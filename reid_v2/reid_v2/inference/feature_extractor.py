from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from configs.autocfg import cfg
from ServiceApp.utils import decode_base64_to_image
from ..models.track_history import TrackFeatureHistory

try:
    from modules.triton_.tritonclient_ import TritonInfer
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False
    logging.getLogger(__name__).warning("Triton client not available - ReID will not work")


logger = logging.getLogger("ReIDFeatureExtractor")


class ReIDFeatureExtractor:
    """
    Extracts features from image crops.

    - Uses deep learning model (Triton/Local) to generate vectors
    - Applies vectorized normalization to all outputs
    - Manages per-track feature history to control extraction interval
    """

    def __init__(
        self,
        model_name: str = "osnet_msmt17",
        model_version: str = "1",
        use_triton: bool = True,
        enable_logging: bool = True,
        metadata_cache: Optional[Any] = None,
    ):
        self.logger = logger
        self.logger.setLevel(logging.INFO if enable_logging else logging.WARNING)

        self.reid_model = None
        self.inference_mode = None

        reid_cfg = getattr(cfg.SUPERVISION_SYSTEM, "REID", None)
        
        # Frame-based extraction interval
        self.extraction_interval = getattr(reid_cfg, "EXTRACTION_INTERVAL_FRAMES", 5) if reid_cfg else 5



        self._imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self._imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        if use_triton and TRITON_AVAILABLE:
            self._init_triton_model(model_name=model_name, model_version=model_version)

        if self.reid_model is None:
            self.logger.error("Triton ReID model initialization failed - ReID will not work")

        self.stream_tracks: Dict[str, Dict[str, TrackFeatureHistory]] = defaultdict(dict)
        self.frame_count = 0
        self._metadata_cache = metadata_cache  # For TrackFeatureHistory delegation

    # ------------------------------------------------------------------
    # Triton helpers
    # ------------------------------------------------------------------
    def _get_triton_connection_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        
        reid_cfg = getattr(cfg.SUPERVISION_SYSTEM, "REID", None)
        triton_node = getattr(reid_cfg, "TRITON", None) if reid_cfg else None
        if triton_node is not None:
            kwargs.update({
                "ip": getattr(triton_node, "IP", None),
                "protocol": getattr(triton_node, "PROTOCOL", "grpc"),
                "http_port": getattr(triton_node, "HTTP_CLIENT_PORT", None),
                "grpc_port": getattr(triton_node, "GRPC_CLIENT_PORT", None),
                "http_concurrency": getattr(triton_node, "HTTP_CLIENT_CONCURRENCY", None),
            })

        default_triton = getattr(cfg, "TRITON_SERVER", None)
        if default_triton is not None:
            kwargs.setdefault("ip", getattr(default_triton, "IP", None))
            kwargs.setdefault("protocol", getattr(default_triton, "PROTOCOL", "grpc"))
            kwargs.setdefault("http_port", getattr(default_triton, "HTTP_CLIENT_PORT", None))
            kwargs.setdefault("grpc_port", getattr(default_triton, "GRPC_CLIENT_PORT", None))
            if hasattr(default_triton, "HTTP_CLIENT_CONCURRENCY"):
                kwargs.setdefault("http_concurrency", getattr(default_triton, "HTTP_CLIENT_CONCURRENCY"))

        # env overrides
        if os.getenv("TRITON_REID_IP"):
            kwargs["ip"] = os.getenv("TRITON_REID_IP")
        for env_k, key in [("TRITON_REID_HTTP_PORT", "http_port"), ("TRITON_REID_GRPC_PORT", "grpc_port")]:
            if os.getenv(env_k):
                try:
                    kwargs[key] = int(os.getenv(env_k))
                except ValueError:
                    self.logger.warning("Invalid %s value: %s", env_k, os.getenv(env_k))
        if os.getenv("TRITON_REID_PROTOCOL"):
            kwargs["protocol"] = os.getenv("TRITON_REID_PROTOCOL").lower()

        def _ensure_int(k: str, default: int) -> int:
            v = kwargs.get(k)
            if v is None:
                return default
            if isinstance(v, int):
                return v
            try:
                return int(v)
            except (TypeError, ValueError):
                self.logger.warning("Invalid Triton %s value: %s", k, v)
                return default

        kwargs["http_port"] = _ensure_int("http_port", 8708)
        kwargs["grpc_port"] = _ensure_int("grpc_port", 8709)
        conc = kwargs.get("http_concurrency")
        if conc is not None and not isinstance(conc, int):
            try:
                kwargs["http_concurrency"] = int(conc)
            except Exception:
                self.logger.warning("Invalid Triton HTTP concurrency value: %s", conc)
                kwargs["http_concurrency"] = None

        kwargs.setdefault("ip", "localhost")
        kwargs.setdefault("protocol", "grpc")
        return kwargs

    def _init_triton_model(self, model_name: str, model_version: str) -> None:
        conn = self._get_triton_connection_kwargs()
        try:
            self.reid_model = TritonInfer(
                rec_name=model_name,
                model_version=model_version,
                ip=conn.get("ip"),
                protocol=conn.get("protocol", "grpc"),
                http_port=conn.get("http_port"),
                grpc_port=conn.get("grpc_port"),
                http_concurrency=conn.get("http_concurrency"),
            )
            self.inference_mode = "triton"
        except Exception as exc:
            self.logger.warning("Triton initialization failed: %s", exc, exc_info=True)
            self.reid_model = None
            self.inference_mode = None

    def _run_inference(self, batch: np.ndarray, frame_number: Optional[int] = None, stream_id: Optional[str] = None) -> Optional[np.ndarray]:
        if self.inference_mode != "triton":
            self.logger.warning("ReID inference requested but no backend initialised")
            return None
        try:
            ctx = {"frame_number": frame_number, "stream_id": stream_id}
            try:
                raw = self.reid_model.forward(batch, context=ctx)
            except TypeError:
                raw = self.reid_model.forward(batch)
        except Exception as exc:
            self.logger.error("Triton inference error: %s", exc, exc_info=True)
            return None

        if raw is None:
            self.logger.warning("Triton returned empty result for ReID batch size=%s", len(batch))
            return None

        arr = np.array(raw)
        if arr.size == 0:
            return None
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        elif arr.ndim > 2:
            last_dim = arr.shape[-1]
            arr = arr.reshape(-1, last_dim)
        return arr.astype(np.float32, copy=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        loop_t0 = time.perf_counter()
        try:
            stream_id = metadata.get("stream_id", "unknown")
            frame_number = metadata.get("frame_number", 0)
            self.frame_count += 1

            person_list = metadata.get("person", [])
            if not person_list:
                metadata["reid_features"] = {"by_track_id": {}, "features": [], "tracks_extracted": []}
                self.logger.warning("[%s] No person data, reid_features empty", stream_id)
                return metadata

            tracks = self.stream_tracks.setdefault(stream_id, {})

            # Frame-aligned extraction cycle
            is_extraction_frame = (frame_number % self.extraction_interval == 0)
            
            indices_to_extract: List[int] = []
            for i, person_obj in enumerate(person_list):
                track_id = person_obj["id"]
                hist = tracks.setdefault(track_id, TrackFeatureHistory(
                    track_id=track_id, 
                    stream_id=stream_id,
                    metadata_cache=self._metadata_cache,
                ))
                hist.register_frame(frame_number)
                
                if is_extraction_frame:
                    hist.last_extract_frame = frame_number
                    hist.extract_count += 1
                    indices_to_extract.append(i)
            

            decode_elapsed = 0.0
            infer_elapsed = 0.0

            features_out: List[List[float]] = []
            tracks_out: List[str] = []
            confidences_out: List[Optional[float]] = []
            features_out_dict: Dict[str, List[float]] = {}

            if indices_to_extract:
                t0 = time.perf_counter()
                selected_caps = [person_list[i]["capture"] for i in indices_to_extract]
                crops = self._decode_crops(selected_caps)
                decode_elapsed = time.perf_counter() - t0

                if crops:
                    t1 = time.perf_counter()
                    body_feats = self._extract_reid_features_batch(crops, frame_number=frame_number, stream_id=stream_id)
                    infer_elapsed = time.perf_counter() - t1

                    for rel_idx, person_idx in enumerate(indices_to_extract):
                        if rel_idx >= len(body_feats):
                            self.logger.warning("[%s] Missing body feature for rel_idx %s/%s", stream_id, rel_idx, len(body_feats))
                            continue
                        person_obj = person_list[person_idx]
                        track_id = person_obj["id"]
                        body_vec = body_feats[rel_idx]
                        hist = tracks[track_id]
                        hist.add_observation(feature=body_vec, frame_number=frame_number)
                        track_key = str(track_id)
                        vec_list = body_vec.tolist()
                        features_out.append(vec_list)
                        features_out_dict[track_key] = vec_list
                        tracks_out.append(track_key)
                        confidences_out.append(person_obj.get("confidence"))

            metadata["reid_features"] = {
                "by_track_id": features_out_dict,
                "features": features_out,
                "tracks_extracted": tracks_out,
                "confidences": confidences_out,
            }

            if not features_out and indices_to_extract:
                self.logger.warning(
                    "[%s] Expected %d ReID features but none produced (frame=%s)",
                    stream_id,
                    len(indices_to_extract),
                    frame_number,
                )
        except Exception:
            pass

        return metadata

    # Crop decode + model preprocessing
    # ------------------------------------------------------------------
    def _decode_crops(self, captures: List[Any]) -> List[Optional[np.ndarray]]:
        """
        Decode captures to numpy arrays.
        Supports:
        - np.ndarray: direct from SHM (zero-copy)
        - str: base64 encoded image
        - None/empty: skipped
        """
        crops: List[Optional[np.ndarray]] = []
        empty = 0
        for cap in captures:
            # SHM: Handle np.ndarray directly
            if isinstance(cap, np.ndarray):
                if cap.size > 0:
                    crops.append(cap)
                else:
                    crops.append(None)
                    empty += 1
                continue
            
            # Base64 string logic
            if not cap:
                crops.append(None)
                empty += 1
                continue
            
            if isinstance(cap, str):
                try:
                    img = decode_base64_to_image(cap)
                except Exception:
                    img = None
                if img is None:
                    empty += 1
                crops.append(img)
            else:
                # Unknown type
                crops.append(None)
                empty += 1
                
        return crops

    def _extract_reid_features_batch(self, crops: List[Optional[np.ndarray]], frame_number: Optional[int] = None, stream_id: Optional[str] = None) -> List[np.ndarray]:
        preprocess_start = time.time()
        preprocessed = []
        valid_idx = []

        for idx, crop in enumerate(crops):
            if crop is None:
                continue
            resized = cv2.resize(crop, (128, 256))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            norm_img = rgb.astype(np.float32) / 255.0
            norm_img = (norm_img - self._imagenet_mean) / self._imagenet_std
            chw = np.transpose(norm_img, (2, 0, 1))
            preprocessed.append(chw.astype(np.float32, copy=False))
            valid_idx.append(idx)

        if not preprocessed:
            return []

        batch = np.stack(preprocessed, axis=0).astype(np.float32, copy=False)
        preprocess_elapsed = (time.time() - preprocess_start) * 1000

        triton_start = time.time()
        raw_feats = self._run_inference(batch, frame_number=frame_number, stream_id=stream_id)
        triton_elapsed = (time.time() - triton_start) * 1000
        
        if raw_feats is None:
            self.logger.warning("Failed to obtain ReID features for batch (size=%s)", len(preprocessed))
            feat_dim = 512
            return [np.zeros(feat_dim, dtype=np.float32) for _ in crops]

        feat_dim = raw_feats.shape[-1]
        
        # Vectorized L2 normalize (faster than loop)
        raw_feats = raw_feats.astype(np.float32)
        norms = np.linalg.norm(raw_feats, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.maximum(norms, 1e-8)
        normed_feats = raw_feats / norms

        if len(normed_feats) != len(valid_idx):
            self.logger.warning("ReID feature count mismatch (features=%s valid_idx=%s)", len(normed_feats), len(valid_idx))

        out: List[np.ndarray] = []
        zeros_template = np.zeros(feat_dim, dtype=np.float32)
        feat_map = {vi: normed_feats[i] for i, vi in enumerate(valid_idx) if i < len(normed_feats)}
        for idx in range(len(crops)):
            out.append(feat_map.get(idx, zeros_template.copy()))
        return out

    def stop(self) -> None:
        pass