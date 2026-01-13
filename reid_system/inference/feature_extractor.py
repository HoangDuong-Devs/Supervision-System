# reid_system/inference/feature_extractor.py

"""
ReID Feature Extractor using ONNX Runtime.

Replaces Triton inference with ONNX Runtime for local inference.
Maintains the EXACT same interface as reid_system.inference.feature_extractor.py
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from ..models.track_metadata import TrackMetadata
from ..utils.vector_utils import l2_normalize

logger = logging.getLogger("ReIDFeatureExtractor")


class ReIDFeatureExtractor:
    """
    Extracts features from image crops using ONNX Runtime.

    Interface matches reid_v2.inference.feature_extractor.ReIDFeatureExtractor exactly.
    Only difference: uses ONNX instead of Triton inference.
    """

    def __init__(
        self,
        model_name: str = "osnet_msmt17",
        model_version: str = "1", 
        use_triton: bool = False,  # Ignored - always use ONNX
        enable_logging: bool = True,
        metadata_cache: Optional[Any] = None,
        onnx_model_path: str = "osnet_x1_0_msmt17.onnx",
    ):
        self.logger = logger
        self.logger.setLevel(logging.INFO if enable_logging else logging.WARNING)

        self.onnx_model_path = onnx_model_path
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: str = ""
        self.output_name: str = ""

        # Frame-based extraction interval
        self.extraction_interval = 5

        # ImageNet normalization constants
        self._imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self._imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        # Initialize ONNX model
        self._init_onnx_model()

        if self.session is None:
            self.logger.error("ONNX ReID model initialization failed - ReID will not work")

        self.frame_count = 0
        self._metadata_cache = metadata_cache

    def _init_onnx_model(self) -> None:
        """Initialize ONNX Runtime session."""
        if not ONNX_AVAILABLE:
            self.logger.error("ONNX Runtime not installed. Run: pip install onnxruntime-gpu")
            return

        try:
            # Try CUDA first, fallback to CPU
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            available_providers = ort.get_available_providers()
            
            providers = [p for p in providers if p in available_providers]
            if not providers:
                providers = ['CPUExecutionProvider']

            self.session = ort.InferenceSession(self.onnx_model_path, providers=providers)
            
            # Get input/output names
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            used_provider = self.session.get_providers()[0] if self.session.get_providers() else "Unknown"
            self.logger.info(
                "[ReIDFeatureExtractor] ONNX model loaded: %s (provider: %s)",
                self.onnx_model_path, used_provider
            )
            
        except Exception as e:
            self.logger.error("[ReIDFeatureExtractor] Failed to load ONNX model: %s", e)
            self.session = None

    def supports_batch_inference(self) -> bool:
        """Check if batch inference is supported."""
        return self.session is not None

    def _preprocess_batch(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Preprocess batch of crops for ONNX inference.
        
        Args:
            crops: List of BGR image crops
            
        Returns:
            Preprocessed tensor (batch_size, 3, H, W)
        """
        if not crops:
            return np.array([])

        batch_size = len(crops)
        batch_data = np.zeros((batch_size, 3, 256, 128), dtype=np.float32)  # OSNet input size
        
        for i, crop in enumerate(crops):
            if crop is None or crop.size == 0:
                continue
                
            # Resize to model input size (128x256)
            resized = cv2.resize(crop, (128, 256))
            
            # Convert BGR to RGB
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Convert to float32 and normalize to [0, 1]
            normalized = rgb.astype(np.float32) / 255.0
            
            # Apply ImageNet normalization
            normalized = (normalized - self._imagenet_mean) / self._imagenet_std
            
            # Transpose to (C, H, W)
            transposed = normalized.transpose(2, 0, 1)
            batch_data[i] = transposed
        
        return batch_data

    def should_extract_features_for_track(
        self,
        track_id: str,
        stream_id: str,
        frame_idx: int,
    ) -> bool:
        """
        Check if features should be extracted for track (interval-based).
        
        Maintains EXACT interface from reid_v2.
        """
        if self._metadata_cache is None:
            # Fallback to simple interval check
            return frame_idx % self.extraction_interval == 0

        # Get track metadata from cache
        cache_key = str(track_id)  # Convert to string to match cache format
        metadata = self._metadata_cache.get_or_create_track(cache_key)
        
        if metadata is None:
            return True  # Extract for new tracks

        # Check last extraction frame
        last_extract_frame = metadata.last_person_vector_frame
        if last_extract_frame is None:
            return True
            
        return (frame_idx - last_extract_frame) >= self.extraction_interval

    def extract_features_batch(
        self,
        crops: List[np.ndarray],
        track_ids: List[str],
        stream_id: str,
        frame_idx: int,
    ) -> List[Optional[np.ndarray]]:
        """
        Extract ReID features from batch of crops.
        
        Maintains EXACT interface from reid_v2.
        
        Args:
            crops: List of BGR image crops
            track_ids: Corresponding track IDs
            stream_id: Stream identifier
            frame_idx: Current frame index
            
        Returns:
            List of feature vectors (or None for skipped tracks)
        """
        if not self.session:
            return [None] * len(crops)

        if len(crops) != len(track_ids):
            self.logger.warning("Crops and track_ids length mismatch")
            return [None] * len(crops)

        # Initialize results for all tracks
        results = [None] * len(crops)
        extract_indices = []
        extract_crops = []

        # Filter tracks that need feature extraction
        for i, (crop, track_id) in enumerate(zip(crops, track_ids)):
            if self.should_extract_features_for_track(track_id, stream_id, frame_idx):
                extract_indices.append(i)
                extract_crops.append(crop)
            else:
                # Use cached features if available
                if self._metadata_cache:
                    cache_key = str(track_id)  # Match cache format
                    metadata = self._metadata_cache.get_or_create_track(track_id)
                    if metadata and metadata.last_person_vector is not None:
                        results[i] = metadata.last_person_vector.copy()

        if not extract_crops:
            return results

        # Process batch
        try:
            t0 = time.perf_counter()
            batch_data = self._preprocess_batch(extract_crops)
            
            if batch_data.size == 0:
                return results
            
            # Run inference
            outputs = self.session.run([self.output_name], {self.input_name: batch_data})
            features_batch = outputs[0]
            
            if features_batch is None:
                self.logger.error("[ReIDFeatureExtractor] ONNX inference returned None")
                return results
                
            elapsed = (time.perf_counter() - t0) * 1000
            self.logger.debug(
                "[ReIDFeatureExtractor] Batch inference: %d crops in %.2fms",
                len(extract_crops), elapsed
            )

            # Insert extracted features at correct positions
            for feat_idx, orig_idx in enumerate(extract_indices):
                if feat_idx < len(features_batch):
                    feature = features_batch[feat_idx]
                    if feature is not None:
                        feature_vector = l2_normalize(feature.flatten())
                        results[orig_idx] = feature_vector
                        
                        # Update metadata cache
                        if self._metadata_cache:
                            track_id = track_ids[orig_idx]
                            metadata = self._metadata_cache.get_or_create_track(track_id)
                            metadata.last_person_vector = feature_vector.copy()
                            metadata.last_person_vector_frame = frame_idx
                            metadata.person_extraction_count += 1
            
            return results

        except Exception as e:
            self.logger.error("[ReIDFeatureExtractor] Batch inference error: %s", e)
            self.logger.error("[ReIDFeatureExtractor] Error details - crops count: %d, track_ids: %s", 
                            len(crops), track_ids[:5] if track_ids else "None")
            import traceback
            self.logger.error("[ReIDFeatureExtractor] Traceback: %s", traceback.format_exc())
            return [None] * len(crops)

    def extract_features_from_crops(
        self,
        crops: List[np.ndarray], 
        track_ids: List[str],
        stream_id: str,
        frame_idx: int,
    ) -> List[Optional[np.ndarray]]:
        """
        Legacy method name - delegates to extract_features_batch.
        
        Maintains backward compatibility with reid_v2.
        """
        return self.extract_features_batch(crops, track_ids, stream_id, frame_idx)

    def process_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process metadata and extract ReID features.
        
        This method maintains the exact interface as reid_v2 ReIDFeatureExtractor.process_metadata()
        but uses ONNX inference instead of Triton.
        """
        loop_t0 = time.perf_counter()
        try:
            stream_id = metadata.get("stream_id", "unknown")
            frame_number = metadata.get("frame_number", 0)
            
            person_list = metadata.get("person", [])
            if not person_list:
                metadata["reid_features"] = {"by_track_id": {}, "features": [], "tracks_extracted": []}
                self.logger.debug("[%s] No person data, reid_features empty", stream_id)
                return metadata

            # For demo: extract features from every frame (simplified)
            indices_to_extract: List[int] = list(range(len(person_list)))
            
            features_out: List[List[float]] = []
            tracks_out: List[str] = []
            confidences_out: List[Optional[float]] = []
            features_out_dict: Dict[str, List[float]] = {}

            if indices_to_extract:
                # Get crops from person data
                crops = []
                for i in indices_to_extract:
                    person_obj = person_list[i]
                    capture = person_obj.get("capture")
                    if isinstance(capture, np.ndarray):
                        crops.append(capture)
                    else:
                        # Skip if no valid crop
                        crops.append(None)

                # Extract features using ONNX
                if crops and any(crop is not None for crop in crops):
                    valid_crops = [crop for crop in crops if crop is not None]
                    valid_track_ids = []
                    
                    # Extract track IDs for valid crops
                    crop_idx = 0
                    for rel_idx, person_idx in enumerate(indices_to_extract):
                        if crops[rel_idx] is not None:
                            person_obj = person_list[person_idx]
                            track_id = person_obj.get("id", f"track_{person_idx}")
                            valid_track_ids.append(track_id)
                            
                    if valid_crops and valid_track_ids:
                        # Get stream_id and frame_idx from metadata
                        stream_id = metadata.get("stream_id", "demo")
                        frame_idx = metadata.get("frame_number", 0)
                        
                        # Call extract_features_batch with all required params
                        body_feats = self.extract_features_batch(
                            valid_crops, 
                            valid_track_ids,
                            stream_id,
                            frame_idx
                        )
                        
                        feat_idx = 0
                        for rel_idx, person_idx in enumerate(indices_to_extract):
                            if crops[rel_idx] is None:
                                continue
                                
                            if feat_idx >= len(body_feats):
                                break
                                
                            person_obj = person_list[person_idx]
                            track_id = person_obj.get("id", f"track_{person_idx}")
                            body_vec = body_feats[feat_idx]
                            
                            # Check if ONNX returned valid feature vector
                            if body_vec is None:
                                self.logger.warning("ONNX returned None for track %s", track_id)
                                feat_idx += 1
                                continue
                            
                            track_key = str(track_id)
                            vec_list = body_vec.tolist()
                            features_out.append(vec_list)
                            features_out_dict[track_key] = vec_list
                            tracks_out.append(track_key)
                            confidences_out.append(person_obj.get("confidence"))
                            feat_idx += 1

            metadata["reid_features"] = {
                "by_track_id": features_out_dict,
                "features": features_out,
                "tracks_extracted": tracks_out,
                "confidences": confidences_out,
            }
            
        except Exception as e:
            self.logger.error("Error in process_metadata: %s", e, exc_info=True)
            metadata["reid_features"] = {"by_track_id": {}, "features": [], "tracks_extracted": []}

        return metadata

    def cleanup_stream_tracks(self, stream_id: str, active_track_ids: List[str]) -> None:
        """
        Clean up tracking state for inactive tracks.
        
        Maintains interface from reid_v2 (may not be needed for ONNX).
        """
        # Not needed for ONNX implementation
        pass

    def stop(self) -> None:
        """Release resources."""
        self.session = None