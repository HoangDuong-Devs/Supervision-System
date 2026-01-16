# pipeline.py

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..inference.feature_extractor import ReIDFeatureExtractor
from .track_manager import TrackManager
from ..storage.ram_vector_store import RAMVectorStore
from ..storage.metadata_cache import TrackMetadataCache
from configs.autocfg import cfg

logger = logging.getLogger(__name__)


class ReIDPipeline:
    """
    Orchestrates the ReID process per frame.

    - Coordinates FeatureExtractor, MetadataCache, and TrackManager
    - Handles data flow from detection inputs to identity assignment
    """
    def __init__(
        self,
        stream_id: str,
        company_id: str,
        qdrant_face_db=None,
        vector_embedder=None,
    ):
        self.stream_id = stream_id
        self.company_id = company_id
        self.qdrant_face_db = qdrant_face_db
        self.vector_embedder = vector_embedder
        self.logger = logger

        self.reid_tester: Optional[ReIDFeatureExtractor] = None
        self.track_manager: Optional[TrackManager] = None
        self.vector_store: Optional[RAMVectorStore] = None
        self.metadata_cache: Optional[TrackMetadataCache] = None

    def initialize(self) -> bool:
        # 1. RAM vector store first (needed by metadata_cache)
        try:
            self._init_ram_vector_store()
        except Exception as exc:
            self.logger.warning("Failed to initialize RAM Vector Store: %s", exc, exc_info=True)
            self.vector_store = None

        # 2. Metadata cache second (needed by reid_tester for delegation)
        try:
            self._init_metadata_cache()
        except Exception as exc:
            self.logger.error("Failed to initialize Metadata Cache: %s", exc, exc_info=True)
            self.metadata_cache = None

        # 3. ReID tester third (receives metadata_cache reference)
        try:
            self._init_reid_tester()
        except Exception as exc:
            self.logger.error("Failed to initialize ReID feature extractor: %s", exc, exc_info=True)
            self.reid_tester = None

        # 4. Track manager last
        try:
            self._init_track_manager()
        except Exception as exc:
            self.logger.error("Failed to initialize Track Manager: %s", exc, exc_info=True)
            self.track_manager = None

        return (self.reid_tester is not None) or (self.track_manager is not None)

    def _init_reid_tester(self) -> None:
        reid_model_cfg = getattr(cfg.REID, "MODEL", None)
        
        # Fallback to defaults if config missing (should be there in default.py)
        model_name = getattr(reid_model_cfg, "NAME", "osnet_msmt17_engine")
        model_version = getattr(reid_model_cfg, "VERSION", "1")

        self.reid_tester = ReIDFeatureExtractor(
            model_name=model_name,
            model_version=model_version,
            use_triton=True,
            enable_logging=True,
            metadata_cache=self.metadata_cache,
        )

    def _init_ram_vector_store(self) -> None:
        """Initialize RAM-based vector storage."""
        # Read config for capacity and limits
        reid_cfg = getattr(cfg.SUPERVISION_SYSTEM, "REID", None)
        
        capacity = 50000  # 50k vectors default
        snapshot_limit = 20  # Default
        
        if reid_cfg:
            capacity = getattr(reid_cfg, "RAM_CAPACITY", 50000)
            snapshot_limit = getattr(reid_cfg, "SNAPSHOT_HISTORY_LIMIT", 20)
        
        self.vector_store = RAMVectorStore(
            capacity=capacity,
            vector_dim=512,
            snapshot_limit=snapshot_limit,
            company_id=self.company_id,
        )

    def _init_track_manager(self) -> None:
        if not self.vector_store:
            self.track_manager = None
            return

        self.track_manager = TrackManager(
            vector_store=self.vector_store,
            metadata_cache=self.metadata_cache,
        )

        if self.metadata_cache is not None:
            self.metadata_cache.track_manager_ref = self.track_manager

    def _init_metadata_cache(self) -> None:
        self.metadata_cache = TrackMetadataCache(
            vector_store=self.vector_store
        )

    def run_reid(
        self,
        metadata: Dict[str, Any],
        face_person_map: Dict[str, str],
        face_recognition_results: Optional[Dict[str, Any]] = None,
        face_data: Optional[List[Dict]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict]]:
        if not self.reid_tester:
            return {}, {}

        # ---- Build person_list for extractor ----
        person_ids = metadata.get("person", {}).get("ids", [])
        person_bboxes = metadata.get("person", {}).get("bboxes", [])
        person_captures = metadata.get("person", {}).get("captures", [])

        person_list = []
        for i, track_id in enumerate(person_ids):
            person_list.append(
                {
                    "id": track_id,
                    "bbox": person_bboxes[i] if i < len(person_bboxes) else {},
                    "capture": person_captures[i] if i < len(person_captures) else "",
                }
            )

        face_crops_dict: Dict[str, np.ndarray] = {}
        face_rec_map: Dict[str, Dict[str, Any]] = {}

        person_to_face_map: Dict[str, str] = {}
        face_to_person_map: Dict[str, str] = {}

        if face_person_map:
            for k, v in face_person_map.items():
                if isinstance(v, (list, tuple)) and v:
                    face_id = v[0]
                    if face_id is None:
                        continue
                    person_id = str(k)
                    face_to_person_map[str(face_id)] = person_id
                    person_to_face_map[person_id] = str(face_id)
                else:
                    if v is None:
                        continue
                    face_to_person_map[str(k)] = str(v)
                    person_to_face_map[str(v)] = str(k)

        if face_data and face_to_person_map:
            for face_obj in face_data:
                fid = face_obj.get("id")
                if fid is None:
                    continue
                pid = face_to_person_map.get(str(fid))
                if not pid:
                    continue

                face_capture = face_obj.get("capture")
                
                # SHM format only: face_capture is np.ndarray
                if face_capture is not None and isinstance(face_capture, np.ndarray):
                    if face_capture.size > 0:
                        face_crops_dict[pid] = face_capture

                if face_recognition_results:
                    face_key = str(fid)
                    if face_key in face_recognition_results:
                        face_rec_map[pid] = face_recognition_results[face_key]

        for person_obj in person_list:
            pid = str(person_obj["id"])
            if pid in face_rec_map:
                person_obj["face_recognition_result"] = face_rec_map[pid]
            if pid in face_crops_dict:
                person_obj["face_crop"] = face_crops_dict[pid]

        reid_metadata = {
            "stream_id": metadata.get("stream_id"),
            "frame_number": metadata.get("frame_number"),
            "timestamp": metadata.get("timestamp"),
            "person": person_list,
        }

        processed_metadata = self.reid_tester.process_metadata(reid_metadata)
        if processed_metadata is None:
            processed_metadata = reid_metadata

        if self.metadata_cache:
            stream_id = metadata.get("stream_id", self.stream_id)
            frame_number = metadata.get("frame_number", 0)

            reid_features = processed_metadata.get("reid_features", {})
            feature_list = reid_features.get("features", [])
            tracks_extracted = reid_features.get("tracks_extracted", [])
            confidences = reid_features.get("confidences", [])

            feature_map: Dict[str, np.ndarray] = {}
            confidence_map: Dict[str, Optional[float]] = {}
            for i, track_id in enumerate(tracks_extracted):
                if i >= len(feature_list):
                    break
                track_key = str(track_id)
                feature_map[track_key] = np.array(feature_list[i], dtype=np.float32)
                if i < len(confidences):
                    confidence_map[track_key] = confidences[i]

            active_ids_this_frame = set()
            tracks_to_batch_commit = []  # Collect (track, vector) for batch commit

            for person_obj in person_list:
                track_key = str(person_obj["id"])
                track = self.metadata_cache.get_or_create_track(track_key)
                active_ids_this_frame.add(track_key)

                # 1) Reset missing count
                self.metadata_cache.on_track_matched(track)

                # 2) update bbox / confidence
                confidence = person_obj.get("confidence")
                if confidence is None:
                    confidence = confidence_map.get(track_key)

                self.metadata_cache.update_track_metadata(
                    track=track,
                    bbox=person_obj.get("bbox"),
                    confidence=confidence,
                )

                # 3) Kalman update
                self.metadata_cache.update_kalman_state(
                    track=track,
                    bbox=person_obj.get("bbox"),
                    confidence=confidence,
                )

                # 4) Process extracted person vector
                if track_key not in feature_map:
                    continue

                person_vector = feature_map[track_key]

                should_upsert = self.metadata_cache.should_upsert_person_vector(
                    track=track,
                    new_person_vector=person_vector,
                    current_frame=frame_number,
                )

                self.metadata_cache.mark_person_extract(
                    track=track,
                    frame_num=frame_number,
                    vector=person_vector,
                )

                # Collect for batch commit (instead of per-track commit)
                if should_upsert and track.global_id:
                    tracks_to_batch_commit.append((track, person_vector))

            # BATCH COMMIT all collected tracks at once (single RAM upsert call)
            if tracks_to_batch_commit:
                try:
                    committed = self.metadata_cache.batch_commit_person_vectors(
                        tracks_to_commit=tracks_to_batch_commit,
                        stream_id=stream_id,
                        reason="batch_steady_state",
                    )
                    # Update last_upsert_frame for throttling
                    for track, _ in tracks_to_batch_commit:
                        track.last_upsert_frame = frame_number
                except Exception as exc:
                    self.logger.warning(
                        "[ReIDPipeline] Batch commit failed: %s", exc
                    )

            try:
                self.metadata_cache.step_kalman_for_missing(
                    active_track_ids=list(active_ids_this_frame),
                )
            except Exception:
                pass

        metadata["reid_features"] = processed_metadata.get("reid_features", {})
        metadata["reid_person_list"] = processed_metadata.get("person", [])

        return face_crops_dict, face_rec_map

    def assign_global_ids(
        self,
        metadata: Dict[str, Any],
        face_crops_dict: Dict[str, np.ndarray],
        face_rec_map: Dict[str, Dict],
    ) -> None:
        assign_start = time.perf_counter()
        
        if not self.track_manager or not self.reid_tester:
            return

        stream_id = metadata.get("stream_id", self.stream_id)
        frame_number = metadata.get("frame_number", 0)

        person_list = metadata.get("person", [])
        if isinstance(person_list, dict):
            ids = person_list.get("ids", [])
            bboxes = person_list.get("bboxes", [])
            new_list = []
            for i, tid in enumerate(ids):
                new_list.append({"id": tid, "bbox": bboxes[i] if i < len(bboxes) else {}})
            person_list = new_list

        reid_features = metadata.get("reid_features", {})
        feature_list = reid_features.get("features", [])
        tracks_extracted = reid_features.get("tracks_extracted", [])

        feature_map: Dict[str, np.ndarray] = {}
        for i, track_id in enumerate(tracks_extracted):
            if i < len(feature_list):
                feature_map[str(track_id)] = np.array(feature_list[i], dtype=np.float32)

        active_track_ids = [p["id"] for p in person_list]
        person_map = {str(p["id"]): p for p in person_list}

        entries: List[Dict[str, Any]] = []
        for person_obj in person_list:
            track_key = str(person_obj["id"])

            body_vector = feature_map.get(track_key)
            
            # ✅ No feature extracted this frame → preserve existing Global ID
            if body_vector is None:
                track_metadata = self.metadata_cache.get_track(track_key) if self.metadata_cache else None
                if track_metadata and track_metadata.global_id:
                    # Preserve existing assignment
                    person_obj["global_id"] = track_metadata.global_id
                    person_obj["state"] = track_metadata.state or "PENDING"
                    person_obj["assignment_method"] = "preserved"
                    person_obj["assignment_score"] = getattr(track_metadata, "assignment_confidence", None)
                else:
                    # No existing assignment
                    person_obj["global_id"] = None
                    person_obj["state"] = "PENDING"
                    person_obj["assignment_method"] = "no_feature"
                    person_obj["assignment_score"] = None
                person_obj["vote_winner"] = None
                continue

            track_metadata = self.metadata_cache.get_track(track_key) if self.metadata_cache else None

            # Get existing assignment from MetadataCache (single source of truth)
            existing_gid = None
            existing_state = None
            if track_metadata:
                existing_gid = track_metadata.global_id
                existing_state = track_metadata.state

            if existing_gid:
                # Track đã ASSIGNED → giữ nguyên, không gửi vào TrackManager
                person_obj["global_id"] = existing_gid
                person_obj["state"] = existing_state or "ASSIGNED"
                person_obj["assignment_method"] = "retain_assigned"
                person_obj["assignment_score"] = getattr(track_metadata, "assignment_confidence", None)
                person_obj["vote_winner"] = None
                continue

            entries.append(
                {
                    "track_id": track_key,
                    "body_vector": body_vector,
                    "track_metadata": track_metadata,
                    "current_frame_idx": frame_number,
                }
            )

        assignment_results: Dict[str, Dict[str, Any]] = {}
        if entries:
            pending_desc = {"count": len(entries), "tracks": [e["track_id"] for e in entries[:5]]}
            
            try:
                assignment_results = self.track_manager.assign_global_ids_batch(
                    entries,
                    stream_id=stream_id,
                    frame_number=frame_number,
                    active_track_ids=active_track_ids,
                )
            except Exception as exc:
                logger.error(
                    "[ReIDPipeline] Track manager assign_global_ids_batch failed: %s",
                    exc, exc_info=True
                )

        # ✅ Only update tracks that have assignment results (were just processed)
        for track_key, person_obj in person_map.items():
            if track_key in assignment_results:
                # Update with new assignment
                info = assignment_results[track_key]
                person_obj["global_id"] = info.get("global_id")
                person_obj["state"] = info.get("state", "PENDING")
                person_obj["assignment_method"] = info.get("method")
                person_obj["assignment_score"] = info.get("score")
                person_obj["vote_winner"] = info.get("vote_winner")
            # else: Keep existing values (already set in loop above for no_feature tracks)

        reid_person_list = metadata.get("reid_person_list")
        if isinstance(reid_person_list, list):
            for reid_person in reid_person_list:
                key = str(reid_person.get("id"))
                if key in person_map:
                    src = person_map[key]
                    reid_person["global_id"] = src.get("global_id")
                    reid_person["state"] = src.get("state", "PENDING")
                    reid_person["assignment_method"] = src.get("assignment_method")
                    reid_person["assignment_score"] = src.get("assignment_score")
                    reid_person["vote_winner"] = src.get("vote_winner")

        objects = metadata.get("objects", [])
        for obj in objects:
            if obj.get("class") != "person":
                continue
            tid = str(obj.get("id"))
            p = person_map.get(tid)
            if not p:
                continue
            obj["global_id"] = p.get("global_id")
            obj["state"] = p.get("state", "PENDING")
            obj["assignment_method"] = p.get("assignment_method")
            obj["assignment_score"] = p.get("assignment_score")


    def stop(self) -> None:
        if self.reid_tester:
            self.reid_tester.stop()