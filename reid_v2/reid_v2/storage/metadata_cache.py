# storage/metadata_cache.py

"""
RAM metadata cache for track management.

- Centralizes state management for all active tracks
- Handles EMA vector updates and renormalization
- coordinates concurrent access with thread locks
"""

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from ..inference.kalman_xywh import KalmanFilterXYWH
except Exception as exc:
    KalmanFilterXYWH = None  # type: ignore
    logging.getLogger(__name__).warning("Kalman filter unavailable: %s", exc)

from ..models.track_metadata import TrackMetadata
from ..utils.geometry import normalize_to_ltwh
from ..utils.vector_utils import l2_normalize, cosine_similarity_normalized
from configs.autocfg import cfg

logger = logging.getLogger(__name__)


class TrackMetadataCache:
    """RAM cache for active tracks.

    - Thread-safe creation/access
    - Kalman bbox prediction (optional)
    - EMA for person vectors
    - Policies to decide when to upsert vectors
    """

    PERSON_SIMILARITY_MIN_CHANGE = 0.11
    PERSON_SIMILARITY_MAX_CHANGE = 0.3
    MAX_UPSERT_OVERLAP_RATIO = 0.5

    FORCE_SNAPSHOT_EVERY_N = 120
    EMA_ALPHA = 0.15
    MAX_HISTORY_VECTORS = 10

    def __init__(self, vector_store: Optional[Any] = None, track_manager_ref: Optional[Any] = None):
        self.local_tracks: Dict[str, TrackMetadata] = {}
        self.vector_store = vector_store
        self.kalman_filter = KalmanFilterXYWH() if KalmanFilterXYWH else None
        self._track_locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()
        self.track_manager_ref = track_manager_ref
        self._lock_cleanup_counter = 0

    # basic accessors

    def _get_track_lock(self, track_id: str) -> threading.Lock:
        with self._global_lock:
            if track_id not in self._track_locks:
                self._track_locks[track_id] = threading.Lock()
            return self._track_locks[track_id]

    def get_or_create_track(self, local_id: Any) -> TrackMetadata:
        key = str(local_id)
        track = self.local_tracks.get(key)
        if track is not None:
            return track

        created = False
        total_active = 0
        with self._global_lock:
            track = self.local_tracks.get(key)
            if track is None:
                track = TrackMetadata(local_id=key)
                self.local_tracks[key] = track
                created = True
                total_active = len(self.local_tracks)


        return track

    def get_track(self, local_id: Any) -> Optional[TrackMetadata]:
        return self.local_tracks.get(str(local_id))

    def remove_track(self, local_id: Any, stream_id: Optional[str] = None) -> None:
        key = str(local_id)
        if key in self.local_tracks:
            track = self.local_tracks[key]
            track_state = getattr(track, "state", "UNKNOWN")
            has_gid = bool(getattr(track, "global_id", None))
            history_len = len(getattr(track, "person_vector_history", []))
            

            if hasattr(track, 'person_vector_history'):
                track.person_vector_history.clear()
            if hasattr(track, 'candidate_cache'):
                track.candidate_cache.clear()
                track.candidate_cache_token = None
            if hasattr(track, 'bboxes'):
                track.bboxes.clear()

            track.kalman_mean = None
            track.kalman_covariance = None
            track.kalman_predicted_mean = None
            track.kalman_predicted_covariance = None
            track.kalman_predicted_bbox = None
            track.last_person_vector = None
            track.person_vector_ema = None


            with self._global_lock:
                if key in self._track_locks:
                    del self._track_locks[key]

            if self.track_manager_ref and stream_id:
                try:
                    if hasattr(self.track_manager_ref, '_clear_pending'):
                        self.track_manager_ref._clear_pending(stream_id, key)
                        
                except Exception:
                    pass

            del self.local_tracks[key]

            self._lock_cleanup_counter += 1
            if self._lock_cleanup_counter >= 100:
                self._cleanup_stale_locks()
                self._lock_cleanup_counter = 0

    def get_all_tracks(self) -> Dict[str, TrackMetadata]:
        return self.local_tracks

    def _cleanup_stale_locks(self) -> None:
        with self._global_lock:
            active_keys = set(self.local_tracks.keys())
            stale_keys = set(self._track_locks.keys()) - active_keys
            for key in stale_keys:
                del self._track_locks[key]

    # bbox helpers

    @staticmethod
    def _bbox_to_xywh(bbox: Optional[Any]) -> Optional[np.ndarray]:
        norm = normalize_to_ltwh(bbox)
        if not norm:
            return None
        w = norm["width"]
        h = norm["height"]
        if w <= 1e-4 or h <= 1e-4:
            return None
        cx = norm["left"] + w / 2.0
        cy = norm["top"] + h / 2.0
        return np.array([cx, cy, w, h], dtype=np.float32)

    @staticmethod
    def _xywh_to_bbox(xywh: np.ndarray) -> Dict[str, float]:
        cx, cy, w, h = xywh.astype(float).tolist()
        w = max(w, 1.0)
        h = max(h, 1.0)
        left = cx - w / 2.0
        top = cy - h / 2.0
        left = max(-100.0, min(left, 10000.0))
        top = max(-100.0, min(top, 10000.0))
        return {"left": float(left), "top": float(top), "width": float(w), "height": float(h)}

    # NOTE: should_extract_person_vector() removed - logic now in feature_extractor.py (global frame cycle)

    def _intersection_over_own_area(self, own_bbox: Optional[Dict], other_bbox: Optional[Dict]) -> float:
        from ..utils.geometry import compute_intersection_ratio_safe
        return compute_intersection_ratio_safe(own_bbox, other_bbox)

    def _are_bboxes_close(self, bbox1: Optional[Dict], bbox2: Optional[Dict], distance_ratio_threshold: float = 0.75) -> bool:
        from ..utils.geometry import are_bboxes_close
        return are_bboxes_close(bbox1, bbox2, threshold=distance_ratio_threshold)

    def _is_heavily_overlapped(self, track: TrackMetadata) -> bool:
        own_bbox = track.latest_bbox
        if own_bbox is None:
            return False

        tracks_snapshot = list(self.local_tracks.values())
        for other in tracks_snapshot:
            if other.local_id == track.local_id:
                continue
            if getattr(other, "missing_count", 0) > 0:
                continue
            other_bbox = other.latest_bbox
            if other_bbox is None:
                continue
            if not self._are_bboxes_close(own_bbox, other_bbox):
                continue
            ratio = self._intersection_over_own_area(own_bbox, other_bbox)
            if ratio > float(self.MAX_UPSERT_OVERLAP_RATIO):
                
                return True
        
        return False

    def _ema_update(self, track: TrackMetadata, new_vec: np.ndarray, alpha: Optional[float] = None) -> None:
        if alpha is None:
            alpha = self.EMA_ALPHA
        if new_vec is None:
            return
        if track.person_vector_ema is None:
            track.person_vector_ema = l2_normalize(new_vec.copy())
            track.person_vector_ema_count = 1
            
        else:
            beta = float(alpha)
            ema = (1.0 - beta) * track.person_vector_ema + beta * new_vec
            track.person_vector_ema = l2_normalize(ema)
            track.person_vector_ema_count += 1
            

    def get_matching_anchor(self, track: TrackMetadata) -> Optional[np.ndarray]:
        if track.person_vector_ema is not None:
            return track.person_vector_ema
        elif track.last_person_vector is not None:
            return track.last_person_vector
        return None

    def should_upsert_person_vector(
        self,
        track: TrackMetadata,
        new_person_vector: np.ndarray,
        current_frame: Optional[int] = None,
    ) -> bool:
        """
        Decide whether to accept/upsert a new person vector.
        
        For ASSIGNED tracks, throttles updates to once every N frames (configurable).
        For PENDING tracks, accepts all vectors for voting diversity.
        """
        lock = self._get_track_lock(track.local_id)
        with lock:
            if track.last_person_vector is not None:
                new_hash = hash(new_person_vector.tobytes())
                last_hash = hash(track.last_person_vector.tobytes())
                if new_hash == last_hash:
                    return False

            try:
                if self._is_heavily_overlapped(track):
                    self._ema_update(track, new_person_vector, alpha=self.EMA_ALPHA / 2.0)
                    return False
            except Exception:
                pass

            if track.last_person_vector is None:
                self._ema_update(track, new_person_vector, alpha=self.EMA_ALPHA)
                return True

            if (track.person_extraction_count > 0 and track.person_extraction_count % self.FORCE_SNAPSHOT_EVERY_N == 0):
                self._ema_update(track, new_person_vector, alpha=self.EMA_ALPHA)
                return True

            # ================================================================
            # PENDING tracks: Always accept new vectors for voting diversity
            # ASSIGNED tracks: Check frame interval + similarity
            # ================================================================
            if not track.global_id or track.state != "ASSIGNED":
                # PENDING: accept all vectors (needed for voting)
                self._ema_update(track, new_person_vector, alpha=self.EMA_ALPHA)
                return True

            # ================================================================
            # ASSIGNED: Frame-based throttling (new feature)
            # Only accept vector every N frames to reduce redundant upserts
            # ================================================================
            reid_cfg = getattr(cfg.SUPERVISION_SYSTEM, "REID", None)
            upsert_interval = getattr(reid_cfg, "UPSERT_INTERVAL_ASSIGNED_FRAMES", 60) if reid_cfg else 60
            
            if current_frame is not None and track.last_upsert_frame is not None:
                frames_since_upsert = current_frame - track.last_upsert_frame
                if frames_since_upsert < upsert_interval:
                    # Skip upsert, but still update EMA with reduced weight
                    self._ema_update(track, new_person_vector, alpha=self.EMA_ALPHA / 4.0)
                    
                    return False

            # ASSIGNED: Check similarity to avoid duplicate upserts
            # Compare with last extracted vector (will be upserted)
            # NOT with EMA because EMA drifts continuously → Qdrant vectors can still be identical
            base = track.last_person_vector
            sim = cosine_similarity_normalized(base, new_person_vector)
            change = 1.0 - sim

            if change < self.PERSON_SIMILARITY_MIN_CHANGE or change > self.PERSON_SIMILARITY_MAX_CHANGE:
                self._ema_update(track, new_person_vector, alpha=self.EMA_ALPHA / 2.0)
                return False

            self._ema_update(track, new_person_vector, alpha=self.EMA_ALPHA)
            return True

    def mark_person_extract(self, track: TrackMetadata, frame_num: int, vector: np.ndarray) -> None:
        lock = self._get_track_lock(track.local_id)
        with lock:
            if track.last_person_vector_frame is not None:
                if frame_num < track.last_person_vector_frame:
                    return

            track.last_person_vector = vector.copy()
            track.last_person_vector_frame = frame_num
            track.person_extraction_count += 1

            if not track.global_id:
                track.person_vector_history.append((
                    vector.copy(),
                    frame_num,
                ))
                
                trimmed = 0
                dropped_info: Optional[int] = None
                while len(track.person_vector_history) > self.MAX_HISTORY_VECTORS:
                    dropped_vector = track.person_vector_history.pop(0)
                    trimmed += 1
                    if dropped_vector and len(dropped_vector) >= 2:
                        dropped_info = dropped_vector[1]  # frame_num
                

    # Kalman & bbox updates

    def update_track_metadata(self, track: TrackMetadata, bbox: Optional[Dict] = None, confidence: Optional[float] = None) -> None:
        if bbox:
            norm = normalize_to_ltwh(bbox)
            if norm is not None:
                track.latest_bbox = norm
                if len(track.bboxes) >= 10:
                    track.bboxes.pop(0)
                track.bboxes.append(norm)
            else:
                track.latest_bbox = bbox
                if len(track.bboxes) >= 10:
                    track.bboxes.pop(0)
                track.bboxes.append(bbox)

        if confidence is not None:
            track.person_confidence = float(confidence)

        track.last_update_time = time.time()

    def update_kalman_state(self, track: TrackMetadata, bbox: Optional[Any], confidence: Optional[float] = None) -> None:
        if self.kalman_filter is None:
            return

        measurement = self._bbox_to_xywh(bbox)
        if measurement is None:
            return

        conf_val = float(confidence) if confidence is not None else 0.0

        try:
            if track.kalman_mean is None or track.kalman_covariance is None:
                mean, cov = self.kalman_filter.initiate(measurement)
            else:
                mean_pred, cov_pred = self.kalman_filter.predict(track.kalman_mean, track.kalman_covariance)
                mean, cov = self.kalman_filter.update(mean_pred, cov_pred, measurement, confidence=conf_val)

            track.kalman_mean = mean
            track.kalman_covariance = cov
            track.kalman_last_observation_time = time.time()

            pred_mean, pred_cov = self.kalman_filter.predict(mean.copy(), cov.copy())
            track.kalman_predicted_mean = pred_mean
            track.kalman_predicted_covariance = pred_cov
            track.kalman_predicted_bbox = self._xywh_to_bbox(pred_mean[:4])

        except Exception as exc:
            logger.warning("[Cache] Kalman update failed for track %s: %s", track.local_id, exc)

    def advance_kalman_state(self, track: TrackMetadata) -> None:
        if self.kalman_filter is None:
            return
        if track.kalman_mean is None or track.kalman_covariance is None:
            return

        # Try new config structure first, fallback to legacy
        stage1_cfg = getattr(cfg.SUPERVISION_SYSTEM, "STAGE1", None)
        if stage1_cfg:
            MAX_MISSING_FOR_KALMAN = getattr(stage1_cfg, "KALMAN_MAX_MISSING_FRAMES", 120)
        else:
            MAX_MISSING_FOR_KALMAN = getattr(cfg.SUPERVISION_SYSTEM.TRACK_MANAGER, "recent_lost_frames", 120)
        if track.missing_count > MAX_MISSING_FOR_KALMAN:
            return

        try:
            mean, cov = self.kalman_filter.predict(track.kalman_mean, track.kalman_covariance)
            track.kalman_mean = mean
            track.kalman_covariance = cov

            track.kalman_predicted_mean = mean
            track.kalman_predicted_covariance = cov
            track.kalman_predicted_bbox = self._xywh_to_bbox(mean[:4])

        except Exception as exc:
            logger.warning("[Cache] Kalman predict failed for track %s: %s", track.local_id, exc)

    def on_track_matched(self, track: TrackMetadata) -> None:
        track.missing_count = 0

    def on_track_missing(self, track: TrackMetadata) -> None:
        track.missing_count += 1
        self.advance_kalman_state(track)

    def step_kalman_for_missing(self, active_track_ids: List[str]) -> None:
        active = set(str(t) for t in (active_track_ids or []))
        for tid, track in self.local_tracks.items():
            if tid in active:
                continue
            self.on_track_missing(track)

    # Vector commit helpers

    def commit_person_vectors(
        self,
        track: TrackMetadata,
        stream_id: str,
        snapshot_suffix: Optional[str] = None,
        reason: str = "manual",
    ) -> bool:
        if self.vector_store is None or not track.global_id or track.last_person_vector is None:
            return False

        ok_snap = self.vector_store.upsert_snapshot(
            track_id=track.local_id,
            stream_id=stream_id,
            body_vector=track.last_person_vector,
            metadata={
                "global_id": track.global_id,
                "state": track.state,
            },
            snapshot_suffix=snapshot_suffix,
        )

        return bool(ok_snap)

    def batch_commit_person_vectors(
        self,
        tracks_to_commit: List[Tuple["TrackMetadata", np.ndarray]],
        stream_id: str,
        reason: str = "batch_steady_state",
    ) -> int:
        """
        Batch commit vectors from multiple assigned tracks in one Qdrant API call.
        
        Args:
            tracks_to_commit: List of (track, vector) tuples
            stream_id: Stream ID
            reason: Commit reason for logging
            
        Returns:
            Number of successfully committed vectors
        """
        if not self.vector_store or not tracks_to_commit:
            return 0
        
        entries = []
        for track, vector in tracks_to_commit:
            if not track.global_id or vector is None:
                continue
            
            entries.append({
                "track_id": track.local_id,
                "vector": vector,
                "global_id": track.global_id,
                "metadata": {
                    "state": track.state,
                },
            })
        
        if not entries:
            return 0
        
        # Single batch upsert call
        committed = self.vector_store.batch_upsert_multi_tracks(
            entries=entries,
            stream_id=stream_id,
        )
        
        
        
        return committed

    def commit_person_vector_history(
        self,
        track: TrackMetadata,
        stream_id: str,
        reason: str = "manual",
    ) -> int:
        if self.vector_store is None:
            return 0

        if not track.global_id:
            logger.warning("[Cache] Track %s: no Global ID, skip history commit (reason=%s)", track.local_id, reason)
            return 0

        if not track.person_vector_history:
            return 0

        # Check if last_person_vector is the same as the last vector in history
        # If so, remove it from history to avoid duplicate upsert
        last_vector_is_in_history = False
        if track.last_person_vector is not None and len(track.person_vector_history) > 0:
            last_hist_vector, last_hist_frame = track.person_vector_history[-1]
            # Compare: same frame number means likely same vector
            if track.last_person_vector_frame == last_hist_frame:
                # Double-check with vector similarity (exact match or very close)
                sim = np.dot(track.last_person_vector, last_hist_vector) / (
                    np.linalg.norm(track.last_person_vector) * np.linalg.norm(last_hist_vector) + 1e-9
                )
                if sim > 0.9999:  # Practically identical
                    last_vector_is_in_history = True
                    

        # ================================================================
        # TRIM HISTORY if _vectors_to_keep is set (from vector transfer logic)
        # This ensures we keep only the most recent N vectors from new track
        # ================================================================
        history_to_commit = list(track.person_vector_history)
        vectors_to_keep = getattr(track, '_vectors_to_keep', None)
        if vectors_to_keep is not None and vectors_to_keep < len(history_to_commit):
            # Keep the LAST N vectors (most recent)
            trimmed_count = len(history_to_commit) - vectors_to_keep
            history_to_commit = history_to_commit[-vectors_to_keep:]
            
            # Clear the flag
            track._vectors_to_keep = None

        # Prepare batch: collect (vector, suffix) tuples
        vectors_with_suffixes = [
            (vector, f"hist_{idx}_{frame_num}")
            for idx, (vector, frame_num) in enumerate(history_to_commit)
        ]

        # Batch upsert all history vectors WITH global_id
        committed = self.vector_store.batch_upsert_snapshots(
            track_id=track.local_id,
            stream_id=stream_id,
            vectors_with_suffixes=vectors_with_suffixes,
            metadata={
                "global_id": track.global_id,
                "state": track.state,
            },
        )

        total_before = len(track.person_vector_history)
        track.person_vector_history.clear()
        
        # Mark that last_person_vector was already committed via history
        if last_vector_is_in_history:
            track._last_vector_committed_via_history = True

        return committed