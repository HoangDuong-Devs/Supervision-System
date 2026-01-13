"""
Track state layers for the supervision system.

- PendingLayer: Short-lived tracks, voting window, vector collection.
- AssignedLayer: Long-lived tracks, global ID, EMA vectors, payload cache.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass
class PendingTrackState:
    """
    In-RAM state for tracks awaiting Global ID assignment.
    
    Manages:
    - Voting window (deque)
    - Vector accumulation (for Stage 1/2)
    - Short-term candidate pool
    """

    stream_id: str
    track_id: str
    # Frame-based tracking
    created_frame: int = 0
    last_update_frame: int = 0
    votes: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=10))
    seq: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)
    last_seen_ts: float = field(default_factory=time.time)
    missing_count: int = 0
    is_active: bool = False
    vector_counter: int = 0
    last_vector_frame: Optional[int] = None
    latest_vector: Optional[np.ndarray] = None
    short_pool: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Cache vectors đã kéo về cho Stage 1 (Motion-based voting)
    pool_vectors_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Frame-based voting timeline
    first_vote_frame: Optional[int] = None

    def snapshot_votes(self) -> List[Dict[str, Any]]:
        with self.lock:
            return [dict(v) for v in self.votes]

    def record_vector(
        self,
        vector: Optional[np.ndarray],
        frame_idx: Optional[int],
    ) -> bool:
        if vector is None:
            return False
        with self.lock:
            # Frame-based dedup: avoid recording for same frame
            if frame_idx is not None and frame_idx == self.last_vector_frame:
                return False
            self.vector_counter += 1
            self.last_vector_frame = frame_idx
            self.latest_vector = vector.copy()
        return True

    def reset_vectors(self) -> None:
        with self.lock:
            self.vector_counter = 0
            self.last_vector_frame = None
            self.latest_vector = None


class PendingTrackStore:
    """
    Thread-safe container for PendingTrackState objects.
    
    - Handles ensure/get/drop/cleanup operations.
    - Manages vector snapshots and short-lost candidate pools.
    """

    def __init__(self, vote_window_size: int, vector_snapshot_limit: int = 1):
        self.vote_window_size = max(1, vote_window_size)
        self.vector_snapshot_limit = max(1, vector_snapshot_limit)
        self._tracks: Dict[Tuple[str, str], PendingTrackState] = {}
        self._lock = threading.Lock()

    # -- lifecycle -----------------------------------------------------

    def ensure(
        self,
        stream_id: str,
        track_id: str,
        frame_idx: Optional[int] = None,
        missing_count: Optional[int] = None,
        is_active: Optional[bool] = None,
    ) -> Tuple[PendingTrackState, bool]:
        key = (stream_id, track_id)
        created = False
        with self._lock:
            state = self._tracks.get(key)
            if state is None:
                votes = deque(maxlen=self.vote_window_size)
                state = PendingTrackState(
                    stream_id=stream_id,
                    track_id=track_id,
                    created_frame=int(frame_idx or 0),
                    last_update_frame=int(frame_idx or 0),
                    votes=votes,
                )
                self._tracks[key] = state
                created = True
            else:
                if frame_idx is not None:
                    state.last_update_frame = int(frame_idx)
            state.last_seen_ts = time.time()
            if missing_count is not None:
                state.missing_count = int(missing_count)
            if is_active is not None:
                state.is_active = bool(is_active)
        return state, created

    def get(self, stream_id: str, track_id: str) -> Optional[PendingTrackState]:
        with self._lock:
            return self._tracks.get((stream_id, track_id))

    def drop(self, stream_id: str, track_id: str) -> Optional[PendingTrackState]:
        with self._lock:
            return self._tracks.pop((stream_id, track_id), None)

    def cleanup(self, predicate) -> List[PendingTrackState]:
        removed: List[PendingTrackState] = []
        with self._lock:
            for key, state in list(self._tracks.items()):
                if predicate(state):
                    removed.append(self._tracks.pop(key))
        return removed

    def values(self) -> Iterable[PendingTrackState]:
        with self._lock:
            return list(self._tracks.values())

    # -- data helpers --------------------------------------------------

    def record_vector(
        self,
        stream_id: str,
        track_id: str,
        vector: Optional[np.ndarray],
        frame_idx: Optional[int],
    ) -> bool:
        state, _ = self.ensure(stream_id, track_id, frame_idx)
        recorded = state.record_vector(vector, frame_idx)
        return recorded

    def vector_snapshot(self, stream_id: str, track_id: str) -> Optional[Dict[str, Any]]:
        state = self.get(stream_id, track_id)
        if state is None:
            return None
        with state.lock:
            if state.vector_counter <= 0:
                return None
            vector = state.latest_vector.copy() if state.latest_vector is not None else None
            return {
                "track_id": track_id,
                "stream_id": stream_id,
                "vector_count": state.vector_counter,
                "last_vector": vector,
                "last_frame": state.last_vector_frame,
            }

    def reset_vector_stats(self, stream_id: str, track_id: str) -> None:
        state = self.get(stream_id, track_id)
        if state:
            state.reset_vectors()

    # -- short-lost candidate pool -----------------------------------

    def short_pool_snapshot(self, stream_id: str, track_id: str) -> List[Dict[str, Any]]:
        state = self.get(stream_id, track_id)
        if state is None:
            return []
        with state.lock:
            snapshot: List[Dict[str, Any]] = []
            for candidate in state.short_pool.values():
                entry = dict(candidate)
                payload = entry.get("payload")
                if isinstance(payload, dict):
                    entry["payload"] = dict(payload)
                snapshot.append(entry)
            return snapshot

    def extend_short_pool(
        self,
        stream_id: str,
        track_id: str,
        candidates: List[Dict[str, Any]],
        max_size: int,
    ) -> int:
        state = self.get(stream_id, track_id)
        if state is None or max_size <= 0:
            return 0
        if not candidates:
            return 0

        added = 0
        with state.lock:
            for candidate in candidates:
                if len(state.short_pool) >= max_size:
                    break
                track_key = str(candidate.get("track_id") or "")
                if not track_key or track_key in state.short_pool:
                    continue

                entry = dict(candidate)
                payload = entry.get("payload")
                if isinstance(payload, dict):
                    entry["payload"] = dict(payload)
                state.short_pool[track_key] = entry
                added += 1
        return added

    def remove_from_short_pool(
        self,
        stream_id: str,
        track_id: str,
        remove_ids: Iterable[str],
    ) -> int:
        state = self.get(stream_id, track_id)
        if state is None:
            return 0
        removed = 0
        with state.lock:
            for rid in remove_ids or []:
                if rid is None:
                    continue
                rid_str = str(rid)
                if rid_str in state.short_pool:
                    state.short_pool.pop(rid_str, None)
                    removed += 1
        return removed

    # -- pool vectors cache (Stage 1) --------------------------------

    def set_pool_vectors_cache(
        self,
        stream_id: str,
        track_id: str,
        vectors_data: Dict[str, Dict[str, Any]],
    ) -> None:
        """
        Cache vectors đã kéo về từ RAM cho Stage 1.
        Chỉ gọi 1 lần khi lock pool.
        
        Args:
            vectors_data: {candidate_track_id: {"vectors": [...], "payload": {...}, "global_id": "..."}}
        """
        state = self.get(stream_id, track_id)
        if state is None:
            return
        with state.lock:
            state.pool_vectors_cache = vectors_data

    def get_pool_vectors_cache(
        self,
        stream_id: str,
        track_id: str,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Lấy vectors cache cho Stage 1.
        Returns empty dict nếu chưa có cache.
        """
        state = self.get(stream_id, track_id)
        if state is None:
            return {}
        with state.lock:
            return dict(state.pool_vectors_cache)

    def remove_from_pool_vectors_cache(
        self,
        stream_id: str,
        track_id: str,
        remove_ids: Iterable[str],
    ) -> int:
        """
        Loại bỏ track_ids khỏi vectors cache (khi track đã active/retired).
        """
        state = self.get(stream_id, track_id)
        if state is None:
            return 0
        removed = 0
        with state.lock:
            for rid in remove_ids or []:
                if rid is None:
                    continue
                rid_str = str(rid)
                if rid_str in state.pool_vectors_cache:
                    state.pool_vectors_cache.pop(rid_str, None)
                    removed += 1
        return removed


@dataclass
class AssignedTrackState:
    """
    In-RAM state for tracks with assigned Global IDs.
    
    Manages:
    - EMA vector state
    - Payload cache (bbox, etc.)
    - Activity status (missing count)
    """

    stream_id: str
    track_id: str
    global_id: str
    payload: Dict[str, Any] = field(default_factory=dict)
    last_seen_ts: float = field(default_factory=time.time)
    is_active: bool = False
    last_vector: Optional[np.ndarray] = None
    ema_vector: Optional[np.ndarray] = None
    ema_count: int = 0
    missing_count: int = 0  # Frame count since last seen

    def update_activity(self, is_active: bool) -> None:
        self.last_seen_ts = time.time()
        # Update missing_count: reset when active, increment when inactive
        if is_active:
            self.missing_count = 0
        else:
            self.missing_count += 1
        self.is_active = bool(is_active)

    def update_vector(self, vector: Optional[np.ndarray], alpha: float) -> None:
        if vector is None:
            return
        vector_copy = vector.copy()
        self.last_vector = vector_copy
        if self.ema_vector is None:
            self.ema_vector = vector_copy
            self.ema_count = 1
            return
        self.ema_vector = alpha * vector_copy + (1.0 - alpha) * self.ema_vector
        self.ema_count += 1


class AssignedTrackStore:
    """
    Thread-safe container for AssignedTrackState objects.
    
    - Indexing by (stream_id, track_id) AND global_id.
    - Handles assignment registration and vector EMA updates.
    """

    def __init__(self, ema_alpha: float = 0.15):
        self.ema_alpha = max(1e-4, float(ema_alpha))
        self._tracks: Dict[Tuple[str, str], AssignedTrackState] = {}
        self._by_global: Dict[str, AssignedTrackState] = {}
        self._lock = threading.Lock()

    def register_assignment(
        self,
        stream_id: str,
        track_id: str,
        global_id: str,
        payload: Optional[Dict[str, Any]] = None,
        pending_snapshot: Optional[Dict[str, Any]] = None,
    ) -> AssignedTrackState:
        with self._lock:
            state = AssignedTrackState(
                stream_id=stream_id,
                track_id=track_id,
                global_id=global_id,
                payload=dict(payload or {}),
            )
            self._tracks[(stream_id, track_id)] = state
            self._by_global[global_id] = state

        if pending_snapshot and pending_snapshot.get("last_vector") is not None:
            state.update_vector(pending_snapshot.get("last_vector"), self.ema_alpha)
        return state

    def observe(
        self,
        stream_id: str,
        track_id: str,
        global_id: Optional[str],
        *,
        is_active: bool,
        body_vector: Optional[np.ndarray] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Optional[AssignedTrackState]:
        if not global_id:
            return None
        key = (stream_id, track_id)
        with self._lock:
            state = self._tracks.get(key)
            if state is None:
                state = self._by_global.get(global_id)
                if state is None:
                    state = AssignedTrackState(
                        stream_id=stream_id,
                        track_id=track_id,
                        global_id=global_id,
                    )
                self._tracks[key] = state
                self._by_global[global_id] = state
            if payload:
                state.payload.update(payload)
        state.update_activity(is_active)
        state.update_vector(body_vector, self.ema_alpha)
        return state

    def get(self, stream_id: str, track_id: str) -> Optional[AssignedTrackState]:
        with self._lock:
            return self._tracks.get((stream_id, track_id))

    def get_by_global_id(self, global_id: str) -> Optional[AssignedTrackState]:
        with self._lock:
            return self._by_global.get(global_id)

    def retire(self, stream_id: str, track_id: str) -> Optional[AssignedTrackState]:
        key = (stream_id, track_id)
        with self._lock:
            state = self._tracks.pop(key, None)
            if state and self._by_global.get(state.global_id) is state:
                self._by_global.pop(state.global_id, None)
        return state

    def get_inactive_assigned_by_stream(
        self,
        stream_id: str,
        exclude_track_ids: Optional[set] = None,
        limit: int = 50,
        max_missing_frames: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get inactive assigned tracks as dict format for Stage 1 filtering.
        
        Returns dicts compatible with existing IoU filter logic:
        {
            "track_id": str,
            "global_id": str,
            "payload": {...},  # Contains bbox_kalman, bbox_current
        }
        
        Sorted by last_seen_ts descending (most recently seen first).
        
        Args:
            max_missing_frames: If set, only return tracks missing < N frames
        """
        exclude_set = exclude_track_ids or set()
        candidates: List[Tuple[float, Dict[str, Any]]] = []
        
        with self._lock:
            for (s_id, t_id), state in self._tracks.items():
                if s_id != stream_id:
                    continue
                if t_id in exclude_set:
                    continue
                if state.is_active:
                    continue
                if not state.global_id:
                    continue
                # Filter by missing frame count
                if max_missing_frames is not None and state.missing_count >= max_missing_frames:
                    continue
                # Build dict compatible with _filter_by_iou format
                candidates.append((state.last_seen_ts, {
                    "track_id": t_id,
                    "global_id": state.global_id,
                    "payload": state.payload,
                    "score": 1.0,  # Dummy score for sorting, will be recalculated
                    "missing_count": state.missing_count,
                }))
        
        # Sort by last_seen_ts descending
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        return [c[1] for c in candidates[:limit]]