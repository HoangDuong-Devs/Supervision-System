import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple
import threading

import numpy as np
# RAM backend

from ..utils.geometry import normalize_to_ltwh, are_bboxes_close, compute_iou
from ..utils.vector_utils import cosine_similarity_normalized
from ..demo_config import cfg
from .track_layers import AssignedTrackStore, PendingTrackStore

logger = logging.getLogger(__name__)


class TrackManager:
    """
    Manages identity assignment logic.
    
    - Implements 3-stage voting process for Global ID assignment
    - Coordinates Candidate Search (Stage 2) and Motion Matching (Stage 1)
    - Manages lifecycle of Pending and Assigned tracks
    """

    def __init__(self, vector_store, metadata_cache=None):
        self.vector_store = vector_store
        self.metadata_cache = metadata_cache
        
        self.global_id_counter = 0
        self._gid_lock = threading.Lock()
        
        # Voting constants
        self.VOTE_WINDOW_SIZE = cfg.SUPERVISION_SYSTEM.VOTING.WINDOW_SIZE
        self.VOTE_RATIO_THRESHOLD = 0.30
        
        # Top-K voting config
        self.VOTE_TOPK = cfg.SUPERVISION_SYSTEM.VOTING.TOPK_PER_VOTE
        self.VOTE_SCORE_FILTER = cfg.SUPERVISION_SYSTEM.VOTING.SCORE_FILTER
        
        # Early exit config
        self.EARLY_EXIT_ENABLED = cfg.SUPERVISION_SYSTEM.VOTING.EARLY_EXIT_ENABLED
        self.EARLY_EXIT_MIN_VOTES = cfg.SUPERVISION_SYSTEM.VOTING.EARLY_EXIT_MIN_VOTES
        self.EARLY_EXIT_RATIO = cfg.SUPERVISION_SYSTEM.VOTING.EARLY_EXIT_RATIO
        self.EARLY_EXIT_MARGIN = cfg.SUPERVISION_SYSTEM.VOTING.EARLY_EXIT_MARGIN
        
        # Final assignment threshold
        self.MIN_VOTE_RATIO_TO_ASSIGN = cfg.SUPERVISION_SYSTEM.VOTING.MIN_RATIO_TO_ASSIGN
        
        self.pending_store = PendingTrackStore(
            vote_window_size=self.VOTE_WINDOW_SIZE,
            vector_snapshot_limit=1
        )
        self.assigned_store = AssignedTrackStore(
            ema_alpha=0.15
        )
        # Reuse history
        self.track_gid_history: Dict[Tuple[str, str], str] = {}
        
        self.retired_tracks: Dict[Tuple[str, str], Tuple[float, Optional[str]]] = {}
        self._retired_lock = threading.Lock()
        
        # Retired track limit
        self._retired_track_limit = cfg.SUPERVISION_SYSTEM.TRACK_MANAGER.RETIRED_TRACK_LIMIT
        
        # Voting timeout
        self.VOTING_TIMEOUT_ENABLED = cfg.SUPERVISION_SYSTEM.VOTING.TIMEOUT_ENABLED
        self.VOTING_MIN_VOTES_TO_COMPLETE = cfg.SUPERVISION_SYSTEM.VOTING.MIN_VOTES_TO_COMPLETE
        
        # Dynamic timeout calculation
        # Formula: Window Size * Interval * Multiplier
        # Example: 10 votes * 5 frames/vote * 2x multiplier = 100 frames timeout
        _extraction_interval = cfg.SUPERVISION_SYSTEM.REID.EXTRACTION_INTERVAL_FRAMES
        _multiplier = cfg.SUPERVISION_SYSTEM.VOTING.TIMEOUT_MULTIPLIER
        self.VOTING_TIMEOUT_FRAMES = self.VOTE_WINDOW_SIZE * _extraction_interval * _multiplier

    # ------------------------------------------------------------------ #
    # Core API
    # ------------------------------------------------------------------ #

    def generate_new_global_id(self) -> str:
        with self._gid_lock:
            self.global_id_counter += 1
            new_gid = f"G_{self.global_id_counter}"

            return new_gid

    def _normalize_global_id(self, global_id: Optional[str]) -> Optional[str]:
        if not global_id or (isinstance(global_id, str) and global_id.startswith("PENDING_")):
            return None
        return global_id

    # ------------------------------------------------------------------ #
    # Pending Vote Management
    # ------------------------------------------------------------------ #
    
    def _ensure_pending(
        self,
        stream_id: str,
        track_id: str,
        current_frame: Optional[int] = None,
        *,
        missing_count: Optional[int] = None,
        is_active: Optional[bool] = None,
    ) -> None:
        state, created = self.pending_store.ensure(
            stream_id=stream_id,
            track_id=track_id,
            frame_idx=current_frame,
            missing_count=missing_count,
            is_active=is_active,
        )

    def _clear_pending(self, stream_id: str, track_id: str) -> None:
        removed = self.pending_store.drop(stream_id, track_id)
        if removed is not None:
            vote_len = len(removed.votes)

    def _cleanup_expired_pending(self) -> int:
        """Remove pending entries for tracks missing 30+ consecutive frames.
        
        Note: Protected tracks with >= MIN_VOTES_TO_COMPLETE will NOT be expired.
        """
        cleanup_threshold = int(cfg.SUPERVISION_SYSTEM.STAGE1.KALMAN_MAX_MISSING_FRAMES) / 2 # Approx logic
        # Fallback if config is too large or different unit, just use 30 as reasonable default for pending cleanup
        cleanup_threshold = 30
        min_votes_protected = self.VOTING_MIN_VOTES_TO_COMPLETE

        def _should_expire(state) -> bool:
            # Protection: Don't expire tracks that have enough votes to complete
            vote_count = len(state.votes) if hasattr(state, 'votes') else 0
            if vote_count >= min_votes_protected:
                return False  # Protected from expiration
            
            missing_count = state.missing_count
            if self.metadata_cache:
                track = self.metadata_cache.local_tracks.get(state.track_id)
                if track is not None:
                    missing_count = getattr(track, "missing_count", missing_count)
            return missing_count >= cleanup_threshold

        expired_states = self.pending_store.cleanup(_should_expire)
        if expired_states:
            ids = [f"{st.stream_id}:{st.track_id}" for st in expired_states]
            
        return len(expired_states)

    def _add_pending_votes(self, stream_id: str, track_id: str, 
                        candidates: List[Tuple[str, float]], 
                        frame_idx: Optional[int] = None,
                        gid_to_track: Optional[Dict[str, str]] = None):
        """Add votes for Top-K candidates (not just winner) to improve robustness."""
        pending_state = self.pending_store.get(stream_id, track_id)
        if pending_state is None:
            return
        
        # Ensure lock exists - create one if missing to prevent race condition
        entry_lock = pending_state.lock
        if entry_lock is None:
            entry_lock = threading.Lock()
            pending_state.lock = entry_lock

        # Build Top-K candidates list (thay vì chỉ winner)
        topk_candidates: List[Dict[str, Any]] = []
        if candidates:
            # Sort by score descending, take top-K
            sorted_cands = sorted(candidates, key=lambda x: x[1], reverse=True)
            for gid, score in sorted_cands[:self.VOTE_TOPK]:
                if gid and score > 0:
                    topk_candidates.append({"gid": gid, "score": float(score)})

        # Always use lock for thread safety
        with entry_lock:
            pending_state.seq += 1
            seq = pending_state.seq
            per_obs = {
                "seq": seq,
                "candidates": topk_candidates,  # Top-K thay vì chỉ winner
                "frame_idx": frame_idx,         # Frame-based primary
                # Backward compatibility: giữ winner/score cho logging
                "winner": topk_candidates[0]["gid"] if topk_candidates else None,
                "score": topk_candidates[0]["score"] if topk_candidates else 0.0,
            }
            pending_state.votes.append(per_obs)
            
            # Track first vote frame for timeout calculation
            if pending_state.first_vote_frame is None and frame_idx is not None:
                pending_state.first_vote_frame = frame_idx

        # Logging (outside locks to keep locks short)
        winner_gid = topk_candidates[0]["gid"] if topk_candidates else None
        winner_score = topk_candidates[0]["score"] if topk_candidates else 0.0
        winner_track = gid_to_track.get(winner_gid, "unknown") if gid_to_track and winner_gid else None
        winner_display = f"{winner_gid}(track={winner_track})" if winner_track else winner_gid
        
        


    def _compute_vote_aggregates(self, stream_id: str, track_id: str, gid_to_track: Optional[Dict[str, str]] = None):
        """Aggregate votes from Top-K candidates with filtered mean and early exit detection."""
        pending_state = self.pending_store.get(stream_id, track_id)
        if not pending_state:
            return {
                "agg": {},
                "total": 0,
                "winner": None,
                "winner_weight": 0,
                "ratio": 0.0,
                "window_fill": 0.0,
                "vote_count": 0,
                "score_sums": {},
                "score_avgs": {},
                "filtered_avgs": {},
                "early_exit": False,
            }

        votes_snapshot = pending_state.snapshot_votes()
        total_votes = len(votes_snapshot)
        
        # Aggregate từ Top-K candidates (không chỉ winner)
        agg: Dict[str, int] = {}  # GID -> vote count (số lần xuất hiện trong top-K)
        all_scores: Dict[str, List[float]] = {}  # GID -> all scores (for filtered mean)
        last_scores: Dict[str, float] = {}
        
        for per_obs in votes_snapshot:
            # Hỗ trợ cả format mới (candidates) và cũ (winner)
            candidates = per_obs.get("candidates", [])
            if candidates:
                # Format mới: Top-K candidates
                for cand in candidates:
                    gid = cand.get("gid")
                    score = float(cand.get("score", 0.0))
                    if gid:
                        agg[gid] = agg.get(gid, 0) + 1
                        all_scores.setdefault(gid, []).append(score)
                        last_scores[gid] = score
            else:
                # Backward compatibility: format cũ (chỉ winner)
                gid = per_obs.get("winner")
                score = float(per_obs.get("score", 0.0))
                if gid:
                    agg[gid] = agg.get(gid, 0) + 1
                    all_scores.setdefault(gid, []).append(score)
                    last_scores[gid] = score

        # If no gid votes at all
        if not agg:
            window_fill = total_votes / (self.VOTE_WINDOW_SIZE or 1)
            return {
                "agg": agg,
                "total": total_votes,
                "winner": None,
                "winner_weight": 0,
                "ratio": 0.0,
                "window_fill": window_fill,
                "vote_count": total_votes,
                "score_sums": {},
                "score_avgs": {},
                "filtered_avgs": {},
                "early_exit": False,
            }

        # Tính filtered mean (loại outliers < VOTE_SCORE_FILTER)
        def _filtered_avg(gid: str) -> float:
            scores = all_scores.get(gid, [])
            if not scores:
                return 0.0
            # Lọc scores vượt ngưỡng (loại frames bị occlusion/blur)
            valid_scores = [s for s in scores if s >= self.VOTE_SCORE_FILTER]
            if not valid_scores:
                # Fallback: dùng all scores nếu không có valid
                valid_scores = scores
            return sum(valid_scores) / len(valid_scores)
        
        def _raw_avg(gid: str) -> float:
            scores = all_scores.get(gid, [])
            if not scores:
                return 0.0
            return sum(scores) / len(scores)

        # Tính score_sums cho backward compatibility
        score_sums: Dict[str, float] = {gid: sum(scores) for gid, scores in all_scores.items()}
        filtered_avgs: Dict[str, float] = {gid: _filtered_avg(gid) for gid in agg}
        raw_avgs: Dict[str, float] = {gid: _raw_avg(gid) for gid in agg}

        # Use fixed denominator -> VOTE_WINDOW_SIZE
        denom = self.VOTE_WINDOW_SIZE if self.VOTE_WINDOW_SIZE > 0 else total_votes or 1
        ratio_map: Dict[str, float] = {gid: weight / denom for gid, weight in agg.items()}
        consensus_gids = [gid for gid, ratio_val in ratio_map.items() if ratio_val >= self.VOTE_RATIO_THRESHOLD]

        winner_gid: Optional[str] = None
        winner_weight = 0
        if consensus_gids:
            # Chọn winner: ưu tiên filtered_avg > ratio > vote count
            winner_gid = max(
                consensus_gids,
                key=lambda gid: (filtered_avgs.get(gid, 0.0), ratio_map.get(gid, 0.0), agg.get(gid, 0)),
            )
            winner_weight = agg.get(winner_gid, 0)
        else:
            def _fallback_key(item: Tuple[str, int]) -> Tuple[int, float]:
                gid, weight = item
                return (weight, filtered_avgs.get(gid, 0.0))
            winner_gid, winner_weight = max(agg.items(), key=_fallback_key)

        ratio = ratio_map.get(winner_gid, 0.0) if winner_gid else 0.0
        window_fill = total_votes / (self.VOTE_WINDOW_SIZE or 1)
        winner_filtered_avg = filtered_avgs.get(winner_gid, 0.0) if winner_gid else 0.0

        # ================================================================
        # EARLY EXIT DETECTION
        # ================================================================
        early_exit = False
        if self.EARLY_EXIT_ENABLED and winner_gid:
            # Điều kiện early exit:
            # 1. Đủ số votes tối thiểu
            # 2. Ratio >= EARLY_EXIT_RATIO
            # 3. Margin với top-2 đủ lớn (ko cần check min_avg vì đã filter ở Stage 1/2)
            if (total_votes >= self.EARLY_EXIT_MIN_VOTES and
                ratio >= self.EARLY_EXIT_RATIO):
                
                # Tìm top-2 để kiểm tra margin
                sorted_by_avg = sorted(
                    filtered_avgs.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                second_avg = 0.0
                if len(sorted_by_avg) >= 2:
                    # Lấy candidate thứ 2 (khác winner)
                    for gid, avg in sorted_by_avg:
                        if gid != winner_gid:
                            second_avg = avg
                            break
                
                margin = winner_filtered_avg - second_avg
                if margin >= self.EARLY_EXIT_MARGIN:
                    early_exit = True
                    



        return {
            "agg": agg,
            "total": total_votes,
            "winner": winner_gid,
            "winner_weight": winner_weight,
            "ratio": ratio,
            "window_fill": window_fill,
            "vote_count": total_votes,
            "score_sums": score_sums,
            "score_avgs": raw_avgs,
            "filtered_avgs": filtered_avgs,
            "ratio_map": ratio_map,
            "last_scores": last_scores,
            "early_exit": early_exit,
            "winner_filtered_avg": winner_filtered_avg,
            "first_vote_frame": pending_state.first_vote_frame,
        }

    def _update_track_layers(
        self,
        *,
        stream_id: str,
        track_id: str,
        global_id: Optional[str],
        is_visible: bool,
        lost_frames: int,
        frame_idx: Optional[int],
        body_vector: Optional[np.ndarray],
        track_metadata: Optional[Any],
    ) -> None:
        if global_id:
            payload = self._build_assigned_payload(track_metadata, global_id)
            self.assigned_store.observe(
                stream_id=stream_id,
                track_id=track_id,
                global_id=global_id,
                is_active=is_visible,
                body_vector=body_vector,
                payload=payload,
            )
            return

        self._ensure_pending(
            stream_id,
            track_id,
            current_frame=frame_idx,
            missing_count=lost_frames,
            is_active=is_visible,
        )
        self.pending_store.record_vector(
            stream_id=stream_id,
            track_id=track_id,
            vector=body_vector,
            frame_idx=frame_idx,
        )

    def _build_assigned_payload(self, track_metadata: Optional[Any], global_id: Optional[str]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "global_id": global_id,
            "state": "ASSIGNED" if global_id else "PENDING",
            "last_seen_ts": time.time(),
        }
        if track_metadata is None:
            return payload

        payload.update(
            {
                "track_id": getattr(track_metadata, "local_id", None),
                "bbox_current": getattr(track_metadata, "latest_bbox", None),
                "bbox_kalman": getattr(track_metadata, "kalman_predicted_bbox", None),
            }
        )
        return payload

    def _register_assignment_layers(
        self,
        *,
        stream_id: str,
        track_id: str,
        global_id: Optional[str],
        track_metadata: Optional[Any],
    ) -> None:
        if not global_id:
            self.pending_store.reset_vector_stats(stream_id, track_id)
            return

        payload = self._build_assigned_payload(track_metadata, global_id)
        pending_snapshot = self.pending_store.vector_snapshot(stream_id, track_id)
        self.assigned_store.register_assignment(
            stream_id=stream_id,
            track_id=track_id,
            global_id=global_id,
            payload=payload,
            pending_snapshot=pending_snapshot,
        )
        self.pending_store.reset_vector_stats(stream_id, track_id)

    def _filter_inactive_assigned(
        self,
        matches: List[Dict],
        stream_id: str,
        active_track_ids: Optional[List[str]],
    ) -> List[Dict]:
        """Post-search safety filter to catch tracks that became active AFTER search.
        
        This handles race conditions where a track's state changed between:
        - When the vector store indexed the must_not filter (stale snapshot)
        - When we process the results (current state)
        
        NOTE: This is NOT redundant - it's protection against timing issues.
        """
        if not matches:
            return []

        active_set = {str(tid) for tid in (active_track_ids or [])}
        filtered: List[Dict] = []
        dropped = 0

        for match in matches:
            payload = (match.get("payload") or {}) if isinstance(match, dict) else {}
            candidate_track = payload.get("track_id")
            candidate_stream = payload.get("stream_id") or stream_id

            if candidate_track is None:
                filtered.append(match)
                continue

            candidate_track_str = str(candidate_track)
            
            # Race condition check: Did this track become active after search?
            if candidate_track_str in active_set:
                dropped += 1
                continue

            # Double-check assigned_store for most current is_active status
            state = self.assigned_store.get(str(candidate_stream), candidate_track_str)
            if state and state.is_active:
                dropped += 1
                continue

            filtered.append(match)

        return filtered

    def _set_assign_winner_context(
        self,
        context: Dict[str, Any],
        winner_gid: str,
        gid_to_meta: Dict[str, Dict[str, Any]],
        matches: List[Dict[str, Any]],
        assign_reason: str,
    ) -> Dict[str, Any]:
        """
        Set context fields for assigning winner global_id.
        
        Common pattern extracted from: early_exit, consensus, window_full_winner, timeout_complete.
        """
        winner_meta = gid_to_meta.get(winner_gid) if winner_gid else None
        winner_payload = (winner_meta or {}).get("payload") or {}
        winner_payload_score = float((winner_meta or {}).get("score", 0.0))
        
        # Fallback to first match if no winner_payload
        if not winner_payload and matches:
            winner_payload = matches[0].get("payload", {}) or {}
            winner_payload_score = float(matches[0].get("score", 0.0))
        
        context["status"] = "assign_global"
        context["candidate_global_id"] = winner_gid
        context["candidate_track_id"] = (
            winner_meta.get("track_id") if winner_meta else winner_payload.get("track_id")
        )
        context["candidate_stream_id"] = (
            winner_meta.get("stream_id") if winner_meta else winner_payload.get("stream_id")
        )
        context["clear_pending"] = True
        context["best_score"] = winner_payload_score
        context["search_method"] = f"voting_{assign_reason}"
        context["assign_reason"] = assign_reason
        return context

    # ------------------------------------------------------------------ #
    # Public Core API
    # ------------------------------------------------------------------ #
    
    def assign_global_id(self, track_id: str, stream_id: str, body_vector: np.ndarray,
                        track_metadata: Optional[Any] = None, force_recompute: bool = False,
                        current_frame_idx: Optional[int] = None,
                        active_track_ids: Optional[List[str]] = None) -> Tuple[str, Dict]:
        
        if body_vector is None:
            return None, {"method": "no_feature", "state": "PENDING"}

        entry = {
            "track_id": str(track_id),
            "body_vector": body_vector,
            "track_metadata": track_metadata,
            "current_frame_idx": current_frame_idx,
            "force_recompute": force_recompute,
        }

        batch_results = self.assign_global_ids_batch([entry], stream_id, 
                                                     current_frame_idx or 0, active_track_ids)
        info = batch_results.get(str(track_id), 
                                {"global_id": None, "method": "no_feature", "state": "PENDING"})
        return info.get("global_id"), info

    def assign_global_ids_batch(self, entries: List[Dict[str, Any]], stream_id: str,
                                frame_number: int, active_track_ids: Optional[List[str]] = None
                                ) -> Dict[str, Dict[str, Any]]:
        """
        Batch assign global IDs with optimized network I/O.
        
        3-Phase Flow:
        1. COLLECT: Identify pool track_ids and search queries for all pending tracks
        2. BATCH EXECUTE: Single batch_retrieve + batch_search_flat calls
        3. DISTRIBUTE: Pass results to per-track processing
        """
        self._cleanup_expired_pending()
        
        # Periodic stale track cleanup (every ~1000 calls)
        if not hasattr(self, "_cleanup_counter"):
            self._cleanup_counter = 0
        self._cleanup_counter += 1
        if self._cleanup_counter >= 1000:
            self._cleanup_counter = 0
            # Periodically cleanup stale tracks from RAM (if supported)
            if hasattr(self.vector_store, "cleanup_stale_tracks"):
                cleanup_age = getattr(cfg.SUPERVISION_SYSTEM.TRACK_MANAGER, "CLEANUP_AGE_SECONDS", 3600)
                removed_keys = self.vector_store.cleanup_stale_tracks(max_age_seconds=cleanup_age)
                # Sync cleanup with Metadata Cache
                if removed_keys and self.metadata_cache:
                    for key in removed_keys:
                        # key format "stream_id:track_id"
                        parts = key.split(":", 1)
                        if len(parts) == 2:
                            self.metadata_cache.remove_track(parts[1], stream_id=parts[0])
                    # Simple log to confirm cleanup sync (since we don't have full logs)
                    # logger.info("[TrackManager] Synced cleanup for %d tracks", len(removed_keys))
        contexts: List[Dict[str, Any]] = []
        batch_start = time.perf_counter()
        
        # ================================================================
        # PHASE 1: COLLECT - Identify what data each track needs
        # ================================================================
        
        # Stage 1: Collect ALL pool track_ids from ALL pending tracks
        all_pool_track_ids: set = set()
        track_to_pool: Dict[str, List[str]] = {}  # pending_track_id -> [pool_track_ids]
        
        # Stage 2: Collect search queries
        stage2_queries: Dict[str, Dict[str, Any]] = {}  # pending_track_id -> {"vector": ..., "filter": ...}
        
        # Build common filter components
        active_set = set(active_track_ids) if active_track_ids else set()
        retired_set = set()
        with self._retired_lock:
            for (ret_stream_id, ret_track_id), _ in self.retired_tracks.items():
                if ret_stream_id == str(stream_id):
                    retired_set.add(ret_track_id)
        
        # Config thresholds
        RECENT_LOST_FRAMES = cfg.SUPERVISION_SYSTEM.STAGE1.KALMAN_MAX_MISSING_FRAMES
        IOU_RECENT_LOST = cfg.SUPERVISION_SYSTEM.STAGE1.IOU_THRESHOLD
        SHORT_POOL_SIZE = cfg.SUPERVISION_SYSTEM.STAGE1.POOL_SIZE
        
        # ================================================================
        # BATCH POOL BUILDING: Fetch candidates once for ALL tracks
        # ================================================================
        
        # Identify tracks that need new pools
        tracks_needing_pools: List[str] = []
        tracks_with_pools: List[str] = []
        
        for entry in entries:
            track_id = str(entry["track_id"])
            if entry.get("body_vector") is None:
                continue
            
            short_pool_entries = self.pending_store.short_pool_snapshot(stream_id, track_id)
            if not short_pool_entries:
                tracks_needing_pools.append(track_id)
            else:
                # Extract and prune existing pool
                pool_ids = [str(e.get("track_id")) for e in short_pool_entries if e.get("track_id")]
                pool_ids = [tid for tid in pool_ids if tid not in active_set and tid not in retired_set]
                track_to_pool[track_id] = pool_ids
                all_pool_track_ids.update(pool_ids)
                tracks_with_pools.append(track_id)
            
            # Build Stage 2 search query for this track
            stage2_queries[track_id] = {
                "vector": entry["body_vector"],
                "filter": {"stream_id": stream_id},
                "exclude_tracks": {track_id, *active_set, *retired_set},
            }
        
        # BATCH: Fetch pool candidates ONCE for all tracks needing pools
        if tracks_needing_pools:
            try:
                # Fetch more candidates to distribute across multiple tracks
                batch_limit = SHORT_POOL_SIZE * max(len(tracks_needing_pools) * 2, 10)
                all_ram_candidates = self.assigned_store.get_inactive_assigned_by_stream(
                    stream_id=stream_id,
                    exclude_track_ids=active_set | retired_set,
                    limit=batch_limit,
                    max_missing_frames=RECENT_LOST_FRAMES,
                )
                
                
                
                # Distribute candidates to each track
                for track_id in tracks_needing_pools:
                    # Filter: exclude self and already-used tracks
                    track_candidates = [
                        c for c in all_ram_candidates 
                        if c.get("track_id") != track_id
                    ]
                    
                    # Deduplicate and pick top N
                    pool_ids = []
                    seen = set()
                    for c in track_candidates:
                        tid = c.get("track_id")
                        gid = self._normalize_global_id(c.get("global_id"))
                        if tid and gid and str(tid) not in seen:
                            seen.add(str(tid))
                            pool_ids.append(str(tid))
                            if len(pool_ids) >= SHORT_POOL_SIZE:
                                break
                    
                    if pool_ids:
                        # Save pool to pending_store
                        pool_entries = [{"track_id": tid} for tid in pool_ids]
                        self.pending_store.extend_short_pool(stream_id, track_id, pool_entries, SHORT_POOL_SIZE)
                        
                    
                    track_to_pool[track_id] = pool_ids
                    all_pool_track_ids.update(pool_ids)
                    
            except Exception as exc:
                logger.warning("[TrackManager] Batch pool building failed: %s", exc)
                # Fallback: tracks will have empty pools
                for track_id in tracks_needing_pools:
                    track_to_pool[track_id] = []

        
        # ================================================================
        # PHASE 2: BATCH EXECUTE
        # ================================================================
        
        # Log pool mapping for debugging
        
        
        # Stage 1: Batch vector retrieval for ALL pool tracks
        if all_pool_track_ids:
            try:
                batch_pool_vectors = self.vector_store.get_vectors_for_tracks(
                    track_ids=list(all_pool_track_ids),
                    stream_id=stream_id
                )
                
                
                
                # Distribute vectors to each track's cache
                for track_id, pool_ids in track_to_pool.items():
                    if pool_ids:
                        track_vectors = {
                            tid: batch_pool_vectors[tid]
                            for tid in pool_ids
                            if tid in batch_pool_vectors
                        }
                        self.pending_store.set_pool_vectors_cache(stream_id, track_id, track_vectors)
                        
            except Exception as exc:
                logger.warning("[TrackManager] Batch vector fetch failed: %s", exc)
        
        # Stage 2: Batch search
        precomputed_matches: Dict[str, List[Dict[str, Any]]] = {}
        if stage2_queries:
            try:
                # Convert dict to list format expected by batch_search_flat
                queries_list = [
                    {"track_id": tid, "vector": q["vector"], "filter": q["filter"]}
                    for tid, q in stage2_queries.items()
                ]
                precomputed_matches = self.vector_store.batch_search_flat(queries_list)
                
                # Post-filter: Remove excluded tracks from results
                for tid, query_data in stage2_queries.items():
                    if tid in precomputed_matches:
                        excluded = query_data.get("exclude_tracks", set())
                        if excluded:
                            # Filter out excluded track_ids
                            precomputed_matches[tid] = [
                                match for match in precomputed_matches[tid]
                                if match.get("payload", {}).get("track_id") not in excluded
                            ]
                
                

            except Exception as exc:
                logger.warning("[TrackManager] Batch search/validate failed, falling back: %s", exc)
                precomputed_matches = {}
        
        # ================================================================
        # PHASE 3: DISTRIBUTE & PROCESS - Per-track with precomputed data
        # ================================================================
        
        for entry in entries:
            track_id = str(entry["track_id"])
            
            track_matches = precomputed_matches.get(track_id) if precomputed_matches else None
            
            ctx = self._prepare_assignment(
                track_id=track_id,
                stream_id=stream_id,
                body_vector=entry["body_vector"],
                track_metadata=entry.get("track_metadata"),
                current_frame_idx=entry.get("current_frame_idx", frame_number),
                active_track_ids=active_track_ids,
                force_recompute=entry.get("force_recompute", False),
                precomputed_matches=track_matches,
            )
            contexts.append(ctx)

        results = self._finalize_assignments(stream_id, contexts)
        return results

    def _prepare_assignment(self, track_id: str, stream_id: str, body_vector: np.ndarray,
                           track_metadata: Optional[Any], current_frame_idx: Optional[int],
                           active_track_ids: Optional[List[str]],
                           force_recompute: bool,
                           precomputed_matches: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:

        context: Dict[str, Any] = {
            "track_id": str(track_id),
            "stream_id": stream_id,
            "track_metadata": track_metadata,
            "status": None,
            "assignment_info": None,
            "existing_global_id": None,
            "existing_state": "PENDING",
            "resolved_state": "PENDING",
            "candidate_global_id": None,
            "candidate_track_id": None,
            "candidate_stream_id": None,
            "best_score": 0.0,
            "search_method": None,
            "vote_winner": None,
            "clear_pending": False,
            "assign_reason": None,
        }


        # ✅ CHECK 0: Retired track detection (resurrection guard)
        # If this track_id was previously retired, it MUST start fresh as PENDING
        retired_key = (str(stream_id), str(track_id))
        is_retired_resurrection = False
        resurrection_global_id: Optional[str] = None
        with self._retired_lock:
            retired_entry = self.retired_tracks.pop(retired_key, None)
        if retired_entry is not None:
            is_retired_resurrection = True
            retired_ts, resurrection_global_id = retired_entry
            
            # Wipe all data - treat track as completely new
            old_history_gid = self.track_gid_history.pop((str(stream_id), str(track_id)), None)
            self.assigned_store.retire(stream_id, track_id)
            self.pending_store.drop(stream_id, track_id)
            self._clear_pending(stream_id, track_id)
            
            if track_metadata is not None:
                track_metadata.global_id = None
                track_metadata.state = "PENDING"

                if hasattr(track_metadata, "candidate_cache"):
                    track_metadata.candidate_cache = None
                    track_metadata.candidate_cache_token = None
        
        # Priority 1: RAM cache (TrackMetadata - freshest source of truth)
        existing_global_id = None
        existing_state = "PENDING"
        
        # Skip reading existing data if resurrection (đã wipe sạch ở trên)
        if not is_retired_resurrection:
            if track_metadata is not None:
                existing_global_id = self._normalize_global_id(
                    getattr(track_metadata, "global_id", None)
                )
                existing_state = getattr(track_metadata, "state", "PENDING")
            
            # Priority 2: History reuse - nếu track_id từng được gán GID, ưu tiên giữ lại
            if existing_global_id is None:
                history_gid = self.track_gid_history.get((str(stream_id), str(track_id)))
                if history_gid:
                    existing_global_id = history_gid
                    existing_state = "ASSIGNED"
                    context["resolved_state"] = "ASSIGNED"
        
        # ✅ CHECK 2: Global ID ownership validation
        # If this track claims a global_id, verify no other track in same stream owns it
        if existing_global_id:
            owner_state = self.assigned_store.get_by_global_id(existing_global_id)
            if owner_state is not None:
                owner_track_id = owner_state.track_id
                owner_stream_id = owner_state.stream_id
                
                # Conflict: different track owns this global_id
                if owner_stream_id == str(stream_id) and owner_track_id != str(track_id):
                    logger.error(
                        "[TrackManager] GLOBAL_ID_CONFLICT: track=%s claims global_id=%s but owner is track=%s → revoking",
                        track_id, existing_global_id, owner_track_id
                    )
                    # Revoke this track's claim
                    existing_global_id = None
                    existing_state = "PENDING"
                    
                    # Clear stale data from RAM cache
                    if track_metadata is not None:
                        track_metadata.global_id = None
                        track_metadata.state = "PENDING"

        context["existing_global_id"] = existing_global_id
        context["existing_state"] = existing_state
        context["resolved_state"] = existing_state if existing_state else (
            "ASSIGNED" if existing_global_id else "PENDING"
        )

        # Config thresholds
        RECENT_LOST_FRAMES = cfg.SUPERVISION_SYSTEM.STAGE1.KALMAN_MAX_MISSING_FRAMES
        IOU_RECENT_LOST = cfg.SUPERVISION_SYSTEM.STAGE1.IOU_THRESHOLD
        
        # HNSW base threshold
        HNSW_SCORE_THRESHOLD = cfg.SUPERVISION_SYSTEM.SEARCH.HNSW_SCORE_THRESHOLD
        MIN_SIM_FILTER = HNSW_SCORE_THRESHOLD
        
        # Stage 1 (Motion-based)
        SIMILARITY_THRESH_STAGE1 = cfg.SUPERVISION_SYSTEM.STAGE1.SIMILARITY_THRESHOLD
        # Stage 2 (Appearance-based)
        SIMILARITY_THRESH_STAGE2 = cfg.SUPERVISION_SYSTEM.STAGE2.SIMILARITY_THRESHOLD
        SHORT_POOL_SIZE = cfg.SUPERVISION_SYSTEM.STAGE1.POOL_SIZE
        FINAL_SCORE_MAX_WEIGHT = cfg.SUPERVISION_SYSTEM.SEARCH.FINAL_SCORE_MAX_WEIGHT
        
        

        active_set = {str(tid) for tid in (active_track_ids or [])}
        is_visible = str(track_id) in active_set

        lost_frames = 0
        if (not is_visible) and (track_metadata is not None):
            lost_frames = int(getattr(track_metadata, "missing_count", 0))

        # State machine
        if is_visible:
            track_state = "ACTIVE"
        elif lost_frames < RECENT_LOST_FRAMES:
            track_state = "LOST_SHORT"
        else:
            track_state = "LOST_LONG"

        self._update_track_layers(
            stream_id=stream_id,
            track_id=track_id,
            global_id=existing_global_id,
            is_visible=is_visible,
            lost_frames=lost_frames,
            frame_idx=current_frame_idx,
            body_vector=body_vector,
            track_metadata=track_metadata,
        )

        detector_bbox = getattr(track_metadata, "latest_bbox", None) if track_metadata else None
        kalman_bbox = getattr(track_metadata, "kalman_predicted_bbox", None) if track_metadata else None
        bbox_context = {"detector": detector_bbox, "kalman": kalman_bbox}

        use_kalman_iou = False
        iou_threshold = 0.0

        if track_state == "ACTIVE":
            use_kalman_iou = False
        elif track_state == "LOST_SHORT":
            if kalman_bbox or detector_bbox:
                use_kalman_iou = True
                iou_threshold = IOU_RECENT_LOST

        # Candidate search (1 lần duy nhất)
        matches: List[Dict] = []
        cache_token = current_frame_idx if track_metadata is not None else None
        
        # Log retired tracks hiện tại để debug (LUÔN LOG)
        with self._retired_lock:
            total_retired = len(self.retired_tracks)
        
        if track_metadata is not None:
            if getattr(track_metadata, "candidate_cache", None) and (
                track_metadata.candidate_cache_token == cache_token
            ):
                matches = [dict(m) for m in track_metadata.candidate_cache]

        # ====================================================================
        # 2-STAGE VOTING: Motion-based + Appearance-based (song song mỗi round)
        # ====================================================================
        
        # Lấy short pool hiện tại (nếu có) - dùng cho Stage 1
        short_pool_entries = self.pending_store.short_pool_snapshot(stream_id, track_id)
        
        short_pool_track_ids: List[str] = []
        
        # Build retired_set để filter
        retired_set = set()
        with self._retired_lock:
            for (ret_stream_id, ret_track_id), _ in self.retired_tracks.items():
                if ret_stream_id == str(stream_id):
                    retired_set.add(ret_track_id)
        
        # ================================================================
        # STAGE 1: Motion-based (Short Pool)
        # - Pool already built at batch level (assign_global_ids_batch)
        # - Just retrieve cached pool + vectors
        # ================================================================
        
        # Retrieve cached pool vectors (already populated by batch)
        pool_vectors_cache = self.pending_store.get_pool_vectors_cache(stream_id, track_id) or {}
        
        # Extract pool track IDs from existing pool
        if short_pool_entries:
            short_pool_track_ids = [
                str(entry.get("track_id"))
                for entry in short_pool_entries
                if entry.get("track_id") is not None
            ]    
            
        else:
            # No pool entries - track may not have been processed in batch
            short_pool_track_ids = []  
        
        # Prune pool: loại bỏ track đã active hoặc retired
        if short_pool_track_ids:
            to_remove = [
                tid for tid in short_pool_track_ids
                if tid in active_set or tid in retired_set
            ]
            if to_remove:
                self.pending_store.remove_from_short_pool(stream_id, track_id, to_remove)
                # Cũng xóa khỏi vectors cache
                self.pending_store.remove_from_pool_vectors_cache(stream_id, track_id, to_remove)
                short_pool_track_ids = [tid for tid in short_pool_track_ids if tid not in to_remove]
                
        
        # Log Stage 1 pool state (combined)
        if short_pool_track_ids:
            cache_keys = list(pool_vectors_cache.keys()) if pool_vectors_cache else []
            
        
        # Tính Stage 1 candidates (motion-based, local sim) - DÙNG CACHE
        candidates_stage1: List[Dict[str, Any]] = []
        stage1_track_ids: set = set()
        
        if short_pool_track_ids and pool_vectors_cache:
            candidates_stage1 = self._compute_stage1_candidates_from_cache(
                body_vector=body_vector,
                pool_track_ids=short_pool_track_ids,
                vectors_cache=pool_vectors_cache,
                stream_id=stream_id,
                similarity_threshold=SIMILARITY_THRESH_STAGE1,  # 0.78
                active_set=active_set,
                retired_set=retired_set,
                min_sim_filter=MIN_SIM_FILTER,  # 0.70 directly
                weight_max=FINAL_SCORE_MAX_WEIGHT,
            )
            # Lưu track_ids từ Stage 1 để loại khỏi Stage 2
            stage1_track_ids = {str(c.get("track_id")) for c in candidates_stage1 if c.get("track_id")}
        # ================================================================
        # STAGE 2: Appearance-based (Full Search)
        # - Full RAM search mỗi round
        # - Filter by Stage 2 threshold (0.85)
        # - LOẠI các track_ids đã có trong Stage 1 (tránh duplicate + tiết kiệm compute)
        # ================================================================
        
        # STAGE 2 SEARCH: Use precomputed_matches if available (from batch), else fallback
        fresh_matches = False
        if not matches:
            if precomputed_matches is not None:
                # Use batch search results
                matches = precomputed_matches
                fresh_matches = True
                
            else:
                # Fallback to single-track search
                matches = self._search_body_reid(body_vector, stream_id, track_id, active_track_ids)
                fresh_matches = True
        
        # Post-process matches: safety filter for active tracks
        if matches and fresh_matches:
            matches = self._filter_inactive_assigned(matches, stream_id, active_track_ids)
            
            if track_metadata is not None:
                track_metadata.candidate_cache = [dict(m) for m in matches]
                track_metadata.candidate_cache_token = cache_token
        
        candidates_stage2: List[Dict[str, Any]] = []
        stage2_entries = self._build_candidate_entries(matches, stream_id)
        
        # Log all track_ids từ HNSW search (trước khi filter)
        stage2_all_track_ids = [str(e.get("track_id", "")) for e in stage2_entries]
        
        
        # Filter by Stage 2 threshold VÀ loại track_ids đã có ở Stage 1
        logged_track_ids: set = set()  # Avoid duplicate logs for same track_id
        for cand in stage2_entries:
            cand_track_id = str(cand.get("track_id", ""))
            cand_gid = cand.get("global_id", "?")
            score = float(cand.get("score", 0.0))
            
            # Skip nếu track_id đã có trong Stage 1
            if cand_track_id in stage1_track_ids:
                if cand_track_id not in logged_track_ids:
                    
                    logged_track_ids.add(cand_track_id)
                continue
            
            if score >= SIMILARITY_THRESH_STAGE2:  # 0.85
                cand["stage"] = "appearance"
                candidates_stage2.append(cand)
                if cand_track_id not in logged_track_ids:
                    
                    logged_track_ids.add(cand_track_id)
            else:
                if cand_track_id not in logged_track_ids:
                    
                    logged_track_ids.add(cand_track_id)
        
        # ================================================================
        # MERGE: Union Stage 1 + Stage 2
        # (Duplicate đã được loại ở Stage 2 bằng track_id filter)
        # ================================================================
        
        voting_candidates: List[Dict[str, Any]] = candidates_stage1 + candidates_stage2

        # ====================================================================
        # BUILD CANDIDATE META VÀ VOTING
        # ====================================================================

        # Không có candidates và track đã có GID → giữ nguyên
        if not voting_candidates:
            if existing_global_id and not force_recompute:
                context["status"] = "retain"
                context["assignment_info"] = {
                    "global_id": existing_global_id,
                    "method": "retain",
                    "state": context["resolved_state"],
                    "score": 0.0,
                    "vote_winner": None,
                }
                return context

        # Build candidate_meta từ voting_candidates (đã dedupe by GID ở bước MERGE)
        candidate_meta: Dict[str, Dict[str, Any]] = {
            cand.get("global_id"): cand
            for cand in voting_candidates
            if cand.get("global_id")
        }

        candidates_list: List[Tuple[str, float]] = sorted(
            [(gid, float(meta.get("score", 0.0))) for gid, meta in candidate_meta.items()],
            key=lambda item: item[1],
            reverse=True,
        )
        gid_to_track = {gid: meta.get("track_id") for gid, meta in candidate_meta.items()}
        gid_to_meta = candidate_meta

        # Log voting candidates
        if candidates_list:
            top_candidates_display = [
                f"{gid}(track={gid_to_track.get(gid, 'unknown')},score={score:.3f})"
                for gid, score in candidates_list[:5]  # Show top 5
            ]
            

        best_score = 0.0
        if candidate_meta:
            best_score = max(float(meta.get("score", 0.0)) for meta in candidate_meta.values())
        context["best_score"] = best_score
        context["search_method"] = "reid" if matches else "none"

        # ✅ UNIFIED VOTING FLOW: All PENDING tracks go through voting
        # regardless of whether they have candidates or not
        if not existing_global_id:
            self._add_pending_votes(stream_id, track_id, candidates_list, frame_idx=current_frame_idx, gid_to_track=gid_to_track)

            agg = self._compute_vote_aggregates(stream_id, track_id, gid_to_track=gid_to_track)
            winner_gid = agg["winner"]
            vote_count = int(agg["vote_count"])
            winner_filtered_avg = float(agg.get("winner_filtered_avg", 0.0))
            early_exit = bool(agg.get("early_exit", False))
            
            window_full = vote_count >= self.VOTE_WINDOW_SIZE
            # IMPORTANT: use fixed-denominator ratio returned by agg
            winner_ratio = float(agg.get("ratio", 0.0))
            # has_consensus CHỈ trigger khi window full VÀ ratio >= threshold
            # Không trigger sớm khi chưa đủ votes!
            has_consensus = (window_full and winner_gid is not None and winner_ratio >= self.VOTE_RATIO_THRESHOLD)
            
            winner_track = gid_to_track.get(winner_gid, "unknown") if winner_gid else None
            winner_display = f"{winner_gid}(track={winner_track})" if winner_track else winner_gid         
            
            # Case 1a: EARLY EXIT - đủ điều kiện assign sớm (không cần chờ hết window)
            # Case 1b: Consensus reached (chờ hết window hoặc đạt consensus ratio)
            if early_exit or has_consensus:
                assign_reason = "early_exit" if early_exit else "consensus"
                
                return self._set_assign_winner_context(
                    context, winner_gid, gid_to_meta, matches, assign_reason
                )
            
            # Case 2: Window full, no consensus → Check if winner meets MIN_VOTE_RATIO_TO_ASSIGN
            if window_full:
                # Nếu có winner và đạt ngưỡng tối thiểu → assign winner
                if winner_gid and winner_ratio >= self.MIN_VOTE_RATIO_TO_ASSIGN:
                    
                    return self._set_assign_winner_context(
                        context, winner_gid, gid_to_meta, matches, "window_full_winner"
                    )
                
                # Không có winner hoặc winner ratio quá thấp → Assign new Global ID
                
                context["status"] = "assign_new"
                context["assign_reason"] = "voting_no_valid_winner"
                context["clear_pending"] = True
                context["best_score"] = 0.0
                return context
            
            # Case 2b: TIMEOUT COMPLETE - đủ MIN_VOTES và vượt timeout window
            # (track có thể đã biến mất trước khi đạt 10 votes)
            # Frame-based timeout (primary)
            first_vote_frame = agg.get("first_vote_frame")
            if (self.VOTING_TIMEOUT_ENABLED and 
                first_vote_frame is not None and 
                current_frame_idx is not None and
                vote_count >= self.VOTING_MIN_VOTES_TO_COMPLETE):
                
                elapsed_frames = current_frame_idx - first_vote_frame
                if elapsed_frames >= self.VOTING_TIMEOUT_FRAMES:
                    
                    
                    # Apply same logic as window_full: assign winner if ratio >= MIN_VOTE_RATIO
                    if winner_gid and winner_ratio >= self.MIN_VOTE_RATIO_TO_ASSIGN:
                        
                        return self._set_assign_winner_context(
                            context, winner_gid, gid_to_meta, matches, "timeout_complete"
                        )
                    
                    # No valid winner → assign new Global ID
                    
                    context["status"] = "assign_new"
                    context["assign_reason"] = "timeout_no_valid_winner"
                    context["clear_pending"] = True
                    context["best_score"] = 0.0
                    return context
            
            # Case 3: Continue voting (window not full yet, no early exit, no timeout)
            context["status"] = "pending"
            context["assignment_info"] = {
                "global_id": None,
                "method": "pending_voting",
                "state": "PENDING",
                "score": best_score,
                "vote_progress": f"{vote_count}/{self.VOTE_WINDOW_SIZE}",
                "vote_winner": winner_gid,
                "vote_ratio": f"{winner_ratio:.3f}",
                "vote_avg": f"{winner_filtered_avg:.3f}",
            }
            return context
        
        # ASSIGNED Track: Retain
        if existing_global_id:
            context["status"] = "retain"
            context["assignment_info"] = {
                "global_id": existing_global_id,
                "method": "retain",
                "state": context["resolved_state"],
                "score": best_score,
                "vote_winner": None,
            }
            return context
        
        # Fallback
        context["status"] = "pending"
        context["assignment_info"] = {
            "global_id": None,
            "method": "pending_no_candidates",
            "state": "PENDING",
            "score": best_score,
            "vote_winner": None,
        }
        return context

    # ------------------------------------------------------------------ #
    # Finalization
    # ------------------------------------------------------------------ #
    
    def _finalize_assignments(self, stream_id: str, contexts: List[Dict[str, Any]]
                             ) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        new_assignments: List[Dict[str, Any]] = []

        for ctx in contexts:
            status = ctx.get("status")
            track_id = ctx["track_id"]
            if status == "assign_global":
                gid = ctx.get("candidate_global_id")
                if gid:
                    grouped.setdefault(gid, []).append(ctx)
                else:
                    ctx["status"] = "assign_new"
                    new_assignments.append(ctx)
            elif status == "assign_new":
                new_assignments.append(ctx)
            else:
                info = ctx.get("assignment_info") or {
                    "global_id": ctx.get("existing_global_id"),
                    "method": status or "pending",
                    "state": ctx.get("resolved_state", "PENDING"),
                    "score": ctx.get("best_score", 0.0),
                    "vote_winner": ctx.get("vote_winner"),
                }
                results[track_id] = info

        # Resolve conflicts per-GID
        for gid, group in grouped.items():
            winner = max(group, key=lambda c: c.get("best_score", 0.0))
            for ctx in group:
                if ctx is winner:
                    info = self._finalize_assign_global(ctx)
                else:
                    info = self._handle_conflict_loser(ctx, gid)
                results[ctx["track_id"]] = info

        # New GIDs
        for ctx in new_assignments:
            info = self._finalize_assign_new(ctx)
            results[ctx["track_id"]] = info

        return results

    def _finalize_assign_global(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        track_id = ctx["track_id"]
        stream_id = ctx["stream_id"]
        global_id = ctx.get("candidate_global_id")
        track_metadata = ctx.get("track_metadata")
        best_score = ctx.get("best_score", 0.0)

        # ✅ CRITICAL: Check ownership conflict before finalizing assignment
        owner_state = self.assigned_store.get_by_global_id(global_id) if global_id else None
        if owner_state is not None:
            owner_track_id = owner_state.track_id
            owner_stream_id = owner_state.stream_id
            
            # Conflict: different track in same stream already owns this global_id AND is still active
            if owner_stream_id == str(stream_id) and owner_track_id != str(track_id) and owner_state.is_active:
                logger.error(
                    "[TrackManager] ASSIGNMENT_CONFLICT: Cannot assign global_id=%s to track=%s "
                    "(owner is track=%s) → assigning NEW global_id instead",
                    global_id, track_id, owner_track_id
                )
                # Downgrade to assign_new
                ctx["status"] = "assign_new"
                ctx["assign_reason"] = f"ownership_conflict_with_{owner_track_id}"
                return self._finalize_assign_new(ctx)

        # Retire matching track
        candidate_track = ctx.get("candidate_track_id")
        candidate_stream = ctx.get("candidate_stream_id")
        
        # Calculate how many vectors to transfer based on new track's history
        new_vector_count = 0
        if track_metadata is not None:
            new_vector_count = len(track_metadata.person_vector_history)
            # +1 for last_person_vector if exists and not in history
            if track_metadata.last_person_vector is not None:
                new_vector_count += 1
        
        if candidate_track and candidate_stream:
            # Get cached vectors from Stage 1 pool_vectors_cache (indexed by candidate track_id)
            cached_vecs = None
            pool_cache = self.pending_store.get_pool_vectors_cache(stream_id, track_id)
            if pool_cache and str(candidate_track) in pool_cache:
                cache_entry = pool_cache[str(candidate_track)]
                cached_vecs = cache_entry.get("vectors", [])
            
            # Get query vector from track_metadata for similarity filtering
            query_vec = None
            if track_metadata is not None:
                query_vec = getattr(track_metadata, "last_person_vector", None)
            
            self._retire_previous_track(
                track_id, stream_id, candidate_track, candidate_stream,
                new_track_vector_count=new_vector_count,
                track_metadata=track_metadata,
                new_global_id=global_id,  # Pass global_id for copied vectors
                cached_vectors=cached_vecs,
                query_vector=query_vec,
            )

        if ctx.get("clear_pending"):
            self._clear_pending(stream_id, track_id)

        if track_metadata is not None:
            track_metadata.global_id = global_id
            track_metadata.state = "ASSIGNED"
            track_metadata.assignment_confidence = best_score
            track_metadata.candidate_cache = []
            track_metadata.candidate_cache_token = None

        self.vector_store.update_global_id(track_id, stream_id, global_id, "ASSIGNED")
        # Lưu lịch sử để reuse nếu track_id tái xuất
        self.track_gid_history[(str(stream_id), str(track_id))] = global_id

        # Batch commit historical vectors
        if track_metadata and self.metadata_cache:
            try:
                history_count = len(track_metadata.person_vector_history)
                
                if history_count > 0:
                    self.metadata_cache.commit_person_vector_history(
                        track=track_metadata,
                        stream_id=stream_id,
                        reason="assignment_final",
                    )
                
                # Only commit last_person_vector if NOT already committed via history
                if track_metadata.last_person_vector is not None:
                    # Check flag set by commit_person_vector_history
                    if getattr(track_metadata, '_last_vector_committed_via_history', False):
                        
                        track_metadata._last_vector_committed_via_history = False  # Reset flag
                    else:
                        self.metadata_cache.commit_person_vectors(
                            track=track_metadata,
                            stream_id=stream_id,
                            snapshot_suffix=f"assigned_{int(time.time())}",
                            reason="assignment_final",
                        )
            except Exception as exc:
                logger.warning("[TrackManager] Failed to commit vectors: %s", exc)

        self._register_assignment_layers(
            stream_id=stream_id,
            track_id=track_id,
            global_id=global_id,
            track_metadata=track_metadata,
        )

        return {
            "global_id": global_id,
            "method": ctx.get("search_method"),
            "state": "ASSIGNED",
            "score": best_score,
            "vote_winner": ctx.get("candidate_global_id"),
        }

    def _finalize_assign_new(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Assign a new Global ID to a PENDING track."""
        track_id = ctx["track_id"]
        stream_id = ctx["stream_id"]
        track_metadata = ctx.get("track_metadata")
        best_score = ctx.get("best_score", 0.0)

        new_global_id = self.generate_new_global_id()
        if ctx.get("clear_pending"):
            self._clear_pending(stream_id, track_id)

        self.vector_store.update_global_id(track_id, stream_id, new_global_id, "ASSIGNED")
        # Lưu lịch sử để reuse nếu track_id tái xuất
        self.track_gid_history[(str(stream_id), str(track_id))] = new_global_id

        if track_metadata is not None:
            track_metadata.global_id = new_global_id
            track_metadata.state = "ASSIGNED"
            track_metadata.assignment_confidence = best_score
            track_metadata.candidate_cache = []
            track_metadata.candidate_cache_token = None

        # Batch commit historical vectors
        if track_metadata and self.metadata_cache:
            try:
                history_count = len(track_metadata.person_vector_history)
                
                if history_count > 0:
                    self.metadata_cache.commit_person_vector_history(
                        track=track_metadata,
                        stream_id=stream_id,
                        reason="assignment_new",
                    )
                
                # Only commit last_person_vector if NOT already committed via history
                if track_metadata.last_person_vector is not None:
                    # Check flag set by commit_person_vector_history
                    if getattr(track_metadata, '_last_vector_committed_via_history', False):
                        
                        track_metadata._last_vector_committed_via_history = False  # Reset flag
                    else:
                        self.metadata_cache.commit_person_vectors(
                            track=track_metadata,
                            stream_id=stream_id,
                            snapshot_suffix=f"new_{int(time.time())}",
                            reason="assignment_new",
                        )
            except Exception as exc:
                logger.warning("[TrackManager] Failed to commit vectors: %s", exc)

        self._register_assignment_layers(
            stream_id=stream_id,
            track_id=track_id,
            global_id=new_global_id,
            track_metadata=track_metadata,
        )

        return {
            "global_id": new_global_id,
            "method": ctx.get("assign_reason", "assign_new"),
            "state": "ASSIGNED",
            "score": best_score,
            "vote_winner": None,
        }

    def _handle_conflict_loser(self, ctx: Dict[str, Any], winner_gid: str) -> Dict[str, Any]:
        track_id = ctx["track_id"]
        stream_id = ctx["stream_id"]
        track_metadata = ctx.get("track_metadata")
        best_score = ctx.get("best_score", 0.0)

        if track_metadata is not None:
            track_metadata.global_id = None
            track_metadata.state = "PENDING"
            track_metadata.assignment_confidence = best_score
            track_metadata.candidate_cache = []
            track_metadata.candidate_cache_token = None

        try:
            self.vector_store.update_global_id(track_id, stream_id, None, "PENDING")
        except Exception as exc:
            logger.warning("Failed to reset global_id for conflicted track %s: %s", track_id, exc)

        

        return {
            "global_id": None,
            "method": "conflict_loser",
            "state": "PENDING",
            "score": best_score,
            "vote_winner": winner_gid,
        }

    # ------------------------------------------------------------------ #
    # Search & Filtering Helpers
    # ------------------------------------------------------------------ #
    
    def _build_candidate_entries(self, match_list: List[Dict], stream_id: str) -> List[Dict[str, Any]]:
        """
        Transform search results into candidate entry format.
        
        Args:
            match_list: List of search results with score and payload
            stream_id: Default stream_id to use if not in payload
            
        Returns:
            List of candidate entries with global_id, track_id, stream_id, score, payload
        """
        entries: List[Dict[str, Any]] = []
        for m in match_list or []:
            if not isinstance(m, dict):
                continue
            payload = m.get("payload") or {}
            gid = self._normalize_global_id(payload.get("global_id"))
            track_candidate = payload.get("track_id")
            if not gid or track_candidate is None:
                continue
            entries.append({
                "global_id": gid,
                "track_id": str(track_candidate),
                "stream_id": str(payload.get("stream_id") or stream_id),
                "score": float(m.get("score", 0.0)),
                "payload": payload,
            })
        return entries

    def _compute_stage1_candidates_from_cache(
        self,
        body_vector: np.ndarray,
        pool_track_ids: List[str],
        vectors_cache: Dict[str, Dict[str, Any]],
        stream_id: str,
        similarity_threshold: float,
        active_set: set,
        retired_set: set,
        min_sim_filter: float = 0.70,
        weight_max: float = 0.70,
    ) -> List[Dict[str, Any]]:
        """
        Stage 1: Motion-based candidates - tính similarity từ CACHED vectors.
        
        Áp dụng CÙNG logic với Stage 2 multi-snap validation:
        - CHỈ dùng vectors có sim >= min_sim_filter (0.7) để loại outliers
        - final_score = weight_max × max_sim + (1 - weight_max) × avg_sim
        
        Args:
            body_vector: Vector query của track hiện tại
            pool_track_ids: Danh sách track_ids trong short pool
            vectors_cache: Cached vectors từ pool_vectors_cache
            stream_id: Stream ID
            similarity_threshold: Ngưỡng Stage 1 (0.78)
            active_set: Set các track_ids đang active
            retired_set: Set các track_ids đã retired
            min_sim_filter: Ngưỡng tối thiểu để lọc vectors lỗi (0.7)
            weight_max: Trọng số cho max_sim (0.7)
            
        Returns:
            List[Dict]: Candidates với final_score >= threshold
        """
        if not pool_track_ids or body_vector is None or not vectors_cache:
            return []
        
        # Prune: loại bỏ track đã active hoặc retired
        valid_track_ids = [
            tid for tid in pool_track_ids
            if tid not in active_set and tid not in retired_set
        ]
        
        # Log filtered tracks for debugging
        filtered_active = [tid for tid in pool_track_ids if tid in active_set]
        filtered_retired = [tid for tid in pool_track_ids if tid in retired_set]
        
        if not valid_track_ids:
            return []
        
        candidates: List[Dict[str, Any]] = []
        
        for track_id_str in valid_track_ids:
            data = vectors_cache.get(track_id_str)
            if not data:
                
                continue
                
            vectors = data.get("vectors", [])
            if not vectors:
                
                continue
            
            # Tính similarity với TẤT CẢ vectors, filter bởi min_sim_filter
            all_similarities = []
            valid_similarities = []
            for vec in vectors:
                sim = cosine_similarity_normalized(body_vector, vec)
                all_similarities.append(sim)
                if sim >= min_sim_filter:  # 0.7 - loại vectors lỗi (occlusion, blur)
                    valid_similarities.append(sim)
            
            # Fallback: nếu không có valid sims, dùng all (candidate có thể không tốt)
            similarities = valid_similarities if valid_similarities else all_similarities
            
            if not similarities:
                
                continue
            
            # Calculate final score: 0.7 × max + 0.3 × avg
            max_sim = max(similarities)
            avg_sim = sum(similarities) / len(similarities)
            final_score = weight_max * max_sim + (1 - weight_max) * avg_sim
            
            global_id = data.get("global_id")
            
            # Log the score comparison
            if final_score < similarity_threshold:
                # Show all individual similarities for debugging
                sim_sorted = sorted(all_similarities, reverse=True)
                cached_uuids = data.get("uuids", [])
                
                continue
            
            if not global_id:
                
                continue
            
            # PASSED
            
            
            payload = data.get("payload", {})
            candidates.append({
                "global_id": global_id,
                "track_id": track_id_str,
                "stream_id": str(payload.get("stream_id") or stream_id),
                "score": float(final_score),
                "max_sim": float(max_sim),
                "avg_sim": float(avg_sim),
                "valid_snaps": len(valid_similarities),
                "total_snaps": len(all_similarities),
                "payload": payload,
                "stage": "motion",  # Đánh dấu nguồn gốc
            })
        
        # Sort by score descending
        candidates.sort(key=lambda x: x["score"], reverse=True)
             
        return candidates
    
    def _search_body_reid(self, body_vector, stream_id, exclude_track_id,
                         active_track_ids: Optional[List[str]] = None):
        """Search for similar tracks with simple dict filter (RAM backend compatible)."""
        stream_id_str = str(stream_id)
        exclude_track_id_str = str(exclude_track_id)
        
        # Build excluded tracks set for filtering
        excluded_tracks = {exclude_track_id_str}
        
        # Add active tracks
        if active_track_ids:
            excluded_tracks.update(str(tid) for tid in active_track_ids)
        
        # Add retired tracks
        with self._retired_lock:
            for (ret_stream_id, ret_track_id), _ in list(self.retired_tracks.items()):
                if ret_stream_id == stream_id_str:
                    excluded_tracks.add(ret_track_id)
        
        # Config: SUPERVISION_SYSTEM.SEARCH
        search_cfg = getattr(cfg.SUPERVISION_SYSTEM, "SEARCH", None)
        top_k = getattr(search_cfg, "ENHANCED_RECALL_LIMIT", 20) if search_cfg else 20
        score_threshold = getattr(search_cfg, "SCORE_THRESHOLD", 0.70) if search_cfg else 0.70
        
        # Search using RAM-optimized simple API
        results = self.vector_store.search(
            query_vector=body_vector,
            stream_id=stream_id_str,
            exclude_tracks=excluded_tracks,
            top_k=top_k,
            score_threshold=score_threshold,
        )
        
        # (Removed verbose SEARCH_RESULTS log)

        return results

    def _filter_by_iou(self, current_bbox_info, matches, threshold, use_kalman: bool = True):
        if not matches:
            return []

        current_variants = []
        if current_bbox_info:
            if use_kalman:
                kal = current_bbox_info.get("kalman")
                if kal is not None:
                    norm = normalize_to_ltwh(kal)
                    if norm:
                        current_variants.append(norm)
            if not current_variants:
                det = current_bbox_info.get("detector")
                if det is not None:
                    norm = normalize_to_ltwh(det)
                    if norm:
                        current_variants.append(norm)

        if not current_variants:
            return matches

        filtered = []
        for match in matches:
            payload = match.get("payload", {}) or {}

            candidate_variants = []
            bbox_kalman = payload.get("bbox_kalman")
            if bbox_kalman:
                norm = normalize_to_ltwh(bbox_kalman)
                if norm:
                    candidate_variants.append(norm)
            else:
                bbox_current = payload.get("bbox_current")
                if bbox_current:
                    norm = normalize_to_ltwh(bbox_current)
                    if norm:
                        candidate_variants.append(norm)

            if not candidate_variants:
                match_copy = dict(match)
                match_copy["_iou"] = 0.0
                filtered.append(match_copy)
                continue

            max_iou = 0.0
            for cand_variant in candidate_variants:
                for cur_variant in current_variants:
                    if not are_bboxes_close(cur_variant, cand_variant):
                        continue
                    iou = compute_iou(cur_variant, cand_variant)
                    if iou > max_iou:
                        max_iou = iou

            match_copy = dict(match)
            match_copy["_iou"] = max_iou

            if threshold > 0 and max_iou < threshold:
                continue

            filtered.append(match_copy)

        if not filtered:
            return []

        filtered.sort(key=lambda m: (m.get("_iou", 0.0), m.get("score", 0.0)), reverse=True)
        return filtered

    def _retire_previous_track(self, current_track_id: str, current_stream_id: str,
                               candidate_track_id: Optional[str], 
                               candidate_stream_id: Optional[str],
                               new_track_vector_count: int = 0,
                               track_metadata: Optional[Any] = None,
                               new_global_id: Optional[str] = None,
                               cached_vectors: Optional[List[np.ndarray]] = None,
                               query_vector: Optional[np.ndarray] = None) -> None:
        """Retire old track and copy vectors to new track.
        
        Args:
            current_track_id: New track that takes over the global_id
            current_stream_id: Stream ID
            candidate_track_id: Old track being retired
            candidate_stream_id: Stream ID of old track
            new_track_vector_count: Number of vectors the new track already has
                                   Used to calculate how many to copy
            track_metadata: TrackMetadata of new track (to set _vectors_to_keep flag)
            new_global_id: Global ID being assigned to new track
            cached_vectors: Pre-fetched vectors from Stage 1/Stage 2 cache (optional)
                           If provided, uses transfer_vectors_from_cache instead of RAM retrieval
            query_vector: Query vector for similarity filtering when using cached vectors
        """
        if not candidate_track_id or not candidate_stream_id:
            return
        
        candidate_stream_id_str = str(candidate_stream_id)
        candidate_track_id_str = str(candidate_track_id)
        
        if candidate_stream_id_str != str(current_stream_id):
            return
        if candidate_track_id_str == str(current_track_id):
            return

        retired_gid = None
        owner_state = self.assigned_store.get(candidate_stream_id_str, candidate_track_id_str)
        if owner_state is not None:
            retired_gid = owner_state.global_id
        self.retired_tracks[(candidate_stream_id_str, candidate_track_id_str)] = (
            time.time(),
            retired_gid,
        )

        try:
            # Use snapshot_limit from vector_store as source of truth
            N = self.vector_store.snapshot_limit
            m = new_track_vector_count
            half_N = N // 2
            
            if m <= half_N:
                # Keep ALL current vectors, fill up from old
                keep_new = m
                needed = N - m
            else:
                # Keep about half of current to avoid bias, fill rest from old
                keep_new = math.ceil(m / 2)
                needed = N - keep_new
            
            # Ensure non-negative
            needed = max(0, needed)
            
            # Note: transfer uses cached vectors when available, falls back to RAM retrieval
            transferred = 0  # Will be unpacked from tuple below
            if needed > 0:
                # Use cached vectors if provided (from Stage 1/Stage 2)
                if cached_vectors:
                    # Apply similarity filter to get quality vectors
                    MIN_SIM_FILTER = cfg.SUPERVISION_SYSTEM.STAGE1.SIMILARITY_THRESHOLD
                    transferred, _ = self.vector_store.transfer_vectors_from_cache(
                        from_track_id=candidate_track_id_str,
                        to_track_id=str(current_track_id),
                        stream_id=candidate_stream_id_str,
                        cached_vectors=cached_vectors,
                        max_vectors=needed,
                        to_global_id=new_global_id,
                        min_sim_filter=MIN_SIM_FILTER,
                        query_vector=query_vector,
                    )
                else:
                    # Fallback to RAM-based transfer
                    transferred, _ = self.vector_store.transfer_vectors_to_track(
                        from_track_id=candidate_track_id_str,
                        to_track_id=str(current_track_id),
                        stream_id=candidate_stream_id_str,
                        max_vectors=needed,
                        to_global_id=new_global_id,
                    )
            
            # Defensive: ensure transferred is int (handle tuple return)
            if isinstance(transferred, tuple):
                transferred = transferred[0]
            
            
            
            if track_metadata is not None:
                track_metadata._vectors_to_keep = keep_new
            
            self.vector_store.retire_track(candidate_stream_id_str, candidate_track_id_str)
            
            
            if self.metadata_cache:
                try:
                    self.metadata_cache.remove_track(candidate_track_id_str)
                    
                except Exception as cache_exc:
                    logger.warning("Failed to cleanup track %s from cache: %s",
                                 candidate_track_id_str, cache_exc)
        except Exception as exc:
            logger.warning("Failed to retire track %s: %s", candidate_track_id, exc)

        # Keep assigned layer consistent
        self.assigned_store.retire(candidate_stream_id_str, candidate_track_id_str)

        # Maintain soft limit
        if len(self.retired_tracks) > self._retired_track_limit:
            target_size = self._retired_track_limit // 2
            sorted_tracks = sorted(self.retired_tracks.items(), key=lambda x: x[1][0])
            to_remove = sorted_tracks[:len(self.retired_tracks) - target_size]
            
            for (s_id, t_id), _ in to_remove:
                self.retired_tracks.pop((s_id, t_id), None)
                if self.metadata_cache:
                    try:
                        self.metadata_cache.remove_track(t_id)
                    except Exception:
                        pass
            
            