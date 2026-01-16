# ram_vector_store.py

"""
RAM-based vector storage for ReID tracking.

Provides high-performance in-memory vector operations:
- Pre-allocated NumPy matrix for zero-copy storage
- FIFO management per track (auto-eviction)
- Optimized cosine similarity search
"""

import logging
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class RAMVectorStore:
    """
    In-memory vector storage with FIFO management and efficient search.
    
    Storage layout:
    - vectors: np.ndarray(shape=(capacity, vector_dim)) - main storage
    - metadata: List[Dict] - parallel metadata array
    - track_to_indices: Dict[str, deque] - track ownership (FIFO)
    - free_indices: deque - available slots for reuse
    """
    
    def __init__(
        self,
        capacity: int = 50000,
        vector_dim: int = 512,
        snapshot_limit: int = 10,
        company_id: str = "default",
    ):
        """
        Initialize RAM vector store.
        
        Args:
            capacity: Maximum number of vectors to store
            vector_dim: Dimension of each vector (default 512 for ReID)
            snapshot_limit: Max vectors per track (FIFO limit)
            company_id: Company identifier (for logging)
        """
        self.capacity = capacity
        self.vector_dim = vector_dim
        self.snapshot_limit = snapshot_limit
        self.company_id = company_id
        
        # Core storage - pre-allocated for performance
        # Shape: (capacity, vector_dim) - C-contiguous for fast matrix ops
        self.vectors = np.zeros((capacity, vector_dim), dtype=np.float32)
        
        # Parallel metadata storage
        self.metadata: List[Optional[Dict[str, Any]]] = [None] * capacity
        
        # Index management
        self.free_indices = deque(range(capacity))  # Available slots
        self.used_indices: set[int] = set()         # Occupied slots
        
        # Track ownership - FIFO per track
        # Key format: "stream_id:track_id"
        # Value: deque with maxlen=snapshot_limit (auto-evict oldest)
        self.track_to_indices: Dict[str, deque] = {}
        
        # Reverse mapping for fast lookup
        self.index_to_track: Dict[int, str] = {}  # index -> track_key
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Statistics
        self.stats = {
            "upserts": 0,
            "searches": 0,
            "deletes": 0,
            "transfers": 0,
            "fifo_evictions": 0,
        }
        
        
    
    # ----------------------------------------------------------------
    # Core Operations
    # ----------------------------------------------------------------
    
    def upsert_snapshot(
        self,
        track_id: str,
        stream_id: str,
        body_vector: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        snapshot_suffix: Optional[str] = None,
    ) -> bool:
        """
        Add vector snapshot with automatic FIFO eviction.
        
        Args:
            track_id: Track identifier
            stream_id: Stream identifier
            body_vector: Feature vector (shape: (512,))
            metadata: Optional metadata dict
            snapshot_suffix: Optional suffix for debugging
            
        Returns:
            True if successful
        """
        if body_vector.shape[0] != self.vector_dim:
            logger.error(
                "[RAMVectorStore] Vector dimension mismatch: expected %d, got %d",
                self.vector_dim, body_vector.shape[0]
            )
            return False
        
        key = f"{stream_id}:{track_id}"
        
        with self._lock:
            # Initialize track if new
            if key not in self.track_to_indices:
                self.track_to_indices[key] = deque(maxlen=self.snapshot_limit)
            
            track_indices = self.track_to_indices[key]
            
            # FIFO: If at limit, oldest will be auto-evicted by deque
            if len(track_indices) >= self.snapshot_limit:
                # Note: deque will auto-remove oldest when we append
                # We need to free the index that will be evicted
                oldest_idx = track_indices[0]  # Will be removed on next append
                self._free_index(oldest_idx)
                self.stats["fifo_evictions"] += 1
            
            # Allocate new index
            if not self.free_indices:
                logger.error("[RAMVectorStore] Capacity exceeded! Cannot allocate new index.")
                return False
            
            new_idx = self.free_indices.popleft()
            
            # Store vector as-is (already normalized by feature extractor)
            # DO NOT normalize again - would cause mismatch with Qdrant!
            self.vectors[new_idx] = body_vector.astype(np.float32, copy=False)
            
            # Store metadata
            meta = metadata or {}
            meta.update({
                "track_id": track_id,
                "stream_id": stream_id,
                "timestamp": time.time(),
                "suffix": snapshot_suffix,
            })
            self.metadata[new_idx] = meta
            
            # Update mappings
            track_indices.append(new_idx)  # Auto FIFO via maxlen
            self.used_indices.add(new_idx)
            self.index_to_track[new_idx] = key
            
            self.stats["upserts"] += 1
        
        return True
    
    def search(
        self,
        query_vector: np.ndarray,
        stream_id: str,
        exclude_tracks: Optional[set] = None,
        top_k: int = 20,
        score_threshold: float = 0.70,
    ) -> List[Dict[str, Any]]:
        """Simple single-query search."""
        queries = [{
            "track_id": "__query__",
            "vector": query_vector,
            "filter": {"exclude_tracks": exclude_tracks or set()}
        }]
        
        results = self.batch_search_flat(
            queries=queries,
            top_k=top_k,
            score_threshold=score_threshold,
        )
        return results.get("__query__", [])
    
    def batch_search_flat(
        self,
        queries: List[Dict[str, Any]],
        top_k: int = 20,
        score_threshold: float = 0.70,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Batch exact cosine similarity search.
        
        Args:
            queries: List of dicts with keys:
                - track_id: str - Query track identifier
                - vector: np.ndarray - Query vector (shape: (512,))
                - filter: Optional - Filter dict (e.g., stream_id)
            top_k: Number of top results per query
            score_threshold: Minimum similarity score
            
        Returns:
            Dict[track_id -> List[result_dicts]]
            Each result: {"id": index, "score": float, "payload": dict}
        """
        if not queries:
            return {}
        
        results = {}
        
        with self._lock:
            # Build active indices list once
            active_indices = sorted(self.used_indices)
            if not active_indices:
                return {q["track_id"]: [] for q in queries}
            
            # Extract active vectors - shape: (N, 512)
            active_vectors = self.vectors[active_indices]
            
            # Pre-build track_id array for fast filtering
            track_ids_array = np.array([
                self.metadata[idx].get("track_id", "") if self.metadata[idx] else ""
                for idx in active_indices
            ])
            
            # Process each query
            for query in queries:
                query_track_id = query.get("track_id")
                query_vec = query.get("vector")
                query_filter = query.get("filter")
                
                if query_track_id is None or query_vec is None:
                    continue
                
                # Skip normalize if vector is already unit norm (within tolerance)
                # Feature extractor already normalizes, so this is usually a no-op
                vec_norm = np.linalg.norm(query_vec)
                if vec_norm < 1e-8:
                    results[query_track_id] = []
                    continue
                if abs(vec_norm - 1.0) > 0.01:  # Only normalize if not already normalized
                    query_normalized = query_vec / vec_norm
                else:
                    query_normalized = query_vec
                
                # Cosine similarity: query @ matrix.T
                similarities = active_vectors @ query_normalized
                
                # Apply exclude filter (single stream, no stream_id filter needed)
                if query_filter and isinstance(query_filter, dict):
                    exclude_tracks = query_filter.get("exclude_tracks")
                    if exclude_tracks:
                        exclude_set = set(exclude_tracks)  # O(1) lookup
                        exclude_mask = np.isin(track_ids_array, list(exclude_set), invert=True)
                        similarities = np.where(exclude_mask, similarities, -1.0)
                
                # Filter by threshold and get top-k in one pass
                valid_mask = similarities >= score_threshold
                valid_indices_local = np.where(valid_mask)[0]
                
                if len(valid_indices_local) == 0:
                    results[query_track_id] = []
                    continue
                
                # Get top-k efficiently
                valid_scores = similarities[valid_indices_local]
                if len(valid_scores) > top_k:
                    # Use argpartition for O(N) instead of full sort O(N log N)
                    top_k_local_indices = np.argpartition(valid_scores, -top_k)[-top_k:]
                    top_k_local_indices = top_k_local_indices[np.argsort(valid_scores[top_k_local_indices])[::-1]]
                else:
                    top_k_local_indices = np.argsort(valid_scores)[::-1]
                
                # Build result list
                track_results = []
                for local_idx in top_k_local_indices:
                    global_idx = active_indices[valid_indices_local[local_idx]]
                    track_results.append({
                        "id": global_idx,
                        "score": float(valid_scores[local_idx]),
                        "payload": self.metadata[global_idx].copy(),
                    })
                
                results[query_track_id] = track_results
            
            self.stats["searches"] += 1
        
        return results
    
    def get_vectors_for_tracks(
        self,
        track_ids: List[str],
        stream_id: str,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve all vectors for specified tracks.
        
        Args:
            track_ids: List of track IDs
            stream_id: Stream identifier
            
        Returns:
            Dict[track_id -> {"vectors": List[np.ndarray], "payload": dict, "global_id": str}]
        """
        if not track_ids:
            return {}
        
        results = {}
        
        with self._lock:
            for track_id in track_ids:
                key = f"{stream_id}:{track_id}"
                indices = self.track_to_indices.get(key, deque())
                
                if not indices:
                    continue
                
                # Collect vectors and metadata
                vectors = [self.vectors[idx].copy() for idx in indices]
                # Use most recent metadata
                latest_meta = self.metadata[indices[-1]].copy() if indices else {}
                global_id = latest_meta.get("global_id")
                
                results[track_id] = {
                    "vectors": vectors,
                    "payload": latest_meta,
                    "global_id": global_id,
                    "uuids": list(indices),  # For debugging
                }
        
        return results
    
    def retire_track(self, stream_id: str, track_id: str) -> bool:
        """
        Delete all vectors of a track.
        
        Args:
            stream_id: Stream identifier
            track_id: Track identifier
            
        Returns:
            True if successful
        """
        key = f"{stream_id}:{track_id}"
        
        with self._lock:
            indices = self.track_to_indices.get(key, deque())
            
            if not indices:
                
                return True
            
            # Free all indices
            for idx in indices:
                self._free_index(idx)
            
            # Remove track mapping
            del self.track_to_indices[key]
            
            self.stats["deletes"] += 1
            
            
        
        return True
    
    def get_vectors(self, track_id: str, stream_id: str) -> List[np.ndarray]:
        """
        Get vectors for a single track (RAM-optimized simple API).
        
        Args:
            track_id: Track identifier
            stream_id: Stream identifier
            
        Returns:
            List of vectors (empty if track not found)
        """
        key = f"{stream_id}:{track_id}"
        
        with self._lock:
            if key not in self.track_to_indices:
                return []
            
            indices = list(self.track_to_indices[key])
            return [self.vectors[idx].copy() for idx in indices]
    
    def transfer_vectors_to_track(
        self,
        from_track_id: str,
        to_track_id: str,
        stream_id: str,
        max_vectors: int = 6,
        to_global_id: Optional[str] = None,
    ) -> Tuple[int, List[int]]:
        """
        Transfer vectors from one track to another by copying.
        
        Args:
            from_track_id: Source track
            to_track_id: Target track
            stream_id: Stream identifier
            max_vectors: Maximum vectors to transfer
            to_global_id: Global ID to assign to transferred vectors
            
        Returns:
            Tuple of (number_transferred, list_of_new_indices)
        """
        if from_track_id == to_track_id:
            return 0, []
        
        from_key = f"{stream_id}:{from_track_id}"
        to_key = f"{stream_id}:{to_track_id}"
        
        with self._lock:
            source_indices = list(self.track_to_indices.get(from_key, deque()))
            
            if not source_indices:
                return 0, []
            
            # Sample indices (interleaved for diversity)
            if len(source_indices) > max_vectors:
                step = len(source_indices) / max_vectors
                indices_to_copy = [source_indices[int(i * step)] for i in range(max_vectors)]
            else:
                indices_to_copy = source_indices
            
            # Initialize target track if needed
            if to_key not in self.track_to_indices:
                self.track_to_indices[to_key] = deque(maxlen=self.snapshot_limit)
            
            target_indices = self.track_to_indices[to_key]
            new_indices = []
            
            # Copy vectors to new indices
            for src_idx in indices_to_copy:
                # Check capacity
                if not self.free_indices:
                    logger.warning("[RAMVectorStore] Capacity exceeded during transfer")
                    break
                
                # Handle FIFO eviction if target is full
                if len(target_indices) >= self.snapshot_limit:
                    oldest_idx = target_indices[0]
                    self._free_index(oldest_idx)
                    self.stats["fifo_evictions"] += 1
                
                # Allocate new index
                new_idx = self.free_indices.popleft()
                
                # Copy vector
                self.vectors[new_idx] = self.vectors[src_idx].copy()
                
                # Copy and update metadata
                new_meta = self.metadata[src_idx].copy()
                new_meta["track_id"] = to_track_id
                new_meta["transferred_from"] = from_track_id
                new_meta["transfer_timestamp"] = time.time()
                if to_global_id:
                    new_meta["global_id"] = to_global_id
                    new_meta["state"] = "ASSIGNED"
                
                self.metadata[new_idx] = new_meta
                
                # Update mappings
                target_indices.append(new_idx)
                self.used_indices.add(new_idx)
                self.index_to_track[new_idx] = to_key
                new_indices.append(new_idx)
            
            self.stats["transfers"] += 1
            
            
        
        return len(new_indices), new_indices
    
    # ----------------------------------------------------------------
    # Compatibility Methods (Qdrant Interface)
    # ----------------------------------------------------------------
    
    def update_global_id(
        self,
        track_id: str,
        stream_id: str,
        global_id: Optional[str],
        state: str = "ASSIGNED"
    ) -> bool:
        """
        Update global_id in metadata for all vectors of a track.
        
        Args:
            track_id: Track identifier
            stream_id: Stream identifier
            global_id: Global ID to assign (or None to clear)
            state: State to set (ASSIGNED, PENDING, etc.)
            
        Returns:
            True if successful
        """
        key = f"{stream_id}:{track_id}"
        
        with self._lock:
            indices = self.track_to_indices.get(key, deque())
            if not indices:
                
                return False
            
            # Update metadata for all vectors of this track
            for idx in indices:
                if self.metadata[idx]:
                    self.metadata[idx]["global_id"] = global_id
                    self.metadata[idx]["state"] = state
            
            
        
        return True
    
    def batch_upsert_snapshots(
        self,
        track_id: str,
        stream_id: str,
        vectors_with_suffixes: List[Tuple[np.ndarray, str]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Batch upsert multiple snapshots for a track (Qdrant-compatible API).
        
        Args:
            track_id: Track ID
            stream_id: Stream ID
            vectors_with_suffixes: List of (vector, suffix) tuples
            metadata: Optional metadata to attach to all points
            
        Returns:
            Number of successfully upserted snapshots
        """
        if not vectors_with_suffixes:
            return 0
        
        count = 0
        for vector, suffix in vectors_with_suffixes:
            # Merge metadata
            full_metadata = metadata.copy() if metadata else {}
            full_metadata["suffix"] = suffix
            
            success = self.upsert_snapshot(
                track_id=track_id,
                stream_id=stream_id,
                body_vector=vector,
                metadata=full_metadata,
                snapshot_suffix=suffix,
            )
            if success:
                count += 1
        

        return count
    
    def batch_retrieve(
        self,
        uuids: List[int],
        with_vectors: bool = True
    ) -> Dict[int, Dict[str, Any]]:
        """
        Retrieve multiple vectors by their indices (UUIDs).
        
        Args:
            uuids: List of indices to retrieve
            with_vectors: Whether to include vectors in results
            
        Returns:
            Dict[uuid -> {"vector": np.ndarray, "payload": dict}]
        """
        results = {}
        
        with self._lock:
            for uuid in uuids:
                if uuid not in self.used_indices:
                    continue
                
                result = {
                    "payload": self.metadata[uuid].copy() if self.metadata[uuid] else {}
                }
                
                if with_vectors:
                    result["vector"] = self.vectors[uuid].copy()
                
                results[uuid] = result
        
        return results
    
    # ----------------------------------------------------------------
    # Helper Methods
    # ----------------------------------------------------------------
    
    def _free_index(self, idx: int) -> None:
        """
        Free an index for reuse (internal, must hold lock).
        
        Args:
            idx: Index to free
        """
        self.used_indices.discard(idx)
        if idx in self.index_to_track:
            del self.index_to_track[idx]
        self.free_indices.append(idx)
        # No need to zero out vector - will be overwritten
    
    def cleanup_stale_tracks(self, max_age_seconds: float = 3600) -> List[str]:
        """
        Cleanup tracks that haven't been updated in max_age_seconds.
        
        Args:
            max_age_seconds: Max age in seconds (default 3600 = 1 hour)
            
        Returns:
            Number of tracks cleaned up
        """
        now = time.time()
        stale_keys = []
        
        with self._lock:
            for key, indices in self.track_to_indices.items():
                if not indices:
                    stale_keys.append(key)
                    continue
                    
                # Check timestamp of newest vector
                newest_idx = indices[-1]
                meta = self.metadata[newest_idx]
                if meta is None:
                    stale_keys.append(key)
                    continue
                    
                last_update = meta.get("timestamp", 0)
                
                # Determine threshold based on assignment status
                # If track has global_id (ASSIGNED), keep for long time (default max_age_seconds = 3600)
                # If track has NO global_id (Pending/Ghost), clean quickly (300s = 5 mins)
                has_global_id = bool(meta.get("global_id"))
                threshold = max_age_seconds if has_global_id else 300.0
                
                if now - last_update > threshold:
                    stale_keys.append(key)
        
        # Cleanup outside main lock iteration
        cleaned_keys = []
        for key in stale_keys:
            parts = key.split(":", 1)
            if len(parts) == 2:
                stream_id, track_id = parts
                if self.retire_track(stream_id, track_id):
                    cleaned_keys.append(key)
        
        if cleaned_keys:
            # Assuming logger is imported, e.g., from logging import getLogger; logger = getLogger(__name__)
            # If not, this line will cause an error. The instruction implies logger exists.
            import logging
            logger = logging.getLogger(__name__)
            logger.info(
                "[RAMVectorStore] Cleaned up %d stale tracks (age > %ds)",
                len(cleaned_keys), int(max_age_seconds)
            )
        
        return cleaned_keys
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        with self._lock:
            return {
                "capacity": self.capacity,
                "used": len(self.used_indices),
                "free": len(self.free_indices),
                "usage_percent": len(self.used_indices) / self.capacity * 100,
                "num_tracks": len(self.track_to_indices),
                **self.stats,
            }
    
    def reset_collection(self) -> bool:
        """Reset all storage (for testing)."""
        with self._lock:
            self.vectors.fill(0)
            self.metadata = [None] * self.capacity
            self.free_indices = deque(range(self.capacity))
            self.used_indices.clear()
            self.track_to_indices.clear()
            self.index_to_track.clear()
            
            for key in self.stats:
                self.stats[key] = 0
            
            
        
        return True
    
    def batch_upsert_multi_tracks(
        self,
        entries: List[Dict[str, Any]],
        stream_id: str,
    ) -> int:
        """
        Batch upsert vectors from MULTIPLE tracks (frame-based extraction).
        
        Args:
            entries: List of dicts with keys: track_id, vector, global_id, metadata (optional)
            stream_id: Stream ID (shared by all tracks)
            
        Returns:
            Number of successfully upserted vectors
        """
        if not entries:
            return 0
        
        success_count = 0
        
        for entry in entries:
            track_id = str(entry.get("track_id", ""))
            vector = entry.get("vector")
            global_id = entry.get("global_id")
            metadata = entry.get("metadata") or {}
            
            if not track_id or vector is None:
                continue
            
            # Upsert using existing method
            vector_array = vector if isinstance(vector, np.ndarray) else np.array(vector, dtype=np.float32)
            
            # Merge metadata
            full_metadata = {
                "global_id": global_id,
                **metadata
            }
            
            success = self.upsert_snapshot(
                track_id=track_id,
                stream_id=stream_id,
                body_vector=vector_array,
                metadata=full_metadata,
            )
            
            if success:
                success_count += 1
        

        return success_count
    
    def transfer_vectors_from_cache(
        self,
        from_track_id: str,
        to_track_id: str,
        stream_id: str,
        cached_vectors: List[np.ndarray],
        max_vectors: int = 6,
        to_global_id: Optional[str] = None,
        min_sim_filter: float = 0.0,
        query_vector: Optional[np.ndarray] = None,
    ) -> Tuple[int, List[int]]:
        """
        Transfer vectors from cache (compatibility wrapper).
        
        RAM has no cache concept - vectors already in memory.
        Ignores cached_vectors and calls transfer_vectors_to_track directly.
        
        Args:
            from_track_id: Source track ID
            to_track_id: Destination track ID
            stream_id: Stream ID
            cached_vectors: Ignored (RAM has no cache)
            max_vectors: Ignored (RAM transfers all)
            to_global_id: Optional global ID to set
            min_sim_filter: Ignored (Qdrant-specific)
            query_vector: Ignored (Qdrant-specific)
            
        Returns:
            Tuple of (number_transferred, empty_list)
        """
        count, indices = self.transfer_vectors_to_track(
            from_track_id=from_track_id,
            to_track_id=to_track_id,
            stream_id=stream_id,
            max_vectors=max_vectors,  # Pass through the limit!
            to_global_id=to_global_id,
        )
        # Return tuple to match Qdrant signature
        return (count, indices)

