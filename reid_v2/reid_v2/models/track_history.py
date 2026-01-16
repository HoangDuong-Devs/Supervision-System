# models/track_history.py

from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..storage.metadata_cache import TrackMetadataCache


class TrackFeatureHistory:
    """
    Feature history for tracks (used by ReIDFeatureExtractor).
    
    Stores extracted features and manages update cycle per track.
    DELEGATES global_id/state to TrackMetadata via metadata_cache.
    """

    def __init__(
        self,
        track_id: str,
        stream_id: str,
        metadata_cache: Optional["TrackMetadataCache"] = None,
    ):
        """
        Initialize track feature history.
        
        Args:
            track_id: Unique track identifier
            stream_id: Stream identifier
            metadata_cache: Reference to TrackMetadataCache for global_id/state delegation
        """
        self.track_id = track_id
        self.stream_id = stream_id
        self._metadata_cache = metadata_cache

        # Frame-based tracking (primary)
        self.frame_process_count = 0      # Number of frames this track was processed
        self.latest_feature: Optional[np.ndarray] = None
        self.last_extract_frame = -1      # Frame of last extraction
        self.extract_count = 0            # Number of extractions performed
        self.first_seen_ts = datetime.utcnow().timestamp()
        
        # Fallback storage when metadata_cache is not available
        self._fallback_global_id: Optional[str] = None
        self._fallback_state: str = "PENDING"

        self.last_seen_frame: Optional[int] = None

    # ----------------------------------------------------------------
    # Delegated properties (Authoritative source: TrackMetadata)
    # ----------------------------------------------------------------
    
    @property
    def global_id(self) -> Optional[str]:
        """Get global_id from TrackMetadata (single source of truth)."""
        if self._metadata_cache is not None:
            track = self._metadata_cache.get_track(self.track_id)
            if track is not None:
                return track.global_id
        return self._fallback_global_id
    
    @global_id.setter
    def global_id(self, value: Optional[str]) -> None:
        """Set global_id on TrackMetadata (authoritative) and fallback."""
        self._fallback_global_id = value
        if self._metadata_cache is not None:
            track = self._metadata_cache.get_track(self.track_id)
            if track is not None:
                track.global_id = value
    
    @property
    def state(self) -> str:
        """Get state from TrackMetadata (single source of truth)."""
        if self._metadata_cache is not None:
            track = self._metadata_cache.get_track(self.track_id)
            if track is not None:
                return track.state or "PENDING"
        return self._fallback_state
    
    @state.setter
    def state(self, value: str) -> None:
        """Set state on TrackMetadata (authoritative) and fallback."""
        self._fallback_state = value
        if self._metadata_cache is not None:
            track = self._metadata_cache.get_track(self.track_id)
            if track is not None:
                track.state = value

    # ----------------------------------------------------------------
    # Frame registration and extraction
    # ----------------------------------------------------------------

    def register_frame(self, frame_number: int) -> None:
        """Update last seen frame and increment count."""
        self.frame_process_count += 1
        self.last_seen_frame = frame_number

    # NOTE: should_extract() removed - extraction now uses global frame cycle in feature_extractor.py
    # The logic is: extract ALL tracks when frame_number % EXTRACTION_INTERVAL_FRAMES == 0

    def add_observation(
        self,
        feature: np.ndarray,
        frame_number: int,
    ) -> None:
        """
        Add new observation with feature.
        
        Updates observation count, latest feature, and continuous run tracking.
        
        Args:
            feature: Extracted ReID feature vector
            frame_number: Current frame number
        """
        self.latest_feature = feature.astype(np.float32, copy=False)
        self.last_seen_frame = frame_number
        # Note: last_extract_frame already updated in should_extract()

    def get_latest_feature(self) -> Optional[np.ndarray]:
        return self.latest_feature