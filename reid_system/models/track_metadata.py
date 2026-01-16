# reid_v2_demo/models/track_metadata.py

"""
Track metadata data model.

Copied exactly from reid_v2.models.track_metadata.py
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class TrackMetadata:
    """
    In-RAM state for 1 tracked person.
    
    Acts as the Single Source of Truth for:
    - Identity (local_id, global_id, state)
    - Spatial data (bbox, history)
    - ReID vectors (last vector, EMA, history)
    - Kalman filter state
    """
    
    local_id: str

    # Identity
    global_id: Optional[str] = None
    state: str = "PENDING"  # PENDING / ASSIGNED

    # Timestamps
    birth_time: float = field(default_factory=time.time)
    last_update_time: float = field(default_factory=time.time)

    # Counters
    person_extraction_count: int = 0

    # Vector Storage (RAM only)
    last_person_vector: Optional[np.ndarray] = None
    last_person_vector_frame: Optional[int] = None
    last_upsert_frame: Optional[int] = None  # Frame when vector was last upserted (for throttling)
    last_crop: Optional[np.ndarray] = None  # For visual debugging crop storage
    
    # Vector history (before assignment)
    person_vector_history: List[Tuple[np.ndarray, int]] = field(default_factory=list)
    
    # Crop history (before assignment) - stored in RAM, written to disk on assignment
    crop_history: List[Tuple[np.ndarray, int]] = field(default_factory=list)  # List of (crop_image, frame_num)

    # EMA Vector (RAM)
    person_vector_ema: Optional[np.ndarray] = None
    person_vector_ema_count: int = 0

    # Confidence
    person_confidence: Optional[float] = None

    # Spatial State
    latest_bbox: Optional[Dict] = None  # {"left","top","width","height"}
    bboxes: list = field(default_factory=list)

    # Lifecycle
    missing_count: int = 0

    # Diagnostic
    assignment_confidence: float = 0.0

    # Kalman State
    kalman_mean: Optional[np.ndarray] = None
    kalman_covariance: Optional[np.ndarray] = None
    kalman_predicted_mean: Optional[np.ndarray] = None
    kalman_predicted_covariance: Optional[np.ndarray] = None
    kalman_predicted_bbox: Optional[Dict[str, float]] = None
    kalman_last_observation_time: float = 0.0

    # Candidate Cache (reused per observation)
    candidate_cache: List[Dict[str, Any]] = field(default_factory=list)
    candidate_cache_token: Optional[int] = None