# reid_system/__init__.py

"""
REID_SYSTEM - Long-term Person Re-Identification (ONNX Backend)
Flow-based ReID system with ONNX inference.
"""

# Core business logic
from .main_logic.pipeline import ReIDPipeline
from .main_logic.track_manager import TrackManager
from .main_logic.track_layers import PendingTrackStore, AssignedTrackStore

# Models
from .models.track_metadata import TrackMetadata

# Inference (ONNX instead of Triton)
from .inference.feature_extractor import ReIDFeatureExtractor

# Storage
from .storage.ram_vector_store import RAMVectorStore
from .storage.metadata_cache import TrackMetadataCache

# Utils
from .utils.geometry import normalize_to_ltwh, are_bboxes_close, compute_iou
from .utils.vector_utils import cosine_similarity_normalized

__all__ = [
    # Core
    "ReIDPipeline",
    "TrackManager",
    "PendingTrackStore",
    "AssignedTrackStore",

    # Models
    "TrackMetadata",

    # Inference
    "ReIDFeatureExtractor",

    # Storage
    "RAMVectorStore",
    "TrackMetadataCache",

    # Utils
    "normalize_to_ltwh",
    "are_bboxes_close",
    "compute_iou",
    "cosine_similarity_normalized",
]
