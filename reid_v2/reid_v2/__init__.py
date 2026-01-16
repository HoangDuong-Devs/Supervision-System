# =============================================================================
# REID_V2 - Long-term Person Re-Identification (RAM Backend)
# =============================================================================
# Flow-based ReID system with RAM vector store, 3-stage voting, and buffer emit.

# Main entry point
from .reid_processor import process_reid_v2, flush_reid_v2, stop_reid_v2, stop_all_v2

# Core business logic
from .main_logic.pipeline import ReIDPipeline
from .main_logic.track_manager import TrackManager
from .main_logic.track_layers import PendingTrackStore, AssignedTrackStore

# Models
from .models.track_metadata import TrackMetadata
from .models.track_history import TrackFeatureHistory

# Inference
from .inference.feature_extractor import ReIDFeatureExtractor
from .inference.kalman_xywh import KalmanFilterXYWH

# Storage
from .storage.ram_vector_store import RAMVectorStore
from .storage.metadata_cache import TrackMetadataCache

# Emit (buffer) - unified from reid module
from reid.emit.global_id_buffer import GlobalIDBuffer


__all__ = [
    # Main entry
    "process_reid_v2",
    "flush_reid_v2",
    "stop_reid_v2",
    "stop_all_v2",
    
    # Core
    "ReIDPipeline",
    "TrackManager",
    "PendingTrackStore",
    "AssignedTrackStore",
    
    # Models
    "TrackMetadata",
    "TrackFeatureHistory",
    
    # Inference
    "ReIDFeatureExtractor",
    "KalmanFilterXYWH",
    
    # Storage
    "RAMVectorStore",
    "TrackMetadataCache",
    
    # Emit
    "GlobalIDBuffer",
]
