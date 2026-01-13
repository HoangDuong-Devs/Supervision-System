# reid_system/config.py

"""
Configuration module for ReID System.

Full configuration matching reid_system module behavior.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class ReIDConfig:
    """
    Full configuration for ReID Demo - matches reid_v2 behavior.
    """
    # =================================================================
    # Model settings
    # =================================================================
    onnx_model_path: str = "osnet_x1_0_msmt17.onnx"
    input_size: Tuple[int, int] = (128, 256)  # (width, height) for OSNet
    input_width: int = 128
    input_height: int = 256
    vector_dim: int = 512
    reid_vector_dim: int = 512  # Alias for compatibility
    
    # =================================================================
    # Feature Extraction settings
    # =================================================================
    extraction_interval: int = 5  # Extract ReID every N frames
    min_track_length: int = 3     # Minimum track length before processing
    
    # =================================================================
    # Voting settings (matching reid_v2)
    # =================================================================
    vote_window_size: int = 10          # Number of votes before final decision
    voting_window_size: int = 10        # Alias for compatibility
    vote_ratio_threshold: float = 0.30  # Min ratio to consider consensus
    vote_topk: int = 3                  # Top-K candidates per vote
    vote_score_filter: float = 0.65     # Filter low-quality votes
    similarity_threshold: float = 0.70  # Main similarity threshold
    
    # Early exit
    early_exit_enabled: bool = True
    early_exit_min_votes: int = 3       # Min votes before early exit
    early_exit_ratio: float = 0.6       # Dominant ratio for early exit
    early_exit_margin: float = 0.05     # Margin over second-best
    
    # Final assignment
    min_vote_ratio_to_assign: float = 0.3
    min_votes_to_complete: int = 5      # Min votes for timeout completion
    
    # Timeout (frames)
    voting_timeout_enabled: bool = True
    voting_timeout_multiplier: float = 2.0  # timeout = window * interval * multiplier
    
    # =================================================================
    # Similarity / Search settings
    # =================================================================
    search_top_k: int = 20              # Top-K in vector search
    search_score_threshold: float = 0.70  # Min cosine similarity
    new_id_threshold: float = 0.50      # Below this, create new ID
    
    # Stage 1: Motion-based matching
    stage1_pool_size: int = 10          # Candidates from recent lost tracks
    stage1_iou_threshold: float = 0.3   # IoU for bbox matching
    stage1_kalman_max_missing: int = 60 # Max frames before track considered gone
    
    # =================================================================
    # Storage settings
    # =================================================================
    snapshot_limit: int = 20            # Max vectors per track (FIFO)
    ram_capacity: int = 50000           # Max total vectors in RAM
    
    # =================================================================
    # EMA settings
    # =================================================================
    ema_alpha: float = 0.15             # EMA smoothing for feature vectors
    
    # =================================================================
    # Track cleanup
    # =================================================================
    pending_cleanup_threshold: int = 30  # Frames before cleanup pending
    retired_track_limit: int = 1000      # Max retired tracks to remember
    
    # =================================================================
    # ImageNet normalization
    # =================================================================
    imagenet_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    imagenet_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
