# reid_system/demo_config.py

"""
Demo configuration matching reid_system config structure.

Provides cfg object with SUPERVISION_SYSTEM.REID settings.
"""

from typing import Any
import logging

logger = logging.getLogger(__name__)

class MockConfig:
    """Mock config to match cfg.SUPERVISION_SYSTEM structure."""
    
    def __init__(self):
        # SUPERVISION_SYSTEM.REID config
        self.REID = MockREIDConfig()
        
        # SUPERVISION_SYSTEM.VOTING config  
        self.VOTING = MockVotingConfig()
        
        # SUPERVISION_SYSTEM.TRACK_MANAGER config
        self.TRACK_MANAGER = MockTrackManagerConfig()
        
        # SUPERVISION_SYSTEM.STAGE1 config (for Kalman filter)
        self.STAGE1 = MockStage1Config()
        
        # SUPERVISION_SYSTEM.STAGE2 config (for advanced matching)
        self.STAGE2 = MockStage2Config()
        
        # SUPERVISION_SYSTEM.SEARCH config (for HNSW search)
        self.SEARCH = MockSearchConfig()

class MockREIDConfig:
    """Mock REID config."""
    
    def __init__(self):
        self.MODEL = MockModelConfig()
        self.EXTRACTION_INTERVAL_FRAMES = 5
        self.RAM_CAPACITY = 50000
        self.SNAPSHOT_HISTORY_LIMIT = 20

class MockModelConfig:
    """Mock model config."""
    
    def __init__(self):
        self.NAME = "osnet_msmt17_engine"
        self.VERSION = "1"

class MockVotingConfig:
    """Mock voting config."""
    
    def __init__(self):
        self.WINDOW_SIZE = 10
        self.TOPK_PER_VOTE = 3
        self.SCORE_FILTER = 0.65
        self.EARLY_EXIT_ENABLED = True
        self.EARLY_EXIT_MIN_VOTES = 5
        self.EARLY_EXIT_RATIO = 0.6
        self.EARLY_EXIT_MARGIN = 0.05
        self.MIN_RATIO_TO_ASSIGN = 0.30
        self.TIMEOUT_ENABLED = True
        self.TIMEOUT_SECONDS = 5
        self.MIN_VOTES_TO_COMPLETE = 3
        self.TIMEOUT_MULTIPLIER = 2.0

class MockTrackManagerConfig:
    """Mock track manager config."""
    def __init__(self):
        self.EXPIRE_TIME = 30
        self.TRACK_THRESHOLD = 0.6
        self.ID_THRESHOLD = 0.5
        self.MAX_TRACKS = 1000
        self.RETIRED_TRACK_LIMIT = 100
        self.CLEANUP_AGE_SECONDS = 3600  # Cleanup age for retired tracks

class MockStage1Config:
    """Mock Stage1 config for Kalman filter settings."""
    def __init__(self):
        self.KALMAN_MAX_MISSING_FRAMES = 30  # Max frames a track can be missing before cleanup
        self.KALMAN_MAX_AGE = 60  # Max age for Kalman predictions
        self.IOU_THRESHOLD = 0.3  # IOU threshold for matching tracks
        self.POOL_SIZE = 100  # Pool size for track matching
        self.SIMILARITY_THRESHOLD = 0.5  # Similarity threshold for ReID matching

class MockStage2Config:
    """Mock Stage2 config for advanced matching settings."""
    def __init__(self):
        self.SIMILARITY_THRESHOLD = 0.6  # Stage 2 similarity threshold for advanced matching

class MockSearchConfig:
    """Mock Search config for HNSW search settings."""
    def __init__(self):
        self.HNSW_SCORE_THRESHOLD = 0.5  # HNSW search score threshold
        self.FINAL_SCORE_MAX_WEIGHT = 0.7  # Final score max weight for search

class MockMainConfig:
    """Mock main config object."""
    
    def __init__(self):
        self.SUPERVISION_SYSTEM = MockConfig()

# Global config instance
cfg = MockMainConfig()

logger.info("Demo config initialized with mock reid_v2 config structure")