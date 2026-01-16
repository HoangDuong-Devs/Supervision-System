# reid_system/reid_processor.py

"""
ReID System Processor - Entry point for ReID processing.

Provides process_reid_system function for demo and production use.
"""

import logging
from typing import Any, Dict, Optional

from .main_logic.pipeline import ReIDPipeline

logger = logging.getLogger(__name__)

# Global pipeline instance (for demo simplicity)
_pipeline_instance: Optional[ReIDPipeline] = None


def process_reid_system(
    stream_id: str,
    metadata: Dict[str, Any],
    onnx_model_path: str = "osnet_x1_0_msmt17.onnx",
    company_id: str = "demo",
) -> Optional[Dict[str, Any]]:
    """
    Process ReID system for a frame.

    Args:
        stream_id: Stream identifier
        metadata: Frame metadata containing 'frame', 'frame_idx', 'objects'
        onnx_model_path: Path to ONNX model
        company_id: Company identifier

    Returns:
        Processed metadata with global IDs assigned to objects
    """
    global _pipeline_instance

    try:
        # Initialize pipeline if not exists
        if _pipeline_instance is None:
            logger.info("Initializing ReID Pipeline for stream: %s", stream_id)
            _pipeline_instance = ReIDPipeline(
                stream_id=stream_id,
                company_id=company_id,
                onnx_model_path=onnx_model_path,
            )

            if not _pipeline_instance.initialize():
                logger.error("Failed to initialize ReID Pipeline")
                return None

        # Extract data from metadata
        frame = metadata.get('frame')
        frame_idx = metadata.get('frame_idx', 0)
        objects = metadata.get('objects', [])

        if not objects:
            # Return metadata unchanged if no objects
            return metadata

        # Process frame through ReID pipeline
        global_id_mappings = _pipeline_instance.process_reid_frame(
            stream_id=stream_id,
            frame=frame,
            objects=objects,
            frame_idx=frame_idx,
        )

        # Update objects with global IDs
        processed_objects = []
        for obj in objects:
            track_id = obj.get('track_id')
            if track_id and track_id in global_id_mappings:
                global_id = global_id_mappings[track_id]
                if global_id:
                    obj = obj.copy()  # Don't modify original
                    obj['global_id'] = global_id

            processed_objects.append(obj)

        # Return processed metadata
        processed_metadata = metadata.copy()
        processed_metadata['objects'] = processed_objects

        return processed_metadata

    except Exception as e:
        logger.error("Error in process_reid_system: %s", e, exc_info=True)
        return None


def reset_reid_pipeline():
    """Reset global pipeline instance (for testing/cleanup)."""
    global _pipeline_instance
    if _pipeline_instance:
        _pipeline_instance.stop()
        _pipeline_instance = None


def stop_reid_system(stream_id: str = "demo", company_id: str = "demo"):
    """Stop the ReID system and cleanup resources."""
    reset_reid_pipeline()
    logger.info("ReID system stopped and resources cleaned up")
