# reid_system/reid_processor.py

"""
ReID System Processor - Entry point cho Long-term ReID với ONNX backend.

Sử dụng:
    from reid_system import process_reid_system
    
    # Trong tracking loop:
    output_metadata = process_reid_system(stream_id, input_metadata, onnx_model_path="model.onnx")
"""

import logging
from typing import Any, Dict, List, Optional

from .main_logic.pipeline import ReIDPipeline

logger = logging.getLogger(__name__)

# Cache per stream
_pipelines: Dict[str, ReIDPipeline] = {}


def process_reid_system(
    stream_id: str, 
    metadata: Dict[str, Any],
    onnx_model_path: str = "osnet_x1_0_msmt17.onnx",
    company_id: str = "default",
) -> Optional[Dict[str, Any]]:
    """
    Process ReID và trả về metadata đã patch Global ID.
    
    Args:
        stream_id: ID của stream
        metadata: Metadata chứa frame data và objects  
        onnx_model_path: Path to ONNX model file
        company_id: Company identifier
        
    Returns:
        Metadata đã patch global_id, hoặc None nếu lỗi.
    """
    try:
        pipeline = _get_pipeline(stream_id, company_id, onnx_model_path)
        
        if not pipeline:
            logger.warning("[ReID_System] No pipeline for stream=%s, fallback", stream_id)
            return metadata

        # Extract frame and objects from metadata
        frame = metadata.get('frame')
        objects = metadata.get('objects', [])
        frame_idx = metadata.get('frame_idx', 0)
        
        logger.debug("[ReID_System] Processing stream=%s, frame_idx=%d, objects=%d", 
                    stream_id, frame_idx, len(objects))
        
        if frame is None or not objects:
            logger.warning("[ReID_System] Missing frame or objects for stream=%s", stream_id)
            return metadata

        # Process through pipeline
        global_id_mappings = pipeline.process_reid_frame(
            stream_id=stream_id,
            frame=frame,
            objects=objects,
            frame_idx=frame_idx,
        )
        
        logger.debug("[ReID_System] Pipeline returned %d mappings: %s", 
                    len(global_id_mappings) if global_id_mappings else 0, 
                    global_id_mappings)
        
        # Patch global IDs back into metadata
        patched_metadata = _patch_global_ids(metadata, global_id_mappings)
        
        return patched_metadata
        
    except Exception as exc:
        logger.error("[ReID_System] Error processing stream=%s: %s", stream_id, exc, exc_info=True)
        return metadata


def _get_pipeline(
    stream_id: str, 
    company_id: str,
    onnx_model_path: str,
) -> Optional[ReIDPipeline]:
    """Get or create pipeline for stream."""
    cache_key = f"{company_id}:{stream_id}"
    
    if cache_key not in _pipelines:
        try:
            pipeline = ReIDPipeline(
                stream_id=stream_id,
                company_id=company_id,
                onnx_model_path=onnx_model_path,
            )
            
            if pipeline.initialize():
                _pipelines[cache_key] = pipeline
                logger.info("[ReID_System] Created pipeline for %s", cache_key)
            else:
                logger.error("[ReID_System] Failed to initialize pipeline for %s", cache_key)
                return None
                
        except Exception as exc:
            logger.error("[ReID_System] Error creating pipeline for %s: %s", cache_key, exc)
            return None
    
    return _pipelines[cache_key]


def _patch_global_ids(
    metadata: Dict[str, Any],
    global_id_mappings: Dict[str, Optional[str]],
) -> Dict[str, Any]:
    """Patch global IDs into metadata objects."""
    patched_metadata = metadata.copy()
    objects = metadata.get('objects', [])
    
    if not objects or not global_id_mappings:
        return patched_metadata
    
    patched_objects = []
    for obj in objects:
        patched_obj = obj.copy()
        track_id = str(obj.get('track_id', ''))
        
        if track_id in global_id_mappings:
            global_id = global_id_mappings[track_id]
            patched_obj['global_id'] = global_id
            patched_obj['has_global_id'] = global_id is not None
        
        patched_objects.append(patched_obj)
    
    patched_metadata['objects'] = patched_objects
    return patched_metadata


def stop_reid_system(stream_id: str, company_id: str = "default") -> None:
    """Stop and cleanup pipeline for specific stream."""
    cache_key = f"{company_id}:{stream_id}"
    
    if cache_key in _pipelines:
        pipeline = _pipelines.pop(cache_key)
        try:
            pipeline.stop()
            logger.info("[ReID_System] Stopped pipeline for %s", cache_key)
        except Exception as exc:
            logger.error("[ReID_System] Error stopping pipeline %s: %s", cache_key, exc)


def stop_all_reid_system() -> None:
    """Stop all pipelines."""
    for cache_key, pipeline in list(_pipelines.items()):
        try:
            pipeline.stop()
            logger.info("[ReID_System] Stopped pipeline %s", cache_key)
        except Exception as exc:
            logger.error("[ReID_System] Error stopping pipeline %s: %s", cache_key, exc)
    
    _pipelines.clear()
    logger.info("[ReID_System] All pipelines stopped")
