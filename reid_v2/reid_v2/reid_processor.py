# reid_v2/reid_processor.py

"""
ReID V2 Processor - Entry point cho Long-term ReID với RAM backend.

Flow-based design (không phải service), giữ nguyên logic từ supervision_system_ram.

Sử dụng:
    from reid_v2 import process_reid_v2
    
    # Trong add_face.py, trước khi gửi transmitter:
    output_metadata = process_reid_v2(stream_id, input_metadata)
"""

import logging
from typing import Any, Dict, List, Optional

from ServiceApp.utils import load_frame_from_shm, crop_roi_from_frame
from .main_logic.pipeline import ReIDPipeline
# Use unified GlobalIDBuffer from reid module
from reid.emit.global_id_buffer import GlobalIDBuffer

logger = logging.getLogger(__name__)

# Cache per stream
_pipelines: Dict[str, ReIDPipeline] = {}
_buffers: Dict[str, GlobalIDBuffer] = {}


def process_reid_v2(stream_id: str, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Process ReID và trả về metadata đã patch Global ID.
    
    Args:
        stream_id: ID của stream
        metadata: Metadata từ add_face (chứa shm_info và objects)
        
    Returns:
        Metadata đã patch global_id (delayed vì buffer),
        hoặc None nếu còn trong buffer.
    """
    try:
        pipeline = _get_pipeline(stream_id)
        buffer = _get_buffer(stream_id)
        
        if not pipeline:
            logger.warning("[ReID_v2] No pipeline for stream=%s, fallback", stream_id)
            return buffer.add_frame(metadata)
        
        # Load frame từ SHM
        frame = _load_frame(metadata)
        frame_number = metadata.get("frame_number", 0)
        
        if frame is None:
            
            # Không có frame → chỉ buffer, không process ReID
            return buffer.add_frame(metadata)
        
        # Extract person data và crop từ frame, rebuild metadata format
        person_data = _extract_person_crops(metadata, frame)
        
        if person_data:
            # Build metadata format expected by run_reid
            person_ids = [p["id"] for p in person_data]
            person_bboxes = [p.get("bbox", {}) for p in person_data]
            person_captures = [p.get("capture") for p in person_data]
            
            reid_metadata = {
                "stream_id": stream_id,
                "frame_number": frame_number,
                "timestamp": metadata.get("timestamp"),
                "person": {
                    "ids": person_ids,
                    "bboxes": person_bboxes,
                    "captures": person_captures,
                },
                "objects": metadata.get("objects", []),
            }
            
            # Step 1: run_reid (feature extraction)
            try:
                face_crops_dict, face_rec_map = pipeline.run_reid(
                    metadata=reid_metadata,
                    face_person_map={},  # No face mapping in simple flow
                    face_recognition_results=None,
                    face_data=None,
                )
            except Exception as exc:
                logger.error("[ReID_v2] run_reid failed: %s", exc, exc_info=True)
                return buffer.add_frame(metadata)
            
            # Step 2: assign_global_ids (voting + assignment)
            try:
                pipeline.assign_global_ids(
                    metadata=reid_metadata,
                    face_crops_dict=face_crops_dict,
                    face_rec_map=face_rec_map,
                )
            except Exception as exc:
                logger.error("[ReID_v2] assign_global_ids failed: %s", exc, exc_info=True)
            
            # Copy results back to original metadata
            for obj in metadata.get("objects", []):
                if obj.get("class") != "person":
                    continue
                # Find matching object in reid_metadata
                for reid_obj in reid_metadata.get("objects", []):
                    if str(reid_obj.get("id")) == str(obj.get("id")):
                        obj["global_id"] = reid_obj.get("global_id")
                        obj["state"] = reid_obj.get("state", "PENDING")
                        break
            
            # Update buffer with ASSIGNED tracks
            for obj in metadata.get("objects", []):
                if obj.get("class") == "person":
                    state = obj.get("state")
                    gid = obj.get("global_id")
                    tid = obj.get("id")
                    if state == "ASSIGNED" and gid and tid:
                        buffer.notify_assigned(str(tid), str(gid))
        
        # Push frame vào buffer, pop frame cũ đã confirm
        return buffer.add_frame(metadata)
        
    except Exception as exc:
        logger.error("[ReID_v2] Error: %s", exc, exc_info=True)
        return metadata  # Fallback


def _load_frame(metadata: Dict[str, Any]) -> Optional[Any]:
    """Load frame từ SHM nếu có shm_info."""
    shm_info = metadata.get("shm_info")
    if not shm_info:
        return None
    
    try:
        frame = load_frame_from_shm(shm_info)
        return frame
    except Exception as exc:
        logger.warning("[ReID_v2] Failed to load frame from SHM: %s", exc)
        return None


def _extract_person_crops(metadata: Dict[str, Any], frame) -> List[Dict[str, Any]]:
    """Extract person crops từ frame dựa trên bbox."""
    from configs.autocfg import cfg
    
    min_conf = getattr(getattr(cfg, "REID", None), "QUALITY_FILTER", None)
    min_conf = getattr(min_conf, "MIN_CONFIDENCE", 0.7) if min_conf else 0.7
    
    person_data = []
    
    for obj in metadata.get("objects", []):
        if obj.get("class") != "person":
            continue
        
        track_id = obj.get("id")
        bbox = obj.get("bbox")
        conf = obj.get("confidence", 1.0)
        
        if track_id is None or bbox is None:
            continue
        
        # Filter theo confidence
        if conf < min_conf:
            continue
        
        # Crop person từ frame
        try:
            # Normalize bbox format for crop_roi_from_frame
            crop_bbox = {
                "topleftx": bbox.get("x1", bbox.get("topleftx", 0)),
                "toplefty": bbox.get("y1", bbox.get("toplefty", 0)),
                "bottomrightx": bbox.get("x2", bbox.get("bottomrightx", 0)),
                "bottomrighty": bbox.get("y2", bbox.get("bottomrighty", 0)),
            }
            crop = crop_roi_from_frame(frame, crop_bbox, crop_option="no_transform")
            if crop is not None and crop.size > 0:
                person_data.append({
                    "id": track_id,
                    "capture": crop,
                    "bbox": bbox,
                    "confidence": conf,
                })
        except Exception as exc:
            continue
    
    return person_data


def _get_pipeline(stream_id: str) -> Optional[ReIDPipeline]:
    """Lazy init pipeline."""
    if stream_id not in _pipelines:
        try:
            import os
            company_id = os.getenv("COMPANY_ID", "default_company")
            pipeline = ReIDPipeline(stream_id=stream_id, company_id=company_id)
            pipeline.initialize()
            _pipelines[stream_id] = pipeline
            
        except Exception as exc:
            logger.error("[ReID_v2] Failed to create pipeline: %s", exc)
            _pipelines[stream_id] = None
    return _pipelines.get(stream_id)


def _get_buffer(stream_id: str) -> GlobalIDBuffer:
    """Lazy init buffer."""
    from configs.autocfg import cfg
    
    if stream_id not in _buffers:
        # Read shared REID config
        reid_cfg = getattr(cfg, "REID", None)
        
        # Buffer Size
        buffer_size = getattr(reid_cfg, "BUFFER_SIZE", 100)
        
        # Override ID
        override_id = getattr(reid_cfg, "OVERRIDE_ID", True)
        
        # Cache Max Size
        cache_max_size = 500
        
        _buffers[stream_id] = GlobalIDBuffer(
            size=buffer_size,
            override_id=override_id,
            cache_max_size=cache_max_size,
        )
        
    return _buffers[stream_id]


def flush_reid_v2(stream_id: str) -> list:
    """Flush tất cả frames còn lại (end of stream)."""
    buffer = _buffers.get(stream_id)
    if buffer:
        return buffer.flush_all()
    return []


def stop_reid_v2(stream_id: str) -> None:
    """Stop và cleanup."""
    pipeline = _pipelines.pop(stream_id, None)
    if pipeline and hasattr(pipeline, 'stop'):
        pipeline.stop()
    _buffers.pop(stream_id, None)


def stop_all_v2() -> None:
    """Stop tất cả."""
    for stream_id in list(_pipelines.keys()):
        stop_reid_v2(stream_id)
