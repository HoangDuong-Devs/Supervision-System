# reid_v2_demo/utils/geometry.py

"""
Centralized Geometry Utilities for ReID V2 Demo.

Copied exactly from reid_v2.utils.geometry.py
"""

from typing import Any, Dict, Optional, Tuple
import math
import logging

logger = logging.getLogger(__name__)


# ========================================================================
#  BBOX FORMAT CONVERSIONS
# ========================================================================

def normalize_to_ltwh(bbox: Any) -> Optional[Dict[str, float]]:
    """
    Convert any bbox format to canonical LTWH format {left, top, width, height}.
    
    Args:
        bbox: Input bbox in any supported format
        
    Returns:
        Dict with keys {left, top, width, height} as float, or None if invalid
    """
    if bbox is None:
        return None
    
    # Already LTWH
    if isinstance(bbox, dict):
        if all(k in bbox for k in ("left", "top", "width", "height")):
            return {
                "left": float(bbox["left"]),
                "top": float(bbox["top"]),
                "width": float(bbox["width"]),
                "height": float(bbox["height"]),
            }
        
        # XYWH format
        if all(k in bbox for k in ("x", "y", "w", "h")):
            return {
                "left": float(bbox["x"]),
                "top": float(bbox["y"]),
                "width": float(bbox["w"]),
                "height": float(bbox["h"]),
            }
        
        # X1Y1X2Y2 format
        if all(k in bbox for k in ("x1", "y1", "x2", "y2")):
            x1 = float(bbox["x1"])
            y1 = float(bbox["y1"])
            x2 = float(bbox["x2"])
            y2 = float(bbox["y2"])
            return {
                "left": x1,
                "top": y1,
                "width": max(0.0, x2 - x1),
                "height": max(0.0, y2 - y1),
            }

        # DeepStream format: {topleftx, toplefty, bottomrightx, bottomrighty}
        if all(k in bbox for k in ("topleftx", "toplefty", "bottomrightx", "bottomrighty")):
            left = float(bbox["topleftx"])
            top = float(bbox["toplefty"])
            right = float(bbox["bottomrightx"])
            bottom = float(bbox["bottomrighty"])
            return {
                "left": left,
                "top": top,
                "width": max(0.0, right - left),
                "height": max(0.0, bottom - top),
            }
    
    # List/tuple format: interpret as [x1, y1, x2, y2]
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        x1 = float(bbox[0])
        y1 = float(bbox[1])
        x2 = float(bbox[2])
        y2 = float(bbox[3])
        return {
            "left": x1,
            "top": y1,
            "width": max(0.0, x2 - x1),
            "height": max(0.0, y2 - y1),
        }
    
    logger.warning("Unknown bbox format for normalization: %s (type: %s)", bbox, type(bbox))
    return None


def compute_iou(bbox1: Dict[str, float], bbox2: Dict[str, float]) -> float:
    """
    Compute IoU (Intersection over Union) between two LTWH bboxes.
    
    Args:
        bbox1, bbox2: Bboxes in LTWH format
        
    Returns:
        IoU score [0, 1]
    """
    # Extract coordinates
    left1, top1, width1, height1 = bbox1["left"], bbox1["top"], bbox1["width"], bbox1["height"]
    left2, top2, width2, height2 = bbox2["left"], bbox2["top"], bbox2["width"], bbox2["height"]
    
    # Compute corners
    right1, bottom1 = left1 + width1, top1 + height1
    right2, bottom2 = left2 + width2, top2 + height2
    
    # Intersection area
    inter_left = max(left1, left2)
    inter_top = max(top1, top2)
    inter_right = min(right1, right2)
    inter_bottom = min(bottom1, bottom2)
    
    if inter_right <= inter_left or inter_bottom <= inter_top:
        return 0.0
    
    inter_area = (inter_right - inter_left) * (inter_bottom - inter_top)
    
    # Union area
    area1 = width1 * height1
    area2 = width2 * height2
    union_area = area1 + area2 - inter_area
    
    if union_area <= 0:
        return 0.0
    
    return inter_area / union_area


def are_bboxes_close(
    bbox1: Dict[str, float], 
    bbox2: Dict[str, float],
    distance_threshold: float = 100.0,
    size_ratio_threshold: float = 3.0
) -> bool:
    """
    Check if two bboxes are spatially close (heuristic for motion matching).
    
    Args:
        bbox1, bbox2: Bboxes in LTWH format
        distance_threshold: Max center distance in pixels
        size_ratio_threshold: Max size ratio between bboxes
        
    Returns:
        True if bboxes are considered close
    """
    # Center distance
    center1_x = bbox1["left"] + bbox1["width"] / 2
    center1_y = bbox1["top"] + bbox1["height"] / 2
    center2_x = bbox2["left"] + bbox2["width"] / 2
    center2_y = bbox2["top"] + bbox2["height"] / 2
    
    distance = math.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
    
    if distance > distance_threshold:
        return False
    
    # Size ratio check
    area1 = bbox1["width"] * bbox1["height"]
    area2 = bbox2["width"] * bbox2["height"]
    
    if area1 <= 0 or area2 <= 0:
        return False
    
    ratio = max(area1, area2) / min(area1, area2)
    
    return ratio <= size_ratio_threshold