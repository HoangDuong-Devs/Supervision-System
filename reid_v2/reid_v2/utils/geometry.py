# utils/geometry.py

"""
Centralized Geometry Utilities for Supervision System.

Merged from bbox_utils.py và spatial_utils.py để tránh phân tán code.

Features:
- Format conversions: LTWH ↔ XYWH ↔ X1Y1X2Y2
- Spatial computations: IoU, intersection ratio, proximity checks
- Optimized proximity heuristics for performance-critical paths
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


def ltwh_to_xywh(bbox: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
    """Convert LTWH to XYWH format."""
    if bbox is None:
        return None
    return {
        "x": bbox.get("left", 0.0),
        "y": bbox.get("top", 0.0),
        "w": bbox.get("width", 0.0),
        "h": bbox.get("height", 0.0),
    }


def ltwh_to_x1y1x2y2(bbox: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
    """Convert LTWH to X1Y1X2Y2 format."""
    if bbox is None:
        return None
    left = bbox.get("left", 0.0)
    top = bbox.get("top", 0.0)
    width = bbox.get("width", 0.0)
    height = bbox.get("height", 0.0)
    return {
        "x1": left,
        "y1": top,
        "x2": left + width,
        "y2": top + height,
    }


def ltwh_to_x1y1x2y2_int(bbox: Optional[Dict[str, float]]) -> Optional[Dict[str, int]]:
    """Convert LTWH to X1Y1X2Y2 format (int)."""
    if bbox is None:
        return None
    left = int(bbox.get("left", 0.0))
    top = int(bbox.get("top", 0.0))
    width = int(bbox.get("width", 0.0))
    height = int(bbox.get("height", 0.0))
    return {
        "x1": left,
        "y1": top,
        "x2": left + width,
        "y2": top + height,
    }


def normalize_to_x1y1x2y2(bbox: Any) -> Optional[Dict[str, int]]:
    """
    Convert any bbox format directly to X1Y1X2Y2 (int) format.
    
    This is the primary output format for implement.py.
    """
    ltwh = normalize_to_ltwh(bbox)
    if ltwh is None:
        return None
    return ltwh_to_x1y1x2y2_int(ltwh)


def normalize_to_xywh(bbox: Any) -> Optional[Dict[str, float]]:
    """
    Convert any bbox format directly to XYWH (float) format.
    """
    ltwh = normalize_to_ltwh(bbox)
    if ltwh is None:
        return None
    return ltwh_to_xywh(ltwh)


def bbox_to_coords_tuple(bbox: Any) -> Optional[Tuple[int, int, int, int]]:
    """
    Convert any bbox format to (x1, y1, x2, y2) tuple for drawing/visualization.
    
    Used by video_recorder.py.
    """
    if bbox is None:
        return None
    
    # Try direct tuple format
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        return (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    
    # Convert via LTWH
    x1y1x2y2 = normalize_to_x1y1x2y2(bbox)
    if x1y1x2y2 is None:
        return None
    
    return (
        x1y1x2y2["x1"],
        x1y1x2y2["y1"],
        x1y1x2y2["x2"],
        x1y1x2y2["y2"],
    )


def format_bbox_for_log(bbox: Any) -> str:
    """
    Format any bbox to a canonical string for logging: "x1,y1,x2,y2" or "None".
    
    This is the standard logging format across the supervision system.
    
    Args:
        bbox: Input bbox in any supported format
        
    Returns:
        String like "100,200,300,400" or "None" if invalid
        
    Example:
        >>> format_bbox_for_log({"left": 10, "top": 20, "width": 100, "height": 50})
        "10,20,110,70"
    """
    x1y1x2y2 = normalize_to_x1y1x2y2(bbox)
    if x1y1x2y2 is None:
        return "None"
    return f"{x1y1x2y2['x1']},{x1y1x2y2['y1']},{x1y1x2y2['x2']},{x1y1x2y2['y2']}"


# ========================================================================
#  SPATIAL COMPUTATIONS
# ========================================================================

def compute_intersection_ratio_safe(own_bbox: Optional[Dict[str, float]],
                                  other_bbox: Optional[Dict[str, float]]) -> float:
    """
    Compute intersection over own area ratio safely.

    Args:
        own_bbox: Bbox in LTWH format {left, top, width, height}
        other_bbox: Bbox in LTWH format {left, top, width, height}

    Returns:
        Intersection area / own area, or 0.0 if invalid inputs or no overlap
    """
    if not own_bbox or not other_bbox:
        return 0.0

    # Extract coordinates
    own_left = own_bbox.get("left", 0.0)
    own_top = own_bbox.get("top", 0.0)
    own_width = own_bbox.get("width", 0.0)
    own_height = own_bbox.get("height", 0.0)

    other_left = other_bbox.get("left", 0.0)
    other_top = other_bbox.get("top", 0.0)
    other_width = other_bbox.get("width", 0.0)
    other_height = other_bbox.get("height", 0.0)

    # Convert to x1,y1,x2,y2 for intersection calculation
    own_x1 = own_left
    own_y1 = own_top
    own_x2 = own_left + own_width
    own_y2 = own_top + own_height

    other_x1 = other_left
    other_y1 = other_top
    other_x2 = other_left + other_width
    other_y2 = other_top + other_height

    # Compute intersection
    inter_x1 = max(own_x1, other_x1)
    inter_y1 = max(own_y1, other_y1)
    inter_x2 = min(own_x2, other_x2)
    inter_y2 = min(own_y2, other_y2)

    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    # Own area
    own_area = own_width * own_height
    if own_area <= 0.0:
        return 0.0

    return inter_area / own_area


def compute_iou(bbox1: Optional[Dict[str, float]],
                bbox2: Optional[Dict[str, float]]) -> float:
    """
    Compute Intersection over Union (IoU) between two bboxes.

    Args:
        bbox1: First bbox in LTWH format {left, top, width, height}
        bbox2: Second bbox in LTWH format {left, top, width, height}

    Returns:
        IoU value between 0.0 and 1.0, or 0.0 if invalid inputs
    """
    if not bbox1 or not bbox2:
        return 0.0

    # Extract coordinates
    b1_left = bbox1.get("left", 0.0)
    b1_top = bbox1.get("top", 0.0)
    b1_width = bbox1.get("width", 0.0)
    b1_height = bbox1.get("height", 0.0)

    b2_left = bbox2.get("left", 0.0)
    b2_top = bbox2.get("top", 0.0)
    b2_width = bbox2.get("width", 0.0)
    b2_height = bbox2.get("height", 0.0)

    # Convert to x1,y1,x2,y2
    b1_x1 = b1_left
    b1_y1 = b1_top
    b1_x2 = b1_left + b1_width
    b1_y2 = b1_top + b1_height

    b2_x1 = b2_left
    b2_y1 = b2_top
    b2_x2 = b2_left + b2_width
    b2_y2 = b2_top + b2_height

    # Compute intersection
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    # Compute union
    b1_area = b1_width * b1_height
    b2_area = b2_width * b2_height
    union_area = b1_area + b2_area - inter_area

    if union_area <= 0.0:
        return 0.0

    return inter_area / union_area


def are_bboxes_close(bbox1: Optional[Dict[str, float]],
                    bbox2: Optional[Dict[str, float]],
                    threshold: float = 0.3) -> bool:
    """
    Check if two bboxes are spatially close using optimized proximity checks.

    Uses multiple heuristics for efficiency:
    1. Center distance check (fast)
    2. IoU check (accurate but slower)

    Args:
        bbox1: First bbox in LTWH format {left, top, width, height}
        bbox2: Second bbox in LTWH format {left, top, width, height}
        threshold: Proximity threshold (0.0-1.0), higher = more lenient

    Returns:
        True if bboxes are considered close
    """
    if not bbox1 or not bbox2:
        return False

    # Extract coordinates
    b1_left = bbox1.get("left", 0.0)
    b1_top = bbox1.get("top", 0.0)
    b1_width = bbox1.get("width", 0.0)
    b1_height = bbox1.get("height", 0.0)

    b2_left = bbox2.get("left", 0.0)
    b2_top = bbox2.get("top", 0.0)
    b2_width = bbox2.get("width", 0.0)
    b2_height = bbox2.get("height", 0.0)

    # Convert to x1,y1,x2,y2
    b1_x1 = b1_left
    b1_y1 = b1_top
    b1_x2 = b1_left + b1_width
    b1_y2 = b1_top + b1_height

    b2_x1 = b2_left
    b2_y1 = b2_top
    b2_x2 = b2_left + b2_width
    b2_y2 = b2_top + b2_height

    # Fast center distance check
    b1_center_x = (b1_x1 + b1_x2) / 2.0
    b1_center_y = (b1_y1 + b1_y2) / 2.0
    b2_center_x = (b2_x1 + b2_x2) / 2.0
    b2_center_y = (b2_y1 + b2_y2) / 2.0

    center_dist = math.sqrt((b1_center_x - b2_center_x)**2 + (b1_center_y - b2_center_y)**2)

    # Average bbox size for normalization
    avg_size = ((b1_width + b1_height + b2_width + b2_height) / 4.0)

    # If centers are too far apart, not close
    if avg_size > 0.0 and center_dist > avg_size * 2.0:
        return False

    # If centers are very close, definitely close
    if avg_size > 0.0 and center_dist < avg_size * 0.5:
        return True

    # Otherwise, check IoU
    iou = compute_iou(bbox1, bbox2)
    return iou >= threshold


def get_bbox_center(bbox: Optional[Dict[str, float]]) -> Optional[Tuple[float, float]]:
    """
    Get the center coordinates of a bbox.

    Args:
        bbox: Bbox in LTWH format {left, top, width, height}

    Returns:
        (center_x, center_y) tuple or None if invalid
    """
    if not bbox:
        return None

    left = bbox.get("left", 0.0)
    top = bbox.get("top", 0.0)
    width = bbox.get("width", 0.0)
    height = bbox.get("height", 0.0)

    center_x = left + width / 2.0
    center_y = top + height / 2.0

    return (center_x, center_y)


def get_bbox_area(bbox: Optional[Dict[str, float]]) -> float:
    """
    Get the area of a bbox.

    Args:
        bbox: Bbox in LTWH format {left, top, width, height}

    Returns:
        Area as float, 0.0 if invalid
    """
    if not bbox:
        return 0.0

    width = bbox.get("width", 0.0)
    height = bbox.get("height", 0.0)

    return width * height