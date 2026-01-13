# reid_v2_demo/utils/vector_utils.py

"""
Vector and similarity computation utilities for ReID V2 Demo.

Copied exactly from reid_v2.utils.vector_utils.py
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """
    L2 normalize vector to unit norm.
    
    Args:
        vec: Input vector (1D or multi-dimensional)
        
    Returns:
        Normalized vector with norm=1.0, or zeros if input has zero norm
    """
    if vec is None:
        return vec
    if vec.ndim > 1:
        vec = vec.reshape(-1)
    n = float(np.linalg.norm(vec))
    if n > 0.0:
        vec = (vec / n).astype(np.float32, copy=False)
    else:
        vec = np.zeros_like(vec, dtype=np.float32)
    return vec


def cosine_similarity_normalized(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity clipped to [0, 1] range.
    
    Range: [0, 1] where 1 = identical, 0.5 = orthogonal, 0 = opposite.
    Useful for similarity metrics where negative scores don't make sense.
    
    NOTE: This function normalizes vectors internally. If vectors are already
    L2-normalized, use cosine_similarity_prenormalized() for better performance.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity clipped to [0, 1]
    """
    if vec1.ndim == 1:
        vec1 = vec1.reshape(1, -1)
    if vec2.ndim == 1:
        vec2 = vec2.reshape(1, -1)
    v1 = vec1 / (np.linalg.norm(vec1, axis=1, keepdims=True) + 1e-8)
    v2 = vec2 / (np.linalg.norm(vec2, axis=1, keepdims=True) + 1e-8)
    sim = float(np.dot(v1, v2.T)[0, 0])
    sim = np.nan_to_num(sim, nan=0.0)
    return float(np.clip(sim, 0.0, 1.0))