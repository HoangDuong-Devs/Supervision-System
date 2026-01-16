# reid_system/inference/pytorch_feature_extractor.py

"""
ReID Feature Extractor using PyTorch (matching BoxMOT implementation).

Uses the exact same preprocessing and normalization as BoxMOT.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch

from ..utils.vector_utils import l2_normalize

logger = logging.getLogger("PyTorchReIDFeatureExtractor")


class PyTorchReIDFeatureExtractor:
    """
    Extract ReID features using PyTorch model (matching BoxMOT's approach).
    """

    def __init__(
        self,
        model_path: str = "osnet_x1_0_msmt17.pt",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        enable_logging: bool = True,
        metadata_cache: Optional[Any] = None,
    ):
        self.logger = logger
        self.logger.setLevel(logging.INFO if enable_logging else logging.WARNING)

        self.model_path = model_path
        self.device = torch.device(device)
        self.model = None
        
        # Frame-based extraction interval
        self.extraction_interval = 5

        # ImageNet normalization constants (matching BoxMOT)
        self.mean_array = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std_array = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        
        # Input shape (matching BoxMOT for OSNet)
        self.input_shape = (256, 128)  # (height, width)

        # Initialize model
        self._init_model()

        self.frame_count = 0
        self._metadata_cache = metadata_cache

    def _init_model(self) -> None:
        """Initialize PyTorch model."""
        try:
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            
            from boxmot.appearance.backbones.osnet import osnet_x1_0
            
            # Load model
            self.model = osnet_x1_0(pretrained=False, num_classes=1041)
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            self.logger.info(
                "[PyTorchReIDFeatureExtractor] Model loaded: %s (device: %s)",
                self.model_path, self.device
            )
            
        except Exception as e:
            self.logger.error("[PyTorchReIDFeatureExtractor] Model init failed: %s", e)
            import traceback
            traceback.print_exc()
            self.model = None

    def get_crops(self, crops_bgr: List[np.ndarray]) -> torch.Tensor:
        """
        Preprocess crops exactly like BoxMOT.
        
        Args:
            crops_bgr: List of BGR image crops (already cropped from frame)
            
        Returns:
            Preprocessed tensor of shape (N, 3, 256, 128)
        """
        num_crops = len(crops_bgr)
        batch = torch.empty(
            (num_crops, 3, *self.input_shape),
            dtype=torch.float,
            device=self.device,
        )

        for i, crop in enumerate(crops_bgr):
            # Resize
            crop_resized = cv2.resize(
                crop,
                (self.input_shape[1], self.input_shape[0]),  # (width, height)
                interpolation=cv2.INTER_LINEAR,
            )
            
            # BGR to RGB
            crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)

            # Convert to tensor (H, W, C) -> (C, H, W)
            crop_tensor = torch.from_numpy(crop_rgb).to(self.device, dtype=torch.float)
            batch[i] = torch.permute(crop_tensor, (2, 0, 1))

        # Normalize to [0, 1]
        batch = batch / 255.0

        # Standardize with ImageNet mean/std
        batch = (batch - self.mean_array) / self.std_array

        return batch

    @torch.no_grad()
    def extract_features_batch(
        self,
        crops: List[np.ndarray],
        track_ids: List[str],
        stream_id: str,
        frame_idx: int,
    ) -> List[Optional[np.ndarray]]:
        """
        Extract ReID features from batch of crops.
        
        Args:
            crops: List of BGR image crops
            track_ids: Corresponding track IDs
            stream_id: Stream identifier
            frame_idx: Current frame index
            
        Returns:
            List of feature vectors (or None for skipped tracks)
        """
        if self.model is None:
            return [None] * len(crops)

        if len(crops) != len(track_ids):
            self.logger.warning("Crops and track_ids length mismatch")
            return [None] * len(crops)

        # Initialize results for all tracks
        results = [None] * len(crops)
        extract_indices = []
        extract_crops = []

        # Filter tracks that need feature extraction
        for i, (crop, track_id) in enumerate(zip(crops, track_ids)):
            if self.should_extract_features_for_track(track_id, stream_id, frame_idx):
                extract_indices.append(i)
                extract_crops.append(crop)
            else:
                # Use cached features if available
                if self._metadata_cache:
                    metadata = self._metadata_cache.get_or_create_track(track_id)
                    if metadata and metadata.last_person_vector is not None:
                        results[i] = metadata.last_person_vector.copy()

        if not extract_crops:
            return results

        # Process batch
        try:
            t0 = time.perf_counter()
            
            # Preprocess (matching BoxMOT)
            batch_tensor = self.get_crops(extract_crops)
            
            # Forward pass
            features_batch = self.model(batch_tensor)
            
            # Convert to numpy
            features_batch = features_batch.cpu().numpy()
            
            # L2 normalize each feature vector (matching BoxMOT)
            features_batch = features_batch / np.linalg.norm(features_batch, axis=-1, keepdims=True)
            
            elapsed = (time.perf_counter() - t0) * 1000
            self.logger.debug(
                "[PyTorchReIDFeatureExtractor] Batch inference: %d crops in %.2fms",
                len(extract_crops), elapsed
            )

            # Insert extracted features at correct positions
            for feat_idx, orig_idx in enumerate(extract_indices):
                if feat_idx < len(features_batch):
                    feature_vector = features_batch[feat_idx]
                    results[orig_idx] = feature_vector
                    
                    # Update metadata cache
                    if self._metadata_cache:
                        track_id = track_ids[orig_idx]
                        metadata = self._metadata_cache.get_or_create_track(track_id)
                        metadata.last_person_vector = feature_vector.copy()
                        metadata.last_person_vector_frame = frame_idx
                        metadata.person_extraction_count += 1
            
            return results

        except Exception as e:
            self.logger.error("[PyTorchReIDFeatureExtractor] Batch inference error: %s", e)
            import traceback
            traceback.print_exc()
            return [None] * len(crops)

    def should_extract_features_for_track(
        self,
        track_id: str,
        stream_id: str,
        frame_idx: int,
    ) -> bool:
        """
        Decide whether to extract features for this track.
        
        Extract every N frames per track.
        """
        if not self._metadata_cache:
            return True

        metadata = self._metadata_cache.get_or_create_track(track_id)
        
        # Extract if no vector exists
        if metadata.last_person_vector is None:
            return True
        
        # Extract every N frames
        frames_since_last = frame_idx - metadata.last_person_vector_frame
        return frames_since_last >= self.extraction_interval

    def supports_batch_inference(self) -> bool:
        """Check if batch inference is supported."""
        return True

    def process_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process metadata and extract ReID features.
        """
        try:
            stream_id = metadata.get("stream_id", "unknown")
            frame_number = metadata.get("frame_number", 0)
            
            person_list = metadata.get("person", [])
            if not person_list:
                metadata["reid_features"] = {"by_track_id": {}, "features": [], "tracks_extracted": []}
                return metadata

            # Extract features from all persons
            indices_to_extract: List[int] = list(range(len(person_list)))
            
            if indices_to_extract:
                # Get crops from person data
                crops = []
                valid_track_ids = []
                
                for i in indices_to_extract:
                    person_obj = person_list[i]
                    capture = person_obj.get("capture")
                    if isinstance(capture, np.ndarray):
                        crops.append(capture)
                        track_id = person_obj.get("id", f"track_{i}")
                        valid_track_ids.append(track_id)

                if crops and valid_track_ids:
                    # Extract features
                    body_feats = self.extract_features_batch(
                        crops, 
                        valid_track_ids,
                        stream_id,
                        frame_number
                    )
                    
                    # Build output
                    features_out_dict: Dict[str, List[float]] = {}
                    features_out: List[List[float]] = []
                    tracks_out: List[str] = []
                    
                    for feat, track_id in zip(body_feats, valid_track_ids):
                        if feat is not None:
                            features_out.append(feat.tolist())
                            tracks_out.append(track_id)
                            features_out_dict[track_id] = feat.tolist()

                    metadata["reid_features"] = {
                        "by_track_id": features_out_dict,
                        "features": features_out,
                        "tracks_extracted": tracks_out,
                    }
                else:
                    metadata["reid_features"] = {"by_track_id": {}, "features": [], "tracks_extracted": []}
            else:
                metadata["reid_features"] = {"by_track_id": {}, "features": [], "tracks_extracted": []}

            return metadata

        except Exception as e:
            self.logger.error("[PyTorchReIDFeatureExtractor] process_metadata error: %s", e)
            import traceback
            traceback.print_exc()
            metadata["reid_features"] = {"by_track_id": {}, "features": [], "tracks_extracted": []}
            return metadata
