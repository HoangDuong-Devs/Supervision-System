# reid_system/demo_tracking.py

"""
Demo Tracking Script with Full ReID Pipeline.

Integrates:
- YOLOv7 for detection
- BoxMOT tracker (WITHOUT built-in ReID)
- ReID System pipeline for long-term identity

This mirrors the reid_system flow:
1. Detection -> Tracking (short-term IDs)
2. ReID feature extraction (interval-based)
3. Voting-based Global ID assignment
4. Vector storage for re-identification
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add yolov7 to path for relative imports BEFORE any yolov7 imports
yolov7_path = Path(__file__).parent.parent / "yolov7"
if str(yolov7_path) not in sys.path:
    sys.path.insert(0, str(yolov7_path))

import time
import logging
from dataclasses import dataclass

import cv2
import numpy as np
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,  # Changed to INFO to reduce verbosity
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

from yolov7.utils.datasets import letterbox
from yolov7.models.experimental import attempt_load
from yolov7.utils.general import non_max_suppression, scale_coords

# BoxMOT tracker
from boxmot.tracker_zoo import create_tracker
from boxmot.utils import TRACKER_CONFIGS


@dataclass
class DemoConfig:
    """Configuration for demo tracking."""
    # Video source
    # source: str = r"rtsp://10.100.140.70:8554/d0bac86ce0fa41978e70592c24bd1cdd_7fdd7585ee7b43cc82864a59950849e1_1"
    source: str = r"test_video\aicity.mp4"
    output: str = "reid_demo_output.avi"
    
    # Detection model
    yolo_weights: str = r"F:\VScode_NHD\boxmot\best.pt"
    detect_class: int = 2  # COCO: 0=head, 2=visible body
    conf_thres: float = 0.3
    iou_thres: float = 0.65
    img_size: int = 640
    
    # Tracker (WITHOUT ReID - we handle ReID separately)
    tracker_type: str = "botsort"
    
    # ReID ONNX model
    onnx_model_path: str = r"F:\VScode_NHD\boxmot\osnet_x1_0_msmt17.onnx"
    extraction_interval: int = 5
    
    # Voting
    vote_window_size: int = 10
    early_exit_min_votes: int = 3
    
    # Display
    max_width: int = 1280
    max_height: int = 720
    show_global_id: bool = True
    show_track_id: bool = True
    show_pending: bool = True
    show_stats: bool = True
    
    # Device
    device: str = "0"
    half: bool = False


# Color palette for visualization
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 0, 0), (0, 128, 0), (0, 0, 128),
    (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (255, 128, 0), (255, 0, 128), (128, 255, 0),
    (0, 255, 128), (128, 0, 255), (0, 128, 255),
    (255, 128, 128), (128, 255, 128), (128, 128, 255),
    (64, 64, 64), (192, 192, 192), (255, 165, 0),
]


def detect_source_type(source: str) -> str:
    """Detect the type of video source.
    
    Returns:
        'webcam' for integer camera IDs
        'rtsp' for RTSP streams
        'http' for HTTP streams
        'file' for local video files
    """
    # Check if it's a camera ID
    try:
        int(source)
        return 'webcam'
    except ValueError:
        pass
    
    # Check for URL protocols
    source_lower = source.lower()
    if source_lower.startswith(('rtsp://', 'rtmp://')):
        return 'rtsp'
    elif source_lower.startswith(('http://', 'https://')):
        return 'http'
    else:
        return 'file'


def open_video_source(source: str):
    """Open video source with appropriate backend."""
    source_type = detect_source_type(source)
    logger.info(f"Detected source type: {source_type} for {source}")
    
    if source_type == 'webcam':
        # Camera input
        cam_id = int(source)
        cap = cv2.VideoCapture(cam_id)
        logger.info(f"Opening webcam ID: {cam_id}")
    elif source_type == 'rtsp':
        # RTSP stream - use FFMPEG backend for better compatibility
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        logger.info(f"Opening RTSP stream: {source}")
        # Set buffer size to reduce latency for live streams
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    elif source_type == 'http':
        # HTTP stream
        cap = cv2.VideoCapture(source)
        logger.info(f"Opening HTTP stream: {source}")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        # Local file
        cap = cv2.VideoCapture(source)
        logger.info(f"Opening video file: {source}")
    
    return cap, source_type


def get_color_for_gid(global_id: Optional[str]) -> Tuple[int, int, int]:
    """Generate a consistent, visually distinct color for a given ID."""
    if not global_id:
        return (128, 128, 128)  # Gray for no ID
    
    # Use hash to generate consistent color
    h = hash(str(global_id)) & 0xFFFFFF
    # Ensure vibrant colors by keeping values in range 80-255
    r = 80 + (h & 0xFF) % 176
    g = 80 + ((h >> 8) & 0xFF) % 176
    b = 80 + ((h >> 16) & 0xFF) % 176
    return (b, g, r)  # BGR for OpenCV


def draw_detections(
    frame: np.ndarray,
    detections: np.ndarray,
    global_ids: Dict[str, Optional[str]],
    config: DemoConfig,
) -> np.ndarray:
    """Draw bounding boxes with track and global IDs with improved visualization."""
    vis = frame.copy()
    H, W = vis.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8  # Increased for better readability
    thickness = 2
    padding = 5  # Increased padding

    # Helper function to draw text with shadow
    def draw_text_with_shadow(img, text, pos, font, scale, color, thickness, shadow_color=(0, 0, 0)):
        # Draw shadow
        cv2.putText(img, text, (pos[0] + 1, pos[1] + 1), font, scale, shadow_color, thickness, cv2.LINE_AA)
        # Draw main text
        cv2.putText(img, text, pos, font, scale, color, thickness, cv2.LINE_AA)

    for det in detections:
        x1, y1, x2, y2 = det[:4].astype(int)
        track_id = str(int(det[4]))
        conf = det[5] if len(det) > 5 else 0
        
        global_id = global_ids.get(track_id)
        color = get_color_for_gid(global_id)
        
        # Draw bbox with thicker line
        thickness_bbox = 3 if global_id else 2
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness_bbox)
        
        # Build label
        parts = []
        if config.show_global_id:
            if global_id:
                parts.append(f"GID:{global_id}")
            elif config.show_pending:
                parts.append("PENDING")
        
        if config.show_track_id:
            parts.append(f"T:{track_id}")
        
        label = " | ".join(parts) if parts else f"T:{track_id}"
        
        # Draw label with semi-transparent background
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        box_w = tw + padding * 2
        box_h = th + padding * 2

        top_x1 = max(0, min(x1, W - box_w))
        top_y2 = max(box_h, y1)
        top_y1 = max(0, top_y2 - box_h)

        # Semi-transparent overlay
        overlay = vis[top_y1:top_y2, top_x1:top_x1 + box_w]
        cv2.rectangle(overlay, (0, 0), (box_w, box_h), color, -1)
        vis[top_y1:top_y2, top_x1:top_x1 + box_w] = cv2.addWeighted(overlay, 0.7, vis[top_y1:top_y2, top_x1:top_x1 + box_w], 0.3, 0)

        draw_text_with_shadow(
            vis, label,
            (top_x1 + padding, top_y2 - padding),
            font, font_scale, (255, 255, 255), thickness
        )
    
    return vis


def draw_stats(
    frame: np.ndarray,
    stats: Dict,
    fps: float,
    avg_fps: float,
    frame_idx: int,
) -> np.ndarray:
    """Draw statistics overlay."""
    lines = [
        f"Frame: {frame_idx} | FPS: {fps:.1f} | Avg: {avg_fps:.1f}",
        f"Tracks: {stats.get('total_tracks', 0)} (Assigned: {stats.get('assigned_tracks', 0)}, Pending: {stats.get('pending_tracks', 0)})",
        f"Global IDs: {stats.get('unique_global_ids', 0)} | Vectors: {stats.get('vector_store_used', 0)}",
    ]
    
    y = 30
    for line in lines:
        # Background
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (15, y - th - 5), (25 + tw, y + 5), (0, 0, 0), -1)
        # Text
        cv2.putText(
            frame, line, (20, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA
        )
        y += 30
    
    return frame


def run_demo(config: DemoConfig) -> None:
    """Run demo tracking with full ReID pipeline."""
    
    # Setup device
    if config.device.lower() == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{config.device}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # =================================================================
    # Load YOLOv7
    # =================================================================
    logger.info(f"Loading YOLOv7 from {config.yolo_weights}")
    yolo_model = attempt_load(config.yolo_weights, map_location="cpu")
    yolo_model.to(device)
    yolo_model.eval()
    
    # =================================================================
    # Create Tracker (WITHOUT ReID - we handle it ourselves)
    # =================================================================
    logger.info(f"Creating tracker: {config.tracker_type} (without built-in ReID)")
    
    # Use bytetrack/ocsort which don't require ReID, or provide dummy weights for others
    tracker_args = {
        "tracker_type": config.tracker_type,
        "tracker_config": TRACKER_CONFIGS / f"{config.tracker_type}.yaml",
        "half": config.half,
        "per_class": False,
        "device": device,
    }
    
    # Trackers that require reid_weights argument
    reid_trackers = ["botsort", "strongsort", "deepocsort", "hybridsort", "boosttrack"]
    
    if config.tracker_type in reid_trackers:
        # These trackers need reid_weights, but we disable with with_reid=False
        # Provide a dummy path (model won't be loaded if with_reid=False)
        tracker_args["reid_weights"] = Path(config.onnx_model_path)  # Use same path
        tracker_args["with_reid"] = False
    
    tracker = create_tracker(**tracker_args)
    
    # =================================================================
    # Initialize ReID System (no async needed - direct call)
    # =================================================================
    from reid_system.reid_processor import process_reid_system
    
    # Initialize reid pipeline for demo
    reid_pipeline = None
    try:
        # Setup reid demo with dummy metadata to initialize pipeline
        dummy_metadata = {
            'frame': None,
            'frame_idx': 0,
            'objects': [],
        }
        process_reid_system("demo", dummy_metadata, onnx_model_path=config.onnx_model_path, company_id="demo")
        reid_pipeline = True  # Flag to indicate reid is ready
        logger.info("ReID System pipeline initialized")
    except Exception as e:
        logger.warning("ReID pipeline initialization failed: %s, continuing with tracking only", e)
        reid_pipeline = None
    
    logger.info("ReID System ready with ONNX model: %s", config.onnx_model_path)
    
    # =================================================================
    # Open Video
    # =================================================================
    cap, source_type = open_video_source(config.source)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {config.source}")
        return
    
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # For live streams, frame count may not be available
    if source_type in ['rtsp', 'http', 'webcam']:
        total_frames = 0  # Unknown for live streams
        logger.info("Live stream detected - frame count unavailable")
    else:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Output size
    scale = min(config.max_width / width, config.max_height / height, 1.0)
    out_w = int(width * scale)
    out_h = int(height * scale)
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(config.output, fourcc, fps_video, (out_w, out_h))
    
    if source_type in ['rtsp', 'http', 'webcam']:
        logger.info(f"Live stream: {width}x{height} @ {fps_video:.1f}fps")
    else:
        logger.info(f"Video: {width}x{height} @ {fps_video:.1f}fps, {total_frames} frames")
    logger.info(f"Output: {out_w}x{out_h} -> {config.output}")
    
    # =================================================================
    # Main Loop
    # =================================================================
    frame_idx = 0
    start_time = time.time()
    prev_time = start_time
    
    logger.info("=" * 60)
    logger.info("Starting tracking loop... Press 'q' to quit")
    logger.info("=" * 60)
    
    # Persist global_ids across frames
    global_ids = {}
    
    try:
        while True:
            frame_start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            
            # FPS calculation
            curr_time = time.time()
            fps_current = 1.0 / max(curr_time - prev_time, 1e-6)
            avg_fps = (frame_idx + 1) / max(curr_time - start_time, 1e-6)
            prev_time = curr_time
            
            # =============================================================
            # Step 1: Detection with YOLOv7
            # =============================================================
            img, ratio, pad = letterbox(frame, new_shape=config.img_size, auto=False)
            img = img.transpose((2, 0, 1))[::-1]
            img = np.ascontiguousarray(img)
            img_tensor = torch.from_numpy(img).to(device).float() / 255.0
            
            if img_tensor.ndimension() == 3:
                img_tensor = img_tensor.unsqueeze(0)
            
            with torch.no_grad():
                pred = yolo_model(img_tensor, augment=False)[0]
                dets = non_max_suppression(pred, conf_thres=config.conf_thres, iou_thres=config.iou_thres)[0]
            
            display_frame = frame.copy()
            tracked = np.array([])
            
            if dets is not None and len(dets):
                # Scale to original frame
                dets[:, :4] = scale_coords(
                    img_tensor.shape[2:], dets[:, :4], frame.shape,
                    ratio_pad=(ratio, pad)
                ).round()
                
                # Filter by class
                dets = dets[dets[:, 5] == config.detect_class]
                
                if len(dets):
                    # =============================================================
                    # Step 2: Tracking (short-term IDs, NO ReID)
                    # =============================================================
                    tracked = tracker.update(dets.cpu().numpy(), frame)
                    
                    if len(tracked):
                        # =============================================================
                        # Step 3: ReID V2 Demo Pipeline (long-term Global IDs)
                        # =============================================================
                        # ONLY process ReID on extraction interval frames
                        should_extract = (frame_idx % config.extraction_interval == 0)
                        
                        if should_extract:
                            # Convert tracked detections to objects format
                            objects = []
                            for det in tracked:
                                if len(det) >= 7:
                                    objects.append({
                                        'track_id': int(det[4]),
                                        'id': int(det[4]),  # Alias
                                        'class': 'person',
                                        'class_id': int(det[6]) if len(det) > 6 else 0,
                                        'confidence': float(det[5]),
                                        'bbox': det[:4].tolist(),  # x1, y1, x2, y2
                                    })
                            
                            # Process through ReID System
                            metadata = {
                                'frame': frame,
                                'frame_idx': frame_idx,
                                'objects': objects,
                            }
                            
                            processed_metadata = process_reid_system(
                                stream_id="demo",
                                metadata=metadata,
                                onnx_model_path=config.onnx_model_path,
                                company_id="demo",
                            )
                            
                            # Update global IDs from processed metadata (preserve existing)
                            if processed_metadata and processed_metadata.get('objects'):
                                for obj in processed_metadata['objects']:
                                    track_id = str(obj.get('track_id', obj.get('id', '')))
                                    global_id = obj.get('global_id')
                                    global_ids[track_id] = global_id
                                
                        # =============================================================
                        # Step 4: Visualization
                        # =============================================================
                        display_frame = draw_detections(
                            display_frame, tracked, global_ids, config
                        )
            
            # Draw stats (simplified for now)
            if config.show_stats:
                # Simple stats without reid_pipeline
                total_tracks = len(tracked) if len(tracked) else 0
                assigned_gids = len([gid for gid in global_ids.values() if gid is not None]) if 'global_ids' in locals() else 0
                
                stats_text = [
                    f"FPS: {avg_fps:.1f}",
                    f"Tracks: {total_tracks}",
                    f"Global IDs: {assigned_gids}",
                ]
                
                # Add frame counter (different format for live vs file)
                if source_type in ['rtsp', 'http', 'webcam']:
                    stats_text.append(f"Frame: {frame_idx} (live)")
                else:
                    stats_text.append(f"Frame: {frame_idx}/{total_frames}")
                
                y_offset = 30
                for line in stats_text:
                    cv2.putText(display_frame, line, (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    y_offset += 25
            
            # Resize for output
            display_frame = cv2.resize(display_frame, (out_w, out_h))
            
            # Display
            cv2.imshow("ReID Demo", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Quit requested by user")
                break
            
            # Write output
            writer.write(display_frame)
            
            # Progress log
            if frame_idx % 100 == 0:
                assigned_gids = len([gid for gid in global_ids.values() if gid is not None]) if 'global_ids' in locals() else 0
                total_tracks = len(tracked) if len(tracked) else 0
                logger.info(
                    f"Frame {frame_idx}/{total_frames} | FPS: {avg_fps:.1f} | "
                    f"Tracks: {total_tracks} | GIDs: {assigned_gids}"
                )
            
            # Frame processing time log
            frame_time = (time.time() - frame_start_time) * 1000  # Convert to ms
            assigned_gids = len([gid for gid in global_ids.values() if gid is not None]) if 'global_ids' in locals() else 0
            pending_gids = len([gid for gid in global_ids.values() if gid is None]) if 'global_ids' in locals() else 0
            logger.info(f"🎯 Frame {frame_idx}: PERF_FRAME_PROCESS: {frame_time:.1f}ms pending={pending_gids} assigned={assigned_gids}")
            
            frame_idx += 1
        
        # End of main loop - calculate final stats  
        total_time = time.time() - start_time
        final_fps = frame_idx / total_time if total_time > 0 else 0
        
        logger.info("=" * 60)
        logger.info(f"✅ Demo finished!")
        logger.info(f"📁 Output saved to: {config.output}")
        logger.info(f"📊 Processed {frame_idx} frames in {total_time:.1f}s ({final_fps:.1f} FPS)")
        
        if reid_pipeline:
            # Simple demo stats
            logger.info(f"📈 Demo stats:")
            logger.info(f"   - ReID pipeline was active")
            logger.info(f"   - Total frames processed: {frame_idx}")
        else:
            logger.info(f"📈 Demo stats:")
            logger.info(f"   - Tracking only (ReID disabled)")
            logger.info(f"   - Total frames processed: {frame_idx}")
        logger.info("=" * 60)
    
    except KeyboardInterrupt:
        logger.info("Quit requested by user")
    except Exception as e:
        logger.error("Error in demo: %s", e, exc_info=True)
    finally:
        # Cleanup
        if 'reid_pipeline' in locals() and reid_pipeline:
            from reid_system.reid_processor import stop_reid_system
            stop_reid_system("demo", "demo")
        if 'cap' in locals():
            cap.release()
        if 'writer' in locals():
            writer.release()
        cv2.destroyAllWindows()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ReID Demo with YOLOv7 + BoxMOT")
    parser.add_argument("--source", "-s", type=str, 
                       help="Video source: file path, RTSP URL, HTTP URL, or webcam ID (0,1,2...)")
    parser.add_argument("--rtsp", type=str, help="RTSP stream URL (shorthand for --source)")
    parser.add_argument("--webcam", type=int, help="Webcam device ID (shorthand for --source)")
    parser.add_argument("--output", "-o", type=str, help="Output video path")
    parser.add_argument("--yolo-weights", type=str, help="YOLOv7 weights")
    parser.add_argument("--onnx-model", type=str, help="ONNX ReID model")
    parser.add_argument("--tracker", type=str, choices=["botsort", "bytetrack", "ocsort"])
    parser.add_argument("--extraction-interval", type=int, help="ReID extraction interval")
    parser.add_argument("--device", type=str, help="CUDA device")
    parser.add_argument("--detect-class", type=int, help="Detection class (0=person)")
    
    args = parser.parse_args()
    
    config = DemoConfig()
    
    # Override from args
    if args.rtsp:
        config.source = args.rtsp
    elif args.webcam is not None:
        config.source = str(args.webcam)
    elif args.source:
        config.source = args.source
    if args.output:
        config.output = args.output
    if args.yolo_weights:
        config.yolo_weights = args.yolo_weights
    if args.onnx_model:
        config.onnx_model_path = args.onnx_model
    if args.tracker:
        config.tracker_type = args.tracker
    if args.extraction_interval:
        config.extraction_interval = args.extraction_interval
    if args.device:
        config.device = args.device
    if args.detect_class is not None:
        config.detect_class = args.detect_class
    
    print("=" * 60)
    print("ReID Demo Configuration:")
    print(f"  Source: {config.source}")
    print(f"  Output: {config.output}")
    print(f"  YOLO: {config.yolo_weights}")
    print(f"  ReID ONNX: {config.onnx_model_path}")
    print(f"  Tracker: {config.tracker_type} (without built-in ReID)")
    print(f"  Extraction interval: {config.extraction_interval} frames")
    print(f"  Device: {config.device}")
    print(f"  Detect class: {config.detect_class}")
    print("=" * 60)
    
    run_demo(config)


if __name__ == "__main__":
    main()
