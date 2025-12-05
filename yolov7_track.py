import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List
import cv2
import numpy as np
import torch
import time

from boxmot.tracker_zoo import create_tracker
from boxmot.utils import TRACKER_CONFIGS

# Add YOLOv7 repo to path so its utils can be imported
YOLOV7_ROOT = Path(__file__).parent / "yolov7"
sys.path.insert(0, str(YOLOV7_ROOT))

from yolov7.utils.datasets import letterbox
from yolov7.models.experimental import attempt_load
from yolov7.utils.general import non_max_suppression, scale_coords

@dataclass
class TrackingConfig:
    source      : str   = r"test_video\leloi1.mp4"
    weights     : str   = "best.pt"
    detect_class: int   = 2
    tracker_type: str   = "deepocsort"  # BotSort hoạt động tốt hơn khi tắt ReID
    device      : str   = "0"
    conf_thres  : float = 0.3
    iou_thres   : float = 0.65
    output      : str   = "demo.avi"
    img_size    : int   = 416
    half        : bool  = False
    per_class   : bool  = False
    max_width   : int   = 1280
    max_height  : int   = 720

def visualize_simple(frame, outputs, colors):
    for det in outputs:
        x1, y1, x2, y2, tid, score, *_ = det
        x1, y1, x2, y2, tid = map(int, (x1, y1, x2, y2, tid))
        color = colors[tid % len(colors)]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"ID:{tid}({score:.2f})"
        cv2.putText(frame, text, (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    return frame

def run_tracking(cfg: TrackingConfig) -> None:
    import torch.cuda
    torch.cuda.set_device(0)  # Force sử dụng GPU 0
    selected_device = torch.device("cuda:0")
    print(f"[INFO] Using device: {selected_device}")

    model = attempt_load(cfg.weights, map_location="cpu")
    model.to(selected_device)
    model.eval()
    print(f"[INFO] YOLOv7 model on: {next(model.parameters()).device}")

    # Tắt ReID cho BotSort bằng cách set reid_weights=None
    reid_weights = None if cfg.tracker_type == "botsort" else Path("osnet_x1_0_msmt17.pt")
    
    # List of trackers that support ReID toggle
    trackers_with_reid = ["strongsort", "botsort", "deepocsort", "hybridsort", "boosttrack"]
    
    tracker = create_tracker(
        tracker_type=cfg.tracker_type,
        tracker_config=TRACKER_CONFIGS / f"{cfg.tracker_type}.yaml",
        half=cfg.half,
        per_class=cfg.per_class,
        reid_weights=reid_weights,
        device=selected_device,
        with_reid=True if cfg.tracker_type in trackers_with_reid else None
    )
    if cfg.tracker_type == "botsort":
        print(f"[INFO] Tracker on: {selected_device} (ReID disabled for BotSort)")
    elif cfg.tracker_type in trackers_with_reid:
        print(f"[INFO] Tracker and ReID model on: {selected_device}")
    else:
        print(f"[INFO] Tracker on: {selected_device}")
    try:
        if hasattr(tracker, 'model') and tracker.model is not None and hasattr(tracker.model, 'parameters'):
            print(f"[INFO] ReID model device: {next(tracker.model.parameters()).device}")
    except:
        print("[INFO] ReID model device check skipped (non-standard backend)")

    cap = cv2.VideoCapture(cfg.source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open source: {cfg.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Resize to fit within max dimensions while keeping aspect ratio
    scale = min(cfg.max_width / width, cfg.max_height / height, 1.0)
    output_width = int(width * scale)
    output_height = int(height * scale)

    out = cv2.VideoWriter(cfg.output, cv2.VideoWriter_fourcc(*"XVID"), fps, (output_width, output_height))

    colors: List[tuple] = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128),
        (128, 128, 0), (128, 0, 128), (0, 128, 128),
        (192, 192, 192), (128, 128, 128), (64, 0, 0),
        (0, 64, 0), (0, 0, 64), (255, 165, 0),
        (255, 105, 180), (173, 216, 230),
    ]

    # === FPS & Frame Index
    frame_idx = 0
    prev_time = time.time()
    start_time = prev_time  # tổng thời gian chạy

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        print(f"Frame: {frame_idx}")
        frame_idx += 1
        curr_time = time.time()
        fps_real = 1.0 / (curr_time - prev_time)
        avg_fps = frame_idx / (curr_time - start_time)  # FPS trung bình tính tại thời điểm này
        prev_time = curr_time

        img, ratio, pad = letterbox(frame, new_shape=cfg.img_size, auto=False)
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        img_tensor = torch.from_numpy(img).to(selected_device).float() / 255.0

        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            pred = model(img_tensor, augment=False)[0]
            det = non_max_suppression(pred, conf_thres=cfg.conf_thres, iou_thres=cfg.iou_thres)[0]

        if det is not None and len(det):
            det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], frame.shape, ratio_pad=(ratio, pad)).round()
            det = det[det[:, 5] == 2]  # Filter class (person)
            if len(det):
                outputs = tracker.update(det.cpu().numpy(), frame)
                display_frame = visualize_simple(frame.copy(), outputs, colors)
            else:
                display_frame = frame.copy()
        else:
            display_frame = frame.copy()

        # === Hiển thị FPS hiện tại & FPS trung bình
        info_text = f"Frame: {frame_idx} | FPS: {fps_real:.2f}"
        avg_text = f"Avg FPS: {avg_fps:.2f}"
        cv2.putText(display_frame, info_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(display_frame, avg_text, (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

        # Resize for output
        display_frame = cv2.resize(display_frame, (output_width, output_height), interpolation=cv2.INTER_LINEAR)

        cv2.imshow('Tracking', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        out.write(display_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # In ra FPS trung bình sau khi xử lý toàn bộ video
    total_time = time.time() - start_time
    avg_fps_final = frame_idx / total_time if total_time > 0 else 0
    print(f"\n✅ Tracking finished. Results saved to {cfg.output}")
    print(f"📊 Average FPS (final): {avg_fps_final:.2f}")


if __name__ == "__main__":
    config = TrackingConfig()
    run_tracking(config)
