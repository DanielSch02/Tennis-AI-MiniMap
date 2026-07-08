"""
Local, GPU-accelerated tennis-court keypoint detection using the
TennisCourtDetector model (yastrebksv) — heatmap-regression, 14 court points,
reported median error ~1.8px. Runs offline on the local GPU (no API calls).

We import only the *architecture* (BallTrackerNet) and the postprocessing
helpers from the cloned repo; all orchestration below is our own code.

Model runs at 640x360; detected keypoints are scaled back to the source frame.

Outputs (into output_videos/):
  * court_local_frame0.jpg          single-frame accuracy check
  * court_local_first_3s.mp4        annotated clip
  * court_local_first_3s.json       per-frame keypoints in source-image pixels
"""

import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Import architecture + postprocessing from the cloned repo (reviewed code).
REPO = os.path.join(os.path.dirname(__file__), "TennisCourtDetector")
sys.path.insert(0, REPO)
from tracknet import BallTrackerNet          # noqa: E402
from postprocess import postprocess, refine_kps  # noqa: E402

INPUT_VIDEO = "input_videos/input_video_1.mp4"
OUTPUT_DIR = "output_videos"
WEIGHTS = os.path.join(REPO, "model_tennis_court_det.pt")

MODEL_W, MODEL_H = 640, 360
SECONDS = 3.0
USE_REFINE = True   # snap keypoints to detected line intersections

def pick_device():
    """Prefer CUDA, but fall back to CPU if the GPU is busy/unavailable."""
    if torch.cuda.is_available():
        try:
            torch.zeros(1).cuda()  # probe the device
            return "cuda"
        except Exception as e:
            print(f"GPU unavailable ({e.__class__.__name__}); using CPU")
    return "cpu"


device = pick_device()


def load_model():
    model = BallTrackerNet(out_channels=15).to(device)
    # weights_only=True: safe loader, restores tensors only, refuses to execute
    # any pickled code embedded in the checkpoint.
    state = torch.load(WEIGHTS, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def detect_keypoints(model, frame):
    """Return list of 14 (x, y) in source-frame pixels (None if a point missed)."""
    h, w = frame.shape[:2]
    # postprocess() outputs coords in a 2x-upscaled space relative to the
    # model input (its internal scale=2), i.e. 1280x720. Map that to the frame.
    sx, sy = w / (MODEL_W * 2), h / (MODEL_H * 2)

    img = cv2.resize(frame, (MODEL_W, MODEL_H))
    inp = torch.tensor(np.rollaxis(img.astype(np.float32) / 255.0, 2, 0)).unsqueeze(0)

    with torch.no_grad():
        out = model(inp.to(device))[0]
        pred = torch.sigmoid(out).cpu().numpy()

    points = []
    for k in range(14):
        heatmap = (pred[k] * 255).astype(np.uint8)
        x_pred, y_pred = postprocess(heatmap, low_thresh=170, max_radius=25)
        if x_pred is not None and y_pred is not None:
            # scale from model space (640x360) to source frame
            x_src, y_src = x_pred * sx, y_pred * sy
            if USE_REFINE and k not in (8, 9, 12):
                try:
                    x_src, y_src = refine_kps(frame, int(y_src), int(x_src))
                except cv2.error:
                    pass  # empty/degenerate crop near frame edge -> keep raw point
            points.append((float(x_src), float(y_src)))
        else:
            points.append((None, None))
    return points


def draw(frame, points):
    for i, (x, y) in enumerate(points):
        if x is None:
            continue
        cv2.circle(frame, (int(x), int(y)), 6, (0, 0, 255), -1)
        cv2.putText(frame, str(i), (int(x) + 6, int(y) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
    return frame


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = load_model()
    print(f"Model loaded on {device}")

    cap = cv2.VideoCapture(INPUT_VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(round(SECONDS * fps))

    writer = cv2.VideoWriter(
        os.path.join(OUTPUT_DIR, "court_local_first_3s.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    all_kps = []
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        pts = detect_keypoints(model, frame)
        all_kps.append({"frame": i, "keypoints": pts})
        annotated = draw(frame.copy(), pts)
        writer.write(annotated)
        if i == 0:
            cv2.imwrite(os.path.join(OUTPUT_DIR, "court_local_frame0.jpg"), annotated)
        found = sum(1 for x, _ in pts if x is not None)
        print(f"  frame {i:>3}: {found}/14 keypoints", end="\r")

    cap.release()
    writer.release()
    with open(os.path.join(OUTPUT_DIR, "court_local_first_3s.json"), "w") as f:
        json.dump(all_kps, f, indent=2)
    print(f"\nDone. Outputs in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
