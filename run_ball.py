"""
Local, GPU-accelerated tennis-ball detection using TrackNet
(yastrebksv/TrackNet) — heatmap regression over 3 consecutive frames.

This is the ball-tracking counterpart to run_court.py. Same paradigm as
the court keypoint model (both are the BallTrackerNet encoder-decoder), but the
ball model's contract differs:

  * input : 3 consecutive frames concatenated -> 9 channels, at 360x640, [0,1].
  * output: (1, 256, 360*640) logits; argmax over the class dim -> 360x640
            heatmap; a Hough-circle peak on that heatmap gives the ball (x, y),
            reported in 1280x720 space (postprocess scale=2), then mapped to the
            source frame.

On top of the raw per-frame detections we run a trajectory-cleanup pass:
  * interpolation to fill short gaps where the ball is occluded / missed, and
  * light smoothing to remove single-frame jitter.
This is what makes the track actually usable — raw TrackNet drops the ball on
occluded / motion-blurred frames, and those are cheap to recover downstream.

Because the first valid detection needs 3 frames, frames 0 and 1 have no raw
detection (they are recovered by interpolation if the track starts early).

Outputs (into output_videos/):
  * ball_local_frame0.jpg           single annotated frame (accuracy check)
  * ball_local.mp4                  annotated clip with ball marker + trail
  * ball_local.json                 per-frame ball (x, y) in source pixels,
                                     with a "visible" flag (True = raw detection,
                                     False = interpolated / missing)
"""

import json
import os
import sys

import cv2
import numpy as np
import torch

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "tracknet_ball")
sys.path.insert(0, MODEL_DIR)
from ball_tracker_net import BallTrackerNet  # noqa: E402

INPUT_VIDEO = "input_videos/input_video_1.mp4"
OUTPUT_DIR = "output_videos"
WEIGHTS = os.path.join(MODEL_DIR, "model_best.pt")

MODEL_W, MODEL_H = 640, 360
POSTPROCESS_SCALE = 2          # postprocess reports coords in 1280x720 space
SECONDS = 3.0                  # process only the first N seconds (match court det)

# Trajectory-cleanup knobs.
MAX_GAP = 8                    # interpolate across gaps up to this many frames
                              # (~0.3s at 25fps); longer gaps stay as "missing"
SMOOTH_WINDOW = 3             # moving-average window for jitter removal (odd)
SUBPIX_RADIUS = 3             # heatmap window (px) for the centroid refinement
SUBPIX_MIN_VAL = 32           # heatmap values below this don't weigh in


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
    model = BallTrackerNet(out_channels=256).to(device)
    # weights_only=True: safe loader, restores tensors only, refuses to execute
    # any pickled code embedded in the checkpoint.
    state = torch.load(WEIGHTS, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def postprocess(feature_map, scale=POSTPROCESS_SCALE):
    """Heatmap (360x640, float in [0,1]) -> (x, y) in scaled space, or (None, None).

    Detection follows the upstream TrackNet postprocess (threshold + single
    Hough circle; zero or multiple circles -> None, keeping false positives out
    of the track). The *position*, however, is refined to sub-pixel with an
    intensity-weighted centroid of the unthresholded heatmap around the peak:
    the Hough centre is quantized at model resolution (640x360), which after
    the x3 upscale to a 1920px frame was the dominant term in the ~4px median
    trajectory noise measured by analyze_bounce3d.py.
    """
    fm = (feature_map * 255).reshape((MODEL_H, MODEL_W)).astype(np.uint8)
    _, heatmap = cv2.threshold(fm, 127, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(
        heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1,
        param1=50, param2=2, minRadius=2, maxRadius=7)
    if circles is None or len(circles) != 1:
        return None, None
    x, y = float(circles[0][0][0]), float(circles[0][0][1])

    xi, yi = int(round(x)), int(round(y))
    x0, x1 = max(0, xi - SUBPIX_RADIUS), min(MODEL_W, xi + SUBPIX_RADIUS + 1)
    y0, y1 = max(0, yi - SUBPIX_RADIUS), min(MODEL_H, yi + SUBPIX_RADIUS + 1)
    win = fm[y0:y1, x0:x1].astype(np.float32)
    win[win < SUBPIX_MIN_VAL] = 0.0
    if win.sum() > 0:
        ys_g, xs_g = np.mgrid[y0:y1, x0:x1]
        x = float((win * xs_g).sum() / win.sum())
        y = float((win * ys_g).sum() / win.sum())
    return x * scale, y * scale


def detect_ball(model, frames3, src_w, src_h):
    """frames3 = [prev2, prev, cur] BGR source frames -> (x, y) in source pixels.

    The 3 frames are resized to the model input, concatenated along channels
    (cur, prev, prev2 order per upstream), normalized, and run through the net.
    """
    # scale from postprocess space (1280x720) to the source frame
    sx = src_w / (MODEL_W * POSTPROCESS_SCALE)
    sy = src_h / (MODEL_H * POSTPROCESS_SCALE)

    resized = [cv2.resize(f, (MODEL_W, MODEL_H)) for f in frames3]
    # upstream concatenates current first, then previous, then previous-previous
    stacked = np.concatenate([resized[2], resized[1], resized[0]], axis=2)
    inp = torch.tensor(
        np.rollaxis(stacked.astype(np.float32) / 255.0, 2, 0)).unsqueeze(0)

    with torch.no_grad():
        out = model(inp.to(device))          # (1, 256, 230400)
        heatmap = out.argmax(dim=1).reshape(MODEL_H, MODEL_W)
        heatmap = (heatmap.float() / 255.0).cpu().numpy()

    x, y = postprocess(heatmap)
    if x is None:
        return None, None
    return x * sx, y * sy


def interpolate_and_smooth(track):
    """track: list of (x, y) or (None, None), one per frame.

    Returns (points, visible) where points has no None gaps up to MAX_GAP
    (linearly interpolated) and is lightly smoothed; visible[i] is True only for
    frames that had a real detection. Gaps longer than MAX_GAP, and the head/tail
    before the first / after the last detection, are left as (None, None).
    """
    n = len(track)
    xs = np.array([p[0] if p[0] is not None else np.nan for p in track], float)
    ys = np.array([p[1] if p[1] is not None else np.nan for p in track], float)
    visible = [not np.isnan(xs[i]) for i in range(n)]

    det_idx = np.where(~np.isnan(xs))[0]
    if len(det_idx) >= 2:
        for a, b in zip(det_idx[:-1], det_idx[1:]):
            if 1 < (b - a) <= MAX_GAP:            # short interior gap -> fill it
                t = np.linspace(0, 1, b - a + 1)[1:-1]
                xs[a + 1:b] = xs[a] + t * (xs[b] - xs[a])
                ys[a + 1:b] = ys[a] + t * (ys[b] - ys[a])

    # light moving-average smoothing over the now-dense interior stretches
    if SMOOTH_WINDOW >= 3:
        half = SMOOTH_WINDOW // 2
        sx, sy = xs.copy(), ys.copy()
        for i in range(n):
            lo, hi = max(0, i - half), min(n, i + half + 1)
            wx, wy = xs[lo:hi], ys[lo:hi]
            m = ~np.isnan(wx)
            if m.any():
                sx[i] = wx[m].mean() if not np.isnan(xs[i]) else np.nan
                sy[i] = wy[m].mean() if not np.isnan(ys[i]) else np.nan
        xs, ys = sx, sy

    points = [(None, None) if np.isnan(xs[i]) else (float(xs[i]), float(ys[i]))
              for i in range(n)]
    return points, visible


def draw(frame, pt, visible, trail):
    """Draw the ball marker (solid if detected, hollow if interpolated) + trail."""
    for j, (tx, ty) in enumerate(trail):
        if tx is None:
            continue
        alpha = (j + 1) / len(trail)
        cv2.circle(frame, (int(tx), int(ty)), 2,
                   (0, int(180 * alpha), int(255 * alpha)), -1)
    if pt[0] is not None:
        color = (0, 255, 255) if visible else (0, 165, 255)
        cv2.circle(frame, (int(pt[0]), int(pt[1])), 6, color, 2)
    return frame


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = load_model()
    print(f"Model loaded on {device}")

    cap = cv2.VideoCapture(INPUT_VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Pass 1: read the first SECONDS of frames, detect raw ball position per frame.
    n_frames = int(round(SECONDS * fps))
    frames = []
    while len(frames) < n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    print(f"Read {len(frames)} frames (first {SECONDS:.0f}s) @ {w}x{h} {fps:.0f}fps")

    raw = [(None, None), (None, None)]  # frames 0,1 have no 3-frame window
    for i in range(2, len(frames)):
        x, y = detect_ball(model, frames[i - 2:i + 1], w, h)
        raw.append((x, y))
        det = sum(1 for p in raw if p[0] is not None)
        print(f"  frame {i:>3}/{len(frames) - 1}: detections so far {det}", end="\r")
    print()

    # Pass 2: interpolate short occlusion gaps + smooth.
    points, visible = interpolate_and_smooth(raw)
    n_raw = sum(visible)
    n_final = sum(1 for p in points if p[0] is not None)
    print(f"Raw detections: {n_raw}/{len(frames)}  ->  "
          f"after interpolation: {n_final}/{len(frames)}")

    # Pass 3: render annotated video + JSON.
    writer = cv2.VideoWriter(
        os.path.join(OUTPUT_DIR, "ball_local.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    records = []
    TRAIL = 8
    for i, frame in enumerate(frames):
        trail = points[max(0, i - TRAIL):i]
        annotated = draw(frame.copy(), points[i], visible[i], trail)
        writer.write(annotated)
        if i == 0 or (points[i][0] is not None and not any(r["visible"] for r in records)):
            cv2.imwrite(os.path.join(OUTPUT_DIR, "ball_local_frame0.jpg"), annotated)
        records.append({
            "frame": i,
            "x": points[i][0],
            "y": points[i][1],
            "visible": visible[i],   # True = raw detection, False = interpolated
        })
    writer.release()

    with open(os.path.join(OUTPUT_DIR, "ball_local.json"), "w") as f:
        json.dump(records, f, indent=2)
    print(f"Done. Outputs in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
