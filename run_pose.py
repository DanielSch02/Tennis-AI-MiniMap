"""
Local, GPU-accelerated 2D human-pose estimation using ViT-Pose
(usyd-community/vitpose-base-simple via HuggingFace transformers).

This is the pose counterpart to run_court.py / run_ball.py.
ViT-Pose is a *top-down* model: it takes the full frame plus person bounding
boxes (here: the RF-DETR player boxes from the combined pipeline) and returns
17 COCO keypoints per box:

     0 nose        1 l_eye    2 r_eye    3 l_ear     4 r_ear
     5 l_shoulder  6 r_shoulder         7 l_elbow    8 r_elbow
     9 l_wrist    10 r_wrist           11 l_hip     12 r_hip
    13 l_knee     14 r_knee            15 l_ankle   16 r_ankle

Left/right are *anatomical* (the player's own left/right), independent of
which way they face the camera — this is what lets the stroke classifier in
main.py tell a forehand from a backhand for both the near
player (back to camera) and the far player (facing it) with the same rule.

Weights are downloaded once from the HF hub and cached in
models/vitpose_base_simple/ so later runs are fully offline.

Standalone demo (python run_pose.py): runs RF-DETR person detection +
pose on the first SECONDS of the input video and writes into output_videos/:
  * pose_local_frame0.jpg   single-frame check
  * pose_local.mp4          annotated clip (all detected persons' skeletons)
"""

import os

import cv2
import numpy as np
import torch
import truststore

# Norton AV intercepts HTTPS with its own root CA; truststore verifies against
# the Windows cert store (which trusts it) instead of certifi's public bundle.
# Must run before the HF hub download.
truststore.inject_into_ssl()

from transformers import AutoProcessor, VitPoseForPoseEstimation  # noqa: E402

HF_MODEL = "usyd-community/vitpose-base-simple"
LOCAL_DIR = os.path.join(os.path.dirname(__file__), "models", "vitpose_base_simple")

INPUT_VIDEO = "input_videos/input_video_1.mp4"
OUTPUT_DIR = "output_videos"
SECONDS = 3.0

# COCO-17 keypoint ids used by downstream consumers.
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12

KEYPOINT_MIN_SCORE = 0.3      # below this a keypoint is treated as missing

SKELETON = [                   # COCO limb pairs for drawing
    (5, 7), (7, 9), (6, 8), (8, 10),          # arms
    (5, 6), (5, 11), (6, 12), (11, 12),       # torso
    (11, 13), (13, 15), (12, 14), (14, 16),   # legs
    (0, 1), (0, 2), (1, 3), (2, 4),           # head
]
# arms drawn brighter: they carry the stroke information
ARM_LIMBS = {(5, 7), (7, 9), (6, 8), (8, 10)}


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
    """Return (model, processor); downloads to models/ on the first run."""
    src = LOCAL_DIR if os.path.isdir(LOCAL_DIR) else HF_MODEL
    processor = AutoProcessor.from_pretrained(src)
    model = VitPoseForPoseEstimation.from_pretrained(src).to(device)
    model.eval()
    if src == HF_MODEL:                     # cache for offline runs
        os.makedirs(LOCAL_DIR, exist_ok=True)
        processor.save_pretrained(LOCAL_DIR)
        model.save_pretrained(LOCAL_DIR)
    return model, processor


def detect_pose(model, processor, frame, boxes):
    """Estimate pose for each (x1, y1, x2, y2) box on a BGR frame.

    Returns a list aligned with `boxes`, each entry a dict with
    "keypoints" (17, 2) float32 in source-frame pixels and "scores" (17,).
    """
    if not boxes:
        return []
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # ViT-Pose's processor takes COCO-format boxes: [x, y, width, height]
    coco_boxes = [[x1, y1, x2 - x1, y2 - y1] for (x1, y1, x2, y2) in boxes]
    inputs = processor(rgb, boxes=[coco_boxes], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_pose_estimation(outputs, boxes=[coco_boxes])[0]
    return [{"keypoints": np.asarray(r["keypoints"].cpu(), dtype=np.float32),
             "scores": np.asarray(r["scores"].cpu(), dtype=np.float32)}
            for r in results]


def kp(pose_entry, idx):
    """(x, y) of keypoint `idx`, or None when below the confidence bar."""
    if pose_entry is None or pose_entry["scores"][idx] < KEYPOINT_MIN_SCORE:
        return None
    return (float(pose_entry["keypoints"][idx][0]),
            float(pose_entry["keypoints"][idx][1]))


def draw(frame, poses):
    """Draw skeletons (arms highlighted) onto the frame in place."""
    for p in poses:
        pts = p["keypoints"]
        ok = p["scores"] >= KEYPOINT_MIN_SCORE
        for a, b in SKELETON:
            if ok[a] and ok[b]:
                color = (0, 255, 0) if (a, b) in ARM_LIMBS else (200, 160, 0)
                cv2.line(frame, (int(pts[a][0]), int(pts[a][1])),
                         (int(pts[b][0]), int(pts[b][1])), color, 2, cv2.LINE_AA)
        for i in range(len(pts)):
            if ok[i]:
                r = 4 if i in (L_WRIST, R_WRIST) else 2
                cv2.circle(frame, (int(pts[i][0]), int(pts[i][1])), r,
                           (0, 0, 255), -1)
    return frame


def main():
    """Standalone demo: RF-DETR persons + ViT-Pose on the first SECONDS."""
    from rfdetr import RFDETRMedium   # only needed for the demo

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model, processor = load_model()
    detector = RFDETRMedium(device=device)
    print(f"Models loaded on {device}")

    cap = cv2.VideoCapture(INPUT_VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(round(SECONDS * fps))

    writer = cv2.VideoWriter(
        os.path.join(OUTPUT_DIR, "pose_local.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        det = detector.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                               threshold=0.3)
        boxes = [tuple(map(float, b))
                 for b, c in zip(det.xyxy, det.class_id) if c == 1]  # person
        poses = detect_pose(model, processor, frame, boxes)
        annotated = draw(frame.copy(), poses)
        writer.write(annotated)
        if i == 0:
            cv2.imwrite(os.path.join(OUTPUT_DIR, "pose_local_frame0.jpg"),
                        annotated)
        print(f"  frame {i + 1}/{n}: {len(boxes)} persons", end="\r")

    cap.release()
    writer.release()
    print(f"\nDone. Outputs in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
