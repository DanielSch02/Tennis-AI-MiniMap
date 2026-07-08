"""
One-time setup: fetch the third-party models this project builds on.

Nothing third-party is committed to this repository (see ATTRIBUTION.md) — this
script pulls each component from its original source into the layout the runners
expect:

    TennisCourtDetector/                    (cloned from GitHub)
        model_tennis_court_det.pt           (court keypoint weights)
    models/tracknet_ball/model_best.pt      (ball TrackNet weights)
    models/vitpose_base_simple/             (auto-downloaded on first run)

Usage:
    python setup_models.py

ViT-Pose and RF-DETR download themselves the first time the pipeline runs (via
HuggingFace / the rfdetr package), so this script only needs to handle the two
components that live as loose files: the TennisCourtDetector repo and the two
TrackNet-style weight files (which are hosted on Google Drive by their authors).

If automatic download fails (e.g. Google Drive quota), the script prints the
exact URL and destination so you can place the file manually.
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

COURT_REPO_URL = "https://github.com/yastrebksv/TennisCourtDetector.git"
COURT_REPO_DIR = os.path.join(ROOT, "TennisCourtDetector")

# Author-hosted weights (Google Drive file IDs from the upstream READMEs).
COURT_WEIGHTS = {
    "gdrive_id": "1f-Co64ehgq4uddcQm1aFBDtbnyZhQvgG",
    "dest": os.path.join(COURT_REPO_DIR, "model_tennis_court_det.pt"),
    "source": "yastrebksv/TennisCourtDetector",
}
# The ball TrackNet model (both the architecture file `ball_tracker_net.py` and
# the `model_best.pt` weights) comes from yastrebksv/TrackNet and is NOT
# committed to this repo. Obtain both from the upstream repo and place them at
# models/tracknet_ball/. Fill in the Google Drive file id for the weights from
# the yastrebksv/TrackNet README to enable auto-download.
BALL_WEIGHTS = {
    "gdrive_id": None,  # <-- set me (see yastrebksv/TrackNet)
    "dest": os.path.join(ROOT, "models", "tracknet_ball", "model_best.pt"),
    "source": "yastrebksv/TrackNet",
}


def clone_court_repo():
    if os.path.isdir(COURT_REPO_DIR):
        print(f"[skip] {COURT_REPO_DIR} already exists")
        return
    print(f"[clone] {COURT_REPO_URL}")
    subprocess.check_call(["git", "clone", COURT_REPO_URL, COURT_REPO_DIR])


def download_gdrive(file_id, dest):
    """Best-effort Google Drive download via gdown; instructive on failure."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        print(f"[skip] {dest} already present")
        return True
    try:
        import gdown  # noqa: WPS433 (optional dep)
    except ImportError:
        print("[info] `pip install gdown` to auto-download weights, or fetch "
              "them manually (see below).")
        return False
    try:
        gdown.download(id=file_id, output=dest, quiet=False)
        return os.path.exists(dest)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] auto-download failed: {e}")
        return False


def fetch_weights(spec, label):
    if spec["gdrive_id"] is None:
        print(f"[manual] {label} weights: set its Google Drive id in "
              f"setup_models.py (see {spec['source']}), or place the file at:\n"
              f"         {spec['dest']}")
        return
    ok = download_gdrive(spec["gdrive_id"], spec["dest"])
    if not ok:
        url = f"https://drive.google.com/uc?id={spec['gdrive_id']}"
        print(f"[manual] Download {label} weights from {spec['source']}:\n"
              f"         {url}\n"
              f"         and save to: {spec['dest']}")


def main():
    print("Setting up third-party models (see ATTRIBUTION.md)...\n")
    clone_court_repo()
    fetch_weights(COURT_WEIGHTS, "court keypoint")
    fetch_weights(BALL_WEIGHTS, "ball TrackNet")
    if not os.path.exists(os.path.join(ROOT, "models", "tracknet_ball",
                                       "ball_tracker_net.py")):
        print("[manual] ball architecture: also copy `ball_tracker_net.py` from "
              "yastrebksv/TrackNet into models/tracknet_ball/ (not "
              "redistributed here).")
    print("\nViT-Pose and RF-DETR download automatically on the first pipeline "
          "run, so no action is needed for those.")
    print("\nDone. Next:  python main.py input_videos/your_clip.mp4")


if __name__ == "__main__":
    sys.exit(main())
