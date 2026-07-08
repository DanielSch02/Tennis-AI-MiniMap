# Attribution & Third-Party Components

This project stands on several excellent open-source models. To respect their
licenses and keep this repository small, **none of the third-party code or
model weights are committed here** — `setup_models.py` fetches them at install
time from their original sources. This file records what is used and where it
comes from.

## Original work in this repository (MIT licensed — see LICENSE)

The contribution of this project is the analysis pipeline built *on top of*
the detectors below. These files are entirely original:

| File | What it does |
|------|--------------|
| `ball_events.py` | Bounce / racket-hit detection and forehand-vs-backhand stroke classification from the raw ball track — deterministic geometry (change-point segmentation under a gravity prior, rally-grammar repair, sub-frame bounce refinement). No model imports. |
| `main.py` | Orchestrates the full pipeline: court keypoints → ball → players → pose → homography → minimap → live match-stats overlay. |
| `match_stats.py` | Rally counter, per-player stroke tally, and distance-covered stats from the detected events. |
| `eval_clips.py` | Offline regression harness — replays cached detections (`stubs/`) through `ball_events.py` against hand annotations, no models needed. |
| `analyze_bounce3d.py` | Exploratory 3D bounce-fitting analysis. |
| `run_court.py`, `run_ball.py`, `run_pose.py` | Thin local runners that load the third-party models below and adapt their I/O. Orchestration is original; the model architectures/weights are not (see below). |

## Third-party models (fetched by `setup_models.py`, NOT redistributed here)

| Component | Source | Used for | License |
|-----------|--------|----------|---------|
| **TennisCourtDetector** | [yastrebksv/TennisCourtDetector](https://github.com/yastrebksv/TennisCourtDetector) | 14-point court keypoint detection + `CourtReference` / homography helpers | ⚠️ No license file upstream — cloned, not redistributed. See note below. |
| **TrackNet (tennis ball)** | [yastrebksv/TrackNet](https://github.com/yastrebksv/TrackNet) | Ball detection — both the `BallTrackerNet` architecture file *and* the weights come from upstream and are **not committed here**; place them at `models/tracknet_ball/`. | See upstream repo |
| **ViT-Pose** | [`usyd-community/vitpose-base-simple`](https://huggingface.co/usyd-community/vitpose-base-simple) via 🤗 Transformers | 2D human pose (17 COCO keypoints) for stroke classification | Apache 2.0 (per model card) |
| **RF-DETR** | [`rfdetr`](https://pypi.org/project/rfdetr/) (Roboflow) | Player / person detection | Apache 2.0 |

### Note on TennisCourtDetector's license

At the time of writing, the upstream `TennisCourtDetector` repository has **no
LICENSE file**, which means its code is *all rights reserved* by default. For
that reason this project does **not** vendor or redistribute it — `setup_models.py`
clones it directly from the author's GitHub so that you obtain it from the
original source under whatever terms the author sets. If you intend to use this
project beyond personal experimentation, please check the upstream repository
for licensing and contact its author as needed.

## Inspiration

The minimap / top-down visualisation idea is inspired by
[ArtLabss/tennis-tracking](https://github.com/ArtLabss/tennis-tracking).
