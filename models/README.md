# models/

Third-party model weights live here. They are **not** committed to this repo
(see [ATTRIBUTION.md](../ATTRIBUTION.md)); run `python setup_models.py` to fetch
them, or place them manually:

```
models/
├── tracknet_ball/
│   ├── ball_tracker_net.py   # architecture, from yastrebksv/TrackNet
│   └── model_best.pt         # ball detection weights
└── vitpose_base_simple/      # ViT-Pose, auto-downloaded from HuggingFace
```

The court model (`TennisCourtDetector/`) is cloned to the repo root by
`setup_models.py`, not stored here.
