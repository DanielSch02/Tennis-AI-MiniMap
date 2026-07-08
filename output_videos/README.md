# output_videos/

Generated results land here — the pipeline writes, per input clip:

- `combined_<clip>.mp4` — annotated video (keypoints, ball, players,
  skeletons, stroke labels, minimap, live match-stats panel)
- `combined_<clip>.json` — per-frame ball/player/pose data plus every
  detected event (bounces + hits with player/stroke labels)
- `combined_<clip>_frame0.jpg` — single-frame check

Nothing in this folder is committed (it's all generated output).
