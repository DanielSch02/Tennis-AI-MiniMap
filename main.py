"""
Combined pipeline: court keypoints + ball detection + player detection +
player pose (ViT-Pose) + homography + minimap + stroke classification.

Reuses the two existing runners (run_court.py, run_ball.py) and the
cloned TennisCourtDetector repo's CourtReference / get_trans_matrix:

  1. Per frame: detect 14 court keypoints, the ball (TrackNet), and people
     (RF-DETR Medium, COCO class 1 "person" — better recall than YOLO26n on
     this footage: it doesn't drop the far player on frames where YOLO26n
     lost it to motion blur, see the RF-DETR switch note below).
  2. The RENDERED ball track (marker + minimap dot/trail) comes from
     ball_events.smooth_track(): the raw track is cleaned with the same
     misfire/wrong-lock rejection event detection uses, then each flight
     segment is drawn from its gravity-constrained fit — smooth within a
     flight, corners kept sharp at the detected events, occlusion gaps
     filled with a plausible arc. (Event detection itself runs on the raw
     track; smoothing only decides where the dot is drawn.)
  3. get_trans_matrix() gives the reference->image homography from the
     keypoints; its inverse maps the ball's image position into the top-down
     court reference (1665x3506 space from CourtReference).
  4. Player boxes are filtered down to the two players by projecting each
     box's bottom-center (feet) through the same inverse homography and
     keeping it only if it lands within a few meters of the court's playing
     area. The margin is generous (~6m behind each baseline, ~3m outside the
     sidelines) because rallying players roam well off their baseline — an
     earlier, tighter margin (150px in court-reference units, ~1.5m) wrongly
     dropped a player standing in a normal rally position. This is still what
     rejects the chair umpire/ball kids/crowd, since a plain confidence
     threshold can't (a small, distant player can score lower than a
     courtside umpire).
  5. Bounces and racket hits are found by splitting the ball track into
     smooth flight segments (change-point detection over gravity-constrained
     quadratics, residuals in court units via the homography scale), then
     classifying the segment boundaries (hits typed by ViT-Pose wrist
     proximity) and finally enforcing the rally's hit/bounce alternation.
     All of that lives in ball_events.py (pure geometry, no model imports)
     so eval_clips.py can replay it offline from cached detections; see its
     module docstring for the full rationale. NOTE: the homography maps
     the *ground plane*,
     so the minimap position is exact only when the ball is on the ground —
     i.e. at bounces. In-flight positions are a ground-plane approximation
     (shown as a moving dot + trail); bounce markers are the trustworthy points.
     The same caveat applies to player minimap dots (feet = ground contact,
     so those are exact whenever the homography is valid).
  6. ViT-Pose (top-down, 17 COCO keypoints, see run_pose.py) runs on the
     two filtered player boxes. The wrist/arm keypoints feed the stroke
     classifier; the full skeletons are drawn on the output and dumped to the
     JSON as extra per-player data.
  7. Each racket hit found in step 5 is then attributed and classified by
     classify_strokes(): the wrist-ball distance pins down the exact contact
     frame and the hitting player, and the contact point's side relative to
     the player's anatomical shoulder axis says forehand vs backhand (serves /
     smashes are split off first by contact-above-head). See ball_events.py
     for the geometry.
  8. A minimap (top-down court) is rendered in the top-left corner with the
     live ball dot, a fading trail, persistent bounce markers, and the two
     players.
  9. A live match-stats panel (top-right corner): rally shot counter +
     clock, per-player stroke tally, distance covered, and the last shot.
     Players are labeled "Player 1" (near end) / "Player 2" (far end) on
     every overlay, each with an identity color on their bounding box and
     minimap dot. All pure post-processing over the events and court
     coords — see match_stats.py.

Inspired by ArtLabss/tennis-tracking's minimap visualisation.

Outputs (into output_videos/):
  * combined_local.mp4        annotated clip (keypoints + ball + players +
                              skeletons + stroke labels + minimap + stats)
  * combined_local_frame0.jpg single-frame check
  * combined_local.json       per-frame ball in image px and court coords,
                              player boxes/court coords/pose keypoints,
                              homography validity, detected events
                              (bounces + hits with player/stroke labels),
                              and the match-stats summary (shot list,
                              distance covered per player)
"""

import json
import os

import cv2
import numpy as np
import truststore
from rfdetr import RFDETRMedium

COCO_PERSON_CLASS_ID = 1

# Norton AV intercepts HTTPS on this machine with its own root CA; truststore
# verifies against the Windows cert store (which trusts it) instead of
# certifi's public bundle.
truststore.inject_into_ssl()

# Reusing the existing runners: importing them also puts TennisCourtDetector
# and models/tracknet_ball on sys.path and picks the device.
import run_ball as ball
import run_court as court
import run_pose as pose

from court_reference import CourtReference      # noqa: E402 (repo, via sys.path)
from homography import get_trans_matrix         # noqa: E402 (repo, via sys.path)

# Event detection + stroke classification live in ball_events.py (pure
# geometry, no model imports) so eval_clips.py can replay them offline from
# cached detections. Re-exported here for analyze_bounce3d.py and callers.
from ball_events import (                        # noqa: E402, F401
    EVENT_BOUNCE_MARGIN, classify_strokes, detect_ball_events, fit_y,
    player_end, prepare_track, refine_bounce_points, seg_vel_at,
    segment_track, smooth_track, to_court_coords,
)
# Match statistics (rally counter, stroke tally, distance covered) — same
# model-free convention as ball_events.py.
from match_stats import (                        # noqa: E402
    PLAYER_LABELS, compute_match_stats, draw_stats_box, stats_summary,
)

INPUT_VIDEO = "input_videos/input_video_2.mp4"
OUTPUT_DIR = "output_videos"
SECONDS = 3.0
N_FRAMES_OVERRIDE = None      # None = process the whole video

# --- minimap layout -------------------------------------------------------
MINIMAP_HEIGHT = 320          # px on the output frame; width follows aspect
MINIMAP_MARGIN = 20           # offset from the top-left corner
TRAIL_LEN = 12                # frames of fading trail on the minimap

# --- event / stroke rendering ----------------------------------------------
BOUNCE_POPUP_FRAMES = 12   # how long the "bounce" text lingers by the ball
STROKE_POPUP_FRAMES = 14   # how long the stroke label lingers at the contact
STROKE_COLORS = {"forehand": (80, 220, 80), "backhand": (200, 80, 255),
                 "forehand volley": (80, 220, 80),
                 "backhand volley": (200, 80, 255),
                 "serve": (0, 200, 255), "overhead": (0, 200, 255)}
# Per-player identity colors (BGR), used on the bounding boxes, their
# "Player 1"/"Player 2" labels, and the minimap dots. Player 1 = near end,
# Player 2 = far end (see match_stats.PLAYER_LABELS). Both stay clear of
# the ball (yellow) and bounce (red) markers.
PLAYER_COLORS = {"near": (255, 160, 0),    # Player 1: azure
                 "far": (0, 140, 255)}     # Player 2: orange

# --- player detection -------------------------------------------------------
PLAYER_CONF = 0.3
# Court-reference units are ~100px/meter (baseline-to-baseline = 2374 units =
# 23.77m). Margins are generous because rallying players roam well off their
# baseline; they only need to be tight enough to exclude the chair umpire
# (~500 units outside the sidelines) and crowd.
PLAYER_COURT_MARGIN_X = 300   # ~3m outside the sidelines
PLAYER_COURT_MARGIN_Y = 600   # ~6m behind each baseline
PLAYER_ANCHOR_RADIUS = 300    # court units (~3m): a player can't move farther
                              # than this between frames; holds each half's
                              # slot to the tracked player when a courtside
                              # ball boy briefly outscores him (see below)
STATIC_BOX_TOL = 6            # px: person boxes whose center stays within this
STATIC_BOX_FRAC = 0.5         # of one spot for this fraction of ALL frames are
                              # furniture (line judge / sitting ball boy /
                              # umpire) — no player is pixel-static for
                              # hundreds of frames. On clip 2 a static person
                              # at (372,311,426,416)+-2px for all 290 frames
                              # outscored the real far player at frame 0
                              # (conf 0.70 vs 0.68) and the continuity anchor
                              # then held the wrong pick for the whole clip.


def load_player_model():
    # Reuses the same CUDA-probe/CPU-fallback as the other two models (a busy
    # 4GB GPU otherwise throws cudaErrorDevicesUnavailable).
    return RFDETRMedium(device=court.device)


def detect_players(model, frame):
    """Run person detection; return list of (x1, y1, x2, y2, conf) boxes."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    det = model.predict(rgb, threshold=PLAYER_CONF)
    boxes = []
    for (x1, y1, x2, y2), class_id, conf in zip(
            det.xyxy, det.class_id, det.confidence):
        if class_id == COCO_PERSON_CLASS_ID:
            boxes.append((float(x1), float(y1), float(x2), float(y2), float(conf)))
    return boxes


def find_static_boxes(raw_players):
    """Centers of person boxes that sit in one spot for most of the clip
    (see STATIC_BOX_TOL/FRAC): line judges, sitting ball kids, the umpire.
    Returns a list of (cx, cy) centers to exclude from player selection."""
    clusters = []                     # {"cx", "cy", "n"} running means
    for boxes in raw_players:
        for (x1, y1, x2, y2, _conf) in boxes:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            for cl in clusters:
                if abs(cx - cl["cx"]) <= STATIC_BOX_TOL \
                        and abs(cy - cl["cy"]) <= STATIC_BOX_TOL:
                    cl["cx"] += (cx - cl["cx"]) / (cl["n"] + 1)
                    cl["cy"] += (cy - cl["cy"]) / (cl["n"] + 1)
                    cl["n"] += 1
                    break
            else:
                clusters.append({"cx": cx, "cy": cy, "n": 1})
    bar = STATIC_BOX_FRAC * len(raw_players)
    static = [(cl["cx"], cl["cy"]) for cl in clusters if cl["n"] >= bar]
    if static:
        print(f"Static person boxes excluded from player selection: "
              f"{[(round(x), round(y)) for x, y in static]}")
    return static


def filter_court_players(boxes, inv_matrix, court_ref, anchors=None,
                         static_boxes=()):
    """Keep (up to) one box per court half whose feet land on the court.

    Projects each box's bottom-center point through the inverse homography
    and rejects it if it falls outside the court's playing area (plus a
    margin) — this is what separates the two players from the chair umpire,
    ball kids, and crowd, which a confidence threshold alone can't do (a
    small, distant player can score lower than a courtside umpire).

    Selection is PER COURT HALF — exactly one player lives on each side of
    the net — instead of the old global top-2 by confidence: on clip 3 a
    ball boy sitting at the net post (2m outside the sideline, inside the
    margin, steady conf 0.89) outscored the small far player on 236/434
    frames and stole his slot. Within a half, candidates near the previous
    frame's pick (`anchors`, court coords keyed "near"/"far") are preferred
    over raw confidence — the moving player, not the static ball boy, is
    where the player was 40ms ago; with no candidate in reach it falls back
    to confidence (graceful reacquire after a long detection gap).
    Returns a list of dicts with the box, confidence, and court coords.
    """
    if inv_matrix is None:
        return []

    x_lo = court_ref.left_court_line[0][0] - PLAYER_COURT_MARGIN_X
    x_hi = court_ref.right_court_line[0][0] + PLAYER_COURT_MARGIN_X
    y_lo = court_ref.baseline_top[0][1] - PLAYER_COURT_MARGIN_Y
    y_hi = court_ref.baseline_bottom[0][1] + PLAYER_COURT_MARGIN_Y

    candidates = []
    for (x1, y1, x2, y2, conf) in boxes:
        bc = ((x1 + x2) / 2, (y1 + y2) / 2)
        if any(abs(bc[0] - sx) <= STATIC_BOX_TOL
               and abs(bc[1] - sy) <= STATIC_BOX_TOL
               for sx, sy in static_boxes):
            continue                   # furniture (see find_static_boxes)
        feet = ((x1 + x2) / 2, y2)
        cx, cy = to_court_coords(feet, inv_matrix)
        if x_lo <= cx <= x_hi and y_lo <= cy <= y_hi:
            candidates.append({
                "box": (x1, y1, x2, y2), "conf": conf, "court": (cx, cy),
            })

    net_y = (court_ref.baseline_top[0][1] + court_ref.baseline_bottom[0][1]) / 2
    picks = []
    for half in ("near", "far"):
        group = [c for c in candidates
                 if (c["court"][1] > net_y) == (half == "near")]
        if not group:
            continue
        anchor = (anchors or {}).get(half)
        if anchor is not None:
            close = [c for c in group
                     if np.hypot(c["court"][0] - anchor[0],
                                 c["court"][1] - anchor[1])
                     <= PLAYER_ANCHOR_RADIUS]
            if close:
                group = close
        picks.append(max(group, key=lambda c: c["conf"]))
    return picks


def build_minimap_base(court_ref, scale):
    """Render the top-down court reference as a small BGR minimap image."""
    lines = court_ref.build_court_reference()          # uint8, lines > 0
    h, w = lines.shape
    mm = np.zeros((h, w, 3), dtype=np.uint8)
    mm[:] = (70, 47, 20)                               # dark blue court bg
    mm[lines > 0] = (255, 255, 255)
    mm = cv2.resize(mm, (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_AREA)
    return mm


def paste_minimap(frame, mm):
    """Paste the minimap into the top-left corner with a thin border."""
    h, w = mm.shape[:2]
    x0, y0 = MINIMAP_MARGIN, MINIMAP_MARGIN
    frame[y0:y0 + h, x0:x0 + w] = mm
    cv2.rectangle(frame, (x0 - 2, y0 - 2), (x0 + w + 1, y0 + h + 1),
                  (255, 255, 255), 2)
    return frame


def main(input_video=None, stem="combined_local"):
    """Run the pipeline on `input_video` (default: INPUT_VIDEO). Output files
    are named <stem>.mp4 / <stem>_frame0.jpg / <stem>.json."""
    input_video = input_video or INPUT_VIDEO
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    court_model = court.load_model()
    ball_model = ball.load_model()
    player_model = load_player_model()
    pose_model, pose_processor = pose.load_model()
    print(f"Models loaded (court on {court.device}, ball on {ball.device}, "
          f"player on {court.device}, pose on {pose.device})")

    print(f"Input: {input_video} -> {OUTPUT_DIR}/{stem}.*")
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = N_FRAMES_OVERRIDE or int(round(SECONDS * fps))

    frames = []
    while N_FRAMES_OVERRIDE is None or len(frames) < n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    print(f"Read {len(frames)} frames @ {w}x{h} {fps:.0f}fps")

    # ---- pass 1: per-frame court keypoints + raw ball + raw player boxes --
    all_kps, raw_ball, raw_players = [], [(None, None), (None, None)], []
    for i, frame in enumerate(frames):
        all_kps.append(court.detect_keypoints(court_model, frame))
        if i >= 2:
            raw_ball.append(ball.detect_ball(ball_model, frames[i - 2:i + 1], w, h))
        raw_players.append(detect_players(player_model, frame))
        print(f"  detect frame {i + 1}/{len(frames)}", end="\r")
    print()

    # ---- pass 2: homography per frame (image -> court reference) ----------
    court_ref = CourtReference()
    inv_matrices, last_inv = [], None
    for kps in all_kps:
        matrix = get_trans_matrix(kps)          # reference -> image, or None
        if matrix is not None:
            last_inv = np.linalg.inv(matrix)
        inv_matrices.append(last_inv)
    n_hom = sum(1 for m in inv_matrices if m is not None)
    print(f"Homography available on {n_hom}/{len(frames)} frames")

    # ---- pass 3: filter raw player boxes down to the 2 on-court players ----
    # sequential: each frame's picks anchor the next frame's per-half search
    net_y = (court_ref.baseline_top[0][1] + court_ref.baseline_bottom[0][1]) / 2
    static_boxes = find_static_boxes(raw_players)
    players, anchors = [], {}
    for boxes, inv in zip(raw_players, inv_matrices):
        picks = filter_court_players(boxes, inv, court_ref, anchors,
                                     static_boxes)
        for p in picks:
            anchors["near" if p["court"][1] > net_y else "far"] = p["court"]
        players.append(picks)
    n_both = sum(1 for p in players if len(p) == 2)
    print(f"Both players found on {n_both}/{len(frames)} frames")

    # ---- pass 4: ViT-Pose on the filtered player boxes ----------------------
    # Runs only on the (up to 2) on-court players, so this stays cheap: one
    # batched top-down forward pass per frame.
    poses = []
    for i, frame in enumerate(frames):
        poses.append(pose.detect_pose(pose_model, pose_processor, frame,
                                      [p["box"] for p in players[i]]))
        print(f"  pose frame {i + 1}/{len(frames)}", end="\r")
    print()
    n_posed = sum(1 for ps in poses for _ in ps)
    print(f"Pose estimated on {n_posed} player boxes")

    # ---- pass 5: ball events (bounces + hits), stroke classification -------
    events = detect_ball_events(raw_ball, players, poses, inv_matrices, court_ref)
    classify_strokes(events, players, poses, raw_ball, inv_matrices, court_ref)

    # ---- pass 6: rendered ball track (cleaned + piecewise-physics smoothed,
    # event frames as segment boundaries) and its court-plane ground shadow --
    points, visible = smooth_track(raw_ball, inv_matrices, court_ref, events)
    n_raw = sum(visible)
    n_final = sum(1 for p in points if p[0] is not None)
    print(f"Ball: raw {n_raw}/{len(frames)} -> rendered {n_final}/{len(frames)}")

    court_pts = []
    for pt, inv in zip(points, inv_matrices):
        if pt[0] is not None and inv is not None:
            court_pts.append(to_court_coords(pt, inv))
        else:
            court_pts.append((None, None))

    bounces = [e for e in events if e["type"] == "bounce"]
    bounce_frames = [e["frame"] for e in bounces]
    for e in bounces:
        # court position from the sub-frame-refined image point (more accurate
        # than the nearest per-frame detection)
        inv = inv_matrices[e["frame"]]
        e["court"] = to_court_coords(e["image"], inv) if inv is not None else None
    # match stats (rally counter, stroke tally, distance covered)
    stats = compute_match_stats(events, players, court_ref, fps)
    dist = stats["cum_dist"]
    print(f"Distance covered: near {dist['near'][-1]:.0f} m, "
          f"far {dist['far'][-1]:.0f} m")
    for e in events:
        extra = ""
        if e["type"] == "hit":
            extra = (f" -> {e.get('player') or '?'} "
                     f"{e.get('stroke') or 'unclassified'}"
                     f" ({e.get('wrist') or '?'} wrist, "
                     f"contact f{e.get('contact_frame')}, "
                     f"{'confirmed' if e.get('confirmed') else 'unconfirmed'})")
        print(f"  {e['type']:6s} at frame {e['frame']} "
              f"(t={e['t']:.2f}, score={e['score']:.0f}){extra}")

    # ---- pass 7: render -----------------------------------------------------
    scale = MINIMAP_HEIGHT / court_ref.court_total_height
    mm_base = build_minimap_base(court_ref, scale)

    writer = cv2.VideoWriter(
        os.path.join(OUTPUT_DIR, f"{stem}.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    records = []
    for i, frame in enumerate(frames):
        out = frame.copy()

        # court keypoints on the main frame
        out = court.draw(out, all_kps[i])

        # player boxes (per-player identity color) + labels + pose skeletons
        for p in players[i]:
            end = player_end(p, court_ref)
            color = PLAYER_COLORS[end]
            x1, y1, x2, y2 = (int(v) for v in p["box"])
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, PLAYER_LABELS[end], (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(out, PLAYER_LABELS[end], (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        out = pose.draw(out, poses[i])

        # ball marker + short trail on the main frame
        for j in range(max(0, i - 8), i):
            if points[j][0] is not None:
                a = (j - (i - 8)) / 8
                cv2.circle(out, (int(points[j][0]), int(points[j][1])), 2,
                           (0, int(180 * a), int(255 * a)), -1)
        if points[i][0] is not None:
            color = (0, 255, 255) if visible[i] else (0, 165, 255)
            cv2.circle(out, (int(points[i][0]), int(points[i][1])), 6, color, 2)

        # "bounce" pop-up: show it for a short window after each bounce, anchored
        # to the refined impact point.
        for e in bounces:
            if 0 <= i - e["frame"] < BOUNCE_POPUP_FRAMES:
                bx, by = int(e["image"][0]), int(e["image"][1])
                cv2.putText(out, "bounce", (bx + 12, by - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4,
                            cv2.LINE_AA)                       # dark outline
                cv2.putText(out, "bounce", (bx + 12, by - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                            cv2.LINE_AA)                       # red text
                break

        # stroke pop-up: "<player> <stroke>" at the racket contact point
        for e in events:
            if e["type"] != "hit" or not e.get("stroke"):
                continue
            cf = e.get("contact_frame", e["frame"])
            if 0 <= i - cf < STROKE_POPUP_FRAMES:
                # anchor at the refined contact frame's ball position (the
                # kinematic hit's e["image"] can sit a few frames of ball
                # travel away from the racket)
                ax_, ay_ = (points[cf] if points[cf][0] is not None
                            else e["image"])
                hx, hy = int(ax_), int(ay_)
                label = f"{PLAYER_LABELS.get(e.get('player'), '?')} {e['stroke']}"
                color = STROKE_COLORS.get(e["stroke"], (255, 255, 255))
                cv2.putText(out, label, (hx + 12, hy + 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4,
                            cv2.LINE_AA)                       # dark outline
                cv2.putText(out, label, (hx + 12, hy + 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
                            cv2.LINE_AA)
                break

        # minimap: trail, bounces so far, live ball
        mm = mm_base.copy()
        for j in range(max(0, i - TRAIL_LEN), i):
            cx, cy = court_pts[j]
            if cx is not None:
                a = (j - (i - TRAIL_LEN)) / TRAIL_LEN
                cv2.circle(mm, (int(cx * scale), int(cy * scale)), 2,
                           (0, int(180 * a), int(255 * a)), -1)
        for e in bounces:
            if e["frame"] <= i and e["court"] is not None:
                bx, by = e["court"]
                cv2.circle(mm, (int(bx * scale), int(by * scale)), 5,
                           (0, 0, 255), -1)                      # bounce: red
        if court_pts[i][0] is not None:
            cx, cy = court_pts[i]
            cv2.circle(mm, (int(cx * scale), int(cy * scale)), 4,
                       (0, 255, 255), -1)                        # live: yellow

        # players on the minimap (feet position = exact ground-plane point),
        # in their identity colors
        for p in players[i]:
            px, py = p["court"]
            cv2.circle(mm, (int(px * scale), int(py * scale)), 6,
                       PLAYER_COLORS[player_end(p, court_ref)], -1)
        out = paste_minimap(out, mm)

        # live match-stats panel (top-right corner)
        out = draw_stats_box(out, i, stats, fps)

        writer.write(out)
        if i == 0:
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{stem}_frame0.jpg"), out)

        records.append({
            "frame": i,
            "ball_image": [points[i][0], points[i][1]],
            "ball_court": [court_pts[i][0], court_pts[i][1]],
            # raw (pre-interpolation) detection + keypoints so downstream
            # analyses (analyze_bounce3d.py) can rerun without the models
            "ball_raw": [raw_ball[i][0], raw_ball[i][1]],
            "keypoints": [[kx, ky] for kx, ky in all_kps[i]],
            "visible": visible[i],
            "homography": inv_matrices[i] is not None,
            "bounce": i in bounce_frames,
            "players": [
                {"box": list(p["box"]), "conf": p["conf"], "court": list(p["court"])}
                for p in players[i]
            ],
            # ALL person boxes (pre-filter) so player-selection logic can be
            # iterated offline against the stubs without rerunning RF-DETR
            "players_raw": [list(b) for b in raw_players[i]],
            # ViT-Pose keypoints, aligned with "players" (17 COCO points each;
            # ids/skeleton documented in run_pose.py)
            "poses": [
                {"keypoints": [[round(float(x), 1), round(float(y), 1)]
                               for x, y in ps["keypoints"]],
                 "scores": [round(float(s), 3) for s in ps["scores"]]}
                for ps in poses[i]
            ],
        })
    writer.release()

    with open(os.path.join(OUTPUT_DIR, f"{stem}.json"), "w") as f:
        json.dump({
            "fps": fps,
            "bounce_frames": bounce_frames,
            "events": [
                {"type": e["type"], "frame": e["frame"], "t": e["t"],
                 "score": e["score"], "image": list(e["image"]),
                 "court": list(e["court"]) if e.get("court") else None,
                 # stroke fields (hits only; None on bounces)
                 "player": e.get("player"), "stroke": e.get("stroke"),
                 "wrist": e.get("wrist"),
                 "contact_frame": e.get("contact_frame"),
                 "contact_ratio": e.get("contact_ratio"),
                 "confirmed": e.get("confirmed")}
                for e in events
            ],
            "stats": stats_summary(stats),
            "frames": records,
        }, f, indent=2)
    print(f"Done. Outputs in {OUTPUT_DIR}/")


if __name__ == "__main__":
    import sys
    # python main.py [video.mp4] — output files are named
    # combined_<video basename>.* so different clips don't overwrite each other.
    inp = sys.argv[1] if len(sys.argv) > 1 else INPUT_VIDEO
    base = os.path.splitext(os.path.basename(inp))[0]
    main(inp, stem=f"combined_{base}")
