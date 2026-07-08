"""
Match statistics for the combined pipeline — pure post-processing over the
per-frame outputs (classified events + player court coords), no model
imports (numpy + cv2 only, same convention as ball_events.py) so it can be
exercised offline from the stubs.

Computed by compute_match_stats():
  * rally shot counter + duration (serve contact to the last event)
  * per-player stroke tally (serves incl. overheads, forehands, backhands,
    volleys), cumulative as the rally progresses
  * distance covered per player, in meters, integrated from the feet-point
    court coords. Feet are exact ground-plane points, but the raw per-frame
    box jitter of a STANDING player (a few court units/frame) would
    integrate into fake meters over a whole clip, hence the moving-average
    smoothing + per-step deadband below.

A per-shot speed estimate (contact ground shadow -> bounce over elapsed
time) was tried and REMOVED: the homography maps the ground plane, so the
number degrades with the ball's height at contact — groundstrokes read a
plausible 45-155 km/h on the stub clips, but serves/volleys (ball 1.5-3 m
up at contact) read 44-313 km/h. A trustworthy speed needs 3D (cf. the
bounce3d analysis).

The pipeline's internal player keys stay "near"/"far" (court-half based,
like ball_events); PLAYER_LABELS maps them to the display names used on
every overlay. draw_stats_box() renders the live numbers into a translucent
panel in the frame's top-right corner (counterpart of the top-left minimap).
"""

import cv2
import numpy as np

COURT_LENGTH_M = 23.77     # baseline-to-baseline; with the court reference's
                           # 2374-unit baseline span this fixes units/meter

# Display names for the internal court-half player keys (used on the stats
# panel, the bounding-box labels, and the stroke pop-ups).
PLAYER_LABELS = {"near": "Player 1", "far": "Player 2"}

# --- distance integration ----------------------------------------------------
DIST_SMOOTH = 9        # frames of centered moving average on the feet court
                       # coords (~0.36 s at 25fps): kills the 2-5 units/frame
                       # box jitter of a standing player without flattening a
                       # real sprint (which moves 40+ units/frame)
DIST_STEP_MIN = 2.0    # court units/frame (~2 cm, ~0.5 m/s): steps below this
                       # are residual jitter of a standing player, not motion
DIST_STEP_MAX = 300.0  # court units: a jump this large between consecutive
                       # detections is a reacquire after a lost track, not
                       # running (cf. PLAYER_ANCHOR_RADIUS in the pipeline)
DIST_GAP_MAX = 5       # frames: don't integrate motion across longer
                       # detection gaps (position after the gap is reliable,
                       # the path taken during it is not)

# --- overlay layout -----------------------------------------------------------
STATS_WIDTH = 340      # px on the output frame
STATS_MARGIN = 20      # offset from the top-right corner (mirrors the minimap)
STATS_PAD = 12
STATS_LINE = 24        # row height
STATS_ALPHA = 0.55     # panel background opacity
FONT = cv2.FONT_HERSHEY_SIMPLEX

STROKE_BUCKETS = (("serves", ("serve", "overhead")),
                  ("forehands", ("forehand",)),
                  ("backhands", ("backhand",)),
                  ("volleys", ("forehand volley", "backhand volley")))


def _stroke_bucket(stroke):
    for name, members in STROKE_BUCKETS:
        if stroke in members:
            return name
    return None


def _nanmovmean(a, k=DIST_SMOOTH):
    """Centered moving average that ignores NaNs and keeps gaps as gaps."""
    out = np.full_like(a, np.nan)
    half = k // 2
    for i in range(len(a)):
        if np.isnan(a[i]):
            continue
        w = a[max(0, i - half):i + half + 1]
        out[i] = np.nanmean(w)
    return out


def _cum_distance(players, court_ref, n, units_per_m):
    """Per-frame cumulative meters covered, keyed "near"/"far"."""
    net_y = (court_ref.baseline_top[0][1] + court_ref.baseline_bottom[0][1]) / 2
    pos = {"near": np.full((n, 2), np.nan), "far": np.full((n, 2), np.nan)}
    for i, picks in enumerate(players):
        for p in picks:
            end = "near" if p["court"][1] > net_y else "far"
            pos[end][i] = p["court"]

    out = {}
    for end, a in pos.items():
        sx, sy = _nanmovmean(a[:, 0]), _nanmovmean(a[:, 1])
        cum = np.zeros(n)
        total, last = 0.0, None
        for i in range(n):
            if not np.isnan(sx[i]):
                if last is not None and i - last <= DIST_GAP_MAX:
                    step = float(np.hypot(sx[i] - sx[last], sy[i] - sy[last]))
                    if DIST_STEP_MIN * (i - last) <= step <= DIST_STEP_MAX:
                        total += step
                last = i
            cum[i] = total
        out[end] = cum / units_per_m
    return out


def _shot_list(events):
    """One entry per hit, in rally order: contact frame, player, stroke."""
    hits = sorted((e for e in events if e["type"] == "hit"),
                  key=lambda e: e["frame"])
    return [{"frame": (int(e["contact_frame"])
                       if e.get("contact_frame") is not None else e["frame"]),
             "player": e.get("player"), "stroke": e.get("stroke")}
            for e in hits]


def compute_match_stats(events, players, court_ref, fps):
    """Everything draw_stats_box() needs, computed once for the whole clip.

    events      classified events (hits carry player/stroke/contact_frame
                from classify_strokes)
    players     per frame: filtered player picks with "court" feet coords
    """
    n = len(players)
    units_per_m = (court_ref.baseline_bottom[0][1]
                   - court_ref.baseline_top[0][1]) / COURT_LENGTH_M
    shots = _shot_list(events)
    seq = sorted(events, key=lambda e: e["frame"])
    return {
        "shots": shots,
        "cum_dist": _cum_distance(players, court_ref, n, units_per_m),
        "rally_start": shots[0]["frame"] if shots else None,
        "rally_end": seq[-1]["frame"] if seq else None,
        "units_per_m": units_per_m,
    }


def stats_summary(stats):
    """JSON-ready summary (shot list + total distance per player)."""
    return {
        "shots": [dict(s) for s in stats["shots"]],
        "distance_m": {end: round(float(cum[-1]), 1)
                       for end, cum in stats["cum_dist"].items()},
    }


def draw_stats_box(frame, i, stats, fps):
    """Render the live stats panel into the frame's top-right corner and
    return the composited frame (cumulative up to frame i)."""
    h, w = frame.shape[:2]
    x0 = w - STATS_MARGIN - STATS_WIDTH
    y0 = STATS_MARGIN
    n_lines = 8
    box_h = STATS_PAD * 2 + n_lines * STATS_LINE

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + STATS_WIDTH, y0 + box_h),
                  (30, 30, 30), -1)
    frame = cv2.addWeighted(overlay, STATS_ALPHA, frame, 1 - STATS_ALPHA, 0)
    cv2.rectangle(frame, (x0 - 2, y0 - 2),
                  (x0 + STATS_WIDTH + 1, y0 + box_h + 1), (255, 255, 255), 2)

    def put(text, x, y, scale=0.55, color=(255, 255, 255), right=None):
        if right is not None:
            (tw, _), _ = cv2.getTextSize(text, FONT, scale, 1)
            x = right - tw
        cv2.putText(frame, text, (int(x), int(y)), FONT, scale, color, 1,
                    cv2.LINE_AA)

    label_x = x0 + STATS_PAD
    near_r = x0 + STATS_WIDTH - 100      # right edge of the NEAR column
    far_r = x0 + STATS_WIDTH - STATS_PAD  # right edge of the FAR column
    gray = (185, 185, 185)

    def row_y(k):
        return y0 + STATS_PAD + 17 + k * STATS_LINE

    # title: shot counter + rally clock (frozen at the last event)
    played = [s for s in stats["shots"] if s["frame"] <= i]
    dur = 0.0
    if stats["rally_start"] is not None and i >= stats["rally_start"]:
        dur = (min(i, stats["rally_end"]) - stats["rally_start"]) / fps
    put(f"RALLY  {len(played)} shot{'s' if len(played) != 1 else ''}",
        label_x, row_y(0), 0.6)
    put(f"{dur:.1f} s", 0, row_y(0), 0.6, right=far_r)

    # per-player stroke tally
    put(PLAYER_LABELS["near"].upper(), 0, row_y(1), 0.45, gray, right=near_r)
    put(PLAYER_LABELS["far"].upper(), 0, row_y(1), 0.45, gray, right=far_r)
    for k, (name, _members) in enumerate(STROKE_BUCKETS):
        put(name, label_x, row_y(2 + k), 0.5, gray)
        for end, right in (("near", near_r), ("far", far_r)):
            cnt = sum(1 for s in played
                      if s["player"] == end and _stroke_bucket(s["stroke"]) == name)
            put(str(cnt), 0, row_y(2 + k), 0.55, right=right)

    # distance covered (feet ground-plane track, exact)
    put("distance", label_x, row_y(6), 0.5, gray)
    j = min(i, len(stats["cum_dist"]["near"]) - 1)
    for end, right in (("near", near_r), ("far", far_r)):
        put(f"{stats['cum_dist'][end][j]:.0f} m", 0, row_y(6), 0.55, right=right)

    # last shot
    last = played[-1] if played else None
    if last is not None:
        who = PLAYER_LABELS.get(last["player"], "?")
        put(f"last: {who} {last['stroke'] or 'shot'}", label_x, row_y(7), 0.5)
    return frame
