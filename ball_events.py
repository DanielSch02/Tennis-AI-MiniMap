"""
Ball-event detection (bounces + racket hits) and stroke classification for
the combined pipeline — pure geometry over already-computed detections.

Extracted from main.py so it can be exercised offline (see
eval_clips.py) against cached per-frame detections without importing any of
the model stacks (torch / transformers / rfdetr): everything here needs only
numpy + cv2. main.py re-exports these names, so
`import main as rc; rc.prepare_track(...)` keeps working.

Inputs use the pipeline's own structures:
  raw_track     list of (x, y) or (None, None) per frame (RAW TrackNet track)
  players       per frame: list of {"box": (x1,y1,x2,y2), "conf", "court"}
  poses         per frame: list aligned with players, each {"keypoints": (17,2),
                "scores": (17,)} (ViT-Pose COCO-17; see run_pose.py)
  inv_matrices  per frame: image->court 3x3 homography or None
  court_ref     TennisCourtDetector CourtReference instance

--- ball event detection (bounces + racket hits) ---------------------------
Three-layer deterministic pipeline (see detect_ball_events):
  1. SEGMENTATION: the track is split into smooth flight segments by
     change-point detection (DP over piecewise-quadratic fits, residuals
     weighted by the local homography scale so one tolerance holds at any
     court depth). Key physics prior: in image space a flight can only curve
     *downward* (gravity), so the y-fit is refused upward curvature — a
     segment can then never absorb a bounce/hit vertex, which a free
     quadratic happily bends around.
  2. CLASSIFICATION: a hit reverses the ball's direction of travel along the
     court (net court-y displacement of whole segments — short-window tests
     fail because a rising ball's ground shadow initially runs the wrong
     way); within a direction-flip group the most racket-like breakpoint —
     minimum ViT-Pose wrist-ball distance in box heights (box distance as a
     pose-less fallback) — is the hit. Wrist beats box distance because a
     bounce at a lunging player's feet is inside the box but ~a body-height
     from the wrists. Bounces must land on the court (+margin) and be a
     descent arrest: not strongly rising in, not strongly falling out
     (kills the mid-flight deceleration artifacts of perspective). Two
     pose-driven repairs on top:
       * a REJECTED bounce within racket reach is promoted to hit — a
         serve's contact produces no direction flip (toss and serve both
         travel toward the receiver) so layer-2 can't see it, but its
         impulse at the racket can't be anything else (clip 1 f15; without
         this the serve steals the flip group's hit slot from the return
         and the real serve bounce gets deleted as a "double bounce").
       * hits closer than EVENT_HIT_MIN_GAP are contact-chaos duplicates
         (TrackNet flip-flops between ball and racket in the swing blur,
         clip 1 f188-193, spawning impulse breakpoints either side of the
         real hit); keep the most racket-like of the cluster. Two genuine
         hits can never be that close — ball flight between players is
         >=0.5 s even at pro pace.
  3. RALLY GRAMMAR: the rally alternates hit/bounce, enforced both ways.
     (a) Between two consecutive bounces there must be a hit (a true double
     bounce ends the point, so a later hit proves one was missed): recover
     it as the strongest corner at a player's racket; failing that, the
     bounce nearest a racket was itself a mislabeled hit; failing that, the
     weaker-scored bounce is a net cord/artifact and is dropped.
     (b) Between two hits there must be a bounce (or none for a volley).
     The half-volley case is checked FIRST: if the ball met the racket at
     the player's feet, bounce and hit coincide — direct physical evidence
     that beats any interval rescan (on clip 1 the rescan's lowered bar
     found a mid-flight noise corner at f197 and masked the true
     half-volley bounce at f216). Only then are missing bounces recovered
     by re-scanning the interval with a lowered corner threshold — a weak
     skid-bounce corner is unambiguous once the grammar says a bounce must
     be there. (a) runs first: a mislabeled hit inside a hit-pair
     otherwise satisfies (b) and masks a real miss.
Accepted bounces then get SUB-FRAME refinement (refine_bounce_points):
the DP breakpoint is the first frame of the OUTGOING flight, so the stored
detection there is systematically late — the ball has traveled onward for
up to one frame interval and is airborne again, and the ground-plane
homography displaces an airborne ball away from the camera by
~height x cot(camera elevation) (~5-6x the ball's height at the far
baseline). The true impact is where the incoming and outgoing image arcs
meet, at height 0, where the homography is exact: both flights are fitted
near the bounce and the arcs' closest approach in continuous time gives
the refined "t"/"image" (the integer "frame" is untouched — eval_clips.py
matches on it).

Runs on the RAW track (gaps bridged by interpolation, no smoothing — a
moving average would smear exactly the discontinuities needed).

smooth_track() reuses the same cleanup + segmentation to produce the
RENDERING track (main-frame ball marker, minimap dot/trail): per-flight
gravity-constrained fits, corners kept sharp at the detected events.

--- pose + stroke classification --------------------------------------------
ViT-Pose keypoints (anatomical left/right, camera-facing independent) turn
each kinematic "hit" event into an attributed, classified stroke:
  * CONTACT: the frame (within +-STROKE_SEARCH_HALF of the kinematic hit)
    where a player's wrist is closest to the ball is the racket contact;
    distances are normalized by box height so near and far player compete
    fairly. Within STROKE_WRIST_MAX box-heights = confirmed contact
    (arm extension + ~70cm of racket + TrackNet blur at peak racket speed).
  * SERVE/SMASH: on an overhead the hitting arm is fully extended upward,
    so the hitting WRIST rises above the head — no groundstroke does that
    (a high backhand meets a head-high ball with the wrist *below* it,
    e.g. clip 1 frame 91). The ball's own height can't be used: TrackNet
    only reacquires a served ball a few frames after contact, by which
    time it reads at head height. Tested across the whole search window.
    The clip's first hit is labeled "serve", later overheads "overhead".
  * FOREHAND/BACKHAND: which side of the body the contact is on, in the
    player's own frame — contact on the racket-hand side = forehand. The
    side comes from the ball's image-x offset from the mid-shoulder point,
    mapped to body side by the broadcast facing prior (near player = back
    to the camera, so his right = image right; far player mirrored). The
    anatomical shoulder AXIS would measure this directly, but it fails two
    ways on real footage (both tried on clip 1): ViT-Pose mirrors
    left/right on the back-facing near player often enough to flip labels
    (f216's forehand read as backhand), and the projected axis collapses
    to noise when the torso rotates toward the camera mid-swing (13px
    shoulder width on f119). The mid-shoulder POINT is flip-invariant.
    Known miss: a jammed shot right at the body reads on the wrong side
    (clip 1 f120, a squeezed forehand next to the hip). Swing-sweep and
    follow-through-side signals were also tested and are too noisy at
    25fps (wrist blur): 3-6/8 vs 7/8 for the contact-side rule.
"""

import cv2
import numpy as np

# COCO-17 keypoint ids (ViT-Pose output; full table in run_pose.py).
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_WRIST, R_WRIST = 9, 10
KEYPOINT_MIN_SCORE = 0.3      # below this a keypoint is treated as missing

EVENT_OUTLIER_PX = 80     # single-frame TrackNet misfires: a detection this
EVENT_OUTLIER_GAIN = 2.0  # far from the line its two real neighbors agree on
                          # (and > GAIN x the neighbors' own spacing) is
                          # physically impossible — a genuine hit vertex
                          # deviates by at most half the velocity change per
                          # frame (~68 px on clip 1's hardest hit) while
                          # misfires measure 127-660 px (the racket or a
                          # background object). The PX floor scales with the
                          # gap span between the neighbors: near an occlusion
                          # gap a real vertex legitimately sits far off the
                          # chord (clip 3 f281 hit). Rejected BEFORE
                          # segmentation: one spike otherwise shatters the DP
                          # into 4-frame fragments and fabricates dv scores
                          # of 800-2400.
EVENT_JUMP_MAX = 250      # px/frame: no ball moves faster in image space.
                          # Speed alone can't flag every wrong lock (a real
                          # serve descent measures 193 px/f while a wrong
                          # lock was entered at 185), so a jump > 100 px/f
                          # that REVERSES the current velocity direction also
                          # counts as a break: a real fast ball continues on
                          # its way, a lock onto a bystander object does not
                          # (a genuine hit also reverses — that just splits
                          # the run at the contact, harmless because flight
                          # runs are longer than EVENT_RUN_MAX). Used to catch
EVENT_RUN_MAX = 8         # multi-frame WRONG-LOCKS: a run of up to this many
                          # detections whose entry AND exit jumps are both
                          # impossible while its flanks reconnect plausibly
                          # is TrackNet latched onto a second ball / person
                          # (clip 2 f62-66: five frames on a far-court object
                          # 750px away, between two perfectly joinable
                          # near-court flights; pointwise tests can't see it
                          # because the run's frames vouch for each other).
EVENT_SEG_MIN = 3         # min flight-segment length (frames)
EVENT_SEG_MAX = 60        # max segment length considered by the DP
EVENT_SEG_PENALTY = 15000.0  # DP cost of adding a segment (court-units^2 of
                          # residual a breakpoint must save; events on this
                          # footage are stable across 10k-20k)
EVENT_A2_TOL = 0.2        # max upward y-curvature allowed in a flight (px/f^2)
EVENT_MIN_DV = 60         # court-units/frame velocity discontinuity to keep a
                          # breakpoint (real events measure 70-320; ~25-30fps)
EVENT_DIR_MIN_LEN = 5     # min segment length for a reliable travel direction
EVENT_DIR_MIN_DISP = 150  # ...and min |net court-y displacement| (~1.5m): a
                          # ball dropping near-vertically has a noise-sign
                          # direction that would fake hit/bounce flips
EVENT_DIR_WEAK_DISP = 50  # weaker displacement bar used only INSIDE a flip
                          # group to tell the hit from a co-grouped bounce: a
                          # short-hop segment (bounce->racket) is too short to
                          # pass the strict bar but its direction still shows
                          # the ball continuing the SAME way after the bounce
                          # (clip 1 f216: +79 toward the hitter), which proves
                          # that breakpoint is NOT the direction reversal
EVENT_VERTEX_MIN = 60     # court-units/f corner for splitting a hit vertex out
                          # of a DP segment: a far-court hit on a RISING ball
                          # makes a sharp V in image-y (rising->falling), which
                          # has POSITIVE curvature — exactly what the gravity
                          # prior allows — so the DP absorbs it (clip 1 f59,
                          # corner -110). A lob apex is the only innocent
                          # rising->falling flip and is gravity-smooth
                          # (~-28 here); perspective deceleration is strong
                          # (-113 at f35) but never flips the sign of vy.
EVENT_IMPULSE_GAIN = 2.0  # a hit's true contact is where per-frame speed first
EVENT_IMPULSE_MIN = 12    # JUMPS to GAIN x the median of the 3 preceding
                          # frames (+MIN px floor); the DP breakpoint can sit
                          # a few frames late when the outgoing flight is
                          # absorbed into the incoming fit (clip 1 serve: bp
                          # f15, contact f11, speed 13 -> 42 px/f at f11->12).
                          # The bar is LOCAL on purpose: a distant baseline
                          # mistakes the smooth perspective speed-up of a ball
                          # approaching the camera for an impulse (f91 -> f87)
EVENT_GRAMMAR_MIN = 35    # lowered corner bar when grammar demands a bounce
EVENT_BOUNCE_MARGIN = 300  # court units (~3m): bounce-on-court plausibility
EVENT_VY_IN_MIN = -40     # units/f: bounce can't be strongly rising into...
EVENT_VY_OUT_MAX = 60     # ...nor strongly falling out of the contact
EVENT_FIT_HALF = 4        # half-window for corner scores / occlusion checks
EVENT_HIT_BOX_PX = 60     # image px: ball this close to a player box counts
                          # as "at the racket" (matches the half-volley margin)
EVENT_HIT_REACH = 0.5     # wrist-ball distance within racket reach, in player-
                          # box heights (arm + racket ~= 0.75 of standing height
                          # fully stretched; measured contacts on clip 1 run
                          # 0.02-0.30, bounces away from a racket 0.42-1.8)
EVENT_HIT_MIN_GAP = 10    # frames (~0.4 s): hits closer than this are contact-
                          # chaos duplicates of one swing, never two real hits
                          # (ball flight between players is >= 0.5 s)

BOUNCE_REFINE_HALF = 12   # max frames of flight fitted on each side of a
                          # bounce: a long flight isn't exactly quadratic in
                          # image space (perspective), so only the arc near
                          # the impact should shape the local fit
BOUNCE_REFINE_SPAN = (-2.0, 1.0)  # continuous-time search window around the
                          # breakpoint: the true impact precedes the DP
                          # breakpoint (= first frame of the outgoing
                          # segment) by up to ~1 frame, plus slack for a
                          # breakpoint that itself sits a frame off
BOUNCE_REFINE_MAX_GAP = 25.0   # px: the two arcs must (nearly) meet at t* —
                          # farther apart means at least one fit is garbage
                          # (a clean crossing measures a few px)
BOUNCE_REFINE_MAX_SHIFT = 80.0  # px: the refined point may not move farther
                          # than this from the breakpoint detection (a real
                          # correction is at most ~1 frame of ball travel)

STROKE_SEARCH_HALF = 4     # frames around a kinematic hit to search for contact
STROKE_WRIST_MAX = 0.9     # confirmed-contact bar, in player-box heights
STROKE_OVERHEAD_MARGIN = 0.2  # hitting wrist this many box heights above the
                              # head -> serve/smash. A fully extended overhead
                              # measures ~0.2+ (clip 1 serve); a whipped
                              # groundstroke follow-through peaks at ~0.12
                              # (clip 2 f259, a forehand) — 0.1 was too low.
                              # (The rally's FIRST hit is labeled serve by
                              # grammar regardless, so this margin only
                              # gates mid-rally smash labels.)
PLAYER_HANDEDNESS = {"near": "right", "far": "right"}  # assumed; edit if known


def kp(pose_entry, idx):
    """(x, y) of keypoint `idx`, or None when below the confidence bar."""
    if pose_entry is None or pose_entry["scores"][idx] < KEYPOINT_MIN_SCORE:
        return None
    return (float(pose_entry["keypoints"][idx][0]),
            float(pose_entry["keypoints"][idx][1]))


def to_court_coords(pt, inv_matrix):
    """Image-pixel (x, y) -> court-reference coords via the inverse homography."""
    src = np.array([[pt]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, inv_matrix)
    return float(dst[0, 0, 0]), float(dst[0, 0, 1])


def prepare_track(raw_track, inv_matrices, court_ref):
    """Bridge the raw track's gaps and compute per-frame ground-shadow court
    coords + homography scale. Shared by detect_ball_events() and
    analyze_bounce3d.py so both analyses run on exactly the same arrays.

    Returns a dict of per-frame arrays (fx/fy bridged image track, real mask,
    cx/cy ground-shadow court coords, scale, inv_fill, lo/hi span), or None
    when the track / homography are too sparse to analyse.
    """
    n = len(raw_track)
    xs = np.array([p[0] if p[0] is not None else np.nan for p in raw_track], float)
    ys = np.array([p[1] if p[1] is not None else np.nan for p in raw_track], float)

    # ---- track cleanup 1: multi-frame wrong-locks (see EVENT_JUMP_MAX) -----
    # A run is a wrong lock when its flanks reconnect plausibly WITHOUT it,
    # at least one of its boundary jumps is impossible, and its points sit
    # far off the flanks' chord. One impossible boundary suffices: a lock
    # onto an object near the ball's path can be ENTERED with a legal jump
    # and only exposed on exit (clip 3 f149-152: entry 185 px/f, exit 434).
    # The chord-deviation test is what keeps real contact-chaos runs (which
    # sit ON the trajectory) safe.
    def speed_between(a, b):
        return np.hypot(xs[b] - xs[a], ys[b] - ys[a]) / (b - a)

    def chord_dev(r, p, q):
        """Mean distance of run r's points from the p->q line (by time)."""
        devs = []
        for t in r:
            f = (t - p) / (q - p)
            devs.append(np.hypot(xs[t] - (xs[p] + f * (xs[q] - xs[p])),
                                 ys[t] - (ys[p] + f * (ys[q] - ys[p]))))
        return float(np.mean(devs))

    def hard_jump(a, b, run):
        """True when a->b cannot be ball flight: impossible speed, or a fast
        direction reversal against the run's recent motion."""
        sp = speed_between(a, b)
        if sp > EVENT_JUMP_MAX:
            return True
        if sp > 100 and run and len(run) >= 2:
            ra, rb = run[max(0, len(run) - 4)], run[-1]
            if rb > ra:
                dot = ((xs[b] - xs[a]) * (xs[rb] - xs[ra])
                       + (ys[b] - ys[a]) * (ys[rb] - ys[ra]))
                return dot < 0
        return False

    for _ in range(2):
        idx = [int(t) for t in np.where(~np.isnan(xs))[0]]
        runs = []
        for t in idx:
            if runs and t - runs[-1][-1] <= 4 \
                    and not hard_jump(runs[-1][-1], t, runs[-1]):
                runs[-1].append(t)
            else:
                runs.append([t])
        dropped = False
        for r0, r, r1 in zip(runs[:-2], runs[1:-1], runs[2:]):
            p, q = r0[-1], r1[0]
            reconnect = speed_between(p, q)
            if len(r) <= EVENT_RUN_MAX and reconnect <= EVENT_JUMP_MAX \
                    and (hard_jump(p, r[0], r0) or hard_jump(r[-1], q, r)) \
                    and chord_dev(r, p, q) > max(150.0, 3.0 * reconnect):
                for t in r:
                    xs[t] = ys[t] = np.nan
                dropped = True
        if not dropped:
            break

    # ---- track cleanup 2: pointwise misfires (see EVENT_OUTLIER_PX) --------
    # Neighbors are the nearest REAL detections within 3 frames on each side;
    # the tolerance widens with the gap span (a real vertex next to a gap
    # sits legitimately far off the chord). Two passes so a double spike
    # whose first frame vouches for the second still falls.
    def nearest_real(t, step):
        for k in range(1, 4):
            u = t + step * k
            if 0 <= u < n and not np.isnan(xs[u]):
                return u
        return None

    for _ in range(2):
        bad = []
        for t in range(1, n - 1):
            if np.isnan(xs[t]):
                continue
            p, q = nearest_real(t, -1), nearest_real(t, +1)
            if p is None or q is None:
                continue
            f = (t - p) / (q - p)
            mx = xs[p] + f * (xs[q] - xs[p])
            my = ys[p] + f * (ys[q] - ys[p])
            dev = np.hypot(xs[t] - mx, ys[t] - my)
            nb = np.hypot(xs[q] - xs[p], ys[q] - ys[p])
            if dev > max(EVENT_OUTLIER_PX * (q - p - 1),
                         EVENT_OUTLIER_GAIN * nb):
                bad.append(t)
        if not bad:
            break
        for t in bad:
            xs[t] = ys[t] = np.nan

    valid = np.where(~np.isnan(ys))[0]
    if len(valid) < 3 * EVENT_SEG_MIN:
        return None
    lo, hi = int(valid[0]), int(valid[-1])
    real = ~np.isnan(xs)

    def bridge(a):
        b = a.copy()
        b[:lo] = a[lo]
        b[hi + 1:] = a[hi]
        nan_mask = np.isnan(b)
        b[nan_mask] = np.interp(np.where(nan_mask)[0], valid, a[valid])
        return b

    fx, fy = bridge(xs), bridge(ys)

    # Per-frame ground shadow (court coords) and homography scale (court units
    # per vertical image px). The scale is evaluated no higher than the far
    # baseline's image row: above it (ball high in the air) the ground-plane
    # scale diverges toward the horizon and would inflate every measure.
    cx = np.full(n, np.nan)
    cy = np.full(n, np.nan)
    scale = np.full(n, np.nan)
    inv_fill = list(inv_matrices)
    for i in range(1, n):                      # forward-fill (head may be None)
        if inv_fill[i] is None:
            inv_fill[i] = inv_fill[i - 1]
    for i in range(n - 2, -1, -1):             # backward-fill the leading gap
        if inv_fill[i] is None:
            inv_fill[i] = inv_fill[i + 1]
    if inv_fill[lo] is None:
        return None
    mid_top = np.array(court_ref.baseline_top, float).mean(axis=0)
    for i in range(lo, hi + 1):
        inv = inv_fill[i]
        far_row = cv2.perspectiveTransform(
            np.array([[mid_top]], np.float32), np.linalg.inv(inv))[0, 0, 1]
        yc = max(fy[i], float(far_row))
        p = cv2.perspectiveTransform(
            np.array([[(fx[i], fy[i]), (fx[i], yc), (fx[i], yc + 1.0)]],
                     np.float32), inv)[0]
        cx[i], cy[i] = float(p[0][0]), float(p[0][1])
        scale[i] = float(np.hypot(*(p[2] - p[1])))

    return {"n": n, "lo": lo, "hi": hi, "real": real, "fx": fx, "fy": fy,
            "cx": cx, "cy": cy, "scale": scale, "inv_fill": inv_fill}


def fit_y(fy, a, b):
    """Quadratic y(t) fit constrained to gravity-consistent curvature
    (flight can only curve downward); linear fallback otherwise."""
    t = np.arange(a, b + 1)
    if len(t) >= 5:
        p = np.polyfit(t, fy[a:b + 1], 2)
        if p[0] >= -EVENT_A2_TOL:
            return p
    return np.polyfit(t, fy[a:b + 1], 1)


def seg_vel_at(fx, fy, a, b, t0):
    """Velocity at t0 from the segment [a, b]'s fit derivatives."""
    t = np.arange(a, b + 1)
    deg = 2 if len(t) >= 6 else 1
    px = np.polyder(np.polyfit(t, fx[a:b + 1], deg))
    py = np.polyder(fit_y(fy, a, b) if deg == 2 else np.polyfit(t, fy[a:b + 1], 1))
    return float(np.polyval(px, t0)), float(np.polyval(py, t0))


def segment_track(fx, fy, scale, lo, hi):
    """Layer-1 DP change-point segmentation of the bridged track into smooth
    flight segments. Returns sorted breakpoints, each the first frame of the
    later segment."""
    def seg_cost(a, b):
        t = np.arange(a, b + 1)
        if len(t) < 5:
            return 0.0
        rx = fx[a:b + 1] - np.polyval(np.polyfit(t, fx[a:b + 1], 2), t)
        ry = fy[a:b + 1] - np.polyval(fit_y(fy, a, b), t)
        return float(np.sum((rx ** 2 + ry ** 2) * scale[a:b + 1] ** 2))

    n_a = hi - lo + 1
    best = np.full(n_a + 1, np.inf)
    best[0] = 0.0
    prev = np.zeros(n_a + 1, int)
    for j in range(EVENT_SEG_MIN, n_a + 1):
        for a in range(max(0, j - EVENT_SEG_MAX), j - EVENT_SEG_MIN + 1):
            if np.isfinite(best[a]):
                c = best[a] + seg_cost(lo + a, lo + j - 1) + EVENT_SEG_PENALTY
                if c < best[j]:
                    best[j], prev[j] = c, a
    bps, j = [], n_a
    while j > 0:
        a = prev[j]
        if a > 0:
            bps.append(lo + a)                 # first frame of the later segment
        j = a
    bps.sort()
    return bps


SMOOTH_MAX_GAP = 8   # frames (~0.3s): render the fitted arc through occlusion
                     # gaps up to this long (same bar as run_ball's
                     # MAX_GAP); farther from any real detection the ball is
                     # genuinely gone (out of frame / long occlusion) and the
                     # rendered dot stays hidden rather than invented


def smooth_track(raw_track, inv_matrices, court_ref, events=None):
    """Piecewise-physics smoothed ball track for RENDERING (the main-frame
    marker and the minimap dot/trail). Event detection keeps using the raw
    track; this only decides where the dot is drawn.

    Replaces the old moving-average smoothing, which had the two visible
    failure modes the raw pipeline showed: TrackNet misfires survived it and
    yanked the minimap dot around, and the 3-frame window rounded off the
    real corners at bounces/hits. Instead:
      1. clean the raw track with prepare_track() — the same wrong-lock and
         misfire rejection event detection runs on, so a detection the event
         layer would ignore can no longer move the rendered dot;
      2. split it into smooth flight segments (segment_track DP), adding the
         detected events' frames as extra breakpoints so segment boundaries
         sit exactly on the hits/bounces;
      3. draw each segment's fit (quadratic x, gravity-constrained y) instead
         of the raw points: sub-pixel smooth within a flight, corners at the
         events stay sharp because no fit spans across them.
    Occlusion gaps inside a flight are filled by that flight's fit — a
    plausible arc rather than a straight chord — except stretches longer
    than SMOOTH_MAX_GAP frames, which stay (None, None).

    Returns (points, visible) in the same shape run_ball's
    interpolate_and_smooth() produced: per-frame (x, y) or (None, None),
    plus a flag list (True = a real detection survived cleanup there).
    """
    n = len(raw_track)
    prep = prepare_track(raw_track, inv_matrices, court_ref)
    if prep is None:                    # too sparse to fit: raw points as-is
        return ([(p[0], p[1]) for p in raw_track],
                [p[0] is not None for p in raw_track])
    lo, hi = prep["lo"], prep["hi"]
    fx, fy, real, scale = prep["fx"], prep["fy"], prep["real"], prep["scale"]

    bps = set(segment_track(fx, fy, scale, lo, hi))
    for e in events or []:
        for f in (e["frame"], e.get("contact_frame")):
            if f is not None and lo < f <= hi:
                bps.add(int(f))
    bounds = [lo] + sorted(bps) + [hi + 1]

    sx, sy = fx.copy(), fy.copy()
    for a, b in zip(bounds[:-1], bounds[1:]):      # segment frames [a, b-1]
        t = np.arange(a, b)
        if len(t) < 3:
            continue          # too short to fit: keep the bridged points
        sx[a:b] = np.polyval(np.polyfit(t, fx[a:b], 2 if len(t) >= 5 else 1), t)
        sy[a:b] = np.polyval(fit_y(fy, a, b - 1) if len(t) >= 5
                             else np.polyfit(t, fy[a:b], 1), t)

    # hide interpolated stretches with no real detection within SMOOTH_MAX_GAP
    ok = np.zeros(n, bool)
    real_idx = np.where(real)[0]
    ok[real_idx] = True
    for a, b in zip(real_idx[:-1], real_idx[1:]):
        if b - a - 1 <= SMOOTH_MAX_GAP:
            ok[a + 1:b] = True

    points = [(float(sx[i]), float(sy[i])) if ok[i] else (None, None)
              for i in range(n)]
    return points, [bool(r) for r in real]


def refine_bounce_points(events, prep, dbg=lambda *a: None):
    """Sub-frame refinement of each bounce's time + image point by
    intersecting the incoming and outgoing flight arcs (in place).

    The DP breakpoint `b` is the first frame of the OUTGOING segment, so the
    per-frame detection stored for the bounce is systematically LATE: the
    ball has already traveled onward for up to one frame interval (40ms at
    25fps) AND is airborne again — and the ground-plane homography displaces
    an airborne ball away from the camera by ~height x cot(camera elevation),
    which at the far baseline (~10 deg elevation on this footage) is ~5-6x
    the ball's height. Both biases push a deep far-court bounce beyond the
    baseline on the minimap ("just in" renders as out).

    At the true impact the ball IS on the ground — and that instant is where
    the incoming and outgoing image arcs meet. So: fit both flights over the
    frames nearest the bounce (quadratic x, gravity-constrained y — the same
    model segmentation used), find the continuous time t* in
    [b + BOUNCE_REFINE_SPAN] where the two arcs come closest, and take their
    meeting point. The fits span 3+ frames each, so the ~4px single-frame
    detection noise is largely averaged out as a bonus.

    Refines "t" and "image"; "frame" stays the integer breakpoint
    (eval_clips.py matches on it, smooth_track uses it as a segment
    boundary). Falls back silently — keeping the detection point — when
    either flight is too short or too interpolated to fit, or the arcs never
    come close (tangent skid, garbage fit): refinement can only replace a
    point with a better-conditioned one, never invent or move an event.
    """
    fx, fy, real = prep["fx"], prep["fy"], prep["real"]
    lo, hi = prep["lo"], prep["hi"]
    ev_frames = sorted(e["frame"] for e in events)

    def local_fit(a, b):
        """x(t), y(t) polynomial fits over frames [a, b]."""
        t = np.arange(a, b + 1)
        deg = 2 if len(t) >= 5 else 1
        px = np.polyfit(t, fx[a:b + 1], deg)
        py = fit_y(fy, a, b) if deg == 2 else np.polyfit(t, fy[a:b + 1], 1)
        return px, py

    for e in events:
        if e["type"] != "bounce":
            continue
        b = e["frame"]
        # flights bounded by the neighboring events (a fit must not span
        # across another hit/bounce corner) and BOUNCE_REFINE_HALF
        prev_ev = max((f for f in ev_frames if f < b), default=lo - 1)
        next_ev = min((f for f in ev_frames if f > b), default=hi + 1)
        a0 = max(lo, prev_ev + 1, b - BOUNCE_REFINE_HALF)
        c1 = min(hi, next_ev - 1, b + BOUNCE_REFINE_HALF)
        # each side needs >= 3 mostly-real frames: an interpolated bridge
        # fits the chord, not the flight
        if b - a0 < 3 or c1 - b + 1 < 3 \
                or real[a0:b].sum() < 3 or real[b:c1 + 1].sum() < 3:
            dbg(f"bounce refine {b}: skipped (flights [{a0},{b - 1}] / "
                f"[{b},{c1}] too short or interpolated)")
            continue
        pxi, pyi = local_fit(a0, b - 1)          # incoming flight
        pxo, pyo = local_fit(b, c1)              # outgoing flight
        ts = np.arange(b + BOUNCE_REFINE_SPAN[0],
                       b + BOUNCE_REFINE_SPAN[1] + 1e-9, 0.02)
        gap = np.hypot(np.polyval(pxi, ts) - np.polyval(pxo, ts),
                       np.polyval(pyi, ts) - np.polyval(pyo, ts))
        k = int(np.argmin(gap))
        t_star = float(ts[k])
        x_star = float((np.polyval(pxi, t_star) + np.polyval(pxo, t_star)) / 2)
        y_star = float((np.polyval(pyi, t_star) + np.polyval(pyo, t_star)) / 2)
        shift = float(np.hypot(x_star - e["image"][0],
                               y_star - e["image"][1]))
        if gap[k] > BOUNCE_REFINE_MAX_GAP or shift > BOUNCE_REFINE_MAX_SHIFT:
            dbg(f"bounce refine {b}: rejected (arc gap {gap[k]:.0f}px, "
                f"shift {shift:.0f}px)")
            continue
        dbg(f"bounce refine {b}: t*={t_star:.2f}, moved {shift:.1f}px "
            f"(arc gap {gap[k]:.1f}px)")
        e["t"] = t_star
        e["image"] = (x_star, y_star)


def detect_ball_events(raw_track, players, poses, inv_matrices, court_ref,
                       debug=False):
    """Find bounce and racket-hit events on the raw ball track.

    Three layers (see the module docstring for the rationale):
      1. DP change-point segmentation into smooth flight segments
         (gravity-constrained piecewise quadratics, court-unit residuals),
      2. classification of the breakpoints (segment travel-direction flips =
         hits, typed by ViT-Pose wrist proximity; descent-arrest + on-court
         checks for bounces; racket-reach promotion for flip-less serve
         contacts; contact-chaos dedup),
      3. rally-grammar repair (a bounce must exist between two hits —
         half-volleys resolved first; recover weak skid bounces the
         segmentation under-rates).

    NaN stretches are bridged by linear interpolation purely for the analysis;
    no smoothing is applied (it would smear the discontinuities). Returns a
    list of {"frame", "t", "image", "type", "score"} dicts, in frame order.
    debug=True traces each stage's decisions (used by eval_clips.py -v).
    """
    def dbg(*a):
        if debug:
            print("[events]", *a)
    prep = prepare_track(raw_track, inv_matrices, court_ref)
    if prep is None:
        return []
    n, lo, hi = prep["n"], prep["lo"], prep["hi"]
    fx, fy, real = prep["fx"], prep["fy"], prep["real"]
    cx, cy, scale = prep["cx"], prep["cy"], prep["scale"]

    # per-frame y-corner (in court units): slope before minus slope after.
    # Positive = descent arrest (bounce-like), negative = a rising ball turned
    # downward (hit-like). Used by the vertex split below and layer 3.
    kf = EVENT_FIT_HALF
    corner = np.full(n, np.nan)
    vy_b = np.full(n, np.nan)              # y slope into / out of each frame
    vy_a = np.full(n, np.nan)
    for i in range(max(kf, lo), min(n - kf, hi + 1)):
        vy_b[i] = np.polyfit(np.arange(i - kf, i + 1), fy[i - kf:i + 1], 1)[0]
        vy_a[i] = np.polyfit(np.arange(i, i + kf + 1), fy[i:i + kf + 1], 1)[0]
        corner[i] = (vy_b[i] - vy_a[i]) * scale[i]

    # ---- layer 1: DP change-point segmentation ----------------------------
    bps = segment_track(fx, fy, scale, lo, hi)

    # layer 1b: split hit vertices the DP absorbed. A far-court hit on a
    # rising ball makes a sharp V in image-y (rising -> falling). That V has
    # positive curvature, which the gravity prior ALLOWS, so a whole flight
    # segment can swallow it (clip 1: segment [50,73] hid the hit at f59).
    # The signature is unambiguous: vy flips sign rising->falling AND the
    # corner is far too sharp for gravity (see EVENT_VERTEX_MIN). One vertex
    # per segment (a flight can contain at most one hit).
    vertex_bps = []
    for a, b in zip([lo] + bps, bps + [hi + 1]):
        cand = [i for i in range(a + EVENT_SEG_MIN, b - EVENT_SEG_MIN)
                if not np.isnan(corner[i]) and vy_b[i] < 0 < vy_a[i]
                and corner[i] <= -EVENT_VERTEX_MIN]
        if cand:
            v = min(cand, key=lambda i: corner[i])
            vertex_bps.append(v)
            dbg(f"vertex split: hit V at {v} inside segment [{a},{b - 1}] "
                f"(corner={corner[v]:.0f})")
    bps = sorted(set(bps) | set(vertex_bps))

    # breakpoint velocities from the segment fits' derivatives: smooth gravity
    # curvature then yields dv ~ 0 (pruned), a real impulse a large dv.
    # Vertex breakpoints are exempt from the dv gate: they already passed an
    # equivalent-strength corner gate, and the quadratic fits smooth their V
    # into a dv just under the bar (clip 1 f60: corner -118 but dv 55).
    bounds = [lo] + bps + [hi + 1]
    kept = []
    for k in range(1, len(bounds) - 1):
        b = bounds[k]
        vbx, vby = seg_vel_at(fx, fy, bounds[k - 1], b - 1, b)
        vax, vay = seg_vel_at(fx, fy, b, bounds[k + 1] - 1, b)
        dv = float(np.hypot(vax - vbx, vay - vby) * scale[b])
        if dv >= EVENT_MIN_DV or b in vertex_bps:
            kept.append((b, dv))

    # occlusion gaps: a breakpoint whose preceding frames are interpolated has
    # a fabricated velocity (gap exit); keep only the gap-entry breakpoint.
    merged = []
    for b, dv in kept:
        if real[max(lo, b - EVENT_FIT_HALF):b].sum() < 2:
            continue
        if merged and real[merged[-1][0]:b + 1].sum() <= 1:
            continue
        merged.append((b, dv))

    # ---- layer 2: classification -------------------------------------------
    x_lo = court_ref.left_court_line[0][0] - EVENT_BOUNCE_MARGIN
    x_hi = court_ref.right_court_line[0][0] + EVENT_BOUNCE_MARGIN
    y_lo = court_ref.baseline_top[0][1] - EVENT_BOUNCE_MARGIN
    y_hi = court_ref.baseline_bottom[0][1] + EVENT_BOUNCE_MARGIN

    def on_court(i):
        return x_lo <= cx[i] <= x_hi and y_lo <= cy[i] <= y_hi

    def box_dist(i):
        best_d = np.inf
        for p in players[i]:
            x1, y1, x2, y2 = p["box"]
            best_d = min(best_d, np.hypot(max(x1 - fx[i], 0, fx[i] - x2),
                                          max(y1 - fy[i], 0, fy[i] - y2)))
        return best_d

    def wrist_ratio(i):
        """Min wrist-ball distance near frame i, in player-box heights (inf
        when no pose). A +-2 frame window rides out contact-frame chaos
        (TrackNet flip-flops between ball and racket in the swing blur)."""
        best = np.inf
        for w in range(max(lo, i - 2), min(hi, i + 2) + 1):
            for pl, ps in zip(players[w], poses[w]):
                box_h = pl["box"][3] - pl["box"][1]
                if box_h <= 0:
                    continue
                for widx in (L_WRIST, R_WRIST):
                    pt = kp(ps, widx)
                    if pt is not None:
                        best = min(best, np.hypot(pt[0] - fx[w],
                                                  pt[1] - fy[w]) / box_h)
        return best

    bps2 = [b for b, _ in merged]
    dvs = {b: dv for b, dv in merged}
    ratios = {b: wrist_ratio(b) for b in bps2}
    bounds = [lo] + bps2 + [hi + 1]

    def seg_dir_at(a, b, min_disp, min_len=EVENT_DIR_MIN_LEN):
        disp = cy[b - 1] - cy[a]
        return (np.sign(disp) if b - a >= min_len
                and abs(disp) >= min_disp else None)

    seg_dir = [seg_dir_at(a, b, EVENT_DIR_MIN_DISP)
               for a, b in zip(bounds[:-1], bounds[1:])]
    # weaker-bar directions, used only INSIDE a flip group (see the constants
    # block): short segments there (a short hop between bounce and racket)
    # carry real direction evidence the strict bar throws away. A 4-frame
    # segment still counts when its displacement is LARGE (clip 2 [38,41],
    # disp -305: the return flight right after the reversal — strong enough
    # that noise can't fake its sign).
    seg_dir_w = [seg_dir_at(a, b, EVENT_DIR_WEAK_DISP)
                 or seg_dir_at(a, b, EVENT_DIR_MIN_DISP, min_len=4)
                 for a, b in zip(bounds[:-1], bounds[1:])]

    dbg("breakpoints kept:",
        [(b, round(dv), round(ratios[b], 2)) for b, dv in merged])

    types = ["bounce"] * len(bps2)
    rel = [k for k, s in enumerate(seg_dir) if s is not None]
    for r0, r1 in zip(rel[:-1], rel[1:]):
        if seg_dir[r0] != seg_dir[r1]:         # travel direction flipped
            group = list(range(r0, r1))
            # Direction evidence beats wrist proximity (at a short-hop the
            # wrist is closest to the BOUNCE, clip 1 f216 vs the hit f223):
            # a breakpoint whose flanking directions agree is provably not
            # the reversal; one whose flanking directions flip provably is.
            # Wrist-ball distance ranks whatever direction can't decide.
            def dir_class(k):                  # bp bps2[k] sits between
                lw, rw = seg_dir_w[k], seg_dir_w[k + 1]   # segments k, k+1
                if lw is not None and rw is not None:
                    return "hit" if lw != rw else "nonhit"
                return "unknown"
            proven = [k for k in group if dir_class(k) == "hit"]
            open_k = [k for k in group if dir_class(k) == "unknown"]
            pool = proven or open_k or group
            hit_k = min(pool,
                        key=lambda k: (ratios[bps2[k]], box_dist(bps2[k])))
            types[hit_k] = "hit"
            dbg(f"flip group bps {[bps2[k] for k in group]} "
                f"(dir: {[dir_class(k) for k in group]}) -> hit {bps2[hit_k]}")

    def corner_near(i):
        """Strongest descent-arrest corner within +-2 of frame i (the DP
        breakpoint can sit a frame or two off the corner's peak)."""
        w = corner[max(lo, i - 2):min(hi, i + 2) + 1]
        return float(np.nanmax(w)) if np.isfinite(w).any() else np.nan

    # racket-reach promotion below is for the SERVE only (its contact has no
    # direction flip: toss and serve both travel toward the receiver), so it
    # only applies before the first flip-typed hit — mid-rally, every real
    # hit flips the travel direction, and an unrestricted promotion turns
    # any ball passing a player's wrist into a phantom hit whose "no bounce
    # right after a hit" shadow then deletes the REAL bounce (clip 2 f269).
    first_flip = min((bps2[k] for k, t in enumerate(types) if t == "hit"),
                     default=n)

    events = []
    for k, (b, t) in enumerate(zip(bps2, types)):
        src = "flip" if t == "hit" else "seg"
        if t == "bounce":
            _, vby = seg_vel_at(fx, fy, bounds[k], b - 1, b)
            _, vay = seg_vel_at(fx, fy, b, bounds[k + 2] - 1, b)
            # a bounce must also SHOW a descent arrest: a breakpoint whose
            # local corner is negative is a fit-edge artifact, not a bounce
            # (clip 3 f251: dv 124 but corner -94, mid-descent)
            if not on_court(b) or vby * scale[b] < EVENT_VY_IN_MIN \
                    or vay * scale[b] > EVENT_VY_OUT_MAX \
                    or not (corner_near(b) > 0):
                # off court / not a descent arrest. Within racket reach that
                # makes it a HIT the direction test can't see (a serve has no
                # travel flip: toss and serve both move toward the receiver).
                if ratios[b] <= EVENT_HIT_REACH and b < first_flip:
                    dbg(f"bounce {b} rejected (vy_in={vby * scale[b]:.0f} "
                        f"vy_out={vay * scale[b]:.0f} on_court={on_court(b)},"
                        f" corner={corner_near(b):.0f})"
                        f" -> promoted to hit (wr={ratios[b]:.2f})")
                    t, src = "hit", "promo"
                else:
                    dbg(f"bounce {b} rejected (vy_in={vby * scale[b]:.0f} "
                        f"vy_out={vay * scale[b]:.0f} on_court={on_court(b)},"
                        f" corner={corner_near(b):.0f},"
                        f" wr={ratios[b]:.2f}) -> dropped")
                    continue
        events.append({"frame": int(b), "t": float(b),
                       "image": (float(fx[b]), float(fy[b])),
                       "type": t, "score": float(dvs[b]), "src": src})

    # contact-chaos dedup: impulse breakpoints within a racket's reach a few
    # frames either side of a real hit are duplicates of one swing — keep the
    # most racket-like of each cluster.
    hit_seq = sorted((e for e in events if e["type"] == "hit"),
                     key=lambda e: e["frame"])
    k = 0
    while k + 1 < len(hit_seq):
        e1, e2 = hit_seq[k], hit_seq[k + 1]
        if e2["frame"] - e1["frame"] <= EVENT_HIT_MIN_GAP:
            # direction evidence outranks wrist proximity: a flip-typed hit
            # is the rally's actual reversal, a promoted one only means "an
            # impulse near a racket" — and the wrist ratio lies exactly when
            # the player's detection box flickered away at contact (clip 3
            # f202, real hit, box missing -> wr 2.65, vs the promoted f209
            # artifact at wr 0.42 seven frames later).
            worse = max((e1, e2), key=lambda e: (e.get("src") != "flip",
                                                 ratios.get(e["frame"], np.inf)))
            dbg(f"hit dedup: {e1['frame']} ({e1.get('src')}) vs "
                f"{e2['frame']} ({e2.get('src')}) -> drop {worse['frame']}")
            events.remove(worse)
            hit_seq.remove(worse)
        else:
            k += 1

    # gap-exit hit relocation: when a hit's breakpoint sits at the ENTRY of
    # an occlusion gap (the gap-exit kink is suppressed by the occlusion
    # rules, so bounce and hit hiding behind the gap collapse onto one
    # gap-entry breakpoint), the true contact is where the ball leaves the
    # player, not where the track went dark — relocate to the last frame at
    # the racket (clip 2: bp f61 at gap entry, ball visibly AT the player
    # f67-70 before departing; GT contact f70).
    for e in events:
        if e["type"] != "hit":
            continue
        b = e["frame"]
        if real[b + 1:b + 4].sum() > 1:
            continue                    # real flight follows: not a gap entry
        # the target must be a DWELL — 2+ REAL detections lingering slow at
        # the racket. Real only: a linear gap bridge sliding past a player's
        # box fakes a perfect dwell (clip 3 f287-289, all interpolated). A
        # lone slow at-box frame is normal far-court flight past a player
        # (12-26 px/f there), and after a real contact the outgoing ball
        # departs fast (clip 1 f121-127, 85+ px/f) — only contact chaos
        # makes real detections linger.
        # bd <= 10, not the usual 60: contact chaos means TrackNet reading
        # the racket/ball ON the player (clip 2 f67-69, bd 0); a slow flyby
        # PAST a player reads 20-50px off the box (clip 3 f207-208).
        dwell = [w for w in range(b + 1, min(hi, b + 10) + 1)
                 if real[w] and box_dist(w) <= 10
                 and np.hypot(fx[w] - fx[w - 1], fy[w] - fy[w - 1]) <= 40]
        if len(dwell) >= 2:
            best = dwell[-1]
            dbg(f"gap-exit relocation: hit {b} -> {best} (dwell at racket)")
            e["frame"], e["t"] = int(best), float(best)
            e["image"] = (float(fx[best]), float(fy[best]))

    # a bounce a few frames AFTER a hit is contact chaos, not physics: even
    # a smash needs ~10 frames to reach the court from the racket, while a
    # groundstroke's contact V (descending ball turned back up) makes a
    # bounce-like POSITIVE corner that can pass every bounce check
    # (clip 2 f136, five frames after the f131 contact; real serve-to-bounce
    # gaps measure 10-12 frames on all three clips).
    hit_frames_now = sorted(e["frame"] for e in events if e["type"] == "hit")
    for e in [e for e in events if e["type"] == "bounce"]:
        if any(0 < e["frame"] - h <= 8 for h in hit_frames_now):
            dbg(f"drop bounce {e['frame']}: within 8 frames after a hit")
            events.remove(e)

    # bounce dedup: two bounces within a few frames and no hit between are
    # one physical impact split by track noise (clip 2 b273+b277 around the
    # true bounce at 275) — a real double bounce is never that fast (>=0.3s
    # even on a failed get). Keep the stronger descent-arrest corner.
    bounce_seq = sorted((e for e in events if e["type"] == "bounce"),
                        key=lambda e: e["frame"])
    k = 0
    while k + 1 < len(bounce_seq):
        e1, e2 = bounce_seq[k], bounce_seq[k + 1]
        if e2["frame"] - e1["frame"] <= 5 and not any(
                e["type"] == "hit" and e1["frame"] < e["frame"] < e2["frame"]
                for e in events):
            worse = min((e1, e2), key=lambda e: (
                v if np.isfinite(v := corner_near(e["frame"])) else -np.inf))
            dbg(f"bounce dedup: {e1['frame']} vs {e2['frame']} -> "
                f"drop {worse['frame']}")
            events.remove(worse)
            bounce_seq.remove(worse)
        else:
            k += 1

    # ---- layer 3: rally-grammar repair (corner[] computed above) -----------
    # grammar (c): the rally STARTS with the serve, so the first event must
    # be a hit. A near-player serve defeats every other serve detector: the
    # toss falls to the racket ON court (its ground shadow, unlike a far
    # serve's, lands mid-court), so the contact passes all bounce checks and
    # would then be deleted by grammar (a) as a double bounce (clip 2 f20).
    # Racket-reach evidence at the event confirms the retype.
    seq0 = sorted(events, key=lambda e: e["frame"])
    if seq0 and seq0[0]["type"] == "bounce" \
            and any(e["type"] == "hit" for e in events):
        b0 = seq0[0]["frame"]
        if ratios.get(b0, wrist_ratio(b0)) <= EVENT_HIT_REACH \
                or box_dist(b0) <= EVENT_HIT_BOX_PX:
            dbg(f"grammar(c): first event b{b0} within racket reach "
                f"-> retype as serve hit")
            seq0[0]["type"] = "hit"

    def corner_cluster(h):
        """Frames belonging to the hit's own corner (its shoulders would
        otherwise fake a bounce right next to the hit)."""
        c, i = {h}, h - 1
        while i > lo and corner[i] >= EVENT_GRAMMAR_MIN:
            c.add(i)
            i -= 1
        i = h + 1
        while i < hi and corner[i] >= EVENT_GRAMMAR_MIN:
            c.add(i)
            i += 1
        return c

    # grammar (a): consecutive bounces with a later hit imply a missed hit.
    # Resolve one violation per pass until none remain (each action below
    # either separates the pair or removes one of its members).
    changed = True
    while changed:
        changed = False
        seq = sorted(events, key=lambda e: e["frame"])
        for e1, e2 in zip(seq[:-1], seq[1:]):
            if not (e1["type"] == "bounce" and e2["type"] == "bounce"
                    and any(e["type"] == "hit" and e["frame"] > e2["frame"]
                            for e in events)):
                continue
            excl = corner_cluster(e1["frame"]) | corner_cluster(e2["frame"])
            cand = [i for i in range(e1["frame"] + 1, e2["frame"])
                    if i not in excl and not np.isnan(corner[i])
                    and box_dist(i) <= EVENT_HIT_BOX_PX
                    and real[max(0, i - 1):i + 2].any()]
            p = max(cand, key=lambda i: corner[i]) if cand else None
            near = min((e1, e2), key=lambda e: box_dist(e["frame"]))
            if p is not None and corner[p] >= EVENT_GRAMMAR_MIN:
                dbg(f"grammar(a) b{e1['frame']}/b{e2['frame']}: "
                    f"insert hit {p} (corner={corner[p]:.0f})")
                events.append({"frame": int(p), "t": float(p),
                               "image": (float(fx[p]), float(fy[p])),
                               "type": "hit", "score": float(corner[p])})
            elif box_dist(near["frame"]) <= EVENT_HIT_BOX_PX:
                dbg(f"grammar(a) b{e1['frame']}/b{e2['frame']}: "
                    f"retype b{near['frame']} as hit")
                near["type"] = "hit"       # the "bounce" was at the racket
            else:                          # net cord / artifact
                # weaker CORNER, not weaker dv: the corner is the bounce's
                # physical signature, while dv can be fabricated by residual
                # track garbage (clip 3 b230: dv 225 from two off-track
                # frames, corner -16; the real b222 had dv 85, corner 179)
                gone = min((e1, e2), key=lambda e: (
                    v if np.isfinite(v := corner_near(e["frame"])) else -np.inf))
                dbg(f"grammar(a) b{e1['frame']}/b{e2['frame']}: "
                    f"drop b{gone['frame']} as artifact "
                    f"(corners {corner_near(e1['frame']):.0f} vs "
                    f"{corner_near(e2['frame']):.0f})")
                events.remove(gone)
            changed = True
            break

    # grammar (b): a bounce must exist between two hits
    hits = sorted((e for e in events if e["type"] == "hit"),
                  key=lambda e: e["frame"])
    for h1, h2 in zip(hits[:-1], hits[1:]):
        if any(e["type"] == "bounce" and h1["frame"] < e["frame"] < h2["frame"]
               for e in events):
            continue
        # half-volley first: the ball bounced within a few frames of the
        # racket contact, so bounce and hit share one kinematic breakpoint and
        # layer 2 typed it (correctly) as the hit. The bounce is the strongest
        # on-court descent-arrest corner just before the contact — direct
        # physical evidence, so it's checked before the weak-corner rescan
        # below, whose lowered bar can latch onto a mid-flight noise corner
        # and mask the real bounce (clip 1 f197 vs the true bounce f216).
        # The window is one corner half-window + 2 (~6 frames): any farther
        # from the racket and it's a normal bounce the rescan should place.
        # Both real half-volleys measure corner 80-150 here, comfortably
        # above the EVENT_GRAMMAR_MIN=35 bar. (An earlier version demanded
        # the ball in the lower 30% of the player's box instead — that broke
        # on a crouching hitter with the ball a body-width to his side,
        # where box fractions say nothing about height above ground.)
        # The hit's own corner cluster is excluded: a VOLLEY's contact makes
        # a strong corner right at the racket, which would otherwise be
        # promoted to a phantom bounce here (clip 2 f164/f221) — the
        # bounce-less hit pair is exactly what a volley looks like. A true
        # half-volley's bounce corner sits just OUTSIDE the hit's cluster
        # (clip 1: bounce corner at f216, hit cluster f220-223).
        h = h2["frame"]
        excl_h = corner_cluster(h)
        cand_hv = [i for i in range(max(lo, h - EVENT_FIT_HALF - 2), h + 1)
                   if i not in excl_h and not np.isnan(corner[i])
                   and on_court(i) and corner[i] >= EVENT_GRAMMAR_MIN]
        if cand_hv:
            v = max(cand_hv, key=lambda i: corner[i])
            dbg(f"grammar(b) h{h1['frame']}/h{h2['frame']}: half-volley "
                f"bounce {v} (corner={corner[v]:.0f})")
            events.append({"frame": int(v), "t": float(v),
                           "image": (float(fx[v]), float(fy[v])),
                           "type": "bounce", "score": float(corner[v])})
            continue
        excl = corner_cluster(h1["frame"]) | corner_cluster(h2["frame"])
        # + 9: same physics as the drop above — the first ~8 frames after
        # h1's contact can only hold h1's own bounce-like corner, never a
        # real bounce (would otherwise resurrect exactly the artifact the
        # drop removed, clip 2 f136)
        cand = [i for i in range(h1["frame"] + 9, h2["frame"])
                if i not in excl and not np.isnan(corner[i])
                and real[max(0, i - 1):i + 2].any() and on_court(i)]
        p = max(cand, key=lambda i: corner[i]) if cand else None
        if p is not None and corner[p] >= EVENT_GRAMMAR_MIN:
            dbg(f"grammar(b) h{h1['frame']}/h{h2['frame']}: rescan "
                f"bounce {p} (corner={corner[p]:.0f})")
            events.append({"frame": int(p), "t": float(p),
                           "image": (float(fx[p]), float(fy[p])),
                           "type": "bounce", "score": float(corner[p])})

    # ---- impulse-onset refinement of hit frames -----------------------------
    # The DP breakpoint can sit a few frames after the true contact when the
    # outgoing flight's first frames get absorbed into the incoming fit (the
    # serve is the worst case: slow toss, then 3-4x the speed — clip 1 bp f15
    # vs contact f11). The contact is where the per-frame speed first jumps to
    # EVENT_IMPULSE_GAIN x the incoming baseline. Only fires on that
    # slow-in/fast-out pattern; symmetric-speed hits keep their breakpoint.
    speed = np.hypot(np.diff(fx), np.diff(fy))         # speed[t] = t -> t+1
    for e in events:
        if e["type"] != "hit":
            continue
        b = e["frame"]
        for t in range(max(lo + 3, b - 5), min(b, len(speed))):
            if box_dist(t) > 2 * EVENT_HIT_BOX_PX:
                continue     # a contact can't be 120px+ from every player —
                             # don't let a perspective speed ramp mid-flight
                             # pull the hit away from the racket (clip 2 f56)
            base = float(np.median(speed[t - 3:t]))    # local: see constants
            bar = max(EVENT_IMPULSE_GAIN * base, base + EVENT_IMPULSE_MIN)
            if speed[t] >= bar:
                dbg(f"impulse refine: hit {b} -> {t} "
                    f"(speed {speed[t]:.0f} vs local base {base:.0f})")
                e["frame"], e["t"] = int(t), float(t)
                e["image"] = (float(fx[t]), float(fy[t]))
                break

    events.sort(key=lambda e: e["frame"])

    # sub-frame bounce refinement: replace each bounce's frame-late,
    # single-detection point with the incoming/outgoing arcs' meeting point
    # (= the moment the ball is actually on the ground; see the docstring)
    refine_bounce_points(events, prep, dbg)

    return events


def player_end(p, court_ref):
    """"near" (bottom of the frame) or "far" from the player's court-y."""
    net_y = (court_ref.baseline_top[0][1] + court_ref.baseline_bottom[0][1]) / 2
    return "near" if p["court"][1] > net_y else "far"


def classify_strokes(events, players, poses, raw_track, inv_matrices, court_ref):
    """Attribute each kinematic hit to a player and classify the stroke.

    Mutates the hit events in place, adding:
      player        "near" / "far" (None if no player found near the ball)
      stroke        "forehand" / "backhand" / "serve" / "overhead" / None
      wrist         "left" / "right" — the hitting hand (nearest to the ball)
      contact_frame frame where wrist-ball distance is minimal (the kinematic
                    hit frame is a velocity breakpoint, which on a smeared
                    TrackNet track can sit a frame or two off the true contact)
      contact_ratio wrist-ball distance at contact, in box heights
      confirmed     contact_ratio <= STROKE_WRIST_MAX (pose corroborates the
                    kinematic hit)

    Geometry (see the module docstring): contact frame + hitting player by
    minimum box-height-normalized wrist-ball distance in a +-STROKE_SEARCH_HALF
    window; overheads split off by contact-above-head; else forehand/backhand
    by the side of the contact along the anatomical shoulder axis, with a
    nearest-wrist fallback in the midline ambiguity band.
    """
    prep = prepare_track(raw_track, inv_matrices, court_ref)
    if prep is None:
        return
    fx, fy, lo, hi = prep["fx"], prep["fy"], prep["lo"], prep["hi"]

    hits = [e for e in events if e["type"] == "hit"]
    for e in hits:
        f = e["frame"]
        best = None                            # (ratio, frame, player_idx, wrist)
        for w in range(max(lo, f - STROKE_SEARCH_HALF),
                       min(hi, f + STROKE_SEARCH_HALF) + 1):
            for j, (pl, ps) in enumerate(zip(players[w], poses[w])):
                box_h = pl["box"][3] - pl["box"][1]
                if box_h <= 0:
                    continue
                for widx, wname in ((L_WRIST, "left"),
                                    (R_WRIST, "right")):
                    pt = kp(ps, widx)
                    if pt is None:
                        continue
                    ratio = np.hypot(pt[0] - fx[w], pt[1] - fy[w]) / box_h
                    if best is None or ratio < best[0]:
                        best = (ratio, w, j, wname)

        e.update({"player": None, "stroke": None, "wrist": None,
                  "contact_frame": f, "contact_ratio": None, "confirmed": False})
        if best is None:
            # no usable pose near the hit: attribute by nearest box only
            if players[f]:
                e["player"] = player_end(
                    min(players[f], key=lambda p: np.hypot(
                        (p["box"][0] + p["box"][2]) / 2 - fx[f],
                        (p["box"][1] + p["box"][3]) / 2 - fy[f])), court_ref)
            continue

        ratio, w, j, wname = best
        pl, ps = players[w][j], poses[w][j]
        end = player_end(pl, court_ref)
        e.update({"player": end, "wrist": wname, "contact_frame": int(w),
                  "contact_ratio": float(ratio),
                  "confirmed": bool(ratio <= STROKE_WRIST_MAX)})

        bx = fx[w]
        hand = PLAYER_HANDEDNESS.get(end, "right")

        # serve / smash: hitting wrist above the head anywhere in the window
        # (see the module docstring); the most-elevated wrist is the hitter —
        # this also corrects the contact-search wrist, which on serves grabs
        # the still-raised toss arm because the tracked ball lags the contact.
        overhead = None                        # (elevation in box heights, wrist)
        for w2 in range(max(lo, f - STROKE_SEARCH_HALF),
                        min(hi, f + STROKE_SEARCH_HALF) + 1):
            for pl2, ps2 in zip(players[w2], poses[w2]):
                if player_end(pl2, court_ref) != end:
                    continue
                bh2 = pl2["box"][3] - pl2["box"][1]
                heads = [pt[1] for i in (0, 1, 2, 3, 4)
                         if (pt := kp(ps2, i)) is not None]  # nose/eyes/ears
                if bh2 <= 0 or not heads:
                    continue
                for widx, wnm in ((L_WRIST, "left"),
                                  (R_WRIST, "right")):
                    pt = kp(ps2, widx)
                    if pt is None:
                        continue
                    elev = (min(heads) - pt[1]) / bh2
                    # the BALL must be above the head too — at the CONTACT
                    # frame, not at w2: a whipped follow-through raises the
                    # wrist over the head on a plain groundstroke while the
                    # just-hit ball is also legitimately high in the later
                    # window frames (clip 2 f257 forehand). On a real smash
                    # the ball is above the head at contact itself.
                    if elev >= STROKE_OVERHEAD_MARGIN and fy[w] <= min(heads) \
                            and (overhead is None or elev > overhead[0]):
                        overhead = (elev, wnm)
        first_hit = e is min(hits, key=lambda h: h["frame"])
        if overhead is not None:
            e["wrist"] = overhead[1]
            e["stroke"] = "serve" if first_hit else "overhead"
            continue
        if first_hit:
            # a rally clip's first hit is the serve even when the wrist-above-
            # head evidence is unavailable — the server's detection box can
            # flicker away exactly around contact (clip 3 f17-19), leaving no
            # pose to test. (A clip that starts mid-rally would mislabel its
            # first hit; broadcast point clips start at the serve.)
            e["stroke"] = "serve"
            continue

        # forehand / backhand: contact side vs the mid-shoulder point, mapped
        # to the player's own left/right by the broadcast facing prior (see
        # the module docstring for why not the anatomical shoulder axis).
        # The side is read from the ball's last INCOMING frames (kinematic
        # hit frame and the 3 before it), not from the wrist-min contact
        # frame: in the swing blur TrackNet flip-flops onto the racket, which
        # sits on whatever side the swing is passing through (clip 1 f190
        # read the racket at x=1136, the wrong side of a backhand contact,
        # while the true incoming ball ran 1231-1249). The approach frames
        # are the clean part of the track and the approach side IS the
        # stroke side.
        def center_x_at(w2):
            for pl2, ps2 in zip(players[w2], poses[w2]):
                if player_end(pl2, court_ref) != end:
                    continue
                ls, rs = kp(ps2, L_SHOULDER), kp(ps2, R_SHOULDER)
                if ls is not None and rs is not None:
                    return (ls[0] + rs[0]) / 2
                return (pl2["box"][0] + pl2["box"][2]) / 2
            return None

        offs = []
        for w2 in range(max(lo, f - 3), f + 1):
            c = center_x_at(w2)
            if c is not None:
                offs.append(fx[w2] - c)
        if not offs:                            # no pose/box: contact frame
            ls, rs = kp(ps, L_SHOULDER), kp(ps, R_SHOULDER)
            c = ((ls[0] + rs[0]) / 2 if ls is not None and rs is not None
                 else (pl["box"][0] + pl["box"][2]) / 2)
            offs = [bx - c]
        right_off = float(np.median(offs)) * (1 if end == "near" else -1)
        side = "right" if right_off > 0 else "left"
        e["stroke"] = "forehand" if side == hand else "backhand"

    # volleys are grammar-level information, not pose geometry: a hit with no
    # bounce since the previous hit took the ball out of the air. (A
    # half-volley HAS its bounce, a few frames before contact, so it
    # correctly stays a plain groundstroke label.)
    seq = sorted(events, key=lambda e: e["frame"])
    hit_seq = [e for e in seq if e["type"] == "hit"]
    for h1, h2 in zip(hit_seq[:-1], hit_seq[1:]):
        if h2.get("stroke") in ("forehand", "backhand") and not any(
                e["type"] == "bounce" and h1["frame"] < e["frame"] < h2["frame"]
                for e in seq):
            h2["stroke"] += " volley"
