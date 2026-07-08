"""
Stage-1 validation of monocular 3D ballistic bounce detection.

Fits a gravity-constrained 3D flight arc to every smooth flight segment the
existing DP segmentation (main.segment_track) finds in the raw
TrackNet ball track, using a camera calibrated from the 14 court keypoints.
Solving each arc's height-above-court h(t) = 0 then gives a *continuous-time*
landing prediction — including for bounces the 2D event detector missed
(e.g. contacts that fall between frames at 25-30 fps broadcast footage).

This stage does NOT change the pipeline's bounce output. It produces a
comparison report so the physics' added value can be judged first:

  * console table: per-segment fit quality, predicted landing, verdict vs.
    the events recorded in combined_local.json
  * output_videos/bounce3d_report.json    the same, machine-readable
  * output_videos/bounce3d_heights.png    fitted height curves + events

Method notes:
  * Camera: intrinsics (single focal length, principal point at the image
    centre, no distortion) via cv2.calibrateCamera on frames where all 14
    keypoints were found; per-frame extrinsics via solvePnP, median-smoothed
    to damp keypoint jitter. Broadcast cameras move little within a rally,
    but the per-frame pose keeps slow pans from biasing the fits.
  * Arc model: constant-velocity ground motion + vertical parabola with known
    g. Drag/spin are ignored at this stage — over sub-second flight segments
    the resulting bias is small next to TrackNet noise, and it shows up
    honestly in the reported reprojection RMS.
  * The fit is a linear DLT least squares: each real (non-interpolated)
    detection contributes two rows linear in the 6 unknowns
    (X0, VX, Y0, VY, H0, VH); a depth-reweighting pass converts the implicit
    DLT weighting back to image-pixel residuals.
  * The landing time is the descending root of h(tau) = H0 + VH*tau -
    0.5*g*tau^2 = 0, i.e. sub-frame timing, and the landing spot follows from
    the ground velocity at that instant.

Inputs: reads keypoints + the raw ball track from combined_local.json when
present (runs of main.py after 2026-07 store them); otherwise
recomputes them with the court/ball models only (no player detection) and
caches to bounce3d_cache.json. The events to compare against always come
from combined_local.json.
"""

import json
import os

import cv2
import numpy as np

import main as rc                  # also sets up sys.path
import run_ball as ball
import run_court as court

from court_reference import CourtReference       # noqa: E402 (repo, via sys.path)
from homography import get_trans_matrix          # noqa: E402 (repo, via sys.path)

COMBINED_JSON = os.path.join(rc.OUTPUT_DIR, "combined_local.json")
CACHE_JSON = os.path.join(rc.OUTPUT_DIR, "bounce3d_cache.json")
REPORT_JSON = os.path.join(rc.OUTPUT_DIR, "bounce3d_report.json")
PLOT_PNG = os.path.join(rc.OUTPUT_DIR, "bounce3d_heights.png")

# Court-reference units per meter (baseline-to-baseline = 2374 units = 23.77m).
UNITS_PER_M = (2935.0 - 561.0) / 23.77
G_M_S2 = 9.81

MIN_FIT_POINTS = 6        # real detections needed for a 6-parameter arc
CALIB_VIEWS = 12          # frames fed to calibrateCamera for the focal length
POSE_SMOOTH = 9           # median window (frames) over rvec/tvec jitter
MATCH_TOL = 3.0           # frames: predicted landing <-> detected event match
LAND_AHEAD = 4.0          # frames past segment end a landing may still refer
                          # to this segment's terminating bounce
RMS_GOOD = 8.0            # px: reprojection RMS above this = unreliable fit
CONT_SIGMA = 75.0         # court units (~0.75m): softness of the position-
                          # continuity constraint between adjacent segments
                          # (model error at a junction is (v_out-v_in) * the
                          # sub-frame contact offset, so it must stay soft)
SIGMA_MAX = 2.5           # frames: landing predictions with a larger 1-sigma
                          # uncertainty are reported but not trusted


# --------------------------------------------------------------------------
# data acquisition
# --------------------------------------------------------------------------

def video_props():
    cap = cv2.VideoCapture(rc.INPUT_VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, w, h


def compute_track_and_kps():
    """Run court keypoint + ball detection over the whole video (no player
    model — not needed here). Same per-frame calls as main."""
    cap = cv2.VideoCapture(rc.INPUT_VIDEO)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    court_model = court.load_model()
    ball_model = ball.load_model()
    print(f"Models loaded (court on {court.device}, ball on {ball.device}); "
          f"{len(frames)} frames to process")

    kps, raw = [], [(None, None), (None, None)]
    for i, frame in enumerate(frames):
        kps.append(court.detect_keypoints(court_model, frame))
        if i >= 2:
            raw.append(ball.detect_ball(ball_model, frames[i - 2:i + 1], w, h))
        print(f"  detect frame {i + 1}/{len(frames)}", end="\r")
    print()
    return kps, raw


def get_data(combined):
    """Return (fps, w, h, keypoints, raw_ball) for the combined run's video,
    from the combined JSON if it has them, else from the cache, else by
    running the two detectors (and caching)."""
    n = len(combined["frames"])
    fps, w, h = video_props()
    fps = combined.get("fps", fps)

    rec0 = combined["frames"][0]
    if "keypoints" in rec0 and "ball_raw" in rec0:
        kps = [[(p[0], p[1]) for p in fr["keypoints"]] for fr in combined["frames"]]
        raw = [(fr["ball_raw"][0], fr["ball_raw"][1]) for fr in combined["frames"]]
        return fps, w, h, kps, raw

    if os.path.exists(CACHE_JSON):
        with open(CACHE_JSON) as f:
            cache = json.load(f)
        if len(cache["keypoints"]) == n:
            kps = [[(p[0], p[1]) for p in fr] for fr in cache["keypoints"]]
            raw = [(p[0], p[1]) for p in cache["ball_raw"]]
            return fps, w, h, kps, raw
        print("Cache frame count doesn't match combined_local.json; recomputing")

    kps, raw = compute_track_and_kps()
    if len(kps) != n:
        raise SystemExit(
            f"Video has {len(kps)} frames but combined_local.json has {n} — "
            f"was it produced from {rc.INPUT_VIDEO}? Re-run main.py.")
    with open(CACHE_JSON, "w") as f:
        json.dump({"keypoints": [[list(p) for p in fr] for fr in kps],
                   "ball_raw": [list(p) for p in raw]}, f)
    return fps, w, h, kps, raw


# --------------------------------------------------------------------------
# camera model from the court keypoints
# --------------------------------------------------------------------------

def calibrate_intrinsics(kps, ref_pts, w, h):
    """Estimate the focal length from frames with well-detected keypoints.
    Pinhole model: principal point fixed at the image centre, square pixels,
    zero distortion — the court plane's homography then determines f."""
    full = [i for i, k in enumerate(kps)
            if sum(1 for p in k if p[0] is not None) >= 12]
    if not full:
        raise SystemExit("No frames with >=12 court keypoints; cannot calibrate")
    step = max(1, len(full) // CALIB_VIEWS)
    views = full[::step][:CALIB_VIEWS]

    obj_pts, img_pts = [], []
    for i in views:
        o, m = [], []
        for (rx, ry), (px, py) in zip(ref_pts, kps[i]):
            if px is not None:
                o.append([rx, ry, 0.0])
                m.append([px, py])
        obj_pts.append(np.array(o, np.float32))
        img_pts.append(np.array(m, np.float32))

    K0 = np.array([[1.2 * w, 0, w / 2.0],
                   [0, 1.2 * w, h / 2.0],
                   [0, 0, 1]], np.float64)
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT |
             cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST |
             cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
    rms, K, _, _, _ = cv2.calibrateCamera(
        obj_pts, img_pts, (w, h), K0, np.zeros(5), flags=flags)
    return K, rms


def per_frame_poses(kps, ref_pts, K, n):
    """solvePnP per frame (previous pose as the iterative seed), gaps filled
    from neighbours, then component-wise median smoothing. Returns a list of
    3x4 projection matrices P = K [R|t]."""
    rvecs = [None] * n
    tvecs = [None] * n
    prev_r = prev_t = None
    for i in range(n):
        pairs = [((rx, ry), (px, py))
                 for (rx, ry), (px, py) in zip(ref_pts, kps[i]) if px is not None]
        if len(pairs) >= 6:
            o = np.array([[rx, ry, 0.0] for (rx, ry), _ in pairs], np.float32)
            m = np.array([im for _, im in pairs], np.float32)
            if prev_r is not None:
                ok, r, t = cv2.solvePnP(o, m, K, None, prev_r.copy(),
                                        prev_t.copy(), True,
                                        cv2.SOLVEPNP_ITERATIVE)
            else:
                ok, r, t = cv2.solvePnP(o, m, K, None,
                                        flags=cv2.SOLVEPNP_ITERATIVE)
            if ok:
                rvecs[i] = r.reshape(3)
                tvecs[i] = t.reshape(3)
                prev_r, prev_t = r, t

    for arr in (rvecs, tvecs):                  # fill missing from neighbours
        for i in range(1, n):
            if arr[i] is None:
                arr[i] = arr[i - 1]
        for i in range(n - 2, -1, -1):
            if arr[i] is None:
                arr[i] = arr[i + 1]
    if rvecs[0] is None:
        raise SystemExit("solvePnP failed on every frame")

    R = np.array(rvecs)
    T = np.array(tvecs)
    half = POSE_SMOOTH // 2
    Rs, Ts = R.copy(), T.copy()
    for i in range(n):
        a, b = max(0, i - half), min(n, i + half + 1)
        Rs[i] = np.median(R[a:b], axis=0)
        Ts[i] = np.median(T[a:b], axis=0)

    Ps = []
    for i in range(n):
        rot, _ = cv2.Rodrigues(Rs[i])
        Ps.append(K @ np.hstack([rot, Ts[i][:, None]]))
    return Ps


def height_axis_sign(P):
    """The court reference's XY plane is left-handed seen from above, so
    whether +Z or -Z points 'up' (away from the ground toward the sky) depends
    on the pose. Probe: raising a mid-court point must move it up in the image."""
    mid = np.array([832.0, 1748.0])
    base = P @ np.array([mid[0], mid[1], 0.0, 1.0])
    plus = P @ np.array([mid[0], mid[1], 100.0, 1.0])
    return 1.0 if plus[1] / plus[2] < base[1] / base[2] else -1.0


# --------------------------------------------------------------------------
# ballistic arc fitting
# --------------------------------------------------------------------------

def world_mats(tau, g_units, s):
    """Affine map from theta = (X0, VX, Y0, VY, H0, VH) to the homogeneous
    world point at segment-relative time tau: W = A @ theta + c."""
    A = np.zeros((4, 6))
    c = np.zeros(4)
    A[0, 0], A[0, 1] = 1.0, tau
    A[1, 2], A[1, 3] = 1.0, tau
    A[2, 4], A[2, 5] = s, s * tau
    c[2] = -s * 0.5 * g_units * tau * tau       # known gravity term
    c[3] = 1.0
    return A, c


def fit_segment(a, ts, us, vs, Ps, g_units, s):
    """Linear DLT least squares for the 6 arc parameters, with two
    depth-reweighting passes so residuals are effectively image pixels.
    Returns (theta, rms_px)."""
    mats = [world_mats(t - a, g_units, s) for t in ts]
    weights = np.ones(len(ts))
    theta = None
    for _ in range(3):
        rows, rhs = [], []
        for k, t in enumerate(ts):
            A, c = mats[k]
            P = Ps[t]
            for coef, row in ((us[k], 0), (vs[k], 1)):
                r = coef * P[2] - P[row]
                rows.append(weights[k] * (r @ A))
                rhs.append(-weights[k] * float(r @ c))
        theta, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
        depths = []
        for k, t in enumerate(ts):
            A, c = mats[k]
            depths.append(abs(float(Ps[t][2] @ (A @ theta + c))))
        weights = 1.0 / np.maximum(np.array(depths), 1e-6)
        weights *= len(ts) / weights.sum()

    err2 = []
    for k, t in enumerate(ts):
        A, c = mats[k]
        p = Ps[t] @ (A @ theta + c)
        err2.append((p[0] / p[2] - us[k]) ** 2 + (p[1] / p[2] - vs[k]) ** 2)
    return theta, float(np.sqrt(np.mean(err2)))


def landing_time(theta, g_units):
    """Descending root of h(tau) = H0 + VH*tau - g/2*tau^2 = 0, or None."""
    h0, vh = theta[4], theta[5]
    disc = vh * vh + 2.0 * g_units * h0
    if disc < 0:
        return None
    tau = (vh + np.sqrt(disc)) / g_units
    return tau if tau >= 0 else None


def landing_sigma(theta, cov6, g_units):
    """1-sigma uncertainty of the landing time via the delta method."""
    h0, vh = theta[4], theta[5]
    disc = vh * vh + 2.0 * g_units * h0
    if disc <= 0:
        return None
    grad = np.zeros(6)
    grad[4] = 1.0 / np.sqrt(disc)
    grad[5] = (1.0 + vh / np.sqrt(disc)) / g_units
    var = float(grad @ cov6 @ grad)
    return float(np.sqrt(var)) if var >= 0 else None


def fit_joint(eligible, fit_idx, cont, fx, fy, Ps, g_units, s):
    """Joint linear fit of all eligible segments' arcs with soft position-
    continuity at the junctions between adjacent fitted segments.

    A lone short far-court segment is badly conditioned monocularly (the arc
    can slide along the view ray almost freely); the ball's position being
    continuous through every contact ties each arc to its neighbours and
    removes that degeneracy. Velocity stays free per segment — that jump IS
    the event. Continuity rows are weighted 1px-equivalent per CONT_SIGMA
    court units, so they steer only the near-degenerate directions.

    Returns ({seg_idx: theta}, {seg_idx: rms_px}, {seg_idx: 6x6 covariance}).
    """
    pos = {seg_i: k for k, seg_i in enumerate(fit_idx)}
    n_par = 6 * len(fit_idx)

    obs = []                                    # (block, t, u, v, A, c)
    for seg_i in fit_idx:
        a, _, ts = eligible[seg_i]
        for t in ts:
            A, c = world_mats(t - a, g_units, s)
            obs.append((pos[seg_i], t, float(fx[t]), float(fy[t]), A, c))

    weights = np.ones(len(obs))
    theta_all = None
    for _ in range(3):
        rows = np.zeros((2 * len(obs) + 3 * len(cont), n_par))
        rhs = np.zeros(rows.shape[0])
        r = 0
        for w_k, (blk, t, u, v, A, c) in zip(weights, obs):
            P = Ps[t]
            for coef, prow in ((u, 0), (v, 1)):
                rr = coef * P[2] - P[prow]
                rows[r, 6 * blk:6 * blk + 6] = w_k * (rr @ A)
                rhs[r] = -w_k * float(rr @ c)
                r += 1
        wc = 1.0 / CONT_SIGMA
        for i, j in cont:
            ai = eligible[i][0]
            aj = eligible[j][0]
            Ai, ci = world_mats(aj - ai, g_units, s)   # seg i at the junction
            Aj, cj = world_mats(0.0, g_units, s)       # seg j at the junction
            for d in range(3):
                rows[r, 6 * pos[i]:6 * pos[i] + 6] = wc * Ai[d]
                rows[r, 6 * pos[j]:6 * pos[j] + 6] -= wc * Aj[d]
                rhs[r] = -wc * float(ci[d] - cj[d])
                r += 1
        theta_all, *_ = np.linalg.lstsq(rows, rhs, rcond=None)
        new_w = []                              # depth reweighting -> px rows
        for blk, t, u, v, A, c in obs:
            lam = abs(float(Ps[t][2] @ (A @ theta_all[6 * blk:6 * blk + 6] + c)))
            new_w.append(1.0 / max(lam, 1e-6))
        weights = np.array(new_w)
        weights *= len(obs) / weights.sum()

    dof = max(rows.shape[0] - n_par, 1)
    sigma2 = float(np.sum((rows @ theta_all - rhs) ** 2)) / dof
    cov = sigma2 * np.linalg.pinv(rows.T @ rows)

    thetas, rmss, covs = {}, {}, {}
    err2 = {seg_i: [] for seg_i in fit_idx}
    for blk, t, u, v, A, c in obs:
        seg_i = fit_idx[blk]
        p = Ps[t] @ (A @ theta_all[6 * blk:6 * blk + 6] + c)
        err2[seg_i].append((p[0] / p[2] - u) ** 2 + (p[1] / p[2] - v) ** 2)
    for seg_i in fit_idx:
        blk = pos[seg_i]
        thetas[seg_i] = theta_all[6 * blk:6 * blk + 6]
        rmss[seg_i] = float(np.sqrt(np.mean(err2[seg_i])))
        covs[seg_i] = cov[6 * blk:6 * blk + 6, 6 * blk:6 * blk + 6]
    return thetas, rmss, covs


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def main():
    with open(COMBINED_JSON) as f:
        combined = json.load(f)
    events = combined["events"]
    fps, w, h, kps, raw = get_data(combined)
    n = len(raw)
    print(f"{n} frames @ {fps:.1f}fps, {len(events)} detected events loaded")

    court_ref = CourtReference()
    ref_pts = court_ref.key_points

    # homographies, same forward-fill as the pipeline
    inv_matrices, last_inv = [], None
    for k in kps:
        m = get_trans_matrix(k)
        if m is not None:
            last_inv = np.linalg.inv(m)
        inv_matrices.append(last_inv)

    prep = rc.prepare_track(raw, inv_matrices, court_ref)
    if prep is None:
        raise SystemExit("Ball track too sparse to analyse")
    lo, hi = prep["lo"], prep["hi"]
    fx, fy, real = prep["fx"], prep["fy"], prep["real"]
    bps = rc.segment_track(fx, fy, prep["scale"], lo, hi)
    bounds = [lo] + bps + [hi + 1]
    print(f"DP segmentation: {len(bounds) - 1} flight segments over "
          f"frames {lo}-{hi}")

    K, calib_rms = calibrate_intrinsics(kps, ref_pts, w, h)
    print(f"Camera: f={K[0, 0]:.0f}px (image {w}x{h}), "
          f"calibration RMS {calib_rms:.2f}px")
    Ps = per_frame_poses(kps, ref_pts, K, n)
    s = height_axis_sign(Ps[lo])
    g_units = G_M_S2 * UNITS_PER_M / fps ** 2

    # court bounds for the landing plausibility check (same margin as events)
    x_lo = court_ref.left_court_line[0][0] - rc.EVENT_BOUNCE_MARGIN
    x_hi = court_ref.right_court_line[0][0] + rc.EVENT_BOUNCE_MARGIN
    y_lo = court_ref.baseline_top[0][1] - rc.EVENT_BOUNCE_MARGIN
    y_hi = court_ref.baseline_bottom[0][1] + rc.EVENT_BOUNCE_MARGIN

    eligible = [(a, b, [t for t in range(a, b) if real[t]])
                for a, b in zip(bounds[:-1], bounds[1:])]   # frames [a, b)
    fit_idx = [i for i, (_, _, ts) in enumerate(eligible)
               if len(ts) >= MIN_FIT_POINTS]
    cont = []                # junctions with real coverage on both sides
    for i, j in zip(fit_idx[:-1], fit_idx[1:]):
        if j == i + 1:
            bj = eligible[j][0]
            if real[max(lo, bj - 4):bj].sum() >= 2 and \
                    real[bj:bj + 4].sum() >= 2:
                cont.append((i, j))
    thetas, rmss, covs = fit_joint(eligible, fit_idx, cont, fx, fy, Ps,
                                   g_units, s)
    print(f"Joint fit: {len(fit_idx)}/{len(eligible)} segments fitted, "
          f"{len(cont)} continuity junctions")

    segments = []
    for idx, (a, b, ts) in enumerate(eligible):
        seg = {"start": int(a), "end": int(b - 1)}
        if idx not in thetas:
            seg["status"] = "skipped"
            seg["reason"] = f"only {len(ts)} real detections"
            segments.append(seg)
            continue
        theta, rms = thetas[idx], rmss[idx]
        tau_land = landing_time(theta, g_units)
        sig = landing_sigma(theta, covs[idx], g_units)

        h_start = theta[4] / UNITS_PER_M
        tau_end = (b - 1) - a
        h_end = (theta[4] + theta[5] * tau_end
                 - 0.5 * g_units * tau_end ** 2) / UNITS_PER_M
        seg.update({
            "status": "fit", "n_points": len(ts), "rms_px": round(rms, 2),
            "theta": [float(v) for v in theta],
            "h_start_m": round(float(h_start), 2),
            "h_end_m": round(float(h_end), 2),
        })

        if tau_land is not None and a + tau_land <= (b - 1) + LAND_AHEAD:
            t_land = a + tau_land
            land_x = theta[0] + theta[1] * tau_land
            land_y = theta[2] + theta[3] * tau_land
            on = bool(x_lo <= land_x <= x_hi and y_lo <= land_y <= y_hi)
            seg.update({
                "t_land": round(float(t_land), 2),
                "t_land_sigma": None if sig is None else round(sig, 2),
                "landing_court": [round(float(land_x), 1),
                                  round(float(land_y), 1)],
                "landing_on_court": on,
            })
            near = min(events, key=lambda e: abs(e["frame"] - t_land))
            delta = near["frame"] - t_land
            if abs(delta) <= MATCH_TOL:
                seg["matched_event"] = {"type": near["type"],
                                        "frame": near["frame"],
                                        "delta": round(float(delta), 2)}
                if near["type"] == "bounce":
                    seg["verdict"] = "confirms detected bounce"
                else:
                    seg["verdict"] = "lands at detected hit (half-volley?)"
            elif sig is None or sig > SIGMA_MAX:
                seg["verdict"] = "landing too uncertain " \
                                 f"(±{sig:.1f}f)" if sig else \
                                 "landing too uncertain"
            elif rms > RMS_GOOD:
                seg["verdict"] = "possible landing, fit too noisy"
            elif not on:
                seg["verdict"] = "landing off court (ignored)"
            else:
                seg["verdict"] = "CANDIDATE MISSED BOUNCE"
        else:
            seg["verdict"] = f"no ground contact (ends {h_end:.1f}m high)"
        segments.append(seg)

    # reverse check: which detected bounces got physics confirmation?
    confirmed = {s["matched_event"]["frame"] for s in segments
                 if s.get("matched_event", {}).get("type") == "bounce"}
    det_bounces = [e["frame"] for e in events if e["type"] == "bounce"]

    # ---- console report ----------------------------------------------------
    print("\n%-11s %4s %7s %7s %12s %9s  %s" % (
        "segment", "pts", "rms_px", "h_end_m", "t_land", "on_court", "verdict"))
    for seg in segments:
        rng = f"{seg['start']}-{seg['end']}"
        if seg["status"] == "skipped":
            print("%-11s %4s %7s %7s %12s %9s  skipped (%s)" % (
                rng, "-", "-", "-", "-", "-", seg["reason"]))
            continue
        me = seg.get("matched_event")
        verdict = seg["verdict"]
        if me:
            verdict += f" @{me['frame']} (d={me['delta']:+.1f}f)"
        tl = "-"
        if "t_land" in seg:
            tl = f"{seg['t_land']:.1f}"
            if seg.get("t_land_sigma") is not None:
                tl += f"±{seg['t_land_sigma']:.1f}"
        print("%-11s %4d %7.2f %7.2f %12s %9s  %s" % (
            rng, seg["n_points"], seg["rms_px"], seg["h_end_m"], tl,
            {True: "yes", False: "no"}.get(seg.get("landing_on_court"), "-"),
            verdict))

    print(f"\nDetected bounces: {det_bounces}")
    print(f"Physics-confirmed: {sorted(confirmed)}")
    unconfirmed = [b for b in det_bounces if b not in confirmed]
    if unconfirmed:
        print(f"Detected but not physics-confirmed: {unconfirmed}")
    cands = [s for s in segments if s.get("verdict") == "CANDIDATE MISSED BOUNCE"]
    if cands:
        print("Candidate missed bounces (arc reaches ground, no detected event):")
        for s in cands:
            print(f"  ~frame {s['t_land']:.1f}, landing at court "
                  f"({s['landing_court'][0]:.0f}, {s['landing_court'][1]:.0f}), "
                  f"fit rms {s['rms_px']:.1f}px")

    with open(REPORT_JSON, "w") as f:
        json.dump({"focal_px": float(K[0, 0]), "calib_rms_px": float(calib_rms),
                   "fps": fps, "g_units_per_frame2": float(g_units),
                   "up_sign": s, "segments": segments,
                   "detected_bounces": det_bounces,
                   "confirmed_bounces": sorted(confirmed)}, f, indent=2)

    # ---- height plot ---------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14, 5))
    for seg in segments:
        if seg["status"] != "fit":
            continue
        a, b = seg["start"], seg["end"]
        th = seg["theta"]
        tt = np.linspace(0, b - a, 40)
        hh = (th[4] + th[5] * tt - 0.5 * g_units * tt ** 2) / UNITS_PER_M
        ax.plot(a + tt, hh, lw=1.8)
        if "t_land" in seg:
            ax.plot(seg["t_land"], 0, "v", color="green", ms=7)
            if seg.get("t_land_sigma"):
                ax.errorbar(seg["t_land"], 0, xerr=seg["t_land_sigma"],
                            color="green", capsize=3, lw=1)
    for e in events:
        color = "red" if e["type"] == "bounce" else "blue"
        ax.axvline(e["frame"], color=color, alpha=0.35, lw=1)
        ax.text(e["frame"], ax.get_ylim()[1] * 0.95, e["type"][0],
                color=color, ha="center", fontsize=8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("frame")
    ax.set_ylabel("fitted ball height (m)")
    ax.set_title("3D ballistic fits per flight segment "
                 "(red=detected bounce, blue=hit, green=predicted landing)")
    fig.tight_layout()
    fig.savefig(PLOT_PNG, dpi=130)
    print(f"\nReport: {REPORT_JSON}\nPlot:   {PLOT_PNG}")


if __name__ == "__main__":
    main()
