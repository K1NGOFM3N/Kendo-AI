#!/usr/bin/env python3
"""
Kendo suburi analyzer -- quick prototype
=========================================
Analyzes a SIDE-VIEW video of men-uchi suburi with footwork (no bogu).

A swing COUNTS as a strike only if ALL of these are true:
  1. Furikaburi: the hands rise high enough above the shoulders
  2. Hikitsuke: right around the strike, the left foot pulls up toward the
     right foot, quickly and far enough

Swings that fail are listed as "not counted" with the reason. Counted strikes
also get technique feedback (strike height, arm extension, posture, timing).

Outputs (in --out-dir):
  annotated.mp4  skeleton, strike counter, feedback
  swings.csv     one row per swing, counted or not, with all measurements
  chart.png      hand height, elbow angle and foot gap over time

Usage:
  python kendo_suburi.py my_suburi.mp4
  python kendo_suburi.py my_suburi.mp4 --no-footwork   # stationary suburi

Recording tips: phone on a tripod at hip height, 3-4 m away, filming from the
side, whole body AND both feet in frame, good light, 60 fps if possible.
"""
import argparse
import csv
import math
import os
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_full/float16/1/pose_landmarker_full.task")
MODEL_PATH = "pose_landmarker_full.task"

# MediaPipe landmark indices ("left" = the person's left)
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 11, 12, 13, 14, 15, 16
L_HIP, R_HIP = 23, 24
L_HEEL, R_HEEL, L_TOE, R_TOE = 29, 30, 31, 32
SKELETON = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
            (23, 24), (23, 25), (25, 27), (24, 26), (26, 28), (27, 29), (29, 31),
            (27, 31), (28, 30), (30, 32), (28, 32)]

# ============================================================================
# SETTINGS -- first guesses. Distances are in torso lengths (shoulder to hip),
# speeds in torso lengths per second. Tune with real footage and a sensei.
# ============================================================================
# --- Finding swings in the motion ---
MIN_SWING = 0.35          # hands must rise and fall at least this much

# --- What it takes for a swing to COUNT as a strike ---
COUNT_MIN_TOP = 0.45      # 1. furikaburi: hands at least this far above shoulders
REQUIRE_FOOTWORK = True   # 2. require the foot pull (or use --no-footwork)
MIN_PULL_SPEED = 2.0      #    left foot must close the gap at least this fast
MIN_PULL_DIST = 0.2       #    ...and close it by at least this much
PULL_FOOT = "left"        #    "left", or "either" for zenshin-kotai suburi
PULL_WINDOW = (-0.2, 0.5) #    seconds around the strike to look for the pull

# --- Technique feedback on counted strikes ---
GOOD_TOP = 0.8            # ideal furikaburi height
STRIKE_LOW = -0.4         # strike stopping below this = too low for men
STRIKE_HIGH = 0.35        # strike stopping above this = cut short
GOOD_ELBOW = 150          # degrees; arms extended at the strike
MAX_LEAN = 15             # degrees of torso lean from vertical
MAX_PULL_DELAY = 0.25     # seconds; left foot should follow the strike quickly


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading pose model (one time, ~9 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def angle(a, b, c):
    """Angle at point b (degrees) formed by a-b-c."""
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return math.degrees(math.acos(np.clip(cos, -1.0, 1.0)))


# ----------------------------------------------------------------------------
# Pass 1: pose estimation
# ----------------------------------------------------------------------------
def extract_landmarks(video_path):
    ensure_model()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=0.5, min_tracking_confidence=0.5)
    frames, i = [], 0
    with vision.PoseLandmarker.create_from_options(opts) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            img = mp.Image(image_format=mp.ImageFormat.SRGB,
                           data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = landmarker.detect_for_video(img, int(i * 1000 / fps))
            if res.pose_landmarks:
                frames.append(np.array([
                    [p.x * w, p.y * h, 1.0 if p.visibility is None else p.visibility]
                    for p in res.pose_landmarks[0]]))
            else:
                frames.append(None)
            i += 1
            if i % 60 == 0:
                print(f"  processed {i} frames")
    cap.release()
    return frames, fps


# ----------------------------------------------------------------------------
# Per-frame measurements
# ----------------------------------------------------------------------------
def foot_x(p, heel, toe):
    """Horizontal position of a foot: middle of heel and toe (visible below hakama)."""
    xs = [p[j, 0] for j in (heel, toe) if p[j, 2] > 0.3]
    return float(np.mean(xs)) if xs else np.nan


def frame_metrics(p):
    if p is None or p[[L_SH, R_SH, L_WR, R_WR, L_HIP, R_HIP], 2].mean() < 0.5:
        return None
    xy = p[:, :2]
    sh = (xy[L_SH] + xy[R_SH]) / 2
    hip = (xy[L_HIP] + xy[R_HIP]) / 2
    torso = np.linalg.norm(sh - hip)
    if torso < 1:
        return None
    hand = (xy[L_WR] + xy[R_WR]) / 2
    m = {
        "torso": torso,
        "hand_height": (sh[1] - hand[1]) / torso,  # + = above shoulders
        "elbow": (angle(xy[L_SH], xy[L_EL], xy[L_WR]) +
                  angle(xy[R_SH], xy[R_EL], xy[R_WR])) / 2,
        "lean": math.degrees(math.atan2(abs(sh[0] - hip[0]), hip[1] - sh[1])),
        "lfoot_x": foot_x(p, L_HEEL, L_TOE),
        "rfoot_x": foot_x(p, R_HEEL, R_TOE),
        "heel_lift": np.nan,
    }
    if p[[L_HEEL, L_TOE], 2].min() > 0.5:
        m["heel_lift"] = (xy[L_TOE][1] - xy[L_HEEL][1]) / torso  # + = heel up
    return m


def fill_smooth(a, k=3):
    a = np.asarray(a, dtype=float)
    idx, ok = np.arange(len(a)), ~np.isnan(a)
    if ok.sum() < 2:
        return a
    a = np.interp(idx, idx[ok], a[ok])
    return np.convolve(np.pad(a, (k // 2, k // 2), mode="edge"),
                       np.ones(k) / k, mode="valid")


# ----------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------
def find_swings(hh):
    """A swing = hands rise >= MIN_SWING from the last low, peak, then fall
    >= MIN_SWING. Returns (top_frame, frame_where_fall_confirmed) pairs."""
    swings, state, low_i, top_i = [], "seek_top", 0, 0
    for i, v in enumerate(hh):
        if state == "seek_top":
            if v < hh[low_i]:
                low_i = i
            if v - hh[low_i] >= MIN_SWING:
                state, top_i = "rising", i
        else:
            if v > hh[top_i]:
                top_i = i
            if hh[top_i] - v >= MIN_SWING:
                swings.append((top_i, i))
                state, low_i = "seek_top", i
    return swings


def find_strike(vel, top_i, hi):
    """Strike = where the FIRST downswing after the top stops: the hands nearly
    halt, or slow clearly and then speed up again (flowing into chudan)."""
    s, vmin = top_i, 0.0
    while s < hi - 1:
        vmin = min(vmin, vel[s])
        if vmin < -1.0:
            if vel[s] > 0.25 * vmin:
                break
            if vel[s] > 0.6 * vmin and vel[s + 1] < vel[s]:
                break
        s += 1
    return s


def find_pull(feet, s, hi, fps):
    """Look for the trailing foot snapping up toward the front foot near the strike."""
    gap, close_v, lx, rx = feet["gap"], feet["close_v"], feet["lx"], feet["rx"]
    n = len(gap)
    lo = max(0, s + int(PULL_WINDOW[0] * fps))
    hi = min(n, hi, s + int(PULL_WINDOW[1] * fps) + 1)
    if hi - lo < 2:
        return None
    pk = lo + int(np.argmax(close_v[lo:hi]))
    speed = close_v[pk]
    if not speed > 0:
        return None
    a = pk  # extend to the whole pull movement
    while a > 0 and close_v[a - 1] > 0.2 * speed:
        a -= 1
    b = pk
    while b < n - 1 and close_v[b + 1] > 0.2 * speed:
        b += 1
    return {
        "pull_speed": round(float(speed), 2),
        "pull_dist": round(float(gap[a] - gap[b]), 2),
        "pull_time": round((b - a) / fps, 2),
        "pull_delay": round((pk - s) / fps, 2),
        "pull_foot": "left" if abs(lx[b] - lx[a]) >= abs(rx[b] - rx[a]) else "right",
        "pull_frame": pk,
    }


def count_check(r, use_footwork):
    """Reasons this swing does NOT count as a strike (empty list = counts)."""
    why = []
    if r["top_height"] < COUNT_MIN_TOP:
        why.append(f"furikaburi too low ({r['top_height']} < {COUNT_MIN_TOP})")
    if use_footwork:
        if r.get("pull_speed") is None:
            why.append("no left-foot pull found")
        else:
            if PULL_FOOT == "left" and r["pull_foot"] != "left":
                why.append("right foot moved, not a left-foot pull")
            if r["pull_speed"] < MIN_PULL_SPEED:
                why.append(f"left-foot pull too slow ({r['pull_speed']} < {MIN_PULL_SPEED})")
            if r["pull_dist"] < MIN_PULL_DIST:
                why.append(f"left foot didn't come in far enough "
                           f"({r['pull_dist']} < {MIN_PULL_DIST})")
    return why


def critique(r, use_footwork):
    notes = []
    if r["top_height"] < GOOD_TOP:
        notes.append("Raise hands higher in furikaburi")
    if r["strike_height"] < STRIKE_LOW:
        notes.append("Strike stops low - aim for men height")
    elif r["strike_height"] > STRIKE_HIGH:
        notes.append("Swing cut short - extend through to men")
    if r["elbow_at_strike"] < GOOD_ELBOW:
        notes.append("Extend arms more at the strike")
    if r["lean_at_strike"] > MAX_LEAN:
        notes.append("Keep upper body upright")
    if use_footwork and r.get("pull_delay") is not None and r["pull_delay"] > MAX_PULL_DELAY:
        notes.append("Pull the left foot in sooner after the strike")
    return notes or ["Good strike"]


def analyze(frames, fps, use_footwork=True):
    ms = [frame_metrics(p) for p in frames]
    get = lambda key: [m[key] if m else np.nan for m in ms]
    hh = fill_smooth(get("hand_height"))
    elbow = fill_smooth(get("elbow"))
    lean = fill_smooth(get("lean"))
    heel = np.array(get("heel_lift"), dtype=float)
    vel = np.gradient(hh) * fps

    # Feet: horizontal positions in torso lengths (camera is fixed, so use the
    # median torso size for the whole clip to keep positions stable)
    torso_med = np.nanmedian(get("torso")) if any(m for m in ms) else 1.0
    lx = fill_smooth(get("lfoot_x")) / torso_med
    rx = fill_smooth(get("rfoot_x")) / torso_med
    gap = np.abs(lx - rx)
    feet_ok = not np.isnan(gap).all()
    close_v = -np.gradient(gap) * fps if feet_ok else gap
    feet = {"gap": gap, "close_v": close_v, "lx": lx, "rx": rx}
    if use_footwork and not feet_ok:
        print("WARNING: feet not visible enough -- footwork check turned off")
        use_footwork = False

    swings = find_swings(hh)
    results, strike_no = [], 0
    for n, (top_i, cross_i) in enumerate(swings):
        next_top = swings[n + 1][0] if n + 1 < len(swings) else len(hh)
        hi = min(len(hh), cross_i + int(0.5 * fps) + 1, next_top)
        s = find_strike(vel, top_i, hi)
        r = {
            "swing": n + 1, "top_frame": top_i, "strike_frame": s,
            "time_s": round(s / fps, 2),
            "top_height": round(float(hh[top_i]), 2),
            "strike_height": round(float(hh[s]), 2),
            "elbow_at_strike": round(float(elbow[s]), 1),
            "lean_at_strike": round(float(lean[s]), 1),
            "tempo_s": round((s - top_i) / fps, 2),
            "pull_speed": None, "pull_dist": None, "pull_time": None,
            "pull_delay": None, "pull_foot": None, "pull_frame": None,
        }
        if feet_ok:
            pull = find_pull(feet, s, next_top, fps)
            if pull:
                r.update(pull)
        r["not_counted_because"] = count_check(r, use_footwork)
        r["counted"] = not r["not_counted_because"]
        if r["counted"]:
            strike_no += 1
            r["strike"] = strike_no
            r["notes"] = critique(r, use_footwork)
        else:
            r["strike"] = None
            r["notes"] = []
        results.append(r)

    valid_heel = heel[~np.isnan(heel)]
    heel_summary = float((valid_heel > 0.02).mean()) if len(valid_heel) > fps else None
    return {"hh": hh, "elbow": elbow, "gap": gap if feet_ok else None,
            "swings": results, "heel_lifted_frac": heel_summary,
            "use_footwork": use_footwork}


# ----------------------------------------------------------------------------
# Outputs
# ----------------------------------------------------------------------------
def save_csv(swings, path):
    cols = ["swing", "counted", "strike", "time_s", "top_height", "strike_height",
            "elbow_at_strike", "lean_at_strike", "tempo_s", "pull_foot",
            "pull_speed", "pull_dist", "pull_time", "pull_delay"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols + ["feedback / not counted because"])
        for r in swings:
            msg = r["notes"] if r["counted"] else r["not_counted_because"]
            w.writerow([r[c] for c in cols] + ["; ".join(msg)])


def save_chart(result, fps, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(len(result["hh"])) / fps
    rows = 3 if result["gap"] is not None else 2
    fig, axes = plt.subplots(rows, 1, figsize=(12, 3 * rows), sharex=True)
    a1, a2 = axes[0], axes[1]
    a1.plot(t, result["hh"], color="#1f4e79")
    a1.axhline(COUNT_MIN_TOP, ls="--", c="orange", lw=0.8, label="min to count")
    a1.axhline(GOOD_TOP, ls="--", c="green", lw=0.8, label="ideal furikaburi")
    a1.axhline(0, ls=":", c="gray", lw=0.8, label="shoulder level")
    for r in result["swings"]:
        x = r["strike_frame"] / fps
        if r["counted"]:
            a1.plot(x, r["strike_height"], "o", c="green")
            a1.annotate(str(r["strike"]), (x, r["strike_height"]),
                        textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8)
        else:
            a1.plot(x, r["strike_height"], "x", c="red", ms=9, mew=2)
        a1.plot(r["top_frame"] / fps, r["top_height"], "^", c="orange")
    a1.set_ylabel("Hand height\n(torso lengths)")
    a1.legend(loc="upper right", fontsize=8)
    a2.plot(t, result["elbow"], color="#7a3b00")
    a2.axhline(GOOD_ELBOW, ls="--", c="green", lw=0.8)
    a2.set_ylabel("Elbow angle (deg)")
    if rows == 3:
        a3 = axes[2]
        a3.plot(t, result["gap"], color="#4b2c7a")
        for r in result["swings"]:
            if r["pull_frame"] is not None:
                a3.plot(r["pull_frame"] / fps, result["gap"][r["pull_frame"]], "v",
                        c="green" if r["counted"] else "red")
        a3.set_ylabel("Gap between feet\n(torso lengths)")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Green dot = counted strike (numbered), red X = swing not counted, "
                 "triangle = foot pull")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def save_annotated(video_path, frames, result, fps, path):
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    swings, i = result["swings"], 0
    sc = max(w, h) / 1280
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        ok, frame = cap.read()
        if not ok or i >= len(frames):
            break
        p = frames[i]
        if p is not None:
            for a, b in SKELETON:
                if p[a, 2] > 0.5 and p[b, 2] > 0.5:
                    cv2.line(frame, tuple(map(int, p[a, :2])), tuple(map(int, p[b, :2])),
                             (255, 200, 0), max(2, int(3 * sc)))
            for j, col in ((L_HEEL, (0, 0, 255)), (L_TOE, (0, 0, 255)),
                           (R_HEEL, (0, 255, 0)), (R_TOE, (0, 255, 0))):
                if p[j, 2] > 0.3:  # left foot red, right foot green
                    cv2.circle(frame, tuple(map(int, p[j, :2])), max(4, int(6 * sc)), col, -1)
        done = [r for r in swings if r["strike_frame"] <= i]
        n_counted = sum(r["counted"] for r in done)
        cv2.rectangle(frame, (0, 0), (int(700 * sc), int(150 * sc)), (0, 0, 0), -1)
        cv2.putText(frame, f"Strikes: {n_counted}", (int(15 * sc), int(45 * sc)),
                    font, 1.3 * sc, (255, 255, 255), 2)
        if done:
            last = done[-1]
            if last["counted"]:
                lines = [(m, (0, 220, 0) if m == "Good strike" else (0, 165, 255))
                         for m in last["notes"][:2]]
            else:
                lines = [("NOT COUNTED: " + last["not_counted_because"][0], (0, 0, 255))]
            for k, (msg, col) in enumerate(lines):
                cv2.putText(frame, msg, (int(15 * sc), int((90 + 40 * k) * sc)),
                            font, 0.8 * sc, col, 2)
        out.write(frame)
        i += 1
    cap.release()
    out.release()


def main():
    ap = argparse.ArgumentParser(description="Kendo suburi analyzer (prototype)")
    ap.add_argument("video")
    ap.add_argument("--out-dir", default="kendo_results")
    ap.add_argument("--no-footwork", action="store_true",
                    help="count strikes from the swing alone (stationary suburi)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Running pose estimation...")
    frames, fps = extract_landmarks(args.video)
    print(f"Pose found in {sum(f is not None for f in frames)}/{len(frames)} frames")

    result = analyze(frames, fps, use_footwork=REQUIRE_FOOTWORK and not args.no_footwork)
    swings = result["swings"]
    counted = [r for r in swings if r["counted"]]
    rejected = [r for r in swings if not r["counted"]]

    print(f"\nFound {len(swings)} swings -> {len(counted)} counted as strikes\n")
    for r in counted:
        pull = (f"  pull {r['pull_speed']}/s in {r['pull_time']}s"
                if r["pull_speed"] is not None else "")
        print(f"Strike {r['strike']:>2} @ {r['time_s']:>5}s  top {r['top_height']:>5}  "
              f"strike {r['strike_height']:>5}  elbow {r['elbow_at_strike']:>5}  "
              f"lean {r['lean_at_strike']:>4}{pull}  -> " + "; ".join(r["notes"]))
    if rejected:
        print("\nNot counted:")
        for r in rejected:
            print(f"  swing @ {r['time_s']:>5}s  -> " + "; ".join(r["not_counted_because"]))
    if counted:
        good = sum(r["notes"] == ["Good strike"] for r in counted)
        print(f"\n{good}/{len(counted)} strikes passed every technique check")
    if result["heel_lifted_frac"] is not None:
        print(f"Left heel looks lifted in {result['heel_lifted_frac']:.0%} of frames "
              "(rough estimate)")

    save_csv(swings, os.path.join(args.out_dir, "swings.csv"))
    save_chart(result, fps, os.path.join(args.out_dir, "chart.png"))
    print("Writing annotated video...")
    save_annotated(args.video, frames, result, fps,
                   os.path.join(args.out_dir, "annotated.mp4"))
    print(f"Done. Results in {args.out_dir}/")


if __name__ == "__main__":
    main()
