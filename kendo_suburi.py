#!/usr/bin/env python3
"""
Kendo suburi analyzer -- quick prototype
=========================================
Analyzes a SIDE-VIEW video of men-uchi suburi (no bogu) and reports, per swing:
  - furikaburi height (how high the hands go)
  - strike height (where the hands stop)
  - arm extension at the strike
  - upper-body lean at the strike
  - swing tempo (top -> strike)
plus a rough check of whether the left heel stays lifted.

Outputs (in --out-dir):
  annotated.mp4  skeleton + live rep counter + feedback
  reps.csv       one row per swing
  chart.png      hand height and elbow angle over time

Usage:
  pip install mediapipe opencv-python numpy matplotlib
  python kendo_suburi.py my_suburi.mp4
  python kendo_suburi.py my_suburi.mp4 --out-dir results

Recording tips: phone on a tripod at hip height, camera 3-4 m away, filming
from the side, whole body in frame, good light, 60 fps if possible.
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
L_HIP, R_HIP, L_HEEL, L_TOE = 23, 24, 29, 31
SKELETON = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
            (23, 24), (23, 25), (25, 27), (24, 26), (26, 28), (27, 29), (29, 31),
            (27, 31), (28, 30), (30, 32), (28, 32)]

# --- Thresholds: first guesses, measured in torso lengths / degrees. ---------
# --- Tune these with real footage and an experienced sensei. -------------
TOP_THRESH = 0.5      # hands this far above shoulders = furikaburi started
STRIKE_THRESH = 0.4   # hands drop below this = swing came down
GOOD_TOP = 0.54        # furikaburi should reach at least this high
STRIKE_LOW = -0.4     # strike stopping below this = too low for men
STRIKE_HIGH = 0.35    # strike stopping above this = cut short
GOOD_ELBOW = 150      # degrees; arms should be extended at the strike
MAX_LEAN = 15         # degrees of torso lean from vertical


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
# Analysis
# ----------------------------------------------------------------------------
def frame_metrics(p):
    """Per-frame measurements from one set of landmarks (pixels + visibility)."""
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
        "hand_height": (sh[1] - hand[1]) / torso,  # + = above shoulders
        "elbow": (angle(xy[L_SH], xy[L_EL], xy[L_WR]) +
                  angle(xy[R_SH], xy[R_EL], xy[R_WR])) / 2,
        "lean": math.degrees(math.atan2(abs(sh[0] - hip[0]), hip[1] - sh[1])),
        "heel_lift": np.nan,
    }
    if p[[L_HEEL, L_TOE], 2].min() > 0.5:
        m["heel_lift"] = (xy[L_TOE][1] - xy[L_HEEL][1]) / torso  # + = heel up
    return m


def fill_smooth(a, k=5):
    a = np.asarray(a, dtype=float)
    idx, ok = np.arange(len(a)), ~np.isnan(a)
    if ok.sum() < 2:
        return a
    a = np.interp(idx, idx[ok], a[ok])
    return np.convolve(np.pad(a, (k // 2, k // 2), mode="edge"),
                       np.ones(k) / k, mode="valid")


def critique(r):
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
    return notes or ["Good swing"]


def analyze(frames, fps):
    ms = [frame_metrics(p) for p in frames]
    get = lambda key: [m[key] if m else np.nan for m in ms]
    hh = fill_smooth(get("hand_height"))
    elbow = fill_smooth(get("elbow"))
    lean = fill_smooth(get("lean"))
    heel = np.array(get("heel_lift"), dtype=float)
    vel = np.gradient(hh) * fps  # torso lengths per second

    # Rep detection: hands go above TOP_THRESH, then come back below STRIKE_THRESH
    swings, state, top_i = [], "down", None
    for i, v in enumerate(hh):
        if state == "down" and v > TOP_THRESH:
            state, top_i = "up", i
        elif state == "up":
            if v > hh[top_i]:
                top_i = i
            if v < STRIKE_THRESH:
                state = "down"
                swings.append((top_i, i))

    reps = []
    for n, (top_i, cross_i) in enumerate(swings, 1):
        hi = min(len(hh), cross_i + int(0.5 * fps) + 1)
        fastest = top_i + int(np.argmin(vel[top_i:hi]))
        s = fastest  # strike = where the downswing has mostly stopped
        while s < hi - 1 and vel[s] < 0.25 * vel[fastest]:
            s += 1
        r = {
            "rep": n, "top_frame": top_i, "strike_frame": s,
            "time_s": round(s / fps, 2),
            "top_height": round(float(hh[top_i]), 2),
            "strike_height": round(float(hh[s]), 2),
            "elbow_at_strike": round(float(elbow[s]), 1),
            "lean_at_strike": round(float(lean[s]), 1),
            "tempo_s": round((s - top_i) / fps, 2),
        }
        r["notes"] = critique(r)
        reps.append(r)

    valid_heel = heel[~np.isnan(heel)]
    heel_summary = None
    if len(valid_heel) > fps:  # need at least ~1 s of visible left foot
        heel_summary = float((valid_heel > 0.02).mean())
    return {"hh": hh, "elbow": elbow, "reps": reps, "heel_lifted_frac": heel_summary}


# ----------------------------------------------------------------------------
# Outputs
# ----------------------------------------------------------------------------
def save_csv(reps, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        cols = ["rep", "time_s", "top_height", "strike_height",
                "elbow_at_strike", "lean_at_strike", "tempo_s", "notes"]
        w.writerow(cols)
        for r in reps:
            w.writerow([r[c] if c != "notes" else "; ".join(r[c]) for c in cols])


def save_chart(result, fps, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(len(result["hh"])) / fps
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    a1.plot(t, result["hh"], color="#1f4e79")
    a1.axhline(GOOD_TOP, ls="--", c="green", lw=0.8, label="target furikaburi")
    a1.axhline(0, ls=":", c="gray", lw=0.8, label="shoulder level")
    for r in result["reps"]:
        a1.plot(r["top_frame"] / fps, r["top_height"], "^", c="orange")
        ok = r["notes"] == ["Good swing"]
        a1.plot(r["strike_frame"] / fps, r["strike_height"], "o",
                c="green" if ok else "red")
    a1.set_ylabel("Hand height\n(torso lengths)")
    a1.legend(loc="upper right", fontsize=8)
    a2.plot(t, result["elbow"], color="#7a3b00")
    a2.axhline(GOOD_ELBOW, ls="--", c="green", lw=0.8)
    a2.set_ylabel("Elbow angle (deg)")
    a2.set_xlabel("Time (s)")
    fig.suptitle("Suburi analysis: triangles = furikaburi top, dots = strike")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def save_annotated(video_path, frames, result, fps, path):
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    reps, i = result["reps"], 0
    scale = max(w, h) / 1280
    while True:
        ok, frame = cap.read()
        if not ok or i >= len(frames):
            break
        p = frames[i]
        if p is not None:
            for a, b in SKELETON:
                if p[a, 2] > 0.5 and p[b, 2] > 0.5:
                    cv2.line(frame, tuple(map(int, p[a, :2])), tuple(map(int, p[b, :2])),
                             (255, 200, 0), max(2, int(3 * scale)))
        done = [r for r in reps if r["strike_frame"] <= i]
        cv2.rectangle(frame, (0, 0), (int(620 * scale), int(150 * scale)), (0, 0, 0), -1)
        cv2.putText(frame, f"Rep {len(done)}", (int(15 * scale), int(45 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3 * scale, (255, 255, 255), 2)
        if done:
            for k, note in enumerate(done[-1]["notes"][:2]):
                color = (0, 220, 0) if note == "Good swing" else (0, 120, 255)
                cv2.putText(frame, note, (int(15 * scale), int((90 + 40 * k) * scale)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9 * scale, color, 2)
        out.write(frame)
        i += 1
    cap.release()
    out.release()


def main():
    ap = argparse.ArgumentParser(description="Kendo suburi analyzer (prototype)")
    ap.add_argument("video")
    ap.add_argument("--out-dir", default="kendo_results")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Running pose estimation...")
    frames, fps = extract_landmarks(args.video)
    detected = sum(f is not None for f in frames)
    print(f"Pose found in {detected}/{len(frames)} frames")

    result = analyze(frames, fps)
    reps = result["reps"]
    print(f"\nDetected {len(reps)} swings\n")
    for r in reps:
        print(f"Rep {r['rep']:>2} @ {r['time_s']:>5}s  top {r['top_height']:>5}  "
              f"strike {r['strike_height']:>5}  elbow {r['elbow_at_strike']:>5}  "
              f"lean {r['lean_at_strike']:>4}  tempo {r['tempo_s']}s  -> "
              + "; ".join(r["notes"]))
    if reps:
        good = sum(r["notes"] == ["Good swing"] for r in reps)
        print(f"\n{good}/{len(reps)} swings passed every check")
    if result["heel_lifted_frac"] is not None:
        print(f"Left heel looks lifted in {result['heel_lifted_frac']:.0%} of frames "
              "(rough estimate)")
    else:
        print("Left foot not visible enough to check the heel")

    save_csv(reps, os.path.join(args.out_dir, "reps.csv"))
    save_chart(result, fps, os.path.join(args.out_dir, "chart.png"))
    print("Writing annotated video...")
    save_annotated(args.video, frames, result, fps,
                   os.path.join(args.out_dir, "annotated.mp4"))
    print(f"Done. Results in {args.out_dir}/")


if __name__ == "__main__":
    main()
