"""
video_compression.py
Sentio Mind · Project 2 · Smart Behavioral Video Compression

Run: python solution.py
Requires: ffmpeg, opencv-python, imagehash, Pillow, numpy
"""

import cv2
import json
import base64
import subprocess
import time
import numpy as np
from pathlib import Path
from PIL import Image
import imagehash

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
VIDEO_IN               = Path("video_sample_1.mov")
VIDEO_OUT              = Path("compressed_output.mp4")
REPORT_HTML_OUT        = Path("compression_report.html")
SEGMENTS_JSON_OUT      = Path("segments_kept.json")

PHASH_THRESHOLD        = 0.95
MOTION_KEEP_THRESH     = 0.15
MOTION_DISCARD_THRESH  = 0.05
CONTEXT_EVERY_SEC      = 3
OUTPUT_FPS             = 12
OUTPUT_CRF             = 28

# Downscale for analysis — full-res frame written to output
ANALYSIS_W,  ANALYSIS_H  = 320, 180   # face detection resolution
FLOW_W,      FLOW_H      = 80,  45    # optical flow resolution (33x faster than 320x180)


# ---------------------------------------------------------------------------
# PERCEPTUAL HASH
# ---------------------------------------------------------------------------

def compute_phash(frame: np.ndarray) -> str:
    """
    Compute a perceptual hash of the frame.
    Returns a string of '0' and '1' characters, length 64.
    """
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    h = imagehash.phash(img, hash_size=8)
    return "".join("1" if x else "0" for x in h.hash.flatten())


def phash_similarity(h1: str, h2: str) -> float:
    """
    Compare two hash strings. Return 1.0 if identical, 0.0 if completely different.
    Formula: 1.0 - (hamming_distance / length)
    """
    if not h1 or not h2 or len(h1) != len(h2):
        return 0.0
    hamming_dist = sum(1 for c1, c2 in zip(h1, h2) if c1 != c2)
    return 1.0 - (hamming_dist / len(h1))


# ---------------------------------------------------------------------------
# MOTION SCORE
# ---------------------------------------------------------------------------

def compute_motion_score(prev_gray, curr_gray: np.ndarray) -> float:
    """
    Dense optical flow between two grayscale frames. Return mean magnitude.
    If prev_gray is None, return 0.0.
    """
    if prev_gray is None:
        return 0.0
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(np.mean(mag))


# ---------------------------------------------------------------------------
# FACE PRESENCE CHECK
# ---------------------------------------------------------------------------

def has_face(frame: np.ndarray, cascade) -> bool:
    """
    True if at least one face detected.
    Equalises histogram on grayscale first for better CCTV detection.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20)
    )
    return len(faces) > 0


# ---------------------------------------------------------------------------
# FRAME KEEP DECISION
# ---------------------------------------------------------------------------

def should_keep_frame(frame: np.ndarray,
                      prev_tiny_gray,        # pre-computed 80x45 gray of prev frame
                      prev_kept_hash: str,
                      last_kept_time_sec: float,
                      current_time_sec: float,
                      cascade) -> tuple:
    """
    Apply the 5-step decision algorithm from README in strict order.
    Returns: (keep: bool, reason: str, motion_score: float, face_found: bool,
              curr_tiny_gray)  — caller stores curr_tiny for next iteration
    """
    # Resize once — small for pHash/face, tiny for flow
    small      = cv2.resize(frame, (ANALYSIS_W, ANALYSIS_H))
    curr_gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    curr_tiny  = cv2.resize(curr_gray, (FLOW_W, FLOW_H))

    # Step 1 — pHash similarity: discard near-duplicates immediately
    curr_hash = compute_phash(small)
    if prev_kept_hash:
        sim = phash_similarity(curr_hash, prev_kept_hash)
        if sim > PHASH_THRESHOLD:
            return False, "discarded_duplicate", 0.0, False, curr_tiny

    # Step 2 — Motion score (run at 80x45 — 33x faster, same result)
    motion = compute_motion_score(prev_tiny_gray, curr_tiny)

    # High motion — keep immediately, skip face detection (saves ~80% of Haar calls)
    if motion > MOTION_KEEP_THRESH:
        return True, "motion_above_threshold", motion, False, curr_tiny

    # Step 3 — Face override: only run Haar on low-motion frames
    # (duplicates already dropped in Step 1, high-motion kept above)
    face_found = has_face(small, cascade)
    if face_found:
        return True, "face_detected", motion, True, curr_tiny

    # Step 4 — Context frame: force-keep every CONTEXT_EVERY_SEC seconds
    if (current_time_sec - last_kept_time_sec) >= CONTEXT_EVERY_SEC:
        return True, "context_frame", motion, False, curr_tiny

    # Discard: low motion, no face, not a context frame
    return False, "discarded_static", motion, False, curr_tiny


# ---------------------------------------------------------------------------
# THUMBNAIL HELPER
# ---------------------------------------------------------------------------

def frame_to_b64_thumb(frame: np.ndarray, width: int = 200) -> str:
    """Resize frame keeping aspect ratio, encode as base64 JPEG."""
    h, w = frame.shape[:2]
    nh = int(h * width / w)
    thumb = cv2.resize(frame, (width, nh), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 72])
    return base64.b64encode(buf).decode("utf-8")


# ---------------------------------------------------------------------------
# VIDEO WRITING
# ---------------------------------------------------------------------------

def write_frames_to_video(kept_frames: list, output_path: Path,
                          fps: float, frame_size: tuple):
    """
    Write kept_frames to a temp AVI, then re-encode with ffmpeg to H.264 MP4.
    kept_frames must be a list of np.ndarray frames.
    """
    temp_avi = "temp_raw.avi"
    out = cv2.VideoWriter(
        temp_avi,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        frame_size
    )
    for f in kept_frames:
        out.write(f)
    out.release()

    subprocess.run([
        "ffmpeg", "-y", "-i", temp_avi,
        "-vcodec", "libx264",
        "-crf", str(OUTPUT_CRF),
        "-preset", "fast",
        str(output_path)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    Path(temp_avi).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# HTML REPORT
# ---------------------------------------------------------------------------

def generate_compression_report(segments: list, stats: dict, output_path: Path):
    """
    Self-contained offline HTML report.
    Inline CSS only. No CDN dependencies.
    """
    seg_cards = ""
    for s in segments:
        seg_cards += f"""
    <div class="card">
      <img src="data:image/jpeg;base64,{s['thumbnail_b64']}" alt="seg {s['segment_id']}">
      <div class="info">
        <span class="badge">{s['reason_kept']}</span>
        <div>{s['start_sec']}s &rarr; {s['end_sec']}s</div>
        <div>{s['frames_in_segment']} frames &nbsp;|&nbsp; motion {s['motion_score_avg']}</div>
      </div>
    </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Compression Report</title>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0f172a;color:#f8fafc;font-family:system-ui,sans-serif;padding:2rem;line-height:1.6}}
h1{{font-size:1.6rem;color:#38bdf8;margin-bottom:.3rem}}
h2{{font-size:1.1rem;color:#94a3b8;margin:2rem 0 1rem}}
.sub{{color:#94a3b8;font-size:.9rem;margin-bottom:2rem}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1rem;margin-bottom:2rem}}
.stat{{background:#1e293b;border-radius:10px;padding:1.2rem 1.4rem}}
.stat .lbl{{font-size:.72rem;color:#64748b;text-transform:uppercase;letter-spacing:.07em}}
.stat .val{{font-size:1.9rem;font-weight:700;color:#38bdf8;margin-top:.2rem}}
.stat .sub2{{font-size:.75rem;color:#64748b;margin-top:.2rem}}
.storyboard{{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:1rem}}
.card{{background:#1e293b;border-radius:8px;overflow:hidden}}
.card img{{width:100%;display:block}}
.info{{padding:.6rem .8rem;font-size:.78rem;color:#94a3b8}}
.badge{{display:inline-block;background:#0369a1;color:#e0f2fe;
  font-size:.68rem;padding:.15rem .5rem;border-radius:20px;margin-bottom:.3rem}}
</style>
</head>
<body>
<h1>Smart Behavioral Video Compression</h1>
<p class="sub">{stats['source_video']} &rarr; {stats['compressed_video']}</p>

<div class="grid">
  <div class="stat">
    <div class="lbl">Size Reduction</div>
    <div class="val">{stats['reduction_pct']}%</div>
    <div class="sub2">{stats['original_size_mb']} MB &rarr; {stats['compressed_size_mb']} MB</div>
  </div>
  <div class="stat">
    <div class="lbl">Frames Kept</div>
    <div class="val">{stats['frames_kept']}</div>
    <div class="sub2">of {stats['frames_original']} total</div>
  </div>
  <div class="stat">
    <div class="lbl">Duration</div>
    <div class="val">{stats['compressed_duration_sec']}s</div>
    <div class="sub2">from {stats['original_duration_sec']}s original</div>
  </div>
  <div class="stat">
    <div class="lbl">Processing Time</div>
    <div class="val">{stats['processing_time_sec']}s</div>
    <div class="sub2">Output: {stats['output_fps']} fps</div>
  </div>
  <div class="stat">
    <div class="lbl">Discarded (duplicate)</div>
    <div class="val">{stats['frames_discarded_reasons']['near_duplicate_phash']}</div>
    <div class="sub2">pHash near-duplicates</div>
  </div>
  <div class="stat">
    <div class="lbl">Discarded (static)</div>
    <div class="val">{stats['frames_discarded_reasons']['low_motion_no_face']}</div>
    <div class="sub2">low motion, no face</div>
  </div>
</div>

<h2>Segment Storyboard</h2>
<div class="storyboard">
{seg_cards}
</div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t_start = time.time()

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap      = cv2.VideoCapture(str(VIDEO_IN))
    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fw       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total / fps_in
    orig_mb  = VIDEO_IN.stat().st_size / 1_000_000 if VIDEO_IN.exists() else 0.0

    print(f"Input: {VIDEO_IN}  |  {total} frames  |  {duration:.1f}s  |  {orig_mb:.1f} MB")

    # ── BONUS: auto-calibrate motion thresholds from first 30 seconds ──────
    print("Calibrating motion thresholds from first 30s ...")
    calib_frames = min(int(fps_in * 30), total)
    motions      = []
    prev_calib_g = None

    for i in range(calib_frames):
        ret, f = cap.read()
        if not ret:
            break
        tiny   = cv2.resize(f, (FLOW_W, FLOW_H))
        curr_g = cv2.cvtColor(tiny, cv2.COLOR_BGR2GRAY)
        if prev_calib_g is not None:
            motions.append(compute_motion_score(prev_calib_g, curr_g))
        prev_calib_g = curr_g

    if motions:
        MOTION_DISCARD_THRESH = float(np.percentile(motions, 20))
        MOTION_KEEP_THRESH    = float(np.percentile(motions, 80))
        print(f"  Discard < {MOTION_DISCARD_THRESH:.4f}  |  Keep > {MOTION_KEEP_THRESH:.4f}")

    # Seek back to start after calibration
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ── MAIN PROCESSING LOOP ───────────────────────────────────────────────
    kept_frames   = []   # list of np.ndarray — held in RAM, written once at end
    segments      = []
    prev_tiny_g   = None  # 80x45 gray of previous frame — no full 3K frame stored
    prev_hash     = ""
    last_kept_t   = -999.0
    cur_seg       = None
    disc_dup      = 0
    disc_stat     = 0
    frame_idx     = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ts = frame_idx / fps_in

        keep, reason, motion, face, curr_tiny_g = should_keep_frame(
            frame, prev_tiny_g, prev_hash, last_kept_t, ts, cascade
        )

        if keep:
            kept_frames.append(frame.copy())

            # Update hash from small frame (consistent with should_keep_frame)
            small_f   = cv2.resize(frame, (ANALYSIS_W, ANALYSIS_H))
            prev_hash = compute_phash(small_f)
            last_kept_t = ts

            if cur_seg is None or (ts - cur_seg["end_sec"]) > 2.5:
                if cur_seg:
                    segments.append(cur_seg)
                cur_seg = {
                    "segment_id":            len(segments) + 1,
                    "start_sec":             round(ts, 2),
                    "end_sec":               round(ts, 2),
                    "frames_in_segment":     1,
                    "reason_kept":           reason,
                    "face_count_in_segment": 1 if face else 0,
                    "motion_score_avg":      round(motion, 3),
                    "thumbnail_b64":         frame_to_b64_thumb(frame),
                }
            else:
                cur_seg["end_sec"]               = round(ts, 2)
                cur_seg["frames_in_segment"]    += 1
                cur_seg["face_count_in_segment"] += 1 if face else 0
                # running average of motion score
                n = cur_seg["frames_in_segment"]
                cur_seg["motion_score_avg"] = round(
                    (cur_seg["motion_score_avg"] * (n-1) + motion) / n, 3)
        else:
            if "duplicate" in reason:
                disc_dup  += 1
            else:
                disc_stat += 1

        prev_tiny_g = curr_tiny_g  # store only 80x45 gray, not full 3K frame
        frame_idx += 1

    if cur_seg:
        segments.append(cur_seg)
    cap.release()

    print(f"Kept {len(kept_frames)} / {total} frames across {len(segments)} segments")
    print("Encoding compressed video ...")
    write_frames_to_video(kept_frames, VIDEO_OUT, OUTPUT_FPS, (fw, fh))

    comp_mb = VIDEO_OUT.stat().st_size / 1_000_000 if VIDEO_OUT.exists() else 0.0
    t_end   = time.time()

    stats = {
        "_readme":                  "Your segments_kept.json must match this structure exactly. The Sentio Mind pipeline reads this file to skip re-scanning the raw video. Do not add, remove, or rename any top-level key.",
        "source_video":             str(VIDEO_IN),
        "compressed_video":         str(VIDEO_OUT),
        "original_size_mb":         round(orig_mb, 2),
        "compressed_size_mb":       round(comp_mb, 2),
        "reduction_pct":            round((1 - comp_mb / (orig_mb + 1e-9)) * 100, 1),
        "original_duration_sec":    round(duration, 2),
        "compressed_duration_sec":  round(len(kept_frames) / OUTPUT_FPS, 2),
        "original_fps":             round(fps_in, 2),
        "output_fps":               OUTPUT_FPS,
        "frames_original":          total,
        "frames_kept":              len(kept_frames),
        "processing_time_sec":      round(t_end - t_start, 2),
        "segments":                 segments,
        "frames_discarded_reasons": {
            "near_duplicate_phash": disc_dup,
            "low_motion_no_face":   disc_stat,
            "total_discarded":      total - len(kept_frames),
        },
    }

    with open(SEGMENTS_JSON_OUT, "w") as f:
        json.dump(stats, f, indent=2)

    generate_compression_report(segments, stats, REPORT_HTML_OUT)

    print()
    print("=" * 55)
    print(f"  Done in {stats['processing_time_sec']}s")
    print(f"  Size:     {orig_mb:.1f} MB  →  {comp_mb:.1f} MB  ({stats['reduction_pct']}% smaller)")
    print(f"  Duration: {duration:.1f}s  →  {stats['compressed_duration_sec']:.1f}s")
    print(f"  Report  → {REPORT_HTML_OUT}")
    print(f"  JSON    → {SEGMENTS_JSON_OUT}")
    print("=" * 55)
