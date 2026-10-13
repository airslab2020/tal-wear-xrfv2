#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import re
import subprocess
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np


BASE_DIR = Path(os.environ.get("XRFV2_SOURCE_ROOT", "./data/xrfv2_source"))
VIDEO_ROOT = BASE_DIR / "kinect_action_cropped_tracks"
COMMAND_DIR = BASE_DIR / "command_time"
MONITOR_ROOT = BASE_DIR / "kinect_rgb_feature_monitor"
FEATURE_ROOT = BASE_DIR / "kinect_rgb_secondwise_feature"
TARGET_LIST = MONITOR_ROOT / "wifi_853_names.txt"
DEFAULT_DURATION_OVERRIDES = MONITOR_ROOT / "duration_overrides.json"
FFPROBE = shutil.which("ffprobe")

TARGET_FPS = 15
CLIP_LEN = 15
FEATURE_STEM = "scheme2_15frames_secondwise"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-list", default=str(TARGET_LIST))
    parser.add_argument("--duration-overrides", default=str(DEFAULT_DURATION_OVERRIDES))
    parser.add_argument("--video-root", default=str(VIDEO_ROOT))
    parser.add_argument("--command-dir", default=str(COMMAND_DIR))
    parser.add_argument("--monitor-root", default=str(MONITOR_ROOT))
    parser.add_argument("--feature-root", default=str(FEATURE_ROOT))
    parser.add_argument("--duration-tolerance", type=float, default=0.5)
    parser.add_argument("--stable-min-age", type=float, default=20.0)
    parser.add_argument("--stability-wait", type=float, default=0.0)
    parser.add_argument("--scan-interval", type=float, default=120.0)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--scan-only", action="store_true")
    parser.add_argument("--extract-ready", action="store_true")
    parser.add_argument("--max-extract", type=int, default=0)
    parser.add_argument("--overwrite-feature", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", default="")
    return parser.parse_args()


def parse_sample_name(sample_id):
    match = re.match(r"^(\d+)_(livingroom|office|kitchen)_(\d+)$", sample_id)
    if not match:
        raise ValueError("bad sample id: {}".format(sample_id))
    return match.group(2)


def load_targets(path):
    targets = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if not value:
                continue
            if value.endswith(".h5"):
                value = value[:-3]
            targets.append(value)
    return sorted(dict.fromkeys(targets))


def parse_hms(value):
    return datetime.strptime(value.strip(), "%H:%M:%S")


def load_command_window(command_dir, sample_id):
    scene = parse_sample_name(sample_id)
    path = Path(command_dir) / scene / "{}.json".format(sample_id)
    if not path.exists():
        return None, "missing command json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list) or not data:
            return None, "empty command json"
        start_text = data[0]["start_time"]
        end_text = data[-1]["end_time"]
        start = parse_hms(start_text)
        end = parse_hms(end_text)
        duration = (end - start).total_seconds()
        if duration < 0:
            duration += 24 * 3600
        return {
            "path": str(path),
            "start_time": start_text,
            "end_time": end_text,
            "duration_sec": int(round(duration)),
            "action_count": len(data),
        }, ""
    except Exception as exc:
        return None, "{}: {}".format(type(exc).__name__, exc)


def load_duration_overrides(path):
    path = Path(path)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError("duration override file must be a JSON object: {}".format(path))
    return data


def stat_video(video_path):
    path = Path(video_path)
    if not path.exists():
        return {"exists": False}
    st = path.stat()
    return {
        "exists": True,
        "size_bytes": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "mtime_iso": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
        "age_sec": max(0.0, time.time() - st.st_mtime),
    }


def parse_rate(value):
    if not value or value == "0/0":
        return None
    num, den = value.split("/")
    den = float(den)
    return float(num) / den if den else None


def probe_video_ffprobe(path):
    if FFPROBE is None:
        raise FileNotFoundError("ffprobe not found: {}".format(FFPROBE))
    proc = subprocess.run([
        str(FFPROBE), "-v", "error", "-count_frames",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,duration,nb_read_frames",
        "-of", "json", str(path),
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=True, timeout=30)
    data = json.loads(proc.stdout)
    stream = data["streams"][0]
    fps = parse_rate(stream.get("avg_frame_rate"))
    frames_raw = stream.get("nb_read_frames")
    frame_count = int(frames_raw) if frames_raw not in (None, "N/A") else None
    duration = None
    if frame_count is not None and fps:
        duration = frame_count / fps
    elif stream.get("duration") not in (None, "N/A"):
        duration = float(stream["duration"])
    return {
        "fps": fps,
        "frame_count": frame_count,
        "video_duration_sec": duration,
        "width": stream.get("width", ""),
        "height": stream.get("height", ""),
    }


def probe_video(path):
    path = Path(path)
    meta_path = path.with_name("meta.json")
    if meta_path.exists():
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        frames = data.get("rgb_frames")
        duration = data.get("rgb_duration_s")
        fps = None
        if frames not in (None, "") and duration not in (None, "", 0):
            fps = float(frames) / float(duration)
        return {
            "fps": fps,
            "frame_count": int(frames) if frames not in (None, "") else None,
            "video_duration_sec": float(duration) if duration not in (None, "") else None,
            "width": "",
            "height": "",
        }
    return probe_video_ffprobe(path)


def feature_paths(feature_root, scene, sample_id):
    out_dir = Path(feature_root) / scene / sample_id
    return {
        "dir": out_dir,
        "pt": out_dir / "{}.pt".format(FEATURE_STEM),
        "npy": out_dir / "{}.npy".format(FEATURE_STEM),
        "ts": out_dir / "{}_timestamps.npy".format(FEATURE_STEM),
        "meta": out_dir / "feature_meta.json",
    }


def read_feature_shape(path):
    try:
        if Path(path).exists():
            arr = np.load(path, mmap_mode="r")
            return "x".join(str(x) for x in arr.shape)
    except Exception:
        return ""
    return ""


def load_state(path):
    if not Path(path).exists():
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(path, state):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(str(tmp), str(path))


def summarize(rows):
    summary = {
        "target_count": len(rows),
        "video_exists": 0,
        "missing": 0,
        "generating": 0,
        "ready": 0,
        "duration_ok": 0,
        "duration_bad": 0,
        "feature_done": 0,
        "feature_failed": 0,
        "command_missing_or_bad": 0,
    }
    for row in rows:
        if row["exists"] == "1":
            summary["video_exists"] += 1
        if row["status"] in summary:
            summary[row["status"]] += 1
        if row["duration_ok"] == "1":
            summary["duration_ok"] += 1
    return summary


def write_csv(path, rows):
    fieldnames = [
        "sample_id", "scene", "status", "exists", "stable",
        "command_start", "command_end", "command_duration_sec", "action_count", "duration_override",
        "video_path", "size_bytes", "mtime_iso", "age_sec",
        "fps", "frame_count", "video_duration_sec", "duration_diff_sec", "duration_ok",
        "feature_exists", "feature_shape", "feature_path", "error",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def scan(args, previous_state):
    video_root = Path(args.video_root)
    command_dir = Path(args.command_dir)
    feature_root = Path(args.feature_root)
    targets = load_targets(args.target_list)
    duration_overrides = load_duration_overrides(args.duration_overrides)
    wanted = {x.strip() for x in args.samples.split(",") if x.strip()}
    if wanted:
        targets = [x for x in targets if x in wanted]

    first_stats = {}
    if args.stability_wait > 0:
        for sample_id in targets:
            scene = parse_sample_name(sample_id)
            first_stats[sample_id] = stat_video(video_root / scene / sample_id / "rgb_cropped.mkv")
        time.sleep(args.stability_wait)

    rows = []
    next_state = {}
    now = time.time()

    for sample_id in targets:
        scene = parse_sample_name(sample_id)
        video_path = video_root / scene / sample_id / "rgb_cropped.mkv"
        command, command_error = load_command_window(command_dir, sample_id)
        duration_override = ""
        if command is not None and sample_id in duration_overrides:
            override = duration_overrides[sample_id]
            if isinstance(override, dict):
                command["duration_sec"] = int(override["duration_sec"])
                command["end_time"] = override.get("end_time", command["end_time"])
                command["action_count"] = override.get("action_count", command["action_count"])
                duration_override = override.get("reason", "duration_override")
            else:
                command["duration_sec"] = int(override)
                duration_override = "duration_override"
        current = stat_video(video_path)
        prev = previous_state.get(sample_id, {})
        stable = False

        if current.get("exists"):
            if args.stability_wait > 0 and first_stats.get(sample_id, {}).get("exists"):
                a = first_stats[sample_id]
                stable = (
                    a.get("size_bytes") == current.get("size_bytes")
                    and a.get("mtime_ns") == current.get("mtime_ns")
                    and current.get("age_sec", 0.0) >= args.stable_min_age
                )
            else:
                stable = (
                    prev.get("size_bytes") == current.get("size_bytes")
                    and prev.get("mtime_ns") == current.get("mtime_ns")
                    and current.get("age_sec", 0.0) >= args.stable_min_age
                )
            next_state[sample_id] = {
                "size_bytes": current.get("size_bytes"),
                "mtime_ns": current.get("mtime_ns"),
                "seen_at": now,
                "stable": stable,
            }

        paths = feature_paths(feature_root, scene, sample_id)
        feature_shape = read_feature_shape(paths["npy"])
        feature_exists = bool(feature_shape)

        error = command_error
        probe = {}
        duration_ok = False
        duration_diff = None

        if not current.get("exists"):
            status = "missing"
        elif not stable:
            status = "generating"
        elif command is None:
            status = "command_missing_or_bad"
        else:
            try:
                probe = probe_video(video_path)
                if probe.get("video_duration_sec") is None:
                    raise RuntimeError("cannot read video duration")
                duration_diff = probe["video_duration_sec"] - command["duration_sec"]
                duration_ok = abs(duration_diff) <= args.duration_tolerance
                if feature_exists:
                    expected_prefix = "{}x".format(command["duration_sec"])
                    status = "feature_done" if feature_shape.startswith(expected_prefix) else "feature_failed"
                    if status == "feature_failed":
                        error = "feature shape {} does not start with expected {}".format(
                            feature_shape, expected_prefix
                        )
                elif duration_ok:
                    status = "ready"
                else:
                    status = "duration_bad"
            except Exception as exc:
                status = "duration_bad"
                error = "{}: {}".format(type(exc).__name__, exc)

        row = {
            "sample_id": sample_id,
            "scene": scene,
            "status": status,
            "exists": "1" if current.get("exists") else "0",
            "stable": "1" if stable else "0",
            "command_start": command["start_time"] if command else "",
            "command_end": command["end_time"] if command else "",
            "command_duration_sec": command["duration_sec"] if command else "",
            "action_count": command["action_count"] if command else "",
            "duration_override": duration_override,
            "video_path": str(video_path),
            "size_bytes": current.get("size_bytes", ""),
            "mtime_iso": current.get("mtime_iso", ""),
            "age_sec": "{:.3f}".format(current.get("age_sec", 0.0)) if current.get("exists") else "",
            "fps": "{:.6f}".format(probe["fps"]) if probe.get("fps") else "",
            "frame_count": probe.get("frame_count", ""),
            "video_duration_sec": "{:.6f}".format(probe["video_duration_sec"]) if probe.get("video_duration_sec") is not None else "",
            "duration_diff_sec": "{:.6f}".format(duration_diff) if duration_diff is not None else "",
            "duration_ok": "1" if duration_ok else "0",
            "feature_exists": "1" if feature_exists else "0",
            "feature_shape": feature_shape,
            "feature_path": str(paths["npy"]),
            "error": error,
        }
        rows.append(row)

    return rows, next_state


class FeatureExtractor:
    def __init__(self, device, batch_size):
        import torch
        from torchvision.models.video import swin3d_b, Swin3D_B_Weights

        self.torch = torch
        self.batch_size = batch_size
        self.device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
        weights = Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1
        model = swin3d_b(weights=weights)
        model.head = torch.nn.Identity()
        model.eval().to(self.device)
        self.model = model
        self.preprocess = weights.transforms()

    def clip_to_tensor(self, clip_frames):
        torch = self.torch
        arr = np.stack(clip_frames, axis=0)
        tensor = torch.from_numpy(arr).permute(0, 3, 1, 2)
        return self.preprocess(tensor)

    def extract_batches(self, clips):
        torch = self.torch
        outputs = []
        with torch.no_grad():
            for i in range(0, len(clips), self.batch_size):
                batch = torch.stack(clips[i:i + self.batch_size], dim=0).to(self.device)
                feats = self.model(batch)
                outputs.append(feats.cpu())
        return torch.cat(outputs, dim=0)


def load_video_frames(video_path):
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("cannot open video: {}".format(video_path))
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError("no frames read from: {}".format(video_path))
    return frames, fps, frame_count


def build_secondwise_clips(extractor, frames, expected_seconds):
    clips = []
    timestamps = []
    for sec in range(expected_seconds):
        start = sec * TARGET_FPS
        if start >= len(frames):
            clip_frames = [frames[-1].copy()]
        else:
            clip_frames = frames[start:start + CLIP_LEN]
        while len(clip_frames) < CLIP_LEN:
            clip_frames.append(clip_frames[-1].copy())
        clips.append(extractor.clip_to_tensor(clip_frames))
        timestamps.append(float(sec))
    return clips, np.asarray(timestamps, dtype=np.float32)


def atomic_np_save(path, array):
    tmp = Path(str(path) + ".tmp.npy")
    np.save(tmp, array)
    os.replace(str(tmp), str(path))


def atomic_text_save(path, text):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(str(tmp), str(path))


def process_feature(row, args, extractor):
    if row["status"] != "ready":
        return "skip"
    scene = row["scene"]
    sample_id = row["sample_id"]
    expected_seconds = int(row["command_duration_sec"])
    paths = feature_paths(args.feature_root, scene, sample_id)
    if paths["npy"].exists() and not args.overwrite_feature:
        shape = read_feature_shape(paths["npy"])
        if shape.startswith("{}x".format(expected_seconds)):
            return "exists"

    paths["dir"].mkdir(parents=True, exist_ok=True)
    frames, video_fps, frame_count = load_video_frames(row["video_path"])
    clips, timestamps = build_secondwise_clips(extractor, frames, expected_seconds)
    features = extractor.extract_batches(clips)
    features_np = features.numpy()

    atomic_np_save(paths["npy"], features_np)
    atomic_np_save(paths["ts"], timestamps)
    tmp_pt = Path(str(paths["pt"]) + ".tmp")
    extractor.torch.save({
        "features": features,
        "timestamps_sec": extractor.torch.tensor(timestamps, dtype=extractor.torch.float32),
        "clip_len": CLIP_LEN,
        "fps": TARGET_FPS,
        "video_fps": float(video_fps),
        "frame_count": int(frame_count),
        "expected_seconds": int(expected_seconds),
        "video_path": row["video_path"],
        "scheme": "scheme2_15frames_secondwise_command_aligned",
    }, tmp_pt)
    os.replace(str(tmp_pt), str(paths["pt"]))
    atomic_text_save(paths["meta"], json.dumps({
        "sample_id": sample_id,
        "scene": scene,
        "video_path": row["video_path"],
        "command_duration_sec": expected_seconds,
        "duration_override": row.get("duration_override", ""),
        "feature_shape": list(features_np.shape),
        "timestamps_shape": list(timestamps.shape),
        "clip_len": CLIP_LEN,
        "target_fps": TARGET_FPS,
        "video_fps": float(video_fps),
        "frame_count": int(frame_count),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }, ensure_ascii=False, indent=2) + "\n")
    return "done"


def write_summary(monitor_root, rows, feature_root):
    summary = summarize(rows)
    summary["updated_at"] = datetime.now().isoformat(timespec="seconds")
    (monitor_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Kinect RGB Feature Monitor",
        "",
        "Updated: `{}`".format(summary["updated_at"]),
        "",
        "- Target samples: `{}`".format(summary["target_count"]),
        "- Videos existing: `{}`".format(summary["video_exists"]),
        "- Missing: `{}`".format(summary["missing"]),
        "- Generating/not stable: `{}`".format(summary["generating"]),
        "- Ready for feature extraction: `{}`".format(summary["ready"]),
        "- Duration bad: `{}`".format(summary["duration_bad"]),
        "- Feature done: `{}`".format(summary["feature_done"]),
        "",
        "Status CSV: `{}`".format(monitor_root / "status.csv"),
        "Feature root: `{}`".format(feature_root),
    ]
    (monitor_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def run_cycle(args, extractor=None):
    monitor_root = Path(args.monitor_root)
    monitor_root.mkdir(parents=True, exist_ok=True)
    state_path = monitor_root / "state.json"
    previous_state = load_state(state_path)
    rows, next_state = scan(args, previous_state)
    write_csv(monitor_root / "status.csv", rows)
    save_state(state_path, next_state)

    extracted = []
    errors = []
    if args.extract_ready and not args.scan_only:
        ready_rows = [row for row in rows if row["status"] == "ready"]
        if args.max_extract > 0:
            ready_rows = ready_rows[:args.max_extract]
        if ready_rows and extractor is None:
            extractor = FeatureExtractor(args.device, args.batch_size)
        for row in ready_rows:
            try:
                result = process_feature(row, args, extractor)
                extracted.append({"sample_id": row["sample_id"], "result": result})
                print("[FEATURE] {} {}".format(row["sample_id"], result), flush=True)
            except Exception as exc:
                msg = "{}: {}".format(type(exc).__name__, exc)
                errors.append({"sample_id": row["sample_id"], "error": msg})
                print("[FEATURE_ERR] {} {}".format(row["sample_id"], msg), flush=True)

        if extracted or errors:
            with (monitor_root / "feature_last_cycle.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["sample_id", "result"])
                writer.writeheader()
                writer.writerows(extracted)
            with (monitor_root / "feature_errors.csv").open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["sample_id", "error"])
                if handle.tell() == 0:
                    writer.writeheader()
                writer.writerows(errors)

        rows, next_state = scan(args, load_state(state_path))
        write_csv(monitor_root / "status.csv", rows)
        save_state(state_path, next_state)

    summary = write_summary(monitor_root, rows, Path(args.feature_root))
    print(
        "[SUMMARY] target={target_count} exists={video_exists} missing={missing} "
        "generating={generating} ready={ready} duration_bad={duration_bad} "
        "feature_done={feature_done}".format(**summary),
        flush=True,
    )
    return summary, extractor


def main():
    args = parse_args()
    extractor = None
    while True:
        summary, extractor = run_cycle(args, extractor)
        if not args.watch:
            break
        if summary["feature_done"] >= summary["target_count"]:
            print("[DONE] all target samples have features", flush=True)
            break
        time.sleep(args.scan_interval)


if __name__ == "__main__":
    main()
