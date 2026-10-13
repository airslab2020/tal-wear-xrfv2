#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


BASE = Path(os.environ.get("XRFV2_SOURCE_ROOT", "./data/xrfv2_source"))
RAW_IMU_DIR = BASE / "imu"
COMMAND_TIME_DIR = BASE / "command_time"
DEVICE_SPLIT_DIR = BASE / "IMU_Device_Split"
COMMAND_SPLIT_DIR = BASE / "IMU_Command_Time_Split"

DEVICE_BINDINGS = {
    "6770": ("WT5300006770", "left_wrist"),
    "6649": ("WT5300006649", "right_wrist"),
    "6748": ("WT5300006748", "left_phone"),
    "6678": ("WT5300006678", "right_phone"),
    "6735": ("WT5300006735", "glasses"),
    "6751": ("WT5300006751", "left_wrist"),
    "6676": ("WT5300006676", "right_wrist"),
    "5126": ("WT5300005126", "left_phone"),
    "6633": ("WT5300006633", "right_phone"),
    "6727": ("WT5300006727", "glasses"),
}

RANGE_RE = re.compile(r"^(?P<person>\d+)_(?P<code>\d+)_(?P<start>\d+)-(?P<end>\d+)\.txt$")
TIMESTAMP_RE = re.compile(r"^\s*(\d{4}-\d{2}-\d{2}) (\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?")


@dataclass(frozen=True)
class CommandWindow:
    command_stem: str
    base_stem: str
    scene: str
    person: str
    trial: int
    json_path: Path
    start_time: str
    end_time: str
    start_sec: float
    end_sec: float
    duration_sec: int


def short_device_id(device_id: str) -> str:
    device_id = device_id.strip()
    if device_id.startswith("WT530000"):
        return device_id.removeprefix("WT530000")
    return device_id


def safe_device_filename(device_id: str) -> str:
    short = short_device_id(device_id)
    full, position = DEVICE_BINDINGS.get(short, (device_id, "unknown_position"))
    return f"{position}_{full}.txt"


def parse_hms_seconds(value: str) -> int:
    h, m, s = value.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def parse_line_seconds(line: str) -> float | None:
    match = TIMESTAMP_RE.match(line)
    if not match:
        return None
    _, hour, minute, second, frac = match.groups()
    sec = int(hour) * 3600 + int(minute) * 60 + int(second)
    if frac:
        sec += float(f"0.{frac}")
    return sec


def command_base_stem(command_stem: str) -> str:
    return command_stem.removesuffix("_bak")


def parse_sample_stem(stem: str) -> tuple[str, str, int]:
    person, scene, trial = stem.split("_")
    return person, scene, int(trial)


def load_command_windows() -> list[CommandWindow]:
    windows: list[CommandWindow] = []
    for path in sorted(COMMAND_TIME_DIR.glob("*/*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list) or not data:
            raise ValueError(f"{path} is not a non-empty command list")

        command_stem = path.stem
        base_stem = command_base_stem(command_stem)
        person, scene, trial = parse_sample_stem(base_stem)

        start_time = data[0]["start_time"]
        end_time = data[-1]["end_time"]
        start_sec = parse_hms_seconds(start_time)
        end_sec = parse_hms_seconds(end_time)
        if end_sec < start_sec:
            end_sec += 86400

        windows.append(
            CommandWindow(
                command_stem=command_stem,
                base_stem=base_stem,
                scene=scene,
                person=person,
                trial=trial,
                json_path=path,
                start_time=start_time,
                end_time=end_time,
                start_sec=float(start_sec),
                end_sec=float(end_sec),
                duration_sec=int(end_sec - start_sec),
            )
        )
    return windows


def prepare_output_dirs() -> None:
    for path in (DEVICE_SPLIT_DIR, COMMAND_SPLIT_DIR):
        if path.exists():
            raise FileExistsError(f"Output directory already exists: {path}")
        path.mkdir(parents=True)


def split_raw_imu_by_device() -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    raw_files = sorted(RAW_IMU_DIR.glob("*/*.txt"), key=lambda p: str(p.relative_to(RAW_IMU_DIR)))

    for raw_path in raw_files:
        rel = raw_path.relative_to(RAW_IMU_DIR)
        out_dir = DEVICE_SPLIT_DIR / rel.parent / raw_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        handles = {}
        counts: Counter[str] = Counter()
        total_lines = 0
        bad_lines = 0

        with raw_path.open("r", encoding="utf-8", errors="replace", newline="") as src:
            header = src.readline()
            for line in src:
                if not line.strip():
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    bad_lines += 1
                    continue
                device_id = parts[1].strip()
                if not device_id:
                    bad_lines += 1
                    continue

                if device_id not in handles:
                    out_path = out_dir / safe_device_filename(device_id)
                    handle = out_path.open("w", encoding="utf-8", newline="")
                    handle.write(header)
                    handles[device_id] = (handle, out_path)
                handles[device_id][0].write(line)
                counts[device_id] += 1
                total_lines += 1

        for handle, _ in handles.values():
            handle.close()

        for device_id, line_count in sorted(counts.items(), key=lambda item: short_device_id(item[0])):
            short = short_device_id(device_id)
            _, position = DEVICE_BINDINGS.get(short, (device_id, "unknown_position"))
            rows.append(
                {
                    "raw_file": str(rel),
                    "split_file": str(handles[device_id][1].relative_to(BASE)),
                    "device_id": device_id,
                    "short_device_id": short,
                    "binding_position": position,
                    "line_count": line_count,
                    "raw_total_data_lines": total_lines,
                    "raw_bad_lines": bad_lines,
                }
            )

    return rows


def build_windows_by_base(windows: list[CommandWindow]) -> dict[str, list[CommandWindow]]:
    by_base: dict[str, list[CommandWindow]] = defaultdict(list)
    for window in windows:
        by_base[window.base_stem].append(window)
    return by_base


def parse_range_file(path: Path) -> tuple[str, str, int, int] | None:
    match = RANGE_RE.match(path.name)
    if not match:
        return None
    person = match.group("person")
    code = match.group("code")
    start = int(match.group("start"))
    end = int(match.group("end"))
    return person, code, start, end


def sample_base_stems_for_range(scene: str, person: str, start: int, end: int) -> list[str]:
    return [f"{person}_{scene}_{idx}" for idx in range(start, end + 1)]


def output_sample_file(window: CommandWindow, split_file: Path) -> Path:
    short = split_file.stem.split("_WT530000")[-1]
    position = split_file.stem.split("_WT530000")[0]
    full = f"WT530000{short}" if short.isdigit() else short
    sample_dir = COMMAND_SPLIT_DIR / window.scene / window.command_stem
    sample_dir.mkdir(parents=True, exist_ok=True)
    return sample_dir / f"{position}_{full}.txt"


def split_device_files_by_command(windows: list[CommandWindow]) -> tuple[list[dict[str, str | int | float]], list[dict[str, str | int]]]:
    windows_by_base = build_windows_by_base(windows)
    rows: list[dict[str, str | int | float]] = []
    unmatched_ranges: list[dict[str, str | int]] = []

    split_files = sorted(DEVICE_SPLIT_DIR.glob("*/*/*.txt"), key=lambda p: str(p.relative_to(DEVICE_SPLIT_DIR)))

    for split_file in split_files:
        rel_parts = split_file.relative_to(DEVICE_SPLIT_DIR).parts
        scene = rel_parts[0]
        range_dir = rel_parts[1]
        range_info = parse_range_file(Path(range_dir + ".txt"))
        if range_info is None:
            unmatched_ranges.append({"split_file": str(split_file.relative_to(BASE)), "reason": "unparsed_range_name"})
            continue
        person, _, start, end = range_info
        base_stems = sample_base_stems_for_range(scene, person, start, end)
        active_windows = [w for stem in base_stems for w in windows_by_base.get(stem, [])]
        if not active_windows:
            unmatched_ranges.append({"split_file": str(split_file.relative_to(BASE)), "reason": "no_command_window"})
            continue

        active_windows.sort(key=lambda w: (w.start_sec, w.end_sec, w.command_stem))

        out_handles = {}
        line_counts: Counter[str] = Counter()
        first_ts: dict[str, float] = {}
        last_ts: dict[str, float] = {}
        bad_timestamp_lines = 0
        total_input_lines = 0

        with split_file.open("r", encoding="utf-8", errors="replace", newline="") as src:
            header = src.readline()
            for line in src:
                if not line.strip():
                    continue
                total_input_lines += 1
                line_sec = parse_line_seconds(line)
                if line_sec is None:
                    bad_timestamp_lines += 1
                    continue

                for window in active_windows:
                    adjusted_sec = line_sec
                    if window.end_sec >= 86400 and adjusted_sec < window.start_sec:
                        adjusted_sec += 86400
                    if window.start_sec <= adjusted_sec <= window.end_sec:
                        if window.command_stem not in out_handles:
                            out_path = output_sample_file(window, split_file)
                            handle = out_path.open("w", encoding="utf-8", newline="")
                            handle.write(header)
                            out_handles[window.command_stem] = (handle, out_path)
                        out_handles[window.command_stem][0].write(line)
                        line_counts[window.command_stem] += 1
                        first_ts.setdefault(window.command_stem, adjusted_sec)
                        last_ts[window.command_stem] = adjusted_sec
                        break

        for handle, _ in out_handles.values():
            handle.close()

        short = split_file.stem.split("_WT530000")[-1]
        position = split_file.stem.split("_WT530000")[0]
        full = f"WT530000{short}" if short.isdigit() else short

        for window in active_windows:
            out_path = output_sample_file(window, split_file)
            count = line_counts.get(window.command_stem, 0)
            rows.append(
                {
                    "command_stem": window.command_stem,
                    "base_stem": window.base_stem,
                    "scene": window.scene,
                    "source_split_file": str(split_file.relative_to(BASE)),
                    "output_file": str(out_path.relative_to(BASE)),
                    "device_id": full,
                    "short_device_id": short,
                    "binding_position": position,
                    "command_start_time": window.start_time,
                    "command_end_time": window.end_time,
                    "command_duration_sec": window.duration_sec,
                    "line_count": count,
                    "first_sample_sec_of_day": first_ts.get(window.command_stem, ""),
                    "last_sample_sec_of_day": last_ts.get(window.command_stem, ""),
                    "input_device_file_lines": total_input_lines,
                    "bad_timestamp_lines": bad_timestamp_lines,
                }
            )

    return rows, unmatched_ranges


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(device_rows: list[dict], command_rows: list[dict], unmatched_ranges: list[dict], windows: list[CommandWindow]) -> None:
    device_sets = Counter()
    files_to_devices: defaultdict[str, set[str]] = defaultdict(set)
    for row in device_rows:
        files_to_devices[str(row["raw_file"])].add(str(row["short_device_id"]))
    for devices in files_to_devices.values():
        device_sets[tuple(sorted(devices))] += 1

    command_counts: defaultdict[str, int] = defaultdict(int)
    command_nonempty_counts: defaultdict[str, int] = defaultdict(int)
    command_zero_files: list[dict] = []
    for row in command_rows:
        stem = str(row["command_stem"])
        command_counts[stem] += 1
        if int(row["line_count"]) > 0:
            command_nonempty_counts[stem] += 1
        else:
            command_zero_files.append(row)

    command_stems = {w.command_stem for w in windows}
    missing_command_dirs = sorted(command_stems - set(command_counts))
    complete_five = sum(1 for stem in command_stems if command_nonempty_counts.get(stem, 0) == 5)
    complete_four = sum(1 for stem in command_stems if command_nonempty_counts.get(stem, 0) == 4)
    other_counts = Counter(command_nonempty_counts.get(stem, 0) for stem in command_stems)

    lines = [
        "# IMU Split Report",
        "",
        f"Raw IMU directory: `{RAW_IMU_DIR}`",
        f"Device split output: `{DEVICE_SPLIT_DIR}`",
        f"Command-time split output: `{COMMAND_SPLIT_DIR}`",
        "",
        "## Step 1: Split By Device",
        "",
        f"Raw IMU txt files: {len(files_to_devices)}",
        f"Device-split txt files: {len(device_rows)}",
        "",
        "| Device set observed in raw file | Raw file count |",
        "|---|---:|",
    ]
    for device_set, count in device_sets.most_common():
        lines.append(f"| {', '.join(device_set)} | {count} |")

    lines.extend(
        [
            "",
            "## Step 2: Split By Command Time",
            "",
            f"Command JSON files: {len(windows)}",
            f"Command-split output txt rows in manifest: {len(command_rows)}",
            f"Command samples with 5 non-empty device files: {complete_five}",
            f"Command samples with 4 non-empty device files: {complete_four}",
            f"Command samples with other non-empty device-file counts: {dict(sorted(other_counts.items()))}",
            f"Command samples with no matched source range: {len(missing_command_dirs)}",
            f"Zero-line output entries: {len(command_zero_files)}",
            f"Device split ranges without command windows: {len(unmatched_ranges)}",
            "",
            "## Notes",
            "",
            "- Output sample directory names keep `_bak` when the source command JSON has `_bak`.",
            "- Matching to IMU range files uses the command stem without `_bak`.",
            "- The command window is from the first action `start_time` to the last action `end_time`, inclusive.",
        ]
    )

    if command_zero_files:
        lines.extend(["", "## Zero-Line Output Entries", "", "| Command | Device | Source split file |", "|---|---|---|"])
        for row in command_zero_files[:200]:
            lines.append(f"| {row['command_stem']} | {row['device_id']} | {row['source_split_file']} |")
        if len(command_zero_files) > 200:
            lines.append(f"| ... | ... | {len(command_zero_files) - 200} more |")

    if unmatched_ranges:
        lines.extend(["", "## Device Split Ranges Without Command Windows", "", "| Split file | Reason |", "|---|---|"])
        for row in unmatched_ranges:
            lines.append(f"| {row['split_file']} | {row['reason']} |")

    (BASE / "imu_split_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    prepare_output_dirs()
    windows = load_command_windows()
    print(f"Loaded command windows: {len(windows)}", flush=True)

    print("Step 1/2: splitting raw IMU files by device...", flush=True)
    device_rows = split_raw_imu_by_device()
    write_csv(BASE / "imu_device_split_manifest_local.csv", device_rows)
    print(f"Device split files: {len(device_rows)}", flush=True)

    print("Step 2/2: splitting device files by command_time windows...", flush=True)
    command_rows, unmatched_ranges = split_device_files_by_command(windows)
    write_csv(BASE / "imu_command_time_split_manifest.csv", command_rows)
    write_csv(BASE / "imu_command_time_unmatched_ranges.csv", unmatched_ranges)
    write_report(device_rows, command_rows, unmatched_ranges, windows)
    print(f"Command-time split manifest rows: {len(command_rows)}", flush=True)
    print(f"Unmatched split ranges: {len(unmatched_ranges)}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
