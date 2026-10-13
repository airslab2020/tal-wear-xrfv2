#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path


BASE = Path(os.environ.get("XRFV2_SOURCE_ROOT", "./data/xrfv2_source"))
INPUT_MANIFEST = BASE / "imu_command_time_split_manifest.csv"
OUTPUT_DIR = BASE / "IMU_Command_Time_50Hz_PerSecond"
OUTPUT_MANIFEST = BASE / "imu_50hz_per_second_manifest.csv"
OUTPUT_SAMPLE_SUMMARY = BASE / "imu_50hz_per_second_sample_summary.csv"
OUTPUT_REPORT = BASE / "imu_50hz_per_second_report.md"

TIMESTAMP_FORMATS = ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S")


def parse_timestamp(value: str) -> datetime:
    value = value.strip()
    for fmt in TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    raise ValueError(f"Unsupported timestamp: {value!r}")


def parse_hms_seconds(value: str) -> int:
    h, m, s = value.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def seconds_of_day(dt: datetime) -> float:
    return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1_000_000


def format_target_timestamp(base_date: datetime, target_sec: float) -> str:
    day_offset = math.floor(target_sec / 86400)
    sec_in_day = target_sec - day_offset * 86400
    target = datetime.combine(base_date.date(), datetime.min.time()) + timedelta(days=day_offset, seconds=sec_in_day)
    return target.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def load_source_rows(path: Path, command_start_sec: float, command_end_sec: float) -> tuple[list[str], str, list[float], list[list[float]], datetime]:
    times: list[float] = []
    values: list[list[float]] = []
    device_id = ""
    first_dt: datetime | None = None

    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        header_line = handle.readline().rstrip("\n")
        header = header_line.split("\t")
        if len(header) < 3:
            raise ValueError(f"Unexpected header in {path}: {header_line!r}")

        for line in handle:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != len(header):
                continue
            dt = parse_timestamp(parts[0])
            if first_dt is None:
                first_dt = dt
            if not device_id:
                device_id = parts[1].strip()
            sec = seconds_of_day(dt)
            if command_end_sec >= 86400 and sec < command_start_sec:
                sec += 86400
            try:
                numeric = [float(x) for x in parts[2:]]
            except ValueError:
                continue
            times.append(sec)
            values.append(numeric)

    if first_dt is None:
        raise ValueError(f"No source data rows in {path}")
    return header, device_id, times, values, first_dt


def interpolate_value(t: float, xs: list[float], ys: list[list[float]]) -> list[float]:
    if not xs:
        raise ValueError("No source points for interpolation")
    if len(xs) == 1 or t <= xs[0]:
        return ys[0]
    if t >= xs[-1]:
        return ys[-1]

    lo = 0
    hi = len(xs) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if xs[mid] <= t:
            lo = mid
        else:
            hi = mid

    x0, x1 = xs[lo], xs[hi]
    y0, y1 = ys[lo], ys[hi]
    if x1 == x0:
        return y0
    ratio = (t - x0) / (x1 - x0)
    return [a + (b - a) * ratio for a, b in zip(y0, y1)]


def rows_for_second(
    second_start: float,
    all_times: list[float],
    all_values: list[list[float]],
    start_index: int,
) -> tuple[list[float], list[list[float]], int]:
    second_end = second_start + 1.0
    n = len(all_times)
    while start_index < n and all_times[start_index] < second_start:
        start_index += 1

    end_index = start_index
    while end_index < n and all_times[end_index] < second_end:
        end_index += 1

    return all_times[start_index:end_index], all_values[start_index:end_index], start_index


def nearest_global_value(t: float, all_times: list[float], all_values: list[list[float]]) -> list[float]:
    if not all_times:
        raise ValueError("No global source points")
    if t <= all_times[0]:
        return all_values[0]
    if t >= all_times[-1]:
        return all_values[-1]
    lo = 0
    hi = len(all_times) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if all_times[mid] <= t:
            lo = mid
        else:
            hi = mid
    return all_values[lo] if abs(t - all_times[lo]) <= abs(all_times[hi] - t) else all_values[hi]


def resample_file(row: dict[str, str]) -> dict[str, str | int | float]:
    source_path = BASE / row["output_file"]
    output_path = OUTPUT_DIR / row["scene"] / row["command_stem"] / source_path.name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command_start_sec = float(parse_hms_seconds(row["command_start_time"]))
    command_end_sec = float(parse_hms_seconds(row["command_end_time"]))
    if command_end_sec < command_start_sec:
        command_end_sec += 86400
    duration_sec = int(command_end_sec - command_start_sec)
    expected_rows = duration_sec * 50

    header, device_id, times, values, first_dt = load_source_rows(source_path, command_start_sec, command_end_sec)

    # Keep only monotonic rows. The original files are usually ordered, but this
    # protects interpolation from occasional timestamp disorder.
    ordered = sorted(zip(times, values), key=lambda item: item[0])
    times = [item[0] for item in ordered]
    values = [item[1] for item in ordered]

    seconds_with_zero_points = 0
    seconds_with_one_point = 0
    seconds_with_under_40_points = 0
    source_points_by_second: Counter[int] = Counter()
    output_rows = 0
    cursor = 0

    with output_path.open("w", encoding="utf-8", newline="") as out:
        out.write("\t".join(header) + "\n")
        for second_idx in range(duration_sec):
            second_start = command_start_sec + second_idx
            sec_times, sec_values, cursor = rows_for_second(second_start, times, values, cursor)
            source_points_by_second[len(sec_times)] += 1
            if len(sec_times) == 0:
                seconds_with_zero_points += 1
            elif len(sec_times) == 1:
                seconds_with_one_point += 1
            if len(sec_times) < 40:
                seconds_with_under_40_points += 1

            for tick in range(50):
                target_t = second_start + tick / 50.0
                if len(sec_times) >= 2:
                    vals = interpolate_value(target_t, sec_times, sec_values)
                elif len(sec_times) == 1:
                    vals = sec_values[0]
                else:
                    vals = nearest_global_value(target_t, times, values)
                timestamp = format_target_timestamp(first_dt, target_t)
                numeric = "\t".join(f"{v:.6f}" for v in vals)
                out.write(f"{timestamp}\t{device_id}\t{numeric}\n")
                output_rows += 1

    return {
        "command_stem": row["command_stem"],
        "scene": row["scene"],
        "device_id": row["device_id"],
        "short_device_id": row["short_device_id"],
        "binding_position": row["binding_position"],
        "source_file": str(source_path.relative_to(BASE)),
        "output_file": str(output_path.relative_to(BASE)),
        "command_start_time": row["command_start_time"],
        "command_end_time": row["command_end_time"],
        "command_duration_sec": duration_sec,
        "source_rows": len(times),
        "expected_50hz_rows": expected_rows,
        "output_rows": output_rows,
        "seconds_with_zero_source_points": seconds_with_zero_points,
        "seconds_with_one_source_point": seconds_with_one_point,
        "seconds_with_under_40_source_points": seconds_with_under_40_points,
        "min_source_points_in_one_second": min(source_points_by_second) if source_points_by_second else "",
        "max_source_points_in_one_second": max(source_points_by_second) if source_points_by_second else "",
        "output_matches_expected": str(output_rows == expected_rows),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_sample_summary(rows: list[dict]) -> list[dict]:
    by_sample: defaultdict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_sample[str(row["command_stem"])].append(row)

    summary: list[dict] = []
    for stem, items in sorted(
        by_sample.items(),
        key=lambda item: (item[0].endswith("_bak"), int(item[0].split("_")[0]), item[0].split("_")[1], int(item[0].split("_")[2])),
    ):
        normal = not stem.endswith("_bak")
        summary.append(
            {
                "command_stem": stem,
                "scene": items[0]["scene"],
                "is_normal_853_sample": str(normal),
                "device_files": len(items),
                "command_duration_sec": items[0]["command_duration_sec"],
                "expected_rows_per_device": items[0]["expected_50hz_rows"],
                "total_output_rows": sum(int(item["output_rows"]) for item in items),
                "devices": ";".join(sorted(f"{item['binding_position']}:{item['short_device_id']}" for item in items)),
                "seconds_with_zero_source_points_total": sum(int(item["seconds_with_zero_source_points"]) for item in items),
                "seconds_with_one_source_point_total": sum(int(item["seconds_with_one_source_point"]) for item in items),
                "seconds_with_under_40_source_points_total": sum(int(item["seconds_with_under_40_source_points"]) for item in items),
                "all_outputs_match_expected": str(all(item["output_matches_expected"] == "True" for item in items)),
            }
        )

    write_csv(OUTPUT_SAMPLE_SUMMARY, summary)
    return summary


def write_report(rows: list[dict], sample_summary: list[dict]) -> None:
    normal_rows = [row for row in rows if not str(row["command_stem"]).endswith("_bak")]
    normal_samples = [row for row in sample_summary if row["is_normal_853_sample"] == "True"]
    device_counts = Counter(int(row["device_files"]) for row in sample_summary)
    normal_device_counts = Counter(int(row["device_files"]) for row in normal_samples)
    zero_seconds_total = sum(int(row["seconds_with_zero_source_points"]) for row in rows)
    one_seconds_total = sum(int(row["seconds_with_one_source_point"]) for row in rows)
    under_40_total = sum(int(row["seconds_with_under_40_source_points"]) for row in rows)

    lines = [
        "# IMU 50Hz Per-Second Resampling Report",
        "",
        f"Input manifest: `{INPUT_MANIFEST}`",
        f"Output directory: `{OUTPUT_DIR}`",
        "",
        "## Method",
        "",
        "- Each command-time sample/device file is resampled independently.",
        "- Each natural one-second window is processed independently.",
        "- Every second produces exactly 50 target rows at offsets `0.00, 0.02, ..., 0.98`.",
        "- If a second has two or more source points, numeric columns are linearly interpolated within that second.",
        "- If a second has one source point, that value is repeated for the 50 target rows in that second.",
        "- If a second has zero source points, the nearest source value from the same device sequence is used.",
        "",
        "## Summary",
        "",
        f"Resampled device files: {len(rows)}",
        f"Resampled normal 853-sample device files: {len(normal_rows)}",
        f"Sample directories represented: {len(sample_summary)}",
        f"Normal sample directories represented: {len(normal_samples)}",
        f"Device-file count distribution across all sample directories: {dict(sorted(device_counts.items()))}",
        f"Device-file count distribution across normal sample directories: {dict(sorted(normal_device_counts.items()))}",
        f"All device outputs match expected `duration_sec * 50`: {all(row['output_matches_expected'] == 'True' for row in rows)}",
        f"Seconds with zero source points: {zero_seconds_total}",
        f"Seconds with one source point: {one_seconds_total}",
        f"Seconds with fewer than 40 source points: {under_40_total}",
    ]

    (OUTPUT_REPORT).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Output directory already exists: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True)

    manifest_rows = list(csv.DictReader(INPUT_MANIFEST.open(newline="", encoding="utf-8")))
    work_rows = [row for row in manifest_rows if int(row["line_count"]) > 0]
    print(f"Input manifest rows: {len(manifest_rows)}", flush=True)
    print(f"Non-empty device files to resample: {len(work_rows)}", flush=True)

    output_rows: list[dict] = []
    for idx, row in enumerate(work_rows, start=1):
        output_rows.append(resample_file(row))
        if idx % 250 == 0:
            print(f"Processed {idx}/{len(work_rows)} device files", flush=True)

    write_csv(OUTPUT_MANIFEST, output_rows)
    sample_summary = write_sample_summary(output_rows)
    write_report(output_rows, sample_summary)
    print(f"Done. Resampled files: {len(output_rows)}", flush=True)
    print(f"Manifest: {OUTPUT_MANIFEST}", flush=True)
    print(f"Report: {OUTPUT_REPORT}", flush=True)


if __name__ == "__main__":
    main()
