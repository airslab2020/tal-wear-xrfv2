#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np


BASE = Path(os.environ.get("XRFV2_SOURCE_ROOT", "./data/xrfv2_source"))
IMU_DIR = BASE / "IMU_Command_Time_50Hz_PerSecond"
WIFI_DIMS_CSV = BASE / "wifi_time_dims.csv"
SAMPLE_NAMES_TXT = BASE / "wifi_853_names.txt"
OUTPUT_H5 = BASE / "imu_50hz_853.h5"
OUTPUT_MANIFEST = BASE / "imu_50hz_853_h5_manifest.csv"
OUTPUT_REPORT = BASE / "imu_50hz_853_h5_report.md"

DEVICE_ORDER = ["left_wrist", "right_wrist", "left_phone", "right_phone", "glasses"]
CHANNEL_ORDER = ["加速度X", "加速度Y", "加速度Z", "角速度X", "角速度Y", "角速度Z"]
CHANNEL_INDICES = [2, 3, 4, 5, 6, 7]


def sample_key(name: str) -> tuple[int, str, int]:
    person, scene, trial = name.split("_")
    return int(person), scene, int(trial)


def load_sample_names() -> list[str]:
    names = [Path(line.strip()).stem for line in SAMPLE_NAMES_TXT.read_text().splitlines() if line.strip()]
    return sorted(names, key=sample_key)


def load_wifi_lengths() -> dict[str, int]:
    with WIFI_DIMS_CSV.open(newline="", encoding="utf-8") as handle:
        return {Path(row["file"]).stem: int(row["n_samples"]) for row in csv.DictReader(handle)}


def find_device_files(sample_dir: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sample_dir.glob("*.txt"):
        for position in DEVICE_ORDER:
            if path.name.startswith(position + "_"):
                if position in result:
                    raise ValueError(f"Duplicate {position} file in {sample_dir}")
                result[position] = path
                break
    return result


def read_imu_6ch(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        missing = [name for name in CHANNEL_ORDER if name not in header]
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")
        indices = [header.index(name) for name in CHANNEL_ORDER]
        for row in reader:
            if not row:
                continue
            if len(row) <= max(indices):
                continue
            rows.append([float(row[idx]) for idx in indices])
    return np.asarray(rows, dtype=np.float32)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if OUTPUT_H5.exists():
        raise FileExistsError(f"Output already exists: {OUTPUT_H5}")

    sample_names = load_sample_names()
    wifi_lengths = load_wifi_lengths()
    manifest_rows: list[dict] = []
    string_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(OUTPUT_H5, "w") as h5:
        h5.attrs["description"] = "853 normal IMU samples resampled per-second to 50Hz"
        h5.attrs["shape_convention"] = "[time, device, channel]"
        h5.attrs["sampling_rate_hz"] = 50
        h5.attrs["dtype"] = "float32"
        h5.attrs["device_axis"] = 1
        h5.attrs["channel_axis"] = 2

        h5.create_dataset("sample_names", data=np.asarray(sample_names, dtype=object), dtype=string_dtype)
        h5.create_dataset("device_order", data=np.asarray(DEVICE_ORDER, dtype=object), dtype=string_dtype)
        h5.create_dataset("channel_order", data=np.asarray(CHANNEL_ORDER, dtype=object), dtype=string_dtype)

        samples_group = h5.create_group("samples")

        for idx, sample in enumerate(sample_names, start=1):
            person, scene, trial = sample.split("_")
            sample_dir = IMU_DIR / scene / sample
            if not sample_dir.exists():
                raise FileNotFoundError(f"Missing IMU sample dir: {sample_dir}")

            files_by_device = find_device_files(sample_dir)
            missing_devices = [device for device in DEVICE_ORDER if device not in files_by_device]
            if missing_devices:
                raise FileNotFoundError(f"{sample} missing device files: {missing_devices}")

            arrays = []
            source_files = []
            for device in DEVICE_ORDER:
                source_path = files_by_device[device]
                arr = read_imu_6ch(source_path)
                arrays.append(arr)
                source_files.append(str(source_path.relative_to(BASE)))

            row_counts = [arr.shape[0] for arr in arrays]
            if len(set(row_counts)) != 1:
                raise ValueError(f"{sample} device row counts differ: {row_counts}")
            n = row_counts[0]
            wifi_n = wifi_lengths.get(sample)
            if wifi_n != n:
                raise ValueError(f"{sample} IMU length {n} != WiFi length {wifi_n}")

            stacked = np.stack(arrays, axis=1).astype(np.float32, copy=False)
            if stacked.shape != (n, 5, 6):
                raise ValueError(f"{sample} unexpected shape: {stacked.shape}")

            sample_group = samples_group.create_group(sample)
            chunks = (min(n, 512), 5, 6)
            dataset = sample_group.create_dataset(
                "imu",
                data=stacked,
                dtype=np.float32,
                chunks=chunks,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
            dataset.attrs["sample_id"] = sample
            dataset.attrs["sampling_rate_hz"] = 50
            dataset.attrs["device_order"] = json.dumps(DEVICE_ORDER, ensure_ascii=False)
            dataset.attrs["channel_order"] = json.dumps(CHANNEL_ORDER, ensure_ascii=False)
            dataset.attrs["source_files"] = json.dumps(source_files, ensure_ascii=False)
            sample_group.attrs["scene"] = scene
            sample_group.attrs["person_id"] = int(person)
            sample_group.attrs["trial_id"] = int(trial)
            sample_group.attrs["duration_sec"] = int(n // 50)
            sample_group.attrs["n_samples"] = int(n)

            manifest_rows.append(
                {
                    "sample_id": sample,
                    "scene": scene,
                    "person_id": person,
                    "trial_id": trial,
                    "shape": str(tuple(stacked.shape)),
                    "n_samples": n,
                    "duration_sec": n // 50,
                    "wifi_n_samples": wifi_n,
                    "device_order": "|".join(DEVICE_ORDER),
                    "channel_order": "|".join(CHANNEL_ORDER),
                    "source_files": "|".join(source_files),
                    "h5_dataset": f"/samples/{sample}/imu",
                }
            )

            if idx % 100 == 0:
                print(f"Processed {idx}/{len(sample_names)} samples", flush=True)

    write_csv(OUTPUT_MANIFEST, manifest_rows)

    durations = [int(row["duration_sec"]) for row in manifest_rows]
    report = [
        "# IMU 50Hz 853 H5 Report",
        "",
        f"Output H5: `{OUTPUT_H5}`",
        f"Samples: {len(manifest_rows)}",
        "Dataset path pattern: `/samples/<sample_id>/imu`",
        "Shape convention: `[time, device, channel]`",
        "",
        "## Axes",
        "",
        f"Device order: `{DEVICE_ORDER}`",
        f"Channel order: `{CHANNEL_ORDER}`",
        "",
        "## Validation",
        "",
        "- All 853 samples have 5 device files.",
        "- All devices within a sample have identical row counts.",
        "- Each sample shape is `[50*T, 5, 6]`.",
        "- Each sample IMU length equals the corresponding WiFi H5 first dimension.",
        "",
        "## Duration",
        "",
        f"Min duration: {min(durations)} sec",
        f"Max duration: {max(durations)} sec",
    ]
    OUTPUT_REPORT.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"Done: {OUTPUT_H5}", flush=True)
    print(f"Manifest: {OUTPUT_MANIFEST}", flush=True)
    print(f"Report: {OUTPUT_REPORT}", flush=True)


if __name__ == "__main__":
    main()
