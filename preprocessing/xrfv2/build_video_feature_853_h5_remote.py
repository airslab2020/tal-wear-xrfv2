#!/usr/bin/env python3
import csv
import json
import os
from pathlib import Path

import h5py
import numpy as np


BASE_DIR = Path(os.environ.get("XRFV2_SOURCE_ROOT", "./data/xrfv2_source"))
FEATURE_ROOT = BASE_DIR / "kinect_rgb_secondwise_feature"
MONITOR_ROOT = BASE_DIR / "kinect_rgb_feature_monitor"
SAMPLE_ORDER_PATH = MONITOR_ROOT / "video_h5_sample_order_853.txt"
OUTPUT_H5 = BASE_DIR / "video_feature_853_video_aligned.h5"
MANIFEST_CSV = MONITOR_ROOT / "video_feature_853_h5_manifest.csv"
REPORT_MD = MONITOR_ROOT / "video_feature_853_h5_report.md"

FEATURE_FILE = "scheme2_15frames_secondwise.npy"
TIMESTAMP_FILE = "scheme2_15frames_secondwise_timestamps.npy"


def scene_from_sample(sample_id):
    return sample_id.split("_")[1]


def read_sample_order(path):
    samples = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if value:
                samples.append(value[:-3] if value.endswith(".h5") else value)
    return samples


def string_dataset(file_handle, name, values):
    dtype = h5py.string_dtype(encoding="utf-8")
    return file_handle.create_dataset(name, data=np.asarray(values, dtype=object), dtype=dtype)


def atomic_write_h5(samples):
    tmp_path = Path(str(OUTPUT_H5) + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    manifest_rows = []
    with h5py.File(tmp_path, "w") as out:
        out.attrs["source_feature_root"] = str(FEATURE_ROOT)
        out.attrs["feature_file"] = FEATURE_FILE
        out.attrs["shape_convention"] = "[time_second, feature_dim]"
        out.attrs["dataset_path_pattern"] = "/samples/<sample_id>/video_feature"
        out.attrs["feature_extractor"] = "torchvision.models.video.swin3d_b"
        out.attrs["feature_scheme"] = "scheme2_15frames_secondwise"
        out.attrs["sample_count"] = len(samples)

        string_dataset(out, "sample_names", samples)
        string_dataset(out, "feature_name", ["swin3d_b_scheme2_15frames_secondwise"])

        sample_group = out.create_group("samples")

        for idx, sample_id in enumerate(samples, 1):
            scene = scene_from_sample(sample_id)
            sample_dir = FEATURE_ROOT / scene / sample_id
            feature_path = sample_dir / FEATURE_FILE
            ts_path = sample_dir / TIMESTAMP_FILE
            meta_path = sample_dir / "feature_meta.json"

            if not feature_path.exists():
                raise FileNotFoundError("missing feature npy: {}".format(feature_path))

            arr = np.load(str(feature_path), mmap_mode="r")
            if arr.ndim != 2 or arr.shape[1] != 1024:
                raise RuntimeError("bad feature shape for {}: {}".format(sample_id, arr.shape))

            group = sample_group.create_group(sample_id)
            n_rows = int(arr.shape[0])
            chunks = (min(n_rows, 64), 1024)
            ds = group.create_dataset(
                "video_feature",
                data=arr,
                dtype=np.float32,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
                chunks=chunks,
            )
            ds.attrs["shape_convention"] = "[T, 1024]"

            timestamps_shape = ""
            if ts_path.exists():
                ts = np.load(str(ts_path), mmap_mode="r")
                if ts.shape[0] == n_rows:
                    group.create_dataset(
                        "timestamps_sec",
                        data=ts.astype(np.float32),
                        dtype=np.float32,
                        compression="gzip",
                        compression_opts=4,
                        shuffle=True,
                        chunks=(min(n_rows, 128),),
                    )
                    timestamps_shape = str(tuple(ts.shape))

            meta = {}
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    meta = {}
            for key in [
                "command_duration_sec",
                "duration_override",
                "clip_len",
                "target_fps",
                "video_fps",
                "frame_count",
                "video_path",
                "scheme",
            ]:
                if key in meta:
                    group.attrs[key] = meta[key]

            manifest_rows.append({
                "sample_id": sample_id,
                "scene": scene,
                "shape": str(tuple(arr.shape)),
                "n_seconds": n_rows,
                "feature_dim": int(arr.shape[1]),
                "dtype": str(arr.dtype),
                "feature_path": str(feature_path),
                "h5_dataset": "/samples/{}/video_feature".format(sample_id),
                "timestamps_shape": timestamps_shape,
                "duration_override": meta.get("duration_override", ""),
            })

            if idx % 50 == 0 or idx == len(samples):
                print("[video_h5] packed {}/{}".format(idx, len(samples)), flush=True)

    os.replace(str(tmp_path), str(OUTPUT_H5))
    return manifest_rows


def write_manifest(rows):
    fields = [
        "sample_id", "scene", "shape", "n_seconds", "feature_dim", "dtype",
        "feature_path", "h5_dataset", "timestamps_shape", "duration_override",
    ]
    with MANIFEST_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(rows):
    seconds = [int(row["n_seconds"]) for row in rows]
    overrides = [row for row in rows if row["duration_override"]]
    lines = [
        "# Video Feature 853 H5 Report",
        "",
        "Output H5: `{}`".format(OUTPUT_H5),
        "Samples: `{}`".format(len(rows)),
        "Dataset path: `/samples/<sample_id>/video_feature`",
        "Shape convention: `[T, 1024]`",
        "Dtype: `float32`",
        "Feature extractor: `swin3d_b`",
        "Scheme: `scheme2_15frames_secondwise`",
        "",
        "## Validation",
        "",
        "- All sample feature npy files were included.",
        "- Every `video_feature` dataset is two-dimensional.",
        "- Every feature dimension is `1024`.",
        "",
        "## Duration",
        "",
        "Min T: `{}` sec".format(min(seconds)),
        "Max T: `{}` sec".format(max(seconds)),
        "Duration-overridden samples: `{}`".format(len(overrides)),
        "",
        "Manifest: `{}`".format(MANIFEST_CSV),
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    samples = read_sample_order(SAMPLE_ORDER_PATH)
    if len(samples) != 853:
        raise RuntimeError("expected 853 samples, got {}".format(len(samples)))
    MONITOR_ROOT.mkdir(parents=True, exist_ok=True)
    rows = atomic_write_h5(samples)
    write_manifest(rows)
    write_report(rows)
    print("output_h5:", OUTPUT_H5, flush=True)
    print("manifest:", MANIFEST_CSV, flush=True)
    print("report:", REPORT_MD, flush=True)


if __name__ == "__main__":
    main()
