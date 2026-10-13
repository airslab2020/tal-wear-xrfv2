#!/usr/bin/env python3
import argparse
import csv
import json
import os
from pathlib import Path

import h5py


BASE = Path(os.environ.get("XRFV2_SOURCE_ROOT", "./data/xrfv2_source"))
OVERRIDE_PATH = BASE / "kinect_rgb_feature_monitor" / "duration_overrides.json"

JOBS = [
    {
        "modality": "wifi",
        "input": BASE / "wifi_50hz_853.h5",
        "output": BASE / "wifi_50hz_853_video_aligned.h5",
        "datasets": ["amp", "pha"],
        "rate": 50,
    },
    {
        "modality": "imu",
        "input": BASE / "imu_50hz_853.h5",
        "output": BASE / "imu_50hz_853_video_aligned.h5",
        "datasets": ["imu"],
        "rate": 50,
    },
    {
        "modality": "airpods",
        "input": BASE / "airpods_25hz_853.h5",
        "output": BASE / "airpods_25hz_853_video_aligned.h5",
        "datasets": ["airpods"],
        "rate": 25,
    },
]


def decode_name(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def copy_attrs(src, dst):
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def create_like(dst_group, name, src_ds, n_rows):
    chunks = src_ds.chunks
    if chunks is not None:
        chunks = (min(chunks[0], n_rows),) + tuple(chunks[1:])
    kwargs = {
        "dtype": src_ds.dtype,
        "shape": (n_rows,) + src_ds.shape[1:],
        "compression": src_ds.compression,
        "compression_opts": src_ds.compression_opts,
        "shuffle": src_ds.shuffle,
        "fletcher32": src_ds.fletcher32,
        "chunks": chunks,
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    dst_ds = dst_group.create_dataset(name, **kwargs)
    dst_ds[...] = src_ds[:n_rows]
    copy_attrs(src_ds, dst_ds)
    return dst_ds


def process_job(job, overrides, manifest_rows):
    input_path = job["input"]
    output_path = job["output"]
    tmp_path = Path(str(output_path) + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    with h5py.File(input_path, "r") as src, h5py.File(tmp_path, "w") as dst:
        copy_attrs(src, dst)
        dst.attrs["video_alignment_duration_overrides"] = json.dumps(overrides, ensure_ascii=False)
        dst.attrs["source_h5"] = str(input_path)

        for key in src.keys():
            if key != "samples":
                src.copy(src[key], dst, name=key)

        dst_samples = dst.create_group("samples")
        copy_attrs(src["samples"], dst_samples)

        sample_names = [decode_name(x) for x in src["sample_names"][()]]
        total = len(sample_names)
        for idx, sample_id in enumerate(sample_names, 1):
            if sample_id not in overrides:
                src.copy(src["samples"][sample_id], dst_samples, name=sample_id)
                if idx % 50 == 0 or idx == total:
                    print("[{}] copied {}/{}".format(job["modality"], idx, total), flush=True)
                continue

            new_t = int(overrides[sample_id]["duration_sec"])
            new_rows = new_t * int(job["rate"])
            src_group = src["samples"][sample_id]
            dst_group = dst_samples.create_group(sample_id)
            copy_attrs(src_group, dst_group)

            old_rows = None
            shapes = {}
            for dataset_name in job["datasets"]:
                src_ds = src_group[dataset_name]
                old_rows = src_ds.shape[0] if old_rows is None else old_rows
                if new_rows > src_ds.shape[0]:
                    raise RuntimeError(
                        "{} {} requests {} rows but source has {}".format(
                            job["modality"], sample_id, new_rows, src_ds.shape[0]
                        )
                    )
                dst_ds = create_like(dst_group, dataset_name, src_ds, new_rows)
                shapes[dataset_name] = str(tuple(dst_ds.shape))

            dst_group.attrs["video_aligned_trimmed"] = 1
            dst_group.attrs["video_aligned_duration_sec"] = new_t
            dst_group.attrs["video_aligned_reason"] = overrides[sample_id].get("reason", "")
            dst_group.attrs["video_aligned_action_count"] = int(overrides[sample_id].get("action_count", 0))

            manifest_rows.append({
                "modality": job["modality"],
                "sample_id": sample_id,
                "rate_hz": job["rate"],
                "old_rows": old_rows,
                "old_duration_sec": old_rows / float(job["rate"]),
                "new_rows": new_rows,
                "new_duration_sec": new_t,
                "trimmed_rows": old_rows - new_rows,
                "trimmed_sec": old_rows / float(job["rate"]) - new_t,
                "reason": overrides[sample_id].get("reason", ""),
                "action_count": overrides[sample_id].get("action_count", ""),
                "end_time": overrides[sample_id].get("end_time", ""),
                "shapes": json.dumps(shapes, ensure_ascii=False),
            })
            print(
                "[{}] trimmed {}: {} -> {} rows".format(
                    job["modality"], sample_id, old_rows, new_rows
                ),
                flush=True,
            )

    os.replace(str(tmp_path), str(output_path))
    print("[{}] wrote {}".format(job["modality"], output_path), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modalities", nargs="+", choices=["imu", "wifi", "airpods"], default=["imu"])
    args = parser.parse_args()
    overrides = json.loads(OVERRIDE_PATH.read_text(encoding="utf-8"))
    manifest_rows = []
    selected_jobs = [job for job in JOBS if job["modality"] in args.modalities]
    for job in selected_jobs:
        process_job(job, overrides, manifest_rows)

    manifest_path = BASE / "video_aligned_modality_trim_manifest.csv"
    fields = [
        "modality", "sample_id", "rate_hz", "old_rows", "old_duration_sec",
        "new_rows", "new_duration_sec", "trimmed_rows", "trimmed_sec",
        "reason", "action_count", "end_time", "shapes",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)

    report_path = BASE / "video_aligned_modality_trim_report.md"
    lines = [
        "# Video-Aligned Modality Trim Report",
        "",
        "Input H5 files:",
        *["- " + str(job["input"].name) for job in selected_jobs],
        "",
        "Output H5 files:",
        *["- " + str(job["output"].name) for job in selected_jobs],
        "",
        "Only the samples listed in `kinect_rgb_feature_monitor/duration_overrides.json` were trimmed.",
        "All other samples were copied unchanged.",
        "",
        "Manifest: `video_aligned_modality_trim_manifest.csv`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("manifest:", manifest_path, flush=True)
    print("report:", report_path, flush=True)


if __name__ == "__main__":
    main()
