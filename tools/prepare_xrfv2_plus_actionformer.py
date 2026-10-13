#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import h5py
import numpy as np


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def parse_sample_name(sample_name: str) -> tuple[int, str, int]:
    parts = sample_name.rsplit("_", 2)
    if len(parts) != 3:
        raise ValueError(f"Unexpected sample name: {sample_name}")
    return int(parts[0]), parts[1], int(parts[2])


def sort_key(sample_name: str) -> tuple[int, str, int]:
    volunteer_id, scene, sequence_id = parse_sample_name(sample_name)
    return volunteer_id, scene.lower(), sequence_id


def clip_annotations(annotations: list[dict], seconds: int) -> list[dict]:
    clipped = []
    for ann in annotations:
        start, end = ann["segment"]
        start = max(0.0, float(start))
        end = min(float(seconds), float(end))
        if end <= start:
            continue
        clipped.append(
            {
                "segment": [start, end],
                "label": ann["label"],
                "label_id": int(ann["label_id"]),
            }
        )
    return clipped


def labels_for_seconds(annotations: list[dict], seconds: int) -> list[str]:
    labels = ["null"] * seconds
    for ann in annotations:
        start, end = ann["segment"]
        start_idx = max(0, int(math.floor(float(start))))
        end_idx = min(seconds, int(math.ceil(float(end))))
        for idx in range(start_idx, end_idx):
            labels[idx] = ann["label"]
    return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--release-root",
        default="./data/XRFv2-tal-caption-qa-imu-wifi-videoFeature-RGB-IR-Depth",
    )
    parser.add_argument(
        "--output-root",
        default="./data/xrfv2_plus",
    )
    parser.add_argument("--imu-hz", type=int, default=50)
    args = parser.parse_args()

    release_root = Path(args.release_root)
    output_root = Path(args.output_root)

    tal_path = (
        release_root
        / "annotations"
        / "tal_annotations_video_aligned"
        / "activitynet_tal_853.json"
    )
    label_map_path = (
        release_root
        / "annotations"
        / "tal_annotations_video_aligned"
        / "label_map.json"
    )
    imu_path = release_root / "data" / "imu_50hz_853_video_aligned.h5"
    video_path = release_root / "data" / "video_feature_853_video_aligned.h5"

    tal = read_json(tal_path)
    label_map = read_json(label_map_path)
    db = tal["database"]

    feat_dir = output_root / "processed" / "combined_features" / "imu_flatten_video_1hz"
    label_dir = output_root / "raw" / "labels_1hz"
    anno_dir = output_root / "annotations" / "loso"
    feat_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    anno_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    id_map = {}
    new_db = {}

    with h5py.File(imu_path, "r") as imu_h5, h5py.File(video_path, "r") as video_h5:
        sample_names = sorted(
            set(db)
            & set(imu_h5["samples"].keys())
            & set(video_h5["samples"].keys()),
            key=sort_key,
        )

        for internal_idx, sample_name in enumerate(sample_names):
            internal_id = f"sbj_{internal_idx}"
            sample_db = db[sample_name]
            volunteer_id = int(sample_db["volunteer_id"])
            duration = int(sample_db["duration"])

            video = video_h5["samples"][sample_name]["video_feature"][:].astype(np.float32)
            imu = imu_h5["samples"][sample_name]["imu"][:].astype(np.float32)

            seconds = min(duration, video.shape[0], imu.shape[0] // args.imu_hz)
            if seconds <= 0:
                continue

            video = video[:seconds]
            imu = imu[: seconds * args.imu_hz]
            imu = imu.reshape(seconds, args.imu_hz, *imu.shape[1:])
            # Match WEAR's channel-major inertial flattening: channels/devices before time.
            imu_flat = np.transpose(imu, (0, 2, 3, 1)).reshape(seconds, -1)

            combined = np.concatenate([imu_flat, video], axis=1).astype(np.float32)
            if combined.shape[1] != 2524:
                raise ValueError(
                    f"{sample_name} produced dim {combined.shape[1]}, expected 2524"
                )
            np.save(feat_dir / f"{internal_id}.npy", combined)

            annotations = clip_annotations(sample_db["annotations"], seconds)
            labels = labels_for_seconds(annotations, seconds)
            with open(label_dir / f"{internal_id}.csv", "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["sbj_id", "label"])
                for label in labels:
                    writer.writerow([internal_idx, label])

            id_map[internal_id] = {
                "sample_id": sample_name,
                "volunteer_id": volunteer_id,
                "scene": sample_db.get("scene"),
                "scene_name": sample_db.get("scene_name"),
                "sequence_id": sample_db.get("sequence_id"),
                "seconds": seconds,
            }
            new_db[internal_id] = {
                "subset": "all",
                "duration": seconds,
                "fps": 1.0,
                "original_sample_id": sample_name,
                "volunteer_id": volunteer_id,
                "scene": sample_db.get("scene"),
                "scene_name": sample_db.get("scene_name"),
                "sequence_id": sample_db.get("sequence_id"),
                "annotations": annotations,
            }
            summary_rows.append(
                {
                    "internal_id": internal_id,
                    "sample_id": sample_name,
                    "volunteer_id": volunteer_id,
                    "seconds": seconds,
                    "feature_shape": list(combined.shape),
                    "num_annotations": len(annotations),
                }
            )

    volunteers = sorted({row["volunteer_id"] for row in summary_rows})
    anno_paths = []
    for volunteer_id in volunteers:
        fold_db = {}
        for internal_id, item in new_db.items():
            fold_item = dict(item)
            fold_item["subset"] = (
                "Validation" if int(item["volunteer_id"]) == volunteer_id else "Training"
            )
            fold_db[internal_id] = fold_item

        fold_obj = {
            "version": "XRFv2 Plus ActionFormer LOSO",
            "description": "IMU 50Hz flattened per second and concatenated with 1Hz video features.",
            "time_unit": "seconds",
            "database": fold_db,
            "label_dict": label_map,
        }
        out_path = anno_dir / f"xrfv2_plus_loso_s{volunteer_id:02d}.json"
        write_json(fold_obj, out_path)
        anno_paths.append(str(out_path))

    write_json(id_map, output_root / "id_map.json")
    write_json(
        {
            "num_samples": len(summary_rows),
            "volunteers": volunteers,
            "feature_dir": str(feat_dir),
            "label_dir": str(label_dir),
            "annotation_dir": str(anno_dir),
            "feature_dim": 2524,
            "imu_dim": 1500,
            "video_dim": 1024,
            "anno_json": anno_paths,
            "samples": summary_rows,
        },
        output_root / "prepare_summary.json",
    )

    print(f"Wrote {len(summary_rows)} feature files to {feat_dir}")
    print(f"Wrote {len(volunteers)} LOSO annotation files to {anno_dir}")
    print(f"Feature dim: 2524 = 1500 IMU + 1024 video")


if __name__ == "__main__":
    main()
