#!/usr/bin/env python3
import csv
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import h5py


BASE = Path(os.environ.get("XRFV2_SOURCE_ROOT", "./data/xrfv2_source"))
COMMAND_DIR = BASE / "command_time_video_aligned"
VIDEO_H5 = BASE / "video_feature_853_video_aligned.h5"
OUT_DIR = BASE / "tal_annotations_video_aligned"
SCENE_TO_ID = {
    "kitchen": 0,
    "livingroom": 1,
    "office": 2,
}


def decode_name(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def parse_hms(value):
    return datetime.strptime(value.strip(), "%H:%M:%S")


def seconds_between(start, end):
    delta = (end - start).total_seconds()
    if delta < 0:
        delta += 24 * 3600
    return int(delta)


def sample_scene(sample_id):
    return sample_id.split("_")[1]


def parse_sample_id(sample_id):
    parts = sample_id.split("_")
    if len(parts) != 3:
        raise ValueError(f"bad sample_id: {sample_id}")
    volunteer_id = int(parts[0])
    scene_name = parts[1]
    sequence_id = int(parts[2])
    if scene_name not in SCENE_TO_ID:
        raise ValueError(f"unknown scene name in sample_id: {sample_id}")
    return volunteer_id, scene_name, SCENE_TO_ID[scene_name], sequence_id


def load_video_durations():
    durations = {}
    with h5py.File(VIDEO_H5, "r") as handle:
        sample_names = [decode_name(x) for x in handle["sample_names"][()]]
        for sample_id in sample_names:
            durations[sample_id] = int(handle["samples"][sample_id]["video_feature"].shape[0])
    return sample_names, durations


def ensure_clean_dir(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main():
    sample_names, video_durations = load_video_durations()
    ensure_clean_dir(OUT_DIR)
    per_sample_root = OUT_DIR / "per_sample"
    per_sample_root.mkdir(parents=True, exist_ok=True)

    label_to_id = {}
    action_rows = []
    sample_rows = []
    database = {}
    errors = []

    for sample_id in sample_names:
        volunteer_id, scene, scene_id, sequence_id = parse_sample_id(sample_id)
        json_path = COMMAND_DIR / scene / f"{sample_id}.json"
        if not json_path.exists():
            errors.append((sample_id, "missing_command_json"))
            continue

        actions = json.loads(json_path.read_text(encoding="utf-8"))
        if not actions:
            errors.append((sample_id, "empty_command_json"))
            continue

        sequence_start = parse_hms(actions[0]["start_time"])
        video_t = int(video_durations[sample_id])
        annotations = []
        sample_errors = []

        for action_index, item in enumerate(actions, 1):
            label = item["action"]
            if label not in label_to_id:
                label_to_id[label] = len(label_to_id)

            start_sec = seconds_between(sequence_start, parse_hms(item["start_time"]))
            end_sec = seconds_between(sequence_start, parse_hms(item["end_time"]))
            if end_sec <= start_sec:
                sample_errors.append(f"non_positive_segment:{action_index}")
            if start_sec < 0 or end_sec > video_t:
                sample_errors.append(f"segment_outside_duration:{action_index}")

            segment = [start_sec, end_sec]
            annotation = {
                "label": label,
                "label_id": label_to_id[label],
                "segment": segment,
                "segment_norm": [start_sec / video_t, end_sec / video_t],
                "action_index": action_index,
                "original_id": item.get("id", action_index),
            }
            annotations.append(annotation)

            action_rows.append({
                "sample_id": sample_id,
                "volunteer_id": volunteer_id,
                "scene": scene,
                "scene_id": scene_id,
                "sequence_id": sequence_id,
                "duration_sec": video_t,
                "action_index": action_index,
                "original_id": item.get("id", action_index),
                "label": label,
                "label_id": label_to_id[label],
                "start_sec": start_sec,
                "end_sec": end_sec,
                "segment_duration_sec": end_sec - start_sec,
                "start_norm": start_sec / video_t,
                "end_norm": end_sec / video_t,
            })

        last_end = annotations[-1]["segment"][1]
        if last_end != video_t:
            sample_errors.append(f"last_end_{last_end}_neq_video_T_{video_t}")

        sample_doc = {
            "sample_id": sample_id,
            "volunteer_id": volunteer_id,
            "scene": scene,
            "scene_name": scene,
            "scene_id": scene_id,
            "sequence_id": sequence_id,
            "duration": video_t,
            "duration_unit": "seconds",
            "annotation_type": "temporal_action_localization",
            "annotations": annotations,
        }

        out_scene_dir = per_sample_root / scene
        out_scene_dir.mkdir(parents=True, exist_ok=True)
        (out_scene_dir / f"{sample_id}.json").write_text(
            json.dumps(sample_doc, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        database[sample_id] = {
            "duration": video_t,
            "subset": "all",
            "scene": scene,
            "scene_name": scene,
            "scene_id": scene_id,
            "volunteer_id": volunteer_id,
            "sequence_id": sequence_id,
            "annotations": [
                {
                    "segment": item["segment"],
                    "label": item["label"],
                    "label_id": item["label_id"],
                }
                for item in annotations
            ],
        }

        sample_rows.append({
            "sample_id": sample_id,
            "volunteer_id": volunteer_id,
            "scene": scene,
            "scene_id": scene_id,
            "sequence_id": sequence_id,
            "duration_sec": video_t,
            "n_actions": len(actions),
            "last_end_sec": last_end,
            "pass": "1" if not sample_errors else "0",
            "errors": ";".join(sample_errors),
            "source_json": str(json_path),
            "tal_json": str(out_scene_dir / f"{sample_id}.json"),
        })

    label_map = {label: idx for label, idx in label_to_id.items()}
    activitynet = {
        "version": "XRF V2 dataset",
        "description": "XRF V2 dataset, powered by XJTU AIRS Lab, Fei Wang.",
        "powered_by": "XJTU AIRS Lab",
        "author": "Fei Wang",
        "time_unit": "seconds",
        "database": database,
        "label_map": label_map,
        "scene_map": SCENE_TO_ID,
    }
    (OUT_DIR / "activitynet_tal_853.json").write_text(
        json.dumps(activitynet, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUT_DIR / "label_map.json").write_text(
        json.dumps(label_map, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    sample_manifest = OUT_DIR / "tal_sample_manifest.csv"
    with sample_manifest.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "sample_id", "volunteer_id", "scene", "scene_id", "sequence_id",
            "duration_sec", "n_actions",
            "last_end_sec", "pass", "errors", "source_json", "tal_json",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sample_rows)

    segment_manifest = OUT_DIR / "tal_segments.csv"
    with segment_manifest.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "sample_id", "volunteer_id", "scene", "scene_id", "sequence_id",
            "duration_sec", "action_index", "original_id",
            "label", "label_id", "start_sec", "end_sec", "segment_duration_sec",
            "start_norm", "end_norm",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(action_rows)

    bad_samples = [row for row in sample_rows if row["pass"] != "1"]
    action_counts = [int(row["n_actions"]) for row in sample_rows]
    durations = [int(row["duration_sec"]) for row in sample_rows]
    report_lines = [
        "# TAL Relative-Time Annotation Report",
        "",
        f"Source command directory: `{COMMAND_DIR}`",
        f"Reference video H5: `{VIDEO_H5}`",
        f"Output directory: `{OUT_DIR}`",
        "",
        "## Summary",
        "",
        f"- Samples: `{len(sample_rows)}`",
        f"- Segments/actions: `{len(action_rows)}`",
        f"- Unique labels: `{len(label_map)}`",
        f"- Scene map: `{SCENE_TO_ID}`",
        f"- Passed samples: `{len(sample_rows) - len(bad_samples)}`",
        f"- Failed samples: `{len(bad_samples)}`",
        f"- Duration range: `{min(durations)}`-`{max(durations)}` sec",
        f"- Actions per sample range: `{min(action_counts)}`-`{max(action_counts)}`",
        "",
        "## Outputs",
        "",
        "- `activitynet_tal_853.json`: ActivityNet-style TAL annotation file.",
        "- `per_sample/<scene>/<sample_id>.json`: one relative-time TAL annotation JSON per sample.",
        "- `tal_sample_manifest.csv`: sample-level validation manifest.",
        "- `tal_segments.csv`: segment-level CSV manifest with seconds and normalized boundaries.",
        "- `label_map.json`: label-to-id mapping.",
        "",
        "No absolute clock-time fields are included in the TAL outputs; original clock times remain only in `command_time_video_aligned`.",
    ]
    if bad_samples:
        report_lines.extend(["", "## Failed Samples"])
        for row in bad_samples[:50]:
            report_lines.append(f"- `{row['sample_id']}`: {row['errors']}")
    (OUT_DIR / "tal_annotation_report.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("samples", len(sample_rows))
    print("segments", len(action_rows))
    print("unique_labels", len(label_map))
    print("failed_samples", len(bad_samples))
    print("output_dir", OUT_DIR)


if __name__ == "__main__":
    main()
