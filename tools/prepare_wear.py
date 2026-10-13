#!/usr/bin/env python3
"""Assemble WEAR's paper input from 25 Hz IMU and released I3D features.

Copies the official 30 fps LOSO annotations; never regenerates their boundaries.
IMU: 50 samples (2 seconds) per window, stride 25 samples (1 second).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imu-dir", required=True, type=Path, help="25 Hz CSVs: sbj_0.csv .. sbj_17.csv")
    parser.add_argument("--i3d-dir", required=True, type=Path, help="2048-D [RGB; Flow] feature NPYs")
    parser.add_argument("--annotation-dir", required=True, type=Path, help="Official annotations/30fps")
    parser.add_argument("--output-root", type=Path, default=Path("data/wear"))
    args = parser.parse_args()
    root = args.output_root.resolve()
    feat_dir = root / "processed/combined_features/30fps/60_frames_30_stride"
    imu_dir = root / "raw/inertial/25hz"
    annotation_dir = root / "annotations/30fps"
    for directory in (feat_dir, imu_dir, annotation_dir):
        directory.mkdir(parents=True, exist_ok=True)
    manifest = []
    for subject in range(18):
        name = f"sbj_{subject}"
        source = args.imu_dir / f"{name}.csv"
        frame = pd.read_csv(source)
        if frame.shape[1] != 14 or "label" not in frame or frame.columns[-1] != "label":
            raise ValueError(f"{source}: expected [subject_id; 12 acceleration channels; label]")
        raw = frame.iloc[:, 1:13].to_numpy(dtype=np.float32)
        if not np.isfinite(raw).all():
            raise ValueError(f"{source}: IMU contains NaN/Inf; clean it before preprocessing")
        # Match the released sliding-window routine's strict end condition.
        starts = list(range(0, len(raw) - 50, 25))
        if not starts:
            raise ValueError(f"{source}: recording must contain more than 50 samples")
        flat = np.stack([raw[i:i + 50].T.reshape(-1) for i in starts])
        video = np.load(args.i3d_dir / f"{name}.npy", mmap_mode="r")
        if video.ndim != 2 or video.shape[1] != 2048:
            raise ValueError(f"{name}: expected I3D [T,2048], got {video.shape}")
        if len(flat) < len(video):
            raise ValueError(f"{name}: fewer IMU windows ({len(flat)}) than I3D rows ({len(video)})")
        combined = np.concatenate([flat[:len(video)], video], axis=1).astype(np.float32)
        np.save(feat_dir / f"{name}.npy", combined)
        destination = imu_dir / source.name
        if source.resolve() != destination:
            shutil.copy2(source, destination)
        manifest.append({"subject": name, "shape": list(combined.shape), "imu_windows": len(flat)})
    for fold in range(1, 19):
        source = args.annotation_dir / f"wear_split_{fold}.json"
        data = json.loads(source.read_text(encoding="utf-8"))
        heldout = [k for k, v in data["database"].items() if v["subset"].lower() == "validation"]
        if heldout != [f"sbj_{fold - 1}"]:
            raise ValueError(f"{source}: unexpected held-out subject {heldout}")
        destination = annotation_dir / source.name
        if source.resolve() != destination:
            shutil.copy2(source, destination)
    (root / "preprocessing_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared 18 subjects and 18 official LOSO folds under {root}")


if __name__ == "__main__":
    main()
