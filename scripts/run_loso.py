#!/usr/bin/env python3
"""Portable launcher for one paper configuration; does not change model code."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

import yaml

REPO = Path(__file__).resolve().parents[1]
DATASETS = ("wear_rgb", "wear_flow", "xrfv2")
MODELS = ("actionformer", "temporalmaxer", "tridet")
FUSIONS = (
    "tinyhar_cat", "tinyhar_gate_sum", "tinyhar_gate_concat",
    "tinyhar_sensor_aware_gate_concat", "tinyhar_ltca_gate_concat",
)
CONTROLS = ("video_only", "raw_imu_only", "tinyhar_imu_only", "direct_cat")
METHODS = FUSIONS + CONTROLS


def config_path(dataset, model, method):
    if method == "raw_imu_only":
        method = "raw1500_imu_only" if dataset == "xrfv2" else "raw600_imu_only"
    folder = "xrfv2_plus" if dataset == "xrfv2" else dataset
    return REPO / "configs" / folder / f"{model}_{method}_loso.yaml"


def resolved_config(dataset, model, method, data_root=None, folds=None):
    source = config_path(dataset, model, method)
    cfg = yaml.safe_load(source.read_text(encoding="utf-8"))
    folder = "xrfv2_plus" if dataset == "xrfv2" else "wear"
    root = Path(data_root).expanduser().resolve() if data_root else REPO / "data" / folder
    cfg["anno_json"] = [
        str(root / "annotations" / ("loso" if dataset == "xrfv2" else "30fps") / Path(p).name)
        for p in cfg["anno_json"]
    ]
    if dataset == "xrfv2":
        cfg["dataset"]["feat_folder"] = str(root / "processed/combined_features/imu_flatten_video_1hz")
        cfg["dataset"]["sens_folder"] = str(root / "raw/labels_1hz")
    else:
        cfg["dataset"]["feat_folder"] = str(root / "processed/combined_features/30fps/60_frames_30_stride")
        cfg["dataset"]["sens_folder"] = str(root / "raw/inertial/25hz")
    full_folds = len(cfg["anno_json"])
    chosen = list(range(full_folds)) if folds is None else folds
    if not chosen or len(set(chosen)) != len(chosen) or any(i < 0 or i >= full_folds for i in chosen):
        raise ValueError(f"Fold indices must be distinct values in 0..{full_folds - 1}")
    cfg["anno_json"] = [cfg["anno_json"][i] for i in chosen]
    cfg["devices"] = [0]  # Physical GPU is selected by CUDA_VISIBLE_DEVICES.
    cfg["release_metadata"] = {
        "dataset": dataset, "model": model, "method": method,
        "source_config": str(source.relative_to(REPO)), "fold_indices": chosen,
        "full_loso_folds": full_folds,
    }
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--data-root", help="Prepared WEAR or XRF V2 dataset root")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--folds", help="Optional zero-based indices, e.g. 0,1; default: complete LOSO")
    parser.add_argument("--ckpt-freq", type=int, default=-1)
    parser.add_argument("--dry-run", action="store_true", help="Show resolved config and command; start no training")
    args = parser.parse_args()
    folds = [int(i) for i in args.folds.split(",")] if args.folds else None
    cfg = resolved_config(args.dataset, args.model, args.method, args.data_root, folds)
    suffix = "_subset" + "-".join(map(str, folds)) if folds is not None else ""
    run_id = f"{args.dataset}_{args.model}_{args.method}_loso_20ep_seed{args.seed}{suffix}"
    output = REPO / ".runs/configs" / f"{run_id}.yaml"
    command = [
        sys.executable, str(REPO / "main.py"), "--config", str(output),
        "--run_id", run_id, "--seed", str(args.seed), "--ckpt-freq", str(args.ckpt_freq),
    ]
    print(f"CUDA_VISIBLE_DEVICES={args.gpu} {shlex.join(command)}", flush=True)
    if args.dry_run:
        print(yaml.safe_dump(cfg, sort_keys=False))
        return 0
    for name in ("feat_folder", "sens_folder"):
        if not Path(cfg["dataset"][name]).is_dir():
            parser.error(f"Missing {name}: {cfg['dataset'][name]}; prepare the dataset or set --data-root")
    missing = [p for p in cfg["anno_json"] if not Path(p).is_file()]
    if missing:
        parser.error(f"Missing annotation: {missing[0]}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    return subprocess.run(command, cwd=REPO, env=env).returncode


if __name__ == "__main__":
    sys.exit(main())
