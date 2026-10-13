#!/usr/bin/env python3
"""Run a bounded experiment matrix, one sequential queue per backbone/GPU."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys

from run_loso import CONTROLS, DATASETS, FUSIONS, METHODS, MODELS, REPO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--methods", nargs="+", default=["all"])
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--wear-root")
    parser.add_argument("--xrfv2-root")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    groups = {"all": METHODS, "fusion": FUSIONS, "controls": CONTROLS}
    methods = []
    for value in args.methods:
        for method in groups.get(value, (value,)):
            if method not in METHODS:
                parser.error(f"Unknown method {method}; choose all, fusion, controls, or {METHODS}")
            if method not in methods:
                methods.append(method)
    if len(args.gpus) != len(args.models) or len(set(args.gpus)) != len(args.gpus):
        parser.error("Provide one distinct GPU per model, in the same order as --models")
    if len(set(args.models)) != len(args.models):
        parser.error("--models must be distinct")

    def queue(model, gpu):
        for dataset in args.datasets:
            root = args.xrfv2_root if dataset == "xrfv2" else args.wear_root
            for method in methods:
                for seed in args.seeds:
                    command = [
                        sys.executable, str(REPO / "scripts/run_loso.py"),
                        "--dataset", dataset, "--model", model, "--method", method,
                        "--gpu", gpu, "--seed", str(seed),
                    ]
                    if root:
                        command += ["--data-root", root]
                    if args.dry_run:
                        print(" ".join(command), flush=True)
                        continue
                    path = REPO / "batch_logs" / f"{dataset}_{model}_{method}_seed{seed}.out"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    # Exclusive creation prevents silently replacing a previous run.
                    with path.open("x", encoding="utf-8") as handle:
                        process = subprocess.run(command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT)
                    if process.returncode:
                        raise RuntimeError(f"Run failed ({process.returncode}); inspect {path}")

    if args.dry_run:
        for model, gpu in zip(args.models, args.gpus):
            queue(model, gpu)
    else:
        with ThreadPoolExecutor(max_workers=len(args.models)) as pool:
            futures = [pool.submit(queue, model, gpu) for model, gpu in zip(args.models, args.gpus)]
            for future in futures:
                future.result()
    return 0


if __name__ == "__main__":
    sys.exit(main())
