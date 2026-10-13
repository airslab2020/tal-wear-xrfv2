from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
STATUS_PATH = REPO / "batch_logs" / "wear_ltca_loso_status.tsv"
FRAMEWORKS = ("actionformer", "temporalmaxer", "tridet")
MODES = ("rgb", "flow")


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_done() -> set[str]:
    if not STATUS_PATH.exists():
        return set()
    with STATUS_PATH.open("r", newline="", encoding="utf-8") as handle:
        return {
            row["run_id"]
            for row in csv.DictReader(handle, delimiter="\t")
            if row.get("status") == "done"
        }


def append_status(row: dict[str, str]) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    exists = STATUS_PATH.exists()
    fields = [
        "time", "dataset", "framework", "video_feature", "config", "run_id",
        "status", "returncode", "gpu",
    ]
    with STATUS_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", required=True, choices=FRAMEWORKS)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--video-feature", action="append", choices=MODES)
    parser.add_argument("--skip-xrfv2", action="store_true")
    parser.add_argument("--seed", default=1, type=int)
    args = parser.parse_args()

    modes = args.video_feature or list(MODES)
    done = read_done()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.chdir(REPO)

    jobs = []
    for mode in modes:
        jobs.append({
            "dataset": "wear",
            "video_feature": mode,
            "config": (
                f"configs/wear_{mode}/{args.framework}_"
                "tinyhar_ltca_gate_concat_loso.yaml"
            ),
            "run_id": (
                f"wear_{mode}_{args.framework}_tinyhar_ltca_gate_concat_"
                f"loso_20ep_seed{args.seed}"
            ),
        })
    if not args.skip_xrfv2:
        jobs.append({
            "dataset": "xrfv2_plus",
            "video_feature": "swin3d",
            "config": (
                f"configs/xrfv2_plus/{args.framework}_"
                "tinyhar_ltca_gate_concat_loso.yaml"
            ),
            "run_id": (
                f"xrfv2_{args.framework}_tinyhar_ltca_gate_concat_"
                f"loso_20ep_seed{args.seed}"
            ),
        })

    for job in jobs:
        config = job["config"]
        run_id = job["run_id"]
        base = {
            "dataset": job["dataset"],
            "framework": args.framework,
            "video_feature": job["video_feature"],
            "config": config,
            "run_id": run_id,
            "gpu": str(args.gpu),
        }
        if run_id in done:
            append_status({
                **base, "time": now(), "status": "skip_done", "returncode": "0"
            })
            continue

        append_status({
            **base, "time": now(), "status": "start", "returncode": ""
        })
        command = [
            str(PYTHON), "main.py", "--config", config,
            "--run_id", run_id, "--seed", str(args.seed),
        ]
        print(
            f"{now()} START dataset={job['dataset']} framework={args.framework} "
            f"video_feature={job['video_feature']} "
            f"gpu={args.gpu} run_id={run_id}",
            flush=True,
        )
        process = subprocess.run(command, env=env)
        status = "done" if process.returncode == 0 else "failed"
        append_status({
            **base,
            "time": now(),
            "status": status,
            "returncode": str(process.returncode),
        })
        print(
            f"{now()} END dataset={job['dataset']} framework={args.framework} "
            f"returncode={process.returncode}",
            flush=True,
        )
        if process.returncode != 0:
            return process.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
