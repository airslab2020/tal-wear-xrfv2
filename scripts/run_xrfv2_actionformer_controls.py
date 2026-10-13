from __future__ import annotations

import csv
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
STATUS_PATH = REPO / "batch_logs" / "xrfv2_actionformer_controls_status.tsv"

QUEUE = [
    {
        "exp_id": "D-AF-video-only",
        "method": "video_only",
        "config": "configs/xrfv2_plus/actionformer_video_only_loso.yaml",
        "run_id": "xrfv2_plus_video_only_loso_20ep_seed1",
    },
    {
        "exp_id": "D-AF-imu-only-raw1500",
        "method": "raw1500_imu_only",
        "config": "configs/xrfv2_plus/actionformer_raw1500_imu_only_loso.yaml",
        "run_id": "xrfv2_plus_raw1500_imu_only_loso_20ep_seed1",
    },
    {
        "exp_id": "D-AF-imu-only-tinyhar",
        "method": "tinyhar_imu_only",
        "config": "configs/xrfv2_plus/actionformer_tinyhar_imu_only_loso.yaml",
        "run_id": "xrfv2_plus_tinyhar_imu_only_loso_20ep_seed1",
    },
]


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def completed_run_ids() -> set[str]:
    if not STATUS_PATH.exists():
        return set()
    done: set[str] = set()
    with STATUS_PATH.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("status") == "done":
                done.add(row["run_id"])
    return done


def append_status(row: dict[str, str]) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    exists = STATUS_PATH.exists()
    fields = ["time", "exp_id", "method", "config", "run_id", "status", "returncode"]
    with STATUS_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    os.chdir(REPO)
    done = completed_run_ids()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "2")

    for item in QUEUE:
        if item["run_id"] in done:
            append_status({**item, "time": now(), "status": "skip_done", "returncode": "0"})
            continue
        append_status({**item, "time": now(), "status": "start", "returncode": ""})
        print("{} START {} {}".format(now(), item["exp_id"], item["run_id"]), flush=True)
        proc = subprocess.run(
            [
                str(PYTHON),
                "main.py",
                "--config",
                item["config"],
                "--run_id",
                item["run_id"],
                "--seed",
                "1",
            ],
            env=env,
        )
        status = "done" if proc.returncode == 0 else "failed"
        append_status(
            {
                **item,
                "time": now(),
                "status": status,
                "returncode": str(proc.returncode),
            }
        )
        print(
            "{} END {} {} rc={}".format(
                now(), item["exp_id"], item["run_id"], proc.returncode
            ),
            flush=True,
        )
        if proc.returncode != 0:
            return proc.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
