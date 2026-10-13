from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
STATUS_PATH = REPO / "batch_logs" / "xrfv2_temporalmaxer_tridet_batch_status.tsv"

METHODS = [
    "direct_cat",
    "tinyhar_cat",
    "video_only",
    "raw1500_imu_only",
    "tinyhar_imu_only",
    "tinyhar_gate_sum",
    "tinyhar_gate_concat",
    "tinyhar_sensor_aware_gate_concat",
]
FRAMEWORKS = ["temporalmaxer", "tridet"]


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def queue(
    framework_filter: set[str] | None = None,
    method_filter: set[str] | None = None,
) -> list[dict[str, str]]:
    items = []
    for framework in FRAMEWORKS:
        if framework_filter and framework not in framework_filter:
            continue
        prefix = "TM" if framework == "temporalmaxer" else "TD"
        for idx, method in enumerate(METHODS):
            if method_filter and method not in method_filter:
                continue
            items.append(
                {
                    "exp_id": f"{prefix}-{idx:02d}",
                    "framework": framework,
                    "method": method,
                    "config": f"configs/xrfv2_plus/{framework}_{method}_loso.yaml",
                    "run_id": f"xrfv2_plus_{framework}_{method}_loso_20ep_seed1",
                }
            )
    return items


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
    fields = [
        "time",
        "exp_id",
        "framework",
        "method",
        "config",
        "run_id",
        "status",
        "returncode",
        "gpu",
    ]
    with STATUS_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def gpu_snapshot(index: int) -> tuple[int, int]:
    cmd = [
        "nvidia-smi",
        f"--id={index}",
        "--query-gpu=utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    util_s, mem_s = [x.strip() for x in out.split(",")]
    return int(util_s), int(mem_s)


def wait_for_gpu(index: int, max_mem: int, max_util: int, poll_seconds: int) -> None:
    while True:
        util, mem = gpu_snapshot(index)
        print(
            f"{now()} GPU{index} util={util}% mem={mem}MiB "
            f"threshold util<={max_util}% mem<={max_mem}MiB",
            flush=True,
        )
        if util <= max_util and mem <= max_mem:
            return
        time.sleep(poll_seconds)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--wait-gpu", action="store_true")
    parser.add_argument("--max-mem", type=int, default=3000)
    parser.add_argument("--max-util", type=int, default=20)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument(
        "--framework",
        action="append",
        choices=FRAMEWORKS,
        help="Limit the queue to one or more frameworks.",
    )
    parser.add_argument(
        "--method",
        action="append",
        choices=METHODS,
        help="Limit the queue to one or more methods.",
    )
    args = parser.parse_args()

    os.chdir(REPO)
    env = os.environ.copy()
    framework_filter = set(args.framework) if args.framework else None
    method_filter = set(args.method) if args.method else None

    for item in queue(framework_filter, method_filter):
        done = completed_run_ids()
        if item["run_id"] in done:
            append_status({**item, "time": now(), "status": "skip_done", "returncode": "0", "gpu": str(args.gpu)})
            continue

        if args.wait_gpu:
            wait_for_gpu(args.gpu, args.max_mem, args.max_util, args.poll_seconds)

        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        append_status({**item, "time": now(), "status": "start", "returncode": "", "gpu": str(args.gpu)})
        cmd = [
            str(PYTHON),
            "main.py",
            "--config",
            item["config"],
            "--run_id",
            item["run_id"],
            "--seed",
            "1",
        ]
        print(f"{now()} START {item['exp_id']} {item['run_id']} gpu={args.gpu}", flush=True)
        proc = subprocess.run(cmd, env=env)
        status = "done" if proc.returncode == 0 else "failed"
        append_status(
            {
                **item,
                "time": now(),
                "status": status,
                "returncode": str(proc.returncode),
                "gpu": str(args.gpu),
            }
        )
        print(f"{now()} END {item['exp_id']} {item['run_id']} rc={proc.returncode}", flush=True)
        if proc.returncode != 0:
            return proc.returncode

    return 0


if __name__ == "__main__":
    sys.exit(main())
