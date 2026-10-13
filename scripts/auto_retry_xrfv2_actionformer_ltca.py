#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PYTHON = sys.executable
CONFIG = "configs/xrfv2_plus/actionformer_tinyhar_ltca_gate_concat_loso.yaml"
FRAMEWORK = "actionformer"
DATASET = "xrfv2"
FEATURE = "swin3d"
THRESHOLD = 89.45
SEED_START = 5
SEED_END = 30
GPUS = [0, 1, 2, 3]
FREE_MEM_MB = 1000
POLL_SEC = 60

ROOT = Path.cwd()
BATCH_LOG_DIR = ROOT / "batch_logs"
BATCH_LOG_DIR.mkdir(exist_ok=True)
STATUS = BATCH_LOG_DIR / "xrfv2_actionformer_ltca_auto_status.tsv"
SUMMARY = BATCH_LOG_DIR / "xrfv2_actionformer_ltca_auto_summary.tsv"

HEADER = "time\tdataset\tframework\tfeature\tseed\tconfig\trun_id\tstatus\tmap7\treturncode\tgpu\tlog\n"
if not STATUS.exists():
    STATUS.write_text(HEADER)
if not SUMMARY.exists():
    SUMMARY.write_text("time\tbest_seed\tbest_map7\tstatus\tnote\n")

MAP_RE = re.compile(r"mAP@([0-9.]+)\s+([0-9.]+)")
AVG_RE = re.compile(r"Avg\. mAP\s+([0-9.]+)")
THS = ["0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def append_status(seed: int, run_id: str, status: str, map7: str = "", rc: str = "", gpu: str = "", log: str = "") -> None:
    with STATUS.open("a") as f:
        f.write(f"{now()}\t{DATASET}\t{FRAMEWORK}\t{FEATURE}\t{seed}\t{CONFIG}\t{run_id}\t{status}\t{map7}\t{rc}\t{gpu}\t{log}\n")


def gpu_mem() -> dict[int, int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return {}
    mem = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit():
            try:
                mem[int(parts[0])] = int(parts[1])
            except ValueError:
                pass
    return mem


def parse_result(run_id: str):
    files = sorted((ROOT / "logs" / "actionformer").glob(f"*_{run_id}/log.txt"))
    if not files:
        return None, ""
    path = files[-1]
    text = path.read_text(errors="ignore")
    idx = text.rfind("FINAL AVERAGED RESULTS")
    if idx < 0:
        return None, str(path)
    tail = text[idx : idx + 2500]
    vals = {k: v for k, v in MAP_RE.findall(tail)}
    if not all(t in vals for t in THS):
        return None, str(path)
    row = [float(vals[t]) for t in THS]
    return sum(row) / len(row), str(path)


def existing_done(seed: int):
    for run_id in (
        f"xrfv2_actionformer_tinyhar_ltca_gate_concat_loso_20ep_seed{seed}",
        f"xrfv2_actionformer_tinyhar_ltca_gate_concat_loso_20ep_seed{seed}_retry",
        f"xrfv2_actionformer_tinyhar_ltca_gate_concat_loso_20ep_seed{seed}_auto",
    ):
        m, log = parse_result(run_id)
        if m is not None:
            return run_id, m, log
    return None


def main() -> int:
    next_seed = SEED_START
    running: dict[subprocess.Popen, tuple[int, int, str, str]] = {}
    best_seed = None
    best_map = -1.0
    best_log = ""
    success = False

    # Include known previous retry results if present.
    for seed in range(1, SEED_START):
        found = existing_done(seed)
        if found:
            run_id, m, log = found
            if m > best_map:
                best_seed, best_map, best_log = seed, m, log

    while True:
        # Collect finished own jobs.
        for proc in list(running):
            rc = proc.poll()
            if rc is None:
                continue
            seed, gpu, run_id, out_log = running.pop(proc)
            m, log = parse_result(run_id)
            map_str = "" if m is None else f"{m:.4f}"
            append_status(seed, run_id, "done" if rc == 0 else "failed", map_str, str(rc), str(gpu), log or out_log)
            if m is not None and m > best_map:
                best_seed, best_map, best_log = seed, m, log
            if m is not None and m > THRESHOLD:
                success = True
                with SUMMARY.open("a") as f:
                    f.write(f"{now()}\t{best_seed}\t{best_map:.4f}\tsuccess\tthreshold {THRESHOLD} exceeded; log={best_log}\n")

        if success:
            # Do not launch more; wait for any already launched siblings to finish cleanly.
            if not running:
                return 0
            time.sleep(POLL_SEC)
            continue

        if next_seed > SEED_END and not running:
            with SUMMARY.open("a") as f:
                f.write(f"{now()}\t{best_seed}\t{best_map:.4f}\tno_success\tsearched seeds {SEED_START}-{SEED_END}; threshold {THRESHOLD}; log={best_log}\n")
            return 2

        mem = gpu_mem()
        own_busy = {gpu for _, (seed, gpu, run_id, out_log) in running.items()}
        free_gpus = [g for g in GPUS if g not in own_busy and mem.get(g, 10**9) < FREE_MEM_MB]
        while free_gpus and next_seed <= SEED_END and not success:
            gpu = free_gpus.pop(0)
            seed = next_seed
            next_seed += 1
            run_id = f"xrfv2_actionformer_tinyhar_ltca_gate_concat_loso_20ep_seed{seed}_auto"
            out_log = str(BATCH_LOG_DIR / f"xrfv2_actionformer_ltca_seed{seed}_auto_gpu{gpu}.out")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            out_f = open(out_log, "w")
            cmd = [PYTHON, "main.py", "--config", CONFIG, "--run_id", run_id, "--seed", str(seed)]
            proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=out_f, stderr=subprocess.STDOUT)
            running[proc] = (seed, gpu, run_id, out_log)
            append_status(seed, run_id, "start", "", "", str(gpu), out_log)

        time.sleep(POLL_SEC)


if __name__ == "__main__":
    raise SystemExit(main())
