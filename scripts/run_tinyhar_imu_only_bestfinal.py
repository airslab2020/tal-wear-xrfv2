from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
RUN_ID = "xrfv2_plus_tinyhar_imu_only_loso_20ep_seed1_bestfinal"
CONFIG = "configs/xrfv2_plus/actionformer_tinyhar_imu_only_loso.yaml"
OUT_TSV = REPO / "batch_logs" / "xrfv2_tinyhar_imu_only_bestfinal.tsv"
STATUS = REPO / "batch_logs" / "xrfv2_actionformer_controls_status.tsv"

THS_030_090 = [round(x / 100, 2) for x in range(30, 91, 5)]


def append_status(status: str, returncode: str = "") -> None:
    if not STATUS.exists():
        STATUS.write_text(
            "time\texp_id\tmethod\tconfig\trun_id\tstatus\treturncode\n",
            encoding="utf-8",
        )
    stamp = subprocess.check_output(["date", "-Iseconds"], text=True).strip()
    with STATUS.open("a", encoding="utf-8") as f:
        f.write(
            "\t".join(
                [
                    stamp,
                    "D-AF-imu-only-tinyhar-bestfinal",
                    "tinyhar_imu_only_bestfinal",
                    CONFIG,
                    RUN_ID,
                    status,
                    returncode,
                ]
            )
            + "\n"
        )


def latest_log_dir() -> Path:
    matches = sorted(
        (REPO / "logs" / "actionformer").glob(f"*_{RUN_ID}"),
        key=lambda p: p.stat().st_mtime,
    )
    if not matches:
        raise FileNotFoundError(f"No log dir for {RUN_ID}")
    return matches[-1]


def parse_avg(line: str) -> tuple[float | None, dict[float, float]]:
    avg_match = re.search(r"Avg\. mAP ([0-9.]+) \(%\)", line)
    vals = {
        round(float(k), 2): float(v)
        for k, v in re.findall(r"mAP@([0-9.]+) ([0-9.]+) \(%\)", line)
    }
    avg = float(avg_match.group(1)) if avg_match else None
    return avg, vals


def map_030_090(vals: dict[float, float]) -> float:
    return sum(vals[t] for t in THS_030_090) / len(THS_030_090)


def summarize_log(log_path: Path) -> None:
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    epochs_by_subject: dict[int, list[dict[str, float | int]]] = {}
    finals_by_subject: dict[int, dict[str, float | int]] = {}
    current_subject: int | None = None
    current_epoch: int | None = None

    for idx, line in enumerate(lines):
        split_match = re.match(r"Split\s+(\d+)\s*/\s*\d+", line)
        if split_match:
            current_subject = int(split_match.group(1)) - 1
            current_epoch = None
            epochs_by_subject.setdefault(current_subject, [])
            continue

        epoch_match = re.match(r"Epoch:\s+\[(\d+)/(\d+)\]", line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            continue

        if current_subject is not None and current_epoch is not None and "Avg. mAP" in line:
            avg, vals = parse_avg(line)
            if avg is not None and all(t in vals for t in THS_030_090):
                epochs_by_subject.setdefault(current_subject, []).append(
                    {
                        "epoch": current_epoch,
                        "avg_map": avg,
                        "map030_090": map_030_090(vals),
                        "map50": vals.get(0.5, float("nan")),
                        "map75": vals.get(0.75, float("nan")),
                        "map90": vals.get(0.9, float("nan")),
                    }
                )
            current_epoch = None
            continue

        final_match = re.match(r"FINAL RESULTS SUBJECT\s+(\d+)", line)
        if final_match:
            subject = int(final_match.group(1))
            for candidate in lines[idx : idx + 8]:
                if "Avg. mAP" not in candidate:
                    continue
                avg, vals = parse_avg(candidate)
                if avg is None or not all(t in vals for t in THS_030_090):
                    continue
                finals_by_subject[subject] = {
                    "avg_map": avg,
                    "map030_090": map_030_090(vals),
                    "map50": vals.get(0.5, float("nan")),
                    "map75": vals.get(0.75, float("nan")),
                    "map90": vals.get(0.9, float("nan")),
                }
                break

    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TSV.open("w", encoding="utf-8") as f:
        f.write(
            "subject\tfinal_epoch\tfinal_avg_map\tfinal_map030_090\tfinal_map50\t"
            "final_map75\tfinal_map90\tbest_avg_epoch\tbest_avg_map\t"
            "best_avg_map030_090\tbest030_epoch\tbest030_map030_090\t"
            "best030_avg_map\tlog_dir\n"
        )
        for subject in sorted(epochs_by_subject):
            epochs = epochs_by_subject[subject]
            if not epochs:
                continue
            final_epoch = max(int(e["epoch"]) for e in epochs)
            final = finals_by_subject.get(subject)
            if final is None:
                final = max(epochs, key=lambda e: int(e["epoch"]))
            best_avg = max(epochs, key=lambda e: float(e["avg_map"]))
            best030 = max(epochs, key=lambda e: float(e["map030_090"]))
            f.write(
                "\t".join(
                    [
                        str(subject),
                        str(final_epoch),
                        f"{float(final['avg_map']):.4f}",
                        f"{float(final['map030_090']):.4f}",
                        f"{float(final['map50']):.4f}",
                        f"{float(final['map75']):.4f}",
                        f"{float(final['map90']):.4f}",
                        str(int(best_avg["epoch"])),
                        f"{float(best_avg['avg_map']):.4f}",
                        f"{float(best_avg['map030_090']):.4f}",
                        str(int(best030["epoch"])),
                        f"{float(best030['map030_090']):.4f}",
                        f"{float(best030['avg_map']):.4f}",
                        str(log_path.parent),
                    ]
                )
                + "\n"
            )


def main() -> int:
    os.chdir(REPO)
    append_status("start")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "1")
    proc = subprocess.run(
        [
            str(PYTHON),
            "main.py",
            "--config",
            CONFIG,
            "--run_id",
            RUN_ID,
            "--seed",
            "1",
        ],
        env=env,
    )
    status = "done" if proc.returncode == 0 else "failed"
    try:
        summarize_log(latest_log_dir() / "log.txt")
    finally:
        append_status(status, str(proc.returncode))
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
