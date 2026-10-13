#!/usr/bin/env python3
"""Export complete and partial LOSO runs without selecting the best seed."""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import re
from statistics import mean, pstdev

THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
METHODS = (
    "tinyhar_sensor_aware_gate_concat", "tinyhar_ltca_gate_concat",
    "tinyhar_gate_concat", "tinyhar_gate_sum", "tinyhar_imu_only",
    "raw1500_imu_only", "raw600_imu_only", "tinyhar_cat", "direct_cat", "video_only",
)
METRIC_RE = re.compile(r"mAP@([0-9.]+)\s+([0-9.]+)\s+\(%\)")


def metrics(line):
    values = {round(float(t), 2): float(v) for t, v in METRIC_RE.findall(line)}
    return values if all(t in values for t in THRESHOLDS) else None


def parse_log(path):
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    folds, final, duplicates = {}, None, False
    for i, line in enumerate(lines):
        subject = re.match(r"FINAL RESULTS SUBJECT\s+(\d+)", line.strip())
        averaged = "FINAL AVERAGED RESULTS" in line
        if not subject and not averaged:
            continue
        values = next((m for candidate in lines[i + 1:i + 9] if (m := metrics(candidate))), None)
        if values is None:
            continue
        if averaged:
            final = values
        else:
            index = int(subject.group(1))
            duplicates |= index in folds
            folds[index] = values
    return folds, final, duplicates


def run_info(path):
    name = path.parent.name
    dataset = "xrfv2" if "xrfv2" in name else "wear_flow" if "wear_flow" in name else "wear_rgb" if "wear_rgb" in name else "unknown"
    model = path.parent.parent.name
    method = next((m for m in METHODS if m in name), "other")
    if method == "other" and "imu_flatten_video" in name:
        method = "direct_cat"
    seed = re.search(r"seed(\d+)", name)
    return dataset, model, method, int(seed.group(1)) if seed else ""


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, default=Path("logs"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    runs, fold_rows = [], []
    cols = [f"mAP@{t:.1f}" for t in THRESHOLDS]
    for path in sorted(args.logs.glob("*/*/log.txt")):
        dataset, model, method, seed = run_info(path)
        if dataset == "unknown" or method == "other":
            continue
        folds, final, duplicates = parse_log(path)
        expected = 16 if dataset == "xrfv2" else 18
        complete = (
            final is not None and len(folds) == expected
            and set(folds) == set(range(expected)) and not duplicates
            and "_subset" not in path.parent.name
        )
        values = final
        if values is None and folds:
            values = {t: mean([v[t] for v in folds.values()]) for t in THRESHOLDS}
        common = {
            "dataset": dataset, "model": model, "method": method, "seed": seed,
            "run": path.parent.name, "source_log": str(path.relative_to(args.logs)),
        }
        row = {
            **common, "status": "complete" if complete else "partial",
            "folds": len(folds), "expected_folds": expected,
            "log_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            **{col: f"{values[t]:.4f}" if values else "" for col, t in zip(cols, THRESHOLDS)},
            "mAP7": f"{mean(values[t] for t in THRESHOLDS):.4f}" if values else "",
        }
        runs.append(row)
        for index, fold in sorted(folds.items()):
            fold_rows.append({
                **common, "fold": index,
                **{col: f"{fold[t]:.2f}" for col, t in zip(cols, THRESHOLDS)},
                "mAP7": f"{mean(fold[t] for t in THRESHOLDS):.4f}",
            })
    write_csv(args.output / "runs.csv", runs, [
        "dataset", "model", "method", "seed", "run", "source_log", "status",
        "folds", "expected_folds", *cols, "mAP7", "log_sha256",
    ])
    write_csv(args.output / "folds.csv", fold_rows, [
        "dataset", "model", "method", "seed", "run", "source_log", "fold", *cols, "mAP7",
    ])
    groups = {}
    for row in runs:
        if row["status"] == "complete" and row["seed"] != "":
            groups.setdefault((row["dataset"], row["model"], row["method"]), []).append(row)
    seed_rows = []
    for key, group in sorted(groups.items()):
        counts = {}
        for row in group:
            counts[row["seed"]] = counts.get(row["seed"], 0) + 1
        # Do not double count reruns of one seed or silently choose the best rerun.
        unique = [row for row in group if counts[row["seed"]] == 1]
        vals = [float(row["mAP7"]) for row in unique]
        seed_rows.append({
            "dataset": key[0], "model": key[1], "method": key[2],
            "seeds": " ".join(str(row["seed"]) for row in unique),
            "n": len(vals), "mean_mAP7": f"{mean(vals):.4f}" if vals else "",
            "population_std_mAP7": f"{pstdev(vals):.4f}" if vals else "",
            "duplicate_seed_runs_excluded": len(group) - len(unique),
        })
    write_csv(args.output / "seed_statistics.csv", seed_rows, [
        "dataset", "model", "method", "seeds", "n", "mean_mAP7",
        "population_std_mAP7", "duplicate_seed_runs_excluded",
    ])
    print(f"Exported {len(runs)} runs ({sum(r['status'] == 'complete' for r in runs)} complete), {len(fold_rows)} folds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
