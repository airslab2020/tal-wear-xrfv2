#!/usr/bin/env python3
"""Export manuscript table cells verbatim and map them to archived run vectors."""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import re

THRESHOLDS = (.3, .4, .5, .6, .7, .8, .9)
COLS = [f"mAP@{t:.1f}" for t in THRESHOLDS]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tex", type=Path, required=True)
    parser.add_argument("--runs", type=Path, default=Path("results/runs.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/paper_tables.csv"))
    args = parser.parse_args()
    source = args.tex.read_text(encoding="utf-8")
    digest = hashlib.sha256(args.tex.read_bytes()).hexdigest()
    rows = []
    for block in re.findall(r"\\begin\{table\*\}(.*?)\\end\{table\*\}", source, flags=re.S):
        label = re.search(r"\\label\{tab:(xrf|wear)-(actionformer|temporalmaxer|tridet)\}", block)
        if label is None:
            continue
        kind, model = label.groups()
        for line in block.splitlines():
            if "&" not in line:
                continue
            match = re.search(r"\\texttt\{([^}]+)\}", line)
            if match is None:
                continue
            method = match.group(1).replace(r"\_", "_")
            cells = line.split("&")
            values = [re.search(r"[0-9]+\.[0-9]+", cell).group() for cell in cells[-8:]]
            dataset = "xrfv2" if kind == "xrf" else "wear_flow" if "Flow" in cells[0] else "wear_rgb"
            rows.append({
                "dataset": dataset, "model": model, "method": method,
                **dict(zip(COLS + ["mAP7"], values)), "manuscript_sha256": digest,
            })
    if len(rows) != 81:
        raise ValueError(f"Expected 81 rows from six full tables, got {len(rows)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "model", "method", *COLS, "mAP7", "manuscript_sha256"]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    if args.runs.exists():
        with args.runs.open(newline="", encoding="utf-8") as handle:
            runs = list(csv.DictReader(handle))
        mappings = []
        for row in rows:
            dataset, method = row["dataset"], row["method"]
            if method == "flow_only":
                method = "video_only"
            if dataset == "wear_flow" and "imu_only" in method:
                dataset = "wear_rgb"  # Paper repeats the same IMU-only result.
            candidates = [
                r for r in runs if r["status"] == "complete"
                and (r["dataset"], r["model"], r["method"]) == (dataset, row["model"], method)
                and all(abs(float(r[c]) - float(row[c])) < .011 for c in COLS)
            ]
            mappings.append({
                "dataset": row["dataset"], "model": row["model"], "method": row["method"],
                "mAP7": row["mAP7"],
                "match_status": "matched" if candidates else "no_exact_archived_log_match",
                "matching_seeds": " ".join(str(r["seed"]) for r in candidates),
                "matching_runs": " | ".join(r["run"] for r in candidates),
                "log_sha256": " | ".join(r["log_sha256"] for r in candidates),
            })
        with (args.output.parent / "paper_sources.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(mappings[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(mappings)
        print(f"Exported {len(rows)} manuscript rows; {sum(r['match_status'] == 'matched' for r in mappings)} matched to complete archived logs")


if __name__ == "__main__":
    main()
