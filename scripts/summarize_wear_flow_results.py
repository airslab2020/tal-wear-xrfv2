from __future__ import annotations

import re
from pathlib import Path
from statistics import mean


REPO = Path(__file__).resolve().parents[1]
FRAMEWORKS = ["actionformer", "temporalmaxer", "tridet"]
METHODS = [
    "direct_cat",
    "tinyhar_cat",
    "video_only",
    "tinyhar_gate_sum",
    "tinyhar_gate_concat",
    "tinyhar_sensor_aware_gate_concat",
]
MAP_RE = re.compile(r"Avg\. mAP ([0-9.]+) \(%\)(.*)")
TH_RE = re.compile(r"mAP@([0-9.]+) ([0-9.]+) \(%\)")


def latest_run_dir(framework: str, run_id: str) -> Path | None:
    root = REPO / "logs" / framework
    matches = sorted(root.glob(f"*_{run_id}"), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def parse_map_line(line: str) -> dict[str, float] | None:
    m = MAP_RE.search(line)
    if not m:
        return None
    out = {"Avg.mAP": float(m.group(1))}
    for key, value in TH_RE.findall(m.group(2)):
        out[f"mAP@{key}"] = float(value)
    return out


def parse_log(log_path: Path) -> tuple[str, int, dict[str, float] | None]:
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    final_avg = None
    folds = []
    for idx, line in enumerate(lines):
        if "FINAL AVERAGED RESULTS" in line:
            for candidate in lines[idx + 1 : idx + 8]:
                parsed = parse_map_line(candidate)
                if parsed:
                    final_avg = parsed
                    break
        if line.startswith("FINAL RESULTS SUBJECT"):
            for candidate in lines[idx + 1 : idx + 8]:
                parsed = parse_map_line(candidate)
                if parsed:
                    folds.append(parsed)
                    break
    if final_avg:
        return "done", len(folds), final_avg
    if folds:
        keys = sorted(set().union(*(x.keys() for x in folds)))
        partial = {k: mean([x[k] for x in folds if k in x]) for k in keys}
        return "running", len(folds), partial
    return "started", 0, None


def map_range(metrics: dict[str, float], lo: float, hi: float) -> float | None:
    vals = []
    for i in range(int(round((hi - lo) / 0.05)) + 1):
        value = lo + 0.05 * i
        keys = [f"mAP@{value:.2g}", f"mAP@{value:.1f}", f"mAP@{value:.2f}"]
        for key in keys:
            if key in metrics:
                vals.append(metrics[key])
                break
    return mean(vals) if vals else None


def main() -> None:
    print("framework\tmethod\tstatus\tfolds\tAvg.mAP\tmAP0.3-0.7\tmAP0.3-0.9\tlog_dir")
    for framework in FRAMEWORKS:
        for method in METHODS:
            run_id = f"wear_flow_{framework}_{method}_loso_20ep_seed1"
            run_dir = latest_run_dir(framework, run_id)
            if run_dir is None:
                print(f"{framework}\t{method}\tnot_started\t0/18\t\t\t\t")
                continue
            status, folds, metrics = parse_log(run_dir / "log.txt")
            if metrics is None:
                print(f"{framework}\t{method}\t{status}\t{folds}/18\t\t\t\t{run_dir}")
                continue
            m03_07 = map_range(metrics, 0.3, 0.7)
            m03_09 = map_range(metrics, 0.3, 0.9)
            print("\t".join([
                framework,
                method,
                status,
                f"{folds}/18",
                f"{metrics.get('Avg.mAP', float('nan')):.2f}",
                "" if m03_07 is None else f"{m03_07:.2f}",
                "" if m03_09 is None else f"{m03_09:.2f}",
                str(run_dir),
            ]))


if __name__ == "__main__":
    main()
