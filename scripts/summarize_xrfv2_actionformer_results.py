from __future__ import annotations

import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
LOG_ROOT = REPO / "logs" / "actionformer"
RUN_IDS = [
    "xrfv2_plus_tinyhar_mravff_loso_20ep_seed1",
    "xrfv2_plus_tinyhar_residual_mravff_3fold_5ep_seed1",
    "xrfv2_plus_tinyhar_residual_mravff_3fold_20ep_seed1",
    "xrfv2_plus_tinyhar_cat_loso_20ep_seed1",
    "xrfv2_plus_video_only_loso_20ep_seed1",
    "xrfv2_plus_raw1500_imu_only_loso_20ep_seed1",
    "xrfv2_plus_tinyhar_imu_only_loso_20ep_seed1",
    "xrfv2_plus_tinyhar_imu_only_loso_20ep_seed1_bestfinal",
    "xrfv2_plus_direct_cat_loso_20ep_seed1",
    "xrfv2_plus_raw1024_mravff_loso_20ep_seed1",
    "xrfv2_plus_tinyhar_gate_sum_loso_20ep_seed1",
    "xrfv2_plus_tinyhar_gate_concat_loso_20ep_seed1",
    "xrfv2_plus_tinyhar_sensor_aware_gate_concat_loso_20ep_seed1",
]

MAP_RE = re.compile(r"Avg\. mAP ([0-9.]+) \(%\)(.*)")


def latest_run_dir(run_id: str) -> Path | None:
    matches = sorted(LOG_ROOT.glob(f"*_{run_id}"), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def parse_final_average(log_path: Path) -> dict[str, str] | None:
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    for idx, line in enumerate(lines):
        if "FINAL AVERAGED RESULTS" not in line:
            continue
        for candidate in lines[idx + 1 : idx + 8]:
            m = MAP_RE.search(candidate)
            if not m:
                continue
            out = {"Avg. mAP": m.group(1)}
            rest = m.group(2)
            for key, value in re.findall(r"mAP@([0-9.]+) ([0-9.]+) \(%\)", rest):
                out[f"mAP@{key}"] = value
            return out
    return None


def main() -> None:
    print("run_id\tstatus\tAvg.mAP\tmAP@0.5\tmAP@0.75\tmAP@0.9\tlog_dir")
    for run_id in RUN_IDS:
        run_dir = latest_run_dir(run_id)
        if run_dir is None:
            print(f"{run_id}\tnot_started\t\t\t\t\t")
            continue
        metrics = parse_final_average(run_dir / "log.txt")
        if metrics is None:
            print(f"{run_id}\trunning_or_incomplete\t\t\t\t\t{run_dir}")
            continue
        print(
            "\t".join(
                [
                    run_id,
                    "done",
                    metrics.get("Avg. mAP", ""),
                    metrics.get("mAP@0.5", ""),
                    metrics.get("mAP@0.75", ""),
                    metrics.get("mAP@0.9", ""),
                    str(run_dir),
                ]
            )
        )


if __name__ == "__main__":
    main()
