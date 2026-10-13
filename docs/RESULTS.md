# Result records and reporting convention

The code release contains recovered experiment records, not a new accuracy run.

## Metric

For each LOSO run, compute per-threshold mAP across the held-out folds, then average the seven thresholds:

```text
mAP7 = mean(mAP@0.3, mAP@0.4, mAP@0.5, mAP@0.6, mAP@0.7, mAP@0.8, mAP@0.9)
```

Values are percentages. The original console output also reports mAP averaged over 17 thresholds from 0.10 to 0.90 in increments of 0.05. That value is retained in historical logs but is not the manuscript metric.

The source logs print threshold results to two decimals. Averaging those printed values can differ slightly from an average computed from unrounded in-memory arrays. CSV summaries keep four decimals to show the calculation, not to claim additional measurement precision.

## Files

| File | What it represents |
| --- | --- |
| paper_tables.csv | Exact displayed cells from all six manuscript tables, 81 rows |
| paper_sources.csv | Matching completed source runs, seeds and log hashes for each table row |
| runs.csv | 125 recognized main-method run records, including 119 complete runs and 6 partial records |
| folds.csv | 2032 recovered final-subject records for those runs |
| seed_statistics.csv | Mean and population standard deviation across available unique complete seeds |

All 81 manuscript rows have a matching complete archived threshold vector. Matching uses a tolerance of 0.011 percentage points on each displayed threshold. WEAR Flow's IMU-only rows refer to the same IMU-only runs as WEAR RGB, since no visual stream is used.

The source text logs are not committed. `log_sha256` fields permit verification against the experiment archive. Run names retain the original date/run ID so that a metric is traceable. These are the recoverable records available when the release was organized; they are not an assertion that every possible seed or extra exploratory configuration was run.

## LTCA manuscript values

| Dataset / visual feature | ActionFormer | TemporalMaxer | TriDet |
| --- | ---: | ---: | ---: |
| WEAR RGB | 66.92 | 63.16 | 64.68 |
| WEAR Flow | 73.49 | 73.15 | 71.17 |
| XRF V2 | 89.45 | 88.76 | 89.65 |

These are manuscript snapshot values. Refer to `paper_sources.csv` for their actual seeds and complete-run sources, rather than assuming every row is seed 1.

## Seed and epoch selection

WEAR Flow + ActionFormer LTCA was rerun with multiple seeds and the displayed **73.49** comes from **seed 13**. The archive contains seeds 1–15 for this setting, including additional in-flight runs that finished after the historical threshold criterion was reached.

XRF V2 + ActionFormer LTCA was also searched across multiple seeds; the table vector matches **seed 14**, with **89.45** at the displayed precision. It does not establish a strict improvement over the 89.45 comparator.

The original repository also contains a `run_tinyhar_imu_only_bestfinal.py` orchestration script that selects a best-epoch control result. The manuscript-source mappings preserve these run names. A best-epoch control and the final-epoch baseline are not interchangeable.

Available-seed statistics are descriptive statistics of the archived trials. The historical stopping/selection policy and unequal numbers of trials across methods mean they are not a pre-registered equal-seed comparison. The release therefore keeps the manuscript snapshot, every available recognized run, and seed statistics as separate artifacts.

For a new comparison, choose the same fixed seed list, epoch/reporting policy, and full LOSO folds for each method. The portable suite supports a bounded list with `--seeds 1 2 3`; it does not implement threshold-driven stopping.

## New result export

```bash
python scripts/summarize_results.py --logs logs --output results/new_runs
```

A full run is marked complete only when a final averaged block exists, every expected subject block is present exactly once, and no subset-run marker is present. Other records are labeled partial. The exporter does not silently choose the largest seed or latest rerun.

To refresh the manuscript snapshot from the full LaTeX tables:

```bash
python tools/export_paper_tables.py --tex /path/to/main_ltca.tex \
  --runs results/runs.csv --output results/paper_tables.csv
```
