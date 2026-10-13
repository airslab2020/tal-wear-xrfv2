# XRFv2 Plus TAL Experiment Plan

Date: 2026-06-15

This plan defines the full 16-fold LOSO experiment matrix for XRFv2 Plus TAL with IMU and video features. The goal is to show that both proposed changes are useful:

1. Replace raw IMU flattening with a learned TinyHAR per-second IMU encoder.
2. Replace plain feature concatenation with MRAVFF multimodal fusion.

All final experiments use 16-fold LOSO, mAP thresholds `0.10:0.05:0.90`, and the same annotation files:

`data/xrfv2_plus/annotations/loso/xrfv2_plus_loso_s00.json` ... `xrfv2_plus_loso_s15.json`

## Current State

Existing direct-cat ActionFormer baseline is preserved:

`configs/xrfv2_plus/actionformer_imu_flatten_video_loso.yaml`

Current full method config:

`configs/xrfv2_plus/actionformer_tinyhar_imu_video_loso.yaml`

Current full method:

`IMU 50 x 30 -> TinyHAR -> Linear projection to 1024 -> MRAVFFFusion with video 1024 -> TAL head`

Current quick result, 5 epochs, 16-fold LOSO:

| Method | Head | Epochs | Avg. mAP |
|---|---:|---:|---:|
| Raw IMU flatten 1500 + video 1024 direct cat | ActionFormer | 5 | 81.60 |
| TinyHAR + MRAVFF | ActionFormer | 5 | 70.24 |

Important finding:

The 5-epoch TinyHAR + MRAVFF result underfits. A 20-epoch single-fold test on LOSO `s02`, the previously bad fold, reached `83.86` Avg. mAP at epoch 20, compared with `38.94` at 5 epochs. Therefore final full-method experiments should use at least 20 epochs.

Currently running:

| Run ID | Purpose |
|---|---|
| `xrfv2_plus_tinyhar_mravff_loso_20ep_seed1` | Full ActionFormer TinyHAR + MRAVFF, 20 epochs, 16-fold LOSO |

## Shared Settings

Use these settings unless an experiment explicitly changes them.

| Setting | Value |
|---|---|
| Dataset root | `/data/XRFv2-tal-caption-qa-imu-wifi-videoFeature-RGB-IR-Depth/` |
| Work repo | `/home/zhangtingting/wear` |
| Feature folder | `data/xrfv2_plus/processed/combined_features/imu_flatten_video_1hz` |
| Label folder | `data/xrfv2_plus/raw/labels_1hz` |
| Video feature | `T x 1024` |
| Raw IMU per second | `50 x 5 x 6 = 50 x 30` |
| Flattened IMU storage | `T x 1500`, channel-major in prepared files, restored to `50 x 30` for TinyHAR |
| mAP thresholds | `[0.10, 0.15, ..., 0.90]` |
| Validation protocol | 16-fold LOSO by volunteer id |
| Seed | `1` first; optional repeat seeds `[1, 2, 3]` after final table is stable |
| Final epochs | `20` minimum for learned fusion methods |
| Quick screen epochs | `5` for smoke tests only |
| Primary metric | Average mAP over `0.10:0.05:0.90` |
| Secondary metrics | mAP@0.50, mAP@0.75, mAP@0.90, Acc, Prec, Rec, F1 |

## Core Method Definitions

These four methods are needed for each TAL head.

| Method ID | Name | IMU branch | Fusion | Purpose |
|---|---|---|---|---|
| `direct_cat` | Raw IMU + video direct cat | Flatten raw IMU `1500` | Concatenate with video, TAL input `2524` | Original baseline |
| `tinyhar_cat` | TinyHAR, no MRAVFF | `50 x 30 -> TinyHAR -> Linear 1024` | Concatenate `[video1024, imu1024]`, then project or feed `2048` to TAL | Tests TinyHAR improvement alone |
| `raw1024_mravff` | No TinyHAR, with MRAVFF | Uniformly sample 1024 dims from raw IMU 1500 | MRAVFF(video1024, rawIMU1024) | Tests MRAVFF improvement without learned IMU encoder |
| `tinyhar_mravff` | Full method | `50 x 30 -> TinyHAR -> Linear 1024` | MRAVFF(video1024, imu1024) | Proposed method |

For `raw1024_mravff`, use a deterministic uniform index map:

```python
indices = np.round(np.linspace(0, 1499, 1024)).astype(np.int64)
imu_1024 = imu_flat_1500[:, indices]
```

Keep this index map identical across ActionFormer, TemporalMaxer, and TriDet.

## TAL Heads

Run the core matrix on three TAL heads:

| Head ID | Source / target implementation |
|---|---|
| `actionformer` | Existing `/home/zhangtingting/wear/camera_baseline/actionformer` plus added multimodal meta-archs |
| `temporalmaxer` | Port/adapt from `/Users/wangfei/Downloads/on_wear/model/with_temporalmaxer` |
| `tridet` | Port/adapt from `/Users/wangfei/Downloads/on_wear/model/with_tridet` or existing `/home/zhangtingting/wear/camera_baseline/tridet` |

## Experiment Matrix A: Main Ablations

These are mandatory. All rows are final 16-fold LOSO.

| Exp ID | Head | Method | Epochs | Expected config |
|---|---|---|---:|---|
| `A-AF-00` | ActionFormer | `direct_cat` | 20 or existing 5 baseline + rerun 20 if needed | `configs/xrfv2_plus/actionformer_direct_cat_loso.yaml` |
| `A-AF-01` | ActionFormer | `tinyhar_cat` | 20 | `configs/xrfv2_plus/actionformer_tinyhar_cat_loso.yaml` |
| `A-AF-02` | ActionFormer | `raw1024_mravff` | 20 | `configs/xrfv2_plus/actionformer_raw1024_mravff_loso.yaml` |
| `A-AF-03` | ActionFormer | `tinyhar_mravff` | 20 | `configs/xrfv2_plus/actionformer_tinyhar_mravff_loso.yaml` |
| `A-TM-00` | TemporalMaxer | `direct_cat` | 20 | `configs/xrfv2_plus/temporalmaxer_direct_cat_loso.yaml` |
| `A-TM-01` | TemporalMaxer | `tinyhar_cat` | 20 | `configs/xrfv2_plus/temporalmaxer_tinyhar_cat_loso.yaml` |
| `A-TM-02` | TemporalMaxer | `raw1024_mravff` | 20 | `configs/xrfv2_plus/temporalmaxer_raw1024_mravff_loso.yaml` |
| `A-TM-03` | TemporalMaxer | `tinyhar_mravff` | 20 | `configs/xrfv2_plus/temporalmaxer_tinyhar_mravff_loso.yaml` |
| `A-TD-00` | TriDet | `direct_cat` | 20 | `configs/xrfv2_plus/tridet_direct_cat_loso.yaml` |
| `A-TD-01` | TriDet | `tinyhar_cat` | 20 | `configs/xrfv2_plus/tridet_tinyhar_cat_loso.yaml` |
| `A-TD-02` | TriDet | `raw1024_mravff` | 20 | `configs/xrfv2_plus/tridet_raw1024_mravff_loso.yaml` |
| `A-TD-03` | TriDet | `tinyhar_mravff` | 20 | `configs/xrfv2_plus/tridet_tinyhar_mravff_loso.yaml` |

Interpretation target:

| Claim | Required comparison |
|---|---|
| TinyHAR helps | `tinyhar_cat > direct_cat` and/or `tinyhar_mravff > raw1024_mravff` |
| MRAVFF helps | `raw1024_mravff > direct_cat` and/or `tinyhar_mravff > tinyhar_cat` |
| Two improvements together are best | `tinyhar_mravff` is best or statistically tied-best across all heads |
| General across TAL frameworks | Same ranking pattern appears for ActionFormer, TemporalMaxer, and TriDet |

## Experiment Matrix B: Fusion Method Comparison

Use TinyHAR encoder for all rows. Compare fusion only.

Fusion methods found in `/Users/wangfei/Downloads/on_wear/model/easy_fusion/fusion.py`:

| Fusion ID | Description |
|---|---|
| `concat_project` | Concatenate video and IMU then project |
| `gate_sum` | Project IMU then gated weighted sum |
| `gate_concat` | Gated IMU plus concat/project |
| `sensor_aware_gate_concat` | Sensor-aware IMU recalibration then gated concat |
| `mravff` | Multi-resolution attention video-feature fusion |

Mandatory fusion comparison rows:

| Exp ID | Head | Fusion | Epochs | Expected config |
|---|---|---|---:|---|
| `F-AF-00` | ActionFormer | `concat_project` | 20 | `configs/xrfv2_plus/actionformer_tinyhar_concat_project_loso.yaml` |
| `F-AF-01` | ActionFormer | `gate_sum` | 20 | `configs/xrfv2_plus/actionformer_tinyhar_gate_sum_loso.yaml` |
| `F-AF-02` | ActionFormer | `gate_concat` | 20 | `configs/xrfv2_plus/actionformer_tinyhar_gate_concat_loso.yaml` |
| `F-AF-03` | ActionFormer | `sensor_aware_gate_concat` | 20 | `configs/xrfv2_plus/actionformer_tinyhar_sensor_aware_gate_concat_loso.yaml` |
| `F-AF-04` | ActionFormer | `mravff` | 20 | Same as `A-AF-03` |
| `F-TM-00` | TemporalMaxer | `concat_project` | 20 | `configs/xrfv2_plus/temporalmaxer_tinyhar_concat_project_loso.yaml` |
| `F-TM-01` | TemporalMaxer | `gate_sum` | 20 | `configs/xrfv2_plus/temporalmaxer_tinyhar_gate_sum_loso.yaml` |
| `F-TM-02` | TemporalMaxer | `gate_concat` | 20 | `configs/xrfv2_plus/temporalmaxer_tinyhar_gate_concat_loso.yaml` |
| `F-TM-03` | TemporalMaxer | `sensor_aware_gate_concat` | 20 | `configs/xrfv2_plus/temporalmaxer_tinyhar_sensor_aware_gate_concat_loso.yaml` |
| `F-TM-04` | TemporalMaxer | `mravff` | 20 | Same as `A-TM-03` |
| `F-TD-00` | TriDet | `concat_project` | 20 | `configs/xrfv2_plus/tridet_tinyhar_concat_project_loso.yaml` |
| `F-TD-01` | TriDet | `gate_sum` | 20 | `configs/xrfv2_plus/tridet_tinyhar_gate_sum_loso.yaml` |
| `F-TD-02` | TriDet | `gate_concat` | 20 | `configs/xrfv2_plus/tridet_tinyhar_gate_concat_loso.yaml` |
| `F-TD-03` | TriDet | `sensor_aware_gate_concat` | 20 | `configs/xrfv2_plus/tridet_tinyhar_sensor_aware_gate_concat_loso.yaml` |
| `F-TD-04` | TriDet | `mravff` | 20 | Same as `A-TD-03` |

Interpretation target:

MRAVFF should outperform or be consistently tied with the simpler fusion methods across all three TAL heads.

## Optional Diagnostic Controls

Run these only if reviewers need extra evidence or if the main matrix has an unexpected failure.

| Exp ID | Head | Method | Purpose |
|---|---|---|---|
| `D-*-video_only` | All heads | Video only `1024` | Shows video baseline |
| `D-*-imu_only_tinyhar` | All heads | TinyHAR IMU only | Shows IMU-only contribution |
| `D-*-imu_norm_sequence` | All heads | TinyHAR + MRAVFF with sequence normalization | Already tested on first 3 ActionFormer folds; did not solve low-fold issue |
| `D-*-epochs_5` | All heads | Full method 5 epochs | Shows underfitting / convergence effect |
| `D-*-epochs_50` | All heads | Full method 50 epochs | Only if 20 epochs has not saturated |

## Implementation Tasks

### Task 1: Common XRFv2 Plus Multimodal Dataset

Extend/keep `xrfv2_plus_multimodal` so it supports:

| Dataset output | Shape | Notes |
|---|---|---|
| `feats` | `1024 x T` | Video feature |
| `additional_feats` | `1500 x T` | Raw IMU window flattened after restoring to time-major `50 x 30` |
| `additional_feats_1024` or model-side sampled IMU | `1024 x T` | For `raw1024_mravff` |
| `segments`, `labels` | ActionFormer/TriDet/TemporalMaxer format | Same as current TAL dataset |

Recommended: do raw-IMU `1500 -> 1024` sampling inside the model or a small transform, not by overwriting saved features.

### Task 2: Model/Fusion Registry

Create a small multimodal registry with these choices:

| Registry key | Options |
|---|---|
| `tal_head` | `actionformer`, `temporalmaxer`, `tridet` |
| `imu_encoder` | `none_raw1024`, `tinyhar_project1024` |
| `fusion` | `direct_cat`, `concat_project`, `gate_sum`, `gate_concat`, `sensor_aware_gate_concat`, `mravff` |

This avoids copying 27 nearly identical model files.

### Task 3: Head Wrappers

Each TAL head should accept a fused sequence:

| Head | Input shape after fusion |
|---|---|
| ActionFormer | `B x C x T` |
| TemporalMaxer | `B x C x T` |
| TriDet | `B x C x T` |

The wrapper should transform `video_list` only once:

```text
video_list with feats + additional_feats
  -> IMU encoder or raw1024 sampler
  -> fusion module
  -> replace item["feats"] with fused feature
  -> call TAL detector/head
```

### Task 4: Config Generation

Generate configs programmatically from a template rather than hand-writing all files.

Suggested script:

`tools/generate_xrfv2_plus_tal_experiment_configs.py`

Inputs:

```text
heads = [actionformer, temporalmaxer, tridet]
methods = [direct_cat, tinyhar_cat, raw1024_mravff, tinyhar_mravff]
fusion_methods = [concat_project, gate_sum, gate_concat, sensor_aware_gate_concat, mravff]
epochs = 20
```

Outputs:

`configs/xrfv2_plus/generated/*.yaml`

### Task 5: Batch Runner

Suggested script:

`tools/run_xrfv2_plus_tal_experiments.py`

Required features:

| Feature | Requirement |
|---|---|
| Resume/skip | Skip an experiment if its log contains `ALL FINISHED` |
| Run ID | Include exp id, head, method, epochs, seed |
| GPU selection | `CUDA_VISIBLE_DEVICES` from CLI |
| Log tracking | Save mapping `{exp_id, config, log_dir}` to JSONL |
| Smoke mode | `--folds 1 --epochs 1` |
| Final mode | `--folds 16 --epochs 20` |

### Task 6: Result Parser

Suggested script:

`tools/collect_xrfv2_plus_tal_results.py`

Parse each experiment log and write:

`results/xrfv2_plus_tal_loso_results.csv`

Required columns:

```text
exp_id, head, method, imu_encoder, fusion, epochs, seed,
avg_mAP,
mAP_0.10, mAP_0.15, ..., mAP_0.90,
acc, prec, rec, f1,
log_dir, config_path, status
```

Also write:

`results/xrfv2_plus_tal_loso_per_fold.csv`

Required columns:

```text
exp_id, fold, volunteer_id, avg_mAP,
mAP_0.10, ..., mAP_0.90,
acc, prec, rec, f1
```

## Recommended Run Order

1. Finish current `ActionFormer + TinyHAR + MRAVFF + 20 epochs + 16-fold` run.
2. Run ActionFormer core ablations:
   - `direct_cat`
   - `tinyhar_cat`
   - `raw1024_mravff`
   - `tinyhar_mravff`
3. Port wrappers to TemporalMaxer and TriDet.
4. Run TemporalMaxer core ablations.
5. Run TriDet core ablations.
6. Run fusion comparison matrix for ActionFormer.
7. Run fusion comparison matrix for TemporalMaxer and TriDet.
8. Collect CSV results and generate tables.

## Expected Paper Tables

### Table 1: Main Ablation Across TAL Heads

Rows: methods. Columns: ActionFormer, TemporalMaxer, TriDet.

| Method | ActionFormer Avg. mAP | TemporalMaxer Avg. mAP | TriDet Avg. mAP |
|---|---:|---:|---:|
| Direct raw IMU/video cat | | | |
| TinyHAR + cat | | | |
| Raw1024 + MRAVFF | | | |
| TinyHAR + MRAVFF | | | |

### Table 2: Fusion Comparison

Rows: fusion methods. Columns: ActionFormer, TemporalMaxer, TriDet.

| Fusion | ActionFormer Avg. mAP | TemporalMaxer Avg. mAP | TriDet Avg. mAP |
|---|---:|---:|---:|
| concat_project | | | |
| gate_sum | | | |
| gate_concat | | | |
| sensor_aware_gate_concat | | | |
| MRAVFF | | | |

### Table 3: Detailed Final Method

For final `TinyHAR + MRAVFF`, report all thresholds.

| Head | Avg. mAP | mAP@0.10 | mAP@0.30 | mAP@0.50 | mAP@0.75 | mAP@0.90 |
|---|---:|---:|---:|---:|---:|---:|
| ActionFormer | | | | | | |
| TemporalMaxer | | | | | | |
| TriDet | | | | | | |

## Notes And Risks

1. Do not overwrite the original direct-cat baseline config or results. It is a comparison method.
2. Use 20 epochs for final learned multimodal methods; 5 epochs is undertrained.
3. If full 20-epoch LOSO is too slow, first run `s00`, `s01`, `s02` as a screen, but final reported results must use all 16 folds.
4. Keep the raw IMU `1500 -> 1024` sampling deterministic and identical across all heads.
5. Keep video feature, annotation, NMS, mAP thresholds, and seed fixed across compared methods.
6. Some validation folds lack samples for some classes, causing per-class accuracy NaNs. Use `np.nanmean`, as done in current code; this does not affect TAL mAP.
