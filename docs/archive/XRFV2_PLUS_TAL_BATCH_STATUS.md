# XRFv2 Plus TAL Batch Status

Updated: 2026-06-16 04:21 Asia/Shanghai

## Completed

| Run ID | Head | Method | Epochs | LOSO folds | Avg. mAP |
|---|---|---|---:|---:|---:|
| `xrfv2_plus_tinyhar_mravff_loso_20ep_seed1` | ActionFormer | TinyHAR + MRAVFF | 20 | 16 | 80.02 |
| `xrfv2_plus_tinyhar_residual_mravff_3fold_5ep_seed1` | ActionFormer | TinyHAR + video-residual MRAVFF | 5 | s00-s02 | 80.99 |

This full-method ActionFormer run did not beat the earlier 5-epoch direct-cat baseline (`81.60` Avg. mAP), so the next batch starts with the ActionFormer core ablations and fusion alternatives.

## Queued Now

These jobs are launched by:

```bash
CUDA_VISIBLE_DEVICES=0 nohup /home/zhangtingting/miniconda3/envs/wear/bin/python scripts/run_xrfv2_actionformer_batch.py > batch_logs/xrfv2_actionformer_batch_YYYYmmdd_HHMMSS.log 2>&1 &
```

Status file:

`batch_logs/xrfv2_actionformer_batch_status.tsv`

Launch log:

`batch_logs/xrfv2_actionformer_batch_20260616_042113.log`

| Exp ID | Method | Config | Run ID |
|---|---|---|---|
| `A-AF-01` | `tinyhar_cat` | `configs/xrfv2_plus/actionformer_tinyhar_cat_loso.yaml` | `xrfv2_plus_tinyhar_cat_loso_20ep_seed1` |
| `A-AF-00` | `direct_cat` | `configs/xrfv2_plus/actionformer_direct_cat_loso.yaml` | `xrfv2_plus_direct_cat_loso_20ep_seed1` |
| `A-AF-02` | `raw1024_mravff` | `configs/xrfv2_plus/actionformer_raw1024_mravff_loso.yaml` | `xrfv2_plus_raw1024_mravff_loso_20ep_seed1` |
| `F-AF-01` | `tinyhar_gate_sum` | `configs/xrfv2_plus/actionformer_tinyhar_gate_sum_loso.yaml` | `xrfv2_plus_tinyhar_gate_sum_loso_20ep_seed1` |
| `F-AF-02` | `tinyhar_gate_concat` | `configs/xrfv2_plus/actionformer_tinyhar_gate_concat_loso.yaml` | `xrfv2_plus_tinyhar_gate_concat_loso_20ep_seed1` |
| `F-AF-03` | `tinyhar_sensor_aware_gate_concat` | `configs/xrfv2_plus/actionformer_tinyhar_sensor_aware_gate_concat_loso.yaml` | `xrfv2_plus_tinyhar_sensor_aware_gate_concat_loso_20ep_seed1` |

## Notes

- The original direct-cat baseline config is preserved at `configs/xrfv2_plus/actionformer_imu_flatten_video_loso.yaml`.
- `actionformer_direct_cat_loso.yaml` is a new fair 20-epoch direct-cat config.
- `actionformer_tinyhar_cat_loso.yaml` is also the `concat_project` fusion comparison row.
- `tinyhar_mravff` is already completed by the full-method run above.
- `actionformer_tinyhar_residual_mravff_loso.yaml` is a candidate replacement for plain MRAVFF. Its 3-fold/5-epoch screen reached `80.99` Avg. mAP and about `77.12` mAP@0.30:0.90. This improves over plain MRAVFF on the screened folds, but is not strong enough yet to stop the current `tinyhar_cat` / `direct_cat_20ep` batch.
- If video-residual MRAVFF beats the fair direct-cat baseline and the `tinyhar_cat` ablation on a stronger screen or full 16-fold run, stop the old MRAVFF-focused queue and relaunch the ablation matrix with residual MRAVFF as the proposed fusion.
