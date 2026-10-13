# IMU–Visual Temporal Action Localization on WEAR and XRF V2

[中文说明](README_zh.md)

Research code for TinyHAR-based inertial–visual temporal action localization (TAL), covering **five fusion methods**, **ActionFormer / TemporalMaxer / TriDet**, and the WEAR RGB, WEAR Flow, and XRF V2 protocols. This release organizes the actual experiment source, configurations, preprocessing scripts, and results used in the PCC manuscript and its LTCA extension.

## Included experiments

| Configuration method | Fusion implementation | Purpose |
| --- | --- | --- |
| `tinyhar_cat` | `ConcatProjectFusion` / `concat_project` | TinyHAR encoding, feature concatenation, and output projection |
| `tinyhar_gate_sum` | `GateSumFusion` | Complementary gates followed by weighted sum |
| `tinyhar_gate_concat` | `GateConcatFusion` | Retain the video stream, gate the IMU stream, concatenate and project |
| `tinyhar_sensor_aware_gate_concat` | `SensorAwareGateConcatFusion` | Recalibrate encoded IMU channels before gated concatenation |
| `tinyhar_ltca_gate_concat` | `LocalTemporalCrossAttentionGateConcatFusion` | Bidirectional local temporal cross-attention, residual normalization, independent gates, concatenation |

The sensor-aware implementation recalibrates **TinyHAR feature channels**, using temporal average pooling and a channel gate; it does not construct explicit per-device tokens. The implementation is shared from `camera_baseline/actionformer/libs/modeling/meta_archs.py` by all three backbones.

The controls are `video_only`, `raw_imu_only`, `tinyhar_imu_only`, and `direct_cat`. They test modality contribution, learned versus flattened IMU representation, and encoding before fusion. `raw_imu_only` is a launcher alias for `raw600_imu_only` on WEAR and `raw1500_imu_only` on XRF V2. WEAR Flow's `video_only` configuration corresponds to `flow_only` in the manuscript.

There are **81 primary configurations**: 3 dataset/visual settings × 3 backbones × (5 fusion methods + 4 controls). Additional MRAV-FF, residual-fusion, raw-feature, and IMU-normalization configurations from the earlier experiments are retained in `configs/xrfv2_plus/`. These additional configurations are not new LTCA component-ablation results.

## Repository layout

```text
camera_baseline/        Three TAL backbones, shared fusion classes and dataset adapters
inertial_baseline/      TinyHAR and upstream inertial baselines
configs/
  wear_rgb/            18-fold WEAR RGB configurations
  wear_flow/           18-fold WEAR Flow configurations
  xrfv2_plus/          16-fold XRF V2 configurations and earlier variants
scripts/
  run_loso.py          Portable single-experiment launcher
  run_suite.py         Bounded matrix, separate GPU/queue for each backbone
  summarize_results.py Seven-threshold / per-fold / seed result exporter
tools/                 WEAR assembler, XRF V2 release bridge, paper-table exporter
preprocessing/xrfv2/   Raw IMU processing, video features, H5 packing and alignment
data_creation/         Upstream WEAR synchronization and annotation construction
postprocessing/        Upstream optional prediction postprocessing
results/               Archived run metrics and exact manuscript table snapshots
tests/                 Synthetic fusion/backbone and preprocessing checks
docs/                  Data, method, results and source-provenance documentation
```

Raw data, extracted feature arrays, checkpoints, local credentials, and full training logs are excluded from Git.

## Installation

The experiment server used Linux, Python **3.10.19**, PyTorch **2.0.1+cu117**, NumPy **1.23.5**, h5py **3.7.0**, and PyYAML **6.0.3**. The requirements preserve the upstream WEAR package pins, with the missing TinyHAR/preprocessing dependencies added. Use a fresh environment:

```bash
conda create -n tal-wear-xrfv2 python=3.10 -y
conda activate tal-wear-xrfv2
python -m pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu117
python -m pip install -r requirements.txt
python -m pip install -r requirements-preprocessing.txt  # Only needed for video preprocessing
```

A C++ compiler with OpenMP support is needed for temporal NMS. The three backbones use the same `nms_1d_cpu` extension, so one installation is sufficient:

```bash
cd camera_baseline/actionformer/libs/utils
python setup.py install
cd ../../../..
python -c "import torch; import nms_1d_cpu"
```

The paper training entry point uses CUDA and `DataParallel`. CPU is sufficient for the synthetic checks, but it is not the supported full-training path. Neptune logging is disabled unless explicitly requested; no Neptune credentials are bundled.

## Data preparation

Obtain WEAR from its [official project](https://github.com/mariusbock/wear) and XRF V2 from the [AIRS dataset repository](https://github.com/airslab2020/XRFV2). Dataset access and usage terms belong to their respective releases.

**WEAR:** place the official processed dataset at `data/wear`, or use `--data-root` below. The required parts are:

```text
data/wear/
  annotations/30fps/wear_split_1.json ... wear_split_18.json
  raw/inertial/25hz/sbj_0.csv ... sbj_17.csv
  processed/combined_features/30fps/60_frames_30_stride/sbj_0.npy ... sbj_17.npy
```

Each combined row is **2648-D = 600-D IMU + 1024-D RGB + 1024-D Flow**. RGB uses columns 600–1623; Flow uses 1624–2647. The paper configuration uses **25 Hz IMU × 50 samples = 2 seconds** and **30 fps video × 60 frames = 2 seconds**, with a **1-second stride**. TinyHAR receives each IMU window in time/channel order after the adapter restores it from the stored channel/time flattening.

To rebuild the combined arrays from the released 25 Hz CSVs and precomputed RGB+Flow I3D features:

```bash
python tools/prepare_wear.py \
  --imu-dir /path/to/WEAR/raw/inertial/25hz \
  --i3d-dir /path/to/WEAR/processed/i3d_features/30fps/60_frames_30_stride \
  --annotation-dir /path/to/WEAR/annotations/30fps \
  --output-root data/wear
```

This assembler copies the official LOSO annotations without changing their action boundaries. I3D/optical-flow extraction is supplied through upstream WEAR's precomputed features; a new I3D training or flow-estimation implementation is not included.

**XRF V2:** the bridge expects the synchronized release files listed in [docs/DATA_PREPROCESSING.md](docs/DATA_PREPROCESSING.md):

```bash
python tools/prepare_xrfv2_plus_actionformer.py \
  --release-root /path/to/XRFv2-tal-caption-qa-imu-wifi-videoFeature-RGB-IR-Depth \
  --output-root data/xrfv2_plus
```

The output contains 16 volunteer-held-out annotation files, label CSVs, and **2524-D = 1500-D IMU + 1024-D Swin3D-B video** feature rows. The 1500 IMU values are 50 samples × 5 devices × 6 channels, representing one second. All recordings from the same volunteer are held out together; an internal `sbj_*` file identifier is a recording ID, not a volunteer ID.

See [the preprocessing guide](docs/DATA_PREPROCESSING.md) for raw IMU splitting/resampling, RGB feature extraction, H5 packing, duration alignment, and required input manifests.

## Run one experiment

```bash
python scripts/run_loso.py \
  --dataset wear_flow --model actionformer \
  --method tinyhar_ltca_gate_concat --gpu 0 --seed 1

python scripts/run_loso.py \
  --dataset xrfv2 --model tridet \
  --method tinyhar_sensor_aware_gate_concat --gpu 2 --seed 1 \
  --data-root /path/to/prepared/xrfv2_plus
```

Add `--dry-run` to display the resolved configuration and command without creating a run. Add `--folds 0,1` for a deliberately partial check; indices are zero-based, and partial-run names contain `subset`. The default runs all 18/16 folds. `--ckpt-freq 5` enables periodic checkpoints; the default does not save checkpoints.

The launcher resolves local data paths into an ignored `.runs/configs/` directory. It keeps the stored model, optimizer, epoch and loader settings unchanged. Use `CUDA_VISIBLE_DEVICES` through `--gpu`; the model configuration always addresses the selected physical GPU as logical device 0.

## Run all models on separate GPUs

This command runs the five fusion methods, first on WEAR RGB, then WEAR Flow, then XRF V2. Each backbone has its own sequential queue:

```bash
python scripts/run_suite.py \
  --datasets wear_rgb wear_flow xrfv2 \
  --models actionformer temporalmaxer tridet \
  --methods fusion --gpus 0 1 2 --seeds 1
```

Use `--methods all` to include the four controls, `--methods controls` for controls only, or supply individual method names. `--seeds 1 2 3` schedules a fixed set of seeds. `--wear-root` and `--xrfv2-root` set external data locations. Queue logs use exclusive file creation so an existing run is not silently overwritten.

To run in the background:

```bash
mkdir -p batch_logs
nohup python -u scripts/run_suite.py \
  --datasets wear_rgb wear_flow xrfv2 \
  --models actionformer temporalmaxer tridet \
  --methods all --gpus 0 1 2 --seeds 1 \
  > batch_logs/paper_suite.out 2>&1 &
```

Earlier per-dataset batch scripts and the historical threshold-based LTCA seed-search scripts are also retained in `scripts/`. They reproduce the original orchestration, including its selection policy; the portable suite above uses an explicitly bounded seed list.

## Evaluation and archived results

The manuscript metric is the arithmetic mean of mAP at **tIoU 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9**, in percent. Original training logs also report an average over 17 thresholds; that printed `Avg. mAP` is not the manuscript's seven-threshold score.

```bash
python scripts/summarize_results.py --logs logs --output results/new_runs
```

The exporter produces `runs.csv`, `folds.csv`, and `seed_statistics.csv`; it labels partial runs, records seeds and log hashes, and never silently keeps only the maximum seed. Seed statistics exclude duplicated reruns of a seed rather than double-counting or selecting one.

The committed result snapshot contains:

- [paper_tables.csv](results/paper_tables.csv): all 81 rows copied from the LTCA manuscript tables, retaining the original cells;
- [paper_sources.csv](results/paper_sources.csv): complete-log matches, source runs and seeds, plus explicit unmatched rows;
- [runs.csv](results/runs.csv) and [folds.csv](results/folds.csv): recovered individual experiments and fold metrics;
- [seed_statistics.csv](results/seed_statistics.csv): statistics across the available unique, complete seeds.

The historical WEAR Flow + ActionFormer LTCA value **73.49** is from **seed 13**, selected after multiple trials. XRF V2 + ActionFormer's displayed **89.45** is tied at the displayed precision, not evidence of a strict improvement over 89.45. Manuscript table snapshots, final-epoch run results, historical best-epoch controls, and multi-seed means are different records; consult [docs/RESULTS.md](docs/RESULTS.md) before comparing them.

## Verification

```bash
python -m unittest discover -s tests -p 'test_*.py'
python tests/smoke_models.py
python scripts/run_suite.py --methods fusion --gpus 0 1 2 --seeds 1 --dry-run
```

The synthetic checks cover all 81 primary configurations, TinyHAR with 12/30 channels, all five fusions, padding masks, finite gradients, and all nine LTCA dataset/backbone combinations. They do not reproduce full LOSO accuracy.

## Attribution and license

This project adapts [WEAR](https://github.com/mariusbock/wear), retaining its [CC BY-NC-SA 4.0 license](LICENSE.md) and existing source notices. The TAL backbones and TinyHAR build on their respective upstream work. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [docs/SOURCE_PROVENANCE.md](docs/SOURCE_PROVENANCE.md) for sources and release changes.
