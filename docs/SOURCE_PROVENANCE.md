# Source provenance

Release organized on 2026-10-13 (Asia/Shanghai) from the actual PCC/LTCA experiment checkout.

The experiment checkout derives from `mariusbock/wear`, upstream commit `cfd4817584a059871fbf5d9163c75a61a7bb9a16`, with uncommitted research adaptations. That upstream commit alone is not the complete research version.

The following files were copied without changing their model implementations:

| File | SHA-256 |
| --- | --- |
| camera_baseline/actionformer/libs/modeling/meta_archs.py | 6946b516a9b6306f21a4e344f98a697a762fa3b124bdfc013f6211d2a0badd98 |
| camera_baseline/temporalmaxer/libs/modeling/meta_archs.py | 9c68951cf6f6970fbf31bee9b85a29b6b4d43783c71d40d3d3c95ff0ef8764ad |
| camera_baseline/tridet/libs/modeling/meta_archs.py | 960374713bbd16415de34af69557ab413dbca9f284442499ce2321e8b6a7954c |
| inertial_baseline/TinyHAR.py | 164fd5226629030bcde599dc68e68a590ddf182c5321264f981f18b814f6b965 |

Release preparation changes:

- Replace fixed WEAR dataset paths with repository-relative defaults in paper configurations.
- Use the current Python interpreter and the script's repository location in batch scripts.
- Add portable single-run and bounded multi-GPU suite launchers.
- Add the WEAR combined-feature assembler, exact manuscript-table exporter, seven-threshold result exporter, and synthetic checks.
- Make XRF V2 source preprocessing paths configurable through `XRFV2_SOURCE_ROOT`; discover ffprobe on PATH; allow IMU-only duration trimming.
- Add missing dependency declarations and retain original licenses/source notices.
- Archive earlier experiment-planning/status documents under `docs/archive/`; these documents are historical and can refer to superseded experiments.

No manuscript, raw dataset, feature array or trained checkpoint is part of this code release. No full LOSO retraining was performed during release preparation. Model verification uses synthetic inputs.
