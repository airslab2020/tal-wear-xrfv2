# Method and configuration map

The shared fusion implementation is in `camera_baseline/actionformer/libs/modeling/meta_archs.py`. TemporalMaxer and TriDet import the same encoder/fusion factory rather than defining separate variants.

| Paper method | Model IMU encoder | Fusion type |
| --- | --- | --- |
| tinyhar_cat | tinyhar | concat_project |
| tinyhar_gate_sum | tinyhar | gate_sum |
| tinyhar_gate_concat | tinyhar | gate_concat |
| tinyhar_sensor_aware_gate_concat | tinyhar | sensor_aware_gate_concat |
| tinyhar_ltca_gate_concat | tinyhar | ltca_gate_concat |
| tinyhar_imu_only | tinyhar | imu_only |

The unencoded controls use the dataset's feature-selection path: video-only, flattened IMU-only, or directly concatenated flattened IMU and video. They use the corresponding original single-stream TAL architecture.

## LTCA

Each modality is projected to 256 channels. IMU queries local video windows, and video queries local IMU windows, using two separate four-head attention modules. The default window length is 5 temporal feature positions, with radius 2. Attention outputs are added to the respective projected features and normalized. A gate network consumes both original projected streams and both enhanced streams; two sigmoid gates independently weight the enhanced modalities. Their concatenation is 512-D.

The implementation pads local windows and masks invalid keys. Invalid padded queries expose zero windows to avoid fully masked attention producing NaNs; the final padding mask removes their output. The default attention/gate dropout is 0.1.

The feature step is one second in both paper settings, but a WEAR feature already represents a two-second clip/window, whereas an XRF V2 feature represents one second. Therefore a five-position attention window must not be confused with five disjoint raw IMU windows on WEAR.

## Controls

| Comparison | Question it evaluates |
| --- | --- |
| video_only vs raw_imu_only | Contribution of the two modalities individually |
| raw_imu_only vs tinyhar_imu_only | Learned IMU representation vs flattening |
| direct_cat vs tinyhar_cat | IMU encoding before multimodal fusion |
| five TinyHAR fusion variants | Effect of the fusion strategy with a shared encoder |
| Wear RGB vs Wear Flow | Effect of visual feature type |
| three backbones | Transfer of fusion modules across TAL architectures |

An LTCA window-size sweep, removal of one attention direction, removal of residuals, or removal of reliability gates would be a separate component ablation. No results for those unrun ablations are claimed in this release.
