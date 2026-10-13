# Third-party sources and attribution

This repository is a research adaptation of [Marius Bock et al.'s WEAR code](https://github.com/mariusbock/wear), retaining its original copyright/source notices and [CC BY-NC-SA 4.0 license text](LICENSE.md). Changes and additional research modules are described in [docs/SOURCE_PROVENANCE.md](docs/SOURCE_PROVENANCE.md).

The incorporated TAL implementations originate from the WEAR adaptation and/or their upstream projects:

- [ActionFormer](https://github.com/happyharrycn/actionformer_release): temporal transformer backbone, detection head, evaluation and temporal NMS utilities.
- [TemporalMaxer](https://github.com/TuanTNG/TemporalMaxer): pooling-based temporal backbone and associated detection components.
- [TriDet](https://github.com/dingfengshi/TriDet): relative boundary modeling and trident detection head.
- TinyHAR: the TinyHAR implementation in `inertial_baseline/TinyHAR.py`, used for per-window IMU encoding. Please cite the TinyHAR paper when using this encoder.

Existing file-level upstream notices remain in the source. Third-party packages and pretrained torchvision / I3D models remain subject to their respective upstream licenses and distribution terms; this release does not relicense them.

Dataset sources:

- WEAR: Marius Bock, Hilde Kuehne, Kristof Van Laerhoven, and Michael Moeller. “WEAR: An Outdoor Sports Dataset for Wearable and Egocentric Activity Recognition.” IMWUT, 2024. DOI: [10.1145/3699776](https://doi.org/10.1145/3699776).
- XRF V2: [airslab2020/XRFV2](https://github.com/airslab2020/XRFV2), for the synchronized IMU/video release and its dataset citation.

Raw datasets and pretrained/experiment weights are not redistributed in this repository.
