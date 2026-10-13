# WEAR / XRF V2 惯性与视觉融合时序动作定位

[English README](README.md)

本仓库整理 PCC 论文及其 LTCA 扩展的实际实验代码，包括 TinyHAR、五种融合方法、ActionFormer / TemporalMaxer / TriDet、单模态与编码消融、数据预处理，以及逐折和多 seed 结果。

## 五种融合方法

| 方法名 | 实际实现 |
| --- | --- |
| `tinyhar_cat` | TinyHAR 编码后拼接，再投影到 TAL 输入维度 |
| `tinyhar_gate_sum` | 两个模态投影后，使用互补门控加权求和 |
| `tinyhar_gate_concat` | 保留视觉特征，门控 IMU 特征，拼接后投影 |
| `tinyhar_sensor_aware_gate_concat` | 对 TinyHAR 的 IMU 特征通道进行重标定，再门控拼接 |
| `tinyhar_ltca_gate_concat` | 局部双向跨模态注意力、残差归一化、独立门控与拼接 |

融合模块统一位于 `camera_baseline/actionformer/libs/modeling/meta_archs.py`，另外两个模型复用该实现。传感器感知版本实际使用的是编码后通道重标定，不是显式的逐设备 token 注意力。

消融配置包括：`video_only`、`raw_imu_only`、`tinyhar_imu_only`、`direct_cat`。三个数据/视觉设置 × 三个骨干 × 九个方法，共 **81 个主要配置**。早期 MRAV-FF、残差融合、原始 IMU 投影及归一化配置也保留在 `configs/xrfv2_plus/` 中；没有把这些旧实验写成已完成的 LTCA 内部组件消融。

## 环境

原实验环境为 Linux + Python 3.10.19 + PyTorch 2.0.1/cu117。完整安装命令见英文 README。训练入口使用 CUDA；预处理和合成输入检查可以使用 CPU。

```bash
conda create -n tal-wear-xrfv2 python=3.10 -y
conda activate tal-wear-xrfv2
python -m pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu117
python -m pip install -r requirements.txt
cd camera_baseline/actionformer/libs/utils
python setup.py install
cd ../../../..
```

## 数据

WEAR 默认放在 `data/wear/`，需要官方 `annotations/30fps`、`raw/inertial/25hz` 和 `processed/combined_features/30fps/60_frames_30_stride`。每行 2648 维，顺序为 `[IMU600; RGB1024; Flow1024]`。

按实际配置，WEAR 的 IMU 是 **25 Hz × 50 点，即 2 秒窗口**，视频是 **30 fps × 60 帧，即 2 秒窗口**，步长均为 1 秒。使用官方预计算 I3D 特征，或通过 `tools/prepare_wear.py` 重新拼接数据；该脚本保留官方标注边界。

XRF V2 通过以下命令生成 TAL 输入及 16 折 LOSO：

```bash
python tools/prepare_xrfv2_plus_actionformer.py \
  --release-root /你的路径/XRFv2同步数据集 \
  --output-root data/xrfv2_plus
```

XRF V2 每行 2524 维，为 `[IMU1500; Video1024]`；IMU 按 50 Hz × 5 设备 × 6 通道组织。LOSO 按志愿者划分，同一志愿者的全部录制序列同时留出。

原始 IMU 的设备/指令切分、50 Hz 插值、H5 打包、视频长度对齐、Swin3D 特征提取等代码见 `preprocessing/xrfv2/`；WEAR 原始同步与标注代码见 `data_creation/`。详细输入文件和步骤见 [数据预处理说明](docs/DATA_PREPROCESSING.md)。

## 实验命令

单个方法：

```bash
python scripts/run_loso.py \
  --dataset wear_flow --model actionformer \
  --method tinyhar_ltca_gate_concat --gpu 0 --seed 1
```

数据在外部目录时添加 `--data-root /你的路径/准备后的数据`。默认运行完整 18/16 折，`--dry-run` 只查看配置与命令；`--folds 0,1` 明确运行部分折。

三个模型分别占用 GPU 0、1、2；每个模型先跑 Wear RGB，再跑 Flow，最后排队 XRF V2：

```bash
python scripts/run_suite.py \
  --datasets wear_rgb wear_flow xrfv2 \
  --models actionformer temporalmaxer tridet \
  --methods all --gpus 0 1 2 --seeds 1
```

`--methods fusion` 只跑五种融合；`--methods controls` 只跑四个对照；`--seeds 1 2 3` 指定固定种子列表。后台执行方式见英文 README。

## 结果

论文 mAP 为 tIoU **0.3、0.4、0.5、0.6、0.7、0.8、0.9** 七阈值平均；原日志的 `Avg. mAP` 可能是 17 阈值平均，不能直接替代。

```bash
python scripts/summarize_results.py --logs logs --output results/new_runs
```

已整理的 `results/` 包含论文 81 行原表数值、实验来源匹配、全部可恢复逐折记录及 seed 统计。Wear Flow + ActionFormer 的 LTCA **73.49 来自 seed 13 的多次运行后选择**；XRF V2 + ActionFormer 的 **89.45 在显示精度上是持平**。原表值、最终轮次结果、历史 best-epoch 对照和多种子平均分别记录，详见 [结果口径说明](docs/RESULTS.md)。

## 检查与代码来源

```bash
python -m unittest discover -s tests -p 'test_*.py'
python tests/smoke_models.py
```

代码来自实际实验服务器，模型实现和 TinyHAR 保留原版本；本次新增了可移植入口、数据组装器、结果导出及测试，整理了固定路径。没有在本次整理中重新训练 LOSO，也没有打包数据集、权重或账号凭据。

保留上游 WEAR 的许可证与来源声明，详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
