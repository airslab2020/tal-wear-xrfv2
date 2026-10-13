#!/usr/bin/env python3
"""Synthetic integration checks. No dataset, training loop, or checkpoints."""
from __future__ import annotations

import argparse
import copy
import importlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import torch
import yaml

from run_loso import DATASETS, FUSIONS, METHODS, MODELS, config_path, resolved_config
from camera_baseline.actionformer.libs.modeling.meta_archs import (
    TinyHAREncoder, make_multimodal_fusion,
)


def check_configs():
    count = 0
    for dataset in DATASETS:
        for model in MODELS:
            for method in METHODS:
                original = yaml.safe_load(config_path(dataset, model, method).read_text())
                cfg = resolved_config(dataset, model, method)
                assert len(cfg["anno_json"]) == (16 if dataset == "xrfv2" else 18)
                for key in ("model", "opt", "loader"):
                    assert original[key] == cfg[key], (dataset, model, method, key)
                thresholds = {round(float(t), 2) for t in cfg["dataset"]["tiou_thresholds"]}
                assert set((.3, .4, .5, .6, .7, .8, .9)) <= thresholds
                count += 1
    print(f"PASS {count} paper configurations", flush=True)


def check_fusions():
    for channels in (12, 30):
        encoder = TinyHAREncoder(input_dim=channels, project_dim=1024).eval()
        raw = torch.randn(2, 50 * channels, 7)
        with torch.no_grad():
            imu = encoder(raw)
        assert imu.shape == (2, 1024, 7)
        mask = torch.ones(2, 1, 7, dtype=torch.bool)
        mask[1, :, 4:] = False
        for method in FUSIONS:
            cfg = resolved_config("xrfv2", "actionformer", method)["model"]
            fusion = make_multimodal_fusion(
                cfg["fusion_type"], video_dim=1024, imu_dim=1024,
                embd_dim=cfg["fusion_dim"], fusion_kwargs=cfg.get("fusion_kwargs", {}),
            )
            video = torch.randn(2, 1024, 7, requires_grad=True)
            inertial = imu.detach().requires_grad_()
            output = fusion(video, inertial, mask)
            assert output.shape == (2, cfg["fusion_dim"], 7)
            assert torch.isfinite(output).all()
            assert torch.count_nonzero(output[1, :, 4:]) == 0
            output.square().mean().backward()
            assert video.grad is not None and torch.isfinite(video.grad).all()
            assert inertial.grad is not None and torch.isfinite(inertial.grad).all()
            if method == "tinyhar_ltca_gate_concat":
                fusion.eval()
                with torch.no_grad():
                    empty = fusion(video.detach(), inertial.detach(), torch.zeros_like(mask))
                    assert torch.isfinite(empty).all() and torch.count_nonzero(empty) == 0
        print(f"PASS TinyHAR ({channels} channels) + five fusions, masks and gradients", flush=True)


def check_backbones():
    for dataset in DATASETS:
        for model in MODELS:
            cfg = resolved_config(dataset, model, "tinyhar_ltca_gate_concat")
            module = importlib.import_module(f"camera_baseline.{model}.libs.core.config")
            cfg = module._update_config(copy.deepcopy(cfg))
            cfg["model"]["max_seq_len"] = 256
            factory = importlib.import_module(f"camera_baseline.{model}.libs.modeling").make_meta_arch
            net = factory(cfg["model"]["model_name"], **cfg["model"])
            net.train()
            item = {
                "video_id": "synthetic", "feats": torch.randn(1024, 64),
                "additional_feats": torch.randn(50 * cfg["model"]["imu_input_dim"], 64),
                "segments": torch.tensor([[4., 20.]]), "labels": torch.tensor([0]),
                "fps": 1., "duration": 64., "feat_stride": 1., "feat_num_frames": 1,
            }
            losses = net([item])
            loss = losses["final_loss"]
            assert torch.isfinite(loss), (dataset, model, losses)
            loss.backward()
            assert any(p.grad is not None for p in net.imu_encoder.parameters())
            assert any(p.grad is not None for p in net.fusion_module.parameters())
            print(f"PASS {dataset}/{model}: end-to-end loss and backward", flush=True)
            del net


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-backbones", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(1)
    check_configs()
    check_fusions()
    if not args.skip_backbones:
        check_backbones()
