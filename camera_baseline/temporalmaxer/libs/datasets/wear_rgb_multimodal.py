import os
import json
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

from .datasets import register_dataset
from .data_utils import truncate_feats


@register_dataset("wear_rgb_multimodal")
class WearRGBMultimodalDataset(Dataset):
    def __init__(
        self,
        is_training,
        split,
        feat_folder,
        sens_folder,
        sampling_rate,
        json_anno,
        feat_stride,
        num_frames,
        default_fps,
        downsample_rate,
        max_seq_len,
        trunc_thresh,
        crop_ratio,
        tiou_thresholds,
        input_dim,
        num_classes,
        file_prefix,
        file_ext,
        force_upsampling,
        imu_flat_dim=600,
        video_dim=1024,
        imu_window_size=50,
        imu_input_dim=12,
        rgb_start=600,
        video_feature="rgb_i3d",
        feature_mode="multimodal",
        imu_normalize="none",
        imu_eps=1e-5,
    ):
        assert os.path.exists(feat_folder) and os.path.exists(json_anno)
        assert isinstance(split, tuple) or isinstance(split, list)
        assert crop_ratio is None or len(crop_ratio) == 2
        self.feat_folder = feat_folder
        self.sens_folder = sens_folder
        self.sampling_rate = sampling_rate
        self.file_prefix = file_prefix if file_prefix is not None else ""
        self.file_ext = file_ext
        self.json_anno = json_anno
        self.db_attributes = {
            "dataset_name": "wear_rgb_multimodal",
            "tiou_thresholds": tiou_thresholds,
            "empty_label_ids": [],
        }
        self.force_upsampling = force_upsampling
        self.split = split
        self.is_training = is_training
        self.feat_stride = feat_stride
        self.num_frames = num_frames
        self.input_dim = input_dim
        self.default_fps = default_fps
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes
        self.label_dict = None
        self.crop_ratio = crop_ratio
        self.imu_flat_dim = int(imu_flat_dim)
        self.video_dim = int(video_dim)
        self.imu_window_size = int(imu_window_size)
        self.imu_input_dim = int(imu_input_dim)
        self.rgb_start = int(rgb_start)
        self.video_feature = video_feature
        self.feature_mode = feature_mode.lower()
        self.imu_normalize = imu_normalize
        self.imu_eps = float(imu_eps)
        assert self.imu_flat_dim == self.imu_window_size * self.imu_input_dim
        assert self.feature_mode in (
            "multimodal",
            "direct_cat",
            "video_only",
            "raw600_imu_only",
            "raw_imu_only",
        )
        dict_db, label_dict = self._load_json_db(self.json_anno)
        assert len(label_dict) == num_classes
        self.data_list = dict_db
        self.label_dict = label_dict

    def get_attributes(self):
        return self.db_attributes

    def _load_json_db(self, json_anno):
        with open(json_anno, "r", encoding="utf-8") as fid:
            json_data = json.load(fid)
        json_db = json_data["database"]

        if self.label_dict is None:
            label_dict = {}
            for _, value in json_db.items():
                for act in value["annotations"]:
                    label_dict[act["label"]] = act["label_id"]

        dict_db = tuple()
        for key, value in json_db.items():
            if value["subset"].lower() not in self.split:
                continue
            feat_file = os.path.join(self.feat_folder, self.file_prefix + key + self.file_ext)
            if not os.path.exists(feat_file):
                continue

            if self.default_fps is not None:
                fps = self.default_fps
            elif "fps" in value:
                fps = value["fps"]
            else:
                assert False, "Unknown video FPS."

            duration = value["duration"] if "duration" in value else 1e8
            if ("annotations" in value) and (len(value["annotations"]) > 0):
                segments, labels = [], []
                for act in value["annotations"]:
                    segments.append(act["segment"])
                    labels.append([label_dict[act["label"]]])
                segments = np.asarray(segments, dtype=np.float32)
                labels = np.squeeze(np.asarray(labels, dtype=np.int64), axis=1)
            else:
                segments, labels = None, None

            dict_db += ({
                "id": key,
                "fps": fps,
                "duration": duration,
                "segments": segments,
                "labels": labels,
            },)
        return dict_db, label_dict

    def __len__(self):
        return len(self.data_list)

    def _split_combined_features(self, combined):
        min_dim = self.rgb_start + self.video_dim
        if combined.shape[1] < min_dim:
            raise ValueError("Expected at least {} dims, got {}".format(min_dim, combined.shape[1]))
        raw_imu = combined[:, : self.imu_flat_dim]
        video = combined[:, self.rgb_start : self.rgb_start + self.video_dim]

        # Stored order is 12 channels x 50 samples; TinyHAR expects 50 samples x 12 channels.
        imu_time_major = raw_imu.reshape(
            combined.shape[0], self.imu_input_dim, self.imu_window_size
        ).transpose(0, 2, 1)

        if self.imu_normalize == "window":
            mean = imu_time_major.mean(axis=1, keepdims=True)
            std = imu_time_major.std(axis=1, keepdims=True)
            imu_time_major = (imu_time_major - mean) / np.maximum(std, self.imu_eps)
        elif self.imu_normalize == "sequence":
            mean = imu_time_major.reshape(-1, self.imu_input_dim).mean(axis=0, keepdims=True)
            std = imu_time_major.reshape(-1, self.imu_input_dim).std(axis=0, keepdims=True)
            imu_time_major = (
                imu_time_major - mean.reshape(1, 1, self.imu_input_dim)
            ) / np.maximum(std.reshape(1, 1, self.imu_input_dim), self.imu_eps)
        elif self.imu_normalize not in ("none", None):
            raise ValueError("Unknown imu_normalize: {}".format(self.imu_normalize))

        imu_time_major = np.nan_to_num(imu_time_major, nan=0.0, posinf=0.0, neginf=0.0)
        tinyhar_imu = imu_time_major.reshape(combined.shape[0], self.imu_flat_dim)
        return video.astype(np.float32), raw_imu.astype(np.float32), tinyhar_imu.astype(np.float32)

    def _select_features(self, combined):
        video, raw_imu, tinyhar_imu = self._split_combined_features(combined)
        if self.feature_mode == "multimodal":
            return video, tinyhar_imu
        if self.feature_mode == "direct_cat":
            return np.concatenate([raw_imu, video], axis=1), None
        if self.feature_mode == "video_only":
            return video, None
        if self.feature_mode in ("raw600_imu_only", "raw_imu_only"):
            return raw_imu, None
        raise ValueError("Unknown feature_mode: {}".format(self.feature_mode))

    def __getitem__(self, idx):
        video_item = self.data_list[idx]
        filename = os.path.join(self.feat_folder, self.file_prefix + video_item["id"] + self.file_ext)
        combined = np.load(filename).astype(np.float32)
        feats, additional_feats = self._select_features(combined)

        if self.force_upsampling:
            feat_stride = float((feats.shape[0] - 1) * self.feat_stride + self.num_frames) / self.max_seq_len
            num_frames = feat_stride
        else:
            feat_stride, num_frames = self.feat_stride, self.num_frames
            feats = feats[:: self.downsample_rate, :]
            if additional_feats is not None:
                additional_feats = additional_feats[:: self.downsample_rate, :]
            feat_stride = self.feat_stride * self.downsample_rate

        feat_offset = 0.5 * num_frames / feat_stride
        feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))
        if additional_feats is not None:
            additional_feats = torch.from_numpy(np.ascontiguousarray(additional_feats.transpose()))

        if (feats.shape[-1] != self.max_seq_len) and self.force_upsampling:
            feats = F.interpolate(feats.unsqueeze(0), size=self.max_seq_len, mode="linear", align_corners=False).squeeze(0)
            if additional_feats is not None:
                additional_feats = F.interpolate(
                    additional_feats.unsqueeze(0), size=self.max_seq_len, mode="linear", align_corners=False
                ).squeeze(0)

        if video_item["segments"] is not None:
            segments = torch.from_numpy(video_item["segments"] * video_item["fps"] / feat_stride - feat_offset)
            labels = torch.from_numpy(video_item["labels"])
        else:
            segments, labels = None, None

        data_dict = {
            "video_id": video_item["id"],
            "feats": feats,
            "segments": segments,
            "labels": labels,
            "fps": video_item["fps"],
            "duration": video_item["duration"],
            "feat_stride": feat_stride,
            "feat_num_frames": self.num_frames,
        }
        if additional_feats is not None:
            data_dict["additional_feats"] = additional_feats

        if self.is_training and (segments is not None):
            data_dict = truncate_feats(data_dict, self.max_seq_len, self.trunc_thresh, feat_offset, self.crop_ratio)
        return data_dict
