import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models.video import swin3d_b, Swin3D_B_Weights

BASE_DIR = Path(os.environ.get("XRFV2_SOURCE_ROOT", "./data/xrfv2_source"))
VIDEO_ROOT = BASE_DIR / "video_cropped"
SAVE_ROOT = BASE_DIR / "video_feature"
SAVE_ROOT.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
target_fps = 15
batch_size = 2

weights = Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1
model = swin3d_b(weights=weights)
model.head = torch.nn.Identity()
model.eval().to(device)

preprocess = weights.transforms()


def get_all_samples():
    samples = []
    for scene_dir in sorted(VIDEO_ROOT.iterdir()):
        if not scene_dir.is_dir():
            continue
        scene_name = scene_dir.name
        for sample_dir in sorted(scene_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            video_path = sample_dir / "cropped_output.mkv"
            if video_path.exists():
                samples.append((scene_name, sample_dir.name, video_path))
    return samples


def load_video_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"no frames read from: {video_path}")

    return frames, fps, frame_count


def clip_to_tensor(clip_frames):
    arr = np.stack(clip_frames, axis=0)                  # (T, H, W, C)
    tensor = torch.from_numpy(arr).permute(0, 3, 1, 2)  # (T, C, H, W)
    tensor = preprocess(tensor)                          # (C, T, H, W)
    return tensor


def build_secondwise_clips(frames, clip_len, fps):
    fps = int(round(fps))
    num_seconds = math.ceil(len(frames) / fps)
    start_indices = [sec * fps for sec in range(num_seconds)]

    clips = []
    timestamps = []

    for start in start_indices:
        if start >= len(frames):
            break

        clip_frames = frames[start:start + clip_len]
        if len(clip_frames) == 0:
            continue

        while len(clip_frames) < clip_len:
            clip_frames.append(clip_frames[-1].copy())

        clips.append(clip_to_tensor(clip_frames))
        timestamps.append(start / fps)

    return clips, timestamps


def extract_features_in_batches(clips):
    outputs = []
    with torch.no_grad():
        for i in range(0, len(clips), batch_size):
            batch = torch.stack(clips[i:i + batch_size], dim=0).to(device)
            feats = model(batch)
            outputs.append(feats.cpu())
    return torch.cat(outputs, dim=0)


def process_one_sample(scene_name, sample_name, video_path):
    save_dir = SAVE_ROOT / scene_name / sample_name
    save_dir.mkdir(parents=True, exist_ok=True)

    frames, fps, frame_count = load_video_frames(video_path)

    clip_len = 15
    clips, timestamps = build_secondwise_clips(
        frames=frames,
        clip_len=clip_len,
        fps=target_fps,
    )

    features = extract_features_in_batches(clips)
    timestamps = torch.tensor(timestamps, dtype=torch.float32)

    torch.save(
        {
            "features": features,
            "timestamps_sec": timestamps,
            "clip_len": clip_len,
            "fps": target_fps,
            "video_path": str(video_path),
            "scheme": "scheme2_15frames_secondwise"
        },
        save_dir / "scheme2_15frames_secondwise.pt"
    )

    np.save(save_dir / "scheme2_15frames_secondwise.npy", features.numpy())
    np.save(save_dir / "scheme2_15frames_secondwise_timestamps.npy", timestamps.numpy())

    print(
        f"[OK] {scene_name}/{sample_name} | "
        f"frames={frame_count} | "
        f"feature={tuple(features.shape)} | "
        f"time={tuple(timestamps.shape)}"
    )


def main():
    samples = get_all_samples()
    if len(samples) == 0:
        raise RuntimeError(f"no cropped videos found under: {VIDEO_ROOT}")

    ok = 0
    err = 0

    for scene_name, sample_name, video_path in samples:
        try:
            process_one_sample(scene_name, sample_name, video_path)
            ok += 1
        except Exception as e:
            print(f"[ERR] {scene_name}/{sample_name}: {e}")
            err += 1

    print("\n======================")
    print(f"success: {ok}")
    print(f"failed: {err}")
    print(f"output dir: {SAVE_ROOT}")
    print("======================")


if __name__ == "__main__":
    main()
