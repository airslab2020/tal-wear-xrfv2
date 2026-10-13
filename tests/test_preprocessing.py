"""Check XRF V2 alignment/LOSO construction and WEAR channel ordering."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


class PreprocessingTest(unittest.TestCase):
    def test_xrfv2_release_bridge(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            release = root / "release"
            annotations = release / "annotations/tal_annotations_video_aligned"
            annotations.mkdir(parents=True)
            (release / "data").mkdir()
            names = ["0_kitchen_1", "1_office_1"]
            database = {
                name: {"volunteer_id": i, "duration": 3, "scene": i,
                       "annotations": [{"segment": [0., 4.], "label": "action", "label_id": 0}]}
                for i, name in enumerate(names)
            }
            (annotations / "activitynet_tal_853.json").write_text(json.dumps({"database": database}))
            (annotations / "label_map.json").write_text(json.dumps({"action": 0}))
            imu = np.arange(150 * 5 * 6, dtype=np.float32).reshape(150, 5, 6)
            with h5py.File(release / "data/imu_50hz_853_video_aligned.h5", "w") as handle:
                for name in names:
                    handle.create_dataset(f"samples/{name}/imu", data=imu)
            with h5py.File(release / "data/video_feature_853_video_aligned.h5", "w") as handle:
                for name in names:
                    handle.create_dataset(f"samples/{name}/video_feature", data=np.ones((2, 1024)))
            output = root / "prepared"
            subprocess.run([
                sys.executable, str(ROOT / "tools/prepare_xrfv2_plus_actionformer.py"),
                "--release-root", str(release), "--output-root", str(output),
            ], check=True)
            combined = np.load(output / "processed/combined_features/imu_flatten_video_1hz/sbj_0.npy")
            self.assertEqual(combined.shape, (2, 2524))
            np.testing.assert_array_equal(combined[0, :1500], imu[:50].transpose(1, 2, 0).reshape(-1))
            for i in range(2):
                fold = json.loads((output / f"annotations/loso/xrfv2_plus_loso_s{i:02d}.json").read_text())
                heldout = [k for k, v in fold["database"].items() if v["subset"] == "Validation"]
                self.assertEqual(heldout, [f"sbj_{i}"])
                self.assertEqual(fold["database"][f"sbj_{i}"]["annotations"][0]["segment"], [0., 2.])

    def test_wear_assembler(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            imu_dir, video_dir, annotation_dir = [root / p for p in ("imu", "video", "annotations")]
            for p in (imu_dir, video_dir, annotation_dir):
                p.mkdir()
            raw = np.arange(100 * 12, dtype=np.float32).reshape(100, 12)
            for subject in range(18):
                frame = pd.DataFrame(raw, columns=[f"acc_{i}" for i in range(12)])
                frame.insert(0, "sbj_id", subject)
                frame["label"] = "null"
                frame.to_csv(imu_dir / f"sbj_{subject}.csv", index=False)
                np.save(video_dir / f"sbj_{subject}.npy", np.ones((2, 2048), dtype=np.float32))
                database = {
                    f"sbj_{i}": {"subset": "Validation" if i == subject else "Training",
                                "duration": 4., "fps": 30., "annotations": []}
                    for i in range(18)
                }
                (annotation_dir / f"wear_split_{subject + 1}.json").write_text(json.dumps({"database": database}))
            output = root / "prepared"
            subprocess.run([
                sys.executable, str(ROOT / "tools/prepare_wear.py"),
                "--imu-dir", str(imu_dir), "--i3d-dir", str(video_dir),
                "--annotation-dir", str(annotation_dir), "--output-root", str(output),
            ], check=True)
            combined = np.load(output / "processed/combined_features/30fps/60_frames_30_stride/sbj_0.npy")
            self.assertEqual(combined.shape, (2, 2648))
            np.testing.assert_array_equal(combined[0, :600], raw[:50].T.reshape(-1))
            np.testing.assert_array_equal(combined[1, :600], raw[25:75].T.reshape(-1))
            self.assertEqual(
                (output / "annotations/30fps/wear_split_1.json").read_bytes(),
                (annotation_dir / "wear_split_1.json").read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()
