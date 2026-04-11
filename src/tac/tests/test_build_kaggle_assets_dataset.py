from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tac.deploy.kaggle import build_kaggle_assets_dataset as mod


class BuildKaggleAssetsDatasetTests(unittest.TestCase):
    def test_render_dataset_readme_lists_runtime_assets(self) -> None:
        text = mod.render_dataset_readme(
            wheel_name="comma_video_lab_ball_pack-0.7.0-py3-none-any.whl",
            archive_name="decode_base_archive.zip",
            saliency_name="posenet_saliency.npy",
        )

        self.assertIn("comma_video_lab_ball_pack-0.7.0-py3-none-any.whl", text)
        self.assertIn("decode_base_archive.zip", text)
        self.assertIn("posenet_saliency.npy", text)

    def test_stage_assets_copies_required_runtime_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            dataset = root / "dataset"
            source.mkdir()
            dataset.mkdir()

            wheel = source / "comma_video_lab_ball_pack-0.7.0-py3-none-any.whl"
            archive = source / "decode_base_archive.zip"
            saliency = source / "posenet_saliency.npy"
            metadata = source / "dataset-metadata.json"
            readme = source / "README.md"

            wheel.write_bytes(b"wheel")
            archive.write_bytes(b"archive")
            saliency.write_bytes(b"saliency")
            metadata.write_text(json.dumps({"id": "alice/private-assets"}))
            readme.write_text("# assets\n")

            staged = mod.stage_assets_dataset(
                dataset_dir=dataset,
                wheel_path=wheel,
                archive_path=archive,
                saliency_path=saliency,
                metadata_path=metadata,
            )

            self.assertEqual(staged["wheel_name"], wheel.name)
            self.assertTrue((dataset / wheel.name).exists())
            self.assertTrue((dataset / archive.name).exists())
            self.assertTrue((dataset / saliency.name).exists())
            readme_text = (dataset / "README.md").read_text()
            self.assertIn(wheel.name, readme_text)
            self.assertIn(archive.name, readme_text)
            self.assertIn(saliency.name, readme_text)

    def test_stage_assets_is_idempotent_when_metadata_and_readme_already_live_in_dataset_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            dataset = root / "dataset"
            source.mkdir()
            dataset.mkdir()

            wheel = source / "comma_video_lab_ball_pack-0.7.0-py3-none-any.whl"
            archive = source / "decode_base_archive.zip"
            saliency = source / "posenet_saliency.npy"
            metadata = dataset / "dataset-metadata.json"
            readme = dataset / "README.md"

            wheel.write_bytes(b"wheel")
            archive.write_bytes(b"archive")
            saliency.write_bytes(b"saliency")
            metadata.write_text(json.dumps({"id": "alice/private-assets"}))
            readme.write_text("# assets\n")

            staged = mod.stage_assets_dataset(
                dataset_dir=dataset,
                wheel_path=wheel,
                archive_path=archive,
                saliency_path=saliency,
                metadata_path=metadata,
            )

            self.assertEqual(staged["wheel_name"], wheel.name)
            self.assertIn(wheel.name, (dataset / "README.md").read_text())


if __name__ == "__main__":
    unittest.main()
