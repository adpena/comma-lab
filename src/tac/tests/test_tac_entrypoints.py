# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
import sys
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.entrypoints import (
    SCHEMA_VERSION,
    build_postfilter_meta,
    make_dilated_default_tag,
    make_fixed_h32_segnet_tag,
    resolve_cloud_asset_bundle,
    resolve_cloud_asset,
    resolve_cloud_archive_source,
    save_best_checkpoint,
    save_final_artifacts,
)


class TacEntrypointsTests(unittest.TestCase):
    def test_build_postfilter_meta_is_typed_and_versioned(self) -> None:
        meta = build_postfilter_meta(variant="dilated", hidden=64, kernel=3, alpha=20.0)

        self.assertEqual(meta["schema_version"], SCHEMA_VERSION)
        self.assertEqual(meta["variant"], "dilated")
        self.assertEqual(meta["hidden"], 64)
        self.assertEqual(meta["kernel"], 3)
        self.assertEqual(meta["alpha"], 20.0)

    def test_default_tags_are_canonical(self) -> None:
        self.assertEqual(make_dilated_default_tag(64, 20.0), "dilated_qat_ema_h64_a20")
        self.assertEqual(make_fixed_h32_segnet_tag(20.0), "cloud_segnet_attack_h32_a20")

    def test_save_best_checkpoint_writes_versioned_meta(self) -> None:
        model = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1)
        shadow = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        meta = build_postfilter_meta(variant="dilated", hidden=4, kernel=3, alpha=20.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            payload = save_best_checkpoint(
                model=model,
                shadow_state=shadow,
                output_dir=out,
                tag="unit",
                meta=meta,
                epoch=3,
                scorer=1.234,
                per_channel_int8=True,
            )
            on_disk = json.loads((out / "postfilter_unit_best_meta.json").read_text())

        self.assertEqual(payload["schema_version"], SCHEMA_VERSION)
        self.assertEqual(on_disk["schema_version"], SCHEMA_VERSION)
        self.assertEqual(on_disk["epoch"], 3)
        self.assertEqual(on_disk["scorer"], 1.234)

    def test_save_final_artifacts_backfills_best_meta(self) -> None:
        model = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1)
        meta = build_postfilter_meta(variant="cloud_segnet_attack_h32", hidden=32, kernel=3, alpha=20.0)
        best_payload = {
            "schema_version": SCHEMA_VERSION,
            "epoch": 7,
            "scorer": 1.111,
            "meta": meta,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            payload = save_final_artifacts(
                model=model,
                output_dir=out,
                tag="unit",
                meta=meta,
                final_metrics={
                    "baseline_loss": 2.5,
                    "final_loss": 1.5,
                    "final_pose": 0.05,
                    "final_seg": 0.01,
                },
                best_eval_payload=best_payload,
            )
            final_meta = json.loads((out / "postfilter_unit_final_meta.json").read_text())
            best_meta = json.loads((out / "postfilter_unit_best_meta.json").read_text())

        self.assertEqual(payload["schema_version"], SCHEMA_VERSION)
        self.assertEqual(final_meta["schema_version"], SCHEMA_VERSION)
        self.assertEqual(best_meta["epoch"], 7)
        self.assertEqual(final_meta["best_eval"]["scorer"], 1.111)

    def test_resolve_cloud_asset_searches_nested_kaggle_mounts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            project_root = root / "project"
            script_path = project_root / "experiments" / "runner.py"
            nested = root / "input" / "datasets" / "comma-lab-private-assets"
            nested.mkdir(parents=True)
            script_path.parent.mkdir(parents=True)
            script_path.write_text("print('x')\n")
            asset = nested / "decode_base_archive.zip"
            asset.write_bytes(b"zip")

            resolved = resolve_cloud_asset(project_root, script_path, "reports/raw/decode_base_archive.zip", input_root=root / "input")

        self.assertEqual(resolved, asset)

    def test_resolve_cloud_archive_source_falls_back_to_extracted_dataset_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            project_root = root / "project"
            script_path = project_root / "experiments" / "runner.py"
            nested_dir = root / "input" / "datasets" / "comma-lab-private-assets" / "decode_base_archive"
            nested_dir.mkdir(parents=True)
            script_path.parent.mkdir(parents=True)
            script_path.write_text("print('x')\n")
            mkv = nested_dir / "0.mkv"
            mkv.write_bytes(b"mkv")

            resolved = resolve_cloud_archive_source(
                project_root,
                script_path,
                "reports/raw/2026-04-06-av1-roi-experiments/decode_base_archive.zip",
                input_root=root / "input",
            )

        self.assertEqual(resolved, mkv)

    def test_resolve_cloud_asset_bundle_resolves_archive_and_saliency(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            project_root = root / "project"
            script_path = project_root / "experiments" / "runner.py"
            input_root = root / "input" / "datasets" / "comma-lab-private-assets"
            input_root.mkdir(parents=True)
            script_path.parent.mkdir(parents=True)
            script_path.write_text("print('x')\n")
            archive_dir = input_root / "decode_base_archive"
            archive_dir.mkdir()
            mkv = archive_dir / "0.mkv"
            mkv.write_bytes(b"mkv")
            saliency = input_root / "posenet_saliency.npy"
            saliency.write_bytes(b"sal")

            bundle = resolve_cloud_asset_bundle(
                project_root,
                script_path,
                archive_relative_path="reports/raw/2026-04-06-av1-roi-experiments/decode_base_archive.zip",
                saliency_relative_path="experiments/masks/posenet_saliency.npy",
                input_root=root / "input",
            )

        self.assertEqual(bundle["archive_path"], mkv)
        self.assertEqual(bundle["saliency_path"], saliency)
