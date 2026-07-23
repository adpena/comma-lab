# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
TOOL = REPO / "tools" / "run_ddm_sn1_segnet_telemetry_asymmetry.py"
SPEC = importlib.util.spec_from_file_location("run_ddm_sn1_segnet_telemetry_asymmetry", TOOL)
assert SPEC and SPEC.loader
tool = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tool
SPEC.loader.exec_module(tool)


def _config(tmp_path: Path) -> dict[str, object]:
    return {
        "schema": tool.CONFIG_SCHEMA,
        "run_id": "test",
        "video_path": "/fixture/0.mkv",
        "video_sha256": "1" * 64,
        "gt_cache_path": "/fixture/gt.npz",
        "gt_cache_sha256": "2" * 64,
        "upstream_root": "/fixture/upstream",
        "upstream_modules_sha256": "3" * 64,
        "segnet_weights_sha256": "4" * 64,
        "output_directory": str(tmp_path / "evidence"),
        "scratch_directory": "/Volumes/VertigoDataTier/pact/test-sn1",
        "pair_count": 600,
        "batch_pairs": 8,
        "torch_threads": 4,
        "seed": 1234,
        "erf_pair_ids": [0, 299, 599],
        "inverse_segment_count": 3,
        "inverse_max_steps": 8,
        "inverse_max_linf": 16,
        "inverse_camera_radius": 64,
        "checkpoint_policy": "atomic_preserve_each_batch_resume_by_sha",
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
    }


def test_typed_config_requires_n600_and_false_authority(tmp_path: Path) -> None:
    value = _config(tmp_path)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    config, digest = tool.load_config(path)
    assert config.pair_count == 600
    assert len(digest) == 64
    value["pair_count"] = 64
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(tool.DDMSN1Error, match="n600"):
        tool.load_config(path)


def test_segment_candidates_are_connected_and_directional() -> None:
    coords = torch.tensor(
        [[0, 1, 1], [0, 1, 2], [0, 1, 3], [0, 5, 5], [0, 5, 6]],
        dtype=torch.int64,
    )
    samples = {
        "Road->Lane": {
            "coordinates_nyx": coords,
            "margins": torch.tensor([0.3, 0.1, 0.2, 0.01, 0.02]),
        }
    }
    rows = tool.select_segment_candidates(
        samples,
        pair_id=7,
        height=8,
        width=8,
    )
    assert len(rows) == 1
    assert rows[0]["orientation"] == "Road->Lane"
    assert rows[0]["coordinates_yx"] == [[1, 2], [1, 3], [1, 1]]
    assert rows[0]["segment_pixel_count"] == 3


def test_accumulator_preserves_missing_final_decoder_skip() -> None:
    accumulator = tool._empty_accumulator()
    summary = {
        "per_class_logit_energy": dict.fromkeys(tool.CLASS_NAMES, 1.0),
        "layer_boundary_energy": {
            "decoder.blocks.4.skip": {
                "present": False,
                "reason": "decoder block has no skip connection",
            },
            "segmentation_head": {
                "present": True,
                "shape": [1, 5, 2, 2],
                "boundary_sample_count": 2,
                "interior_sample_count": 2,
                "boundary_mean_square": 3.0,
                "interior_mean_square": 1.0,
            },
        },
    }
    tool._accumulate_summary(accumulator, summary)
    assert accumulator["layers"]["decoder.blocks.4.skip"]["present"] is False
    assert accumulator["layers"]["segmentation_head"]["boundary_energy_sum"] == 6.0


def test_batch_resume_refuses_implementation_drift(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_config(tmp_path)), encoding="utf-8")
    config, config_sha = tool.load_config(config_path)
    implementation = {
        "schema": "ddm_sn1_implementation_identity.v1",
        "files": {},
        "bundle_sha256": "a" * 64,
    }
    margin_path = tmp_path / "margins.npz"
    margin_path.write_bytes(b"fixture")
    batch_path = tmp_path / "batch.json"
    batch = {
        "schema": tool.BATCH_SCHEMA,
        "run_id": config.run_id,
        "config_sha256": config_sha,
        "implementation_identity": implementation,
        "pair_window": [0, 8],
        "pair_count": 8,
        "argmax_cache_mismatch_count": 0,
        "margin_arrays": {
            "path": str(margin_path),
            "bytes": margin_path.stat().st_size,
            "sha256": tool.sha256_file(margin_path),
        },
    }
    batch_path.write_text(json.dumps(batch), encoding="utf-8")
    assert tool._valid_batch(
        batch_path,
        config=config,
        config_sha256=config_sha,
        implementation=implementation,
        start=0,
        stop=8,
    )
    assert not tool._valid_batch(
        batch_path,
        config=config,
        config_sha256=config_sha,
        implementation={**implementation, "bundle_sha256": "b" * 64},
        start=0,
        stop=8,
    )
