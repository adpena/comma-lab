# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from experiments.plan_c067_decoded_delta_overlay_mask_topology import (
    RUN_STRUCT_NAME,
    _encode_overlay_payload,
    _mask_tensor_sha256,
    _runs_from_selected,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
BUILDER_PATH = REPO_ROOT / "experiments" / "build_c067_reversed_base_cdo1_candidate.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _runtime_archive(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("renderer.bin", b"R" * 10_001)
        zf.writestr("masks.mkv", b"M" * 1_200)
        zf.writestr("optimized_poses.bin", b"P" * 2_000)


def test_reversed_base_builder_rebuilds_planner_payload_and_archive(tmp_path: Path) -> None:
    builder = _load_module("build_reversed_base_cdo1_candidate_test", BUILDER_PATH)
    base = np.zeros((2, 384, 512), dtype=np.uint8)
    target = base.copy()
    target[0, 12, 3:7] = 1
    target[1, 13, 9:11] = 1
    selected = base != target
    runs = _runs_from_selected(base, target, selected)
    raw = _encode_overlay_payload(
        runs=runs,
        header={
            "schema": "c067_decoded_delta_overlay_payload_v1",
            "producer": "experiments/plan_c067_reversed_base_cdo1_overlay_economics.py",
            "score_claim": False,
            "base_mask_tensor_sha256": _mask_tensor_sha256(base),
            "target_mask_tensor_sha256": _mask_tensor_sha256(target),
            "reconstructed_mask_u8_sha256": _mask_tensor_sha256(target),
            "shape": [2, 384, 512],
            "pair_index_basis": "video_frame_pair_index",
            "run_struct": RUN_STRUCT_NAME,
            "run_count": len(runs),
            "selected_pixel_count": int(selected.sum()),
            "selected_pair_indices": [0],
            "policy_id": "full_repair_to_c067_decoded_mask",
        },
    )
    base_npz = tmp_path / "base.npz"
    target_npy = tmp_path / "target.npy"
    np.savez_compressed(base_npz, masks=base)
    np.save(target_npy, target)
    economics = {
        "schema": "c067_reversed_base_cdo1_overlay_economics_v1",
        "target_decoded_mask": {"path": str(target_npy)},
        "gates": {"max_residual_disagreement_fraction": 0.001},
        "all_candidates": [
            {
                "candidate_id": "tiny__full_repair_to_c067_decoded_mask",
                "base": {"decoded_mask_array": str(base_npz)},
                "policy": {
                    "policy_id": "full_repair_to_c067_decoded_mask",
                    "selection": "all_base_target_differences",
                    "pair_indices": [],
                    "class_ids": [],
                },
                "cdo1_payload": {"raw_sha256": builder._sha256_bytes(raw)},
            }
        ],
    }
    economics_json = tmp_path / "economics.json"
    economics_json.write_text(__import__("json").dumps(economics))
    base_archive = tmp_path / "base.zip"
    _runtime_archive(base_archive)

    report = builder.build_reversed_base_candidate(
        economics_json=economics_json,
        candidate_id="tiny__full_repair_to_c067_decoded_mask",
        base_archive=base_archive,
        output_archive=tmp_path / "out" / "archive.zip",
        manifest_json=tmp_path / "out" / "manifest.json",
        overlay_compressor="lzma_xz",
        pack_output_payload=False,
        repo_root=REPO_ROOT,
    )

    assert report["score_claim"] is False
    assert report["promotion_eligible"] is False
    assert report["rebuilt_cdo1_payload"]["rebuilt_raw_sha256"] == builder._sha256_bytes(raw)
    summary = report["selected_policy_summary"]
    assert summary["candidate_id"] == "tiny__full_repair_to_c067_decoded_mask"
    assert summary["policy_id"] == "full_repair_to_c067_decoded_mask"
    assert summary["selected_pair_indices"] == [0]
    assert summary["rebuilt_selected_pixel_count"] == int(selected.sum())
    assert summary["expected_raw_sha256"] == builder._sha256_bytes(raw)
    members = {
        info.filename
        for info in zipfile.ZipFile(tmp_path / "out" / "archive.zip").infolist()
    }
    assert "masks.cdo1.xz" in members


def test_reversed_base_builder_preserves_nonzero_half_frame_pair_basis(
    tmp_path: Path,
) -> None:
    builder = _load_module("build_reversed_base_cdo1_candidate_half_frame_test", BUILDER_PATH)
    base = np.zeros((8, 384, 512), dtype=np.uint8)
    target = base.copy()
    target[3, 12:14, 3:5] = 1
    target[6, 20:22, 9:11] = 1
    target[7, 21:23, 13:15] = 1
    selected = np.zeros_like(target, dtype=bool)
    selected[3] = target[3] != base[3]
    repaired = base.copy()
    repaired[selected] = target[selected]
    runs = _runs_from_selected(base, target, selected)
    raw = _encode_overlay_payload(
        runs=runs,
        header={
            "schema": "c067_decoded_delta_overlay_payload_v1",
            "producer": "experiments/plan_c067_reversed_base_cdo1_overlay_economics.py",
            "score_claim": False,
            "base_mask_tensor_sha256": _mask_tensor_sha256(base),
            "target_mask_tensor_sha256": _mask_tensor_sha256(target),
            "reconstructed_mask_u8_sha256": _mask_tensor_sha256(repaired),
            "shape": [8, 384, 512],
            "pair_index_basis": "half_frame_pair_index",
            "run_struct": RUN_STRUCT_NAME,
            "run_count": len(runs),
            "selected_pixel_count": int(selected.sum()),
            "selected_pair_indices": [3],
            "policy_id": "budget4000",
        },
    )
    base_npz = tmp_path / "base.npz"
    target_npy = tmp_path / "target.npy"
    np.savez_compressed(base_npz, masks=base)
    np.save(target_npy, target)
    economics = {
        "schema": "c067_reversed_base_cdo1_overlay_economics_v1",
        "target_decoded_mask": {"path": str(target_npy)},
        "gates": {
            "pair_index_basis": "half_frame_pair_index",
            "max_residual_disagreement_fraction": 0.001,
        },
        "all_candidates": [
            {
                "candidate_id": "tiny__budget4000",
                "base": {"decoded_mask_array": str(base_npz)},
                "mask_disagreement": {"pair_index_basis": "half_frame_pair_index"},
                "policy": {
                    "policy_id": "budget4000",
                    "selection": "trust_plan_pair_target_class",
                    "pair_indices": [3],
                    "class_ids": [1],
                },
                "cdo1_payload": {"raw_sha256": builder._sha256_bytes(raw)},
            }
        ],
    }
    economics_json = tmp_path / "economics.json"
    economics_json.write_text(__import__("json").dumps(economics))
    base_archive = tmp_path / "base.zip"
    _runtime_archive(base_archive)

    report = builder.build_reversed_base_candidate(
        economics_json=economics_json,
        candidate_id="tiny__budget4000",
        base_archive=base_archive,
        output_archive=tmp_path / "out" / "archive.zip",
        manifest_json=tmp_path / "out" / "manifest.json",
        overlay_compressor="lzma_xz",
        pack_output_payload=False,
        repo_root=REPO_ROOT,
    )

    summary = report["selected_policy_summary"]
    assert summary["pair_index_basis"] == "half_frame_pair_index"
    assert summary["base_vs_target_pair_indices"] == [3, 6, 7]
    assert summary["selected_pair_indices"] == [3]
    assert summary["residual_vs_target_pair_indices_after_overlay"] == [6, 7]
    rebuilt = report["rebuilt_cdo1_payload"]
    assert rebuilt["payload_header"]["pair_index_basis"] == "half_frame_pair_index"
    assert rebuilt["payload_header"]["selected_pair_indices"] == [3]
    assert rebuilt["rebuilt_selected_pixel_count"] == int(selected.sum())
    assert report["archive_report"]["cdo1_overlay"]["pair_index_basis"] == "half_frame_pair_index"


def test_reversed_base_builder_rejects_candidate_pair_basis_mismatch(tmp_path: Path) -> None:
    builder = _load_module("build_reversed_base_cdo1_candidate_basis_mismatch_test", BUILDER_PATH)
    base = np.zeros((4, 384, 512), dtype=np.uint8)
    target = base.copy()
    target[1, 12, 3:7] = 1
    base_npz = tmp_path / "base.npz"
    target_npy = tmp_path / "target.npy"
    np.savez_compressed(base_npz, masks=base)
    np.save(target_npy, target)
    economics = {
        "schema": "c067_reversed_base_cdo1_overlay_economics_v1",
        "target_decoded_mask": {"path": str(target_npy)},
        "gates": {
            "pair_index_basis": "half_frame_pair_index",
            "max_residual_disagreement_fraction": 0.001,
        },
        "all_candidates": [
            {
                "candidate_id": "tiny__budget4000",
                "base": {"decoded_mask_array": str(base_npz)},
                "mask_disagreement": {"pair_index_basis": "video_frame_pair_index"},
                "policy": {
                    "policy_id": "budget4000",
                    "selection": "trust_plan_pair_target_class",
                    "pair_indices": [1],
                    "class_ids": [1],
                },
                "cdo1_payload": {"raw_sha256": ""},
            }
        ],
    }
    economics_json = tmp_path / "economics.json"
    economics_json.write_text(__import__("json").dumps(economics))
    base_archive = tmp_path / "base.zip"
    _runtime_archive(base_archive)

    with pytest.raises(builder.ReversedBaseBuildError, match="pair_index_basis"):
        builder.build_reversed_base_candidate(
            economics_json=economics_json,
            candidate_id="tiny__budget4000",
            base_archive=base_archive,
            output_archive=tmp_path / "out" / "archive.zip",
            manifest_json=tmp_path / "out" / "manifest.json",
            repo_root=REPO_ROOT,
        )
