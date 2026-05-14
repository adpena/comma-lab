# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "experiments" / "plan_predictive_mask_hotspot.py"


def _load_planner():
    spec = importlib.util.spec_from_file_location("plan_predictive_mask_hotspot_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _tiny_hotspot_masks() -> np.ndarray:
    masks = np.zeros((4, 4, 8), dtype=np.uint8)
    masks[:, :, :] = 0
    masks[1, :, :] = 2
    masks[1, 0, :] = 3
    masks[1, 2, :] = 3
    masks[1, 3, :] = 0
    masks[2, :, :] = 1
    masks[3, 1:3, 2:6] = 2
    return masks


def test_pmg_hotspot_plan_emits_non_score_confusion_guard(tmp_path: Path) -> None:
    planner = _load_planner()
    mask_path = tmp_path / "decoded_mask_array.npy"
    np.save(mask_path, _tiny_hotspot_masks(), allow_pickle=False)

    manifest = planner.build_plan(
        decoded_mask_array=mask_path,
        output_dir=tmp_path / "plan",
        hotspot_pairs=(1,),
        protected_confusions=((2, 3), (0, 3)),
        row_strides=(2,),
        compressors=("zlib9",),
        baseline_mask_stream_bytes=8000,
        frontier_archive_bytes=10000,
        target_savings_bytes=100,
        archive_wrapper_overhead_bytes=0,
        max_selected_atoms=128,
        command=["unit-test"],
    )

    assert manifest["schema"] == planner.SCHEMA
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["gpu_required"] is False
    assert manifest["cuda_jobs_launched"] is False
    assert manifest["cloud_jobs_dispatched"] is False
    assert manifest["scorer_network_loaded"] is False
    assert "contest_auth_eval.py --device cuda" in manifest["canonical_score_source_required"]
    assert manifest["hotspot_contract"]["hotspot_pairs"] == [1]
    assert manifest["hotspot_contract"]["hotspot_frame_records"][0]["frame_indices"] == [1]

    candidate = manifest["best_candidate"]
    trust = candidate["trust_region_metrics"]
    assert candidate["score_claim"] is False
    assert candidate["dispatch_relevance"]["dispatchable_now"] is False
    assert trust["protected_confusion_hotspot_pixels_before_residual"] > 0
    assert trust["protected_confusion_hotspot_pixels_after_selected_residual"] == 0
    assert trust["protected_confusion_complete"] is True
    assert candidate["protected_atoms"]["selected_atom_count"] > 0
    assert any(atom["reason"] == "protected_confusion" for atom in candidate["protected_atoms"]["selected_atoms"])
    assert candidate["byte_estimates"]["archive_bytes_if_replaces_mask_estimate"] < 10000
    assert manifest["recommendation"]["confusion_guard_candidate_exists"] is True

    manifest_path = tmp_path / "plan" / planner.REPORT_NAME
    assert json.loads(manifest_path.read_text()) == manifest


def test_pmg_hotspot_truncated_atoms_are_trust_no_go(tmp_path: Path) -> None:
    planner = _load_planner()
    mask_path = tmp_path / "decoded_mask_array.npy"
    np.save(mask_path, _tiny_hotspot_masks(), allow_pickle=False)

    manifest = planner.build_plan(
        decoded_mask_array=mask_path,
        output_dir=tmp_path / "plan",
        hotspot_pairs=(1,),
        protected_confusions=((2, 3),),
        row_strides=(2,),
        compressors=("zlib9",),
        baseline_mask_stream_bytes=8000,
        frontier_archive_bytes=10000,
        target_savings_bytes=100,
        archive_wrapper_overhead_bytes=0,
        max_selected_atoms=0,
    )

    candidate = manifest["best_candidate"]
    assert candidate["protected_atoms"]["selected_atom_count"] == 0
    assert candidate["protected_atoms"]["selected_atoms_truncated"] is True
    assert candidate["trust_region_metrics"]["protected_confusion_complete"] is False
    assert candidate["dispatch_relevance"]["relevance_class"] == "trust_no_go_unprotected_confusion"
    assert manifest["recommendation"]["confusion_guard_candidate_exists"] is False


def test_pmg_hotspot_candidate_table_is_deterministic(tmp_path: Path) -> None:
    planner = _load_planner()
    mask_path = tmp_path / "decoded_mask_array.npy"
    np.save(mask_path, _tiny_hotspot_masks(), allow_pickle=False)
    kwargs = {
        "decoded_mask_array": mask_path,
        "hotspot_pairs": (1, 3),
        "protected_confusions": ((2, 3), (0, 3)),
        "row_strides": (1, 2),
        "compressors": ("zlib9",),
        "baseline_mask_stream_bytes": 8000,
        "frontier_archive_bytes": 10000,
        "target_savings_bytes": 100,
        "archive_wrapper_overhead_bytes": 0,
        "max_selected_atoms": 128,
    }

    first = planner.build_plan(output_dir=tmp_path / "plan-a", **kwargs)
    second = planner.build_plan(output_dir=tmp_path / "plan-b", **kwargs)

    assert first["candidate_table"] == second["candidate_table"]
    assert first["best_candidate"] == second["best_candidate"]
    assert first["recommendation"] == second["recommendation"]
