# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
PLANNER_PATH = REPO_ROOT / "experiments" / "plan_cmg3_pixel_lagrangian_atoms.py"
CMG3A_BUILDER_PATH = REPO_ROOT / "experiments" / "build_cmg3_adaptive_runs_candidate.py"


def _load_planner():
    spec = importlib.util.spec_from_file_location("plan_cmg3_pixel_lagrangian_atoms_test", PLANNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_source_candidate(tmp_path: Path, source: np.ndarray, candidate: np.ndarray) -> tuple[Path, Path]:
    source_path = tmp_path / "source_masks.npy"
    candidate_path = tmp_path / "candidate_masks.npy"
    np.save(source_path, source.astype(np.uint8), allow_pickle=False)
    np.save(candidate_path, candidate.astype(np.uint8), allow_pickle=False)
    return source_path, candidate_path


def test_row_residual_run_segments_split_by_gap_and_source_class() -> None:
    planner = _load_planner()
    diff = np.array([False, True, True, True, False, True, True], dtype=bool)
    source = np.array([0, 1, 1, 2, 0, 2, 2], dtype=np.uint8)

    assert planner._row_residual_run_segments(  # noqa: SLF001 - regression for vectorized helper
        diff_row=diff,
        source_row=source,
    ) == [
        (1, 3, 1),
        (3, 4, 2),
        (5, 7, 2),
    ]


def test_residual_atoms_are_deterministic_and_use_cmg3_run_cost(tmp_path: Path) -> None:
    planner = _load_planner()
    source = np.zeros((4, 6, 8), dtype=np.uint8)
    candidate = np.zeros_like(source)
    source[0, 1, 1:4] = 1
    source[0, 2, 1:4] = 1
    source[2, 4, 4:7] = 2
    source_path, candidate_path = _write_source_candidate(tmp_path, source, candidate)

    kwargs = {
        "source_mask_array": source_path,
        "candidate_mask_array": candidate_path,
        "atom_families": ("row_run", "pair", "connected_row_run"),
        "max_atoms": 32,
        "boundary_bonus": 0.0,
        "long_run_bonus": 0.0,
    }
    first = planner.build_ledger(output_json=tmp_path / "ledger_a.json", **kwargs)
    second = planner.build_ledger(output_json=tmp_path / "ledger_b.json", **kwargs)

    assert first == second
    assert json.loads((tmp_path / "ledger_a.json").read_text()) == first
    assert first["tensor"]["residual_pixels"] == 9
    assert first["atom_family_counts"]["row_run"] == 3

    row_atom = next(atom for atom in first["top_atoms"] if atom["atom_family"] == "row_run")
    assert row_atom["cost_model"]["estimated_charged_bytes"] == 6
    assert row_atom["cost_model"]["formula"] == "estimated_charged_bytes = touched_row_count*1 + run_count*5"

    pair0 = next(
        atom
        for atom in first["top_atoms"]
        if atom["atom_family"] == "pair" and atom["identity"]["pair_index"] == 0
    )
    assert pair0["run_count"] == 2
    assert pair0["touched_row_count"] == 2
    assert pair0["cost_model"]["estimated_charged_bytes"] == 12


def test_foveal_and_component_pair_weights_change_ranking(tmp_path: Path) -> None:
    planner = _load_planner()
    source = np.zeros((4, 8, 10), dtype=np.uint8)
    candidate = np.zeros_like(source)
    source[0, 2, 1:5] = 1
    source[2, 6, 6:10] = 1
    source_path, candidate_path = _write_source_candidate(tmp_path, source, candidate)

    near_pair0 = planner.build_ledger(
        source_mask_array=source_path,
        candidate_mask_array=candidate_path,
        output_json=tmp_path / "near_pair0.json",
        atom_families=("row_run",),
        foveal_center=(2.5, 2.0),
        foveal_sigma=1.0,
        foveal_strength=6.0,
        boundary_bonus=0.0,
        long_run_bonus=0.0,
    )
    near_pair1 = planner.build_ledger(
        source_mask_array=source_path,
        candidate_mask_array=candidate_path,
        output_json=tmp_path / "near_pair1.json",
        atom_families=("row_run",),
        foveal_center=(7.5, 6.0),
        foveal_sigma=1.0,
        foveal_strength=6.0,
        boundary_bonus=0.0,
        long_run_bonus=0.0,
    )

    assert near_pair0["top_atoms"][0]["identity"]["frame_index"] == 0
    assert near_pair1["top_atoms"][0]["identity"]["frame_index"] == 2

    trace = {
        "score_claim": False,
        "evidence_grade": "diagnostic_component_trace",
        "samples": [
            {
                "pair_index": 0,
                "score_combined_contribution_first_order": 0.1,
            },
            {
                "pair_index": 1,
                "score_combined_contribution_first_order": 1.0,
            },
        ],
    }
    trace_path = tmp_path / "component_trace.json"
    trace_path.write_text(json.dumps(trace, sort_keys=True) + "\n")
    pair_weighted = planner.build_ledger(
        source_mask_array=source_path,
        candidate_mask_array=candidate_path,
        output_json=tmp_path / "pair_weighted.json",
        component_trace_json=trace_path,
        atom_families=("pair",),
        boundary_bonus=0.0,
        long_run_bonus=0.0,
    )

    assert pair_weighted["inputs"]["component_trace_json"]["top_pair_indices"][:2] == [1, 0]
    assert pair_weighted["top_atoms"][0]["identity"]["pair_index"] == 1
    assert pair_weighted["top_atoms"][0]["weights"]["component_pair_weight_pixel_mean"] > 1.0


def test_dynamic_per_frame_foveation_changes_ranking_and_records_hash(tmp_path: Path) -> None:
    planner = _load_planner()
    source = np.zeros((4, 8, 10), dtype=np.uint8)
    candidate = np.zeros_like(source)
    source[0, 2, 1:5] = 1
    source[2, 6, 6:10] = 1
    source_path, candidate_path = _write_source_candidate(tmp_path, source, candidate)
    foveation_path = tmp_path / "dynamic_foveation.json"
    foveation_path.write_text(
        json.dumps(
            {
                "frame_centers": [
                    [2.5, 2.0],
                    [2.5, 2.0],
                    [7.5, 6.0],
                    [7.5, 6.0],
                ],
                "sigma": 1.0,
                "strength": 6.0,
            },
            sort_keys=True,
        )
        + "\n"
    )

    ledger = planner.build_ledger(
        source_mask_array=source_path,
        candidate_mask_array=candidate_path,
        output_json=tmp_path / "dynamic.json",
        atom_families=("row_run",),
        foveation_json=foveation_path,
        boundary_bonus=0.0,
        long_run_bonus=0.0,
    )

    assert ledger["inputs"]["foveation"]["config"]["mode"] == "dynamic_per_frame"
    assert ledger["inputs"]["foveation"]["config"]["frame_center_count"] == 4
    assert "frame_centers_sha256" in ledger["inputs"]["foveation"]["config"]
    top_frames = {atom["identity"]["frame_index"] for atom in ledger["top_atoms"][:2]}
    assert top_frames == {0, 2}


def test_dynamic_foveation_frame_count_must_match_source(tmp_path: Path) -> None:
    planner = _load_planner()
    source = np.zeros((2, 4, 5), dtype=np.uint8)
    candidate = np.zeros_like(source)
    source[0, 1, 1:4] = 1
    source_path, candidate_path = _write_source_candidate(tmp_path, source, candidate)
    foveation_path = tmp_path / "bad_dynamic_foveation.json"
    foveation_path.write_text(
        json.dumps(
            {
                "frame_centers": [[2.0, 1.0]],
                "sigma": 1.0,
                "strength": 2.0,
            },
            sort_keys=True,
        )
        + "\n"
    )

    try:
        planner.build_ledger(
            source_mask_array=source_path,
            candidate_mask_array=candidate_path,
            output_json=tmp_path / "bad.json",
            atom_families=("row_run",),
            foveation_json=foveation_path,
        )
    except planner.PlannerError as exc:
        assert "dynamic foveation frame center count" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected dynamic foveation frame-count failure")


def test_dynamic_foveation_full_frame_centers_average_into_pair_masks(tmp_path: Path) -> None:
    planner = _load_planner()
    source = np.zeros((2, 4, 8), dtype=np.uint8)
    candidate = np.zeros_like(source)
    source[0, 1, 1:4] = 1
    source[1, 2, 5:8] = 1
    source_path, candidate_path = _write_source_candidate(tmp_path, source, candidate)
    foveation_path = tmp_path / "full_frame_dynamic_foveation.json"
    foveation_path.write_text(
        json.dumps(
            {
                "frame_centers": [
                    [2.0, 1.0],
                    [2.0, 1.0],
                    [6.0, 2.0],
                    [6.0, 2.0],
                ],
                "sigma": 0.75,
                "strength": 5.0,
            },
            sort_keys=True,
        )
        + "\n"
    )

    ledger = planner.build_ledger(
        source_mask_array=source_path,
        candidate_mask_array=candidate_path,
        output_json=tmp_path / "full_to_pair.json",
        atom_families=("row_run",),
        foveation_json=foveation_path,
        boundary_bonus=0.0,
        long_run_bonus=0.0,
    )

    cfg = ledger["inputs"]["foveation"]["config"]
    assert cfg["frame_center_count"] == 4
    assert cfg["frame_center_indexing"] == "pair_average_from_full_frames"
    assert ledger["inputs"]["foveation"]["source_frame_count"] == 2
    top_frames = {atom["identity"]["frame_index"] for atom in ledger["top_atoms"][:2]}
    assert top_frames == {0, 1}


def test_ledger_records_no_score_claim_grade_and_break_even_math(tmp_path: Path) -> None:
    planner = _load_planner()
    source = np.zeros((2, 4, 5), dtype=np.uint8)
    candidate = np.zeros_like(source)
    source[1, 1, 1:4] = 3
    source_path, candidate_path = _write_source_candidate(tmp_path, source, candidate)

    ledger = planner.build_ledger(
        source_mask_array=source_path,
        candidate_mask_array=candidate_path,
        output_json=tmp_path / "ledger.json",
        atom_families=("row_run", "pair", "class"),
        max_atoms=16,
    )

    assert ledger["score_claim"] is False
    assert ledger["no_score_claim"] is True
    assert ledger["promotion_eligible"] is False
    assert ledger["evidence_grade"] == "planning_only"
    assert ledger["cuda_jobs_launched"] is False
    assert ledger["remote_jobs_dispatched"] is False
    assert math.isclose(ledger["formulas"]["lambda_rate"], 25.0 / 37_545_489)
    assert math.isclose(ledger["formulas"]["break_even_bytes_per_score"], 37_545_489 / 25.0)
    assert "1 / lambda_rate" in ledger["formulas"]["break_even_bytes_per_score_formula"]

    atom = ledger["top_atoms"][0]
    assert atom["score_claim"] is False
    assert atom["no_score_claim"] is True
    assert atom["evidence_grade"] == "planning_only"
    assert math.isclose(atom["lagrangian"]["break_even_bytes_per_score"], 37_545_489 / 25.0)
    assert "score_saved_needed_to_pay_rate" in atom["lagrangian"]


def test_cmg3a_adaptive_manifest_reconstructs_candidate_for_planning(tmp_path: Path) -> None:
    planner = _load_planner()
    builder = _load_module(CMG3A_BUILDER_PATH, "cmg3a_manifest_planner_test")
    source = np.zeros((1, 384, 512), dtype=np.uint8)
    source[0, 10, 0:120] = 1
    source[0, 10, 200:280] = 2
    source[0, 20, 300:305] = 4
    source_path = tmp_path / "source.npy"
    np.save(source_path, source, allow_pickle=False)

    _stream, recon, stats, policy = builder.encode_adaptive_run_stream(
        source,
        base_runs_per_row=1,
        target_extra_runs=1,
        adaptive_max_runs_per_row=3,
        compressor="raw",
        class_weights={1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
        foveal_row_weight=0.0,
        foveal_col_weight=0.0,
        boundary_detail_weight=0.0,
        rank_decay=1.0,
    )
    manifest = {
        "schema": "cmg3a_adaptive_nonzero_row_runs_candidate_v1",
        "score_claim": False,
        "policy": policy,
        "cmg3": {
            "mode": "nonzero_row_runs_topk_v1",
            "run_stats": stats,
        },
    }
    manifest_path = tmp_path / "cmg3a_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")

    ledger = planner.build_ledger(
        source_mask_array=source_path,
        candidate_manifest=manifest_path,
        output_json=tmp_path / "ledger.json",
        atom_families=("row_run",),
        max_atoms=16,
        boundary_bonus=0.0,
        long_run_bonus=0.0,
    )

    assert ledger["inputs"]["candidate"]["mode"] == "reconstructed_from_cmg3a_adaptive_manifest"
    assert ledger["inputs"]["candidate"]["selected_extra_runs"] == 1
    assert ledger["tensor"]["residual_pixels"] == int((source != recon).sum())
