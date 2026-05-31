# SPDX-License-Identifier: MIT
"""Tests for reusable Z8 freeze-vs-KKT allocator comparison primitives."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from tac.substrates.z8_hierarchical_predictive_coding.allocator_comparison import (
    FREEZE_ALLOCATOR_ARM,
    IMPLICIT_KKT_ALLOCATOR_ARM,
    allocator_deadzone_mask,
    apply_allocator_arm_to_archive,
    classify_allocator_comparison,
    match_allocator_rows_on_deadzone,
    segnet_detail_saliency_for_scored_frame,
)
from tac.substrates.z8_hierarchical_predictive_coding.archive import parse_z8hpc1_archive_bytes
from tac.substrates.z8_hierarchical_predictive_coding.canonical_quadruple_binding import (
    build_canonical_quadruple_binding_from_z8_config,
    build_z8hpc1_archive_bytes_from_canonical_quadruple,
)
from tac.substrates.z8_hierarchical_predictive_coding.joint_p18_p19_deadzone_rate_attack import (
    flatten_detail_coefficients,
    parse_pair_blobs_from_wavelet_blob,
)
from tac.substrates.z8_hierarchical_predictive_coding.mlx_renderer import Z8HierarchicalConfig


def _archive_bytes(*, num_pairs: int = 2) -> bytes:
    rng = np.random.RandomState(88)
    f0 = rng.uniform(0, 1, size=(num_pairs, 16, 16, 3)).astype(np.float32)
    f1 = rng.uniform(0, 1, size=(num_pairs, 16, 16, 3)).astype(np.float32)
    cfg = Z8HierarchicalConfig(
        num_levels=3,
        num_groups_per_level=(4, 3, 2),
        num_categories_per_level=(16, 8, 4),
        base_channels=8,
        decoder_latent_dim=12,
        num_pairs=num_pairs,
        deterministic_state_dim=16,
        gumbel_temperature=1.0,
        use_straight_through=True,
        eval_size=(16, 16),
    )
    binding = build_canonical_quadruple_binding_from_z8_config(cfg)
    return build_z8hpc1_archive_bytes_from_canonical_quadruple(binding, f0, f1)


def _per_pair_joint_for_archive(archive_bytes: bytes) -> list[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    sections = parse_z8hpc1_archive_bytes(archive_bytes)
    wavelet_start, wavelet_len = sections["wavelet_blob"]
    pyramids = parse_pair_blobs_from_wavelet_blob(archive_bytes[wavelet_start : wavelet_start + wavelet_len])
    rows: list[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]] = []
    for pyramid in pyramids:
        per_frame: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for frame_key in ("frame_0", "frame_1"):
            n_atoms = int(flatten_detail_coefficients(pyramid[f"{frame_key}_details"]).size)
            seg = np.linspace(0.0, 1.0, n_atoms, dtype=np.float64)
            pose = np.linspace(1.0, 0.0, n_atoms, dtype=np.float64)
            pose_null = pose <= 0.5
            per_frame[frame_key] = (seg, pose, pose_null)
        rows.append(per_frame)
    return rows


def test_allocator_masks_are_false_authority_and_share_priority_input() -> None:
    keep_priority = np.asarray([0.0, 1.0, 10.0, 100.0], dtype=np.float64)

    freeze_mask, freeze_report = allocator_deadzone_mask(
        keep_priority,
        arm=FREEZE_ALLOCATOR_ARM,
        knob=0.5,
    )
    kkt_mask, kkt_report = allocator_deadzone_mask(
        keep_priority,
        arm=IMPLICIT_KKT_ALLOCATOR_ARM,
        knob=0.25,
    )

    assert freeze_mask.tolist() == [True, True, False, False]
    assert kkt_mask.shape == keep_priority.shape
    assert kkt_mask[0] and not kkt_mask[-1]
    assert freeze_report["solver_blockers"] == []
    assert kkt_report["allocator"]["schema"] == "joint_p18_p19_implicit_kkt_dykstra_allocator.v1"


def test_match_and_classify_allocator_rows_are_schemaed_and_non_authority() -> None:
    freeze = [
        {"knob": 0.5, "deadzone_fraction": 0.5, "rate": 1.0, "d_seg": 0.1, "d_pose": 0.2, "contest_score": 10.0},
        {"knob": 0.25, "deadzone_fraction": 0.75, "rate": 0.8, "d_seg": 0.2, "d_pose": 0.3, "contest_score": 9.0},
    ]
    kkt = [
        {"knob": 0.01, "deadzone_fraction": 0.48, "rate": 0.9, "d_seg": 0.1, "d_pose": 0.2, "contest_score": 9.5},
        {"knob": 0.02, "deadzone_fraction": 0.76, "rate": 0.7, "d_seg": 0.2, "d_pose": 0.3, "contest_score": 8.7},
    ]

    matched = match_allocator_rows_on_deadzone(freeze, kkt)
    verdict = classify_allocator_comparison(matched, noise_band=0.05)

    assert [row["schema"] for row in matched] == [
        "z8_p18_p19_allocator_matched_operating_point.v2",
        "z8_p18_p19_allocator_matched_operating_point.v2",
    ]
    assert [row["match_metric"] for row in matched] == [
        "rate_then_deadzone_fraction",
        "rate_then_deadzone_fraction",
    ]
    assert matched[0]["implicit_kkt_minus_freeze_S"] == -0.5
    assert verdict["schema"] == "z8_p18_p19_allocator_comparison_verdict.v1"
    assert verdict["winner"] == "IMPLICIT_KKT"
    assert verdict["score_claim"] is False
    assert verdict["ready_for_exact_eval_dispatch"] is False


def test_classifier_refuses_kkt_winner_when_solver_blockers_are_present() -> None:
    matched = [
        {
            "schema": "z8_p18_p19_allocator_matched_operating_point.v2",
            "deadzone_fraction_gap": 0.0,
            "implicit_kkt_minus_freeze_S": -1.0,
            "implicit_kkt_solver_blockers": ["joint_p18_p19_allocator_dykstra_projection_mismatch"],
        }
    ]

    verdict = classify_allocator_comparison(matched, noise_band=0.01)

    assert verdict["winner"] is None
    assert verdict["blocker"] == "implicit_kkt_solver_blockers_present"
    assert verdict["score_claim"] is False


def test_segnet_detail_saliency_only_applies_to_scored_frame_1() -> None:
    detail = [
        {
            "lh": np.ones((2, 2, 1), dtype=np.float64),
            "hl": np.full((2, 2, 1), 2.0, dtype=np.float64),
            "hh": np.full((2, 2, 1), 3.0, dtype=np.float64),
        }
    ]

    frame0 = segnet_detail_saliency_for_scored_frame(detail, frame_key="frame_0")
    frame1 = segnet_detail_saliency_for_scored_frame(detail, frame_key="frame_1")

    assert all(float(np.max(arr)) == 0.0 for level in frame0 for arr in level.values())
    assert float(np.max(frame1[0]["hh"])) == 3.0


def test_apply_allocator_arm_to_archive_rewrites_valid_z8_archive() -> None:
    archive = _archive_bytes(num_pairs=2)
    per_pair_joint = _per_pair_joint_for_archive(archive)

    result = apply_allocator_arm_to_archive(
        archive,
        arm=FREEZE_ALLOCATOR_ARM,
        knob=0.5,
        per_pair_joint=per_pair_joint,
    )

    assert isinstance(result["archive_bytes"], bytes)
    assert result["coefficients_total"] > 0
    assert result["coefficients_zeroed"] > 0
    assert 0.0 < result["deadzone_fraction"] < 1.0
    assert result["score_claim"] is False
    assert parse_z8hpc1_archive_bytes(result["archive_bytes"])["wavelet_blob"][1] > 0
    assert result["archive_bytes"] != archive

    original_sections = parse_z8hpc1_archive_bytes(archive)
    mutated_sections = parse_z8hpc1_archive_bytes(result["archive_bytes"])
    original_start, original_len = original_sections["wavelet_blob"]
    mutated_start, mutated_len = mutated_sections["wavelet_blob"]
    original_pyramids = parse_pair_blobs_from_wavelet_blob(archive[original_start : original_start + original_len])
    mutated_pyramids = parse_pair_blobs_from_wavelet_blob(
        result["archive_bytes"][mutated_start : mutated_start + mutated_len]
    )
    original_coeffs = flatten_detail_coefficients(original_pyramids[0]["frame_0_details"])
    mutated_coeffs = flatten_detail_coefficients(mutated_pyramids[0]["frame_0_details"])
    newly_zeroed = (original_coeffs != 0.0) & (mutated_coeffs == 0.0)
    assert bool(np.any(newly_zeroed))


def test_freeze_vs_kkt_cli_has_no_tool_to_tool_imports() -> None:
    script = Path(__file__).resolve().parents[5] / "tools" / "z8_p18_p19_freeze_vs_implicit_kkt_comparison.py"
    tree = ast.parse(script.read_text(encoding="utf-8"))

    imported_modules = [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    ]

    assert not [module for module in imported_modules if module == "tools" or module.startswith("tools.")]
