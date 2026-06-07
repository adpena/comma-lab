# SPDX-License-Identifier: MIT
"""Tests for measured inverse-scorer ActionEffect materialization.

Fixtures here are synthetic validation inputs only.  They exercise the real
``ActionEffect`` score math, generator, commutator, and CLI paths, but carry no
empirical or contest authority.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tac.analysis.action_commutator import build_commutator_ledger
from tac.analysis.action_effect import ActionEffect, append_action_effect, read_action_effects
from tac.analysis.inverse_scorer_actions import (
    BLOCKER_NO_COMPOSITE,
    generate_inverse_scorer_candidates,
)
from tac.analysis.pr110_baseline_reproduction import (
    BLOCKER_GLOBAL_K,
    BLOCKER_SELECTOR_BITS,
    BLOCKER_SELECTOR_PAIR_COUNT,
    build_pr110_k16_baseline_reproduction_from_action_effects,
    validate_pr110_k16_baseline_reproduction,
)


def _frame0_pose_effect() -> ActionEffect:
    return ActionEffect.build(
        action_id="real_frame0_pose_source",
        family="hinerv",
        action_kind="frame0_pose_target_only",
        authority="batch_local_live_mlx",
        producer="fixture",
        consumer="fixture",
        pair_ids=[0],
        trained_groups=["compensation_head_rgb_0"],
        old_d_seg=0.50,
        new_d_seg=0.50,
        old_d_pose=0.20,
        new_d_pose=0.15,
        receiver_surface={
            "posenet_input_delta_linf": 1.0,
            "pose_output_l2_delta": 1.25,
        },
        posenet_input_delta_linf_pair=1.0,
        pose_output_l2_delta=1.25,
        exact_score_decision="accept",
    )


def _frame1_birth_effect() -> ActionEffect:
    return ActionEffect.build(
        action_id="real_frame1_birth_source",
        family="hinerv",
        action_kind="target_region_birth",
        authority="batch_local_live_mlx",
        producer="fixture",
        consumer="fixture",
        pair_ids=[0],
        class_ids=[4],
        region_ids=["b0/c4/r1"],
        trained_groups=["high_grid", "output_head_rgb_1"],
        old_d_seg=0.50,
        new_d_seg=0.48,
        old_d_pose=0.20,
        new_d_pose=0.20,
        receiver_surface={
            "uint8_changed_pixels": 32,
            "seg_input_delta_linf": 1.0,
            "seg_argmax_changed_pixels": 7,
            "seg_wrong_to_target_count": 5,
        },
        uint8_changed_count_region=32,
        seg_input_delta_linf_region=1.0,
        argmax_changed_count_region=7,
        wrong_to_target=5,
        exact_score_decision="accept",
    )


def _composite_effect() -> ActionEffect:
    return ActionEffect.build(
        action_id="real_birth_plus_pose_composite_source",
        family="hinerv",
        action_kind="independent_birth_plus_frame0_pose",
        authority="batch_local_live_mlx",
        producer="fixture",
        consumer="fixture",
        pair_ids=[0],
        class_ids=[4],
        region_ids=["b0/c4/r1"],
        trained_groups=["high_grid", "output_head_rgb_1", "compensation_head_rgb_0"],
        old_d_seg=0.50,
        new_d_seg=0.46,
        old_d_pose=0.20,
        new_d_pose=0.14,
        receiver_surface={
            "uint8_changed_pixels": 48,
            "seg_input_delta_linf": 1.0,
            "posenet_input_delta_linf": 1.0,
            "seg_argmax_changed_pixels": 11,
            "seg_wrong_to_target_count": 8,
            "pose_output_l2_delta": 1.5,
        },
        uint8_changed_count_region=48,
        seg_input_delta_linf_region=1.0,
        posenet_input_delta_linf_pair=1.0,
        argmax_changed_count_region=11,
        wrong_to_target=8,
        pose_output_l2_delta=1.5,
        interaction_or_commutator=-0.25,
        exact_score_decision="accept",
    )


def _pr110_k1_replay_effect() -> ActionEffect:
    return ActionEffect.build(
        action_id="lfv1v2_k01_replay",
        family="pr110",
        action_kind="selector_replay",
        authority="[macOS-CPU advisory] pr110_selector_replay",
        normalization_scope="full_video_equiv_estimate",
        producer="fixture",
        consumer="fixture",
        pair_ids=[43],
        payload_sections=["lfv1v2", "k01"],
        old_d_seg=0.00056039,
        new_d_seg=0.00056039,
        old_d_pose=2.943e-05,
        new_d_pose=2.943e-05,
        old_bytes=178_517,
        new_bytes=178_674,
        restore_state_pass=True,
        inflate_survived=True,
    )


def test_inverse_scorer_generator_reemits_measured_candidates_and_ordered_composite() -> None:
    result = generate_inverse_scorer_candidates(
        [_frame0_pose_effect(), _frame1_birth_effect(), _composite_effect()]
    )

    assert result["passed"] is True
    assert result["candidate_count"] == 3
    effects = result["action_effects"]
    assert {effect.candidate_status for effect in effects} == {"measured"}
    assert all(effect.promotion_eligible is False for effect in effects)

    singles = [effect for effect in effects if effect.frame_index != "both"]
    composites = [effect for effect in effects if effect.frame_index == "both"]
    assert len(singles) == 2
    assert len(composites) == 1
    composite_id = composites[0].action_id
    assert singles[0].action_id in composite_id or singles[1].action_id in composite_id

    ledger = build_commutator_ledger(singles, composites)
    assert ledger["measured_commutator_count"] == 1
    assert ledger["needs_measurement_count"] == 1
    assert ledger["rows"][0]["authority"] == "batch_local_live_mlx"

    queue = result["candidate_queue"]
    assert len(queue) == 3
    assert all(row["menu_ilp_allowed"] is False for row in queue)
    assert all("pr110_k16_baseline_reproduction_missing" in row["menu_ilp_blockers"] for row in queue)


def test_inverse_scorer_generator_names_missing_composite_without_inventing_row() -> None:
    result = generate_inverse_scorer_candidates([_frame0_pose_effect(), _frame1_birth_effect()])

    assert result["passed"] is False
    assert result["candidate_count"] == 2
    assert BLOCKER_NO_COMPOSITE in result["blockers"]


def test_pr110_k16_reproduction_from_sparse_k1_row_is_precise_blocker() -> None:
    proof = build_pr110_k16_baseline_reproduction_from_action_effects([_pr110_k1_replay_effect()])

    assert proof["passed"] is False
    assert proof["global_k"] == 1
    assert proof["pair_count"] == 1
    assert BLOCKER_GLOBAL_K in proof["blockers"]
    assert BLOCKER_SELECTOR_PAIR_COUNT in proof["blockers"]
    assert BLOCKER_SELECTOR_BITS in proof["blockers"]

    validation = validate_pr110_k16_baseline_reproduction(proof)
    assert validation["passed"] is False
    assert BLOCKER_SELECTOR_PAIR_COUNT in validation["blockers"]
    assert BLOCKER_SELECTOR_BITS in validation["blockers"]


def test_generate_inverse_evaluate_actions_cli_writes_artifacts(tmp_path: Path) -> None:
    seed_ledger = tmp_path / "seed_action_effects.jsonl"
    pr110_ledger = tmp_path / "pr110_action_effects.jsonl"
    out_dir = tmp_path / "out"
    for effect in (_frame0_pose_effect(), _frame1_birth_effect(), _composite_effect()):
        append_action_effect(effect, seed_ledger)
    append_action_effect(_pr110_k1_replay_effect(), pr110_ledger)

    repo_root = Path(__file__).resolve().parents[3]
    proc = subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools" / "generate_inverse_evaluate_actions.py"),
            "--seed-action-effects",
            str(seed_ledger),
            "--pr110-action-effects",
            str(pr110_ledger),
            "--output-dir",
            str(out_dir),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout)
    assert summary["schema"] == "tac.inverse_evaluate_action_materialization.v1"
    assert summary["inverse_candidate_count"] == 3
    assert summary["pr110_replay_row_count"] == 1
    assert summary["menu_ilp_allowed"] is False
    assert BLOCKER_GLOBAL_K in summary["pr110_k16_blockers"]

    rows = read_action_effects(out_dir / "action_effect_rows.jsonl")
    assert len(rows) == 4
    assert any(row.family == "pr110" for row in rows)
    assert (out_dir / "inverse_candidate_queue.jsonl").is_file()
    assert (out_dir / "commutator_summary.json").is_file()
    assert (out_dir / "next_blocker.md").is_file()

    commutator = json.loads((out_dir / "commutator_summary.json").read_text(encoding="utf-8"))
    assert commutator["measured_commutator_count"] == 1
    queued = commutator["measurement_queue"][0]
    assert queued["measurement_command_available"] is False
    assert queued["first_measurement_command"] is None
    assert "inverse_scorer_reverse_order_composite_producer_missing" in queued["measurement_command_blockers"]
    assert "inverse_scorer_composite_base_identity_producer_missing" in queued["measurement_command_blockers"]


def test_convert_real_pr110_k16_packet_to_action_effect_clears_reproduction_gate(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    packet_manifest = (
        repo_root
        / "experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex"
        / "packet_manifest.json"
    )
    archive_manifest = (
        repo_root
        / "experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex"
        / "archive_manifest.json"
    )
    ledger = tmp_path / "pr110_k16_action_effects.jsonl"

    proc = subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools" / "convert_pr110_k16_packet_to_action_effect.py"),
            "--packet-manifest",
            str(packet_manifest),
            "--archive-manifest",
            str(archive_manifest),
            "--output-jsonl",
            str(ledger),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0, proc.stderr
    conversion = json.loads(proc.stdout)
    assert conversion["pair_count"] == 600
    assert conversion["selector_bits"] == 1944
    assert conversion["validation"]["passed"] is True

    rows = read_action_effects(ledger)
    assert len(rows) == 1
    proof = build_pr110_k16_baseline_reproduction_from_action_effects(rows)
    assert proof["passed"] is True
    assert proof["blockers"] == []
    assert proof["global_k"] == 16
    assert proof["pair_count"] == 600
    assert proof["selector_bits"] == 1944
