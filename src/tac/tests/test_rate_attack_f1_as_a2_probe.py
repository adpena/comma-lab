# SPDX-License-Identifier: MIT
"""Tests for corrected F1-as-A2 RGB invariance probes."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from tac.contest_exploits import f1_as_a2_rgb_invariance as f1
from tac.contest_exploits import hydra_dim_invariance as direct_dim


def test_lsb_perturbation_changes_physical_rgb_values_deterministically():
    rgb = np.full((1, 2, 8, 8, 3), 128, dtype=np.uint8)

    perturbed_a, meta_a = f1.apply_deterministic_lsb_perturbation(
        rgb,
        pixel_stride=4,
        amplitude=1,
        seed=7,
    )
    perturbed_b, meta_b = f1.apply_deterministic_lsb_perturbation(
        rgb,
        pixel_stride=4,
        amplitude=1,
        seed=7,
    )

    assert np.array_equal(perturbed_a, perturbed_b)
    assert meta_a == meta_b
    assert meta_a["physical_rgb_changed"] is True
    assert meta_a["changed_rgb_values_total"] > 0
    assert not np.array_equal(rgb, perturbed_a)


def test_corrected_a2_capacity_report_proceeds_only_as_local_non_score_authority():
    baseline_pose = np.zeros((2, 12), dtype=np.float64)
    perturbed_pose = baseline_pose.copy()
    perturbed_pose[:, 6:12] = 0.5
    seg = np.zeros((2, 4, 4), dtype=np.int64)
    perturbation = {
        "changed_rgb_values_total": 32,
        "attempted_payload_bits_total": 32,
        "physical_rgb_changed": True,
        "pixel_stride": 4,
        "amplitude": 1,
        "seed": 0,
    }

    report = f1.build_f1_as_a2_report_from_outputs(
        baseline_pose=baseline_pose,
        perturbed_pose=perturbed_pose,
        baseline_seg_argmax=seg,
        perturbed_seg_argmax=seg.copy(),
        perturbation=perturbation,
        pose_0_5_rmse_threshold=1e-9,
        seg_delta_threshold=0.0,
    )

    assert report["f1_framing_version"] == "corrected_A2"
    assert report["channel_realization_surface"] == "in_memory_rgb_tensor_probe"
    assert (
        report["candidate_channel_realization_surface"]
        == "rgb_archive_output_pending_archive_proof"
    )
    assert report["direct_hydra_dim_channel_verdict"] == "DEFER"
    assert report["verdict"] == "PROCEED"
    assert report["blocker_status"] == "advisory"
    assert report["metrics"]["changed_rgb_values_total"] == 32
    assert report["metrics"]["changed_rgb_values_per_pair"] == 16.0
    assert "accepted_bits_per_pair" not in report["metrics"]
    assert report["metrics"]["recovered_payload_bits_total"] == 0
    assert report["authority"]["score_claim"] is False
    assert report["authority"]["ready_for_exact_eval_dispatch"] is False
    assert report["authority"]["payload_recovery_authority"] is False
    assert report["measurement_axis"].endswith("advisory]")
    assert report["probe_outcome_kwargs"]["substrate"] == (
        "rate_attack_f1_as_a2_posenet_rgb_invariance"
    )
    assert report["probe_outcome_kwargs"]["metric_name"] == "changed_rgb_values_per_pair"
    assert report["direct_hydra_dim_probe_outcome_kwargs"]["substrate"] == (
        "rate_attack_f1_hydra_dims_7_12"
    )
    assert report["direct_hydra_dim_probe_outcome_kwargs"]["verdict"] == "DEFER"
    assert report["direct_hydra_dim_probe_outcome_kwargs"]["blocker_status"] == "blocking"


def test_corrected_a2_capacity_report_blocks_when_pose_first_six_moves():
    baseline_pose = np.zeros((1, 12), dtype=np.float64)
    perturbed_pose = baseline_pose.copy()
    perturbed_pose[:, 0] = 0.01
    seg = np.zeros((1, 4, 4), dtype=np.int64)

    report = f1.build_f1_as_a2_report_from_outputs(
        baseline_pose=baseline_pose,
        perturbed_pose=perturbed_pose,
        baseline_seg_argmax=seg,
        perturbed_seg_argmax=seg.copy(),
        perturbation={"changed_rgb_values_total": 8, "physical_rgb_changed": True},
        pose_0_5_rmse_threshold=1e-4,
        seg_delta_threshold=0.0,
    )

    assert report["verdict"] == "DEFER"
    assert report["blocker_status"] == "blocking"
    assert "pose_0_5_delta_exceeds_threshold" in report["blockers"]
    assert report["metrics"]["recovered_payload_bits_total"] == 0


def test_direct_hydra_dim_report_is_internal_only_and_blocking():
    pose = np.zeros((2, 12), dtype=np.float64)

    report = direct_dim.build_hydra_dim_invariance_report_from_outputs(pose)

    assert report["authority"]["internal_posenet_score_invariance"] is True
    assert report["authority"]["archive_transport_channel_authority"] is False
    assert report["verdict"] == "DEFER"
    assert report["blocker_status"] == "blocking"
    assert (
        report["cargo_cult_audit"]["dims_7_12_are_free_archive_bytes"]
        == "CARGO_CULTED_FALSE_AS_STATED"
    )


def test_probe_tools_synthetic_emit_corrected_a2_framing():
    repo = Path(__file__).resolve().parents[3]
    for rel in (
        "tools/probe_f1_as_a2_posenet_rgb_invariance.py",
        "tools/probe_hydra_dim_7_12_score_invariance.py",
    ):
        result = subprocess.run(
            [sys.executable, str(repo / rel), "--synthetic", "--pair-count", "1"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        payload = json.loads(result.stdout)
        assert payload["f1_framing_version"] == "corrected_A2"
        assert payload["channel_realization_surface"] == "in_memory_rgb_tensor_probe"
        assert (
            payload["candidate_channel_realization_surface"]
            == "rgb_archive_output_pending_archive_proof"
        )
        assert payload["authority"]["score_claim"] is False
        assert payload["authority"]["payload_recovery_authority"] is False
        assert "accepted_bits_per_pair" not in payload["metrics"]
        assert payload["measurement_axis"].endswith("advisory]")
        assert payload["hardware_substrate"]
        assert payload["probe_command"]


def test_corrected_a2_report_uses_changed_values_not_payload_bits():
    baseline_pose = np.zeros((1, 12), dtype=np.float64)
    seg = np.zeros((1, 4, 4), dtype=np.int64)

    report = f1.build_f1_as_a2_report_from_outputs(
        baseline_pose=baseline_pose,
        perturbed_pose=baseline_pose.copy(),
        baseline_seg_argmax=seg,
        perturbed_seg_argmax=seg.copy(),
        perturbation={"changed_rgb_values_total": 5, "physical_rgb_changed": True},
    )

    assert report["metrics"]["changed_rgb_values_per_pair"] == 5.0
    assert report["metrics"]["recovered_payload_bits_per_pair"] == 0.0
    assert report["probe_outcome_kwargs"]["threshold_token"] == (
        "RGB_CHANGED_VALUES_PER_PAIR_GT_0_NOT_PAYLOAD_BITS"
    )
