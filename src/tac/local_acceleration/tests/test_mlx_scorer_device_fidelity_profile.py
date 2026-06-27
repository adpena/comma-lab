# SPDX-License-Identifier: MIT
"""Contract tests for the canonical MLX scorer device fidelity×speed profile.

The profile is a pure-literal system-intelligence surface (no MLX/torch needed): it codifies
the measured 2026-06-27 benchmark + proven drift diagnosis + optimal device policy so the
verdict path / level-set sibling / autopilot consume the policy instead of re-deriving it.
"""

from __future__ import annotations

from tac.local_acceleration.mlx_scorer_torch_parity import (
    DEVICE_FIDELITY_PROFILE_SCHEMA_VERSION,
    mlx_scorer_device_fidelity_speed_profile,
)


def test_schema_and_false_authority_flags():
    p = mlx_scorer_device_fidelity_speed_profile()
    assert p["schema_version"] == DEVICE_FIDELITY_PROFILE_SCHEMA_VERSION
    # MLX rows are NEVER a contest-axis score authority.
    assert p["score_claim"] is False
    assert p["promotion_eligible"] is False
    assert p["rank_or_kill_eligible"] is False
    assert p["ready_for_exact_eval_dispatch"] is False


def test_gpu_is_fastest_scorer_but_drifts():
    sp = mlx_scorer_device_fidelity_speed_profile()["scorer_port"]
    # MLX-GPU is the fastest forward but reduced-order (drifts vs the torch-CPU oracle).
    assert sp["mlx_gpu"]["fwd_pairs_per_s"] > sp["torch_cpu"]["fwd_pairs_per_s"]
    assert sp["mlx_gpu"]["speedup_vs_torch_cpu"] > 1.0
    assert sp["mlx_gpu"]["seg_argmax_px_per_pair_max"] > 0.0


def test_mlx_cpu_is_exact_but_slower_than_torch_cpu():
    # The measured wall-clock finding: MLX-CPU is fp32-exact (0 px) yet SLOWER than torch-CPU,
    # so it offers NO speed win for the verdict — torch-CPU stays the verdict scorer.
    sp = mlx_scorer_device_fidelity_speed_profile()["scorer_port"]
    assert sp["mlx_cpu"]["seg_argmax_px_per_pair"] == 0.0
    assert sp["mlx_cpu"]["fwd_pairs_per_s"] < sp["torch_cpu"]["fwd_pairs_per_s"]
    assert sp["mlx_cpu"]["speedup_vs_torch_cpu"] < 1.0


def test_gpu_not_bit_exact_with_named_blockers():
    p = mlx_scorer_device_fidelity_speed_profile()
    assert p["gpu_bit_exact_achievable"] is False
    blockers = p["gpu_bit_exact_blockers"]
    assert any("deterministic_reduction" in b for b in blockers)
    assert any("fp64" in b for b in blockers)


def test_diagnosis_is_reduction_order_not_precision_or_fastmath():
    d = mlx_scorer_device_fidelity_speed_profile()["diagnosis"]
    assert d["root_cause"] == "fp32_reduction_order_non_associativity"
    assert d["not_reduced_precision_accum"] is True
    assert d["not_fast_math_transcendentals"] is True


def test_optimal_policy_device_split():
    pol = mlx_scorer_device_fidelity_speed_profile()["optimal_policy"]
    assert pol["training_gradient"] == "mlx_gpu"
    assert pol["verdict_scorer_device"] == "torch_cpu"
    assert "numpy_fp32" in pol["scored_verdict"]
    # The 4th measurement artifact: mlx-gpu render is forbidden for any scored verdict.
    assert pol["forbidden_for_scored_verdict"] == "mlx_gpu_render"


def test_witness_render_artifact_ranges():
    wr = mlx_scorer_device_fidelity_speed_profile()["witness_render"]
    assert wr["numpy_fp32"]["argmax_px_per_pair"] == 0
    lo, hi = wr["mlx_gpu"]["argmax_px_per_pair_range"]
    # The catastrophic 4th-artifact range, far above the 143-px sub-0.15 d_seg budget.
    assert lo >= 143 and hi > lo
