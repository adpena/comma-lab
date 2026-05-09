"""Tests for tac.unified_action — GR-style unified action skeleton.

Per ``feedback_unified_lagrangian_action_principle_GR_style_20260509`` the
council's eureka was that the contest objective decomposes as a single scalar
action. This test module covers:

  - Each track contributes correctly to S_total
  - gradient() is autograd-consistent (sum of per-track grads = grad of S_total)
  - step() drives theta toward minimum (smoke convergence)
  - dual_update callback is invoked with grad-norm dict
  - Inactive tracks (lambda=0) drop from S_total + active_tracks
  - migration_status() emits valid JSON-serializable dict
  - Edge cases: zero theta, no active tracks, mismatched device, requires_grad=False
"""
from __future__ import annotations

import json

import pytest
import torch

from tac.unified_action import (
    ACTION_EVIDENCE_GRADE,
    ACTION_SCHEMA_VERSION,
    Action,
    DualVariables,
    TrackKind,
    make_action_from_track_callables,
)


# ── Per-track callable factories (deterministic, autograd-friendly) ─────────


def _quadratic(theta: torch.Tensor) -> torch.Tensor:
    """L = 0.5 * sum(theta**2). Gradient = theta."""
    return 0.5 * torch.sum(theta**2)


def _linear(theta: torch.Tensor) -> torch.Tensor:
    """L = sum(theta). Gradient = ones_like(theta)."""
    return torch.sum(theta)


def _negative_quadratic(theta: torch.Tensor) -> torch.Tensor:
    """L = -0.5 * sum(theta**2). Gradient = -theta. Used to test cancellation."""
    return -0.5 * torch.sum(theta**2)


# ── Basic construction + S_total ────────────────────────────────────────────


def test_action_with_no_tracks_returns_zero_tensor():
    action = Action()
    theta = torch.zeros(3)
    S = action.S_total(theta)
    assert S.shape == ()
    assert S.item() == 0.0


def test_action_with_seg_only_returns_lambda_seg_times_seg():
    action = make_action_from_track_callables(seg=_quadratic, duals=DualVariables(lambda_seg=2.0))
    theta = torch.tensor([1.0, 2.0, 3.0])
    expected = 2.0 * 0.5 * (1.0 + 4.0 + 9.0)
    assert action.S_total(theta).item() == pytest.approx(expected)


def test_action_with_three_baselines_sums_correctly():
    action = make_action_from_track_callables(
        seg=_quadratic,
        pose=_linear,
        rate=lambda t: torch.tensor(5.0),
        duals=DualVariables(lambda_seg=1.0, lambda_pose=2.0, lambda_rate=3.0),
    )
    theta = torch.tensor([1.0, 1.0])
    # seg = 1.0 * 0.5 * (1+1) = 1.0
    # pose = 2.0 * 2.0 = 4.0
    # rate = 3.0 * 5.0 = 15.0
    assert action.S_total(theta).item() == pytest.approx(1.0 + 4.0 + 15.0)


def test_action_with_t7_t8_t11_active_when_lambda_nonzero():
    action = make_action_from_track_callables(
        seg=_quadratic,
        t7_fisher_rao=_quadratic,
        t8_sinkhorn_w2=_quadratic,
        t11_lovasz_hinge=_quadratic,
        duals=DualVariables(lambda_seg=1.0, lambda_t7=0.5, lambda_t8=0.25, lambda_t11=0.0),
    )
    theta = torch.tensor([1.0, 1.0])
    # base = 0.5 * 2 = 1.0 each
    # seg=1.0, t7=0.5, t8=0.25, t11=0.0 (deactivated)
    assert action.S_total(theta).item() == pytest.approx(1.0 + 0.5 + 0.25 + 0.0)


def test_active_tracks_excludes_zero_lambda_refinement_tracks():
    action = make_action_from_track_callables(
        seg=_quadratic,
        pose=_linear,
        t7_fisher_rao=_quadratic,
        t11_lovasz_hinge=_quadratic,
        duals=DualVariables(lambda_t7=0.5, lambda_t11=0.0),
    )
    actives = action.active_tracks()
    assert TrackKind.SEG_BASELINE in actives
    assert TrackKind.POSE_BASELINE in actives
    assert TrackKind.T7_FISHER_RAO in actives
    assert TrackKind.T11_LOVASZ_HINGE not in actives


def test_active_tracks_baselines_always_active_even_at_lambda_zero():
    """Baselines (seg/pose/rate) are the contest scorer; setting lambda=0 still
    keeps them active because they ARE the scorer. The dual is a SCALING factor."""
    action = make_action_from_track_callables(
        seg=_quadratic,
        duals=DualVariables(lambda_seg=0.0),
    )
    actives = action.active_tracks()
    # Baseline track is still listed as active (it's mathematically there at lambda=0
    # because lambda_seg modulates magnitude not active/inactive).
    assert TrackKind.SEG_BASELINE in actives


# ── Gradient + autograd consistency ─────────────────────────────────────────


def test_gradient_requires_grad_raises_when_not_set():
    action = make_action_from_track_callables(seg=_quadratic)
    theta = torch.tensor([1.0, 2.0])  # requires_grad=False
    with pytest.raises(ValueError, match="requires_grad"):
        action.gradient(theta)


def test_gradient_per_track_keys_match_active_tracks():
    action = make_action_from_track_callables(
        seg=_quadratic, pose=_linear,
        t7_fisher_rao=_quadratic,
        duals=DualVariables(lambda_t7=1.0),
    )
    theta = torch.tensor([1.0, 2.0], requires_grad=True)
    grads = action.gradient(theta)
    assert set(grads.keys()) == {TrackKind.SEG_BASELINE, TrackKind.POSE_BASELINE, TrackKind.T7_FISHER_RAO}


def test_gradient_sum_equals_S_total_grad():
    """Sum of per-track gradients == grad of S_total. Autograd consistency."""
    action = make_action_from_track_callables(
        seg=_quadratic, pose=_linear,
        t7_fisher_rao=_quadratic,
        duals=DualVariables(lambda_seg=1.0, lambda_pose=2.0, lambda_t7=0.5),
    )
    theta = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    per_track_grads = action.gradient(theta)
    total_per_track = sum(per_track_grads.values(), start=torch.zeros_like(theta))

    # Autograd grad of S_total directly.
    theta2 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    action2 = make_action_from_track_callables(
        seg=_quadratic, pose=_linear,
        t7_fisher_rao=_quadratic,
        duals=DualVariables(lambda_seg=1.0, lambda_pose=2.0, lambda_t7=0.5),
    )
    S2 = action2.S_total(theta2)
    direct_grad = torch.autograd.grad(S2, theta2)[0]

    assert torch.allclose(total_per_track, direct_grad, atol=1e-6)


def test_gradient_seg_only_matches_quadratic_grad():
    action = make_action_from_track_callables(seg=_quadratic, duals=DualVariables(lambda_seg=3.0))
    theta = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    grads = action.gradient(theta)
    # grad of (3.0 * 0.5 * sum(theta**2)) = 3.0 * theta
    expected = 3.0 * torch.tensor([1.0, 2.0, 3.0])
    assert torch.allclose(grads[TrackKind.SEG_BASELINE], expected, atol=1e-6)


def test_gradient_cancellation_seg_plus_negative_t7():
    """seg + (-t7) at equal lambdas should give zero gradient."""
    action = make_action_from_track_callables(
        seg=_quadratic, t7_fisher_rao=_negative_quadratic,
        duals=DualVariables(lambda_seg=1.0, lambda_t7=1.0),
    )
    theta = torch.tensor([1.0, 2.0], requires_grad=True)
    grads = action.gradient(theta)
    total = grads[TrackKind.SEG_BASELINE] + grads[TrackKind.T7_FISHER_RAO]
    assert torch.allclose(total, torch.zeros_like(theta), atol=1e-6)


def test_gradient_independent_of_theta_returns_zero_for_unused_track():
    """Track that doesn't depend on theta returns zero grad (allow_unused=True)."""
    action = make_action_from_track_callables(
        seg=_quadratic,
        pose=lambda t: torch.tensor(5.0),  # constant — doesn't depend on theta
    )
    theta = torch.tensor([1.0, 2.0], requires_grad=True)
    grads = action.gradient(theta)
    assert torch.allclose(grads[TrackKind.POSE_BASELINE], torch.zeros_like(theta))


# ── Step + variational descent ──────────────────────────────────────────────


def test_step_requires_grad_raises():
    action = make_action_from_track_callables(seg=_quadratic)
    theta = torch.tensor([1.0, 2.0])
    with pytest.raises(ValueError, match="requires_grad"):
        action.step(theta, lr=0.1)


def test_step_lr_must_be_positive():
    action = make_action_from_track_callables(seg=_quadratic)
    theta = torch.tensor([1.0, 2.0], requires_grad=True)
    with pytest.raises(ValueError, match="lr must be > 0"):
        action.step(theta, lr=0.0)
    with pytest.raises(ValueError, match="lr must be > 0"):
        action.step(theta, lr=-0.5)


def test_step_descends_quadratic_toward_zero():
    """Repeated steps drive theta toward zero on a pure quadratic action."""
    action = make_action_from_track_callables(seg=_quadratic, duals=DualVariables(lambda_seg=1.0))
    theta = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
    initial_norm = theta.norm().item()
    for _ in range(20):
        theta, _ = action.step(theta, lr=0.1)
    final_norm = theta.norm().item()
    assert final_norm < initial_norm * 0.2  # >80% reduction in 20 steps


def test_step_dual_update_called_with_grad_norms():
    received: dict[TrackKind, float] = {}

    def my_dual_update(duals: DualVariables, grad_norms: dict[TrackKind, torch.Tensor]) -> DualVariables:
        received.update(grad_norms)
        return duals

    action = make_action_from_track_callables(
        seg=_quadratic, pose=_linear,
        duals=DualVariables(lambda_seg=1.0, lambda_pose=1.0),
    )
    theta = torch.tensor([1.0, 2.0], requires_grad=True)
    _, _ = action.step(theta, lr=0.1, dual_update=my_dual_update)

    assert TrackKind.SEG_BASELINE in received
    assert TrackKind.POSE_BASELINE in received
    # grad_norms are floats (we extract .item())
    assert all(isinstance(v, float) for v in received.values())


def test_step_dual_update_must_return_DualVariables():
    def bad_update(duals, grad_norms):
        return "not a DualVariables"

    action = make_action_from_track_callables(seg=_quadratic, duals=DualVariables(lambda_seg=1.0))
    theta = torch.tensor([1.0], requires_grad=True)
    with pytest.raises(TypeError, match="DualVariables"):
        action.step(theta, lr=0.1, dual_update=bad_update)


def test_step_with_dual_update_returns_new_duals():
    def adaptive_dual(duals, grad_norms):
        # Boyd-style: increase lambda when its grad is dominant.
        return DualVariables(lambda_seg=duals.lambda_seg * 1.5)

    action = make_action_from_track_callables(seg=_quadratic, duals=DualVariables(lambda_seg=2.0))
    theta = torch.tensor([1.0], requires_grad=True)
    _, new_duals = action.step(theta, lr=0.01, dual_update=adaptive_dual)
    assert new_duals.lambda_seg == pytest.approx(3.0)


# ── Migration status ────────────────────────────────────────────────────────


def test_migration_status_returns_json_serializable():
    action = make_action_from_track_callables(
        seg=_quadratic, pose=_linear,
        duals=DualVariables(lambda_seg=1.0, lambda_pose=2.0),
        metadata={"experiment_id": "smoke_001"},
    )
    status = action.migration_status()
    # Should round-trip through JSON.
    s = json.dumps(status)
    parsed = json.loads(s)
    assert parsed["schema"] == ACTION_SCHEMA_VERSION
    assert parsed["evidence_grade"] == ACTION_EVIDENCE_GRADE
    assert "seg_baseline" in parsed["active_tracks"]
    assert "pose_baseline" in parsed["active_tracks"]
    assert parsed["duals"]["lambda_seg"] == 1.0
    assert parsed["duals"]["lambda_pose"] == 2.0
    assert parsed["metadata"]["experiment_id"] == "smoke_001"


def test_migration_status_canonical_trainer_targets_present():
    action = Action()
    status = action.migration_status()
    targets = status["canonical_trainer_migration_targets"]
    assert "phase_1_balle_hyperprior" in targets
    assert "lane_12_v2_nerv_as_renderer" in targets
    assert "score_gradient_pr101_finetune" in targets


# ── DualVariables ───────────────────────────────────────────────────────────


def test_dual_variables_default_values():
    duals = DualVariables()
    # Baselines default to 1.0 (Boyd-style equal weight).
    assert duals.lambda_seg == 1.0
    assert duals.lambda_pose == 1.0
    assert duals.lambda_rate == 1.0
    # Refinement tracks default to 0.0 (off).
    assert duals.lambda_t7 == 0.0
    assert duals.lambda_t8 == 0.0
    assert duals.lambda_t11 == 0.0
    assert duals.lambda_t13 == 0.0
    assert duals.lambda_t20 == 0.0
    assert duals.lambda_t22 == 0.0


def test_dual_variables_immutable():
    duals = DualVariables(lambda_seg=1.0)
    with pytest.raises((AttributeError, Exception)):
        duals.lambda_seg = 99.0  # frozen=True


# ── TrackKind enum ──────────────────────────────────────────────────────────


def test_track_kind_values_match_canonical_landings():
    """Every TrackKind name must be a canonical landed track from CLAUDE.md."""
    expected_tracks = {
        "seg_baseline", "pose_baseline", "rate_baseline",
        "t7_fisher_rao", "t8_sinkhorn_w2", "t11_lovasz_hinge",
        "t13_joint_source_rd", "t19_adaptive_rho",
        "t20_kl_pose_distill", "t22_temporal_consistency",
        "lane_12_v2_nerv_as_renderer",
    }
    actual = {k.value for k in TrackKind}
    assert actual == expected_tracks


# ── Smoke: full integration with all tracks active ──────────────────────────


def test_smoke_all_tracks_active_returns_finite_S_total():
    """All 9 contributions active simultaneously; S_total must be finite."""
    action = make_action_from_track_callables(
        seg=_quadratic, pose=_linear, rate=lambda t: torch.tensor(1.0),
        t7_fisher_rao=_quadratic,
        t8_sinkhorn_w2=_quadratic,
        t11_lovasz_hinge=_quadratic,
        t13_joint_source_rd=_quadratic,
        t20_kl_pose_distill=_quadratic,
        t22_temporal_consistency=_quadratic,
        duals=DualVariables(
            lambda_seg=1.0, lambda_pose=1.0, lambda_rate=1.0,
            lambda_t7=0.1, lambda_t8=0.1, lambda_t11=0.1,
            lambda_t13=0.1, lambda_t20=0.1, lambda_t22=0.1,
        ),
    )
    theta = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    S = action.S_total(theta)
    assert torch.isfinite(S)
    assert S.item() > 0.0


def test_smoke_all_tracks_active_gradient_finite_and_correct_shape():
    action = make_action_from_track_callables(
        seg=_quadratic,
        t7_fisher_rao=_quadratic,
        t8_sinkhorn_w2=_quadratic,
        duals=DualVariables(lambda_seg=1.0, lambda_t7=0.1, lambda_t8=0.1),
    )
    theta = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    grads = action.gradient(theta)
    for kind, g in grads.items():
        assert g.shape == theta.shape, f"{kind}: shape mismatch"
        assert torch.all(torch.isfinite(g)), f"{kind}: non-finite grad"
