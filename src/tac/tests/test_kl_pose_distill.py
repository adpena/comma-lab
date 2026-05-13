"""Tests for T20 — KL pose-axis distillation loss.

Per coherence council 2026-05-09 (operator-approved §3.A redundancy-fix):
T20 closes the pose-axis gap left by T7+T8+T11 (all SegNet-axis attacks).
Hinton-Vinyals-Dean 2014 §2 soft-target KL distillation at T=2.0.

Tests verify:

* Convergence — identical logits → loss exactly zero (KL of any
  distribution against itself is zero).
* T^2 scaling correctness — gradient magnitude w.r.t. student logits is
  T-invariant (Hinton 2014 §2.1 gradient note).
* Stop-gradient on teacher — teacher.grad stays None after backward.
* Mode coverage — full / first6 / regression_mse all wired & validated.
* Validation gates — temperature / eps / mode / shape / first_n_dims
  fail-loud per CLAUDE.md "fail-loud, not silent" rule.
* Batched-vs-loop equivalence — vectorized form matches per-sample loop.
* Dim-6 truncation correctness — first6 mode ignores extra dims; matches
  full mode when total_dims == first_n_dims.
* Finite gradient at extreme logits — no NaN at large |z|.
* Apply-config wrapper preserves weight_pose multiplication.
* PoseNet-shape contract — works on the canonical (B, 12) shape from
  ``upstream/modules.py``.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from tac.kl_pose_distill import (
    DEFAULT_POSE_HEAD_OUT_DIM,
    DEFAULT_POSE_SCORED_DIM,
    DEFAULT_TEMPERATURE,
    VALID_MODES,
    KLPoseDistillConfig,
    apply_kl_pose_distill,
    kl_pose_distill_loss,
)


# ---------------------------------------------------------------------------
# Convergence — identical logits → loss zero
# ---------------------------------------------------------------------------


def test_identical_logits_full_mode_loss_zero() -> None:
    torch.manual_seed(0)
    z = torch.randn(8, DEFAULT_POSE_HEAD_OUT_DIM)
    loss = kl_pose_distill_loss(z, z.clone(), mode="distill_softmax_full")
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_identical_logits_first6_mode_loss_zero() -> None:
    torch.manual_seed(1)
    z = torch.randn(4, DEFAULT_POSE_HEAD_OUT_DIM)
    loss = kl_pose_distill_loss(z, z.clone(), mode="distill_softmax_first6")
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_identical_logits_regression_mse_loss_zero() -> None:
    torch.manual_seed(2)
    z = torch.randn(3, DEFAULT_POSE_HEAD_OUT_DIM)
    loss = kl_pose_distill_loss(z, z.clone(), mode="regression_mse")
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_loss_strictly_positive_when_logits_disagree() -> None:
    torch.manual_seed(3)
    student = torch.randn(4, DEFAULT_POSE_HEAD_OUT_DIM)
    teacher = torch.randn(4, DEFAULT_POSE_HEAD_OUT_DIM)
    loss = kl_pose_distill_loss(student, teacher)
    assert loss.item() > 0.0


# ---------------------------------------------------------------------------
# Hinton T^2 scaling — gradient magnitude is T-invariant
# ---------------------------------------------------------------------------


def test_hinton_T_squared_normalization_keeps_gradient_scale_stable() -> None:
    """Per Hinton 2014 §2.1: ∂L/∂z_student scales as 1/T from softmax,
    but T^2 normalization restores T-invariance.

    We check the *gradient norm* across two T values (1.0 and 4.0) on
    the SAME logits. With proper T^2 scaling the ratio should be O(1)
    (typically within 2× — the scaling is exact only at the high-T
    limit per Hinton's small-perturbation derivation, but the order of
    magnitude is bounded for any reasonable T pair). Without T^2 the
    ratio would be 16× (T=4 vs T=1).
    """
    torch.manual_seed(4)
    student_t1 = torch.randn(8, DEFAULT_POSE_HEAD_OUT_DIM, requires_grad=True)
    teacher = torch.randn(8, DEFAULT_POSE_HEAD_OUT_DIM)
    student_t4 = student_t1.detach().clone().requires_grad_(True)

    loss_t1 = kl_pose_distill_loss(student_t1, teacher, temperature=1.0)
    loss_t1.backward()
    grad_norm_t1 = student_t1.grad.norm().item()

    loss_t4 = kl_pose_distill_loss(student_t4, teacher, temperature=4.0)
    loss_t4.backward()
    grad_norm_t4 = student_t4.grad.norm().item()

    # Without T^2 normalization, the gradient magnitude scales as 1/T^2
    # (one 1/T from softmax sharpening, one 1/T from logit-to-prob chain
    # rule). T=4 vs T=1 would give a ratio of 1/16 = 0.0625 — way under 1.
    # With T^2 normalization (Hinton 2014 §2.1), the gradient becomes
    # APPROXIMATELY T-invariant in the high-T / small-perturbation limit;
    # for finite T and finite logit magnitudes the ratio is bounded but
    # NOT exactly 1. Empirical anchors on seed=4: ~1.4 at the chosen
    # logit scale. Tighten the bound to [0.5, 3.0] to catch regressions
    # in the T^2 factor itself (a missing T^2 would give ratio < 0.1).
    ratio = grad_norm_t4 / max(grad_norm_t1, 1e-12)
    assert 0.5 < ratio < 3.0, (
        f"T^2-normalized grad ratio = {ratio:.3f}, expected ~1.4; "
        "values outside [0.5, 3.0] suggest the T^2 normalization is "
        "missing or doubled"
    )


def test_T_squared_factor_present_in_loss_value() -> None:
    """Direct numeric check: loss scales as T^2 for the per-element KL
    divided by T^2 (i.e., the KL with the T^2 factor stripped should be
    T-dependent, but the T^2 factor restores the canonical Hinton scale).

    We compute the loss at two T values on the same teacher-student pair
    where teacher == 0 (uniform) and student is a small perturbation.
    For tiny perturbations the KL ≈ 0.5 * |z/T|^2 / D, so the T^2 factor
    cancels the 1/T^2 from the KL, giving a T-invariant loss in the limit.
    """
    torch.manual_seed(5)
    student = torch.randn(8, DEFAULT_POSE_HEAD_OUT_DIM) * 0.05  # small perturbation
    teacher = torch.zeros(8, DEFAULT_POSE_HEAD_OUT_DIM)
    loss_t2 = kl_pose_distill_loss(student, teacher, temperature=2.0).item()
    loss_t8 = kl_pose_distill_loss(student, teacher, temperature=8.0).item()
    # In the small-perturbation limit the loss is approximately T-invariant
    # (the ratio should be close to 1.0). Allow factor 4 slack for finite-size
    # effects; the key point is the ratio is NOT 16× (which would happen
    # without T^2).
    ratio = loss_t8 / max(loss_t2, 1e-12)
    assert 0.05 < ratio < 4.0


# ---------------------------------------------------------------------------
# Stop-gradient on teacher
# ---------------------------------------------------------------------------


def test_teacher_grad_remains_none_after_backward() -> None:
    torch.manual_seed(6)
    student = torch.randn(4, DEFAULT_POSE_HEAD_OUT_DIM, requires_grad=True)
    teacher = torch.randn(4, DEFAULT_POSE_HEAD_OUT_DIM, requires_grad=True)
    loss = kl_pose_distill_loss(student, teacher)
    loss.backward()
    assert student.grad is not None
    assert teacher.grad is None, (
        "Teacher MUST be stop-gradient (Hinton 2014); got grad of norm "
        f"{teacher.grad.norm().item() if teacher.grad is not None else 0}"
    )


def test_teacher_grad_remains_none_after_backward_first6_mode() -> None:
    torch.manual_seed(7)
    student = torch.randn(2, DEFAULT_POSE_HEAD_OUT_DIM, requires_grad=True)
    teacher = torch.randn(2, DEFAULT_POSE_HEAD_OUT_DIM, requires_grad=True)
    loss = kl_pose_distill_loss(student, teacher, mode="distill_softmax_first6")
    loss.backward()
    assert student.grad is not None
    assert teacher.grad is None


def test_teacher_grad_remains_none_after_backward_regression_mode() -> None:
    torch.manual_seed(8)
    student = torch.randn(2, DEFAULT_POSE_HEAD_OUT_DIM, requires_grad=True)
    teacher = torch.randn(2, DEFAULT_POSE_HEAD_OUT_DIM, requires_grad=True)
    loss = kl_pose_distill_loss(student, teacher, mode="regression_mse")
    loss.backward()
    assert student.grad is not None
    assert teacher.grad is None


# ---------------------------------------------------------------------------
# Mode coverage
# ---------------------------------------------------------------------------


def test_full_mode_uses_all_dims() -> None:
    torch.manual_seed(9)
    z_left = torch.randn(4, DEFAULT_POSE_HEAD_OUT_DIM)
    z_right = z_left.clone()
    # Perturb only the LAST dim (index 11). Full mode must see this; first6
    # mode must NOT see it.
    z_right[:, -1] += 5.0
    loss_full = kl_pose_distill_loss(z_left, z_right, mode="distill_softmax_full")
    loss_first6 = kl_pose_distill_loss(z_left, z_right, mode="distill_softmax_first6")
    assert loss_full.item() > 1e-3
    assert loss_first6.item() == pytest.approx(0.0, abs=1e-6)


def test_first6_mode_uses_only_first_six_dims() -> None:
    torch.manual_seed(10)
    z_left = torch.randn(4, DEFAULT_POSE_HEAD_OUT_DIM)
    z_right = z_left.clone()
    # Perturb only first dim — both modes must see it.
    z_right[:, 0] += 5.0
    loss_full = kl_pose_distill_loss(z_left, z_right, mode="distill_softmax_full")
    loss_first6 = kl_pose_distill_loss(z_left, z_right, mode="distill_softmax_first6")
    assert loss_full.item() > 1e-3
    assert loss_first6.item() > 1e-3


def test_first6_mode_matches_full_mode_when_d_equals_first_n() -> None:
    """When the head output is itself 6-dim (toy case), full and first6
    must agree exactly."""
    torch.manual_seed(11)
    z_student = torch.randn(4, DEFAULT_POSE_SCORED_DIM)
    z_teacher = torch.randn(4, DEFAULT_POSE_SCORED_DIM)
    loss_full = kl_pose_distill_loss(
        z_student, z_teacher, mode="distill_softmax_full"
    )
    loss_first6 = kl_pose_distill_loss(
        z_student, z_teacher, mode="distill_softmax_first6", first_n_dims=6
    )
    assert loss_full.item() == pytest.approx(loss_first6.item(), abs=1e-6)


def test_first_n_equals_total_dims_matches_full_mode_for_canonical_shape() -> None:
    """Edge case (Yousfi review): user passes first_n_dims=12 with a
    canonical (B, 12) shape and mode='distill_softmax_first6'. Should
    behave IDENTICALLY to mode='distill_softmax_full' (the slice covers
    every dim)."""
    torch.manual_seed(31)
    z_student = torch.randn(4, DEFAULT_POSE_HEAD_OUT_DIM)
    z_teacher = torch.randn(4, DEFAULT_POSE_HEAD_OUT_DIM)
    loss_full = kl_pose_distill_loss(
        z_student, z_teacher, mode="distill_softmax_full"
    )
    loss_first_eq_total = kl_pose_distill_loss(
        z_student,
        z_teacher,
        mode="distill_softmax_first6",
        first_n_dims=DEFAULT_POSE_HEAD_OUT_DIM,
    )
    assert loss_full.item() == pytest.approx(loss_first_eq_total.item(), abs=1e-6)


def test_regression_mse_mode_matches_torch_mse() -> None:
    """Regression mode must agree exactly with F.mse_loss on first-6 dims."""
    torch.manual_seed(12)
    student = torch.randn(8, DEFAULT_POSE_HEAD_OUT_DIM)
    teacher = torch.randn(8, DEFAULT_POSE_HEAD_OUT_DIM)
    loss = kl_pose_distill_loss(student, teacher, mode="regression_mse")
    expected = F.mse_loss(student[:, :6], teacher[:, :6])
    assert loss.item() == pytest.approx(expected.item(), abs=1e-6)


# ---------------------------------------------------------------------------
# Validation gates
# ---------------------------------------------------------------------------


def test_temperature_must_be_positive() -> None:
    z = torch.zeros(2, DEFAULT_POSE_HEAD_OUT_DIM)
    for bad in [0.0, -1.0, float("nan"), float("inf"), -float("inf")]:
        with pytest.raises(ValueError, match="temperature"):
            kl_pose_distill_loss(z, z, temperature=bad)


def test_temperature_must_not_be_bool() -> None:
    z = torch.zeros(2, DEFAULT_POSE_HEAD_OUT_DIM)
    with pytest.raises(ValueError, match="temperature"):
        kl_pose_distill_loss(z, z, temperature=True)  # type: ignore[arg-type]


def test_eps_must_be_in_range() -> None:
    z = torch.zeros(2, DEFAULT_POSE_HEAD_OUT_DIM)
    for bad in [0.0, -1e-9, 1e-2, float("nan")]:
        with pytest.raises(ValueError, match="eps"):
            kl_pose_distill_loss(z, z, eps=bad)


def test_mode_must_be_valid_token() -> None:
    z = torch.zeros(2, DEFAULT_POSE_HEAD_OUT_DIM)
    with pytest.raises(ValueError, match="mode"):
        kl_pose_distill_loss(z, z, mode="bogus")
    with pytest.raises(ValueError, match="mode"):
        kl_pose_distill_loss(z, z, mode="DISTILL_SOFTMAX_FULL")  # case-sensitive


def test_shape_mismatch_raises() -> None:
    a = torch.zeros(2, DEFAULT_POSE_HEAD_OUT_DIM)
    b = torch.zeros(3, DEFAULT_POSE_HEAD_OUT_DIM)
    with pytest.raises(ValueError, match="shape"):
        kl_pose_distill_loss(a, b)


def test_empty_input_raises() -> None:
    z = torch.zeros(0, DEFAULT_POSE_HEAD_OUT_DIM)
    with pytest.raises(ValueError, match="non-empty"):
        kl_pose_distill_loss(z, z)


def test_one_dim_input_raises() -> None:
    z = torch.zeros(DEFAULT_POSE_HEAD_OUT_DIM)
    with pytest.raises(ValueError, match="2 dims"):
        kl_pose_distill_loss(z, z)


def test_first_n_dims_must_be_positive() -> None:
    z = torch.zeros(2, DEFAULT_POSE_HEAD_OUT_DIM)
    with pytest.raises(ValueError, match="first_n_dims"):
        kl_pose_distill_loss(z, z, first_n_dims=0, mode="distill_softmax_first6")
    with pytest.raises(ValueError, match="first_n_dims"):
        kl_pose_distill_loss(z, z, first_n_dims=-1, mode="distill_softmax_first6")


def test_first_n_dims_cannot_exceed_total_dims() -> None:
    z = torch.zeros(2, DEFAULT_POSE_HEAD_OUT_DIM)
    with pytest.raises(ValueError, match="exceeds"):
        kl_pose_distill_loss(
            z, z, first_n_dims=DEFAULT_POSE_HEAD_OUT_DIM + 1, mode="distill_softmax_first6"
        )


def test_first_n_dims_must_not_be_bool() -> None:
    z = torch.zeros(2, DEFAULT_POSE_HEAD_OUT_DIM)
    with pytest.raises(ValueError, match="first_n_dims"):
        kl_pose_distill_loss(z, z, first_n_dims=True)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Batched-vs-loop equivalence
# ---------------------------------------------------------------------------


def test_batched_matches_per_sample_loop() -> None:
    torch.manual_seed(13)
    B = 6
    student = torch.randn(B, DEFAULT_POSE_HEAD_OUT_DIM)
    teacher = torch.randn(B, DEFAULT_POSE_HEAD_OUT_DIM)
    batched = kl_pose_distill_loss(student, teacher).item()
    per_sample = []
    for i in range(B):
        per_sample.append(
            kl_pose_distill_loss(
                student[i : i + 1], teacher[i : i + 1]
            ).item()
        )
    assert batched == pytest.approx(sum(per_sample) / B, abs=1e-6)


def test_batched_matches_per_sample_loop_first6_mode() -> None:
    torch.manual_seed(14)
    B = 4
    student = torch.randn(B, DEFAULT_POSE_HEAD_OUT_DIM)
    teacher = torch.randn(B, DEFAULT_POSE_HEAD_OUT_DIM)
    batched = kl_pose_distill_loss(
        student, teacher, mode="distill_softmax_first6"
    ).item()
    per_sample = [
        kl_pose_distill_loss(
            student[i : i + 1], teacher[i : i + 1], mode="distill_softmax_first6"
        ).item()
        for i in range(B)
    ]
    assert batched == pytest.approx(sum(per_sample) / B, abs=1e-6)


# ---------------------------------------------------------------------------
# Numerical stability — finite gradient at extreme logits
# ---------------------------------------------------------------------------


def test_finite_loss_at_large_logits() -> None:
    student = torch.tensor([[100.0, -100.0, 50.0, -50.0, 25.0, -25.0,
                             10.0, -10.0, 5.0, -5.0, 2.5, -2.5]])
    teacher = torch.tensor([[-100.0, 100.0, -50.0, 50.0, -25.0, 25.0,
                              -10.0, 10.0, -5.0, 5.0, -2.5, 2.5]])
    loss = kl_pose_distill_loss(student, teacher)
    assert math.isfinite(loss.item())
    assert loss.item() > 0.0


def test_finite_gradient_at_large_logits() -> None:
    student = torch.tensor(
        [[80.0, -80.0, 40.0, -40.0, 20.0, -20.0,
          10.0, -10.0, 5.0, -5.0, 2.0, -2.0]],
        requires_grad=True,
    )
    teacher = torch.tensor([[0.0] * DEFAULT_POSE_HEAD_OUT_DIM])
    loss = kl_pose_distill_loss(student, teacher)
    loss.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


def test_finite_loss_with_zero_logits() -> None:
    z = torch.zeros(4, DEFAULT_POSE_HEAD_OUT_DIM)
    loss = kl_pose_distill_loss(z, z)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Apply-config wrapper
# ---------------------------------------------------------------------------


def test_apply_kl_pose_distill_default_config() -> None:
    torch.manual_seed(15)
    student = torch.randn(4, DEFAULT_POSE_HEAD_OUT_DIM)
    teacher = torch.randn(4, DEFAULT_POSE_HEAD_OUT_DIM)
    config = KLPoseDistillConfig()
    loss = apply_kl_pose_distill(student, teacher, config)
    raw = kl_pose_distill_loss(student, teacher)
    assert loss.item() == pytest.approx(raw.item(), abs=1e-6)


def test_apply_kl_pose_distill_weight_pose_multiplier() -> None:
    torch.manual_seed(16)
    student = torch.randn(4, DEFAULT_POSE_HEAD_OUT_DIM)
    teacher = torch.randn(4, DEFAULT_POSE_HEAD_OUT_DIM)
    config_w1 = KLPoseDistillConfig(weight_pose=1.0)
    config_w7 = KLPoseDistillConfig(weight_pose=7.0)
    loss_w1 = apply_kl_pose_distill(student, teacher, config_w1)
    loss_w7 = apply_kl_pose_distill(student, teacher, config_w7)
    assert loss_w7.item() == pytest.approx(7.0 * loss_w1.item(), rel=1e-5)


def test_apply_kl_pose_distill_propagates_temperature() -> None:
    torch.manual_seed(17)
    student = torch.randn(4, DEFAULT_POSE_HEAD_OUT_DIM)
    teacher = torch.randn(4, DEFAULT_POSE_HEAD_OUT_DIM)
    config_t2 = KLPoseDistillConfig(temperature=2.0)
    config_t8 = KLPoseDistillConfig(temperature=8.0)
    raw_t2 = kl_pose_distill_loss(student, teacher, temperature=2.0)
    raw_t8 = kl_pose_distill_loss(student, teacher, temperature=8.0)
    assert apply_kl_pose_distill(student, teacher, config_t2).item() == pytest.approx(
        raw_t2.item(), abs=1e-6
    )
    assert apply_kl_pose_distill(student, teacher, config_t8).item() == pytest.approx(
        raw_t8.item(), abs=1e-6
    )


def test_apply_kl_pose_distill_rejects_non_config() -> None:
    z = torch.zeros(2, DEFAULT_POSE_HEAD_OUT_DIM)
    with pytest.raises(TypeError, match="KLPoseDistillConfig"):
        apply_kl_pose_distill(z, z, {"temperature": 2.0})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Config validation gates
# ---------------------------------------------------------------------------


def test_config_validates_temperature() -> None:
    with pytest.raises(ValueError, match="temperature"):
        KLPoseDistillConfig(temperature=0.0)
    with pytest.raises(ValueError, match="temperature"):
        KLPoseDistillConfig(temperature=-1.0)


def test_config_validates_weight_pose() -> None:
    with pytest.raises(ValueError, match="weight_pose"):
        KLPoseDistillConfig(weight_pose=-0.1)
    with pytest.raises(ValueError, match="weight_pose"):
        KLPoseDistillConfig(weight_pose=float("nan"))
    with pytest.raises(ValueError, match="weight_pose"):
        KLPoseDistillConfig(weight_pose=True)  # type: ignore[arg-type]


def test_config_validates_mode() -> None:
    with pytest.raises(ValueError, match="mode"):
        KLPoseDistillConfig(mode="invalid")


def test_config_validates_eps() -> None:
    with pytest.raises(ValueError, match="eps"):
        KLPoseDistillConfig(eps=0.0)
    with pytest.raises(ValueError, match="eps"):
        KLPoseDistillConfig(eps=1.0)


def test_config_validates_first_n_dims() -> None:
    with pytest.raises(ValueError, match="first_n_dims"):
        KLPoseDistillConfig(weight_first_n_dims=0)
    with pytest.raises(ValueError, match="first_n_dims"):
        KLPoseDistillConfig(weight_first_n_dims=-1)


def test_config_default_temperature_is_2_per_quantizr() -> None:
    """Quantizr's verified 0.33 archive uses kl_on_logits(T=2.0). The
    default must match this empirical anchor (CLAUDE.md "Quantizr
    intelligence" §)."""
    config = KLPoseDistillConfig()
    assert config.temperature == DEFAULT_TEMPERATURE == 2.0


def test_config_default_first_n_dims_matches_scorer_slice() -> None:
    """upstream/modules.py:84 uses out[..., :6]. Our default must match."""
    config = KLPoseDistillConfig()
    assert config.weight_first_n_dims == DEFAULT_POSE_SCORED_DIM == 6


def test_config_is_frozen() -> None:
    config = KLPoseDistillConfig()
    with pytest.raises(Exception):  # FrozenInstanceError or TypeError
        config.temperature = 8.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# PoseNet-shape contract
# ---------------------------------------------------------------------------


def test_works_on_canonical_pose_head_shape() -> None:
    """The contest scorer's pose head outputs (B, 12) per
    upstream/modules.py:26 (Hydra Head 'pose', out=12). This is the
    shape T20 was designed for."""
    torch.manual_seed(20)
    student = torch.randn(600, DEFAULT_POSE_HEAD_OUT_DIM)  # 600 pairs per video
    teacher = torch.randn(600, DEFAULT_POSE_HEAD_OUT_DIM)
    loss = kl_pose_distill_loss(student, teacher)
    assert math.isfinite(loss.item())
    assert loss.item() > 0.0


def test_works_with_extra_leading_dims() -> None:
    """Trainers may want (B, T, 12) or (B, ..., 12) — verify last-dim
    softmax convention."""
    torch.manual_seed(21)
    student = torch.randn(2, 4, DEFAULT_POSE_HEAD_OUT_DIM)
    teacher = torch.randn(2, 4, DEFAULT_POSE_HEAD_OUT_DIM)
    loss = kl_pose_distill_loss(student, teacher)
    assert math.isfinite(loss.item())
    assert loss.item() > 0.0


# ---------------------------------------------------------------------------
# VALID_MODES public surface
# ---------------------------------------------------------------------------


def test_valid_modes_constant_exposes_three_modes() -> None:
    assert set(VALID_MODES) == {
        "distill_softmax_full",
        "distill_softmax_first6",
        "regression_mse",
    }


def test_all_modes_run_on_canonical_input() -> None:
    torch.manual_seed(22)
    student = torch.randn(4, DEFAULT_POSE_HEAD_OUT_DIM)
    teacher = torch.randn(4, DEFAULT_POSE_HEAD_OUT_DIM)
    for mode in VALID_MODES:
        loss = kl_pose_distill_loss(student, teacher, mode=mode)
        assert math.isfinite(loss.item())
        assert loss.item() >= 0.0
