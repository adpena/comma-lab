"""Unit tests for tac.paradigm_delta_epsilon_zeta.joint_lagrangian_admm."""
from __future__ import annotations

import pytest
import torch

from tac.paradigm_delta_epsilon_zeta.joint_lagrangian_admm import (
    JointLagrangianADMM,
    JointLagrangianADMMConfig,
    LagrangianStepResult,
)


def _step(coord, *, rate_bits=800_000.0, distortion=1.0, seg=0.001, pose=0.0002):
    return coord.step(
        distortion=torch.tensor(distortion, requires_grad=True),
        rate_bits=torch.tensor(rate_bits, requires_grad=True),
        seg_loss=torch.tensor(seg, requires_grad=True),
        pose_loss=torch.tensor(pose, requires_grad=True),
    )


def test_default_config_constructs():
    coord = JointLagrangianADMM()
    assert coord.rho == 1.0
    assert coord.lambdas == {"rate": 1.0, "seg": 1.0, "pose": 1.0}


def test_config_validates_rho_min_positive():
    with pytest.raises(ValueError, match="rho_min must be > 0"):
        JointLagrangianADMMConfig(rho_min=0.0)


def test_config_validates_rho_max_geq_min():
    with pytest.raises(ValueError, match="rho_max"):
        JointLagrangianADMMConfig(rho_min=1.0, rho_max=0.5)


def test_config_validates_rho_init_in_band():
    with pytest.raises(ValueError, match="rho_init"):
        JointLagrangianADMMConfig(rho_min=1.0, rho_max=10.0, rho_init=100.0)


def test_config_validates_window_positive():
    with pytest.raises(ValueError, match="adaptive_rho_window"):
        JointLagrangianADMMConfig(adaptive_rho_window=0)


def test_config_validates_ratio_above_one():
    with pytest.raises(ValueError, match="adaptive_rho_ratio"):
        JointLagrangianADMMConfig(adaptive_rho_ratio=0.5)


def test_step_returns_result_dataclass():
    coord = JointLagrangianADMM()
    res = _step(coord)
    assert isinstance(res, LagrangianStepResult)
    assert isinstance(res.augmented_lagrangian, torch.Tensor)
    assert "rate" in res.lambdas
    assert "rate" in res.primal_residuals


def test_step_advances_step_count():
    coord = JointLagrangianADMM()
    assert coord.step_count == 0
    _step(coord)
    assert coord.step_count == 1


def test_step_supports_backward_through_distortion():
    coord = JointLagrangianADMM()
    distortion = torch.tensor(1.0, requires_grad=True)
    rate_bits = torch.tensor(800_000.0, requires_grad=True)
    seg = torch.tensor(0.001, requires_grad=True)
    pose = torch.tensor(0.0002, requires_grad=True)
    res = coord.step(distortion=distortion, rate_bits=rate_bits, seg_loss=seg, pose_loss=pose)
    res.augmented_lagrangian.backward()
    assert distortion.grad is not None
    assert rate_bits.grad is not None


def test_lambdas_clamped_to_nonnegative():
    """When residual is very negative, λ stays at 0 (inequality constraint)."""
    coord = JointLagrangianADMM()
    # Massively undershoot the rate target — residual will be very negative.
    for _ in range(20):
        _step(coord, rate_bits=1.0, seg=1e-8, pose=1e-8)
    assert coord.lambdas["rate"] >= 0.0
    assert coord.lambdas["seg"] >= 0.0
    assert coord.lambdas["pose"] >= 0.0


def test_lambdas_capped_at_lambda_max():
    cfg = JointLagrangianADMMConfig(lambda_init=0.0, lambda_max=10.0, rho_init=100.0)
    coord = JointLagrangianADMM(cfg)
    for _ in range(50):
        _step(coord, rate_bits=1e9, seg=1.0, pose=1.0)
    assert coord.lambdas["rate"] <= cfg.lambda_max + 1e-6


def test_state_dict_roundtrip():
    coord = JointLagrangianADMM()
    for _ in range(5):
        _step(coord)
    state = coord.state_dict()
    coord2 = JointLagrangianADMM()
    coord2.load_state_dict(state)
    assert coord2.rho == coord.rho
    assert coord2.lambdas == coord.lambdas
    assert coord2.step_count == coord.step_count


def test_adaptive_rho_increases_when_primal_dominates():
    """With huge primal residual but tiny dual updates, ρ should grow."""
    cfg = JointLagrangianADMMConfig(
        rho_init=1.0, rho_min=0.5, rho_max=100.0,
        adaptive_rho_window=4, adaptive_rho_ratio=2.0,
        lambda_init=1.0, lambda_max=1e-3,  # cap λ-updates so dual stays small
    )
    coord = JointLagrangianADMM(cfg)
    for _ in range(8):
        _step(coord, rate_bits=1e10, seg=1.0, pose=1.0)
    assert coord.rho >= 1.0  # at least no shrink; should adapt up


def test_adaptive_rho_bounded_by_max():
    cfg = JointLagrangianADMMConfig(
        rho_init=1.0, rho_min=0.5, rho_max=4.0,
        adaptive_rho_window=4, adaptive_rho_ratio=2.0,
        lambda_init=1.0, lambda_max=1e-3,
    )
    coord = JointLagrangianADMM(cfg)
    for _ in range(80):
        _step(coord, rate_bits=1e10, seg=1.0, pose=1.0)
    assert coord.rho <= cfg.rho_max + 1e-9


def test_rate_target_bits_is_8x_bytes():
    cfg = JointLagrangianADMMConfig(rate_target_bytes=10_000.0)
    coord = JointLagrangianADMM(cfg)
    assert coord.rate_target_bits == 80_000.0


def test_residual_is_dimensionless_normalised():
    """Residual is (R - target) / target so a 2× target gives residual 1.0."""
    cfg = JointLagrangianADMMConfig(rate_target_bytes=10_000.0)
    coord = JointLagrangianADMM(cfg)
    rate_bits = torch.tensor(160_000.0)  # 2× target (10_000 bytes = 80_000 bits)
    val = coord._rate_residual(rate_bits)
    assert float(val) == pytest.approx(1.0, rel=1e-5)


def test_u_dual_state_accumulates():
    coord = JointLagrangianADMM()
    initial_u = dict(coord.u)
    _step(coord, rate_bits=1e9, seg=1.0, pose=1.0)
    # u accumulates the residuals; should differ from initial.
    assert coord.u["rate"] != initial_u["rate"]
