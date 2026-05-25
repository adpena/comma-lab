# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")
F = pytest.importorskip("torch.nn.functional")

from tac.local_acceleration.pr95_hnerv_mlx import stage_smoke_config  # noqa: E402
from tac.local_acceleration.pr95_hnerv_mlx_stage_losses import (  # noqa: E402
    PR95_MLX_STAGE_LOSS_CONTRACT_SCHEMA,
    PR95_MLX_STAGE_SCORER_LOSS_SURFACE,
    PR95_SEG_LOSS_CE,
    PR95_SEG_LOSS_L7_SOFTPLUS,
    PR95_SEG_LOSS_SMOOTH_DISAGREEMENT,
    PR95_SEG_LOSS_TAU_SOFTPLUS,
    pr95_mlx_cross_entropy_seg_loss,
    pr95_mlx_l7_softplus_seg_loss,
    pr95_mlx_pose_loss,
    pr95_mlx_smooth_disagreement_seg_loss,
    pr95_mlx_stage_loss_contract_from_training_config,
    pr95_mlx_stage_scorer_surrogate_loss,
    pr95_mlx_stage_seg_loss,
    pr95_mlx_tau_softplus_seg_loss,
)


def _fixture_logits_targets() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(9501)
    logits = rng.normal(0.0, 2.0, size=(2, 5, 4, 3)).astype(np.float32)
    targets = rng.integers(0, 5, size=(2, 4, 3), dtype=np.int64)
    return logits, targets


def _torch_tau_softplus(logits: np.ndarray, targets: np.ndarray, *, tau: float = 0.3) -> float:
    seg_logits = torch.tensor(logits)
    hard = torch.tensor(targets)
    target_logits = seg_logits.gather(1, hard.unsqueeze(1))
    masked = seg_logits.clone()
    masked.scatter_(1, hard.unsqueeze(1), -1e9)
    margin = target_logits - masked.max(dim=1, keepdim=True)[0]
    return float((tau * F.softplus(-margin / tau)).mean().item())


def _torch_smooth(logits: np.ndarray, targets: np.ndarray, *, tau: float = 0.3) -> float:
    seg_logits = torch.tensor(logits)
    hard = torch.tensor(targets)
    target_logits = seg_logits.gather(1, hard.unsqueeze(1))
    masked = seg_logits.clone()
    masked.scatter_(1, hard.unsqueeze(1), -1e9)
    margin = target_logits - masked.max(dim=1, keepdim=True)[0]
    return float(torch.sigmoid(-margin / tau).mean().item())


def _torch_l7(
    logits: np.ndarray,
    targets: np.ndarray,
    *,
    tau: float = 0.3,
    l7_threshold: float = 1.0,
    l7_mult: float = 4.0,
) -> float:
    seg_logits = torch.tensor(logits)
    hard = torch.tensor(targets)
    target_logits = seg_logits.gather(1, hard.unsqueeze(1))
    masked = seg_logits.clone()
    masked.scatter_(1, hard.unsqueeze(1), -1e9)
    margin = target_logits - masked.max(dim=1, keepdim=True)[0]
    per_pixel = tau * F.softplus(-margin / tau)
    with torch.no_grad():
        weights = 1.0 + l7_mult * (margin < l7_threshold).float()
        weights = weights / weights.mean()
    return float((per_pixel * weights).mean().item())


def _mlx_scalar(value) -> float:
    mx.eval(value)
    return float(np.asarray(value))


def test_pr95_mlx_segmentation_losses_match_public_torch_math() -> None:
    logits, targets = _fixture_logits_targets()
    logits_mx = mx.array(logits)
    targets_mx = mx.array(targets)

    ce_mlx = _mlx_scalar(pr95_mlx_cross_entropy_seg_loss(logits_mx, targets_mx))
    ce_torch = float(F.cross_entropy(torch.tensor(logits), torch.tensor(targets)).item())
    tau_mlx = _mlx_scalar(pr95_mlx_tau_softplus_seg_loss(logits_mx, targets_mx))
    smooth_mlx = _mlx_scalar(pr95_mlx_smooth_disagreement_seg_loss(logits_mx, targets_mx))
    l7_mlx = _mlx_scalar(pr95_mlx_l7_softplus_seg_loss(logits_mx, targets_mx))

    assert ce_mlx == pytest.approx(ce_torch, abs=1e-6)
    assert tau_mlx == pytest.approx(_torch_tau_softplus(logits, targets), abs=1e-6)
    assert smooth_mlx == pytest.approx(_torch_smooth(logits, targets), abs=1e-6)
    assert l7_mlx == pytest.approx(_torch_l7(logits, targets), abs=1e-6)


@pytest.mark.parametrize(
    ("family", "expected_fn"),
    [
        (
            PR95_SEG_LOSS_CE,
            lambda logits, targets: float(
                F.cross_entropy(torch.tensor(logits), torch.tensor(targets)).item()
            ),
        ),
        (PR95_SEG_LOSS_TAU_SOFTPLUS, _torch_tau_softplus),
        (PR95_SEG_LOSS_SMOOTH_DISAGREEMENT, _torch_smooth),
        (PR95_SEG_LOSS_L7_SOFTPLUS, _torch_l7),
    ],
)
def test_pr95_mlx_stage_seg_loss_dispatch_matches_torch(
    family: str,
    expected_fn,
) -> None:
    logits, targets = _fixture_logits_targets()

    actual = _mlx_scalar(
        pr95_mlx_stage_seg_loss(family, mx.array(logits), mx.array(targets))
    )

    assert actual == pytest.approx(expected_fn(logits, targets), abs=1e-6)


def test_pr95_mlx_pose_and_stage_surrogate_loss_match_public_torch_math() -> None:
    rng = np.random.default_rng(9502)
    logits, targets = _fixture_logits_targets()
    pose_pred = rng.normal(size=(2, 6)).astype(np.float32)
    pose_target = rng.normal(size=(2, 6)).astype(np.float32)
    cat_entropy = np.float32(3.25)

    mlx_pose = _mlx_scalar(
        pr95_mlx_pose_loss(mx.array(pose_pred), mx.array(pose_target))
    )
    torch_pose = float(
        torch.sqrt(
            10.0
            * F.mse_loss(torch.tensor(pose_pred), torch.tensor(pose_target))
            + 1e-12
        ).item()
    )
    total_mlx = _mlx_scalar(
        pr95_mlx_stage_scorer_surrogate_loss(
            seg_logits_nchw=mx.array(logits),
            targets_hard_nhw=mx.array(targets),
            pose_pred_first6=mx.array(pose_pred),
            pose_target_first6=mx.array(pose_target),
            loss_family=PR95_SEG_LOSS_L7_SOFTPLUS,
            cat_entropy_term=mx.array(cat_entropy),
            cat_lambda=0.02,
        )
    )
    total_torch = 100.0 * _torch_l7(logits, targets) + torch_pose + 0.02 * float(cat_entropy)

    assert mlx_pose == pytest.approx(torch_pose, abs=1e-6)
    assert total_mlx == pytest.approx(total_torch, abs=5e-5)


def test_pr95_stage_loss_contracts_are_wired_into_mlx_stage_config() -> None:
    stage1 = stage_smoke_config(1).source_stage_loss_contract
    stage8 = stage_smoke_config(8).source_stage_loss_contract

    assert stage1["schema"] == PR95_MLX_STAGE_LOSS_CONTRACT_SCHEMA
    assert stage1["loss_surface"] == PR95_MLX_STAGE_SCORER_LOSS_SURFACE
    assert stage1["seg_loss_family"] == PR95_SEG_LOSS_CE
    assert stage1["mlx_loss_primitives_implemented"] is True
    assert stage1["scorer_network_forward_gradient_wired"] is False
    assert stage8["seg_loss_family"] == PR95_SEG_LOSS_L7_SOFTPLUS
    assert stage8["cat_lambda"] == pytest.approx(0.02)
    assert stage8["cat_sigma"] == pytest.approx(0.1)
    assert stage8["uses_qat"] is True
    assert stage8["uses_muon"] is True
    assert stage8["score_claim"] is False
    assert stage8["ready_for_exact_eval_dispatch"] is False


def test_pr95_stage_loss_contract_rejects_unknown_family() -> None:
    with pytest.raises(ValueError, match="unsupported PR95 loss family"):
        pr95_mlx_stage_loss_contract_from_training_config(
            {"stage_loss_family": "not_a_pr95_loss"},
            stage_index=1,
        )
