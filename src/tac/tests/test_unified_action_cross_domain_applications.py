# SPDX-License-Identifier: MIT
"""Cross-domain application demonstration tests.

Per the synthesis memo §"OTHER APPLICATIONS" the SAME
``make_action_from_track_callables`` factory + the 3 utility curves +
the 4 wire-in solvers can serve 27 cross-domain applications spanning
per-tensor / per-pixel / per-byte score-axis surfaces. This test file
exercises 5 representative applications + 1 composition test to prove
the canonical surface composes across domains.

Applications covered (numbering matches the synthesis memo):

  #1  per-tensor archive bit budget — water-filling on R(D) curve
  #10 per-region UNIWARD perturbation — water-filling on inverse-variance
  #15 per-tensor ADMM consensus       — ADMM on R(D) curves
  #24 per-byte master-gradient sens.  — bit-allocator on per-byte gradient
  #27 multi-solver bandit             — choose_solver across all 3 utilities

Plus #composability: same Action object scored across all 3 utilities to
prove the unified surface is genuinely cross-pollination-friendly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
import torch

from tac.codec_magic_registry import CodecMagicEntry
from tac.joint_admm_coordinator import (
    AdmmResult,
    JointADMMConfig,
    ProximalStepResult,
)
from tac.unified_action import (
    Action,
    DualVariables,
    SolverChoice,
    SurfaceKind,
    choose_solver,
    evaluate_with_admm,
    evaluate_with_magic_codec,
    evaluate_with_water_filling,
    make_action_from_track_callables,
)
from tac.utility_curves import (
    per_byte_master_gradient_utility,
    per_pixel_inverse_variance_utility,
    per_tensor_rate_distortion_utility,
)


# ── Application #1: per-tensor archive bit budget via water-filling on R(D) ──


def test_application_1_per_tensor_archive_bit_budget_water_filling_on_rd():
    """Synthesis memo §OTHER APPLICATIONS #1.

    The Action's rate term IS the per-tensor R(D) utility. The water-filling
    router consumes per-tensor Hessian/variance/channel-count and emits a
    bit allocation that the autopilot ranker can compare against alternative
    treatments.
    """
    theta = torch.tensor([0.1, 0.3, 0.7, 1.5], requires_grad=True)
    action = make_action_from_track_callables(
        rate=per_tensor_rate_distortion_utility,
        duals=DualVariables(lambda_rate=1.0),
    )
    # Action's S_total exposes the R(D) contribution.
    S = action.S_total(theta)
    assert torch.isfinite(S)
    assert S.grad_fn is not None
    # Route the per-tensor rate through water-filling: build a 1-channel
    # toy quantization problem matching the R(D) curve.
    key = "stub.weight"
    h = {key: torch.ones(4, dtype=torch.float64)}
    v = {key: torch.ones(4, dtype=torch.float64)}
    counts = {key: [16, 16, 16, 16]}
    qint_max, duals_out = evaluate_with_water_filling(
        action,
        hessians=h,
        variances=v,
        channel_element_counts=counts,
        total_bit_budget=16 * 4 * 3,
    )
    assert len(qint_max[key]) == 4
    # The cross-pollination IS the wire-in: per-tensor R(D) utility from
    # tac.utility_curves + per-tensor water-filling from tac.water_filling_codec.
    assert all(q in (1, 3, 7, 15, 31) for q in qint_max[key])


# ── Application #10: per-region UNIWARD via water-filling on inverse-variance ──


def test_application_10_per_region_uniward_perturbation_water_filling():
    """Synthesis memo §OTHER APPLICATIONS #10.

    The Action's segmentation term IS the per-pixel UNIWARD utility. The
    cross-pollination: UNIWARD inverse-local-variance utility feeds the
    Action's seg-axis; a follow-on water-filling allocator (with per-region
    Hessian/variance estimates derived from local-variance statistics) can
    then sweep per-region perturbation budgets.

    This test exercises the utility + Action wire-up; the per-region
    water-filling sweep itself is an autopilot-ranker concern.
    """
    img = torch.randn(16, 16, requires_grad=True)
    action = make_action_from_track_callables(
        seg=per_pixel_inverse_variance_utility,
        duals=DualVariables(lambda_seg=2.0),
    )
    S = action.S_total(img)
    assert torch.isfinite(S)
    assert S.grad_fn is not None
    assert float(S) > 0.0  # UNIWARD inverse-variance is strictly positive


# ── Application #15: per-tensor ADMM consensus on R(D) curves ──


@dataclass
class _RateDistortionStream:
    """Stream wrapping the per-tensor R(D) utility as a StreamProximalCodec."""

    name: str
    sigma2: float

    def proximal_step(self, target_bytes: float, dual: float) -> ProximalStepResult:
        # Toy proximal step: realised_bytes = round(target_bytes); score scales
        # with how close we are to the R(D) optimum at sigma².
        encoded = int(max(round(target_bytes), 0))
        # R(D) = 0.5 * log2(sigma² / D) where D shrinks as encoded grows.
        D = max(self.sigma2 / max(encoded, 1), 1e-9)
        score = 0.5 * math.log2(max(self.sigma2 / D, 1.0))
        marginal = -1.0 / (max(D, 1e-9) * math.log(2.0))
        return ProximalStepResult(
            encoded_bytes=encoded,
            score_delta=float(score),
            marginal=float(marginal),
        )


def test_application_15_per_tensor_admm_consensus_on_rd_curves():
    """Synthesis memo §OTHER APPLICATIONS #15.

    Same R(D) utility on the Action's rate term, but solved jointly across
    multiple streams via ADMM consensus instead of per-tensor water-filling.
    """
    action = make_action_from_track_callables(
        rate=per_tensor_rate_distortion_utility,
        duals=DualVariables(lambda_rate=1.0),
    )
    streams = [
        _RateDistortionStream(name="A", sigma2=1.0),
        _RateDistortionStream(name="B", sigma2=4.0),
        _RateDistortionStream(name="C", sigma2=9.0),
    ]
    cfg = JointADMMConfig(
        rho_init=1.0,
        max_iters=10,
        byte_budget=300.0,
    )
    result = evaluate_with_admm(action, streams=streams, config=cfg)
    assert isinstance(result, AdmmResult)
    assert len(result.final_bytes_per_stream) == 3
    # ADMM with toy non-convex score curves may not converge in 10 iters;
    # the demo asserts the wire-up works end-to-end (returns finite bytes
    # per stream) rather than asserting global feasibility.
    for b in result.final_bytes_per_stream:
        assert isinstance(b, (int, float))
        assert b >= 0.0
        assert math.isfinite(b)


# ── Application #24: per-byte master-gradient sensitivity → bit allocator ──


def test_application_24_per_byte_master_gradient_sensitivity():
    """Synthesis memo §OTHER APPLICATIONS #24.

    The Action's rate term IS the per-byte master-gradient utility from
    tac.utility_curves. The autopilot ranker can consume the resulting
    Action scalar as a sensitivity-aware rate-axis cost and route it
    through the canonical bit_allocator_end_to_end (downstream consumer).
    """
    n_bytes = 50
    archive_bytes = torch.zeros(n_bytes)
    master_gradient = torch.randn(n_bytes, 3, requires_grad=True)

    def rate_callable(theta):
        # Closure: bind archive_bytes; theta is master_gradient.
        return per_byte_master_gradient_utility(archive_bytes, theta)

    action = make_action_from_track_callables(
        rate=rate_callable,
        duals=DualVariables(lambda_rate=1.0),
    )
    S = action.S_total(master_gradient)
    assert torch.isfinite(S)
    assert S.grad_fn is not None
    # Backwards pass should produce a sensible gradient over master_gradient.
    S.backward()
    assert master_gradient.grad is not None
    assert torch.isfinite(master_gradient.grad).all()


# ── Application #27: multi-solver bandit across all 3 wired solvers ──


def test_application_27_multi_solver_bandit_explores_all_three():
    """Synthesis memo §OTHER APPLICATIONS #27 — bandit picks across solvers.

    Multiple choose_solver calls with varied epsilon eventually exercise
    each of the 3 wired solvers (water_filling / admm / magic_codec).
    """
    picks: set[str] = set()
    for seed in range(50):
        choice = choose_solver(
            SurfaceKind.MASTER_GRADIENT_BOUNDARY,
            epsilon=0.9,  # high explore probability
            rng_seed=seed,
        )
        assert isinstance(choice, SolverChoice)
        picks.add(choice.solver)
    # With 50 seeds at epsilon=0.9, the explore branch should sample all 3.
    assert picks == {"water_filling", "admm", "magic_codec"}


# ── Composability: same Action object scored across all 3 utilities ──


def test_composability_same_action_factory_handles_all_3_utilities():
    """Demonstrates the synthesis memo's canonical-naming directive:
    ONE ``make_action_from_track_callables`` factory + ONE ``S_total``
    method handles per-tensor / per-pixel / per-byte utilities.
    """
    # Per-tensor R(D)
    theta_1d = torch.tensor([0.1, 0.3, 0.7, 1.5], requires_grad=True)
    action_rd = make_action_from_track_callables(
        rate=per_tensor_rate_distortion_utility,
        duals=DualVariables(lambda_rate=1.0),
    )
    S_rd = action_rd.S_total(theta_1d)
    assert torch.isfinite(S_rd)

    # Per-pixel UNIWARD
    img = torch.randn(16, 16, requires_grad=True)
    action_uni = make_action_from_track_callables(
        seg=per_pixel_inverse_variance_utility,
        duals=DualVariables(lambda_seg=1.0),
    )
    S_uni = action_uni.S_total(img)
    assert torch.isfinite(S_uni)

    # Per-byte master-gradient
    archive_bytes = torch.zeros(20)
    grad = torch.randn(20, 3, requires_grad=True)
    action_mg = make_action_from_track_callables(
        rate=lambda theta: per_byte_master_gradient_utility(archive_bytes, theta),
        duals=DualVariables(lambda_rate=1.0),
    )
    S_mg = action_mg.S_total(grad)
    assert torch.isfinite(S_mg)

    # All 3 Action objects produced finite scalars from the SAME factory ⇒
    # the canonical-naming-directive pay-off (per the synthesis memo) is real.
    assert S_rd.ndim == S_uni.ndim == S_mg.ndim == 0


# ── Magic-codec wire-in cross-domain: codec sniffing in archive composition ──


def test_application_archive_codec_sniffing_via_magic_codec_wire_in():
    """Demonstrates the magic-codec wire-in in a cross-domain context.

    Given a heterogeneous archive (per-tensor renderer + per-pixel mask +
    per-byte sidecar) the cross-pollination wire-in identifies the canonical
    decode routing for each section via ``evaluate_with_magic_codec``.
    """
    action = make_action_from_track_callables(
        seg=per_pixel_inverse_variance_utility,
        rate=per_tensor_rate_distortion_utility,
        duals=DualVariables(lambda_seg=1.0, lambda_rate=1.0),
    )
    tensor_bytes = {
        "renderer.water_filled": b"OWV2" + b"\x00" * 100,
        "renderer.sensitivity_v3": b"OWV3" + b"\x00" * 100,
        "renderer.imp_pruned": b"IMPS" + b"\x00" * 100,
        "renderer.legacy_unknown": b"WUT?" + b"\x00" * 100,
    }
    out = evaluate_with_magic_codec(action, tensor_bytes_by_name=tensor_bytes)
    assert isinstance(out["renderer.water_filled"], CodecMagicEntry)
    assert isinstance(out["renderer.sensitivity_v3"], CodecMagicEntry)
    assert isinstance(out["renderer.imp_pruned"], CodecMagicEntry)
    assert out["renderer.legacy_unknown"] is None  # unknown magic → caller decides


# ── Synthesis memo regression: ensure 26+ symbol public surface preserved ──


def test_unified_action_public_surface_size_at_least_30():
    """After the cross-pollination wire-in landing, the public surface
    grows from 26 to >=30 (4 new functions + SolverChoice dataclass).
    """
    import tac.unified_action as ua

    public = [s for s in dir(ua) if not s.startswith("_")]
    # Filter out imported re-exports (Mapping / Callable / etc.) from
    # the __all__ surface specifically.
    assert "evaluate_with_water_filling" in ua.__all__
    assert "evaluate_with_admm" in ua.__all__
    assert "evaluate_with_magic_codec" in ua.__all__
    assert "choose_solver" in ua.__all__
    assert "SolverChoice" in ua.__all__
    # public symbols include __all__ + Python stdlib imports; total >= 30
    assert len(public) >= 30, (
        f"Expected public surface >=30 after cross-pollination wire-ins; "
        f"got {len(public)}"
    )
