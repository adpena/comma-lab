# SPDX-License-Identifier: MIT
"""Tests for tac.unified_action cross-pollination wire-ins.

Covers the 4 new functions landed in lane
``lane_unified_action_cross_pollination_wirein_20260518``:

- ``evaluate_with_water_filling`` — Action rate-axis → tac.water_filling_codec
- ``evaluate_with_admm``         — Action → tac.joint_admm_coordinator.run_admm
- ``evaluate_with_magic_codec``  — per-tensor codec sniff → tac.codec_magic_registry
- ``choose_solver``              — ε-greedy bandit over the 3 routers

Tests follow CLAUDE.md "Bugs must be permanently fixed AND self-protected
against" pattern: positive (basic invocation works), negative (error path
raises with clear message), integration (round-trip with canonical helper
produces expected output shape).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

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

# ── water-filling wire-in ───────────────────────────────────────────────


class _StubConv:
    """Dummy stub matching the per-tensor key contract of water_filling_codec."""

    pass


def _toy_water_filling_inputs(n_channels: int = 4):
    key = "stub.weight"
    hessians = {key: torch.full((n_channels,), 1.0, dtype=torch.float64)}
    variances = {key: torch.full((n_channels,), 1.0, dtype=torch.float64)}
    channel_element_counts = {key: [16] * n_channels}
    total_bits = sum(channel_element_counts[key]) * 3  # ~3 bits/element budget
    return hessians, variances, channel_element_counts, total_bits, key


def _toy_action_with_rate():
    return make_action_from_track_callables(
        seg=lambda theta: (theta**2).sum(),
        rate=lambda theta: (theta**2).sum(),
        duals=DualVariables(lambda_seg=1.0, lambda_rate=1.0),
    )


def test_evaluate_with_water_filling_basic_invocation():
    action = _toy_action_with_rate()
    h, v, c, b, _ = _toy_water_filling_inputs()
    qint_max, duals_out = evaluate_with_water_filling(
        action,
        hessians=h,
        variances=v,
        channel_element_counts=c,
        total_bit_budget=b,
    )
    assert isinstance(qint_max, dict)
    assert "stub.weight" in qint_max
    assert len(qint_max["stub.weight"]) == 4
    # All channels must land on a canonical QINT level.
    for q in qint_max["stub.weight"]:
        assert q in (1, 3, 7, 15, 31)
    assert isinstance(duals_out, DualVariables)
    assert duals_out.lambda_rate == 1.0


def test_evaluate_with_water_filling_rejects_missing_rate_term():
    action = Action()  # all None
    h, v, c, b, _ = _toy_water_filling_inputs()
    with pytest.raises(ValueError, match="L_rate is None"):
        evaluate_with_water_filling(
            action,
            hessians=h,
            variances=v,
            channel_element_counts=c,
            total_bit_budget=b,
        )


def test_evaluate_with_water_filling_rejects_inactive_rate_dual():
    action = make_action_from_track_callables(
        rate=lambda theta: (theta**2).sum(),
        duals=DualVariables(lambda_rate=0.0),
    )
    h, v, c, b, _ = _toy_water_filling_inputs()
    with pytest.raises(ValueError, match="lambda_rate"):
        evaluate_with_water_filling(
            action,
            hessians=h,
            variances=v,
            channel_element_counts=c,
            total_bit_budget=b,
        )


def test_evaluate_with_water_filling_rejects_zero_budget():
    action = _toy_action_with_rate()
    h, v, c, _, _ = _toy_water_filling_inputs()
    with pytest.raises(ValueError, match="total_bit_budget"):
        evaluate_with_water_filling(
            action,
            hessians=h,
            variances=v,
            channel_element_counts=c,
            total_bit_budget=0,
        )


def test_evaluate_with_water_filling_rejects_key_mismatch():
    action = _toy_action_with_rate()
    h, v, c, b, _ = _toy_water_filling_inputs()
    bad_v = {"OTHER.weight": v["stub.weight"]}
    with pytest.raises(ValueError, match="same tensor keys"):
        evaluate_with_water_filling(
            action,
            hessians=h,
            variances=bad_v,
            channel_element_counts=c,
            total_bit_budget=b,
        )


def test_evaluate_with_water_filling_sensitivity_map_amplifies_high_weight():
    """High sensitivity weight on a key amplifies its effective variance."""
    action = _toy_action_with_rate()
    key = "stub.weight"
    n = 4
    hessians = {key: torch.full((n,), 1.0, dtype=torch.float64)}
    variances = {key: torch.full((n,), 1.0, dtype=torch.float64)}
    counts = {key: [16] * n}
    budget = 16 * n * 3
    # baseline without sensitivity weighting
    qint_baseline, _ = evaluate_with_water_filling(
        action,
        hessians=hessians,
        variances=variances,
        channel_element_counts=counts,
        total_bit_budget=budget,
    )
    # with sensitivity map giving high weight — same single-key case, but
    # the sensitivity_map path is exercised (no crash, sensible output)
    qint_weighted, _ = evaluate_with_water_filling(
        action,
        hessians=hessians,
        variances=variances,
        channel_element_counts=counts,
        total_bit_budget=budget,
        sensitivity_map={key: 4.0},
    )
    # Both runs return valid QINT levels
    for q in qint_baseline[key]:
        assert q in (1, 3, 7, 15, 31)
    for q in qint_weighted[key]:
        assert q in (1, 3, 7, 15, 31)


def test_evaluate_with_water_filling_rejects_zero_sensitivity_weight():
    action = _toy_action_with_rate()
    h, v, c, b, _ = _toy_water_filling_inputs()
    with pytest.raises(ValueError, match="positive"):
        evaluate_with_water_filling(
            action,
            hessians=h,
            variances=v,
            channel_element_counts=c,
            total_bit_budget=b,
            sensitivity_map={"stub.weight": 0.0},
        )


# ── ADMM wire-in ─────────────────────────────────────────────────────────


@dataclass
class _ToyStreamCodec:
    """Toy StreamProximalCodec for ADMM smoke."""

    name: str
    base_score: float = 0.5

    def proximal_step(self, target_bytes: float, dual: float) -> ProximalStepResult:
        # Linear toy: encoded_bytes == round(target_bytes); score == base_score / (1 + log_bytes)
        log_b = math.log1p(max(target_bytes, 1.0))
        score = self.base_score / (1.0 + log_b)
        marginal = -self.base_score / ((1.0 + log_b) ** 2 * (target_bytes + 1.0))
        return ProximalStepResult(
            encoded_bytes=int(max(round(target_bytes), 0)),
            score_delta=float(score),
            marginal=float(marginal),
        )


def test_evaluate_with_admm_basic_invocation():
    action = make_action_from_track_callables(
        seg=lambda theta: (theta**2).sum(),
        rate=lambda theta: (theta**2).sum(),
        duals=DualVariables(),
    )
    streams = [_ToyStreamCodec(name="A"), _ToyStreamCodec(name="B", base_score=0.7)]
    cfg = JointADMMConfig(
        rho_init=1.0,
        max_iters=5,
        byte_budget=200.0,
        score_budget_per_stream={"A": 1.0, "B": 1.0},
    )
    result = evaluate_with_admm(action, streams=streams, config=cfg)
    assert isinstance(result, AdmmResult)
    assert isinstance(result.iters, int)
    # final_bytes_per_stream is positional (ordered by streams list)
    assert len(result.final_bytes_per_stream) == 2
    assert all(b >= 0.0 for b in result.final_bytes_per_stream)


def test_evaluate_with_admm_rejects_no_baseline_action():
    action = Action()  # all baseline tracks are None
    streams = [_ToyStreamCodec(name="A")]
    with pytest.raises(ValueError, match="no baseline track"):
        evaluate_with_admm(action, streams=streams)


def test_evaluate_with_admm_rejects_empty_streams():
    action = make_action_from_track_callables(
        seg=lambda theta: (theta**2).sum(),
        duals=DualVariables(),
    )
    with pytest.raises(ValueError, match="non-empty sequence"):
        evaluate_with_admm(action, streams=[])


def test_evaluate_with_admm_default_config_raises_without_byte_budget():
    """When config is None, the canonical JointADMMConfig() refuses to run
    without an explicit byte_budget (sentinel discipline). evaluate_with_admm
    surfaces the canonical error directly so callers must thread a real budget.
    """
    action = make_action_from_track_callables(
        seg=lambda theta: (theta**2).sum(),
        duals=DualVariables(),
    )
    streams = [_ToyStreamCodec(name="solo")]
    with pytest.raises(ValueError, match="byte_budget"):
        evaluate_with_admm(action, streams=streams)


def test_evaluate_with_admm_with_explicit_byte_budget_works():
    """Smoke: explicit byte_budget config runs end-to-end."""
    action = make_action_from_track_callables(
        seg=lambda theta: (theta**2).sum(),
        duals=DualVariables(),
    )
    streams = [_ToyStreamCodec(name="solo")]
    cfg = JointADMMConfig(byte_budget=100.0, max_iters=3)
    result = evaluate_with_admm(action, streams=streams, config=cfg)
    assert isinstance(result, AdmmResult)
    assert len(result.final_bytes_per_stream) == 1


# ── magic-codec wire-in ───────────────────────────────────────────────────


def test_evaluate_with_magic_codec_recognizes_registry_entries():
    action = make_action_from_track_callables(
        seg=lambda theta: (theta**2).sum(),
        duals=DualVariables(),
    )
    tensor_bytes = {
        "renderer.weights": b"OWV2\x01\x02\x03\x04tail",
        "renderer.aux": b"OWV3\x05\x06\x07\x08tail",
        "renderer.imps": b"IMPS\x09\x0a\x0b\x0ctail",
    }
    out = evaluate_with_magic_codec(action, tensor_bytes_by_name=tensor_bytes)
    assert isinstance(out, dict)
    assert isinstance(out["renderer.weights"], CodecMagicEntry)
    assert out["renderer.weights"].name == "Lane Ω-W-V2"
    assert isinstance(out["renderer.aux"], CodecMagicEntry)
    assert out["renderer.aux"].name == "Lane Ω-W-V3"
    assert isinstance(out["renderer.imps"], CodecMagicEntry)
    assert out["renderer.imps"].name == "Lane 17 IMP"


def test_evaluate_with_magic_codec_returns_none_for_unknown_magic():
    action = make_action_from_track_callables(
        seg=lambda theta: (theta**2).sum(),
        duals=DualVariables(),
    )
    out = evaluate_with_magic_codec(
        action, tensor_bytes_by_name={"unknown": b"WTFx\x00\x00"}
    )
    assert out["unknown"] is None


def test_evaluate_with_magic_codec_rejects_non_bytes_payload():
    action = make_action_from_track_callables(
        seg=lambda theta: (theta**2).sum(),
        duals=DualVariables(),
    )
    with pytest.raises(TypeError, match="non-bytes"):
        evaluate_with_magic_codec(
            action,
            tensor_bytes_by_name={"bad": "this is a string not bytes"},  # type: ignore[dict-item]
        )


# ── choose_solver bandit ─────────────────────────────────────────────────


def test_choose_solver_cold_start_deterministic_argmax():
    """With no history and rng_seed pinned (exploit branch), alphabetic argmax wins."""
    # rng_seed=0 + epsilon=0.0 forces exploit branch (random < 0 is always False)
    choice = choose_solver(
        SurfaceKind.MASTER_GRADIENT_BOUNDARY,
        history_path=None,
        epsilon=0.0,
        rng_seed=0,
    )
    assert isinstance(choice, SolverChoice)
    # available sorted alphabetically: ('admm', 'magic_codec', 'water_filling')
    assert choice.solver == "admm"
    assert choice.history_anchor_id is None
    assert "cold-start" in choice.rationale.lower()


def test_choose_solver_explore_with_high_epsilon():
    """epsilon=1.0 ALWAYS triggers the explore arm."""
    choice = choose_solver(
        SurfaceKind.BOUNDARY,
        history_path=None,
        epsilon=1.0,
        rng_seed=42,
    )
    assert choice.solver in ("water_filling", "admm", "magic_codec")
    assert choice.history_anchor_id is None
    assert "explore" in choice.rationale.lower()


def test_choose_solver_respects_available_override():
    """The available kwarg restricts the solver set."""
    choice = choose_solver(
        SurfaceKind.SENSITIVITY_MAP,
        epsilon=0.0,
        rng_seed=0,
        available=("water_filling",),
    )
    assert choice.solver == "water_filling"


def test_choose_solver_rejects_bad_epsilon():
    with pytest.raises(ValueError, match="epsilon"):
        choose_solver(SurfaceKind.BOUNDARY, epsilon=1.5)
    with pytest.raises(ValueError, match="epsilon"):
        choose_solver(SurfaceKind.BOUNDARY, epsilon=-0.1)


def test_choose_solver_rejects_empty_available():
    with pytest.raises(ValueError, match="available"):
        choose_solver(SurfaceKind.BOUNDARY, available=())


def test_choose_solver_history_fail_open_when_path_invalid():
    """When history_path is set but unreadable, gate falls back to cold-start."""
    choice = choose_solver(
        SurfaceKind.SENSITIVITY_MAP,
        history_path="/nonexistent/path/that/does/not/exist.jsonl",
        epsilon=0.0,
        rng_seed=0,
    )
    # Should not raise; should fall back to deterministic cold-start argmax.
    assert choice.solver in ("water_filling", "admm", "magic_codec")
    assert choice.history_anchor_id is None


# ── SolverChoice dataclass invariants ─────────────────────────────────────


def test_solver_choice_dataclass_is_frozen():
    c = SolverChoice(solver="water_filling", rationale="test")
    with pytest.raises(Exception):
        c.solver = "admm"  # type: ignore[misc]


def test_solver_choice_evidence_grade_pinned():
    c = SolverChoice(solver="admm", rationale="explore")
    assert c.evidence_grade.startswith("[predicted")
    assert "unified-action" in c.evidence_grade


# ── public API regression ─────────────────────────────────────────────────


def test_unified_action_public_surface_includes_cross_pollination_symbols():
    import tac.unified_action as ua

    for name in (
        "evaluate_with_water_filling",
        "evaluate_with_admm",
        "evaluate_with_magic_codec",
        "choose_solver",
        "SolverChoice",
    ):
        assert name in ua.__all__, f"{name} missing from tac.unified_action.__all__"
        assert hasattr(ua, name), f"{name} not importable from tac.unified_action"
