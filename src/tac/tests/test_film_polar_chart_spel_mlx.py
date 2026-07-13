# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the resumable FiLM polar-chart MCSD/SPEL finisher."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tac.canonical_equations.witness_film_polar_chart_spel_20260713 import (
    build_witness_film_polar_chart_spel_v1,
    chart_reconstruction_relative_fro,
    stiefel_tangent_residual_fro,
)
from tac.optimization.film_polar_chart_spel_mlx import (
    RESUME_PREFIX,
    FilmPolarChartSPELState,
    muon_aspect_ratio_scale,
    polar_chart_numpy,
    spel_step_mlx_arrays,
    spel_step_numpy,
)
from tac.optimization.muon_finisher_mlx import _adamw_bias_correction_for
from tac.witness_control.resume_registry import DIRECT_CONTROLLER_NAMES, ResumeRegistry
from tac.witness_dsl import curriculum_dsl as cd
from tac.witness_dsl import lever_registry as lever_registry


def _full_rank_weight(seed: int = 7, shape: tuple[int, int] = (31, 7)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    weight = rng.normal(size=shape).astype(np.float32)
    weight += np.eye(*shape, dtype=np.float32)
    return weight


def _grad(seed: int = 19, shape: tuple[int, int] = (31, 7)) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=shape).astype(np.float32)


def test_polar_chart_preserves_boundary_function_and_stiefel_constraint():
    weight = _full_rank_weight()
    q, h0 = polar_chart_numpy(weight)
    relative_reconstruction = np.linalg.norm(q @ h0 - weight) / np.linalg.norm(weight)
    orthogonality = np.linalg.norm(q.T @ q - np.eye(q.shape[1]), ord="fro")
    assert relative_reconstruction < 2e-6
    assert orthogonality < 2e-6
    assert chart_reconstruction_relative_fro(weight) < 2e-6
    assert stiefel_tangent_residual_fro(q, _grad(shape=q.shape)) < 2e-5
    assert muon_aspect_ratio_scale((768, 19)) == pytest.approx((768 / 19) ** 0.5)


def test_canonical_equation_names_approximation_and_triality_consumers():
    equation = build_witness_film_polar_chart_spel_v1()
    assert equation.equation_id == "witness_film_polar_chart_spel_v1"
    assert "NOT exact" in equation.domain_of_validity["approximation_status"]
    consumers = "\n".join(equation.canonical_consumers)
    assert "curriculum_dsl" in consumers
    assert "train_levelset_witness_realized_through_R_mlx" in consumers
    assert "DAG_FEED" in consumers


def test_numpy_spel_step_is_seed_deterministic_and_stays_on_manifold():
    q, h0 = polar_chart_numpy(_full_rank_weight())
    zero = np.zeros_like(q)
    kwargs = {
        "learning_rate": 2e-3,
        "momentum_beta": 0.95,
        "nesterov": True,
        "ns_steps": 5,
        "ema_decay": 0.997,
    }
    first = spel_step_numpy(q, h0, zero, q, _grad(), **kwargs)
    second = spel_step_numpy(q, h0, zero, q, _grad(), **kwargs)
    for name in ("q", "h0", "momentum", "q_ema", "weight"):
        assert np.array_equal(getattr(first, name), getattr(second, name))
    assert first.orthogonality_residual_fro < 2e-6
    assert first.tangent_residual_fro < 2e-5
    assert first.direction_spectral_norm <= 1.0 + 2e-6
    assert not np.array_equal(first.weight, q @ h0), "a nonzero task gradient must move FiLM"


def test_registry_roundtrip_and_split_resume_match_uninterrupted():
    weight = _full_rank_weight()
    gradients = [_grad(seed=i) for i in range(6)]

    uninterrupted = FilmPolarChartSPELState()
    uninterrupted.initialize_numpy(weight)
    for grad in gradients:
        uninterrupted.step_numpy(grad, learning_rate=2e-3, ema_decay=0.997)

    split = FilmPolarChartSPELState()
    split.initialize_numpy(weight)
    for grad in gradients[:3]:
        split.step_numpy(grad, learning_rate=2e-3, ema_decay=0.997)
    registry = ResumeRegistry()
    registry.register("film_polar_chart_spel", RESUME_PREFIX, split)
    checkpoint = registry.state_arrays()
    assert checkpoint and all(
        key.startswith(RESUME_PREFIX) for key in checkpoint
    ), "non-event controller remains legacy-compatible and does not invent a manifest"

    resumed = FilmPolarChartSPELState()
    restore_registry = ResumeRegistry()
    restore_registry.register("film_polar_chart_spel", RESUME_PREFIX, resumed)
    report = restore_registry.restore(checkpoint)
    assert report.restored == {"film_polar_chart_spel": True}
    for grad in gradients[3:]:
        resumed.step_numpy(grad, learning_rate=2e-3, ema_decay=0.997)

    assert resumed.step == uninterrupted.step == 6
    assert resumed.source_weight_sha256 == uninterrupted.source_weight_sha256
    for name in ("q", "h0", "momentum", "q_ema"):
        assert np.array_equal(getattr(resumed, name), getattr(uninterrupted, name))
    assert np.array_equal(resumed.deploy_weight_numpy(), uninterrupted.deploy_weight_numpy())
    assert np.array_equal(resumed.live_weight_numpy(), uninterrupted.live_weight_numpy())


def test_resume_payload_rejects_typed_config_drift():
    state = FilmPolarChartSPELState()
    state.initialize_numpy(_full_rank_weight())
    payload = state.state_arrays(RESUME_PREFIX)
    payload[RESUME_PREFIX + "ns_steps"] = np.asarray(4, dtype=np.int64)
    with pytest.raises(ValueError, match="ns_steps differs"):
        FilmPolarChartSPELState(ns_steps=5).restore_from_cfg(RESUME_PREFIX, payload)


def test_outgoing_adam_first_moment_pulls_back_to_tangent():
    state = FilmPolarChartSPELState()
    state.initialize_numpy(_full_rank_weight())
    seeded = state.warm_start_momentum_numpy(_grad())
    q = np.asarray(state.q)
    residual = np.linalg.norm(q.T @ seeded + seeded.T @ q, ord="fro")
    assert residual < 2e-5
    assert np.linalg.norm(seeded) > 0.0


def test_dsl_lever_is_real_default_off_composable_and_duty_registered(tmp_path):
    lever = cd.FilmPolarChartSPELManifoldMuon()
    assert lever.overrides == {"--film-polar-chart-spel": True}
    assert lever.epochs_delta == 0
    assert "FilmPolarChartSPELManifoldMuon" in lever_registry.name_composable_levers()
    assert "--film-polar-chart-spel" not in lever_registry.completeness().unmapped
    assert "film_polar_chart_spel" in DIRECT_CONTROLLER_NAMES

    from tac.witness_dsl.activation_ledger import duty_to_measure, known_levers

    empty = tmp_path / "activation_ledger.jsonl"
    assert "FilmPolarChartSPELManifoldMuon" in known_levers()
    assert "FilmPolarChartSPELManifoldMuon" in duty_to_measure(path=empty)
    assert cd.FilmPolarChartSPELManifoldMuon(start_epoch=276).overrides == {
        "--film-polar-chart-spel": True,
        "--muon-start-epoch": 276,
    }
    with pytest.raises(ValueError, match="start_epoch must be >= 1"):
        cd.FilmPolarChartSPELManifoldMuon(start_epoch=0)


def test_real_trainer_parser_accepts_compiled_film_polar_flag():
    parser = cd.build_real_trainer_parser()
    parsed = parser.parse_args(["--out-dir", "unused", "--film-polar-chart-spel"])
    assert parsed.film_polar_chart_spel is True
    assert parser.parse_args(["--out-dir", "unused"]).film_polar_chart_spel is False


def test_reference_adamw_semantics_is_default_off_typed_and_threads_finisher():
    lever = cd.AdamWReferenceSemantics()
    assert lever.overrides == {"--adamw-reference-semantics": True}
    assert "AdamWReferenceSemantics" in lever_registry.name_composable_levers()
    assert "--adamw-reference-semantics" not in lever_registry.completeness().unmapped
    parser = cd.build_real_trainer_parser()
    assert parser.parse_args(["--out-dir", "unused"]).adamw_reference_semantics is False
    assert parser.parse_args(
        ["--out-dir", "unused", "--adamw-reference-semantics"]
    ).adamw_reference_semantics is True
    assert _adamw_bias_correction_for(0.999) is False
    assert _adamw_bias_correction_for(0.999, reference_semantics=True) is True
    assert _adamw_bias_correction_for(0.9999999) is True


def test_checkpoint_boundary_control_is_a_typed_muon_factory():
    lever = cd.MuonAtCheckpointBoundary(276, window=2)
    assert lever.overrides == {"--muon-start-epoch": 276}
    assert lever.epochs_delta == 2
    with pytest.raises(ValueError, match="start_epoch must be >= 1"):
        cd.MuonAtCheckpointBoundary(0)


def test_mlx_step_matches_numpy_fp32_reference_when_device_is_available():
    """Parity authority is NumPy; a headless no-Metal host records an honest skip."""

    try:
        import mlx.core as mx

        weight = _full_rank_weight(shape=(17, 5))
        q, h0 = polar_chart_numpy(weight)
        momentum = np.zeros_like(q)
        grad = _grad(shape=q.shape)
        expected = spel_step_numpy(
            q,
            h0,
            momentum,
            q,
            grad,
            learning_rate=2e-3,
            ema_decay=0.997,
        )
        actual = spel_step_mlx_arrays(
            mx.array(q),
            mx.array(h0),
            mx.array(momentum),
            mx.array(q),
            mx.array(grad),
            learning_rate=2e-3,
            ema_decay=0.997,
        )
        mx.eval(*actual)
    except RuntimeError as exc:
        if "No Metal device available" in str(exc):
            pytest.skip(json.dumps({"authority": "BLOCKED", "reason": str(exc)}))
        raise

    for got, want in zip(
        actual,
        (expected.q, expected.momentum, expected.q_ema, expected.weight),
        strict=True,
    ):
        np.testing.assert_allclose(np.asarray(got), want, rtol=3e-5, atol=3e-6)
