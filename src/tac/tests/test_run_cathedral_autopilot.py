"""Tests for ``experiments/run_cathedral_autopilot``."""

from __future__ import annotations

import importlib.util
import pathlib



def _load_autopilot_module():
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    path = repo_root / "experiments" / "run_cathedral_autopilot.py"
    spec = importlib.util.spec_from_file_location("run_cathedral_autopilot", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_synthetic_substrate_has_correct_schema() -> None:
    mod = _load_autopilot_module()
    sd = mod.synthetic_substrate(seed=0)
    from tac.pr101_split_brotli_codec import FIXED_STATE_SCHEMA
    assert len(sd) == len(FIXED_STATE_SCHEMA)
    for name, shape in FIXED_STATE_SCHEMA:
        assert name in sd
        assert tuple(sd[name].shape) == shape
        assert sd[name].requires_grad


def test_total_rate_h0_returns_positive_scalar() -> None:
    mod = _load_autopilot_module()
    sd = mod.synthetic_substrate(seed=0)
    rate = mod.total_rate_h0(sd)
    assert rate.ndim == 0
    assert float(rate) > 0


def test_solve_inner_loose_budget_converges() -> None:
    """At a loose rate budget, the solver should hit KKT convergence quickly."""
    mod = _load_autopilot_module()
    sd = mod.synthetic_substrate(seed=0)
    ref = mod.reference_substrate(seed=1)
    pt = mod.solve_rate_constrained_inner(
        sd, ref, R_target_bytes=200_000, n_steps=20,
    )
    # At loose budget the rate is unconstrained; converged should be True
    # because primal violation is ~0 throughout.
    assert pt.converged
    assert pt.rate_bytes < 200_000 + 1.0  # within numerical slack


def test_solve_inner_tight_budget_pushes_lambda_up() -> None:
    """At a very tight budget, λ should grow above 0 to push rate down."""
    mod = _load_autopilot_module()
    sd = mod.synthetic_substrate(seed=0)
    ref = mod.reference_substrate(seed=1)
    pt = mod.solve_rate_constrained_inner(
        sd, ref, R_target_bytes=10_000, n_steps=20, dual_lr=1e-2,
    )
    # Tight budget; λ should have ramped up.
    assert pt.lambda_rate > 0


def test_trace_pareto_frontier_returns_n_points() -> None:
    mod = _load_autopilot_module()
    targets = [50_000, 100_000, 150_000]
    points = mod.trace_pareto_frontier(R_targets=targets, n_inner_steps=10)
    assert len(points) == 3
    for pt, target in zip(points, targets):
        assert pt.R_target_bytes == target


def test_summarize_frontier_returns_markdown_table() -> None:
    mod = _load_autopilot_module()
    points = mod.trace_pareto_frontier(R_targets=[100_000], n_inner_steps=5)
    md = mod.summarize_frontier(points)
    assert "| R_target" in md
    assert "100,000" in md
