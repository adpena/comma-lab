# SPDX-License-Identifier: MIT
"""Tests for ``tools/pr101_omega_opt_joint_admm_allocation_empirical.py``.

Verifies:

- λ-bisection terminates within the declared iteration budget for
  reasonable rms_target ranges.
- ``lagrangian_allocate`` is monotone in λ: larger λ -> lower achieved
  RMS rel_err (or equal, if at the floor).
- At the boundary λ -> 0, allocation matches the "highest-sparsity per
  tensor" greedy choice.
- At λ -> ∞, allocation matches the lossless (rel_err = 0) choice.
- ``bisect_lambda_for_rms_target`` returns a manifest whose RMS satisfies
  the target (or the closest achievable bound).
- The encoded result is byte-deterministic for fixed input curves.

These tests use synthetic per-tensor cost curves (no torch state_dict
load) so they are fast and deterministic on CPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import pr101_omega_opt_joint_admm_allocation_empirical as mod  # noqa: E402


def _synthetic_curves() -> list[list[dict]]:
    """3 tensors, 5 codec choices each. Designed so:
    - tensor 0: cheap-distortion (sparsity scales bytes 100->50 gracefully)
    - tensor 1: expensive-distortion (any sparsity collapses)
    - tensor 2: middling
    """
    return [
        # tensor 0: cheap distortion
        [
            {"alpha": 0.0, "bytes": 100, "rel_err": 0.0},
            {"alpha": 0.3, "bytes": 80, "rel_err": 0.05},
            {"alpha": 0.5, "bytes": 65, "rel_err": 0.10},
            {"alpha": 0.7, "bytes": 55, "rel_err": 0.20},
            {"alpha": 0.9, "bytes": 50, "rel_err": 0.40},
        ],
        # tensor 1: expensive distortion (steep cost curve)
        [
            {"alpha": 0.0, "bytes": 200, "rel_err": 0.0},
            {"alpha": 0.3, "bytes": 195, "rel_err": 0.30},
            {"alpha": 0.5, "bytes": 190, "rel_err": 0.50},
            {"alpha": 0.7, "bytes": 185, "rel_err": 0.70},
            {"alpha": 0.9, "bytes": 180, "rel_err": 0.90},
        ],
        # tensor 2: middling
        [
            {"alpha": 0.0, "bytes": 150, "rel_err": 0.0},
            {"alpha": 0.3, "bytes": 130, "rel_err": 0.10},
            {"alpha": 0.5, "bytes": 115, "rel_err": 0.20},
            {"alpha": 0.7, "bytes": 105, "rel_err": 0.35},
            {"alpha": 0.9, "bytes": 100, "rel_err": 0.50},
        ],
    ]


def test_lagrangian_allocate_lambda_zero_picks_minimum_bytes() -> None:
    """λ -> 0: each tensor picks the lowest-bytes codec ignoring rel_err."""
    curves = _synthetic_curves()
    bytes_total, rms, rel_errs = mod.lagrangian_allocate(curves, lam=0.0)
    # Lowest-bytes per tensor: 50 + 180 + 100 = 330
    assert bytes_total == 50 + 180 + 100
    # Achieved rel_errs should be the per-tensor maxima
    assert rel_errs == [0.40, 0.90, 0.50]
    assert rms > 0.0


def test_lagrangian_allocate_lambda_infinity_picks_lossless() -> None:
    """λ -> ∞: each tensor picks rel_err=0 codec (lossless brotli)."""
    curves = _synthetic_curves()
    bytes_total, rms, rel_errs = mod.lagrangian_allocate(curves, lam=1e15)
    # Lossless per tensor: 100 + 200 + 150 = 450
    assert bytes_total == 100 + 200 + 150
    assert rel_errs == [0.0, 0.0, 0.0]
    assert rms == 0.0


def test_lagrangian_allocate_monotone_in_lambda() -> None:
    """Larger λ -> RMS rel_err is non-increasing."""
    curves = _synthetic_curves()
    prev_rms = float("inf")
    for lam in [0.0, 1.0, 10.0, 100.0, 1000.0, 1e6, 1e12]:
        _, rms, _ = mod.lagrangian_allocate(curves, lam=lam)
        assert rms <= prev_rms + 1e-9, (
            f"non-monotone: λ={lam} rms={rms} > prev_rms={prev_rms}"
        )
        prev_rms = rms


def test_bisect_lambda_for_rms_target_satisfies_constraint() -> None:
    """Bisect must return a result whose RMS <= target (or be at the floor)."""
    curves = _synthetic_curves()
    target = 0.10
    result = mod.bisect_lambda_for_rms_target(curves, rms_target=target)
    # Either constraint satisfied OR at lossless floor
    assert (
        result["achieved_rms_rel_err"] <= target + 1e-6
        or result["achieved_rms_rel_err"] == 0.0
    ), result


def test_bisect_lambda_terminates_within_budget() -> None:
    """The declared 60-iteration cap must be sufficient for reasonable
    targets. Indirectly tested by ensuring the function returns a finite
    λ even at extreme targets."""
    curves = _synthetic_curves()
    for target in [0.001, 0.01, 0.1, 0.3, 0.6, 1.0]:
        result = mod.bisect_lambda_for_rms_target(curves, rms_target=target)
        assert isinstance(result["lambda"], float)
        # λ must be finite (no inf) — converged within budget
        assert result["lambda"] < 1e15 or result["achieved_rms_rel_err"] == 0.0


def test_bisect_lambda_deterministic_for_fixed_curves() -> None:
    """Same curves + same target -> same λ + same allocation bytes."""
    curves = _synthetic_curves()
    r1 = mod.bisect_lambda_for_rms_target(curves, rms_target=0.20)
    r2 = mod.bisect_lambda_for_rms_target(curves, rms_target=0.20)
    assert r1["lambda"] == r2["lambda"]
    assert r1["total_bytes"] == r2["total_bytes"]
    assert r1["achieved_rms_rel_err"] == r2["achieved_rms_rel_err"]


def test_greedy_uniform_budget_gets_most_aggressive_admissible() -> None:
    """Each tensor picks lowest-bytes codec with rel_err <= budget."""
    curves = _synthetic_curves()
    bytes_total, rms = mod.greedy_uniform_budget(curves, budget=0.20)
    # tensor 0: alpha=0.7 (rel_err=0.20, bytes=55)
    # tensor 1: alpha=0.0 (only 0.0<=0.20, bytes=200)
    # tensor 2: alpha=0.5 (rel_err=0.20, bytes=115)
    assert bytes_total == 55 + 200 + 115
    assert rms > 0.0


def test_admm_beats_or_matches_greedy_at_same_target() -> None:
    """Joint-ADMM must NOT be worse than greedy at the same RMS target.

    This is the contract: ADMM's degree of freedom (per-tensor distortion
    allocation) is a strict superset of greedy's uniform per-tensor budget,
    so it must achieve >= bytes saving (or tie).
    """
    curves = _synthetic_curves()
    for target in [0.10, 0.20, 0.30, 0.50]:
        g_bytes, _ = mod.greedy_uniform_budget(curves, budget=target)
        admm = mod.bisect_lambda_for_rms_target(curves, rms_target=target)
        # ADMM achieves RMS <= target; bytes should be <= greedy's for same
        # RMS bound. (Greedy's RMS may be different but is bounded by target.)
        if admm["achieved_rms_rel_err"] <= target + 1e-6:
            assert admm["total_bytes"] <= g_bytes, (
                f"ADMM ({admm['total_bytes']} B) > greedy ({g_bytes} B) at "
                f"target={target} (admm_rms={admm['achieved_rms_rel_err']:.4f})"
            )


def test_falsification_scope_set_in_manifest() -> None:
    """Per CLAUDE.md ``forbidden_premature_class_level_falsification``, the
    manifest must declare a non-empty ``falsification_scope`` when
    ``family_falsified=False``. This guards the tool's evidence-emission
    contract."""
    # Inspect module-level constants embedded in the manifest dict the tool
    # builds — the run_experiment() function returns these directly.
    # We import without running torch.load by checking the source for
    # the literal strings.
    src = (REPO_ROOT / "tools" / "pr101_omega_opt_joint_admm_allocation_empirical.py").read_text()
    assert '"family_falsified": False' in src
    assert '"falsification_scope":' in src
    # The scope string must mention the tested mechanism
    assert "lagrangian_allocation" in src
