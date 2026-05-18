# SPDX-License-Identifier: MIT
"""Tests for BUCKET C (autopilot consumption of sister #817 per-pair sidecars).

LOW gap closure widened wave 2026-05-17 — `lane_low_gap_closure_widened_8_modules_
plus_autopilot_wire_in_20260517` BUCKET C. Exercises the
``adjust_predicted_delta_for_per_pair_sister_817_sidecars`` helper + its
integration into ``apply_z1_empirical_revision_to_candidate_delta``.

Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag" + sister Q2+Q3
v2 cascade discipline: sidecar ABSENT → 1.0× passthrough (no fake reward);
sidecar PRESENT → conservative multiplicative bonus per the BUCKET C reward
bands.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_autopilot_module():
    """Load the autopilot loop as a module so we can call its internal helpers."""
    spec = importlib.util.spec_from_file_location(
        "autopilot_loop",
        REPO_ROOT / "tools" / "cathedral_autopilot_autonomous_loop.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("autopilot_loop", mod)
    spec.loader.exec_module(mod)
    return mod


def _write_per_pair_bit_allocation_sidecar(
    root: Path, sha: str, *, cascade: str
) -> Path:
    """Stage a synthetic per_pair_bit_allocation sidecar at canonical path."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"per_pair_bit_allocation_{sha[:12]}_20260517T220000.json"
    path.write_text(
        json.dumps(
            {
                "schema": "tac_bit_allocator_per_pair_v1",
                "archive_sha256": sha,
                "cascade_path_used": cascade,
                "per_byte_bit_allocation": {0: 10, 1: 20},
                "total_allocated_bytes": 30,
                "n_bytes_classified_shared_prior": 0,
                "n_bytes_classified_pair_specific": 2,
                "n_bytes_classified_mixed": 0,
                "optimal_plan_consumed": cascade == "optimal_plan",
                "sensitivity_reweight_consumed": cascade == "wyner_ziv_composition",
                "per_pair_gradient_consumed": cascade == "wyner_ziv_composition",
                "total_bit_budget": 100,
                "rationale": "synthetic test fixture",
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "evidence_grade": "[predicted; bit allocator per-pair v1]",
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_per_pair_fisher_importance_sidecar(
    root: Path, sha: str, *, aggregate_fisher_l1: float
) -> Path:
    """Stage a synthetic per_pair_fisher_importance sidecar at canonical path."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"per_pair_fisher_importance_{sha[:12]}_20260517T220100.json"
    path.write_text(
        json.dumps(
            {
                "schema": "tac_jacobian_fisher_importance_allocator_per_pair_v1",
                "archive_sha256": sha,
                "per_byte_fisher_importance": {0: 0.5, 1: 0.8},
                "n_bytes": 2,
                "n_pairs": 4,
                "aggregate_fisher_l1": aggregate_fisher_l1,
                "aggregate_fisher_l2": 0.95,
                "top_k_byte_indices_by_importance": [1, 0],
                "bottom_k_byte_indices_by_importance": [0, 1],
                "catalog_123_invariant_satisfied": True,
                "per_pair_gradient_consumed": True,
                "rationale": "synthetic test fixture",
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "evidence_grade": (
                    "[predicted; jacobian-fisher per-pair v1; "
                    "score-gradient-derived NOT weight-domain]"
                ),
            }
        ),
        encoding="utf-8",
    )
    return path


# ── Helper unit tests ────────────────────────────────────────────────────────


def test_sister_817_passthrough_when_no_sidecars(tmp_path, monkeypatch):
    """No sidecars present → 1.0× passthrough (NO FAKE REWARD)."""
    mod = _load_autopilot_module()
    monkeypatch.setattr(mod, "_PER_PAIR_SIDECAR_SCAN_ROOT", tmp_path)
    out = mod.adjust_predicted_delta_for_per_pair_sister_817_sidecars(
        -0.05, "deadbeef1234567890abcdef"
    )
    assert abs(out - (-0.05)) < 1e-12


def test_sister_817_empty_sha_returns_passthrough():
    mod = _load_autopilot_module()
    out = mod.adjust_predicted_delta_for_per_pair_sister_817_sidecars(
        -0.05, ""
    )
    assert abs(out - (-0.05)) < 1e-12


def test_sister_817_optimal_plan_sidecar_reward_factor(tmp_path, monkeypatch):
    """per_pair_bit_allocation cascade=optimal_plan → 0.95× reward."""
    mod = _load_autopilot_module()
    monkeypatch.setattr(mod, "_PER_PAIR_SIDECAR_SCAN_ROOT", tmp_path)
    sha = "deadbeef1234567890abcdef"
    _write_per_pair_bit_allocation_sidecar(tmp_path, sha, cascade="optimal_plan")
    factor = mod._per_pair_bit_allocation_sidecar_reward_factor(sha)
    assert factor == 0.95


def test_sister_817_wyner_ziv_sidecar_reward_factor(tmp_path, monkeypatch):
    """per_pair_bit_allocation cascade=wyner_ziv_composition → 0.98× reward."""
    mod = _load_autopilot_module()
    monkeypatch.setattr(mod, "_PER_PAIR_SIDECAR_SCAN_ROOT", tmp_path)
    sha = "deadbeef1234567890abcdef"
    _write_per_pair_bit_allocation_sidecar(
        tmp_path, sha, cascade="wyner_ziv_composition"
    )
    factor = mod._per_pair_bit_allocation_sidecar_reward_factor(sha)
    assert factor == 0.98


def test_sister_817_aggregate_fallback_sidecar_reward_factor(tmp_path, monkeypatch):
    """per_pair_bit_allocation cascade=aggregate_fallback → 1.0× (NO REWARD)."""
    mod = _load_autopilot_module()
    monkeypatch.setattr(mod, "_PER_PAIR_SIDECAR_SCAN_ROOT", tmp_path)
    sha = "deadbeef1234567890abcdef"
    _write_per_pair_bit_allocation_sidecar(
        tmp_path, sha, cascade="aggregate_fallback"
    )
    factor = mod._per_pair_bit_allocation_sidecar_reward_factor(sha)
    assert factor == 1.0


def test_sister_817_fisher_sidecar_reward_factor_present(tmp_path, monkeypatch):
    """per_pair_fisher_importance sidecar present + agg_l1>0 → 0.97× reward."""
    mod = _load_autopilot_module()
    monkeypatch.setattr(mod, "_PER_PAIR_SIDECAR_SCAN_ROOT", tmp_path)
    sha = "deadbeef1234567890abcdef"
    _write_per_pair_fisher_importance_sidecar(
        tmp_path, sha, aggregate_fisher_l1=1.5
    )
    factor = mod._per_pair_fisher_importance_sidecar_reward_factor(sha)
    assert factor == 0.97


def test_sister_817_fisher_sidecar_zero_aggregate_no_reward(tmp_path, monkeypatch):
    """per_pair_fisher_importance with agg_l1==0 → 1.0× (NO FAKE REWARD)."""
    mod = _load_autopilot_module()
    monkeypatch.setattr(mod, "_PER_PAIR_SIDECAR_SCAN_ROOT", tmp_path)
    sha = "deadbeef1234567890abcdef"
    _write_per_pair_fisher_importance_sidecar(
        tmp_path, sha, aggregate_fisher_l1=0.0
    )
    factor = mod._per_pair_fisher_importance_sidecar_reward_factor(sha)
    assert factor == 1.0


def test_sister_817_combined_optimal_plan_plus_fisher(tmp_path, monkeypatch):
    """BOTH sidecars present at best path → composes multiplicatively."""
    mod = _load_autopilot_module()
    monkeypatch.setattr(mod, "_PER_PAIR_SIDECAR_SCAN_ROOT", tmp_path)
    sha = "deadbeef1234567890abcdef"
    _write_per_pair_bit_allocation_sidecar(tmp_path, sha, cascade="optimal_plan")
    _write_per_pair_fisher_importance_sidecar(
        tmp_path, sha, aggregate_fisher_l1=1.5
    )
    base_delta = -0.05
    out = mod.adjust_predicted_delta_for_per_pair_sister_817_sidecars(
        base_delta, sha
    )
    # Expected: -0.05 × 0.95 × 0.97 = -0.046075
    expected = -0.05 * 0.95 * 0.97
    assert abs(out - expected) < 1e-12


def test_sister_817_malformed_sidecar_returns_passthrough(tmp_path, monkeypatch):
    """A malformed sidecar JSON should NOT crash; returns passthrough."""
    mod = _load_autopilot_module()
    monkeypatch.setattr(mod, "_PER_PAIR_SIDECAR_SCAN_ROOT", tmp_path)
    sha = "deadbeef1234567890abcdef"
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / f"per_pair_bit_allocation_{sha[:12]}_bad.json").write_text(
        "not valid json {{{",
        encoding="utf-8",
    )
    factor = mod._per_pair_bit_allocation_sidecar_reward_factor(sha)
    assert factor == 1.0  # safe passthrough


def test_sister_817_constants_pinned():
    """Reward constants are pinned at the documented values."""
    mod = _load_autopilot_module()
    assert mod._PER_PAIR_BIT_ALLOCATION_SIDECAR_REWARD_OPTIMAL_PLAN == 0.95
    assert mod._PER_PAIR_BIT_ALLOCATION_SIDECAR_REWARD_WYNER_ZIV == 0.98
    assert mod._PER_PAIR_BIT_ALLOCATION_SIDECAR_REWARD_AGGREGATE_FALLBACK == 1.0
    assert mod._PER_PAIR_FISHER_IMPORTANCE_SIDECAR_REWARD_PRESENT == 0.97
    assert mod._PER_PAIR_FISHER_IMPORTANCE_SIDECAR_REWARD_ABSENT == 1.0


# ── Integration: apply_z1_empirical_revision composes new factor ─────────────


def test_apply_z1_empirical_revision_composes_sister_817_factor(
    tmp_path, monkeypatch
):
    """The BUCKET C wire-in is composed multiplicatively at the END of the
    chain (after venn_v2). Verify by staging an optimal_plan sidecar and
    checking that apply_z1_empirical_revision applies the 0.95× factor."""
    mod = _load_autopilot_module()
    monkeypatch.setattr(mod, "_PER_PAIR_SIDECAR_SCAN_ROOT", tmp_path)
    # Also disable venn classification sidecar so v2 cascade falls through to
    # passthrough (1.0×). The new BUCKET C wire-in then applies its 0.95×.
    monkeypatch.setattr(
        mod, "_VENN_CLASSIFICATION_SIDECAR_ROOT", tmp_path / "venn_no_op"
    )
    sha = "deadbeef1234567890abcdef"
    _write_per_pair_bit_allocation_sidecar(tmp_path, sha, cascade="optimal_plan")

    # Build a minimal CandidateRow stand-in. The canonical CandidateRow
    # dataclass has many fields; we use a SimpleNamespace shim that supplies
    # the fields apply_z1_empirical_revision touches.
    from types import SimpleNamespace

    c = SimpleNamespace(
        predicted_score_delta=-0.10,
        mdl_density=None,
        mdl_tier_c_density=None,
        lane_class=None,
        literature_anchor="",
        notes="",
        composition_alpha=None,
        predicted_dispatch_risk=None,
        archive_sha256=sha,
    )
    # Mock _candidate_literature_anchor_rank_reward_suppressed to return False
    monkeypatch.setattr(
        mod, "_candidate_literature_anchor_rank_reward_suppressed",
        lambda _c: False,
    )
    d = mod.apply_z1_empirical_revision_to_candidate_delta(c)
    # Expected: -0.10 (base) → -0.10 (no mdl) → -0.10 (no tier C) → -0.10
    # (no class shift) → -0.10 (no composition alpha) → -0.10 (no risk) →
    # -0.10 (venn passthrough; no sidecar) → -0.10 × 0.95 = -0.095
    expected = -0.10 * 0.95
    assert abs(d - expected) < 1e-9


def test_apply_z1_empirical_revision_preserves_v2_replace_semantics(
    tmp_path, monkeypatch
):
    """When the v2 venn cascade has REPLACE semantics (OptimalPlan present),
    the BUCKET C wire-in applies AFTER on the replaced delta — verifying the
    spec's 'ADDITIONAL factor, NOT a replacement' rule."""
    mod = _load_autopilot_module()
    monkeypatch.setattr(mod, "_PER_PAIR_SIDECAR_SCAN_ROOT", tmp_path)
    monkeypatch.setattr(mod, "_VENN_CLASSIFICATION_SIDECAR_ROOT", tmp_path)
    sha = "deadbeef1234567890abcdef"
    # The v2 cascade calls tac.master_gradient_consumers.load_optimal_plan_for_archive
    # which scans the CANONICAL CONSUMER_OUTPUT_ROOT. We monkeypatch the
    # function to simulate a plan being found.
    import tac.master_gradient_consumers as mgc

    def fake_load_optimal_plan(archive_sha256, *, root=None):
        return {
            "schema": "master_gradient_consumer_optimal_per_pair_treatment_plan_v1",
            "consumer_id": "per_pair_optimal_treatment_plan_via_lagrangian_dual",
            "archive_sha256": archive_sha256,
            "predicted_score_delta": -0.07,
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }

    monkeypatch.setattr(mgc, "load_optimal_plan_for_archive", fake_load_optimal_plan)
    # Also stage BUCKET C sidecar
    _write_per_pair_bit_allocation_sidecar(tmp_path, sha, cascade="optimal_plan")
    from types import SimpleNamespace

    c = SimpleNamespace(
        predicted_score_delta=-0.10,  # base value (REPLACED by v2 → -0.07)
        mdl_density=None,
        mdl_tier_c_density=None,
        lane_class=None,
        literature_anchor="",
        notes="",
        composition_alpha=None,
        predicted_dispatch_risk=None,
        archive_sha256=sha,
    )
    monkeypatch.setattr(
        mod, "_candidate_literature_anchor_rank_reward_suppressed",
        lambda _c: False,
    )
    d = mod.apply_z1_empirical_revision_to_candidate_delta(c)
    # Expected: v2 REPLACES base -0.10 with -0.07 (planner-derived); then
    # BUCKET C applies 0.95× → -0.07 × 0.95 = -0.0665
    expected = -0.07 * 0.95
    assert abs(d - expected) < 1e-9
