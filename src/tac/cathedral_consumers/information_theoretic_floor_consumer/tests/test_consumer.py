# SPDX-License-Identifier: MIT
"""Tests for information_theoretic_floor_consumer Tier B promotion.

Per WAVE-3-DIM-6-STEP-6.5 RESUME-2 2026-05-20. Covers:

- Canonical contract compliance per Catalog #335 (CONSUMER_NAME /
  CONSUMER_VERSION / CONSUMER_TIER / CONSUMER_HOOK_NUMBERS /
  update_from_anchor / consume_candidate).
- CONSUMER_TIER == TIER_B_SCORE_CONTRIBUTING declared.
- Tier A baseline branch returns 0.0 + AxisDecomposition (Catalog #341
  observability-only invariants preserved).
- Tier B solver branch returns NON-ZERO delta within ``[-0.05, 0.05]``
  safety rail per CATHEDRAL-SMARTER-DESIGN-MEMO Dim 6 Step 6.5 + sister
  META-LAGRANGIAN-WIRE-1 Phase 1 bounded 5% band.
- Tier B contribution validates against canonical
  ``validate_tier_b_contribution`` (Catalog #357 runtime contract).
- Paired-comparison mode (default) emits BOTH branches and appends a
  paired row to the fcntl-locked APPEND-ONLY canonical JSONL store.
- ``CONSUMER_TIER_B_MODE`` env honored (tier_a_baseline /
  tier_b_solver / paired_comparison).
- Catalog #335 / #341 / #356 / #357 gates remain at 0 violations.
- Live-repo regression guard.
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from tac.cathedral.consumer_contract import (
    ConsumerTier,
    HookNumber,
    validate_consumer_module,
    validate_tier_b_contribution,
)


MODULE_PATH = "tac.cathedral_consumers.information_theoretic_floor_consumer"


def _load_module() -> Any:
    return importlib.import_module(MODULE_PATH)


# ---------------------------------------------------------------------------
# Canonical contract compliance per Catalog #335
# ---------------------------------------------------------------------------


def test_consumer_satisfies_canonical_contract() -> None:
    mod = _load_module()
    res = validate_consumer_module(mod, module_path=MODULE_PATH)
    assert res.contract_compliant, f"validation errors: {res.validation_errors}"


def test_consumer_declares_canonical_metadata() -> None:
    mod = _load_module()
    assert mod.CONSUMER_NAME == "information_theoretic_floor_consumer"
    assert mod.CONSUMER_VERSION == "2.0.0"
    assert isinstance(mod.CONSUMER_HOOK_NUMBERS, tuple)
    assert len(mod.CONSUMER_HOOK_NUMBERS) >= 1
    for hook in mod.CONSUMER_HOOK_NUMBERS:
        assert isinstance(hook, HookNumber)


def test_consumer_declares_tier_b_score_contributing() -> None:
    mod = _load_module()
    assert mod.CONSUMER_TIER == ConsumerTier.TIER_B_SCORE_CONTRIBUTING


def test_hook_numbers_match_125_declaration() -> None:
    """Per Catalog #125: hooks #1 + #2 + #4 + #5 + #6 ACTIVE."""
    mod = _load_module()
    assert HookNumber.SENSITIVITY_MAP in mod.CONSUMER_HOOK_NUMBERS
    assert HookNumber.PARETO_CONSTRAINT in mod.CONSUMER_HOOK_NUMBERS
    assert HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH in mod.CONSUMER_HOOK_NUMBERS
    assert HookNumber.CONTINUAL_LEARNING_POSTERIOR in mod.CONSUMER_HOOK_NUMBERS
    assert HookNumber.PROBE_DISAMBIGUATOR in mod.CONSUMER_HOOK_NUMBERS


# ---------------------------------------------------------------------------
# Tier A baseline branch: Catalog #341 observability-only invariants
# ---------------------------------------------------------------------------


def test_tier_a_baseline_returns_zero_delta_adjustment() -> None:
    mod = _load_module()
    cand = {
        "information_theoretic_floor": 0.05,
        "floor_estimate_mode": "cramer_rao",
        "m_contest_array_sha256": "a" * 64,
        "current_best_empirical_score": 0.2,
    }
    res = mod.consume_candidate_tier_a_baseline(cand)
    assert res["predicted_delta_adjustment"] == 0.0
    assert res["promotable"] is False
    assert res["axis_tag"] == "[predicted]"
    assert res["consumer_branch_kind"] == "tier_a_baseline"


def test_tier_a_baseline_returns_axis_decomposition() -> None:
    mod = _load_module()
    cand = {
        "information_theoretic_floor": 0.05,
        "floor_estimate_mode": "cramer_rao",
        "m_contest_array_sha256": "b" * 64,
        "per_axis_floor": {"seg": -0.001, "pose": -0.0005, "rate_bytes": -42},
    }
    res = mod.consume_candidate_tier_a_baseline(cand)
    decomp = res["predicted_axis_decomposition"]
    assert "predicted_d_seg_delta" in decomp
    assert "predicted_d_pose_delta" in decomp
    assert "predicted_archive_bytes_delta" in decomp
    assert "axis_tag" in decomp
    assert "canonical_provenance" in decomp
    assert decomp["predicted_d_seg_delta"] == -0.001
    assert decomp["predicted_d_pose_delta"] == -0.0005
    assert decomp["predicted_archive_bytes_delta"] == -42


# ---------------------------------------------------------------------------
# Tier B solver branch: bounded non-zero delta + canonical Catalog #357
# contract per validate_tier_b_contribution
# ---------------------------------------------------------------------------


def test_tier_b_solver_returns_non_zero_within_safety_rail() -> None:
    mod = _load_module()
    cand = {
        "information_theoretic_floor": 0.05,
        "floor_estimate_mode": "cramer_rao",
        "m_contest_array_sha256": "c" * 64,
        "per_axis_floor": {"seg": -0.0001, "pose": -0.001, "rate_bytes": -100},
    }
    res = mod.consume_candidate_solver_derived(cand)
    delta = res["predicted_delta_adjustment"]
    # NON-ZERO ranking signal
    assert delta != 0.0
    # Safety rail
    assert -0.05 <= delta <= 0.05
    assert res["consumer_branch_kind"] == "tier_b_solver"


def test_tier_b_solver_clipped_to_safety_rail_min() -> None:
    """Aggressively-negative per-axis floor saturates at -0.05."""
    mod = _load_module()
    cand = {
        "information_theoretic_floor": 0.05,
        "floor_estimate_mode": "cramer_rao",
        "m_contest_array_sha256": "d" * 64,
        "per_axis_floor": {"seg": -1.0, "pose": -1.0, "rate_bytes": -1_000_000},
    }
    res = mod.consume_candidate_solver_derived(cand)
    assert res["predicted_delta_adjustment"] == -0.05
    assert "clipped_to_safety_rail_min" in res["tier_b_derivation_note"]


def test_tier_b_solver_clipped_to_safety_rail_max() -> None:
    """Aggressively-positive per-axis floor saturates at +0.05."""
    mod = _load_module()
    cand = {
        "information_theoretic_floor": 0.05,
        "floor_estimate_mode": "cramer_rao",
        "m_contest_array_sha256": "e" * 64,
        "per_axis_floor": {"seg": 1.0, "pose": 1.0, "rate_bytes": 1_000_000},
    }
    res = mod.consume_candidate_solver_derived(cand)
    assert res["predicted_delta_adjustment"] == 0.05
    assert "clipped_to_safety_rail_max" in res["tier_b_derivation_note"]


def test_tier_b_solver_uses_diagnostic_cuda_axis_tag() -> None:
    """Catalog #357 requires empirically-grounded axis_tag (NOT [predicted])."""
    mod = _load_module()
    cand = {
        "information_theoretic_floor": 0.05,
        "floor_estimate_mode": "cramer_rao",
        "m_contest_array_sha256": "f" * 64,
        "per_axis_floor": {"seg": -0.0001, "pose": -0.0001, "rate_bytes": -10},
    }
    res = mod.consume_candidate_solver_derived(cand)
    assert res["axis_tag"] == "[diagnostic-CUDA]"
    assert res["axis_tag"] != "[predicted]"


def test_tier_b_solver_preserves_promotable_false() -> None:
    """Catalog #357 requires promotable=False (paired-axis evidence
    required for promotion per CLAUDE.md 'Submission auth eval')."""
    mod = _load_module()
    cand = {
        "information_theoretic_floor": 0.05,
        "floor_estimate_mode": "cramer_rao",
        "m_contest_array_sha256": "g" * 64,
        "per_axis_floor": {"seg": -0.0001, "pose": -0.0001, "rate_bytes": -10},
    }
    res = mod.consume_candidate_solver_derived(cand)
    assert res["promotable"] is False


def test_tier_b_solver_threads_canonical_provenance() -> None:
    """Catalog #323/#357: provenance field present + non-empty Mapping."""
    mod = _load_module()
    cand = {
        "information_theoretic_floor": 0.05,
        "floor_estimate_mode": "cramer_rao",
        "m_contest_array_sha256": "h" * 64,
        "per_axis_floor": {"seg": -0.0001, "pose": -0.0001, "rate_bytes": -10},
    }
    res = mod.consume_candidate_solver_derived(cand)
    prov = res["provenance"]
    assert isinstance(prov, dict)
    assert len(prov) > 0
    assert prov.get("evidence_grade") == "predicted"
    assert prov.get("promotion_eligible") is False
    assert prov.get("score_claim_valid") is False


def test_tier_b_solver_validates_per_canonical_contract() -> None:
    """Canonical validate_tier_b_contribution accepts the Tier B output."""
    mod = _load_module()
    cand = {
        "information_theoretic_floor": 0.05,
        "floor_estimate_mode": "cramer_rao",
        "m_contest_array_sha256": "i" * 64,
        "per_axis_floor": {"seg": -0.0001, "pose": -0.0001, "rate_bytes": -10},
    }
    res = mod.consume_candidate_solver_derived(cand)
    ok, errs = validate_tier_b_contribution(res)
    assert ok, f"Tier B contribution failed canonical validation: {errs}"


def test_tier_b_solver_handles_missing_per_axis_floor_gracefully() -> None:
    """No per_axis_floor => zero raw delta + note 'no_per_axis_floor'."""
    mod = _load_module()
    cand = {
        "information_theoretic_floor": None,
        "floor_estimate_mode": "cramer_rao",
        "m_contest_array_sha256": "j" * 64,
    }
    res = mod.consume_candidate_solver_derived(cand)
    assert res["predicted_delta_adjustment"] == 0.0
    assert "no_per_axis_floor" in res["tier_b_derivation_note"]


# ---------------------------------------------------------------------------
# CONSUMER_TIER_B_MODE env var honored
# ---------------------------------------------------------------------------


def test_default_mode_is_paired_comparison() -> None:
    mod = _load_module()
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("CONSUMER_TIER_B_MODE", None)
        assert mod._resolve_tier_b_mode() == "paired_comparison"


def test_env_var_tier_a_baseline_routes_to_baseline() -> None:
    mod = _load_module()
    with patch.dict(os.environ, {"CONSUMER_TIER_B_MODE": "tier_a_baseline"}):
        assert mod._resolve_tier_b_mode() == "tier_a_baseline"
        cand = {
            "information_theoretic_floor": 0.05,
            "floor_estimate_mode": "cramer_rao",
            "m_contest_array_sha256": "k" * 64,
            "per_axis_floor": {"seg": -0.0001, "pose": -0.0001, "rate_bytes": -10},
        }
        res = mod.consume_candidate(cand)
        assert res["consumer_branch_kind"] == "tier_a_baseline"
        assert res["predicted_delta_adjustment"] == 0.0


def test_env_var_tier_b_solver_routes_to_solver() -> None:
    mod = _load_module()
    with patch.dict(os.environ, {"CONSUMER_TIER_B_MODE": "tier_b_solver"}):
        assert mod._resolve_tier_b_mode() == "tier_b_solver"
        cand = {
            "information_theoretic_floor": 0.05,
            "floor_estimate_mode": "cramer_rao",
            "m_contest_array_sha256": "l" * 64,
            "per_axis_floor": {"seg": -0.0001, "pose": -0.0001, "rate_bytes": -10},
        }
        res = mod.consume_candidate(cand)
        assert res["consumer_branch_kind"] == "tier_b_solver"


def test_env_var_invalid_value_falls_back_to_default() -> None:
    mod = _load_module()
    with patch.dict(os.environ, {"CONSUMER_TIER_B_MODE": "garbage"}):
        assert mod._resolve_tier_b_mode() == "paired_comparison"


# ---------------------------------------------------------------------------
# Paired-comparison mode: emits BOTH branches to JSONL canonical store
# ---------------------------------------------------------------------------


def test_paired_comparison_mode_appends_jsonl_row(tmp_path: Path) -> None:
    """Default mode fires BOTH branches and appends to fcntl-locked
    APPEND-ONLY canonical JSONL store."""
    mod = _load_module()
    fake_state = tmp_path / ".omx" / "state"
    fake_state.mkdir(parents=True, exist_ok=True)
    fake_repo = tmp_path
    (fake_repo / "src").mkdir(parents=True, exist_ok=True)

    with patch.dict(
        os.environ,
        {
            "CONSUMER_TIER_B_MODE": "paired_comparison",
            "PACT_REPO_ROOT": str(fake_repo),
        },
    ):
        store_path = fake_state / "consumer_tier_b_promotion_posterior.jsonl"
        assert not store_path.exists()

        cand = {
            "information_theoretic_floor": 0.05,
            "floor_estimate_mode": "cramer_rao",
            "m_contest_array_sha256": "m" * 64,
            "per_axis_floor": {"seg": -0.0001, "pose": -0.0001, "rate_bytes": -10},
            "substrate_name": "test_substrate",
        }
        res = mod.consume_candidate(cand)

        # Paired-comparison returns Tier A authoritative + Tier B annotation
        assert res["consumer_branch_kind"] == "paired_comparison_authoritative_tier_a"
        assert res["predicted_delta_adjustment"] == 0.0  # Tier A authoritative
        assert "tier_b_paired_payload" in res
        tier_b = res["tier_b_paired_payload"]
        assert tier_b["consumer_branch_kind"] == "tier_b_solver"
        assert tier_b["predicted_delta_adjustment"] != 0.0

        # JSONL store was created and contains the paired row
        assert store_path.exists()
        from tac.cathedral_consumers.information_theoretic_floor_consumer._posterior_store import (
            load_paired_comparison_rows_strict,
        )
        rows = load_paired_comparison_rows_strict(store_path)
        assert len(rows) == 1
        row = rows[0]
        assert row["consumer_name"] == "information_theoretic_floor_consumer"
        assert row["consumer_version"] == "2.0.0"
        assert "tier_a_payload" in row
        assert "tier_b_payload" in row
        assert "divergence" in row
        # Tier A delta is 0 by invariant; Tier B delta is non-zero
        div = row["divergence"]
        assert div["predicted_delta_adjustment_diff"] != 0.0
        assert div["branch_kind_a"] == "tier_a_baseline"
        assert div["branch_kind_b"] == "tier_b_solver"


def test_consume_candidate_with_explicit_modes() -> None:
    """Per-mode dispatch returns the expected branch kind."""
    mod = _load_module()
    cand = {
        "information_theoretic_floor": 0.05,
        "floor_estimate_mode": "cramer_rao",
        "m_contest_array_sha256": "n" * 64,
        "per_axis_floor": {"seg": -0.0001, "pose": -0.0001, "rate_bytes": -10},
    }
    with patch.dict(os.environ, {"CONSUMER_TIER_B_MODE": "tier_a_baseline"}):
        assert mod.consume_candidate(cand)["consumer_branch_kind"] == "tier_a_baseline"
    with patch.dict(os.environ, {"CONSUMER_TIER_B_MODE": "tier_b_solver"}):
        assert mod.consume_candidate(cand)["consumer_branch_kind"] == "tier_b_solver"


# ---------------------------------------------------------------------------
# Catalog #335 / #341 / #356 / #357 gate compatibility
# ---------------------------------------------------------------------------


def test_catalog_357_gate_passes_post_promotion() -> None:
    """Catalog #357 STRICT gate accepts this consumer's Tier B contract."""
    from tac.preflight import (
        check_cathedral_consumer_tier_b_declares_canonical_contract,
    )
    violations = check_cathedral_consumer_tier_b_declares_canonical_contract(
        strict=False, verbose=False
    )
    # Live-repo regression guard: count must remain at 0 post-promotion
    assert len(violations) == 0, f"Catalog #357 violations: {violations}"


def test_catalog_341_gate_passes_post_promotion() -> None:
    """Catalog #341 routing-markers gate still passes for Tier A branch."""
    from tac.preflight import (
        check_cathedral_consumer_mps_prescreen_routing_carries_canonical_markers,
    )
    violations = check_cathedral_consumer_mps_prescreen_routing_carries_canonical_markers(
        strict=False, verbose=False
    )
    assert len(violations) == 0, f"Catalog #341 violations: {violations}"


def test_catalog_335_gate_passes() -> None:
    from tac.preflight import (
        check_cathedral_consumer_directory_package_exposes_canonical_contract,
    )
    violations = check_cathedral_consumer_directory_package_exposes_canonical_contract(
        strict=False, verbose=False
    )
    assert len(violations) == 0, f"Catalog #335 violations: {violations}"


# ---------------------------------------------------------------------------
# update_from_anchor smoke
# ---------------------------------------------------------------------------


def test_update_from_anchor_accepts_arbitrary_anchor_smoke() -> None:
    mod = _load_module()
    # update_from_anchor should not raise on any anchor shape (defensive
    # observability surface per Catalog #305)
    mod.update_from_anchor(None)
    mod.update_from_anchor({"foo": "bar"})
    mod.update_from_anchor(object())


# ---------------------------------------------------------------------------
# Producer surface smoke
# ---------------------------------------------------------------------------


def test_estimate_cramer_rao_lower_bound_smoke() -> None:
    """Producer surface kernel returns finite non-negative floor estimate."""
    pytest.importorskip("numpy")
    import numpy as np

    mod = _load_module()
    rng = np.random.default_rng(seed=42)
    m_contest = rng.normal(size=(2, 3, 8, 8)).astype(np.float64)
    for mode in ("cramer_rao", "fisher_trace", "shannon_lower"):
        floor = mod.estimate_cramer_rao_lower_bound(m_contest, mode=mode)
        assert isinstance(floor, float)
        assert floor >= 0.0 or mode == "shannon_lower"  # shannon_lower can be negative for log-sum-exp


def test_estimate_floor_rejects_wrong_shape() -> None:
    pytest.importorskip("numpy")
    import numpy as np
    mod = _load_module()
    with pytest.raises(ValueError, match="must have shape"):
        mod.estimate_cramer_rao_lower_bound(np.zeros((2, 4, 8, 8)))


def test_estimate_floor_rejects_invalid_mode() -> None:
    pytest.importorskip("numpy")
    import numpy as np
    mod = _load_module()
    with pytest.raises(ValueError, match="must be one of"):
        mod.estimate_cramer_rao_lower_bound(np.zeros((2, 3, 8, 8)), mode="bogus")
