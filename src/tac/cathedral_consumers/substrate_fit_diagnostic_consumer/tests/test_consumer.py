# SPDX-License-Identifier: MIT
"""Tests for substrate_fit_diagnostic_consumer Tier B promotion.

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


MODULE_PATH = "tac.cathedral_consumers.substrate_fit_diagnostic_consumer"


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
    assert mod.CONSUMER_NAME == "substrate_fit_diagnostic_consumer"
    assert mod.CONSUMER_VERSION == "2.0.0"
    assert isinstance(mod.CONSUMER_HOOK_NUMBERS, tuple)
    assert len(mod.CONSUMER_HOOK_NUMBERS) >= 1
    for hook in mod.CONSUMER_HOOK_NUMBERS:
        assert isinstance(hook, HookNumber)


def test_consumer_declares_tier_b_score_contributing() -> None:
    mod = _load_module()
    assert mod.CONSUMER_TIER == ConsumerTier.TIER_B_SCORE_CONTRIBUTING


def test_hook_numbers_match_125_declaration() -> None:
    """Per Catalog #125: hooks #1 + #4 + #5 + #6 ACTIVE."""
    mod = _load_module()
    assert HookNumber.SENSITIVITY_MAP in mod.CONSUMER_HOOK_NUMBERS
    assert HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH in mod.CONSUMER_HOOK_NUMBERS
    assert HookNumber.CONTINUAL_LEARNING_POSTERIOR in mod.CONSUMER_HOOK_NUMBERS
    assert HookNumber.PROBE_DISAMBIGUATOR in mod.CONSUMER_HOOK_NUMBERS


# ---------------------------------------------------------------------------
# Tier A baseline branch: Catalog #341 observability-only invariants
# ---------------------------------------------------------------------------


def test_tier_a_baseline_returns_zero_delta_adjustment() -> None:
    mod = _load_module()
    cand = {
        "substrate_fit_scores": {"sub_x": 0.8},
        "substrate_name": "sub_x",
        "m_contest_array_sha256": "a" * 64,
    }
    res = mod.consume_candidate_tier_a_baseline(cand)
    assert res["predicted_delta_adjustment"] == 0.0
    assert res["promotable"] is False
    assert res["axis_tag"] == "[predicted]"
    assert res["consumer_branch_kind"] == "tier_a_baseline"


def test_tier_a_baseline_returns_axis_decomposition() -> None:
    mod = _load_module()
    cand = {
        "substrate_fit_scores": {"sub_y": 0.9},
        "substrate_name": "sub_y",
        "per_axis_residuals": {"seg": -0.001, "pose": -0.0005},
        "m_contest_array_sha256": "b" * 64,
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
    # substrate-fit is observability of rendered-frame gradient, not codec
    assert decomp["predicted_archive_bytes_delta"] == 0


# ---------------------------------------------------------------------------
# Tier B solver branch: bounded non-zero delta + canonical contract
# ---------------------------------------------------------------------------


def test_tier_b_solver_returns_non_zero_within_safety_rail_per_axis_residuals() -> None:
    mod = _load_module()
    cand = {
        "substrate_fit_scores": {"sub_z": 0.7},
        "substrate_name": "sub_z",
        "per_axis_residuals": {"seg": -0.0001, "pose": -0.0005},
        "m_contest_array_sha256": "c" * 64,
    }
    res = mod.consume_candidate_solver_derived(cand)
    delta = res["predicted_delta_adjustment"]
    assert delta != 0.0
    assert -0.05 <= delta <= 0.05
    assert res["consumer_branch_kind"] == "tier_b_solver"


def test_tier_b_solver_uses_fit_score_when_no_per_axis_residuals() -> None:
    mod = _load_module()
    cand = {
        "substrate_fit_scores": {"sub_w": 0.8},
        "substrate_name": "sub_w",
        "m_contest_array_sha256": "d" * 64,
    }
    res = mod.consume_candidate_solver_derived(cand)
    # fit_score=0.8 => raw_delta = 0.05 - 0.10*0.8 = -0.03 (within rail)
    delta = res["predicted_delta_adjustment"]
    assert delta == pytest.approx(-0.03, abs=1e-9)
    assert "substrate_fit_score=0.8000" in res["tier_b_derivation_note"]


def test_tier_b_solver_clipped_to_safety_rail_min() -> None:
    mod = _load_module()
    cand = {
        "substrate_name": "sub_huge",
        "per_axis_residuals": {"seg": -1.0, "pose": -1.0},
        "m_contest_array_sha256": "e" * 64,
    }
    res = mod.consume_candidate_solver_derived(cand)
    assert res["predicted_delta_adjustment"] == -0.05
    assert "clipped_to_safety_rail_min" in res["tier_b_derivation_note"]


def test_tier_b_solver_clipped_to_safety_rail_max() -> None:
    mod = _load_module()
    cand = {
        "substrate_name": "sub_huge",
        "per_axis_residuals": {"seg": 1.0, "pose": 1.0},
        "m_contest_array_sha256": "f" * 64,
    }
    res = mod.consume_candidate_solver_derived(cand)
    assert res["predicted_delta_adjustment"] == 0.05
    assert "clipped_to_safety_rail_max" in res["tier_b_derivation_note"]


def test_tier_b_solver_no_signal_returns_zero() -> None:
    """Missing both per_axis_residuals and substrate_fit_scores =>
    bounded delta = 0 + note 'no_signal'."""
    mod = _load_module()
    cand = {"m_contest_array_sha256": "g" * 64}
    res = mod.consume_candidate_solver_derived(cand)
    assert res["predicted_delta_adjustment"] == 0.0
    assert res["tier_b_derivation_note"] == "no_signal"


def test_tier_b_solver_uses_diagnostic_cpu_axis_tag() -> None:
    """Catalog #357 requires empirically-grounded axis_tag (NOT [predicted])."""
    mod = _load_module()
    cand = {
        "substrate_fit_scores": {"sub_a": 0.5},
        "substrate_name": "sub_a",
        "per_axis_residuals": {"seg": -0.0001, "pose": -0.0001},
        "m_contest_array_sha256": "h" * 64,
    }
    res = mod.consume_candidate_solver_derived(cand)
    assert res["axis_tag"] == "[diagnostic-CPU]"
    assert res["axis_tag"] != "[predicted]"


def test_tier_b_solver_preserves_promotable_false() -> None:
    mod = _load_module()
    cand = {
        "substrate_fit_scores": {"sub_b": 0.5},
        "substrate_name": "sub_b",
        "per_axis_residuals": {"seg": -0.0001, "pose": -0.0001},
        "m_contest_array_sha256": "i" * 64,
    }
    res = mod.consume_candidate_solver_derived(cand)
    assert res["promotable"] is False


def test_tier_b_solver_threads_canonical_provenance() -> None:
    """Catalog #323/#357: provenance field present + non-empty Mapping."""
    mod = _load_module()
    cand = {
        "substrate_fit_scores": {"sub_c": 0.5},
        "substrate_name": "sub_c",
        "per_axis_residuals": {"seg": -0.0001, "pose": -0.0001},
        "m_contest_array_sha256": "j" * 64,
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
        "substrate_fit_scores": {"sub_d": 0.5},
        "substrate_name": "sub_d",
        "per_axis_residuals": {"seg": -0.0001, "pose": -0.0001},
        "m_contest_array_sha256": "k" * 64,
    }
    res = mod.consume_candidate_solver_derived(cand)
    ok, errs = validate_tier_b_contribution(res)
    assert ok, f"Tier B contribution failed canonical validation: {errs}"


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
            "substrate_fit_scores": {"sub_e": 0.5},
            "substrate_name": "sub_e",
            "per_axis_residuals": {"seg": -0.0001, "pose": -0.0001},
            "m_contest_array_sha256": "l" * 64,
        }
        res = mod.consume_candidate(cand)
        assert res["consumer_branch_kind"] == "tier_a_baseline"
        assert res["predicted_delta_adjustment"] == 0.0


def test_env_var_tier_b_solver_routes_to_solver() -> None:
    mod = _load_module()
    with patch.dict(os.environ, {"CONSUMER_TIER_B_MODE": "tier_b_solver"}):
        assert mod._resolve_tier_b_mode() == "tier_b_solver"
        cand = {
            "substrate_fit_scores": {"sub_f": 0.5},
            "substrate_name": "sub_f",
            "per_axis_residuals": {"seg": -0.0001, "pose": -0.0001},
            "m_contest_array_sha256": "m" * 64,
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
            "substrate_fit_scores": {"sub_paired": 0.7},
            "substrate_name": "sub_paired",
            "per_axis_residuals": {"seg": -0.0001, "pose": -0.0001},
            "m_contest_array_sha256": "n" * 64,
        }
        res = mod.consume_candidate(cand)

        assert res["consumer_branch_kind"] == "paired_comparison_authoritative_tier_a"
        assert res["predicted_delta_adjustment"] == 0.0
        assert "tier_b_paired_payload" in res
        tier_b = res["tier_b_paired_payload"]
        assert tier_b["consumer_branch_kind"] == "tier_b_solver"
        assert tier_b["predicted_delta_adjustment"] != 0.0

        assert store_path.exists()
        from tac.cathedral_consumers.substrate_fit_diagnostic_consumer._posterior_store import (
            load_paired_comparison_rows_strict,
        )
        rows = load_paired_comparison_rows_strict(store_path)
        assert len(rows) == 1
        row = rows[0]
        assert row["consumer_name"] == "substrate_fit_diagnostic_consumer"
        assert row["consumer_version"] == "2.0.0"
        assert "tier_a_payload" in row
        assert "tier_b_payload" in row
        assert "divergence" in row
        div = row["divergence"]
        assert div["predicted_delta_adjustment_diff"] != 0.0
        assert div["branch_kind_a"] == "tier_a_baseline"
        assert div["branch_kind_b"] == "tier_b_solver"


def test_consume_candidate_with_explicit_modes() -> None:
    mod = _load_module()
    cand = {
        "substrate_fit_scores": {"sub_g": 0.5},
        "substrate_name": "sub_g",
        "per_axis_residuals": {"seg": -0.0001, "pose": -0.0001},
        "m_contest_array_sha256": "o" * 64,
    }
    with patch.dict(os.environ, {"CONSUMER_TIER_B_MODE": "tier_a_baseline"}):
        assert mod.consume_candidate(cand)["consumer_branch_kind"] == "tier_a_baseline"
    with patch.dict(os.environ, {"CONSUMER_TIER_B_MODE": "tier_b_solver"}):
        assert mod.consume_candidate(cand)["consumer_branch_kind"] == "tier_b_solver"


# ---------------------------------------------------------------------------
# Catalog #335 / #341 / #356 / #357 gate compatibility
# ---------------------------------------------------------------------------


def test_catalog_357_gate_passes_post_promotion() -> None:
    from tac.preflight import (
        check_cathedral_consumer_tier_b_declares_canonical_contract,
    )
    violations = check_cathedral_consumer_tier_b_declares_canonical_contract(
        strict=False, verbose=False
    )
    assert len(violations) == 0, f"Catalog #357 violations: {violations}"


def test_catalog_341_gate_passes_post_promotion() -> None:
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
    mod.update_from_anchor(None)
    mod.update_from_anchor({"foo": "bar"})
    mod.update_from_anchor(object())


# ---------------------------------------------------------------------------
# Producer surface smoke
# ---------------------------------------------------------------------------


def test_compute_substrate_fit_score_smoke() -> None:
    """Producer surface returns dict of fit scores in [0, 1]."""
    pytest.importorskip("numpy")
    import numpy as np

    mod = _load_module()
    rng = np.random.default_rng(seed=42)
    m_contest = rng.normal(size=(2, 3, 8, 8)).astype(np.float64)
    m_inflated_a = m_contest.copy()  # perfect fit
    m_inflated_b = -m_contest.copy()  # anti-aligned (cosine = -1 => clipped to 0)
    res = mod.compute_substrate_fit_score(
        m_contest,
        {"perfect": m_inflated_a, "anti": m_inflated_b},
    )
    assert res["perfect"] == pytest.approx(1.0, abs=1e-9)
    assert res["anti"] == 0.0


def test_compute_substrate_fit_score_rejects_shape_mismatch() -> None:
    pytest.importorskip("numpy")
    import numpy as np
    mod = _load_module()
    # Non-zero m_contest so the all-zero short-circuit doesn't fire before
    # the per-substrate shape check
    m_contest = np.ones((2, 3, 8, 8))
    m_wrong = np.zeros((2, 3, 4, 4))
    with pytest.raises(ValueError, match="!= M_contest shape"):
        mod.compute_substrate_fit_score(m_contest, {"bad": m_wrong})


def test_compute_substrate_fit_score_empty_input() -> None:
    pytest.importorskip("numpy")
    import numpy as np
    mod = _load_module()
    m_contest = np.zeros((2, 3, 8, 8))
    assert mod.compute_substrate_fit_score(m_contest, {}) == {}


def test_compute_substrate_fit_score_zero_contest_returns_perfect_fit() -> None:
    """All-zero M_contest => every substrate trivially perfect fit."""
    pytest.importorskip("numpy")
    import numpy as np
    mod = _load_module()
    m_contest = np.zeros((2, 3, 8, 8))
    m_inflated = np.ones((2, 3, 8, 8))
    res = mod.compute_substrate_fit_score(m_contest, {"x": m_inflated})
    assert res == {"x": 1.0}
