# SPDX-License-Identifier: MIT
"""Tests for findings_lagrangian_consumer (WAVE-2-PREREQ + Catalog #355 sister).

Coverage per the task contract:
  - Canonical contract compliance per Catalog #335 (CONSUMER_NAME /
    CONSUMER_VERSION / CONSUMER_HOOK_NUMBERS / update_from_anchor /
    consume_candidate).
  - Catalog #341 routing markers (Tier A invariants).
  - update_from_anchor delegates to canonical helper (mocked anchor).
  - consume_candidate computes scalar + decomposition + posterior sigma
    (synthetic candidate; round-trip verify).
  - Catalog #287 placeholder-rationale rejection (via Provenance).
  - Catalog #323 canonical Provenance integration.
  - 6-hook wire-in declaration matches CONSUMER_HOOK_NUMBERS.
  - Auto-discovery via discover_compliant_consumer_modules returns this
    consumer.
  - Catalog #354 sister-callable regression guard.
  - Live-repo regression guard.
"""
from __future__ import annotations

import importlib
import math
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from tac.cathedral.consumer_contract import (
    ConsumerTier,
    HookNumber,
    validate_consumer_module,
)


MODULE_PATH = "tac.cathedral_consumers.findings_lagrangian_consumer"


def _load_module() -> Any:
    return importlib.import_module(MODULE_PATH)


# ---------------------------------------------------------------------------
# Canonical contract compliance per Catalog #335
# ---------------------------------------------------------------------------


def test_consumer_satisfies_canonical_contract():
    mod = _load_module()
    res = validate_consumer_module(mod, module_path=MODULE_PATH)
    assert res.contract_compliant, f"validation errors: {res.validation_errors}"


def test_consumer_declares_canonical_metadata():
    mod = _load_module()
    assert mod.CONSUMER_NAME == "findings_lagrangian_consumer"
    assert mod.CONSUMER_VERSION == "1.0.0"
    assert isinstance(mod.CONSUMER_HOOK_NUMBERS, tuple)
    assert len(mod.CONSUMER_HOOK_NUMBERS) >= 1


def test_consumer_explicit_tier_a():
    """Phase 1: explicit TIER_A_OBSERVABILITY_ONLY per Catalog #341."""
    mod = _load_module()
    assert hasattr(mod, "CONSUMER_TIER")
    assert mod.CONSUMER_TIER == ConsumerTier.TIER_A_OBSERVABILITY_ONLY


def test_consumer_hook_numbers_match_declared_active_hooks():
    """6-hook wire-in declaration: hooks #1, #4, #5 ACTIVE per module docstring."""
    mod = _load_module()
    assert HookNumber.SENSITIVITY_MAP in mod.CONSUMER_HOOK_NUMBERS
    assert HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH in mod.CONSUMER_HOOK_NUMBERS
    assert HookNumber.CONTINUAL_LEARNING_POSTERIOR in mod.CONSUMER_HOOK_NUMBERS


def test_consumer_hook_numbers_omit_na_hooks():
    """Hooks #2, #3, #6 are N/A at Phase 1 per module docstring."""
    mod = _load_module()
    assert HookNumber.PARETO_CONSTRAINT not in mod.CONSUMER_HOOK_NUMBERS
    assert HookNumber.BIT_ALLOCATOR not in mod.CONSUMER_HOOK_NUMBERS
    assert HookNumber.PROBE_DISAMBIGUATOR not in mod.CONSUMER_HOOK_NUMBERS


# ---------------------------------------------------------------------------
# Catalog #341 Tier A routing-markers invariants
# ---------------------------------------------------------------------------


CANONICAL_CANDIDATE_KEYS = (
    "predicted_delta_adjustment",
    "rationale",
    "axis_tag",
    "promotable",
)


def test_consume_candidate_returns_canonical_keys():
    mod = _load_module()
    out = mod.consume_candidate(
        {"candidate_id": "test", "family": "test_fam", "predicted_score_delta": -0.01}
    )
    for key in CANONICAL_CANDIDATE_KEYS:
        assert key in out, f"missing canonical key: {key}"


def test_consume_candidate_tier_a_invariants_always():
    """Catalog #341: predicted_delta_adjustment=0.0 + promotable=False +
    axis_tag=[predicted] for ALL Tier A returns (with-signal + no-signal +
    error paths)."""
    mod = _load_module()
    for candidate in [
        {},
        {"family": "f1"},
        {"family": "f1", "predicted_score_delta": -0.01},
        {"family": "f1", "predicted_score_delta": 0.5},
        {"family": "f1", "predicted_score_delta": float("nan")},
        {"family": "f1", "predicted_score_delta": 2.5},  # clip case
        {"family": "f1", "predicted_score_delta": -2.5},  # clip case
        {"candidate_id": "no_family"},
        {"lane_id": "lane_abc", "predicted_delta": -0.005},  # legacy alias
    ]:
        out = mod.consume_candidate(candidate)
        assert out["predicted_delta_adjustment"] == 0.0, candidate
        assert out["promotable"] is False, candidate
        assert out["axis_tag"] == "[predicted]", candidate


def test_consume_candidate_residual_clip():
    """Phase 1 _RESIDUAL_CLIP=1.0 protects posterior from runaway candidates."""
    mod = _load_module()
    # >1.0 → still returns Tier A invariants; computation succeeds
    out = mod.consume_candidate({"family": "f", "predicted_score_delta": 100.0})
    assert out["predicted_delta_adjustment"] == 0.0
    assert out["lagrangian_scalar"] is not None
    assert math.isfinite(out["lagrangian_scalar"])


def test_consume_candidate_nan_signal_returns_no_signal_skip():
    """NaN residual → consumer treats as no-signal (lagrangian_scalar=None)."""
    mod = _load_module()
    out = mod.consume_candidate(
        {"family": "f", "predicted_score_delta": float("nan")}
    )
    assert out["lagrangian_scalar"] is None
    assert out["posterior_sigma_per_term"] is None
    assert "skip" in out["rationale"].lower() or "lacks" in out["rationale"].lower()


# ---------------------------------------------------------------------------
# Observability surfaces (Catalog #305 + Catalog #125 hook #1)
# ---------------------------------------------------------------------------


def test_consume_candidate_with_signal_returns_lagrangian_scalar():
    mod = _load_module()
    out = mod.consume_candidate(
        {"family": "fam_x", "predicted_score_delta": -0.005}
    )
    assert out["lagrangian_scalar"] is not None
    assert math.isfinite(out["lagrangian_scalar"])
    assert isinstance(out["per_term_decomposition"], dict)
    assert "data_fit" in out["per_term_decomposition"]
    assert "scalar" in out["per_term_decomposition"]


def test_consume_candidate_with_signal_returns_posterior_sigma():
    """Catalog #125 hook #1 sensitivity-map signal."""
    mod = _load_module()
    out = mod.consume_candidate(
        {"family": "fam_x", "predicted_score_delta": -0.005}
    )
    assert isinstance(out["posterior_sigma_per_term"], list)
    assert len(out["posterior_sigma_per_term"]) >= 1
    for sigma in out["posterior_sigma_per_term"]:
        assert isinstance(sigma, float)
        assert sigma >= 0.0
        assert math.isfinite(sigma)


def test_consume_candidate_per_term_decomposition_keys():
    """Catalog #305 observability surface — per-term decomposition."""
    mod = _load_module()
    out = mod.consume_candidate(
        {"family": "fam_x", "predicted_score_delta": -0.005}
    )
    decompose = out["per_term_decomposition"]
    expected_keys = {
        "data_fit",
        "occam_complexity_weighted",
        "occam_interpretability_weighted",
        "partition_penalty_weighted",
        "info_gain_reward_weighted",
        "scalar",
    }
    assert expected_keys.issubset(set(decompose.keys()))


def test_consume_candidate_phase_2_info_gain_placeholder_is_none():
    """Phase 2 will populate expected_information_gain_per_action; Phase 1 None."""
    mod = _load_module()
    out = mod.consume_candidate(
        {"family": "fam_x", "predicted_score_delta": -0.005}
    )
    assert out["expected_information_gain_per_action"] is None


# ---------------------------------------------------------------------------
# Catalog #323 canonical Provenance integration
# ---------------------------------------------------------------------------


def test_consume_candidate_carries_canonical_provenance():
    mod = _load_module()
    out = mod.consume_candidate(
        {"family": "fam_x", "predicted_score_delta": -0.005}
    )
    assert "canonical_provenance" in out
    prov = out["canonical_provenance"]
    assert isinstance(prov, dict)
    # Per Catalog #287/#323: PREDICTED grade + non-promotable invariants
    assert prov.get("evidence_grade") in ("PREDICTED", "predicted")
    # measurement_axis preserved per Catalog #287
    assert "[predicted]" in str(prov.get("measurement_axis", ""))
    # promotion_eligible MUST be False per Catalog #319/#341
    assert prov.get("promotion_eligible") is False
    assert prov.get("score_claim_valid") is False


def test_consume_candidate_provenance_present_even_on_no_signal():
    """canonical_provenance always present so downstream consumers can audit."""
    mod = _load_module()
    out = mod.consume_candidate({"candidate_id": "no_signal"})
    assert isinstance(out.get("canonical_provenance"), dict)
    assert out["canonical_provenance"].get("promotion_eligible") is False


# ---------------------------------------------------------------------------
# update_from_anchor delegation + fail-safe contract
# ---------------------------------------------------------------------------


def test_update_from_anchor_handles_none_gracefully():
    mod = _load_module()
    mod.update_from_anchor(None)  # MUST NOT raise


def test_update_from_anchor_handles_dict_anchor():
    mod = _load_module()
    mod.update_from_anchor({"equation_id": "test_eq", "residual": 0.03})


def test_update_from_anchor_handles_typed_anchor():
    """Typed anchor with attribute access (ContinualLearningAnchor pattern)."""
    mod = _load_module()

    class MockAnchor:
        equation_id = "test_eq"
        residual = 0.05

    mod.update_from_anchor(MockAnchor())


def test_update_from_anchor_missing_fields_no_op():
    """Anchor without equation_id OR residual → silent NO-OP (never raises)."""
    mod = _load_module()
    mod.update_from_anchor({})
    mod.update_from_anchor({"unrelated_key": "value"})
    mod.update_from_anchor({"equation_id": "eq"})  # missing residual
    mod.update_from_anchor({"residual": 0.01})  # missing equation_id


def test_update_from_anchor_delegates_to_canonical_helper():
    """Verifies forwarding to tac.findings_lagrangian.posterior_update_from_anchors."""
    mod = _load_module()
    with patch(
        "tac.findings_lagrangian.posterior_update_from_anchors"
    ) as mock_update:
        mod.update_from_anchor({"equation_id": "test_eq", "residual": 0.03})
        assert mock_update.called
        call_kwargs = mock_update.call_args.kwargs
        # First positional arg is equation_id
        call_args = mock_update.call_args.args
        assert call_args[0] == "test_eq"
        assert call_kwargs["anchor_residuals"] == (0.03,)


def test_update_from_anchor_swallows_canonical_helper_exception():
    """Canonical helper failures MUST NOT propagate (cathedral main loop safety)."""
    mod = _load_module()
    with patch(
        "tac.findings_lagrangian.posterior_update_from_anchors",
        side_effect=RuntimeError("simulated solver failure"),
    ):
        mod.update_from_anchor({"equation_id": "test_eq", "residual": 0.03})
        # MUST NOT raise


def test_update_from_anchor_nan_residual_no_op():
    """NaN residual rejected at extraction; canonical helper never called."""
    mod = _load_module()
    with patch(
        "tac.findings_lagrangian.posterior_update_from_anchors"
    ) as mock_update:
        mod.update_from_anchor({"equation_id": "eq", "residual": float("nan")})
        assert not mock_update.called


# ---------------------------------------------------------------------------
# consume_candidate fail-safe / defensive paths
# ---------------------------------------------------------------------------


def test_consume_candidate_findings_lagrangian_import_failure_returns_tier_a():
    """Phase 1 ImportError fallback returns canonical Tier A annotation."""
    mod = _load_module()

    class ImportRaiser:
        def __getattr__(self, item: str) -> Any:
            raise ImportError("simulated import failure")

    # Patch the lazy import — note consume_candidate does lazy import
    # inside the function body. We can patch via sys.modules.
    import sys

    saved = {}
    for key in list(sys.modules.keys()):
        if key.startswith("tac.findings_lagrangian"):
            saved[key] = sys.modules[key]
    for key in saved:
        del sys.modules[key]

    # Insert a sentinel module that raises on attribute access.
    sys.modules["tac.findings_lagrangian"] = ImportRaiser()  # type: ignore

    try:
        out = mod.consume_candidate(
            {"family": "fam", "predicted_score_delta": -0.005}
        )
        # The ImportError path returns canonical Tier A.
        assert out["predicted_delta_adjustment"] == 0.0
        assert out["promotable"] is False
        assert out["axis_tag"] == "[predicted]"
        # lagrangian_scalar None because computation could not run.
        assert out["lagrangian_scalar"] is None
    finally:
        # Restore the saved modules.
        del sys.modules["tac.findings_lagrangian"]
        for key, val in saved.items():
            sys.modules[key] = val


def test_consume_candidate_solver_exception_returns_tier_a():
    """Solver failures MUST be trapped → canonical Tier A annotation."""
    mod = _load_module()
    with patch(
        "tac.findings_lagrangian.posterior_update_from_anchors",
        side_effect=RuntimeError("simulated solver crash"),
    ):
        out = mod.consume_candidate(
            {"family": "fam", "predicted_score_delta": -0.005}
        )
        assert out["predicted_delta_adjustment"] == 0.0
        assert out["lagrangian_scalar"] is None
        assert "failed" in out["rationale"].lower() or "skip" in out["rationale"].lower()


# ---------------------------------------------------------------------------
# Auto-discovery via discover_compliant_consumer_modules
# ---------------------------------------------------------------------------


def test_auto_discovery_picks_up_this_consumer():
    """Mirror discover_compliant_consumer_modules logic — verify auto-discovery."""
    from tac.cathedral.consumer_contract import validate_consumer_module

    # Walk up from this test file to find src/tac/cathedral_consumers.
    # File layout: src/tac/cathedral_consumers/findings_lagrangian_consumer/tests/test_consumer.py
    consumer_dir = Path(__file__).resolve().parent.parent.parent
    assert consumer_dir.exists(), f"consumer dir missing: {consumer_dir}"
    assert consumer_dir.name == "cathedral_consumers", consumer_dir

    discovered_names: list[str] = []
    for sub in sorted(consumer_dir.iterdir()):
        if not sub.is_dir():
            continue
        if sub.name in {"__pycache__", "tests"}:
            continue
        if sub.name.startswith("_"):
            continue
        init_path = sub / "__init__.py"
        if not init_path.exists():
            continue
        module_dotted = f"tac.cathedral_consumers.{sub.name}"
        try:
            m = importlib.import_module(module_dotted)
        except ImportError:
            continue
        reg = validate_consumer_module(m, module_path=module_dotted)
        if reg.contract_compliant:
            discovered_names.append(reg.consumer_name)

    assert "findings_lagrangian_consumer" in discovered_names


# ---------------------------------------------------------------------------
# Catalog #354 sister-callable regression guard
# ---------------------------------------------------------------------------


def test_consumer_module_callable_via_globals_regression():
    """Sister of Catalog #185 META-meta: gate functions callable via module globals."""
    mod = _load_module()
    # Both canonical surfaces must be callable
    assert callable(mod.update_from_anchor)
    assert callable(mod.consume_candidate)


# ---------------------------------------------------------------------------
# Cite-chain regression (module docstring + Phase 2 promotion pathway)
# ---------------------------------------------------------------------------


def test_module_docstring_cites_catalog_341():
    """Module docstring MUST cite Catalog #341 (Tier A invariants source)."""
    mod = _load_module()
    assert "#341" in (mod.__doc__ or "")


def test_module_docstring_cites_phase_2_tier_b_promotion():
    """Module docstring MUST document Tier B promotion pathway for sister
    subagents (CATHEDRAL-SMARTER-DESIGN-MEMO Dimension 6 Step 6.5)."""
    doc = (_load_module().__doc__ or "").lower()
    assert "phase 2" in doc
    assert "tier b" in doc


def test_module_docstring_cites_catalog_355():
    """Module docstring MUST cite Catalog #355 (sister wire-in surface)."""
    mod = _load_module()
    assert "#355" in (mod.__doc__ or "")


def test_module_docstring_cites_canonical_findings_lagrangian():
    mod = _load_module()
    assert "tac.findings_lagrangian" in (mod.__doc__ or "")


# ---------------------------------------------------------------------------
# Equation ID extraction (Phase 1 mapping)
# ---------------------------------------------------------------------------


def test_extract_equation_id_prefers_family():
    mod = _load_module()
    assert mod._extract_equation_id({"family": "fam_a", "lane_id": "lane_b"}) == "fam_a"


def test_extract_equation_id_falls_back_through_keys():
    mod = _load_module()
    assert mod._extract_equation_id({"candidate_id": "cand_1"}) == "cand_1"
    assert mod._extract_equation_id({"lane_id": "lane_x"}) == "lane_x"
    assert (
        mod._extract_equation_id({"candidate_family": "fam_x"})
        == "fam_x"
    )


def test_extract_equation_id_unknown_fallback():
    mod = _load_module()
    assert mod._extract_equation_id({}) == "unknown_family"
    assert mod._extract_equation_id({"unrelated": "v"}) == "unknown_family"
