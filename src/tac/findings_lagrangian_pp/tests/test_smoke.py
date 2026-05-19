# SPDX-License-Identifier: MIT
"""TRACK B smoke tests: import + optional NumPyro contract.

Per T3 grand council 3-round consolidated verdict + operator-frontier-override
2026-05-19 verbatim *"we shoud pursue PP in parallel"*: TRACK B
(`tac.findings_lagrangian_pp`) is the parallel-pursuit NumPyro hierarchical
posterior track to TRACK A (closed-form Gaussian per slot 20 binding).

Per Carmack ULTRA-MVP Q9 AMEND: Phase 1.A is ONE equation only — the
canonical Gaussian conjugate update at `tac.findings_lagrangian.posterior`.
TRACK B's smoke test here is minimal:

    1. Verify the package imports cleanly without NumPyro installed.
    2. Verify the canonical _optional_numpyro guard contract is honored
       (try_import_numpyro returns None when NumPyro missing; require_numpyro
       raises TrackBNumPyroUnavailableError with explicit remediation).
    3. Verify TRACK B exposes the canonical UnifiedPrediction-compatible
       interface so cathedral autopilot can consume both tracks symmetrically.

Heavier NumPyro-specific tests (MCMC convergence + hierarchical model
validation + cross-architecture posterior interpretation) are DEFERRED to
Phase 1.B per Q9 AMEND + Catalog #345 strict-flip gate. NumPyro install
requires explicit operator decision per CLAUDE.md "Deployment version
checklist" non-negotiable; we do NOT auto-install in this dispatch.

Cite-chain:
- Operator-frontier-override 2026-05-19 PARALLEL TRACK A + TRACK B
- T3 grand council slot 20 Q5 RATIFIED hand-rolled Gaussian (TRACK A;
  TRACK B per operator override is parallel-pursuit)
- Carmack ULTRA-MVP Q9 AMEND (slot 20-second-supplemental)
- Slot A recovery (commit 5de1a96f1; scaffold ~3500 LOC / 14 modules)
- CLAUDE.md "Forbidden silent-skip cascades" (NumPyro unavailable raises;
  does NOT silently fall back)
"""
from __future__ import annotations

import pytest


def test_track_b_package_imports_cleanly_without_numpyro() -> None:
    """tac.findings_lagrangian_pp imports without requiring NumPyro at module load.

    Per CLAUDE.md "Forbidden silent-skip cascades": TRACK B's module-level
    imports MUST not depend on NumPyro being present (otherwise importing
    the package on a fresh checkout / CI without numpyro installed would
    raise ImportError, breaking Catalog #188 test-import discipline).

    The canonical pattern uses the _optional_numpyro guard module which
    exposes `TRACK_B_NUMPYRO_AVAILABLE` flag + `require_numpyro()` raising
    helper. Top-level package import must succeed regardless.
    """
    import tac.findings_lagrangian_pp  # noqa: F401 — import check
    # The package's _optional_numpyro guard exposes the canonical flag:
    from tac.findings_lagrangian_pp._optional_numpyro import (
        TRACK_B_NUMPYRO_AVAILABLE,
        TrackBNumPyroUnavailableError,
        require_numpyro,
        try_import_numpyro,
    )
    # Flag is a boolean — True iff NumPyro + jax actually importable.
    assert isinstance(TRACK_B_NUMPYRO_AVAILABLE, bool)


def test_track_b_numpyro_guard_contract_honored() -> None:
    """The _optional_numpyro guard module follows CLAUDE.md fail-loud contract.

    Per CLAUDE.md "Forbidden silent-skip cascades": when NumPyro is missing,
    `require_numpyro()` raises `TrackBNumPyroUnavailableError` with explicit
    installation remediation; does NOT silently return None or fall back.

    When NumPyro IS available, `require_numpyro()` returns the (numpyro, jax)
    module tuple. This test exercises the BRANCH that fires in the current
    test environment (which may or may not have numpyro installed).
    """
    from tac.findings_lagrangian_pp._optional_numpyro import (
        TRACK_B_NUMPYRO_AVAILABLE,
        TrackBNumPyroUnavailableError,
        require_numpyro,
        try_import_numpyro,
    )

    try_result = try_import_numpyro()
    if TRACK_B_NUMPYRO_AVAILABLE:
        # Branch: numpyro IS installed
        assert try_result is not None
        assert len(try_result) == 2  # (numpyro, jax)
        # require_numpyro returns the modules
        numpyro, jax = require_numpyro()
        assert numpyro is not None
        assert jax is not None
    else:
        # Branch: numpyro is NOT installed (the canonical CI / fresh-checkout case)
        assert try_result is None
        # require_numpyro MUST raise with remediation per CLAUDE.md
        with pytest.raises(TrackBNumPyroUnavailableError) as exc_info:
            require_numpyro()
        msg = str(exc_info.value)
        # Remediation includes pip-install command
        assert "pip install" in msg
        assert "numpyro" in msg
        # Operator-override cite-chain preserved per CLAUDE.md HISTORICAL_PROVENANCE
        assert "operator-frontier-override" in msg or "2026-05-19" in msg
        # Documented degraded path: UnifiedPrediction degrades to TRACK A
        assert "TRACK A" in msg or "ensemble_prediction_from_tracks" in msg


def test_track_b_exposes_unified_prediction_compatible_interface() -> None:
    """TRACK B + TRACK A both emit ScalarPrediction through the canonical interface.

    Per `tac.findings_lagrangian.unified.ensemble_prediction_from_tracks`: both
    tracks emit `ScalarPrediction` objects with `source_track` discriminator
    (`track_a_handrolled` vs `track_b_numpyro` vs `ensemble`).

    This test verifies the canonical `ScalarPrediction` accepts a
    `track_b_numpyro` source_track value — proving the canonical contract
    is symmetric across the two tracks. The actual TRACK B implementation
    construction is deferred to Phase 1.B (requires NumPyro install).
    """
    from tac.findings_lagrangian.unified import (
        EnsembleError,
        ScalarPrediction,
        UnifiedPrediction,
        ensemble_prediction_from_tracks,
    )

    # Build a TRACK A ScalarPrediction
    track_a_pred = ScalarPrediction(
        predicted_value=0.5,
        uncertainty_sigma=0.1,
        axis_tag="[predicted]",
        source_track="track_a_handrolled",
        equation_id="test_eq_unified_interface_v1",
        n_anchors_consumed=3,
        rationale="canonical TRACK A test fixture per Phase 1.A smoke",
    )
    assert track_a_pred.source_track == "track_a_handrolled"
    # Build a synthetic TRACK B ScalarPrediction — the canonical interface
    # accepts source_track='track_b_numpyro' even when NumPyro is unavailable
    # in the test env. (TRACK B *implementation* requires NumPyro; the
    # *contract* does not.)
    track_b_pred = ScalarPrediction(
        predicted_value=0.55,
        uncertainty_sigma=0.12,
        axis_tag="[predicted]",
        source_track="track_b_numpyro",
        equation_id="test_eq_unified_interface_v1",
        n_anchors_consumed=3,
        rationale="canonical TRACK B test fixture per Phase 1.A smoke",
    )
    assert track_b_pred.source_track == "track_b_numpyro"
    # Test the ensemble combination
    unified = ensemble_prediction_from_tracks(track_a_pred, track_b_pred)
    assert isinstance(unified, UnifiedPrediction)
    assert unified.equation_id == "test_eq_unified_interface_v1"
    # Ensemble degrades gracefully when track_b is None (the standard
    # behavior when NumPyro is unavailable + cathedral autopilot routes
    # to TRACK A-only ensemble per documented fallback).
    unified_degraded = ensemble_prediction_from_tracks(track_a_pred, track_b=None)
    assert isinstance(unified_degraded, UnifiedPrediction)
    assert unified_degraded.track_b_prediction is None
    assert unified_degraded.track_a_weight == 1.0
    assert unified_degraded.track_b_weight == 0.0
    # Ensemble degenerated to TRACK A's prediction.
    assert unified_degraded.ensemble_prediction.predicted_value == pytest.approx(
        track_a_pred.predicted_value, abs=1e-12
    )


def test_track_b_package_documents_operator_override_provenance() -> None:
    """tac.findings_lagrangian_pp documents its operator-override origin.

    Per CLAUDE.md "Apples-to-apples evidence discipline" + "Submission auth
    eval" + Catalog #110/#113 HISTORICAL_PROVENANCE non-negotiables: TRACK B
    is operator-mandated via 2026-05-19 frontier-override. The package's
    canonical guard module documents this cite-chain.
    """
    from tac.findings_lagrangian_pp import _optional_numpyro

    docstring = _optional_numpyro.__doc__ or ""
    # Operator-override cite-chain preserved
    assert "operator-frontier-override" in docstring
    assert "2026-05-19" in docstring
    # Slot 20 binding decision cite-chain
    assert "slot 20" in docstring.lower() or "Slot 20" in docstring
    # NumPyro install remediation
    assert "pip install" in docstring
    # Canonical degradation path
    assert "ensemble_prediction_from_tracks" in docstring or "TRACK A" in docstring
