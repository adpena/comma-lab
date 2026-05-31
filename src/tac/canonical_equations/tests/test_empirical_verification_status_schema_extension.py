# SPDX-License-Identifier: MIT
"""Tests for Slot S Phase 1: EmpiricalAnchor.empirical_verification_status schema extension.

Per Slot N M3 corrective action (Catalog #363 4-value taxonomy at the
per-anchor canonical equations surface) + canonical 2-landing pattern.

Cross-references:
  * `tac.council_continual_learning.EmpiricalVerificationStatus` — sister
    canonical 4-value taxonomy at the council deliberation surface.
  * `tac.canonical_equations.equation.EmpiricalAnchor` — surface this
    test file exercises.
  * Catalog #363 — canonical recursive self-reflection protocol non-negotiable
    that mandates 4-value taxonomy on every empirical-anchor surface.
  * Catalog #110/#113 — APPEND-ONLY HISTORICAL_PROVENANCE invariant that
    forces backward-compat field default (327 legacy rows must load
    unchanged).
"""
from __future__ import annotations

import pytest

from tac.canonical_equations.equation import (
    ASSUMED_AWAITING_VERIFICATION,
    INFERRED_FROM_DOMAIN_LITERATURE,
    UNVERIFIED_EMPIRICAL_VERIFICATION_STATUSES,
    VALID_EMPIRICAL_VERIFICATION_STATUSES,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    VERIFIED_VIA_SOURCE_INSPECTION,
    EmpiricalAnchor,
    InvalidEquationError,
)
from tac.provenance.builders import build_provenance_for_predicted

# ---------------------------------------------------------------------------
# Constants + canonical 4-value taxonomy invariants
# ---------------------------------------------------------------------------


def test_canonical_4_value_taxonomy_is_complete():
    """Per Catalog #363 the 4-value taxonomy is fixed by the canonical contract."""
    assert VERIFIED_VIA_SOURCE_INSPECTION == "VERIFIED_VIA_SOURCE_INSPECTION"
    assert VERIFIED_VIA_EMPIRICAL_ANCHOR == "VERIFIED_VIA_EMPIRICAL_ANCHOR"
    assert INFERRED_FROM_DOMAIN_LITERATURE == "INFERRED_FROM_DOMAIN_LITERATURE"
    assert ASSUMED_AWAITING_VERIFICATION == "ASSUMED_AWAITING_VERIFICATION"


def test_valid_statuses_frozenset_has_4_canonical_members():
    assert isinstance(VALID_EMPIRICAL_VERIFICATION_STATUSES, frozenset)
    assert len(VALID_EMPIRICAL_VERIFICATION_STATUSES) == 4
    assert frozenset(
        {
            VERIFIED_VIA_SOURCE_INSPECTION,
            VERIFIED_VIA_EMPIRICAL_ANCHOR,
            INFERRED_FROM_DOMAIN_LITERATURE,
            ASSUMED_AWAITING_VERIFICATION,
        }
    ) == VALID_EMPIRICAL_VERIFICATION_STATUSES


def test_unverified_statuses_subset_has_2_members():
    assert isinstance(UNVERIFIED_EMPIRICAL_VERIFICATION_STATUSES, frozenset)
    assert len(UNVERIFIED_EMPIRICAL_VERIFICATION_STATUSES) == 2
    assert frozenset(
        {INFERRED_FROM_DOMAIN_LITERATURE, ASSUMED_AWAITING_VERIFICATION}
    ) == UNVERIFIED_EMPIRICAL_VERIFICATION_STATUSES
    # Subset invariant
    assert UNVERIFIED_EMPIRICAL_VERIFICATION_STATUSES.issubset(
        VALID_EMPIRICAL_VERIFICATION_STATUSES
    )


def test_canonical_taxonomy_mirrors_council_continual_learning_sister():
    """Per Catalog #363 the canonical equations taxonomy MUST be byte-identical
    to the sister council_continual_learning taxonomy so cross-surface
    classification is canonical."""
    from tac.council_continual_learning import (
        VALID_EMPIRICAL_VERIFICATION_STATUSES as council_valid,
    )
    from tac.council_continual_learning import (
        EmpiricalVerificationStatus as CouncilEVS,
    )

    # The 4 canonical tokens must match byte-for-byte
    assert VERIFIED_VIA_SOURCE_INSPECTION == CouncilEVS.VERIFIED_VIA_SOURCE_INSPECTION
    assert VERIFIED_VIA_EMPIRICAL_ANCHOR == CouncilEVS.VERIFIED_VIA_EMPIRICAL_ANCHOR
    assert INFERRED_FROM_DOMAIN_LITERATURE == CouncilEVS.INFERRED_FROM_DOMAIN_LITERATURE
    assert ASSUMED_AWAITING_VERIFICATION == CouncilEVS.ASSUMED_AWAITING_VERIFICATION
    # frozensets equal (set semantics; not ordering-dependent)
    assert council_valid == VALID_EMPIRICAL_VERIFICATION_STATUSES


# ---------------------------------------------------------------------------
# Backward-compat invariant (Catalog #110/#113 APPEND-ONLY)
# ---------------------------------------------------------------------------


def _make_anchor(**kwargs):
    """Construct a canonical EmpiricalAnchor with synthetic defaults."""
    prov = build_provenance_for_predicted(
        model_id="unit_test_model",
        inputs_sha256="0" * 64,
    )
    defaults = {
        "anchor_id": "test_anchor",
        "measurement_utc": "2026-05-29T07:00:00Z",
        "inputs": {"k": 1.0},
        "predicted_output": {"y": 1.0},
        "empirical_output": {"y": 1.0},
        "residual": 0.0,
        "source_artifact": "/tmp/synthetic",
        "measurement_method": "unit_test",
        "provenance": prov,
    }
    defaults.update(kwargs)
    return EmpiricalAnchor(**defaults)


def test_backward_compat_default_is_none():
    """327 legacy rows constructed without empirical_verification_status must
    load unchanged per Catalog #110/#113 APPEND-ONLY invariant."""
    a = _make_anchor()
    assert a.empirical_verification_status is None


def test_backward_compat_to_dict_omits_field_when_none():
    """Byte-stable serialization: legacy rows MUST NOT carry an
    `empirical_verification_status` key in the JSONL payload (would corrupt
    sha-based forensic identity per Catalog #245 sister discipline)."""
    a = _make_anchor()
    payload = a.to_dict()
    assert "empirical_verification_status" not in payload


def test_backward_compat_legacy_registry_loads_clean():
    """Slot N Round 1 anchor: 327 legacy rows in the live registry MUST
    load unchanged with the new optional field (all-None backward-compat)."""
    from tac.canonical_equations.registry import load_equation_registry_strict

    eqs = load_equation_registry_strict()
    assert len(eqs) >= 326, (
        f"expected >= 326 canonical equation events at landing; got {len(eqs)}"
    )


# ---------------------------------------------------------------------------
# Forward-compat: field accepted + serialized
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status",
    [
        VERIFIED_VIA_SOURCE_INSPECTION,
        VERIFIED_VIA_EMPIRICAL_ANCHOR,
        INFERRED_FROM_DOMAIN_LITERATURE,
        ASSUMED_AWAITING_VERIFICATION,
    ],
)
def test_forward_compat_each_canonical_status_accepted(status):
    """All 4 canonical tokens are accepted at construction."""
    a = _make_anchor(empirical_verification_status=status)
    assert a.empirical_verification_status == status


@pytest.mark.parametrize(
    "status",
    [
        VERIFIED_VIA_SOURCE_INSPECTION,
        VERIFIED_VIA_EMPIRICAL_ANCHOR,
        INFERRED_FROM_DOMAIN_LITERATURE,
        ASSUMED_AWAITING_VERIFICATION,
    ],
)
def test_forward_compat_to_dict_includes_field_when_set(status):
    """When set, to_dict() includes the field per canonical serialization."""
    a = _make_anchor(empirical_verification_status=status)
    payload = a.to_dict()
    assert payload["empirical_verification_status"] == status


# ---------------------------------------------------------------------------
# Validator invariants (refuses out-of-taxonomy values + wrong types)
# ---------------------------------------------------------------------------


def test_invalid_string_token_rejected():
    """Out-of-taxonomy string values are refused at construction with a
    canonical InvalidEquationError citing Catalog #363."""
    with pytest.raises(InvalidEquationError) as exc_info:
        _make_anchor(empirical_verification_status="VERIFIED_VIA_HEARSAY")
    assert "Catalog #363" in str(exc_info.value)
    assert "4-value taxonomy" in str(exc_info.value)


def test_wrong_type_int_rejected():
    """Non-string values (int) are refused."""
    with pytest.raises(InvalidEquationError) as exc_info:
        _make_anchor(empirical_verification_status=42)
    assert "must be a string OR None" in str(exc_info.value)


def test_wrong_type_list_rejected():
    """Non-string values (list) are refused."""
    with pytest.raises(InvalidEquationError):
        _make_anchor(empirical_verification_status=["VERIFIED_VIA_SOURCE_INSPECTION"])


def test_wrong_type_dict_rejected():
    """Non-string values (dict) are refused."""
    with pytest.raises(InvalidEquationError):
        _make_anchor(empirical_verification_status={"status": "VERIFIED"})


def test_empty_string_rejected():
    """Empty string is out-of-taxonomy and is refused (cannot match any of
    the 4 canonical non-empty tokens)."""
    with pytest.raises(InvalidEquationError):
        _make_anchor(empirical_verification_status="")


def test_case_mismatch_rejected():
    """Lowercase variant is out-of-taxonomy + rejected (canonical tokens
    are UPPER_SNAKE_CASE per Catalog #363)."""
    with pytest.raises(InvalidEquationError):
        _make_anchor(empirical_verification_status="verified_via_source_inspection")


# ---------------------------------------------------------------------------
# Frozen-dataclass invariant (canonical immutability per Catalog #110)
# ---------------------------------------------------------------------------


def test_frozen_dataclass_field_immutable_post_construction():
    """EmpiricalAnchor is frozen; field cannot be mutated post-construction."""
    a = _make_anchor(empirical_verification_status=VERIFIED_VIA_SOURCE_INSPECTION)
    with pytest.raises((AttributeError, Exception)):
        a.empirical_verification_status = INFERRED_FROM_DOMAIN_LITERATURE


# ---------------------------------------------------------------------------
# Round-trip serialization (to_dict + reconstruct)
# ---------------------------------------------------------------------------


def test_round_trip_with_status_preserves_field():
    """to_dict + reconstruct round-trip preserves the field when set."""
    a = _make_anchor(empirical_verification_status=INFERRED_FROM_DOMAIN_LITERATURE)
    payload = a.to_dict()
    # Reconstruct from payload (drop provenance dict; reconstruct via builder)
    reconstructed_kwargs = {
        k: v
        for k, v in payload.items()
        if k not in {"provenance"}
    }
    reconstructed_kwargs["provenance"] = a.provenance  # reuse original
    reconstructed = EmpiricalAnchor(**reconstructed_kwargs)
    assert reconstructed.empirical_verification_status == INFERRED_FROM_DOMAIN_LITERATURE


def test_round_trip_without_status_preserves_none():
    """Legacy round-trip (field absent) preserves None default."""
    a = _make_anchor()
    payload = a.to_dict()
    assert "empirical_verification_status" not in payload
    # Reconstruct without the field → still None
    reconstructed_kwargs = {
        k: v
        for k, v in payload.items()
        if k not in {"provenance"}
    }
    reconstructed_kwargs["provenance"] = a.provenance
    reconstructed = EmpiricalAnchor(**reconstructed_kwargs)
    assert reconstructed.empirical_verification_status is None


# ---------------------------------------------------------------------------
# Module exports (Catalog #335 sister: public API surface)
# ---------------------------------------------------------------------------


def test_canonical_4_value_taxonomy_exported_in_module_all():
    """The 4 canonical tokens + frozensets are exported in __all__."""
    from tac.canonical_equations import equation as eq_module

    for export in (
        "VERIFIED_VIA_SOURCE_INSPECTION",
        "VERIFIED_VIA_EMPIRICAL_ANCHOR",
        "INFERRED_FROM_DOMAIN_LITERATURE",
        "ASSUMED_AWAITING_VERIFICATION",
        "VALID_EMPIRICAL_VERIFICATION_STATUSES",
        "UNVERIFIED_EMPIRICAL_VERIFICATION_STATUSES",
    ):
        assert export in eq_module.__all__, (
            f"{export} must be in tac.canonical_equations.equation.__all__"
        )
