# SPDX-License-Identifier: MIT
from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from tac import preflight as preflight_module
from tac.preflight import (
    PreflightError,
    _check_293_source_scope_violations,
    check_cathedral_literature_anchors_have_source_scope,
)


def _row(**kwargs: object) -> SimpleNamespace:
    defaults = {
        "substrate_id": "z3_balle_hyperprior_bolton",
        "literature_anchor": "balle_2018",
        "source_supports": "Scale hyperpriors support image-compression rate-distortion.",
        "paper_claim_scope": "Natural-image learned compression, not Pact score evidence.",
        "pact_must_prove": "Byte-closed contest archive score and runtime custody.",
        "decode_complexity_evidence": "T4 inflate timing required before promotion.",
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_check_293_live_repo_zero_violations() -> None:
    assert check_cathedral_literature_anchors_have_source_scope(strict=False) == []


def test_check_293_flags_missing_substrate_scope_field() -> None:
    violations = _check_293_source_scope_violations(
        substrate_rows=[_row(source_supports="")],
        pareto_rows=[],
    )

    assert len(violations) == 1
    assert "canonical_substrate_inventory" in violations[0]
    assert "source_supports" in violations[0]
    assert "literature_anchor='balle_2018'" in violations[0]


def test_check_293_flags_missing_serialized_pareto_scope_field() -> None:
    clean = _row()
    pareto_row = {
        "substrate_id": "z3_balle_hyperprior_bolton",
        "literature_anchor": "balle_2018",
        "source_supports": clean.source_supports,
        "paper_claim_scope": clean.paper_claim_scope,
        "pact_must_prove": "TBD",
        "decode_complexity_evidence": clean.decode_complexity_evidence,
    }

    violations = _check_293_source_scope_violations(
        substrate_rows=[clean],
        pareto_rows=[pareto_row],
    )

    assert len(violations) == 1
    assert "serialized_pareto_rows" in violations[0]
    assert "pact_must_prove" in violations[0]


def test_check_293_ignores_unanchored_rows() -> None:
    violations = _check_293_source_scope_violations(
        substrate_rows=[_row(literature_anchor="", source_supports="")],
        pareto_rows=[{"substrate_id": "control", "literature_anchor": ""}],
    )

    assert violations == []


def test_check_293_strict_raises_on_synthetic_missing_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    from tac.optimization import substrate_composition_matrix as scm

    monkeypatch.setattr(scm, "canonical_substrate_inventory", lambda: [_row(pact_must_prove="")])
    monkeypatch.setattr(scm, "per_substrate_pareto_rows", lambda: [])
    monkeypatch.setattr(scm, "serialize_pareto_rows", lambda rows: [])

    with pytest.raises(PreflightError, match="Catalog #293"):
        check_cathedral_literature_anchors_have_source_scope(strict=True)


def test_check_293_wired_into_preflight_all_strict() -> None:
    src = inspect.getsource(preflight_module.preflight_all)
    needle = "check_cathedral_literature_anchors_have_source_scope("
    idx = src.find(needle)
    assert idx >= 0, "Catalog #293 must be wired into preflight_all"
    call_window = src[idx : idx + 140]
    assert "strict=True" in call_window
