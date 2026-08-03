"""Behaviour tests for the force x (class|edge) x verb ledger (task #809)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tac.witness_control.force_class_edge_ledger import (
    CLASSES,
    EDGES,
    LEDGER,
    VERBS,
    ForceLedgerRow,
    by_force,
    by_scope,
    coverage,
    harms,
    to_jsonl,
    unmeasured,
    unprotected_harms,
    validate_ledger,
)

REPO = Path(__file__).resolve().parents[4]


# --------------------------------------------------------------------------- #
# schema validation
# --------------------------------------------------------------------------- #

def _base_row(**over) -> dict:
    kw = dict(
        row_id="t.row",
        scope_kind="CLASS",
        scope="Lane",
        force="f",
        force_kind="OBJECTIVE",
        verb="ANNIHILATE",
        verdict="HARMS",
        measured=True,
        evidence="e",
        evidence_source="s",
        verdict_scope="INSTANCE",
        protection="ABSENT",
    )
    kw.update(over)
    return kw


def test_ledger_validates() -> None:
    validate_ledger()


def test_duplicate_row_id_refused() -> None:
    rows = (ForceLedgerRow(**_base_row()), ForceLedgerRow(**_base_row()))
    with pytest.raises(ValueError, match="duplicate"):
        validate_ledger(rows)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("verb", "SMASH", "bad verb"),
        ("verdict", "MAYBE", "bad verdict"),
        ("verdict_scope", "COSMIC", "bad verdict_scope"),
        ("protection", "HOPED", "bad protection"),
        ("force_kind", "VIBE", "bad force_kind"),
        ("scope_kind", "SOMETHING", "bad scope_kind"),
    ],
)
def test_bad_vocab_refused(field: str, value: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ForceLedgerRow(**_base_row(**{field: value}))


def test_bad_class_refused() -> None:
    with pytest.raises(ValueError, match="bad class"):
        ForceLedgerRow(**_base_row(scope="Sky"))


def test_bad_edge_refused() -> None:
    with pytest.raises(ValueError, match="bad edge"):
        ForceLedgerRow(**_base_row(scope_kind="EDGE", scope="Road<->Sky"))


def test_bad_directed_edge_refused() -> None:
    with pytest.raises(ValueError, match="bad directed edge"):
        ForceLedgerRow(**_base_row(scope_kind="DIRECTED_EDGE", scope="Lane->Lane"))


def test_directed_edge_accepted() -> None:
    ForceLedgerRow(**_base_row(scope_kind="DIRECTED_EDGE", scope="Lane->Road"))


def test_unmeasured_cannot_claim_measured() -> None:
    with pytest.raises(ValueError, match="UNMEASURED"):
        ForceLedgerRow(**_base_row(verdict="UNMEASURED", measured=True))


def test_signed_verdict_requires_measurement_or_derivation() -> None:
    with pytest.raises(ValueError, match="signed verdict"):
        ForceLedgerRow(**_base_row(verdict="HARMS", measured=False))
    # DERIVATION scope is the labelled analytic escape hatch
    ForceLedgerRow(**_base_row(verdict="HARMS", measured=False, verdict_scope="DERIVATION"))


# --------------------------------------------------------------------------- #
# vocabulary is gt2's lexicon
# --------------------------------------------------------------------------- #

def test_verbs_adopt_gt2_lexicon() -> None:
    for verb in (
        "DISPLACE", "TRANSFER", "ERODE", "GOUGE", "ANNIHILATE",
        "BIRTH", "FRAGMENT", "AMPLITUDE", "PHASE",
    ):
        assert verb in VERBS


def test_classes_canonical_order() -> None:
    assert CLASSES == ("Road", "Lane", "Undrivable", "Movable", "MyCar")


def test_nine_edges() -> None:
    assert len(EDGES) == 9
    assert EDGES[0] == "Road<->Lane"  # the hub edge, largest flip mass


# --------------------------------------------------------------------------- #
# ledger content invariants (the charter's load-bearing rows)
# --------------------------------------------------------------------------- #

def test_governing_law_row_present() -> None:
    row = next(r for r in LEDGER if r.row_id == "law.aggregate_cashing_dead")
    assert row.measured
    assert "7 independent instruments" in row.evidence


def test_structural_fact_row_cites_modules_py() -> None:
    row = next(r for r in LEDGER if r.row_id == "struct.d_seg_pricing")
    assert "modules.py" in row.evidence_source
    assert row.verdict_scope == "STRUCTURAL"


def test_lane_annihilation_row() -> None:
    row = next(r for r in LEDGER if r.row_id == "tr1.lane.annihilate")
    assert row.scope == "Lane" and row.verb == "ANNIHILATE" and row.verdict == "HARMS"
    assert "58.23%" in row.evidence


def test_margin_floor_is_inert_not_harms() -> None:
    row = next(r for r in LEDGER if r.row_id == "mg1.margin_floor.inert")
    assert row.verdict == "INERT"
    assert row.verdict_scope == "FORMULATION"


def test_every_harms_row_has_protection_status() -> None:
    for row in harms():
        assert row.protection in ("BUILT", "BUILT_UNFIRED", "DESIGNED", "ABSENT", "N_A")
        if row.protection in ("BUILT", "BUILT_UNFIRED", "DESIGNED"):
            assert row.protection_ref, f"{row.row_id}: protected but no protection_ref"


def test_every_measured_row_cites_an_artifact() -> None:
    for row in LEDGER:
        if row.measured:
            assert row.evidence_source.strip(), row.row_id


# --------------------------------------------------------------------------- #
# queries
# --------------------------------------------------------------------------- #

def test_harms_sorted_by_magnitude() -> None:
    hs = harms()
    mags = [r.magnitude_s or 0.0 for r in hs]
    assert mags == sorted(mags, reverse=True)


def test_unprotected_harms_ranked_absent_first() -> None:
    ranks = {"ABSENT": 0, "BUILT_UNFIRED": 1, "DESIGNED": 2}
    out = unprotected_harms()
    vals = [ranks[r.protection] for r in out]
    assert vals == sorted(vals)
    assert all(r.verdict == "HARMS" for r in out)


def test_by_force_and_scope() -> None:
    lane_rows = by_scope("Lane")
    assert any(r.row_id == "tr1.lane.annihilate" for r in lane_rows)
    tr1 = by_force("tr1_seg_leg_composite")
    assert len(tr1) >= 5


def test_unmeasured_rows_exist_and_are_honest() -> None:
    rows = unmeasured()
    assert rows, "an all-measured ledger would be a fake (m50)"
    for r in rows:
        assert not r.measured


# --------------------------------------------------------------------------- #
# coverage: the m50 denominator
# --------------------------------------------------------------------------- #

def test_coverage_reports_denominator() -> None:
    cov = coverage()
    assert cov["scope_cells_possible"] == cov["forces"] * (len(CLASSES) + len(EDGES))
    assert cov["scope_cells_with_explicit_rows"] <= cov["scope_cells_possible"]
    assert cov["scope_cells_unmeasured_implicit"] > 0, (
        "claiming full coverage would be vacuous PASS (m50)"
    )
    assert "VACUOUS" in cov["honest_statement"]


def test_coverage_per_force_denominators() -> None:
    cov = coverage()
    for name, v in cov["per_force"].items():
        assert v["class_cells_possible"] == 5, name
        assert v["edge_cells_possible"] == 9, name
        assert v["class_cells_covered"] <= 5
        assert v["edge_cells_covered"] <= 9


# --------------------------------------------------------------------------- #
# export + CLI
# --------------------------------------------------------------------------- #

def test_jsonl_roundtrip(tmp_path: Path) -> None:
    p = to_jsonl(tmp_path / "ledger.jsonl")
    lines = p.read_text().strip().splitlines()
    assert len(lines) == len(LEDGER)
    first = json.loads(lines[0])
    assert set(first) >= {
        "row_id", "scope_kind", "scope", "force", "verb", "verdict",
        "measured", "evidence", "evidence_source", "verdict_scope", "protection",
    }


def test_cli_coverage_runs() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "tac.witness_control.force_class_edge_ledger", "--coverage"],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert "scope_cells_possible" in out


def test_cli_unprotected_runs() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "tac.witness_control.force_class_edge_ledger", "--unprotected"],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert proc.returncode == 0, proc.stderr
    assert "HARMS" in proc.stdout
