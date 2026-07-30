"""Tests for ddm_ja1 (QA73) — the JOINT waterfill allocator SENSE law.

The organ gains ONE standing SENSE law `joint_exchange_rate_allocator_v1` (co9 pattern):
the digest surfaces the top of the committed joint waterfill table instead of axis-scoped
duties. Advisory; actuation NONE; score_claim False; pointer 0.1910828242 UNMOVED. The three
committed deliverables (atlas / waterfill table / order DAG) exist and are self-consistent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]

from tac.ddm_costate_organ import (  # noqa: E402
    _arc_evidence_rows,
    _deferral_ledger_source,
    _sense_laws,
    build_live_ddm_costate,
    digest_lines,
)

RESEARCH = REPO / ".omx" / "research"
ATLAS = RESEARCH / "ddm_ja1_joint_sensitivity_atlas_20260731.json"
TABLE = RESEARCH / "ddm_ja1_joint_waterfill_table_20260731.json"
ORDER = RESEARCH / "ddm_ja1_order_of_operations_dag_20260731.json"


def _arc_index():
    return {row["finding_id"]: row for row in _arc_evidence_rows(REPO) if row.get("available")}


def test_joint_allocator_law_present_and_labeled():
    laws = _sense_laws(_arc_index(), ledger_source=_deferral_ledger_source(REPO))
    rows = {row["law_id"]: row for row in laws["rows"]}
    assert "joint_exchange_rate_allocator_v1" in rows
    law = rows["joint_exchange_rate_allocator_v1"]
    assert law["epistemic_status"] == "MEASURED_ALLOCATOR"
    assert law["evidence_axis"] == "[macOS-CPU advisory]"
    # standing law: fire on joint exchange rate, not axis identity.
    assert "axis identity" in law["statement"]
    assert "NON-ADDITIVE" in law["statement"]
    # the committed table pointer is present and points at a real file.
    assert law["committed_table"] == ".omx/research/ddm_ja1_joint_waterfill_table_20260731.json"
    assert (REPO / law["committed_table"]).exists()
    assert "QA73" in law["ledger_rows"]


def test_allocation_surprise_is_the_seg_saturation_finding():
    laws = _sense_laws(_arc_index(), ledger_source=_deferral_ledger_source(REPO))
    law = {r["law_id"]: r for r in laws["rows"]}["joint_exchange_rate_allocator_v1"]
    surprise = law["allocation_surprise"]
    # the surprise is exactly: biggest axis (seg) has NO cheap byte lever (saturated knee);
    # cheap byte levers are POSE; seg only moves via a heavy re-burn.
    assert "SATURATED" in surprise and "POSE" in surprise
    assert "QA24" in surprise  # the heavy seg re-burn (capacity pool, parallel)
    top = law["table_top"]
    assert any("QA66" in t and "REALIZED-live-base" in t for t in top)  # rank-1 measured lever


def test_organ_stays_advisory_actuation_none():
    laws = _sense_laws(_arc_index(), ledger_source=_deferral_ledger_source(REPO))
    assert laws["actuation"] == "NONE"
    assert laws["score_claim"] is False


def test_committed_deliverables_exist_and_are_self_consistent():
    for p in (ATLAS, TABLE, ORDER):
        assert p.exists(), p
        obj = json.loads(p.read_text())
        assert obj["score_claim"] is False
        assert obj["pointer"] == "0.1910828242 [contest-CPU] UNMOVED"
    table = json.loads(TABLE.read_text())
    # rank-1 is the ONLY realized-live-base ranked rung and it is negative (a real win).
    r1 = table["ranked_rungs"][0]
    assert r1["id"] == "QA66" and r1["label"] == "REALIZED-live-base"
    assert r1["dS"] < 0
    # the three saturated pools are labeled realized-live-base (measured, not asserted).
    assert len(table["saturated_do_not_spend"]) == 3
    for pool in table["saturated_do_not_spend"]:
        assert "REALIZED-live-base" in pool["label"]
    # order DAG: cheap pose build comes first, seg re-burn last.
    order = json.loads(ORDER.read_text())
    builds = order["ordered_next_3_builds"]
    assert "v4d" in builds[0]["build"]
    assert "reburn" in builds[-1]["build"]


def test_digest_surfaces_the_joint_allocator_line():
    try:
        report = build_live_ddm_costate(repo_root=REPO)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"live DDM fleet unavailable: {type(exc).__name__}: {exc}")
    if not report.get("available"):
        pytest.skip(f"live DDM fleet incomplete: {report.get('missing_required')}")
    joined = "\n".join(digest_lines(report))
    assert "DDM-joint[ja1 allocator, QA73]:" in joined
    assert "JOINT exchange rate not axis identity" in joined
    assert "QA66" in joined  # the rank-1 measured lever surfaces
    # organ stays advisory: no contest-authority tag anywhere.
    assert "[contest-CPU]" not in joined and "[contest-CUDA]" not in joined
    assert "actuation NONE" in joined
