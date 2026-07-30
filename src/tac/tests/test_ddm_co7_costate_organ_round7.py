"""Tests for ddm_co7 costate-organ round 7.

Covers the late-2026-07-28 arc join (fd2 / tb1 / eg1), the two new SENSE laws
(ema_gate_basis_v1 + basin_solve_handoff_v1), the per-parent band placement read
from the committed sealed ticket, the conditional-validity precondition schema +
substrate-change trigger, the pending-producer registry (lv1 / rv1 stay uncounted
until committed), the endgame duty chain, and the CO5 co7 re-check. Every advisory
surface is score_claim=False / actuation=NONE.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]

from tac.ddm_campaign_costate import _CO5_ENHANCEMENT_GATES  # noqa: E402
from tac.ddm_costate_organ import (  # noqa: E402
    ARC_EVIDENCE_SPECS,
    PENDING_PRODUCER_SPECS,
    PRECONDITION_SCHEMA,
    T3_SEALED_TICKET,
    _arc_evidence_rows,
    _band_position,
    _band_position_parents,
    _conditional_validity_review,
    _endgame_chain_duties,
    _pending_producers,
    _refreshed_duties,
    _sense_laws,
)

CO7_FINDING_IDS = {
    "fd2_zero_accept_disambiguation",
    "tb1_t2_race_verdict",
    "tb1_t3_sealed_ticket",
    "eg1_e1_byteclose_rehearsal",
    "eg1_e2_stop_policy",
    "eg1_e3_qdbs_finisher",
    "eg1_e3_pose_finisher",
}


def _arc():
    return _arc_evidence_rows(REPO)


def _arc_index():
    return {row["finding_id"]: row for row in _arc() if row.get("available")}


# ── R1: late-arc evidence join ─────────────────────────────────────────────────
def test_late_arc_rows_committed_and_content_hashed():
    rows = _arc()
    assert len(rows) == len(ARC_EVIDENCE_SPECS) == 15
    index = {row["finding_id"]: row for row in rows}
    for fid in CO7_FINDING_IDS:
        row = index[fid]
        assert row["available"] is True, f"{fid}: {row.get('reason')}"
        assert len(row["sha256"]) == 64
        assert row["actuation"] == "NONE" and row["score_claim"] is False


def test_sealed_ticket_row_hashes_the_actual_ticket():
    row = _arc_index()["tb1_t3_sealed_ticket"]
    assert row["artifact"] == T3_SEALED_TICKET
    ticket = json.loads((REPO / T3_SEALED_TICKET).read_text())
    assert ticket["schema"] == "ddm_tb1_tr1_sealed_ticket.v1"
    assert ticket["score_claim"] is False
    # The headline hashes/hash-prefix must match the committed ticket, not prose.
    assert ticket["ticket_hash"].startswith("007d8eacf402c4fe")
    assert "READY_TO_FIRE_UNDER_STANDING_GO" in ticket["adjudication"]["status"]


# ── R2: SENSE laws ─────────────────────────────────────────────────────────────
def test_gate_basis_law_anchored_to_committed_tb1():
    laws = _sense_laws(_arc_index())
    rows = {row["law_id"]: row for row in laws["rows"]}
    gate = rows["ema_gate_basis_v1"]
    assert "W = 2/(1-d)" in gate["statement"]
    assert gate["status"] == "ACTIVE_INSTRUMENT_VALIDITY_PRECONDITION"
    assert gate["code_commit"] == "17166ee9c4"
    assert gate["source"] is not None and len(gate["source"]["sha256"]) == 64
    assert "#85" in gate["empirical_anchor"]


def test_basin_solve_law_is_armed_not_a_stopper():
    laws = _sense_laws(_arc_index())
    rows = {row["law_id"]: row for row in laws["rows"]}
    basin = rows["basin_solve_handoff_v1"]
    assert basin["never_auto_stop"] is True
    assert set(basin["detector_guards"]) == {"#344_ncde", "#216_saddle_staircase", "#475_grokking"}
    assert basin["live_evaluation"]["status"] == "ARMED_NO_TRIGGER"
    assert "MEASURE_FINISHER_QUOTE" in basin["executable_form"]
    assert basin["source"] is not None  # anchored to the eg1 policy receipt


def test_sense_laws_fail_open_without_arc():
    laws = _sense_laws({})
    rows = {row["law_id"]: row for row in laws["rows"]}
    assert rows["ema_gate_basis_v1"]["source"] is None
    assert rows["basin_solve_handoff_v1"]["source"] is None


# ── R3: per-parent band placement ──────────────────────────────────────────────
def test_band_parents_read_from_sealed_ticket_and_all_explode():
    bp = _band_position(REPO)
    if not bp.get("available"):
        pytest.skip(f"live-base receipt absent: {bp.get('reason')}")
    parents = _band_position_parents(REPO, bp)
    rows = {row["parent"]: row for row in parents["rows"]}
    ticket = json.loads((REPO / T3_SEALED_TICKET).read_text())
    lotto = rows["tr1_lotto_t2_full_confirm"]
    assert lotto["base_d_seg"] == pytest.approx(
        ticket["adjudication"]["arithmetic"]["lotto"]["full_dseg"]
    )
    assert lotto["counted_bytes"] == 534597
    # PREMISE FLIP (co9, ng1 §2 row 10): at co7 every enumerated parent (W_joint 0.0705,
    # tr1 ~0.0138-0.0141) was above the 1e-2 band upper. co9 added the tb1 burn ENDPOINT
    # (0.00389, read from the committed pfs1 D1 eval receipt) as the CURRENT live base — it
    # is INSIDE the band, so any_parent_in_band is now True and the pre-arc parents (which
    # remain "explode") are relabeled as the stale bases the digest was wrongly parenting on.
    #   was: assert all(row["regime"] == "explode" for row in parents["rows"])
    #        assert parents["any_parent_in_band"] is False
    assert all(
        row["regime"] == "explode"
        for row in parents["rows"]
        if row["parent"] != "tb1_burn_endpoint"
    )
    endpoint = rows["tb1_burn_endpoint"]
    assert endpoint["regime"] == "correct"
    assert 5.02e-4 < endpoint["base_d_seg"] < 1.0e-2  # inside the rational band
    assert "BREAK_EVEN" in endpoint["measured_correction_value_at_base"]  # white-jitter caveat travels
    assert parents["any_parent_in_band"] is True
    assert parents["sealed_ticket"]["available"] is True
    assert parents["sealed_ticket"]["winner_arm"] == "lotto"


def test_band_parents_fail_open_without_ticket(tmp_path):
    parents = _band_position_parents(tmp_path, {"available": False})
    assert parents["rows"] == []
    assert parents["sealed_ticket"]["available"] is False
    assert parents["any_parent_in_band"] is False


# ── R4: conditional validity (precondition schema + substrate-change trigger) ──
def test_precondition_tags_present_on_negative_rows():
    specs = {spec.finding_id: spec for spec in ARC_EVIDENCE_SPECS}
    assert len(specs["sp1_support_race"].preconditions) == 2
    assert len(specs["pp1_direct_partition_price"].preconditions) == 1
    assert len(specs["fd1_zero_accept_window"].preconditions) == 1
    for spec in ARC_EVIDENCE_SPECS:
        for pre in spec.preconditions:
            assert set(pre) == {"precondition_id", "kind", "holds_when", "invalidated_by"}


def test_fd1_precondition_regraded_by_committed_successors():
    bp = _band_position(REPO)
    if not bp.get("available"):
        pytest.skip("live-base receipt absent")
    review = _conditional_validity_review(_arc(), _band_position_parents(REPO, bp))
    assert review["live_parent"] == "tr1_lotto_sealed"
    rows = {row["precondition_id"]: row for row in review["rows"]}
    fd1 = rows["fd1_fixed_capacity_wjoint_parametrization"]
    assert fd1["schema"] == PRECONDITION_SCHEMA
    assert fd1["status"] == "BROKEN"
    assert fd1["disposition"] == "RE_GRADED_BY_COMMITTED_SUCCESSOR"
    assert "fd2_zero_accept_disambiguation" in fd1["successors"]
    # sp1's parent precondition breaks (live parent is tr1) and its disposition is a re-grade
    # candidate — unchanged by co9.
    sp1_parent = rows["sp1_copy_base_parent"]
    assert sp1_parent["status"] == "BROKEN"
    assert sp1_parent["disposition"] == "REGRADE_CANDIDATE"
    # PREMISE FLIP (co9, ng1 §2 row 10): the tb1 burn endpoint entered the band, so the live
    # band-regime precondition is now BROKEN and the re-grade duty is DUE (the co8-registered
    # ARMED->DUE flip firing on the REAL in-band base, not the synthetic one below). The
    # white-jitter break-even prior travels with the DUE via the DDM-parents caveat + sense law.
    #   was: assert rows["sp1_band_regime_explode"]["status"] == "HOLDS"
    #        assert duties["sp1_copy_base_parent"]["status"] == "ARMED_NOT_DUE_BAND_STILL_EXPLODE"
    assert rows["sp1_band_regime_explode"]["status"] == "BROKEN"
    duties = {d["precondition_id"]: d for d in review["re_grade_duties"]}
    assert duties["sp1_copy_base_parent"]["status"] == "DUE"
    assert "rv1" in review["table_owner"]


def test_substrate_change_trigger_fires_when_a_parent_enters_the_band():
    # Synthetic in-band state: the trigger must promote the armed re-grade to DUE and
    # break the sp1 band precondition — proof the review is a function of the inputs.
    in_band_parents = {"any_parent_in_band": True, "rows": []}
    review = _conditional_validity_review(_arc(), in_band_parents)
    rows = {row["precondition_id"]: row for row in review["rows"]}
    assert rows["sp1_band_regime_explode"]["status"] == "BROKEN"
    duties = {d["precondition_id"]: d for d in review["re_grade_duties"]}
    assert duties["sp1_band_regime_explode"]["status"] == "DUE"
    assert duties["sp1_copy_base_parent"]["status"] == "DUE"


# ── R5: pending producers (NO-FAKE: uncounted until committed) ─────────────────
def test_pending_producers_registered_not_folded():
    assert {spec.name for spec in PENDING_PRODUCER_SPECS} == {
        "lv1_token_stack_prices",
        "rv1_conditional_validity_table",
    }
    rows = {row["producer"]: row for row in _pending_producers(REPO)}
    for name, row in rows.items():
        assert row["named_gate"]
        assert row["actuation"] == "NONE" and row["score_claim"] is False
        if not row["available"]:
            assert row["reason"] == "PENDING_COMMITTED_PRODUCER", name
    # No charter-cited lv1 price may appear as a typed number anywhere in the specs
    # (folding an uncommitted arm's numbers would be NO-FAKE #4).
    blob = json.dumps([spec.__dict__ for spec in ARC_EVIDENCE_SPECS])
    for uncommitted_number in ("364.6", "531.1", "125.4", "134.7"):
        assert uncommitted_number not in blob


# ── R6: endgame duty chain ─────────────────────────────────────────────────────
def test_endgame_chain_matches_charter_order_with_receipts():
    bp = _band_position(REPO)
    if not bp.get("available"):
        pytest.skip("live-base receipt absent")
    parents = _band_position_parents(REPO, bp)
    chain = _endgame_chain_duties(_arc_index(), _pending_producers(REPO), parents)
    duties = [row["duty"] for row in chain["chain"]]
    assert duties == [
        "B_VERDICT_WATCH",
        "T3_BURN_FIRE",
        "FIRST_GATES",
        "T1_VALIDITY_GATE",
        "BYTE_CLOSE_CHAIN_READY",
    ]
    by_duty = {row["duty"]: row for row in chain["chain"]}
    assert by_duty["T3_BURN_FIRE"]["status"] == "READY_OPERATOR_GO"
    assert by_duty["B_VERDICT_WATCH"]["cites"] == ["eg1_e2_stop_policy"]
    assert by_duty["FIRST_GATES"]["law"] == "ema_gate_basis_v1"
    assert by_duty["T1_VALIDITY_GATE"]["status"].startswith("PENDING") or by_duty[
        "T1_VALIDITY_GATE"
    ]["status"].startswith("PRODUCER_COMMITTED")
    assert by_duty["BYTE_CLOSE_CHAIN_READY"]["cites"] == ["eg1_e1_byteclose_rehearsal"]
    assert chain["head_actionable"] == "T3_BURN_FIRE"
    assert all(row["actuation"] == "NONE" for row in chain["chain"])


def test_endgame_chain_not_ready_without_seal():
    index = {k: v for k, v in _arc_index().items() if k != "tb1_t3_sealed_ticket"}
    chain = _endgame_chain_duties(index, [], {"any_parent_in_band": False})
    by_duty = {row["duty"]: row for row in chain["chain"]}
    assert by_duty["T3_BURN_FIRE"]["status"] == "NOT_SEALED"
    assert chain["head_actionable"] == "B_VERDICT_WATCH"


def test_refreshed_duty_head_hands_off_to_endgame_chain_at_full_arc():
    bp = _band_position(REPO)
    if not bp.get("available"):
        pytest.skip("live-base receipt absent")
    legacy = {"live_ranked": [{"rank": 1, "duty": "J_paint"}]}
    refreshed = _refreshed_duties(legacy, _arc(), bp)
    assert refreshed["live_ranked"][0]["duty"] == "ENDGAME_CHAIN_HANDOFF"
    dispositions = {d["disposition"] for d in refreshed["demoted"]}
    assert "MATERIALIZED_BY_TB1_SEALED_TICKET" in dispositions


# ── R7: CO5 co7 re-check ───────────────────────────────────────────────────────
def test_co5_recheck_keeps_all_gates_held_with_producer_state():
    for name, gate in _CO5_ENHANCEMENT_GATES.items():
        assert gate["producer_state"], name
    assert _CO5_ENHANCEMENT_GATES["compression_progress_per_effort"][
        "producer_state"
    ].startswith("PARTIALLY_BUILT_AWAITING_BURN")
    assert (
        _CO5_ENHANCEMENT_GATES["regret_bounded_duty_allocation"]["producer_state"]
        == "DOWNSTREAM_OF_COMPRESSION_PROGRESS_GATE"
    )


def test_co5_enhancement_state_carries_producer_state_and_recheck():
    from tac.ddm_campaign_costate import _co5_enhancement_state, load_campaign_sources

    try:
        sources, payloads = load_campaign_sources(REPO)
    except Exception as exc:
        pytest.skip(f"campaign source fleet unavailable: {type(exc).__name__}: {exc}")
    state = _co5_enhancement_state(
        ct1_payload=payloads["ct1_campaign_telemetry"],
        ct1_source=sources["ct1_campaign_telemetry"],
        evidence_join=payloads["ev1_campaign_evidence_join"],
        evidence_source=sources["ev1_campaign_evidence_join"],
    )
    # KEEP-GATED: nothing fires; the re-check is recorded with its reason.
    assert state["re_premised_count"] == 4 and state["active_count"] == 0
    assert "re_checked" in state and "co7" in state["re_checked"]
    assert state["duty_to_measure"][0]["enhancement_id"] == "compression_progress_per_effort"
    assert state["duty_to_measure"][0]["producer_state"].startswith("PARTIALLY_BUILT")


# ── R8: report + digest wiring ─────────────────────────────────────────────────
def test_report_and_digest_surface_the_new_sections():
    from tac.ddm_costate_organ import build_live_ddm_costate, digest_lines

    try:
        report = build_live_ddm_costate(repo_root=REPO)
    except Exception as exc:
        pytest.skip(f"live DDM fleet unavailable: {type(exc).__name__}: {exc}")
    if not report.get("available"):
        pytest.skip(f"live DDM fleet incomplete: {report.get('missing_required')}")
    for key in (
        "band_position_parents",
        "sense_laws",
        "pending_producers",
        "conditional_validity",
        "duties_endgame",
    ):
        assert key in report, key
    lines = digest_lines(report)
    joined = "\n".join(lines)
    assert "DDM-chain[endgame]:" in joined
    assert "head=T3_BURN_FIRE" in joined
    assert "DDM-laws:" in joined and "basin-solve=ARMED_NO_TRIGGER" in joined
    # PREMISE FLIP (co9, ng1 §2 row 10): the burn endpoint entered the band, so the digest no
    # longer emits "corrections DEAD at every live parent" — it emits the in-band re-grade with
    # the white-jitter break-even caveat welded on.
    #   was: assert "DDM-parents:" in joined and "corrections DEAD at every live parent" in joined
    assert "DDM-parents:" in joined and "tb1_burn_endpoint IN-BAND" in joined
    assert "MEASURED BREAK-EVEN at this base" in joined
    # PREMISE FLIP (co8, 2026-07-28): at co7 rv1 was an UNCOMMITTED parallel arm, so the
    # digest read table-owner=rv1[pending]. rv1 landed on main (merge 6cf5454509) and co8
    # folded its table (rv1_table section) -> the owner state is now consumed-co8. The
    # original co7 assertion is preserved below as the documented prior state.
    #   was: assert "table-owner=rv1[pending]" in joined
    assert "DDM-validity:" in joined and "table-owner=rv1[consumed-co8]" in joined
    assert "DDM-pending:" in joined and "lv1_token_stack_prices=" in joined
    # Pointer-first honesty is owned by tools/costate_digest.py section_pointer (co6);
    # the organ's lines never carry a contest-authority tag.
    assert "[contest-CPU]" not in joined and "[contest-CUDA]" not in joined
