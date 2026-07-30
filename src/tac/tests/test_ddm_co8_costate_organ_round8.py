"""Tests for ddm_co8 costate-organ round 8.

Covers: (1) the COMMITTED rv1 conditional-validity table folded as typed rows (8
reactivations, each a PROPOSAL with a named measurement; 12 honest non-reactivations;
2 charter-framing corrections; the pending-producer row flips to CONSUMED); (2) the pn1
evidence joins (VOI ranking as a SENSE input, the nu-pivot decision node, the S1
Stage-A/Stage-B duty rows, the granularity-ladder LAW + ARMED_NOT_DUE contingent race,
the rebalance-event watch); (3) the allocator-law duty waterfill (priced ranking, pools
law, and the honest CO5 adjudication: pn1's VOI table is NOT the named CT1-v2-class
producer — everything stays gated); (4) rung 6, the organ-freshness gate (event-driven
rounds: registry derived from live module constants, measured-derived threshold,
re-routing-class detection, fail-open WARN-only hook). Every surface is
score_claim=False / actuation=NONE.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]

from tac.ddm_costate_organ import (  # noqa: E402
    PENDING_PRODUCER_SPECS,
    PN1_MEMO,
    RV1_CHARTER_CORRECTIONS,
    RV1_NON_REACTIVATION_SPECS,
    RV1_REACTIVATION_SPECS,
    RV1_TABLE_MEMO,
    T3_SEALED_TICKET,
    _arc_evidence_rows,
    _band_position,
    _band_position_parents,
    _conditional_validity_review,
    _duty_allocator_waterfill,
    _pending_producers,
    _pn1_nodes,
    _rv1_conditional_validity_table,
    _sense_laws,
    consumed_evidence_registry,
)


def _arc():
    return _arc_evidence_rows(REPO)


def _arc_index():
    return {row["finding_id"]: row for row in _arc() if row.get("available")}


# ── rung 1: rv1 table consumed ────────────────────────────────────────────────
def test_rv1_table_folds_committed_memo_with_content_hash():
    table = _rv1_conditional_validity_table(REPO)
    assert table["available"] is True
    assert table["source"]["path"] == RV1_TABLE_MEMO
    assert len(table["source"]["sha256"]) == 64
    assert table["counts"]["reactivations"] == len(RV1_REACTIVATION_SPECS) == 8
    assert table["counts"]["non_reactivations"] == len(RV1_NON_REACTIVATION_SPECS) == 12
    # rv1 §2b accounting (R1:3 R2:4 R3:1 R4:3 R5:1 R6:3 R7:4 R8:1 = 20), DERIVED from
    # the typed rows so the count can never drift from the negatives lists.
    assert table["counts"]["distinct_negatives_regraded"] == 20
    assert sum(len(r["negatives"]) for r in table["reactivation_rows"]) == 20
    assert len(table["charter_corrections"]) == len(RV1_CHARTER_CORRECTIONS) == 2
    assert table["actuation"] == "NONE" and table["score_claim"] is False


def test_rv1_reactivations_are_proposals_with_named_measurements():
    table = _rv1_conditional_validity_table(REPO)
    for row in table["reactivation_rows"]:
        # NOTHING reactivates until its measurement lands (rv1 memo boundary).
        assert row["reactivated"] is False
        assert row["measurement"], row["row_id"]
        assert row["consumer"], row["row_id"]
        assert row["negatives"], row["row_id"]
        assert row["source"]["sha256"] == table["source"]["sha256"]


def test_rv1_statuses_match_the_committed_ranking():
    rows = {row["row_id"]: row for row in _rv1_conditional_validity_table(REPO)["reactivation_rows"]}
    # rv1 §1 ranking: R1 first by leverage but POST_BURN; R7/R4 are $0 measurable NOW.
    assert rows["R1_terminal_band_discrete_search"]["rank"] == 1
    assert rows["R1_terminal_band_discrete_search"]["status"] == "POST_BURN"
    assert rows["R7_token_stream_coder_race"]["status"] == "MEASURABLE_NOW_T2_DUMPS"
    assert rows["R4_token_granularity_correction_probe"]["status"] == "MEASURABLE_NOW_T2_CHECKPOINTS"
    assert rows["R2_correction_stream_band_repriced"]["status"] == "BAND_ENTRY_ARMED"
    assert rows["R6_lane_channel_intraining_entrants"]["status"] == "BURN_WINDOW_RACE_FIRST_ITEM"
    assert rows["R5_step_hosc_head_conditional"]["status"] == "CONDITIONAL_TRIGGER_ONLY"
    # Ranks are the rv1 leverage x cheapness ordering, contiguous 1..8.
    assert sorted(r["rank"] for r in rows.values()) == list(range(1, 9))


def test_rv1_non_reactivations_are_tagged_closed_verdicts():
    table = _rv1_conditional_validity_table(REPO)
    dispositions = {row["row_id"]: row["disposition"] for row in table["non_reactivation_rows"]}
    assert dispositions["X4_island_homotopy_ladder_323"] == "DISSOLVED_ON_TOKEN_VEHICLE"
    assert dispositions["X5_chroma_sub2px"] == "RECLASSIFIED_POSE_SAFE_EXPLOIT"
    assert dispositions["X6_posthoc_stored_pose_family"] == "STAYS_DEAD_PRECONDITION_STRENGTHENED"
    assert dispositions["X11_eikonal_viscosity_era"] == "RETRACTED_NO_REACTIVATION_VENUE_HERE"
    assert dispositions["X12_witness_era_formulation_negatives"] == "SWEPT_REMAIN_CLOSED"
    for row in table["non_reactivation_rows"]:
        assert row["reason"], row["row_id"]


def test_rv1_table_fails_open_without_memo(tmp_path):
    table = _rv1_conditional_validity_table(tmp_path)
    assert table["available"] is False
    assert table["reason"] == "RV1_COMMITTED_TABLE_ABSENT"


def test_rv1_pending_producer_row_flips_to_consumed():
    spec = {s.name: s for s in PENDING_PRODUCER_SPECS}
    assert spec["rv1_conditional_validity_table"].consumed_by == "co8_rv1_conditional_validity_table"
    assert spec["lv1_token_stack_prices"].consumed_by is None
    rows = {row["producer"]: row for row in _pending_producers(REPO)}
    rv1 = rows["rv1_conditional_validity_table"]
    assert rv1["available"] is True
    assert rv1["status"] == "COMMITTED_CONSUMED_BY_CO8_RV1_CONDITIONAL_VALIDITY_TABLE"
    # lv1 stays PENDING: its worktree numbers remain unfoldable (NO-FAKE #4).
    lv1 = rows["lv1_token_stack_prices"]
    if not lv1["available"]:
        assert lv1["reason"] == "PENDING_COMMITTED_PRODUCER"
    else:  # a committed lv1 landing must NOT silently absorb — fold is a named next round
        assert lv1["status"] == "COMMITTED_ARTIFACT_PRESENT_FOLD_ON_NEXT_ROUND"


def test_conditional_validity_table_owner_flips_with_rv1():
    review = _conditional_validity_review(_arc(), {"any_parent_in_band": False}, rv1_table_available=True)
    assert review["table_owner_state"] == "CONSUMED_CO8"
    assert "rv1" in review["table_owner"] and "CONSUMED" in review["table_owner"]
    legacy = _conditional_validity_review(_arc(), {"any_parent_in_band": False})
    assert legacy["table_owner_state"] == "PENDING"
    assert "pending producer" in legacy["table_owner"]


# ── rung 2: pn1 evidence joins ────────────────────────────────────────────────
def test_pn1_nodes_content_hashed_and_voi_top_is_rerouting_class():
    nodes = _pn1_nodes(REPO, {"any_parent_in_band": False})
    assert nodes["available"] is True
    assert nodes["source"]["path"] == PN1_MEMO and len(nodes["source"]["sha256"]) == 64
    voi = nodes["voi_ranking"]
    assert voi["rows"][0]["measurement"] == "S2_NU_AUDIT_PLUS_R7_CODER_RACE"
    assert voi["rows"][0]["voi_class"] == "REROUTING_PRE_BURN"
    # S2+R7 outrank the schedulable actions BECAUSE they can re-route the sealed burn.
    assert "RE-ROUTE" in voi["rows"][0]["value"]
    assert voi["rows"][1]["voi_class"] == "NOT_SCHEDULABLE_ARRIVES_WITH_BURN"
    assert "ASSUMED" in voi["priors_label"]
    # pn1 FEED-pn1f: attach under existing heads, never new heads.
    assert voi["no_new_heads"] is True
    assert voi["attach_under"]["S2_NU_AUDIT_PLUS_R7_CODER_RACE"].startswith("FIRST_GATES")
    assert voi["attach_under"]["S1_STAGE_A_LOCAL_FULL_N600"] == "BYTE_CLOSE_CHAIN_READY"


def test_nu_pivot_is_a_named_decision_node():
    pivot = _pn1_nodes(REPO, {"any_parent_in_band": False})["nu_pivot"]
    assert pivot["node_id"] == "nu_fiber_fraction_g4_feasibility_pivot"
    assert pivot["pivot_window"] == [0.55, 0.75]
    assert "(1 - nu) * h_vis <= 0.578" in pivot["feasibility_condition"]
    assert pivot["status"] == "UNMEASURED_REROUTING_CLASS_PRE_BURN"
    assert "T2 lotto dump" in pivot["measurement"]
    assert "(D,c,levels)" in pivot["decision_routed"]


def test_s1_rehearsal_stage_a_free_stage_b_operator_go():
    s1 = _pn1_nodes(REPO, {"any_parent_in_band": False})["s1_rehearsal"]
    stages = {row["duty"]: row for row in s1["stages"]}
    a = stages["S1_STAGE_A_LOCAL_FULL_N600"]
    assert a["cost"] == "$0" and a["status"] == "READY_QUIET_SLOT_ZERO_DOLLARS"
    assert a["actuation"] == "NONE"
    b = stages["S1_STAGE_B_MODAL_CPU_FLIGHT"]
    assert b["status"] == "OPERATOR_GO_PAID_DISPATCH"
    assert "<$2" in b["cost"] and "$20" in b["cost"]
    assert "dispatch_modal_paired_auth_eval" in b["spec"]
    assert s1["attach_under"] == "BYTE_CLOSE_CHAIN_READY"


def test_granularity_race_armed_pattern_flips_due_in_band():
    armed = _pn1_nodes(REPO, {"any_parent_in_band": False})["granularity_race_duty"]
    assert armed["status"] == "ARMED_NOT_DUE_BAND_STILL_EXPLODE"
    assert "[5e-4, 1e-2]" in armed["trigger"]
    assert "rv1 R4" in armed["instrument"]
    due = _pn1_nodes(REPO, {"any_parent_in_band": True})["granularity_race_duty"]
    assert due["status"] == "DUE"


def test_rebalance_watch_names_the_rerouting_events():
    watch = _pn1_nodes(REPO, {"any_parent_in_band": False})["rebalance_watch"]
    assert watch["status"] == "STANDING_WATCH"
    joined = " ".join(watch["events"])
    for token in ("nu measured", "R7 coder-race", "band-entry", "lv1", "Stage-A drift"):
        assert token in joined, token


def test_pn1_nodes_fail_open_without_memo(tmp_path):
    nodes = _pn1_nodes(tmp_path, {"any_parent_in_band": False})
    assert nodes["available"] is False
    assert nodes["reason"] == "PN1_COMMITTED_MEMO_ABSENT"


def test_sense_laws_gain_granularity_ladder_and_stay_backward_compatible():
    laws = _sense_laws(_arc_index(), pn1_source={"path": PN1_MEMO, "sha256": "0" * 64})
    rows = {row["law_id"]: row for row in laws["rows"]}
    # co8 laws stay backward-compatible; co9 appends three MEASURED laws (ng1/QA consumption).
    assert set(rows) == {
        "ema_gate_basis_v1",
        "basin_solve_handoff_v1",
        "correction_granularity_ladder_v1",
        "posenet_far_field_photometrics_bidirectional_v1",
        "token_sensitivity_spread_nu_pivot_v1",
        "seg_is_base_quality_white_jitter_v1",
    }
    ladder = rows["correction_granularity_ladder_v1"]
    assert ladder["kind"] == "DERIVED_LAW"
    assert "rung inversion" in ladder["falsifier"]
    assert "CONTINGENT" in ladder["race_posture"]
    assert ladder["source"]["path"] == PN1_MEMO
    # co7 call shape still works (no pn1 source) and the ladder fails open unsourced.
    bare = _sense_laws({})
    bare_rows = {row["law_id"]: row for row in bare["rows"]}
    assert bare_rows["correction_granularity_ladder_v1"]["source"] is None


# ── rung 3: allocator waterfill + CO5 adjudication ────────────────────────────
def test_allocator_waterfill_ranks_priced_rows_and_honors_pools_law():
    pn1 = _pn1_nodes(REPO, {"any_parent_in_band": False})
    rv1 = _rv1_conditional_validity_table(REPO)
    alloc = _duty_allocator_waterfill(pn1, rv1)
    assert alloc["available"] is True
    duties = [row["duty"] for row in alloc["rows"]]
    assert duties[0] == "S2_NU_AUDIT" and duties[1] == "R7_CODER_RACE"
    rows = {row["duty"]: row for row in alloc["rows"]}
    # Pools law: S2 and R7 share the token-rate pool -> compete, R7 depends on S2.
    assert rows["S2_NU_AUDIT"]["pool"] == rows["R7_CODER_RACE"]["pool"] == "token_rate_axis"
    assert "S2_NU_AUDIT" in rows["R7_CODER_RACE"]["depends_on"]
    # Paid + contingent rows are NOT organ-schedulable.
    assert rows["S1_STAGE_B"]["schedulable"] is False
    assert rows["GRANULARITY_LADDER_RACE"]["schedulable"] is False
    # Every row labels its pricing basis (DERIVED vs ASSUMED vs CONTINGENT).
    for row in alloc["rows"]:
        assert any(
            row["pricing_basis"].startswith(tag)
            for tag in ("DERIVED", "ASSUMED", "CONTINGENT")
        ), row["duty"]
    assert "GATED_RE_PREMISE" in alloc["co5_regret_allocator"]


def test_allocator_fails_open_without_sources(tmp_path):
    alloc = _duty_allocator_waterfill({"available": False}, {"available": False})
    assert alloc["available"] is False


def test_co5_adjudication_pn1_voi_is_not_the_named_producer():
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
    # HONEST adjudication: the VOI table has ZERO exact n600 S-endpoints -> keep gated.
    assert state["re_premised_count"] == 4 and state["active_count"] == 0
    assert "co8" in state["re_checked"]
    assert "NOT the named CT1-v2-class producer" in state["re_checked"]
    assert "co7" in state["re_checked"]  # append-only history preserved


# ── rung 6: organ-freshness gate ──────────────────────────────────────────────
def test_consumed_registry_derived_from_live_module_constants():
    reg = consumed_evidence_registry()
    assert reg["schema"] == "ddm_costate_consumed_evidence_registry.v1"
    assert RV1_TABLE_MEMO in reg["paths"]
    assert PN1_MEMO in reg["paths"]
    assert T3_SEALED_TICKET in reg["paths"]
    # arc artifacts + campaign sources are present (derived, not hand-listed).
    assert ".omx/research/ddm_fd1_family_d_gn_description_engine_20260728.md" in reg["paths"]
    assert any("ddm_lv1_*" in g for g in reg["globs"])
    assert any("ddm_rv1_*" in g for g in reg["globs"])


def _gate_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "organ_freshness_gate", REPO / "tools" / "organ_freshness_gate.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_freshness_threshold_derived_never_hardcoded():
    gate = _gate_module()
    # Insufficient history -> floor with recorded basis.
    th = gate.derive_threshold([100], [])
    assert th["n"] == 2 and th["basis"] == "FLOOR_DEFAULT_INSUFFICIENT_ROUND_HISTORY"
    # Measured history: 2 windows x {6, 8} landings -> mean 7 -> N = ceil(7/2) = 4.
    th = gate.derive_threshold(
        [0, 10, 20],
        [1] * 6 + [11] * 8,
    )
    assert th["n"] == 4 and th["windows"] == 2 and th["mean_landings_per_round"] == 7.0
    # Duplicate round commit-times (marker-file clusters) are deduped, not zero-width.
    th_dup = gate.derive_threshold([0, 0, 10, 10, 20], [1] * 6 + [11] * 8)
    assert th_dup["windows"] == 2 and th_dup["n"] == 4


def test_freshness_candidate_and_rerouting_classification(tmp_path):
    gate = _gate_module()
    assert gate._is_candidate_landing(".omx/research/ddm_new_arm_20260729.md")
    assert not gate._is_candidate_landing(".omx/research/ddm_co9_organ_round9_20260729.md")
    assert not gate._is_candidate_landing(".omx/research/ddm_new_arm_DAG_FEED_20260729.md")
    assert not gate._is_candidate_landing(".omx/research/sub/ddm_nested_20260729.md")
    assert not gate._is_candidate_landing(".omx/research/other_memo_20260729.md")
    memo = tmp_path / "m.md"
    memo.write_text("---\nverdict: SETTLED\n---\nVOI ranking with a precondition trigger\n")
    markers = gate._rerouting_markers(memo)
    assert "frontmatter_verdict" in markers and "voi" in markers and "precondition_trigger" in markers
    plain = tmp_path / "p.md"
    plain.write_text("# a plain landing memo with no routing content\n")
    assert gate._rerouting_markers(plain) == []


def test_freshness_consumed_matching_paths_and_globs():
    gate = _gate_module()
    reg = {"paths": [RV1_TABLE_MEMO], "globs": [".omx/research/ddm_lv1_*"]}
    assert gate._is_consumed(RV1_TABLE_MEMO, reg)
    assert gate._is_consumed(".omx/research/ddm_lv1_receipt_20260729.md", reg)
    assert not gate._is_consumed(".omx/research/ddm_unknown_arm_20260729.md", reg)


def test_freshness_gate_live_run_is_fail_open_and_rv1_pn1_consumed():
    gate = _gate_module()
    report = gate.evaluate(REPO)
    # rv1 + pn1 landed after co7 but are registered-consumed by co8 -> never unconsumed.
    assert RV1_TABLE_MEMO not in report["unconsumed"]
    assert PN1_MEMO not in report["unconsumed"]
    assert report["threshold"]["n"] >= 2
    assert report["actuation"] == "NONE" and report["score_claim"] is False


def test_freshness_hook_invocation_always_exits_zero():
    out = subprocess.run(
        [sys.executable, str(REPO / "tools" / "organ_freshness_gate.py"), "--quiet-ok"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
        env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"},
    )
    assert out.returncode == 0, out.stderr


def test_freshness_hook_wired_warn_only_in_settings():
    settings = json.loads((REPO / ".claude" / "settings.json").read_text())
    stop_cmds = [
        hook["command"]
        for group in settings["hooks"]["Stop"]
        for hook in group["hooks"]
    ]
    matches = [cmd for cmd in stop_cmds if "organ_freshness_gate.py" in cmd]
    assert len(matches) == 1
    assert "--quiet-ok" in matches[0]  # WARN-only, silent unless due


# ── report + digest wiring ────────────────────────────────────────────────────
def test_report_and_digest_surface_the_co8_nodes():
    from tac.ddm_costate_organ import build_live_ddm_costate, digest_lines

    try:
        report = build_live_ddm_costate(repo_root=REPO)
    except Exception as exc:
        pytest.skip(f"live DDM fleet unavailable: {type(exc).__name__}: {exc}")
    if not report.get("available"):
        pytest.skip(f"live DDM fleet incomplete: {report.get('missing_required')}")
    for key in ("rv1_table", "pn1_nodes", "duty_allocator"):
        assert key in report, key
    lines = digest_lines(report)
    joined = "\n".join(lines)
    assert "DDM-rv1[consumed]: reactivations=8" in joined
    assert "closed=12" in joined
    assert "DDM-voi[pn1]: top=S2_NU_AUDIT+R7" in joined
    assert "nu-pivot=[0.55, 0.75]" in joined
    assert "DDM-alloc[waterfill]:" in joined
    assert "regret-allocator=GATED(CO5)" in joined
    assert "table-owner=rv1[consumed-co8]" in joined
    # No contest-authority tag on any organ line (pointer honesty is section_pointer's).
    assert "[contest-CPU]" not in joined and "[contest-CUDA]" not in joined


def test_band_parents_unchanged_burn_chain_intact():
    # rung 4 boundary: the B-verdict/burn chain is UNCHANGED by the band re-parent.
    bp = _band_position(REPO)
    if not bp.get("available"):
        pytest.skip(f"live-base receipt absent: {bp.get('reason')}")
    parents = _band_position_parents(REPO, bp)
    # PREMISE FLIP (co9, ng1 §2 row 10): the tb1 burn endpoint (0.00389) entered the band, so
    # any_parent_in_band is now True. The invariant this test guards is that the ENDGAME/BURN
    # CHAIN head is unaffected by the correction-band re-parent — that still holds.
    #   was: assert parents["any_parent_in_band"] is False
    assert parents["any_parent_in_band"] is True
    from tac.ddm_costate_organ import _endgame_chain_duties

    chain = _endgame_chain_duties(_arc_index(), _pending_producers(REPO), parents)
    assert chain["head_actionable"] == "T3_BURN_FIRE"
    assert [row["duty"] for row in chain["chain"]] == [
        "B_VERDICT_WATCH",
        "T3_BURN_FIRE",
        "FIRST_GATES",
        "T1_VALIDITY_GATE",
        "BYTE_CLOSE_CHAIN_READY",
    ]
