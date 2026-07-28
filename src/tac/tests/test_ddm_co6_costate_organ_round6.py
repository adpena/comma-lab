"""Tests for ddm_co6 costate-organ round 6.

Covers the 2026-07-28 arc-evidence join, the band-position SENSE law wiring, the
refreshed-duty derivation, the CO5 per-enhancement re-adjudication, and the honest
digest pointer line. Every advisory surface here is score_claim=False / actuation=NONE.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]

from tac.ddm_campaign_costate import (  # noqa: E402
    _CO5_ENHANCEMENT_GATES,
    _co5_disposition,
)
from tac.ddm_costate_organ import (  # noqa: E402
    ARC_EVIDENCE_SPECS,
    _arc_evidence_rows,
    _band_position,
    _refreshed_duties,
)


# ── R1: arc-evidence join ──────────────────────────────────────────────────────
def test_arc_evidence_rows_present_and_content_hashed():
    rows = _arc_evidence_rows(REPO)
    assert len(rows) == len(ARC_EVIDENCE_SPECS) == 8
    for row in rows:
        # Every committed 07-28 artifact is git-tracked, so all rows must be present.
        assert row["available"] is True, f"{row['finding_id']}: {row.get('reason')}"
        assert len(row["sha256"]) == 64
        assert row["actuation"] == "NONE"
        assert row["score_claim"] is False
        assert row["crux_status"] in {"SETTLED", "LIVE_CRUX", "SOLVED", "LAW", "APPARATUS"}
    ids = {row["finding_id"] for row in rows}
    assert {"fd1_zero_accept_window", "pp1_band_lemma", "rp1_cells_hold", "sc1_ep_rank1_pose"} <= ids


def test_arc_evidence_absent_artifact_is_fail_open(tmp_path):
    rows = _arc_evidence_rows(tmp_path)  # empty dir -> no committed artifacts
    assert all(row["available"] is False for row in rows)
    assert all(row["reason"] == "COMMITTED_ARTIFACT_ABSENT" for row in rows)


def test_no_uncommitted_fd2_or_quantum_cure_row():
    # NO-FAKE: fd2 has no committed artifact and "quantum cure" appears in none; the arc join
    # must ground only committed evidence (the fd1 realization ladder), never those.
    blob = json.dumps([spec.__dict__ for spec in ARC_EVIDENCE_SPECS]).lower()
    assert "quantum cure" not in blob
    assert "fd2" not in blob  # no fd2 arm cited as committed evidence


# ── R2: band-position SENSE law ────────────────────────────────────────────────
def test_band_position_places_live_base_above_band():
    bp = _band_position(REPO)
    if not bp.get("available"):
        pytest.skip(f"live-base receipt absent: {bp.get('reason')}")
    # The live base d_seg (~0.0705) is far above the band upper (1e-2) -> corrections explode.
    assert bp["regime"] == "explode"
    assert bp["base_d_seg"] > bp["band_upper"]
    assert bp["correction_duty_multiplier"] == 0.0
    assert bp["correction_class_regime"].startswith("ABOVE_BAND")
    assert bp["equation_id"] == "ddm_pp1_correction_stream_position_band_v1"
    assert bp["actuation"] == "NONE" and bp["score_claim"] is False
    assert len(bp["source"]["sha256"]) == 64


def test_band_position_regimes_are_the_registered_law():
    # Sanity: the multiplier is 0 outside the rational band, 1 inside — matching the equation.
    from tac.canonical_equations.ddm_pp1_correction_stream_position_band_20260728 import (
        position_cost_band,
    )

    assert position_cost_band(3e-4)["regime"] == "concede"  # below rho_c
    assert position_cost_band(5e-3)["regime"] == "correct"  # in band
    assert position_cost_band(7e-2)["regime"] == "explode"  # above upper


# ── R3: refreshed-duty derivation ──────────────────────────────────────────────
def _legacy_stub():
    return {
        "live_ranked": [
            {"rank": 1, "duty": "J_paint", "block_id": "j_paint_dv1_persistent_ground"},
            {"rank": 2, "duty": "R6_rehearsal"},
            {"rank": 3, "duty": "DDM_iteration_curves"},
        ],
        "current_legacy_rows_retained": 115,
    }


def test_refreshed_duty_head_displaces_j_paint():
    arc = _arc_evidence_rows(REPO)
    bp = _band_position(REPO)
    if not bp.get("available"):
        pytest.skip("live-base receipt absent")
    refreshed = _refreshed_duties(_legacy_stub(), arc, bp)
    ranked = refreshed["live_ranked"]
    assert ranked[0]["duty"] == "FD1_REALIZATION_LADDER_MATERIALIZATION"
    assert ranked[0]["rank"] == 1
    assert refreshed["superseded_legacy_head"] == "J_paint"
    dispositions = {d["duty"]: d["disposition"] for d in refreshed["demoted"]}
    assert any("cell-space" in k for k in dispositions)
    assert any(v.startswith("BAND_DEAD") for v in dispositions.values())
    for row in ranked:
        assert row["actuation"] == "NONE"


def test_refreshed_duty_not_hand_ordered_reacts_to_band():
    # If the base were in-band (corrections rational) AND zero-accept not sealed, the ladder is
    # not elevated -> the derivation is a real function of the inputs, not a fixed order.
    arc_no_zero_accept = [r for r in _arc_evidence_rows(REPO) if r["finding_id"] != "fd1_zero_accept_window"]
    in_band = {
        "available": True,
        "regime": "correct",
        "correction_duty_multiplier": 1.0,
        "base_d_seg": 5e-3,
    }
    refreshed = _refreshed_duties(_legacy_stub(), arc_no_zero_accept, in_band)
    ladder = next(r for r in refreshed["live_ranked"] if r["kind"] == "capacity_ladder")
    assert ladder["priority"] == 1.0  # not elevated when base in-band + descent not exhausted


# ── R4: CO5 per-enhancement re-adjudication ────────────────────────────────────
def test_co5_disposition_derivation():
    assert _co5_disposition(True, "SOME_GATE")[0] == "ACTIVATE_ADVISORY"
    assert _co5_disposition(False, "SOME_GATE")[0] == "RE_PREMISE"
    assert _co5_disposition(False, None)[0] == "RETIRE"


def test_co5_gate_constant_covers_all_four_with_dependency():
    assert set(_CO5_ENHANCEMENT_GATES) == {
        "pontryagin_bellman_transition_residual",
        "m34_per_state_dual_consistency",
        "compression_progress_per_effort",
        "regret_bounded_duty_allocation",
    }
    # compression_progress unblocks regret (dependency topology drives the duty-to-measure rank).
    assert _CO5_ENHANCEMENT_GATES["compression_progress_per_effort"]["unblocks"] == (
        "regret_bounded_duty_allocation",
    )
    for spec in _CO5_ENHANCEMENT_GATES.values():
        assert spec["named_gate"] and spec["producer"]


def test_co5_enhancement_state_re_premised_not_silent():
    # Requires the full campaign source fleet (ct1 + ev1); skip if not present in this checkout.
    from tac.ddm_campaign_costate import _co5_enhancement_state, load_campaign_sources

    try:
        sources, payloads = load_campaign_sources(REPO)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"campaign source fleet unavailable: {type(exc).__name__}: {exc}")
    state = _co5_enhancement_state(
        ct1_payload=payloads["ct1_campaign_telemetry"],
        ct1_source=sources["ct1_campaign_telemetry"],
        evidence_join=payloads["ev1_campaign_evidence_join"],
        evidence_source=sources["ev1_campaign_evidence_join"],
    )
    assert state["status"] == "RE_PREMISED_TRACKED_QUEUE_4_NAMED_GATES"
    assert "PREMISE_FALSIFIED" not in state["status"]
    assert state["re_premised_count"] == 4
    assert state["retired_count"] == 0
    assert all(v == "RE_PREMISE" for v in state["dispositions"].values())
    # The duty-to-measure queue is a real, ranked, drainable queue.
    dtm = state["duty_to_measure"]
    assert dtm[0]["enhancement_id"] == "compression_progress_per_effort"  # unblocks the most
    assert [r["rank"] for r in dtm] == [1, 2, 3, 4]
    assert state["actuation"] == "NONE" and state["score_claim"] is False


# ── R5: honest digest pointer line ─────────────────────────────────────────────
def _load_digest_module():
    spec = importlib.util.spec_from_file_location("cdg_co6", REPO / "tools" / "costate_digest.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cdg_co6"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_pointer_labels_the_non_submission_bank(tmp_path):
    mod = _load_digest_module()
    ptr = tmp_path / "pointer.json"
    ptr.write_text(
        json.dumps(
            {
                "our_local_frontier_contest_cpu": {
                    "score": 0.1880443979880752,
                    "measured_at_utc": "2026-07-12T06:35:14+00:00",
                    "archive_sha256": "196acd18e4ca10a3ab0d826436aa46014a44cba8a55eb4abf9931876cc7e98b5",
                },
                "effective_frontier": {"score": 0.172},
            }
        )
    )
    mod._POINTER_JSON = ptr
    line, data = mod.section_pointer()
    assert "submittable 0.1910828" in line
    assert "NON-SUBMISSION bank" in line
    assert "effective bar 0.172" in line
    assert data["non_submission_bank"] is True
    assert data["score"] == pytest.approx(0.1910828242)
    assert data["effective_bar"] == 0.172


def test_pointer_reads_submittable_cpu_frontier_directly_when_not_bank(tmp_path):
    mod = _load_digest_module()
    ptr = tmp_path / "pointer.json"
    ptr.write_text(
        json.dumps(
            {
                "our_local_frontier_contest_cpu": {
                    "score": 0.1495,
                    "measured_at_utc": "2026-08-01T00:00:00+00:00",
                    "archive_sha256": "deadbeef" + "0" * 56,
                },
                "effective_frontier": {"score": 0.1495},
            }
        )
    )
    mod._POINTER_JSON = ptr
    line, data = mod.section_pointer()
    assert "0.14950 [contest-CPU] UNMOVED" in line
    assert data["non_submission_bank"] is False
    assert data["score"] == pytest.approx(0.1495)
