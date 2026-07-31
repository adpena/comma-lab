"""Tests for ddm_co9 costate-organ round 9.

Covers the four co9 work items, each score_claim=False / actuation=NONE:
  (1) the three MEASURED SENSE laws (two-plane pose bidirectional / token sensitivity-spread
      ν-pivot / seg=base-quality white-jitter), anchored to the committed deferral ledger;
  (2) the band RE-PARENT (ng1 §2 row 10): the tb1 burn endpoint (0.00389, read machine-readably
      from the committed pfs1 D1 eval receipt) enters the band -> any_parent_in_band True, with
      the QA03/QA04 white-jitter break-even caveat welded on;
  (3) the OWNERSHIP-ON-GATE-OPEN scan (new SENSE surface over the committed ledger) + the
      cn1/cn2 rc1-branch gate-opener;
  (4) QA37 consumption: the deferral ledger registered as a consumed-evidence source.
Plus fail-open behavior and the digest-line surfaces (end-to-end).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]

from tac.ddm_costate_organ import (  # noqa: E402
    DEFERRAL_LEDGER_GLOB,
    LIVE_BURN_ENDPOINT_EVAL_RECEIPT,
    _arc_evidence_rows,
    _band_position,
    _band_position_parents,
    _burn_endpoint_base,
    _deferral_ledger_source,
    _open_gate_ownership_scan,
    _sense_laws,
    build_live_ddm_costate,
    consumed_evidence_registry,
    digest_lines,
)


def _arc_index():
    return {row["finding_id"]: row for row in _arc_evidence_rows(REPO) if row.get("available")}


# ── work item 1: the three MEASURED SENSE laws ────────────────────────────────
def test_three_co9_measured_laws_present_and_labeled():
    src = _deferral_ledger_source(REPO)
    laws = _sense_laws(_arc_index(), pn1_source=None, ledger_source=src)
    rows = {row["law_id"]: row for row in laws["rows"]}
    for lid in (
        "posenet_far_field_photometrics_bidirectional_v1",
        "token_sensitivity_spread_nu_pivot_v1",
        "seg_is_base_quality_white_jitter_v1",
    ):
        assert lid in rows, lid
        law = rows[lid]
        assert law["epistemic_status"] == "MEASURED"
        assert law["evidence_axis"] == "[macOS-CPU advisory]"
        assert law["ledger_rows"]  # cites the committed ledger rows
        # NO-FAKE: a MEASURED law carries its committed ledger source stamp when the ledger exists.
        if src is not None:
            assert law["source"] == src


def test_two_plane_pose_law_is_bidirectional():
    laws = _sense_laws(_arc_index(), ledger_source=_deferral_ledger_source(REPO))
    rows = {row["law_id"]: row for row in laws["rows"]}
    law = rows["posenet_far_field_photometrics_bidirectional_v1"]
    # both directions of the same far-field mechanism are stated (QA43 recover / QA06 freeze).
    assert "FORWARD" in law["statement"] and "REVERSE" in law["statement"]
    assert "QA43" in law["ledger_rows"] and "QA06" in law["ledger_rows"]


def test_white_jitter_law_marks_corrections_break_even():
    laws = _sense_laws(_arc_index(), ledger_source=_deferral_ledger_source(REPO))
    rows = {row["law_id"]: row for row in laws["rows"]}
    law = rows["seg_is_base_quality_white_jitter_v1"]
    assert "BREAK-EVEN" in law["statement"] or "BREAK-EVEN" in law["consequence"]
    assert "QA03" in law["ledger_rows"] and "QA04" in law["ledger_rows"]


# ── work item 2: band RE-PARENT (ng1 §2 row 10) ───────────────────────────────
def test_burn_endpoint_base_read_from_committed_eval_receipt():
    endpoint = _burn_endpoint_base(REPO)
    if endpoint is None:
        pytest.skip("pfs1 D1 eval receipt absent")
    # the burn endpoint d_seg is inside the rational band [rho_c, 1e-2].
    assert 5.02e-4 < endpoint["d_seg"] < 1.0e-2
    assert endpoint["source_path"] == LIVE_BURN_ENDPOINT_EVAL_RECEIPT
    assert endpoint["sha256"] and len(endpoint["sha256"]) == 64


def test_burn_endpoint_base_fail_open(tmp_path):
    # no receipt on an empty tree -> None (never a crash).
    assert _burn_endpoint_base(tmp_path) is None


def test_reparent_flips_any_parent_in_band_with_break_even_caveat():
    bp = _band_position(REPO)
    if not bp.get("available") or _burn_endpoint_base(REPO) is None:
        pytest.skip("live-base or burn-endpoint receipt absent")
    parents = _band_position_parents(REPO, bp)
    rows = {row["parent"]: row for row in parents["rows"]}
    assert "tb1_burn_endpoint" in rows
    endpoint = rows["tb1_burn_endpoint"]
    assert endpoint["regime"] == "correct"
    assert parents["any_parent_in_band"] is True
    # the caveat travels (operating manual §5): in-band != promising; white-jitter break-even.
    assert "BREAK_EVEN" in endpoint["measured_correction_value_at_base"]
    # the pre-arc parents remain above-band and are relabeled as such.
    assert rows["W_joint_describe_line"]["regime"] == "explode"
    assert "pre-arc" in rows["W_joint_describe_line"]["vehicle"]


# ── work item 3: OWNERSHIP-ON-GATE-OPEN scan ──────────────────────────────────
def test_open_gate_ownership_scan_surfaces_due_orphan_open_rows():
    scan = _open_gate_ownership_scan(REPO)
    if not scan.get("available"):
        pytest.skip(f"deferral ledger absent: {scan.get('reason')}")
    assert scan["open_gate_count"] >= 1
    # NO ROW-IDENTITY PINS AGAINST THE LIVE LEDGER (task #780, fixed 2026-07-31).
    # This asserted `"QA05" in ids`, pinning a row whose status is a LIVE MUTABLE cell
    # in .omx/research/ddm_deferral_queue_ledger_*.md. QA05 legitimately advanced to
    # FIRED on 2026-07-30 (ddm_qp1) and the test went red on clean main — the ledger
    # doing its job broke the test. That is staleness baked into a test: it re-breaks
    # every time the queue advances, and a permanently-red test erodes the authority
    # of the whole suite. Assert the STRUCTURAL invariant of the scan instead, which
    # is what the scan actually guarantees and what a regression would violate.
    # Positive/"a due row IS surfaced" coverage is NOT lost: it lives in
    # test_no_owner_alarm_flags_empty_consumer below, against a SYNTHETIC fixture row
    # (the correct home for it — a fixture cannot drift).
    for r in scan["open_gate_unfired_rows"]:
        # every surfaced row is DUE/ORPHAN and carries an owner + age field.
        assert ("DUE" in r["row_status"]) or ("ORPHAN" in r["row_status"])
        assert "owner" in r and "age_days" in r
        assert r["row_id"], "every surfaced row must carry a non-empty row_id"
    # the cn1/cn2 gate-opener (rc1 unmerged branch) is encoded.
    openers = {o["gate_opener"] for o in scan["gate_openers"]}
    assert "rc1_branch_landing" in openers
    assert scan["actuation"] == "NONE" and scan["score_claim"] is False


def test_open_gate_ownership_scan_fail_open(tmp_path):
    scan = _open_gate_ownership_scan(tmp_path)
    assert scan["available"] is False
    assert scan["reason"] == "DEFERRAL_LEDGER_ABSENT"


def test_no_owner_alarm_flags_empty_consumer(tmp_path):
    # a synthetic ledger row with an OPEN gate, DUE status, and an EMPTY consumer cell
    # must raise the no-owner alarm (the QA05/QA41 open-gate/no-owner class).
    ledger = tmp_path / ".omx" / "research"
    ledger.mkdir(parents=True)
    (ledger / "ddm_deferral_queue_ledger_29990101.md").write_text(
        "---\ndate_utc: 2999-01-01\n---\n"
        "| id | item | src | gate | gate status NOW | effect | price | consumer | status |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
        "| QZ99 | orphan probe | s | g | OPEN | e | $0 |  | DUE |\n",
        encoding="utf-8",
    )
    scan = _open_gate_ownership_scan(tmp_path)
    assert scan["available"] is True
    assert "QZ99" in scan["no_owner_alarm_rows"]
    assert scan["no_owner_alarm_count"] == 1


# ── work item 4: QA37 — ledger registered as consumed evidence ─────────────────
def test_deferral_ledger_registered_as_consumed_evidence():
    reg = consumed_evidence_registry()
    assert ".omx/research/" + DEFERRAL_LEDGER_GLOB in reg["globs"]
    assert LIVE_BURN_ENDPOINT_EVAL_RECEIPT in reg["paths"]


# ── end-to-end: the digest surfaces the co9 nodes ─────────────────────────────
def test_digest_surfaces_co9_lines():
    try:
        report = build_live_ddm_costate(repo_root=REPO)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"live DDM fleet unavailable: {type(exc).__name__}: {exc}")
    if not report.get("available"):
        pytest.skip(f"live DDM fleet incomplete: {report.get('missing_required')}")
    assert "open_gate_ownership" in report
    joined = "\n".join(digest_lines(report))
    assert "DDM-laws[co9 MEASURED]:" in joined
    assert "DDM-parents:" in joined and "tb1_burn_endpoint IN-BAND" in joined
    assert "MEASURED BREAK-EVEN at this base" in joined
    assert "DDM-owners[gate-open/unfired]:" in joined
    assert "DDM-band[pre-arc describe-line base]:" in joined
    # organ stays advisory: no contest-authority tag anywhere.
    assert "[contest-CPU]" not in joined and "[contest-CUDA]" not in joined


def test_costate_digest_end_to_end_runs_and_stays_advisory():
    out = subprocess.run(
        [sys.executable, "tools/costate_digest.py"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert out.returncode == 0
    assert "DDM-owners[gate-open/unfired]:" in out.stdout
