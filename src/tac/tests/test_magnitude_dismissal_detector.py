"""Tests for the magnitude-dismissal detector Stop hook + its confound-gate sister.

Covers the pure classifier (positive / relative-sig-exempt / measured-un-recoverable-
exempt / non-dismissal-exempt / waiver / opt-out), the fmtools fail-open surface, the
sister preflight gate (tac.confound_gates.check_no_unjustified_magnitude_dismissal), and
an integration smoke proving the real hook exits 0 (fail-open) on the live repo.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys

_REPO = pathlib.Path(__file__).resolve().parents[3]
_TOOL = _REPO / "tools" / "magnitude_dismissal_detector.py"


def _load():
    spec = importlib.util.spec_from_file_location("magnitude_dismissal_detector", _TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


D = _load()


# ------------------------------- POSITIVE (catches the bug) -----------------------------
def test_positive_defer_lever_as_weak():
    lines = ["We defer horizon-margin #169 because its measured ΔS is weak."]
    cands = D.magnitude_dismissal_candidates(lines)
    assert len(cands) == 1
    assert D.deterministic_flags(lines, source="memo.md")


def test_positive_downgrade_as_negligible():
    lines = ["Downgrade the lane-orbit lever to WEAK: negligible impact, not worth it."]
    assert D.magnitude_dismissal_candidates(lines)


def test_positive_orphan_as_noise():
    lines = ["Orphan the chroma lever — the effect is just noise."]
    assert D.magnitude_dismissal_candidates(lines)


def test_positive_kill_small_delta():
    lines = ["KILL #212: too small a gain to bother with."]
    assert D.magnitude_dismissal_candidates(lines)


def test_positive_across_two_lines_window():
    # dismissal on one line, magnitude on the next — the 3-line window catches it.
    lines = ["We should shelve this lever.", "Its improvement is negligible."]
    assert D.magnitude_dismissal_candidates(lines)


def test_positive_causal_wrap_across_two_lines():
    lines = ["We defer the lever because its measured", "delta-S effect is weak."]
    assert D.magnitude_dismissal_candidates(lines)


def test_positive_magnitude_led_continuation_across_two_lines():
    lines = ["We defer the lever.", "Negligible impact at this operating point."]
    assert D.magnitude_dismissal_candidates(lines)


def test_positive_same_table_row_still_refuses():
    lines = ["| lever | DEFER because its effect is negligible |"]
    assert D.magnitude_dismissal_candidates(lines)


# ------------------------------- NEGATIVE (a): relative significance ---------------------
def test_negative_relative_significance_stated():
    lines = ["Defer #169? Its ΔS 0.012–0.024 is ~13–27% of the remaining gap to target — "
             "keep it (relative significance is significant)."]
    assert D.magnitude_dismissal_candidates(lines) == []


def test_negative_fraction_of_remaining_descent():
    lines = ["This lever looks weak in absolute terms but buys a large fraction of the "
             "remaining descent, so we do NOT defer it."]
    assert D.magnitude_dismissal_candidates(lines) == []


# ------------------------------- NEGATIVE (b): measured un-recoverable -------------------
def test_negative_measured_unrecoverable():
    lines = ["Defer the dash-recovery lever: the residual is un-recoverable (measured), "
             "at the information floor — no term can predict it."]
    assert D.magnitude_dismissal_candidates(lines) == []


def test_negative_structurally_superseded():
    lines = ["Downgrade this to weak — it is structurally superseded by the directional "
             "basis lever."]
    assert D.magnitude_dismissal_candidates(lines) == []


def test_negative_label_noise_with_measurement():
    lines = ["Defer: the residual is label-noise, measured un-recoverable at n600 "
             "(exit criterion met)."]
    assert D.magnitude_dismissal_candidates(lines) == []


# ------------------------------- NEGATIVE: non-dismissal magnitude usages ----------------
def test_negative_weak_supervision():
    lines = ["We drop the auxiliary head under weak supervision noise conditions."]
    assert D.magnitude_dismissal_candidates(lines) == []


def test_negative_noise_floor_phrase():
    lines = ["The residual sits at the noise floor, so we shelve further tuning."]
    # "noise floor" is a measured-un-recoverability cue → exempt.
    assert D.magnitude_dismissal_candidates(lines) == []


def test_negative_noise_injection_mechanism():
    lines = ["Sigma noise injection weakly regularizes; we abandon the schedule tweak."]
    assert D.magnitude_dismissal_candidates(lines) == []


def test_negative_plain_prose_no_cooccurrence():
    lines = ["The lane markings are weak in this frame.",
             "Separately, we launched the n600 run."]
    # magnitude word but no dismissal decision in the same window.
    assert D.magnitude_dismissal_candidates(lines) == []


def test_negative_provenance_class_cannot_borrow_from_other_table_row():
    lines = [
        "| safety bound | ORPHAN-LITERAL | unowned extension |",
        "| schedule | IMPORTED | same-object gate exists |",
        "| stop predicate | DERIVED-IN-PLACE | one flip, marginal, objective |",
    ]
    assert D.magnitude_dismissal_candidates(lines) == []


def test_negative_live_m1r5c_review_passes_without_waiver():
    review = _REPO / ".omx" / "research" / "ddm_m1r4_20260808" / "M1R5C_REVIEW.md"
    assert D.deterministic_flags(review.read_text().splitlines(), source=str(review)) == []


def test_negative_source_inventory_does_not_borrow_label_noise():
    lines = [
        "- Deep-era bounded corpus: evaluator-inverse orphan inventory; tasks #52-#59,",
        "  #66/#67 and #150-#158; B-WITNESS #95/#96; #141 label-noise memo;",
        "  survival/capacity-wall and later generator-power-law corrections.",
    ]
    assert D.magnitude_dismissal_candidates(lines) == []


# ------------------------------- waiver + opt-out ---------------------------------------
def test_waiver_respected():
    lines = ["Defer #169 as weak.  # MAGNITUDE_DISMISSAL_OK: council ratified, "
             "duplicate of #212 already re-ranked"]
    assert D.magnitude_dismissal_candidates(lines) == []


def test_waiver_placeholder_rejected():
    lines = ["Defer #169 as weak.  # MAGNITUDE_DISMISSAL_OK: <rationale>"]
    # placeholder rationale must NOT exempt.
    assert D.magnitude_dismissal_candidates(lines)


def test_opt_out_token_window_wide():
    assert D.is_opted_out(["chore: prune stale memos [magnitude-ok]"]) is True
    assert D.is_opted_out(["chore: prune stale memos [skip-magnitude]"]) is True
    assert D.is_opted_out(["witness: defer #169 as weak"]) is False


def test_discussion_cue_exempt():
    lines = ["Reminder: never dismiss a lever as weak on absolute ΔS — re-rank at the "
             "current operating point instead."]
    assert D.magnitude_dismissal_candidates(lines) == []


# ------------------------------- fmtools fail-open --------------------------------------
def test_fm_confirm_absent_returns_not_ran(monkeypatch):
    monkeypatch.setenv("MAGNITUDE_DISMISSAL_FM_PYTHON", "/nonexistent/python")
    monkeypatch.setenv("DASH_FM_PYTHON", "/nonexistent/python")
    monkeypatch.setattr(D.os.path, "expanduser", lambda p: "/nonexistent/python")
    ids, ran = D.fm_confirm([{"id": "c0", "passage": "defer x as weak"}])
    assert ids == [] and ran is False


def test_fm_confirm_empty_candidates():
    assert D.fm_confirm([]) == ([], False)


def test_build_reason_labels_owed_when_fm_absent():
    r = D.build_reason(["memo.md:3: ... — \"defer as weak\""], fm_ran=False)
    assert "confirmation OWED" in r or "confirmation\nOWED" in r or "OWED" in r
    r2 = D.build_reason(["memo.md:3: ..."], fm_ran=True)
    assert "OWED" not in r2


# ------------------------------- integration smoke (fail-open) --------------------------
def test_hook_exits_zero_on_live_repo():
    proc = subprocess.run(
        [sys.executable, str(_TOOL)],
        input=json.dumps({"cwd": str(_REPO), "stop_hook_active": True}),
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0


def test_hook_fail_open_on_garbage_stdin():
    proc = subprocess.run(
        [sys.executable, str(_TOOL)], input="not json at all",
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0


# ------------------------------- sister preflight gate ----------------------------------
def _gate():
    from tac.confound_gates import check_no_unjustified_magnitude_dismissal
    return check_no_unjustified_magnitude_dismissal


def test_preflight_gate_catches_crafted_memo(tmp_path):
    research = tmp_path / ".omx" / "research"
    research.mkdir(parents=True)
    (research / "verdict_x_20260708.md").write_text(
        "# Verdict\n\nWe defer lever #169 because its ΔS is weak and negligible.\n")
    viol = _gate()(repo_root=tmp_path, strict=False, verbose=False)
    assert viol, "gate should flag the crafted unjustified magnitude-dismissal"


def test_preflight_gate_clean_on_relative_sig(tmp_path):
    research = tmp_path / ".omx" / "research"
    research.mkdir(parents=True)
    (research / "verdict_ok_20260708.md").write_text(
        "# Verdict\n\nLever #169 ΔS 0.02 is ~half the remaining gap to target — keep.\n")
    assert _gate()(repo_root=tmp_path, strict=False, verbose=False) == []


def test_preflight_gate_respects_waiver(tmp_path):
    research = tmp_path / ".omx" / "research"
    research.mkdir(parents=True)
    (research / "verdict_waived_20260708.md").write_text(
        "# Verdict\n\nDefer #169 as weak.  # MAGNITUDE_DISMISSAL_OK: duplicate of "
        "#212 already re-ranked at operating point\n")
    assert _gate()(repo_root=tmp_path, strict=False, verbose=False) == []


def test_preflight_gate_is_warn_only_in_confound_registry():
    # #404 must NOT be in the strict set (warn-only until the historical re-audit).
    from tac.confound_gates import CONFOUND_GATES, check_no_unjustified_magnitude_dismissal
    assert check_no_unjustified_magnitude_dismissal in CONFOUND_GATES


def test_preflight_gate_strict_raises_on_violation(tmp_path):
    from tac.preflight import PreflightError
    research = tmp_path / ".omx" / "research"
    research.mkdir(parents=True)
    (research / "verdict_strict_20260708.md").write_text(
        "# Verdict\n\nOrphan the chroma lever — negligible, not worth it.\n")
    try:
        _gate()(repo_root=tmp_path, strict=True, verbose=False)
        raised = False
    except PreflightError:
        raised = True
    assert raised
