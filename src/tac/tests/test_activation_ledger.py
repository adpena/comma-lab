"""Tests for the lever ACTIVATION ledger (tac.witness_dsl.activation_ledger) — the "'off' is a tracked
queue" apparatus (#247 SENSE / CLAUDE.md orphaned-signal non-negotiable)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tac.witness_dsl import activation_ledger as al


@pytest.fixture
def ledger(tmp_path):
    return tmp_path / "lever_activation_ledger.jsonl"


# --- record / read ---------------------------------------------------------
def test_record_and_read_roundtrip(ledger):
    row = al.record_activation("EikonalViscosity", al.EVENT_FIRED, run_ref="run_x", path=ledger)
    assert row["lever"] == "EikonalViscosity" and row["event"] == "fired" and row["ts"]
    evs = al._read_events(ledger)
    assert len(evs) == 1 and evs[0]["run_ref"] == "run_x"


def test_invalid_event_raises(ledger):
    with pytest.raises(ValueError):
        al.record_activation("X", "bogus", path=ledger)


def test_folded_and_queued_events_require_reason(ledger):
    for event in (al.EVENT_FOLDED, al.EVENT_QUEUED, al.EVENT_RETIRED):
        with pytest.raises(ValueError, match="requires a non-empty reason"):
            al.record_activation("X", event, path=ledger)


def test_empty_lever_raises(ledger):
    with pytest.raises(ValueError):
        al.record_activation("", al.EVENT_FIRED, path=ledger)


def test_read_missing_file_is_empty(tmp_path):
    assert al._read_events(tmp_path / "nope.jsonl") == []


def test_read_skips_corrupt_lines(ledger):
    al.record_activation("A", al.EVENT_FIRED, path=ledger)
    with open(ledger, "a") as f:
        f.write("{not json\n\n")
    al.record_activation("B", al.EVENT_FIRED, path=ledger)
    assert {e["lever"] for e in al._read_events(ledger)} == {"A", "B"}


# --- state machine ---------------------------------------------------------
def test_state_never_fired_for_unknown(ledger):
    st = al.activation_status("SegFocalGamma", ledger)
    assert st.state == al.STATE_NEVER_FIRED and not st.ever_fired and not st.ever_measured


def test_state_fired_unmeasured(ledger):
    al.record_activation("SegFocalGamma", al.EVENT_FIRED, path=ledger)
    st = al.activation_status("SegFocalGamma", ledger)
    assert st.state == al.STATE_FIRED_UNMEASURED and st.ever_fired and not st.ever_measured
    assert st.n_fired == 1


def test_state_measured(ledger):
    al.record_activation("SegFocalGamma", al.EVENT_FIRED, path=ledger)
    al.record_activation("SegFocalGamma", al.EVENT_MEASURED, verdict_ref="v1", path=ledger)
    st = al.activation_status("SegFocalGamma", ledger)
    assert st.state == al.STATE_MEASURED and st.ever_measured and st.n_measured == 1


def test_retired_is_terminal(ledger):
    al.record_activation("SegFocalGamma", al.EVENT_FIRED, path=ledger)
    al.record_activation("SegFocalGamma", al.EVENT_MEASURED, path=ledger)
    al.record_activation("SegFocalGamma", al.EVENT_RETIRED, reason="dominated", path=ledger)
    st = al.activation_status("SegFocalGamma", ledger)
    assert st.state == al.STATE_RETIRED and st.retired


# --- known_levers / never_fired / duty_to_measure --------------------------
def test_known_levers_are_the_dsl_factories():
    known = al.known_levers()
    assert "EikonalViscosity" in known and "SegFocalGamma" in known and "BoundaryDistance" in known
    assert known == tuple(sorted(known))  # sorted, deterministic


def test_empty_ledger_all_known_are_never_fired(ledger):
    known = al.known_levers()
    nf = al.never_fired(known, path=ledger)
    assert set(nf) == set(known)  # honest: nothing fired via the DSL path yet


def test_never_fired_drops_a_fired_lever(ledger):
    known = ("A", "B", "C")
    al.record_activation("B", al.EVENT_FIRED, path=ledger)
    assert set(al.never_fired(known, path=ledger)) == {"A", "C"}


def test_never_fired_excludes_retired(ledger):
    known = ("A", "B")
    al.record_activation("A", al.EVENT_RETIRED, reason="x", path=ledger)
    assert set(al.never_fired(known, path=ledger)) == {"B"}


def test_duty_to_measure_includes_fired_unmeasured(ledger):
    known = ("A", "B", "C")
    al.record_activation("A", al.EVENT_FIRED, path=ledger)                 # fired, unmeasured -> owed
    al.record_activation("B", al.EVENT_FIRED, path=ledger)
    al.record_activation("B", al.EVENT_MEASURED, path=ledger)             # measured -> not owed
    owed = set(al.duty_to_measure(known, path=ledger))
    assert owed == {"A", "C"} and "B" not in owed


# --- CLOSE: levers_fired_for_run / record_measured_for_run ----------------
def test_levers_fired_for_run_exact_and_containment(ledger):
    al.record_activation("A", al.EVENT_FIRED, run_ref="/runs/exp_1", path=ledger)
    al.record_activation("B", al.EVENT_FIRED, run_ref="/runs/exp_1/checkpoints", path=ledger)
    al.record_activation("C", al.EVENT_FIRED, run_ref="/runs/exp_2", path=ledger)
    # ckpt-dir == the run dir: matches A (exact) + B (subdir contained by the run dir)
    got = set(al.levers_fired_for_run("/runs/exp_1", path=ledger))
    assert got == {"A", "B"} and "C" not in got


def test_levers_fired_for_run_no_prefix_false_match(ledger):
    al.record_activation("A", al.EVENT_FIRED, run_ref="/runs/exp_2", path=ledger)
    # /runs/exp_20 must NOT match /runs/exp_2 (component-wise, not raw substring)
    assert al.levers_fired_for_run("/runs/exp_20", path=ledger) == ()


def test_record_measured_for_run_drains_duty(ledger):
    known = ("A", "B", "C")
    al.record_activation("A", al.EVENT_FIRED, run_ref="/runs/exp_1", path=ledger)
    al.record_activation("B", al.EVENT_FIRED, run_ref="/runs/exp_1", path=ledger)
    # C never fired for this run
    assert set(al.duty_to_measure(known, path=ledger)) == {"A", "B", "C"}
    rows = al.record_measured_for_run("/runs/exp_1", verdict_ref="v.json", path=ledger)
    assert {r["lever"] for r in rows} == {"A", "B"}
    # A, B now measured (drained); C still owed (never fired)
    assert set(al.duty_to_measure(known, path=ledger)) == {"C"}
    assert al.activation_status("A", ledger).state == al.STATE_MEASURED


def test_record_measured_for_run_idempotent(ledger):
    al.record_activation("A", al.EVENT_FIRED, run_ref="/runs/exp_1", path=ledger)
    al.record_measured_for_run("/runs/exp_1", path=ledger)
    # second call records nothing (A already measured)
    assert al.record_measured_for_run("/runs/exp_1", path=ledger) == ()


# --- report ordering -------------------------------------------------------
def test_activation_report_surfaces_never_fired_first(ledger):
    known = ("A", "B", "C")
    al.record_activation("A", al.EVENT_FIRED, path=ledger)
    al.record_activation("A", al.EVENT_MEASURED, path=ledger)   # A measured
    al.record_activation("B", al.EVENT_FIRED, path=ledger)      # B fired-unmeasured
    # C never fired
    rows = al.activation_report(known, path=ledger)
    assert [r["lever"] for r in rows] == ["C", "B", "A"]  # never < fired-unmeasured < measured
    assert all(r["default"] == "off" for r in rows)


# --- significance-key canonicalization (#377 build-wave: built-but-mislabeled-unbuilt fix) --------
def _sig_row(lever, est=0.02, axis="d_seg", label=al.SIG_LABEL_ESTIMATED):
    return {"lever": lever, "est_delta_s": est, "axis": axis, "delta_s_label": label}


def test_canonicalize_moves_legacy_key_onto_held_factory():
    sig = {"d_seg_aware_taper_121": _sig_row("d_seg_aware_taper_121")}
    out = al.canonicalize_significance_keys(sig, {"DsegAwareTaper"})
    assert "DsegAwareTaper" in out and "d_seg_aware_taper_121" not in out
    assert out["DsegAwareTaper"]["lever"] == "DsegAwareTaper"
    assert out["DsegAwareTaper"]["_alias_from"] == "d_seg_aware_taper_121"
    assert out["DsegAwareTaper"]["est_delta_s"] == 0.02  # ΔS row carried over intact


def test_canonicalize_noop_when_target_not_a_factory():
    # factory absent -> the row correctly stays legacy (a real build gap, not a mislabel).
    sig = {"d_seg_aware_taper_121": _sig_row("d_seg_aware_taper_121")}
    out = al.canonicalize_significance_keys(sig, set())
    assert out == sig and "DsegAwareTaper" not in out


def test_canonicalize_preserves_explicit_canonical_row():
    # an explicit canonical row must NOT be clobbered by the legacy alias (latest-wins preserved).
    sig = {
        "d_seg_aware_taper_121": _sig_row("d_seg_aware_taper_121", est=0.01),
        "DsegAwareTaper": _sig_row("DsegAwareTaper", est=0.99),
    }
    out = al.canonicalize_significance_keys(sig, {"DsegAwareTaper"})
    assert out["DsegAwareTaper"]["est_delta_s"] == 0.99  # explicit row wins
    assert "d_seg_aware_taper_121" in out  # legacy left as-is when canonical already present


def test_canonicalize_is_pure_and_idempotent():
    sig = {"horizon_weighted_margin_169": _sig_row("horizon_weighted_margin_169")}
    original = {k: dict(v) for k, v in sig.items()}
    out1 = al.canonicalize_significance_keys(sig, {"HorizonWeightedMargin"})
    out2 = al.canonicalize_significance_keys(out1, {"HorizonWeightedMargin"})
    assert sig == original  # input not mutated
    assert out2 == out1     # idempotent


def test_ranked_marks_aliased_lever_registered_not_unbuilt(tmp_path, monkeypatch):
    # end-to-end: a legacy-keyed significance row for a held factory ranks as REGISTERED never-fired
    # (duty-to-MEASURE) instead of unregistered (duty-to-BUILD ~unbuilt).
    import json
    sig_path = tmp_path / "sig.jsonl"
    sig_path.write_text(json.dumps(_sig_row("d_seg_aware_taper_121", est=0.02)) + "\n", encoding="utf-8")
    ledger_path = tmp_path / "ledger.jsonl"  # empty -> never-fired
    ranked = al.duty_to_measure_ranked(
        s_current=0.19110, known=("DsegAwareTaper",), path=ledger_path, sig_path=sig_path
    )
    row = next(r for r in ranked if r["lever"] == "DsegAwareTaper")
    assert row["registered"] is True
    assert row["activation_state"] == "never-fired"
    assert row["est_delta_s"] == 0.02
    assert not any(r["lever"] == "d_seg_aware_taper_121" for r in ranked)


# --- terminal compiled-config join (ddm_ip1) --------------------------------


def _compiled(*names):
    return {"dsl_program_manifest": {"expected_active_levers": list(names)}}


def test_terminal_join_accepts_fired_folded_and_queued_evidence(ledger):
    al.record_activation("A", al.EVENT_FIRED, run_ref="run-a", path=ledger)
    al.record_activation("B", al.EVENT_FOLDED, reason="folded into A", path=ledger)
    al.record_activation("C", al.EVENT_QUEUED, reason="fire when terminal binds", path=ledger)
    receipt = al.terminal_activation_join(_compiled("A", "B", "C"), path=ledger)
    assert receipt.status is al.TerminalJoinStatus.PASS
    assert [row.disposition for row in receipt.rows] == [
        al.TerminalJoinDisposition.FIRED,
        al.TerminalJoinDisposition.FOLDED,
        al.TerminalJoinDisposition.QUEUED,
    ]
    assert receipt.missing_levers == ()


def test_terminal_join_refuses_any_non_default_lever_without_row(ledger):
    al.record_activation("A", al.EVENT_FIRED, path=ledger)
    receipt = al.terminal_activation_join(_compiled("A", "missing"), path=ledger)
    assert receipt.status is al.TerminalJoinStatus.REFUSE
    assert receipt.missing_levers == ("missing",)
    assert receipt.to_dict()["execution_allowed"] is False


def test_terminal_join_excludes_explicit_default_levers(ledger):
    config = {"typed_config": {"levers": [
        {"name": "on", "non_default": True},
        {"name": "off", "default": True},
    ]}}
    al.record_activation("on", al.EVENT_MEASURED, verdict_ref="v.json", path=ledger)
    receipt = al.terminal_activation_join(config, path=ledger)
    assert receipt.non_default_levers == ("on",)
    assert receipt.status is al.TerminalJoinStatus.PASS


def test_terminal_join_refuses_config_without_compiled_lever_surface(ledger):
    with pytest.raises(ValueError, match="expected_active_levers"):
        al.terminal_activation_join({"name": "not-compiled"}, path=ledger)


def test_terminal_join_refuses_duplicate_or_blank_lever_names(ledger):
    with pytest.raises(ValueError, match="duplicate"):
        al.terminal_activation_join(_compiled("A", "A"), path=ledger)
    with pytest.raises(ValueError, match="non-empty"):
        al.terminal_activation_join(_compiled(""), path=ledger)
    with pytest.raises(ValueError, match="default marker must be boolean"):
        al.terminal_activation_join(
            {"levers": [{"name": "A", "default": "false"}]}, path=ledger
        )


def test_current_v752_era_compiled_config_is_positive_against_live_ledger():
    from tac.witness_autoconfig import compile_crucible_v752_launch_config

    compiled = compile_crucible_v752_launch_config(
        "/dev/null", num_pairs=8, epochs=3000, self_orient=False
    )
    config = {"dsl_program_manifest": compiled.dsl_program_manifest}
    receipt = al.terminal_activation_join(config)
    assert receipt.status is al.TerminalJoinStatus.PASS
    assert len(receipt.rows) == 9
    assert all(row.disposition is al.TerminalJoinDisposition.FIRED for row in receipt.rows)


def test_terminal_join_cli_returns_nonzero_and_emits_refusal_receipt(tmp_path, ledger):
    config_path = tmp_path / "compiled.json"
    config_path.write_text(json.dumps(_compiled("missing")), encoding="utf-8")
    tool = Path(__file__).resolve().parents[3] / "tools" / "report_terminal_activation_join.py"
    result = subprocess.run(
        [sys.executable, str(tool), str(config_path), "--ledger", str(ledger)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert json.loads(result.stdout)["status"] == "REFUSE"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
