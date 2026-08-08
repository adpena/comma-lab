"""Tests for the DDM seal orchestrator (tools/ddm_seal_orchestrator.py).

The orchestrator's whole value is behavioural, so these tests exercise BEHAVIOUR
(gate ordering, idempotent skip, fail-closed stop, counter reset, unit honesty),
never constants. Replacing any gate body with a canonical-marker return would
fail these tests - the anti-pattern CLAUDE.md's NO-FAKE class 2 names.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import ddm_seal_orchestrator as orch  # noqa: E402


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")


def _ticket(tmp_path: Path, **overrides) -> Path:
    run = tmp_path / "sigma" / "run_1"
    probe_out = tmp_path / "probe" / "mem_probe_result.json"
    ticket = {
        "mem_probe_command": ["true", "--out", str(probe_out)],
        "mem_probe_receipt_path": str(probe_out.parent / "mem_probe_receipt.json"),
        "argv_sigma_fp16_run1": [
            "true",
            "--out",
            str(run / "result.json"),
            "--fire-guard-verdict",
            str(run / "fire_guard_verdict.json"),
            "--fire-argv-key",
            "argv_sigma_fp16_run1",
        ],
        "sigma_calibration": {"sanity_sigma_measured": None},
    }
    ticket.update(overrides)
    path = tmp_path / "ticket.json"
    _write(path, ticket)
    return path


def _pass_probe(tmp_path: Path) -> None:
    _write(
        tmp_path / "probe" / "mem_probe_receipt.json",
        {"status": "passed", "peak": {"peak_mlx_reported": 8.49, "peak_rss": 2.02}},
    )


def _pass_guard(tmp_path: Path) -> None:
    _write(tmp_path / "sigma" / "run_1" / "fire_guard_verdict.json", {"status": "passed"})


def _pass_run(tmp_path: Path, loss: float = 0.000377) -> None:
    _write(
        tmp_path / "sigma" / "run_1" / "result.json",
        {"mlx_train": {"status": "passed", "history": [{"step": 5, "loss": loss}], "seconds_per_step": 22.1}},
    )


# --------------------------------------------------------------------- argv custody


def test_argv_flag_reads_the_tickets_own_value():
    argv = ["prog", "--out", "/x/result.json", "--steps", "5"]
    assert orch.argv_flag(argv, "--out") == "/x/result.json"
    assert orch.argv_flag(argv, "--steps") == "5"


def test_argv_flag_returns_none_for_absent_and_dangling_flags():
    assert orch.argv_flag(["prog", "--out"], "--out") is None
    assert orch.argv_flag(["prog"], "--missing") is None


def test_gate_build_refuses_a_sigma_key_without_a_verdict_path(tmp_path):
    path = _ticket(tmp_path, argv_sigma_fp16_run1=["true", "--out", "/x/r.json"])
    with pytest.raises(orch.TicketError):
        orch.build_gates(orch.load_ticket(path), path)


def test_gate_build_derives_probe_receipt_from_the_commands_own_out(tmp_path):
    path = _ticket(tmp_path)
    gates = orch.build_gates(orch.load_ticket(path), path)
    probe = next(g for g in gates if g.name == "mem_probe_fp16")
    assert probe.receipt == tmp_path / "probe" / "mem_probe_receipt.json"


# ------------------------------------------------------------------ status walking


def test_status_walk_is_read_only_and_reports_pending(tmp_path):
    path = _ticket(tmp_path)
    report = orch.walk(path, run=False, only=None, max_gates=64, verbose=False)
    assert report["executed"] == []
    assert report["gates"][0]["status"] == orch.PENDING
    assert not (tmp_path / "probe" / "mem_probe_receipt.json").exists()


def test_satisfied_gates_are_skipped_on_a_run_walk(tmp_path):
    path = _ticket(tmp_path)
    _pass_probe(tmp_path)
    _pass_guard(tmp_path)
    _pass_run(tmp_path)
    report = orch.walk(path, run=True, only=None, max_gates=64, verbose=False)
    by_name = {row["name"]: row for row in report["gates"]}
    already_satisfied = {"mem_probe_fp16", "guard::argv_sigma_fp16_run1", "run::argv_sigma_fp16_run1"}
    assert already_satisfied.isdisjoint(report["executed"]), "idempotence: satisfied gates must not re-run"
    for name in already_satisfied:
        assert by_name[name]["status"] == orch.SATISFIED


def test_failed_probe_receipt_blocks_and_stops_the_walk(tmp_path):
    path = _ticket(tmp_path)
    _write(tmp_path / "probe" / "mem_probe_receipt.json", {"status": "refused"})
    report = orch.walk(path, run=False, only=None, max_gates=64, verbose=False)
    assert report["blocked"] == ["mem_probe_fp16"]
    assert len(report["gates"]) == 1, "fail-closed: no gate is evaluated past a blocker"


def test_stale_probe_receipt_is_pending_not_satisfied(tmp_path, monkeypatch):
    path = _ticket(tmp_path)
    _pass_probe(tmp_path)
    receipt = tmp_path / "probe" / "mem_probe_receipt.json"
    old = receipt.stat().st_mtime - (orch.RECEIPT_FRESHNESS_WINDOW_SECONDS + 60)
    import os

    os.utime(receipt, (old, old))
    report = orch.walk(path, run=False, only=None, max_gates=64, verbose=False)
    row = report["gates"][0]
    assert row["status"] == orch.PENDING and "stale" in row["reason"]


def test_run_gate_blocks_when_its_dependency_is_unsatisfied(tmp_path):
    path = _ticket(tmp_path)
    _pass_probe(tmp_path)
    # guard verdict deliberately absent -> the run must not execute
    report = orch.walk(path, run=True, only="run::argv_sigma_fp16_run1", max_gates=64, verbose=False)
    names = {row["name"]: row for row in report["gates"]}
    assert "run::argv_sigma_fp16_run1" not in report["executed"]
    assert names["guard::argv_sigma_fp16_run1"]["status"] == orch.PENDING


def test_fire_gate_reports_hold_and_is_structurally_unrunnable(tmp_path):
    path = _ticket(
        tmp_path,
        fire_argv_key="argv_burn",
        # `false` returns rc=1: if execute_gate ever SPAWNED it, the rc would surface
        argv_burn=["false", "--fire-guard-verdict", str(tmp_path / "burn_verdict.json")],
    )
    report = orch.walk(path, run=False, only=None, max_gates=64, verbose=False)
    fire = next(row for row in report["gates"] if row["name"] == "FIRE")
    assert fire["status"] == orch.MANUAL and "held" in fire["reason"]

    ticket = orch.load_ticket(path)
    gate = next(g for g in orch.build_gates(ticket, path) if g.kind == orch.KIND_FIRE)
    rc, note = orch.execute_gate(gate, ticket, path, verbose=False)
    assert rc == 0 and "not executed" in note, "containment: heavy Metal fire is never auto-run"


# ------------------------------------------------------------------ review counter


@pytest.mark.parametrize(
    "rounds,expected_clean",
    [
        ([], 0),
        ([{"findings": 0}], 1),
        ([{"findings": 0}, {"findings": 2}], 0),          # a finding RESETS
        ([{"findings": 0}, {"findings": 2}, {"findings": 0}], 1),
        ([{"findings": 0}, {"findings": 0}, {"findings": 0}], 3),
    ],
)
def test_review_counter_resets_on_any_finding(tmp_path, rounds, expected_clean):
    path = _ticket(tmp_path, review_passes=rounds)
    ticket = orch.load_ticket(path)
    gate = orch.Gate(name="review_passes", kind=orch.KIND_REVIEW, detail="", extra={"required": 3})
    state = orch._eval_review(gate, ticket)
    assert f"{expected_clean}/3" in state.reason
    assert state.status == (orch.SATISFIED if expected_clean >= 3 else orch.PENDING)


# ----------------------------------------------------------------- sigma harvest


def _harvest_ticket(tmp_path, losses: list[float], fp32: float | None) -> Path:
    keys = {}
    for index, loss in enumerate(losses, start=1):
        run = tmp_path / "sigma" / f"run_{index}"
        keys[f"argv_sigma_fp16_run{index}"] = [
            "true",
            "--out",
            str(run / "result.json"),
            "--fire-guard-verdict",
            str(run / "verdict.json"),
        ]
        _write(
            run / "result.json",
            {"mlx_train": {"status": "passed", "history": [{"step": 5, "loss": loss}]}},
        )
    if fp32 is not None:
        run = tmp_path / "sigma" / "run_fp32"
        keys["argv_sigma_fp32_ref"] = [
            "true",
            "--out",
            str(run / "result.json"),
            "--fire-guard-verdict",
            str(run / "verdict.json"),
        ]
        _write(
            run / "result.json",
            {"mlx_train": {"status": "passed", "history": [{"step": 5, "loss": fp32}]}},
        )
    return _ticket(tmp_path, **keys)


def test_harvest_computes_real_sigma_and_writes_both_receipt_and_ticket(tmp_path):
    path = _harvest_ticket(tmp_path, [1.0, 2.0, 3.0], fp32=4.0)
    ticket = orch.load_ticket(path)
    gates = orch.build_gates(ticket, path)
    gate = next(g for g in gates if g.kind == orch.KIND_HARVEST)
    assert orch.harvest_sigma(gate, ticket, path) == 0

    receipt = json.loads(gate.receipt.read_text())
    measured = receipt["measured"]
    assert measured["n"] == 3
    assert measured["mean"] == pytest.approx(2.0)
    assert measured["sigma"] == pytest.approx(1.0)          # stdev([1,2,3]) == 1.0
    assert receipt["fp16_fp32_delta"]["abs_delta"] == pytest.approx(2.0)
    assert receipt["fp16_fp32_delta"]["delta_in_sigma"] == pytest.approx(2.0)
    # the ticket itself is amended - the harvest is the writer, not a human
    assert json.loads(path.read_text())["sigma_calibration"]["sanity_sigma_measured"]["n"] == 3


def test_harvest_labels_scope_and_refuses_score_authority(tmp_path):
    path = _harvest_ticket(tmp_path, [1.0, 1.0], fp32=None)
    ticket = orch.load_ticket(path)
    gate = next(g for g in orch.build_gates(ticket, path) if g.kind == orch.KIND_HARVEST)
    orch.harvest_sigma(gate, ticket, path)
    receipt = json.loads(gate.receipt.read_text())
    assert receipt["score_claim"] is False
    assert receipt["axis"] == "[macOS-MLX research-signal]"
    assert "same-seed" in receipt["measured"]["scope"]


def test_harvest_emits_unit_mismatch_rather_than_comparing_loss_to_a_dseg_bar(tmp_path):
    path = _harvest_ticket(tmp_path, [1.0, 2.0], fp32=1.5)
    ticket = orch.load_ticket(path)
    gate = next(g for g in orch.build_gates(ticket, path) if g.kind == orch.KIND_HARVEST)
    orch.harvest_sigma(gate, ticket, path)
    receipt = json.loads(gate.receipt.read_text())
    for name, row in receipt["falsifiers"].items():
        assert row["fired"] is None, f"{name} must not be adjudicated across units"
        assert "UNIT_MISMATCH" in row["reason"]
        assert row["bar_unit"] != row["measured_unit"]
    assert receipt["resolving_commands"]["d_seg_unit_sigma"]


def test_unevaluable_falsifier_blocks_the_harvest_gate(tmp_path):
    path = _harvest_ticket(tmp_path, [1.0, 2.0], fp32=1.5)
    ticket = orch.load_ticket(path)
    gate = next(g for g in orch.build_gates(ticket, path) if g.kind == orch.KIND_HARVEST)
    orch.harvest_sigma(gate, ticket, path)
    state = orch._eval_harvest(gate, orch.load_ticket(path))
    assert state.status == orch.BLOCKED and "unevaluable" in state.reason


def test_missing_run_result_fails_the_harvest_closed(tmp_path):
    path = _harvest_ticket(tmp_path, [1.0, 2.0], fp32=None)
    ticket = orch.load_ticket(path)
    gate = next(g for g in orch.build_gates(ticket, path) if g.kind == orch.KIND_HARVEST)
    (tmp_path / "sigma" / "run_2" / "result.json").unlink()
    assert orch.harvest_sigma(gate, ticket, path) == 4


def test_fired_falsifier_blocks(tmp_path):
    path = _harvest_ticket(tmp_path, [1.0, 2.0], fp32=None)
    ticket = orch.load_ticket(path)
    gate = next(g for g in orch.build_gates(ticket, path) if g.kind == orch.KIND_HARVEST)
    orch.harvest_sigma(gate, ticket, path)
    payload = json.loads(gate.receipt.read_text())
    payload["falsifiers"]["F1_sigma_below_fp16_guard_bar"]["fired"] = True
    gate.receipt.write_text(json.dumps(payload))
    state = orch._eval_harvest(gate, orch.load_ticket(path))
    assert state.status == orch.BLOCKED and "falsifier fired" in state.reason


# ------------------------------------------------------------------------- CLI


def test_cli_status_returns_zero_and_renders_a_table(tmp_path, capsys):
    path = _ticket(tmp_path)
    assert orch.main(["--ticket", str(path), "--quiet"]) == 0
    out = capsys.readouterr().out
    assert "DDM SEAL ORCHESTRATOR" in out and "mem_probe_fp16" in out


def test_cli_returns_three_when_blocked(tmp_path, capsys):
    path = _ticket(tmp_path)
    _write(tmp_path / "probe" / "mem_probe_receipt.json", {"status": "refused"})
    assert orch.main(["--ticket", str(path), "--quiet"]) == 3


def test_cli_returns_three_on_a_ticket_error(tmp_path, capsys):
    assert orch.main(["--ticket", str(tmp_path / "absent.json"), "--quiet"]) == 3
