"""Focused regression and replay tests for the ddm_gov3 apparatus cures."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools import cell_admission as ca
from tools import launch_detached_process as launcher
from tools import safe_run as sr

_REPO = Path(__file__).resolve().parents[3]
_SAFE_RUN = _REPO / "tools" / "safe_run.py"


def _cell(cell_id: str, *, metal: bool, rss: float = 1.5) -> ca.LiveCell:
    return ca.LiveCell(
        cell_id=cell_id,
        pid=os.getpid(),
        alive=True,
        declared_peak_gib=rss,
        current_rss_gib=rss,
        manifest_path=Path(f"/{cell_id}/launch_manifest.json"),
        config_path=None,
        run_dir=None,
        total_steps=None,
        completed_steps=None,
        arm_name=None,
        arm_role=None,
        purpose="gov3 test",
        argv=("python", f"{cell_id}.py", "--metal") if metal else ("python", f"{cell_id}.py"),
        metal_footprint_gib=38.62 if metal else None,
        metal_evidence=("argv:--metal",) if metal else (),
    )


def _fixed_memory(monkeypatch: pytest.MonkeyPatch, *, reclaimable: float, committed: float) -> None:
    monkeypatch.setattr(ca.mem_basis, "conservative_free_gib", lambda default=0.0: reclaimable)
    monkeypatch.setattr(ca.mem_basis, "true_committed_gib", lambda default=0.0: committed)


class TestMetalOccupancy:
    def test_cl3_direct_trainer_argv_counts_as_an_occupant(self, monkeypatch):
        monkeypatch.setattr(ca, "process_tree_rss_gib", lambda _pid: 1.5)
        rows = [
            ca.ProcessRow(
                pid=101,
                ppid=1,
                argv=("python", "tools/safe_run.py", "--", "python", "cl3.py", "--metal"),
            )
        ]
        cells = ca.live_cells_from_process_table(rows)
        assert len(cells) == 1
        assert cells[0].is_cell is False
        assert cells[0].is_metal_occupant is True
        assert "argv:--metal" in cells[0].metal_evidence

    def test_run_config_cell_counts_as_an_occupant(self, tmp_path, monkeypatch):
        config = tmp_path / "authorized.json"
        config.write_text(json.dumps({"cell_id": "md3", "total_steps": 5000, "output": str(tmp_path)}))
        monkeypatch.setattr(ca, "process_tree_rss_gib", lambda _pid: 1.75)
        rows = [ca.ProcessRow(102, 1, ("python", "md3.py", "run-config", str(config)))]
        cell = ca.live_cells_from_process_table(rows)[0]
        assert cell.is_cell is True
        assert cell.is_metal_occupant is True
        assert "argv:run-config-cell" in cell.metal_evidence

    def test_cpu_pricer_does_not_count_as_an_occupant(self, monkeypatch):
        monkeypatch.setattr(ca, "process_tree_rss_gib", lambda _pid: 5.7)
        rows = [
            ca.ProcessRow(
                pid=103,
                ppid=1,
                argv=("python", "tools/safe_run.py", "--", "python", "price_archive.py", "--cpu"),
            )
        ]
        cell = ca.live_cells_from_process_table(rows)[0]
        assert cell.is_metal_occupant is False

    def test_ledger_ratio_above_three_counts_without_mps_argv(self, tmp_path, monkeypatch):
        ledger = tmp_path / "peaks.jsonl"
        monkeypatch.setenv("TAC_MEASURED_PEAKS_LEDGER", str(ledger))
        ca.measured_peaks.append_row(
            {
                "schema": ca.measured_peaks.MEASURED_PEAK_SCHEMA,
                "family": "cl3",
                "governed_peak_gib": 38.62,
                "peak_rss_gib": 1.5,
                "system_availability_delta_gib": 38.62,
            }
        )
        monkeypatch.setattr(ca, "process_tree_rss_gib", lambda _pid: 1.5)
        rows = [
            ca.ProcessRow(
                104,
                1,
                ("python", "tools/safe_run.py", "--", "python", "cl3.py"),
            )
        ]
        cell = ca.live_cells_from_process_table(rows)[0]
        assert cell.is_metal_occupant is True
        assert cell.metal_footprint_gib == pytest.approx(38.62)
        assert cell.declared_peak_gib == pytest.approx(38.62)
        assert cell.declared_peak_source == "measured_metal_footprint"

    def test_two_occupants_refuse_with_the_rule_chain(self, tmp_path):
        verdict = ca.metal_occupancy_verdict(
            [_cell("md3", metal=True)],
            candidate_is_metal=True,
            admission_rows=[],
            cpu_context={"logical_cpus": 18, "load_avg_1m": 6.5},
        )
        assert verdict.admits is False
        assert verdict.live_occupant_names == ("md3",)
        assert "0/3 measured dual-Metal FIT windows" in verdict.reasons[0]

    def test_seed_table_has_profiles_but_no_dual_fit_proof(self):
        rows = ca.read_metal_admission_rows()
        assert len(rows) == 3
        assert {row["occupant"] for row in rows} == {"cl3 trainer", "md3 Metal cell", "sj1 shard"}
        assert ca.dual_metal_fit_windows(rows) == 0

    def test_duplicate_fit_rows_do_not_fake_three_windows(self):
        row = {
            "schema": ca.METAL_ADMISSION_WINDOW_SCHEMA,
            "recorded_utc": "2026-09-08T10:00:00Z",
            "measured_window": True,
            "concurrent_metal_occupants": 2,
            "outcome": "FIT",
        }
        assert ca.dual_metal_fit_windows([row, dict(row), dict(row)]) == 1

    def test_cpu_candidate_does_not_inherit_live_metal_contention(self, tmp_path, monkeypatch):
        _fixed_memory(monkeypatch, reclaimable=100.0, committed=4.0)
        decision = ca.decide_admission(
            1.0,
            live_cells=[_cell("m1", metal=True), _cell("m2", metal=True)],
            ledger_path=tmp_path / "empty.jsonl",
            ceiling_gib=116.0,
            include_naive_contrast=False,
            candidate_is_metal=False,
            cpu_context={"logical_cpus": 18, "load_avg_1m": 6.5},
        )
        assert decision.verdict == "ADMIT"
        assert decision.throughput.evidence == "SOLE_CELL_NO_CONTENTION"


class TestProgressBudget:
    @staticmethod
    def _run(tmp_path: Path, *args: str) -> tuple[subprocess.CompletedProcess[str], dict]:
        receipt = tmp_path / "status.json"
        command = [
            sys.executable,
            str(_SAFE_RUN),
            "--skip-admission-gate",
            "--rss-mb",
            "2048",
            "--poll",
            "0.02",
            "--status-receipt",
            str(receipt),
            *args,
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
        return result, json.loads(receipt.read_text())

    def test_stalled_heartbeat_kills(self, tmp_path):
        history = tmp_path / "history.jsonl"
        history.write_text('{"completed_steps":1}\n')
        result, receipt = self._run(
            tmp_path,
            "--timeout",
            "0.12",
            "--far-timeout",
            "0.9",
            "--progress-stall-minutes",
            "0.003",
            "--progress-history-jsonl",
            str(history),
            "--",
            sys.executable,
            "-c",
            "import time; time.sleep(2)",
        )
        assert result.returncode == sr.EXIT_TIMEOUT
        assert receipt["status"] == "progress_stall"
        assert receipt["last_completed_steps"] == 1

    def test_slow_advancing_heartbeat_survives_the_old_cap(self, tmp_path):
        history = tmp_path / "history.jsonl"
        child = (
            f"import json,time; p={str(history)!r}; f=open(p,'a'); "
            "[(f.write(json.dumps({'completed_steps':i})+'\\n'),f.flush(),time.sleep(.1)) "
            "for i in range(1,7)]; f.close()"
        )
        result, receipt = self._run(
            tmp_path,
            "--timeout",
            "0.15",
            "--far-timeout",
            "0.9",
            "--progress-stall-minutes",
            "0.005",
            "--progress-history-jsonl",
            str(history),
            "--",
            sys.executable,
            "-c",
            child,
        )
        assert result.returncode == 0
        assert receipt["status"] == "ok"
        assert receipt["elapsed_s"] > 0.15
        assert receipt["last_completed_steps"] >= 5

    def test_no_heartbeat_preserves_the_old_timeout(self, tmp_path):
        result, receipt = self._run(
            tmp_path,
            "--timeout",
            "0.15",
            "--far-timeout",
            "0.9",
            "--progress-stall-minutes",
            "0.005",
            "--progress-history-jsonl",
            str(tmp_path / "absent.jsonl"),
            "--",
            sys.executable,
            "-c",
            "import time; time.sleep(2)",
        )
        assert result.returncode == sr.EXIT_TIMEOUT
        assert receipt["status"] == "timeout"
        assert receipt["heartbeat_seen"] is False


class TestProjectionRefusal:
    @staticmethod
    def _gate_namespace(*, escape: str | None = None) -> argparse.Namespace:
        return argparse.Namespace(
            skip_admission_gate=False,
            projected_gib=49.6,
            rss_mb=116 * 1024,
            admission_override_rationale=None,
            admit_over_projection=escape,
        )

    @staticmethod
    def _fake_governor(*, enforcing: bool = False) -> argparse.Namespace:
        decision = argparse.Namespace(
            admit=False,
            projected_system_used_gib=125.2,
            adaptive_ceiling_gib=116.0,
            reason="projected system-used 125.2 GiB EXCEEDS adaptive ceiling 116.0 GiB",
        )
        return argparse.Namespace(
            live_admission_decision=lambda **_kwargs: argparse.Namespace(decision=decision),
            admission_enforcing=lambda: enforcing,
        )

    def test_125_2_vs_116_is_a_hard_refusal(self):
        row = sr.projection_refusal_policy(
            admits=False,
            projected_system_used_gib=125.2,
            adaptive_ceiling_gib=116.0,
            enforcing=False,
            admit_over_projection=None,
        )
        assert row["action"] == "REFUSE"
        assert row["hard"] is True

    def test_118_vs_116_remains_advisory(self):
        row = sr.projection_refusal_policy(
            admits=False,
            projected_system_used_gib=118.0,
            adaptive_ceiling_gib=116.0,
            enforcing=False,
            admit_over_projection=None,
        )
        assert row["action"] == "ADVISORY"
        assert row["hard"] is False

    def test_substantive_escape_admits_and_placeholder_refuses(self):
        admitted = sr.projection_refusal_policy(
            admits=False,
            projected_system_used_gib=125.2,
            adaptive_ceiling_gib=116.0,
            enforcing=False,
            admit_over_projection="operator accepts this bounded replay risk",
        )
        refused = sr.projection_refusal_policy(
            admits=False,
            projected_system_used_gib=125.2,
            adaptive_ceiling_gib=116.0,
            enforcing=False,
            admit_over_projection="tbd",
        )
        assert admitted["action"] == "ADMIT" and admitted["override_used"] is True
        assert refused["action"] == "REFUSE"
        assert "placeholder" in refused["reason"]

    def test_integrated_gate_prints_hard_rule_chain_and_escape(self, monkeypatch, capsys):
        monkeypatch.setitem(sys.modules, "system_memory_governor", self._fake_governor())
        monkeypatch.setitem(
            sys.modules,
            "spawn_durable_daemon",
            argparse.Namespace(reconcile_dead_daemons=lambda **_kwargs: None),
        )
        assert sr._system_admission_gate(self._gate_namespace(), ["trainer"]) == 5
        refused = capsys.readouterr().err
        assert "REFUSED" in refused
        assert "overage 9.2 GiB > attribution error 5.0 GiB" in refused

        rationale = "operator accepts this bounded replay risk"
        escaped_ns = self._gate_namespace(escape=rationale)
        assert sr._system_admission_gate(escaped_ns, ["trainer"]) is None
        escaped = capsys.readouterr().err
        assert "OVER-PROJECTION ADMISSION ESCAPE" in escaped
        assert rationale in escaped
        assert escaped_ns.projection_override_used is True


def test_launcher_derives_progress_far_cap_and_records_escape(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(_REPO / "tools"))
    monkeypatch.setattr(
        launcher,
        "_metal_launch_profile",
        lambda _cmd: {
            "metal_occupant": True,
            "metal_footprint_gib": 38.62,
            "metal_occupancy_evidence": ["ledger:metal_footprint_gib"],
        },
    )
    config = tmp_path / "authorized.json"
    output = tmp_path / "run"
    config.write_text(json.dumps({"output": str(output), "total_steps": 30}))
    args = argparse.Namespace(
        derive_resource_budgets=True,
        measured_peak_rss_gib=20.0,
        measured_thread_need=2,
        walltime_cap_s=18.0,
        progress_stall_minutes=None,
        admit_over_projection="operator accepts this bounded replay risk",
    )
    wrapped, budget = launcher._derive_resource_budget(
        args,
        out=tmp_path,
        cmd=[sys.executable, "trainer.py", "run-config", str(config)],
        env={},
        cwd=_REPO,
    )
    assert budget["progress_stall_minutes"] == pytest.approx(0.1)
    assert budget["walltime_cap_s"] == pytest.approx(54.0)
    assert budget["progress_history_jsonl"] == str(output / "history.jsonl")
    assert budget["metal_occupant"] is True
    assert budget["measured_peak_rss_gib"] == pytest.approx(20.0)
    assert budget["projected_peak_gib"] == pytest.approx(38.62)
    assert budget["admit_over_projection_rationale"] == args.admit_over_projection
    assert wrapped[wrapped.index("--far-timeout") + 1] == "54.0"
    assert wrapped[wrapped.index("--projected-gib") + 1] == "38.62"


class TestHistoricalReplays:
    def test_1612_cl3_beside_md3_refuses(self, tmp_path, monkeypatch):
        _fixed_memory(monkeypatch, reclaimable=100.0, committed=49.572)
        decision = ca.decide_admission(
            38.62,
            live_cells=[_cell("md3", metal=True, rss=1.75)],
            ledger_path=tmp_path / "empty-contention.jsonl",
            ceiling_gib=116.0,
            include_naive_contrast=False,
            cpu_context={"logical_cpus": 18, "load_avg_1m": 6.5},
        )
        assert decision.verdict == "REFUSE"
        assert decision.metal is not None and "md3" in decision.metal.live_occupant_names

    def test_2230_md3_beside_five_cpu_shards_admits(self, tmp_path, monkeypatch):
        _fixed_memory(monkeypatch, reclaimable=100.0, committed=28.0)
        shards = [_cell(f"sj1_{index}", metal=False, rss=5.7) for index in range(5)]
        decision = ca.decide_admission(
            49.572,
            live_cells=shards,
            ledger_path=tmp_path / "empty-contention.jsonl",
            ceiling_gib=116.0,
            include_naive_contrast=False,
            cpu_context={"logical_cpus": 18, "load_avg_1m": 6.5},
        )
        assert decision.verdict == "ADMIT"
        assert decision.metal is not None and decision.metal.live_occupant_names == ()

    def test_2105_projection_refuses(self):
        row = sr.projection_refusal_policy(
            admits=False,
            projected_system_used_gib=125.2,
            adaptive_ceiling_gib=116.0,
            enforcing=False,
            admit_over_projection=None,
        )
        assert row["action"] == "REFUSE"
