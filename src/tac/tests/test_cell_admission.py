"""Tests for ``tools/cell_admission.py`` -- governed admission for concurrent local cells.

Coverage: live-cell discovery from launch manifests (cell vs non-cell job, dead PID skipped,
config-derived run dir / step budget), the unrealized-growth charge that the hand-written shell
rule omitted, the reclaimable-aware memory arithmetic (both the relative-headroom and the absolute
operator-ceiling leg), fail-closed behaviour on an unmeasurable basis, the Metal-contention ledger
(append under lock, schema filter, serial baseline, throughput verdict at every evidence grade),
the composed decision, and the CLI exit codes the fire scripts branch on.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_MODULE_PATH = _REPO / "tools" / "cell_admission.py"

_spec = importlib.util.spec_from_file_location("cell_admission_under_test", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
ca = importlib.util.module_from_spec(_spec)
# Register BEFORE exec: ``@dataclasses.dataclass`` resolves ``cls.__module__`` through
# ``sys.modules`` while the class body is being processed.
sys.modules[_spec.name] = ca
_spec.loader.exec_module(ca)


# ── fixtures ────────────────────────────────────────────────────────────────────────────────────


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _make_cell(
    root: Path,
    name: str,
    *,
    pid: int | None = None,
    peak_gib: float = 10.0,
    total_steps: int | None = 5000,
    history_rows: int = 0,
    with_config: bool = True,
) -> Path:
    """Build a launch manifest (+ config + history) that :mod:`cell_admission` can read."""
    pid = os.getpid() if pid is None else pid
    run_dir = root / name / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    if history_rows:
        run_dir.joinpath("history.jsonl").write_text(
            "".join(json.dumps({"completed_steps": i + 1}) + "\n" for i in range(history_rows)),
            encoding="utf-8",
        )

    config_path = root / name / "authorized_configs" / f"{name}.json"
    argv = ["/bin/python", "trainer.py"]
    if with_config:
        payload: dict = {
            "schema": "ddm_qbr1_fairform_config.v1",
            "cell_id": name,
            "output": str(run_dir),
            "arm_name": f"{name}_arm",
            "arm_role": "treatment",
        }
        if total_steps is not None:
            payload["total_steps"] = total_steps
        _write_json(config_path, payload)
        argv += ["run-config", str(config_path)]

    manifest = root / name / "launch" / name / "launch_manifest.json"
    _write_json(
        manifest,
        {
            "schema": "detached_local_process_launch.v2",
            "pid": pid,
            "argv": argv,
            "purpose": f"test cell {name}",
            "resource_budget": {"measured_peak_rss_gib": peak_gib},
        },
    )
    return manifest


@pytest.fixture
def unmeasurable_basis(monkeypatch):
    """Force :mod:`mem_basis` to report no measurement (the fail-closed path)."""
    monkeypatch.setattr(ca.mem_basis, "conservative_free_gib", lambda default=0.0: float(default))
    monkeypatch.setattr(ca.mem_basis, "true_committed_gib", lambda default=0.0: float(default))


def _fixed_basis(monkeypatch, *, reclaimable: float, committed: float) -> None:
    monkeypatch.setattr(ca.mem_basis, "conservative_free_gib", lambda default=0.0: reclaimable)
    monkeypatch.setattr(ca.mem_basis, "true_committed_gib", lambda default=0.0: committed)


# ── live-cell discovery ─────────────────────────────────────────────────────────────────────────


class TestDiscovery:
    def test_discovers_a_live_cell_with_config_derived_fields(self, tmp_path):
        _make_cell(tmp_path, "alpha", peak_gib=12.5, total_steps=5000, history_rows=137)
        cells = ca.discover_live_cells([tmp_path])
        assert len(cells) == 1
        cell = cells[0]
        assert cell.cell_id == "alpha"
        assert cell.is_cell is True
        assert cell.declared_peak_gib == 12.5
        assert cell.total_steps == 5000
        assert cell.completed_steps == 137
        assert cell.arm_role == "treatment"
        assert cell.progress_fraction == pytest.approx(137 / 5000)

    def test_dead_pid_is_not_live(self, tmp_path):
        # PID 2**22 is above the macOS/Linux default pid_max and is reliably absent.
        _make_cell(tmp_path, "dead", pid=4194304)
        assert ca.discover_live_cells([tmp_path]) == []

    def test_non_cell_job_is_discovered_but_not_flagged_a_cell(self, tmp_path):
        _make_cell(tmp_path, "plainjob", with_config=False, peak_gib=12.0)
        cells = ca.discover_live_cells([tmp_path])
        assert len(cells) == 1
        assert cells[0].is_cell is False
        # Still charged: a non-cell job holds real memory.
        assert cells[0].declared_peak_gib == 12.0

    def test_config_without_step_budget_is_not_a_cell(self, tmp_path):
        _make_cell(tmp_path, "nosteps", total_steps=None)
        assert ca.discover_live_cells([tmp_path])[0].is_cell is False

    def test_foreign_manifest_schema_is_ignored(self, tmp_path):
        _write_json(
            tmp_path / "x" / "launch_manifest.json",
            {"schema": "something_else.v1", "pid": os.getpid()},
        )
        assert ca.discover_live_cells([tmp_path]) == []

    def test_missing_root_is_tolerated(self, tmp_path):
        assert ca.discover_live_cells([tmp_path / "absent"]) == []

    def test_same_pid_is_deduplicated(self, tmp_path):
        _make_cell(tmp_path, "one")
        _make_cell(tmp_path, "two")  # same PID (this process)
        assert len(ca.discover_live_cells([tmp_path])) == 1

    def test_config_path_from_argv_prefers_run_config(self):
        argv = ["python", "trainer.py", "run-config", "/x/cfg.json", "--other", "/y/z.json"]
        assert ca._config_path_from_argv(argv) == Path("/x/cfg.json")

    def test_config_path_from_argv_falls_back_to_trailing_json(self):
        assert ca._config_path_from_argv(["python", "t.py", "/x/tail.json"]) == Path("/x/tail.json")

    def test_config_path_from_argv_none_when_absent(self):
        assert ca._config_path_from_argv(["python", "t.py", "--flag"]) is None

    def test_count_history_steps_handles_missing(self, tmp_path):
        assert ca.count_history_steps(None) is None
        assert ca.count_history_steps(tmp_path / "nope") is None

    def test_process_tree_rss_includes_self(self):
        rss = ca.process_tree_rss_gib(os.getpid())
        assert rss is not None and rss > 0.0

    def test_process_tree_rss_none_for_dead_pid(self):
        assert ca.process_tree_rss_gib(4194304) is None

    def test_pid_alive_rejects_unsafe_pids(self):
        assert ca.pid_alive(0) is False
        assert ca.pid_alive(1) is False
        assert ca.pid_alive(os.getpid()) is True


# ── unrealized growth (the charge the shell rule omitted) ───────────────────────────────────────


class TestUnrealizedGrowth:
    def _cell(self, **kw):
        base = {
            "cell_id": "c",
            "pid": 1,
            "alive": True,
            "declared_peak_gib": 41.5,
            "current_rss_gib": 0.6,
            "manifest_path": Path("/m"),
            "config_path": Path("/c"),
            "run_dir": Path("/r"),
            "total_steps": 5000,
            "completed_steps": 1,
            "arm_name": None,
            "arm_role": None,
            "purpose": None,
        }
        base.update(kw)
        return ca.LiveCell(**base)

    def test_growth_is_peak_minus_current(self):
        assert self._cell().unrealized_growth_gib == pytest.approx(40.9)

    def test_growth_never_negative_when_over_peak(self):
        assert self._cell(current_rss_gib=50.0).unrealized_growth_gib == 0.0

    def test_unreadable_rss_charges_the_whole_peak(self):
        """Fail-closed: an unreadable RSS must not make a live cell look free."""
        assert self._cell(current_rss_gib=None).unrealized_growth_gib == pytest.approx(41.5)

    def test_progress_fraction_none_without_budget(self):
        assert self._cell(total_steps=None).progress_fraction is None

    def test_progress_fraction_clamped_at_one(self):
        assert self._cell(completed_steps=9999, total_steps=5000).progress_fraction == 1.0


# ── memory arithmetic ───────────────────────────────────────────────────────────────────────────


class TestMemoryVerdict:
    def test_admits_with_headroom(self, monkeypatch):
        _fixed_basis(monkeypatch, reclaimable=80.0, committed=20.0)
        verdict = ca.memory_verdict(40.0, [], margin_gib=16.0, include_naive_contrast=False)
        assert verdict.admits is True
        assert verdict.required_gib == pytest.approx(56.0)
        assert verdict.headroom_gib == pytest.approx(24.0)

    def test_refuses_when_reclaimable_short(self, monkeypatch):
        _fixed_basis(monkeypatch, reclaimable=20.0, committed=20.0)
        verdict = ca.memory_verdict(40.0, [], margin_gib=16.0, include_naive_contrast=False)
        assert verdict.admits is False
        assert verdict.headroom_gib == pytest.approx(-36.0)
        assert any("reclaimable" in reason for reason in verdict.reasons)

    def test_live_unrealized_growth_is_charged(self, monkeypatch):
        """The regression that matters: the shell rule ignored live cells entirely."""
        _fixed_basis(monkeypatch, reclaimable=60.0, committed=10.0)
        live = ca.LiveCell(
            cell_id="ng3",
            pid=1,
            alive=True,
            declared_peak_gib=41.5,
            current_rss_gib=0.6,
            manifest_path=Path("/m"),
            config_path=None,
            run_dir=None,
            total_steps=None,
            completed_steps=None,
            arm_name=None,
            arm_role=None,
            purpose=None,
        )
        without = ca.memory_verdict(20.0, [], margin_gib=16.0, include_naive_contrast=False)
        with_live = ca.memory_verdict(20.0, [live], margin_gib=16.0, include_naive_contrast=False)
        assert without.admits is True
        assert with_live.admits is False
        assert with_live.live_unrealized_gib == pytest.approx(40.9)
        assert with_live.required_gib == pytest.approx(76.9)

    def test_operator_ceiling_leg_can_refuse_alone(self, monkeypatch):
        """Plenty of reclaimable headroom, but the absolute committed ceiling still binds."""
        _fixed_basis(monkeypatch, reclaimable=200.0, committed=110.0)
        verdict = ca.memory_verdict(
            20.0, [], margin_gib=0.0, ceiling_gib=116.0, include_naive_contrast=False
        )
        assert verdict.headroom_gib > 0
        assert verdict.ceiling_headroom_gib == pytest.approx(-14.0)
        assert verdict.admits is False
        assert any("ceiling" in reason for reason in verdict.reasons)

    def test_fail_closed_when_basis_unmeasurable(self, unmeasurable_basis):
        verdict = ca.memory_verdict(1.0, [], include_naive_contrast=False)
        assert verdict.measurable is False
        assert verdict.admits is False
        assert any("fail-closed" in reason for reason in verdict.reasons)

    def test_basis_names_the_canonical_helper_not_raw_vm_stat(self, monkeypatch):
        _fixed_basis(monkeypatch, reclaimable=80.0, committed=10.0)
        verdict = ca.memory_verdict(1.0, [], include_naive_contrast=False)
        assert "mem_basis" in verdict.basis

    def test_negative_candidate_is_clamped(self, monkeypatch):
        _fixed_basis(monkeypatch, reclaimable=80.0, committed=10.0)
        assert ca.memory_verdict(
            -5.0, [], margin_gib=0.0, include_naive_contrast=False
        ).candidate_peak_gib == 0.0

    def test_as_dict_is_json_serializable(self, monkeypatch):
        _fixed_basis(monkeypatch, reclaimable=80.0, committed=10.0)
        payload = ca.memory_verdict(1.0, [], include_naive_contrast=False).as_dict()
        json.dumps(payload)
        assert payload["reasons"]

    def test_operator_ceiling_env_override(self, monkeypatch):
        monkeypatch.setenv(ca.OPERATOR_CEILING_ENV, "90")
        assert ca.operator_ceiling_gib() == 90.0
        monkeypatch.setenv(ca.OPERATOR_CEILING_ENV, "not-a-number")
        assert ca.operator_ceiling_gib() == ca.OPERATOR_CEILING_GIB_FALLBACK
        monkeypatch.setenv(ca.OPERATOR_CEILING_ENV, "0")
        assert ca.operator_ceiling_gib() == ca.OPERATOR_CEILING_GIB_FALLBACK
        monkeypatch.delenv(ca.OPERATOR_CEILING_ENV)
        assert ca.operator_ceiling_gib() == ca.OPERATOR_CEILING_GIB_FALLBACK


# ── Metal-contention ledger ─────────────────────────────────────────────────────────────────────


class TestContentionLedger:
    def _row(self, concurrency: int, total: float, when: str) -> dict:
        return {
            "schema": ca.THROUGHPUT_ROW_SCHEMA,
            "recorded_utc": when,
            "concurrency": concurrency,
            "total_steps_per_min": total,
        }

    def test_append_and_read_roundtrip(self, tmp_path):
        ledger = tmp_path / "ledger.jsonl"
        ca.append_contention_row(self._row(1, 28.0, "2026-09-04T10:00:00Z"), ledger)
        ca.append_contention_row(self._row(2, 35.0, "2026-09-04T11:00:00Z"), ledger)
        rows = ca.read_contention_rows(ledger)
        assert [row["concurrency"] for row in rows] == [1, 2]

    def test_append_stamps_schema_and_time(self, tmp_path):
        ledger = tmp_path / "ledger.jsonl"
        written = ca.append_contention_row({"concurrency": 1, "total_steps_per_min": 1.0}, ledger)
        assert written["schema"] == ca.THROUGHPUT_ROW_SCHEMA
        assert written["recorded_utc"].endswith("Z")

    def test_read_missing_ledger_is_empty(self, tmp_path):
        assert ca.read_contention_rows(tmp_path / "absent.jsonl") == []

    def test_read_skips_foreign_schema_and_garbage(self, tmp_path):
        ledger = tmp_path / "ledger.jsonl"
        ledger.write_text(
            json.dumps({"schema": "other.v1", "concurrency": 1}) + "\n"
            + "not json\n"
            + "\n"
            + json.dumps(self._row(2, 35.0, "2026-09-04T11:00:00Z")) + "\n",
            encoding="utf-8",
        )
        rows = ca.read_contention_rows(ledger)
        assert len(rows) == 1 and rows[0]["concurrency"] == 2

    def test_serial_baseline_is_the_best_single_cell_row(self, tmp_path):
        rows = [
            self._row(1, 28.0, "2026-09-04T10:00:00Z"),
            self._row(1, 24.0, "2026-09-04T10:30:00Z"),
            self._row(2, 35.0, "2026-09-04T11:00:00Z"),
        ]
        assert ca.serial_baseline_steps_per_min(rows) == 28.0

    def test_serial_baseline_none_without_a_single_cell_row(self):
        assert ca.serial_baseline_steps_per_min([self._row(2, 35.0, "t")]) is None


class TestThroughputVerdict:
    def _row(self, concurrency: int, total: float, when: str) -> dict:
        return {
            "schema": ca.THROUGHPUT_ROW_SCHEMA,
            "recorded_utc": when,
            "concurrency": concurrency,
            "total_steps_per_min": total,
        }

    def test_no_observation_admits_but_labels_the_absence(self):
        verdict = ca.throughput_verdict([], live_count=2)
        assert verdict.admits is True
        assert verdict.evidence == "NO_CONCURRENT_OBSERVATION"

    def test_concurrency_pays_admits(self):
        """MAIN's measured row: 28 serial -> 20 + 15 = 35 concurrent."""
        rows = [
            self._row(1, 28.0, "2026-09-04T10:00:00Z"),
            self._row(2, 35.0, "2026-09-04T11:00:00Z"),
        ]
        verdict = ca.throughput_verdict(rows, live_count=2)
        assert verdict.evidence == "MEASURED"
        assert verdict.admits is True
        assert verdict.ratio == pytest.approx(35.0 / 28.0)

    def test_contention_below_serial_refuses(self):
        rows = [
            self._row(1, 28.0, "2026-09-04T10:00:00Z"),
            self._row(2, 21.0, "2026-09-04T11:00:00Z"),
        ]
        verdict = ca.throughput_verdict(rows, live_count=2)
        assert verdict.admits is False
        assert verdict.ratio == pytest.approx(0.75)
        assert "28.00" in verdict.reasons[0]

    def test_latest_concurrent_row_wins(self):
        rows = [
            self._row(1, 28.0, "2026-09-04T10:00:00Z"),
            self._row(2, 35.0, "2026-09-04T11:00:00Z"),
            self._row(2, 20.0, "2026-09-04T12:00:00Z"),
        ]
        assert ca.throughput_verdict(rows, live_count=2).admits is False

    def test_concurrent_without_baseline_is_labelled(self):
        verdict = ca.throughput_verdict([self._row(2, 35.0, "t")], live_count=2)
        assert verdict.evidence == "NO_SERIAL_BASELINE"
        assert verdict.admits is True
        assert verdict.serial_baseline_steps_per_min is None

    def test_as_dict_is_json_serializable(self):
        json.dumps(ca.throughput_verdict([], live_count=2).as_dict())


# ── composed decision + CLI ─────────────────────────────────────────────────────────────────────


class TestDecision:
    def test_admit_requires_both_legs(self, tmp_path, monkeypatch):
        _fixed_basis(monkeypatch, reclaimable=200.0, committed=5.0)
        ledger = tmp_path / "ledger.jsonl"
        ca.append_contention_row(
            {"concurrency": 1, "total_steps_per_min": 28.0, "recorded_utc": "2026-09-04T10:00:00Z"},
            ledger,
        )
        ca.append_contention_row(
            {"concurrency": 2, "total_steps_per_min": 10.0, "recorded_utc": "2026-09-04T11:00:00Z"},
            ledger,
        )
        decision = ca.decide_admission(
            1.0, roots=[tmp_path / "none"], ledger_path=ledger, include_naive_contrast=False
        )
        assert decision.memory.admits is True
        assert decision.throughput.admits is False
        assert decision.verdict == "REFUSE"

    def test_throughput_counts_training_cells_not_every_governed_job(self, tmp_path, monkeypatch):
        """VACUITY==PASS guard: unrelated live jobs must not inflate the required concurrency.

        Two live TRAINING CELLS plus one unrelated job means the candidate would be the 3rd cell,
        so a concurrency-3 ledger row is what matters -- not concurrency 4. Counting every job
        would push the requirement past every recorded row and silently make the leg vacuous.
        """
        _fixed_basis(monkeypatch, reclaimable=500.0, committed=5.0)
        ledger = tmp_path / "ledger.jsonl"
        ca.append_contention_row(
            {"concurrency": 1, "total_steps_per_min": 28.0, "recorded_utc": "2026-09-04T10:00:00Z"},
            ledger,
        )
        ca.append_contention_row(
            {"concurrency": 3, "total_steps_per_min": 10.0, "recorded_utc": "2026-09-04T11:00:00Z"},
            ledger,
        )
        _make_cell(tmp_path, "cell_one", pid=os.getpid(), history_rows=1)
        cells = ca.discover_live_cells([tmp_path])
        # One discovered cell; synthesise a second training cell and one non-cell job.
        second = dataclasses.replace(cells[0], cell_id="cell_two", pid=cells[0].pid)
        job = dataclasses.replace(
            cells[0], cell_id="a_job", config_path=None, total_steps=None, declared_peak_gib=0.0
        )
        decision = ca.decide_admission(
            1.0,
            live_cells=[cells[0], second, job],
            ledger_path=ledger,
            include_naive_contrast=False,
        )
        # The concurrency-3 row (10 steps/min, below the 28 baseline) must BIND, not be skipped.
        assert decision.throughput.evidence == "MEASURED"
        assert decision.throughput.concurrency_observed == 3
        assert decision.throughput.admits is False
        assert decision.verdict == "REFUSE"

    def test_admit_when_both_legs_pass(self, tmp_path, monkeypatch):
        _fixed_basis(monkeypatch, reclaimable=200.0, committed=5.0)
        decision = ca.decide_admission(
            1.0,
            roots=[tmp_path / "none"],
            ledger_path=tmp_path / "empty.jsonl",
            include_naive_contrast=False,
        )
        assert decision.verdict == "ADMIT"
        assert decision.admits is True

    def test_decision_dict_is_non_promotable_and_serializable(self, tmp_path, monkeypatch):
        _fixed_basis(monkeypatch, reclaimable=200.0, committed=5.0)
        payload = ca.decide_admission(
            1.0,
            roots=[tmp_path / "none"],
            ledger_path=tmp_path / "empty.jsonl",
            include_naive_contrast=False,
        ).as_dict()
        json.dumps(payload)
        assert payload["schema"] == ca.ADMISSION_SCHEMA
        assert payload["score_claim"] is False

    def test_human_lines_show_the_arithmetic(self, tmp_path, monkeypatch):
        _fixed_basis(monkeypatch, reclaimable=200.0, committed=5.0)
        _make_cell(tmp_path, "live", peak_gib=41.5, history_rows=5)
        lines = "\n".join(
            ca.decide_admission(
                1.0,
                roots=[tmp_path],
                ledger_path=tmp_path / "empty.jsonl",
                include_naive_contrast=False,
            ).human_lines()
        )
        assert "headroom" in lines
        assert "live-unrealized" in lines
        assert "live" in lines


class TestCLI:
    def test_admit_returns_zero_when_admitted(self, tmp_path, monkeypatch, capsys):
        _fixed_basis(monkeypatch, reclaimable=200.0, committed=5.0)
        rc = ca.main(
            [
                "admit",
                "--candidate-peak-gib", "1.0",
                "--root", str(tmp_path / "none"),
                "--ledger", str(tmp_path / "empty.jsonl"),
                "--json",
            ]
        )
        assert rc == ca.RC_ADMIT
        assert json.loads(capsys.readouterr().out)["verdict"] == "ADMIT"

    def test_admit_returns_two_when_refused(self, tmp_path, monkeypatch, capsys):
        """rc=2 is the code the fire scripts already branch on."""
        _fixed_basis(monkeypatch, reclaimable=5.0, committed=5.0)
        rc = ca.main(
            [
                "admit",
                "--candidate-peak-gib", "41.5",
                "--root", str(tmp_path / "none"),
                "--ledger", str(tmp_path / "empty.jsonl"),
            ]
        )
        assert rc == ca.RC_REFUSE
        assert "REFUSE" in capsys.readouterr().out

    def test_admit_returns_three_when_unmeasurable(self, tmp_path, unmeasurable_basis, capsys):
        rc = ca.main(
            [
                "admit",
                "--candidate-peak-gib", "1.0",
                "--root", str(tmp_path / "none"),
                "--ledger", str(tmp_path / "empty.jsonl"),
            ]
        )
        assert rc == ca.RC_UNMEASURABLE
        capsys.readouterr()

    def test_cells_json_lists_discovered_cells(self, tmp_path, capsys):
        _make_cell(tmp_path, "alpha", history_rows=3)
        assert ca.main(["cells", "--root", str(tmp_path), "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["count"] == 1
        assert payload["cells"][0]["cell_id"] == "alpha"
        assert payload["cells"][0]["is_cell"] is True

    def test_cells_human_mode_reports_none(self, tmp_path, capsys):
        assert ca.main(["cells", "--root", str(tmp_path / "none")]) == 0
        assert "NONE" in capsys.readouterr().out

    def test_sample_measures_without_writing(self, tmp_path, capsys):
        _make_cell(tmp_path, "alpha", history_rows=3)
        ledger = tmp_path / "ledger.jsonl"
        rc = ca.main(
            ["sample", "--window-s", "0", "--root", str(tmp_path), "--ledger", str(ledger), "--no-write"]
        )
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["concurrency"] == 1
        assert payload["score_claim"] is False
        assert not ledger.exists()

    def test_contention_summary_reports_baseline(self, tmp_path, capsys):
        ledger = tmp_path / "ledger.jsonl"
        ca.append_contention_row(
            {"concurrency": 1, "total_steps_per_min": 28.0, "recorded_utc": "2026-09-04T10:00:00Z"},
            ledger,
        )
        assert ca.main(["contention", "--ledger", str(ledger)]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["serial_baseline_steps_per_min"] == 28.0


class TestSampling:
    def test_sample_rates_zero_window_yields_zero_rate(self, tmp_path):
        _make_cell(tmp_path, "alpha", history_rows=10)
        cells = ca.discover_live_cells([tmp_path])
        rates = ca.sample_cell_rates(cells, window_s=0.0)
        assert len(rates) == 1
        assert rates[0].steps_delta == 0

    def test_sample_rates_skips_cells_without_history(self, tmp_path):
        _make_cell(tmp_path, "nohist", with_config=False)
        cells = ca.discover_live_cells([tmp_path])
        assert ca.sample_cell_rates(cells, window_s=0.0) == []


class TestRepoIntegration:
    """The module must stay usable against the real tree without touching live runs."""

    def test_default_roots_are_the_ssd_tiers(self):
        assert any("APDataStore" in str(root) for root in ca.DEFAULT_MANIFEST_ROOTS)

    def test_contention_ledger_lives_in_omx_state(self):
        assert ca.CONTENTION_LEDGER.parent == _REPO / ".omx" / "state"

    def test_default_margin_matches_the_operator_blessed_rule(self):
        """MAIN's hand-written rule was ``reclaimable >= peak + 16``; do not silently loosen it."""
        assert ca.DEFAULT_MARGIN_GIB == 16.0

    def test_module_does_not_use_raw_psutil_virtual_memory_as_a_basis(self):
        """CLASS-1 discipline: the system basis must come from ``tools/mem_basis.py`` only.

        Checked over the AST, not the raw text: the module docstring deliberately NAMES
        ``psutil.virtual_memory().available`` while explaining the bug class it cures, and a
        substring scan would wrongly flag that prose.
        """
        import ast

        tree = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))
        offenders = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr == "virtual_memory"
        ]
        assert offenders == [], f"raw psutil.virtual_memory() used at lines {offenders}"
        assert "mem_basis.conservative_free_gib" in _MODULE_PATH.read_text(encoding="utf-8")
