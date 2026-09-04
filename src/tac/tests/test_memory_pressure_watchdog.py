"""Tests for ``tools/memory_pressure_watchdog.py`` (ddm_gov2).

The thresholds under test are DERIVED from one real event, and the tests replay it: on
2026-09-04 the VM compressor went from 0.66 GiB to CRITICAL in 19.2 s and jetsam killed background
daemons.  Every sample below is verbatim from ``.omx/state/memory_blackbox.jsonl``.
"""

from __future__ import annotations

import importlib.util
import json
import signal
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_MODULE_PATH = _REPO / "tools" / "memory_pressure_watchdog.py"

_spec = importlib.util.spec_from_file_location("memory_pressure_watchdog", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
wd = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("memory_pressure_watchdog", wd)
_spec.loader.exec_module(wd)


#: The 2026-09-04 collapse, verbatim: (offset seconds, compressor GiB, swap GiB, free GiB, level).
MEASURED_RAMP = [
    (0.000, 0.66, 2.30, 0.03, 1),
    (3.150, 19.68, 2.30, 0.05, 1),
    (5.883, 46.42, 2.29, 0.04, 1),
    (8.274, 63.92, 2.29, 0.02, 2),
    (19.213, 73.99, 22.03, 0.07, 4),
]


def _sample(compressor=1.0, swap=1.0, free=40.0, level=1, monotonic=0.0, total=128.0):
    return wd.PressureSample(
        total_gib=total,
        compressor_gib=compressor,
        swap_used_gib=swap,
        free_gib=free,
        pressure_level=level,
        pressure={1: "normal", 2: "warn", 4: "critical"}.get(level, "normal"),
        monotonic=monotonic,
    )


class TestParsers:
    _VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                             3105187.
Pages inactive:                          756483.
Pages wired down:                       3466799.
Pages stored in compressor:              393054.
Pages occupied by compressor:             86670.
"""

    def test_reads_occupied_not_stored(self):
        """``stored in compressor`` is the PRE-compression count and is ~4.5x larger.

        MEASURED on this box at the same instant: stored 393,054 pages vs occupied 86,670.
        Reading the wrong line inflates every alarm by that factor.
        """
        compressor, free = wd.parse_vm_stat(self._VM_STAT)
        assert compressor == pytest.approx(86670 * 16384 / 1024**3, abs=1e-6)
        assert free == pytest.approx(3105187 * 16384 / 1024**3, abs=1e-6)

    def test_honours_the_reported_page_size(self):
        text = self._VM_STAT.replace("page size of 16384", "page size of 4096")
        compressor, _ = wd.parse_vm_stat(text)
        assert compressor == pytest.approx(86670 * 4096 / 1024**3, abs=1e-6)

    def test_empty_vm_stat_is_zero_not_a_crash(self):
        assert wd.parse_vm_stat("") == (0.0, 0.0)

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("vm.swapusage: total = 2048.00M  used = 1141.31M  free = 906.69M  (encrypted)", 1141.31 / 1024),
            ("vm.swapusage: total = 72.00G  used = 64.08G  free = 8.00G", 64.08),
            ("vm.swapusage: total = 1024.00K  used = 512.00K  free = 512.00K", 512.0 / 1024 / 1024),
            ("nonsense", 0.0),
        ],
    )
    def test_swapusage_units(self, text, expected):
        assert wd.parse_swapusage(text) == pytest.approx(expected, abs=1e-6)

    @pytest.mark.parametrize(
        "text,level",
        [
            ("System-wide memory pressure: normal", 1),
            ("System-wide memory pressure: warn", 2),
            ("System-wide memory pressure: critical", 4),
            ("no such line", 1),
        ],
    )
    def test_pressure_levels(self, text, level):
        assert wd.parse_memory_pressure(text)[0] == level


class TestClassify:
    def test_quiet_machine_is_ok(self):
        level, reasons = wd.classify(_sample())
        assert level == wd.LEVEL_OK and reasons == []

    def test_compressor_warn_threshold(self):
        level, reasons = wd.classify(_sample(compressor=16.0))
        assert level == wd.LEVEL_WARN
        assert "compressor" in reasons[0]

    def test_compressor_critical_threshold(self):
        assert wd.classify(_sample(compressor=48.0))[0] == wd.LEVEL_CRITICAL

    def test_swap_thresholds(self):
        assert wd.classify(_sample(swap=4.0))[0] == wd.LEVEL_WARN
        assert wd.classify(_sample(swap=16.0))[0] == wd.LEVEL_CRITICAL

    def test_os_pressure_is_ground_truth(self):
        assert wd.classify(_sample(level=2))[0] == wd.LEVEL_WARN
        assert wd.classify(_sample(level=4))[0] == wd.LEVEL_CRITICAL

    def test_growth_rate_fires_before_any_absolute_threshold(self):
        """The measured ramp was +6.03 and +9.79 GiB/s; the rule catches it on the first delta."""
        previous = _sample(compressor=0.66, monotonic=0.0)
        current = _sample(compressor=19.68, monotonic=3.150)
        level, reasons = wd.classify(current, previous)
        assert level == wd.LEVEL_CRITICAL
        assert any("GiB/s" in reason for reason in reasons)

    def test_a_slow_climb_is_not_a_growth_alarm(self):
        previous = _sample(compressor=1.0, monotonic=0.0)
        current = _sample(compressor=2.0, monotonic=5.0)  # 0.2 GiB/s
        assert wd.classify(current, previous)[0] == wd.LEVEL_OK

    def test_free_memory_is_not_a_trigger(self):
        """REJECTED on its base rate: ``free < 1 GiB`` was true 28.99% of the day's 13,235 samples.

        It also sat at 0.02-0.3 GiB for ~40 s BEFORE the collapse without anything being wrong.
        """
        assert wd.classify(_sample(free=0.02))[0] == wd.LEVEL_OK

    def test_thresholds_scale_with_ram(self):
        small = wd.classify(_sample(compressor=8.0, total=64.0))
        assert small[0] == wd.LEVEL_WARN, "16/128 of a 64 GiB box is 8 GiB"

    def test_replay_of_the_measured_collapse_fires_with_runway(self):
        """The whole event is 19.2 s; the alarm must fire while there is still time to act."""
        fired_at = None
        previous = None
        for offset, compressor, swap, free, level in MEASURED_RAMP:
            sample = _sample(compressor=compressor, swap=swap, free=free, level=level, monotonic=offset)
            verdict, _ = wd.classify(sample, previous)
            previous = sample
            if verdict == wd.LEVEL_CRITICAL and fired_at is None:
                fired_at = offset
        assert fired_at is not None
        jetsam_at = MEASURED_RAMP[-1][0]
        assert jetsam_at - fired_at >= 15.0, f"only {jetsam_at - fired_at:.1f} s of runway"


class TestSignalDiscipline:
    def test_sigkill_is_structurally_impossible(self):
        """A watchdog that can kill a run is a bigger hazard than the pressure it watches."""
        with pytest.raises(ValueError):
            wd._signal_tree(os_pid := 1, signal.SIGKILL)
        assert os_pid == 1

    def test_sigterm_is_also_refused(self):
        with pytest.raises(ValueError):
            wd._signal_tree(1, signal.SIGTERM)

    def test_signal_failure_is_reported_not_raised(self):
        result = wd._signal_tree(2**30, signal.SIGCONT)
        assert result["delivered"] == []
        assert result["failed"] and "error" in result["failed"][0]


class TestWatchLoop:
    def _fixed_sampler(self, samples):
        iterator = iter(samples)
        last = samples[-1]

        def sampler():
            nonlocal last
            try:
                last = next(iterator)
            except StopIteration:
                pass
            return last

        return sampler

    def test_report_only_never_signals(self, tmp_path, monkeypatch):
        signalled = []
        monkeypatch.setattr(wd, "_signal_tree", lambda pid, sig: signalled.append((pid, sig)) or {})
        monkeypatch.setattr(wd, "newest_training_cell", lambda: {"cell_id": "c", "pid": 1, "stop_pid": 2})
        monkeypatch.setattr(wd, "_push", lambda *a: None)
        summary = wd.watch(
            duration_s=0,
            report_only=True,
            ledger=tmp_path / "a.jsonl",
            sampler=self._fixed_sampler([_sample(compressor=60.0)]),
            sleeper=lambda _s: None,
            now=lambda: 0.0,
        )
        assert signalled == []
        assert wd.LEVEL_CRITICAL in summary["levels"]
        row = json.loads((tmp_path / "a.jsonl").read_text().strip().splitlines()[0])
        assert row["action"]["kind"] == "WOULD_SIGSTOP"
        assert row["event"] == "confound_alarm"

    def test_critical_stops_the_newest_cell(self, tmp_path, monkeypatch):
        sent = []
        monkeypatch.setattr(wd, "_signal_tree", lambda pid, sig: sent.append((pid, sig)) or {"delivered": [pid]})
        monkeypatch.setattr(wd, "newest_training_cell", lambda: {"cell_id": "newest", "pid": 9, "stop_pid": 99})
        monkeypatch.setattr(wd, "_push", lambda *a: None)
        wd.watch(
            duration_s=0,
            ledger=tmp_path / "a.jsonl",
            sampler=self._fixed_sampler([_sample(level=4)]),
            sleeper=lambda _s: None,
            now=lambda: 0.0,
        )
        assert sent[0] == (99, signal.SIGSTOP), "the newest cell's TRAINER pid is the target"

    def test_no_target_is_alarmed_not_crashed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(wd, "newest_training_cell", lambda: None)
        monkeypatch.setattr(wd, "_push", lambda *a: None)
        wd.watch(
            duration_s=0,
            ledger=tmp_path / "a.jsonl",
            sampler=self._fixed_sampler([_sample(level=4)]),
            sleeper=lambda _s: None,
            now=lambda: 0.0,
        )
        row = json.loads((tmp_path / "a.jsonl").read_text().strip().splitlines()[-1])
        assert row["action"]["kind"] == "NO_TARGET"

    def test_resume_waits_for_the_clear_hold(self, tmp_path, monkeypatch):
        sent = []
        clock = {"t": 0.0}
        monkeypatch.setattr(wd, "_signal_tree", lambda pid, sig: sent.append((pid, sig)) or {"delivered": [pid]})
        monkeypatch.setattr(wd, "newest_training_cell", lambda: {"cell_id": "c", "pid": 9, "stop_pid": 99})
        monkeypatch.setattr(wd, "_push", lambda *a: None)

        def sleeper(_s):
            clock["t"] += 30.0

        samples = [_sample(level=4), _sample(), _sample(), _sample(), _sample()]
        wd.watch(
            duration_s=100.0,
            clear_hold_s=60.0,
            ledger=tmp_path / "a.jsonl",
            sampler=self._fixed_sampler(samples),
            sleeper=sleeper,
            now=lambda: clock["t"],
        )
        assert sent[0] == (99, signal.SIGSTOP)
        assert (99, signal.SIGCONT) in sent
        # not resumed on the first clear poll -- the hold must elapse
        assert sent.index((99, signal.SIGCONT)) == 1
        rows = [json.loads(line) for line in (tmp_path / "a.jsonl").read_text().strip().splitlines()]
        assert any(row["action"] and row["action"]["kind"] == "SIGCONT" for row in rows)

    def test_a_stopped_cell_is_always_resumed_on_exit(self, tmp_path, monkeypatch):
        """A watchdog that pauses a run and then exits is worse than no watchdog at all."""
        sent = []
        monkeypatch.setattr(wd, "_signal_tree", lambda pid, sig: sent.append((pid, sig)) or {"delivered": [pid]})
        monkeypatch.setattr(wd, "newest_training_cell", lambda: {"cell_id": "c", "pid": 9, "stop_pid": 99})
        monkeypatch.setattr(wd, "_push", lambda *a: None)
        wd.watch(
            duration_s=0,
            ledger=tmp_path / "a.jsonl",
            sampler=self._fixed_sampler([_sample(level=4)]),
            sleeper=lambda _s: None,
            now=lambda: 0.0,
        )
        assert sent == [(99, signal.SIGSTOP), (99, signal.SIGCONT)]

    def test_a_second_critical_does_not_stop_a_second_cell(self, tmp_path, monkeypatch):
        sent = []
        monkeypatch.setattr(wd, "_signal_tree", lambda pid, sig: sent.append((pid, sig)) or {"delivered": [pid]})
        monkeypatch.setattr(wd, "newest_training_cell", lambda: {"cell_id": "c", "pid": 9, "stop_pid": 99})
        monkeypatch.setattr(wd, "_push", lambda *a: None)
        clock = {"t": 0.0}
        wd.watch(
            duration_s=10.0,
            ledger=tmp_path / "a.jsonl",
            sampler=self._fixed_sampler([_sample(level=4), _sample(level=4)]),
            sleeper=lambda _s: clock.__setitem__("t", clock["t"] + 5.0),
            now=lambda: clock["t"],
        )
        stops = [entry for entry in sent if entry[1] == signal.SIGSTOP]
        assert len(stops) == 1

    def test_warn_alarms_without_acting(self, tmp_path, monkeypatch):
        sent = []
        monkeypatch.setattr(wd, "_signal_tree", lambda pid, sig: sent.append((pid, sig)) or {})
        monkeypatch.setattr(wd, "_push", lambda *a: None)
        wd.watch(
            duration_s=0,
            ledger=tmp_path / "a.jsonl",
            sampler=self._fixed_sampler([_sample(compressor=20.0)]),
            sleeper=lambda _s: None,
            now=lambda: 0.0,
        )
        assert sent == []
        row = json.loads((tmp_path / "a.jsonl").read_text().strip().splitlines()[0])
        assert row["level"] == wd.LEVEL_WARN and row["action"] is None


class TestAlarmLedger:
    def test_rows_are_typed_and_non_promotable(self, tmp_path):
        row = wd.emit_alarm(wd.LEVEL_WARN, _sample(), ["because"], None, ledger=tmp_path / "a.jsonl")
        assert row["schema"] == wd.ALARM_SCHEMA
        assert row["event"] == "confound_alarm"
        assert row["score_claim"] is False
        assert "derivation" in row

    def test_append_is_never_a_rewrite(self, tmp_path):
        ledger = tmp_path / "a.jsonl"
        wd.emit_alarm(wd.LEVEL_WARN, _sample(), ["a"], None, ledger=ledger)
        wd.emit_alarm(wd.LEVEL_WARN, _sample(), ["b"], None, ledger=ledger)
        assert len(ledger.read_text().strip().splitlines()) == 2

    def test_default_ledger_lives_in_omx_state(self):
        assert wd.ALARM_LEDGER.parent == _REPO / ".omx" / "state"


class TestLiveIntegration:
    """Read-only against the real machine.  Never signals anything."""

    def test_once_reads_the_real_machine(self, capsys):
        rc = wd.main(["once"])
        assert rc in (0, 1, 2)
        payload = json.loads(capsys.readouterr().out)
        assert payload["sample"]["total_gib"] > 0
        assert payload["score_claim"] is False

    def test_thresholds_carry_their_derivation(self):
        thresholds = wd.thresholds_dict(128.0)
        assert thresholds["critical_compressor_gib"] == 48.0
        assert thresholds["warn_compressor_gib"] == 16.0
        assert "2026-09-04" in thresholds["derived_from"]
        assert "28.99%" in thresholds["rejected_trigger"]

    def test_newest_cell_helper_never_raises(self):
        wd.newest_training_cell()
