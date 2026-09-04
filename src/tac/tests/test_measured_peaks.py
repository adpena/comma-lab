"""Tests for ``tools/measured_peaks.py`` -- the measured-peak ledger (ddm_gov2).

The defect under test is a HAND-TYPED memory declaration.  ng2 and ng3 both ran under
``--measured-peak-rss-gib 2.3959503173828125``; MEASURED from the real receipts on 2026-09-04 the
same family's system-availability cost was **49.572 GiB**, a 20.7x under-declaration, and two such
cells running concurrently drove the VM compressor to 76.978 GiB with 72.0 GiB of swap.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_MODULE_PATH = _REPO / "tools" / "measured_peaks.py"

_spec = importlib.util.spec_from_file_location("measured_peaks", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
mp = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("measured_peaks", mp)
_spec.loader.exec_module(mp)


def _receipt(path: Path, **overrides) -> Path:
    payload = {
        "schema": "safe_run_status_receipt.v1",
        "argv": ["/p/python", "/p/experiments/some_trainer.py", "run-config", "/cfg.json"],
        "status": "ok",
        "exit": 0,
        "elapsed_s": 600.0,
        "start_utc": "2026-09-04T21:55:36Z",
        "peak_rss_mib": 1787.703,
        "peak_rss_observed": True,
    }
    payload.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def _blackbox(path: Path, rows: list[tuple[str, float]]) -> Path:
    path.write_text("\n".join(json.dumps({"ts_iso": ts, "available_gib": value}) for ts, value in rows) + "\n")
    return path


class TestRowFromStatusReceipt:
    def test_reads_peak_rss_from_an_untouched_receipt(self, tmp_path):
        """Works on SEALED-SOURCE launcher output: nothing had to be instrumented at launch."""
        receipt = _receipt(tmp_path / "launch" / "resource_safe_run_status.json")
        row = mp.row_from_status_receipt(receipt, ledger=tmp_path / "l.jsonl")
        assert row is not None
        assert row.peak_rss_gib == pytest.approx(1787.703 / 1024.0, abs=1e-6)
        assert row.peak_rss_observed is True

    def test_family_is_the_trainer_entry_point_not_the_seed(self, tmp_path):
        """Three cells of one trainer share a memory profile; keying on cell_id gives none a row."""
        receipt = _receipt(tmp_path / "resource_safe_run_status.json")
        row = mp.row_from_status_receipt(receipt, ledger=tmp_path / "l.jsonl")
        assert row.family == "some_trainer"

    def test_rejects_a_foreign_json_file(self, tmp_path):
        other = tmp_path / "resource_safe_run_status.json"
        other.write_text(json.dumps({"schema": "something_else"}))
        assert mp.row_from_status_receipt(other, ledger=tmp_path / "l.jsonl") is None

    def test_governed_peak_takes_the_availability_delta_when_rss_is_blind(self, tmp_path):
        """THE ng4 CASE, MEASURED: RSS 1.746 GiB, availability delta 49.572 GiB (28.4x).

        ``ps rss`` cannot see a Metal allocation, so on a GPU cell the RSS peak is a floor and the
        availability delta is the real cost.
        """
        receipt = _receipt(tmp_path / "resource_safe_run_status.json")
        blackbox = _blackbox(
            tmp_path / "bb.jsonl",
            [
                ("2026-09-04T21:55:00+00:00", 77.609),
                ("2026-09-04T21:56:00+00:00", 28.037),
                ("2026-09-04T22:00:00+00:00", 30.0),
            ],
        )
        row = mp.row_from_status_receipt(receipt, blackbox_path=blackbox, ledger=tmp_path / "l.jsonl")
        assert row.pre_launch_available_gib == pytest.approx(77.609)
        assert row.min_available_while_live_gib == pytest.approx(28.037)
        assert row.system_availability_delta_gib == pytest.approx(49.572, abs=1e-3)
        assert row.governed_peak_gib == pytest.approx(49.572, abs=1e-3)
        assert row.attribution_grade == mp.GRADE_SOLE_CELL

    def test_governed_peak_takes_rss_when_rss_is_the_larger_instrument(self, tmp_path):
        """The CPU-smoke regime, also MEASURED: RSS 40.9 GiB vs availability delta 22.4 GiB.

        Both instruments are partial in opposite regimes, so the fail-closed answer is the max.
        """
        receipt = _receipt(tmp_path / "resource_safe_run_status.json", peak_rss_mib=40.903 * 1024)
        blackbox = _blackbox(
            tmp_path / "bb.jsonl",
            [("2026-09-04T21:55:00+00:00", 60.0), ("2026-09-04T21:56:00+00:00", 37.6)],
        )
        row = mp.row_from_status_receipt(receipt, blackbox_path=blackbox, ledger=tmp_path / "l.jsonl")
        assert row.system_availability_delta_gib == pytest.approx(22.4, abs=0.01)
        assert row.governed_peak_gib == pytest.approx(40.903, abs=0.01)

    def test_missing_blackbox_coverage_is_graded_not_guessed(self, tmp_path):
        """ng2 started 2026-09-04T14:16Z; the blackbox on disk starts 16:12Z. No pre-launch row."""
        receipt = _receipt(tmp_path / "resource_safe_run_status.json")
        blackbox = _blackbox(tmp_path / "bb.jsonl", [("2026-09-04T23:00:00+00:00", 50.0)])
        row = mp.row_from_status_receipt(receipt, blackbox_path=blackbox, ledger=tmp_path / "l.jsonl")
        assert row.system_availability_delta_gib is None
        assert row.attribution_grade == mp.GRADE_NO_PRE_LAUNCH

    def test_pre_launch_reading_must_be_recent(self, tmp_path):
        """A sample from an hour before the launch describes a different machine."""
        receipt = _receipt(tmp_path / "resource_safe_run_status.json")
        blackbox = _blackbox(
            tmp_path / "bb.jsonl",
            [("2026-09-04T20:00:00+00:00", 100.0), ("2026-09-04T21:56:00+00:00", 20.0)],
        )
        row = mp.row_from_status_receipt(receipt, blackbox_path=blackbox, ledger=tmp_path / "l.jsonl")
        assert row.pre_launch_available_gib is None
        assert row.attribution_grade == mp.GRADE_NO_PRE_LAUNCH

    def test_overlapping_cell_confounds_the_attribution(self, tmp_path):
        """Two live cells means the availability delta is not attributable to either one."""
        ledger = tmp_path / "l.jsonl"
        mp.append_row(
            {
                "schema": mp.MEASURED_PEAK_SCHEMA,
                "family": "other",
                "start_utc": "2026-09-04T21:50:00Z",
                "elapsed_s": 3600.0,
                "status_receipt_path": "/other/resource_safe_run_status.json",
                "governed_peak_gib": 1.0,
            },
            ledger,
        )
        receipt = _receipt(tmp_path / "resource_safe_run_status.json")
        blackbox = _blackbox(
            tmp_path / "bb.jsonl",
            [("2026-09-04T21:55:00+00:00", 77.6), ("2026-09-04T21:56:00+00:00", 28.0)],
        )
        row = mp.row_from_status_receipt(receipt, blackbox_path=blackbox, ledger=ledger)
        assert row.attribution_grade == mp.GRADE_CONFOUNDED
        assert row.system_availability_delta_gib is not None, "still recorded, just labelled"

    def test_availability_delta_is_never_negative(self, tmp_path):
        receipt = _receipt(tmp_path / "resource_safe_run_status.json")
        blackbox = _blackbox(
            tmp_path / "bb.jsonl",
            [("2026-09-04T21:55:00+00:00", 10.0), ("2026-09-04T21:56:00+00:00", 90.0)],
        )
        row = mp.row_from_status_receipt(receipt, blackbox_path=blackbox, ledger=tmp_path / "l.jsonl")
        assert row.system_availability_delta_gib == 0.0

    def test_declared_peak_is_carried_so_the_gap_stays_visible(self, tmp_path):
        """The 2.396 GiB fiction must be recorded NEXT TO the truth, not silently replaced."""
        manifest = tmp_path / "launch_manifest.json"
        manifest.write_text(json.dumps({"resource_budget": {"measured_peak_rss_gib": 2.3959503173828125}}))
        receipt = _receipt(tmp_path / "resource_safe_run_status.json")
        row = mp.row_from_status_receipt(receipt, manifest_path=manifest, ledger=tmp_path / "l.jsonl")
        assert row.declared_peak_gib == pytest.approx(2.3959503173828125)

    def test_manifest_is_found_beside_the_receipt_without_being_named(self, tmp_path):
        (tmp_path / "launch_manifest.json").write_text(json.dumps({"resource_budget": {"measured_peak_rss_gib": 7.0}}))
        receipt = _receipt(tmp_path / "resource_safe_run_status.json")
        row = mp.row_from_status_receipt(receipt, ledger=tmp_path / "l.jsonl")
        assert row.declared_peak_gib == 7.0


class TestLedger:
    def test_append_and_read_round_trip(self, tmp_path):
        ledger = tmp_path / "l.jsonl"
        receipt = _receipt(tmp_path / "resource_safe_run_status.json")
        row = mp.row_from_status_receipt(receipt, ledger=ledger)
        mp.append_row(row.as_dict(), ledger)
        rows = mp.read_rows(ledger)
        assert len(rows) == 1 and rows[0]["family"] == "some_trainer"

    def test_rows_of_a_foreign_schema_are_ignored(self, tmp_path):
        ledger = tmp_path / "l.jsonl"
        ledger.write_text(json.dumps({"schema": "other.v1"}) + "\nnot json\n")
        assert mp.read_rows(ledger) == []

    def test_lookup_takes_the_maximum_not_the_mean(self, tmp_path):
        """A reservation that holds for the average run fails on the run that matters."""
        ledger = tmp_path / "l.jsonl"
        for peak in (1.6, 49.572, 2.4):
            mp.append_row(
                {"schema": mp.MEASURED_PEAK_SCHEMA, "family": "f", "governed_peak_gib": peak},
                ledger,
            )
        found = mp.lookup_family("f", path=ledger)
        assert found["governed_peak_gib"] == pytest.approx(49.572)
        assert found["row_count"] == 3

    def test_lookup_of_an_unknown_family_is_none(self, tmp_path):
        assert mp.lookup_family("never-run", path=tmp_path / "absent.jsonl") is None

    def test_ledger_path_honours_the_env_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TAC_MEASURED_PEAKS_LEDGER", str(tmp_path / "custom.jsonl"))
        assert mp.ledger_path() == tmp_path / "custom.jsonl"

    def test_default_ledger_lives_in_omx_state(self):
        assert mp.MEASURED_PEAKS_LEDGER.parent == _REPO / ".omx" / "state"
        assert mp.MEASURED_PEAKS_LEDGER.name == "measured_peaks.jsonl"

    def test_append_is_never_a_rewrite(self, tmp_path):
        ledger = tmp_path / "l.jsonl"
        mp.append_row({"schema": mp.MEASURED_PEAK_SCHEMA, "family": "a"}, ledger)
        mp.append_row({"schema": mp.MEASURED_PEAK_SCHEMA, "family": "b"}, ledger)
        assert len(ledger.read_text().strip().splitlines()) == 2


class TestHarvest:
    def test_finds_receipts_and_prunes_bulk(self, tmp_path):
        keep = _receipt(tmp_path / "launch" / "cell" / "resource_safe_run_status.json")
        for bulk in ("retained", "runs", "sealed_source_abc", "step_000100"):
            _receipt(tmp_path / bulk / "resource_safe_run_status.json")
        assert mp.find_status_receipts(tmp_path) == [keep]

    def test_missing_root_is_tolerated(self, tmp_path):
        assert mp.find_status_receipts(tmp_path / "nope") == []

    def test_already_recorded_skips_an_unchanged_run(self, tmp_path):
        ledger = tmp_path / "l.jsonl"
        receipt = _receipt(tmp_path / "launch" / "resource_safe_run_status.json")
        assert mp.already_recorded(receipt, ledger=ledger) is False
        row = mp.row_from_status_receipt(receipt, ledger=ledger)
        mp.append_row(row.as_dict(), ledger)
        assert mp.already_recorded(receipt, ledger=ledger) is True

    def test_a_run_that_moved_on_is_recorded_again(self, tmp_path):
        """A LIVE cell's receipt updates every poll; a longer elapsed time is a new observation."""
        ledger = tmp_path / "l.jsonl"
        receipt = _receipt(tmp_path / "launch" / "resource_safe_run_status.json")
        mp.append_row(mp.row_from_status_receipt(receipt, ledger=ledger).as_dict(), ledger)
        _receipt(receipt, elapsed_s=1200.0)
        assert mp.already_recorded(receipt, ledger=ledger) is False

    def test_cli_harvest_dry_run_writes_nothing(self, tmp_path, capsys):
        _receipt(tmp_path / "launch" / "resource_safe_run_status.json")
        ledger = tmp_path / "l.jsonl"
        rc = mp.main(["harvest", "--root", str(tmp_path), "--ledger", str(ledger), "--dry-run"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["recorded"] == 1
        assert payload["score_claim"] is False
        assert not ledger.exists()


class TestCLI:
    def test_record_then_lookup(self, tmp_path, capsys):
        ledger = tmp_path / "l.jsonl"
        receipt = _receipt(tmp_path / "launch" / "resource_safe_run_status.json")
        assert mp.main(["record", "--status-receipt", str(receipt), "--ledger", str(ledger)]) == 0
        capsys.readouterr()
        assert mp.main(["lookup", "--family", "some_trainer", "--ledger", str(ledger)]) == 0
        assert json.loads(capsys.readouterr().out)["found"] is True

    def test_lookup_missing_family_returns_two(self, tmp_path, capsys):
        rc = mp.main(["lookup", "--family", "nope", "--ledger", str(tmp_path / "l.jsonl")])
        assert rc == 2
        assert json.loads(capsys.readouterr().out)["found"] is False

    def test_record_of_a_non_receipt_returns_two(self, tmp_path, capsys):
        bad = tmp_path / "resource_safe_run_status.json"
        bad.write_text("{}")
        assert mp.main(["record", "--status-receipt", str(bad), "--ledger", str(tmp_path / "l.jsonl")]) == 2
        capsys.readouterr()

    def test_families_lists_every_measured_family(self, tmp_path, capsys):
        ledger = tmp_path / "l.jsonl"
        for family in ("a", "b"):
            mp.append_row({"schema": mp.MEASURED_PEAK_SCHEMA, "family": family, "governed_peak_gib": 1.0}, ledger)
        assert mp.main(["families", "--ledger", str(ledger)]) == 0
        assert json.loads(capsys.readouterr().out)["count"] == 2


class TestBlackboxWindow:
    def test_absent_blackbox_is_tolerated(self, tmp_path):
        assert mp.blackbox_window(dt.datetime.now(dt.UTC), blackbox_path=tmp_path / "nope.jsonl") == (None, None, 0)

    def test_malformed_lines_are_skipped(self, tmp_path):
        path = tmp_path / "bb.jsonl"
        path.write_text('{"available_gib": 1\nnot json\n' + json.dumps({"ts_iso": "x", "available_gib": 5.0}) + "\n")
        assert mp.blackbox_window(dt.datetime(2026, 9, 4, tzinfo=dt.UTC), blackbox_path=path) == (None, None, 0)

    def test_counts_only_samples_inside_the_run_window(self, tmp_path):
        path = _blackbox(
            tmp_path / "bb.jsonl",
            [
                ("2026-09-04T21:55:00+00:00", 70.0),
                ("2026-09-04T21:56:00+00:00", 30.0),
                ("2026-09-04T23:59:00+00:00", 5.0),
            ],
        )
        start = dt.datetime(2026, 9, 4, 21, 55, 36, tzinfo=dt.UTC)
        end = dt.datetime(2026, 9, 4, 22, 0, 0, tzinfo=dt.UTC)
        pre, low, count = mp.blackbox_window(start, end, blackbox_path=path)
        assert (pre, low, count) == (70.0, 30.0, 1), "the 23:59 sample is outside the run"


def test_module_makes_no_score_claim():
    text = _MODULE_PATH.read_text(encoding="utf-8")
    assert '"score_claim": False' in text or '"score_claim"] = False' in text
