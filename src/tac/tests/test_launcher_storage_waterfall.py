"""Tests for the launcher's storage waterfall (ddm_gov2, mechanism 5).

CLAUDE.md's "Local Disk, SSD Spill, Auto-Cleanup, And Provenance" says to *fail closed if no
SSD/local tier has enough free space*.  That rule had no enforcement point, and on 2026-09-04 the
boot volume reached **344 MiB free** and threw ENOSPC twice mid-run.

Two facts from the dk1 arm shape what a REFUSAL must say, and are pinned here:
deleting bulk frees ~0 bytes until APFS local snapshots are thinned, and the launcher must never
thin them itself.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_MODULE_PATH = _REPO / "tools" / "launch_detached_process.py"

_spec = importlib.util.spec_from_file_location("launch_detached_process_under_test", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
ldp = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ldp
_spec.loader.exec_module(ldp)


class TestVolumeResolution:
    def test_boot_and_data_volume_agree_under_disk_usage(self):
        """dk1 warned that ``df /`` reads the SEALED system volume.

        MEASURED here: ``shutil.disk_usage`` already reports the DATA volume for both paths
        (211.1 GiB free from either on this box), so the hazard is a ``df`` hazard. We resolve to
        ``/System/Volumes/Data`` explicitly anyway so the intent survives a refactor -- this test
        pins the agreement rather than the coincidence.
        """
        root_free = ldp._free_gib(Path("/"))
        data_free = ldp._free_gib(ldp.BOOT_DATA_VOLUME)
        if root_free is None or data_free is None:
            pytest.skip("disk_usage unavailable")
        assert abs(root_free - data_free) < 1.0

    def test_external_volumes_are_not_the_boot_volume(self):
        assert ldp._is_on_boot_volume(Path("/Users/x/Projects/pact/.omx/tmp")) is True
        assert ldp._is_on_boot_volume(Path("/Volumes/APDataStore/pact/x")) is False

    def test_free_space_resolves_through_a_path_that_does_not_exist_yet(self, tmp_path):
        """An output dir is created BY the launch, so the check must walk up to a live ancestor."""
        assert ldp._free_gib(tmp_path / "not" / "yet" / "there") is not None

    def test_existing_ancestor_terminates_at_root(self):
        assert ldp._existing_ancestor(Path("/definitely/not/here/at/all")).exists()


class TestWaterfall:
    def test_passes_on_a_tier_with_room(self, monkeypatch, tmp_path):
        monkeypatch.setattr(ldp, "_free_gib", lambda path: 88.0)
        record = ldp._storage_waterfall(
            Path("/Volumes/VertigoDataTier/pact/x"), cmd=["python", "t.py"], artifact_budget_gib=2.0
        )
        assert record["verdict"] == "PASS"
        assert record["artifact_budget_gib"] == 2.0

    def test_refuses_a_boot_volume_launch_below_the_floor(self, monkeypatch):
        """The 2026-09-04 state, verbatim: 344 MiB free on the boot data volume."""
        monkeypatch.setattr(ldp, "_free_gib", lambda path: 0.336)
        monkeypatch.setattr(ldp, "_local_snapshot_report", lambda: {"available": True, "count": 21})
        with pytest.raises(ldp.LaunchRefusal) as excinfo:
            ldp._storage_waterfall(
                Path("/Users/x/Projects/pact/.omx/tmp/run"), cmd=["python", "t.py"], artifact_budget_gib=0.1
            )
        assert excinfo.value.rc == 11
        assert "boot volume" in str(excinfo.value)

    def test_refusal_reports_snapshot_pinning_so_a_delete_is_not_the_default_cure(self, monkeypatch):
        """dk1 MEASURED: a certified 32.97 GiB delete freed +1 GiB; thinning released +65 GiB."""
        monkeypatch.setattr(ldp, "_free_gib", lambda path: 0.336)
        monkeypatch.setattr(
            ldp, "_local_snapshot_report", lambda: {"available": True, "count": 21, "note": "pinned"}
        )
        with pytest.raises(ldp.LaunchRefusal) as excinfo:
            ldp._storage_waterfall(Path("/Users/x/p/out"), cmd=["python", "t.py"], artifact_budget_gib=0.1)
        detail = excinfo.value.detail
        assert detail["local_snapshots"]["count"] == 21
        assert "thin" in detail["cure"].lower()

    def test_refuses_when_the_target_volume_cannot_hold_the_artifact_budget(self, monkeypatch):
        monkeypatch.setattr(ldp, "_free_gib", lambda path: 3.0)
        monkeypatch.setattr(ldp, "_local_snapshot_report", lambda: {"available": False})
        with pytest.raises(ldp.LaunchRefusal) as excinfo:
            ldp._storage_waterfall(
                Path("/Volumes/APDataStore/pact/x"), cmd=["python", "t.py"], artifact_budget_gib=20.0
            )
        assert excinfo.value.rc == 11
        assert "artifact budget" in str(excinfo.value)

    def test_the_refusal_names_the_tier_with_the_most_room(self, monkeypatch):
        free = {"/Volumes/VertigoDataTier/pact": 88.0, "/Volumes/APDataStore/pact": 17.0}
        monkeypatch.setattr(ldp, "_free_gib", lambda path: free.get(str(path), 0.5))
        monkeypatch.setattr(ldp, "_local_snapshot_report", lambda: {"available": False})
        with pytest.raises(ldp.LaunchRefusal) as excinfo:
            ldp._storage_waterfall(Path("/Users/x/p/out"), cmd=["python", "t.py"], artifact_budget_gib=1.0)
        assert "VertigoDataTier" in excinfo.value.detail["cure"]

    def test_the_launcher_never_thins_snapshots(self):
        """Thinning is destructive and operator-level; a launcher must only refuse with a reason.

        Checked over the AST, not the raw text: the module deliberately NAMES
        ``tmutil thinlocalsnapshots`` in prose while explaining the cure, and a substring scan
        would flag that explanation (it did, on the first pass).
        """
        import ast

        tree = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))
        offenders = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            for sub in ast.walk(node)
            if isinstance(sub, ast.Constant)
            and isinstance(sub.value, str)
            and "thinlocalsnapshots" in sub.value
        ]
        assert offenders == [], f"the launcher must not invoke thinning (lines {offenders})"

    def test_a_boot_launch_with_room_is_allowed(self, monkeypatch):
        """The floor is a floor, not a ban: the boot volume is a legal target when it has room."""
        monkeypatch.setattr(ldp, "_free_gib", lambda path: 211.0)
        record = ldp._storage_waterfall(Path("/Users/x/p/out"), cmd=["python", "t.py"], artifact_budget_gib=2.0)
        assert record["verdict"] == "PASS"
        assert record["on_boot_volume"] is True


class TestArtifactBudgetProvenance:
    def test_defaults_to_the_measured_family_size_when_one_exists(self, tmp_path, monkeypatch):
        import json

        ledger = tmp_path / "peaks.jsonl"
        ledger.write_text(
            json.dumps(
                {"schema": "measured_peak.v1", "family": "trainer", "governed_peak_gib": 1.0, "artifact_gib": 1.1733}
            )
            + "\n"
        )
        monkeypatch.setenv("TAC_MEASURED_PEAKS_LEDGER", str(ledger))
        budget, provenance = ldp._measured_artifact_budget_gib(["python", "/p/trainer.py", "run-config", "/c.json"])
        assert budget == pytest.approx(2.3466, abs=1e-3), "2x the measured 1.1733 GiB"
        assert provenance.startswith("MEASURED")

    def test_falls_back_to_a_floor_for_an_unmeasured_family(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TAC_MEASURED_PEAKS_LEDGER", str(tmp_path / "absent.jsonl"))
        budget, provenance = ldp._measured_artifact_budget_gib(["python", "/p/never_run.py"])
        assert budget == ldp.FALLBACK_ARTIFACT_BUDGET_GIB
        assert provenance.startswith("FALLBACK")

    def test_a_broken_ledger_never_blocks_a_launch(self, tmp_path, monkeypatch):
        bad = tmp_path / "peaks.jsonl"
        bad.write_text("not json\n")
        monkeypatch.setenv("TAC_MEASURED_PEAKS_LEDGER", str(bad))
        budget, _ = ldp._measured_artifact_budget_gib(["python", "/p/trainer.py"])
        assert budget == ldp.FALLBACK_ARTIFACT_BUDGET_GIB

    def test_an_operator_declaration_overrides_the_derivation(self, monkeypatch):
        monkeypatch.setattr(ldp, "_free_gib", lambda path: 88.0)
        record = ldp._storage_waterfall(
            Path("/Volumes/VertigoDataTier/pact/x"), cmd=["python", "t.py"], artifact_budget_gib=7.5
        )
        assert record["artifact_budget_gib"] == 7.5
        assert record["artifact_budget_provenance"] == "operator-declared --artifact-budget-gib"


class TestSnapshotCensus:
    def test_census_is_fail_open(self, monkeypatch):
        def boom(*_a, **_k):
            raise OSError("tmutil missing")

        monkeypatch.setattr(ldp.subprocess, "run", boom)
        report = ldp._local_snapshot_report()
        assert report["available"] is False and "error" in report

    def test_census_counts_snapshots_on_the_real_machine(self):
        report = ldp._local_snapshot_report()
        assert "available" in report
        if report["available"]:
            assert isinstance(report["count"], int)


def test_the_flag_exists_and_defaults_to_derivation():
    text = _MODULE_PATH.read_text(encoding="utf-8")
    assert '"--artifact-budget-gib"' in text
    assert "storage_waterfall" in text, "the record must reach the manifest"
