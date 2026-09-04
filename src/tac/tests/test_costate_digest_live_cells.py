"""Tests for ``costate_digest.section_live_cells`` -- the live-cell SENSE surface (ddm_gv1).

The digest is a SessionStart hook: it must never crash a session, never sleep, never actuate, and
never make a score claim.  These tests pin exactly that, plus the content the operator reads
(progress, rate, ETA, admission headroom, contention ratio), and the wire-in into ``build_digest``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_DIGEST_PATH = _REPO / "tools" / "costate_digest.py"
_ADMISSION_PATH = _REPO / "tools" / "cell_admission.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ca = _load("cell_admission_for_digest_test", _ADMISSION_PATH)


@pytest.fixture
def digest(monkeypatch):
    """Import the digest with ``tools`` on the path (it imports ``cell_admission`` lazily)."""
    monkeypatch.syspath_prepend(str(_REPO / "tools"))
    if "costate_digest" in sys.modules:
        del sys.modules["costate_digest"]
    import costate_digest

    return costate_digest


def _live_cell(**overrides) -> ca.LiveCell:
    base = {
        "cell_id": "cell_x",
        "pid": os.getpid(),
        "alive": True,
        "declared_peak_gib": 41.5,
        "current_rss_gib": 0.6,
        "manifest_path": Path("/m/launch_manifest.json"),
        "config_path": Path("/c/cfg.json"),
        "run_dir": Path("/r"),
        "total_steps": 5000,
        "completed_steps": 1000,
        "arm_name": "tau_band",
        "arm_role": "control",
        "purpose": "test",
    }
    base.update(overrides)
    return ca.LiveCell(**base)


def _stub(monkeypatch, digest, *, cells, rows):
    """Point the digest's lazily-imported ``cell_admission`` at fixed inputs."""
    module = sys.modules.get("cell_admission")
    if module is None:
        module = _load("cell_admission", _ADMISSION_PATH)
    monkeypatch.setattr(module, "discover_live_cells", lambda *a, **kw: list(cells))
    monkeypatch.setattr(module, "read_contention_rows", lambda *a, **kw: list(rows))
    return module


class TestSectionLiveCells:
    def test_no_live_cells_reports_none(self, digest, monkeypatch):
        _stub(monkeypatch, digest, cells=[], rows=[])
        lines, data = digest.section_live_cells()
        assert lines == ["live cells: NONE (no governed local job is alive)"]
        assert data["live_job_count"] == 0
        assert data["actuation"] == "SENSE_ONLY"
        assert data["score_claim"] is False

    def test_reports_progress_rate_and_eta(self, digest, monkeypatch):
        rows = [
            {
                "schema": ca.THROUGHPUT_ROW_SCHEMA,
                "recorded_utc": "2026-09-04T15:55:37Z",
                "concurrency": 2,
                "total_steps_per_min": 31.2854,
                "cells": [{"cell_id": "cell_x", "steps_per_min": 16.0}],
            },
            {
                "schema": ca.THROUGHPUT_ROW_SCHEMA,
                "recorded_utc": "2026-09-04T15:32:00Z",
                "concurrency": 1,
                "total_steps_per_min": 28.0,
            },
        ]
        _stub(monkeypatch, digest, cells=[_live_cell()], rows=rows)
        lines, data = digest.section_live_cells()
        body = "\n".join(lines)
        assert "1000/5000" in body
        assert "16.0/min" in body
        # 4000 remaining steps at 16/min = 250 min = 4.2 h
        assert "ETA 4.2 h" in body
        assert data["cells"][0]["eta_minutes"] == pytest.approx(250.0)

    def test_reports_the_contention_ratio_and_whether_concurrency_pays(self, digest, monkeypatch):
        rows = [
            {"schema": ca.THROUGHPUT_ROW_SCHEMA, "recorded_utc": "b", "concurrency": 2,
             "total_steps_per_min": 31.2854, "cells": []},
            {"schema": ca.THROUGHPUT_ROW_SCHEMA, "recorded_utc": "a", "concurrency": 1,
             "total_steps_per_min": 28.0},
        ]
        _stub(monkeypatch, digest, cells=[_live_cell()], rows=rows)
        body = "\n".join(digest.section_live_cells()[0])
        assert "ratio 1.12" in body
        assert "concurrency PAYS" in body

    def test_contention_costs_is_reported_when_below_serial(self, digest, monkeypatch):
        rows = [
            {"schema": ca.THROUGHPUT_ROW_SCHEMA, "recorded_utc": "b", "concurrency": 2,
             "total_steps_per_min": 20.0, "cells": []},
            {"schema": ca.THROUGHPUT_ROW_SCHEMA, "recorded_utc": "a", "concurrency": 1,
             "total_steps_per_min": 28.0},
        ]
        _stub(monkeypatch, digest, cells=[_live_cell()], rows=rows)
        assert "concurrency COSTS" in "\n".join(digest.section_live_cells()[0])

    def test_non_cell_job_is_labelled_but_still_counted(self, digest, monkeypatch):
        job = _live_cell(cell_id="plainjob", config_path=None, total_steps=None, declared_peak_gib=12.0)
        _stub(monkeypatch, digest, cells=[job], rows=[])
        lines, data = digest.section_live_cells()
        assert "not a training cell" in "\n".join(lines)
        assert data["live_job_count"] == 1
        assert data["training_cell_count"] == 0

    def test_admission_headroom_is_surfaced(self, digest, monkeypatch):
        module = _stub(monkeypatch, digest, cells=[_live_cell()], rows=[])
        monkeypatch.setattr(module.mem_basis, "conservative_free_gib", lambda default=0.0: 15.0)
        monkeypatch.setattr(module.mem_basis, "true_committed_gib", lambda default=0.0: 100.0)
        lines, data = digest.section_live_cells()
        body = "\n".join(lines)
        assert "admission:" in body
        assert "REFUSE" in body
        assert data["admission_probe_peak_gib"] == 41.5
        assert data["admission"]["verdict"] == "REFUSE"

    def test_missing_rate_degrades_honestly(self, digest, monkeypatch):
        _stub(monkeypatch, digest, cells=[_live_cell()], rows=[])
        body = "\n".join(digest.section_live_cells()[0])
        assert "rate unmeasured" in body
        assert "ETA unknown" in body

    def test_fails_open_and_never_raises(self, digest, monkeypatch):
        module = sys.modules.get("cell_admission") or _load("cell_admission", _ADMISSION_PATH)

        def _boom(*a, **kw):
            raise RuntimeError("disk on fire")

        monkeypatch.setattr(module, "discover_live_cells", _boom)
        lines, data = digest.section_live_cells()
        assert data["available"] is False
        assert data["status"] == "FAIL_OPEN"
        assert data["actuation"] == "NONE"
        assert lines and "unavailable" in lines[0]

    def test_section_payload_is_json_serializable(self, digest, monkeypatch):
        _stub(monkeypatch, digest, cells=[_live_cell()], rows=[])
        json.dumps(digest.section_live_cells()[1], default=str)

    def test_section_does_not_sleep(self, digest, monkeypatch):
        """A SessionStart hook must not block: the section reads, it never samples a window."""
        module = _stub(monkeypatch, digest, cells=[_live_cell()], rows=[])

        def _no_sleep(*a, **kw):
            raise AssertionError("section_live_cells must never sleep")

        monkeypatch.setattr(module.time, "sleep", _no_sleep)
        digest.section_live_cells()


class TestWireIn:
    def test_build_digest_calls_the_section(self, digest, monkeypatch):
        called: dict = {}

        def _section():
            called["hit"] = True
            return ["live cells: STUB"], {"available": True, "live_job_count": 0}

        monkeypatch.setattr(digest, "section_live_cells", _section)
        lines, data = digest.build_digest(include_fm=False)
        assert called.get("hit") is True
        assert "live cells: STUB" in lines
        assert data["live_cells"]["available"] is True

    def test_build_digest_survives_a_broken_section(self, digest, monkeypatch):
        """The section is fail-open internally; build_digest must still produce a digest."""
        monkeypatch.setattr(
            digest,
            "section_live_cells",
            lambda: (["live cells: unavailable (RuntimeError: x)"], {"available": False}),
        )
        lines, data = digest.build_digest(include_fm=False)
        assert data["live_cells"]["available"] is False
        assert len(lines) > 1
