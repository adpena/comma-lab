# SPDX-License-Identifier: MIT
"""Tests for tools/check_modal_harvest_freshness.py."""
from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "check_modal_harvest_freshness.py"


def _load_module():
    import sys as _sys

    name = "check_modal_harvest_freshness"
    if name in _sys.modules:
        return _sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TOOL_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    _sys.modules[name] = mod  # required for dataclasses with module lookup
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load_module()


def _make_metadata(parent: Path, name: str, harvested: bool) -> Path:
    lane = parent / f"lane_{name}_modal"
    lane.mkdir(parents=True)
    meta = lane / "modal_metadata.json"
    meta.write_text(json.dumps({"label": name, "call_id": "fc-test"}))
    if harvested:
        (lane / "harvested_artifacts").mkdir()
    return meta


def _setup_repo_root(tmp_path: Path, *, harvested: list[str], unharvested: list[str]) -> Path:
    results = tmp_path / "experiments" / "results"
    results.mkdir(parents=True)
    for n in harvested:
        _make_metadata(results, n, harvested=True)
    for n in unharvested:
        _make_metadata(results, n, harvested=False)
    return tmp_path


def test_find_unharvested_returns_only_missing_dirs(tmp_path, mod):
    repo = _setup_repo_root(tmp_path, harvested=["a", "b"], unharvested=["c"])
    out = mod.find_unharvested(repo)
    assert len(out) == 1
    assert out[0].parent.name == "lane_c_modal"


def test_find_unharvested_handles_missing_results_dir(tmp_path, mod):
    assert mod.find_unharvested(tmp_path) == []


def test_build_report_no_summary_no_pending_is_fresh(tmp_path, mod):
    repo = _setup_repo_root(tmp_path, harvested=[], unharvested=[])
    rep = mod.build_report(repo)
    assert rep.summary_exists is False
    assert rep.unharvested_count == 0
    assert rep.is_stale is False
    assert "no pending" in rep.note.lower() or "no harvest summary and no pending" in rep.note


def test_build_report_no_summary_with_pending_is_stale(tmp_path, mod):
    repo = _setup_repo_root(tmp_path, harvested=[], unharvested=["x"])
    rep = mod.build_report(repo)
    assert rep.summary_exists is False
    assert rep.is_stale is True
    assert rep.suggested_command is not None
    assert "harvest_modal_calls.py --execute" in rep.suggested_command


def test_build_report_fresh_summary_below_threshold(tmp_path, mod):
    repo = _setup_repo_root(tmp_path, harvested=[], unharvested=["x"])
    summary = repo / "experiments" / "results" / "_modal_harvest_summary.json"
    summary.write_text("[]")
    # mtime now - very fresh
    rep = mod.build_report(repo, threshold_hours=4.0, now=time.time())
    assert rep.summary_exists is True
    assert rep.is_stale is False
    assert rep.unharvested_count == 1


def test_build_report_stale_summary_with_pending(tmp_path, mod):
    repo = _setup_repo_root(tmp_path, harvested=[], unharvested=["x"])
    summary = repo / "experiments" / "results" / "_modal_harvest_summary.json"
    summary.write_text("[]")
    # Force mtime to 10h ago
    old = time.time() - (10 * 3600)
    import os
    os.utime(summary, (old, old))
    rep = mod.build_report(repo, threshold_hours=4.0)
    assert rep.is_stale is True
    assert rep.summary_age_hours is not None and rep.summary_age_hours > 4.0


def test_build_report_stale_summary_but_no_pending_is_not_stale(tmp_path, mod):
    repo = _setup_repo_root(tmp_path, harvested=["a"], unharvested=[])
    summary = repo / "experiments" / "results" / "_modal_harvest_summary.json"
    summary.write_text("[]")
    import os
    old = time.time() - (10 * 3600)
    os.utime(summary, (old, old))
    rep = mod.build_report(repo, threshold_hours=4.0)
    # Stale-by-time but nothing pending => not actionable; is_stale=False
    assert rep.is_stale is False
    assert rep.suggested_command is None


def test_main_returns_1_on_stale(tmp_path, mod, capsys):
    repo = _setup_repo_root(tmp_path, harvested=[], unharvested=["x"])
    rc = mod.main(["--repo-root", str(repo), "--threshold-hours", "4.0"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "STALE" in out or "is_stale:            True" in out


def test_main_returns_0_on_fresh(tmp_path, mod, capsys):
    repo = _setup_repo_root(tmp_path, harvested=[], unharvested=[])
    rc = mod.main(["--repo-root", str(repo)])
    assert rc == 0


def test_main_json_emits_full_report(tmp_path, mod, capsys):
    repo = _setup_repo_root(tmp_path, harvested=[], unharvested=["x"])
    rc = mod.main(["--repo-root", str(repo), "--json"])
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert "is_stale" in parsed
    assert "unharvested_count" in parsed
    assert parsed["unharvested_count"] == 1
    # rc=1 because is_stale (no summary + pending)
    assert rc == 1


def test_main_returns_2_on_missing_repo(tmp_path, mod, capsys):
    rc = mod.main(["--repo-root", str(tmp_path / "nope")])
    assert rc == 2


def test_render_text_includes_action_when_stale(tmp_path, mod):
    repo = _setup_repo_root(tmp_path, harvested=[], unharvested=["x"])
    rep = mod.build_report(repo)
    out = mod.render_text(rep)
    assert "ACTION:" in out
    assert "harvest_modal_calls.py" in out


def test_render_text_no_action_when_fresh(tmp_path, mod):
    repo = _setup_repo_root(tmp_path, harvested=[], unharvested=[])
    rep = mod.build_report(repo)
    out = mod.render_text(rep)
    assert "ACTION:" not in out


def test_main_handles_live_repo_smoke():
    """Smoke-test against live repo if present; never hard-fail."""
    if not (REPO_ROOT / "experiments" / "results").exists():
        pytest.skip("live results dir not present")
    mod = _load_module()
    rc = mod.main(["--repo-root", str(REPO_ROOT), "--json"])
    assert rc in (0, 1)  # accepts either fresh or stale; just must not crash
