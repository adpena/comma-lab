"""Tests for tools/costate_digest.py section_telemetry_binding — the #404 P0 wire-in.

The digest surfaces the binding-vs-inert lever readback (amber clip rate / chroma share /
pose-gate liveness / EMA-lag / D27b terminal-band) for the live run. Read-only + score-neutral
=> defaults ON. Fail-open: a broken analyzer must never break a session; absent telemetry omits
the row."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import costate_digest as cd  # noqa: E402


def _write_run_log(run_dir: Path, *, n_verdict: int = 12) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"stage": "witness_stability_resolved", "grad_clip": 0.5, "per_group_grad_clip": True}]
    for i in range(n_verdict):
        ep = 25 * (i + 1)
        rows.append({"stage": "loss_terms", "ep": ep, "accum_batch": 0,
                     "terms": {"seg": 0.5, "pose": 0.1, "chroma_boundary": 0.0},
                     "total": 1.0, "gnorm": 0.1})
        rows.append({"stage": "verdict", "epoch": ep, "d_seg": 0.01, "d_pose": 0.001,
                     "implied_S": 0.4})
        rows.append({"stage": "jacobian_basin", "epoch": ep, "median_sigma_min": 0.01,
                     "sigma_min_plateau_est": 0.01, "would_have_fired": False})
    (run_dir / "run.log").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_section_present_on_telemetry(tmp_path) -> None:
    _write_run_log(tmp_path)
    line, data = cd.section_telemetry_binding(tmp_path)
    assert line is not None and line.startswith("telem-binding:")
    assert "amber=INERT_NEVER_BINDS" in line
    assert data is not None
    assert data["pose_gate"]["verdict"] == "OK"
    assert data["terminal_band"]["d27b_ready"] is False


def test_section_omitted_when_run_dir_none() -> None:
    assert cd.section_telemetry_binding(None) == (None, None)


def test_section_omitted_on_missing_run_log(tmp_path) -> None:
    assert cd.section_telemetry_binding(tmp_path) == (None, None)


def test_section_fail_open_on_analyzer_exception(tmp_path, monkeypatch) -> None:
    _write_run_log(tmp_path)
    from tac.witness_control import telemetry_binding as tb

    def boom(*_a, **_k):  # pragma: no cover - trivial
        raise RuntimeError("synthetic")

    monkeypatch.setattr(tb, "audit_rows", boom)
    assert cd.section_telemetry_binding(tmp_path) == (None, None)


def test_build_digest_never_raises_with_telemetry_binding_wired() -> None:
    lines, data = cd.build_digest()
    assert isinstance(lines, list) and lines
    assert "telemetry_binding" in data  # key present even when None (no live run)
