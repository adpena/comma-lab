"""Tests for tools/costate_digest.py section_ncde + format_ncde_line — the #344 wire-in.

The digest surfaces the Linear-NCDE trajectory detector's score-relevant d_seg hit->solve
advisory (asymptote + ETA-to-threshold + BASIN/HANDOFF + instrument-invalid guard) for the
live run. Read-only + score-neutral => defaults ON. Fail-open: a broken probe must never
break a session; short/absent telemetry omits the row.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import costate_digest as cd  # noqa: E402


# ─────────────────────── format_ncde_line (pure formatter) ───────────────────────
def test_format_basin_fire_row() -> None:
    import math

    row = {
        "epoch": 320.0, "fire": True, "reason": "BASIN: predicted remaining ...",
        "predicted_asymptote": math.log(0.0046), "eta_handoff_epochs": 12.0,
        "remaining_descent_frac": 0.031, "fit_r2": 0.94, "stable": True,
    }
    line = cd.format_ncde_line(row)
    assert line.startswith("ncde-trajectory: ep320 d_seg BASIN-FIRE")
    assert "plateau d_seg~0.00460" in line
    assert "ETA-handoff 12ep" in line
    assert "advisory NON-PROMOTABLE" in line


def test_format_instrument_invalid_row_surfaces_guard() -> None:
    row = {
        "epoch": 1000.0, "fire": False,
        "reason": "NO-FIRE: fit not trustworthy (stable=False, r2=0.665 ...); "
                  "instrument-invalid, no verdict",
        "predicted_asymptote": None, "eta_handoff_epochs": None,
        "remaining_descent_frac": None, "fit_r2": 0.665, "stable": False,
    }
    line = cd.format_ncde_line(row)
    assert "INSTRUMENT-INVALID" in line
    assert "stable=False" in line
    # invalid rows must NOT assert an asymptote/ETA (untrustworthy instrument)
    assert "plateau" not in line and "ETA-handoff" not in line


def test_format_no_fire_still_descending() -> None:
    row = {
        "epoch": 210.0, "fire": False, "reason": "NO-FIRE: still descending ...",
        "predicted_asymptote": -6.0, "eta_handoff_epochs": None,
        "remaining_descent_frac": 0.42, "fit_r2": 0.88, "stable": True,
    }
    line = cd.format_ncde_line(row)
    assert "NO-FIRE (still descending)" in line
    assert "rem 42.0% of travelled" in line


# ─────────────────────── section_ncde (integration, fail-open) ───────────────────────
def _write_synthetic_run_log(run_dir: Path, *, n_verdict: int) -> None:
    """Emit a run.log with loss_terms (control) + a decaying-d_seg verdict series."""
    run_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for k in range(1, 40):
        rows.append({
            "stage": "loss_terms", "ep": k, "total": 1.0 / k + 0.1,
            "gnorm": 0.5, "terms": {"seg": 0.3, "island_amplify": 0.05},
            "softmax_temp": 1.0, "hosc_beta": 1.0,
        })
    for i in range(n_verdict):
        ep = 5 * (i + 1)
        d_seg = 0.02 * (0.85 ** i) + 0.004  # smooth decay toward an asymptote
        rows.append({"stage": "verdict", "epoch": ep, "d_seg": d_seg,
                     "ep_loss": d_seg * 2.0})
    (run_dir / "run.log").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_section_present_on_sufficient_telemetry(tmp_path) -> None:
    cd._NCDE_CACHE.clear()
    rd = tmp_path / "run_ok"
    _write_synthetic_run_log(rd, n_verdict=14)
    line, data = cd.section_ncde(rd)
    assert line is not None and line.startswith("ncde-trajectory:")
    assert data is not None and data.get("advisory")


def test_section_omitted_on_short_telemetry(tmp_path) -> None:
    cd._NCDE_CACHE.clear()
    rd = tmp_path / "run_short"
    _write_synthetic_run_log(rd, n_verdict=4)  # < 8 verdict pts -> no d_seg path
    line, data = cd.section_ncde(rd)
    assert line is None and data is None


def test_section_omitted_on_missing_run_log(tmp_path) -> None:
    cd._NCDE_CACHE.clear()
    assert cd.section_ncde(None) == (None, None)
    empty = tmp_path / "no_log"
    empty.mkdir()
    assert cd.section_ncde(empty) == (None, None)


def test_section_fail_open_on_probe_exception(tmp_path, monkeypatch) -> None:
    cd._NCDE_CACHE.clear()
    rd = tmp_path / "run_boom"
    _write_synthetic_run_log(rd, n_verdict=14)
    import ncde_trajectory_probe as ntp

    def boom(*_a, **_k):  # pragma: no cover - trivial
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(ntp, "run_probe", boom)
    assert cd.section_ncde(rd) == (None, None)


def test_section_cache_hit_avoids_recompute(tmp_path, monkeypatch) -> None:
    cd._NCDE_CACHE.clear()
    rd = tmp_path / "run_cache"
    _write_synthetic_run_log(rd, n_verdict=14)
    line1, _ = cd.section_ncde(rd)  # populates cache

    import ncde_trajectory_probe as ntp
    calls = {"n": 0}
    orig = ntp.run_probe

    def counting(*a, **k):  # pragma: no cover - not expected to fire on cache hit
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(ntp, "run_probe", counting)
    line2, _ = cd.section_ncde(rd)  # should hit cache, NOT call run_probe
    assert line2 == line1
    assert calls["n"] == 0


def test_build_digest_never_raises_with_ncde_wired() -> None:
    cd._NCDE_CACHE.clear()
    lines, data = cd.build_digest()
    assert lines and "ncde" in data
