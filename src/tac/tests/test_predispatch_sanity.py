"""Tests for tools/predispatch_sanity.py — the 5-gate ladder before any paid GPU dispatch.

The CRITICAL test reproduces the exact apogee_int4 failure mode: with only PR106
as a calibration anchor and predicted band [0.155, 0.180] (the historical wrong
band), all gates either pass or fail in a way that BLOCKS the dispatch. The
historical bug was that we dispatched anyway; this gate would have caught it.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PREDISPATCH = REPO_ROOT / "tools" / "predispatch_sanity.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("predispatch_sanity_test", PREDISPATCH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_anchors_file(tmp_path: Path, anchors: list[dict]) -> Path:
    """Create a test-only calibration anchors file."""
    cal_dir = tmp_path / "calibration"
    cal_dir.mkdir()
    out = cal_dir / "anchors_test_lane.json"
    out.write_text(json.dumps(anchors))
    return cal_dir


def test_apogee_int4_with_only_pr106_anchor_BLOCKS(tmp_path: Path) -> None:
    """The exact 2026-05-05 failure mode: only 1 anchor → predispatch blocks."""
    mod = _load_module()
    anchors_dir = _make_anchors_file(tmp_path, [
        {
            "lane_id": "lane_pr106_baseline",
            "rel_err_pct_per_weight": 0.0,
            "archive_bytes": 186239,
            "contest_cuda_score": 0.20945673,
            "avg_pose_dist": 3.4e-5,
            "avg_seg_dist": 0.00067819,
            "rate_unscaled": 0.00496015,
            "measured_utc": "2026-05-05T17:25Z",
            "job_id": "pr106-baseline",
            "archive_sha256": "0af839ab",
        },
    ])
    # Use any existing archive as a stand-in.
    archive = REPO_ROOT / "pyproject.toml"

    result = mod.predispatch_sanity(
        archive_path=archive,
        predicted_low=0.155,
        predicted_high=0.180,
        rel_err_pct=7.09,
        lane_class="test_lane",
        distortion_proxy_was_run=False,
        anchors_dir=anchors_dir,
    )
    assert not result.passed, "expected refusal with only 1 anchor"
    # At minimum the anchors_sufficient gate must fail.
    gate_names_failed = [g.name for g in result.gates if not g.passed]
    assert "anchors_sufficient" in gate_names_failed
    # And the distortion proxy gate too (rel_err 7.09% > 1% threshold).
    assert "distortion_proxy_local" in gate_names_failed


def test_lossy_better_than_lossless_BLOCKS(tmp_path: Path) -> None:
    """Sanity gate fires when predicted_high < lossless score."""
    mod = _load_module()
    anchors_dir = _make_anchors_file(tmp_path, [
        {
            "lane_id": "lane_pr106_baseline",
            "rel_err_pct_per_weight": 0.0,
            "archive_bytes": 186239,
            "contest_cuda_score": 0.20945673,
            "avg_pose_dist": 3.4e-5, "avg_seg_dist": 0.00067819, "rate_unscaled": 186239 / 37545489,
            "measured_utc": "2026-05-05T17:25Z", "job_id": "pr106", "archive_sha256": "ab",
        },
        {
            "lane_id": "lane_apogee_int8",
            "rel_err_pct_per_weight": 0.24,
            "archive_bytes": 187731, "contest_cuda_score": 0.21119,
            "avg_pose_dist": 3.38e-5, "avg_seg_dist": 0.000678, "rate_unscaled": 187731 / 37545489,
            "measured_utc": "2026-05-05T18:02Z", "job_id": "int8", "archive_sha256": "cd",
        },
        {
            "lane_id": "lane_apogee_int4",
            "rel_err_pct_per_weight": 7.09,
            "archive_bytes": 109996, "contest_cuda_score": 1.4287,
            "avg_pose_dist": 0.0237, "avg_seg_dist": 0.00868, "rate_unscaled": 0.00293,
            "measured_utc": "2026-05-05T17:40Z", "job_id": "int4", "archive_sha256": "ef",
        },
    ])
    archive = REPO_ROOT / "pyproject.toml"
    # Predicting [0.10, 0.15] with rel_err > 0 says "lossy beats lossless 0.20946" — incoherent.
    result = mod.predispatch_sanity(
        archive_path=archive,
        predicted_low=0.10,
        predicted_high=0.15,
        rel_err_pct=2.0,
        lane_class="test_lane",
        distortion_proxy_was_run=True,
        anchors_dir=anchors_dir,
    )
    sanity_gate = next(g for g in result.gates if g.name == "sanity_lossy_vs_lossless")
    assert not sanity_gate.passed, f"sanity gate should fire: {sanity_gate.detail}"
    assert "lossy" in sanity_gate.detail.lower()


def test_high_rel_err_without_proxy_BLOCKS(tmp_path: Path) -> None:
    """rel_err > 1% requires the distortion proxy to have been run."""
    mod = _load_module()
    anchors_dir = _make_anchors_file(tmp_path, [
        {"lane_id": f"a{i}", "rel_err_pct_per_weight": 0.0 + i * 0.5, "archive_bytes": 100000,
         "contest_cuda_score": 0.21 + i * 0.01, "avg_pose_dist": 1e-4, "avg_seg_dist": 1e-3,
         "rate_unscaled": 100000 / 37545489, "measured_utc": "2026-05-05T00:00Z", "job_id": f"j{i}", "archive_sha256": f"{i:02x}"}
        for i in range(3)
    ])
    archive = REPO_ROOT / "pyproject.toml"
    result = mod.predispatch_sanity(
        archive_path=archive,
        predicted_low=0.30,
        predicted_high=0.40,
        rel_err_pct=2.5,
        lane_class="test_lane",
        distortion_proxy_was_run=False,
        anchors_dir=anchors_dir,
    )
    proxy_gate = next(g for g in result.gates if g.name == "distortion_proxy_local")
    assert not proxy_gate.passed
    assert "proxy" in proxy_gate.detail.lower()


def test_low_rel_err_without_proxy_PASSES_proxy_gate(tmp_path: Path) -> None:
    """rel_err ≤ 1% does NOT require the proxy."""
    mod = _load_module()
    anchors_dir = _make_anchors_file(tmp_path, [
        {"lane_id": f"a{i}", "rel_err_pct_per_weight": i * 0.5, "archive_bytes": 100000,
         "contest_cuda_score": 0.21 + i * 0.01, "avg_pose_dist": 1e-4, "avg_seg_dist": 1e-3,
         "rate_unscaled": 100000 / 37545489, "measured_utc": "2026-05-05T00:00Z", "job_id": f"j{i}", "archive_sha256": f"{i:02x}"}
        for i in range(3)
    ])
    archive = REPO_ROOT / "pyproject.toml"
    result = mod.predispatch_sanity(
        archive_path=archive,
        predicted_low=0.21,
        predicted_high=0.22,
        rel_err_pct=0.24,  # int8 regime — under 1% threshold
        lane_class="test_lane",
        distortion_proxy_was_run=False,
        anchors_dir=anchors_dir,
    )
    proxy_gate = next(g for g in result.gates if g.name == "distortion_proxy_local")
    assert proxy_gate.passed, f"proxy gate should pass for low rel_err: {proxy_gate.detail}"


def test_missing_archive_BLOCKS(tmp_path: Path) -> None:
    """Missing archive blocks early — operator typo defense."""
    mod = _load_module()
    anchors_dir = _make_anchors_file(tmp_path, [{"lane_id": "x", "rel_err_pct_per_weight": 0.0, "archive_bytes": 100000,
        "contest_cuda_score": 0.2, "avg_pose_dist": 1e-4, "avg_seg_dist": 1e-3, "rate_unscaled": 100000 / 37545489,
        "measured_utc": "2026-05-05T00:00Z", "job_id": "j", "archive_sha256": "00"}] * 3)
    result = mod.predispatch_sanity(
        archive_path=tmp_path / "does_not_exist.zip",
        predicted_low=0.20, predicted_high=0.21, rel_err_pct=0.1,
        lane_class="test_lane", anchors_dir=anchors_dir,
    )
    assert not result.passed
    assert any("archive not found" in r for r in result.refusal_reasons)


def test_short_override_reason_REJECTED() -> None:
    """Override reason <40 chars is refused by main()."""
    mod = _load_module()
    archive = REPO_ROOT / "pyproject.toml"
    rc = mod.main([
        "--archive", str(archive),
        "--predicted-low", "0.155",
        "--predicted-high", "0.180",
        "--rel-err-pct", "7.09",
        "--lane-class", "apogee_intN",
        "--override-reason", "too short",
    ])
    # Either 64 (no override accepted) or all gates passed (rc 0). With a bogus
    # lane-class the anchors_sufficient gate fails, so we expect 64.
    assert rc == 64
