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
    # And the distortion model gate too (rel_err 7.09% > 1% threshold).
    assert "distortion_model_gate" in gate_names_failed


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


def test_lossy_smaller_archive_with_parity_evidence_allows_rate_term_gain(tmp_path: Path) -> None:
    """Smaller lossy archives may beat lossless by the official rate term."""
    mod = _load_module()
    anchors_dir = _make_anchors_file(tmp_path, [
        {
            "lane_id": "lane_pr106_baseline",
            "rel_err_pct_per_weight": 0.0,
            "archive_bytes": 186_239,
            "contest_cuda_score": 0.20945673,
            "avg_pose_dist": 3.4e-5,
            "avg_seg_dist": 0.00067819,
            "rate_unscaled": 186_239 / 37_545_489,
            "measured_utc": "2026-05-05T17:25Z",
            "job_id": "pr106",
            "archive_sha256": "ab",
        },
        {
            "lane_id": "lane_apogee_int8",
            "rel_err_pct_per_weight": 0.24,
            "archive_bytes": 187_731,
            "contest_cuda_score": 0.21119,
            "avg_pose_dist": 3.38e-5,
            "avg_seg_dist": 0.000678,
            "rate_unscaled": 187_731 / 37_545_489,
            "measured_utc": "2026-05-05T18:02Z",
            "job_id": "int8",
            "archive_sha256": "cd",
        },
        {
            "lane_id": "lane_apogee_int4",
            "rel_err_pct_per_weight": 7.09,
            "archive_bytes": 109_996,
            "contest_cuda_score": 1.4287,
            "avg_pose_dist": 0.0237,
            "avg_seg_dist": 0.00868,
            "rate_unscaled": 109_996 / 37_545_489,
            "measured_utc": "2026-05-05T17:40Z",
            "job_id": "int4",
            "archive_sha256": "ef",
        },
    ])
    archive = tmp_path / "apogee_int6_archive.zip"
    archive.write_bytes(b"x" * 170_450)
    evidence = tmp_path / "parity.json"
    evidence.write_text(
        json.dumps(
            {
                "candidate_archive_sha256": mod.sha256_file(archive),
                "evidence_semantics": "scorer_basin_parity_gate",
                "parity_report": {
                    "pose_dist_delta": 0.0,
                    "pose_dist_lossless": 1.0e-4,
                    "pose_dist_quantized": 1.0e-4,
                    "seg_dist_delta": 0.0,
                },
                "ready_for_exact_eval_dispatch": True,
                "evidence_grade": "empirical",
                "scorer_basin_parity_status": "passed",
            }
        ),
        encoding="utf-8",
    )

    result = mod.predispatch_sanity(
        archive_path=archive,
        predicted_low=0.199,
        predicted_high=0.204,
        rel_err_pct=2.0,
        lane_class="test_lane",
        distortion_proxy_was_run=True,
        anchors_dir=anchors_dir,
        readiness_evidence_json=evidence,
    )

    sanity_gate = next(g for g in result.gates if g.name == "sanity_lossy_vs_lossless")
    assert sanity_gate.passed, sanity_gate.detail
    assert "official rate-distortion floor" in sanity_gate.detail


def test_lossy_smaller_archive_blocks_when_parity_deltas_overwhelm_rate_gain(tmp_path: Path) -> None:
    """Parity readiness is not score-lowering evidence when deltas are positive."""
    mod = _load_module()
    anchors_dir = _make_anchors_file(tmp_path, [
        {
            "lane_id": "lane_pr106_baseline",
            "rel_err_pct_per_weight": 0.0,
            "archive_bytes": 186_239,
            "contest_cuda_score": 0.20945673,
            "avg_pose_dist": 3.4e-5,
            "avg_seg_dist": 0.00067819,
            "rate_unscaled": 186_239 / 37_545_489,
            "measured_utc": "2026-05-05T17:25Z",
            "job_id": "pr106",
            "archive_sha256": "ab",
        },
        {
            "lane_id": "lane_apogee_int8",
            "rel_err_pct_per_weight": 0.24,
            "archive_bytes": 187_731,
            "contest_cuda_score": 0.21119,
            "avg_pose_dist": 3.38e-5,
            "avg_seg_dist": 0.000678,
            "rate_unscaled": 187_731 / 37_545_489,
            "measured_utc": "2026-05-05T18:02Z",
            "job_id": "int8",
            "archive_sha256": "cd",
        },
        {
            "lane_id": "lane_apogee_int4",
            "rel_err_pct_per_weight": 7.09,
            "archive_bytes": 109_996,
            "contest_cuda_score": 1.4287,
            "avg_pose_dist": 0.0237,
            "avg_seg_dist": 0.00868,
            "rate_unscaled": 109_996 / 37_545_489,
            "measured_utc": "2026-05-05T17:40Z",
            "job_id": "int4",
            "archive_sha256": "ef",
        },
    ])
    archive = tmp_path / "apogee_int6_archive.zip"
    archive.write_bytes(b"x" * 170_450)
    evidence = tmp_path / "parity.json"
    evidence.write_text(
        json.dumps(
            {
                "candidate_archive_sha256": mod.sha256_file(archive),
                "evidence_semantics": "scorer_basin_parity_gate",
                "parity_report": {
                    "pose_dist_lossless": 0.0001678224216448143,
                    "pose_dist_quantized": 0.00027574982959777117,
                    "seg_dist_delta": 0.0009618123876862228,
                },
                "ready_for_exact_eval_dispatch": True,
                "evidence_grade": "empirical",
                "scorer_basin_parity_status": "passed",
            }
        ),
        encoding="utf-8",
    )

    result = mod.predispatch_sanity(
        archive_path=archive,
        predicted_low=0.199,
        predicted_high=0.204,
        rel_err_pct=2.0,
        lane_class="test_lane",
        distortion_proxy_was_run=True,
        anchors_dir=anchors_dir,
        readiness_evidence_json=evidence,
    )

    sanity_gate = next(g for g in result.gates if g.name == "sanity_lossy_vs_lossless")
    assert not sanity_gate.passed
    assert "rate-distortion floor" in sanity_gate.detail
    assert "not score-lowering evidence" in sanity_gate.detail


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
    proxy_gate = next(g for g in result.gates if g.name == "distortion_model_gate")
    assert not proxy_gate.passed
    assert "proxy" in proxy_gate.detail.lower()


def test_high_rel_err_with_proxy_without_generic_readiness_evidence_BLOCKS() -> None:
    """A local proxy trace is telemetry, not generic dispatch authorization."""
    mod = _load_module()
    archive = REPO_ROOT / "pyproject.toml"

    gate = mod._gate_distortion_proxy(
        rel_err_pct=2.5,
        distortion_proxy_was_run=True,
        archive_path=archive,
        evidence_json_path=None,
    )

    assert not gate.passed
    assert gate.name == "distortion_model_gate"
    assert "non-promotable" in gate.detail
    assert "readiness-evidence-json" in gate.detail


def test_high_rel_err_with_exact_byte_generic_readiness_evidence_passes(tmp_path: Path) -> None:
    """Generic high-rel-err dispatch requires exact SHA-tied non-proxy evidence."""
    mod = _load_module()
    archive = REPO_ROOT / "pyproject.toml"
    evidence = tmp_path / "parity.json"
    evidence.write_text(
        json.dumps(
            {
                "candidate_archive_sha256": mod.sha256_file(archive),
                "evidence_semantics": "scorer_basin_parity_gate",
                "ready_for_exact_eval_dispatch": True,
                "evidence_grade": "A",
                "scorer_basin_parity_status": "passed",
            }
        ),
        encoding="utf-8",
    )

    gate = mod._gate_distortion_proxy(
        rel_err_pct=2.5,
        distortion_proxy_was_run=True,
        archive_path=archive,
        evidence_json_path=evidence,
    )

    assert gate.passed
    assert gate.name == "distortion_model_gate"
    assert "scorer_basin_parity_gate" in gate.detail


def test_apogee_proxy_ran_still_requires_non_proxy_readiness_evidence(tmp_path: Path) -> None:
    """Running the local proxy is not enough to authorize Apogee GPU dispatch."""
    mod = _load_module()
    anchors_dir = _make_anchors_file(tmp_path, [
        {
            "lane_id": "lane_pr106_baseline",
            "rel_err_pct_per_weight": 0.0,
            "archive_bytes": 186239,
            "contest_cuda_score": 0.20945673,
            "avg_pose_dist": 3.4e-5,
            "avg_seg_dist": 0.00067819,
            "rate_unscaled": 186239 / 37545489,
            "measured_utc": "2026-05-05T17:25Z",
            "job_id": "pr106",
            "archive_sha256": "ab",
        },
        {
            "lane_id": "lane_apogee_int8",
            "rel_err_pct_per_weight": 0.24,
            "archive_bytes": 187731,
            "contest_cuda_score": 0.21119,
            "avg_pose_dist": 3.38e-5,
            "avg_seg_dist": 0.000678,
            "rate_unscaled": 187731 / 37545489,
            "measured_utc": "2026-05-05T18:02Z",
            "job_id": "int8",
            "archive_sha256": "cd",
        },
        {
            "lane_id": "lane_apogee_int4",
            "rel_err_pct_per_weight": 7.09,
            "archive_bytes": 109996,
            "contest_cuda_score": 1.4287,
            "avg_pose_dist": 0.0237,
            "avg_seg_dist": 0.00868,
            "rate_unscaled": 109996 / 37545489,
            "measured_utc": "2026-05-05T17:40Z",
            "job_id": "int4",
            "archive_sha256": "ef",
        },
    ])
    archive = REPO_ROOT / "pyproject.toml"

    result = mod.predispatch_sanity(
        archive_path=archive,
        predicted_low=0.211,
        predicted_high=0.220,
        rel_err_pct=0.24,
        lane_class="apogee_intN",
        distortion_proxy_was_run=True,
        anchors_dir=anchors_dir,
    )

    evidence_gate = next(g for g in result.gates if g.name == "apogee_evidence_semantics")
    assert not evidence_gate.passed
    assert "cannot dispatch from byte-only" in evidence_gate.detail


def test_apogee_readiness_evidence_rejects_local_proxy_json(tmp_path: Path) -> None:
    mod = _load_module()
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"apogee bytes")
    evidence = tmp_path / "proxy.json"
    evidence.write_text(
        json.dumps(
            {
                "candidate_archive_sha256": mod.sha256_file(archive),
                "evidence_semantics": "local_distortion_proxy",
                "ready_for_exact_eval_dispatch": True,
                "distortion_model_status": "passed",
            }
        )
    )

    gate = mod._gate_apogee_evidence_semantics(
        lane_class="apogee_intN",
        archive_path=archive,
        evidence_json_path=evidence,
    )
    assert not gate.passed
    assert "unsupported evidence_semantics" in gate.detail


def test_apogee_readiness_evidence_accepts_parity_gate_for_exact_sha(tmp_path: Path) -> None:
    mod = _load_module()
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"apogee bytes")
    evidence = tmp_path / "parity.json"
    evidence.write_text(
        json.dumps(
            {
                "candidate_archive_sha256": mod.sha256_file(archive),
                "evidence_semantics": "scorer_basin_parity_gate",
                "ready_for_exact_eval_dispatch": True,
                "scorer_basin_parity_status": "passed",
            }
        )
    )

    gate = mod._gate_apogee_evidence_semantics(
        lane_class="apogee_intN",
        archive_path=archive,
        evidence_json_path=evidence,
    )
    assert gate.passed


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
    proxy_gate = next(g for g in result.gates if g.name == "distortion_model_gate")
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
