# SPDX-License-Identifier: MIT
"""Tests for tac.master_gradient_pr101_mps_axis_probe (codex op7 iteration #4).

Per CLAUDE.md "MPS auth eval is NOISE" + Catalog #192 + Catalog #317: the MPS
axis is non-promotable research-signal only; the dataclass invariants in
MpsScoreResponseRecord enforce this structurally.
"""
from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import pytest

from tac.master_gradient_pr101_mps_axis_probe import (
    AUTH_EVAL_SCHEMA,
    MPS_AXIS_TAG,
    MPS_HARDWARE_SUBSTRATE_PREFIX,
    MpsAxisProbeError,
    MpsScoreResponseRecord,
    build_cross_device_comparison_table,
    build_mps_axis_provenance,
    build_mps_research_signal_record,
    validate_cauchy_schwarz_bound_on_measured_gap,
)


def test_canonical_constants() -> None:
    assert MPS_AXIS_TAG == "[MPS-research-signal]"
    assert MPS_HARDWARE_SUBSTRATE_PREFIX == "macos_arm64"
    assert AUTH_EVAL_SCHEMA == "tac_mps_axis_score_response_probe_v1"


def test_mps_record_minimal_valid() -> None:
    rec = MpsScoreResponseRecord(
        archive_path="experiments/results/foo/archive.zip",
        archive_sha256="b" * 64,
        archive_bytes=178258,
        seg_term=0.056,
        pose_term=0.018,
        rate_term=0.119,
        score=0.193,
        n_samples=600,
        runtime_tree_sha256=None,
    )
    assert rec.measurement_axis == MPS_AXIS_TAG
    assert rec.score_claim_valid is False
    assert rec.promotion_eligible is False


def test_mps_record_rejects_short_sha() -> None:
    with pytest.raises(MpsAxisProbeError):
        MpsScoreResponseRecord(
            archive_path="x",
            archive_sha256="abc",
            archive_bytes=10,
            seg_term=0.0,
            pose_term=0.0,
            rate_term=0.0,
            score=0.0,
            n_samples=1,
            runtime_tree_sha256=None,
        )


def test_mps_record_rejects_zero_bytes() -> None:
    with pytest.raises(MpsAxisProbeError):
        MpsScoreResponseRecord(
            archive_path="x",
            archive_sha256="a" * 64,
            archive_bytes=0,
            seg_term=0.0,
            pose_term=0.0,
            rate_term=0.0,
            score=0.0,
            n_samples=1,
            runtime_tree_sha256=None,
        )


def test_mps_record_refuses_promotion_eligible_true() -> None:
    """Per CLAUDE.md MPS auth eval is NOISE: promotion_eligible MUST be False."""
    with pytest.raises(MpsAxisProbeError):
        MpsScoreResponseRecord(
            archive_path="x",
            archive_sha256="a" * 64,
            archive_bytes=10,
            seg_term=0.0,
            pose_term=0.0,
            rate_term=0.0,
            score=0.0,
            n_samples=1,
            runtime_tree_sha256=None,
            promotion_eligible=True,
        )


def test_mps_record_refuses_score_claim_valid_true() -> None:
    with pytest.raises(MpsAxisProbeError):
        MpsScoreResponseRecord(
            archive_path="x",
            archive_sha256="a" * 64,
            archive_bytes=10,
            seg_term=0.0,
            pose_term=0.0,
            rate_term=0.0,
            score=0.0,
            n_samples=1,
            runtime_tree_sha256=None,
            score_claim_valid=True,
        )


def test_mps_record_refuses_wrong_axis_tag() -> None:
    with pytest.raises(MpsAxisProbeError):
        MpsScoreResponseRecord(
            archive_path="x",
            archive_sha256="a" * 64,
            archive_bytes=10,
            seg_term=0.0,
            pose_term=0.0,
            rate_term=0.0,
            score=0.0,
            n_samples=1,
            runtime_tree_sha256=None,
            measurement_axis="[contest-CUDA]",
        )


def test_mps_record_as_dict_serializable() -> None:
    rec = MpsScoreResponseRecord(
        archive_path="x",
        archive_sha256="a" * 64,
        archive_bytes=10,
        seg_term=0.1,
        pose_term=0.2,
        rate_term=0.3,
        score=0.6,
        n_samples=600,
        runtime_tree_sha256="d" * 64,
    )
    d = rec.as_dict()
    import json as j
    s = j.dumps(d)
    assert "MPS-research-signal" in s
    assert d["score_claim_valid"] is False


def test_build_mps_research_signal_record_from_real_archive(tmp_path: Path) -> None:
    # Build a synthetic archive on disk
    archive = tmp_path / "synthetic.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("x", b"hello world")
    rec = build_mps_research_signal_record(
        archive,
        seg_term=0.05,
        pose_term=0.02,
        rate_term=0.01,
        score=0.08,
        n_samples=600,
    )
    assert rec.archive_bytes > 0
    # sha256 over the archive bytes
    expected_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert rec.archive_sha256 == expected_sha


def test_build_mps_record_missing_archive_raises(tmp_path: Path) -> None:
    with pytest.raises(MpsAxisProbeError):
        build_mps_research_signal_record(
            tmp_path / "nonexistent.zip",
            seg_term=0.0,
            pose_term=0.0,
            rate_term=0.0,
            score=0.0,
            n_samples=1,
        )


def test_build_mps_provenance_routes_canonical(tmp_path: Path) -> None:
    """MPS provenance MUST route through Catalog #323 canonical builder."""
    archive = tmp_path / "synthetic.zip"
    archive.write_bytes(b"abc")
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    prov = build_mps_axis_provenance(
        archive_sha256=sha, source_path=str(archive)
    )
    # The canonical Catalog #323 mps_proxy builder sets:
    assert prov.measurement_axis == "[MPS-PROXY]"
    assert prov.score_claim_valid is False
    assert prov.promotion_eligible is False


def test_cross_device_comparison_table_basic() -> None:
    baseline = {
        "contest_cpu": {"seg_term": 0.056, "pose_term": 0.018, "rate_term": 0.119, "score": 0.193, "n_samples": 600},
        "contest_cuda": {"seg_term": 0.056, "pose_term": 0.018, "rate_term": 0.119, "score": 0.193, "n_samples": 600},
        "mps_research_signal": {"seg_term": 0.057, "pose_term": 0.019, "rate_term": 0.119, "score": 0.195, "n_samples": 600},
    }
    candidate = {
        "contest_cpu": {"seg_term": 0.0576, "pose_term": 0.0182, "rate_term": 0.119, "score": 0.1945, "n_samples": 600},
        "contest_cuda": {"seg_term": 0.0574, "pose_term": 0.0179, "rate_term": 0.119, "score": 0.1943, "n_samples": 600},
        "mps_research_signal": {"seg_term": 0.0578, "pose_term": 0.0190, "rate_term": 0.119, "score": 0.1958, "n_samples": 600},
    }
    table = build_cross_device_comparison_table(
        baseline,
        candidate,
        baseline_archive_sha256="b" * 64,
        candidate_archive_sha256="c" * 64,
    )
    assert table["schema"] == AUTH_EVAL_SCHEMA
    assert table["baseline_archive_sha256"] == "b" * 64
    assert table["candidate_archive_sha256"] == "c" * 64
    # Per-axis comparison should have 3 entries
    assert len(table["per_axis"]) == 3
    # Cross-device drift should be computed
    assert "mps_vs_cuda_aggregate_fraction" in table["cross_device_drift_on_candidate"]
    assert "mps_vs_cpu_aggregate_fraction" in table["cross_device_drift_on_candidate"]
    # MPS-vs-CUDA absolute gap should be small
    gap = table["cross_device_drift_on_candidate"]["mps_vs_cuda_absolute_gap"]
    assert 0.0 <= gap < 0.01  # tight gap for tiny synthetic delta


def test_cross_device_comparison_evidence_discipline_present() -> None:
    table = build_cross_device_comparison_table(
        {"contest_cpu": {"score": 0.1, "seg_term": 0.0, "pose_term": 0.0, "rate_term": 0.0, "n_samples": 600}},
        {"contest_cpu": {"score": 0.1, "seg_term": 0.0, "pose_term": 0.0, "rate_term": 0.0, "n_samples": 600}},
        baseline_archive_sha256="a" * 64,
        candidate_archive_sha256="b" * 64,
    )
    ed = table["evidence_discipline"]
    assert ed["mps_axis_tag"] == MPS_AXIS_TAG
    assert ed["mps_score_claim_valid"] is False
    assert ed["mps_promotion_eligible"] is False
    assert "MPS auth eval is NOISE" in ed["claude_md_anchor"]


def test_validate_cauchy_schwarz_bound_satisfied() -> None:
    v = validate_cauchy_schwarz_bound_on_measured_gap(
        predicted_gap_upper_bound=0.10, measured_gap=0.05
    )
    assert v["bound_satisfied"] is True
    assert v["verdict"] == "BOUND_SATISFIED"
    assert v["headroom"] == pytest.approx(0.05)


def test_validate_cauchy_schwarz_bound_violated() -> None:
    v = validate_cauchy_schwarz_bound_on_measured_gap(
        predicted_gap_upper_bound=0.05, measured_gap=0.10
    )
    assert v["bound_satisfied"] is False
    assert v["verdict"] == "BOUND_VIOLATED"
    assert v["headroom"] == pytest.approx(-0.05)
    assert "exceeds" in v["action_recommendation"].lower()


def test_validate_bound_rejects_negative_predicted() -> None:
    with pytest.raises(MpsAxisProbeError):
        validate_cauchy_schwarz_bound_on_measured_gap(-0.1, 0.05)


def test_validate_bound_rejects_negative_measured() -> None:
    with pytest.raises(MpsAxisProbeError):
        validate_cauchy_schwarz_bound_on_measured_gap(0.1, -0.05)


def test_validate_bound_zero_zero_satisfies() -> None:
    v = validate_cauchy_schwarz_bound_on_measured_gap(0.0, 0.0)
    assert v["bound_satisfied"] is True


def test_module_exports_canonical_api() -> None:
    import tac.master_gradient_pr101_mps_axis_probe as mod

    expected = {
        "AUTH_EVAL_SCHEMA",
        "MPS_AXIS_TAG",
        "MPS_HARDWARE_SUBSTRATE_PREFIX",
        "MpsAxisProbeError",
        "MpsScoreResponseRecord",
        "build_cross_device_comparison_table",
        "build_mps_axis_provenance",
        "build_mps_research_signal_record",
        "validate_cauchy_schwarz_bound_on_measured_gap",
    }
    assert set(mod.__all__) == expected
