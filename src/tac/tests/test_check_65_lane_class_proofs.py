# SPDX-License-Identifier: MIT
"""Tests for Check 65 — lane classes must have a pipeline proof.

These tests use a tmp repo skeleton so we can mutate scripts/ + .omx/state/
without touching the real repo.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.preflight import (
    LANE_CLASS_PROOFS_REL,
    MetaBugViolation,
    _classify_lane_script,
    check_lane_classes_have_pipeline_proof,
)


@pytest.fixture
def tmp_repo(tmp_path: Path) -> Path:
    """Build a minimal repo skeleton with scripts/ and .omx/state/."""
    (tmp_path / "scripts").mkdir()
    (tmp_path / ".omx" / "state").mkdir(parents=True)
    return tmp_path


def _write_lane(repo: Path, name: str) -> Path:
    p = repo / "scripts" / f"remote_lane_{name}.sh"
    p.write_text("#!/usr/bin/env bash\n# stub lane script\n")
    return p


def _write_proofs(repo: Path, proofs: dict) -> None:
    (repo / LANE_CLASS_PROOFS_REL).write_text(json.dumps(proofs, indent=2))


def test_classify_recognizes_canonical_classes():
    assert _classify_lane_script(Path("remote_lane_a_pose_tto.sh")) == "pose-tto"
    assert _classify_lane_script(Path("remote_lane_d_halfframe.sh")) == "halfframe-mask"
    assert _classify_lane_script(Path("remote_lane_b_fp4_qat.sh")) == "fp4-qat"
    assert _classify_lane_script(Path("remote_lane_w_self_compress.sh")) == "self-compress"
    assert _classify_lane_script(Path("remote_lane_ea_entropy_archive.sh")) == "entropy-archive"


def test_classify_falls_back_to_uncategorized():
    assert _classify_lane_script(Path("remote_lane_zzz_random.sh")) == "uncategorized"


def test_no_lanes_no_violations(tmp_repo):
    """Empty scripts/ directory => no violations."""
    out = check_lane_classes_have_pipeline_proof(repo_root=tmp_repo, strict=False, verbose=False)
    assert out == []


def test_missing_proofs_file_flags_every_class(tmp_repo):
    """No proofs file => every detected class violates."""
    _write_lane(tmp_repo, "a_pose_tto")
    _write_lane(tmp_repo, "b_fp4_qat")
    out = check_lane_classes_have_pipeline_proof(repo_root=tmp_repo, strict=False, verbose=False)
    assert len(out) == 2
    classes_violating = {v.split("'")[1] for v in out}
    assert "pose-tto" in classes_violating
    assert "fp4-qat" in classes_violating


def test_proof_present_clears_violation(tmp_repo):
    _write_lane(tmp_repo, "a_pose_tto")
    _write_proofs(tmp_repo, {
        "pose-tto": {
            "proven_by_lane": "lane_a_pose_tto",
            "timestamp_utc": "2026-04-28T00:00:00Z",
            "score": 1.05,
        }
    })
    out = check_lane_classes_have_pipeline_proof(repo_root=tmp_repo, strict=False, verbose=False)
    assert out == []


def test_proof_missing_required_field_flags_violation(tmp_repo):
    _write_lane(tmp_repo, "a_pose_tto")
    _write_proofs(tmp_repo, {
        "pose-tto": {"proven_by_lane": "x"},  # missing timestamp_utc
    })
    out = check_lane_classes_have_pipeline_proof(repo_root=tmp_repo, strict=False, verbose=False)
    assert len(out) == 1
    assert "timestamp_utc" in out[0]


def test_proof_missing_proven_by_lane_flags_violation(tmp_repo):
    _write_lane(tmp_repo, "a_pose_tto")
    _write_proofs(tmp_repo, {
        "pose-tto": {"timestamp_utc": "2026-04-28T00:00:00Z"},  # missing proven_by_lane
    })
    out = check_lane_classes_have_pipeline_proof(repo_root=tmp_repo, strict=False, verbose=False)
    assert len(out) == 1
    assert "proven_by_lane" in out[0]


def test_strict_mode_raises_on_violations(tmp_repo):
    _write_lane(tmp_repo, "a_pose_tto")
    with pytest.raises(MetaBugViolation) as exc:
        check_lane_classes_have_pipeline_proof(
            repo_root=tmp_repo, strict=True, verbose=False,
        )
    assert "Check 65" in str(exc.value)
    assert "Lane RM-d" in str(exc.value)


def test_partial_coverage_only_flags_unproven_classes(tmp_repo):
    """If 1 of 2 classes is proven, only the unproven one should flag."""
    _write_lane(tmp_repo, "a_pose_tto")
    _write_lane(tmp_repo, "b_fp4_qat")
    _write_proofs(tmp_repo, {
        "pose-tto": {
            "proven_by_lane": "lane_a_pose_tto",
            "timestamp_utc": "2026-04-28T00:00:00Z",
        }
    })
    out = check_lane_classes_have_pipeline_proof(repo_root=tmp_repo, strict=False, verbose=False)
    assert len(out) == 1
    assert "fp4-qat" in out[0]
    assert "pose-tto" not in out[0]


def test_corrupt_proofs_file_treated_as_empty(tmp_repo):
    """Malformed JSON should not crash the check; treated as zero proofs."""
    _write_lane(tmp_repo, "a_pose_tto")
    (tmp_repo / LANE_CLASS_PROOFS_REL).write_text("{not valid json")
    out = check_lane_classes_have_pipeline_proof(repo_root=tmp_repo, strict=False, verbose=False)
    assert len(out) == 1


def test_check_runs_against_real_repo_without_crash():
    """Smoke: the check executes against the real repo without raising
    (warn-only mode)."""
    out = check_lane_classes_have_pipeline_proof(strict=False, verbose=False)
    # We don't assert a specific count — just that it returns a list.
    assert isinstance(out, list)
