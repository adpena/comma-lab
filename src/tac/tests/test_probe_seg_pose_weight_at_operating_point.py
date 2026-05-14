# SPDX-License-Identifier: MIT
"""Tests for the council A-1 probe-disambiguator.

Verifies operating-point-aware seg/pose weight derivation against the
closed-form ``tac.score_geometry.score_gradient`` formula. The probe MUST:

- agree with the score-gradient at OLD 1.x and PR106 r2 anchors,
- flip the legacy seg-dominant ratio at PR106 r2 frontier,
- report the importance_flip_threshold consistent with score_geometry.

This is the council A-1 Option C deliverable (8/10 FOR).
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Load the probe module directly from tools/.
_PROBE_PATH = REPO_ROOT / "tools" / "probe_seg_pose_weight_at_operating_point.py"
_spec = importlib.util.spec_from_file_location(
    "probe_seg_pose_weight_at_operating_point", _PROBE_PATH
)
assert _spec is not None and _spec.loader is not None
probe = importlib.util.module_from_spec(_spec)
sys.modules["probe_seg_pose_weight_at_operating_point"] = probe
_spec.loader.exec_module(probe)


# ── Closed-form agreement ───────────────────────────────────────────────────


def test_probe_consumes_score_gradient_from_tac_score_geometry() -> None:
    """The probe MUST use the closed-form gradient module — no parallel impl."""
    from tac import score_geometry  # noqa: F401

    # The probe must call score_gradient; check import-level dependency.
    assert hasattr(probe, "score_gradient")
    assert probe.SEG_COEFFICIENT == score_geometry.SEG_COEFFICIENT
    assert (
        probe.POSE_COEFFICIENT_INSIDE_SQRT
        == score_geometry.POSE_COEFFICIENT_INSIDE_SQRT
    )


def test_probe_returns_seg_weight_equal_to_seg_coefficient() -> None:
    """w_seg is pinned to SEG_COEFFICIENT (=100.0)."""
    out = probe.compute_optimal_weights(d_pose=1e-4, d_seg=5e-4)
    assert out["seg_weight_optimal"] == 100.0


def test_probe_pose_weight_matches_score_gradient_at_pr106_r2() -> None:
    """At PR106 r2 the pose marginal is 5/sqrt(10*3.4e-5) ≈ 271.4."""
    out = probe.compute_optimal_weights(d_pose=3.4e-5, d_seg=6.7e-4)
    expected_pose_grad = 5.0 / math.sqrt(10.0 * 3.4e-5)
    # pose_weight = seg_weight * (pose_grad / seg_grad)
    expected_pose_weight = 100.0 * expected_pose_grad / 100.0
    assert math.isclose(
        out["pose_weight_optimal"], expected_pose_weight, rel_tol=1e-9
    )


def test_probe_ratio_flips_at_pr106_r2_vs_old_1x() -> None:
    """At PR106 r2 the ratio pose/seg EXCEEDS the legacy 0.1 default."""
    out_old = probe.compute_optimal_weights(d_pose=0.18, d_seg=0.001)
    out_pr106 = probe.compute_optimal_weights(d_pose=3.4e-5, d_seg=6.7e-4)
    assert out_old["ratio_pose_over_seg"] < 0.1  # old 1.x: pose < legacy
    assert out_pr106["ratio_pose_over_seg"] > 1.0  # PR106 r2: pose dominant
    assert (
        out_pr106["ratio_pose_over_seg"]
        > out_old["ratio_pose_over_seg"]
    )


def test_probe_ratio_at_threshold_equals_one() -> None:
    """At d_pose == importance_flip_threshold(), pose/seg ratio == 1.0."""
    from tac.score_geometry import importance_flip_threshold

    thresh = importance_flip_threshold()
    out = probe.compute_optimal_weights(d_pose=thresh, d_seg=1e-4)
    assert math.isclose(out["ratio_pose_over_seg"], 1.0, rel_tol=1e-9)


def test_probe_rejects_zero_d_pose() -> None:
    """d_pose == 0 is the singularity; the probe must refuse."""
    with pytest.raises(ValueError, match="d_pose"):
        probe.compute_optimal_weights(d_pose=0.0, d_seg=1e-3)


def test_probe_rejects_negative_d_seg() -> None:
    with pytest.raises(ValueError, match="d_seg"):
        probe.compute_optimal_weights(d_pose=1e-4, d_seg=-1.0)


# ── Report shape ────────────────────────────────────────────────────────────


def test_probe_report_contains_required_keys() -> None:
    report = probe.build_report(d_pose=3.4e-5, d_seg=6.7e-4)
    required = {
        "operating_point",
        "seg_weight_optimal",
        "pose_weight_optimal",
        "ratio_pose_over_seg",
        "marginal_d_seg",
        "marginal_d_pose",
        "basis",
        "legacy_ratio_pose_over_seg",
        "ratio_flip_threshold_d_pose",
        "score_coefficients",
        "wire_in",
        "evidence_tag",
    }
    assert required.issubset(report.keys())
    assert report["basis"] == (
        "closed-form gradient per tac.score_geometry.score_gradient"
    )
    assert report["legacy_ratio_pose_over_seg"] == 0.1


def test_probe_report_serializes_to_json() -> None:
    report = probe.build_report(d_pose=3.4e-5, d_seg=6.7e-4)
    txt = json.dumps(report)
    parsed = json.loads(txt)
    assert parsed["operating_point"]["d_pose"] == 3.4e-5


def test_probe_evidence_tag_links_score_geometry_lines() -> None:
    """The evidence tag must point at the closed-form derivation file."""
    report = probe.build_report(d_pose=1e-4, d_seg=5e-4)
    assert "score_geometry.py" in report["evidence_tag"]


# ── CLI smoke ───────────────────────────────────────────────────────────────


def test_probe_cli_named_operating_point(capsys: pytest.CaptureFixture[str]) -> None:
    rc = probe.main(["--operating-point", "pr106_r2"])
    assert rc == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["operating_point"]["d_pose"] == 3.4e-5
    assert parsed["ratio_pose_over_seg"] > 1.0


def test_probe_cli_explicit_d_pose_d_seg(capsys: pytest.CaptureFixture[str]) -> None:
    rc = probe.main(["--d-pose", "1e-4", "--d-seg", "5e-4"])
    assert rc == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["operating_point"]["d_pose"] == 1e-4
    assert parsed["operating_point"]["d_seg"] == 5e-4


def test_probe_cli_requires_d_seg_when_d_pose_supplied(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = probe.main(["--d-pose", "1e-4"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "--d-seg" in err


def test_probe_cli_writes_output_json(tmp_path: Path) -> None:
    out_path = tmp_path / "probe.json"
    rc = probe.main(
        [
            "--operating-point",
            "pr106_r2",
            "--output-json",
            str(out_path),
        ]
    )
    assert rc == 0
    assert out_path.exists()
    parsed = json.loads(out_path.read_text())
    assert parsed["operating_point"]["d_pose"] == 3.4e-5
