"""Tests for tools/xray_cpu_cuda_drift_per_arch_class.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import xray_cpu_cuda_drift_per_arch_class as xd  # noqa: E402


# ── medal_band_verdict logic ─────────────────────────────────────────────


def test_inside_when_point_below_floor_and_band_tight():
    v = xd.medal_band_verdict(
        predicted_cpu_score=0.190,
        predicted_cpu_score_high=0.193,
        predicted_cpu_score_low=0.187,
        medal_floor=0.19538,
        medal_tolerance=0.005,
    )
    assert v["verdict"] == "INSIDE"


def test_inside_with_uncertainty_when_band_high_above_borderline():
    v = xd.medal_band_verdict(
        predicted_cpu_score=0.194,
        predicted_cpu_score_high=0.210,
        predicted_cpu_score_low=0.190,
        medal_floor=0.19538,
        medal_tolerance=0.005,
    )
    assert v["verdict"] == "INSIDE_with_uncertainty"


def test_borderline_when_point_in_tolerance_band():
    v = xd.medal_band_verdict(
        predicted_cpu_score=0.198,
        predicted_cpu_score_high=0.200,
        predicted_cpu_score_low=0.196,
        medal_floor=0.19538,
        medal_tolerance=0.005,
    )
    assert v["verdict"] == "BORDERLINE"


def test_uncertain_when_band_low_within_borderline():
    v = xd.medal_band_verdict(
        predicted_cpu_score=0.220,
        predicted_cpu_score_high=0.250,
        predicted_cpu_score_low=0.198,
        medal_floor=0.19538,
        medal_tolerance=0.005,
    )
    assert v["verdict"] == "UNCERTAIN"


def test_outside_when_point_and_band_above_borderline():
    v = xd.medal_band_verdict(
        predicted_cpu_score=0.230,
        predicted_cpu_score_high=0.235,
        predicted_cpu_score_low=0.225,
        medal_floor=0.19538,
        medal_tolerance=0.005,
    )
    assert v["verdict"] == "OUTSIDE"


# ── actionable_next_step ─────────────────────────────────────────────────


def test_next_step_inside_recommends_dispatch():
    s = xd.actionable_next_step("INSIDE", n_anchors=5)
    assert "DISPATCH" in s


def test_next_step_outside_recommends_drop():
    s = xd.actionable_next_step("OUTSIDE", n_anchors=5)
    assert "DROP" in s


def test_next_step_uncertain_low_anchors_recommends_hold():
    s = xd.actionable_next_step("UNCERTAIN", n_anchors=0)
    assert "HOLD" in s


def test_next_step_uncertain_calibrated_recommends_dispatch():
    s = xd.actionable_next_step("UNCERTAIN", n_anchors=5)
    assert "DISPATCH" in s


def test_next_step_borderline_recommends_dispatch_with_caveat():
    s = xd.actionable_next_step("BORDERLINE", n_anchors=5)
    assert "DISPATCH" in s
    assert "borderline" in s.lower() or "caveat" in s.lower()


def test_next_step_unknown_verdict_returns_review():
    s = xd.actionable_next_step("XYZ", n_anchors=5)
    assert "REVIEW" in s


# ── predict_cpu_band ─────────────────────────────────────────────────────


def test_predict_cpu_band_with_metadata_only():
    pred = xd.predict_cpu_band(
        archive_path=None,
        cuda_score=0.22933,
        metadata={"architecture_class": "hnerv_ft_microcodec"},
    )
    assert pred["architecture_class"] == "hnerv_ft_microcodec"
    assert "predicted_cpu_score" in pred
    # HNeRV cluster gap is ~0.033 → predicted CPU should be ~0.196
    assert 0.18 < pred["predicted_cpu_score"] < 0.21


def test_predict_cpu_band_unknown_class_falls_back():
    pred = xd.predict_cpu_band(
        archive_path=None,
        cuda_score=0.300,
        metadata={"architecture_class": "DEFINITELY_NOT_A_REAL_CLASS"},
    )
    # Bad class string falls through to unknown_uncalibrated
    assert pred["architecture_class"] == "unknown_uncalibrated"


def test_predict_cpu_band_records_n_anchors():
    pred = xd.predict_cpu_band(
        archive_path=None,
        cuda_score=0.229,
        metadata={"architecture_class": "hnerv_ft_microcodec"},
    )
    # HNeRV cluster bootstrapped from PR100/101/102/103/105 → 5 anchors
    assert pred["n_anchors"] == 5


# ── main / CLI ───────────────────────────────────────────────────────────


def test_main_no_archive_no_metadata_returns_2(tmp_path):
    rc = xd.main([
        "--cuda-score", "0.229",
        "--manual-cuda-score-justification", "unit-test diagnostic no-input guard",
        "--output-dir", str(tmp_path / "out"),
    ])
    assert rc == 2


def test_main_metadata_path_writes_outputs(tmp_path):
    md = tmp_path / "md.json"
    md.write_text(json.dumps({"architecture_class": "hnerv_ft_microcodec"}))
    out = tmp_path / "out"
    rc = xd.main([
        "--metadata-json", str(md),
        "--cuda-score", "0.22933",
        "--manual-cuda-score-justification", "unit-test diagnostic fixture",
        "--label", "pr107_apogee",
        "--output-dir", str(out),
    ])
    assert rc == 0
    assert (out / "drift_prediction.json").exists()
    assert (out / "drift_prediction.md").exists()
    rep = json.loads((out / "drift_prediction.json").read_text())
    assert rep["score_claim"] is False
    assert rep["label"] == "pr107_apogee"


def test_main_records_verdict_and_next_step(tmp_path):
    md = tmp_path / "md.json"
    md.write_text(json.dumps({"architecture_class": "hnerv_ft_microcodec"}))
    out = tmp_path / "out"
    xd.main([
        "--metadata-json", str(md),
        "--cuda-score", "0.22933",
        "--manual-cuda-score-justification", "unit-test diagnostic fixture",
        "--output-dir", str(out),
    ])
    rep = json.loads((out / "drift_prediction.json").read_text())
    assert rep["medal_verdict"]["verdict"] in {
        "INSIDE", "INSIDE_with_uncertainty", "BORDERLINE", "UNCERTAIN", "OUTSIDE"
    }
    assert isinstance(rep["recommended_next_step"], str)
    assert len(rep["recommended_next_step"]) > 10


def test_markdown_includes_diagnostic_tag(tmp_path):
    md = tmp_path / "md.json"
    md.write_text(json.dumps({"architecture_class": "hnerv_ft_microcodec"}))
    out = tmp_path / "out"
    xd.main([
        "--metadata-json", str(md),
        "--cuda-score", "0.229",
        "--manual-cuda-score-justification", "unit-test diagnostic fixture",
        "--output-dir", str(out),
    ])
    md_text = (out / "drift_prediction.md").read_text()
    assert "[diagnostic: cpu-vs-cuda drift prediction]" in md_text
    assert "predicted; learning-layer registry posterior" in md_text


def test_main_invalid_metadata_returns_2(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("not json {")
    rc = xd.main([
        "--metadata-json", str(bad),
        "--cuda-score", "0.229",
        "--manual-cuda-score-justification", "unit-test diagnostic fixture",
        "--output-dir", str(tmp_path / "out"),
    ])
    assert rc == 2


def test_main_default_medal_floor_matches_pr102_silver(tmp_path):
    md = tmp_path / "md.json"
    md.write_text(json.dumps({"architecture_class": "hnerv_ft_microcodec"}))
    out = tmp_path / "out"
    xd.main([
        "--metadata-json", str(md),
        "--cuda-score", "0.229",
        "--manual-cuda-score-justification", "unit-test diagnostic fixture",
        "--output-dir", str(out),
    ])
    rep = json.loads((out / "drift_prediction.json").read_text())
    # PR102 silver score
    assert abs(rep["medal_floor"] - 0.19538) < 1e-6


def test_main_refuses_unjustified_manual_cuda_score(tmp_path):
    md = tmp_path / "md.json"
    md.write_text(json.dumps({"architecture_class": "hnerv_ft_microcodec"}))
    rc = xd.main([
        "--metadata-json", str(md),
        "--cuda-score", "0.229",
        "--output-dir", str(tmp_path / "out"),
    ])
    assert rc == 2


def test_main_accepts_cuda_auth_eval_json_score_source(tmp_path):
    md = tmp_path / "md.json"
    md.write_text(json.dumps({"architecture_class": "hnerv_ft_microcodec"}))
    artifact = tmp_path / "contest_auth_eval.json"
    artifact.write_text(json.dumps({
        "score_recomputed_from_components": 0.22933,
        "avg_posenet_dist": 3.4e-5,
        "avg_segnet_dist": 6.7e-4,
        "rate_unscaled": 0.005,
        "archive_size_bytes": 185578,
        "n_samples": 600,
        "score_axis": "contest_cuda",
        "evidence_grade": "contest-CUDA",
        "score_claim_valid": True,
        "promotion_eligible": False,
        "provenance": {
            "device": "cuda",
            "gpu_t4_match": True,
            "archive_sha256": "a" * 64,
        },
    }))
    out = tmp_path / "out"
    rc = xd.main([
        "--metadata-json", str(md),
        "--cuda-auth-eval-json", str(artifact),
        "--output-dir", str(out),
    ])
    assert rc == 0
    rep = json.loads((out / "drift_prediction.json").read_text())
    assert rep["cuda_score"] == 0.22933
    assert rep["cuda_score_source"] == "contest_cuda_auth_eval_json"
    assert rep["cuda_auth_eval_json"] == str(artifact)


def test_main_refuses_wrong_shape_metadata_that_resolves_unknown(tmp_path):
    md = tmp_path / "analysis.json"
    md.write_text(json.dumps({"schema": "device_axis_eval_matrix_analysis.v1", "entries": []}))
    rc = xd.main([
        "--metadata-json", str(md),
        "--cuda-score", "0.229",
        "--manual-cuda-score-justification", "unit-test diagnostic fixture",
        "--output-dir", str(tmp_path / "out"),
    ])
    assert rc == 2


def test_main_can_explicitly_allow_unknown_architecture_class(tmp_path):
    md = tmp_path / "analysis.json"
    md.write_text(json.dumps({"schema": "device_axis_eval_matrix_analysis.v1", "entries": []}))
    out = tmp_path / "out"
    rc = xd.main([
        "--metadata-json", str(md),
        "--cuda-score", "0.229",
        "--manual-cuda-score-justification", "unit-test diagnostic fixture",
        "--allow-unknown-architecture-class",
        "--output-dir", str(out),
    ])
    assert rc == 0
    rep = json.loads((out / "drift_prediction.json").read_text())
    assert rep["prediction"]["architecture_class"] == "unknown_uncalibrated"


def test_default_constants_match_dossier():
    assert abs(xd.DEFAULT_MEDAL_FLOOR - 0.19538) < 1e-6
    assert abs(xd.DEFAULT_MEDAL_TOLERANCE - 0.005) < 1e-6


def test_inside_verdict_when_exactly_at_floor():
    v = xd.medal_band_verdict(
        predicted_cpu_score=0.19538,
        predicted_cpu_score_high=0.196,
        predicted_cpu_score_low=0.194,
        medal_floor=0.19538,
        medal_tolerance=0.005,
    )
    assert v["verdict"] == "INSIDE"


# ── Adversarial-input sanity (R4 fix) ─────────────────────────────────────


@pytest.mark.parametrize("cuda_score", [0.0, 0.001, 0.5, 1.0, 5.0])
def test_predict_cpu_band_handles_wide_score_range(cuda_score):
    pred = xd.predict_cpu_band(
        archive_path=None,
        cuda_score=cuda_score,
        metadata={"architecture_class": "hnerv_ft_microcodec"},
    )
    # Predicted CPU score should be a finite float
    import math
    assert math.isfinite(pred["predicted_cpu_score"])
    assert math.isfinite(pred["predicted_cpu_score_low"])
    assert math.isfinite(pred["predicted_cpu_score_high"])


def test_medal_verdict_returns_label_for_negative_score():
    # Negative scores can occur in a buggy registry update; verdict should
    # still produce a valid label (INSIDE since point < floor)
    v = xd.medal_band_verdict(
        predicted_cpu_score=-1.0,
        predicted_cpu_score_high=-0.5,
        predicted_cpu_score_low=-1.5,
        medal_floor=0.19538,
        medal_tolerance=0.005,
    )
    assert v["verdict"] in {
        "INSIDE", "INSIDE_with_uncertainty", "BORDERLINE", "UNCERTAIN", "OUTSIDE"
    }


def test_medal_verdict_returns_label_for_very_large_score():
    v = xd.medal_band_verdict(
        predicted_cpu_score=10.0,
        predicted_cpu_score_high=10.5,
        predicted_cpu_score_low=9.5,
        medal_floor=0.19538,
        medal_tolerance=0.005,
    )
    assert v["verdict"] == "OUTSIDE"
