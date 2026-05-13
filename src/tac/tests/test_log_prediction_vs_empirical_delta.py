"""Tests for :mod:`tools.log_prediction_vs_empirical_delta`.

Coverage:
- predicted-anchors JSONL parsing (well-formed + skip-foreign-schema + missing-file)
- continual_learning_posterior JSON parsing (schema-validated + missing-file)
- lane_id ↔ architecture_class substring + token-overlap matching
- delta + verdict computation (within / over / under)
- calibration recommendations (tighten / widen / leave_as_is)
- write_outputs: JSON + CSV emit + (optional) plot fallback when matplotlib missing
- end-to-end main() CLI smoke

Lane: ``lane_other_priorities_parallel_sweep_20260513``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Import via the canonical script path.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import log_prediction_vs_empirical_delta as delta_logger  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_predictions(path: Path, rows: list[dict]) -> None:
    lines = [json.dumps(r) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_posterior(path: Path, history: list[dict]) -> None:
    doc = {
        "schema": "tac_continual_learning_posterior_v1",
        "schema_version": 1,
        "accepted_anchor_count": len(history),
        "refused_anchor_count": 0,
        "accepted_anchors": [],
        "accepted_anchor_history": history,
        "evidence_grade": "[empirical]",
        "last_updated_utc": "2026-05-13T00:00:00Z",
        "source_rho_posteriors": {},
        "track_correction_posteriors": {},
    }
    path.write_text(json.dumps(doc), encoding="utf-8")


def _pred_row(
    lane_id: str = "lane_test_substrate_20260513",
    low: float = 0.180,
    high: float = 0.200,
    source: str = "[prediction; test]",
) -> dict:
    return {
        "schema": "tac_predicted_anchor_v1",
        "schema_version": 1,
        "anchor_id": f"prediction_{lane_id}",
        "lane_id": lane_id,
        "predicted_score_band_low": low,
        "predicted_score_band_high": high,
        "predicted_archive_bytes": 100000,
        "predicted_seg_distortion_avg": 6.5e-4,
        "predicted_pose_distortion_avg": 3.0e-5,
        "evidence_grade": "[prediction]",
        "source": source,
        "logged_at_utc": "2026-05-13T00:00:00Z",
    }


def _emp_row(
    architecture_class: str = "test_substrate",
    score: float = 0.193,
    axis: str = "cuda",
    observed: str = "2026-05-13T10:00:00+00:00",
) -> dict:
    return {
        "axis": axis,
        "architecture_class": architecture_class,
        "evidence_tag": f"[contest-{axis.upper()}]",
        "archive_sha256": "deadbeef" * 8,
        "archive_bytes": 100000,
        "score_value": score,
        "hardware_substrate": "linux_x86_64_t4",
        "observed_at_utc": observed,
        "track_updates": [],
        "source_rho_estimate": None,
    }


# ---------------------------------------------------------------------------
# load_predictions
# ---------------------------------------------------------------------------


def test_load_predictions_well_formed(tmp_path: Path) -> None:
    p = tmp_path / "preds.jsonl"
    _write_predictions(p, [_pred_row(lane_id="lane_a1_20260513")])
    preds = delta_logger.load_predictions(p)
    assert len(preds) == 1
    assert preds[0].lane_id == "lane_a1_20260513"
    assert preds[0].predicted_midpoint == pytest.approx(0.190)


def test_load_predictions_skips_foreign_schema(tmp_path: Path) -> None:
    p = tmp_path / "preds.jsonl"
    rows = [
        _pred_row(lane_id="lane_real_substrate_20260513"),
        {"schema": "some_other_schema_v1", "lane_id": "lane_ignored_20260513"},
    ]
    _write_predictions(p, rows)
    preds = delta_logger.load_predictions(p)
    assert len(preds) == 1
    assert preds[0].lane_id == "lane_real_substrate_20260513"


def test_load_predictions_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="predicted-anchors"):
        delta_logger.load_predictions(tmp_path / "nonexistent.jsonl")


def test_load_predictions_skips_blank_and_comment_lines(tmp_path: Path) -> None:
    p = tmp_path / "preds.jsonl"
    body = "\n".join(
        [
            "# leading comment",
            "",
            json.dumps(_pred_row(lane_id="lane_keep_20260513")),
            "",
        ]
    )
    p.write_text(body, encoding="utf-8")
    preds = delta_logger.load_predictions(p)
    assert len(preds) == 1


# ---------------------------------------------------------------------------
# load_empirical
# ---------------------------------------------------------------------------


def test_load_empirical_well_formed(tmp_path: Path) -> None:
    p = tmp_path / "post.json"
    _write_posterior(p, [_emp_row(architecture_class="hnerv_test", score=0.190)])
    emp = delta_logger.load_empirical(p)
    assert len(emp) == 1
    assert emp[0].architecture_class == "hnerv_test"
    assert emp[0].score_value == pytest.approx(0.190)


def test_load_empirical_schema_mismatch_raises(tmp_path: Path) -> None:
    p = tmp_path / "post.json"
    p.write_text(json.dumps({"schema": "wrong_schema"}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema mismatch"):
        delta_logger.load_empirical(p)


def test_load_empirical_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="continual_learning_posterior"):
        delta_logger.load_empirical(tmp_path / "nonexistent.json")


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def test_match_substring_in_lane_id() -> None:
    pred = delta_logger.PredictionRecord(
        lane_id="lane_a1_plus_lapose_composition_20260513",
        predicted_score_band_low=0.187,
        predicted_score_band_high=0.192,
        predicted_archive_bytes=4096,
        predicted_seg_distortion_avg=6.7e-4,
        predicted_pose_distortion_avg=3.0e-5,
        source="",
    )
    emp = delta_logger.EmpiricalRecord(
        architecture_class="a1_host",
        axis="cuda",
        evidence_tag="[contest-CUDA]",
        archive_sha256="x" * 64,
        archive_bytes=178262,
        score_value=0.193,
        hardware_substrate="linux_x86_64_t4",
        observed_at_utc="2026-05-13T10:00:00+00:00",
    )
    matched = delta_logger.match_prediction_to_empirical(pred, [emp])
    assert matched is not None
    assert matched.architecture_class == "a1_host"


def test_match_no_overlap_returns_none() -> None:
    pred = delta_logger.PredictionRecord(
        lane_id="lane_completely_distinct_20260513",
        predicted_score_band_low=0.1,
        predicted_score_band_high=0.2,
        predicted_archive_bytes=None,
        predicted_seg_distortion_avg=None,
        predicted_pose_distortion_avg=None,
        source="",
    )
    emp = delta_logger.EmpiricalRecord(
        architecture_class="hnerv_ft_microcodec",
        axis="cpu",
        evidence_tag="[contest-CPU]",
        archive_sha256="y" * 64,
        archive_bytes=100000,
        score_value=0.195,
        hardware_substrate="linux_x86_64_gha_cpu",
        observed_at_utc="2026-05-13T10:00:00+00:00",
    )
    assert delta_logger.match_prediction_to_empirical(pred, [emp]) is None


def test_match_multiple_candidates_picks_most_recent() -> None:
    pred = delta_logger.PredictionRecord(
        lane_id="lane_hnerv_test_substrate_20260513",
        predicted_score_band_low=0.18,
        predicted_score_band_high=0.20,
        predicted_archive_bytes=None,
        predicted_seg_distortion_avg=None,
        predicted_pose_distortion_avg=None,
        source="",
    )
    older = delta_logger.EmpiricalRecord(
        architecture_class="hnerv_test",
        axis="cuda",
        evidence_tag="[contest-CUDA]",
        archive_sha256=None,
        archive_bytes=100000,
        score_value=0.195,
        hardware_substrate="linux_x86_64",
        observed_at_utc="2026-05-01T00:00:00+00:00",
    )
    newer = delta_logger.EmpiricalRecord(
        architecture_class="hnerv_test",
        axis="cuda",
        evidence_tag="[contest-CUDA]",
        archive_sha256=None,
        archive_bytes=100000,
        score_value=0.193,
        hardware_substrate="linux_x86_64",
        observed_at_utc="2026-05-13T00:00:00+00:00",
    )
    matched = delta_logger.match_prediction_to_empirical(pred, [older, newer])
    assert matched is not None
    assert matched.score_value == pytest.approx(0.193)


# ---------------------------------------------------------------------------
# Delta verdict
# ---------------------------------------------------------------------------


def test_compute_delta_within_band() -> None:
    pred = delta_logger.PredictionRecord(
        lane_id="lane_x_20260513", predicted_score_band_low=0.18,
        predicted_score_band_high=0.20, predicted_archive_bytes=None,
        predicted_seg_distortion_avg=None, predicted_pose_distortion_avg=None,
        source="",
    )
    emp = delta_logger.EmpiricalRecord(
        architecture_class="x", axis="cuda", evidence_tag="[contest-CUDA]",
        archive_sha256=None, archive_bytes=100000, score_value=0.190,
        hardware_substrate="linux_x86_64", observed_at_utc="2026-05-13T00:00:00+00:00",
    )
    row = delta_logger.compute_delta_row(pred, emp)
    assert row.band_contained_empirical is True
    assert row.over_or_under == "within_band"
    assert row.delta == pytest.approx(0.0)


def test_compute_delta_over_predicted() -> None:
    pred = delta_logger.PredictionRecord(
        lane_id="lane_y_20260513", predicted_score_band_low=0.15,
        predicted_score_band_high=0.17, predicted_archive_bytes=None,
        predicted_seg_distortion_avg=None, predicted_pose_distortion_avg=None,
        source="",
    )
    emp = delta_logger.EmpiricalRecord(
        architecture_class="y", axis="cuda", evidence_tag="[contest-CUDA]",
        archive_sha256=None, archive_bytes=100000, score_value=0.21,
        hardware_substrate="linux_x86_64", observed_at_utc="2026-05-13T00:00:00+00:00",
    )
    row = delta_logger.compute_delta_row(pred, emp)
    assert row.band_contained_empirical is False
    assert row.over_or_under == "over_predicted"
    assert row.delta == pytest.approx(0.21 - 0.16)


def test_compute_delta_under_predicted() -> None:
    pred = delta_logger.PredictionRecord(
        lane_id="lane_z_20260513", predicted_score_band_low=0.22,
        predicted_score_band_high=0.24, predicted_archive_bytes=None,
        predicted_seg_distortion_avg=None, predicted_pose_distortion_avg=None,
        source="",
    )
    emp = delta_logger.EmpiricalRecord(
        architecture_class="z", axis="cuda", evidence_tag="[contest-CUDA]",
        archive_sha256=None, archive_bytes=100000, score_value=0.19,
        hardware_substrate="linux_x86_64", observed_at_utc="2026-05-13T00:00:00+00:00",
    )
    row = delta_logger.compute_delta_row(pred, emp)
    assert row.band_contained_empirical is False
    assert row.over_or_under == "under_predicted"


# ---------------------------------------------------------------------------
# Calibration recommendations
# ---------------------------------------------------------------------------


def test_calibration_recommendations_includes_per_substrate_records() -> None:
    rows = [
        # 1 within-band but very wide → tighten
        delta_logger.DeltaRow(
            lane_id="lane_wide_band_20260513",
            matched_empirical_architecture_class="wb",
            predicted_midpoint=0.190,
            predicted_score_band_low=0.10,
            predicted_score_band_high=0.28,  # band width 0.18 >> 4 * |0| = 0
            empirical_score=0.190,
            delta=0.0,
            band_contained_empirical=True,
            over_or_under="within_band",
            empirical_axis="cuda", empirical_evidence_tag="[contest-CUDA]",
            empirical_archive_sha256=None,
            empirical_observed_at_utc="2026-05-13T00:00:00+00:00",
        ),
        # 1 outside band → widen
        delta_logger.DeltaRow(
            lane_id="lane_missed_20260513",
            matched_empirical_architecture_class="m",
            predicted_midpoint=0.16,
            predicted_score_band_low=0.15,
            predicted_score_band_high=0.17,
            empirical_score=0.22,
            delta=0.06,
            band_contained_empirical=False,
            over_or_under="over_predicted",
            empirical_axis="cuda", empirical_evidence_tag="[contest-CUDA]",
            empirical_archive_sha256=None,
            empirical_observed_at_utc="2026-05-13T00:00:00+00:00",
        ),
    ]
    cal = delta_logger.build_calibration_report(rows)
    recs = {r["lane_id"]: r["recommendation"] for r in cal["per_substrate_recommendations"]}
    assert recs["lane_wide_band_20260513"] == "tighten_band"
    assert recs["lane_missed_20260513"] == "widen_band"
    assert cal["n_matched_substrates"] == 2
    assert cal["n_within_band"] == 1
    assert cal["n_over_predicted"] == 1


# ---------------------------------------------------------------------------
# Output IO
# ---------------------------------------------------------------------------


def test_write_outputs_emits_json_csv_calibration(tmp_path: Path) -> None:
    rows = [
        delta_logger.DeltaRow(
            lane_id="lane_io_test_20260513",
            matched_empirical_architecture_class="io",
            predicted_midpoint=0.19,
            predicted_score_band_low=0.18,
            predicted_score_band_high=0.20,
            empirical_score=0.193,
            delta=0.003,
            band_contained_empirical=True,
            over_or_under="within_band",
            empirical_axis="cuda", empirical_evidence_tag="[contest-CUDA]",
            empirical_archive_sha256="ff" * 32,
            empirical_observed_at_utc="2026-05-13T00:00:00+00:00",
        )
    ]
    calibration = delta_logger.build_calibration_report(rows)
    outputs = delta_logger.write_outputs(rows, [], calibration, tmp_path, plot=False)
    assert (tmp_path / "deltas.json").exists()
    assert (tmp_path / "deltas.csv").exists()
    assert (tmp_path / "calibration_report.json").exists()
    deltas_doc = json.loads((tmp_path / "deltas.json").read_text())
    assert deltas_doc["schema"] == "tac_prediction_vs_empirical_delta_v1"
    assert deltas_doc["score_claim"] is False
    assert deltas_doc["promotion_eligible"] is False
    csv_text = (tmp_path / "deltas.csv").read_text()
    assert "lane_io_test_20260513" in csv_text


# ---------------------------------------------------------------------------
# End-to-end main()
# ---------------------------------------------------------------------------


def test_main_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    preds_path = tmp_path / "preds.jsonl"
    post_path = tmp_path / "posterior.json"
    out_dir = tmp_path / "out"
    _write_predictions(preds_path, [
        _pred_row(lane_id="lane_a1_substrate_20260513", low=0.18, high=0.20),
    ])
    _write_posterior(post_path, [_emp_row(architecture_class="a1", score=0.193)])
    rc = delta_logger.main([
        "--predicted-anchors", str(preds_path),
        "--empirical-posterior", str(post_path),
        "--out-dir", str(out_dir),
    ])
    assert rc == 0
    captured = capsys.readouterr()
    assert "matched=1" in captured.out
    assert (out_dir / "deltas.json").exists()
    assert (out_dir / "calibration_report.json").exists()


def test_main_unmatched_lane_ids_surfaced(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    preds_path = tmp_path / "preds.jsonl"
    post_path = tmp_path / "posterior.json"
    out_dir = tmp_path / "out"
    _write_predictions(preds_path, [
        _pred_row(lane_id="lane_nonexistent_substrate_20260513"),
    ])
    _write_posterior(post_path, [_emp_row(architecture_class="completely_other", score=0.193)])
    rc = delta_logger.main([
        "--predicted-anchors", str(preds_path),
        "--empirical-posterior", str(post_path),
        "--out-dir", str(out_dir),
    ])
    assert rc == 0
    captured = capsys.readouterr()
    assert "matched=0" in captured.out
    assert "unmatched=1" in captured.out
    deltas_doc = json.loads((out_dir / "deltas.json").read_text())
    assert "lane_nonexistent_substrate_20260513" in deltas_doc["unmatched_lane_ids"]
