# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from tac.optimization.decoder_q_selective_window_bridge import (
    DecoderQSelectiveWindowBridgeError,
    build_decoder_q_selective_window_bridge_plan,
    render_decoder_q_selective_window_bridge_markdown,
)
from tac.optimization.normalized_objective import RATE_SCORE_PER_BYTE


def _false_authority() -> dict:
    return {
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
    }


def _write_zip(path: Path, payload: bytes = b"decoder-q") -> str:
    info = zipfile.ZipInfo("x")
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.compress_type = zipfile.ZIP_STORED
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, payload)
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(tmp_path: Path, *, rank: int, start: int, gain: float) -> dict:
    candidate_path = tmp_path / f"candidate_pair_{start:04d}_{start + 1:04d}.json"
    baseline_path = tmp_path / f"baseline_pair_{start:04d}_{start + 1:04d}.json"
    candidate_path.write_text(json.dumps({"pair": [start, start + 1]}), encoding="utf-8")
    baseline_path.write_text(json.dumps({"baseline": [start, start + 1]}), encoding="utf-8")
    normalized_gain = gain / 600.0
    normalized_break_even = normalized_gain / RATE_SCORE_PER_BYTE
    return {
        "schema": "mlx_effective_spend_triage_candidate_row.v1",
        **_false_authority(),
        "candidate_generation_only": True,
        "archive_materialization_required": True,
        "requires_exact_auth_eval_before_score_claim": True,
        "selection_basis": "normalized_full_video_mlx_singleton_response_gain",
        "rank": rank,
        "row_id": f"row-{rank}",
        "family": "mlx_decoder_q",
        "candidate_id": f"mlx_scorer_response:window:{start}:{start + 1}",
        "pair_indices": [start, start + 1],
        "source_pair_window": [start, start + 1],
        "source_n_samples": 1,
        "source_batch_pairs": 1,
        "full_video_denominator": 600,
        "added_archive_bytes": 0,
        "normalized_full_video_scorer_gain_vs_baseline": normalized_gain,
        "projected_full_video_delta_vs_baseline_score": -normalized_gain,
        "break_even_added_bytes_from_normalized_full_video_gain": (
            normalized_break_even
        ),
        "normalized_full_video_byte_budget_margin_vs_break_even": normalized_break_even,
        "source_path": str(candidate_path),
        "window_baseline_source_path": str(baseline_path),
        "archive_sha256": "a" * 64,
        "raw_sha256": "b" * 64,
        "source_inflated_outputs_aggregate_sha256": "c" * 64,
        "source_candidate_cache_array_sha256": {"pair_indices": "d" * 64},
        "source_reference_cache_array_sha256": {"pair_indices": "e" * 64},
        "window_baseline_candidate_cache_array_sha256": {"pair_indices": "f" * 64},
        "window_baseline_reference_cache_array_sha256": {"pair_indices": "1" * 64},
        "source_posenet_sha256": "2" * 64,
        "source_segnet_sha256": "3" * 64,
        "observed_scorer_gain_vs_baseline": gain,
        "observed_scorer_delta_vs_baseline": -gain,
        "observed_delta_vs_baseline_score": -gain,
        "byte_budget_margin_vs_break_even": gain * 1000.0,
        "predicted_delta_vs_baseline_score": 0.001,
        "prediction_agrees_with_observed_gain": False,
    }


def _selection(tmp_path: Path) -> dict:
    return {
        "schema": "mlx_effective_spend_triage_candidate_selection.v1",
        **_false_authority(),
        "candidate_generation_only": True,
        "archive_materialization_required": True,
        "requires_exact_auth_eval_before_score_claim": True,
        "evidence_grade": "macOS-MLX-research-signal",
        "evidence_tag": "[macOS-MLX research-signal]",
        "producer": "fixture",
        "selection_policy": {"top_k": 3},
        "gates": {"effective_mlx_spend_triage_gate": {"status": "strict_pass"}},
        "summary": {"selected_count": 3},
        "selected_rows": [
            _row(tmp_path, rank=1, start=10, gain=0.002),
            _row(tmp_path, rank=2, start=11, gain=0.001),
            _row(tmp_path, rank=3, start=20, gain=0.0005),
        ],
    }


def _manifest(tmp_path: Path) -> dict:
    archive = tmp_path / "archive.zip"
    archive_sha = _write_zip(archive)
    return {
        "schema": "fec6_decoder_q_materialized_candidate_v1",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "candidate_id": "d1f1e56e042692f2",
        "archive_zip_path": str(archive),
        "archive_zip_bytes": archive.stat().st_size,
        "archive_zip_sha256": archive_sha,
        "archive_bin_sha256": "4" * 64,
        "mutation_row": {
            "fixed_length_runtime_compatible": True,
            "length_delta": 0,
            "source_decoder_sha256": "5" * 64,
            "mutated_decoder_sha256": "6" * 64,
            "q_before": 70,
            "q_after": 71,
            "mutation": {
                "tensor_name": "rgb_1.weight",
                "q_offset": 0,
                "delta": 1,
            },
            "tensor": {"raw_q_range": {"start": 159626, "end": 160112, "length": 486}},
            "op3v3_target_evidence": {
                "approx_compressed_range": {"start": 113039, "end": 113383, "length": 344}
            },
        },
    }


def test_bridge_plan_preserves_false_authority_and_requires_dqs1_materialization(tmp_path: Path) -> None:
    selection = _selection(tmp_path)
    manifest = _manifest(tmp_path)
    for row in selection["selected_rows"]:
        row["archive_sha256"] = manifest["archive_zip_sha256"]

    plan = build_decoder_q_selective_window_bridge_plan(
        selection,
        manifest,
        repo_root=tmp_path,
        lane_id="lane_decoder_q_bridge_test",
        coalesce_gap=0,
    )

    assert plan["score_claim"] is False
    assert plan["ready_for_exact_eval_dispatch"] is False
    assert plan["bridge_status"] == "ready_for_dqs1_tail_trailer_materialization"
    assert plan["summary"]["selected_window_count"] == 3
    assert plan["summary"]["coalesced_run_count"] == 2
    assert plan["summary"]["all_units_prediction_disagree_count"] == 3
    assert plan["materialized_decoder_q_candidate"]["mutation"]["tensor_name"] == "rgb_1.weight"
    assert plan["work_units"][0]["pair_window"] == [10, 11]
    assert plan["work_units"][0]["observed_mlx_window_gain"] == pytest.approx(0.002)
    assert "observed_mlx_gain" not in plan["work_units"][0]
    assert plan["work_units"][0]["normalized_full_video_gain"] == pytest.approx(0.002 / 600.0)
    assert plan["work_units"][0][
        "break_even_added_bytes_from_normalized_full_video_gain"
    ] == pytest.approx((0.002 / 600.0) / RATE_SCORE_PER_BYTE)
    assert "byte_budget_margin_vs_break_even" not in plan["work_units"][0]
    assert plan["coalesced_runs"][0]["pair_window"] == [10, 12]
    assert "local_mlx_gain_sum_non_authoritative" not in plan["coalesced_runs"][0]
    assert plan["coalesced_runs"][0]["normalized_full_video_gain_sum_non_authoritative"] == pytest.approx(
        0.003 / 600.0
    )
    assert "DQS1 packet materialization not run for this bridge plan" in plan["dispatch_blockers"]
    assert (
        plan["bridge_policy"]["runtime_strategy"]
        == "dqs1_tail_trailer_selective_runtime"
    )

    markdown = render_decoder_q_selective_window_bridge_markdown(plan)
    assert "Decoder-Q Selective Window Bridge Plan" in markdown
    assert "[macOS-MLX research-signal]" in markdown


def test_bridge_backfills_legacy_observed_selection_rows_to_normalized_objective(
    tmp_path: Path,
) -> None:
    selection = _selection(tmp_path)
    manifest = _manifest(tmp_path)
    for row in selection["selected_rows"]:
        row["archive_sha256"] = manifest["archive_zip_sha256"]
        row["selection_basis"] = "observed_strict_gated_mlx_singleton_response_gain"
        for key in (
            "source_n_samples",
            "full_video_denominator",
            "normalized_full_video_scorer_gain_vs_baseline",
            "projected_full_video_delta_vs_baseline_score",
            "break_even_added_bytes_from_normalized_full_video_gain",
            "normalized_full_video_byte_budget_margin_vs_break_even",
        ):
            row.pop(key)

    plan = build_decoder_q_selective_window_bridge_plan(
        selection,
        manifest,
        repo_root=tmp_path,
        lane_id="lane_decoder_q_bridge_test",
    )

    unit = plan["work_units"][0]
    assert unit["source_selection_basis"] == "normalized_full_video_mlx_singleton_response_gain"
    assert unit["legacy_selection_basis"] == "observed_strict_gated_mlx_singleton_response_gain"
    assert unit["normalized_objective_backfilled"] is True
    assert unit["source_n_samples"] == 1
    assert unit["full_video_denominator"] == 600
    assert unit["normalized_full_video_gain"] == pytest.approx(0.002 / 600.0)
    assert unit["break_even_added_bytes_from_normalized_full_video_gain"] == pytest.approx(
        (0.002 / 600.0) / RATE_SCORE_PER_BYTE
    )


def test_bridge_rejects_archive_sha_mismatch(tmp_path: Path) -> None:
    selection = _selection(tmp_path)
    manifest = _manifest(tmp_path)

    with pytest.raises(
        DecoderQSelectiveWindowBridgeError,
        match="does not match materialized candidate",
    ):
        build_decoder_q_selective_window_bridge_plan(
            selection,
            manifest,
            repo_root=tmp_path,
            lane_id="lane_decoder_q_bridge_test",
        )


def test_bridge_rejects_non_improving_projected_full_video_delta(tmp_path: Path) -> None:
    selection = _selection(tmp_path)
    manifest = _manifest(tmp_path)
    for row in selection["selected_rows"]:
        row["archive_sha256"] = manifest["archive_zip_sha256"]
    selection["selected_rows"][0]["projected_full_video_delta_vs_baseline_score"] = 0.0

    with pytest.raises(
        DecoderQSelectiveWindowBridgeError,
        match="projected full-video delta must be negative",
    ):
        build_decoder_q_selective_window_bridge_plan(
            selection,
            manifest,
            repo_root=tmp_path,
            lane_id="lane_decoder_q_bridge_test",
        )


def test_bridge_rejects_normalized_break_even_mismatch(tmp_path: Path) -> None:
    selection = _selection(tmp_path)
    manifest = _manifest(tmp_path)
    for row in selection["selected_rows"]:
        row["archive_sha256"] = manifest["archive_zip_sha256"]
    selection["selected_rows"][0]["break_even_added_bytes_from_normalized_full_video_gain"] = (
        selection["selected_rows"][0]["observed_scorer_gain_vs_baseline"]
        / RATE_SCORE_PER_BYTE
    )

    with pytest.raises(
        DecoderQSelectiveWindowBridgeError,
        match="normalized_full_video_break_even_mismatch",
    ):
        build_decoder_q_selective_window_bridge_plan(
            selection,
            manifest,
            repo_root=tmp_path,
            lane_id="lane_decoder_q_bridge_test",
        )


def test_bridge_rejects_rows_with_score_authority(tmp_path: Path) -> None:
    selection = _selection(tmp_path)
    manifest = _manifest(tmp_path)
    for row in selection["selected_rows"]:
        row["archive_sha256"] = manifest["archive_zip_sha256"]
    selection["selected_rows"][0]["score_claim"] = True

    with pytest.raises(DecoderQSelectiveWindowBridgeError, match="score_claim"):
        build_decoder_q_selective_window_bridge_plan(
            selection,
            manifest,
            repo_root=tmp_path,
            lane_id="lane_decoder_q_bridge_test",
        )


def test_bridge_rejects_duplicate_window_units(tmp_path: Path) -> None:
    selection = _selection(tmp_path)
    manifest = _manifest(tmp_path)
    for row in selection["selected_rows"]:
        row["archive_sha256"] = manifest["archive_zip_sha256"]
    selection["selected_rows"][1]["source_pair_window"] = [10, 11]
    selection["selected_rows"][1]["pair_indices"] = [10, 11]

    with pytest.raises(DecoderQSelectiveWindowBridgeError, match="duplicate pair windows"):
        build_decoder_q_selective_window_bridge_plan(
            selection,
            manifest,
            repo_root=tmp_path,
            lane_id="lane_decoder_q_bridge_test",
        )
