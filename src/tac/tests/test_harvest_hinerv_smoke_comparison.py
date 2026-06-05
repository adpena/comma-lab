# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

from tools import harvest_hinerv_smoke_comparison as harvest_cli


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_harvest_hinerv_smoke_comparison_emits_false_authority_refresh_rows(
    tmp_path: Path,
) -> None:
    runner_dir = tmp_path / "hinerv_directlive_allpixel_argmax_7ep_metricselect"
    row = {
        "schema": "nerv_candidate_feedback_row.v1",
        "family": "hi_nerv",
        "candidate_id": "hinerv_candidate_directlive",
        "archive_bytes": 120_000,
        "measured_num_pairs": 600,
        "feedback_ready": False,
        "receiver_proof_attached": False,
        "full_video_local_prefilter_attached": False,
        "local_cpu_replay_gate_attached": False,
        "blockers": [
            "hi_nerv_receiver_proof_missing",
            "hi_nerv_full_video_local_prefilter_missing",
        ],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    _write_json(
        runner_dir / "compact_renderer_mlx_spine_runner_report.json",
        {
            "schema": "compact_renderer_mlx_spine_runner.v1",
            "execute_family": "hi_nerv",
            "training_executed": True,
            "num_pairs": 600,
            "archive_bytes": 121_000,
            "candidate_feedback": {"row": row},
            "blockers": ["hi_nerv_receiver_proof_missing"],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )
    _write_json(
        runner_dir / "hprc_spine_acquisition_report.json",
        {
            "schema": "hprc_spine_acquisition_report.v1",
            "row_count": 1,
            "rows": [
                {
                    "schema": "hprc_spine_acquisition_row.v1",
                    "family": "hi_nerv",
                    "effective_archive_bytes": 119_500,
                    "promotable": False,
                    "recommended_next_action": "shrink_base_renderer_before_any_residual_sidecar",
                    "ceiling_results": [{"fits": False}],
                }
            ],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )
    _write_json(
        runner_dir
        / "hi_nerv_mlx_training"
        / "ema_archive_selection"
        / "live"
        / "hi_nerv_bitstream_preparation.json",
        {
            "schema": "hi_nerv_bitstream_preparation.v1",
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )

    report = harvest_cli.build_hinerv_smoke_comparison(artifact_roots=(tmp_path,))

    assert report["schema"] == "hinerv_smoke_comparison_harvest.v1"
    assert report["score_claim"] is False
    assert report["promotion_eligible"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["row_count"] == 1
    assert report["embedded_candidate_feedback_row_count"] == 1
    assert report["variant_counts"]["direct_live"] == 1
    assert report["variant_counts"]["argmax"] == 1
    assert report["variant_counts"]["bitstream"] == 1
    harvested = report["rows"][0]
    assert harvested["candidate_id"] == "hinerv_candidate_directlive"
    assert harvested["archive_bytes"] == 121_000
    assert harvested["best_acquisition_bytes"] == 119_500
    assert harvested["bitstream_preparation_count"] == 1
    assert harvested["next_action"] == "run_receiver_proof"
    refresh = report["feedback_refresh"]
    assert refresh["schema"] == "nerv_queue_training_feedback_refresh.v1"
    assert refresh["refreshed_row_count"] == 1
    assert refresh["rows"][0]["row"]["schema"] == "nerv_candidate_feedback_row.v1"
    assert refresh["score_claim"] is False
    assert refresh["ready_for_exact_eval_dispatch"] is False


def test_harvest_hinerv_smoke_comparison_writes_comparison_and_refresh(
    tmp_path: Path,
) -> None:
    export_dir = tmp_path / "hinerv_waterfill_bitstream_export"
    _write_json(
        export_dir / "hinerv_checkpoint_archive_export.json",
        {
            "schema": "hinerv_checkpoint_archive_export.v1",
            "family": "hi_nerv",
            "candidate_id": "hinerv_candidate_waterfill",
            "archive_bytes": 98_000,
            "archive_sha256": "a" * 64,
            "receiver_proof_ready": True,
            "local_mlx_prefilter_written": True,
            "blockers": ["hi_nerv_local_cpu_replay_gate_missing"],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )
    report = harvest_cli.build_hinerv_smoke_comparison(artifact_roots=(tmp_path,))
    out_json = tmp_path / "out" / "comparison.json"
    refresh_json = tmp_path / "out" / "refresh.json"
    out_md = tmp_path / "out" / "comparison.md"

    write = harvest_cli.write_hinerv_smoke_comparison(
        report=report,
        output_json=out_json,
        feedback_refresh_json=refresh_json,
        output_md=out_md,
    )

    assert Path(write["report_path"]).is_file()
    assert Path(write["feedback_refresh_path"]).is_file()
    assert Path(write["markdown_path"]).is_file()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["byte_frontier_row"]["run_id"] == "hinerv_waterfill_bitstream_export"
    assert payload["variant_counts"]["waterfill"] == 1
    assert payload["variant_counts"]["bitstream"] == 1
    refresh = json.loads(refresh_json.read_text(encoding="utf-8"))
    assert refresh["schema"] == "nerv_queue_training_feedback_refresh.v1"
    assert refresh["refreshed_row_count"] == 0
