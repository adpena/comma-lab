# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

from tac.analysis.hinerv_distortion_stabilization_queue import (
    SCHEMA,
    build_hinerv_distortion_stabilization_queue,
)
from tools.build_hinerv_distortion_stabilization_queue import main as cli_main


def test_hinerv_distortion_queue_blocks_dynamic_range_before_local_replay(
    tmp_path: Path,
) -> None:
    report = build_hinerv_distortion_stabilization_queue(
        candidate_feedback_rows=[_feedback(dynamic_ok=False, full_video=False)],
        checkpoint_export_reports=[_export(dynamic_ok=False, prefilter=True)],
        output_root=tmp_path / "queue",
        min_free_bytes=0,
        allow_local_output=True,
        generated_utc="2026-06-05T00:00:00+00:00",
    )

    assert report["schema"] == SCHEMA
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["checkpoint_export_evidence"]["byte_closed_archive_export"] is True
    assert report["checkpoint_export_evidence"]["receiver_proof_ready"] is True
    dynamic = _node(report, "dynamic_range_scorer_input_stabilization")
    assert dynamic["blocked"] is True
    assert "candidate_segnet_last_rgb_dynamic_range_too_low" in dynamic["blockers"]
    assert "hinerv_checkpoint_fit_scale_gate_failed" in dynamic["blockers"]
    local = _node(report, "local_cpu_replay_gate")
    assert local["blocked"] is True
    assert "hi_nerv_full_video_local_prefilter_missing" in local["blockers"]
    exact = _node(report, "exact_cpu_cuda_dispatch_gate")
    assert exact["blocked"] is True


def test_hinerv_distortion_queue_marks_replay_ready_only_after_all_local_proofs(
    tmp_path: Path,
) -> None:
    report = build_hinerv_distortion_stabilization_queue(
        candidate_feedback_rows=[_feedback(dynamic_ok=True, full_video=True)],
        checkpoint_export_reports=[_export(dynamic_ok=True, prefilter=True)],
        waterfill_reports=[_waterfill(recon_pixel=True)],
        replay_actuator_reports=[_replay(receiver_rows=1)],
        output_root=tmp_path / "queue",
        min_free_bytes=0,
        allow_local_output=True,
        generated_utc="2026-06-05T00:00:00+00:00",
    )

    ready_nodes = {
        row["node_id"]
        for row in report["dag_nodes"]
        if row["status"] == "ready_no_authority"
    }
    assert "dynamic_range_scorer_input_stabilization" in ready_nodes
    assert "byte_closed_archive_export" in ready_nodes
    assert "receiver_archive_replay_proof" in ready_nodes
    assert "ema_archive_in_loop_selection" in ready_nodes
    assert "decoder_weight_waterfill_recon_pixel_proof" in ready_nodes
    assert "full_video_mlx_prefilter_gate" in ready_nodes
    assert "local_cpu_replay_gate" in ready_nodes
    assert _node(report, "exact_cpu_cuda_dispatch_gate")["blocked"] is True
    assert report["local_cpu_replay_allowed"] is False
    assert report["exact_cpu_cuda_dispatch_allowed"] is False


def test_hinerv_distortion_queue_cli_writes_artifacts(tmp_path: Path) -> None:
    feedback_path = tmp_path / "feedback.json"
    export_path = tmp_path / "export.json"
    output_root = tmp_path / "out"
    output_json = output_root / "queue.json"
    output_md = output_root / "queue.md"
    feedback_path.write_text(json.dumps(_feedback(dynamic_ok=False, full_video=False)))
    export_path.write_text(json.dumps(_export(dynamic_ok=False, prefilter=False)))

    rc = cli_main(
        [
            "--candidate-feedback-row",
            feedback_path.as_posix(),
            "--checkpoint-export-report",
            export_path.as_posix(),
            "--output-root",
            output_root.as_posix(),
            "--output-json",
            output_json.as_posix(),
            "--output-md",
            output_md.as_posix(),
            "--allow-local-output",
            "--min-free-bytes",
            "0",
        ]
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["schema"] == SCHEMA
    assert payload["feedback_evidence"]["source_path"] == feedback_path.as_posix()
    assert "HiNeRV Distortion Stabilization Queue" in output_md.read_text(
        encoding="utf-8"
    )


def _node(report: dict, node_id: str) -> dict:
    return next(row for row in report["dag_nodes"] if row["node_id"] == node_id)


def _feedback(*, dynamic_ok: bool, full_video: bool) -> dict[str, object]:
    blockers = []
    if not dynamic_ok:
        blockers.extend(
            [
                "candidate_segnet_last_rgb_dynamic_range_too_low",
                "mlx_renderer_prefilter_scorer_input_out_of_distribution",
                "hi_nerv_score_aware_training_direct_live_segnet_candidate_argmax_collapsed",
            ]
        )
    if not full_video:
        blockers.extend(
            [
                "hi_nerv_full_video_local_prefilter_missing",
                "hi_nerv_local_cpu_replay_gate_missing",
            ]
        )
    if dynamic_ok:
        blockers.append("contest_cpu_cuda_exact_eval_not_executed")
    return {
        "schema": "nerv_candidate_feedback_row.v1",
        "family": "hi_nerv",
        "created_utc": "2026-06-05T00:00:00+00:00",
        "candidate_id": "hinerv_unit",
        "measured_num_pairs": 600 if full_video else 16,
        "measured_archive_bytes": 123456,
        "mlx_prefilter_has_full_video": full_video,
        "local_cpu_replay_gate_has_full_video_mlx_prefilter": full_video,
        "local_cpu_replay_gate_local_replay_mlx_prefilter_passed": full_video,
        "mlx_prefilter_blockers": [] if dynamic_ok and full_video else blockers,
        "blockers": blockers,
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _export(*, dynamic_ok: bool, prefilter: bool) -> dict[str, object]:
    fit_blockers = [] if dynamic_ok else ["hinerv_checkpoint_fit_scale_gate_failed"]
    cache_blockers = [] if dynamic_ok else ["candidate_segnet_last_rgb_far_from_reference_fit_gate"]
    blockers = ["contest_cpu_cuda_exact_eval_not_executed"]
    if not prefilter:
        blockers.append("hinerv_checkpoint_mlx_prefilter_pending")
    return {
        "schema": "hinerv_checkpoint_archive_export.v1",
        "candidate_id": "hinerv_unit",
        "checkpoint_epoch": 10,
        "checkpoint_state_kind": "ema",
        "archive_path": "/Volumes/VertigoDataTier/pact/unit/archive.zip",
        "archive_bytes": 123456,
        "archive_sha256": "a" * 64,
        "receiver_closed": True,
        "receiver_contract_satisfied": True,
        "receiver_proof_ready": True,
        "receiver_proof_path": "/Volumes/VertigoDataTier/pact/unit/proof.json",
        "receiver_proof_sha256": "b" * 64,
        "local_mlx_prefilter_written": prefilter,
        "receiver_fit_scale_guard": {
            "schema": "hinerv_checkpoint_fit_scale_guard.v1",
            "gate_passed": dynamic_ok,
            "blockers": fit_blockers,
        },
        "local_mlx_prefilter_profile": {
            "cache_quality_gate": {
                "schema": "mlx_cache_quality_gate.v1",
                "fit_gate_passed": dynamic_ok,
                "blockers": cache_blockers,
            }
        },
        "blockers": blockers,
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _waterfill(*, recon_pixel: bool) -> dict[str, object]:
    return {
        "schema": "hinerv_archive_ladder_waterfill.v1",
        "report_path": "/Volumes/VertigoDataTier/pact/unit/waterfill.json",
        "recon_pixel_weight_proof_passed": recon_pixel,
        "rows": [{"row_id": "unit", "recon_pixel_weight_proof_passed": recon_pixel}],
        "blockers": [] if recon_pixel else ["hinerv_recon_pixel_weight_proof_missing"],
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _replay(*, receiver_rows: int) -> dict[str, object]:
    return {
        "schema": "hinerv_archive_ladder_replay_actuator.v1",
        "receiver_proof_ready_row_count": receiver_rows,
        "blockers": ["contest_cpu_cuda_exact_eval_not_executed"],
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }
