# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

import tac.substrates.hprc.archive_candidate as hprc_candidate
from tac.optimization.archive_bound_candidate_runtime_bridge import (
    build_archive_bound_candidate_runtime_package,
)
from tac.substrates.hprc.archive import HprcSectionKind, pack_hprc_packet
from tac.substrates.hprc.campaign import build_hprc_queue_followup_report
from tac.substrates.hprc.inflate import hprc_preview_digest


def test_hprc_preview_changes_for_pixel_sections_but_not_manifest() -> None:
    packet = pack_hprc_packet(
        {
            HprcSectionKind.DECODER_QW: b"decoder",
            HprcSectionKind.LATENTS_RC: b"latents",
            HprcSectionKind.MANIFEST_JSON: b'{"note":"metadata"}',
        }
    )
    base = hprc_preview_digest(packet)
    decoder_mut = pack_hprc_packet(
        {
            HprcSectionKind.DECODER_QW: b"Decoder",
            HprcSectionKind.LATENTS_RC: b"latents",
            HprcSectionKind.MANIFEST_JSON: b'{"note":"metadata"}',
        }
    )
    manifest_mut = pack_hprc_packet(
        {
            HprcSectionKind.DECODER_QW: b"decoder",
            HprcSectionKind.LATENTS_RC: b"latents",
            HprcSectionKind.MANIFEST_JSON: b'{"note":"changed"}',
        }
    )

    assert hprc_preview_digest(decoder_mut) != base
    assert hprc_preview_digest(manifest_mut) == base


def test_hprc_section_mutation_proof_separates_metadata() -> None:
    packet = hprc_candidate.build_minimal_hprc_v0_packet()

    proof = hprc_candidate.build_hprc_section_mutation_proof(packet)

    assert proof["section_mutation_preview_ready"] is True
    assert proof["blockers"] == []
    per_section = {row["section"]: row for row in proof["per_section"]}
    assert per_section["decoder_qw"]["receiver_preview_changed"] is True
    assert per_section["latents_rc"]["receiver_preview_changed"] is True
    assert per_section["manifest_json"]["receiver_preview_changed"] is False
    assert per_section["manifest_json"]["metadata_only_expected"] is True


def test_hprc_export_emits_archive_bound_package(tmp_path: Path, monkeypatch) -> None:
    def fake_emit_runtime_package(**kwargs):
        proof = {
            "schema": kwargs["proof_schema"],
            "proof_path": "receiver_proof/hprc_receiver_proof.json",
            "runtime_consumption_proof_ready": True,
            "receiver_contract_satisfied": True,
            "blockers": [],
            "inflate_argv": ["inflate.sh", "archive_dir", "out", "file_list"],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
        return build_archive_bound_candidate_runtime_package(
            adapter_id=kwargs["adapter_id"],
            candidate_family=kwargs["candidate_family"],
            candidate_id_prefix=kwargs["candidate_id_prefix"],
            transform_kind=kwargs["transform_kind"],
            archive_zip_path=kwargs["archive_zip_path"],
            archive_sha256=kwargs["archive_sha256"],
            archive_bytes=kwargs["archive_bytes"],
            submission_dir=kwargs["submission_dir"],
            output_dir=kwargs["output_dir"],
            repo_root=kwargs["repo_root"],
            receiver_proof=proof,
            receiver_contract_kind=kwargs["receiver_contract_kind"],
            runtime_adapter_manifest_extra=kwargs["runtime_adapter_manifest_extra"],
            candidate_row_schema=kwargs["candidate_row_schema"],
            wrapper_schema=kwargs["wrapper_schema"],
            input_artifacts=kwargs["input_artifacts"],
            extra_blockers=kwargs["extra_blockers"],
            mlx_triage_argv=kwargs["mlx_triage_argv"],
        )

    monkeypatch.setattr(
        hprc_candidate,
        "emit_archive_bound_candidate_runtime_package",
        fake_emit_runtime_package,
    )

    archive_zip_path, archive_sha256, archive_bytes = (
        hprc_candidate.export_hprc_archive_bytes(
            hprc_candidate.build_minimal_hprc_v0_packet(),
            tmp_path,
            repo_root=tmp_path,
        )
    )

    assert archive_zip_path.is_file()
    assert len(archive_sha256) == 64
    assert archive_bytes == archive_zip_path.stat().st_size
    package_path = tmp_path / "archive_bound_candidate_adapter_package.json"
    package = json.loads(package_path.read_text(encoding="utf-8"))
    row = package["archive_bound_candidate_adapter_package"]["candidate_rows"][0]
    assert row["candidate_family"] == hprc_candidate.HPRC_ARCHIVE_CANDIDATE_FAMILY
    assert row["byte_closed_candidate_materialized"] is True
    assert row["runtime_consumption_proof_ready"] is True
    assert row["ready_for_exact_eval_dispatch"] is False
    assert "hprc_v0_receiver_scaffold_not_trained_renderer" in row["blockers"]
    contract = row["archive_bound_candidate_contract"]
    runtime_payload = contract["runtime_payload_consumption"]
    assert runtime_payload["declared"] is True
    assert runtime_payload["predictive_stack"] is True
    assert runtime_payload["full_stack_pixel_consumption_proven"] is False
    assert runtime_payload["status"] == (
        "section_pixel_consumption_proven_full_stack_claim_blocked"
    )
    assert "decoder_qw" in runtime_payload["pixel_consumed_archive_sections"]
    assert runtime_payload["next_materializer_tasks"] == [
        "replace_hprc_v0_receiver_scaffold_with_trained_renderer_export",
        "attach_z8_scorer_weighted_residual_sidecar",
        "prove_mamba_dreamer_wyner_ziv_sections_drive_receiver_pixels",
    ]
    assert package["archive_bound_candidate_adapter_package"][
        "runtime_payload_materializer_backlog_count"
    ] == 3
    assert (tmp_path / "hprc_archive_byte_ledger.json").is_file()


def test_hprc_queue_followup_report_blocks_partial_and_missing_z8(tmp_path: Path) -> None:
    result_path = tmp_path / "hprc_compact_receiver_training_run_result.json"
    result_path.write_text(
        json.dumps(
            {
                "schema": "hprc_compact_receiver_training_run_result.v1",
                "artifact": {
                    "archive_path": (tmp_path / "archive.zip").as_posix(),
                    "archive_sha256": "a" * 64,
                    "archive_bytes": 1234,
                    "score_claim": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        encoding="utf-8",
    )

    report = build_hprc_queue_followup_report(
        training_result_path=result_path,
        decode_pairs=32,
        repo_root=tmp_path,
    )

    assert report["schema"] == "hprc_queue_followup_report.v1"
    assert report["local_replay_gate"]["required"] is False
    assert "partial_pair_campaign_not_full_video_replay_candidate" in report[
        "local_replay_gate"
    ]["blockers"]
    assert report["z8_residual_sidecar_followup"]["status"] == "blocked"
    assert report["full_video_p18_p19_allocator_followup"]["status"] == "blocked"
    assert "exact_auth_gate_not_executed_or_missing" in report["exact_auth_gate"][
        "blockers"
    ]
    assert report["ready_for_exact_eval_dispatch"] is False


def test_hprc_queue_followup_report_accepts_local_replay_but_keeps_false_authority(
    tmp_path: Path,
) -> None:
    result_path = tmp_path / "hprc_compact_receiver_training_run_result.json"
    result_path.write_text(
        json.dumps(
            {
                "schema": "hprc_compact_receiver_training_run_result.v1",
                "artifact": {
                    "archive_path": (tmp_path / "archive.zip").as_posix(),
                    "archive_sha256": "b" * 64,
                    "archive_bytes": 4321,
                    "score_claim": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        encoding="utf-8",
    )
    replay_path = tmp_path / "local_replay.json"
    replay_path.write_text(
        json.dumps(
            {
                "schema": "local_submission_replay.v1",
                "evaluation_passed": True,
                "axis_tag": "[macOS-CPU advisory]",
                "local_score_estimate": 0.18,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        encoding="utf-8",
    )
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(
        json.dumps(
            {
                "schema": "local_candidate_exact_auth_gate.v1",
                "exact_auth_dispatch_recommended": True,
                "next_required_action": "claim_lane_and_run_exact_cpu_auth_eval",
                "blockers": [],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        encoding="utf-8",
    )
    z8_archive = tmp_path / "z8.bin"
    z8_archive.write_bytes(b"z8")
    z8_surface = tmp_path / "surface.npz"
    z8_surface.write_bytes(b"surface")

    report = build_hprc_queue_followup_report(
        training_result_path=result_path,
        decode_pairs=600,
        local_replay_summary_path=replay_path,
        exact_auth_gate_path=gate_path,
        z8_archive_bin_path=z8_archive,
        z8_surface_path=z8_surface,
        repo_root=tmp_path,
    )

    assert report["local_replay_gate"]["evaluation_passed"] is True
    assert report["exact_auth_gate"]["exact_auth_dispatch_recommended"] is True
    assert report["full_video_p18_p19_allocator_followup"]["status"] == (
        "ready_for_queue_execution"
    )
    assert report["promotion_gate"]["ready_for_exact_eval_dispatch"] is False
    assert "contest_cpu_cuda_exact_eval_not_executed" in report["promotion_gate"]["blockers"]
