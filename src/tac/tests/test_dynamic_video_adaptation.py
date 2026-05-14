# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.adaptation.dynamic_byte_allocator import (
    DynamicByteAllocatorError,
    build_dynamic_byte_atom_ledger,
)
from tac.adaptation.video_telemetry import (
    DynamicVideoTelemetryError,
    TelemetryPairRow,
    build_dynamic_video_telemetry,
    telemetry_to_hard_pair_indices,
    write_hard_pair_indices_file,
)
from tac.packet_compiler.dynamic_video_adaptive_packet import (
    DynamicVideoAdaptivePacketError,
    build_dynamic_packet_manifest,
)


def _telemetry() -> dict:
    rows = [
        TelemetryPairRow(
            pair_idx=7,
            frame_indices=(14, 15),
            seg_dist=0.004,
            pose_dist=0.030,
            score_contribution=0.53,
            hard_pair_rank=1,
            hard_frame_flags=("pose_hard",),
            xray_artifact_refs=("xray/per_pair.json",),
        ),
        TelemetryPairRow(
            pair_idx=2,
            frame_indices=(4, 5),
            seg_dist=0.002,
            pose_dist=0.010,
            score_contribution=0.28,
            hard_pair_rank=2,
        ),
    ]
    return build_dynamic_video_telemetry(
        video_sha256="1" * 64,
        runtime_tree_sha256="2" * 64,
        source_archive_sha256="a" * 64,
        axis_label="[macOS-CPU advisory]",
        rows=rows,
        scorer_cache_ref="experiments/results/cache/gt_scorer_cache.json",
    )


def test_dynamic_video_telemetry_schema_is_proxy_safe():
    telemetry = _telemetry()
    assert telemetry["schema"] == "dynamic_video_telemetry_v1"
    assert telemetry["score_claim"] is False
    assert telemetry["ready_for_exact_eval_dispatch"] is False
    assert telemetry["inflate_allowed"] is False
    assert telemetry["scorer_allowed_at_inflate"] is False
    assert telemetry["hard_pair_indices"] == [7, 2]
    assert "bit_allocator" in telemetry["wire_in_hooks_engaged"]


def test_dynamic_video_telemetry_rejects_ambiguous_axis():
    with pytest.raises(DynamicVideoTelemetryError, match="axis_label"):
        build_dynamic_video_telemetry(
            video_sha256="1" * 64,
            runtime_tree_sha256="2" * 64,
            axis_label="[contest-ish]",
            rows=[
                TelemetryPairRow(
                    pair_idx=0,
                    frame_indices=(0, 1),
                    seg_dist=0.0,
                    pose_dist=0.0,
                    score_contribution=0.0,
                    hard_pair_rank=1,
                )
            ],
        )


def test_dynamic_video_telemetry_rejects_malformed_sha256():
    with pytest.raises(DynamicVideoTelemetryError, match="video_sha256"):
        build_dynamic_video_telemetry(
            video_sha256="not-a-sha",
            runtime_tree_sha256="2" * 64,
            axis_label="[proxy]",
            rows=[
                TelemetryPairRow(
                    pair_idx=0,
                    frame_indices=(0, 1),
                    seg_dist=0.0,
                    pose_dist=0.0,
                    score_contribution=0.0,
                    hard_pair_rank=1,
                )
            ],
        )


def test_dynamic_video_telemetry_contest_axis_requires_eval_custody():
    with pytest.raises(DynamicVideoTelemetryError, match="eval_custody"):
        build_dynamic_video_telemetry(
            video_sha256="1" * 64,
            runtime_tree_sha256="2" * 64,
            axis_label="[contest-CUDA]",
            rows=[
                TelemetryPairRow(
                    pair_idx=0,
                    frame_indices=(0, 1),
                    seg_dist=0.0,
                    pose_dist=0.0,
                    score_contribution=0.0,
                    hard_pair_rank=1,
                )
            ],
        )


def test_dynamic_video_telemetry_accepts_contest_axis_with_eval_custody():
    telemetry = build_dynamic_video_telemetry(
        video_sha256="1" * 64,
        runtime_tree_sha256="2" * 64,
        axis_label="[contest-CUDA]",
        rows=[
            TelemetryPairRow(
                pair_idx=0,
                frame_indices=(0, 1),
                seg_dist=0.0,
                pose_dist=0.0,
                score_contribution=0.0,
                hard_pair_rank=1,
            )
        ],
        eval_custody={
            "command": "modal run experiments/modal_auth_eval.py",
            "hardware": "T4",
            "sample_count": 600,
            "component_recomputed": True,
            "auth_eval_json_sha256": "3" * 64,
        },
    )
    assert telemetry["eval_custody"]["sample_count"] == 600
    assert telemetry["eval_custody"]["auth_eval_json_sha256"] == "3" * 64


def test_hard_pair_indices_file_feeds_mdl_ablation_cli(tmp_path: Path):
    telemetry = _telemetry()
    assert telemetry_to_hard_pair_indices(telemetry, top_k=1) == [7]
    out = write_hard_pair_indices_file(telemetry, tmp_path / "pairs.json", top_k=2)
    assert json.loads(out.read_text()) == [7, 2]


def test_dynamic_byte_allocator_emits_planning_atoms_only():
    plan = build_dynamic_byte_atom_ledger(
        _telemetry(),
        top_k=2,
        section="decoder_blob",
        operation="ibps1_byte_patch",
        byte_delta_per_pair=0,
        predicted_score_delta_per_pair=-0.01,
        confidence=0.4,
        evidence_source_path="experiments/results/dvar1/telemetry.json",
    )
    assert plan["schema"] == "dynamic_byte_atom_ledger_v1"
    assert plan["score_claim"] is False
    assert plan["ready_for_exact_eval_dispatch"] is False
    assert plan["pair_count"] == 2
    assert plan["atoms"][0]["pair_idx"] == 7
    assert plan["atoms"][0]["rankable"] is False
    assert "requires_byte_closed_packet" in plan["atoms"][0]["dispatch_blockers"]
    assert "atom_ledger" in plan


def test_dynamic_byte_allocator_rejects_unknown_operation():
    with pytest.raises(DynamicByteAllocatorError, match="operation"):
        build_dynamic_byte_atom_ledger(
            _telemetry(),
            top_k=1,
            section="decoder_blob",
            operation="magic_unregistered",
            byte_delta_per_pair=0,
            predicted_score_delta_per_pair=-0.01,
            confidence=0.5,
            evidence_source_path="x",
        )


def test_dynamic_packet_manifest_blocks_promotion_until_exact_eval():
    manifest = build_dynamic_packet_manifest(
        target_profile="contest_one_video_replay",
        baseline_archive_sha256="a" * 64,
        candidate_archive_sha256="b" * 64,
        baseline_archive_bytes=224481,
        candidate_archive_bytes=224481,
        runtime_tree_sha256="c" * 64,
        parser_section_manifest={
            "decoder_blob": {
                "start": 65507,
                "length": 145198,
                "sha256": "d" * 64,
            }
        },
        consumed_sections=("decoder_blob",),
        runtime_consumption_proof={
            "inflate_entrypoint": "inflate.sh",
            "consumed_sections": ("decoder_blob",),
            "proof_artifacts": ("no_op_proof.json",),
            "scorer_loaded_at_inflate": False,
        },
        no_op_proof_inputs=("archive_byte_diff", "inflate_runtime_consumes_decoder_blob"),
        atom_ledger_ref="experiments/results/dvar1/atoms.json",
        telemetry_ref="experiments/results/dvar1/telemetry.json",
        reproduction_commands=(["python", "tools/build_ibps1_byte_patch_archive.py"],),
    )
    assert manifest["schema"] == "dynamic_packet_manifest_v1"
    assert manifest["contest_dispatch_candidate"] is True
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["scorer_allowed_at_inflate"] is False
    assert manifest["byte_delta"] == 0
    assert "requires_exact_eval_before_score_claim" in manifest["dispatch_blockers"]


def test_dynamic_packet_manifest_requires_no_op_proof_inputs():
    with pytest.raises(DynamicVideoAdaptivePacketError, match="no_op_proof"):
        build_dynamic_packet_manifest(
            target_profile="contest_one_video_replay",
            baseline_archive_sha256="a" * 64,
            candidate_archive_sha256="b" * 64,
            baseline_archive_bytes=1,
            candidate_archive_bytes=1,
            runtime_tree_sha256="c" * 64,
            parser_section_manifest={"x": {"start": 0, "length": 1, "sha256": "d" * 64}},
            consumed_sections=("x",),
            runtime_consumption_proof={
                "inflate_entrypoint": "inflate.sh",
                "consumed_sections": ("x",),
                "proof_artifacts": ("no_op_proof.json",),
                "scorer_loaded_at_inflate": False,
            },
            no_op_proof_inputs=(),
            atom_ledger_ref="atoms.json",
            telemetry_ref="telemetry.json",
        )


def test_dynamic_packet_manifest_rejects_malformed_runtime_sha256():
    with pytest.raises(DynamicVideoAdaptivePacketError, match="runtime_tree_sha256"):
        build_dynamic_packet_manifest(
            target_profile="contest_one_video_replay",
            baseline_archive_sha256="a" * 64,
            candidate_archive_sha256="b" * 64,
            baseline_archive_bytes=1,
            candidate_archive_bytes=1,
            runtime_tree_sha256="runtime-tree-not-sha",
            parser_section_manifest={"x": {"start": 0, "length": 1, "sha256": "d" * 64}},
            consumed_sections=("x",),
            runtime_consumption_proof={
                "inflate_entrypoint": "inflate.sh",
                "consumed_sections": ("x",),
                "proof_artifacts": ("no_op_proof.json",),
                "scorer_loaded_at_inflate": False,
            },
            no_op_proof_inputs=("archive_byte_diff",),
            atom_ledger_ref="atoms.json",
            telemetry_ref="telemetry.json",
        )


def test_dynamic_packet_manifest_requires_consumed_section_manifest_sha():
    with pytest.raises(DynamicVideoAdaptivePacketError, match="decoder_blob.sha256"):
        build_dynamic_packet_manifest(
            target_profile="contest_one_video_replay",
            baseline_archive_sha256="a" * 64,
            candidate_archive_sha256="b" * 64,
            baseline_archive_bytes=1,
            candidate_archive_bytes=1,
            runtime_tree_sha256="c" * 64,
            parser_section_manifest={"decoder_blob": {"start": 0, "length": 1}},
            consumed_sections=("decoder_blob",),
            runtime_consumption_proof={
                "inflate_entrypoint": "inflate.sh",
                "consumed_sections": ("decoder_blob",),
                "proof_artifacts": ("no_op_proof.json",),
                "scorer_loaded_at_inflate": False,
            },
            no_op_proof_inputs=("archive_byte_diff",),
            atom_ledger_ref="atoms.json",
            telemetry_ref="telemetry.json",
        )


def test_dynamic_packet_manifest_rejects_scorer_loaded_runtime_proof():
    with pytest.raises(DynamicVideoAdaptivePacketError, match="scorer"):
        build_dynamic_packet_manifest(
            target_profile="contest_one_video_replay",
            baseline_archive_sha256="a" * 64,
            candidate_archive_sha256="b" * 64,
            baseline_archive_bytes=1,
            candidate_archive_bytes=1,
            runtime_tree_sha256="c" * 64,
            parser_section_manifest={
                "decoder_blob": {"start": 0, "length": 1, "sha256": "d" * 64}
            },
            consumed_sections=("decoder_blob",),
            runtime_consumption_proof={
                "inflate_entrypoint": "inflate.sh",
                "consumed_sections": ("decoder_blob",),
                "proof_artifacts": ("no_op_proof.json",),
                "scorer_loaded_at_inflate": True,
            },
            no_op_proof_inputs=("archive_byte_diff",),
            atom_ledger_ref="atoms.json",
            telemetry_ref="telemetry.json",
        )
