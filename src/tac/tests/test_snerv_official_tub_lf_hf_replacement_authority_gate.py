# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

from tac.analysis.snerv_official_tub_lf_hf_replacement_authority_gate import (
    FULL_REPLAY_BLOCKER,
    NUMERICAL_SOURCE_FORWARD_PROOF_BLOCKER,
    SCHEMA,
    SOURCE_AUTHORITY_BLOCKER,
    build_snerv_official_tub_lf_hf_replacement_authority_gate,
)
from tac.tests.snerv_source_forward_fixtures import (
    valid_snerv_source_forward_action_effect,
)
from tools.build_snerv_official_tub_lf_hf_replacement_authority_gate import (
    main as cli_main,
)


def test_authority_gate_blocks_current_export_bound_receiver_replay_shape(
    tmp_path: Path,
) -> None:
    report = build_snerv_official_tub_lf_hf_replacement_authority_gate(
        source_forward_artifacts=[
            _source_forward_artifact(
                official_export_bound=True,
                receiver_consumes_output2=True,
                source_authority=False,
                full_tub_parity=False,
            )
        ],
        checkpoint_export_reports=[_checkpoint_export_report(trained_mapping=False)],
        output_root=tmp_path / "gate",
        min_free_bytes=0,
        allow_local_output=True,
        generated_utc="2026-06-05T00:00:00+00:00",
    )

    assert report["schema"] == SCHEMA
    assert report["official_checkpoint_export_binding_ready"] is True
    assert report["receiver_output2_frame_replay_ready"] is True
    assert report["tub_source_fixture_replay_ready"] is True
    assert report["trained_checkpoint_state_dict_mapping_ready"] is False
    assert report["full_tub_source_forward_replay_ready"] is False
    assert report["official_tub_lf_hf_decoder_replacement_ready"] is False
    blockers = set(report["queue_blockers"])
    assert "snerv_official_mfu_hfr_tub_export_not_bound" not in blockers
    assert "snerv_official_mfu_hfr_tub_receiver_payload_not_bound" not in blockers
    assert (
        "snerv_official_mfu_hfr_tub_receiver_payload_not_source_forward_authority"
        in blockers
    )
    assert "snerv_official_mfu_hfr_tub_full_stack_source_forward_replay_missing" in blockers
    assert "snerv_official_trained_checkpoint_state_dict_mapping_missing" in blockers
    assert (
        "snerv_official_tub_portable_output2_decoder_weight_mapping_missing"
        in blockers
    )
    assert "snerv_official_tub_source_fixture_replay_missing" not in blockers
    assert (
        "snerv_official_tub_frame_reconstruction_source_forward_replay_missing"
        in report["closed_campaign_blockers"]
    )
    fixture_gate = next(
        row for row in report["gate_rows"] if row["gate_id"] == "tub_source_fixture_replay"
    )
    assert fixture_gate["blocked"] is False
    tub_state = report["tub_source_forward_evidence"]
    assert tub_state["selected_artifact_source"] == "nested_in_source_forward_artifact"
    assert tub_state["fixture_source_replay_passed"] is True
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False


def test_authority_gate_can_open_without_score_or_dispatch_authority(
    tmp_path: Path,
) -> None:
    report = build_snerv_official_tub_lf_hf_replacement_authority_gate(
        source_forward_artifacts=[
            _source_forward_artifact(
                official_export_bound=True,
                receiver_consumes_output2=True,
                source_authority=True,
                full_tub_parity=True,
            )
        ],
        checkpoint_export_reports=[_checkpoint_export_report(trained_mapping=True)],
        output_root=tmp_path / "gate",
        min_free_bytes=0,
        allow_local_output=True,
        generated_utc="2026-06-05T00:00:00+00:00",
    )

    assert report["official_tub_lf_hf_decoder_replacement_ready"] is True
    assert report["blocked_gate_row_count"] == 0
    assert report["tub_source_fixture_replay_ready"] is True
    assert report["queue_blockers"] == []
    assert report["score_claim"] is False
    assert report["promotion_eligible"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["local_mlx_long_training_allowed"] is False


def test_authority_gate_rejects_metadata_only_source_forward_authority(
    tmp_path: Path,
) -> None:
    source = _source_forward_artifact(
        official_export_bound=True,
        receiver_consumes_output2=True,
        source_authority=True,
        full_tub_parity=True,
    )
    source["source_forward_replay_proof"] = {
        "section_names": ["MFU", "HFR", "TUB"],
        "shapes_match": True,
    }
    source["source_forward_replay_authority"] = True
    source["source_forward_replay_closed_blockers"] = [FULL_REPLAY_BLOCKER]
    source["source_forward_authority_closed_blockers"] = [SOURCE_AUTHORITY_BLOCKER]

    report = build_snerv_official_tub_lf_hf_replacement_authority_gate(
        source_forward_artifacts=[source],
        checkpoint_export_reports=[_checkpoint_export_report(trained_mapping=True)],
        output_root=tmp_path / "gate",
        min_free_bytes=0,
        allow_local_output=True,
        generated_utc="2026-06-05T00:00:00+00:00",
    )

    assert report["official_tub_lf_hf_decoder_replacement_ready"] is False
    assert report["full_tub_source_forward_replay_ready"] is False
    source_evidence = report["source_forward_evidence"]
    assert source_evidence["source_forward_replay_numerical_proof_complete"] is False
    assert (
        source_evidence["source_forward_replay_proof_status"][
            "source_forward_replay_proof_status"
        ]
        == "metadata_only_or_incomplete_source_forward_proof"
    )
    blockers = set(report["queue_blockers"])
    assert NUMERICAL_SOURCE_FORWARD_PROOF_BLOCKER in blockers
    assert SOURCE_AUTHORITY_BLOCKER in blockers
    assert FULL_REPLAY_BLOCKER in blockers
    assert NUMERICAL_SOURCE_FORWARD_PROOF_BLOCKER not in report[
        "closed_campaign_blockers"
    ]
    assert SOURCE_AUTHORITY_BLOCKER not in report["closed_campaign_blockers"]
    assert FULL_REPLAY_BLOCKER not in report["closed_campaign_blockers"]


def test_authority_gate_rejects_fixture_scope_source_forward_action_effect(
    tmp_path: Path,
) -> None:
    source = _source_forward_artifact(
        official_export_bound=True,
        receiver_consumes_output2=True,
        source_authority=True,
        full_tub_parity=True,
    )
    proof = valid_snerv_source_forward_action_effect()
    proof["surface_provenance"]["official_torch"]["source_scope"] = (
        "official_source_fixture_state"
    )
    proof["surface_provenance"]["official_torch"]["capture_origin"] = (
        "official_upstream_source_fixture"
    )
    source["source_forward_replay_proof"] = proof

    report = build_snerv_official_tub_lf_hf_replacement_authority_gate(
        source_forward_artifacts=[source],
        checkpoint_export_reports=[_checkpoint_export_report(trained_mapping=True)],
        output_root=tmp_path / "gate",
        min_free_bytes=0,
        allow_local_output=True,
        generated_utc="2026-06-05T00:00:00+00:00",
    )

    assert report["official_tub_lf_hf_decoder_replacement_ready"] is False
    proof_status = report["source_forward_evidence"]["source_forward_replay_proof_status"]
    assert proof_status["source_forward_replay_action_effect_valid"] is False
    assert (
        "snerv_source_forward_official_torch_trained_checkpoint_source_scope_missing"
        in proof_status["source_forward_replay_action_effect_blockers"]
    )
    blockers = set(report["queue_blockers"])
    assert NUMERICAL_SOURCE_FORWARD_PROOF_BLOCKER in blockers
    assert SOURCE_AUTHORITY_BLOCKER in blockers
    assert FULL_REPLAY_BLOCKER in blockers


def test_authority_gate_keeps_residual_source_forward_blockers_sticky(
    tmp_path: Path,
) -> None:
    source = _source_forward_artifact(
        official_export_bound=True,
        receiver_consumes_output2=True,
        source_authority=True,
        full_tub_parity=True,
    )
    source["blockers"] = [
        "official_weight_tensor_mapping_not_loaded",
        "full_official_mfu_forward_artifact_not_emitted",
        "snerv_official_pytorch_wavelets_runtime_dependency_missing",
    ]
    source["receiver_payload_frame_replay"][
        "source_forward_authority_residual_blockers"
    ] = [
        "official_weight_tensor_mapping_not_loaded",
        "full_official_mfu_forward_artifact_not_emitted",
        "snerv_official_pytorch_wavelets_runtime_dependency_missing",
    ]

    report = build_snerv_official_tub_lf_hf_replacement_authority_gate(
        source_forward_artifacts=[source],
        checkpoint_export_reports=[_checkpoint_export_report(trained_mapping=True)],
        output_root=tmp_path / "gate",
        min_free_bytes=0,
        allow_local_output=True,
        generated_utc="2026-06-05T00:00:00+00:00",
    )

    assert report["official_tub_lf_hf_decoder_replacement_ready"] is False
    assert report["full_tub_source_forward_replay_ready"] is False
    assert (
        report["source_forward_evidence"]["receiver_payload_source_forward_authority"]
        is False
    )
    assert report["source_forward_authority_residual_blockers"] == [
        "official_weight_tensor_mapping_not_loaded",
        "full_official_mfu_forward_artifact_not_emitted",
        "snerv_official_pytorch_wavelets_runtime_dependency_missing",
    ]
    assert report["source_forward_evidence"][
        "source_forward_authority_residual_blockers"
    ] == report["source_forward_authority_residual_blockers"]
    blockers = set(report["queue_blockers"])
    assert (
        "snerv_official_mfu_hfr_tub_receiver_payload_not_source_forward_authority"
        in blockers
    )
    assert "snerv_official_mfu_hfr_tub_full_stack_source_forward_replay_missing" in (
        blockers
    )
    assert "official_weight_tensor_mapping_not_loaded" in blockers
    assert "full_official_mfu_forward_artifact_not_emitted" in blockers
    assert "snerv_official_pytorch_wavelets_runtime_dependency_missing" in blockers
    assert (
        "snerv_official_mfu_hfr_tub_receiver_payload_not_source_forward_authority"
        not in report["closed_campaign_blockers"]
    )


def test_authority_gate_consumes_source_artifact_closed_blockers(
    tmp_path: Path,
) -> None:
    source = _source_forward_artifact(
        official_export_bound=True,
        receiver_consumes_output2=True,
        source_authority=True,
        full_tub_parity=True,
    )
    source["blockers"] = []
    source["source_forward_replay_closed_blockers"] = [
        "snerv_official_mfu_hfr_tub_full_stack_source_forward_replay_missing",
        "snerv_official_snerv_t_trained_full_tub_source_forward_parity_missing",
        "snerv_official_trained_checkpoint_source_forward_replay_missing",
    ]
    source["source_forward_authority_closed_blockers"] = [
        "snerv_official_mfu_hfr_tub_receiver_payload_not_source_forward_authority"
    ]
    source["source_forward_component_closed_blockers"] = [
        "official_weight_tensor_mapping_not_loaded",
        "full_official_mfu_forward_artifact_not_emitted",
        "official_hfr_weight_tensor_mapping_not_loaded",
        "full_official_hfr_forward_artifact_not_emitted",
        "snerv_official_pytorch_wavelets_runtime_dependency_missing",
    ]
    source["source_forward_replay_authority"] = True
    source["official_tub_source_forward_replay"]["blockers"] = [
        "snerv_official_snerv_t_trained_full_tub_source_forward_parity_missing",
        "snerv_official_pytorch_wavelets_runtime_dependency_missing",
    ]
    source["official_tub_source_forward_replay"]["preserved_blockers"] = [
        "snerv_official_snerv_t_trained_full_tub_source_forward_parity_missing",
        "snerv_official_pytorch_wavelets_runtime_dependency_missing",
    ]

    report = build_snerv_official_tub_lf_hf_replacement_authority_gate(
        source_forward_artifacts=[source],
        checkpoint_export_reports=[_checkpoint_export_report(trained_mapping=True)],
        output_root=tmp_path / "gate",
        min_free_bytes=0,
        allow_local_output=True,
        generated_utc="2026-06-05T00:00:00+00:00",
    )

    assert report["official_tub_lf_hf_decoder_replacement_ready"] is True
    assert report["queue_blockers"] == []
    assert report["source_forward_authority_residual_blockers"] == []
    assert "snerv_official_pytorch_wavelets_runtime_dependency_missing" in report[
        "closed_campaign_blockers"
    ]
    assert (
        "snerv_official_mfu_hfr_tub_receiver_payload_not_source_forward_authority"
        in report["closed_campaign_blockers"]
    )
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False


def test_authority_gate_closes_stale_source_mapping_residuals_from_source_manifest(
    tmp_path: Path,
) -> None:
    source = _source_forward_artifact(
        official_export_bound=True,
        receiver_consumes_output2=True,
        source_authority=True,
        full_tub_parity=True,
    )
    residuals = [
        "official_weight_tensor_mapping_not_loaded",
        "official_hfr_weight_tensor_mapping_not_loaded",
        "full_official_mfu_forward_artifact_not_emitted",
        "snerv_official_pytorch_wavelets_runtime_dependency_missing",
    ]
    source["blockers"] = list(residuals)
    source["receiver_payload_frame_replay"][
        "source_forward_authority_residual_blockers"
    ] = list(residuals)
    source["official_trained_checkpoint_mapping_manifest"] = {
        "schema": "snerv_official_trained_checkpoint_state_dict_mapping_manifest.v1",
        "official_trained_checkpoint_loaded": True,
        "official_trained_checkpoint_state_dict_mapping_verified": True,
        "official_mfu_hfr_trained_checkpoint_weight_mapping_proven": True,
        "official_mfu_trained_checkpoint_weight_mapping_proven": True,
        "official_hfr_trained_checkpoint_weight_mapping_proven": True,
        "official_tub_temporal_encoder_weight_mapping_proven": True,
        "official_tub_output2_decoder_weight_mapping_proven": True,
        "blockers": [],
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }

    checkpoint = _checkpoint_export_report(trained_mapping=False)
    checkpoint["official_checkpoint_export_binding"]["blockers"].append(
        "snerv_official_trained_checkpoint_source_forward_replay_missing"
    )

    report = build_snerv_official_tub_lf_hf_replacement_authority_gate(
        source_forward_artifacts=[source],
        checkpoint_export_reports=[checkpoint],
        output_root=tmp_path / "gate",
        min_free_bytes=0,
        allow_local_output=True,
        generated_utc="2026-06-05T00:00:00+00:00",
    )

    assert report["official_tub_lf_hf_decoder_replacement_ready"] is False
    assert report["trained_checkpoint_state_dict_mapping_ready"] is True
    assert report["tub_temporal_output2_weight_mapping_ready"] is True
    assert report["source_forward_authority_residual_blockers"] == [
        "full_official_mfu_forward_artifact_not_emitted",
        "snerv_official_pytorch_wavelets_runtime_dependency_missing",
    ]
    blockers = set(report["queue_blockers"])
    assert "official_weight_tensor_mapping_not_loaded" not in blockers
    assert "official_hfr_weight_tensor_mapping_not_loaded" not in blockers
    assert "snerv_official_trained_checkpoint_hfr_weight_mapping_incomplete" not in blockers
    assert "snerv_official_trained_checkpoint_mfu_weight_mapping_incomplete" not in blockers
    assert "snerv_official_trained_checkpoint_source_forward_replay_missing" not in blockers
    assert "full_official_mfu_forward_artifact_not_emitted" in blockers
    assert "snerv_official_pytorch_wavelets_runtime_dependency_missing" in blockers
    assert "official_weight_tensor_mapping_not_loaded" in report[
        "closed_campaign_blockers"
    ]
    assert "official_hfr_weight_tensor_mapping_not_loaded" in report[
        "closed_campaign_blockers"
    ]
    assert "snerv_official_trained_checkpoint_source_forward_replay_missing" in report[
        "closed_campaign_blockers"
    ]
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False


def test_authority_gate_preserves_source_and_checkpoint_blockers_for_partial_mapping(
    tmp_path: Path,
) -> None:
    source = _source_forward_artifact(
        official_export_bound=True,
        receiver_consumes_output2=True,
        source_authority=True,
        full_tub_parity=True,
    )
    source["blockers"] = [
        "snerv_official_snerv_t_trained_full_tub_source_forward_parity_missing"
    ]
    checkpoint = _checkpoint_export_report(trained_mapping=False)
    binding = checkpoint["official_checkpoint_export_binding"]
    binding["official_trained_checkpoint_state_dict_slice_present"] = True
    binding["official_hfr_trained_checkpoint_weight_mapping_proven"] = True
    binding["official_mfu_receiver_activation_payload_bound"] = True
    binding["blockers"] = [
        "snerv_official_mfu_native_receiver_activation_payload_not_upstream_weight_mapping",
        "snerv_official_trained_checkpoint_state_dict_mapping_missing",
        "snerv_official_tub_trained_temporal_encoder_decoder_weights_not_loaded",
        "snerv_official_tub_portable_temporal_encoder_weight_mapping_missing",
        "snerv_official_tub_portable_output2_decoder_weight_mapping_missing",
    ]
    binding["preserved_blockers"] = list(binding["blockers"])

    report = build_snerv_official_tub_lf_hf_replacement_authority_gate(
        source_forward_artifacts=[source],
        checkpoint_export_reports=[checkpoint],
        output_root=tmp_path / "gate",
        min_free_bytes=0,
        allow_local_output=True,
        generated_utc="2026-06-05T00:00:00+00:00",
    )

    assert report["official_tub_lf_hf_decoder_replacement_ready"] is False
    assert report["full_tub_source_forward_replay_ready"] is False
    assert report["trained_checkpoint_state_dict_mapping_ready"] is False
    assert report["tub_temporal_output2_weight_mapping_ready"] is False
    blockers = set(report["queue_blockers"])
    assert (
        "snerv_official_mfu_native_receiver_activation_payload_not_upstream_weight_mapping"
        in blockers
    )
    assert "snerv_official_trained_checkpoint_state_dict_mapping_missing" in blockers
    assert "snerv_official_tub_trained_temporal_encoder_decoder_weights_not_loaded" in (
        blockers
    )
    assert "snerv_official_tub_portable_temporal_encoder_weight_mapping_missing" in (
        blockers
    )
    assert "snerv_official_tub_portable_output2_decoder_weight_mapping_missing" in (
        blockers
    )
    assert (
        "snerv_official_mfu_hfr_tub_receiver_payload_not_source_forward_authority"
        in blockers
    )
    assert "snerv_official_mfu_hfr_tub_full_stack_source_forward_replay_missing" in (
        blockers
    )
    assert (
        "snerv_official_mfu_native_receiver_activation_payload_not_upstream_weight_mapping"
        not in report["closed_campaign_blockers"]
    )


def test_authority_gate_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    source_path = tmp_path / "source.json"
    checkpoint_path = tmp_path / "checkpoint.json"
    output_root = tmp_path / "out"
    output_json = output_root / "gate.json"
    output_md = output_root / "gate.md"
    source_path.write_text(
        json.dumps(
            _source_forward_artifact(
                official_export_bound=True,
                receiver_consumes_output2=True,
                source_authority=False,
                full_tub_parity=False,
            )
        ),
        encoding="utf-8",
    )
    checkpoint_path.write_text(
        json.dumps(_checkpoint_export_report(trained_mapping=False)),
        encoding="utf-8",
    )

    rc = cli_main(
        [
            "--source-forward-artifact",
            source_path.as_posix(),
            "--checkpoint-export-report",
            checkpoint_path.as_posix(),
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
    markdown = output_md.read_text(encoding="utf-8")
    assert rc == 0
    assert payload["schema"] == SCHEMA
    assert payload["source_forward_evidence"]["source_path"] == source_path.as_posix()
    assert (
        payload["checkpoint_export_evidence"]["source_path"]
        == checkpoint_path.as_posix()
    )
    assert "SNeRV Official TUB LF/HF Replacement Authority Gate" in markdown
    assert "trained checkpoint mapping ready" in markdown
    assert "TUB source fixture replay ready" in markdown
    assert payload["score_claim"] is False


def _source_forward_artifact(
    *,
    official_export_bound: bool,
    receiver_consumes_output2: bool,
    source_authority: bool,
    full_tub_parity: bool,
) -> dict[str, object]:
    blockers = []
    if not full_tub_parity:
        blockers.append("snerv_official_snerv_t_trained_full_tub_source_forward_parity_missing")
    artifact: dict[str, object] = {
        "schema": "snerv_official_mfu_hfr_tub_forward_parity.v1",
        "generated_utc": "20260605T000000Z",
        "_source_path": "/ssd/source_forward.json",
        "_source_sha256": "a" * 64,
        "official_export_bound": official_export_bound,
        "official_checkpoint_export_binding_evidence": {
            "schema": "snerv_official_checkpoint_export_binding_evidence.v1",
            "official_export_bound": official_export_bound,
        },
        "full_tub_source_forward_parity_proven": full_tub_parity,
        "official_tub_source_forward_replay": {
            "schema": "snerv_official_tub_source_forward_replay.v1",
            "source_forward_replay_executed": True,
            "official_tub_temporal_encoder_output2_source_fixture_replay_passed": True,
            "full_tub_source_forward_parity_proven": full_tub_parity,
            "source_forward_parity_proven": full_tub_parity,
            "closed_blockers": [
                "snerv_official_tub_graph_inputs_only_not_full_source_forward_parity",
                "snerv_official_snerv_t_output2_fusion_source_forward_replay_missing",
                "snerv_official_tub_portable_output2_fusion_receiver_mapping_missing",
                "snerv_official_tub_frame_reconstruction_source_forward_replay_missing",
            ],
            "preserved_blockers": (
                []
                if full_tub_parity
                else [
                    "snerv_official_tub_trained_temporal_encoder_decoder_weights_not_loaded",
                    "snerv_official_tub_portable_temporal_encoder_weight_mapping_missing",
                    "snerv_official_tub_portable_output2_decoder_weight_mapping_missing",
                    "snerv_official_snerv_t_trained_full_tub_source_forward_parity_missing",
                ]
            ),
            "blockers": (
                []
                if full_tub_parity
                else [
                    "snerv_official_tub_trained_temporal_encoder_decoder_weights_not_loaded",
                    "snerv_official_tub_portable_temporal_encoder_weight_mapping_missing",
                    "snerv_official_tub_portable_output2_decoder_weight_mapping_missing",
                    "snerv_official_snerv_t_trained_full_tub_source_forward_parity_missing",
                ]
            ),
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
        },
        "receiver_payload_frame_replay": {
            "schema": "snerv_official_mfu_hfr_tub_receiver_payload_frame_replay.v1",
            "receiver_runtime_decode_proven": True,
            "frame_producing_official_payload_replay_proven": True,
            "receiver_frame_decode_consumes_output2": receiver_consumes_output2,
            "source_forward_replay_authority": source_authority,
            "decoded_frames_shape": [2, 3, 16, 24],
            "decoded_frames_sha256": "b" * 64,
            "payload_bytes": 13052,
            "payload_sha256": "c" * 64,
        },
        "blockers": blockers,
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }
    if source_authority and full_tub_parity:
        artifact["source_forward_replay_proof"] = _complete_source_forward_proof()
    return artifact


def _complete_source_forward_proof() -> dict[str, object]:
    return valid_snerv_source_forward_action_effect()


def _checkpoint_export_report(*, trained_mapping: bool) -> dict[str, object]:
    binding_blockers = (
        []
        if trained_mapping
        else [
            "snerv_official_trained_checkpoint_state_dict_not_loaded",
            "snerv_official_tub_trained_temporal_encoder_decoder_weights_not_loaded",
        ]
    )
    return {
        "schema": "snerv_checkpoint_archive_export.v1",
        "report_path": "/ssd/snerv_checkpoint_archive_export.json",
        "checkpoint_epoch": 3999,
        "archive_bytes": 123456,
        "archive_sha256": "d" * 64,
        "packet_bytes": 13052,
        "packet_sha256": "e" * 64,
        "official_checkpoint_export_binding": {
            "schema": "snerv_official_checkpoint_export_binding.v1",
            "selected_packet_status": "frame_producing_official_export",
            "native_checkpoint_export_bound_to_official_payload": True,
            "official_receiver_payload_bound": True,
            "official_receiver_tensor_map_verified": True,
            "official_trained_checkpoint_state_dict_slice_present": trained_mapping,
            "official_trained_checkpoint_state_dict_mapping_verified": trained_mapping,
            "official_mfu_hfr_trained_checkpoint_weight_mapping_proven": trained_mapping,
            "official_tub_temporal_encoder_weight_mapping_proven": trained_mapping,
            "official_tub_output2_decoder_weight_mapping_proven": trained_mapping,
            "official_trained_checkpoint_mapping_manifest": {
                "schema": (
                    "snerv_official_trained_checkpoint_state_dict_mapping_manifest.v1"
                ),
                "official_trained_checkpoint_loaded": trained_mapping,
                "official_mfu_hfr_trained_checkpoint_weight_mapping_proven": (
                    trained_mapping
                ),
                "official_tub_temporal_encoder_weight_mapping_proven": (
                    trained_mapping
                ),
                "official_tub_output2_decoder_weight_mapping_proven": (
                    trained_mapping
                ),
                "blockers": binding_blockers,
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            },
            "blockers": binding_blockers,
            "preserved_blockers": binding_blockers,
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
        },
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }


# ---------------------------------------------------------------------------
# C2 binding — the authority gate consumes the MFU/HFR/TUB DROP_OR_REIFY proof
# (NO-FAKE: real proof object drives the binding gate; behavioral assertions).
# ---------------------------------------------------------------------------

import pytest  # noqa: E402

from tac.analysis.snerv_official_source_forward_harness import (  # noqa: E402
    DEFAULT_OFFICIAL_SNERV_REPO,
    build_snerv_official_mfu_hfr_tub_drop_or_reify_source_forward_proof,
)
from tac.analysis.snerv_official_tub_lf_hf_replacement_authority_gate import (  # noqa: E402
    MFU_HFR_TUB_DROP_OR_REIFY_MISSING_BLOCKER,
    MFU_HFR_TUB_DROP_OR_REIFY_NO_CAUSAL_FACET_BLOCKER,
    MFU_HFR_TUB_DROP_OR_REIFY_PROOF_SCHEMA,
    MFU_HFR_TUB_DROP_OR_REIFY_SCORER_PENDING_BLOCKER,
)


def test_authority_gate_omits_binding_row_without_proof(tmp_path: Path) -> None:
    report = build_snerv_official_tub_lf_hf_replacement_authority_gate(
        output_root=tmp_path,
        allow_local_output=True,
    )
    gate_ids = [row["gate_id"] for row in report["gate_rows"]]
    # The prior DAG is preserved unchanged when no proof is supplied.
    assert "mfu_hfr_tub_drop_or_reify_binding" not in gate_ids
    evidence = report["mfu_hfr_tub_drop_or_reify_evidence"]
    assert evidence["proof_supplied"] is False
    assert evidence["binding_ready"] is False
    assert MFU_HFR_TUB_DROP_OR_REIFY_MISSING_BLOCKER in evidence["blockers"]


def test_authority_gate_binds_real_drop_or_reify_proof(tmp_path: Path) -> None:
    if not DEFAULT_OFFICIAL_SNERV_REPO.exists():
        pytest.skip(f"official SNeRV checkout is absent: {DEFAULT_OFFICIAL_SNERV_REPO}")
    proof = build_snerv_official_mfu_hfr_tub_drop_or_reify_source_forward_proof(
        official_repo_dir=DEFAULT_OFFICIAL_SNERV_REPO,
        generated_utc="20260609T000000Z",
    )
    report = build_snerv_official_tub_lf_hf_replacement_authority_gate(
        output_root=tmp_path,
        allow_local_output=True,
        mfu_hfr_tub_drop_or_reify_proofs=[proof],
    )
    gate_ids = [row["gate_id"] for row in report["gate_rows"]]
    assert "mfu_hfr_tub_drop_or_reify_binding" in gate_ids
    evidence = report["mfu_hfr_tub_drop_or_reify_evidence"]
    # The binding question is answered: the source graph ran and >=1 facet is
    # receiver-causal (REIFY/REIFY_PENDING_SCORER), so binding_ready is True.
    assert evidence["proof_supplied"] is True
    assert evidence["source_graph_executed"] is True
    assert evidence["binding_ready"] is True
    assert evidence["reify_pending_scorer_facets"], "expected causal facets"
    assert evidence["dropped_facets"], "expected metadata-only DROP facets"
    # Research-only proof => scorer ΔS still pending (the C4 blocker is surfaced).
    assert (
        MFU_HFR_TUB_DROP_OR_REIFY_SCORER_PENDING_BLOCKER in evidence["blockers"]
    )
    # The binding gate row is correctly BLOCKED on the pending scorer (binding
    # proven, score authority pending) — never falsely "ready".
    row = next(
        r for r in report["gate_rows"]
        if r["gate_id"] == "mfu_hfr_tub_drop_or_reify_binding"
    )
    assert row["blocked"] is True
    assert MFU_HFR_TUB_DROP_OR_REIFY_SCORER_PENDING_BLOCKER in row["blockers"]
    assert report["score_claim"] is False
    assert report["promotable"] is False


def test_authority_gate_binding_blocks_on_no_causal_facet() -> None:
    # A synthetic proof with NO receiver-causal facet must block the binding gate
    # with the no-causal-facet blocker (the receiver consumes nothing -> no bytes).
    synthetic = {
        "schema": MFU_HFR_TUB_DROP_OR_REIFY_PROOF_SCHEMA,
        "source_graph_executed": True,
        "real_scorer_supplied": False,
        "drop_or_reify_verdict": "DROP",
        "family_verdicts": {"mfu": "DROP", "hfr": "DROP", "tub": "DROP"},
        "reified_facets": [],
        "reify_pending_scorer_facets": [],
        "dropped_facets": ["mfu_low", "mfu_skip_mid"],
        "candidate_action_evaluations": [],
        "generated_utc": "20260609T000000Z",
    }
    from tac.analysis.snerv_official_tub_lf_hf_replacement_authority_gate import (
        _mfu_hfr_tub_drop_or_reify_state,
    )

    state = _mfu_hfr_tub_drop_or_reify_state([synthetic])
    assert state["proof_supplied"] is True
    assert state["binding_ready"] is False
    assert MFU_HFR_TUB_DROP_OR_REIFY_NO_CAUSAL_FACET_BLOCKER in state["blockers"]


def test_authority_gate_binding_ignores_wrong_schema_proofs() -> None:
    from tac.analysis.snerv_official_tub_lf_hf_replacement_authority_gate import (
        _mfu_hfr_tub_drop_or_reify_state,
    )

    state = _mfu_hfr_tub_drop_or_reify_state([{"schema": "not_the_proof", "x": 1}])
    assert state["proof_supplied"] is False
    assert MFU_HFR_TUB_DROP_OR_REIFY_MISSING_BLOCKER in state["blockers"]
