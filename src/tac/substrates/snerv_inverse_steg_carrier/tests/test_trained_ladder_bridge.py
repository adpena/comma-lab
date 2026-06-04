# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from tac.substrates.snerv_inverse_steg_carrier.carrier import (
    SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER,
    SNERV_SPECTRA_PRESERVING_ADAPTER,
)
from tac.substrates.snerv_inverse_steg_carrier.trained_ladder_bridge import (
    build_snerv_trained_ladder_row_from_advisory,
)


def test_bridge_emits_strict_false_authority_row_from_real_packet(
    tmp_path: Path,
) -> None:
    packet = b"SNAR1-real-packet"
    packet_path = tmp_path / "candidate.snar"
    packet_path.write_bytes(packet)
    advisory = _advisory(packet)

    payload = build_snerv_trained_ladder_row_from_advisory(
        advisory_result=advisory,
        archive_path=packet_path,
        archive_path_kind="receiver_snar_packet",
        target_bits_per_coeff=2.5,
        repo_root=tmp_path,
    )

    row = payload["rows"][0]
    assert payload["status"] == "trained_ladder_row_blocked"
    assert payload["archive_custody"]["computed_from_file"] is True
    assert row["archive_bytes"] == len(packet)
    assert row["archive_sha256"] == hashlib.sha256(packet).hexdigest()
    assert row["fc_dim"] == 12
    assert row["official_controls"]["fc_dim"] == 12
    assert row["official_controls"]["emb_size"] == 4
    assert row["lf_payload_codec"] == "portfolio_auto"
    assert row["receiver_archive_replay_verified"] is True
    assert row["receiver_proof_identity_bound"] is False
    assert row["byte_closed_receiver_proof"] is False
    assert row["score_claim"] is False
    assert row["ready_for_exact_eval_dispatch"] is False
    assert "receiver_proof_identity_missing" in payload["blockers"]
    assert "sample_pair_count_below_full600" in payload["blockers"]
    assert "required_emission_field_missing:official_controls.--modelsize" in payload[
        "blockers"
    ]
    assert (
        "required_emission_field_false:official_controls.source_faithful_stack"
        in payload["blockers"]
    )
    assert "required_emission_field_false:official_controls.mfu_enabled" in payload[
        "blockers"
    ]
    assert "required_emission_field_false:official_controls.hfr_enabled" in payload[
        "blockers"
    ]
    assert row["official_controls"]["source_faithful_stack"] is False
    assert row["official_controls"]["official_parity_status"] == (
        "blocked_official_mfu_hfr_not_implemented"
    )
    local_modelsize = row["official_controls"]["local_modelsize_analogue"]
    assert local_modelsize["schema"] == "snerv_local_modelsize_analogue.v1"
    assert local_modelsize["fc_dim"] == 12
    assert local_modelsize["emb_size"] == 4
    assert local_modelsize["adapter"] == "snerv_inverse_steg_principled_fork"
    assert local_modelsize["decoder_feature_count"] == 16
    assert local_modelsize["official_modelsize_authority"] is False
    assert "required_emission_field_missing:official_controls.fc_dim" not in payload[
        "blockers"
    ]
    assert "modelsize_or_fc_dim_missing" not in payload["blockers"]
    assert "required_emission_field_missing:qat_bits" in payload["blockers"]
    assert "required_emission_field_missing:lf_payload_codec" not in payload[
        "blockers"
    ]


def test_bridge_preserves_file_backed_receiver_proof_identity(
    tmp_path: Path,
) -> None:
    packet = b"SNAR1-file-backed-proof"
    packet_path = tmp_path / "candidate.snar"
    packet_path.write_bytes(packet)
    proof_path = tmp_path / "receiver_proof.json"
    proof_path.write_text(
        (
            '{"schema":"snerv_inverse_steg_generated_receiver_proof.v1",'
            '"receiver_contract_satisfied":true,'
            '"runtime_consumption_proof_ready":true,'
            '"runtime_consumption_proof_passed":true,'
            f'"archive_bytes":{packet_path.stat().st_size},'
            f'"archive_sha256":"{hashlib.sha256(packet).hexdigest()}",'
            '"receiver_output_bytes":123,'
            '"expected_receiver_output_bytes":123,'
            '"blockers":[]}\n'
        ),
        encoding="utf-8",
    )

    payload = build_snerv_trained_ladder_row_from_advisory(
        advisory_result=_advisory(packet),
        archive_path=packet_path,
        archive_path_kind="receiver_snar_packet",
        receiver_proof={
            "schema": "snerv_inverse_steg_generated_receiver_proof.v1",
            "receiver_archive_replay_verified": True,
            "receiver_contract_satisfied": True,
            "runtime_consumption_proof_ready": True,
            "receiver_proof_path": proof_path.as_posix(),
            "receiver_proof_sha256": hashlib.sha256(
                proof_path.read_bytes()
            ).hexdigest(),
        },
        target_bits_per_coeff=2.5,
        repo_root=tmp_path,
    )

    row = payload["rows"][0]
    assert row["receiver_archive_replay_verified"] is True
    assert row["receiver_proof_identity_bound"] is True
    assert row["byte_closed_receiver_proof"] is True
    assert "receiver_proof_identity_missing" not in payload["blockers"]
    assert "sample_pair_count_below_full600" in payload["blockers"]


def test_bridge_preserves_actual_lf_payload_codec(tmp_path: Path) -> None:
    packet = b"SNAR1-legacy-lf-codec-packet"
    packet_path = tmp_path / "legacy_codec.snar"
    packet_path.write_bytes(packet)

    payload = build_snerv_trained_ladder_row_from_advisory(
        advisory_result=_advisory(packet, lf_payload_codec="legacy"),
        archive_path=packet_path,
        archive_path_kind="receiver_snar_packet",
        target_bits_per_coeff=2.5,
        repo_root=tmp_path,
    )

    row = payload["rows"][0]
    assert row["lf_payload_codec"] == "legacy"
    assert row["lf_payload_codec"] != "snerv_lf_quant_payload.v1"
    assert "required_emission_field_missing:lf_payload_codec" not in payload[
        "blockers"
    ]


def test_bridge_blocks_missing_lf_payload_codec_instead_of_faking_v1(
    tmp_path: Path,
) -> None:
    packet = b"SNAR1-missing-lf-codec-packet"
    packet_path = tmp_path / "missing_codec.snar"
    packet_path.write_bytes(packet)

    payload = build_snerv_trained_ladder_row_from_advisory(
        advisory_result=_advisory(packet, lf_payload_codec=None),
        archive_path=packet_path,
        archive_path_kind="receiver_snar_packet",
        target_bits_per_coeff=2.5,
        repo_root=tmp_path,
    )

    row = payload["rows"][0]
    assert row["lf_payload_codec"] is None
    assert "required_emission_field_missing:lf_payload_codec" in payload[
        "blockers"
    ]


def test_bridge_rejects_unknown_archive_path_kind(tmp_path: Path) -> None:
    packet_path = tmp_path / "candidate.snar"
    packet_path.write_bytes(b"packet")

    with pytest.raises(ValueError, match="archive_path_kind"):
        build_snerv_trained_ladder_row_from_advisory(
            advisory_result=_advisory(b"packet"),
            archive_path=packet_path,
            archive_path_kind="unknown",
            repo_root=tmp_path,
        )


def test_bridge_records_receiver_visible_mfu_hfr_adapter_controls(
    tmp_path: Path,
) -> None:
    packet = b"SNAR1-adapter-packet"
    packet_path = tmp_path / "adapter.snar"
    packet_path.write_bytes(packet)

    payload = build_snerv_trained_ladder_row_from_advisory(
        advisory_result=_advisory(
            packet,
            snerv_model_size_adapter=SNERV_SPECTRA_PRESERVING_ADAPTER,
            snerv_mfu_scales=(1, 3),
            snerv_hfr_gain=0.25,
        ),
        archive_path=packet_path,
        archive_path_kind="receiver_snar_packet",
        target_bits_per_coeff=2.5,
        repo_root=tmp_path,
    )

    controls = payload["rows"][0]["official_controls"]
    assert controls["source_faithful_stack"] is False
    assert controls["adapter"] == SNERV_SPECTRA_PRESERVING_ADAPTER
    assert controls["mfu_enabled"] is True
    assert controls["hfr_enabled"] is True
    assert controls["snerv_t_enabled"] is False
    assert controls["mfu_scales"] == [1, 3]
    assert controls["hfr_gain"] == 0.25
    assert controls["official_parity_status"] == (
        "receiver_safe_mfu_hfr_temporal_adapter_present__official_oss_"
        "parity_still_required"
    )
    assert "required_emission_field_false:official_controls.mfu_enabled" not in payload[
        "blockers"
    ]
    assert "required_emission_field_false:official_controls.hfr_enabled" not in payload[
        "blockers"
    ]


def test_bridge_records_official_mfu_hfr_tub_primitives_fail_closed_until_export_bound(
    tmp_path: Path,
) -> None:
    packet = b"SNAR1-official-primitives-packet"
    packet_path = tmp_path / "official_primitives.snar"
    packet_path.write_bytes(packet)

    payload = build_snerv_trained_ladder_row_from_advisory(
        advisory_result=_advisory(
            packet,
            snerv_model_size_adapter=SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER,
            snerv_mfu_scales=(1, 2, 4),
            snerv_hfr_gain=0.125,
            snerv_temporal_context=1,
            snerv_temporal_mode="official_haar_dwt1d_lowpass",
        ),
        archive_path=packet_path,
        archive_path_kind="receiver_snar_packet",
        target_bits_per_coeff=2.5,
        repo_root=tmp_path,
    )

    controls = payload["rows"][0]["official_controls"]
    assert controls["source_faithful_stack"] is False
    assert controls["adapter"] == SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER
    assert controls["mfu_enabled"] is True
    assert controls["receiver_safe_mfu_adapter_present"] is False
    assert controls["official_mfu_hfr_tub_numeric_primitives_requested"] is True
    assert controls["official_mfu_hfr_tub_primitives_present"] is True
    assert controls["official_mfu_hfr_tub_export_bound"] is False
    assert controls["official_mfu_hfr_tub_receiver_payload_bound"] is False
    assert controls["official_mfu_hfr_tub_source_forward_replay_bound"] is False
    assert controls["official_mfu_hfr_tub_source_forward_replay_authority"] is False
    assert controls["official_parity_status"] == (
        "official_mfu_hfr_tub_numeric_primitives_present__receiver_export_"
        "and_source_forward_replay_required"
    )
    assert controls["official_mfu_hfr_tub_export_blockers"] == [
        "snerv_official_mfu_hfr_tub_native_mlx_export_not_bound_to_official_payload",
        "snerv_official_mfu_hfr_tub_weight_mapping_missing",
        "snerv_official_mfu_hfr_tub_source_forward_replay_missing",
    ]
    assert controls["hfr_enabled"] is True
    assert controls["snerv_t_enabled"] is True
    assert controls["snerv_temporal_mode"] == "official_haar_dwt1d_lowpass"
    assert "required_emission_field_false:official_controls.mfu_enabled" not in payload[
        "blockers"
    ]
    assert "required_emission_field_false:official_controls.hfr_enabled" not in payload[
        "blockers"
    ]
    assert (
        "required_emission_field_false:official_controls.source_faithful_stack"
        in payload["blockers"]
    )


def test_bridge_refuses_top_level_official_export_bound_without_binding(
    tmp_path: Path,
) -> None:
    packet = b"SNAR1-top-level-official-bound-only"
    packet_path = tmp_path / "top_level_bound_only.snar"
    packet_path.write_bytes(packet)

    payload = build_snerv_trained_ladder_row_from_advisory(
        advisory_result=_advisory(
            packet,
            snerv_model_size_adapter=SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER,
            snerv_mfu_scales=(1, 2, 4),
            snerv_hfr_gain=0.125,
            snerv_temporal_context=1,
            snerv_temporal_mode="official_haar_dwt1d_lowpass",
            snerv_official_mfu_hfr_tub_export_bound=True,
        ),
        archive_path=packet_path,
        archive_path_kind="receiver_snar_packet",
        target_bits_per_coeff=2.5,
        repo_root=tmp_path,
    )

    controls = payload["rows"][0]["official_controls"]
    assert controls["official_mfu_hfr_tub_export_bound"] is False
    assert controls["official_mfu_hfr_tub_receiver_payload_bound"] is False
    assert controls["official_mfu_hfr_tub_source_forward_replay_bound"] is False
    assert controls["official_receiver_tensor_map_verified"] is False
    assert controls["official_receiver_tensor_map_custody"]["blockers"] == [
        "snerv_official_receiver_tensor_map_binding_missing"
    ]
    assert controls["official_mfu_hfr_tub_export_blockers"] == [
        "snerv_official_mfu_hfr_tub_native_mlx_export_not_bound_to_official_payload",
        "snerv_official_mfu_hfr_tub_weight_mapping_missing",
        "snerv_official_mfu_hfr_tub_source_forward_replay_missing",
    ]
    assert controls["source_faithful_stack"] is False
    assert payload["rows"][0]["score_claim"] is False
    assert payload["rows"][0]["ready_for_exact_eval_dispatch"] is False


def test_bridge_consumes_receiver_bound_official_payload_evidence(
    tmp_path: Path,
) -> None:
    packet = b"SNAR1-official-bound-packet"
    packet_path = tmp_path / "official_bound.snar"
    packet_path.write_bytes(packet)

    payload = build_snerv_trained_ladder_row_from_advisory(
        advisory_result=_advisory(
            packet,
            snerv_model_size_adapter=SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER,
            snerv_mfu_scales=(1, 2, 4),
            snerv_hfr_gain=0.125,
            snerv_temporal_context=1,
            snerv_temporal_mode="official_haar_dwt1d_lowpass",
            snerv_official_mfu_hfr_tub_export_bound=True,
            snerv_official_mfu_hfr_tub_frame_producing_export=True,
            official_primitive_binding={
                "official_export_bound": True,
                "export_bound_to_receiver_packet": True,
                "official_receiver_payload_contract_emitted": True,
                "receiver_runtime_decode_authority": True,
                "official_receiver_tensor_map": _official_tensor_map(),
            },
            blockers=[
                "snerv_official_mfu_hfr_tub_weight_mapping_missing",
                "snerv_official_mfu_hfr_tub_source_forward_replay_missing",
            ],
        ),
        archive_path=packet_path,
        archive_path_kind="receiver_snar_packet",
        target_bits_per_coeff=2.5,
        repo_root=tmp_path,
    )

    controls = payload["rows"][0]["official_controls"]
    assert controls["official_mfu_hfr_tub_export_bound"] is True
    assert controls["official_mfu_hfr_tub_export_bound_semantics"] == (
        "receiver_payload_bound_not_source_forward_parity"
    )
    assert controls["official_mfu_hfr_tub_receiver_payload_bound"] is True
    assert controls["official_mfu_hfr_tub_source_forward_replay_bound"] is False
    assert controls["official_mfu_hfr_tub_source_forward_replay_authority"] is False
    assert controls["official_parity_status"] == (
        "official_mfu_hfr_tub_receiver_payload_bound__source_forward_"
        "replay_required"
    )
    assert (
        "snerv_official_mfu_hfr_tub_native_mlx_export_not_bound_to_official_payload"
        not in controls["official_mfu_hfr_tub_export_blockers"]
    )
    assert controls["official_mfu_hfr_tub_export_blockers"] == [
        "snerv_official_mfu_hfr_tub_source_forward_replay_missing",
    ]
    assert controls["official_receiver_tensor_map_verified"] is True
    custody = controls["official_receiver_tensor_map_custody"]
    assert custody["receiver_tensor_map_verified"] is True
    assert custody["row_count"] == 2
    assert custody["total_tensor_bytes"] == 28
    assert custody["tensor_manifest_sha256"] == "b" * 64
    assert custody["blockers"] == []
    assert controls["source_faithful_stack"] is False
    assert payload["rows"][0]["score_claim"] is False
    assert payload["rows"][0]["ready_for_exact_eval_dispatch"] is False


def test_bridge_refuses_boolean_only_official_receiver_tensor_map(
    tmp_path: Path,
) -> None:
    packet = b"SNAR1-official-forged-tensor-map"
    packet_path = tmp_path / "official_forged_tensor_map.snar"
    packet_path.write_bytes(packet)

    payload = build_snerv_trained_ladder_row_from_advisory(
        advisory_result=_advisory(
            packet,
            snerv_model_size_adapter=SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER,
            snerv_mfu_scales=(1, 2, 4),
            snerv_hfr_gain=0.125,
            snerv_temporal_context=1,
            snerv_temporal_mode="official_haar_dwt1d_lowpass",
            snerv_official_mfu_hfr_tub_export_bound=True,
            official_primitive_binding={
                "official_export_bound": True,
                "export_bound_to_receiver_packet": True,
                "official_receiver_payload_contract_emitted": True,
                "receiver_runtime_decode_authority": True,
                "official_receiver_tensor_map": {
                    "receiver_tensor_map_verified": True,
                },
            },
            blockers=[
                "snerv_official_mfu_hfr_tub_weight_mapping_missing",
                "snerv_official_mfu_hfr_tub_source_forward_replay_missing",
            ],
        ),
        archive_path=packet_path,
        archive_path_kind="receiver_snar_packet",
        target_bits_per_coeff=2.5,
        repo_root=tmp_path,
    )

    controls = payload["rows"][0]["official_controls"]
    assert controls["official_mfu_hfr_tub_export_bound"] is True
    assert controls["official_receiver_tensor_map_verified"] is False
    custody = controls["official_receiver_tensor_map_custody"]
    assert custody["receiver_tensor_map_verified"] is False
    assert "snerv_official_receiver_tensor_map_rows_missing" in custody["blockers"]
    assert (
        "snerv_official_receiver_tensor_map_manifest_sha_missing"
        in custody["blockers"]
    )
    assert "snerv_official_mfu_hfr_tub_weight_mapping_missing" in controls[
        "official_mfu_hfr_tub_export_blockers"
    ]
    assert (
        "snerv_official_mfu_hfr_tub_native_mlx_export_not_bound_to_official_payload"
        not in controls["official_mfu_hfr_tub_export_blockers"]
    )
    assert controls["source_faithful_stack"] is False
    assert payload["rows"][0]["score_claim"] is False
    assert payload["rows"][0]["ready_for_exact_eval_dispatch"] is False


def test_bridge_records_official_modelsize_solution_without_score_authority(
    tmp_path: Path,
) -> None:
    packet = b"SNAR1-official-modelsize"
    packet_path = tmp_path / "official_modelsize.snar"
    packet_path.write_bytes(packet)
    official_solution = {
        "schema": "official_snerv_modelsize_to_fc_dim.v1",
        "source": "official_snerv_train_snerv_modelsize_quadratic_fc_dim_resolver_bound",
        "modelsize_mparams": 0.05,
        "full_data_length": 4,
        "final_size": 384 * 512,
        "enc_strds": [5, 4, 2, 2, 2],
        "dec_strds": [5, 4, 2, 2, 2],
        "fc_dim": 12,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }

    payload = build_snerv_trained_ladder_row_from_advisory(
        advisory_result=_advisory(
            packet,
            official_modelsize_solution=official_solution,
        ),
        archive_path=packet_path,
        archive_path_kind="receiver_snar_packet",
        target_bits_per_coeff=2.5,
        repo_root=tmp_path,
    )

    row = payload["rows"][0]
    controls = row["official_controls"]
    assert row["modelsize_mparams"] == 0.05
    assert controls["--modelsize"] == 0.05
    assert controls["official_modelsize_solution"]["fc_dim"] == 12
    assert controls["source_bound_modelsize_control"]["fc_dim"] == 12
    assert controls["source_bound_modelsize_control"]["score_claim"] is False
    assert controls["local_modelsize_analogue"]["official_modelsize_authority"] is True
    assert (
        "required_emission_field_missing:official_controls.--modelsize"
        not in payload["blockers"]
    )
    assert row["score_claim"] is False
    assert row["ready_for_exact_eval_dispatch"] is False


def test_bridge_refuses_official_controls_override_of_source_faithful_stack_without_parity_proof(
    tmp_path: Path,
) -> None:
    packet = b"SNAR1-override-guard"
    packet_path = tmp_path / "override_guard.snar"
    packet_path.write_bytes(packet)

    payload = build_snerv_trained_ladder_row_from_advisory(
        advisory_result=_advisory(packet),
        archive_path=packet_path,
        archive_path_kind="receiver_snar_packet",
        target_bits_per_coeff=2.5,
        official_controls={
            "source_faithful_stack": True,
            "safe_extra_control": "kept",
        },
        repo_root=tmp_path,
    )

    controls = payload["rows"][0]["official_controls"]
    assert controls["safe_extra_control"] == "kept"
    assert controls["source_faithful_stack"] is False
    guard = controls["official_control_override_guard"]
    assert guard["schema"] == "snerv_official_control_override_guard.v1"
    assert guard["ignored_overrides"]["source_faithful_stack"] is True
    assert guard["score_claim"] is False
    assert (
        "required_emission_field_false:official_controls.source_faithful_stack"
        in payload["blockers"]
    )


def test_bridge_keeps_official_mfu_hfr_tub_parity_status_fail_closed_when_controls_are_supplied(
    tmp_path: Path,
) -> None:
    packet = b"SNAR1-parity-guard"
    packet_path = tmp_path / "parity_guard.snar"
    packet_path.write_bytes(packet)

    payload = build_snerv_trained_ladder_row_from_advisory(
        advisory_result=_advisory(
            packet,
            snerv_model_size_adapter=SNERV_SPECTRA_PRESERVING_ADAPTER,
            snerv_hfr_gain=0.25,
        ),
        archive_path=packet_path,
        archive_path_kind="receiver_snar_packet",
        target_bits_per_coeff=2.5,
        official_controls={
            "official_parity_status": "official_mfu_hfr_tub_parity_proven",
            "official_mfu_hfr_tub_numeric_primitives_requested": True,
            "official_mfu_hfr_tub_primitives_present": True,
            "official_mfu_hfr_tub_export_bound": True,
            "official_mfu_hfr_tub_export_blockers": [],
        },
        repo_root=tmp_path,
    )

    controls = payload["rows"][0]["official_controls"]
    assert controls["official_parity_status"] == (
        "receiver_safe_mfu_hfr_temporal_adapter_present__official_oss_"
        "parity_still_required"
    )
    assert controls["official_control_override_guard"]["ignored_overrides"] == {
        "official_mfu_hfr_tub_export_blockers": [],
        "official_mfu_hfr_tub_export_bound": True,
        "official_mfu_hfr_tub_numeric_primitives_requested": True,
        "official_mfu_hfr_tub_primitives_present": True,
        "official_parity_status": "official_mfu_hfr_tub_parity_proven",
    }
    assert controls["official_mfu_hfr_tub_numeric_primitives_requested"] is False
    assert controls["official_mfu_hfr_tub_primitives_present"] is False
    assert controls["official_mfu_hfr_tub_export_bound"] is False
    assert controls["official_mfu_hfr_tub_export_blockers"] == []
    assert controls["official_control_override_guard"]["ready_for_exact_eval_dispatch"] is False


def _advisory(packet: bytes, **overrides: object) -> SimpleNamespace:
    values = {
        "snerv_model_size_adapter": "snerv_inverse_steg_principled_fork",
        "snerv_mfu_scales": (),
        "snerv_hfr_gain": 0.0,
        "snerv_temporal_context": 0,
        "lf_payload_codec": "portfolio_auto",
    }
    values.update(overrides)
    return SimpleNamespace(
        n_pairs=2,
        levels=1,
        wavelet="haar",
        snerv_fc_dim=12,
        snerv_emb_size=4,
        snerv_patch_radius=1,
        decoder_feature_count=16,
        receiver_archive_packet_bytes=len(packet),
        receiver_archive_sha256=hashlib.sha256(packet).hexdigest(),
        receiver_archive_replay_verified=True,
        decoder_payload_codec="mixed_magnitude_symmetric",
        linf_steps_payload_codec="snerv_step_map_coder.v1",
        hf_decoder_fit_mode="least_squares",
        hf_decoder_saliency_component="combined",
        d_seg_mean_linf=0.01,
        d_pose_mean_linf=0.04,
        score_linf=2.0,
        axis_tag="[macOS-CPU advisory]",
        **values,
    )


def _official_tensor_map() -> dict[str, object]:
    return {
        "schema": "snerv_official_mfu_hfr_tub_receiver_tensor_map.v1",
        "receiver_tensor_map_verified": True,
        "official_decoder_payload_selected": True,
        "row_count": 2,
        "total_tensor_bytes": 28,
        "category_counts": {"hfr": 1, "mfu": 1},
        "category_bytes": {"hfr": 12, "mfu": 16},
        "tensor_manifest_sha256": "b" * 64,
        "rows": [
            {
                "name": "mfu.upsample_mid.weight",
                "category": "mfu",
                "shape": [1, 1, 2, 2],
                "dtype": "float32",
                "bytes": 16,
                "manifest_byte_key": "nbytes",
                "sha256": "a" * 64,
            },
            {
                "name": "hfr.lh.conv1.weight",
                "category": "hfr",
                "shape": [3, 1, 1, 1],
                "dtype": "float32",
                "bytes": 12,
                "manifest_byte_key": "nbytes",
                "sha256": "c" * 64,
            },
        ],
        "blockers": [],
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
