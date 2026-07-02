# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tac.analysis.snerv_lf_hf_runtime_binding import (
    NATIVE_TUB_LF_HF_OUTPUT2_BINDING_SCHEMA,
    SCHEMA,
    build_snerv_lf_hf_runtime_binding_proof,
    build_snerv_native_tub_lf_hf_output2_runtime_binding,
)
from tac.substrates.snerv_inverse_steg_carrier.joint_lf_hf_codebook import (
    build_joint_lf_hf_factorized_codebook_receiver_proof,
)
from tac.substrates.snerv_inverse_steg_carrier.lf_conditioned_hf_residual import (
    build_lf_conditioned_hf_residual_receiver_proof,
)
from tac.substrates.snerv_inverse_steg_carrier.lf_latent_hyperprior import (
    build_lf_latent_hyperprior_receiver_proof,
)
from tac.substrates.snerv_inverse_steg_carrier.lf_super_resolution_tiny_anchor import (
    build_lf_super_resolution_tiny_anchor_receiver_proof,
)
from tac.substrates.snerv_inverse_steg_carrier.spectral_band_allocator import (
    build_score_tethered_spectral_band_allocator_receiver_proof,
)
from tac.substrates.snerv_inverse_steg_carrier.temporal_lf_predictor import (
    build_temporal_lf_predictor_receiver_proof,
)
from tools.build_snerv_lf_hf_runtime_binding_proof import main as cli_main


def test_runtime_binding_reopens_and_decodes_exact_payload_bytes(tmp_path: Path) -> None:
    proofs = _write_all_payload_proofs(tmp_path)

    report = build_snerv_lf_hf_runtime_binding_proof(
        proofs,
        generated_utc="2026-06-06T00:00:00+00:00",
    )

    assert report["schema"] == SCHEMA
    assert report["runtime_binding_row_count"] == 6
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    closed = set(report["closed_campaign_blockers"])
    assert (
        "snerv_lf_conditioned_hf_residual_receiver_runtime_binding_missing"
        in closed
    )
    assert (
        "snerv_joint_lf_hf_factorized_codebook_receiver_runtime_binding_missing"
        in closed
    )
    assert "snerv_temporal_lf_predictor_receiver_runtime_binding_missing" in closed
    assert "snerv_lf_super_resolution_receiver_runtime_binding_missing" in closed
    assert (
        "snerv_score_tethered_lf_hf_band_allocator_runtime_binding_missing"
        in closed
    )
    assert "snerv_lf_latent_hyperprior_runtime_binding_missing" in closed
    assert "snerv_lf_conditioned_hf_bounded_training_binding_missing" not in closed
    assert "snerv_joint_lf_hf_bounded_training_binding_missing" not in closed
    for row in report["runtime_binding_rows"]:
        assert row["runtime_binding_proven"] is True
        assert row["payload_bytes_actual"] == row["payload_bytes_expected"]
        assert row["payload_sha256_actual"] == row["payload_sha256_expected"]
        assert row["decoded_summary"]["all_finite"] is True
        assert row["decoded_summary"]["element_count"] > 0
        assert row["score_claim"] is False


def test_runtime_binding_blocks_payload_sha_mismatch(tmp_path: Path) -> None:
    proof = _write_all_payload_proofs(tmp_path)[0]
    proof["payload_sha256"] = "0" * 64

    report = build_snerv_lf_hf_runtime_binding_proof([proof])

    row = report["runtime_binding_rows"][0]
    assert row["runtime_binding_proven"] is False
    assert "snerv_lf_hf_runtime_binding_payload_sha256_mismatch" in row["blockers"]
    assert report["closed_campaign_blockers"] == []


def test_native_tub_lf_hf_output2_binding_clears_source_identical_false_authority() -> None:
    report = build_snerv_native_tub_lf_hf_output2_runtime_binding(
        {
            "schema": "compact_runner_snerv_mlx_native_export_attachment.v1",
            "packet_path": "/artifacts/native_packet.snar",
            "packet_sha256": "1" * 64,
            "archive_path": "/artifacts/archive.zip",
            "archive_sha256": "2" * 64,
            "snerv_lf_hf_solution_family": "official_tub_lf_hf_decoder_replacement",
            "snerv_official_tub_source_fixture_binding": {
                "schema": "snerv_official_tub_source_fixture_binding.v1",
                "source_fixture_replay_bound": True,
                "official_tub_temporal_encoder_output2_source_fixture_replay_passed": True,
                "source_forward_replay_authority": False,
            },
            "snerv_official_tub_source_fixture_replay_bound": True,
            "snerv_official_tub_source_fixture_replay_passed": True,
            "official_primitive_binding": {
                "official_receiver_payload_bound": True,
                "selected_packet_frame_producing_official_export": True,
                "selected_packet_receiver_payload_frame_replay_passed": True,
            },
            "selected_official_authority": {
                "frame_producing_official_export": True,
                "receiver_payload_frame_replay_passed": True,
                "official_tub_output2_storage": {
                    "section": "decoder_payload.output_2",
                    "stored": True,
                    "source_payload_present": True,
                    "receiver_executes_output2_fusion_from_payload": True,
                    "receiver_frame_decode_consumes_output2": True,
                    "receiver_output2_frame_shape_match": True,
                },
                "official_tub_output2_payload_source_available": True,
                "official_tub_output2_payload_export_bound": True,
                "official_tub_output2_payload_stored": True,
                "official_tub_output2_receiver_fusion_from_payload": True,
                "official_tub_output2_receiver_executed": True,
                "receiver_frame_decode_consumes_output2": True,
                "official_tub_output2_receiver_frame_bound": True,
                "official_tub_output2_receiver_output2_frame_shape_match": True,
            },
        },
        generated_utc="2026-06-13T00:00:00+00:00",
    )

    assert report["schema"] == NATIVE_TUB_LF_HF_OUTPUT2_BINDING_SCHEMA
    assert report["runtime_binding_proven"] is True
    assert report["output2_source_identical"] is True
    assert report["output2_boundary_verdict"]["verdict"] == "SOURCE_IDENTICAL"
    assert report["output2_boundary_verdict"]["passed"] is True
    assert report["source_forward_replay_authority"] is False
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert "migration_required_before_runner_execution" in report["closed_campaign_blockers"]


def test_native_tub_lf_hf_output2_binding_blocks_missing_receiver_frame_bound() -> None:
    report = build_snerv_native_tub_lf_hf_output2_runtime_binding(
        {
            "schema": "compact_runner_snerv_mlx_native_export_attachment.v1",
            "snerv_lf_hf_solution_family": "official_tub_lf_hf_decoder_replacement",
            "snerv_official_tub_source_fixture_replay_bound": True,
            "snerv_official_tub_source_fixture_replay_passed": True,
            "official_primitive_binding": {
                "official_receiver_payload_bound": True,
                "selected_packet_frame_producing_official_export": True,
                "selected_packet_receiver_payload_frame_replay_passed": True,
            },
        }
    )

    assert report["runtime_binding_proven"] is False
    assert "snerv_native_output2_payload_not_export_bound" in report["blockers"]
    assert report["output2_boundary_verdict"]["verdict"] == (
        "DROP_OUTPUT2_USE_MFU_HFR_TUB_BASIS"
    )
    assert report["ready_for_exact_eval_dispatch"] is False


def test_runtime_binding_cli_writes_json(tmp_path: Path) -> None:
    proof = _write_all_payload_proofs(tmp_path)[4]
    proof_path = tmp_path / "spectral_proof.json"
    proof_path.write_text(json.dumps(proof, sort_keys=True), encoding="utf-8")
    out_json = tmp_path / "runtime_binding.json"

    rc = cli_main(
        [
            "--spectral-band-allocator-receiver-payload-proof",
            proof_path.as_posix(),
            "--output-json",
            out_json.as_posix(),
        ]
    )

    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["schema"] == SCHEMA
    assert payload["runtime_binding_row_count"] == 1
    assert payload["closed_campaign_blockers"] == [
        "snerv_score_tethered_lf_hf_band_allocator_runtime_binding_missing"
    ]


def _write_all_payload_proofs(tmp_path: Path) -> list[dict[str, object]]:
    frames = (
        np.arange(1 * 2 * 3 * 16 * 24, dtype=np.float32).reshape(1, 2, 3, 16, 24)
        % 251.0
    )
    pair_indices = [0]
    builders = [
        (
            "lf_residual",
            "slhr",
            lambda payload_path: build_lf_conditioned_hf_residual_receiver_proof(
                frames,
                pair_indices=pair_indices,
                packet_path="/ssd/candidate.snar",
                source_packet_sha256="1" * 64,
                payload_path=payload_path.as_posix(),
            ),
        ),
        (
            "joint_codebook",
            "sjlc",
            lambda payload_path: build_joint_lf_hf_factorized_codebook_receiver_proof(
                frames,
                pair_indices=pair_indices,
                packet_path="/ssd/candidate.snar",
                source_packet_sha256="2" * 64,
                payload_path=payload_path.as_posix(),
            ),
        ),
        (
            "temporal_lf",
            "stlp",
            lambda payload_path: build_temporal_lf_predictor_receiver_proof(
                frames,
                pair_indices=pair_indices,
                packet_path="/ssd/candidate.snar",
                source_packet_sha256="3" * 64,
                payload_path=payload_path.as_posix(),
            ),
        ),
        (
            "tiny_anchor",
            "slsr",
            lambda payload_path: build_lf_super_resolution_tiny_anchor_receiver_proof(
                frames,
                pair_indices=pair_indices,
                packet_path="/ssd/candidate.snar",
                source_packet_sha256="4" * 64,
                payload_path=payload_path.as_posix(),
            ),
        ),
        (
            "spectral_allocator",
            "ssba",
            lambda payload_path: build_score_tethered_spectral_band_allocator_receiver_proof(
                frames,
                pair_indices=pair_indices,
                packet_path="/ssd/candidate.snar",
                source_packet_sha256="5" * 64,
                payload_path=payload_path.as_posix(),
            ),
        ),
        (
            "lf_hyperprior",
            "slhp",
            lambda payload_path: build_lf_latent_hyperprior_receiver_proof(
                frames,
                pair_indices=pair_indices,
                packet_path="/ssd/candidate.snar",
                source_packet_sha256="6" * 64,
                payload_path=payload_path.as_posix(),
            ),
        ),
    ]
    proofs: list[dict[str, object]] = []
    for name, suffix, builder in builders:
        payload_path = tmp_path / f"{name}.{suffix}"
        proof, payload = builder(payload_path)
        payload_path.write_bytes(payload)
        proof["_source_path"] = (tmp_path / f"{name}_proof.json").as_posix()
        proof["_source_sha256"] = "a" * 64
        proofs.append(proof)
    return proofs
