# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tac.analysis.snerv_lf_hf_runtime_binding import (
    SCHEMA,
    build_snerv_lf_hf_runtime_binding_proof,
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
