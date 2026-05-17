# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile, ZipInfo

import pytest
import torch

from tac.analysis.z3v2_payload_profile import (
    Z3V2PayloadProfileError,
    profile_z3v2_archive,
    render_markdown,
    write_profile_outputs,
)
from tac.substrates.z3_balle_hyperprior_bolton.archive_v2 import (
    A1_DECODER_SECTION_TOTAL,
    A1_LATENT_BLOB_LEN,
    A1_LATENT_DIM,
    A1_N_PAIRS,
    build_z3v2_payload_bytes,
    encode_z3hv2_section,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _write_archive(path: Path, member: str, payload: bytes) -> None:
    info = ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_STORED
    info.external_attr = 0o644 << 16
    with ZipFile(path, "w") as zf:
        zf.writestr(info, payload)


def _synthetic_a1_payload() -> bytes:
    return (
        struct.pack("<I", A1_DECODER_SECTION_TOTAL)
        + bytes([7]) * (A1_DECODER_SECTION_TOTAL - 4)
        + bytes([11]) * A1_LATENT_BLOB_LEN
        + b"synthetic-sidecar"
    )


def _direct_z3_payload() -> bytes:
    section = encode_z3hv2_section(
        hyperprior_weights_int8=b"",
        w_hat_int8=b"",
        residual_int8=bytes([3]) * (A1_N_PAIRS * A1_LATENT_DIM),
        latent_offset=torch.zeros(A1_LATENT_DIM),
        latent_scale=torch.ones(A1_LATENT_DIM),
        hyper_dim=8,
        int8_w_scale=1.0,
        quant_step=1.0,
        min_sigma=1.0e-3,
        max_sigma=32.0,
        factorized_half_range=7.0,
    )
    return build_z3v2_payload_bytes(a1_bytes=_synthetic_a1_payload(), z3hv2_section=section)


def test_profile_z3v2_archive_classifies_direct_residual_control(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    _write_archive(archive, "0.bin", _direct_z3_payload())

    profile = profile_z3v2_archive(archive)

    assert profile["schema"] == "z3v2_payload_profile_v1"
    assert profile["classification"] == "direct_residual_control"
    assert profile["direct_residual_control"] is True
    assert profile["weights_int8_bytes"] == 0
    assert profile["w_hat_int8_bytes"] == 0
    assert profile["residual_int8_bytes"] == A1_N_PAIRS * A1_LATENT_DIM
    assert profile["residual_coding"] == "brotli_direct_int8_residual"
    assert profile["balle_entropy_residual_decoder_active"] is False
    assert profile["score_claim"] is False
    assert profile["promotion_eligible"] is False
    assert profile["ready_for_exact_eval_dispatch"] is False
    assert "hyperprior_weights_and_w_hat_slots_empty" in profile["result_review_blockers"]
    assert "current_z3hv2_runtime_has_no_active_balle_entropy_residual_decoder" in profile[
        "result_review_blockers"
    ]


def test_profile_z3v2_archive_classifies_sideinfo_without_entropy_decoder(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive.zip"
    section = encode_z3hv2_section(
        hyperprior_weights_int8=bytes([1, 2, 3, 4]),
        w_hat_int8=bytes([1]) * (A1_N_PAIRS * 2),
        residual_int8=bytes([3]) * (A1_N_PAIRS * A1_LATENT_DIM),
        latent_offset=torch.zeros(A1_LATENT_DIM),
        latent_scale=torch.ones(A1_LATENT_DIM),
        hyper_dim=2,
        int8_w_scale=1.0,
        quant_step=1.0,
        min_sigma=1.0e-3,
        max_sigma=32.0,
        factorized_half_range=7.0,
    )
    payload = build_z3v2_payload_bytes(a1_bytes=_synthetic_a1_payload(), z3hv2_section=section)
    _write_archive(archive, "0.bin", payload)

    profile = profile_z3v2_archive(archive)

    assert profile["classification"] == (
        "hyperprior_sideinfo_present_but_residual_still_direct_brotli_int8"
    )
    assert profile["direct_residual_control"] is False
    assert profile["weights_int8_bytes"] == 4
    assert profile["w_hat_int8_bytes"] == A1_N_PAIRS * 2
    assert profile["balle_entropy_residual_decoder_active"] is False
    assert "hyperprior_weights_and_w_hat_slots_empty" not in profile[
        "result_review_blockers"
    ]


def test_profile_z3v2_archive_rejects_non_z3_payload(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    _write_archive(archive, "x", _synthetic_a1_payload())

    with pytest.raises(Z3V2PayloadProfileError, match="not a valid Z3HV2 payload"):
        profile_z3v2_archive(archive)


def test_profile_z3v2_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    json_out = tmp_path / "profile.json"
    md_out = tmp_path / "profile.md"
    _write_archive(archive, "0.bin", _direct_z3_payload())

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools/profile_z3v2_payload_contract.py"),
            "--archive",
            str(archive),
            "--output-json",
            str(json_out),
            "--output-md",
            str(md_out),
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    payload = json.loads(json_out.read_text())
    assert payload["classification"] == "direct_residual_control"
    assert "Z3HV2 Payload Authority Profile" in md_out.read_text()


def test_render_and_write_profile_outputs(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    json_out = tmp_path / "profile.json"
    md_out = tmp_path / "profile.md"
    _write_archive(archive, "0.bin", _direct_z3_payload())
    profile = profile_z3v2_archive(archive)

    assert "direct_residual_control" in render_markdown(profile)
    write_profile_outputs(profile, json_out=json_out, markdown_out=md_out)

    assert json.loads(json_out.read_text())["schema"] == "z3v2_payload_profile_v1"
    assert "score_claim" in md_out.read_text()
