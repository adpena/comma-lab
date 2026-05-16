# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tac.packet_compiler.pr106_context_recode import (
    encode_adaptive_context_recode_section,
    load_pr106_context_source_from_archive,
    prove_pr106_context_archive_identity,
)

REPO = Path(__file__).resolve().parents[3]
FORMAT0C_ARCHIVE = (
    REPO
    / "experiments/results/pr106_format0c_exact_radix_candidate_20260515_codex/"
    "candidates/"
    "pr101_hdm9_hlm3_magicless_exact_radix_dim_fixed_meta_noop_rank_elided_sidecar_format_0x0c.archive.zip"
)
FORMAT0D_ARCHIVE = (
    REPO
    / "experiments/results/pr106_format0d_latent_score_table_materialized_20260515_codex/"
    "sidecar_archive.zip"
)


def test_context_recode_parses_magicless_format0c_packetir_archive() -> None:
    if not FORMAT0C_ARCHIVE.exists():
        pytest.skip("Format0C PacketIR artifact is not present")

    source = load_pr106_context_source_from_archive(FORMAT0C_ARCHIVE)
    proof = prove_pr106_context_archive_identity(archive_path=FORMAT0C_ARCHIVE)

    assert source.wrapper is not None
    assert source.wrapper["format_id"] == "0x0C"
    assert source.source["wrapper_unwrapped_for_section_context_model"] is True
    assert [section.name for section in source.sections[:3]] == [
        "packed_header_ff_len24",
        "decoder_packed_brotli",
        "latents_and_sidecar_brotli",
    ]
    assert proof["context_packet_ir_identity_passed"] is True
    assert proof["score_claim"] is False
    assert proof["ready_for_exact_eval_dispatch"] is False


def test_adaptive_context_recode_roundtrips_without_static_model_table() -> None:
    if not FORMAT0C_ARCHIVE.exists():
        pytest.skip("Format0C PacketIR artifact is not present")

    source = load_pr106_context_source_from_archive(FORMAT0C_ARCHIVE)
    section = source.section("latents_and_sidecar_brotli")

    prototype = encode_adaptive_context_recode_section(
        section.name,
        section.data,
        context_order=2,
    ).manifest()

    assert prototype["codec"] == "adaptive_section_local_context_range_prototype_v1"
    assert prototype["lossless_roundtrip_proven"] is True
    assert prototype["no_op_detector_passed"] is True
    assert prototype["range_stream_bytes"] + prototype["prefix_bytes"] == prototype[
        "integrated_section_bytes"
    ]
    assert "context_model_bytes" not in prototype
    assert prototype["score_claim"] is False
    assert prototype["ready_for_exact_eval_dispatch"] is False


def test_l5_v2_packetir_section_entropy_matrix_records_charged_prototype_floor(
    tmp_path: Path,
) -> None:
    if not FORMAT0C_ARCHIVE.exists() or not FORMAT0D_ARCHIVE.exists():
        pytest.skip("Format0C/Format0D PacketIR artifacts are not present")

    json_out = tmp_path / "matrix.json"
    md_out = tmp_path / "matrix.md"
    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools/build_l5_v2_packetir_section_entropy_matrix.py"),
            "--candidate-id",
            "format_0x0c_exact_radix",
            "--candidate-id",
            "format_0x0d_latent_score_table",
            "--build-prototypes",
            "--prototype-orders",
            "2",
            "--build-adaptive-prototypes",
            "--adaptive-orders",
            "2",
            "--json-out",
            str(json_out),
            "--md-out",
            str(md_out),
        ],
        check=True,
        text=True,
    )

    matrix = json.loads(json_out.read_text(encoding="utf-8"))
    assert matrix["schema"] == "l5_v2_packetir_section_entropy_matrix_v1"
    assert matrix["score_claim"] is False
    assert matrix["promotion_eligible"] is False
    assert matrix["ready_for_exact_eval_dispatch"] is False
    assert matrix["profiled_candidate_count"] == 2
    assert matrix["prototype_row_count"] == 4
    assert matrix["rate_positive_prototype_row_count"] == 0
    assert matrix["adaptive_prototype_row_count"] == 4
    assert matrix["rate_positive_adaptive_prototype_row_count"] == 0
    assert matrix["best_adaptive_prototype"]["delta_bytes_vs_source_section"] == 1
    assert all(
        "prototype_not_rate_positive_after_model_overhead"
        in prototype["blockers"]
        for row in matrix["rows"]
        for prototype in row["prototype_rows"]
    )
    assert all(
        "adaptive_prototype_not_rate_positive_after_integrated_overhead"
        in prototype["blockers"]
        for row in matrix["rows"]
        for prototype in row["adaptive_prototype_rows"]
    )
    assert md_out.exists()
