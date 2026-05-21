# SPDX-License-Identifier: MIT
"""Tests for ATW2 cdf_table_blob dead-section probes."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from tac.substrates.atw_codec_v2 import (
    ATWv2Codec,
    ATWv2CodecConfig,
    ATWv2Variant,
    pack_archive,
    parse_archive,
)
from tac.substrates.atw_codec_v2.cdf_dead_section import (
    analyze_atw2_cdf_section,
    mutate_atw2_cdf_table_bytes,
    prove_atw2_cdf_decode_influence,
)

REPO_ROOT = Path(__file__).resolve().parents[5]


def _make_archive() -> bytes:
    cfg = ATWv2CodecConfig(
        variant=ATWv2Variant.B_WZ_ONLY,
        latent_dim=8,
        encoder_input_channels=3,
        encoder_hidden_dim=16,
        decoder_embed_dim=8,
        decoder_initial_grid_h=2,
        decoder_initial_grid_w=2,
        decoder_channels=(6, 4, 4, 4, 4, 4),
        decoder_num_upsample_blocks=2,
        num_pairs=4,
        output_height=16,
        output_width=24,
        scorer_class_prior_dim=8,
        wz_head_hidden_dim=8,
        g1_distill_hidden_dim=8,
    )
    torch.manual_seed(123)
    model = ATWv2Codec(cfg).eval()
    with torch.no_grad():
        model.scorer_class_prior_table.normal_(0.0, 0.2)
        model.cdf_table.copy_(
            torch.linspace(0.001, 0.999, model.cdf_table.numel()).view_as(
                model.cdf_table
            )
        )
    meta: dict[str, Any] = {
        "decoder_embed_dim": cfg.decoder_embed_dim,
        "decoder_initial_grid_h": cfg.decoder_initial_grid_h,
        "decoder_initial_grid_w": cfg.decoder_initial_grid_w,
        "decoder_channels": list(cfg.decoder_channels),
        "decoder_num_upsample_blocks": cfg.decoder_num_upsample_blocks,
        "encoder_input_channels": cfg.encoder_input_channels,
        "encoder_hidden_dim": cfg.encoder_hidden_dim,
        "output_height": cfg.output_height,
        "output_width": cfg.output_width,
        "wz_head_hidden_dim": cfg.wz_head_hidden_dim,
        "g1_distill_hidden_dim": cfg.g1_distill_hidden_dim,
        "latent_init_std": cfg.latent_init_std,
    }
    return pack_archive(
        model.encoder.state_dict(),
        model.decoder.state_dict(),
        model.wz_side_info_head.state_dict(),
        model.g1_distill_head.state_dict(),
        model.latents.detach().cpu(),
        model.scorer_class_prior_table.detach().cpu(),
        model.cdf_table.detach().cpu(),
        meta,
        variant=1,
    )


def test_analyze_atw2_cdf_section_reports_rate_only_estimate() -> None:
    archive = _make_archive()
    analysis = analyze_atw2_cdf_section(archive)
    assert analysis.cdf_classes == 5
    assert analysis.cdf_symbols == 256
    assert analysis.cdf_bytes == 2560
    assert analysis.parser_visible is True
    assert analysis.current_runtime_decode_visible is False
    assert analysis.conservative_bytes_saved == 2528
    assert analysis.conservative_delta_s_rate_only < 0
    assert analysis.score_claim is False


def test_mutate_atw2_cdf_table_changes_only_cdf_table() -> None:
    archive = _make_archive()
    analysis = analyze_atw2_cdf_section(archive)
    mutated = mutate_atw2_cdf_table_bytes(archive)
    assert len(mutated) == len(archive)
    diffs = [i for i, (a, b) in enumerate(zip(archive, mutated)) if a != b]
    assert diffs[0] == analysis.cdf_offset
    assert diffs[-1] == analysis.cdf_offset + analysis.cdf_bytes - 1
    assert len(diffs) == analysis.cdf_bytes
    assert parse_archive(mutated).cdf_table.shape == (5, 256)


def test_cdf_table_xor_preserves_current_inflate_raw_output() -> None:
    archive = _make_archive()
    proof = prove_atw2_cdf_decode_influence(archive, device="cpu")
    assert proof.raw_equal is True
    assert proof.max_abs_raw_byte_delta == 0
    assert proof.mutated_byte_count == 2560
    assert proof.source_archive_sha256 != proof.mutated_archive_sha256
    assert proof.source_raw_sha256 == proof.mutated_raw_sha256
    assert proof.score_claim is False


def test_probe_atw2_cdf_dead_section_cli_synthetic_smoke() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "probe_atw2_cdf_dead_section.py"),
            "--synthetic-smoke",
            "--device",
            "cpu",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    payload = json.loads(proc.stdout)
    assert payload["raw_equal"] is True
    assert payload["analysis"]["cdf_bytes"] == 2560
    assert payload["ready_for_exact_eval_dispatch"] is False

