# SPDX-License-Identifier: MIT
"""Tests for procedural replacement surface routing."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tac.procedural_replacement_surfaces import (
    DEFAULT_SEED_BYTES,
    build_default_surface_matrix,
    build_surface_matrix_payload,
    rank_surface_matrix,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL = REPO_ROOT / "tools" / "plan_procedural_replacement_surfaces.py"


def _by_surface():
    return {(s.substrate_id, s.surface_id): s for s in build_default_surface_matrix()}


def test_surface_matrix_contains_current_cascade_candidates():
    surfaces = _by_surface()
    assert ("atw_codec_v2", "cdf_table_blob") in surfaces
    assert ("pretrained_driving_prior_dp1", "codebook_blob") in surfaces
    assert ("vq_vae", "codebook_inside_decoder_blob") in surfaces
    assert ("grayscale_lut_glv1", "chroma_lut") in surfaces
    assert ("pr101_fec6_frontier", "master_gradient_null_bytes") in surfaces


def test_atw2_cdf_table_is_cleanest_parser_visible_surface():
    surface = _by_surface()[("atw_codec_v2", "cdf_table_blob")]
    assert surface.context == "atw_v2_codec_quantizer_lut"
    assert surface.current_archive_surface is True
    assert surface.parser_visible is True
    assert surface.raw_byte_mutation_parse_safe is True
    assert surface.whole_section_replacement_surface is True
    assert surface.original_payload_bytes == 5 * 256 * 2
    assert surface.predicted_bytes_saved == 5 * 256 * 2 - DEFAULT_SEED_BYTES
    assert surface.predicted_delta_s == pytest.approx(
        -25.0 * (2560 - 32) / 37_545_489
    )


def test_dp1_is_whole_section_replacement_not_raw_byte_mutation_surface():
    surface = _by_surface()[("pretrained_driving_prior_dp1", "codebook_blob")]
    assert surface.parser_visible is True
    assert surface.raw_byte_mutation_parse_safe is False
    assert surface.whole_section_replacement_surface is True
    assert surface.requires_archive_adapter is True
    assert surface.predicted_bytes_saved == 4096 - DEFAULT_SEED_BYTES


def test_vqvae_requires_adapter_because_codebook_is_inside_decoder_blob():
    surface = _by_surface()[("vq_vae", "codebook_inside_decoder_blob")]
    assert surface.context == "intermediate_transform_quantizer"
    assert surface.current_archive_surface is False
    assert surface.parser_visible is False
    assert surface.requires_archive_adapter is True
    assert surface.original_payload_bytes == 512 * 8 * 2
    assert surface.predicted_bytes_saved == 8192 - DEFAULT_SEED_BYTES


def test_glv1_chroma_lut_is_not_current_parser_visible_surface():
    surface = _by_surface()[("grayscale_lut_glv1", "chroma_lut")]
    assert surface.context == "chroma_lut_replacement"
    assert surface.current_archive_surface is False
    assert surface.parser_visible is False
    assert surface.candidate_status == "BLOCKED_NO_CURRENT_SURFACE"
    assert "GLV2" in surface.blocker


def test_pr101_fec6_negative_control_saves_zero_bytes():
    surface = _by_surface()[("pr101_fec6_frontier", "master_gradient_null_bytes")]
    assert surface.candidate_status == "BLOCKED_PARSER_SAFE_SUBSET_EMPTY"
    assert surface.context_in_domain is False
    assert surface.predicted_bytes_saved == 0
    assert surface.predicted_delta_s == 0.0


def test_ranker_puts_ready_dp1_before_design_deferred_atw():
    ranked = rank_surface_matrix()
    assert (ranked[0].substrate_id, ranked[0].surface_id) == (
        "pretrained_driving_prior_dp1",
        "codebook_blob",
    )
    assert (ranked[1].substrate_id, ranked[1].surface_id) == (
        "atw_codec_v2",
        "cdf_table_blob",
    )


def test_payload_is_non_promotional_and_serializable():
    payload = build_surface_matrix_payload()
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["rank_or_kill_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    encoded = json.dumps(payload, sort_keys=True)
    assert "procedural_replacement_surface_matrix_v1_20260521" in encoded
    assert "atw_v2_codec_quantizer_lut" in encoded


def test_cli_json_outputs_surface_matrix():
    proc = subprocess.run(
        [sys.executable, str(TOOL), "--json"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["schema_version"] == "procedural_replacement_surface_matrix_v1_20260521"
    assert len(payload["surfaces"]) >= 6
    assert payload["surfaces"][0]["substrate_id"] == "pretrained_driving_prior_dp1"


def test_cli_writes_json_and_markdown(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(TOOL), "--output-dir", str(tmp_path)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert (tmp_path / "surface_matrix.json").is_file()
    assert (tmp_path / "surface_matrix.md").is_file()
    assert "surface_matrix.json" in proc.stdout
    md = (tmp_path / "surface_matrix.md").read_text()
    assert "score_claim=false" in md
    assert "pretrained_driving_prior_dp1" in md
    assert "BLOCKED_PARSER_SAFE_SUBSET_EMPTY" in md
