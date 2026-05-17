# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

from tac.analysis.a1_byte_escape_profile import (
    build_a1_byte_escape_profile,
    latent_lzma_sweep,
    read_a1_archive_sections,
    render_a1_byte_escape_markdown,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
A1_ARCHIVE = REPO_ROOT / "submissions/a1/archive.zip"


def test_a1_archive_sections_parse_prefixed_layout() -> None:
    sections = read_a1_archive_sections(A1_ARCHIVE)

    assert sections.archive_bytes == 178_262
    assert sections.member_name == "x"
    assert sections.member_bytes == 178_162
    assert sections.section_total == 162_168
    assert len(sections.section_header) == 4
    assert len(sections.decoder_blob) == 162_164
    assert len(sections.latent_blob) == 15_387
    assert len(sections.sidecar_blob) == 607


def test_a1_latent_lzma_sweep_does_not_reopen_generic_arithmetic_retread() -> None:
    sections = read_a1_archive_sections(A1_ARCHIVE)
    profile = latent_lzma_sweep(sections.latent_blob, dict_sizes=(4096, 8192))

    assert profile["source_filter_roundtrip_exact"] is True
    assert profile["raw_latent_bytes"] == 16_912
    assert profile["best"]["bytes"] == 15_387
    assert profile["best"]["delta_vs_source_bytes"] == 0
    assert profile["best_beats_source"] is False


def test_a1_byte_escape_profile_is_planning_only_and_sidecar_saturated() -> None:
    profile = build_a1_byte_escape_profile(A1_ARCHIVE, repo_root=REPO_ROOT)

    assert profile["authority"]["score_claim"] is False
    assert profile["authority"]["promotion_eligible"] is False
    assert profile["authority"]["ready_for_exact_eval_dispatch"] is False
    assert profile["sidecar_huff_enum"]["current_sidecar_bytes"] == 607
    assert profile["sidecar_huff_enum"]["n_valid"] == 597
    assert profile["sidecar_huff_enum"]["noop_count"] == 3
    assert profile["sidecar_huff_enum"]["choice_values_fit_u8_runtime_format"] is False
    assert profile["sidecar_huff_enum"]["runtime_min_supported_length_usable_for_current_semantics"] == 607
    assert profile["byte_escape_summary"]["best_supported_delta_bytes_without_runtime_change"] == 0
    assert profile["byte_escape_summary"]["classification"] == "saturated_byte_only_current_runtime"


def test_a1_byte_escape_markdown_and_json_are_serializable() -> None:
    profile = build_a1_byte_escape_profile(A1_ARCHIVE, repo_root=REPO_ROOT)
    text = render_a1_byte_escape_markdown(profile)

    assert "# A1 Rule #6 byte-escape profile" in text
    assert "score_claim: false" in text
    assert "Do not retread generic arithmetic" in text
    json.dumps(profile, sort_keys=True, allow_nan=False)
