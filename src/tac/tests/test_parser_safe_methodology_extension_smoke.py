# SPDX-License-Identifier: MIT
"""Tests for WAVE-3 PARSER-SAFE METHODOLOGY EXTENSION smoke tool.

Sister of ``src/tac/tests/test_parser_safe_subset_smoke.py``; same
methodology applied to 4 IN-DOMAIN substrates (DP1 + VQ-VAE +
grayscale_lut + ATW V2). Per CLAUDE.md "MPS auth eval is NOISE" +
Catalog #192: all artifacts are non-promotable macOS-CPU advisory.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


def _load_module():
    """Load the smoke tool module by file path (it lives under tools/)."""
    spec = importlib.util.spec_from_file_location(
        "_parser_safe_methodology_extension_smoke",
        REPO_ROOT / "tools" / "run_parser_safe_methodology_extension_smoke.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # required for dataclass.__module__ resolution
    spec.loader.exec_module(mod)
    return mod


# Module-level load (mirrors sister test pattern; avoids per-test
# re-execution and resolves dataclass.__module__ during construction).
_SMOKE = _load_module()


@pytest.fixture(scope="module")
def smoke_mod():
    return _SMOKE


def test_canonical_constants_present(smoke_mod):
    """Canonical kind taxonomy constants are present + immutable."""
    assert smoke_mod.KIND_STRUCT_FIELD == "struct_field"
    assert smoke_mod.KIND_BROTLI_STREAM == "brotli_stream"
    assert smoke_mod.KIND_RAW_BYTE_SECTION == "raw_byte_section"
    assert smoke_mod.KIND_JSON_METADATA == "json_metadata"
    assert smoke_mod.SCORE_RELEVANCE_SCORE_AFFECTING == "score_affecting"
    assert smoke_mod.SCORE_RELEVANCE_SCORE_OPAQUE == "score_opaque"
    assert smoke_mod.SCORE_RELEVANCE_UNKNOWN == "unknown"
    # Parser-safe = RAW only; everything else parser-essential.
    assert smoke_mod.KIND_RAW_BYTE_SECTION in smoke_mod.PARSER_SAFE_KINDS
    assert smoke_mod.KIND_BROTLI_STREAM in smoke_mod.PARSER_ESSENTIAL_KINDS
    assert smoke_mod.KIND_STRUCT_FIELD in smoke_mod.PARSER_ESSENTIAL_KINDS
    assert smoke_mod.KIND_JSON_METADATA in smoke_mod.PARSER_ESSENTIAL_KINDS


def test_synthesize_dp1_archive_valid(smoke_mod):
    """DP1 synthesizer produces a valid archive that re-parses."""
    blob, sha = smoke_mod._synthesize_dp1_archive()
    assert isinstance(blob, bytes)
    assert len(blob) > 28  # at least header
    assert blob[:4] == b"DP1\x00"
    assert len(sha) == 64


def test_synthesize_vqv1_archive_valid(smoke_mod):
    """VQV1 synthesizer produces a valid archive."""
    blob, sha = smoke_mod._synthesize_vqv1_archive()
    assert isinstance(blob, bytes)
    assert blob[:4] == b"VQV1"
    assert len(sha) == 64


def test_synthesize_glv1_archive_valid(smoke_mod):
    """GLV1 synthesizer produces a valid archive."""
    blob, sha = smoke_mod._synthesize_glv1_archive()
    assert isinstance(blob, bytes)
    assert blob[:4] == b"GLV1"
    assert len(sha) == 64


def test_synthesize_atw2_archive_valid(smoke_mod):
    """ATW2 synthesizer produces a valid archive."""
    blob, sha = smoke_mod._synthesize_atw2_archive()
    assert isinstance(blob, bytes)
    assert blob[:4] == b"ATW2"
    assert len(sha) == 64


def test_classify_dp1_all_parser_essential(smoke_mod):
    """DP1 has ZERO parser-safe bytes (all sections brotli/struct/JSON)."""
    blob, _ = smoke_mod._synthesize_dp1_archive()
    result = smoke_mod.classify_dp1(blob)
    assert result.substrate_id == "dp1_pretrained_driving_prior"
    assert result.parser_safe_subset_total_bytes == 0
    # Every region must be parser-essential.
    for region in result.regions:
        assert region.parser_essential is True
        assert region.parser_kind in smoke_mod.PARSER_ESSENTIAL_KINDS


def test_classify_vqv1_indices_blob_parser_safe(smoke_mod):
    """VQV1 indices_blob is parser-safe (RAW int16) but score-affecting."""
    blob, _ = smoke_mod._synthesize_vqv1_archive()
    result = smoke_mod.classify_vqv1(blob)
    assert result.substrate_id == "vq_vae"
    # Find the indices_blob region.
    indices_region = next(r for r in result.regions if r.region_name == "indices_blob")
    assert indices_region.parser_kind == smoke_mod.KIND_RAW_BYTE_SECTION
    assert indices_region.parser_essential is False
    assert indices_region.score_relevance == smoke_mod.SCORE_RELEVANCE_SCORE_AFFECTING
    # Total parser-safe = indices_blob size; ALL score-affecting.
    assert result.parser_safe_subset_total_bytes == indices_region.size()
    assert result.parser_safe_score_affecting_bytes == indices_region.size()
    assert result.parser_safe_score_opaque_bytes == 0


def test_classify_glv1_all_parser_essential(smoke_mod):
    """GLV1 has ZERO parser-safe bytes."""
    blob, _ = smoke_mod._synthesize_glv1_archive()
    result = smoke_mod.classify_glv1(blob)
    assert result.substrate_id == "grayscale_lut"
    assert result.parser_safe_subset_total_bytes == 0


def test_classify_atw2_three_raw_sections(smoke_mod):
    """ATW2 has 3 RAW sections (latent_residual + class_prior_table + cdf_table)."""
    blob, _ = smoke_mod._synthesize_atw2_archive()
    result = smoke_mod.classify_atw2(blob)
    assert result.substrate_id == "atw_codec_v2"
    raw_names = {
        r.region_name
        for r in result.regions
        if r.parser_kind == smoke_mod.KIND_RAW_BYTE_SECTION
    }
    assert raw_names == {
        "latent_residual_blob",
        "class_prior_table_blob",
        "cdf_table_blob",
    }
    # All RAW sections must be score-affecting (decoder side-information).
    for region in result.regions:
        if region.parser_kind == smoke_mod.KIND_RAW_BYTE_SECTION:
            assert region.score_relevance == smoke_mod.SCORE_RELEVANCE_SCORE_AFFECTING
    assert result.parser_safe_subset_total_bytes > 0
    assert result.parser_safe_score_opaque_bytes == 0


def test_classify_all_substrates_returns_4(smoke_mod):
    """classify_all_substrates returns 4 results (1 per in-scope substrate)."""
    results = smoke_mod.classify_all_substrates()
    assert len(results) == 4
    substrate_ids = {r.substrate_id for r in results}
    assert substrate_ids == {
        "dp1_pretrained_driving_prior",
        "vq_vae",
        "grayscale_lut",
        "atw_codec_v2",
    }


def test_aggregate_verdict_mixed_score_affecting(smoke_mod):
    """Aggregate verdict on the canonical 4 substrates is MIXED (parser-safe but score-affecting)."""
    results = smoke_mod.classify_all_substrates()
    verdict, rationale = smoke_mod.compute_aggregate_verdict(results)
    # Sanity: parser-safe total > 0 (VQ-VAE + ATW V2 contribute), all
    # score-affecting (no score-opaque RAW regions).
    assert sum(r.parser_safe_subset_total_bytes for r in results) > 0
    assert sum(r.parser_safe_score_opaque_bytes for r in results) == 0
    assert verdict == "METHODOLOGY_EXTENSION_MIXED_PARSER_SAFE_BUT_SCORE_AFFECTING"
    assert "score-affecting" in rationale.lower()


def test_aggregate_verdict_taxonomy_pinned(smoke_mod):
    """Verdict classifier handles ALL_EMPTY and SCORE_OPAQUE cases."""
    # Synthesize a fake all-empty result.
    empty_result = smoke_mod.SubstrateClassification(
        substrate_id="x",
        archive_magic="X",
        archive_bytes=0,
        archive_sha256="0" * 64,
        regions=(),
        parser_safe_subset_total_bytes=0,
        parser_safe_score_affecting_bytes=0,
        parser_safe_score_opaque_bytes=0,
        parser_safe_unknown_bytes=0,
        canonical_equation_26_eligible_contexts=(),
    )
    verdict, _ = smoke_mod.compute_aggregate_verdict([empty_result])
    assert verdict == "METHODOLOGY_EXTENSION_ALL_EMPTY"

    # Synthesize a score-opaque result.
    opaque_region = smoke_mod.SubstrateRegion(
        region_name="opaque",
        start_byte=0,
        end_byte=10,
        parser_kind=smoke_mod.KIND_RAW_BYTE_SECTION,
        parser_essential=False,
        score_relevance=smoke_mod.SCORE_RELEVANCE_SCORE_OPAQUE,
        role="latent_stream",
        rationale="hypothetical opaque RAW region",
    )
    opaque_result = smoke_mod.SubstrateClassification(
        substrate_id="y",
        archive_magic="Y",
        archive_bytes=10,
        archive_sha256="0" * 64,
        regions=(opaque_region,),
        parser_safe_subset_total_bytes=10,
        parser_safe_score_affecting_bytes=0,
        parser_safe_score_opaque_bytes=10,
        parser_safe_unknown_bytes=0,
        canonical_equation_26_eligible_contexts=(),
    )
    verdict, _ = smoke_mod.compute_aggregate_verdict([opaque_result])
    assert verdict == "METHODOLOGY_EXTENSION_PARSER_SAFE_AND_SCORE_OPAQUE"


def test_smoke_writes_artifacts_with_provenance(smoke_mod, tmp_path):
    """End-to-end smoke writes JSON + MD with canonical Provenance."""
    results = smoke_mod.classify_all_substrates()
    verdict, rationale = smoke_mod.compute_aggregate_verdict(results)
    json_path = smoke_mod._write_smoke_result_json(
        tmp_path, results, verdict, rationale
    )
    md_path = smoke_mod._write_smoke_result_md(
        tmp_path, results, verdict, rationale
    )
    assert json_path.is_file()
    assert md_path.is_file()
    import json as _json

    payload = _json.loads(json_path.read_text())
    # Canonical non-promotable Provenance per Catalog #192 / #323.
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["axis_tag"] == "[macOS-CPU advisory]"
    assert payload["evidence_grade"] == "macOS-CPU-advisory"
    assert payload["provenance"]["evidence_grade"] in (
        "macOS_cpu_advisory",
        "macos_cpu_advisory",
        "macOS-CPU-advisory",
    )
    assert payload["verdict_label"] == verdict
    assert payload["substrate_count"] == 4
    # Catalog disciplines explicitly enumerated.
    assert "#192 macOS-CPU non-promotable" in payload["catalog_disciplines_honored"]
    assert "#323 canonical Provenance umbrella" in payload[
        "catalog_disciplines_honored"
    ]
    assert "#344 canonical equation cross-ref" in payload[
        "catalog_disciplines_honored"
    ]


def test_main_dry_run_returns_zero(smoke_mod, capsys):
    """Main --dry-run prints plan + returns 0."""
    rc = smoke_mod.main(["--dry-run"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "dry_run" in captured.out
    assert "in_scope_substrates" in captured.out


def test_smoke_synthesizer_determinism(smoke_mod):
    """Synthesizers are deterministic — same shape -> same sha256."""
    blob_a, sha_a = smoke_mod._synthesize_dp1_archive()
    blob_b, sha_b = smoke_mod._synthesize_dp1_archive()
    assert sha_a == sha_b
    assert blob_a == blob_b
