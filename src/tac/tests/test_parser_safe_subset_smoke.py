# SPDX-License-Identifier: MIT
"""Tests for ``tools/run_parser_safe_subset_smoke.py``.

The smoke maps master-gradient-null bytes to the actual PR101/FEC6 parser
regions. Its key contract is conservative: a null-gradient byte is not
replaceable unless it is also downstream of parser dispatch.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "run_parser_safe_subset_smoke.py"


def _load_tool_module():
    spec = importlib.util.spec_from_file_location("run_parser_safe_subset_smoke", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


SMOKE = _load_tool_module()


def test_tool_module_imports_canonical_surfaces():
    assert hasattr(SMOKE, "static_classify_regions")
    assert hasattr(SMOKE, "construct_parser_safe_subset")
    assert hasattr(SMOKE, "classify_null_indices_per_region")
    assert hasattr(SMOKE, "_write_smoke_result_json")
    assert hasattr(SMOKE, "_write_smoke_result_md")


def test_canonical_constants_match_contest_formula():
    assert SMOKE.CANONICAL_RATE_DENOM_BYTES == 37_545_489
    assert SMOKE.CANONICAL_RATE_MULTIPLIER == 25.0
    assert SMOKE.CANONICAL_SEG_MULTIPLIER == 100.0
    assert SMOKE.CANONICAL_POSE_SQRT_INNER == 10.0
    assert SMOKE.EPSILON_GRADIENT_NULL == 1e-9


def test_fec6_frontier_archive_sha_pinned():
    assert SMOKE.FEC6_FRONTIER_SHA256 == (
        "6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf"
    )


def test_load_master_gradient_null_indices_expected_count_and_order():
    null_indices = SMOKE._load_master_gradient_null_indices()
    assert isinstance(null_indices, np.ndarray)
    assert len(null_indices) == 16_292
    assert null_indices.dtype == np.int64
    assert bool((np.diff(null_indices) > 0).all())


def test_read_fec6_archive_verifies_sha_and_member():
    archive_bytes, inner_bytes, member_name = SMOKE._read_fec6_archive()
    assert len(archive_bytes) >= len(inner_bytes)
    assert len(inner_bytes) == 178_417
    assert member_name == "x"


def test_static_classify_regions_partitions_archive_exactly():
    _, inner_bytes, _ = SMOKE._read_fec6_archive()
    regions = SMOKE.static_classify_regions(inner_bytes)
    assert [region.name for region in regions] == [
        "A_fp11_outer_wrapper",
        "B_pr101_decoder_brotli",
        "C_pr101_latent_lzma",
        "D_pr101_sidecar_brotli",
        "E_fec6_selector_len_uint16",
        "F_fec6_selector_payload",
    ]
    assert regions[0].start_byte == 0
    assert regions[-1].end_byte == len(inner_bytes)
    for left, right in zip(regions, regions[1:]):
        assert left.end_byte == right.start_byte
    assert all(region.parser_essential for region in regions)


def test_region_classification_matches_parser_safe_anchor_counts():
    null_indices = SMOKE._load_master_gradient_null_indices()
    _, inner_bytes, _ = SMOKE._read_fec6_archive()
    regions = SMOKE.static_classify_regions(inner_bytes)
    results = SMOKE.classify_null_indices_per_region(null_indices, regions)
    assert {r.region_name: r.null_index_count for r in results} == {
        "A_fp11_outer_wrapper": 8,
        "B_pr101_decoder_brotli": 39,
        "C_pr101_latent_lzma": 15_387,
        "D_pr101_sidecar_brotli": 607,
        "E_fec6_selector_len_uint16": 2,
        "F_fec6_selector_payload": 249,
    }
    assert sum(r.null_index_count for r in results) == 16_292


def test_construct_parser_safe_subset_empty_for_fec6_frontier():
    null_indices = SMOKE._load_master_gradient_null_indices()
    _, inner_bytes, _ = SMOKE._read_fec6_archive()
    regions = SMOKE.static_classify_regions(inner_bytes)
    subset = SMOKE.construct_parser_safe_subset(null_indices, regions)
    assert isinstance(subset, np.ndarray)
    assert subset.dtype == np.int64
    assert len(subset) == 0


def test_build_variant_archive_empty_subset_is_identity_for_modified_variants():
    _, inner_bytes, member_name = SMOKE._read_fec6_archive()
    empty_subset = np.array([], dtype=np.int64)
    baseline_zip, baseline_sha, baseline_bytes = SMOKE._build_variant_archive(
        inner_bytes, member_name, empty_subset, "V_BASELINE"
    )
    for variant_name in ("V_ZERO", "V_HALF", "V_RANDOM"):
        variant_zip, variant_sha, variant_bytes = SMOKE._build_variant_archive(
            inner_bytes, member_name, empty_subset, variant_name
        )
        assert variant_zip == baseline_zip
        assert variant_sha == baseline_sha
        assert variant_bytes == baseline_bytes


def test_classify_verdict_empty_subset_is_terminal_negative():
    verdict, rationale = SMOKE._classify_verdict(0, None, None, None, None, 0)
    assert verdict == "PARSER_SAFE_SUBSET_EMPTY"
    assert "ALL 16,292" in rationale
    assert "NO null-gradient region downstream of parser dispatch" in rationale


def test_main_static_only_writes_non_promotional_artifacts(tmp_path):
    rc = SMOKE.main(["--static-only", "--output-dir", str(tmp_path)])
    assert rc == 0
    json_path = tmp_path / "smoke_result.json"
    md_path = tmp_path / "smoke_result.md"
    assert json_path.is_file()
    assert md_path.is_file()
    payload = json.loads(json_path.read_text())
    assert payload["parser_safe_subset_size"] == 0
    assert payload["verdict_label"] == "PARSER_SAFE_SUBSET_EMPTY"
    assert payload["axis_tag"] == "[macOS-CPU advisory]"
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["rank_or_kill_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["variants"] == []


def test_cli_dry_run_succeeds_without_writing_auth_eval_artifacts(capsys):
    rc = SMOKE.main(["--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["dry_run"] is True
    assert payload["parser_safe_subset_size"] == 0
    assert payload["null_index_count"] == 16_292
