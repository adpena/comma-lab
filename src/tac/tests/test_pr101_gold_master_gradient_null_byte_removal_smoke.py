# SPDX-License-Identifier: MIT
"""Tests for PR101 GOLD master-gradient-null-byte REMOVAL smoke.

Covers tools/run_pr101_gold_master_gradient_null_byte_removal_smoke.py CLI
parsing, fec6 archive load + null-byte indices load, 4-variant byte
modification correctness, score recording correctness, hypothesis
disambiguation logic (H1/H2/H3), and Provenance per Catalog #323.
"""

from __future__ import annotations

import importlib.util
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SMOKE_SCRIPT_PATH = REPO_ROOT / "tools/run_pr101_gold_master_gradient_null_byte_removal_smoke.py"


def _load_smoke_module():
    spec = importlib.util.spec_from_file_location(
        "pr101_gold_null_byte_removal_smoke", SMOKE_SCRIPT_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["pr101_gold_null_byte_removal_smoke"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def smoke():
    return _load_smoke_module()


def test_smoke_script_exists():
    assert SMOKE_SCRIPT_PATH.is_file(), f"smoke script not found at {SMOKE_SCRIPT_PATH}"


def test_smoke_canonical_constants_match_contest(smoke):
    assert smoke.CANONICAL_RATE_DENOM_BYTES == 37_545_489
    assert smoke.CANONICAL_RATE_MULTIPLIER == 25.0
    assert smoke.CANONICAL_SEG_MULTIPLIER == 100.0
    assert smoke.CANONICAL_POSE_SQRT_INNER == 10.0
    assert smoke.EPSILON_GRADIENT_NULL == 1e-9


def test_smoke_fec6_archive_sha_pinned(smoke):
    """The canonical fec6 frontier archive sha must match canonical_frontier_pointer.json."""
    assert smoke.FEC6_FRONTIER_SHA256 == (
        "6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf"
    )


def test_smoke_load_master_gradient_null_indices_returns_expected_count(smoke):
    """Master-gradient .npy must yield exactly 16,292 null-byte indices per probe matrix anchor."""
    null_indices = smoke._load_master_gradient_null_indices()
    assert isinstance(null_indices, np.ndarray)
    assert len(null_indices) == 16_292
    # Per CLAUDE.md "Apples-to-apples evidence discipline": MUST be ascending unique indices
    assert (np.diff(null_indices) > 0).all()


def test_smoke_read_fec6_archive_verifies_sha(smoke):
    """fec6 archive load must verify sha matches canonical pointer."""
    archive_bytes, inner_bytes, member_name = smoke._read_fec6_archive()
    assert len(inner_bytes) == 178_417
    assert member_name == "x"


def test_smoke_derive_random_bytes_deterministic(smoke):
    """PCG64-derived random bytes must be deterministic across calls."""
    a = smoke._derive_random_bytes_for_indices(16_292)
    b = smoke._derive_random_bytes_for_indices(16_292)
    assert a == b
    assert len(a) == 16_292
    assert isinstance(a, bytes)


def test_smoke_build_variant_archive_baseline_is_identity(smoke):
    """V_BASELINE variant must NOT mutate inner bytes."""
    _, inner_bytes, member_name = smoke._read_fec6_archive()
    null_indices = smoke._load_master_gradient_null_indices()
    archive_zip_bytes, sha, n_bytes = smoke._build_variant_archive(
        inner_bytes, member_name, null_indices, "V_BASELINE", "control"
    )
    # Decode + verify inner bytes preserved
    with zipfile.ZipFile(io.BytesIO(archive_zip_bytes), mode="r") as zf:
        recovered = zf.read(member_name)
    assert recovered == inner_bytes


def test_smoke_build_variant_archive_zero_mutates_only_null_indices(smoke):
    """V_ZERO variant must set 0x00 at null-indices and preserve other bytes."""
    _, inner_bytes, member_name = smoke._read_fec6_archive()
    null_indices = smoke._load_master_gradient_null_indices()
    archive_zip_bytes, _, _ = smoke._build_variant_archive(
        inner_bytes, member_name, null_indices, "V_ZERO", "0x00"
    )
    with zipfile.ZipFile(io.BytesIO(archive_zip_bytes), mode="r") as zf:
        mutated = zf.read(member_name)
    # All null-indices must be 0x00
    null_set = set(int(i) for i in null_indices)
    for idx in null_indices[:100]:  # spot-check first 100
        assert mutated[int(idx)] == 0x00
    # At least one NON-null index must equal the original (preserves non-null bytes)
    non_null_idx = next(i for i in range(178_417) if i not in null_set)
    assert mutated[non_null_idx] == inner_bytes[non_null_idx]


def test_smoke_build_variant_archive_half_uses_0x80(smoke):
    """V_HALF must set 0x80 at null-indices."""
    _, inner_bytes, member_name = smoke._read_fec6_archive()
    null_indices = smoke._load_master_gradient_null_indices()
    archive_zip_bytes, _, _ = smoke._build_variant_archive(
        inner_bytes, member_name, null_indices, "V_HALF", "0x80"
    )
    with zipfile.ZipFile(io.BytesIO(archive_zip_bytes), mode="r") as zf:
        mutated = zf.read(member_name)
    for idx in null_indices[:100]:
        assert mutated[int(idx)] == 0x80


def test_smoke_build_variant_archive_random_deterministic(smoke):
    """V_RANDOM must be deterministic + apply PCG64 bytes at null-indices."""
    _, inner_bytes, member_name = smoke._read_fec6_archive()
    null_indices = smoke._load_master_gradient_null_indices()
    a, _, _ = smoke._build_variant_archive(
        inner_bytes, member_name, null_indices, "V_RANDOM", "pcg64"
    )
    b, _, _ = smoke._build_variant_archive(
        inner_bytes, member_name, null_indices, "V_RANDOM", "pcg64"
    )
    assert a == b


def test_smoke_build_variant_archive_rejects_unknown_variant(smoke):
    """Unknown variants must raise NullByteRemovalSmokeError per Catalog #287 placeholder rejection pattern."""
    _, inner_bytes, member_name = smoke._read_fec6_archive()
    null_indices = smoke._load_master_gradient_null_indices()
    with pytest.raises(smoke.NullByteRemovalSmokeError):
        smoke._build_variant_archive(
            inner_bytes, member_name, null_indices, "V_UNKNOWN", "test"
        )


def test_smoke_classify_hypothesis_h1_score_irrelevant(smoke):
    """H1: all variants within HYPOTHESIS_H1_THRESHOLD."""
    label, rationale = smoke._classify_hypothesis(0.19205, 1e-5, -5e-5, 8e-5)
    assert label == "H1_SCORE_IRRELEVANT"
    assert "BUILD justified" in rationale


def test_smoke_classify_hypothesis_h2_partially_relevant(smoke):
    """H2: variants diverge but not catastrophic."""
    label, rationale = smoke._classify_hypothesis(0.19205, 0.001, -0.002, 0.005)
    assert label == "H2_PARTIALLY_RELEVANT"
    assert "cascade pivot recommended" in rationale


def test_smoke_classify_hypothesis_h3_opaque_to_scorer(smoke):
    """H3: catastrophic divergence (substitution breaks decoded output)."""
    label, rationale = smoke._classify_hypothesis(0.19205, 0.5, -0.3, 1.2)
    assert label == "H3_OPAQUE_TO_SCORER"
    assert "rescope" in rationale


def test_smoke_classify_hypothesis_h3_inflate_failure_short_circuit(smoke):
    """H3: ANY inflate failure short-circuits to H3 regardless of scalar dS."""
    label, rationale = smoke._classify_hypothesis(
        0.19205, 0.0, 0.0, 0.0, n_inflate_failures=1
    )
    assert label == "H3_OPAQUE_TO_SCORER"
    assert "bit-essential for archive parser" in rationale
    assert "PR101 magic header" in rationale


def test_smoke_cli_dry_run_succeeds(smoke):
    """--dry-run path verifies fixture paths without running auth_eval."""
    rc = smoke.main(["--dry-run"])
    assert rc == 0


def test_smoke_provenance_per_catalog_323(smoke):
    """smoke uses canonical Provenance builder per Catalog #323."""
    # Verify the canonical builder import is present + works
    from tac.provenance import build_provenance_for_macos_cpu_advisory

    prov = build_provenance_for_macos_cpu_advisory(
        archive_sha256="0" * 64,
        source_path="experiments/results/test",
    )
    assert prov.score_claim_valid is False
    assert prov.promotion_eligible is False
    assert prov.measurement_axis == "[macOS-CPU advisory]"
    assert prov.evidence_grade.value == "macos_cpu_advisory"
