# SPDX-License-Identifier: MIT
"""Tests for ``tools/run_magic_codec_pair_2_sparse_packet_ir_fec6_null_byte_smoke.py``.

Per CLAUDE.md "Subagent coherence-by-default" + Catalog #344 + Catalog #272
+ Catalog #192 + the MAGIC CODEC × CASCADE STACKING ANALYSIS memo
``.omx/research/magic_codec_x_todays_cascade_stacking_analysis_20260520.md``.

Sister of ``src/tac/tests/test_magic_codec_dense_streams_dwt_residual_smoke.py``
(pair #1; disjoint scope — pair #1 measures DWT detail-subband residual
byte budget; THIS pair #2 measures fec6 frontier master-gradient
null-byte residual byte budget via sparse_packet_ir SRL1).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = (
    REPO_ROOT
    / "tools"
    / "run_magic_codec_pair_2_sparse_packet_ir_fec6_null_byte_smoke.py"
)


def _load_tool_module():
    """Import the tool module as a runtime module (tools/ is not a package)."""
    spec = importlib.util.spec_from_file_location(
        "run_magic_codec_pair_2_sparse_packet_ir_fec6_null_byte_smoke",
        TOOL_PATH,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


SMOKE = _load_tool_module()


def test_tool_module_imports_canonical_surfaces():
    """Catalog #229 PV: every canonical helper the smoke depends on is importable."""
    assert hasattr(SMOKE, "run_smoke")
    assert hasattr(SMOKE, "append_third_empirical_anchor")
    assert hasattr(SMOKE, "compute_residual_int16")
    assert hasattr(SMOKE, "encode_config_a_baseline_brotli")
    assert hasattr(SMOKE, "encode_config_c_procedural_plus_srl1")
    assert hasattr(SMOKE, "encode_config_c_alt_brotli_residual")
    assert hasattr(SMOKE, "load_fec6_frontier_archive_bytes")
    assert hasattr(SMOKE, "extract_inner_member_bytes_from_archive")
    assert hasattr(SMOKE, "find_master_gradient_null_byte_indices")
    assert hasattr(SMOKE, "load_fec6_null_byte_anchor_metadata")
    assert hasattr(SMOKE, "PAIR_2_PREDICTED_DELTA_S")
    assert hasattr(SMOKE, "CANONICAL_RATE_MULTIPLIER")
    assert hasattr(SMOKE, "CANONICAL_RATE_DENOM_BYTES")
    assert hasattr(SMOKE, "FEC6_FRONTIER_SHA256_PREFIX")


def test_canonical_constants_match_contest_formula():
    """The smoke uses the canonical contest formula constants (Catalog #344 sister)."""
    assert SMOKE.CANONICAL_RATE_MULTIPLIER == 25.0
    assert SMOKE.CANONICAL_RATE_DENOM_BYTES == 37_545_489
    # Pair #2 prediction from stacking analysis memo §7
    assert SMOKE.PAIR_2_PREDICTED_DELTA_S == -0.00109
    assert SMOKE.PAIR_2_COMPOSITION_ALPHA_ESTIMATE == 0.9
    # fec6 frontier sha prefix per Catalog #343 canonical pointer
    assert SMOKE.FEC6_FRONTIER_SHA256_PREFIX == "6bae0201fb08"


def test_compute_residual_int16_round_trip_self_is_zero():
    """Residual of empirical against itself is all-zero."""
    arr = np.array([0, 64, 128, 200, 255], dtype=np.uint8)
    residual = SMOKE.compute_residual_int16(arr, arr)
    assert residual.dtype == np.int16
    assert residual.shape == arr.shape
    assert int(np.abs(residual).sum()) == 0


def test_compute_residual_int16_byte_exact_reconstruction():
    """synthetic.astype(int16) + residual == empirical (lossless)."""
    rng = np.random.RandomState(42)
    empirical = rng.randint(0, 256, size=500, dtype=np.uint8)
    synthetic = rng.randint(0, 256, size=500, dtype=np.uint8)
    residual = SMOKE.compute_residual_int16(empirical, synthetic)
    reconstructed = synthetic.astype(np.int16) + residual
    assert np.array_equal(reconstructed, empirical.astype(np.int16))


def test_compute_residual_int16_rejects_dtype_mismatch():
    """Catalog #229 type discipline: residual inputs must be uint8."""
    a = np.array([0, 64, 100], dtype=np.uint8)
    b = np.array([0, 64, 100], dtype=np.int8)
    with pytest.raises(ValueError, match="uint8"):
        SMOKE.compute_residual_int16(a, b)


def test_compute_residual_int16_rejects_shape_mismatch():
    """Catalog #229 type discipline: residual inputs must be same shape."""
    a = np.array([0, 64, 128], dtype=np.uint8)
    b = np.array([[0, 64], [128, 200]], dtype=np.uint8)
    with pytest.raises(ValueError, match="shape mismatch"):
        SMOKE.compute_residual_int16(a, b)


def test_encode_config_a_baseline_brotli_returns_byte_count():
    """Configuration A direct empirical brotli baseline emits a positive byte count."""
    arr = np.array([1, 2, 3, 4, 5] * 100, dtype=np.uint8)
    n_bytes = SMOKE.encode_config_a_baseline_brotli(arr)
    assert isinstance(n_bytes, int)
    assert n_bytes > 0
    # Brotli compresses 500 repetitive bytes well below the raw 500
    assert n_bytes < 500


def test_encode_config_a_baseline_brotli_all_zeros_compresses_tiny():
    """All-zero byte array compresses to a near-trivial brotli payload."""
    arr = np.zeros(16292, dtype=np.uint8)
    n_bytes = SMOKE.encode_config_a_baseline_brotli(arr)
    # 16,292 zeros should compress to < 50 bytes under brotli q=11
    assert n_bytes < 50


def test_encode_config_c_returns_total_bytes_plus_selection_log():
    """Configuration C emits total_bytes + selection_log with SRL1 canonical keys."""
    rng = np.random.RandomState(42)
    residual = rng.randint(-128, 128, size=200, dtype=np.int16)
    seed = b"\x00" * 32
    total_bytes, selection_log = SMOKE.encode_config_c_procedural_plus_srl1(
        residual=residual,
        seed_bytes=seed,
    )
    assert isinstance(total_bytes, int)
    assert total_bytes > 0
    assert isinstance(selection_log, dict)
    assert selection_log["seed_bytes_len"] == 32
    assert selection_log["srl1_serialized_payload_len"] > 0
    assert total_bytes == 32 + selection_log["srl1_serialized_payload_len"]
    assert selection_log["codec_name"] == "sparse_packet_ir_srl1"
    assert selection_log["codec_id"] == "SRL1"
    assert "residual_sparsity_ratio" in selection_log
    assert 0.0 <= selection_log["residual_sparsity_ratio"] <= 1.0
    assert "srl1_serialized_payload_sha256" in selection_log
    assert len(selection_log["srl1_serialized_payload_sha256"]) == 64
    # Envelope overhead canonical = 13 bytes per the SRL1 wire format
    # (magic(4) + total_length(4) + n_nonzero(4) + dtype_code(1))
    assert selection_log["srl1_envelope_overhead_bytes"] == 13


def test_encode_config_c_seed_sensitivity():
    """Different seeds produce different SRL1 payloads (Catalog #272 prep)."""
    rng = np.random.RandomState(42)
    residual_a = rng.randint(-128, 128, size=100, dtype=np.int16)
    residual_b = rng.randint(-128, 128, size=100, dtype=np.int16)
    seed_a = b"\x00" * 32
    seed_b = b"\xff" * 32
    _, log_a = SMOKE.encode_config_c_procedural_plus_srl1(
        residual=residual_a, seed_bytes=seed_a
    )
    _, log_b = SMOKE.encode_config_c_procedural_plus_srl1(
        residual=residual_b, seed_bytes=seed_b
    )
    # Different residuals → different SRL1 payload sha
    assert (
        log_a["srl1_serialized_payload_sha256"]
        != log_b["srl1_serialized_payload_sha256"]
    )


def test_find_master_gradient_null_byte_indices_canonical_fec6_frontier():
    """The canonical fec6 frontier master gradient has exactly 16,292 null bytes."""
    canonical_npy = (
        REPO_ROOT
        / "experiments"
        / "results"
        / "master_gradient_per_archive_fp64_extraction_wave_20260519T012404Z"
        / "master_gradient_pr101_fec6_frontier_macos_cpu_advisory_8pair_fp64_20260518.npy"
    )
    if not canonical_npy.exists():
        pytest.skip("canonical fec6 frontier master_gradient npy not present")
    indices = SMOKE.find_master_gradient_null_byte_indices(canonical_npy)
    # Per feedback_null_byte_probe_matrix_landed_20260520.md the fec6
    # frontier has 16,292 master-gradient-null bytes (9.13% of 178,417)
    assert indices.dtype == np.uint64
    assert indices.size == 16292
    # Indices are strictly increasing
    assert np.all(np.diff(indices.astype(np.int64)) > 0)
    # All indices fit in the inner-member size (178,417 bytes)
    assert int(indices.max()) < 178_417


def test_load_fec6_frontier_archive_bytes_verifies_sha_prefix():
    """The fec6 frontier archive loader verifies sha256 prefix per Catalog #343."""
    canonical_archive = (
        REPO_ROOT
        / "experiments"
        / "results"
        / "pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex"
        / "archive.zip"
    )
    if not canonical_archive.exists():
        pytest.skip("canonical fec6 frontier archive not present")
    archive_bytes, archive_sha = SMOKE.load_fec6_frontier_archive_bytes(
        canonical_archive
    )
    assert isinstance(archive_bytes, bytes)
    assert len(archive_bytes) == 178517  # canonical fec6 frontier size
    assert archive_sha.startswith(SMOKE.FEC6_FRONTIER_SHA256_PREFIX)


def test_extract_inner_member_bytes_canonical_fec6_frontier():
    """The fec6 frontier archive has exactly one inner member 'x' with 178,417 bytes."""
    canonical_archive = (
        REPO_ROOT
        / "experiments"
        / "results"
        / "pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex"
        / "archive.zip"
    )
    if not canonical_archive.exists():
        pytest.skip("canonical fec6 frontier archive not present")
    inner = SMOKE.extract_inner_member_bytes_from_archive(canonical_archive)
    assert isinstance(inner, bytes)
    assert len(inner) == 178_417


def test_run_smoke_end_to_end_emits_canonical_keys(tmp_path):
    """End-to-end smoke on real fec6 frontier emits all canonical custody fields."""
    canonical_archive = (
        REPO_ROOT
        / "experiments"
        / "results"
        / "pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex"
        / "archive.zip"
    )
    canonical_matrix = (
        REPO_ROOT
        / "experiments"
        / "results"
        / "null_byte_probe_matrix_20260520T223742Z"
        / "null_byte_matrix.json"
    )
    if not canonical_archive.exists() or not canonical_matrix.exists():
        pytest.skip("canonical fec6 frontier archive or null_byte_matrix not present")
    base_seed = hashlib.sha256(b"test_seed_pair_2_fec6_null_byte").digest()
    result = SMOKE.run_smoke(
        archive_path=canonical_archive,
        matrix_path=canonical_matrix,
        base_seed_bytes=base_seed,
        generator_kind="pcg64",
        max_null_indices=512,  # fast smoke for regression test
    )
    # Custody triple per Catalog #127 + #192 + #323
    assert result["axis_tag"] == "[macOS-CPU advisory]"
    assert result["hardware_substrate"] == "darwin_arm64_m5_max_macos_cpu_advisory"
    assert result["evidence_grade"] == "local_cpu_smoke_advisory"
    assert result["promotion_eligible"] is False
    assert result["score_claim_valid"] is False
    assert result["score_claim_axis"] is None
    # fec6 frontier custody
    assert result["fec6_archive_sha256"].startswith(SMOKE.FEC6_FRONTIER_SHA256_PREFIX)
    assert result["fec6_archive_size_bytes"] == 178517
    assert result["fec6_inner_member_size_bytes"] == 178_417
    assert result["fec6_frontier_lane_id"] == SMOKE.FEC6_FRONTIER_LANE_ID
    # 3-way comparison fields
    assert "config_a_in_place_charged_bytes" in result
    assert "config_a_baseline_brotli_bytes" in result
    assert "config_b_procedural_only_bytes" in result
    assert "config_c_procedural_plus_srl1_bytes" in result
    assert "bytes_saved_c_vs_a" in result
    assert "empirical_delta_s" in result
    assert "predicted_delta_s_pair_2" in result
    assert result["predicted_delta_s_pair_2"] == -0.00109
    # Cap honored
    assert result["n_null_used_in_smoke"] == 512
    # Canonical equation linkage (NEW IN-DOMAIN context per Catalog #344)
    assert (
        result["canonical_equation_id"]
        == "procedural_codebook_from_seed_compression_savings_v1"
    )
    assert result["canonical_equation_in_domain_context"] == (
        "sparse_packet_ir_srl1_correction_on_fec6_frontier_null_bytes"
    )
    # SRL1 selection log canonical fields
    assert result["config_c_selection_log"]["codec_id"] == "SRL1"
    assert result["config_c_selection_log"]["codec_name"] == "sparse_packet_ir_srl1"
    # Cascade verdict is one of the canonical taxonomy
    assert result["cascade_verdict"] in {
        "PAIR_2_VALIDATED_PROCEED_TO_PAIR_3_OR_PAIR_4_ORTHOGONALITY",
        "PARTIAL_RESCUE_NET_SAVINGS_BUT_OUTSIDE_PREDICTED_BAND",
        "PAIR_2_FALSIFIED_CASCADE_FURTHER_NARROWS_PIVOT_TO_PAIR_4_OR_DP1_ONLY",
        "INDETERMINATE_REQUIRES_PAIRED_LINUX_X86_64_VERIFICATION",
    }


def test_run_smoke_full_anchor_count_matches_matrix():
    """Full smoke (no max_null_indices cap) processes all 16,292 fec6 null bytes."""
    canonical_archive = (
        REPO_ROOT
        / "experiments"
        / "results"
        / "pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex"
        / "archive.zip"
    )
    canonical_matrix = (
        REPO_ROOT
        / "experiments"
        / "results"
        / "null_byte_probe_matrix_20260520T223742Z"
        / "null_byte_matrix.json"
    )
    if not canonical_archive.exists() or not canonical_matrix.exists():
        pytest.skip("canonical fec6 frontier archive or null_byte_matrix not present")
    base_seed = hashlib.sha256(b"test_seed_pair_2_full_anchor_count").digest()
    result = SMOKE.run_smoke(
        archive_path=canonical_archive,
        matrix_path=canonical_matrix,
        base_seed_bytes=base_seed,
        generator_kind="pcg64",
        max_null_indices=None,
    )
    assert result["null_byte_matrix_anchor_n_null_expected"] == 16292
    assert result["null_byte_matrix_anchor_n_null_observed"] == 16292
    assert result["null_byte_matrix_anchor_match_observed_vs_expected"] is True
    assert result["n_null_used_in_smoke"] == 16292
    # Catalog #272 byte-mutation smoke is seed-sensitive on the real
    # canonical fec6 frontier null-byte set
    assert result["byte_mutation_smoke_verdict_seed_sensitive"] is True
    # Pair #2 verdict is structurally CARGO-CULTED at full scale because
    # the procedural-codebook + SRL1 stack adds residual overhead that
    # exceeds the in-place 16,292-byte baseline (see landing memo)
    assert (
        result["canonical_equation_verdict_HARD_EARNED_or_CARGO_CULTED_at_2sigma"]
        == "CARGO-CULTED"
    )
    assert result["cascade_verdict"] == (
        "PAIR_2_FALSIFIED_CASCADE_FURTHER_NARROWS_PIVOT_TO_PAIR_4_OR_DP1_ONLY"
    )


def test_main_with_skip_anchor_append_writes_artifacts(tmp_path):
    """`--skip-canonical-equation-append` emits JSON + MD without touching registry."""
    canonical_archive = (
        REPO_ROOT
        / "experiments"
        / "results"
        / "pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex"
        / "archive.zip"
    )
    canonical_matrix = (
        REPO_ROOT
        / "experiments"
        / "results"
        / "null_byte_probe_matrix_20260520T223742Z"
        / "null_byte_matrix.json"
    )
    if not canonical_archive.exists() or not canonical_matrix.exists():
        pytest.skip("canonical fec6 frontier archive or null_byte_matrix not present")
    rc = SMOKE.main(
        [
            "--archive-path",
            str(canonical_archive),
            "--null-byte-matrix-path",
            str(canonical_matrix),
            "--max-null-indices",
            "256",  # very small for fast regression
            "--output-dir",
            str(tmp_path),
            "--skip-canonical-equation-append",
        ]
    )
    assert rc == 0
    assert (tmp_path / "smoke_result.json").exists()
    assert (tmp_path / "smoke_result.md").exists()
    payload = json.loads((tmp_path / "smoke_result.json").read_text())
    # Re-verify canonical custody on the persisted artifact
    assert payload["axis_tag"] == "[macOS-CPU advisory]"
    assert payload["evidence_grade"] == "local_cpu_smoke_advisory"
    # Verdict is one of the canonical taxonomy
    assert payload["canonical_equation_verdict_HARD_EARNED_or_CARGO_CULTED_at_2sigma"] in {
        "HARD-EARNED",
        "CARGO-CULTED",
    }


def test_main_refuses_missing_archive():
    """CLI fails fast on missing archive file (rc=2 per UNIX convention)."""
    rc = SMOKE.main(
        [
            "--archive-path",
            "/nonexistent/archive.zip",
            "--skip-canonical-equation-append",
        ]
    )
    assert rc == 2


def test_main_refuses_missing_matrix():
    """CLI fails fast on missing null_byte_matrix file (rc=2)."""
    canonical_archive = (
        REPO_ROOT
        / "experiments"
        / "results"
        / "pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex"
        / "archive.zip"
    )
    if not canonical_archive.exists():
        pytest.skip("canonical fec6 frontier archive not present")
    rc = SMOKE.main(
        [
            "--archive-path",
            str(canonical_archive),
            "--null-byte-matrix-path",
            "/nonexistent/null_byte_matrix.json",
            "--skip-canonical-equation-append",
        ]
    )
    assert rc == 2
