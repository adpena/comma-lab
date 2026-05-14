"""Tests for DP1 canonical composition API (``compose_with``/``decompose``).

Per CLAUDE.md "Subagent coherence-by-default" + "Beauty, simplicity, and
developer experience": the composition API is the canonical reuse surface
that downstream substrates (A1, PR101, HDM8, YUCR, TT5L, sane_hnerv) hit.
These tests pin its byte-stable behavior, leakage refusal, license
propagation, codebook-tampering detection, roundtrip, and per-substrate
1-line usage examples.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tac.substrates.pretrained_driving_prior import (
    DPCOMP_HEADER_SIZE,
    DPCOMP_MAGIC,
    DPCOMP_SCHEMA_VERSION,
    Comma2k19FrameIterator,
    DistillationConfig,
    compose_from_files,
    compose_with,
    decompose,
    distill_codebook,
    known_base_substrates,
    pack_archive,
    verify_composition,
)


def _make_dp1_archive_bytes(*, num_pairs: int = 4, seed: int = 0xDA5C) -> bytes:
    """Build a minimal valid DP1 archive for composition fixture use."""
    iterator = Comma2k19FrameIterator(synthetic=True, n_frames=16, seed=seed)
    cfg = DistillationConfig(
        dataset_name="synthetic_test",
        dataset_sha256="",
        max_frames=16,
        random_seed=seed,
    )
    codebook = distill_codebook(cfg, frames=iter(iterator))
    state_dict = {
        "input_proj.weight": torch.zeros(64, 4, dtype=torch.float32),
        "input_proj.bias": torch.zeros(64, dtype=torch.float32),
        "output_proj.weight": torch.zeros(3, 64, dtype=torch.float32),
        "output_proj.bias": torch.zeros(3, dtype=torch.float32),
    }
    per_pair_bytes = 6
    residual = bytes([0] * (num_pairs * per_pair_bytes))
    meta = {
        "residual_int8_scale": 64.0,
        "lane_id": "lane_dp1_phase_2_hardening_v2_20260514",
        "evidence_grade": "[proxy]",
        "score_claim": False,
    }
    return pack_archive(
        codebook,
        state_dict,
        residual,
        meta,
        num_pairs=num_pairs,
        output_height=64,
        output_width=96,
        per_pair_bytes=per_pair_bytes,
    )


# --- Section 1: header layout invariants ------------------------------------


def test_dpcomp_header_size_invariant():
    """DPCOMP header must be exactly 13 bytes per the public constant."""
    assert DPCOMP_HEADER_SIZE == 13
    assert len(DPCOMP_MAGIC) == 4
    assert DPCOMP_SCHEMA_VERSION == 1


def test_dpcomp_magic_distinct_from_dp1_magic():
    """DPCOMP magic must NOT collide with DP1 magic (different grammar)."""
    from tac.substrates.pretrained_driving_prior import DP1_MAGIC

    assert DPCOMP_MAGIC != DP1_MAGIC
    assert DPCOMP_MAGIC == b"DPC\x00"


# --- Section 2: compose_with happy path -------------------------------------


def test_compose_with_minimal_a1_base_returns_byte_stable_archive():
    dp1_bytes = _make_dp1_archive_bytes()
    a1_bytes = b"FAKE_A1_ARCHIVE_FOR_COMPOSITION_TEST" + bytes(64)
    composed = compose_with(dp1_bytes, a1_bytes, base_substrate="a1")
    assert len(composed) == DPCOMP_HEADER_SIZE + len(dp1_bytes) + len(a1_bytes)
    # Repeating composition with identical inputs is byte-stable.
    composed_again = compose_with(dp1_bytes, a1_bytes, base_substrate="a1")
    assert composed == composed_again


def test_compose_with_each_known_base_substrate():
    """Every base substrate in known_base_substrates() must compose cleanly."""
    dp1_bytes = _make_dp1_archive_bytes()
    fake_base = b"FAKE_BASE_ARCHIVE_FOR_COMPOSITION_TEST" + bytes(128)
    for substrate in known_base_substrates():
        composed = compose_with(dp1_bytes, fake_base, base_substrate=substrate)
        round = decompose(composed)
        assert round.base_substrate == substrate
        assert round.base_archive_bytes == fake_base


def test_known_base_substrates_contains_expected_set():
    """Catalog #211 + composition tests pin the set of supported substrates."""
    expected = {"a1", "pr101", "hdm8", "yucr", "time_traveler_l5", "sane_hnerv"}
    actual = set(known_base_substrates())
    assert actual == expected, (
        f"unexpected base-substrate set {actual} vs {expected}; if adding a "
        f"new substrate, update this test + _KNOWN_BASE_TAGS docstring + "
        f"composition.py docstring + Catalog #211 fixture"
    )


# --- Section 3: decompose roundtrip -----------------------------------------


def test_compose_decompose_roundtrip_preserves_base_bytes():
    dp1_bytes = _make_dp1_archive_bytes()
    pr101_bytes = b"PR101_FAKE_ARCHIVE_FOR_TEST" + bytes(256)
    composed = compose_with(dp1_bytes, pr101_bytes, base_substrate="pr101")
    decomposed = decompose(composed)
    assert decomposed.base_substrate == "pr101"
    assert decomposed.base_archive_bytes == pr101_bytes
    # DP1 archive must parse identically (same num_pairs / out_h / out_w).
    assert decomposed.dp1_archive.num_pairs == 4
    assert decomposed.dp1_archive.output_height == 64
    assert decomposed.dp1_archive.output_width == 96


def test_decompose_empty_base_bytes_accepted():
    """Edge case: composition with zero-length base archive (degenerate)."""
    dp1_bytes = _make_dp1_archive_bytes()
    composed = compose_with(dp1_bytes, b"", base_substrate="hdm8")
    decomposed = decompose(composed)
    assert decomposed.base_archive_bytes == b""
    assert decomposed.base_substrate == "hdm8"


# --- Section 4: validation refusals -----------------------------------------


def test_compose_with_unknown_base_substrate_raises():
    dp1_bytes = _make_dp1_archive_bytes()
    with pytest.raises(ValueError, match="unknown base_substrate"):
        compose_with(dp1_bytes, b"", base_substrate="bogus_substrate_name")


def test_compose_with_invalid_dp1_bytes_raises():
    """Composition fails fast if DP1 bytes are corrupted (per CLAUDE.md
    fail-fast at every boundary)."""
    with pytest.raises(ValueError):
        compose_with(b"NOT_DP1_BYTES", b"", base_substrate="a1")


def test_decompose_short_bytes_raises():
    with pytest.raises(ValueError, match="too short for header"):
        decompose(b"\x00\x01")


def test_decompose_wrong_magic_raises():
    bad = b"XXXX" + bytes(DPCOMP_HEADER_SIZE)
    with pytest.raises(ValueError, match="magic mismatch"):
        decompose(bad)


def test_decompose_wrong_version_raises():
    """Future version bump path: refuse decode of unknown schema_version."""
    import struct

    bad_header = struct.pack(
        "<4sBI4s", DPCOMP_MAGIC, 99, 0, b"A1\x00\x00"
    )
    with pytest.raises(ValueError, match="schema version"):
        decompose(bad_header)


def test_decompose_unknown_base_tag_raises():
    import struct

    bad_header = struct.pack(
        "<4sBI4s", DPCOMP_MAGIC, DPCOMP_SCHEMA_VERSION, 0, b"ZZZZ"
    )
    with pytest.raises(ValueError, match="base_tag"):
        decompose(bad_header)


def test_decompose_truncated_dp1_blob_raises():
    """Header declares a DP1 blob length but the bytes are missing."""
    import struct

    bad_header = struct.pack(
        "<4sBI4s", DPCOMP_MAGIC, DPCOMP_SCHEMA_VERSION, 10_000, b"A1\x00\x00"
    )
    with pytest.raises(ValueError, match="truncated"):
        decompose(bad_header)


# --- Section 5: verify_composition forensic surface -------------------------


def test_verify_composition_returns_canonical_report():
    dp1_bytes = _make_dp1_archive_bytes(num_pairs=8)
    base_bytes = b"FAKE_BASE" + bytes(64)
    composed = compose_with(dp1_bytes, base_bytes, base_substrate="yucr")
    report = verify_composition(composed)
    assert report["base_substrate"] == "yucr"
    assert report["composed_total_bytes"] == len(composed)
    assert report["dp1_blob_bytes"] == len(dp1_bytes)
    assert report["base_blob_bytes"] == len(base_bytes)
    assert report["num_pairs"] == 8
    assert report["output_height"] == 64
    assert report["output_width"] == 96
    assert report["schema_version"] == DPCOMP_SCHEMA_VERSION
    assert isinstance(report["dp1_license_tags"], list)
    assert "synthetic-test-only" in report["dp1_license_tags"]
    assert report["dp1_dataset_provenance"] == "synthetic_test"
    assert len(report["dp1_basis_sha256"]) == 64  # SHA-256 hex digest


def test_verify_composition_with_expected_substrate_passes():
    dp1_bytes = _make_dp1_archive_bytes()
    composed = compose_with(dp1_bytes, b"BASE", base_substrate="sane_hnerv")
    report = verify_composition(
        composed, expected_base_substrate="sane_hnerv"
    )
    assert report["base_substrate"] == "sane_hnerv"


def test_verify_composition_with_wrong_expected_substrate_raises():
    dp1_bytes = _make_dp1_archive_bytes()
    composed = compose_with(dp1_bytes, b"BASE", base_substrate="a1")
    with pytest.raises(ValueError, match="base_substrate"):
        verify_composition(composed, expected_base_substrate="pr101")


def test_verify_composition_detects_codebook_tampering():
    """If the DP1 basis_sha256 changes between distill and verify, the
    forensic surface MUST detect it.

    This is the operator's load-bearing assurance that future federated
    composition cannot smuggle a tampered prior into a downstream archive.
    """
    dp1_bytes = _make_dp1_archive_bytes()
    composed = compose_with(dp1_bytes, b"BASE", base_substrate="time_traveler_l5")
    # Recover the legitimate basis_sha256 from a verify pass.
    legit_report = verify_composition(composed)
    legit_sha = legit_report["dp1_basis_sha256"]
    assert isinstance(legit_sha, str) and len(legit_sha) == 64

    # Same composition should pass against its own basis_sha256.
    verify_composition(
        composed, expected_dp1_basis_sha256=legit_sha
    )

    # A different expected SHA should fail (proves we're checking it).
    fake_sha = "0" * 64
    with pytest.raises(ValueError, match="basis_sha256"):
        verify_composition(
            composed, expected_dp1_basis_sha256=fake_sha
        )


# --- Section 6: license propagation -----------------------------------------


def test_license_tags_propagate_to_verification_report():
    """Catalog #210 sister test: license attribution flows from DP1 metadata
    into the composition forensic report so downstream replay can audit it."""
    dp1_bytes = _make_dp1_archive_bytes()
    composed = compose_with(dp1_bytes, b"BASE", base_substrate="a1")
    report = verify_composition(composed)
    tags = report["dp1_license_tags"]
    assert isinstance(tags, list)
    # Synthetic codebook -> ["synthetic-test-only"]; real Comma2k19 would be
    # ["comma2k19:MIT", "github.com/commaai/comma2k19"]. Either is acceptable
    # by license-tag policy; the test pins that the field is non-empty.
    assert len(tags) >= 1


# --- Section 7: compose_from_files end-to-end ------------------------------


def test_compose_from_files_roundtrip(tmp_path: Path):
    dp1_bytes = _make_dp1_archive_bytes()
    base_bytes = b"FAKE_BASE_FROM_DISK" + bytes(128)
    dp1_path = tmp_path / "dp1.bin"
    base_path = tmp_path / "base.bin"
    out_path = tmp_path / "composed.bin"
    dp1_path.write_bytes(dp1_bytes)
    base_path.write_bytes(base_bytes)
    report = compose_from_files(
        dp1_path, base_path, out_path, base_substrate="hdm8"
    )
    assert out_path.is_file()
    assert report["base_substrate"] == "hdm8"
    assert report["composed_total_bytes"] == out_path.stat().st_size
    # Sidecar must exist with byte-stable JSON.
    sidecar = out_path.with_suffix(out_path.suffix + ".meta.json")
    assert sidecar.is_file()
    parsed_sidecar = json.loads(sidecar.read_text())
    assert parsed_sidecar["base_substrate"] == "hdm8"
    # The composed_sha256 is recorded in the returned report (added after
    # sidecar write so the sidecar JSON doesn't depend on its own contents).
    assert "composed_sha256" in report
    assert (
        hashlib.sha256(out_path.read_bytes()).hexdigest()
        == report["composed_sha256"]
    )


def test_compose_from_files_refuses_tmp_paths(tmp_path: Path):
    """Per CLAUDE.md 'Forbidden /tmp paths in any persisted artifact'."""
    dp1_bytes = _make_dp1_archive_bytes()
    dp1_path = tmp_path / "dp1.bin"
    base_path = tmp_path / "base.bin"
    dp1_path.write_bytes(dp1_bytes)
    base_path.write_bytes(b"")
    # Explicit /tmp/ output path triggers refusal.
    forbidden_out = Path("/tmp/dp1_composed_should_refuse.bin")
    with pytest.raises(ValueError, match="transient path"):
        compose_from_files(
            dp1_path, base_path, forbidden_out, base_substrate="a1"
        )


# --- Section 8: cross-substrate composition matrix -------------------------


def test_dp1_x_a1_composition_byte_stable():
    """DP1 × A1 (sole verified sub-0.20 anchor) — pin byte-stable composition."""
    dp1_bytes = _make_dp1_archive_bytes(seed=0xA1)
    a1_bytes = b"A1_SUBSTRATE_BYTES_FIXTURE" + bytes(256)
    composed_1 = compose_with(dp1_bytes, a1_bytes, base_substrate="a1")
    composed_2 = compose_with(dp1_bytes, a1_bytes, base_substrate="a1")
    assert composed_1 == composed_2
    sha_1 = hashlib.sha256(composed_1).hexdigest()
    sha_2 = hashlib.sha256(composed_2).hexdigest()
    assert sha_1 == sha_2


def test_dp1_x_pr101_composition_byte_stable():
    """DP1 × PR101 (gold 0.193 baseline) — pin byte-stable composition."""
    dp1_bytes = _make_dp1_archive_bytes(seed=0xC2)
    pr101_bytes = b"PR101_SUBSTRATE_BYTES_FIXTURE" + bytes(512)
    composed_1 = compose_with(dp1_bytes, pr101_bytes, base_substrate="pr101")
    composed_2 = compose_with(dp1_bytes, pr101_bytes, base_substrate="pr101")
    assert composed_1 == composed_2


def test_dp1_x_time_traveler_l5_composition_round_robin():
    """DP1 × TT5L declared composition in L1 scaffold — pin roundtrip."""
    dp1_bytes = _make_dp1_archive_bytes(seed=0x715)
    tt5l_bytes = b"TT5L_WORLD_MODEL_FIXTURE" + bytes(128)
    composed = compose_with(dp1_bytes, tt5l_bytes, base_substrate="time_traveler_l5")
    parsed = decompose(composed)
    assert parsed.base_substrate == "time_traveler_l5"
    assert parsed.base_archive_bytes == tt5l_bytes


# --- Section 9: cross-seed reproducibility (Phase 2 hardening) -------------


def test_cross_seed_codebook_distillation_quality_band():
    """Different seeds should produce DIFFERENT codebooks but with the same
    quality band (basis_sha256 differs, but provenance metadata structure
    remains stable).

    Per B.2 distillation quality metric hardening.
    """
    cfg_a = DistillationConfig(
        dataset_name="synthetic_test",
        random_seed=1,
        max_frames=32,
    )
    cfg_b = DistillationConfig(
        dataset_name="synthetic_test",
        random_seed=2,
        max_frames=32,
    )
    iter_a = Comma2k19FrameIterator(synthetic=True, n_frames=32, seed=1)
    iter_b = Comma2k19FrameIterator(synthetic=True, n_frames=32, seed=2)
    book_a = distill_codebook(cfg_a, frames=iter(iter_a))
    book_b = distill_codebook(cfg_b, frames=iter(iter_b))
    # Different seed -> different basis bytes.
    assert book_a.metadata["basis_sha256"] != book_b.metadata["basis_sha256"]
    # Same structural metadata keys + provenance.
    assert book_a.metadata["dataset_provenance"] == book_b.metadata["dataset_provenance"]
    assert book_a.metadata["distillation_version"] == book_b.metadata["distillation_version"]
    assert sorted(book_a.metadata.keys()) == sorted(book_b.metadata.keys())


def test_cross_run_codebook_byte_identical_for_same_seed():
    """Same seed + same data -> bit-identical codebook (deterministic
    reproducibility).

    Per B.2 distillation quality metric + B.5 cross-machine reproducibility.
    """
    cfg = DistillationConfig(
        dataset_name="synthetic_test",
        random_seed=42,
        max_frames=32,
    )
    iter_1 = Comma2k19FrameIterator(synthetic=True, n_frames=32, seed=42)
    iter_2 = Comma2k19FrameIterator(synthetic=True, n_frames=32, seed=42)
    book_1 = distill_codebook(cfg, frames=iter(iter_1))
    book_2 = distill_codebook(cfg, frames=iter(iter_2))
    # Bit-identical: basis bytes match, sha256 matches, scale matches.
    assert np.array_equal(book_1.road_plane_basis, book_2.road_plane_basis)
    assert np.array_equal(book_1.sky_horizon_profile, book_2.sky_horizon_profile)
    assert book_1.metadata["basis_sha256"] == book_2.metadata["basis_sha256"]
    assert book_1.metadata["road_plane_scale"] == book_2.metadata["road_plane_scale"]
