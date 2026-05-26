# SPDX-License-Identifier: MIT
"""L0 SCAFFOLD smoke + shape + Catalog #139 byte-mutation tests.

Per Catalog #91 ENCODE_INFLATE_ROUNDTRIP + Catalog #139 byte-mutation
no_op_proof + Catalog #240 L0 SCAFFOLD posture verification.

These tests verify the substrate package's STRUCTURAL invariants without
requiring MLX to be installed (MLX-dependent tests are guarded by import
detection and skipped when MLX is unavailable).
"""

from __future__ import annotations

import hashlib

import pytest


def test_module_imports_without_mlx() -> None:
    """Top-level package import must succeed without MLX installed."""
    import tac.substrates.boost_nerv_pr110_residual as mod

    assert mod.ARCHIVE_MAGIC == b"BPR1\x00"
    assert mod.ARCHIVE_VERSION == 1
    assert mod.BPR1_HEADER_LEN == 29
    assert mod.DEFAULT_NUM_BOOSTING_ROUNDS == 1
    assert mod.DEFAULT_RESIDUAL_BUDGET_BYTES == 8192


def test_module_exposes_canonical_public_api() -> None:
    """__all__ surface must be narrow + explicit per Catalog #335 contract."""
    import tac.substrates.boost_nerv_pr110_residual as mod

    expected = {
        "ARCHIVE_MAGIC",
        "ARCHIVE_VERSION",
        "BPR1_HEADER_FMT",
        "BPR1_HEADER_LEN",
        "DEFAULT_NUM_BOOSTING_ROUNDS",
        "DEFAULT_RESIDUAL_BUDGET_BYTES",
        "BoostNervPr110ResidualConfig",
    }
    assert set(mod.__all__) == expected


def test_config_dataclass_lazy_loads_via_getattr() -> None:
    """Catalog #168 AnnAssign discipline + lazy MLX import: BoostNervPr110ResidualConfig
    must be accessible via module attribute without forcing MLX at import time."""
    import tac.substrates.boost_nerv_pr110_residual as mod
    from tac.substrates.boost_nerv_pr110_residual.architecture import (
        BoostNervPr110ResidualConfig,
    )

    cfg = mod.BoostNervPr110ResidualConfig()
    assert isinstance(cfg, BoostNervPr110ResidualConfig)
    assert cfg.pr110_latent_dim == 24
    assert cfg.residual_hidden_dim == 12
    assert cfg.num_boosting_rounds == 1
    assert cfg.boosting_gain_clamp == 0.05
    assert cfg.residual_spatial_h == 96
    assert cfg.residual_spatial_w == 128
    assert cfg.residual_quant_bits == 8


def test_curriculum_canonical_5_plus_1_stages_declared() -> None:
    """Catalog #305 observability surface: curriculum is machine-readable at rest."""
    from tac.substrates.boost_nerv_pr110_residual.boosting_curriculum import (
        CANONICAL_CURRICULUM,
    )

    assert len(CANONICAL_CURRICULUM.stages) == 6  # 5 canonical + 1 optional L1+
    stage_ids = [s.stage_id for s in CANONICAL_CURRICULUM.stages]
    assert stage_ids == [0, 1, 2, 3, 4, 5]

    # Stage 0 (PR110 extraction) is NOT MLX-trained.
    assert CANONICAL_CURRICULUM.stage_by_id(0).mlx_implementable is False
    # Stage 1 (residual targets) is NOT MLX-trained.
    assert CANONICAL_CURRICULUM.stage_by_id(1).mlx_implementable is False
    # Stage 2 (warm-up) IS MLX-trained.
    assert CANONICAL_CURRICULUM.stage_by_id(2).mlx_implementable is True
    # Stage 3 (score-aware fine-tune) IS MLX-trained.
    assert CANONICAL_CURRICULUM.stage_by_id(3).mlx_implementable is True
    # Stage 4 (archive build + Catalog #1265 gate) is NOT MLX-trained (uses subprocess).
    assert CANONICAL_CURRICULUM.stage_by_id(4).mlx_implementable is False
    # Stage 5 (optional round 2) IS MLX-trained (when reactivated at L1+).
    assert CANONICAL_CURRICULUM.stage_by_id(5).mlx_implementable is True


def test_curriculum_total_wallclock_estimate_excludes_optional_by_default() -> None:
    """L0 wallclock budget excludes round-2 optional stage."""
    from tac.substrates.boost_nerv_pr110_residual.boosting_curriculum import (
        CANONICAL_CURRICULUM,
    )

    without_optional = CANONICAL_CURRICULUM.total_wallclock_seconds_estimate(
        include_optional=False
    )
    with_optional = CANONICAL_CURRICULUM.total_wallclock_seconds_estimate(
        include_optional=True
    )
    assert with_optional > without_optional
    # Stage 5 estimate is 1200 s; difference matches.
    assert with_optional - without_optional == pytest.approx(1200.0, rel=1e-6)


def test_full_main_stub_raises_per_catalog_240() -> None:
    """L0 SCAFFOLD posture: _full_main_stub_raises must refuse dispatch."""
    from tac.substrates.boost_nerv_pr110_residual.boosting_curriculum import (
        _full_main_stub_raises,
    )

    with pytest.raises(NotImplementedError, match="L0 SCAFFOLD per Catalog #240"):
        _full_main_stub_raises()


def test_bpr1_header_pack_round_trip() -> None:
    """Catalog #91 ENCODE_INFLATE_ROUNDTRIP: pack + parse = identity."""
    from tac.substrates.boost_nerv_pr110_residual.archive import (
        pack_bpr1_header,
        parse_bpr1_header,
    )

    sha_prefix = bytes(range(16))
    packed = pack_bpr1_header(
        num_boosting_rounds=2,
        pr110_sha256_prefix=sha_prefix,
        residual_blob_len=12345,
    )
    assert len(packed) == 29

    parsed = parse_bpr1_header(packed)
    assert parsed.magic == b"BPR1\x00"
    assert parsed.version == 1
    assert parsed.num_boosting_rounds == 2
    assert parsed.pr110_sha256_prefix == sha_prefix
    assert parsed.residual_blob_len == 12345


def test_bpr1_header_invalid_magic_rejected() -> None:
    """Mis-magic'd sidecar must be refused at parse time."""
    from tac.substrates.boost_nerv_pr110_residual.archive import parse_bpr1_header

    bad_buf = b"NOPE\x00" + bytes(24)
    with pytest.raises(ValueError, match="BPR1 magic mismatch"):
        parse_bpr1_header(bad_buf)


def test_bpr1_header_invalid_version_rejected() -> None:
    """Wrong version must be refused at parse time."""
    from tac.substrates.boost_nerv_pr110_residual.archive import (
        pack_bpr1_header,
        parse_bpr1_header,
    )
    import struct
    from tac.substrates.boost_nerv_pr110_residual import (
        ARCHIVE_MAGIC,
        BPR1_HEADER_FMT,
    )

    bad_buf = struct.pack(
        BPR1_HEADER_FMT,
        ARCHIVE_MAGIC,
        99,  # bad version
        1,
        0,
        bytes(16),
        0,
        0,
    )
    with pytest.raises(ValueError, match="BPR1 version mismatch"):
        parse_bpr1_header(bad_buf)


def test_pack_rejects_bad_num_rounds() -> None:
    """num_boosting_rounds must be in [1, 4]."""
    from tac.substrates.boost_nerv_pr110_residual.archive import pack_bpr1_header

    with pytest.raises(ValueError, match="num_boosting_rounds must be in"):
        pack_bpr1_header(0, bytes(16), 0)
    with pytest.raises(ValueError, match="num_boosting_rounds must be in"):
        pack_bpr1_header(5, bytes(16), 0)


def test_pack_rejects_bad_sha_prefix_length() -> None:
    """SHA prefix must be exactly 16 bytes."""
    from tac.substrates.boost_nerv_pr110_residual.archive import pack_bpr1_header

    with pytest.raises(ValueError, match="must be exactly 16 bytes"):
        pack_bpr1_header(1, bytes(15), 0)
    with pytest.raises(ValueError, match="must be exactly 16 bytes"):
        pack_bpr1_header(1, bytes(17), 0)


def test_compose_archive_binds_to_pr110_sha256() -> None:
    """Compose computes sha256 prefix from the actual PR110 base bytes."""
    from tac.substrates.boost_nerv_pr110_residual.archive import (
        compose_archive,
        parse_bpr1_header,
    )

    pr110_base = b"\xde\xad\xbe\xef" * 1024  # 4 KB synthetic base
    residual = b"\x00\x01" * 100  # 200-byte synthetic residual
    composed = compose_archive(
        pr110_base_archive_bytes=pr110_base,
        residual_blob=residual,
        num_boosting_rounds=1,
    )

    expected_sha_prefix = hashlib.sha256(pr110_base).digest()[:16]
    header = parse_bpr1_header(composed)
    assert header.pr110_sha256_prefix == expected_sha_prefix
    assert header.residual_blob_len == 200
    assert header.num_boosting_rounds == 1


def test_split_composed_archive_round_trip() -> None:
    """Catalog #91 ENCODE_INFLATE_ROUNDTRIP: compose + split = identity (excluding header)."""
    from tac.substrates.boost_nerv_pr110_residual.archive import (
        compose_archive,
        split_composed_archive,
    )

    pr110_base = b"PR110\x00BASE" * 200  # 2000 bytes
    residual = b"RESIDUAL_DATA" * 50  # 650 bytes
    composed = compose_archive(
        pr110_base_archive_bytes=pr110_base,
        residual_blob=residual,
        num_boosting_rounds=2,
    )

    header, recovered_residual, recovered_base = split_composed_archive(composed)
    assert header.num_boosting_rounds == 2
    assert recovered_residual == residual
    assert recovered_base == pr110_base


def test_split_refuses_non_matching_pr110_base() -> None:
    """Catalog #139 byte-mutation discipline: mutated PR110 base must be refused.

    The sha_prefix binding is the structural-extinction primitive that
    prevents the sidecar from being silently mis-applied to a non-PR110 base.
    """
    from tac.substrates.boost_nerv_pr110_residual.archive import (
        compose_archive,
        split_composed_archive,
    )

    pr110_base = b"PR110\x00BASE" * 200
    residual = b"RESIDUAL_DATA" * 50
    composed = compose_archive(
        pr110_base_archive_bytes=pr110_base,
        residual_blob=residual,
        num_boosting_rounds=1,
    )

    # Mutate one byte in the PR110 base section (after header + residual).
    composed_array = bytearray(composed)
    from tac.substrates.boost_nerv_pr110_residual import BPR1_HEADER_LEN
    base_start = BPR1_HEADER_LEN + len(residual)
    composed_array[base_start] ^= 0x01  # flip one bit
    mutated = bytes(composed_array)

    with pytest.raises(ValueError, match="PR110 base archive sha256 prefix mismatch"):
        split_composed_archive(mutated)


def test_byte_mutation_no_op_proof_per_catalog_139() -> None:
    """Mutating the residual blob changes the parsed header's blob bytes
    (downstream forward pass would consume different residual weights →
    different output frames). The no_op_proof here is at the archive layer."""
    from tac.substrates.boost_nerv_pr110_residual.archive import (
        compose_archive,
        split_composed_archive,
    )

    pr110_base = b"PR110_BASE_CONST" * 100
    residual = b"\x10\x20\x30\x40" * 50  # 200 bytes
    composed = compose_archive(
        pr110_base_archive_bytes=pr110_base,
        residual_blob=residual,
        num_boosting_rounds=1,
    )

    # Mutate one byte in the residual blob region.
    composed_array = bytearray(composed)
    from tac.substrates.boost_nerv_pr110_residual import BPR1_HEADER_LEN
    composed_array[BPR1_HEADER_LEN] ^= 0xFF  # flip byte 0 of residual
    mutated = bytes(composed_array)

    # Header still parses (sha_prefix unchanged); residual blob differs.
    _, mutated_residual, _ = split_composed_archive(mutated)
    assert mutated_residual != residual
    assert mutated_residual[0] != residual[0]


def test_write_composed_archive_to_zip_deterministic(tmp_path) -> None:
    """Catalog #19 deterministic ZIP: two writes produce byte-identical archives."""
    from tac.substrates.boost_nerv_pr110_residual.archive import (
        compose_archive,
        write_composed_archive_to_zip,
    )

    pr110_base = b"PR110_BYTES_FOR_DETERMINISM_TEST" * 50
    residual = b"RESIDUAL_BYTES_FOR_DETERMINISM" * 10
    composed = compose_archive(
        pr110_base_archive_bytes=pr110_base,
        residual_blob=residual,
        num_boosting_rounds=1,
    )

    zip_a = tmp_path / "archive_a.zip"
    zip_b = tmp_path / "archive_b.zip"
    write_composed_archive_to_zip(composed, zip_a)
    write_composed_archive_to_zip(composed, zip_b)

    assert zip_a.read_bytes() == zip_b.read_bytes()
    # SHA-stable across runs.
    sha_a = hashlib.sha256(zip_a.read_bytes()).hexdigest()
    sha_b = hashlib.sha256(zip_b.read_bytes()).hexdigest()
    assert sha_a == sha_b


def test_residual_extraction_plan_refuses_tmp_paths(tmp_path) -> None:
    """Catalog #113 'Forbidden /tmp paths' discipline at construction time."""
    from pathlib import Path

    from tac.substrates.boost_nerv_pr110_residual.residual_extraction import (
        Pr110BaseExtractionPlan,
    )

    # Build a fake-but-real archive zip + inflate sh + video at tmp_path
    # (Pr110BaseExtractionPlan refuses non-existent paths first).
    fake_archive = tmp_path / "archive.zip"
    fake_archive.write_bytes(b"fake_zip_bytes")
    fake_inflate = tmp_path / "inflate.sh"
    fake_inflate.write_text("#!/bin/bash")
    fake_video = tmp_path / "0.mkv"
    fake_video.write_bytes(b"fake_video_bytes")

    # cache_root pointed at /tmp/foo → refused.
    with pytest.raises(ValueError, match="must not be under /tmp/"):
        Pr110BaseExtractionPlan(
            pr110_archive_zip=fake_archive,
            pr110_inflate_sh=fake_inflate,
            cache_root=Path("/tmp/foo"),
            upstream_video_path=fake_video,
        )


def test_residual_extraction_plan_refuses_missing_files() -> None:
    """Catalog #229 PV at construction time."""
    from pathlib import Path

    from tac.substrates.boost_nerv_pr110_residual.residual_extraction import (
        Pr110BaseExtractionPlan,
    )

    with pytest.raises(FileNotFoundError, match="PR110 archive.zip not found"):
        Pr110BaseExtractionPlan(
            pr110_archive_zip=Path("/nonexistent/archive.zip"),
            pr110_inflate_sh=Path("/nonexistent/inflate.sh"),
            cache_root=Path("/Users/runner/.omx/state/cache"),
            upstream_video_path=Path("/nonexistent/0.mkv"),
        )


def test_diagnose_residual_target_magnitude_verdicts() -> None:
    """Stage 1 diagnostic verdict taxonomy per design memo."""
    import numpy as np

    from tac.substrates.boost_nerv_pr110_residual.residual_extraction import (
        diagnose_residual_target_magnitude,
    )

    # Case A: near-zero residual (PR110 already near-optimal) → DEFER.
    zero_residual_0 = np.zeros((10, 3, 96, 128), dtype=np.float32)
    zero_residual_1 = np.zeros((10, 3, 96, 128), dtype=np.float32)
    diag_zero = diagnose_residual_target_magnitude(zero_residual_0, zero_residual_1)
    assert "NEAR_OPTIMAL" in diag_zero["verdict"]
    assert diag_zero["score_claim"] is False
    assert diag_zero["promotion_eligible"] is False
    assert diag_zero["axis_tag"] == "[macOS-MLX research-signal]"

    # Case B: large residuals (PR110 has headroom) → PROCEED.
    rng = np.random.default_rng(42)
    large_residual_0 = (rng.random((10, 3, 96, 128)).astype(np.float32) - 0.5) * 0.4
    large_residual_1 = (rng.random((10, 3, 96, 128)).astype(np.float32) - 0.5) * 0.4
    diag_large = diagnose_residual_target_magnitude(large_residual_0, large_residual_1)
    assert "PROCEED_TO_STAGE_2" in diag_large["verdict"]
    assert diag_large["residual_magnitude_p99_rgb_range"] >= 0.05


def test_diagnose_rejects_shape_mismatch() -> None:
    """Per-pair residual targets must have matching shapes."""
    import numpy as np

    from tac.substrates.boost_nerv_pr110_residual.residual_extraction import (
        extract_per_pair_residual_targets,
    )

    a = np.zeros((10, 3, 96, 128), dtype=np.float32)
    b = np.zeros((10, 3, 96, 128), dtype=np.float32)
    c = np.zeros((11, 3, 96, 128), dtype=np.float32)  # bad shape
    with pytest.raises(ValueError, match="shape mismatch"):
        extract_per_pair_residual_targets(a, b, c, b)


def test_write_diagnostic_manifest_rejects_score_claim(tmp_path) -> None:
    """Catalog #127 + #341: diagnostic must carry non-promotable markers."""
    from tac.substrates.boost_nerv_pr110_residual.residual_extraction import (
        write_diagnostic_manifest,
    )

    bad_diag = {
        "residual_magnitude_p50_rgb_range": 0.01,
        "score_claim": True,  # FORBIDDEN per Catalog #127
        "promotion_eligible": False,
        "axis_tag": "[macOS-MLX research-signal]",
    }
    with pytest.raises(ValueError, match="score_claim=False"):
        write_diagnostic_manifest(bad_diag, tmp_path / "diag.json")


def test_num_residual_parameters_matches_paper_calculation() -> None:
    """Param count = sanity check on design memo §9 EXTREME OPTIMIZATION row."""
    from tac.substrates.boost_nerv_pr110_residual.architecture import (
        BoostNervPr110ResidualConfig,
        num_residual_parameters,
    )

    cfg = BoostNervPr110ResidualConfig()
    n = num_residual_parameters(cfg)
    # Hand calculation for default config:
    # z_proj: 24 * 12 + 12 = 300
    # conv1: (3 + 12) * 12 * 9 + 12 = 1632
    # conv2: 12 * 3 * 1 + 3 = 39
    # total per-round: 300 + 1632 + 39 = 1971
    # num_rounds=1: 1971
    assert n == 1971


def test_num_residual_parameters_scales_with_rounds() -> None:
    """Per-round residual head; multi-round multiplies parameter count."""
    from dataclasses import replace

    from tac.substrates.boost_nerv_pr110_residual.architecture import (
        BoostNervPr110ResidualConfig,
        num_residual_parameters,
    )

    cfg_1 = BoostNervPr110ResidualConfig(num_boosting_rounds=1)
    cfg_3 = replace(cfg_1, num_boosting_rounds=3)
    assert num_residual_parameters(cfg_3) == 3 * num_residual_parameters(cfg_1)


@pytest.mark.skipif(
    not pytest.importorskip("mlx.core", reason="MLX not installed"),
    reason="MLX-dependent test",
)
def test_residual_head_mlx_forward_shape() -> None:
    """MLX residual head forward pass produces correct shape.

    Only runs when MLX is installed (skipped on non-Apple-Silicon CI).
    """
    import mlx.core as mx

    from tac.substrates.boost_nerv_pr110_residual.architecture import (
        BoostNervPr110ResidualConfig,
        ResidualHeadMLX,
    )

    cfg = BoostNervPr110ResidualConfig()
    head = ResidualHeadMLX(cfg)
    # MLX-native NHWC convention.
    rgb_base = mx.zeros((2, cfg.residual_spatial_h, cfg.residual_spatial_w, 3))
    z_pr110 = mx.zeros((2, cfg.pr110_latent_dim))
    residual = head.forward(rgb_base, z_pr110)
    assert residual.shape == (2, cfg.residual_spatial_h, cfg.residual_spatial_w, 3)


@pytest.mark.skipif(
    not pytest.importorskip("mlx.core", reason="MLX not installed"),
    reason="MLX-dependent test",
)
def test_compose_pr110_base_plus_residual_clamps_correctly() -> None:
    """Canonical composition: bounded output in [0, 1]."""
    import mlx.core as mx

    from tac.substrates.boost_nerv_pr110_residual.architecture import (
        compose_pr110_base_plus_residual,
    )

    # NHWC single-pixel: (B=1, H=1, W=1, C=1)
    rgb_base = mx.array([[[[0.5]]]], dtype=mx.float32)
    # Massive residual that should be clamped to ±gain then composed +
    # final clamp to [0, 1].
    residual_huge_positive = mx.array([[[[10.0]]]], dtype=mx.float32)
    composed = compose_pr110_base_plus_residual(
        rgb_base, residual_huge_positive, gain_clamp=0.05
    )
    # 0.5 + clip(10.0, -0.05, 0.05) = 0.5 + 0.05 = 0.55, in [0, 1].
    assert float(composed[0, 0, 0, 0]) == pytest.approx(0.55, abs=1e-5)

    residual_huge_negative = mx.array([[[[-10.0]]]], dtype=mx.float32)
    composed = compose_pr110_base_plus_residual(
        rgb_base, residual_huge_negative, gain_clamp=0.05
    )
    assert float(composed[0, 0, 0, 0]) == pytest.approx(0.45, abs=1e-5)
