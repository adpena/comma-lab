# SPDX-License-Identifier: MIT
"""D1 substrate test suite.

Per CLAUDE.md "Beauty, simplicity, and developer experience" + "Apples-to-
apples evidence discipline" + Catalog #124/#146/#157/#164/#192. NO score
claims. NO /tmp paths. Every reconstruction-error tag is ``[proxy]``.
"""

from __future__ import annotations

import hashlib
import math
import subprocess
import sys

import numpy as np
import pytest
import torch

from tac.substrates.d1_segnet_margin_polytope import (
    D1POLY1_HEADER_FMT,
    D1POLY1_HEADER_SIZE,
    D1POLY1_MAGIC,
    D1POLY1_SCHEMA_VERSION,
    D1POLY_DEFAULT_BASE_SUBSTRATE,
    POLYTOPE_LATTICE_LEVELS,
    POLYTOPE_LATTICE_VALUES,
    D1PolytopeConfig,
    D1PolytopeLossWeights,
    D1PolytopeSidecar,
    MarginMapMode,
    allocate_noise_within_polytope,
    build_readiness_manifest,
    compute_logit_margin_map_dummy,
    compute_safe_perturbation_budget,
    decode_polytope_payload,
    dequantize_margin_map_int8,
    encode_polytope_payload,
    pack_archive,
    parse_archive,
    quantize_margin_map_int8,
)
from tac.substrates.d1_segnet_margin_polytope.architecture import (
    D1POLY_OVERHEAD_TARGET_BYTES_MAX,
    D1POLY_OVERHEAD_TARGET_BYTES_MIN,
    _BaseArchiveDescriptor,
    compose_with_base,
    estimate_overhead_bytes,
)

# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_d1_config_defaults_are_self_consistent():
    cfg = D1PolytopeConfig()
    assert cfg.base_substrate_id == D1POLY_DEFAULT_BASE_SUBSTRATE
    assert cfg.margin_map_mode == "segnet_top1_minus_top2"
    assert cfg.polytope_payload_bits == 8000
    assert cfg.margin_map_resolution == (384, 512)
    assert math.isclose(cfg.pose_sqrt_weight, math.sqrt(10.0))
    assert math.isclose(cfg.seg_weight, 100.0)
    assert cfg.jacobian_lipschitz > 0
    assert cfg.margin_threshold > 0


def test_d1_config_rejects_unknown_base():
    with pytest.raises(ValueError, match="not in"):
        D1PolytopeConfig(base_substrate_id="some_unknown_substrate")


def test_d1_config_rejects_unknown_margin_mode():
    with pytest.raises(ValueError, match="margin_map_mode"):
        D1PolytopeConfig(margin_map_mode="moonbeam")


def test_d1_config_rejects_non_positive_lipschitz():
    with pytest.raises(ValueError, match="jacobian_lipschitz"):
        D1PolytopeConfig(jacobian_lipschitz=0.0)
    with pytest.raises(ValueError, match="jacobian_lipschitz"):
        D1PolytopeConfig(jacobian_lipschitz=-1.0)


def test_d1_config_rejects_huge_lipschitz():
    with pytest.raises(ValueError, match="jacobian_lipschitz"):
        D1PolytopeConfig(jacobian_lipschitz=2000.0)


def test_d1_config_rejects_zero_budget_bits():
    with pytest.raises(ValueError, match="polytope_payload_bits"):
        D1PolytopeConfig(polytope_payload_bits=0)


def test_d1_config_rejects_negative_margin_threshold():
    with pytest.raises(ValueError, match="margin_threshold"):
        D1PolytopeConfig(margin_threshold=-0.1)


# ---------------------------------------------------------------------------
# Margin map (dummy mode for tests; real mode covered downstream)
# ---------------------------------------------------------------------------


def test_margin_map_dummy_returns_correct_shape_and_value():
    mm = compute_logit_margin_map_dummy(resolution=(48, 64), constant_value=2.5)
    assert mm.shape == (48, 64)
    assert mm.dtype == torch.float32
    assert torch.all(mm == 2.5)


def test_margin_map_dummy_rejects_non_positive_constant():
    with pytest.raises(ValueError, match="constant_value"):
        compute_logit_margin_map_dummy(constant_value=0.0)


def test_quantize_margin_map_int8_roundtrip():
    mm = torch.tensor([[0.5, 2.0, 3.0], [1.5, 0.0, 0.25]])
    int8, scale = quantize_margin_map_int8(mm, scale=127.0)
    assert int8.dtype == torch.int8
    assert int8.shape == mm.shape
    assert scale > 0
    # int8 values must be non-negative (margin is non-negative).
    assert torch.all(int8 >= 0)
    recovered = dequantize_margin_map_int8(int8, recovered_scale=scale)
    err = (mm - recovered).abs().max().item()
    assert err < 0.05, f"int8 quantize error {err} too large"


def test_quantize_margin_map_int8_rejects_negative_values():
    mm = torch.tensor([[1.0, -0.1], [0.5, 2.0]])
    with pytest.raises(ValueError, match="negative"):
        quantize_margin_map_int8(mm, scale=127.0)


def test_quantize_margin_map_int8_rejects_bad_scale():
    mm = torch.ones(8, 8)
    with pytest.raises(ValueError, match="scale"):
        quantize_margin_map_int8(mm, scale=0.0)
    with pytest.raises(ValueError, match="scale"):
        quantize_margin_map_int8(mm, scale=200.0)


def test_dequantize_margin_rejects_non_positive_scale():
    int8 = torch.zeros(4, 4, dtype=torch.int8)
    with pytest.raises(ValueError, match="recovered_scale"):
        dequantize_margin_map_int8(int8, recovered_scale=-1.0)


def test_margin_map_mode_enum_values():
    assert MarginMapMode.SEGNET_TOP1_MINUS_TOP2.value == "segnet_top1_minus_top2"
    assert MarginMapMode.UNIFORM.value == "uniform"
    assert MarginMapMode.DUMMY_CONSTANT.value == "dummy_constant"


# ---------------------------------------------------------------------------
# Polytope encoder — safe budget + allocator
# ---------------------------------------------------------------------------


def test_polytope_lattice_levels_match_values():
    assert len(POLYTOPE_LATTICE_VALUES) == POLYTOPE_LATTICE_LEVELS
    assert -2 in POLYTOPE_LATTICE_VALUES and 2 in POLYTOPE_LATTICE_VALUES
    assert 0 in POLYTOPE_LATTICE_VALUES


def test_safe_budget_basic_division():
    mm = torch.tensor([[10.0, 20.0], [5.0, 0.0]])
    safe = compute_safe_perturbation_budget(mm, jacobian_lipschitz=10.0)
    # safe = margin / L
    expected = torch.tensor([[1.0, 2.0], [0.5, 0.0]])
    assert torch.allclose(safe, expected, atol=1e-6)


def test_safe_budget_rejects_zero_lipschitz():
    mm = torch.ones(4, 4)
    with pytest.raises(ValueError, match="jacobian_lipschitz"):
        compute_safe_perturbation_budget(mm, jacobian_lipschitz=0.0)


def test_safe_budget_rejects_negative_margin():
    mm = torch.tensor([[1.0, -1.0]])
    with pytest.raises(ValueError, match="negative"):
        compute_safe_perturbation_budget(mm, jacobian_lipschitz=10.0)


def test_allocate_noise_uniform_safe_budget():
    # Uniform safe budget — bisection should drive every pixel to the
    # entropy cap (or distribute uniformly within budget).
    safe = np.ones(1024, dtype=np.float32) * 4.0  # B_safe = 4 -> log2(4) = 2
    result = allocate_noise_within_polytope(
        safe, budget_bits=2048, jacobian_lipschitz=10.0,
    )
    assert result.noise_levels.shape == (1024,)
    assert math.isfinite(result.lambda_star)
    assert result.total_entropy_bits <= 2048 + 1.0
    assert math.isclose(result.jacobian_lipschitz, 10.0)
    # Interior fraction should be 1.0 (all pixels have B_safe > eps).
    assert math.isclose(result.polytope_interior_fraction, 1.0, abs_tol=1e-6)


def test_allocate_noise_boundary_pixels_get_zero_entropy():
    # Mix of high-safe-budget pixels and zero-safe-budget (boundary) pixels.
    # The boundary pixels must receive ZERO noise.
    safe = np.array([4.0] * 100 + [0.0] * 100, dtype=np.float32)
    result = allocate_noise_within_polytope(
        safe, budget_bits=200, jacobian_lipschitz=10.0,
    )
    boundary_levels = result.noise_levels[100:]
    # Every boundary pixel should be 0 (no entropy allocated).
    assert np.all(boundary_levels == 0), (
        "polytope encoder violated SegNet argmax-stability invariant: "
        "boundary pixels received noise allocation"
    )


def test_allocate_noise_polytope_interior_fraction_correct():
    # 60% interior, 40% boundary.
    safe = np.concatenate([
        np.ones(60, dtype=np.float32) * 2.0,
        np.zeros(40, dtype=np.float32),
    ])
    result = allocate_noise_within_polytope(
        safe, budget_bits=100, jacobian_lipschitz=10.0,
    )
    assert math.isclose(result.polytope_interior_fraction, 0.6, abs_tol=1e-6)


def test_allocate_noise_rejects_empty_safe_budget():
    with pytest.raises(ValueError, match="empty"):
        allocate_noise_within_polytope(
            np.array([], dtype=np.float32),
            budget_bits=64,
            jacobian_lipschitz=10.0,
        )


def test_allocate_noise_rejects_zero_budget():
    with pytest.raises(ValueError, match="budget_bits"):
        allocate_noise_within_polytope(
            np.ones(16, dtype=np.float32),
            budget_bits=0,
            jacobian_lipschitz=10.0,
        )


def test_allocate_noise_rejects_2d_safe_budget():
    with pytest.raises(ValueError, match="1D"):
        allocate_noise_within_polytope(
            np.ones((4, 4), dtype=np.float32),
            budget_bits=64,
            jacobian_lipschitz=10.0,
        )


def test_allocate_noise_rejects_negative_safe_budget():
    safe = np.array([1.0, -0.5, 2.0], dtype=np.float32)
    with pytest.raises(ValueError, match="negative"):
        allocate_noise_within_polytope(
            safe, budget_bits=64, jacobian_lipschitz=10.0,
        )


def test_allocate_noise_all_boundary_pathological():
    # All pixels boundary -> no entropy allocatable. Should not crash.
    safe = np.zeros(64, dtype=np.float32)
    result = allocate_noise_within_polytope(
        safe, budget_bits=64, jacobian_lipschitz=10.0,
    )
    assert np.all(result.noise_levels == 0)
    assert result.total_entropy_bits == 0
    assert math.isclose(result.polytope_interior_fraction, 0.0, abs_tol=1e-6)


def test_encode_decode_polytope_payload_roundtrip():
    rng = np.random.RandomState(42)
    # Synthetic margin map with high-interior + boundary mix.
    mm = torch.tensor(
        (rng.rand(48, 64).astype(np.float32) * 3.0).clip(min=0.0)
    )
    payload = encode_polytope_payload(
        mm, jacobian_lipschitz=10.0, budget_bits=2000,
    )
    assert isinstance(payload, bytes)
    assert len(payload) > 18  # at least header + brotli overhead
    decoded = decode_polytope_payload(payload)
    assert decoded.noise_levels.shape == (48 * 64,)
    assert math.isclose(decoded.jacobian_lipschitz, 10.0)
    assert np.all(decoded.noise_levels >= POLYTOPE_LATTICE_VALUES[0])
    assert np.all(decoded.noise_levels <= POLYTOPE_LATTICE_VALUES[-1])


def test_encode_polytope_payload_rejects_invalid_shape():
    mm = torch.ones(4)  # 1D
    with pytest.raises(ValueError, match="2D or 3D"):
        encode_polytope_payload(mm, jacobian_lipschitz=10.0, budget_bits=64)


def test_decode_polytope_payload_rejects_truncated_blob():
    import brotli  # type: ignore[import-not-found]

    short = brotli.compress(b"x" * 4, quality=9)  # 4 < header size
    with pytest.raises(ValueError, match="too short"):
        decode_polytope_payload(short)


def test_decode_polytope_payload_rejects_bad_magic():
    import struct

    import brotli  # type: ignore[import-not-found]

    bad_header = struct.pack("<4sBIfBf", b"XXXX", 1, 16, 0.0, 5, 10.0)
    blob = brotli.compress(bad_header + b"\x00" * 16, quality=9)
    with pytest.raises(ValueError, match="bad magic"):
        decode_polytope_payload(blob)


# ---------------------------------------------------------------------------
# Archive grammar (D1POLY1 0.bin) — the byte-stability backbone
# ---------------------------------------------------------------------------


def _make_dummy_archive_inputs():
    cfg = D1PolytopeConfig(
        margin_map_resolution=(48, 64),
        polytope_payload_bits=2000,
        jacobian_lipschitz=10.0,
    )
    mm = compute_logit_margin_map_dummy(
        resolution=(48, 64), constant_value=2.0
    )
    payload = encode_polytope_payload(
        mm, jacobian_lipschitz=10.0, budget_bits=2000
    )
    base_sha = "a" * 64
    return cfg, mm, payload, base_sha


def test_d1poly1_header_invariants():
    assert D1POLY1_MAGIC == b"D1PY"
    assert D1POLY1_SCHEMA_VERSION == 1
    assert D1POLY1_HEADER_SIZE == 31
    import struct

    assert struct.calcsize(D1POLY1_HEADER_FMT) == 31


def test_pack_parse_roundtrip_byte_stable():
    cfg, mm, payload, base_sha = _make_dummy_archive_inputs()
    blob1 = pack_archive(
        margin_map=mm,
        polytope_payload=payload,
        jacobian_lipschitz=10.0,
        base_substrate_id="a1",
        base_archive_sha256=base_sha,
        base_archive_bytes=200_000,
        config=cfg,
        extra_meta={"trainer_hash": "abc123"},
    )
    blob2 = pack_archive(
        margin_map=mm,
        polytope_payload=payload,
        jacobian_lipschitz=10.0,
        base_substrate_id="a1",
        base_archive_sha256=base_sha,
        base_archive_bytes=200_000,
        config=cfg,
        extra_meta={"trainer_hash": "abc123"},
    )
    assert blob1 == blob2, "pack_archive must be byte-deterministic"
    parsed = parse_archive(blob1)
    assert parsed.base_substrate_id == "a1"
    assert parsed.base_archive_sha256_truncated == base_sha[:16]
    assert parsed.height == 48 and parsed.width == 64
    assert parsed.schema_version == 1
    assert math.isclose(parsed.jacobian_lipschitz, 10.0)


def test_parse_rejects_bad_magic():
    cfg, mm, payload, base_sha = _make_dummy_archive_inputs()
    blob = pack_archive(
        margin_map=mm,
        polytope_payload=payload,
        jacobian_lipschitz=10.0,
        base_substrate_id="a1",
        base_archive_sha256=base_sha,
        base_archive_bytes=100,
        config=cfg,
        extra_meta={},
    )
    corrupted = b"BADM" + blob[4:]
    with pytest.raises(ValueError, match="bad magic"):
        parse_archive(corrupted)


def test_parse_rejects_size_mismatch():
    cfg, mm, payload, base_sha = _make_dummy_archive_inputs()
    blob = pack_archive(
        margin_map=mm,
        polytope_payload=payload,
        jacobian_lipschitz=10.0,
        base_substrate_id="a1",
        base_archive_sha256=base_sha,
        base_archive_bytes=100,
        config=cfg,
        extra_meta={},
    )
    with pytest.raises(ValueError, match="size"):
        parse_archive(blob + b"\x00\x00")


def test_parse_recovers_margin_map_within_int8_tolerance():
    cfg = D1PolytopeConfig(
        margin_map_resolution=(16, 32),
        polytope_payload_bits=200,
        jacobian_lipschitz=10.0,
    )
    rng = np.random.RandomState(7)
    mm = torch.tensor(
        (rng.rand(16, 32).astype(np.float32) * 3.0).clip(min=0.0)
    )
    payload = encode_polytope_payload(
        mm, jacobian_lipschitz=10.0, budget_bits=200
    )
    blob = pack_archive(
        margin_map=mm,
        polytope_payload=payload,
        jacobian_lipschitz=10.0,
        base_substrate_id="pr101_lc_v2_clone",
        base_archive_sha256="b" * 64,
        base_archive_bytes=1000,
        config=cfg,
        extra_meta={},
    )
    parsed = parse_archive(blob)
    recovered = parsed.margin_map_float()
    assert recovered.shape == (16, 32)
    err = np.abs(mm.numpy() - recovered).max()
    assert err < 0.05, f"int8 quantization tolerance breached: {err}"


def test_pack_rejects_bad_base_id():
    cfg, mm, payload, _ = _make_dummy_archive_inputs()
    with pytest.raises(ValueError, match="base_substrate_id length"):
        pack_archive(
            margin_map=mm,
            polytope_payload=payload,
            jacobian_lipschitz=10.0,
            base_substrate_id="",
            base_archive_sha256="c" * 64,
            base_archive_bytes=100,
            config=cfg,
            extra_meta={},
        )


def test_pack_rejects_bad_sha():
    cfg, mm, payload, _ = _make_dummy_archive_inputs()
    with pytest.raises(ValueError, match="base_archive_sha256"):
        pack_archive(
            margin_map=mm,
            polytope_payload=payload,
            jacobian_lipschitz=10.0,
            base_substrate_id="a1",
            base_archive_sha256="short",
            base_archive_bytes=100,
            config=cfg,
            extra_meta={},
        )


def test_pack_rejects_non_positive_lipschitz():
    cfg, mm, payload, _ = _make_dummy_archive_inputs()
    with pytest.raises(ValueError, match="jacobian_lipschitz"):
        pack_archive(
            margin_map=mm,
            polytope_payload=payload,
            jacobian_lipschitz=0.0,
            base_substrate_id="a1",
            base_archive_sha256="d" * 64,
            base_archive_bytes=100,
            config=cfg,
            extra_meta={},
        )


def test_pack_rejects_extra_meta_collision():
    cfg, mm, payload, _ = _make_dummy_archive_inputs()
    with pytest.raises(ValueError, match="collides with reserved"):
        pack_archive(
            margin_map=mm,
            polytope_payload=payload,
            jacobian_lipschitz=10.0,
            base_substrate_id="a1",
            base_archive_sha256="e" * 64,
            base_archive_bytes=100,
            config=cfg,
            extra_meta={"score_claim": True},
        )


def test_pack_meta_carries_apples_to_apples_discipline():
    cfg, mm, payload, _ = _make_dummy_archive_inputs()
    blob = pack_archive(
        margin_map=mm,
        polytope_payload=payload,
        jacobian_lipschitz=10.0,
        base_substrate_id="a1",
        base_archive_sha256="f" * 64,
        base_archive_bytes=100,
        config=cfg,
        extra_meta={},
    )
    parsed = parse_archive(blob)
    # Per CLAUDE.md "Apples-to-apples evidence discipline" D1 cannot
    # claim a score without paired CUDA + CPU exact eval — the archive
    # MUST self-tag as proxy/non-promotable.
    assert parsed.meta["score_claim"] is False
    assert parsed.meta["evidence_grade"] == "proxy"
    assert parsed.meta["ready_for_exact_eval_dispatch"] is False


# ---------------------------------------------------------------------------
# Composability with base substrates
# ---------------------------------------------------------------------------


def test_compose_with_base_packs_archive_with_correct_id_binding():
    cfg = D1PolytopeConfig(
        margin_map_resolution=(16, 32),
        polytope_payload_bits=200,
        jacobian_lipschitz=10.0,
    )
    mm = compute_logit_margin_map_dummy(resolution=(16, 32))
    payload = encode_polytope_payload(
        mm, jacobian_lipschitz=10.0, budget_bits=200
    )
    desc = _BaseArchiveDescriptor(
        base_substrate_id="a1",
        base_archive_sha256="f" * 64,
        base_archive_bytes=200_000,
    )
    blob = compose_with_base(
        base_archive_descriptor=desc,
        margin_map=mm,
        polytope_payload=payload,
        config=cfg,
        extra_meta={"predicted_delta_score": -0.008},
    )
    parsed = parse_archive(blob)
    assert parsed.base_substrate_id == "a1"
    assert parsed.base_archive_sha256_truncated == "f" * 16
    assert parsed.meta["base_archive_sha256_full"] == "f" * 64
    assert parsed.meta["predicted_delta_score"] == -0.008


def test_compose_supports_yucr_base_for_d1_yucr_cross_axis_stacking():
    # D1 composes with YUCR base for the bidirectional cooperative-receiver
    # exploit (frame-0 nullspace + frame-1 polytope-interior).
    cfg = D1PolytopeConfig(
        base_substrate_id="yucr",
        margin_map_resolution=(16, 32),
        polytope_payload_bits=200,
    )
    mm = compute_logit_margin_map_dummy(resolution=(16, 32))
    payload = encode_polytope_payload(
        mm, jacobian_lipschitz=cfg.jacobian_lipschitz, budget_bits=200
    )
    desc = _BaseArchiveDescriptor(
        base_substrate_id="yucr",
        base_archive_sha256="0" * 64,
        base_archive_bytes=2_000,
    )
    blob = compose_with_base(
        base_archive_descriptor=desc,
        margin_map=mm,
        polytope_payload=payload,
        config=cfg,
        extra_meta={},
    )
    parsed = parse_archive(blob)
    assert parsed.base_substrate_id == "yucr"


def test_compose_rejects_margin_map_resolution_mismatch():
    cfg = D1PolytopeConfig(
        margin_map_resolution=(16, 32),
        polytope_payload_bits=200,
    )
    mm_wrong = compute_logit_margin_map_dummy(resolution=(8, 8))
    payload = encode_polytope_payload(
        mm_wrong, jacobian_lipschitz=cfg.jacobian_lipschitz, budget_bits=200
    )
    desc = _BaseArchiveDescriptor(
        base_substrate_id="a1",
        base_archive_sha256="0" * 64,
        base_archive_bytes=100,
    )
    with pytest.raises(ValueError, match="margin_map_resolution"):
        compose_with_base(
            base_archive_descriptor=desc,
            margin_map=mm_wrong,
            polytope_payload=payload,
            config=cfg,
            extra_meta={},
        )


def test_base_archive_descriptor_validates_id():
    with pytest.raises(ValueError, match="not in"):
        _BaseArchiveDescriptor(
            base_substrate_id="ghost_substrate",
            base_archive_sha256="0" * 64,
            base_archive_bytes=100,
        )


def test_base_archive_descriptor_validates_sha_length():
    with pytest.raises(ValueError, match="sha256"):
        _BaseArchiveDescriptor(
            base_substrate_id="a1",
            base_archive_sha256="too_short",
            base_archive_bytes=100,
        )


def test_estimate_overhead_in_target_band():
    cfg = D1PolytopeConfig(
        margin_map_resolution=(48, 64), polytope_payload_bits=2000
    )
    estimated = estimate_overhead_bytes(config=cfg)
    # Allow up to 2x of MAX since estimate is conservative pre-brotli.
    assert D1POLY_OVERHEAD_TARGET_BYTES_MIN <= estimated <= (
        D1POLY_OVERHEAD_TARGET_BYTES_MAX * 2
    )


# ---------------------------------------------------------------------------
# Loss weights validation
# ---------------------------------------------------------------------------


def test_loss_weights_defaults_match_contest_formula():
    w = D1PolytopeLossWeights()
    assert math.isclose(w.seg_weight, 100.0)
    assert math.isclose(w.pose_sqrt_weight, math.sqrt(10.0))
    assert w.rate_weight == 25.0
    assert 0 <= w.lambda_d1 <= 10.0
    assert 0 <= w.margin_threshold <= 10.0


def test_loss_weights_reject_negative_lambda():
    with pytest.raises(ValueError, match="lambda_d1"):
        D1PolytopeLossWeights(lambda_d1=-0.1)


def test_loss_weights_reject_giant_lambda():
    with pytest.raises(ValueError, match="lambda_d1"):
        D1PolytopeLossWeights(lambda_d1=20.0)


def test_loss_weights_reject_negative_margin_threshold():
    with pytest.raises(ValueError, match="margin_threshold"):
        D1PolytopeLossWeights(margin_threshold=-0.1)


# ---------------------------------------------------------------------------
# Readiness manifest — non-promotion + apples-to-apples (Catalog #192)
# ---------------------------------------------------------------------------


def test_readiness_manifest_has_no_score_claim():
    """Manifest never sets score_claim or promotion_eligible regardless of
    runtime_overlay_consumed."""
    cfg = D1PolytopeConfig()
    manifest = build_readiness_manifest(
        base_substrate_id="a1",
        base_archive_bytes=200_000,
        d1_overhead_bytes=1_500,
        config=cfg,
    )
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False


def test_readiness_manifest_l1_no_op_explicit_false():
    """When the caller asserts L1 NO-OP overlay (runtime_overlay_consumed=False)
    the manifest must surface the dispatch blocker per Catalog #220."""
    cfg = D1PolytopeConfig()
    manifest = build_readiness_manifest(
        base_substrate_id="a1",
        base_archive_bytes=200_000,
        d1_overhead_bytes=1_500,
        config=cfg,
        runtime_overlay_consumed=False,
    )
    assert manifest["score_claim"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["runtime_overlay_consumed"] is False
    assert manifest["current_runtime_effect"] == "base_renderer_plus_rate_only"
    assert manifest["predicted_score_evidence_grade"] == "blocked_l1_noop_overlay"
    assert "d1_runtime_overlay_not_consumed" in manifest["dispatch_blockers"]


def test_readiness_manifest_records_total_archive_bytes():
    cfg = D1PolytopeConfig()
    manifest = build_readiness_manifest(
        base_substrate_id="a1",
        base_archive_bytes=178_000,
        d1_overhead_bytes=2_000,
        config=cfg,
    )
    assert manifest["total_archive_bytes"] == 178_000 + 2_000


def test_readiness_manifest_l1_no_op_blocks_score_band_records_l2_projection():
    # When the caller explicitly asserts L1 NO-OP overlay (runtime_overlay_consumed=False)
    # the manifest blocks current band but still records the L2 projection.
    cfg = D1PolytopeConfig()
    manifest = build_readiness_manifest(
        base_substrate_id="a1",
        base_archive_bytes=178_000,
        d1_overhead_bytes=2_000,
        config=cfg,
        runtime_overlay_consumed=False,
    )
    assert manifest["predicted_score_band_low"] is None
    assert manifest["predicted_score_band_high"] is None
    assert 0.18 <= manifest["l2_projected_score_band_low"] <= 0.19
    assert 0.18 <= manifest["l2_projected_score_band_high"] <= 0.19


def test_readiness_manifest_l2_default_overlay_ready_exposes_dispatchable_projection():
    """L2 INTEGRATION default: runtime_overlay_consumed defaults to True
    (the inflate runtime applies the polytope overlay)."""
    cfg = D1PolytopeConfig()
    manifest = build_readiness_manifest(
        base_substrate_id="a1",
        base_archive_bytes=178_000,
        d1_overhead_bytes=2_000,
        config=cfg,
    )
    assert manifest["runtime_overlay_consumed"] is True
    assert manifest["ready_for_exact_eval_dispatch"] is True
    assert manifest["promotion_eligible"] is False
    assert manifest["predicted_score_evidence_grade"] == "first-principles-bound"
    assert 0.18 <= manifest["predicted_score_band_low"] <= 0.19
    assert 0.18 <= manifest["predicted_score_band_high"] <= 0.19


# ---------------------------------------------------------------------------
# D1 substrate handle
# ---------------------------------------------------------------------------


def test_d1_substrate_default_constructible():
    sub = D1PolytopeSidecar()
    assert sub.config.base_substrate_id == D1POLY_DEFAULT_BASE_SUBSTRATE


def test_d1_substrate_carries_typed_config():
    cfg = D1PolytopeConfig(
        base_substrate_id="pr101_lc_v2_clone",
        polytope_payload_bits=4000,
    )
    sub = D1PolytopeSidecar(config=cfg)
    assert sub.config.polytope_payload_bits == 4000
    assert sub.config.base_substrate_id == "pr101_lc_v2_clone"


# ---------------------------------------------------------------------------
# Inflate runtime (Catalog #146 contract)
# ---------------------------------------------------------------------------


def test_inflate_module_exposes_main():
    from tac.substrates.d1_segnet_margin_polytope import inflate

    assert callable(getattr(inflate, "main", None))


def test_inflate_loc_under_substrate_engineering_budget():
    """HNeRV parity L4: inflate.py <= 200 LOC substrate-engineering waiver."""
    from pathlib import Path

    inflate_path = (
        Path(__file__).resolve().parent.parent / "inflate.py"
    )
    line_count = sum(1 for _ in inflate_path.read_text().splitlines())
    assert line_count <= 200, (
        f"D1 inflate.py is {line_count} LOC; budget is 200"
    )


def test_inflate_locator_fails_when_d1_bin_absent(tmp_path):
    from tac.substrates.d1_segnet_margin_polytope.inflate import (
        _locate_d1_archive,
    )

    with pytest.raises(FileNotFoundError, match="D1 sidecar not found"):
        _locate_d1_archive(tmp_path)


def test_default_a1_base_exposes_package_inflate_adapter():
    import importlib

    base_inflate = importlib.import_module("tac.substrates.a1.inflate")
    assert hasattr(base_inflate, "main")


def test_inflate_base_archive_sha_mismatch_refuses(tmp_path):
    from tac.substrates.d1_segnet_margin_polytope.inflate import (
        _verify_base_archive_match,
    )

    base_path = tmp_path / "a1.bin"
    base_path.write_bytes(b"some-base-bytes")
    actual_sha = hashlib.sha256(base_path.read_bytes()).hexdigest()
    wrong_truncated = (
        "0" * 16
        if not actual_sha.startswith("0" * 16)
        else "1" * 16
    )
    with pytest.raises(ValueError, match="sha mismatch"):
        _verify_base_archive_match(
            tmp_path,
            base_substrate_id="a1",
            base_sha_truncated=wrong_truncated,
        )


def test_inflate_base_archive_sha_match_accepts(tmp_path):
    from tac.substrates.d1_segnet_margin_polytope.inflate import (
        _verify_base_archive_match,
    )

    base_path = tmp_path / "a1.bin"
    base_path.write_bytes(b"some-base-bytes")
    truncated = hashlib.sha256(base_path.read_bytes()).hexdigest()[:16]
    located = _verify_base_archive_match(
        tmp_path,
        base_substrate_id="a1",
        base_sha_truncated=truncated,
    )
    assert located == base_path


def test_no_tmp_paths_in_d1_module_source():
    """CLAUDE.md "Forbidden /tmp paths in any persisted artifact" sanity check.

    Scan the D1 substrate package for any /tmp/ literal in source code
    (not inside test fixtures).
    """
    import re
    from pathlib import Path

    d1_root = Path(__file__).resolve().parent.parent
    forbidden = re.compile(r"['\"]/tmp/")
    for py in d1_root.glob("*.py"):
        text = py.read_text()
        assert not forbidden.search(text), (
            f"forbidden /tmp/ literal found in {py}; per CLAUDE.md "
            "FORBIDDEN_PATTERNS use experiments/results/<lane_id>_<timestamp>/ "
            "for build artifacts"
        )


def test_sidecar_payload_roundtrip_is_structurally_parseable(tmp_path):
    """Catalog #105 / #139 guard: archive payload bytes are parseable.

    A roundtrip pack -> parse -> roundtrip must produce identical bytes,
    and the parsed archive must carry the polytope payload + margin map.
    Runtime score movement is still blocked until L2 overlay wiring.
    """
    cfg, mm, payload, base_sha = _make_dummy_archive_inputs()
    blob = pack_archive(
        margin_map=mm,
        polytope_payload=payload,
        jacobian_lipschitz=10.0,
        base_substrate_id="a1",
        base_archive_sha256=base_sha,
        base_archive_bytes=100,
        config=cfg,
        extra_meta={},
    )
    parsed = parse_archive(blob)
    assert parsed.margin_map_int8.size > 0
    assert len(parsed.polytope_payload) > 0
    # Re-roundtrip should match (deterministic pack).
    blob2 = pack_archive(
        margin_map=mm,
        polytope_payload=payload,
        jacobian_lipschitz=10.0,
        base_substrate_id="a1",
        base_archive_sha256=base_sha,
        base_archive_bytes=100,
        config=cfg,
        extra_meta={},
    )
    assert blob == blob2


def test_trainer_runtime_emits_d1_sidecar_verifier(tmp_path):
    import importlib

    trainer = importlib.import_module(
        "experiments.train_substrate_d1_segnet_margin_polytope"
    )
    submission_dir = tmp_path / "submission"
    trainer._write_runtime(submission_dir)
    assert (submission_dir / "d1_verify.py").is_file()
    assert (submission_dir / "a1_inflate.py").is_file()
    inflate_sh = (submission_dir / "inflate.sh").read_text(encoding="utf-8")
    assert "$HERE/inflate.py" in inflate_sh
    assert "polytope overlay" in inflate_sh
    inflate_py = (submission_dir / "inflate.py").read_text(encoding="utf-8")
    assert "_apply_overlay" in inflate_py
    assert "changed zero bytes" in inflate_py

    base_bytes = b"some-a1-base"
    base_path = tmp_path / "a1.bin"
    base_path.write_bytes(base_bytes)
    cfg, mm, payload, _ = _make_dummy_archive_inputs()
    blob = pack_archive(
        margin_map=mm,
        polytope_payload=payload,
        jacobian_lipschitz=10.0,
        base_substrate_id="a1",
        base_archive_sha256=hashlib.sha256(base_bytes).hexdigest(),
        base_archive_bytes=len(base_bytes),
        config=cfg,
        extra_meta={},
    )
    d1_path = tmp_path / "d1_polytope.bin"
    d1_path.write_bytes(blob)
    result = subprocess.run(
        [
            sys.executable,
            str(submission_dir / "d1_verify.py"),
            str(d1_path),
            str(base_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "[d1-verify] parsed D1 sidecar" in result.stderr


def test_generated_d1_runtime_applies_overlay_and_fails_closed(tmp_path):
    import importlib

    trainer = importlib.import_module(
        "experiments.train_substrate_d1_segnet_margin_polytope"
    )
    submission_dir = tmp_path / "submission"
    trainer._write_runtime(submission_dir)

    # Replace the expensive A1 renderer with a deterministic one-pair raw
    # writer. This keeps the test focused on the generated D1 driver.
    fake_a1 = (
        "from pathlib import Path\n"
        "import sys\n"
        "camera_h, camera_w = 874, 1164\n"
        "frame_bytes = camera_h * camera_w * 3\n"
        "Path(sys.argv[2]).write_bytes(bytes([128]) * (2 * frame_bytes))\n"
    )
    (submission_dir / "a1_inflate.py").write_text(
        fake_a1, encoding="utf-8"
    )

    base_bytes = b"some-a1-base"
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    (archive_dir / "a1.bin").write_bytes(base_bytes)

    cfg, mm, payload, _ = _make_dummy_archive_inputs()
    blob = pack_archive(
        margin_map=mm,
        polytope_payload=payload,
        jacobian_lipschitz=10.0,
        base_substrate_id="a1",
        base_archive_sha256=hashlib.sha256(base_bytes).hexdigest(),
        base_archive_bytes=len(base_bytes),
        config=cfg,
        extra_meta={},
    )
    (archive_dir / "d1_polytope.bin").write_bytes(blob)
    file_list = tmp_path / "file_list.txt"
    file_list.write_text("0.mkv\n", encoding="utf-8")
    output_dir = tmp_path / "out"

    import importlib.util

    from tac.substrates.d1_segnet_margin_polytope.overlay import (
        _build_camera_resolution_overlay,
    )

    spec = importlib.util.spec_from_file_location(
        "generated_d1_inflate", submission_dir / "inflate.py"
    )
    assert spec is not None
    assert spec.loader is not None
    generated_inflate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generated_inflate)
    decoded = decode_polytope_payload(payload)
    margin_hw = generated_inflate._decode_margin(
        generated_inflate._parse_d1_sidecar(archive_dir / "d1_polytope.bin")[
            "margin_blob"
        ],
        height=48,
        width=64,
        scale=parse_archive(blob).margin_map_scale,
    )
    np.testing.assert_array_equal(
        generated_inflate._decode_overlay(
            payload,
            height=48,
            width=64,
            expected_lipschitz=10.0,
            margin_hw=margin_hw,
        ),
        _build_camera_resolution_overlay(
            noise_levels_flat=decoded.noise_levels,
            encoder_grid_h=48,
            encoder_grid_w=64,
        ),
    )

    result = subprocess.run(
        [
            sys.executable,
            str(submission_dir / "inflate.py"),
            str(archive_dir),
            str(output_dir),
            str(file_list),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "OVERLAY_TOTAL" in result.stderr
    raw_bytes = (output_dir / "0.raw").read_bytes()
    assert raw_bytes != bytes([128]) * len(raw_bytes)

    zero_archive_dir = tmp_path / "archive_zero"
    zero_archive_dir.mkdir()
    (zero_archive_dir / "a1.bin").write_bytes(base_bytes)
    zero_margin = torch.zeros_like(mm)
    zero_blob = pack_archive(
        margin_map=zero_margin,
        polytope_payload=encode_polytope_payload(
            zero_margin, jacobian_lipschitz=10.0, budget_bits=2000
        ),
        jacobian_lipschitz=10.0,
        base_substrate_id="a1",
        base_archive_sha256=hashlib.sha256(base_bytes).hexdigest(),
        base_archive_bytes=len(base_bytes),
        config=cfg,
        extra_meta={},
    )
    (zero_archive_dir / "d1_polytope.bin").write_bytes(zero_blob)
    zero_result = subprocess.run(
        [
            sys.executable,
            str(submission_dir / "inflate.py"),
            str(zero_archive_dir),
            str(tmp_path / "out_zero"),
            str(file_list),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert zero_result.returncode != 0
    assert "changed zero bytes" in zero_result.stderr


def test_d1_substrate_lane_class_marker_present():
    """Catalog #124 HNeRV parity L7 substrate_engineering waiver: ensure
    the lane_class declaration is present in module docstring so the
    AST walker observes it.
    """
    from pathlib import Path

    init_path = (
        Path(__file__).resolve().parent.parent / "__init__.py"
    )
    text = init_path.read_text()
    assert "lane_class=substrate_engineering" in text, (
        "D1 must declare lane_class=substrate_engineering in module "
        "docstring per HNeRV parity L7"
    )
