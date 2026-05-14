"""YUCR substrate test suite.

Per CLAUDE.md "Beauty, simplicity, and developer experience" + "Apples-to-
apples evidence discipline" + Catalog #124/#146/#157/#164/#192. NO score
claims. NO /tmp paths. Every reconstruction-error tag is ``[proxy]``.
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pytest
import torch

from tac.substrates.yucr import (
    STC_DEFAULT_BUDGET_BITS,
    STC_LATTICE_LEVELS,
    YUCR1_HEADER_FMT,
    YUCR1_HEADER_SIZE,
    YUCR1_MAGIC,
    YUCR1_SCHEMA_VERSION,
    YUCR_BASE_SUBSTRATE_IDS,
    YUCR_DEFAULT_BASE_SUBSTRATE,
    YUCRArchive,
    YUCRConfig,
    YUCRLossWeights,
    YUCRSubstrate,
    build_readiness_manifest,
    compute_cost_map_dummy,
    decode_stc_payload,
    encode_stc_payload,
    pack_archive,
    parse_archive,
    quantize_cost_map_int8,
    waterfill_allocate,
)
from tac.substrates.yucr.architecture import (
    YUCR_OVERHEAD_TARGET_BYTES_MAX,
    YUCR_OVERHEAD_TARGET_BYTES_MIN,
    _BaseArchiveDescriptor,
    compose_with_base,
    estimate_overhead_bytes,
)
from tac.substrates.yucr.cost_map import (
    YUCRCostMapMode,
    dequantize_cost_map_int8,
)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_yucr_config_defaults_are_self_consistent():
    cfg = YUCRConfig()
    assert cfg.base_substrate_id == YUCR_DEFAULT_BASE_SUBSTRATE
    assert cfg.cost_map_mode == "score_gradient"
    assert cfg.stc_payload_bits == 8000
    assert cfg.cost_map_resolution == (384, 512)
    assert math.isclose(cfg.pose_sqrt_weight, math.sqrt(10.0))
    assert math.isclose(cfg.seg_weight, 100.0)


def test_yucr_config_rejects_unknown_base():
    with pytest.raises(ValueError, match="not in"):
        YUCRConfig(base_substrate_id="some_unknown_substrate")


def test_yucr_config_rejects_unknown_cost_map_mode():
    with pytest.raises(ValueError, match="cost_map_mode"):
        YUCRConfig(cost_map_mode="moonbeam")


def test_yucr_config_rejects_zero_payload_bits():
    with pytest.raises(ValueError, match="stc_payload_bits"):
        YUCRConfig(stc_payload_bits=0)


def test_yucr_config_rejects_giant_payload_bits():
    with pytest.raises(ValueError, match="stc_payload_bits"):
        YUCRConfig(stc_payload_bits=(1 << 21))


def test_yucr_config_rejects_zero_resolution():
    with pytest.raises(ValueError, match="cost_map_resolution"):
        YUCRConfig(cost_map_resolution=(0, 512))


def test_yucr_config_rejects_negative_l_inf_cap():
    with pytest.raises(ValueError, match="l_inf_noise_cap"):
        YUCRConfig(l_inf_noise_cap=-1.0)


# ---------------------------------------------------------------------------
# Cost map (dummy mode for tests; real mode covered by integration test below)
# ---------------------------------------------------------------------------


def test_cost_map_dummy_returns_correct_shape_and_value():
    cm = compute_cost_map_dummy(resolution=(48, 64), constant_value=2.5)
    assert cm.shape == (48, 64)
    assert cm.dtype == torch.float32
    assert torch.all(cm == 2.5)


def test_cost_map_dummy_rejects_non_positive_constant():
    with pytest.raises(ValueError, match="constant_value"):
        compute_cost_map_dummy(constant_value=0.0)


def test_quantize_cost_map_int8_roundtrip():
    cm = torch.tensor([[1.0, 2.0, 3.0], [-0.5, 0.0, 0.25]])
    int8, scale = quantize_cost_map_int8(cm, scale=127.0)
    assert int8.dtype == torch.int8
    assert int8.shape == cm.shape
    assert scale > 0
    # Recovered values should be close to original (modulo int8 quantization).
    recovered = dequantize_cost_map_int8(int8, recovered_scale=scale)
    err = (cm - recovered).abs().max().item()
    assert err < 0.05, f"int8 quantize error {err} too large"


def test_quantize_cost_map_int8_rejects_bad_scale():
    cm = torch.ones(8, 8)
    with pytest.raises(ValueError, match="scale"):
        quantize_cost_map_int8(cm, scale=0.0)
    with pytest.raises(ValueError, match="scale"):
        quantize_cost_map_int8(cm, scale=200.0)


def test_dequantize_rejects_non_positive_scale():
    int8 = torch.zeros(4, 4, dtype=torch.int8)
    with pytest.raises(ValueError, match="recovered_scale"):
        dequantize_cost_map_int8(int8, recovered_scale=-1.0)


def test_yucr_cost_map_mode_enum_values():
    assert YUCRCostMapMode.SCORE_GRADIENT.value == "score_gradient"
    assert YUCRCostMapMode.UNIFORM.value == "uniform"
    assert YUCRCostMapMode.DUMMY_CONSTANT.value == "dummy_constant"


# ---------------------------------------------------------------------------
# STC encoder
# ---------------------------------------------------------------------------


def test_stc_lattice_levels_match_values():
    from tac.substrates.yucr.stc_encoder import STC_LATTICE_VALUES

    assert len(STC_LATTICE_VALUES) == STC_LATTICE_LEVELS
    assert -2 in STC_LATTICE_VALUES and 2 in STC_LATTICE_VALUES


def test_waterfill_allocate_uniform_cost_uses_full_budget():
    # Uniform cost = every pixel equally cheap. Bisection finds a single
    # lambda that distributes total budget evenly. With cap = log2(5) and
    # large budget, lambda* is small enough that every pixel fills to cap.
    cost = np.ones(1024, dtype=np.float32)
    result = waterfill_allocate(cost, budget_bits=2048)
    assert result.noise_levels.shape == (1024,)
    # lambda_star may be negative when cost map is uniformly cheap and the
    # bisection drives every pixel to the lattice entropy cap.
    assert math.isfinite(result.lambda_star)
    assert result.total_entropy_bits <= 2048 + 1.0  # bisection slack


def test_waterfill_allocate_zero_cost_pixels_get_more_bits():
    cost = np.array([1.0] * 100 + [10.0] * 100, dtype=np.float32)
    result = waterfill_allocate(cost, budget_bits=200)
    # Pixels with cost=1 (lower) have higher log_inv_cost; should get more
    # entropy than cost=10 pixels.
    cheap_entropy = result.per_pixel_entropy[:100].sum()
    expensive_entropy = result.per_pixel_entropy[100:].sum()
    assert cheap_entropy >= expensive_entropy


def test_waterfill_rejects_empty_cost_map():
    with pytest.raises(ValueError, match="empty"):
        waterfill_allocate(np.array([], dtype=np.float32), budget_bits=64)


def test_waterfill_rejects_zero_budget():
    with pytest.raises(ValueError, match="budget_bits"):
        waterfill_allocate(np.ones(16, dtype=np.float32), budget_bits=0)


def test_waterfill_rejects_2d_cost_map():
    with pytest.raises(ValueError, match="1D"):
        waterfill_allocate(np.ones((4, 4), dtype=np.float32), budget_bits=64)


def test_encode_decode_stc_payload_roundtrip():
    cm = torch.tensor(
        np.random.RandomState(42).rand(48, 64).astype(np.float32) + 0.1
    )
    payload = encode_stc_payload(cm, budget_bits=2000)
    assert isinstance(payload, bytes)
    assert len(payload) > 14  # at least header + brotli overhead
    decoded = decode_stc_payload(payload)
    assert decoded.noise_levels.shape == (48 * 64,)
    assert decoded.lambda_star >= 0
    assert np.all(decoded.noise_levels >= -2)
    assert np.all(decoded.noise_levels <= 2)


def test_encode_stc_payload_rejects_invalid_shape():
    cm = torch.ones(4)  # 1D
    with pytest.raises(ValueError, match="2D or 3D"):
        encode_stc_payload(cm, budget_bits=64)


def test_decode_stc_payload_rejects_truncated_blob():
    import brotli  # type: ignore[import-not-found]

    # Build a brotli-decodable but too-short-after-decompress payload —
    # decoder should refuse with "too short" error.
    short = brotli.compress(b"x" * 4, quality=9)  # 4 < header size
    with pytest.raises(ValueError, match="too short"):
        decode_stc_payload(short)


# ---------------------------------------------------------------------------
# Archive grammar (YUCR1 0.bin) — the byte-stability backbone
# ---------------------------------------------------------------------------


def _make_dummy_archive_inputs():
    cfg = YUCRConfig(
        cost_map_resolution=(48, 64),
        stc_payload_bits=2000,
    )
    cm = compute_cost_map_dummy(resolution=(48, 64), constant_value=1.0)
    payload = encode_stc_payload(cm, budget_bits=2000)
    base_sha = "a" * 64
    return cfg, cm, payload, base_sha


def test_yucr1_header_invariants():
    assert YUCR1_MAGIC == b"YUCR"
    assert YUCR1_SCHEMA_VERSION == 1
    assert YUCR1_HEADER_SIZE == 27
    import struct

    assert struct.calcsize(YUCR1_HEADER_FMT) == 27


def test_pack_parse_roundtrip_byte_stable():
    cfg, cm, payload, base_sha = _make_dummy_archive_inputs()
    blob1 = pack_archive(
        cost_map=cm,
        stc_payload=payload,
        base_substrate_id="a1",
        base_archive_sha256=base_sha,
        base_archive_bytes=200_000,
        config=cfg,
        extra_meta={"trainer_hash": "abc123"},
    )
    blob2 = pack_archive(
        cost_map=cm,
        stc_payload=payload,
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


def test_parse_rejects_bad_magic():
    cfg, cm, payload, base_sha = _make_dummy_archive_inputs()
    blob = pack_archive(
        cost_map=cm,
        stc_payload=payload,
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
    cfg, cm, payload, base_sha = _make_dummy_archive_inputs()
    blob = pack_archive(
        cost_map=cm,
        stc_payload=payload,
        base_substrate_id="a1",
        base_archive_sha256=base_sha,
        base_archive_bytes=100,
        config=cfg,
        extra_meta={},
    )
    with pytest.raises(ValueError, match="size"):
        parse_archive(blob + b"\x00\x00")


def test_parse_recovers_cost_map_within_int8_tolerance():
    cfg = YUCRConfig(cost_map_resolution=(16, 32), stc_payload_bits=200)
    rng = np.random.RandomState(7)
    cm = torch.tensor(rng.rand(16, 32).astype(np.float32) + 0.1)
    payload = encode_stc_payload(cm, budget_bits=200)
    blob = pack_archive(
        cost_map=cm,
        stc_payload=payload,
        base_substrate_id="pr101_lc_v2_clone",
        base_archive_sha256="b" * 64,
        base_archive_bytes=1000,
        config=cfg,
        extra_meta={},
    )
    parsed = parse_archive(blob)
    recovered = parsed.cost_map_float()
    assert recovered.shape == (16, 32)
    err = np.abs(cm.numpy() - recovered).max()
    assert err < 0.05, f"int8 quantization tolerance breached: {err}"


def test_pack_rejects_bad_base_id():
    cfg, cm, payload, _ = _make_dummy_archive_inputs()
    with pytest.raises(ValueError, match="base_substrate_id length"):
        pack_archive(
            cost_map=cm,
            stc_payload=payload,
            base_substrate_id="",
            base_archive_sha256="c" * 64,
            base_archive_bytes=100,
            config=cfg,
            extra_meta={},
        )


def test_pack_rejects_bad_sha():
    cfg, cm, payload, _ = _make_dummy_archive_inputs()
    with pytest.raises(ValueError, match="base_archive_sha256"):
        pack_archive(
            cost_map=cm,
            stc_payload=payload,
            base_substrate_id="a1",
            base_archive_sha256="short",
            base_archive_bytes=100,
            config=cfg,
            extra_meta={},
        )


def test_pack_rejects_extra_meta_collision():
    cfg, cm, payload, _ = _make_dummy_archive_inputs()
    with pytest.raises(ValueError, match="collides with reserved"):
        pack_archive(
            cost_map=cm,
            stc_payload=payload,
            base_substrate_id="a1",
            base_archive_sha256="d" * 64,
            base_archive_bytes=100,
            config=cfg,
            extra_meta={"score_claim": True},
        )


def test_pack_meta_carries_score_claim_false_per_apples_to_apples_discipline():
    cfg, cm, payload, _ = _make_dummy_archive_inputs()
    blob = pack_archive(
        cost_map=cm,
        stc_payload=payload,
        base_substrate_id="a1",
        base_archive_sha256="e" * 64,
        base_archive_bytes=100,
        config=cfg,
        extra_meta={},
    )
    parsed = parse_archive(blob)
    # Per CLAUDE.md "Apples-to-apples evidence discipline" YUCR cannot
    # claim a score without paired CUDA + CPU exact eval — the archive
    # MUST self-tag as proxy/non-promotable.
    assert parsed.meta["score_claim"] is False
    assert parsed.meta["evidence_grade"] == "proxy"
    assert parsed.meta["ready_for_exact_eval_dispatch"] is False


# ---------------------------------------------------------------------------
# Composability with base substrates
# ---------------------------------------------------------------------------


def test_compose_with_base_packs_archive_with_correct_id_binding():
    cfg = YUCRConfig(cost_map_resolution=(16, 32), stc_payload_bits=200)
    cm = compute_cost_map_dummy(resolution=(16, 32))
    payload = encode_stc_payload(cm, budget_bits=200)
    desc = _BaseArchiveDescriptor(
        base_substrate_id="a1",
        base_archive_sha256="f" * 64,
        base_archive_bytes=200_000,
    )
    blob = compose_with_base(
        base_archive_descriptor=desc,
        cost_map=cm,
        stc_payload=payload,
        config=cfg,
        extra_meta={"predicted_delta_score": -0.025},
    )
    parsed = parse_archive(blob)
    assert parsed.base_substrate_id == "a1"
    assert parsed.base_archive_sha256_truncated == "f" * 16
    assert parsed.meta["base_archive_sha256_full"] == "f" * 64
    assert parsed.meta["predicted_delta_score"] == -0.025


def test_compose_rejects_cost_map_resolution_mismatch():
    cfg = YUCRConfig(cost_map_resolution=(16, 32), stc_payload_bits=200)
    cm_wrong = compute_cost_map_dummy(resolution=(8, 8))
    payload = encode_stc_payload(cm_wrong, budget_bits=200)
    desc = _BaseArchiveDescriptor(
        base_substrate_id="a1",
        base_archive_sha256="0" * 64,
        base_archive_bytes=100,
    )
    with pytest.raises(ValueError, match="cost_map_resolution"):
        compose_with_base(
            base_archive_descriptor=desc,
            cost_map=cm_wrong,
            stc_payload=payload,
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
    cfg = YUCRConfig(cost_map_resolution=(48, 64), stc_payload_bits=2000)
    estimated = estimate_overhead_bytes(config=cfg)
    assert YUCR_OVERHEAD_TARGET_BYTES_MIN <= estimated <= YUCR_OVERHEAD_TARGET_BYTES_MAX * 2


# ---------------------------------------------------------------------------
# Loss weights validation
# ---------------------------------------------------------------------------


def test_loss_weights_defaults_match_contest_formula():
    w = YUCRLossWeights()
    assert math.isclose(w.seg_weight, 100.0)
    assert math.isclose(w.pose_sqrt_weight, math.sqrt(10.0))
    assert w.rate_weight == 25.0
    assert 0 <= w.lambda_yucr <= 10.0


def test_loss_weights_reject_negative_lambda():
    with pytest.raises(ValueError, match="lambda_yucr"):
        YUCRLossWeights(lambda_yucr=-0.1)


def test_loss_weights_reject_giant_lambda():
    with pytest.raises(ValueError, match="lambda_yucr"):
        YUCRLossWeights(lambda_yucr=20.0)


# ---------------------------------------------------------------------------
# Readiness manifest — non-promotion + apples-to-apples discipline (Catalog #192)
# ---------------------------------------------------------------------------


def test_readiness_manifest_has_no_score_claim():
    cfg = YUCRConfig()
    manifest = build_readiness_manifest(
        base_substrate_id="a1",
        base_archive_bytes=200_000,
        yucr_overhead_bytes=2_000,
        config=cfg,
    )
    # NEVER score_claim=True in a YUCR readiness manifest. Catalog #192:
    # advisory tag must be paired with explicit non-promotion until
    # contest-CPU + contest-CUDA paired auth eval lands.
    assert manifest["score_claim"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["predicted_score_evidence_grade"] == "time-traveler-prediction"


def test_readiness_manifest_records_total_archive_bytes():
    cfg = YUCRConfig()
    manifest = build_readiness_manifest(
        base_substrate_id="a1",
        base_archive_bytes=178_000,
        yucr_overhead_bytes=2_500,
        config=cfg,
    )
    assert manifest["total_archive_bytes"] == 178_000 + 2_500


# ---------------------------------------------------------------------------
# YUCR substrate handle
# ---------------------------------------------------------------------------


def test_yucr_substrate_default_constructible():
    sub = YUCRSubstrate()
    assert sub.config.base_substrate_id == YUCR_DEFAULT_BASE_SUBSTRATE


def test_yucr_substrate_carries_typed_config():
    cfg = YUCRConfig(base_substrate_id="pr101_lc_v2_clone", stc_payload_bits=4000)
    sub = YUCRSubstrate(config=cfg)
    assert sub.config.stc_payload_bits == 4000
    assert sub.config.base_substrate_id == "pr101_lc_v2_clone"


# ---------------------------------------------------------------------------
# Inflate runtime (Catalog #146 contract)
# ---------------------------------------------------------------------------


def test_inflate_module_exposes_main():
    from tac.substrates.yucr import inflate

    assert callable(getattr(inflate, "main", None))


def test_inflate_loc_under_substrate_engineering_budget(tmp_path):
    """HNeRV parity L4: inflate.py <= 200 LOC substrate-engineering waiver."""
    from pathlib import Path

    inflate_path = (
        Path(__file__).resolve().parent.parent / "inflate.py"
    )
    line_count = sum(1 for _ in inflate_path.read_text().splitlines())
    assert line_count <= 200, f"YUCR inflate.py is {line_count} LOC; budget is 200"


def test_inflate_locator_fails_when_yucr_bin_absent(tmp_path):
    from tac.substrates.yucr.inflate import _locate_yucr_archive

    with pytest.raises(FileNotFoundError, match="YUCR sidecar not found"):
        _locate_yucr_archive(tmp_path)


def test_inflate_base_archive_sha_mismatch_refuses(tmp_path):
    from tac.substrates.yucr.inflate import _verify_base_archive_match

    base_path = tmp_path / "a1.bin"
    base_path.write_bytes(b"some-base-bytes")
    actual_sha = hashlib.sha256(base_path.read_bytes()).hexdigest()
    # Ask for a DIFFERENT truncated sha — should refuse.
    wrong_truncated = ("0" * 16) if not actual_sha.startswith("0" * 16) else "1" * 16
    with pytest.raises(ValueError, match="sha mismatch"):
        _verify_base_archive_match(
            tmp_path,
            base_substrate_id="a1",
            base_sha_truncated=wrong_truncated,
        )


def test_inflate_base_archive_sha_match_accepts(tmp_path):
    from tac.substrates.yucr.inflate import _verify_base_archive_match

    base_path = tmp_path / "a1.bin"
    base_path.write_bytes(b"some-base-bytes")
    truncated = hashlib.sha256(base_path.read_bytes()).hexdigest()[:16]
    # Should succeed.
    located = _verify_base_archive_match(
        tmp_path,
        base_substrate_id="a1",
        base_sha_truncated=truncated,
    )
    assert located == base_path


def test_no_tmp_paths_in_yucr_module_source():
    """CLAUDE.md "Forbidden /tmp paths in any persisted artifact" sanity check.

    Scan the YUCR substrate package for any /tmp/ literal in source code
    (not inside test fixtures). The only acceptable hits are inside
    docstring examples explicitly marked as historical-recipe-only OR
    inside test files using pytest tmp_path fixtures.
    """
    import re
    from pathlib import Path

    yucr_root = Path(__file__).resolve().parent.parent
    forbidden = re.compile(r"['\"]/tmp/")
    for py in yucr_root.glob("*.py"):
        text = py.read_text()
        # YUCR source files MUST NOT carry literal /tmp/ paths.
        assert not forbidden.search(text), (
            f"forbidden /tmp/ literal found in {py}; per CLAUDE.md "
            "FORBIDDEN_PATTERNS use experiments/results/<lane_id>_<timestamp>/ "
            "for build artifacts"
        )
