# SPDX-License-Identifier: MIT
"""Tests for ``tac.fec6_format0d_extra`` codec (Ext 4 of fec6 stacking wave).

Design memo: ``.omx/research/fec6_plus_format0d_extra_design_20260517.md``
Lane: ``lane_fec6_stacking_wave_5_grammar_extensions_20260517``

Test surface:
- Round-trip byte-identity (encode → decode produces identical bytes)
- Sanity (canonical decode matches PR106 format0d apply_sidecar_corrections semantics)
- Edge cases (empty pairs, NO_OP sentinels, max u16 length)
- Byte-determinism (two encodes with identical inputs produce identical bytes)
- Wrap/unwrap symmetry (wrap then unwrap recovers original bytes)
- Negative cases (malformed payload rejected with typed exception)

Per CLAUDE.md Catalog #158 (deterministic byte-emit) + Catalog #287
(evidence-tag) + Catalog #295 (self-contained inflate.py).
"""
from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest

from tac.fec6_format0d_extra import (
    DEFAULT_DELTA_SCALE,
    EXTRA_MAGIC,
    NO_OP_DIM,
    Format0dExtraDecodeError,
    Format0dExtraEncodeError,
    decode_format0d_extra_payload,
    encode_format0d_extra_payload,
    unwrap_fec6_archive_with_extra,
    wrap_fec6_archive_with_extra,
)
from tools.build_fec6_plus_format0d_extra_packet import build_packet


@pytest.fixture
def canonical_pairs() -> tuple[np.ndarray, np.ndarray]:
    """A small canonical fixture mirroring PR106 format0d sidecar patterns."""
    # 10-pair fixture; some NO_OP, some corrections.
    dim_arr = np.array([5, 255, 12, 3, 27, 255, 14, 255, 0, 9], dtype=np.uint8)
    delta_q_arr = np.array([2, 0, -3, 1, 5, 0, -7, 0, 4, -1], dtype=np.int8)
    return dim_arr, delta_q_arr


def test_encode_emits_expected_total_size(canonical_pairs):
    dim_arr, delta_q_arr = canonical_pairs
    n_pairs = len(dim_arr)
    payload = encode_format0d_extra_payload(dim_arr=dim_arr, delta_q_arr=delta_q_arr)
    assert len(payload) == 4 + n_pairs + n_pairs + 2  # u32 + dim_arr + delta_q_arr + fp16
    assert len(payload) == 2 * n_pairs + 6


def test_encode_then_decode_byte_identical_roundtrip(canonical_pairs):
    dim_arr, delta_q_arr = canonical_pairs
    payload = encode_format0d_extra_payload(
        dim_arr=dim_arr, delta_q_arr=delta_q_arr, scale=DEFAULT_DELTA_SCALE
    )
    dim_back, delta_back, scale_back = decode_format0d_extra_payload(payload)
    assert np.array_equal(dim_back, dim_arr)
    assert np.array_equal(delta_back, delta_q_arr)
    assert scale_back == pytest.approx(DEFAULT_DELTA_SCALE, abs=1e-5)


def test_encode_determinism_two_encodes_byte_identical(canonical_pairs):
    """Catalog #158 byte-determinism: same input → same bytes."""
    dim_arr, delta_q_arr = canonical_pairs
    a = encode_format0d_extra_payload(dim_arr=dim_arr, delta_q_arr=delta_q_arr)
    b = encode_format0d_extra_payload(dim_arr=dim_arr, delta_q_arr=delta_q_arr)
    assert a == b


def test_encode_rejects_mismatched_length():
    dim_arr = np.zeros(10, dtype=np.uint8)
    delta_q_arr = np.zeros(11, dtype=np.int8)
    with pytest.raises(Format0dExtraEncodeError, match="must have same length"):
        encode_format0d_extra_payload(dim_arr=dim_arr, delta_q_arr=delta_q_arr)


def test_encode_rejects_2d_arrays():
    dim_arr = np.zeros((2, 3), dtype=np.uint8)
    delta_q_arr = np.zeros(6, dtype=np.int8)
    with pytest.raises(Format0dExtraEncodeError, match="must be 1-D"):
        encode_format0d_extra_payload(dim_arr=dim_arr, delta_q_arr=delta_q_arr)


def test_encode_rejects_zero_pairs():
    dim_arr = np.array([], dtype=np.uint8)
    delta_q_arr = np.array([], dtype=np.int8)
    with pytest.raises(Format0dExtraEncodeError, match="n_pairs must be positive"):
        encode_format0d_extra_payload(dim_arr=dim_arr, delta_q_arr=delta_q_arr)


def test_decode_rejects_too_short_payload():
    with pytest.raises(Format0dExtraDecodeError, match="too short"):
        decode_format0d_extra_payload(b"\x00\x00\x00")  # 3 bytes < 6 min


def test_decode_rejects_payload_size_mismatch():
    # Header claims 5 pairs but payload only has 2 bytes of data after header.
    bad = b"\x05\x00\x00\x00" + b"\x00" * 2  # u32 n_pairs=5, 2 trailing bytes
    with pytest.raises(Format0dExtraDecodeError, match="payload size mismatch"):
        decode_format0d_extra_payload(bad)


def test_apply_sidecar_corrections_semantics_match_pr106_format0d(canonical_pairs):
    """Sanity: the (dim_arr, delta_q_arr, scale) tuple applied per the
    PR106 format0d formula yields the expected per-pair corrections."""
    dim_arr, delta_q_arr = canonical_pairs
    scale = DEFAULT_DELTA_SCALE
    payload = encode_format0d_extra_payload(
        dim_arr=dim_arr, delta_q_arr=delta_q_arr, scale=scale
    )
    dim_back, delta_back, scale_back = decode_format0d_extra_payload(payload)

    # Simulate the inflate-side apply (mirroring PR106 format0d apply_sidecar_corrections)
    n_pairs = len(dim_arr)
    latent_dim = 28  # PR101's LATENT_DIM
    latents = np.zeros((n_pairs, latent_dim), dtype=np.float32)
    for p in range(n_pairs):
        d = int(dim_back[p])
        if d == NO_OP_DIM:
            continue
        if d >= latent_dim:
            # PR106 format0d would reject this; canonical dim values are < 28
            continue
        latents[p, d] += float(delta_back[p]) * scale_back

    # Verify expected applied corrections per the fixture.
    # Tolerance: fp16 round-trip of scale=0.01 introduces ~2e-4 relative
    # error (0.01 is not exactly representable in fp16 — the nearest
    # fp16 value is ~0.0100021362...). Use rel=2e-3 to absorb fp16
    # quantization noise while still catching real wire-format bugs
    # (a wrong dim_arr or delta_q_arr value would shift by >100%).
    tol = {"rel": 2e-3, "abs": 1e-6}
    assert latents[0, 5] == pytest.approx(2 * scale, **tol)
    assert np.all(latents[1, :] == 0.0)  # NO_OP_DIM pair
    assert latents[2, 12] == pytest.approx(-3 * scale, **tol)
    assert latents[3, 3] == pytest.approx(1 * scale, **tol)
    assert latents[4, 27] == pytest.approx(5 * scale, **tol)
    assert np.all(latents[5, :] == 0.0)
    assert latents[6, 14] == pytest.approx(-7 * scale, **tol)
    assert np.all(latents[7, :] == 0.0)
    assert latents[8, 0] == pytest.approx(4 * scale, **tol)
    assert latents[9, 9] == pytest.approx(-1 * scale, **tol)


def test_wrap_unwrap_roundtrip_preserves_base_and_extra(canonical_pairs):
    dim_arr, delta_q_arr = canonical_pairs
    extra_payload = encode_format0d_extra_payload(
        dim_arr=dim_arr, delta_q_arr=delta_q_arr
    )
    base = b"<<< fake fec6 base wrapper bytes >>>" * 100  # 3700 bytes
    wrapped = wrap_fec6_archive_with_extra(
        fec6_archive_bytes=base, extra_payload=extra_payload
    )
    assert wrapped.startswith(base)
    assert wrapped.endswith(extra_payload)
    base_back, extra_back = unwrap_fec6_archive_with_extra(wrapped)
    assert base_back == base
    assert extra_back == extra_payload


def test_unwrap_returns_none_when_no_extra_slot():
    base = b"a fec6 archive without an EXTRA slot" * 10
    base_back, extra_back = unwrap_fec6_archive_with_extra(base)
    assert base_back == base
    assert extra_back is None


def test_unwrap_handles_short_input():
    short = b"abc"  # less than 6 bytes
    base_back, extra_back = unwrap_fec6_archive_with_extra(short)
    assert base_back == short
    assert extra_back is None


def test_wrap_rejects_extra_payload_over_u16():
    base = b"x" * 100
    too_big = b"y" * (2**16)  # > u16 max
    with pytest.raises(Format0dExtraEncodeError, match="exceeds u16 max"):
        wrap_fec6_archive_with_extra(fec6_archive_bytes=base, extra_payload=too_big)


def test_extra_magic_pinned():
    """Regression guard: EXTRA_MAGIC is a stable wire-format constant."""
    assert EXTRA_MAGIC == b"FE6E"
    assert len(EXTRA_MAGIC) == 4


def test_default_delta_scale_matches_pr106_format0d():
    """The default scale matches PR106 format0d DELTA_SCALE for direct compatibility."""
    assert DEFAULT_DELTA_SCALE == 0.01


def test_no_op_dim_sentinel_matches_pr106_format0d():
    """The NO_OP_DIM sentinel matches PR106 format0d NO_OP_DIM for direct compatibility."""
    assert NO_OP_DIM == 255


def test_unwrap_with_extra_magic_substring_in_base_handled_correctly():
    """Adversarial: if base bytes contain EXTRA_MAGIC literal, the unwrap
    correctly finds the canonical trailing slot at the right position."""
    dim_arr = np.array([1, 2, 3], dtype=np.uint8)
    delta_q_arr = np.array([1, -1, 0], dtype=np.int8)
    extra_payload = encode_format0d_extra_payload(
        dim_arr=dim_arr, delta_q_arr=delta_q_arr
    )
    # Base bytes contain the magic; the unwrap should still find the
    # canonical trailing position via the u16 length anchor.
    base = b"prefix" + EXTRA_MAGIC + b"random-bytes" + EXTRA_MAGIC + b"more"
    wrapped = wrap_fec6_archive_with_extra(
        fec6_archive_bytes=base, extra_payload=extra_payload
    )
    base_back, extra_back = unwrap_fec6_archive_with_extra(wrapped)
    assert base_back == base
    assert extra_back == extra_payload


def test_encode_with_all_no_op_pairs_produces_valid_payload():
    """All-NO_OP pairs should still produce a valid wire-format payload."""
    dim_arr = np.full(50, NO_OP_DIM, dtype=np.uint8)
    delta_q_arr = np.zeros(50, dtype=np.int8)
    payload = encode_format0d_extra_payload(
        dim_arr=dim_arr, delta_q_arr=delta_q_arr
    )
    dim_back, delta_back, scale_back = decode_format0d_extra_payload(payload)
    assert np.array_equal(dim_back, dim_arr)
    assert np.array_equal(delta_back, delta_q_arr)
    assert all(d == NO_OP_DIM for d in dim_back)


def test_build_packet_manifest_fails_closed_while_runtime_scaffold_only(tmp_path):
    """The Phase 1 builder must not emit dispatch authority before inflate wiring."""
    fec6_archive = tmp_path / "fec6_source.zip"
    with zipfile.ZipFile(fec6_archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("x", b"FP11 fake fec6 inner member")

    corrections_json = tmp_path / "corrections.json"
    corrections_json.write_text(
        json.dumps(
            {
                "n_pairs": 3,
                "dim_arr": [1, NO_OP_DIM, 2],
                "delta_q_arr": [3, 0, -2],
                "scale": DEFAULT_DELTA_SCALE,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "packet"
    manifest = build_packet(
        fec6_archive=fec6_archive,
        extra_corrections_json=corrections_json,
        output_dir=output_dir,
    )

    assert (output_dir / "archive.zip").is_file()
    assert (output_dir / "inflate.py").is_file()
    assert "NotImplementedError" in (output_dir / "inflate.py").read_text(
        encoding="utf-8"
    )

    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["rank_or_kill_eligible"] is False
    assert manifest["ready_for_provider_dispatch"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["dispatch_attempted"] is False
    assert manifest["runtime_scaffold_only"] is True
    assert manifest["runtime_consumption_proof"] is False
    assert manifest["byte_consumption_proof"] is False
    assert "fec6_base_inflate_path_not_wired" in manifest["dispatch_blockers"]
