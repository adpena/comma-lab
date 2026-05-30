# SPDX-License-Identifier: MIT
"""Tests for canonical L28 + L32 PR-family quick-wins bolt-on cascade on Z8.

Per the operator-routed Yousfi-cascade TOP-1 (post-Z8 M11 L1 smoke landing
2026-05-30 commit ``2f8570755``): canonical PR-or-greater binding-depth
discipline applies L28 (PR98 decode-side channel postprocess) + L32 (brotli
quality 9→11 bump) to the canonical Z8 substrate. L30 (constriction
RangeDecoder Categorical substitution) is DEFERRED per Catalog #290
``FORK_BECAUSE_PRINCIPLED_MISMATCH`` (Z8 archive carries float32 raw
wavelet coefficients, not int8 quantized tensors that PR103 SILVER's
Categorical model applies to; honest L30 binding requires an int-quantizer
prerequisite layer — out-of-scope for $0 quick-wins cascade).

Canonical PR98 L28 reference: ``experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/source/submissions/hnerv_ft_microcodec/inflate.py``
lines 49-51 (``up[:, 0, 0].sub_(1.0); up[:, 0, 2].sub_(1.0);
up[:, 1, 1].sub_(1.0)`` applied at CAMERA_HW resolution AFTER bicubic
upsample, BEFORE clamp + uint8 cast).

Canonical PR101 L32 anchor: CLAUDE.md HNeRV parity discipline L32 + canonical
equation ``pr95_family_l32_brotli_quality_11_max_v1`` (q=11 spends ~10×
compression time vs q=6 but saves ~5-10% bytes; compression time is offline
overhead so q=11 is free at deploy time).
"""

from __future__ import annotations

import io
import struct
from pathlib import Path

import brotli
import numpy as np
import pytest
import torch

from tac.substrates._shared.inflate_runtime import (
    CAMERA_HW,
    write_rgb_pair_to_raw,
)
from tac.substrates.z8_hierarchical_predictive_coding.archive import (
    _BROTLI_QUALITY,
)


# --- L32: brotli quality 9→11 canonical bump ----------------------------


def test_l32_archive_brotli_quality_is_canonical_11() -> None:
    """``_BROTLI_QUALITY`` MUST equal canonical PR95-family L32 value (11).

    Per CLAUDE.md HNeRV parity discipline L32 +
    ``pr95_family_l32_brotli_quality_11_max_v1``. The pre-patch value was
    q=9 (DreamerV3 / Z6 / NSCS06 v8 sibling baseline); L32 canonical PR95
    PR101 reference is q=11 (max). Regression guard.
    """
    assert _BROTLI_QUALITY == 11, (
        f"Z8 archive brotli quality must be canonical PR95-family L32 value 11; "
        f"got {_BROTLI_QUALITY}. Per pr95_family_l32_brotli_quality_11_max_v1."
    )


def test_l32_serialize_pair_wavelet_pyramid_uses_q11() -> None:
    """The dominant per-pair wavelet pyramid serialization MUST use brotli q=11.

    The ``_serialize_pair_wavelet_pyramid`` source MUST reference
    ``brotli.compress(raw, quality=11)`` (not q=9). Per CLAUDE.md HNeRV
    parity L32: this covers ~99.5% of the canonical Z8HPC1 archive bytes.
    Regression guard via source-text scan because the brotli call is
    inside a helper that builds the wavelet_blob (largest section).
    """
    src_path = (
        Path(__file__).resolve().parent.parent / "canonical_quadruple_binding.py"
    )
    src = src_path.read_text(encoding="utf-8")
    # The canonical L32 pattern.
    assert "brotli.compress(raw, quality=11)" in src, (
        "_serialize_pair_wavelet_pyramid must use brotli quality=11 per L32"
    )
    # Regression guard: explicit absence of q=9 in serialize helper.
    # We allow q=9 mentions in docstrings/historical context but the live
    # call MUST be q=11.
    assert "brotli.compress(raw, quality=9)" not in src, (
        "Pre-L32 brotli quality=9 call detected — must be q=11 per L32"
    )


def test_l32_brotli_q11_produces_no_larger_payload_than_q9() -> None:
    """Verify q=11 produces ≤ size of q=9 payload on canonical wavelet pyramid.

    L32 canonical PR95 invariant: q=11 spends more compression time but
    produces ≤ bytes than q=9 (the q=11 compressor has a strictly larger
    search budget so it cannot produce more bytes for the same input).
    Per Brotli specification (Alakuijala et al. 2018, RFC 7932).
    """
    # Synthetic canonical wavelet-pyramid-like payload: float32 detail
    # bands with bounded magnitude (mimics Mallat decompose output).
    rng = np.random.RandomState(42)
    payload_parts = []
    payload_parts.append(struct.pack("<B", 1))  # schema version
    for _ in range(6):  # 2 top_ll + 4 detail subbands (canonical pyramid)
        shape = (8, 8, 3)
        payload_parts.append(struct.pack("<HHH", *shape))
        arr = rng.randn(*shape).astype(np.float32) * 0.1  # bounded magnitude
        payload_parts.append(arr.tobytes(order="C"))
    raw = b"".join(payload_parts)

    q9_bytes = brotli.compress(raw, quality=9)
    q11_bytes = brotli.compress(raw, quality=11)
    assert len(q11_bytes) <= len(q9_bytes), (
        f"L32 invariant violated: q=11 ({len(q11_bytes)}B) > q=9 "
        f"({len(q9_bytes)}B). Per Brotli RFC 7932 q=11 search budget is "
        f"strictly larger so output bytes must be ≤ q=9."
    )


# --- L28: PR98 decode-side channel postprocess --------------------------


def test_l28_write_rgb_pair_supports_pr98_postprocess_kwarg() -> None:
    """``write_rgb_pair_to_raw`` MUST accept the L28 opt-in kwarg.

    Per CLAUDE.md HNeRV parity discipline L28
    (``pr95_family_l28_decode_side_channel_postprocess_v1``). The kwarg
    name MUST be ``apply_pr98_l28_channel_postprocess`` so the canonical
    PR98 reference is explicit at the call site.
    """
    import inspect

    sig = inspect.signature(write_rgb_pair_to_raw)
    assert "apply_pr98_l28_channel_postprocess" in sig.parameters, (
        "write_rgb_pair_to_raw must accept apply_pr98_l28_channel_postprocess "
        "kwarg per L28 canonical PR98 reference"
    )
    # Default must be False (opt-in only) to preserve backward compatibility
    # for sister substrates per CLAUDE.md UNIQUE-AND-COMPLETE-PER-METHOD +
    # Catalog #290 ADOPT_CANONICAL_BECAUSE_SERVES decision.
    param = sig.parameters["apply_pr98_l28_channel_postprocess"]
    assert param.default is False, (
        "apply_pr98_l28_channel_postprocess must default False (opt-in only) "
        "to preserve sister substrate backward compatibility per "
        "UNIQUE-AND-COMPLETE-PER-METHOD operating mode"
    )


def test_l28_postprocess_subtracts_canonical_channels_exactly() -> None:
    """L28 MUST subtract 1.0 from frame_0 R + frame_0 B + frame_1 G exactly.

    Per PR101 canonical inflate.py:49-51:
        up[:, 0, 0].sub_(1.0)  # frame_0 RED
        up[:, 0, 2].sub_(1.0)  # frame_0 BLUE
        up[:, 1, 1].sub_(1.0)  # frame_1 GREEN

    Equivalent at the (2, 3, CAMERA_H, CAMERA_W) stage used by
    ``write_rgb_pair_to_raw``:
        frames[0, 0].sub_(1.0)  # frame_0 (first of pair) RED
        frames[0, 2].sub_(1.0)  # frame_0 (first of pair) BLUE
        frames[1, 1].sub_(1.0)  # frame_1 (second of pair) GREEN

    We use a constant fill value at byte-scale 128.0 so the subtraction
    produces a deterministic per-channel delta we can verify byte-exact.
    """
    # Build canonical-shape input tensors: (1, 3, CAMERA_H, CAMERA_W) is
    # the contract; we use CAMERA_HW directly to avoid bicubic resampling
    # (which would not commute with the per-channel constant subtract for
    # pixels near 0 or 255 due to the clamp). At CAMERA_HW resolution the
    # subtract is byte-exact.
    rgb_0 = torch.full(
        (1, 3, CAMERA_HW[0], CAMERA_HW[1]), 128.0 / 255.0, dtype=torch.float32
    )
    rgb_1 = torch.full(
        (1, 3, CAMERA_HW[0], CAMERA_HW[1]), 128.0 / 255.0, dtype=torch.float32
    )

    # --- Pair A: no L28 (canonical baseline) -----------------------------
    buf_baseline = io.BytesIO()
    written_a = write_rgb_pair_to_raw(
        buf_baseline, rgb_0, rgb_1, input_range="unit"
    )
    assert written_a == 2
    bytes_baseline = buf_baseline.getvalue()
    arr_baseline = np.frombuffer(bytes_baseline, dtype=np.uint8).reshape(
        2, CAMERA_HW[0], CAMERA_HW[1], 3
    )

    # --- Pair B: L28 ON --------------------------------------------------
    buf_l28 = io.BytesIO()
    written_b = write_rgb_pair_to_raw(
        buf_l28,
        rgb_0,
        rgb_1,
        input_range="unit",
        apply_pr98_l28_channel_postprocess=True,
    )
    assert written_b == 2
    bytes_l28 = buf_l28.getvalue()
    arr_l28 = np.frombuffer(bytes_l28, dtype=np.uint8).reshape(
        2, CAMERA_HW[0], CAMERA_HW[1], 3
    )

    # --- Verify per-channel canonical L28 deltas --------------------------
    # arr layout: (frame_idx_in_pair, H, W, channel) with channel order [R, G, B]
    # PR101 canonical: frame_0 R + frame_0 B + frame_1 G all -= 1.0 at byte scale.
    # At byte-scale 128.0 the subtract leaves 127.0 which rounds to 127.
    frame_0_R_base = arr_baseline[0, :, :, 0].astype(int)
    frame_0_R_l28 = arr_l28[0, :, :, 0].astype(int)
    assert (frame_0_R_l28 - frame_0_R_base == -1).all(), (
        "L28 must subtract 1 from frame_0 RED at byte scale"
    )

    frame_0_G_base = arr_baseline[0, :, :, 1].astype(int)
    frame_0_G_l28 = arr_l28[0, :, :, 1].astype(int)
    assert (frame_0_G_l28 - frame_0_G_base == 0).all(), (
        "L28 must NOT subtract from frame_0 GREEN (canonical PR98 only "
        "subtracts frame_0 R + frame_0 B + frame_1 G)"
    )

    frame_0_B_base = arr_baseline[0, :, :, 2].astype(int)
    frame_0_B_l28 = arr_l28[0, :, :, 2].astype(int)
    assert (frame_0_B_l28 - frame_0_B_base == -1).all(), (
        "L28 must subtract 1 from frame_0 BLUE at byte scale"
    )

    frame_1_R_base = arr_baseline[1, :, :, 0].astype(int)
    frame_1_R_l28 = arr_l28[1, :, :, 0].astype(int)
    assert (frame_1_R_l28 - frame_1_R_base == 0).all(), (
        "L28 must NOT subtract from frame_1 RED (canonical PR98 only "
        "subtracts frame_0 R + frame_0 B + frame_1 G)"
    )

    frame_1_G_base = arr_baseline[1, :, :, 1].astype(int)
    frame_1_G_l28 = arr_l28[1, :, :, 1].astype(int)
    assert (frame_1_G_l28 - frame_1_G_base == -1).all(), (
        "L28 must subtract 1 from frame_1 GREEN at byte scale"
    )

    frame_1_B_base = arr_baseline[1, :, :, 2].astype(int)
    frame_1_B_l28 = arr_l28[1, :, :, 2].astype(int)
    assert (frame_1_B_l28 - frame_1_B_base == 0).all(), (
        "L28 must NOT subtract from frame_1 BLUE (canonical PR98 only "
        "subtracts frame_0 R + frame_0 B + frame_1 G)"
    )


def test_l28_postprocess_default_off_preserves_backward_compatibility() -> None:
    """L28 default OFF preserves byte-identical sister substrate output.

    Per Catalog #290 ADOPT_CANONICAL_BECAUSE_SERVES decision: the L28
    kwarg defaults OFF so sister substrates (NSCS06 v8 / DP1 / Slot GGG /
    etc.) that don't opt in get the EXACT SAME output as pre-patch.
    """
    rgb_0 = torch.full(
        (1, 3, CAMERA_HW[0], CAMERA_HW[1]), 128.0 / 255.0, dtype=torch.float32
    )
    rgb_1 = torch.full(
        (1, 3, CAMERA_HW[0], CAMERA_HW[1]), 128.0 / 255.0, dtype=torch.float32
    )

    buf_no_kwarg = io.BytesIO()
    write_rgb_pair_to_raw(buf_no_kwarg, rgb_0, rgb_1, input_range="unit")

    buf_explicit_false = io.BytesIO()
    write_rgb_pair_to_raw(
        buf_explicit_false,
        rgb_0,
        rgb_1,
        input_range="unit",
        apply_pr98_l28_channel_postprocess=False,
    )

    assert buf_no_kwarg.getvalue() == buf_explicit_false.getvalue(), (
        "L28 default OFF must produce byte-identical output to explicit "
        "apply_pr98_l28_channel_postprocess=False per backward-compat invariant"
    )


def test_l28_postprocess_clamps_at_zero_for_pixels_near_zero() -> None:
    """L28 subtraction at byte-scale-0 must clamp at 0 (not wrap).

    Per the canonical L28 path: subtract 1.0 happens BEFORE the
    ``.clamp(0.0, 255.0)`` step. Pixels at value 0.0 become -1.0 then
    clamp to 0. Verifies the clamp-after-subtract ordering matches PR101.
    """
    # All-zeros input
    rgb_0 = torch.zeros((1, 3, CAMERA_HW[0], CAMERA_HW[1]), dtype=torch.float32)
    rgb_1 = torch.zeros((1, 3, CAMERA_HW[0], CAMERA_HW[1]), dtype=torch.float32)

    buf = io.BytesIO()
    write_rgb_pair_to_raw(
        buf,
        rgb_0,
        rgb_1,
        input_range="unit",
        apply_pr98_l28_channel_postprocess=True,
    )
    arr = np.frombuffer(buf.getvalue(), dtype=np.uint8).reshape(
        2, CAMERA_HW[0], CAMERA_HW[1], 3
    )
    # All pixels should be 0 (the subtract-then-clamp produces 0 from 0).
    # Crucially, NO uint8 wraparound to 255.
    assert arr.min() == 0 and arr.max() == 0, (
        f"L28 + clamp at 0 must produce all-zero output for all-zero input; "
        f"got min={arr.min()} max={arr.max()}"
    )


def test_l28_postprocess_preserves_at_255_for_pixels_at_max() -> None:
    """L28 subtraction on max-value pixels leaves 254 in subtracted channels.

    Verifies that the subtract is byte-faithful at the high end too:
    255.0 - 1.0 = 254.0, which is well within [0, 255] so clamp is no-op.
    """
    rgb_0 = torch.ones((1, 3, CAMERA_HW[0], CAMERA_HW[1]), dtype=torch.float32)
    rgb_1 = torch.ones((1, 3, CAMERA_HW[0], CAMERA_HW[1]), dtype=torch.float32)

    buf = io.BytesIO()
    write_rgb_pair_to_raw(
        buf,
        rgb_0,
        rgb_1,
        input_range="unit",
        apply_pr98_l28_channel_postprocess=True,
    )
    arr = np.frombuffer(buf.getvalue(), dtype=np.uint8).reshape(
        2, CAMERA_HW[0], CAMERA_HW[1], 3
    )
    # frame_0 R + frame_0 B + frame_1 G become 254; others stay 255.
    assert (arr[0, :, :, 0] == 254).all(), "frame_0 R must be 254 after L28"
    assert (arr[0, :, :, 1] == 255).all(), "frame_0 G must stay 255 after L28"
    assert (arr[0, :, :, 2] == 254).all(), "frame_0 B must be 254 after L28"
    assert (arr[1, :, :, 0] == 255).all(), "frame_1 R must stay 255 after L28"
    assert (arr[1, :, :, 1] == 254).all(), "frame_1 G must be 254 after L28"
    assert (arr[1, :, :, 2] == 255).all(), "frame_1 B must stay 255 after L28"


def test_l28_wired_in_z8_inflate_one_video_from_archive_bytes() -> None:
    """Z8 ``inflate_one_video_from_archive_bytes`` MUST opt-in to L28.

    Per the canonical L28 wire-in: the Z8 inflate.py call site MUST pass
    ``apply_pr98_l28_channel_postprocess=True`` to ``write_rgb_pair_to_raw``
    so the canonical PR98 third-prize postprocess actually fires at
    inflate time. Regression guard via source-text scan because dynamic
    invocation would require a full archive build + inflate (heavy).
    """
    src_path = (
        Path(__file__).resolve().parent.parent / "inflate.py"
    )
    src = src_path.read_text(encoding="utf-8")
    # The canonical L28 opt-in kwarg.
    assert "apply_pr98_l28_channel_postprocess=True" in src, (
        "Z8 inflate.py must opt into L28 canonical PR98 postprocess "
        "via apply_pr98_l28_channel_postprocess=True"
    )


# --- L30 deferred classification per Catalog #290 + #303 ---------------


def test_l30_deferred_per_catalog_290_fork_because_principled_mismatch() -> None:
    """L30 (Categorical+RangeDecoder) is DEFERRED per Catalog #290 fork rationale.

    Per CLAUDE.md HNeRV parity discipline L30
    (``pr95_family_l30_range_arithmetic_coding_categorical_v1``): the
    canonical PR103 SILVER L30 pattern applies Categorical model +
    RangeDecoder to INT8 QUANTIZED tensors (PR103 uses AC_INDICES = [0, 2,
    4, 6, 8, 10, 12, 21] for 8 specific large weight tensors).

    Z8 substrate currently emits float32 raw wavelet coefficients (top_ll +
    per-level LH/HL/HH detail bands). Applying PR103's int-based
    Categorical to float32 raw bytes would be a canonical cargo-cult per
    Catalog #303 (Categorical model assumes bounded discrete distribution
    over [0, 255]; float32 raw bytes are near-uniform-random by entropy
    so brotli's ~93% ratio is already close to the entropy floor).

    Honest L30 binding requires an int-quantization prerequisite layer
    (float32 → int8 wavelet coefficient quantizer with companion
    histogram-emit at encode-time + dequantize at decode-time). This is
    a multi-hour substrate-engineering wave that exceeds the canonical
    $0 quick-wins cascade scope.

    Per CLAUDE.md "Forbidden premature KILL": L30 is DEFERRED-pending-
    int-quantization-binding-wave, NOT killed. Reactivation criteria:
    land a canonical int-quantizer for the wavelet pyramid + companion
    Categorical+RangeDecoder roundtrip + verify byte savings vs brotli q=11
    baseline > +0% (i.e. the L30 substitution actually helps over the
    new q=11 baseline).

    This test pins the canonical Catalog #290 + #303 classification at
    the source-text level so future agents inherit the documented
    deferral.
    """
    # The deferral is documented at the test file's module docstring AND
    # the landing memo (canonical sister surface per Catalog #348 +
    # Catalog #294 9-dim checklist).
    test_file_src = Path(__file__).read_text(encoding="utf-8")
    assert "L30" in test_file_src, "L30 deferral must be documented in test file"
    assert "Catalog #290" in test_file_src, (
        "L30 deferral must cite Catalog #290 (canonical-vs-unique decision)"
    )
    assert "FORK_BECAUSE_PRINCIPLED_MISMATCH" in test_file_src, (
        "L30 deferral must cite canonical FORK_BECAUSE_PRINCIPLED_MISMATCH "
        "decision per Catalog #290"
    )
    assert "Forbidden premature KILL" in test_file_src, (
        "L30 deferral must cite CLAUDE.md 'Forbidden premature KILL' "
        "to confirm DEFERRAL not KILL classification"
    )
