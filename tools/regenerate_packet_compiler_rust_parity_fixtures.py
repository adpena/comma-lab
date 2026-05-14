#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Regenerate the binary input fixtures consumed by Rust parity tests.

# no-argparse-OK: parameterless regenerator; no operator options to surface.


The Rust port of `tac.packet_compiler` lives in
`runtime-rs/crates/tac-packet-compiler/`. Its byte-for-byte parity tests
(`tests/golden_vector_parity.rs`) read the canonical Python inputs from
`src/tac/packet_compiler/golden_vectors/*_v1_*.bin` so they don't need to
reproduce numpy's RNG / linspace / fp16 quantisation behaviour in Rust.

Run this script whenever the Python recipe changes (e.g. seed update,
new column count). The Python golden-vector `*.json` files MUST also be
regenerated in lockstep — those carry the SHA-256 of the bytes the Rust
port must produce.

Per CLAUDE.md "Beauty, simplicity, and developer experience": this is the
documented deterministic recipe behind the binary fixtures so an OSS
implementer can reproduce them in any language.

No GPU, no MPS, no /tmp paths.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


_GOLDEN_DIR = Path(__file__).resolve().parent.parent / "src/tac/packet_compiler/golden_vectors"


def write_centered_delta_uint8_inputs() -> None:
    """fp32 row-major `(40, 6)` + fp16 mins + fp16 scales (PR101 recipe).

    Matches `test_centered_delta_golden_vector` in
    `src/tac/tests/test_packet_compiler_pr101_sidecar_grammar.py`.
    """
    rng = np.random.default_rng(seed=20260511)
    mins = np.linspace(-1.0, -0.5, 6, dtype=np.float16)
    scales = np.full(6, 2.0 / 255.0, dtype=np.float16)
    values = (
        mins.astype(np.float32)[None, :]
        + rng.uniform(0.0, 1.8, size=(40, 6)).astype(np.float32)
    )
    (_GOLDEN_DIR / "centered_delta_uint8_v1_input.bin").write_bytes(
        values.astype("<f4").tobytes()
    )
    (_GOLDEN_DIR / "centered_delta_uint8_v1_mins.bin").write_bytes(
        np.asarray(mins, dtype="<f2").tobytes()
    )
    (_GOLDEN_DIR / "centered_delta_uint8_v1_scales.bin").write_bytes(
        np.asarray(scales, dtype="<f2").tobytes()
    )


def write_split_brotli_streams_input() -> None:
    """Pinned 3-stream test input for the PR101 split-Brotli primitive.

    Format on disk: `[n_streams: u32 LE][len_0: u32 LE][bytes_0][len_1: u32 LE][bytes_1]…`.
    Matches `test_split_brotli_golden_vector`.
    """
    streams = [
        b"PR101 sidecar grammar conformance vector A " * 16,
        b"Reusable byte primitives, deterministic build " * 12,
        b"Future Rust/Zig port must match these bytes " * 8,
    ]
    out = bytearray()
    out += struct.pack("<I", len(streams))
    for s in streams:
        out += struct.pack("<I", len(s))
        out += s
    (_GOLDEN_DIR / "split_brotli_self_delim_v1_streams.bin").write_bytes(bytes(out))


def write_latent_hi_inputs() -> None:
    """uint16 latents + fp64 histogram for PR103 latent-hi arithmetic.

    Matches `test_latent_hi_arithmetic_golden_vector`.
    """
    rng = np.random.default_rng(seed=20260511)
    n = 1000
    latents = rng.integers(0, 16, size=n).astype(np.uint16)
    latents[::50] = 300  # inject some high-byte values
    hi = ((latents.astype(np.int32) >> 8) & 0xFF).astype(np.int32)
    histogram = np.bincount(hi, minlength=256).astype(np.float64) + 1.0
    (_GOLDEN_DIR / "latent_hi_arithmetic_v1_latents.bin").write_bytes(
        np.asarray(latents, dtype="<u2").tobytes()
    )
    (_GOLDEN_DIR / "latent_hi_arithmetic_v1_histogram.bin").write_bytes(
        np.asarray(histogram, dtype="<f8").tobytes()
    )


def write_merged_range_stream_inputs() -> None:
    """Multi-tensor int32 flat symbol stream + per-tensor fp64 histogram for
    PR103 merged-range-stream arithmetic.

    Matches `test_merged_range_stream_golden_vector` in
    ``src/tac/tests/test_packet_compiler_pr103_arithmetic_coding.py``.

    Fixture file layout:

    ``flat.bin``    - concatenated int32 LE symbols (60 + 80 + 36 = 176
                      values; each tensor is in C-order, post-offset-128)
    ``hist0.bin``   - fp64 LE histogram for tensor 0 (256 floats)
    ``hist1.bin``   - fp64 LE histogram for tensor 1 (256 floats)
    ``hist2.bin``   - fp64 LE histogram for tensor 2 (256 floats)

    The shape metadata `(60,)`, `(10, 8)`, `(3, 3, 4)` is pinned in the
    Rust test (matching the JSON manifest's ``tensor_shapes`` field).
    """
    rng = np.random.default_rng(seed=20260511)
    shapes = [(60,), (10, 8), (3, 3, 4)]
    flat_int32: list[np.ndarray] = []
    histograms: list[np.ndarray] = []
    for sh in shapes:
        size = int(np.prod(sh))
        raw = rng.integers(-12, 13, size=size).astype(np.int32)
        symbols = (raw + 128).astype(np.int32).reshape(sh)
        hist = np.bincount(
            symbols.reshape(-1), minlength=256
        ).astype(np.float64) + 1.0
        flat_int32.append(symbols.reshape(-1))
        histograms.append(hist)
    flat = np.concatenate(flat_int32).astype(np.int32)
    (_GOLDEN_DIR / "merged_range_stream_v1_flat.bin").write_bytes(
        np.asarray(flat, dtype="<i4").tobytes()
    )
    for i, hist in enumerate(histograms):
        (_GOLDEN_DIR / f"merged_range_stream_v1_hist{i}.bin").write_bytes(
            np.asarray(hist, dtype="<f8").tobytes()
        )


def write_pr93_delta_varint_pose_inputs() -> None:
    """fp32 (16, 4) poses + fp32 (4,) lo + fp32 (4,) scale for PR93 codec.

    Matches `test_delta_varint_pose_golden_vector` in
    ``src/tac/tests/test_packet_compiler_pr93_pose_codec.py``.

    The Python recipe is:

    ``rng = np.random.default_rng(seed=20260511); poses = rng.uniform(0, 1, (16, 4))``
    ``lo = zeros(4); scale = ones(4) / 255.0; bits = 8``
    """
    rng = np.random.default_rng(seed=20260511)
    n_rows, n_dims = 16, 4
    poses = rng.uniform(0.0, 1.0, size=(n_rows, n_dims)).astype(np.float32)
    lo = np.full(n_dims, 0.0, dtype=np.float32)
    scale = np.full(n_dims, 1.0 / 255.0, dtype=np.float32)
    (_GOLDEN_DIR / "pr93_delta_varint_pose_v1_poses.bin").write_bytes(
        np.asarray(poses, dtype="<f4").tobytes()
    )
    (_GOLDEN_DIR / "pr93_delta_varint_pose_v1_lo.bin").write_bytes(
        np.asarray(lo, dtype="<f4").tobytes()
    )
    (_GOLDEN_DIR / "pr93_delta_varint_pose_v1_scale.bin").write_bytes(
        np.asarray(scale, dtype="<f4").tobytes()
    )


def write_pr91_arithmetic_coder_inputs() -> None:
    """int32 symbol stream + (n_symbols, alphabet) fp64 probability matrix
    for PR91 per-symbol arithmetic coder.

    Matches `test_arithmetic_coder_constriction_golden_vector` in
    ``src/tac/tests/test_packet_compiler_pr91_hpac_grammar.py``.

    The Python recipe builds a `(200, 8)` deterministic peaked-at-0 prob
    matrix that drifts across positions, samples 200 symbols by inverse-cdf
    via ``rng.random``, then range-encodes with one Categorical per symbol.
    """
    rng = np.random.default_rng(seed=20260511)
    n_symbols = 200
    alphabet = 8
    probs = np.full((n_symbols, alphabet), 0.02, dtype=np.float64)
    peak = (np.arange(n_symbols) // 50).astype(np.int64) % alphabet
    for i in range(n_symbols):
        probs[i, peak[i]] = 1.0 - 0.02 * (alphabet - 1)
    cdf = np.cumsum(probs, axis=1)
    u = rng.random(n_symbols)
    symbols = np.array(
        [int(np.searchsorted(cdf[i], u[i])) for i in range(n_symbols)],
        dtype=np.int32,
    )
    (_GOLDEN_DIR / "pr91_arithmetic_coder_constriction_v1_symbols.bin").write_bytes(
        np.asarray(symbols, dtype="<i4").tobytes()
    )
    (_GOLDEN_DIR / "pr91_arithmetic_coder_constriction_v1_probs.bin").write_bytes(
        np.asarray(probs, dtype="<f8").tobytes()
    )


def write_pr84_adaptive_mask_inputs() -> None:
    """(n_contexts, alphabet) fp64 cdf table + int32 symbols + int32 contexts
    for PR84 adaptive-context arithmetic coding.

    Matches `test_adaptive_mask_context_golden_vector` in
    ``src/tac/tests/test_packet_compiler_pr84_adaptive_mask.py``.

    Recipe: 4-context × 5-class cdf where each context peaks (0.8) on its
    own symbol; 256 symbols whose context cycles 0..3 in raster fashion;
    each symbol drawn by inverse-cdf from ``rng.random``.
    """
    n_contexts = 4
    alphabet = 5
    cdf = np.full((n_contexts, alphabet), 0.05, dtype=np.float64)
    for ctx in range(n_contexts):
        cdf[ctx, ctx] = 0.8
    cdf /= cdf.sum(axis=1, keepdims=True)
    rng = np.random.default_rng(seed=20260511)
    n_symbols = 256
    context_ids = (np.arange(n_symbols) % n_contexts).astype(np.int32)
    u = rng.random(n_symbols)
    cum = np.cumsum(cdf, axis=1)
    symbols = np.array(
        [int(np.searchsorted(cum[context_ids[i]], u[i])) for i in range(n_symbols)],
        dtype=np.int32,
    )
    (_GOLDEN_DIR / "pr84_adaptive_mask_context_v1_cdf.bin").write_bytes(
        np.asarray(cdf, dtype="<f8").tobytes()
    )
    (_GOLDEN_DIR / "pr84_adaptive_mask_context_v1_symbols.bin").write_bytes(
        np.asarray(symbols, dtype="<i4").tobytes()
    )
    (_GOLDEN_DIR / "pr84_adaptive_mask_context_v1_contexts.bin").write_bytes(
        np.asarray(context_ids, dtype="<i4").tobytes()
    )


def write_pr81_fp4_codebook_inputs() -> None:
    """fp32 (64,) values + fp32 (2,) scales for the PR81 FP4 codebook +
    pack-nibbles golden vector.

    Matches `test_fp4_codebook_golden_vector` in
    ``src/tac/tests/test_packet_compiler_pr81_quantizr.py``.

    Recipe: ``rng=np.random.default_rng(20260511);
    values = rng.uniform(-3, 3, 64).astype(np.float32);
    scales = np.array([1.5, 0.75], dtype=np.float32);
    block_size=32``.
    """
    rng = np.random.default_rng(seed=20260511)
    values = rng.uniform(-3.0, 3.0, size=64).astype(np.float32)
    scales = np.array([1.5, 0.75], dtype=np.float32)
    (_GOLDEN_DIR / "pr81_fp4_codebook_v1_values.bin").write_bytes(
        np.asarray(values, dtype="<f4").tobytes()
    )
    (_GOLDEN_DIR / "pr81_fp4_codebook_v1_scales.bin").write_bytes(
        np.asarray(scales, dtype="<f4").tobytes()
    )


def write_pr81_router_action_inputs() -> None:
    """uint8 (600,) action stream for the PR81 ROUTER_ACTION packing
    golden vector.

    Matches `test_router_action_golden_vector` in
    ``src/tac/tests/test_packet_compiler_pr81_quantizr.py``.

    Recipe: ``rng=np.random.default_rng(20260511);
    actions = rng.integers(0, 8, size=600, dtype=np.uint8); bits=3``.
    """
    rng = np.random.default_rng(seed=20260511)
    actions = rng.integers(0, 8, size=600, dtype=np.uint8)
    (_GOLDEN_DIR / "pr81_router_action_v1_actions.bin").write_bytes(
        np.asarray(actions, dtype=np.uint8).tobytes()
    )


def write_pr92_rmc_joint_stream_inputs() -> None:
    """uint8 (128,) seg + uint8 (120,) actions for the PR92 RMC1 joint
    stream golden vector.

    Matches `test_rmc_joint_stream_golden_vector` in
    ``src/tac/tests/test_packet_compiler_pr92_joint_stream.py``.

    Recipe: ``rng=np.random.default_rng(20260511);
    seg_bytes = rng.integers(0, 256, size=128, dtype=np.uint8);
    actions = rng.integers(0, 8, size=120, dtype=np.uint8);
    action_bits=3, table_id=2``.
    """
    rng = np.random.default_rng(seed=20260511)
    seg_bytes = rng.integers(0, 256, size=128, dtype=np.uint8)
    actions = rng.integers(0, 8, size=120, dtype=np.uint8)
    (_GOLDEN_DIR / "pr92_rmc_joint_stream_v1_seg.bin").write_bytes(
        np.asarray(seg_bytes, dtype=np.uint8).tobytes()
    )
    (_GOLDEN_DIR / "pr92_rmc_joint_stream_v1_actions.bin").write_bytes(
        np.asarray(actions, dtype=np.uint8).tobytes()
    )


def write_ranked_no_op_sidecar_inputs() -> None:
    """int64 dims + int64 delta_indices for PR101 ranked-no-op sidecar.

    Matches `test_ranked_sidecar_golden_vector` in
    ``src/tac/tests/test_packet_compiler_pr101_sidecar_grammar.py``.

    Recipe: schema(n_pairs=24, n_dims=8, deltas=PR101_DELTAS, min=2, max=8);
    dims initialised to no_op_sentinel (255); for i in [0, 3, 6, .., 21]:
        dims[i] = (2 + i // 3) % 8
        delta_indices[i] = (5 + (i // 3) * 3) % 16
    """
    schema_n_pairs = 24
    schema_n_dims = 8
    schema_n_deltas = 16
    no_op_sentinel = 255
    dims = np.full(schema_n_pairs, no_op_sentinel, dtype=np.int64)
    delta_idx = np.zeros(schema_n_pairs, dtype=np.int64)
    for i in range(0, schema_n_pairs, 3):
        dims[i] = (2 + i // 3) % schema_n_dims
        delta_idx[i] = (5 + (i // 3) * 3) % schema_n_deltas
    (_GOLDEN_DIR / "ranked_no_op_sidecar_v1_dims.bin").write_bytes(
        np.asarray(dims, dtype="<i8").tobytes()
    )
    (_GOLDEN_DIR / "ranked_no_op_sidecar_v1_delta_indices.bin").write_bytes(
        np.asarray(delta_idx, dtype="<i8").tobytes()
    )


def write_sparse_rle_of_zeros_inputs() -> None:
    """int8 dense input for the sparse RLE-of-zeros primitive.

    Matches `test_golden_vector_sparse_rle` in
    `src/tac/tests/test_packet_compiler_sparse_packet_ir.py`:

    ::

        rng = np.random.default_rng(20260511)
        dense = np.zeros(1024, dtype=np.int8)
        positions = rng.choice(1024, size=64, replace=False)
        values = rng.integers(1, 32, size=64, dtype=np.int8)
        dense[positions] = values
    """
    rng = np.random.default_rng(20260511)
    dense = np.zeros(1024, dtype=np.int8)
    positions = rng.choice(1024, size=64, replace=False)
    values = rng.integers(1, 32, size=64, dtype=np.int8)
    dense[positions] = values
    (_GOLDEN_DIR / "sparse_rle_of_zeros_v1_dense.bin").write_bytes(dense.tobytes())


def write_sparse_arithmetic_coefficients_inputs() -> None:
    """int32 values for the sparse arithmetic coefficients primitive.

    Matches `test_golden_vector_sparse_arithmetic`:

    ::

        rng = np.random.default_rng(20260511)
        raw = rng.integers(-8, 9, size=500, dtype=np.int32)
    """
    rng = np.random.default_rng(20260511)
    raw = rng.integers(-8, 9, size=500, dtype=np.int32)
    (_GOLDEN_DIR / "sparse_arithmetic_coefficients_v1_values.bin").write_bytes(
        raw.astype("<i4").tobytes()
    )


def write_sparse_temporal_subsampled_inputs() -> None:
    """K-of-N int8 per-frame residuals for the sparse temporal-subsampled primitive.

    Matches `test_golden_vector_sparse_temporal`:

    ::

        rng = np.random.default_rng(20260511)
        N = 50; per_frame = 20
        frames: list[np.ndarray | None] = []
        for i in range(N):
            if i % 5 == 0:
                frames.append(rng.integers(-10, 11, size=per_frame, dtype=np.int8))
            else:
                frames.append(None)

    On disk we store ONLY the K signal-carrying frames concatenated as a flat
    int8 stream. The Rust port reconstructs the indicator vector from
    ``i % 5 == 0`` over ``i in [0, N)``. K = ceil(N/5) = 10 frames at 20
    bytes each = 200 bytes.
    """
    rng = np.random.default_rng(20260511)
    n_frames = 50
    per_frame = 20
    parts: list[bytes] = []
    for i in range(n_frames):
        if i % 5 == 0:
            arr = rng.integers(-10, 11, size=per_frame, dtype=np.int8)
            parts.append(arr.tobytes())
    flat = b"".join(parts)
    assert len(flat) == 10 * 20, f"expected 200 bytes, got {len(flat)}"
    (_GOLDEN_DIR / "sparse_temporal_subsampled_v1_signal_frames.bin").write_bytes(flat)


def main() -> None:
    _GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    write_centered_delta_uint8_inputs()
    write_split_brotli_streams_input()
    write_latent_hi_inputs()
    write_merged_range_stream_inputs()
    write_pr93_delta_varint_pose_inputs()
    write_pr91_arithmetic_coder_inputs()
    write_pr84_adaptive_mask_inputs()
    write_pr81_fp4_codebook_inputs()
    write_pr81_router_action_inputs()
    write_pr92_rmc_joint_stream_inputs()
    write_ranked_no_op_sidecar_inputs()
    write_sparse_rle_of_zeros_inputs()
    write_sparse_arithmetic_coefficients_inputs()
    write_sparse_temporal_subsampled_inputs()
    written = sorted(p.name for p in _GOLDEN_DIR.glob("*_v1_*.bin"))
    print(f"Wrote {len(written)} fixture(s) to {_GOLDEN_DIR.relative_to(_GOLDEN_DIR.parent.parent.parent.parent)}:")
    for name in written:
        path = _GOLDEN_DIR / name
        print(f"  {name}: {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
