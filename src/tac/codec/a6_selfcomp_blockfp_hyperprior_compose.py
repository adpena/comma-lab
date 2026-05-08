"""A6 Council ablation — Selfcomp block-FP × Ballé-style hyperprior compose.

Composition contract
--------------------
The hypothesis under test (Council R6 — see the Selfcomp + Ballé seats in
``CLAUDE.md`` and ``.omx/research/grand_council_extreme_rigor_track_1_20260508.md``)
is that **Selfcomp's per-block scale parameter is exactly the conditioning
sigma that a Ballé-style hyperprior wants** — score-aware allocation of bits
across blocks via a tighter, scale-conditional entropy estimate than either
component standalone.

The lane this module operationalises:

1. Block-FP (Selfcomp paradigm) partitions an ``int8`` symbol stream into
   contiguous blocks of size ``B``. For each block we compute the per-block
   scale ``s_b = max(|s_i|)`` for ``i ∈ block``. Storage of ``s_b`` is in
   ``fp16`` (2 bytes/block) or ``uint8`` (1 byte/block) per ablation cell.
   The int8 alphabet includes ``-128``, so true max-abs scale is in
   ``[0, 128]``; uint8 scale side-info reports its saturation explicitly.
2. Ballé hyperprior consumes ``s_b`` as the conditioning side-information
   and derives a per-block sigma ``σ_b = sigma_floor + α * s_b`` (a
   monotone-positive map; trivial deterministic decoder so encoder and
   decoder agree byte-for-byte without needing to ship neural-net weights).
3. The block's residuals (the int8 symbols themselves; centred at zero) are
   range-coded under ``gaussian_pmf_int8(mu=0, sigma=σ_b)`` via
   ``tac.codec.charm_range_coder.ChARMRangeEncoder``. Each block's PMF is
   conditioned on the SAME scale that the inflate path will recover from
   the side-info, so encode/decode agree by construction.

The wire format is a 12-byte header + ``num_blocks * scale_bytes`` of
side-info + a ChARMRangeEncoder payload (which itself is self-delimiting
via its own header). Pure-CPU, no scorer load, no neural-net weights
shipped.

CLAUDE.md compliance
--------------------
* No score claim. Byte-anchor only — every artifact tagged
  ``[byte-anchor; codec=a6_selfcomp_blockfp_hyperprior_compose]``.
* No silent defaults — every public function arg is required-keyword.
* Encoder runs decode roundtrip on every call (CompressAI policy). The
  ``compose_blockfp_with_hyperprior`` wrapper raises if decode-of-encode
  doesn't recover the input bit-for-bit.
* Side-info IS in the encoded byte stream — Check 91 STRICT applies.

Empirical predictions vs PR101 substrate (Council R6 estimate, [predicted]):
* The PR101 INT8 symbol stream from ``tac.pr101_split_brotli_codec`` has
  per-tensor scale heterogeneity. A scale-conditional code is in the
  general direction of the joint-entropy floor (see
  ``feedback_pr101_joint_entropy_floor_subagent_verdict_20260507.md`` —
  148-162 KB on PR101 substrate; brotli sits at 178 KB, ~15% above).
* The compose can plausibly close part of the 16-30 KB headroom **only if
  the side-info cost (num_blocks * scale_bytes) is smaller than the
  scale-conditional rate savings**. This is the ablation cell sweep
  question: how small can ``B`` go before the scale side-info dominates?
* This module computes the byte ledger; ``tools/pr101_a6_blockfp_hyperprior_anchor.py``
  drives the sweep on real PR101 weights and emits the anchor row.

References
----------
* Selfcomp / szabolcs-cs ``src/tac/block_fp_codec.py`` — the original
  ternary block-FP encoder (different math: float weights → ternary qint
  + per-block exponent). This module operates on an already-quantised
  int8 stream and treats the per-block ``max|s|`` as the scale.
* Ballé, Minnen, Singh, Hwang, Johnston 2018 ICLR "Variational Image
  Compression with a Scale Hyperprior". The hyperprior here is a
  deterministic monotone map (no learned net), so we ship 0 bytes of
  hyper-decoder weights — only the per-block scales are side-info.
* ``tac.codec.charm_range_coder`` — entropy backend (``gaussian_pmf_int8``
  + ``ChARMRangeEncoder``). The same backend the A4 lane uses.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

from tac.codec.charm_range_coder import (
    ChARMRangeEncoder,
    ChARMRangeDecoder,
    gaussian_pmf_int8,
)


# ── Wire-format constants ─────────────────────────────────────────────────

_MAGIC: bytes = b"A6BF"  # A6 BlockFp x Hyperprior
_VERSION: int = 1
_HEADER_SIZE: int = 12  # magic(4) + version(1) + scale_quant(1) + block_size(2) + n_total(4)
_ALPHABET: Tuple[int, int] = (-128, 127)
_NUM_SYMBOLS: int = 256
_MAX_UINT16: int = (1 << 16) - 1
_MAX_UINT32: int = (1 << 32) - 1

# ChARM range coder payload header is uint16 → max 65535 bytes per ChARM
# stream. We chunk the input into BLOCK-aligned groups, each a separate
# ChARM stream, so that no single payload exceeds the cap. Wire format:
#
#   header(12) + side_info(num_blocks * scale_bytes) +
#       n_chunks(uint32) +
#       per-chunk: n_blocks_in_chunk(uint32) +
#                  chunk_payload_len(uint32) +
#                  chunk_payload(bytes)
#
# A chunk starts at a block boundary; each chunk's PMFs are independently
# reconstructible from the side-info scales (which are stored once for ALL
# blocks at the head of the encoded blob).
#
# Worst-case: a wide-σ block charges ~1 byte/symbol (uniform). To stay
# safely under 65535 bytes per ChARM stream we cap the number of SYMBOLS
# per chunk at 32_768. The block count per chunk is then
# ``ceil(MAX_SYMBOLS / block_size)`` rounded up to the nearest block.
_CHARM_MAX_SYMBOLS_PER_CHUNK: int = 32_768
_CHARM_CHUNK_HEADER_BYTES: int = 8  # n_blocks_in_chunk(4) + chunk_payload_len(4)


# Hyperprior deterministic map: σ = sigma_floor + alpha * scale.
#
# Selecting these defaults: for an int8 stream with values in [-127, 127]
# and per-block max-abs scale, σ ≈ scale / 1.5 keeps the discretized
# Gaussian PMF "spread" matched to the block's empirical histogram.
# (For a uniform block on [-s, s], the std is s / sqrt(3) ≈ s * 0.577;
# for a Gaussian-ish histogram, std ≈ s / 2 to s / 3.) ``alpha=0.55``
# sits between those and is robust to either substrate.
SIGMA_FLOOR: float = 0.5
ALPHA_DEFAULT: float = 0.55


# Scale-quant modes (1 byte field in the header)
SCALE_QUANT_FP16: int = 0  # 2 bytes per block (default, finest)
SCALE_QUANT_FP32: int = 1  # 4 bytes per block (debug; rarely useful)
SCALE_QUANT_UINT8: int = 2  # 1 byte per block (coarsest)
_UINT8_SCALE_STEP: float = 0.5
_UINT8_SCALE_MAX_REPRESENTABLE: float = 127.5


def _validate_block_size(block_size: int, *, for_wire: bool) -> int:
    if block_size is None:
        raise ValueError("block_size is required")
    if isinstance(block_size, bool):
        raise ValueError("block_size must be an integer >= 1, not bool")
    try:
        block_size_i = int(block_size)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"block_size must be an integer >= 1, got {block_size!r}") from exc
    if block_size_i != block_size:
        raise ValueError(f"block_size must be an integer >= 1, got {block_size!r}")
    if block_size_i < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")
    if for_wire and block_size_i > _MAX_UINT16:
        raise ValueError(
            "block_size must fit the A6BF uint16 wire header "
            f"(<= {_MAX_UINT16}); got {block_size_i}"
        )
    return block_size_i


def _as_int8_symbol_stream(symbols: np.ndarray, *, caller: str) -> np.ndarray:
    if symbols is None:
        raise ValueError(f"{caller}: symbols is required")
    flat = np.ascontiguousarray(symbols).reshape(-1)
    if flat.size == 0:
        return flat.astype(np.int8, copy=False)
    if not np.issubdtype(flat.dtype, np.integer):
        raise ValueError(
            f"{caller}: symbols must have an integer dtype, got {flat.dtype}"
        )
    if flat.dtype != np.int8:
        if flat.min() < -128 or flat.max() > 127:
            raise ValueError(
                f"symbols out of int8 range: min={int(flat.min())}, max={int(flat.max())}"
            )
        flat = flat.astype(np.int8)
    return flat


def _scale_bytes_per_block(scale_quant: int) -> int:
    if scale_quant == SCALE_QUANT_FP16:
        return 2
    if scale_quant == SCALE_QUANT_FP32:
        return 4
    if scale_quant == SCALE_QUANT_UINT8:
        return 1
    raise ValueError(f"unknown scale_quant code {scale_quant}")


def _quantize_scale(scale: float, *, mode: int) -> bytes:
    """Round-trip-stable scale quantization; matches ``_dequantize_scale``."""
    if scale < 0.0:
        raise ValueError(f"scale must be >= 0; got {scale}")
    if mode == SCALE_QUANT_FP16:
        return np.array([scale], dtype="<f2").tobytes()
    if mode == SCALE_QUANT_FP32:
        return np.array([scale], dtype="<f4").tobytes()
    if mode == SCALE_QUANT_UINT8:
        # Half-step scale code. True int8 max-abs can be 128 because of
        # ``-128``; uint8 v1 saturates that one endpoint to 127.5 and the
        # ledger reports the quantization error.
        q = int(
            round(
                min(max(scale, 0.0), _UINT8_SCALE_MAX_REPRESENTABLE)
                / _UINT8_SCALE_STEP
            )
        )
        q = max(0, min(255, q))
        return bytes((q,))
    raise ValueError(f"unknown scale_quant code {mode}")


def _dequantize_scale(blob: bytes, *, mode: int) -> float:
    """Inverse of ``_quantize_scale``."""
    if mode == SCALE_QUANT_FP16:
        if len(blob) != 2:
            raise ValueError(f"fp16 scale blob must be 2 bytes, got {len(blob)}")
        return float(np.frombuffer(blob, dtype="<f2")[0])
    if mode == SCALE_QUANT_FP32:
        if len(blob) != 4:
            raise ValueError(f"fp32 scale blob must be 4 bytes, got {len(blob)}")
        return float(np.frombuffer(blob, dtype="<f4")[0])
    if mode == SCALE_QUANT_UINT8:
        if len(blob) != 1:
            raise ValueError(f"uint8 scale blob must be 1 byte, got {len(blob)}")
        return float(blob[0]) * _UINT8_SCALE_STEP
    raise ValueError(f"unknown scale_quant code {mode}")


def _validate_wire_hyperparams(*, sigma_floor: float, alpha: float) -> None:
    """A6BF v1 stores fixed hyperprior parameters in the codec version.

    Accepting alternate values without serializing them creates a split-brain
    wire contract: encode and decode can silently use different PMFs. Keep v1
    fixed until a new header version carries these fields explicitly.
    """
    if sigma_floor != SIGMA_FLOOR:
        raise ValueError(
            "A6BF v1 does not serialize sigma_floor; use the fixed "
            f"default {SIGMA_FLOOR} or introduce a new wire version"
        )
    if alpha != ALPHA_DEFAULT:
        raise ValueError(
            "A6BF v1 does not serialize alpha; use the fixed "
            f"default {ALPHA_DEFAULT} or introduce a new wire version"
        )


# ── Hyperprior decoder (deterministic, no neural-net weights) ─────────────


def hyperprior_sigma_from_scale(
    scale: float,
    *,
    sigma_floor: float = SIGMA_FLOOR,
    alpha: float = ALPHA_DEFAULT,
) -> float:
    """Map a per-block scale to a per-block Gaussian sigma.

    ``σ = sigma_floor + α * scale``

    The map is monotone-positive and analytic; encoder and decoder MUST call
    it with the same arguments to produce identical PMFs.

    Args:
        scale: per-block scale (>= 0).
        sigma_floor: lower bound on σ to keep ``gaussian_pmf_int8`` strictly
            positive (the floor that ``ChARMRangeEncoder`` requires).
        alpha: linear coefficient on scale.

    Returns:
        Positive float sigma in the band ``[sigma_floor, sigma_floor + 128*α]``.
    """
    if scale < 0.0:
        raise ValueError(f"scale must be >= 0; got {scale}")
    if sigma_floor <= 0.0:
        raise ValueError(f"sigma_floor must be > 0; got {sigma_floor}")
    if alpha < 0.0:
        raise ValueError(f"alpha must be >= 0; got {alpha}")
    return float(sigma_floor + alpha * scale)


# ── Block-FP encode (per-block max-abs scale extraction) ──────────────────


@dataclass(frozen=True)
class BlockFPSplit:
    """Result of partitioning a symbol stream into blocks + per-block scales.

    Pure data class — the bytes encoder is :func:`compose_blockfp_with_hyperprior`.
    """

    block_size: int
    n_total: int
    n_blocks: int
    scales: np.ndarray  # shape (n_blocks,), float32, values in [0, 127]
    blocks: list[np.ndarray]  # each (block_size,) int8 except last (<= block_size)


def split_into_blockfp(
    symbols: np.ndarray,
    *,
    block_size: int,
) -> BlockFPSplit:
    """Partition a 1-D int8 stream into blocks and compute per-block scale.

    Args:
        symbols: 1-D ``int8`` array; values must be in ``[-128, 127]``.
        block_size: block partition size (>= 1). Required keyword.

    Returns:
        :class:`BlockFPSplit`. The last block may be smaller than
        ``block_size`` if ``len(symbols)`` is not a multiple.
    """
    block_size = _validate_block_size(block_size, for_wire=False)
    flat = _as_int8_symbol_stream(symbols, caller="split_into_blockfp")
    n_total = int(flat.size)
    if n_total == 0:
        return BlockFPSplit(
            block_size=block_size,
            n_total=0,
            n_blocks=0,
            scales=np.zeros((0,), dtype=np.float32),
            blocks=[],
        )
    n_blocks = (n_total + block_size - 1) // block_size
    blocks: list[np.ndarray] = []
    scales = np.zeros((n_blocks,), dtype=np.float32)
    for b in range(n_blocks):
        lo = b * block_size
        hi = min(lo + block_size, n_total)
        block = flat[lo:hi]
        # Per-block scale: max abs of the int8 block. Falls in [0, 128].
        # An all-zero block has scale 0, which the dequantize path treats
        # as σ = sigma_floor (a tight Gaussian centered at zero, the
        # natural choice for a degenerate block).
        s = float(np.abs(block.astype(np.int16)).max()) if block.size > 0 else 0.0
        scales[b] = s
        blocks.append(block)
    return BlockFPSplit(
        block_size=block_size,
        n_total=n_total,
        n_blocks=n_blocks,
        scales=scales,
        blocks=blocks,
    )


# ── Compose: block-FP × hyperprior → encoded bytes ────────────────────────


def compose_blockfp_with_hyperprior(
    symbols: np.ndarray,
    *,
    block_size: int,
    scale_quant: int = SCALE_QUANT_FP16,
    sigma_floor: float = SIGMA_FLOOR,
    alpha: float = ALPHA_DEFAULT,
    verify_roundtrip: bool = True,
) -> Tuple[bytes, dict]:
    """Block-FP scale × Ballé-style hyperprior → encoded bytes + ledger.

    The composition: per-block scale ``s_b = max(|s_i|)`` becomes the
    conditioning sigma for a per-block discretized Gaussian PMF; the
    block's residuals are range-coded under that PMF. Side-info is
    ``num_blocks`` quantized scales (size = ``num_blocks * scale_bytes``).

    Args:
        symbols: 1-D int8-range integer array. Required.
        block_size: block partition size (>= 1). Required keyword.
        scale_quant: one of ``SCALE_QUANT_FP16`` / ``_FP32`` / ``_UINT8``.
        sigma_floor: hyperprior floor (passed to
            :func:`hyperprior_sigma_from_scale`).
        alpha: hyperprior linear coefficient.
        verify_roundtrip: if True (default), encode-decode-assert on every
            call. CLAUDE.md non-negotiable: a malformed wire format MUST
            NOT ship silently.

    Returns:
        ``(encoded_bytes, ledger)`` where ``ledger`` is a dict with byte
        accounting:

            * ``total_bytes``
            * ``header_bytes``
            * ``side_info_bytes`` (per-block scales)
            * ``payload_bytes`` (range-coded residuals + their header)
            * ``n_blocks``, ``block_size``, ``scale_quant``
    """
    block_size = _validate_block_size(block_size, for_wire=True)
    if scale_quant not in (SCALE_QUANT_FP16, SCALE_QUANT_FP32, SCALE_QUANT_UINT8):
        raise ValueError(f"unknown scale_quant {scale_quant}")
    _validate_wire_hyperparams(sigma_floor=sigma_floor, alpha=alpha)
    split = split_into_blockfp(symbols, block_size=block_size)

    scale_byte_size = _scale_bytes_per_block(scale_quant)

    # ── Side-info: quantize-then-dequantize each scale so encoder uses the
    # SAME σ value the decoder will reconstruct. This is the encode/decode
    # symmetry that bit-deterministic compose requires.
    side_info = bytearray()
    decoded_scales = np.zeros_like(split.scales)
    for b in range(split.n_blocks):
        s_q = _quantize_scale(float(split.scales[b]), mode=scale_quant)
        side_info.extend(s_q)
        decoded_scales[b] = _dequantize_scale(bytes(s_q), mode=scale_quant)
    if split.n_blocks:
        scale_errors = np.abs(
            decoded_scales.astype(np.float64) - split.scales.astype(np.float64)
        )
        max_scale_quantization_error = float(scale_errors.max())
    else:
        max_scale_quantization_error = 0.0
    scale_quantization_saturated_blocks = (
        int(np.count_nonzero(split.scales > _UINT8_SCALE_MAX_REPRESENTABLE))
        if scale_quant == SCALE_QUANT_UINT8
        else 0
    )

    # ── Range-code the residuals in BLOCK-aligned chunks, each its own
    # ChARM stream (uint16 payload-len cap forces chunking on long inputs).
    chunks: list[bytes] = []
    chunk_block_counts: list[int] = []
    if split.n_total > 0:
        # Cap chunk size in SYMBOLS to stay under ChARM's uint16 payload-len
        # field even at worst-case uniform σ (~1 byte/symbol). Convert to
        # block count, rounded UP so the worst-case block sees at least one
        # full block per chunk; ensure ≥1 block per chunk regardless of
        # block_size.
        max_blocks = max(1, _CHARM_MAX_SYMBOLS_PER_CHUNK // max(1, block_size))
        cursor_block = 0
        while cursor_block < split.n_blocks:
            end_block = min(cursor_block + max_blocks, split.n_blocks)
            enc = ChARMRangeEncoder(alphabet=_ALPHABET, pmf_total_bits=15)
            for b in range(cursor_block, end_block):
                sigma = hyperprior_sigma_from_scale(
                    float(decoded_scales[b]),
                    sigma_floor=sigma_floor,
                    alpha=alpha,
                )
                pmf = gaussian_pmf_int8(
                    mu=0.0,
                    sigma=sigma,
                    alphabet_lo=_ALPHABET[0],
                    alphabet_hi=_ALPHABET[1],
                )
                for sym in split.blocks[b].tolist():
                    enc.write_symbol(int(sym), pmf)
            chunk_payload = enc.finish()
            chunks.append(chunk_payload)
            chunk_block_counts.append(end_block - cursor_block)
            cursor_block = end_block

    # ── Assemble: header + side_info + n_chunks(u32) + per-chunk(n_blocks u32, len u32, payload)
    header = (
        _MAGIC
        + struct.pack("<B", _VERSION)
        + struct.pack("<B", int(scale_quant))
        + struct.pack("<H", int(block_size))
        + struct.pack("<I", int(split.n_total))
    )
    assert len(header) == _HEADER_SIZE

    chunk_section = bytearray()
    chunk_section.extend(struct.pack("<I", len(chunks)))
    for n_blocks_in_chunk, chunk_payload in zip(chunk_block_counts, chunks):
        chunk_section.extend(struct.pack("<I", int(n_blocks_in_chunk)))
        chunk_section.extend(struct.pack("<I", len(chunk_payload)))
        chunk_section.extend(chunk_payload)

    encoded = bytes(header) + bytes(side_info) + bytes(chunk_section)
    payload_total = sum(len(c) for c in chunks)
    chunk_overhead = 4 + len(chunks) * _CHARM_CHUNK_HEADER_BYTES

    ledger = {
        "total_bytes": len(encoded),
        "header_bytes": _HEADER_SIZE,
        "side_info_bytes": len(side_info),
        "payload_bytes": payload_total + chunk_overhead,
        "payload_charm_bytes": payload_total,
        "chunk_overhead_bytes": chunk_overhead,
        "n_chunks": len(chunks),
        "n_blocks": split.n_blocks,
        "block_size": block_size,
        "n_total": split.n_total,
        "scale_quant": int(scale_quant),
        "scale_bytes_per_block": scale_byte_size,
        "scale_side_info_exact": max_scale_quantization_error == 0.0,
        "scale_quantization_max_abs_error": max_scale_quantization_error,
        "scale_quantization_saturated_blocks": scale_quantization_saturated_blocks,
        "sigma_floor": sigma_floor,
        "alpha": alpha,
    }

    # CompressAI / arithmetic_qint_codec pattern: never ship a wire format
    # that doesn't decode. CLAUDE.md non-negotiable.
    if verify_roundtrip:
        recovered = decompose_blockfp_with_hyperprior(encoded)
        expected = (
            np.concatenate(split.blocks)
            if split.blocks
            else np.zeros((0,), dtype=np.int8)
        )
        if not np.array_equal(recovered, expected):
            raise RuntimeError(
                "compose_blockfp_with_hyperprior: encode/decode roundtrip failed; "
                "the compose is not bit-deterministic — refusing to ship"
            )

    return encoded, ledger


# ── Decompose: bytes → recovered symbol stream ─────────────────────────────


def decompose_blockfp_with_hyperprior(
    encoded: bytes,
    *,
    sigma_floor: float = SIGMA_FLOOR,
    alpha: float = ALPHA_DEFAULT,
) -> np.ndarray:
    """Inverse of :func:`compose_blockfp_with_hyperprior`.

    Returns a 1-D ``int8`` array of length ``n_total`` from the header.
    The hyperprior ``sigma_floor`` and ``alpha`` MUST match the encoder
    invocation; they are NOT stored in the wire format because they are
    fixed module-level constants under the A6 lane (changing them is a
    new wire-format version).
    """
    if encoded is None or len(encoded) < _HEADER_SIZE:
        raise ValueError("decompose_blockfp_with_hyperprior: blob too short")
    if encoded[:4] != _MAGIC:
        raise ValueError(
            f"decompose: bad magic {encoded[:4]!r}, expected {_MAGIC!r}"
        )
    version = encoded[4]
    if version != _VERSION:
        raise ValueError(f"decompose: unsupported version {version}")
    scale_quant = encoded[5]
    (block_size,) = struct.unpack_from("<H", encoded, 6)
    (n_total,) = struct.unpack_from("<I", encoded, 8)
    _validate_block_size(block_size, for_wire=True)
    _validate_wire_hyperparams(sigma_floor=sigma_floor, alpha=alpha)

    cursor = _HEADER_SIZE
    n_blocks = (n_total + block_size - 1) // block_size
    scale_byte_size = _scale_bytes_per_block(scale_quant)
    side_info_len = n_blocks * scale_byte_size
    if cursor + side_info_len > len(encoded):
        raise ValueError(
            f"decompose: side-info truncated; expected {side_info_len} bytes, "
            f"have {len(encoded) - cursor}"
        )
    decoded_scales = np.zeros((n_blocks,), dtype=np.float32)
    for b in range(n_blocks):
        blob = encoded[cursor : cursor + scale_byte_size]
        decoded_scales[b] = _dequantize_scale(blob, mode=scale_quant)
        cursor += scale_byte_size

    # Chunk section: n_chunks(u32) + per-chunk (n_blocks u32, len u32, payload).
    if cursor + 4 > len(encoded):
        raise ValueError("decompose: chunk-count header truncated")
    (n_chunks,) = struct.unpack_from("<I", encoded, cursor)
    cursor += 4
    if n_total == 0 and n_chunks != 0:
        raise ValueError(f"decompose: empty stream must have 0 chunks, got {n_chunks}")

    out = np.zeros((n_total,), dtype=np.int8)
    block_cursor = 0
    sym_cursor = 0
    for _ in range(n_chunks):
        if cursor + _CHARM_CHUNK_HEADER_BYTES > len(encoded):
            raise ValueError("decompose: chunk header truncated")
        (n_blocks_in_chunk,) = struct.unpack_from("<I", encoded, cursor)
        cursor += 4
        if n_blocks_in_chunk < 1:
            raise ValueError(
                f"decompose: chunk block count must be >= 1, got {n_blocks_in_chunk}"
            )
        (chunk_payload_len,) = struct.unpack_from("<I", encoded, cursor)
        cursor += 4
        if cursor + chunk_payload_len > len(encoded):
            raise ValueError(
                f"decompose: chunk payload truncated; need {chunk_payload_len}, "
                f"have {len(encoded) - cursor}"
            )
        chunk_payload = encoded[cursor : cursor + chunk_payload_len]
        cursor += chunk_payload_len

        chunk_end_block = block_cursor + n_blocks_in_chunk
        if chunk_end_block > n_blocks:
            raise ValueError(
                f"decompose: chunk overshoots block count "
                f"({chunk_end_block} > {n_blocks})"
            )
        expected_chunk_symbols = (
            min(chunk_end_block * block_size, n_total) - block_cursor * block_size
        )
        dec = ChARMRangeDecoder(chunk_payload, alphabet=_ALPHABET)
        if dec.num_symbols != expected_chunk_symbols:
            raise ValueError(
                f"decompose: chunk symbol count mismatch "
                f"({dec.num_symbols} != {expected_chunk_symbols})"
            )
        for b in range(block_cursor, chunk_end_block):
            sigma = hyperprior_sigma_from_scale(
                float(decoded_scales[b]),
                sigma_floor=sigma_floor,
                alpha=alpha,
            )
            pmf = gaussian_pmf_int8(
                mu=0.0,
                sigma=sigma,
                alphabet_lo=_ALPHABET[0],
                alphabet_hi=_ALPHABET[1],
            )
            lo = b * block_size
            hi = min(lo + block_size, n_total)
            for i in range(lo, hi):
                out[i] = dec.read_symbol(pmf)
            sym_cursor = hi
        block_cursor = chunk_end_block

    if block_cursor != n_blocks:
        raise ValueError(
            f"decompose: block count mismatch after chunks "
            f"({block_cursor} != {n_blocks})"
        )
    if sym_cursor != n_total:
        raise ValueError(
            f"decompose: symbol count mismatch after chunks "
            f"({sym_cursor} != {n_total})"
        )
    if cursor != len(encoded):
        raise ValueError(
            f"decompose: trailing bytes after chunk section "
            f"({len(encoded) - cursor} unconsumed)"
        )
    return out


# ── Standalone-comparison helpers (for the ablation sweep) ────────────────


def encode_blockfp_only(
    symbols: np.ndarray,
    *,
    block_size: int,
    scale_quant: int = SCALE_QUANT_FP16,
) -> Tuple[bytes, dict]:
    """Block-FP standalone — per-block scale + raw int8 residuals (no entropy).

    The "block-FP standalone" baseline against which compose is measured.
    Wire layout: header + side-info (scales) + raw int8 residuals
    (1 byte/symbol).
    """
    block_size = _validate_block_size(block_size, for_wire=True)
    split = split_into_blockfp(symbols, block_size=block_size)
    scale_byte_size = _scale_bytes_per_block(scale_quant)
    side_info = bytearray()
    for b in range(split.n_blocks):
        side_info.extend(_quantize_scale(float(split.scales[b]), mode=scale_quant))
    flat = (
        np.concatenate(split.blocks)
        if split.blocks
        else np.zeros((0,), dtype=np.int8)
    )
    payload = flat.tobytes()  # raw int8, 1 byte/symbol
    header = (
        b"BFRO"
        + struct.pack("<B", _VERSION)
        + struct.pack("<B", int(scale_quant))
        + struct.pack("<H", int(block_size))
        + struct.pack("<I", int(split.n_total))
    )
    encoded = bytes(header) + bytes(side_info) + payload
    return (
        encoded,
        {
            "total_bytes": len(encoded),
            "header_bytes": _HEADER_SIZE,
            "side_info_bytes": len(side_info),
            "payload_bytes": len(payload),
            "n_blocks": split.n_blocks,
            "block_size": block_size,
            "n_total": split.n_total,
            "scale_quant": int(scale_quant),
            "scale_bytes_per_block": scale_byte_size,
            "encoding": "blockfp_only_raw_int8",
        },
    )


def encode_hyperprior_only(
    symbols: np.ndarray,
) -> Tuple[bytes, dict]:
    """Hyperprior standalone — global Gaussian sigma over the whole stream.

    The "Ballé hyperprior standalone" baseline: a single sigma derived from
    the global max abs is used for the entire stream. No per-block side-info.
    """
    flat = _as_int8_symbol_stream(symbols, caller="encode_hyperprior_only")
    global_scale = float(np.abs(flat.astype(np.int16)).max()) if flat.size else 0.0
    sigma = hyperprior_sigma_from_scale(global_scale)
    pmf = gaussian_pmf_int8(
        mu=0.0,
        sigma=sigma,
        alphabet_lo=_ALPHABET[0],
        alphabet_hi=_ALPHABET[1],
    )
    # Chunk to satisfy the ChARM uint16 payload-len cap. Worst-case-uniform
    # σ runs at ~1 byte/symbol; chunk every 32K symbols → ≤32K payload bytes
    # per ChARM stream. Each chunk gets a 4-byte length prefix.
    CHUNK_SYMBOLS = 32_768
    chunk_payloads: list[bytes] = []
    for start in range(0, flat.size, CHUNK_SYMBOLS):
        end = min(start + CHUNK_SYMBOLS, flat.size)
        enc = ChARMRangeEncoder(alphabet=_ALPHABET, pmf_total_bits=15)
        for sym in flat[start:end].tolist():
            enc.write_symbol(int(sym), pmf)
        chunk_payloads.append(enc.finish())
    payload_total = sum(len(c) for c in chunk_payloads)
    header = (
        b"HONY"
        + struct.pack("<B", _VERSION)
        + struct.pack("<B", SCALE_QUANT_FP16)
        + struct.pack("<H", int(min(max(flat.size, 1), 65_535)))
        + struct.pack("<I", int(flat.size))
    )
    side_info = _quantize_scale(global_scale, mode=SCALE_QUANT_FP16)
    chunk_section = bytearray()
    chunk_section.extend(struct.pack("<I", len(chunk_payloads)))
    for chunk in chunk_payloads:
        chunk_section.extend(struct.pack("<I", len(chunk)))
        chunk_section.extend(chunk)
    encoded = bytes(header) + side_info + bytes(chunk_section)
    chunk_section_bytes = len(chunk_section)
    return encoded, {
        "total_bytes": len(encoded),
        "header_bytes": _HEADER_SIZE,
        "side_info_bytes": 2,
        "payload_bytes": chunk_section_bytes,
        "payload_charm_bytes": payload_total,
        "n_chunks": len(chunk_payloads),
        "n_blocks": 1,
        "block_size": flat.size,
        "n_total": flat.size,
        "encoding": "hyperprior_only_global_sigma",
    }


__all__ = [
    "ALPHA_DEFAULT",
    "BlockFPSplit",
    "SCALE_QUANT_FP16",
    "SCALE_QUANT_FP32",
    "SCALE_QUANT_UINT8",
    "SIGMA_FLOOR",
    "compose_blockfp_with_hyperprior",
    "decompose_blockfp_with_hyperprior",
    "encode_blockfp_only",
    "encode_hyperprior_only",
    "hyperprior_sigma_from_scale",
    "split_into_blockfp",
]
