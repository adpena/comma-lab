# SPDX-License-Identifier: MIT
"""Carmack-Hotz STRIP-EVERYTHING codec — closed-form bit allocator + arithmetic coder.

NO neural primitives. NO PyTorch dependency outside the compress-time scorer
query (compress-side is allowed anything per contest README rule 2). At inflate
time only the OUTPUT of these helpers (bytes) lives in the archive; the
decoder is pure numpy arithmetic.

Three primitive classes:

* :class:`GrayscalePalette` — Quantizr PR #56 paradigm: quantize the per-frame
  grayscale field to a small palette (default 16 levels). The palette indices
  encode dramatically more efficiently than fp16 / fp32 grayscale.

* :class:`ClassConditionalCDF` — per-class cumulative distribution over palette
  indices, derived from the SegNet argmax at compress time. The decoder uses
  the SAME palette + per-class CDF (both stored in the archive) for arithmetic
  decoding. NO scorer load required at inflate.

* :class:`ArithmeticCoder` — minimal hand-rolled binary arithmetic coder with
  32-bit state. Encode emits a byte stream; decode consumes the same.

And one closed-form allocator:

* :func:`allocate_bits_closed_form` — given the per-pixel SegNet importance
  (the inverse-steganalysis lens: bits spent where the scorer cares), return a
  per-cell bit budget. NO training; the scorer's class-conditional entropy IS
  the allocation.

Why "closed-form"? The Carmack-Hotz radical premise: a SegNet-trained scorer
already encodes class-importance via its softmax distribution. At compress
time, we can compute the per-pixel ``H(class | pixel)`` and allocate bits in
proportion. This is the inverse-steganalysis lens (Fridrich UNIWARD: weight by
inverse local variance; here we weight by scorer-implied importance).

References:
- Quantizr PR #56 grayscale-LUT analog mask paradigm (Selfcomp; PR101 0.193)
- Witten et al. 1987 (arithmetic coding canonical reference)
- Fridrich 2009 UNIWARD (inverse-steganalysis cost weighting)
- CLAUDE.md "Fridrich inverse steganalysis - how to beat the scorer"

CLAUDE.md compliance:
- Deterministic (sorted CDF, fixed-precision arithmetic state)
- No /tmp paths
- No scorer load (this module operates on scorer OUTPUTS, not loads the model)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Constants — Carmack iteration: pick the smallest knobs that could possibly work
# ---------------------------------------------------------------------------

DEFAULT_PALETTE_SIZE: int = 16
"""Quantizr PR #56 paradigm: 16 grayscale levels (4 bits/cell) is the sweet
spot empirically for natural-video luminance. 8 levels = 3 bits but ringing
artifacts; 32 levels = 5 bits without measurable quality gain."""

NUM_SEGNET_CLASSES: int = 5
"""SegNet outputs 5-class logits per CLAUDE.md "Exact scorer architectures".
The class-conditional CDF table has NUM_SEGNET_CLASSES rows."""

CDF_PRECISION_BITS: int = 16
"""Arithmetic coder uses 16-bit CDF precision; cumfreq_max = 2**16 - 1."""

CDF_MAX: int = (1 << CDF_PRECISION_BITS) - 1


# ---------------------------------------------------------------------------
# Grayscale palette (Quantizr PR #56 paradigm)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GrayscalePalette:
    """Quantized grayscale palette: PALETTE_SIZE float levels in [0, 255].

    Built compress-side via k-means-style binning over the GT video's
    luminance histogram. Stored in the archive as PALETTE_SIZE * 1 byte
    (uint8 levels in [0, 255]).
    """

    levels: np.ndarray  # shape (PALETTE_SIZE,) uint8 in [0, 255]

    def __post_init__(self) -> None:
        if self.levels.dtype != np.uint8:
            raise ValueError(f"palette levels must be uint8; got {self.levels.dtype}")
        if self.levels.ndim != 1:
            raise ValueError(f"palette must be 1-D; got shape {self.levels.shape}")
        if len(self.levels) < 2 or len(self.levels) > 256:
            raise ValueError(f"palette size out of [2, 256]: {len(self.levels)}")

    @property
    def size(self) -> int:
        return len(self.levels)

    def quantize(self, gray_u8: np.ndarray) -> np.ndarray:
        """Quantize uint8 grayscale to palette indices.

        Returns palette-index array (uint8 in [0, PALETTE_SIZE)).
        Uses argmin-distance bucketing; the palette is small so the (H, W, P)
        broadcast cost is acceptable at compress time.
        """
        if gray_u8.dtype != np.uint8:
            raise ValueError(f"input must be uint8; got {gray_u8.dtype}")
        # (..., 1) - (P,) -> (..., P)
        dists = np.abs(
            gray_u8.astype(np.int16)[..., None] - self.levels.astype(np.int16)
        )
        return np.argmin(dists, axis=-1).astype(np.uint8)

    def dequantize(self, indices: np.ndarray) -> np.ndarray:
        """Map palette indices -> uint8 grayscale levels."""
        if indices.dtype != np.uint8:
            raise ValueError(f"indices must be uint8; got {indices.dtype}")
        if int(indices.max()) >= self.size:
            raise ValueError(
                f"palette index {int(indices.max())} out of range [0, {self.size})"
            )
        return self.levels[indices]


def build_grayscale_palette(
    gray_u8_samples: np.ndarray, palette_size: int = DEFAULT_PALETTE_SIZE
) -> GrayscalePalette:
    """Build a palette by greedy 1-D k-means-style binning over a histogram.

    Carmack iteration: avoid sklearn KMeans dependency; use the histogram
    cumulative percentile bins, which is the optimal 1-D quantizer for
    uniform-codelength assumption and matches Lloyd-Max for natural images.
    """
    if gray_u8_samples.dtype != np.uint8:
        raise ValueError(f"samples must be uint8; got {gray_u8_samples.dtype}")
    if palette_size < 2 or palette_size > 256:
        raise ValueError(f"palette_size out of [2, 256]: {palette_size}")
    flat = gray_u8_samples.ravel()
    if flat.size == 0:
        raise ValueError("cannot build palette from empty samples")
    # Cumulative-percentile bins (Lloyd-Max approximation for 1-D).
    quantiles = np.linspace(0.0, 1.0, palette_size + 1)[1:-1]
    if quantiles.size == 0:
        levels = np.array([int(flat.mean())], dtype=np.uint8)
    else:
        edges = np.quantile(flat.astype(np.float64), quantiles)
        bin_edges = np.concatenate([[0.0], edges, [255.0]])
        levels = []
        for i in range(palette_size):
            lo = bin_edges[i]
            hi = bin_edges[i + 1]
            mask = (flat >= lo) & (flat <= hi)
            if mask.any():
                levels.append(round(float(flat[mask].mean())))
            else:
                levels.append(round((lo + hi) / 2.0))
        levels = np.array(levels, dtype=np.uint8)
    return GrayscalePalette(levels=levels)


# ---------------------------------------------------------------------------
# Class-conditional CDFs (inverse-steganalysis lens; closed-form allocator)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClassConditionalCDF:
    """Per-(SegNet-class) cumulative distribution over palette indices.

    Shape: ``(NUM_SEGNET_CLASSES, palette_size + 1)`` uint16. The last
    column is always CDF_MAX so the coder has a closed [0, CDF_MAX] interval.

    Stored in the archive as ``NUM_SEGNET_CLASSES * palette_size * 2`` bytes.
    """

    cdf: np.ndarray  # shape (NUM_SEGNET_CLASSES, palette_size + 1) uint16

    def __post_init__(self) -> None:
        if self.cdf.dtype != np.uint16:
            raise ValueError(f"CDF must be uint16; got {self.cdf.dtype}")
        if self.cdf.ndim != 2 or self.cdf.shape[0] != NUM_SEGNET_CLASSES:
            raise ValueError(
                f"CDF must be ({NUM_SEGNET_CLASSES}, P+1); got {self.cdf.shape}"
            )
        if self.cdf.shape[1] < 2:
            raise ValueError("CDF must have at least 2 columns (palette_size >= 1)")
        # Monotone non-decreasing; first 0, last CDF_MAX
        if not np.all(self.cdf[:, 0] == 0):
            raise ValueError("CDF[:, 0] must be 0")
        if not np.all(self.cdf[:, -1] == CDF_MAX):
            raise ValueError(f"CDF[:, -1] must be CDF_MAX={CDF_MAX}")
        diffs = np.diff(self.cdf.astype(np.int32), axis=1)
        if (diffs < 0).any():
            raise ValueError("CDF must be non-decreasing along axis=1")

    @property
    def palette_size(self) -> int:
        return int(self.cdf.shape[1] - 1)


def build_class_conditional_cdf(
    palette_indices: np.ndarray,
    class_labels: np.ndarray,
    palette_size: int,
) -> ClassConditionalCDF:
    """Empirical CDF P(palette_idx | class) from compress-time samples.

    Args:
        palette_indices: uint8 array of palette indices (any shape).
        class_labels: uint8 array of SegNet argmax labels (same shape).
        palette_size: number of palette entries.

    The class-conditional CDF is the heart of the closed-form allocator:
    pixels in a high-importance class get bits allocated by THEIR class's
    PMF, not the global PMF. This is the inverse-steganalysis lens
    (Fridrich UNIWARD adapted to scorer-derived importance).
    """
    if palette_indices.shape != class_labels.shape:
        raise ValueError(
            f"palette_indices.shape={palette_indices.shape} != class_labels.shape={class_labels.shape}"
        )
    if palette_indices.dtype != np.uint8 or class_labels.dtype != np.uint8:
        raise ValueError("palette_indices and class_labels must both be uint8")
    if int(class_labels.max()) >= NUM_SEGNET_CLASSES:
        raise ValueError(
            f"class_labels max {int(class_labels.max())} >= NUM_SEGNET_CLASSES={NUM_SEGNET_CLASSES}"
        )
    pi_flat = palette_indices.ravel()
    cls_flat = class_labels.ravel()
    cdf = np.zeros((NUM_SEGNET_CLASSES, palette_size + 1), dtype=np.int64)
    for c in range(NUM_SEGNET_CLASSES):
        mask = cls_flat == c
        if not mask.any():
            # Uniform PMF for empty class so the arithmetic coder still works.
            counts = np.ones(palette_size, dtype=np.int64)
        else:
            counts = np.bincount(pi_flat[mask], minlength=palette_size)
            counts = counts + 1  # Laplace smoothing (no zero-prob symbol)
        cum = np.cumsum(counts)
        # Scale to CDF_MAX
        cum_scaled = (cum.astype(np.float64) / float(cum[-1]) * CDF_MAX).astype(np.int64)
        cum_scaled[-1] = CDF_MAX
        # Ensure strict monotone (defends against rounding ties)
        for i in range(1, palette_size):
            if cum_scaled[i] <= cum_scaled[i - 1]:
                cum_scaled[i] = cum_scaled[i - 1] + 1
        cum_scaled[-1] = CDF_MAX
        # Ensure cum_scaled[-1] stayed at CDF_MAX after the previous pass
        if cum_scaled[-2] >= CDF_MAX:
            cum_scaled[-2] = CDF_MAX - 1
        cdf[c, 0] = 0
        cdf[c, 1:] = cum_scaled
    return ClassConditionalCDF(cdf=cdf.astype(np.uint16))


def allocate_bits_closed_form(
    class_importance_map: np.ndarray,
    total_byte_budget: int,
) -> np.ndarray:
    """Allocate bytes per spatial cell proportional to scorer-derived importance.

    Carmack-Hotz closed-form: byte_i = budget * (importance_i / sum_j importance_j).

    Args:
        class_importance_map: float32 array of per-cell importance (any shape).
            Typically computed as ``softmax_entropy(SegNet(x))`` — pixels
            where the scorer is uncertain get more bits (high entropy = high
            information content w.r.t. the scoring objective).
        total_byte_budget: total bytes available for this stream.

    Returns:
        int32 array of same shape with per-cell byte allocation (sum
        approximately equals total_byte_budget within rounding error).
    """
    if total_byte_budget < 0:
        raise ValueError(f"total_byte_budget must be >= 0; got {total_byte_budget}")
    if class_importance_map.ndim < 1:
        raise ValueError("class_importance_map must have at least 1 dim")
    weights = class_importance_map.astype(np.float64)
    if (weights < 0).any():
        raise ValueError("class_importance_map must be non-negative")
    total = weights.sum()
    if total <= 0:
        # Uniform allocation if scorer outputs no signal
        n = weights.size
        per_cell = total_byte_budget // n
        out = np.full(weights.shape, per_cell, dtype=np.int32)
    else:
        out = np.round(weights / total * float(total_byte_budget)).astype(np.int32)
        # Rounding can push the sum slightly over/under; distribute the
        # delta to the highest-importance cells.
        delta = int(total_byte_budget - int(out.sum()))
        if delta != 0:
            flat_imp = weights.ravel()
            order = np.argsort(-flat_imp)  # descending
            flat_out = out.ravel()
            step = 1 if delta > 0 else -1
            for k in range(abs(delta)):
                idx = order[k % flat_imp.size]
                if step > 0 or flat_out[idx] > 0:
                    flat_out[idx] += step
            out = flat_out.reshape(weights.shape)
    return out


# ---------------------------------------------------------------------------
# Minimal hand-rolled binary arithmetic coder (32-bit state)
# ---------------------------------------------------------------------------


class ArithmeticCoder:
    """Minimal binary arithmetic coder with 32-bit state and 16-bit CDFs.

    Encode/decode round-trip is byte-stable. Hotz-style minimal: no carry
    handling beyond renormalization, no rangecoder optimization, no
    bit-rotated state — just the canonical Witten-Neal-Cleary algorithm.

    Usage:

        coder = ArithmeticCoder()
        for sym, cdf_row in zip(symbols, cdf_rows):
            coder.encode_symbol(sym, cdf_row)
        encoded = coder.finish_encoding()

        coder2 = ArithmeticCoder.from_bytes(encoded)
        decoded = [coder2.decode_symbol(cdf_row) for cdf_row in cdf_rows]
    """

    _STATE_BITS = 32
    _STATE_MAX = (1 << _STATE_BITS) - 1
    _QUARTER = 1 << (_STATE_BITS - 2)
    _HALF = 2 * _QUARTER
    _THREE_QUARTER = 3 * _QUARTER

    def __init__(self) -> None:
        self._low = 0
        self._high = self._STATE_MAX
        self._pending_bits = 0
        self._bit_buf: list[int] = []
        # Decoder state
        self._dec_value = 0
        self._dec_pos = 0
        self._dec_bytes = b""

    def encode_symbol(self, symbol: int, cdf_row: np.ndarray) -> None:
        """Encode one symbol given its class-conditional CDF row."""
        if cdf_row.dtype != np.uint16:
            raise ValueError("cdf_row must be uint16")
        if symbol < 0 or symbol >= len(cdf_row) - 1:
            raise ValueError(
                f"symbol {symbol} out of range [0, {len(cdf_row) - 1})"
            )
        cdf_lo = int(cdf_row[symbol])
        cdf_hi = int(cdf_row[symbol + 1])
        rng = self._high - self._low + 1
        self._high = self._low + (rng * cdf_hi) // CDF_MAX - 1
        self._low = self._low + (rng * cdf_lo) // CDF_MAX
        self._renormalize_encoder()

    def _renormalize_encoder(self) -> None:
        while True:
            if self._high < self._HALF:
                self._emit_bit(0)
            elif self._low >= self._HALF:
                self._emit_bit(1)
                self._low -= self._HALF
                self._high -= self._HALF
            elif self._low >= self._QUARTER and self._high < self._THREE_QUARTER:
                self._pending_bits += 1
                self._low -= self._QUARTER
                self._high -= self._QUARTER
            else:
                break
            self._low <<= 1
            self._high = (self._high << 1) | 1
            # Clamp to 32-bit state
            self._low &= self._STATE_MAX
            self._high &= self._STATE_MAX

    def _emit_bit(self, bit: int) -> None:
        self._bit_buf.append(bit)
        for _ in range(self._pending_bits):
            self._bit_buf.append(1 - bit)
        self._pending_bits = 0

    def finish_encoding(self) -> bytes:
        """Finalize encoder and return the byte stream."""
        self._pending_bits += 1
        if self._low < self._QUARTER:
            self._emit_bit(0)
        else:
            self._emit_bit(1)
        # Pad to byte boundary
        while len(self._bit_buf) % 8 != 0:
            self._bit_buf.append(0)
        out = bytearray()
        for i in range(0, len(self._bit_buf), 8):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | self._bit_buf[i + j]
            out.append(byte)
        return bytes(out)

    @classmethod
    def from_bytes(cls, data: bytes) -> ArithmeticCoder:
        """Construct decoder over previously-encoded bytes."""
        c = cls()
        c._dec_bytes = data
        c._dec_pos = 0
        c._dec_value = 0
        for _ in range(c._STATE_BITS):
            c._dec_value = (c._dec_value << 1) | c._read_bit()
        return c

    def _read_bit(self) -> int:
        byte_idx = self._dec_pos // 8
        bit_idx = 7 - (self._dec_pos % 8)
        self._dec_pos += 1
        if byte_idx >= len(self._dec_bytes):
            return 0
        return (self._dec_bytes[byte_idx] >> bit_idx) & 1

    def decode_symbol(self, cdf_row: np.ndarray) -> int:
        """Decode one symbol given its class-conditional CDF row."""
        if cdf_row.dtype != np.uint16:
            raise ValueError("cdf_row must be uint16")
        rng = self._high - self._low + 1
        # Position in [0, CDF_MAX]
        scaled = ((self._dec_value - self._low + 1) * CDF_MAX - 1) // rng
        # Binary search for symbol s s.t. cdf_row[s] <= scaled < cdf_row[s+1]
        symbol = int(np.searchsorted(cdf_row, scaled, side="right") - 1)
        if symbol < 0:
            symbol = 0
        max_s = len(cdf_row) - 2
        if symbol > max_s:
            symbol = max_s
        cdf_lo = int(cdf_row[symbol])
        cdf_hi = int(cdf_row[symbol + 1])
        self._high = self._low + (rng * cdf_hi) // CDF_MAX - 1
        self._low = self._low + (rng * cdf_lo) // CDF_MAX
        self._renormalize_decoder()
        return symbol

    def _renormalize_decoder(self) -> None:
        while True:
            if self._high < self._HALF:
                pass
            elif self._low >= self._HALF:
                self._dec_value -= self._HALF
                self._low -= self._HALF
                self._high -= self._HALF
            elif self._low >= self._QUARTER and self._high < self._THREE_QUARTER:
                self._dec_value -= self._QUARTER
                self._low -= self._QUARTER
                self._high -= self._QUARTER
            else:
                break
            self._low <<= 1
            self._high = (self._high << 1) | 1
            self._dec_value = (self._dec_value << 1) | self._read_bit()
            self._low &= self._STATE_MAX
            self._high &= self._STATE_MAX
            self._dec_value &= self._STATE_MAX
