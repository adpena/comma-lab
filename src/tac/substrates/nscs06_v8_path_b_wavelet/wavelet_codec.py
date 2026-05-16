# SPDX-License-Identifier: MIT
"""DB4 depth-2 separable 2D DWT + per-subband Mallat hierarchical arithmetic coding.

This is the UNIQUE substrate-distinguishing core of v8 Path B. The compress
side decomposes each (gray, Cb, Cr) channel of each frame into 7 subbands
(LL2 + LH2/HL2/HH2 + LH1/HL1/HH1) via separable 2D DWT, then per-(subband,
class) Laplacian-prior arithmetic-codes the quantized coefficients per
Mallat 1989 hierarchical scale-by-scale coding canon.

Per the design memo Section 4 + Section 8 + Section 16:

* **DB4 depth-2** is the canonical natural-image decorrelation choice
  (Mallat 1989; Daubechies 1992 Ch. 6) — HARD-EARNED assumption.
* **Laplacian prior** with per-class scale b_c is the max-entropy distribution
  under the mean-absolute constraint (MacKay 2003 ITILA Ch. 2) — HARD-EARNED.
* **Per-subband uniform quantization** (Antonini-Barlaud-Mathieu-Daubechies
  1992) — preserved as the empirical sweet spot.

Module surface:

* :data:`DB4_DECOMP_LO` / :data:`DB4_DECOMP_HI` / :data:`DB4_RECON_LO` /
  :data:`DB4_RECON_HI` — canonical Daubechies-4 wavelet filter coefficients
  (length-8 orthonormal filter bank per Daubechies 1992 Table 6.1).
* :func:`dwt2_db4_depth2` / :func:`idwt2_db4_depth2` — forward/inverse
  separable 2D DWT at depth 2, using pywavelets ``pywt.wavedec2`` / ``waverec2``
  with periodic boundary handling for byte-deterministic roundtrip.
* :func:`quantize_subband` / :func:`dequantize_subband` — per-subband uniform
  quantization with per-band step sizes from :data:`PER_SUBBAND_QUANT_STEPS`.
* :func:`build_per_subband_laplacian_priors` — empirical Laplacian scale b_c
  per (subband, SegNet class) from compress-time GT video samples.
* :func:`laplacian_cdf_uint16` — convert continuous Laplacian PMF to uint16
  CDF compatible with the v7-canonical :class:`ArithmeticCoder` from
  :mod:`tac.substrates.nscs06_carmack_hotz_strip_everything.codec`.
* :func:`encode_subband_arith` — encode a quantized subband via the canonical
  per-class arith coder.

This module is UNIQUE per UNIQUE-AND-COMPLETE-PER-METHOD operating mode.
It does NOT route through any canonical helper that would force-fit a
neural-substrate gradient path onto this analytical codec.

References:
- Daubechies 1992 *Ten Lectures on Wavelets* Ch. 6 (DB4 filter coefficients)
- Mallat 1989 *Multiresolution approximation and wavelet orthonormal bases*
- Antonini-Barlaud-Mathieu-Daubechies 1992 *Image coding using wavelet transform*
- Witten-Neal-Cleary 1987 *Arithmetic coding for data compression*
- MacKay 2003 *Information Theory, Inference, and Learning Algorithms* Ch. 2
- Wyner-Ziv 1976 (consumer; sister module wyner_ziv_temporal.py)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pywt

from tac.substrates.nscs06_carmack_hotz_strip_everything.codec import (
    CDF_MAX,
    NUM_SEGNET_CLASSES,
    ArithmeticCoder,
)

# ---------------------------------------------------------------------------
# Canonical Daubechies-4 filter coefficients (Daubechies 1992 Table 6.1)
# ---------------------------------------------------------------------------

# pywt's "db4" wavelet uses length-8 filter coefficients.
_DB4 = pywt.Wavelet("db4")
DB4_DECOMP_LO: np.ndarray = np.asarray(_DB4.dec_lo, dtype=np.float64)
DB4_DECOMP_HI: np.ndarray = np.asarray(_DB4.dec_hi, dtype=np.float64)
DB4_RECON_LO: np.ndarray = np.asarray(_DB4.rec_lo, dtype=np.float64)
DB4_RECON_HI: np.ndarray = np.asarray(_DB4.rec_hi, dtype=np.float64)

DWT_LEVEL: int = 2
"""Mallat hierarchical decomposition depth. Per design memo Section 6.3,
depth-2 is the empirical sweet spot for natural video: ~98% energy compaction
in 25% of coefficients while leaving sufficient detail-band bandwidth for
motion residuals (deeper levels saturate)."""

NUM_SUBBANDS: int = 7
"""depth-2 separable 2D DWT yields 7 subbands: LL2 + (LH2, HL2, HH2) at
depth-2 + (LH1, HL1, HH1) at depth-1. Per pywt's wavedec2 ordering convention."""

SUBBAND_LABELS: tuple[str, ...] = ("LL2", "LH2", "HL2", "HH2", "LH1", "HL1", "HH1")
"""Canonical ordering of the 7 subbands. INVARIANT — the WLV2 archive grammar
indexes streams by this order."""

# Per-subband quantization step sizes per Mallat 1989: detail bands get coarser
# steps than approximation. The schedule below is empirically derived for
# natural-video grayscale + chroma; smoke/full will calibrate step sizes per
# the design memo Section 7 "post-training" adaptive search.
PER_SUBBAND_QUANT_STEPS: tuple[int, ...] = (1, 4, 4, 8, 4, 4, 8)
"""Per-subband quantization step:
(LL2=1, LH2=4, HL2=4, HH2=8, LH1=4, HL1=4, HH1=8) — coarser for diagonal +
deeper bands per Mallat 1989."""

# Quantized-coefficient support range. T=20 (signed) means quantized values
# clamp to [-20, +20]; the Laplacian-prior CDF table is (2T+1)=41 levels wide.
QUANT_CLAMP_T: int = 20
QUANT_LEVELS: int = 2 * QUANT_CLAMP_T + 1
QUANT_ZERO_INDEX: int = QUANT_CLAMP_T  # quantized 0 maps to symbol index 20

# ---------------------------------------------------------------------------
# DB4 depth-2 separable 2D DWT (pywt-backed; periodic boundary for byte-stability)
# ---------------------------------------------------------------------------


def dwt2_db4_depth2(image: np.ndarray) -> list[np.ndarray]:
    """Forward DB4 depth-2 separable 2D DWT.

    Args:
        image: float64 2D array (H, W). H + W must be divisible by 4 so
            depth-2 yields integer-sized subbands.

    Returns:
        List of 7 subbands in :data:`SUBBAND_LABELS` order:
        [LL2, LH2, HL2, HH2, LH1, HL1, HH1].

    Per Daubechies 1992 + pywt convention: depth-2 decomposes the LL1
    approximation (from depth-1) into (LL2, LH2, HL2, HH2). Pywt's
    ``wavedec2`` returns ``[LL2, (LH2, HL2, HH2), (LH1, HL1, HH1)]`` so
    we flatten to a 7-list in our canonical ordering.

    Boundary mode: "periodization" gives byte-stable roundtrip without
    extension-pad ambiguity per design memo Section 6.3.
    """
    if image.ndim != 2:
        raise ValueError(f"image must be 2D; got shape {image.shape}")
    if image.dtype != np.float64:
        image = image.astype(np.float64)
    h, w = image.shape
    if h % 4 != 0 or w % 4 != 0:
        raise ValueError(
            f"DB4 depth-2 requires H and W divisible by 4; got ({h}, {w})"
        )
    coeffs = pywt.wavedec2(image, wavelet="db4", level=DWT_LEVEL, mode="periodization")
    # coeffs = [LL2, (LH2, HL2, HH2), (LH1, HL1, HH1)]
    LL2 = coeffs[0]
    LH2, HL2, HH2 = coeffs[1]
    LH1, HL1, HH1 = coeffs[2]
    return [LL2, LH2, HL2, HH2, LH1, HL1, HH1]


def idwt2_db4_depth2(subbands: list[np.ndarray], output_shape: tuple[int, int]) -> np.ndarray:
    """Inverse DB4 depth-2 separable 2D DWT.

    Args:
        subbands: list of 7 subbands in :data:`SUBBAND_LABELS` order.
        output_shape: (H, W) — required because pywavelets may pad/truncate
            by 1 sample on boundaries depending on input size. We use the
            stored OUTPUT_HEIGHT/OUTPUT_WIDTH from the archive header.

    Returns:
        float64 2D array (H, W) reconstructed image.
    """
    if len(subbands) != NUM_SUBBANDS:
        raise ValueError(
            f"expected {NUM_SUBBANDS} subbands; got {len(subbands)}"
        )
    LL2, LH2, HL2, HH2, LH1, HL1, HH1 = subbands
    coeffs = [LL2, (LH2, HL2, HH2), (LH1, HL1, HH1)]
    rec = pywt.waverec2(coeffs, wavelet="db4", mode="periodization")
    h, w = output_shape
    # Pywt may produce slightly off-by-one shape if input was not a power of 2
    # exactly aligned to the wavelet length; crop to declared output_shape.
    rec = rec[:h, :w]
    return rec.astype(np.float64)


# ---------------------------------------------------------------------------
# Per-subband uniform quantization
# ---------------------------------------------------------------------------


def quantize_subband(coeff: np.ndarray, step: int) -> np.ndarray:
    """Quantize wavelet subband to int8 symbols in [-QUANT_CLAMP_T, +QUANT_CLAMP_T].

    Args:
        coeff: float64 wavelet subband.
        step: positive int quantization step size from PER_SUBBAND_QUANT_STEPS.

    Returns:
        int8 array same shape as `coeff` with quantized symbols.
        Coefficients beyond [-T*step, T*step] clip to ±T.
    """
    if step <= 0:
        raise ValueError(f"step must be > 0; got {step}")
    q = np.round(coeff / float(step)).astype(np.int32)
    q = np.clip(q, -QUANT_CLAMP_T, QUANT_CLAMP_T)
    return q.astype(np.int8)


def dequantize_subband(q: np.ndarray, step: int) -> np.ndarray:
    """Inverse uniform quantization: int8 symbols -> float64."""
    if step <= 0:
        raise ValueError(f"step must be > 0; got {step}")
    return q.astype(np.float64) * float(step)


# ---------------------------------------------------------------------------
# Per-(subband, class) Laplacian prior + CDF (MacKay 2003 Ch. 2 max-entropy)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerSubbandLaplacianPriors:
    """Empirical Laplacian scale b_c per (subband, SegNet class).

    Shape: ``(NUM_SUBBANDS, NUM_SEGNET_CLASSES)`` float32. Stored in the
    archive as ``NUM_SUBBANDS * NUM_SEGNET_CLASSES * 4`` bytes.
    """

    scales: np.ndarray  # (NUM_SUBBANDS, NUM_SEGNET_CLASSES) float32

    def __post_init__(self) -> None:
        if self.scales.dtype != np.float32:
            raise ValueError(f"scales must be float32; got {self.scales.dtype}")
        if self.scales.shape != (NUM_SUBBANDS, NUM_SEGNET_CLASSES):
            raise ValueError(
                f"scales must be ({NUM_SUBBANDS}, {NUM_SEGNET_CLASSES}); "
                f"got {self.scales.shape}"
            )
        if not np.all(self.scales > 0):
            raise ValueError("all scales must be positive")


def build_per_subband_laplacian_priors(
    quantized_subbands_per_class: dict[int, list[np.ndarray]],
) -> PerSubbandLaplacianPriors:
    """Estimate per-(subband, class) Laplacian scale b_c = mean(|coeff|).

    Per MacKay 2003 ITILA Ch. 2: the Laplacian distribution p(x) = (1/2b) exp(-|x|/b)
    is the maximum-entropy distribution under the mean-absolute constraint
    E[|X|] = b. The empirical b_c is therefore mean(|coeff|) over the samples
    for class c at that subband.

    Args:
        quantized_subbands_per_class: dict mapping class_idx -> list of
            quantized subband arrays. Each list must have NUM_SUBBANDS entries.

    Returns:
        :class:`PerSubbandLaplacianPriors`.
    """
    scales = np.zeros((NUM_SUBBANDS, NUM_SEGNET_CLASSES), dtype=np.float32)
    for c in range(NUM_SEGNET_CLASSES):
        if c not in quantized_subbands_per_class:
            # No samples; use a small default scale (sparse prior)
            scales[:, c] = 1.0
            continue
        bands = quantized_subbands_per_class[c]
        if len(bands) != NUM_SUBBANDS:
            raise ValueError(
                f"class {c} got {len(bands)} subbands; expected {NUM_SUBBANDS}"
            )
        for s, band in enumerate(bands):
            if band.size == 0:
                scales[s, c] = 1.0
                continue
            b = float(np.mean(np.abs(band.astype(np.float64))))
            # Clamp to [0.5, 50] to keep CDF well-conditioned numerically.
            scales[s, c] = max(0.5, min(50.0, b))
    return PerSubbandLaplacianPriors(scales=scales)


def laplacian_cdf_uint16(scale: float) -> np.ndarray:
    """Build a uint16 CDF over QUANT_LEVELS symbols from a Laplacian PMF.

    The PMF at integer symbol k (k in [-T, +T]) is:
        p(k) ∝ exp(-|k| / scale)
    The CDF is the cumulative sum scaled to CDF_MAX. The resulting array is
    compatible with the v7 :class:`ArithmeticCoder` interface (which expects
    a (QUANT_LEVELS + 1,) uint16 CDF with CDF[0]=0 and CDF[-1]=CDF_MAX).

    Args:
        scale: Laplacian scale parameter b > 0. Larger b = flatter PMF.

    Returns:
        uint16 array of length QUANT_LEVELS + 1.
    """
    if scale <= 0:
        raise ValueError(f"scale must be > 0; got {scale}")
    ks = np.arange(-QUANT_CLAMP_T, QUANT_CLAMP_T + 1, dtype=np.float64)
    pmf = np.exp(-np.abs(ks) / float(scale))
    pmf = pmf / pmf.sum()  # normalize
    # Build integer counts with Laplace smoothing (no zero-prob symbol).
    counts = (pmf * (1 << 20)).astype(np.int64)
    counts = np.maximum(counts, 1)
    cum = np.cumsum(counts)
    # Scale so cum[-1] -> CDF_MAX exactly; intermediate values floor-truncate.
    cum_scaled = (cum.astype(np.float64) / float(cum[-1]) * CDF_MAX).astype(np.int64)
    cum_scaled[-1] = CDF_MAX
    # Strict monotone from the front. If a forward fix would push above
    # CDF_MAX-1 at the penultimate slot (so the final cum_scaled[-1]=CDF_MAX
    # would not strictly exceed it), back off the offender immediately.
    for i in range(1, len(cum_scaled) - 1):
        if cum_scaled[i] <= cum_scaled[i - 1]:
            cum_scaled[i] = cum_scaled[i - 1] + 1
    # Ensure penultimate < CDF_MAX so cum_scaled[-1]=CDF_MAX is strictly
    # greater. If forward repair pushed it ≥ CDF_MAX, scan backwards and
    # squash each colliding entry until the chain is monotone-strict.
    for i in range(len(cum_scaled) - 1, 0, -1):
        if cum_scaled[i - 1] >= cum_scaled[i]:
            cum_scaled[i - 1] = cum_scaled[i] - 1
    cum_scaled[-1] = CDF_MAX
    cdf = np.zeros(QUANT_LEVELS + 1, dtype=np.int64)
    cdf[0] = 0
    cdf[1:] = cum_scaled
    return cdf.astype(np.uint16)


# ---------------------------------------------------------------------------
# Per-subband arith encoding (consumes the v7 ArithmeticCoder primitive)
# ---------------------------------------------------------------------------


def encode_subband_arith(
    quantized: np.ndarray,
    class_labels: np.ndarray,
    priors: PerSubbandLaplacianPriors,
    subband_index: int,
) -> bytes:
    """Arith-encode a quantized subband with per-class Laplacian-prior CDFs.

    Args:
        quantized: int8 subband (any shape) with values in [-T, +T].
        class_labels: uint8 SegNet class label per coefficient (same shape).
        priors: per-(subband, class) Laplacian scales.
        subband_index: which subband (0..NUM_SUBBANDS-1) to look up scales for.

    Returns:
        Arith-coded byte stream. Decoder needs (priors, subband_index,
        class_labels) to reverse.
    """
    if quantized.shape != class_labels.shape:
        raise ValueError(
            f"quantized.shape={quantized.shape} != class_labels.shape={class_labels.shape}"
        )
    if quantized.dtype != np.int8:
        raise ValueError(f"quantized must be int8; got {quantized.dtype}")
    if class_labels.dtype != np.uint8:
        raise ValueError(f"class_labels must be uint8; got {class_labels.dtype}")
    if subband_index < 0 or subband_index >= NUM_SUBBANDS:
        raise ValueError(
            f"subband_index out of [0, {NUM_SUBBANDS}); got {subband_index}"
        )
    # Pre-build per-class CDFs for this subband.
    cdfs_per_class: list[np.ndarray] = []
    for c in range(NUM_SEGNET_CLASSES):
        scale = float(priors.scales[subband_index, c])
        cdfs_per_class.append(laplacian_cdf_uint16(scale))
    coder = ArithmeticCoder()
    flat_q = quantized.ravel()
    flat_c = class_labels.ravel()
    for q, c in zip(flat_q, flat_c):
        # Map signed quant value in [-T, +T] to symbol index in [0, QUANT_LEVELS)
        symbol = int(q) + QUANT_ZERO_INDEX
        if symbol < 0 or symbol >= QUANT_LEVELS:
            raise ValueError(f"symbol {symbol} out of [0, {QUANT_LEVELS})")
        coder.encode_symbol(symbol, cdfs_per_class[int(c)])
    return coder.finish_encoding()


def decode_subband_arith(
    encoded: bytes,
    class_labels: np.ndarray,
    priors: PerSubbandLaplacianPriors,
    subband_index: int,
) -> np.ndarray:
    """Inverse of :func:`encode_subband_arith`."""
    if class_labels.dtype != np.uint8:
        raise ValueError(f"class_labels must be uint8; got {class_labels.dtype}")
    if subband_index < 0 or subband_index >= NUM_SUBBANDS:
        raise ValueError(
            f"subband_index out of [0, {NUM_SUBBANDS}); got {subband_index}"
        )
    cdfs_per_class: list[np.ndarray] = []
    for c in range(NUM_SEGNET_CLASSES):
        scale = float(priors.scales[subband_index, c])
        cdfs_per_class.append(laplacian_cdf_uint16(scale))
    coder = ArithmeticCoder.from_bytes(encoded)
    flat_c = class_labels.ravel()
    out = np.zeros(flat_c.size, dtype=np.int8)
    for i, c in enumerate(flat_c):
        symbol = coder.decode_symbol(cdfs_per_class[int(c)])
        out[i] = symbol - QUANT_ZERO_INDEX
    return out.reshape(class_labels.shape)
