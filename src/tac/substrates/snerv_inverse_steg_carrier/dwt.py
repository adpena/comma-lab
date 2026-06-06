# SPDX-License-Identifier: MIT
"""Orthonormal multi-level 2D DWT for the SNeRV inverse-steganalysis carrier.

The SNeRV carrier (arXiv 2501.01681 — Spectra-preserving NeRV) stores ONLY the low-frequency
(coarse) approximation subband of a multi-level 2D wavelet decomposition and
GENERATES the high-frequency detail subbands with a small decoder. This cures the
"Z8 disease" (`feedback_z8_*`): the byte-heavy detail blob is *generated*, not
stored, so "wavelet blobs are too big" does not apply.

Why this exact transform:

  * **Square orthonormal by construction.** ``pywt`` with ``mode="periodization"``
    on even-dimensioned inputs produces a coefficient count equal to ``H*W`` — the
    analysis operator ``A`` is a square orthonormal matrix. For an orthonormal
    transform ``A^T == A^{-1} == synthesis``, so the synthesis path IS the exact
    adjoint (closes the G3 adjoint requirement by construction, rel-residual 0.0
    by floating point — see ``test_dwt_adjoint.py``).
  * **Exact reconstruction.** ``idwt2(dwt2(x)) == x`` to ~1e-15 (orthonormal
    perfect reconstruction, Mallat 1989 §7.5).
  * **Detail subbands are a structural dead-zone for ``d_seg``.** SegNet is
    ``smp.Unet('tu-efficientnet_b2')`` with a stride-2 stem; spatial detail below
    the stem's effective resolution is invisible to the argmax-flip metric, so the
    finest detail subbands carry near-zero ``d_seg`` saliency (Yousfi lens, design
    memo §3.1). Generating them rather than storing them is therefore cheap-by-
    construction in the detector's eyes.

[verified-against: Mallat 1989 "A theory for multiresolution signal
decomposition" §7.5 (orthonormal Daubechies synthesis = time-reversed analysis);
Daubechies 1988 orthonormal compactly-supported wavelets; SNeRV arXiv 2501.01681
(store-LF / generate-HF). Adjoint exactness proven by the dot-product test in
``tests/test_dwt_adjoint.py`` (rel-residual 0.0).]

Catalog #290 (canonical-vs-unique): pywt is ADOPTED canonical because its
orthonormal periodization DWT gives a square transform with the exact synthesis
    adjoint for FREE on the padded canvas. Native frame scoring uses the cropped
    synthesis path, so pixel-domain cotangents are zero-embedded onto that padded
    canvas before analysis (``dwt2_native_synthesis_adjoint``). A fork (hand-rolled
    separable filter banks, e.g. the Z8 ``mallat_dwt_adapter``) would suppress this
    by coupling to the dirty Z8 dir and re-implementing a tested primitive; FORK
    rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

try:  # pragma: no cover - import guard
    import pywt
except ImportError:  # pragma: no cover
    pywt = None

__all__ = [
    "DEFAULT_WAVELET",
    "DWT_MODE",
    "OFFICIAL_SNERV_HAAR_MODE_PROOF",
    "SNERV_RECEIVER_DWT_RUNTIME_CUSTODY_PROOF",
    "SnervDwtError",
    "WaveletPyramid",
    "dwt2_multilevel",
    "dwt2_native_synthesis_adjoint",
    "idwt2_multilevel",
    "lf_coeff_count",
    "synthesis_adjoint_residual",
]

# The ONLY mode that makes the 2D DWT a square orthonormal operator on even dims.
DWT_MODE: Final[str] = "periodization"
DEFAULT_WAVELET: Final[str] = "db2"
OFFICIAL_SNERV_HAAR_MODE_PROOF: Final[str] = (
    "official_snerv_haar_j1_compatible_numpy_periodized_orthonormal_path"
)
SNERV_RECEIVER_DWT_RUNTIME_CUSTODY_PROOF: Final[str] = (
    "receiver_can_decode_wavelet_haar_without_pywavelets_dependency"
)


class SnervDwtError(ValueError):
    """Raised when the SNeRV DWT contract is violated."""


def _pad_to_square_dims(hw: tuple[int, int], levels: int) -> tuple[int, int]:
    """Round (H, W) UP to the next multiple of ``2**levels``.

    Only when both spatial dims are multiples of ``2**levels`` is the
    periodization 2D DWT a SQUARE orthonormal operator (coeff count == H*W), which
    is what makes the synthesis the EXACT adjoint (rel-residual 0.0). Native
    dashcam frames (874x1164) have odd factors that break this; we pad to the next
    multiple, transform on the padded canvas, and crop the padding on decode. This
    is the canonical SNeRV practice and keeps the G3 adjoint exact by construction.
    """
    m = 1 << int(levels)
    h, w = hw
    ph = ((int(h) + m - 1) // m) * m
    pw = ((int(w) + m - 1) // m) * m
    return ph, pw


def _pad_reflect(image: np.ndarray, padded_hw: tuple[int, int]) -> np.ndarray:
    h, w = image.shape
    ph, pw = padded_hw
    if (ph, pw) == (h, w):
        return image
    return np.pad(image, ((0, ph - h), (0, pw - w)), mode="reflect")


def _zero_embed_native(image: np.ndarray, padded_hw: tuple[int, int]) -> np.ndarray:
    h, w = image.shape
    ph, pw = padded_hw
    if ph < h or pw < w:
        raise SnervDwtError(
            f"padded_hw {padded_hw} must cover native image shape {(h, w)}"
        )
    canvas = np.zeros((ph, pw), dtype=np.float64)
    canvas[:h, :w] = image
    return canvas


@dataclass(frozen=True)
class WaveletPyramid:
    """A multi-level orthonormal 2D wavelet pyramid for one (H, W) channel.

    ``coeffs`` is the ``pywt.wavedec2`` list: ``[LL_L, (LH_L, HL_L, HH_L), ...,
    (LH_1, HL_1, HH_1)]`` where ``LL_L`` is the coarsest approximation (the LF
    subband the SNeRV carrier STORES) and the detail tuples are the HF subbands
    the SNeRV decoder GENERATES.

    ``orig_hw`` is the NATIVE frame shape (the decode crop target). ``padded_hw``
    is the square canvas the transform actually ran on (== orig_hw when native is
    already a multiple of ``2**levels``). The transform is square+orthonormal on
    ``padded_hw`` so the synthesis is the exact adjoint there.
    """

    coeffs: list  # pywt wavedec2 coefficient list (numpy arrays)
    levels: int
    wavelet: str
    orig_hw: tuple[int, int]
    padded_hw: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.padded_hw is None:
            object.__setattr__(
                self, "padded_hw", _pad_to_square_dims(self.orig_hw, self.levels)
            )

    @property
    def lf(self) -> np.ndarray:
        """The coarsest LL approximation subband (the stored LF coefficients)."""
        return np.asarray(self.coeffs[0], dtype=np.float64)

    @property
    def details(self) -> list:
        """The HF detail tuples (the generated subbands), coarsest-first."""
        return list(self.coeffs[1:])

    def total_coeff_count(self) -> int:
        n = int(self.lf.size)
        for lh, hl, hh in self.details:
            n += int(lh.size) + int(hl.size) + int(hh.size)
        return n


def _validate_image(image: np.ndarray, levels: int) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim != 2:
        raise SnervDwtError(f"image must be 2D (H, W); got shape {arr.shape}")
    if levels < 1:
        raise SnervDwtError(f"levels must be >= 1; got {levels}")
    return arr


def _validate_levels_for_padded_shape(
    orig_hw: tuple[int, int],
    padded_hw: tuple[int, int],
    levels: int,
    wavelet: str,
) -> None:
    if _is_numpy_haar(wavelet):
        if min(padded_hw) < (1 << int(levels)):
            raise SnervDwtError(
                f"levels={levels} exceeds Haar capacity for native shape {orig_hw} "
                f"(padded to {padded_hw})"
            )
        return
    _require_pywt()
    max_lvl = pywt.dwtn_max_level(padded_hw, pywt.Wavelet(wavelet))
    if levels > max_lvl:
        raise SnervDwtError(
            f"levels={levels} exceeds max {max_lvl} for native shape {orig_hw} "
            f"(padded to {padded_hw}) with wavelet {wavelet!r}"
        )


def dwt2_multilevel(
    image: np.ndarray,
    *,
    levels: int = 3,
    wavelet: str = DEFAULT_WAVELET,
) -> WaveletPyramid:
    """Forward multi-level orthonormal 2D analysis DWT (the carrier encoder).

    Returns a :class:`WaveletPyramid`. The LF subband (``.lf``) is what the SNeRV
    carrier stores; the detail subbands (``.details``) are the HF the decoder
    generates. With ``mode="periodization"`` the total coefficient count equals
    ``H*W`` exactly (square orthonormal transform).
    """
    arr = _validate_image(image, levels)
    orig_hw = (int(arr.shape[0]), int(arr.shape[1]))
    padded_hw = _pad_to_square_dims(orig_hw, levels)
    _validate_levels_for_padded_shape(orig_hw, padded_hw, levels, wavelet)
    canvas = _pad_reflect(arr, padded_hw)
    coeffs = (
        _haar_wavedec2_periodized(canvas, levels=levels)
        if _is_numpy_haar(wavelet)
        else pywt.wavedec2(canvas, wavelet, mode=DWT_MODE, level=levels)
    )
    return WaveletPyramid(
        coeffs=coeffs,
        levels=int(levels),
        wavelet=str(wavelet),
        orig_hw=orig_hw,
        padded_hw=padded_hw,
    )


def dwt2_native_synthesis_adjoint(
    pixel_cotangent: np.ndarray,
    *,
    levels: int = 3,
    wavelet: str = DEFAULT_WAVELET,
) -> WaveletPyramid:
    """Adjoint of native cropped synthesis: pixels -> all wavelet coefficients.

    ``idwt2_multilevel`` decodes coefficients on the padded orthonormal canvas and
    crops the reconstructed image back to native ``orig_hw``. The exact adjoint of
    that native synthesis path is therefore:

        native pixel cotangent -> zero-embed into padded canvas -> DWT analysis

    This differs from :func:`dwt2_multilevel`, which reflect-pads an input image for
    encoder analysis. Reflect padding is correct for encoding native pixels, but it
    is NOT the adjoint of the crop operation on odd native dimensions.
    """
    arr = _validate_image(pixel_cotangent, levels)
    orig_hw = (int(arr.shape[0]), int(arr.shape[1]))
    padded_hw = _pad_to_square_dims(orig_hw, levels)
    _validate_levels_for_padded_shape(orig_hw, padded_hw, levels, wavelet)
    canvas = _zero_embed_native(arr, padded_hw)
    coeffs = (
        _haar_wavedec2_periodized(canvas, levels=levels)
        if _is_numpy_haar(wavelet)
        else pywt.wavedec2(canvas, wavelet, mode=DWT_MODE, level=levels)
    )
    return WaveletPyramid(
        coeffs=coeffs,
        levels=int(levels),
        wavelet=str(wavelet),
        orig_hw=orig_hw,
        padded_hw=padded_hw,
    )


def idwt2_multilevel(pyramid: WaveletPyramid) -> np.ndarray:
    """Inverse multi-level orthonormal 2D synthesis DWT (= the exact adjoint).

    For the orthonormal periodization transform this is BOTH the inverse
    (perfect reconstruction) AND the adjoint of the forward analysis (Mallat
    §7.5). Returns the reconstructed ``(H, W)`` image, cropped to ``orig_hw``
    (periodization can round dims up to the next multiple of the filter length).
    """
    if _is_numpy_haar(pyramid.wavelet):
        recon = _haar_waverec2_periodized(pyramid.coeffs)
    else:
        _require_pywt()
        recon = pywt.waverec2(pyramid.coeffs, pyramid.wavelet, mode=DWT_MODE)
    h, w = pyramid.orig_hw
    return np.asarray(recon[:h, :w], dtype=np.float64)


def lf_coeff_count(
    orig_hw: tuple[int, int],
    levels: int,
    wavelet: str = DEFAULT_WAVELET,
) -> int:
    """The number of LF (LL approximation) coefficients the carrier STORES.

    This is the SNeRV rate lever: the LF subband at level ``L`` is ~``(H*W) / 4^L``
    coefficients (each level halves both spatial dims). The detail subbands hold
    the rest and are generated, not stored.

    The count is wavelet-independent for periodized orthonormal 2D DWT on the
    padded ``2**levels`` lattice, so this helper intentionally avoids executing
    PyWavelets. Budget/modelsize controllers call this path frequently and must
    stay receiver-safe even when the training DWT backend is unavailable.
    """
    h, w = (int(orig_hw[0]), int(orig_hw[1]))
    if h <= 0 or w <= 0:
        raise SnervDwtError(f"orig_hw must be positive; got {orig_hw}")
    if int(levels) < 1:
        raise SnervDwtError(f"levels must be >= 1; got {levels}")
    if not str(wavelet).strip():
        raise SnervDwtError("wavelet must be a non-empty string")
    ph, pw = _pad_to_square_dims((h, w), int(levels))
    scale = 1 << int(levels)
    return int((ph // scale) * (pw // scale))


def synthesis_adjoint_residual(
    orig_hw: tuple[int, int],
    *,
    levels: int = 3,
    wavelet: str = DEFAULT_WAVELET,
    seed: int = 0,
) -> float:
    """The G3 native adjoint exactness number from a live dot-product test.

    Proves ``<S c, g> == <c, S^T g>`` where ``S`` is the native decode path
    (orthonormal synthesis on the padded canvas, then crop to ``orig_hw``) and
    ``S^T`` is :func:`dwt2_native_synthesis_adjoint`. For odd native dimensions,
    this specifically verifies the crop-aware zero-embed adjoint rather than the
    encoder's reflect-padded analysis path.

    Returns the relative residual ``|lhs - rhs| / (|lhs| + 1e-30)``.
    """
    rng = np.random.default_rng(seed)
    ph, pw = _pad_to_square_dims(orig_hw, levels)
    _validate_levels_for_padded_shape(orig_hw, (ph, pw), levels, wavelet)
    template_coeffs = (
        _haar_wavedec2_periodized(np.zeros((ph, pw), dtype=np.float64), levels=levels)
        if _is_numpy_haar(wavelet)
        else pywt.wavedec2(
            np.zeros((ph, pw), dtype=np.float64), wavelet, mode=DWT_MODE, level=levels
        )
    )
    c_struct = _random_like_coeffs(template_coeffs, rng)
    c_vec = _flatten_coeffs(c_struct)
    pixel_cotangent = rng.standard_normal(orig_hw).astype(np.float64)
    synth_native = idwt2_multilevel(
        WaveletPyramid(
            coeffs=c_struct,
            levels=int(levels),
            wavelet=str(wavelet),
            orig_hw=orig_hw,
            padded_hw=(ph, pw),
        )
    )
    adjoint_coeffs = dwt2_native_synthesis_adjoint(
        pixel_cotangent, levels=levels, wavelet=wavelet
    ).coeffs

    lhs = float(np.dot(synth_native.ravel(), pixel_cotangent.ravel()))
    rhs = float(np.dot(c_vec, _flatten_coeffs(adjoint_coeffs)))
    return abs(lhs - rhs) / (abs(lhs) + 1e-30)


def _flatten_coeffs(coeffs: list) -> np.ndarray:
    parts = [np.asarray(coeffs[0], dtype=np.float64).ravel()]
    for lh, hl, hh in coeffs[1:]:
        parts.append(np.asarray(lh, dtype=np.float64).ravel())
        parts.append(np.asarray(hl, dtype=np.float64).ravel())
        parts.append(np.asarray(hh, dtype=np.float64).ravel())
    return np.concatenate(parts)


def _random_like_coeffs(coeffs: list, rng: np.random.Generator) -> list:
    out: list = [rng.standard_normal(np.asarray(coeffs[0]).shape)]
    for lh, hl, hh in coeffs[1:]:
        out.append(
            (
                rng.standard_normal(np.asarray(lh).shape),
                rng.standard_normal(np.asarray(hl).shape),
                rng.standard_normal(np.asarray(hh).shape),
            )
        )
    return out


def _require_pywt() -> None:
    if pywt is None:
        raise ImportError(
            "snerv_inverse_steg_carrier.dwt requires pywt (PyWavelets) for "
            "non-Haar wavelets. Use wavelet='haar' for the NumPy receiver path."
        )


def _is_numpy_haar(wavelet: str) -> bool:
    return str(wavelet).strip().lower() in {"haar", "db1"}


def _haar_wavedec2_periodized(image: np.ndarray, *, levels: int) -> list:
    current = np.asarray(image, dtype=np.float64)
    details: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for _level in range(int(levels)):
        current, detail = _haar_dwt2_level(current)
        details.append(detail)
    return [current, *reversed(details)]


def _haar_waverec2_periodized(coeffs: list) -> np.ndarray:
    current = np.asarray(coeffs[0], dtype=np.float64)
    for detail in coeffs[1:]:
        current = _haar_idwt2_level(current, detail)
    return current


def _haar_dwt2_level(
    image: np.ndarray,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] % 2 or arr.shape[1] % 2:
        raise SnervDwtError(f"Haar level requires even 2D shape; got {arr.shape}")
    a = arr[0::2, 0::2]
    b = arr[0::2, 1::2]
    c = arr[1::2, 0::2]
    d = arr[1::2, 1::2]
    ll = (a + b + c + d) * 0.5
    lh = (a + b - c - d) * 0.5
    hl = (a - b + c - d) * 0.5
    hh = (a - b - c + d) * 0.5
    return ll, (lh, hl, hh)


def _haar_idwt2_level(
    ll: np.ndarray,
    detail: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    ll = np.asarray(ll, dtype=np.float64)
    lh, hl, hh = (np.asarray(item, dtype=np.float64) for item in detail)
    if lh.shape != ll.shape or hl.shape != ll.shape or hh.shape != ll.shape:
        raise SnervDwtError("Haar detail shapes must match LL shape")
    out = np.empty((ll.shape[0] * 2, ll.shape[1] * 2), dtype=np.float64)
    out[0::2, 0::2] = (ll + lh + hl + hh) * 0.5
    out[0::2, 1::2] = (ll + lh - hl - hh) * 0.5
    out[1::2, 0::2] = (ll - lh + hl - hh) * 0.5
    out[1::2, 1::2] = (ll - lh - hl + hh) * 0.5
    return out
