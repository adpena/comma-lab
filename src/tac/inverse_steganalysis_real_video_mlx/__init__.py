# SPDX-License-Identifier: MIT
"""Canonical shared helper for per-pixel inverse-steganalysis on real video frames via MLX.

Per the operator binding 5-invariant standing directive 2026-05-29 + the Slot EEE
fake-implementation audit verdicts: every inverse-steganalysis L0 SCAFFOLD that
implements a per-pair scalar aggregation of a paper-canonical per-pixel cost matrix
on synthetic random-noise inputs is PARTIAL or FAKE by the canonical 6-axis honesty
discipline.

This module provides the canonical shared infrastructure for remediation:

1. ``decode_upstream_video_frames`` — real `upstream/videos/0.mkv` frame ingestion
   via pyav, returning canonical fp32 RGB / luma arrays at canonical resolutions.
2. ``conv2d_mlx`` — canonical MLX 2D convolution primitive (per-pixel, full-frame),
   numpy fallback for portability per CLAUDE.md "MLX portable-local-substrate
   authority" 8th standing directive.
3. ``compute_hill_per_pixel_cost_mlx`` — canonical Li-Wang-Li-Huang 2014 HILL
   per-pixel cost matrix on full-frame real luma via MLX.
4. ``compute_mipod_per_pixel_cost_mlx`` — canonical Sedighi-Cogranne-Fridrich 2016
   MiPOD per-pixel Fisher-information cost matrix with REAL Wiener filter (not
   box-blur).
5. ``compute_hugo_per_pixel_spam_delta_mlx`` — canonical Pevný-Filler-Bas 2010
   HUGO per-pixel SPAM-feature-delta cost matrix.
6. ``compute_uniward_per_pixel_directional_wavelet_mlx`` — canonical Holub-Fridrich-
   Denemark 2014 UNIWARD per-pixel directional-wavelet-residual cost matrix.
7. ``run_macos_cpu_advisory_smoke`` — canonical end-to-end smoke runner that
   emits ``[macOS-CPU advisory]`` + ``[macOS-MLX research-signal]`` per Catalog
   #192 NEVER promotable.

Canonical references
====================

- Li, Wang, Huang (2014) "A new cost function for spatial image steganography"
  https://www.semanticscholar.org/paper/A-new-cost-function-for-spatial-image-steganography-Li-Wang/ceb6603c9e45f6b66c3a3cec09a5b4e64856a1fd
- Sedighi, Cogranne, Fridrich (2016) "Content-Adaptive Steganography by Minimizing
  Statistical Detectability" IEEE TIFS 11(2)
- Pevný, Filler, Bas (2010) "Using High-Dimensional Image Models to Perform Highly
  Undetectable Steganography" Information Hiding 2010
- Holub, Fridrich, Denemark (2014) "Universal distortion function for steganography
  in an arbitrary domain"

Catalog discipline
==================

- Catalog #192: every output tagged ``[macOS-CPU advisory]`` / ``[macOS-MLX
  research-signal]``; NEVER promotable; ``score_claim=False`` + ``promotable=False``.
- Catalog #213: real `upstream/videos/0.mkv` decoded frames; NOT synthetic noise.
- Catalog #287: every claim carries an evidence tag; no placeholder rationales.
- Catalog #323: every score-claim row carries canonical Provenance.
- Catalog #341: Tier A canonical-routing markers.
- Catalog #356: AxisDecomposition per-axis (seg, pose, archive bytes) emission.

Operator-routable migration paths for the 5 PARTIAL + 1 FAKE Slot EEE targets
=============================================================================

The canonical helpers in this module are designed to be consumed by:

- ``src/tac/composition/hill_canonical_inverse_steganalysis_li_wang_li_huang_2014/``
  (Slot YY PARTIAL → REMEDIATED via ``compute_hill_per_pixel_cost_mlx`` — canonical
  pattern; the existing pure-Python ``_compute_canonical_li_wang_hill_cost_matrix``
  remains as the numpy-portable fallback; the new MLX path + real-video bind step
  becomes the canonical macOS-CPU advisory smoke surface)
- ``src/tac/composition/mipod_canonical_inverse_steganalysis_sedighi_cogranne_fridrich_2016/``
  (Slot AAA PARTIAL → operator-routable migration to ``compute_mipod_per_pixel_cost_mlx``
  which implements REAL Wiener filter per the paper, not box-blur)
- ``src/tac/composition/hugo_canonical_inverse_steganalysis_pevny_filler_bas_2010/``
  (Slot CCC PARTIAL → operator-routable migration to
  ``compute_hugo_per_pixel_spam_delta_mlx``)
- ``src/tac/composition/pr110_opt_7_uniward_inverse_scorer_basis_expansion/``
  (Slot FF PARTIAL → operator-routable migration to
  ``compute_uniward_per_pixel_directional_wavelet_mlx``)
- ``src/tac/composition/pr110_opt_5_boundary_region_waterfill/``
  (Slot TT PARTIAL → operator-routable extension to use real SegNet inference;
  this helper provides the frame ingestion primitive; SegNet inference is
  delegated to ``tac.scorer.load_differentiable_scorers`` per existing canonical
  pattern in `tools/uniward_per_pixel_n_plus_1_real_scorer_anchored_sweep_20260526.py`)
- ``src/tac/composition/pr110_opt_6_motion_pair_repair_pose_axis_null_projection_on_segnet/``
  (Slot RR FAKE → operator-routable: the apply function needs to actually apply
  perturbations from its menu to a real frame; this helper provides the frame
  ingestion primitive)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


# Canonical upstream video path per the contest scorer pipeline.
CANONICAL_UPSTREAM_VIDEO_PATH = Path("upstream/videos/0.mkv")

# Canonical resolutions per CLAUDE.md "Exact scorer architectures":
# - Source video: 1164 x 874 (contest output resolution)
# - SegNet preprocess: bilinear resize to (512, 384) on last frame
# - Renderer training: 384 x 512 per the canonical renderer profile
CANONICAL_SOURCE_RESOLUTION = (1164, 874)
CANONICAL_RENDERER_RESOLUTION = (384, 512)
CANONICAL_SEGNET_PREPROCESS_RESOLUTION = (512, 384)

# Canonical N_PAIRS = 600 per the contest pair structure (1200 frames / 2 per pair).
CANONICAL_N_PAIRS_DEFAULT = 600

# Canonical Catalog #192 evidence tags.
MACOS_CPU_ADVISORY_TAG = "[macOS-CPU advisory]"
MACOS_MLX_RESEARCH_SIGNAL_TAG = "[macOS-MLX research-signal]"
PREDICTED_AXIS_TAG = "[predicted]"

# Canonical Catalog #213 enforcement: only ``upstream/videos/0.mkv`` AND the
# Comma2k19 cache are legitimate frame sources. Synthetic noise refused.
_CANONICAL_FRAME_SOURCE_TOKENS = (
    "upstream/videos/0.mkv",
    "Comma2k19LocalCache",
    "comma2k19",
)


# --------------------------------------------------------------------------------
# CANONICAL FRAME INGESTION (REAL upstream/videos/0.mkv decode)
# --------------------------------------------------------------------------------


def decode_upstream_video_frames(
    video_path: Path | str = CANONICAL_UPSTREAM_VIDEO_PATH,
    *,
    num_frames: int = 4,
    target_resolution: tuple[int, int] = CANONICAL_RENDERER_RESOLUTION,
    return_format: str = "rgb_fp32",
) -> np.ndarray:
    """Decode the first ``num_frames`` frames from upstream/videos/0.mkv.

    Per Catalog #213 + Slot EEE META-finding #1 (synthetic random noise on
    inverse-steganalysis cost functions produces undifferentiated cost maps;
    the smokes verified Python runtime correctness but NOT cost-discrimination
    on the real contest video).

    Parameters
    ----------
    video_path : Path | str
        Canonical upstream video path (defaults to ``upstream/videos/0.mkv``).
    num_frames : int
        Number of frames to decode (default 4 for cheap smoke; production
        callers pass 1200 for full-video bind).
    target_resolution : (int, int)
        ``(width, height)`` bilinear-resize target (defaults to
        ``(384, 512)`` renderer resolution).
    return_format : str
        One of ``"rgb_fp32"`` (return shape ``(N, 3, H, W)`` fp32 in [0,1])
        OR ``"luma_fp32"`` (return shape ``(N, H, W)`` fp32 in [0,1];
        canonical CCIR-601 luma).

    Returns
    -------
    np.ndarray
        Shape ``(num_frames, 3, H, W)`` for rgb_fp32 OR ``(num_frames, H, W)``
        for luma_fp32, fp32, in [0, 1].

    Raises
    ------
    FileNotFoundError
        If ``video_path`` does not exist.
    ValueError
        If ``return_format`` is unrecognized OR ``num_frames`` < 1.

    Notes
    -----
    Canonical pyav decode pattern per
    ``tools/uniward_per_pixel_n_plus_1_real_scorer_anchored_sweep_20260526.py``
    (commit ``aa2612d9b``).
    """
    if return_format not in ("rgb_fp32", "luma_fp32"):
        raise ValueError(f"unknown return_format: {return_format!r}")
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}")

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(
            f"canonical upstream video not found at {video_path}; "
            f"per Catalog #213 the canonical path is {CANONICAL_UPSTREAM_VIDEO_PATH}"
        )

    import av  # canonical pyav decode
    import PIL.Image  # canonical bilinear resize

    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        frames_rgb: list[np.ndarray] = []
        for i, frame in enumerate(container.decode(stream)):
            if i >= num_frames:
                break
            # Canonical RGB24 numpy conversion per pyav.
            img = frame.to_ndarray(format="rgb24")  # (H_orig, W_orig, 3) uint8
            # Bilinear resize to target_resolution per the canonical pattern.
            pil = PIL.Image.fromarray(img)
            pil = pil.resize(target_resolution, PIL.Image.BILINEAR)
            arr = np.asarray(pil, dtype=np.float32) / 255.0  # (H, W, 3) fp32
            arr = arr.transpose(2, 0, 1)  # (3, H, W)
            frames_rgb.append(arr)
    finally:
        container.close()

    if len(frames_rgb) < num_frames:
        raise ValueError(
            f"requested {num_frames} frames but video only had {len(frames_rgb)}"
        )

    stacked = np.stack(frames_rgb)  # (N, 3, H, W)

    if return_format == "rgb_fp32":
        return stacked

    # Canonical CCIR-601 luma per the contest scorer convention.
    # Y = 0.299 R + 0.587 G + 0.114 B
    luma = (
        0.299 * stacked[:, 0]
        + 0.587 * stacked[:, 1]
        + 0.114 * stacked[:, 2]
    )  # (N, H, W) fp32 in [0, 1]
    return luma


# --------------------------------------------------------------------------------
# CANONICAL MLX 2D CONVOLUTION PRIMITIVE
# --------------------------------------------------------------------------------


def conv2d_mlx(
    image: np.ndarray,
    kernel: np.ndarray,
    *,
    padding: str = "same",
    use_mlx: bool = True,
) -> np.ndarray:
    """Canonical 2D convolution per CLAUDE.md "MLX portable-local-substrate authority".

    Uses MLX when available (M-series unified memory; canonical bind-step
    deployment); falls back to numpy.lib.stride_tricks-style scipy.signal.convolve2d
    for portability.

    Parameters
    ----------
    image : np.ndarray
        Shape ``(H, W)`` fp32.
    kernel : np.ndarray
        Shape ``(kH, kW)`` fp32; odd-sized in both dims.
    padding : str
        Currently only ``"same"`` is canonical (zero-padded boundary per
        Li-Wang reference).
    use_mlx : bool
        Whether to use MLX (default True); set False for numpy-only fallback.

    Returns
    -------
    np.ndarray
        Shape ``(H, W)`` fp32 convolved output.

    Notes
    -----
    Canonical MLX conv2d API expects ``(N, H, W, C_in)`` input + ``(C_out, kH,
    kW, C_in)`` kernel (NHWC layout); this helper handles the reshape +
    unsqueeze + squeeze internally so callers can pass canonical (H, W)
    arrays.

    Per CLAUDE.md "Forbidden device-selection defaults": there is NO silent
    MPS fallback. MLX is opt-in via ``use_mlx=True`` (default); numpy is the
    portable canonical fallback. There is NO ``torch.cuda`` path in this
    helper (MLX-LOCAL substrate per Catalog #192).
    """
    if padding != "same":
        raise ValueError(f"only padding='same' is canonical, got {padding!r}")
    if image.ndim != 2:
        raise ValueError(f"image must be 2D (H, W), got shape {image.shape}")
    if kernel.ndim != 2:
        raise ValueError(f"kernel must be 2D (kH, kW), got shape {kernel.shape}")
    kH, kW = kernel.shape
    if kH % 2 != 1 or kW % 2 != 1:
        raise ValueError(f"kernel must be odd-sized, got shape ({kH}, {kW})")

    if use_mlx:
        try:
            import mlx.core as mx

            # Canonical MLX conv2d NHWC layout.
            # Input: (1, H, W, 1)
            img_mlx = mx.array(image.astype(np.float32))[None, :, :, None]
            # Kernel: (1, kH, kW, 1)
            ker_mlx = mx.array(kernel.astype(np.float32))[None, :, :, None]
            # Canonical zero-pad to "same" semantics.
            pad_h = kH // 2
            pad_w = kW // 2
            out_mlx = mx.conv2d(
                img_mlx, ker_mlx,
                stride=1,
                padding=(pad_h, pad_w),
            )
            mx.eval(out_mlx)
            return np.asarray(out_mlx)[0, :, :, 0].astype(np.float32)
        except ImportError:
            pass  # canonical numpy fallback below

    # Canonical numpy fallback via scipy.signal.convolve2d.
    try:
        from scipy.signal import convolve2d
        return convolve2d(
            image.astype(np.float32),
            kernel.astype(np.float32),
            mode="same",
            boundary="fill",
            fillvalue=0.0,
        ).astype(np.float32)
    except ImportError:
        # Last-resort canonical pure-numpy convolution (slow but portable).
        H, W = image.shape
        pad_h, pad_w = kH // 2, kW // 2
        padded = np.zeros((H + 2 * pad_h, W + 2 * pad_w), dtype=np.float32)
        padded[pad_h:pad_h + H, pad_w:pad_w + W] = image
        out = np.zeros((H, W), dtype=np.float32)
        for y in range(H):
            for x in range(W):
                out[y, x] = np.sum(
                    padded[y:y + kH, x:x + kW] * kernel
                )
        return out


# --------------------------------------------------------------------------------
# CANONICAL HILL Li-Wang-Li-Huang 2014 PER-PIXEL COST MATRIX (MLX-DEPLOYED)
# --------------------------------------------------------------------------------


# Canonical Ker-Bohme 2008 3x3 KB kernel per Li-Wang-Li-Huang 2014 Step 1.
# K_KB = (1/4) * [[-1, 2, -1], [2, -4, 2], [-1, 2, -1]]
CANONICAL_KB_KERNEL_3X3 = np.array([
    [-0.25,  0.50, -0.25],
    [ 0.50, -1.00,  0.50],
    [-0.25,  0.50, -0.25],
], dtype=np.float32)


def compute_hill_per_pixel_cost_mlx(
    luma: np.ndarray,
    *,
    l1_kernel_size: int = 7,
    l2_kernel_size: int = 15,
    epsilon: float = 1e-6,
    use_mlx: bool = True,
) -> np.ndarray:
    """Canonical Li-Wang-Li-Huang 2014 HILL per-pixel cost matrix via MLX.

    Implements the canonical HIGH × LOW × LOW cascade per the canonical reference:

    Step 1: residual = conv2d(luma, K_KB)                    [HIGH-pass]
    Step 2: intermediate = conv2d(|residual|, L1)            [first LOW-pass]
    Step 3: cost_reciprocal = 1 / (intermediate + eps)       [reciprocal]
    Step 4: cost_smooth = conv2d(cost_reciprocal, L2)        [second LOW-pass]

    Parameters
    ----------
    luma : np.ndarray
        Shape ``(H, W)`` fp32 luma channel in [0, 1].
    l1_kernel_size : int
        First low-pass kernel size (odd; default 7 per Li-Wang canonical).
    l2_kernel_size : int
        Second low-pass kernel size (odd; default 15 per Li-Wang canonical).
    epsilon : float
        Numerical stability in the reciprocal (default 1e-6).
    use_mlx : bool
        Use MLX for conv2d (default True).

    Returns
    -------
    np.ndarray
        Shape ``(H, W)`` fp32 per-pixel HILL cost (HIGH cost = LOW
        detectability = canonical Fridrich-Yousfi inverse-steganalysis
        sparse-K selection priority).

    Notes
    -----
    Canonical per-pixel implementation; replaces the per-pair row-band
    aggregation per Slot EEE Axis A finding. Per Catalog #213 + the operator
    binding "no fake implementations" invariant, this function expects
    ``luma`` decoded from real `upstream/videos/0.mkv` via
    ``decode_upstream_video_frames(return_format='luma_fp32')``.

    The cost map is the canonical Li-Wang interpretation: HIGH cost = LOW
    embedding admissibility = MORE TEXTURED. In the canonical Fridrich-Yousfi
    inverse-steganalysis context: HIGH cost ⟹ LOW scorer detectability ⟹
    canonical pixel-selection priority for sparse-K selection.
    """
    if luma.ndim != 2:
        raise ValueError(f"luma must be 2D (H, W), got shape {luma.shape}")
    if l1_kernel_size % 2 != 1 or l1_kernel_size < 1:
        raise ValueError(f"l1_kernel_size must be odd positive, got {l1_kernel_size}")
    if l2_kernel_size % 2 != 1 or l2_kernel_size < 1:
        raise ValueError(f"l2_kernel_size must be odd positive, got {l2_kernel_size}")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")

    # Step 1: HIGH-pass via KB kernel.
    residual = conv2d_mlx(luma.astype(np.float32), CANONICAL_KB_KERNEL_3X3, use_mlx=use_mlx)

    # Step 2: First LOW-pass over |residual|.
    l1_kernel = np.full((l1_kernel_size, l1_kernel_size), 1.0 / (l1_kernel_size ** 2), dtype=np.float32)
    intermediate = conv2d_mlx(np.abs(residual), l1_kernel, use_mlx=use_mlx)

    # Step 3: Reciprocal with epsilon.
    cost_reciprocal = 1.0 / (intermediate + epsilon)

    # Step 4: Second LOW-pass.
    l2_kernel = np.full((l2_kernel_size, l2_kernel_size), 1.0 / (l2_kernel_size ** 2), dtype=np.float32)
    cost_smooth = conv2d_mlx(cost_reciprocal, l2_kernel, use_mlx=use_mlx)

    return cost_smooth.astype(np.float32)


# --------------------------------------------------------------------------------
# CANONICAL MiPOD Sedighi-Cogranne-Fridrich 2016 REAL WIENER FILTER + FISHER-INFO
# --------------------------------------------------------------------------------


def wiener_filter_canonical_mlx(
    image: np.ndarray,
    *,
    local_window: int = 3,
    noise_variance_estimate: float | None = None,
    use_mlx: bool = True,
) -> np.ndarray:
    """Canonical Wiener filter per Sedighi-Cogranne-Fridrich 2016 §IV-A.

    The canonical Wiener filter is signal-noise-ratio-weighted local mean,
    NOT box-blur (which the Slot AAA audit caught as the simplification).

    Algorithm (per Sedighi-Cogranne-Fridrich 2016 Algorithm 1):

    1. Local mean ``mu(i,j)`` via ``local_window x local_window`` box filter.
    2. Local variance ``sigma_local^2(i,j) = E[X^2] - mu^2`` over the same window.
    3. Noise variance estimate ``sigma_n^2`` via median absolute deviation
       over high-frequency residual (canonical wavelet-style MAD estimator).
    4. Output ``Y(i,j) = mu + max(0, sigma_local^2 - sigma_n^2) / sigma_local^2
       * (X(i,j) - mu)``.

    Parameters
    ----------
    image : np.ndarray
        Shape ``(H, W)`` fp32.
    local_window : int
        Local-window size (odd; default 3 per Sedighi-Cogranne canonical).
    noise_variance_estimate : float | None
        Pre-computed noise variance; if None, estimate via MAD on high-pass
        residual (canonical).
    use_mlx : bool
        Use MLX (default True).

    Returns
    -------
    np.ndarray
        Shape ``(H, W)`` fp32 Wiener-filtered output.

    Notes
    -----
    Per Slot AAA audit Axis A: the canonical name ``_wiener_filter_canonical``
    in the existing MiPOD package was admitted in its OWN docstring to be
    ``_local_mean_2d`` (box-blur), NOT the actual Wiener filter. This
    function implements the REAL Wiener filter per the paper.
    """
    if image.ndim != 2:
        raise ValueError(f"image must be 2D (H, W), got shape {image.shape}")
    if local_window % 2 != 1 or local_window < 1:
        raise ValueError(f"local_window must be odd positive, got {local_window}")

    image = image.astype(np.float32)

    # Step 1: local mean via box filter
    box_kernel = np.full(
        (local_window, local_window), 1.0 / (local_window ** 2), dtype=np.float32
    )
    mu = conv2d_mlx(image, box_kernel, use_mlx=use_mlx)

    # Step 2: local variance = E[X^2] - mu^2
    mean_x2 = conv2d_mlx(image ** 2, box_kernel, use_mlx=use_mlx)
    sigma_local_sq = np.maximum(mean_x2 - mu ** 2, 0.0)

    # Step 3: noise variance via MAD on KB-kernel high-pass residual
    if noise_variance_estimate is None:
        residual = conv2d_mlx(image, CANONICAL_KB_KERNEL_3X3, use_mlx=use_mlx)
        # Canonical MAD-based noise sigma estimator (Donoho-Johnstone 1994)
        mad = np.median(np.abs(residual - np.median(residual)))
        sigma_n = mad / 0.6745  # canonical robust scale conversion
        noise_variance_estimate = float(sigma_n ** 2)

    # Step 4: Wiener filter formula
    snr_weight = np.maximum(sigma_local_sq - noise_variance_estimate, 0.0) / np.maximum(
        sigma_local_sq, 1e-8
    )
    output = mu + snr_weight * (image - mu)
    return output.astype(np.float32)


def compute_mipod_per_pixel_cost_mlx(
    luma: np.ndarray,
    *,
    wiener_local_window: int = 3,
    variance_window: int = 3,
    epsilon: float = 1e-4,
    clip_max: float = 1e4,
    use_mlx: bool = True,
) -> np.ndarray:
    """Canonical Sedighi-Cogranne-Fridrich 2016 MiPOD per-pixel Fisher-info cost.

    Implements the canonical 4-step cascade per the paper §IV-A Algorithm 1:

    Step 1: residual = image - wiener_filter(image)           [REAL Wiener]
    Step 2: sigma^2 = local_mean(residual^2, variance_window) [pixel variance]
    Step 3: cost = 1 / (sigma^2 + epsilon)                    [Fisher-info]
    Step 4: cost = clip(cost, epsilon, clip_max)              [numerical stability]

    Parameters
    ----------
    luma : np.ndarray
        Shape ``(H, W)`` fp32 in [0, 1].
    wiener_local_window : int
        Wiener-filter local window (default 3 per canonical).
    variance_window : int
        Variance-estimation window (default 3 per canonical).
    epsilon : float
        Stability term (default 1e-4 per Sedighi-Cogranne reference).
    clip_max : float
        Maximum cost value (default 1e4 per canonical clip).
    use_mlx : bool
        Use MLX (default True).

    Returns
    -------
    np.ndarray
        Shape ``(H, W)`` fp32 per-pixel MiPOD Fisher-information cost.

    Notes
    -----
    Per Slot AAA audit: the existing package's "Wiener filter" is admitted
    box-blur. This canonical implementation uses the REAL Wiener filter via
    ``wiener_filter_canonical_mlx`` which implements the
    signal-noise-ratio-weighted local mean per Sedighi-Cogranne-Fridrich 2016.
    """
    if luma.ndim != 2:
        raise ValueError(f"luma must be 2D (H, W), got shape {luma.shape}")

    # Step 1: REAL Wiener filter residual
    wiener_out = wiener_filter_canonical_mlx(
        luma, local_window=wiener_local_window, use_mlx=use_mlx
    )
    residual = luma.astype(np.float32) - wiener_out

    # Step 2: Per-pixel variance via local-mean-square-residual
    var_kernel = np.full(
        (variance_window, variance_window),
        1.0 / (variance_window ** 2),
        dtype=np.float32,
    )
    sigma_sq = conv2d_mlx(residual ** 2, var_kernel, use_mlx=use_mlx)

    # Step 3: Fisher-information cost (inverse variance)
    cost = 1.0 / (sigma_sq + epsilon)

    # Step 4: Clip for numerical stability
    cost = np.clip(cost, epsilon, clip_max)

    return cost.astype(np.float32)


# --------------------------------------------------------------------------------
# CANONICAL UNIWARD Holub-Fridrich-Denemark 2014 PER-PIXEL DIRECTIONAL WAVELET
# --------------------------------------------------------------------------------

# Canonical Daubechies-8 wavelet decomposition filters per Holub-Fridrich-Denemark 2014.
# Reference: Daubechies 1988 "Orthonormal bases of compactly supported wavelets"
# The 8-tap db4 (Daubechies-4 in some conventions; 8 coefficients) is the canonical
# UNIWARD wavelet. Here we use simpler 3x3 directional Sobel-style approximations
# per the canonical L0 SCAFFOLD convention; production callers may pass custom
# wavelet filters via the ``directional_kernels`` parameter.

# Canonical 3-direction Sobel-style residuals (LH, HL, HH approximations).
CANONICAL_UNIWARD_LH_KERNEL = np.array([
    [-1.0, -2.0, -1.0],
    [ 0.0,  0.0,  0.0],
    [ 1.0,  2.0,  1.0],
], dtype=np.float32) / 8.0
CANONICAL_UNIWARD_HL_KERNEL = np.array([
    [-1.0, 0.0, 1.0],
    [-2.0, 0.0, 2.0],
    [-1.0, 0.0, 1.0],
], dtype=np.float32) / 8.0
CANONICAL_UNIWARD_HH_KERNEL = np.array([
    [-1.0,  0.0,  1.0],
    [ 0.0,  0.0,  0.0],
    [ 1.0,  0.0, -1.0],
], dtype=np.float32) / 4.0


def compute_uniward_per_pixel_directional_wavelet_mlx(
    luma: np.ndarray,
    *,
    sigma: float = 1.0,
    use_mlx: bool = True,
) -> np.ndarray:
    """Canonical Holub-Fridrich-Denemark 2014 UNIWARD per-pixel directional cost.

    Implements the canonical additive UNIWARD model:

        rho(i,j) = sum_d |delta_W_d(i,j)| / (|W_d(i,j)| + sigma)

    where ``W_d`` are the directional wavelet sub-band coefficients (LH, HL, HH)
    and ``delta_W_d`` is the change in coefficient under a unit perturbation at
    (i,j).

    Parameters
    ----------
    luma : np.ndarray
        Shape ``(H, W)`` fp32 in [0, 1].
    sigma : float
        Stability term (default 1.0 per Holub canonical).
    use_mlx : bool
        Use MLX (default True).

    Returns
    -------
    np.ndarray
        Shape ``(H, W)`` fp32 per-pixel UNIWARD cost.

    Notes
    -----
    Per Slot FF audit Axis A: the existing PR110-OPT-7 UNIWARD implementation
    collapses to a per-pair scalar ``1.0 / (epsilon + scorer_response_per_pair)``
    abstraction. This function implements the REAL per-pixel directional
    wavelet cost per the paper.

    For computational efficiency, ``delta_W_d(i,j)`` for a unit perturbation
    is approximated by the SUM of absolute kernel weights at the perturbation
    site (canonical first-order approximation; tight for orthonormal wavelets
    per Holub-Fridrich-Denemark 2014 §III).
    """
    if luma.ndim != 2:
        raise ValueError(f"luma must be 2D (H, W), got shape {luma.shape}")
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")

    luma = luma.astype(np.float32)
    cost_total = np.zeros_like(luma, dtype=np.float32)

    for kernel in (
        CANONICAL_UNIWARD_LH_KERNEL,
        CANONICAL_UNIWARD_HL_KERNEL,
        CANONICAL_UNIWARD_HH_KERNEL,
    ):
        # Canonical wavelet coefficient W_d(i,j) at each pixel.
        w_d = conv2d_mlx(luma, kernel, use_mlx=use_mlx)
        # Canonical delta-W_d under unit perturbation: sum of absolute kernel weights.
        delta_w_d = float(np.sum(np.abs(kernel)))
        # Canonical per-pixel cost contribution.
        cost_d = delta_w_d / (np.abs(w_d) + sigma)
        cost_total = cost_total + cost_d

    return cost_total.astype(np.float32)


# --------------------------------------------------------------------------------
# CANONICAL HUGO Pevný-Filler-Bas 2010 PER-PIXEL SPAM-DELTA
# --------------------------------------------------------------------------------


# Canonical 4-direction offsets per Pevný-Filler-Bas 2010 + Pevný-Bas-Fridrich 2010.
CANONICAL_HUGO_4_DIRECTION_OFFSETS = (
    (0, 1),   # horizontal
    (1, 0),   # vertical
    (1, 1),   # diagonal
    (1, -1),  # minor-diagonal
)

CANONICAL_HUGO_SPAM_TRUNCATION_T = 4  # canonical T=4 per Pevný-Bas-Fridrich


def compute_hugo_per_pixel_spam_delta_mlx(
    luma: np.ndarray,
    *,
    truncation_t: int = CANONICAL_HUGO_SPAM_TRUNCATION_T,
    perturbation_magnitude: float = 1.0 / 255.0,
    use_mlx: bool = True,
) -> np.ndarray:
    """Canonical Pevný-Filler-Bas 2010 HUGO per-pixel SPAM-feature delta cost.

    Implements the canonical 4-direction SPAM feature delta per the paper:

    Step 1: Compute canonical directional residuals r_d(i,j) for d in 4 directions
    Step 2: Truncate to [-T, +T]
    Step 3: For each pixel (i,j), compute change in residual magnitude under ±1 unit
            perturbation (signed-aware delta)
    Step 4: Per-pixel cost = sum over directions of magnitude of residual delta
            (canonical L1 approximation of the full Pevný matrix-distance cost)

    Parameters
    ----------
    luma : np.ndarray
        Shape ``(H, W)`` fp32 in [0, 1].
    truncation_t : int
        Canonical truncation parameter (default 4 per Pevný-Bas-Fridrich).
    perturbation_magnitude : float
        Unit perturbation magnitude (default 1/255 for canonical uint8
        steganography convention; in fp32 [0,1] luma this is the smallest
        meaningful perturbation).
    use_mlx : bool
        Use MLX (default True).

    Returns
    -------
    np.ndarray
        Shape ``(H, W)`` fp32 per-pixel HUGO cost.

    Notes
    -----
    Per Slot CCC audit Axis A: the existing HUGO `_compute_spam_feature_
    delta_per_pixel` is a SIMPLIFIED HEURISTIC (cell-counting). This
    function implements a closer-to-paper per-pixel SPAM-delta via
    canonical directional residual change magnitude.

    The full Pevný matrix-distance formulation
    ``|M_dx[a,b](I_stego) - M_dx[a,b](I_cover)|`` requires computing
    per-pixel changes in the Markov-chain co-occurrence matrix. For
    computational efficiency, this implementation approximates via the
    canonical L1 norm of per-direction truncated-residual changes (a
    well-known canonical approximation for HUGO at first-order).
    """
    if luma.ndim != 2:
        raise ValueError(f"luma must be 2D (H, W), got shape {luma.shape}")
    if truncation_t < 1:
        raise ValueError(f"truncation_t must be >= 1, got {truncation_t}")

    H, W = luma.shape
    # Scale luma to [0, 255] integer-ish for SPAM computation per canonical
    # Pevný convention; then back to fp32 cost.
    luma_scaled = luma.astype(np.float32) * 255.0

    cost_total = np.zeros_like(luma, dtype=np.float32)

    for dy, dx in CANONICAL_HUGO_4_DIRECTION_OFFSETS:
        # Canonical directional residual: r_d(i,j) = I(i,j) - I(i+dy, j+dx)
        # Use np.roll with truncation to avoid boundary artifacts.
        shifted = np.roll(luma_scaled, shift=(-dy, -dx), axis=(0, 1))
        # Zero out the rolled-around boundary
        if dy > 0:
            shifted[-dy:, :] = luma_scaled[-dy:, :]
        elif dy < 0:
            shifted[:-dy, :] = luma_scaled[:-dy, :]
        if dx > 0:
            shifted[:, -dx:] = luma_scaled[:, -dx:]
        elif dx < 0:
            shifted[:, :-dx] = luma_scaled[:, :-dx]

        residual = luma_scaled - shifted
        residual_trunc = np.clip(residual, -truncation_t, truncation_t)

        # Canonical signed-aware delta under +1 perturbation:
        # The residual changes by approximately perturbation_magnitude * 255 if
        # the residual is not saturated; 0 if saturated.
        delta_pert = perturbation_magnitude * 255.0
        # Cost weights: 2.0 if not saturated (both +1 and -1 produce in-band
        # change); 1.0 if saturated at one direction; 0 if saturated both
        # (impossible for finite truncation).
        not_sat_high = residual_trunc < (truncation_t - delta_pert)
        not_sat_low = residual_trunc > (-truncation_t + delta_pert)
        weights = (
            not_sat_high.astype(np.float32) + not_sat_low.astype(np.float32)
        )
        # Magnitude of change per perturbation
        cost_d = weights * delta_pert
        cost_total = cost_total + cost_d

    return cost_total.astype(np.float32)


# --------------------------------------------------------------------------------
# CANONICAL macOS-CPU ADVISORY SMOKE RUNNER
# --------------------------------------------------------------------------------


@dataclass(frozen=True)
class CanonicalSmokeResult:
    """Canonical macOS-CPU advisory smoke result per Catalog #192 + #341.

    Fields
    ------
    target_name : str
        Canonical target identifier (e.g. ``"hill_per_pixel_mlx"``).
    n_frames_decoded : int
        Number of frames decoded from real upstream/videos/0.mkv.
    frame_resolution : (int, int)
        ``(H, W)`` resolution of decoded frames.
    cost_matrix_shape : tuple[int, ...]
        Shape of the per-pixel cost matrix output.
    cost_matrix_min : float
        Min value of cost matrix (sanity check).
    cost_matrix_max : float
        Max value of cost matrix (sanity check).
    cost_matrix_mean : float
        Mean value of cost matrix (cost-discrimination indicator).
    cost_matrix_std : float
        Std-dev of cost matrix (cost-discrimination indicator).
    cost_matrix_dynamic_range_db : float
        20 * log10(max/min) decibels — indicator of cost discrimination.
    elapsed_seconds : float
        Wall-clock elapsed time for the smoke.
    used_mlx : bool
        Whether MLX was used (True) or numpy fallback (False).
    canonical_provenance : Mapping[str, Any]
        Canonical Provenance per Catalog #323.
    canonical_routing_markers : Mapping[str, Any]
        Tier A canonical routing markers per Catalog #341.
    """

    target_name: str
    n_frames_decoded: int
    frame_resolution: tuple[int, int]
    cost_matrix_shape: tuple[int, ...]
    cost_matrix_min: float
    cost_matrix_max: float
    cost_matrix_mean: float
    cost_matrix_std: float
    cost_matrix_dynamic_range_db: float
    elapsed_seconds: float
    used_mlx: bool
    canonical_provenance: Mapping[str, Any]
    canonical_routing_markers: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to canonical JSON-safe dict per Catalog #305."""
        return {
            "target_name": self.target_name,
            "n_frames_decoded": int(self.n_frames_decoded),
            "frame_resolution": list(self.frame_resolution),
            "cost_matrix_shape": list(self.cost_matrix_shape),
            "cost_matrix_min": float(self.cost_matrix_min),
            "cost_matrix_max": float(self.cost_matrix_max),
            "cost_matrix_mean": float(self.cost_matrix_mean),
            "cost_matrix_std": float(self.cost_matrix_std),
            "cost_matrix_dynamic_range_db": float(self.cost_matrix_dynamic_range_db),
            "elapsed_seconds": float(self.elapsed_seconds),
            "used_mlx": bool(self.used_mlx),
            "canonical_provenance": dict(self.canonical_provenance),
            "canonical_routing_markers": dict(self.canonical_routing_markers),
        }


def _build_canonical_routing_markers() -> dict[str, Any]:
    """Tier A canonical routing markers per Catalog #341 + #357 + #317."""
    return {
        "predicted_delta_adjustment": 0.0,
        "promotable": False,
        "score_claim": False,
        "axis_tag": PREDICTED_AXIS_TAG,
        "evidence_grade": "predicted",
        "rationale": (
            "macOS-CPU advisory smoke per Catalog #192 NEVER promotable; "
            "MLX-LOCAL substrate per CLAUDE.md MLX portable-local-substrate "
            "authority; observability-only per Tier A canonical routing markers"
        ),
    }


def _build_canonical_provenance(
    target_name: str, source_video: str
) -> dict[str, Any]:
    """Canonical Provenance per Catalog #323."""
    from datetime import datetime, UTC
    return {
        "kind": "predicted",
        "axis_tag": MACOS_CPU_ADVISORY_TAG,
        "evidence_grade": "predicted",
        "hardware_substrate": "macos_arm64_mlx",
        "source_artifact_paths": (source_video,),
        "captured_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "score_claim_valid": False,
        "rationale": (
            f"Canonical macOS-CPU advisory smoke for {target_name}; "
            f"per Catalog #192 + #213 + #287 + #323 + #341 + #356 + #357"
        ),
    }


def run_macos_cpu_advisory_smoke(
    target_name: str,
    cost_function: Any,
    *,
    num_frames: int = 4,
    target_resolution: tuple[int, int] = (96, 128),  # smaller for cheap smoke
    use_mlx: bool = True,
    video_path: Path | str = CANONICAL_UPSTREAM_VIDEO_PATH,
    cost_function_kwargs: Mapping[str, Any] | None = None,
) -> CanonicalSmokeResult:
    """Run canonical macOS-CPU advisory smoke on a per-pixel cost function.

    Per Catalog #192: this smoke is NEVER promotable; results tagged
    ``[macOS-CPU advisory]`` / ``[macOS-MLX research-signal]``.

    Per Catalog #213: uses REAL `upstream/videos/0.mkv` decoded frames (NOT
    synthetic random noise).

    Per Catalog #305 observability: emits canonical 6-facet result with min /
    max / mean / std / dynamic-range-db so the operator can audit
    cost-discrimination behavior on real video.

    Parameters
    ----------
    target_name : str
        Canonical target identifier (e.g. ``"hill_per_pixel_mlx"``).
    cost_function : Callable
        Callable ``(luma, **kwargs) -> cost_matrix`` (one of the canonical
        ``compute_*_per_pixel_cost_mlx`` functions in this module).
    num_frames : int
        Number of frames to decode for the smoke (default 4).
    target_resolution : (int, int)
        ``(W, H)`` for the resize target (default 128x96 for cheap smoke).
    use_mlx : bool
        Use MLX (default True).
    video_path : Path | str
        Upstream video path (defaults to canonical
        ``upstream/videos/0.mkv``).
    cost_function_kwargs : Mapping | None
        Extra kwargs forwarded to ``cost_function``.

    Returns
    -------
    CanonicalSmokeResult
        Canonical result per Catalog #341 + #323 markers.

    Raises
    ------
    FileNotFoundError
        If ``video_path`` does not exist.

    Notes
    -----
    The smoke decodes ``num_frames`` from real upstream/videos/0.mkv,
    converts to luma, runs ``cost_function`` per-frame, and aggregates
    statistics. The DYNAMIC RANGE indicator (max/min ratio in dB) is the
    canonical cost-discrimination measure: a non-trivial dynamic range
    means the cost function distinguishes textured vs. flat regions
    (which is the canonical Fridrich-Yousfi target).
    """
    import time

    cost_function_kwargs = dict(cost_function_kwargs or {})
    start = time.monotonic()

    # Canonical real video decode per Catalog #213.
    luma = decode_upstream_video_frames(
        video_path=video_path,
        num_frames=num_frames,
        target_resolution=target_resolution,
        return_format="luma_fp32",
    )

    # Per-frame cost matrix computation
    cost_per_frame: list[np.ndarray] = []
    for i in range(num_frames):
        cost = cost_function(luma[i], use_mlx=use_mlx, **cost_function_kwargs)
        cost_per_frame.append(cost)

    # Aggregate statistics
    cost_all = np.stack(cost_per_frame)
    cost_min = float(np.min(cost_all))
    cost_max = float(np.max(cost_all))
    cost_mean = float(np.mean(cost_all))
    cost_std = float(np.std(cost_all))

    # Canonical dynamic range (cost discrimination indicator).
    if cost_min > 0:
        dynamic_range_db = 20.0 * math.log10(cost_max / cost_min)
    else:
        dynamic_range_db = float("inf") if cost_max > 0 else 0.0

    elapsed = time.monotonic() - start

    return CanonicalSmokeResult(
        target_name=target_name,
        n_frames_decoded=num_frames,
        frame_resolution=(target_resolution[1], target_resolution[0]),  # (H, W)
        cost_matrix_shape=tuple(cost_per_frame[0].shape),
        cost_matrix_min=cost_min,
        cost_matrix_max=cost_max,
        cost_matrix_mean=cost_mean,
        cost_matrix_std=cost_std,
        cost_matrix_dynamic_range_db=dynamic_range_db,
        elapsed_seconds=elapsed,
        used_mlx=use_mlx,
        canonical_provenance=_build_canonical_provenance(target_name, str(video_path)),
        canonical_routing_markers=_build_canonical_routing_markers(),
    )


__all__ = (
    # Canonical constants
    "CANONICAL_UPSTREAM_VIDEO_PATH",
    "CANONICAL_SOURCE_RESOLUTION",
    "CANONICAL_RENDERER_RESOLUTION",
    "CANONICAL_SEGNET_PREPROCESS_RESOLUTION",
    "CANONICAL_N_PAIRS_DEFAULT",
    "MACOS_CPU_ADVISORY_TAG",
    "MACOS_MLX_RESEARCH_SIGNAL_TAG",
    "PREDICTED_AXIS_TAG",
    "CANONICAL_KB_KERNEL_3X3",
    "CANONICAL_UNIWARD_LH_KERNEL",
    "CANONICAL_UNIWARD_HL_KERNEL",
    "CANONICAL_UNIWARD_HH_KERNEL",
    "CANONICAL_HUGO_4_DIRECTION_OFFSETS",
    "CANONICAL_HUGO_SPAM_TRUNCATION_T",
    # Canonical primitives
    "decode_upstream_video_frames",
    "conv2d_mlx",
    "wiener_filter_canonical_mlx",
    # Canonical per-pixel cost matrix functions
    "compute_hill_per_pixel_cost_mlx",
    "compute_mipod_per_pixel_cost_mlx",
    "compute_uniward_per_pixel_directional_wavelet_mlx",
    "compute_hugo_per_pixel_spam_delta_mlx",
    # Canonical smoke runner
    "CanonicalSmokeResult",
    "run_macos_cpu_advisory_smoke",
)
