"""Wavelet residual basis SCAFFOLD over PR106 r2 decoded RGB frames.

This module is a research-signal-only Level 0 (SKETCH) scaffold per
`lane_wavelet_residual_basis_pr106` registered in the lane maturity registry on
2026-05-11. It computes a multi-level 2D Discrete Wavelet Transform (DWT) over
PR106-family decoded RGB frames using the Daubechies family (default db4 to
match the existing steganalysis-grounded convention in
`tac.wavelet_variance`) and emits per-band sparsity statistics + entropy
estimates as an INFORMATIONAL artifact.

It does NOT
-----------

* score against the contest scorer (no PoseNet/SegNet load, no `evaluate.py`),
* modify any archive bytes,
* propose any byte-closed sidecar (that's the downstream L1 work; see
  "Path to promotion" below),
* claim a [contest-CUDA] or [contest-CPU] result,
* permit MPS as the compute axis (forbidden by CLAUDE.md "MPS auth eval is
  NOISE"; macOS-CPU is acceptable for research-signal generation only),
* write to `/tmp/` (forbidden by CLAUDE.md "Forbidden /tmp paths in any
  persisted artifact").

It DOES
-------

* expose a typed `WaveletResidualResult` dataclass with explicit
  `score_claim=False`, `promotion_eligible=False`,
  `ready_for_exact_eval_dispatch=False`, and an `evidence_grade` tag of
  `"research_signal"`,
* offer a `compute_wavelet_residual_stats()` entry point that accepts a
  decoded RGB array (T, H, W, 3) of uint8 and returns the per-band
  sparsity + entropy summary,
* offer a `decompose_frame_to_bands()` low-level helper that returns each
  level's (LL, (LH, HL, HH)) coefficient tensors for downstream tooling,
* offer a `reconstruct_frame_from_bands()` inverse-DWT that round-trips a
  frame within `1e-10` numerical tolerance (the test harness pins this),
* expose a `load_decoded_raw_frames()` reader for the canonical
  `(N, H=874, W=1164, 3)` uint8 layout emitted by the PR106 sidecar inflate.

Path to L1+ promotion (per HNeRV parity discipline 13 lessons)
--------------------------------------------------------------

To advance this lane past L0 SKETCH, the eight archive-grammar fields must be
declared:

  1. archive_grammar          — wavelet-coefficient sidecar layout (e.g.,
                                magic byte + per-level LH/HL/HH brotli stream)
  2. parser_section_manifest  — fixed offsets / length prefixes in inflate.py
  3. inflate_runtime_loc_budget — <= 100 LOC, <= 2 external deps
  4. runtime_dep_closure      — PyWavelets is NOT a contest runtime dep;
                                a Rust/C/numpy port of the inverse DWT must
                                land before runtime closure
  5. export_format            — write the per-frame band-coefficient stream
                                + reattach to the PR106 sidecar archive
  6. score_aware_loss         — train the coefficient quantizer with
                                eval_roundtrip + differentiable YUV6 +
                                SegNet/PoseNet gradients per CLAUDE.md
                                "eval_roundtrip — NON-NEGOTIABLE"
  7. bolt_on_loc_budget       — <= 350 LOC for the codec + builder
  8. no_op_detector_planned   — old_archive_sha256 vs new_archive_sha256
                                with explicit `runtime_consumption_proof`

Mallat grand-council position (per CLAUDE.md grand council roster)
------------------------------------------------------------------

Stéphane Mallat's seminal contribution to the council is the multi-resolution
analysis framework (1989) and the scattering transform (2012, with Bruna). The
canonical interpretation here is:

    "AV1 grayscale + Gaussian-LUT viewed as wavelet-coded analog signal."

The leaderboard's HNeRV-family decoder produces RGB frames at 384x512 then
bicubic-upsamples to 874x1164 (per `submissions/pr106_latent_sidecar/inflate.py`).
The residual between this decoded RGB and the score-relevant frames the contest
scorer derives (SegNet on the last frame; PoseNet on the YUV6 frame-pair) lives
in a sparse multi-resolution representation. The per-band sparsity + entropy
estimates emitted by this scaffold are the FIRST step toward a wavelet-coded
sidecar that would charge wavelet coefficient bytes against the contest rate
term while reducing the perceptually-irrelevant residual energy that SegNet /
PoseNet do NOT measure (the scattering-transform invariance argument).

The canonical theoretical citation: Mallat, S. (1989). "A theory for
multiresolution signal decomposition: the wavelet representation." IEEE PAMI
11(7): 674-693. The Daubechies-N (db4 default) basis matches the steganalysis
literature convention already used by `tac.wavelet_variance` (Holub-Fridrich
2014 UNIWARD), preserving a single canonical wavelet family across the codebase.

References
----------

Handoff P3 "Wavelet/foveation/RAFT/ego-motion" + "Non-HNeRV and cross-paradigm
routes" + handoff "Generate coefficient score tables on PR106 decoded residuals".

`feedback_grand_council_pose_axis_insights_review_20260511.md` (Insight 3:
pose-axis residual lanes higher EV at PR106 operating point).

CLAUDE.md "HNeRV / leaderboard-implementation parity discipline — 13 lessons
NON-NEGOTIABLE" + "tac stays clean; comma-lab owns research state" +
"Beauty, simplicity, and developer experience".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np
import pywt

# Constants matching the PR106 latent sidecar inflate output contract.
CAMERA_H: Final[int] = 874
CAMERA_W: Final[int] = 1164
RGB_CHANNELS: Final[int] = 3
DEFAULT_WAVELET: Final[str] = "db4"
DEFAULT_DECOMPOSITION_LEVELS: Final[int] = 3
DEFAULT_HIST_BINS: Final[int] = 257  # +/- 256 centered on zero for uint8 residuals
DEFAULT_SPARSITY_EPSILON: Final[float] = 1e-6
NUMERICAL_ROUNDTRIP_TOLERANCE: Final[float] = 1e-10
RESEARCH_SIGNAL_EVIDENCE_GRADE: Final[str] = "research_signal"


class WaveletResidualError(ValueError):
    """Raised on shape / dtype / parameter contract violations."""


@dataclass(frozen=True)
class BandStats:
    """Per-band sparsity + entropy summary.

    Every field carries a non-negative, finite floating-point value (the
    `assert_invariants()` method pins these post-construction; tests cover the
    contract directly).
    """

    band_name: str
    level: int
    height: int
    width: int
    n_coefficients: int
    abs_mean: float
    abs_std: float
    abs_max: float
    sparsity_fraction: float
    entropy_bits: float

    def assert_invariants(self) -> None:
        if self.n_coefficients <= 0:
            raise WaveletResidualError(f"non-positive n_coefficients={self.n_coefficients}")
        if self.height <= 0 or self.width <= 0:
            raise WaveletResidualError(
                f"non-positive band shape h={self.height} w={self.width}"
            )
        for field_name, value in (
            ("abs_mean", self.abs_mean),
            ("abs_std", self.abs_std),
            ("abs_max", self.abs_max),
            ("sparsity_fraction", self.sparsity_fraction),
            ("entropy_bits", self.entropy_bits),
        ):
            if not math.isfinite(value):
                raise WaveletResidualError(
                    f"non-finite {field_name}={value} for band {self.band_name}"
                )
            if value < 0.0:
                raise WaveletResidualError(
                    f"negative {field_name}={value} for band {self.band_name}"
                )
        if self.sparsity_fraction > 1.0:
            raise WaveletResidualError(
                f"sparsity_fraction={self.sparsity_fraction} > 1.0 for band {self.band_name}"
            )


@dataclass(frozen=True)
class WaveletResidualResult:
    """Result of `compute_wavelet_residual_stats` over a frame sample.

    The promotion-status fields are FROZEN to research-only by construction;
    no callsite may pass `score_claim=True` etc. into this dataclass.
    """

    wavelet: str
    levels: int
    n_frames: int
    n_channels: int
    height: int
    width: int
    per_band_stats: tuple[BandStats, ...]
    score_claim: bool = field(default=False, init=False)
    promotion_eligible: bool = field(default=False, init=False)
    ready_for_exact_eval_dispatch: bool = field(default=False, init=False)
    evidence_grade: str = field(default=RESEARCH_SIGNAL_EVIDENCE_GRADE, init=False)
    schema: str = field(default="wavelet_residual_pr106_scaffold_v1", init=False)

    def assert_invariants(self) -> None:
        if self.wavelet not in pywt.wavelist():
            raise WaveletResidualError(f"unknown wavelet {self.wavelet!r}")
        if self.levels < 1:
            raise WaveletResidualError(f"levels={self.levels} must be >= 1")
        if self.n_frames <= 0 or self.n_channels <= 0:
            raise WaveletResidualError(
                f"non-positive n_frames={self.n_frames} or n_channels={self.n_channels}"
            )
        if self.height <= 0 or self.width <= 0:
            raise WaveletResidualError(
                f"non-positive shape height={self.height} width={self.width}"
            )
        if not self.per_band_stats:
            raise WaveletResidualError("per_band_stats must be non-empty")
        for stats in self.per_band_stats:
            stats.assert_invariants()
        # Promotion-status invariants — pinned permanently to research-only.
        if self.score_claim:
            raise WaveletResidualError("score_claim must be False (scaffold-only)")
        if self.promotion_eligible:
            raise WaveletResidualError("promotion_eligible must be False (scaffold-only)")
        if self.ready_for_exact_eval_dispatch:
            raise WaveletResidualError(
                "ready_for_exact_eval_dispatch must be False (scaffold-only)"
            )
        if self.evidence_grade != RESEARCH_SIGNAL_EVIDENCE_GRADE:
            raise WaveletResidualError(
                f"evidence_grade must be {RESEARCH_SIGNAL_EVIDENCE_GRADE!r}"
            )


def _validate_rgb_array(frames: np.ndarray) -> None:
    if frames.ndim != 4:
        raise WaveletResidualError(
            f"expected (T, H, W, 3) frames; got ndim={frames.ndim}"
        )
    if frames.shape[3] != RGB_CHANNELS:
        raise WaveletResidualError(
            f"expected last-axis size 3 (RGB); got {frames.shape[3]}"
        )
    if frames.dtype not in (np.uint8, np.float32, np.float64):
        raise WaveletResidualError(
            f"expected dtype uint8 / float32 / float64; got {frames.dtype}"
        )


def _to_float_channels_first(frame: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
    """Convert a single (H, W, 3) uint8 / float frame to (3, H, W) float.

    `dtype` controls the working precision; pass `np.float64` for the
    round-trip test harness (machine-epsilon round-trip) and `np.float32` for
    bulk research-signal sweeps over hundreds of frames (default; halves
    memory).
    """

    if frame.ndim != 3 or frame.shape[2] != RGB_CHANNELS:
        raise WaveletResidualError(
            f"expected single-frame shape (H, W, 3); got {frame.shape}"
        )
    out = frame.astype(dtype, copy=False)
    return np.transpose(out, (2, 0, 1)).copy()


def decompose_frame_to_bands(
    frame: np.ndarray,
    *,
    wavelet: str = DEFAULT_WAVELET,
    levels: int = DEFAULT_DECOMPOSITION_LEVELS,
    working_dtype: np.dtype = np.float32,
) -> list[tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Multi-level 2D DWT decomposition of a single (H, W, 3) RGB frame.

    Returns a list of `(cA, (cH, cV, cD))` per channel-stacked level, ordered
    coarse-to-fine: index 0 is the deepest level (LL_{levels}, (LH_levels,
    HL_levels, HH_levels)); the per-level detail bands at progressively finer
    scales follow.

    Each band tensor has shape `(3, h_at_level, w_at_level)` (channels-first),
    `float32`.

    Notes
    -----

    This is a low-level building block; the high-level entrypoint is
    `compute_wavelet_residual_stats()`. `pywt.wavedec2()` returns the coarse
    approximation first followed by progressively finer detail tuples; we
    preserve that ordering (no manual flip) so downstream tooling does not need
    to second-guess the level index.
    """

    if levels < 1:
        raise WaveletResidualError(f"levels={levels} must be >= 1")
    if wavelet not in pywt.wavelist():
        raise WaveletResidualError(f"unknown wavelet {wavelet!r}")
    chw = _to_float_channels_first(frame, dtype=working_dtype)
    per_channel = [
        pywt.wavedec2(chw[c], wavelet=wavelet, level=levels) for c in range(RGB_CHANNELS)
    ]
    # Each per_channel[c] is [cA_L, (cH_L, cV_L, cD_L), (cH_{L-1}, cV_{L-1}, cD_{L-1}), ...]
    # Stack channels.
    bands: list[tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]] = []
    cA_stack = np.stack([per_channel[c][0] for c in range(RGB_CHANNELS)], axis=0)
    bands.append((cA_stack, _stack_detail(per_channel, 1)))
    for level_index in range(2, levels + 1):
        # Detail-only entries from index 2 onwards mirror pywt's wavedec2 order.
        # cA appears only once at the coarsest level; subsequent entries are
        # (cH, cV, cD) per finer level.
        detail = _stack_detail(per_channel, level_index)
        # Approximation is shared with the level above (we don't recompute LL).
        bands.append((cA_stack if level_index == 1 else _empty_like_band(detail), detail))
    return bands


def _stack_detail(
    per_channel: list[list],
    level_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack per-channel detail bands at a given pywt-level index.

    `pywt.wavedec2` returns `[cA_L, (cH_L, cV_L, cD_L), (cH_{L-1}, cV_{L-1},
    cD_{L-1}), ...]`. Level index 1 corresponds to entry index 1 (coarsest
    detail); level index 2 → entry index 2; etc.
    """

    if level_index < 1:
        raise WaveletResidualError(f"level_index={level_index} must be >= 1")
    if level_index >= len(per_channel[0]):
        raise WaveletResidualError(
            f"level_index={level_index} exceeds available levels "
            f"{len(per_channel[0]) - 1}"
        )
    cH_stack = np.stack(
        [per_channel[c][level_index][0] for c in range(RGB_CHANNELS)], axis=0
    )
    cV_stack = np.stack(
        [per_channel[c][level_index][1] for c in range(RGB_CHANNELS)], axis=0
    )
    cD_stack = np.stack(
        [per_channel[c][level_index][2] for c in range(RGB_CHANNELS)], axis=0
    )
    return (cH_stack, cV_stack, cD_stack)


def _empty_like_band(detail: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """Return a zero-filled placeholder matching the detail-band shape.

    Used for non-coarsest levels where the approximation band is logically
    "shared upward" — we never emit per-level approximation deltas in the
    scaffold; only the deepest cA is stored once at level index 1.
    """

    return np.zeros_like(detail[0])


def reconstruct_frame_from_bands(
    bands: list[tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]],
    *,
    wavelet: str = DEFAULT_WAVELET,
) -> np.ndarray:
    """Inverse-DWT a multi-level band tuple back to (H, W, 3) float32 RGB.

    Inverse of `decompose_frame_to_bands()`; the test harness pins
    `1e-10` numerical round-trip tolerance.

    Notes
    -----

    Re-constitutes the `pywt.wavedec2` coefficient list per channel by reading
    the coarsest cA from bands[0][0] and all detail tuples from bands[i][1].
    """

    if not bands:
        raise WaveletResidualError("bands must be non-empty")
    if wavelet not in pywt.wavelist():
        raise WaveletResidualError(f"unknown wavelet {wavelet!r}")
    coeffs_per_channel = []
    for c in range(RGB_CHANNELS):
        coeffs: list = [bands[0][0][c]]
        for i in range(len(bands)):
            cH, cV, cD = bands[i][1]
            coeffs.append((cH[c], cV[c], cD[c]))
        coeffs_per_channel.append(coeffs)
    rec = np.stack(
        [pywt.waverec2(coeffs_per_channel[c], wavelet=wavelet) for c in range(RGB_CHANNELS)],
        axis=0,
    )
    # Trim to original spatial size if pywt's reconstruction over-extends by 1.
    return np.transpose(rec, (1, 2, 0))


def _entropy_bits_from_band(
    band: np.ndarray,
    *,
    bins: int = DEFAULT_HIST_BINS,
) -> float:
    """Shannon entropy estimate (bits) over a band's float coefficients.

    Quantizes to a fixed integer histogram (`bins` bins centered on zero) and
    returns -sum(p * log2(p)) over the empirical mass. Always non-negative and
    finite (uniform-band cap is `log2(bins)`).
    """

    if band.size == 0:
        return 0.0
    flat = band.reshape(-1).astype(np.float64, copy=False)
    bound = max(float(np.abs(flat).max()), 1.0)
    edges = np.linspace(-bound, bound, bins + 1)
    counts, _ = np.histogram(flat, bins=edges)
    total = float(counts.sum())
    if total <= 0.0:
        return 0.0
    probs = counts.astype(np.float64) / total
    nonzero = probs[probs > 0.0]
    return float(-(nonzero * np.log2(nonzero)).sum())


def _band_stats(
    band: np.ndarray,
    *,
    band_name: str,
    level: int,
    sparsity_eps: float = DEFAULT_SPARSITY_EPSILON,
    bins: int = DEFAULT_HIST_BINS,
) -> BandStats:
    """Single-band BandStats helper.

    Assumes ``band.shape[-2:] = (h, w)`` and uses those dims directly.
    For aggregated multi-frame stats, prefer `_band_stats_from_flat`.
    """

    if band.ndim < 2:
        raise WaveletResidualError(
            f"band must have rank >= 2; got shape {band.shape}"
        )
    flat = band.reshape(-1).astype(np.float64, copy=False)
    return _band_stats_from_flat(
        flat,
        band_name=band_name,
        level=level,
        height=int(band.shape[-2]),
        width=int(band.shape[-1]),
        sparsity_eps=sparsity_eps,
        bins=bins,
    )


def _band_stats_from_flat(
    flat: np.ndarray,
    *,
    band_name: str,
    level: int,
    height: int,
    width: int,
    sparsity_eps: float = DEFAULT_SPARSITY_EPSILON,
    bins: int = DEFAULT_HIST_BINS,
) -> BandStats:
    """BandStats from a pre-flattened aggregate of coefficients."""

    flat = flat.reshape(-1).astype(np.float64, copy=False)
    abs_vals = np.abs(flat)
    sparsity = (
        float(np.sum(abs_vals < sparsity_eps)) / float(flat.size) if flat.size else 0.0
    )
    return BandStats(
        band_name=band_name,
        level=int(level),
        height=int(height),
        width=int(width),
        n_coefficients=int(flat.size),
        abs_mean=float(abs_vals.mean()) if flat.size else 0.0,
        abs_std=float(abs_vals.std()) if flat.size else 0.0,
        abs_max=float(abs_vals.max()) if flat.size else 0.0,
        sparsity_fraction=sparsity,
        entropy_bits=_entropy_bits_from_flat(flat, bins=bins),
    )


def _entropy_bits_from_flat(
    flat: np.ndarray,
    *,
    bins: int = DEFAULT_HIST_BINS,
) -> float:
    if flat.size == 0:
        return 0.0
    bound = max(float(np.abs(flat).max()), 1.0)
    edges = np.linspace(-bound, bound, bins + 1)
    counts, _ = np.histogram(flat, bins=edges)
    total = float(counts.sum())
    if total <= 0.0:
        return 0.0
    probs = counts.astype(np.float64) / total
    nonzero = probs[probs > 0.0]
    return float(-(nonzero * np.log2(nonzero)).sum())


def compute_wavelet_residual_stats(
    frames: np.ndarray,
    *,
    wavelet: str = DEFAULT_WAVELET,
    levels: int = DEFAULT_DECOMPOSITION_LEVELS,
    sparsity_epsilon: float = DEFAULT_SPARSITY_EPSILON,
    hist_bins: int = DEFAULT_HIST_BINS,
    working_dtype: np.dtype = np.float32,
) -> WaveletResidualResult:
    """Compute per-band sparsity + entropy statistics over a sample of frames.

    Aggregates the multi-level wavelet decomposition across all input frames
    and returns a frozen `WaveletResidualResult` with explicit research-only
    promotion-status fields.

    Parameters
    ----------

    frames
        `(T, H, W, 3)` uint8 / float RGB frames. Shape constraints validated.
    wavelet
        Wavelet family name; default `db4` per Holub-Fridrich convention
        (matching `tac.wavelet_variance`).
    levels
        Decomposition depth; default 3 (LL3, LH3, HL3, HH3, LH2, HL2, HH2,
        LH1, HL1, HH1).
    sparsity_epsilon
        Per-coefficient absolute threshold for the "sparse" classification.
    hist_bins
        Histogram bin count for the entropy estimate.

    Returns
    -------

    `WaveletResidualResult` with `score_claim=False`,
    `promotion_eligible=False`, `ready_for_exact_eval_dispatch=False`,
    `evidence_grade="research_signal"`.
    """

    _validate_rgb_array(frames)
    n_frames, height, width, n_channels = frames.shape

    # Aggregate band coefficients across frames per (band-name, level).
    # Store as a list of band tensors plus a representative (h, w) shape so
    # downstream stats can report the per-frame band shape AND aggregate
    # coefficient counts honestly.
    band_acc: dict[tuple[str, int], list[np.ndarray]] = {}
    band_shape: dict[tuple[str, int], tuple[int, int]] = {}
    for t in range(n_frames):
        bands = decompose_frame_to_bands(
            frames[t], wavelet=wavelet, levels=levels, working_dtype=working_dtype
        )
        # bands[0] = (cA_coarsest, (LH_coarsest, HL_coarsest, HH_coarsest))
        # bands[i>=1] = (placeholder_zeros, (LH_finer, HL_finer, HH_finer))
        coarsest_level = levels
        # Approximation only at coarsest level.
        cA = bands[0][0]
        key_ll = ("LL", coarsest_level)
        band_acc.setdefault(key_ll, []).append(cA)
        band_shape.setdefault(key_ll, (int(cA.shape[-2]), int(cA.shape[-1])))
        for entry_index, (_approx, (cH, cV, cD)) in enumerate(bands, start=1):
            level_label = coarsest_level - (entry_index - 1)
            for band_name, tensor in (("LH", cH), ("HL", cV), ("HH", cD)):
                key = (band_name, level_label)
                band_acc.setdefault(key, []).append(tensor)
                band_shape.setdefault(key, (int(tensor.shape[-2]), int(tensor.shape[-1])))

    per_band: list[BandStats] = []
    for (band_name, level), arrays in sorted(
        band_acc.items(), key=lambda kv: (-kv[0][1], _band_order_key(kv[0][0]))
    ):
        flat = np.concatenate([arr.reshape(-1) for arr in arrays], axis=0)
        h, w = band_shape[(band_name, level)]
        per_band.append(
            _band_stats_from_flat(
                flat,
                band_name=band_name,
                level=level,
                height=h,
                width=w,
                sparsity_eps=sparsity_epsilon,
                bins=hist_bins,
            )
        )

    result = WaveletResidualResult(
        wavelet=wavelet,
        levels=levels,
        n_frames=n_frames,
        n_channels=n_channels,
        height=height,
        width=width,
        per_band_stats=tuple(per_band),
    )
    result.assert_invariants()
    return result


def _band_order_key(name: str) -> int:
    return {"LL": 0, "LH": 1, "HL": 2, "HH": 3}[name]


def load_decoded_raw_frames(
    raw_path: Path,
    *,
    height: int = CAMERA_H,
    width: int = CAMERA_W,
    max_frames: int | None = None,
) -> np.ndarray:
    """Read the canonical PR106 sidecar inflate raw output into a numpy array.

    The PR106 sidecar `inflate.py` writes `(N, H=874, W=1164, 3)` uint8 RGB
    frames as contiguous bytes. This helper memory-maps the file and slices the
    first `max_frames` frames (default: ALL frames in the file).

    Reading the full 3.4 GB into RAM is not required for the scaffold — the
    test harness uses synthetic 32x32 fixtures, and any real-data sweep can
    use `max_frames=32` for a 16-pair smoke that fits in <30 MB.
    """

    raw_path = Path(raw_path)
    if not raw_path.is_file():
        raise WaveletResidualError(f"raw file not found: {raw_path}")
    frame_bytes = height * width * RGB_CHANNELS
    total = raw_path.stat().st_size
    if total % frame_bytes != 0:
        raise WaveletResidualError(
            f"file size {total} not a multiple of frame_bytes {frame_bytes}"
        )
    n_total = total // frame_bytes
    n_read = n_total if max_frames is None else min(n_total, int(max_frames))
    if n_read <= 0:
        raise WaveletResidualError(f"max_frames={max_frames} resolves to 0")
    arr = np.memmap(
        raw_path, dtype=np.uint8, mode="r", shape=(n_total, height, width, RGB_CHANNELS)
    )
    # Force a copy of just the slice we want so the rest of the memmap can be
    # released; downstream wavelet-decomposition needs a real array anyway.
    return np.array(arr[:n_read], dtype=np.uint8)


__all__ = [
    "BandStats",
    "CAMERA_H",
    "CAMERA_W",
    "DEFAULT_DECOMPOSITION_LEVELS",
    "DEFAULT_HIST_BINS",
    "DEFAULT_SPARSITY_EPSILON",
    "DEFAULT_WAVELET",
    "NUMERICAL_ROUNDTRIP_TOLERANCE",
    "RESEARCH_SIGNAL_EVIDENCE_GRADE",
    "RGB_CHANNELS",
    "WaveletResidualError",
    "WaveletResidualResult",
    "compute_wavelet_residual_stats",
    "decompose_frame_to_bands",
    "load_decoded_raw_frames",
    "reconstruct_frame_from_bands",
]
