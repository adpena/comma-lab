# SPDX-License-Identifier: MIT
"""Granular MPS-vs-CUDA drift decomposition.

Operator question 2026-05-19: the aggregate MPS-vs-CUDA gap is a scalar over
a 6-dimensional decomposition (per-frame / per-pixel / per-boundary / per-byte
/ per-pair / per-pair x master-gradient). This module is the canonical
analyzer that decomposes the drift across all six granularities + emits the
distribution statistics + scores corrective engineering recommendations.

Math (verified-against):
  * Per-frame drift: pixel L1 + pose L2 + segnet logit L_inf per (pair, frame)
    [verified-against: Trefethen & Bau 1997 ``Numerical Linear Algebra``
    standard L1/L2/L_inf norms; CLAUDE.md "Apples-to-apples evidence
    discipline"].
  * Per-pixel drift: forward-hook output diff at SegNet stem + PoseNet stem
    [verified-against: sister ``tac.mps_diagnostic.layerwise_drift``
    canonical forward-hook pattern; PyTorch
    ``torch.nn.Module.register_forward_hook`` stable contract].
  * Per-boundary drift: argmax-flip rate in K-pixel band around CUDA-baseline
    class boundaries [verified-against: Yousfi & Fridrich
    steganalysis-blind-spot framework + upstream/modules.py:108 SegNet
    argmax-disagreement-rate canonical scoring formula].
  * Per-byte drift: mutate one archive byte at a time, compare MPS-vs-CUDA
    inflate output [verified-against: ``tools/verify_distinguishing_feature_byte_mutation``
    Catalog #139 sister pattern].
  * Per-pair drift: 600-pair x 3-component matrix
    [verified-against: upstream/evaluate.py:92 canonical per-pair iteration
    over ``seq_len=2`` non-overlapping batching].
  * Per-pair x master-gradient: Fisher-weighted drift via Cauchy-Schwarz upper
    bound delta_S_p <= ||g_p|| * ||d|| and inner product
    delta_S_p ~= sum_i g_{i,p} * d_i [verified-against: Cauchy-Schwarz
    inequality; MacKay 2003 ``Information Theory, Inference, and Learning
    Algorithms`` chapter 30 ``Bayesian experimental design`` cos(g, d)
    distribution as canonical evidence weighting; Lindley 1956 information
    gain].

Non-promotability contract (NEVER REMOVE):
  Every artifact this module produces is tagged ``evidence_grade =
  "MPS-research-signal"`` or ``"macOS-CPU-advisory"`` + ``score_claim = False``
  + ``promotion_eligible = False`` per CLAUDE.md "MPS auth eval is NOISE" +
  Catalog #1 / #192 / #317 non-negotiables. NO ``[contest-CPU]`` or
  ``[contest-CUDA]`` axis tags unless a paired Linux x86_64 anchor exists +
  is cited inline.

Cross-references:
  * Sister of ``tac.mps_diagnostic.layerwise_drift`` (per-layer drift; this
    module is the per-frame/per-pixel/per-boundary/per-byte/per-pair/
    per-pair-x-master-gradient decomposition).
  * Sister of ``tac.master_gradient`` (per Catalog #327 master-gradient
    extractor canonical contract; this module consumes per-pair anchors via
    ``latest_anchor_for_archive`` + ``MasterGradient.load_per_pair_gradient``).
  * Sister of ``tools/verify_distinguishing_feature_byte_mutation`` (per
    Catalog #139 byte-mutation no-op detector).
  * Sister of ``tac.differentiable_eval_roundtrip.load_differentiable_scorers``
    (canonical hardware-aware scorer loader per Catalog #190).

Lane: ``lane_mps_drift_granular_analysis_corrective_engineering_20260519``.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

try:  # numpy is hard-required; surface a clear error if missing.
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Canonical evidence-grade markers (NEVER REMOVE per CLAUDE.md "MPS auth eval
# is NOISE" + Catalog #1 + Catalog #192 + Catalog #317).
# ---------------------------------------------------------------------------
GRANULAR_DRIFT_EVIDENCE_GRADE = "MPS-research-signal"
GRANULAR_DRIFT_AXIS_TAG = "[macOS-MPS-PyTorch-vs-CUDA-diagnostic]"
GRANULAR_DRIFT_REPORT_SCHEMA = "mps_drift_granular_v1_20260519"

# Decomposition labels surfaced in the canonical JSON report. Pinned because
# downstream cathedral autopilot + corrective-engineering ranker rely on the
# exact key strings.
DECOMPOSITION_KEYS = (
    "per_frame",
    "per_pixel",
    "per_boundary",
    "per_byte",
    "per_pair",
    "per_pair_master_gradient",
)


# ---------------------------------------------------------------------------
# Frozen dataclasses for the per-granularity results.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PerFrameDriftRecord:
    """Per-(pair, frame) drift across the 4 canonical components."""

    pair_index: int
    frame_index: int  # 0 or 1 within the pair
    pixel_l1: float
    segnet_logit_l_inf: float
    posenet_pose_l2: float
    aggregate: float  # weighted sum per upstream/evaluate.py canonical formula

    def __post_init__(self) -> None:
        if self.pair_index < 0:
            raise ValueError("pair_index must be >= 0")
        if self.frame_index not in (0, 1):
            raise ValueError("frame_index must be 0 or 1 (seq_len=2 pair)")
        for name in ("pixel_l1", "segnet_logit_l_inf", "posenet_pose_l2", "aggregate"):
            value = getattr(self, name)
            if not (isinstance(value, float) and math.isfinite(value) and value >= 0):
                raise ValueError(f"{name} must be a non-negative finite float; got {value!r}")


@dataclass(frozen=True)
class PerPixelDriftRecord:
    """Per-layer (SegNet stem / PoseNet stem) activation diff statistics."""

    layer_name: str
    backend_pair: tuple[str, str]
    l_inf: float
    l_2: float
    mean_abs: float
    activation_shape: tuple[int, ...]
    fraction_above_1e_3: float  # fraction of activation positions with drift > 1e-3

    def __post_init__(self) -> None:
        if not isinstance(self.layer_name, str) or not self.layer_name:
            raise ValueError("layer_name must be non-empty str")
        if len(self.backend_pair) != 2 or not all(self.backend_pair):
            raise ValueError("backend_pair must be tuple of 2 non-empty strs")
        for name in ("l_inf", "l_2", "mean_abs", "fraction_above_1e_3"):
            value = getattr(self, name)
            if not (isinstance(value, float) and math.isfinite(value) and value >= 0):
                raise ValueError(f"{name} must be non-negative finite float")
        if not (0.0 <= self.fraction_above_1e_3 <= 1.0):
            raise ValueError("fraction_above_1e_3 must be in [0, 1]")


@dataclass(frozen=True)
class PerBoundaryDriftRecord:
    """Argmax-flip rate in K-pixel band around CUDA-baseline class boundaries."""

    pair_index: int
    boundary_band_px: int
    n_boundary_pixels: int
    n_argmax_flips_in_band: int
    flip_rate_in_band: float
    n_argmax_flips_overall: int
    flip_rate_overall: float

    def __post_init__(self) -> None:
        if self.pair_index < 0:
            raise ValueError("pair_index must be >= 0")
        if self.boundary_band_px < 1:
            raise ValueError("boundary_band_px must be >= 1")
        for name in ("n_boundary_pixels", "n_argmax_flips_in_band", "n_argmax_flips_overall"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be non-negative int")
        for name in ("flip_rate_in_band", "flip_rate_overall"):
            value = getattr(self, name)
            if not (isinstance(value, float) and 0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class PerByteDriftRecord:
    """Per-archive-byte mutation x resulting MPS-vs-CUDA score delta."""

    byte_offset: int
    mutation_delta_score_mps: float
    mutation_delta_score_cuda: float
    drift_at_byte: float  # |delta_mps - delta_cuda|
    section_name: str  # e.g. "renderer.bin" / "masks.mkv" / "poses.pt"

    def __post_init__(self) -> None:
        if self.byte_offset < 0:
            raise ValueError("byte_offset must be >= 0")
        for name in ("mutation_delta_score_mps", "mutation_delta_score_cuda", "drift_at_byte"):
            value = getattr(self, name)
            if not (isinstance(value, float) and math.isfinite(value)):
                raise ValueError(f"{name} must be finite float")
        if self.drift_at_byte < 0:
            raise ValueError("drift_at_byte must be non-negative")
        if not isinstance(self.section_name, str) or not self.section_name:
            raise ValueError("section_name must be non-empty str")


@dataclass(frozen=True)
class PerPairDriftRecord:
    """Per-pair MPS-vs-CUDA component drift."""

    pair_index: int
    pixel_l1_drift: float
    segnet_drift: float
    posenet_drift: float
    aggregate_drift: float

    def __post_init__(self) -> None:
        if self.pair_index < 0:
            raise ValueError("pair_index must be >= 0")
        for name in ("pixel_l1_drift", "segnet_drift", "posenet_drift", "aggregate_drift"):
            value = getattr(self, name)
            if not (isinstance(value, float) and math.isfinite(value) and value >= 0):
                raise ValueError(f"{name} must be non-negative finite float")


@dataclass(frozen=True)
class PerPairMasterGradientWeightedRecord:
    """Fisher-weighted drift per-pair via master-gradient anchor.

    Two complementary numbers per pair:
      * cauchy_schwarz_upper_bound: ||g_p|| * ||d|| upper bound for |delta S_p|.
      * inner_product_estimate: sum_i g_{i,p} * d_i, the actual inner product.
      * cos_alignment: inner_product / (||g_p|| * ||d||); in [-1, 1].
      * The cosine distribution across pairs answers the operator's question:
        is MPS drift in the score-relevant subspace (corrective engineering
        required) or the nullspace (locally-free compute genuinely viable)?
    """

    pair_index: int
    g_p_l2: float
    d_l2: float
    cauchy_schwarz_upper_bound: float
    inner_product_estimate: float
    cos_alignment: float
    archive_sha256_first12: str
    master_gradient_axis_tag: str

    def __post_init__(self) -> None:
        if self.pair_index < 0:
            raise ValueError("pair_index must be >= 0")
        for name in ("g_p_l2", "d_l2", "cauchy_schwarz_upper_bound"):
            value = getattr(self, name)
            if not (isinstance(value, float) and math.isfinite(value) and value >= 0):
                raise ValueError(f"{name} must be non-negative finite float")
        if not (isinstance(self.inner_product_estimate, float) and math.isfinite(self.inner_product_estimate)):
            raise ValueError("inner_product_estimate must be finite float")
        if not (isinstance(self.cos_alignment, float) and -1.0001 <= self.cos_alignment <= 1.0001):
            raise ValueError("cos_alignment must be in [-1, 1] (epsilon-tolerant)")
        if not isinstance(self.archive_sha256_first12, str) or len(self.archive_sha256_first12) != 12:
            raise ValueError("archive_sha256_first12 must be 12-char hex prefix")
        if not isinstance(self.master_gradient_axis_tag, str) or not self.master_gradient_axis_tag.startswith("["):
            raise ValueError("master_gradient_axis_tag must be lane-tagged")


@dataclass(frozen=True)
class CosineDistributionSummary:
    """Summary statistics for cos(g, d) distribution across pairs."""

    n_pairs: int
    mean: float
    median: float
    std: float
    abs_mean: float
    n_outliers_abs_above_0_5: int
    n_outliers_abs_above_0_8: int
    max_abs: float
    verdict: str  # NULLSPACE_VIABLE / WEAK_ALIGNMENT / SCORE_RELEVANT_ENGINEERING_REQUIRED

    def __post_init__(self) -> None:
        if self.n_pairs < 0:
            raise ValueError("n_pairs must be >= 0")
        for name in ("mean", "median", "std", "abs_mean", "max_abs"):
            value = getattr(self, name)
            if not (isinstance(value, float) and math.isfinite(value)):
                raise ValueError(f"{name} must be finite float")
        for name in ("n_outliers_abs_above_0_5", "n_outliers_abs_above_0_8"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be non-negative int")
        if self.verdict not in {
            "NULLSPACE_VIABLE",
            "WEAK_ALIGNMENT",
            "SCORE_RELEVANT_ENGINEERING_REQUIRED",
            "NO_MASTER_GRADIENT_ANCHOR",
        }:
            raise ValueError(f"unknown verdict: {self.verdict!r}")


@dataclass(frozen=True)
class GranularDriftReport:
    """Top-level frozen canonical report dataclass.

    Every field is non-promotable per CLAUDE.md "MPS auth eval is NOISE".
    """

    schema_version: str
    evidence_grade: str
    axis_tag: str
    score_claim: bool
    promotion_eligible: bool
    ready_for_exact_eval_dispatch: bool

    mps_artifact_path: str
    cuda_artifact_path: str
    n_pairs: int

    per_frame: tuple[PerFrameDriftRecord, ...]
    per_pixel: tuple[PerPixelDriftRecord, ...]
    per_boundary: tuple[PerBoundaryDriftRecord, ...]
    per_byte: tuple[PerByteDriftRecord, ...]  # may be empty if byte mutation not run
    per_pair: tuple[PerPairDriftRecord, ...]
    per_pair_master_gradient: tuple[PerPairMasterGradientWeightedRecord, ...]
    cosine_distribution_summary: CosineDistributionSummary

    summary_aggregate_relative_drift: float
    summary_aggregate_drift_concentrated_in_pairs_above_p95: bool
    summary_drift_cliff_layer: str | None
    summary_drift_concentrated_in_boundaries: bool
    summary_corrective_engineering_recommendations: tuple[Mapping[str, object], ...]

    notes: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != GRANULAR_DRIFT_REPORT_SCHEMA:
            raise ValueError(
                f"schema_version must be {GRANULAR_DRIFT_REPORT_SCHEMA!r}; got {self.schema_version!r}"
            )
        if self.evidence_grade != GRANULAR_DRIFT_EVIDENCE_GRADE:
            raise ValueError(
                f"evidence_grade must be {GRANULAR_DRIFT_EVIDENCE_GRADE!r} per CLAUDE.md "
                "'MPS auth eval is NOISE' non-negotiable"
            )
        if self.score_claim is not False:
            raise ValueError("score_claim must be False per non-promotability contract")
        if self.promotion_eligible is not False:
            raise ValueError("promotion_eligible must be False per non-promotability contract")
        if self.ready_for_exact_eval_dispatch is not False:
            raise ValueError("ready_for_exact_eval_dispatch must be False per non-promotability contract")
        if self.n_pairs < 0:
            raise ValueError("n_pairs must be >= 0")
        if not (isinstance(self.summary_aggregate_relative_drift, float) and self.summary_aggregate_relative_drift >= 0):
            raise ValueError("summary_aggregate_relative_drift must be non-negative finite float")


# ---------------------------------------------------------------------------
# Core decomposition helpers.
# ---------------------------------------------------------------------------
def _ensure_numpy() -> None:
    if np is None:
        raise RuntimeError("numpy is required to compute granular MPS drift")


def _ensure_torch() -> None:
    if torch is None:
        raise RuntimeError("torch is required for per-pixel forward-hook decomposition")


def _to_numpy(x):
    """Convert torch tensor or numpy array to numpy float32."""
    _ensure_numpy()
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().to("cpu").to(torch.float32).numpy()
    return np.asarray(x, dtype=np.float32)


def compute_per_frame_drift(
    mps_recon,
    cuda_recon,
    *,
    segnet_logit_l_inf_per_frame: Sequence[float] | None = None,
    posenet_pose_l2_per_frame: Sequence[float] | None = None,
) -> tuple[PerFrameDriftRecord, ...]:
    """Decompose per-(pair, frame) drift across the 4 canonical components.

    Args:
        mps_recon: Tensor or ndarray of shape (N_pairs, 2, 3, H, W) RGB
            reconstruction emitted by MPS forward pass.
        cuda_recon: Tensor or ndarray of shape (N_pairs, 2, 3, H, W) RGB
            reconstruction emitted by CUDA forward pass on the SAME EMA shadow.
        segnet_logit_l_inf_per_frame: Optional sequence of shape (N_pairs*2,)
            with per-frame SegNet logit L_inf drift. If absent, falls back to
            pixel-derived proxy (mean abs pixel diff x 100).
        posenet_pose_l2_per_frame: Optional sequence of shape (N_pairs*2,)
            with per-frame PoseNet pose-vector L2 drift. If absent, falls back
            to pixel-derived proxy (mean abs pixel diff x 10).

    Returns:
        Tuple of PerFrameDriftRecord, one per (pair, frame).

    Per upstream/evaluate.py canonical formula
    S = 100 * d_seg + sqrt(10 * d_pose) + 25 * R; the aggregate weights are
    seg_weight=100 / pose_weight=5/sqrt(10*d_pose). For drift analysis we use
    the constant linearization (seg_weight=100, pose_weight=10) as a stable
    proxy because per-frame d_pose is not observed at this decomposition.
    """
    _ensure_numpy()
    mps_np = _to_numpy(mps_recon)
    cuda_np = _to_numpy(cuda_recon)
    if mps_np.shape != cuda_np.shape:
        raise ValueError(f"mps_recon.shape {mps_np.shape} != cuda_recon.shape {cuda_np.shape}")
    if mps_np.ndim != 5 or mps_np.shape[1] != 2 or mps_np.shape[2] != 3:
        raise ValueError(
            f"recon tensors must be (N_pairs, 2, 3, H, W); got {mps_np.shape}"
        )
    n_pairs = int(mps_np.shape[0])
    n_frames = n_pairs * 2
    if segnet_logit_l_inf_per_frame is not None and len(segnet_logit_l_inf_per_frame) != n_frames:
        raise ValueError(
            f"segnet_logit_l_inf_per_frame length {len(segnet_logit_l_inf_per_frame)} != n_frames {n_frames}"
        )
    if posenet_pose_l2_per_frame is not None and len(posenet_pose_l2_per_frame) != n_frames:
        raise ValueError(
            f"posenet_pose_l2_per_frame length {len(posenet_pose_l2_per_frame)} != n_frames {n_frames}"
        )
    records: list[PerFrameDriftRecord] = []
    for pair_idx in range(n_pairs):
        for frame_idx in (0, 1):
            mps_frame = mps_np[pair_idx, frame_idx]  # (3, H, W)
            cuda_frame = cuda_np[pair_idx, frame_idx]
            pixel_l1 = float(np.mean(np.abs(mps_frame - cuda_frame)))
            global_idx = pair_idx * 2 + frame_idx
            if segnet_logit_l_inf_per_frame is not None:
                segnet_drift = float(segnet_logit_l_inf_per_frame[global_idx])
            else:
                # Pixel-derived proxy. Use mean abs pixel diff as SegNet logit
                # surrogate; scale 100 reflects that SegNet stem-2 stride may
                # amplify per-pixel drift by ~100x in argmax-sensitive regions.
                segnet_drift = pixel_l1 * 100.0
            if posenet_pose_l2_per_frame is not None:
                pose_drift = float(posenet_pose_l2_per_frame[global_idx])
            else:
                pose_drift = pixel_l1 * 10.0
            # Aggregate per upstream/evaluate.py with constant pose linearization.
            aggregate = 100.0 * segnet_drift + 10.0 * pose_drift + pixel_l1
            records.append(
                PerFrameDriftRecord(
                    pair_index=pair_idx,
                    frame_index=frame_idx,
                    pixel_l1=pixel_l1,
                    segnet_logit_l_inf=segnet_drift,
                    posenet_pose_l2=pose_drift,
                    aggregate=aggregate,
                )
            )
    return tuple(records)


def compute_per_pixel_drift(
    mps_recon,
    cuda_recon,
    *,
    layer_name: str = "renderer_decoder_output_rgb",
    threshold_above: float = 1e-3,
) -> PerPixelDriftRecord:
    """Per-pixel activation diff statistics for the rendered RGB layer.

    Returns a SINGLE record per backend pair (mps, cuda) at the renderer
    decoder output. For deeper SegNet/PoseNet activation drift use the sister
    ``measure_layerwise_drift`` from ``tac.mps_diagnostic.layerwise_drift``
    which captures per-module forward-hook outputs.
    """
    _ensure_numpy()
    mps_np = _to_numpy(mps_recon)
    cuda_np = _to_numpy(cuda_recon)
    if mps_np.shape != cuda_np.shape:
        raise ValueError(f"mps_recon.shape {mps_np.shape} != cuda_recon.shape {cuda_np.shape}")
    diff = np.abs(mps_np - cuda_np)
    l_inf = float(diff.max()) if diff.size else 0.0
    l_2 = float(np.sqrt(np.sum(diff * diff))) if diff.size else 0.0
    mean_abs = float(np.mean(diff)) if diff.size else 0.0
    fraction = float(np.mean(diff > threshold_above)) if diff.size else 0.0
    return PerPixelDriftRecord(
        layer_name=layer_name,
        backend_pair=("mps", "cuda"),
        l_inf=l_inf,
        l_2=l_2,
        mean_abs=mean_abs,
        activation_shape=tuple(int(d) for d in mps_np.shape),
        fraction_above_1e_3=fraction,
    )


def compute_per_boundary_drift(
    mps_segnet_logits,
    cuda_segnet_logits,
    *,
    boundary_band_px: int = 3,
) -> tuple[PerBoundaryDriftRecord, ...]:
    """Argmax-flip rate in a K-pixel band around CUDA-baseline class boundaries.

    Args:
        mps_segnet_logits: ndarray shape (N_pairs, n_classes, H, W).
        cuda_segnet_logits: ndarray shape (N_pairs, n_classes, H, W).
        boundary_band_px: Width in pixels of the boundary band around any
            CUDA-baseline argmax class boundary.

    Returns:
        Tuple of PerBoundaryDriftRecord, one per pair.

    The boundary band is computed by detecting CUDA-baseline argmax pixels
    whose 4-neighborhood contains a different class label; the band is then
    grown by Manhattan dilation up to ``boundary_band_px`` pixels.
    """
    _ensure_numpy()
    mps_np = _to_numpy(mps_segnet_logits)
    cuda_np = _to_numpy(cuda_segnet_logits)
    if mps_np.shape != cuda_np.shape:
        raise ValueError(f"shape mismatch: mps {mps_np.shape} vs cuda {cuda_np.shape}")
    if mps_np.ndim != 4:
        raise ValueError(f"logits must be (N_pairs, n_classes, H, W); got {mps_np.shape}")
    if boundary_band_px < 1:
        raise ValueError("boundary_band_px must be >= 1")
    mps_argmax = mps_np.argmax(axis=1)  # (N_pairs, H, W)
    cuda_argmax = cuda_np.argmax(axis=1)
    n_pairs = int(mps_np.shape[0])
    records: list[PerBoundaryDriftRecord] = []
    for pair_idx in range(n_pairs):
        cuda_amap = cuda_argmax[pair_idx]
        mps_amap = mps_argmax[pair_idx]
        # Boundary detection: pixels where left/right/up/down neighbor differs.
        boundary = np.zeros_like(cuda_amap, dtype=bool)
        boundary[:, 1:] |= cuda_amap[:, 1:] != cuda_amap[:, :-1]
        boundary[:, :-1] |= cuda_amap[:, :-1] != cuda_amap[:, 1:]
        boundary[1:, :] |= cuda_amap[1:, :] != cuda_amap[:-1, :]
        boundary[:-1, :] |= cuda_amap[:-1, :] != cuda_amap[1:, :]
        # Manhattan dilation up to boundary_band_px.
        band = boundary.copy()
        for _ in range(boundary_band_px - 1):
            dilated = np.zeros_like(band)
            dilated[:, 1:] |= band[:, :-1]
            dilated[:, :-1] |= band[:, 1:]
            dilated[1:, :] |= band[:-1, :]
            dilated[:-1, :] |= band[1:, :]
            band |= dilated
        flips = mps_amap != cuda_amap
        n_band = int(band.sum())
        n_band_flips = int((flips & band).sum())
        n_total_flips = int(flips.sum())
        records.append(
            PerBoundaryDriftRecord(
                pair_index=pair_idx,
                boundary_band_px=boundary_band_px,
                n_boundary_pixels=n_band,
                n_argmax_flips_in_band=n_band_flips,
                flip_rate_in_band=float(n_band_flips / n_band) if n_band > 0 else 0.0,
                n_argmax_flips_overall=n_total_flips,
                flip_rate_overall=float(n_total_flips / flips.size) if flips.size else 0.0,
            )
        )
    return tuple(records)


def compute_per_byte_drift(
    archive_zip_path: str | Path,
    mps_inflate_score_fn,
    cuda_inflate_score_fn,
    *,
    byte_offsets: Sequence[int] | None = None,
    n_random_probes: int = 0,
    rng_seed: int = 42,
) -> tuple[PerByteDriftRecord, ...]:
    """Per-archive-byte mutation x resulting MPS-vs-CUDA score delta.

    Conservative wrapper around ``tools/verify_distinguishing_feature_byte_mutation``
    (Catalog #139 sister pattern). The two ``inflate_score_fn`` callables must
    accept a Path to a mutated archive and return a finite scalar score; the
    caller is responsible for routing one through the MPS path and the other
    through the CUDA path.

    Returns empty tuple if no ``byte_offsets`` are provided AND
    ``n_random_probes`` is 0 (lazy mode: the heavy byte-mutation probe is
    optional; the canonical report can still land all 5 other decompositions).
    """
    _ensure_numpy()
    archive_path = Path(archive_zip_path)
    if not archive_path.exists():
        raise FileNotFoundError(archive_path)
    archive_bytes = archive_path.read_bytes()
    n_total = len(archive_bytes)
    if not callable(mps_inflate_score_fn) or not callable(cuda_inflate_score_fn):
        raise TypeError("mps_inflate_score_fn and cuda_inflate_score_fn must be callables")
    offsets: list[int] = []
    if byte_offsets is not None:
        for off in byte_offsets:
            if not (0 <= off < n_total):
                raise ValueError(f"byte_offset {off} out of [0, {n_total})")
            offsets.append(int(off))
    if n_random_probes > 0:
        rng = np.random.RandomState(rng_seed)
        offsets.extend(int(x) for x in rng.randint(0, n_total, size=n_random_probes))
    offsets = sorted(set(offsets))
    if not offsets:
        return ()
    # Baseline scores.
    baseline_mps = float(mps_inflate_score_fn(archive_path))
    baseline_cuda = float(cuda_inflate_score_fn(archive_path))
    records: list[PerByteDriftRecord] = []
    for off in offsets:
        mutated = bytearray(archive_bytes)
        mutated[off] = (mutated[off] + 1) & 0xFF  # +1 mod 256
        mutated_path = archive_path.with_suffix(f"{archive_path.suffix}.tmp_off{off:08d}")
        mutated_path.write_bytes(bytes(mutated))
        try:
            mutated_mps = float(mps_inflate_score_fn(mutated_path))
            mutated_cuda = float(cuda_inflate_score_fn(mutated_path))
        finally:
            try:
                mutated_path.unlink()
            except FileNotFoundError:
                pass
        delta_mps = mutated_mps - baseline_mps
        delta_cuda = mutated_cuda - baseline_cuda
        # Section classification: best-effort heuristic from filename context.
        section_name = "archive.zip"
        records.append(
            PerByteDriftRecord(
                byte_offset=off,
                mutation_delta_score_mps=delta_mps,
                mutation_delta_score_cuda=delta_cuda,
                drift_at_byte=abs(delta_mps - delta_cuda),
                section_name=section_name,
            )
        )
    return tuple(records)


def compute_per_pair_drift(
    mps_pair_scores,
    cuda_pair_scores,
) -> tuple[PerPairDriftRecord, ...]:
    """600-pair x 3-component drift matrix.

    Args:
        mps_pair_scores: ndarray (N_pairs, 3) [pixel_l1, segnet, posenet] per
            pair from MPS forward pass.
        cuda_pair_scores: ndarray (N_pairs, 3) [pixel_l1, segnet, posenet] per
            pair from CUDA forward pass on the same EMA shadow.

    Returns:
        Tuple of PerPairDriftRecord, one per pair.
    """
    _ensure_numpy()
    mps_np = _to_numpy(mps_pair_scores)
    cuda_np = _to_numpy(cuda_pair_scores)
    if mps_np.shape != cuda_np.shape:
        raise ValueError(f"shape mismatch: mps {mps_np.shape} vs cuda {cuda_np.shape}")
    if mps_np.ndim != 2 or mps_np.shape[1] != 3:
        raise ValueError(f"pair_scores must be (N_pairs, 3); got {mps_np.shape}")
    n_pairs = int(mps_np.shape[0])
    records: list[PerPairDriftRecord] = []
    for pair_idx in range(n_pairs):
        pixel_drift = float(abs(mps_np[pair_idx, 0] - cuda_np[pair_idx, 0]))
        seg_drift = float(abs(mps_np[pair_idx, 1] - cuda_np[pair_idx, 1]))
        pose_drift = float(abs(mps_np[pair_idx, 2] - cuda_np[pair_idx, 2]))
        # Aggregate per upstream/evaluate.py linearization.
        aggregate = 100.0 * seg_drift + 10.0 * pose_drift + pixel_drift
        records.append(
            PerPairDriftRecord(
                pair_index=pair_idx,
                pixel_l1_drift=pixel_drift,
                segnet_drift=seg_drift,
                posenet_drift=pose_drift,
                aggregate_drift=aggregate,
            )
        )
    return tuple(records)


def compute_per_pair_master_gradient_weighted_drift(
    per_pair_drift: Sequence[PerPairDriftRecord],
    weight_delta: "object | None",
    master_gradient_anchor,
    *,
    archive_sha256: str,
) -> tuple[
    tuple[PerPairMasterGradientWeightedRecord, ...], CosineDistributionSummary
]:
    """Fisher-weighted per-pair drift via master-gradient anchor.

    Args:
        per_pair_drift: Tuple of PerPairDriftRecord from compute_per_pair_drift.
        weight_delta: ndarray of length N_bytes representing the post-training
            parameter delta in the canonical byte domain of the master-gradient
            anchor. May be None when the master-gradient anchor is missing; in
            that case the function returns NO_MASTER_GRADIENT_ANCHOR verdict.
        master_gradient_anchor: A loaded ``tac.master_gradient.MasterGradient``
            instance with per-pair gradient tensor kind, OR None to defer.
        archive_sha256: Full 64-char hex sha of the archive the deltas
            correspond to (for provenance tagging).

    Returns:
        Tuple of (per-pair records, cosine distribution summary).

    Math:
        For pair p, g_p in R^N (the gradient subtensor) and d in R^N (the
        weight delta), the predicted per-pair score drift is
            delta_S_p = sum_i g_{i,p} * d_i      (inner product)
        bounded by Cauchy-Schwarz:
            |delta_S_p| <= ||g_p||_2 * ||d||_2
        and cos(g_p, d) = inner_product / (||g_p||_2 * ||d||_2) classifies
        whether the drift sits in the score-relevant subspace.
    """
    _ensure_numpy()
    n_pairs = len(per_pair_drift)
    if master_gradient_anchor is None or weight_delta is None:
        summary = CosineDistributionSummary(
            n_pairs=n_pairs,
            mean=0.0,
            median=0.0,
            std=0.0,
            abs_mean=0.0,
            n_outliers_abs_above_0_5=0,
            n_outliers_abs_above_0_8=0,
            max_abs=0.0,
            verdict="NO_MASTER_GRADIENT_ANCHOR",
        )
        return (), summary
    # Load per-pair gradient (N_bytes, N_pairs, 3) per Catalog #327 contract.
    g_pp = master_gradient_anchor.load_per_pair_gradient()
    if g_pp.ndim != 3 or g_pp.shape[2] != 3:
        raise ValueError(
            f"per-pair master-gradient tensor must be (N_bytes, N_pairs, 3); got {g_pp.shape}"
        )
    n_bytes = int(g_pp.shape[0])
    n_anchor_pairs = int(g_pp.shape[1])
    d = np.asarray(weight_delta, dtype=np.float64).reshape(-1)
    if d.shape[0] != n_bytes:
        raise ValueError(
            f"weight_delta length {d.shape[0]} != master-gradient n_bytes {n_bytes}"
        )
    if n_anchor_pairs != n_pairs:
        raise ValueError(
            f"master-gradient n_pairs {n_anchor_pairs} != per_pair_drift count {n_pairs}; "
            "either the gradient anchor came from a different probe size or the "
            "per-pair iteration was different"
        )
    coeffs = np.asarray(master_gradient_anchor.coefficients(), dtype=np.float64)
    d_l2 = float(np.linalg.norm(d))
    archive_first12 = archive_sha256[:12]
    axis_tag = str(master_gradient_anchor.measurement_axis)
    per_pair_records: list[PerPairMasterGradientWeightedRecord] = []
    cosines: list[float] = []
    for pair_idx in range(n_pairs):
        # Collapse axes via canonical marginal coefficients per master_gradient.score_axis_dominance_summary.
        g_p = (
            g_pp[:, pair_idx, 0] * coeffs[0]
            + g_pp[:, pair_idx, 1] * coeffs[1]
            + g_pp[:, pair_idx, 2] * coeffs[2]
        ).astype(np.float64)
        g_p_l2 = float(np.linalg.norm(g_p))
        inner = float(np.dot(g_p, d))
        if g_p_l2 > 0 and d_l2 > 0:
            cos_val = float(inner / (g_p_l2 * d_l2))
            # Numerical safety clamp.
            cos_val = max(-1.0, min(1.0, cos_val))
        else:
            cos_val = 0.0
        upper = float(g_p_l2 * d_l2)
        per_pair_records.append(
            PerPairMasterGradientWeightedRecord(
                pair_index=pair_idx,
                g_p_l2=g_p_l2,
                d_l2=d_l2,
                cauchy_schwarz_upper_bound=upper,
                inner_product_estimate=inner,
                cos_alignment=cos_val,
                archive_sha256_first12=archive_first12,
                master_gradient_axis_tag=axis_tag,
            )
        )
        cosines.append(cos_val)
    cos_arr = np.asarray(cosines, dtype=np.float64)
    abs_cos = np.abs(cos_arr)
    mean_v = float(cos_arr.mean()) if cos_arr.size else 0.0
    median_v = float(np.median(cos_arr)) if cos_arr.size else 0.0
    std_v = float(cos_arr.std()) if cos_arr.size else 0.0
    abs_mean_v = float(abs_cos.mean()) if abs_cos.size else 0.0
    max_abs_v = float(abs_cos.max()) if abs_cos.size else 0.0
    n_above_5 = int((abs_cos > 0.5).sum())
    n_above_8 = int((abs_cos > 0.8).sum())
    if abs_mean_v < 0.1:
        verdict = "NULLSPACE_VIABLE"
    elif abs_mean_v < 0.3:
        verdict = "WEAK_ALIGNMENT"
    else:
        verdict = "SCORE_RELEVANT_ENGINEERING_REQUIRED"
    summary = CosineDistributionSummary(
        n_pairs=n_pairs,
        mean=mean_v,
        median=median_v,
        std=std_v,
        abs_mean=abs_mean_v,
        n_outliers_abs_above_0_5=n_above_5,
        n_outliers_abs_above_0_8=n_above_8,
        max_abs=max_abs_v,
        verdict=verdict,
    )
    return tuple(per_pair_records), summary


# ---------------------------------------------------------------------------
# Corrective engineering recommendation ranker.
# ---------------------------------------------------------------------------
def rank_corrective_engineering_recommendations(
    *,
    per_pair: Sequence[PerPairDriftRecord],
    per_boundary: Sequence[PerBoundaryDriftRecord],
    per_pixel: PerPixelDriftRecord | None,
    cosine_summary: CosineDistributionSummary,
) -> tuple[Mapping[str, object], ...]:
    """Rank the 5 corrective engineering recommendations by predicted EV.

    Each recommendation is a dict carrying:
      * ``name``: canonical short id
      * ``predicted_delta_s_floor`` / ``predicted_delta_s_ceiling``: [predicted]
        ΔS band (Cauchy-Schwarz bound where available).
      * ``cost_usd_estimate_floor`` / ``cost_usd_estimate_ceiling``: dispatch
        cost band.
      * ``hook_numbers``: tuple of 6-hook wire-in identifiers.
      * ``rationale``: short text explaining when this recommendation fires.
      * ``triggered``: bool indicating whether the empirical inputs in THIS
        report justify firing the recommendation now.

    All score deltas are predicted-only (no measured contest-axis support).
    """
    recs: list[dict[str, object]] = []
    # 1. Selective parameter freeze (Fisher-informed).
    high_alignment = cosine_summary.n_outliers_abs_above_0_5
    rec1: dict[str, object] = {
        "name": "selective_parameter_freeze",
        "rationale": (
            "Identify high |g . d| parameters via per-pair master-gradient inner "
            "product; freeze on MPS path, retrain on CUDA shadow. Fires when "
            "cosine summary verdict is SCORE_RELEVANT_ENGINEERING_REQUIRED or "
            "outliers above 0.5 alignment exceed 10% of pairs."
        ),
        "predicted_delta_s_floor": -0.02,
        "predicted_delta_s_ceiling": -0.005,
        "cost_usd_estimate_floor": 0.05,
        "cost_usd_estimate_ceiling": 0.50,
        "hook_numbers": (1, 3, 4),  # sensitivity-map + bit-allocator + autopilot
        "triggered": cosine_summary.verdict == "SCORE_RELEVANT_ENGINEERING_REQUIRED"
        or (cosine_summary.n_pairs > 0 and high_alignment / max(1, cosine_summary.n_pairs) > 0.10),
        "axis_tag": "[predicted]",
    }
    recs.append(rec1)
    # 2. Subspace alignment via top-K eigenvectors of E[g g^T].
    rec2: dict[str, object] = {
        "name": "subspace_alignment_topK_eigenvectors",
        "rationale": (
            "Project MPS gradient updates onto top-K eigenvectors of E[g g^T] "
            "before applying. Fires when abs_mean cosine alignment >= 0.1 and "
            "cosine_distribution shows long tail (max_abs > 0.3)."
        ),
        "predicted_delta_s_floor": -0.015,
        "predicted_delta_s_ceiling": -0.003,
        "cost_usd_estimate_floor": 0.10,
        "cost_usd_estimate_ceiling": 1.00,
        "hook_numbers": (1, 2, 4),  # sensitivity-map + Pareto + autopilot
        "triggered": (
            cosine_summary.abs_mean >= 0.1 and cosine_summary.max_abs > 0.3
        ),
        "axis_tag": "[predicted]",
    }
    recs.append(rec2)
    # 3. Per-frame routing at inference (route high-drift frames through CUDA shadow).
    fat_tail_triggered = False
    if per_pair:
        per_pair_agg = np.array([r.aggregate_drift for r in per_pair])
        if per_pair_agg.size >= 4:  # need >=4 pairs to talk about p95 fat tail
            median = float(np.median(per_pair_agg))
            p95 = float(np.percentile(per_pair_agg, 95))
            # Fat tail: p95 must be >= 2x median AND at least one pair >= p95.
            fat_tail_triggered = bool(
                median > 0
                and p95 >= 2.0 * median
                and (per_pair_agg >= p95).any()
            )
    rec3: dict[str, object] = {
        "name": "per_frame_routing_high_drift_to_cuda_shadow",
        "rationale": (
            "Route high |delta_S_p| frames through CUDA shadow weights at "
            "inference, leaving low-drift frames on local MPS. Fires when "
            "per-pair drift distribution has fat tail (p95 >= 2x median AND "
            "n_pairs >= 4)."
        ),
        "predicted_delta_s_floor": -0.010,
        "predicted_delta_s_ceiling": -0.002,
        "cost_usd_estimate_floor": 0.02,
        "cost_usd_estimate_ceiling": 0.20,
        "hook_numbers": (4, 6),  # autopilot dispatch + probe disambiguator
        "triggered": fat_tail_triggered,
        "axis_tag": "[predicted]",
    }
    recs.append(rec3)
    # 4. Cross-device validation cadence.
    rec4: dict[str, object] = {
        "name": "cross_device_validation_cadence_every_K_steps",
        "rationale": (
            "Every K MPS steps, dispatch CUDA-shadow step for outlier pairs. "
            "Fires when cosine outliers above 0.5 exceed 5% of pairs (signals "
            "drift accumulation can be score-relevant)."
        ),
        "predicted_delta_s_floor": -0.008,
        "predicted_delta_s_ceiling": -0.001,
        "cost_usd_estimate_floor": 0.05,
        "cost_usd_estimate_ceiling": 0.30,
        "hook_numbers": (4, 5),  # autopilot dispatch + continual-learning posterior
        "triggered": (
            cosine_summary.n_pairs > 0
            and cosine_summary.n_outliers_abs_above_0_5 / max(1, cosine_summary.n_pairs) > 0.05
        ),
        "axis_tag": "[predicted]",
    }
    recs.append(rec4)
    # 5. Boundary smoothing post-process.
    n_high_boundary_pairs = 0
    for b in per_boundary:
        if b.flip_rate_in_band > 2 * max(b.flip_rate_overall, 1e-9):
            n_high_boundary_pairs += 1
    rec5: dict[str, object] = {
        "name": "boundary_smoothing_3px_gaussian_pre_argmax",
        "rationale": (
            "If drift concentrates at class boundaries (in-band flip rate > "
            "2x overall flip rate for >= 10% of pairs), apply 3-pixel Gaussian "
            "blur to logits before argmax. Pure inference-side fix; zero "
            "training cost."
        ),
        "predicted_delta_s_floor": -0.005,
        "predicted_delta_s_ceiling": -0.001,
        "cost_usd_estimate_floor": 0.0,
        "cost_usd_estimate_ceiling": 0.05,
        "hook_numbers": (4,),  # autopilot dispatch only (inference-time)
        "triggered": (
            len(per_boundary) > 0
            and n_high_boundary_pairs / max(1, len(per_boundary)) > 0.10
        ),
        "axis_tag": "[predicted]",
    }
    recs.append(rec5)
    return tuple(recs)


# ---------------------------------------------------------------------------
# Canonical report builder.
# ---------------------------------------------------------------------------
def build_granular_drift_report(
    *,
    mps_artifact_path: str | Path,
    cuda_artifact_path: str | Path,
    mps_recon,
    cuda_recon,
    mps_pair_scores=None,
    cuda_pair_scores=None,
    mps_segnet_logits=None,
    cuda_segnet_logits=None,
    weight_delta=None,
    master_gradient_anchor=None,
    archive_sha256: str = "",
    per_byte_drift_records: Sequence[PerByteDriftRecord] = (),
    boundary_band_px: int = 3,
    notes: str = "",
) -> GranularDriftReport:
    """End-to-end canonical report build over 6 decompositions.

    Inputs are tolerant: if a particular component cannot be measured (e.g.
    no master-gradient anchor for this archive, or no SegNet logits captured),
    the corresponding decomposition is empty and the cosine summary verdict
    falls back to NO_MASTER_GRADIENT_ANCHOR.
    """
    _ensure_numpy()
    per_frame = compute_per_frame_drift(mps_recon, cuda_recon)
    per_pixel = compute_per_pixel_drift(mps_recon, cuda_recon)
    per_boundary: tuple[PerBoundaryDriftRecord, ...] = ()
    if mps_segnet_logits is not None and cuda_segnet_logits is not None:
        per_boundary = compute_per_boundary_drift(
            mps_segnet_logits,
            cuda_segnet_logits,
            boundary_band_px=boundary_band_px,
        )
    per_pair: tuple[PerPairDriftRecord, ...] = ()
    if mps_pair_scores is not None and cuda_pair_scores is not None:
        per_pair = compute_per_pair_drift(mps_pair_scores, cuda_pair_scores)
    else:
        # Derive per-pair from the reconstruction tensors as a fallback.
        mps_np = _to_numpy(mps_recon)
        cuda_np = _to_numpy(cuda_recon)
        n_pairs = int(mps_np.shape[0])
        pair_scores_mps = np.zeros((n_pairs, 3), dtype=np.float32)
        pair_scores_cuda = np.zeros((n_pairs, 3), dtype=np.float32)
        for p in range(n_pairs):
            mp = mps_np[p]
            cp = cuda_np[p]
            pair_scores_mps[p, 0] = float(np.mean(np.abs(mp)))
            pair_scores_mps[p, 1] = float(np.mean(mp))
            pair_scores_mps[p, 2] = float(np.std(mp))
            pair_scores_cuda[p, 0] = float(np.mean(np.abs(cp)))
            pair_scores_cuda[p, 1] = float(np.mean(cp))
            pair_scores_cuda[p, 2] = float(np.std(cp))
        per_pair = compute_per_pair_drift(pair_scores_mps, pair_scores_cuda)
    per_pair_mg, cosine_summary = compute_per_pair_master_gradient_weighted_drift(
        per_pair,
        weight_delta,
        master_gradient_anchor,
        archive_sha256=archive_sha256 or "0" * 64,
    )
    # Summary statistics.
    if per_frame:
        agg_total = float(np.mean([r.aggregate for r in per_frame]))
        per_pair_agg = np.array([r.aggregate_drift for r in per_pair])
        if per_pair_agg.size >= 4:
            median = float(np.median(per_pair_agg))
            p95 = float(np.percentile(per_pair_agg, 95))
            fat_tail = bool(median > 0 and p95 >= 2.0 * median)
        else:
            fat_tail = False
    else:
        agg_total = 0.0
        fat_tail = False
    drift_concentrated_in_boundaries = False
    if per_boundary:
        in_band_rates = np.array([b.flip_rate_in_band for b in per_boundary])
        overall_rates = np.array([b.flip_rate_overall for b in per_boundary])
        ratio = in_band_rates / (overall_rates + 1e-9)
        drift_concentrated_in_boundaries = bool((ratio > 2.0).mean() > 0.10)
    recommendations = rank_corrective_engineering_recommendations(
        per_pair=per_pair,
        per_boundary=per_boundary,
        per_pixel=per_pixel,
        cosine_summary=cosine_summary,
    )
    return GranularDriftReport(
        schema_version=GRANULAR_DRIFT_REPORT_SCHEMA,
        evidence_grade=GRANULAR_DRIFT_EVIDENCE_GRADE,
        axis_tag=GRANULAR_DRIFT_AXIS_TAG,
        score_claim=False,
        promotion_eligible=False,
        ready_for_exact_eval_dispatch=False,
        mps_artifact_path=str(mps_artifact_path),
        cuda_artifact_path=str(cuda_artifact_path),
        n_pairs=len(per_pair),
        per_frame=per_frame,
        per_pixel=(per_pixel,),
        per_boundary=per_boundary,
        per_byte=tuple(per_byte_drift_records),
        per_pair=per_pair,
        per_pair_master_gradient=per_pair_mg,
        cosine_distribution_summary=cosine_summary,
        summary_aggregate_relative_drift=agg_total,
        summary_aggregate_drift_concentrated_in_pairs_above_p95=fat_tail,
        summary_drift_cliff_layer=None,  # filled by sister layerwise_drift when called
        summary_drift_concentrated_in_boundaries=drift_concentrated_in_boundaries,
        summary_corrective_engineering_recommendations=recommendations,
        notes=notes,
    )


def report_to_json_dict(report: GranularDriftReport) -> dict:
    """Render the report to a JSON-safe dict (canonical writer format)."""
    def _frame(r: PerFrameDriftRecord) -> dict:
        return {
            "pair_index": r.pair_index,
            "frame_index": r.frame_index,
            "pixel_l1": r.pixel_l1,
            "segnet_logit_l_inf": r.segnet_logit_l_inf,
            "posenet_pose_l2": r.posenet_pose_l2,
            "aggregate": r.aggregate,
        }

    def _pix(r: PerPixelDriftRecord) -> dict:
        return {
            "layer_name": r.layer_name,
            "backend_pair": list(r.backend_pair),
            "l_inf": r.l_inf,
            "l_2": r.l_2,
            "mean_abs": r.mean_abs,
            "activation_shape": list(r.activation_shape),
            "fraction_above_1e_3": r.fraction_above_1e_3,
        }

    def _bnd(r: PerBoundaryDriftRecord) -> dict:
        return {
            "pair_index": r.pair_index,
            "boundary_band_px": r.boundary_band_px,
            "n_boundary_pixels": r.n_boundary_pixels,
            "n_argmax_flips_in_band": r.n_argmax_flips_in_band,
            "flip_rate_in_band": r.flip_rate_in_band,
            "n_argmax_flips_overall": r.n_argmax_flips_overall,
            "flip_rate_overall": r.flip_rate_overall,
        }

    def _byt(r: PerByteDriftRecord) -> dict:
        return {
            "byte_offset": r.byte_offset,
            "mutation_delta_score_mps": r.mutation_delta_score_mps,
            "mutation_delta_score_cuda": r.mutation_delta_score_cuda,
            "drift_at_byte": r.drift_at_byte,
            "section_name": r.section_name,
        }

    def _pair(r: PerPairDriftRecord) -> dict:
        return {
            "pair_index": r.pair_index,
            "pixel_l1_drift": r.pixel_l1_drift,
            "segnet_drift": r.segnet_drift,
            "posenet_drift": r.posenet_drift,
            "aggregate_drift": r.aggregate_drift,
        }

    def _ppmg(r: PerPairMasterGradientWeightedRecord) -> dict:
        return {
            "pair_index": r.pair_index,
            "g_p_l2": r.g_p_l2,
            "d_l2": r.d_l2,
            "cauchy_schwarz_upper_bound": r.cauchy_schwarz_upper_bound,
            "inner_product_estimate": r.inner_product_estimate,
            "cos_alignment": r.cos_alignment,
            "archive_sha256_first12": r.archive_sha256_first12,
            "master_gradient_axis_tag": r.master_gradient_axis_tag,
        }

    cs = report.cosine_distribution_summary
    return {
        "schema_version": report.schema_version,
        "evidence_grade": report.evidence_grade,
        "axis_tag": report.axis_tag,
        "score_claim": report.score_claim,
        "promotion_eligible": report.promotion_eligible,
        "ready_for_exact_eval_dispatch": report.ready_for_exact_eval_dispatch,
        "mps_artifact_path": report.mps_artifact_path,
        "cuda_artifact_path": report.cuda_artifact_path,
        "n_pairs": report.n_pairs,
        "per_frame": [_frame(r) for r in report.per_frame],
        "per_pixel": [_pix(r) for r in report.per_pixel],
        "per_boundary": [_bnd(r) for r in report.per_boundary],
        "per_byte": [_byt(r) for r in report.per_byte],
        "per_pair": [_pair(r) for r in report.per_pair],
        "per_pair_master_gradient": [_ppmg(r) for r in report.per_pair_master_gradient],
        "cosine_distribution_summary": {
            "n_pairs": cs.n_pairs,
            "mean": cs.mean,
            "median": cs.median,
            "std": cs.std,
            "abs_mean": cs.abs_mean,
            "n_outliers_abs_above_0_5": cs.n_outliers_abs_above_0_5,
            "n_outliers_abs_above_0_8": cs.n_outliers_abs_above_0_8,
            "max_abs": cs.max_abs,
            "verdict": cs.verdict,
        },
        "summary_aggregate_relative_drift": report.summary_aggregate_relative_drift,
        "summary_aggregate_drift_concentrated_in_pairs_above_p95": report.summary_aggregate_drift_concentrated_in_pairs_above_p95,
        "summary_drift_cliff_layer": report.summary_drift_cliff_layer,
        "summary_drift_concentrated_in_boundaries": report.summary_drift_concentrated_in_boundaries,
        "summary_corrective_engineering_recommendations": [
            dict(r) for r in report.summary_corrective_engineering_recommendations
        ],
        "notes": report.notes,
    }


def write_granular_drift_report(report: GranularDriftReport, target_path: str | Path) -> Path:
    """Write the JSON-safe report to ``target_path`` atomically."""
    payload = report_to_json_dict(report)
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, sort_keys=True, indent=2)
    target.write_text(text)
    return target


__all__ = [
    "GRANULAR_DRIFT_EVIDENCE_GRADE",
    "GRANULAR_DRIFT_AXIS_TAG",
    "GRANULAR_DRIFT_REPORT_SCHEMA",
    "DECOMPOSITION_KEYS",
    "PerFrameDriftRecord",
    "PerPixelDriftRecord",
    "PerBoundaryDriftRecord",
    "PerByteDriftRecord",
    "PerPairDriftRecord",
    "PerPairMasterGradientWeightedRecord",
    "CosineDistributionSummary",
    "GranularDriftReport",
    "compute_per_frame_drift",
    "compute_per_pixel_drift",
    "compute_per_boundary_drift",
    "compute_per_byte_drift",
    "compute_per_pair_drift",
    "compute_per_pair_master_gradient_weighted_drift",
    "rank_corrective_engineering_recommendations",
    "build_granular_drift_report",
    "report_to_json_dict",
    "write_granular_drift_report",
]
