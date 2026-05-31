# SPDX-License-Identifier: MIT
"""Canonical multi-granularity contest-score sensitivity decomposition.

This module raises the signal for mathematically-optimal substrate optimization by
decomposing contest-score sensitivity along TWO orthogonal axes simultaneously:

  * the SCORE axis  (seg / pose / rate), grounded in the exact contest marginals
    ``∂S/∂d_seg = 100``, ``∂S/∂d_pose = 5/sqrt(10·d_pose)``, ``∂S/∂byte = 25/N``
    (``tac.master_gradient.compute_marginal_coefficients``); and
  * the GRANULARITY axis  (byte → frame → pair → region → boundary), because the
    contest scores 600 non-overlapping pairs / 1200 frames and ``d_seg`` is a
    per-pixel argmax-disagreement RATE whose signal concentrates at SegNet class
    boundaries (``upstream/modules.py`` ``SegNet.compute_distortion`` :
    ``(out1.argmax(1) != out2.argmax(1)).float().mean()``).

[verified-against: tac.master_gradient.compute_marginal_coefficients canonical
contest-score marginals] [verified-against: upstream.modules SegNet.compute_distortion
argmax-disagreement-rate + PoseNet.compute_distortion MSE-on-first-half-pose-dims]

Three classes of output, each carrying canonical Provenance per Catalog #323 / #341 /
#192 / #127 (every score-relevant number is NON-PROMOTABLE ``[predicted]`` /
``[macOS-MLX research-signal]`` evidence — it is a sensitivity decomposition, NOT a
contest-score claim):

  1. REAL-MEASURED ($0, no GPU): :func:`byte_axis_sensitivity_from_master_gradient`
     reads the existing ``(n_bytes, 3)`` per-byte axis-decomposed gradient arrays
     already in the master-gradient ledger and produces per-axis byte-sensitivity
     concentration statistics. No forward pass, no fabricated number.

  2. PURE-MATH KERNELS (work on supplied tensors; caller owns the forward pass):
     :func:`segnet_boundary_band_weights` (the margin-based ``w_i = exp(-margin/τ)``
     boundary-band map that the optimal seg-distillation objective concentrates on),
     :func:`per_pose_dim_score_contribution` (the per-dim ``(Δpose_k)²·∂S/∂d_pose``
     contribution the optimal pose-distillation objective weights by), and
     :func:`per_pair_axis_score_contribution` (the per-pair seg/pose/rate score
     decomposition over the 600 contest pairs).

  3. DESIGNED-PENDING-MEASUREMENT: :func:`design_input_domain_sensitivity_measurement`
     returns the EXACT, reproducible recipe for the per-frame / per-pair / per-region
     input-domain SegNet/PoseNet-forward sensitivity that cannot be computed at $0
     (it needs a scorer forward pass). It fabricates NO numbers (Catalog #307) — it
     emits a ``PendingMeasurement`` with the command + reactivation criteria, tagged
     research-only-pending-measurement, so the operator-attended GPU slot can fill it.

The module is import-light: numpy + the master-gradient ledger reader at module load;
torch is imported lazily only inside the tensor kernels so reading byte-domain
sensitivity from anchors never requires torch.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tac.master_gradient import (
    CONTEST_RATE_DENOM_BYTES,
    OperatingPoint,
    compute_marginal_coefficients,
    latest_anchor_for_archive,
)

__all__ = [
    "GRANULARITIES",
    "SCORE_AXES",
    "AxisSensitivityStats",
    "BoundaryBandStats",
    "ByteAxisSensitivityReport",
    "MultiGranularitySensitivityError",
    "PendingMeasurement",
    "PerPairAxisContribution",
    "PoseDimContribution",
    "byte_axis_sensitivity_from_master_gradient",
    "design_input_domain_sensitivity_measurement",
    "non_promotable_provenance_dict",
    "per_pair_axis_score_contribution",
    "per_pose_dim_score_contribution",
    "segnet_boundary_band_weights",
]

#: Canonical contest score axes (mirrors ``tac.master_gradient.SCORE_AXIS_LABELS``
#: but ordered seg/pose/rate to match the master-gradient ``(n_bytes, 3)`` array).
SCORE_AXES: tuple[str, str, str] = ("seg", "pose", "rate")

#: Canonical granularity ladder. ``byte`` is real-measured from the ledger;
#: ``frame``/``pair`` are the contest's native units (1200 frames / 600 pairs);
#: ``region``/``boundary`` are the d_seg spatial-concentration units.
GRANULARITIES: tuple[str, ...] = ("byte", "frame", "pair", "region", "boundary")

#: Numerical floor so the hyperbolic pose marginal ``5/sqrt(10·d_pose)`` is finite.
_POSE_FLOOR = 1e-9


class MultiGranularitySensitivityError(ValueError):
    """Raised on malformed inputs / unreadable anchors (fail-closed)."""


# ---------------------------------------------------------------------------
# Provenance helper — every score-relevant number is non-promotable.
# ---------------------------------------------------------------------------
def non_promotable_provenance_dict(
    *,
    model_id: str,
    inputs_sha256: str,
    measurement_axis: str = "[predicted]",
    hardware_substrate: str = "unknown",
) -> dict[str, Any]:
    """Return the canonical NON-PROMOTABLE Provenance dict for a sensitivity row.

    Per Catalog #341 / #192 / #127 / #323 sensitivity decompositions are
    observability-only — ``promotion_eligible=False`` + ``score_claim_valid=False``
    by construction. Delegates to the canonical builder + serializer.
    """
    from tac.provenance.builders import build_provenance_for_predicted
    from tac.provenance.validator import provenance_to_dict

    prov = build_provenance_for_predicted(
        model_id=model_id,
        inputs_sha256=inputs_sha256,
        measurement_axis=measurement_axis,
        hardware_substrate=hardware_substrate,
    )
    return provenance_to_dict(prov)


# ---------------------------------------------------------------------------
# 1. REAL-MEASURED byte-domain axis sensitivity (from the ledger; $0, no GPU).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AxisSensitivityStats:
    """Per-axis concentration statistics of a byte-domain sensitivity vector.

    ``share`` is the axis' mean fraction of total marginal-weighted sensitivity.
    ``gini`` is the Gini concentration of |sensitivity| across bytes for this axis
    (0 = uniform, →1 = a few bytes carry all the signal — the bit-allocator's prior).
    ``top_decile_mass`` is the fraction of total |sensitivity| held by the most
    sensitive 10% of bytes.
    """

    axis: str
    share: float
    gini: float
    top_decile_mass: float
    dominant_byte_count: int
    max_abs: float


@dataclass(frozen=True)
class ByteAxisSensitivityReport:
    """REAL-measured byte × axis sensitivity decomposition from a ledger anchor."""

    archive_sha256: str
    measurement_axis: str
    n_bytes: int
    operating_point: dict[str, float]
    marginal_coefficients: dict[str, float]
    per_axis: tuple[AxisSensitivityStats, ...]
    gradient_array_path: str
    provenance: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "archive_sha256": self.archive_sha256,
            "measurement_axis": self.measurement_axis,
            "n_bytes": self.n_bytes,
            "operating_point": dict(self.operating_point),
            "marginal_coefficients": dict(self.marginal_coefficients),
            "per_axis": [
                {
                    "axis": a.axis,
                    "share": a.share,
                    "gini": a.gini,
                    "top_decile_mass": a.top_decile_mass,
                    "dominant_byte_count": a.dominant_byte_count,
                    "max_abs": a.max_abs,
                }
                for a in self.per_axis
            ],
            "gradient_array_path": self.gradient_array_path,
            "provenance": dict(self.provenance),
            "schema": "multi_granularity_byte_axis_sensitivity_v1",
        }


def _gini(values: np.ndarray) -> float:
    """Gini concentration coefficient of |values| in [0, 1] (0=uniform)."""
    v = np.abs(np.asarray(values, dtype=np.float64)).ravel()
    if v.size == 0:
        return 0.0
    total = float(v.sum())
    if total <= 0.0:
        return 0.0
    v_sorted = np.sort(v)
    n = v_sorted.size
    # Gini = (2·Σ i·x_i)/(n·Σ x_i) − (n+1)/n
    idx = np.arange(1, n + 1, dtype=np.float64)
    g = (2.0 * float((idx * v_sorted).sum())) / (n * total) - (n + 1.0) / n
    return float(max(0.0, min(1.0, g)))


def _top_decile_mass(values: np.ndarray) -> float:
    v = np.abs(np.asarray(values, dtype=np.float64)).ravel()
    if v.size == 0:
        return 0.0
    total = float(v.sum())
    if total <= 0.0:
        return 0.0
    k = max(1, math.ceil(0.1 * v.size))
    top = np.sort(v)[-k:]
    return float(top.sum() / total)


def byte_axis_sensitivity_from_master_gradient(
    archive_sha256: str,
    *,
    axis: str | None = None,
    ledger_path: Path | None = None,
) -> ByteAxisSensitivityReport:
    """REAL-measured byte × (seg,pose,rate) sensitivity for an archive ($0, no GPU).

    Reads the existing ``(n_bytes, 3)`` per-byte axis-decomposed master-gradient
    array (already on disk in the ledger; no forward pass) and computes per-axis
    marginal-weighted concentration statistics. The sensitivity of byte ``b`` to
    axis ``a`` is ``|grad[b,a]| · ∂S/∂a`` at the anchor's operating point — i.e.
    the contest-score-faithful byte sensitivity, NOT the raw gradient.

    Args:
        archive_sha256: archive whose anchor to read.
        axis: optional contest axis filter (``[contest-CUDA]`` / ``[contest-CPU]`` /
            ``[macOS-CPU advisory]``); ``None`` = latest anchor of any axis.
        ledger_path: optional override for the master-gradient ledger.

    Returns:
        :class:`ByteAxisSensitivityReport` with REAL-measured per-axis stats +
        non-promotable Provenance.

    Raises:
        MultiGranularitySensitivityError: no anchor / no array / shape mismatch.
    """
    anchor = latest_anchor_for_archive(archive_sha256, axis=axis, path=ledger_path)
    if anchor is None:
        raise MultiGranularitySensitivityError(
            f"no master-gradient anchor for archive_sha256={archive_sha256[:12]} "
            f"axis={axis!r}; run tools/extract_master_gradient.py first"
        )
    arr_path = anchor.get("gradient_array_path")
    if not arr_path:
        raise MultiGranularitySensitivityError(
            f"anchor for {archive_sha256[:12]} has no gradient_array_path"
        )
    p = Path(arr_path)
    if not p.exists():
        raise MultiGranularitySensitivityError(
            f"gradient array missing on disk: {arr_path}"
        )
    grad = np.load(p)
    if grad.ndim != 2 or grad.shape[1] != 3:
        raise MultiGranularitySensitivityError(
            f"expected (n_bytes, 3) array; got {grad.shape!r} from {arr_path}"
        )
    op_raw = anchor.get("operating_point") or {}
    try:
        op = OperatingPoint(
            d_seg=float(op_raw.get("d_seg", 0.0)),
            d_pose=float(op_raw.get("d_pose", _POSE_FLOOR)),
            rate=float(op_raw.get("rate", 0.0)),
            score=float(op_raw.get("score", 0.0)),
        )
    except Exception as exc:  # malformed operating point in the anchor
        raise MultiGranularitySensitivityError(
            f"anchor operating_point invalid: {op_raw!r} ({exc})"
        ) from exc
    seg_m, pose_m, byte_m = compute_marginal_coefficients(op)
    marginals = np.array([seg_m, pose_m, byte_m * CONTEST_RATE_DENOM_BYTES], dtype=np.float64)
    # NOTE: the byte-domain rate marginal is per-byte (25/N); multiplying by N
    # converts to a per-unit-rate marginal so all three axes are comparable in
    # the share computation. Concentration stats (gini/top-decile) are
    # marginal-INVARIANT (a positive scalar does not change the |.|-ordering),
    # so they are computed on the RAW per-axis gradient column.
    n_bytes = int(grad.shape[0])
    # marginal-weighted |sensitivity| per axis, summed → share
    weighted = np.abs(grad.astype(np.float64)) * marginals[np.newaxis, :]
    axis_totals = weighted.sum(axis=0)
    grand = float(axis_totals.sum())
    per_axis: list[AxisSensitivityStats] = []
    for j, ax in enumerate(SCORE_AXES):
        col = grad[:, j].astype(np.float64)
        share = float(axis_totals[j] / grand) if grand > 0 else 0.0
        per_axis.append(
            AxisSensitivityStats(
                axis=ax,
                share=share,
                gini=_gini(col),
                top_decile_mass=_top_decile_mass(col),
                dominant_byte_count=int((np.abs(col) > 0).sum()),
                max_abs=float(np.abs(col).max()) if col.size else 0.0,
            )
        )
    prov = non_promotable_provenance_dict(
        model_id="multi_granularity_byte_axis_sensitivity",
        inputs_sha256=str(anchor.get("gradient_subject_sha256") or archive_sha256),
        measurement_axis="[predicted]",
        hardware_substrate=str(anchor.get("measurement_hardware") or "unknown"),
    )
    return ByteAxisSensitivityReport(
        archive_sha256=archive_sha256,
        measurement_axis=str(anchor.get("measurement_axis") or "unknown"),
        n_bytes=n_bytes,
        operating_point=op.as_dict() if hasattr(op, "as_dict") else dict(op_raw),
        marginal_coefficients={"seg": seg_m, "pose": pose_m, "rate_per_byte": byte_m},
        per_axis=tuple(per_axis),
        gradient_array_path=str(arr_path),
        provenance=prov,
    )


# ---------------------------------------------------------------------------
# 2. PURE-MATH KERNELS (work on supplied tensors; caller owns the forward pass).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PerPairAxisContribution:
    """Per-pair seg/pose/rate score contribution over the contest's 600 pairs."""

    pair_index: int
    d_seg: float
    d_pose: float
    seg_score_contribution: float  # 100 · d_seg
    pose_score_contribution: float  # sqrt(10 · d_pose)
    dominant_axis: str


def per_pair_axis_score_contribution(
    d_seg_per_pair: Sequence[float] | np.ndarray,
    d_pose_per_pair: Sequence[float] | np.ndarray,
) -> list[PerPairAxisContribution]:
    """Decompose the contest score into per-pair seg + pose contributions.

    Pure contest-score math (``upstream/modules.py``): the contest sums seg over
    pairs as ``100·d_seg`` and pose as ``sqrt(10·d_pose)``. This identifies WHICH
    of the 600 pairs dominate each axis — the per-pair signal the operator asked
    for. ``d_seg``/``d_pose`` are caller-supplied (from a scorer forward pass);
    this kernel does the score-faithful decomposition, fabricating no numbers.

    Note: ``sqrt(10·d_pose)`` is SUPER-ADDITIVE across pairs only if pose is summed
    pre-sqrt; the contest pose term is ``sqrt(10·mean_pairs(d_pose))`` at the video
    level, so per-pair ``sqrt(10·d_pose)`` here is the pair-local pose magnitude
    used for RANKING which pairs carry pose signal, not an additive decomposition.
    """
    ds = np.asarray(d_seg_per_pair, dtype=np.float64).ravel()
    dp = np.asarray(d_pose_per_pair, dtype=np.float64).ravel()
    if ds.shape != dp.shape:
        raise MultiGranularitySensitivityError(
            f"d_seg_per_pair {ds.shape} and d_pose_per_pair {dp.shape} must match"
        )
    out: list[PerPairAxisContribution] = []
    for i in range(ds.size):
        seg_c = 100.0 * float(ds[i])
        pose_c = math.sqrt(10.0 * max(float(dp[i]), 0.0))
        dominant = "seg" if seg_c >= pose_c else "pose"
        out.append(
            PerPairAxisContribution(
                pair_index=i,
                d_seg=float(ds[i]),
                d_pose=float(dp[i]),
                seg_score_contribution=seg_c,
                pose_score_contribution=pose_c,
                dominant_axis=dominant,
            )
        )
    return out


@dataclass(frozen=True)
class PoseDimContribution:
    """Per-pose-dimension score contribution for d_pose (the AIL-weight source)."""

    dim_index: int
    delta_sq: float  # (student_pose_k − teacher_pose_k)²
    in_contest_window: bool  # k < out//2 (only these enter the contest)
    score_contribution: float  # delta_sq · ∂S/∂d_pose, 0 outside window


def per_pose_dim_score_contribution(
    student_pose: Sequence[float] | np.ndarray,
    teacher_pose: Sequence[float] | np.ndarray,
    *,
    contest_window_dims: int | None = None,
    d_pose_running: float | None = None,
) -> list[PoseDimContribution]:
    """Per-dim score contribution of a pose error (which of the 6 dims dominate).

    Contest pose distortion is ``mean_k((Δpose_k)²)`` over ``k < out//2``. The
    score contribution of dim ``k`` is ``(Δpose_k)² · ∂S/∂d_pose`` where
    ``∂S/∂d_pose = 5/sqrt(10·d_pose)`` (the per-pair hyperbolic marginal). Dims at
    or beyond the contest window contribute 0 (the scorer ignores them). This is
    the empirical source of the optimal pose-distillation per-dim weight (§2 of the
    teacher design memo): weight dim ``k`` by its measured score contribution, not
    uniformly.
    """
    s = np.asarray(student_pose, dtype=np.float64).ravel()
    t = np.asarray(teacher_pose, dtype=np.float64).ravel()
    if s.shape != t.shape:
        raise MultiGranularitySensitivityError(
            f"student_pose {s.shape} and teacher_pose {t.shape} must match"
        )
    n = s.size
    window = contest_window_dims if contest_window_dims is not None else n // 2
    if window < 0 or window > n:
        raise MultiGranularitySensitivityError(
            f"contest_window_dims={window} out of range for {n} pose dims"
        )
    if d_pose_running is None:
        # use the realized d_pose over the contest window as the operating point
        in_win = (s[:window] - t[:window]) ** 2
        d_pose_running = float(in_win.mean()) if window > 0 else _POSE_FLOOR
    pose_marginal = 5.0 / math.sqrt(10.0 * max(float(d_pose_running), _POSE_FLOOR))
    out: list[PoseDimContribution] = []
    for k in range(n):
        dsq = float((s[k] - t[k]) ** 2)
        in_win = k < window
        out.append(
            PoseDimContribution(
                dim_index=k,
                delta_sq=dsq,
                in_contest_window=in_win,
                score_contribution=(dsq * pose_marginal) if in_win else 0.0,
            )
        )
    return out


@dataclass(frozen=True)
class BoundaryBandStats:
    """Statistics of the margin-based SegNet boundary-band weight map."""

    boundary_band_fraction: float  # fraction of pixels with w_i above the band threshold
    mean_weight: float
    weight_gini: float  # concentration of w_i (how tight is the band)
    n_pixels: int
    tau: float
    margin_threshold: float | None


def segnet_boundary_band_weights(
    teacher_logits: Any,
    *,
    tau: float = 1.0,
    margin_threshold: float | None = None,
) -> tuple[Any, BoundaryBandStats]:
    """Margin-based SegNet boundary-band weight map ``w_i = exp(-margin_i/τ)``.

    This is the score-faithful spatial weight the OPTIMAL seg-distillation objective
    concentrates on (teacher design memo §1.4). ``d_seg`` is the per-pixel
    argmax-disagreement RATE, so a rendering perturbation can only move ``d_seg`` at a
    pixel whose top-2 teacher-logit margin ``m_i = z_(1) − z_(2)`` is small (the
    decision-boundary band). The weight ``w_i = exp(-m_i/τ)`` is ≈1 on the boundary
    and →0 in confident interiors — exactly where ``∂d_seg/∂rendered ≠ 0``.

    PURE TENSOR MATH: the caller supplies ``teacher_logits`` (e.g. the real SegNet
    logits on a decoded frame — that forward pass needs a GPU/CPU scorer; THIS kernel
    does not). Accepts torch or numpy ``(B, C, H, W)`` or ``(C, H, W)``.

    Args:
        teacher_logits: per-pixel class logits, channel axis = classes.
        tau: soft-band temperature; smaller = tighter band.
        margin_threshold: if given, ``boundary_band_fraction`` counts pixels with
            margin < this threshold (hard band); else counts ``w_i > exp(-1)`` (the
            ``m_i < τ`` soft-band).

    Returns:
        ``(weights, BoundaryBandStats)`` — ``weights`` same backend/shape as the
        spatial map (B,H,W) or (H,W); stats carries the band fraction + concentration.
    """
    import torch  # lazy: only the kernels need torch

    if tau <= 0.0:
        raise MultiGranularitySensitivityError(f"tau must be > 0; got {tau}")
    if isinstance(teacher_logits, np.ndarray):
        logits = torch.from_numpy(teacher_logits.astype(np.float32))
    elif torch.is_tensor(teacher_logits):
        logits = teacher_logits.detach().float()
    else:
        raise MultiGranularitySensitivityError(
            "teacher_logits must be a torch tensor or numpy array"
        )
    if logits.ndim == 3:
        logits = logits.unsqueeze(0)  # (1, C, H, W)
    if logits.ndim != 4:
        raise MultiGranularitySensitivityError(
            f"expected (C,H,W) or (B,C,H,W) logits; got ndim={logits.ndim}"
        )
    if logits.shape[1] < 2:
        raise MultiGranularitySensitivityError(
            f"need >=2 classes for a top-2 margin; got C={logits.shape[1]}"
        )
    # top-2 margin per pixel
    top2 = torch.topk(logits, k=2, dim=1).values  # (B, 2, H, W)
    margin = (top2[:, 0, :, :] - top2[:, 1, :, :]).clamp_min(0.0)  # (B, H, W)
    weights = torch.exp(-margin / tau)  # (B, H, W) in (0, 1]
    if margin_threshold is not None:
        band = (margin < float(margin_threshold)).float()
    else:
        # soft band: m_i < tau  ⇔  w_i > exp(-1)
        band = (weights > math.exp(-1.0)).float()
    w_np = weights.reshape(-1).cpu().numpy()
    stats = BoundaryBandStats(
        boundary_band_fraction=float(band.mean().item()),
        mean_weight=float(weights.mean().item()),
        weight_gini=_gini(w_np),
        n_pixels=int(weights.numel()),
        tau=float(tau),
        margin_threshold=margin_threshold,
    )
    out_weights = weights.squeeze(0) if weights.shape[0] == 1 else weights
    return out_weights, stats


# ---------------------------------------------------------------------------
# 3. DESIGNED-PENDING-MEASUREMENT (no fabricated numbers; Catalog #307).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PendingMeasurement:
    """A reproducible input-domain sensitivity measurement that needs a scorer forward.

    Fabricates NO numbers — it is the EXACT recipe + reactivation criteria so the
    operator-attended GPU/CPU-scorer slot can fill it. Per CLAUDE.md "Substrate
    scaffolds MUST be COMPLETE or RESEARCH-ONLY" + Catalog #307: a sensitivity that
    needs a forward pass we cannot do at $0 is research-only-pending-measurement.
    """

    granularity: str
    score_axis: str
    description: str
    measurement_recipe: tuple[str, ...]
    requires_forward_pass: bool
    reactivation_criteria: str
    research_only: bool = True
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "granularity": self.granularity,
            "score_axis": self.score_axis,
            "description": self.description,
            "measurement_recipe": list(self.measurement_recipe),
            "requires_forward_pass": self.requires_forward_pass,
            "reactivation_criteria": self.reactivation_criteria,
            "research_only": self.research_only,
            "provenance": dict(self.provenance),
            "schema": "multi_granularity_pending_measurement_v1",
        }


def design_input_domain_sensitivity_measurement(
    granularity: str,
    score_axis: str,
    *,
    video_path: str = "upstream/videos/0.mkv",
) -> PendingMeasurement:
    """Emit the EXACT recipe for an input-domain (pixel/region/boundary/frame/pair)
    sensitivity measurement that needs a SegNet/PoseNet forward pass.

    No fabricated numbers. The recipe is reproducible and reactivation-pinned so the
    GPU-holding (operator-attended) slot fills the actual numbers.
    """
    if granularity not in GRANULARITIES:
        raise MultiGranularitySensitivityError(
            f"granularity must be one of {GRANULARITIES}; got {granularity!r}"
        )
    if score_axis not in SCORE_AXES:
        raise MultiGranularitySensitivityError(
            f"score_axis must be one of {SCORE_AXES}; got {score_axis!r}"
        )
    prov = non_promotable_provenance_dict(
        model_id=f"pending_input_domain_{granularity}_{score_axis}_sensitivity",
        inputs_sha256="0" * 64,  # no inputs measured yet (pending)
        measurement_axis="[research-signal]",
        hardware_substrate="unknown",
    )
    if granularity == "boundary" and score_axis == "seg":
        return PendingMeasurement(
            granularity="boundary",
            score_axis="seg",
            description=(
                "Per-pixel d_seg flip-sensitivity: for each pixel, the smallest "
                "rendering perturbation that flips the SegNet argmax (the empirical "
                "Yousfi cost-map). The boundary-band weight map (segnet_boundary_band_"
                "weights) is the PURE-MATH upper bound on WHERE flips can occur; the "
                "actual per-pixel flip cost needs a scorer forward + perturbation sweep."
            ),
            measurement_recipe=(
                f"1. decode pairs from {video_path} (tac.research.segnet_boundary_floor.decode_video + build_pairs)",
                "2. forward real SegNet (smp.Unet EfficientNet-B2) on the LAST frame of each pair → logits (B,5,H,W)",
                "3. w_i = segnet_boundary_band_weights(logits) → margin-based boundary band (this module, pure math)",
                "4. for each boundary pixel, sweep a small RGB perturbation ε and record min‖ε‖ that flips argmax → flip-cost map",
                "5. d_seg_sensitivity[pixel] = boundary_band_fraction-normalized inverse-flip-cost",
                "6. aggregate to region (e.g. 32x32 tiles) → region-seg-sensitivity for the bit-allocator prior",
            ),
            requires_forward_pass=True,
            reactivation_criteria=(
                "operator-attended GPU/CPU-scorer slot frees (sister z8 long-run done); "
                "run on upstream/videos/0.mkv; tag [macOS-MLX research-signal] / [contest-*] per device"
            ),
            provenance=prov,
        )
    if granularity in ("frame", "pair"):
        axis_desc = {
            "seg": "d_seg per frame/pair (argmax-disagreement rate of the last frame)",
            "pose": "d_pose per pair (MSE on first out//2 pose dims)",
            "rate": "per-frame/pair byte attribution (which frames' latents cost the most archive bytes)",
        }[score_axis]
        return PendingMeasurement(
            granularity=granularity,
            score_axis=score_axis,
            description=(
                f"Per-{granularity} {score_axis} sensitivity over the contest's "
                f"{'1200 frames' if granularity == 'frame' else '600 pairs'}: {axis_desc}. "
                "per_pair_axis_score_contribution / per_pose_dim_score_contribution (this "
                "module) do the score-faithful decomposition; the d_seg/d_pose inputs need "
                "a scorer forward on the reconstructed-vs-GT pairs."
            ),
            measurement_recipe=(
                f"1. decode GT pairs from {video_path}; reconstruct the candidate archive's pairs",
                "2. forward real SegNet+PoseNet on (GT, reconstructed) → d_seg/d_pose per pair",
                "3. per_pair_axis_score_contribution(d_seg, d_pose) → per-pair seg/pose dominance ranking (this module)",
                "4. (pose) per_pose_dim_score_contribution(student, teacher) per pair → per-dim dominance",
            ),
            requires_forward_pass=True,
            reactivation_criteria=(
                "operator-attended scorer slot frees; the contest scores 600 non-overlapping "
                "pairs / 1200 frames so this is the native-granularity sensitivity"
            ),
            provenance=prov,
        )
    return PendingMeasurement(
        granularity=granularity,
        score_axis=score_axis,
        description=(
            f"Per-{granularity} {score_axis} input-domain sensitivity. The byte-domain "
            "axis sensitivity IS available at $0 (byte_axis_sensitivity_from_master_"
            "gradient); the input-domain (pixel/region) decomposition needs a scorer forward."
        ),
        measurement_recipe=(
            f"forward the scorer on {video_path} pairs; decompose {score_axis} at {granularity} granularity",
        ),
        requires_forward_pass=True,
        reactivation_criteria="operator-attended scorer slot frees",
        provenance=prov,
    )
