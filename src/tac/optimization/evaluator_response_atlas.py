# SPDX-License-Identifier: MIT
"""EVALUATOR RESPONSE ATLAS ENGINE — the typed per-pair scorer-sensitivity INDEX
over the full 600-pair contest video.

The contest objective is an evaluator quotient
(``100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37_545_489``), so the binding
question for every byte that touches a frame pair is: *where, across the 600
pairs, does the scorer-fragile mass concentrate, and where is the free budget?*
Answering that requires the per-pair PoseNet pixel-Jacobian + SegNet argmax-flip
margin field for ALL 600 pairs — the missing piece this engine computes and
indexes.

WHAT THIS IS (and is NOT)
=========================
Per the orphan inventory ``.omx/research/evaluator_inverse_orphan_inventory_20260609.md``
(REUSE PLAN, task #36) the atlas is an **INDEX over existing surfaces**, NOT a
new registry:

* The per-pair score-effect quantities (seg margin field, pose Jacobian field,
  joint safe cone radius) are produced by the canonical #35 cone producer
  :func:`tac.optimization.frame1_joint_safe_cone.compute_frame1_joint_safe_cone`
  (real CPU-torch scorers + differentiable-YUV6 patch + fail-closed-on-zero
  guard). This engine ORCHESTRATES that producer across 600 pairs and SUMMARISES
  each pair into one typed row — it copies pointers + summary stats, never
  tensors. The full per-pixel fields live in the ``.npz`` cone maps the producer
  emits (which the LF waterfiller :mod:`tac.optimization.lf_payload_rate_distortion`
  ``Frame1ConeMap`` already reads).
* The atom score-unit record / admission law / atom vocabulary are the existing
  :mod:`tac.analysis.action_effect` IR +
  :mod:`tac.optimization.evaluator_action_waterfill` +
  :mod:`tac.contest_exploits` families. The atlas references them by family name;
  it does not re-author them.

So the atlas row is a join key: ``{pair_index, seg field stats, pose Jacobian
stats, joint cone summary, per-family sensitivity references (cone-map path,
spectral cell, exploit-atom family)}`` that the waterfiller (#46), rate-attack
repair campaign, and PR110++ selector-improvement consume to *target* the
highest-budget pairs/regions instead of sweeping uniformly.

MLX-FIRST (operator doctrine 2026-06-09)
========================================
The per-pair scorer forwards are REAL torch on the **exact CPU path** (NEVER
MPS — MPS corrupted 95.5% of the frontier's mode picks; any MPS-derived table is
contamination). The contest scorers (SegNet EfficientNet-B2 / PoseNet
FastViT-T12) only exist as torch modules, and the differentiable-YUV6
non-negotiable + fail-closed-zero-Jacobian guard live in the torch producer, so
the per-pair Jacobian/margin measurement uses **CPU-torch** (stated explicitly
per the apples-to-apples discipline).

The **cross-video reduction** — collapsing 600 per-pair field-statistic vectors
into the atlas headline (fragile-mass concentration, pose-binds fraction across
the video, top-K highest-budget pairs) — is the embarrassingly-parallel reduce
that saturates the M5 Max 128GB unified memory. That reduction is the MLX kernel
(:func:`atlas_cross_video_reduce`), with a **numpy reference**
(:func:`atlas_cross_video_reduce_numpy`) as the canonical contract per the
``canonical_kernels`` Backend pattern (Catalog #383). The two agree to fp32
tolerance; the MLX path runs on Apple unified memory, the numpy path is the
portability oracle.

EVIDENCE GRADE
==============
Every quantity is ``[macOS-CPU advisory]`` / ``[macOS-MLX research-signal]`` —
local macOS is NOT 1:1 contest hardware. Non-promotable per Catalog
#192/#341/#127/#323. The atlas proposes *where to spend bytes*; the full-video
exact DistortionNet replay on contest hardware ratifies. $0 local, NO cloud,
NO paid GPU, NO MPS.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

EVALUATOR_RESPONSE_ATLAS_SCHEMA = "evaluator_response_atlas.v1"
ATLAS_PAIR_ROW_SCHEMA = "evaluator_response_atlas_pair_row.v1"

# Canonical Tier A non-promotable false-authority markers (Catalog #341/#323).
# Local macOS is never 1:1 contest hardware; the atlas is an advisory index.
ATLAS_PROVENANCE: dict[str, Any] = {
    "evidence_grade": "macOS-CPU advisory",
    "axis_tag": "[macOS-CPU advisory]",
    "score_claim": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "promotable": False,
    "hardware_substrate": "local_macos_cpu",
}

# Canonical contest-exploit atom families the atlas indexes (the score-effect
# codebook vocabulary; the atlas references these by name, it does NOT re-author
# them — orphan inventory task #36 REUSE PLAN step 3). Each maps a sensitivity
# regime -> the canonical atom family whose mechanism exploits it.
ATLAS_EXPLOIT_ATOM_FAMILIES: tuple[str, ...] = (
    "per_class_chroma_anchor",
    "tropical_argmax_boundary_grammar",
    "decoy_mosaic_residual_basis",
    "stable_orbit_packet_diet",
    "a1_specialized_inverter",
    "precomputed_inference_outputs",
)


class EvaluatorResponseAtlasError(ValueError):
    """Raised on malformed atlas inputs / contract violations (fail-closed)."""


# ---------------------------------------------------------------------------
# Typed atlas row: an INDEX entry (pointers + summary stats; no copied tensors)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SegMarginFieldStats:
    """Reduced statistics of the per-pair SegNet argmax-flip margin field.

    The SegNet ``d_seg`` term is the per-pixel argmax-flip RATE; the top1-top2
    logit margin is the distance to flip. Small margin = fragile boundary mass.
    These stats summarise the 384x512 margin field into the join-key the atlas
    indexes (the full field is in the cone-map ``.npz``).
    """

    mean: float
    median: float
    p10: float
    p90: float
    min: float
    max: float
    # fraction of pixels with margin below half a uint8 step (fragile boundary).
    fragile_fraction: float
    # mean boundary slope ||d(margin)/d(frame1 pixel)|| (seg sensitivity).
    mean_boundary_slope: float


@dataclass(frozen=True)
class PoseJacobianFieldStats:
    """Reduced statistics of the per-pair PoseNet frame1-channel pixel-Jacobian
    norm field (the pose sensitivity / pose-null structure).

    PoseNet reads both frames; ``d_pose`` is MSE on the first ``out//2`` dims.
    The frame1-channel Jacobian norm is the pose response per unit frame1-pixel
    perturbation. Low norm = pose-null (free pose budget). The compute path
    fails closed if this field is identically zero (the YUV6-severed-gradient
    signature) so a non-reachable gradient can never masquerade as pose-null.
    """

    mean: float
    median: float
    p10: float
    p90: float
    min: float
    max: float
    l2_norm: float
    # fraction of pixels below pose_null_fraction * max (pose-null = free).
    pose_null_fraction: float
    compute_path: str  # "cpu_torch" — stated explicitly (NEVER mps)


@dataclass(frozen=True)
class JointConeSummary:
    """Summary of the #35 joint safe cone for this pair (consumes the producer).

    The cone radius is ``min(seg_margin_budget, pose_budget)`` — the per-pixel
    frame1 perturbation budget BOTH scorers permit. ``pair_budget`` is the
    integrated free budget the rate-attack can spend on this pair.
    """

    usable_budget_fraction: float
    empty_cone_fraction: float
    pose_binds_fraction: float
    seg_binds_fraction: float
    pose_null_fraction: float
    mean_radius_usable: float
    # the integrated per-pair budget = sum of usable radii (the rate-attack
    # ranking key: high pair_budget = where free bytes live for this pair).
    pair_budget: float
    pose_ail_gain: float


@dataclass(frozen=True)
class AtlasPairRow:
    """One typed atlas index row for one contest pair.

    An INDEX entry: pointers (cone-map path + producer sha) + reduced summary
    stats + per-family sensitivity references. NOT a tensor copy. The full
    per-pixel fields live in the referenced cone-map ``.npz`` (which the LF
    waterfiller already reads).
    """

    schema: str
    pair_index: int
    seg_margin_field_stats: SegMarginFieldStats
    pose_jacobian_norm_stats: PoseJacobianFieldStats
    joint_cone_summary: JointConeSummary
    # per-family sensitivity references (the atlas's atom vocabulary join):
    #   cone_map_path / cone_map_sha256 -> the spatial budget surface (#46 reads)
    #   spectral_cell_ref -> scorer_spectral_sensitivity.v2 band cell (advisory)
    #   exploit_atom_families -> canonical contest_exploits vocabulary
    sensitivity_refs: dict[str, Any]
    # per-SegNet-class region aggregates (region key -> radius stats).
    per_region: dict[int, dict[str, float]]
    provenance: dict[str, Any] = field(default_factory=lambda: dict(ATLAS_PROVENANCE))

    def to_json_obj(self) -> dict[str, Any]:
        """Serialise to a JSONL-safe dict (the persisted INDEX row)."""

        return {
            "schema": self.schema,
            "pair_index": self.pair_index,
            "seg_margin_field_stats": _dc_to_dict(self.seg_margin_field_stats),
            "pose_jacobian_norm_stats": _dc_to_dict(self.pose_jacobian_norm_stats),
            "joint_cone_summary": _dc_to_dict(self.joint_cone_summary),
            "sensitivity_refs": self.sensitivity_refs,
            "per_region": {str(k): v for k, v in self.per_region.items()},
            "provenance": self.provenance,
        }

    @classmethod
    def from_json_obj(cls, obj: Mapping[str, Any]) -> AtlasPairRow:
        return cls(
            schema=str(obj["schema"]),
            pair_index=int(obj["pair_index"]),
            seg_margin_field_stats=SegMarginFieldStats(**obj["seg_margin_field_stats"]),
            pose_jacobian_norm_stats=PoseJacobianFieldStats(**obj["pose_jacobian_norm_stats"]),
            joint_cone_summary=JointConeSummary(**obj["joint_cone_summary"]),
            sensitivity_refs=dict(obj["sensitivity_refs"]),
            per_region={int(k): dict(v) for k, v in obj.get("per_region", {}).items()},
            provenance=dict(obj.get("provenance", ATLAS_PROVENANCE)),
        )


def _dc_to_dict(dc: Any) -> dict[str, Any]:
    from dataclasses import asdict

    return asdict(dc)


# ---------------------------------------------------------------------------
# Build one atlas row from a #35 cone (reuse, not rebuild)
# ---------------------------------------------------------------------------


def build_atlas_row_from_cone(
    *,
    pair_index: int,
    cone: Any,
    cone_map_path: str | None = None,
    cone_map_sha256: str | None = None,
    compute_path: str = "cpu_torch",
    spectral_cell_ref: str | None = None,
    fragile_uint8_threshold: float = 0.5,
) -> AtlasPairRow:
    """Summarise one :class:`Frame1JointSafeCone` into a typed atlas index row.

    Reuses the producer's measured fields (``seg_margin``, ``seg_margin_budget``,
    ``pose_jacobian_norm``, ``joint_cone_radius``, ``fragile_cone_mask``,
    ``per_region``, ``summary``). No tensor is copied into the row — only reduced
    statistics + the pointer to the cone-map ``.npz`` (where the full fields
    live, read by the LF waterfiller).

    ``compute_path`` MUST be ``"cpu_torch"`` for promotion-honesty: the scorer
    forwards are the exact CPU path, NEVER MPS. The atlas refuses any other
    string except an explicit ``"numpy_synthetic"`` (test fixtures).
    """

    if compute_path not in ("cpu_torch", "numpy_synthetic"):
        raise EvaluatorResponseAtlasError(
            f"compute_path must be 'cpu_torch' (real scorer, never MPS) or "
            f"'numpy_synthetic' (test fixture); got {compute_path!r}"
        )

    seg_margin = np.asarray(cone.seg_margin, dtype=np.float64)
    pose_jac = np.asarray(cone.pose_jacobian_norm, dtype=np.float64)
    radius = np.asarray(cone.joint_cone_radius, dtype=np.float64)
    fragile = np.asarray(cone.fragile_cone_mask, dtype=bool)
    summary = dict(cone.summary)

    # mean boundary slope: derive from seg_budget = tol*margin/(slope+eps).
    # We have summary['mean_seg_boundary_slope'] from the producer; prefer it.
    mean_slope = float(summary.get("mean_seg_boundary_slope", 0.0))
    pose_null_frac = float(summary.get("pose_null_fraction", 0.0))

    seg_stats = SegMarginFieldStats(
        mean=float(seg_margin.mean()),
        median=float(np.median(seg_margin)),
        p10=float(np.percentile(seg_margin, 10)),
        p90=float(np.percentile(seg_margin, 90)),
        min=float(seg_margin.min()),
        max=float(seg_margin.max()),
        fragile_fraction=float((seg_margin < fragile_uint8_threshold).mean()),
        mean_boundary_slope=mean_slope,
    )
    pose_stats = PoseJacobianFieldStats(
        mean=float(pose_jac.mean()),
        median=float(np.median(pose_jac)),
        p10=float(np.percentile(pose_jac, 10)),
        p90=float(np.percentile(pose_jac, 90)),
        min=float(pose_jac.min()),
        max=float(pose_jac.max()),
        l2_norm=float(np.sqrt(np.square(pose_jac).sum())),
        pose_null_fraction=pose_null_frac,
        compute_path=compute_path,
    )
    # The integrated free budget = sum of usable (non-fragile) cone radii.
    usable = ~fragile
    pair_budget = float(radius[usable].sum()) if usable.any() else 0.0
    cone_summary = JointConeSummary(
        usable_budget_fraction=float(summary.get("usable_budget_fraction", float(usable.mean()))),
        empty_cone_fraction=float(summary.get("empty_cone_fraction", float(fragile.mean()))),
        pose_binds_fraction=float(summary.get("pose_binds_fraction", 0.0)),
        seg_binds_fraction=float(summary.get("seg_binds_fraction", 0.0)),
        pose_null_fraction=pose_null_frac,
        mean_radius_usable=float(summary.get("mean_radius_usable", 0.0)),
        pair_budget=pair_budget,
        pose_ail_gain=float(summary.get("pose_ail_gain", 0.0)),
    )

    sensitivity_refs: dict[str, Any] = {
        "cone_map_path": cone_map_path,
        "cone_map_sha256": cone_map_sha256,
        "cone_producer_schema": getattr(cone, "schema", None),
        # the scorer_spectral_sensitivity.v2 band cell this pair maps to
        # (advisory pointer; resolved by the LF waterfiller / spectral atlas).
        "spectral_cell_ref": spectral_cell_ref,
        # the canonical contest_exploits atom families whose mechanism this
        # pair's sensitivity regime admits (vocabulary reference, not a claim).
        "exploit_atom_families": list(ATLAS_EXPLOIT_ATOM_FAMILIES),
    }

    per_region = {
        int(k): {kk: float(vv) for kk, vv in v.items()}
        for k, v in dict(getattr(cone, "per_region", {})).items()
    }

    return AtlasPairRow(
        schema=ATLAS_PAIR_ROW_SCHEMA,
        pair_index=int(pair_index),
        seg_margin_field_stats=seg_stats,
        pose_jacobian_norm_stats=pose_stats,
        joint_cone_summary=cone_summary,
        sensitivity_refs=sensitivity_refs,
        per_region=per_region,
        provenance=dict(ATLAS_PROVENANCE),
    )


# ---------------------------------------------------------------------------
# Cross-video reduction kernel: MLX (unified memory) + numpy reference
# ---------------------------------------------------------------------------
# Catalog #383 canonical kernel contract: the numpy reference is the canonical
# oracle; the MLX forward runs on Apple unified memory and agrees to fp32 tol.


CROSS_VIDEO_REDUCE_FP32_ATOL = 1e-4


def _stack_pair_features(rows: Sequence[AtlasPairRow]) -> np.ndarray:
    """Stack per-pair scalar features into a ``(n_pairs, n_features)`` matrix.

    The features are the cross-video reduction inputs: the per-pair scalars the
    headline statistics reduce over. Deterministic feature order so the MLX +
    numpy paths reduce identical columns.
    """

    if not rows:
        raise EvaluatorResponseAtlasError("cannot reduce an empty atlas")
    feats = []
    for r in rows:
        feats.append(
            [
                r.joint_cone_summary.pair_budget,
                r.joint_cone_summary.usable_budget_fraction,
                r.joint_cone_summary.empty_cone_fraction,
                r.joint_cone_summary.pose_binds_fraction,
                r.joint_cone_summary.seg_binds_fraction,
                r.pose_jacobian_norm_stats.pose_null_fraction,
                r.seg_margin_field_stats.fragile_fraction,
                r.pose_jacobian_norm_stats.l2_norm,
                r.seg_margin_field_stats.mean_boundary_slope,
            ]
        )
    return np.asarray(feats, dtype=np.float64)


_FEATURE_NAMES: tuple[str, ...] = (
    "pair_budget",
    "usable_budget_fraction",
    "empty_cone_fraction",
    "pose_binds_fraction",
    "seg_binds_fraction",
    "pose_null_fraction",
    "fragile_fraction",
    "pose_jacobian_l2",
    "seg_boundary_slope",
)


def atlas_cross_video_reduce_numpy(rows: Sequence[AtlasPairRow]) -> dict[str, Any]:
    """Canonical numpy reference reduction over the per-pair feature matrix.

    Returns the cross-video headline: per-feature mean/std/min/max + the
    fragile-mass concentration (Gini of the fragile fraction) + the
    pose-binds-fraction across the video + the integrated free-budget
    distribution. This is the portability oracle the MLX forward matches.
    """

    feats = _stack_pair_features(rows)
    means = feats.mean(axis=0)
    stds = feats.std(axis=0)
    mins = feats.min(axis=0)
    maxs = feats.max(axis=0)

    per_feature = {
        name: {
            "mean": float(means[i]),
            "std": float(stds[i]),
            "min": float(mins[i]),
            "max": float(maxs[i]),
        }
        for i, name in enumerate(_FEATURE_NAMES)
    }

    # fragile-mass concentration: Gini of per-pair fragile fraction (how
    # unevenly the binding-constraint mass concentrates across pairs).
    fragile_col = feats[:, _FEATURE_NAMES.index("fragile_fraction")]
    budget_col = feats[:, _FEATURE_NAMES.index("pair_budget")]
    gini_fragile = _gini_numpy(fragile_col)
    gini_budget = _gini_numpy(budget_col)

    return {
        "n_pairs": int(feats.shape[0]),
        "per_feature": per_feature,
        "fragile_mass_gini": float(gini_fragile),
        "budget_concentration_gini": float(gini_budget),
        # pose-binds fraction across the WHOLE video = mean over pairs of the
        # per-pair pose-binds fraction (the CLAUDE.md marginal-value flip).
        "video_pose_binds_fraction": float(means[_FEATURE_NAMES.index("pose_binds_fraction")]),
        "video_usable_budget_fraction": float(means[_FEATURE_NAMES.index("usable_budget_fraction")]),
        "total_free_budget": float(budget_col.sum()),
        "reduce_path": "numpy_reference",
    }


def _gini_numpy(x: np.ndarray) -> float:
    """Gini concentration coefficient of a non-negative vector (numpy ref)."""

    a = np.sort(np.asarray(x, dtype=np.float64).ravel())
    n = a.size
    if n == 0:
        return 0.0
    s = a.sum()
    if s <= 0.0:
        return 0.0
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * (idx * a).sum()) / (n * s) - (n + 1.0) / n)


def atlas_cross_video_reduce_mlx(rows: Sequence[AtlasPairRow]) -> dict[str, Any]:
    """MLX (Apple unified memory) reduction over the per-pair feature matrix.

    Mathematically identical to :func:`atlas_cross_video_reduce_numpy` (matched
    to fp32 tolerance per the canonical_kernels Backend contract). Runs the
    column reductions + Gini sort on MLX arrays in the M5 Max 128GB unified
    memory. Raises if MLX is not installed (caller falls back to numpy).
    """

    import mlx.core as mx

    feats_np = _stack_pair_features(rows)
    feats = mx.array(feats_np.astype(np.float32))
    means = mx.mean(feats, axis=0)
    stds = mx.sqrt(mx.mean((feats - means) ** 2, axis=0))
    mins = mx.min(feats, axis=0)
    maxs = mx.max(feats, axis=0)
    mx.eval(means, stds, mins, maxs)

    means_np = np.asarray(means)
    stds_np = np.asarray(stds)
    mins_np = np.asarray(mins)
    maxs_np = np.asarray(maxs)

    per_feature = {
        name: {
            "mean": float(means_np[i]),
            "std": float(stds_np[i]),
            "min": float(mins_np[i]),
            "max": float(maxs_np[i]),
        }
        for i, name in enumerate(_FEATURE_NAMES)
    }

    fragile_col = feats[:, _FEATURE_NAMES.index("fragile_fraction")]
    budget_col = feats[:, _FEATURE_NAMES.index("pair_budget")]
    gini_fragile = _gini_mlx(fragile_col)
    gini_budget = _gini_mlx(budget_col)

    return {
        "n_pairs": int(feats_np.shape[0]),
        "per_feature": per_feature,
        "fragile_mass_gini": float(gini_fragile),
        "budget_concentration_gini": float(gini_budget),
        "video_pose_binds_fraction": float(means_np[_FEATURE_NAMES.index("pose_binds_fraction")]),
        "video_usable_budget_fraction": float(means_np[_FEATURE_NAMES.index("usable_budget_fraction")]),
        "total_free_budget": float(np.asarray(mx.sum(budget_col))),
        "reduce_path": "mlx_unified_memory",
    }


def _gini_mlx(x: Any) -> float:
    """Gini concentration via MLX sort + cumulative reduction (unified mem)."""

    import mlx.core as mx

    a = mx.sort(x)
    n = int(a.shape[0])
    if n == 0:
        return 0.0
    s = mx.sum(a)
    mx.eval(s)
    if float(np.asarray(s)) <= 0.0:
        return 0.0
    idx = mx.arange(1, n + 1, dtype=mx.float32)
    num = 2.0 * mx.sum(idx * a)
    mx.eval(num)
    return float(np.asarray(num) / (n * float(np.asarray(s))) - (n + 1.0) / n)


def atlas_cross_video_reduce(
    rows: Sequence[AtlasPairRow], *, prefer_mlx: bool = True
) -> dict[str, Any]:
    """Cross-video reduction with MLX-first + numpy-portable fallback.

    Per the operator MLX-first doctrine: try the MLX unified-memory kernel; if
    MLX is unavailable, fall back to the canonical numpy reference (the
    portability contract). The returned ``reduce_path`` states which ran.
    """

    if prefer_mlx:
        try:
            return atlas_cross_video_reduce_mlx(rows)
        except ImportError:
            pass
    return atlas_cross_video_reduce_numpy(rows)


# ---------------------------------------------------------------------------
# The atlas: a typed collection of rows + a query surface
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluatorResponseAtlas:
    """The typed per-pair scorer-sensitivity INDEX over the contest video.

    Holds the ordered atlas rows + a cross-video headline + provenance. The
    query surface (by pair / region / family / budget) is the consumer-facing
    API the waterfiller / rate-attack / PR110++ selector use to TARGET the
    highest-budget pairs/regions.
    """

    schema: str
    rows: tuple[AtlasPairRow, ...]
    headline: dict[str, Any]
    provenance: dict[str, Any] = field(default_factory=lambda: dict(ATLAS_PROVENANCE))

    # -- query surface -----------------------------------------------------

    def by_pair(self, pair_index: int) -> AtlasPairRow:
        """Return the atlas row for a specific pair index (fail-closed)."""

        for r in self.rows:
            if r.pair_index == pair_index:
                return r
        raise EvaluatorResponseAtlasError(f"no atlas row for pair_index={pair_index}")

    def top_budget_pairs(self, k: int = 10) -> list[AtlasPairRow]:
        """Top-k pairs by integrated free budget (``pair_budget``).

        These are where the rate attack should spend bytes first: the pairs
        whose frame1 carries the most joint-safe perturbation budget.
        """

        if k <= 0:
            raise EvaluatorResponseAtlasError("k must be >= 1")
        return sorted(
            self.rows, key=lambda r: r.joint_cone_summary.pair_budget, reverse=True
        )[: int(k)]

    def most_fragile_pairs(self, k: int = 10) -> list[AtlasPairRow]:
        """Top-k pairs by fragile fraction (the PROTECT set — least free budget).

        These are the binding-constraint pairs no rate attack should touch.
        """

        if k <= 0:
            raise EvaluatorResponseAtlasError("k must be >= 1")
        return sorted(
            self.rows,
            key=lambda r: r.seg_margin_field_stats.fragile_fraction,
            reverse=True,
        )[: int(k)]

    def pose_bound_pairs(self, threshold: float = 0.5) -> list[AtlasPairRow]:
        """Pairs where the POSE budget binds for > ``threshold`` of pixels.

        The CLAUDE.md marginal-value flip: at the frontier operating point pose
        is the binding half-cone. These pairs need pose-targeted lanes.
        """

        return [
            r
            for r in self.rows
            if r.joint_cone_summary.pose_binds_fraction > float(threshold)
        ]

    def by_region_budget(self, region_class: int) -> list[tuple[int, float]]:
        """Per-pair usable-budget fraction for a SegNet source-class region.

        Returns ``[(pair_index, region_usable_fraction), ...]`` sorted by free
        budget — where in the video a given semantic region (road/sky/etc.)
        carries the most free frame1 budget.
        """

        out: list[tuple[int, float]] = []
        for r in self.rows:
            reg = r.per_region.get(int(region_class))
            if reg is not None:
                out.append((r.pair_index, float(reg.get("usable_budget_fraction", 0.0))))
        return sorted(out, key=lambda t: t[1], reverse=True)

    def by_family(self, family: str) -> dict[str, Any]:
        """Reference the canonical exploit-atom family across the atlas.

        Returns the pairs whose sensitivity regime admits ``family`` + the
        vocabulary pointer. The atlas references the canonical
        :mod:`tac.contest_exploits` families; this is a JOIN to the existing
        score-effect codebook, not a new registry.
        """

        if family not in ATLAS_EXPLOIT_ATOM_FAMILIES:
            raise EvaluatorResponseAtlasError(
                f"unknown exploit atom family {family!r}; known: "
                f"{ATLAS_EXPLOIT_ATOM_FAMILIES}"
            )
        # All pairs reference the canonical vocabulary; the per-pair budget tells
        # the consumer where the family's mechanism has the most room.
        return {
            "family": family,
            "canonical_module": f"tac.contest_exploits.{family}",
            "pairs_by_budget": [
                {"pair_index": r.pair_index, "pair_budget": r.joint_cone_summary.pair_budget}
                for r in self.top_budget_pairs(len(self.rows))
            ],
        }

    # -- persistence -------------------------------------------------------

    def to_jsonl_lines(self) -> list[str]:
        """Serialise to JSONL lines: one header line + one per pair row.

        The header carries the schema + headline + provenance + producer
        artifact references (sha-cited); each subsequent line is one
        :class:`AtlasPairRow`. This is the persisted INDEX (pointers + summary
        stats; no copied tensors).
        """

        header = {
            "schema": self.schema,
            "record_type": "atlas_header",
            "n_pairs": len(self.rows),
            "headline": self.headline,
            "provenance": self.provenance,
        }
        lines = [json.dumps(header, sort_keys=True)]
        for r in self.rows:
            obj = r.to_json_obj()
            obj["record_type"] = "atlas_pair_row"
            lines.append(json.dumps(obj, sort_keys=True))
        return lines

    @classmethod
    def from_jsonl_lines(cls, lines: Iterable[str]) -> EvaluatorResponseAtlas:
        """Load an atlas from its persisted JSONL lines (fail-closed)."""

        header: dict[str, Any] | None = None
        rows: list[AtlasPairRow] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rt = obj.get("record_type")
            if rt == "atlas_header":
                header = obj
            elif rt == "atlas_pair_row":
                rows.append(AtlasPairRow.from_json_obj(obj))
            else:
                raise EvaluatorResponseAtlasError(
                    f"unknown atlas record_type {rt!r} in JSONL"
                )
        if header is None:
            raise EvaluatorResponseAtlasError("atlas JSONL has no header record")
        return cls(
            schema=str(header["schema"]),
            rows=tuple(sorted(rows, key=lambda r: r.pair_index)),
            headline=dict(header.get("headline", {})),
            provenance=dict(header.get("provenance", ATLAS_PROVENANCE)),
        )


def build_atlas(
    rows: Sequence[AtlasPairRow], *, prefer_mlx: bool = True
) -> EvaluatorResponseAtlas:
    """Assemble the atlas from per-pair rows + compute the cross-video headline.

    The headline reduction runs MLX-first (unified memory) with numpy fallback.
    """

    if not rows:
        raise EvaluatorResponseAtlasError("cannot build an atlas from zero rows")
    headline = atlas_cross_video_reduce(rows, prefer_mlx=prefer_mlx)
    # augment the headline with the top-budget / most-fragile pair indices.
    ordered = sorted(rows, key=lambda r: r.joint_cone_summary.pair_budget, reverse=True)
    fragile = sorted(
        rows, key=lambda r: r.seg_margin_field_stats.fragile_fraction, reverse=True
    )
    headline = dict(headline)
    headline["top10_budget_pair_indices"] = [r.pair_index for r in ordered[:10]]
    headline["top10_budget_values"] = [
        float(r.joint_cone_summary.pair_budget) for r in ordered[:10]
    ]
    headline["top10_fragile_pair_indices"] = [r.pair_index for r in fragile[:10]]
    return EvaluatorResponseAtlas(
        schema=EVALUATOR_RESPONSE_ATLAS_SCHEMA,
        rows=tuple(sorted(rows, key=lambda r: r.pair_index)),
        headline=headline,
        provenance=dict(ATLAS_PROVENANCE),
    )


def iter_atlas_rows(atlas: EvaluatorResponseAtlas) -> Iterator[AtlasPairRow]:
    """Iterate the atlas rows in pair-index order."""

    yield from atlas.rows


__all__ = [
    "ATLAS_EXPLOIT_ATOM_FAMILIES",
    "ATLAS_PAIR_ROW_SCHEMA",
    "ATLAS_PROVENANCE",
    "CROSS_VIDEO_REDUCE_FP32_ATOL",
    "EVALUATOR_RESPONSE_ATLAS_SCHEMA",
    "AtlasPairRow",
    "EvaluatorResponseAtlas",
    "EvaluatorResponseAtlasError",
    "JointConeSummary",
    "PoseJacobianFieldStats",
    "SegMarginFieldStats",
    "atlas_cross_video_reduce",
    "atlas_cross_video_reduce_mlx",
    "atlas_cross_video_reduce_numpy",
    "build_atlas",
    "build_atlas_row_from_cone",
    "iter_atlas_rows",
]
