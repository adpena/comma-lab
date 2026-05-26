# SPDX-License-Identifier: MIT
"""NSCS06 v8 chroma-LUT MLX-LOCAL ITERATION extension (Path 3 candidate #C, 2026-05-26).

Per operator NON-NEGOTIABLE 2026-05-26 + the corrected #1258 empirical anchor
(``|S_MLX − S_PyTorch| = 0.000011``; 72x smaller than the PR110 frontier delta
0.000789) + #1265 canonical PASS/FAIL gate
``tools/gate_mlx_candidate_contest_equivalence.py``: MLX-LOCAL iteration is now
contest-grade at the frontier-tightening granularity, so cargo-cult-unwind
research over this substrate's compress-side LUT-derivation policies can fire
locally at $0 instead of paying for repeated paid Modal dispatch.

Background: NSCS06 v8 chroma_lut went through multiple paid Modal dispatch
attempts (per task ledger ``#1195``/``#1207``/``#1208``/``#1209``/``#1213``/``#1219``)
that returned rc=22 / rc=1. The cascade-mortality memo
``#1135``/``#1170 OVERNIGHT-F`` catalogued the substrate as DEFERRED-pending-
research. Per CLAUDE.md "Forbidden premature KILL without research exhaustion"
+ "KILL/FALSIFIED memory verdicts" non-negotiables the paradigm is INTACT;
MLX-local iteration is the next research path that lets the operator (or sister
subagents) iterate cargo-cult unwinds without paying paid CUDA per arm.

This module is **research_only**: it produces MLX-iteration metadata + optional
archive bytes BUT the score signal is ``[macOS-MLX research-signal]`` per
CLAUDE.md "MLX portable-local-substrate authority" non-negotiable. Promotion to
``[contest-CUDA]`` still requires a paired Linux x86_64 + NVIDIA auth-eval on
the EXACT archive bytes per CLAUDE.md "Submission auth eval - BOTH CPU AND
CUDA". The #1265 contest-equivalence gate is the canonical pre-dispatch
disambiguator.

The substrate's INFLATE runtime is already ``numpy + Pillow only`` (no torch);
the MLX value-add is at COMPRESS time:

1. **MLX SegNet argmax** for per-pixel class labels (replaces the torch SegNet
   query in ``experiments/train_substrate_nscs06_v8_chroma_lut.py::_full_main``
   Stage 4); MLX↔PyTorch parity on this specific code path is verified by
   :func:`verify_mlx_segnet_argmax_parity_with_torch`.
2. **Cargo-cult-unwind iteration**: the v8 substrate has a documented set of
   cargo-cult assumptions (Catalog #303 sister discipline; see
   ``.omx/research/nscs06_v8_chroma_lut_design_20260521.md``); MLX-local
   iteration runs candidate ``(grayscale_levels, aggregation_policy)`` arms at
   $0 and emits per-arm predicted ΔS + LUT bytes for operator ranking.

6-hook wire-in declaration per Catalog #125:

* hook #1 sensitivity-map = ACTIVE (per-arm chroma LUT bytes-saved table is
  surfaceable as sensitivity signal for the rate-axis Pareto polytope).
* hook #2 Pareto constraint = ACTIVE via canonical equation #26 IN-DOMAIN
  context ``nscs06_v8_chroma_lut`` (sister to ``procedural_variant.py``).
* hook #3 bit-allocator = N/A at this layer (the canonical 32-byte seed slot
  replaces the 4096-byte chroma LUT slot at archive build; bit allocation IS
  the canonical equation #26 closed-form).
* hook #4 cathedral autopilot dispatch = ACTIVE via the dual-tier consumer
  contract (this module emits Tier A observability-only routing payloads per
  Catalog #341; promotion is gated by the #1265 contest-equivalence gate).
* hook #5 continual-learning posterior = N/A at MLX-iteration layer (paired
  CUDA + CPU anchors per CLAUDE.md "Submission auth eval - BOTH CPU AND CUDA"
  remain required before any posterior anchor lands).
* hook #6 probe-disambiguator = ACTIVE (the MLX↔PyTorch parity verifier IS the
  canonical disambiguator between "MLX-LUT-derivation-is-faithful" and
  "MLX-LUT-derivation-needs-paid-CUDA-confirmation").

Canonical-vs-unique decision per layer (per CLAUDE.md
"UNIQUE-AND-COMPLETE-PER-METHOD operating mode" + Catalog #290):

| Layer | Decision | Rationale |
|---|---|---|
| MLX SegNet adapter | ADOPT canonical | ``tac.local_acceleration.mlx_scorer_adapters.torch_segnet_to_mlx`` |
| MLX SegNet forward | ADOPT canonical | ``run_mlx_segnet_nchw`` |
| Chroma LUT aggregation | ADOPT canonical | ``architecture.build_chroma_lut_from_ground_truth`` |
| Seed derivation | ADOPT canonical | ``tac.procedural_codebook_generator.derive_codebook_from_seed`` |
| Non-promotable markers | ADOPT canonical | ``[macOS-MLX research-signal]`` per Catalog #1/#127/#192/#317/#341 |
| Cargo-cult-unwind axis enumeration | UNIQUE | the (levels, aggregation) policy arms are substrate-specific |
| MLX↔PyTorch parity verifier | UNIQUE | substrate-specific argmax-flip metric on the v8 LUT-binning surface |

Hard-stamped non-promotable contract (per CLAUDE.md "MLX portable-local-substrate
authority" + the #1264 anchor + #1265 gate):

* ``score_claim=False``
* ``promotion_eligible=False``
* ``ready_for_exact_eval_dispatch=False``
* ``rank_or_kill_eligible=False``
* ``axis_tag="[macOS-MLX research-signal]"``
* ``evidence_grade="research-signal"``
* ``contest_equivalence_gate_required_before_dispatch=True``
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .architecture import (
    GRAYSCALE_LEVELS_DEFAULT,
    NUM_SEGNET_CLASSES,
    PROCEDURAL_SEED_SIZE_BYTES,
    build_chroma_lut_from_ground_truth,
)
from .procedural_variant import (
    CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT,
    predicted_archive_bytes_saved,
    predicted_delta_s,
)

__all__ = [
    "CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT",
    "DEFAULT_MLX_AXIS_TAG",
    "DEFAULT_PARITY_TOLERANCE_ARGMAX_FLIP_FRACTION",
    "DEFAULT_SEGNET_NOISE_FLOOR_FRACTION_PATH3CPRIME",
    "MLXIterationArm",
    "MLXIterationError",
    "MLXIterationVerdict",
    "MLXParityVerdict",
    "MLX_NON_PROMOTABLE_PROVENANCE",
    "SegNetArgmaxDisplacementVerdict",
    "derive_chroma_lut_via_mlx_scorer",
    "enumerate_cargo_cult_unwind_arms",
    "is_mlx_available",
    "iterate_chroma_lut_policies_via_mlx",
    "measure_v8_chroma_lut_segnet_argmax_displacement_from_baseline",
    "verify_mlx_segnet_argmax_parity_with_torch",
]


DEFAULT_SEGNET_NOISE_FLOOR_FRACTION_PATH3CPRIME: float = 1e-3
"""Canonical SegNet noise-floor threshold per Path 3 C' Phase 2 Section 3c.

Per Phase 1 audit cargo-cult #8 (SegNet stride-2 stem noise-floor sensitivity
to LUT chroma differentiation UNKNOWN): if v8 chroma LUT produces SegNet
argmax-flip-fraction BELOW 1e-3 (i.e. < 0.1% of pixels change class vs
GT-RGB baseline), the v8 distinguishing feature is BELOW SegNet's
sensitivity floor → substrate is FALSIFIED-AT-IMPLEMENTATION-LEVEL per
Catalog #307 at $0 cost via MLX-local probe BEFORE paid dispatch.

Per CLAUDE.md "MLX portable-local-substrate authority": this is a
research-signal, NOT a contest score authority. PASS verdict means the
substrate is WORTH paid dispatch; FAIL verdict means the substrate's
LUT differentiation is below the scorer's noise floor and paid dispatch
will produce a downstream-of-noise result."""


DEFAULT_MLX_AXIS_TAG: str = "[macOS-MLX research-signal]"
"""Per CLAUDE.md FORBIDDEN_PATTERNS + MLX portable-local-substrate authority."""

DEFAULT_PARITY_TOLERANCE_ARGMAX_FLIP_FRACTION: float = 0.02
"""Default SegNet argmax-flip-fraction tolerance for MLX vs PyTorch parity.

Rationale: the corrected #1258 empirical anchor reports SegNet argmax-flip
fraction ``1.58e-5`` on full-resolution PR95 HNeRV decoded frames. The v8
substrate's compress-side input is the GT RGB frame (not a decoded
reconstruction), so MLX↔PyTorch SegNet drift on the substrate's actual code
path is expected to be smaller than the #1258 anchor. We set the default
tolerance at ``2%`` (1000x the empirical anchor) so the parity verifier admits
the canonical PyTorch path under expected MLX drift and only refuses
catastrophic divergence (e.g. SegNet weight upload failure or NHWC/NCHW axis
mismatch). Operators MAY tighten via the ``tolerance`` kwarg for asymptotic
PV.
"""


MLX_NON_PROMOTABLE_PROVENANCE: dict[str, Any] = {
    "score_claim": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "rank_or_kill_eligible": False,
    "promotable": False,
    "axis_tag": DEFAULT_MLX_AXIS_TAG,
    "evidence_grade": "research-signal",
    "contest_equivalence_gate_required_before_dispatch": True,
    "canonical_equation_in_domain_context": CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT,
    "blockers": (
        "macos_mlx_research_signal_not_contest_authority",
        "requires_paired_contest_cpu_plus_cuda_for_score_claim",
        "requires_pass_verdict_from_gate_mlx_candidate_contest_equivalence",
    ),
}
"""Canonical non-promotable provenance attached to every MLX-iteration result.

Sister of ``tac.optimization.mps_research_signal`` /
``tac.optimization.macos_cpu_advisory_signal`` markers (Catalog #1 + #127 +
#192 + #317 + #341). Promotion to a contest score claim requires (a) a PASS
verdict from ``tools/gate_mlx_candidate_contest_equivalence.py`` AND (b) a
paired Linux x86_64 + NVIDIA auth-eval on the EXACT archive bytes per
CLAUDE.md "Submission auth eval - BOTH CPU AND CUDA on 1:1 contest-compliant
hardware" non-negotiable.
"""


class MLXIterationError(RuntimeError):
    """Raised when an MLX iteration step cannot be honored faithfully."""


def is_mlx_available() -> bool:
    """Return True iff ``mlx.core`` imports cleanly (Apple Silicon expected).

    The MLX-local iteration extension is non-functional on non-Apple platforms;
    callers MUST gate via this helper and fall back to the canonical PyTorch
    compress path when MLX is unavailable.
    """
    try:  # pragma: no cover - exercised on Apple Silicon only
        import mlx.core  # noqa: F401
    except Exception:  # pragma: no cover - import guard
        return False
    return True


# ---------------------------------------------------------------------------
# MLX SegNet compress-side helper
# ---------------------------------------------------------------------------


def derive_chroma_lut_via_mlx_scorer(
    rgb_pairs: np.ndarray,
    mlx_segnet_adapter: Any,
    *,
    grayscale_levels: int = GRAYSCALE_LEVELS_DEFAULT,
    num_segnet_classes: int = NUM_SEGNET_CLASSES,
    chunk_size: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive the (levels, classes, 3) chroma LUT using an MLX SegNet adapter.

    Sister of the torch path in
    ``experiments/train_substrate_nscs06_v8_chroma_lut.py::_full_main`` Stage 4
    (per-pixel SegNet argmax) + Stage 6 (canonical
    ``build_chroma_lut_from_ground_truth`` aggregation). The MLX path replaces
    the torch SegNet forward; the LUT aggregation step adopts the canonical
    numpy helper unchanged (Catalog #290 canonical-vs-unique adoption per
    layer).

    Args:
        rgb_pairs: ``(N, 3, H, W)`` uint8 RGB frames (compress-time GT). H, W
            should match the contest scorer's expected resolution (typically
            ``EVAL_HW = (384, 512)`` per substrate trainer canonical).
        mlx_segnet_adapter: an ``MLXSegNetAdapter`` returned by
            ``tac.local_acceleration.mlx_scorer_adapters.torch_segnet_to_mlx``.
        grayscale_levels: Number of luma quantization levels (default 16).
        num_segnet_classes: Number of SegNet semantic classes (default 5).
        chunk_size: Mini-batch size for MLX SegNet forward (default 8 mirrors
            Catalog #218 torch sister chunk-size pattern).

    Returns:
        ``(chroma_lut, cls_full)`` where:

        * ``chroma_lut``: ``(grayscale_levels, num_segnet_classes, 3)`` uint8.
        * ``cls_full``: ``(N, H, W)`` uint8 per-pixel SegNet argmax labels
          (returned so callers can persist them or build other LUT shapes).

    Raises:
        MLXIterationError: if ``rgb_pairs`` shape/dtype is invalid OR if the
            MLX adapter is unavailable.
    """
    if not is_mlx_available():
        raise MLXIterationError(
            "MLX is not available; this helper requires Apple Silicon + MLX install"
        )
    if rgb_pairs.dtype != np.uint8:
        raise MLXIterationError(
            f"rgb_pairs must be uint8; got {rgb_pairs.dtype}"
        )
    if rgb_pairs.ndim != 4 or rgb_pairs.shape[1] != 3:
        raise MLXIterationError(
            f"rgb_pairs must be (N, 3, H, W); got {rgb_pairs.shape}"
        )
    if chunk_size < 1:
        raise MLXIterationError(f"chunk_size must be >= 1; got {chunk_size}")

    from tac.local_acceleration.mlx_scorer_adapters import run_mlx_segnet_nchw

    n = int(rgb_pairs.shape[0])
    h = int(rgb_pairs.shape[2])
    w = int(rgb_pairs.shape[3])
    cls_chunks: list[np.ndarray] = []

    # MLX SegNet expects float32 NCHW in 0..255 (mirrors torch SegNet
    # `preprocess_input` contract per upstream/modules.py + sister torch
    # path in train_substrate_nscs06_v8_chroma_lut.py:662).
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        # Use ODD frame (last frame) for SegNet per upstream/modules.py
        # `x[:, -1, ...]` convention; the rgb_pairs caller passes the odd frame
        # directly (sister torch path uses pair_tensor[:, 0] for odd; the v8
        # trainer passes odd RGB to SegNet at line 662 of train_substrate_*).
        chunk = rgb_pairs[start:stop].astype(np.float32)
        seg_logits_nchw = run_mlx_segnet_nchw(mlx_segnet_adapter, chunk)
        # seg_logits_nchw shape: (chunk, 5, H_out, W_out). We need argmax in
        # class dim at the SAME spatial resolution as input RGB. The SegNet
        # output may be at a different resolution per its internal stride
        # (upstream/modules.py SegNet interpolates internally to (512, 384)).
        # For LUT binning we need per-pixel labels at INPUT resolution; if
        # SegNet output resolution differs, nearest-neighbor upsample is
        # sufficient since argmax labels are categorical (no smoothing).
        cls_chunk = np.argmax(seg_logits_nchw, axis=1).astype(np.uint8)
        if cls_chunk.shape[-2:] != (h, w):
            # Cheap nearest-neighbor resample to (h, w) via numpy indexing.
            ch, cw = cls_chunk.shape[-2:]
            row_idx = (np.arange(h, dtype=np.int64) * ch // h).clip(0, ch - 1)
            col_idx = (np.arange(w, dtype=np.int64) * cw // w).clip(0, cw - 1)
            cls_chunk = cls_chunk[:, row_idx[:, None], col_idx[None, :]]
        cls_chunks.append(cls_chunk)
    cls_full = np.concatenate(cls_chunks, axis=0).astype(np.uint8)

    # Adopt canonical numpy aggregation (Catalog #290 ADOPT canonical).
    chroma_lut = build_chroma_lut_from_ground_truth(
        rgb_pairs,
        cls_full,
        grayscale_levels=grayscale_levels,
        num_segnet_classes=num_segnet_classes,
    )
    return chroma_lut, cls_full


# ---------------------------------------------------------------------------
# Cargo-cult-unwind iteration arms (Catalog #303 sister discipline)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MLXIterationArm:
    """One cargo-cult-unwind candidate for MLX-local iteration.

    Per Catalog #303 ``## Cargo-cult audit per assumption`` sister discipline +
    the existing v8 ``revisions.py`` per-assumption ablation ladder pattern.
    Each arm probes a single cargo-cult assumption in the v8 substrate's
    compress-side LUT-derivation path:

    * ``grayscale_levels``: the 4-bit luma quantization assumption (default
      16). Alternative arms test 8 / 32 / 64.
    * ``num_segnet_classes_aggregation_policy``: the 5-class aggregation
      assumption (default ``per_class``). Alternative arms test
      ``per_class_pair`` (grouping classes into background-vs-foreground)
      or ``per_class_triplet``.
    * ``arm_label``: a human-readable id for the arm (used in operator
      ranking + iteration result manifests).
    """

    arm_label: str
    grayscale_levels: int
    num_segnet_classes_aggregation_policy: str = "per_class"

    def __post_init__(self) -> None:
        if self.grayscale_levels < 1 or self.grayscale_levels > 256:
            raise MLXIterationError(
                f"grayscale_levels={self.grayscale_levels} outside [1, 256]"
            )
        valid_policies = {"per_class", "binary_foreground", "merged_road_lane"}
        if self.num_segnet_classes_aggregation_policy not in valid_policies:
            raise MLXIterationError(
                f"num_segnet_classes_aggregation_policy="
                f"{self.num_segnet_classes_aggregation_policy!r} "
                f"not in {sorted(valid_policies)}"
            )
        if not self.arm_label:
            raise MLXIterationError("arm_label must be a non-empty string")


def enumerate_cargo_cult_unwind_arms() -> tuple[MLXIterationArm, ...]:
    """Canonical cargo-cult-unwind arms for MLX-local iteration.

    Each arm probes a documented cargo-cult assumption from the v8 substrate's
    design memo (``.omx/research/nscs06_v8_chroma_lut_design_20260521.md`` +
    sister ``revisions.py`` ablation ladder). The canonical baseline arm
    ``baseline_4bit_per_class`` reproduces the v7 -> v8 default; alternative
    arms unwind specific cargo-cult assumptions inherited from the v7 design
    that the v8 dispatches kept failing on.

    Returns:
        Tuple of frozen :class:`MLXIterationArm` records covering the
        canonical cargo-cult-unwind axes. The baseline arm is FIRST.
    """
    return (
        # Baseline (default v8 behavior; mirror of trainer Stage 5+6).
        MLXIterationArm(
            arm_label="baseline_4bit_per_class",
            grayscale_levels=16,
            num_segnet_classes_aggregation_policy="per_class",
        ),
        # Cargo-cult #1 (inherited from v7): 4-bit luma is "always enough".
        # UNWIND: try 3-bit (8 levels) and 5-bit (32 levels) variants.
        MLXIterationArm(
            arm_label="cargo_cult_1_unwind_3bit_per_class",
            grayscale_levels=8,
            num_segnet_classes_aggregation_policy="per_class",
        ),
        MLXIterationArm(
            arm_label="cargo_cult_1_unwind_5bit_per_class",
            grayscale_levels=32,
            num_segnet_classes_aggregation_policy="per_class",
        ),
        # Cargo-cult #2 (inherited from v7): 5-class per-class is "always
        # optimal". UNWIND: try binary foreground + merged road/lane variants.
        MLXIterationArm(
            arm_label="cargo_cult_2_unwind_binary_foreground",
            grayscale_levels=16,
            num_segnet_classes_aggregation_policy="binary_foreground",
        ),
        MLXIterationArm(
            arm_label="cargo_cult_2_unwind_merged_road_lane",
            grayscale_levels=16,
            num_segnet_classes_aggregation_policy="merged_road_lane",
        ),
    )


def _apply_aggregation_policy(
    cls_labels: np.ndarray,
    policy: str,
) -> tuple[np.ndarray, int]:
    """Apply a cargo-cult-unwind aggregation policy to per-pixel class labels.

    Returns ``(remapped_labels, effective_num_classes)``.
    """
    if policy == "per_class":
        return cls_labels.astype(np.uint8), NUM_SEGNET_CLASSES
    if policy == "binary_foreground":
        # Background (class 0) -> 0; all foreground -> 1.
        remapped = (cls_labels > 0).astype(np.uint8)
        return remapped, 2
    if policy == "merged_road_lane":
        # Background=0; lane(1) + road(3) -> 1; vehicle(2) -> 2; sky-or-other(4) -> 3
        remap_table = np.array([0, 1, 2, 1, 3], dtype=np.uint8)
        # cls_labels is already in [0, 4]; safe to index
        clipped = np.clip(cls_labels, 0, 4)
        remapped = remap_table[clipped]
        return remapped, 4
    raise MLXIterationError(f"unknown aggregation policy: {policy!r}")


# ---------------------------------------------------------------------------
# MLX iteration verdict + canonical contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MLXIterationVerdict:
    """Per-arm verdict from MLX-local iteration over the chroma LUT policies.

    Carries the canonical non-promotable contract per CLAUDE.md "MLX
    portable-local-substrate authority" + Catalog #341 dual-tier consumer
    architecture (Tier A observability-only). Promotion to a contest score
    claim requires PASS verdict from
    ``tools/gate_mlx_candidate_contest_equivalence.py`` PLUS paired Linux
    x86_64 + NVIDIA auth-eval per CLAUDE.md "Submission auth eval - BOTH CPU
    AND CUDA on 1:1 contest-compliant hardware".
    """

    arm_label: str
    grayscale_levels: int
    num_segnet_classes_aggregation_policy: str
    chroma_lut_bytes_full: int
    procedural_seed_size_bytes: int
    predicted_archive_bytes_saved: int
    predicted_delta_s: float
    chroma_lut_sha256: str
    procedural_seed_sha256: str
    cls_label_remap_effective_classes: int
    # Canonical non-promotable contract (Catalog #1 + #127 + #192 + #317 + #341)
    axis_tag: str = DEFAULT_MLX_AXIS_TAG
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False
    rank_or_kill_eligible: bool = False
    evidence_grade: str = "research-signal"
    contest_equivalence_gate_required_before_dispatch: bool = True
    canonical_equation_in_domain_context: str = CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT
    blockers: tuple[str, ...] = field(
        default_factory=lambda: (
            "macos_mlx_research_signal_not_contest_authority",
            "requires_paired_contest_cpu_plus_cuda_for_score_claim",
            "requires_pass_verdict_from_gate_mlx_candidate_contest_equivalence",
        )
    )

    def __post_init__(self) -> None:
        # The non-promotable contract is the structural protection per Catalog
        # #1 + #127 + #192 + #317 + #341. Reject any construction attempt that
        # weakens it.
        if self.score_claim is not False:
            raise MLXIterationError("score_claim MUST be False for MLX iteration verdict")
        if self.promotion_eligible is not False:
            raise MLXIterationError("promotion_eligible MUST be False for MLX iteration verdict")
        if self.ready_for_exact_eval_dispatch is not False:
            raise MLXIterationError(
                "ready_for_exact_eval_dispatch MUST be False for MLX iteration verdict"
            )
        if self.rank_or_kill_eligible is not False:
            raise MLXIterationError(
                "rank_or_kill_eligible MUST be False for MLX iteration verdict"
            )
        if self.axis_tag != DEFAULT_MLX_AXIS_TAG:
            raise MLXIterationError(
                f"axis_tag MUST be {DEFAULT_MLX_AXIS_TAG!r} for MLX iteration verdict; "
                f"got {self.axis_tag!r}"
            )
        if self.evidence_grade != "research-signal":
            raise MLXIterationError(
                f"evidence_grade MUST be 'research-signal' for MLX iteration verdict; "
                f"got {self.evidence_grade!r}"
            )

    def as_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict (for persisted iteration manifests)."""
        return {
            "arm_label": self.arm_label,
            "grayscale_levels": self.grayscale_levels,
            "num_segnet_classes_aggregation_policy": (
                self.num_segnet_classes_aggregation_policy
            ),
            "chroma_lut_bytes_full": self.chroma_lut_bytes_full,
            "procedural_seed_size_bytes": self.procedural_seed_size_bytes,
            "predicted_archive_bytes_saved": self.predicted_archive_bytes_saved,
            "predicted_delta_s": self.predicted_delta_s,
            "chroma_lut_sha256": self.chroma_lut_sha256,
            "procedural_seed_sha256": self.procedural_seed_sha256,
            "cls_label_remap_effective_classes": (
                self.cls_label_remap_effective_classes
            ),
            "axis_tag": self.axis_tag,
            "score_claim": self.score_claim,
            "promotion_eligible": self.promotion_eligible,
            "ready_for_exact_eval_dispatch": self.ready_for_exact_eval_dispatch,
            "rank_or_kill_eligible": self.rank_or_kill_eligible,
            "evidence_grade": self.evidence_grade,
            "contest_equivalence_gate_required_before_dispatch": (
                self.contest_equivalence_gate_required_before_dispatch
            ),
            "canonical_equation_in_domain_context": (
                self.canonical_equation_in_domain_context
            ),
            "blockers": list(self.blockers),
        }


# ---------------------------------------------------------------------------
# Iteration loop
# ---------------------------------------------------------------------------


def iterate_chroma_lut_policies_via_mlx(
    rgb_pairs: np.ndarray,
    mlx_segnet_adapter: Any,
    *,
    arms: tuple[MLXIterationArm, ...] | None = None,
    chunk_size: int = 8,
) -> tuple[MLXIterationVerdict, ...]:
    """Iterate the canonical cargo-cult-unwind arms via MLX-local SegNet.

    Each arm produces a :class:`MLXIterationVerdict` with the substrate-
    specific LUT bytes-saved prediction + canonical non-promotable markers.
    The iteration runs entirely in MLX at $0; the operator (or sister
    subagents) then ranks arms by ``predicted_delta_s`` and selects which to
    promote to a paid Modal dispatch.

    Args:
        rgb_pairs: ``(N, 3, H, W)`` uint8 RGB frames (compress-time GT;
            typically the odd-frame slice of the pair tensor per upstream
            SegNet convention).
        mlx_segnet_adapter: an ``MLXSegNetAdapter`` instance.
        arms: tuple of arms to iterate; defaults to
            :func:`enumerate_cargo_cult_unwind_arms`.
        chunk_size: per-call SegNet forward batch size.

    Returns:
        Tuple of :class:`MLXIterationVerdict` in the same order as ``arms``.
    """
    if arms is None:
        arms = enumerate_cargo_cult_unwind_arms()
    if not arms:
        raise MLXIterationError("arms tuple must be non-empty")

    # Run MLX SegNet ONCE; reuse cls_full across arms by applying aggregation
    # policy in numpy. This keeps the MLX cost amortized across N arms.
    _, cls_full_per_class = derive_chroma_lut_via_mlx_scorer(
        rgb_pairs,
        mlx_segnet_adapter,
        grayscale_levels=GRAYSCALE_LEVELS_DEFAULT,
        num_segnet_classes=NUM_SEGNET_CLASSES,
        chunk_size=chunk_size,
    )

    verdicts: list[MLXIterationVerdict] = []
    for arm in arms:
        remapped, eff_classes = _apply_aggregation_policy(
            cls_full_per_class, arm.num_segnet_classes_aggregation_policy
        )
        chroma_lut = build_chroma_lut_from_ground_truth(
            rgb_pairs,
            remapped,
            grayscale_levels=arm.grayscale_levels,
            num_segnet_classes=eff_classes,
        )
        chroma_bytes = chroma_lut.tobytes()
        chroma_sha = hashlib.sha256(chroma_bytes).hexdigest()
        # Derive a deterministic procedural seed from the LUT bytes (sister
        # to the canonical hash-based seed in the v8 trainer Stage 9).
        seed_bytes = hashlib.sha256(chroma_bytes).digest()[:PROCEDURAL_SEED_SIZE_BYTES]
        seed_sha = hashlib.sha256(seed_bytes).hexdigest()
        # Canonical equation #26 closed-form predicted bytes saved.
        bytes_saved = predicted_archive_bytes_saved()
        delta_s = predicted_delta_s()
        verdict = MLXIterationVerdict(
            arm_label=arm.arm_label,
            grayscale_levels=arm.grayscale_levels,
            num_segnet_classes_aggregation_policy=(
                arm.num_segnet_classes_aggregation_policy
            ),
            chroma_lut_bytes_full=len(chroma_bytes),
            procedural_seed_size_bytes=len(seed_bytes),
            predicted_archive_bytes_saved=bytes_saved,
            predicted_delta_s=delta_s,
            chroma_lut_sha256=chroma_sha,
            procedural_seed_sha256=seed_sha,
            cls_label_remap_effective_classes=eff_classes,
        )
        verdicts.append(verdict)
    return tuple(verdicts)


# ---------------------------------------------------------------------------
# MLX <-> PyTorch parity verifier (probe-disambiguator per Catalog #125 hook #6)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MLXParityVerdict:
    """Verdict from :func:`verify_mlx_segnet_argmax_parity_with_torch`.

    Carries non-promotable markers per the same canonical contract as
    :class:`MLXIterationVerdict`. The parity verdict is the canonical
    probe-disambiguator (Catalog #125 hook #6) between
    "MLX-LUT-derivation-is-faithful" and "MLX-LUT-derivation-needs-paid-
    CUDA-confirmation" routing decisions.
    """

    num_pairs_compared: int
    argmax_flip_fraction: float
    tolerance: float
    parity_ok: bool
    max_abs_logit_drift: float
    mean_abs_logit_drift: float
    axis_tag: str = DEFAULT_MLX_AXIS_TAG
    score_claim: bool = False
    promotion_eligible: bool = False
    evidence_grade: str = "research-signal"

    def __post_init__(self) -> None:
        if self.score_claim is not False:
            raise MLXIterationError("score_claim MUST be False for MLX parity verdict")
        if self.promotion_eligible is not False:
            raise MLXIterationError(
                "promotion_eligible MUST be False for MLX parity verdict"
            )
        if self.axis_tag != DEFAULT_MLX_AXIS_TAG:
            raise MLXIterationError(
                f"axis_tag MUST be {DEFAULT_MLX_AXIS_TAG!r}; got {self.axis_tag!r}"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "num_pairs_compared": self.num_pairs_compared,
            "argmax_flip_fraction": self.argmax_flip_fraction,
            "tolerance": self.tolerance,
            "parity_ok": self.parity_ok,
            "max_abs_logit_drift": self.max_abs_logit_drift,
            "mean_abs_logit_drift": self.mean_abs_logit_drift,
            "axis_tag": self.axis_tag,
            "score_claim": self.score_claim,
            "promotion_eligible": self.promotion_eligible,
            "evidence_grade": self.evidence_grade,
        }


def verify_mlx_segnet_argmax_parity_with_torch(
    rgb_pairs: np.ndarray,
    mlx_segnet_adapter: Any,
    torch_segnet: Any,
    *,
    tolerance: float = DEFAULT_PARITY_TOLERANCE_ARGMAX_FLIP_FRACTION,
    chunk_size: int = 4,
) -> MLXParityVerdict:
    """Compare MLX vs PyTorch SegNet argmax on the substrate's compress-side path.

    Sister of the #1258 corrected anchor methodology. The v8 chroma-LUT
    substrate's compress-side only uses SegNet argmax (per-pixel class
    labels); this verifier measures the argmax-flip-fraction between MLX and
    PyTorch on the SAME GT RGB input to disambiguate "MLX-LUT-is-faithful"
    from "MLX-LUT-needs-paid-CUDA-confirmation".

    Args:
        rgb_pairs: ``(N, 3, H, W)`` uint8 RGB frames (typically a small
            calibration window).
        mlx_segnet_adapter: an ``MLXSegNetAdapter`` instance.
        torch_segnet: the canonical PyTorch SegNet (eval mode).
        tolerance: maximum acceptable argmax-flip fraction.
        chunk_size: per-call SegNet forward batch size.

    Returns:
        :class:`MLXParityVerdict` with the empirical drift metrics.
    """
    if not is_mlx_available():
        raise MLXIterationError(
            "MLX is not available; this helper requires Apple Silicon + MLX install"
        )
    if rgb_pairs.dtype != np.uint8:
        raise MLXIterationError(f"rgb_pairs must be uint8; got {rgb_pairs.dtype}")
    if rgb_pairs.ndim != 4 or rgb_pairs.shape[1] != 3:
        raise MLXIterationError(
            f"rgb_pairs must be (N, 3, H, W); got {rgb_pairs.shape}"
        )
    if chunk_size < 1:
        raise MLXIterationError(f"chunk_size must be >= 1; got {chunk_size}")
    if not (0.0 <= tolerance <= 1.0):
        raise MLXIterationError(
            f"tolerance must be in [0.0, 1.0]; got {tolerance}"
        )

    import torch

    from tac.local_acceleration.mlx_scorer_adapters import run_mlx_segnet_nchw

    n = int(rgb_pairs.shape[0])
    flip_count = 0
    total_pixels = 0
    max_abs_drift = 0.0
    sum_abs_drift = 0.0
    drift_voxels = 0
    torch_device = next(torch_segnet.parameters()).device
    torch_segnet.eval()

    with torch.no_grad():
        for start in range(0, n, chunk_size):
            stop = min(start + chunk_size, n)
            chunk = rgb_pairs[start:stop].astype(np.float32)
            # MLX path
            mlx_logits = run_mlx_segnet_nchw(mlx_segnet_adapter, chunk)
            mlx_argmax = np.argmax(mlx_logits, axis=1)
            # Torch path with the same canonical preprocess used by the
            # substrate trainer (per
            # train_substrate_nscs06_v8_chroma_lut.py:662). Upstream SegNet
            # expects BTCHW; we wrap with T=1 to mirror the trainer pattern.
            t_chunk = torch.from_numpy(chunk).to(torch_device).float()
            t_btchw = t_chunk.unsqueeze(1)  # (chunk, 1, 3, H, W)
            torch_logits = torch_segnet(torch_segnet.preprocess_input(t_btchw))
            torch_argmax = torch.argmax(torch_logits, dim=1).cpu().numpy()
            # Resample to common spatial shape if needed (nearest-neighbor
            # since labels are categorical).
            if mlx_argmax.shape != torch_argmax.shape:
                ch, cw = mlx_argmax.shape[-2:]
                th, tw = torch_argmax.shape[-2:]
                row_idx = (np.arange(th, dtype=np.int64) * ch // th).clip(
                    0, ch - 1
                )
                col_idx = (np.arange(tw, dtype=np.int64) * cw // tw).clip(
                    0, cw - 1
                )
                mlx_argmax = mlx_argmax[:, row_idx[:, None], col_idx[None, :]]
            flips = int((mlx_argmax != torch_argmax).sum())
            flip_count += flips
            total_pixels += int(mlx_argmax.size)
            # Logit drift requires aligned spatial shapes; resample MLX
            # logits to torch shape if needed (linear interp avoided -- we
            # report stats per voxel where shapes already match).
            t_logits_np = torch_logits.cpu().numpy()
            if mlx_logits.shape == t_logits_np.shape:
                diff = np.abs(mlx_logits - t_logits_np)
                max_abs_drift = max(max_abs_drift, float(diff.max()))
                sum_abs_drift += float(diff.sum())
                drift_voxels += int(diff.size)

    argmax_flip_fraction = flip_count / max(1, total_pixels)
    mean_abs_drift = (
        sum_abs_drift / max(1, drift_voxels) if drift_voxels > 0 else 0.0
    )
    parity_ok = argmax_flip_fraction <= tolerance
    return MLXParityVerdict(
        num_pairs_compared=n,
        argmax_flip_fraction=argmax_flip_fraction,
        tolerance=tolerance,
        parity_ok=parity_ok,
        max_abs_logit_drift=max_abs_drift,
        mean_abs_logit_drift=mean_abs_drift,
    )


# ---------------------------------------------------------------------------
# Path 3 C' Phase 3 Section 3c — SegNet noise-floor probe (UNWIND cargo-cult #8)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SegNetArgmaxDisplacementVerdict:
    """Verdict from the $0 MLX-local SegNet noise-floor probe per Path 3 C' Phase 2 Section 3c.

    Carries the canonical non-promotable contract per CLAUDE.md "MLX
    portable-local-substrate authority" + Catalog #287 + #323 + #341
    (dual-tier consumer architecture Tier A observability-only).

    Per Phase 1 audit cargo-cult #8 (CARGO-CULTED-CRITICAL): the v8 chroma
    LUT MIGHT produce differentiation BELOW the SegNet stride-2 stem's
    noise floor. If so, the substrate is FALSIFIED-AT-IMPLEMENTATION-LEVEL
    per Catalog #307 at $0 BEFORE any paid Modal dispatch fires.

    The verdict reports:
    - argmax_displacement_fraction: fraction of pixels whose SegNet argmax
      class CHANGES between (GT-RGB baseline) and (v8-LUT-rendered RGB).
    - in_noise_floor: True iff displacement is below the canonical threshold
      :data:`DEFAULT_SEGNET_NOISE_FLOOR_FRACTION_PATH3CPRIME` (1e-3).
    - recommended_proceed: True iff displacement is ABOVE the noise floor
      (i.e. substrate's distinguishing feature IS detectable by SegNet).

    Expected outcomes:
    - PASS (recommended_proceed=True; displacement > 1e-3): v8 substrate's
      LUT differentiation is SegNet-detectable; paid dispatch is justified
      per Catalog #325 per-substrate symposium PROCEED criteria.
    - FAIL (recommended_proceed=False; displacement <= 1e-3): v8 substrate's
      LUT differentiation is BELOW SegNet noise floor; paid dispatch will
      produce a downstream-of-noise result; recommend Path 2 redesign per
      Catalog #307 IMPLEMENTATION-LEVEL falsification.
    """

    num_pairs_probed: int
    argmax_displacement_fraction: float
    noise_floor_threshold: float
    in_noise_floor: bool
    recommended_proceed: bool
    max_per_pair_displacement: float
    mean_per_pair_displacement: float
    # Canonical non-promotable contract per Catalog #1 + #127 + #192 + #317 + #341
    axis_tag: str = DEFAULT_MLX_AXIS_TAG
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False
    rank_or_kill_eligible: bool = False
    evidence_grade: str = "research-signal"
    blockers: tuple[str, ...] = field(
        default_factory=lambda: (
            "macos_mlx_research_signal_not_contest_authority",
            "requires_paired_contest_cpu_plus_cuda_for_score_claim",
            "noise_floor_probe_is_GO_NO_GO_recommendation_not_score_claim",
        )
    )

    def __post_init__(self) -> None:
        if self.score_claim is not False:
            raise MLXIterationError(
                "score_claim MUST be False for SegNet displacement verdict"
            )
        if self.promotion_eligible is not False:
            raise MLXIterationError(
                "promotion_eligible MUST be False for SegNet displacement verdict"
            )
        if self.ready_for_exact_eval_dispatch is not False:
            raise MLXIterationError(
                "ready_for_exact_eval_dispatch MUST be False for SegNet displacement verdict"
            )
        if self.rank_or_kill_eligible is not False:
            raise MLXIterationError(
                "rank_or_kill_eligible MUST be False for SegNet displacement verdict"
            )
        if self.axis_tag != DEFAULT_MLX_AXIS_TAG:
            raise MLXIterationError(
                f"axis_tag MUST be {DEFAULT_MLX_AXIS_TAG!r}; got {self.axis_tag!r}"
            )
        if self.evidence_grade != "research-signal":
            raise MLXIterationError(
                f"evidence_grade MUST be 'research-signal'; got {self.evidence_grade!r}"
            )
        if not 0.0 <= self.argmax_displacement_fraction <= 1.0:
            raise MLXIterationError(
                f"argmax_displacement_fraction={self.argmax_displacement_fraction} "
                f"outside [0.0, 1.0]"
            )
        if self.noise_floor_threshold <= 0.0 or self.noise_floor_threshold >= 1.0:
            raise MLXIterationError(
                f"noise_floor_threshold={self.noise_floor_threshold} outside (0.0, 1.0)"
            )
        # Verify the in_noise_floor + recommended_proceed booleans are consistent
        # with the canonical threshold semantics.
        expected_in_noise_floor = (
            self.argmax_displacement_fraction <= self.noise_floor_threshold
        )
        if self.in_noise_floor != expected_in_noise_floor:
            raise MLXIterationError(
                f"in_noise_floor inconsistent: argmax_displacement_fraction="
                f"{self.argmax_displacement_fraction} vs threshold="
                f"{self.noise_floor_threshold}; expected in_noise_floor="
                f"{expected_in_noise_floor}, got {self.in_noise_floor}"
            )
        if self.recommended_proceed == self.in_noise_floor:
            raise MLXIterationError(
                "recommended_proceed must be the NEGATION of in_noise_floor "
                "(proceed when NOT in noise floor)"
            )

    def as_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict per Catalog #287 + #323."""
        return {
            "num_pairs_probed": self.num_pairs_probed,
            "argmax_displacement_fraction": self.argmax_displacement_fraction,
            "noise_floor_threshold": self.noise_floor_threshold,
            "in_noise_floor": self.in_noise_floor,
            "recommended_proceed": self.recommended_proceed,
            "max_per_pair_displacement": self.max_per_pair_displacement,
            "mean_per_pair_displacement": self.mean_per_pair_displacement,
            "axis_tag": self.axis_tag,
            "score_claim": self.score_claim,
            "promotion_eligible": self.promotion_eligible,
            "ready_for_exact_eval_dispatch": self.ready_for_exact_eval_dispatch,
            "rank_or_kill_eligible": self.rank_or_kill_eligible,
            "evidence_grade": self.evidence_grade,
            "blockers": list(self.blockers),
        }


def measure_v8_chroma_lut_segnet_argmax_displacement_from_baseline(
    rgb_pairs_gt: np.ndarray,
    chroma_lut_v8: np.ndarray,
    mlx_segnet_adapter: Any,
    *,
    noise_floor_threshold: float = DEFAULT_SEGNET_NOISE_FLOOR_FRACTION_PATH3CPRIME,
    chunk_size: int = 8,
    cls_full_at_probe_time: np.ndarray | None = None,
) -> SegNetArgmaxDisplacementVerdict:
    """$0 MLX-local SegNet noise-floor probe — UNWIND Path 3 C' cargo-cult #8.

    THIS IS THE CANONICAL CARGO-CULT #8 UNWIND PROBE per Path 3 C' Phase 2
    Section 3c. Measures whether the v8 chroma LUT's inflate-reconstructed
    RGB frames produce SegNet argmax-flip-fraction ABOVE or BELOW the
    canonical noise-floor threshold (default 1e-3).

    Operational contract:

    1. For each GT pair, run MLX SegNet on GT RGB → get baseline argmax labels.
    2. Reconstruct the v8 inflated frame_0 RGB via the canonical
       :func:`tac.substrates.nscs06_v8_chroma_lut.architecture.lookup_rgb_via_chroma_lut`
       using GT-derived (gray, cls) maps as inputs.
    3. Run MLX SegNet on reconstructed RGB → get mutated argmax labels.
    4. Compute per-pair displacement = (mutated_argmax != baseline_argmax).mean().
    5. Aggregate across pairs → mean displacement; threshold-check.

    Cost: $0 (Apple Silicon MLX; ~30-90s per pair on M5 Max baseline).

    Args:
        rgb_pairs_gt: ``(N, 3, H, W)`` uint8 GT RGB pairs (typically the
            odd-frame slice; H=384 W=512 contest resolution).
        chroma_lut_v8: ``(grayscale_levels, num_segnet_classes, 3)`` uint8
            v8 chroma LUT (canonical 16 × 5 × 3 = 240 dense bytes).
        mlx_segnet_adapter: an ``MLXSegNetAdapter`` instance from
            ``tac.local_acceleration.mlx_scorer_adapters.torch_segnet_to_mlx``.
        noise_floor_threshold: canonical threshold for in-noise-floor verdict.
            Default :data:`DEFAULT_SEGNET_NOISE_FLOOR_FRACTION_PATH3CPRIME`
            (1e-3 = 0.1% pixel flip).
        chunk_size: per-call SegNet forward batch size (default 8).
        cls_full_at_probe_time: optional ``(N, H, W)`` uint8 SegNet-derived
            class labels for the LUT lookup. If None, derived via the
            canonical :func:`derive_chroma_lut_via_mlx_scorer` from
            ``rgb_pairs_gt`` (sister-canonical pattern). Providing this
            kwarg lets callers supply L1-promotion-style cls_stream-derived
            labels for a UNWIND-test of cargo-cult #5 in concert with
            cargo-cult #8.

    Returns:
        :class:`SegNetArgmaxDisplacementVerdict`.

    Raises:
        MLXIterationError: if MLX is unavailable OR rgb_pairs shape invalid
            OR chroma_lut shape invalid OR threshold outside (0, 1).
    """
    if not is_mlx_available():
        raise MLXIterationError(
            "MLX is not available; SegNet displacement probe requires Apple Silicon + MLX install"
        )
    if rgb_pairs_gt.dtype != np.uint8:
        raise MLXIterationError(
            f"rgb_pairs_gt must be uint8; got {rgb_pairs_gt.dtype}"
        )
    if rgb_pairs_gt.ndim != 4 or rgb_pairs_gt.shape[1] != 3:
        raise MLXIterationError(
            f"rgb_pairs_gt must be (N, 3, H, W); got {rgb_pairs_gt.shape}"
        )
    if chroma_lut_v8.dtype != np.uint8:
        raise MLXIterationError(
            f"chroma_lut_v8 must be uint8; got {chroma_lut_v8.dtype}"
        )
    if chroma_lut_v8.ndim != 3 or chroma_lut_v8.shape[2] != 3:
        raise MLXIterationError(
            f"chroma_lut_v8 must be (grayscale_levels, num_classes, 3); got "
            f"{chroma_lut_v8.shape}"
        )
    if noise_floor_threshold <= 0.0 or noise_floor_threshold >= 1.0:
        raise MLXIterationError(
            f"noise_floor_threshold={noise_floor_threshold} outside (0.0, 1.0)"
        )
    if chunk_size < 1:
        raise MLXIterationError(f"chunk_size must be >= 1; got {chunk_size}")

    # Lazy import to avoid hard MLX dep at module-import time on non-Apple platforms.
    from tac.local_acceleration.mlx_scorer_adapters import run_mlx_segnet_nchw

    from .architecture import lookup_rgb_via_chroma_lut

    n = int(rgb_pairs_gt.shape[0])
    h = int(rgb_pairs_gt.shape[2])
    w = int(rgb_pairs_gt.shape[3])
    grayscale_levels, num_segnet_classes, _ = chroma_lut_v8.shape

    # Step 1: derive cls_full labels for the LUT lookup (either user-provided
    # OR via canonical MLX SegNet helper).
    if cls_full_at_probe_time is not None:
        cls_full = cls_full_at_probe_time
        if cls_full.dtype != np.uint8:
            raise MLXIterationError(
                f"cls_full_at_probe_time must be uint8; got {cls_full.dtype}"
            )
        if cls_full.shape != (n, h, w):
            raise MLXIterationError(
                f"cls_full_at_probe_time shape {cls_full.shape} != ({n}, {h}, {w})"
            )
    else:
        # Use the canonical helper to derive cls labels via MLX SegNet.
        _, cls_full = derive_chroma_lut_via_mlx_scorer(
            rgb_pairs_gt,
            mlx_segnet_adapter,
            grayscale_levels=grayscale_levels,
            num_segnet_classes=num_segnet_classes,
            chunk_size=chunk_size,
        )

    # Step 2: compute per-pixel grayscale for the LUT lookup (BT.601 luma).
    r = rgb_pairs_gt[:, 0].astype(np.float32)
    g = rgb_pairs_gt[:, 1].astype(np.float32)
    b = rgb_pairs_gt[:, 2].astype(np.float32)
    luma = (0.299 * r + 0.587 * g + 0.114 * b).clip(0.0, 255.0).astype(np.uint8)

    per_pair_displacements: list[float] = []

    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        gt_chunk = rgb_pairs_gt[start:stop].astype(np.float32)
        # Baseline SegNet on GT RGB.
        baseline_logits = run_mlx_segnet_nchw(mlx_segnet_adapter, gt_chunk)
        baseline_argmax = np.argmax(baseline_logits, axis=1)
        # Reconstruct each pair's v8-LUT-rendered RGB.
        recon_rgb_list: list[np.ndarray] = []
        for p_local in range(stop - start):
            p_global = start + p_local
            recon = lookup_rgb_via_chroma_lut(
                luma[p_global], cls_full[p_global], chroma_lut_v8
            )  # (H, W, 3) uint8
            # Transpose to (3, H, W) for SegNet input.
            recon_rgb_list.append(recon.transpose(2, 0, 1))
        recon_chunk_uint8 = np.stack(recon_rgb_list, axis=0).astype(np.uint8)
        recon_chunk = recon_chunk_uint8.astype(np.float32)
        # Mutated SegNet on reconstructed RGB.
        mutated_logits = run_mlx_segnet_nchw(mlx_segnet_adapter, recon_chunk)
        mutated_argmax = np.argmax(mutated_logits, axis=1)
        # Resample if SegNet output spatial shape differs from input.
        if baseline_argmax.shape != mutated_argmax.shape:
            # Defensive — should match since both went through identical pipeline.
            raise MLXIterationError(
                f"baseline_argmax shape {baseline_argmax.shape} != "
                f"mutated_argmax shape {mutated_argmax.shape}; SegNet pipeline "
                f"inconsistency"
            )
        # Per-pair displacement.
        for p_local in range(stop - start):
            disp = float(
                (baseline_argmax[p_local] != mutated_argmax[p_local]).mean()
            )
            per_pair_displacements.append(disp)

    mean_disp = (
        sum(per_pair_displacements) / len(per_pair_displacements)
        if per_pair_displacements
        else 0.0
    )
    max_disp = max(per_pair_displacements) if per_pair_displacements else 0.0
    in_noise_floor = mean_disp <= noise_floor_threshold
    recommended_proceed = not in_noise_floor

    return SegNetArgmaxDisplacementVerdict(
        num_pairs_probed=n,
        argmax_displacement_fraction=mean_disp,
        noise_floor_threshold=noise_floor_threshold,
        in_noise_floor=in_noise_floor,
        recommended_proceed=recommended_proceed,
        max_per_pair_displacement=max_disp,
        mean_per_pair_displacement=mean_disp,
    )
