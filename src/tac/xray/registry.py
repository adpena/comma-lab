# SPDX-License-Identifier: MIT
"""Canonical xray primitive registry.

Maps primitive ``name`` strings to :class:`XRayPrimitiveSpec` rows.
Used by the solver stack (sensitivity_map / Pareto / bit_allocator /
autopilot / continual_learning / probe-disambiguator) to discover
which xray primitives produce which wire-in signals.

This is a SISTER registry to :func:`tac.composition.registry.canonical_primitive_inventory`
— that registry covers PACKET-COMPILER primitives (archive-building);
THIS registry covers ANALYTIC primitives (archive/substrate/scorer
inspection).

Lane: ``lane_xray_canon_math_findings_wire_in_20260514``.

Cross-references
----------------
- Sister registry (packet-compiler primitives):
  :func:`tac.composition.registry.canonical_primitive_inventory`
- Catalog #169 (canonical primitive inventory pattern)
- CLAUDE.md "Subagent coherence-by-default" 6-hook wire-in non-negotiable
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from tac.xray.base import CANONICAL_WIRE_IN_HOOKS, EvidenceGrade, WireInHook

XRAY_REGISTRY_SCHEMA_VERSION = "tac_xray_registry_v1"


@dataclass(frozen=True)
class XRayPrimitiveSpec:
    """One row in the xray primitive registry.

    Each spec is FROZEN metadata: a primitive's spec NEVER changes after
    landing. Updating a primitive's metadata means bumping
    :data:`XRAY_REGISTRY_SCHEMA_VERSION`.

    Attributes
    ----------
    primitive_name : str
        Canonical name (registry key). Matches the primitive class's
        ``.name`` attribute.
    canonical_module : str
        Module path the primitive class lives at (e.g.,
        ``"tac.xray.mdl_scorer_conditional"``).
    canonical_symbol : str
        Class/function name within the module (e.g.,
        ``"ScorerConditionalMDLEstimator"``).
    category : str
        Top-level category for grouping. One of:
        ``"information-geometric"``, ``"scorer-internal"``,
        ``"codec-axis"``, ``"unified-action"``, ``"codec-primitive"``.
    description : str
        One-paragraph summary of what the primitive computes.
    primary_finding : str
        The empirical / mathematical anchor (e.g.,
        ``"80.7% of camera-pixel directions in bilinear resize nullspace"``).
        Tagged with evidence grade.
    evidence_grade : EvidenceGrade
        Evidence grade of the primary finding.
    wire_in_hooks : tuple[WireInHook, ...]
        Non-empty subset of the 6 canonical hooks this primitive engages.
    composes_with : tuple[str, ...]
        Other primitive names this primitive composes with naturally.
    upstream_memo : str
        Path to the math memo / research ledger that anchors this primitive
        (relative to repo root).
    """

    primitive_name: str
    canonical_module: str
    canonical_symbol: str
    category: str
    description: str
    primary_finding: str
    evidence_grade: EvidenceGrade
    wire_in_hooks: tuple[WireInHook, ...]
    composes_with: tuple[str, ...] = ()
    upstream_memo: str = ""

    def __post_init__(self) -> None:
        if not self.primitive_name:
            raise ValueError("primitive_name must be non-empty")
        if not self.wire_in_hooks:
            raise ValueError(
                f"primitive {self.primitive_name!r} declared zero wire-in "
                "hooks — orphan-work failure mode per CLAUDE.md non-negotiable"
            )
        for hook in self.wire_in_hooks:
            if hook not in CANONICAL_WIRE_IN_HOOKS:
                raise ValueError(
                    f"primitive {self.primitive_name!r} declared unknown "
                    f"hook {hook!r}; must be subset of {CANONICAL_WIRE_IN_HOOKS}"
                )


# ── Canonical xray primitive inventory ────────────────────────────────────


def canonical_xray_primitive_inventory() -> list[XRayPrimitiveSpec]:
    """Return the canonical xray primitives wired into the solver stack.

    Per the 2026-05-14 wire-in sweep ``lane_xray_canon_math_findings_wire_in_20260514``,
    these 13 primitives promote one-off math findings + research-memo
    derivations into canonical composable tools that the solver stack
    consumes.

    Categories (5):
      information-geometric (F1, F2, F4):
        1. ``mdl_scorer_conditional`` — Z1 multi-tier MDL ablation
        2. ``shannon_vector_r_d`` — vector R(D) bound estimator (Shannon 1959)
        4. ``score_lipschitz`` — Lipschitz constant via finite-difference

      codec-axis (F3, F5, F6, F10):
        3. ``bilinear_resize_nullspace`` — 80.7% nullspace finding
        5. ``vq_codebook_coverage`` — K=4096 codebook coverage
        6. ``wavelet_hf_energy`` — HF-energy floor estimate
        10. ``yuv6_sublattice_geometry`` — 4-luma + 2-chroma sublattice

      scorer-internal (F7, F8, F9, F14):
        7. ``segnet_margin_polytope`` — per-pixel logit margin map
        8. ``posenet_se3_lie_algebra`` — Cartan-Killing pose distance
        9. ``per_pair_score_decomposition`` — heterogeneous pair selector
        14. ``pairset_component_marginal`` — exact-axis component deltas

      unified-action (F11):
        11. ``unified_action_principle`` — Wasserstein × Fisher × tropical

      codec-primitive (F12 pair):
        12. ``predictive_coding_hierarchy`` — Rao-Ballard 1999 residual codec
        13. ``foveation_ego_motion`` — Gibson 1950 ego-motion-weighted budget
    """
    return [
        # ── Information-geometric (F1, F2, F4) ───────────────────────────
        XRayPrimitiveSpec(
            primitive_name="mdl_scorer_conditional",
            canonical_module="tac.xray.mdl_scorer_conditional",
            canonical_symbol="ScorerConditionalMDLEstimator",
            category="information-geometric",
            description=(
                "Three-tier MDL estimator (structural / sampled byte-level / "
                "post-decode perturbation) returning per-section byte entropy "
                "+ scorer-conditional entropy delta. Wraps the canonical "
                "tac.analysis.scorer_conditional_mdl manifest into a typed "
                "XRayPrimitive surface and delegates structural tier to the "
                "existing build_scorer_conditional_mdl_ablation flow."
            ),
            primary_finding=(
                "Z1 ablation: A1 archive (178,262 B) shows ~95% scorer-"
                "indifferent byte density on the within-class delta — "
                "consistent with 3-order-of-magnitude Shannon headroom "
                "[mathematical-derivation; deep_math §1.3 + zen_floor §Z1]"
            ),
            evidence_grade="mathematical-derivation",
            wire_in_hooks=(
                "continual_learning",
                "probe_disambiguator",
                "cathedral_autopilot",
            ),
            composes_with=("shannon_vector_r_d", "per_pair_score_decomposition"),
            upstream_memo=".omx/research/zen_floor_field_medal_grade_council_20260514.md",
        ),
        XRayPrimitiveSpec(
            primitive_name="shannon_vector_r_d",
            canonical_module="tac.xray.shannon_vector_r_d",
            canonical_symbol="ShannonVectorRDEstimator",
            category="information-geometric",
            description=(
                "Shannon 1959 vector R(D) lower bound for the contest's "
                "3-axis (d_seg, d_pose, rate) Pareto. Returns a "
                "(bits_floor, confidence_band) tuple. Used by the Pareto "
                "solver as the lower-left vertex of the achievable region."
            ),
            primary_finding=(
                "At PR101 operating point (seg=0.067, pose=0.018), "
                "R_min ≈ 100 bytes vs A1's 178,262 B archive — ~3 orders "
                "of magnitude theoretical headroom [first-principles-bound]"
            ),
            evidence_grade="first-principles-bound",
            wire_in_hooks=(
                "pareto_constraint",
                "sensitivity_map",
                "probe_disambiguator",
            ),
            composes_with=("score_lipschitz", "mdl_scorer_conditional"),
            upstream_memo=".omx/research/deep_math_geometry_manifolds_synthesis_20260514.md",
        ),
        XRayPrimitiveSpec(
            primitive_name="score_lipschitz",
            canonical_module="tac.xray.score_lipschitz",
            canonical_symbol="ScoreVsArchiveLipschitz",
            category="information-geometric",
            description=(
                "Estimates the score-vs-archive Lipschitz constant L such "
                "that |S(A) - S(B)| <= L * d_hamming(A, B). Computed via "
                "finite-difference single-bit / single-byte flips at random "
                "sample positions. inverse(L) bounds the zen-floor's minimal "
                "byte-budget granularity (Tao shower-thought, deep-math §13)."
            ),
            primary_finding=(
                "L_max approx 8.32e-8 score-units per archive bit at PR101 "
                "operating point [mathematical-derivation; deep_math §1.3 "
                "Lagrangian-dual derivation]"
            ),
            evidence_grade="mathematical-derivation",
            wire_in_hooks=(
                "pareto_constraint",
                "bit_allocator",
                "probe_disambiguator",
            ),
            composes_with=("shannon_vector_r_d", "bilinear_resize_nullspace"),
            upstream_memo=".omx/research/deep_math_geometry_manifolds_synthesis_20260514.md",
        ),
        # ── Codec-axis (F3, F5, F6, F10) ─────────────────────────────────
        XRayPrimitiveSpec(
            primitive_name="bilinear_resize_nullspace",
            canonical_module="tac.xray.bilinear_resize_nullspace",
            canonical_symbol="BilinearResizeNullspace",
            category="codec-axis",
            description=(
                "Computes the (1164, 874) -> (384, 512) bilinear resize "
                "matrix's left-nullspace dimension. The (camera_size -> "
                "scorer_input_size) projection is a low-rank linear map; "
                "camera-pixel directions in the nullspace are FREE BITS — "
                "scorer cannot distinguish perturbations along them."
            ),
            primary_finding=(
                "80.7% of camera-pixel directions (820,728 of 1,016,536) "
                "are in the bilinear-resize left-nullspace — invisible to "
                "SegNet/PoseNet preprocess pipeline [mathematical-derivation; "
                "deep_math §2.4 + structural-code-contract upstream/modules.py:73,108-109]"
            ),
            evidence_grade="mathematical-derivation",
            wire_in_hooks=(
                "sensitivity_map",
                "bit_allocator",
                "probe_disambiguator",
            ),
            composes_with=("yuv6_sublattice_geometry", "segnet_margin_polytope"),
            upstream_memo=".omx/research/deep_math_geometry_manifolds_synthesis_20260514.md",
        ),
        XRayPrimitiveSpec(
            primitive_name="vq_codebook_coverage",
            canonical_module="tac.xray.vq_codebook_coverage",
            canonical_symbol="VQCodebookCoverage",
            category="codec-axis",
            description=(
                "Computes a vector-quantization codebook's coverage rate "
                "over patches drawn from a target video. Returns "
                "(coverage_fraction, codebook_byte_budget, "
                "per_patch_assignment_entropy). Used to lower-bound the "
                "zen-floor's codebook-bytes axis."
            ),
            primary_finding=(
                "van den Oord shower-thought: K=4096 codebook of 16x16 "
                "driving patches covers ~95% of upstream/videos/0.mkv "
                "patch distribution [council-deliberation; zen_floor §13]"
            ),
            evidence_grade="council-deliberation",
            wire_in_hooks=(
                "bit_allocator",
                "sensitivity_map",
                "probe_disambiguator",
            ),
            composes_with=("wavelet_hf_energy", "shannon_vector_r_d"),
            upstream_memo=".omx/research/zen_floor_field_medal_grade_council_20260514.md",
        ),
        XRayPrimitiveSpec(
            primitive_name="wavelet_hf_energy",
            canonical_module="tac.xray.wavelet_hf_energy",
            canonical_symbol="WaveletHFEnergy",
            category="codec-axis",
            description=(
                "Computes high-frequency energy of each frame in a "
                "wavelet basis (default db8). The HF energy above a "
                "distortion threshold lower-bounds the zen-floor's "
                "spatial-frequency-axis allocator (Mallat 1989 + the "
                "Mallat shower-thought in zen_floor §13)."
            ),
            primary_finding=(
                "Mallat shower-thought: wavelets diagonalize spatial "
                "covariance — zen-floor = Sum(HF energy above distortion "
                "threshold) [council-deliberation; zen_floor §13]"
            ),
            evidence_grade="council-deliberation",
            wire_in_hooks=(
                "bit_allocator",
                "sensitivity_map",
            ),
            composes_with=("vq_codebook_coverage", "bilinear_resize_nullspace"),
            upstream_memo=".omx/research/zen_floor_field_medal_grade_council_20260514.md",
        ),
        XRayPrimitiveSpec(
            primitive_name="yuv6_sublattice_geometry",
            canonical_module="tac.xray.yuv6_sublattice_geometry",
            canonical_symbol="YUV6SublatticeGeometry",
            category="codec-axis",
            description=(
                "Decomposes RGB frames into the YUV6 sublattice geometry "
                "the contest scorer uses: [Y00, Y10, Y01, Y11, U_sub, "
                "V_sub] at half-resolution. Computes per-sublattice score "
                "gradient norms to identify which luma sublattice is the "
                "bit-efficient channel for sensitivity-weighted bit allocation."
            ),
            primary_finding=(
                "rgb_to_yuv6 produces 4 luma + 2 chroma half-resolution "
                "sublattices; each RGB pixel belongs to ONE luma sublattice "
                "via (y%2, x%2) indexing [structural-code-contract upstream/"
                "frame_utils.py:50-78 + deep_math §2.2]"
            ),
            evidence_grade="structural-code-contract",
            wire_in_hooks=(
                "sensitivity_map",
                "bit_allocator",
            ),
            composes_with=("bilinear_resize_nullspace", "segnet_margin_polytope"),
            upstream_memo=".omx/research/deep_math_geometry_manifolds_synthesis_20260514.md",
        ),
        # ── Scorer-internal (F7, F8, F9) ─────────────────────────────────
        XRayPrimitiveSpec(
            primitive_name="segnet_margin_polytope",
            canonical_module="tac.xray.segnet_margin_polytope",
            canonical_symbol="SegNetLogitMarginPolytope",
            category="scorer-internal",
            description=(
                "Computes the per-pixel SegNet logit-margin map M(x, y) = "
                "top1_logit - top2_logit, plus a safe-perturbation polytope "
                "budget T such that any logit-space perturbation with "
                "norm <= T preserves argmax. Used by D1 substrate + future "
                "argmax-margin-aware codecs."
            ),
            primary_finding=(
                "SegNet distortion is argmax-disagreement-rate (Bernoulli "
                "via 5-class softmax); margin polytope is the natural "
                "safe-perturbation budget [structural-code-contract upstream/"
                "modules.py:111-113 + deep_math §2.5]"
            ),
            evidence_grade="structural-code-contract",
            wire_in_hooks=(
                "sensitivity_map",
                "bit_allocator",
                "probe_disambiguator",
            ),
            composes_with=(
                "bilinear_resize_nullspace",
                "yuv6_sublattice_geometry",
            ),
            upstream_memo=".omx/research/deep_math_geometry_manifolds_synthesis_20260514.md",
        ),
        XRayPrimitiveSpec(
            primitive_name="posenet_se3_lie_algebra",
            canonical_module="tac.xray.posenet_se3_lie_algebra",
            canonical_symbol="PoseNetSE3LieAlgebra",
            category="scorer-internal",
            description=(
                "Treats PoseNet's 6-D output as a Lie-algebra coordinate on "
                "se(3) and computes pose residuals via the Cartan-Killing "
                "metric (not Euclidean MSE). For high-motion pairs the "
                "curvature term dominates; the canonical PoseNet MSE loss "
                "is sub-optimal in that regime."
            ),
            primary_finding=(
                "10% of pairs (high-motion) sit in se(3) regions where "
                "Cartan-Killing diverges from Euclidean MSE; canonical loss "
                "is sub-optimal there [mathematical-derivation; deep_math §2.6]"
            ),
            evidence_grade="mathematical-derivation",
            wire_in_hooks=(
                "sensitivity_map",
                "probe_disambiguator",
            ),
            composes_with=("per_pair_score_decomposition",),
            upstream_memo=".omx/research/deep_math_geometry_manifolds_synthesis_20260514.md",
        ),
        XRayPrimitiveSpec(
            primitive_name="per_pair_score_decomposition",
            canonical_module="tac.xray.per_pair_score_decomposition",
            canonical_symbol="PerPairScoreDecomposition",
            category="scorer-internal",
            description=(
                "Decomposes the total contest score (1/N) Sum(100*seg + "
                "sqrt(10*pose)) + 25*B/N into per-pair contributions and "
                "returns a heterogeneous per-pair priority vector. The "
                "cathedral autopilot consumes the top-K pairs for "
                "selective bit-budget allocation."
            ),
            primary_finding=(
                "Per-pair score contribution is heterogeneous; top-10% of "
                "pairs typically dominate >50% of total score "
                "[mathematical-derivation; deep_math §4.1 contest formula]"
            ),
            evidence_grade="mathematical-derivation",
            wire_in_hooks=(
                "cathedral_autopilot",
                "sensitivity_map",
                "bit_allocator",
            ),
            composes_with=("posenet_se3_lie_algebra", "segnet_margin_polytope"),
            upstream_memo=".omx/research/deep_math_geometry_manifolds_synthesis_20260514.md",
        ),
        XRayPrimitiveSpec(
            primitive_name="pairset_component_marginal",
            canonical_module="tac.xray.pairset_component_marginal",
            canonical_symbol="PairsetComponentMarginalXRay",
            category="scorer-internal",
            description=(
                "Canonicalizes exact-axis pairset component marginals from "
                "auth-eval component traces. The primitive exposes per-axis "
                "safe/protected drop pairs, CPU/CUDA transfer diagnostics, "
                "and canonical refs into the master-gradient consumers and "
                "pairset component marginal equation."
            ),
            primary_finding=(
                "DQS1 drop-one pair0371 improves [contest-CPU] by the one-byte "
                "rate credit but regresses [contest-CUDA T4] because SegNet "
                "penalty exceeds rate credit [empirical-anchor; "
                "codex_findings_dqs1_pairset_observation_feedback_20260522]"
            ),
            evidence_grade="empirical-anchor",
            wire_in_hooks=(
                "sensitivity_map",
                "pareto_constraint",
                "bit_allocator",
                "cathedral_autopilot",
                "continual_learning",
                "probe_disambiguator",
            ),
            composes_with=(
                "per_pair_score_decomposition",
                "segnet_margin_polytope",
                "posenet_se3_lie_algebra",
                "score_lipschitz",
            ),
            upstream_memo=(
                ".omx/research/codex_findings_dqs1_pairset_observation_feedback_"
                "20260522T164706Z_codex.md"
            ),
        ),
        # ── Unified-action (F11) ──────────────────────────────────────────
        XRayPrimitiveSpec(
            primitive_name="unified_action_principle",
            canonical_module="tac.xray.unified_action_principle",
            canonical_symbol="UnifiedActionPrinciple",
            category="unified-action",
            description=(
                "Synthesizes the Amari Fisher metric, the Wasserstein W_2 "
                "projection, and the tropical-semiring zen-floor projection "
                "into a single Lagrangian objective. Theta* = argmin "
                "[W_2(p_source, p_theta) * g_Fisher(theta) * T_trop(z(theta))]. "
                "Used by the meta-Lagrangian solver as the META objective."
            ),
            primary_finding=(
                "GR-style action principle for archive optimization: ONE "
                "scalar action S_total, all track-Lagrangians composed via "
                "delta_S / delta_theta = 0 [mathematical-derivation; "
                "deep_math §5.4 + feedback_unified_lagrangian_action_principle_GR_style_20260509.md]"
            ),
            evidence_grade="mathematical-derivation",
            wire_in_hooks=(
                "sensitivity_map",
                "pareto_constraint",
                "bit_allocator",
                "cathedral_autopilot",
            ),
            composes_with=(),  # The unified action is terminal; doesn't compose with sub-primitives
            upstream_memo=".omx/research/deep_math_geometry_manifolds_synthesis_20260514.md",
        ),
        # ── Codec-primitive (F12 pair) ────────────────────────────────────
        XRayPrimitiveSpec(
            primitive_name="predictive_coding_hierarchy",
            canonical_module="tac.xray.predictive_coding_hierarchy",
            canonical_symbol="PredictiveCodingHierarchy",
            category="codec-primitive",
            description=(
                "Rao-Ballard 1999 hierarchical predictive-coding analyzer: "
                "given an existing PredictiveCodingWeights residual-coding "
                "model, estimates the per-pair top-down prediction error "
                "and the residual byte budget at each level of the "
                "hierarchy. Surfaces opportunities to deepen the hierarchy "
                "or re-balance level capacities."
            ),
            primary_finding=(
                "Time-Traveler L5 substrate uses a single-level Rao-Ballard "
                "hierarchy; deeper hierarchies (2-3 levels) predicted to "
                "yield additional rate savings via cross-frame redundancy "
                "[council-deliberation; time_traveler_architecture_reverse_engineered]"
            ),
            evidence_grade="council-deliberation",
            wire_in_hooks=(
                "bit_allocator",
                "sensitivity_map",
                "probe_disambiguator",
            ),
            composes_with=("foveation_ego_motion",),
            upstream_memo=".omx/research/time_traveler_architecture_reverse_engineered_20260513.md",
        ),
        XRayPrimitiveSpec(
            primitive_name="foveation_ego_motion",
            canonical_module="tac.xray.foveation_ego_motion",
            canonical_symbol="FoveationEgoMotionAnalyzer",
            category="codec-primitive",
            description=(
                "Gibson 1950 ego-motion-matched foveation analyzer: given a "
                "stream of ego-motion pose deltas (from PoseNet or LAPose), "
                "computes a per-pixel bit-budget weighting that allocates "
                "more bits to the focus-of-expansion region and fewer bits "
                "to the parafoveal periphery. Used by D4 + future "
                "ego-motion-aware substrates."
            ),
            primary_finding=(
                "Focus-of-expansion concentrates ~70% of usable visual "
                "information in ~25% of pixels (driving sequences) — "
                "per-pixel sensitivity weighting from ego motion "
                "[council-deliberation; time_traveler L5 + Gibson 1950]"
            ),
            evidence_grade="council-deliberation",
            wire_in_hooks=(
                "sensitivity_map",
                "bit_allocator",
                "probe_disambiguator",
            ),
            composes_with=("predictive_coding_hierarchy", "posenet_se3_lie_algebra"),
            upstream_memo=".omx/research/time_traveler_architecture_reverse_engineered_20260513.md",
        ),
    ]


def get_xray_primitive_spec(name: str) -> XRayPrimitiveSpec:
    """Return the spec for ``name``; raise ValueError if not found."""
    for spec in canonical_xray_primitive_inventory():
        if spec.primitive_name == name:
            return spec
    raise ValueError(
        f"unknown xray primitive {name!r}; canonical inventory: "
        f"{[s.primitive_name for s in canonical_xray_primitive_inventory()]}"
    )


def specs_by_hook(hook: WireInHook) -> list[XRayPrimitiveSpec]:
    """Return all specs whose ``wire_in_hooks`` includes ``hook``.

    Used by hook-side consumers (sensitivity_map / Pareto / bit_allocator /
    autopilot / continual_learning / probe-disambiguator) to discover
    which xray primitives produce signals they should consume.
    """
    if hook not in CANONICAL_WIRE_IN_HOOKS:
        raise ValueError(
            f"unknown hook {hook!r}; must be one of {CANONICAL_WIRE_IN_HOOKS}"
        )
    return [
        s for s in canonical_xray_primitive_inventory() if hook in s.wire_in_hooks
    ]


def specs_by_category(category: str) -> list[XRayPrimitiveSpec]:
    """Return all specs in ``category``."""
    return [
        s
        for s in canonical_xray_primitive_inventory()
        if s.category == category
    ]


def serialize_xray_inventory() -> list[Mapping[str, Any]]:
    """JSON-friendly serialization of the xray inventory."""
    out: list[Mapping[str, Any]] = []
    for spec in canonical_xray_primitive_inventory():
        d = dataclasses.asdict(spec)
        d["wire_in_hooks"] = list(spec.wire_in_hooks)
        d["composes_with"] = list(spec.composes_with)
        out.append(d)
    return out


__all__ = [
    "XRAY_REGISTRY_SCHEMA_VERSION",
    "XRayPrimitiveSpec",
    "canonical_xray_primitive_inventory",
    "get_xray_primitive_spec",
    "serialize_xray_inventory",
    "specs_by_category",
    "specs_by_hook",
]
