"""16-substrate x 16-substrate composition orthogonality matrix.

Per operator directive 2026-05-11 ("wiring and integration") + the 16
non-HNeRV substrates landed across this session (5 residual basis from the
``tac.residual_basis.*`` family + 11 substrate families landed by sibling
subagents FF/GG/HH/II/KK + 2 NeRV-family bolt-ons), this module builds a
typed CompositionResult matrix that classifies the pairwise composability
of every substrate combination and exposes per-substrate Pareto rows that
the cathedral autopilot dispatch ranker (Deliverable 2) and theoretical
floor analyzer (Deliverable 3) consume.

Per CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable, the matrix is
**typed**, **machine-readable**, **planning_only**, and produces NO score
claims. Every cell carries an explicit ``[predicted; ...]`` evidence-grade
tag; no ``[contest-CUDA]`` claims are produced by this module.

Per CLAUDE.md "Cross-paradigm composition rules" + "substrate vs codec
composition meta-pattern" + the May-2026 race postmortem, the rules below
encode the **composability classes** the council has empirically validated:

- Two **residual** substrates compose orthogonally if they target
  DIFFERENT score axes (wavelet=spatial-high-freq + c3=temporal-low-freq
  -> orthogonal; wavelet+SIREN -> redundant because both spatial-frequency).
- Two **substrate-replacement** substrates are mutually exclusive (HNeRV
  vs NeRV vs VQ-VAE vs HNeRV-family-derivatives; only ONE full-RGB
  renderer can sit in an archive at the score-relevant slot).
- A **residual + a replacement** is stackable_serial (residual sits ON
  TOP of replacement's RGB output as a side-channel correction).
- **Self-compression** (SC++/Hessian-block-FP/MDL-FP4-TTO) composes
  orthogonally with any substrate as a quantization layer (it acts on
  parameters; substrate acts on representation).
- **Pose-axis** lanes (foveation/RAFT/LAPose) compose orthogonally with
  spatial substrates because they target the pose term directly via
  per-frame side-channels that don't intersect the renderer slot.
- **Categorical/ANR** is a substrate-replacement at the renderer slot.
- The **magic codec** (sister analysis at
  ``src/tac/packet_compiler/magic_codec.py``) composes orthogonally with
  every substrate because it is a META-codec on byte streams, but its
  row is deliberately zero-EV until a vendored byte-closed runtime and
  full-frame inflate-parity proof land.

Cross-references
----------------
- :mod:`tac.optimization.cross_paradigm_atoms` — adapter that consumes
  composition rows for meta-Lagrangian planning.
- :mod:`tools.cathedral_autopilot_autonomous_loop` — Deliverable 2 ranker
  consumes :func:`rank_substrates_by_ev_per_dollar` from THIS module.
- :mod:`tools.theoretical_floor_solver_v2` — Deliverable 3 refresh
  consumes :func:`per_substrate_predicted_floor` for Pareto bounds.

CLAUDE.md compliance tags
-------------------------
- ``planning_only_no_score_claim``
- ``no_mps_authoritative``
- ``no_tmp_paths``
- ``substrate_composition_orthogonality_matrix_v1``
- ``halt_and_ask_default_for_dispatch_recommendations``
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

# Schema constants pinned to v1 so downstream consumers (autopilot ranker,
# theoretical floor refresh) detect schema drift loudly.
SCHEMA_VERSION = "tac_substrate_composition_matrix_v1"

# Forbidden /tmp paths per CLAUDE.md FORBIDDEN_PATTERNS.
PLANNING_ONLY = True
SCORE_CLAIM = False
PROMOTION_ELIGIBLE = False
READY_FOR_EXACT_EVAL_DISPATCH = False


# ── Substrate taxonomy ────────────────────────────────────────────────────


class SubstrateClass(StrEnum):
    """Top-level taxonomy class.

    The class governs composability with siblings. Two ``RENDERER_REPLACEMENT``
    substrates are mutually exclusive at the renderer slot; two ``RESIDUAL``
    substrates compose if they target DIFFERENT score axes; ``SELF_COMPRESSION``
    composes with everything; ``META_CODEC`` composes with everything.
    """

    RESIDUAL = "residual"  # Sits on top of an existing substrate's RGB output.
    RENDERER_REPLACEMENT = "renderer_replacement"  # Full RGB renderer.
    SELF_COMPRESSION = "self_compression"  # Acts on parameters / weights.
    POSE_AXIS_SIDECHANNEL = "pose_axis_sidechannel"  # Targets pose term directly.
    META_CODEC = "meta_codec"  # Auto-selecting codec wrapper (e.g., magic codec).
    BOLT_ON = "bolt_on"  # Composes with a host substrate (e.g., FiLM modulator).


class ScoreAxis(StrEnum):
    """Which score-axis term the substrate primarily targets.

    Per CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent",
    at PR106 r2 frontier the marginal-value ranking is POSE > SEG > RATE.
    The axis classification informs orthogonality between residual substrates.
    """

    RATE = "rate"  # Bytes-axis improvement.
    SEG = "seg"  # SegNet distortion improvement.
    POSE = "pose"  # PoseNet distortion improvement.
    MIXED = "mixed"  # Affects multiple axes simultaneously.


class Composability(StrEnum):
    """Pairwise composability verdict.

    The 8 classes are:
    - ``orthogonal`` -- additive byte savings AND distortion improvements
      can be assumed independent (alpha = 1.0); the composition's predicted
      delta is the sum of the per-substrate predicted deltas.
    - ``redundant`` -- both substrates capture the same score-axis signal;
      composition's predicted delta is approximately the larger of the two
      individual deltas (not the sum).
    - ``antagonistic`` -- one substrate erodes the other's gain; predicted
      delta is LESS than the larger individual.
    - ``replacement`` -- mutually exclusive (only one renderer slot exists);
      composition is forbidden at the archive level.
    - ``stackable_serial`` -- substrate B sits on top of substrate A's
      output (e.g., wavelet residual on top of NeRV's RGB).
    - ``stackable_parallel`` -- substrates run independently and their
      bytes accumulate side-by-side (e.g., RAFT pose stream + foveation
      field; both are pose-axis sidechannels but on different mechanisms).
    - ``stackable_cascade`` -- substrate B refines substrate A's output
      iteratively (e.g., MDL-FP4-TTO refining SC++ Stage-1).
    - ``incompatible`` -- composition would violate a hard contract (e.g.,
      a substrate's runtime closure list excludes the other's dependency).
    """

    ORTHOGONAL = "orthogonal"
    REDUNDANT = "redundant"
    ANTAGONISTIC = "antagonistic"
    REPLACEMENT = "replacement"
    STACKABLE_SERIAL = "stackable_serial"
    STACKABLE_PARALLEL = "stackable_parallel"
    STACKABLE_CASCADE = "stackable_cascade"
    INCOMPATIBLE = "incompatible"


@dataclass(frozen=True)
class SubstrateRow:
    """One substrate inventory row.

    Every substrate that ships in the matrix appears exactly once. Fields
    are typed and frozen so downstream consumers cannot tamper with the
    declared compatibility surface.
    """

    substrate_id: str
    name: str
    substrate_class: SubstrateClass
    target_axis: ScoreAxis
    format_id: int  # The 1-byte format_id allocated in archive grammar.
    magic_bytes: str  # 4-char ASCII magic.
    runtime_dep_closure: tuple[str, ...]  # e.g., ("torch", "brotli").
    byte_budget_band: tuple[int, int]  # (min_bytes, max_bytes) for archive bytes.
    predicted_delta_alone_band: tuple[float, float]  # (low, high) score delta.
    requires_score_aware_training: bool
    landed_at: str  # e.g., "2026-05-11" or "2026-05-09".
    landing_memo: str  # e.g., "feedback_*_landed_*.md" filename token.

    def predicted_delta_alone_midpoint(self) -> float:
        return 0.5 * (self.predicted_delta_alone_band[0] + self.predicted_delta_alone_band[1])


@dataclass(frozen=True)
class CompositionResult:
    """Typed pairwise composition verdict.

    Per CLAUDE.md "score_claim_must_be_False_until_contest_CUDA" the
    ``score_claim`` invariant is hard-coded False. ``expected_alpha`` is a
    PREDICTED multiplier (alpha=1 means independent additive; alpha=0
    means fully redundant; alpha<0 means antagonistic; alpha>1 is super-
    additive per Volterra correction in the theoretical floor solver).
    """

    substrate_a: str
    substrate_b: str
    composability: Composability
    expected_alpha: float
    byte_overhead: int
    score_axis_target_a: ScoreAxis
    score_axis_target_b: ScoreAxis
    format_id_collision_risk: bool
    rationale: str
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False


# ── Canonical 16-substrate inventory (FROZEN per landing memos) ───────────
#
# Per CLAUDE.md "Lane maturity registry" + "Cross-paradigm composition rules"
# the 16-substrate inventory must match what actually shipped in the
# 2026-05-09 -> 2026-05-11 work envelope. The inventory below is the
# canonical mapping; if a substrate is added to the codebase, its
# SubstrateRow MUST land in this same file or the matrix consumers will
# silently miss it.


def canonical_substrate_inventory() -> list[SubstrateRow]:
    """Return the 48-row substrate inventory.

    Composition:
    - 24 legacy rows (residual basis 5 + pose-axis 3 + self-compression 3 +
      NeRV-family 5 + NeRV/MNeRV/VQVAE 3 + ANR/categorical 2 +
      magic codec 1 + bolt-ons 2)
    - 15 FIX-J substrate-scaffold rows (Fields-medal 2026-05-12 council
      design under `src/tac/substrates/<name>/`)
    - 9 WAVE-A-2 TRADITION 2 rows for older single-file `<name>_renderer.py`
      substrates that pre-date the substrate-scaffold subpackage discipline
      (CANON-1.A explicit-taxonomy resolution, 2026-05-12)

    Order is stable (alphabetical by substrate_class then substrate_id) to
    preserve deterministic matrix construction across runs.

    Cross-references for each row:
    - Residual basis (5 rows): ``feedback_wavelet_residual_basis_pr106_scaffold_landed_20260511.md``
      + ``feedback_nonhnerv_residual_basis_scaffolds_landed_20260511.md``
    - Pose-axis (3 rows): ``feedback_pose_axis_lanes_full_scaffolds_landed_20260511.md``
    - Self-compression (3 rows): ``feedback_self_compression_family_scpp_hessian_mdl_landed_20260511.md``
    - NeRV-family (5 rows from KK + 3 from HH = 8 total renderer-replacement rows):
      ``feedback_nerv_family_expansion_blocknerv_ffnerv_dsnerv_hinerv_tcnerv_landed_20260511.md``
      + ``feedback_nerv_mnerv_vqvae_full_renderer_substrate_trainers_landed_20260511.md``
    - ANR/categorical (2 rows): ``feedback_anr_token_renderer_categorical_full_substrate_landed_20260511.md``
    - Bolt-ons (2 rows): NeRV-Enc/Dec separated + FiLM pose conditioning
      from ``feedback_nerv_family_expansion_blocknerv_ffnerv_dsnerv_hinerv_tcnerv_landed_20260511.md``
    - Magic codec (1 row): ``feedback_magic_codec_auto_selector_landed_20260511.md``
    - FIX-J substrate-scaffold packages (15 rows, 2026-05-12 Fields-medal council):
      ``feedback_fix_j_substrate_compressai_inventory_wire_in_landed_20260512.md``
      Covers sane_hnerv (alpha), balle_renderer (beta), hybrid_renderer_residual
      (gamma), self_compress_nn (delta), pr101_lc_v2_clone (forensic), cool_chic_full_renderer,
      wavelet_full_renderer, grayscale_lut, vq_vae_substrate, siren_substrate,
      block_nerv_substrate, tc_nerv_substrate, ff_nerv_substrate,
      ds_nerv_substrate, hi_nerv_substrate.
    - WAVE-A-2 TRADITION 2 single-file substrates (9 rows, 2026-05-12 CANON-1.A):
      ``feedback_wave_a_2_taxonomy_inventory_drift_landed_20260512.md``
      Adds: cnerv, e_nerv, ego_nerv, lane_12_v2_nerv_as_renderer, nervdc, quantizr_faithful,
      mlx_mask_renderer, dp_sims_renderer, diffusion_renderer.
      Cross-ref tradition taxonomy memo at
      ``.omx/research/substrate_tradition_taxonomy_20260512.md``.
    """
    rows: list[SubstrateRow] = [
        # ── 5 RESIDUAL basis substrates ──
        SubstrateRow(
            substrate_id="wavelet_residual",
            name="Wavelet residual basis (Mallat)",
            substrate_class=SubstrateClass.RESIDUAL,
            target_axis=ScoreAxis.RATE,
            format_id=0x10,
            magic_bytes="WVRS",
            runtime_dep_closure=("torch", "brotli", "numpy"),
            byte_budget_band=(2_000, 8_000),
            predicted_delta_alone_band=(-0.0010, -0.0001),
            requires_score_aware_training=True,
            landed_at="2026-05-11",
            landing_memo="feedback_wavelet_residual_basis_pr106_scaffold_landed_20260511",
        ),
        SubstrateRow(
            substrate_id="cool_chic_residual",
            name="Cool-Chic hierarchical pyramid residual",
            substrate_class=SubstrateClass.RESIDUAL,
            target_axis=ScoreAxis.RATE,
            format_id=0x11,
            magic_bytes="CCRS",
            runtime_dep_closure=("torch", "brotli", "numpy"),
            byte_budget_band=(3_000, 12_000),
            predicted_delta_alone_band=(-0.0015, -0.0002),
            requires_score_aware_training=True,
            landed_at="2026-05-11",
            landing_memo="feedback_nonhnerv_residual_basis_scaffolds_landed_20260511",
        ),
        SubstrateRow(
            substrate_id="c3_residual",
            name="C3 (Cool-Chic + temporal hyperprior) residual",
            substrate_class=SubstrateClass.RESIDUAL,
            target_axis=ScoreAxis.MIXED,  # Captures temporal motion + spatial.
            format_id=0x12,
            magic_bytes="C3RS",
            runtime_dep_closure=("torch", "brotli", "numpy"),
            byte_budget_band=(3_500, 13_000),
            predicted_delta_alone_band=(-0.0018, -0.0003),
            requires_score_aware_training=True,
            landed_at="2026-05-11",
            landing_memo="feedback_nonhnerv_residual_basis_scaffolds_landed_20260511",
        ),
        SubstrateRow(
            substrate_id="siren_residual",
            name="SIREN sinusoidal coordinate-MLP residual",
            substrate_class=SubstrateClass.RESIDUAL,
            target_axis=ScoreAxis.RATE,
            format_id=0x13,
            magic_bytes="SIRS",
            runtime_dep_closure=("torch", "numpy"),
            byte_budget_band=(2_500, 10_000),
            predicted_delta_alone_band=(-0.0012, -0.0001),
            requires_score_aware_training=True,
            landed_at="2026-05-11",
            landing_memo="feedback_nonhnerv_residual_basis_scaffolds_landed_20260511",
        ),
        SubstrateRow(
            substrate_id="coordinate_mlp_residual",
            name="Coordinate-MLP family-agnostic residual",
            substrate_class=SubstrateClass.RESIDUAL,
            target_axis=ScoreAxis.RATE,
            format_id=0x14,
            magic_bytes="CMRS",
            runtime_dep_closure=("torch", "numpy"),
            byte_budget_band=(2_500, 10_000),
            predicted_delta_alone_band=(-0.0010, -0.0001),
            requires_score_aware_training=True,
            landed_at="2026-05-11",
            landing_memo="feedback_nonhnerv_residual_basis_scaffolds_landed_20260511",
        ),
        # ── 3 POSE_AXIS_SIDECHANNEL substrates ──
        SubstrateRow(
            substrate_id="foveation_field",
            name="Telescopic foveation field",
            substrate_class=SubstrateClass.POSE_AXIS_SIDECHANNEL,
            target_axis=ScoreAxis.POSE,
            format_id=0x30,
            magic_bytes="FOVE",
            runtime_dep_closure=("torch", "brotli"),
            byte_budget_band=(200, 500),
            predicted_delta_alone_band=(-0.0008, -0.0002),
            requires_score_aware_training=True,
            landed_at="2026-05-11",
            landing_memo="feedback_pose_axis_lanes_full_scaffolds_landed_20260511",
        ),
        SubstrateRow(
            substrate_id="raft_pose_stream",
            name="RAFT optical-flow pose stream",
            substrate_class=SubstrateClass.POSE_AXIS_SIDECHANNEL,
            target_axis=ScoreAxis.POSE,
            format_id=0x31,
            magic_bytes="RAFT",
            runtime_dep_closure=("torch", "brotli"),
            byte_budget_band=(2_000, 4_096),
            predicted_delta_alone_band=(-0.0015, -0.0004),
            requires_score_aware_training=False,  # Compress-time RAFT is precomputed.
            landed_at="2026-05-11",
            landing_memo="feedback_pose_axis_lanes_full_scaffolds_landed_20260511",
        ),
        SubstrateRow(
            substrate_id="lapose_motion_atom_allocator",
            name="LAPose inverse-dynamics motion-atom allocator",
            substrate_class=SubstrateClass.POSE_AXIS_SIDECHANNEL,
            target_axis=ScoreAxis.POSE,
            format_id=0x32,
            magic_bytes="LAPO",
            runtime_dep_closure=("torch", "brotli"),
            byte_budget_band=(500, 1_024),
            predicted_delta_alone_band=(-0.0010, -0.0001),  # research_only.
            requires_score_aware_training=True,
            landed_at="2026-05-11",
            landing_memo="feedback_pose_axis_lanes_full_scaffolds_landed_20260511",
        ),
        # ── 3 SELF_COMPRESSION substrates ──
        SubstrateRow(
            substrate_id="scpp_substrate",
            name="SC++ block-FP self-compression substrate",
            substrate_class=SubstrateClass.SELF_COMPRESSION,
            target_axis=ScoreAxis.RATE,
            format_id=0x40,
            magic_bytes="SCPP",
            runtime_dep_closure=("torch", "brotli"),
            byte_budget_band=(40_000, 80_000),
            predicted_delta_alone_band=(-0.0050, 0.0050),
            requires_score_aware_training=True,
            landed_at="2026-05-11",
            landing_memo="feedback_self_compression_family_scpp_hessian_mdl_landed_20260511",
        ),
        SubstrateRow(
            substrate_id="hessian_block_fp",
            name="Hessian block-FP allocator (Boyd ADMM water-filling)",
            substrate_class=SubstrateClass.SELF_COMPRESSION,
            target_axis=ScoreAxis.RATE,
            format_id=0x41,
            magic_bytes="HBFP",
            runtime_dep_closure=("torch",),
            byte_budget_band=(0, 0),  # Acts on existing substrate weights.
            predicted_delta_alone_band=(-0.0030, -0.0010),
            requires_score_aware_training=False,  # Closed-form allocator.
            landed_at="2026-05-11",
            landing_memo="feedback_self_compression_family_scpp_hessian_mdl_landed_20260511",
        ),
        SubstrateRow(
            substrate_id="mdl_fp4_tto",
            name="MDL/FP4 test-time training (Stage-5 final TTO)",
            substrate_class=SubstrateClass.SELF_COMPRESSION,
            target_axis=ScoreAxis.MIXED,  # Refines weights against rate+distortion.
            format_id=0x42,
            magic_bytes="MFP4",
            runtime_dep_closure=("torch",),
            byte_budget_band=(0, 0),  # Acts on existing weights.
            predicted_delta_alone_band=(-0.0020, -0.0005),
            requires_score_aware_training=True,
            landed_at="2026-05-11",
            landing_memo="feedback_self_compression_family_scpp_hessian_mdl_landed_20260511",
        ),
        # ── 5 NeRV-family RENDERER_REPLACEMENT substrates (KK landing) ──
        SubstrateRow(
            substrate_id="blocknerv",
            name="BlockNeRV (tile-decomposed)",
            substrate_class=SubstrateClass.RENDERER_REPLACEMENT,
            target_axis=ScoreAxis.MIXED,
            format_id=0x60,
            magic_bytes="BNRV",
            runtime_dep_closure=("torch", "brotli"),
            byte_budget_band=(120_000, 200_000),
            predicted_delta_alone_band=(-0.0050, 0.0150),
            requires_score_aware_training=True,
            landed_at="2026-05-11",
            landing_memo="feedback_nerv_family_expansion_blocknerv_ffnerv_dsnerv_hinerv_tcnerv_landed_20260511",
        ),
        SubstrateRow(
            substrate_id="ffnerv",
            name="FFNeRV (Fourier-features)",
            substrate_class=SubstrateClass.RENDERER_REPLACEMENT,
            target_axis=ScoreAxis.MIXED,
            format_id=0x61,
            magic_bytes="FFNV",
            runtime_dep_closure=("torch", "brotli"),
            byte_budget_band=(120_000, 200_000),
            predicted_delta_alone_band=(-0.0050, 0.0150),
            requires_score_aware_training=True,
            landed_at="2026-05-11",
            landing_memo="feedback_nerv_family_expansion_blocknerv_ffnerv_dsnerv_hinerv_tcnerv_landed_20260511",
        ),
        SubstrateRow(
            substrate_id="dsnerv",
            name="DSNeRV (diffusion-supervised)",
            substrate_class=SubstrateClass.RENDERER_REPLACEMENT,
            target_axis=ScoreAxis.MIXED,
            format_id=0x62,
            magic_bytes="DSNV",
            runtime_dep_closure=("torch", "brotli"),
            byte_budget_band=(120_000, 200_000),
            predicted_delta_alone_band=(-0.0030, 0.0200),
            requires_score_aware_training=True,
            landed_at="2026-05-11",
            landing_memo="feedback_nerv_family_expansion_blocknerv_ffnerv_dsnerv_hinerv_tcnerv_landed_20260511",
        ),
        SubstrateRow(
            substrate_id="hinerv",
            name="HiNeRV (hierarchical)",
            substrate_class=SubstrateClass.RENDERER_REPLACEMENT,
            target_axis=ScoreAxis.MIXED,
            format_id=0x63,
            magic_bytes="HiNV",
            runtime_dep_closure=("torch", "brotli"),
            byte_budget_band=(120_000, 200_000),
            predicted_delta_alone_band=(-0.0070, 0.0100),
            requires_score_aware_training=True,
            landed_at="2026-05-11",
            landing_memo="feedback_nerv_family_expansion_blocknerv_ffnerv_dsnerv_hinerv_tcnerv_landed_20260511",
        ),
        SubstrateRow(
            substrate_id="tcnerv",
            name="TCNeRV (temporal-conv)",
            substrate_class=SubstrateClass.RENDERER_REPLACEMENT,
            target_axis=ScoreAxis.POSE,  # Temporal convolution targets motion.
            format_id=0x64,
            magic_bytes="TCNV",
            runtime_dep_closure=("torch", "brotli"),
            byte_budget_band=(120_000, 200_000),
            predicted_delta_alone_band=(-0.0050, 0.0120),
            requires_score_aware_training=True,
            landed_at="2026-05-11",
            landing_memo="feedback_nerv_family_expansion_blocknerv_ffnerv_dsnerv_hinerv_tcnerv_landed_20260511",
        ),
        # ── 3 RENDERER_REPLACEMENT (HH landing: NeRV/MNeRV/VQ-VAE) ──
        SubstrateRow(
            substrate_id="nerv_as_renderer",
            name="NeRV-as-renderer (Lane 12-v2)",
            substrate_class=SubstrateClass.RENDERER_REPLACEMENT,
            target_axis=ScoreAxis.MIXED,
            format_id=0x70,
            magic_bytes="NRVR",
            runtime_dep_closure=("torch", "brotli"),
            byte_budget_band=(150_000, 220_000),
            predicted_delta_alone_band=(-0.0080, 0.0050),  # PR100-band.
            requires_score_aware_training=True,
            landed_at="2026-05-11",
            landing_memo="feedback_nerv_mnerv_vqvae_full_renderer_substrate_trainers_landed_20260511",
        ),
        SubstrateRow(
            substrate_id="mnerv",
            name="MNeRV (Mallat scattering bandpass cascade)",
            substrate_class=SubstrateClass.RENDERER_REPLACEMENT,
            target_axis=ScoreAxis.MIXED,
            format_id=0x71,
            magic_bytes="MNRV",
            runtime_dep_closure=("torch", "brotli"),
            byte_budget_band=(150_000, 220_000),
            predicted_delta_alone_band=(-0.0070, 0.0080),
            requires_score_aware_training=True,
            landed_at="2026-05-11",
            landing_memo="feedback_nerv_mnerv_vqvae_full_renderer_substrate_trainers_landed_20260511",
        ),
        SubstrateRow(
            substrate_id="vqvae_as_full_renderer",
            name="VQ-VAE-as-full-renderer",
            substrate_class=SubstrateClass.RENDERER_REPLACEMENT,
            target_axis=ScoreAxis.MIXED,
            format_id=0x72,
            magic_bytes="VQRR",
            runtime_dep_closure=("torch", "brotli"),
            byte_budget_band=(120_000, 180_000),
            predicted_delta_alone_band=(-0.0090, 0.0060),
            requires_score_aware_training=True,
            landed_at="2026-05-11",
            landing_memo="feedback_nerv_mnerv_vqvae_full_renderer_substrate_trainers_landed_20260511",
        ),
        # ── 2 RENDERER_REPLACEMENT (II landing: ANR + categorical) ──
        SubstrateRow(
            substrate_id="anr_token_renderer_v62",
            name="ANR TokenRendererV62 (PR95 jas0xf-derived)",
            substrate_class=SubstrateClass.RENDERER_REPLACEMENT,
            target_axis=ScoreAxis.MIXED,
            format_id=0x50,
            magic_bytes="ANRV",
            runtime_dep_closure=("torch", "constriction", "pyppmd"),
            byte_budget_band=(175_000, 195_000),
            predicted_delta_alone_band=(-0.0050, 0.0100),  # PR95-band 0.193.
            requires_score_aware_training=True,
            landed_at="2026-05-11",
            landing_memo="feedback_anr_token_renderer_categorical_full_substrate_landed_20260511",
        ),
        SubstrateRow(
            substrate_id="categorical_substrate",
            name="Categorical per-pixel SegNet-class palette renderer",
            substrate_class=SubstrateClass.RENDERER_REPLACEMENT,
            target_axis=ScoreAxis.SEG,  # Conditioned on SegNet's argmax.
            format_id=0x51,
            magic_bytes="CATG",
            runtime_dep_closure=("torch", "constriction"),
            byte_budget_band=(120_000, 165_000),
            predicted_delta_alone_band=(-0.0020, 0.0200),
            requires_score_aware_training=True,
            landed_at="2026-05-11",
            landing_memo="feedback_anr_token_renderer_categorical_full_substrate_landed_20260511",
        ),
        # ── 2 BOLT_ON substrates (compose with host substrates) ──
        SubstrateRow(
            substrate_id="nerv_enc_dec_separated",
            name="NeRV Encoder/Decoder separated bolt-on",
            substrate_class=SubstrateClass.BOLT_ON,
            target_axis=ScoreAxis.RATE,  # Encoder is COMPRESS-TIME ONLY.
            format_id=0x80,
            magic_bytes="NEDB",
            runtime_dep_closure=("torch",),
            byte_budget_band=(0, 0),  # Encoder not shipped in archive.
            predicted_delta_alone_band=(-0.0010, 0.0010),  # Marginal alone.
            requires_score_aware_training=True,
            landed_at="2026-05-11",
            landing_memo="feedback_nerv_family_expansion_blocknerv_ffnerv_dsnerv_hinerv_tcnerv_landed_20260511",
        ),
        SubstrateRow(
            substrate_id="film_pose_conditioning",
            name="FiLM pose conditioning bolt-on",
            substrate_class=SubstrateClass.BOLT_ON,
            target_axis=ScoreAxis.POSE,
            format_id=0x81,
            magic_bytes="FILM",
            runtime_dep_closure=("torch",),
            byte_budget_band=(2_000, 8_000),  # Small MLP weights.
            predicted_delta_alone_band=(-0.0015, -0.0002),
            requires_score_aware_training=True,
            landed_at="2026-05-11",
            landing_memo="feedback_nerv_family_expansion_blocknerv_ffnerv_dsnerv_hinerv_tcnerv_landed_20260511",
        ),
        # ── 1 META_CODEC substrate (composes with everything) ──
        SubstrateRow(
            substrate_id="magic_codec",
            name="Magic codec auto-selector (meta-codec, planning-only)",
            substrate_class=SubstrateClass.META_CODEC,
            target_axis=ScoreAxis.RATE,
            format_id=0xF0,
            magic_bytes="MGIC",
            runtime_dep_closure=("repo_tac_required_until_vendored", "brotli"),
            byte_budget_band=(0, 1_000),  # Wrapper overhead.
            predicted_delta_alone_band=(0.0, 0.0),
            requires_score_aware_training=False,
            landed_at="2026-05-11",
            landing_memo="feedback_magic_codec_auto_selector_landed_20260511",
        ),
        # ── 15 FIX-J substrate-scaffold packages (Catalog #124 fields-medal) ──
        # Per LOOPCLOSE finding 2026-05-12: the substrate-scaffold subpackages
        # under ``src/tac/substrates/`` (Fields-medal grand council 2026-05-12
        # design) were absent from the canonical inventory. Each has an
        # archive grammar with a 4-byte magic and a Catalog #124-compliant
        # 8-fields-at-design-time declaration. Cross-references:
        # ``feedback_grand_council_fields_medal_substrate_design_20260512.md``
        # + ``feedback_fix_j_substrate_compressai_inventory_wire_in_landed_20260512.md``.
        # Format-id band 0x90-0x9E reserved for FIX-J scaffolds; 0x43 reserved
        # for self_compress_nn (SELF_COMPRESSION class extension).
        SubstrateRow(
            substrate_id="sane_hnerv",
            name="Sane HNeRV (Score-Aware HNeRV-extended, alpha primary)",
            substrate_class=SubstrateClass.RENDERER_REPLACEMENT,
            target_axis=ScoreAxis.MIXED,
            format_id=0x90,
            magic_bytes="SHV1",
            runtime_dep_closure=("torch", "brotli", "numpy"),
            byte_budget_band=(150_000, 220_000),
            predicted_delta_alone_band=(-0.0080, 0.0050),
            requires_score_aware_training=True,
            landed_at="2026-05-12",
            landing_memo="feedback_fix_j_substrate_compressai_inventory_wire_in_landed_20260512",
        ),
        SubstrateRow(
            substrate_id="balle_renderer",
            name="Ballé hyperprior-as-renderer (beta parallel scaffold)",
            substrate_class=SubstrateClass.RENDERER_REPLACEMENT,
            target_axis=ScoreAxis.RATE,
            format_id=0x91,
            magic_bytes="BRV1",
            runtime_dep_closure=("torch", "brotli", "compressai"),
            byte_budget_band=(150_000, 220_000),
            predicted_delta_alone_band=(-0.0060, 0.0050),
            requires_score_aware_training=True,
            landed_at="2026-05-12",
            landing_memo="feedback_fix_j_substrate_compressai_inventory_wire_in_landed_20260512",
        ),
        SubstrateRow(
            substrate_id="hybrid_renderer_residual",
            name="Hybrid renderer+residual basis (gamma deferred scaffold)",
            substrate_class=SubstrateClass.RENDERER_REPLACEMENT,
            target_axis=ScoreAxis.MIXED,
            format_id=0x92,
            magic_bytes="HRR1",
            runtime_dep_closure=("torch", "brotli"),
            byte_budget_band=(150_000, 220_000),
            predicted_delta_alone_band=(-0.0060, 0.0050),
            requires_score_aware_training=True,
            landed_at="2026-05-12",
            landing_memo="feedback_fix_j_substrate_compressai_inventory_wire_in_landed_20260512",
        ),
        SubstrateRow(
            substrate_id="self_compress_nn",
            name="Self-Compress NN (delta MDL weight clustering scaffold)",
            substrate_class=SubstrateClass.SELF_COMPRESSION,
            target_axis=ScoreAxis.RATE,
            format_id=0x43,
            magic_bytes="SCV1",
            runtime_dep_closure=("torch", "brotli"),
            byte_budget_band=(40_000, 90_000),
            predicted_delta_alone_band=(-0.0040, 0.0040),
            requires_score_aware_training=True,
            landed_at="2026-05-12",
            landing_memo="feedback_fix_j_substrate_compressai_inventory_wire_in_landed_20260512",
        ),
        SubstrateRow(
            substrate_id="pr101_lc_v2_clone",
            name="PR101 lc_v2 forensic apples-to-apples clone (research_only)",
            substrate_class=SubstrateClass.RENDERER_REPLACEMENT,
            target_axis=ScoreAxis.MIXED,
            format_id=0x93,
            magic_bytes="PR12",  # PR101 v2; no archive magic on disk yet.
            runtime_dep_closure=("torch", "brotli"),
            byte_budget_band=(175_000, 195_000),
            predicted_delta_alone_band=(-0.0010, 0.0010),
            requires_score_aware_training=True,
            landed_at="2026-05-12",
            landing_memo="feedback_fix_j_substrate_compressai_inventory_wire_in_landed_20260512",
        ),
        SubstrateRow(
            substrate_id="cool_chic_full_renderer",
            name="Cool-Chic full per-frame AR-latent renderer (L0 SKETCH)",
            substrate_class=SubstrateClass.RENDERER_REPLACEMENT,
            target_axis=ScoreAxis.RATE,
            format_id=0x94,
            magic_bytes="CCV1",
            runtime_dep_closure=("torch", "brotli"),
            byte_budget_band=(120_000, 200_000),
            predicted_delta_alone_band=(-0.0050, 0.0050),
            requires_score_aware_training=True,
            landed_at="2026-05-12",
            landing_memo="feedback_fix_j_substrate_compressai_inventory_wire_in_landed_20260512",
        ),
        SubstrateRow(
            substrate_id="wavelet_full_renderer",
            name="Wavelet (Mallat scattering) full renderer (L0 SKETCH)",
            substrate_class=SubstrateClass.RENDERER_REPLACEMENT,
            target_axis=ScoreAxis.RATE,
            format_id=0x95,
            magic_bytes="WLV1",
            runtime_dep_closure=("torch", "brotli"),
            byte_budget_band=(120_000, 200_000),
            predicted_delta_alone_band=(-0.0050, 0.0050),
            requires_score_aware_training=True,
            landed_at="2026-05-12",
            landing_memo="feedback_fix_j_substrate_compressai_inventory_wire_in_landed_20260512",
        ),
        SubstrateRow(
            substrate_id="grayscale_lut",
            name="Grayscale-LUT (Selfcomp analog mask paradigm, L0 SKETCH)",
            substrate_class=SubstrateClass.RENDERER_REPLACEMENT,
            target_axis=ScoreAxis.MIXED,
            format_id=0x96,
            magic_bytes="GLV1",
            runtime_dep_closure=("torch", "brotli"),
            byte_budget_band=(100_000, 180_000),
            predicted_delta_alone_band=(-0.0040, 0.0080),
            requires_score_aware_training=True,
            landed_at="2026-05-12",
            landing_memo="feedback_fix_j_substrate_compressai_inventory_wire_in_landed_20260512",
        ),
        SubstrateRow(
            substrate_id="vq_vae_substrate",
            name="VQ-VAE per-frame discrete-token substrate (L0 SKETCH)",
            substrate_class=SubstrateClass.RENDERER_REPLACEMENT,
            target_axis=ScoreAxis.MIXED,
            format_id=0x97,
            magic_bytes="VQV1",
            runtime_dep_closure=("torch", "brotli"),
            byte_budget_band=(120_000, 180_000),
            predicted_delta_alone_band=(-0.0070, 0.0060),
            requires_score_aware_training=True,
            landed_at="2026-05-12",
            landing_memo="feedback_fix_j_substrate_compressai_inventory_wire_in_landed_20260512",
        ),
        SubstrateRow(
            substrate_id="siren_substrate",
            name="SIREN sinusoidal coordinate-MLP renderer (L0 SKETCH)",
            substrate_class=SubstrateClass.RENDERER_REPLACEMENT,
            target_axis=ScoreAxis.RATE,
            format_id=0x98,
            magic_bytes="SRV1",
            runtime_dep_closure=("torch", "brotli"),
            byte_budget_band=(80_000, 160_000),
            predicted_delta_alone_band=(-0.0040, 0.0050),
            requires_score_aware_training=True,
            landed_at="2026-05-12",
            landing_memo="feedback_fix_j_substrate_compressai_inventory_wire_in_landed_20260512",
        ),
        SubstrateRow(
            substrate_id="block_nerv_substrate",
            name="Per-pair block-decoder NeRV substrate scaffold (L0 SKETCH)",
            substrate_class=SubstrateClass.RENDERER_REPLACEMENT,
            target_axis=ScoreAxis.MIXED,
            format_id=0x99,
            magic_bytes="BNV1",
            runtime_dep_closure=("torch", "brotli"),
            byte_budget_band=(150_000, 220_000),
            predicted_delta_alone_band=(-0.0060, 0.0080),
            requires_score_aware_training=True,
            landed_at="2026-05-12",
            landing_memo="feedback_fix_j_substrate_compressai_inventory_wire_in_landed_20260512",
        ),
        SubstrateRow(
            substrate_id="tc_nerv_substrate",
            name="Temporal-consistency NeRV substrate scaffold (L0 SKETCH)",
            substrate_class=SubstrateClass.RENDERER_REPLACEMENT,
            target_axis=ScoreAxis.POSE,
            format_id=0x9A,
            magic_bytes="TCV1",
            runtime_dep_closure=("torch", "brotli"),
            byte_budget_band=(120_000, 200_000),
            predicted_delta_alone_band=(-0.0050, 0.0070),
            requires_score_aware_training=True,
            landed_at="2026-05-12",
            landing_memo="feedback_fix_j_substrate_compressai_inventory_wire_in_landed_20260512",
        ),
        SubstrateRow(
            substrate_id="ff_nerv_substrate",
            name="Frequency-domain (DCT) NeRV substrate scaffold (L0 SKETCH)",
            substrate_class=SubstrateClass.RENDERER_REPLACEMENT,
            target_axis=ScoreAxis.RATE,
            format_id=0x9B,
            magic_bytes="FFV1",
            runtime_dep_closure=("torch", "brotli"),
            byte_budget_band=(120_000, 200_000),
            predicted_delta_alone_band=(-0.0050, 0.0060),
            requires_score_aware_training=True,
            landed_at="2026-05-12",
            landing_memo="feedback_fix_j_substrate_compressai_inventory_wire_in_landed_20260512",
        ),
        SubstrateRow(
            substrate_id="ds_nerv_substrate",
            name="Depth-separable NeRV substrate scaffold (L0 SKETCH)",
            substrate_class=SubstrateClass.RENDERER_REPLACEMENT,
            target_axis=ScoreAxis.MIXED,
            format_id=0x9C,
            magic_bytes="DSV1",
            runtime_dep_closure=("torch", "brotli"),
            byte_budget_band=(80_000, 150_000),
            predicted_delta_alone_band=(-0.0040, 0.0050),
            requires_score_aware_training=True,
            landed_at="2026-05-12",
            landing_memo="feedback_fix_j_substrate_compressai_inventory_wire_in_landed_20260512",
        ),
        SubstrateRow(
            substrate_id="hi_nerv_substrate",
            name="Hierarchical NeRV substrate scaffold (L0 SKETCH)",
            substrate_class=SubstrateClass.RENDERER_REPLACEMENT,
            target_axis=ScoreAxis.MIXED,
            format_id=0x9D,
            magic_bytes="HIV1",
            runtime_dep_closure=("torch", "brotli"),
            byte_budget_band=(140_000, 210_000),
            predicted_delta_alone_band=(-0.0060, 0.0070),
            requires_score_aware_training=True,
            landed_at="2026-05-12",
            landing_memo="feedback_fix_j_substrate_compressai_inventory_wire_in_landed_20260512",
        ),
        # ── 9 WAVE-A-2 TRADITION 2 single-file substrate rows ──
        # Per CANON-1.A (`canonicalization_dedup_oss_rigor_ledger_20260512.md`)
        # + tradition taxonomy memo
        # (`.omx/research/substrate_tradition_taxonomy_20260512.md`):
        # the older single-file `<name>_as_renderer.py` / `<name>_renderer.py`
        # substrates pre-date the substrate-scaffold subpackage discipline
        # (Fields-medal 2026-05-12). They are PRODUCTION-MATURE and ship into
        # the canonical inventory so the autopilot / Pareto solver /
        # sensitivity-map see them. Format-id band 0xA0-0xAF reserved for
        # WAVE-A-2 single-file rows; magic-byte band ASCII *-V1 (already
        # used by FIX-J) extended via 4-char distinct tokens.
        SubstrateRow(
            substrate_id="cnerv",
            name="CNeRV (fully-Convolutional NeRV; convolutional stem)",
            substrate_class=SubstrateClass.RENDERER_REPLACEMENT,
            target_axis=ScoreAxis.MIXED,
            format_id=0xA0,
            magic_bytes="CNRV",
            runtime_dep_closure=("torch", "brotli"),
            byte_budget_band=(120_000, 200_000),
            predicted_delta_alone_band=(-0.0050, 0.0150),
            requires_score_aware_training=True,
            landed_at="2026-05-11",
            landing_memo="feedback_wave_a_2_taxonomy_inventory_drift_landed_20260512",
        ),
        SubstrateRow(
            substrate_id="e_nerv",
            name="E-NeRV (Encoder-NeRV; standalone substrate w/ encoder identity)",
            substrate_class=SubstrateClass.RENDERER_REPLACEMENT,
            target_axis=ScoreAxis.MIXED,
            format_id=0xA1,
            magic_bytes="ENRV",
            runtime_dep_closure=("torch", "brotli"),
            byte_budget_band=(120_000, 200_000),
            predicted_delta_alone_band=(-0.0050, 0.0150),
            requires_score_aware_training=True,
            landed_at="2026-05-11",
            landing_memo="feedback_wave_a_2_taxonomy_inventory_drift_landed_20260512",
        ),
        SubstrateRow(
            substrate_id="ego_nerv",
            name="Ego-NeRV (egocentric pose-conditioning NeRV)",
            substrate_class=SubstrateClass.RENDERER_REPLACEMENT,
            target_axis=ScoreAxis.POSE,  # Pose-conditioned by construction.
            format_id=0xA2,
            magic_bytes="EGOV",
            runtime_dep_closure=("torch", "brotli"),
            byte_budget_band=(120_000, 200_000),
            predicted_delta_alone_band=(-0.0060, 0.0120),
            requires_score_aware_training=True,
            landed_at="2026-05-11",
            landing_memo="feedback_wave_a_2_taxonomy_inventory_drift_landed_20260512",
        ),
        SubstrateRow(
            substrate_id="lane_12_v2_nerv_as_renderer",
            name="Lane 12-v2 NeRV-as-renderer (Phase B reference)",
            substrate_class=SubstrateClass.RENDERER_REPLACEMENT,
            target_axis=ScoreAxis.MIXED,
            format_id=0xA3,
            magic_bytes="L12V",
            runtime_dep_closure=("torch", "brotli"),
            byte_budget_band=(150_000, 220_000),
            predicted_delta_alone_band=(-0.0080, 0.0050),
            requires_score_aware_training=True,
            landed_at="2026-05-09",
            landing_memo="feedback_wave_a_2_taxonomy_inventory_drift_landed_20260512",
        ),
        SubstrateRow(
            substrate_id="nervdc",
            name="NeRVdc (NeRV with explicit decoder-conditioning)",
            substrate_class=SubstrateClass.RENDERER_REPLACEMENT,
            target_axis=ScoreAxis.MIXED,
            format_id=0xA4,
            magic_bytes="NDCV",
            runtime_dep_closure=("torch", "brotli"),
            byte_budget_band=(120_000, 200_000),
            predicted_delta_alone_band=(-0.0050, 0.0120),
            requires_score_aware_training=True,
            landed_at="2026-05-11",
            landing_memo="feedback_wave_a_2_taxonomy_inventory_drift_landed_20260512",
        ),
        SubstrateRow(
            substrate_id="quantizr_faithful",
            name="Quantizr-faithful (1:1 PR55 reverse-engineering renderer)",
            substrate_class=SubstrateClass.RENDERER_REPLACEMENT,
            target_axis=ScoreAxis.MIXED,
            format_id=0xA5,
            magic_bytes="QZRV",
            runtime_dep_closure=("torch", "brotli"),
            byte_budget_band=(60_000, 100_000),  # FiLM 88K-param renderer band.
            predicted_delta_alone_band=(-0.0080, 0.0100),
            requires_score_aware_training=True,
            landed_at="2026-05-11",
            landing_memo="feedback_wave_a_2_taxonomy_inventory_drift_landed_20260512",
        ),
        SubstrateRow(
            substrate_id="mlx_mask_renderer",
            name="MLX MaskRenderer (Apple Silicon dev/research)",
            substrate_class=SubstrateClass.RENDERER_REPLACEMENT,
            target_axis=ScoreAxis.MIXED,
            format_id=0xA6,
            magic_bytes="MLXR",
            runtime_dep_closure=("mlx",),  # Apple-only; CUDA dispatch refuses.
            byte_budget_band=(50_000, 120_000),
            # MLX is `[macOS-CPU advisory only]` per CLAUDE.md; no CUDA-axis
            # claim. Predicted band reflects research-signal scope only.
            predicted_delta_alone_band=(-0.0030, 0.0050),
            requires_score_aware_training=False,
            landed_at="2026-05-11",
            landing_memo="feedback_wave_a_2_taxonomy_inventory_drift_landed_20260512",
        ),
        SubstrateRow(
            substrate_id="dp_sims_renderer",
            name="DP-SIMS renderer (semantic image synthesis CVPR 2024)",
            substrate_class=SubstrateClass.RENDERER_REPLACEMENT,
            target_axis=ScoreAxis.SEG,  # Mask-conditioned generation -> seg axis.
            format_id=0xA7,
            magic_bytes="DPSV",
            runtime_dep_closure=("torch", "brotli"),
            byte_budget_band=(180_000, 260_000),
            predicted_delta_alone_band=(-0.0050, 0.0150),
            requires_score_aware_training=True,
            landed_at="2026-05-11",
            landing_memo="feedback_wave_a_2_taxonomy_inventory_drift_landed_20260512",
        ),
        SubstrateRow(
            substrate_id="diffusion_renderer",
            name="Diffusion-based renderer (contrib lane, research_only)",
            substrate_class=SubstrateClass.RENDERER_REPLACEMENT,
            target_axis=ScoreAxis.MIXED,
            format_id=0xA8,
            magic_bytes="DIFV",
            runtime_dep_closure=("torch", "brotli"),
            byte_budget_band=(200_000, 320_000),  # Diffusion weights heavier.
            predicted_delta_alone_band=(-0.0050, 0.0200),
            requires_score_aware_training=True,
            landed_at="2026-05-11",
            landing_memo="feedback_wave_a_2_taxonomy_inventory_drift_landed_20260512",
        ),
    ]
    # Stable sort: by class then id.
    rows.sort(key=lambda r: (r.substrate_class.value, r.substrate_id))
    return rows


# ── Composability rule engine ─────────────────────────────────────────────


def classify_pairwise_composability(
    a: SubstrateRow, b: SubstrateRow
) -> CompositionResult:
    """Apply the council-validated composition rules to ONE pair.

    Per CLAUDE.md "Cross-paradigm composition rules":
    1. Two RENDERER_REPLACEMENT -> REPLACEMENT (mutually exclusive).
    2. Two RESIDUAL on DIFFERENT axes -> ORTHOGONAL.
    3. Two RESIDUAL on SAME axis -> REDUNDANT.
    4. RESIDUAL + RENDERER_REPLACEMENT -> STACKABLE_SERIAL.
    5. SELF_COMPRESSION + anything -> ORTHOGONAL (acts on params).
    6. POSE_AXIS_SIDECHANNEL + spatial substrate -> ORTHOGONAL.
    7. Two POSE_AXIS_SIDECHANNEL on DIFFERENT mechanisms -> STACKABLE_PARALLEL.
    8. Two POSE_AXIS_SIDECHANNEL on SAME mechanism (e.g., RAFT vs LAPose
       both estimating frame-pair pose) -> REDUNDANT.
    9. META_CODEC + anything -> ORTHOGONAL (byte-stream-level).
    10. BOLT_ON + RENDERER_REPLACEMENT/RESIDUAL -> STACKABLE_SERIAL.
    11. BOLT_ON + BOLT_ON -> ORTHOGONAL if axes differ; REDUNDANT otherwise.
    12. Self-with-self (a.substrate_id == b.substrate_id) -> REDUNDANT alpha=0.
    13. SELF_COMPRESSION + SELF_COMPRESSION -> STACKABLE_CASCADE
        (e.g., SC++ Stage 1 -> Hessian-block-FP -> MDL-FP4-TTO).
    14. RESIDUAL + BOLT_ON -> STACKABLE_PARALLEL (residual sits on host
        renderer's RGB; bolt-on modulates the host renderer).
    15. POSE_AXIS_SIDECHANNEL + SELF_COMPRESSION -> ORTHOGONAL.

    The format_id collision check is a hard gate independent of the
    composability classification: if both substrates declare the same
    format_id but different magic, downstream archive grammar would
    silently misroute bytes.
    """
    # Format-ID collision check (independent of composability semantics).
    fid_collision = a.format_id == b.format_id and a.substrate_id != b.substrate_id

    # Self-with-self: redundant by definition (composition with self adds nothing).
    if a.substrate_id == b.substrate_id:
        return CompositionResult(
            substrate_a=a.substrate_id,
            substrate_b=b.substrate_id,
            composability=Composability.REDUNDANT,
            expected_alpha=0.0,
            byte_overhead=0,
            score_axis_target_a=a.target_axis,
            score_axis_target_b=b.target_axis,
            format_id_collision_risk=fid_collision,
            rationale="self-with-self composition is redundant by definition",
        )

    a_class = a.substrate_class
    b_class = b.substrate_class

    # Rule 1: two RENDERER_REPLACEMENT -> REPLACEMENT.
    if a_class == SubstrateClass.RENDERER_REPLACEMENT and b_class == SubstrateClass.RENDERER_REPLACEMENT:
        return CompositionResult(
            substrate_a=a.substrate_id,
            substrate_b=b.substrate_id,
            composability=Composability.REPLACEMENT,
            expected_alpha=0.0,  # Only one renderer slot exists.
            byte_overhead=0,
            score_axis_target_a=a.target_axis,
            score_axis_target_b=b.target_axis,
            format_id_collision_risk=fid_collision,
            rationale=(
                "Both substrates target the renderer slot; archive can hold "
                "only ONE full RGB renderer per HNeRV parity discipline lesson 5"
            ),
        )

    # Rule 5: SELF_COMPRESSION + anything -> ORTHOGONAL (special-case Rule 13 below).
    if a_class == SubstrateClass.SELF_COMPRESSION and b_class == SubstrateClass.SELF_COMPRESSION:
        # Rule 13: SELF_COMPRESSION + SELF_COMPRESSION -> STACKABLE_CASCADE.
        return CompositionResult(
            substrate_a=a.substrate_id,
            substrate_b=b.substrate_id,
            composability=Composability.STACKABLE_CASCADE,
            expected_alpha=0.7,  # Diminishing returns at each stage.
            byte_overhead=0,
            score_axis_target_a=a.target_axis,
            score_axis_target_b=b.target_axis,
            format_id_collision_risk=fid_collision,
            rationale=(
                "Self-compression cascade (e.g., SC++ Stage 1 -> Hessian-block-FP "
                "-> MDL-FP4-TTO); each stage refines the previous"
            ),
        )
    if SubstrateClass.SELF_COMPRESSION in (a_class, b_class):
        return CompositionResult(
            substrate_a=a.substrate_id,
            substrate_b=b.substrate_id,
            composability=Composability.ORTHOGONAL,
            expected_alpha=1.0,
            byte_overhead=0,
            score_axis_target_a=a.target_axis,
            score_axis_target_b=b.target_axis,
            format_id_collision_risk=fid_collision,
            rationale=(
                "Self-compression acts on parameters; substrate acts on "
                "representation. Independent additive savings"
            ),
        )

    # Rule 9: META_CODEC + anything -> ORTHOGONAL.
    if SubstrateClass.META_CODEC in (a_class, b_class):
        return CompositionResult(
            substrate_a=a.substrate_id,
            substrate_b=b.substrate_id,
            composability=Composability.ORTHOGONAL,
            expected_alpha=1.0,
            byte_overhead=64,  # Magic codec wrapper header overhead.
            score_axis_target_a=a.target_axis,
            score_axis_target_b=b.target_axis,
            format_id_collision_risk=fid_collision,
            rationale=(
                "Meta-codec operates at byte-stream level; orthogonal to "
                "any substrate's representation choice"
            ),
        )

    # Rule 4: RESIDUAL + RENDERER_REPLACEMENT -> STACKABLE_SERIAL.
    if {a_class, b_class} == {SubstrateClass.RESIDUAL, SubstrateClass.RENDERER_REPLACEMENT}:
        return CompositionResult(
            substrate_a=a.substrate_id,
            substrate_b=b.substrate_id,
            composability=Composability.STACKABLE_SERIAL,
            expected_alpha=0.85,  # Residual captures most of decoder error.
            byte_overhead=128,  # Section header in archive.
            score_axis_target_a=a.target_axis,
            score_axis_target_b=b.target_axis,
            format_id_collision_risk=fid_collision,
            rationale=(
                "Residual sits on top of replacement renderer's RGB output "
                "as a side-channel correction"
            ),
        )

    # Rule 2/3: two RESIDUAL.
    if a_class == SubstrateClass.RESIDUAL and b_class == SubstrateClass.RESIDUAL:
        if a.target_axis == b.target_axis:
            return CompositionResult(
                substrate_a=a.substrate_id,
                substrate_b=b.substrate_id,
                composability=Composability.REDUNDANT,
                expected_alpha=0.3,  # Both capture similar signal.
                byte_overhead=128,
                score_axis_target_a=a.target_axis,
                score_axis_target_b=b.target_axis,
                format_id_collision_risk=fid_collision,
                rationale=(
                    f"Both residuals target {a.target_axis.value} axis; "
                    "they capture overlapping signal"
                ),
            )
        return CompositionResult(
            substrate_a=a.substrate_id,
            substrate_b=b.substrate_id,
            composability=Composability.ORTHOGONAL,
            expected_alpha=1.0,
            byte_overhead=128,
            score_axis_target_a=a.target_axis,
            score_axis_target_b=b.target_axis,
            format_id_collision_risk=fid_collision,
            rationale=(
                f"Residuals target different axes ({a.target_axis.value} "
                f"vs {b.target_axis.value}); independent additive savings"
            ),
        )

    # Rule 6/7/8: POSE_AXIS_SIDECHANNEL.
    if a_class == SubstrateClass.POSE_AXIS_SIDECHANNEL and b_class == SubstrateClass.POSE_AXIS_SIDECHANNEL:
        # Mechanism check: foveation/RAFT/LAPose all estimate per-pair pose
        # but via different mechanisms; treat as STACKABLE_PARALLEL with mild
        # redundancy (alpha < 1.0).
        return CompositionResult(
            substrate_a=a.substrate_id,
            substrate_b=b.substrate_id,
            composability=Composability.STACKABLE_PARALLEL,
            expected_alpha=0.6,  # Both target pose; partial overlap.
            byte_overhead=128,
            score_axis_target_a=a.target_axis,
            score_axis_target_b=b.target_axis,
            format_id_collision_risk=fid_collision,
            rationale=(
                "Both pose-axis sidechannels target POSE term but via "
                "different mechanisms (foveation = attention; RAFT = optical "
                "flow; LAPose = inverse dynamics); partial redundancy"
            ),
        )
    if SubstrateClass.POSE_AXIS_SIDECHANNEL in (a_class, b_class):
        # Pose-axis + spatial substrate.
        return CompositionResult(
            substrate_a=a.substrate_id,
            substrate_b=b.substrate_id,
            composability=Composability.ORTHOGONAL,
            expected_alpha=1.0,
            byte_overhead=128,
            score_axis_target_a=a.target_axis,
            score_axis_target_b=b.target_axis,
            format_id_collision_risk=fid_collision,
            rationale=(
                "Pose-axis sidechannel composes orthogonally with spatial "
                "substrate; targets pose term directly via per-frame side-channel"
            ),
        )

    # Rule 10/11/14: BOLT_ON.
    if a_class == SubstrateClass.BOLT_ON and b_class == SubstrateClass.BOLT_ON:
        if a.target_axis == b.target_axis:
            return CompositionResult(
                substrate_a=a.substrate_id,
                substrate_b=b.substrate_id,
                composability=Composability.REDUNDANT,
                expected_alpha=0.4,
                byte_overhead=64,
                score_axis_target_a=a.target_axis,
                score_axis_target_b=b.target_axis,
                format_id_collision_risk=fid_collision,
                rationale=f"Both bolt-ons target {a.target_axis.value} axis",
            )
        return CompositionResult(
            substrate_a=a.substrate_id,
            substrate_b=b.substrate_id,
            composability=Composability.ORTHOGONAL,
            expected_alpha=1.0,
            byte_overhead=64,
            score_axis_target_a=a.target_axis,
            score_axis_target_b=b.target_axis,
            format_id_collision_risk=fid_collision,
            rationale="Bolt-ons target different axes; orthogonal",
        )
    if SubstrateClass.BOLT_ON in (a_class, b_class):
        # BOLT_ON + RESIDUAL/RENDERER_REPLACEMENT -> STACKABLE_PARALLEL.
        # (The bolt-on modulates the host renderer; residual sits on
        # host's RGB output. They run in parallel rather than serial.)
        return CompositionResult(
            substrate_a=a.substrate_id,
            substrate_b=b.substrate_id,
            composability=Composability.STACKABLE_PARALLEL,
            expected_alpha=0.85,
            byte_overhead=64,
            score_axis_target_a=a.target_axis,
            score_axis_target_b=b.target_axis,
            format_id_collision_risk=fid_collision,
            rationale=(
                "Bolt-on modulates host renderer (e.g., FiLM pose conditioning "
                "modulates NeRV decoder); composes with residual / replacement "
                "as a parallel parameter modification"
            ),
        )

    # Fallback: unhandled combination. Mark INCOMPATIBLE so callers see the gap.
    return CompositionResult(
        substrate_a=a.substrate_id,
        substrate_b=b.substrate_id,
        composability=Composability.INCOMPATIBLE,
        expected_alpha=0.0,
        byte_overhead=0,
        score_axis_target_a=a.target_axis,
        score_axis_target_b=b.target_axis,
        format_id_collision_risk=fid_collision,
        rationale=(
            f"Unhandled combination ({a_class.value}, {b_class.value}); "
            "extend the rule engine to classify"
        ),
    )


# ── Matrix construction ──────────────────────────────────────────────────


@dataclass(frozen=True)
class CompositionMatrix:
    """Typed matrix bundle.

    The matrix is a dict-of-dicts (substrate_a_id -> substrate_b_id ->
    CompositionResult). The substrate inventory is preserved alongside so
    consumers can iterate Pareto rows without re-fetching.
    """

    schema_version: str
    substrates: tuple[SubstrateRow, ...]
    cells: dict[str, dict[str, CompositionResult]]
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False

    def n_substrates(self) -> int:
        return len(self.substrates)

    def n_cells(self) -> int:
        n = self.n_substrates()
        return n * n

    def get(self, a_id: str, b_id: str) -> CompositionResult:
        """Return the result for one pair; raises KeyError if missing."""
        return self.cells[a_id][b_id]

    def n_format_id_collisions(self) -> int:
        """Count distinct (a, b) pairs (a < b lexicographic) with format-ID collision."""
        seen: set[tuple[str, str]] = set()
        for a_id, by_a in self.cells.items():
            for b_id, cell in by_a.items():
                if a_id == b_id:
                    continue
                key = tuple(sorted((a_id, b_id)))
                if key in seen:
                    continue
                if cell.format_id_collision_risk:
                    seen.add(key)
        return len(seen)

    def find_pairs(self, composability: Composability) -> list[tuple[str, str]]:
        """Return distinct (a_id, b_id) pairs (a < b lexicographic) with the given verdict."""
        out: set[tuple[str, str]] = set()
        for a_id, by_a in self.cells.items():
            for b_id, cell in by_a.items():
                if a_id == b_id:
                    continue
                if cell.composability != composability:
                    continue
                key = (min(a_id, b_id), max(a_id, b_id))
                out.add(key)
        return sorted(out)


def build_composition_matrix(
    substrates: list[SubstrateRow] | None = None,
) -> CompositionMatrix:
    """Build the full pairwise matrix.

    Per CLAUDE.md "Beauty, simplicity, and developer experience": the matrix
    is built once and consumed many times. The construction is O(N^2) but
    pure-numpy-free so it ships in any deployment.
    """
    rows = substrates if substrates is not None else canonical_substrate_inventory()
    if not rows:
        raise ValueError("substrate inventory is empty; refusing to build matrix")

    # Refuse duplicate substrate_ids.
    ids = [r.substrate_id for r in rows]
    if len(set(ids)) != len(ids):
        dupes = [i for i in ids if ids.count(i) > 1]
        raise ValueError(f"duplicate substrate_ids in inventory: {sorted(set(dupes))}")

    cells: dict[str, dict[str, CompositionResult]] = {}
    for a in rows:
        cells[a.substrate_id] = {}
        for b in rows:
            cells[a.substrate_id][b.substrate_id] = classify_pairwise_composability(a, b)
    return CompositionMatrix(
        schema_version=SCHEMA_VERSION,
        substrates=tuple(rows),
        cells=cells,
    )


# ── EV/dollar Pareto rows ─────────────────────────────────────────────────


@dataclass(frozen=True)
class ParetoRow:
    """One per-substrate Pareto candidate row.

    Per CLAUDE.md "Forbidden score claims": every numeric here is
    ``[predicted; substrate composition matrix v1]``. Promotion to a real
    score claim requires a ``[contest-CUDA]`` anchor on the EXACT archive.
    """

    substrate_id: str
    name: str
    substrate_class: SubstrateClass
    target_axis: ScoreAxis
    predicted_delta_alone_midpoint: float
    estimated_dispatch_cost_usd: float
    expected_information_gain: float
    eig_per_dollar: float
    notes: str = ""
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False


# Per-substrate dispatch-cost estimate band (see landing memos for sources).
# Using midpoint-of-published-band as the representative estimate.
DISPATCH_COST_USD_MIDPOINT: dict[str, float] = {
    # Residual basis: small sidecar archives, ~$0.30-1.00 on T4.
    "wavelet_residual": 0.65,
    "cool_chic_residual": 0.65,
    "c3_residual": 0.65,
    "siren_residual": 0.65,
    "coordinate_mlp_residual": 0.65,
    # Pose-axis sidechannels: small archives + per-frame compute.
    "foveation_field": 1.50,
    "raft_pose_stream": 1.50,
    "lapose_motion_atom_allocator": 1.50,
    # Self-compression: full trainer required for SC++; closed-form for Hessian.
    "scpp_substrate": 45.0,  # $30-60 band midpoint.
    "hessian_block_fp": 0.50,  # Allocator runs offline.
    "mdl_fp4_tto": 1.00,  # Refinement on existing weights.
    # NeRV-family renderer-replacements: full trainer ~5-12h on T4.
    "blocknerv": 40.0,
    "ffnerv": 40.0,
    "dsnerv": 40.0,
    "hinerv": 50.0,
    "tcnerv": 40.0,
    # NeRV/MNeRV/VQ-VAE (HH landing): full trainers.
    "nerv_as_renderer": 40.0,
    "mnerv": 50.0,
    "vqvae_as_full_renderer": 45.0,
    # ANR / categorical (II landing): full trainers.
    "anr_token_renderer_v62": 60.0,
    "categorical_substrate": 45.0,
    # Bolt-ons: compose with host substrates; no separate trainer cost.
    "nerv_enc_dec_separated": 0.0,
    "film_pose_conditioning": 0.0,
    # Magic codec: planning-only until runtime is byte-closed/vendored.
    "magic_codec": 0.0,
    # ── FIX-J substrate-scaffold packages (2026-05-12) ──
    # Per LOOPCLOSE wire-in. Banded as full-trainer NeRV-family analogs;
    # research_only scaffolds use lower band midpoints reflecting L0 SKETCH
    # status (council design memo + 13 HNeRV-parity-discipline lessons but
    # no L1+ trainer cost yet).
    "sane_hnerv": 50.0,  # alpha primary; full Sane-HNeRV trainer ~5-12h T4.
    "balle_renderer": 50.0,  # beta parallel; hyperprior trainer ~5-12h T4.
    "hybrid_renderer_residual": 45.0,  # gamma deferred; full trainer + residual basis.
    "self_compress_nn": 30.0,  # delta deferred; MDL clustering full trainer.
    "pr101_lc_v2_clone": 25.0,  # Forensic apples-to-apples replay.
    "cool_chic_full_renderer": 40.0,  # L0 SKETCH; full Cool-Chic AR trainer.
    "wavelet_full_renderer": 40.0,  # L0 SKETCH; Mallat scattering trainer.
    "grayscale_lut": 35.0,  # Selfcomp grayscale-LUT full trainer.
    "vq_vae_substrate": 45.0,  # L0 SKETCH; VQ-VAE codebook training.
    "siren_substrate": 35.0,  # L0 SKETCH; SIREN coord-MLP training.
    "block_nerv_substrate": 45.0,  # L0 SKETCH; per-pair LoRA training.
    "tc_nerv_substrate": 40.0,  # L0 SKETCH; temporal-conv NeRV training.
    "ff_nerv_substrate": 40.0,  # L0 SKETCH; DCT-frequency NeRV training.
    "ds_nerv_substrate": 35.0,  # L0 SKETCH; depth-separable NeRV training.
    "hi_nerv_substrate": 50.0,  # L0 SKETCH; hierarchical NeRV training.
    # ── WAVE-A-2 TRADITION 2 single-file substrate cost band (2026-05-12) ──
    # Per CANON-1.A explicit-taxonomy resolution + tradition memo
    # ``.omx/research/substrate_tradition_taxonomy_20260512.md``. Each row's
    # dispatch cost reflects "full trainer on T4 ~5-12h" for the NeRV-family
    # entries; Quantizr-faithful is bandwidth-light (smaller model);
    # mlx_mask_renderer is `[macOS-CPU advisory only]` so $0 (no paid
    # remote-GPU dispatch path); dp_sims/diffusion are heavier.
    "cnerv": 40.0,
    "e_nerv": 40.0,
    "ego_nerv": 40.0,
    "lane_12_v2_nerv_as_renderer": 45.0,
    "nervdc": 40.0,
    "quantizr_faithful": 25.0,
    "mlx_mask_renderer": 0.0,  # `[macOS-CPU advisory only]`; no paid remote.
    "dp_sims_renderer": 55.0,
    "diffusion_renderer": 70.0,
}


def per_substrate_pareto_rows(
    matrix: CompositionMatrix | None = None,
) -> list[ParetoRow]:
    """Compute per-substrate Pareto rows ranked by EV/$.

    Per CLAUDE.md "Meta-Lagrangian/Pareto solver" + "Bayesian experimental
    design": the EV/$ ranking is the canonical primary signal for the
    autopilot dispatch ranker (Deliverable 2).

    EV is approximated as ``|predicted_delta_alone_midpoint|`` (the absolute
    expected score improvement). Per CLAUDE.md "Forbidden empirical-claim-
    without-evidence-tag", the EV is ``[predicted; substrate matrix v1]``,
    not a measurement.
    """
    matrix = matrix or build_composition_matrix()
    rows: list[ParetoRow] = []
    for s in matrix.substrates:
        cost = DISPATCH_COST_USD_MIDPOINT.get(s.substrate_id, 0.0)
        delta_mid = s.predicted_delta_alone_midpoint()
        # Information gain: |predicted delta|; clamps to 0 for zero-delta.
        eig = abs(delta_mid)
        # Cost-zero is treated as cost-unknown (missing estimation), NOT free.
        # Emit eig_per_dollar=0.0 (sorts LAST per reverse=True ranking) and
        # surface a `cost_estimation_required` blocker so consumers see the
        # underlying gap. Previously emitted float("inf") which (a) violated
        # RFC 8259 when JSON-serialized and (b) falsely promoted cost-unknown
        # rows to the top of the ranking, masking real signal.
        cost_estimation_pending = cost <= 0.0
        eig_per_dollar = 0.0 if cost_estimation_pending else eig / cost
        notes = (
            f"[predicted; substrate composition matrix v1] "
            f"target_axis={s.target_axis.value}, class={s.substrate_class.value}"
        )
        if cost_estimation_pending:
            notes += "; cost_estimation_required (cost=0.0 treated as unknown, not free)"
        rows.append(
            ParetoRow(
                substrate_id=s.substrate_id,
                name=s.name,
                substrate_class=s.substrate_class,
                target_axis=s.target_axis,
                predicted_delta_alone_midpoint=delta_mid,
                estimated_dispatch_cost_usd=cost,
                expected_information_gain=eig,
                eig_per_dollar=eig_per_dollar,
                notes=notes,
            )
        )
    return rows


def rank_substrates_by_ev_per_dollar(
    matrix: CompositionMatrix | None = None,
    *,
    descending: bool = True,
) -> list[ParetoRow]:
    """Return Pareto rows sorted by EV/$ (best-first by default).

    Cost-zero substrates (bolt-ons, allocators) are sorted to the top with
    EV/$ = +inf when they have non-zero EV. Substrates with zero EV go last.
    """
    rows = per_substrate_pareto_rows(matrix=matrix)
    if descending:
        return sorted(rows, key=lambda r: r.eig_per_dollar, reverse=True)
    return sorted(rows, key=lambda r: r.eig_per_dollar)


# ── Composition-aware ranking (consumed by autopilot Deliverable 2) ──────


def filter_pareto_dominated(
    rows: list[ParetoRow],
    matrix: CompositionMatrix | None = None,
) -> list[ParetoRow]:
    """Drop substrates dominated by a redundant sibling already in the list.

    A row is REDUNDANT-DOMINATED if there exists another row in the input
    list with (a) the same target_axis, (b) the same substrate_class, and
    (c) a STRICTLY higher EV/$. The dominated row is dropped so the
    autopilot doesn't double-dispatch substrates that capture overlapping
    signal (e.g., wavelet vs SIREN both targeting RATE-axis residual).

    Per CLAUDE.md "Pareto constraint": this is a Pareto frontier filter on
    the (axis, class) projection — not a hard guarantee that the dropped
    substrate cannot win, but a default ordering for parallel-dispatch.
    """
    matrix = matrix or build_composition_matrix()
    out: list[ParetoRow] = []
    for r in rows:
        dominated = False
        for sibling in rows:
            if sibling.substrate_id == r.substrate_id:
                continue
            if sibling.target_axis != r.target_axis:
                continue
            if sibling.substrate_class != r.substrate_class:
                continue
            if sibling.eig_per_dollar > r.eig_per_dollar:
                # Verify the council's matrix actually flags them as REDUNDANT.
                cell = matrix.get(r.substrate_id, sibling.substrate_id)
                if cell.composability == Composability.REDUNDANT:
                    dominated = True
                    break
        if not dominated:
            out.append(r)
    return out


def predicted_composite_delta(
    substrate_ids: Iterable[str],
    matrix: CompositionMatrix | None = None,
) -> dict[str, Any]:
    """Predict composite score delta for a multi-substrate dispatch.

    Per CLAUDE.md "Cross-paradigm composition rules": the composite delta
    accounts for the per-pair composability alpha. For an N-substrate
    dispatch with predicted deltas d_i and pairwise composability alpha_ij,
    the composite delta is approximated as:

        delta_total ≈ sum_i d_i * (1 - sum_{j != i} (1 - alpha_ij) / (N-1))

    This is a first-order Volterra-style correction; super-additive cases
    (alpha > 1) or antagonistic cases (alpha < 0) are flagged in the
    returned dict for downstream consumer attention.

    Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag", the
    output carries ``score_claim=False`` and is tagged ``[predicted]``.
    """
    matrix = matrix or build_composition_matrix()
    ids = list(substrate_ids)
    if not ids:
        return {
            "schema": SCHEMA_VERSION,
            "substrate_ids": [],
            "predicted_composite_delta": 0.0,
            "evidence_grade": "planning_only_no_substrates_provided",
            "score_claim": False,
            "warnings": [],
        }
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate substrate_ids in dispatch: {ids}")
    by_id = {s.substrate_id: s for s in matrix.substrates}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise ValueError(
            f"substrate_ids {missing!r} not in canonical inventory; "
            "extend canonical_substrate_inventory() first"
        )

    n = len(ids)
    warnings: list[str] = []
    composite = 0.0

    # Refuse REPLACEMENT pairs (mutually exclusive at archive level).
    for i in range(n):
        for j in range(i + 1, n):
            cell = matrix.get(ids[i], ids[j])
            if cell.composability == Composability.REPLACEMENT:
                raise ValueError(
                    f"substrates ({ids[i]}, {ids[j]}) are mutually exclusive "
                    f"({cell.rationale}); cannot dispatch together"
                )
            if cell.composability == Composability.INCOMPATIBLE:
                raise ValueError(
                    f"substrates ({ids[i]}, {ids[j]}) marked INCOMPATIBLE "
                    f"({cell.rationale}); cannot dispatch together"
                )
            if cell.format_id_collision_risk:
                warnings.append(
                    f"format-ID collision between {ids[i]!r} and {ids[j]!r}; "
                    "archive grammar rejection at parse time"
                )
            if cell.composability == Composability.ANTAGONISTIC:
                warnings.append(
                    f"({ids[i]}, {ids[j]}) flagged ANTAGONISTIC "
                    f"(alpha={cell.expected_alpha:.2f})"
                )

    # First-order Volterra-style composite delta.
    if n == 1:
        s = by_id[ids[0]]
        composite = s.predicted_delta_alone_midpoint()
    else:
        for i, sid_i in enumerate(ids):
            d_i = by_id[sid_i].predicted_delta_alone_midpoint()
            penalty_sum = 0.0
            for j, sid_j in enumerate(ids):
                if i == j:
                    continue
                alpha_ij = matrix.get(sid_i, sid_j).expected_alpha
                penalty_sum += (1.0 - alpha_ij)
            penalty = penalty_sum / max(n - 1, 1)
            composite += d_i * (1.0 - penalty)

    return {
        "schema": SCHEMA_VERSION,
        "substrate_ids": ids,
        "predicted_composite_delta": composite,
        "evidence_grade": "predicted_substrate_composition_matrix_v1",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "warnings": warnings,
        "n_substrates": n,
    }


# ── Serialization ────────────────────────────────────────────────────────


def _composition_to_dict(c: CompositionResult) -> dict[str, Any]:
    d = dataclasses.asdict(c)
    d["composability"] = c.composability.value
    d["score_axis_target_a"] = c.score_axis_target_a.value
    d["score_axis_target_b"] = c.score_axis_target_b.value
    return d


def _row_to_dict(r: SubstrateRow) -> dict[str, Any]:
    d = dataclasses.asdict(r)
    d["substrate_class"] = r.substrate_class.value
    d["target_axis"] = r.target_axis.value
    d["runtime_dep_closure"] = list(r.runtime_dep_closure)
    d["byte_budget_band"] = list(r.byte_budget_band)
    d["predicted_delta_alone_band"] = list(r.predicted_delta_alone_band)
    return d


def serialize_matrix(matrix: CompositionMatrix) -> dict[str, Any]:
    """JSON-safe serialization of the matrix for downstream consumers."""
    cells_dict = {
        a_id: {b_id: _composition_to_dict(c) for b_id, c in by_a.items()}
        for a_id, by_a in matrix.cells.items()
    }
    return {
        "schema": matrix.schema_version,
        "n_substrates": matrix.n_substrates(),
        "n_cells": matrix.n_cells(),
        "n_format_id_collisions": matrix.n_format_id_collisions(),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": "planning_only_substrate_composition_matrix_v1",
        "substrates": [_row_to_dict(s) for s in matrix.substrates],
        "cells": cells_dict,
        "claude_md_compliance_tags": [
            "planning_only_no_score_claim",
            "no_mps_authoritative",
            "no_tmp_paths",
            "substrate_composition_orthogonality_matrix_v1",
        ],
    }


def serialize_pareto_rows(rows: list[ParetoRow]) -> list[dict[str, Any]]:
    """JSON-safe serialization of Pareto rows."""
    out = []
    for r in rows:
        d = dataclasses.asdict(r)
        d["substrate_class"] = r.substrate_class.value
        d["target_axis"] = r.target_axis.value
        out.append(d)
    return out


def write_matrix_json(matrix: CompositionMatrix, path: str) -> None:
    """Write the matrix as pretty-printed JSON.

    Per CLAUDE.md "Forbidden /tmp paths": callers must point ``path`` at a
    durable location (``experiments/results/`` / ``reports/`` / ``.omx/``).
    The caller is responsible for parent-dir creation.
    """
    if path.startswith("/tmp/") or "/private/tmp/" in path or "/var/tmp/" in path:
        raise ValueError(f"refusing to write to forbidden /tmp path: {path!r}")
    payload = serialize_matrix(matrix)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


__all__ = [
    "DISPATCH_COST_USD_MIDPOINT",
    "SCHEMA_VERSION",
    "Composability",
    "CompositionMatrix",
    "CompositionResult",
    "ParetoRow",
    "ScoreAxis",
    "SubstrateClass",
    "SubstrateRow",
    "build_composition_matrix",
    "canonical_substrate_inventory",
    "classify_pairwise_composability",
    "filter_pareto_dominated",
    "per_substrate_pareto_rows",
    "predicted_composite_delta",
    "rank_substrates_by_ev_per_dollar",
    "serialize_matrix",
    "serialize_pareto_rows",
    "write_matrix_json",
]
