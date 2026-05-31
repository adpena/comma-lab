# SPDX-License-Identifier: MIT
"""HPRC family-binding registry.

This file is the anti-local-minimum map for the new rate-first paradigm. It
classifies existing substrate families by what they are allowed to contribute
to HPRC, so PR95/HNeRV, RNeRV/PACT, C3/Cool-Chic, SIREN/COIN, RAFT, CLade/
SPADE, and Z8 teacher work compose through one archive contract instead of
becoming disconnected bolt-ons.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from tac.substrates.hprc.archive import HprcSectionKind


class HprcRole(StrEnum):
    """Role a research family may play in the HPRC stack."""

    BASE_RECEIVER = "base_receiver"
    LATENT_STREAM = "latent_stream"
    RESIDUAL_TOKENIZER = "residual_tokenizer"
    MOTION_SIDE_INFO = "motion_side_info"
    SEMANTIC_CONDITIONER = "semantic_conditioner"
    ENTROPY_MODEL = "entropy_model"
    SCORER_ALLOCATOR = "scorer_allocator"
    TEACHER_ONLY = "teacher_only"
    BASELINE_ORACLE = "baseline_oracle"


class HprcOptimizationLever(StrEnum):
    """Optimization levers HPRC is allowed to test under byte accounting."""

    RECEIVER_WEIGHT_QUANTIZATION = "receiver_weight_quantization"
    RECEIVER_WEIGHT_PRUNING = "receiver_weight_pruning"
    LOW_RANK_ADAPTERS = "low_rank_adapters"
    LATENT_ENTROPY_CODING = "latent_entropy_coding"
    LATENT_PREDICTION = "latent_prediction"
    VECTOR_QUANTIZATION = "vector_quantization"
    SHARED_CODEBOOKS = "shared_codebooks"
    RESIDUAL_TOKEN_WATERFILL = "residual_token_waterfill"
    BITPLANE_TRUNCATION = "bitplane_truncation"
    SIGNIFICANCE_TREE_CODING = "significance_tree_coding"
    MOTION_COMPENSATED_SIDE_INFO = "motion_compensated_side_info"
    SEMANTIC_CONDITIONING = "semantic_conditioning"
    SCORER_WEIGHTED_ABLATION = "scorer_weighted_ablation"
    SEGNET_BOUNDARY_REPAIR = "segnet_boundary_repair"
    POSENET_NULL_ALLOCATION = "posenet_null_allocation"
    QAT_LSQ_NOISE_SHAPING = "qat_lsq_noise_shaping"
    MUON_ADAMW_CURRICULUM = "muon_adamw_curriculum"
    RANGE_ANS_ARITHMETIC_CODING = "range_ans_arithmetic_coding"
    BROTLI_REPACK_ORDERING = "brotli_repack_ordering"
    PACKETIR_SECTION_COMPILATION = "packetir_section_compilation"
    NATIVE_RUST_ZIG_DECODE = "native_rust_zig_decode"
    FULL_VIDEO_BUNDLE_KKT_ALLOCATION = "full_video_bundle_kkt_allocation"
    EXACT_REPLAY_ACCEPTANCE = "exact_replay_acceptance"
    INVENTED_RECEIVER_PARADIGM = "invented_receiver_paradigm"
    GENERATED_OPERATOR_SEARCH = "generated_operator_search"
    CROSS_FAMILY_STACK_SYNTHESIS = "cross_family_stack_synthesis"
    UNKNOWN_FUTURE_LEVER = "unknown_future_lever"


HPRC_OPTIMIZATION_LEVERS: tuple[dict[str, object], ...] = (
    {
        "lever": HprcOptimizationLever.RECEIVER_WEIGHT_QUANTIZATION.value,
        "stage": "receiver",
        "sections": [HprcSectionKind.DECODER_QW.name],
        "gate": "full-frame parity or measured scorer delta; bytes include scales/tables",
    },
    {
        "lever": HprcOptimizationLever.RECEIVER_WEIGHT_PRUNING.value,
        "stage": "receiver",
        "sections": [HprcSectionKind.DECODER_QW.name],
        "gate": "scorer-weighted ablation with exact archive replay",
    },
    {
        "lever": HprcOptimizationLever.LOW_RANK_ADAPTERS.value,
        "stage": "receiver",
        "sections": [HprcSectionKind.DECODER_QW.name, HprcSectionKind.LATENTS_RC.name],
        "gate": "adapter bytes beat full-weight delta at equal local replay",
    },
    {
        "lever": HprcOptimizationLever.LATENT_ENTROPY_CODING.value,
        "stage": "latent_stream",
        "sections": [HprcSectionKind.LATENTS_RC.name],
        "gate": "bijective decode and byte-positive versus raw latent stream",
    },
    {
        "lever": HprcOptimizationLever.LATENT_PREDICTION.value,
        "stage": "latent_stream",
        "sections": [HprcSectionKind.LATENTS_RC.name, HprcSectionKind.RECEIVER_STATE.name],
        "gate": "predictor plus residual bytes beat direct latents",
    },
    {
        "lever": HprcOptimizationLever.VECTOR_QUANTIZATION.value,
        "stage": "residual_or_latent",
        "sections": [HprcSectionKind.CODEBOOKS_Q.name, HprcSectionKind.RESIDUAL_RC.name],
        "gate": "codebook plus indices beats scalar residual coding",
    },
    {
        "lever": HprcOptimizationLever.SHARED_CODEBOOKS.value,
        "stage": "residual_or_latent",
        "sections": [HprcSectionKind.CODEBOOKS_Q.name, HprcSectionKind.SELECTORS_RC.name],
        "gate": "shared dictionaries amortize across pairs/GOPs",
    },
    {
        "lever": HprcOptimizationLever.RESIDUAL_TOKEN_WATERFILL.value,
        "stage": "residual_sidecar",
        "sections": [HprcSectionKind.SELECTORS_RC.name, HprcSectionKind.RESIDUAL_RC.name],
        "gate": "P18/P19 marginal value beats measured byte cost",
    },
    {
        "lever": HprcOptimizationLever.BITPLANE_TRUNCATION.value,
        "stage": "residual_sidecar",
        "sections": [HprcSectionKind.RESIDUAL_RC.name, HprcSectionKind.RDO_PLAN.name],
        "gate": "EBCOT-style pass truncation wins after base receiver collapse",
    },
    {
        "lever": HprcOptimizationLever.SIGNIFICANCE_TREE_CODING.value,
        "stage": "residual_sidecar",
        "sections": [HprcSectionKind.RESIDUAL_RC.name],
        "gate": "tree metadata stays cheaper than block/bitplane stream",
    },
    {
        "lever": HprcOptimizationLever.MOTION_COMPENSATED_SIDE_INFO.value,
        "stage": "receiver_prediction",
        "sections": [HprcSectionKind.RECEIVER_STATE.name, HprcSectionKind.SELECTORS_RC.name],
        "gate": "motion state is procedural or compact; no dense flow field storage",
    },
    {
        "lever": HprcOptimizationLever.SEMANTIC_CONDITIONING.value,
        "stage": "receiver_prediction",
        "sections": [HprcSectionKind.SELECTORS_RC.name, HprcSectionKind.RDO_PLAN.name],
        "gate": "semantic state is derived or compactly charged; no hidden scorer outputs",
    },
    {
        "lever": HprcOptimizationLever.SCORER_WEIGHTED_ABLATION.value,
        "stage": "allocator",
        "sections": [HprcSectionKind.RDO_PLAN.name],
        "gate": "full-video scorer surfaces, not RGB-only pruning",
    },
    {
        "lever": HprcOptimizationLever.SEGNET_BOUNDARY_REPAIR.value,
        "stage": "allocator",
        "sections": [HprcSectionKind.SELECTORS_RC.name, HprcSectionKind.RESIDUAL_RC.name],
        "gate": "valid receiver patch reduces total score after byte cost",
    },
    {
        "lever": HprcOptimizationLever.POSENET_NULL_ALLOCATION.value,
        "stage": "allocator",
        "sections": [HprcSectionKind.RDO_PLAN.name],
        "gate": "Mahalanobis pose-null budget is rechecked after replay",
    },
    {
        "lever": HprcOptimizationLever.QAT_LSQ_NOISE_SHAPING.value,
        "stage": "training",
        "sections": [HprcSectionKind.DECODER_QW.name, HprcSectionKind.LATENTS_RC.name],
        "gate": "quantization noise in training improves hard archive replay",
    },
    {
        "lever": HprcOptimizationLever.MUON_ADAMW_CURRICULUM.value,
        "stage": "training",
        "sections": [HprcSectionKind.DECODER_QW.name, HprcSectionKind.LATENTS_RC.name],
        "gate": "timed MLX stage smokes before long run",
    },
    {
        "lever": HprcOptimizationLever.RANGE_ANS_ARITHMETIC_CODING.value,
        "stage": "entropy",
        "sections": [
            HprcSectionKind.LATENTS_RC.name,
            HprcSectionKind.SELECTORS_RC.name,
            HprcSectionKind.RESIDUAL_RC.name,
        ],
        "gate": "native/vectorized decoder if Python entropy decode threatens auth window",
    },
    {
        "lever": HprcOptimizationLever.BROTLI_REPACK_ORDERING.value,
        "stage": "container",
        "sections": [section.name for section in HprcSectionKind],
        "gate": "after pre-entropy shaping; never expected to fix random mantissas",
    },
    {
        "lever": HprcOptimizationLever.PACKETIR_SECTION_COMPILATION.value,
        "stage": "compiler",
        "sections": [section.name for section in HprcSectionKind],
        "gate": "section manifests and receiver proof stay canonical",
    },
    {
        "lever": HprcOptimizationLever.NATIVE_RUST_ZIG_DECODE.value,
        "stage": "runtime",
        "sections": [HprcSectionKind.RESIDUAL_RC.name, HprcSectionKind.LATENTS_RC.name],
        "gate": "byte-identical output and real inflate speedup",
    },
    {
        "lever": HprcOptimizationLever.FULL_VIDEO_BUNDLE_KKT_ALLOCATION.value,
        "stage": "allocator",
        "sections": [HprcSectionKind.RDO_PLAN.name],
        "gate": "full-video exact chunk reduction before update",
    },
    {
        "lever": HprcOptimizationLever.EXACT_REPLAY_ACCEPTANCE.value,
        "stage": "promotion",
        "sections": [section.name for section in HprcSectionKind],
        "gate": "local CPU replay then exact auth only for true local winners",
    },
    {
        "lever": HprcOptimizationLever.INVENTED_RECEIVER_PARADIGM.value,
        "stage": "architecture_search",
        "sections": [
            HprcSectionKind.DECODER_QW.name,
            HprcSectionKind.LATENTS_RC.name,
            HprcSectionKind.RECEIVER_STATE.name,
        ],
        "gate": "new receiver class must beat PR95 control on byte-counted train/export/archive proof",
    },
    {
        "lever": HprcOptimizationLever.GENERATED_OPERATOR_SEARCH.value,
        "stage": "architecture_search",
        "sections": [
            HprcSectionKind.SELECTORS_RC.name,
            HprcSectionKind.RESIDUAL_RC.name,
            HprcSectionKind.RDO_PLAN.name,
        ],
        "gate": "generated operators must compile to deterministic receiver code and section bytes",
    },
    {
        "lever": HprcOptimizationLever.CROSS_FAMILY_STACK_SYNTHESIS.value,
        "stage": "architecture_search",
        "sections": [section.name for section in HprcSectionKind],
        "gate": "stack member contributions are position-disjoint or measured for interaction",
    },
    {
        "lever": HprcOptimizationLever.UNKNOWN_FUTURE_LEVER.value,
        "stage": "reserved",
        "sections": [section.name for section in HprcSectionKind],
        "gate": "register role, byte budget, receiver proof, and replay gate before implementation",
    },
)


@dataclass(frozen=True)
class HprcFamilyBinding:
    """A single research/code family mapped into HPRC's packet sections."""

    family_id: str
    roles: tuple[HprcRole, ...]
    allowed_sections: tuple[HprcSectionKind, ...]
    existing_surfaces: tuple[str, ...]
    promotion_gate: str
    rate_axis_risk: str
    contest_fit: str
    notes: str

    def as_dict(self) -> dict[str, object]:
        return {
            "family_id": self.family_id,
            "roles": [role.value for role in self.roles],
            "allowed_sections": [section.name for section in self.allowed_sections],
            "existing_surfaces": list(self.existing_surfaces),
            "promotion_gate": self.promotion_gate,
            "rate_axis_risk": self.rate_axis_risk,
            "contest_fit": self.contest_fit,
            "notes": self.notes,
        }


HPRC_FAMILY_BINDINGS: tuple[HprcFamilyBinding, ...] = (
    HprcFamilyBinding(
        family_id="pr95_hnerv_control",
        roles=(HprcRole.BASE_RECEIVER, HprcRole.LATENT_STREAM, HprcRole.BASELINE_ORACLE),
        allowed_sections=(
            HprcSectionKind.DECODER_QW,
            HprcSectionKind.LATENTS_RC,
            HprcSectionKind.MANIFEST_JSON,
        ),
        existing_surfaces=(
            "src/tac/substrates/pr95_lora_dora",
            "tools/materialize_hnerv_generated_schema_codec.py",
            "experiments/results/public_pr_intake_full/public_pr95_intake_20260505_auto/archive.zip",
        ),
        promotion_gate=(
            "full 600-pair train/export/archive proof, full-frame inflate parity, "
            "then local CPU replay before exact auth"
        ),
        rate_axis_risk="low bytes proven by public PR95; innovation ceiling limited if copied directly",
        contest_fit="control arm and warm-start receiver, not the final new paradigm alone",
        notes="Compact decoder plus latents is the byte-scale target HPRC must match or beat.",
    ),
    HprcFamilyBinding(
        family_id="rnerv_pact_nerv_base",
        roles=(HprcRole.BASE_RECEIVER, HprcRole.LATENT_STREAM),
        allowed_sections=(
            HprcSectionKind.DECODER_QW,
            HprcSectionKind.LATENTS_RC,
            HprcSectionKind.MANIFEST_JSON,
        ),
        existing_surfaces=(
            "src/tac/substrates/pact_nerv_ia3",
            "src/tac/substrates/pact_nerv_selector_v4",
            ".omx/research/rnerv_pact_z8_residual_lane_design_20260531T222520Z_codex.md",
        ),
        promotion_gate=(
            "MLX train smoke with bytes-in-archive export; no hidden hypernetwork or "
            "checkpoint state; measured archive bytes beat PR95-scale baseline at useful distortion"
        ),
        rate_axis_risk="medium; can regress into larger NeRV variants without hard byte pressure",
        contest_fit="primary rate-collapse candidate for HPRC V1",
        notes="Use RNeRV/PACT ideas to improve receiver efficiency, but optimize contest action not PSNR.",
    ),
    HprcFamilyBinding(
        family_id="z8_hpc_teacher_residual",
        roles=(
            HprcRole.TEACHER_ONLY,
            HprcRole.RESIDUAL_TOKENIZER,
            HprcRole.SCORER_ALLOCATOR,
        ),
        allowed_sections=(
            HprcSectionKind.SELECTORS_RC,
            HprcSectionKind.RESIDUAL_RC,
            HprcSectionKind.RDO_PLAN,
            HprcSectionKind.MANIFEST_JSON,
        ),
        existing_surfaces=(
            "src/tac/substrates/z8_hierarchical_predictive_coding",
            "tools/profile_z8_hpc_archive_bytes.py",
            "tools/z8_top_ll_entropy_headroom_report.py",
            ".omx/research/codex_findings_z8_wavelet_quantization_research_20260531T221839Z_subagent.md",
        ),
        promotion_gate=(
            "residual sidecar bytes only; explicit top-LL/detail field store cannot be "
            "the primary archive representation"
        ),
        rate_axis_risk="very high if raw/quantized per-pair wavelet fields remain primary payload",
        contest_fit="teacher, residual codec, and P18/P19 allocator surface",
        notes="The profiler proved explicit Z8 payloads are tens of MB; HPRC uses them to train what not to store.",
    ),
    HprcFamilyBinding(
        family_id="c3_cool_chic_overfit_codec",
        roles=(HprcRole.BASE_RECEIVER, HprcRole.ENTROPY_MODEL, HprcRole.LATENT_STREAM),
        allowed_sections=(
            HprcSectionKind.DECODER_QW,
            HprcSectionKind.LATENTS_RC,
            HprcSectionKind.CODEBOOKS_Q,
            HprcSectionKind.MANIFEST_JSON,
        ),
        existing_surfaces=(
            "src/tac/substrates/cool_chic",
            "src/tac/substrates/nscs03_end_to_end_balle_joint_codec",
        ),
        promotion_gate=(
            "decoder complexity fits inflate budget; entropy tables are archive-bound; "
            "local replay beats PR95/HPRC control at equal bytes"
        ),
        rate_axis_risk="medium; model overhead can dominate at the contest's tiny byte scale",
        contest_fit="strong inspiration for overfitted neural codec with exact entropy-coded latents",
        notes="Useful for learned latent priors and low-complexity decoder discipline.",
    ),
    HprcFamilyBinding(
        family_id="siren_coin_coordinate_basis",
        roles=(HprcRole.BASE_RECEIVER, HprcRole.RESIDUAL_TOKENIZER),
        allowed_sections=(
            HprcSectionKind.DECODER_QW,
            HprcSectionKind.LATENTS_RC,
            HprcSectionKind.RESIDUAL_RC,
            HprcSectionKind.MANIFEST_JSON,
        ),
        existing_surfaces=(
            "src/tac/substrates/siren",
            "src/tac/substrates/coin_plus_plus",
            "src/tac/substrates/coin_pp_implicit_neural_representation",
        ),
        promotion_gate="must beat image-wise receiver at bytes-per-pixel after runtime/code cost",
        rate_axis_risk="high for full-frame coordinate MLPs; useful if constrained to residual patches/atoms",
        contest_fit="patch/residual implicit basis, not default full-video carrier",
        notes="Coordinate methods can model hard residual regions when selected by P18/P19, but are not free.",
    ),
    HprcFamilyBinding(
        family_id="raft_motion_side_information",
        roles=(HprcRole.MOTION_SIDE_INFO, HprcRole.RESIDUAL_TOKENIZER),
        allowed_sections=(
            HprcSectionKind.SELECTORS_RC,
            HprcSectionKind.RESIDUAL_RC,
            HprcSectionKind.RECEIVER_STATE,
            HprcSectionKind.MANIFEST_JSON,
        ),
        existing_surfaces=(
            "src/tac/substrates/d4_wyner_ziv_frame_0",
            "src/tac/substrates/pretrained_driving_prior",
        ),
        promotion_gate=(
            "motion state is regenerated or archive-bound; flow bytes plus residual bytes beat "
            "direct residual coding"
        ),
        rate_axis_risk="medium-high; dense flow fields are too large unless proceduralized or heavily quantized",
        contest_fit="side-information generator for frame1/residual prediction",
        notes="Use RAFT-like flow as encoder teacher or compact motion prior; do not serialize dense flow.",
    ),
    HprcFamilyBinding(
        family_id="clade_spade_semantic_conditioning",
        roles=(HprcRole.SEMANTIC_CONDITIONER, HprcRole.SCORER_ALLOCATOR),
        allowed_sections=(
            HprcSectionKind.SELECTORS_RC,
            HprcSectionKind.RDO_PLAN,
            HprcSectionKind.MANIFEST_JSON,
        ),
        existing_surfaces=(
            "src/tac/substrates/d1_segnet_margin_polytope",
            "src/tac/substrates/cascade_c_prime_frame_1_segnet_waterfill",
            "src/tac/substrates/uniward_per_pixel_distortion",
        ),
        promotion_gate=(
            "semantic maps/classes are derived or compactly charged; no hidden SegNet outputs "
            "inside the receiver"
        ),
        rate_axis_risk="high if class maps are stored; useful as allocator/conditioning prior",
        contest_fit="scorer-region bit allocation and decoder conditioning",
        notes="CLade/SPADE ideas matter as cheap conditioning if the semantics are implicit or tiny.",
    ),
    HprcFamilyBinding(
        family_id="ebcot_spiht_wavelet_residual_coder",
        roles=(HprcRole.RESIDUAL_TOKENIZER, HprcRole.ENTROPY_MODEL),
        allowed_sections=(
            HprcSectionKind.CODEBOOKS_Q,
            HprcSectionKind.SELECTORS_RC,
            HprcSectionKind.RESIDUAL_RC,
            HprcSectionKind.RDO_PLAN,
            HprcSectionKind.MANIFEST_JSON,
        ),
        existing_surfaces=(
            "src/tac/substrates/z8_hierarchical_predictive_coding/per_subband_rd_waterfill_solver.py",
            "src/tac/substrates/z8_hierarchical_predictive_coding/joint_coefficient_waterfill.py",
        ),
        promotion_gate="block/pass truncation emits a receiver-proven residual stream and beats current portfolio",
        rate_axis_risk="medium; second-order unless paired with base receiver collapse",
        contest_fit="best classical residual sidecar path once base prediction is compact",
        notes="Use RD pass truncation on residuals, not full explicit frame fields.",
    ),
)


def get_binding(family_id: str) -> HprcFamilyBinding:
    """Return a binding by ID."""

    for binding in HPRC_FAMILY_BINDINGS:
        if binding.family_id == family_id:
            return binding
    raise KeyError(f"unknown HPRC family binding {family_id!r}")


def hprc_campaign_manifest() -> dict[str, object]:
    """Return the current HPRC campaign manifest.

    The manifest is intentionally score-authority false; it is a planner input
    and audit surface. Exact candidate promotion must flow through archive
    packaging, receiver proof, local replay, and contest-axis auth gates.
    """

    return {
        "schema": "hprc_family_binding_manifest.v1",
        "primary_lane": "hprc_hierarchical_predictive_receiver_codec",
        "score_claim": False,
        "promotion_eligible": False,
        "rate_first_rules": [
            "primary payload must be compact decoder plus latent/code streams, not explicit per-pair fields",
            "Z8 wavelets are teacher/residual/allocator surfaces unless a residual sidecar is byte-closed",
            "q=0.25 and other probe payload numbers are advisory until packed into archive.zip with receiver proof",
            "byte ceilings include hprc.bin, inflate runtime, decoder code, config, entropy tables, and ZIP/container bytes",
            "authority manifests use full SHA-256 digests, not digest prefixes",
            "section mutation proof requires valid semantic mutation plus full receiver replay, not just raw byte flips",
            "motion/semantic maps are derived or charged; hidden eval-time state is refused",
            "MLX/local rows are acquisition signals only until local CPU and exact auth gates pass",
            "new invented levers are welcome only after they register role, section mapping, byte budget, proof gate, and replay gate",
        ],
        "bindings": [binding.as_dict() for binding in HPRC_FAMILY_BINDINGS],
        "optimization_levers": list(HPRC_OPTIMIZATION_LEVERS),
    }


def primary_rate_collapse_candidates() -> tuple[str, ...]:
    """Families allowed to compete as the primary compact receiver."""

    candidates: list[str] = []
    for binding in HPRC_FAMILY_BINDINGS:
        if HprcRole.BASE_RECEIVER in binding.roles and HprcRole.TEACHER_ONLY not in binding.roles:
            candidates.append(binding.family_id)
    return tuple(candidates)


def residual_sidecar_candidates() -> tuple[str, ...]:
    """Families allowed to emit residual token streams."""

    return tuple(
        binding.family_id
        for binding in HPRC_FAMILY_BINDINGS
        if HprcRole.RESIDUAL_TOKENIZER in binding.roles
    )
