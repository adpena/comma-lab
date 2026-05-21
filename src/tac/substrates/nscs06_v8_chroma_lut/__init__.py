# SPDX-License-Identifier: MIT
"""tac.substrates.nscs06_v8_chroma_lut — chroma-LUT replacement variant (L0 SCAFFOLD).

Per WAVE-3-NSCS06-V8-CHROMA-LUT-SUBSTRATE-BUILD 2026-05-21 + CASCADE
COMPRESSION symposium commit ``d125af6c3`` PRIORITY 3 (Daubechies + Mallat
multi-scale partition discovery framing; 2nd IN-DOMAIN procedural-variant
substrate after grayscale_lut) + HONEST CASCADE-MORTALITY ASSESSMENT
commit ``d884dd6aa`` Rank 2 + NSCS06 v6 -> v7 cargo-cult-unwind methodology
empirically validated rescue path commit ``4292c8ce2``.

**Canonical equation #26 IN-DOMAIN context**: ``nscs06_v8_chroma_lut`` (per
``src/tac/canonical_equations/procedural_codebook_savings.py:102``
``_INCLUDED_CONTEXTS``). Predicted savings
``ΔS = -25 * (4096 - 32) / 37_545_489 ≈ -0.002706`` [prediction;
canonical-equation-26-grounded; per-substrate-symposium-pending].

The v8 chroma-LUT substrate scales the v7 per-class chroma palette from
~15 bytes (5 classes x 3 RGB anchors) to a ~4096-byte
**per-grayscale-level x per-class** chroma table. The expansion has two
parallel realizations:

- **v1 INLINE LUT**: the full 4096-byte dense LUT lives inside the archive
  at fixed offset (``CH08_SCHEMA_VERSION_INLINE_LUT``).
- **v2 PROCEDURAL SEED**: the LUT slot is replaced by a 32-byte PCG64 seed
  re-derived deterministically at inflate via
  ``tac.procedural_codebook_generator.derive_codebook_from_seed`` per sister
  PROCEDURAL VARIANT pattern (grayscale_lut commit ``f037d1144`` +
  DP1 commit ``9cbfa471c`` + VQ-VAE commit ``6fea30f22``).

This is the canonical NSCS06 v8 substrate per the operator's "fix the third
slot approved" directive + CASCADE COMPRESSION symposium PRIORITY 3 +
HONEST CASCADE-MORTALITY ASSESSMENT Rank 2.

13 HNeRV parity-discipline lessons compliance — design-time declaration:

| Lesson | Status | Notes |
|---|---|---|
| L1 substrate must be score-aware | PASS | scorer queried at COMPRESS time; class-conditional LUT derivation |
| L2 export-first archive grammar | PASS | CH08 grammar declared BEFORE training (this package) |
| L3 monolithic 0.bin | PASS | single-file fixed-offset CH08 grammar |
| L4 inflate <= 100 LOC | substrate_engineering exception | ~120 LOC w/ chroma LUT lookup |
| L5 full RGB renderer | PASS | per-pair RGB via per-pixel (level, class) lookup |
| L6 score-domain Lagrangian | N/A | NO training; bit allocation closed-form (sister to v7) |
| L7 bolt-on <= 350 LOC | substrate_engineering exception | total ~700 LOC across 4 files |
| L8 eval-roundtrip + diff yuv6 | N/A | NO training; simulated at compress only |
| L9 runtime closure | PASS | numpy + Pillow + tac.procedural_codebook_generator (transitive numpy) |
| L10 mask/pose coupling | PASS | pose deltas drive frame-1 affine warp from frame-0 |
| L11 no-op detector | PASS | Catalog #139 byte-mutation smoke planned + scaffolded |
| L12 single-LOC review discipline | PASS | each file reviewable in 30s |
| L13 KILL last resort | PASS | DEFERRED-pending-per-substrate-symposium per Catalog #325 |

Catalog #124 archive-grammar 8 fields:

    archive_grammar:            monolithic single-file 0.bin CH08 fixed offsets
    parser_section_manifest:    parse_archive() -> Nscs06V8Archive (header / lut_payload / pose / grayscale)
    inflate_runtime_loc_budget: ~120 LOC (numpy + Pillow + tac.procedural_codebook_generator)
    runtime_dep_closure:        numpy, Pillow, tac.procedural_codebook_generator (v2 only)
    export_format:              custom (CH08: hand-rolled binary; no fp16/brotli/torch)
    score_aware_loss:           custom (NO TRAINING; closed-form LUT derivation at compress)
    bolt_on_loc_budget:         ~700 LOC (substrate_engineering exception per L7)
    no_op_detector_planned:     Catalog #139 byte-mutation smoke planned + scaffolded

Catalog #272 distinguishing-feature contract (v1 vs v2):

    distinguishing_feature_name:      chroma_lut_replacement_via_procedural_seed
    distinguishing_bytes_path:        CH08 v2 LUT_PAYLOAD slot (32-byte seed; substitutes 4096-byte LUT)
    inflate_consumer_function:        tac.substrates.nscs06_v8_chroma_lut.inflate._resolve_chroma_lut
    byte_mutation_smoke_passes:       SCAFFOLDED (Catalog #139); the smoke
                                       MUTATES one seed byte -> different derived
                                       LUT bytes -> different rendered frames.

Canonical-vs-unique decision per layer (per CLAUDE.md
"UNIQUE-AND-COMPLETE-PER-METHOD operating mode" + Catalog #290):

| Layer | Decision | Rationale |
|---|---|---|
| Architecture (this package) | UNIQUE | The (levels x classes) LUT shape is structurally distinct from v7's (classes,) anchor and from grayscale_lut's (256,) chroma table |
| Compress-side LUT derivation | UNIQUE | Per-bin median over (level, class) bins from GT pixels; no canonical helper exists for this aggregation |
| Inflate runtime | UNIQUE | Numpy + Pillow only; per-pixel LUT lookup |
| Procedural seed derivation | ADOPT canonical | ``tac.procedural_codebook_generator.derive_codebook_from_seed`` |
| Auth eval routing | ADOPT canonical | ``tac.substrates._shared.smoke_auth_eval_gate.gate_auth_eval_call`` (Catalog #226) |
| NVML/Modal/CUDA env hygiene | ADOPT canonical | Catalog #244 NVML block in remote driver |
| Mount manifest | ADOPT canonical | ``tac.deploy.modal.mount_manifest.build_training_image`` (Catalog #153) |
| eval_roundtrip simulation | PRESERVE HARD-EARNED | 384->874->uint8->384 simulated at compress-time |
| strict-scorer-rule | PRESERVE HARD-EARNED | inflate.py imports ZERO scorer code |
| Catalog #220 operational mechanism | PRESERVE HARD-EARNED | LUT payload IS the operational mechanism (v1 inline OR v2 seed) |
| Trainer skeleton helpers | ADOPT canonical | ``tac.substrates._shared.trainer_skeleton`` |
| device_or_die | ADOPT canonical | per Catalog #178 TF32-helper consolidation |
| SubstrateContract decoration | ADOPT canonical | ``@register_substrate(NSCS06_V8_CHROMA_LUT_SUBSTRATE_CONTRACT)`` per Catalog #241/#242 |
| Lane registry | ADOPT canonical | ``tools/lane_maturity.py`` per Catalog #126 pre-registration |

Reactivation criteria (L0 -> L1 per Catalog #325 per-substrate symposium):
    Per-substrate symposium per Catalog #325 lands a PROCEED verdict (1-2
    week window from 2026-05-21 -> 2026-06-04). Pre-symposium dispatch is
    BLOCKED via ``dispatch_enabled: false`` + ``research_only: true`` in
    the operator-authorize recipe. The 14-day symposium window is operator-
    routed; the first paid Modal T4 smoke is queued post-symposium.

CLAUDE.md compliance:
- No silent device defaults (compress side runs scorer via canonical helper)
- No scorer loading at inflate (inflate.py has ZERO torch / scorer imports)
- No /tmp paths (compress writes under args.output_dir; inflate honors $1/$2/$3)
- No KILL verdicts (DEFER-pending-per-substrate-symposium per Catalog #325)
- Apples-to-apples axis labels on every score CLAIM ([contest-CUDA] only; this
  scaffold makes NO score claim — predicted ΔS is PREDICTED-only per Catalog #287 + #323)
"""

from .architecture import (
    CHROMA_LUT_BYTES_DEFAULT,
    CHROMA_LUT_DTYPE_DEFAULT,
    GRAYSCALE_LEVELS_DEFAULT,
    NUM_SEGNET_CLASSES,
    PROCEDURAL_SEED_SIZE_BYTES,
    Nscs06V8ChromaLutConfig,
    build_chroma_lut_from_ground_truth,
    lookup_rgb_via_chroma_lut,
)
from .archive import (
    CH08_HEADER_FMT,
    CH08_HEADER_SIZE,
    CH08_MAGIC,
    CH08_SCHEMA_VERSION_INLINE_LUT,
    CH08_SCHEMA_VERSION_PROCEDURAL_SEED,
    GENERATOR_KIND_TAG,
    GENERATOR_KIND_TAG_INVERSE,
    POSE_DIMS,
    Nscs06V8Archive,
    pack_archive,
    parse_archive,
)
from .inflate import (
    inflate_one_video,
    main_cli,
    select_inflate_device,
)
from .procedural_variant import (
    CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT,
    PROCEDURAL_LUT_SENTINEL,
    ProceduralVariantConfig,
    ProceduralVariantError,
    derive_procedural_chroma_lut_replacement,
    predicted_archive_bytes_saved,
    predicted_delta_s,
    verify_procedural_lut_in_domain,
    verify_seed_mutation_changes_lut_bytes,
)
from .revisions import (
    CANONICAL_GENERATOR_KIND_ABLATION_AXIS,
    CANONICAL_LUMA_QUANTIZATION_LEVEL_ABLATION_AXIS,
    CANONICAL_PER_LEVEL_CLASS_AGGREGATION_ABLATION_AXIS,
    PER_ASSUMPTION_ABLATION_DIR_NAME,
    PER_ASSUMPTION_ABLATION_TABLE_SCHEMA_VERSION,
    CarmackMvpFirstPreSmokeVerificationVerdict,
    CarmackMvpFirstStepResult,
    MultiScaleDykstraFeasibilityVerdict,
    PerAssumptionAblationArm,
    PerAssumptionAblationLadder,
    build_per_assumption_ablation_ladder,
    build_per_assumption_ablation_table_path,
    emit_per_assumption_ablation_table_json,
    run_carmack_mvp_first_pre_smoke_verification,
    verify_multi_scale_dykstra_feasibility,
)
from .substrate_contract import NSCS06_V8_CHROMA_LUT_SUBSTRATE_CONTRACT

PROCEDURAL_VARIANT_AVAILABLE: bool = True
"""Flag set True at scaffold landing (sister of grayscale_lut + DP1 + VQ-VAE
``PROCEDURAL_VARIANT_AVAILABLE``). Trainers + cathedral consumers may key off
this flag to detect that the v8 chroma-LUT procedural variant envelope is
importable. This is NOT a score-eligible replacement archive until the per-
substrate symposium per Catalog #325 lands a PROCEED verdict + paired-smoke
contest-CUDA + contest-CPU anchor per CLAUDE.md "Submission auth eval - BOTH
CPU AND CUDA"."""


__all__ = [
    "CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT",
    "CANONICAL_GENERATOR_KIND_ABLATION_AXIS",
    "CANONICAL_LUMA_QUANTIZATION_LEVEL_ABLATION_AXIS",
    "CANONICAL_PER_LEVEL_CLASS_AGGREGATION_ABLATION_AXIS",
    "CH08_HEADER_FMT",
    "CH08_HEADER_SIZE",
    "CH08_MAGIC",
    "CH08_SCHEMA_VERSION_INLINE_LUT",
    "CH08_SCHEMA_VERSION_PROCEDURAL_SEED",
    "CHROMA_LUT_BYTES_DEFAULT",
    "CHROMA_LUT_DTYPE_DEFAULT",
    "CarmackMvpFirstPreSmokeVerificationVerdict",
    "CarmackMvpFirstStepResult",
    "GENERATOR_KIND_TAG",
    "GENERATOR_KIND_TAG_INVERSE",
    "GRAYSCALE_LEVELS_DEFAULT",
    "MultiScaleDykstraFeasibilityVerdict",
    "NSCS06_V8_CHROMA_LUT_SUBSTRATE_CONTRACT",
    "NUM_SEGNET_CLASSES",
    "Nscs06V8Archive",
    "Nscs06V8ChromaLutConfig",
    "PER_ASSUMPTION_ABLATION_DIR_NAME",
    "PER_ASSUMPTION_ABLATION_TABLE_SCHEMA_VERSION",
    "POSE_DIMS",
    "PROCEDURAL_LUT_SENTINEL",
    "PROCEDURAL_SEED_SIZE_BYTES",
    "PROCEDURAL_VARIANT_AVAILABLE",
    "PerAssumptionAblationArm",
    "PerAssumptionAblationLadder",
    "ProceduralVariantConfig",
    "ProceduralVariantError",
    "build_chroma_lut_from_ground_truth",
    "build_per_assumption_ablation_ladder",
    "build_per_assumption_ablation_table_path",
    "derive_procedural_chroma_lut_replacement",
    "emit_per_assumption_ablation_table_json",
    "inflate_one_video",
    "lookup_rgb_via_chroma_lut",
    "main_cli",
    "pack_archive",
    "parse_archive",
    "predicted_archive_bytes_saved",
    "predicted_delta_s",
    "run_carmack_mvp_first_pre_smoke_verification",
    "select_inflate_device",
    "verify_multi_scale_dykstra_feasibility",
    "verify_procedural_lut_in_domain",
    "verify_seed_mutation_changes_lut_bytes",
]
