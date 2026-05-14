# SPDX-License-Identifier: MIT
"""tac.substrates.pr101_lc_v2_clone — forensic apples-to-apples PR101 GOLD clone.

L0 SKETCH substrate, ``research_only=true``. The clone re-implements PR101's
``hnerv_ft_microcodec`` architecture (the 0.193 [contest-CUDA] GOLD submission
per public claim) inside ``tac.*`` so that future work has a faithful baseline
that consumes Subagent C's 3 PR101 GOLD primitives end-to-end (storage order,
conv4 perms, byte maps).

Operator decision #1 surfaced in
``feedback_pr101_gold_primitive_port_landed_20260512.md`` (2026-05-12 approved
"all are approved"). The forensic value is apples-to-apples vs PR101's public
GOLD claim, NOT a new architecture or new method claim.

Catalog #109 compliance (FORBIDDEN in-place edits to public PR intake clones):
this module READS bytes/values from the intake clone documentation but writes
NOTHING into the intake clone tree. All new code lives under
``src/tac/substrates/pr101_lc_v2_clone/``.

13 HNeRV parity-discipline lessons compliance — design-time declaration:

| Lesson | Status |
|---|---|
| L1 substrate must be score-aware | PASS (score_aware_loss.py wired) |
| L2 export-first archive grammar | PASS (archive.py declared BEFORE training) |
| L3 monolithic 0.bin | PASS (single-file fixed offsets per PR101 contract) |
| L4 inflate <= 100 LOC, <= 2 deps | WAIVED (<= 200 LOC; PR101 fidelity requires sidecar parser) |
| L5 full RGB renderer | PASS (NOT a mask codec; full RGB pair) |
| L6 score-domain Lagrangian | PASS (B(theta)/N + d_seg + sqrt(d_pose)) |
| L7 bolt-on <= 350 LOC | substrate_engineering exception |
| L8 eval-roundtrip + diff yuv6 | PASS (wired via tac.differentiable_eval_roundtrip) |
| L9 runtime closure | PASS (torch + brotli + numpy; declared) |
| L10 mask/pose coupling | N/A (renderer replaces full RGB slot) |
| L11 no-op detector | PASS (byte-mutation smoke in tests) |
| L12 single-LOC review discipline | PASS (each file reviewable in <60s) |
| L13 KILL last resort | PASS (research_only; never killed prematurely) |

Catalog #124 archive-grammar 8 fields (declared inline below for the lane
registry notes; STRICT preflight checks these at design time):

    archive_grammar:           monolithic single-file 0.bin fixed offsets per PR101
    parser_section_manifest:   parse_archive() returns (decoder_sd, latents, meta)
    inflate_runtime_loc_budget: <= 200 LOC (waiver for PR101 sidecar fidelity)
    runtime_dep_closure:       torch, brotli, numpy
    export_format:             multi-brotli-stream decoder + lzma latents + raw sidecar
    score_aware_loss:          L = alpha*B/N + beta*d_seg + gamma*sqrt(d_pose)
    bolt_on_loc_budget:        substrate_engineering tag (~600 LOC)
    no_op_detector_planned:    byte-mutation smoke verifies bytes change archive

Primitive composition (Subagent C anchor):

    encode pipeline:
        state_dict -> per-tensor int8 quantise
            -> apply_storage_perm(...) for 4D conv tensors (CONV4_STORAGE_PERMS)
            -> encode_byte_map(arr_i8, strategy)              for sign encoding
            -> reorder_tensors_for_storage(tensors, schema)   for storage order
            -> partition_buffer_by_stream_ends(buffer, ...)   for split-brotli
            -> brotli.compress(stream)                        per stream
            -> concat(brotli_streams) -> DECODER_BLOB

    decode pipeline (inflate-time, mirrored):
        DECODER_BLOB -> N brotli streams -> concat raw bytes
            -> per-tensor decode_byte_map(payload, strategy)
            -> apply_inverse_perm(...) for 4D conv tensors
            -> reshape + dequantise -> state_dict

NEGZIG -128 precondition is enforced at encode time (runtime guard inside
``encode_decoder_compact``); violation raises ``ValueError`` so a trained
checkpoint that quantises any negzig-tagged tensor entry to -128 is caught
BEFORE archive bytes are written.

CLAUDE.md compliance summary:
* No scorer load at inflate time (strict-scorer-rule honored)
* No /tmp paths (uses tac canonical state paths)
* No MPS / silent-fallback device defaults
* No score claims (lane stays research_only; promotion_eligible=false)
* No archive bytes modified outside the typed primitives
* Co-authored commit via tools/subagent_commit_serializer.py per Catalog #117
"""

from __future__ import annotations

from .architecture import Pr101LcV2CloneConfig, Pr101LcV2CloneSubstrate
from .archive import (
    PR101_LC_V2_ARCHIVE_GRAMMAR,
    Pr101LcV2Archive,
    decode_decoder_compact,
    encode_decoder_compact,
    pack_archive,
    parse_archive,
)
from .curriculum import (
    CURRICULUM_STAGES,
    CurriculumStageConfig,
    Muon,
    apply_qat,
    cat_entropy_v2,
    ema_update,
    get_seg_loss_fn,
    partition_params_for_muon,
    pose_loss,
    restore_qat,
)
from .curriculum_enhanced import (
    ENHANCEMENT_KEYS,
    EnhancedCurriculumConfig,
    TernaryStageBudget,
    apply_cross_block_skip,
    apply_logit_softcap,
    audit_enhanced_curriculum_against_hnerv_parity_lessons,
    build_enhanced_stages,
    build_faithful_stages,
    build_optimizer_for_enhanced_stage,
    compute_wsd_lr,
    default_ternary_schedule,
    enhancement_summary,
    logit_softcap_30,
    stage0_pretrained_driving_prior_bootstrap,
    validate_enhanced_curriculum_config,
)
from .score_aware_loss import (
    Pr101LcV2CloneScoreAwareLoss,
    Pr101LcV2ScoreAwareLossWeights,
)

__all__ = [
    "CURRICULUM_STAGES",
    "CurriculumStageConfig",
    "ENHANCEMENT_KEYS",
    "EnhancedCurriculumConfig",
    "Muon",
    "PR101_LC_V2_ARCHIVE_GRAMMAR",
    "Pr101LcV2Archive",
    "Pr101LcV2CloneConfig",
    "Pr101LcV2CloneScoreAwareLoss",
    "Pr101LcV2CloneSubstrate",
    "Pr101LcV2ScoreAwareLossWeights",
    "TernaryStageBudget",
    "apply_cross_block_skip",
    "apply_logit_softcap",
    "apply_qat",
    "audit_enhanced_curriculum_against_hnerv_parity_lessons",
    "build_enhanced_stages",
    "build_faithful_stages",
    "build_optimizer_for_enhanced_stage",
    "cat_entropy_v2",
    "compute_wsd_lr",
    "decode_decoder_compact",
    "default_ternary_schedule",
    "ema_update",
    "encode_decoder_compact",
    "enhancement_summary",
    "get_seg_loss_fn",
    "logit_softcap_30",
    "pack_archive",
    "parse_archive",
    "partition_params_for_muon",
    "pose_loss",
    "restore_qat",
    "stage0_pretrained_driving_prior_bootstrap",
    "validate_enhanced_curriculum_config",
]
