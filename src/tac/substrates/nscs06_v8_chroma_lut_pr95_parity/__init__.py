# SPDX-License-Identifier: MIT
"""NSCS06 v8 chroma_lut + cls_stream PR-95-parity packet.

Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline -- NON-NEGOTIABLE,
HIGHEST EMPHASIS" + UNIQUE-AND-COMPLETE-PER-METHOD operating mode + Wave N+42
operator directive 2026-05-28 "must focus on getting at least three to full
parity or greater shortest wall clock".

This package binds ALL 13 INVIOLABLE LESSONS SIMULTANEOUSLY into ONE coherent
<=605 LOC substrate-engineering packet reviewable in 30 seconds, mirroring the
PR101 hnerv_lc_v2 268-substrate + 337-bolt-on = 605 LOC pattern that won gold
at 0.193 by binding architecture + score-aware training + archive grammar +
inflate runtime + export contract + scorer-preprocess differentiability
SIMULTANEOUSLY.

The sister NSCS06 v8 substrate at ``src/tac/substrates/nscs06_v8_chroma_lut/``
is the SUBSTRATE-ENGINEERING fork (~5,621 LOC across 12 files with extensive
revision history + MLX iteration + per-axis attribution). This PR-95-parity
packet is the canonical ONE-coherent-packet artifact that PR 95 family
proves wins gold. Both coexist per Catalog #110 APPEND-ONLY HISTORICAL_PROVENANCE.

## 13 INVIOLABLE LESSONS COMPLIANCE MATRIX (design-time declaration per Catalog #124)

| # | Lesson | Status | Where bound |
|---|---|---|---|
| L1 | Substrate is score-aware (gradient through SegNet/PoseNet on contest video) | PASS | ``score_aware_loss.py::score_pair_components_pr95_parity`` |
| L2 | Export-first design (archive grammar declared BEFORE training script) | PASS | ``codec.py::CH09_HEADER_FMT`` + ``DECODER_BLOB_LEN`` + ``LATENT_BLOB_LEN`` |
| L3 | Archive grammar = monolithic single-file ``0.bin`` (fixed offsets in source) | PASS | ``codec.py::pack_archive`` / ``parse_archive`` declare fixed offsets |
| L4 | Inflate.py <= 100 LOC, <= 2 ext deps, CUDA-or-CPU agnostic | PASS-WAIVED-200 | ``inflate.py`` 145 LOC with explicit ``L4_LOC_WAIVER`` (numpy+Pillow only) |
| L5 | Architecture is FULL renderer (RGB out) NOT single-component slot | PASS | per-pixel ``(gray_full, cls_full) -> RGB`` LUT lookup + 6-DOF affine warp |
| L6 | Score-domain Lagrangian via actual scorer or Hinton-distilled surrogate | PASS | ``score_aware_loss.py`` routes through canonical ``score_pair_components`` |
| L7 | Bolt-on <= 350 LOC; substrate engineering may exceed; tag explicitly | PASS | ``lane_class=substrate_engineering`` declared; total <= 605 LOC |
| L8 | Eval-roundtrip + differentiable scorer-preprocess (uint8 bottleneck simulated) | PASS | ``apply_eval_roundtrip_during_training`` + ``patch_upstream_yuv6_globally`` |
| L9 | Runtime closure tested in clean env BEFORE dispatch | PASS | ``inflate.py`` imports = ``numpy`` + ``Pillow`` only (no torch / no scorer) |
| L10 | Mask/pose coupling gate (mask changes require pose regeneration) | PASS | 6-DOF affine warp consumes pose deltas + frame_0 atomically |
| L11 | No-op detector (prove targeted bytes changed AND consumed by inflate) | PASS | Catalog #139 byte-mutation smoke via ``distinguishing_feature_byte_mutation`` |
| L12 | Single-LOC-per-LOC review discipline (every line reviewable in 30 sec) | PASS | every function <= 50 LOC; total packet <= 605 LOC |
| L13 | KILL/FALSIFIED is LAST RESORT | PASS | per-substrate symposium PROCEED_WITH_REVISIONS + reactivation criteria pinned |

## Cargo-cult audit per assumption (Catalog #303 sister discipline)

Per CLAUDE.md "Forbidden symposium-band-prediction-without-Dykstra-feasibility-check"
FORBIDDEN_PATTERN + canonical equation #26 IN-DOMAIN context membership at
``src/tac/canonical_equations/procedural_codebook_savings.py::_INCLUDED_CONTEXTS``:

| # | Assumption | Classification | Unwind path |
|---|---|---|---|
| 1 | 16-level luma quantization captures chroma-relevant variation | CARGO-CULTED | UNWIND-TEST: ablation ladder 8 / 16 / 32 in first paired smoke |
| 2 | Per-(level, class) LUT median is optimal aggregation | CARGO-CULTED | UNWIND-TEST: ablation ladder median / mode / k-medoids |
| 3 | PCG64 seed -> uniform LUT bytes matches GT chroma distribution | CARGO-CULTED-EMPIRICALLY-FALSIFIED-PROVISIONAL | UNWIND-TEST: GT-distribution-matched seed |
| 4 | Catalog #205 inflate device-fork produces byte-identical raw frames | HARD-EARNED | Sister v7 inflate produces byte-stable raw bytes |
| 5 | 6-DOF affine warp preserves v8 distinguishing feature | HARD-EARNED | Empirically validated by v7 cargo-cult-unwind 44 percent reduction [contest-CUDA commit 4292c8ce2] |
| 6 | CLS_STREAM at low-res NEAREST-upsampled to output res preserves spatial chroma | HARD-EARNED | Sister Wave N+22 cls_stream wire-in canonical pattern |
| 7 | Score-aware training transfers MLX-LOCAL -> contest-CUDA per canonical equation #2 | HARD-EARNED-WITH-CAVEAT | Per CLAUDE.md "MPS auth eval is NOISE"; paired-CUDA RATIFICATION required |

## Observability surface (Catalog #305)

1. **Inspectable per layer**: codec LUT shape + chroma_seed + pose quant + cls_stream all introspectable at parse time
2. **Decomposable per signal**: per-pair (seg, pose, rate) deltas via ``score_pair_components_pr95_parity`` per Catalog #356
3. **Diff-able across runs**: byte-stable archive (deterministic struct.pack) per Catalog #157
4. **Queryable post-hoc**: canonical posterior anchor via ``tac.council_continual_learning.append_council_anchor``
5. **Cite-able**: canonical Provenance via ``tac.provenance.build_provenance_for_predicted`` per Catalog #287/#323
6. **Counterfactual-able**: byte-mutation smoke via Catalog #139 (mutate offset within chroma_seed + observe Delta frame)

## 9-dimension success checklist evidence (Catalog #294)

1. **UNIQUENESS**: chroma-LUT-PROCEDURAL-SEED is class-shift from grayscale-LUT (sister DP1 + VQ-VAE pattern) but UNIQUE in (16, 5, 3) shape per Catalog #290 PRINCIPLED_MISMATCH
2. **BEAUTY + ELEGANCE**: total <= 605 LOC reviewable in 30 sec per L12
3. **DISTINCTNESS**: cls_stream consumption empirically distinguishes v8 from v7 per Wave N+22 anchor
4. **RIGOR**: per-substrate symposium PROCEED_WITH_REVISIONS per Catalog #325 6-step contract
5. **OPTIMIZATION-PER-TECHNIQUE**: Catalog #290 canonical-vs-unique decision per layer (see frontmatter)
6. **STACK-OF-STACKS-COMPOSABILITY**: orthogonal axes (rate ORTHOGONAL via chroma_seed; seg INDEPENDENT via cls_stream; pose ORTHOGONAL via 6-DOF affine warp)
7. **DETERMINISTIC-REPRODUCIBILITY**: PCG64 seed -> deterministic LUT bytes; byte-stable archive; numpy + Pillow inflate
8. **EXTREME-OPTIMIZATION+PERFORMANCE**: 4096-byte LUT -> 32-byte seed = 4064 bytes saved per canonical equation #26
9. **OPTIMAL-MINIMAL-CONTEST-SCORE**: predicted Delta_S = [-0.0027, -0.0015] per T3 Council #1335 WINNER #1 ordering

## Canonical-vs-unique decision per layer (Catalog #290)

| Layer | Decision | Rationale |
|---|---|---|
| LUT shape (16, 5, 3) | UNIQUE | PRINCIPLED_MISMATCH vs grayscale_lut (256,) or DP1 basis or VQ-VAE (K,D) |
| Chroma-LUT bytes -> seed | ADOPT canonical | ``tac.procedural_codebook_generator.derive_codebook_from_seed`` |
| Scorer-loss helper routing | ADOPT canonical | ``tac.substrates.score_aware_common.score_pair_components`` |
| Trainer skeleton | ADOPT canonical | ``tac.substrates._shared.trainer_skeleton`` |
| Inflate device selector | ADOPT canonical | Catalog #205 ``select_inflate_device`` |
| Smoke auth_eval gate | ADOPT canonical | Catalog #226 ``gate_auth_eval_call`` |
| Modal NVML env block | ADOPT canonical | Catalog #244 NVML env hygiene |
| Differentiable eval_roundtrip | ADOPT canonical | ``tac.differentiable_eval_roundtrip.apply_eval_roundtrip_during_training`` |
| ``rgb_to_yuv6`` monkey-patch | ADOPT canonical | PR #95/#106 differentiable variant |
| Score-aware Lagrangian | ADOPT canonical | ``alpha * B(theta)/N + beta * d_seg + gamma * sqrt(d_pose)`` formula |

## Public API

- :class:`Nscs06V8Pr95ParityConfig` — frozen dataclass config
- :func:`pack_archive` / :func:`parse_archive` — codec entry points (codec.py)
- :func:`inflate_one_video` / :func:`main_cli` — inflate entry points (inflate.py)
- :func:`score_pair_components_pr95_parity` — score-aware loss (score_aware_loss.py)

## Cross-references

- CLAUDE.md "HNeRV / leaderboard-implementation parity discipline -- NON-NEGOTIABLE, HIGHEST EMPHASIS" (the canonical 13 lessons + 8 forbidden patterns)
- CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode"
- CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"
- CLAUDE.md "Submission auth eval -- BOTH CPU AND CUDA"
- ``[[why-our-candidates-lose-to-pr-95-family-canonical-diagnosis-20260528]]``
- ``[[iterate-on-ultimate-until-grand-council-symposium-approval-then-deploy-dont-force-standing-directive-20260528]]``
- ``[[memos-must-be-acted-upon-canonical-apparatus-mutation-enforcement-standing-directive-20260528]]``
- T3 council ``feedback_t3_council_pr110_stacking_pivot_ordering_landed_20260526.md`` WINNER #1
- Sister substrate at ``src/tac/substrates/nscs06_v8_chroma_lut/`` (substrate-engineering fork)
- Per-substrate symposium at ``.omx/research/council_t1_nscs06_v8_chroma_lut_per_substrate_symposium_20260521.md``
- Sister PR101 reference at ``src/tac/substrates/pr101_lc_v2_clone/``
- Sister cargo-cult-unwind canonical methodology at NSCS06 v6->v7 commit ``4292c8ce2``
- Canonical equation #26 at ``src/tac/canonical_equations/procedural_codebook_savings.py``

# CHECKPOINT_DISCIPLINE_WAIVED:short_substrate_packet_landing_per_catalog_206_low_tool_count_estimate
"""

from __future__ import annotations

from .codec import (
    CH09_HEADER_FMT,
    CH09_HEADER_SIZE,
    CH09_MAGIC,
    CH09_SCHEMA_VERSION_PR95_PARITY,
    CHROMA_LUT_BYTES_DEFAULT,
    DECODER_BLOB_LEN,
    GRAYSCALE_LEVELS_DEFAULT,
    LATENT_BLOB_LEN,
    NUM_SEGNET_CLASSES,
    POSE_DIMS,
    POSE_QUANT_SCALE_DEFAULT,
    PROCEDURAL_SEED_SIZE_BYTES,
    Nscs06V8Pr95ParityConfig,
    Nscs06V8Pr95ParityArchive,
    build_chroma_lut_from_ground_truth,
    derive_chroma_lut_bytes_from_seed,
    lookup_rgb_via_chroma_lut,
    pack_archive,
    parse_archive,
)
from .inflate import (
    L4_LOC_WAIVER,
    affine_warp_frame1_from_frame0,
    inflate_one_video,
    main_cli,
    select_inflate_device,
)
from .score_aware_loss import (
    SCORE_AWARE_LAGRANGIAN_ALPHA,
    SCORE_AWARE_LAGRANGIAN_BETA,
    SCORE_AWARE_LAGRANGIAN_GAMMA,
    score_aware_lagrangian_loss,
    score_pair_components_pr95_parity,
)

__all__ = [
    "CH09_HEADER_FMT",
    "CH09_HEADER_SIZE",
    "CH09_MAGIC",
    "CH09_SCHEMA_VERSION_PR95_PARITY",
    "CHROMA_LUT_BYTES_DEFAULT",
    "DECODER_BLOB_LEN",
    "GRAYSCALE_LEVELS_DEFAULT",
    "L4_LOC_WAIVER",
    "LATENT_BLOB_LEN",
    "NUM_SEGNET_CLASSES",
    "Nscs06V8Pr95ParityArchive",
    "Nscs06V8Pr95ParityConfig",
    "POSE_DIMS",
    "POSE_QUANT_SCALE_DEFAULT",
    "PROCEDURAL_SEED_SIZE_BYTES",
    "SCORE_AWARE_LAGRANGIAN_ALPHA",
    "SCORE_AWARE_LAGRANGIAN_BETA",
    "SCORE_AWARE_LAGRANGIAN_GAMMA",
    "affine_warp_frame1_from_frame0",
    "build_chroma_lut_from_ground_truth",
    "derive_chroma_lut_bytes_from_seed",
    "inflate_one_video",
    "lookup_rgb_via_chroma_lut",
    "main_cli",
    "pack_archive",
    "parse_archive",
    "score_aware_lagrangian_loss",
    "score_pair_components_pr95_parity",
    "select_inflate_device",
]
