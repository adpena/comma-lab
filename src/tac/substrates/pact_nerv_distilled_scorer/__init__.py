# SPDX-License-Identifier: MIT
"""tac.substrates.pact_nerv_distilled_scorer — Pact-NeRV-DistilledScorer (L0 SKETCH).

Group 2 variant #6 of PACT-NERV-ULTIMATE (research memo
`.omx/research/pact_nerv_ultimate_research_and_design_20260520T193443Z.md`).
Mid-LOC apparatus-aligned variant compatible with the WAVE-3-NERV-LITERATURE-L0-RESCOPED
canonical pattern + sister PACT-NERV-IA3 substrate scaffold.

Literature anchor: Hinton-Vinyals-Dean 2015 *"Distilling the Knowledge in a
Neural Network"*, arXiv:1503.02531. Canonical OSS reference repo:
`peterliht/knowledge-distillation-pytorch`. KL-T=2.0 distillation IS the
canonical Quantizr 0.33 [contest-CUDA] technique per CLAUDE.md "Quantizr
intelligence — verified competitive data" + cross-references in the inner
council Hinton seat. <!-- HISTORICAL_SCORE_LITERAL_OK:quantizr_0_33_canonical_anchor_2026-04-21_landed -->

The distinguishing primitive vs sister IA3 substrate: Pact-NeRV-DistilledScorer
distills frozen-teacher SegNet + PoseNet logits into a compact internal
scorer surrogate via KL-T=2.0 (Hinton 1503.02531 §3) THEN uses the surrogate
to condition the HNeRV-class base decoder. The internal scorer surrogate
replaces direct scorer routing at training time, providing (a) ~5x faster
gradient flow per Quantizr empirical anchor, (b) reduced VRAM at smoke
time, and (c) Hinton's dissent fulfillment per PACT-NERV-DESIGN-SYMPOSIUM
op-routable #9.

Status: **L0 SKETCH** (research_only=true). NO full main, NO dispatch.

Council design memo:
    `.omx/research/pact_nerv_distilled_scorer_l0_scaffold_design_20260520T<UTC>.md`

13 HNeRV parity-discipline lessons compliance — design-time declaration:

| Lesson | Status |
|---|---|
| L1 substrate must be score-aware | PLANNED (score_aware_loss.py wired; distilled surrogate at L1) |
| L2 export-first archive grammar | PASS (archive.py declared BEFORE training; PDS magic) |
| L3 monolithic 0.bin | PASS (single-file fixed-offset grammar, 22-byte header) |
| L4 inflate ≤ 200 LOC | PASS (target ~130 LOC; torch + brotli) |
| L5 full RGB renderer | PASS (NOT a mask codec; outputs (T, 3, H, W) per HNeRV parity L5) |
| L6 score-domain Lagrangian | PASS (B(theta)/N + d_seg + sqrt(d_pose)) |
| L7 bolt-on ≤ 350 LOC | substrate_engineering exception (~250 total per task spec) |
| L8 eval-roundtrip + diff yuv6 | PASS (patches diff yuv6 BEFORE scorer; eval_roundtrip MANDATORY) |
| L9 runtime closure | PASS (torch + brotli; declared) |
| L10 mask/pose coupling | N/A (renderer replaces full slot) |
| L11 no-op detector | PASS (executable byte-mutation smoke in tests) |
| L12 single-LOC review discipline | PASS (each file reviewable in 30s) |
| L13 KILL last resort | PASS (losing variants DEFERRED per Forbidden premature KILL) |

Catalog #124 archive-grammar 8 fields:
    archive_grammar:           monolithic single-file 0.bin fixed offsets (PDS)
    parser_section_manifest:   parse_archive() -> 5 sections (header +
                               distilled_surrogate_blob + decoder_blob +
                               latent_blob + meta_blob)
    inflate_runtime_loc_budget: ≤ 150 LOC
    runtime_dep_closure:       torch, brotli
    export_format:             FP4+Brotli (Quantizr canonical) for weights;
                               int16 latents
    score_aware_loss:          L = alpha*B/N + beta*d_seg + gamma*sqrt(d_pose)
    bolt_on_loc_budget:        ~250 LOC (substrate_engineering tag; task-cap)
    no_op_detector_planned:    Catalog #139 _build_no_op_proof + byte-mutation
                               smoke

Cargo-cult audit per Catalog #303:
- KL-T=2.0 distill = HARD-EARNED (Hinton 1503.02531 §3 + Quantizr 0.33 [contest-CUDA]
  empirical anchor + Hinton inner council seat). <!-- HISTORICAL_SCORE_LITERAL_OK:quantizr_0_33_anchor_in_cargo_cult_audit -->
- Internal scorer surrogate REPLACES direct scorer routing at training time =
  CARGO-CULTED-MAY-BE-PROMISING at L0 (alternative: hybrid — distill for early
  epochs, direct scorer for late epochs; Stage 1 dispatch validates the choice).
- Cross-attention-to-frozen-scorer-features = CARGO-CULTED at L0 (alternative:
  KL-on-logits-only per pure Hinton; Stage 1 sweep validates).

Sister NeRV-family packages (post-landing: 22 total).
"""

from .architecture import (
    DistilledScorerSurrogate,
    PactNervDistilledScorerConfig,
    PactNervDistilledScorerSubstrate,
)
from .archive import (
    PDS_HEADER_SIZE,
    PDS_MAGIC,
    PDS_SCHEMA_VERSION,
    PactNervDistilledScorerArchive,
    pack_archive,
    parse_archive,
)
from .score_aware_loss import (
    PactNervDistilledScorerScoreAwareLoss,
    ScoreAwareLossWeights,
)

__all__ = [
    "DistilledScorerSurrogate",
    "PDS_HEADER_SIZE",
    "PDS_MAGIC",
    "PDS_SCHEMA_VERSION",
    "PactNervDistilledScorerArchive",
    "PactNervDistilledScorerConfig",
    "PactNervDistilledScorerScoreAwareLoss",
    "PactNervDistilledScorerSubstrate",
    "ScoreAwareLossWeights",
    "pack_archive",
    "parse_archive",
]
