# SPDX-License-Identifier: MIT
"""tac.substrates.pact_nerv_selector_v2 - Pact-NeRV-SELECTOR-V2 (substrate L0 SKETCH).

Group 3 (SELECTOR-PARADIGM-EXTENSIONS) variant per PACT-NERV-ULTIMATE (commit
``e3ad4243a``) variant #11 + task-spec G3 family (arithmetic coding extension
of the FEC6 fixed-Huffman k=16 frame-exploit selector). Direct empirical
extension of CROSS-CANDIDATE finding #1 (+259 bytes / +0.00333 ratio per
PACT-NERV-ULTIMATE Section 5).

Literature anchor: Witten-Neal-Cleary 1987 *"Arithmetic Coding for Data
Compression"* + Said 2004 *"Introduction to Arithmetic Coding -- Theory
and Practice"*. Arithmetic coding achieves fractional-bit precision (vs
Huffman's integer-bit code-lengths) so a 16-mode palette whose mode
frequencies are non-power-of-2 gains a fraction-of-a-byte per selector
on average; for the FEC6 600-pair selector that translates to ~30-100
bytes of additional rate savings vs the canonical FEC6 fixed-Huffman k=16.

Hypothesis (per PACT-NERV-ULTIMATE Variant #11 + CROSS-CANDIDATE finding
#1 empirical anchor): the FEC6 fixed-Huffman k=16 wastes fractional bits
on every selector. Arithmetic coding tightens the code-length without
changing the 16-mode palette. The substrate is the FEC6 frame-exploit
selector + an arithmetic coder over the same palette; the rate-axis
prediction is ``-30..-100 bytes / +0.000..-0.003 [contest-CPU]``.

Architecture (L0 SCAFFOLD; council-pending Stage 1 dispatch operator-gated):

    HNeRV-class base decoder (mirrors pact_nerv_ia3 / boost_nerv canonical
    sister; FP11 weights + brotli)
       |
       v
    Per-pair selector index in [0, 16) per FEC6 k=16 palette
       |
       v
    Arithmetic coder (fixed per-palette static probability table; ~50 LOC
    primitive per Witten 1987)
       |
       v
    rgb_0 / rgb_1: standard HNeRV decoder forward + FEC6 selector-conditioned
    deterministic frame-0 transforms per the canonical FES1 palette.

Status: **L0 SKETCH** (research_only=true). NO full main, NO dispatch.

13 HNeRV parity-discipline lessons compliance - design-time declaration:

| Lesson | Status |
|---|---|
| L1 score-aware | PLANNED (score_aware_loss.py wired; trainer at L1) |
| L2 export-first | PASS (archive.py declared BEFORE training) |
| L3 monolithic 0.bin | PASS (PSV2 magic; single-file fixed-offset grammar) |
| L4 inflate ≤ 200 LOC | PASS (target ~140 LOC; torch + brotli) |
| L5 full RGB renderer | PASS (NOT a mask codec; outputs (T, 3, H, W)) |
| L6 score-domain Lagrangian | PASS |
| L7 bolt-on ≤ 350 LOC | PASS (~150 LOC arithmetic-coder primitive over fec6 selector) |
| L8 eval-roundtrip + diff yuv6 | PASS (eval_roundtrip MANDATORY DEFAULT) |
| L9 runtime closure | PASS (torch + brotli; declared) |
| L10 mask/pose coupling | N/A (renderer replaces full slot) |
| L11 no-op detector | PASS (executable byte-mutation smoke in tests) |
| L12 single-LOC review | PASS |
| L13 KILL last resort | PASS |

Catalog #124 archive-grammar 8 fields:
    archive_grammar:           monolithic single-file 0.bin (PSV2 magic)
    parser_section_manifest:   header + base_decoder_blob (brotli FP11) +
                               selector_arithmetic_blob (arithmetic-coded
                               selector indices over FEC6 k=16 palette) +
                               latents_blob + meta_blob
    inflate_runtime_loc_budget: <= 200 LOC (PR101 GOLD reference)
    runtime_dep_closure:       torch, brotli
    export_format:             FP4+Brotli for decoder weights; arithmetic-coded
                               u8 selectors over FEC6 k=16 palette
    score_aware_loss:          L = alpha*B/N + beta*d_seg + gamma*sqrt(d_pose)
    bolt_on_loc_budget:        ~150 LOC arithmetic-coder primitive
    no_op_detector_planned:    Catalog #139 _build_no_op_proof; selector
                               perturbation MUST change rendered frames

Cargo-cult audit per Catalog #303 (HARD-EARNED vs CARGO-CULTED):

- Arithmetic coding over the FEC6 16-mode palette = HARD-EARNED-LITERATURE
  (Witten 1987 + Said 2004; fractional-bit precision is the canonical
  rate-extremal entropy coder).
- FEC6 k=16 palette inherited = HARD-EARNED-EMPIRICAL (the +259-byte /
  +0.00333 anchor in CROSS-CANDIDATE finding #1 validates the palette).
- Static (non-adaptive) per-palette probability table at L0 = CARGO-CULTED
  (alternative at L1: per-pair adaptive context model; sister of SELECTOR-V3
  per-pair-difficulty conditioning).
- Per-symbol arithmetic encoding (vs per-pair grouped coding) = CARGO-CULTED
  at L0 (sweep at L1 after Stage 1 dispatch lands first empirical anchor).

Sister NeRV-family packages (17 total after this lands; sister of
PACT-NERV-IA3 canonical pattern at commit `9cf9bdb16` + NERV-LITERATURE-L0
sister boost_nerv at commit `d9aaf7c13`):

- pact_nerv_ia3 (Stage 1 HYBRID; sister Pact-NeRV variant)
- pact_nerv_selector_v2 (THIS variant; arithmetic coding extension)
- pact_nerv_selector_v3 (sister G3; Rice-Golomb)
- pact_nerv_selector_v4 (sister G3; RLE)
- pact_nerv_ia3_multi (sister G3; multi-layer IA3 + per-pair difficulty)
- pact_nerv_asymmetric_boundary (sister G3; per-class boundary)
"""

from .architecture import (
    PactNervSelectorV2Config,
    PactNervSelectorV2Substrate,
    ArithmeticSelectorCoder,
)
from .archive import (
    PactNervSelectorV2Archive,
    PSV2_HEADER_SIZE,
    PSV2_MAGIC,
    PSV2_SCHEMA_VERSION,
    pack_archive,
    parse_archive,
)
from .score_aware_loss import (
    PactNervSelectorV2ScoreAwareLoss,
    ScoreAwareLossWeights,
)

__all__ = [
    "ArithmeticSelectorCoder",
    "PSV2_HEADER_SIZE",
    "PSV2_MAGIC",
    "PSV2_SCHEMA_VERSION",
    "PactNervSelectorV2Archive",
    "PactNervSelectorV2Config",
    "PactNervSelectorV2ScoreAwareLoss",
    "PactNervSelectorV2Substrate",
    "ScoreAwareLossWeights",
    "pack_archive",
    "parse_archive",
]
