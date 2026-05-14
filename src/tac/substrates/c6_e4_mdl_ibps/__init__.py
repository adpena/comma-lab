# SPDX-License-Identifier: MIT
"""tac.substrates.c6_e4_mdl_ibps — Minimum Description Length × Information Bottleneck Predictive Substrate.

The zen-Z1 LARGEST single bet for substrate-class shift per
`.omx/research/adjusted_theoretical_floor_v3_post_pr106_falsification_20260513.md` +
`.omx/research/zen_floor_band_v2_post_z1_ablation_20260514.md`. Z1 empirically
proved the HNeRV-family substrate class is saturated at ~99.3% MDL density on
A1; sub-0.10 requires a DIFFERENT substrate class, not a more efficient
encoding of the HNeRV grammar. C6 is the across-class-shift substrate.

Core insight (Tishby-Zaslavsky 2015 Information Bottleneck + Rissanen 1978 MDL)
------------------------------------------------------------------------------

Jointly minimize:

    L = ||scorer(decoded) - scorer(GT)||² + β · I(z; frames)

where:

    z ∈ R^d_z          : per-pair latent (10-50 dims; ULTRA-LOW-RATE)
    encoder            : variational q(z | frames) (small MLP/CNN)
    decoder            : reconstructs frames from z ONLY (no per-pair latent rows)
    β                  : IB Lagrangian; controls bit budget for I(z; frames)
    p(z) (prior)       : N(0, I) or learned

The IB-mutual-information `I(z; frames)` is upper-bounded by the variational
`KL(q(z|frames) || p(z))` (Tishby-Zaslavsky 2017 — VIB upper bound).

**Across-class differentiation from HNeRV:**
- HNeRV stores (decoder + per-pair latent_blob); latent_blob is 100% structurally
  required (Z1 finding: LZMA-encoded; every byte desyncs the entropy decoder).
- C6 stores (encoder + decoder + per-pair `z`); `z` is a TINY low-rate sufficient
  statistic of frames-given-scorer. The discardable information (everything
  scorer is invariant to: high-frequency texture, lighting, occluded geometry)
  is COMPRESSED OUT at the encoder, not encoded then decompressed.
- HNeRV's latent_blob (~15KB) carries ~all of the source-entropy.
- C6's per-pair `z` carries only the scorer-relevant subspace (predicted ~3-5 KB total).

**Predicted ΔS:** -0.030 to -0.080 vs PR101 0.193 → predicted score band
**[0.113, 0.163]** `[mathematical-derivation]` per E4 ledger.

Catalog #124 archive-grammar 8 fields (declared inline so AST walker observes them)
-----------------------------------------------------------------------------------

- ``archive_grammar``: monolithic single-file ``0.bin`` (substrate-engineering)
- ``parser_section_manifest``: IBPS1 header + encoder_blob + decoder_blob +
  latent_blob (per-pair z int8) + meta JSON
- ``inflate_runtime_loc_budget``: ≤200 LOC substrate-engineering waiver
  (encoder/decoder + latent dequant + bilinear → contest HW)
- ``runtime_dep_closure``: torch + brotli only (HNeRV parity L4 ≤ 2 deps)
- ``export_format``: IBPS1 monolithic single-zip-member ``0.bin``
- ``score_aware_loss``: ``MDLIBPSScoreAwareLoss`` routes through
  ``score_pair_components`` per Catalog #164; eval-roundtrip mandatory
- ``bolt_on_loc_budget``: ``lane_class=substrate_engineering`` (HNeRV parity L7);
  encoder + decoder + IB regularizer composition is substrate engineering
- ``no_op_detector_planned``: emit/parse roundtrip preserves bytes byte-for-byte;
  decoder_blob + latent_blob are frame-affecting at inflate, meta/header are
  parse/config gates, and encoder_blob is training/provenance-only because
  inflate calls ``frames_for_encoder=None``. Encoder-only byte changes must not
  be claimed as score-affecting without a future runtime path that consumes
  q(z|frames).

target_modes: ``contest_one_video_replay``, ``contest_generalized``, ``research_substrate``
lane_class: ``substrate_engineering``
research_only: false (export-first, all 8 fields declared)
canary_status: ``independent_substrate`` (substrate-CLASS shift from HNeRV-family)

Probe-disambiguator (Catalog #125 hook #6)
------------------------------------------

Two defensible interpretations of "what dominates ΔS":

1. **Decoder-class hypothesis** — the IB decoder's ability to reconstruct
   scorer-relevant features from the low-dim `z` dominates ΔS.
2. **Encoder-bottleneck hypothesis** — the IB encoder's information-theoretic
   compression to MDL-optimal `z` dominates ΔS.

Both modes are inside the same substrate; the probe sweeps `β` over [0.01, 10]
and observes which axis (decoder reconstruction error OR encoder I(z;frames))
correlates with the empirical ΔS. Memo at
``tools/probe_c6_decoder_vs_encoder_dominance.py`` (planned post-smoke).

6-hook wire-in (Catalog #125 NON-NEGOTIABLE)
-------------------------------------------

1. **Sensitivity-map** — IB latent dims are the per-dim sensitivity primitive;
   register `sensitivity_map.c6_mdl_ibps_v1` (planned post-Stage-1).
2. **Pareto constraint** — `(rate_latent ≤ 5 KB) ∩ (rate_encoder ≤ 60 KB) ∩
   (rate_decoder ≤ 60 KB) ∩ (S(θ) ≥ 0.10 [HNeRV-class floor; Z1 anchor])`;
   register `tac.pareto.mdl_ibps_v1`.
3. **Bit-allocator** — `β` IS the bit-allocator knob (Tishby-Zaslavsky
   Lagrangian); register `bit_allocator.ib_beta_aware_v1`.
4. **Cathedral autopilot dispatch hook** — recipe registered; gated by
   Catalog #167 smoke-before-full.
5. **Continual-learning posterior** — Z1 ablation on C6 archive seeds the
   posterior with the FIRST across-class anchor; expected MDL density < 0.90
   would confirm the substrate-class shift.
6. **Probe-disambiguator** — `β`-sweep IS the probe; see above.

Cross-references
----------------

- Master roadmap: `.omx/research/long_term_multi_year_campaign_roadmap_20260514.md` (C6 EV/$ #1)
- Campaign ledger: `.omx/research/campaign_lane_c6_e4_mdl_ibps_substrate_20260514.md`
- Floor v3: `.omx/research/adjusted_theoretical_floor_v3_post_pr106_falsification_20260513.md`
- Z1 ablation: `.omx/research/zen_floor_band_v2_post_z1_ablation_20260514.md`
- D4 sister substrate: `tac.substrates.d4_wyner_ziv_frame_0` (canonical L1 pattern)
- Tishby & Zaslavsky 2015 "Deep Learning and the Information Bottleneck Principle"
- Rissanen 1978 "Modeling by shortest data description"
- MacKay 2003 *Information Theory, Inference, and Learning Algorithms* chs. 28-30

Lane: ``lane_c6_e4_mdl_ibps_substrate_20260514``
"""

from tac.substrates.c6_e4_mdl_ibps.architecture import (
    EVAL_HW,
    NUM_PAIRS,
    TOTAL_ARCHIVE_TARGET_BYTES_MAX,
    TOTAL_ARCHIVE_TARGET_BYTES_MIN,
    MDLIBPSConfig,
    MDLIBPSSubstrate,
)
from tac.substrates.c6_e4_mdl_ibps.archive import (
    IBPS1_HEADER_FMT,
    IBPS1_HEADER_SIZE,
    IBPS1_MAGIC,
    IBPS1_SCHEMA_VERSION,
    IBPS1_SECTION_ROLES,
    MDLIBPSArchive,
    pack_archive,
    parse_archive,
    parse_ibps1_archive_bytes,
)
from tac.substrates.c6_e4_mdl_ibps.ib_decoder import IBDecoder
from tac.substrates.c6_e4_mdl_ibps.ib_encoder import IBEncoder
from tac.substrates.c6_e4_mdl_ibps.mdl_loss import IBMDLLoss
from tac.substrates.c6_e4_mdl_ibps.score_aware_loss import (
    MDLIBPSLossWeights,
    MDLIBPSScoreAwareLoss,
)

__all__ = [
    "EVAL_HW",
    "IBDecoder",
    "IBEncoder",
    "IBMDLLoss",
    "IBPS1_HEADER_FMT",
    "IBPS1_HEADER_SIZE",
    "IBPS1_MAGIC",
    "IBPS1_SCHEMA_VERSION",
    "IBPS1_SECTION_ROLES",
    "MDLIBPSArchive",
    "MDLIBPSConfig",
    "MDLIBPSLossWeights",
    "MDLIBPSScoreAwareLoss",
    "MDLIBPSSubstrate",
    "NUM_PAIRS",
    "TOTAL_ARCHIVE_TARGET_BYTES_MAX",
    "TOTAL_ARCHIVE_TARGET_BYTES_MIN",
    "pack_archive",
    "parse_archive",
    "parse_ibps1_archive_bytes",
]
