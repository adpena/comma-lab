# SPDX-License-Identifier: MIT
"""tac.substrates.rudin_floor_interpretable_ml — Rudin floor compositional decoder.

Per ``.omx/research/rudin_floor_interpretable_ml_substrate_asymptotic_pursuit_scoping_design_20260516.md``
(L0 SKETCH design memo, ratified into L1 SCAFFOLD by subagent D per the
2026-05-16 omnibus dispatch + the Rudin floor memo op-routable #2).

The substrate is a TRIPLE CLASS-SHIFT per the abandon-within-class taxonomy:
(1) **architecture class** — NO neural network at inflate; pure rule application;
structurally distinct from every renderer / codec / latent-stream substrate in
the portfolio; (2) **decode-time contract** — zero PyTorch; ≤200 LOC pure Python;
CPU-only inflate by construction; (3) **scorer-relationship** — the Rashomon
ensemble preserves K=8 epistemic-diverse interpretations of the scorer's
evaluation, structurally distinct from canonical eval_roundtrip + score-aware
loss patterns.

The substrate IS the META-layer Rudin-Daubechies autopilot (Catalogs #273-#278;
`src/tac/autopilot_rudin_daubechies/`) RE-ARCHITECTED as the decode-time
architecture. The encoder is a GOSDT-compiled sparse decision tree (Lin-Zhong-
Hu-Hu-Rudin-Seltzer 2020); the decoder is a Wang-Rudin 2015 falling-rule-list
with K=4-6 rules; the loss is the Rashomon ensemble (K=8 SLIM-scored rule
lists per Semenova-Rudin-Parr 2020); the archive is the RDIF (Rudin Decoder
Interchange Format) monolithic 0.bin with SLIM-coded integer coefficients +
per-rule conditions + falling-rule-list ordering.

Canonical-vs-unique decision per layer (per CLAUDE.md
``UNIQUE-AND-COMPLETE-PER-METHOD operating mode`` + Catalog #290 +
design memo §15):

============================================  ===================  ==========================================
Layer                                         Decision             Rationale
============================================  ===================  ==========================================
Encoder GOSDT decision tree                   ADOPT canonical      ``tac.autopilot_rudin_daubechies.gosdt_dispatcher`` (Catalog #278); depth ≤ 4 + leaves ≤ 16 fits exactly
Falling-rule-list decoder                     ADOPT canonical      ``tac.autopilot_rudin_daubechies.falling_rule_list`` (Catalog #274); first-match-wins semantics
SLIM integer-coefficient solver               ADOPT canonical      ``tac.autopilot_rudin_daubechies.slim_ranker`` (Catalog #273); greedy + coordinate descent
Rashomon ensemble bootstrap                   ADOPT canonical      ``tac.autopilot_rudin_daubechies.rashomon_ensemble`` (Catalog #275); K=8 bootstrap
Compressive sensing measurement               ADOPT canonical      ``tac.autopilot_rudin_daubechies.compressive_landscape`` (Catalog #276); K=O(√N)
Wavelet multi-scale rule ranker               ADOPT canonical      ``tac.autopilot_rudin_daubechies.wavelet_multi_scale_ranker`` (Catalog #277); coarse-gates-fine
Predicate vocabulary (substrate-specific)     FORK                 No canonical helper exists; substrate-specific tokens unique to the rendering decoder
RDIF v1 archive grammar                       FORK                 No canonical archive grammar fits rule-list semantics
Per-pair pose-rule application                FORK                 Pose is not a class-classification problem; SegNet-argmax pattern doesn't fit
``gate_auth_eval_call``                       ADOPT canonical      Catalog #226 hard-earned auth-eval CLI flag stability
``posterior_update_locked``                   ADOPT canonical      Catalogs #128/#131 hard-earned fcntl-locked discipline
``subagent_commit_serializer``                ADOPT canonical      Catalogs #117/#157/#174 hard-earned commit-swap protection
EMA decay 0.997                               N/A                  No neural weights to EMA-average
eval_roundtrip                                N/A                  Closed-form; uint8 roundtrip captured in rule-list compilation
``detect_hardware_substrate``                 ADOPT canonical      Catalog #190 hard-earned hardware-substrate routing
============================================  ===================  ==========================================

Catalog #124 archive-grammar 8 fields (declared inline so AST walker observes them)
-----------------------------------------------------------------------------------

- ``archive_grammar``: RDIF v1 monolithic single-file ``0.bin`` per HNeRV
  parity L3 (fixed offsets per ``RDIFv1Codec``)
- ``parser_section_manifest``: header (34 bytes) + encoder_tree_blob +
  rule_list_blob + scorer_priors_blob + frame_0_init_blob +
  wavelet_residuals_blob + pose_residuals_blob + per_pair_rule_indices_blob
  + rashomon_disagreement_blob + archive_sha256 trailer
- ``inflate_runtime_loc_budget``: ≤200 LOC pure Python (HNeRV L4
  substrate_engineering exception; ``numpy`` + ``PIL`` + stdlib only)
- ``runtime_dep_closure``: numpy + Pillow + standard library only (NO PyTorch)
- ``export_format``: custom (RDIF v1; substrate-specific)
- ``score_aware_loss``: custom (Rashomon ensemble consensus; NOT a
  PyTorch backprop loss — combinatorial over SLIM integer-coefficient space)
- ``bolt_on_loc_budget``: substrate_engineering exception per HNeRV L7
  (substrate engineering exceeds bolt-on budget by construction)
- ``no_op_detector_planned``: Catalog #272 distinguishing-feature byte-mutation
  smoke (sister ``tools/verify_distinguishing_feature_byte_mutation.py``);
  the rule-application IS the byte-consumption proof

target_modes: ``research_substrate``
lane_class: ``substrate_engineering``
research_only: false (Phase 1b L2 integration implements trainer _full_main
and the operator recipe is dispatch-enabled for claimed smoke/exact-eval runs;
promotion still requires paired CPU/CUDA evidence, lane claim custody, and
exact-eval artifacts per Catalog #240 discipline)
horizon_class: ``asymptotic_pursuit`` (per T4 SYMPOSIUM 4×4 floor matrix)

6-hook wire-in (Catalog #125 NON-NEGOTIABLE)
-------------------------------------------

1. **Sensitivity-map** — K=8 Rashomon disagreement queue contributes
   per-substrate / per-rule / per-pixel sensitivity signals.
2. **Pareto constraint** — adds ``interpretability_tax_lower_bound = 0.03 ×
   Shannon_floor`` per Rudin canonical (interpretability tax bound).
3. **Bit-allocator** — per-rule byte budget (≤200 bytes per rule × 6 rules
   ≤ 1.2KB rule-list bytes) registers per-rule importance.
4. **Cathedral autopilot dispatch hook** — first asymptotic-pursuit class
   registered in the autopilot; predicted-band [0.150, 0.180] Mid; cost-band
   $3-15 smoke per design memo §18.
5. **Continual-learning posterior** — empirical anchors flow through
   ``tac.continual_learning.posterior_update_locked`` per Catalogs #128/#131;
   Rashomon K=8 members also update via canonical
   ``rashomon_ensemble.RashomonEnsembleRanker.update_all(..., store_path=...)``
   per Catalog #252.
6. **Probe-disambiguator** — K=8 Rashomon disagreement queue IS the
   canonical probe-disambiguator; sister
   ``tools/probe_rudin_floor_substrate_disambiguator.py`` consumes the
   substrate at design-time (no GPU spend).

Observability surface (per CLAUDE.md "Observability surface" + Catalog #305)
----------------------------------------------------------------------------

1. **Per-layer inspection.** Per-rule hit-rate via
   ``parsed.rashomon_disagreement`` carrying K=8 σ per rule.
2. **Per-signal decomposition.** Per-rule seg/pose/rate contributions
   decomposable via ``rule_list_blob[rule_idx]`` byte attribution.
3. **Run-to-run diff.** Byte-deterministic compress (seed-pinned;
   sorted-keys JSON meta) ⇒ identical ``0.bin`` sha256 across runs.
4. **Post-hoc query interface.** ``rdif_compress_observability.jsonl``
   (per-pair row) + ``rdif_inflate_observability.jsonl`` (per-frame row).
5. **Cite-chain.** Per Catalog #245 modal_call_id_ledger:
   ``(substrate_id='rudin_floor_interpretable_ml', commit_sha,
   modal_call_id, recipe_sha, trainer_sha, upstream_snapshot_sha256,
   seed, K_rules, K_rashomon, slim_coeff_bound, gosdt_depth)``.
6. **Counterfactual hooks.** Per Catalog #272 byte-mutation discipline:
   the sister ``tools/verify_distinguishing_feature_byte_mutation.py`` can
   mutate one byte in ``rule_list_blob`` and observe the inflate output
   change. Every byte is traceable to a rule.

Public API (narrow per CLAUDE.md "Beauty, simplicity, and developer experience")
-------------------------------------------------------------------------------

* :class:`RDIFv1Header` — fixed 34-byte header parser/builder
* :class:`RudinFallingRule` — one rule (predicate + action + SLIM coefficients)
* :class:`RudinRuleList` — K=4-6 rules; first-match-wins semantics
* :func:`pack_archive` — closed-form RDIF v1 packer (compress-side)
* :func:`parse_archive` — RDIF v1 parser (inflate-side; pure stdlib)
* :func:`inflate_one_video` — per-frame PNG emission via rule application
* :data:`RDIF_MAGIC` — ``b"RDF1"`` magic constant
* :data:`RDIF_VERSION` — uint16 ``0x0001``
* :data:`RDIF_HEADER_SIZE` — 34 bytes fixed
* :data:`CANONICAL_K_RULES` — 6 (Wang-Rudin 2015 canonical falling-rule depth)
* :data:`CANONICAL_K_RASHOMON` — 8 (Semenova-Rudin-Parr 2020 canonical ensemble)
* :data:`CANONICAL_SLIM_COEFF_BOUND` — 10 (Ustun-Rudin 2016 canonical integer bound)
* :data:`CANONICAL_GOSDT_DEPTH` — 4 (Lin-Zhong-Hu-Hu-Rudin-Seltzer 2020 canonical)
"""

from __future__ import annotations

from tac.substrates.rudin_floor_interpretable_ml.archive import (
    CANONICAL_GOSDT_DEPTH,
    CANONICAL_K_RASHOMON,
    CANONICAL_K_RULES,
    CANONICAL_SLIM_COEFF_BOUND,
    RDIF_HEADER_SIZE,
    RDIF_MAGIC,
    RDIF_VERSION,
    RDIFv1Archive,
    RDIFv1Header,
    pack_archive,
    parse_archive,
)
from tac.substrates.rudin_floor_interpretable_ml.inflate import inflate_one_video
from tac.substrates.rudin_floor_interpretable_ml.rule_list import (
    RudinFallingRule,
    RudinRuleList,
)

# Phase 1b L2 integration: archive grammar + inflate runtime + rule_list +
# trainer _full_main IMPLEMENTED and recipe dispatch-enabled. Score promotion
# still requires paired exact-eval evidence.
IMPLEMENTATION_STATUS = (
    "phase_1b_l2_dispatch_enabled_full_main_integrated_archive_inflate_rule_list"
)
RESEARCH_ONLY = False

PLANNED_PUBLIC_API = (
    "RDIFv1Archive",
    "RDIFv1Header",
    "RudinFallingRule",
    "RudinRuleList",
    "pack_archive",
    "parse_archive",
    "inflate_one_video",
)

__all__ = [
    "CANONICAL_GOSDT_DEPTH",
    "CANONICAL_K_RASHOMON",
    "CANONICAL_K_RULES",
    "CANONICAL_SLIM_COEFF_BOUND",
    "IMPLEMENTATION_STATUS",
    "PLANNED_PUBLIC_API",
    "RDIF_HEADER_SIZE",
    "RDIF_MAGIC",
    "RDIF_VERSION",
    "RESEARCH_ONLY",
    "RDIFv1Archive",
    "RDIFv1Header",
    "RudinFallingRule",
    "RudinRuleList",
    "inflate_one_video",
    "pack_archive",
    "parse_archive",
]
