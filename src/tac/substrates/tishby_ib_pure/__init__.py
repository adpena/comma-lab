# SPDX-License-Identifier: MIT
"""tac.substrates.tishby_ib_pure — Tishby IB-pure substrate (full Information Bottleneck Lagrangian).

Per the 2026-05-16 design memo
``.omx/research/tishby_ib_pure_substrate_asymptotic_pursuit_scoping_design_20260516.md``.
PRIMARY architecture (NOT bolt-on like Z4 / ATW v1 / ATW v2): the entire
codec IS the variational IB Lagrangian + Atick-Redlich cooperative-receiver
framing + Wyner-Ziv side-information construction operationalized as a single
coherent substrate.

L1 SCAFFOLD landing per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or
RESEARCH-ONLY" (Catalog #240):
  - trainer ``_full_main`` raises NotImplementedError (Phase 2 council approval
    required to lift; gated on D4 probe MEANINGFUL_CONDITIONING + VIB
    tractability TRACTABLE per design memo §19)
  - recipe carries ``research_only: true`` + ``dispatch_enabled: false``
  - registered_substrate.py declares ``recipe_research_only=True`` per the
    META layer canonical contract (Catalog #241/#242)

Empirical anchors at landing:
  - **VIB tractability probe**: TRACTABLE (gradient SNR ≈ 6.75) on synthetic
    Gaussian smoke per ``tools/check_variational_ib_tractability.py``
  - **D4 H(latent|scorer_class) probe** on A1 latents: INDEPENDENT
    (MI ≈ 0.006 bits/symbol; degenerate single-class SegNet output on
    dashcam footage). Per CLAUDE.md "Forbidden premature KILL" + design
    memo §19: this is DEFER-pending-research, NOT KILLED. Reactivation
    criteria: (a) re-run probe with per-pair multi-class signature beyond
    composite-majority (e.g. spatial-bin class proportions); (b) train
    Tishby IB-pure with SCORER-CONDITIONAL CDF range coding to see if the
    SUBSTRATE'S OWN encoder produces a non-degenerate latent-class
    distribution (the A1 substrate's latents may simply not exhibit the
    class-conditional structure the IB Lagrangian would learn de novo);
    (c) operator-approved Phase 2 council deliberation per design memo §19.2.

Two operationalization paths per §4:
  - **Path-VIB** (Alemi 2017 canonical reparam-trick): DEFAULT at v1
    SCAFFOLD; tractability empirically validated by ``check_variational_ib_tractability.py``.
  - **Path-MINE** (Belghazi 2018 MINE statistic-network): v2 fallback if
    Path-VIB tractability fails on real-scorer Modal A100 smoke.

Distinguishing-feature (per Catalog #272):
  - ``distinguishing_feature_name``: variational_ib_encoder_with_wyner_ziv_side_info_scorer_class_conditional_cdf_range_coding
  - ``distinguishing_bytes_path``: DECODER_BLOB + LATENT_T_BLOB + SCORER_CLASS_PRIOR_BLOB + CDF_TABLE_BLOB

Catalog #124 archive-grammar 8 fields (declared inline so AST walker observes them)
-----------------------------------------------------------------------------------

* ``archive_grammar``: TIBP1 monolithic single-file ``0.bin`` with variational
  encoder (optional) + variational decoder + int8 latent table + scorer-class
  prior side-info + scorer-conditional fp16 CDF table; bytewise-distinct
  from ATW1/ATW2/Z3HV2/A1 (magic ``b"TIBP"``).
* ``parser_section_manifest``: TIBP1 header + encoder_blob (optional) +
  decoder_blob + statistic_net_blob (Path-MINE only) + latent_t_blob +
  scorer_class_prior_blob + cdf_table_blob + meta_blob (8 sections).
* ``inflate_runtime_loc_budget``: <=200 LOC substrate-engineering waiver per
  HNeRV parity L4 + L7 (variational decoder + WZ side-info offset + range
  decode + per-pair render loop).
* ``runtime_dep_closure``: torch + brotli + numpy (HNeRV L4 <=3 deps; numpy
  for range coding lookups).
* ``export_format``: TIBP1 monolithic single-zip-member ``0.bin``.
* ``score_aware_loss``: ``TishbyIBPureScoreAwareLoss`` routes through canonical
  ``tac.codec.cooperative_receiver.atick_redlich.cooperative_receiver_loss``
  per Catalog #164 for the reconstruction term + closed-form KL.
* ``bolt_on_loc_budget``: ``lane_class=substrate_engineering`` (HNeRV L7);
  ~400-550 LOC scaffold (Path-VIB) or ~550-700 LOC (Path-MINE).
* ``no_op_detector_planned``: variational decoder + WZ side-info offset +
  scorer-conditional CDF range coding are structurally consumed at inflate;
  empirical byte-mutation smoke per Catalog #220 + #272 verifies non-trivial
  contribution.

target_modes: ``research_substrate``
lane_class: ``substrate_engineering``
research_only: true (Phase 2 council approval required to lift _full_main
NotImplementedError per CLAUDE.md "Substrate scaffolds MUST be COMPLETE
or RESEARCH-ONLY" + the D4 probe INDEPENDENT verdict at landing per
``.omx/state/h_latent_given_scorer_class_tishby_ib_pure.json``)
canary_status: ``independent_substrate`` (no canary dependency at v1 since
D4 probe + VIB-tractability check ARE the pre-dispatch gates)

Observability surface
---------------------

Per the MAX-OBSERVABILITY-INTO-BEHAVIOR standing directive 2026-05-16 +
Catalog #305. The Tishby IB-pure substrate exposes:

1. **Per-layer inspection**: trainer emits ``stage_log`` per provenance schema;
   each forward pass logs (mu norm, log_sigma mean, KL per-dim, reconstruction
   per-pair) to provenance JSON.
2. **Per-signal decomposition**: L_VIB decomposes into
   (reconstruction_term, KL_term, rate_term) per Tishby IB Lagrangian; each
   serialized per-step + per-pair for empirical R(D) curve construction.
3. **Run-to-run diff**: TIBP1 archive byte-identical reproducible under
   ``(seed, commit_sha, upstream_snapshot_sha256, beta_value)`` tuple per
   Catalog #166 Modal HEAD-parity ledger + Catalog #245 modal_call_id_ledger.
4. **Post-hoc query**: ``experiments/results/lane_tishby_ib_pure_*/`` carries
   contest_auth_eval_<axis>.json + modal_metadata.json + observability/*.jsonl.
5. **D4 probe verdict** at ``.omx/state/h_latent_given_scorer_class_tishby_ib_pure.json``
   (LANDED at L1 scaffold: INDEPENDENT, MI ≈ 0.006 bits/symbol; reactivation
   criteria per the §19.1 V1 lift gate).
6. **VIB tractability verdict** at ``.omx/state/variational_ib_tractability_tishby_ib_pure.json``
   (TRACTABLE at synthetic-data smoke; council-grade Modal A100 100ep proxy
   pending per design memo §19.1 V1 lift gate criterion #2).

Probe-disambiguator (Catalog #125 hook #6)
-------------------------------------------

Three layers per design memo §19.3:

1. **D4 probe** ($3-5 CPU; LANDED): disambiguates Wyner-Ziv side-info hypothesis.
2. **VIB tractability check** ($0 CPU smoke / $5-10 Modal A100 council-grade):
   disambiguates Path-VIB vs Path-MINE choice via gradient SNR measurement.
3. **β-sweep regime arbitration** ($20-40 paired; 4× Modal A100 100ep at
   β ∈ {0.001, 0.01, 0.1, 1.0}): empirically produces R(D) curve; council
   picks operating point.

6-hook wire-in (Catalog #125 NON-NEGOTIABLE)
---------------------------------------------

1. **Sensitivity-map**: ``tac.sensitivity_map.variational_ib_per_pair``
   (planned; consumes KL term per-pair gradient norm).
2. **Pareto constraint**: ``tac.pareto.tishby_ib_pure_lagrangian``
   (``I(X;T) - β·I(T;Y) <= K_IB(β)`` constraint).
3. **Bit-allocator hook**: ``bit_allocator.tishby_ib_pure_variational_v1``
   (per-pair archive bytes by KL term per-pair contribution).
4. **Cathedral autopilot dispatch hook**: recipe registered warn-only at
   landing per Catalog #167; promotes to dispatch-eligible upon D4
   MEANINGFUL_CONDITIONING + VIB-tractability TRACTABLE on real scorer.
5. **Continual-learning posterior update**: full anchor seeds posterior
   paired with D4 MI value + VIB tractability SNR per Catalog #128 locked
   write.
6. **Probe-disambiguator**: D4 + VIB-tractability + β-sweep per §19.3.

Cross-references
----------------

* ``.omx/research/tishby_ib_pure_substrate_asymptotic_pursuit_scoping_design_20260516.md``
  (PRIMARY design memo)
* ``.omx/research/atw_codec_v2_cooperative_receiver_full_stack_design_20260516.md``
  (sister BOLT-ON variant)
* ``src/tac/substrates/atw_codec_v2/`` (sister substrate package — V2 BOLT-ON
  with G1 + B3 + WZ side-info head)
* ``src/tac/substrates/wyner_ziv_cooperative_receiver/`` (sister Wyner-Ziv
  DISCUS substrate)
* ``src/tac/substrates/z4_cooperative_receiver_loss/`` (Z4 β-only branch sister)
* ``src/tac/codec/cooperative_receiver/atick_redlich.py`` (canonical primitive
  the reconstruction term routes through)
* ``tools/check_variational_ib_tractability.py`` (NEW canonical probe per
  design memo §22 op-routable #3 — LANDED at L1)
* ``tools/probe_latent_conditional_entropy_h_latent_given_scorer_class.py``
  (D4 probe; commit d72f50985)
* ``experiments/results/tishby_ib_pure_d4_probe_20260516T212557Z/`` (D4 probe
  execution artifacts: per-pair SegNet class JSON + verdict JSON)

Lane: ``lane_tishby_ib_pure_l1_scaffold_substrate_build_plus_probes_20260516``
"""

from __future__ import annotations

from tac.substrates.tishby_ib_pure.architecture import (
    DEFAULT_BETA,
    DEFAULT_LATENT_DIM,
    EVAL_HW,
    NUM_PAIRS,
    NUM_SEGNET_CLASSES,
    TishbyIBPureCodec,
    TishbyIBPureCodecConfig,
    TishbyIBPurePathVariant,
)
from tac.substrates.tishby_ib_pure.archive import (
    TIBP1_HEADER_FMT,
    TIBP1_HEADER_SIZE,
    TIBP1_MAGIC,
    TIBP1_SCHEMA_VERSION,
    TIBP1_SECTION_ROLES,
    TishbyIBPureArchive,
    pack_archive,
    parse_archive,
    parse_tibp1_archive_bytes,
)
from tac.substrates.tishby_ib_pure.inflate import inflate_one_video, main_cli
from tac.substrates.tishby_ib_pure.score_aware_loss import (
    TishbyIBPureLossOutput,
    TishbyIBPureLossWeights,
    TishbyIBPureScoreAwareLoss,
)

LANE_ID = "lane_tishby_ib_pure_l1_scaffold_substrate_build_plus_probes_20260516"
DESIGN_MEMO_PATH = (
    ".omx/research/tishby_ib_pure_substrate_asymptotic_pursuit_scoping_design_20260516.md"
)
RESEARCH_ONLY = True
IMPLEMENTATION_STATUS = (
    "l1_scaffold_architecture_archive_inflate_loss_modules_research_only_pending_"
    "phase_2_council_lift_per_d4_independent_verdict_reactivation"
)

# Empirical anchors at L1 SCAFFOLD landing (per .omx/state/ JSON artifacts)
D4_PROBE_VERDICT = "INDEPENDENT"
D4_PROBE_MUTUAL_INFORMATION_BITS = 0.0064
VIB_TRACTABILITY_VERDICT = "TRACTABLE"
VIB_TRACTABILITY_SNR_MEAN = 6.75

__all__ = [
    "D4_PROBE_MUTUAL_INFORMATION_BITS",
    "D4_PROBE_VERDICT",
    "DEFAULT_BETA",
    "DEFAULT_LATENT_DIM",
    "DESIGN_MEMO_PATH",
    "EVAL_HW",
    "IMPLEMENTATION_STATUS",
    "LANE_ID",
    "NUM_PAIRS",
    "NUM_SEGNET_CLASSES",
    "RESEARCH_ONLY",
    "TIBP1_HEADER_FMT",
    "TIBP1_HEADER_SIZE",
    "TIBP1_MAGIC",
    "TIBP1_SCHEMA_VERSION",
    "TIBP1_SECTION_ROLES",
    "VIB_TRACTABILITY_SNR_MEAN",
    "VIB_TRACTABILITY_VERDICT",
    "TishbyIBPureArchive",
    "TishbyIBPureCodec",
    "TishbyIBPureCodecConfig",
    "TishbyIBPureLossOutput",
    "TishbyIBPureLossWeights",
    "TishbyIBPurePathVariant",
    "TishbyIBPureScoreAwareLoss",
    "inflate_one_video",
    "main_cli",
    "pack_archive",
    "parse_archive",
    "parse_tibp1_archive_bytes",
]
