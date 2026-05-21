# SPDX-License-Identifier: MIT
"""tac.substrates.atw_codec_v2 — ATW Codec V2 (Atick-Tishby-Wyner full-stack cooperative-receiver codec).

Per the 2026-05-16 ATW v2 design memo at
``.omx/research/atw_codec_v2_cooperative_receiver_full_stack_design_20260516.md``
(commit ``fcdcc1112``). V2 is the productionized successor of the V1 SCAFFOLD
that lifts ``_full_main NotImplementedError`` per the Phase 2 council gate AND
adds three structural primitives (G1 + B3 + Wyner-Ziv side-info head closed-form)
that the V1 design memo named as reactivation criteria.

V2 binds Atick-Redlich + Tishby IB + Wyner-Ziv into ONE substrate-engineering
scaffold whose distinguishing-feature (per Catalog #272) is the
**scorer-class-conditional latent residual codec** — the encoder uses the
cooperative receiver's class assignments as side-information that the decoder
reconstructs WITHOUT loading the scorer at inflate time.

Two variants per design memo §4:

* **Variant A** (three-knob κ_IB / λ_WZ / λ_pixel): preserves V1 probe-
  disambiguator regime sweep. ~450-600 LOC bolt-on. Selected when D4 probe
  verdict is ``MEANINGFUL_CONDITIONING`` AND council adjudication requires
  knob-zero regime sweep arbitration.
* **Variant B** (single-knob WZ-only): UNIQUE-AND-COMPLETE substrate-optimal
  engineering per the operator standing directive
  ``feedback_pr95_lesson_now_at_meta_level_unique_and_complete_per_method_default_20260515.md``.
  Drops Tishby IB regularizer + Z3 pixel-MSE residual; binds ONLY the
  Wyner-Ziv side-info residual mechanism. ~250-350 LOC bolt-on. **DEFAULT.**

The V2 distinguishing-feature triple per Wunderkind candidates
(``feedback_wunderkind_visionary_scorer_as_cooperative_receiver_paradigm_shift_20260515.md``):

* **G1** scorer-class distill head: 1KB MLP ``g(decoded_latent_per_pixel) ->
  5-way SegNet softmax``. Replaces Ballé-style 50KB hyperprior. ΔS prediction
  ``-0.005 to -0.015`` rate-axis per design memo §3 [prediction;
  first-principles-bound; D4-probe-conditional].
* **B3** scorer-conditional CDF table: range-encode latents conditional on
  scorer's softmax distribution per pixel. CDF table precomputed at compress,
  shipped ~2KB side-info. ΔS prediction ``-0.003 to -0.010`` rate-axis
  [prediction; first-principles-bound].
* **G2-PARTIAL** posterior-matching codec: stateless decoder via scorer-
  conditional Langevin sampler. Deferred to V3 per design memo §4.

The V2 Lagrangian (Variant B, default):

::

    L_ATW_v2_B = alpha * B(theta)/N           (rate from archive bytes)
               + beta_seg * d_seg(theta)      (Atick-Redlich SegNet term)
               + gamma_pose * sqrt(d_pose)    (Atick-Redlich PoseNet term)
               + lambda_WZ * R_WZ(z | s)      (Wyner-Ziv residual term)

Variant A additionally carries ``kappa_IB * I(T; Y_predicted)`` and
``lambda_pixel * MSE(decoded, GT)`` terms; the four-corner regime sweep
``(kappa_IB, lambda_WZ, lambda_pixel) in {(0,0,0), (0,1,0), (0.1,0,0), (0,0,1)}``
recovers Atick-only / ATW canonical / Tishby IB pure / Z3 baseline.

Catalog #124 archive-grammar 8 fields (declared inline so AST walker observes them)
-----------------------------------------------------------------------------------

* ``archive_grammar``: ATW2 monolithic single-file ``0.bin`` with WZ side-info
  head + G1 distill head + B3 scorer-conditional CDF table; bytewise-distinct
  from ATW1 (different magic ``b"ATW2"``) and from sister Z3HP1/Z4CR1.
* ``parser_section_manifest``: ATW2 header + encoder_blob + decoder_blob +
  wz_head_blob + distill_head_blob + latent_residual_blob + class_prior_table_blob
  + cdf_table_blob + meta_blob (9 sections; +2 vs ATW1).
* ``inflate_runtime_loc_budget``: <=200 LOC substrate-engineering waiver per
  HNeRV parity L4 + L7 (decoder + WZ head + distill head + range-decode +
  composition).
* ``runtime_dep_closure``: torch + brotli + numpy (HNeRV L4 <=3 deps; numpy
  for range coding lookups).
* ``export_format``: ATW2 monolithic single-zip-member ``0.bin``.
* ``score_aware_loss``: ``ATWv2ScoreAwareLoss`` routes through canonical
  ``tac.codec.cooperative_receiver.atick_redlich.cooperative_receiver_loss``
  per Catalog #164 + Wunderkind E1 substitution recommendation.
* ``bolt_on_loc_budget``: ``lane_class=substrate_engineering`` (HNeRV L7);
  ~450-600 LOC scaffold (Variant A) or ~250-350 LOC (Variant B).
* ``no_op_detector_planned``: WZ side-info head + G1 distill head + B3 CDF table
  are structurally consumed at inflate; empirical byte-mutation smoke per
  Catalog #220 + #272 verifies non-trivial contribution.

target_modes: ``research_substrate``
lane_class: ``substrate_engineering``
research_only: true (Phase 2 council approval required to lift _full_main
NotImplementedError per CLAUDE.md "Substrate scaffolds MUST be COMPLETE
or RESEARCH-ONLY". The 2026-05-16 D4 probe on A1 latents returned
``INDEPENDENT`` at ``I(latent; scorer_class)=0.006385502752`` bits/symbol, so
the measured A1-latent/class-conditioning surface is deferred and ATW v2 must
remain research-only unless a richer side-information reactivation probe lands.)
canary_status: ``post_canary_dependent``
canary_dependency: ``lane_atw_codec_design_v1_20260515``

Observability surface
---------------------

Per the MAX-OBSERVABILITY-INTO-BEHAVIOR standing directive 2026-05-16 + Catalog
#305. The V2 substrate exposes the following observability surfaces:

1. **Per-layer inspection**: trainer emits ``stage_log`` entries per provenance
   schema; each forward pass logs (z_residual norm, z_predicted norm, WZ residual
   ratio, per-class CDF entropy) to provenance JSON.
2. **Per-signal decomposition**: ATW Lagrangian decomposes into
   (rate, seg, pose, wz_residual, [ib, pixel for Variant A]) terms; each
   serialized per-step.
3. **Run-to-run diff**: ATW2 archive byte-identical reproducible under
   ``(seed, commit_sha, upstream_snapshot_sha256)`` tuple per Catalog #166.
4. **Post-hoc query**: ``experiments/results/lane_atw_codec_v2_*/`` carries
   contest_auth_eval_<axis>.json + modal_metadata.json + observability/*.jsonl.
5. **D4 probe verdict** at
   ``.omx/research/atw_codec_v2_d4_probe_verdict_20260516_codex.json``
   (INDEPENDENT on A1 latents; PRE-DISPATCH GATE per design memo §19).

Probe-disambiguator (Catalog #125 hook #6)
------------------------------------------

Three layers per design memo §19:

1. **D4 probe** ($0 CPU reuse of sister class artifact; LANDED): disambiguates
   Wyner-Ziv hypothesis directly. Current verdict: ``INDEPENDENT`` on A1
   latents, therefore no ATW v2 Phase-2 dispatch authority from this signal.
2. **Three-knob regime sweep** (Variant A only; $30 paired): four-corner
   ablations Atick-only / ATW canonical / Tishby IB pure / Z3 baseline.
3. **Variant A vs B paired smoke** ($10-20): if D4 = MEANINGFUL and Council
   adjudication is split, paired smoke arbitrates Variant choice empirically.

6-hook wire-in (Catalog #125 NON-NEGOTIABLE)
-------------------------------------------

1. **Sensitivity-map**: ``tac.sensitivity_map.scorer_class_conditional_v2``
   (planned; consumes D4 MI gradient).
2. **Pareto constraint**: ``tac.pareto.atw_v2_wz_residual_entropy``
   (R_WZ >= H(latent | scorer_class) constraint).
3. **Bit-allocator hook**: ``bit_allocator.atw_v2_wz_residual_v1``
   (per-pair archive bytes by scorer-class entropy).
4. **Cathedral autopilot dispatch hook**: recipe registered warn-only at
   landing per Catalog #167; promotes to dispatch-eligible upon D4
   MEANINGFUL_CONDITIONING.
5. **Continual-learning posterior update**: full anchor seeds posterior
   paired with D4 MI value per Catalog #128 locked write.
6. **Probe-disambiguator**: see above.

Cross-references
----------------

* `.omx/research/atw_codec_v2_cooperative_receiver_full_stack_design_20260516.md`
  (V2 design memo; commit fcdcc1112)
* `.omx/research/atw_codec_atick_tishby_wyner_v1_design_20260515.md` (V1 design memo)
* `.omx/research/atw_codec_v1_cargo_cult_unwind_design_20260516.md` (V1 unwind)
* ``feedback_wunderkind_visionary_scorer_as_cooperative_receiver_paradigm_shift_20260515.md``
  (G1 + B3 + G2-PARTIAL substitution candidates)
* ``src/tac/codec/cooperative_receiver/atick_redlich.py`` (canonical primitive)
* ``src/tac/substrates/atw_codec_v1/`` (V1 sister package; V2 is a NEW package,
  NOT a rename)
* ``tools/probe_latent_conditional_entropy_h_latent_given_scorer_class.py``
  (D4 probe; commit d72f50985)

Lane: ``lane_atw_codec_v2_substrate_build_20260516``
"""

from __future__ import annotations

from tac.optimization.atw_v2_phase2_gate import (
    ATW_V2_D4_VERDICT_ARTIFACT_PATH,
    ATW_V2_PHASE2_GATE_STATUS_SCHEMA,
    atw_v2_phase2_gate_status,
)
from tac.substrates.atw_codec_v2.architecture import (
    CDF_TABLE_NUM_SYMBOLS,
    DEFAULT_SCORER_CLASS_PRIOR_DIM,
    EVAL_HW,
    NUM_PAIRS,
    NUM_SEGNET_CLASSES,
    TOTAL_ARCHIVE_TARGET_BYTES_MAX,
    TOTAL_ARCHIVE_TARGET_BYTES_MIN,
    ATWv2Codec,
    ATWv2CodecConfig,
    ATWv2Variant,
)
from tac.substrates.atw_codec_v2.archive import (
    ATW2_HEADER_FMT,
    ATW2_HEADER_SIZE,
    ATW2_MAGIC,
    ATW2_SCHEMA_VERSION,
    ATW2_SECTION_ROLES,
    ATWv2CodecArchive,
    pack_archive,
    parse_archive,
    parse_atw2_archive_bytes,
)
from tac.substrates.atw_codec_v2.cdf_dead_section import (
    Atw2CdfDecodeInfluenceProof,
    Atw2CdfSectionAnalysis,
    analyze_atw2_cdf_section,
    mutate_atw2_cdf_table_bytes,
    prove_atw2_cdf_decode_influence,
)
from tac.substrates.atw_codec_v2.inflate import inflate_one_video, main_cli
from tac.substrates.atw_codec_v2.score_aware_loss import (
    ATWv2LossOutput,
    ATWv2LossWeights,
    ATWv2ScoreAwareLoss,
)

LANE_ID = "lane_atw_codec_v2_substrate_build_20260516"
DESIGN_MEMO_PATH = (
    ".omx/research/atw_codec_v2_cooperative_receiver_full_stack_design_20260516.md"
)
RESEARCH_ONLY = True
IMPLEMENTATION_STATUS = "l1_architecture_archive_inflate_and_loss_modules_available_research_only"
MISSING_IMPLEMENTATION_MODULES: tuple[str, ...] = ()
D4_PROBE_VERDICT = "INDEPENDENT"
D4_PROBE_MUTUAL_INFORMATION_BITS = 0.006385502752311645
D4_PROBE_PHASE2_STATUS = "defer_measured_a1_latent_class_conditioning_surface"
D4_PROBE_NEXT_ACTION = "do_not_dispatch_atw_v2_phase2_from_this_signal"

__all__ = [
    "ATW2_HEADER_FMT",
    "ATW2_HEADER_SIZE",
    "ATW2_MAGIC",
    "ATW2_SCHEMA_VERSION",
    "ATW2_SECTION_ROLES",
    "ATW_V2_D4_VERDICT_ARTIFACT_PATH",
    "ATW_V2_PHASE2_GATE_STATUS_SCHEMA",
    "CDF_TABLE_NUM_SYMBOLS",
    "D4_PROBE_MUTUAL_INFORMATION_BITS",
    "D4_PROBE_NEXT_ACTION",
    "D4_PROBE_PHASE2_STATUS",
    "D4_PROBE_VERDICT",
    "DEFAULT_SCORER_CLASS_PRIOR_DIM",
    "DESIGN_MEMO_PATH",
    "EVAL_HW",
    "IMPLEMENTATION_STATUS",
    "LANE_ID",
    "MISSING_IMPLEMENTATION_MODULES",
    "NUM_PAIRS",
    "NUM_SEGNET_CLASSES",
    "RESEARCH_ONLY",
    "TOTAL_ARCHIVE_TARGET_BYTES_MAX",
    "TOTAL_ARCHIVE_TARGET_BYTES_MIN",
    "Atw2CdfDecodeInfluenceProof",
    "Atw2CdfSectionAnalysis",
    "ATWv2Codec",
    "ATWv2CodecArchive",
    "ATWv2CodecConfig",
    "ATWv2LossOutput",
    "ATWv2LossWeights",
    "ATWv2ScoreAwareLoss",
    "ATWv2Variant",
    "atw_v2_phase2_gate_status",
    "analyze_atw2_cdf_section",
    "inflate_one_video",
    "main_cli",
    "mutate_atw2_cdf_table_bytes",
    "pack_archive",
    "parse_archive",
    "parse_atw2_archive_bytes",
    "prove_atw2_cdf_decode_influence",
]
