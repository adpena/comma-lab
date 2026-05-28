# SPDX-License-Identifier: MIT
"""Wyner-Ziv pipeline-stage codec L0 SCAFFOLD substrate package.

The Wyner-Ziv 1976 source-coding-with-side-information theorem applied at the
**pipeline-stage** surface: this substrate is a wrapper that takes any existing
substrate's pre-entropy byte stream, splits it via Y-derivable-prefix detection
against decoder-side side-info Y, and feeds the residual into the entropy
coder. The decoder reconstructs the original pre-entropy bytes from
``(main_compressed, side_compressed_baked, Y)``. The contest scorer is
decoder-deterministic (every Modal/Vast.ai/Lightning/Linux-x86_64-CPU runner
loads the same upstream/modules.py PoseNet + SegNet weights), so PoseNet
output IS canonical side-info Y at inflate time: encoded bytes can be
CONDITIONAL on PoseNet predictions WITHOUT including PoseNet weights in the
archive per CLAUDE.md "Strict scorer rule" non-negotiable.

This is the canonical Atick-Tishby-Wyner cooperative-receiver triple per
Catalog #311 grand council roster (Atick-Redlich + Tishby-Zaslavsky + Wyner
sister seats):

* **Atick-Redlich 1990 cooperative-receiver**: encoder optimizes loss against
  what the DECODER sees, not the source. Wyner-Ziv is the bytes-side analog.
* **Tishby-Zaslavsky 2015 information bottleneck**: encoder transmits I(X;T)
  bits, decoder consumes T as a sufficient-statistic side-info.
* **Wyner-Ziv 1976**: R(D|Y) achievable rate when decoder has side-info Y
  (vs R(D) without). At pipeline-stage primitive surface: bytes Y "covers"
  contribute zero archive cost.

L0 SCAFFOLD scope per HNeRV parity discipline L7 (substrate-engineering tier;
``lane_class=substrate_engineering``; ``research_only=true``;
``dispatch_enabled=false``). MLX-first training (M5 Max local; 8th MLX-FIRST
standing directive 2026-05-26) + numpy-portable inflate (no MLX dep at runtime
per HNeRV parity L4 ≤2 deps). Per CLAUDE.md "INDIVIDUALLY-FRACTAL" standing
directive 2026-05-27, every per-method engineering decision is Wyner-Ziv-
pipeline-stage-codec-OPTIMAL — NOT shared-helper shortcut from Z4 cooperative-
receiver primitive NOR from Z6-v2 cargo-cult-unwind NOR from D4 Wyner-Ziv
frame_0 (D4 derives frame_0 from frame_1 + parametric motion; this substrate
operates one abstraction level UP at the pipeline-stage byte-split surface).

Routing-through-canonical-primitive
-----------------------------------

This substrate is the substrate-scope CONSUMER of the canonical
``tac.codec.wyner_ziv_layer`` primitive (lane
``lane_wyner_ziv_pipeline_stage_codec_primitive_20260517`` at L1; 740 LOC +
64 tests; sister landing 2026-05-17). The primitive exposes the WZ stage as
a black-box codec; this substrate wires it into a complete substrate package
per the canonical 36-field ``SubstrateContract`` per Catalog #241/#242 +
contributes to the cathedral autopilot ranker via auto-discovery per
Catalog #335 + emits canonical Provenance per Catalog #323.

Distinction from sister Wyner-Ziv substrates:

* ``tac.substrates.d4_wyner_ziv_frame_0`` (lane 20260514) — derives frame_0
  from frame_1 + parametric motion + photometric residual. Operates at the
  FRAME-PAIR mathematical surface. Standalone substrate composing with an
  external base archive.
* ``tac.substrates.wyner_ziv_cooperative_receiver`` (lane 20260513) — DISCUS
  Slepian-Wolf coset binning of the source pair against SegNet+PoseNet as
  cooperative-receiver. Operates at the BIT-PLANE surface.
* ``tac.substrates.wyner_ziv_pipeline_stage_codec`` (THIS substrate; lane
  20260528) — pipeline-stage codec wrapper. Operates at the COMPRESSION-
  PIPELINE-STAGE byte-stream surface. Composes orthogonally with ANY existing
  substrate by sitting between its pre-entropy byte production and its
  entropy stage.

Predicted ΔS band derivation (per Catalog #296 Dykstra-feasibility + sister
primitive empirical anchors):

* The sister primitive's pre-entropy prober (Q1 sister; lane
  ``lane_wyner_ziv_deliverability_prober_20260517``) measured ratio 0.217-
  0.228 on raw fp16 state_dict bytes via lzma; translating to deliverable
  score-savings at the contest rate scale: ~0.47 score reduction per substrate
  IF Y-derivable-prefix overlap reaches the same regime.
* For THIS substrate at L0 SCAFFOLD predicted band is **operator-routable
  pending L1 paired smoke** because the routing-through-primitive composition
  is novel; the empirical anchor inherits from the primitive but the
  composition factor (Catalog #227 alpha) is unmeasured. Predicted band
  declared in design memo as ``[-0.0470, -0.0050]`` with explicit
  Dykstra-feasibility citation.

Six-hook wire-in declaration per Catalog #125:

* hook #1 sensitivity-map = ACTIVE (per-pair byte-coverage exposed via
  ``architecture.layerwise_inspector``; routes through
  ``tac.sensitivity_map.wyner_ziv_reweight`` per Wire-in #2 sister)
* hook #2 Pareto constraint = ACTIVE (``inflate_runtime_loc_budget`` ≤200
  per HNeRV parity L4 waiver; side_bytes_compressed_baked ≤ budget)
* hook #3 bit-allocator = ACTIVE (main_compressed + side_compressed_baked
  reduce per-substrate byte budget via canonical
  ``tac.codec.wyner_ziv_layer.insert_wyner_ziv_layer``)
* hook #4 cathedral autopilot dispatch = ACTIVE via auto-discovery per
  Catalog #335 canonical contract + Catalog #336 invocation discipline
* hook #5 continual-learning posterior = ACTIVE via
  ``tac.council_continual_learning.append_council_anchor`` per Catalog #355
  meta-Lagrangian wire-in
* hook #6 probe-disambiguator = ACTIVE; canonical predecessor
  ``tools/probe_wyner_ziv_composition_alpha_disambiguator.py`` (sister stub
  landed 2026-05-17; reactivation-pending-empirical per primitive landing
  memo) IS the apples-to-apples arbiter between additive (α=1.0) vs
  saturating (α=0.5) composition with the FEC6 0.19205 frontier per Catalog
  #227 substrate_composition_matrix discipline.
"""

from __future__ import annotations

from tac.substrate_registry import (
    SubstrateContract,
    register_substrate,
)


__all__ = (
    "SUBSTRATE_ID",
    "LANE_ID",
    "WYNER_ZIV_PIPELINE_STAGE_CODEC_CONTRACT",
    "OBSERVABILITY_SURFACE",
)

SUBSTRATE_ID = "wyner_ziv_pipeline_stage_codec"
LANE_ID = "lane_wyner_ziv_pipeline_stage_codec_l0_scaffold_20260528"


# 6-facet observability surface per Catalog #305. Each facet is a fact about
# what THIS substrate exposes at runtime so a reviewer can decompose its
# behavior without re-instrumentation per CLAUDE.md "Max observability —
# non-negotiable".
OBSERVABILITY_SURFACE = {
    "inspectable_per_layer": (
        "Per-pair (offset_in_y, prefix_len) split tuple exposed via the "
        "trainer's ``layerwise_inspector`` hook (one tuple per inserted "
        "Wyner-Ziv stage); per-stage (main_bytes_raw, main_bytes_compressed, "
        "side_bytes_raw, side_bytes_compressed_baked) byte counts exposed "
        "via ``architecture.report_stage_byte_counts()``."
    ),
    "decomposable_per_signal": (
        "Per-substrate-iteration total loss decomposed into (1) main-stream "
        "rate term (post-entropy bytes), (2) side-stream bake-in cost "
        "(inflate.py LOC overhead), (3) cooperative-receiver discrepancy "
        "(decoder-side PoseNet output deviation from encoder-time PoseNet "
        "prediction), (4) HNeRV parity L4 ≤200 LOC budget headroom."
    ),
    "diff_able_across_runs": (
        "Archive bytes byte-stable under deterministic seed + fixed Y "
        "derivation (PoseNet output is decoder-deterministic per upstream/"
        "modules.py); two runs of the same Wyner-Ziv config + same source "
        "pre-entropy bytes produce sha256-identical ``main_compressed`` + "
        "``side_compressed_baked`` payloads (regression test "
        "``tests/test_wyner_ziv_pipeline_stage_codec_smoke.py::"
        "test_archive_byte_stability_under_deterministic_seed``)."
    ),
    "queryable_post_hoc": (
        "Training emits ``wyner_ziv_pipeline_stage_codec_training_"
        "observability_<utc>.jsonl`` with per-iteration (iter_idx, "
        "main_bytes_compressed, side_bytes_compressed_baked, "
        "score_savings_estimate, inflate_py_loc_added, "
        "decoder_complexity_estimate_seconds) — fcntl-locked APPEND-ONLY "
        "per Catalog #128/#131 sister discipline."
    ),
    "cite_able": (
        "Every persisted artifact carries Provenance per Catalog #323 with "
        "(commit_sha, lane_id=LANE_ID, substrate_id=SUBSTRATE_ID, "
        "canonical_helper_invocation='tac.codec.wyner_ziv_layer."
        "insert_wyner_ziv_layer', hardware_substrate, intercept_location, "
        "side_info_source); ``score_claim=False`` + ``promotable=False`` + "
        "``axis_tag='[predicted]'`` at L0 per Catalog #341 Tier A markers."
    ),
    "counterfactual_able": (
        "Catalog #105/#139/#220/#272 byte-mutation-smoke discipline: every "
        "byte in the Wyner-Ziv side stream (offset_in_y + prefix_len bake-in) "
        "is paired with a counterfactual probe that mutates one byte + reruns "
        "``reconstruct_from_wyner_ziv_layer`` + verifies decoded "
        "pre_entropy_bytes diverge from source. The encode-decode roundtrip "
        "IS the canonical Catalog #139 packet compiler no-op detector "
        "applied at the pipeline-stage primitive surface."
    ),
}


# Per Catalog #241 canonical contract registration. 36-field schema per
# ``tac.substrate_registry.contract.SubstrateContract`` (read inline 2026-05-28
# PV per Catalog #229).
#
# Field-by-field canonical-vs-unique decisions per CLAUDE.md
# UNIQUE-AND-COMPLETE-PER-METHOD operating mode (full table in the sister
# design memo):
#
# | Field | Decision | Rationale |
# |---|---|---|
# | id / lane_id | ADOPT_CANONICAL_BECAUSE_SERVES | Catalog #241/#242 schema |
# | target_modes | UNIQUE | research_substrate at L0; routes to contest_one_video_replay at L1 |
# | archive_grammar | UNIQUE | monolithic_single_file_wzpsc01 (this substrate's MAGIC) |
# | parser_section_manifest | UNIQUE | header + main_stream + side_stream + Y_derivation_meta + sorted-keys JSON |
# | inflate_runtime_loc_budget | UNIQUE | 200 LOC waiver per HNeRV parity L4 (substrate-engineering tier) |
# | runtime_dep_closure | UNIQUE | (numpy, brotli) — 8th MLX-FIRST + HNeRV L4 ≤2 deps |
# | export_format | ADOPT_CANONICAL ("custom") | bridge MLX state_dict → npz → ZIP-member → numpy |
# | score_aware_loss | ADOPT_CANONICAL ("custom") | bridges Atick-Redlich coop-receiver MI to bytes |
# | bolt_on_loc_budget | N/A_SUBSTRATE_ENGINEERING | 999 sentinel; HNeRV parity L7 |
# | no_op_detector_planned | ADOPT_CANONICAL (True) | Catalog #105/#139 byte-mutation smoke |
# | score_improvement_mechanism_status | UNIQUE (RESEARCH_ONLY at L0) | flips to OPERATIONAL at L1 |
# | runtime_overlay_consumed | UNIQUE (False at L0) | paired with RESEARCH_ONLY per __post_init__ |
# | recipe_smoke_only / recipe_research_only | ADOPT_CANONICAL (True/True) | L0 scaffold contract |
# | recipe_min_smoke_gpu | UNIQUE (T4) | placeholder; actual ladder lands at L1 |
# | recipe_min_vram_gb | UNIQUE (12) | T4 16GB floor; revised per measurement at L1 |
# | recipe_pyav_decode_strategy | ADOPT_CANONICAL | cpu_thread_async_upload (sister Z6-v2 pattern) |
# | recipe_canary_status | UNIQUE (independent_substrate) | this substrate routes through any base substrate's pre-entropy stage |
# | recipe_video_input_strategy | ADOPT_CANONICAL | readonly_mmap (substrate-engineering default) |
# | cost_band_epochs | UNIQUE (50) | L0 smoke cap; consistent with recipe_smoke_only |
# | cost_band_gpu_key / cost_band_platform_key | ADOPT_CANONICAL | T4/modal placeholder |
# | cost_band_p50_usd | UNIQUE (0.30) | L0 smoke estimate; revised post-empirical |
# | hook_* (6) | ALL ACTIVE | per class-shift routing per Catalog #311 |
# | hook_autopilot_ranker_class_shift_token | UNIQUE | "pipeline_stage_codec_decoder_side_posenet_side_info_wyner_ziv_1976_per_catalog_311" |
# | hook_continual_learning_anchor_kind | UNIQUE (macos_cpu_advisory) | MLX-local non-promotable per Catalog #192 |
# | hook_probe_disambiguator | ADOPT_CANONICAL | sister probe at tools/probe_wyner_ziv_composition_alpha_disambiguator.py |
# | catalog_compliance_declarations | UNIQUE | 5 canonical compliance tokens declared |
#
# ``recipe_research_only=True`` per CLAUDE.md "Substrate scaffolds MUST be
# COMPLETE or RESEARCH-ONLY" non-negotiable. ``score_improvement_mechanism_
# status="RESEARCH_ONLY"`` + ``runtime_overlay_consumed=False`` per the
# contract's __post_init__ invariant: the inflate path WILL operationally
# consume archive bytes for Wyner-Ziv decoder-side reconstruction at L1+
# promotion (the L0 scaffold's mechanism is declared but the L1 trainer
# implementation is the next operator-routable step).
WYNER_ZIV_PIPELINE_STAGE_CODEC_CONTRACT = SubstrateContract(
    # 2.1 Identity & lifecycle (5)
    id=SUBSTRATE_ID,
    lane_id=LANE_ID,
    target_modes=("research_substrate",),
    deployment_target="desktop_research",
    council_verdict_provenance=(
        ".omx/research/wyner_ziv_pipeline_stage_codec_design_20260528.md"
    ),
    # 2.2 Architecture & runtime (8 per Catalog #124)
    archive_grammar="monolithic_single_file_wzpsc01",
    parser_section_manifest={
        "header": "magic_wzpsc01_plus_version_plus_section_offsets",
        "main_stream": "wyner_ziv_main_stream_post_entropy_bytes",
        "side_stream": "wyner_ziv_side_stream_offset_prefix_len_baked",
        "y_derivation_meta": "side_info_source_plus_intercept_location_plus_codec_choices_json",
        "meta": "json_meta_provenance_quantization_scales",
    },
    inflate_runtime_loc_budget=200,  # HNeRV L4 waiver per substrate-engineering scope
    runtime_dep_closure=("numpy", "brotli"),  # HNeRV L4 <=2 deps per 8th MLX-first directive
    export_format="custom",  # MLX state_dict -> npz -> ZIP-member -> numpy inflate bridge
    score_aware_loss="custom",  # cooperative-receiver MI-max via decoder-side PoseNet side-info
    bolt_on_loc_budget=999,  # N/A; substrate_engineering scope per HNeRV parity L7
    no_op_detector_planned=True,  # per Catalog #105/#139 byte-mutation smoke discipline
    # 2.3 Operational mechanism (3 per Catalog #220)
    archive_bytes_added=None,  # L0 scaffold has no archive yet; declared at L1 promotion
    score_improvement_mechanism_status="RESEARCH_ONLY",  # L0; flips to OPERATIONAL at L1
    runtime_overlay_consumed=False,  # paired with RESEARCH_ONLY per contract invariant
    # 2.4 Recipe schema (8)
    recipe_smoke_only=True,  # L0 scaffold is smoke-only by construction
    recipe_research_only=True,  # transparent non-promotable per CLAUDE.md non-negotiable
    recipe_min_smoke_gpu="T4",  # placeholder; recipe lands at L1 with actual ladder
    recipe_min_vram_gb=12,  # T4 floor; revised per actual measurement at L1
    recipe_pyav_decode_strategy="cpu_thread_async_upload",
    recipe_canary_status="independent_substrate",
    recipe_video_input_strategy="readonly_mmap",
    recipe_canary_dependency=None,
    # 2.5 Cost band & GPU envelope (4)
    cost_band_epochs=50,  # L0 smoke cap; consistent with recipe_smoke_only=True (<=100)
    cost_band_gpu_key="T4",  # matches recipe_min_smoke_gpu per MacKay M1 invariant
    cost_band_platform_key="modal",
    cost_band_p50_usd=0.30,  # L0 smoke estimate; revised post-empirical
    # 2.6 6-hook wire-in (6 per Catalog #125) — all ACTIVE for class-shift routing
    hook_sensitivity_contribution="scorer_conditional_entropy_map_v1",
    hook_pareto_constraint="rate_distortion_v1",
    hook_bit_allocator_class="per_tensor_uniform",
    hook_autopilot_ranker_class_shift_token=(
        "pipeline_stage_codec_decoder_side_posenet_side_info_wyner_ziv_1976_per_catalog_311"
    ),
    hook_continual_learning_anchor_kind="macos_cpu_advisory",  # MLX-local non-promotable per Catalog #192
    hook_probe_disambiguator=(
        "tools/probe_wyner_ziv_composition_alpha_disambiguator.py"
    ),  # sister stub landed 2026-05-17; reactivation-pending-empirical
    # 2.7 Compliance + 2.8 not-applicable rationales
    catalog_compliance_declarations=(
        "catalog_146_3arg_archive_grammar_honored",
        "catalog_164_scorer_preprocess_input_called",
        "catalog_205_select_inflate_device_used",
        "catalog_220_operational_mechanism_declared",
        "catalog_226_gate_auth_eval_call_used",
    ),
    hook_not_applicable_rationale={},  # all 6 hooks ACTIVE; no rationale needed
)

register_substrate(WYNER_ZIV_PIPELINE_STAGE_CODEC_CONTRACT)
