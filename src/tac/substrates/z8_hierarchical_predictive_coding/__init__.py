# SPDX-License-Identifier: MIT
# LEGACY_SUBSTRATE_PRE_META_LAYER:z8_l0_scaffold_pending_meta_layer_register_substrate_decorator_per_phase_2_substrate_class_shift_canonical_quadruple_binding_per_catalog_312
"""tac.substrates.z8_hierarchical_predictive_coding — Z8 hierarchical predictive coding L0 SCAFFOLD.

Path 3 substrate-class-shift candidate F per
``.omx/research/path_3_candidate_inventory_for_next_wave_spawning_20260526.md``
Tier 1. FRESH SUBSTRATE DESIGN (2-phase methodology) per operator binding
directive 2026-05-26:

  *"The MLX first requirement might also force us out of the issue we were
  having before where we had great ideas but we're building them as Boltons to
  the same substrates over and over again; we want to design the substrate and
  curriculum and then optimize the design the whole stack around it for
  extreme optimization and performance and optimal score lowering"*

Z8 binds Catalog #312's canonical quadruple SIMULTANEOUSLY per HNeRV parity
discipline L7 (substrate engineering UNIQUE-IFIES; binds ALL ingredients NOT
incrementally):

1. **Rao-Ballard 1999** hierarchical predictive coding (multi-level generative
   model with top-down prediction + bottom-up error encoding).
2. **Mallat 1989** wavelet multi-scale (orthogonal wavelet packet decomposition
   of detail bands per level; Daubechies-CDF entropy coding).
3. **Hafner DreamerV3 2023** latent dynamics (discrete-categorical posterior at
   each level + deterministic GRU state + stochastic categorical state).
4. **Wyner-Ziv 1976** source coding with side information (top-level latent
   Wyner-Ziv-coded against frame_0's decoded latent).

Z8 is the asymptotic-pursuit terminal of the F-asymptote-trajectory; Z6
(sister D landed) is single-layer FiLM low-engineering-risk anchor and Z7-Mamba-2
(sister B' in-flight) is intermediate. Z8 supersedes Z6+Z7 architecturally (NOT
in dispatch order — Z6 is canonical FIRST empirical anchor per parent scoping
memo's engineering-risk-minimization recommendation) by binding ALL four
canonical primitives in a single coherent stack-of-stacks substrate.

Design memo (parent):
``.omx/research/path_3_f_z8_hierarchical_predictive_coding_substrate_design_20260526.md``

Sister substrate citation (Catalog #230)
-----------------------------------------

**LANDED** (reference as research INPUT not bolt-on target):

- **A=DreamerV3 RSSM** (``69253a1cc``) — canonical per-level categorical
  posterior + Gumbel-Softmax STE primitive. Z8 reuses the categorical-posterior
  primitive at each hierarchy level (per Catalog #290
  ADOPT_CANONICAL_BECAUSE_SERVES decision).
- **D=Z6 predictive coding** (``83b9ee3e2``) — canonical FiLM-conditioned
  next-frame predictor + ego-motion conditioning. Z8 generalizes to 3-level
  hierarchical predictor.
- **E=BoostNeRV against PR110** (``83910e54e``) — residual-learner-against-PR110
  pattern; Z8 is orthogonal.

**IN-FLIGHT** (avoid file collision per Catalog #340):
- B'=Z7-Mamba-2; C'=NSCS06 v8 chroma_lut.

**CONCURRENT SISTER SPAWNS**: G=NIRVANA cascading NeRV; H=ATW V2
cooperative-receiver. Disjoint scope.

Catalog #124 archive-grammar 8 fields (declared inline so AST walker observes them)
-----------------------------------------------------------------------------------

- ``archive_grammar``: monolithic single-file ``0.bin`` with Z8HPC1 magic
  prefix; per-level decoder + per-level categorical indices + per-level Mallat
  wavelet coeffs + Wyner-Ziv top-level coded blob + DreamerV3 state init +
  meta JSON.
- ``parser_section_manifest``: Z8HPC1_HEADER + DECODER_BLOB + INDICES_BLOB +
  WAVELET_BLOB + WYNER_ZIV_BLOB + DREAMER_STATE_BLOB + META_BLOB; canonical
  Z8HPC1_HEADER_FMT below.
- ``inflate_runtime_loc_budget``: ≤200 LOC substrate-engineering waiver per
  HNeRV parity L4 explicit waiver (per parent scoping memo Z8 estimate ~280
  LOC; targeting ≤200 via canonical helper reuse).
- ``runtime_dep_closure``: torch + brotli + numpy only (HNeRV L4 ≤2 deps; numpy
  is universal foundation NOT a substrate-engineering dep).
- ``export_format``: Z8HPC1 monolithic single-file ``0.bin`` (fp16 + brotli +
  Daubechies-CDF wavelet coeffs + Wyner-Ziv conditional coding).
- ``score_aware_loss``: ``Z8HierarchicalPredictiveCodingScoreAwareLoss`` routes
  through ``score_pair_components`` per Catalog #164; eval-roundtrip mandatory;
  per-level Rao-Ballard residual L2 + Mallat sub-band entropy + Wyner-Ziv
  conditional entropy + canonical seg+pose+rate decomposition.
- ``bolt_on_loc_budget``: ``lane_class=substrate_engineering`` (HNeRV L7);
  multi-level RSSM stack + per-level Mallat wavelet codec + DreamerV3 latent
  dynamics + Wyner-Ziv top-level coder is substrate engineering.
- ``no_op_detector_planned``: every section MUST be operationally consumed
  by the inflate runtime; 4 distinguishing primitives (multi-level RSSM /
  Mallat wavelet / DreamerV3 RSSM per-level / Wyner-Ziv) byte-mutation-tested
  per ``tools/verify_distinguishing_feature_byte_mutation.py``. Catalog #272
  distinguishing-feature integration contract honored.

target_modes: ``contest_one_video_replay``, ``contest_generalized``,
``research_substrate``
lane_class: ``substrate_engineering``
research_only: true (L0 SCAFFOLD pending Phase 2 council approval to lift
``_full_main raises NotImplementedError`` per CLAUDE.md "Substrate scaffolds
MUST be COMPLETE or RESEARCH-ONLY")
canary_status: ``independent_substrate`` (substrate-CLASS shift binding 4
canonical primitives; parallel sister Path 3 candidate to A=DreamerV3 / D=Z6 /
E=BoostNeRV / G=NIRVANA / H=ATW V2)

axis_tag: ``[macOS-MLX research-signal]`` per CLAUDE.md "MLX portable-local-substrate
authority"
score_claim: false
promotable: false
ready_for_exact_eval_dispatch: false

Probe-disambiguator (Catalog #125 hook #6)
------------------------------------------

Per-primitive ablation IS the canonical disambiguator. The Z8 trainer (Phase 2)
supports ``--ablate-rao-ballard`` / ``--ablate-mallat`` /
``--ablate-dreamer-categorical`` / ``--ablate-wyner-ziv`` argparse flags; each
disables ONE primitive and measures ΔS contribution per primitive empirically.
The 4-way ablation cross-tabulates the multiplicative bound vs the actual
additive contribution at the empirical operating point.

6-hook wire-in (Catalog #125 NON-NEGOTIABLE)
-------------------------------------------

1. **Sensitivity-map** — per-level prediction error L2 norm IS the per-tensor
   importance signal at each hierarchy level; register
   ``sensitivity_map.z8_hierarchical_predictive_coding_v1`` post Phase 2.
2. **Pareto constraint** — adds
   ``per_level_prediction_error_entropy ≤ ε_level`` to the convex feasibility
   region per level; register ``tac.pareto.z8_hierarchical_predictive_coding_v1``
   post-smoke.
3. **Bit-allocator hook** — per-level prediction-error bit allocation derives
   from per-level Mallat wavelet detail-band sparsity; register
   ``bit_allocator.z8_hierarchical_predictive_coding_per_level_v1`` post-smoke.
4. **Cathedral autopilot dispatch hook** — recipe planned at
   ``.omx/operator_authorize_recipes/substrate_z8_hierarchical_predictive_coding_modal_a100_dispatch.yaml``;
   gated by Catalog #167 smoke-before-full + Catalog #325 per-substrate
   symposium (REQUIRED before paid dispatch); ranker v2 receives
   ``literature_anchor=Rao-Ballard1999+Mallat1989+Hafner2023+Wyner-Ziv1976``
   as source-basis metadata only.
5. **Continual-learning posterior** — every Z8 empirical anchor seeds the
   posterior via ``posterior_update_locked`` (Catalog #128).
6. **Probe-disambiguator** — per-primitive ablation IS the probe; see above.

Cross-references
----------------

- Design memo (parent):
  ``.omx/research/path_3_f_z8_hierarchical_predictive_coding_substrate_design_20260526.md``
- Sister Z6 L1 scaffold (parent-substrate research input):
  ``src/tac/substrates/time_traveler_l5_z6/``
- Sister DreamerV3 RSSM (per-level categorical posterior research input):
  ``src/tac/substrates/dreamer_v3_rssm/``
- Parent scoping memo (Z6/Z7/Z8 design):
  ``.omx/research/time_traveler_l5_z6_z7_z8_predictive_coding_world_models_asymptotic_pursuit_scoping_design_20260516.md``
- Rao & Ballard (1999) "Predictive coding in the visual cortex: a functional
  interpretation of some extra-classical receptive-field effects" Nature
  Neuroscience 2(1):79-87
- Mallat (1989) "A theory for multiresolution signal decomposition: the
  wavelet representation" IEEE PAMI 11(7):674-693
- Hafner et al. (2023) DreamerV3 arXiv:2301.04104
- Wyner & Ziv (1976) IEEE Trans. Inf. Theory IT-22:1

Observability surface (per Catalog #305 MAX-OBSERVABILITY directive)
--------------------------------------------------------------------

This substrate honors the 6-facet observability surface per the design memo
Section 7. Per-layer inspection / per-signal decomposition / run-to-run diff /
post-hoc query / cite-chain / counterfactual hooks all enumerated.

Lane: ``lane_path_3_f_z8_hierarchical_predictive_coding_canonical_quadruple_20260526``
"""

from __future__ import annotations

from tac.substrates.z8_hierarchical_predictive_coding.archive import (
    Z8HPC1_HEADER_FMT,
    Z8HPC1_HEADER_SIZE,
    Z8HPC1_MAGIC,
    Z8HPC1_SCHEMA_VERSION,
    Z8HierarchicalArchive,
    pack_archive,
    parse_archive,
    parse_z8hpc1_archive_bytes,
)
from tac.substrates.z8_hierarchical_predictive_coding.mlx_renderer import (
    DEFAULT_NUM_LEVELS,
    EVAL_HW,
    NUM_PAIRS,
    Z8HierarchicalConfig,
    Z8HierarchicalPredictiveCoderMLX,
    z8_decoder_param_count,
)

# Catalog #124 8-field representation-lane declaration (canonical tokens for
# the AST walker per the gate's regex set). DO NOT remove without operator
# review per Catalog #229 premise verification.
ARCHIVE_GRAMMAR_FIELDS: dict[str, str] = {
    "archive_grammar": "monolithic_single_file_0bin_z8hpc1",
    "parser_section_manifest": (
        "z8hpc1_header + decoder_blob + indices_blob + wavelet_blob "
        "+ wyner_ziv_blob + dreamer_state_blob + meta_blob"
    ),
    "inflate_runtime_loc_budget": (
        "le_200_loc_substrate_engineering_waiver_per_hnerv_parity_l4_plus_l7"
    ),
    "runtime_dep_closure": (
        "torch_brotli_numpy_only_per_hnerv_parity_l4"
    ),
    "export_format": (
        "z8hpc1_monolithic_single_zip_member_0bin_with_wavelet_and_wyner_ziv_coded_sections"
    ),
    "score_aware_loss": (
        "pending_phase_2_trainer_canonical_helper_score_pair_components_catalog_164_"
        "plus_per_level_rao_ballard_residual_plus_mallat_subband_entropy_plus_wyner_ziv_conditional"
    ),
    "bolt_on_loc_budget": (
        "substrate_engineering_lane_class_per_hnerv_parity_l7_waiver_"
        "binds_canonical_quadruple_simultaneously_per_catalog_312"
    ),
    "no_op_detector_planned": (
        "byte_mutation_gate_catalog_139_plus_105_plus_272_distinguishing_feature_"
        "four_primitives_multi_level_rssm_plus_mallat_wavelet_plus_dreamer_categorical_plus_wyner_ziv"
    ),
}

# Canonical equation references per Catalog #344 (registered in
# .omx/state/canonical_equations_registry.jsonl).
CANONICAL_EQUATION_IDS: tuple[str, ...] = (
    "mlx_pytorch_full_decoder_downstream_scorer_drift_propagation_v1",
    "scorer_conditional_joint_rate_distortion_floor_v1",
    "categorical_posterior_capacity_vs_continuous_gaussian_v1",
    "ego_motion_concentration_prior_v1",
    "cross_codec_super_additive_orthogonality_predictor_v1",
)

IMPLEMENTATION_STATUS = (
    "l0_research_only_mlx_renderer_archive_scaffold"
)
RESEARCH_ONLY = True

PLANNED_PUBLIC_API = (
    "DEFAULT_NUM_LEVELS",
    "EVAL_HW",
    "NUM_PAIRS",
    "Z8HierarchicalArchive",
    "Z8HierarchicalConfig",
    "Z8HierarchicalPredictiveCoderMLX",
    "Z8HPC1_HEADER_FMT",
    "Z8HPC1_HEADER_SIZE",
    "Z8HPC1_MAGIC",
    "Z8HPC1_SCHEMA_VERSION",
    "pack_archive",
    "parse_archive",
    "parse_z8hpc1_archive_bytes",
    "z8_decoder_param_count",
)


__all__ = [
    "ARCHIVE_GRAMMAR_FIELDS",
    "CANONICAL_EQUATION_IDS",
    "DEFAULT_NUM_LEVELS",
    "EVAL_HW",
    "IMPLEMENTATION_STATUS",
    "NUM_PAIRS",
    "PLANNED_PUBLIC_API",
    "RESEARCH_ONLY",
    "Z8HPC1_HEADER_FMT",
    "Z8HPC1_HEADER_SIZE",
    "Z8HPC1_MAGIC",
    "Z8HPC1_SCHEMA_VERSION",
    "Z8HierarchicalArchive",
    "Z8HierarchicalConfig",
    "Z8HierarchicalPredictiveCoderMLX",
    "pack_archive",
    "parse_archive",
    "parse_z8hpc1_archive_bytes",
    "z8_decoder_param_count",
]
