# SPDX-License-Identifier: MIT
# LEGACY_SUBSTRATE_PRE_META_LAYER:l0_scaffold_landed_20260526_path_3_j_mdl_ibps_discrete_categorical_mine_hybrid_cargo_cult_first_3_axis_per_operator_directives_substrate_design_not_bolt_on_plus_adversarial_cargo_cult_pass_first_plus_3_axis_recursive_adversarial_review_discipline_mlx_first_meta_layer_register_substrate_decorator_pending_phase_2_council_symposium_per_catalog_325_and_substrate_contract_canonical_helper_adoption_when_mdlibpsj1_archive_grammar_per_pair_categorical_indices_blob_plus_mine_critic_blob_fields_are_added_to_contract_schema
"""mdl_ibps_j_discrete_categorical_mine_hybrid — Path 3 J=MDL-IBPS L0 SCAFFOLD.

Path 3 candidate #J per operator binding directives 2026-05-26 verbatim:

*"The MLX first requirement might also force us out of the issue we were
having before where we had great ideas but we're building them as Boltons
to the same substrates over and over again; we want to design the
substrate and curriculum and then optimize the design the whole stack
around it for extreme optimization and performance and optimal score
lowering"*

*"Never simply extend unless a rigorous adversarial cargo cult pass has
been done first"*

*"we also need adversarial review against all landing recursive for math
and scientific and engineering rigor and for MLX drift minimization and
portability via numpy"*

This substrate is an EXTENSION of the C6 MDL-IBPS scaffold (parent at
``src/tac/substrates/c6_e4_mdl_ibps/`` preserved per Catalog #110/#113
HISTORICAL_PROVENANCE; C6 v1 recipe in `phantom_random_init` per Catalog
#324 post-training Tier-C re-measurement empirical falsification anchor
2026-05-19); the EXTENSION followed the 3-phase methodology per binding
directive #2 (Phase 1 cargo-cult audit BEFORE extension):

- Phase 1: ``.omx/research/path_3_j_mdl_ibps_cargo_cult_audit_of_c6_scaffold_20260526.md``
  (11 CCs classified HARD-EARNED vs CARGO-CULTED; CC-J-1 critical anchor =
  predicted_band-from-random-init-Tier-C-density assumption; CC-J-2 through
  CC-J-6 + CC-J-10 + CC-J-11 enumerate substrate-design decision space)
- Phase 2: ``.omx/research/path_3_j_mdl_ibps_substrate_design_decision_20260526.md``
  (Path (a) DISCRETE-CATEGORICAL-MINE-HYBRID chosen with explicit DISTINCTNESS
  from sisters A=DreamerV3 / F=Z8 / K=COIN++ / parent C6 v1)
- Phase 3: this scaffold + sister memo ``.omx/research/path_3_j_mdl_ibps_substrate_design_20260526.md``
  + landing memo ``.omx/research/path_3_j_mdl_ibps_L0_scaffold_landed_20260526.md``

Paradigm anchor: Minimum Description Length (Rissanen 1978) × Information
Bottleneck (Tishby-Zaslavsky 2015) with cargo-cult-resurrected reducer
methodology per T3 v2 cargo-cult resurrection symposium 2026-05-19. Sister
F=Z8 (LANDED commit 5ff5d2ab9) binds Tishby IB hierarchically per Rao-Ballard
quadruple; J=MDL-IBPS forks to SINGLE-SCALE + MINE-tight-MI-lower-bound +
DISCRETE-CATEGORICAL-POSTERIOR + HYBRID procedural-coord-MLP-decoder +
sparse-Laplacian regularizer. Sister A=DreamerV3 (LANDED commit 69253a1cc)
uses categorical posterior at K=256 × G=24 = 192 bits/sample at the
substrate-latent surface; J forks to K=16 × G=12 = 48 bits/sample for
matched MDL-optimal per-pair bit-budget target.

Architecture (DISCRETE-CATEGORICAL-MINE-HYBRID; Phase 2 chosen):

    Per-pair categorical-index modulation m_i in {0, 1, ..., K-1}^G
        [K=16 categorical alphabet, G=12 groups; 48 bits per pair]
       |
       v   (Gumbel-Softmax reparametrization for training; argmax for inference)
       v
    FiLM modulation: scale_i, shift_i = linear_film_proj(one_hot(m_i))
                                       -> R^(HIDDEN_DIM * 2)
       |
       v
    Procedural coord-MLP base F_phi:
        Input: (x, y, t) in [0,1]^2 x {0,1}
            Sinusoidal positional encoding: coord -> R^(POS_DIM * 2 * 3)
        Hidden layers: 3 x HIDDEN_DIM=64 with FiLM modulation per layer:
            h <- sin( linear(h) * scale_i + shift_i )
        Output: linear(h) -> R^3 -> sigmoid -> rgb in [0, 1]^3
       |
       v   per pixel (x, y) in [0, 384) x [0, 512)
       v
    Stack: rgb_0, rgb_1 in (B, 3, 384, 512)   [FULL contest scorer resolution; CC-J-6 unwind]

Loss composition:

    L_score = score_pair_components(rgb_pair, gt_pair, segnet, posenet,
                                    eval_roundtrip=True)   [canonical Catalog #164]
    L_IB = beta * MINE_lower_bound_critic(z, frames)        [Belghazi 2018; CC-J-4 unwind]
    L_sparse = lambda_sparse * |W_film|_1                   [MacKay sparse-Laplacian; Path B5 influence]
    L_total = L_score + L_IB + L_sparse

Distinct from sister candidates by reducer methodology + architecture composition:
- A=DreamerV3 RSSM (LANDED): categorical posterior K=256 x G=24 = 192 bits/sample
- F=Z8 hierarchical predictive coding (LANDED): Rao-Ballard hierarchical quadruple
- G=NIRVANA cascading NeRV (LANDED): hierarchical residual decoder cascade
- K=COIN++ (LANDED): meta-learned modulated coord-MLP via continuous FiLM
- C6 v1 (PARENT; phantom_random_init): 24-dim continuous Gaussian posterior, beta=0.01, 48x64 decoder
- **J=MDL-IBPS DISCRETE-CATEGORICAL-MINE-HYBRID (THIS L0)**: K=16 x G=12 = 48 bits/sample +
  MINE tight-MI lower bound + HYBRID procedural-FiLM-modulated decoder +
  full 384x512 resolution + empirical beta-sweep + sparse-Laplacian regularizer

Catalog #124 archive-grammar 8 fields (declared inline so AST walker observes them)
-----------------------------------------------------------------------------------

- ``archive_grammar``: monolithic single-file ``0.bin`` (MDLIBPS-J1 grammar)
- ``parser_section_manifest``: MIBJ1 header (32 bytes) + BASE_BLOB +
  MINE_BLOB + INDICES_BLOB + META_BLOB
- ``inflate_runtime_loc_budget``: <=200 LOC substrate-engineering waiver
  (procedural-coord-MLP forward + FiLM modulation + sigmoid +
  canonical bilinear -> contest HW per Catalog #205 device-fork helper)
- ``runtime_dep_closure``: torch + brotli only (HNeRV parity L4 <= 2 deps;
  canonical Catalog #146 + #205 + #295 self-containment)
- ``export_format``: MDLIBPS-J1 monolithic single-zip-member ``0.bin``
- ``score_aware_loss``: ``MDLIBPSJScoreAwareLoss`` routes through canonical
  ``score_pair_components`` per Catalog #164; eval-roundtrip mandatory
- ``bolt_on_loc_budget``: ``lane_class=substrate_engineering`` (HNeRV
  parity L7); procedural-coord-MLP + FiLM + MINE + categorical-indices
  composition is substrate engineering
- ``no_op_detector_planned``: emit/parse roundtrip preserves bytes
  byte-for-byte; INDICES_BLOB + BASE_BLOB are frame-affecting at inflate
  per Catalog #220/#272 distinguishing-feature contract; MINE_BLOB is
  provenance/training-only (Catalog #220 declaration: NOT score-affecting
  at inflate); META_BLOB is parse/config gate

target_modes: ``contest_one_video_replay``, ``contest_generalized``, ``research_substrate``
lane_class: ``substrate_engineering``
research_only: true  (L0 SCAFFOLD; full pipeline pending Phase 3 follow-on smoke + Catalog #325 symposium)
canary_status: ``independent_substrate`` (substrate-CLASS shift from HNeRV-family + distinct
    from sister DREAMER_V3_RSSM via categorical-alphabet bit-budget fork)
dispatch_enabled: false  (Catalog #240 (c) opt-out; _full_main raises NotImplementedError)
predicted_band_validation_status: pending_post_training  (Catalog #324; CC-J-1 unwind)

CLAUDE.md compliance:

- No silent device defaults (MLX explicit; PyTorch export path uses canonical
  ``tac.substrates._shared.inflate_runtime.select_inflate_device`` per Catalog #205)
- No scorer load at inflate time (only procedural-coord-MLP forward + FiLM
  modulation + sigmoid + canonical bilinear upscale to camera HW + uint8 cast)
- No /tmp paths in persisted artifacts (Catalog #113 forbidden-path)
- Every file reviewable in 30 seconds per HNeRV parity L12 (mlx_renderer +
  numpy_reference + archive + inflate + ib_loss_mine + __init__ + tests/test_basic)
- All artifacts carry ``[macOS-MLX research-signal]`` + ``score_claim=false`` +
  ``promotion_eligible=false`` + ``ready_for_exact_eval_dispatch=false`` per
  Catalog #192/#317 (CC-J-10 unwind compliance)

6-hook wire-in declaration (Catalog #125 NON-NEGOTIABLE)
-------------------------------------------------------

1. **Sensitivity-map** - per-pair categorical-index distribution is the
   per-pair sensitivity primitive; register
   ``sensitivity_map.path_3_j_mdl_ibps_v1`` (planned post-Stage-1 smoke).
2. **Pareto constraint** - ``(rate_categorical_indices <= 4 KB) intersect
   (rate_base_decoder <= 50 KB) intersect (rate_film_matrices <= 10 KB)
   intersect (S(theta) >= canonical_frontier_pointer - eps)``; register
   ``tac.pareto.mdl_ibps_j_v1``.
3. **Bit-allocator** - ``beta`` + ``lambda_sparse`` + ``K=16`` + ``G=12``
   are the bit-allocator knobs; register ``bit_allocator.mdl_ibps_j_v1``.
4. **Cathedral autopilot dispatch hook** - planned cathedral consumer at
   ``tac.cathedral_consumers.mdl_ibps_j_routing_consumer/`` (Phase 3
   follow-on per Catalog #335 canonical contract) routes substrate
   candidates per Catalog #341 canonical non-promotable markers.
5. **Continual-learning posterior** - every empirical anchor (per-arm
   beta + Tier-C density + final-score decomposition) emits canonical
   posterior anchor per Catalog #300 v2 frontmatter via
   ``tac.council_continual_learning.append_council_anchor``.
6. **Probe-disambiguator** - planned beta-sweep probe
   ``tools/probe_path_3_j_mdl_ibps_beta_sweep_disambiguator.py`` (Phase 3
   follow-on) emits per-arm score decomposition for empirical beta-optimum
   derivation per Catalog #1265 MLX-first gate + post-training Tier-C
   verdict per Catalog #324.

Cross-references
----------------

- Master inventory: ``.omx/research/path_3_candidate_inventory_for_next_wave_spawning_20260526.md`` (Tier 2 J)
- Phase 1 audit: ``.omx/research/path_3_j_mdl_ibps_cargo_cult_audit_of_c6_scaffold_20260526.md``
- Phase 2 design decision: ``.omx/research/path_3_j_mdl_ibps_substrate_design_decision_20260526.md``
- Phase 3 design memo: ``.omx/research/path_3_j_mdl_ibps_substrate_design_20260526.md``
- Phase 3 landing memo: ``.omx/research/path_3_j_mdl_ibps_L0_scaffold_landed_20260526.md``
- Parent C6 substrate: ``src/tac/substrates/c6_e4_mdl_ibps/``
- Parent C6 cargo-cult-unwind: ``.omx/research/c6_e4_mdl_ibps_cargo_cult_unwind_design_20260516.md``
- Parent C6 Tier-C re-measurement (Catalog #324 anchor): ``.omx/research/c6_ibps_post_training_tier_c_remeasurement_landed_20260519.md``
- Sister A=DreamerV3 RSSM: ``src/tac/substrates/dreamer_v3_rssm/``
- Sister F=Z8 hierarchical: ``src/tac/substrates/z8_hierarchical_predictive_coding/``
- Sister K=COIN++: ``src/tac/substrates/coin_pp_implicit_neural_representation/`` (3-axis pattern reference)
- T3 v2 cargo-cult resurrection symposium: ``.omx/research/council_t3_cargo_cult_resurrection_c6_ibps_v2_20260519.md``
- Tishby & Zaslavsky 2015 "Deep Learning and the Information Bottleneck Principle"
- Rissanen 1978 "Modeling by shortest data description"
- Belghazi et al. 2018 "MINE: Mutual Information Neural Estimation"
- Jang et al. 2016 "Categorical Reparameterization with Gumbel-Softmax"
- Higgins et al. 2017 "beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework"
- Hafner et al. 2024 "DreamerV3" categorical posterior
- Perez et al. 2017 "FiLM: Visual Reasoning with a General Conditioning Layer"
- MacKay 2003 *Information Theory, Inference, and Learning Algorithms* chs. 28-30
- Olshausen-Field 1996 sparse coding canonical
- CLAUDE.md Catalogs #110, #113, #124, #125, #146, #164, #192, #205, #215, #220, #226,
  #229, #240, #244, #270, #287, #290, #292, #294, #295, #296, #303, #305, #309, #317,
  #324, #325, #341, #344, #1265

Lane: ``lane_path_3_j_mdl_ibps_information_bottleneck_cargo_cult_first_20260526``
"""

# Substrate-design constants (declared at module level per Catalog #124 AST walker)

EVAL_HW: tuple[int, int] = (384, 512)
"""Contest scorer-resolution (height, width); full per CC-J-6 unwind."""

NUM_PAIRS: int = 600
"""Contest pair count (1200 frames / 2 frames per pair)."""

CATEGORICAL_K: int = 16
"""Categorical alphabet size per group (CC-J-2 unwind; matched to MDL-optimal per-pair bit budget)."""

CATEGORICAL_G: int = 12
"""Number of independent categorical groups (CC-J-2 unwind; G * log2(K) = 48 bits per pair)."""

BITS_PER_PAIR: int = 48
"""Per-pair latent bit budget = G * log2(K); independent of training (Catalog #287 evidence tag)."""

HIDDEN_DIM: int = 64
"""Hidden dimensionality of procedural coord-MLP (CC-J-5 unwind; sister K=COIN++ pattern)."""

NUM_HIDDEN_LAYERS: int = 3
"""Number of FiLM-modulated hidden layers."""

POS_DIM: int = 8
"""Sinusoidal positional encoding dimensionality (per coordinate axis)."""

MINE_HIDDEN_DIM: int = 128
"""MINE critic hidden dimensionality (Belghazi 2018; CC-J-4 unwind)."""

# Archive byte targets (per Phase 2 §curriculum stage 0).
TOTAL_ARCHIVE_TARGET_BYTES_MIN: int = 45_000
"""Predicted minimum total archive bytes (procedural base + FiLM matrices + categorical indices)."""

TOTAL_ARCHIVE_TARGET_BYTES_MAX: int = 80_000
"""Predicted maximum total archive bytes (room for larger MINE-critic checkpoint)."""

# Default IB Lagrangian sweep (CC-J-3 unwind; Higgins-memorial verdict).
DEFAULT_BETA_SWEEP: tuple[float, ...] = (1e-5, 1e-4, 1e-3, 1e-2)
"""Empirical beta sweep (4 arms) per Higgins 2017 empirical-beta-tuning canonical."""

# Default sparse-Laplacian regularizer (Path B5 influence).
DEFAULT_LAMBDA_SPARSE: float = 1e-4

# Substrate identity
SUBSTRATE_ID = "path_3_j_mdl_ibps"
SUBSTRATE_FAMILY = "mdl_ib_discrete_categorical_mine_hybrid"
LANE_ID = "lane_path_3_j_mdl_ibps_information_bottleneck_cargo_cult_first_20260526"

__all__ = [
    "BITS_PER_PAIR",
    "CATEGORICAL_G",
    "CATEGORICAL_K",
    "DEFAULT_BETA_SWEEP",
    "DEFAULT_LAMBDA_SPARSE",
    "EVAL_HW",
    "HIDDEN_DIM",
    "LANE_ID",
    "MINE_HIDDEN_DIM",
    "NUM_HIDDEN_LAYERS",
    "NUM_PAIRS",
    "POS_DIM",
    "SUBSTRATE_FAMILY",
    "SUBSTRATE_ID",
    "TOTAL_ARCHIVE_TARGET_BYTES_MAX",
    "TOTAL_ARCHIVE_TARGET_BYTES_MIN",
]
