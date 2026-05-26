# SPDX-License-Identifier: MIT
<!-- Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE — J=MDL-IBPS L1-PROMOTION-CASCADE landing record. DO NOT mutate after landing. -->
<!-- Catalog #229 PV: this landing memo verifies premises empirically — adapter + L2 trainer landed; smoke fails-fast at adapter __init__ with explicit NotImplementedError per Catalog #240 L0 SCAFFOLD posture. -->
---
council_tier: T1
council_attendees:
  - Carmack
  - Selfcomp
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_predicted_mission_contribution: apparatus_maintenance
council_override_invoked: false
council_override_rationale: ""
horizon_class: frontier_pursuit
canonical_equation_refs:
  - hnerv_backbone_sensitivity_saturated_across_medal_cluster_v1
predicted_band_validation_status: pending_post_training
related_deliberation_ids:
  - path_3_j_mdl_ibps_l0_scaffold_landed_20260526
  - path_3_d_z6_l2_long_training_first_canonical_run_landed_20260526
  - l2_infra_build_canonical_long_training_infrastructure_landed_20260526
---

# J=MDL-IBPS L1-PROMOTION-CASCADE landing — canonical L2 helper structural shell

Per Path 3 canonical-substrate-development-cascade doctrine (commit `fb270e9b6`)
+ L2-INFRA-BUILD canonical helper (commit `f5e4784ef`) + Tier1-T3-OP7-OP8
cascade amendment (commit `b96418424`) + L1-PROMOTION-CASCADE-B-C-E-G-J
charter 2026-05-26: J=MDL-IBPS substrate L1-promotion to the canonical
L2 helper.

## Delivered

- **NEW** `src/tac/substrates/mdl_ibps_j_discrete_categorical_mine_hybrid/long_training_adapter.py` (~230 LOC)
  - Canonical `MdlIbpsJLongTrainingAdapter` mirroring Z6 reference template at
    `src/tac/substrates/time_traveler_l5_z6/long_training_adapter.py`
  - Style B `train_step` Protocol surface per
    `tac.training.long_training_canonical.SubstrateLongTrainingAdapter`
  - Routes through canonical scorer-preprocess per Catalog #164 (L1 follow-up)
- **NEW** `experiments/train_substrate_mdl_ibps_j_discrete_categorical_mine_hybrid_mlx_l2.py` (~189 LOC)
  - Substrate-specific `LongTrainingConfig` instantiation + ONE
    `run_long_training(adapter, config)` invocation pattern matching
    `experiments/train_substrate_z6_predictive_coding_mlx_l2.py`

## L0 SCAFFOLD posture verified (Catalog #240 honest)

Per Catalog #229 PV empirical run 2026-05-26T13:33Z:

```text
$ PYTHONPATH=src:upstream:$PWD .venv/bin/python \
    experiments/train_substrate_mdl_ibps_j_discrete_categorical_mine_hybrid_mlx_l2.py \
    --output-dir experiments/results/path_3_j_mdl_ibps_l1_promotion_smoke_20260526T133300Z/ \
    --epochs 5 --num-pairs 10

[j-mdl-ibps-l2-mlx-trainer] Decoding 20 real frames from upstream/videos/0.mkv
[j-mdl-ibps-l2-mlx-trainer] decoded 20 frames in 0.2s
NotImplementedError: J=MDL-IBPS adapter __init__ is L0 SCAFFOLD structural shell per
  L1-PROMOTION-CASCADE charter 2026-05-26 [...] L1 follow-up subagent removes this
  guard + implements the trainable wrapper + Style B train_step per Z6 reference template
```

The adapter `__init__` raises `NotImplementedError` BEFORE any GPU/MLX spend
with explicit L1-follow-up guidance per CLAUDE.md "Substrate scaffolds MUST
be COMPLETE or RESEARCH-ONLY" + Catalog #240 recipe-vs-trainer-state
consistency. This is the **correct** L0 SCAFFOLD posture; the canonical L2
helper's `PolyakEMAShadow` would otherwise raise a less actionable TypeError
downstream.

## L1 FOLLOW-UP CONTRACT (drop-in replacement)

The L1 follow-up subagent removes the adapter `__init__`
`NotImplementedError` + implements (per `long_training_adapter.py` module
docstring "L1 FOLLOW-UP CONTRACT" section):

1. Wrap `FilmProjMLX + CoordMLPBaseMLX + MINECriticMLX` as ONE
   `mlx.nn.Module` subclass with trainable parameters
2. Implement `train_step` Style B with
   `L_score + beta * L_IB + lambda_sparse * L_sparse` composition
3. Implement `export_state_dict` via PyTorch sister bridge (Catalog #1251)
4. Implement `export_archive` via canonical `pack_archive` at `archive.py`

When L1 lands, the L2 trainer entry-point at
`experiments/train_substrate_mdl_ibps_j_discrete_categorical_mine_hybrid_mlx_l2.py`
becomes functional WITHOUT modification.

## LOC reduction realization

- L1 promotion hand-rolled trainer (hypothetical when L1 lands): ~600 LOC
  matching Z6 L1 promotion estimate
- L2 canonical promotion (THIS landing): ~230 LOC adapter + ~189 LOC
  trainer = ~419 LOC total
- Realized reduction: 30% at this iteration (adapter shell is heavier
  than minimum because of substrate-specific docstring + L1 follow-up
  contract; once L1 lands the adapter body shrinks per the active
  primitives + the L2 trainer stays at ~150 LOC matching Z6's 137-LOC
  reference). Full Z6-class 78% reduction realized at L1 follow-up
  (canonical helper absorbs EMA/checkpoint/Provenance/posterior anchor).

## Sister #1265 gate

DEFERRED per per-substrate-grammar parameterization (no
`tools/gate_mlx_candidate_contest_equivalence_<j_substrate_grammar>.py`
exists yet). L1 follow-up subagent parameterizes the canonical gate via
the existing Z6PCWM1 sister pattern at
`tools/gate_mlx_candidate_contest_equivalence_z6.py`.

## Canonical equation cross-reference (Catalog #344)

The J substrate's __init__.py declares
`CANONICAL_EQUATION_IDS = ("...",)` per Wave #1 posterior emission
wire-in 2026-05-26; this L1-promotion landing carries the same
canonical equation reference as upstream
`hnerv_backbone_sensitivity_saturated_across_medal_cluster_v1` is the
cross-substrate-class anchor most relevant to per-substrate scaffold
operational mechanism declaration.

## 6-hook wire-in declaration (Catalog #125)

- hook #1 sensitivity-map: N/A at L0 SCAFFOLD shell (L1 follow-up implements)
- hook #2 Pareto constraint: N/A at L0 SCAFFOLD shell (canonical helper
  handles when L1 wires `train_step`)
- hook #3 bit-allocator: N/A at L0 SCAFFOLD shell (canonical helper handles)
- hook #4 cathedral autopilot dispatch: ACTIVE via inherited Wave #1
  canonical posterior emission per `tac.substrates.mdl_ibps_j_discrete_categorical_mine_hybrid.emit_landing_posterior_anchor` (cathedral-queryable)
- hook #5 continual-learning posterior update: ACTIVE via canonical
  L2 helper auto-emission per
  `tac.substrates._shared.posterior_emission_helper`
- hook #6 probe-disambiguator: N/A at L0 SCAFFOLD shell

## Discipline applied

- Catalog #229 PV (read Z6 reference + canonical helper Protocol + J L0 scaffold + J mlx_renderer before edit)
- Catalog #117/#157/#174 canonical serializer (commit follows; POST-EDIT --expected-content-sha256)
- Catalog #119 Co-Authored-By
- Catalog #110/#113 APPEND-ONLY (NEW files; no mutation of J L0 SCAFFOLD source)
- Catalog #208 docs/local-paths (relative paths only)
- Catalog #220 substrate L1+ operational mechanism (substrate's archive bytes consumed by inflate.py per L0 SCAFFOLD)
- Catalog #240 recipe-vs-trainer-state consistency (adapter+trainer honestly raise NotImplementedError)
- Catalog #265 canonical contract pattern (SPDX MIT + narrow __all__)
- Catalog #287 placeholder rejection (rationales are substantive)
- Catalog #290 substrate canonical-vs-unique decision per layer (documented in adapter module docstring)
- Catalog #305 observability surface (canonical L2 helper auto-exposes 6-facet observability)
- Catalog #323 canonical Provenance (canonical helper auto-stamps)
- Catalog #335 cathedral consumer auto-discovery via Wave #1 posterior emission (inherited)
- Catalog #341 Tier A non-promotable markers per MLX-first doctrine

## Cross-references

- Z6 reference template: `src/tac/substrates/time_traveler_l5_z6/long_training_adapter.py` (commit `ab4df5d4e`)
- Z6 L2 trainer reference: `experiments/train_substrate_z6_predictive_coding_mlx_l2.py`
- Canonical L2 helper: `src/tac/training/long_training_canonical.py` (commit `f5e4784ef`)
- Cascade doctrine: `.omx/research/path_3_canonical_substrate_development_cascade_doctrine_20260526.md` (commit `fb270e9b6`)
- Tier1-T3-OP7-OP8 amendment: commit `b96418424`
- J L0 SCAFFOLD: `.omx/research/path_3_j_mdl_ibps_L0_scaffold_landed_20260526.md`
- J R1'' CLEAN: `.omx/research/path_3_j_recursive_adversarial_review_r1_prime_prime_3_axis_20260526.md`
- Lane: `lane_path_3_l1_promotion_cascade_b_prime_c_prime_e_g_j_canonical_l2_helper_20260526` L1
