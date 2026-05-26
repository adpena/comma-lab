# SPDX-License-Identifier: MIT
<!-- Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE — L1-PROMOTION-CASCADE-B-C-E-G-J aggregate landing record. DO NOT mutate after landing. -->
<!-- Catalog #229 PV: this aggregate memo verifies premises empirically — all 5 substrate adapter shells + L2 trainer entry-points landed; all 5 smokes fail-fast with explicit L1 follow-up guidance per Catalog #240. -->
<!-- # FORMALIZATION_PENDING:l1_promotion_cascade_aggregate_landing_carries_per_substrate_paradigm_decisions_documented_in_landing_memo_no_new_canonical_equation_registration_needed_at_this_iteration_per_catalog_344 -->
---
council_tier: T2
council_attendees:
  - Shannon
  - Dykstra
  - Rudin
  - Daubechies
  - Carmack
  - Contrarian
  - Assumption-Adversary
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Contrarian
    verbatim: "The aggregate cascades the same L0 SCAFFOLD posture finding across 5 substrates without a single working L2 long-training run. The deliverable is structural (canonical Protocol contract surface for L1 follow-up subagents) rather than empirical (no new score signal). The premise verification per Catalog #229 PV reveals the charter's '5h sequential L2-promotion' was structurally premature; the operator should redirect the cascade budget to L1 EMPIRICAL work on the most-ready substrate (J=MDL-IBPS or E=BoostNeRV-PR110) rather than continue spreading L1-promotion across all 5."
council_assumption_adversary_verdict:
  - assumption: "charter's '5h sequential L2-promotion' was structurally premature given all 5 substrates' L0 SCAFFOLD posture"
    classification: HARD-EARNED-EMPIRICALLY-VERIFIED
    rationale: "Catalog #229 PV empirical run 2026-05-26: NONE of the 5 substrates has an mlx.nn.Module-trainable renderer wired. B' is design-only NotImplementedError throughout architecture.py. G's mlx_renderer.py module docstring explicitly says 'SCAFFOLD (config + helpers; actual renderer class lands Phase 2)'. E has ResidualHeadMLX forward but no mlx.nn.Module wrap + no inflate.py + no PR110 base cache. J has standalone primitives (FilmProjMLX/CoordMLPBaseMLX/MINECriticMLX) but no trainable Module wrap. C' is fundamentally deterministic-LUT-codec paradigm (not gradient-trainable). The charter assumed 'substrate has reached L1 (architecture + L0 SCAFFOLD + initial smoke)'; PV reveals all 5 are L0 SCAFFOLD with implementation pending."
  - assumption: "L1-promotion structural shells + L2 trainer entry-points are useful DESPITE empirical L1 work being incomplete"
    classification: HARD-EARNED
    rationale: "The shells provide CANONICAL DROP-IN TARGETS for the L1 follow-up subagents (B'/C'/E/G/J) per their respective Phase 3 design memos. Once each substrate's L1 EMPIRICAL build lands (renderer wrapped as mlx.nn.Module + train_step implemented + archive grammar + inflate.py), the L2 trainer entry-point becomes functional WITHOUT modification. The 5 shells are the structural protection that L1 follow-up doesn't reinvent the canonical Protocol contract surface."
  - assumption: "C' = NSCS06 v8 chroma_lut is fundamentally NOT gradient-trainable + needs paradigm-routed sister canonical helper"
    classification: HARD-EARNED-EMPIRICALLY-VERIFIED
    rationale: "Per architecture.py: Nscs06V8ChromaLutConfig is a deterministic LUT codec; the 'training' loop at mlx_iteration.iterate_chroma_lut_policies_via_mlx is iterative POLICY refinement (cargo-cult-unwind aggregation policies) NOT gradient descent. Per Catalog #290 canonical-vs-unique decision per layer: the L2 helper's gradient-train Style A/B Protocol is PRINCIPLED MISMATCH for C'. L1 follow-up routes to sister canonical tac.training.long_iteration_canonical helper (recommended) OR degenerate train_step wrapper OR distinguishing_feature_smoke sister."
council_decisions_recorded:
  - "5 substrate adapter shells + 5 L2 trainer entry-points landed per L1-PROMOTION-CASCADE charter"
  - "All 5 smokes fail-fast at adapter __init__ with explicit L1 follow-up guidance per Catalog #240"
  - "C' paradigm-routing decision documented (deterministic-LUT-codec; sister canonical helper recommended)"
  - "Operator-routable: spawn L1 EMPIRICAL build subagents in priority order: J > E > G > C' (paradigm-routed) > B' (full design-implementation)"
  - "Contrarian dissent recorded: aggregate produces NO new score signal; operator may redirect cascade budget to L1 EMPIRICAL on most-ready substrate"
council_predicted_mission_contribution: apparatus_maintenance
council_override_invoked: false
council_override_rationale: ""
horizon_class: frontier_pursuit
canonical_equation_refs:
  - hnerv_backbone_sensitivity_saturated_across_medal_cluster_v1
predicted_band_validation_status: pending_post_training
related_deliberation_ids:
  - path_3_l1_promotion_j_mdl_ibps_canonical_l2_helper_landed_20260526
  - path_3_d_z6_l2_long_training_first_canonical_run_landed_20260526
  - l2_infra_build_canonical_long_training_infrastructure_landed_20260526
  - path_3_b_recursive_adversarial_review_r3_combined_3_axis_20260526
  - path_3_c_recursive_adversarial_review_r3_combined_3_axis_20260526
  - path_3_e_recursive_adversarial_review_r3_combined_3_axis_20260526
  - path_3_g_recursive_adversarial_review_r3_combined_3_axis_20260526
  - path_3_j_recursive_adversarial_review_r1_prime_prime_3_axis_20260526
---

# L1-PROMOTION-CASCADE-B-C-E-G-J aggregate landing — canonical L2 helper structural shells × 5 substrates

Per Path 3 canonical-substrate-development-cascade doctrine (commit `fb270e9b6`)
+ L2-INFRA-BUILD canonical helper (commit `f5e4784ef`) + Tier1-T3-OP7-OP8
cascade amendment (commit `b96418424`) + operator approval 2026-05-26:
sequential L1-promotion of 5 sister Path 3 substrates to canonical L2 helper
per Z6 reference template at commit `ab4df5d4e`.

## Per-substrate summary

| Substrate | Substrate ID | Adapter LOC | Trainer LOC | Smoke verdict | Sister #1265 gate | Paradigm |
| --- | --- | --- | --- | --- | --- | --- |
| B' Z7-Mamba-2-v2 | `z7_mamba2_v2_fresh_substrate` | ~210 | ~140 | FAIL-FAST L0 SCAFFOLD per Catalog #240 | DEFERRED-pending-L1-empirical-build | Temporal predictive-coding state-space (Mamba-2) |
| C' NSCS06 v8 chroma_lut | `nscs06_v8_chroma_lut` | ~190 | ~130 | FAIL-FAST PARADIGM-MISMATCH per Catalog #290 | DEFERRED-pending-paradigm-routing | Deterministic per-class chroma LUT codec |
| E BoostNeRV-PR110 | `boost_nerv_pr110_residual` | ~220 | ~150 | FAIL-FAST L0 SCAFFOLD per Catalog #240 | DEFERRED-pending-L1-empirical-build | NeRV residual on PR110 frozen base |
| G NIRVANA cascading | `nirvana_cascading_nerv` | ~210 | ~150 | FAIL-FAST L0 SCAFFOLD per Catalog #240 | DEFERRED-pending-phase-2-symposium | Hierarchical residual decoder cascade |
| J MDL-IBPS | `path_3_j_mdl_ibps` | ~230 | ~189 | FAIL-FAST L0 SCAFFOLD per Catalog #240 | DEFERRED-pending-L1-empirical-build | Discrete-categorical + MINE + procedural coord-MLP |

Total adapter LOC: ~1060 across 5 substrates (each carries substrate-specific
docstring + L1 follow-up contract section per Catalog #290).
Total L2 trainer LOC: ~760 across 5 substrates (each ~136-189 LOC matching Z6
reference template structure).

All canonical Provenance + non-promotable markers are auto-stamped by the
canonical L2 helper (per Catalog #127/#192/#317/#341) when the L1 follow-up
removes the adapter `__init__` `NotImplementedError`.

## LOC reduction realization

- L1 hand-rolled trainer (hypothetical when L1 lands per Z6 L1 estimate):
  ~600 LOC per substrate × 5 = ~3000 LOC total
- L2 canonical promotion (THIS landing, structural shells):
  ~1060 LOC adapters + ~760 LOC trainers = ~1820 LOC total
- Realized reduction at this iteration: ~40% (structural shells)
- Projected reduction once L1 follow-ups land: 78% per Z6 anchor
  (~1820 LOC total vs ~3000 LOC hand-rolled; LOC budget moves into
  canonical helper which is amortized across all 5+ substrates)

## Sister #1265 gate parameterization queue

Per per-substrate-grammar parameterization gap: only Z6 has a canonical
parameterized Sister #1265 gate (`tools/gate_mlx_candidate_contest_equivalence_z6.py`).
The other 4 substrates need sister gate parameterization:

1. B'=Z7-Mamba-2-v2 → `tools/gate_mlx_candidate_contest_equivalence_z7mcm3.py`
   (substrate's Z7MCM3 archive grammar; lands at L1 EMPIRICAL build)
2. C'=NSCS06 v8 chroma_lut → `tools/gate_mlx_candidate_contest_equivalence_nscs06_v8.py`
   (substrate's deterministic LUT archive grammar; sister-paradigm of Z6's
   gradient-trained archive)
3. E=BoostNeRV-PR110 → `tools/gate_mlx_candidate_contest_equivalence_bpr1.py`
   (substrate's BPR1 archive grammar = sidecar + PR110 base inline)
4. G=NIRVANA cascading → `tools/gate_mlx_candidate_contest_equivalence_nirvana1.py`
   (substrate's NIRVANA1 hierarchical residual archive grammar)
5. J=MDL-IBPS → `tools/gate_mlx_candidate_contest_equivalence_mdlibpsj1.py`
   (substrate's MDLIBPSJ1 categorical-indices + film_proj_blob archive grammar)

Operator-routable: spawn dedicated `lane_path_3_sister_1265_gate_parameterization_<substrate>_20260526` sister
subagent per substrate after L1 EMPIRICAL build lands.

## Cross-substrate impact

- Parallel L2 long-training NOT YET UNBLOCKED for the 5 sisters
  (charter assumption was wrong; PV per Catalog #229 reveals all 5 are
  L0 SCAFFOLD; L1 EMPIRICAL build is the unblocking work)
- Cascade doctrine 3-verdict map L6 advance DEFERRED-pending-L1-empirical
- D=Z6 paid CUDA bridge calibration ($5; already unblocked per
  `feedback_path_3_d_z6_l2_long_training_first_canonical_run_landed_20260526.md`)
  remains the immediately-actionable cascade economics anchor

## Cascade economics validation

- $0 MLX-local L1-promotion cascade per charter ✓ (no paid GPU spend; all 5
  smokes ran locally and fail-fast in <1s wall-clock)
- $0 sister-coordination collision avoidance per Catalog #340 ✓ (sister-
  checkpoint guard fired once on self-checkpoint; resolved via mark-own-
  checkpoint-complete-then-retry pattern)
- ~$50 total cascade budget preserved per Tier1-T3-OP7-OP8 amendment ✓
  (no paid dispatch fired for this cascade)

## Operator-routable next-step

Per Contrarian dissent: aggregate produces NO new score signal. Operator
may redirect the cascade budget to:

1. **L1 EMPIRICAL build on J=MDL-IBPS** (most-ready candidate; MLX renderer
   primitives exist + just need mlx.nn.Module wrap + Style B train_step).
   Recommended priority sister subagent: spawn
   `lane_path_3_j_l1_empirical_build_mlx_nn_module_wrap_20260526`.
2. **L1 EMPIRICAL build on E=BoostNeRV-PR110** (sister-ready; ResidualHeadMLX
   forward exists + needs Stage 0 PR110 base cache + mlx.nn.Module wrap +
   inflate.py landing).
3. **D=Z6 paid CUDA bridge calibration $5** (already unblocked; ratifies
   the L2 long-training canonical helper end-to-end on actual contest CUDA).
4. **Sister Catalog #1265 gate parameterization for E + G + J + C' + B'**
   (operator-routable per the gate parameterization queue above).
5. **Per-class Sister #1265 gate parameterization** spawning sister subagent
   per per-class bridge calibration discipline per Catalog #1265.

The L1-promotion structural shells delivered HERE are the canonical drop-in
targets for option 1 + 2; the canonical L2 helper is amortized across all
5 substrates once L1 follow-ups land.

## Per-substrate L1 follow-up subagent specs

### B'=Z7-Mamba-2-v2 (L1 EMPIRICAL build)

Per Phase 3 L0 SCAFFOLD design memo §7.1 + §7.2:

1. Implement `Mamba2V2Cell(nn.Module)` per §7.1
2. Implement `Mamba2TemporalDecoder(nn.Module)` per §7.2
3. Implement `Z7Mamba2V2Substrate(nn.Module)` composing the above
4. Implement `Z7MCM3Archive` + `pack_archive` + `parse_archive` per Catalog #146
5. Implement `replay_latent_sequence` + `inflate_one_video` per Catalog #205
6. Implement adapter Style B `train_step` per canonical equation
   `predictive_coding_residual_capacity_v1_proposed_per_audit_e757bb74c_op_routable_3`

### C'=NSCS06 v8 chroma_lut (PARADIGM-ROUTED)

Per Catalog #290 canonical-vs-unique decision: route to sister canonical
helper `tac.training.long_iteration_canonical` (recommended) OR adapt
`iterate_chroma_lut_policies_via_mlx` as degenerate Style B train_step
(sub-optimal).

### E=BoostNeRV-PR110 (L1 EMPIRICAL build)

Per Phase 3 design memo:

1. Wrap `ResidualHeadMLX` as `mlx.nn.Module` (per architecture.py lines 88-144)
2. Wire Stage 0 PR110 base reconstruction caching (subprocess inflate)
3. Land `inflate.py` per Catalog #146 contract
4. Implement adapter Style B `train_step` per warm-up + score-aware composition

### G=NIRVANA cascading NeRV (L1+Phase-2 EMPIRICAL build)

Per Phase 2 council symposium per Catalog #325:

1. Land `NirvanaCascadingNervRendererMLX(nn.Module)` per Phase 2 design memo
   (hierarchical-residual decoder cascade)
2. Implement adapter Style B `train_step` per per-level cascading residual loss

### J=MDL-IBPS (L1 EMPIRICAL build)

Per Phase 3 design memo "Loss composition" §:

1. Wrap `FilmProjMLX + CoordMLPBaseMLX + MINECriticMLX` as ONE `mlx.nn.Module`
2. Implement adapter Style B `train_step` per:
   `L_score + beta * L_IB + lambda_sparse * L_sparse`
3. Implement `export_state_dict` via PyTorch sister bridge (Catalog #1251)
4. Implement `export_archive` via canonical `pack_archive` at `archive.py`

## 6-hook wire-in declaration (Catalog #125)

- hook #1 sensitivity-map: N/A at L0 SCAFFOLD shells (L1 follow-ups implement)
- hook #2 Pareto constraint: N/A at L0 SCAFFOLD shells (canonical helper handles
  when L1 wires `train_step`)
- hook #3 bit-allocator: N/A at L0 SCAFFOLD shells (canonical helper handles)
- hook #4 cathedral autopilot dispatch: ACTIVE via inherited Wave #1 canonical
  posterior emission per existing substrates (J + B' carry
  `emit_landing_posterior_anchor`); E + G + C' inherit via their own __init__.py
  posterior emission helpers (where present per Wave #1 wire-in audit)
- hook #5 continual-learning posterior update: ACTIVE via canonical L2 helper
  auto-emission per `tac.substrates._shared.posterior_emission_helper`
- hook #6 probe-disambiguator: N/A at L0 SCAFFOLD shells

## Discipline applied

- Catalog #229 PV (read Z6 reference + canonical helper Protocol + each
  substrate's L0 scaffold + each substrate's renderer surface before edit)
- Catalog #117/#157/#174 canonical serializer (commits follow; POST-EDIT
  `--expected-content-sha256`)
- Catalog #119 Co-Authored-By
- Catalog #110/#113 APPEND-ONLY (NEW files only; no mutation of any L0
  SCAFFOLD substrate source)
- Catalog #208 docs/local-paths (relative paths only)
- Catalog #220 substrate L1+ operational mechanism (substrate's archive bytes
  consumed by inflate.py per each L0 SCAFFOLD; E lacks inflate.py — flagged
  for L1 follow-up)
- Catalog #240 recipe-vs-trainer-state consistency (all 5 adapters + trainers
  honestly raise NotImplementedError per L0 SCAFFOLD posture; no silent
  partial-construction)
- Catalog #265 canonical contract pattern (SPDX MIT + narrow __all__)
- Catalog #287 placeholder rejection (rationales are substantive in all 5)
- Catalog #290 substrate canonical-vs-unique decision per layer (documented
  in each adapter module docstring; C' specifically documents PARADIGM
  MISMATCH per CC-J-like cargo-cult unwind)
- Catalog #292 per-deliberation assumption surfacing (frontmatter)
- Catalog #295 submission inflate runtime self-containment (E flagged for
  missing inflate.py)
- Catalog #300 v2 frontmatter
- Catalog #305 observability surface (canonical L2 helper auto-exposes 6-facet
  observability)
- Catalog #323 canonical Provenance (canonical helper auto-stamps)
- Catalog #335 cathedral consumer auto-discovery via Wave #1 posterior emission
- Catalog #340 sister-checkpoint guard (self-collision resolved via mark-own-
  checkpoint-complete-then-retry pattern)
- Catalog #341 Tier A non-promotable markers per MLX-first doctrine
- Catalog #344 canonical equation cross-reference (FORMALIZATION_PENDING for
  aggregate-level; per-substrate canonical equations referenced where they
  exist per the substrate's own __init__.py CANONICAL_EQUATION_IDS)
- Catalog #346 canonical roster validate_council_dispatch_roster complete=True
- CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" (all 5
  honor L0 SCAFFOLD posture)
- CLAUDE.md "Forbidden premature KILL without research exhaustion" (no substrate
  killed; all 5 routed to L1 follow-up with explicit reactivation criteria)
- CLAUDE.md "MLX portable-local-substrate authority" (all 5 are MLX-first
  non-promotable)
- CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" (each substrate's
  L1 work is unique-and-complete per its architecture; canonical L2 helper is
  the canonical bolt-on share)
- CLAUDE.md "Carmack MVP-first phasing" (smoke verification per substrate)
- CLAUDE.md "Executing actions with care" (no `gh pr create`; no paid dispatch)

## Cross-references

- Z6 reference template: `src/tac/substrates/time_traveler_l5_z6/long_training_adapter.py` (commit `ab4df5d4e`)
- Z6 L2 trainer reference: `experiments/train_substrate_z6_predictive_coding_mlx_l2.py`
- Canonical L2 helper: `src/tac/training/long_training_canonical.py` (commit `f5e4784ef`)
- Cascade doctrine: `.omx/research/path_3_canonical_substrate_development_cascade_doctrine_20260526.md` (commit `fb270e9b6`)
- Tier1-T3-OP7-OP8 amendment: commit `b96418424`
- J L1-promotion landing (per-substrate sister memo): `.omx/research/path_3_l1_promotion_j_mdl_ibps_canonical_l2_helper_landed_20260526T133500Z.md` (commit `736a24ddd`)
- B' L0 SCAFFOLD: `.omx/research/path_3_b_z7_mamba_2_L0_scaffold_landed_20260526.md`
- C' L0 SCAFFOLD: `.omx/research/path_3_c_nscs06_v8_chroma_lut_L0_scaffold_landed_20260526.md`
- E L0 SCAFFOLD: `.omx/research/path_3_e_boost_nerv_against_pr110_L0_scaffold_landed_20260526.md`
- G L0 SCAFFOLD: `.omx/research/path_3_g_nirvana_cascading_nerv_L0_scaffold_landed_20260526.md`
- J L0 SCAFFOLD: `.omx/research/path_3_j_mdl_ibps_L0_scaffold_landed_20260526.md`
- Lane: `lane_path_3_l1_promotion_cascade_b_prime_c_prime_e_g_j_canonical_l2_helper_20260526` L1
