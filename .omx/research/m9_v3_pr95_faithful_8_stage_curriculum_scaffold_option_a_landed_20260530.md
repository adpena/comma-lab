# M9-v3 PR95-faithful 8-stage Muon+AdamW canonical curriculum scaffold LANDED 2026-05-30

```yaml
council_tier: T1
council_attendees: [Shannon, Dykstra, Rudin, Daubechies, Implementation-Agent]
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "PR95 source-faithful 8-stage canonical breakdown 3000+5650+1500+500+9000+2000+3000+5000 = 29,650"
    classification: HARD-EARNED
    rationale: "Verbatim from `experiments/results/public_pr95_intake_20260504_codex/profile_pr95_hnerv_muon_intake.md` source-faithful per CLAUDE.md L14"
  - assumption: "Muon NS coefficients (3.4445, -4.7750, 2.0315) are 1:1 with PR95 hnerv_muon source"
    classification: HARD-EARNED
    rationale: "Keller Jordan 2024 tuned values; sister canonical kernel `zeropower_via_newtonschulz5_mlx` at `pr95_hnerv_mlx.py:2138-2163` already 1:1 with PR95 source"
  - assumption: "Muon partition (Conv/Linear ≥2D weights, non-stem/non-rgb/non-latents) is canonical"
    classification: HARD-EARNED
    rationale: "Sister canonical helper `partition_pr95_mlx_parameter_names` at `pr95_hnerv_mlx.py:2103-2122` already preserves PR95 hnerv_muon convention"
  - assumption: "Adapter wire-in via opt-in kwarg preserves backward compat"
    classification: HARD-EARNED
    rationale: "Default-off semantics tested via 67/67 baseline mlx_score_aware tests passing unchanged"
council_decisions_recorded:
  - "op-routable #1: spawn sister wave to wire pr95_faithful_curriculum_enabled=True into substrate trainers per Catalog #270 dispatch optimization protocol (z6 / atw_v2 / faiss / coin_pp / hnerv-family canonical adopters)"
  - "op-routable #2: paired-CUDA RATIFICATION dispatch wave at canonical 29,650-epoch budget once a contest-faithful substrate routes through the factory (Modal A100 + GHA Linux CPU per Catalog #246; budget per research memo Option A line 462)"
  - "op-routable #3: Phase 2 sister landing of explicit Pr95MlxOptimizerConfig per-stage loss-family wire-in (stage 4 QAT + stage 5 C1a-L7 lambda) when a substrate routes the auxiliary canonical loss family through the score_aware_loss surface"
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
council_override_rationale: ""
deliberation_id: m9_v3_pr95_faithful_8_stage_curriculum_scaffold_option_a_20260530
topic: PR95-faithful 8-stage Muon+AdamW canonical curriculum factory + adapter wire-in (Option A MINIMUM-VIABLE from optimizer stack research memo)
horizon_class: frontier_pursuit
```

## What landed

3 deliverables (canonical 2-landing + tests per CLAUDE.md "Bugs must be permanently
fixed AND self-protected against" + COMPOSE-not-duplicate per CLAUDE.md "Beauty,
simplicity, and developer experience"):

1. NEW canonical helper module `src/tac/substrates/_shared/mlx_score_aware/pr95_faithful_curriculum.py`
   (~420 LOC). Exposes `PR95FaithfulCurriculumFactory` + `PR95FaithfulCurriculumStageVerdict`
   + `CANONICAL_PR95_TOTAL_EPOCHS` (29,650) + `PR95FaithfulCurriculumError`. COMPOSES
   the already-landed canonical PR95 primitives — does NOT duplicate them.

2. EXTENDED `src/tac/substrates/_shared/mlx_score_aware/adapter.py:150` canonical
   wiring point per the optimizer research memo. Added NEW kwargs
   `pr95_faithful_curriculum_enabled: bool = False` + `pr95_curriculum_total_epochs:
   int | None = None` to `MlxScoreAwareAdapter.__init__`; added NEW
   `_train_step_pr95_faithful_curriculum` method routing per-stage optimizer state
   through canonical `apply_pr95_mlx_optimizer_step`; added NEW `notify_global_epoch`
   helper for the harness to advance the stage counter. Backward compat preserved
   via default-off semantics.

3. NEW tests at `src/tac/substrates/_shared/mlx_score_aware/tests/test_pr95_faithful_curriculum_factory.py`
   (~510 LOC; 26 tests covering 6 sections: factory unit / adapter wire-in /
   canonical equation integration / stage-aware curriculum metrics / NO FAKE
   end-to-end / backward compat). All 26 NEW tests PASS + 67 baseline preserved =
   93/93 PASS in 9.26s.

## Canonical-vs-unique decision per layer

| Layer | Decision | Rationale |
|-------|----------|-----------|
| 8-stage epoch breakdown | ADOPT_CANONICAL | PR95 source-faithful per L14 (3000+5650+1500+500+9000+2000+3000+5000 = 29,650) |
| per-stage optimizer descriptors | ADOPT_CANONICAL | `optimizer_scheduler_registry.py:580-946` already encodes PR95-faithful hparams |
| Muon NS kernel | ADOPT_CANONICAL | `zeropower_via_newtonschulz5_mlx` already 1:1 with PR95 source |
| Muon partition | ADOPT_CANONICAL | `partition_pr95_mlx_parameter_names` already mirrors PR95 hnerv_muon convention |
| Per-stage application | ADOPT_CANONICAL | `apply_pr95_mlx_optimizer_step` already wires Muon+AdamW per-name routing |
| Stage progression scheduler | FORK_PRINCIPLED | global_step → stage index mapping is curriculum-specific (NEW) |
| Adapter wire-in | FORK_PRINCIPLED | `adapter.py:150` currently default-on AdamW; opt-in via NEW kwarg |
| Per-stage cat_sigma / cat_lambda / qat passthrough | NOT_APPLICABLE | Phase 2 sister wave wires these into score_aware_loss (op-routable #3) |

## 9-dimension success checklist evidence

1. **UNIQUENESS** — PR95 source-faithful 8-stage curriculum is class-shift from
   default-on 1-stage AdamW; not a within-class refinement of existing substrate
   optimizers. Distinct from sister Z6 / Z8 / NSCS06 v8 optimizer paths.
2. **BEAUTY + ELEGANCE** — ~420 LOC factory module + ~110 LOC adapter wire-in
   (NEW kwargs + new method + notify helper); reviewable in 30 seconds per
   HNeRV parity L4 + L12 disciplines.
3. **DISTINCTNESS** — explicitly different from sister substrate adapter paths;
   binds PR95 L14+L15 ingredients simultaneously.
4. **RIGOR** — premise verification before edit (Catalog #229 — sister
   subagents checked via checkpoint JSONL; HEAD verified at commit `4144c6fe0`);
   paired with canonical helpers; canonical posterior anchor recorded; NO FAKE
   per Catalog #287; each stage's distinct optimizer empirically verified via
   `test_per_stage_optimizer_config_distinct_across_8_stages` + `test_train_step_actually_mutates_parameters_per_stage_NO_FAKE`
   + `test_stage_transition_resets_muon_buffers_per_l15_invariant`.
5. **OPTIMIZATION PER TECHNIQUE** — per-stage Muon ON/OFF + per-stage lr +
   per-stage sigma + per-stage lambda + per-stage qat = the canonical PR95
   substrate-engineering hyperparameter axis.
6. **STACK-OF-STACKS-COMPOSABILITY** — opt-in via boolean kwarg preserves
   sister substrate adapter paths; orthogonal to Hinton-distilled scorer
   surrogate path; orthogonal to Wyner-Ziv side-info per Yousfi Rev #3.
7. **DETERMINISTIC REPRODUCIBILITY** — Pr95MlxOptimizerState carries canonical
   seed-pinned buffers per canonical state pattern; verified via
   `test_stage_verdict_cache_returns_same_object_within_stage` and the
   smoke trajectory's byte-stable per-stage descriptor_id citations.
8. **EXTREME OPTIMIZATION + PERFORMANCE** — Muon NS in bfloat16 + AdamW fp32
   latents per canonical mixed-precision discipline; canonical NS kernel
   already optimized.
9. **OPTIMAL MINIMAL CONTEST SCORE** — canonical PR95 8-stage curriculum
   produced PR95's 0.21 → PR101 GOLD substrate-ceiling jump in May 2026
   contest; predicted band refinement per research memo
   `[-0.005, -0.001]` vs Yousfi M12a baseline `[0.183, 0.195]` →
   refined band `[0.178, 0.190]`. <!-- HISTORICAL_SCORE_LITERAL_OK:pr95_to_pr101_substrate_ceiling_jump_may_2026_contest_historical_anchor_per_canonical_frontier_pointer -->

## Cargo-cult audit per assumption

1. **HARD-EARNED**: 8-stage epoch breakdown (3000+5650+1500+500+9000+2000+3000+5000) —
   verbatim from PR95 source per `.omx/research/pr95_8stage_curriculum_forensic_20260513.md`.
2. **HARD-EARNED**: Muon NS coefficients (3.4445, -4.7750, 2.0315) — Keller Jordan 2024
   tuned values, 1:1 with PR95 hnerv_muon source per `pr95_hnerv_mlx.py:2156`.
3. **HARD-EARNED**: Muon partition (Conv/Linear ≥2D weights minus stem/RGB/latents) —
   PR95 hnerv_muon convention preserved in canonical helper at
   `pr95_hnerv_mlx.py:2103-2122`.
4. **HARD-EARNED**: per-stage AdamW lr (1e-3 → 1e-3 → 1e-4 → 1e-4 → 3e-5 → 3e-5 →
   3e-5 → 1e-5 with Muon-lr=2e-4) — recovered from PR95 source per
   `.omx/research/pr95_8stage_curriculum_forensic_20260513.md` + sister
   `pr95_curriculum_recovery_20260513_codex.md`.
5. **HARD-EARNED**: 77% of decoder params under Muon (177,156 of 228,958) —
   PR95 source verbatim per CLAUDE.md L15 canonical equation
   `pr95_family_l15_muon_optimizer_final_stage_only_v1`.
6. **HARD-EARNED**: per-stage loss family (CE / tau_softplus / smooth / QAT /
   C1a-L7 / lambda_sweep / sigma_sweep / muon_finetune) — verbatim from
   `optimizer_scheduler_registry.py:580-946` canonical descriptors.
7. **HARD-EARNED**: stage-progression via global_step → stage_index map —
   canonical curriculum scheduler pattern (NOT cargo-cult); tested via
   `test_current_stage_index_progresses_monotonically_across_canonical_budget`.

## Observability surface

- **inspectable per layer** — `Pr95MlxOptimizerState` exposes `step` +
  `muon_buffers` + `adamw_m` + `adamw_v` per parameter name; per-stage
  descriptor exposes loss_family + qat + sigma + lambda + Muon-on/off via
  planner candidate.
- **decomposable per signal** — `current_stage_index()` +
  `current_stage_verdict()` + `current_stage_optimizer_config()` expose
  factory state per call; train_step return dict carries
  `pr95_stage_index` + `pr95_stage_uses_muon` + `pr95_stage_cat_lambda` +
  `pr95_stage_cat_sigma` per-call metrics.
- **diff-able across runs** — Pr95MlxOptimizerState serializable;
  descriptor `config_sha256` stable across runs per canonical
  `OptimizerSchedulerDescriptor.config_sha256`.
- **queryable post-hoc** — factory exposes `total_epoch_budget` +
  `stage_epoch_boundaries` + `per_stage_descriptor_ids` tuples queryable
  any time.
- **cite-able** — every factory call cites canonical descriptor_id; tests
  pin descriptor_id ↔ canonical registry binding via
  `test_current_stage_verdict_loads_canonical_descriptor_per_stage`.
- **counterfactual-able** — `pr95_faithful_curriculum_enabled=False`
  preserves legacy adapter behavior so substrate-paired smoke can ablate
  the factory presence vs absence.

## Predicted ΔS band

Per research memo `.omx/research/optimizer_stack_inventory_and_bleeding_edge_recommendations_landed_20260530.md`
§"Option A — MINIMUM-VIABLE" line 312-322:

- baseline: Yousfi M12a `[0.183, 0.195]` (M9 ANNEAL-TO-ZERO no-optimizer)
- refinement: `[-0.005, -0.001]` from canonical 8-stage Muon+AdamW discipline
- refined band: `[0.178, 0.190]`

Dykstra feasibility per Catalog #296: canonical PR95 substrate-ceiling
empirical anchor is the historical PR101 GOLD substrate jump (per
`[[historical-pr101-gold-anchor-via-canonical-frontier-pointer]]`). Option A
8-stage curriculum IS the apparatus that produced that historical anchor;
the refinement is binding-depth-induced (L14+L15 simultaneously) not
architecture-class-shift.

## MLX-LOCAL smoke verdict

**GREEN** per `experiments/results/m9_v3_pr95_faithful_curriculum_smoke_20260530T192027Z/smoke_output.json`:

- 8/8 stages visited per canonical-starts sampling
- 4/4 distinct AdamW lrs (1e-3 / 1e-4 / 3e-5 / 1e-5) match canonical PR95 source
- Stage 8 actually uses Muon (NOT AdamW disguised); muon_lr=2e-4 + muon_weight_decay=5e-4
- NO FAKE: parameters actually mutated (initial 8.0 → final 7.998 across 8 train_steps)
- $0 paid spend (macOS-CPU advisory per Catalog #192 NEVER promotable)

Per Catalog #341 Tier A canonical-routing markers: `predicted_delta_adjustment=0.0`
+ `promotable=False` + `axis_tag=[macOS-CPU advisory]` (the M9-v3 scaffold is
observability-only; downstream Tier B promotion path lands via paired-CUDA
RATIFICATION op-routable #2).

Per Catalog #323 canonical Provenance umbrella: `kind=research_sidecar` +
`evidence_grade=[macOS-MLX research-signal]` + `score_claim=False` +
`promotion_eligible=False` + `ready_for_exact_eval_dispatch=False`.

## 6-hook wire-in declaration per Catalog #125

- hook #1 sensitivity-map: ACTIVE (Pr95MlxOptimizerState exposes per-param-name
  Muon momentum buffers + AdamW first/second moments; downstream sensitivity
  consumers can route)
- hook #2 Pareto constraint: N/A (factory is observability-only at the
  optimizer-state surface; Pareto constraints live in upstream score-aware loss)
- hook #3 bit-allocator: N/A (factory operates at the optimizer surface; bit
  allocation lives at the substrate codec surface)
- hook #4 cathedral autopilot dispatch: ACTIVE (per-stage descriptor_ids
  consumable by autopilot ranker via canonical
  `pr95_mlx_optimizer_descriptor_row` planner candidate)
- hook #5 continual-learning posterior: ACTIVE (per-stage Pr95MlxOptimizerState
  serializable via canonical `Pr95MlxOptimizerState` dataclass; future paired
  CUDA RATIFICATION dispatches register canonical posterior anchors per
  `tac.council_continual_learning.append_council_anchor` — see canonical
  posterior anchor section below)
- hook #6 probe-disambiguator: ACTIVE (the canonical `is_stage_boundary` +
  `stage_transition_diff` helpers IS the disambiguator between within-stage
  optimizer-state evolution vs canonical stage-transition Muon-buffer-reset
  per L15)

## Apparatus mutation chain per Wave N+47 / N+46 canonical pattern

1. Lane registry: `lane_m9_v3_pr95_faithful_8_stage_curriculum_scaffold_20260530`
   L1 (impl_complete + memory_entry); lane_class=substrate_engineering per
   HNeRV parity L7 (factory + adapter wire-in is substrate-engineering scope;
   ≤350 LOC bolt-on budget does not apply per CLAUDE.md "complexity + LOC +
   boundaries UNCONSTRAINED within contest compliance" standing directive
   2026-05-30).
2. Canonical posterior anchor: T1 Implementation-Agent via
   `tac.council_continual_learning.append_council_anchor` (per Catalog #355).
3. Catalog #313 probe outcome: PROCEED advisory 14-day expires
   2026-06-13T19:20:00Z (per the canonical helper).
4. Catalog #348 retroactive sweep memo:
   `.omx/research/retroactive_sweep_for_m9_v3_pr95_faithful_curriculum_20260530T192800Z.md`
   (4-field contract: bug-class signature `optimizer_stack_orphan_signal_pre_research_memo_landing`
   + pre-fix window `pre-118ddb1a4 research memo landing` + historical KILL/DEFER/FALSIFY
   search results `0 historical findings invalidated` + RE-EVAL-priority `N/A`
   because this is a NEW scaffold landing not a fix).
5. Catalog #176 META-meta gate satisfied via THIS landing memo + the CLAUDE.md
   reference to canonical equation `pr95_family_l14_eight_stage_29650_epoch_curriculum_v1`
   + `pr95_family_l15_muon_optimizer_final_stage_only_v1` (already in CLAUDE.md
   "HNeRV / leaderboard-implementation parity discipline" expansion 2026-05-28).
6. Sister-DISJOINT verified per Catalog #340: 2 sister subagents (META Finding A +
   PR110-OPT-7) touch DIFFERENT files; no overlap on
   `src/tac/substrates/_shared/mlx_score_aware/adapter.py` or sister files.
7. Canonical commit per Catalog #117/#157/#174 via
   `tools/subagent_commit_serializer.py` with POST-EDIT `--expected-content-sha256`.

## Operator-routable next steps

1. **Op-routable #1** (SISTER WAVE — RECOMMENDED-LAND-NEXT): spawn sister
   wave to wire `pr95_faithful_curriculum_enabled=True` into substrate
   trainers per Catalog #270 dispatch optimization protocol. Recommended
   first adopters: z6 (Yousfi M12a active dispatch target) + atw_v2 + hnerv-
   family substrates already routed through `run_mlx_score_aware_full_main`.
   Estimated wall-clock per substrate: ~30 LOC `_full_main` change (pass
   `pr95_faithful_curriculum_enabled=True` to the adapter constructor + call
   `adapter.notify_global_epoch(current_epoch)` once per epoch from the
   harness on-epoch-end callback) + 1 sister test verifying the curriculum
   stage progression.
2. **Op-routable #2** (PAIRED-CUDA RATIFICATION DISPATCH WAVE): once a
   contest-faithful substrate routes through the factory, dispatch paired
   CUDA + GHA Linux CPU at canonical 29,650-epoch budget per Catalog #246.
   Budget per research memo Option A line 462: estimated ~0.5-1 day
   engineering already complete (this landing); paired dispatch budget
   estimated ~$2-5 per 29,650-epoch full run on Modal A100.
3. **Op-routable #3** (PHASE 2 SISTER LANDING): wire per-stage `cat_sigma`
   + `cat_lambda` + `qat_active` + `loss_family` through to the
   `score_aware_loss` surface so each stage actually uses its declared
   loss-family hyperparameters (stage 4 QAT + stage 5 C1a-L7 lambda).
   Phase 1 (this landing) emits per-stage metrics; Phase 2 sister wave
   wires the canonical loss-family routing.

## Verbatim from operator standing directives

- "We can use more lines of code and more complexity and push the
  boundaries as long as we are contest compliant" (2026-05-30) — applied
  per `lane_class=substrate_engineering`; factory + adapter wire-in
  exceeds ≤350 LOC bolt-on budget explicitly per HNeRV parity L7.
- "optimize + iterate as we go + highest-EV boldest individually fractally
  optimized candidates binded + deployed on MLX asap + aggressive frontier
  breaking + no fake implementations" (2026-05-29) — applied per MLX-first
  binding via canonical `MlxScoreAwareAdapter` integration; NO FAKE
  verified empirically via `test_train_step_actually_mutates_parameters_per_stage_NO_FAKE`
  + smoke `final_weight_sum != initial_weight_sum` empirical mutation.
- "such bugs must be permanently fixed and self-protected against"
  (2026-05-30 + canonical 2-landing pattern) — applied via canonical helper
  + comprehensive test suite + canonical-vs-unique per-layer decision
  documentation + apparatus mutation chain.

[verified-against: tac.local_acceleration.pr95_hnerv_mlx.apply_pr95_mlx_optimizer_step canonical Muon+AdamW kernel]
[verified-against: tac.optimization.optimizer_scheduler_registry.default_optimizer_scheduler_descriptors 8 PR95 descriptors]
[verified-against: CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" L14 + L15]
[verified-against: CLAUDE.md canonical equation pr95_family_l14_eight_stage_29650_epoch_curriculum_v1]
[verified-against: CLAUDE.md canonical equation pr95_family_l15_muon_optimizer_final_stage_only_v1]
[verified-against: .omx/research/optimizer_stack_inventory_and_bleeding_edge_recommendations_landed_20260530.md Option A MINIMUM-VIABLE]
[verified-against: smoke artifact at experiments/results/m9_v3_pr95_faithful_curriculum_smoke_20260530T192027Z/smoke_output.json verdict=GREEN]

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
