# Tier Dispatch Authorizations — Grand Council 2026-05-14
**Sister memo of:** `.omx/research/grand_council_tiered_parallel_plan_full_authority_20260514.md`
**Tag:** `journal_grade_v1=true`; `research_only=true` (THIS memo); execution subagents listed below are EXECUTION-AUTHORIZED, not yet spawned
**Authorization date:** 2026-05-14
**Authorizing body:** Grand Council 11/11 UNANIMOUS (per parent memo Section 7.1)
**Operator delegation:** FULL DECISIONMAKING AUTHORITY granted 2026-05-14

---

## Purpose

This memo enumerates the **execution-authorized subagent assignments** approved by the Grand Council per parent memo Section 7.2-7.5. The parent memo establishes the council's plan + math + reactivation criteria; THIS memo is the concrete dispatch list the parent agent uses to spawn execution waves.

Each entry is a complete **subagent prompt scope** ready to be lifted into an `Agent` tool invocation by the parent agent.

---

## Recursive trust ladder

Every execution subagent below INHERITS the directive chain:
1. `.omx/research/recovery_session_20260514_directive_absolute_no_signal_loss_20260514.md` (original 7-rule)
2. `.omx/research/recursive_no_signal_loss_protocol_20260514.md` (R1-R4 extension)
3. `.omx/research/journal_lab_grade_documentation_standard_directive_20260514.md` (11-element)
4. `.omx/research/grand_council_tiered_parallel_plan_full_authority_20260514.md` (parent memo)
5. THIS memo (authorization scope)
6. `[[feedback_parallel_wave_dispatch_vs_editor_serialization_pattern_20260514]]` (Phase 1/2 sequencing)

Every execution subagent's prompt MUST include these references in mandatory pre-flight.

---

## WAVE STRUCTURE

**Wave 1 (Phase 1; Tier 0 only; START IMMEDIATELY):**
- Spawn ALL Tier 0 subagents simultaneously (T0-A through T0-F)
- Each writes checkpoint per Catalog #206
- Parent agent BLOCKS for `--step complete` on all 6
- Estimated wave-1 wall-clock: 2-4h

**Wave 2 (Phase 2; Tier 1 + Tier 2 startable + Tier 3 queue):**
- After Wave 1 quiescence (parent verifies via `find src tools experiments scripts -newer <T-3s>` returns empty)
- Spawn Tier 1 dispatchers simultaneously (T1-A through T1-F)
- Spawn Tier 2 candidates that DO NOT depend on Tier 1 anchors (T2-D STC pose; T2-G Dasher arithmetic pose)
- Estimated wave-2 wall-clock: 12-24h (Tier 1 fulls) + 24h (independent Tier 2)

**Wave 3 (Tier 2 dependent):**
- After Tier 1 anchors harvested
- Spawn dependent Tier 2 subagents (T2-A through T2-F)
- Estimated wave-3 wall-clock: 2-5 days

**Wave 4 (Tier 3; operator-decision-gated):**
- Surface to operator at Checkpoint 5
- Spawn ONLY after explicit per-wave operator approval (≥$200/wave threshold)

---

## TIER 0 SUBAGENT ASSIGNMENTS

### T0-A: HARVEST-INFLIGHT-AND-LANDING-AUDIT

**Subagent ID format:** `t0_a_harvest_inflight_20260514_<random>`
**Lane:** `lane_tier_0_harvest_inflight_20260514` (pre-register at L0 first)
**Budget:** $0 GPU; 1h wall-clock
**Owner scope:** `.omx/state/subagent_progress.jsonl` (append-only) + harvested artifacts under `experiments/results/recovered_*/` + Modal call_id harvest

**Mandate:**
1. Read CLAUDE.md + AGENTS.md per Catalog #206
2. Verify checkpoint discipline; write step 1
3. Harvest in-flight Modal call_ids:
   - C6 smoke `fc-01KRKBF28G2M3N73FS7PDCB6AZ` via `.venv/bin/python tools/harvest_modal_calls.py` filter
   - D4 timeout `fc-01KRKB7GFKQE8Y1JNKRYBWS3RJ` (rc=124 expected)
   - D4 A10G `fc-01KRKA5DA13RH1CP5BQNDVAM3C` (rc=124 sister)
4. For each harvest result, write `harvested_artifacts/` next to `modal_metadata.json`
5. Audit `tools/claim_lane_dispatch.py summary` — confirm 5 active claims (4 codex hdm8 + 1 Z3 pending); mark any landed as terminal
6. Surface to next subagent: archived artifact paths + score-extraction if available
7. Checkpoint `--step complete`

**Deferral:** if harvest finds non-recoverable corruption, surface forensic blocker; do not invent data
**Crash-resume:** read latest checkpoint via `tools/subagent_checkpoint.py read --subagent-id t0_a_harvest_inflight_*`

### T0-B: Z1-MDL-ABLATION-C6

**Subagent ID format:** `t0_b_z1_mdl_ablation_c6_20260514_<random>`
**Lane:** existing `lane_zen_floor_scorer_conditional_mdl_ablation_20260514` (extend with C6 anchor)
**Budget:** $0 GPU; 2h wall-clock
**Owner scope:** `tools/mdl_scorer_conditional_ablation.py` + `experiments/results/mdl_ablation_c6_*/`

**Mandate:**
1. Read parent memo Section 2.2 (within-class saturation finding)
2. Once T0-A confirms C6 IBPS1 archive available, run MDL ablation on C6
3. Compute scorer-conditional MDL density per A1/PR106 protocol
4. Write `experiments/results/mdl_ablation_c6_*/c6_mdl_ablation.json` with `archive_sha256` + `mdl_density_estimate_lo`
5. Verify: if C6 density < 0.95, C6 is CLASS-SHIFT empirical anchor (NOT within-HNeRV-class saturated) — surface as high-priority signal
6. Update zen-floor band per reactivation criteria (parent memo section 7.9)
7. Add C6 to `autopilot_candidate_queue_v2_post_z1_revision_20260514.jsonl` per Catalog #219

**Deferral:** if C6 archive not yet available (T0-A still in flight), defer this subagent's start; do not spawn until T0-A `--step complete`
**Conditional dependency:** requires T0-A's harvest result

### T0-C: PROBE-DISAMBIGUATORS

**Subagent ID format:** `t0_c_probe_disambiguators_20260514_<random>`
**Lane:** NEW pre-register `lane_uniward_weighting_d4_probe_20260514` + `lane_z9_world_model_vs_hyperprior_disambiguator_20260514`
**Budget:** $0 GPU; 2h wall-clock
**Owner scope:** `tools/probe_uniward_weighting_on_d4_residual.py` + `tools/probe_z9_world_model_vs_hyperprior_disambiguator.py` (new files)

**Mandate:**
1. Per Fridrich Round 2 addition: build `tools/probe_uniward_weighting_on_d4_residual.py` that estimates UNIWARD-weighting ΔS-impact on D4 residual codec via offline analysis (no GPU)
2. Per zen-floor council Z9: build `tools/probe_z9_world_model_vs_hyperprior_disambiguator.py` that tests which interpretation (world-model-IS-hyperprior vs predictive-coding) better fits empirical anchors
3. Both probes write structured JSON verdicts under `.omx/research/probe_outputs/`
4. Surface findings to T0-D (informs Tier 1 priority)

**Independence:** no dependencies on other Tier 0 subagents
**Crash-resume:** probe code self-contained; resume reads existing partial files

### T0-D: PARALLEL-HARVEST-ACTUATOR-BUILD

**Subagent ID format:** `t0_d_parallel_harvest_actuator_20260514_<random>`
**Lane:** NEW pre-register `lane_tier_1_parallel_harvest_actuator_20260514`
**Budget:** $0 GPU; 2h wall-clock
**Owner scope:** `tools/harvest_tier_1_parallel_wave.py` (new file)

**Mandate (per Carmack Round 2 intervention; CLAUDE.md "Race-mode rigor inversion + parallel-dispatch first" Rule 1):**
1. Build `tools/harvest_tier_1_parallel_wave.py` that:
   - Reads `.omx/state/subagent_progress.jsonl` for all `t1_*` checkpoints
   - For each Modal call_id in checkpoints, runs `tools/harvest_modal_calls.py` filter
   - For each Lightning job, runs canonical Lightning state-reader
   - For each Vast.ai instance, runs `tools/vastai_orphan_cleanup.py` audit
   - Aggregates all results into `reports/tier_1_harvest_$(date +%Y%m%dT%H%M%SZ).json`
   - Includes score-extraction for any contest-CUDA / contest-CPU evals
   - Tags every result by axis per CLAUDE.md "Forbidden score claims"
2. Add unit tests for `parse_tier_1_state_chain()` + `aggregate_provider_harvests()`
3. Wire into CLAUDE.md catalog if applicable (Catalog # to claim if STRICT gate needed)

**Independence:** no other Tier 0 dependencies
**Reusability:** this actuator survives across all future waves; first-class deliverable

### T0-E: CATALOG-226-FINISH

**Subagent ID format:** `t0_e_catalog_226_finish_20260514_<random>`
**Lane:** existing `lane_catalog_226_trainer_auth_eval_canonical_helper_20260514`
**Budget:** $0 GPU; 2h wall-clock
**Owner scope:** Complete in-flight refactor inherited from CATALOG-226-REFACTOR `ad33b810`

**Mandate:**
1. Pick up CATALOG-226-REFACTOR's predecessor checkpoint if it crashed/stalled
2. Complete the 18-trainer auth_eval canonical-helper refactor (in flight)
3. Atomic strict-flip Catalog #226 once live count = 0
4. Add row in CLAUDE.md catalog table per Catalog #176 (strict callsite has CLAUDE.md entry)
5. Run preflight smoke

**Sister-subagent coordination:** if CATALOG-226-REFACTOR is still active, defer; check `tools/subagent_checkpoint.py read --latest-incomplete`

### T0-F: C1-SMOKE-DISPATCH

**Subagent ID format:** `t0_f_c1_smoke_20260514_<random>`
**Lane:** existing `lane_c1_world_model_foveation_campaign_20260514`
**Budget:** $1 GPU; 1h wall-clock
**Owner scope:** C1 smoke dispatch via canonical operator-authorize wrapper

**Mandate:**
1. Per parent memo Section 7.2, dispatch C1 smoke at 100ep T4
2. Use canonical `tools/operator_authorize.py --recipe substrate_c1_world_model_foveation_modal_t4_dispatch.yaml --run-smoke-before-full`
3. Set paired-env-var bypass per Catalog #199: `OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1` + `OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=1.50`
4. Claim lane dispatch first per CLAUDE.md "CROSS-AGENT DISPATCH COORDINATION"
5. After smoke, harvest + write empirical anchor to `tac.cost_band_calibration` with `outcome=smoke_complete`
6. Surface to operator at Checkpoint 1

**Catalog #165 quiescence:** mandatory before fire; verify mtime stability
**Deferral if quiescence violation:** retry every 5 min for up to 1h; then surface as blocker

---

## TIER 1 SUBAGENT ASSIGNMENTS (Phase 2 — fire AFTER Tier 0 wave quiescence)

### T1-A: Z3-FULL-MAIN-DISPATCH

**Subagent ID format:** `t1_a_z3_full_main_20260514_<random>`
**Lane:** existing `lane_z3_balle_hyperprior_bolton_campaign_20260514`
**Budget:** $2 GPU; 8h wall-clock
**Owner scope:** `experiments/train_substrate_z3_balle_hyperprior_bolton.py` + recipe + remote driver

**Mandate:**
1. Z3 `_full_main` already implemented + council-approved 6/6 PROCEED per `feedback_z3_full_main_impl_phase_2_council_20260514.md`
2. Dispatch full $2 Modal A100 (recipe pinned)
3. Predicted ΔS: -0.0003 to -0.0009 per `[Ballé 2018 §IV.A]` 5-15% byte savings bound
4. Smoke-before-full mandatory per Catalog #167; smoke wrapper auto-fires
5. Auth-eval at completion via Catalog #226 canonical helper
6. Update posterior via `posterior_update_locked` per Catalog #128
7. Surface result at Checkpoint 3

**Falsification probe role:** if ΔS ≥ 0, escalates Tier 2 class-shift priority (parent memo section 6.2)

### T1-B: Z4-COOPERATIVE-RECEIVER

**Subagent ID format:** `t1_b_z4_cooperative_receiver_20260514_<random>`
**Lane:** existing `lane_z4_cooperative_receiver_loss_20260514`
**Budget:** $5-8 GPU; 12h wall-clock
**Owner scope:** `src/tac/substrates/z4_cooperative_receiver_loss/*` + trainer + recipe

**Mandate:**
1. Z4 trainer + recipe landed per `feedback_time_traveler_l5_staircase_steps_2_3_landed_20260514.md`
2. Dispatch full Modal T4
3. Predicted ΔS: -0.005 to -0.010 per `[Atick-Redlich 1990]` cooperative-receiver theory
4. Same wrapper pattern as T1-A
5. Smoke-before-full mandatory

**Composition opportunity:** Z4 + Z3 mutually-additive; if both land positive, Tier 2 cell `Z4×Z3` is high-priority

### T1-C: D4-WYNERZIV-FULL

**Subagent ID format:** `t1_c_d4_wynerziv_full_20260514_<random>`
**Lane:** existing `lane_d4_wyner_ziv_frame_0_substrate_20260514`
**Budget:** $10 GPU; 12h wall-clock
**Owner scope:** `src/tac/substrates/d4_wyner_ziv_frame_0/*` + trainer + recipe

**Mandate:**
1. D4 mini-batch fix landed per `feedback_d4_oom_fix_minibatch_landed_20260514.md` + `feedback_d4_unblock_landed_20260514.md`
2. Smoke validation FIRST (smoke-before-full mandatory)
3. If smoke ΔS in expected band, dispatch full Modal T4
4. Predicted ΔS: -0.025 to -0.045 (highest individual variance Tier 1 substrate)
5. Composition with Ballé hyperprior is Tier 2 cell

**Catalog #218 sister:** mini-batch reconstruct_pair STRICT gate already protects against OOM regression

### T1-D: D1-MARGIN-POLYTOPE-FULL

**Subagent ID format:** `t1_d_d1_margin_polytope_full_20260514_<random>`
**Lane:** existing `lane_d1_segnet_margin_polytope_encoder_20260514`
**Budget:** $10 GPU; 8h wall-clock
**Owner scope:** `src/tac/substrates/d1_segnet_margin_polytope/*` + trainer + recipe

**Mandate:**
1. D1 L2 integration landed per `feedback_d1_l2_integration_plus_permanent_gate_landed_20260514.md`
2. Smoke + full dispatch
3. Predicted ΔS: -0.010 to -0.020 (frame-1 polytope geometric-nullspace; sister of D4)
4. Same provider + wrapper pattern

**Composition with D4:** sister substrates; A1×D4×D1 cell is potentially additive; Tier 2 candidate

### T1-E: C6-MDL-IBPS-FULL

**Subagent ID format:** `t1_e_c6_mdl_ibps_full_20260514_<random>`
**Lane:** existing `lane_c6_e4_mdl_ibps_substrate_20260514`
**Budget:** $15 GPU; 12h wall-clock
**Owner scope:** `src/tac/substrates/c6_e4_mdl_ibps/*` + trainer + recipe

**Mandate:**
1. C6 auth_eval CLI fix landed per `feedback_recovery_2_c6_finish_and_modal_harvest_landed_20260514.md`
2. C6 IBPS1 grammar already added to MDL ablation tool (predecessor wave)
3. Dependent on T0-B Z1 ablation output (if C6 density measured < 0.95 vs A1's 99.29%, C6 IS class-shift candidate; promote priority)
4. Dispatch full Modal A100 (preferred for IBPS architecture; T4 fallback)
5. Predicted ΔS: -0.010 to -0.030

**Conditional dependency:** T0-B must complete first; T1-E start gated on its anchor

### T1-F: Z3xC6-COMPOSITION-PROBE

**Subagent ID format:** `t1_f_z3xc6_composition_probe_20260514_<random>`
**Lane:** NEW pre-register `lane_z3xc6_composition_probe_20260514`
**Budget:** $1 GPU; 4h wall-clock
**Owner scope:** `experiments/composition_probes/z3_x_c6_substrate_probe_20260514.py` (new file)

**Mandate:**
1. Probe composition cell Z3 (Ballé) + C6 (MDL-IBPS) on shared dataset
2. Cheap T4 dispatch; SMOKE only (not anchor)
3. Predicted ΔS cumulative: -0.001 to -0.005 (small but real test)
4. Optional; only fires if T0-D harvester is online (so Z3+C6 results can be combined post-T1-A and T1-E completion)
5. Surface composition matrix output to operator at Checkpoint 3

**Optionality:** can be deferred to Tier 2 if Wave 2 envelope crowded

---

## TIER 2 SUBAGENT ASSIGNMENTS (multi-stage; Wave 3; conditional)

### T2-A: Z5-PREDICTIVE-CODING-FULL
**Lane:** existing `lane_z5_predictive_coding_world_model_20260514`; budget $10/24h
**Condition:** Z4 lands negative OR T0-C probe-disambiguator output points to predictive-coding hypothesis
**Predicted ΔS:** -0.030 to -0.060 per `[Rao & Ballard 1999]`

### T2-B: A1xZ3-COMPOSITION-CELL
**Lane:** NEW `lane_a1xz3_composition_cell_20260514`; budget $10/12h
**Condition:** Z3 lands at predicted range
**Predicted ΔS:** -0.005 to -0.010 cumulative

### T2-C: D4xBALLE-COMPOSITION-CELL
**Lane:** NEW `lane_d4xballe_composition_cell_20260514`; budget $15/24h
**Condition:** D4 lands at [0.148, 0.168]
**Predicted ΔS:** -0.005 to -0.020 cumulative

### T2-D: STC-POSE-SUBSTRATE (Fridrich Round 2 add)
**Lane:** NEW `lane_stc_pose_substrate_20260514`; budget $10/24h
**Condition:** Independent; can fire in Wave 2 alongside Tier 1
**Predicted ΔS:** -0.005 to -0.010 per `[Filler 2011]` syndrome-trellis pose

### T2-E: C1-PHASE-3-MULTI-STAGE
**Lane:** existing `lane_c1_world_model_foveation_campaign_20260514`; budget $30-50/3-5 days
**Condition:** T0-F smoke valid + Z5 lands positive
**Predicted ΔS:** -0.040 to -0.070 (substrate-class shift)

### T2-F: COOPERATIVE-RECEIVER-MATURE
**Lane:** NEW `lane_z4_cooperative_receiver_mature_20260514`; budget $30/3 days
**Condition:** Z4 lands positive
**Predicted ΔS:** -0.015 to -0.030 cumulative

### T2-G: DASHER-ARITHMETIC-POSE (MacKay Round 1 add)
**Lane:** NEW `lane_dasher_arithmetic_pose_substrate_20260514`; budget $5/12h
**Condition:** Independent; can fire in Wave 2 alongside Tier 1
**Predicted ΔS:** -0.001 to -0.005 (Dasher-style arithmetic-coded pose residual)

---

## TIER 3 SUBAGENT ASSIGNMENTS (queued; awaiting operator $200/wave approval)

### T3-A: Z7-PREDICTIVE-RECEIVER-MATURE
**Lane:** NEW `lane_z7_predictive_receiver_mature_substrate_20260514`; budget $50-100/8-12 weeks
**Operator action:** Approve as standalone wave per CLAUDE.md "Long-burn campaign default"
**Predicted ΔS:** to Time-Traveler asymptote [0.03, 0.07]

### T3-B: C7-DARTS-SUPERNET
**Lane:** NEW `lane_c7_darts_supernet_substrate_search_20260514`; budget $100-300/4-8 weeks
**Operator action:** Approve as standalone wave
**Predicted ΔS:** unknown variance; substrate-class search

### T3-C: C2-MATURE-L5-AUTONOMY
**Lane:** NEW `lane_c2_mature_l5_autonomy_20260514`; budget $50-100/8-12 weeks
**Operator action:** Approve as standalone wave
**Predicted ΔS:** matches Time-Traveler reverse-engineered architecture in posterior

### T3-D: C3-MULTI-YEAR-SUB-0.05
**Lane:** NEW `lane_c3_multi_year_sub_0_05_campaign_20260514`; budget $500-2000/1-3 years
**Operator action:** STRATEGIC DECISION per long-term campaign roadmap; surface explicitly
**Predicted ΔS:** to sub-0.05 long-term goal

---

## Parent agent spawn instructions

When the parent agent reads this memo, the spawn-orchestration sequence is:

```bash
# Wave 1 (Tier 0; all parallel; ~2-4h)
spawn t0_a_harvest_inflight_20260514 # T0-A
spawn t0_b_z1_mdl_ablation_c6_20260514 # T0-B
spawn t0_c_probe_disambiguators_20260514 # T0-C
spawn t0_d_parallel_harvest_actuator_20260514 # T0-D
spawn t0_e_catalog_226_finish_20260514 # T0-E
spawn t0_f_c1_smoke_20260514 # T0-F (waits for mtime quiescence)

# Block until ALL --step complete
.venv/bin/python tools/subagent_checkpoint.py read --latest-incomplete \
    | grep -c '"status": "in_progress"' # should be 0

# Verify mtime quiescence
find src tools experiments scripts -newer $(date -v-3S +%Y%m%dT%H%M%S) | head # should be empty

# Wave 2 (Tier 1 + Tier 2 independents; all parallel; ~12-24h)
spawn t1_a_z3_full_main_20260514 # T1-A ($2)
spawn t1_b_z4_cooperative_receiver_20260514 # T1-B ($5-8)
spawn t1_c_d4_wynerziv_full_20260514 # T1-C ($10)
spawn t1_d_d1_margin_polytope_full_20260514 # T1-D ($10)
spawn t1_e_c6_mdl_ibps_full_20260514 # T1-E ($15; awaits T0-B)
spawn t1_f_z3xc6_composition_probe_20260514 # T1-F ($1; optional)
spawn t2_d_stc_pose_substrate_20260514 # T2-D Tier 2 independent ($10)
spawn t2_g_dasher_arithmetic_pose_20260514 # T2-G Tier 2 independent ($5)

# Wave 3 (Tier 2 dependent; after Wave 2 anchors harvested)
# spawn t2_a / t2_b / t2_c / t2_e / t2_f conditional on harvests

# Wave 4 (Tier 3 queued; ONLY after operator $200/wave approval)
# spawn t3_a / t3_b / t3_c / t3_d only after explicit operator decision
```

---

## Operator-routable decision summary

Per parent memo Section 12, 7 operator-routable decisions. Routing recommendations:

| # | Decision | Cost | Pre-approval status | Recommended action |
|---|---|---:|---|---|
| A | Tier 0 + Tier 1 envelope | $0-$46 | PRE-APPROVED (council 11/11) | Launch Wave 1 |
| B | Tier 2 envelope ≤$200/wave | $110-$170 | PRE-APPROVED (within threshold) | Approve at Checkpoint 4 |
| C | Tier 3 envelope $700-$2500 | varies | NOT PRE-APPROVED | Wave-by-wave operator decision |
| D | Phase 1 / Phase 2 sequencing | $0 | OPERATIONAL | Parent agent enforces |
| E | Probe-disambiguator priority | $0 | PRE-APPROVED | Tier 0 work; auto-fires |
| F | Harvest actuator first-class | $0 | PRE-APPROVED | Tier 0 work; auto-fires |
| G | Quarterly review cadence | $0 | OPERATOR-SET | Set 90-day reminder |

---

## Crash-resume

If this dispatch authorization memo is interrupted mid-spawn:
1. Read parent memo Section 7.2-7.5 for tier plan
2. Read `.omx/state/subagent_progress.jsonl` for which subagents have spawned
3. Check `tools/claim_lane_dispatch.py summary` for active dispatches
4. Re-survey if any Tier 1 wave-2 subagent has --status complete with anchor; consider whether Tier 2 dependents are ready

**No KILL.** All deferrals carry reactivation criteria per parent memo Section 7.9.

🌀🏛️🛰️
