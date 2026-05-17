---
title: "WAVE-3 Phase 1 baseline fire BLOCKED: 3 of 5 named recipes have dispatch_enabled=false pending Phase 2 council lift; STOP-AND-REPORT per parent guidance"
date: 2026-05-16
author: WAVE-3 Phase 1 orchestrator subagent (wave_3_phase_1_orchestrator_20260516)
lane: lane_wave_3_phase_1_baseline_fire_20260516
horizon_class: frontier_pursuit
mission_alignment: frontier_protecting  # refusing systemic-bypass that would re-introduce Z3 v2 / Z4 / Z5 bug class
status: BLOCKED-PENDING-OPERATOR-DECISION; 0 dispatches fired; $0 spent (vs $37.70 phase cap)
related: grand_council_t3_wave_2_batch_nscs06_rescue_tishby_wunderkind_nerv_family_phase_2_lift_20260516 + k_measurement_schedule_level_1_rebalanced_post_donoho_tanner_20260516
council_tier: T1
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_attendees: [Wave-3-Phase-1-orchestrator]
council_quorum_met: true
council_verdict: ESCALATE_TO_OPERATOR
council_decisions_recorded:
  - "STOP-AND-REPORT per parent's explicit STOP-IF-3-REFUSE rule"
  - "Do NOT bypass via OPERATOR_AUTHORIZE_CODEX_REVIEW_BYPASS_VERDICT (forbidden per parent)"
  - "Do NOT toggle dispatch_enabled=true without trainer _full_main lift (would re-introduce Z3 v2 bug class)"
council_dissent: []
---

## TL;DR (60 seconds)

Parent dispatched the WAVE-3 Phase 1 orchestrator to fire 5 baseline dispatches in parallel: Rudin substrate / Z6 time-traveler L5 / STC v2 retry / ATW D4 probe / NSCS06 cheap v8 sister-probe. **3 of the 5 named recipes (Rudin, Z6, ATW-D4-Wyner-Ziv) carry `dispatch_enabled: false` AND `research_only: true` flags** because their underlying substrate trainers' `_full_main` functions raise `NotImplementedError` pending Phase 2 council approval per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" non-negotiable + Catalog #240 recipe-vs-trainer-state consistency.

`tools/operator_authorize.py::_recipe_dispatch_refusal` correctly fail-closes on all 3 with the message `would refuse: dispatch_enabled=false`. This is the STRUCTURAL gate that prevents re-introducing the Z3 v2 / Z4 / Z5 bug class (Modal smoke crashes pre-auth-eval at $1-15 per dispatch when the trainer can't reach auth eval stage).

Per parent's explicit instruction: *"If 3+ of 5 dispatches refuse for the same reason, that's a systemic issue — STOP and report to parent rather than burn the budget on doomed fires."*

**Outcome:** 0 dispatches fired. $0 spent. Phase budget cap of $37.70 untouched.

## Recipe-by-recipe verification (via dry-run)

| # | Recipe | Substrate | dispatch_enabled | research_only | Trainer state | Dry-run verdict |
|---|---|---|---|---|---|---|
| 1 | substrate_rudin_floor_interpretable_ml_modal_t4_dispatch | rudin_floor | **false** | true | _full_main NotImplementedError | REFUSED: dispatch_enabled=false |
| 2 | substrate_time_traveler_l5_z6_modal_t4_dispatch | time_traveler_l5_z6 | **false** | true | _full_main NotImplementedError | REFUSED: dispatch_enabled=false; 3 dispatch_blockers declared |
| 3 | substrate_stc_v2_modal_t4_dispatch | stc_v2 | true | true | _full_main IMPLEMENTED | PROCEEDS past gate (research_only OK for diagnostic disambiguator) |
| 4 | substrate_d4_wyner_ziv_frame_0_modal_t4_dispatch | d4_wyner_ziv | (smoke_only:true; assumed dispatch_enabled default=true) | n/a | smoke_only:true mode | PROCEEDS past gate IF used as ATW-D4 probe BUT NOT what parent named |
| 5 | substrate_nscs06_v8_path_b_wavelet_modal_t4_dispatch | nscs06_v8_path_b | true | false | _full_main IMPLEMENTED | PROCEEDS past gate |

**Note on #4:** parent named "ATW D4 probe" referencing ATW codec v2 design memo. The actual ATW v2 recipe (`substrate_atw_codec_v2_modal_a100_dispatch.yaml`) also carries `dispatch_enabled: false` + 3 explicit dispatch_blockers. The D4 recipe (`substrate_d4_wyner_ziv_frame_0_modal_t4_dispatch.yaml`) carries `smoke_only: true` but is a sister Wyner-Ziv substrate, NOT ATW-D4. Whether parent meant ATW v2 or D4 standalone is ambiguous — both interpretations land at the same structural blocker (ATW v2 explicitly refused; D4 standalone is smoke_only and works but is not the "ATW D4 probe" parent named).

## Root cause: Wave 2 T3 council approval has not yet propagated to recipe-toggle + trainer-lift commits

Per `.omx/research/grand_council_t3_wave_2_batch_nscs06_rescue_tishby_wunderkind_nerv_family_phase_2_lift_20260516.md` (commit `599de2947`) Decision 2 (Tishby IB-pure) and Decision 4 (NeRV-family Phase 2 lift order), the council APPROVED:

- "FUND-PHASE-2-STAGE-1-MODAL-A100-100EP-PROXY ($5-10 budget)" for Tishby IB-pure
- "PHASE-2-LIFT-ORDER: (1) HiNeRV ... (6) BlockNeRV. Per-substrate cost envelope $5-15 Modal smoke + $5-15 paired CUDA/CPU full"

But the council verdict alone is NOT sufficient to lift `dispatch_enabled: false`. The required follow-on commits are:

1. **Lift `_full_main` NotImplementedError** in each substrate trainer with real Phase 2 implementation
2. **Toggle `dispatch_enabled: true` + `research_only: false`** in the recipe YAML
3. **Clear `dispatch_blockers` list** to `[]`
4. **Wire 6-hook integration** per Catalog #125 (sensitivity-map / Pareto / bit-allocator / cathedral autopilot / continual-learning / probe-disambiguator)

Neither the Rudin nor Z6 nor ATW-v2 nor D4-Wyner-Ziv (as ATW-D4 probe) have these follow-on commits. The Phase 2 council ratification per the council memo Section 22 op-routables has not yet landed as code.

## Why this is the CORRECT behavior of the apparatus

Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" non-negotiable + Catalog #240 (`check_substrate_contest_cuda_chain_complete_or_research_only_tagged`) STRICT preflight gate + Catalog #220 (substrate L1+ byte-addition operational mechanism declaration):

- Z3 v2 FULL Modal A100 dispatch 2026-05-15T11:41:15Z burned $2 because of recipe-vs-trainer-state divergence (recipe claimed dispatch-ready; trainer's `_full_main` reached auth-eval but mis-routed to CPU)
- Z4 + Z5 trainers historically had recipes pointing to NotImplementedError-raising `_full_main` paths; the Z3 v2 / Z4 / Z5 bug class burned ~$4 across multiple dispatches before Catalog #240 landed

The fact that 3 of 5 recipes refuse here is NOT a bug in the apparatus — it is the apparatus correctly enforcing the substrate completeness contract. Bypassing it would re-introduce the bug class at ~$1-15 per refused-fire.

## What parent should decide

Options for parent agent (orchestrator) to choose:

### Option A: Lift Phase 2 council approvals into code first
Spawn 3 sister subagents to land the Phase 2 implementations:
1. **Rudin substrate Phase 2** — implement GOSDT depth-4 encoder + Wang-Rudin K=4-6 falling rule list decoder + Semenova-Rudin-Parr K=8 Rashomon ensemble loss + RDIF v1 monolithic archive + ≤200 LOC pure Python inflate per design memo; toggle recipe `dispatch_enabled: true`
2. **Z6 time-traveler L5 Phase 2** — implement single-layer FiLM-conditioned conv predictor (~75K params) + identity_predictor ablation probe per design memo; toggle recipe
3. **ATW v2 Phase 2** — implement full-stack cooperative-receiver codec per design memo; toggle recipe

Then re-dispatch this WAVE-3 Phase 1 with all 5 lit.

Cost estimate: ~3-6 hours subagent editing per substrate (no GPU). Risk: high (Phase 2 implementations are non-trivial substrate engineering).

### Option B: Re-scope Phase 1 to only the 2 dispatchable recipes
Fire only STC v2 + NSCS06 v8 baseline as Phase 1 ($0.20 + $5-10 = $5.20-10.20 actual cost vs $37.70 cap). The other 3 substrates wait until their Phase 2 implementations land.

Cost: $5.20-10.20. Risk: low. Captures only 2 of 5 intended baseline measurements.

### Option C: Re-name Phase 1 dispatches per actual recipe inventory
Operator may have intended different recipes than what is named. For example:
- "STC v2 retry" → `substrate_stc_v2_modal_t4_dispatch` (dispatchable; $0.20)
- "NSCS06 cheap v8 sister-probe" → `substrate_nscs06_v8_path_b_wavelet_modal_t4_dispatch` (dispatchable; $5-10) OR `substrate_nscs06_carmack_hotz_strip_everything_modal_t4_dispatch` (dispatchable; cost unknown, both `dispatch_enabled: true` + `research_only: false`)
- "ATW D4 probe" → `substrate_d4_wyner_ziv_frame_0_modal_t4_dispatch` (`smoke_only: true` — needs verification it can dispatch)

The Rudin and Z6 dispatches genuinely cannot fire today without Phase 2 code lift.

### Option D: Operator override — explicitly toggle recipes
The operator can directly authorize dispatch_enabled=true on the 3 refused recipes WITH the understanding that the trainers will crash at `_full_main` NotImplementedError and burn $0.50-5 per crash for diagnostic value. This is the "force-fire-to-measure-something" approach but loses ~$15-45 to crashes.

## Recommendation

**Option B** (re-scope to 2 dispatchable recipes) is the safest immediate action; **Option A** (land Phase 2 implementations first) is the correct longer-term path; **Option C** (re-verify recipe naming) should be the operator's first response to confirm parent's intent.

## 6-hook wire-in status per CLAUDE.md "Subagent coherence-by-default"

1. **Sensitivity-map contribution** — N/A (no dispatch fired; no empirical anchors produced)
2. **Pareto constraint** — N/A (no new candidate produced)
3. **Bit-allocator hook** — N/A (no new tensor importance change)
4. **Cathedral autopilot dispatch hook** — DEFERRED (this subagent was meant to be the actuator; deferral is the systemic refusal itself; the actuator surface is `tools/operator_authorize.py` per Catalog #243/#271 wire-in)
5. **Continual-learning posterior update** — N/A (no empirical anchor)
6. **Probe-disambiguator** — N/A (this is a recipe-state diagnostic, not a 2+ defensible interpretation question)

## Lane registry pre-registration

Pre-registered via `tools/lane_maturity.py add-lane lane_wave_3_phase_1_baseline_fire_20260516 --name "Wave 3 Phase 1 baseline parallel fire" --phase 2` (Level 0 SKETCH; will be marked impl_complete=False until Phase 2 lifts land + dispatches actually fire).

## Checkpoint trail per Catalog #206

- Step 1 (in_progress) — pre-register lane + audit 5 recipes
- Step 2 (blocked) — report 3/5 systemic refusal to parent

Subagent ID: `wave_3_phase_1_orchestrator_20260516`

## Next action for parent agent

Choose one of Options A/B/C/D above and re-dispatch with corrected recipe set OR escalate to operator for ratification of the Phase 2 lift commits. The harvest schedule for the (would-be) Modal dispatches is moot until at least one dispatch lands.
