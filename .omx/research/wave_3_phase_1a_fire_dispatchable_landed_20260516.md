---
title: "WAVE-3 Phase 1a baseline fire LANDED: 2 of 2 dispatchable recipes (STC v2 + NSCS06 v8) fired via canonical operator_authorize.py with Catalog #202 paired-env bypass"
date: 2026-05-16
author: WAVE-3 Phase 1a dispatcher subagent (wave_3_phase_1a_fire_dispatchable_20260516)
lane: lane_wave_3_phase_1a_fire_dispatchable_20260516
horizon_class: frontier_pursuit
mission_alignment: frontier_breaking
status: BOTH-DISPATCHES-FIRED; awaiting harvest 6-24h
related: feedback_wave_3_phase_1_systemic_refusal_3_of_5_recipes_dispatch_disabled_20260516 (predecessor STOP-AND-REPORT) + grand_council_t3_wave_2_batch_nscs06_rescue_tishby_wunderkind_nerv_family_phase_2_lift_20260516
council_tier: T1
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
council_attendees: [WAVE-3-Phase-1a-dispatcher]
council_quorum_met: true
council_verdict: PROCEED
council_decisions_recorded:
  - "Option B (re-scoped Phase 1 to only 2 dispatchable recipes) selected per predecessor's recommendation"
  - "Catalog #202 paired-env bypass invoked after independent sentinel verification per gate's actionable option [2]"
  - "Both dispatches fired via canonical tools/operator_authorize.py (Catalog #176 + #243 + #271 wire-in)"
council_dissent: []
---

## TL;DR

Per WAVE-3 Phase 1 predecessor's Option B recommendation, fired only the 2 dispatchable recipes (STC v2 + NSCS06 v8) leaving Rudin / Z6 / ATW-v2 to sister LIFT-* subagents for Phase 2 trainer lifts. Both dispatches landed successfully through the canonical apparatus.

## Dispatch outcomes

| # | Recipe | Modal call_id | Job | Status |
|---|---|---|---|---|
| 1 | substrate_stc_v2_modal_t4_dispatch | fc-01KRSVKF9VEESQY2FS33FF4WDM | substrate_stc_v2_modal_t4_dispatch_20260517T021720Z | active_dispatch on Modal T4 (0.5h timeout) |
| 2 | substrate_nscs06_v8_path_b_wavelet_modal_t4_dispatch | fc-01KRSVGE57MT5XSAWCGNQFQPBP | substrate_nscs06_v8_path_b_wavelet_modal_t4_dispatch_20260517T021548Z | active_dispatch on Modal T4 (1.5h timeout) |

Both registered in `.omx/state/modal_call_id_ledger.jsonl` per Catalog #245.

## Cost summary

- STC v2: predicted p50 $0.07 (empirical_posterior N=5; p10/p50/p90 = $0.03/$0.07/$0.18)
- NSCS06 v8: predicted p50 $0.07 (same empirical_posterior table); hand-calibrated fallback $15.00 is conservative cold-start estimate per recipe
- **Total expected spend: $0.14 at p50; $0.36 at p90** — well within $10.20 hard cap (parent's slice of $246 envelope)

## Required apparatus interactions

### First-attempt blocker: Catalog #166 --require-clean-head FATAL (both dispatches)

Both first attempts (02:14:11Z STC v2, 02:14:14Z NSCS06 v8) refused with `FATAL [Catalog #166]: --require-clean-head is set and the working tree has 10 uncommitted edit(s)`. Dirty files were 100% LIFT-Z6 / codex-sister-subagent owned per Catalog #230 ownership map:
- codex sister research ledger (5): `.omx/research/l5_v2_lattice_measurement_schedule_20260516_codex.{json,md}` + `.omx/research/l5_v2_paired_measurement_dispatch_plan_20260516_codex.{json,md}` + `.omx/research/l5_v2_tt5l_sideinfo_variant_packets_20260517_codex.{json,md}`
- operator-owned source (5): `src/tac/optimization/l5_v2_measurement_schedule.py` + `src/tac/tests/test_l5_v2_measurement_schedule.py` + `src/tac/optimization/tt5l_sideinfo_variant_packets.py` + `src/tac/tests/test_tt5l_sideinfo_variant_packets.py` + `tools/build_tt5l_sideinfo_variant_packets.py`

### Resolution: Catalog #202 paired-env bypass (independent sentinel verification)

Per gate's actionable next-step option [2]: independently verified all 12 sentinel files (6 per dispatch) clean via `git status --porcelain`:
```
experiments/modal_train_lane.py — clean
tools/operator_authorize.py — clean
tools/run_modal_smoke_before_full.py — clean
src/tac/deploy/modal/mount_manifest.py — clean
scripts/remote_lane_substrate_stc_v2.sh — clean
experiments/train_substrate_stc_v2.py — clean
scripts/remote_lane_substrate_nscs06_v8_path_b_wavelet.sh — clean
experiments/train_substrate_nscs06_v8_path_b_wavelet.py — clean
```

Then set:
```bash
export OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK=1
export OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED=1
```

Per Catalog #166 worker-side hash check still runs to validate sentinel parity at the Modal worker (defense-in-depth preserved).

### Secondary observation: Catalog #165 Modal mount mtime stability retry

STC v2 first attempt also triggered `Catalog #165 detected a pre-spawn Modal mount upload race; waiting for source mtimes to settle before retrying in 5.0s` — sister TT5L test file modified during build. Auto-retried 2/3 times before the Catalog #166 fatal terminated the chain. Working as designed (race detection prevented torn upload).

### All 9 local pre-deploy checks PASS per Catalog #243

Both retried dispatches passed all 9 strict pre-deploy harness checks:
- py_compile / trainer_importable / full_main_implemented / archive_grammar / auth_eval_reachability / canonical_inflate_device / deterministic_zip / recipe_status_consistent_with_trainer_state / dispatch_optimization_protocol (Catalog #270 Tier 1/2/3 all complete)

## 6-hook wire-in status per Catalog #125

1. **Sensitivity-map contribution** — N/A (no empirical anchors produced YET; dispatches in-flight)
2. **Pareto constraint** — DEFERRED-pending-harvest (will fire once auth_eval scores land)
3. **Bit-allocator hook** — N/A (these are codec dispatches, not tensor-importance-updating training)
4. **Cathedral autopilot dispatch hook** — ACTIVE (`tools/operator_authorize.py` IS the actuator; both dispatches consumed via canonical wire-in)
5. **Continual-learning posterior update** — DEFERRED-pending-harvest (cost_band posterior + lane registry update on outcome)
6. **Probe-disambiguator** — N/A (these are first-anchor dispatches per their design memos; STC v2 IS the MPS-vs-CUDA disambiguator; NSCS06 v8 IS the v7-Path-A vs v8-Path-B band test)

## Next actions for parent

1. **HARVEST in 6-24h** — STC v2 timeout 0.5h so likely complete by 03:00Z UTC; NSCS06 v8 timeout 1.5h so likely complete by 04:00Z UTC. Per CLAUDE.md HARVEST OR LOSE non-negotiable, do not let either result expire from Modal cache.
2. **Update Modal call_id ledger** on harvest with `--status completed_...` outcome rows per Catalog #245 schema.
3. **Mark lane gates** as evidence lands: `real_archive_empirical` (when archives produced) → `contest_cuda` (when paired CUDA auth_eval lands) → `three_clean_review` (Phase 1b adversarial review).
4. **Tier C MDL ablation** on resulting archives per Catalog #227 if either lands as L2-promotion candidate.
5. **Sister LIFT-* status**: monitor Rudin / Z6 / ATW-v2 Phase 2 lifts (separate sister subagents) before re-dispatch of those recipes.

## Lane maturity status

- L0 → L1 (impl_complete=true) per this landing
- 6 remaining gates pending harvest + downstream evidence

## Checkpoint trail per Catalog #206

- Step 0 (in_progress) — read predecessor + pre-flight reads
- Step 1 (in_progress) — dispatches fired, waiting for retry resolution
- Step complete on this landing memo commit

## Subagent ID

`wave_3_phase_1a_fire_dispatchable_20260516`

## Cross-refs

- Predecessor: `feedback_wave_3_phase_1_systemic_refusal_3_of_5_recipes_dispatch_disabled_20260516.md` (Option B selected)
- Sister LIFT subagents (DO NOT TOUCH): LIFT-Rudin / LIFT-Z6 / LIFT-ATW-v2
- Modal call_id ledger: `.omx/state/modal_call_id_ledger.jsonl` (Catalog #245)
- Active dispatch claims: `.omx/state/active_lane_dispatch_claims.md` (Catalog #186)
- Catalog #202 paired-env bypass design: CLAUDE.md "Catalog #202 ... `--require-clean-head` bypass" section
- Catalog #166 worker-side hash: `experiments/modal_train_lane.py::_run_lane_inner` (defense-in-depth preserved)
