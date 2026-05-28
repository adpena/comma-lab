---
landing_utc: 2026-05-28T04:17:00Z
lane_id: lane_pact_nerv_ia3_lane_script_catalog_240_drift_fix_20260528
task_id: 1437
parent_session: feeder_session_2026_05_27
subagent_id: task_1437_pact_nerv_ia3_lane_script_catalog_240_drift_fix
council_tier: T1
council_attendees: [WorkingGroup_PACT_NERV_IA3_LaneScriptDriftFix]
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_decisions_recorded:
  - "Driver script lines 63-69 stale override extincted per Catalog #240"
  - "Trainer docstring superseded per Catalog #110/#113 APPEND-ONLY"
  - "Canonical Z6 sister mode-resolution pattern adopted (Catalog #326)"
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
related_deliberation_ids:
  - council_per_substrate_symposium_pact_nerv_ia3_20260520T185500Z
  - pact_nerv_ia3_paired_cuda_modal_t4_dispatch_attempt_DEFER_landed_20260528
---

# PACT-NeRV-IA3 Lane Script Catalog #240 Drift Fix Landed 2026-05-28

## Summary

Extincted a stale-driver-belief override in
`scripts/remote_lane_substrate_pact_nerv_ia3.sh` lines 63-69 that forced
`PACT_NERV_IA3_SMOKE="1"` (smoke mode) even when the recipe correctly
declared full-mode (`PACT_NERV_IA3_SMOKE=0` + `SMOKE_ONLY=0`). The
override's docstring rationale ("trainer _full_main raises
NotImplementedError per Catalog #240") was stale post-commit `259292757`
(PACT-NERV-FULL-MAIN-IMPLEMENTATION-WAVE) which extincted the
`NotImplementedError` and wired the canonical
`tac.substrates._shared.pact_nerv_full_main` helper + score-aware loss +
`gate_auth_eval_call` Catalog #226 routing.

## Empirical Anchor

Per task #1436 DEFER landing memo
`pact_nerv_ia3_paired_cuda_modal_t4_dispatch_attempt_DEFER_landed_20260528`,
the paired-CUDA Modal T4 dispatch attempt for PACT-NeRV-IA3 surfaced 2
structural blockers; blocker #2 was this driver-script drift. Worker
stdout receipt:

```
[lane-pact-nerv-ia3-l0] WARNING: PACT_NERV_IA3_SMOKE=0; trainer _full_main
raises NotImplementedError per Catalog #240. Forcing smoke.
```

The recipe's explicit `PACT_NERV_IA3_SMOKE: "0"` + `SMOKE_ONLY: "0"`
env_overrides were silently overridden by the driver, identical META-class
to the Catalog #326 Z6-v2 Wave 2 DEFER anchor (call_ids
`fc-01KRW7RHFHP640BHTQ0FZM3M38` + `fc-01KRW7ZCYK5XF6MSHD24R71A46`).

## Root Cause

PRE-fix driver had 3 stale surfaces:

1. **Header docstring lines 22-26**: stated "Recipe MUST set
   PACT_NERV_IA3_SMOKE=1 OR SMOKE_ONLY=1" + "Full mode is unreachable
   from this driver because the trainer's _full_main raises
   NotImplementedError" — both clauses stale post-commit `259292757`.

2. **Mode-resolution block lines 57-69**: defaulted `PACT_NERV_IA3_SMOKE`
   to `${SMOKE_ONLY:-1}` (smoke-by-default) AND added two stale-override
   branches that FORCED `PACT_NERV_IA3_SMOKE="1"` whenever either
   `PACT_NERV_IA3_TRAINER_MODE=full` OR `PACT_NERV_IA3_SMOKE!=1` —
   making full-mode structurally unreachable even with a recipe that
   correctly opted in.

3. **Stage 4 trainer invocation lines 172-183**: unconditionally passed
   `--smoke` flag even when the resolved mode said full.

Trainer docstring lines 8-9, 41, 54 still claimed `_full_main raises
NotImplementedError` despite the function being IMPLEMENTED at commit
`259292757`.

## Fix

### Driver script (`scripts/remote_lane_substrate_pact_nerv_ia3.sh`)

Adopted the canonical Catalog #326 Z6 sister mode-resolution pattern
(per `scripts/remote_lane_substrate_time_traveler_l5_z6.sh` lines 87-105):

1. **Mode resolution lines 70-89**: canonical precedence
   `PACT_NERV_IA3_TRAINER_MODE` > `PACT_NERV_IA3_SMOKE` > `SMOKE_ONLY` >
   default=smoke (with loud WARN if nothing set). `case` statement
   handles `smoke|SMOKE|Smoke` and `full|FULL|Full`; FATAL on invalid
   token.
2. **Stage 4 trainer invocation lines 183-208**: conditional `--smoke`
   flag based on resolved mode (canonical Z6 sister pattern using
   `TRAIN_FLAG_ARGS` bash array + `${ARR[@]+"${ARR[@]}"}` empty-array
   guard per CLAUDE.md "Forbidden silent-skip cascades" + Catalog #189).
3. **Header docstring lines 22-44**: APPEND-ONLY supersession marker
   citing commit `259292757` + Catalog #240 + #326 canonical pattern.

### Trainer (`experiments/train_substrate_pact_nerv_ia3.py`)

Per Catalog #110/#113 HISTORICAL_PROVENANCE APPEND-ONLY discipline:
preserved original SCAFFOLD claim verbatim + appended 2026-05-28
supersession marker citing commit `259292757` extinction of the
`NotImplementedError` at the function-body level. Stale "raises
NotImplementedError" claims at lines 8-9, 41, 54 superseded with
explicit "IS IMPLEMENTED post-2026-05-28 commit 259292757" text.

## Verification

### Catalog #326 Audit (Empirical Receipt)

```
PRE-fix:  CONSUMES_ENV_MULTI_KEY_DEFAULT_RECIPE_OK
POST-fix: CONSUMES_ENV_NO_HARDCODE
Bug class drivers: 0 (PRE) -> 0 (POST; better verdict class)
```

Verdict class improvement: PRE-fix was "MULTI_KEY_DEFAULT_RECIPE_OK"
(safe because recipes opt out) → POST-fix is "NO_HARDCODE" (canonical
no-hardcode-at-all pattern; mode resolved purely from env vars per
recipe). Both verdicts are bug-class-clean per Catalog #326; the fix is
a quality improvement at the mode-resolution surface.

### Empirical Mode-Resolution Tests (4 PASS)

```
TEST_1 paired_recipe_full_PACT_NERV_IA3_SMOKE=0_SMOKE_ONLY=0: resolved=0 (expect 0=full)  PASS
TEST_2 smoke_recipe_PACT_NERV_IA3_SMOKE=1:                    resolved=1 (expect 1=smoke) PASS
TEST_3 canonical_TRAINER_MODE=full:                           resolved=0 (expect 0=full)  PASS
TEST_4 nothing_set:                                           resolved=1 (expect 1=smoke) PASS
```

### Shell Syntax

`bash -n scripts/remote_lane_substrate_pact_nerv_ia3.sh` → OK.

## Recipe Status (No Changes Needed)

`.omx/operator_authorize_recipes/substrate_pact_nerv_ia3_modal_t4_paired_dispatch.yaml`
already declares the canonical full-mode env_overrides at lines 264-266:

```yaml
PACT_NERV_IA3_DEVICE: cuda
PACT_NERV_IA3_SMOKE: "0"
SMOKE_ONLY: "0"
```

Post-fix the driver correctly honors these settings (verified via TEST_1).

## Catalog #325 14-Day Window

PACT-NERV-DESIGN-SYMPOSIUM dated 2026-05-20 + 14d = **2026-06-03**.
Today is 2026-05-28. **6 days remaining** in symposium window.

## Mission Contribution

`frontier_protecting`: extincts the driver-script-drift bug class for
PACT-NeRV-IA3 specifically; unblocks the paired-CUDA Modal T4 dispatch
re-attempt within the Catalog #325 symposium window; prevents future
recurrence at this driver surface.

## Operator-Routable TOP-1 Next-Step

POST-fix, the paired-CUDA Modal T4 paired-dispatch is unblocked. Sister
operator-attended dispatch command per task #1436 DEFER landing:

```bash
# Operator must run interactively (paired-env bypass requires explicit
# rationale per Catalog #199; canonical entry point is operator_authorize).
.venv/bin/python tools/operator_authorize.py \
    --recipe substrate_pact_nerv_ia3_modal_t4_paired_dispatch \
    --paired-axes
```

Per CLAUDE.md "Executing actions with care": ONLY operator can flip
`research_only: false` + `dispatch_enabled: true` on the paired-dispatch
recipe (operator_override_rationale already declares the canonical
2026-05-28 verbatim quote per Catalog #300 Consequence 1). The Catalog
#325 14-day window remains open through 2026-06-03 (6 days remaining).

## Sister-Coherence (Catalog #340)

Disjoint scope from sister subagents:
- Slot 2 COMBINED-AUDIT-WAVE: cathedral autopilot + integration audit
  (no overlap with this driver script or trainer)
- Task #1437 owns ONLY: `scripts/remote_lane_substrate_pact_nerv_ia3.sh`
  + `experiments/train_substrate_pact_nerv_ia3.py` (docstring fix only;
  no `_full_main` body changes) + this landing memo

## Cross-References

- Catalog #240 (recipe-vs-trainer-state-consistency; the bug class this fix extincts)
- Catalog #326 (driver-mode-hardcode-extinction; the canonical pattern adopted)
- Catalog #110/#113 (HISTORICAL_PROVENANCE APPEND-ONLY; trainer docstring discipline)
- Catalog #325 (per-substrate symposium; 14-day window 2026-06-03 still open)
- Catalog #300 Consequence 1 (operator-frontier-override on paired-dispatch recipe)
- Commit `259292757` (PACT-NERV-FULL-MAIN-IMPLEMENTATION-WAVE; the
  `NotImplementedError` extinction)
- Sister driver `scripts/remote_lane_substrate_time_traveler_l5_z6.sh`
  lines 87-105 (canonical Z6 mode-resolution pattern)
- Task #1436 paired-CUDA dispatch attempt DEFER landing memo
- `feedback_driver_fix_smoke_hardcode_plus_new_catalog_gate_cross_substrate_audit_landed_20260518.md`
  (Catalog #326 canonical META-class anchor)

## 6-Hook Wire-In Declaration (Catalog #125)

- hook #1 sensitivity-map: N/A (driver-script engineering fix)
- hook #2 Pareto constraint: N/A
- hook #3 bit-allocator: N/A
- hook #4 cathedral autopilot dispatch: **ACTIVE** (unblocks paired-CUDA
  dispatch path for PACT-NeRV-IA3; sister cathedral consumer awaits
  Phase 5 dispatch + Phase 6 archive)
- hook #5 continual-learning posterior: N/A (no anchor lands here; the
  paid dispatch (operator-routable) is what lands an anchor)
- hook #6 probe-disambiguator: N/A

## Discipline Verification

- Catalog #229 PV: ✅ read full driver + trainer + recipe + 2 sister
  canonical drivers (Z6 + DS-NeRV) BEFORE editing
- Catalog #117/#157/#174/#235 canonical serializer: ✅ will commit via
  `tools/subagent_commit_serializer.py` with POST-EDIT
  `--expected-content-sha256` for both files
- Catalog #206 crash-resume: ✅ 2 checkpoints landed (Phase 1+2 PV
  complete; Phase 3+4 fix landed verified)
- Catalog #110/#113 APPEND-ONLY: ✅ trainer docstring preserves original
  SCAFFOLD claim verbatim + appends 2026-05-28 supersession marker
- Catalog #287 placeholder rejection: ✅ all waivers ≥4 chars substantive
- Catalog #340 sister-checkpoint guard: ✅ disjoint scope from Slot 2
  sister; own only 2 source files + this memo
- Catalog #208 docs/local-paths: ✅ only scratch context `/tmp/` for test
  scripts (excluded by gate per `/tmp/` substring inside word marker)
- 7th META AUTOMATED+COMPOUNDING+OPTIMAL: ✅ canonical Z6 pattern
  PROPAGATED (automated re-use); FIX COMPOUNDS (extincts bug class for
  PACT-NeRV-IA3); OPTIMAL (one canonical pattern vs ad-hoc hack)
- 8th MLX-first standing directive: ✅ $0 GPU verified throughout (this
  fix is engineering-only; no MLX/CUDA/Modal dispatches fired)
- 11th INDIVIDUALLY-FRACTAL: ✅ fix is PER-SUBSTRATE for PACT-NeRV-IA3
  specifically (NOT shared-helper update; each substrate driver has its
  own canonical mode-routing per its specific architecture's semantics)
- 13th OPTIMAL-TRIO: ✅ minimal LOC delta; canonical pattern adopted;
  empirically verified

$0 GPU verified. ~30 min wall-clock.
