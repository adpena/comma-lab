# Z6 Wave 2 Re-Fire Probe-Predecessor Blocker - 2026-05-18

## Scope

This ledger records the attempted Z6-v2 Candidate 1 Wave 2 re-fire after the
Catalog #326 driver-mode repair landed at commit `c10dec618`.

No Modal provider job was created in this attempt. `operator_authorize.py`
refused before lane-claim/provider setup because Catalog #313 still contains a
blocking predecessor probe verdict for the previous Z6 Wave 2 dispatch attempt.

## Pre-Dispatch Evidence

- Branch/head: `main` / `c10dec618`
- Z6 repair landing: `.omx/research/z6_v2_wave_2_landing_state_snapshot_20260517.md`
- Active dispatch summary immediately before launch attempt:
  - `active_count=0`
  - `stale_nonterminal_count=0`
- Catalog #326 audit:
  - `.venv/bin/python tools/audit_substrate_driver_mode_hardcode.py --format summary`
  - `Bug class count: 0`
- Focused tests:
  - `.venv/bin/python -m pytest src/tac/tests/test_time_traveler_l5_z6_remote_driver.py src/tac/tests/test_check_326_substrate_driver_consumes_trainer_mode_env_var.py -q`
  - `35 passed in 1.13s`
- Z6 committed sentinel set was clean at the repair commit before launch.

## Attempted Command

```bash
OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1 \
OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=3.00 \
OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK=1 \
OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED='Z6 repair bundle committed at c10dec618; Z6 sentinel files clean in git status; unrelated partner WIP only; Catalog166 worker-side sentinel hash check remains active' \
.venv/bin/python tools/run_modal_smoke_before_full.py \
  --recipe substrate_z6_v2_candidate_1_multi_layer_film_modal_t4_smoke_dispatch \
  --operator-handle codex:z6_wave2_refire_after_catalog326_20260518
```

## Refusal

`operator_authorize.py` failed closed before provider creation:

- substrate: `time_traveler_l5_z6_v2_candidate_1_multi_layer_film_depth_3_300k`
- probe id: `z6_v2_wave_2_dispatch_smoke_before_full_paired_2026_05_18`
- blocking verdict: `DEFER`
- metric: `wave_2_dispatch_outcome_implementation_path_executed=0.0`
- threshold: `1.0`
- adjudicated at: `2026-05-18T00:34:23Z`
- staleness expiry: `2026-06-17T00:34:23Z`
- refusal class: `failed_pre_provider_catalog313_predecessor_probe_blocking_defer`

The refusal is correct as a guard behavior. The previous probe outcome remains
blocking until a fresh council/operator override, a superseding probe row, or a
more specific alternative dispatch path clears it.

## Why I Did Not Override

The Z6 landing snapshot states that the next operator routable is a fork:

- Candidate 1 re-fire (`$2.50`)
- Candidate 4c scorer-logit conditioning (`$2.50`, higher leverage for both
  Z6 and ATW V2 per the ATW V2 symposium)

Clearing Catalog #313 via `OPERATOR_AUTHORIZE_PROBE_PREDECESSOR_BYPASS_VERDICT`
or by appending a probe-ledger `operator_override` row would silently choose
Candidate 1 over Candidate 4c. That is not a custody-safe action because the
current `.omx` state explicitly frames it as an operator decision fork.

## Current Dispatchability

Candidate 1 is mechanically launch-ready after an explicit decision:

1. Driver-mode bug repaired and committed at `c10dec618`.
2. Recipe env overrides set `Z6_TRAINER_MODE="full"` and `SMOKE_ONLY="0"`.
3. Focused driver tests prove full mode does not pass `--smoke`.
4. Catalog #326 reports zero live bug-class findings.
5. Active dispatch claims are clean.

The remaining blocker is not a code readiness blocker. It is an explicit
dispatch-choice/custody blocker:

`NEEDS-OPERATOR-DECISION: Candidate 1 re-fire versus Candidate 4c scorer-logit path.`

## Six-Hook Wire-In

- Sensitivity map: no empirical deltas; no update.
- Pareto constraint: no score axis exists; no Pareto status change.
- Bit allocator: no archive bytes/SHA; no allocator update.
- Cathedral autopilot dispatch: keep Candidate 1 refused by Catalog #313 until
  operator choice or probe-ledger override; Candidate 4c should remain a
  first-class alternative in the queue.
- Continual-learning posterior: update dispatcher reliability only with the
  fail-closed classification above.
- Probe-disambiguator: Catalog #313 is functioning; next action is either
  explicit predecessor override for Candidate 1 or build/dispatch the Candidate
  4c scorer-logit disambiguator path.
