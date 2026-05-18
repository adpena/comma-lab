# Z6 Full Canary Smoke-Misroute Terminal Claim - 2026-05-18

## Scope

This ledger terminalizes the stale active dispatch claim for the Z6-v2 Candidate
1 Wave 2 full canary after the Modal call was already harvested as a
non-promotable smoke-path execution.

## Registry Action

- Lane: `lane_z6_v2_candidate_1_wave_2_build_trainer_extension_and_recipe_20260517`
- Instance/job: `substrate_z6_v2_candidate_1_multi_layer_film_modal_t4_smoke_dispatch_20260518T003206Z__full__100ep`
- Platform: `modal`
- Call id: `fc-01KRW7ZCYK5XF6MSHD24R71A46`
- Terminal row written to `.omx/state/active_lane_dispatch_claims.md`
- Terminal status: `failed_z6_full_canary_driver_smoke_mode_misroute`
- Verification: `.venv/bin/python tools/claim_lane_dispatch.py summary --live-only --format json` reported `active_count=0`

## Evidence

The Modal ledger already contained matching dispatch and harvest rows in
`.omx/state/modal_call_id_ledger.jsonl`.

- Dispatch row:
  - `event_type=dispatched`
  - `expected_axis=cuda`
  - `gpu=T4`
  - `max_seconds=7200`
  - `mounted_code_git_head=f4f6c379c4253b5bd4cdc3d4a8988e72362fffaa`
- Harvest row:
  - `event_type=harvested`
  - `rc=0`
  - `elapsed_seconds=9.137`
  - `score=null`
  - `score_axis=null`
  - `archive_bytes=null`
  - `archive_sha256=null`
  - notes state that the driver forced `_smoke_main` / `smoke=1`
  - notes state that the run built the 27850-parameter synthetic cfg, not the council-binding depth=3 hidden_dim=96 full spec

Local metadata exists at
`experiments/results/lane_substrate_z6_v2_candidate_1_multi_layer_film_modal_t4_smoke_dispatch_20260518T003206Z__full__100ep_modal/modal_metadata.json`
and points to the same call id, lane id, label, and lane script.

## Classification

This is an implementation/custody failure:

- `score_claim=false`
- `promotion_eligible=false`
- no `[contest-CUDA]` score
- no `[contest-CPU]` score
- no `[macOS-CPU advisory]` score
- no archive identity suitable for byte-closed promotion
- no method-negative conclusion for Z6-v2 Candidate 1

The measured failure is only that the allegedly full canary followed the smoke
trainer path. The full council-binding Z6 architecture was not empirically
tested by this call.

## Reactivation Criteria

Before refiring this lane, require all of the following:

1. Driver proof that full/smoke mode is controlled by the operator recipe or
   explicit environment, not hardcoded to smoke.
2. A local or dry-run proof that the full path builds the intended depth=3,
   hidden_dim=96, roughly 300K-parameter architecture instead of the 27850-param
   synthetic cfg.
3. A clean live dispatch summary before any new Modal/Vast/Lightning spend.
4. A fresh lane claim for the new job id before provider creation.
5. Harvest evidence that records archive bytes/SHA or a precise pre-score
   failure class.

## Current Local Driver Proof

After terminalizing the stale claim, the focused local driver test was updated
to match the repaired contract:

- `Z6_TRAINER_MODE=full` overrides legacy `SMOKE_ONLY=1`
- full mode does not pass `--smoke` to the trainer
- provenance records `smoke_only=0`
- Wave 2 architecture, target param count, ego source, paired-control init, and
  disambiguator flags still reach trainer argv

Verification:

```bash
.venv/bin/python -m pytest src/tac/tests/test_time_traveler_l5_z6_remote_driver.py -q
```

Result: `2 passed in 0.24s`.

## Six-Hook Wire-In

- Sensitivity map: no empirical component deltas; hold Z6 sensitivity unchanged.
- Pareto constraint: no Pareto status change because no valid score axis exists.
- Bit allocator: no allocation update; archive bytes/SHA are null.
- Cathedral autopilot dispatch: stale claim cleared; future Z6 dispatch must pass
  the driver-mode proof above.
- Continual-learning posterior: update only the dispatcher reliability prior for
  `remote_lane_substrate_time_traveler_l5_z6.sh` full/smoke routing.
- Probe-disambiguator: the arbitration is driver-mode proof before refire, not a
  Candidate 1 vs Candidate 4c method pivot.
