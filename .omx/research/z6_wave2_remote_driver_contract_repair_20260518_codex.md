---
review_kind: dispatch_contract_repair
review_id: z6_wave2_remote_driver_contract_repair_20260518_codex
review_date: "2026-05-18"
lane_id: lane_z6_v2_candidate_1_wave_2_build_trainer_extension_and_recipe_20260517
substrate: z6_v2_candidate_1_multi_layer_film
evidence_axis: pre_dispatch_contract_guard
score_claim: false
promotion_eligible: false
provider_spend: false
research_only: true
---

# Z6 Wave 2 Remote Driver Contract Repair

## Summary

The Z6 Wave 2 recipe declared the multi-layer FiLM architecture and identity
disambiguator env overrides, but the shared Z6 remote driver still forwarded
only the Z6-v1 trainer flags. A Wave 2 smoke launched through this driver would
either verify the wrong lane claim, run the single-layer default architecture,
or omit the identity-predictor disambiguator archive. That would make any
returned result a harness/config artifact rather than a valid Wave 2 empirical
anchor.

No provider job was dispatched. No [contest-CUDA] or [contest-CPU] score claim
is made here.

## Repair

- `scripts/remote_lane_substrate_time_traveler_l5_z6.sh` now accepts
  `Z6_LANE_ID`/`PACT_DISPATCH_LANE_ID` instead of hard-coding the Z6-v1 lane.
- The driver records the actual `Z6_RECIPE_PATH` in provenance.
- The driver forwards the Wave 2 trainer flags:
  `--predictor-architecture`, `--predictor-param-count-target`, `--ego-source`,
  `--enable-paired-control-initialization`,
  `--paired-control-disambiguator-decision-criterion-delta-s`, and the
  boolean `--emit-identity-predictor-disambiguator-archive`.
- The heartbeat cleanup now kills and waits on the heartbeat child so scripted
  invocations do not hang on inherited stdout/stderr pipes after successful
  driver exit.
- The Wave 2 recipe now supplies `Z6_LANE_ID`, `Z6_RECIPE_PATH`, and a Wave 2
  `TAG` through `env_overrides`.

## Verification

- `bash -n scripts/remote_lane_substrate_time_traveler_l5_z6.sh`
- `.venv/bin/python -m pytest src/tac/tests/test_time_traveler_l5_z6_remote_driver.py -q`
  - `2 passed`
- `.venv/bin/python -m pytest src/tac/substrates/time_traveler_l5_z6/tests/test_multi_layer_film_predictor.py src/tac/tests/test_z6_v2_candidate_1_wave_2_build.py src/tac/tests/test_time_traveler_l5_z6_remote_driver.py -q`
  - `36 passed`
- `.venv/bin/python tools/audit_predicted_band_provenance.py --strict`
  - `Recipes scanned: 73`
  - `In-scope: 19`
  - `PASS: 19`
  - `FAIL: 0`
- `git diff --check -- scripts/remote_lane_substrate_time_traveler_l5_z6.sh src/tac/tests/test_time_traveler_l5_z6_remote_driver.py .omx/operator_authorize_recipes/substrate_z6_v2_candidate_1_multi_layer_film_modal_t4_smoke_dispatch.yaml`
  - clean

## Six-Hook Wire-In

1. Sensitivity-map contribution: N/A. This is a dispatch-contract repair; no
   new empirical anchor or sensitivity tensor.
2. Pareto constraint: protects the declared Wave 2 rate/distortion hypothesis
   by preventing accidental Z6-v1 default execution.
3. Bit-allocator hook: N/A. No archive bytes changed.
4. Cathedral autopilot dispatch hook: active guardrail. The remote driver now
   consumes the recipe lane/env contract that operator-authorized dispatches
   rely on.
5. Continual-learning posterior update: this ledger records the harness defect
   class and its repair before any paid smoke.
6. Probe-disambiguator: active. The identity-predictor disambiguator flag is
   now actually forwarded to the trainer, preserving the Wave 2 arbitration
   contract before any Wave 3 decision.

## Reactivation Criteria

The Z6 Wave 2 smoke remains non-promotional until a claimed dispatch produces
paired custody artifacts and the identity-vs-full predictor disambiguator is
reviewed on its declared evidence axis. Any future Wave 2 result lacking
`predictor_architecture=multi_layer_film_depth_3_300k`,
`emit_identity_predictor_disambiguator_archive=true`, the Wave 2 lane id, and
the Wave 2 recipe path in provenance must be classified as
`indeterminate_harness_or_runtime_mismatch`, not as a method result.
