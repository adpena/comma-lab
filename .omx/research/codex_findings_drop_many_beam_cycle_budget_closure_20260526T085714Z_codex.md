# Codex Findings - Drop-Many Beam + Cycle Budget Closure - 2026-05-26T08:57:14Z

## Verdict

This tranche closes two drift points in the rate-attack loop:

1. `drop_many_beam_pairwise_interaction_waterfill` is no longer only a
   selector name with a skeleton probe. It now has a reusable local-planning
   beam helper with Dykstra feasibility and waterfill budget accounting.
2. The frontier feedback cycle no longer stops after building a targeted
   component-correction queue. It now also writes the queue-owned response
   harvest, materialization requests, and operation-chain work orders so freed
   rate budget keeps moving toward targeted SegNet/PoseNet correction without
   manual refresh glue.

## What changed

- Added `tac.optimization.dqs1_drop_many_beam` as the reusable local-only beam
  primitive for pairset drop-many planning.
- Updated decoder-q pairset acquisition so the
  `drop_many_beam_pairwise_interaction_waterfill` selector uses the executable
  beam/Dykstra/waterfill path before falling back to the older bounded tail
  heuristic.
- Updated `tools/probe_dqs1_drop_many_beam_pairwise_interaction_disambiguator.py`
  so the probe emits executable zero-interaction beam candidates instead of
  stopping at `NotImplementedError` stubs. BUILD-1 empirical interaction matrix
  population remains the next calibration step.
- Updated `frontier_rate_attack_feedback_cycle.py` so targeted correction
  acquisition automatically produces response harvest, materialization request,
  and targeted operation-chain work-order artifacts in the same cycle.
- Added cycle integration edges for:
  - `targeted_component_correction_queue_to_response_harvest_and_materialization_requests`
  - `targeted_component_materialization_requests_to_operation_chain_queue`

## Authority

All new signals remain false-authority:

- no score claim;
- no promotion authority;
- no rank/kill authority;
- no exact-eval dispatch readiness;
- no targeted correction budget spend until runtime proof, component replay,
  total Lagrangian improvement, and exact auth evidence exist.

## Verification

- `ruff` on touched beam, acquisition, probe, cycle, CLI, and test files: passed.
- `pytest src/tac/tests/test_frontier_rate_attack_feedback.py -q`: 31 passed.
- `pytest src/tac/tests/test_dqs1_drop_many_beam.py src/tac/tests/test_decoder_q_pairset_acquisition.py src/tac/tests/test_dqs1_local_first_queue_builder.py -q`: 64 passed.
- `tools/probe_dqs1_drop_many_beam_pairwise_interaction_disambiguator.py --json`
  emits `dqs1_drop_many_beam_pairwise_interaction_waterfill_probe.v1` with
  executable beam candidates and explicit false-authority markers.

## Remaining Work

Highest-EV next step is still the non-leaf closure path: execute the targeted
component correction queue locally where safe, harvest measured response rows,
then allow only component-improving rows to become materialization requests and
operation-chain candidates. The beam helper should also receive a real BUILD-1
pairwise interaction matrix from existing fp64/master-gradient evidence, but
that is selector-quality calibration rather than the primary budget-spend
autonomy blocker.
