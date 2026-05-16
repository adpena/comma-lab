# L5 v2 remote smoke and asymptotic semantics hardening

- schema: `l5_v2_remote_smoke_and_asymptotic_semantics_hardening_v1`
- created_at_utc: `2026-05-16T22:45:00Z`
- score_claim: `false`
- promotion_eligible: `false`
- ready_for_exact_eval_dispatch: `false`

## Issue

Two small but dispatch-facing L5 v2 defects were still live after the latest
asymptotic lane-registry hardening:

1. `tools/all_lanes_preflight.py` only rejected ambiguous
   `ready_for_l1_build_semantics` when `ready_for_l1_build=true`. A stale row
   with an already-present L1 scaffold could still carry ambiguous semantics or
   mark the completed action as ready again.
2. `scripts/remote_lane_substrate_time_traveler_l5_z6.sh` defaulted
   `Z6_EPOCHS=300` even on the default `SMOKE_ONLY=1` path. The trainer caps
   smoke internally, but the remote provenance and operator surface still
   advertised the full-run epoch count for a smoke dispatch.
3. The TT5L paired-axis plan action still exposed a hand-written
   `claim_lane_dispatch.py ... && <run ...>` template. That is non-executable
   placeholder prose, but it pushes operators toward manual single-lifecycle
   dispatch instead of the canonical paired CPU/CUDA dispatcher.
4. The asymptotic candidate surface needed a compact machine-readable
   next-action status object so operator briefing and preflight can compare
   ledger SHA, lane-registry state, first-artifact status, and prerequisite
   readiness without re-inferring them from prose.
4. The L5 v2 asymptotic candidate payload carried most custody facts as flat
   compatibility fields, but did not expose a named
   `l5_v2_asymptotic_next_action_status` contract with the exact
   `ledger_present`, `ledger_sha256`, and structured `next_prerequisite_status`
   fields needed by operator briefing, preflight, and future autopilot consumers.

## Fix

- Hardened the operator-briefing dispatch gate to validate the semantic enum on
  every L5 v2 asymptotic candidate, reject completed L1 scaffolds with
  ready-to-start semantics, and reject completed/superseded actions that still
  present as ready for the recommended next action.
- Added `l5_v2_asymptotic_next_action_status_v1` rows to the L5 v2 candidate
  surface and threaded them through `tools/operator_briefing.py` and
  `tools/all_lanes_preflight.py`.
- Changed the Z6 remote driver so `SMOKE_ONLY=1` defaults `Z6_EPOCHS=3`, while
  full mode still defaults to `300` when the operator explicitly lifts the
  non-smoke path.
- Replaced the TT5L paired-axis action template with
  `tools/dispatch_modal_paired_auth_eval.py`, marked standalone preclaim as
  forbidden, and recorded the paired dispatcher/wrappers as claim lifecycle
  owners.
- Added a named `l5_v2_asymptotic_next_action_status` surface while preserving
  the existing flat candidate fields for Cathedral/autopilot compatibility.
  The status contract fails closed when a source ledger exists but the lane id is
  absent from `.omx/state/lane_registry.json`, unless a future canonical
  replacement lane is explicitly marked registered.
- Added regression tests for both surfaces.

## Evidence

- `src/tac/tests/test_all_lanes_operator_briefing_gate.py`
- `src/tac/tests/test_time_traveler_l5_z6_remote_driver.py`
- `src/tac/tests/test_l5_staircase_v2.py`

No score, rank, or dispatch claim is made by this hardening. This only removes
stale-authority and smoke-cost ambiguity before the next Z6/L5 v2 actuation.
