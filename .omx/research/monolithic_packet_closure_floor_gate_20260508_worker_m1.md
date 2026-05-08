# Monolithic Packet Closure/Floor Gate - Worker M1 - 2026-05-08

Evidence grade: `empirical_guard_no_score`.
Score claim: false.
Dispatch performed: false.

## Scope

Implemented the dispatch-readiness gate recommended by the cross-paradigm
dispatch readiness review and the monolithic bridge review. The gate is a
reusable JSON-artifact guard for monolithic HNeRV section-replacement
candidates before exact CUDA spend.

## Guard Contract

The new gate fails closed unless all dispatch-critical closure facts are bound:

- parser-proven logical section mutation exists in the candidate manifest;
- runtime proof binds the exact candidate archive SHA-256, rebuilt member
  SHA-256, command/log SHA-256s, and every changed logical section SHA-256;
- an active Level-2 dispatch claim is exported from the canonical claim file,
  unless the caller explicitly sets dry-run mode;
- rate-only candidates at or above the active `185578` byte PR103-on-PR106
  A++ floor are marked non-dispatchable before exact-eval spend.

Dry-run mode can review closure without a lane claim, but it never sets
`ready_for_exact_eval_dispatch=true`.

## Implementation

- `src/tac/monolithic_packet_closure_gate.py` adds the reusable pure guard.
- `tools/check_monolithic_packet_closure_gate.py` exposes the guard as a
  deterministic CLI.
- `tools/run_monolithic_candidate_preflight.py` now delegates to the closure
  gate instead of trusting a manifest's own readiness flag.
- `tools/build_monolithic_runtime_consumption_proof.py` now emits
  `rebuilt_member_sha256` alongside the existing `new_member_sha256` field.
- `src/tac/monolithic_packet_candidate.py` now blocks readiness when runtime
  proof does not bind the rebuilt member SHA-256.

## Verification

- `.venv/bin/python -m pytest -q src/tac/tests/test_monolithic_packet_closure_gate.py`
  - `6 passed`
- `.venv/bin/python -m pytest -q src/tac/tests/test_run_monolithic_candidate_preflight.py`
  - `6 passed`
- `.venv/bin/python -m pytest -q src/tac/tests/test_build_monolithic_runtime_consumption_proof.py`
  - `3 passed`
- `.venv/bin/python -m pytest -q src/tac/tests/test_monolithic_packet_candidate.py`
  - `13 passed`
- `.venv/bin/python -m pytest -q src/tac/tests/test_monolithic_packet_closure_gate.py src/tac/tests/test_monolithic_packet_candidate.py src/tac/tests/test_build_monolithic_runtime_consumption_proof.py src/tac/tests/test_run_monolithic_candidate_preflight.py src/tac/tests/test_monolithic_codec_op_replacement.py`
  - `36 passed`
- `.venv/bin/python -m ruff check src/tac/monolithic_packet_closure_gate.py tools/check_monolithic_packet_closure_gate.py src/tac/monolithic_packet_candidate.py tools/run_monolithic_candidate_preflight.py tools/build_monolithic_runtime_consumption_proof.py src/tac/tests/test_monolithic_packet_closure_gate.py src/tac/tests/test_run_monolithic_candidate_preflight.py src/tac/tests/test_build_monolithic_runtime_consumption_proof.py src/tac/tests/test_monolithic_packet_candidate.py`
  - `All checks passed`
- `.venv/bin/python tools/check_monolithic_packet_closure_gate.py --candidate-manifest experiments/results/monolithic_stack_candidate_pr106x_lgblock16_20260508_codex/manifest_from_replacement_manifest.json --dry-run`
  - blocked on `runtime_consumption_proof_missing`;
  - blocked on `rate_only_candidate_not_below_active_pr103_pr106_a_plus_plus_floor:185578`;
  - lane-claim absence was allowed only because this was dry-run mode.

## Dispatch Status

No candidate was dispatched. The reviewed `pr106x_lgblock16` monolithic
candidate remains a valid archive-construction control but is non-dispatchable
as-is: it is rate-only at `186079` bytes, above the active `185578` byte exact
floor, and still lacks runtime-consumption proof.

## Codex Integration Update - 2026-05-08T13:34Z

Runtime-consumption proof is now present for the same candidate:

- runtime probe:
  `experiments/results/monolithic_stack_candidate_pr106x_lgblock16_20260508_codex/runtime_consumption_proof_pr106x_lgblock16.json`
- canonical proof-from-log:
  `experiments/results/monolithic_stack_candidate_pr106x_lgblock16_20260508_codex/runtime_consumption_proof_from_log_pr106x_lgblock16.json`
- runtime log:
  `experiments/results/monolithic_stack_candidate_pr106x_lgblock16_20260508_codex/runtime_consumption_pr106x_lgblock16.log`

The refreshed closure/preflight artifacts bind the candidate archive SHA,
rebuilt member SHA, and changed `decoder_packed_brotli` section SHA; runtime
blockers are empty. The candidate is still **not dispatchable**:

- dry-run closure blocker:
  `rate_only_candidate_not_below_active_pr103_pr106_a_plus_plus_floor:185578`
- non-dry-run closure blockers:
  `active_lane_claim_missing` and
  `rate_only_candidate_not_below_active_pr103_pr106_a_plus_plus_floor:185578`

This supersedes the earlier `still lacks runtime-consumption proof` sentence
for this candidate. It remains a non-dispatchable rate-only control, not a
score-lowering exact-eval target.
