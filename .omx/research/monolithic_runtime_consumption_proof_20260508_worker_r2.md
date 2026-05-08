# Monolithic Runtime-Consumption Proof - Worker R2 - 2026-05-08

Evidence grade: `empirical_runtime_consumption_no_score`.
Score claim: false.
Dispatch performed: false.
GPU claim or remote job: none.

## Scope

Built the next concrete runtime-consumption proof path for monolithic HNeRV
section candidates. The path is CPU-only and read-only with respect to dispatch
state: it consumes the exact single ZIP member using the PR106 runtime wire
grammar, verifies the Brotli logical sections decode, emits a deterministic
runtime log, and converts that log into the strict
`tac_runtime_consumption_proof_v1` consumed by the monolithic closure gate.

## Candidate Advanced

Candidate manifest:
`experiments/results/monolithic_stack_candidate_pr106x_lgblock16_20260508_codex/manifest_from_replacement_manifest.json`

- candidate archive:
  `experiments/results/monolithic_stack_candidate_pr106x_lgblock16_20260508_codex/pr106x_lgblock16_monolithic_candidate_from_manifest.zip`
- archive bytes: `186079`
- archive SHA-256:
  `866dc135e9168d61fab02b6b1c218c4b1d6eed779154a6dc3095fd05e48024f2`
- rebuilt member SHA-256:
  `0a83096defc59120ee551c45e73f69e089165df78ae706fbbe2be3e9bc284765`
- runtime-consumed changed section: `decoder_packed_brotli`
- changed section SHA-256:
  `a812f1e837afd0e463a7f133b680ea6c027339ff8816db7012dd41253435afbf`

## New Proof Artifacts

- Runtime-style proof emitter output:
  `experiments/results/monolithic_stack_candidate_pr106x_lgblock16_20260508_codex/runtime_consumption_proof_pr106x_lgblock16.json`
  - artifact SHA-256:
    `7fa74ad8045ae4636fb0706cbc9f21d56324cfcf1f0895c0afe801c08c674329`
  - `ready_for_exact_eval_runtime=true`
  - `blockers=[]`
  - confirms `ff_header`, `decoder_packed_brotli`, and
    `latents_and_sidecar_brotli` are parsed from the scored member
  - confirms both PR106 Brotli streams decompress in the runtime-style parser
- Runtime log:
  `experiments/results/monolithic_stack_candidate_pr106x_lgblock16_20260508_codex/runtime_consumption_pr106x_lgblock16.log`
  - log SHA-256:
    `f7396c712e3ac4574710c8c3152d9c2b629edf890963fb6245762f229a136f23`
  - contains the exact candidate archive SHA, rebuilt member SHA, and changed
    section SHA tokens required by the canonical proof builder
- Canonical proof built from the runtime log:
  `experiments/results/monolithic_stack_candidate_pr106x_lgblock16_20260508_codex/runtime_consumption_proof_from_log_pr106x_lgblock16.json`
  - artifact SHA-256:
    `81f6ebbca792de92d7d075b0d1b5c65bbc8a9ca068a40f22099e79a2447093b8`
  - `ready_for_exact_eval_runtime=true`
  - `blockers=[]`
  - binds archive SHA, rebuilt member SHA, command SHA, log SHA, and
    `decoder_packed_brotli` section SHA
- Closure gate with runtime proof and no lane claim:
  `experiments/results/monolithic_stack_candidate_pr106x_lgblock16_20260508_codex/closure_gate_with_runtime_proof_no_claim_pr106x_lgblock16.json`
  - artifact SHA-256:
    `71e632ab7bd7f74231c15e0f661ccb38c647589ff44be76507c4cabc2844a3f2`
  - `runtime_blockers=[]`
  - remaining blockers:
    `active_lane_claim_missing`,
    `rate_only_candidate_not_below_active_pr103_pr106_a_plus_plus_floor:185578`
- Read-only candidate preflight with runtime proof and no lane claim:
  `experiments/results/monolithic_stack_candidate_pr106x_lgblock16_20260508_codex/preflight_with_runtime_proof_no_claim_pr106x_lgblock16.json`
  - artifact SHA-256:
    `b646a941c81b78dc4c8ebd6e4a4bdf8378b7cf030c23f11bc639560cc2d00c0a`
  - `runtime_proof.ready_for_exact_eval_runtime=true`
  - `closure_gate.runtime_blockers=[]`
  - remaining blockers match the closure gate above

Dry-run gate/preflight artifacts were also emitted to show the lane-claim
absence can be waived only for review; the rate-only floor blocker remains.

## Outcome

The candidate progressed from byte-anchor/archive-construction evidence to a
strict runtime-consumption proof: the missing runtime log/SHA binding is now
closed for the `decoder_packed_brotli` section on the exact archive bytes.

This does not make the candidate dispatchable. It is still a rate-only
`186079` byte archive, above the active `185578` byte A++ floor, and no active
Level-2 dispatch claim exists. No exact CUDA eval was run, and no score,
promotion, rank, or kill claim is permitted from these artifacts.

## Commands

```bash
.venv/bin/python tools/prove_monolithic_runtime_consumption.py --candidate-manifest experiments/results/monolithic_stack_candidate_pr106x_lgblock16_20260508_codex/manifest_from_replacement_manifest.json --runtime-log-out experiments/results/monolithic_stack_candidate_pr106x_lgblock16_20260508_codex/runtime_consumption_pr106x_lgblock16.log --command-text ".venv/bin/python tools/prove_monolithic_runtime_consumption.py --candidate-manifest experiments/results/monolithic_stack_candidate_pr106x_lgblock16_20260508_codex/manifest_from_replacement_manifest.json --runtime-log-out experiments/results/monolithic_stack_candidate_pr106x_lgblock16_20260508_codex/runtime_consumption_pr106x_lgblock16.log --json-out experiments/results/monolithic_stack_candidate_pr106x_lgblock16_20260508_codex/runtime_consumption_proof_pr106x_lgblock16.json" --json-out experiments/results/monolithic_stack_candidate_pr106x_lgblock16_20260508_codex/runtime_consumption_proof_pr106x_lgblock16.json --fail-if-not-ready
.venv/bin/python tools/build_monolithic_runtime_consumption_proof.py --candidate-manifest experiments/results/monolithic_stack_candidate_pr106x_lgblock16_20260508_codex/manifest_from_replacement_manifest.json --command-text ".venv/bin/python tools/prove_monolithic_runtime_consumption.py --candidate-manifest experiments/results/monolithic_stack_candidate_pr106x_lgblock16_20260508_codex/manifest_from_replacement_manifest.json --runtime-log-out experiments/results/monolithic_stack_candidate_pr106x_lgblock16_20260508_codex/runtime_consumption_pr106x_lgblock16.log --json-out experiments/results/monolithic_stack_candidate_pr106x_lgblock16_20260508_codex/runtime_consumption_proof_pr106x_lgblock16.json" --runtime-log experiments/results/monolithic_stack_candidate_pr106x_lgblock16_20260508_codex/runtime_consumption_pr106x_lgblock16.log --json-out experiments/results/monolithic_stack_candidate_pr106x_lgblock16_20260508_codex/runtime_consumption_proof_from_log_pr106x_lgblock16.json --fail-if-not-ready
.venv/bin/python tools/check_monolithic_packet_closure_gate.py --candidate-manifest experiments/results/monolithic_stack_candidate_pr106x_lgblock16_20260508_codex/manifest_from_replacement_manifest.json --runtime-proof-json experiments/results/monolithic_stack_candidate_pr106x_lgblock16_20260508_codex/runtime_consumption_proof_from_log_pr106x_lgblock16.json --json-out experiments/results/monolithic_stack_candidate_pr106x_lgblock16_20260508_codex/closure_gate_with_runtime_proof_no_claim_pr106x_lgblock16.json
.venv/bin/python tools/check_monolithic_packet_closure_gate.py --candidate-manifest experiments/results/monolithic_stack_candidate_pr106x_lgblock16_20260508_codex/manifest_from_replacement_manifest.json --runtime-proof-json experiments/results/monolithic_stack_candidate_pr106x_lgblock16_20260508_codex/runtime_consumption_proof_from_log_pr106x_lgblock16.json --dry-run --json-out experiments/results/monolithic_stack_candidate_pr106x_lgblock16_20260508_codex/closure_gate_with_runtime_proof_dry_run_pr106x_lgblock16.json
.venv/bin/python tools/run_monolithic_candidate_preflight.py --candidate-manifest experiments/results/monolithic_stack_candidate_pr106x_lgblock16_20260508_codex/manifest_from_replacement_manifest.json --runtime-proof-json experiments/results/monolithic_stack_candidate_pr106x_lgblock16_20260508_codex/runtime_consumption_proof_from_log_pr106x_lgblock16.json --json-out experiments/results/monolithic_stack_candidate_pr106x_lgblock16_20260508_codex/preflight_with_runtime_proof_no_claim_pr106x_lgblock16.json
.venv/bin/python tools/run_monolithic_candidate_preflight.py --candidate-manifest experiments/results/monolithic_stack_candidate_pr106x_lgblock16_20260508_codex/manifest_from_replacement_manifest.json --runtime-proof-json experiments/results/monolithic_stack_candidate_pr106x_lgblock16_20260508_codex/runtime_consumption_proof_from_log_pr106x_lgblock16.json --dry-run --json-out experiments/results/monolithic_stack_candidate_pr106x_lgblock16_20260508_codex/preflight_with_runtime_proof_dry_run_pr106x_lgblock16.json
```

## Verification

- `.venv/bin/python -m py_compile tools/prove_monolithic_runtime_consumption.py`
  - passed
- `.venv/bin/python -m pytest -q src/tac/tests/test_prove_monolithic_runtime_consumption.py`
  - `3 passed`
- `.venv/bin/python -m ruff check tools/prove_monolithic_runtime_consumption.py src/tac/tests/test_prove_monolithic_runtime_consumption.py`
  - `All checks passed`
- `.venv/bin/python -m pytest -q src/tac/tests/test_build_monolithic_runtime_consumption_proof.py src/tac/tests/test_monolithic_packet_closure_gate.py src/tac/tests/test_run_monolithic_candidate_preflight.py src/tac/tests/test_prove_monolithic_runtime_consumption.py`
  - `18 passed`
