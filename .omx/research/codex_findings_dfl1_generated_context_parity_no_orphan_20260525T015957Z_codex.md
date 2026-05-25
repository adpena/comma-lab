# Codex findings: DFL1 generated-context parity no-orphan bridge

Timestamp: 2026-05-25T01:59:57Z
Lane: `dfl1_generated_context_parity_no_orphan`
Authority axis: local scheduler / materializer custody only; no score claim, no
promotion eligibility, no rank/kill authority.

## Finding

The DFL1 shell-parity DAG existed, but generated final-byte-operation contexts
could still orphan the parity signal. Artifact maps produced or consumed by the
automated materializer path carried archive/member/output data, while DFL1
runtime, candidate runtime, full-frame file-list, and parity-output hints were
not guaranteed to reach `build_materializer_execution_queue(...)`. That could
leave generated DFL1 materializer candidates with materialization and harvest
steps but no automatic full-frame shell-parity proof step.

An adversarial audit also caught a fake-green class: a caller-provided
`--full-frame-file-list-claim` could previously make a partial one-entry file
list look strict if the outputs matched. Strict DFL1 parity now requires
expected full-frame file-list identity, not just a caller assertion.

## Landing

- `src/comma_lab/scheduler/final_byte_operation_contexts.py` now preserves DFL1
  source runtime, inflate runtime, candidate runtime, full-frame file-list,
  file-list entries, expected full-frame file-list SHA-256, expected full-frame
  entry count, file-list source, and parity-output-dir aliases when compiling
  artifact-map rows into materializer contexts. Missing DFL1 parity identity is
  now a generated-context blocker instead of a silent downstream skip.
- `src/comma_lab/scheduler/byte_shaving_campaign_queue.py` now carries the
  expected file-list identity into the DFL1 parity step, asserts the identity in
  postconditions, and passes the parity proof directory as an allowed artifact
  root to harvest when scheduler workload roots live outside the repo.
- `tools/prove_shell_inflate_parity.py` now refuses a full-frame parity claim
  unless the measured file list matches the expected full-frame SHA-256 and
  entry count and records the file-list source in the proof.
- `src/comma_lab/scheduler/materializer_chain_harvest.py` and
  `tools/harvest_materializer_chain_candidates.py` now allow explicit
  non-repo artifact roots for sidecar parity proofs while still rejecting
  unapproved external or symlinked proofs.
- `tools/build_byte_shaving_campaign_queue.py` now surfaces DFL1 parity
  requested/enabled/blocked counts and blocker summaries in CLI stdout.
- `tools/run_byte_shaving_materializer_campaign.py` now exposes DFL1 parity
  artifact-map flags and rejects those flags unless the packet-member target is
  `renderer_payload_dfl1`, preventing cross-family context pollution. Invalid
  or missing full-frame identity remains fail-closed in context compilation and
  queue follow-up blockers.
- `src/tac/tests/test_final_byte_operation_contexts.py` asserts the compiler
  carries those hints into the work-row DFL1 parity context.
- `src/tac/tests/test_byte_shaving_campaign_queue.py` adds a CLI-level generated
  context regression proving artifact-map -> generated contexts -> work queue ->
  execution queue emits the DFL1 parity step and harvest sidecar proof wiring.
- `src/tac/tests/test_byte_shaving_materializer_campaign_runner.py` adds runner
  regressions for generated DFL1 parity artifact maps, follow-up DAG emission,
  and fail-closed misuse on non-DFL1 packet targets.
- `src/tac/tests/test_materializer_chain_harvest_scheduler.py` and
  `src/tac/tests/test_optimizer_exact_readiness.py` cover the stronger
  full-frame file-list identity checks and external-root sidecar proof custody.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  src/tac/tests/test_family_agnostic_materializers.py \
  src/tac/tests/test_materializer_chain_harvest_scheduler.py \
  src/tac/tests/test_optimizer_exact_readiness.py \
  src/tac/tests/test_final_byte_operation_contexts.py \
  src/tac/tests/test_byte_shaving_campaign_queue.py \
  src/tac/tests/test_byte_shaving_materializer_campaign_runner.py -q
```

Result: 267 passed in 11.44s.

```bash
ruff check tools/prove_shell_inflate_parity.py \
  tools/harvest_materializer_chain_candidates.py \
  tools/build_byte_shaving_campaign_queue.py \
  tools/run_byte_shaving_materializer_campaign.py \
  src/comma_lab/scheduler/byte_shaving_campaign_queue.py \
  src/comma_lab/scheduler/final_byte_operation_contexts.py \
  src/comma_lab/scheduler/materializer_chain_harvest.py \
  src/tac/optimization/family_agnostic_materializers.py \
  src/tac/tests/test_byte_shaving_materializer_campaign_runner.py \
  src/tac/tests/test_final_byte_operation_contexts.py \
  src/tac/tests/test_byte_shaving_campaign_queue.py \
  src/tac/tests/test_materializer_chain_harvest_scheduler.py \
  src/tac/tests/test_optimizer_exact_readiness.py
```

Result: all checks passed.

```bash
git diff --check
```

Result: clean.

## Remaining boundary

This landing makes the generated local materializer DAG preserve DFL1 shell
parity proof signal and prevents partial-list proof self-attestation from
becoming strict parity. It does not claim a contest score and does not authorize
promotion. Exact CPU/CUDA auth-axis evaluation still owns score and promotion
authority after a byte-closed candidate is produced and strict parity proof
passes.
