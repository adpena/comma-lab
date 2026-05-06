# Joint Stack No-Op Manifest

Date: 2026-05-06
Author: codex
Evidence grade: planning/component empirical plus typed fixture manifest
Score claim: false
Dispatch attempted: false

## Context

PARADIGM-gamma had ADMM, Ballé hyperprior, and AQ/static arithmetic pieces
landed individually, but that did not prove they compose as a contest archive
stack. The current tranche adds a deterministic fixture-only manifest for the
canonical order:

`representation -> prediction -> quantization -> hyperprior -> arithmetic -> pack`

## Artifact

- Manifest:
  `experiments/results/joint_stack_noop_manifest_20260506_codex/manifest.json`
- Builder:
  `tools/build_joint_stack_noop_manifest.py`
- Contract implementation:
  `src/tac/stack_compositions.py::build_joint_admm_balle_arithmetic_noop_manifest`

## Non-Promotion Boundaries

This is intentionally not dispatchable and not a score artifact. The manifest
records:

- `score_claim=false`
- `dispatch_attempted=false`
- `remote_jobs_dispatched=false`
- `ready_for_exact_eval_dispatch=false`
- `fixture_only=true`
- `candidate_non_noop=false`

The blockers are:

- fixture-only candidate, not dispatchable
- no byte-closed JCSP archive member
- no runtime loader parity for a JCSP archive member
- no exact CUDA auth eval for a stacked archive
- no lane dispatch claim
- component empirical results do not prove stack composability

## Next Gate

The next promotable patch is to replace the fixture-only contract with a real
byte-closed JCSP archive member, prove runtime loader parity in the canonical
inflate path, claim the lane, and then run exact CUDA auth eval on the exact
archive bytes.
