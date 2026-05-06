# Geometry Feedback Runtime Consumer Contract

Date: 2026-05-06
Author: codex
Evidence grade: pre-dispatch readiness hardening
Score claim: false
Dispatch attempted: false

## Context

LA-Pose-inspired atoms, telescopic foveation fields, RAFT flow, and openpilot
pose priors are useful planning signals, but they are not dispatch evidence by
themselves. They can only affect a scored archive through deterministic charged
bytes that the inflate runtime actually consumes.

## Contract

`src/tac/geometry_feedback_readiness.py` defines
`charged_geometry_feedback_runtime_consumer_v1`. The contract records:

- `score_claim=false`
- `dispatch_attempted=false`
- `ready_for_exact_eval_dispatch=false`
- charged artifact proofs, if any
- runtime consumer proofs, if any
- fail-closed blockers for uncharged feedback, missing candidate archive
  manifests, missing geometry component gates, and missing exact CUDA auth eval

The cross-paradigm frontier inventory attaches this contract to:

- `lapose_motion_atom_allocator`
- `telescopic_foveation_field`
- `raft_radial_openpilot_pose`

## Boundary

This artifact does not claim a score and does not authorize remote/GPU
dispatch. It prevents geometry feedback from being treated as dispatchable
until a candidate-specific archive manifest proves charged runtime consumption
and the normal exact CUDA gates apply.
