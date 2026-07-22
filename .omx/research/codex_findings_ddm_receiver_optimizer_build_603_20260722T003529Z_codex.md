---
title: Codex round-1 findings — DDM receiver and optimizer build
utc: 2026-07-22T00:35:29Z
task: 603
verdict: CLEAN_AFTER_FIX_FOR_CUSTODY_SCOPE
verdict_scope: local deterministic n64 receiver, byte custody, and search continuation only
research_only: true
execution_allowed: false
---

# Round-1 outcome

The implementation is clean after three round-1 corrections. This is one reviewed custody landing,
not a three-clean-pass seal and not PRIMARY launch readiness.

## Findings fixed

1. `HIGH`: the first multi-stream coordinate enumerator flattened streams then truncated, so a stage
   naming four streams searched only its first stream. It now round-robins coordinates and the receipt
   records positive coverage for every declared stream.
2. `HIGH`: the draft blocker register called canonical resume-registry/cadence wiring green when only
   the local runner's immutable stage checkpoints and disk continuation were proven. That blocker is
   red again; local resume remains measured separately.
3. `MEDIUM`: the first n64 coverage receipt listed class/stratum names without binding them to pairs.
   It now persists 64 explicit pair assignments and verifies all 11 applicable class/stratum
   combinations.

Fix review: focused tests, Ruff, Python compilation, byte-diff checks, fresh typed CLI smoke, and
receipt re-harvest are green. Checkpoint tampering and config-identity mismatch both fail closed.

## Remaining attack surface

- The 8x8 RGB uint8 field is deliberately a custody fixture, not a contest-shape receiver or scorer
  preimage. Full-shape n600 receiver/evaluator closure remains red.
- `DirectDescriptionOptimizerCheckpointV1` is the real local runner schema; the older S4-bound
  `DirectDescriptionStageCheckpointV1` remains a legacy static audit. MAIN should review the versioned
  split and reject any claim that V1 legacy checkpoints became runnable.
- Exact ZIP byte homes now exist, but task-value per byte is unknown. No Fisher/KKT, d_seg, d_pose, or
  score conclusion transfers.
- Canonical `WitnessProgram`/`TypedWitnessConfig`, canonical resume registry, governed launcher/memory
  adapter, cleanup/cold-store path for a heavy n600 run, and external attestation remain absent.

## STORES CONSULTED

- Delegated authority, `CLAUDE.md`, `AGENTS.md`, project memory top, `PROGRAM.md`, `HANDOFF.md`,
  `SYSTEM_MAP.md`, and [the operating manual](../../docs/operating_manual_craft_handoff.md).
- v7.5 §8 operating contract, v8 spec, DDM PRIMARY spec, predecessor builder/DAG/equation artifacts,
  S4 composer/parser/receiver source, owner bundle, current pointer/lane/subagent/task state, and both
  live inboxes.
- The fresh custody archive, receipt, all nine preserved checkpoint files, focused tests, and the full
  base-to-working-tree diff.

Pointer `0.1910828242 [contest-CPU]` unchanged. MAIN landing review required.
