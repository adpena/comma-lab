# L5 v2 Parallel Audit Findings

Date: 2026-05-17
Author: codex
Authority: read-only subagent synthesis; `score_claim=false`;
`promotion_eligible=false`; `ready_for_exact_eval_dispatch=false`;
`dispatch_attempted=false`.

## Context

Five parallel read-only audits returned while preparing the `main` push. None
edited files. This memo preserves their L5/L5 v2 findings so the signal is
tracked in git rather than stranded in chat.

## Findings To Act On

1. **Harvest-cell bridge needs a single canonical owner.**
   One audit reported no existing end-to-end post-harvest converter from the
   Lightning paired-axis plan to `tools/build_l5_v2_sideinfo_effect_curve.py`
   cell JSON. A second audit reported a tracked adapter surface plus possible
   successor WIP names. Before implementation, inspect the current repo and
   choose one owner path. Required invariant: the bridge must consume the
   Lightning plan, prefer `contest_auth_eval.adjudicated.json`, normalize via
   `exact_eval_evidence_from_auth_eval_artifact`, and emit exactly the 5x2
   `variant x axis` cells required by the side-info effect-curve validator.

2. **Dykstra feasibility must gate promotion-adjacent L5 v2 side-info claims.**
   Dry-run parsing may remain allowed without
   `.omx/state/dykstra_feasibility_time_traveler_l5.json`, but side-info proof,
   paired-anchor readiness, timing-smoke authority, or promotion surfaces must
   record `dykstra_feasibility_present=true` and validate the artifact schema.

3. **Source-custody drift coverage is too narrow for post-plan tools.**
   The TT5L architecture-lock source set should include the execution bundle,
   dry-run verifier, harvest-cell bridge, and related `src/tac/optimization/*`
   modules. Otherwise future drift in omitted post-plan code can be invisible
   while `source_custody_current_for_execution` still looks true.

4. **Readiness booleans need sharper names and stricter blockers.**
   Current dry-run/bundle readiness can be true while global blockers such as
   missing Dykstra evidence remain. Preserve parse readiness, but do not let it
   imply provider dispatch, exact-eval readiness, side-info proof, rank/kill, or
   promotion authority.

5. **Dry-run verifier must validate custody, not only launcher parse success.**
   Required checks: local archive bytes/SHA match plan, preflight, and variant
   manifest; runtime files and runtime manifest hashes match; all ten cells are
   present with no duplicates or extras; CPU/CUDA axes preserve distinct
   commands/devices; stdout JSON and state JSON agree; `source_spec_command_sha256`
   matches the paired-axis plan command SHA.

6. **Axis and identity mismatches must fail before source-metadata fallback.**
   Reject harvested artifacts whose raw `score_axis`, command, eval device,
   inflate device, hardware, `pair_group_id`, or `run_id` conflict with the
   plan. Do not mask raw mismatches by injecting plan metadata during merge.

7. **Temporary test artifacts should avoid real result-tree pollution.**
   Tests that currently write under
   `experiments/results/time_traveler_l5_v2/test_sideinfo_lightning_paired_axis_*`
   should move to `tmp_path` or clean through a safer fixture so interrupted
   tests cannot leave ignored residue that looks like real frontier state.

## Next Concrete Patch Queue

- Add or harden the canonical harvest-cell bridge with tests for missing,
  complete, mismatched archive SHA/bytes, axis mismatch, raw `pair_group_id` /
  `run_id` mismatch, adjudicated-preferred behavior, duplicate/extra cells, and
  downstream effect-curve consumption.
- Expand TT5L source-custody static paths to include post-plan execution,
  dry-run, and harvest code.
- Split dry-run parse readiness from provider/promotion readiness in JSON/MD
  reports and tests.
- Add Dykstra-gated assertions for any side-info proof or promotion-adjacent
  readiness surface.

No provider dispatch was attempted. No score claim is made.
