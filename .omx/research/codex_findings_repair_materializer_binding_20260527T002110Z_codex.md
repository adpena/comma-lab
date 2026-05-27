# Codex Findings: Repair Materializer Binding

UTC: 2026-05-27T00:21:10Z

## Summary

Repair-budget waterfill queue actuation was coherent at the parent queue level,
but the live queue still bound repair materialization audits to the blocked
generic operation materializer queue. The targeted component chain materializer
queue already had receiver-closed materializer manifests, so those manifests
were invisible to repair-budget binding and execution reports.

This pass wires repair-budget binding to prefer the targeted component chain
materializer work/execution queues, while falling back to the generic operation
materializer queue only when no targeted queue exists. Manifest discovery now
accepts explicit `--output-manifest` outputs and manifest-named postconditions,
and rejects unrelated exact-readiness reports such as harvest, dispatch, and
bridge JSONs.

## Landed Changes

- `frontier_rate_attack_feedback.py`
  - Filters expected materializer manifest paths to real manifest artifacts.
  - Deduplicates manifest paths discovered from work and execution queues.
  - Reads step-level `--output-manifest` command outputs.

- `frontier_rate_attack_feedback_cycle.py`
  - Passes targeted-chain materializer work/execution queues into repair
    waterfill queue construction when available.

- `tools/build_frontier_rate_attack_feedback_refresh.py`
  - Mirrors the targeted-chain preference in the operator refresh CLI.

- `src/tac/tests/test_frontier_rate_attack_feedback.py`
  - Adds regression coverage for manifest discovery filtering.
  - Verifies refresh-produced repair queues bind to targeted-chain materializer
    queues instead of the generic operation materializer queue.
  - Fixes direct tool-module import setup for the cycle auxiliary-key test.

- Materializer hardening preserved from concurrent worktree changes:
  - Archive-section entropy recode receiver proof now accepts full-member
    brotli payload identity even when compressed section length changes.
  - Family-agnostic materializer CLI derives expected existing hashes during
    `--allow-overwrite` idempotent reruns.
  - Submission closure now writes a false-authority static custody/refusal
    report instead of throwing away byte-closed candidate archive evidence when
    runtime resolution is missing.

## Live Artifact Refresh

Regenerated:

- `.omx/research/frontier_mlx_repair_dynamics_paired_reference_20260526T155611Z/repair_budget_waterfill_queue.json`
- Three repair-budget materializer binding reports under
  `experiments/results/frontier_mlx_repair_dynamics_paired_reference_20260526T155611Z/frontier_repair_budget_waterfill/`
- Three repair-budget materialization execution reports under the same root.

Current honest state after refresh:

- `repair_budget_waterfill_queue.json` validates cleanly.
- Repair binding reports now point to
  `targeted_component_correction_chain_materializer_work_queue.json` and
  `targeted_component_correction_chain_materializer_execution_queue.json`.
- Each repair binding report discovers exactly two real targeted-chain
  materializer manifests.
- Each campaign remains `candidate_archive_materialized_count = 1 / 6`:
  the rate-only parent is receiver-consumed, while spent-budget repair children
  still need direct composed repair-candidate manifests plus component replay.
- No score claim, promotion eligibility, rank/kill eligibility, budget spend
  authority, or exact dispatch authority is granted.

## Verification

- `ruff check` on touched scheduler/tool/test/materializer files: passed.
- `pytest src/tac/tests/test_frontier_rate_attack_feedback.py -q`: 50 passed.
- `pytest src/tac/tests/test_repair_budget_materialization_execution.py -q`: 4 passed.
- `pytest src/tac/tests/test_family_agnostic_materializers.py -q`: 43 passed.
- `pytest src/tac/tests/test_materializer_submission_closure.py -q`: 5 passed.
- `pytest src/tac/tests/test_optimizer_exact_readiness.py -q -k "materializer or closure or runtime"`: 22 passed, 43 deselected.
- `tools/experiment_queue.py validate` for repair waterfill queue: valid.
- `tools/experiment_queue.py validate` for targeted-chain materializer execution queue: valid.
- `tools/lane_maturity.py validate`: 1424 lanes validated cleanly.

## Remaining Gaps

This is now wired and automated for local MLX/CPU advisory queue actuation, but
not theoretically complete. The next blocker is direct composed spent-budget
repair materialization: child rows must receive receiver-consumed manifests
that explicitly bind to repair candidate-chain ids and then replay exact-axis
SegNet/PoseNet component response. Until then, the repaired archive is not a
budget-spend authority and cannot claim score.
