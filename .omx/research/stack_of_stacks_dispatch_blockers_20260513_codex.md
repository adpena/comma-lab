# Stack-of-Stacks Dispatch Blockers - Codex Review - 2026-05-13

Lane: `lane_stack_of_stacks_composition_implementation_20260513`
Context: local commit `970602cd` landed the reusable stack-of-stacks composer
module and tests. Additional local files exist for a Modal dispatch recipe and
trainer, but they are not promoted in this review.

## Verdict

`research_only=true` until the dispatch wrapper and emitted runtime close.

## Blocking Findings

1. Missing remote driver:
   `.omx/operator_authorize_recipes/substrate_stack_of_stacks_modal_a100_dispatch.yaml`
   references `scripts/remote_lane_substrate_stack_of_stacks.sh`, but that file
   is absent in the worktree.

2. The trainer's emitted `inflate.py` template currently exits successfully
   without rendering frames when per-arm decoder hooks are absent. That is
   acceptable as an internal scaffold only if it fails closed before auth eval;
   it is not a contest-evaluable runtime.

3. The recipe predicts a score band and a smoke-before-full path, but the
   current runtime cannot produce a byte-closed exact-eval packet. Dispatching
   it would produce low-signal infrastructure failure rather than score
   evidence.

## Required Promotion Work

- Add `scripts/remote_lane_substrate_stack_of_stacks.sh` using the canonical
  source-only bootstrap pattern and active lane-claim verification.
- Change the emitted runtime to either render frames through a verified arm-0
  decoder path or fail nonzero with `research_only_runtime_hooks_missing`.
- Add a focused test proving the trainer refuses exact-eval readiness while
  per-arm decoder hooks are missing.
- Only then wire the recipe into operator-authorize as a dispatchable path.

No score claim, no promotion eligibility, no exact-eval readiness.
