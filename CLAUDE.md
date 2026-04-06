# AGENTS

You are operating inside a dual-track lab for the comma video compression challenge.

Read `PROGRAM.md` before making changes.

## Primary duties

1. Keep `submissions/exact_current` runnable under the current published workflow.
2. Keep `submissions/robust_current` improving under a stricter, rule-faithful interpretation.
3. Leave durable state so a fresh agent iteration can resume work without relying on chat memory.

## Mutation frontier

You may edit only:

- `configs/**`
- `docs/**`
- `prompts/**`
- `src/comma_lab/**`
- `submissions/robust_current/**`
- `runtime-rs/**`
- `cuda/**`
- `jax/**`
- `mojo/**`
- `.omx/**`
- `.ralph/**`
- `.agents/**`
- `reports/**`
- `experiments/**`

You must not edit without explicit human approval:

- the pinned upstream snapshot
- `submissions/exact_current/inflate.py`
- `submissions/exact_current/inflate.sh`
- `start.sh`
- `LICENSE`
- `THIRD_PARTY_NOTICES.md`

## Operating rules

- Prefer at most 3 experiments per cycle.
- Prefer small, reversible changes.
- Never claim a win without a measured score.
- Do not confuse `current_workflow` accounting with `rule_faithful` accounting.
- Keep both tracks healthy even if one looks dominant.
- Use JAX, Mojo, CUDA, or Rust only when they clearly reduce wall-clock cost or artifact size.
- Treat speculative ideas as side lanes unless evidence forces promotion.

## Required durable state

After each serious cycle, update at least:

- `.omx/state/current_focus.md`
- `.omx/state/next_experiments.md`
- `.omx/research/findings.md`
- `.ralph/run_log.md`
- `reports/latest.md`

## Promotion rules

A candidate may be promoted only after:

1. packaging succeeds
2. inflation succeeds
3. shape/frame-count checks pass
4. proxy evaluation looks promising
5. full evaluation confirms the gain or records the failure

## Track-specific guidance

### Track A: `exact_current`

- Preserve transparency.
- Use it as a live test of the currently published workflow.
- If upstream changes invalidate the exploit assumptions, demote it immediately to a research note and keep the repo useful.

### Track B: `robust_current`

- Start with safer codec improvements and task-aware pre/post processing.
- Add sparse residuals before adding heavier learned components.
- Only promote a neural side-model if its bytes and runtime clearly justify themselves.

## Ralph-style execution model

Treat files and git as memory.
Each iteration should be resumable from disk.
Do not rely on long chat context for continuity.
