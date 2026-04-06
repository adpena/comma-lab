$deep-interview "Read AGENTS.md, PROGRAM.md, docs/challenge_snapshot.md, docs/current_workflow_vs_rule_faithful.md, docs/omx_ralph_runbook.md, and reports/latest.md if it exists. Clarify the project state and the highest-leverage next step from the files alone. Do not ask the human follow-up questions unless a missing credential or missing binary is a hard blocker."

$ralplan "Approve a small, reversible plan that gets this repo from cold start to the first measured progress while keeping both tracks healthy. The plan must: (1) verify the upstream snapshot, (2) confirm exact_current is still wired, (3) confirm robust_current packaging still works, (4) queue the next 3 experiments with expected payoff, cost, and risk, and (5) record all evidence to disk."

$ralph "Execute the approved plan to completion. Stay inside the mutation frontier in AGENTS.md. Prefer measured progress over elegance. Leave the repo resumable for the next fresh loop. Keep both packaging views explicit: current_workflow and rule_faithful. If a speculative lane looks attractive, record it under docs/speculative_lanes.md and only promote it when evidence justifies it."

Extra operating constraints for this repo:

- The scorer is the authority.
- Keep two submission tracks alive at all times: `exact_current` and `robust_current`.
- Do not edit the pinned upstream snapshot or the exact-current inflator.
- Prefer direct, small changes over sprawling rewrites.
- Update `.omx/state/current_focus.md`, `.omx/state/next_experiments.md`, `.omx/research/findings.md`, `.ralph/run_log.md`, and `reports/latest.md` before stopping.
- If the current-workflow exploit path appears broken, demote it quickly and shift energy to Track B.
- If a heavy model, JAX lane, Mojo lane, or CUDA lane does not earn its bytes or runtime, cut it.
- Use teams only when parallelism is obviously worth it. Default to a single-owner Ralph loop.
