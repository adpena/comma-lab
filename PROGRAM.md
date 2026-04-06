# program

Mission: minimize the official challenge score on a pinned upstream snapshot while keeping a patched-world submission viable at all times.

## Primary objectives

1. Keep `submissions/exact_current` runnable under the current published workflow.
2. Keep `submissions/robust_current` improving under a more rule-faithful interpretation.
3. Collect clean evidence for the best-write-up track from day one.

## Mutation frontier

The agent may edit only:

- `configs/**`
- `docs/**`
- `prompts/**`
- `src/comma_lab/**`
- `submissions/robust_current/**`
- `runtime-rs/**`
- `cuda/**`
- `mojo/**`
- `jax/**`
- `.omx/**`
- `.ralph/**`
- `.agents/**`
- `reports/**`
- `experiments/**`

The agent may **not** edit without explicit human approval:

- the pinned upstream snapshot
- `scripts/bootstrap.sh`
- `scripts/start_lab.sh`
- `start.sh`
- `submissions/exact_current/inflate.py`
- `submissions/exact_current/inflate.sh`
- `LICENSE`
- `THIRD_PARTY_NOTICES.md`

## Evidence rules

- Never claim an improvement without a measured score.
- Prefer the official evaluator over proxies.
- Use proxy evaluation only to rank cheap candidates before promotion.
- Record config, command, artifact size, and score breakdown for each promoted run.

## Packaging rules

Always maintain two packaging views:

- `current_workflow`: reflects the published GitHub Action behavior.
- `rule_faithful`: reflects what a stricter interpretation would count.

Never confuse the two in reports.

## Operating loop

At each cycle:

1. propose at most 3 experiments
2. estimate expected payoff and cost
3. run smoke checks
4. run proxy evals
5. promote only the best candidate(s) to full eval
6. summarize what changed and what the evidence says
7. update the next experiment queue

## Track gates

- If `exact_current` stops producing near-zero or near-zero-like distortion with tiny rate, demote it immediately to a research note and move effort to `robust_current`.
- If `robust_current` does not beat the published baseline floor after a focused x265 search, shrink the search space before adding complexity.
- If a learned postfilter or decoder cannot justify its bytes or runtime, cut it.

## Reporting standards

Every promoted run should record:

- upstream snapshot hash
- submission track
- packaging mode
- archive size
- measured score
- segnet distortion
- posenet distortion
- rate
- runtime notes
- exact commands or config diff

## Style

Be direct.
Prefer small edits over sprawling rewrites.
Prefer reversible experiments.
Prefer measured evidence over narratives.
