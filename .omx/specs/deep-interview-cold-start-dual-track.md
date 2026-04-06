# Execution-Ready Spec: cold-start-dual-track

## Metadata
- Profile: standard
- Rounds: 0 interactive rounds; repo-evidence crystallization only
- Final ambiguity: 0.14
- Threshold: 0.20
- Context type: brownfield
- Context snapshot: `.omx/context/cold-start-dual-track-20260403T195000Z.md`

## Intent
Establish a trustworthy cold-start operating baseline for the comma challenge lab without violating the mutation frontier, so future iterations can optimize from evidence instead of assumptions.

## Desired Outcome
- Upstream snapshot is verified against the live checkout.
- `exact_current` is confirmed alive or explicitly demoted with evidence.
- `robust_current` packaging is confirmed alive and at least one measured baseline is captured if feasible.
- The next 3 experiments are queued with payoff/cost/risk.
- Durable repo state is updated for seamless continuation.

## In Scope
- Read and synthesize repo guidance/status files.
- Verify upstream snapshot commit/digests.
- Smoke/evaluate Track A and Track B.
- Record evidence in `.omx/*`, `.ralph/*`, and `reports/latest.md`.
- Add/update a speculative lane note only if a speculative idea appears during execution.

## Out of Scope / Non-goals
- Editing the upstream snapshot.
- Editing `submissions/exact_current/inflate.py` or `inflate.sh`.
- Large refactors, new heavy models, or CUDA/JAX/Mojo lanes without measured justification.
- Claiming improvement absent measured score.

## Decision Boundaries
- OMX may decide exact_current demotion if live evidence shows the exploit path is broken.
- OMX may make small reversible edits inside the allowed mutation frontier to keep robust_current packaging/evaluation healthy.
- OMX should use bat00 only if CUDA work becomes clearly justified by evidence.

## Constraints
- Keep both packaging views explicit: `current_workflow` and `rule_faithful`.
- Prefer direct small changes and <=3 experiments.
- Update required durable state files before stopping.

## Testable Acceptance Criteria
1. Live upstream commit and pinned-file digests are checked and recorded.
2. `exact_current` is run or its failure mode is recorded with concrete output.
3. `robust_current` packaging succeeds or its failure mode is recorded with concrete output.
4. At least one measured result (smoke/package/eval evidence; ideally full score for a track) is written to disk.
5. `reports/latest.md`, `.omx/state/current_focus.md`, `.omx/state/next_experiments.md`, `.omx/research/findings.md`, and `.ralph/run_log.md` are updated.

## Assumptions exposed + resolutions
- Assumption: the stored snapshot still matches the live upstream checkout. Resolution: must be verified before trusting any workflow inference.
- Assumption: exact_current is still wired in the installed upstream checkout. Resolution: must be tested, not inferred from copied files.
- Assumption: robust_current packaging is a safe initial baseline lane. Resolution: verify packaging first, then decide whether to measure full eval or iterate on packaging.

## Pressure-pass finding
A tempting shortcut was to start x265 tuning immediately, but repo evidence says the highest-leverage risk is stale grounding. Therefore the first execution loop should spend its budget on snapshot verification + track wiring before any tuning.

## Brownfield evidence vs inference notes
- Evidence-backed: snapshot JSON, runbook/docs, installed submission trees, packaging scripts.
- Inference: the best first measured progress is likely a robust_current baseline if packaging succeeds and exact_current remains only a current_workflow exploit lane.
