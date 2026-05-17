# Exact Dispatch Authority Active-Claim Policy

Date: 2026-05-17

Scope: shared exact-eval dispatch authority, Lightning exact-eval submit claim
matching, and L5/frontier provider-spend hardening.

This is a no-score-claim hardening artifact.

## Finding

The shared exact-dispatch authority helper had one claim behavior: before a
fan-out actuator fires, an already-active same-lane claim is a conflict. That is
correct for queue fan-out, but it is not the correct provider-submit predicate.
At provider-submit time, the non-dry-run exact-eval command must prove the
lane/job already has an active claim.

Keeping those two predicates outside the shared helper lets future actuators
recreate the same ambiguity:

- queue fan-out needs `preclaim_conflict_check`;
- provider submit needs `require_active_claim`.

The distinction matters for score lowering because paid exact-eval dispatch is
the boundary where L5/TT5L, C1, Z5, and stack candidates become empirical
anchors. A dispatch launched without a claim can orphan custody and make
returned positive or negative scores harder to trust.

## Change

- Added `claim_policy` to `tac.optimizer.exact_dispatch_authority`.
- Added public `active_dispatch_claim_present(...)` matching with optional
  platform and job-id filters.
- Added `ignore_active_claim_conflicts` to `readiness_blockers(...)` so the
  provider-submit policy can require an active claim without treating the same
  claim as a conflict.
- Wired `scripts/launch_lightning_batch_job.py` exact-eval/component submit
  claim validation through the shared matcher instead of its local table parser.
- Kept existing submit error behavior and break-glass override semantics.
- Added focused regression coverage in
  `src/tac/tests/test_exact_dispatch_authority.py`.

## Authority

This artifact does not authorize score, rank, promotion, paid dispatch, or exact
eval by itself:

- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `dispatch_attempted=false`

The helper only controls whether a candidate or submit command has enough
custody to be launched by an actuator that already has operator intent and a
byte-closed archive/runtime packet.

## Validation

Focused checks run:

```bash
.venv/bin/ruff check \
  src/tac/optimizer/exact_dispatch_authority.py \
  src/tac/optimizer/exact_readiness.py \
  src/tac/tests/test_exact_dispatch_authority.py \
  scripts/launch_lightning_batch_job.py

.venv/bin/python -m pytest \
  src/tac/tests/test_exact_dispatch_authority.py \
  src/tac/tests/test_lightning_batch_jobs.py::test_non_dry_run_studio_submit_requires_active_dispatch_claim \
  -q

git diff --check
```

## Next Score-Lowering Implication

For the L5 v2 / TT5L side-info effect-curve dispatch wave, use
`preclaim_conflict_check` while selecting/fanning out work units and
`require_active_claim` inside the provider submit path. This keeps the actuation
fast while preserving exact custody for every paired `[contest-CPU]` and
`[contest-CUDA]` result that feeds the side-info curve and architecture-lock
packet.
