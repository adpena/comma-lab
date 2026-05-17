# Catalog 315 Optimal Form Before Paid Dispatch Gate

Date: 2026-05-17
Author: codex
Authority: preflight hardening; strict landing; `score_claim=false`;
`promotion_eligible=false`; `ready_for_exact_eval_dispatch=false`;
`ready_for_provider_dispatch=false`; `dispatch_attempted=false`.

## Context

During the L5 v2 push cycle, the local preflight hook surfaced a generated
Catalog #315 patch in `src/tac/preflight.py`. The gate matches the operator's
non-negotiable "do not test lifted-trainer form as if it falsifies the concept"
concern: paid dispatch should not proceed when a substrate has
`impl_complete=true` but the latest council anchor is still
`PROCEED_WITH_REVISIONS`.

## Gate Behavior

`check_substrate_at_optimal_form_before_paid_dispatch(strict=True)` scans
`.omx/state/lane_registry.json` and joins in-scope L1+ substrate lanes to
`.omx/state/council_deliberation_posterior.jsonl` via
`deferred_substrate_id`.

It raises in strict mode when all of these are true:

- lane is substrate-like and L1+
- `gates.impl_complete.status=true`
- latest joined council verdict is `PROCEED_WITH_REVISIONS`
- no later unconditional `PROCEED` anchor supersedes it
- no explicit opt-out applies

Accepted opt-outs are `research_only=true`, `lane_class=substrate_engineering`,
archived state, or a same-line waiver
`# OPTIMAL_FORM_DISPATCH_OK:<rationale>` with a real rationale. Placeholder
waivers such as `<rationale>` are rejected.

The gate is wired into `preflight_all()` as strict because the live count was
verified at zero in this checkout on 2026-05-17. The known
`PROCEED_WITH_REVISIONS` anchors are structurally opted out through
`research_only=true` / `lane_class=substrate_engineering`; future lifted-trainer
substrates without opt-out now fail closed before paid dispatch.

## Tests

Added `src/tac/tests/test_check_315_substrate_optimal_form_before_dispatch.py`
covering:

- unresolved `PROCEED_WITH_REVISIONS` emits a warning and raises in strict mode
- a later unconditional `PROCEED` anchor clears the warning
- research-only, substrate-engineering, and real waiver opt-outs pass
- placeholder waiver text does not self-waive

Verification:

```bash
.venv/bin/python -m pytest src/tac/tests/test_check_315_substrate_optimal_form_before_dispatch.py -q
.venv/bin/python -m py_compile src/tac/preflight.py src/tac/tests/test_check_315_substrate_optimal_form_before_dispatch.py
.venv/bin/python - <<'PY'
from tac.preflight import check_substrate_at_optimal_form_before_paid_dispatch
print(len(check_substrate_at_optimal_form_before_paid_dispatch(strict=False)))
PY
```

Initial local results before commit:

- `61 passed`
- combined L5 dry-run custody + execution-bundle + Catalog #315 sweep:
  `76 passed`
- direct gate call: `0`
- `py_compile` clean
- `git diff --check` clean
- first focused test pass caught and fixed a generated-gate waiver bug:
  `_check_315_collect_lane_text()` lowercases notes before waiver scanning, so
  `_CHECK_315_WAIVER_RE` must be case-insensitive or valid
  `# OPTIMAL_FORM_DISPATCH_OK:<rationale>` waivers never match.

## Evidence Discipline

No provider dispatch was attempted. This gate changes only dispatch hygiene and
does not classify any lane, score, or substrate as dead. A bad paid result still
requires post-result adversarial classification; Catalog #315 only prevents
prematurely spending on a substrate form that the council already marked as
requiring iteration.
