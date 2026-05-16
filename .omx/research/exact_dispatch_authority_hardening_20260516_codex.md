# Exact Dispatch Authority Hardening

- date: `2026-05-16`
- agent: `codex`
- scope: paid exact-eval fan-out authority, `parallel_dispatch_top_k`, L5/autopilot safety surface
- score_claim: `false`
- promotion_eligible: `false`
- ready_for_exact_eval_dispatch: `false`

## Why

Multiple dispatch-adjacent surfaces can see a row with
`ready_for_exact_eval_dispatch=true`. That flag is useful, but it must not be
treated as sufficient authority for paid exact eval. Authority requires live
archive/runtime custody at the actuator boundary: archive bytes/SHA, strict ZIP,
submission runtime, `inflate.sh`, report, archive manifest, target metadata, and
the current runtime tree.

The recurring bug class is false authority by stale or incomplete ready rows.
The current fix makes the ready flag an input fact, not the final decision.

## Landed

Added `tac.optimizer.exact_dispatch_authority` with:

- `ExactDispatchAuthorityVerdict`
- `exact_dispatch_authority(...)`

The helper delegates to the existing exact-readiness custody checker and also
compares any declared `runtime_tree_sha256` against the runtime tree computed
from the live submission directory.

`tools/parallel_dispatch_top_k.py` now calls this helper inside
`_candidate_blockers()` before any paid fan-out. The existing exact-ready queue
audit remains in place; this adds a row-level authority check so a ready flag
without live custody is blocked before dispatch.

## Regression

`test_parallel_dispatch_ready_flag_still_requires_live_custody` builds a
candidate with:

- `ready_for_exact_eval_dispatch=true`
- contest target metadata
- valid archive bytes/SHA
- valid-looking runtime tree SHA

but omits report and archive manifest custody. The actuator now blocks it with:

- `exact_dispatch_authority:archive_manifest_missing`
- `exact_dispatch_authority:report_txt_missing`

## Follow-Up

Next exact-dispatch hardening should route Cathedral autonomous loop, direct
Modal CUDA/CPU actuators, and Lightning direct submitters through the same
helper so the rule is uniform across providers.

## Verification

- `.venv/bin/python -m pytest tests/test_parallel_dispatch_top_k_exact_ready_audit.py::test_parallel_dispatch_ready_flag_still_requires_live_custody -q` -> `1 passed`
- `.venv/bin/python -m pytest tests/test_parallel_dispatch_top_k_exact_ready_audit.py -q` -> `4 passed`
- `.venv/bin/python -m pytest tests/test_audit_exact_ready_queues_cli.py tests/test_parallel_dispatch_top_k_exact_ready_audit.py -q` -> `10 passed`
- `.venv/bin/python -m ruff check src/tac/optimizer/exact_dispatch_authority.py tools/parallel_dispatch_top_k.py tests/test_parallel_dispatch_top_k_exact_ready_audit.py` -> clean
- `.venv/bin/python -m py_compile src/tac/optimizer/exact_dispatch_authority.py tools/parallel_dispatch_top_k.py` -> clean
- `git diff --check` -> clean
