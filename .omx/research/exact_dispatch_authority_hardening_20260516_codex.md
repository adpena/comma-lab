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

## Cathedral Autonomous Loop Extension

Landed a second actuator-bound use of the shared authority helper in
`tools/cathedral_autopilot_autonomous_loop.py`.

Previously, the le-$5 Cathedral self-authorization path accepted a local
dispatch-authority checklist: lane id, contest target mode, dispatch readiness,
and syntactically valid archive/runtime hashes. That was better than a naked
`ready_for_exact_eval_dispatch=true` flag, but still too weak: a stale row with
valid-looking hashes and no live archive path, byte count, manifest, report, or
score-affecting-change proof could reach the self-authorization decision before
the lane-claim write.

The loop now carries exact custody fields on `CandidateRow`:

- `archive_path`
- `submission_dir`
- `archive_manifest_path`
- `candidate_archive_bytes`
- `deployment_target`
- `score_affecting_payload_changed`
- `charged_bits_changed`
- `score_affecting_runtime_changed`

`OperatorAuthorizedModeConfig.can_authorize()` now passes the candidate through
`exact_dispatch_authority(...)` before any autonomous dispatch can be tagged as
authorized. `make_dispatch_halt_event(...)` forwards the caller's claim ledger
path so the same live custody decision sees the same dispatch-claim surface
that the subsequent claim write will use. The authorization journal also records
the custody paths and byte count so future analysis can reconstruct exactly
which packet crossed the gate.

### Regression

The Cathedral tests now include a tiny byte-closed exact-ready fixture:

- strict `archive.zip` with member `0.bin`
- executable `inflate.sh`
- `report.txt`
- `archive_manifest.json`
- computed runtime tree SHA from the real contest auth-eval runtime manifest
- `score_affecting_payload_changed=true`
- `charged_bits_changed=true`

Hash-only rows now fail closed with
`exact_dispatch_authority:archive_path_missing`,
`exact_dispatch_authority:archive_bytes_missing_or_invalid`, and
`exact_dispatch_authority:score_affecting_change_proof_missing`. Positive
Cathedral self-authorization tests must use the byte-closed fixture.

### Residual Work

Direct Modal CUDA/CPU actuators and Lightning direct submitters still need the
same helper at their paid-dispatch boundary. That work should be next if a
future review finds a path that can launch exact eval from row metadata without
first proving live archive/runtime/report/manifest custody.

## Verification

- `.venv/bin/python -m pytest tests/test_parallel_dispatch_top_k_exact_ready_audit.py::test_parallel_dispatch_ready_flag_still_requires_live_custody -q` -> `1 passed`
- `.venv/bin/python -m pytest tests/test_parallel_dispatch_top_k_exact_ready_audit.py -q` -> `4 passed`
- `.venv/bin/python -m pytest tests/test_audit_exact_ready_queues_cli.py tests/test_parallel_dispatch_top_k_exact_ready_audit.py -q` -> `10 passed`
- `.venv/bin/python -m ruff check src/tac/optimizer/exact_dispatch_authority.py tools/parallel_dispatch_top_k.py tests/test_parallel_dispatch_top_k_exact_ready_audit.py` -> clean
- `.venv/bin/python -m py_compile src/tac/optimizer/exact_dispatch_authority.py tools/parallel_dispatch_top_k.py` -> clean
- `.venv/bin/python -m pytest src/tac/tests/test_cathedral_autopilot_autonomous_loop.py -q` -> `148 passed`
- `.venv/bin/python -m ruff check tools/cathedral_autopilot_autonomous_loop.py src/tac/tests/test_cathedral_autopilot_autonomous_loop.py` -> clean
- `git diff --check` -> clean
