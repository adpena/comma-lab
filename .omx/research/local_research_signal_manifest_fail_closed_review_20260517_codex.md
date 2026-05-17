# Local Research-Signal Manifest Fail-Closed Review - 2026-05-17

## Context

Active objective: drive score-lowering work through byte-closed, axis-labelled,
contest-compliant artifacts while preserving every proxy/advisory signal. The
current WIP adds the operator-facing `--target local-mps` / `--target local-cpu`
surface so cheap local research signals can be used for dev velocity without
polluting `[contest-CPU]` or `[contest-CUDA]` custody.

## Finding

The local MPS and local CPU dispatcher paths wrote their dispatcher row through
the canonical JSONL helpers, but manifest append failure was catch-and-warn:

- `tools/operator_authorize.py::_dispatch_local_mps`
- `tools/operator_authorize.py::_dispatch_local_cpu`

That was a no-signal-loss violation. A successful local subprocess could return
`rc=0` even if the canonical `.omx/state/*_signal_manifest.jsonl` row failed to
land. The result would be neither promotable nor safely preserved as advisory
evidence.

## Fix

- Manifest append failure now raises `SystemExit` with a fatal message.
- The message records the local subprocess rc and explicitly says it is
  refusing success to avoid signal loss.
- Catalog #317's required source tokens now include the fail-closed phrase so a
  future refactor cannot silently downgrade this back to best-effort warning.
- `CLAUDE.md` Catalog #317 text was corrected to include manifest-write
  fail-closed semantics and the actual focused test files.

## Verification

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_operator_authorize_local_signal_manifest.py -q
```

Result: `3 passed`.

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_dispatch_protocol_tool_scope.py \
  src/tac/tests/test_local_pre_deploy_check.py \
  src/tac/tests/test_operator_authorize_local_signal_manifest.py -q
```

Result after the pinned dispatch-kind assertion was updated for
`local_research_signal`: `38 passed`.

```bash
.venv/bin/python -m ruff check \
  tools/operator_authorize.py \
  src/tac/tests/test_operator_authorize_local_signal_manifest.py
```

Result: `All checks passed!`.

## Authority

- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `ready_for_provider_dispatch=false`
- `dispatch_attempted=false`

No provider dispatch was launched. No lane claim was opened. This is a
dispatcher-custody hardening patch, not a score or promotion claim.
