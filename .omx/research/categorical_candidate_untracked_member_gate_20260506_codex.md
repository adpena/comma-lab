# Categorical Candidate Untracked Member Gate - 2026-05-06

## Context

Categorical/openpilot-label candidates are only useful as stackable contest
artifacts when every archive byte is named, charged, and consumed by the
declared runtime contract. Unexpected sidecars can hide debug state, uncharged
conditioning tables, or no-op evidence ambiguity.

## Finding

`audit_categorical_candidate_manifest()` compared archive members against the
declared charged-member manifest but treated untracked archive members as a
warning. That allowed `ready_for_exact_eval_dispatch=true` for a ZIP with extra
members that were not represented in the charged-member contract.

Evidence grade: `empirical` custody hardening, not score evidence.

## Change

- Add `candidate_archive_untracked_members` as a fail-closed dispatch blocker.
- Preserve the exact sorted list at
  `candidate_archive.untracked_members` in readiness JSON.
- Add a regression test with an undeclared `debug_sidecar.json` member.

## Verification

Focused:

```text
.venv/bin/python -m pytest src/tac/tests/test_categorical_candidate_readiness.py -q
```

Full preflight:

```text
.venv/bin/python tools/all_lanes_preflight.py --timings
```

## Promotion Status

This does not dispatch GPU work and does not claim score. It prevents
categorical archive candidates from reaching exact-eval readiness until the
archive member set is byte-closed and contract-complete.
