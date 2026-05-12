# Preflight SourceIndex Casefold Cold-Path Greenup (2026-05-12)

## Scope

Fixed the hosted-runner preflight timeout class without weakening the 30s DX
fuse or dropping any checks.

## Finding

`SourceIndex.text_facts()` computed `text.casefold()` and scanned the full
default needle set twice for every source file. Only a small subset of checks
needs case-insensitive matching, so cold-cache preflight paid unnecessary
string-copy and substring-search cost before any semantic guard could report.

The source-index cache schema is bumped to `pact.source_text_facts.v24` so old
wide-casefold rows do not silently carry forward.

## Patch

- Added `_DEFAULT_CASEFOLD_TEXT_FACT_NEEDLES` as a tiny explicit set for
  case-insensitive scanner prefilters.
- Limited casefold computation to that set and cached folded needle pairs.
- Replaced `len(text.splitlines())` with `_source_line_count(text)` to avoid a
  full line-list allocation on every file.
- Added process-local `Path.resolve()` memoization for SourceIndex keying after
  true-cold profiling showed repeated path resolution dominated hosted-runner
  scanner cost.
- Disabled persistent text-facts read/write by default on GitHub Actions,
  because hosted CI starts from a cold checkout and discards `.omx/cache`; the
  persistent serializer is useful for local dev loops but pure first-run
  overhead in CI.
- Kept exact-match `_DEFAULT_TEXT_FACT_NEEDLES` unchanged, so strict scanner
  coverage is preserved.

## Evidence

- `PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest src/tac/tests/test_source_index.py -q`
- `PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest src/tac/substrates/*/tests/test_score_aware_loss_real_scorer_forward.py -q`
- `PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest src/tac/tests/test_preflight_proactive_custody_concurrency_audit.py src/tac/tests/test_preflight_custody_validator_and_locked_writes.py -q`
- `env GITHUB_ACTIONS=true PACT_PREFLIGHT_DISABLE_INCREMENTAL_CACHE=1 PACT_PREFLIGHT_PARALLEL_WORKERS=8 PYTHONPATH=src:upstream:$PWD /usr/bin/time -lp .venv/bin/python -m tac.preflight`

Result: strict developer preflight passed locally in 8.83s cold GHA-mode with
all checks enabled.
