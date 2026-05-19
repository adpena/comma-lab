# Codex Findings - Full Preflight Timing Profile

**UTC:** 2026-05-19T18:30:00Z
**Scope:** `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m tac.preflight --scope all --allow-slow-preflight --timings-json /tmp/pact_preflight_all_timing_20260519.json`
**Score claim:** none

## Finding

The all-scope preflight did not die silently. It emitted a timing JSON profile,
failed normally with `PreflightError`, and reached the pre-existing
`check_substrate_at_optimal_form_before_paid_dispatch` blocker.

Observed timing payload:

- `status`: failed
- `wall_elapsed_s`: 35.562608
- `scope`: all
- `error_type`: PreflightError
- `step_rows`: 89

Top measured costs:

1. `preflight_filename_contract`: 12.242261s
2. `check_no_proxy_metric_drives_decision`: 10.320776s
3. `check_no_compromised_lightning_supply_chain`: 10.025872s
4. `preflight_dead_resolvers`: 9.166034s
5. `check_codebase_drift`: 6.836078s
6. `preflight_loader_format_safety`: 6.710071s
7. `check_silent_default_audit_clean`: 5.485057s
8. `check_callsite_contracts_satisfied`: 5.056899s
9. `check_no_misleading_device_named_output_directories`: 4.631206s
10. `check_uv_torch_install_has_driver_version_pin`: 1.953849s

## Interpretation

This is a real performance surface. The current release-scope sweep is
correctness-first but broad-scan-heavy. The slowest checks are mostly repeated
repo-wide file enumeration and text scanning. The next optimization should
preserve fail-closed semantics while sharing a single immutable file snapshot
or scan cache across compatible preflight checks.

## Recommended Follow-Up

Build a preflight scan-context layer that memoizes:

- `rg --files` / tracked file lists by scope and ignore policy;
- file text loads keyed by path + mtime + size;
- parsed Python ASTs for checks that currently rescan source text;
- `.omx/state/lane_registry.json` and council-anchor query results for
  lane/council gates.

Independent read-only perf review refined the first patch targets:

- add per-file result caches for `preflight_dead_resolvers` and
  `check_no_proxy_metric_drives_decision`, including failing-result lists, keyed
  by path + size + `mtime_ns` + scanner version;
- add lossless lexical prefilters before AST work in
  `check_no_misleading_device_named_output_directories`;
- cache `preflight_filename_contract` extraction outputs per file instead of
  invalidating a whole-corpus source cache when one file changes;
- replace repeated `ast.unparse` hot calls in loader-format safety with
  `ast.get_source_segment(text, node)` plus existing fallback.

Do this as a separate landing from Catalog #287-v2. The Cluster C correctness
landing should remain small enough to review and revert independently.

## Optimization Follow-Up — 2026-05-19T18:55Z

First tranche landed the two lowest-risk scanner reductions:

- `preflight_loader_format_safety`: replaced repeated `ast.unparse` on hot
  call/function nodes with direct source-segment extraction, then narrowed the
  no-SourceIndex fallback to `rg -l 'torch\.load|torch\.frombuffer'` with the
  same `experiments/results/**` exclusion used by `_iter_python_files`.
- `check_no_misleading_device_named_output_directories`: added a lossless
  lexical prefilter before AST parse and routed the file enumerator through
  `SourceIndex.files_containing_substrings` when a shared SourceIndex exists.

Adversarial subagent review caught one important correctness trap before
commit: Python AST columns are UTF-8 byte offsets, not Python string code-point
offsets. The final implementation slices encoded source bytes, decodes the
segment, and falls back to `ast.unparse` only on malformed offsets. Regression:
`test_preflight_loader_source_segment_handles_non_ascii_before_call`.

Re-profile command:

`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m tac.preflight --scope all --allow-slow-preflight --timings-json /tmp/pact_preflight_all_timing_after3_20260519.json`

Observed after profile:

- `status`: failed normally with the same
  `check_substrate_at_optimal_form_before_paid_dispatch` 17-lane blocker
- `wall_elapsed_s`: 30.201587
- `hot_step_count`: 20
- `sum(hot_step elapsed_s)`: 59.961286

Measured hot-step deltas versus the 18:30Z profile:

- `check_no_misleading_device_named_output_directories`: 4.631206s -> 1.516838s
- `preflight_loader_format_safety`: 6.710071s -> 4.488789s
- `preflight_dead_resolvers`: 9.166034s -> 6.999458s (incidental cache warmth;
  not attributed to this patch)
- `check_no_proxy_metric_drives_decision`: 10.320776s -> 8.510246s (incidental
  cache warmth; not attributed to this patch)
- `preflight_filename_contract`: 12.242261s -> 10.041400s (still the top
  remaining owned hot path)

Next tranche should implement the subagent-recommended per-file result caches
for `preflight_dead_resolvers` and `check_no_proxy_metric_drives_decision`,
including cached failure replay and dependency-fingerprint invalidation for
dead imports. Do not land whole-check clean caching; the invalidation surface
is too coarse for this repo's churn rate.
