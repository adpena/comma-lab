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
