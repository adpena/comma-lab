# Codex Findings: 5D Follow-up Search Roots and Custody Hardening

UTC: 2026-05-27T13:27:12Z

## Summary

The 5D follow-up automation now searches the real artifact surfaces that can
contain reusable submission bundles and audited MLX caches, and its cache
custody gate is stricter before a local MLX follow-up can become runnable.
This closes the prior paradox where the queue owned the binder but could still
miss the artifacts produced by the surrounding frontier refresh/cycle stack.

## Subagent Inputs Consumed

- Search-root audit: add refresh output roots, results roots, component-response
  cache roots, frontier artifact roots, and `submissions/` when present.
- Operator-surface audit: expose readiness paths, inspect commands, child-worker
  output paths, and safe acquisition worker bounds in both refresh writers.
- MLX custody audit: reject schema/evidence spoofing, authority-smuggling
  fields, uncustodied explicit archive sizes, and global ambiguous bindings.

## What Changed

- `build_pair_frame_5d_coverage_acquisition_queue(...)` now accepts
  `followup_search_roots` and passes them to the binder command.
- `tools/build_5d_canvas_coverage_acquisition_queue.py` exposes
  `--followup-search-root`.
- Frontier refresh library and standalone CLI both pass expanded follow-up
  search roots and expose them in the 5D acquisition summary.
- The standalone refresh CLI exposes
  `--pair-frame-5d-followup-search-root`.
- Operator commands now include post-acquisition inspect commands for the input
  binding and readiness reports.
- Post-acquisition child worker commands now write the advertised worker result
  path.
- Acquisition worker bounds were widened so the binder/child queue rows are not
  skipped by a too-small experiment cap.
- `cache_audit_stamp_blockers(...)` now validates audit stamp schema versions.
- The 5D MLX cache gate now requires canonical cache schema/evidence tags,
  canonical array hash domain, shape and pair-count consistency, false-authority
  fields, and pair-index semantic hashes.
- Explicit archive-size binding must match a discovered submission bundle.
- Binding-report global blockers now clear per-row readiness.

## Authority Contract

All outputs remain planning/local execution surfaces only. The new checks make
local MLX negative-delta execution harder to spoof, but they do not grant score,
promotion, rank/kill, dispatch, or exact-eval authority. Exact-axis rows still
require a byte-closed submission bundle and remain operator-gated.

## Verification

- `ruff check` on changed scheduler/tool/test/cache-audit files passed.
- `pytest src/tac/tests/test_pair_frame_5d_coverage_acquisition_queue.py src/tac/tests/test_pair_frame_5d_extended_operator_queue.py src/tac/tests/test_frontier_rate_attack_refresh_pair_frame_cli.py src/tac/tests/test_frontier_rate_attack_feedback.py -q`
  passed: 82 tests.
- `pytest src/tac/tests/test_mlx_cache_audit.py src/tac/tests/test_mlx_scorer_response.py src/tac/tests/test_mlx_production_contract.py -q`
  passed: 84 tests.
- `tools/lane_maturity.py validate` passed: 1430 lanes validated cleanly.

## Remaining Edge

The queue can now discover and bind real local inputs, but it still needs a
fresh frontier refresh pointed at current receiver-closed bundles and audited
component-response cache roots to drain live follow-up work. The next tranche
should run that refresh, harvest the bound/blocked split, and turn remaining
blocker classes into either queue-owned builders or explicit exact-axis claims.
