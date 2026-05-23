# Codex Findings: Reports/Latest Frontier Table Parser Guard

UTC: 2026-05-23T08:57:33Z
Lane: `lane_codex_frontier_scan_reports_latest_table_parser_guard_20260523`
Agent: Codex

## Finding

`tac.frontier_scan` parsed `reports/latest.md` with the same permissive
axis-tagged citation scanner used for compact state docs. That was unsafe for
the generated frontier report because `reports/latest.md` intentionally carries
historical prose, advisory deltas, and exact-eval notes that mention
`[contest-CPU]` / `[contest-CUDA]` near non-frontier decimals.

The live symptom was severe enough to be operator-routing noise: the scanner
reported the cited `reports/latest.md` frontier as `0.000001` /
`0.000022`, i.e. local/prose improvement deltas, while canonical state and the
report's `### Current best` table both carried `0.1920282830` /
`0.2053300290`.

## Landing

Added a `reports/latest.md`-specific parser that prefers the generated
`### Current best` Markdown table when present. If that authoritative section
exists but is malformed, the parser now returns no citation rather than falling
back to historical prose. The generic citation parser is retained as a fallback
only for older/simple fixtures without that section and for state control docs
such as `.omx/state/current_focus.md` and `.omx/state/next_experiments.md`.

`build_frontier_scan_payload()` now routes `reports/latest.md` through that
specialized parser, so the CLI, Catalog #316 preflight, and operator surfaces
consume the same authoritative table values.

## Verification

Commands run:

```bash
.venv/bin/python -m pytest -q src/tac/tests/test_frontier_scan.py src/tac/tests/test_frontier_scan_and_check_316.py src/tac/tests/test_frontier_scan_repo_root_str.py
.venv/bin/ruff check src/tac/frontier_scan.py src/tac/tests/test_frontier_scan.py src/tac/tests/test_frontier_scan_and_check_316.py
.venv/bin/python tools/scan_best_anchor_per_axis.py --check-drift
.venv/bin/python -c "from tac.preflight import check_reports_latest_md_not_stale_vs_canonical_frontier as f; v=f(strict=False, verbose=True); raise SystemExit(1 if v else 0)"
git diff --check
```

Results:

- frontier scan focused suite: `41 passed`
- ruff: passed
- diff check: clean
- live scanner now reports `reports/latest.md: contest_cpu=0.192028,
  contest_cuda=0.205330` instead of `0.000001` / `0.000022`

## Integration Notes

This is a false-signal guardrail, not a score movement. It protects Catalog
#316, operator briefings, autopilot inputs, and future parser consumers from
ranking or routing against prose deltas. No `.gitignore` change was needed: no
new generated artifact namespace or rebuildable bulk output was introduced.
