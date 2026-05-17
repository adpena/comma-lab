# Frontier Citation Surface Drift Hardening - 2026-05-17

## Status

- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `dispatch_attempted=false`
- Commit target: `main`

## Finding

After landing the canonical frontier scanner and `reports/latest.md` citation
refresh, the live control plane was still split-brain:

- `reports/latest.md` correctly cited:
  - `0.1920513168811056` `[contest-CPU; GHA Linux x86_64 1:1]`
    from archive `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`.
  - `0.20533002902019143` `[contest-CUDA T4]`
    from archive `9cb989cef519ed1771f6c9dc18c988ee93d01a2925da1913d63f9015d6247cf4`.
- `.omx/state/current_focus.md` still described A1 as the best local
  public-axis floor at `0.19284757743677347` `[contest-CPU]` and
  `0.2263520234784395` `[contest-CUDA]`.
- `.omx/state/next_experiments.md` still described A1 as the active floor
  rather than the Rule #6 control substrate.

This is the same frontier-signal-loss bug class as Catalog #316, just one
surface deeper: the report was fresh, but the operator control-plane Markdown
would steer future work from the stale A1 floor.

## Fix

Extended `tac.frontier_scan` from a `reports/latest.md` parser into a
multi-surface citation scanner for:

- `reports/latest.md`
- `.omx/state/current_focus.md`
- `.omx/state/next_experiments.md`

Parser hardening added:

- Markdown-table axis rows.
- Same-line score+axis citations with Markdown backticks/bold between score and
  axis.
- State-doc split citations where the score line is followed by an axis-label
  line.
- Protection against comparison-delta false positives such as "we beat by
  0.00095" in prose.

Catalog #316 now checks all three citation surfaces and reports the exact file
whose cited frontier drifts from canonical state.

## State Refresh

Updated `.omx/state/current_focus.md` and `.omx/state/next_experiments.md`:

- Canonical best CPU anchor is `6bae0201...` at `0.1920513168811056`
  `[contest-CPU; GHA Linux x86_64 1:1]`.
- Canonical best CUDA anchor is `9cb989ce...` at `0.20533002902019143`
  `[contest-CUDA T4]`.
- A1 is explicitly retained as the Rule #6 control substrate, not the best
  current axis floor.

## Verification

Required before landing:

- `tools/scan_best_anchor_per_axis.py --check-drift` must return 0.
- Focused scanner and Catalog #316 tests must pass.
- Existing pre-submission and L5/autopilot/operator-briefing suites must stay
  green or receive a recorded stale-artifact refresh.

## Next Action

With the control plane de-split, resume the active P0 queue:

1. TT5L: materialize missing `report.txt` + per-variant `archive_manifest.json`
   to unblock exact-dispatch authority.
2. Rule #6: continue A1 Ballé/VQ/new-consumed-grammar bolt-ons, comparing
   against the scanner-derived `0.1920513168811056` CPU floor rather than stale
   A1-as-frontier language.
