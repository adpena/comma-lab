# Repair Dynamics Palette Prior Binding

Codex canonicalized the empirical live-archive K=16 repair palette as a
fail-closed repair-dynamics prior in the repair-budget materializer binding
path.

## Finding

The live 6bae0201 archive manifest reports a canonical selector palette with
16 modes: one identity mode, 15 frame-0 modes, and zero frame-1 modes. The
non-identity modes are global frame-0 chroma, luma, RGB-bias, and one roll
operator, which is evidence that repair allocation should treat frame-0 global
color/geometry calibration as a first-class interaction prior before spending
bytes on leaf pixel repairs.

## Wiring Landed

- `frontier_rate_attack_repair_dynamics_palette_prior.v1` records mode counts,
  frame counts, family counts, frame-0 fractions, zero-frame-1 status, and
  action-functional hints.
- Materializer manifests can provide palette modes through `selector_palette`,
  `canonical_palette`, `palette_modes`, nested `archive_manifest`, or related
  selector manifest fields.
- `tools/build_frontier_repair_budget_materializer_binding_report.py` accepts
  `--repair-palette-mode` so operator-supplied palette facts can be captured
  without becoming score, spend, promotion, or dispatch authority.
- The repair-budget waterfill queue now has a concrete binding step before the
  execution audit, so parent/child candidate chains must bind to
  receiver-consumed materializer manifests before the final exact-eval refusal
  can clear materialization blockers.

## Authority Status

The prior is planning evidence only. It is false-authority throughout:
`score_claim=false`, `promotion_eligible=false`,
`rank_or_kill_eligible=false`, `budget_spend_allowed=false`, and
`ready_for_exact_eval_dispatch=false`.

## Verification

- `.venv/bin/ruff check` on touched scheduler/tool/test files passed.
- `pytest src/tac/tests/test_repair_budget_materialization_execution.py -q`
  passed: 3 tests.
- `pytest src/tac/tests/test_frontier_rate_attack_feedback.py -q` passed:
  33 tests.
- `tools/lane_maturity.py validate` passed: 1410 lanes clean.
- Review policy checks passed on the touched scheduler, CLIs, and tests.

## Next Integration Target

The prior should next feed the waterfill allocator as a weighted interaction
term: prioritize frame-0 global chroma/luma/RGB/roll repair probes, require
parent-child synergy remeasurement before spending freed bytes, and keep all
frame-1 repair proposals unranked until exact component-response evidence
exists.
