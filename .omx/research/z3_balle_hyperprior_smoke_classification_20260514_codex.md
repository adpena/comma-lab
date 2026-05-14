# Z3 Ballé Hyperprior Smoke Classification - 2026-05-14

## Result

Z3 Modal T4 smoke returned successfully, but it is not a score-bearing result.

- Output directory:
  `experiments/results/lane_substrate_z3_balle_hyperprior_bolton_modal_t4_dispatch_20260514T154958Z__smoke__100ep_modal`
- Harvest summary:
  `experiments/results/lane_substrate_z3_balle_hyperprior_bolton_modal_t4_dispatch_20260514T154958Z__smoke__100ep_modal/harvested_artifacts/_harvest_summary.json`
- Stats:
  `experiments/results/lane_substrate_z3_balle_hyperprior_bolton_modal_t4_dispatch_20260514T154958Z__smoke__100ep_modal/harvested_artifacts/lane_substrate_z3_balle_hyperprior_bolton_results/output/stats.json`
- Return code: `0`
- Elapsed: `13.881049523` seconds
- Estimated cost: `$0.0022749497829361114`
- Archive ZIP bytes: `2,538`
- Archive ZIP SHA-256:
  `a46b98fbf837c126ace245667ddd46b1f7e4f2c2a942755e4789d827c412819c`
- Evidence grade: `smoke-no-scorer`
- Score claim: `false`
- Promotion eligible: `false`
- Ready for exact eval dispatch: `false`

## Classification

`completed_modal_training_recovered_no_score_claim`.

The smoke proved that the tiny Z3 hyperprior trainer runs on Modal T4 and emits
a byte artifact. It did not load or run the contest scorer, and therefore it
must not be used for ranking, retirement, promotion, or score movement.

Blocking fields from `stats.json`:

- `smoke_no_scorer_load`
- `requires_separate_auth_eval_result_review_before_score_claim`

## Next Gate

Do not dispatch a Z3 exact eval from this smoke artifact. The next useful Z3
step is a full non-smoke composition archive with an operational consumed
runtime effect over a real base archive, followed by the normal claimed exact
auth-eval path.
