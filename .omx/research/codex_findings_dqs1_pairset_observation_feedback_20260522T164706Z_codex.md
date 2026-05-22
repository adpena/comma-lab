# Codex Findings: DQS1 Pairset Observation Feedback

Date: 2026-05-22T16:47:06Z

## Verdict

`pairset_diversity_k002` is byte-closed and contest-CPU valid, but it is not a
frontier move. The exact Modal CPU score is `0.19205563890644933`
`[contest-CPU]`, a regression of `+0.00002669009035946579` versus the current
DQS1 top32 gap-ULEB CPU frontier `0.19202894881608987`.

The regression decomposes as:

- PoseNet: `+0.0`
- SegNet: `+0.00004600000000001131`
- Rate: `-0.00001930990964053858`

Interpretation: this pairset bought rate, but selected frames raised SegNet
more than the rate savings paid back. Do not CUDA-promote k002 from this
evidence.

## Artifacts

`pairset_diversity_k002` exact CPU:

- Modal result:
  `experiments/results/modal_auth_eval_cpu/dqs1_pairset_diversity_k002_selective_decoderq_cpu_20260522T162726Z/contest_auth_eval.json`
- Modal wrapper result:
  `experiments/results/modal_auth_eval_cpu/dqs1_pairset_diversity_k002_selective_decoderq_cpu_20260522T162726Z/modal_cpu_auth_eval_result.json`
- Archive SHA-256:
  `4432525de41c9df0c9edecab0447fa3320d9d5e5047675fced069ca8213e630e`
- Archive bytes: `178531`
- Runtime tree SHA-256:
  `af4ba9dfcfb0d5091f96add104fb811ade169f682b02d3111a97613e95790a08`
- Inflated aggregate SHA-256:
  `02eaa49d0dd6f12cb16791491f3c38121fbee8bac35beef65846058f6efae9a7`

`pairset_diversity_k002` local advisory sign calibration:

- Advisory JSON:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/pairset_diversity_k002_materialization_20260522T162726Z/dqs1_pairset_diversity_k002_cpu_advisory_venv.json`
- Feedback JSON:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/pairset_diversity_k002_materialization_20260522T162726Z/decoder_q_selective_runtime_feedback.json`
- Local advisory score: `0.19206763890644935` `[macOS-CPU advisory]`
- Local baseline score: `0.19206131688110561`
- Local score delta: `+0.000006322025343730164`
- Sign label: `+1` regression
- Advisory raw SHA-256:
  `dab3f18f7affcecc71ee566b0113526daa3c28ade59f60d6f185c3af7b2c293a`
- Combined sign-calibration summary:
  `label_count=3`, `regression_label_count=2`, `improvement_label_count=1`

Planner feedback:

- Observation JSONL:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/dynamic_learned_sweep/dqs1_dynamic_sweep_observations.jsonl`
- Observation-aware portfolio:
  `experiments/results/cross_family_candidate_portfolio_20260522T164000Z_observed_pairset/portfolio.json`
- New recommended candidate:
  `pairset_diversity_k004`
- New recommended action:
  `materialize_pairset_archive_and_run_local_controls`

## Code Change

`tools/plan_cross_family_candidate_portfolio.py` now accepts
`--observation-jsonl` and `--incumbent-score-by-axis AXIS=SCORE`.
`tac.optimization.cross_family_candidate_portfolio` validates observation rows
through `tac.optimization.mlx_dynamic_sweep_observations`, attaches exact-axis
feedback to matching candidate IDs, and demotes same-axis repeat operator
actions without granting score, rank, promotion, or dispatch authority.

This closes the loop that previously let the portfolio re-recommend an already
observed candidate.

## Next Candidate

`pairset_diversity_k004` was materialized from the observation-aware portfolio.

- Selected pairs: `[26, 242, 440, 588]`
- Selected frames under `pair_all_frames`:
  `[52, 53, 484, 485, 880, 881, 1176, 1177]`
- Archive:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/pairset_diversity_k004_materialization_20260522T164149Z/submission_dir/archive.zip`
- Archive SHA-256:
  `fc01aca62f07cd4959f98e7fe33f99c6f1bc0f5812b26fe40151ad8fb0e8b392`
- Archive bytes: `178535`
- DQS1 payload bytes: `18`
- DQS1 payload SHA-256:
  `8e97f763175aa9056fe1dc29ff8534cc69927d149316ba94137d0c0ec32e1024`

Locality controls passed:

- Locality report:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/pairset_diversity_k004_materialization_20260522T164149Z/locality_controls.json`
- Selected frame mismatch count: `0`
- Unselected frame mismatch count: `0`
- Raw size mismatch count: `0`
- Missing raw file count: `0`

`pairset_diversity_k004` was then dispatched CPU-first and recovered cleanly.
It also regressed versus the compact DQS1 top32 CPU frontier.

- Modal result:
  `experiments/results/modal_auth_eval_cpu/dqs1_pairset_diversity_k004_selective_decoderq_cpu_20260522T164149Z/contest_auth_eval.json`
- Modal call id:
  `fc-01KS89ERK62GJHGQM4VZHDKN66`
- Exact score:
  `0.19205830234226182` `[contest-CPU]`
- Delta versus compact DQS1 top32 CPU frontier:
  `+0.000029353526171949085`
- Component deltas: PoseNet `+0.0`, SegNet
  `+0.00004600000000001131`, rate `-0.000016646473828055286`
- Runtime tree SHA-256:
  `602eee3dce811530a79ea1199fb8f9eb8257a489b1cd241c15a02ddf59487863`
- Inflated aggregate SHA-256:
  `8b6418747af5bc7f35263cc90959ccd1a1613c63ade88c745b4fbd0a2b68eb0c`

The observation ledger now has `3` exact CPU calibration rows:
`prefix_k028`, `pairset_diversity_k002`, and `pairset_diversity_k004`.
The refreshed observation-aware portfolio is:

- `experiments/results/cross_family_candidate_portfolio_20260522T165500Z_observed_pairsets/portfolio.json`
- New recommended candidate: `pairset_diversity_k008`
- New recommended action: `materialize_pairset_archive_and_run_local_controls`

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_cross_family_candidate_portfolio.py -q`
- `.venv/bin/python -m pytest src/tac/tests/test_cross_family_candidate_portfolio.py src/tac/tests/test_mlx_dynamic_sweep_observations.py -q`
- `.venv/bin/python -m pytest src/tac/tests/test_mlx_dynamic_learned_sweep.py -q`
- `.venv/bin/ruff check src/tac/optimization/cross_family_candidate_portfolio.py tools/plan_cross_family_candidate_portfolio.py src/tac/tests/test_cross_family_candidate_portfolio.py`
- `tools/recover_modal_auth_eval.py` on k002 Modal CPU call `fc-01KS88E8S109WQN8FHETA7227T`
- `experiments/contest_auth_eval.py --device cpu` on k002 local advisory
- `tools/build_decoder_q_selective_runtime_feedback.py` on k002 advisory/locality/materialization artifacts
- `tools/run_decoder_q_selective_runtime_locality_controls.py` on k004
- `tools/recover_modal_auth_eval.py` on k004 Modal CPU call `fc-01KS89ERK62GJHGQM4VZHDKN66`

## Authority

- `score_claim=false` for planner, advisory, feedback, and locality artifacts.
- The k002 Modal CPU JSON is exact `[contest-CPU]` evidence only, not CUDA
  promotion evidence.
- k004 now has exact `[contest-CPU]` evidence only, not CUDA promotion
  evidence.
