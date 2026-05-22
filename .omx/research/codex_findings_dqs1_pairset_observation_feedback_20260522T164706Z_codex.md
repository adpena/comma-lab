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

- `experiments/results/cross_family_candidate_portfolio_20260522T165500Z_k004_observed/portfolio.json`
- `experiments/results/cross_family_candidate_portfolio_20260522T165500Z_k004_observed/portfolio.md`
- New recommended candidate: `pairset_diversity_k008`
- New recommended action: `materialize_pairset_archive_and_run_local_controls`

`pairset_diversity_k008` was also materialized, locality-checked, dispatched
CPU-first, and recovered cleanly. It was the best of the tested diversity
pairsets, but it still did not beat compact DQS1 top32.

- Locality report:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/pairset_diversity_k008_materialization_20260522T165616Z/locality_controls.json`
- Archive SHA-256:
  `d18812203b765e63ac925677088bd7abe6c3e30b2daf5584033ab29adc216a06`
- Archive bytes: `178536`
- Selected pairs:
  `[26, 109, 229, 296, 378, 459, 501, 588]`
- Selected frame mismatch count: `0`
- Unselected frame mismatch count: `0`
- Modal result:
  `experiments/results/modal_auth_eval_cpu/dqs1_pairset_diversity_k008_selective_decoderq_cpu_20260522T165616Z/contest_auth_eval.json`
- Modal call id:
  `fc-01KS8A13JRWMBKK8BC1B4GKY6M`
- Exact score:
  `0.19204896820121495` `[contest-CPU]`
- Delta versus compact DQS1 top32 CPU frontier:
  `+0.000020019385125080724`
- Component deltas: PoseNet `+0.0`, SegNet
  `+0.00003600000000000825`, rate `-0.000015980614874927523`
- Runtime tree SHA-256:
  `e9841d6b6c9fc60621e07845c121a6c90cc2c4462b5c106a1e5f67b25409c4d5`
- Inflated aggregate SHA-256:
  `182198a6124e5597235ebd53d1106d4544931686eb83a76b48a96bba75a10373`

The observation ledger now has `4` exact CPU calibration rows:
`prefix_k028`, `pairset_diversity_k002`, `pairset_diversity_k004`, and
`pairset_diversity_k008`. The latest refreshed observation-aware portfolio is:

- `experiments/results/cross_family_candidate_portfolio_20260522T170400Z_observed_pairsets/portfolio.json`
- `experiments/results/cross_family_candidate_portfolio_20260522T170400Z_observed_pairsets/portfolio.md`
- New recommended candidate: `pairset_diversity_k012`
- New recommended action: `materialize_pairset_archive_and_run_local_controls`

`pairset_diversity_k012` was then materialized, locality-checked, dispatched
CPU-first, and recovered cleanly. It is the best tested diversity pairset so
far, but it still does not beat compact DQS1 top32.

- Locality report:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/pairset_diversity_k012_materialization_20260522T170735Z/locality_controls.json`
- Archive SHA-256:
  `afa6be3d8a71226810e4c93b707daf778c4cbe751fd4df28206d70d4e61b2d56`
- Archive bytes: `178540`
- Selected pairs:
  `[26, 98, 134, 167, 257, 320, 376, 430, 467, 492, 520, 588]`
- Selected frame mismatch count: `0`
- Unselected frame mismatch count: `0`
- Modal result:
  `experiments/results/modal_auth_eval_cpu/dqs1_pairset_diversity_k012_selective_decoderq_cpu_20260522T170735Z/contest_auth_eval.json`
- Modal call id:
  `fc-01KS8APGE4NYE8GB3Q27A711D8`
- Exact score:
  `0.1920486316370274` `[contest-CPU]`
- Delta versus compact DQS1 top32 CPU frontier:
  `+0.000019682820937533263`
- Component deltas: PoseNet `+0.0`, SegNet
  `+0.00003300000000000525`, rate `-0.000013317179062439082`
- Runtime tree SHA-256:
  `74f702f758632e50c70aa46e0fe4da828cda3d8b0e91c37c0c39390c7143055e`
- Inflated aggregate SHA-256:
  `d037a6ccf7245ce77e198807d5c82096f866adcd959605a9b678cfe208e131ba`

The observation ledger now has `5` exact CPU calibration rows:
`prefix_k028`, `pairset_diversity_k002`, `pairset_diversity_k004`,
`pairset_diversity_k008`, and `pairset_diversity_k012`. The latest refreshed
observation-aware portfolio is:

- `experiments/results/cross_family_candidate_portfolio_20260522T171700Z_observed_pairsets_k012/portfolio.json`
- `experiments/results/cross_family_candidate_portfolio_20260522T171700Z_observed_pairsets_k012/portfolio.md`
- New recommended candidate: `pairset_diversity_k016`
- New recommended action: `materialize_pairset_archive_and_run_local_controls`

`pairset_diversity_k016` was also materialized, locality-checked, dispatched
CPU-first, and recovered cleanly. It improves over k012 but remains a
regression versus compact DQS1 top32.

- Locality report:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/pairset_diversity_k016_materialization_20260522T171859Z/locality_controls.json`
- Archive SHA-256:
  `408247c02d8c316562492f8247a0ede0711ef05b65729edb5b6cef8c79d230bc`
- Archive bytes: `178544`
- Selected pairs:
  `[26, 68, 109, 134, 167, 242, 259, 320, 376, 412, 440, 467, 492, 501, 544, 588]`
- Selected frame mismatch count: `0`
- Unselected frame mismatch count: `0`
- Modal result:
  `experiments/results/modal_auth_eval_cpu/dqs1_pairset_diversity_k016_selective_decoderq_cpu_20260522T171859Z/contest_auth_eval.json`
- Modal call id:
  `fc-01KS8BAYCXTYEXS13FABSDDR2E`
- Exact score:
  `0.19204229507283993` `[contest-CPU]`
- Delta versus compact DQS1 top32 CPU frontier:
  `+0.000013346256750063068`
- Component deltas: PoseNet `+0.0`, SegNet
  `+0.000024000000000010124`, rate `-0.000010653743249947056`
- Runtime tree SHA-256:
  `3565b57b547af74cc1b63887e60f9c0b0fbf63514890ea764b2a67c6df1a68ac`
- Inflated aggregate SHA-256:
  `05e6fef94492bcdfea0e1229164691bc282c006facff891b443fdc73f397dcf9`

The observation ledger now has `6` exact CPU calibration rows:
`prefix_k028`, `pairset_diversity_k002`, `pairset_diversity_k004`,
`pairset_diversity_k008`, `pairset_diversity_k012`, and
`pairset_diversity_k016`. The latest refreshed observation-aware portfolio is:

- `experiments/results/cross_family_candidate_portfolio_20260522T172900Z_observed_pairsets_k016/portfolio.json`
- `experiments/results/cross_family_candidate_portfolio_20260522T172900Z_observed_pairsets_k016/portfolio.md`
- New recommended candidate: `pairset_diversity_k024`
- New recommended action: `materialize_pairset_archive_and_run_local_controls`

`pairset_diversity_k024` was materialized, locality-checked, dispatched
CPU-first, and recovered cleanly. It is the best tested diversity pairset so
far and is close to compact DQS1 top32, but still does not beat it.

- Locality report used for dispatch:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/pairset_diversity_k024_materialization_20260522T173059Z/locality_controls_rerun_20260522T1738Z.json`
- First locality report:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/pairset_diversity_k024_materialization_20260522T173059Z/locality_controls.json`
- First locality note: the 180-second run emitted a raw-size mismatch while
  inflate children were still writing; the 600-second rerun completed with
  stable full-size raw outputs and zero mismatches.
- Archive SHA-256:
  `a904d580436b44f4871eb6b7bb83f700ea5e493721bd1bf5845476131382e7ff`
- Archive bytes: `178552`
- Selected pairs:
  `[26, 59, 98, 109, 112, 151, 167, 229, 257, 259, 296, 327, 371, 378, 412, 430, 459, 467, 479, 496, 501, 520, 555, 588]`
- Selected frame mismatch count on rerun: `0`
- Unselected frame mismatch count on rerun: `0`
- Modal result:
  `experiments/results/modal_auth_eval_cpu/dqs1_pairset_diversity_k024_selective_decoderq_cpu_20260522T173059Z/contest_auth_eval.json`
- Modal call id:
  `fc-01KS8C4SX0TQ5G50VE7Z6M47PQ`
- Exact score:
  `0.19203562194446488` `[contest-CPU]`
- Delta versus compact DQS1 top32 CPU frontier:
  `+0.000006673128375017656`
- Component deltas: PoseNet `+0.0`, SegNet
  `+0.000012000000000005062`, rate `-0.000005326871624980467`
- Runtime tree SHA-256:
  `d8431f5ca579e93fcc1eb3f357b831e351e056beeac765572a3c95609519a4f5`
- Inflated aggregate SHA-256:
  `34aa715e5871c930843259970988c56ff4c4c2fbcf61a29084d897ded7903b12`

The observation ledger now has `7` exact CPU calibration rows:
`prefix_k028`, `pairset_diversity_k002`, `pairset_diversity_k004`,
`pairset_diversity_k008`, `pairset_diversity_k012`,
`pairset_diversity_k016`, and `pairset_diversity_k024`. The latest refreshed
observation-aware portfolio is:

- `experiments/results/cross_family_candidate_portfolio_20260522T174300Z_observed_pairsets_k024/portfolio.json`
- `experiments/results/cross_family_candidate_portfolio_20260522T174300Z_observed_pairsets_k024/portfolio.md`
- New recommended candidate: `pairset_drop_one_rank013_pair0327`
- New recommended action: `materialize_pairset_archive_and_run_local_controls`

`pairset_drop_one_rank013_pair0327` was materialized, locality-checked,
dispatched CPU-first, and recovered cleanly. It is the closest tested
pairset-family candidate to compact DQS1 top32, but still misses by
`3.3414104685935975e-7` on exact `[contest-CPU]`.

- Locality report:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/pairset_drop_one_rank013_pair0327_materialization_20260522T174354Z/locality_controls.json`
- Archive SHA-256:
  `fa8ccca5331525668433a56c4bcf861a3efbecc81460c707fca79857cf792697`
- Archive bytes: `178559`
- Selected pairs:
  `[26, 59, 68, 98, 109, 112, 134, 151, 167, 229, 242, 257, 259, 296, 320, 371, 376, 378, 412, 430, 440, 459, 467, 479, 492, 496, 501, 520, 544, 555, 588]`
- Selected frame mismatch count: `0`
- Unselected frame mismatch count: `0`
- Modal result:
  `experiments/results/modal_auth_eval_cpu/dqs1_pairset_drop_one_rank013_pair0327_selective_decoderq_cpu_20260522T174354Z/contest_auth_eval.json`
- Modal call id:
  `fc-01KS8CRN8VP339K8BQKAM6BA4H`
- Exact score:
  `0.19202928295713673` `[contest-CPU]`
- Delta versus compact DQS1 top32 CPU frontier:
  `+0.00000033414104685935975`
- Component deltas: PoseNet `+0.0`, SegNet
  `+0.000001000000000007939`, rate `-0.0000006658589531277626`
- Runtime tree SHA-256:
  `eaf208407c6f301b830c0f76059da1a930c2ef7fc2996b24bf86032a249a679f`
- Inflated aggregate SHA-256:
  `7695b3bf0f2ac64f47f82d9ac9a4469612b8209617056049c9ff61d92189939e`

The observation ledger now has `8` exact CPU calibration rows:
`prefix_k028`, `pairset_diversity_k002`, `pairset_diversity_k004`,
`pairset_diversity_k008`, `pairset_diversity_k012`,
`pairset_diversity_k016`, `pairset_diversity_k024`, and
`pairset_drop_one_rank013_pair0327`. The latest refreshed observation-aware
portfolio was regenerated after adversarial review fixed selector-family
leakage, regression-only extrapolation, and missing selected-pair identity
checks. The exact pairset observation-response model is active only for
selector families with sufficient matching observations:

- `experiments/results/cross_family_candidate_portfolio_20260522T180248Z_identity_required_selector_scoped/portfolio.json`
- `experiments/results/cross_family_candidate_portfolio_20260522T180248Z_identity_required_selector_scoped/portfolio.md`
- `experiments/results/cross_family_candidate_portfolio_20260522T180248Z_identity_required_selector_scoped/action_summary.json`
- Model: selector-scoped linear selected-pair-count prior on `contest_cpu`;
  active selector kind `diversity_spaced`, intercept `0.192059196213965`,
  slope per pair `-9.95745144e-07`, residual MSE
  `5.017957909820203e-12`, selected-pair identity verified for all `7`
  candidate-id matches, regression-only cap active at best observed score
  `0.19203562194446488`, false-authority fields preserved.
- New recommended candidate: `pairset_drop_two_r028_021_p0257_0371`
- New recommended action: `materialize_pairset_archive_and_run_local_controls`

`pairset_drop_two_r028_021_p0257_0371` was recovered from the parallel CPU
dispatch. It bought two bytes versus compact top32 but regressed on SegNet, so
it is not a frontier move.

- Modal result:
  `experiments/results/modal_auth_eval_cpu/dqs1_pairset_drop_two_r028_021_p0257_0371_selective_decoderq_cpu_20260522T1804Z/contest_auth_eval.json`
- Modal call id:
  `fc-01KS8DZ3GCDWECP5P80DDGPBG1`
- Exact score:
  `0.1920296170981836` `[contest-CPU]`
- Delta versus compact DQS1 top32 CPU frontier:
  `+0.0000006682820937464751`
- Component deltas: PoseNet `+0.0`, SegNet
  `+0.0000019999999999989644`, rate `-0.0000013317179062443428`
- Archive SHA-256:
  `3ca6c7bb54e98ab04e7ca71a5c709e743bd467eb5ad4501ebe2f9a1dac22222c`
- Archive bytes: `178558`
- Runtime tree SHA-256:
  `af0d5587d0a6c25eb66cd411ebef0797b0ade9a82e1de7271f39d54b73d328fe`
- Inflated aggregate SHA-256:
  `fe5d3d0a9511ce2e776aeb96ff765f2993be92ec5f67af6b2e9e23622bd2c7e6`

`pairset_drop_one_rank021_pair0371` was materialized, locality-checked,
dispatched CPU-first, and recovered cleanly. It is a new exact
`[contest-CPU]` frontier by one charged archive byte with unchanged rounded
PoseNet and SegNet components versus compact top32.

- Locality report:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/pairset_drop_one_rank021_pair0371_materialization_20260522T180446Z/locality_controls.json`
- Archive SHA-256:
  `7a0da5d0fc327cba3f7d1387a544fd5ce5f05bc56ecc8e12cd5097141672f4fe`
- Archive bytes: `178559`
- Selected pairs:
  `[26, 59, 68, 98, 109, 112, 134, 151, 167, 229, 242, 257, 259, 296, 320, 327, 376, 378, 412, 430, 440, 459, 467, 479, 492, 496, 501, 520, 544, 555, 588]`
- Selected frame mismatch count: `0`
- Unselected frame mismatch count: `0`
- Modal result:
  `experiments/results/modal_auth_eval_cpu/dqs1_pairset_drop_one_rank021_pair0371_selective_decoderq_cpu_20260522T180446Z/contest_auth_eval.json`
- Modal call id:
  `fc-01KS8E07E7MJVKXCQ0WW26WQZH`
- Exact score:
  `0.19202828295713675` `[contest-CPU]`
- Delta versus compact DQS1 top32 CPU frontier:
  `-0.0000006658589531138848`
- Component deltas: PoseNet `+0.0`, SegNet `+0.0`, rate
  `-0.0000006658589531221714`
- Runtime tree SHA-256:
  `da4a69cb7412cd879e460ed0ffe3ad9d48f2b1aa73215696ce8de492b7dac4b4`
- Inflated aggregate SHA-256:
  `abd960605e57a3f1a6a8fe8b21ebfd4ff4d50cd15300482d8658b3d53e63ddb5`

The observation ledger now has `10` exact CPU calibration rows, including the
new drop-two and drop-one outcomes. The latest refreshed observation-aware
portfolio is:

- `experiments/results/cross_family_candidate_portfolio/20260522T181725Z_observed_pairsets_new_frontier/portfolio.json`
- `experiments/results/cross_family_candidate_portfolio/20260522T181725Z_observed_pairsets_new_frontier/portfolio.md`
- `experiments/results/cross_family_candidate_portfolio/20260522T181725Z_observed_pairsets_new_frontier/action_summary.json`
- Model: selector-scoped linear selected-pair-count prior on `contest_cpu`;
  active selector kind `diversity_spaced`, selected-pair identity verified for
  all `9` candidate-id matches in the acquisition surface, false-authority
  fields preserved.
- New recommended candidate: `pairset_drop_one_rank010_pair0376`
- New recommended action: `materialize_pairset_archive_and_run_local_controls`

The next harvested results changed this from a one-off interpretation into a
component marginal model consumed by the portfolio planner.

- `pairset_drop_one_rank021_pair0371` exact `[contest-CUDA T4]` replay:
  `experiments/results/modal_auth_eval/dqs1_pairset_drop_one_rank021_pair0371_selective_decoderq_cuda_20260522T1816Z/contest_auth_eval.json`
- Exact CUDA score: `0.22619176954300405`
- Delta versus compact DQS1 top32 CUDA:
  `+0.00000133414104686`
- Component deltas versus compact DQS1 top32 CUDA: PoseNet `+0.0`, SegNet
  `+0.000002`, rate `-0.00000066585895312`
- CUDA inflated aggregate SHA-256:
  `8d09f5082e42d69205c3f4ad118af892cdf654d8e2a9395d2fb775aa30a44166`
- CUDA runtime tree SHA-256:
  `6b6e385bfc1368a34c058392a5dc0b0f3eb42c1f340f262845b3dea4dd5fb5f1`
- `pairset_drop_one_rank010_pair0376` exact `[contest-CPU]` replay:
  `experiments/results/modal_auth_eval_cpu/dqs1_pairset_drop_one_rank010_pair0376_selective_decoderq_cpu_20260522T182154Z/contest_auth_eval.json`
- Exact CPU score: `0.19202928295713673`
- Delta versus current rank021 CPU frontier:
  `+0.0000010000000000`
- Component deltas versus compact DQS1 top32 CPU: PoseNet `+0.0`,
  SegNet `+0.000001`, rate `-0.00000066585895312`
- Rank010 archive SHA-256:
  `1533283b3e4f6ad10a2eb736a098df739f0b161d0385df56b89ba94d190a8237`
- Rank010 inflated aggregate SHA-256:
  `8c13092466131b70228ee09993f2f5b29ac2d3f146bf44be200eecbf91d05e7e`
- `pairset_drop_one_rank026_pair0320` exact `[contest-CPU]` replay:
  `experiments/results/modal_auth_eval_cpu/dqs1_pairset_drop_one_rank026_pair0320_selective_decoderq_cpu_20260522T183308Z/contest_auth_eval.json`
- Exact CPU score: `0.19202928295713673`
- Delta versus current rank021 CPU frontier:
  `+0.0000010000000000`
- Component deltas versus compact DQS1 top32 CPU: PoseNet `+0.0`,
  SegNet `+0.000001`, rate `-0.00000066585895312`
- Rank026 archive SHA-256:
  `bf6c2135ce874c7a99b727af76cdbf0dc31deca128b5ee462334dde460731296`
- Rank026 inflated aggregate SHA-256:
  `d8f43d23d84ca3b7631a7a5109d95f8f1472b903b91fedf50f8176949f1377d2`

Canonicalization/wire-in:

- Commit `130f09683` added `pairset_component_marginal_model.v1` to
  `tac.optimization.cross_family_candidate_portfolio`.
- The follow-up landing adds reusable component-marginal helpers under
  `tac.optimization.pairset_component_marginal`, the canonical equation
  `pairset_component_marginal_score_decomposition_v1`, and the xray primitive
  `tac.xray.pairset_component_marginal`.
- The model consumes exact-axis observations with selected-pair identity,
  component deltas, and acquisition operations.
- Current generated portfolio:
  `experiments/results/cross_family_candidate_portfolio/20260522T185121Z_observed_pairsets_rank026_xray_equation_refs/portfolio.json`
- Current action summary:
  `experiments/results/cross_family_candidate_portfolio/20260522T185121Z_observed_pairsets_rank026_xray_equation_refs/action_summary.json`
- Portfolio SHA-256:
  `ce035dc78a07bce924bddca50646d60f7e36c353a848848bce9ba250af542154`
- Action summary SHA-256:
  `a8255b7d3d2b6c69bcd9257864cc757603df4c05dd29b0225423504e1c160220`
- Component marginal summary: CPU-safe observed drop pair `[371]`;
  CPU-protected observed drop pairs `[327, 376, 320]`; CUDA-protected observed
  drop pair `[371]`; cross-axis transfer diagnostic present for
  `pairset_drop_one_rank021_pair0371`.
- New recommended candidate:
  `pairset_drop_one_rank027_pair0378`
- New recommended action:
  `materialize_pairset_archive_and_run_local_controls`
- The regenerated portfolio contains `canonical_signal_refs` pointing this
  exact-axis component ledger to xray primitive `pairset_component_marginal`,
  canonical equation `pairset_component_marginal_score_decomposition_v1`, and
  master-gradient consumers including `per_pair_difficulty_atlas`,
  `per_pair_pareto_envelope`, `per_pair_lagrangian_lambda_bisection`, and
  `per_pair_coding_budget_allocation`.

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
- `tools/recover_modal_auth_eval.py` on k008 Modal CPU call `fc-01KS8A13JRWMBKK8BC1B4GKY6M`
- `tools/run_decoder_q_selective_runtime_locality_controls.py` on k012
- `tools/recover_modal_auth_eval.py` on k012 Modal CPU call `fc-01KS8APGE4NYE8GB3Q27A711D8`
- `tools/run_decoder_q_selective_runtime_locality_controls.py` on k016
- `tools/recover_modal_auth_eval.py` on k016 Modal CPU call `fc-01KS8BAYCXTYEXS13FABSDDR2E`
- `tools/run_decoder_q_selective_runtime_locality_controls.py` on k024 first 180-second run
- `tools/run_decoder_q_selective_runtime_locality_controls.py` on k024 600-second rerun
- `tools/recover_modal_auth_eval.py` on k024 Modal CPU call `fc-01KS8C4SX0TQ5G50VE7Z6M47PQ`
- `tools/run_decoder_q_selective_runtime_locality_controls.py` on drop-one rank013 pair0327
- `tools/recover_modal_auth_eval.py` on drop-one rank013 pair0327 Modal CPU call `fc-01KS8CRN8VP339K8BQKAM6BA4H`
- `tools/recover_modal_auth_eval.py` on drop-two r028/r021 Modal CPU call `fc-01KS8DZ3GCDWECP5P80DDGPBG1`
- `tools/run_decoder_q_selective_runtime_locality_controls.py` on drop-one rank021 pair0371
- `tools/recover_modal_auth_eval.py` on drop-one rank021 pair0371 Modal CPU call `fc-01KS8E07E7MJVKXCQ0WW26WQZH`
- `tools/recover_modal_auth_eval.py` on drop-one rank021 pair0371 Modal CUDA call `fc-01KS8ERK4DVX1EBGVHP80H4QRW`
- `tools/recover_modal_auth_eval.py` on drop-one rank010 pair0376 Modal CPU call `fc-01KS8EY13VWJZD0QDND9ER6G34`
- `tools/run_decoder_q_selective_runtime_locality_controls.py` on drop-one rank026 pair0320
- `tools/recover_modal_auth_eval.py` on drop-one rank026 pair0320 Modal CPU call `fc-01KS8FKFADCWKVAN8FGDFW5M69`
- `.venv/bin/python -m pytest src/tac/xray/tests/test_pairset_component_marginal.py src/tac/canonical_equations/tests/test_pairset_component_marginal.py src/tac/xray/tests/test_registry.py src/tac/xray/tests/test_integration_with_solver_stack.py src/tac/xray/tests/test_compositional_integration.py src/tac/tests/test_cross_family_candidate_portfolio.py src/tac/tests/test_plan_cross_family_candidate_portfolio_cli.py -q`
- `.venv/bin/python -m pytest src/tac/tests/test_pairset_component_marginal.py src/tac/xray/tests/test_pairset_component_marginal.py src/tac/canonical_equations/tests/test_pairset_component_marginal.py -q`
- `.venv/bin/ruff check src/tac/tests/test_pairset_component_marginal.py src/tac/xray/tests/test_pairset_component_marginal.py src/tac/canonical_equations/tests/test_pairset_component_marginal.py src/tac/optimization/pairset_component_marginal.py src/tac/canonical_equations/pairset_component_marginal.py src/tac/xray/pairset_component_marginal.py`
- `.venv/bin/python tools/plan_cross_family_candidate_portfolio.py --incumbent-score 0.205330029 --incumbent-score-by-axis contest_cpu=0.19202828295713675 --pairset-acquisition experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/pairset_acquisition/dqs1_pairset_acquisition_dense_tail_20260522T1812Z.json --observation-jsonl experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/dynamic_learned_sweep/dqs1_dynamic_sweep_observations.jsonl --json-out experiments/results/cross_family_candidate_portfolio/20260522T185121Z_observed_pairsets_rank026_xray_equation_refs/portfolio.json --md-out experiments/results/cross_family_candidate_portfolio/20260522T185121Z_observed_pairsets_rank026_xray_equation_refs/portfolio.md --summary-json-out experiments/results/cross_family_candidate_portfolio/20260522T185121Z_observed_pairsets_rank026_xray_equation_refs/action_summary.json --top-k 32 --top-actions 8`
- `.venv/bin/python -m pytest src/tac/tests/test_cross_family_candidate_portfolio.py src/tac/tests/test_plan_cross_family_candidate_portfolio_cli.py src/tac/tests/test_mlx_dynamic_sweep_observations.py src/tac/tests/test_decoder_q_pairset_acquisition.py -q`
- `.venv/bin/python -m pytest src/tac/tests/test_cross_family_candidate_portfolio.py src/tac/tests/test_mlx_dynamic_sweep_observations.py -q`
- `.venv/bin/python -m pytest src/tac/tests/test_mlx_dynamic_sweep_observations.py src/tac/tests/test_cross_family_candidate_portfolio.py src/tac/tests/test_plan_cross_family_candidate_portfolio_cli.py -q`
- `.venv/bin/ruff check src/tac/optimization/cross_family_candidate_portfolio.py src/tac/optimization/mlx_dynamic_sweep_observations.py tools/plan_cross_family_candidate_portfolio.py src/tac/tests/test_cross_family_candidate_portfolio.py`
- `.venv/bin/ruff check src/tac/tests/test_mlx_dynamic_sweep_observations.py src/tac/tests/test_plan_cross_family_candidate_portfolio_cli.py`

## Authority

- `score_claim=false` for planner, advisory, feedback, and locality artifacts.
- The k002 Modal CPU JSON is exact `[contest-CPU]` evidence only, not CUDA
  promotion evidence.
- k004 now has exact `[contest-CPU]` evidence only, not CUDA promotion
  evidence.
- k008 now has exact `[contest-CPU]` evidence only, not CUDA promotion
  evidence.
- k012 now has exact `[contest-CPU]` evidence only, not CUDA promotion
  evidence.
- k016 now has exact `[contest-CPU]` evidence only, not CUDA promotion
  evidence.
- k024 now has exact `[contest-CPU]` evidence only, not CUDA promotion
  evidence.
- drop-one rank013 pair0327 now has exact `[contest-CPU]` evidence only, not
  CUDA promotion evidence.
- drop-two r028/r021 now has exact `[contest-CPU]` evidence only, not CUDA
  promotion evidence.
- drop-one rank021 pair0371 now has exact `[contest-CPU]` and
  `[contest-CUDA T4]` evidence. It remains CPU-frontier only; the CUDA replay
  regressed, so there is no CUDA promotion evidence.
- drop-one rank010 pair0376 has exact `[contest-CPU]` evidence only and
  regressed versus the current rank021 CPU frontier.
- drop-one rank026 pair0320 has exact `[contest-CPU]` evidence only and
  regressed versus the current rank021 CPU frontier.
