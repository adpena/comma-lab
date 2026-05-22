# Codex Findings: DQS1 Pairset Diversity K002 Materialization and CPU Dispatch

Date: 2026-05-22T16:35:00Z

## Verdict

TERMINAL NON-PROMOTIONAL. `pairset_diversity_k002` was materialized into a
byte-closed DQS1 selective-runtime archive, passed local inflate locality
controls, and completed detached Modal CPU auth eval. The exact CPU score is
`0.19205563890644933 [contest-CPU]`, which is worse than the current DQS1
CPU-axis frontier `0.19202894881608987 [contest-CPU]` by
`+0.00002669009035946579`.

No CUDA dispatch was launched. No promotion, rank/kill, or public score claim is
made here.

## Materialized Candidate

Source planner recommendation:

- portfolio artifact:
  `experiments/results/cross_family_candidate_portfolio_20260522T162213Z/cross_family_candidate_portfolio.json`
- candidate: `pairset_diversity_k002`
- selected pairs: `[26, 588]`
- selected frames under `pair_all_frames`: `[52, 53, 1176, 1177]`

Packet plan:

- `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/pairset_diversity_k002_materialization_20260522T162726Z/decoder_q_selective_runtime_packet_plan.json`
- status: `runtime_packet_l0_byte_plan_ready`
- pair encoding: `sorted_gap_uleb`
- payload bytes: `14`
- pair-index payload bytes: `3`

Materialized submission:

- submission dir:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/pairset_diversity_k002_materialization_20260522T162726Z/submission_dir`
- archive:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/pairset_diversity_k002_materialization_20260522T162726Z/submission_dir/archive.zip`
- archive SHA-256:
  `4432525de41c9df0c9edecab0447fa3320d9d5e5047675fced069ca8213e630e`
- archive bytes: `178531`
- DQS1 tail SHA-256:
  `f6e7bd14b5abc01e79ce621305b7e162bc4980df84c4fd47fc64411a9f4516d1`

## Locality Controls

Report:

- `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/pairset_diversity_k002_materialization_20260522T162726Z/locality_controls.json`

Result:

- locality controls passed: `true`
- selected frame mismatch count: `0`
- unselected frame mismatch count: `0`
- raw size mismatch count: `0`
- missing raw file count: `0`

Interpretation:

- selected frames in the selective runtime match the global decoder-q mutation
- unselected frames in the selective runtime match the parent runtime
- this is a no-score local inflate/control artifact, not score authority

## CPU Exact Eval Dispatch

Detached Modal CPU auth eval was launched after local controls passed and was
recovered successfully.

- lane id:
  `lane_dqs1_pairset_diversity_k002_selective_decoderq_exact_cpu_20260522`
- instance/job id:
  `dqs1_pairset_diversity_k002_cpu_20260522T162726Z`
- Modal call id:
  `fc-01KS88E8S109WQN8FHETA7227T`
- output dir:
  `experiments/results/modal_auth_eval_cpu/dqs1_pairset_diversity_k002_selective_decoderq_cpu_20260522T162726Z`
- uploaded CPU runtime tree SHA-256:
  `af4ba9dfcfb0d5091f96add104fb811ade169f682b02d3111a97613e95790a08`
- single-axis waiver:
  `CPU-first after local locality controls; launch CUDA only if CPU survives.`

Current recovery status:

- `status = completed_contest_cpu_modal_auth_eval_recovered`
- `score = 0.19205563890644933 [contest-CPU]`
- `archive_size_bytes = 178531`
- `avg_segnet_dist = 0.00056024`
- `avg_posenet_dist = 0.00002943`
- `score_seg_contribution = 0.056024000000000004`
- `score_pose_contribution = 0.017155174146594957`
- `score_rate_contribution = 0.11887646475985437`
- `raw_output_aggregate_sha256 =
  02eaa49d0dd6f12cb16791491f3c38121fbee8bac35beef65846058f6efae9a7`
- `score_claim = false` for this findings memo and replanning observation
- `ready_for_exact_eval_dispatch = false`

Recovered artifacts:

- `experiments/results/modal_auth_eval_cpu/dqs1_pairset_diversity_k002_selective_decoderq_cpu_20260522T162726Z/contest_auth_eval.json`
- `experiments/results/modal_auth_eval_cpu/dqs1_pairset_diversity_k002_selective_decoderq_cpu_20260522T162726Z/modal_cpu_auth_eval_result.json`

## Replanning Feedback

The exact CPU result was appended to the ignored local dynamic-sweep observation
JSONL for replanning only:

- `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/dynamic_learned_sweep/dqs1_dynamic_sweep_observations.jsonl`
- observation row count after de-duplication: `2`
- K002 observation status in the regenerated portfolio:
  `observed_exact_axis_regressed_vs_axis_baseline`
- K002 CPU delta versus DQS1 CPU baseline:
  `+0.00002669009035946579`

The cross-family portfolio planner was hardened to consume exact-axis
observation JSONL without cross-axis score comparison. CPU observations require
a `contest_cpu` baseline before they can be classified as improved/regressed;
they are not compared to the CUDA incumbent.

Axis-corrected regenerated portfolio:

- `experiments/results/cross_family_candidate_portfolio_20260522T164315Z_axis_corrected/portfolio.json`
- `experiments/results/cross_family_candidate_portfolio_20260522T164315Z_axis_corrected/portfolio.md`
- recommended next candidate: `pairset_diversity_k004`
- recommended next action:
  `materialize_pairset_archive_and_run_local_controls`
- `pairset_diversity_k002` action rank after observation feedback: `40`

## Authority

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `dispatch_attempted=true` only for the CPU auth-eval call above
- CUDA dispatch not launched because CPU-axis filter failed

## Next Step

Materialize `pairset_diversity_k004`, run the same local locality controls, and
only spend exact CPU if the byte/locality gates pass. Do not revisit
`pairset_diversity_k002` on the same CPU axis unless a later upstream scorer,
runtime, or baseline change invalidates this observation.
