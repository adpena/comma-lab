# Codex Findings: DQS1 Rank024 Local Advisory And Boundary Hardening

UTC: 2026-05-22T20:56:06Z
Agent: Codex
Lane: `lane_codex_dqs1_rank024_local_first_execution_20260522`
Authority: observation only; no score claim; no promotion; no rank/kill.

## Result

Candidate: `pairset_drop_one_rank024_pair0112`

Selected pair indices:

`26,59,68,98,109,134,151,167,229,242,257,259,296,320,327,371,376,378,412,430,440,459,467,479,492,496,501,520,544,555,588`

Local-first queue outcome:

- `plan_packet`: succeeded.
- `materialize`: succeeded.
- `locality_controls`: succeeded.
- `local_cpu_advisory`: succeeded.

Artifacts:

- Packet plan:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/selector_pareto/packet_plans/drop_rank024_pair0112.json`
- Materialization manifest:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank024_pair0112/materialization_manifest.json`
- Locality controls:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank024_pair0112/locality_controls.json`
- Local advisory:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank024_pair0112/local_cpu_advisory.json`
- Eureka signal:
  `.omx/research/local_cpu_contest_drift_eureka_pairset_drop_one_rank024_pair0112_20260522T205606Z.json`

Custody hashes:

- Archive SHA-256:
  `5dee03930e42c5a46834f4d0d7d7d38f00e475ebff3ad7a0943b9758a8afcc2c`
- Archive bytes: `178559`
- Runtime tree SHA-256:
  `e5e307c5ede2a94846f38bc516ac7572c58be11831630ba784e3759c09de2374`
- Runtime content tree SHA-256:
  `7caa6375ee623217a7cefc0a49da587f1d8f10337f8f1c002bfdbf9835b8c68c`
- Inflated output manifest SHA-256:
  `8577e363fabdaad7da024d19167cb981402551d87fd8f978a20812f851826594`
- Inflated output aggregate SHA-256:
  `3b497807d2548379d30fb302a188bc07c65ab12244a5f91b2d16eaea5e600fc0`

Local advisory metrics:

- Axis: `[macOS-CPU advisory]`
- `canonical_score`: `0.19203828295713676`
- `avg_segnet_dist`: `0.00055988`
- `avg_posenet_dist`: `0.00002943`
- `rate_unscaled`: `0.0047558043524216715`
- `score_claim`: `false`
- `promotion_eligible`: `false`
- `rank_or_kill_eligible`: `false`

Drift/eureka interpretation:

- Calibration: `.omx/research/local_cpu_contest_drift_calibration_dqs1_fec6_20260522T194800Z.json`
- Trust region: `dqs1_fec6_like_same_archive_segnet_rounding`
- Calibration bias local-minus-contest: `0.000010000000000010001`
- Guard band: `0.000003`
- Projected contest CPU score: `0.19202828295713675`
- Conservative projected contest CPU score:
  `0.19203128295713676`
- Current auth frontier score:
  `0.19202828295713675`
- Eureka trigger: `false`
- Eureka margin: `-0.0000030000000000030003`
- Recommended action: `observe_only`

Interpretation: rank024 appears to tie the current CPU frontier after local
CPU drift correction, but the conservative guard band is worse than the
frontier. It should not trigger exact CPU/CUDA spend by itself.

## Boundary Findings Integrated This Turn

Read-only boundary review found that `.omx/state/current_focus.md` and
`.omx/state/next_experiments.md` still advertised the superseded compact top32
CPU anchor as the current CPU frontier. This turn updated both mirrors to the
canonical rank021/pair0371 CPU frontier and added
`src/tac/tests/test_frontier_state_docs.py` so these control-plane docs must
mirror `.omx/state/canonical_frontier_pointer.json`.

The same review found that component-marginal action-prior wiring was
implemented but under-discoverable. This turn added exact producer/consumer refs
for:

- `tools/canonicalize_pairset_component_marginal_signal.py`
- `tac.optimization.cross_family_candidate_portfolio._component_marginal_action_prior`
- `tac.optimization.cross_family_candidate_portfolio._operator_action_priority`

## Queue Hardening Integrated

A default-state worker was already running rank024 when a separate `/tmp`
state worker was launched. The duplicate worker re-ran plan/materialize/locality
against the same artifact paths. The duplicate `/tmp` worker was stopped after
the default-state worker had already completed locality and entered the local
CPU advisory step.

This exposed a real hardening target: executing the same queue with
noncanonical state can bypass SQLite claim coordination and duplicate writes.
The landing now:

- refuses `run-once --execute` and `run-worker --execute` on noncanonical state
  unless `--noncanonical-state-rationale` is provided;
- refuses execution when blocking orphaned reroute rows remain unless an
  explicit `--orphaned-state-rationale` is provided;
- adds `retire-orphans` to mark blocking stale reroute rows as `skipped`;
- records override rationales in worker telemetry;
- tightens queue false-authority postconditions so strings and integers cannot
  masquerade as `false`;
- generates the DQS1 queue from the latest schema-validated action summary via
  `tools/build_dqs1_local_first_queue.py`;
- requires completed local-advisory skip detection to have `cpu_advisory`
  semantics, `600` samples, a finite score, archive custody, matching archive
  path, and matching SHA-256.

After rank023 and rank024 local observe-only results, the generated queue
routes the next candidate to `pairset_drop_one_rank018_pair0588`.

## Verification

- `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml validate`
- `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml status`
- `.venv/bin/python tools/calibrate_local_cpu_contest_drift.py --calibration-json .omx/research/local_cpu_contest_drift_calibration_dqs1_fec6_20260522T194800Z.json --candidate-id pairset_drop_one_rank024_pair0112 --candidate-local-json experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank024_pair0112/local_cpu_advisory.json --auth-frontier-score 0.19202828295713675 --eureka-out .omx/research/local_cpu_contest_drift_eureka_pairset_drop_one_rank024_pair0112_20260522T205606Z.json --min-margin 0.0 --source-artifact experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank024_pair0112/local_cpu_advisory.json`
- `.venv/bin/ruff check src/tac/canonical_equations/pairset_component_marginal.py src/tac/optimization/pairset_component_marginal.py src/tac/canonical_equations/tests/test_pairset_component_marginal.py`
- `.venv/bin/python -m pytest src/tac/canonical_equations/tests/test_pairset_component_marginal.py src/tac/canonical_equations/tests/test_canonical_equations_initial_population.py src/tac/tests/test_cross_family_candidate_portfolio.py -q`
- `.venv/bin/ruff check src/tac/tests/test_frontier_state_docs.py`
- `.venv/bin/python -m pytest src/tac/tests/test_frontier_state_docs.py -q`
- `.venv/bin/ruff check src/comma_lab/scheduler/experiment_queue.py src/comma_lab/scheduler/dqs1_local_first_queue.py tools/experiment_queue.py tools/build_dqs1_local_first_queue.py src/tac/tests/test_experiment_queue.py src/tac/tests/test_dqs1_local_first_queue_builder.py`
- `.venv/bin/python -m pytest src/tac/tests/test_experiment_queue.py src/tac/tests/test_dqs1_local_first_queue_builder.py -q`
- `.venv/bin/python tools/build_dqs1_local_first_queue.py --action-summary latest --write`
- `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml retire-orphans --reason rerouted_to_rank018_after_rank024_observe_only`

## Next Action

Do not promote rank024. Continue with rank018 through the canonical local-first
queue after committing the generator/orphan-state guard.
