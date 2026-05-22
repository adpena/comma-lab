# Codex Session Summary: Local-First Pairset Scale-Up And MLX Boundary Hardening

UTC: 2026-05-22T21:11:58Z
Agent: Codex
Authority: no score claim; no promotion; no rank/kill; local CPU/MLX signals remain advisory.

## Landed Work

- DQS1 local-first queue generation now selects the next unobserved
  false-authority pairset candidate from the canonical component-marginal action
  summary and routes `pairset_drop_one_rank018_pair0588`.
- Queue execution is hardened against duplicate noncanonical SQLite state and
  stale reroute rows. Execution with a noncanonical state path or blocking
  orphaned rows now requires an explicit non-placeholder audit rationale, and
  `retire-orphans` records stale rows as skipped instead of silently ignoring
  them.
- Queue postconditions now use `json_false_authority` so local plan,
  materialization, locality, and advisory artifacts cannot pass while carrying
  score, promotion, rank/kill, or dispatch authority.
- The queue builder now accepts general `pairset_*` candidates, including group
  rows such as `pairset_drop_two_*`, instead of only singleton `drop_one` IDs.
  This makes pair groups executable through the same local-first fail-closed
  path.
- MLX score calibration now validates auth-axis payloads through exact-eval
  custody: runtime tree, hardware/devices, auth command, log/artifact path and
  SHA, inflated-output manifest, raw aggregate SHA, archive hash, and score
  formula closure.
- MLX spend triage `--min-observed-gain` can only raise the calibrated minimum
  gain. Attempts to lower the safety gap fail closed.
- Current focus mirrors now point at the canonical rank021/pair0371 CPU
  frontier instead of the superseded compact top32 CPU anchor, with regression
  coverage against `.omx/state/canonical_frontier_pointer.json`.
- Rank027/rank031 exact CPU results were backfilled into result-review packets
  and autopilot evidence so component-negative evidence is consumable by the
  planner.

## New Observation

`pairset_drop_one_rank024_pair0112` completed local
plan/materialize/locality/advisory. Local macOS CPU advisory score was
`0.19203828295713676` with `score_claim=false`. Drift calibration projected a
point estimate tied with the current CPU frontier, but the conservative
projection was worse by the guard band, so the recommended action is
`observe_only`.

## Scale-Up Methodology

The pair/group/null-remove loop should be one integrated acquisition funnel:

1. Generate candidates from canonical pairset acquisition, master-gradient,
   X-ray, waterbucket, and null/remove exploit surfaces.
2. Normalize every candidate into a false-authority portfolio row with
   operation metadata (`drop_one`, `drop_two`, prefix, diversity, swap-in,
   null/remove, byte-recode, or future group operation).
3. Execute cheap local materialization, locality controls, and CPU/MLX advisory
   probes through the queue substrate.
4. Convert local observations into component-marginal, calibration, and
   response-surface training rows.
5. Trigger exact CPU/CUDA spend only when custody, calibration, and eureka
   gates pass.
6. Feed exact outcomes back into the portfolio/action summary and regenerate
   the queue.

The important boundary is that local CPU and MLX may select follow-up work, but
they do not claim score, promote, rank/kill, or skip exact auth eval.

## Verification

- `.venv/bin/python -m py_compile src/comma_lab/scheduler/dqs1_local_first_queue.py tools/build_dqs1_local_first_queue.py src/comma_lab/scheduler/experiment_queue.py tools/experiment_queue.py src/tac/local_acceleration/mlx_score_calibration.py src/tac/optimization/mlx_effective_spend_triage_selection.py tools/select_mlx_effective_spend_triage_candidates.py`
- `.venv/bin/ruff check ...` on edited queue, MLX, frontier-doc, and component-marginal files.
- `.venv/bin/python -m pytest src/tac/tests/test_experiment_queue.py src/tac/tests/test_dqs1_local_first_queue_builder.py src/tac/tests/test_frontier_state_docs.py src/tac/canonical_equations/tests/test_pairset_component_marginal.py src/tac/canonical_equations/tests/test_canonical_equations_initial_population.py src/tac/tests/test_cross_family_candidate_portfolio.py src/tac/tests/test_mlx_score_calibration.py src/tac/tests/test_mlx_effective_spend_triage_selection.py -q`
- `.venv/bin/python tools/build_dqs1_local_first_queue.py --action-summary latest --write --output configs/experiment_queues/dqs1_pairset_local_first.yaml`
- `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml validate`
- `.venv/bin/python tools/lane_maturity.py validate`
- `git diff --check`

## Next Tranche

- Execute rank018 local-first through the canonical queue, then calibrate its
  eureka signal and regenerate the queue.
- Extend the local-first queue runner from one active candidate to a bounded
  local cascade that can continuously retire observed candidates, regenerate
  the queue, and execute the next safe pair/group row without false authority.
- Add null/remove exploit candidates as first-class portfolio operations with
  byte-closed materialization and locality/advisory gates.
- Add non-singleton MLX batch-invariance manifests before any grouped MLX row
  can be used for spend triage.
- Backfill Modal CPU recovery through the same exact-custody validator used by
  CUDA and MLX calibration.
- Prototype tiny Mamba-3/Gated-DeltaNet-2 inspired predictor features only as a
  planner-side local smoke; reject runtime inclusion unless the byte budget is
  positive.
