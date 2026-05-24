# Codex Findings: PR95 MLX Harvest Intelligence Wiring

UTC: 2026-05-24T09:32:51Z
Agent: Codex
Lane: `codex_pr95_mlx_optimizer_matrix_queue_20260524`

## Scope

Continuation of the PR95/HNeRV MLX reproduction and automation lane. The goal
was to make completed local MLX queue results feed reusable solver surfaces
instead of stopping at a harvested candidate queue.

## Landed Changes

1. Added `local_training_optimizer_harvest_intelligence.v1` in
   `src/tac/optimization/local_training_harvest_intelligence.py`.
   It converts harvested `optimizer_candidate_queue_v1` rows into:
   - neutral optimizer atom ledgers via the existing optimizer atom adapter;
   - `optimizer_scheduler_telemetry.v1` rows from embedded runtime profiles.

2. Extended `optimizer_scheduler_telemetry.v1` to represent
   `seconds_per_step`, which is the natural timing metric for one-step PR95 MLX
   timing smokes.

3. Wired intelligence output into operator flows:
   - `tools/harvest_local_training_optimizer_candidates.py --intelligence-output`
   - `tools/materialize_optimizer_signal_atoms.py --scheduler-telemetry-json-out`
   - `tools/materialize_optimizer_signal_atoms.py --intelligence-json-out`

4. Hardened learned-sweep intake so raw `optimizer_candidate_queue_v1` timing
   queues fail closed instead of being silently ignored or accidentally wrapped
   as score input. Explicit `candidates[]` now require either candidate-specific
   exact-calibrated evidence or `mlx_dynamic_learned_sweep_quality_evidence.v1`
   with strict pass gates.

5. The existing optimizer atom path now preserves runtime-cost observations and
   scheduler-cost-prior hints inside atom metadata while keeping predicted score
   impact neutral until exact auth or calibrated posterior evidence exists.

## Live Artifact

The corrected PR95 MLX matrix queue at
`experiments/results/pr95_mlx_optimizer_matrix_queue_20260524T091004Z/` now has
an ignored intelligence sidecar:

- `optimizer_harvest_intelligence.json`
- `schema=local_training_optimizer_harvest_intelligence.v1`
- `atom_count=3`
- `telemetry_record_count=3`
- `telemetry_refusal_count=0`
- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`

## Authority Boundary

This landing deliberately does not create score, promotion, rank/kill, or exact
dispatch authority. The PR95 MLX timing rows are cost and scheduler-placement
signal only. Learned sweep must not consume `rank_score` from timing queues as
quality. Exact score movement still requires byte-closed contest export, runtime
and receiver proof, parity, and contest CPU/CUDA auth eval.

## Verification

- `.venv/bin/python -m ruff check src/tac/optimization/__init__.py src/tac/optimization/local_training_harvest_intelligence.py src/tac/optimization/mlx_dynamic_learned_sweep.py src/tac/optimization/optimizer_scheduler_registry.py src/tac/optimization/optimizer_signal_atoms.py src/tac/tests/test_local_training_optimizer_candidate_harvest.py src/tac/tests/test_mlx_dynamic_learned_sweep.py src/tac/tests/test_optimizer_scheduler_registry.py src/tac/tests/test_optimizer_signal_atoms.py tools/harvest_local_training_optimizer_candidates.py tools/materialize_optimizer_signal_atoms.py tools/plan_mlx_dynamic_learned_sweep.py`
- `.venv/bin/python -m pytest src/tac/tests/test_local_training_optimizer_candidate_harvest.py src/tac/tests/test_optimizer_scheduler_registry.py src/tac/tests/test_optimizer_signal_atoms.py src/tac/tests/test_mlx_dynamic_learned_sweep.py -q`
- `.venv/bin/python tools/harvest_local_training_optimizer_candidates.py --queue experiments/results/pr95_mlx_optimizer_matrix_queue_20260524T091004Z/experiment_queue.json --state .omx/state/experiment_queue_pr95_mlx_optimizer_matrix_20260524T091004Z.sqlite --repo-root . --output experiments/results/pr95_mlx_optimizer_matrix_queue_20260524T091004Z/optimizer_candidate_queue_intelligence_rerun.json --intelligence-output experiments/results/pr95_mlx_optimizer_matrix_queue_20260524T091004Z/optimizer_harvest_intelligence.json`
- `.venv/bin/python tools/materialize_optimizer_signal_atoms.py --candidate-queue experiments/results/pr95_mlx_optimizer_matrix_queue_20260524T091004Z/optimizer_candidate_queue.json --json-out experiments/results/pr95_mlx_optimizer_matrix_queue_20260524T091004Z/optimizer_signal_atoms_from_harvest.json --scheduler-telemetry-json-out experiments/results/pr95_mlx_optimizer_matrix_queue_20260524T091004Z/optimizer_scheduler_telemetry_from_harvest.json --intelligence-json-out experiments/results/pr95_mlx_optimizer_matrix_queue_20260524T091004Z/optimizer_harvest_intelligence_from_materialize_tool.json`

## Next Tasks

1. Add the explicit quality adapter from strict MLX effective-spend-triage
   selections into learned-sweep `candidates[]`.
2. Add `tac_materializer_receiver_runtime_proof_v1` exact-readiness support for
   family-agnostic materializer receiver proofs.
3. Use the PR95 telemetry ledger as a cost prior for longer Stage 1/5/8 MLX
   timing runs and grouped optimizer sweeps.
