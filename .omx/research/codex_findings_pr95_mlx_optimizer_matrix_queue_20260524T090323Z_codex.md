# Codex Findings: PR95 MLX Optimizer Matrix Queue

UTC: 2026-05-24T09:03:23Z

Lane: `codex_pr95_mlx_optimizer_matrix_queue_20260524`

## What Landed

`tools/build_pr95_mlx_optimizer_matrix_queue.py` is now the queue-owned fan-out
entry point for PR95/HNeRV MLX timing probes. It emits one plan-only
`tools/run_pr95_mlx_timing_smoke.py` packet per selected stage, optimizer
descriptor, and seed, then compiles those packets into `experiment_queue.v1`
via the existing local-training queue compiler.

The first operator-ready artifact is:

`experiments/results/pr95_mlx_optimizer_matrix_queue_20260524T090145Z/matrix_manifest.json`

It contains three default local-MLX cells:

- stage 1 + `pr95_stage1_adamw_baseline_mlx`
- stage 5 + `pr95_stage5_adamw_baseline_mlx`
- stage 8 + `pr95_stage8_muon_adamw_mlx`

The generated queue validates with local auto-parallelism of 4:

`experiments/results/pr95_mlx_optimizer_matrix_queue_20260524T090145Z/experiment_queue.json`

The queue path now also supports public-runtime byte closure:

- `--write-pr95-public-archive-export` writes a deterministic PR95-compatible
  `pr95_public_archive.zip` plus `pr95_public_archive_export.json`.
- `--prove-pr95-runtime-consumption` runs the public PR95 `inflate.sh` and
  records `runtime_consumption_proof.json`.
- The local-training queue watches those artifacts as first-class
  postconditions, including `runtime_consumption_proven=true`.

Live end-to-end artifact:

`experiments/results/pr95_mlx_optimizer_matrix_runtime_proof_20260524T091705Z/`

That artifact executed one scheduler-owned local-MLX cell:

- Queue ID: `pr95_mlx_matrix_runtime_proof_20260524T091705Z`
- Candidate: `pr95_hnerv_mlx_stage1_pr95_stage1_adamw_baseline_mlx_seed31_steps1_c4`
- Archive export: `pr95_hnerv_archive_export.v1`
- Archive ZIP bytes: `8537`
- Archive ZIP SHA-256:
  `00267f7e3f9c4d7d49d2a6f57030aeef85b750a94bc5886933cea6f965ed5f22`
- Export summary records `runtime_consumption_proof_present=true`
- Runtime proof: `pr95_hnerv_public_runtime_consumption_proof.v1`
- Runtime consumed raw bytes: `6104016 / 6104016`
- Runtime consumed raw SHA-256:
  `4e4b656d7cdd7b33e784885a7d77de03e6170f6aab03cf99c528c3886d14b222`
- Harvest output:
  `experiments/results/pr95_mlx_optimizer_matrix_runtime_proof_20260524T091705Z/optimizer_candidate_queue.json`
- Harvest dispatch-ready count: `0`

## Guardrails

- Descriptor-only and incompatible optimizer recipes are recorded as refused
  matrix rows and are not queued for execution.
- Each matrix cell carries a stable SHA-256 `matrix_cell_id` over stage,
  descriptor, seed, steps, batch size, synthetic-pair count, base channels, and
  latent dimension.
- Local-training queue compilation now refuses duplicate output manifests and
  duplicate representation sidecars.
- Queue postconditions now bind completed sidecars to expected
  `candidate_id`, `stage_index`, `seed`, optimizer descriptor, optimizer config
  hash, and parameter-group policy identity.
- Queue postconditions can bind PR95 public archive export and runtime proof
  artifacts without treating those artifacts as score authority.
- Local-training harvest now rejects completed sidecars whose identity differs
  from the queue experiment metadata.
- All matrix, plan, queue, and harvest artifacts remain false-authority:
  no score claim, no promotion, no rank/kill, and no exact-dispatch readiness.

## Recovered PR95 Runtime Proof Signal

The interrupted PR95 runtime-consumption proof tool was adopted and hardened:

`tools/prove_pr95_public_archive_runtime_consumption.py`

It proves a PR95-compatible native export is consumed by the actual public PR95
`inflate.sh` runtime on a one-pair packet. This is still a runtime smoke, not
full public-packet inflate parity and not score authority.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_pr95_hnerv_mlx.py src/tac/tests/test_pr95_mlx_optimizer_matrix_queue.py src/tac/tests/test_local_training_execution_queue.py src/tac/tests/test_local_training_optimizer_candidate_harvest.py src/tac/tests/test_run_pr95_mlx_timing_smoke.py src/tac/tests/test_run_pr95_mlx_timing_smoke_plan.py -q` (`38 passed`)
- `.venv/bin/ruff check src/tac/local_acceleration/pr95_hnerv_mlx.py tools/run_pr95_mlx_timing_smoke.py tools/build_pr95_mlx_optimizer_matrix_queue.py src/comma_lab/scheduler/local_training_queue.py src/tac/tests/test_run_pr95_mlx_timing_smoke.py src/tac/tests/test_run_pr95_mlx_timing_smoke_plan.py src/tac/tests/test_pr95_mlx_optimizer_matrix_queue.py`
- `.venv/bin/python tools/experiment_queue.py --queue experiments/results/pr95_mlx_optimizer_matrix_queue_20260524T090145Z/experiment_queue.json validate`
- `.venv/bin/python tools/harvest_local_training_optimizer_candidates.py --queue experiments/results/pr95_mlx_optimizer_matrix_runtime_proof_20260524T091705Z/queue.json --state experiments/results/pr95_mlx_optimizer_matrix_runtime_proof_20260524T091705Z/queue_state.sqlite --repo-root . --output experiments/results/pr95_mlx_optimizer_matrix_runtime_proof_20260524T091705Z/optimizer_candidate_queue.json`

## Next Engineering Step

Promote the family-agnostic materializer branch from registry-visible to
executable by landing a generic receiver/runtime proof schema accepted by exact
readiness, then implement `packet_member_recompress_v1`,
`archive_section_entropy_recode_v1`, and `tensor_factorize_v1` in that order.

For the PR95 reproduction lane, the next source-faithful blockers are full
source-runtime versus rebuilt-runtime inflate parity, PR95-specific MLX to
PyTorch forward parity, GPU MLX drift classification, and connecting MLX-trained
outputs to full contest-video PR95 archive grammar rather than one-pair
synthetic runtime smokes.
