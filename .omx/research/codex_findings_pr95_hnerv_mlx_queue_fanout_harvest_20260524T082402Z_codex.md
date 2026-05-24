# Codex Findings: PR95/HNeRV MLX Queue Fan-Out And Harvest

UTC: 2026-05-24T08:24:02Z

## Artifact

Run root:
`experiments/results/pr95_hnerv_mlx_queue_smoke_20260524T082140Z`

The run used the canonical queue path:

1. `tools/run_pr95_mlx_timing_smoke.py --plan-only` for stages 1, 5, and 8.
2. `tools/build_local_training_execution_queue.py` to compile
   `experiment_queue.v1`.
3. `tools/experiment_queue.py --queue ... run-worker --execute` with
   `local_mlx` concurrency 3.
4. `tools/build_optimizer_candidate_queue.py` over the harvested
   `representation_training_manifest.json` sidecars.

## Observed Local MLX Timing

| Stage | Module | Seconds/step | Smoke archive bytes |
|---:|---|---:|---:|
| 1 | `stage1_v328_ce` | `0.3005089579964988` | `2908` |
| 5 | `stage5_c1a_l7` | `0.891868082981091` | `2906` |
| 8 | `stage8_muon_finetune` | `3.4647309170104563` | `2922` |

Worker result:

- `steps_started=3`
- `success_count=3`
- `failure_count=0`
- `claim_refused_count=0`
- failed postconditions: none

## Queue/Optimizer Integration Patch

The live harvest exposed that runtime-only representation-training rows entered
the optimizer queue with no `rank_score`. I patched both generic and PR95-specific
training adapters to use the best runtime timing as a planning rank when no
quality/auth score is available:

- `rank_score=<seconds>`
- `rank_score_field=<timing_field>_cost_signal_not_score`

The rebuilt optimizer queue now sorts the PR95 MLX Stage 1/5/8 rows by measured
seconds/step while keeping `dispatch_ready_count=0`.

## Authority Boundary

The artifacts remain `[macOS-MLX research-signal]` cost/throughput evidence only.
They are not quality, score, promotion, rank/kill, or exact-dispatch authority.
Every row still carries the exact-readiness blockers for runtime consumption
proof, receiver proof, PyTorch export parity, byte-closed contest archive export,
and exact CPU/CUDA auth eval.

## Verification

- `.venv/bin/ruff check src/tac/optimization/representation_training_probe_integration.py src/tac/optimization/pr95_muon_local_training_integration.py src/tac/tests/test_representation_training_probe_integration.py src/tac/tests/test_pr95_muon_local_training_integration.py`
- `.venv/bin/python -m pytest src/tac/tests/test_representation_training_probe_integration.py src/tac/tests/test_pr95_muon_local_training_integration.py src/tac/tests/test_optimizer_candidate_queue.py -q`

Observed result: 49 tests passed.

## Remaining Gap

The next engineering step is source-checkpoint PyTorch-to-MLX forward parity.
Once that is anchored, the same queue path can run non-synthetic PR95/HNeRV
local MLX training windows and feed higher-EV stage/optimizer candidates into
the inverse-scorer acquisition pipeline.
