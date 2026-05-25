<!-- SPDX-License-Identifier: MIT -->
---
schema: codex_findings_v1
topic: hinton_distilled_scorer_surrogate_mlx_long_training_validation
created_at_utc: 2026-05-25T21:55:00Z
author: codex
lane_id: lane_hinton_distilled_scorer_surrogate_mlx_long_training_validation_20260525
score_claim: false
promotion_eligible: false
rank_or_kill_eligible: false
ready_for_exact_eval_dispatch: false
evidence_grade: macOS-MLX-research-signal
evidence_tag: "[macOS-MLX research-signal]"
---

# Hinton-distilled scorer surrogate MLX validation

## Finding

The initial Hinton MLX smoke surface was not safe to absorb as a dispatch gate.
The KL branch was self-referential: student logits and stopped teacher logits
were both computed from the decoded frame through the same provider, so the
distillation term collapsed toward self-KL. Its artifact also used
`paid_dispatch_authorization_signal=GATE_OPEN` while carrying mock-teacher and
CPU/CUDA authority blockers.

## Landing

This correction makes the signal non-vacuous and queue-visible:

- `make_hinton_custom_loss_fn` now computes student logits from decoded frames
  and stopped teacher logits from target frames.
- A regression proves that enabling the distillation weight increases the loss
  over MSE-only on mismatched decoded/target frames.
- `tools/run_hinton_mlx_long_training_smoke.py` now emits
  `LOCAL_MLX_QUEUE_READY` for local queue follow-up on convergence, while paid
  dispatch remains blocked by contest-teacher and paired CPU/CUDA requirements.
- `hinton_mlx_long_training_smoke_verdict.v1` is accepted by the local training
  queue and surfaced by the experiment queue observer.
- The generated plan compiles to
  `.omx/research/hinton_mlx_long_training_smoke_20260525/local_training_queue.json`.
- The stale `GATE_OPEN` probe outcome was superseded through the canonical
  probe-outcomes ledger, and a corrected non-vacuous queue-visible row was
  appended.

## Empirical Receipt

Corrected 100-epoch macOS-MLX smoke:

- initial loss: `0.1863129436969757`
- final loss: `0.002592975040897727`
- loss reduction: `98.60826897506648%`
- convergence verdict: `CONVERGES_CONSISTENTLY`
- local queue signal: `LOCAL_MLX_QUEUE_READY`
- paid dispatch signal:
  `PAID_DISPATCH_BLOCKED_REQUIRES_CONTEST_TEACHER_AND_CPU_CUDA_AUTH_EVAL`

Artifact:
`.omx/research/hinton_mlx_long_training_smoke_20260525/executed_smoke_100ep_verdict.json`

## Authority Boundary

This remains `[macOS-MLX research-signal]` only. It does not claim a score,
promote, rank/kill, or authorize paid dispatch. The next useful step is a
queue-owned contest-teacher replacement or wider local proof, followed by
paired contest CPU/CUDA authority before any score movement claim.

## Verification

- `.venv/bin/python -m ruff check src/tac/local_acceleration/mlx_production_contract.py src/tac/optimization/scorer_response_dataset.py src/comma_lab/scheduler/local_training_queue.py src/comma_lab/scheduler/experiment_queue_observer.py src/tac/substrates/hinton_distilled_scorer_surrogate tools/run_hinton_mlx_long_training_smoke.py src/tac/tests/test_mlx_production_contract.py src/tac/tests/test_scorer_response_dataset.py src/tac/tests/test_local_training_execution_queue.py src/tac/tests/test_experiment_queue_observer.py`
- `.venv/bin/python -m pytest src/tac/tests/test_mlx_production_contract.py src/tac/tests/test_scorer_response_dataset.py src/tac/substrates/hinton_distilled_scorer_surrogate/tests src/tac/tests/test_local_training_execution_queue.py src/tac/tests/test_experiment_queue_observer.py -q`
  - `221 passed`
- `.venv/bin/python tools/run_hinton_mlx_long_training_smoke.py --output-report .omx/research/hinton_mlx_long_training_smoke_20260525/executed_smoke_100ep_verdict.json --telemetry-path .omx/research/hinton_mlx_long_training_smoke_20260525/executed_smoke_100ep_telemetry.jsonl --max-frames 4 --smoke-epochs 100 --checkpoint-every-epochs 100 --hash-source-video --execute-smoke`
- `.venv/bin/python tools/build_local_training_execution_queue.py --plan .omx/research/hinton_mlx_long_training_smoke_20260525/plan_only_report.json --output .omx/research/hinton_mlx_long_training_smoke_20260525/local_training_queue.json --queue-id hinton_mlx_long_training_smoke_20260525 --repo-root . --local-mlx-concurrency 1`
