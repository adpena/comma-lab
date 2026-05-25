<!-- SPDX-License-Identifier: MIT -->
---
schema: codex_findings_v1
topic: hinton_mlx_queue_signal_observation
created_at_utc: 2026-05-25T22:00:26Z
author: codex
lane_id: lane_codex_hinton_mlx_queue_signal_observation_20260525
score_claim: false
promotion_eligible: false
rank_or_kill_eligible: false
ready_for_exact_eval_dispatch: false
dispatch_attempted: false
research_only: false
---

# Hinton MLX Queue Signal Observation

## Finding

The Hinton MLX smoke verdict was visible in queue observation while the step was
running, but a succeeded verdict could fall out of downstream planning because
the observer only preserved succeeded materializer/DQS1 artifacts and the local
training harvester only accepted `representation_training_manifest.json`
candidate manifests.

That made `LOCAL_MLX_QUEUE_READY` a real but orphan-prone signal: useful for the
operator, not yet durable queue intelligence.

## Landing

This patch adds `local_training_signal_observation.v1` for succeeded
false-authority local-training verdict artifacts. The first supported source is
`hinton_mlx_long_training_smoke_verdict.v1`.

The observer now surfaces:

- `succeeded_signal_steps`
- `local_training_signal_observations`
- `local_training_signal_observation_count`

The local-training harvester now preserves Hinton queue-ready verdicts as
non-candidate signal observations when no representation manifest exists. It
returns a zero-candidate optimizer queue with the signal attached instead of
raising or dropping the result.

## Authority Boundary

The signal observation is planning-only. It preserves convergence verdict,
readiness blockers, and the next local/teacher action, but keeps:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`

The recommended next action remains to build a contest-teacher or strict
surrogate queue before any paid dispatch or exact-eval authority.

## Verification

Focused regressions cover both paths:

- observer: a running Hinton verdict remains visible, then a succeeded Hinton
  verdict becomes a `local_training_signal_observation.v1`;
- harvester: a succeeded Hinton-only local-training queue returns zero optimizer
  candidates and one preserved queue-ready signal observation.

This closes the succeeded-state orphan without converting local MLX evidence
into score, promotion, rank/kill, or dispatch authority.
