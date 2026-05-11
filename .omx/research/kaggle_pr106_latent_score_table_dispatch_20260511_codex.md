# Kaggle PR106 latent score-table dispatch (2026-05-11)

Purpose: use free Kaggle GPU capacity for the PR106 latent sidecar score-table
search while preserving the proxy/evidence boundary. This is score-lowering
work because a positive table can materialize a byte-closed sidecar archive, but
Kaggle itself is not a contest score authority.

## Dispatch

- lane_id: `lane_pr106_latent_sidecar`
- instance_job_id: `kaggle_pr106_latent_score_table`
- platform: `kaggle`
- kernel: `adpena/comma-lab-pr106-latent-score-table`
- source dataset: `adpena/comma-lab-pr106-latent-source`
- pushed_at_utc: `2026-05-11T14:09:41Z`
- observed_status_after_push: `KernelWorkerStatus.RUNNING`
- score_claim: `false`
- promotion_eligible: `false`
- rank_or_kill_eligible: `false`

## Commands

```bash
.venv/bin/python tools/claim_lane_dispatch.py claim \
  --lane-id lane_pr106_latent_sidecar \
  --platform kaggle \
  --instance-job-id kaggle_pr106_latent_score_table \
  --agent codex:gpt-5.5 \
  --predicted-eta-utc 2026-05-11T17:15:00Z \
  --status active_dispatching \
  --notes 'PR106 latent score-table private Kaggle CUDA producer; score_claim=false; Kaggle diagnostic/proxy only pending contest-CUDA adjudication'

.venv/bin/python tools/kaggle_build_pr106_latent_score_table.py --write-source-bundle
uv run --with kaggle kaggle datasets create -p experiments/kaggle_datasets/comma-lab-pr106-latent-source
uv run --with kaggle kaggle kernels push -p experiments/kaggle_kernels/comma-lab-pr106-latent-score-table
uv run --with kaggle kaggle kernels status adpena/comma-lab-pr106-latent-score-table
```

## Boundary

This run may produce a useful latent candidate table and a diagnostic
CUDA/provider result, but it cannot promote, rank, kill, or claim contest score.
Promotion requires:

1. ingest Kaggle output with the canonical paginated ingester;
2. verify source/runtime/archive custody and score table manifest hashes;
3. materialize a byte-closed archive whose inflate path consumes the sidecar;
4. run no-op proof and exact readiness gates;
5. dispatch separate contest-CUDA auth eval under a fresh lane claim.

Do not compare this result directly to `[contest-CUDA]`, `[contest-CPU]`, or
macOS CPU anchors. It is a provider/search artifact until exact CUDA replay
adjudicates the materialized packet.
