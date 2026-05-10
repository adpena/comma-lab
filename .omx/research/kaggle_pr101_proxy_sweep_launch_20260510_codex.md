# Kaggle PR101 proxy sweep launch — Codex ledger

Date: 2026-05-10

## Launch

- Lane: `kaggle_pr101_proxy_sweep`
- Platform: Kaggle
- Kernel: `adpena/pr101-proxy-sweep`
- URL: `https://www.kaggle.com/code/adpena/pr101-proxy-sweep`
- Kernel version: `1`
- Claim status: `active_proxy_dispatch`
- Agent: `codex:kaggle_proxy_readiness`
- Pushed command:

```bash
uv run --with kaggle kaggle kernels push -p experiments/kaggle_kernels/pr101_proxy_sweep
```

Immediate status after push:

```text
adpena/pr101-proxy-sweep has status "KernelWorkerStatus.RUNNING"
```

## Evidence boundary

This is a proxy/config-search substrate only:

- `score_claim=false`
- `score_claim_valid=false`
- `proxy_only=true`
- `ready_for_exact_eval_dispatch=false`
- `contest_cuda_auth_eval=false`
- `archive_zip_emitted=false`
- `inflate_runtime_emitted=false`

No score movement, promotion, rank, kill, or retirement may be inferred from
the Kaggle result. A useful `best_proxy_candidate.json` must be promoted through
a separate archive builder or training dispatch, then exact contest-CUDA eval
with a fresh lane claim.

## Harvest

Check status:

```bash
uv run --with kaggle kaggle kernels status adpena/pr101-proxy-sweep
```

When terminal, close the active proxy claim with the exact outcome. Template:

```bash
.venv/bin/python tools/claim_lane_dispatch.py claim \
  --lane-id kaggle_pr101_proxy_sweep \
  --platform kaggle \
  --instance-job-id kaggle:adpena/pr101-proxy-sweep \
  --agent codex:kaggle_proxy_readiness \
  --status completed_proxy_or_failed_proxy_SET_EXACT_STATUS \
  --notes 'Set exact Kaggle terminal status and artifact path; still score_claim=false' \
  --force
```
