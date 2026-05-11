# PR106 latent sidecar Modal exact-CUDA dispatch — Codex

Date: 2026-05-11

## Dispatch

Dispatched the materialized PR106 latent-sidecar archive from the Kaggle score
table into Modal T4 exact CUDA auth eval.

- Lane: `lane_pr106_latent_sidecar`
- Instance/job id: `pr106_latent_sidecar_modal_exact_cuda_20260511T150517Z`
- Modal call id: `fc-01KRBS58Z6EW2A7A3FXJXB3N6S`
- Modal app run: `https://modal.com/apps/adpena/main/ap-7rGIsZsXyNsnpQPXQYVtl0`
- Modal call URL: `https://modal.com/id/fc-01KRBS58Z6EW2A7A3FXJXB3N6S`
- Archive:
  `experiments/results/pr106_latent_sidecar_from_kaggle_table_20260511_codex/sidecar_archive.zip`
- Archive bytes: `186808`
- Archive SHA-256:
  `947b85e8a69db295d4dcf80b0b528639c47839f40f289a2c05b70a2064658b48`
- Runtime: `submissions/pr106_latent_sidecar`
- Inflate wrapper: `inflate.sh`
- Scorer device: `cuda`
- GPU: Modal T4
- Output dir:
  `experiments/results/modal_auth_eval/pr106_latent_sidecar_20260511T150517Z`

Command:

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/modal run --detach experiments/modal_auth_eval.py \
  --archive experiments/results/pr106_latent_sidecar_from_kaggle_table_20260511_codex/sidecar_archive.zip \
  --output-dir experiments/results/modal_auth_eval/pr106_latent_sidecar_20260511T150517Z \
  --submission-dir submissions/pr106_latent_sidecar \
  --inflate-sh inflate.sh \
  --gpu T4 \
  --scorer-device cuda \
  --inflate-timeout 1800 \
  --evaluate-timeout 1800 \
  --lane-id lane_pr106_latent_sidecar \
  --instance-job-id pr106_latent_sidecar_modal_exact_cuda_20260511T150517Z \
  --claim-agent codex:gpt-5.5 \
  --claim-notes 'PR106 latent sidecar exact CUDA from Kaggle score table; archive_sha256=947b85e8a69db295d4dcf80b0b528639c47839f40f289a2c05b70a2064658b48; bytes=186808; score_claim=false until recovered/adjudicated' \
  --detach \
  --provider-detach-ack
```

## Current status

Initial recover returned `status=pending` at `2026-05-11T15:06:27Z`.

```bash
.venv/bin/python tools/recover_modal_auth_eval.py \
  --output-dir experiments/results/modal_auth_eval/pr106_latent_sidecar_20260511T150517Z
```

This is not score evidence yet.

- `score_claim=false`
- `promotion_eligible=false`
- no component distances harvested yet
- no rank/kill/lane-status decision until recovery produces exact CUDA auth
  eval JSON and adversarial adjudication

## Pre-dispatch hardening

Commit `1928a7b4` hardened `submissions/pr106_latent_sidecar/inflate.sh` so the
runtime works in uploaded self-contained Modal submission trees instead of
requiring repo-root `submissions.<name>` imports. Verification before dispatch:

- `bash -n submissions/pr106_latent_sidecar/inflate.sh`
- `.venv/bin/python -m pytest src/tac/tests/test_pr106_latent_sidecar.py -q`
  passed `20 passed`
- `.venv/bin/python -m ruff check src/tac/tests/test_pr106_latent_sidecar.py`
- `.venv/bin/python tools/all_lanes_preflight.py --timings --timeout-s 30`
  passed all 29 checks in `2.28s`

## Recovery and classification rule

On terminal recovery, classify as one of:

- `completed_contest_cuda_positive` if exact CUDA auth eval reports zero
  schema blockers and improves the active CUDA floor after formula
  recomputation;
- `completed_contest_cuda_regression` if exact CUDA auth eval is valid but
  worse than the active CUDA floor;
- `failed_modal_auth_eval_*` if inflate/eval/runtime/dependency/custody fails;
- `indeterminate_no_score_claim` if artifacts are incomplete or schema blockers
  remain.

Do not compare the Kaggle score table, macOS CPU, or Linux CPU numbers directly
to this Modal CUDA result. The score table is a builder input only; the
promotable axis is the recovered
`archive.zip -> inflate.sh -> upstream/evaluate.py --device cuda` artifact.
