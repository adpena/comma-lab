# Provider expansion score-lowering dispatch matrix (2026-05-10)

Generated at 2026-05-10T07:39:31Z by codex.

## Live dispatch state

- `track1_phase_a1_score_gradient_modal_20260510T0738Z_codex` is active on Modal.
- Modal call id: `fc-01KR8D302GXGKGT49ETYMA0BZC`.
- Local metadata: `experiments/results/track1_phase_a1_score_gradient_modal_20260510T0738Z_codex/modal_metadata.json`.
- Immediate recover result: queued/running; re-run `experiments/modal_phase_a1_score_gradient_pr101.py recover --label track1_phase_a1_score_gradient_modal_20260510T0738Z_codex` within Modal's result-cache TTL.

## Provider status

| Provider | Current status | Score-lowering role | Blocker / next action |
|---|---|---|---|
| Modal | Active A1 job dispatched; CLI present (`modal client version: 1.4.2`). | Primary CUDA path right now for A1 PR101 archive-in-loop and short exact-eval jobs. | Harvest active A1 call; if credits fail, terminal claim row will classify. |
| Kaggle | Auth works via `uv run --with kaggle`; `kaggle kernels list --mine` succeeds; `comma-gpu-lane-smoke` status is `COMPLETE`. | Free GPU sweep/config substrate only; not auth-eval authority. Use for Optuna/CMA-ES proxy curves, HNeRV/T1 warm starts, and throughput probes. | Build a new Kaggle kernel for score-domain training smoke with clear `score_claim=false`; do not use as `[contest-CUDA]`. |
| GCP | `gcloud auth` active for `adpena@gmail.com`; project `personal-mbp-2026`; compute API enabled. | Future CUDA auth-eval/training provider after billing/quota. | Billing is not enabled for the current project; GPU quota commands fail before quota data. Enable billing or switch project, then query T4/L4 GPU quotas. |
| AWS | AWS CLI installed, but SSO/session expired. | Future EC2/GPU fallback after budget/identity gate. | Run `aws login`; then inspect account credits/budgets and G-family quota before any EC2 launch. |
| Azure | `az` installed but not logged in. | Future spot VM fallback after login/quota/budget. | Run `az login`; then use dry-run `scripts/launch_lane_azure.py`; no VM creation until claim and budget gates pass. |
| Lightning | Not viable per operator: likely out of credits; local Batch submit also failed on teamspace resolution. | Keep as dry-run/spec generator only until identity/credit returns. | Do not use for score-lowering dispatch in this tranche. |

## 2026-05-10T07:51Z codex hardening addendum

- Added `tools/cloud_provider_readiness.py`, a read-only provider inventory that emits `score_claim=false` and `ready_for_exact_eval_dispatch=false` for the whole inventory payload.
- Latest inventory artifacts:
  - JSON: `experiments/results/cloud_provider_readiness_20260510_codex.json`
  - Markdown: `.omx/research/cloud_provider_readiness_20260510_codex.md`
- Hardened `scripts/kaggle_check.py`:
  - falls back to `uv run --with kaggle kaggle` when the Kaggle CLI is not installed in `.venv`;
  - treats `KernelWorkerStatus.ERROR` / `KernelWorkerStatus.CANCEL_ACKNOWLEDGED` as failures instead of only matching bare `ERROR`;
  - skips nested output directories and has `--log-timeout-s` so failed-kernel log harvest cannot hang the operator loop.
- Current Kaggle lane status:
  - `adpena/comma-gpu-lane-smoke`: `KernelWorkerStatus.COMPLETE`;
  - `adpena/comma-lab-debug-mount`: `KernelWorkerStatus.COMPLETE`;
  - `adpena/comma-lab-asym-warp-base`: `KernelWorkerStatus.ERROR`;
  - `adpena/comma-lab-asym-warp-raft-only`: `KernelWorkerStatus.ERROR`;
  - `adpena/comma-lab-constrained-gen-smoke`: `KernelWorkerStatus.ERROR`;
  - `adpena/comma-lab-asym-warp-supervised`: not found.

Classification: Kaggle is usable now as a free proxy/config substrate, but the failed legacy kernels must not be used as score-lowering evidence until their logs are harvested and the kernels are rebuilt. Kaggle remains `proxy_only=true`; any useful config must be promoted to an exact CUDA provider before score claims.

## Score-lowering priority queue

1. Harvest active Modal A1. If it returns claimable `[contest-CUDA]`, pair with GHA Linux CPU and only then consider submission/promotion policy.
2. Prepare Kaggle no-score sweep substrate for PR101/T1 hyperparameter search: CMA-ES or Optuna should materialize candidate configs and proxy curves only, with `ready_for_exact_eval_dispatch=false`.
3. Create a real provider-job wrapper for T1. The remote T1 script now supports packet compile plus contest-CUDA auth eval, but a provider job id and copied active claim ledger are still required.
4. Re-enable GCP/AWS/Azure only after identity and billing/quota checks pass; all launches must claim first and emit terminal rows on failure.

## Evidence boundaries

- MPS and Kaggle are proxy/sweep substrates only.
- Modal/GCP/AWS/Azure can become exact CUDA evidence only when the scorer path is `archive.zip -> inflate.sh -> upstream/evaluate.py --device cuda`, `n_samples=600`, archive/runtime SHA custody is present, and `tac.auth_eval_schema.required_contest_cuda_evidence_blockers(...)` returns zero blockers.
- A single contest-CUDA result is not promotion/submission readiness. Promotion remains blocked on paired CPU axis and operator submission policy.
