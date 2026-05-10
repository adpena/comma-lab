# Provider free-tier setup TODO for score-lowering capacity

Date: 2026-05-10

Purpose: add AWS, Azure, GCP, Kaggle, and other free-tier/provider capacity to
the durable TODO surface without weakening the provider contract in
`AGENTS.md`. This is capacity setup only until each provider has a real
claim-before-launch, runtime-closure, harvest, and adjudication lifecycle.

## TODO

1. AWS: create/verify account credits, IAM role boundary, spot quota, GPU quota,
   S3 artifact bucket, budget alarm, and a provider adapter under
   `src/tac/deploy/aws/`.
2. Azure: create/verify free-credit subscription, quota request path for T4/A10
   class GPUs, storage account, budget alert, and `src/tac/deploy/azure/`
   lifecycle parity with locked active-VM state.
3. GCP: create/verify free-credit project, GPU quota path, service-account
   boundary, GCS artifact bucket, budget alert, and
   `src/tac/deploy/gcp/` scaffold with `exact_cuda_eval_supported=false`
   until proven.
4. Kaggle: keep using it only as proxy/search capacity unless a byte-closed
   exact archive/runtime packet is exported and rerun through a claimed exact
   CUDA path. Maintain wheel/dataset freshness checks before kernel push.
5. Modal: continue active T1 harvest/custody, but treat Modal CPU/MPS/proxy
   signals as non-promotional. Modal exact-CUDA support remains only for
   wrappers with full runtime closure and adjudicated exact-eval output.
6. Provider-neutral: add conformance tests for each new provider contract:
   plan-only default, explicit execute flag, lane claim before job creation,
   terminal claim on all outcomes, mounted-code/shipped-tarball manifest,
   artifact harvest, no public secrets/URLs, no score claim from proxy rows.

## Score-lowering priority

Provider setup is instrumental, not the objective. The objective remains
sub-0.17 and then sub-0.15 exact score. Providers should first unblock:

- T1 Balle end-to-end harvest/continuation if Modal cannot schedule.
- A1/A3 exact CUDA checks for byte-closed candidates.
- Phase 1/Phase 2 dispatches only after the local exact-readiness promoter,
  lane claim, runtime closure, and pre-submission custody gates are green.
