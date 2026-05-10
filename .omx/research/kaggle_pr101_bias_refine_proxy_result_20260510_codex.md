# Kaggle PR101 Bias-Refine Proxy Result (2026-05-10)

## Scope

This ledger records the `pr101_bias_refine` Kaggle proxy run launched after
commit `d88eab54`. It is proxy/search evidence only. It is not contest CPU,
not contest CUDA, not a score claim, and not promotion evidence.

## Custody

- Kernel: `adpena/pr101-bias-refine`
- Lane claim: `kaggle_pr101_bias_refine`
- Status: completed proxy run
- Harvested output root:
  `experiments/results/kaggle_pr101_bias_refine_20260510_codex/pr101_bias_refine/`
- Best candidate:
  `experiments/results/kaggle_pr101_bias_refine_20260510_codex/pr101_bias_refine/best_proxy_candidate.json`
- Runtime packet:
  `experiments/results/kaggle_pr101_bias_refine_20260510_codex/pr101_bias_refine/proxy_runtime_packet/runtime_packet_manifest.json`
- Runtime-consumption proof:
  `experiments/results/kaggle_pr101_bias_refine_20260510_codex/pr101_bias_refine/proxy_runtime_packet/runtime_consumption_proof.json`
- Exact-ready local queue:
  `experiments/results/optimizer_candidate_queue_20260510_codex/pr101_bias_refine_exact_ready_queue.json`
- Exact-ready report:
  `experiments/results/optimizer_candidate_queue_20260510_codex/pr101_bias_refine_exact_ready_report.json`

## Best Proxy Candidate

- Candidate ID: `bias_refine_cmaes_0050`
- Profile: `pr101_bias_refine`
- Param schema: `pr101_kaggle_proxy_bias_runtime_params_v1`
- Proxy objective: `0.19285462481263735`
- Runtime-consumed params:
  - `bias_b = -1.0027525485325404`
  - `bias_g = -0.9922764812932092`
  - `bias_r = -1.0055585926234436`

The older six-parameter `pr101_proxy_sweep` best candidate had proxy objective
`0.19287550335547282`; this narrowed bias-only run improved the proxy objective
while removing non-runtime-consumed search dimensions. This is not an exact
score improvement until the byte-closed packet is evaluated through the
canonical contest auth-eval path.

## Adversarial Classification

- `score_claim=false`
- `ready_for_exact_eval_dispatch=false` in proxy/runtime packet manifests
- Runtime-consumption proof: present and true for supported bias params
- Promotion blocker checklist verdict: `BLOCKED_PROXY_ONLY_NOT_PROMOTABLE`
- Exact-readiness gate: created a local exact-ready queue row only after proof
  validation
- Required next gate: claimed exact CUDA auth eval on the exact-ready queue row

## Next Action

Dispatch
`experiments/results/optimizer_candidate_queue_20260510_codex/pr101_bias_refine_exact_ready_queue.json`
only after a fresh Level-2 lane claim. Lightning is not assumed available.
Use Modal/Vast or another CUDA provider with the canonical auth-eval path; do
not use Kaggle or MPS as auth-eval evidence.
