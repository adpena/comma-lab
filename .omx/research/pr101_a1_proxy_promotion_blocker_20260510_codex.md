# PR101/A1 proxy promotion blocker - 2026-05-10

Generated: `2026-05-10T13:39:24Z`

`research_only=true`; `score_claim=false`; `dispatch_attempted=false`;
`remote_gpu_jobs_run=false`.

## Scope

Operator scope was PR101/A1 exact-runtime score-lowering opportunities while T1
runs. I inspected the PR101 Kaggle proxy bridge, the local runtime-consumption
proof, the inflate op-cost xray, and canonical A1 exact-eval artifacts. I did
not touch T1 Modal files or preflight files.

## Evidence Inspected

- PR101 proxy bridge:
  `tools/materialize_kaggle_pr101_proxy_candidate.py`,
  `tools/build_pr101_kaggle_proxy_runtime_packet.py`
- PR101 runtime-consumption proof:
  `tools/prove_pr101_kaggle_proxy_runtime_consumption.py`
- Existing proxy packet:
  `experiments/results/kaggle_pr101_proxy_sweep_20260510_codex/pr101_proxy_sweep/proxy_runtime_packet/`
- Inflate op-cost xray:
  `experiments/results/xray_inflate_op_cost_profiler_20260509T104122Z/op_catalog.json`
- Canonical A1 exact CUDA anchor:
  `experiments/results/track1_phase_a1_score_gradient_bestproxy_lr2e6_20260509_codex/harvested_artifacts/eval_work/contest_auth_eval.json`

## Current Checklist Verdict

New checker:

```bash
.venv/bin/python tools/check_pr101_proxy_promotion_blocker.py
```

Current output exits nonzero and reports:

```json
{
  "verdict": "BLOCKED_PROXY_ONLY_NOT_PROMOTABLE",
  "promotable": false,
  "candidate_id": "proxy_cmaes_0037",
  "blockers": [
    "full_runtime_consumption_not_proven",
    "no_candidate_contest_cuda_auth_eval",
    "stale_unsupported_proxy_contract"
  ]
}
```

Interpretation:

- The xray/op-cost tool correctly finds the three PR101 per-channel bias
  mutation sites, so the proxy idea is pointed at a real consumed runtime
  surface.
- The existing packet is stale against the current bias-only contract: its
  manifest/proof still carry removed unsupported-param blockers for
  `delta_scale`, `latent_delta_scale`, and `smooth_weight`.
- Even after regenerating under the current contract, the proxy packet would
  still be blocked until full runtime execution and exact contest-CUDA auth eval
  land on the byte-closed archive/runtime packet.
- The canonical A1 anchor is valid exact CUDA evidence, but it does not promote
  this proxy packet. It is only the baseline/control anchor.

## A1 Anchor

Canonical A1 evidence path:

`experiments/results/track1_phase_a1_score_gradient_bestproxy_lr2e6_20260509_codex/harvested_artifacts/eval_work/contest_auth_eval.json`

Key fields:

- `[contest-CUDA]`, `evidence_grade=A++`, `score_axis=contest_cuda`
- `score_recomputed_from_components=0.2263520234784395`
- `archive_size_bytes=178262`
- `archive_sha256=87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5`
- `n_samples=600`

## Wire-In Declaration

- Sensitivity-map contribution: N/A; no new empirical anchor or saliency map.
- Pareto constraint: N/A; current verdict is a blocker, not a candidate row.
- Bit-allocator hook: N/A; no per-tensor allocation change.
- Cathedral autopilot dispatch hook: blocked by the checker returning nonzero
  unless promotion evidence exists.
- Continual-learning posterior update: N/A; no new exact-eval result.
- Probe-disambiguator: N/A; no 2-mode design choice was introduced.

## Verification

```bash
.venv/bin/python -m pytest -q \
  tests/test_check_pr101_proxy_promotion_blocker.py \
  tests/test_build_pr101_kaggle_proxy_runtime_packet.py \
  tests/test_prove_pr101_kaggle_proxy_runtime_consumption.py
# 18 passed

.venv/bin/python -m py_compile tools/check_pr101_proxy_promotion_blocker.py
```

## Next Exact-Eval Candidate Or Blocker

No exact-eval candidate should be dispatched from the current PR101 proxy
packet. The active blocker is now explicit and machine-checkable:

1. Regenerate the proxy packet only if the operator wants a clean local packet
   artifact under the current bias-only contract.
2. Do not promote or dispatch until a candidate packet has full runtime
   execution proof and exact contest-CUDA auth eval under a fresh claim.
3. Keep the stale existing packet classified as proxy-only / not promotable.
