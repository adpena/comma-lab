# Kaggle Score-Lowering Proxy Runbook

Kaggle is free GPU capacity for search, curves, and warm starts. It is not a
score surface. Every Kaggle artifact must keep:

- `score_claim=false`
- `ready_for_exact_eval_dispatch=false`
- `proxy_only=true`
- no rank, kill, promotion, or contest-CUDA authority

Promising Kaggle outputs become useful only after a separate builder emits a
byte-closed `archive.zip` plus `inflate.sh`, the exact-readiness gate promotes
that packet, a lane claim is recorded, and contest-CUDA auth eval lands.

## Current Kaggle Profiles

Build kernels with the canonical builder:

```bash
.venv/bin/python tools/build_kaggle_proxy_sweep_kernel.py \
  --profile pr101_proxy_sweep \
  --force
```

```bash
.venv/bin/python tools/build_kaggle_proxy_sweep_kernel.py \
  --profile pr101_bias_refine \
  --force
```

`pr101_proxy_sweep` is the legacy six-parameter proxy profile. It is useful for
coarse optimizer behavior, but only `bias_b`, `bias_g`, and `bias_r` are routed
by the PR101 runtime packet builder.

`pr101_bias_refine` is the score-lowering profile to prefer now. It emits only
the three runtime-consumed bias parameters and declares
`param_schema=pr101_kaggle_proxy_bias_runtime_params_v1`, so downstream
materialization cannot silently drop searched parameters.

## Required Launch Sequence

Use the command strings in each generated `proxy_sweep_build_manifest.json`.
The sequence is always:

1. Dry-run the lane claim.
2. Record the active lane claim.
3. Push the Kaggle kernel.
4. Sync status into `.omx/status/`.
5. Download/ingest outputs.
6. Close the claim with a terminal proxy status.

Never push a Kaggle kernel outside an active lane claim, and never leave a
completed kernel with a nonterminal claim.

## Promotion Boundary

For PR101 bias outputs:

```bash
.venv/bin/python tools/materialize_kaggle_pr101_proxy_candidate.py \
  --candidate <downloaded>/best_proxy_candidate.json \
  --output-dir <local_materialization> \
  --force

.venv/bin/python tools/build_pr101_kaggle_proxy_runtime_packet.py \
  --handoff <local_materialization>/archive_builder_handoff.json \
  --packet-dir <proxy_runtime_packet> \
  --force

.venv/bin/python tools/prove_pr101_kaggle_proxy_runtime_consumption.py \
  --manifest <proxy_runtime_packet>/runtime_packet_manifest.json

.venv/bin/python tools/check_pr101_proxy_promotion_blocker.py \
  --manifest <proxy_runtime_packet>/runtime_packet_manifest.json \
  --proof <proxy_runtime_packet>/runtime_consumption_proof.json \
  --allow-blocked

.venv/bin/python tools/build_optimizer_candidate_queue.py \
  --source <proxy_runtime_packet>/runtime_packet_manifest.json \
  --output <promotion_dir>/optimizer_queue.json

.venv/bin/python tools/promote_optimizer_candidate_for_exact_eval.py \
  --queue <promotion_dir>/optimizer_queue.json \
  --candidate-id <candidate_id>_pr101_proxy_runtime_packet \
  --output <promotion_dir>/exact_ready_queue.json \
  --report-output <promotion_dir>/exact_ready_report.json
```

The runtime packet manifest remains conservative and blocked because it is
proxy-derived. The exact-readiness report is the only local gate allowed to say
that archive/runtime custody is ready for claimed exact eval. That is still not
a score claim.

## Adversarial Priority

1. Exact-eval already byte-closed Kaggle-derived packets on a real CUDA
   provider when capacity exists.
2. Use Kaggle for `pr101_bias_refine` if exact CUDA is blocked and free GPU
   capacity would otherwise sit idle.
3. Do not rerun `pr101_proxy_sweep` without a reason to search non-runtime
   parameters and a builder that consumes them.
4. Do not use Kaggle to kill or promote PR95/T1/HNeRV training; use it for
   warm starts and curves only, then promote through byte-closed exact eval.
