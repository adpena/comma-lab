# Exact eval state refresh - 2026-05-13

## Scope

Refresh exact-eval custody state after the preflight wall-clock tranche. This
is a state ledger, not a new score claim. All numbers below are copied from
byte-closed artifacts or live provider status queries, with CPU and CUDA axes
kept separate.

## Modal provider state

Command:

```bash
.venv/bin/modal app list
PYTHONPATH=src:upstream:$PWD .venv/bin/python tools/harvest_modal_calls.py
```

Observed state:

- Modal app list shows all comma training/auth-eval apps at `Tasks=0`.
- `tools/harvest_modal_calls.py` scanned 65 dispatched lanes.
- Aggregate status counts:
  - `already_harvested`: 57
  - `error_NotFoundError`: 6
  - `expired`: 1
  - `error_ModuleNotFoundError`: 1
- `not_ready_rows`: 0

Classification: no active Modal job remains to harvest or close. Historical
NotFound/expired/error rows are terminal provider-custody states, not model
results.

## lane_g_v3 contest-CPU closure

Artifact:
`experiments/results/gha_cpu_eval/lane_g_v3_retry7_25772267506/contest_cpu_eval-lane_g_v3-25772267506/contest_auth_eval.json`

Workflow:
`https://github.com/adpena/comma-lab/actions/runs/25772267506`

Evidence:

- archive SHA-256:
  `9b20bdfca246d8e32cc19da966c84cdae7e34f6b247161d107ec43cb9ef6870b`
- archive bytes: `694074`
- axis: `[contest-CPU GHA Linux x86_64]`
- canonical score: `1.0375828860270313`
- displayed score: `1.04`
- SegNet avg: `0.00400702`
- PoseNet avg: `0.0030529`
- rate unscaled: `0.018486215481172717`
- samples: `600`

Classification: lane_g_v3 has the requested Linux x86_64 `[contest-CPU]`
closure. This is not a CUDA promotion claim; its CUDA anchor remains the
separate Lane G v3 CUDA evidence chain.

## PR106 latent-sidecar R2 PR101 grammar CPU validation

Artifact copied locally from GHA run 25773960214:
`experiments/results/gha_cpu_eval/lane_pr106_latent_sidecar_r2_pr101_grammar_25773960214/contest_cpu_eval-lane_pr106_latent_sidecar_r2_pr101_grammar-25773960214/contest_auth_eval.json`

Workflow:
`https://github.com/adpena/comma-lab/actions/runs/25773960214`

Evidence:

- archive SHA-256:
  `c48631e11a9bb18d051da9100ca4d5773558a8a81ac38dc8f6f4e8b6119d0383`
- archive bytes: `186780`
- axis: `[contest-CPU GHA Linux x86_64]`
- canonical score: `0.22806551797550428`
- displayed score: `0.23`
- SegNet avg: `0.00063197`
- PoseNet avg: `0.00016402`
- rate unscaled: `0.004974765410566366`
- samples: `600`

Classification: validates the partner PR106/A1 CPU-CUDA axis finding's R2 CPU
cell within the same score neighborhood. It remains a CPU-axis artifact only;
no CUDA inference is made.

## A1 CPU-CUDA axis validation

Artifact:
`experiments/results/a1_dual_cuda_dispatch_20260509T163400Z/dual_eval_adjudicated.json`

Evidence:

- archive SHA-256:
  `87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5`
- archive bytes: `178262`
- `[contest-CPU GHA Linux x86_64]` score: `0.19284757743677347`
- `[contest-CUDA Tesla T4]` score: `0.2263520234784395`
- CUDA minus CPU: `+0.033504446041666`
- CUDA/CPU pose ratio: `5.20420572122337`
- CUDA/CPU seg ratio: `1.18342283709908`
- rate unchanged: `true`

Classification: A1 is a strong CPU/public-axis anchor but is not a
CUDA-internal frontier improvement. The CPU-CUDA gap is a measured
per-archive/per-runtime property, not a universal conversion rule.

## Immediate Score-Lowering Implication

The exact-eval short-term state is now clearer:

1. There are no active Modal jobs blocking harvest/custody.
2. lane_g_v3 has its requested `[contest-CPU]` closure.
3. A1's dual-axis evidence is complete and argues against more blind
   inflate-time bias sweeps for CUDA-frontier work.
4. PR106/R2 sidecar work remains interesting as a CPU/CUDA mechanism probe, but
   byte-format changes must not become score claims until runtime consumption
   and exact paired-axis eval are both present.

The highest-value next exact-evaluable work remains: claimed CUDA/CPU paired
evaluation of packet-consumed PR106 sidecar/compiler candidates, and wiring
score-aware HNeRV/PR95 parity or SIREN/Ballé substrate trainers into byte-closed
archives before further remote spend.
