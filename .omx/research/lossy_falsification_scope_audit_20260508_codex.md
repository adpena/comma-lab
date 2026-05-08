# Lossy falsification scope audit — 2026-05-08

## Question

Was the `lossy_int4` / lossy-weight-compression work actually falsified?

## Answer

No broad family falsification is justified. The honest status is narrower:
several measured configurations are non-dispatchable from CPU/MPS/proxy
evidence, while the family remains open until the remaining calibration paths
and byte-closed runtime packets receive exact CUDA auth eval.

Evidence grades in this audit are intentionally conservative. CPU/MPS
roundtrip error and byte counts are proxy evidence only. They cannot promote,
rank, kill, or anchor a score claim.

## Current measured scope

| Lane | Current evidence | Honest verdict |
| --- | --- | --- |
| Naive lossy int4 PTQ | `100799` bytes, `37.42%` proxy rel_err | Measured config not dispatchable. Not a family kill. |
| Int4 QAT | `115641` bytes, `28.478992843129316%` proxy rel_err, MPS research signal | Measured QAT config not dispatchable. QAT recipe is not exhausted globally. |
| Int4 per-channel scales | `115958` bytes, `30.4108214365326%` proxy rel_err | Measured per-channel config not dispatchable. |
| Mixed int4/int6/int8 | `187785` bytes, `4.8953699576675485%` proxy rel_err | Distortion reopens the lane, but this point is byte-dominated by PR101 brotli (`178144` bytes) and lossy coarsening. Not dispatchable. |
| GPTQ | `101106` bytes, `45.69922978599979%` proxy rel_err, CPU synthetic-latent calibration smoke | Measured GPTQ config not dispatchable. This is not a family kill and not score evidence. |
| AWQ | `100726` bytes, `37.41986206728719%` proxy rel_err, CPU synthetic-latent calibration smoke | Measured AWQ config not dispatchable. This is not a family kill and not score evidence. |
| Lossy coarsening analytical | `156344` bytes at `0.03856566284611934` relative error proxy | Best current proxy byte/rel_err point. Still not a score claim until byte-closed runtime and exact CUDA auth eval. |

## Guardrail changes made

- CPU/MPS/proxy int4 tools now emit fail-closed fields:
  `score_claim=false`, `promotion_eligible=false`,
  `rank_or_kill_eligible=false`, `ready_for_exact_eval_dispatch=false`,
  `dispatch_attempted=false`, `proxy_row=true`.
- Proxy manifests now include `family_falsified=false` and an explicit
  `falsification_scope`.
- Mixed-precision byte accounting now checks the estimated raw payload bytes
  against the actual packed payload bytes before reporting.
- Mixed-precision proxy classification now marks larger-than-baseline lossy
  points as byte-dominated instead of `CONDITIONAL` dispatch candidates.
- PR106 monolithic surgery reports now distinguish archive-container rewrites,
  charged byte-count changes, and actual score-affecting payload changes.

## Remaining work

1. Finish byte-closed runtime/dequant path for lossy coarsening.
2. Harvest the Lightning `lossy_coarsening_analytical_cuda` dispatch before
   using it as score evidence.
3. Re-run autopilot evidence ingestion after stale `FALSIFIED` rows are
   superseded by newer scoped evidence rows.

## Bottom line

The correct language is `measured configuration not dispatchable`, not
`family falsified`. The only current positive proxy point is lossy coarsening;
it is a serious score-lowering candidate but remains non-promotable until
exact CUDA auth eval lands on the exact byte-closed archive.

## 2026-05-08 GPTQ/AWQ addendum

GPTQ and AWQ are no longer untested in the local proxy sense. The newest CPU
smoke manifests are:

- `reports/raw/pr101_lossy_int4_gptq_20260508T025726Z/manifest.json`: GPTQ
  `8` synthetic calibration samples, `block_size_gptq=64`, `101106` archive
  bytes, `45.69922978599979%` weighted proxy rel_err,
  `MEASURED_CONFIG_NOT_DISPATCHABLE`.
- `reports/raw/pr101_lossy_int4_awq_20260508T025725Z/manifest.json`: AWQ
  `8` synthetic calibration samples, alpha grid `[0.0, 0.5, 1.0]`, `100726`
  archive bytes, `37.41986206728719%` weighted proxy rel_err,
  `MEASURED_CONFIG_NOT_DISPATCHABLE`.

Both manifests are proxy-only: `score_claim=false`,
`promotion_eligible=false`, `rank_or_kill_eligible=false`,
`ready_for_exact_eval_dispatch=false`, `proxy_row=true`, and
`family_falsified=false`. The exact blockers remain byte-closed int4 runtime
packet missing, no int4 decoder runtime built, synthetic activations rather
than PR106 video latents, and missing exact CUDA auth eval. These rows close
the GPTQ/AWQ evidence gap only for the measured proxy configs; they do not
kill the broader lossy-int4 family.
