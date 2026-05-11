# Full-stack score-lowering synthesis (2026-05-11)

## Scope

This memo records the xhigh full-pipeline synthesis requested by the operator:
bit/byte mechanics, runtime architecture, scorer/loss/training mechanics,
optimization, provider paths, and broad algorithmic search space. It is a
read-only synthesis converted into durable next actions; no score claim is
created here.

## Correct current frontier separation

Evidence axes remain separate:

- active internal `[contest-CUDA]` floor:
  PR106 latent sidecar, score `0.20739428085403283`, archive bytes
  `186808`, archive SHA-256
  `947b85e8a69db295d4dcf80b0b528639c47839f40f289a2c05b70a2064658b48`,
  adjudicated at
  `experiments/results/modal_auth_eval/pr106_latent_sidecar_20260511T150517Z/contest_auth_eval.adjudicated.json`;
- best exact `[contest-CPU]` public-axis artifact found locally:
  A1 PR101-derived, score `0.19284757743677347`, archive bytes `178262`,
  archive SHA-256
  `87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5`;
- previous exact `[contest-CUDA]` HNeRV rate anchor:
  PR103-on-PR106 AC repack, score `0.20898105277982337`, archive bytes
  `185578`, archive SHA-256
  `ec0890c2d2317dcad903ed37ffddb2794cd19c1df9effa057cb7f05af205e1ce`;
- PR103 `-16B` clean-runtime packet:
  exact Modal T4 rate-only positive versus PR103 source, but score
  `0.22776742708207615`, so it is not the active CUDA floor.

Do not rank CPU, CUDA, macOS CPU, MPS, or proxy signals against each other.
MPS remains configuration discovery only, never auth eval.

## Highest-EV execution order

## Multi-scale review discipline

Every score-lowering decision must be reviewed at five scales before it changes
the queue:

1. **Bit/byte scale.** Prove which charged bytes changed, which runtime code
   consumes them, whether the stream is entropy-saturated, and whether output
   parity or score movement follows from the exact `inflate.sh` contract.
2. **Packet/runtime scale.** Treat the archive plus runtime as a small compiler
   target: typed sections, deterministic lowering, strict ZIP/runtime custody,
   no-op detection, and reproducible decoder behavior across CPU/CUDA paths.
3. **Training/loss scale.** Optimize the contest score-domain objective, not
   uncalibrated weight error. The PR95/PR101 lesson is archive-in-loop,
   eval-roundtrip-in-loop, differentiable color/preprocess, EMA/export, and
   exact runtime closure.
4. **Portfolio scale.** Rank work by expected information gain per wall-clock
   and dispatch dollar. Stop shaving locally saturated basins when substrate
   training, grammar-aware packet transforms, or dual-axis custody will change
   decisions faster.
5. **Floor scale.** Keep the Shannon/IB/MDL lower bound visible: byte-only
   wins of a few dozen bytes are real but cannot substitute for lower
   SegNet/PoseNet distortion. Large movements require score-aware
   representation and correction bits that the packet actually carries.

This prevents two recurring errors: promoting CPU/proxy/MPS evidence onto the
CUDA axis, and mistaking a local packet-byte optimum for a global score optimum.

## Highest-EV execution order

1. **Exploit PR106 latent sidecar on the CUDA floor.**
   PR106 latent sidecar is now the active exact-CUDA floor. Paired Linux CPU
   replay exists and is worse (`0.2286802845175232`), with the regression
   concentrated in PoseNet. This makes the next score-lowering move the
   radius-2 latent table and byte-closed materialization path, not another
   PR103-on-PR106 rate-only replay. CPU/CUDA remains packet-specific evidence,
   never a monotone rule.

2. **T1 Ballé/HNeRV substrate harvest, not duplicate launch.**
   Active Modal call `fc-01KR955JSYQAVTTYZA48VAV7WJ` remains pending for lane
   `t1_balle_128k_endtoend`. Do not duplicate. Recover/close terminally before
   any same-lane spend or status change.

3. **Score-aware HNeRV/PR95 parity work.**
   Score-domain training and export-first packet construction dominate after
   entropy-saturated byte streams. Required stack: RGB renderer,
   eval-roundtrip-in-training, differentiable YUV6, score/scorer-domain loss,
   EMA/export discipline, archive builder in loop, strict runtime closure,
   and exact eval custody.

4. **PR103/PR106 grammar-aware byte work only on consumed sections.**
   Generic recompression of `decoder.merged_ac` is low EV because it is near
   entropy saturation. Continue only through parser-proven consumed grammar:
   `decoder.hists`, hardcoded constants, ZIP/header bytes, latent-hi histogram
   grammar, or new deterministic packet compiler transforms with parity proof.

5. **Per-pair latent/correction search only as byte-closed runtime packets.**
   CMA-ES/Optuna/BO are appropriate for low-dimensional corrections and codec
   knobs, but proxy winners must materialize into an archive/runtime packet
   with no-op proof before exact eval.

6. **Native/Rust/Zig/C/ASM acceleration as conformance-backed PacketIR ports.**
   Native coders are justified after Python golden vectors prove bitstream
   semantics and >100B net savings or substantial runtime/DX savings. Do not
   let native become an unreviewed parallel truth.

## Tooling and math map

- CMA-ES: continuous latent, bias, correction, and low-dimensional runtime
  constants.
- Optuna/TPE: integer codec knobs, Brotli `lgwin`, histogram deltas, block
  sizes, side-channel vocabulary.
- Bayesian EI/EIG: exact-eval dispatch selection under limited provider budget.
- ADMM/water-filling: shared byte allocation across decoder, latents,
  correction atoms, and residual streams.
- Entropy/MDL/IB: decide whether a lane is information-efficient before GPU
  spend; penalize non-exportable or non-consumed representation complexity.
- Range/ANS/arithmetic/bit packing/categorical coding: first-class byte
  transducers, but only after parser-proven stream semantics and conformance
  vectors.

## CPU/GPU gap falsification

Required experiments should isolate:

- PyAV vs DALI/NVDEC decode/preprocess;
- CPU vs CUDA SegNet/PoseNet kernels;
- TF32/cudnn nondeterminism;
- runtime/inflate color/layout differences;
- exact same archive/runtime across both axes.

Use dual-device plan artifacts as the top-level custody guard, then targeted
drift probes for mechanism localization. Do not extrapolate one axis from the
other.

## Next concrete commands

```bash
.venv/bin/python tools/claim_lane_dispatch.py summary --format json
.venv/bin/python tools/modal_function_status.py fc-01KR955JSYQAVTTYZA48VAV7WJ --get-timeout-s 0
.venv/bin/python tools/plan_dual_device_auth_eval.py \
  --archive experiments/results/lightning_batch/pr103_pr106_ac_repack_exact_eval_t4_20260507T181300Z/archive.zip \
  --inflate-sh submissions/pr103_pr106_final_runtime/inflate.sh \
  --label pr103_pr106_active_floor \
  --cuda-artifact-json experiments/results/lightning_batch/pr103_pr106_ac_repack_exact_eval_t4_20260507T181300Z/contest_auth_eval.adjudicated.json \
  --json-out experiments/results/dual_device_auth_eval/pr103_pr106_active_floor_plan_20260511.json
```

The CPU replay itself still needs a fresh lane claim before dispatch.
