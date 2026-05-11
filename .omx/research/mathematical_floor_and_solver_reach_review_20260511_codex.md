# Mathematical floor and solver reach review (2026-05-11)

## Scope

This memo is a second adversarial pass after the senior handoff review. It
formalizes what can and cannot be claimed about the theoretical floor, and it
maps the micro/macro search space to concrete solvers and proof obligations.

Score claim: `false`.
Dispatch attempted: `false`.

## Score functional

The contest score is:

```text
S(B, d_seg, d_pose)
  = 25 * B / 37,545,489
  + 100 * d_seg
  + sqrt(10 * d_pose)
```

For the current PR106 latent sidecar R2 exact-T4 measured frontier, using the
rounded-report recorded components:

- `B = 186822`
- `d_seg = 0.00064260`
- `d_pose = 0.00003236`
- `dS/dB = 6.658589531e-7` score per byte
- `dS/dd_seg = 100`
- `dS/dd_pose = 277.9494`
- pose marginal is `2.779x` the seg marginal at this operating point

Equivalent local break-even thresholds:

| Extra charged bytes | Score cost | Required seg reduction | Required pose reduction |
|---:|---:|---:|---:|
| 1 | 6.6586e-7 | 6.6586e-9 | 2.3956e-9 |
| 10 | 6.6586e-6 | 6.6586e-8 | 2.3956e-8 |
| 100 | 6.6586e-5 | 6.6586e-7 | 2.3956e-7 |
| 1000 | 6.6586e-4 | 6.6586e-6 | 2.3956e-6 |
| 10000 | 6.6586e-3 | 6.6586e-5 | 2.3956e-5 |

This is the local rounded-report calculus behind the current priority shift
toward pose-axis sidecars and residual atoms. It is a prioritization prior, not
a theorem that pose candidates have higher expected value per charged byte; that
requires measured or bounded `d_pose/db` and `d_seg/db` on the same packet and
device axis.

## What can be proven

The global Shannon floor for the finite contest video and scorer is not
presently proven. A rigorous floor statement must be scoped.

Valid proof forms:

1. **Trivial global bound:** `S >= 0`. True but operationally useless.
2. **Grammar-scoped lower bound:** for a fixed archive grammar and runtime
   decoder class, prove entropy or code-length lower bounds for streams under
   that grammar, then prove no candidate in that class can beat a target score.
3. **Trust-region lower bound:** for a local family of sidecar or residual
   atoms, exhaust or branch-and-bound the discrete search space with certified
   scorer response intervals.
4. **Constructive upper bound:** produce an exact packet with custody and
   exact eval. This is the only way to lower the actual frontier.
5. **Matched bound:** combine a lower-bound certificate and a constructive
   packet whose score is within epsilon. This is the only honest way to say a
   scoped floor is solved.

Invalid proof forms:

- proxy/MPS/macOS curves promoted to exact floor statements;
- solver rows without charged-byte materialization;
- entropy estimates that ignore decoder grammar and overhead;
- additive component forecasts without measured interaction terms;
- CPU rows ranked against CUDA rows without paired axis custody;
- "HNeRV is optimal" or "HNeRV is exhausted" without non-HNeRV packet tests.

## Optimization program

The right abstraction is a mixed discrete/continuous rate-distortion problem:

```text
minimize_x   25 * B(x) / N + 100 * d_seg_device(x)
             + sqrt(10 * d_pose_device(x))
subject to   materialize(x) emits charged archive bytes
             inflate consumes those bytes
             runtime is deterministic and contest compliant
             archive/runtime custody is exact
             no-op proof passes
             device axis is explicit
```

Relaxed local allocation heuristic for active byte atoms:

```text
[-partial S_i / partial b_i] - interaction_penalty_i ~= lambda
```

Inactive atoms should not exceed the active marginal value under the same
relaxation. This is not a proof condition for discrete ZIP bytes, thresholded
scorer behavior, nonconvex decoder parameters, or sign-changing interactions
unless a specific convex relaxation or discrete exchange bound is supplied. If
interactions are large or sign-changing, the additive model is invalid and the
stack must be measured as a joint packet.

## Solver map

Use solvers by evidence class, not by fashion:

- coordinate search: exact small vocabularies, PR106 sidecar radii, bias
  constants, one-dim perturbation tables;
- dynamic programming / knapsack: per-pair atoms with additive byte costs and
  bounded interaction;
- Optuna/TPE: categorical and integer codec knobs such as block size, Brotli
  lgwin, stream split, sidecar vocabulary, and transform family;
- CMA-ES: low-dimensional continuous corrections and residual bases where the
  archive compiler is deterministic;
- Bayesian EIG: choose the next exact eval under provider budget;
- ADMM/water-filling: coupled bit allocation across decoder, latent, sidecar,
  pose, and residual streams;
- branch-and-bound / ILP: global bit-packing or vocabulary choices with exact
  discrete constraints;
- differentiable training: only once RGB renderer, eval-roundtrip, YUV/scorer
  preprocessing, export, and archive build are in the loop.

## Micro-to-macro execution stack

1. **Bit level:** ZIP/section offsets, entropy, range/ANS/arithmetic coding,
   bit packing, hardcoded lengths, consumed-byte proof.
2. **Atom level:** per-pair latent deltas, y-shift, pose atoms, residual basis
   coefficients, categorical tokens.
3. **Stream level:** sidecar grammar, latent stream, mask stream, pose stream,
   residual stream, decoder weights.
4. **Packet level:** deterministic compiler, release surface, strict
   compliance, raw-output SHA.
5. **Device level:** paired CPU/CUDA matrix and loader/kernel drift
   localization.
6. **Training level:** PR95 discipline: full RGB renderer, score-aware loss,
   eval-roundtrip, differentiable preprocessing, EMA/export, archive-in-loop.
7. **Portfolio level:** expected score decrease per wall-clock and dollar,
   with exact-candidate gating.
8. **Floor level:** scoped lower-bound certificate plus constructive exact
   packet.

## Immediate score-lowering implications

1. R2 compliance/custody is not optional. It turns the best exact score into a
   promotion-grade artifact.
2. R2 paired CPU/CUDA matrix is high value because device-axis inversion is
   packet-specific.
3. PR106 sidecar compression should target pose-relevant correction bytes first
   unless entropy/grammar analysis shows a better rate-only payoff.
4. PR101/PR103 grammar work belongs in a deterministic packet compiler, not a
   one-off script.
5. PR97 and PR81/84/91/92/93 public non-HNeRV mechanisms should be converted
   into PR106 residual or sidecar score-table producers before any broad
   substrate replacement claim.
6. T1/Ballé/CompressAI is still strategically important, but only as an
   export-first, score-aware, archive-in-loop renderer/transform path.

## Frontier proof target

The next honest "floor proof" milestone is not a global theorem. It is:

1. define a scoped PacketIR grammar for PR106-style HNeRV sidecar packets;
2. prove identity re-emission and malformed-input fail-closed vectors;
3. build certified code-length lower bounds for each consumed stream including
   grammar/container overhead;
4. solve or bound the local sidecar atom search under that grammar;
5. produce the best constructive packet from that solved class;
6. exact-eval it on CUDA and CPU;
7. state separate epsilon gaps:
   `S_cuda(packet) - L_cuda` and `S_cpu(packet) - L_cpu`.

That would convert the current search from "many clever lanes" into a real
compiler-theoretic floor result for one important substrate class, while still
leaving room for non-HNeRV substrates to beat it.
