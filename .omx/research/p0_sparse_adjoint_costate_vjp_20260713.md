---
title: "P0 sparse/structured adjoint on exact task455 input costates"
date_utc: "2026-07-13"
lane_id: "p0_sparse_adjoint"
research_only: true
score_claim: false
pointer_moved: false
training_performed: false
live_run_mutated: false
verdict: "NO-GO-dense-fullrank"
verdict_scope: "source-bound task455 n600 replay; frozen EfficientNet-B2 U-Net SegNet CE input costate; 4.7366% output masks and high-fidelity cross-state low-rank basis; macOS CPU advisory training-gradient axis"
---

# P0 sparse adjoint / costate VJP — n600 build + measure

## Outcome first

**Input-costate concentration: MEASURED 26.2857% L1 mass (61.0134% L2 energy) in the top
4.7366% pixels. Low rank: MEASURED raw `r95=68/120`, `r99=100/120`; rank 64 still has 23.8192%
relative Frobenius error. Masked adjoint: `NO-GO` for exact and for the tested 4.7366% bounded
approximation. Backward saving: DERIVED ideal custom spatial-sparse ceiling `2.208577x`, but exact
and current dense-kernel speedups are `1.0x`. Scoped verdict: `NO-GO-dense-fullrank`.**

This is a **MEANS** result. It neither evaluates an archive nor moves the contest pointer.

## Verdict and verdict scope

The tested arm is **`NO-GO-dense-fullrank`**, with three separate formulation-level negatives:

1. **`FORMULATION NO-GO — exact masked adjoint`.** Finite-fp32 CE has zero exact-zero spatial
   support on all 600 states; the frozen scorer has a 685-pixel exact local halo and 23 global
   squeeze-excite reductions. Exact source coverage is 100% and exact sparse-backward speedup is
   `1.0x`.
2. **`FORMULATION NO-GO — 4.7366% bounded-error masked adjoint`.** Even the post-hoc oracle that
   selects pixels by the actual output-gradient L2 magnitude has `0.363536` global input-costate
   relative L2 error. The deployable cached source-margin mask has `0.793434` error. A cheap mask
   cannot be admitted merely because its output CE gradient is concentrated.
3. **`FORMULATION NO-GO — high-fidelity fixed low-rank cohort basis`.** The raw spectrum needs
   `68/120` modes for 95% energy and `100/120` for 99%; centered and row-normalized spectra are
   less compressible at high fidelity. The input-costate cohort is not low rank in the sense needed
   for a small-`r` replacement.

`verdict_scope`: the exact frozen task455 scorer/objective and source-bound n600 replay; binary
spatial logit-gradient masks, especially area `4.7365976969%`; and linear fixed-basis compression
across the 120 full-grid heldout costates. This does **not** kill a learned current-witness mask at
larger support, decoder-only sparse kernels after one dense encoder, stale/donated-SE approximations,
the #484 pre-SE local scorer, the #485 JEPA-latent provider, nonlinear costate manifolds, CUDA sparse
kernels, or any score axis.

## STORES CONSULTED

- `CLAUDE.md`, `AGENTS.md`, `docs/operating_manual_craft_handoff.md`, the v7.5 §8 operating
  contract, and the v8 per-class decomposition spec.
- `reports/latest.md`; lane, maturity, subagent, task, equation, probe-outcome, cost-band, continual-
  learning, and modal-call canonical state; current sister findings/session/design/council memos; all
  operator directive files from the preceding 24 hours.
- `.omx/research/per_epoch_detailed_accounting_20260713.md`,
  `.omx/research/onpolicy_surrogate_95kill_20260713.md`,
  `.omx/research/frozen_replay_convex_head_95kill_20260713.md`, its DAG FEED and terminal receipt,
  and `.omx/research/cheapen_real_95_tilehalo_fp16_20260713.md`.
- Read-only task455 state assignments, three source checkpoints, fixed `gt_n600.npz`, frozen SegNet
  weights, 480 train-cache sufficient-stat records, and 120 heldout reduced-metric records.
- Frozen scorer source and the existing exact-costate/renderer paths. No live trainer, live-run
  file, archive, cloud/provider, GPU, submission, or evaluator surface was actuated.

## What was measured

The prior task455 landing intentionally retained no raw 1.4-GB costate tree. This probe therefore
**regenerated**, rather than presumed, the exact batch-1 costate for every immutable state:

```text
x_s = exact cached replay frame for state s
z_s = F_frozen(x_s)
L_s = CE(z_s, cached exact label_s)
g_s = dL_s/dz_s
lambda_s = dL_s/dx_s = J_F(x_s)^T g_s.
```

Every regenerated target was checked against existing task455 content custody before use:

- train `480`: exact cached stride-8 target hash;
- heldout `120`: exact cached full-grid target hash;
- result: **MEASURED `600/600` matches, zero mismatches**.

All-600 concentration uses regenerated full grids. Masked-VJP curves and per-convolution numerical
support use the 120 heldout full-grid states. The exact spectrum uses the matrix
`G in R^(120 x 589824)`, i.e. exactly **70,778,880 costate elements**. Raw arrays were success-only
scratch: the run certified their paths, bytes, per-file SHA-256 values, deterministic rebuild command,
source receipt, and derived spectrum hash before removing them. No evidence path cites `/tmp`.

Measurement receipt:
`experiments/results/p0_sparse_adjoint_costate_vjp_20260713/measurement_receipt.json`, SHA-256
`52a22f4b60367fc27ca0fca7293b0741da4b809724479cd3ef7e92291c250cef`.

Axis: **`[macOS-CPU advisory; Torch/NumPy-fp32 training-gradient MEANS only]`**. Seed `455`;
deterministic Torch algorithms enabled; no training.

## Adjoint derivation — what “exact masking” actually requires

Let frozen scorer logits be `F(x)`, its Jacobian be `J_F(x)`, the surrogate logit adjoint be `g`,
and a binary spatial mask broadcast across classes be `M`. Then

```text
lambda     = J_F(x)^T g
lambda_M   = J_F(x)^T M g
error      = lambda - lambda_M = J_F(x)^T (I-M) g

||error||_2 <= ||J_F(x)||_2 ||(I-M)g||_2.
```

Therefore **DERIVED** exactness is

```text
lambda_M = lambda  iff  J_F(x)^T(I-M)g = 0.
```

“CE is saturated” is not that condition. Small omitted gradients give a bounded approximation only;
they are exact only if zero or in the current Jacobian-transpose nullspace. Here the measured omitted
VJP is nonzero.

The global-support obstruction is explicit for a squeeze-excite block. With spatial index `p`,
`m=P^-1 sum_p x(p)`, and `y(p)=s(m) odot x(p)`, its adjoint is

```text
lambda_x(p) = s(m) odot lambda_y(p)
            + P^-1 J_s(m)^T sum_q[x(q) odot lambda_y(q)].
```

The second term is shared across every `p` and is generically nonzero even when `lambda_y` begins on
a small annulus. The frozen EfficientNet encoder has 23 such reductions. Independently, the derived
exact local halo is 685 pixels, larger than the relevant image extent. Thus output sparsity does not
imply exact input-support sparsity.

## Concentration — output annulus versus input costate

All rows below are **MEASURED on all 600 real cached states**. “Top” is an oracle sort by the field's
own per-pixel channel-reduced magnitude. “Source margin” selects the lowest cached source-label
margin and is deliberately not equated with #333's d_seg-mass annulus.

| Top-area fraction | Output-gradient L1 | Output-gradient L2 energy | Input-costate L1 | Input-costate L2 energy |
|---:|---:|---:|---:|---:|
| 0.1% | 1.9575% | 3.6280% | 1.6057% | 9.6194% |
| 1% | 18.3681% | 31.4302% | 9.1692% | 32.6307% |
| 2% | 33.9519% | 53.8531% | 14.8951% | 44.1605% |
| **4.7366%** | **64.3042%** | **86.4134%** | **26.2857%** | **61.0134%** |
| 10% | 87.6430% | 98.7970% | 41.2293% | 76.2527% |
| 20% | 95.6922% | 99.9373% | 59.5683% | 88.5579% |
| 50% | 98.9216% | 99.9973% | 86.6054% | 98.2341% |

#333's approximately 97% `d_seg`-mass-in-4.7%-area result concerns a discrete error field, not this
CE logit adjoint or its input VJP. The measured 64.30% output-L1 and 26.29% input-L1 rows therefore
do not contradict #333; they reject transferring that concentration percentage across fields.

| Field / selector at 4.7366% area | Global L1 mass | Global L2 energy | Per-state median L1 | Interpretation |
|---|---:|---:|---:|---|
| output CE gradient, own top-L2 pixels | 64.3042% | 86.4134% | 65.4753% | output is strongly concentrated, not sparse |
| output CE gradient, source-margin mask | 38.0659% | 45.6424% | 38.4490% | fixed source margin misses current CE saliency |
| input costate, own top-L2 pixels | **26.2857%** | **61.0134%** | 26.3203% | VJP spreads L1 mass substantially |
| input costate, source-margin mask | **10.3070%** | **16.3785%** | 10.5065% | practical boundary mask poorly localizes costate |

The spatial exact-zero fraction is **MEASURED `0.0` for both output gradient and input costate in
every state**. Thus the CE adjoint is numerically concentrated but not exactly sparse; the input
costate is dense and materially less L1-concentrated.

## Masked-VJP error curves

Two interpretations were shipped and measured, not arbitrated in prose:

- `top_output`: post-hoc oracle selecting the actual largest output-gradient L2 pixels. This is a
  best-case diagnostic and not a deployable free mask.
- `source_margin`: deployable cached lowest-source-margin pixels, composing most directly with the
  #484 boundary-feature idea.

All values are **MEASURED over 120 exact full-grid heldout costates**. `out-L1` is the median output
gradient L1 retained; cosine and relative L2 are global over all costate elements.

| Area | Oracle out-L1 | Oracle costate cosine | Oracle rel-L2 error | Source-margin out-L1 | Source-margin cosine | Source-margin rel-L2 error |
|---:|---:|---:|---:|---:|---:|---:|
| 1% | 18.7073% | 0.809980 | 0.725305 | 10.0442% | 0.694980 | 0.953148 |
| 2% | 34.5934% | 0.886032 | 0.578681 | 19.0040% | 0.716644 | 0.908258 |
| **4.7366%** | **65.8149%** | **0.954554** | **0.363536** | **38.8890%** | **0.775455** | **0.793434** |
| 10% | 89.1995% | 0.993710 | 0.135394 | 63.3836% | 0.835265 | 0.646846 |
| 20% | 95.9273% | 0.999733 | 0.026206 | 75.6593% | 0.888395 | 0.540174 |
| 50% | 98.9764% | 0.999992 | 0.004357 | 84.7239% | 0.938003 | 0.416658 |
| 80% | 99.8488% | 1.000000 | 0.000997 | 91.8470% | 0.964705 | 0.311997 |

The 20% oracle row is the first interesting child (`2.6206%` global relative L2), but it is not an
admission: the mask uses the exact output gradient it aims to avoid, and no renderer-gradient regret,
optimizer-step, or full-facet gate was run. The source-margin interpretation remains poor even at
80%, showing that source margin is not a sufficient current-witness mask here.

## Backward FLOP saving — ideal support bound versus realized dense graph

For each Conv2d input-gradient call, the probe records nominal dense multiply-add FLOPs and propagates
the numerical cotangent support through the real backward graph. The **DERIVED spatial-sparse upper
bound** charges a whole output spatial site when any channel is active. This is intentionally less
optimistic than element sparsity and still assumes unavailable custom support-aware kernels.

| 120-state aggregate, 4.7366% mask | Dense nominal conv VJP FLOPs | Ideal spatial-support FLOPs | DERIVED ceiling | Dense-kernel arithmetic speedup |
|---|---:|---:|---:|---:|
| oracle top-output | 2,378,240,148,480 | 1,076,819,892,485 | **2.208577x** | **1.0x** |
| source-margin | 2,378,240,148,480 | 1,065,091,563,460 | **2.232897x** | **1.0x** |
| full exact CE | 2,378,240,148,480 | 2,378,240,148,480 | **1.0x** | **1.0x** |

For the oracle mask, the family ceilings are **DERIVED** `21.1111x` at the segmentation head,
`3.5676x` in the decoder, but only `1.0168x` in the encoder. The encoder nearly densifies because of
receptive-field growth and global SE. Therefore the whole-network ceiling is only `2.2086x`, not the
naive `1/0.047366 = 21.11x`.

This bound is not a wall-time measurement. Ordinary Torch convolution still executes the full dense
graph. On the same 120 heldout states, **MEASURED medians** were `0.3172 s` for the full VJP,
`0.3045 s` for oracle-masked VJP, and `0.3109 s` for source-margin VJP; one pass per state and no
interleaved repeats do not resolve that few-percent ordering from host noise. No wall-time speedup is
claimed. The only structurally honest realized dense-kernel figure is `1.0x` arithmetic work.

## Full-cohort low-rank spectrum

The spectrum is not a randomized sketch: the probe materialized the 120 full fp32 costates as
certified success-only scratch, formed the complete `120 x 120` Gram matrix with NumPy-fp32
accumulation, and eigendecomposed its fp64 cast. Rank is bounded by the state count `120`, not by the
pixel dimension `589824`. “Full” here means every heldout state and element, not infinite-precision
linear algebra.

| Matrix treatment | r50 | r80 | r90 | r95 | r97 | r99 | Stable rank | Entropy effective rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw costates | 5 | 25 | 47 | **68** | 80 | **100** | 3.4268 | 28.0230 |
| mean-centered | 9 | 33 | 55 | 74 | 85 | 102 | 6.6648 | 43.3511 |
| row-L2 normalized | 6 | 37 | 64 | 85 | 96 | 110 | 3.4560 | 35.0553 |

By Eckart-Young-Mirsky, **DERIVED from the measured singular values**,

```text
min_rank(Gr)<=r ||G-Gr||_F / ||G||_F
  = sqrt(1 - sum_{i<=r} sigma_i^2 / sum_i sigma_i^2).
```

For raw costates the optimal errors are `0.8415` at rank 1, `0.6287` at rank 8, `0.5249` at rank
16, `0.3978` at rank 32, **`0.238192` at rank 64**, and `0.1120` at rank 96. Numerical raw rank is
`50` at a `1e-2` relative-eigenvalue threshold, `115` at `1e-3`, and the full `120` at `1e-4`.
There are useful coarse common modes, but the high-fidelity tail is broad.

### Why “randomized VJP at O(r)” does not close this arm

One scalar surrogate loss already yields `J_F(x)^T g` in **one** reverse pass. Computing `r` basis
VJPs costs `r` reverse passes unless those basis VJPs are cached and reusable. Across states both
`J_F(x_s)` and `g_s` change, so the cohort spectrum alone does not make such reuse exact. Hutchinson
estimators are appropriate for traces or Jacobian statistics; they do not beat the one-pass exact VJP
for this single vector without extra structure.

**DERIVED conclusion:** the measured spectrum can support an offline costate compressor or a learned
provider with cheaply inferred coefficients, but not a direct `O(r)<O(1)` replacement for a fresh
VJP. Any such provider returns to the already-governed surrogate/current-state fidelity problem and
must pass exact n600 renderer-gradient and optimizer-regret gates.

## Accounting caveat — the 82% premise remains unresolved in-loop

The operator-routed pivot is based on prior diagnostic **MEASURED** `537.045 ms` forward and
`3009.070 ms` forward+backward, hence **DERIVED** `2472.024 ms` / `82.15%` backward. The source memo
also records that this diagnostic is approximately 12x heavier than the real loop and says the ratio
is unverified in-loop.

This probe's exact batch-1, one-thread replay measured all-600 median forward `0.606252 s` and median
backward `0.288869 s`, the opposite ordering. These are again diagnostic component timers, not live
MLX-loop attribution; contention inflates their means. **INFERRED:** substrate, thread policy, cache
state, and harness composition materially affect the ratio. **Verdict:** neither diagnostic may be
promoted to the current in-loop split. The D-A in-loop timer remains the authority needed to settle
the 82% premise. This does not change the sparse-adjoint fidelity negative.

## Composition with #484 and #485

- **#484 pre-SE boundary features:** this result says the source-margin mask is insufficient and
  global SE is the exact-support blocker. A child may become viable only by moving the cut before
  global SE or donating/staling SE state, learning saliency on the **current witness**, and measuring
  the larger-support error/FLOP frontier. The 20% oracle row is a target, not an available mask.
- **#485 JEPA latent surrogate:** the raw spectrum's low stable rank but high r95/r99 suggests a
  coarse latent plus structured residual, not a tiny linear basis. JEPA must be judged as a current-
  state provider with full costate/renderer-gradient regret; the spectrum is a capacity prior only.

## Triality and system-intelligence landing

- **Equation:** `src/tac/canonical_equations/sparse_adjoint_support_closure_20260713.py` registers the
  masked-error inequality, SE global-support term, Eckart-Young rank error, exactness boundary, and
  empirical anchor. Its focused tests populate only a temporary locked registry.
- **DAG:** `.omx/research/p0_sparse_adjoint_costate_vjp_DAG_FEED_20260713.md` is the isolated FEED.
  Shared append is deferred because the canonical append-only registry/DAG surfaces were already hot.
- **DSL:** no arm is admitted, so no new flag is invented and no trainer/witness-control file is
  edited. Existing typed scorer-gradient policy must retain `full_teacher` fallback.
- **Sensitivity/Pareto:** the output/input mass curves and input-costate error versus ideal sparse
  FLOPs form the reusable sensitivity/Pareto surface.
- **Bit allocator:** no archive-byte actuator exists; this is training-compute evidence only.
- **Cathedral/autopilot:** consume `NO_GO_DENSE_FULLRANK`; do not dispatch the tested arm.
- **Continual learning:** memo, receipt, equation, DAG FEED, and probe are the durable posterior.
- **Disambiguator:** both top-output oracle and source-margin masks were measured at seven support
  fractions; no prose-only choice was made.

## Verification, custody, and pointer-delta honesty

- Probe: `tools/probe_sparse_adjoint_costate_vjp.py`; resumable per pair, atomic progress and stage
  manifests, single-writer lock, deterministic seed, source seals, exact hash gates, and certified
  success-only cleanup.
- Storage preflight: `.omx/research/p0_sparse_adjoint_storage_preflight_20260713.json`; SSD tiers were
  unavailable in this sandbox, so an explicit small local tier was selected with sufficient space.
- Receipt JSON is finite; `600` pair records exist; temporary raw-costate scratch is absent after
  certification; cleanup status is `CERTIFIED_REBUILDABLE_SCRATCH_REMOVED`.
- No training, paid dispatch, live-run mutation, trainer/costate/witness-control edit, archive
  mutation, or evaluator call occurred.
- Focused equation tests and probe source checks are recorded in the handoff verification.
- **Pointer delta: `NONE`.** Evidence is MEANS-only and uncommitted for main review, as requested.

## Reactivation rule

Do not build the tested 4.7366% masked or fixed low-rank adjoint. Reopen only a changed formulation
that supplies all of: (1) a cheap current-witness mask, initially tested around the measured 20%
oracle frontier; (2) custom sparse decoder/head kernels with measured wall time; (3) explicit
handling/removal/donation of global SE; (4) n600 input-costate and renderer-gradient error; (5)
matched optimizer-step regret and holistic CE/`d_seg`/`d_pose`/rate gates; and (6) a fresh in-loop
D-A accounting anchor. Until then, full dense teacher VJP is the fail-closed path.
