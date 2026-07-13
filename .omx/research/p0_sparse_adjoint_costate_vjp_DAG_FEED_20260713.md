# DAG FEED — P0 sparse-adjoint costate VJP — 2026-07-13

`research_only=true` · `score_claim=false` · `pointer_moved=false` · `$0 local` ·
`training_performed=false` · `live_run_mutated=false` · `shared_DAG_append=DEFERRED_MAIN`

Lane: `p0_sparse_adjoint`  
Equation: `sparse_adjoint_mask_error_and_se_support_closure_v1`  
Receipt SHA-256: `52a22f4b60367fc27ca0fca7293b0741da4b809724479cd3ef7e92291c250cef`

## Terminal state

`NO_GO_DENSE_FULLRANK` for the tested source-bound 4.7366%-mask and high-fidelity
cross-state low-rank formulations. This is a training-gradient MEANS verdict, not a score,
submission, or pointer verdict.

## Executable dependency graph

```text
sealed task455 n600 replay + frozen SegNet + exact objective + GT cache
  -> regenerate exact batch-1 input costate for each of 600 states
  -> hash gate
       train 480: cached stride-8 target hash equality
       heldout 120: cached full-grid target hash equality
  -> all-600 concentration
       output CE logit adjoint top-area and source-margin capture
       input costate top-area and source-margin capture
       exact-zero support gate
  -> heldout-120 mask disambiguator
       M_oracle = top output-gradient L2 pixels
       M_margin = lowest cached source-margin pixels
       areas = {1,2,4.7366,10,20,50,80}%
       -> exact VJP lambda_M = J_F(x)^T M g
       -> relative-L2/cosine error curve
  -> per-convolution cotangent-support propagation
       -> dense nominal backward-data FLOPs
       -> ideal element/spatial sparse FLOP upper bounds
       -> dense-kernel realized arithmetic path = unchanged
  -> full heldout Gram matrix over 120 x 589824 costate matrix
       (NumPy-fp32 Gram accumulation; fp64 eigendecomposition)
       -> raw / centered / row-normalized spectra
       -> Eckart-Young rank-error curves
  -> conjunctive decision
       exact mask: REFUSE (zero exact-zero support; global SE; full source area)
       4.7366% bounded mask: REFUSE (oracle rel-L2 0.3635; margin rel-L2 0.7934)
       high-fidelity low rank: REFUSE (raw r95=68/120; r99=100/120)
  -> retain full dense teacher VJP
```

No edge authorizes trainer wiring, a live launch, paid dispatch, or evaluator use.

## Canonical law

For frozen scorer logits `F(x)`, surrogate logit adjoint `g`, and binary spatial mask `M`,

```text
lambda       = J_F(x)^T g
lambda_M     = J_F(x)^T M g
lambda-lambda_M = J_F(x)^T (I-M) g

||lambda-lambda_M||_2 <= ||J_F(x)||_2 ||(I-M)g||_2
lambda_M = lambda  iff  J_F(x)^T(I-M)g = 0.
```

For squeeze-excite `y(p)=s(mean_p x(p)) odot x(p)`, its backward contains a spatially
constant global term:

```text
lambda_x(p) = s odot lambda_y(p)
            + P^-1 J_s(mean x)^T sum_q[x(q) odot lambda_y(q)].
```

That term is generically nonzero at every pixel. The frozen scorer has 23 such reductions and an
exact local halo of 685 pixels, so exact source coverage is 100%.

For costate matrix `G` and singular values `sigma_i`, the best possible rank-`r` error is

```text
min_rank(Gr)<=r ||G-Gr||_F / ||G||_F
  = sqrt(1 - sum_{i<=r} sigma_i^2 / sum_i sigma_i^2).
```

## Measured anchors

- All `600/600` regenerated target hashes match task455 custody.
- At 4.7366% area: output top-gradient capture is 64.3042% L1 / 86.4134% L2 energy;
  input top-costate capture is 26.2857% L1 / 61.0134% L2 energy.
- The practical source-margin mask captures 10.3070% input L1 / 16.3785% input L2 energy.
- Both output and input spatial exact-zero fractions are 0 for every state.
- Best-case 4.7366% oracle-mask input-costate error is 0.363536 global relative L2;
  source-margin error is 0.793434.
- Ideal custom spatial-sparse backward-data arithmetic ceiling is 2.208577x for the oracle mask;
  exact and ordinary dense-kernel speedups are 1.0x.
- The raw 120-state spectrum needs ranks 68 and 100 for 95% and 99% energy. Rank 64 still has
  0.238192 relative Frobenius error.

## Scoped verdicts and open children

- `FORMULATION NO-GO`: exact output-masked adjoint for the frozen task455 EfficientNet-B2 U-Net.
- `FORMULATION NO-GO`: 4.7366% approximate mask, including the best post-hoc output-gradient oracle.
- `FORMULATION NO-GO`: high-fidelity fixed cross-state low-rank costate basis on the measured cohort.
- `OPEN`: a current-witness learned mask at a larger measured support (the 20% oracle row reaches
  0.026206 global relative L2) plus custom decoder/head sparse kernels and renderer-gradient regret.
- `OPEN`: #484 pre-SE/local scorer, because removing/donating global SE changes the support law.
- `OPEN`: #485 JEPA-latent provider, because it replaces rather than sparsifies the frozen VJP.

## Triality and six-hook wire-in

- **Equation:** new isolated canonical-equation module with empirical anchor; locked shared registry
  append deferred because that hot append-only surface was already modified by siblings.
- **DAG:** this standalone FEED; shared DAG append deferred to main review.
- **DSL:** no lever is admitted, so no flag is invented. Existing typed scorer-gradient policy keeps
  `full_teacher` as fail-closed fallback.
- **Sensitivity map:** output/input concentration and mask-error curves are reusable saliency evidence.
- **Pareto:** explicit error versus ideal sparse FLOPs; no proxy-only admission.
- **Bit allocator:** no direct archive-byte actuator; evidence may rank training-compute acquisition only.
- **Cathedral/autopilot:** consume terminal no-go; do not dispatch this formulation.
- **Continual learning:** memo, receipt, equation module, and this FEED are the durable posterior update.
- **Probe disambiguator:** oracle-output and source-margin masks are both measured at seven areas.

## Custody and pointer

The probe used immutable cached states only, generated no training update, preserved source run
directories read-only, and removed its temporary 120-costate spectrum tree only after recording
path, bytes, per-file hashes, deterministic rebuild command, and derived spectrum hash. Evidence is
local advisory training-gradient MEANS. Pointer delta is exactly `NONE`.
