# Alternative algebraic forms for the conv-preimage wall — exact scope, $0 n600 bound, and build order

**Date:** 2026-07-18. **Lane:** `lane_alternative_forms_conv_wall_20260718` (`research_only=true`).

- **[MEASURED]** No training, provider dispatch, evaluator call, archive mutation, score claim, or pointer
  mutation occurred in this unit. The submittable pointer remains `0.19108` (`0.1910828242` in the
  consulted contest-CPU receipt).
- **[MEASURED]** The sacred live directory
  `experiments/results/levelset_n600_witness_20260717T113932Z/` was read only. Its banked epoch-725 EMA
  checkpoint hashes to `b0a431e9259cd3c54ae53b677076823f36e096b27eb0d9ba74ed7c54c9113cef`.
- **[DERIVED]** The winner-cell wall is not one problem with four interchangeable exact solvers. The
  frozen head is an exact four-dimensional polyhedral problem; the full frozen SegNet pullback is a
  mixed-activation, curved nonconvex inverse problem; the receiver lattice and archive grammar add
  separate feasibility constraints.
- **[INFERRED]** The highest-EV new build is a **Cole–Hopf heat/entropic target initializer pulled back
  through the existing factorized adjoint and followed by a short full-loss repair**, not a standalone
  Sinkhorn loss. It earns adoption only through the preregistered n600 gate in §6.

## Labels and verdict scope

- **[DERIVED]** `EXACT` means equality for the stated variables and model class, not merely a useful
  analogy. `RELAXATION` means a superset/lower bound or a surrogate objective. `FAST-SOLVE` means the
  reduced subproblem has a direct or rapidly convergent numerical solution; it does not imply that its
  solution has an RGB/uint8/archive preimage.
- **[DERIVED]** Every negative below is scoped to its named formulation, checkpoint, model class, or
  evidence axis. No family or paradigm is killed.
- **[MEASURED]** All new numeric evidence in this memo is `[macOS-CPU numpy advisory]`, non-promotable,
  and `score_claim=false`.

**[DERIVED] Algebraic-form decision matrix**

| form | exact / relaxation / fast-solve status | what it buys | hard boundary |
|---|---|---|---|
| tropical / max-plus | **[DERIVED]** exact for the affine head; exact-implicit for a declared PL surrogate; approximation for the mixed scorer | **[DERIVED]** active facets, rank-four constraint chart, local branch oracle | **[DERIVED]** no automatic minimal RGB representation or legal preimage |
| Cole–Hopf / Gibbs | **[DERIVED]** exact for quadratic viscous-HJ and per-pixel entropy smoothing; initializer elsewhere | **[INFERRED]** cheap target construction and dominant-part preconditioning | **[DERIVED]** residual neural, receiver, overlap, lattice, Pose, and rate terms remain |
| entropic OT / Sinkhorn | **[DERIVED]** fast exact solve of the stated regularized two-marginal transport problem | **[INFERRED]** spatial/class coupling when a real second marginal and cost are supplied | **[DERIVED]** marginal agreement does not imply pointwise partition or witness feasibility |
| SDP / QC | **[DERIVED]** outer relaxation/certificate; exactness retained only in the already-affine head QP | **[INFERRED]** local infeasibility lower bounds and warm-start gap brackets | **[DERIVED]** full mixed-network scale and payload-byte coupling |
| curvelet series + KKT | **[DERIVED]** approximate sparse representation plus exact reduced convex allocation | **[INFERRED]** boundary-adapted atoms priced by a reusable dual allocator | **[DERIVED]** receiver custody, equal-byte evidence, and full nonconvex response remain |

## 1. The wall after exact factorization

Let `R` include receiver rendering, camera-size uint8 realization, and the scorer's shared bilinear
resize `A`. Let `F` be the frozen SegNet up to its penultimate 16-channel, 3x3 patch, and let the five
head logits be

```text
z_c(f) = w_c^T f + b_c,              f in R^144,
C_y(mu) = {f : (w_y-w_k)^T f + b_y-b_k >= mu for all k != y}.
```

- **[MEASURED]** The centered five-row head has singular values `(3.128, 2.154, 2.025, 1.796, 0)` and
  rank-four reconstruction error `5.96e-8`; its decision quotient is exactly four-dimensional at fp32
  precision.
- **[DERIVED]** `C_y(0)` is an exact convex polyhedron described by four rival inequalities. The
  feature-space distance to leave the current winner cell is the minimum active-facet distance
  `m_yk/||w_y-w_k||`; projecting into a specified target cell is an exact small active-set QP.
- **[MEASURED]** Direct model introspection found 68 SiLU modules and 10 ReLU modules. The full frozen
  scorer is therefore neither a globally ReLU/PL network nor globally smooth: inside a fixed ReLU
  activity pattern, the SiLU path remains curved. Its input-space winner-cell pullback
  `F(R(.))^-1(C_y)` is generally piecewise-smooth, curved, and nonconvex.
- **[DERIVED]** The hard part is the pullback from an exact head target through `F o R` into legal
  witness parameters and then the uint8/archive lattice. Replacing the head description cannot by
  itself remove that inverse.

## 2. Tropical / max-plus form

### Classification: exact at the head; exact-implicit only for a PL surrogate; approximation for the actual scorer

- **[DERIVED]** For affine logits, `max_c z_c(f)` is one max-plus polynomial. Its tie locus is the
  tropical hypersurface, and its five winner cells are the same affine/Laguerre power diagram. This is
  an exact alternative description of the frozen head, not a new actuator.
- **[DERIVED]** For a feed-forward ReLU network in the theorem's scope, every output is a tropical
  rational function. A common-denominator construction can encode all class comparisons in an implicit
  tropical polynomial/rational arrangement without enumerating every linear region. The theorem is the
  one proved by [Zhang, Naitzat, and Lim (ICML 2018)](https://proceedings.mlr.press/v80/zhang18i.html).
- **[DERIVED]** An implicit tropical expression is not an inverse and is not automatically minimal.
  Redundant tropical monomials can encode the same function, common-denominator expansion can move the
  exponential complexity into expression size, and deciding a legal RGB preimage still requires solving
  the composed inequalities plus `R`, overlap, and lattice constraints.
- **[DERIVED]** A global five-site power diagram exists in the **head feature chart**. A deep PL
  composition generally subdivides input space into many cells; it does not turn the full input-space
  partition into one five-site convex power diagram.
- **[MEASURED]** The actual mixed SiLU/ReLU scorer falls outside the global PL theorem. Tropicalization
  is exact only for its affine head and for an explicitly declared PL approximation/local linear region.
- **[MEASURED]** The prior `maxplus_annulus_fit.py` K<=64 max-of-quadratics formulation reached at most
  `0.4136` annulus argmax agreement on its eight preregistered real frames and failed its `0.95` gate.
  **verdict_scope:** that K<=64, per-frame max-of-concave-quadratics annulus formulation only; other
  elements, larger K, hybrids, and local head QPs remain open.
- **[MEASURED]** #311 TropNNC then tested the stronger zero-d_seg structured-prune claim at n600 on the
  dense beta=1/tau=1 v752 trunk: `0/600` pairs retained exact SegNet argmax after even k=1 pruning, so
  exact-preserving bytes saved were zero. **verdict_scope:** structured neuron pruning on that fully-soft
  dense checkpoint; low-tau saturated checkpoints remain an explicit reactivation path.

### What this form buys

- **[DERIVED]** It buys a compact **constraint chart**: exact class-cell inequalities, exact pair
  normals, active facets, and a four-coordinate head target for QP/branch-and-bound.
- **[INFERRED]** It can reduce branching if the solver branches only on observed active facets and
  uses the exact rank-four head target before pulling back through the factorized adjoint.
- **[DERIVED]** It does **not** make the full head preimage both exact and minimal in RGB space. “One
  tropical polynomial” is exact only for the appropriate PL representation and remains an implicit
  representation whose size/minimization and preimage feasibility are separate problems.

**Build verdict:** **[INFERRED] REUSE, DO NOT BUILD A GLOBAL TROPICAL COMPILER.** Reuse
`segnet_head_rank4_linear_flipdist_v1` and the active-set target-cell QP. Add only a branch-and-bound/local
region oracle if a measured pullback needs certification. **verdict_scope:** a global compiler for this
frozen mixed-activation scorer and current witness grammar; exact affine-head charts and declared local PL
surrogates remain authorized.

## 3. Cole–Hopf, logsumexp, entropic OT, and Sinkhorn

### 3.1 Exact identities, and the point where the analogy stops

For the quadratic viscous Hamilton–Jacobi equation

```text
partial_t phi + 1/2 ||grad phi||^2 = nu Delta phi,
u = exp(-phi/(2 nu)),
```

**[DERIVED]** The transformed field solves `partial_t u = nu Delta u`, so `u(t)` is a heat-kernel
convolution and `phi(t) = -2 nu log u(t)`.

- **[DERIVED]** This linearization is exact for the quadratic Hamiltonian and compatible boundary
  conditions. It is the Cole transform's classical scope; see
  [Cole (1951)](https://www.ams.org/journals/qam/1951-09-03/S0033-569X-1951-42889-X/).
- **[DERIVED]** `L_tau(z)=tau log sum_c exp(z_c/tau)` is the entropy-smoothed maximum, and
  `softmax(z/tau)` is the exact Gibbs/simplex optimizer. It uses the same exp/log algebra as
  Cole–Hopf, but a static logsumexp does **not** prove that the witness's parameter dynamics obey the
  quadratic viscous-HJ PDE.
- **[DERIVED]** Sinkhorn solves a different exact reduced problem: entropy-regularized transport with
  a fixed cost kernel and **two prescribed marginals**, by alternating matrix scaling
  ([Cuturi 2013](https://proceedings.neurips.cc/paper_files/paper/2013/hash/af21d0c97db2e27e13572cbf59eb343d-Abstract.html)).
  A per-pixel five-class simplex has only the row constraint, so its exact solve is already softmax;
  Sinkhorn becomes nontrivial only after adding a second spatial/class-mass constraint and a cost.
- **[MEASURED]** The existing `sinkhorn_w2_mask_distortion_per_pixel` is a per-pixel categorical
  training surrogate with default class cost `1-I`; it does not move mass spatially and is not a
  conv-preimage solver.
- **[MEASURED]** The prior n600 semi-discrete OT class-mass solve converged to maximum mass error
  `2.82e-11` in eight Newton iterations yet worsened realized d_seg from `0.0031436` to `0.0048921`.
  **verdict_scope:** raw global class-frequency matching at tau=1 on the mod32cap ep650 checkpoint;
  spatially conditioned transport and preimage pullbacks remain open.
- **[DERIVED]** That negative is the key fixed-point warning: matching class marginals can converge
  exactly while landing on the wrong pointwise partition.
- **[MEASURED]** The current training flow also contains margin, phase/subpixel, length, neural
  parameterization, shared `R`, and uint8 realization; the broader family additionally includes
  eikonal and curvature terms even when their current weights are zero. These are not one quadratic
  Hamiltonian.
- **[INFERRED]** Cole–Hopf is therefore credible as a dominant-part target initializer or
  preconditioner, followed by a short full-loss trust-region repair. A claim of a full closed-form
  witness solve would be false.

### 3.2 New $0 full-n600 cache-bound measurement

**Inputs**

- **[MEASURED]** Frozen-SegNet GT logits:
  `/Volumes/VertigoDataTier/pact/lever_b_score_native_argmax_smoke_20260610/teacher_logits_n600/gt_segnet_logits.f16`,
  shape `(600,5,384,512)`, fp16, `1,179,648,000` bytes, manifest SHA-256
  `41d3ef535f5b5855fe17aab678580114a50309dc48d04948af62c2f563ed3b52`; its builder used the real video,
  `A`, and frozen CPU SegNet.
- **[MEASURED]** Exact cached hard labels:
  `gt_segnet_argmax.u8`, `117,964,800` bytes, current SHA-256
  `36c6be718916de9b0a62fec0c1229c94e38f84c3313a1fad1357c9a24eef8b68`.
- **[MEASURED]** Fixed temperature `tau=0.21682465292832676` was read from the sacred epoch-725 EMA
  checkpoint; no schedule value was guessed.
- **[MEASURED]** Measurement environment: base git `5119af53a07140311249eee85054baa4886d85ec`,
  Python `3.14.6`, NumPy `2.4.3`, `macOS-26.4-arm64`. The timer was `time.perf_counter`.

**Result**

- **[MEASURED]** A chunked NumPy Gibbs/logsumexp pass processed all `117,964,800` real n600 scorer
  pixels in `2.5133 s` (`46.94 Mpx/s`). The fp16-logit argmax differed from the u8 authority cache on
  `2,629` pixels (`2.2286e-5`), consistent with the cache manifest's fp16 tie-rounding caveat.
- **[MEASURED]** Mean top-one Gibbs probability was `0.9958448`; `2.6664%` of pixels had top-one
  probability below `0.99`; `2.2713%` had entropy above `0.1` nat.
- **[MEASURED]** Mean `L_tau(z)-max(z)` was `0.0010723`, maximum `0.2375571`, below the exact bound
  `tau log(5)=0.3489658`.
- **[MEASURED]** The sacred run's last-100 non-outlier epoch median was `119.4161 s`; a 25-epoch
  interval is therefore `2,985.4 s` **[DERIVED]** at that observed median. The target-side Gibbs solve
  used `0.0842%` of that interval **[DERIVED]**.
- **[DERIVED]** This is an n600-real **wall-localization bound**, not the requested preimage PoC: it
  proves that computing the entropy-smoothed target partition is already cheap and concentrated near
  the boundary annulus. It does not produce witness parameters whose RGB traverses `R+SegNet` to that
  partition.
- **[DERIVED]** Therefore no end-to-end training-speedup is claimed. Counting the 2.513-second target
  solve as a replacement for 25 epochs would confuse the codomain target with its preimage.

**[DERIVED] Reproduction kernel:** memmap the manifest-bound fp16 tensor as `(600,5,384,512)`; for each
frame compute `a=(z-max_c z)/tau`, `p=exp(a)/sum_c exp(a)`,
`L_tau-max=tau*log(sum_c exp(a))`, entropy, and argmax; aggregate all pixels. Quantiles in the receipt
were deterministic every-128th-pixel samples (`921,600` samples); counts, means, maxima, and wallclock
used all `117,964,800` pixels.

## 4. SDP / quadratic neural-verification relaxation

### Classification: outer relaxation and certificate; exact only for the already-affine head subproblem

- **[DERIVED]** Raghunathan, Steinhardt, and Liang construct an SDP outer relaxation for robustness of
  ReLU networks, yielding a bound on worst-case loss rather than an exact inverse
  ([NeurIPS 2018](https://proceedings.neurips.cc/paper/2018/hash/29c0605a3bab4229e46723f89cf59d83-Abstract.html)).
- **[MEASURED]** The actual frozen SegNet contains 68 SiLU and 10 ReLU modules. Directly applying the
  ReLU SDP is therefore not sound without a declared PL surrogate or certified quadratic/linear
  envelopes for SiLU and the other operators.
- **[DERIVED]** At the rank-four affine head, the minimal feature perturbation and target-cell
  projection are already closed-form/QP. An SDP is strictly more machinery and buys no tighter answer
  there.
- **[DERIVED]** For a bounded local scorer block, an SDP/QC relaxation can give a lower bound `L` on
  required visible perturbation. A rounded candidate repaired and evaluated through real `R+SegNet`
  gives a feasible upper bound `U`; `L<=opt<=U` is an honest bracket.
- **[DERIVED]** A perturbation-norm lower bound is not a payload-byte lower bound. To bound payload TTO,
  the relaxation must include a grammar-specific coefficient/count/quantization model; archive bytes
  are discrete and nonconvex.
- **[INFERRED]** A whole-frame SDP over the full U-Net activation graph is not the tractable form worth
  building here. A decomposed certificate on one hard patch, the stride-2 16-channel skip, or one
  class-pair active set could be useful for falsifying impossible local targets and evaluating the
  gap of the factorized-adjoint warm start.

**Negative verdict:** **[INFERRED] DEFER FULL-NET SDP.** **verdict_scope:** full n600 SiLU scorer and
payload objective. Local envelope certificates remain open and should be built only when a lower-bound
decision will change an allocator or carrier.

## 5. Curvelet sparse series plus KKT dual

### Classification: approximation/sparse recovery plus an exact reduced convex allocator

- **[DERIVED]** A curvelet/shearlet expansion changes the boundary representation from region/vertex
  enumeration to sparse coefficient recovery. It is an approximation form unless the finite dictionary
  spans the exact receiver image.
- **[MEASURED]** Genuine finite polar-curvelet and compact-shearlet structural receipts exist, but the
  saved-OFF n600 receiver ordering at equal scalar/support count was Fourier `0.4097223`, shearlet
  `0.4288604`, curvelet `0.5048240`; the comparison was not equal bytes, not fresh training, and not
  selection-eligible.
- **[MEASURED]** Literal curvelet training/receiver custody and a byte-matched through-R A/B remain
  blocked; the legacy Fourier control remains governed and unchanged. **verdict_scope:** the saved-OFF,
  equal-scalar/support comparison and current receiver-custody state; a receiver-closed, equal-byte
  curvelet/shearlet treatment remains open.
- **[MEASURED]** KKT reverse-waterfill is already implemented for the separable convex sensitivity/rate
  model in `frontier_exact_bitalloc.py`; its reduced allocation solve is sub-second. **verdict_scope:**
  exact for that fitted separable model, not for the unknown full nonconvex scorer response.
- **[DERIVED]** The unexploited form is a **stratum-specific union**, not one global basis: PL/dash-phase
  generators for Road-Lane discontinuities, curvelet-like atoms for `C^2` arcs after custody closure, static
  seed for Road-MyCar, and group-sparse residual atoms priced by the existing KKT dual.
- **[INFERRED]** The allocator should consume realized `delta d_seg`, nonlinear Pose score-term delta,
  and exact byte delta per atom/group, with a primal feasible archive and dual residual. Equal coefficient
  count is not an acceptable proxy for equal rate.

**Build verdict:** **[INFERRED] REUSE KKT NOW; DEFER A CURVELET DEFAULT.** The next representation build is
the per-stratum dictionary/grammar and honest receiver closure, followed by byte-matched through-R
measurement. Do not rebuild waterfill and do not infer curvelet superiority. **verdict_scope:** selecting a
global curvelet default before receiver closure and equal-byte evidence; per-stratum sparse atoms remain open.

## 6. Falsifiable end-to-end Cole–Hopf / entropic preimage PoC

**Status:** **[MEASURED] NOT FIRED** under this unit's no-training authority. The target-side n600 cache
bound in §3.2 is complete; the candidate-side preimage gate below is the only result that can authorize
the initializer.

### Fixed config

| field | preregistered value / rule |
|---|---|
| source state | **[MEASURED]** read-only stage-Muon boundary weights plus its complete resume state from the sacred run; copy before use; never mutate the run |
| data | **[MEASURED]** all 600 pairs from `gt_n600.npz`; no subset verdict |
| scorer path | **[DERIVED]** exact renderer -> `_torch_R_to_camera_uint8` -> frozen CPU SegNet; preserve batch/kernel custody and return fp32 logits before argmax |
| tau | **[MEASURED]** fixed `0.21682465292832676` for both arms; no annealing confound |
| baseline B25 | **[DERIVED]** 25 ordinary full-loss epochs from the copied state, seed 0, same optimizer state and no stage transition |
| treatment H5 | **[INFERRED]** one heat/Schrödinger target solve for the quadratic dominant part, matrix-free pullback through the existing factorized `J/J^T`, then at most 5 ordinary full-loss repair epochs |
| negative controls | **[MEASURED/DERIVED]** zero initializer; class-mass-only OT (#288 formulation); per-pixel categorical Sinkhorn (`1-I`, no spatial coupling) |
| heat kernel | **[INFERRED]** separable Gaussian/discrete heat semigroup with the treatment identification `2*nu=tau`; boundary convention and diffusion time chosen before reading treatment metrics and shared by B25's quadratic comparison |
| transport cost | **[INFERRED]** convolutional spatial kernel x explicit 5x5 class cost; no dense pixel-pixel matrix; class-only mass agreement is insufficient |
| pullback | **[DERIVED]** solve the damped least-squares target in the rank-four pair-margin chart with matrix-free CG and a trust radius; `R`, overlap, uint8, and Pose are checked only by the real forward |
| bytes | **[DERIVED]** initializer changes training wallclock only; final checkpoint/archive grammar must be identical before any score-unit comparison |

### Measurements and preregistered decision

Define `p_tau^*` from real GT logits and `p_tau(theta)` from the candidate's real through-R frozen-SegNet
logits. Define `D_tau(theta)=mean_x KL(p_tau^*(x) || p_tau(theta,x))`.

- **[DERIVED] PROCEED** to an initializer build only if, on all n600, treatment H5 satisfies
  `D_tau(H5) <= D_tau(B25)`, realized hard `d_seg(H5) <= d_seg(B25)`, the Pose score term does not
  regress beyond a same-command deterministic replay floor measured before treatment, and total H5
  wallclock is `<=0.10 *` B25 wallclock. Exact hard-map equality to B25 is recorded as a stronger pass,
  but same-or-better target debt is the adoption criterion.
- **[DERIVED] REVISE-PULLBACK** if the heat/Sinkhorn codomain target meets the soft criterion but the
  rendered candidate fails hard d_seg or Pose. **verdict_scope:** inverse realization/Jacobian/trust
  region, not entropic target construction.
- **[DERIVED] REVISE-COST** if quality passes but wallclock is above `0.10 *`; retain the math and replace
  the dense/costly operator with separable convolution, lower-rank active facets, or better CG
  preconditioning.
- **[DERIVED] FORMULATION-NEGATIVE** if only class marginals improve while pointwise `D_tau`/d_seg do not,
  or if the treatment converges to the #288-style different fixed point. **verdict_scope:** the tested
  kernel/cost/tau/checkpoint formulation only.
- **[DERIVED] NO-VERDICT** if candidate logits are not produced through the exact real path, if only
  target logits are solved, if fewer than 600 pairs are used for the final comparison, or if batch,
  seed, resume, Pose, or wallclock custody is missing.

### Harness reuse and missing build

- **[MEASURED]** Reuse `witness_annulus_convergence.render_ckpt_maps` /
  `witness_per_stage_annulus_attribution` for receiver-faithful checkpoint rendering, and reuse the exact
  `R` plus frozen-SegNet batch path from `train_levelset_witness_realized_through_R_mlx.py`.
- **[MEASURED]** Reuse `src/tac/witness_control/factorized_adjoint.py` for the matrix-free pullback and
  `segnet_head_rank4_linear_flipdist_v1` for the exact head chart.
- **[DERIVED]** The harness must add fp32 logit return, the convolutional heat/Sinkhorn target, copied
  resume-state arms, and a single receipt that binds input hashes, tau, solver residuals, full n600
  partition metrics, Pose, and wallclock. No new trainer flag is authorized before that receipt.

## 7. Ranked recommendation

1. **[INFERRED] BUILD FIRST — entropic heat target + exact-head/factorized-adjoint pullback + <=5-step
   repair.** It is the only listed new form with plausible order-of-magnitude training-wallclock leverage.
   The §3.2 bound proves the target solve is cheap; §6 prevents that cheap half from being mistaken for a
   preimage result.
2. **[INFERRED] REUSE AS THE CONSTRAINT CHART — rank-four tropical/Laguerre head active-set QP.** It is
   already exact and minimal in quotient dimension; build no global tropical enumerator. Extend only the
   local-region/branch oracle needed by the treatment pullback.
3. **[INFERRED] BUILD AFTER RECEIVER CLOSURE — per-stratum PL/curvelet sparse grammar priced by existing
   KKT waterfill.** This attacks rate after the training-cost experiment, with equal bytes and real scorer
   marginals rather than equal coefficient count.
4. **[INFERRED] DEFER — full SDP.** Use a local SDP/QC envelope only when its lower bound will decide
   whether a hard patch/carrier is impossible or whether a warm-start gap is worth closing.

## 8. Triality and DAG FEED

- **[DERIVED] Equation candidate:** `gibbs_partition_target_solve_is_not_preimage_solve_v1`:
  `softmax(z/tau)` exactly solves the per-pixel entropy-regularized codomain problem, while witness
  feasibility additionally requires membership in `range(F o R o render)` and the uint8/archive lattice.
- **[MEASURED] Candidate anchor:** full n600 target-cache Gibbs solve in `2.5133 s`, mean top-one
  `0.9958448`, with no candidate preimage produced.
- **[DERIVED] Equation-registration debt:** this memo records the candidate but does not register it as a
  verified runtime law because the candidate-side §6 gate is unmeasured. MAIN should register it only
  after the receipt distinguishes codomain solve from realized pullback; registering target-only timing as
  an end-to-end law would be false authority.
- **[DERIVED] DSL leg:** N/A-with-rationale for this advisory unit. No trainer lever may be compiled until
  §6 produces a typed, resume-safe treatment and receipt; the eventual surface is an initializer policy,
  not a raw flag.
- **[DERIVED] DAG FEED:** `FEED-alternative-forms-conv-wall-20260718` — exact rank-four tropical head is
  reused; global tropical inversion is rejected at formulation scope for the actual mixed SiLU/ReLU
  scorer; n600
  target Gibbs solve localizes the remaining cost to the preimage; build the heat/entropic target plus
  factorized-adjoint pullback and admit only on the full-n600 same-partition-in-<=0.10x gate; reuse KKT,
  keep curvelet governed, and defer full-net SDP.

## 9. Round-1 adversarial self-review

1. **[DERIVED] Attack: “The entropic solve reached the same partition.” Finding: false for this unit.**
   Only the GT codomain partition was solved. No treatment RGB was passed through `R+SegNet`; §6 remains
   `NOT FIRED` and the result is a wall-localization bound.
2. **[MEASURED/DERIVED] Attack: “Sinkhorn convergence proves d_seg improvement.” Finding: refuted by the
   existing n600 mass-match arm.** Numerical marginal convergence and pointwise partition agreement are
   different. The new gate reads pixelwise KL plus hard d_seg and retains class-mass-only as a negative
   control.
3. **[DERIVED] Attack: “logsumexp is Cole–Hopf, so the full flow linearizes.” Finding: overclaim.** The
   exp/log algebra matches, but exact heat linearization requires the quadratic viscous-HJ PDE. The neural,
   margin, topology, `R`, and lattice terms survive as residual constraints.
4. **[MEASURED/DERIVED] Attack: “The tropical map is exact for the scorer.” Finding: exact only at the
   affine head.** Zhang–Naitzat–Lim's global theorem is for ReLU/PL networks; the frozen SegNet mixes
   68 SiLU with 10 ReLU modules.
5. **[DERIVED] Attack: “One tropical polynomial is minimal and invertible.” Finding: false.** It can be an
   exact implicit PL representation, but redundancy/expression size and the input preimage remain.
6. **[DERIVED] Attack: “The SDP lower bound is a payload floor.” Finding: false without grammar coupling.**
   Norm/robustness bounds and encoded bytes are different currencies; only a grammar-specific relaxation
   plus a feasible archive produces a TTO bracket.
7. **[MEASURED/DERIVED] Attack: “Curvelets are already the winning tractable representation.” Finding:
   unsupported.** The saved-OFF row is not equal bytes or fresh training and ranks Fourier first; literal
   receiver/trainer custody remains owed.
8. **[DERIVED] Attack: “A 2.513-second target solve implies ~1,188x training speedup.” Finding: false.**
   That ratio compares different subproblems. The memo reports it only to locate the wall and grants
   speedup authority exclusively to §6.

## 10. Stores consulted and custody

- **[MEASURED]** Protocol/canonical state: `docs/operating_manual_craft_handoff.md`; `CLAUDE.md`;
  `AGENTS.md`; v7.5 §8; v8 spec; lane/subagent state; per-arm and broadcast inboxes; graph-memory recall.
- **[MEASURED]** Math/scorer state: `frozen_scorer_exact_factorization_20260715.md`;
  `segnet_recursive_fractal_factorization_20260715.md`; `necessity_solver_inverse_factorization_20260715.md`;
  `deepmath_lens_tropical_ot_powerdiagram_20260704.md`; #311 DAG closure; n600 OT-offset verdict;
  `collateral_coupling_geometry_and_film_flicker_sidecar_20260718.md`.
- **[MEASURED]** Series/allocator state: genuine curvelet/shearlet final receipt and Codex C1 audit;
  `no_fourier_basis_DAG_FEED_20260715.md`; `solve_dont_train_inventory_20260709.md`.
- **[MEASURED]** Cached inputs: full n600 frozen-SegNet logits and argmax on SSD; sacred epoch-725
  checkpoint and read-only wallclock telemetry.
- **[MEASURED]** `python3 tools/lane_maturity.py validate` reports 110 inherited missing-evidence-path
  errors in other lanes in this artifact-sparse isolated worktree; it reports no error for
  `lane_alternative_forms_conv_wall_20260718`.
- **[MEASURED]** Pointer delta is exactly zero. MAIN landing review is required before this memo is
  merged, any equation is registered, or the §6 harness becomes a build/launch surface.
