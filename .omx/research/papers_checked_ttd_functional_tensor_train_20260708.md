# Papers-checked: Tensor Train Diffusion (TTD / FTT-HJB sampler) — 2026-07-08

**Paper:** Gruhlke, Berner, Sommer, Richter, *"Tensor Train Diffusion: Leveraging Low-Rank
Structures for High-Dimensional Score-Based Sampling"* (ICML 2026; OpenReview DDQX97Xi1Z; code
`github.com/robertgruhlke/TTD`). Read: main body pp.1–9 + appendix pp.13–35 (all pages) + repo
skim (structure/solver/bases verified via WebFetch).

**One-line verdict:** NOT-A-LEVER for the current witness. TT-as-training-substrate is
NOT-APPLICABLE (two independent derivations); the low-rank RATE lever is already armed (D18) but
DOMINATED (rate near floor); the ONE positive draw-from is a *discipline* refinement to the
already-registered #341/#342 solve-don't-train surface (adaptive-Tikhonov + sample-complete
full-P regression = the named cure for #341's measured k=8 subset-overfit). Pointer **0.19110
UNMOVED** — this is research/means; nothing here fires a launch or touches the v7/#205 endgame
config. `[no-triality]` (negative verdict + the positive grain routes to already-registered
surfaces; no new law — see §5).

**Stores consulted (proactive recall):** CLAUDE.md §papers-checked anti-re-research +
§WITNESS-CAPSTONE + NO-FAKE #6/#8; the DAG FEED-05z (#318 DE-derivation), FEED-07u (#341/#342
TerminalSolve + SOLVE-DON'T-TRAIN inventory); `basin_finisher_head_solve_probe_measured_20260707.md`
(#341, MEASURED); `mod_dim_dynamics_telemetry_20260708.md` (landed today — effective_rank/k90 +
`k90_truncate_bytes_estimate` + deferral D18); `reference_papers_checked_not_relevant_or_watch_item_ledger_20260701.md`
(GaussianQuant/KSI/Nielsen sisters, the "rate near i.i.d. floor" + "surrogate≠authority through
argmax" verdicts); MEMORY L55/L77.

---

## 1. What the paper ACTUALLY does (mechanism, not abstract)

Goal: sample from an unnormalized density `ρ_target` (no samples available, `Z` intractable) by
time-reversing a noising OU SDE. The reverse drift needs the score `∇log p(x,t)`; via the
**Hopf–Cole** transform the log-density `V := −log p̃` satisfies a **Hamilton-Jacobi-Bellman**
PDE (Lemma 2.1, eq 4). Fokker-Planck is the *linearization* of that HJB, so the optimal control
(= score) is written through **backward SDEs (BSDEs)** → per-step **regression** problems.

The solver (Algorithm 1, backward-in-time; Alg 2/3 ALS):
1. Partition `[0,T]` into a grid `t_0<…<t_N`. Terminal condition `V(·,T) = −log ρ_target`.
2. March BACKWARD: for each interval, the previously-computed `V̂(·,t_{n+1})` is the terminal
   condition for solving `V̂(·,t_n)`. Each step is a discrete BSDE loss (eq 9/10).
3. **The load-bearing fact:** the discrete loss is **AFFINE (linear) in the coefficient tensor**
   `C` — "the loss has the convenient property that it is almost surely zero at the solution…
   the affine dependence in V̂_n makes it well-suited for regression-based methods" (p.4). Even
   the gradient-dependent BSDE loss stays linear because `(Id + Σ·∇)` is a linear operator on
   `C` (eq 92–95). **So every fit is a regularized LINEAR LEAST-SQUARES solve, not SGD.**
4. **FTT format:** `V(x) ≈ C[Φ(x)] = Σ_α C[α] φ_α(x)`, with `C` a functional tensor train (TT
   ranks `r`) and `φ` a **tensor-PRODUCT** of univariate bases (Legendre / B-spline / Fourier /
   extended-Fourier), `H = H²_mix` Sobolev-mixed-smoothness-2. Storage `O(d·max(m)·max(r)²)` —
   linear in dimension `d`.
5. **Solve = ALS** (`ttd/solvers/als.py`, PyTorch): sweep the TT cores; each micro-step shifts
   the orthogonal core to position `i` and solves `min_C ||A_i·vec(C) − y||² + τ||C||²` (eq
   16/78/95) — a local linear normal-equations solve. Stack principle reuses left/right
   contractions (O(d) speedup).
6. **Three adaptivities:** τ_n (adaptive Tikhonov, eq 101, `τ_n = γ·‖A c − y‖²/‖C‖²`, γ=0.1);
   **rank** via HOSVD singular-value thresholding (Alg 4, δ=1e-4, Dörfler marking); basis-degree.
7. **Rank theory (A.7):** Gaussians have FTT rank `r≡2` (isotropic) or bounded by subdiagonal-block
   ranks of the precision matrix (Thm A.5). **Smoothness → low rank** (anisotropic mixed Sobolev
   `r_i = ⌈ε^{−1/β_i}⌉`, A.7.2). Product functions have rank 1. Non-smooth / axis-misaligned
   structure → HIGH rank (the theory explicitly does NOT cover discontinuous targets — future-work
   caveat p.9: FTT ranks for HJB solutions "only partially understood," unbounded-domain theory
   missing).

**Measured claims:** Multiwell (d=10/50), Ginzburg-Landau (φ⁴), Kitagawa. Converges in **<3 outer
iterations**; significantly faster + more stable than neural samplers (DIS/PIS) at equal ESS/logZ
error; the whole appeal is *"circumvents SGD-based optimization with long training times and
hyperparameter sensitivity."*

---

## 2. Honest relevance map (MEASURED / DERIVED / INFERRED / ASSUMED)

Our witness = a coordinate-INR amortizing the **frozen SegNet argmax partition** (piecewise-constant
codim-1 separatrix, ~8-dim lane manifold), scored `d_seg` through the R operator + frozen CPU-torch
scorer. It is NOT a density-sampling problem; there is no `Z`, no reverse SDE, no score-to-sample.
The HJB resonance is with our LEVEL-SET/eikonal PDE (#318), not with the paper's density-HJB — same
PDE class, different `V`.

### (a) Solve-don't-train — which of our blocks match the "linear-after-Hopf-Cole" structure — **DERIVED → DRAW-FROM-NOW (discipline), feeds #342**
The paper's decisive property is *loss affine in the coefficient tensor → per-core regularized
linear least-squares*. Mapping onto our #342 blocks:
- **#341 quadratic head** (`out_sdf/out_tex/palette`, ~791 affine params): **MEASURED** near-quadratic
  (Levenberg ρ = 0.847/0.868, `basin_finisher_...20260707`). This IS our instance of the paper's
  linear-after structure. **But the probe MEASURED the failure = k=8 SUBSET OVERFIT** (+5.2%
  held-out d_seg on 592 pairs). The paper AVOIDS exactly this by (i) `K≈2^15` samples per regression
  and (ii) **adaptive Tikhonov τ_n** (eq 101) sized relative to the data term. That is the *named
  cure* for #341's measured negative: the only admissible form (per the probe) was already "full-P
  in-trainer GPU," and the paper adds the *regularization discipline* that makes a large-sample
  solve well-posed. **Concrete #342 row:** the three-condition solvability test (in-chart convexity
  × weak coupling × fixed topology) gains a **FOURTH condition made explicit by TTD — SAMPLE-COMPLETE
  regularized LS** (enough samples relative to solved DOF + adaptive Tikhonov); #341's negative lives
  precisely on this axis (791 params ≫ 8 pairs).
- **FiLM gains given frozen features:** the render is NOT affine in the gains *through argmax* — #341
  explicitly EXCLUDED FiLM gains ("not affine"). A logit-surrogate would be linear, but
  surrogate≠authority through argmax. → not the linear-after structure end-to-end.
- **per-class λ costate:** adjoint/backward — see (d).

> **Verdict (a):** DERIVED — TTD's "affine-in-coefficient → per-core regularized linear
> least-squares" is exactly #341's measured near-quadratic head chart, and its adaptive-Tikhonov +
> large-sample regression is the named cure for #341's MEASURED k=8 subset-overfit → **DRAW-FROM-NOW**
> as the full-P in-trainer TerminalSolve discipline (add "sample-complete regularization" as the 4th
> #342 solvability condition), NOT a new solver.

### (b) TT/low-rank as a RATE lever — **DERIVED → REGISTERED-duty-to-measure (D18), but DOMINATED**
Counted bytes ARE weights (rule 118). Post-hoc TT/SVD decomposition of the trained witness's weight
tensors at Δd_seg≈0 is the sister of #311 (TropNNC) / #157 (sensitivity bit-alloc) / #308 (grids-vs-INR).
The **mod-dim telemetry landed today** already: measures `effective_rank` (~17.8 autopsy), `k90` (~20 ≈
Whitney 2·8+1), emits `k90_truncate_bytes_estimate = round(code_bytes·k90/mod_dim)` (~37% at 20/32),
and armed **deferral D18** (truncate-code-to-k90 byte-close A/B, fires at v7 stop). TTD *validates the
criterion*: its rank truncation is exactly SVD-singular-value-thresholding on reshaped cores (Alg 4,
δ=1e-4) — the same math as k90/effective-rank.
- **Honest caveat resolved:** TT ranks blow up on axis-misaligned *output* discontinuities (A.7.2),
  but D18/k90 compress the **smooth latent/weight table**, NOT the discontinuous partition — the
  argmax discontinuity is applied at the END, after the smooth INR. So SVD/TT-truncation of the
  weights is legitimate; the mod-dim spectrum is the right feasibility instrument.
- **Why DOMINATED:** rate sits near the i.i.d. entropy floor (~6.66e-7 S/byte; the DAG "rate DEAD"
  relabel + the GaussianQuant/Delétang ledger entries) — **d_seg is the wall, not rate.** Same crux
  that made GaussianQuant a WATCH-not-GO.

> **Verdict (b):** DERIVED — post-hoc SVD/TT-truncation of the witness's smooth latent/weight tensors
> (not the discontinuous output) is legitimate and ALREADY armed as **D18 + the emitted
> `k90_truncate_bytes_estimate`**; TTD's HOSVD singular-value-threshold validates the truncation
> criterion → **REGISTERED-duty-to-measure** (D18, fires at v7 byte-close), but **DOMINATED** because
> rate is near the entropy floor while d_seg is the binding wall.

### (c) TT as a TRAINING substrate (replace the MLP) — **DERIVED → NOT-APPLICABLE**
Replacing the coordinate-INR with an FTT `V̂_C[Φ(x)]` over pixel coords fails on two independent
grounds:
1. **Dimensionality collapse.** TT's entire value proposition (dimension-free MC rates, `O(d)`
   storage vs curse) is for HIGH `d`. Our render coordinate space is `d=2` (pixel x,y) [+ a little
   conditioning]. The paper's own statement: `d=1` = a linear basis, **`d=2` = plain SVD**. At `d=2`
   the tensor-train machinery *degenerates to a matrix SVD* — no curse-of-dimensionality win exists
   to harvest. The witness's genuinely high-dim object is the ~8-dim lane manifold, but that is an
   EMERGENT structure, not a coordinate grid one regresses a value function over.
2. **Basis anti-match on the separatrix.** The FTT uses **axis-aligned tensor-PRODUCT** bases
   (Legendre/B-spline/Fourier ⊗). A curved codim-1 separatrix oriented at an angle to the pixel axes
   has HIGH/full TT rank (the classic TT failure on rotated discontinuities; A.7.2's smoothness→low-rank
   theory explicitly EXCLUDES discontinuous targets; Fig 7 shows splines failing on localized
   features, Fourier getting Gibbs). This **directly contradicts our MEASURED result**: the
   all-class-**DIRECTIONAL** (anisotropic/curvelet) Fourier basis gives **−48% d_seg** — i.e. the
   right chart is boundary-tangent-ORIENTED, precisely the non-separable structure the tensor-product
   FTT cannot represent without rank blowup. The FTT would need to REDISCOVER the curvelet/directional
   basis its format forbids.

> **Verdict (c):** DERIVED — **NOT-APPLICABLE** as an MLP replacement, on two independent grounds:
> (i) the render coordinate space is `d=2` where TT degenerates to plain SVD (the paper's own `d=2`
> case) so there is no curse-of-dimensionality win, and (ii) the axis-aligned tensor-product FTT basis
> has full/blowup rank on a curved, axis-misaligned codim-1 separatrix (A.7.2 excludes discontinuous
> targets), contradicting our MEASURED −48% all-class-DIRECTIONAL/curvelet result.

### (d) BSDE/backward-regression for the COSTATE controller (#247/#303) — **INFERRED → NOT-APPLICABLE (thin WATCH grain)**
The λ dynamics ARE adjoint/backward (Pontryagin costate), and TTD's backward-in-time BSDE iteration
is the canonical numerical scheme for exactly such backward equations — the structural analogy is
real. BUT: our costate is a **~5-D-per-step advisory SCHEDULE** (one λ per class over training
steps), not a high-dimensional value-function solve; the FTT low-rank machinery is overkill (`d=5`,
no curse), and the controller's job is to RANK never-fired levers (duty-to-measure), not to solve a
`V`. Our costate is also advisory-only by containment (no autonomous heavy-GPU). Only ONE grain
transfers: *solve the λ schedule backward-from-a-terminal-condition rather than forward-heuristically*
— a note for the #247 shadow controller, not a draw-from.

> **Verdict (d):** INFERRED — the analogy is real (our λ costate is an adjoint/backward equation; TTD's
> BSDE backward-iteration is the canonical solver for such equations), but **NOT-APPLICABLE** because
> the costate is a ~5-D-per-step advisory schedule, not a high-dim value-function solve (FTT overkill);
> the only transferable grain is a backward-from-terminal-condition discipline for the #247 shadow
> controller (WATCH, not draw-from).

---

## 3. Papers-checked ledger line (L55 index-fold format, for main to fold into MEMORY.md)

```
TTD/FTT(ICML26, tensor-train HJB density sampler)=NOT-A-LEVER: TT-TRAIN N/A (render d=2→SVD, no curse-win; axis tensor-product basis vs our MEASURED −48% all-class-DIRECTIONAL/curvelet separatrix — A.7.2 excludes discontinuous targets) · solve-don't-train CONFIRMS #341 (loss-affine→linear-LS = our head chart; adaptive-Tikhonov+K≈2^15 = named cure for the MEASURED k=8 subset-overfit → 4th #342 solvability cond: sample-complete regularization) · low-rank RATE=D18-armed (k90/SVD-truncate validated by their HOSVD-threshold) but DOMINATED (rate≈entropy floor, d_seg is wall) · costate-BSDE=WATCH(adjoint analogy real, but λ is 5-D advisory schedule not value-fn solve)
```

---

## 4. What we should NOT do (anti-shiny-object)

- **Do NOT rebuild the witness as an FTT.** That is a from-scratch vehicle swap = the exact
  "capacity-sweep / wrong-vehicle" trap the operating manual §8.2 names; forbidden mid-endgame, and
  independently NOT-APPLICABLE per §2(c). The crucible v7/#205 endgame is the critical path.
- **Do NOT open a rate campaign off this.** Rate is near the i.i.d. floor; TT-weight-compression is
  DOMINATED (same verdict as GaussianQuant). D18 already captures the only measurable form; it fires
  at v7 byte-close, not before.
- **Nothing here preempts v7.** The one positive grain (a) is a *discipline* note that composes with
  the already-gated #341 Phase-2 (full-P in-trainer TerminalSolve) and adds one condition to #342 —
  a next-run/terminal-solve item, not a v7-config change and not a launch.

## 5. Triality note

`[no-triality]` — negative verdict (per the negative-verdict registration rule in the basin-finisher
memo), and the one actionable positive routes to ALREADY-registered surfaces (#341 TerminalSolve DSL
primitive + #342 inventory + D18 deferral + the mod-dim telemetry's `k90_truncate_bytes_estimate`).
No new law is DERIVED that isn't already an anchor: the "sample-complete regularization" fourth
solvability condition is an APPEND to #342's existing three-condition test (a task refinement, not a
canonical equation), and (c)'s TT-rank-blowup-on-directional-separatrix is a NEGATIVE that the
negative-verdict rule says not to register. If #342 is later formalized as a canonical equation, the
4th condition folds in there.

Pointer contest-CPU **0.19110 UNMOVED** — research/means only.
