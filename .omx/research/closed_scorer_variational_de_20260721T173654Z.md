# Closed scorer variational differential equations — task-space closure

UTC 2026-07-21T17:36:54Z · lane `lane_closed_scorer_variational_de_20260721` ·
`research_only=true` · pointer delta **ZERO** · **MAIN landing review required**.

## Verdict

**D1 is CONFIRMED on its stated surface:** the frozen SegNet final-head decision is an exact
rank-4 Laguerre/power diagram and, equivalently, a negative-entropy Bregman Voronoi decision.
The independent native-f32 power assignment and Bregman vertex assignment each had **0 / 20,480**
disagreements against live frozen-SegNet argmax on 20 seed-1234 held-out real tiles. This confirms
the task-coordinate factorization, not inverse reachability from legal archive bytes.

**D3 is fail-closed:** the exact minimum under `B <= 154,600` and sub-0.15 reachability are
**UNRESOLVED_REQUIRES_BYTE_CLOSED_WITNESS**. The prior `S_floor ~= 0.118` is **REFUTED AS AN
ESTABLISHED EXACT MINIMUM**: its cited 177,169-byte achiever is outside this byte cap, and the same
source explicitly says there is no nontrivial proved Kolmogorov lower bound. It remains a useful
empirical achiever/rate reference, not a theorem or a feasible capped witness.

No score was measured, no provider/GPU work was dispatched, and the `[contest-CPU]` pointer remains
`0.1910828242` from `reports/latest.md`.

## D1 — closed functional on the right task space

Let `C` be a legal counted archive description, `G(C)` its deterministic receiver output, and
`R_8` the exact resize/round/uint8 realization chain. Define the coupled task map

```text
(q(C), xi(C)) = (Pi_4 h_seg, h_pose) o R_8 o G(C),
```

where `q_p in R^4` is the quotient coordinate of the frozen SegNet final affine head at output
pixel `p`, and `xi in R^(N x 6)` is the first-six frozen PoseNet output. The closed action is

```text
S[C] = 100 D_seg(q(C), c*) + sqrt(10 D_pose(xi(C), xi*))
       + 25 L_MDL(C) / 37,545,489,

D_seg = |Omega|^-1 sum_p 1[L(q_p) != c*_p],
D_pose = (6N)^-1 ||xi - xi*||_2^2,
L_MDL(C) = exact legal archive bytes (or an exact entropy-code length plus measured container residual).
```

This is a closed finite-dimensional task functional. The CNNs occur only in the coupling map that
determines feasible `(q,xi)`; they are not expanded into a load-bearing symbolic 200-layer ODE.

### Rank-4 Laguerre / power-diagram segmentation term

For the final affine logits `z_c(q)=a_c dot q+b_c`, set

```text
s_c = a_c / 2,       omega_c = b_c + ||s_c||^2.
```

Then

```text
argmax_c z_c(q)
= argmax_c [2 q dot s_c + omega_c - ||s_c||^2]
= argmin_c [||q-s_c||^2 - omega_c].
```

Thus each target class constraint is the intersection of half-spaces

```text
(a_c* - a_j) dot q + (b_c* - b_j) >= 0,   for every rival j,
```

and the global segmentation debt is the measure of disagreement with these Laguerre cells.

### Bregman-Voronoi form and Fisher locality

For negative entropy `F*(p)=sum_c p_c log p_c`, `p=softmax(z)`, and class vertex `e_c`,

```text
D_F*(e_c || p) = KL(e_c || p) = logsumexp(z) - z_c.
```

The minimum-debt vertex is exactly `argmax z_c`; pulling that decision back through the rank-4
affine head yields the same Laguerre partition above. The local Hessian of log-sum-exp is categorical
Fisher, so margin/Fisher curvature co-locates at the cell boundary. Finite KL, a dual-Euclidean
metric, and a Fisher-natural inverse are not interchangeable. This landing preserves the V9 label
`GAUGE_IDENTITY_VERIFIED_NOT_MODEL_FACTORIZED`.

### Pose quadratic and coupling

In scorer-output coordinates the pose debt is exactly quadratic, not merely approximated:

```text
D_pose(xi,xi*) = (6N)^-1 ||xi-xi*||^2,    Hessian_xi D_pose = I/(3N).
```

For decoder coordinates `theta`, its local pullback is Gauss-Newton
`J_xi(theta)^T J_xi(theta)/(3N)` plus second-order residual terms. An `SE(3)` twist interpretation
requires a measured chart/coupling; it is not inferred from the six output numbers alone.

### D1 empirical fidelity gate

Command:

```text
/Users/adpena/Projects/pact/.venv/bin/python tools/verify_closed_scorer_variational_de.py \
  --upstream /Users/adpena/Projects/pact/upstream \
  --gt-cache /Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --seed 1234 --heldout-tiles 20 --tile-size 32 \
  --out .omx/research/closed_scorer_variational_de_20260721T173654Z.json
```

Measured `[macOS-CPU advisory real frozen-SegNet tiles; NON-PROMOTABLE]`:

| check | residual |
|---|---:|
| native-f32 rank-4 power vs live argmax | 0 / 20,480 = 0 |
| generic-f64 rank-4 power vs live argmax | 0 / 20,480 = 0 |
| negative-entropy Bregman vertex vs live argmax | 0 / 20,480 = 0 |
| cached target vs live batch-1 argmax, diagnostic only | 0 / 20,480 = 0 |

Frames were 228, 584, 591, and 597; five disjoint deterministic 32x32 tiles were selected per
frame. The algebraic comparisons use the live captured head features, so they establish final-head
closure. They do not establish a task-space inverse, receiver feasibility, exact bytes, n600 batch-32
authority, or score.

## D2 — Euler-Lagrange / KKT system

The constrained problem is

```text
min_C S[C]
subject to
  q_p(C) in P_{c*_p}                         (target argmax cells),
  ||xi(C)-xi*||^2/(6N) <= rho_pose^2         (pose tube),
  L_MDL(C) <= 154,600                        (hard byte cap),
  C in A_legal, (q,xi)=T(R_8(G(C)))          (receiver/coupling feasibility).
```

On a differentiable fixed-cell stratum, with hard-cap multiplier `mu_B >= 0`, stationarity in a
shared code chart is

```text
0 in 100 partial_C D_seg
     + sqrt(10)/(2 sqrt(D_pose)) J_xi^T (xi-xi*)/(3N-normalization)
     + (lambda_rate + mu_B) partial_C L_MDL
     + N_F(C),

lambda_rate = 25/37,545,489 = 6.658589531221714e-7,
mu_B (L_MDL-154,600) = 0.
```

`lambda_rate` is the fixed objective byte price. It is **not** the KKT multiplier of the hard cap;
that multiplier is `mu_B`. Treating them as the same number would be a units and complementarity bug.

The three coupled differential laws are:

1. **Segmentation / level set.** Within a diffuse relaxation, class potentials obey a viscous
   Hamilton-Jacobi descent
   `partial_tau phi_c + H_c(grad phi; lambda_rate,mu_B) = epsilon Delta phi_c`.
   Target cell walls are `phi_c=phi_j`; the `epsilon -> 0+` limit is interpreted as a viscosity
   solution/subgradient flow. Raw argmax is piecewise constant, so a globally smooth ODE is invalid.
   Realization forces must use `flip_margin_step_law_v1` (first-order + secant + QP), not a bare
   first-order image gradient.
2. **Pose / SE(3).** In a calibrated twist chart `g in SE(3)`, the pose update is a geodesic
   trust-region flow `nabla_tau dot g = -grad_g S_pose`, with `xi=log(g*^-1 g)`. The scorer-output
   quadratic is exact; the `decoder -> R_8 -> PoseNet -> xi` pullback remains empirical.
3. **Rate / coder.** For entropy symbols, optimal ideal lengths satisfy `ell_j=-log2 p_j` under
   Kraft equality and reverse-waterfill admits a symbol only while measured marginal
   `-Delta distortion / Delta byte > 25/37,545,489`. Exact archive length, parse-back, headers, and
   codebook payload close the residual between entropy and real bytes.

At cell interfaces, KKT is a normal-cone/subgradient inclusion; across integer code-length changes it
is a mixed discrete-continuous condition. `stationarity_residual()` therefore labels its output
`SMOOTH_STRATUM_ONLY` and refuses the square-root derivative at zero debt with nonzero residual.

## D3 — analytic solution, bounds, and reachability

Analytically closed quantities:

```text
rate price per byte = 25/37,545,489
                    = 6.658589531221714e-7 exactly as a rational coefficient;

rate at 154,600 B  = 25*154,600/37,545,489
                    = 0.10294179415268769;

sub-0.15 distortion allowance at the cap
                    = 0.15 - rate
                    = 0.04705820584731231.
```

Therefore a 154,600-byte witness reaches sub-0.15 iff

```text
100 D_seg + sqrt(10 D_pose) < 0.04705820584731231.
```

For any smaller exact byte count `B`, replace the right side by
`0.15 - 25B/37,545,489`. This condition is necessary and sufficient **once a legal receiver witness
and its realized debts exist**; it is not an existence proof.

The exact constrained minimum is

```text
S*_{154600} = inf { S[C] : C in A_legal, L_MDL(C)<=154,600 }.
```

Current certified bracket is deliberately weak:

- universal lower bound: `S* >= 0` (DERIVED; no nontrivial conditional-Kolmogorov lower bound exists);
- inspected under-cap advisory incumbent: the 91,062-byte Einstein-Kolmogorov v3 packet scores
  `35.955425463668846` on the retained `[macOS-CPU advisory]` n600 receipt, so it is only a finite
  feasibility/upper-bound witness on that advisory axis, not a competitive or promotable bound;
- no exact archive <=154,600 with `S<0.15` is present in the consulted stores.

A numerical optimizer cannot close `S*` without selecting and measuring the legal description
language `A_legal`, the receiver map, and the exact MDL residual. Substituting a proxy entropy or the
unconstrained 177,169-byte thought-experiment achiever would fake the answer. The executable
`reachability_certificate()` therefore returns `numeric_infimum=None` in the receipt.

### Comparison with `S_floor ~= 0.118`

- `25*177,169/37,545,489 = 0.11796956486570198` is correct arithmetic.
- The cited achiever is 22,569 bytes above this task's cap.
- Its zero-distortion value is a hypothetical/empirical achiever upper bound, not a lower bound on all
  legal programs; the source explicitly calls the true conditional Kolmogorov minimum uncomputable.

Verdict: **REFUTE_AS_ESTABLISHED_EXACT_MINIMUM**; retain only
`EMPIRICAL_ACHIEVER_RATE_REFERENCE_AT_177169_BYTES_OUTSIDE_CAP`. This tightens epistemic scope, not
the numeric lower bound.

## D4 — route into `einstein_kolmogorov_ultra`

| ultra stage | canonical equation | obligation |
|---|---|---|
| U1 stationarity DE | `closed_scorer_viscosity_kkt_stationarity_v1` | Solve viscosity/subgradient Seg force + calibrated SE(3) pose pullback + exact-rate KKT in one decoder chart. |
| U2 lower bound | `closed_scorer_archive_reachability_bound_v1` | Produce a valid relaxation/lower bound for an explicit legal decoder language; never promote proxy entropy as archive bytes. |
| U3 `S*` / reachability | `closed_scorer_archive_reachability_bound_v1` + exact evaluator | Emit archive <=154,600, parse-back/runtime custody, realized n600 Seg/Pose, exact CPU/CUDA replay, or preserve `UNRESOLVED`. |

U1 consumes the D1 EmpiricalAnchor with residual 0 on 20 real tiles. That anchor licenses the frozen
head coordinate system only. U2/U3 remain gated on receiver/rate custody.

## Triality and system wire-in

- **DSL:** no new launch flag. The task map is callable through
  `tac.canonical_equations.closed_scorer_variational_de_20260721`; any future launcher must bind it
  through typed DSL/LawRef rather than inventing CLI flags.
- **DAG:** `closed_scorer_variational_de_DAG_FEED_20260721T173654Z.md` routes U1 -> U2 -> U3 and
  preserves the archive/receiver gates.
- **Equations:** three append-only canonical registry rows carry executable evaluators and the D1
  empirical anchor.
- **Sensitivity map:** cell-interface force uses the rank-4 winner/rival margin and the corrected
  realization secant/QP law.
- **Pareto / bit allocator:** exact byte price is the reverse-waterfill stop rule; entropy estimates
  are inadmissible without exact-byte residual closure.
- **Autopilot:** research-only U1/U2/U3 route; no dispatch authorization.
- **Continual learning:** the D1 residual is a typed EmpiricalAnchor; D3 stays awaiting verification.
- **Probe disambiguator:** finite Bregman vertex KL and Fisher-natural pullback remain separate APIs;
  exact archive bytes arbitrate entropy versus real rate.

## Stores consulted

- `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, `docs/operating_manual_craft_handoff.md` (fully read)
- `reports/latest.md`
- `.omx/state/lane_registry.json`, `subagent_progress.jsonl`,
  `master_gradient_anchors.jsonl`, `modal_call_id_ledger.jsonl`,
  `cost_band_posterior.jsonl`, `continual_learning_posterior.jsonl`
- latest Codex findings/session summary, latest T3 council, V9 design, v7.5/v8 SPECs
- `canonical_equations_registry.jsonl`, `probe_outcomes.jsonl`, `canonical_task_status.jsonl`
- `frozen_scorer_exact_factorization_20260715.md`,
  `information_theoretic_floor_report_v1_20260610T102335Z.md`,
  `einstein_kolmogorov_crux_v3_20260720.json`
- real frozen scorer weights and `gt_n600.npz` named in the JSON receipt
- per-arm and broadcast inboxes before checkpoints

## Reactivation / promotion gate

Promotion requires all of: an explicit legal archive language, exact archive <=154,600 bytes,
deterministic receiver parse-back and runtime custody, exact realized-through-R n600 Seg/Pose,
stationarity or constructive witness evidence, and contest-CPU/CUDA replay. Until then the family is
open, the numeric minimum is unresolved, and the pointer is unchanged.
