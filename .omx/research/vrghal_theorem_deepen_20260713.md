# VR-GHAL theorem-body deepening and live-family re-adjudication

**Date:** 2026-07-13  
**Mode:** `MEANS`; `research_only=true`; design/analysis plus read-only cached-artifact audit  
**Authority:** paper theorem and local source/artifact audit only; no training, paid dispatch,
live-run mutation, scorer call, archive mutation, evaluator run, or pointer mutation  
**Number labels:** `MEASURED-FROM-PAPER`, `MEASURED-ARTIFACT`, `DERIVED`, `INFERRED`,
`ASSUMED`  
**Paper:** Jelena Diakonikolas, *Solving Stochastic Fixed-Point Equations with High
Probability*, arXiv:2607.09097v1, 10 July 2026
([abstract](https://arxiv.org/abs/2607.09097),
[full paper](https://arxiv.org/pdf/2607.09097)).  
**Deepens, does not re-run:** task #462 and
`.omx/research/vrghal_95kill_fixedpoint_20260713.md`; the theorem-body node that #462
correctly left `BLOCKED/UNKNOWN` is the only recalled node being unblocked here.

## Executive answer — MEANS, not the pointer

**One-line scoped verdict:** `NO-GO-for-current-pre_se-convex-rung / DOMINATED-by-certified-
exact-solve`; `NON-DOMINATED-here: NONE CURRENTLY THEOREM-ADMITTED`; the only actual solve in
the audited stack that is forced to iterate is the **frozen-stage, frozen-replay witness-SGD
window**, and VR-GHAL remains conditional there until a fixed nonexpansive/contractive update
map and its unbiased bounded-second-moment oracle are proved.

The #462 paper-theorem node is now **MEASURED-FROM-PAPER**, not `BLOCKED/UNKNOWN`. The full
paper confirms #462's fixed-map objection to the live moving-distribution map. It also makes the
current `pre_se` family decision sharper: although that rung uses a fixed replay and a convex
objective, it does not need a stochastic fixed-point solver. It constructs deterministic
sufficient statistics and solves the retained normal equations directly by symmetric EVD and
Moore--Penrose inversion, with a stored optimum certificate for every head.

## 1. Exact theorem domain and assumptions

All statements in this section are **MEASURED-FROM-PAPER** from arXiv:2607.09097v1.

### Fixed operator and oracle

The object is one fixed operator `T:E->E` satisfying

```text
||T(x)-T(y)|| <= gamma ||x-y||,     gamma in (0,1].
```

For `gamma=1`, the paper assumes that a fixed point exists. Oracle draws are i.i.d. from a fixed
distribution, independent of the algorithmic past, and `tau(x;xi)` is jointly measurable and
Bochner integrable. In the native norm,

```text
E[tau(x;xi)] = T(x),
E[||tau(x;xi)-T(x)||^2] <= sigma^2             for every x.
```

The base theorem does **not** assume cocoercivity. It assumes a fixed contractive or nonexpansive
operator. Cocoercivity appears only in the paper's discussion of an application equivalence in
Hilbert/Euclidean settings.

### Space geometry

`E` is a real separable quadratically smoothable Banach space. There is a compatible norm
`||.||_+` and constants `C>=1`, `B>0` for which

```text
||x|| <= ||x||_+ <= C ||x||,
||x+y||_+^2 <= ||x||_+^2 + D(||.||_+^2)(x)[y] + B||y||_+^2,
kappa_E := B C^2.
```

The squared compatible norm is continuously Frechet differentiable. This is not an ignorable
technicality: `kappa_E` appears explicitly in the concentration and residual constants.

### Optional stronger oracle assumptions

1. **Assumption 1, multi-query:** the same seed `xi` can be queried at two or more points.
2. **Assumption 2, expected Lipschitzness:**
   `E||tau(x;xi)-tau(y;xi)||^2 <= L^2||x-y||^2`.
3. **Assumption 3, samplewise regularity:** for almost every `xi`,
   `||tau(x;xi)-tau(y;xi)|| <= gamma||x-y||`.

Assumptions 1--3 are improvements to the base bounded-variance result, not hidden premises of the
base theorem.

## 2. Exact clipped-difference estimator and VR-GHAL recursion

This section is **MEASURED-FROM-PAPER**. To undo a typography collision in Algorithm 3, write
`bar_lambda_k` for the paper's pre-normalized schedule and `lambda_k` for the actual Halpern
anchor weight. Write `bar_gamma in [gamma,1]` for the supplied upper bound used by clipping.

### Clipped difference — the material correction to #462

For an unbiased stochastic difference

```text
Delta(x,y) = tau(x;xi_1) - tau(y;xi_2),
E Delta(x,y) = T(x)-T(y),
```

the paper defines

```text
Cl_bar_gamma Delta(x,y)
  = min{1, bar_gamma ||x-y|| / ||Delta(x,y)||} Delta(x,y),
```

with value zero when `Delta=0`. Under Assumption 1 the two oracle calls share a seed; otherwise
their seeds are independent. The radius is exactly `bar_gamma||x-y||`: there is no free
calibration constant `c`. Thus #462's generic `R_k=c gamma ||x-y||` was properly labeled a
reconstruction, but it is superseded as a transcription of this paper.

If `E||Delta-E Delta||^2 <= sigma_xy^2`, Lemma 5 gives the exact constants

```text
||Cl-E Cl|| <= 2 bar_gamma ||x-y||,
||E Cl-E Delta|| <= sigma_xy,
E||Cl-E Cl||^2 <= 9 sigma_xy^2.
```

### Recursive estimator

For epoch `k`, inner step `j>=1`, and copy `ell`, the minibatch difference is

```text
Delta_(k,j)^(ell)
  = (1/m_(k,j)) sum_i [tau(y_j;xi_i^(1))-tau(y_(j-1);xi_i^(2))].
```

The epoch-start estimate is a high-probability mean estimator, concretely MoME:

```text
Ttilde_(k-1,0) = MoME(epsilon_k/2, delta_k/2, xhat_(k-1)).
```

It is then updated by `n_k` independently formed clipped minibatch differences:

```text
Ttilde_(k-1,j)
  = Ttilde_(k-1,j-1)
    + (1/n_k) sum_(ell=1)^n_k Cl_bar_gamma Delta_(k,j)^(ell).
```

The MoME call certifies
`||Ttilde_(k-1,0)-T(xhat_(k-1))|| <= (epsilon_k/2)sqrt(kappa_E)sigma`
with failure probability `delta_k/2`, and uses
`N_k=O(log(1/delta_k)/epsilon_k^2)` oracle calls. This last statement is asymptotic; the paper
does not expose a universal numeric leading constant for `N_k`.

The time-uniform Banach-space martingale inequality used by the estimator has the paper's exact
displayed constants: with probability at least `1-delta`, simultaneously for every prefix `m`,

```text
||S_m|| <= sqrt(2 kappa_E ln(2/delta) sum_i sigma_i^2)
           + C ln(2/delta) ||b_(1:n)||_infinity / 3.
```

### Algorithm 3

Inputs are `x_0`, `beta in (0,1)`, `delta in (0,1)`, `bar_gamma in [gamma,1]`, and epoch count
`K`. Initialization is

```text
delta_0=(1-beta)delta/beta,  epsilon_0=1,  bar_lambda_0=1,
xhat_0=x_0,
Ttilde_(0,0)=MoME(beta epsilon_0/2, beta delta_0/2, xhat_0).
```

At epoch `k>=1`:

```text
epsilon_k    = beta epsilon_(k-1),
bar_lambda_k = beta bar_lambda_(k-1),
lambda_k     = bar_lambda_k/(1+bar_lambda_k),
delta_k      = beta delta_(k-1),
y_0          = xhat_(k-1),
J_k          = ceil[ln(beta)/ln((1-lambda_k)bar_gamma)].
```

For `j=0,...,J_k-1`,

```text
y_(j+1) = lambda_k y_0 + (1-lambda_k) Ttilde_(k-1,j),
```

followed by the recursive estimator update above. Then `xhat_k=y_(J_k)` and the next epoch's
MoME anchor is constructed. There is no per-iteration SGD step-size schedule: the gradual
schedule is geometric in `epsilon_k`, confidence budget, and the pre-normalized Halpern anchor.

## 3. High-probability constants and rates

### Master condition and anytime bound

Let `sigma_(k,i)^2` bound the conditional central second moment of one minibatch difference and
let

```text
Rtilde_k := ||Ttilde_(k,0)-xhat_k||.
```

**MEASURED-FROM-PAPER, Assumption 4:** the minibatches must ensure

```text
sum_(i=1)^J_k sigma_(k,i)
 + 3 sqrt(2 kappa_E ln(4/delta_k) sum_(i=1)^J_k sigma_(k,i)^2)
 <= epsilon_k (a_0+a_1 sqrt(k)).
```

Theorem 1 sets

```text
n_k = max{1,
          ceil[2(1-lambda_k)bar_gamma Rtilde_(k-1) ln(4/delta_k)
               /(3 epsilon_k)]}.
```

Define

```text
A_0   = ||T(xhat_0)-xhat_0|| + beta epsilon_0 sqrt(kappa_E)sigma/2,
A_1   = 2 bar_lambda_0 ||x*-x_0|| + beta epsilon_0 sqrt(kappa_E)sigma/2,
A_2   = epsilon_0 [C + sqrt(kappa_E)sigma/2 + a_0],
A_5/2 = (4sqrt(2)/5) epsilon_0 a_1.
```

On one event of probability at least `1-delta`, simultaneously for every epoch,

```text
Rtilde_k <= beta^k [A_0 + A_1 k + A_2(k+2)^2 + A_5/2(k+2)^(5/2)].
```

For target residual `epsilon`, let

```text
L_p=max{1, log_(1/beta)(8A_p/epsilon)},  p in {0,1,2,5/2}.
```

Then `||T(xhat_K)-xhat_K||<=epsilon` with probability at least `1-delta` whenever

```text
K >= ceil max_p {L_p + p log_(1/beta)L_p + O_beta(1)}.
```

The event is independent of `K`, which is the paper's anytime guarantee. **Custody limit:** the
displayed `A_p`, factors `2`, `3`, `4`, `8`, and `4sqrt(2)/5` are exact. The final epoch threshold
still contains an unspecified `O_beta(1)`, described as universal when `beta` is bounded away
from one. Therefore it would be false to claim a completely numeric end-to-end epoch constant.

### Concrete minibatch schedules

**MEASURED-FROM-PAPER, bounded variance only (Lemma 7):** with
`rho_k=bar_gamma(1-lambda_k)`, it suffices to use

```text
m_(k,i)=ceil(4J_k^2/epsilon_k^2),
a_0=sqrt(kappa_E)sigma[1+3sqrt(2kappa_E)sqrt(ln(4/((1-beta)delta)))],
a_1=3sqrt(2)kappa_E sigma sqrt(ln(1/beta)).
```

**MEASURED-FROM-PAPER, multi-query plus expected Lipschitzness (Lemma 8):** it suffices to use

```text
m_(k,j)=ceil(4 Rtilde_(k-1)^2/[epsilon_k^2(1-rho_k)^2]),
a_0=sqrt(L^2+gamma^2)sqrt(kappa_E)
    [1+3sqrt(2kappa_E)sqrt(ln(4/((1-beta)delta)))],
a_1=3sqrt(kappa_E)sqrt(2(L^2+gamma^2)ln(1/beta)).
```

For samplewise `gamma`-Lipschitz oracle calls, clipping never activates and the paper states that
`m_(k,j)=ceil(4Rtilde_(k-1)^2/[epsilon_k^2(1-rho_k)])` suffices.

### Oracle-complexity corollaries

For `beta=1/2`, `epsilon_0=bar_lambda_0=1`:

```text
bounded variance, D=||x*-x_0||+sigma:
  Otilde_(kappa_E,ln(1/delta))[
    D^2/epsilon^2 + min{D^6/epsilon^5, D^3/((1-gamma)^3 epsilon^2)}]

expected Lipschitz, D_L=||x*-x_0||+sigma+sqrt(L^2+gamma^2):
  Otilde_(kappa_E,ln(1/delta))[
    D_L^2/epsilon^2 + min{D_L^6/epsilon^3, D_L^3/(1-gamma)^3}]

samplewise Lipschitz, D=||x*-x_0||+sigma:
  Otilde_(kappa_E,ln(1/delta))[
    D^2/epsilon^2 + min{D^4/epsilon, D^3/(1-gamma)}].
```

These are **MEASURED-FROM-PAPER rate expressions**, but `Otilde` suppresses logarithmic factors
and constants. They are not fully explicit numerical oracle budgets.

## 4. Re-adjudication against the live `pre_se` convex rung

### What the rung actually solves

**MEASURED-ARTIFACT/CODE.** The protected `pre_se_locus_20260713` receipt is a fixed V9 `n600`
replay: `480` immutable cached train targets plus `120` fresh exact heldout costates, with `600/600`
teacher starts and zero retries. At each of two loci, each of `20` ordered class-pair heads forms
deterministic RankRLS sufficient statistics. Thus there are `40` convex certificates.

For each head, code forms a symmetric quadratic

```text
q_b(w)=w^T G_b w - 2 r_b^T w + c_b,
```

symmetrizes `G_b`, computes `G_b=U diag(mu) U^T` with `numpy.linalg.eigh`, retains eigenvalues
above `eps * width * mu_max`, and returns the minimum-norm retained-space optimizer

```text
w_b*=U_+ diag(mu_+)^-1 U_+^T r_b.
```

All `40/40` blocks record `normal_equation_optimum_certified=true`. The present rung has
rank-truncated Moore--Penrose solves and **no positive Tikhonov ridge**; “ridge” is a related
family description, not the current receipt. The heavy-tail sibling correctly keeps a future
positive-ridge reliability row separate from this MP optimization certificate.

The full-space gradient can retain a null-space component when the accumulated RHS is not exactly
in the numerical range of `G_b`; the rung therefore certifies the retained normal equation, which
is precisely its preregistered finite-precision objective. This is an exact one-shot numerical
solve for that objective, not a claim of symbolic real-arithmetic exactness.

### Dominance derivation

**DERIVED.** One could invent a fixed-point map such as `T_eta(w)=w-eta(G_bw-r_b)`, but that does
not create a stochastic-oracle need. `G_b` and `r_b` are already materialized deterministic
sufficient statistics. The EVD returns `w_b*` directly and exposes a first-order certificate.
Replacing it with VR-GHAL would add:

1. a stochastic oracle and its variance estimate where none is required;
2. clipping bias and a nonzero failure budget `delta` where the current solve is deterministic;
3. inner/outer iterations and `Otilde` oracle costs where the current solve is one shot; and
4. a residual certificate weaker than the available retained-space normal-equation certificate.

Therefore VR-GHAL buys **nothing** on the current convex rung and is strictly dominated by the
available direct solve at the actual `188`/`332` feature widths and fixed `n600` cache. This
algorithmic verdict does not alter the rung's measured modeling verdict: convex retained mass is
`0.20233024422907497` at block2 and `0.09314654496850622` at block3, both below the preregistered
`0.47` gate.

**Scoped verdict:** `NO-GO-for-the-convex-rung`,
`verdict_scope=FORMULATION x CURRENT-FIXED-N600-PRE_SE-CONVEX-RUNG x SOLVER-SELECTION`.

**req-R reactivation evidence:** re-open VR-GHAL for a convex head only if all of the following
become true: the exact sufficient statistics/direct factorization violate a recorded memory or
wall-clock budget; deterministic exact-enough sparse/direct/Krylov alternatives do not satisfy
the same certificate; only an unbiased stochastic oracle is affordable; one fixed `T` with
`gamma<=1`, the Banach geometry, and the native-norm second moment are custodied; and a matched
cached replay shows lower wall time or query cost at no worse objective/certificate. A larger
future dataset alone is not sufficient reactivation evidence.

## 5. Where an iterative stochastic fixed-point solve is actually forced

| Audited locus | Fixed map? | Iteration forced? | VR-GHAL status |
|---|---:|---:|---|
| current `pre_se` convex rung | yes | no | **DOMINATED** by the certified MP solve |
| task #454 costate reuse gate | no fixed-point problem | no | **INAPPLICABLE**; this is a refresh/safety decision, and clipping can hide the drift that should trigger refresh |
| task #455 on-policy lambda/costate ORGAN map | no | potentially | **NO-GO under paper theorem**; the witness-induced sample distribution and teacher target move with the witness |
| costate-reuse iteration | no | potentially | **NO-GO under paper theorem**; reused costates are deliberately refreshed as the witness leaves the trust region |
| witness descent inside one frozen stage/replay/config window | conditionally | **yes** | **ONLY CANDIDATE NON-DOMINATED LOCUS**, but not currently theorem-admitted |

**MEASURED-CODE.** The witness trainer actually performs repeated pair-sampled gradient updates;
there is no closed-form optimizer for the nonlinear renderer/witness parameters. Thus the only
audited place where solving is genuinely forced to iterate is a narrowly named
`frozen-stage/frozen-replay/fixed-loss witness-SGD solve window`.

**DERIVED theorem gate.** Across curriculum boundaries, optimizer transitions, costate refreshes,
and on-policy witness changes, the map is moving. Even inside one frozen window, the live witness
objective is nonconvex and no artifact proves that its stochastic update map is fixed and
`gamma`-Lipschitz with `gamma<=1` in a chosen native norm. No artifact also establishes the exact
unbiased-oracle and bounded-second-moment premises for the proposed update map. Hence
`NON-DOMINATED-here: NONE CURRENTLY THEOREM-ADMITTED` is the honest current answer.

**req-R for the only candidate:** freeze replay, stage, loss weights, optimizer geometry, and
teacher semantics; define the exact update operator; prove a registered trust region on which it
is averaged/nonexpansive or contractive; measure `bar_gamma`, `sigma`, `kappa_E`, and oracle
unbiasedness in the native norm; then compare VR-GHAL with the incumbent deterministic/full-batch
and stochastic optimizer at matched realized-through-R downstream debt. Until those receipts
exist, VR-GHAL is a conditional method family, not an actuator recommendation.

## 6. Complementarity with the heavy-tail fold

The concurrent Zhu--Lu fold in
`.omx/research/heavy_tail_interp_fold_20260713.md` governs the **estimator/statistical leg**: rare
bad fits and hard-edge sensitivity of ridgeless versus fixed-positive-ridge interpolation, with
literal rates explicitly withheld from Pact's regime-mismatched designs. VR-GHAL governs a
different **algorithmic iteration leg**: conditional on a fixed nonexpansive/contractive operator
and an unbiased noisy oracle, it bounds the fixed-point residual over iterative oracle calls with
high probability. These legs compose by allocating failure probability to estimator error and
optimization residual separately; they do not substitute for one another. On the current one-shot
MP `pre_se` solve, the iterative optimization-error leg is absent, so VR-GHAL contributes zero,
while the heavy-tail fold can still motivate a separate positive-ridge reliability comparison.

## 7. Triality, provenance, and pointer honesty

- **Equation leg:** `vrghal_high_probability_fixed_operator_law_v2` supersedes only #462's
  generic reconstructed clipping radius/recursion as a paper transcription. It does not
  supersede `EQ-VRGHAL-455-MOVING-OPERATOR-DEBT-v1` or
  `EQ-VRGHAL-455-QUERY-TO-TEACHER-v1`.
- **DAG leg:** `.omx/research/vrghal_theorem_deepen_DAG_FEED_20260713.md` changes
  `PAPER_2607_09097_THEOREMS` from `BLOCKED/UNKNOWN` to `MEASURED` and encodes the direct-solve
  dominance edge. Shared canonical DAG mutation is deferred to main review.
- **DSL leg:** no DSL/config edge is admitted. No flag, launcher, controller, or live actuation was
  added.
- **n600 authority:** the `n600` cache/receipt is authoritative only for the stated fixed-replay
  solver-selection and retained-mass facts. It is not a contest score or live pointer.
- **Pointer delta:** `NONE`.
- **STORES CONSULTED:** full arXiv v1 PDF/HTML; task #462 memo/equation/DAG artifacts;
  `pre_se_locus_20260713` source, solver source, tool, and protected receipt; task #454 trust-region
  and Jacobian-drift memos; task #455 on-policy memo; costate-reuse memo; witness trainer source;
  current heavy-tail fold; lane registry and subagent progress ledger.
