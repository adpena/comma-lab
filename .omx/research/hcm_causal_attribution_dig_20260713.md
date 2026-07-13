---
title: "Hierarchical causal attribution for Pact run telemetry: identification audit, CONFOUND-L4, and organ credit"
date_utc: "2026-07-13"
lane_id: "lane_hcm_causal_attribution_dig_20260713"
research_only: true
status: "DESIGN_ANALYSIS__UNCOMMITTED_MAIN_REVIEW"
identified_existing_lever_queries: 0
confound_l4: "DERIVED_GRAPH_FALSIFICATION_LAYER__NOT_CONFOUND_CERTIFICATION"
organ_partial_pooling: "CONDITIONAL_GO_PREDICTIVE_GATE__NO_GO_CAUSAL_CREDIT_CURRENT_N1"
canonical_equation: "NONE__IDENTIFICATION_LAW_DID_NOT_CLOSE"
score_claim: false
pointer_delta: "NONE"
---

# HCM causal-attribution dig: no free run-level lever effects, but a real L4 and a better organ estimator

## Answer first

> **{identified-queries verdict: `COUNT=0` for existing run-level lever→ΔS effects; top-three duty queries `DsegAwareTaper`, `HorizonWeightedMargin`, and `StepNativeActivation` are all `NO-GO` under the current corpus because the treatment is outside the inner plate and the run-level backdoor/positivity conditions do not close + confound-L4 check derived: `YES`, as a cross-fitted whole-run mechanism-invariance/negative-control falsification test, not a certificate that no hidden confounder exists + organ partial-pooling verdict: `CONDITIONAL GO` for shrinkage and uncertainty in the existing predictive backtest, `NO-GO` for current causal campaign credit or self-activation at one real trajectory; compose after FORE support correction, never instead of it}.**

`DERIVED` — Weinstein and Blei's hierarchy advantage is real, but it is treatment-location
specific. Their positive identification result exploits treatment variation *inside* a unit. Pact's
named levers are selected once per run and are constant across that run's pairs and epochs. The
paper's no-benefit result for unit-level treatments therefore applies as the structural warning;
the Pact query-specific graph then supplies the decisive failure: an unblocked
`A_r <- U_r -> Y_r` path with no observed adjustment set, no valid instrument/front-door route, and
zero treatment support for two of the three highest-ranked candidate formulations.

`DERIVED` — No canonical equation is registered in this unit. An identification equation would be
fake authority because the causal law did not close. The equation leg is explicitly
`N/A-with-reason`, while the graph-falsification statistic and partial-pooling model remain
design equations inside this memo.

`MEASURED` — This unit launched no trainer, scorer, evaluator, provider, GPU job, or process. It
performed read-only source/corpus inspection and writes only this memo plus its standalone DAG feed.
It makes no evaluator score claim and moves no frontier pointer.

## 0. Epistemic contract and source boundary

Every substantive claim below begins with one of the required labels:

- `MEASURED`: read directly from named paper/repository bytes or computed read-only from current
  local files in this unit.
- `DERIVED`: follows from a stated graph, theorem, or algebraic manipulation.
- `INFERRED`: a Pact mapping that is plausible but needs a schema or empirical gate.
- `ASSUMED`: a proposed modeling/design choice, not established fact.

`MEASURED` — Primary source: Eli Weinstein and David M. Blei, *Hierarchical Causal Models*, JMLR
27(37), 2026: [JMLR article](https://www.jmlr.org/papers/v27/25-0899.html),
[official PDF](https://www.jmlr.org/papers/volume27/25-0899/25-0899.pdf), and
[authors' code](https://github.com/EWeinstein/HCM). The source was read for the HCM definition,
collapse construction, Assumptions 1--3, Proposition 7, Theorems 6/12/13, hierarchical empirical
distribution rates, limitations, simulations, and eight-schools reanalysis.

`MEASURED` — The official code repository currently exposes reproduction notebooks for the
simulations and eight-schools analysis. It is source evidence for the paper's examples, not a
drop-in causal-attribution implementation for Pact.

`MEASURED` — The sibling FORE memo was present and read:
`.omx/research/fore_occupancy_ratio_dig_20260713.md`. Its current-organ verdict is that the one
deterministic regime sequence is adequate for the stated forecasting gate but does not identify
causal off-policy value for unlogged schedule arms; Markov transition custody, target-action
support, and cross-run support are missing.

## 1. What the HCM theorem actually buys—and what needs units

### 1.1 Population identification versus estimation

`MEASURED` — The paper's nonparametric identification setup assumes access to the joint law of
observed unit variables and the unit-specific subunit distribution. Its learnability discussion
takes both the number of units `N` and subunits per unit `M` to infinity. The collapse theorem uses
`M -> infinity` to turn each unit's empirical subunit distribution into a random distribution-valued
node; learning the population law over those nodes still needs `N -> infinity`.

`MEASURED` — For the hierarchical empirical distribution, the paper gives two distinct sampling
debts. For nested test functions the mean-squared error is of order

\[
O\!\left(\frac{1}{N}+\frac{1}{M}\right),
\]

and a Wasserstein statement has separate unit and subunit rates. Large `M` pays the within-unit
term; it does not erase the `1/N` term.

`DERIVED` — Identification is a property of the population graph/law, not a blessing conferred by
a Bayesian prior at finite sample size. Hierarchical Bayes can regularize a weakly estimated but
identified functional. It cannot convert an unblocked backdoor path or a structural zero in
treatment support into identification.

`MEASURED` — The paper's asymptotic/structural roles separate as follows:

| paper component | what rich within-unit data can pay | what still needs units |
|---|---|---|
| Theorem 6 collapse | `M -> infinity` identifies a unit's subunit-law node under mechanism convergence | it does not learn the population distribution over unit mechanisms |
| Assumption 1 / collapsed observational law | increasingly accurate unit-specific empirical distributions | `N -> infinity` is still used to learn their joint population law with unit variables |
| Assumption 2 / subunit positivity | within-unit variation can establish support for a treatment that actually varies by subunit | it says nothing about a treatment constant inside every unit |
| Proposition 7 / collapsed do-calculus | permits ordinary graphical identification on the collapsed graph after the collapse and positivity assumptions | population identification still inherits the unit-level graph and support |
| Theorem 12 hierarchy advantage | under the theorem's no-hidden-subunit-confounding setup, exploits a treatment inside the plate when either no prohibited bidirected path connects it to a direct unit descendant or a valid subunit instrument is available | effect estimation/generalization still needs a population of units |
| Theorem 13 unit treatment | supplies no within-unit natural experiment | the erased-plate unit-treatment nonidentification remains |
| hierarchical Bayes | reduces variance and shares strength in a finite parametric model | does not alter any graphical identification or positivity condition |

`MEASURED` — The collapsed node `Q_r` is a distribution-valued summary of subunit behavior. Thus
the theory can avoid retaining every raw row once the empirical measure is faithfully represented;
it does **not** say that an arbitrary scalar run mean, last epoch, or nine-interval digest preserves
all identifying information. Pact's current aggregates cannot be promoted to a sufficient `Q_r`
without proving that sufficiency for the named query.

### 1.2 Which HCM result exploits within-unit structure

`MEASURED` — The paper's positive hierarchy theorem (Theorem 12) concerns a *subunit-level*
treatment. Under the theorem's graph conditions, hierarchy can identify an intervention that the
erased-inner-plate graph cannot, notably through within-unit treatment variation or a subunit
instrument. The construction is a natural experiment inside each unit.

`DERIVED` — The exact first failed condition for the Pact lever query is type-level:
Theorem 12 requires the intervention node `A` to be a subunit variable inside the plate, whereas
`A_r` is a unit variable. Its positive routes therefore never start. In the present graph the
prohibited unit-level backdoor is concrete, and the paper's unit-positivity requirement additionally
fails wherever an exact candidate has zero support conditional on the target run parents.

`MEASURED` — The paper's Theorem 13 is explicitly a no-benefit result for a *unit-level* treatment:
when the erased-inner-plate effect is not identified, adding the hierarchy does not identify the
full observed postintervention distribution. The paper's intuition is exact for the present crux:
there is no within-unit randomization of a treatment that is constant for the whole unit.

`DERIVED` — Theorem 13 alone is not used to overclaim that every imaginable scalar functional is
nonidentified; its statement is about the observed postintervention distribution and the paper
notes functional-specific qualifications. The zero-count verdict here is query-specific: in the
Pact graph below, each named scalar `Delta S` query retains an unblocked run-selection backdoor,
and no alternate identification route is present in the recorded variables.

`MEASURED` — The paper assumes exchangeable units and exchangeable subunits for this theory and
lists hidden subunit confounding, selection, interference, and nonexchangeability among its
limitations.

`DERIVED` — Pact epochs are ordered states with feedback, not exchangeable subunits. Pair outcomes
also share one trained trunk: a gradient contribution from pair `p` changes later outcomes for
other pairs. Therefore the faithful mapping is an inner plate over pairs plus a longitudinal state
DAG, with interference carried by the shared training state. Treating `600 pairs x epochs` as iid
inner-plate replication would violate both the causal order and no-interference premise.

## 2. The Pact HCM and its inner-plate graph

### 2.1 Variables

`ASSUMED` — For run `r`, pair `p`, and ordered epoch/verdict boundary `t`, define:

- `A_r`: the run-level lever/configuration treatment whose causal effect is queried.
- `Z_r`: observed pretreatment run parents: base checkpoint and hashes, declared config, seed,
  machine/backend/axis, dataset/order manifest, and planned stage schedule.
- `U_r`: unobserved run-level selection and state: operator choice, unrecorded initialization,
  stale controller belief, runtime load, and other causes of both `A_r` and outcomes.
- `B_p`: pair identity/baseline difficulty and scorer-margin features, with source custody.
- `H_rt`: shared longitudinal training state before row `t`: checkpoint/EMA lag, stage, optimizer
  moments, spike-guard and accepted-update state, temperatures, controller memory, and prior history.
- `Y_rpt`: latent realized-through-R pair outcome, including per-pair `d_seg`, `d_pose`, and a
  preregistered contribution to the score law.
- `G_rt`: apparatus state that should be represented by the declared graph: frozen-parameter
  status, `weights_stepped`, guard path, scorer/cache/version, and measurement mode.
- `Y_tilde_rpt`: recorded measurement after the apparatus channel.
- `Q_r`: the collapsed HCM distribution-valued summary of pair outcomes/mechanisms for run `r`;
  this is not the apparatus variable `G_rt`.

### 2.2 Graph

`ASSUMED` — The smallest honest graph is:

```text
OUTER PLATE: run r

       U_r ----------------> A_r ----------------------+
        |                    |                          |
        |                    v                          v
        +-----------------> H_r,t ------------------> Y_r,p,t ----> Y_tilde_r,p,t
        |                    ^       ^                  ^                 ^
        |                    |       |                  |                 |
       Z_r ------------------+       |                 B_p               G_r,t
        |                            |                  |
        +--------------------------> +------------------+
                                     |
                              Y_r,*,t-1 / optimizer history

                    INNER PLATE p = pair identity
                    ---------------------------------
                    | B_p -> Y_r,p,t -> Q_r        |
                    ---------------------------------

                    t is an ordered longitudinal DAG, not an iid plate.
```

`DERIVED` — `A_r` sits outside the pair plate. Conditional on a run, all pair rows have the same
treatment. Repeating `Y_rpt` over 600 pairs or many epochs estimates a run's response surface and
apparatus mechanism more precisely, but creates no additional assignment of `A`.

`DERIVED` — The causal path of interest is `A_r -> H_rt -> Y_rpt -> Delta S_r`. The confounding
path `A_r <- U_r -> H_rt/Y_rpt` remains open. Stage, EMA, and spike-guard state are not a generic
adjustment cure: much of `H_rt` is post-treatment, so conditioning on it can block part of the
effect or open collider paths. This is the precise HCM form of the config-orphan confound class.

`DERIVED` — A stage transition that happens at different epochs is not automatically a valid
subunit treatment. It is a deterministic descendant of the run schedule and prior trajectory.
Identifying its effect needs longitudinal sequential exchangeability/positivity or an actual
within-run randomization, neither of which follows from the HCM hierarchy.

`INFERRED` — A future pair-randomized loss intervention could place a treatment inside the pair
plate, but only if the intervention has a stable receiver meaning and interference through the
shared trunk is either eliminated, explicitly modeled, or bounded. Random pair weights applied to
one common optimizer generally violate the simple subunit no-interference graph.

## 3. Existing-corpus audit

### 3.1 What is present

`MEASURED` — The current duty-to-measure apparatus returned 72 actual owed DSL levers. Its merged
ranking surface returned 77 rows because it also includes significance records not registered as
owed DSL levers. The current top three actual duty rows are:

| rank | query | current evidence label | current ledger state | causal question |
|---:|---|---|---|---|
| 1 | `DsegAwareTaper` | `ESTIMATED Delta S=0.03`; convergence revalidation owed | `fired-unmeasured` | `E[Delta S_r | do(A_r=1)] - E[Delta S_r | do(A_r=0)]` |
| 2 | `HorizonWeightedMargin` | `MEASURED oracle ceiling midpoint=0.018`, not a treatment effect | `never-fired` | same run-level contrast |
| 3 | `StepNativeActivation` | `MEASURED screen estimate=0.013`; adopt verdict owed | `never-fired` | same run-level contrast for the exact registered formulation |

`MEASURED` — Read-only launch-script inspection found 90 `launch.sh` files, 32 containing
`--dseg-aware-taper`, zero containing `--seg-horizon-margin-weight`, and zero matching the exact
registered beta-end-8 StepNative candidate. Launch-script presence is configuration evidence only;
dry starts, descendants, and unmatched co-treatment runs are not completed exchangeable causal
units.

`MEASURED` — Read-only inspection found 78 `experiments/results/**/telemetry.jsonl` files containing
94,185 valid JSON rows and no parse failures. Of those, 91,046 are epoch-indexed rows across 21
stage names; 91,046 carry `loss_components`, 30,772 have a nonempty value, and 17,203 have a
nonempty per-axis decomposition.

`MEASURED` — In that shared telemetry schema, zero rows contain any of `pair`, `pair_idx`,
`pair_index`, `d_seg`, `d_pose`, `loss_terms`, `seed`, `machine`, `lever`, `levers`, or
`active_levers`. Specialized pair artifacts and level-set daemon/verdict logs do exist elsewhere,
but the common `telemetry.jsonl` surface is not a standardized causal panel joining pair outcomes,
treatment, pretreatment parents, apparatus state, and run custody.

`DERIVED` — Consequently the paper's Assumption 1—the observable joint law needed by the collapsed
model—is not available nonparametrically from the common telemetry surface. This is an estimation
and schema failure in addition to, not a substitute for, the graph-identification failure.

### 3.2 Identification verdict on the top three

| query | HCM identification verdict | exact failed condition | verdict scope |
|---|---|---|---|
| `DsegAwareTaper -> Delta S` | **`NO-GO`, not identified** | treatment is unit-level; `A <- U -> Y` remains open; the 32 scripts do not form a matched completed treatment/control set with only this lever changed; conditioning on descendant stage/EMA/guard is invalid as blanket adjustment | `FORMULATION x EXISTING HETEROGENEOUS CORPUS x RUN-LEVEL TREATMENT`; taper family and a governed matched A/B remain open |
| `HorizonWeightedMargin -> Delta S` | **`NO-GO`, not identified** | unit-level treatment plus empirical positivity failure: exact treatment support is zero in inspected launch scripts, `P(A=1 | target stratum)=0` | `FORMULATION x EXISTING CORPUS x NEVER-FIRED EXACT ARM`; margin family remains open |
| `StepNativeActivation -> Delta S` | **`NO-GO`, not identified** | unit-level treatment plus empirical positivity failure for the exact registered formulation; other HOSC beta schedules are different treatments and cannot be silently pooled | `FORMULATION x EXISTING CORPUS x EXACT REGISTERED STEP-NATIVE ARM`; activation family remains open |

`DERIVED` — Identified existing-corpus lever queries: **`COUNT=0`**. The violated hierarchy
condition common to all three is that the treatment does not vary inside the inner plate. The
query-specific backdoor and positivity failures then prevent even ordinary flat adjustment.

`DERIVED` — The 600 pair rows may reduce conditional measurement noise within a run. They cannot
reduce uncertainty about the treatment assignment mechanism below what the number and overlap of
independent runs permit. The effective treatment-replication count is the number of compatible
run assignments, not the number of pair/epoch rows.

### 3.3 What can still be estimated without lying

`DERIVED` — Existing bytes can support descriptive associations, within-run response trajectories,
pair-difficulty maps, loss-term closure checks, and prediction of later telemetry from earlier
telemetry. Those are useful for sensing and acquisition but are not `do(A)` effects.

`ASSUMED` — A descriptive hierarchical model could be

\[
\widetilde Y_{rpt}
\sim t_\nu\!\left(
  \alpha_r + \delta_p + f_g(t,H_{rt}) + A_r\beta_g + \gamma^\top Z_r,
  \sigma_g
\right),
\tag{1}
\]

with run, pair, and regime effects. Under the present graph, posterior `beta_g` is an associational
regularized coefficient. It must not be reported as a causal lever effect.

`DERIVED` — No hierarchical-Bayes estimator for the requested causal top-three is supplied because
no top-three functional is identified. Producing one would replace an identification failure with a
prior-sensitive number and violate NO-FAKE.

## 4. CONFOUND immune system L4

### 4.1 Verdict

`DERIVED` — **YES, a principled L4 can be derived**, with a narrow meaning:
`HCM-GRAPH-FALSIFICATION`. Do-calculus itself is an identification calculus, not a hidden-confound
detector. The assumed HCM/collapsed graph nevertheless entails conditional-independence,
exchangeability, and mechanism-invariance restrictions. Their violation proves that the declared
graph/schema/apparatus model is wrong on the tested surface.

`DERIVED` — L4 cannot certify that no hidden confounder exists. A hidden cause can be observationally
equivalent, a graph may be saturated, and finite tests can miss small violations. Therefore:

```text
L4 fires     -> graph/apparatus validity REFUSED; L3 verdict clearance withheld.
L4 is quiet  -> tested implications not rejected; no "unconfounded" certificate.
```

### 4.2 Concrete telemetry-row check

`ASSUMED` — For a future standardized telemetry row, declare before fitting:

- causal parents `X_rpt=(A_r,Z_r,B_p,H_rt)` allowed to predict `Y_tilde_rpt`;
- apparatus/negative-control tags `N_rt` that the valid graph says have no residual path to the
  outcome after conditioning, such as scorer/cache fingerprint, frozen-parameter sentinel,
  measurement worker, or a guarded no-update mode;
- the exact outcome vector, including pair `d_seg`, pair/pairwise `d_pose` attribution where valid,
  and typed loss terms/weights;
- whole-run folds; no random row split.

`DERIVED` — Fit the conditional mean only on training runs and form held-out-run residuals:

\[
\widehat m^{(-r)}(X)=E[\widetilde Y\mid X],
\qquad
R_{rpt}=\widetilde Y_{rpt}-\widehat m^{(-r)}(X_{rpt}).
\tag{2}
\]

For every preregistered negative-control feature `h_j(N)`, the valid graph implies the conditional
moment

\[
E\!\left[R_{rpt}\,h_j(N_{rt})\right]=0
\quad\text{after conditioning on the declared parents.}
\tag{3}
\]

`DERIVED` — A concrete scalar discrepancy is

\[
T_{\mathrm{L4}}
=\max_{j,g}
\frac{\left|\sum_r\sum_{p,t}R_{rpt}h_j(N_{rt})1\{G_{rt}=g\}\right|}
     {\widehat{\mathrm{SE}}_{\mathrm{whole\mbox{-}run}}(j,g)},
\tag{4}
\]

augmented by preregistered residual scale, lag-1 autocorrelation, and pair-rank discrepancies.
Calibrate (4) by permutation/wild bootstrap of *whole runs*, never by treating pair/epoch rows as
iid. Control the familywise or false-discovery error across the fixed diagnostic vector.

`DERIVED` — A typed algebraic apparatus sentinel should run in the same layer:

\[
C_{rt}=L_{rt}^{\mathrm{total}}-\sum_k w_{k,rt}L_{k,rt}.
\tag{5}
\]

When the DSL/compiler declares exact weighted-term closure, nonzero `C_rt` beyond serialization
tolerance or systematic dependence of `C_rt`/`R_rpt` on frozen, `weights_stepped`, spike-guard,
machine, cache, or scorer-version tags refuses apparatus validity. Equation (5) is an integrity
identity, not causal identification; it complements the graph moments in (3)--(4).

`ASSUMED` — The positive-control sentinel is the known meta-confound class: a frozen/no-update
record that a controller would otherwise label "converging." The L4 implementation must flag its
residual/mechanism break at a preregistered threshold before it is trusted on real confound hunts.

`DERIVED` — Apparatus validity is a precondition. If term closure, source hashes, treatment capture,
or pair-outcome custody fail, the test returns `INVALID_INPUT`, not a p-value. This places L4 before
L3 verdict clearance and preserves the existing L1 runtime alarms and L2 strict gates.

### 4.3 Current-instance disposition

`MEASURED` — The inspected common telemetry rows lack the pair outcomes, treatment capture, seed,
machine, and apparatus fields required by (2)--(5); `loss_components` also lacks a cross-corpus
typed guarantee that permits reconstructing exact weighted closure.

`DERIVED` — L4 design verdict: `YES`. Current shared-corpus execution verdict:
`NO-GO`, `verdict_scope=FORMULATION x CURRENT COMMON TELEMETRY SCHEMA`. Specialized artifacts may
be adapted after a schema/custody audit; the HCM/negative-control family remains open.

## 5. Organ credit assignment and partial pooling

### 5.1 What partial pooling can improve today

`MEASURED` — The sealed organ envelope describes one real trajectory, one vehicle, one regime
sequence, and initially eight intervals; the current ledger contains repeated snapshots of the
same `levelset_v752_baseline_20260710T185913Z` trajectory at nine intervals and three named regimes.
Repeated ledger blocks are re-analyses/snapshots, not additional independent runs.

`MEASURED` — The current regime dispatcher reports seven walk-forward folds, a nominal dispatcher
MAE of 0.001596 versus 0.001852 for its global-single-best forecaster and 0.002792 for persistence,
but only three wins and one loss on the differing folds (`p about 0.31`) and an in-sample-derived
policy. These are `[macOS advisory] NON-PROMOTABLE` predictive results, not causal schedule effects.

`DERIVED` — Partial pooling is well suited to the *predictive* gate: it shrinks noisy
regime-by-forecaster contrasts toward a family mean, represents paired errors on the same outcome,
and returns honest posterior uncertainty. It is not allowed to convert repeated folds into new
trajectories.

`ASSUMED` — For predictive arm `a`, regime `g`, real run `r`, and interval `i`, use a paired robust
model such as

\[
e_{rgia}=\ell(\widehat y_{rgia},y_{rgi}),
\qquad
e_{rgia}\sim t_\nu(\alpha_{rgi}+\theta_{ga},\sigma_g),
\tag{6}
\]

\[
\theta_{ga}\sim N(\mu_a+x_g^\top b_a,\tau_a^2),
\qquad
\mu_a\sim N(0,s_a^2).
\tag{7}
\]

The shared `alpha_rgi` preserves paired comparisons across forecasters. A longitudinal
AR/state-space residual is required if interval errors remain autocorrelated.

`DERIVED` — At one run, `tau_a`, run variance, and regime-transfer behavior are weakly or not
separately learned; posterior rankings will be prior-sensitive. Thus partial pooling is a
`CONDITIONAL GO` for conservative ranking/uncertainty in the existing backtest and a `NO-GO` for
autonomous activation, graduation, or cross-run causal claims.

`DERIVED` — The eight-schools analogy is useful only after keeping identification separate from
estimation. In the HCM reanalysis, within-school treatment variation supplies causal information and
hierarchical Bayes pools school-specific effects. Pooling is not what creates the treatment
variation. Pact currently has the pooling opportunity but not the analogous causal assignments.

### 5.2 Campaign outcome credit

`MEASURED` — The existing campaign-outcome-credit design is explicitly the Monte-Carlo-return
complement to the local adjoint/costate signal. Existing memos already state that a one/few-run
version degenerates toward rollback gain and that the named historical comparison changed many
things at once.

`DERIVED` — HCM improves the data model for campaign credit by separating:

```text
between-run variation: seed, machine, base checkpoint, config, campaign context
within-run longitudinal variation: regime/state transitions and chosen actions
within-state predictive variation: multiple forecasters scored on the same target
```

It does not identify causal credit for a lever that was never chosen in a state or for run-level
arms confounded with the entire campaign configuration.

`ASSUMED` — After multiple real runs with logged actions and support, a causal response model may be

\[
R_{ri}\sim t_\nu\!\left(
  b_r+\eta_{g_{ri}}+\theta_{g_{ri},A_{ri}}+\gamma^\top Z_{ri},
  \sigma_{g_{ri}}
\right),
\qquad
\theta_{ga}\sim N(\mu_a+x_g^\top b_a,\tau_a^2).
\tag{8}
\]

Run effects `b_r` are the unit layer; regime-arm effects are partially pooled. Regime/arm cells with
zero logged support return `NOT_IDENTIFIED`, not a prior-only number.

## 6. Explicit composition with FORE

`MEASURED` — The landed FORE memo derives target-over-logged occupancy weighting
`omega_pi=d_pi,gamma/d_nu_log`, a direct weighted-value estimator, and an optional doubly robust
correction. It also finds the present organ log lacks full Markov transitions, target-action
positivity, and cross-run support.

`DERIVED` — The two contributions compose in this order:

```text
causal graph/schema gate (HCM)
  -> action and transition support gate
  -> cross-fitted FORE occupancy ratio (target distribution / logged distribution)
  -> cross-fitted doubly robust pseudo-outcome or run-level weighted return
  -> HCM hierarchical Bayes pooling across real runs, regimes, and supported arms
  -> posterior decision with overlap/ESS and prior-sensitivity gates
```

`DERIVED` — FORE corrects distribution shift under a stable conditional transition/reward law.
HCM states the unit/subunit causal structure and pools heterogeneous supported effects. FORE does not
close the run-level `A <- U -> Y` path; HCM pooling does not create FORE action support. Their debts
are complementary and cumulative.

`ASSUMED` — Let the cross-fitted FORE doubly robust contribution be

\[
\phi_{ri}^{\pi}
=(1-\gamma)\,E_{d_0}\widehat Q^{(-r)}(Z_0)
+\widehat\omega^{(-r)}(Z_{ri},A_{ri})
\left[R_{ri}+\gamma E_{a'\sim\pi}\widehat Q^{(-r)}(Z'_{ri},a')
-\widehat Q^{(-r)}(Z_{ri},A_{ri})\right].
\tag{9}
\]

Aggregate `phi` to a declared run/regime estimand and place those aggregates—not raw pair rows as
independent units—under model (8). Carry ratio uncertainty with run-level bootstrap/posterior draws;
report support exclusions, maximum weight, normalization error, and effective sample size.

`DERIVED` — Raw importance weights in a naive Bayesian likelihood can overstate information. If a
generalized/power likelihood is used, weights must be normalized at the independent-run level and
uncertainty calibrated by whole-run resampling. Zero support remains a refusal; clipping is not an
identification repair.

`DERIVED` — Current composition verdict:
`NO-GO`, `verdict_scope=FORMULATION x CURRENT SINGLE-TRAJECTORY DETERMINISTIC ORGAN LOG`, for causal
off-policy campaign credit. `CONDITIONAL GO`, `verdict_scope=PREDICTIVE BACKTEST x CURRENT ORGAN`,
for partial pooling of forecaster errors with explicit prior-sensitivity and no activation authority.

## 7. Smallest reformulation queue

1. `ASSUMED` — **Causal manifest first.** Join run ID, exact treatment vector, base checkpoint/hash,
   seed, machine/backend/axis, data-order manifest, stage plan, scorer/cache hashes, pair ID, ordered
   state, apparatus flags, and realized-through-R outcomes in a typed append-only schema.
2. `ASSUMED` — **Matched run-level A/B remains the honest default.** From one common checkpoint,
   randomize exact typed control/treatment arms, preserve stage checkpoints, and forbid other config
   changes. This pays the unit-treatment identification debt directly.
3. `ASSUMED` — **Within-run experiment only when scientifically real.** A pair-level randomized
   intervention may exploit HCM identification only after an interference-aware graph/probe shows
   that shared-trunk coupling does not invalidate the estimand. Otherwise use cluster/randomized
   run assignments.
4. `ASSUMED` — **Instrument route.** A valid subunit instrument must affect the pair-level treatment,
   have no direct outcome path, and have support. Stage timing, guard firing, or controller choice
   are not instruments by assertion.
5. `ASSUMED` — **L4 schema and sentinel.** Land equations (2)--(5) as a strict read-only gate only
   after the telemetry schema supports them; require the frozen-run positive control.
6. `ASSUMED` — **Organ accrual.** Keep the real-only graduation rules: synthetic data may shape
   priors but does not count as real trajectories. Fit/pool only after multiple independent real
   records; retain the existing at-least-three-record learned-arm gate and at-least-five-run
   canonical-equation bar.
7. `ASSUMED` — **FORE after support.** Log full `(Z,A,R,Z')` transitions and target-action support,
   then estimate occupancy ratios and DR pseudo-outcomes before HCM pooling.

## 8. Triality, apparatus wire-in, and pointer honesty

`DERIVED` — **DAG leg:** standalone feed
`.omx/research/hcm_causal_attribution_DAG_FEED_20260713.md`. The shared hot DAG is untouched for main
review.

`DERIVED` — **Equation leg:** `N/A-with-reason`. No requested lever-effect identification law closes,
so no file is added to `src/tac/canonical_equations/`. Equations (2)--(9) are design/check equations,
not registered laws.

`DERIVED` — **DSL leg:** no lever, flag, trainer argv, controller, or activation surface is added.
A future `CausalAttributionSpec` would need typed graph/version, treatment/outcome identities,
pretreatment covariates, negative controls, clustering unit, support policy, and verdict scope;
default OFF until implementation and tests exist.

`INFERRED` — **Sensitivity-map hook:** supported posterior lever effects could eventually update the
lever marginal-value map; current output is `NOT_IDENTIFIED`, so no numeric sensitivity update is
admissible.

`DERIVED` — **Pareto hook:** binding debts are independent run count, treatment overlap, scorer
custody, storage/telemetry cost, and estimator variance; no score/byte benefit is claimed.

`DERIVED` — **Bit allocator hook:** non-binding. This work attributes training/control decisions,
not archive bits.

`DERIVED` — **Cathedral/autopilot hook:** fail-closed refusal token for unpaired run-level treatment,
zero support, invalid apparatus, or row-level pseudo-replication. No dispatch hook is added.

`DERIVED` — **Continual-learning hook:** this memo and DAG preserve the unit-treatment no-free-lunch,
the L4 graph-falsification design, and the FORE→DR→pooling order.

`ASSUMED` — **Probe disambiguator:** future modes must remain distinct:
`run_randomized_unit_treatment`, `interference_aware_subunit_treatment`, and
`predictive_only_partial_pooling`. Schema/support receipts, not taste, select the mode.

`MEASURED` — **Checkpoint discipline:** no long job was launched, so run checkpoint obligations are
not activated. The two durable design artifacts are the only checkpoint for this unit.

`MEASURED` — **Lane/anti-collision:** the canonical lane/shared DAG stores were already dirty from
live siblings. Under the prompt's own-files-only boundary, this research-only lane was not appended
to those shared files; main review owns any later registration/merge.

`MEASURED` — **Pointer delta: `NONE`.** No score, archive, evaluator result, live run, or promotion
state changed. Files remain uncommitted for main review.

## 9. Verdict-scope registry

| negative/refusal | exact scope | explicitly remains open |
|---|---|---|
| zero identified top-three queries | `FORMULATION x EXISTING HETEROGENEOUS CORPUS x RUN-LEVEL LEVERS` | matched/randomized run A/B; valid adjusted study with overlap; genuine subunit randomized treatment |
| hierarchy gives no free lever attribution | `TREATMENT OUTSIDE INNER PLATE` | HCM identification for treatments that really vary within units under its graph assumptions |
| current telemetry cannot run L4 | `CURRENT COMMON TELEMETRY SCHEMA` | specialized-artifact adapter and future typed causal telemetry |
| L4 cannot certify no confounding | `OBSERVATIONAL GRAPH FALSIFICATION` | detection of violations with observable negative controls/positive sentinels |
| organ causal pooling refused | `CURRENT ONE-TRAJECTORY DETERMINISTIC LOG` | predictive pooling now; causal pooling after real supported multi-run accrual |
| FORE composition refused | `CURRENT LOGS WITHOUT MARKOV/ACTION SUPPORT` | cross-fitted FORE/DR after transition and overlap custody |
| canonical equation refused | `THIS UNIDENTIFIED LEVER-EFFECT FORMULATION` | registration after an identification law and empirical anchor close |

## 10. STORES CONSULTED

`MEASURED` — Preflight/operating authority read: full `CLAUDE.md`, full `AGENTS.md`,
`docs/operating_manual_craft_handoff.md`, SPEC-v7.5 operating contract, latest directives and sister
handoff memos, current frontier report, lane registry, subagent-progress ownership map, and top
project memory entries.

`MEASURED` — Causal/duty/corpus surfaces read: `src/tac/witness_dsl/activation_ledger.py`, current
activation/significance stores through its read API, `experiments/results/**/telemetry.jsonl`,
`experiments/results/**/launch.sh`, relevant curriculum factories, default-off decision table, and
specialized level-set logs/artifacts read-only.

`MEASURED` — Organ/credit surfaces read:
`.omx/research/costate_organ_capabilities_limits_envelope_20260711.md`,
`.omx/research/costate_organ_trajectory_ledger.md`,
`.omx/research/organ_regime_conditional_dispatch_436_20260711.md`,
`.omx/research/simpletes_costate_controller_assessment_20260705.md`,
`.omx/research/synthetic_data_for_costate_organ_supercharge_20260711.md`,
`.omx/research/amortized_operator_pontryagin_loop_cluster_20260711.md`, and the landed FORE memo.

`MEASURED` — No live run directory was modified. No sibling deliverable, shared DAG, canonical
equation, DSL, ledger, controller, or pointer was edited.
