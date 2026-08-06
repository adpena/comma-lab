# ddm_lw1 - Lam/Wang EO+ Crosswalk

Date: 2026-08-06

Arm: `ddm_lw1`

Status: COMPLETE. Scorer-free, analysis-only, no launch, no paid dispatch, no
archive, no `upstream/evaluate.py`, no score claim.

Paper: Henry Lam and Tianyu Wang, "Achieving First-Order Statistical
Improvements in Data-Driven Optimization: From No-Free-Lunch to Amplified
Decision Perturbation", arXiv:2608.04312v1, submitted 2026-08-05.
Sources read: arXiv abstract page <https://arxiv.org/abs/2608.04312> and arXiv
HTML full text <https://arxiv.org/html/2608.04312v1>. No local PDF or TeX hash is
claimed.

## Answer First

The paper does not apply to Pact's end objective: our score-lowering problem is
a deterministic n=1 overfit against fixed contest scorers and exact archive
bytes. It does apply to our instruments: small-n gates, subset estimators,
noisy thresholding, and scale choices that estimate or select against n600
population quantities.

Ranked headline:

| Count | Grade | Rows |
|---:|---|---|
| 1 | ADOPT | control-variate-corrected gate replay |
| 2 | ADOPT-CLASS | side-information residual-correlation admission; excess-risk/bootstrap coefficient selection for gate corrections |
| 1 | RACE | aggressive regularizer/damping scale only after a positive side-information test |
| 2 | LESSON-ONLY | blind perturbation no-free-lunch; EO+ is not an archive-score theory |
| 1 | N-A | direct EO+ adoption as a Pact candidate generator |

Control-variate verdict: **ADOPT as a $0 retro-test, not as a theorem claim.**
The named test is `lw1_control_variate_gate_replay`: on banked gate-vs-n600 rows,
compare raw gate estimates against
`gate_estimate - beta * (control_subset - control_n600)`, with `beta` fit by
leave-one-run/window-out regression separately by axis. Accept only if RMSE and
sign-decision errors shrink on held-out banked rows without erasing the
axis-specific subset-bias caveats. If the JD/a1 gate rows cannot be joined to
same-basis n600 endpoint arrays, emit `BLOCKED_NO_JOINABLE_GATE_ROWS`.

## Paper Deep Read

### EO and improvement orders

The paper studies stochastic optimization from iid samples. Plain empirical
optimization (EO) has an influence-function expansion, and excess risk is
quadratic in the estimation error, giving an order `1/n` leading term in the
standard smooth setting. "First-order improvement" means reducing that leading
constant without changing the rate exponent; "second-order improvement" means
only the next term changes.

Pact transfer: this vocabulary is about statistical estimators, not our final
archive. It is useful for deciding whether a gate correction reduces the leading
variance/bias term of an n600 estimator.

### The EO+ umbrella and no-free-lunch theorem

The directionally perturbed EO solution has the form
`theta_hat = theta_EO + H * mean(M(...))`, up to projection. EO+ covers many
optimization-enhanced and statistics-enhanced methods, including regularization,
DRO, transfer, shrinkage, CVaR, GMM-like constructions, and contextual weighted
EO.

The core theorem says first-order gain requires:

1. Correct side information: the perturbation has zero or adequately controlled
   population bias at the optimum.
2. Geometric effectiveness: the perturbation is non-orthogonal to the EO
   fluctuation/influence direction.
3. Non-vanishing scale: the adjustment magnitude is constant-order in the
   standard setting, not a vanishing literature-default scale.

If the side information is biased, or if the side information is orthogonal to
the EO fluctuation, the method gets at most second-order gain. This is the paper's
no-free-lunch result: perturbation alone is not magic.

Pact transfer: a gate covariate or stratum only helps if its subset-vs-population
residual is correlated with the gate error we need to remove. "Known n600 value"
alone is insufficient; "correlated with held-out gate residual" is the operational
test.

### Optimizing the adjustment

For correct side information, the paper's optimal adjustment is the regression
matrix

```text
H* = - E[IF * M_tilde^T] * E[M_tilde * M_tilde^T]^dagger
```

and it gives analytical and bootstrap procedures for estimating it. The bootstrap
approach is the important engineering analogue: use resampled or banked paired
rows to fit the correction coefficient without manually deriving influence
functions.

Pact transfer: do not hand-pick a correction constant for subset gates. Fit the
coefficient from banked gate-vs-n600 pairs, validate by leave-one-window/run-out,
and keep separate coefficients by axis (`d_seg`, `d_pose`, rate) and selector
mode.

### Control-variate connection

The paper explicitly connects its optimal perturbation to Monte Carlo control
variates: subtract a correlated auxiliary quantity centered by its known mean,
and choose the coefficient that minimizes variance. Pact has exactly this shape
for instruments: many gate reads have a small subset estimate plus banked n600
values for correlated observables such as prior-epoch per-pair losses,
per-class/block mass, baseline pose difficulty, and token/rate controls.

The required Pact difference: our estimator is finite-population and
temporally-correlated, not iid. Therefore the test is empirical on banked rows;
the theorem supplies the regression shape, not authority.

### Contextual and side-information sections

The contextual theorem keeps the same two conditions in scaled form: almost
correct side information and non-orthogonality to local fluctuations. The paper's
practical side-information sources are shape information, invariant information,
and local/conditional mean structure.

Pact transfer: stratification should be selected because it predicts gate error,
not because it is semantically appealing. Existing prefix-bias laws already show
that "video-order prefix" is often a different population; side information must
be admitted by residual correlation against n600 truth.

### Larger hyperparameters

The paper's "larger than literature" result is conditional. Large scales are
licensed only when side information passes correctness and non-orthogonality.
Otherwise the optimal scale shrinks with `n` or yields only second-order changes.

Pact transfer: this is not a blanket order to raise weights. It is a race rule:
for `en1` margin-weight, Q3 constraint strength, GN damping, or trust-region
radii, aggressive scales are admissible only after the side-information/covariate
used by the scale passes a banked residual-correlation check.

## Recall Evidence

| Query/source | Evidence found beyond charter seeds | What changed |
|---|---|---|
| Governing files: `_common_contract.md`, `lw1_prompt.md`, `PROGRAM.md`, `CLAUDE.md`, `AGENTS.md`, `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md` | This arm is scorer-free and must not touch upstream, protected files, the index, or unrelated dirty work. Hot state says current own-vehicle frontier is `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`; live scorer slot is owned elsewhere. | Kept scope to new markdown in `.omx/research/ddm_lw1_20260806/`; no scorer or launch planned. |
| `rg -i "control[- ]?variate|variance reduction|side information|non-orthogonal|excess risk"` over `.omx/research`, docs, reports, code | Existing "control variate" mentions are cached replay, surrogate/costate, or discrete estimator contexts; no already-landed gate-vs-n600 control-variate correction was found in searched scope. | Treated surface 1 as a new retro-test, not a duplicate implementation. |
| `ddm_ffm1_20260806/RECEIPT.md` and `NEXT_IF_RESUMED.md` | ffm1 already queued a strong-consistency replay over banked n600 rows comparing prefix, strided, seeded-random, and stratified estimators. | lw1 narrows that follow-on: add a control-variate residual correction on top of those estimators instead of duplicating the consistency replay. |
| `src/tac/subset_selection.py` and `src/tac/canonical_anti_patterns/na3_subset_bias_builders.py` | Prefix bias is already encoded as a mechanical law: pose prefixes can be 2.54x-4.21x harder while seg prefixes are 3-5% easier; n=8 banks nothing. | Any lw1 correction must preserve axis-specific caveats and may not promote prefix gates globally. |
| `ddm_na4_20260805/NA4_RECEIPT.md` | Rate-axis prefix behavior is stream/coder dependent, observed `0.989000x` to `1.029800x` full-population B/pair; OD9 direct prefix/full rate bias was not measured. | Rate control variates must be stream/coder-specific; no scalar "prefix cheaper" law is admissible. |
| `ddm_tq1_20260805/RECEIPT.md`, `tq1c/RECEIPT.md`, and final message | TQ1B/TQ1C provide measured small accepted moves and a noise-floor context, but they are candidate edits, not gate-vs-n600 paired training data for a correction coefficient. | Used as a no-free-lunch/small-yield lesson, not as the main CV training set. |
| `ddm_jd1`, `ddm_jd3`, `ddm_jd4` receipts and `experiments/ddm_jd4_endpoint_n600_both_bases.py` | The JD endpoint instrument carries n600 per-pair `d_seg`/`d_pose` arrays and a gate36 positive control; comments state the gate must land near the final a1 telemetry row. | This is the highest-probability data source for `lw1_control_variate_gate_replay`, pending join to telemetry rows by run/window/basis. |
| Canonical equations via `tools/list_canonical_equations.py --json` and targeted text search | Broad registry exists; relevant nearby entries include trajectory stopping, EMA laws, variance-cost ideas, and exact score arithmetic, but no gate CV equation was identified in bounded search. | No canonical equation registered in this arm; next replay can register only after measured residual shrinkage. |
| `CANONICAL_RESEARCH_INDEX_20260629.md` and `sub015_DAG_*.md` targeted search | Found many subset/stratified warnings and fast subset screening laws, but no pre-existing Lam/Wang-style CV correction for gate estimates. | The receipt records bounded absence in those scopes and routes to a specific follow-on. |

Bounded absence statement: these searches do not prove no related work exists
globally. They cover the queries and scopes above.

## Ranked Crosswalk

| Rank | Surface | Grade | Honesty | Pact adjudication | Named consumer | Falsifier |
|---:|---|---|---|---|---|---|
| 1 | Control-variate-corrected gate estimator | ADOPT | DERIVED plus retro-testable on banked rows | Use known n600 correlated observables to reduce small-gate estimator error. This is exactly the paper's control-variate shape, adapted as a finite-population regression. | `src/tac/subset_selection_gate.py`, `src/tac/subset_selection.py`, a1/gate36 protocol, ffm1 strong-consistency replay | `lw1_control_variate_gate_replay` shows no held-out RMSE/sign-decision reduction vs raw stratified/seeded estimator, or no joinable banked rows exist. |
| 2 | Side-information residual-correlation admission | ADOPT-CLASS | DERIVED | A stratum/control is "geometrically effective" only if it predicts the target gate residual on banked n600 truth. Semantic plausibility is not enough. | future gate-design receipts; NA3/NA4-style bias audits; hard-pair registry screening | Candidate side information has near-zero held-out correlation with gate residual or becomes axis-inverting after stratification. |
| 3 | Bootstrap/excess-risk coefficient selection | ADOPT-CLASS | DERIVED | Use leave-one-run/window-out or bootstrap to fit correction coefficients and uncertainty bands; do not borrow constants. | `lw1_control_variate_gate_replay`; future equation registration if measured | Coefficients fail stability across held-out windows or confidence bands do not cover n600 more often than raw gates. |
| 4 | Larger-than-literature hyperparameters | RACE | INFERRED | Large scale is licensed only after a positive side-info test. Candidate knobs: `en1` margin-weight scale, Q3 constraint strength, GN damping, trust-region radius. No retuning in this arm. | en1/Q3/GN/trust-region owners | Aggressive scale harms same-object n600 S or the side-info residual-correlation test is negative. |
| 5 | Second-order no-free-lunch for blind perturbations | LESSON-ONLY | DERIVED | Perturbation families without correct/effective side information should be expected to produce small/noisy second-order effects. TQ1-style small accepted snaps remain real, but they are not evidence for a broad blind-perturbation campaign. | negative-audit and perturbation-family triage docs | A side-info-free perturbation family repeatedly lands n600 byte-closed equal-byte improvements above noise, with exact/advisory custody. |
| 6 | EO+ theory as final archive-score theory | LESSON-ONLY | DERIVED | The paper's stochastic excess-risk frame is not Pact's deterministic archive objective. Use it for instruments and campaign-layer choices only. | all lw1 readers | A future Pact problem is explicitly framed as iid decision estimation with independent training/test distributions and an EO solution. |
| 7 | Direct EO+ method adoption as candidate generator | N-A | DERIVED | No DRO/regularization/transfer objective is adopted as a new archive vehicle in this arm. | none | A later build defines a receiver-closed archive vehicle whose legal payload and exact score are driven by an EO+ construction. |

## Control-Variate Retro-Test Spec

Name: `lw1_control_variate_gate_replay`

Inputs:

- Banked rows with both a small-gate estimate and same-basis n600 truth.
- Per-axis targets: `d_seg`, `d_pose`, and rate/bytes when available.
- Candidate controls with known n600 values at gate time: prior-epoch per-pair
  losses, base per-pair pose/dseg, per-class or block masses, token/rate rows, or
  any declared control whose subset value and n600 value can be recomputed.

Estimator:

```text
raw_j = subset_gate_estimate_j
delta_control_j = control_subset_j - control_n600_j
corrected_j = raw_j - beta * delta_control_j
beta = argmin_beta sum_train (truth_j - raw_j + beta * delta_control_j)^2
```

Use scalar `beta` first; allow ridge multivariate controls only after the scalar
test is positive. Fit and report separately by axis, selector mode, vehicle, and
basis. Validation is leave-one-run/window-out; do not train and test on the same
window.

Acceptance:

- Report denominators, selection mode, basis, and axis for every row.
- Accept a control only if held-out RMSE and sign-decision errors improve versus
  the raw estimator and versus the already-queued ffm1 strong-consistency
  baseline.
- If prefix remains biased after correction, classify as `INSTANCE` or
  `FORMULATION`, not population authority.
- If join fields are missing, emit `BLOCKED_NO_JOINABLE_GATE_ROWS`.

## Follow-Ons

| Status | Item | Fire order |
|---|---|---|
| QUEUED-WITH-FIRE-ORDER | `lw1_control_variate_gate_replay` | Next subset/gate calibration toucher joins JD/a1 gate rows to endpoint n600 arrays, then runs the estimator above. If the join fails, emit `BLOCKED_NO_JOINABLE_GATE_ROWS` with missing keys. |
| QUEUED-WITH-FIRE-ORDER | `lw1_sideinfo_residual_correlation_admission` | Implement as a report field in the same replay: every proposed stratum/control gets held-out residual correlation by axis before it can be used as an admission rule. |
| QUEUED-WITH-FIRE-ORDER | `lw1_large_scale_license_gate` | Before raising en1/Q3/GN/trust-region scale based on side information, cite a positive residual-correlation row or record `FOLDED_NO_SIDEINFO_LICENSE`. |
| FOLDED | `lw1_direct_eo_plus_archive_vehicle` | No action. The paper does not supply a receiver-closed archive vehicle. |
| FOLDED | `lw1_blind_perturbation_campaign` | No action beyond citation as a lesson in perturbation-family triage. |

## Boundaries

Measured in this unit: no scorer values, no archive bytes, no `d_seg`, no
`d_pose`, no runtime, no exact score.

Derived in this unit: theorem-level crosswalk, local recall, ranked dispositions,
and the precise control-variate retro-test spec.

Not done: no local PDF/source hash, no code implementation, no canonical-equation
registration, no launch, no paid dispatch, no exact eval.

Own-vehicle frontier line remains as read from `.omx/state/main_hot_state.md`:
`S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`; contest pointer
borrowed/unmoved.
