# Codex findings — Bregman framework applied to all V9·CGauge surfaces — 2026-07-14

**POINTER STATUS: UNCHANGED.** Submittable
`0.1910828242 [contest-CPU Linux x86_64]`; local PR128 defensive bank
`0.1880443979880752` remains non-submission. No archive, evaluator call, MPS
score, heavy launch, paid dispatch, or pointer mutation occurred.

## Source custody

The exact operator-named `DOxML2026Frank.pdf` attachment was not recoverable
from the workspace or approved local-app surfaces. I read all 31 pages of Frank
Nielsen's public 2025 DOxML deck, “Short stories on Bregman divergences — Flat
and curved,” and checked the primary curved-Bregman, generalized-Legendre/gauge,
cumulant-free exponential-family KL, and extended nonnegative Monte Carlo
f-divergence papers. This is an explicit source substitution, not a claim that
the inaccessible 2026 attachment is byte- or page-identical.

Primary sources:

- https://franknielsen.github.io/SlidesVideo/FrankNielsen-DOxML-Kyoto-May16-2025.pdf
- https://arxiv.org/abs/2504.05654
- https://arxiv.org/abs/2507.20577
- https://arxiv.org/abs/2003.02469
- https://franknielsen.github.io/NonNegativeMonteCarlo-fdivergence.pdf

## Ranked all-surfaces result

| Rank | Surface | Correct Bregman result | New module/equation | MEASURED benefit | V9 wire-in / strict status |
|---:|---|---|---|---|---|
| 1 | Extended sampled KL | For iid `x~p`, use `log(p/q)+q/p-1`; every contribution is nonnegative and its expectation is unchanged | existing canonical helper consumed by `info_gain.monte_carlo_kl_fallback`; `extended_nonnegative_mc_kl_v1` | Exact Gaussian KL `5.0000000000105516e-05`; seed-26 one-sample naive estimate `-0.019200644981038906`; extended `+0.0003236638484699489` | Live consumer patched; strict iid-from-p boundary; spatial/generic `batchmean` bug class repaired and scanner clean |
| 2 | Metric / dual coordinates | Fixed-state PSD `M` is a local quadratic Bregman/Mahalanobis metric. Exact ordinary-Hessian dual uses `H^-1`; raw `||delta_eta||^2` is the distinct no-solve `H^2` geometry | `local_hessian_dual_geometry_summary`; existing `argmax_native_vjp_fidelity_v1`; premise-falsification memo | Synthetic SPD 600-state max errors: `5.68e-14` for primal/exact-dual and `9.09e-13` for raw-dual/`H^2`; false equality on `600/600` | Single canonical metric consumed wholesale; real n600 M selection remains `NO_VERDICT_DATA_CUSTODY` |
| 3 | Affine-Legendre gauge | `Fbar(theta)=lambda F(A theta+b)+<c,theta>+d` implies `B_Fbar(theta1:theta2)=lambda B_F(A theta1+b:A theta2+b)` and `eta_bar=lambda A^T eta+c` | `AffineLegendreGauge`; `affine_legendre_gauge_covariance_v1` | Quadratic identity error `8.881784197001252e-16` | Deterministic receipt and binding are sealed into the V9 DSL policy; activation remains false and status is `GAUGE_IDENTITY_VERIFIED_NOT_MODEL_FACTORIZED` |
| 4 | Exponential-family sigma points | Expectation matching with positive weights and `s<=D+1` makes weighted point log ratios exact; error is `(theta_p-theta_q)^T(eta_hat-eta_p)` | deterministic scalar-RREF `caratheodory_reduce`; `verify_exp_family_sigma_kl`; `exp_family_sigma_point_kl_v1` | Synthetic D=4/N=600: exact `600->5` (120x point count) and KL `0.29780435845399555`; fresh one-run local timing advisory measured `112.637327x`; five-class categorical is `5->5`, exactly 1x | No applicable custodied live EF-KL term: `NO_VERDICT_LIVE_THROUGHPUT_NO_APPLICABLE_CUSTODIED_EF_KL` |
| 5 | N-point to one projection | For right orientation, the constrained centroid is the right Bregman projection of the full arithmetic centroid. Current joint waterfill already is the Euclidean instance | existing joint allocator; `curved_bregman_centroid_projection_v1` | Retained n=50,000 allocation parity is exact; waterfill `0.0046488328s`, FISTA `0.0037170839s`; no new speedup | No duplicate solver. TerminalSolve is designed-not-built; #423 arithmetic collapse is Jensen-invalid; noncommutative evaluator actions remain iterative |
| 6 | Chernoff operating point | `P_alpha proportional P1^(1-alpha)P2^alpha`; solve `D(P_alpha||P1)=D(P_alpha||P2)` on one normalized support | `categorical_chernoff_bisector`; `categorical_chernoff_bisector_v1` | Synthetic categorical `alpha*=0.5064439201178175`, stable bisector residual `5.252465129501616e-13`; endpoint and support bytes are content-bound | Seg/Pose scalar score contributions are not endpoint distributions: `NO_VERDICT_DISTRIBUTION_CUSTODY`; existing score-gradient operating law remains exact fallback |
| 7 | Cosine | On the unit sphere, `1-cos(x,y)=0.5||x-y||^2`, the ambient Euclidean Bregman restriction; `F` restricted to the sphere is constant, so it is not a proper intrinsic flat Bregman generator | `unit_sphere_cosine_curved_bregman`; `unit_sphere_cosine_curved_bregman_v1` | Identity exact on deterministic orthogonal unit vectors | Diagnostic grounding only; no replacement license |

All synthetic math numbers are `MEASURED [local CPU deterministic math
fixture]`, not real-through-R n600 score evidence. The separately hashed sigma
timing receipt is explicitly nondeterministic wall-clock advisory: it excludes
one-time Caratheodory setup from the repeated-log-ratio timing, varied on fresh
repeats, is not a canonical equation constant, and is not live V9 throughput
authority. The exact 120x point-count reduction is the reproducible result.

## CGauge no-fake verdict

`cgauge_master_action_v1` is real as a canonical derivation graph, and the live
trainer has through-R/frozen-scorer terms plus partial `(xi,R)` regularizers.
It is not undefined. The executable vehicle still has no affine-Legendre
transform pair, transformed-action/divergence equality receipt, divergence-unit
field, or proof that free per-frame latent codes factor exclusively as
`Hol_xi(base) + phase + events`.

Verdict: **IMPLEMENTATION-CUSTODY GAP ONLY.** The historical V9 formulation's
best `d_seg=0.03482035319010417 @ ep150` is a real retained row, but it does not
prove missing covariance caused the run's erosion or the separate best-vs-
control gap. Scope:
`HISTORICAL_FORMULATION_ONLY_NOT_CGAUGE_FAMILY`. The reformulation queue is an
executable transform pair, divergence/action equality on identical configured
bytes, explicit unit custody, and zero-uncustodied-residual `(xi,R)`
factorization.

## Extended-KL bug-class extinction

- `monte_carlo_kl_fallback` now delegates to the one sibling-owned canonical
  extended estimator and preserves iid-from-`posterior_after` custody.
- Spatial KL in U-DIE and NSCS02 now sums class support then averages batch and
  pixels instead of silently multiplying by `H*W`.
- The generic pause/distill helper now reduces `(B,...,C)` correctly.
- Frozen-teacher `batchmean` refuses non-flat logits.
- Flat DINO CLS/flattened-patch uses and the flat regression fixture carry
  explicit scanner waivers.
- The older categorical helper now refuses empty, nonfinite, non-simplex input
  and clamps only floating-point-scale negative residue.

These changes do not relabel MINE/DV lower bounds as KL estimates and do not
apply the iid estimator to importance samples, MCMC, unnormalized densities,
or mixture families.

## Triality and V9 composition

- **Equations:** six typed events were appended through the locked canonical
  registry API under `bregman_apply_all_surfaces`, in addition to the existing
  single `argmax_native_vjp_fidelity_v1` metric law.
- **DSL:** one argv-inert `bregman_geometry` Lever binds all seven surfaces,
  exact LawRefs, real-or-null runtime consumers, receipt schemas, strict
  statuses, and the existing canonical metric binding. Unknown, missing, or
  drifted fields refuse; the policy cannot self-assert live composition.
- **DAG/receipts:**
  `.omx/research/bregman_v9_all_surfaces_DAG_FEED_20260714.md`, measurement
  receipt SHA-256
  `12b82ca3f9809339746cc03b48a3237643861dec9e9baec19852a184fa7f358c`,
  and sealed V9 binding SHA-256
  `bdc01ae586c4467b18ebf4deee206242426ddc51da3dc47d8a2b8fff6cab8481`.
  The separate timing advisory SHA-256 is
  `004c9dbc72e1082e82b0eaf0608173bf6f770c552a3e6cdf0c93391feed1b897`.
- **Live wiring:** the exclusive provenance owner composed the sealed
  argv-inert Lever and binding into V9 and cross-bound its metric payload to the
  top-level canonical metric. The policy truthfully reports
  `composition_verified_by_policy=false`; only the external provenance checker
  certifies composition. The sealed policy additionally recomputes the exact
  eight-file receipt source closure, verifies its normalized self-source, and
  exact-compares the complete canonical binding payload. Trainer activation
  remains false. The basis source stabilized, but a later exclusive-owner edit
  reopened the scientific-declaration seal: expected `6cfa9845...`, live
  `5c926130...`. The current strict group is `16 passed, 2 failed, 24 errors`,
  all rooted in that fail-closed mismatch before the Bregman assertions are
  reached. The previously observed stale owner-test Bregman hash literals also
  remain unresolved.

## Verification and round-1 status

The final clean-pass count and independent round-1 disposition are recorded in
`.omx/research/codex_review_bregman_v9_all_surfaces_round1_20260714.md`.
Local surface gates are green and the deterministic receipt regenerated to the
same bytes twice. Fresh review's volatile `resolved_at`, missing-null verifier,
nested metric drift, Chernoff custody, Caratheodory determinism, receipt source
closure, exact binding validation, and provenance findings were repaired. The
remaining red is exclusively owned provenance/config transactional state: the
reopened declaration seal plus the previously observed stale hash assertion.
Final exact pass counts and owner disposition belong in the round-1 artifact.

## Ranked next actions

1. Provenance owner adds the **semantic** V9 gauge receipt: typed transform
   pair, pre/post divergence and master-action equality, divergence unit, and
   explicit `(xi,R)` factorization custody. No activation before it passes.
2. Recover exact real 0..599 selected-M data custody before using the cheap
   squared-Hessian diagnostic or any Bregman metric for provider admission.
3. Apply sigma points only when a live regular EF-KL consumer retains natural
   parameters, expectation coordinates, point IDs/weights, density-ratio
   parity, and matched end-to-end timing. Five-class categorical and analytic
   Gaussian KL do not benefit.
4. Build TerminalSolve only as a resume-safe, per-stage-checkpointed full-P
   consumer with exact n600 accept/rollback. Do not replace #423 or
   noncommutative actions by an arithmetic centroid.
5. Define seg-optimal and pose-optimal normalized distributions on one sealed
   candidate/trajectory support before asking for a live Chernoff alpha.

## STORES CONSULTED

`CLAUDE.md`; `AGENTS.md`; `PROGRAM.md`; operating manual; v7.5 §8 and v8
specs; reports/latest; canonical equation, lane, task, subagent, probe, and
cost stores; latest sister findings/session/design/council/directive memos;
live V9 DSL/compiler/trainer/metric/optimization/KL source and tests; retained
waterfill and V9 receipts; watched arm and fleet inboxes; public sources listed
above.

Pointer delta: exactly zero.
