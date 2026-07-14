# Bregman all-surfaces → V9·CGauge DAG FEED — 2026-07-14

Pointer: `0.1910828242 [contest-CPU Linux x86_64]`; local PR128 defensive
bank `0.1880443979880752` remains non-submission. **UNCHANGED.** This feed is
`$0`, local-only, and carries no MPS, score, promotion, or launch authority.

Source scope: the exact operator-named `DOxML2026Frank.pdf` attachment was not
recoverable from the workspace or approved local-app surfaces. All 31 pages of
Nielsen's public DOxML 2025 “Short stories on Bregman divergences — Flat and
curved” deck were read, together with the primary curved-Bregman,
generalized-Legendre/gauge, cumulant-free exponential-family KL, and extended
nonnegative Monte Carlo f-divergence papers. This is a source-substitution
disclosure, not a claim that the unavailable 2026 attachment had identical
pages.

## Ranked surface feed

| Rank | Surface | Bregman result | Code/equation | Measured benefit | V9 wire-in/status |
|---:|---|---|---|---|---|
| 1 | MC KL | Replace `mean(log p/q)` with `log(p/q)+q/p-1` for iid-from-p samples | canonical helper consumed by `findings_lagrangian.info_gain`; `extended_nonnegative_mc_kl_v1` | Real defect reproduced: exact KL `5.0e-5`; seed-26 one-sample naïve estimate `-0.01920064498`; extended estimate `+0.00032366385` | Live consumer patched; strict pointwise guard; no IS/MCMC laundering |
| 2 | Metric | Fixed-state PSD `M` is local quadratic Bregman/Mahalanobis. Exact ordinary-Hessian dual form uses `H^-1`; raw `||Δη||²` is the no-solve `H²` geometry | existing `argmax_native_vjp_fidelity_v1`; `local_hessian_dual_geometry_summary` | Synthetic 600-state SPD fixture: max primal/exact-dual error `5.68e-14`; max raw-dual/`H²` error `9.09e-13`; raw dual differs from ordinary Hessian on `600/600` | Existing V9 optimal-metric binding consumed; real n600 selection remains `NO_VERDICT_DATA_CUSTODY` |
| 3 | Gauge | `Fbar(θ)=λF(Aθ+b)+c·θ+d` gives `B_Fbar(θ1:θ2)=λ B_F(Aθ1+b:Aθ2+b)` and `ηbar=λAᵀη+c` | `AffineLegendreGauge`; `affine_legendre_gauge_covariance_v1` | Quadratic fixture identity error `8.88e-16` | Deterministic local receipt and binding are sealed into the V9 DSL policy; semantic label remains `GAUGE_IDENTITY_VERIFIED_NOT_MODEL_FACTORIZED` because executable model covariance is absent |
| 4 | Sigma points | A finite EF expectation reduces to at most `D+1`; KL is weighted log-ratio at those points; error is exactly `(θp-θq)ᵀ(ηhat-ηp)` | `caratheodory_reduce`; `verify_exp_family_sigma_kl`; `exp_family_sigma_point_kl_v1` | Synthetic D=4/N=600: exact `600→5`, 120× point count, KL `0.29780435845`; separately hashed fresh one-run wall-clock advisory `112.637327×`; five-class categorical is `5→5`, exactly 1× | No applicable custodied live EF-KL surface; `NO_VERDICT_LIVE_THROUGHPUT` |
| 5 | Centroid/projection | Right curved-Bregman centroid is projection of the full right centroid; current joint waterfill already is one Euclidean Bregman projection | existing joint allocator; `curved_bregman_centroid_projection_v1` | Retained n=50,000: identical allocation, waterfill `0.0046488s` vs FISTA `0.0037171s`; no new speedup and receipt prose is inconsistent | Existing consumer only; no duplicate solver. TerminalSolve designed-not-built; #423 arithmetic-centroid substitution Jensen-invalid |
| 6 | Chernoff | `Pα ∝ P1^(1-α)P2^α`; choose bisector `D(Pα||P1)=D(Pα||P2)` | `categorical_chernoff_bisector`; `categorical_chernoff_bisector_v1` | Synthetic Bernoulli `α*=0.5064439201`, residual `5.25e-13` | Seg/Pose scalar score terms are not distributions: `NO_VERDICT_DISTRIBUTION_CUSTODY`; retain score-gradient law |
| 7 | Cosine | On the unit sphere, `1-cos=1/2||x-y||²`, the restriction of ambient Euclidean Bregman; `F|sphere` is constant, so it is not a proper flat Bregman generator | `unit_sphere_cosine_curved_bregman_v1` | Exact local identity on orthogonal unit vectors | Diagnostic grounding only; never replacement license |

## Exact DAG

```text
cgauge_master_action_v1
  ├─ argmax_native_vjp_fidelity_v1
  │    ├─ categorical_bregman_geometry.v1
  │    ├─ extended_nonnegative_kl_mc.v1
  │    └─ v9_cgauge_optimal_metric_binding.v1
  ├─ affine_legendre_gauge_covariance_v1
  │    └─ affine_legendre_gauge_semantic_receipt.v1
  │         └─ GAUGE_IDENTITY_VERIFIED_NOT_MODEL_FACTORIZED
  ├─ exp_family_sigma_point_kl_v1
  │    ├─ Caratheodory expectation rule (≤D+1)
  │    └─ NO_APPLICABLE_CUSTODIED_EF_KL_SURFACE
  ├─ curved_bregman_centroid_projection_v1
  │    └─ existing joint P18/P19 Euclidean projection consumer
  └─ categorical_chernoff_bisector_v1
       └─ NO_VERDICT_DISTRIBUTION_CUSTODY

bregman_geometry (one argv-inert DSL Lever)
  └─ compile_v9_bregman_geometry_binding
       └─ provenance_canonicalize_fix_all_fakes (exclusive hot-file owner)
            ├─ sealed spec_v9_cgauge policy binding COMPLETE
            ├─ nested Bregman == top-level canonical metric COMPLETE
            ├─ deterministic local receipt/binding seal COMPLETE
            └─ owner test hash expectation refresh OWED
```

## CGauge no-fake verdict

`cgauge_master_action_v1` is real as a registered derivation graph, and the
trainer really contains through-R/frozen-scorer terms plus partial `(xi,R)`
regularizers. The repository has no executable affine-Legendre transform,
transformed-pair equality test, divergence-unit field, or content-bound
covariance receipt in the live trainer/launcher. Free per-frame latent codes
also remain. Verdict: **implementation-custody gap**, not undefined CGauge and
not a family negative.

The historical V9 formulation achieved best `d_seg=0.03482035319010417` at
epoch 150. Its later erosion and the separate roughly sevenfold best-vs-control
gap do not prove that missing affine covariance caused divergence. Scope:
`HISTORICAL_FORMULATION_ONLY_NOT_CGAUGE_FAMILY`; reformulation queue remains
full semantic gauge receipt plus explicit `Hol_xi + phase + events`
factorization custody.

## Triality and owner handoff

- Equations: `tac.canonical_equations.bregman_v9_surfaces_20260714` plus the
  pre-existing canonical metric equation.
- DSL: `tac.witness_dsl.bregman_geometry_policy`, one empty-override Lever,
  exact binding verifier, no invented flags.
- DAG/receipts: this feed and
  `.omx/research/bregman_v9_all_surfaces_measurement_20260714.json`.
- The exclusive provenance owner composed the sealed typed Lever and binding in
  the sole V9 DSL path. The sealed binding artifact is
  `.omx/research/bregman_v9_all_surfaces_binding_20260714.json` (SHA-256
  `bdc01ae586c4467b18ebf4deee206242426ddc51da3dc47d8a2b8fff6cab8481`).
  Receipt SHA-256 is
  `12b82ca3f9809339746cc03b48a3237643861dec9e9baec19852a184fa7f358c`.
  It regenerated byte-identically twice and binds all source bytes; the policy
  source is normalized only at the two unavoidable embedded receipt/binding
  hash literals. The policy recomputes the complete eight-file direct source
  closure, verifies the normalized policy source, and exact-compares the full
  canonical binding payload. Fresh standalone and independent verification are
  green. A later exclusive-owner edit reopened the declaration-table seal:
  expected `6cfa9845...`, live `5c926130...`. The latest strict V9 group is
  `16 passed, 2 failed, 24 errors`; every failure/error is downstream of that
  fail-closed seal mismatch, before the Bregman assertions are reached. The
  previously observed stale owner-test receipt/binding literals therefore also
  remain unresolved. Trainer activation is false and model covariance remains
  an implementation-custody gap.

## STORES CONSULTED

`CLAUDE.md`; `AGENTS.md`; v7.5 §8 and v8 specs; canonical equation registry and
`cgauge_master_action_v1`; `reports/latest.md`; lane registry; subagent progress;
latest sister findings/session/design/council/directive memos; V9 DSL/compiler,
trainer, metric, basis, optimization, and KL source/tests; retained waterfill
and V9 result receipts; watched arm and fleet inboxes; public Nielsen deck and
primary papers listed in the findings memo.
