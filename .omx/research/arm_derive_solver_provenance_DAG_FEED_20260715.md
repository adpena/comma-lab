# FEED-ARM-DERIVE — solver, provenance, rate, and marked telemetry

Date: 2026-07-15 UTC  
Lane: `lane_arm_derive_solver_provenance_20260715`  
Authority: `$0` local build/derivation only; pointer unchanged

```text
[frozen source + canonical metric argmax_native_vjp_fidelity_v1]
          |
          +-- categorical p, quotient-compatible g
          |      +-- Helmert H^-1 solve [BUILT NumPy-fp32]
          |      +-- local KL trust projection [BUILT]
          |      `-- MLX parity [BUILT / METAL EXECUTION OWED]
          |                    |
          |                    `-- real checkpoint consumer + A/B OWED
          |
          +-- delta_KL + checkpoint lambda_max(F) + measured delta_R
          |      +-- MorseContinuationSchedule [BUILT]
          |      |      muon_lr=sqrt(2 delta_KL/lambda_max)
          |      |      l7_mult=0; l7_threshold=m_safe
          |      `-- numerical schedule / stage A/B OWED
          |
          +-- V9 sharpening provenance
          |      +-- hosc beta 1->2->4->8 [DERIVED / SINGLE OWNER]
          |      `-- m_safe=2*0.019590163230895963
          |             =0.039180326461791926 [DERIVED]
          |
          +-- V9 EMA-best n600 rate receipt
          |      +-- D37 net 384637.90888513427 bits [REFRESHED]
          |      `-- D38 H_cov atlas [TYPED]
          |             +-- exact-array overlap instance GLUES
          |             `-- action/coefficient/cocycle/charged receiver UNBOUND
          |
          +-- event detectors
          |      +-- D39 priority mark + causal-manifest append [BUILT]
          |      +-- durable checkpoint cursor [BUILT]
          |      `-- calibrated H(E|X,C) / allocator OWED
          |
          +-- frozen SegNet/source/cache pins
          |      +-- D24a n600 x 2-query resume harness [BUILT]
          |      `-- real 1200-row radius/block-Jacobian run OWED
          |
          `-- #503 recursive-fractal route
                 +-- fullstack_unique_home_assignment_v1 [CONSUMED]
                 +-- BasisLeverSpec + one metric [CONSUMED]
                 +-- epoch schedule [THIS ARM]
                 `-- DCB encoder/live consumer/archive A/B [EXTERNAL OWNER / OWED]
```

## Verdict scopes

- Fisher solver: `FORMULATION x LOCAL_CATEGORICAL_QUOTIENT`; no activation or
  checkpoint efficacy claim.
- #302 schedule: `FORMULATION x CONFIG_COMPILATION`; no numerical curvature
  custody or A/B.
- D37: `FORMULATION x V9_EMA_BEST_N600_EMPIRICAL_SURFACE`.
- D38: exact-array restriction gluing only; global rate descent unbound.
- D39: observability-only schema/producer; no entropy, causal, codec, or score
  authority.
- D24a: harness-only; no historical tail confirmation and no locality verdict.
- #503: route-only; representation families open and receiver-rate custody absent.

## Six hooks

1. **Sensitivity:** categorical natural step and future D24a/D39 measured
   surprises; no proxy is promoted before receipt closure.
2. **Pareto:** no distortion/rate benefit without through-R, Pose, counted byte,
   and receiver survival custody.
3. **Bit allocator:** D37 class-edge and future event terms are charged with
   their context/model/receiver bytes, never gross-only.
4. **Autopilot:** D39 may select regular versus marked branch only when a typed
   mark is present; build status alone does not dispatch.
5. **Continual learning:** terminal empirical receipts, not build tests, update
   posteriors.
6. **Disambiguators:** delta convention is explicit; gauge projection is
   explicit; local split/global gluing and historical/current vehicle constants
   stay separate.

## Triality

- DSL: Fisher policy, continuation schedule, V9 LawRefs/config custody, D39
  typed manifest producer, D24a plan.
- Equations: seven owned/updated laws named in the companion findings memo;
  #503 consumes the existing unique-home law.
- DAG: this file plus the refreshed D37-D39 rate-law feed.

Runtime closure is intentionally absent where the DAG says `OWED`. Pointer
movement, score, launch, and promotion: none.
