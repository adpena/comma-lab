# Codex findings — ARM-DERIVE solver, provenance, rate, and telemetry closure

Date: 2026-07-15 UTC  
Lane: `lane_arm_derive_solver_provenance_20260715`  
Scope: `$0` local build/derivation/tests only; no GPU, provider, training,
evaluator, archive mutation, promotion, or pointer update

## Verdict first

**`BUILD_COMPLETE_LOCAL__MEASUREMENT_AND_MAIN_REVIEW_OWED`.** The categorical
Fisher `H^-1` quotient solve, witness-native continuation schedule, both V9
constant reconciliations, D37 refresh, D38 global-gluing type, D39 marked-event
producer, and D24a receipt harness are built and tested. Task #503 is routed to
the already-owned recursive-fractal lane without duplicating its metric, basis,
equation, or DecisionCarrierBundle implementation.

The submittable score pointer is unchanged. Nothing in this arm is score or
promotion authority.

## Raw results by requested item

### 1. #500/#504 categorical-Fisher `H^-1` solver — BUILT

- NumPy-fp32 authority solves
  `(Q^T(diag(p)-pp^T)Q + lambda I)v = -Q^T g`, `u=Qv`, in the explicit Helmert
  zero-sum quotient chart. It never takes an ambient inverse.
- The local trust projection enforces `u^T H u <= delta_quad`, with the
  convention explicit: `delta_quad=2*delta_KL`. Returned fp32 bytes are
  rechecked after casting; exact finite categorical KL is reported separately.
- Non-zero-sum cotangents fail closed. Gauge projection exists only through the
  explicit `project_gauge=True` surface.
- The MLX float32 implementation is independent and carries non-lowerable
  parity gates: correlation `>=0.9997`, step max error `<=3e-5`, quadratic max
  error `<=3e-6`, gauge error `<=3e-6`.
- The DSL policy emits no trainer argv and reports
  `built_not_activated_measurement_owed`; no fake live consumer was declared.

**Exact blocker:** MLX execution parity was not measured because this sandbox
has no usable Metal device. A checkpoint A/B and activation consumer are later
work; the solver family itself is built.

### 2. #302 witness-native continuation — DERIVED and BUILT

The three inherited PR95 scalars now come from one first-class DSL schedule:

```text
muon_lr     = sqrt(2 * delta_KL / lambda_max(F))
l7_mult     = 0
l7_threshold = m_safe
```

`delta_KL` and `lambda_max(F)` have no defaults: a launch compiler must provide
checkpoint-local custody. `l7_mult=0` is structural under unified tau; the hard
L7 indicator is not a second owner of the same sharpening force.
`l7_threshold` remains the R-survival boundary for telemetry/resume
compatibility. The schedule emits only flags parsed by the real trainer.

**Exact blocker:** no checkpoint-local Fisher-curvature receipt was supplied,
so no numerical Muon LR and no efficacy claim were minted.

### 3. Provenance defects — RECONCILED

#### `hosc_beta_end`

- V6 `10.0` remains a scoped clock-replica pin.
- V7 `3.177` remains a scoped event-boundary value.
- V9 has one owner, `v9_hosc_beta_endpoint_v1`, with the dyadic interface-width
  continuation `1 -> 2 -> 4 -> 8`; therefore the V9 endpoint is **DERIVED 8.0**.
- V9 config, argv, manifest, LawRef resolution, and regression tests are gated
  to the same value. The historical vehicles were not rewritten.

#### `MarginBandSatisficing.m_safe`

The bound report `reports/delta_R_noise_floor.json` has SHA-256
`a9e57041ff0e252527396136311daec0735a93f9143fbfa9e341c4074ed53c4e`.
It supplies:

```text
delta_R                = 0.019590163230895963  MEASURED
full-R annulus p95     = 0.03712034225463867   MEASURED
headroom               = ceil(p95/delta_R) = 2 DERIVED
m_safe                 = 2*delta_R
                       = 0.039180326461791926 DERIVED
```

V9 argv/manifest and `MarginBandSatisficing` now resolve through the same
LawRef. `0.06` is not the default; headroom 3 remains a future treatment only.

**Exact blocker:** efficacy of headroom 2 versus 3 remains an A/B question; the
provenance discrepancy is closed, not the performance question.

### 4. D37 rate law and D38 global `H_cov` gluing — REFRESHED / TYPED

D37 is refreshed from the V9 EMA-best n600 receipt, SHA-256
`60dd6a4837706d100932416cf8fdf77fce0e7c171b1ef58fd3f1154021428308`:

| Quantity | Value | Label |
|---|---:|---|
| class-aware gross gain | 467,373.90888513427 bits | MEASURED |
| flat-table charge | 10,342 B | CONFIGURED CHARGE |
| class-aware net | 384,637.90888513427 bits | DERIVED FROM MEASURED |
| net 95% interval | [373,674.7586229076, 395,236.54874890414] bits | MEASURED BOOTSTRAP |
| phase-aware gross | 464,657.2100251259 bits | MEASURED |
| phase table charge | 56,552 B | CONFIGURED CHARGE |
| phase-aware net | 12,241.210025125882 bits | DERIVED FROM MEASURED |
| phase-aware 95% interval | [957.5520005738945, 23,180.749990467913] bits | MEASURED BOOTSTRAP |

Verdict:
`RESIDUAL_NON_GAUGE_STRUCTURE_DETECTED__M_NOT_SUFFICIENT`, scoped to
`FORMULATION x V9_EMA_BEST_N600_EMPIRICAL_SURFACE`. The stale `+318,586` DAG
scalar is removed from the active successor feed.

D38 now types charts, pair overlaps, H-covariant intertwiners, triple cocycles,
coefficient objects, receiver section, and charged section bits. The exact-array
quadrant receipt binds **19,660,800** pairwise overlap points and glues exactly,
but its status remains
`TYPED_LOCAL_DATA_GLOBAL_RATE_DESCENT_UNBOUND`.

**Exact blocker:** action/coefficient bindings, overlap 2-cells, triple-cocycle
receipts, a real receiver section, and its charged bits are absent. Local split
or exact array restriction does not authorize global rate descent.

### 5. D39 marked events and D24a margin-gradient tail — BUILT / RUN OWED

#### D39

- `pact.causal_manifest.v1` now admits strict `event_mark` rows for topology,
  chart, and receiver-lattice families.
- Rows bind class edge, quantized spacetime mark, non-count-only incidence and
  attachment, before/after stratum, actual receiver state, evidence, canonical
  ID, and stage-local resume key.
- Priority is deterministic: `topology > chart > receiver_lattice`, while all
  matched detectors remain recorded.
- Identical append is a no-op; same identity with changed bytes conflicts.
  The checkpoint cursor advances only after durable append or proof that the
  identical row already exists. Old manifest rows round-trip unchanged.
- Status is `BUILT_RESUME_SAFE_OBSERVABILITY_ONLY`.

**Exact blocker:** no calibrated `H(E|X,C)` model, marked-branch allocator, or
empirical event corpus exists. Telemetry is not causal, codec, score, or
promotion authority.

#### D24a

The later-run harness now binds the frozen scorer, n600 source, and cache as
file/tree SHA-256 artifacts. It refuses anything except 600 pairs,
frozen-SegNet batch size 32, radii `(64,128,192)`, both minimum-margin and
high-margin-control queries, and same/adjacent/remote edge blocks. Raw rows are
append-idempotent and immutable. A terminal receipt refuses until the exact
1,200-row matrix is complete.

The canonical definitions are:

```text
T_r(q) = sum_{||u-q||>r} ||grad_x d_q(u)||^2 / sum_u ||grad_x d_q(u)||^2
B_ab   = ||partial d(edge_a) / partial x(region_b)||_F^2
```

The terminal receipt deliberately says
`NO_VERDICT_THRESHOLD_NOT_PREREGISTERED`; it cannot turn a local Jacobian into
a neighborhood Lipschitz proof or invent a locality threshold.

**Exact blocker:** real scorer/source/cache paths were not supplied and the
n600 run was explicitly deferred. Historical tail percentages remain
unconfirmed and are not embedded as expected values.

### 6. #503 optimal-per-dimension re-derivation — VERIFIED and ROUTED

No duplicate #503 implementation was added. The re-derived composition is:

| Dimension | Current route | Status in this arm |
|---|---|---|
| pixel | readout only; dense RGB is final scorer-boundary materialization, not a presumed low-rank store | prior pair-0 low-rank assumption refuted; no new measurement |
| class/chroma | external-owned DecisionCarrierBundle / v8 merge-diff-correct | `V9_INTEGRATION_BLOCKED_OWNER` |
| boundary/frequency | exactly one `BasisLeverSpec`, ranked by `argmax_native_vjp_fidelity_v1` | existing owner consumed; no second basis/metric |
| frame | decision keyframe plus receiver-necessary warp/correction | encoder/parse-back `NO_VERDICT` |
| pair | one `se(3)` `xi`; Pose6 residual only when receiver evidence requires it | unique-home law consumed |
| epoch | this arm's `MorseContinuationSchedule` at preserved stage boundaries | built; curvature/effect measurement owed |
| scale | `fullstack_unique_home_assignment_v1`; parent stores transport plus receiver-proven residual only | derived; real encoder owed |

The single metric is `argmax_native_vjp_fidelity_v1`; selected `M` remains
`NO-VERDICT_DATA_CUSTODY`. The latest sibling #502 handoff is preserved as
non-promotable advisory evidence: receipt
`031a1569c600bf5d1a3551a4da668db67bb80380d0678f964ca5476e0b293c33`
ranks equal-value/support saved-OFF receiver rows Fourier `0.4097223155`,
shearlet `0.4288604312`, curvelet `0.5048239560`, fixed non-PoU mix
`0.5303014119`. They are not equal bytes; literal decoder-boundary PoU and the
target-boundary inverse remain blocked. No family or byte winner follows.

**Exact blocker:** an actual DecisionCarrierBundle encoder, live V9 parser and
consumer, receiver parse-back, counted alternate archive, and full-n600 exact
A/B are absent from this arm. Status remains
`NO_VERDICT_RECEIVER_RATE_CUSTODY`.

## Triality and system wire-in

- **DSL:** Fisher policy (argv-inert until a real consumer),
  `MorseContinuationSchedule`, V9 LawRefs/config bijection, event-row/producer
  schema, and D24a typed probe plan.
- **Equations:** `categorical_fisher_natural_trust_region_solve_v1`,
  `witness_native_morse_continuation_v1`, `v9_hosc_beta_endpoint_v1`,
  `margin_band_satisficing_threshold_v1`, refreshed
  `rate_law_ladder_v2_measured`, `hcov_global_gluing_descent_v1`, and
  `segnet_margin_gradient_tail_block_jacobian_v1`. #503 consumes rather than
  duplicates `fullstack_unique_home_assignment_v1`.
- **DAG:** `arm_derive_solver_provenance_DAG_FEED_20260715.md` plus the refreshed
  rate-law feed makes every activation/measurement/receiver dependency explicit.
- **Sensitivity/Pareto/allocator:** Fisher steps and D24a/D39 measurements remain
  non-authorizing until real receipts; D37 can propose a contour grammar only
  after context and receiver costs are jointly charged.
- **Autopilot/continual learning:** marked events may route a future branch but
  cannot change policy until calibrated; no posterior or pointer was updated
  from build-only evidence.

## Verification

Targeted suites cover quotient/KKT/gauge/trust behavior, DSL/equation custody,
V9 constant bijection, D37 arithmetic, D38 fail-closed typing, manifest legacy
compatibility, D39 replay/conflict semantics, D24a completion/refusal, and #503
route consumers. Ruff, py_compile, and `git diff --check` are clean for the
owned implementation surfaces. Final targeted aggregate: **136 passed, 1
skipped in 8.58 s**. The skip is the MLX parity execution test: MLX is installed
but the sandbox exposes no Metal device.

## STORES CONSULTED

- `CLAUDE.md`, `AGENTS.md`, `docs/operating_manual_craft_handoff.md`, and
  `.omx/research/P0_campaign_queue_20260715.md`.
- V9/v7.5 operating specs, current `reports/latest.md`, lane/subagent/task
  state, canonical equations, latest Codex/Claude findings, and both live inbox
  files through `2026-07-14T20:32:37Z`.
- D36-D39 rate-law memo/spec/ticket/receipts, V9 EMA-best receipt, delta-R
  report, Task503 build spec/FEED, unique-home equation, canonical basis/metric
  surface, and final #502 broadcast custody.
- No provider, GPU, live run, protected run directory, evaluator, or paid state
  was touched.

## Pointer-delta honesty

Pointer movement: **none**. Score claim: **none**. Promotion eligibility:
**false**. The branch advances means, custody, and build gates only.
