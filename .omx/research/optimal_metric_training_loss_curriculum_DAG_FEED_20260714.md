# DAG FEED — reachable-decision training metric and metric anneal

**FEED:** `FEED-500-optimal-metric-training-loss-curriculum`  
**Lane:** `lane_optimal_metric_training_loss_curriculum_20260714`  
**Status:** `BUILD_GREEN / MEASUREMENT_NO_VERDICT_DATA_CUSTODY / NO_LAUNCH_AUTHORITY`  
**Pointer:** `[contest-CPU Linux x86_64] 0.1910828242` unchanged.

## Nodes

| Node | Evidence | State |
|---|---|---|
| canonical real-n600 GT cache | MEASURED custody, SHA `cf8d8360...8cd6` | retained; has frames/labels/GT margins/poses, not candidate logits/steps |
| retained surrogate pullback receipt | MEASURED advisory, real-n600-source heldout-n120, SHA `c4116ff0...3753` | reachable alignment lift `12.5x-51.4x`; no flip selector authority |
| `argmax_native_vjp_fidelity_v1` + `tac.scorer_surrogate.vjp_fidelity` | sibling canonical equation/helper; 24 focused tests | strict NumPy-fp32 generic pullback/preconditioner/selector surface; exact-n600 completeness gate; no duplication here |
| `#141/#274` margin saliency/reachability | MEASURED/landed partial metric shaping | supplies scalar support inside `W_s` |
| `#382 sigma_cc'` | landed class-edge anisotropy; fitted instance triangle violation measured | requires metric closure before use as a distance block |
| `#360/#459` MarginBandSatisficing | landed derived R-headroom hinge | terminal constraint, not metric |
| `#430` coherent curriculum | model-based backtest + typed schedule realization | canonical stage order: island-birth -> boundary-form -> tau-sharpen+repair -> finish |
| `DecisionMetric / MetricStage / MetricAnneal` | BUILD, 13 new + 140 relevant existing tests pass | typed, argv-inert, canonical law/schema IDs distinct, support gap surfaced, malformed-input safe, measured labels custody-gated |
| full-n600 CE-vs-reachable-metric flip selector | required empirical node | `NO_VERDICT_DATA_CUSTODY`; optimal-form replay queued |
| byte-closed exact archive | terminal authority | absent; pointer unchanged |

## Edges

1. `R(theta) -> frozen SegNet logits -> centered quotient C -> stage selector Q_s -> J_q`.
2. `J_q + W_s -> G_s = J_q^T W_s J_q + lambda I -> damped natural step G_s^dagger grad L`.
3. `#141/#274 -> W_s support`; `metric_closure(#382) -> W_s class-edge block`; `#360 -> terminal feasible-set constraint`.
4. `#430 stage event -> MetricStage`; metric changes only on a stage boundary.
5. `saved checkpoint + full n600 logits/Jacobians + equal-trust finite step -> actual through-R flip/Pose deltas -> metric-selection receipt`.
6. `selection receipt(n=600, SHA, through_R=true) -> DSL measured status -> reviewed trainer consumer -> governed resumable A/B`.
7. `A/B -> byte-closed archive -> exact contest-CPU and contest-CUDA rows -> pointer decision`.

## Canonical law

`q_s(theta)=Q_s C z(R(theta))`

`G_s(theta)=D_theta q_s^T W_s D_theta q_s + lambda_s I`

`L_M,s=1/2 ||q_s(theta)-q_s(theta*)||^2_{W_s}`

`u_s=-eta G_s^dagger grad L_s`

Negative-entropy Bregman/CE supplies categorical Fisher curvature; the new object is the reachable quotient pullback plus its damped inverse. In `h_T^T M h_S`, `M=G_s^dagger` by convention.

## Metric cascade

| #430 stage | Typed metric | Exit |
|---|---|---|
| `island_birth` | global through-R CE / Euclidean optimizer geometry | `birth_completion` |
| `boundary_form` | centered categorical-Fisher pullback on margin/reachability annulus | `annulus_plateau` |
| `tau_sharpen_repair` | active winner-rival pullback + separate #360 constraint | `powerlaw_meat` |
| `finish` | finite applied-step flip preservation + Pose trust region | governed stop |

All stage choices beyond incumbent CE are `DERIVED_UNMEASURED` until the full-n600 finite-effect selector lands.

## Blocker classification and reformulation

`NO_VERDICT_DATA_CUSTODY` is not a negative metric verdict. The retained state lacks candidate logits/probabilities, winner-rival directional Jacobians, applied optimizer steps, and finite before/after flip outcomes. Reformulation queue: replay saved stage checkpoints without training; retain full n600 state; compare equal-trust CE/identity/Fisher/winner-rival steps; design-split select and held-out validate; attach immutable receipt.

## Triality

- **DSL:** `tac.witness_dsl.curriculum_dsl::{DecisionMetric,MetricStage,MetricAnneal,optimal_decision_metric_anneal}`.
- **Equation:** shared `argmax_native_vjp_fidelity_v1`; training specialization above/build spec; no fake n600 empirical anchor.
- **DAG:** this feed, with explicit terminal exact-eval edge and pointer stop.

## Ranked queue

1. Full-n600 no-training finite-step metric selector from hashed stage checkpoints.
2. Per-stage held-out comparison across CE/Tau/Muon/L7 checkpoints.
3. Typed resume-compatible trainer consumer only after a measured selector.
4. Operator-GO short matched A/B, then byte-close/exact dual-axis evaluation.
