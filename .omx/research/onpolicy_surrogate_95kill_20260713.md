# Task #455 — on-policy amortized input-costate surrogate for the frozen-SegNet forward

**ONE-LINE OUTCOME:** the nonlinear on-policy provider reduced isolated inference to
`127.524528 ms` (`4.211311x` versus the same-run exact forward and `12.985737x` DERIVED versus the
operator-supplied `1656 ms` comparator), but it is **NOT ADMITTED**: the tested formulation is
globally `NO-GO`, boundary is anchor-only `NEEDS-MORE`, and full-K20 fidelity is separately UNKNOWN.

**DATE:** 2026-07-13 UTC  
**LANE:** `lane_455_onpolicy_forward_surrogate_20260713`  
**TASK:** canonical Task `455`  
**REVIEW STATUS:** externally tracked against this memo's content hash in the review ledger and
canonical Task `455` record; do not infer review count from this self-referential document.  
**AUTHORITY:** training-signal research only; never `d_seg`, archive-score, promotion, MPS,
contest-CPU, or contest-CUDA authority  
**POINTER DELTA:** none; defensive `[contest-CPU]` pointer remains `0.1880443979880752`

## Verdict and scope

The campaign verdict is **`NO-GO`** under conjunctive pass-all-regimes precedence. The early and late
saved regimes decisively reject this registered formulation: early's final EMA provider is not admitted,
and late's surrogate-driven exact `d_seg` trajectory departs from its zero deterministic-repeat
floor. Boundary ends after a single exact-anchor update, so its exact trace match contains no
surrogate-inference step and cannot validate replacement. No regime reaches the configured W5
horizon, and none validates the recurring K20 target.

`verdict_scope`: the registered nonlinear 3/5-receptive-field formulation; pair 0; seed 455; saved
early/boundary/late renderer states; five exact-labelled student-owned collection steps; joint
CE/through-R-`d_seg`/`d_pose` exact control; event-conditioned windows of one to three updates;
macOS-arm64 CPU / Torch-fp32 advisory training-gradient axis. This is not a surrogate-family kill,
not full-K20 evidence, and not score authority.

## STORES CONSULTED

- `CLAUDE.md`, `AGENTS.md`, `docs/operating_manual_craft_handoff.md`, and the v7.5/v8 canonical specs.
- `reports/latest.md` and `tac.frontier_scan.build_frontier_scan_payload`.
- Lane, task, subagent, equation, probe-outcome, and DAG registries under `.omx/state/` and
  `.omx/research/`.
- The sealed task-455 dependency; methodology-falsified and superseded historical receipts; final
  source bundles; early/boundary/late final receipts; their two-slot/stage checkpoints; and the
  content-addressed campaign reducer.
- Official arXiv record for OPD, arXiv:2602.12125. Only its dense teacher supervision on
  student-generated trajectories is transferred here; its LLM reward-extrapolation results are not
  evidence for this vision-costate formulation.

No cloud, paid provider, GPU, protected live run, `upstream/evaluate.py`, or submission surface was
read or actuated for this measurement.

**Imported-method citation at derivation:** Wenkai Yang, Weijie Liu, Ruobing Xie, Kai Yang,
Saiyong Yang, and Yankai Lin (2026), “Learning beyond Teacher: Generalized On-Policy Distillation
with Reward Extrapolation,” arXiv:2602.12125, DOI 10.48550/arXiv.2602.12125. The official arXiv
abstract page was resolved on 2026-07-13. Only student-trajectory data collection is imported; no
paper result is treated as evidence for this vision-costate formulation.

## What was built

### Nonlinear amortized costate provider

The provider predicts the input costate `lambda_t = dL_teacher/dx_t`, not `d_seg`. With exact anchor
frame `x_a`, exact anchor costate `lambda_a`, current realized-through-R frame `x_t`, RGB scale
`c=255`, and data-derived costate RMS `s_a`, the typed architecture is

```text
z_t = concat(x_t/c, (x_t-x_a)/c, lambda_a/s_a)
h_t = GELU(Conv1x1(z_t))
r_t = Mix1x1(concat(Phi_3(h_t), Phi_5(h_t)))
g_t = tanh(sqrt(mean_channel(((x_t-x_a)/c)^2)))
lambda_hat_t = lambda_a + s_a * g_t * r_t
```

`Phi_3` and `Phi_5` are nonlinear depthwise-plus-pointwise branches. The multiplicative gate gives
structural reference cancellation: `x_t=x_a => lambda_hat_t=lambda_a` for every parameter value.
This is a nonlinear multi-receptive-field model, not the formulation-falsified linearization.

The dense per-transition loss is equal-weight normalized MSE plus cosine debt,

```text
ell_t = mean(((lambda_hat_t-lambda_t)/s_a)^2)
        + max(0, 1-cos(lambda_hat_t/s_a, lambda_t/s_a)).
```

The zero floor follows from mathematical nonnegativity and removes a measured fp32
self-similarity artifact (`-0.00015485286712646484`) found in a superseded receipt. The live learner
updates the only served provider through `EMA <- 0.8*EMA + 0.2*live`; admission is controlled solely
by EMA-shadow loss.

### On-policy collection, anchors, and injection

Every collection state is produced by the current witness/student trajectory. At each of five
contiguous steps, the exact frozen teacher labels that state with `dL_teacher/d(frame)` and the
learner performs two updates. There is no fixed offline tensor-dataset API.

The exact-anchor target cadence is derived from the declared skip target:

```text
K = ceil(1/(1-q)), q=0.95 => K=20.
W = ceil(sqrt(K)) => 5 configured smoke steps; never decisive.
```

The joint exact controller accepts a step only after strict CE descent, nonworsening `d_seg`, and
nonworsening `d_pose`. Fresh review found that the launch harness incorrectly called fp32
parameter-quantization exhaustion a terminal floor. Those completion labels are revoked: the three
runs provide accepted measured prefixes of one, two, and three updates, followed by a BLOCKED line
search. Only an exactly zero renderer-gradient certificate can close a future window as a floor.

The optimizer learning rate and maximum fractional step are ASSUMED constant control laws at
`0.01`; their anchors are the explicit typed argv preserved in every final receipt. They are tested
formulation values, not derived optima or recommendations. `K=20` is the executable
decisive cadence. `W=5` is only the derived smoke horizon and cannot decide admission. EMA decay is
the self-deriving formula `1-1/W=0.8`.

Between exact anchors the chain-rule seam is

```text
L_inject(theta) = <stop_gradient(lambda_hat_t), R(theta)>
grad_theta L_inject = J_R(theta)^T lambda_hat_t.
```

Only the detached EMA costate is injected. The surrogate is never an evaluator and never changes
`d_seg`/`d_pose` authority.

### Matched controls and resumability

For each saved regime, the exact branch derives a fractional step under the joint authority gate.
The surrogate branch receives the identical immutable parameter-step norm. The common start row and
every candidate row record exact CE, exact through-R `d_seg`, and frozen PoseNet `d_pose`. A second
deterministic exact run derives per-metric floors; all six repeated metric traces have zero maximum
delta.

Each run preserves atomic two-slot rolling checkpoints plus distinct collection, exact-window,
repeat-window, and surrogate-window stage checkpoints. Checkpoints include live/EMA model,
optimizer, trajectory position, theta, anchor frame/costate, timing samples, teacher-call counts,
and common schedules/traces. Resume restores with zero teacher calls. The final receipts also bundle
and hash every launch source. All three 200 MB storage plans were explicit local opt-ins because the
available SSD root was not writable in this sandbox; all evidence paths are durable and retained.

## Measurements

Final campaign receipt:
`experiments/results/onpolicy_costate_matched_campaign_final_20260713T043000Z.json`, SHA-256
`5b73396f4990a0d7d44fd358d64fc87d4b3e442dc7ac7a34f1264013fae5aff8`.

All values are `[macOS-CPU advisory training-gradient]`. Line-search/validation work is separately
accounted and excluded from the symmetric operational windows. Sums of complete per-step timers,
not isolated component timers, determine the whole-window speedup; each complete timer includes
render, provider, renderer VJP, and candidate update.

| Quantity | Label | Value | Samples / basis |
|---|---|---:|---|
| Exact frozen forward only | MEASURED | `537.045463 ms` | `n=9`, same-run saved states |
| Exact costate forward+backward | MEASURED | `3009.069611 ms` | `n=6` |
| Surrogate inference only | MEASURED | `127.524528 ms` | `n=3` non-anchor steps |
| Same-run forward-only speedup | DERIVED from measured | `4.211311x` | `537.045463/127.524528` |
| Same-run forward-only reduction | DERIVED from measured | `76.254426%` | `1-127.524528/537.045463` |
| Operator reference | OPERATOR-SUPPLIED | `1656 ms` | not sampled by these runs |
| Operator-reference speedup | DERIVED | `12.985737x` | `1656/127.524528` |
| Operator-reference reduction | DERIVED | `1528.475472 ms` (`92.299243%`) | comparator minus inference |
| Dense anchor fit | MEASURED | `2398.843922 ms` | `n=15`; training cost, not inference |
| Exact renderer VJP | MEASURED | `157.620632 ms` | `n=6` |
| Surrogate renderer VJP | MEASURED | `116.851188 ms` | `n=6` |
| Surrogate non-anchor operational step | MEASURED | `1073.518778 ms` | `n=3`; render+provider+VJP+update |
| Three exact event-conditioned windows | MEASURED | `25.228484 s` | sum of symmetric complete per-step timers |
| Three surrogate event-conditioned windows | MEASURED | `11.969160 s` | identical per-regime schedules |
| Whole-window speedup | DERIVED from measured windows | `2.107791x` | `52.557896%` reduction |
| Full K20 fidelity/economics | UNKNOWN | not measured | target cadence not reached |

The `12.985737x` number is explicitly a DERIVED comparison to the operator-supplied `1656 ms`, not
a same-host measurement. The same-run measured forward-only comparison is `4.211311x`. The whole
training-step economics are smaller (`2.107791x`) because rendering, exact anchors, and renderer VJPs
remain in the operational path. A prior corrected rerun measured `1.329916x`; therefore the observed
corrected-repeat whole-window range is `1.329916x` to `2.107791x`, and the shared-host timing noise
floor remains UNKNOWN. Both measurements save less than the operator-supplied 78% forward share.

## Fidelity re-derivation

| Regime | Exact start -> end `d_seg` | Target end `d_seg` | EMA admitted | Observed skip | Fidelity result |
|---|---:|---:|---|---:|---|
| early | `0.004714965820 -> 0.004592895508` | `0.004592895508` | `false` | `0.5` | `NO-GO`; `d_seg` matches, but EMA fails and CE/pose drift |
| boundary | `0.003524780273 -> 0.003509521484` | `0.003509521484` | `true` | `0.0` | canonical prefix `GO`; mission `NEEDS-MORE` because only the exact anchor ran |
| late | `0.003992716471 -> 0.003977457682` | `0.003997802734` | `true` | `0.666667` | `NO-GO`; `d_seg` first fails at step 3 by `0.0000203450521` |

Early's maximum CE and `d_pose` deltas are `6.239861249923706e-8` and
`0.000042936140914662246`, respectively, even though discrete `d_seg` remains bit-equal. Late first
departs in CE/`d_pose` at step 2 and in `d_seg` at step 3; its maximum CE, `d_pose`, and `d_seg`
deltas are `4.153698682785034e-7`, `0.011055387948260886`, and
`0.000020345052083333044`. Boundary contains no non-anchor inference and therefore cannot answer
the replacement question.

The requested decisive gate is therefore not met. The formulation shows a real local inference
economy, but fidelity rejects it where non-anchor deployment is actually observed. No live witness
trainer integration was performed.

## Historical receipts and falsified interpretations

- `experiments/results/onpolicy_scorer_surrogate_20260713T020600Z/measurement_receipt.json` is
  preserved but methodology-falsified for sparse anchors, live/EMA mismatch, incomparable schedules,
  and incomplete resume state.
- `experiments/results/onpolicy_costate_matched_early_20260713T030500Z/measurement_receipt.json` is
  preserved but superseded by the cosine-debt numerical floor repair.
- `experiments/results/onpolicy_costate_matched_campaign_20260713T031500Z.json` is preserved as the
  source-compatible pre-joint-control campaign, but superseded by the final joint-control receipts.
- The earlier `034000Z` early result is preserved and superseded by `034200Z` because the final
  harness added symmetric candidate-update timing. No evidence bytes were deleted or rewritten.
- The stale reducers in the `034000Z` through `044000Z` timestamp range are preserved as historical
  reductions of superseded receipts. The `040800Z` reducer is the first corrected rerun and consumes
  the source-frozen `040200Z` receipts. The `043000Z` reducer is authoritative because it consumes the
  final `042500Z` receipts under the final source hashes; lexical timestamps alone do not define
  evidence authority.
- All corrected `040200Z` receipts record `valid_terminal_floor=false` and
  `completion_reason=LINE_SEARCH_BLOCKED_AFTER_MEASURED_PREFIX`. They also apply and account an
  excluded exact-forward warm-up before each surrogate operational step to match the exact arm's
  SegNet cache treatment.
- The `040800Z` aggregate and `040200Z` receipts are preserved as the first corrected rerun. The
  authoritative `043000Z` aggregate consumes the final `042500Z` rerun after the blocked isolated-
  timing wording and warm-up regression guard were fixed. Fidelity traces are identical; timing
  variation is reported above rather than selecting one run.

## Triality and durable surfaces

- **DSL:** `src/tac/witness_dsl/onpolicy_scorer_surrogate_policy.py`.
- **Canonical equation harvest artifact:** `onpolicy_input_costate_surrogate_v1` in the untracked
  `src/tac/canonical_equations/onpolicy_input_costate_surrogate_20260713.py`; the locked registry still
  contains the older campaign row because the serializer landing is blocked.
- **DAG FEED harvest artifact:** the intended final task-455 campaign block is the clean-index patch
  `experiments/results/onpolicy_costate_symmetric_timing_20260713T034500Z/task455_symmetric_timing_dag.patch`;
  the shared DAG still contains the superseded `040800Z` row and must not be described as final.
- **Implementation:** `src/tac/scorer_surrogate/amortized_onpolicy_costate.py`,
  `src/tac/scorer_surrogate/onpolicy_matched_verdict.py`,
  `tools/probe_onpolicy_costate_matched_window.py`, and
  `tools/aggregate_onpolicy_costate_matched_campaign.py`.
- **Evidence:** final early/boundary/late receipt directories and the final campaign JSON above.

## Reactivation rule

Do not wire this formulation into the live trainer. A successor must pre-register a changed capacity,
EMA/update law, or provider representation and retain dense student-owned labels, exact joint anchors,
EMA-only serving/admission, common schedules, full CE/`d_seg`/`d_pose` custody, symmetric timing, and
zero-teacher resume. It must first clear real non-anchor fidelity in every saved regime and then measure
the full K20 trajectory before claiming the recurring 95% skip.
