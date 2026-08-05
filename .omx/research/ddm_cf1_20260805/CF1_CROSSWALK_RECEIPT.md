# CF1 Conformal Anomaly Crosswalk Receipt

Date: 2026-08-05

Task: CF1, Conformal Anomaly Detection in Python (`nonconform`, arXiv 2605.13642) crosswalk.

Verdict: **ADOPT_CLASS_WITH_INSTANCE_ALREADY_EMBODIED**.

The CF1 paper/package is relevant to Pact as an apparatus-calibration standard, not as a score mover. It should be adopted for L1 alarm-bank calibration, multiple-alarm false-discovery control, and shift-aware invalidation of stale nulls. It does not provide score authority, candidate promotion, or any replacement for the frozen contest scorer/evaluator.

No frozen scorer forwards, no `upstream/evaluate.py`, no n600 row, no launch, no paid dispatch, and no protected-file edits were performed.

## Paper And Package Custody

Primary sources inspected:

| Source | Custody note | Crosswalk-relevant content |
|---|---|---|
| arXiv abstract page, `https://arxiv.org/abs/2605.13642` | Abstract page reported submitted 2026-05-13 and latest revision 2026-07-24. | Scores are converted into calibrated p-values; supports split, data-efficient, and shift-aware variants plus false-discovery-rate selection. |
| arXiv PDF, `https://arxiv.org/pdf/2605.13642` | PDF footer reported `arXiv:2605.13642v3 [stat.ML] 4 Aug 2026`, which does not exactly match the abstract-page revision display. No local hash claimed. | Full text defines split conformal p-values with finite-sample correction, BH/FDR selection, exchangeability limits, weighted conformal selection, and martingale separation. |
| PyPI `nonconform`, `https://pypi.org/project/nonconform/` | PyPI latest visible release was `1.0.1`, uploaded 2026-05-20, Python `>=3.12`; no vendoring or install performed. | README mirrors the conformal-detector/FDR/weighted/martingale APIs and states the exchangeability limitation for temporal or spatial autocorrelation. |
| GitHub package repo, `https://github.com/OliverHennhoefer/nonconform` | Public source repository inspected through browser; no dependency imported. | Confirms package shape: sklearn/PyOD/custom detector wrapping, post-hoc FDP certificates, weighted workflows, and martingales. |
| GitHub paper repo, `https://github.com/OliverHennhoefer/nonconform-paper` | Companion evaluation repo inspected through browser. | Lists paper evaluations for conformalization strategy, calibration conditional, weighted conformal, and exchangeability martingale. |

The useful method is simple enough that Pact should implement the first pass internally: split-conformal p-values over already-owned alarm scores, plus BH selection over alarm families. Importing `nonconform` immediately is folded because the repo currently needs a scoped, scorer-free alarm calibration layer, not a new Python>=3.12 dependency.

## RECALL EVIDENCE

| Recall scope | Evidence found | Impact on CF1 verdict |
|---|---|---|
| `.omx/research/ddm_lp1_lane_program_20260803.md` | The #934 lane ratchet premise was refuted on shipped telemetry. The old elapsed-horizon default engaged 3/64, but the true total horizon engaged 0/64. A noise-null check with n=20000 placed observed sum-rises at percentile 0.7 and rise-count at percentile 0.6, so the fires were false positives. | The specific #934 instance is already corrected by an empirical null. CF1 still generalizes this into an auditable p-value/FDR contract for future guard fires. |
| `.omx/research/ddm_gd1_undecided_defaults_audit_20260731.md` | Gate estimators, windows, cadences, seeds, and dependency defaults were often undecided yet load-bearing. A1 used a 36-pair block plus SRS mean that over-weighted the block by 16.67x. | CF1 should cover free-variable threshold/cadence/horizon choices with predeclared nulls and multiple-testing accounting. It cannot repair a wrong estimand by itself. |
| `.omx/research/ddm_gc14_first_descent_20260731.md` | Alarm predicates are per-vehicle calibration objects; first fire is a calibration event. The UNDRIV alarm lacked a GT-reference term, and `n_points=5` aliased a 30-gate oscillation. | Conformal p-values are admissible only after the statistic is semantically correct and the calibration population is scoped to vehicle/stage. |
| `.omx/research/ddm_df1_retrain_contamination_classification_20260803.md` | A within-run statistic was incorrectly used as a between-run threshold. Real between-run floors were 11.5x to 67.9x larger at relevant rows; bias-correction conclusions were n=1, diagonal-only, and lacked a measured noise floor. | CF1 shift-aware tools help detect invalid calibration, but do not solve n=1 treatment design, missing randomized cells, or hidden effective-LR dose. |
| `.omx/research/ddm_na3_20260805/ddm_na3_receipt.md` | Prefix sampling created axis-dependent bias: pose prefixes were 2.535x to 4.207x harder than the population, while stratified n120 matched the population. | Calibration sets must be sampled from the target alarm population. Prefix defaults are not exchangeable calibration evidence. |
| `.omx/state/main_hot_state.md` | Own-vehicle pointer remained `S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]`; contest pointer borrowed/unmoved. | CF1 is apparatus-only. It does not move the frontier. |

## Per-Alarm Calibration Table

| Alarm or decision surface | Null and calibration population | Exchangeability grade | Procedure | Consumer | Falsifier |
|---|---|---|---|---|---|
| `lane_guard.ratchet` / `inertness_alarm` | Planned comparison horizon over same vehicle/window; use historical no-actuation windows or block bootstrap over differenced guard statistics. | Conditional yes if stage/window is stationary; weak otherwise. | Convert max/sum-rises anomaly score to conformal p-value, then apply BH across active lane-guard alarms. | #934 successor, b4s burn reseal, lane guard fire order. | Held-out null p-values are not super-uniform, or a martingale/change-point invalidates the calibration window. |
| `term_domination` | Per-stage term-share distribution from historical rows under same schedule and vehicle family. | Partial. Stage changes break pooling. | Split-conformal p-value per term family; BH over monitored terms. | v9 telemetry port and burn supervisor. | Same-stage held-out rows fail calibration, or the fire is driven by an unmodeled stage transition. |
| `term_inert` / inert-adaptive alarms | Engaged-but-inert historical rows, scoped by stage and vehicle. | Partial. Serial dependence requires block calibration. | Block-calibrated conformal score on term movement or residual debt. | Force-stack and curriculum gates. | Alarm fires disappear under block calibration or p-values bunch low on null rows. |
| `gnorm_hijack` | Per-stage gradient-norm share distribution, with heavy-tail robust score. | Partial and fragile. | Conformal p-value for excess share; optional martingale detects null drift. | Force caps and optimizer watchdogs. | Nonstationary gradients make calibration invalid before the fire. |
| `spike_deadlock`, skip-rate ceiling, partial-freeze/open-band accepted fraction | Bernoulli or beta-binomial score under stable data-loader and schedule state. | Partial. | Block conformal or exact-binomial calibration, with BH across sibling liveness alarms. | Run monitor and confound gates. | Serial dependence or loader/schedule drift dominates the calibrated null. |
| `frozen_epoch` / `ep_loss == 0` | Degenerate invariant; not an anomaly-score threshold. | No conformal use. | Fail fast as semantic liveness invariant. | Witness run monitor. | None. A p-value would be category error. |
| `A1_REALIZATION_GAP_ALARM` | Realized-vs-loss gap under matched estimator and per-pair telemetry. | Partial only after estimator repair. | Use HT/per-pair vector, not the old unweighted 36-pair gate mean, then calibrate gap score by stage. | A1 and realization-gap gates. | Old biased estimator remains in the calibration population. |
| `UNDRIV_EROSION` / class-topology alarms | Corrected GT-reference and topology statistic, scoped by class/stage/vehicle. | No until statistic is fixed; partial after repair. | Rebuild statistic first, then calibrate per-class p-values and apply BH. | GC14/R2 alarm recalibration. | The statistic still lacks the reference term or aliases oscillation. |
| Deadband horizon, `n_points`, gate cadence | Planned comparison count and autocorrelation structure. | Conditional. | Treat horizon/cadence selection as multiplicity-bearing decisions; derive calibration before guard use. | Gate scheduler and stop rules. | Horizon/cadence is chosen after seeing the alarm stream. |

## Seed Verdicts

1. **L1 alarm bank: ADOPT.** CF1 gives Pact a small, auditable standard: each alarm family needs a score, calibration population, p-value direction, exchangeability scope, and FDR consumer. This should cover `term_domination`, `term_inert`, `gnorm_hijack`, `spike_deadlock`, `frozen_epoch` exceptions, skip-rate ceilings, partial-freeze/open-band accepted fraction, A1 realization gaps, UNDRIV erosion, and lane ratchet guards. Consumer: confound gates, burn supervisor, run monitor. Falsifier: no held-out or block-calibrated null can produce super-uniform p-values.

2. **#934/lp1 ratchet fires: INSTANCE ALREADY EMBODIED, CLASS ADOPT.** The old 3/64 engagements were already classified as false positives by a noise-null/true-horizon analysis. A conformal/BH layer would also suppress them because their observed give-back/rise scores were not high-tail anomalies under the null. Consumer: #934 successor fire order and lane guard. Falsifier: a preregistered conformal p-value at the derived true horizon becomes significant and survives BH on fresh matched data.

3. **Shift-aware extensions: ADOPT AS INVALIDATION GUARD, not as cure.** Martingales and weighted/shift-aware conformal tools are useful for detecting stale calibration under boundary jumps, resume effects, stage transitions, and sample-selection drift. They do not fix the DF1 failure modes by themselves: n=1 treatment comparisons, missing off-diagonal cells, no measured between-run noise floor, non-randomized treatment assignment, and hidden effective-LR dose still require proper experimental design. Consumer: GC14 stop-rule recalibration, #848 controllable-treated-as-constant audits, and ratchet deadband horizon derivation. Falsifier: the alarm stream is too sparse or nonstationary for any calibration window to be meaningful.

4. **Hard boundary: ADOPT.** CF1 p-values can calibrate diagnostics only. They cannot authorize a score, replace frozen scorer cells, promote or kill a candidate, or infer contest-CPU/CUDA behavior from advisory rows. Consumer: all promotion gates and pointer updates. Falsifier: none; this is a governance boundary.

5. **Beyond-seed dependency decision: FOLD IMPORT, ADOPT INTERNAL CORE.** Implement split-conformal p-values and BH in-tree first for alarm metadata. Revisit `nonconform` only if weighted conformal, post-hoc FDP certificates, or martingales become active enough to justify the dependency and Python-version surface. Consumer: future L1 alarm registry. Falsifier: internal implementation grows beyond the small core or diverges from the package/paper semantics.

## Follow-Ons

| Status | Follow-on | Fire order |
|---|---|---|
| QUEUED-WITH-FIRE-ORDER | `cf1_calibrate_lane_guard_ratchet_null` | After #934 existence-hinge A/B or the next live lane-guard rise, compute preregistered conformal p-values over the derived true horizon and apply BH across active guard alarms. |
| QUEUED-WITH-FIRE-ORDER | `cf1_l1_alarm_calibration_table` | Before the next guard-bank expansion, add an alarm registry with score direction, calibration population, exchangeability scope, consumer, and FDR family. Start with lane guard, `term_domination`, `term_inert`, and A1 realization gap. |
| FOLDED | `import_nonconform_now` | Do not import now. The needed first pass is a small in-repo calibrator and JSON schema. |
| QUEUED | `cf1_shift_martingale_gate` | Add only as a calibration-validity monitor for p-value streams; never use it as score authority. |

## Boundaries

This artifact measured no candidate score and performed no authority evaluation. It did not run frozen scorer forwards, `upstream/evaluate.py`, n600 replay, CUDA, paid dispatch, or any long-running job. It does not claim a lower exact score.

It did classify where CF1 belongs: alarm calibration and false-discovery accounting. Any future conformal alarm must carry its calibration source, sample scope, exchangeability caveat, p-value direction, FDR family, consumer, and falsifier.

## NEXT_IF_RESUMED

1. Build the scorer-free alarm-calibration registry schema.
2. Add a tiny internal split-conformal p-value plus BH utility for L1 alarm rows.
3. Apply it first to the lane-guard ratchet null and A1 realization-gap alarm after their statistics and estimator surfaces are fixed.
4. Keep all outputs advisory/apparatus-only unless a separate exact byte-closed row moves the pointer.

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
