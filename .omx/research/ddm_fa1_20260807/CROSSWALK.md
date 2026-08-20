# ddm_fa1 FlowAdam Crosswalk

## Rigor Triage First

| FlowAdam claim | Grade | Crosswalk consequence |
|---|---|---|
| Clipped gradient-flow component is descent-directed under the paper assumptions. | DERIVED-SOUND | Usable as a transition-replay diagnostic only. |
| Soft injection gives a bounded, less discontinuous momentum transition than hard replacement. | DERIVED-SOUND for the bound; PLAUSIBLE-UNVERIFIED for reported performance | Admissible as an optimizer-state transition class, not as imported constants. |
| EMA difficulty mode switching chooses when/how to inject. | PLAUSIBLE-UNVERIFIED | Observer-only against existing Pact event gates. |
| Reported benchmark gains are real enough to consider as prior art. | PLAUSIBLE-UNVERIFIED | Not enough for Pact adoption without local state/update custody. |
| Implicit regularization explains lower test error despite higher training error. | PLAUSIBLE-UNVERIFIED | Lesson-only unless tied to a named DE-derived consumer. |
| FlowAdam should replace the current Pact optimizer stack. | SUSPECT | No adoption. |

## Ranked Crosswalk Table

| Rank | Requested row | Verdict | Named consumers | Falsifiers / blockers |
|---:|---|---|---|---|
| 1 | TRANSITION LAWS: FlowAdam soft injection vs our warm-start/reset corpus | ADOPT-CLASS | Existing consumers: `adam_v_variance_warmup_length_v1`, J4 warm-start reform, GC15 reset-operator analysis. Proposed exact lever form: `StageTransitionSoftVelocityBlend(gamma, source={prev_m, proposal_delta}, clip_multiple, warmup_steps)`, default-off. Proposed DSL home: curriculum/stage-transition DSL next to existing event-triggered curriculum wiring, consumed only by the current MLX witness trainer's stage-boundary reset path. | Falsify if a $0 optimizer-state replay does not reduce the GC15-style boundary effective-LR spike versus existing `v <- v_prev` / bias-corrected controls at matched update RMS, or if it points updates against the component descent direction. Blocker: no local update-matrix or boundary-gradient custody means no numerical adoption today. |
| 2 | MODE-SWITCH TRIGGER: FlowAdam EMA difficulty vs Pact event-driven schedule | ALREADY-EMBODIED | `EventBackstopGate`, `schedule_provenance_gate`, `campaign.decide_next_stage`, `trajectory_derived_stopping_law_v1`, TP1 typed plateau/tail exits, and the #344 NCDE observer path. | Falsify the "already better" stance only if an observer replay on logged trajectories predicts exact-positive stage handoffs earlier and with fewer component regressions than the typed local events. No control-plane replacement now. |
| 3 | COUPLED-PARAMETER REGIME: diagonal preconditioning failure vs low-rank/FiLM blocks | ADOPT-CLASS | Candidate consumers only after custody: QA83 factorized output head, QA84 rowband, FiLM/rank-collapse surfaces, sc1 rank-1 `e_p`, #140 low-rank pose codec, and coupled seg/pose JD1-style finishing states. Smallest raced design: one default-off update-RMS-matched replay on a saved coupled block, FlowAdam-style soft transition versus existing Adam/Muon/bias-corrected reset, no scorer slot. | Falsify if local gradient/curvature receipts show mostly diagonal behavior, if update-RMS matching erases the advantage, or if component metrics regress. Blocker: current recall found coupling evidence, not a FlowAdam-specific winning update. |
| 4 | IMPLICIT-REGULARIZATION leg vs DE-derivation framework | LESSON-ONLY | Conceptual consumer: closed-scorer variational/KKT/trajectory-stopping framework. Practical consumer: none named strongly enough for adoption. | Falsify by naming a concrete DE consumer where train-loss-worse/score-better is predicted and measured through the actual receiver/scorer. Until then this is explanatory prior art only. |
| 5 | Import FlowAdam optimizer or constants directly | SUSPECT | None. | Paper limitations are directly relevant: mini-batch trigger sensitivity, no full hybrid convergence proof, ODE overhead, manual mode choice, and small/medium benchmarks. Pact also lacks local update-state custody for this optimizer. |
| 6 | Treat paper benchmark percentages as Pact score forecasts | N-A | None. | External RMSE/error gains do not map to `100*d_seg + sqrt(10*d_pose) + 25*bytes/37545489`; no byte-closed archive or realized-through-R evidence exists. |

## Highest-Value Row

The highest-value row is Rank 1: `StageTransitionSoftVelocityBlend` as a class, not FlowAdam as an optimizer.

Reason: GC15 already found a Pact-native boundary reset failure with a concrete mechanism and no scorer cost to backtest. FlowAdam's soft-injection math gives a disciplined shape for testing a smoother transition law, but the test must be local: replay the stage-boundary optimizer state, compare against the existing `v <- v_prev` / bias-corrected reset controls, and report update spike, update RMS, descent alignment, and component-replay deltas. No paper constants become defaults without that local receipt.

## Non-Adoptions

- No direct `FlowAdam` optimizer import.
- No adoption of paper defaults for `alpha_s`, `alpha_c`, `tau`, `gamma`, warmup length, or clipping.
- No replacement of Pact event gates with the paper EMA detector.
- No score, frontier, archive, or launch claim.
