# Task #455 — on-policy frozen-SegNet forward surrogate: terminal assessment

**REVIEW STATUS:** `fresh-eyes-reviewed(2)-finding-producing`. The receipt/custody pass was clean-partial; two
subsequent method/architecture reviews found load-bearing evidence defects. This memo is
`recovery-written-UNREVIEWED` until the required final three clean passes complete.

**STORES CONSULTED:** `tools/corpus_query.py` loaded research (5715), equations (622), memory (1893), DAG (505),
council (277), tasks (96), and docs (92); also loaded `CLAUDE.md`, `AGENTS.md`, the operating manual, the v7.5/v8
specs, the latest sister findings/design/session memos, the lane/task/progress/equation registries, the frozen-SegNet
necessity memo, goldmine memo, SHARE_GE2 memo, OPD memo, current task source/tests, and the terminal receipt. Deliberately
not consulted or actuated: paid/cloud providers, the live trainer, protected runs, `upstream/evaluate.py`, or any
contest CPU/CUDA score surface.

## Answer first

**NEEDS-MORE — formulation and evidence-pipeline scope, not a family verdict.** `score_claim=false`.
The pointer remains expected-unmoved. The nonlinear provider is real and cheap at its own seam, but the current
experiment does not establish that replacing only the frozen-SegNet forward preserves matched exact-teacher descent,
nor does it isolate the requested whole-step economics.

The terminal evidence receipt is
`experiments/results/onpolicy_scorer_surrogate_20260713T020600Z/measurement_receipt.json`, SHA-256
`2812cd3a984fb063845d0690d79e319f2875aad4954630a87b43cda94d22211b`, run-contract SHA-256
`3638ee876adc702e062e319547ad6c9b8d1eb0377f6c01c55705f719c13c5c05`. It is `MEASURED / NEEDS-MORE` on
`[macOS-CPU advisory; torch-fp32; n=1 pair0]`. Its receipt bytes and 14 referenced checkpoints authenticate, but
deterministic source custody is **BLOCKED**: two uncommitted launch-source byte streams were not preserved, and three
of ten current source paths differ from launch. The machine-readable blocker is
`experiments/results/onpolicy_scorer_surrogate_20260713T020600Z/reproducibility_blocker.json`. The later `T023424Z`
rerun is interrupted non-rebuildable evidence; `interruption_correction.json` supersedes its inaccurate rebuildable
and resume-command fields.

## What was measured

- **MEASURED:** the K=1 exact controls accepted 5 early, 10 boundary, and 27 late updates, then the fractional
  halving law reached bit-identical completion. A fixed 40-step matched endpoint therefore did not exist.
- **MEASURED:** zero of three K=20 arms completed. At the first cycle endpoint, step 20, exact CE worsened in all
  three regimes. Costate cosine was `-0.1615319077` early, `0.1028255194` boundary, and `0.1603897922` late.
- **MEASURED:** all three K=20 step-21 refits increased the on-policy fit objective: early
  `2.3105893135 -> 2.3106775284`; boundary `1.9572508335 -> 1.9573390484`; late
  `1.8517711163 -> 1.8518996239`.
- **MEASURED:** boundary K=4 completed with `2.6229493094x` reported cycle speedup and `61.874978%` saved, but
  d_seg worsened `0.0031636556 -> 0.0042470296`. Late K=4 completed with `2.6257356434x` and `61.915435%`
  saved, but d_seg worsened `0.0036468506 -> 0.0048929850` and d_pose worsened
  `136.1073739131 -> 136.1448635502`.
- **MEASURED:** the provider-only mean was `0.027131992 s` over 135 non-anchor calls. The matched K=1 exact-costate
  provider mean was `2.155015959 s` over 42 calls. These are seam timings, not admission-grade whole-step economics.
- **MEASURED:** deterministic repeated d_seg/d_pose noise was zero on this single-seed spine. Across-seed variance
  is **UNKNOWN**. The sign-reversed negative canary passed only in boundary and failed in early and late.

## Why the result is not GO or a clean NO-GO

1. **DERIVED, load-bearing:** the fixed-horizon K=1 arm is not matched after its event-conditioned floor. The raw
   classifier filters blocked regimes before target classification, so it cannot turn the K=20 failures into a
   clean matched-arm negative.
2. **DERIVED, load-bearing:** the gate compares each arm with its own start and with sparse exact cycle checks. It
   does not compare a K=20 surrogate trajectory against a same-length, same-condition exact-teacher trajectory on
   d_seg/d_pose. Endpoint nonworsening is not trajectory parity.
3. **DERIVED, load-bearing:** the reported K=4 economics change more than the frozen forward. Exact anchors pay an
   adaptive exact line search and refit; non-anchor steps use the last anchor fraction. The measured speedup therefore
   conflates forward replacement, control-law change, and calibration cost. It does not isolate the requested
   `K*t_exact/(t_exact+(K-1)*t_surrogate)` counterfactual with an otherwise identical step.
4. **DERIVED:** training labels occur only at anchors. That is on-trajectory, but it is sparse anchor supervision,
   not dense student-trajectory OPD. At the first K=20 endpoint, the combined provider-and-stale-anchor-fraction
   controller fails exact CE descent in every regime. Only the early-regime costate cosine is negative, so this
   receipt does not isolate a CNN direction-loss claim.
5. **DERIVED:** the EMA shadow is used for inference and saved, but the fit-completion predicate tests live-model
   loss rather than EMA-shadow loss. Resume reconstructs the anchor costate with an extra exact teacher call outside
   recorded arm economics.
6. **MEASURED canary boundary:** two regime-level negative controls failed. Under P4, those directional readings
   cannot be promoted as clean family evidence.
7. **MEASURED custody boundary:** receipt and checkpoint hashes authenticate, but the exact uncommitted launch bytes
   for the provider and DSL were not preserved. This receipt is historical measurement evidence, not a deterministically
   reproducible or resumable authority surface.

**VERDICT_SCOPE for every negative above:** the 9-channel residual costate student; hidden width 8; eight fit steps;
EMA 0.997; K={1,4,20}; pair 0; saved early/boundary/late states; single seed 455; macOS CPU advisory. The broader
nonlinear on-policy forward-surrogate family remains open.

## Control laws and reactivation gate

- Target cadence is **DERIVED constant** `K=ceil(1/(1-0.95))=20`.
- K=4 is the named `K4_cadence_interpolation_canary` recess measurement; it is not a promotion arm.
- Width 8 and eight fit steps are **ASSUMED first-probe constants**, not derived laws. The named recess measurement
  that sets them is `shared_horizon_width_fit_grid_hidden_4_8_16_steps_4_8_16`; it must use the shared event horizon
  and identical controller required below.
- Exact step size uses a **fractional completion gate**: start at 1% of parameter norm, halve until exact CE descends
  or the candidate becomes bit-identical.
- The only admissible follow-up is a changed, pre-registered formulation that (a) stops every arm at a shared
  event-conditioned horizon before the exact floor; (b) runs exact and surrogate trajectories under the identical
  step controller; (c) measures matched d_seg/d_pose trajectory regret through R; (d) isolates only the teacher
  forward in timing; (e) evaluates the EMA shadow; and (f) supplies dense student-trajectory labels or a named
  tested sparse-label predicate. Until all six hold, `research_only=true` and no live-trainer argv is legal.

## Triality and custody

- DSL: `tac.witness_dsl.onpolicy_scorer_surrogate_policy` is default-off and emits no live-trainer argv.
- Equation: `onpolicy_input_costate_surrogate_v1` records the measured partial anchor and `NEEDS-MORE` disposition.
- DAG: `FEED-task455-terminal-correction` supersedes the earlier interrupted-status snapshot.
- Checkpoints: all 14 receipt-referenced records pass byte and SHA-256 checks. Resume authority is nevertheless blocked
  by missing launch-source bytes and by uncharged exact anchor-costate reconstruction. The repaired future harness
  preserves an authenticated source bundle and records reconstruction calls as operational resume overhead; no rerun
  was launched. The terminal result tree is 1.5 MiB, within the explicit 10 MiB local opt-in.
- Pointer delta: none. No score claim, provider dispatch, protected-run touch, or live-trainer edit occurred.

## Imported-method citation

The only imported method used at derivation is the on-policy-data principle: train the student on states generated
by the student trajectory, rather than a fixed teacher/offline distribution. The paper is **Wenkai Yang, Weijie Liu,
Ruobing Xie, Kai Yang, Saiyong Yang, and Yankai Lin (2026), “Learning beyond Teacher: Generalized On-Policy
Distillation with Reward Extrapolation,” arXiv:2602.12125, DOI 10.48550/arXiv.2602.12125**. The arXiv abstract page
was resolved directly on 2026-07-13. Its language-model KL/reward results are not transferred as a theorem here;
only the student-generated-trajectory sampling principle is borrowed. No paper establishes that this costate-CNN
formulation preserves SegNet descent, and this memo makes no such claim.
