# Task #455 — terminal joint-control assessment of the on-policy costate surrogate

**REVIEW STATUS:** `recovery-written-UNREVIEWED`. This new terminal artifact remains unreviewed until
three clean fresh-context passes complete. Earlier task-455 reviews were finding-producing and do not
count toward that seal.

**STORES CONSULTED:** `tools/corpus_query.py` loaded research (5715), equations (622), memory (1893),
DAG (505), council (277), tasks (96), and docs (92). Also loaded `CLAUDE.md`, `AGENTS.md`, the operating
manual, v7.5/v8 specs, current canonical pointer/task/lane/subagent surfaces, the frozen-SegNet necessity,
goldmine, SHARE_GE2, OPD, predecessor task-455 receipts, final source bundle, checkpoints, and terminal
receipt. Deliberately not consulted or actuated: paid/cloud providers, GPU dispatch, the live trainer,
protected runs, `upstream/evaluate.py`, or contest CPU/CUDA score surfaces.

## Answer first

**NO-GO — `fresh-eyes-review-pending`; formulation scope, not surrogate-family scope.** The nonlinear
EMA input-costate provider did not preserve the exact branch's descent trajectory under a valid common
controller. The tested provider must not be activated in the live witness trainer. `score_claim=false`;
the defensive contest-CPU pointer remains `0.1880443979880752` and was not moved.

Terminal receipt:
`experiments/results/onpolicy_costate_symmetric_timing_20260713T034500Z/boundary/measurement_receipt.json`,
SHA-256 `dc245467f9bd1fb63f3cfc0bbc1f092d133997a26d418a356738e8642c9abdf4`,
run-contract SHA-256 `db28e776a6616354c5bf059b0896bad1d7c2b64494a8ac13df0de3021a29305b`.
Axis: `[macOS-CPU advisory training-gradient]`; pair 0; saved boundary regime; seed 455; Torch fp32.

## Measured result

- **MEASURED:** the exact controller began from the explicit 1% parameter-norm maximum and halved
  until strict exact CE descent plus non-worsening exact through-R `d_seg` and exact PoseNet `d_pose`.
  It admitted three updates, with norms `2.1457042545080184e-4`, `3.3530351356603206e-6`, and
  `6.548907549586147e-9`, then reached bit-identical completion. This is a completion-guaranteed
  fractional control law, not an open-ended tuned constant.
- **MEASURED:** exact CE moved `0.018926450982689857 -> 0.01892581768333912`; exact `d_seg` moved
  `0.0035349527994791665 -> 0.003509521484375`; exact `d_pose` moved
  `157.90490388909893 -> 157.8940636790291`. The deterministic exact repeat was identical, so the
  measured CE, `d_seg`, and `d_pose` repeat floors were all zero. Across-seed variance is **UNKNOWN**.
- **MEASURED:** the EMA provider passed its fit admission gate. At matched step 2, however, the
  surrogate CE exceeded exact by `1.341104507446289e-7` and surrogate `d_pose` exceeded exact by
  `5.682410546068218e-4`. Max window regret was `1.7881393432617188e-7` for CE and
  `6.329965646330038e-4` for `d_pose`. Surrogate `d_seg` matched exact on this short window.
- **MEASURED:** the exact operational window was `9.635061959270388 s`; the surrogate window was
  `5.2367684580385685 s`; their ratio was `1.8398869525117787x`, or `45.6488346398229%` saved.
  Each window is explicitly the sum of symmetric complete per-step timers that include render,
  provider, renderer VJP, and candidate update. Validation and controller-search calls were excluded
  and separately hook-counted; this is not represented as an independent outer-window timer.
- **MEASURED:** this terminal floor yielded three updates, one exact anchor, and two surrogate
  non-anchor updates, so observed teacher skipping was `66.66666666666667%`. It does not validate
  the target K20 cadence or 95% recurring skip.
- **DERIVED, non-admissible projection:** applying the requested
  `K*t_exact/(t_exact+(K-1)*t_surrogate)` formula to measured step means gives
  `2.7652661849351374x` at K20. The receipt marks this projection as non-authority; fidelity failed
  before K20 and the exact branch was already at its event-conditioned floor.
- **MEASURED custody:** hook counts reconcile exactly at 160 SegNet forwards and 70 PoseNet forwards.
  Nine launch-source files verify against the preserved source bundle. Four distinct stage checkpoints
  verify by byte count and SHA-256. Restoring the terminal rolling checkpoint reconstructed stage
  `surrogate_window`, next step 3, without a teacher callback; recorded resume teacher calls are zero.

## Verdict derivation and boundaries

The pre-registered falsifier was any matched-step exact-metric drift above the deterministic repeat
floor after a valid exact control. It fired at step 2 for CE and `d_pose`. Therefore the tested
3/5-receptive-field, hidden-width-16, two-updates-per-label, EMA-0.8 formulation is **NO-GO** on this
single saved boundary regime and seed. The negative does not close nonlinear on-policy surrogates,
other capacities/update laws, other pairs, other seeds, or other hardware. The single-seed spine
leaves across-seed variance unknown, but this deterministic paired instance is sufficient to reject
this formulation's admission claim.

The economics fail independently: measured matched-window savings were 45.65%, below the frozen
forward's operator-supplied 78% share. The K20 projection also saves only 63.84% and cannot repair
the measured fidelity failure. No archive/evaluator score was measured.

## Triality and provenance

- DSL: `tac.witness_dsl.onpolicy_scorer_surrogate_policy` contains the default-off K20 target,
  nonlinear architecture, and joint fractional exact-control contract; it emits no live-trainer argv.
- Equation: `onpolicy_input_costate_surrogate_v1` records this terminal empirical anchor and scoped
  formulation verdict.
- DAG: `FEED-task455-joint-control-terminal` records the receipt, scope, economics, and pointer delta.
- Resumability: authenticated source custody, two rolling slots, and collection/exact/repeat/surrogate
  stage checkpoints are retained in the terminal result directory. No bytes were deleted or moved.

The only imported method principle is student-trajectory data collection from **Wenkai Yang, Weijie
Liu, Ruobing Xie, Kai Yang, Saiyong Yang, and Yankai Lin (2026), “Learning beyond Teacher:
Generalized On-Policy Distillation with Reward Extrapolation,” arXiv:2602.12125,
DOI 10.48550/arXiv.2602.12125**. The official arXiv abstract page was resolved on 2026-07-13.
Only its on-policy sampling discipline is borrowed; its language-model reward result is not treated as
a theorem about vision costates. No supporting paper was found that proves this CNN costate surrogate
preserves SegNet/PoseNet descent, and no such claim is made.
