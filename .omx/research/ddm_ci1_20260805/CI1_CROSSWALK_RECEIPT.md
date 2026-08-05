---
arm: ddm_ci1
title: "CI1 crosswalk receipt - Causal Inference with Unstructured Outcomes"
date_utc: 2026-08-05
paper: "Causal Inference with Unstructured Outcomes"
paper_url: "https://arxiv.org/abs/2608.03085"
authors: "Kevin Christian Wibisono, Yixin Wang"
submitted: "2026-08-04"
axis: "[research-only] scorer-free literature and repo crosswalk"
score_claim: false
promotion_eligible: false
scorer_forwards_run: 0
evaluate_run: false
launches_run: 0
pointer_moved: false
tokens: "[no-triality] [p0-ledger-ok]"
---

# CI1 Crosswalk Receipt

## Answer First

CI1's useful import is **not a new score, not a replacement objective, and not a
promotion instrument**. The Maximally Contrasting Feature (MCF) idea is useful
only as a **post-measurement diagnostic and A/B attribution layer** over decoded
frames, argmax maps, margins, edge/verb ledgers, and persisted scorer fields.
It should choose or report the feature family where a matched treatment/control
contrast is most pronounced, then name the changed structure. It must never rank
archives, promote rows, or replace exact `upstream/evaluate.py`.

Verdict: **ADOPT a constrained MCF-style contrast report; mark much of the
instrumentation ALREADY-EMBODIED; FOLD all authority uses.** The repo already
has fixed feature maps for class, edge, flip, per-pixel margin/saliency, score
atlas, and subset validity. CI1 adds a clean rule for how to select and label a
contrast feature after matching, plus a reminder that positivity/overlap and
population exchangeability are identification conditions, not polish.

Source boundary: I did not obtain the 68-page PDF/full text through the current
toolchain. The arXiv abstract page was available and states the paper's title,
authors, 2026-08-04 submission, MCF framing, HTE scoring, and unstructured-
treatment extension. The paper PDF/HTML/source were not retrievable through the
available browser/shell path; the in-app browser was unavailable; DeepXiv access
for this non-demo paper was not available. Claims about CI1 below are therefore
`ABSTRACT_GROUNDED` or `INFERRED_FROM_ABSTRACT`; the unstructured-treatment
extension is cross-checked against the related Wibisono-Wang paper at
`https://arxiv.org/abs/2608.00657`, not treated as a substitute for CI1's full
text.

No scorer, no `evaluate.py`, no n600 job, no frozen-scorer forward, and no
launch was run. This is report-only.

## Recall Evidence

| query / scope | source(s) checked | finding | plan impact |
|---|---|---|---|
| `#931`, `na3`, prefix bias, sign inversion, subset selection | `src/tac/subset_selection.py`; `src/tac/subset_selection_gate.py`; `src/tac/canonical_anti_patterns/na3_subset_bias_builders.py`; `.omx/research/ddm_na2_negative_audit_20260803.md`; `.omx/research/ddm_ng1_20260805/ng1_negative_results_audit.md`; `.omx/research/ddm_p4x_lane_existence_birth_matrix_20260803.md` | Prefixes are a different population: pose prefixes are 2.54x to 4.21x harder while seg prefixes are about 3% to 5% easier; serial effective N is 40.22/600; n8 banks nothing. | CI1 HTE cannot bless prefix evidence. It routes to explicit mode/seed/stratum receipts and keeps n>=32, with n120/stratified evidence preferred when a row is to be banked. |
| `m91`, `pc2`, Road Lane, per-edge | `.omx/research/ddm_pc2_perclass_road_edges_20260802.md`; `src/tac/witness_control/force_class_edge_ledger.py` | Per-class summaries hide the object: Road<->Lane carries 49.2% of flips, and Road-incident edges carry 87.8%. The force ledger records class/edge/verb with verdict scope and magnitude kind. | MCF feature families should be edge/verb/pair/margin keyed first, not class-only. |
| `g3 score atlas`, hard-pair registry | `.omx/research/codex_findings_ddm_g3_score_atlas_20260722T204813Z_codex.md`; `.omx/research/ddm_g3_score_atlas_n600_20260722T204000Z/ddm_g3_score_atlas_receipt.json`; `tools/build_ddm_g3_score_atlas.py`; `src/tac/optimization/ddm_g3_score_atlas.py` | The score atlas already persists n600 typed pair rows, class x margin x topology mass, pose debt/sensitivity, byte ledger attribution, and hard-pair replay correlation, all advisory/non-promotable. | CI1 is a selector/report over existing atlas fields, not a new scorer pass. |
| `mp1`, per-pixel field, margin saliency | `.omx/research/ddm_mp1_lsb_misplacement_margin_join_20260802.md`; `src/tac/margin_saliency_map.py` | Per-pixel misplacement and margin-saliency instruments exist; they are diagnostic/loss-weighting assets, never score authority. | MCF can consume persisted fields only when their provenance and authority labels are carried. |
| `#824`, `eta(t)`, reset operator, confound gates | `src/tac/witness_dsl/spec_tr1_renderer_20260728.py`; `tools/build_ddm_bp1_arm_tickets.py`; `src/tac/confound_gates.py`; `.omx/research/ddm_df1_retrain_contamination_classification_20260803.md`; `.omx/research/ddm_na2_negative_audit_20260803.md` | The #824/#863 evidence is a two-cell diagonal from two non-independent runs. A later correction says the charter's "1.56 beta2 time constants" duration is wrong; the measured window was 3.00 tau, but n=1/no-noise-floor/diagonal-only and magnitude mismatch still make the verdict ungrounded. | CI1 identification language should name the real violation: non-randomized treatment, hidden effective-LR dose, missing off-diagonal cells, and no noise floor. |
| canonical equation registry, margin/stratum/score quotient terms | local `query_equations()` listing and targeted registry search | Existing ids cover the CI1 surface: `ddm_score_quotient_functional_v1`, `closed_scorer_taskspace_variational_functional_v1`, `margin_saliency_reachability_replaces_texture_proxy_v1`, `perclass_stratum_residual_carrier_taxonomy_v1`, and `ddm_v4b_composed_gate_instrument_fidelity_v1`. | No new canonical equation registration. CI1 crosswalk points to existing laws. |
| current hot state / live scorer ownership | `.omx/state/main_hot_state.md`; `.omx/research/ddm_bf1_20260805/BF1_RECEIPT_20260805.md` | Live own-vehicle line is `S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]`; od3 owns the scorer slot; BF1 queued its n600 scorer only after the slot is free. | CI1 performs no scorer work and queues no competing launch. |

## Paper Crosswalk

CI1 problem: ordinary ATE language is ill-posed when the outcome is an image,
text, or other unstructured object because there is no canonical subtraction
between outcomes. The paper's abstract says MCF learns a scalar feature-scoring
function that exposes the sharpest treated/control contrast, then extends the
idea to covariate-dependent scoring for heterogeneous treatment effects and to
unstructured treatments.

Pact mapping:

| CI1 object | Pact analogue | permitted use |
|---|---|---|
| treatment | a lever/config/archive/action, e.g. bias correction, reset state, regional phase, BF1 section form |
| unstructured outcome | decoded frames, RGB/YUV tensors, argmax maps, margin fields, per-edge flip fields |
| learned scalar score | diagnostic contrast feature over persisted fields |
| MCF contrast | "where did this treatment actually change the scorer-relevant object?" |
| HTE | per-stratum effects by edge, verb, pair difficulty, margin bin, pose population ratio, and receiver section |
| identification | same-seed/matched-base/control completeness, positivity/overlap of strata, no hidden runtime/config confound |

The hard boundary is unchanged: exact score authority is the byte-closed archive
under the contest evaluator. MCF is a lens on already-measured objects, not an
object that can be optimized and called a score.

## Seed Verdicts

| seed | verdict | honesty label | conclusion | falsifier | named consumer |
|---|---|---|---|---|---|
| 1. A/B attribution instrument | **ADOPT + ALREADY-EMBODIED** | `ABSTRACT_GROUNDED`, `REPO_MEASURED`, `NO_BUILD` | Build no scorer. Specify an MCF-style contrast report over existing `g3`, `mp1`, margin-saliency, subset, and force-ledger fields. Existing instruments already provide fixed feature maps; CI1 adds the constrained "select the largest valid contrast from a predeclared feature family" wrapper and forces the selected feature to be named. | If `g3`/`mp1`/force-ledger outputs already reproduce the same maximum-contrast feature under matched A/B arms with provenance, the wrapper is redundant. If the MCF-selected feature has no change in exact argmax/margin fields, it is a false diagnostic and is folded. | `tools/build_ddm_g3_score_atlas.py`; `src/tac/optimization/ddm_g3_score_atlas.py`; `src/tac/witness_control/force_class_edge_ledger.py`; future verdict-clearance receipts. |
| 2. HTE / per-stratum verdict problem | **ADOPT** | `ABSTRACT_GROUNDED`, `REPO_MEASURED`, `DESIGN_RULE_ONLY` | Use covariate-dependent feature scoring only to decide which strata must be reported: edge, verb, pair difficulty, margin bin, pose/seg axis, and selection mode. This improves bounded-n power by preventing scalar collapse across heterogeneous effects, but it does not lower the n floor. n8 still banks nothing; n>=32 remains the minimum; n120/stratified is the bankable direction when available. | If stratified/random n120 still sign-inverts outside declared null bands, the feature family was not sufficient. If a CI1 stratum report chooses strata after seeing the outcome without predeclared families/cross-fitting, it is a fishing instrument and invalid. | `src/tac/subset_selection.py`; `src/tac/subset_selection_gate.py`; `src/tac/canonical_anti_patterns/na3_subset_bias_builders.py`; `tools/cathedral_autopilot_autonomous_loop.py`. |
| 3. Identification x confound apparatus | **ALREADY-EMBODIED + ADOPT SMALL CURE** | `REPO_MEASURED`, `DERIVED_DESIGN` | Existing L1 runtime alarms, L2 STRICT gates, and L3 verdict-clearance already encode most identification hygiene. CI1 adds naming: treatment assignment must be conditionally exchangeable given recorded base checkpoint, seed, selection mode, window, runtime, and hidden dose variables; strata need overlap/positivity. The cheap cure is a receipt row that names which assumption would fail before a treatment effect is read. | A matched A/B with same parent checkpoint, same selection receipt, same seed policy, same window budget, complete off-diagonal controls when needed, and no hidden dose shift still shows the contrast. | `src/tac/confound_gates.py`; `tools/build_ddm_bp1_arm_tickets.py`; `src/tac/witness_dsl/spec_tr1_renderer_20260728.py`; L3 verdict-clearance receipts. |
| 4. Exact scorer authority boundary | **ADOPT AS REFUSAL** | `GOVERNANCE_BOUNDARY`, `NO_SCORE_CLAIM` | MCF-learned scalar scores are diagnostics only. They cannot be authority, ranking currency, promotion evidence, or a substitute for `upstream/evaluate.py` on exact bytes. | The only thing that can supersede this is the contest oracle itself changing. Until then, any row promoting an MCF score is a NO-FAKE violation. | `tools/preflight_hook.py`; `src/tac/preflight.py`; every score/promotion receipt. |
| 5. Beyond-seed: unstructured treatments | **N-A / LIMITED TRANSFER** | `SIBLING_PAPER_CROSSCHECK`, `NOT_CI1_FULLTEXT` | Our treatments are mostly structured DSL flags, typed archives, and run programs. The unstructured-treatment extension transfers only as a caution: feature scores must distinguish modifiable treatment content from immutable context or style artifacts. It does not justify free-form treatment mutation outside typed DSL/receiver custody. | If a future treatment is genuinely unstructured, e.g. natural-language prompt policy or learned edit script, and can be encoded with reproducible treatment provenance and overlap/modifiability checks, reopen. | `src/tac/witness_dsl/`; `src/tac/followon_backlog_join.py`; future launch-ticket builders. |

## Identification Assumptions Crosswalk

| CI1-style assumption | Pact failure mode | existing apparatus | cheap cure |
|---|---|---|---|
| conditional exchangeability | comparing arms with different parent checkpoints, seeds, windows, runtime state, or implicit LR dose | reset-race DSL, optimizer-state gate, confound gates, L3 clearance | receipt row: `exchangeability_basis`, naming checkpoint/seed/window/selection/dose variables |
| positivity / overlap | stratum exists in one arm but not the other, or prefix selects a scene block rather than the population | subset selector, subset gate, NA3 anti-patterns | require selection provenance plus per-stratum denominators before reading treatment effect |
| consistency / SUTVA-like unit stability | archive/receiver change means the "same" treatment outcome is not on the same unit | receiver parse-back, byte-close receipts, force ledger magnitude kinds | require treatment/control to share receiver grammar and decoded-unit definition |
| no post-treatment feature leakage | learned diagnostic chases treatment-specific artifacts unrelated to scorer cells | exact scorer authority boundary, margin/argmax field checks | predeclare feature families; cross-fit/select on one slice and report on held-out or already-measured fields |
| estimand clarity | "works" collapsed into one scalar S hides edge/axis sign inversions | score atlas, per-edge ledger, subset ratios | report stratum and axis denominators before aggregate S prose |

## Authority Boundary

The MCF scalar is a **diagnostic projection**. It may summarize treatment
differences in decoded outcomes after the treatment/control objects are already
matched and measured. It may not be the objective that certifies improvement,
the ranker that orders archives, the substitute for frozen SegNet/PoseNet
cells, or the evidence for a pointer move. All promotion remains byte-closed,
receiver-closed, exact-evaluator authority on the exact archive bytes.

This also means CI1 cannot rescue proxy losses. A contrast feature that points
at Road<->Lane flips, low-margin pixels, or pose-hard strata is useful only if a
consumer then measures the real score cell it claims to explain.

## Follow-On Disposition

| item | disposition | fire order |
|---|---|---|
| Build a full learned MCF scorer now | **FOLDED** | Folded because scorer authority would be invalid, the full CI1 text was not available, and existing instruments already cover the fixed feature maps needed for a zero-scorer report. |
| CI1 contrast receipt over the next completed A/B | **QUEUED-WITH-FIRE-ORDER** | After a completed matched A/B has persisted decoded/argmax/margin/edge fields, run a scorer-free report over predeclared features: edge x verb x pair difficulty x margin bin x axis x selection mode. No new frozen-scorer forward. The report may only choose a diagnostic contrast and route the next measurement. |
| Canonical equation registration | **FOLDED** | Existing canonical equations cover the crosswalk. No new empirical law was derived here. |
| #863/#824 reinterpretation under CI1 | **FOLDED INTO EXISTING RESOLUTION** | Treat the current verdict as underpowered/diagonal/confounded, with the charter duration corrected to 3.00 beta2 tau per `ddm_na2`; fire only the already registered magnitude-matched race if that lane resumes. |
| #931 prevention | **ALREADY-EMBODIED** | `subset_selection`, `subset_selection_gate`, and NA3 anti-patterns already mechanize the prevention. CI1 only supplies vocabulary: population exchangeability and covariate-dependent strata. |

## Canonical Equation Registration

No new canonical equation was registered. The crosswalk maps onto existing
surfaces:

- `ddm_score_quotient_functional_v1`
- `closed_scorer_taskspace_variational_functional_v1`
- `margin_saliency_reachability_replaces_texture_proxy_v1`
- `perclass_stratum_residual_carrier_taxonomy_v1`
- `ddm_v4b_composed_gate_instrument_fidelity_v1`

Because this unit made no new measurement and derived no new law beyond those
surfaces, a new registry entry would be duplicate apparatus, not signal.

## What Was Not Measured

- I did not run `upstream/evaluate.py`.
- I did not run SegNet/PoseNet forwards.
- I did not launch n600, n120, or subset scorer work.
- I did not read the full CI1 PDF; see source boundary above.
- I did not change any protected file or upstream file.
- I did not move the contest pointer.

## Next If Resumed

1. If full CI1 PDF access becomes available, refresh this receipt against the
   actual paper text and mark every currently abstract-grounded inference that
   survives or changes.
2. Implement the queued contrast receipt only when a completed matched A/B
   already has persisted decoded/argmax/margin/edge fields. Do not consume the
   scorer slot to build the report.
3. Enforce the MCF boundary in review: any learned contrast is a diagnostic, and
   every score/promotion statement must still point to exact bytes and the
   official scorer.

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
