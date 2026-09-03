# ddm_qbt2b_r10 — third-doubling ENDPOINT: the pre-registered STOP rule FIRED (counter 698)

Date: 2026-08-29. Owner: MAIN. Axis: **[macOS-MPS governed n32 frozen-scorer advisory]**,
score_claim=false everywhere in this memo. Continuation of
`ddm_qbt2b_r9_constrained_margin_verdict_20260829.md` (the §5 decision rule binds this memo);
config `AUTHORIZED_N32_R10_10020_20260829.json` (canonical 36a40bdf…, identity 17a0769a…),
init = r9 stage-03 EMA endpoint (sha 8d35cbcf…). Run: counter 698, pid 22681/22693, rc=0,
elapsed 21,388 s (2.135 s/step), peak RSS 2,453 MiB, all payloads retained, keeper receipt +
safe_run status=ok concordant. Cumulative margin steps 30,020 → 40,020.

## 1. The pre-registered rule fired: STOP the n32 constrained-margin chase

The r9 memo §5 rule, recorded BEFORE this run ended: segment exponent e(30k→40k)
endpoint-to-endpoint on matched tail-25 means; **e ≤ −0.85 → chase credible; e > −0.85 → STOP
per the trajectory-stopping law (caps=genus)**. Measured:

- seg_expected_flip_realized tail-25 **0.002969** (tail-100 0.002965, final 0.003113) vs the
  r9-law projection 0.002890 — realized 1.027× high again.
- **e(30k→40k) = ln(0.002969/0.003530)/ln(40020/30020) = −0.6022 > −0.85 → STOP.**
- The segment sequence is now r7→r8 −0.78 · r8→r9 −0.696 · r9→r10 −0.602 — MONOTONICALLY
  FLATTENING across three independent segments. The r8 window-fit −0.963 stands confirmed as
  the outlier. Two consecutive flat segments were the rule's bar; three landed.
- Box-class projection under the r10 segment law: flip 0.00116 at ≈190,700 cumulative steps
  (in-window basis) / gate-basis d_seg_hat 0.002518 → ≈145k — either way +60–90 h of Metal
  WITH a still-flattening exponent. The chase closes on its measured curve, not on a step cap.

Pose: pose_mse_realized tail-25 4.283e-4 (final 4.119e-4), 1.21× better in-window; gate-basis
d_pose_hat 5.757e-4 (r9 9.578e-4 — 1.66× better, 4.6× above the m110 budget 1.25e-4).
Constraints: bounds held in tail-mean form, no ceiling contact (mechanism stable to 40k).
Stage-05 gate: admitted=false, correctly (d_seg_hat 2.17× from the 0.00116 box need).

## 2. The vehicle decision (owed by the rule; adjudicated here)

The rule's three options were n600-scale · qbt family table · main-line. **Adjudication:
MAIN-LINE.** (a) The flattening law gives no basis to expect n32→n600 transfer to beat its
18.75× per-step cost — scaling a decelerating law is buying more of a measured shortfall.
(b) The family's portable asset is BANKED: the QBFLOW body is the ONLY measured rate-feasible
family for the sub-0.12 box (B_hat ≈ 122 KB, ~15.8 KB spare), the constrained-margin
mechanism is PROVEN stable (35k steps, λ at ~2.5% of ceiling, zero existence breaches), and
every stage checkpoint + the full history journal are retained for warm-start. (c) The main
line holds a measured, receiver-closed −3,756 B rate opening (fcd1, rate-only ΔS −0.0025)
with its distortion legs executing NOW (ddm_fcd2) — orders of magnitude better EV per
compute-hour than +60–90 h of flattening descent.

**verdict_scope: VEHICLE (n32 seeded-stratified qbt2b constrained-margin chase) —
DEFERRED-pending-training-law-breakthrough, NOT a family kill.** Reactivation criteria:
(1) a mechanism change with a measured re-steepened segment (e ≤ −0.85) over ≥5k steps from
any retained checkpoint; (2) a warm-start from a materially better field (e.g. an fcd2-class
GT-benefit-edited field) re-entering the margin stage; (3) an operator directive to buy the
n600 read directly. All r7–r10 stage checkpoints + journals retained on APDataStore.

## 3. Apparatus: the history-slimming build PROVEN at scale (first live consumer)

`checkpoint_history_mode: "sidecar"` (commit 108fcd41f7) carried this run end-to-end:
- ALL 301 periodic checkpoints byte-IDENTICAL at **1,604,253 B** (step 33 == step 10,000) vs
  r9's embedded growth 1,637,341 → 9,443,677 B — the O(steps²) retention wall is CURED, not
  just projected: total periodic retention **541 MB** vs r9's multi-GB at a shorter run.
- `history_journal.jsonl`: 15,159,053 B, 10,318 events (10 birth + 10,000 margin rows + 308
  patch events), fsync-per-event, torn-final tolerated by design; stage-end checkpoint kept
  full embedded history (identity checks unchanged).
- Storage projection at compile (2.45 GiB sidecar vs ~15.7 GiB embedded) verified
  conservative against the realized 541 MB + journal.

## 4. Lane closure + routing

Terminal rows appended for `ddm_qbt2b_r10_scorer_20260829` + `ddm_qbt2b_r10_metal_20260829`
(completed). The scorer + Metal surfaces pass to the fcd2 distortion-legs execution
(`ddm_fcd2_distortion_legs_execute_20260829` charter, spawned at this boundary — its fire
trigger WAS this run's terminal state). Own-vehicle frontier UNMOVED this unit: the exact
pointer stands at gb1 S 0.14811799921260607 @ 180,215 B [contest-CUDA T4 n600] — this unit
produced the pre-registered trajectory close of the n32 chase, the at-scale retention-wall
proof, and the freed surfaces the fcd1 opening's distortion adjudication now runs on.

---

## ADDENDUM (ddm_eq1, 2026-09-04) — the equations leg

**Law:** `trajectory_derived_stopping_law_v1` — `tac.canonical_equations.trajectory_derived_stopping_20260805` (`tac.canonical_equations`). **Relation:** IN-DOMAIN ANCHOR (the law's STOP verdict, fired).

The memo already names it in prose ('the trajectory-stopping law (caps=genus)'); this line puts the id in the equations leg. The pre-registered exponent rule fired on the 30,020 → 40,020 segment and STOPPED the n32 constrained-margin chase.

This memo's Catalog #344 trigger was the word **stratified** — `"ratified"` is a substring of it, and the gate matched plainly. MEASURED by this arm: 16 of the 29 live memos (55.2%) tripped the gate ONLY that way, i.e. the gate was flagging the memos that did their sampling right. Fixed in the same batch (`(?<!st)ratified`); the disposition above stands on its own merit, not on the misfire.
