# ddm_qbt2b_r9 — 10,020-step constrained margin ENDPOINT VERDICT (counter 697)

Date: 2026-08-29. Owner: MAIN. Axis: **[macOS-MPS governed n32 frozen-scorer advisory]**,
score_claim=false everywhere in this memo. Continuation of
`ddm_qbt2b_r8_constrained_margin_verdict_20260828.md` (same basis, same constraint tuple);
config `AUTHORIZED_N32_R9_10020_20260829.json` (canonical 1a2aed88…, identity 7655bed1…),
init = r8 stage-03 EMA endpoint. Run: counter 697, pid 42182/42183/42201, rc=0, elapsed
20,684 s (2.066 s/step realized), peak RSS 2,376 MiB, all_payloads_retained=true, 10,010
history rows retained (10 birth + 10,000 margin; cumulative steps 20,020→30,020).

## 1. Headline

**The milestone-adjacent flip landed 6.0% ABOVE the r8-law projection and the descent law
FLATTENED: segment exponent −0.696 (r7 window −0.781 · r8 window −0.963 · r9 segment
−0.696).** Not a wall — the flip still descends monotonically at segment scale — but the
box-class projection moves from ~89.8k to **~148.5k cumulative steps (+67.1 h of Metal at
the measured 2.066 s/step)** if the r9 segment law holds. The third-doubling read at
40,020 is now the DECISION evidence, and r10 is fired to buy it (§5).

- **seg_expected_flip_realized**: tail-25 **0.003530** (tail-100 0.003517, final
  0.003498) vs the r8-law projection 0.003331 at 30,020 — realized **+6.0%** high. The
  half-r6 milestone (0.005) stays comfortably crossed; the r8 fit's predictive skill
  degraded from ~18% step error (r7→r8) to a +6.0% level error over a half-doubling.
- **Exponent honesty note (instrument):** a naive 40-window OLS over the r9 window
  returned exponent −2.034 (coeff 4.57e6) — REJECTED as a fit artifact: the window
  contains the warm-start transient plus a flattening tail, and a power law fitted
  across that curvature is dominated by neither regime. The honest segment read is
  endpoint-to-endpoint on matched tail-25 means: ln(0.003530/0.004680)/ln(30020/20020)
  = **−0.6961**. The r8 memo's "steepened −0.9632" carried the same window-fit caveat
  ("windows/methods differ … the SIGN is the finding"); with the r9 segment measured,
  the sign has now REVERSED — the trend across segments is −0.78 → −0.96 → −0.70, i.e.
  NOT stabilizing, with the middle window-fit now the likely outlier.
- **Pose improved 1.45× in-window**: pose_mse_realized tail-25 5.162e-4 (final
  4.295e-4) vs r8's 7.472e-4. Gate-basis d_pose_hat 9.578e-4 vs r8's 9.965e-4 (only
  1.04× — the gate measures the full-selection EMA basis, the in-window number is the
  chunk-16 realized; both still 4–8× above the m110 budget 1.25e-4, riding down jointly).
- **Constraints held (tail-mean form), falsifier NOT fired**: Lane realized werr tail-25
  0.120297 chattering AT the 0.12 bound (final 0.105522, max 0.3595 early transient,
  ZERO 0.50-existence breaches in 10,000 steps); Movable tail-25 0.009082 at the 0.009
  bound (final 0.008626). λ maxima Lane 0.1259 / Movable 0.1234 (r8: 0.1355/0.1070),
  **0/10,000 ceiling contacts**. The primal-dual is holding both classes exactly at
  their bounds at ~2.5% of the λ ceiling — the constraint remains cheap.

## 2. Stage-05 admission gate (honest refusal, gap collapsed again)

`stage_05_gate`: admitted=false, correctly. **B_hat 122,171 B** (rate-feasible,
15,815 B under the 137,986 B sub-0.12 demand). **S_hat 0.4851** vs r8's 0.6406 (1.32×):
d_seg_hat **0.003059** (r8 0.004593, 1.50× — now 2.64× from the ~0.00116 box need) ·
d_pose_hat 9.578e-4 (r8 9.965e-4). Failed gates: s_hat · d_pose_hat ·
same_budget_qbw1_control (REFUSED_MISSING — discharged by refutation per the r8 §2
clause; the sealed QBW1 fire order adjudicated its own launch FOLDED_BY_SCORER_FREE_GATE
at 336,286 B vs the 137,986 B cap; no Modal spend).

## 3. Apparatus this unit

1. **History-slimming LANDED mid-window (commit 108fcd41f7)** — the r8 memo's item 3.2
   (the O(steps²) periodic-checkpoint retention wall) is PAID: sidecar journal mode
   (append-only `history_journal.jsonl` = history SoT for periodic checkpoints,
   row+patch events mirroring the in-memory mutation order, canonical-hash fail-closed
   at load, KEEP-THE-PAYLOAD pre-reanchor retention, torn-final tolerated /
   interior-corruption refused), stage-end checkpoints keep full embedded history (all
   identity checks unchanged), mode-aware storage projection, legacy configs
   byte-stable. 35/35 tests, 2 review passes. **Measured at the r10 compile: projected
   retention 2.45 GiB sidecar vs ~15.7 GiB embedded at the same cadence** — the wall
   cure is live. r9 itself ran the pre-slimming code (unaffected by construction).
2. **Receipt-semantics laws applied cleanly**: the keeper watcher (ARM … FINISHED rc=0)
   was treated as the authoritative completion signal; the run-dir Monitor was keyed on
   the REAL `--done` path from the launcher argv (the r8 false-positive cure) and
   confirmed the receipt without a false alarm. Both watchers agreed.

## 4. What r9 settles and what it opens

SETTLED (n32 seeded-stratified cohort, this basis):
- The constrained-margin mechanism remains stable and cheap: constraints chatter AT
  their bounds with λ at ~2.5% of ceiling and zero existence breaches across a further
  10,000 steps; pose continues descending jointly.
- The vehicle remains rate-feasible for the sub-0.12 box (122,171 B, 15,815 B spare).
- The r8 "steepening" did NOT hold: the segment law flattened to −0.696.

OPEN:
- Exponent stability now stands at 2.5 of the 3 pre-registered doublings with a SIGN
  REVERSAL in the trend — the third-doubling completion (40,020) is the decisive read.
- Box-class flip needs ~3.0× from here; under the r9 segment law that is ~148.5k
  cumulative (~67 h more Metal), vs ~90k under the r8 law.
- d_pose_hat 9.58e-4 vs the 1.25e-4 budget — 7.7×, still un-gated by a separate leg
  (m110 ratio discipline holds).

## 5. ROUTING — r10 FIRED + the pre-registered decision rule

**r10 FIRED (counter 698, pid 22681)**: margin_steps 10,000 from the r9 stage-03 EMA
endpoint (sha 8d35cbcf…) → **40,020 cumulative = the third doubling complete**, ~5.7 h.
Config `AUTHORIZED_N32_R10_10020_20260829.json` (canonical 36a40bdf…, identity
17a0769a…), **`checkpoint_history_mode: "sidecar"` — the history-slimming build's first
live consumer**; storage projection PASSED with 28.86 GiB headroom. Claims
ddm_qbt2b_r10_{scorer,metal}_20260829; done receipt QBT2B_R10_DONE; Monitor armed on the
real keeper path.

**Pre-registered r10 endpoint decision rule** (recorded BEFORE the run ends; segment
exponent computed endpoint-to-endpoint on matched tail-25 means, the §1 method):
- **e(30k→40k) ≤ −0.85 (re-steepens)**: the r9 flattening was transient — the box chase
  stays credible (re-project; ~110k-class); derive the r11/continuation decision from
  the refreshed law.
- **e(30k→40k) > −0.85 (stays flat or flattens further)**: two consecutive segments
  flatter than the r8 window-fit ⇒ the flattening IS the trend ⇒ STOP the n32
  constrained-margin chase per the trajectory-stopping law (caps=genus; a cap-free
  descending-but-decelerating trajectory is closed on its measured curve, not on a step
  cap), and take the vehicle-level decision (n600-scale vs the qbt family table vs the
  main-line campaign) on the accumulated 40k evidence. Projection under the r9 segment
  law: flip ≈ 0.002890 at 40,020.

## 6. Lane closure

Terminal rows appended for `ddm_qbt2b_r9_scorer_20260829` + `ddm_qbt2b_r9_metal_20260829`
(completed). Own-vehicle frontier UNMOVED this unit: the exact pointer stands at gb1
S 0.14811799921260607 @ 180,215 B [contest-CUDA T4 n600] — this unit produced a
law-flattening measurement, an honest advisory-gate refusal, and the retention-wall
cure, not an exact row.

---

## ADDENDUM (ddm_eq1, 2026-09-04) — the equations leg

**Law:** `trajectory_derived_stopping_law_v1` — `tac.canonical_equations.trajectory_derived_stopping_20260805` (`tac.canonical_equations`). **Relation:** IN-DOMAIN (trajectory segment, second doubling; §5 pre-registers the STOP rule).

r9's §5 records the decision rule BEFORE the next run ends — e ≤ −0.85 → chase credible, e > −0.85 → STOP — which is this law's projected-tail-gain test written as a pre-registered exponent bar.

This memo's Catalog #344 trigger was the word **stratified** — `"ratified"` is a substring of it, and the gate matched plainly. MEASURED by this arm: 16 of the 29 live memos (55.2%) tripped the gate ONLY that way, i.e. the gate was flagging the memos that did their sampling right. Fixed in the same batch (`(?<!st)ratified`); the disposition above stands on its own merit, not on the misfire.
