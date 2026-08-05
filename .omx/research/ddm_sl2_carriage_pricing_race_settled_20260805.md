# sl2 carriage pricing → the seg carrier race is SETTLED — 2026-08-05

Axis: [scorer-free byte pricing on persisted solve bytes] + recall of measured verdicts.
score_claim=false. Rows: /Volumes/VertigoDataTier/pact/ddm_sl2_20260805/carriage_pricing_rows.json

## 1. Measured: the explicit-edit family is DEAD at three orders of magnitude

All 32 persisted sq2 solves (band positions + paint values, 874×1164 camera res):

| stream | mean B/pair | note |
|---|---|---|
| positions (delta-coded, brotli-q11) | 1,030 | band structure compresses 40:1 — cheap |
| values raw (10,252 sites × RGB) | 30,756 | |
| values brotli-q11 | **28,180** | only 8% off raw — structureless |
| values lzma-9 | 27,901 | same |
| unique colors | 10,135 of 10,252 sites | the GN solve paints arbitrary continuous RGB per site |

n600 projection at measured prices: ~29.2 KB/pair → **17.5 MB → rate +11.7 S**. The corridor
(~90 KB total correction budget) allows ~150 B/pair ≈ **0.12 bits/site** — two orders below any
explicit per-site code (even a heroic 1 bit/site quantization-survival = 750 KB = dead; even
100× minimal-support pruning ≈ 165 KB ≈ net −0.02 marginal, on an assumption fp1/et1's band-family
deaths argue against). **No scorer run needed — the corridor arithmetic closes the family
analytically.** This SHARPENS the od-line ship-the-solve verdict (1.21 MB wholesale) with the
mechanism: it is the VALUES that are incompressible, because the solver's feasible-set member is
carriage-blind. (Named residual idea, tracked not fired: a rate-aware re-solve that picks the
CHEAPEST feasible-set member — but the floor is still ≥ positions ≈ 1 KB/pair = 600 KB = dead.)

## 2. Recall: distill-from-solve is already measured DEAD (dw1, 2026-07-30)

ddm_dw1_qa75_distill_window: KD arm ASCENDED at 2.01× the rate plain continuation descended
(split B−A = −3.82e-4 at 12.8× noise); teacher dark-knowledge does not survive uint8-STE/R at the
near-floor operating point; chart relaxation did not rescue it. Plain continuation itself still
paid (E2 0.0052766→0.0051147).

## 3. THE SETTLEMENT — all roads lead through the live training chain

The −0.128 S seg+pose prize (sl2 population projection, ledger 0d8eb8b9fc §5) transfers through
exactly ONE live route: **weights-as-carrier — the w3/w4 → jd1 chain.** Explicit description:
dead (§1, measured + analytic). Distill: dead (§2, measured). Grammar-description: same corridor
arithmetic as §1 unless amortized — and the renderer IS the amortization.

Consequences (binding on slot allocation under the standing GO):
1. The scorer/training slot serves the chain EXCLUSIVELY: window boundaries (canonical
   adjudicate_tail_slope) → w4 extensions while censored_still_descending → jd1 pose-finish at
   the winning endpoint → margin-weight A/B (#925) in the following window.
2. No further slot-time on post-hoc correction families without a NEW mechanism that beats
   0.12 bits/site AMORTIZED.
3. Chain worth, honest projection: burn to solve-proven d_seg 0.0010 at ~258–267 KB counted +
   jd1 pose ≈ S 0.56–0.65 territory vs 0.754 — the chain is worth −0.10 to −0.19 S over the
   coming windows IF the descent continues to the solve floor (gate d_seg 0.003884 and counted
   bytes both still falling at w3 ep~1195; solve floor 0.0010 is EXISTENCE-proven by sl2).
4. sl2's persisted frames remain the TEACHER/verifier surface (in-loop targets, NOT KD — dw1
   scopes the KD death to teacher-imitation objectives; jd1's in-loop scorer loss is not that).
