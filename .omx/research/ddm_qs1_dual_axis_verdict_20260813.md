# QS1 six-pair Schur coupled candidate — dual-axis T4 verdict (2026-08-13, MAIN)

## The row [contest-CUDA T4 dual-axis component instrument, n600, batch=16 — matched-base]
- Candidate: sha `e474d4528aa2917db1433f8ef0ef63a943a15a511628542f98af45d8c972db9d` @ 186,329 B
  (+77 B vs cp135). Dispatch call `fc-01KZYKRGDZVRRQHPYQCBVFNMWE`, run-id
  `ddm_qs1_dual_axis_20260813_r2`, 712.0 s Tesla T4, ~$0.16 (#381, ≈$2.9 of $20).
  Retention: volume `comma-ddm-js1b-argmax-retained/ddm_qs1_dual_axis_20260813_r2`
  (seg field + PoseNet 6-vectors + inputs + repeat) — payload law honored; deterministic
  repeat d_pose IDENTICAL. First row bought on the js6b dual-axis worker: BOTH axes, one dispatch.
- **VERDICT: REFUSED. Net realized ΔS = +2.425702e-5 (matched component instruments).**
  - seg: 34,938 vs 34,970 flips → **−32 net flips, −2.712674e-5 S** (189 changed pixels)
  - pose: d_pose 6.885829861857928e-6 vs 6.885642960696714e-6 → **+1.126177e-7 S**
  - rate: +77 B → **+5.127114e-5 S** — the dominating term
- `score_claim: false` (component instrument; the floor is untouched).

## What this PROVES (the family-opening measurement)
**The Schur frame-0 pose compensation WORKS at the exact instrument.** Pose leakage per
six-pair candidate: +1.13e-7 S — vs re1's +5.7e-6 (one pair, unprojected) and JO1's
+2.05e-4 (six events, unprojected). The family law's killer (pose-dominated cell edits,
7–40× the seg gain) is CURED by ~50–1800×. Integer/receiver-realized cancellation
99.995% held through the full public decode. And the seg leg realized the campaign
line's largest measured win (−32 flips).

## What this REFUTES (instance) + the measured price structure
The candidate loses on RATE: compensation coding cost 77 B for 6 pairs (~12.8 B/pair).
- **Breakeven law (derived from S): 0.785 realized flips per compensation byte.**
- This candidate: 32 flips / 77 B = **0.416 flips/B — 1.9× short.**
- Realization efficiency: 32 net flips from ~189 changed pixels ≈ the screen's
  realized-seg uncertainty resolving LOW (~17–20% of screened target). Calibrate all
  future screens with this number.

## The two named levers (family OPEN, reopen conditions measured)
1. **Cheaper compensation coding** — get ≤6.8 B/pair (vs 12.8) at equal cancellation:
   coarser dc0 quantization along the pose-null slack, shared codebook across pairs, or
   restrict to the highest-cancellation subset (several pairs cancelled >99.8% — drop
   the expensive tail).
2. **Higher-value pairs** — target pairs whose screened seg value survives realization at
   >0.785 flips/B; the js6 bank's larger-target proposals were held, not exhausted.

## Consequences
- Task #1034 closed with this verdict; per-pair retained fields on the volume support the
  exact per-pair decomposition if the family reopens.
- Floor UNCHANGED: cp135 composed S 0.16195513827824176 @ 186,252 B [contest-CUDA T4, n600].
- Cadence: this IS a byte-closed dual-axis row (honest negative, banked). Three candidate
  rows today (JO1, re1, qs1), each cheaper and closer than the last: +2.16e-4 → +4.0e-6 →
  +2.4e-5-with-pose-cured. The remaining distance is ONE rate-coding rung.
