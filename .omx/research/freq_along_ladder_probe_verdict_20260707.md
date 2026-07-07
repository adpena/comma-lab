# freq_along ladder probe — MEASURED verdict (2026-07-07, RECOVERED post-credit-death)

**Agent:** freq_along ladder + warp-vs-noise probe (report-only). Pre-registration committed
`721c764fd`; the n600 measurement COMPLETED (15:49) but the agent died to Fable credit exhaustion
before writing this verdict memo. Data durable-on-disk (gitignored results):
`experiments/results/freq_along_ladder_probe_20260707/{freq_along_ladder_n600_20260707.json,
probe_state.ckpt.npz}`. **Authority:** `[macOS-CPU advisory] NON-PROMOTABLE`. Pointer 0.19110 UNMOVED.

## The question
Is the measured 3.2× along-tangent deficit the Candès–Donoho **parabolic-scaling ceiling**
(freq_along 8 = √64 = √freq_across; FEED-08f), such that **raising freq_along closes the deficit** —
or is 8=√64 a config coincidence? The probe ran the **ORACLE-CAPACITY form (form b)**: inject the
analytic lane band at each freq_along and score through R + frozen SegNet, n600. (Form a —
re-rendering the frozen ep650 field at a changed freq_along — was declared UNSUPPORTED: the trained
dir_feats keep the old dimension; honest scope note.)

## Measured table (n600, mod32cap ep650; GT-conditioned validity control per the 4db610af2 lesson)
| cond | f_along | d_seg | lane_recall | lane_fp | scoreable? |
|---|---|---|---|---|---|
| ctrl_GT | — | **0.00000** | 1.0000 | 0.00000 | control passes (GT scores itself perfectly) |
| c1_witness | — | 0.00315 | 0.7795 | 0.00060 | reference |
| cSOLID | — | 0.01356 | 0.7326 | 0.00428 | solid-band baseline (worse) |
| cDC | 0 | 0.00731 | 0.7948 | 0.00315 | **yes** (gt_sep 0.189) |
| cF8 | 8 | 0.00756 | 0.7472 | 0.00264 | **yes** (gt_sep 0.256) |
| cF16 | 16 | 0.00727 | 0.7404 | 0.00243 | NO (gt_sep 0.022 < 0.05 floor) |
| cF25 | 25 | 0.00716 | 0.7367 | 0.00234 | NO (gt_sep 0.011) |
| cF32 | 32 | 0.00714 | 0.7353 | 0.00231 | NO (gt_sep nan) |
| cCOMB | — | **0.00695** | 0.7291 | 0.00221 | best oracle-capacity condition |

## VERDICT — parabolic-ceiling "raise freq_along" hypothesis: NOT CONFIRMED (indeterminate-leaning-negative in this form); the COMB is favored
1. **The ladder is FLAT.** Across freq_along 0→32 the scoreable d_seg is 0.00731→0.00756→(0.00727→
   0.00714, the last three NOT scoreable) — a tiny, non-monotone-at-the-start drift ≈ noise, NOT the
   monotone collapse the "parabolic ceiling closes if you raise the along-budget" reading predicts.
2. **The GT-control gate did its job** (the adversarial-review lesson APPLIED): freq_along ≥16 rungs
   have GT-vs-degraded separation below the 0.05 floor (0.022 / 0.011 / nan) → the instrument CANNOT
   discriminate them; those rungs are **INDETERMINATE-at-this-resolution**, not evidence. Only the
   coarse (0, 8) rungs are scoreable, and they are flat.
3. **cCOMB is the best condition** (0.00695 < every freq_along rung) — consistent with the Mallat
   second-order-scattering ranking (comb = carrier×envelope, FEED-08f) and the group-theory
   orbit-coding framing (FEED-08k): the dash structure wants the **modulation carrier (comb)**, not a
   higher linear along-frequency budget.

**Council consequence (T5 gate item 7, §23):** the freq_along basis lever is NOT a confirmed simple
win; the in-training comb A/B (FEED-08c) remains THE arbiter for the lane class. 8=√64 stands as a
config observation, not a demonstrated actionable ceiling in the oracle-capacity form. The
form-a (train a witness at higher freq_along) route is the only untested discriminator → T5 candidate.

**NOT resolved (honest):** the equal-L² warp-vs-noise (Mallat GIS row 5) probe design was
pre-registered but its results are not in this JSON (the agent died mid-second-probe); form-a
(retrain at higher freq_along) is unbuilt by construction. Both are T5 crucible candidates, not gate
blockers.
