# ddm_rp1 — RATE-directed token pre-distortion: change the field where the shipped tail coder charges most, admitted only where the realized argmax does not move

Tokens: `[no-triality] [p0-ledger-ok]` · Arm: ddm_rp1 (Opus, close-supervision) · Charter:
`.omx/research/charters/ddm_rp1_rate_directed_token_predistortion_seg_neutral_pose_resolved_20260909.md`
· Live pointer at spawn: **S 0.13867171823146562 @ 181,521 B [contest-CUDA T4 n600]** (sj1 pass 4,
sha `b0ca809c…`), gap to sub-0.12 **0.018671718 S = 28,041.6 B** at 6.658589531e-7 S/B.
Axes: bytes EXACT (real encode); d_seg `[macOS-CPU advisory, jg1 instrument, DALI GT lineage]`;
d_pose `[macOS-CPU advisory, frozen CPU-torch PoseNet]`. `score_claim=false`. MAIN fires.

## 1. The inversion, and why it is not another rate arm

Every rate arm on this body has re-coded the field **as given** — rc1 and rc2 the model sections,
rc3 the model rows, tc1 the tail's context mixer, pc1/pc2 the carrier lattice, sm1 the container.
Every seg arm has **changed the field** to repair the rendered argmax, and paid for it. sj1's live
field carries **9,209 changed tokens** and its token stream is **120,367 B against the pristine
field's 113,419 B** — **6,948 B spent, 6.0358 measured bits per changed token**
(`ddm_sj1_multipass_token_predistortion/pricing/retained/S1_encode_sj1_pass4_subset.json`).

Nobody has changed the field to make it **cheaper** while holding the argmax. That is this arm.
The composition law ([[m148]]) says a closed leg survives only if another leg changes its object;
after gs3 Addendum 17 the seg leg is at a capability floor on the shipped renderer, so the object
that can still change is the field — and the leg that can still take it is rate.

## 2. Pre-registered estimator (written before the sizing data existed — commit order is the receipt)

The sizing tests a **stratified sample** of each pair's ranked proposals, because a head-only sample
cannot distinguish "no neutral proposals exist" from "the neutral ones are ranked below the cut" —
the false-negative shape that would kill this arm on a sampling artefact. Two projections are
therefore reported, and the **stop rule binds on the second**:

- **(a) sampled projection** — accepted first-order bits in the sample, scaled 600/`n_pairs`. This is
  what the tested proposals alone are worth; it understates a full search by construction.
- **(b) coverage-corrected projection** — for each rank stratum `s` with per-pair population `N_s`,
  tested `n_s`, neutral `k_s`, mean neutral saving `ŝ_s`:
  `bits_per_pair = Σ_s (k_s / n_s) · N_s · ŝ_s`, `bytes_n600 = 600 · bits_per_pair / 8`.

Both are **first-order** (the coder's own per-position price, before the placement law). sj1 measured
`realized / modelled = 1.2793` on its own seg-directed writes; the sign of that correction for
rate-directed writes is **not assumed** — fs2's law is that −log2 p is direction-dependent, so the
projection RANKS and a real 600-frame encode CHARGES.

**FALSIFIER (pre-registered, charter §PRIOR-LAW PREDICTION):** if (b) < 300 B, stop after the sizing —
the field's cheap tokens are already cheap and its expensive ones are the boundary, and the rate corner
is closed to field pre-distortion at formulation scope. If the n600 admitted total (real encode, pose
re-solved) is < 150 B, no candidate.

**PRIOR-LAW PREDICTION:** neutral fraction 5–20 % (sj1's own influence probe measured
`moves_with_any_argmax_change = 200 / 216`, i.e. **7.4 % neutral**, on a *seg-directed* family —
`probe/probe_0.json`); 2–6 bits per neutral change; central prediction **−800 to −3,000 B**.

## 3. Identity control and the coder census (MEASURED)

_To be completed from `rank/RANK.json`._

## 4. Sizing (MEASURED)

_To be completed from `sizing/SIZING.json`._

## 5. Verdict

_To be completed._

## 6. Frontier line

`sj1 S 0.13867171823146562 @ 181,521 B [contest-CUDA T4 n600]`
