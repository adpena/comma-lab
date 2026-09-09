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

## 3. Identity control and the coder census (MEASURED, n600, exact)

**The identity control passed.** The instrumented encode emitted **120,367 B, sha
`0720536b5a9e90ef87a8d6a9d73a6be98130855aeaf7f6fb5dfc4862dd70d1b1`** — byte-identical to
the pointer's own shipped token stream (`tail_sj1_pass4_subset.bin`). A second, independent
check fell out of the logs: at frame 475 this run's `code_bytes_so_far` reads
`91738.63625354234`, the same eleven decimals as sj1's own pass-4 encode log. The prices
below are the shipping coder's prices, not a model of them.
Receipt: `rank/RANK.json`, 1,289 s, one process.

**The census — where the tail's 962,934 bits actually are:**

| quantity | measured |
|---|---|
| tokens | 117,964,800 |
| tail bits / bytes | 962,934.008 bits = 120,366.75 B (shipped 120,367 B) |
| mean cost per token | 0.00816289 bits |
| tokens that ARE the coder's argmax | 117,728,167 = **99.79940 %** |
| tokens that are NOT | 236,633 = **0.20060 %** |
| bits spent on those 0.2 % | 677,261.47 = **70.333 %** of the tail |
| bits spent on the other 99.8 % | 285,672.54 = 35,709.07 B, at 0.0024265 bits each |

So the tail splits cleanly into two objects. **35,709 B is the "yes, it is the prediction"
flag mass** — 117.7 M tokens each paying a quarter of a hundredth of a bit. No field change
can touch it: a token cannot be cheaper than the prediction it already equals. That mass is
the coder's business (tc1, rc2, cmp1), not the field's. **The other 84,658 B sits on 236,633
tokens**, and that is the entire object this arm can act on.

**The ceiling, exact and first-order** (rewrite each position to the coder's own most
probable class; `Σ log2(p_max/p_sym)`):

| saving threshold | positions | bytes if all were free | ΔS if all were free |
|---|---|---|---|
| ≥ 0.5 bits | 189,733 | **70,046.0 B** | −0.046641 |
| ≥ 1 bit | 151,933 | 66,555.8 B | −0.044317 |
| ≥ 2 bits | 99,499 | 56,965.5 B | −0.037931 |
| ≥ 4 bits | 43,119 | 36,855.6 B | −0.024541 |
| ≥ 8 bits | 11,288 | 15,024.8 B | −0.010004 |

**The corner is inside the ceiling: 70,046 B is 2.498× the 28,041.6 B demand.** That is the
first real news — every prior rate arm on this body was working against a few-hundred-byte
headroom, and this one is not. Whether any of it is *reachable* is the sizing's question and
nothing above answers it.

## 4. Sizing — realized argmax neutrality (MEASURED, 12 seeded pairs, 384 realized proposals)

Every proposal applied ALONE, re-rendered through the receiver's own renderer at
`semantic_batch=1`, re-segmented by the frozen CPU SegNet, accepted only on argmax identity
across all 196,608 cells. Receipts: `sizing/SIZING.json`, `sizing/SIZING_ROWS.jsonl`.

**Instrument controls, both clean:** the batch control (the unmodified plane pushed through
the batch-4 path and required to equal its batch-1 argmax) disagreed at **0 cells on 12 of
12 pairs**; the composite re-render of each pair's accepted set was neutral on **12 of 12**
with the greedy fallback never firing, so accepted sets compose additively at this size.

| | measured |
|---|---|
| proposals realized | 384 (12 pairs × 32, stratified over the ranking) |
| **neutral** | **18 = 4.6875 %** |
| mean saving, neutral proposals | 5.169 bits |
| mean saving, refused proposals | 6.441 bits (1.246× the neutral ones) |
| wall clock | 0.68 s per realized proposal, 25.8 s per pair |
| projection (a), sampled | **581.5 B** |
| projection (b), coverage-corrected | **2,431.2 B → ΔS −1.6188e-3** |
| **stop rule (binds on (b), threshold 300 B)** | **CONTINUE_TO_N600** |

**PRIOR-LAW PREDICTION vs MEASURED, carried openly:**

| | predicted | measured | verdict |
|---|---|---|---|
| neutral fraction | 5–20 % (sj1 probe prior 7.4 %) | **4.69 %** | **BELOW the band** |
| bits per neutral change | 2–6 | 5.169 | inside |
| yield | −800 to −3,000 B | (b) 2,431 B | inside |

The yield landed inside the band for the wrong reason: the neutral fraction came in *under*
the band and the per-change saving came in at the top of its band, and the two errors
cancelled. Recording that rather than the headline is the point of pre-registering.

**Two structural findings the search now has to obey.**

1. **Neutrality is FLAT in rank.** By absolute rank in the pair: 2.08 % (ranks 0–4),
   4.94 % (4–12), 4.55 % (12–32), 2.13 % (32–100), 5.93 % (100–400). There is no cheap
   region. What varies by 40× is not the *hit rate* but the *prize*: mean neutral saving
   falls 10.19 → 8.25 → 3.33 → 0.879 bits across those same strata. Value per realized test
   is therefore 0.20 / 0.50 / 0.38 / 0.071 / 0.052 bits, and **ranks 4–32 are worth 7–10×
   ranks 32–400 per unit of CPU.**
2. **The closed-form isolation predictor FAILS.** Neutrality against the fraction of the
   render's own 19×19 receptive window already equal to the proposed class: 4.64 % / 1.82 %
   / 6.73 % / 10.53 % across `[0,0.5) [0.5,0.8) [0.8,0.95) [0.95,0.99)`. Weak, non-monotone,
   and not usable as a filter. That is consistent rather than surprising: sj1's own influence
   probe measured a p99 Chebyshev influence radius of **343 token cells**, so what decides
   whether the argmax moves is SegNet's global context, not the token's local consensus.
   **There is no cheap screen for this actuator; neutrality has to be realized.**

**Honest uncertainty on (b).** 42.5 % of projection (b) (13.78 of 32.42 bits per pair) comes
from the rank-100–400 stratum, which rests on **7 neutral events in 118 tests** — a Poisson
interval of roughly ±38 % on that term alone. (b) is a projection, not a row.

## 5. Status and what is running

The stop rule cleared, so the n600 pass is running: **all 600 pairs, ranks 0–32 per pair**
(the value-per-test optimum above), singles mode, 5 interleaved shards, 2 live while sj1's
pass-5 shards hold the CPU. The first-order sub-projection for exactly that window is
0.794 + 4.025 + 7.502 = **12.32 bits per pair → 924 B**, and it is fully covered rather than
extrapolated, because ranks 0–32 were sampled exhaustively in the sizing.

Then: real-encode pricing of the accepted field (`experiments/ddm_rp1_price.sh`, twin
encodes), stale pose, per-pair carrier re-solve, a rate-objective/pose-constraint admission,
and the seal. **Composition note:** cmp1 (rc3 + tc1, −749 B) and sm1 (−263 B) are SEAL READY
and change the CODER and the semantic section, not the field, so they compose with this arm
by re-pricing rather than re-searching — but the byte delta of this field change must then be
re-measured under whichever tail coder is live at seal time.

## 6. Frontier line

`sj1 S 0.13867171823146562 @ 181,521 B [contest-CUDA T4 n600]`
