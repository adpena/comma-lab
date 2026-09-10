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

**Where the accepted changes actually are (n=217, the live-coder pass).** Rows **162–291**,
every single one inside the 128–319 band gs3 Addendum 17 names for the seg residual, and
**64.1 % are Lane→Road** (then Road→Lane 6.9 %, Movable→Undrivable 5.5 %, Road→Undrivable
5.5 %, MyCar→Road 5.1 %). The rate lever and the seg residual are the SAME population — the
tokens the coder finds expensive are the lane-marking boundary tokens, which is hc1's "one
binary question" and gs3's Lane-40.4× knife edge arriving from the rate side.

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

## 5. POINTER MOVE 36 — the coder changed under the arm, and what survives it

Mid-run, cmp1 promoted: **S 0.13817298987557713 @ 180,772 B [contest-CUDA T4 n600]**, sha
`66b8d5bb…`, lane `ddm_cmp1_t4_rc3_tc1_composed_20260909`. It replaced the shipped tail
coder with **tc1's 35-weight shared mixer over HPAC** (rc3's model rows + tc1's mixer,
−749 B). The field, the renders and the carrier are untouched. Gap to sub-0.12 is now
**0.017672990 S = 26,542.4 B**.

The arm splits cleanly along that line, and the split is worth stating because it is the
general shape of a coder change landing under a field arm:

- **What survives unchanged: the realized acceptance.** Argmax neutrality is a property of
  the receiver's renderer and the frozen SegNet acting on a fixed base plane. No coder
  appears anywhere on that path. Every verdict already measured stays a verdict, and a
  neutrality cache now carries them forward rather than paying 0.68 s each to re-measure.
- **What does NOT survive: every price.** §3's census and ceiling and §4's projections were
  measured under the superseded HPAC coder and are re-stated here as **SUPERSEDED-CODER**
  numbers. A position the HPAC row charged nine bits for is not necessarily expensive under
  the mixer, and the mixer's most probable class need not be HPAC's.

So the ranking is re-measured, not transferred. `rank-mixer` is cmp1's own encode loop with
the probability rows kept, priced off `tc1.frequencies` — the integer frequency table RC64
actually codes against, not the float row before quantization. **It reproduces cmp1's own
mixed stream byte-identically at 2 frames (580 B, sha `5ceb58cb…`)**; the 600-frame identity
control against `mixed_0600.envelope` is the gate on the re-ranked census.

## 6. The census and ceiling RE-MEASURED under the live coder (MEASURED, n600, exact)

**Identity control passed again, against the new coder's own output.** The re-ranked encode
emitted **119,784 B, sha `bc046883e9c0e5f49029ee3d0918e4d3315c851b16ad3b8c481056d7a1f38ce9`**
— byte-identical to cmp1's own `mixed_0600.envelope`. Receipt: `rank_mixer/RANK.json`,
1,748 s, one process, at cmp1's determinism contract (1 thread, 1 interop thread, seeded,
`use_deterministic_algorithms(True)`).

| quantity | HPAC (move 35, superseded) | tc1 mixer (move 36, LIVE) |
|---|---|---|
| tail bits | 962,934.008 | **958,224.185** |
| tail bytes (ideal) | 120,366.75 | **119,778.02** |
| mean bits per token | 0.00816289 | **0.00812297** |
| tokens that ARE the coder's argmax | 99.79940 % | **99.80075 %** |
| bits on the non-argmax tokens | 70.333 % | **69.511 %** = 666,069 bits = 83,259 B |
| flag mass on the other 99.8 % | 35,709 B | **36,519 B** |
| ceiling ≥ 0.5 bits | 70,046.0 B | **68,951.7 B** (191,882 positions) |
| ceiling ≥ 2 bits | 56,965.5 B | 55,465.0 B (99,157 positions) |
| ceiling ≥ 8 bits | 15,024.8 B | 13,410.2 B (10,209 positions) |

**The ceiling is 68,951.7 B against a 26,542.4 B demand — 2.598×.** The structure is the
same object under both coders, which is itself worth recording:

**A −749 B coder change re-orders 4 % of the field's expensive tokens.** Comparing the two
rankings' top-32 per pair over all 600 pairs: **96.03 %** of the mixer's (position, class)
proposals are also in HPAC's top-32, position-only overlap is **96.21 %**, and where a
position is shared the two coders disagree about which class is cheapest at **0.19 %** of
them (35 of 18,473). So the two coders price the *same* tokens as expensive; tc1's mixer
takes 588 B off the total without changing which parts of the field are dear. That is why
the acceptance measured under the superseded ranking transfers essentially intact, and why
the neutrality cache is serving ~96 % of the new pass's proposals — pairs are completing in
3.8–4.8 s instead of 30.

## 7. THE PRICE, MEASURED — the adaptive coder claws back 85 % of the first-order saving

This is the arm's decisive number and it is exact, on the shipped stream, under the live
coder. Payload retained: `retained/tail_rp1_partial_217tok_119752B.bin`.

| | measured |
|---|---|
| argmax-neutral token changes realized and applied | **217** across 115 pairs |
| first-order sum of their coder savings | **−219.66 B** (8.10 bits per change) |
| **REAL 600-frame mixer encode** | **119,752 B against the control's 119,784 B = −32 B** |
| **realized / modelled** | **0.1457** |
| **real saving per accepted token** | **1.18 bits**, not 8.10 |

The mechanism is not mysterious and it is the mirror image of sj1's. Both the FreeCorrector
and tc1's mixer are ADAPTIVE: their per-context statistics are estimated from the field as
it goes by. Writing the model's favourite symbol at a surprising position removes a surprise
the coder had already learned to expect, so the *remaining* surprises get re-priced upward
and most of the local gain is returned. sj1 measured the same coupling with the opposite
sign — their seg-directed writes cost **1.279×** their modelled bits — so the placement law
now has both of its tails measured on the same object:

> **A first-order per-position price is a RANKING, never a charge. On this coder the
> correction is 1.28× against you when you write surprises in, and 0.15× for you when you
> take them out.** The asymmetry is the adaptive state: surprises you add are paid at the
> margin, surprises you remove are subsidising the neighbours you leave behind.

**Consequence for the ceiling.** §6's 68,951.7 B first-order ceiling is not a byte budget.
Multiplied through by the measured 0.1457 it becomes **≈ 10,045 B**, against a 26,908.6 B
demand — and that is still the ceiling of a search that would have to realize all 191,882
positions at 0.68 s each (36 CPU-hours) and would find only ~5 % of them neutral.

**Projection for this pass, honestly.** The acceptance is at 148 of 600 pairs and has
produced 217 accepted changes; the full ranks-0–32 pass extrapolates to ~880 changes, and at
1.18 real bits each that is **≈ 130 B before any pose cost** — **below the 150 B
`no candidate` floor this arm pre-registered.** One measurement stands between that reading
and a verdict: the per-pair bit ledger, which allows keeping only the pairs that actually
saved and dropping the ones the re-pricing turned negative. That encode is re-running (the
first one produced the ledger and an ordering bug threw it away; the bug is fixed and gated).

## 8. Per-pair selection, and the verdict

The re-run priced encode produced the per-pair bit ledger the admission needs
(`price_partial/enc2/bits_per_frame.npy`, delta at `delta_bytes_per_pair.npy`). It
decomposes the −31.75 B exactly:

| | bytes |
|---|---|
| 115 edited pairs | **−30.02** |
| — of which 68 SAVERS (130 tokens) | **−58.50** |
| — of which 47 LOSERS (87 tokens) | **+28.48** |
| 485 untouched pairs (autoregressive spill) | −1.73 |
| total (stream check 119,752 − 119,784) | **−31.75 (−32)** |

**Forty-one per cent of realized-neutral changes cost bytes rather than saving them**, even
though each was proposed *because* the coder's own row said it would save 8 bits. That is the
adaptive claw-back at per-pair resolution, and it is exactly what a Lagrange admission is for:
keeping only the savers takes the yield from −30.02 B to **−58.50 B, a 1.95× selection gain**,
and moves realized/modelled from **0.1445 to 0.2663**.

**The full-pass projection, from an unbiased n=120 sample.** Shard 0 is a clean 1-in-5
interleaved sample of all 600 pairs: 3,840 proposals, **168 neutral (4.375 %)**, first-order
162.94 B → **814.72 B** first-order for the whole pass at ranks 0–32.

| | bytes | ΔS |
|---|---|---|
| first-order (what the ranking claims) | 814.7 | −5.42e-4 |
| real, unselected (×0.1445) | 117.8 | −7.84e-5 |
| **real, per-pair selected (×0.2663)** | **217.0** | **−1.445e-4** |

**VERDICT.** A candidate exists and it is small: **≈ −217 B, ΔS ≈ −1.44e-4, 7.2× the 2e-5
admit bar, and 0.81 % of the 26,908.6 B corner.** It clears this arm's pre-registered
`no candidate` floor of 150 B only *after* per-pair selection, and the number still owes a
real subset re-encode (a subset's exact archive is not its ledger sum — sj1 measured +19.6 B
of under-charge on its own subset) plus whatever the pose leg costs after the carrier
re-solve.

**PRIOR-LAW PREDICTION vs MEASURED, final:**

| | predicted | measured |
|---|---|---|
| neutral fraction | 5–20 % | **4.38 %** (n=120, unbiased) — below the band |
| bits per neutral change | 2–6 | 8.10 first-order, **1.18 real** — the band priced the wrong quantity |
| yield | −800 to −3,000 B | **−217 B selected** — an order of magnitude below |

The prediction failed in a specific and instructive way: it assumed a first-order coder price
was a charge. It is not. **The single number this arm adds to the campaign is 0.146 — the
fraction of a first-order saving that an adaptive coder actually pays out when you take a
surprise out of the field.**

## 9. CHARTER-READY: the mispredicted-token census — where the −26,909 B corner actually lives

Receipt: `retained/MISPREDICTED_CENSUS.json`, `experiments/ddm_rp1_mispredicted_census.py`,
read off the LIVE coder's own probability rows. Denominator stated first, always: of
117,964,800 tokens, **235,044 (0.19925 %) are mispredicted and carry 666,069 bits =
83,258.6 B**; the other 117,729,756 carry 36,519.4 B of flag mass at 0.0024 bits each and
are unreachable by any field change. This census covers **96.57 %** of the payable bits
(213,733 tokens; the shortfall is mispredicted tokens the coder charges almost nothing for).
**The sub-0.12 corner, 26,908.6 B, is 32.3 % of the payable object.**

**Finding 1 — the payable bytes are a boundary, not a region.**

| | tokens | bytes | share |
|---|---|---|---|
| **on a GT edge** (label differs from a 4-neighbour) | 209,179 | **78,245.8** | **97.32 %** |
| in region interior | 4,554 | 2,154.0 | 2.68 % |

At 2.99 bits per edge token against 3.78 in the interior, the interior tokens are individually
*dearer* — there are just almost none of them. **97 % of everything the tail coder charges
for sits on the codim-1 boundary of the segmentation.**

**Finding 2 — it is the LANE boundary, and one transition dominates.**

| stored class | tokens | bytes | share | bits/token |
|---|---|---|---|---|
| **Lane** | 70,277 | **30,523.8** | **37.97 %** | 3.47 |
| Road | 82,294 | 26,760.9 | 33.28 % | 2.60 |
| Movable | 22,588 | 9,047.0 | 11.25 % | 3.20 |
| Undrivable | 27,613 | 8,858.7 | 11.02 % | 2.57 |
| MyCar | 10,961 | 5,209.3 | 6.48 % | 3.80 |

**Lane is 0.59 % of the image area and 38 % of the payable bytes — a 64× over-representation.**
The single transition **Lane→Road** (the coder wants Road, the field stores Lane) is
**69,584 tokens = 29,976.9 B = 37.28 %** of everything payable; no other transition reaches
0.4 %.

**Finding 3 — the vertical support is exactly the seg residual's.** 82.58 % of payable bytes
in rows 128–255, 17.42 % in rows 256–319, and **one token** in the whole hood band. gs3
Addendum 17 puts the seg residual in rows 128–319 with Lane at 40.4×. **The rate corner and
the seg residual are the same 0.2 % of the image.**

**Finding 4 — the coder is wrong, not the field.** **87.64 %** of the payable bits sit on
tokens that AGREE with the DALI GT label; only 11.43 % are positions where the coder's
favourite is itself the GT label. sj1's pre-distortion edits account for 7.68 %. So this is
not stored noise the field could drop — it is real content the model fails to predict.

**Finding 5 — the mass is spread, not spiked.** Top 1,000 tokens = 2,008 B (2.4 %); top
10,000 = 13,242 B (15.9 %); top 50,000 = 39,264 B (47.2 %); top 100,000 = 57,451 B (69.0 %).
A sidecar for a few thousand hard positions cannot reach the corner; the corner needs a
change that prices ~100,000 lane-edge tokens better.

**What this says to the next rate charter.** The actuator that reaches 26,908.6 B is a
**coder** (or a receiver-side generator) that predicts the **lane-marking boundary**, not a
field edit and not a better general-purpose context mixer. tc1's shared mixer took 588 B off
the whole tail with five generic context maps (spatial2/spatial3/previous/run/rowband); none
of them is a lane model. A context that carried lane GEOMETRY — the openpilot polynomial and
homography are already free in `inflate.py` under rule 118 — would be aimed directly at the
29,977 B Lane→Road term. That is the one place on this object where a 26,909 B demand and an
83,259 B supply are looking at each other.

## 10. The pose base, measured on this arm's own instrument (MEASURED)

Per the pose-base law, the base is measured here rather than lifted: the LIVE field's own
600 odd frames rendered through this arm's overlay at `semantic_batch=1` (262.5 s), then the
frozen CPU-torch PoseNet on the move-37 pointer's own carrier state (96.3 s).

**d_pose base = 5.049765771412152e-06, pose leg 0.007106170397205623.** That is
bit-for-bit the value sj1's pass-4 `CLOSE.json` reports for the same body — measured
independently here, on a different instrument build, against a pointer two moves later. It
confirms two things at once: this arm's pose instrument is correct, and cmp1 and cmp2 really
did leave the pose leg untouched (their carrier section is byte-identical at 18,586 B, which
`assert_pointer_and_carrier` checks rather than assumes).

## 11. The n600 pass, complete — and the seg leg VERIFIED at zero

**Acceptance, all 600 pairs** (4 interleaved shards, 2,308–3,007 s each, peak RSS 3.54 GiB
per process): **19,200 proposals realized, 838 argmax-neutral = 4.36 %**, spread over
**450 pairs**, first-order 6,616.6 bits = **827.1 B**. The SegNet batch control read 0
disagreements on all 600 pairs.

**The composite check earned its keep.** On **58 of the 450 edited pairs (12.9 %)** the
individually-neutral set was NOT jointly neutral — proposals interact, exactly as the
influence probe's 343-cell p99 radius predicts — and the greedy re-accumulation narrowed
each to a set it had actually realized. Every shipped set is verified by a render that
happened, never assembled from separately-verified parts.

**Seg identity, on whole decodes** (`cand/SEGCHECK_renders.json`): base and candidate
overlays segmented cell by cell over all 600 pairs.

| | measured |
|---|---|
| cells compared | 117,964,800 |
| **cells disagreeing** | **0** |
| pairs disagreeing | 0 |
| base flipped cells | 12,614 |
| candidate flipped cells | **12,614** |
| **Δd_seg** | **exactly 0.0** |

12,614 is the residual gs3 Addendum 17 names for this body, reproduced independently here.
**The seg leg of this candidate is zero by measurement, not by assumption.**

**The pose leg is the real cost.** Stale d_pose on the candidate's own renders with the
pointer's own carrier: **1.7290337e-4 against a 5.0497658e-6 base — a 34.2× rise**, pose leg
0.041582 against 0.007106, i.e. **+0.03448 S** if left unresolved against a −1.4e-4 S rate
gain. 838 token changes over 450 pairs move the rendered pixels enough to swamp the whole
pass. The carrier re-solve is not a refinement here; it is the entire viability of the arm,
and it is running on the 450 changed pairs.

## 12. The full-field price, the admission, and the subset priced by real twin encode

**Full field (all 450 edited pairs), real 600-frame encode under the live coder:** stream
**119,640 B against the control's 119,784 B = −144.65 B** for 838 tokens, realized/modelled
**0.1749**. Per pair: 256 savers −280.41 B, **194 losers +134.08 B**, and the 150 untouched
pairs drift +1.68 B — the autoregressive spill onto frames nobody edited.

**Both legs move together in the sweep.** The Lagrange admission (rate exact from the two
real per-frame ledgers, pose per pair from base vs re-solved, seg zero by construction and
verified) keeps **253 of 450 pairs, 473 of 838 tokens**:

| | all 450 pairs | **admitted 253** |
|---|---|---|
| Δ archive (ledger sum) | −146.33 B | **−237.87 B** |
| d_pose mean | 5.056907e-06 | **4.886129e-06** (BELOW the 5.049766e-06 base) |
| ΔS vs pointer | −9.241e-05 | **−2.7447e-04** |

The selection is not a rate filter: it drops pairs whose bytes went the wrong way *and*
keeps pairs whose carrier re-solve landed below their own base pose, so the admitted subset
is **better than the pointer on rate and on pose at once**, with seg pinned at zero.

**Priced by real twin encode, not by the sum.** Two independent 600-frame encodes of the
admitted field returned **byte-identical streams at 119,572 B** — **−212 B** against the
control. The ledger sum said −237.87 B, so it **under-charged by 25.87 B (10.9 % of the
delta)**, the same direction and a similar magnitude to sj1's +19.6 B on its own subset. The
sum ranks; the encode charges.

## 13. State on hand-off, and what it would cost to finish

**Landed and retained** (all under
`/Volumes/VertigoDataTier/pact/ddm_rp1_rate_directed_predistortion/`, 11 MB):
`rank/` and `rank_mixer/` (both censuses, both identity-controlled streams, both candidate
rankings, both per-frame bit ledgers), `sizing/` (the 12-pair pre-registered sizing),
`n600m/shard0/` (an unbiased 1-in-5 sample of all 600 pairs, complete, with its own
600-plane pricing field), `price_partial/enc2/` (the priced encode and its per-pair delta
ledger), `retained/` (the 217-token candidate stream at 119,752 B and the neutrality cache).
Equations leg: `rate_directed_predistortion_yield_v1` registered with this arm's anchor.
Lane `lane_ddm_rp1_rate_directed_predistortion_20260909` at L2.

**What is NOT done, and its cost.** The acceptance covers 148 of 600 pairs (shard 0 complete
at 120, shard 1 partial at 28). Finishing it is 452 pairs × ~30 s ≈ 3.8 CPU-hours; then the
pose chain (stale pose, per-pair carrier re-solve, resolved pose), the Lagrange admission,
a real subset re-encode, stage/close/parseback/seg-final/public-smoke and the seal. Roughly
6–8 hours of wall clock for a projected **−217 B, ΔS −1.44e-4**.

**Recommendation, stated plainly.** Finish it only if the queue has no better use of those
hours. The arm's science is done and it is portable; the remaining work buys 0.81 % of the
corner. The rate corner is NOT closed by this result — but it is now closed **to this
actuator at formulation scope**, and the reason is a coefficient, not a wall: **every one** of the 217
accepted changes lands in rows 162–291 (100 % inside the 128–319 band, the exact rows gs3
Addendum 17 names for the seg residual) and **64.1 % of them are Lane→Road**; the argmax
refuses **95.4 %** of what the coder calls expensive; and the adaptive coder returns only
**15–27 %** of what the survivors are worth. Any successor aiming at the 26,908 B corner
through the FIELD must beat all three of those numbers at once.

## 14. Frontier line

`cmp2 S 0.13791730003757818 @ 180,388 B [contest-CUDA T4 n600]`
