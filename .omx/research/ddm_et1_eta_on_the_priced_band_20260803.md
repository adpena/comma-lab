---
title: "eta measured: the seg-address band is DEAD, the regional phase field is seg-LIVE but pose-BLOCKED — and both published ceilings were measured on the wrong object"
unit: ddm_et1
task: 927 (TOP RESUME, named by ob1) + 935 folded in + ph1 block16 fold-in
date_utc: 2026-08-03
axis: "[macOS-CPU frozen-scorer advisory] NON-PROMOTABLE"
score_claim: false
promotion_eligible: false
pointer: "0.1910828242 [contest-CPU] UNMOVED"
own_vehicle_frontier: "S = 0.7910689 @ 353,805 B [macOS-CPU advisory] — UNMOVED by this unit"
verdict_scope_default: FORMULATION
---

# ddm_et1 — eta, measured on the objects that are actually priced

## §0 ANSWER FIRST

Two rungs were handed to me with published ceilings. **Neither ceiling was measured on the
object it was being spent against**, and finding that out changed both verdicts — one down,
one up.

| rung | published | what it was really measured on | **re-measured on OUR shipped vehicle** | verdict |
|---|---|---|---|---|
| gp1 A3 free band r=1 | 97.264% capture, −0.17466 S | a band seeded from a field **99.884% GT-agreeing** | capture **83.334%**, break-even η **0.61491** | **DEAD** (η = 0.3017 pose-viable) |
| ph1 block16 phase | +0.11186 S net | the **burn/ep399** field (458,739 flips) | gross **0.18039 S**, break-even η **0.1707** | **seg-LIVE** (η = 0.4817), **pose-BLOCKED** |

**Why one died and one lived is price, not physics.** The band's address costs 331,824 B; the
phase field's costs 46,247 B. Break-even η is `rate/gross`, so the band must clear **0.61491**
while the phase field need only clear **0.1707** — a **3.6× lower bar for the same realizer**.
That is gt2x's address-dominance law (~78% of explicit-production bytes are WHERE) paying out:
the phase field encodes *where* implicitly, in per-block offsets, instead of naming it.

**And the phase field is not bankable yet.** Its seg economics are strong, but pose-neutrality
**fails at scale** (§5): d_pose ratios **[1.064, 1.190, 1.000, 3.652]**. One pair in four costs
3.65×. No row here is byte-closed and the frontier is UNMOVED.

## §1 The vehicle catch — both, with receipts

**ph1's ceiling is a different vehicle's, provable from ph1's own arithmetic.** Its table gives
block16 `flips_left = 275,766` and `ΔS_seg = +0.15511`. Solving: `0.15511 / S_per_flip = 182,973`
flips fixed, so its baseline is `275,766 + 182,973 = 458,739` flips = **d_seg 0.0038888** — the
burn/ep399 field. **Our live-best ships 508,640 flips = d_seg 0.0043118**, 9.8% more to start
from. Per **L18** (ANCESTOR = LESSONS not NUMBERS) the +0.11186 was never ours to bank.

**gp1's "FREE" band is oracle-seeded.** Its seed is the frozen SegNet's argmax of the
`qa75_solve` render: **99.884% GT-agreeing, 3.717× more accurate than the render we ship**, and
already equal to GT at **90.053% of all 508,640 flips**. gp1 labelled it a proxy and marked its
capture BOUND — honestly. What was never done is **pricing the optimism**: **13.93 pp of capture**.

## §2 What I refute — in my charter, in my own first analysis, and in the fold-in

1. **Charter: "2.20× gp1's area at r=1."** 2.200× is the *structuring-element footprint* ratio
   (5 px → 11 px), confirmed in closed form. Realized band area is **1.299×**, bytes **1.004×**.
2. **My own over-correction.** I then wrote the SE divergence is "nearly free". True in **bytes**,
   false in **η**: sq1's 0.5406 → my **0.3017** on the narrower priced SE, **−44%**. SegNet reads
   regions, so a narrower band gives the solver less leverage. **The charter's instinct was right
   for a reason neither it nor I had stated.**
3. **Charter: "gp1's true Chebyshev convention."** gp1's is **L1/von-Neumann**. Neither
   implementation is Chebyshev though both docstrings say so (gp1 5 px, sq1 11 px, Chebyshev 9 px).
4. **The dominant divergence is the SEED, not the SE** — 13.93 pp vs 3.37 pp, **4.1×**.
5. **sq1 §4.2 cross-band transfer**: an η measured on its band (capture 86.701%) applied to gp1's
   gross (capture 97.264%) — **1.1218× inflation**. Corrected on its own band, A3 at η = 0.5406 is
   **+0.043881**, not +0.018003: **2.4× further from break-even**.
6. **sq1 priced against a stale gap** (0.654355209256714); live gap is **0.6189279**.
7. **Coordinator's "#939" does not exist as a repo id** — the **m89** two-store split. The
   272,869 B Lane-crop is rl1's and carries **no realization efficiency** (implicit η = 1.0 on an
   n32×600 extrapolation).

## §3 The r=1 band verdict — ob1's ladder closes DEAD

n=4, matched pair-for-pair to sq1's stratified selection (paired A/B; the SE is the only
variable). Break-even **0.61491** is DERIVED from the band's own bytes and capture — the same
derivation reproduces gp1's published F1 as **0.58408 vs 0.583**, the consistency check that
licenses everything else.

| arm | η pooled | ± sd | vs bar | seg-only net S | d_pose [subset] |
|---|---:|---:|---|---:|---:|
| (a) truth paint | −2.9861 | — | — | +1.29391 | 1074× |
| (b) solved, unconstrained | **+0.7328** | 0.0281 | clears | −0.04237 | **13.72×** |
| (c) solved + Q3 pose-null | **+0.3017** | 0.0246 | **fails 2.04×** | **+0.11253** | 1.0015× |

**12.7σ below the bar.** Arm (b) clears it but is pose-catastrophic; arm (c) holds pose and
misses by half. **`verdict_scope: FORMULATION`** — this realizer, band, and budget. Not killed:
arm (c) is 100% cap-pinned so 0.3017 is a **floor** — but it must **more than double**, and the
bar *rises* with radius, so widening cannot rescue it.

**Settled independently of η:** break-even rises monotonically — r=1 **0.61491**, r=2 **0.65755**,
r=3 **0.67888**. Capture buys more slowly than bytes, so **the band cannot be widened into
profitability**. That answers ob1's ">r=1" fork from the rate side without the realizer at all.
`verdict_scope: FORMULATION` (band-dilation as addressing, at this byte model); it does **not**
touch selective/ordered addressing.

**The uncaptured 16.666%** (the charter's 16.67% exactly) is two edges: **Lane→Road 76.67%** and
Movable→Undrivable 14.95% — 91.6% together. **I hypothesised lane erasure and refuted it:** the
render keeps **80.48%** of GT lane pixels and **no pair has zero**. Lanes are thinned and
displaced, not erased. `verdict_scope: MECHANISM`, lane addressability only.

## §4 The regional phase field — transfer vs re-solve

n600, our shipped field. Offsets are GT-solved and therefore **COUNTED payload**, priced with a
real coder (LZMA1 measured; the SMEVR column applies ph1's measured 0.80× advantage as a
**labelled PROJECTION**, never a coder we ran).

| arm | reach | gross S | bytes | net @ η=1 | break-even η |
|---|---:|---:|---:|---:|---:|
| ph1 published (its own vehicle) | 39.89% | 0.15511 | 64,953 | −0.11186 | 0.2788 |
| (A) TRANSFER ph1's offsets → ours | 25.62% | 0.11047 | 62,738 | −0.06869 | 0.3782 |
| **(B) RE-SOLVED on our field** | **41.84%** | **0.18039** | **46,247** | **−0.14959** | **0.1707** |

Transferring loses **36% of the reach** — those offsets were solved against a different field.
Re-solving recovers it and **exceeds ph1's published gross**, because our vehicle has more flips
to fix.

## §5 REALIZED eta on the phase field — seg LIVE, pose BLOCKED

Same instrument as §3, pointed at the phase field. Target is the **translated field** (what the
46,247 B buys), never GT; realizer is solve-from-frozen-head, never truth-paint (sq1's −3.764).

**Pooled η = +0.4817** (per-pair 0.4806 ± 0.0513, **n=4 of 32 at time of writing**) — **2.8×
above the 0.1707 bar.** Seg-only net **−0.05610 S = 9.06% of gap**, projecting S ≈ **0.7350**.

**Pose does NOT hold, and this blocks the row.** Ratios **[1.064, 1.190, 1.000, 3.652]**. The
smoke's 1.003× was **not** representative: one pair in four costs **3.65×**. Quoted as a
**subset-scoped gate, never folded into net S** — these pairs are 0.2692× of population on d_pose
(pose axis 4.6× skewed vs 1.05× for seg), so a subset pose delta cannot become a population ΔS.
**Consequence:** the phase-field row is **seg-LIVE and pose-BLOCKED**; it is not bankable until
pose is handled, and the Q3 rank-6 projection is the named cure whose cost on *this* field is
**unmeasured** (on a band it cost 44% of η — §3).

**Fidelity — a correction I owed.** The harness's `target_fidelity` is whole-frame, where ~97% of
pixels are untouched: **0.9984 is dilution, not evidence.** Band-restricted: **0.3168**. Off-band
collateral **0.00020**. The realizer reproduces the target on under a third of the band and still
clears the bar 2.8× — **because the bar is low, not because the realization is good.** The
misleading key is retained in the receipt as `fidelity_whole_frame_MISLEADING`.

## §6 Is block16 the knee? — it depends on η, and we land on the crossover

| rung | reach | gross S | bytes | break-even η |
|---|---:|---:|---:|---:|
| block16 (re-solved) | 41.84% | 0.18039 | 46,247 | **0.1707** |
| block8 (re-solved) | 60.03% | 0.25886 | 104,450 | 0.2687 |

Marginal: **+58,203 B buys +0.07847 gross**, so they cross at **η = 0.4939**. Measured η =
**0.4817 ± 0.0513** — *inside one sd of the crossover*. At the measured value block16 wins by
0.001 S (−0.05610 vs −0.05515): **not decisive**. block16 is the right pick because it is **2.3×
cheaper in bytes at statistically indistinguishable net** — ph1's risk-adjusted conclusion,
reached here for a sharper reason: **the knee is a function of the realizer's efficiency**, which
is exactly why realization had to be measured rather than assumed.

## §7 Convergence — a criterion, not a cap

- **The offset search (`solve_blocks`) is EXHAUSTIVE** over the (2·rmax+1)² integer lattice. No
  iteration, no stopping rule — it evaluates every admissible offset and keeps the argmax. It is
  **EXACT**, so the sm1/#874 class cannot apply to it.
- **The paint solve** (inherited from sq1) is Adam + best-realized-iterate under a **step cap**,
  **cap-pinned in 100% of runs**. Every η here is a **FLOOR**. Per #874 the cap is not
  raised-and-quoted; the budget **response** is measured on one pair at three budgets:

| steps | η | fidelity IN BAND | fidelity whole-frame | **d_pose** |
|---:|---:|---:|---:|---:|
| 10 | +0.3345 | 0.3168 | 0.9981 | 1.003× |
| 25 | +0.4085 | 0.4679 | 0.9984 | 1.064× |
| 50 | +0.4613 | 0.5259 | 0.9986 | **1.368×** |

**STILL-RISING at 50** — so every η in this memo is a floor, never an optimum.

**Two things this ladder settles that were previously assumptions.**

1. **η and pose damage are COUPLED, monotonically.** d_pose climbs 1.003 → 1.064 → **1.368×**
   as the solver is given more room. The seg gain is *bought with* pose, on this realizer.
   **Consequence: raising the budget is not a free lever** — it moves along a seg/pose tradeoff
   rather than up a pure-gain curve. This also explains §5's per-pair pose scatter: pairs where
   the solve found more useful descent are the pairs that pay more pose. Fire-order-3 (budget)
   and fire-order-1 (pose) are therefore **the same axis, not independent** — re-ordered in §10.
2. **The whole-frame fidelity metric is provably blind.** It sits at 0.998x across *all three*
   budgets while the band-restricted metric moves 0.3168 → 0.5259. A metric that cannot
   distinguish a 66% improvement in the thing being measured is not a weak metric, it is a
   non-metric. `verdict_scope: INSTRUMENT`.

## §8 Denominators

gap **0.6189279** = 0.7910689 − 0.172141 · seg leg **0.431179 S** (508,640 flips × S_per_flip
8.478009259e-7) · rate **25·B/37,545,489** · dS/dd_pose **31.3026** at the CURRENT operating point
(K3: never a shelf price) · all band/reach geometry is **n600 with no subsetting**, so **m96
cannot apply there by construction**; η is n=3–4 on sq1's **stratified systematic** selection
(0.9973 on flips/pair; a prefix would have been 0.9160).

## §9 Self-caught defects (mine, not inherited)

1. **Offset-search tie-break** resolved toward (−5,−5), buying entropy for identical reach.
   Seeding with the zero-offset agreement so (0,0) wins ties saved **15,490 B (25.1%)** at
   unchanged reach and dropped the bar **0.2279 → 0.1707**. Largest single move of the unit.
2. **Aggregator folded a subset-mean d_pose into a population ΔS** — the exact trap sq1 §1.6
   warns about. Net S is now seg-only; pose is an explicitly subset-scoped gate.
3. **Whole-frame fidelity** (§5) — flagged before it was quoted, then replaced.
4. A guard fix exposed a second latent crash in the recalibration script; the probe returned a
   real negative, which is the point of having one.

## §10 Follow-ons — FIRED / FOLDED / QUEUED-WITH-FIRE-ORDER

- **FIRED** — r=1 band verdict (§3) · block8 knee (§6) · band-restricted fidelity (§5) ·
  convergence criterion (§7). All four owed items measured, none deferred.
- **FOLDED** — sm1/#935 (cap-pinning instrumented in every arm) · sq1 §2.8's snap-tax IOU
  (discharged: **1.074× in bytes**, not the 1.1744× in pixels it was flagged as).
- **QUEUED, fire order 1 — POSE, and it now gates everything.** Measure the Q3 rank-6 projection
  **on the phase field** (not the band). Fire condition: the n=32 pose ratio confirms §5's
  scatter. If Q3 costs what it cost on a band (−44% η → 0.27), the row still clears 0.1707 and
  **survives**; that is the single decisive number left. Owner: this unit's successor.
- **QUEUED, fire order 2 — finish n=32 and byte-close** through `tac.submission_chain` (canonical
  chain, never a probe script; a PROFILES entry if the offset section needs grammar). Fire
  condition: fire-order-1 green. **Done-marker, resumable by anyone:**
  `/Volumes/VertigoDataTier/pact/ddm_et1_20260803/et1_b16_realization_n32.json` — the harness
  checkpoints after **every** pair, so harvest is pooling over `rows`; nothing is lost if killed.
  Relaunch command is in `et1_b16_n32.log`'s header; **single writer only.**
- **QUEUED, fire order 3 — RE-FRAMED by §7's ladder, do NOT fire independently.** "Raise the paint
  budget until the η slope flattens" was queued as the cheapest remaining gain. The measured
  ladder **refutes that framing**: d_pose rises monotonically with budget (1.003 → 1.064 →
  1.368×), so budget is **not** a free lever — it walks a seg/pose tradeoff. Fire it **only as a
  joint sweep with fire-order-1**, reporting the (η, d_pose) pair at each budget, and pick the
  budget by *net* S rather than by η. Firing it alone would buy seg and silently sell pose —
  precisely the seg-only verdict `sf1`/`uv1` forbid.
- **NOT QUEUED, with reason** — truth-paint at any band size (measured anti-productive to 79.3%
  of the field) · re-deriving D-support privacy / 22.70% blind (m86), frame_1 pose relativity
  (m87), the rank-6 yuv6 null (ph5o) — all reproduced here.

## §11 Pointer honesty

**The exact pointer did NOT move.** `0.1910828242 [contest-CPU]` UNMOVED. Own-vehicle frontier
**S = 0.7910689 @ 353,805 B [macOS-CPU advisory]** UNMOVED. Nothing is byte-closed; no archive was
built. A re-priced ladder, a corrected bar, and a measured realization efficiency are **MEANS**.
The phase-field row is the first candidate since pu2 that *could* move the frontier — projected
S ≈ 0.7350 on seg at η = 0.4817 — but it is **pose-blocked**, and until it is byte-closed and
evaluated **this unit has not achieved the goal.**

## §12 Receipts + STORES CONSULTED

Scripts (committed): `ddm_et1_band_convention_recalibration.py` `852a99e81b` ·
`ddm_et1_eta_on_priced_band.py` `3533db8780` · `ddm_et1_aggregate.py` `9caa43a35d` ·
`ddm_et1_ph1_block16_on_our_vehicle.py` + `ddm_et1_block16_realization.py` `79dc5c8c0f` ·
`ddm_et1_fidelity_and_budget_probe.py`.
Receipts: `/Volumes/VertigoDataTier/pact/ddm_et1_20260803/` — `et1_band_recalibration.json`
(n600, 18 combinations) · `et1_eta_priced_n32.json` · `et1_band_aggregate.json` ·
`et1_ph1_block16_our_vehicle.json` · `et1_block8_our_vehicle.json` ·
`et1_b16_realization_n32.json` · `et1_fidelity_budget.json`.
**Controls:** C2 (`SegNet(shipped frames) == cx1_argmax`) **PASS** · C3 **PASS** · C4 (full-frame
GT paste → η = 1.000, flips_after = 0) **PASS** — on the live-best pu2 decode, which independently
confirms **pu2's win was pure pose** (its frames give exactly the cx1 seg field).
Stores: `ddm_{sq1,gp1,gt3,ob1,sm1,si1,ph1}` memos · `experiments/ddm_gp1_free_band_and_net.py:77`
(the dilate defining the priced convention) · `ddm_sq1_eta_seg_realization.py:149` (the divergent
one) · `ddm_b2b_qa75_field_20260730/field_pass_manifest.json` (`source: qa75_solve`) ·
`ddm_ph1_20260803/offsets_n600_rmax5.npz` · CLAUDE.md class-order + authority ladder.
