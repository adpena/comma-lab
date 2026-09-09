# ddm_fe1 — per-pair FRAME-EMBEDDING pre-distortion: the realized search on the shipped renderer

Tokens: `[no-triality] [p0-ledger-ok]` · Lane `lane_ddm_fe1_frame_embedding_predistortion_20260908` ·
Charter `.omx/research/charters/ddm_fe1_per_pair_frame_embedding_realized_search_20260908.md` ·
Axes: d_seg `[macOS-CPU advisory, jg1/sj1 instrument, DALI GT lineage]`; d_pose `[cpu_torch fp32, DALI GT, n600]`;
bytes exact through the shipped container. `score_claim=false` everywhere below — no T4 row was fired by this arm.

Live pointer at every stage of this work: **S 0.13900437796841966 @ 181,645 B [contest-CUDA T4 n600]**,
lane `ddm_sj1_t4_token_predistortion_pass3_20260906`, archive sha `06c44dc4…`.

---

## 0. THE HEADLINE LAW — the renderer-coupling wall does NOT transfer to a per-pair change

The renderer arms are closed on coupling: rf1 measured 166.8, ft1 217.3, and pr1 still sat 41.5× over the payable
bar after a re-solve (`renderer_seg_pose_coupling_170_220_two_arms_20260903`). The stated reason was that renderer
weights move all 600 pairs at once, so there is no per-pair admission lever. **`frame_embed` is that lever, and on
it the wall is gone.**

MEASURED here, end to end on the shipped chain:

| pair · move | d_pose base | d_pose STALE | d_pose after `jg5.refine_pair` | recovery | resolved cost in S |
|---|---|---|---|---|---|
| 118 · d2 +0→+2 | 6.198610e-08 | 2.576839e-04 (**4157×**) | **8.439129e-08** (1.36× base) | **3053×** | **+2.6e-08** |
| 382 · d6 +0→−1 | 1.265234e-06 | 5.570474e-04 (**440×**) | **8.665944e-07** (0.68× base) | **643×** | (not credited — see §5) |

The stale rise is larger than any renderer arm has recorded — 440× to 4157× — and the carrier re-solve gives back
essentially all of it. Post-re-solve the pose leg costs **+2.6e-08 S per moved pair, about 3% of what one repaired
seg cell is worth.**

**The law: a change confined to ONE pair's render is payable on pose, however violent, because that pair's twelve
carrier coefficients are free to re-aim after it.** A change that moves all 600 renders is not, because the same
twelve coefficients per pair must then absorb a perturbation they were not solved against. The coupling number was
never a property of "touching the renderer"; it was a property of touching every pair at once. Any future arm that
can localise its render change to a pair should read the coupling memo as scoped to the all-pairs case and price
its own pose leg rather than inheriting 170–220×.

What this arm did NOT clear is a different wall, and it is worth naming beside the good news: the shipped 3-bit
FiLM lattice is too coarse to buy much seg (§4, §9). Pose stopped being the question; supply became the question.

---

## 1. What was verified at source (the charter's premise, corrected)

The charter said `frame_embed.weight_q (600, 8) i1` — an int4 field with 15 alternatives per code.
On the LIVE body it is **3 bits, not 4**.

| Fact | Value | verified-at-source |
|---|---|---|
| `frame_embed.weight` shape | `(600, 8)` | `candidate_pass3/candidate_runtime/cpr1/inflate.py:96` with `N = 600` (`:20`), `SEMANTIC_FRAME_DIM = 8` (`:28`) |
| FiLM consumption | one 8-vector drives `film = Linear(8, 192)` in ALL FOUR `TokenBlock`s for that pair | `cpr1/inflate.py:81, 84-86, 127` |
| shipped code depth | **3 bits** → signed domain `[-4, 3]`, **7 alternatives per code**, not 15 | SM3R mode-6 depth table, walked with the receiver's own `walk_sm3r` (`cpr1/rc1_adaptive_model_sections.py:249`) |
| codes run in the SM3R body | offset 466, length 1,800 B, count 4,800 | same walk |
| column scales (8 × fp16, offset 450) | 1.208984, 1.010742, 1.009766, 1.0, 1.204102, 1.045898, 0.873047, 1.068359 | same walk |
| shipped code histogram over 4,800 codes | `-3:28  -2:230  -1:892  0:2505  +1:904  +2:216  +3:25` | measured |
| semantic section chain | `brotli → CK2 2-plane un-interleave → RC1 rider` | `runtime/residual_archive.py:193, 237, 248`; RX1 `reserved = 0x7a` (CK2-semantic 0x2 SET, SZ1 0x1 clear) |
| section sizes | hpac 12,343 · semantic 30,246 · carrier 18,621 · tail 120,321 · header 14 · zip 100 | RX1 header of the live archive |

**Consequence recorded before any search:** one code step changes that dimension by ≈ 1.0 (the scales are ≈ 1.0)
against embedding entries that never exceed 3.6. The lattice is coarse; a "small nudge" is not available on this axis.

## 2. Controls (both PASS)

* **Identity control.** Re-rendering frame `2p+1` with the SHIPPED embedding reproduces the receiver's own decode
  **byte for byte**: 5 pairs, 15,258,600 pixels, `max |delta| = 0`, at the receiver's `semantic_batch = 1`.
  Receipt `probe/CONTROL.json`.
* **Instrument agreement.** My render→SegNet argmax equals sj1's decode argmax **pixel for pixel** on pairs 0, 199, 599,
  and the flip counts match (33/33, 26/26, 23/23).
* **Instrument agreement at scale, not on spot checks.** For every pair the n600 search has read, its base flip
  count (computed from MY render → SegNet argmax) is compared against sj1's count from the receiver's DECODE
  argmax. At 321 searched pairs: **321 of 321 agree exactly**, 6,669 flips both ways. The realized objective this
  arm accepts on is the one the shipped bytes produce. This control also closes a gap in the first two shards'
  receipts: they were launched before the end-of-shard "renderer restored to the shipped weights" assertion was
  added, so their receipts carry `restored_to_shipped_weights: null`. Their 200 pairs are nonetheless proven
  undrifted, by measurement rather than by assertion — every one of their base flip counts still equals the
  decode's.
* **Base seg leg reproduced exactly** (five figures and beyond): **12,866 flipped cells,
  d_seg 0.00010906643337673611**, carried to T4 by sj1's own instrument ratio as 1.0913879636e-04.
  All 600 pairs carry flips; median 19, max 100. Receipt `probe/base_flips_per_pair.npy`.
* **Coder identity.** `apply_semantic(restore_semantic(stream)) == stream` byte for byte, and re-encoding the SHIPPED
  codes reproduces the shipped RC1 stream (31,792 B) and, through the container, the header's **30,246 B**.

## 3. CLOSED-FORM-FIRST — the sign of ∂(boundary)/∂(code)

The FiLM path is affine in the embedding, so the whole realized chain (renderer → resample → SegNet) was linearized
about the shipped codes with eight forward-mode JVPs per pair. The one inexactness is DECLARED: the receiver's
`round` to uint8 at 874×1164 has zero derivative, so the linearization uses the straight-through surrogate there.
Per pixel the model repairs a wrong pixel when its top1-vs-GT margin goes negative and breaks a right pixel when its
top1-vs-top2 margin does.

Measured against the realized 56-move enumeration on the same pairs (`probe/SIGN.json` vs `sizing/SIZE_shard_*.json`):

| pair | Pearson r (predicted vs realized Δflips, n=56) | per-dim best DIRECTION agreement |
|---|---|---|
| 118 | **0.966** | 8/8 |
| 382 | **0.856** | 8/8 |
| 516 | **0.937** | 5/8 |

**The derivation ORDERS well and its MAGNITUDES are optimistic.** Sign agreement is 21/24 dims (87.5%), but the
first-order model predicts −9 flips where the realized move gives +3, because a full 3-bit code step is far outside
the linear regime. Verdict: usable as a proposal ordering, **never** as an admission signal — which is why every
number below the sign table is a realized argmax count.

## 4. SIZING — 12 seeded pairs, every single-code move (SCOPE, no verdict)

Seed 20260908, pairs `118 127 131 180 238 244 382 383 452 516 543 568`; 8 dims × 7 alternatives = 56 realized
evaluations per pair; **672 realized argmax evaluations**.

| pair | base flips | best Δ | best Δ % | reducing / 56 | worst Δ | best move |
|---|---|---|---|---|---|---|
| 118 | 17 | **−1** | −5.9% | 2 | +33 | d0 +0→+2 |
| 127 | 22 | +1 | +4.5% | 0 | +30 | d6 +0→−1 |
| 131 | 14 | 0 | 0.0% | 0 | +33 | d5 +0→−1 |
| 180 | 17 | +2 | +11.8% | 0 | +25 | d0 −1→−2 |
| 238 | 22 | +1 | +4.5% | 0 | +17 | d1 −1→−2 |
| 244 | 9 | +1 | +11.1% | 0 | +32 | d2 +0→+1 |
| 382 | 10 | **−1** | −10.0% | 1 | +20 | d6 +0→−1 |
| 383 | 12 | +2 | +16.7% | 0 | +22 | d6 +0→−1 |
| 452 | 33 | +1 | +3.0% | 0 | +53 | d4 −1→+0 |
| 516 | 59 | +3 | +5.1% | 0 | +45 | d1 +2→+1 |
| 543 | 46 | +4 | +8.7% | 0 | +41 | d4 +0→+1 |
| 568 | 23 | +2 | +8.7% | 0 | +46 | d3 −1→−2 |

* **664 of 672 moves make d_seg WORSE**; 5 are neutral; **3 reduce, each by exactly −1 cell.**
* Median move **+9 flips** on a base of ~24. Maximum **+53**.
* The best available single-code move is a **LOSS on 10 of the 12 pairs**. Summed best-per-pair: **+15 flips**.
* By step size, minimum realized Δ: step −1 → −1, +1 → 0, +2 → −1, −2 → +1, and **every one of the 178 moves with
  |step| ≥ 3 made the pair worse (minimum +4)**. This is the evidence for the n600 search's declared SCOPE
  reduction to |step| ≤ 2 — a reduction in how much of the lattice is walked, never in the acceptance mechanism.

**Stop rule (charter): "axis INERT if the best move changes flips by < 2% on every one of 12 pairs."
The rule does NOT fire — the axis is violently active (up to +230% of a pair's flips).** The axis moves boundaries;
it moves them almost entirely the wrong way. The shipped codes sit at a per-pair local minimum of the realized flip
count on this lattice.

## 5. Pose — the charter's expected killer, MEASURED not to be

Priced end to end on the three realized winners (`admission/PRICE_sizing_winners.json`): render the moved frame,
splice it over the live decode, measure d_pose stale, re-solve that pair's carrier from the LIVE coefficients with
`jg5.refine_pair` (damped GN + the ±2 lattice polish, the shipped chain), measure d_pose resolved.

| pair · move | d_pose base | d_pose STALE | d_pose RESOLVED | recovery |
|---|---|---|---|---|
| 118 · d2 +0→+2 | 6.198610e-08 | 2.576839e-04 (**4157×**) | **8.439129e-08** | **3053×** |
| 382 · d6 +0→−1 | 1.265234e-06 | 5.570474e-04 (**440×**) | **8.665944e-07** | **643×** |

The stale rise is enormous, exactly as the charter predicted; **the carrier re-solve absorbs essentially all of it.**
Post-re-solve, pair 118 sits at 1.36× its base and pair 382 lands *below* its base. In score terms the resolved pose
leg is +2.6e-08 S on pair 118 — three per-cent of one repaired cell's value.

Two honesty notes on this table:
* Pair 382's apparent pose GAIN was recorded here as **not creditable to the move** pending a control, because a
  re-solve on the UNMOVED render might have found some of it too. **That control has now RUN and it clears the
  move.** The admission path re-solves every candidate pair on its unmoved render first; on the two pairs measured
  so far the unmoved re-solve moves d_pose by **exactly 0.00e+00** — the live carrier is already converged for
  them — so the post-move resolved value is the move's own:

  | pair | base | control re-solve, UNMOVED | stale | resolved after the move |
  |---|---|---|---|---|
  | 6 | 5.5695e-07 | **5.5695e-07 (Δ 0)** | 1.6460e-04 (296×) | **4.9734e-07 (0.89× base)** |
  | 42 | 2.6859e-07 | **2.6859e-07 (Δ 0)** | 2.9497e-03 (**10,982×**) | **6.5550e-08 (0.24× base)** |

  So on this axis the pose leg is not merely payable — it can be a **GAIN**. A move perturbs frame `2p+1`, the
  twelve coefficients are re-solved against the new render, and the new optimum can sit below the old one. The
  10,982× stale rise on pair 42 is the largest this campaign has recorded anywhere, and it is fully recovered.
* The run that produced this table priced two moves on pair 118. The overlay holds one frame per pair, so both rows
  carry the LAST move's render. The pricer now refuses a repeated pair unless `--allow-repeat-pairs` is passed.

**This is a genuine correction to the prior.** The renderer-coupling law (`renderer_seg_pose_coupling_170_220_two_arms_20260903`,
rf1 166.8 / ft1 217.3 / pr1 k_post 13.82) closed renderer WEIGHT changes because they move all 600 pairs at once.
A per-pair FiLM move does not inherit that: the carrier has twelve free coefficients for that pair and they are
enough to re-aim the pose after an arbitrary re-render of that pair's frame. **Pose is not the wall on this axis.**

## 6. Rate — the wall is the CONTAINER, and it is a fixed fee

The semantic section is `brotli(CK2-interleave(RC1 stream))`. The RC1 payload is **range-coded**: changing one symbol
re-randomises every bit after it, so the brotli layer above it loses the matches it had on the shipped bytes.
Measured at the shipped container shape (`admission/RATE_LAW.json`, exact re-encodes, scorer-free):

| changed codes N | archive section Δ at the SHIPPED shape | Δ with a full container search | B per code (searched) |
|---|---|---|---|
| 1 | +65.0 ± 39.8 B | **+1.3 B** | +1.3 |
| 72 | +68.1 ± 29.4 B | **+28.0 B** | +0.39 |
| 200 | +80.8 ± 29.0 B | — | — |

Two findings, both reusable:

1. **The penalty is a fixed container-break fee, not a per-code price.** From 1 code to 200 the shipped-shape penalty
   is flat at ~+70 B; the marginal per-code term is ≈ +0.08 B. It is also NOT a length effect — length-preserving
   code changes still cost +40 B.
2. **A container search recovers most of it.** Brotli streams are self-describing and the CK2 choice rides in the RX1
   `reserved` byte, so quality, window and interleave are encoder-only choices the receiver is never told about.
   Searching q ∈ {9,10,11} × lgwin ∈ {16,18,20,22,24} × {ck2, plain} recovers **63.5 B at N=1** and **29.2 B at N=72**.
   The shipped body's own shape is `(ck2, 11, 16)`. Pricing an edit at the shipped shape overstates it about 2×.

**A length-only container tie-break ships a different stream; pin the shape by BYTE identity.** This is not a
footnote to the law above, it is part of it. `(ck2, 11, 16)` and `(ck2, 11, 24)` both compress the shipped rider to
**exactly 30,246 B** and they are **different 30,246 bytes** — brotli records its window size in its own header.
This arm's first null build therefore produced the right byte COUNT for the whole archive (181,645, delta 0) with
**30,129 of the semantic section's 30,246 bytes different**, while header, hpac, carrier and tail all matched. A
container search that minimises on length alone will happily ship a stream that differs from the pointer's for no
reason, and it will do it silently. The shipped shape is `(ck2, 11, 24)`, pinned by byte identity; ties in the
search go to it; and with that fix the null build is byte-identical to the live pointer, sha `06c44dc4…`. Same
genus as `available-field-vs-authoritative-field`: equal on the field you read is not equal on the field that binds.

**Control — the container search is NOT a free rate lever on the shipped bytes.** Re-compressing all three model
sections of the live archive over a 56-shape grid (q ∈ {5…11} × lgwin ∈ {10…24}) finds **+0 B** on every one:
hpac 12,343, semantic 30,246, carrier 18,621, all already at their brotli optimum. The search matters only for
EDITED bytes, where the shipped shape was chosen for a payload that no longer exists. Nobody should read this law
as free bytes lying on the pointer.

## 7. The exchange, and what the arm must clear

* one repaired cell = 100/117,964,800 = **−8.4771e-07 S**
* one archive byte = 25/37,545,489 = **+6.6586e-07 S**
* resolved pose ≈ **+2.6e-08 S per moved pair** (pair 118, the uncontaminated row)

With the container search, break-even against the −2e-5 admit bar needs roughly **N > 44 pairs** each repairing at
least one cell with one changed code. Priced at the shipped container shape instead, it needs **N > 80**.

## 8. PRIOR-LAW PREDICTION vs MEASURED

| charter prediction | measured | verdict |
|---|---|---|
| sizing: best move changes flips by ≥ ±10% per pair, or axis INERT at < 2% | up to +230%; best move is a LOSS on 10/12 pairs | axis ACTIVE, stop rule does not fire, but the motion is the wrong sign |
| 25–45% of pairs admit a move | **see §9** | — |
| admitted pairs repair 8–15% of their residual (≈1,000–2,500 cells n600) | maximum realized repair anywhere: **1 cell** (0.45% of the 672 sizing moves reduced at all) | **refuted, ~1 order of magnitude** |
| seg buys ≥ 4× its rate at ≤ 300 B | rate is a ~+28 B fixed fee (searched); seg buys ~2× it at N=72 | rate far cheaper than predicted, seg far thinner |
| d_pose rises 10–100× pre-re-solve; re-solve recovers ≥ 10× | rises **440–4157×**; re-solve recovers **643–3053×** | prediction directionally right, magnitudes both ~40× larger |
| pose-bound on >80% of seg-repairing pairs = the reason | pose costs ~3% of one repaired cell after re-solve | **not pose-bound** |

## 9. n600 realized search — COMPLETE (all 600 pairs)

**19,736 realized SegNet-argmax evaluations**, every one on the receiver's own render at `semantic_batch = 1`
against the DALI GT table. Base flips over the 600 pairs: **12,866** — the pointer's seg leg, reproduced, and
**600 of 600 pairs agree exactly** with the receiver's decode argmax.

| | |
|---|---:|
| pairs offering a realized repair | **30 (5.0%)** |
| cells repaired | **39** |
| codes changed | **34** (1.13 per offering pair) |
| pairs needing two codes | 4 |
| single-code moves evaluated | 18,906 |
| … that REDUCE flips | **39 (0.21%)** |
| … neutral | 243 (1.3%) |
| … that WORSEN | 18,624 (98.5%), median **+6**, max **+59** |

Repairs per offering pair: 23 pairs × 1 cell, 5 × 2 cells, 2 × 3 cells.

**The shipped codes sit at a per-pair local minimum of the realized flip count on this lattice, on 95% of pairs.**
Where a descent direction exists at all it is worth one to three cells.

### 9.3 The rate leg, measured by real encode on the 30-pair edit

The whole 30-pair, 34-code edit re-encodes through the shipped RC1 coder to a semantic section of **30,231 B against
the shipped 30,246 B — the edit is 15 bytes SMALLER**, at container shape `(ck2, 10, 16)`. The break fee is not
merely recovered here; the perturbed payload happens to compress better than the shipped one under a re-searched
shape. That is inside the ±25 B draw-to-draw spread §6 measured and it is EXACT for this set, not a draw.

So before pose:

| leg | value |
|---|---:|
| d_seg: 39 cells | **−3.3061e-05** |
| rate: −15 B of semantic section | **−9.9879e-06** |
| **seg + rate** | **−4.3049e-05** |

against an admit bar of −2e-5.


### 9.1 PRE-REGISTRATION — written before the n600 search finished

This section was committed while the search was still in flight (24 of 600 pairs read), so the verdict below is
read against a number written first, not fitted to the answer.

Every coefficient is measured, not guessed:

| term | value | source |
|---|---|---|
| one repaired cell | −8.4771e-07 S | 100 / 117,964,800 |
| one archive byte | +6.6586e-07 S | 25 / 37,545,489 |
| semantic section, container-searched | ≈ 0.15·N + 8 B for N ≥ 25 | `admission/RATE_LAW.json` (+2.8 B at N=25, +8.8 at N=50, +21.0 at N=72, +13.3 at N=100) |
| resolved pose per admitted pair | +2.6e-08 S | pair 118, the uncontaminated row of `admission/PRICE_sizing_winners.json` |
| carrier bytes per re-solved pair | +1.94e-08 S | sj1 pass-3: +13 B over 445 re-solved pairs |

With `N` admitted pairs repairing `C` cells,

```
ΔS = −8.4771e-07·C + (0.15·N + 8)·6.6586e-07 + N·(2.6e-08 + 1.94e-08)
   = −8.4771e-07·C + 1.453e-07·N + 5.33e-06
```

At the measured shape (one cell per admitted pair, `C ≈ N`) this crosses the −2e-5 admit bar at **N > 36.1**, so:

* **PRE-REGISTERED THRESHOLD: the arm needs ≥ 37 admitted pairs.** Fewer than 37 and it cannot clear the bar
  even at zero pose cost.
* **PRE-REGISTERED POINT PREDICTION** (from the in-flight rate of 4 offering pairs in 24, 6 cells):
  N ≈ 100 pairs (binomial 1σ: 45–145), C ≈ 150 cells, **ΔS ≈ −1.05e-04 S**, projected S ≈ **0.138899**.
* **FALSIFIER (charter's, unchanged): if the admitted set's total ΔS ≥ −2e-5 through the REAL archive build,
  the FiLM axis cannot pay on this renderer** (verdict_scope: formulation — single- and paired-code moves at
  |step| ≤ 2 on the shipped 3-bit lattice, this body, realized cpu_torch SegNet acceptance). Count it plainly and stop.

### 9.2 Declared SCOPE reductions in the n600 search (mechanism unchanged)

1. **Steps of at most two code units.** Evidence: all 178 sizing moves with |step| ≥ 3 made their pair worse,
   minimum +4 flips, and both realized repairs came from steps of −1 and +2. Reduces how much of the lattice is
   walked; acceptance is unchanged.
2. **Greedy, not exhaustive, on code pairs.** The second code is searched only over the 7 remaining dimensions
   given the first, not over all (8×4)² combinations.
3. **The greedy descends only through a STRICTLY reducing first move.** A pair whose best single move is exactly
   NEUTRAL is not walked further, so a plateau-then-descend path is not searched. In the sizing that shape appeared
   on 1 of 12 pairs (pair 131, best Δ = 0). This is a real, named residue: a successor that wants the last of this
   axis should allow one neutral step before pruning. It was NOT changed mid-run, because a search whose rule
   changes partway through is no longer one n600 measurement.

MECHANISM reductions: none. Acceptance is the frozen cpu_torch SegNet argmax on the receiver's own render at
`semantic_batch = 1`, on the DALI GT table, for every one of the 600 pairs.

## 9.4 Admission — every cut priced by a REAL archive build

Rebased twice while this ran: move 33 (rc2, hpac) then move 34 (pc2, carrier). Both were verified section by
section with each tree's own reader before anything was carried; pc2's carrier change made the 12 pose rows already
measured against rc2's carrier inadmissible, and they are RETAINED under
`admission/pose_superseded_rc2_carrier/` with a README rather than deleted.

**Pose, all 30 candidate pairs, on pc2's carrier.** The unmoved-render re-solve control moves d_pose by
**exactly 0.00e+00 on all 30 pairs** — the live carrier is converged everywhere, so every resolved value is the
move's own. Stale rise: min 2×, median **1,021×**, max **572,269×**. Base d_pose was re-measured over all 600
pairs on this instrument rather than inherited: **5.090164724404211e-06** on pc2 (and on rc2 the same instrument
reproduced sj1's sealed 5.0928018072772644e-06 to the last digit).

**The sweep.** 47 subsets were built into REAL archives and parsed back — never a ledger sum, because the section's
cost is a container-break fee and is not additive. Prefix cuts by per-pair value, then leave-one-out and
add-one-back around the winner. The archive size is genuinely non-monotonic in the edit set: 15 pairs gives
181,305 B and 27 pairs gives 181,405 B.

**ADMITTED: 15 pairs / 23 cells, archive 181,305 B (-68 vs pc2), container `('ck2', 10, 16)`.**
Pairs: [6, 42, 48, 71, 73, 188, 237, 247, 268, 289, 323, 329, 382, 420, 451].

| leg | value |
|---|---:|
| d_seg (23 cells repaired, 12,866 → 12,843) | **-1.949734e-05** |
| d_pose (re-solved; the subset's pose is BETTER than base) | **-4.375859e-06** |
| rate (-68 B, exact) | **-4.527841e-05** |
| **net ΔS vs pc2** | **-6.915161e-05** |
| projected S | **0.13875411272345906** |

against an admit bar of −2e-5: **the admitted set clears it 3.5×.**

**Read the legs honestly.** The arm set out to buy seg and the seg leg is the SMALLEST of the three: −1.95e-5 from
23 repaired cells, against −4.53e-5 of rate. Most of this candidate's win is the 68 bytes the edited semantic
section happens to compress to under a re-searched container — a consequence of §6's law, not of the FiLM
mechanism. That is a real, exact, shipping number, and it is also the finding that matters most for what comes
next (§11, ITEM 5).

## 9.5 The candidate, measured on its own shipped bytes

| | |
|---|---|
| archive | **181,305 B**, sha `a0c33d7ba30555cdc8ba1649ae2dfe75fe94c9e69ea042f881aed42fd8eb7d0d` (−68 B vs pc2) |
| d_seg, parse-back | **12,843 cells = 0.0001088714599609375**; the admission predicted 12,843 and the shipped bytes measured 12,843 — **zero cells of disagreement** |
| d_seg carried to T4 | 0.00010894369357914811 |
| d_pose, re-solved | **5.083922691393623e-06** |
| **projected S** | **0.13874809002955826** |
| **net ΔS vs pc2** | **−7.517430361217436e-05** (3.8× the −2e-5 bar) |

Legs: seg 0.010894369357914812 · pose 0.007130163175828182 · rate 0.12072355749581527.

Receiver proof, all PASS:

* **public entrypoint smoke, both legs, both roles** — candidate REACHED_TOKEN_DECODE (240.0 s) and
  REACHED_CUDA_GATE (2.35 s); frontier (pc2) REACHED_TOKEN_DECODE and REACHED_CUDA_GATE.
* **parse-back** through the candidate tree's own `inflate.py _verify_input` + `runtime.f26_inflate`, device cpu,
  num_threads 4: `0.raw` 3,662,409,600 B sha `30964f53…`, wall 1,030.5 s.
* **decoded token field byte-identical to the live field**, 0 differing cells of 117,964,800 — this arm changed
  only the semantic model section and the carrier.
* **staged tree differs from pc2's in exactly two files**: `archive.zip` and its own `inflate.py` pins.
* **null build** reproduces pc2 byte-identically.

Seal: `SEAL_ddm_fe1_frame_embedding_predistortion_contest_cuda.json`, sha
`e15da89eba4124352008946f0e0fb60afd2a531f60c044ea186ffa78478af0cd`, **VALIDATED SEAL_VALID**.

### 9.6 PRIOR-LAW PREDICTION vs MEASURED — counted plainly

| pre-registered | measured | verdict |
|---|---|---|
| ≥ 37 admitted pairs needed to clear the bar | **30 offered, 15 admitted** | the pair count MISSED the threshold |
| point prediction ~100 offering pairs | **30 (5.0%)** | **MISS, 3.3× over-predicted** |
| point prediction ~150 cells | **39 found, 23 admitted** | **MISS, 3.8–6.5× over-predicted** |
| point prediction ΔS ≈ −1.05e-04 | **−7.52e-05** | within 1.4× — but for the WRONG reason |
| charter: 25–45% of pairs admit | 5.0% offer, 2.5% admit | **refuted, an order of magnitude** |
| charter: pairs repair 8–15% of their residual | 1–3 cells, ~7% at best, usually 1 | **refuted** |
| charter: seg buys ≥ 4× its rate | seg −1.95e-5 vs rate −4.53e-5 — the rate leg is 2.3× the seg leg and has the **same sign** | the framing itself was wrong |
| charter: pose-bound is the expected failure | pose is a GAIN of −3.42e-6 | **refuted** |

**The honest reading.** The arm cleared the bar and the pair-count threshold it pre-registered was MISSED — it
admitted 15 pairs where it needed 37 to clear on the seg mechanism alone. It cleared anyway because the rate leg
turned out negative, which the pre-registration did not anticipate in either direction. Counting this as a
vindication of per-pair FiLM pre-distortion would be reading the total and ignoring the legs.

## 10. What this arm hands the next one, whichever way the verdict falls

* **The move ledger is the durable asset.** `search/search_rows_*.jsonl` retains, for every one of the 600 pairs,
  the realized flip count of every single-code move at |step| ≤ 2 and the greedy second code where the first
  reduced. Because the section's cost is a container-break fee and NOT a per-code price, those moves are nearly
  free to anyone who is **already** re-encoding the semantic section: the marginal is +0.17 B per changed code.
  **fe1's repairs are free riders on any arm that perturbs the semantic section for its own reasons.** That is the
  composition law (`sy2`: a closed leg survives only if another leg changes its object first) applied here.
* **The per-pair pose law (§0)** re-opens a class the coupling memo closed: localise your render change to one pair
  and its twelve carrier coefficients will pay for it.
* **The container-break law (§6)** is registered as `model_section_edit_container_break_fee_v1` and any arm editing
  a model section on this body must price through it.

## 11. Owed items

## ITEM 1 — price the fe1 move ledger as a rider on the next semantic-section edit
The ledger's repairs cost +0.17 B/code once someone else has paid the container-break fee. When pc2, rc2, or any
successor re-encodes the semantic section, re-price the fe1 admitted set against THAT candidate's bytes rather than
the live row's — the seg gain is unchanged and the rate leg nearly vanishes.

## ITEM 2 — is the break fee the same on the hpac section?
The hpac section is the same container family with the same rider magic (RC1, `reserved` bit 0x40), so the fee is
PREDICTED to apply, and it is NOT measured. One `rate-law` run against hpac codes settles it and would tell rc2
what its own edits actually cost.

## ITEM 3 — the unmoved-render re-solve control (ANSWERED for the pairs measured, still owed at n600 scale)
The control is wired into the admission path and runs for every candidate. On the first two candidates it moves
d_pose by exactly 0.00e+00, so the live carrier is converged there and the moves' pose gains are their own. If the
control turns up material gains on OTHER pairs at full scale, those gains are a rate-free pose lever belonging to a
carrier arm, not to fe1, and must be subtracted from fe1's credit before it seals.

## ITEM 5 — the container-break fee cuts BOTH ways: search seg-NEUTRAL, byte-NEGATIVE edits
This is the strongest thing this arm found and it belongs to a rate arm, not to fe1. The admitted candidate's
biggest leg is −68 B of semantic section, and that is not a property of the 23 repaired cells: it is the edited
range-coded payload happening to compress better under a re-searched container. The n600 search already measured
**243 single-code moves that leave the realized flip count EXACTLY UNCHANGED**. Those moves are free on the seg
axis and free on pose (nothing to re-solve if no frame changes materially), and each of them re-randomises the RC1
payload. A search over seg-neutral code sets, scored purely on the real archive byte count, is a pure rate lever
with no distortion risk at all — and this arm's ledger already contains the moves it would search. Sizing from the
draws in §6: the spread of the archive delta over random edit sets is roughly ±25 B with a best-of observed at
−68 B, so the lever is worth tens of bytes, and tens of bytes is 1–3× the whole admit bar.

## ITEM 4 — a finer FiLM lattice is a different question
Every negative in §4 is scoped to the shipped **3-bit** code domain. The renderer's `frame_embed` could be re-trained
or re-quantised at 4 bits (+600 B of codes at the shipped packing, before the coder), which would offer sub-step
moves the current lattice cannot express. Nothing here says the FiLM axis is dead; it says the shipped lattice is
too coarse to fine-tune on.


---

## 13. CHARTER-READY — the seg-neutral rate lever (ITEM 5, routed to a rate arm)

MAIN routes this to a codex rate arm. Everything below is measured by this arm and on disk; nothing in it needs
fe1 to run again.

### 13.1 PRE-REGISTRATION — written before the first build (ITEM 5 runs as fe1's own measurement)

MAIN routed ITEM 5 back to this arm rather than to a separate one, for a reason this arm's own tooling makes
plain: another arm editing the semantic section would move it, and `ddm_fe1_rebase.py` REFUSES a semantic-section
move. So it runs here, now, as a measurement on the current base (pc2), and the winner is folded into the re-based
candidate later — one seal, one T4 call.

**The candidate set, extracted and retained before any build** (`item5/NEUTRAL_MOVES.json`): of the 243
single-code moves the n600 search measured as changing the realized flip count by exactly 0, **21 sit on pairs the
FiLM candidate already edits** and are excluded so the two sets compose without conflict, leaving **222 free
neutral moves over 133 distinct pairs** (50 pairs offer more than one). Step sizes: −2: 12, −1: 108, +1: 91, +2: 11.

**Objective.** Maximise `−ΔB`, where ΔB is the EXACT archive byte delta of a REAL build on pc2's archive with the
container searched over q ∈ {9,10,11} × lgwin ∈ {16,18,20,22,24} × {ck2, plain}. Score
`ΔS = ΔB · 25/37,545,489 + (pose leg after the per-pair re-solve)`. **No seg term by construction** — that is the
whole point of choosing neutral moves, and it means this search cannot lose on the axis that is hardest to win.

**Search.** At most ONE move per pair, so the moves are independent by construction (`frame_embed` is per-pair, and
two moves on different pairs cannot interact). Stage A: price all 222 single-move builds and rank by measured ΔB.
Stage B: greedy accumulation over that ranking, re-pricing by a real build at every step, accepting a move only if
it lowers the archive. Stage C: a bounded neighbourhood sweep (leave-one-out and add-one-back) around the greedy
winner. Budget a few hundred builds.

**Seg-neutrality is VERIFIED on the composed set, not assumed.** Moves neutral alone need not be neutral together
in general; here the one-move-per-pair rule makes them independent, so the composition should be exactly neutral —
and that prediction is itself checked, by re-rendering every pair in the final set and comparing realized flip
counts against the base. Any pair that is not neutral in composition is dropped and the fact recorded.

**PRIOR-LAW PREDICTION (falsifiable, named).** From this arm's own draws: a single random edit moves the searched
archive delta by −7.2 ± 25.4 B, so the best of 222 measured single moves should already land near
**−70 B** (≈ mean − 2.5σ) — about the whole −68 B the FiLM candidate got from a set never chosen for bytes. Greedy
accumulation over a non-additive objective then buys less than linearly but should still compound.
**I predict the best subset reaches ΔB ≤ −150 B**, i.e. beats −68 B by ≥ 82 B, and that the winning set is
substantially smaller than 222 moves.

**FALSIFIER (MAIN's, adopted verbatim).** If the best subset does not beat −68 B by at least 20 B — that is, if it
does not reach ΔB ≤ −88 B — then the FiLM candidate's draw was already near the achievable floor, the byte spread
of §6 is not exploitable by selection, and **ITEM 5 closes with the table**.

### 13.2 MEASURED — the falsifier FIRED, and it taught the law something

Ran on pc2's archive, 250 REAL builds in total (133 stage-A singles + 117 greedy/neighbourhood),
container searched on every one.

**Stage A — every neutral move priced alone, 133 pairs, one move per pair:**

| | |
|---|---:|
| best single | **−61 B** |
| 5th percentile | −50 B |
| median | −1 B |
| worst | +98 B |
| moves that REDUCE the archive | **67 of 133** |
| **mean · sd** | **+0.1 B · 34.8 B** |

Ten best: −61, −60, −60, −54, −53, −52, −51, −50, −50, −49.

**Stages B and C — greedy accumulation and neighbourhood, 117 real builds: NOTHING beat the single best move.**
Stage B accepted exactly one move (pair 331, −61 B) and every one of the 66 further additions raised the archive.
Stage C's drop-and-add-back round found no improvement either. Final set: **one move, −61 B**, verified
seg-neutral by re-rendering (pair 331, 21 → 21 flips, Δ 0).

| pre-registered | measured | verdict |
|---|---|---|
| best single near −70 B | **−61 B** | close; the prediction was mildly optimistic |
| best subset ≤ −150 B | **−61 B** | **REFUTED** |
| falsifier: fails to reach −88 B → the FiLM draw was near the floor | −61 B | **FIRED** |

**ITEM 5 CLOSES.** And the reason is a refinement to §6's law worth more than the bytes would have been:

> **The container-break delta is a property of the PERTURBED PAYLOAD, not of the number of perturbations.
> It is a ONE-SAMPLE LOTTERY, not a searchable additive budget.**

The measurement is unambiguous: single moves are distributed with **mean +0.1 B and sd 34.8 B** — a zero-mean
lottery, exactly as a re-randomised range-coded payload under a match-finding compressor should be. Adding a
second move to a good draw does not compound the good draw; it RE-SAMPLES, and 117 builds conditioned on the best
draw never beat it. There is no gradient for a greedy search to climb because there is no landscape — every subset
is one more ticket in the same lottery.

That also explains, retroactively, why the FiLM candidate's **−68 B** and the best single neutral move's **−61 B**
are so close despite one carrying 23 repaired cells and the other carrying none: **both are single draws from the
same ±35 B distribution.** The FiLM candidate was not lucky in a way a byte search could have improved on — it was
one ticket, and one ticket is all anyone gets.

**What this means operationally.** Do not build a byte-search arm on this axis; the search space has no structure.
Do take ONE cheap sample: when any arm edits a model section, price a handful of seg-neutral variants of its own
edit and ship the smallest, which costs a few real builds and buys a draw from a −60…+98 B distribution instead of
whatever draw the edit happened to land on. And do NOT add the neutral move to the FiLM set expecting −129 B: the
two are not additive, and at re-base time the honest test is to build FiLM-alone, FiLM-plus-331 and 331-alone and
take the smallest measured archive — three builds, no arithmetic.

### The observation that makes it a lever

fe1's admitted candidate is 181,305 B against pc2's 181,373 — **−68 B** — and its seg leg is worth only −1.95e-05
against a rate leg of −4.53e-05. The 68 bytes are not a property of the 23 repaired cells. They are the edited
range-coded RC1 payload compressing better than the shipped one under a re-searched brotli container. **If bytes
can fall from an edit whose distortion value is small, they can fall from an edit whose distortion value is ZERO.**

### The supply, already measured and retained

The n600 search evaluated 18,906 single-code moves through the frozen cpu_torch SegNet argmax on the receiver's own
render. Of those, **243 moves change the realized flip count by exactly 0** — spread over the pairs, at |step| ≤ 2
on the shipped 3-bit lattice. Every one is in `search/search_rows_*.jsonl` with its pair, dimension, old code, new
code and realized flip count, so the candidate set is a `jq` away and costs no SegNet time to rediscover.

A seg-neutral move is also very close to pose-free: it changes the pair's render, so its carrier still wants a
re-solve, but fe1 measured the re-solve recovering 440×–572,269× stale rises down to a mean pose leg that came out
NEGATIVE on the admitted set. A neutral-move set should be priced with the same per-pair re-solve, not assumed free.

### The objective

Maximise `−ΔB` over subsets of the 243 neutral moves, where `ΔB` is the EXACT archive byte delta from a real build
with the container searched over `q ∈ {9,10,11} × lgwin ∈ {16,18,20,22,24} × {ck2, plain}`. Score is
`ΔS = ΔB · 25/37,545,489 + (pose leg after re-solve)`. There is no seg term by construction — and that is the
point: this arm can only win, never lose, on the axis that is hardest to win on.

### What is already built and reusable

* `experiments/ddm_fe1_pose_price.py::archive_section_bytes` — the container search, anchored: it reproduces the
  shipped section byte-identically.
* `experiments/ddm_fe1_admit_and_build.py::build_candidate_archive` — a real archive build with the null-build
  identity control and a parse-back that refuses unless the bytes decode to the requested codes with a
  byte-identical token tail.
* `experiments/ddm_fe1_admit_and_build.py cmd_pose` — per-pair re-solve with the unmoved-render control.
* `model_section_edit_container_break_fee_v1` — the rate model to price against.

### Sizing, from this arm's own draws (§6)

Random edit sets moved the searched archive delta over roughly ±25 B, and fe1's ONE non-random set landed at
**−68 B**. So the lever is worth tens of bytes, and tens of bytes is 1–3× the whole −2e-5 admit bar. A search that
can evaluate a few hundred subsets — each build is seconds — should beat a single lucky draw.

### PRIOR-LAW PREDICTION for that arm to write down before it runs

fe1 got −68 B from a set it did not choose for bytes. A search that optimises FOR bytes over 243 neutral moves
should beat that. If a few hundred real builds cannot find a subset below −68 B, then fe1's draw was not luck but
close to the achievable floor, and the byte spread §6 measured is not exploitable — record that plainly.

### The honest caveat

This is rate engineering on exact deterministic bytes, not a distortion mechanism, and it does not generalise off
this body: it exists because the semantic section is a range-coded payload under a match-finding compressor
(§6's domain of validity). A body whose model section is not range-coded owes nothing.


## 12. Verdict

**ADMITTED.** `ddm_fe1_frame_embedding_predistortion`, 181,305 B, sha `a0c33d7b…`, projected S
**0.13874809002955826**, net ΔS **−7.517430361217436e-05** against pc2's 0.13882326433317044 — 3.8× the admit bar.
SEAL VALID; MAIN fires.

Three findings outlast the candidate:

1. **The renderer-coupling wall does not transfer to a per-pair change** (§0). 440×–572,269× stale, fully
   recovered, +2.6e-08 S per moved pair. The coupling number was a property of touching every pair at once.
2. **A range-coded model section charges a fixed ~58.9 B container-break fee for ANY edit, and an encoder-side
   container search pays most of it back** (§6), registered as `model_section_edit_container_break_fee_v1`.
   Pricing at the shipped shape would have killed this live axis 3× too expensively.
3. **The FiLM lattice is the wrong tool for this residual** (§4, §9). 18,624 of 18,906 single-code moves make
   d_seg worse; the shipped codes sit at a per-pair local minimum on 95% of pairs. The 3-bit lattice cannot
   express a small enough step. That is a formulation-scope closure, not a family closure — ITEM 4 names the
   re-quantised lattice that would reopen it.

**verdict_scope:** formulation — single- and greedy paired-code moves at |step| ≤ 2 on the shipped 3-bit
`frame_embed` lattice of this body, under realized frozen-cpu_torch-SegNet acceptance. Not a verdict on the FiLM
axis at a finer lattice, nor on any other per-pair carrier.

pc2 S 0.13882326433317044 @ 181,373 B [contest-CUDA T4 n600]
