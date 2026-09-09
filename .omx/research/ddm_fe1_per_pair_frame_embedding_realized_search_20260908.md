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
  argmax. Over the first 97 searched pairs: **97 of 97 agree exactly**, 1,871 flips both ways. The realized
  objective this arm accepts on is the one the shipped bytes produce.
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

## 9. n600 realized search — IN FLIGHT

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

## ITEM 4 — a finer FiLM lattice is a different question
Every negative in §4 is scoped to the shipped **3-bit** code domain. The renderer's `frame_embed` could be re-trained
or re-quantised at 4 bits (+600 B of codes at the shipped packing, before the coder), which would offer sub-step
moves the current lattice cannot express. Nothing here says the FiLM axis is dead; it says the shipped lattice is
too coarse to fine-tune on.

