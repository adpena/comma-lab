# ddm_tw1 — the waterfill's per-unit byte price is a FUNCTION OF STATE, not a constant (task #869)

UTC 2026-08-01 · axis **[macOS-CPU advisory, rate-only]** · actuation **NONE** · `score_claim=false` ·
`promotion_eligible=false` · `pointer_moved=false` · `review_status=pre-registered + own-round-1-reviewed`.
No scorer job, no staged gate, no training, no dispatch, no upstream edit ran.

**Pointer honesty first.** The exact frontier did **NOT** move. This unit produced no byte-closed exact
contest row. Every byte below is MEASURED through the real shipped r7 coder; **no d_seg or d_pose was
computed here** and none is claimed. This is MEANS.

---

## 0. The seed I was handed was materially STALE — four corrections

My dispatch prompt carried four claims as premises. Re-derived at source, **three are wrong** and one is
right-but-incomplete. This is the whole reason the re-derivation clause exists, so I record it plainly.

| # | seed claim | re-derived verdict |
|---|---|---|
| 1 | "QA07's 4-rung ladder has ZERO code in `src/`, `experiments/`, `tools/` — VERIFY, then BUILD it" | **FALSE.** Fully built and already measured: `experiments/ddm_gr1_granularity_rerace.py::coarsen_alloc` (l.84–95) takes `step_map ∈ {1,2,4,0}` = exactly `{L16, L8, L4, drop-to-mode}`, and `build_allocations` (l.206) emits `tok_rung_{a,b,c}` and `cell_rung_{a,b,c}` at both granularities, wired to the real SMEVR encode path. gr1 (2026-07-30) measured it and returned **DOMINATED at both granularities** — carrying gr1's own scope label, that verdict "rests on **n48** with large margins + strict domination" (gr1 l.185); only the cell-drop winner was n600-confirmed. Building it again would have been the named `built_new_machinery_instead_of_paying_identified_debt` poison. **I did not build it.** |
| 2 | "the wr1 staged gate has NOT fired; DO NOT fire it" | **The gate ALREADY FIRED and was REJECTED.** `kneeA_gate_run.log` on SSD holds a real `evaluate.py` n600 run: d_seg 0.00553676, d_pose 0.28002128, 274,333 B, **S = 2.41 vs ref 2.2566 = +0.153 net → REJECT at INSTANCE scope.** The ledger already carries this row. (I did not re-fire it.) |
| 3 | "QA08 has never been run at ANY drop level; wr1 says it can only improve the byte column" | **Half true, and the wr1 claim is now FALSIFIED as stated.** QA08's named coder *is* implemented — it is r7 codec id 8, `kt_o8_prev5_backoff` — and had been raced only at the undropped field. I raced it (and 7 others) at the dropped streams: see §3. It does not improve the byte column at any drop level; it is **9.7% → 21.4% worse** and gets worse as you drop. |
| 4 | "waterfill is over 768 cells (24×32), 1:1 with argmax blocks; anchors k=0→569,996 B, Knee-A k=486→274,333 B, Knee-B k=600→174,578 B, k=768→14,303 B; wr1 never sums, it re-encodes per tranche" | **CONFIRMED**, and independently re-derived: `cell == row*32 + col` holds for all 768 records, 486 zero-flip cells, bands 288 sky / 288 road / 192 hood, and my own harness reproduces wr1's published `tokens_bytes` at k=486 (261,590) and k=600 (161,835) **exactly**. |

---

## 1. The question, and why the existing curves cannot answer it

Operator, 2026-08-01: *"It's also hard to know what the water fill at higher rates might do with svemr
and jrd"* → *"Perhaps we can actually determine that on a token by token basis."*

wr1 and gr1 both published **candidate points** — 10 tranches and 12 coarsenings respectively, each an
honest joint re-encode. Neither measured the **per-unit exchange rate**: the marginal byte price of *one
more* drop, and whether that price is stable as you move along the curve. Every allocator anyone would
build on top of the waterfill — a price table, a knapsack, a Lagrangian sweep, a greedy — assumes it is.

That assumption is testable at `$0` and had never been tested. SMEVR is a **context** coder: it factors
the lattice into a per-cell temporal `base` plus a `(value − base) mod 16` residual and codes the residual
under contexts derived from its neighbours. A drop therefore changes the coding context of cells that were
*not* dropped. If that coupling is material, a one-shot price table is a **linearization of a
non-separable function** and any allocator built on it is mis-specified.

**Harness:** `experiments/ddm_tw1_token_waterfill_state_dependence.py` (deterministic, ruff-clean, stdlib
+ numpy + the shipped r7 coder). Receipts on SSD `/Volumes/VertigoDataTier/pact/ddm_tw1_20260801/`.

### Controls — the meter is proved before any reading is emitted (P4)

`run_controls` aborts the run if any fails; readings from an unvalidated meter are never written.

| control | expected | observed | |
|---|---:|---:|---|
| positive — re-encode the untouched field | 557,253 B | 557,253 B | PASS |
| null — drop the empty set | 557,253 B | 557,253 B | PASS |
| state reconstruction k=486 vs wr1's receipt | 261,590 B | 261,590 B | PASS |
| state reconstruction k=600 vs wr1's receipt | 161,835 B | 161,835 B | PASS |

The state-reconstruction controls are the strong ones: they are exact integers from an *independent*
receipt, and they fail under a wrong index map, outer-product instead of pairwise fancy indexing, wrong
drop semantics, wrong codec, or wrong `levels`.

**One confound closed by measurement, not assumption.** My design claims a drop leaves the `base` stream
invariant, so the whole measured price is a residual/context effect. I tested it rather than asserting it:
at k ∈ {0, 486, 600} the re-factored `base` is `array_equal` to the original and every dropped cell's
residual is exactly zero. **Base invariance: MEASURED TRUE.**

---

## 2. MEASURED — the per-unit price rises monotonically with drop depth

Method: hold a set of cells **out of every state under test**, then price each one's marginal drop from
each state. Same cell, same content, same coder — only the surrounding drop state differs. Comparing
different cells across states would confound the state effect with a cell effect, so the sample is held
out by construction and the harness refuses if a sampled cell is already dropped in any state.

### Run A — 29 cells held out of Knee-B, priced from 5 states (seed 869)

168 cells survive to k=600 — **159 road + 9 hood, and zero sky** (wr1 has dropped all 288 sky by then).
The sampler's `per_band` cap is 20, so run A takes **20 of the 159 available road cells (12.6%) and all
9 hood**, not a census of the survivors. Run A is therefore naturally concentrated on the road midband,
which is exactly the razor zone, but its road coverage is a sample and is labelled as one. (The harness
now prints and records `available_after_exclusion` per band, so this denominator can no longer be
inferred wrongly from the row count — see §8.)

| state | baseline tokens B | mean marginal B/cell | median | min | max |
|---|---:|---:|---:|---:|---:|
| k0 | 557,253 | **771.8** | 807 | 371 | 928 |
| k100 | 469,999 | 845.1 | 870 | 378 | 998 |
| k300 | 333,928 | 849.5 | 880 | 390 | 1014 |
| kneeA (486) | 261,590 | 835.1 | 857 | 395 | 1000 |
| kneeB (600) | 161,835 | **871.2** | 900 | 420 | 1029 |

**Per-cell ratio kneeB/k0: mean 1.1312, median 1.1307, range 1.0641 – 1.2487. The price rose on 29 of 29
cells. Zero exceptions.**

### Run B — all three bands, 24 cells held out of k=300 (seed 8691)

Run A cannot see the sky band. Run B closes that hole with an independent seed and a shallower state
triple, so sky cells are still held out.

The `k300/k0` column is the **mean of the per-cell ratios**, not the ratio of the displayed means (the
two differ slightly; run A's ratios are the same statistic and are labelled explicitly there).

| band | n | k0 | k100 | k300 | mean per-cell k300/k0 | rose |
|---|---:|---:|---:|---:|---:|---:|
| `mycar_hood_bottom` | 8 | 252 | 270 | 284 | 1.108 | 7/8 |
| `road_lane_midband` | 8 | 791 | 874 | 882 | 1.119 | 8/8 |
| `sky_undriv_top` | 8 | 394 | 433 | 456 | **1.165** | 8/8 |
| **all** | 24 | | | | **1.1305** | **23/24** |

**Across both runs and both seeds: 52 of 53 cells rose in price. The single exception is one hood cell at
0.987.** The effect is largest in sky (+16.5%), which is precisely the band Knee-A spends most of its
budget on.

### The law

> **TW1-1 (MEASURED).** On the pfs1 D1 `[600,24,32,4]` L16 lattice under the shipped SMEVR coder, the
> marginal byte saving of dropping a fixed cell is **not a constant**: it *increases* with the number of
> cells already dropped, by **+13.1% (mean) from k=0 to the Knee-B state**, on 52/53 cells across three
> bands and two seeds, with the `base` stream measured invariant.
> Scope: **INSTANCE/FORMULATION** — this vehicle, this alphabet, this coder. Not a family claim.

**This reverses the confound.** wr1's drop order is flip-risk ascending, tie-broken by residual-mass
**descending** — it drops the *fattest safe* cells first. That ordering predicts per-cell savings should
**fall** along the descent. On fixed cells we measure the opposite, so the coder-context effect is strong
enough to reverse the ordering effect. This is the same mechanism gr1 named from the other side ("SMEVR
conditions on the per-cell temporal mode, so scattered token drops fight the coder"), now measured as a
signed, sized per-unit quantity.

---

## 3. MEASURED — the additivity defect is SUPERadditive, and it shrinks with depth

**Pre-registered, with a disclosure.** `tw1_prereg_additivity_sign.md` was written to SSD at
2026-08-02T00:55:17Z, **before any additivity row for the 29-cell sample existed** (that receipt was
written 2 m 18 s later, at 00:57:35Z — both timestamps independently checkable on disk). It predicted
`defect < 0` (superadditive) at every state, **and** that `|defect|` would grow with depth, with an
explicit kill criterion.

**Disclosure (found by my own fresh-context verifier, not by me).** The 2-cell controls-first smoke
receipt the prereg cites by path was written 3 minutes *earlier* and already contained additivity rows:
k0 `0`, kneeA `0`, kneeB `+1`. The prereg quoted that receipt's *marginal* numbers and did not mention
its additivity rows. At n=2 those values sit on the byte-quantization floor and carry essentially no
information — but `+1` is nominally the sign my own kill criterion names, so reporting only half of a
cited receipt is a real pre-registration defect. The prediction is therefore **pre-registered with
respect to the 29-cell result and NOT blind with respect to a 2-cell prior**. I record it rather than
let the stronger phrasing stand.

`defect = Σ(singleton marginals from state S) − (joint measured saving of the same set from S)`.

| state | Σ singletons | joint measured | defect | |
|---|---:|---:|---:|---:|
| k0 | 22,382 | 24,064 | **−1,682** | −7.0% |
| k100 | 24,509 | 24,971 | −462 | −1.9% |
| k300 | 24,636 | 25,087 | −451 | −1.8% |
| kneeA | 24,217 | 25,020 | −803 | −3.2% |
| kneeB | 25,264 | 25,495 | −231 | −0.9% |

Run B independently: k0 −488 (−4.1%), k100 −108 (−0.8%), k300 −41 (−0.3%).

- **Primary prediction CONFIRMED**: the defect is negative at every state in both runs. A set of cells
  jointly saves **more** than the sum of its individually-measured marginals.
- **Secondary prediction FALSIFIED**: `|defect|` does **not** grow with depth. It is *largest at k=0*
  (−7.0%) and decays to −0.9% at Knee-B. I report this against myself. The consistent reading is that the
  context coupling available to be exploited is largely consumed by the first tranches.

> **TW1-2 (MEASURED).** Cell drops on this lattice are **superadditive** under SMEVR: joint saving exceeds
> the singleton sum by up to **7.0%**, greatest at the high-rate (undropped) end and decaying toward the
> knee. Scope: **INSTANCE/FORMULATION.**

**Consequence, and it is the operator's answer.** Both measured effects point the same way: a one-shot
per-cell price table computed at high rate **understates** what is actually available deeper in the
descent — once because each unit gets cheaper-to-remove as neighbours go (TW1-1), and again because sets
beat their own singletons (TW1-2). A linearized allocator on this coder is **conservative: it will stop
dropping too early.** It is not merely noisy — it is biased, with a known sign.

**wr1's own published curve is NOT affected**: it re-encodes jointly per tranche and never sums. The
finding governs any *future* allocator, which is exactly what the operator was asking about.

---

## 4. MEASURED — the coder race re-run at each drop state (the SMEVR / QA08 question)

The lossless race was decided at the **undropped** field. Dropping 486–600 cells removes most of the
residual mass and changes the source statistics the race was decided under, so I re-ran it where the
waterfill actually leaves the stream. All 9 r7 codecs, 3 states, real encoder, real bytes.

| codec | k0 | Knee-A (486) | Knee-B (600) |
|---|---:|---:|---:|
| **smevr** | **557,253** | **261,590** | **161,835** |
| brotli11 | 642,703 (+15.3%) | 287,567 (+9.9%) | 186,561 (+15.3%) |
| lzma1 | 656,908 (+17.9%) | 291,543 (+11.5%) | 187,172 (+15.7%) |
| kt_prev1 | 607,352 (+9.0%) | 313,923 (+20.0%) | 197,506 (+22.0%) |
| **kt_o8_prev5_backoff** (= QA08) | 611,496 (+9.7%) | 303,422 (+16.0%) | 196,459 (+21.4%) |
| cae_inspired_identity_inter | 631,283 (+13.3%) | 320,680 (+22.6%) | 217,335 (+34.3%) |
| rans_o0 | 662,614 (+18.9%) | 379,198 (+45.0%) | 252,968 (+56.3%) |
| rans_o0_on_adjacent_innovation | 769,644 (+38.1%) | 415,402 (+58.8%) | 280,049 (+73.1%) |
| huffman_nibble | 675,029 (+21.1%) | 450,853 (+72.4%) | 367,516 (+127.1%) |

Three readings:

1. **SMEVR wins at every drop state, and its lead WIDENS**: +8.99% over the best alternative at k0,
   +9.93% at Knee-A, **+15.28% at Knee-B**. The waterfill does not erode the coder choice; it reinforces
   it. This is the direct answer to *"what the waterfill at higher rates might do with SMEVR."*
2. **The runner-up REORDERS**: `kt_prev1` is second at k0, `brotli11` is second at both knees. So the race
   is genuinely state-dependent *below* the winner — only the winner happens to be invariant. Any future
   coder race must be re-run at the intended operating point, not inherited from k0.
3. **QA08 is measured DEAD on the dropped streams, and worsening**: +9.7% → +16.0% → +21.4%.

> **TW1-3 (MEASURED) — a wr1 claim falsified on its own terms.** wr1 §6 frames QA08 as *"a LOSSLESS coder
> swap"* that *"lowers bytes for the SAME surviving token values at EVERY drop level (shifts the whole
> curve down) at zero seg cost,"* and calls for *"a QA08 re-race on the Knee-A/B dropped streams"* that
> *"can only improve the byte column."* That is exactly the experiment run here, and **it does not.** No
> available r7 codec improves on SMEVR at any drop level; the best alternative is 9–15% worse, and QA08's
> own coder is 10–21% worse and degrades with depth.
>
> I originally softened this by saying wr1 "meant" the ideal ≤1,617 B mixing bound. That was my
> misattribution, caught by my own verifier: wr1 §6 never mentions 1,617 B, xi1, or ba29 (zero grep hits
> in that file), and it names a coder swap and a re-race, not a bound. **wr1 §6 was wrong on its own
> terms** and I should not have handed it an escape hatch it did not write.
>
> Scope: **INSTANCE/FORMULATION** — this codec menu, these states. An unbuilt mixer could still close some
> of the ideal gap; nothing here bounds that. But note that the ≈0.3% figure is itself a **k0-measured**
> quantity (1,617 B against SMEVR's *undropped* 555,836 B delta stream), and by TW1-1 and by bullet 2
> above, k0-measured quantities do not transfer to the dropped states. **The ideal headroom at Knee-A/B is
> UNMEASURED**, not ≈0.3%.

---

## 5. BOTH S COLUMNS — and the harder finding, that the ceiling column is not a ceiling

Task constraint (a) required my curve to carry `S_ref_flipfree` **and** `S_ref_ceiling`, because wr1
published its descent without its own ceiling column. Reconstructed from wr1's receipt (`d_pose` held at
the reference 0.22144216, which is what both wr1 columns assume). Rows k=200 and k=400 are omitted here
for width only — they carry `dropped_flip_mass = 0` and identical columns, and the "identical for all
k ≤ 486" statement below was checked over **all** 11 published rows, not this subset:

| k | archive B | rate | d_seg flipfree | d_seg ceiling | **S_flipfree** | **S_ceiling** | dropped flip mass |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 569,996 | 0.37954 | 0.00389 | 0.00389 | 2.2566 | 2.2566 | — |
| 100 | 482,742 | 0.32144 | 0.00389 | 0.00389 | 2.1985 | 2.1985 | 0 |
| 300 | 346,671 | 0.23083 | 0.00389 | 0.00389 | 2.1079 | 2.1079 | 0 |
| **486** | **274,333** | 0.18267 | 0.00389 | 0.00389 | **2.0598** | **2.0598** | **0** |
| 540 | 227,327 | 0.15137 | 0.00389 | 0.00391 | 2.0285 | 2.0304 | 2,313 |
| **600** | **174,578** | 0.11624 | 0.00389 | 0.00401 | **1.9933** | **2.0058** | 14,659 |
| 660 | 118,245 | 0.07873 | 0.00389 | 0.00444 | 1.9558 | 2.0108 | 64,823 |
| 730 | 51,128 | 0.03404 | 0.00389 | 0.00596 | 1.9111 | 2.1186 | 244,705 |
| 768 | 14,303 | 0.00952 | 0.00389 | 0.00778 | 1.8866 | 2.2755 | 458,738 |

Carrying the column exposes something worse than its omission, and it is the most important line in this
memo:

**For every k ≤ 486 the two columns are IDENTICAL, and both are wrong.** wr1's ceiling is built by adding
the *current* flip mass of the dropped cells. Knee-A drops only zero-flip cells, so the ceiling term is
zero **by construction** — at the operating point wr1 actually recommended, the "ceiling" was vacuous, not
conservative. The MEASURED realized gate at exactly that point:

| | d_seg | d_pose | archive B | S |
|---|---:|---:|---:|---:|
| wr1 predicted (both columns) | 0.00389011 | 0.22144216 | 274,333 | **2.0598** |
| **MEASURED** (`evaluate.py`, n600) | **0.00553676** | **0.28002128** | 274,333 | **2.4097** |
| miss | +42.3% | +26.5% | exact | **+0.3499** |

Decomposed against the reference row: seg **+0.1647**, pose **+0.1853**, rate **−0.1969** ⇒ net
**+0.1531**, break-even REJECT. The rate lever landed *exactly* as predicted; both distortion terms did
not.

> **TW1-4 (MEASURED, n=1).** wr1's `S_ceiling` is **not a bound**. It can only price flips that already
> exist inside the dropped cells; it is structurally blind to flips *created* by dropping. Zero *current*
> flips and zero *created* flips are different predicates, and only the second is the one the score
> charges. This is the same class gr1 measured independently at token granularity — "the exact-zero-
> GRADIENT tokens are NOT flip-free under a finite drop".
>
> **Sizing, decomposed — the pierce is not all one mechanism.** The total miss is +0.3499 S, but only
> **+0.1647 (47%) is the d_seg term** that flip-blindness explains; the larger **+0.1853 (53%) is d_pose**,
> which the ceiling never modelled at all and which §6 attributes separately. Quoting +0.3499 against the
> flip-blindness mechanism over-attributes it by ~2.1×. The honest statement is: *the ceiling is not a
> bound on d_seg (missed by +42.3%), and it is not a bound on S at all because it prices no pose term.*

So constraint (a) is satisfied, but the honest deliverable is stronger than the constraint asked for: **a
finer-granularity curve must not carry these two columns as its error bars.** It needs a third,
created-flip column, and that column cannot be derived from the flip atlas — it has to be rendered.

## 6. The razor (constraint b) — what I can and cannot say

Constraint (b) required the per-cell curve to be pose-leak-constrained on the road midband. I can now put
numbers on both halves of the razor, and must be explicit that only one half is measured here.

- **Bytes (MEASURED).** Road-midband cells are the *most expensive* cells to keep and the most valuable to
  drop. Within run A (k0→Knee-B, matched states): road 828→928 B/cell versus hood 647→746. Within run B
  (k0→k300, matched states, the only run that can see sky): road 791→882 versus hood 252→284 and sky
  394→456. Road leads every band at every state measured; the two runs are not directly comparable to each
  other because their state ranges differ.
- **The ~100 road drops, priced state-matched.** wr1's k=486→600 tranche is **114 cells = 101 road + 13
  hood**, and it saved 99,755 B. Predicting that from my per-unit marginals requires choosing *which
  state's* marginals to use, and the choice decides the sign: at Knee-A marginals
  `101×891.75 + 13×709.11 = 99,285` ⇒ defect **−470 B (−0.47%, superadditive, consistent with TW1-2)`;
  at Knee-B marginals `101×927.5 + 13×746.0 = 103,376` ⇒ **+3,620 B (+3.63%, subadditive)**. An earlier
  draft of this memo quoted "≈90 KB" by counting road only and then credited the entire ~9.7 KB gap to
  superadditivity; that was wrong twice over — most of the gap is simply the 13 uncounted hood cells, and
  the residual's sign depends on a state the sentence never named. **Corrected reading: the tranche is
  worth ~100 KB, and the superadditive residual over it is small (≲0.5%) when priced at the tranche's own
  base state.**
- **Distortion (NOT MEASURED HERE, and the razor got sharper).** The Knee-A gate is the calibration:
  it froze **288 sky + 170 hood + 28 road** cells (486 total) and moved d_pose from 0.22144 to 0.28002,
  **+26.5%**. wr1 typed the sky and hood drops as pose-free and the 28 road cells as "stable interior".
  **The gate cannot separate these**: with 28 road-plane cells in the same drop set, the +26.5% is not
  cleanly attributable to the far field, and an earlier draft of this memo asserted a far-field sign
  reversal that the experiment does not support. What is measured is only that **the composite typed
  prediction was wrong in direction** — Knee-A was predicted pose-favorable and came back +26.5%. Which
  band carries it is **UNRESOLVED** and needs a band-split gate (drop sky+hood only, then sky+hood+road)
  to separate. At the solved-pose operating point the seed's razor arithmetic
  (`d(term)/d(d_pose) ≈ 327`, so a 2e-5 leak ≈ +0.0153 S ≈ 57 KB of rate) therefore stands, **but wr1's
  per-band pose typing cannot be used to steer around it** — it has never been checked band-by-band.

**I did not fire the gate and did not run any scorer job.** Every distortion number above is quoted from
runs that already exist.

---

## 7. What remains unmeasured, and the exact measurement that settles it

| open question | the exact measurement | cost |
|---|---|---|
| Does the rising marginal price change the *chosen* knee? | Re-run the descent with a **greedy-under-joint-remeasure** selector (re-price surviving cells every N drops) instead of a fixed pre-ordering, and compare the byte curve to wr1's. My harness already exposes `state_bytes` over arbitrary state sets, so this is a driver, not new machinery. | `$0`, ~1 encode/candidate at 1–5 s; a 768-cell greedy at 16-cell rounds ≈ 48 rounds × ~280 encodes ≈ 4 h CPU |
| What is the *created-flip* cost per cell — the column wr1's ceiling cannot supply? | gr1's validated fast path: inject coarsened codes → `render_frame` → torch-R uint8 → frozen CPU SegNet argmax vs GT `lstars`. It is calibrated against the real evaluator to **Δ ≤ 2.8e-5** and ran n600 in 162.9 s. **This is a scorer job — out of scope for me by instruction.** | one scorer slot, ~3 min/candidate at n600 |
| Is Knee-B's realized d_pose survivable after a pose re-solve on the dropped base? | The composed gate: Knee-B base + P3v2/ck1 re-solved pose (ck1 already shows recovery parity 0.98×), then `stage_wr1_realized_gate.sh kneeB`. The Knee-A REJECT is INSTANCE-scoped on **stale pose params**, so this is not a family kill. | ~1 h re-solve + ~17 min gate |
| Can the ≈0.3% ideal mixing gap be closed by a *built* mixer? | Not by anything in the r7 menu (§4). Would need a new coder raced against SMEVR at Knee-A/B, not at k0. | unbuilt |

**Owed, and not done here:** this memo's TW1-1/2/3 are measured laws and by the triality contract belong
in `tac.canonical_equations`. My dispatch instruction specified `[no-triality]` commits, so I did not
register them; I record the debt rather than let it pass silently.

---

## 8. Own round-1 adversarial review (contract #337)

Run against my own output before handing over. Findings I acted on are marked.

1. **Traced every assumed key/unit.** `drop_rank` semantics, `cell = row*32+col`, `levels=16`, the
   `marginal_bytes` state keys, and the archive-floor constant were each re-derived from a primary
   artifact, not inherited. ✔
2. **Would my tests pass if the code were broken?** The two state-reconstruction controls compare against
   exact integers from an independent receipt and fail under wrong index mapping, outer-product instead of
   pairwise fancy indexing, wrong drop semantics, wrong codec, or wrong `levels`. This is the control that
   makes the readings load-bearing. ✔
3. **Confound: is the price change really the residual?** Not assumed — **measured**: `base` is
   `array_equal` before/after at k ∈ {0,486,600}. ✔
4. **Confound: is the effect an artifact of *which* cells wr1 held out?** Run A's held-out set is by
   construction the highest-flip-risk cells. **Acted on**: added run B (different seed, different state
   triple, all three bands, 468-cell held-out pool) — same +13%. ✔
5. **Sample size honesty.** n=29 and n=24, single-seed each (two seeds total). The direction is 52/53 with
   one exception at 0.987, which I report rather than round away. Across-seed variance beyond these two is
   **UNKNOWN**.
6. **Did I overclaim?** The finding is a **byte** law. No S movement is claimed; no d_seg/d_pose was
   computed by this module. The §5/§6 distortion numbers are quoted from pre-existing runs and labelled.
7. **Class vs instance.** I built a meter, not a fix. It generalizes: `state_bytes` / `marginal_price_by_state`
   / `coder_race_by_state` accept arbitrary state sets and codecs, so the same meter prices any future
   allocator's states.
8. **Against myself.** My own secondary pre-registered prediction (|defect| grows with depth) was
   falsified by my own data and is reported as falsified in §3.

**Unresolved after round 1:** the greedy-remeasure driver (row 1 of §7) is the direct consequence of
TW1-1 and I did not run it — it is ~4 h CPU and would have exceeded this unit's scope. It is named with
its exact form rather than left implicit.

## 8b. Round 2 — fresh-context verifier (the round that actually found things)

Self-critique caught none of the following. A fresh-context verifier re-derived every load-bearing number
from the primary artifacts and returned **13 defects**. Every *number* in the memo passed — all 20 cells
of the run-A table, all 8 additivity rows, all 27 race rows, all 11 S-column rows, all 6 gate figures were
recomputed exact. **Every defect was an attribution or scope error, which is precisely the class
self-review is worst at.** Fixed in this revision:

| # | defect | fix |
|---|---|---|
| D1 | §3 claimed the prereg predated "any additivity row"; a 2-cell smoke receipt 3 min earlier already held rows (0/0/**+1**) and the prereg quoted only its marginals | scope restated + the 2-cell rows disclosed, incl. that `+1` is nominally my own kill-criterion sign |
| D2 | "only road (20) and hood (9) survive to k=600" — 159 road survive; 20 was the sampler cap | corrected to 20-of-159 (12.6%); harness now records the denominator |
| D3 | §6 omitted the **28 road cells** Knee-A also freezes, then blamed the far field | composition corrected to 288 sky + 170 hood + 28 road; far-field attribution **withdrawn as UNRESOLVED**, band-split gate named |
| D4 | TW1-4 sized flip-blindness at the full +0.3499 S | decomposed: only +0.1647 is d_seg; +0.1853 is pose. Over-attribution was ~2.1× |
| D5, D6 | TW1-3 handed wr1 an "ideal bound" escape hatch wr1 never wrote, while §4 bullet 2 forbids inheriting k0 quantities | escape hatch withdrawn; wr1 §6 quoted verbatim and falsified **on its own terms**; ideal headroom at the knees relabelled **UNMEASURED** |
| D7 | §6's "≈90 KB" counted road only and credited a 9.7 KB gap to superadditivity; the residual's sign flips with the state chosen | recomputed state-matched both ways (−0.47% at Knee-A, +3.63% at Knee-B) and the ambiguity stated |
| D8 | harness: a missing wr1 receipt row silently `continue`d, shrinking the control set with `all(passed)` still True — the repo's **vacuity-reads-as-pass** class | unrunnable control now emits `passed=False`; canary added and run (verified it returns `all_passed=False`) |
| D9 | receipts carried no seed / `per_band` / argv / git head | `invocation` block added |
| D10 | hold-out used `max(states, key=len)`, correct only because these states nest | now the explicit union |
| D11 | gr1's token-DOMINATED reported without its n48 scope label | label restored |
| D12, D13 | run-B ratio statistic unlabelled; §5 silently showed 9 of 11 rows | both labelled |

**The fixes are unreviewed new code and reset the clean-pass counter to 0.** What they were regression-
checked against: the repaired harness reproduces the 2-cell smoke byte-for-byte (additivity 1,639/1,639,
1,795/1,795, 1,880/1,879 — identical to the pre-fix run), all four controls still PASS, and the new
fail-closed path was positive-controlled with a stripped receipt. **No measured byte in §2–§4 changed**;
the D1–D7 fixes are corrections to claims *about* those bytes, not to the bytes.

---

## Receipts (SSD `/Volumes/VertigoDataTier/pact/ddm_tw1_20260801/`)

- `marginal/tw1_state_dependence_receipt.json` — run A: 29 cells × 5 states, controls, additivity.
- `allbands/tw1_state_dependence_receipt.json` — run B: 24 cells × 3 states, all three bands.
- `race/tw1_state_dependence_receipt.json` — 9 codecs × 3 drop states.
- `smoke/tw1_state_dependence_receipt.json` — 2-cell controls-first smoke.
- `tw1_prereg_additivity_sign.md` — the pre-registration, written before the result.
- Regenerator: `experiments/ddm_tw1_token_waterfill_state_dependence.py` (deterministic, ruff-clean).

Schema `ddm_tw1_token_waterfill_state_dependence.v1`. Source archive
`ddm_pfs1_20260729/d1/eval_root/submissions/pfs1/archive.zip`, tokens sha `85e2b15d28bddd1b…`.
