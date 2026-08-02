---
title: "ddm_dc1 — the LABEL half of the correction price, measured; QA03's censored solver; and the §B.5 sign"
date: 2026-08-01
agent: ddm_dc1
task_row: 832
evidence_axis: "[macOS-CPU advisory] NON-PROMOTABLE"
score_claim: false
promotion_eligible: false
pointer: "0.1910828242 [contest-CPU] UNMOVED — this arm is scorer-free and moves nothing"
review_status: "pre-registered-controls + own-round-1 + fresh-eyes-verifier(1)"
---

# ddm_dc1 — the correction-stream LABEL cost, measured; the QA03 solver, named; the §B.5 sign, adjudicated

**Pointer honesty FIRST: 0.1910828242 [contest-CPU] UNMOVED. Our own-vehicle frontier v4d
0.9639878 UNMOVED. Every number below is `[macOS-CPU advisory]`, `score_claim=false`. This arm
fired no scorer job and shipped no bytes.**

STORES CONSULTED: `tools/corpus_query.py "aimed correction band law position label cost QA03"` ·
`.omx/research/ddm_ba31_negative_surfaces_20260731.md` §B.1–B.3, §10.B ·
`.omx/research/ddm_sb1_seg_batch_20260729.md` (the QA03 producer memo) ·
`/Volumes/VertigoDataTier/pact/ddm_sb1_20260729/qa03/{qa03_receipt.json,qa03_instances.jsonl}` ·
`.omx/research/ddm_pp1_band_lemma_receipt_20260728.json` ·
`src/tac/canonical_equations/ddm_pp1_correction_stream_position_band_20260728.py` ·
`experiments/ddm_pp1_band_lemma_curve.py` · `tools/measure_contour_string_flip_coding.py` ·
`tools/sb1_seg_batch.py` · `.omx/research/ddm_gr1_granularity_rerace_20260730.md` ·
`.omx/research/ddm_of1_offset_field_and_flicker_coherence_20260729.md`.
DELIBERATELY NOT loaded: the r7/SMEVR coder arm and the xi1 byte-pricing handoff (named as the
owner of QA03's true shipping price; out of this arm's scope), and every training/burn artifact.

---

## §0 Answer first

1. **The seed's framing of Job 1 does not survive contact with the primary artifact.** QA03 has
   **no position stream and no label stream to split.** Its 2,709 B is the *tr1 archive re-encode
   delta* after in-place token-value edits — entropy inflation of an existing stream. The receipt
   says so itself. The 1.4518 B/flip therefore cannot be compared to the water level or to the
   pp1 band law at all: it is a different object in a different currency.
2. **The measurement that *was* decidable, I ran.** The label half of a real correction stream is
   now MEASURED on pp1's own supports with pp1's own coders. **Label = 0.082–0.255 B/flip**, which
   is **0.28–0.88×** the blind 5-ary bound (best of two coders). **ba31's hypothesis is REFUTED as
   stated**: the neighbour-conditioned coder alone buys only **1.14×** at the band edge, and a
   GENERIC control coder beats it at 6 of 7 densities.
3. **But the strategy's sign survives paying it.** At our live base the full position+label price
   is **0.883–0.915 B/flip = 0.69–0.72× water**, and the idealised full-residual correction still
   nets **−0.132 S (ja1) / −0.111 S (burn)** — **16.7% / 14.0% of the 0.7918468 gap-to-bar**.
   ba31's −0.204 was over-stated by 55% because it priced only half the cost. The direction is not
   the error; the magnitude was.
4. **The band's lower edge moves 2.56×** — from the registered position-only `rho_c = 5.02e-4` to
   **1.285e-3** with label paid. Our base (3.9–4.3e-3) is inside the band on either reading.
5. **The named solver deficiency is a hard cap that fired before the convergence test on 42.5% of
   instances**, and those censored instances produced **64.7% of all realized flips**. Fixed,
   wired, tested. Re-measuring requires a scorer slot, which I did not take.
6. **§B.5 does carry a sign defect — but not the one wr2 flagged.** gr1's "DOMINATED" is *correct*.
   The defect is that ba31 places bytes-saved-per-flip-**added** and bytes-spent-per-flip-**fixed**
   in one "vs water" column and concludes they are "opposite ends of the same real line." They are
   opposite-signed economies; "below water" means WIN for one and LOSE for the other.

---

## §1 What I re-derived, and where the seed was wrong

| seed claim | verdict | source |
|---|---|---|
| QA03 total = **1.4500** B/flip | **WRONG (small)** — it is **1.45177** | `qa03_receipt.json`: 2709/1866. `ba29` had 1.4518 right; `ba31` rounded to 1.4500 |
| water = 1.2731, registered | **VERIFIED** | `ddm_pp1_..._20260728.py:49` |
| law's uniform position bound at burn rho = **1.0007** | **VERIFIED** | log2(1/3.89011e-3)/8 = 1.00073 |
| law's coherent position at burn rho = **0.7024** | **NEAR** — I measure **0.6981** (0.6% apart; interpolation-method difference, not a defect) | dc1 n600 vs ba31 §B.2 |
| ba31 §B.3 coherent position at ja1 rho = **0.6702** | **VERIFIED to 4 dp** by independent measurement | dc1 measures 0.6702 |
| the law "explicitly excludes the LABEL cost" | **VERIFIED verbatim** | `domain_of_validity["excluded"]` |
| "≥51.6% of QA03's total is **label + solver overhead**" | **REFUTED AS ATTRIBUTED** — the residue is real, but it is not label cost; QA03 has no label stream | see §2 |
| swing 0.264 S "larger than the entire rate axis" | arithmetic VERIFIED, but the swing is between two prices of **different objects** | see §2, §4 |

**Unchecked premise I could not close:** the seed (via ba31 §B.2) assumes QA03's flip support is
"comparably coherent to the receipt's margin-thresholded synthetic support." QA03 stored only
`(pair, cell, net_flips)` — no per-pixel support — so its coherence is **UNKNOWN** and cannot be
recovered without re-running the scorer. Every price I quote for a correction stream is therefore
quoted on pp1's supports, not QA03's, and I say so at each use.

---

## §2 Job 1 — the split, and why the object had to change

### 2.1 QA03 has nothing to split (the finding that reframes the pool)

`tools/sb1_seg_batch.py:242` commits `rtp.codes[p, gy, gx, ch] += sign` — an **in-place token
value edit**. `:257` then calls `rtp.price_bytes()`, one full re-encode. The receipt's own
`byte_delta_note` reads verbatim:

> *"tr1.ddt1 re-encode; token edits byte-neutral at first order (ru1); **true shipping price =
> r7 SMEVR coder (xi1 handoff)**"*

So the receipt already declares that 1.4518 is **not the shipping price**. Three distinct
currencies were being compared as one:

| quantity | what it measures | direction |
|---|---|---|
| water 1.2731 B/flip | region-merge **concede** price | the alternative to correcting |
| pp1 band law | **position coding** of an explicit correction stream | bytes spent per flip fixed |
| QA03 1.4518 B/flip | **entropy inflation** of an existing token stream under edits | neither — a side effect |

The pp1 law's `domain_of_validity["object"]` is *"correction/support stream position coding over
the n600 GT scorer geometry."* An in-place token edit re-encoded through tr1 is **not in that
object**. The comparison "QA03 1.45 ≈ water 1.27, break-even" is therefore out of domain on its
face — before the label exclusion is even reached. I have added this to the new law's `excluded`
list so the category error is refused structurally rather than re-argued.

### 2.2 The measurement I ran instead

`experiments/ddm_pp1_band_lemma_curve.py:79` builds the class maps as
`cmaps = [np.zeros((H, W), np.int64) for _ in range(N)]  # positions only; class stream trivial`
and `position_price()` sums only `counts + anchor + chain`, dropping the `cls` stream. **The #307
coder has always had a label stream; pp1 zeroed it by construction.** The missing half was one
call away.

`experiments/ddm_dc1_label_stream_price.py` (new) runs the **same supports** (`margins < tau`),
the **same coder**, the **same n600 GT**, with the **real** class map, and splits
`stream_bytes` into POSITION (`counts+anchor+chain`) and LABEL (`cls`). Second label coder: labels
at support in raster order → LZMA1-x9e, as the **generic control**.

**Controls (P4 — no meter without a canary), both PASS at all 7 densities:**
* **POSITIVE** — position streams are byte-identical under zeroed / real / random class maps, and
  reproduce pp1's registered `b_per_err_best` **to 4 dp** at τ ∈ {0.008, 0.02, 0.05, 0.1, 0.2}.
  The position-only water crossing recomputes to **5.0146e-4** against the registered **5.02e-4**
  (0.11% apart). The instrument *is* the pp1 instrument.
* **NEGATIVE** — i.i.d. uniform random labels on the same support cost ≥ 0.90× the blind bound, so
  the label coder is not smuggling position information into the class stream.
* **Bit-exact decode of flips AND labels** verified on a leading 8-frame subset.

### 2.3 MEASURED (n600, receipt `.omx/research/ddm_dc1_label_price_n600_20260801.json`)

| ρ | POSITION | LABEL (#307 ctx) | LABEL (LZMA ctrl) | TOTAL (verified pair) | TOTAL / water | LABEL / blind |
|---:|---:|---:|---:|---:|---:|---:|
| 2.2475e-4 | 1.4687 | 0.2551 | 0.2678 | 1.7238 | 1.354 | 0.879 |
| 5.6320e-4 | 1.2448 | 0.2502 | 0.2483 | 1.4950 | 1.174 | 0.862 |
| 1.4131e-3 | 1.0054 | 0.2422 | **0.2295** | 1.2476 | 0.980 | 0.834 |
| 2.8235e-3 | 0.8026 | 0.2323 | **0.2042** | 1.0349 | 0.813 | 0.800 |
| 5.6219e-3 | 0.5872 | 0.2010 | **0.1657** | 0.7882 | 0.619 | 0.692 |
| 1.1124e-2 | 0.3796¹ | 0.1508 | **0.1236** | 0.5304 | 0.417 | 0.519 |
| 2.1705e-2 | 0.2484¹ | 0.0982 | **0.0817** | 0.3300 | 0.259 | 0.338 |

TOTAL = position(#307) + label(#307), the **fully round-trip-verified** pairing — the conservative
reading. Blind 5-ary bound = log2(5)/8 = **0.2902410**.
¹ at τ ∈ {0.4, 0.8} pp1's best position coder is LZMA (0.3593 / 0.1999), not the contour walk;
dc1 reports contour only, so **both** rows over-price position (by 0.0203 / 0.0485 B/err). Immaterial: the band edge
and our operating point both sit in the region where the contour coder wins and dc1 reproduces pp1
exactly.

### 2.4 The hypothesis, tested

> ba31: *"the label stream may be largely predictable from neighbouring labels — the same boundary
> coherence the position coder already exploits for 1.28× — in which case the composite falls
> toward the position price. **It may also not.**"*

**It does not. REFUTED as stated.** The neighbour-conditioned `cls` stream buys only **1.14×** at
the band edge and **1.25×** at our base against the blind bound — and the **GENERIC** LZMA raster
control **BEATS** the neighbour-conditioned coder at **6 of the 7** measured densities — every
ρ ≥ 5.6e-4, the coherent coder winning only at the sparsest (2.2e-4). The #307 docstring's own
claim (*"class coherence within a component → near-free after adaptation"*) is not borne out at
correction-stream densities. This is the ddm_rh1 lesson reproduced on a second, independent
surface: **a derivation buys a better candidate, never a skipped race.**

The honest bound in our favour: the coded class map is the GT class **alone**. A deployed receiver
also knows the base partition's wrong class at each site and can exclude it, so every label price
above is a conservative **UPPER** bound. The conclusions below hold at that upper bound.

---

## §3 What the label term does to the decision

### 3.1 The band's lower edge moves 2.56×

| edge | ρ_c | source |
|---|---:|---|
| position-only, registered | 5.02e-4 | pp1 law |
| position-only, recomputed by dc1 | **5.0146e-4** | the canary — 0.11% apart |
| **position + label** | **1.2853e-3** | dc1, this arm |

The rational-correction band is **~[1.3e-3, 1e-2]**, not [5e-4, 1e-2]. The carrier design spec
relaxes with it: a carrier must be natively **≤ ~1.3e-3** (not ≤ 5e-4) to make a correction stream
pointless. PR130's 3e-4 rail still clears it comfortably. Registered as
`ddm_dc1_correction_stream_label_cost_v1` (append-only sister; the pp1 law is **not** mutated).

### 3.2 ba31 §B.3 re-priced

**ja1/v4c base, d_seg 0.00431179 → 508,639 flips, seg 0.431179 S:**

| price basis | B/flip | bytes | rate | **NET S** | % of gap-to-bar |
|---|---:|---:|---:|---:|---:|
| ba31 §B.3 coherent POSITION only | 0.6702 | 340,882 | +0.226980 | **−0.204199** win | 25.79% |
| **MEASURED position + label (dc1)** | **0.8832** | 449,252 | +0.299139 | **−0.132040** win | **16.67%** |
| MEASURED position + BLIND label bound | 0.9604 | 488,510 | +0.325279 | −0.105900 win | 13.37% |
| UNIFORM position bound + BLIND label | 1.2724 | 647,207 | +0.430948 | −0.000231 win | 0.03% |
| QA03 composite (a different object) | 1.4518 | 738,426 | +0.491688 | +0.060509 lose | 7.64% |

**burn-4 ep854, d_seg 0.003943024 → 465,138 flips:** −0.178074 / **−0.110828** / −0.088182 /
+0.004783 / +0.055334.

Three things this says that the position-only table did not:

1. **The sign survives.** Paying the measured label cost costs 55% of ba31's claimed win but does
   not invert it: −0.132 S is still 16.67% of the gap-to-bar.
2. **The most pessimistic priceable construction sits exactly ON water.** "Uniform position bound
   + blind label" lands at −0.0002 S (ja1) / +0.0048 S (burn) — break-even to four decimals. That
   is a strong independent confirmation that 1.2731 is the right yardstick, and it means the win
   depends entirely on the *coders being real*, which they now measurably are.
3. **The largest single term in the swing was never a price at all.** ba31's 0.264 S swing runs
   from the position-only price to QA03's composite. Of that, the genuine measured correction to
   the position-only number is 0.072 S (label). The remaining 0.192 S is the gap to an object
   that is not a correction stream.

### 3.3 The caveat that governs all of §3.2 — carried, not buried

This is the **idealised "correct every flip to zero residual"** calculation. QA03 addressed 1,866
flips = **0.367%** of the residual. §3.2 prices the **SLOPE**, extrapolated ~272×; it is **not a
realized move**, and no correction stream fixing 508,639 flips has been built, coded, or scored.
The family is **NOT proven a win**. What §3.2 establishes is narrower and sufficient: the number
the pool was closed on could not have decided it, and the decidable number does not close it.

---

## §4 Job 2 — the solver, named

### 4.1 The deficiency

`tools/sb1_seg_batch.py`, `qa03_gn_solve`, the per-instance descent loop. Two exits existed:

* a **genuine convergence test** — `if best is None or best[0] >= cur: break` (no improving
  single-quantum move). This is present and correct, and is the *intended* terminator.
* a **hard cap** — `for _step in range(args.max_quanta)` with `--max-quanta` **default 4**.

**The cap fired first on 42.5% of instances.** MEASURED from `qa03_instances.jsonl` (120 rows):

| accepted steps | 0 | 1 | 2 | 3 | **4 (the cap)** |
|---|---:|---:|---:|---:|---:|
| instances | 12 | 11 | 27 | 19 | **51** |
| flips | 0 | 72 | 285 | 302 | **1,207** |

* The histogram **spikes 2.68× at the cap** (51 vs 19 at step 3) instead of decaying. That is the
  censoring signature, and it is **unconfounded**: instances stopping at k<4 stopped *voluntarily*
  (convergence), instances at k=4 stopped *forced*, each with an improving move still available.
* The censored 42.5% produced **64.7% of all realized flips** (1,207 / 1,866).
* Therefore **1,866 is a strict LOWER BOUND** on this formulation's converged yield. The `sb1`
  verdict — *"the −0.046…−0.138 free-solve band is NOT reachable by this formulation class"* — was
  drawn on a censored solve. `sb1` itself flagged "51/120 saturated at 4 quanta (per-cell line
  search NOT exhausted)"; the flag was recorded and then not carried into the verdict, and the
  receipt surfaced neither the count nor the fact.

**Sibling-defect check vs `terminal_pose_gn` (#850):** related but not identical. `terminal_pose_gn`
was capped at 2–3 relinearizations with **no convergence test at all**. QA03 **has** the convergence
test — it is simply outranked by an unreasoned cap. Same class (unreasoned iteration bound), milder
instance.

**A second, separate deficiency, named but not fixed:** `_best_single_quantum` probes 8 candidates
(4 channels × ±1) from the current point and takes the best — greedy coordinate descent with unit
steps on an integer lattice. There is no Newton step and no line search along the chosen direction,
so the memo's and receipt's label *"damped-Newton line search"* **overstates the method**. Extending
an accepted direction while it helps costs 1 scorer evaluation per extension versus 8 for a fresh
probe round — an 8× cheaper way to buy the same descent. Left unbuilt: it changes the *method*, and
a method change must be raced, not assumed.

**Third:** `top_k = 120` of ~648 non-zero atlas cells — 81% of the aimed population never touched.

### 4.2 The fix — built, wired, tested

Cheap: the convergence test already existed, so the fix is to stop letting the bound outrank it.

* `--max-quanta` default **4 → 32**, re-documented as a *safety bound, not a target*.
* Per-instance `stop_reason ∈ {converged, no_move, cap}`, emitted via `for/else` — the only
  construct that distinguishes "ran out" from "converged at k".
* Receipt-level `n_cap_saturated` + `cap_saturated_frac` + a note stating that any run with
  `n_cap_saturated > 0` is **CENSORED, not solved**. The resume path infers the legacy rows'
  stop reason from their step count so historical runs are counted too.

That last part is the **class** fix, not the instance fix: raising the cap fixes one run; making
the stopping rule a first-class receipt field means no future consumer can read a censored solve
as a converged one without the receipt saying so.

`tools/tests/test_sb1_seg_batch_stop_reason.py` — 10 tests, scorer-free, all pass. They include a
regression that pins the exact descent that is censored at 4 and converges at 32, and a guard that
**fails if the default returns to 4**. They are not tests that would pass on broken code: reverting
either the default or the `for/else` fails them.

### 4.3 What re-measuring would take

`SegRuntime.pair_flips` (`sb1_seg_batch.py:129-134`) runs `render_frame1_camera_uint8` then the
frozen CPU-torch SegNet per candidate. Re-running QA03 is a **scorer job**, which this arm's
charter forbids, so I did not take the slot. The quote: the original 120 instances × ~2.7 accepted
steps cost 26.6 min wall; uncapping continues only the 51 censored instances, at 8 SegNet
evaluations per additional step. **Order 30–60 min on one scorer slot, resumable from the existing
`qa03_instances.jsonl`** (the resume path replays accepted edits and skips processed instances, so
the *converged* 69 instances are not re-run). That is the cheapest measurement in this whole
document and it is the one I could not take.

---

## §5 The §B.5 sign — adjudicated

wr2 flagged that ba31 §B.5 *"may carry a sign inversion labelling under-water carriers
'dominated'."* **There is a sign defect in §B.5. It is ba31's, not gr1's.**

**The two rows measure opposite-signed quantities:**

| row | quantity | "below water" means |
|---|---|---|
| gr1 token-granular, 0.04–0.51 B/flip | bytes **SAVED** per flip **INTRODUCED** (coarsening) | **LOSE** — flips sold below their concede value |
| W1-COH phase carrier, 0.075–0.141 B/err | bytes **SPENT** per flip **FIXED** (correction) | **WIN** — cheaper than conceding |

gr1 states its own convention explicitly: *"Water break-even = **1.273 bytes saved / flip
introduced**."* of1 states its own: *"B/err = phase_bytes / **flicker_flips_fixed** vs the 1.2731
water."* Same number line, opposite preferred directions.

**(a) gr1's "DOMINATED" is CORRECT**, and not for the water reason. gr1's verdict rests on two
things measured on its own axis: every token-granular candidate is worse than the current point on
realized seg+rate (best `tok_drop27` at **+0.086**), and cell-granular drop **strictly** dominates
at every byte budget (cell has both lower bytes *and* lower d_seg). Its under-water position is the
*reason* the verdict is right, not evidence against it.

**(b) of1's "PRICED, OPEN door" is also CORRECT** — 9–17× under water in the spend-to-fix direction
is a genuine open door, gated (as of1 says) on receiver support-derivability, not on coherence.

**(c) The defect is ba31's inference.** §B.5 puts both directions in one `vs water 1.2731` column
and then reasons: *"the two carrier families that were called dominated are dominated by an order of
magnitude, while the correction that was called break-even is 13.9% over… they are not the same
verdict at different confidence; **they are opposite ends of the same real line**."* Two errors:
W1-COH was never called dominated (of1 called it an open door), and the "same real line" only holds
under a shared sign convention, which these rows do not share. The paragraph reads gr1's
deep-under-water position as evidence its DOMINATED verdict was too harsh; it is the opposite.

**(d) Compounding:** the same table's QA03 row (1.45, "1.139× ABOVE") is a third object again — the
token-stream re-encode delta of §2.1, neither spent-to-fix nor saved-by-adding. §B.5 has three
currencies in one column.

**Correct form of the §B.5 table** (proposed for whoever owns ba31; APPEND-ONLY, I did not mutate it):
split into two tables by sign convention — *spend-to-fix* (want below water: W1-COH 0.075–0.141
WIN; correction position+label at our base 0.883–0.915 WIN) and *save-by-adding* (want above water:
gr1 0.04–0.51 LOSE) — and move QA03 out of both with the note that it prices neither.

---

## §6 Verdict on the aimed-correction family

**Scope: FORMULATION.** Not INSTANCE (two independent defects, not one bad run), and emphatically
not FAMILY.

* What the 1.4518 composite supports: *this solver, at this base, censored at 4 quanta, priced by
  re-encoding an existing token stream* did not beat conceding.
* What it does not support: that a correction stream is priced out at this base. It prices neither
  half of a correction stream's cost.
* What is now MEASURED: a correction stream at our base costs **0.883–0.915 B/flip** (position +
  label, real round-trip-verified coders, blind-to-base upper bound) = **0.69–0.72× water**, and
  the idealised full correction nets **−0.132 / −0.111 S** = **16.7% / 14.0% of the gap-to-bar**.
* What remains UNMEASURED and gates any spend: (i) whether an *aimed* solver can reach a support
  large enough for the slope to matter — QA03 reached 0.367% of the residual, censored; (ii) the
  coherence of a **real** aimed support versus pp1's margin-thresholded supports (QA03 stored no
  per-pixel support, so this is UNKNOWN, not small); (iii) the **solver** cost, which neither law
  prices; (iv) the r7/SMEVR shipping price that the QA03 receipt itself defers to.

**Recommended pool disposition:** `do_not_spend` was set on an undecomposed number that could not
decide it, and should not stand on that basis. But nothing here earns `spend` either — the reach
question (i) is untouched. The correct state is **`decidable_next`**, with the deciding measurement
already named, costed, and resumable: uncap QA03 and re-run the 51 censored instances on one scorer
slot. If the converged yield stays near 1,866 the formulation verdict stands on honest ground for
the first time; if it moves materially, the `sb1` band re-booking has to be redone.

**UNDEFINED, not small:** the coherence of a real aimed correction support. Everything in §3.2 is
quoted on pp1's supports and inherits their coherence. There is no second term to compare against.

---

## §7 Round-1 adversarial review of my own output

| # | challenge | outcome |
|---|---|---|
| 1 | Did the position numbers actually reproduce pp1, or did I read the same file back? | **Independent.** dc1 recomputes from `gt_n600.npz` through `contour_encode_frames`; pp1's JSON is never read by the tool. Match to 4 dp at 5 taus and 0.11% on the derived crossing. |
| 2 | Is `position + label` a *decodable* composition? | The all-#307 pairing is round-trip verified (flips **and** labels, 8 frames). The mixed (contour position + LZMA labels) pairing is decodable in principle but **not** jointly verified — so §3.2 and the §2.3 TOTAL column use the **all-#307, verified** pairing, which is the more expensive one. |
| 3 | Would my tests pass on broken code? | No. Reverting `--max-quanta` to 4 fails `test_max_quanta_default_is_not_the_censoring_value`; replacing `for/else` with a length check fails `test_cap_branch_is_reached_via_for_else_not_a_post_hoc_length_check`. |
| 4 | Is the "+7.77 marginal flips at the cap step" claim sound? | **Withdrawn from the argument.** It is a cross-instance mean difference, and 4-step instances are selected for being productive. Confounded. The load-bearing claims are instead the **2.68× histogram spike** and the **voluntary-vs-forced stop** asymmetry, neither of which depends on yield. |
| 5 | Did I fix the class or the instance? | Both, separately: raising the cap fixes the run; `stop_reason` + `n_cap_saturated` in the receipt fixes the invisibility, which is the class. The *greedy-vs-line-search* defect is named and left unbuilt — it is a method change and must be raced. |
| 6 | Any silent instrument? | One found and fixed mid-arm: my first §B.3 re-pricing passed a **descending** x array to `np.interp`, which silently returned the endpoint and made every row "LOSE". Caught because the ja1 position number failed to reproduce ba31's 0.6702. The reproduction check was the canary that caught my own bug. |
| 7 | Am I shading toward the exciting answer? | The headline hypothesis (**label is near-free**) is the one I **refuted**, and the generic control **beat** my coherent coder. Both cut against the interesting story. |

---

## §7b Fresh-eyes verifier round (independent, fresh context) — and what it changed

A fresh-context verifier re-derived all five load-bearing claims from the primary artifacts.
It **confirmed** §2.1 (QA03 has no correction stream — it independently quoted
`sb1_seg_batch.py:21-25, 97-101` and confirmed `price_bytes()` re-encodes **all four archive
sections**, so the 2,709 B is a whole-ARCHIVE delta, not even a tokens-section delta), §2.2
(pp1 zeroed the class map; with zeroed maps the `cls` stream is **12–18 bytes**, so pp1's
"trivial" comment was accurate *for its own run*), the canary at 5 taus with **matching support
sizes**, and the whole of §5. It found **seven defects, four of them mine**, and I fixed all four:

| # | defect in MY work | fix |
|---|---|---|
| 1 | Docstring promised a cross-instrument positive control (reproduce pp1's receipt) that the code **never implemented** — only within-run invariance was checked | **Implemented** `pp1_cross_check()`: loads the pp1 receipt, requires equality at the 5 taus where pp1's best coder was the contour walk and one-sided "not cheaper" at the 2 where it was LZMA; result serialised into the receipt as `pp1_cross_check` |
| 2 | Receipt `evidence_axis` claimed "round-trip-verified" for the n600 rows; only an 8-frame subset is verified | `evidence_axis` **rewritten** to state the round-trip scope explicitly |
| 3 | "generic beats coherent at ρ ≥ 1.4e-3" — wrong threshold | corrected to **ρ ≥ 5.6e-4, 6 of 7 densities** (coherent wins only at 2.2e-4) |
| 4 | `log2(5)/8` written as 0.29036 in prose (code was correct) | corrected to **0.2902410** throughout |

It also caught that **τ=0.8 disagrees with pp1 for the same reason as τ=0.4** (I had named only
0.4) — footnote corrected — and independently re-priced §B.3 using the best-of-two label coder,
landing **−0.143 S** against my conservative all-contour **−0.132 S**. Both are wins; the 0.011 S
spread is the coder-pairing choice, and I report the more expensive one.

Three defects it found in `ba31` that I had not: **(i)** ba31's §B.2 decomposition subtracts a
position-stream price from a whole-archive re-encode delta and attributes the remainder to
"label + solver overhead" — ba31 half-sees this (*"the 1.45 B/flip is not a position-cost
measurement at all"*) and then decomposes it anyway; **(ii)** §B.5 labels gr1's rows
`token-granular **corrections**` when gr1's candidates are **coarsenings** — the same category
error in miniature; **(iii)** §B.5 says W1-COH was *"called dominated"* when of1's actual verdict
is CHANNEL ADMISSIBLE / open door.

### Remaining gaps — named, not closed

1. **No control on a REAL flip support.** Both pp1 and dc1 use `margins < tau` as a stand-in for
   "where a base is wrong." Position cost depends on support *geometry*; **label cost depends on
   class *composition***, and error sites are heavily class-skewed (ba31 §A.4: Lane error-rate
   25.72% vs Undrivable 0.10%). The label half is therefore **more** proxy-sensitive than the
   position half was. This is the largest gap in this arm and it is why every price above is
   quoted on pp1's supports. Closing it needs the ru1 atlas flip records or a pa1r `control_tail`
   field — cheap, scorer-free, and the natural next rung.
2. **The base-class-exclusion slack is argued, not measured.** No 4-ary control (log2(4)/8 = 0.25)
   and no exclusion-coded run. The "upper bound" claim is sound reasoning, not a meter.
3. **The mixed coder pairing** (contour position + LZMA labels) is decodable in principle but not
   jointly round-trip verified; §3.2 therefore uses the all-#307 verified pairing.

## §8 6-hook wire-in (Catalog #125)

sensitivity-map: N/A (no new per-axis weights) · Pareto: **ACTIVE** (the band's lower edge moves
2.56×; the carrier native-error spec relaxes from ≤5e-4 to ≤1.3e-3) · bit-allocator: **ACTIVE**
(label stream is a priced section; the generic raster coder wins above ρ 5.6e-4) · cathedral
autopilot: N/A (no dispatchable archive) · continual-learning: **ACTIVE** (canonical equation
`ddm_dc1_correction_stream_label_cost_v1` registered; receipt mirrored) · probe-disambiguator:
**ACTIVE** (this arm IS the disambiguator between "corrections are priced out" and "corrections
were never priced").

## §9 DAG FEED — ddm_dc1 (2026-08-01)

- **FEED-dc1-a [MEASURED]** correction-stream **LABEL** cost = 0.082–0.255 B/flip over ρ
  2.2e-4..2.2e-2 = **0.28–0.88× the blind 5-ary bound** (best of two coders). ba31's "label largely predictable from
  neighbours" **REFUTED as stated** (gain 1.14–1.93×, not near-free). A **generic** LZMA raster
  label coder **beats** the neighbour-conditioned #307 `cls` stream at 6 of 7 densities (ρ ≥ 5.6e-4) — the rh1
  lesson on a second surface. Positive control: position reproduces pp1 to 4 dp, position-only
  crossing 5.0146e-4 vs registered 5.02e-4. Negative control + bit-exact flip/label decode PASS.
  Receipt `.omx/research/ddm_dc1_label_price_n600_20260801.json`. `[macOS-CPU advisory]`.
- **FEED-dc1-b [MEASURED]** the rational-correction band's **lower edge moves 2.56×** with label
  paid: ρ_c 5.02e-4 → **1.285e-3**. Carrier design spec relaxes to ≤ ~1.3e-3 (PR130's 3e-4 rail
  still clears). Registered `ddm_dc1_correction_stream_label_cost_v1` (append-only sister; pp1 law
  unmutated).
- **FEED-dc1-c [MEASURED / category]** **QA03's 1.4518 B/flip is not a correction-stream price.**
  It is the tr1 archive re-encode delta of in-place token edits; the receipt's own `byte_delta_note`
  defers the true shipping price to the r7 SMEVR coder. Comparing it to the water level or the pp1
  position law is out of domain. Added to the new law's `excluded` list.
- **FEED-dc1-d [MEASURED]** **QA03's solver was CENSORED, not converged**: `--max-quanta` default 4
  outranked the convergence test on **51/120 (42.5%)** of instances, and those produced **64.7%** of
  realized flips; the step histogram spikes **2.68×** at the cap. 1,866 flips is a strict LOWER
  BOUND. FIXED (default 32; `stop_reason`; `n_cap_saturated` in the receipt) + 10 scorer-free tests.
  Re-measure = ~30–60 min on one scorer slot, resumable. The *"damped-Newton line search"* label
  also overstates the method (greedy unit-step coordinate descent, no line search) — named, unbuilt.
- **FEED-dc1-e [DERIVED]** ba31 §B.3 re-priced with the measured label term: idealised full
  correction nets **−0.132 S (ja1) / −0.111 S (burn)** = **16.7% / 14.0% of gap-to-bar** — sign
  survives, magnitude down 55% from ba31's −0.204. The most pessimistic priceable construction
  (uniform position bound + blind label) lands within 0.0002 S of break-even. **Still the idealised
  slope, not a realized move** (QA03 reached 0.367% of the residual).
- **FEED-dc1-f [ADJUDICATED]** ba31 §B.5 **does** carry a sign defect, but gr1's "DOMINATED" is
  **correct**. gr1 prices bytes-**saved**-per-flip-**added** (below water = LOSE); of1/W1-COH prices
  bytes-**spent**-per-flip-**fixed** (below water = WIN). ba31 puts both — plus QA03's third
  currency — in one "vs water" column and concludes they are "opposite ends of the same real line."
  They are opposite-signed economies. Proposed correction: split the table by sign convention.
