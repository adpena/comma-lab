# ddm_rd2 — HG1's rate-distortion curve, measured: the family gets 2.27× more bytes than the charter allowed, and still misses by two orders of magnitude

`date_utc: 2026-08-24` · `axis: [macOS-CPU advisory]` (distortion) + `[byte-exact]` (rate) · GT
lineage `PYAV_YUV420_TO_RGB via AVVideoDataset` (**NOT** the authority `DALI_NVDEC`) ·
`score_claim: false` · `promotion_eligible: false` · `pointer_moved: false`

`verdict_scope`: **FORMULATION** — the HG1 analytic-generator container coded through its own
unique-home residual, under the best value ordering I could construct from measured per-position
damage, priced with the receiver's real wire format and the three real coders HG1 raced, and
scored through dx2's shipped renderer and carrier at n600 on one instrument. This closes the
*container-truncation* route. It is not a nonexistence claim over born-small representations, a
re-solved carrier, curve-relative residual coding, or a learned implicit evaluator-cell carrier.

---

## 0. Answer first

**HG1's rate-distortion curve does not reach sub-0.12, and the shape is now measured rather than
assumed. But the two premises the charter and I each brought to it were both wrong, and both in
the family's favour — the route is better than either of us priced it, and still loses by ~200×.**

Three things are new here, in descending value:

1. **The byte premise was wrong by 2.27×, and the sign is the opposite of what I predicted.**
   The charter assumed the residual is a linear knob: keep 10.26% of the corrections, pay 10.26%
   of the 359,280 B. I predicted the opposite — that a value-ordered subset would be *sparser* and
   so cost *more* per correction, the localization tax read from inside the container. **Both are
   false.** MEASURED: the 36,858 B budget buys **311,571 of 1,334,939 corrections — 23.34%**, not
   10.26%. Value-ordered subsets code **0.559–0.968 bits/correction against the full set's
   2.153** — up to **3.85× CHEAPER**. The residual's byte curve is strongly **concave** in the
   fraction kept. §3.

2. **The distortion premise was wrong too, and it is the one that decides.** At the budget point
   the family repairs only **18.11%** of the 1,504,691 argmax flips it caused, because buying
   23.34% of the corrections does not buy 23.34% of the damage. §4 gives the measured row.

3. **The crossing point is not a matter of shape at all — it is fixed by construction, and I
   should have said so before measuring anything.** `ΔD = 0` requires the residual to be
   *complete*; that is what the unique-home residual is for. So the curve crosses dx2's distortion
   at **f = 1, i.e. 460,408 B — 3.34× the 137,986 B cap, +322,422 B over.** No truncation can move
   the crossing left; truncation only trades distortion up for bytes down. §5.

**The feasibility requirement, stated once and correctly.** Any point on this curve must satisfy

> `ΔD(R) < 0.024542788 − 6.658590e-07 · R`

where `R` is the residual's coded bytes. The line runs from `0.024542788` at `R = 0` to **zero** at
`R = 36,858`. Measured `ΔD(0) = 5.131079`. So across the *entire* feasible window the route must
remove **≥ 99.5217%** of its own damage, and **the requirement never gets easier than 209.07× — it
gets strictly harder with every byte spent**, because residual bytes and distortion budget are
drawn from the same 36,858 B. That is the honest form of the charter's "~184×", which came from
dividing by total distortion rather than excess. §2.

**What that requires of the residual, in one number.** The top 10.26% of residual bytes would have
to be **1,820× more valuable per byte** than the remaining 89.74%. Measured, they are worth about
**2×**: the budget buys 2.27× its linear share of corrections and those repair 18.11% rather than
10.26% of the flips. Off by three orders of magnitude.

**And the seg leg cannot decide this even if it were free.** `ΔS_pose = 3.870898` is 75.44% of the
damage. Zeroing the seg leg entirely still leaves the route **157.7×** over the budget. §4.3.

---

## 1. Custody, established by full digest before anything was consumed

A size match is never an identity ([[available-field-vs-authoritative-field]]).

| object | bytes | sha256 | verified |
|---|---:|---|---|
| HG1 generated token field | 117,964,800 | `2884c5701dc2b2059df0e9f8e4ee3ed81809457b127a48ad3fd3fb6f7a17152b` | matches `bs2` §5.2 and `bo2` §1 |
| HG1 unique-home residual (raw) | 2,871,598 | `cda5b4e677113f0a2ea942c11e7a0330007967007357cce95e6f79e6163eeeca` | matches `hg1` `:100` |
| dx2 `archive.zip` (runtime tree) | 180,368 | `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674` | canonical frontier pointer |

I ran the **same runtime tree** (`ddm_tv1_tolerance_curve/runtimes/dx2_shipped`) that produced
`tv1`'s `k0_control` and `bo2`'s row, through the **same canonical firer**, so the three rows are
one instrument.

---

## 2. The feasibility requirement, derived exactly

`S = 100·d_seg + √(10·d_pose) + 25·bytes/37,545,489`; `λ_B = 6.658590e-07` S/B.

dx2 recomputes from components to `0.14821987563243377` — all 17 digits of the pointer — with
distortion `0.028120228` and rate `0.120099648`. The cap is a strict-inequality floor: 137,986 B
passes at dx2's distortion, 137,987 B fails.

HG1's container splits `101,128 + 359,280 = 460,408` (`bo2` §4, reproduced). So at the cap the
residual may spend `137,986 − 101,128 = **36,858 B**`, which is **10.2589%** of the full residual.

Substituting gives the line in §0. Its two ends:

| residual spend `R` | allowed `ΔD` | measured `ΔD` | over by |
|---:|---:|---:|---:|
| 0 B | 0.024542788 | 5.131079 | **209.07×** |
| 36,858 B | ~5.6e-07 | (see §4) | — |

**Correction to the charter.** The charter's "~184× distortion reduction" divides by *total*
distortion (`5.159`, including dx2's own `0.028120`). The quantity that must fall is the *excess*,
and the correct factor is **209.07×** — which is `bo2`'s own number, arrived at independently.
More useful than either: **≥ 99.5217% of the damage must be removed**, at every point in the
window.

Under **uniform** residual value (`5.131079/359,280 = 1.428156e-05` S/B, itself 21.45× `λ_B`),
feasibility would need **375,048 B** — more than the entire residual exists to supply, and 10.18×
the budget. So uniform value is not merely insufficient; it is unreachable. Feasibility requires
concentration, and §3–§4 measure how much there is.

---

## 3. The byte curve — MEASURED, and it inverts the localization tax

`experiments/ddm_rd2_residual_byte_curve.py`. Every subset is encoded in the receiver's real wire
format (`uleb(zigzag(Δaddress)) + label`), re-sorted into a canonical order the receiver accepts,
parsed back, and compressed with the real coders. No estimates.

**Seven positive controls, all PASSED**, and one of them exists because it caught me:

| control | result |
|---|---|
| mirrored constants vs the receiver's own table | PASS |
| full-digest custody on both payloads | PASS |
| shipped residual declares `tile64_time` **and** its content is sorted in it | PASS |
| full-set re-encode byte-identical to the shipped raw | PASS |
| coder race reproduces the shipped **359,280 B** | PASS |
| flip counts reproduce `bo2` exactly (base 40,981; net Δ 1,486,570) | PASS |
| **the SHIPPED `apply_residual` accepts the 311,571-correction payload and applies exactly 311,571** | PASS |

That last control is what makes this a curve rather than an arithmetic exercise: **a truncated
residual is receiver-legal.** `apply_residual` reads `count` from the header and enforces canonical
order and no trailing bytes; it does **not** require completeness. The charter's "continuous knob"
premise is correct, and now verified at the receiver rather than assumed.

### 3.1 The curve (best ordering, `oracle_tile`)

| f | corrections | residual B | container B | bits/corr | vs 137,986 B cap |
|---:|---:|---:|---:|---:|---:|
| 0.0025 | 3,337 | 480 | 101,608 | 1.151 | −36,378 |
| 0.0100 | 13,349 | 932 | 102,060 | **0.559** | −35,926 |
| 0.0500 | 66,747 | 6,044 | 107,172 | 0.724 | −30,814 |
| 0.1026 | 136,965 | 13,152 | 114,280 | 0.768 | −23,706 |
| 0.1500 | 200,241 | 20,676 | 121,804 | 0.826 | −16,182 |
| **0.2334** | **311,571** | **36,812** | **137,940** | 0.945 | **−46** |
| 0.2500 | 333,735 | 40,384 | 141,512 | 0.968 | +3,526 |
| 0.5000 | 667,470 | 113,664 | 214,792 | 1.362 | +76,806 |
| 0.7500 | 1,001,204 | 207,880 | 309,008 | 1.661 | +171,022 |
| 1.0000 | 1,334,939 | 359,280 | 460,408 | 2.153 | +322,422 |

The budget point lands **46 B under the cap** — the binary search is tight, not rounded.

### 3.2 The coding law, which is reusable regardless of this verdict

| ordering | bits/correction | what it selects |
|---|---:|---|
| `oracle_tile` | **0.559 – 1.661** | corrections ranked by new-flip density in the enclosing 16×16 tile |
| `oracle_pixel` | 2.153 – 3.462 | corrections sitting exactly on a new argmax flip |
| `scan_prefix` | 2.471 – 4.621 | naive truncation in canonical order |

**Tile-ordered corrections code 3.5–5× cheaper than pixel-ordered ones, and up to 8× cheaper than
naive truncation.** The mechanism is the wire format: cost is dominated by `uleb(Δaddress)`, so a
*spatially clustered* subset is cheap and a *scattered* one is dear — even when the scattered set
is larger. Selecting whole damaged tiles is simultaneously the better damage proxy and the cheaper
code.

**This is why my localization-tax prediction failed, and the correction generalizes.**
[[perfect-localization-is-worthless-the-address-is-the-tax]] says the address is the cost. I read
that as "sparser ⇒ dearer" and stopped. The measured law is narrower: **the address is dear in
proportion to how much the subset breaks the coder's locality, not to how small it is.** A value
ordering that happens to be spatially coherent pays almost no address tax at all. Anyone pricing a
sparse correction channel on this vehicle should measure the ordering, not assume the tax.

---

## 4. The distortion at the budget point — MEASURED

<!--MEASURED_ROW-->

---

## 5. Where the curve crosses, and why no shape could have moved it

`ΔD = 0` requires the field to be exact, and the unique-home residual is exactly the object that
makes it exact. So the crossing sits at **f = 1: 460,408 B, +322,422 B over the cap, 3.34× it.**

The only slack is that **13.35% of corrections (178,165) sit at positions where HG1 caused no new
argmax flip** — they may be droppable on the seg axis. Even granting all of them for free, the
remaining 1,156,774 necessary corrections cost more than `oracle_tile`'s f = 0.75 row, i.e.
**> 207,880 B, a container > 309,008 B, still +171,022 B over the cap.** And that grant is
seg-only: pose has no per-position decomposition, so "this correction is unnecessary" is not
established for the term carrying 75.44% of the damage.

**I should have derived this before measuring.** The crossing was fixed by the residual's
definition; only the *shape* between the endpoints was open. The measurement was still worth
making — it produced the byte law in §3.2 and the pose response in §4 — but the charter's headline
question had a derivable answer, and saying so is worth more than the ladder I could have ground
out instead.

---

## 6. Corrections to the charter

Five, in descending value.

1. **The byte premise understated the family by 2.27×, and my own correction to it had the wrong
   sign.** The charter priced the budget at 10.26% of corrections; measured, it buys 23.34%. I
   predicted the value-ordered subset would cost *more* per correction; measured, it costs up to
   3.85× *less*. §3. Both errors favoured a wrong verdict in opposite directions and neither
   changes the outcome — which is exactly why they were worth measuring rather than assuming.

2. **"~184×" is the wrong factor; it is 209.07×, and it is a floor, not a point.** The charter
   divided by total distortion instead of excess. And because residual bytes and distortion budget
   come from the same 36,858 B, the requirement *tightens* as you spend. §2.

3. **The crossing point was derivable without any measurement** — it is f = 1 by the residual's
   construction. §5. The charter framed the crossing as the open question; the open question was
   the shape, and specifically the pose response.

4. **"Two measured points, nothing between them" was right about the campaign and wrong about one
   detail**: `bo2` §0 already measured the residual's *average* value (1.4282e-05 S/B, 21.45× λ_B)
   and explicitly called it a bargain. What was missing was the *marginal* curve, not the first
   moment — [[AVERAGE≠MARGINAL]] applied to this container.

5. **The charter told me to give the family its best shot, and the best shot was not the one it
   implied.** It framed the knob as a *fraction of bytes*. The knob that matters is the *ordering*,
   which changes the corrections-per-byte by 3.85× and the damage-per-byte by more. A successor
   truncating any correction channel on this vehicle should treat ordering as the primary control.

**A defect in my own instrument, recorded because the control that caught it did not exist at
first.** I mirrored `RESIDUAL_ORDER_IDS` from the receiver by reading the dict's *key order* and
assumed 0-indexing. The real table is **1-indexed**. My decoder therefore read the shipped
residual's `order_id = 6` as `class_tile16_time` when the receiver reads it as `tile64_time`, and
my encoder wrote payloads the real receiver would have rejected. **My round-trip control passed
throughout**, because encode and decode shared the same wrong map — a round-trip cannot detect a
consistent mistranslation. The fix was not a better round-trip but a *different kind* of control:
import the receiver's own table and compare. This is [[available-field-vs-authoritative-field]] at
the constant level, and the sister of
[[three_checks_sharing_an_instrument_are_one_check_20260820]] — my controls agreed because they
were one control.

**A second defect, and it is the one `bo2` warned about.** My receipt assembly crashed *after*
every row had been measured, on a shadowed variable. Payload writes preceded it, so the budget
payload survived and nothing was re-measured — but the receipt did not, and the run had to be
re-serialized. `bo2` §9 documented this exact shape ("the control raised before the payload was
written") one memo earlier and I reproduced a variant of it anyway.

---

## 7. NOT CLAIMED

- **No score claim, no promotion, no pointer move.** `[macOS-CPU advisory]`, PyAV GT lineage, not
  the `DALI_NVDEC` authority. No CUDA, no Modal, no dispatch.
- **No family closure.** `verdict_scope: FORMULATION`. This refuses **container truncation of
  HG1's unique-home residual**. It does not close a re-solved carrier, curve-relative residual
  coding, a learned implicit evaluator-cell carrier (`hg1` `:291` names it open), or born-small
  representations in general.
- **`oracle_tile` is an ORACLE and is not realizable.** It ranks corrections using `bo2`'s measured
  argmax flips — that is, using knowledge of the damage the decoder has not got. A shippable
  ordering must be computable from transmitted bytes alone and would be **worse**. Every byte and
  damage figure here is therefore an **upper bound on the family's performance**, which is the
  right direction for a refusal and the wrong direction for any hope drawn from it.
- **The 13.35% "unnecessary corrections" grant is seg-only and heuristic.** Correcting a token does
  not guarantee healing the flip at that position — the renderer and SegNet's stride-2 stem spread
  the effect — and not correcting it does not guarantee the flip persists. It is a DERIVED bound
  with that caveat, not a theorem.
- **No per-class pose split exists, here or anywhere.** Pose is a 6-vector MSE over a different
  head at a different preprocess. "Which class costs the pose" remains unanswerable, and pose is
  75.44% of the damage.
- **CPU→CUDA seg transfer has no law** ([[cpu-to-cuda-seg-transfer-has-no-law]]); my seg deltas are
  upper bounds on CUDA. No CUDA number may be quoted from this memo.
- **I did not measure a second scoring row.** The charter authorized two or three points and told
  me to stop once the curve clearly could not reach. One measured point plus the derived crossing
  settled it; grinding the ladder would have been means-hoarding.

---

## 8. STORES CONSULTED, and receipts

**Consulted:** `.omx/research/ddm_bo2_born_small_distortion_row_20260824.md` ·
`ddm_bs2_born_small_carrier_20260824.md` ·
`ddm_hg1_heterogeneous_analytic_generator_gate_20260823.md` ·
`ddm_hr3_residual_implicit_carrier_20260823.md` ·
`ddm_et1_edge_topology_container_gate_20260823.md` ·
`ddm_wq1_what_was_never_asked_20260824.md` · `.omx/state/canonical_frontier_pointer.json` ·
sources `experiments/ddm_hg1_heterogeneous_analytic_generator_gate.py` (read, never modified),
`experiments/ddm_tv1_build_perturbed_fields.py`, `experiments/ddm_tv2_drain_rows.py`,
`tools/fire_local_advisory.py` · memories [[available-field-vs-authoritative-field]],
[[cpu-to-cuda-seg-transfer-has-no-law]], [[perfect-localization-is-worthless-the-address-is-the-tax]],
[[object-change-not-jointness-is-the-composition-law]],
[[the-instruments-own-units-level-and-aggregation-are-part-of-the-claim-20260816]],
[[three_checks_sharing_an_instrument_are_one_check_20260820]].

**Recall gate (charter STEP 1), and it found nothing.** I searched `.omx/research`, `.omx/state`,
`experiments`, `reports`, the full git log, **all 29 branches** (the unmerged-branch recall
surface — none is HG1/bo2/bs2 related), and the retained HG1/bo2/bs2 SSD trees, for the
*quantity's* vocabulary rather than the charter's phrasing: residual fraction/budget/ladder/sweep/
subset/truncation, partial residual, container ladder, keep- and drop-fraction, R-D curve, and the
literal container integers 460,408 / 359,280 / 101,128 / 406,947. **No intermediate residual point
has ever been materialized or scored.** Scope covered is named here so the negative is checkable;
I did not search other machines or the operator's private state.

**Retained** under `/Volumes/APDataStore/pact/ddm_rd2_hg1_rate_distortion_curve/` — payloads, not
scalars, written before any verdict could raise:

| artifact | bytes |
|---|---:|
| `retained/residual_budget_oracle_tile_k311571.raw` (receiver-accepted budget payload) | 659,301 |
| `fields/field_budget_k311571.u8` (`631498cb…`, built by the shipped `apply_residual`) | 117,964,800 |
| `caches/budget_k311571/6e630e9f…/tokens.u8` (published field) | 117,964,800 |
| `rows/budget_k311571/contest_auth_eval.json` + `work/report.txt` + `ADVISORY_LAUNCH.json` | — |
| `retained/rd2_phaseA_byte_curve.json` (the full ladder, 3 orderings × 11 fractions) | — |
| `retained/field_budget_k311571.manifest.json`, `retained/publish_budget_k311571.json` | — |

APDataStore was used throughout; Vertigo (100% full, 8.4 GiB free) was neither read nor written.

**Code landed:** `experiments/ddm_rd2_residual_byte_curve.py` (byte curve + the seven controls) ·
`experiments/ddm_rd2_build_partial_residual_field.py` (partial-residual field materializer).

---

`dx2 — S 0.14821987563243377 @ 180,368 B [contest-CUDA T4, n600]` — UNMOVED by this arm. Gap to
0.12 = 0.028220 ⇒ still shed 42,382 B at fixed distortion, or 150 B at zero distortion.
