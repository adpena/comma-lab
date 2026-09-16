# ddm_pd6 — the PRICE-FIRST generator on the move-52 field

Tokens: `[no-triality] [p0-ledger-ok]`

Lane `ddm_pd6_price_first_generator_20260916`. Base: **move 52**, S 0.13620226906030858 @ 179,332 B,
archive sha `ae59c5109597968409c4b581337b8e23b5b3c4262fd3710c315cc83913b3b20e`, re-derived from
`.omx/state/canonical_frontier_pointer.json` at bind time. Axis
`[macOS-CPU advisory, frozen CPU-torch PoseNet + SegNet, DALI GT lineage, n600]` for the distortion
legs; bytes EXACT through the shipped RLC1 coder and the shipped container. **No score is claimed.
Only `upstream/evaluate.py` on shipped bytes is a score, and MAIN fires.** No Modal, no authorization,
no fire, no completion, no packet.

Pose solver, stated once: every per-pair credit comes from
`ddm_jg5_pose_resolve_on_edited_renders.refine_pair` at PoseNet batch 1 on the moved render, with
`outer_rounds=40, max_gn_iterations=400` (pd4's cluster-search reference values); every n600 number
comes from `ddm_sj1_joint_admission`'s `pose` stage (`ddm_up2_shipping_pose_solve.measure_pose`,
batch 8) on the composed overlay this arm rendered.

---

## 1. What this arm inverted

Every pre-distortion pass on this object — sj1 passes 1–8, pd1 through pd5 — generated its proposals
by POSE SALIENCY and priced them afterwards. pd5 measured the consequence and closed the multi-token
formulation, and its §9 scoped that closure to exclude exactly one thing: the reverse order. This
arm runs that order and only that order.

The generator is the shipped coder's own model. `runtime.rlc1_mixer.LaneMixer` codes **every** one
of a frame's 196,608 positions (`end_frame` refuses a frame unless `self.seen.all()`), and every
coded position carries a full K = 5 probability row. So ONE real encode of move 52's own field yields
the first-order price of EVERY single-token change in the whole 600 × 384 × 512 plane:

```
delta_bits(position, symbol) = -log2 p_model(symbol) + log2 p_model(truth)
```

That is a RANKING, never a charge — it does not carry the group-causal geometry update or the
adaptive response the change itself causes. The CHARGE is a real 600-frame sheet encode, and no
credit is measured on anything the coder has not first charged.

## 2. The capture, and the two controls that make it credible

`experiments/ddm_pd6_price_first.py capture` wraps `LaneMixer.coding` and `.end_frame` as pure
observers — each calls the shipped method and returns its value unchanged — and runs the pricer's own
control encode.

| control | outcome |
|---|---|
| **the pricer re-packs move 52's own archive** | **179,332 B sha `ae59c5109597968409c4b581337b8e23b5b3c4262fd3710c315cc83913b3b20e`**, byte-identical to the live pointer, twins identical in-process, stream 119,097 B, decoded plane sha `117951bbd6800949…` — **with the capture wrappers installed** |
| **the captured rows ARE the charged rows** | per frame, the sum of −log2 p over the TRUE symbols must equal the pricer's own `per_frame_bits`. MEASURED over all 600 frames: **max absolute 0.00047 bits, max relative 3.216e−07** (float32 rounding) |
| the inherited decode | 3,662,409,600 B sha `ccb89e3e73bf61ac…eced`, re-hashed by this arm (pd3 and pd4 each parsed the same shipped bytes back cold and produced a bit-identical raw; pd4 measured max abs per-pair pose difference 0.000e+00 over 600 pairs) |

Capture wall: 1,213.6 s for the 600-frame encode plus the per-frame flush.

## 3. The cheap half, MEASURED

Per frame the capture keeps the 1,024 cheapest `(position, symbol)` pairs inside the argmax interior
(radius 3, which admits a median 178,453 of 196,608 positions — 90.8 %). Floor pairs are excluded;
they are pp1's twelve, and they are **exactly the twelve largest-d_pose pairs, holding 56.7 % of the
n600 pose mass**, so every number in this memo is drawn on the remaining 43.3 % over 588 pairs.

| quantity | MEASURED |
|---|---|
| pairs with candidates | **588** of 600 (the 12 floor pairs have none) |
| candidates emitted | **4,291** — 3,233 single-token, 1,058 two-token |
| held-fixed price-control pairs | **59** (every 10th pair contributes one candidate, repeated across all sheets) |
| pairs whose cheapest change is NEGATIVE-priced | **219 of 588 (37.2 %)** — the model would spend FEWER bits on the changed symbol |
| cheapest change per pair, first-order bits | median **0.771**, min **−31.258**, max **3.577** |
| the price ladder, ranks 0→7 (median) | **0.861 → 1.763 → 2.272 → 2.657 → 2.989 → 3.222 → 3.411 → 3.575** bits |
| population median candidate | **29.27** bits — the cheap tail is thin: the 1st percentile of all interior candidates is 11.9 bits |
| pairs with a cheap 8-neighbour of the head | **97.7 %** (the 2-token family is populated almost everywhere) |

**The ladder is the headline of the generator.** A pair does not have one cheap change; it has eight
well-separated ones, and the eighth still prices under four first-order bits. That is the breadth the
inversion was supposed to buy, and it is real.

### Where the cheap half lives

| SegNet class | share of the cheap half | area share of the frame |
|---|---:|---:|
| **Road** | **90.99 %** | 23.2 % |
| Undrivable | 6.40 % | 49.5 % |
| Movable | 2.46 % | 1.24 % |
| MyCar | 0.14 % | 25.4 % |
| Lane | 0.01 % | 0.59 % |

602,112 cells over 588 pairs; mean row 212.4 of 384, with 77.6 % of cells in rows 192–287. The cheap
half is the road surface just ahead of the car — where the ego-motion parallax lives — and the
dominant move is token 0 → token 1 (90 % of cells). Class indices are the canonical comma10k order,
self-checked by area and vertical centroid, never luma-sorted.

### The two pools are not the same object

pd4 and pd5 together priced **714** distinct cells. Of this arm's 588 pairs, the fraction whose
cheapest cells touch ANY of them:

| head | pairs touching the prior pool |
|---|---:|
| 8 cheapest cells | **3 of 588 (0.51 %)** |
| 64 cheapest | 5 (0.85 %) |
| 1,024 cheapest | 26 (4.42 %) |

The price-first generator nominates a part of the plane the saliency-directed passes never priced.
Whatever this arm measures is not pd5 under another name.

## 4. What the lever needs from credit — DERIVED before any credit existed

At move 52's operating point: S per archive byte 6.658589531221714e-07, S per seg cell
8.482752943113527e-07, S per unit of one pair's d_pose 1.2847092313591597 (DERIVED at this base
mean; it EXPIRES at the next pointer move).

A pair can credit at most its own d_pose. Walking the whole n600 base against that ceiling:

| recovered fraction of the pair's own d_pose | at 3 real bits/token | at 6 real bits/token |
|---|---|---|
| 5 % | 55 pairs pay, **−2.76e−05 S** | 26 pairs, −1.82e−05 S |
| 10 % | 87 pairs, −7.30e−05 S | 55 pairs, −5.53e−05 S |
| 20 % | 158 pairs, −2.09e−04 S | 108 pairs, −1.83e−04 S |

**The bar is −2e−05 S. Recovering 5 % of each pair's own d_pose at 3 real bits per token already
clears it.** That is the whole shape of the bet, written down before a single credit was realized
(`PREREGISTRATION.json`, `HEADROOM.json`). pd5's own rows say the break-even price for ITS median
credit is 7.255 bits/token — below pd5's own measured 12.300 median and inside this arm's cheap band.

## 5. The seg screen is a PER-PAIR budget, derived rather than picked

One flipped cell costs 8.482752943113527e-07 S. A pair can therefore only ever afford
`floor(base[pair] * pose_unit / seg_cell)` cells. MEASURED over the 588 non-floor pairs: **398 can
pay for none, 65 for one, 125 for two or more.** A flat screen is wrong in both directions — at 0
cells it discards the rows the heavy pairs can afford, at 2 cells it spends a 25-second refine on
light pairs that could never pay. Every refused row is written to disk with its own `d_cells`, so the
screen's cost is a measurement rather than an assumption.

## 6. THE CHARGE — eight sheets, 4,291 proposals, one real 600-frame encode each

Every candidate was charged before any credit was measured. Each sheet carries one proposal per
pair, so one encode charges the whole sheet frame-locally; twins byte-identical inside every encode,
every encode output-lossless.

| sheet | family | tokens | Δ bytes vs move 52 | **bits/token at sheet density** |
|---|---|---:|---:|---:|
| 00 | single, rank 0 | 588 | **+159** | **2.163** |
| 01 | single, rank 1 | 588 | +429 | 5.837 |
| 02 | single, rank 2 | 588 | +521 | 7.088 |
| 03 | single, rank 3 | 588 | +551 | 7.497 |
| 04 | single, rank 4 | 588 | +572 | 7.782 |
| 05 | single, rank 5 | 588 | +586 | 7.973 |
| **06** | **two-token, rank 6** | **1,117** | **+452** | **3.237** |
| **07** | **two-token, rank 7** | **1,117** | **+520** | **3.724** |

**pd5's eight sheets came back at 12.900 – 14.271 bits/token at the same kind of density. This arm's
cheapest sheet is 2.163, and its two-token sheets are 3.237 and 3.724.** pd4 DERIVED that 10.891
bits per changed token would clear the −2e−05 bar; the price lever is no longer the constraint —
it is beaten by a factor of five.

Per PROPOSAL, frame-locally:

| family | proposals | tokens | pooled bits/token | median | min |
|---|---:|---:|---:|---:|---:|
| single-token | 3,233 | 3,233 | 6.636 | 7.020 | −31.934 |
| **two-token** | 1,058 | 2,116 | **3.496** | **3.621** | −30.619 |

The context discount pd5 measured on the SECOND token survives being the generator rather than an
afterthought, and it is larger here: a two-token proposal costs about half as much per token as a
single one, because both members are cheap AND adjacent 97.7 % of the time.

**The cheap half is literally half.** Over all 4,291 charged proposals the median is 6.065
bits/token; 2,119 (49.4 %) price at ≤ 6. **389 proposals (9.1 %) carry a NEGATIVE real charge** —
the coder spends fewer bits on the changed field than on move 52's own.

| threshold | proposals | pairs | fraction of charged | median bits/token |
|---|---:|---:|---:|---:|
| ≤ 3 bits/token | 1,033 | 388 | 24.1 % | 0.943 |
| **≤ 5** | 1,728 | 506 | 40.3 % | 2.491 |
| **≤ 6 (the charter's threshold)** | **2,119** | **544** | **49.4 %** | **3.094** |
| ≤ 8 | 2,876 | 575 | 67.0 % | 4.244 |
| ≤ 12 | 3,830 | 585 | 89.3 % | 5.443 |

**The price's own noise floor, MEASURED and WORSE than pd5's.** 59 held-fixed control pairs are
repeated across all eight sheets: median spread **6.758 bits** (mean 6.851, max 14.867) against a
within-pair price range of median **12.424 bits** over 529 pairs. Signal is **1.84×** the noise,
where pd5 measured 3.3× (2.905 against 9.555). The reason is structural and worth saying plainly:
this arm's sheets edit **588** pairs where pd5's edited ~156, so every proposal's frame-local price
carries more spill from a crowd of neighbours. F3 does not fire, but the per-proposal price here is
a coarser instrument than pd5's, which is exactly why the SET re-price below is load-bearing rather
than a formality.

## 7. CREDIT AFTERWARDS — the wave

4,307 screens (render + frozen argmax), 3,104 refused by the per-pair seg budget, **1,203 realized
rows over 414 pairs**, seven shards, no prefix stop.

| quantity | MEASURED |
|---|---|
| credit improves the pair | **571 of 1,203 (47.5 %)** — the coin flip a pose-blind generator predicts (pre-registered 0.45–0.70) |
| best single credit | **−4.064e−06** d_pose |
| recovered fraction of the pair's own d_pose, per row | median −1.2 %, p75 **+17.5 %**, p90 **+56.5 %**, max **+98.5 %** |
| **best-of-n per pair** | **67.4 % of the 414 pairs improve**; recovered median **+10.9 %**, p75 **+46.2 %** |
| seg cells on realized rows | **51 repaid**, 687 neutral, 272 cost one, 193 cost two |
| seg screen pass rate | 27.9 % (single 30.8 %, two-token 19.2 %); refused rows' median d_cells 3 |
| **pairs with ANY positive resolved-pose credit** | **279** — F4's bar is 50 |
| **pairs whose best row pays at its OWN real price** | **128** |
| modelled net at the sheet price | **−1.2461e−04 S** |

**The carrier re-solve is doing almost all of the work, and this is the control that says so.**
Before the re-solve, a cheap token change makes the pair's pose **56.99× worse at the median** and
up to 2.4 million× worse; the re-solve recovers **98.0 %** of that damage. The credit this arm
reports is what survives AFTER the re-solve, never the stale number.

**The generator is pose-blind and the measurement says so.** Per row the median recovery is slightly
negative; the value is entirely in the tail and in being allowed eight shots per pair. That is the
honest shape of a price-first pool: it does not know where the pose is, it is simply cheap enough
that a coin flip can pay.

## 8. THE SET PRICE — and the thing only a real encode could show

pd4 measured that a per-bit rule under-charges the subset it chooses by 36.3 %, because the argmin
of a few noisy real prices is biased low. pd5's cure is to re-encode the SELECTED SET as one field.
This arm's noise floor is 6.758 bits — more than twice pd5's — so the cure is not optional here.

| iteration | pairs / tokens | ledger bits | real frame-local bits | **residual bias** | exact archive |
|---:|---:|---:|---:|---:|---|
| 0 | 128 / 147 | −547.56 | −420.31 | **−23.24 %** | **179,290 B** sha `127cac07…` (**−42 B**) |
| 1 | 110 / 129 | −563.44 | −531.03 | **−5.75 %** | **179,273 B** sha `b884a8bf…` (**−59 B**) |

**One iteration takes the residual from −23.2 % to −5.8 %**, inside F6's 10 % bar. The spill onto
unedited frames runs 26.1 % → 13.7 %, so the frame-local rail becomes MORE complete as the set
shrinks — the opposite of pd5's 10.4 % → 19.6 %, and worth saying plainly because it means pd5's
"the converged set is the worst" does not generalise: on this pool the smaller set is the better one.

**And the ledger's sign is the pass's second surprise.** The selected set does not cost bytes. Its
own exact encode comes back **SMALLER than move 52** — the negative-price tail of the cheap half is
real, it survives its own re-encode, and the rate leg pays the candidate instead of the other way
round. Every price-first pass this campaign has run assumed the rate leg was a cost to be minimised.
On this object, generated this way, it is a credit.

## 9. THE COMPOSITION, re-verified on one object

A 600-pair overlay rendered from the 128-pair set-00 field (**1,831,204,800 B, 182.0 s**), an n600
pose leg on it, and a carrier re-solve over all 600 pairs starting from the SHIPPED codes.

| leg | value |
|---|---:|
| per-pair sum of credits | −4.89711e−05 |
| **composed, re-verified (n600 mean shift)** | **−8.1529e−08 × 600 = −4.89174e−05** |
| **realized fraction** | **0.9989** |
| stale pose (candidate renders, shipped carrier) | 3.53511e−04 = **84.0× base** |
| carrier coordinates changed | **926** over 126 pairs (set01: 791 over 108) |

The composed n600 resolved pose is **4.126007089534244e−06** against the base
4.207535785085938e−06. F-composition does not fire: the composition realized **99.89 %** of the
per-pair sum, so the per-pair estimator is neither optimistic nor pessimistic on this vehicle
(pd5 measured 1.0219, pd4 0.9992).

## 10. THE BYTE-CLOSE — the three legs on the candidate's OWN shipped bytes

The archives in §8 are the PRICER's: they carry move 52's own carrier. A real candidate must carry
the RE-SOLVED one, and that is what `ddm_sj1_joint_admission close` builds from a staged body.

**A DEFECT this arm hit, and the gate that caught it.** `stage-tail` splices the new stream as
`tail[:RESIDUAL_COMPACT_BYTES] + stream`. On an RLC1-era pointer the tail is
`prefix(96) + RIDER(64) + stream`, so handing it the bare `stream_0.rc64` silently DROPPED the
64-byte RLC1 rider: the staged archive was 64 B smaller than a legal one and the receiver fell
straight through to the TC1M rider path (`invalid or truncated TC1M rider`). The first closed
candidate therefore read 179,223 B — **64 bytes better than the truth, and undecodable.** The
parse-back refused it, which is the gate doing its job; the correct input is the tail SUFFIX
(`rider + stream`), and every number below is measured on archives rebuilt that way.
**LAW: on an RLC1 pointer, `stage-tail --stream` wants the tail suffix, not the raw rc64 stream;
a byte claim taken from the bare stream is 64 B optimistic and does not parse.**

**MEASURED: the re-solved carrier costs +14 bytes** (18,470 → 18,484) for 791 moved coordinates
over 110 pairs. The `close` identity control passed and every other frame-1 section is
byte-identical.

### The SIZE ladder, on real CLOSED archives

| candidate | pairs / tokens | **exact closed archive** | rate | seg | pose | **net** | **bars** |
|---|---:|---|---:|---:|---:|---:|---:|
| set 00 | 128 / 147 | 179,305 B `4a301a0c…` (−27 B) | −1.7978e−05 | −1.6117e−05 | −6.5051e−05 | −9.914644e−05 | 4.957 |
| **set 01 — THE CANDIDATE** | **110 / 129** | **179,287 B `18b7e729505deb88524c922c4579cc631fcf699a6962032ccc374e374d4be00f` (−45 B)** | **−2.9964e−05** | **−1.6117e−05** | **−6.1375e−05** | **−1.074558e−04** | **5.373** |
| pd6sub (the λ-sweep's own pick) | 106 / — | 179,286 B `a09134a7…` (−46 B) | −3.0630e−05 | −1.5269e−05 | −6.1205e−05 | −1.071037e−04 | 5.355 |

pd5's handoff said: converge the ledger, then price the SIZE ladder on real archives and take the
best. Doing exactly that here picks **set 01** — neither the largest set, nor the smallest archive,
nor the λ-sweep's own choice. The smallest archive (pd6sub, 179,286 B) is NOT the best score,
because it repairs one fewer SegNet cell. That is pd5's warning reproduced from the other side.

### The candidate's three legs

| leg | value | how |
|---|---:|---|
| rate | **−2.99637e−05** | **−45 B EXACT**, the candidate's own closed archive, re-solved carrier included |
| seg | **−1.61172e−05** | **19 cells REPAIRED** on the frozen argmax, carried to T4 by the same-instrument ratio 1.0006662543837985 |
| pose | **−6.13750e−05** | RESOLVED 4.130731e−06 against the base 4.21e−06, from the n600 composed re-solve at 0.9989 of the per-pair sum |
| **S projected** | **0.13609481321602535** | 100·0.00010287882769408085 + √(10·4.130731e−06) + 25·179,287/37,545,489 |
| **net vs move 52** | **−1.074558e−04** | **5.373 bars** · **4.64× the 34.8 B container-break sd** |

**All three legs are negative.** The candidate is smaller, repairs SegNet cells, and lowers pose. No
pass on this object has had that before: every prior move bought pose with bytes and paid seg on the
side.

**Twins.** The candidate was closed twice — from the FIRST and the SECOND in-process encoder's
stream, staged and closed in separate directories — and both produce
**179,287 B sha `18b7e729505deb88524c922c4579cc631fcf699a6962032ccc374e374d4be00f`**.

## 11. THE COLD PARSE-BACK — and the control that changed the verdict

The candidate tree was decoded cold on CPU at the contest thread count (4), through the tree's
own `inflate.py` and `runtime.f26_inflate`, 1,471.1 s (token decode 572.0 s, neural render
845.6 s), raw 3,662,409,600 B sha `baca511f93f8f6b53c3f70b189539ecaed4c9111aaeb635b1911e476c05589fc`.

| control | outcome |
|---|---|
| the decoded TOKEN PLANE is the admitted field | **PASSES** — 129 tokens over 110 pairs, plane sha `e4d1a999bfe9eab59fc451100c65fd43a94c2b0ea302c80ff5ae032ae4594b92`, **zero cells differing** |
| **d_seg on the DECODE** (`step0 --raw`, the same jg1 instrument move 52's leg was measured with) | **12,129 flips against move 52's 12,147 — 18 cells repaired**, where the per-pair census predicted 19 |
| **d_pose on the DECODE** (`up2.measure_pose` batch 8 on the candidate's own `0.raw` and its own carrier) | **2.0241378451781925e−05 — 4.90× the overlay's 4.130731e−06, and 4.81× move 52's own base** |

**That third row is the pass's most important measurement, and it is a REFUTATION of the
candidate.** pd5 named this gap and could not reach it ("the seg leg was never measured on a
decode, because no candidate was decoded"); pd4's F9 was "not reached" twice. This arm reached it,
and the answer is that the composed-overlay pose leg is NOT the shipped pose leg.

### Where the disagreement lives, MEASURED

| pairs | decode vs admission |
|---|---|
| all 490 UNKEPT pairs | **identical to all digits** (Σ|Δ| = 0.0) |
| 101 of the 110 kept pairs | **identical to all digits** |
| **9 kept pairs** (82, 154, 167, 238, 268, 305, 487, 508, 532) | decode pose is **10³–10⁴× worse**; Σ|Δ| = 9.666e−03 |

And the cause is not the carrier: those nine pairs' re-solved codes move by 1–15 units, the same
range as the pairs that reproduce, and nothing is near the int12 rail. It is the RENDER. Comparing
the decoded odd frame against the overlay's own odd frame:

| pair | odd-frame max |Δ| decode vs overlay |
|---|---:|
| 238 / 487 / 268 (bad) | **114 / 154 / 117 grey levels** |
| 397 / 398 / 14 (good) | **0 / 0 / 0** |

**LAW, measured here for the first time on this object: `render-edits` reproduces the receiver's
own odd frame EXACTLY for most edited pairs and NOT for all of them, so a pose credit measured on
the overlay is only valid for the pairs whose overlay render the decode reproduces. The overlay is
a PROPOSAL; the decode is the charge.** Every prior pass on this object bought its pose leg from
the overlay and never checked. This one checked, and 9 pairs of 110 were bought on a render the
shipped bytes do not produce.

### What that does to the candidate

On its own decoded bytes the 110-pair archive scores
100·0.00010288731044702396 + √(10·2.0241378451781925e−05) + 25·179,287/37,545,489 — the pose leg
alone is **+7.739e−03 S**. The 110-pair candidate **does NOT clear the bar**; it is far worse than
move 52. The −1.066e−04 net of §10 was real arithmetic on an unreal pose leg.

## 12. THE SUCCESSOR — drop the nine, and the decode reproduces the admission EXACTLY

The fix the refutation implies is not subtle: drop the nine pairs whose render the decode does not
reproduce, keep the 101 it does, and measure again on the new bytes' own decode.

| | 110-pair candidate | **101-pair candidate** |
|---|---|---|
| pairs / tokens | 110 / 129 | **101 / 118** |
| exact closed archive | 179,287 B `18b7e729…` (−45 B) | **179,286 B `aab908d3b32c65582b7de1a3855b67a4f6fdb9373b0dc81e3049ee8164ec3957` (−46 B)** |
| encoder twins (2 processes) | `b884a8bf…` both | **`b3d5c52988f2a182…` both** |
| cold parse-back | 1,471.1 s | **936.7 s**, raw sha `8a14f55a6a8b141501836f511f4222dde75dca21714d923ad865b5f4757ef66b` |
| decoded plane == admitted field | yes (129 tokens / 110 pairs) | **yes (118 tokens / 101 pairs)**, plane sha `4cb0b147bdae8ce0618938453bb6ccc5b7b5a4927d6ad4e1b0bd9dd9ff46119c` |
| d_seg on the decode | 12,129 flips (18 repaired) | **12,128 flips (19 repaired)** |
| **d_pose on the decode** | **2.0241e−05 — 4.90× the admission** | **4.13644120761059e−06 — IDENTICAL TO ALL DIGITS to the admission; max per-pair |Δ| 0.0** |

**That last cell is the pass's proof.** Removing exactly the nine pairs the render control named
takes the decode/admission pose difference from 1.611e−05 to **0.0 — not "close", not "within
noise", identical in every digit over all 600 pairs.** The law in §11 is not a hedge; it is a
diagnosis that predicted its own cure, and the cure is measured.

### The three legs of the candidate, EVERY ONE on the shipped bytes' own decode

| leg | value | how |
|---|---:|---|
| rate | **−3.062951e−05** | **−46 B EXACT** — the closed archive, re-solved carrier included |
| seg | **−1.611723e−05** | **19 cells REPAIRED**, MEASURED by `step0 --raw` on the candidate's own `0.raw` through the same jg1 instrument move 52's leg was measured with; carried to T4 by the same-instrument ratio 1.0006662543837985 |
| pose | **−5.693419e−05** | **4.1364412076105896e−06** MEASURED by `up2.measure_pose` batch 8 on the candidate's own `0.raw` with the candidate's own carrier |
| **S projected** | **0.13609858812864767** | 100·0.00010287882769408085 + √(10·4.1364412076105896e−06) + 25·179,286/37,545,489 |
| **net vs move 52** | **−1.036809e−04** | **5.184 bars** · **4.47× the 34.8 B container-break sd** |

**All three legs are negative and all three are measured on the decode.** The candidate is 46 bytes
SMALLER than move 52, repairs 19 SegNet cells, and lowers pose by 1.7 %. It is not a projection off
an overlay: it is what the bytes do.



<!-- # FORMALIZATION_PENDING: a measurement pass; the score arithmetic used throughout is the
registered S = 100*d_seg + sqrt(10*d_pose) + 25*B/37,545,489. -->
