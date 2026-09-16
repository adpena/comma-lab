# ddm_pd4 — pose-directed token pre-distortion, PASS 4 on the move-52 field: the PRICE lever

Tokens: `[no-triality] [p0-ledger-ok]`

Lane `ddm_pd4_pose_directed_pass4_price_lever_20260913`. Base: **move 52**,
S 0.13620226906030858 @ 179,332 B, archive sha
`ae59c5109597968409c4b581337b8e23b5b3c4262fd3710c315cc83913b3b20e`. Axis
`[macOS-CPU advisory, frozen CPU-torch PoseNet + SegNet, DALI GT lineage, n600]` for the
distortion legs; bytes EXACT through the shipped RLC1 coder and the shipped container.
**No score is claimed. Only `upstream/evaluate.py` on shipped bytes is a score, and MAIN
fires.** No Modal, no authorization, no fire, no completion, no packet.

Pose solver, stated once: every per-pair credit here comes from
`ddm_jg5_pose_resolve_on_edited_renders.refine_pair` (pd1's, pd2's, pd3's and pass 8's same
object at these budgets), driven at PoseNet batch 1 on the moved render; every n600 number
comes from `ddm_up2_shipping_pose_solve.measure_pose` at batch 8, on move 52's own cold
parse-back for the base and on the CANDIDATE's own cold parse-back for the candidate.

Provenance pins: HEAD at charter `356ee9d17`, producer commit `5c85fba14`; charter
`4bd0b39b1ff4fb9a…`; pd3 memo `89cefe7604100648…`; move-52 packet `bc2d8f3d7901ea90…`;
pass-8 memo (the 8.93 bits/token clustered price) `.omx/research/ddm_sj1_t4_token_predistortion_pass8_20260912.md`.

---

## 1. The binding, and the gate that pd3 relied on which is only 96 % effective

`ddm_pp1_pose_actuation` and `ddm_sj1_multipass_token_predistortion` pin the live pointer at
**MOVE 49** in module globals evaluated at import time; `ddm_pd1_pose_directed` pins move 49
in its own constants; `ddm_pd2_pose_directed_pass2` pins **MOVE 50**; `ddm_pd3_pose_directed_pass3`
pins **MOVE 51**. Move 52 is a real pointer move, so **five** shared modules are stale for
this arm and the failure would be silent. `ddm_pd4_pose_directed_pass4.bind_move52` re-points
all five PROCESS-LOCALLY, fails closed, and re-asserts through the modules that will be used.

| checked before binding | result |
|---|---|
| move-52 archive bytes / sha256 | 179,332 B / `ae59c510…3b20e`, MEASURED |
| move-52 argmax sha256 | `622b284eada5023d…`, MEASURED |
| move-52 token field npz sha256 | `0577bbb88e7218c7…`, MEASURED |
| move-52 decode bytes / sha256 | 3,662,409,600 B / `ccb89e3e…eced`, MEASURED (`--verify-raw`) |
| move 52's three components recompute its packet S | 0.13620226906030858 exactly |
| re-asserted THROUGH `pp1.assert_pointer_identity`, `sj1.assert_carrier_is_pointer`, and `pd1`/`pd2`/`pd3` `seg_s_per_cell()` | all report move 52 |

**And one gate that is weaker than pd3 believed.** pd3 wrote that the base-band gate
"refuses any row whose batch-1 base is not move 51's — which is how *re-search the pairs the
pointer edited from their NEW renders* is enforced structurally rather than by hand."
MEASURED here against move 52: of the **26** pairs move 52 itself edited, **25 are refused by
the band and pair 569 is not** — its predecessor rows sit **9.311e-10** from the new base,
well inside the 2.2642595483754955e-08 band, because that pair's own d_pose barely moved.
The band is therefore a **96.2 %-effective proxy** for "this render did not move", not a proof
of it. This arm excludes the 26 explicitly AND keeps the band as the second gate. The gap
is small in this instance — one pair — but the failure mode it opens is the silent one: a
predecessor's row measured on a render that no longer exists, admitted as if it were current.

---

## 2. The base, MEASURED on this arm's own decode of the bytes the T4 row scored

This arm did not inherit pd3's numbers: it copied move 52's runtime tree (51 files, every one
sha-verified against the source), parsed the archive back cold through that copy, and measured
the pose on its own `0.raw`.

| quantity | measured |
|---|---:|
| cold parse-back wall | **1,148.57 s** |
| raw bytes / sha256 | 3,662,409,600 / `ccb89e3e73bf61ac…` — **equal to pd3's own retained decode** |
| decoded token plane sha256 | `117951bbd6800949…`, `decoded_field_matches_admitted: true` |
| n600 mean d_pose on that decode | **4.207535785085938e-06** |
| ratio vs the T4 print 4.21e-06 | **0.9994146757923842** |
| median / max | 2.9126860401929944e-07 / 2.1283759625946122e-04 |
| top-12 share | 56.69 % (the same twelve pp1 floor pairs) |
| instrument d_seg | 0.00010297139485677083 (**12,147** flipped cells, DALI leg) |

**The control.** Against pd3's own `pose/pose_on_decode.npy` — the vector pd3 measured on the
same shipped bytes — the maximum absolute per-pair difference is **0.000e+00** over all 600
pairs. Two arms, two independent parse-backs, one instrument, identical to the last bit.

Receipts: `parseback/PARSEBACK_RESULT.json`, `base/POSE_BASE_MOVE52.json`,
`base/pose_base_move52.npy`, `base/codes_move52.npy`.

---

## 3. The economics at move 52 — they EXPIRE, so they are re-derived

| term | value |
|---|---:|
| S per archive byte | 6.658589531221714e-07 |
| S per flipped SegNet cell (T4-carried through the instrument ratio) | 8.482752943113527e-07 |
| S per unit of **one pair's** d_pose | **1.2847092313591597** |

Five MEASURED real-encode prices exist for this edit shape on this coder — pd1's 41-token
field at 16.98, pd2's 165-token field at 16.630 and its 40-token subset at 15.400, pd3's
271-token field at 16.443 and its 26-token subset at **12.923** — and pass 8 measured **8.93**
bits/token on a CLUSTERED 163-token/96-pair seg-repair field. None is a law.

**The price ladder, computed on the carried pool BEFORE any pass-4 row existed** (140 pairs
with a paying single-token proposal, 574 paying proposals, the 26 move-52 pairs and the 12
floor pairs excluded):

| price, bits/token | pairs able to pay with their best-benefit proposal | modelled net ΔS | bars of the −2e−05 bar |
|---:|---:|---:|---:|
| 16.443 (pd3's full field) | 7 | −1.770e−06 | 0.09 |
| **12.923 (pd3's shipped subset)** | **19** | **−5.577e−06** | **0.28** |
| 10.000 | 33 | −1.156e−05 | 0.58 |
| **8.930 (pass 8's clustered field)** | **43** | **−1.485e−05** | **0.74** |
| 7.000 | 64 | −2.328e−05 | 1.16 |

That table is the charter's thesis made arithmetic: on the ALREADY-SEARCHED pool, moving the
price from 12.923 to 8.93 bits/token multiplies the payable pairs by **2.26×** and the
modelled net by **2.66×** — and even at 8.93 the carried pool alone reaches only **0.74** of
the bar. The pass-4 yield therefore has to come from three places at once: the price, the 26
pairs move 52 re-rendered, and the clustered family.

Receipts: `PREREGISTRATION.json` (written before any pass-4 search row, price sheet or
admission existed, with ten pre-registered falsifiers), `SEARCH_PLAN.json`.

---

## 4. The smoke: K_refine, the pair budget, and the PRICE method, all declared from measurement

Eight pairs drawn at ranks 0/20/40/60/80/100/120/139 of the expected-credit order, run as **8
concurrent cluster-search shards at 2 threads each, CONCURRENT with the smoke price encode and
the 26-pair re-search** — the fleet condition this arm actually ran in, not an idle machine.

| shard | pair | screened | refined | wall s |
|---:|---:|---:|---:|---:|
| 0 | 85 | 16 | 12 | 491.0 |
| 1 | 418 | 8 | 4 | 132.9 |
| 2 | 446 | 8 | 4 | 148.4 |
| 3 | 284 | 8 | 0 | 19.6 |
| 4 | 428 | 5 | 2 | 69.0 |
| 5 | 262 | 8 | 7 | 255.3 |
| 6 | 385 | 8 | 5 | 144.8 |
| 7 | 373 | 8 | 8 | 395.5 |

| quantity | measured |
|---|---:|
| wall per pair | 19.6 – 491.0 s (mean 207.1, median 146.6) |
| seconds per refine | mean **35.0**, median 35.5 — pd3 measured **37.6** for its single-token walk |
| fleet throughput at 8 shards, mean-wall basis | 2.318 pairs/min |
| projected: the 132-pair cluster population | **57 min** (mean basis); 135 min on the slowest-shard basis |

**K_refine = 12 is DECLARED**, and the reason is that the depth is NOT this arm's declared
delta: pd3 MEASURED K=12 strictly better than K=8 on 24 of 125 shared pairs and worse on none,
so keeping 12 holds the depth fixed while the PRICE moves. On clusters the cap binds only on
the richest pair (85, 16 screened / 12 refined); the other seven are limited by the **seg
screen**, not by K — 5 of 8 pairs produced ≤ 5 refinable proposals and pair 284 produced none.
That is itself a measurement: a second token next to an already-moved one usually costs seg
cells, and `max-cells 2` refuses it.

### The price half of the smoke — and it is the finding of the pass

The rail: `ddm_sj1_rlc1_price.encode` drives the SHIPPED receiver's own decode loop
(`rx.decode_production_tokens` on move 52's runtime) with the true symbols injected at
`NativeDecoder.decode`, feeds the receiver's own probability rows to twin arithmetic encoders,
and records `per_frame_bits[f] = Σ −log2 p_model(symbol)` over the frame's coded positions. It
is not a first-order token price: it carries the adaptive model's full response WITHIN the
frame. What it omits is the spill into later frames, and that omission is MEASURED, not
assumed (below).

**The control encode passed first.** Two independent processes, each with its own natively
compiled rc64 backend, re-encoded move 52's OWN field and packed an archive with sha
`ae59c5109597968409c4b581337b8e23b5b3c4262fd3710c315cc83913b3b20e` at **179,332 B** — byte
identical to move 52's own archive, `control_identity_passed: true`, stream 119,097 B matching
the tail's own stream sha `1ba05d88f800f201…`. **Falsifier F1 does not fire.**

**Then eight single-token proposals — the best-benefit proposal on each of the eight smoke
pairs — were priced by a real encode of a field that changes only those eight pairs:**

| pair | REAL Δbits (frame-local) | its own break-even bits | pays? |
|---:|---:|---:|:--|
| 373 | **7.117** | 0.1 | no |
| 385 | 14.771 | 1.6 | no |
| 284 | 16.769 | 7.4 | no |
| 446 | 17.235 | 9.0 | no |
| 428 | 18.073 | 4.9 | no |
| 262 | 18.182 | 3.4 | no |
| 418 | 19.224 | 12.4 | no |
| 85 | **28.947** | 26.8 | no |

| aggregate | measured |
|---|---:|
| pooled frame-local price over the eight | **17.540 bits/token** |
| whole-field ideal delta | 148.20 bits = 18.525 bits/token |
| **realized archive delta** | **+18 B for 8 tokens = 18.000 bits/token** |
| realized ÷ first-order ideal | 0.972 |

**Not one of the eight pays its own break-even, and the pair with the largest benefit in the
entire carried pool — 85, worth 26.8 bits — costs 28.947.** The move-52 field prices a
single isolated token at **17.5–18.5 bits**, ABOVE pd1's 16.98, pd2's 16.630/15.400 and pd3's
16.443/12.923, and at **2.0×** pass 8's 8.93 clustered price. Four passes of pre-distortion
have spent the cheap tokens; what is left is expensive.

**The spill, measured twice on two different objects.** On pd3's own 271-token field
(its retained `bits_rlc1_control.npy` and `bits_rlc1_pd3full.npy`) the signed delta is 4,449.05
bits, of which **97.21 %** lands on the edited frames. On this 8-token field the signed spill
onto the 592 unedited frames is **7.88 of 148.20 bits = 5.3 %**, with an absolute spill of
340.93 bits and a **maximum per-frame |Δ| of 2.86 bits**. So a frame-local price is a
95–97 %-complete REAL price, its worst-case noise is under 3 bits, and the measured spread
BETWEEN proposals is 7.1 → 28.9 bits — an order of magnitude above that floor.
**Falsifier F3 does not fire: the per-proposal price is resolvable.**

**The cost of the rail, which is why sheets exist.** One 600-frame encode took **~1,700 s**
under this fleet condition and prices at most one proposal per pair — but it prices ALL of
them in that single pass. Pricing the 496 proposals this arm carries one encode at a time
would be 496 × 1,700 s ≈ **234 hours**; six sheets price the same 496 in six concurrent
encodes, about **280× cheaper**, and the sheet construction keeps the edit DENSITY identical
across sheets so the neighbour spill is common to every sheet and cancels in the per-pair
comparison.

---

## 5. The two searches

**(a) The 26 pairs move 52 itself edited, re-searched from their NEW renders** — single token,
K_refine 12, pd1's reference form otherwise unchanged (cells 16, interior radius 3, max-cells
2, deltas −1/+1). All 26 walked to completion: **265 realized rows**. Only **4 of 26** carry a
break-even above the measured ~17.5-bit price — 591 at **65.0** bits, 324 at **42.8**, 59 at
**31.4**, 465 at **24.6** — but those four are the richest assets in the entire pass, and all
four come from re-rendered pairs. pd3's finding that a re-render re-opens credit holds a
fourth time.

**(b) The clustered family** — a second token in the 8-neighbourhood of the pair's best
single-token move, realized JOINTLY (one render, one frozen-argmax, one carrier re-solve),
its credit measured against the pair's OWN base and never summed from two rows. **PREFIX
STOP** per the pre-registered `STOP_RULE.json`: **63 of the 140 anchor pairs (45 %)** were
walked before the cluster shards were stopped by PID to free the cores for the six price
encodes. Because the walk order is round-robin over 8 shards in descending benefit, that 45 %
is a near-uniform prefix of the GLOBAL ranking, not the top of one shard — but every cluster
verdict below is scoped to that prefix.

---

## 6. THE PRICE, measured per PROPOSAL — the charter's rung

Six sheets, 156 pairs, **496 proposals**, one real 600-frame encode each, twins byte-identical
inside every encode. Sheet archives came back at 179,668–179,686 B (+336…+354 B over move 52).

| family | proposals | tokens | total Δbits | **bits/token pooled** | median | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| single-token | 374 | 374 | 5,923.73 | **15.839** | 15.418 | **−17.173** | 33.443 |
| **clustered (2 tokens)** | 122 | 244 | 3,368.14 | **13.804** | **11.416** | 4.502 | 29.453 |

**Falsifier F5 does NOT fire: clustered edits are measurably cheaper per token** — 13.804
against 15.839 pooled (−12.9 %), 11.416 against 15.418 by median (−26.0 %). It is not pass 8's
8.93 — that was a different field, a different edit shape and a different pointer — but the
DIRECTION pass 8 measured reproduces on this vehicle, on 122 jointly-realized proposals.

**And the price resolves.** Three independent measurements of its noise:

| control | measured |
|---|---:|
| the SAME proposal priced in ≥2 sheets that differ at other pairs (116 proposals) | spread median **2.518** bits, mean 2.507, max 5.587 |
| the same 8 proposals priced in an 8-token SPARSE field vs a 156-pair sheet | max \|Δ\| **2.643** bits, median 1.166, mean +0.196 |
| spill onto UNEDITED frames | 5.3 % of the signed total here, 2.79 % on pd3's 271-token field |

against a **within-pair price range of median 6.73 bits (max 45.30) over 110 pairs**. The
signal is 2.7× the noise at the median. **F3 does not fire.**

### What ranking by credit per REAL bit actually changed

| ranking | pairs that pay | tokens | Δbits | **bits/token** | modelled net ΔS |
|---|---:|---:|---:|---:|---:|
| by benefit alone (pd3's rule) | 13 | 20 | 231.98 | 11.599 | −1.1073e−05 |
| **by resolved-pose credit per REAL bit** | **18** | **24** | 211.36 | **8.807** | **−1.4682e−05** |

**The rankings disagree on 31 of 156 pairs.** The per-bit rule finds **5 more payable pairs**,
spends **20.6 fewer bits**, and improves the modelled net by **+32.6 %** — and it drops the
admitted subset's price to **8.807 bits/token**, below pd3's shipped **12.923** and essentially
at pass 8's **8.93**. **Falsifier F7 does NOT fire: the price lever works, and it is the first
thing in this family that has.** Six of the 18 payable pairs are CLUSTERS; three proposals
(pairs 409, 476, 7) carry a NEGATIVE price — they shorten the stream while gaining pose.

---

## 7. The composition, re-verified on one object, and the admission

The per-pair search realizes each edit alone, so the composed object was RE-MEASURED: a
600-pair overlay rendered from the 40-pair carried field (1,831,204,800 B, 200.4 s), an n600
pose leg on it, and a carrier re-solve over all 600 pairs that starts from the **SHIPPED**
codes, never from the search's own.

| leg | per-pair sum | composed, re-verified | realized fraction |
|---|---:|---:|---:|
| pose, all 40 carried | −4.5904610e−05 | **−4.5866457e−05** | **0.999169** |

**Falsifier F6 does not fire.** The two null controls:

* the STALE pose — candidate renders, shipped carrier — is **2.0403e−04**, **48.5×** the base.
  The re-solve recovers all of it: composed resolved **4.1304e−06** against base 4.2075e−06.
* of the **560 unedited pairs, 559 are bit-identical to their base** (|resolved − base| =
  0.000e+00) and the stale leg is bit-identical on all 560. **One** unedited pair is not: its
  carrier re-solve found **−4.063e−07** of d_pose on a render that did not move. That is a
  real, small, SEPARATE lever — a carrier-only re-solve worth about −5.2e−07 S against a
  splice cost of roughly 0.19 B (≈ +1.3e−07 S) at pd3's measured splice rate — and this arm
  does NOT take it: the admission gates on `edited`, so the pair keeps its base leg and its
  shipped codes. It is handed to the next arm, not banked here.

### The three-leg Lagrange admission, on the RESOLVED pose

The sweep keeps **18 of the 40 carried pairs** at λ = 1.2735, with the rate leg taken from the
MEASURED per-frame bit ledgers (`rate_leg_is_modelled_not_measured: false`) rather than
apportioned. Admitted pairs: 7, 14, 48, 59, 82, 168, 170, 217, 221, 228, 265, 324, 407, 409,
465, 476, 499, 591 — **exactly the 18 the per-bit carry said would pay**, reached
independently by a sweep over the composed object.

| quantity | modelled |
|---|---:|
| pairs kept | 18 (24 tokens; 6 of them clustered pairs) |
| d_seg instrument | 0.00010297139 → 0.00010301378 (**+5 cells: the pose-directed edits COST seg here**) |
| d_pose | 4.2075358e−06 → **4.1601727e−06** |
| archive bytes | 179,332 → 179,358.42 (**+26.42 B**, from the spliced per-sheet ledgers) |
| **net vs move 52** | **−1.6678e−05** | 

**That is 0.83 bars of the −2e−05 admit bar.** The rate leg above is a splice of per-sheet
frame-local measurements and is NOT the charge; the admitted subset's own full real encode is,
and it is measured next.

---

## 8. The rate leg by REAL encode — the authority, and it fires the bar

`ddm_sj1_rlc1_price` re-encoded the 18-pair / 24-token ADMITTED field through the shipped
RLC1 loop, twice, in two independent processes each with its own natively compiled rc64
backend:

| field | pairs / tokens | exact archive | Δ vs move 52 | bits/changed token | realized ÷ first-order ideal | twins |
|---|---:|---:|---:|---:|---:|---|
| control (move 52's own field) | 0 / 0 | **179,332 B** sha `ae59c510…3b20e` | 0 | — | — | **byte-identical, 2 processes** |
| the six price sheets | 156 / 176–184 | 179,668 – 179,686 B | +336 … +354 B | **14.957 – 15.514** | — | byte-identical within each |
| the 8-token SPARSE smoke field | 8 / 8 | 179,350 B | +18 B | 18.000 | 0.972 | byte-identical, 1 process |
| **the ADMITTED subset** | **18 / 24** | **179,368 B** sha `d4b52ebc91c05d881cd96ee6aaa081a5f0f46be73697c387fe2a5a77f8cca5ba` | **+36 B** | **12.000** | 0.98797 | **byte-identical, 2 processes** |

### The three legs on those exact bytes

| leg | value | how |
|---|---:|---|
| rate | **+2.39709e−05** | **+36 B EXACT**, the subset's own real encode, twins |
| seg | **+4.24138e−06** | **+5 flipped cells** — the admitted edits COST seg on this field |
| pose | **−3.66120e−05** | RESOLVED 4.1601727e−06 against the base 4.2075358e−06, re-verified n600 on the composed object at 0.999169 of the per-pair sum |
| **S projected** | **0.13619197013064016** | 100·0.00010308241376471558 + √(10·4.1601727e−06) + 25·179,368/37,545,489 |
| **net vs move 52** | **−1.029893e−05** | **0.515 bars** · **0.444× the 34.8 B container-break sd** |

**Falsifier F8 FIRES. The pass does not clear the −2e−05 bar, and it is not close: the margin
is smaller than the campaign's standing container-break lottery.** No candidate was built, no
archive staged, no parse-back run, nothing sealed. Building past this would have been spending
on a row MAIN cannot fire — pass 8's precedent, applied.

### The ledger's own error, measured

The spliced per-sheet ledger predicted **211.363 bits** for these 18 pairs; the real encode
says **288 bits**. The ledger **under-charged by 36.3 %** — far more than pd3's 2.57 B
(+0.0014 %) or sj1's +19.6 B (+0.0108 %). The cause is not the frame-local rail, which the
controls show is 95–97 % complete with a ≈2.5-bit noise floor. **It is SELECTION on a noisy
estimator**: the per-bit rule takes the argmin of ~3 prices per pair, and the argmin of noisy
draws is biased low — about the noise floor per pair, ≈ 2.5 × 18 ≈ 45 bits, plus the spill.
Three of the 18 winners carried a NEGATIVE frame-local price; none of them can actually pay
the container back. **A successor that ranks by a measured per-proposal price must debias the
selection, or price the chosen subset before it believes the subset's number.**

Re-running the sweep with the ledger rescaled by the measured 1.3626 — a single-point
calibration, labelled as such — moves the optimum to **11 pairs** and **−1.1810e−05 (0.59
bars)**. The verdict does not depend on which of the two rate legs is used.

---

## 9. The yield, and what price would have cleared the bar

The best proposal per pair over all 156 priced pairs, as a fraction of that pair's own d_pose:

| band | pairs | admitted |
|---|---:|---:|
| −100 % … −50 % | **74** | **8** |
| −50 % … −25 % | 31 | 7 |
| −25 % … −10 % | 17 | 2 |
| −10 % … −5 % | 10 | 0 |
| −5 % … −2 % | 11 | 0 |
| −2 % … 0 % | 10 | 1 |
| ≥ 0 % | 3 | 0 |

Median carried **−45.78 %**, median admitted **−44.82 %**, best **−99.25 %**. The seg cost of
each pair's best proposal: **−2** on 1 pair, **−1** on 11, **0** on 114, **+1** on 20, **+2**
on 10 — 126 of 156 cost zero or negative cells, and 13 of the 18 admitted do.

**The pass is not short of credit. It is short of PRICE.** The same pool, re-run against the
ladder with the tokens each proposal actually moves:

| price, bits/changed token | pairs that pay | modelled net ΔS | bars |
|---:|---:|---:|---:|
| 16.443 (pd3's full field) | 10 | −8.901e−06 | 0.45 |
| 12.923 (pd3's shipped subset) | 21 | −1.455e−05 | 0.73 |
| **12.000 (THIS subset, MEASURED)** | **25** | **−1.674e−05** | **0.84** |
| **10.891 (DERIVED: the bar)** | **32** | **−2.000e−05** | **1.00** |
| 8.930 (pass 8's clustered field) | 49 | −2.782e−05 | 1.39 |
| 7.000 | 67 | −3.914e−05 | 1.96 |

**DERIVED by bisection: this pool clears −2e−05 at 10.891 bits per changed token. The measured
price is 12.000. The pass misses by 1.109 bits per token — 9.2 %.**

---

## 10. The verdict on the price lever

**The price lever is REAL, it is MEASURED, and on the move-52 field it is 9.2 % short.**

| falsifier | outcome |
|---|---|
| F1 the pricer reproduces move 52's own archive | **does not fire** — 179,332 B sha `ae59c510…`, twins, `control_identity_passed` |
| F2 frame-local pricing is ≥ 90 % of the signed delta | **does not fire** — 97.21 % on pd3's 271-token field, 94.7 % here |
| F3 the per-proposal price resolves above its noise | **does not fire** — noise median 2.518 bits (116 repeated proposals), sparse-vs-dense max 2.643 bits, against a within-pair range of median 6.73 bits |
| F4 the per-bit ranking changes the admitted set | **does not fire** — the two rankings disagree on 31 of 156 pairs; the per-bit rule finds 5 more payable pairs |
| F5 clustered edits are cheaper per token | **does not fire** — 13.804 pooled / 11.416 median against 15.839 / 15.418 for isolated tokens |
| F6 composition is realized, not summed | **does not fire** — realized fraction 0.999169 over 40 pairs; 559 of 560 unedited pairs bit-identical |
| F7 the subset beats pd3's 12.923 bits/token | **does not fire, but only just** — **12.000** measured, a 7.1 % improvement, not the 8.807 the frame-local ranking predicted |
| **F8 the admitted set clears −2e−05** | **FIRES** — **−1.029893e−05, 0.515 bars** |
| F9 seg on the shipped bytes equals the admission's | **not reached** — no candidate was built |
| F10 the base is move 52's own decode | **does not fire** — this arm's own parse-back reproduces `ccb89e3e…`, and its pose reproduces pd3's at max abs difference **0.000e+00** |

**What this pass settles.**

1. **The price on this field is 12–18 bits per changed token, not 8.93.** An isolated token
   costs 15.839 pooled (374 proposals); the best-selected 24-token subset costs **12.000** by
   its own real encode. Four passes of pre-distortion have spent the cheap tokens. pass 8's
   8.93 was measured on the move-49 field with a seg-repair edit shape, and it does not
   transfer — the number is a HYPOTHESIS on any other field until re-measured, which is
   exactly what this arm did.
2. **Ranking by credit per REAL bit is worth +32.6 % of modelled net and 5 extra payable
   pairs**, and it is the first thing in this family that lowered the price rather than
   chasing more credit. It is a keeper.
3. **Clustering is cheaper per token — measured, on this vehicle, at −12.9 % pooled and
   −26.0 % by median** — and 6 of the 18 admitted pairs are clusters. But the second token
   usually costs seg cells: of 8 smoke pairs, 5 produced ≤ 5 refinable proposals and one
   produced none, because `max-cells 2` refuses the rest.
4. **Selection on a measured price needs debiasing.** The argmin of ~3 noisy prices per pair
   under-charged the chosen subset by **36.3 %**. That is the single most useful thing a
   successor can take from this pass, and it is a new failure mode for this campaign: not a
   wrong estimator, a wrongly-SELECTED one.
5. **The gap is 1.109 bits per token.** At 10.891 the same pool clears the bar with 32 pairs.
   Every lever that lowers the price by ~10 % — a larger cluster (3 tokens), edits packed into
   one coder context, a token whose neighbours already moved, or a re-ordering of the stream —
   is now worth more than any lever that finds more pose credit.

**verdict_scope: FORMULATION, on the move-52 field.** What is measured is pose-directed
single-token and 8-neighbourhood two-token pre-distortion on move 52's field, proposals ranked
by resolved-pose credit per REAL measured bit, admitted on the resolved pose against a real
subset encode. The cluster family was walked to **45 % of its 140-pair population** under a
pre-registered prefix stop, so its verdict is scoped to that prefix. Three-token clusters, a
re-ordered stream, and a price-first (rather than credit-first) proposal generator are
untouched.

---

## 11. What this does NOT claim

1. **No score of any kind.** Every S here is a PROJECTION on measured legs. Only
   `upstream/evaluate.py` on the shipped bytes is a score, and MAIN fires. This arm ran no
   Modal call, wrote no authorization, no completion and no packet.
2. **No candidate, no archive, no seal.** Nothing was built past the admission, because the
   admission does not clear the bar. `179,368 B sha d4b52ebc…` is the PRICER's archive — the
   object that measured the rate leg — not a staged candidate: it carries move 52's own
   carrier, not a re-solved one, and no runtime tree was built around it.
3. **The pose numbers are `[macOS-CPU advisory]`**, measured on a frozen CPU-torch PoseNet
   against DALI-lineage GT. The instrument's base is 4.207535785085938e−06 against the T4
   print 4.21e−06 (ratio 0.999415).
4. **The seg leg is carried onto T4 by the same-instrument ratio** (1.0006662543837985), not
   measured there, and it was never measured on a decode because no candidate was decoded.
5. **12.000 bits/token is THIS subset at THIS edit shape on THIS field**, measured once, as
   15.839 is this pool's isolated-token price and 13.804 its clustered price. None is a law.
6. **The 10.891-bit threshold is DERIVED**, by bisection on the modelled ladder over the best
   proposal per pair — it assumes each pair's best proposal keeps its credit at a lower price,
   which is true by construction, and that the subset's real encode tracks the ladder, which
   this pass measured to be wrong by 36.3 % in the optimistic direction when the subset is
   chosen by argmin. Read it as the ORDER of the gap, not as a target.
7. **The cluster verdict is scoped to a 45 % prefix** of the 140-pair anchor population.
8. **pp1's twelve floor pairs stay excluded** on pp1's measurement. Nothing here reopens them.
9. **The one unedited pair whose carrier re-solve found −4.063e−07 was NOT banked** and is not
   counted anywhere in this pass's arithmetic.
10. **This arm fired nothing.** MAIN fires.

---

## 12. Custody

Store **`/Volumes/APDataStore/pact/ddm_pd4/`**. APDataStore rather than Vertigo because
Vertigo held **38 GiB** free — below its 40 GiB reserve. **The reserve was never lowered and
nothing was written there.** APDataStore free space MEASURED at each heavy step: 29 GiB at
start, 26 GiB before the searches, 25 GiB before the sheets, 24 GiB before the overlay,
**22 GiB at landing**.

**MEASURED: 5,929,016,619 B over 1,026 files, every one hashed — under the 6 GiB cap.**
`RETENTION_MANIFEST.json` carries bytes and sha256 for each, including the losers: every
realized search row and every screened proposal, not only the 18 that would have shipped.

| path | what |
|---|---|
| `base/move52_runtime/` · `parseback/0.raw` · `PARSEBACK_RESULT.json` | this arm's verified copy of move 52's 51-file tree and its own cold decode (3,662,409,600 B sha `ccb89e3e…`, 1,148.57 s) |
| `base/pose_base_move52.npy` · `codes_move52.npy` · `POSE_BASE_MOVE52.json` | the base pose (4.207535785085938e−06) and carrier |
| `search/` · `smoke/` | **every realized row and every screened proposal** — 265 re-search rows over all 26 pairs, 336 cluster rows over 63 pairs, winners and losers |
| `sheets/sheet_0{0..5}.npz` · `SHEETS.json` · `priced_rows.jsonl` · `PRICE_MERGE.json` | the six price sheets, their manifest and the 496 per-proposal REAL prices |
| `rlc1_price/` | the pricer's INPUTS, both control encodes, the smoke-price encode, all six sheet encodes, both subset encodes, the per-pair bit ledgers, `PRICE_pd4sub.json` |
| `assemble/` · `admission/` · `admission_calibrated/` | the carry ranking, the candidate field, the pass rows, the spliced and calibrated ledgers, both Lagrange sweeps and their traces |
| `pose/overlay_pd4/` · `pose_stale.npy` · `pose_resolved.npy` · `refine/` | the 600-pair overlay the pose leg scored, the two n600 vectors and the per-pair re-solve |
| `PREREGISTRATION.json` · `SEARCH_PLAN.json` · `SMOKE_TIMING.json` · `STOP_RULE.json` · `PREFIX_STOP.json` · `YIELD_HISTOGRAM.json` · `PRICE_THRESHOLD.json` · `LEDGER_CALIBRATION.json` | the pre-registration and every control receipt |
| `PIPELINE_PLAN.md` · `make_retention.py` · `make_census.py` · `make_seal_inputs.py` · `pose_on_decode.py` | the chain as exact commands and the producers the branch that did not fire would have used |

Producer: `experiments/ddm_pd4_pose_directed_pass4.py` (`bind | base | run | prereg | plan |
smoke-timing | cluster-search | sheets | price-merge | carry | histogram | assemble`),
ruff-clean, plus pd1's `ddm_pd1_pose_directed.py search`, sj1's `ddm_sj1_rlc1_price.py` and
`ddm_sj1_joint_admission.py`, all unchanged. Nothing under
`/Volumes/APDataStore/pact/ddm_pd1`, `ddm_pd2`, `ddm_pd3`, `/Volumes/VertigoDataTier/pact/…`
was written; every module that imports from a custody tree sets `sys.dont_write_bytecode = True`.

## 13. What this hands the next arm

1. **Stop paying for credit; start paying for price.** The pool has 74 pairs whose best
   proposal removes more than half that pair's own d_pose. At 12.000 bits/token 25 of 156 can
   pay; at 10.891 the pool clears the bar. **A 10 % price cut is worth more than any amount of
   further search on this field.**
2. **Three concrete price levers this arm did not run:** a THREE-token cluster (the 2-token
   discount measured −26 % by median, and the mechanism is context sharing, so it should
   compound); a proposal generator that ranks by PRICE first and only then checks credit
   (this arm generated by pose saliency and priced afterwards, so the cheap half of the token
   plane was never proposed); and packing admitted edits into adjacent coder contexts across
   PAIRS rather than within one.
3. **Debias the selection.** The argmin of ~3 noisy per-proposal prices under-charged by
   36.3 %. Either price each pair's chosen proposal a second time in a field where it is the
   only edit, or shrink the candidate set per pair, or apply a measured −2.5-bit-per-pair
   selection penalty before the sweep.
4. **The sheet rail is cheap and it is reusable.** Six 600-frame encodes priced 496 proposals
   — 13.7 s per priced proposal against 267 s when a field carries only eight. Any successor
   that wants per-proposal prices should build sheets, not fields.
5. **One unedited pair's carrier re-solve found −4.063e−07 with no token edit at all.** 559 of
   560 were bit-identical, so this is one pair, not a family — but it is free pose credit at
   roughly 0.19 B of splice, and nobody has swept the carrier alone on this pointer.

<!-- # FORMALIZATION_PENDING: a measurement and a closure verdict; no pointer row is produced, so there is no equations leg for tools/pointer_move_packet.py to write. The score arithmetic used throughout is the registered S = 100*d_seg + sqrt(10*d_pose) + 25*B/37,545,489. -->

Own-vehicle frontier (unchanged by this arm):
**S 0.13620226906030858 @ 179,332 B [contest-CUDA T4 n600]** (move 52).
