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

<!-- SECTIONS 4 ONWARD ARE WRITTEN WHEN THEIR MEASUREMENTS LAND -->
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

