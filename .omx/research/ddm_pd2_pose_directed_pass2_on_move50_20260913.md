# ddm_pd2 — pose-directed token pre-distortion, PASS 2 on the move-50 field

Tokens: `[no-triality] [p0-ledger-ok]`

Lane `ddm_pd2_pose_directed_pass2_first_measurement_20260913`. Base: **move 50**,
S 0.13628342713679067 @ 179,195 B, archive sha
`1ea274f612a26183f31f6d439505d0289bd3751b4da9467343fc156203503cd7`. Axis
`[macOS-CPU advisory, frozen CPU-torch PoseNet + SegNet, DALI GT lineage, n600]` for the
distortion legs; bytes EXACT through the shipped RLC1 coder and the shipped container.
**No score is claimed. Only `upstream/evaluate.py` on shipped bytes is a score, and MAIN
fires.** No Modal, no authorization, no fire, no completion, no packet.

Pose solver, stated once: every resolved pose number here comes from
`ddm_jg5_pose_resolve_on_edited_renders.refine_pair` (pd1's, pass 8's and cr1's same object
at these budgets), driven at PoseNet batch 1 on the moved render; every n600 number comes
from `ddm_up2_shipping_pose_solve.measure_pose` at batch 8 on move 50's own cold parse-back
`0.raw`.

---

## 1. The binding, and why it is the first thing this arm did

`ddm_pp1_pose_actuation` and `ddm_sj1_multipass_token_predistortion` pin the live pointer at
**MOVE 49** in module globals evaluated at import time — `POINTER_TREE`, `POINTER_ARCHIVE_SHA256`,
`POINTER_D_SEG_T4`, `POINTER_D_POSE_T4`, `LIVE_RAW`, `LIVE_FIELD`, `LIVE_ARGMAX`. Move 50 is a
real pointer move, so **every stage this arm runs would otherwise stand on a stale pointer**, and
the failure would be silent: the carrier anchor, the seg baseline, the pose base and the rate
control would each be the predecessor's while every receipt still said "live".

`ddm_pd2_pose_directed_pass2.bind_move50` re-points them PROCESS-LOCALLY. The shared modules are
not edited — an arm that does not import pd2 is unaffected — and the binder fails closed:

| checked before binding | result |
|---|---|
| move-50 archive bytes / sha256 | 179,195 B / `1ea274f6…03cd7`, MEASURED |
| move-50 argmax sha256 | `b3db8fb550fa33ff…`, MEASURED |
| move-50 decode bytes | 3,662,409,600 B; sha `135c9b3a…` verified on the `base` run |
| move 50's three components recompute its packet S | 0.13628342713679065 vs 0.13628342713679067 (2.8e-17) |
| re-asserted THROUGH `pp1.assert_pointer_identity` and `sj1.assert_carrier_is_pointer` | both report move 50 |

Receipt: `BIND_RECEIPT.json`.

---

## 2. The base, MEASURED on the bytes the T4 row scored

| quantity | measured |
|---|---:|
| n600 mean d_pose on move 50's own decode | **4.448947350080121e-06** |
| ratio vs the T4 print 4.45e-06 | **0.999763449456207** |
| median / max | 3.3184780582363554e-07 / 2.1283759625946122e-04 |
| top-12 share | 53.61 % (the same twelve pp1 floor pairs) |
| instrument d_seg | 0.00010286966959635416 (**12,135** flipped cells, DALI leg) |

**The control that makes it load-bearing.** pd1 PROJECTED the post-admission pose vector for
the 20 pairs it shipped. This arm measured the SHIPPED BYTES independently — a cold parse-back,
`up2.measure_pose` at batch 8, move 50's own carrier — and the two vectors are **bit-identical:
max absolute difference 0.0 over all 600 pairs**, mean equal to all 16 digits. pd1's projection
is exactly what the shipped object measures on this instrument.

Receipts: `base/POSE_BASE_MOVE50.json`, `base/pose_base_move50.npy` (sha `5d0f182e…`),
`base/codes_move50.npy`.

**The base band is pd1's, ABSOLUTE, and re-derived rather than quoted:** pd1's own 40 rows
spanning the whole d_pose range give a maximum absolute batch-1 vs batch-8 gap of
**2.2642595483754955e-09**; the gate is 10× that, **2.2642595483754955e-08**. A relative gate
would be ~400× too loose on the smallest pairs.

---

## 3. The economics at move 50 — they EXPIRE, so they are re-derived

| term | value |
|---|---:|
| S per archive byte | 6.658589531221714e-07 |
| S per flipped SegNet cell (T4-carried through the instrument ratio) | 8.482900700453235e-07 |
| S per unit of **one pair's** d_pose | **1.249367259651861** |

And the price, which is the leg that decides everything. pd1 pre-registered 8.93 bits/token
(pass 8's clustered field) and its own real encode measured **16.98** on a 41-token field with
one token per pair — **1.90× optimistic**. This arm therefore SEARCHES at the measured price:

| price, bits/token | required per-pair credit | non-floor pairs able to pay |
|---|---:|---:|
| 8.93 (pass 8, a different edit shape) | 5.949e-07 | 219 |
| 16.00 (pd1's admitted subset, its own real encode) | 1.0659e-06 | 169 |
| **16.98 (pd1's 41-pair field, its own real encode)** | **1.1312e-06** | **164** |

`|credit| ≤ base` is a hard bound, not a heuristic: one token cannot drive a pair's resolved
d_pose below zero. Two tiers follow, and both are DERIVED, not measured:

* **tier 1 — 164 pairs** whose base exceeds 1.1312e-06 and can pay on pose alone;
* **tier 2 — 83 more** (base > 4.5223e-07) that can pay only if the same edit also REPAIRS a
  seg cell, which pd1 measured on 4 of its 41 carried pairs.

Receipts: `PREREGISTRATION.json`, `search_population.json`.

---

## 4. K_refine, chosen by a MEASURED timing smoke

Eight pairs drawn across tier 1, run as **8 concurrent shards at 2 threads each** — the fleet
condition, not an idle-machine number — at pd1's screen (16 pose-saliency cells masked to
argmax-interior positions, radius 3, survivors at ≤ 2 cells) and **K_refine = 8**:

| quantity | measured |
|---|---:|
| wall per pair | 56.8 – 342.2 s (mean 256.4, median 270.8) |
| fit | seconds = **33.34 × refines + 27.22** (load + screen) |
| fleet throughput at K_refine = 8 | **1.633 pairs/min** |
| projected: 164 tier-1 pairs | **100 min** |
| projected: 247 pairs (tier 1 + tier 2) | **151 min** |

**K_refine = 8 is DECLARED** on that measurement: it lands inside the charter's 3–4 h envelope
with margin for the two concurrent real encodes, and pd1's K_refine = 3 left up to ~20 screened
survivors per pair unrefined. Everything else is pd1's reference form unchanged (cells 16,
interior radius 3, max-cells 2, deltas −1/+1), so the delta is attributable.

Receipt: `SMOKE_TIMING.json`.

---

## 5. The reproduction control: pd2 on move 50 returns pd1 on move 49 bit-identically

For a pair whose render did NOT move between the two rows, the object is the same and the
actuator must return the same numbers. Over the **15 proposals** the two arms share:

**15 of 15 bit-identical** — `d_cells`, `d_pose_base`, `d_pose_resolved` and all twelve carrier
coordinates equal to all digits.

That control does three things at once: it proves the move-50 binding did not perturb the
actuator, it makes pd1's rows on unmoved pairs valid PASS-2 rows, and it shows the re-solve is
deterministic. And the base-band gate then enforces the charter's "re-search the twenty pairs
move 50 edited from their NEW renders" STRUCTURALLY rather than by hand: a pd1 row for one of
those pairs carries the move-49 base, which is outside the measured band, so it is refused.

Receipt: `F_REPRODUCTION_VS_PD1.json`.

---

## 6. The rate control: the pricer reproduces move 50's own token stream

`ddm_sj1_rlc1_price.py init` on move 50's tree splits the shipped member and refuses unless the
split is lossless and the null rebuild is byte-identical to the live archive:

| section | bytes |
|---|---:|
| header | 14 |
| hpac | 11,629 |
| semantic | 29,862 (BYTE-IDENTICAL to moves 48/49 — the renderer did not move) |
| carrier | 18,452 |
| tail | 119,138 = 96 prefix + 64 rider + **118,978 stream** |

The **118,978 B** token stream is the charter's number, MEASURED here on move 50's own bytes
(sha `e639941ca900c49a…`). The decoded plane's digest, read from move 50's T4 cold report, is
`975b0e5e1a63795a4df5a863e70aa165e9cd599aa5fcc766542aa69f2cafadfd` — the same sha as the
pricer's control field, so the object priced IS the object shipped.

Receipt: `rlc1_price/INPUTS.json`.

---

## 7. The search

<!-- SEARCH -->

---

## 8. The admission

<!-- ADMISSION -->

---

## 9. The contract route — CLOSED, and the blocker is move 50's own decode

**The headline of this section: under the frozen contract there is no route today that can seal
ANY distortion-moving candidate built on move 50, and the cause is move 50's own T4 decode, not
this arm's candidate.** Four doors, each read at source and one of them EXECUTED:

| door | requirement | status |
|---|---|---|
| normal seal, `--inherit-decode-wall-clock` | `source.archive_sha256 == pointer_archive_sha256` and the source is a DIRECT leg | **CLOSED** — move 50's archive has no direct leg and none can be minted |
| minting move 50's own `t4_direct` leg | `seconds <= 1260.0` | **CLOSED, EXECUTED**: `build_t4_direct_leg` on move 50's own retained harvest raises `decode_wall_clock: measured T4 decode 1375.753169s exceeds 1260.0s` |
| normal seal, `--decode-wall-clock` mode `measured` | `local_600 × cpu_to_t4_ratio <= 1260` | **CLOSED on the arithmetic** — move 50 decodes locally in 1,124.75 s against 1,375.75 s on T4 (ratio 1.2232); a successor with a comparable local decode projects ~1,376 s |
| first-measurement pre-fire intent | `raw["candidate_raw"]["sha256"] == raw["pointer_raw"]["sha256"]` | **CLOSED for every distortion move** — the intent is a RATE-ONLY instrument by construction (`build_prefire_intent` writes `admit_bar.rate_only_precheck` with `raw_identity_required: True` and derives its net ΔS from BYTES ALONE) |
| within that, `measured_t4_identity_class_envelope` | every leg ≤ 1,260 s AND every leg decoded the CANDIDATE's token plane | **CLOSED twice over** |

The envelope's second leg is worth stating on its own, because it is a property of the mode and
not of this candidate. pr19's identity class is *(this receiver, this decoded token plane)* — the
validator's own `_pf_cold_work_facts` docstring calls the plane digest "the one identity that
crosses the device boundary". Moves 44/46/47/48 all decoded `a92e7d90…`, which is exactly why
hpr1 could build an envelope; move 49 decoded `fdf2255f…`; move 50 decoded `975b0e5e…`. **A
token-changing candidate decodes a fourth plane that no measured leg has ever seen, so its
identity class is EMPTY by construction.** The charter's declared mode is unavailable to any
token-changing row, not merely to this one — and its declared envelope maximum, move 50's
1,375.8 s, is itself inadmissible as a leg.

### The cure is a MEASUREMENT, not an amendment — and it is MAIN's

Mint move 50's `t4_direct` leg by re-measuring its EXACT archive on T4. The class's own measured
decode spread on the same receiver is **1,023.2616 s** (move 48, 179,111 B), **1,106.2192 s**
(move 49, 179,153 B), **1,140.8051 s** (move 46, 180,001 B) and **1,375.7532 s** (move 50,
179,195 B). Move 50 is 42 B larger than move 49 and 806 B SMALLER than move 46, so +269.5 s over
move 49 cannot be payload work — most of it is host variance, and the class straddles the
1,260 s limit. One re-measurement of bytes we already hold would unblock every successor of move
50 through the ordinary inheritance route the pointer has used since move 43. This arm fires
nothing; the call and the dispatch are MAIN's.

Receipts: `SEAL_ROUTE_CLOSURE.json`, `DOOR_CONTROL_MOVE50_LEG.json`, `TIMING_ROUTE_FINDING.json`.

### What this arm did NOT do about it

It did not amend the contract, did not edit `src/tac/candidate_seal.py` or
`src/tac/decode_wall_clock.py`, and did not manufacture a risk object in the one mode still
arithmetically open (`completed_t4_receiver_delta`, whose projection off move 48's 1,023.2616 s
leg would admit a local cost fraction up to 0.2313) — because that mode's own non-timing gate is
the raw-identity gate this candidate cannot pass either. Choosing a route is an adjudication
about the contract, not a measurement. The doors, the refusals, the legs and the arithmetic go
to MAIN; the candidate sits byte-closed and retained, ready for the ordinary inheritance route
the moment move 50 has a leg.

---

## 10. What this does NOT claim

<!-- NOT_CLAIMED -->

---

## 11. Custody

<!-- CUSTODY -->

<!-- # FORMALIZATION_PENDING: the equations leg is written by tools/pointer_move_packet.py --equations-leg at harvest, and only on an exact row. The score arithmetic used throughout is the registered S = 100*d_seg + sqrt(10*d_pose) + 25*B/37,545,489. -->
