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

## 0. The headline

**A candidate is built, byte-closed and retained: 179,285 B, archive sha
`42e47d0bae1b0647d08db8a5061fe3eec6368b2169d89eb62b14f832fdb0978f`, projected S
0.13622721373953445 — net −5.6213e−05 vs move 50, 2.81 bars, clearing the −2e−05 bar. 40 pairs, 40
changed tokens, 416 re-solved carrier coordinates.**

**It is SEALED, on the NORMAL inheritance path.**
`/Volumes/APDataStore/pact/ddm_pd2/SEAL_ddm_pd2_pose_directed_pass2_contest_cuda.json` — file sha
`e52bb733c9191b3b4bf8a28d8d18c09dfc3adeb66835e74bc466f7976f21d908`, 10,590 B, seal sha
`6e4e9131b1434988c71d39085a007f1653aac360e8264a70dc42c01075b9d685`, **SEAL_VALID with zero problems**,
inheriting move 50's own `t4_direct` leg. **NO Modal, NO fire: MAIN fires.**

That sentence had to be rewritten. When this arm finished its build, every seal door refused, and the
blocker was move 50's own 1,375.753 s T4 decode — measured, executed, and reported as closed (§9).
MAIN then did the one thing this arm named as the cure: **re-measured move 50's exact archive on the
T4** (call `fc-01M2CC2116RHEAJ3B37C9Z2EGW`, identical score components, decode **1,123.329 s** ≤
1,260) and minted the leg. The prediction underneath the closure — that move 50's +269.5 s over move
49 was host variance and not decode work — held: the re-measurement came in 252.4 s faster, and this
arm's own candidate had already decoded 11.2 % faster than move 50 locally.

| leg | value | how |
|---|---:|---|
| seg | **+1.10278e−05** | 12,135 → **12,148** flipped cells, MEASURED on the candidate's own cold parse-back; the admission predicted 12,148 with **zero cells disagreeing** |
| pose | **−1.27168e−04** | RESOLVED 4.281953276180728e−06 against the base 4.448947350080121e−06, re-verified n600 on the composed object |
| rate | **+5.99273e−05** | **+90 B EXACT** (tail +77 by the subset's own real encode, twins byte-identical; carrier +13 from the splice) |
| **net vs the pointer** | **−5.6213e−05** | **2.81 bars** |

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

Eight shards, `--threads 2`, walking tier 1 in DESCENDING d_pose so a prefix stop keeps the pairs
that can pay. pd1's actuator unchanged; **K_refine = 8** the only search delta.

| quantity | measured |
|---|---:|
| pairs walked | **166** (all 156 of tier 1, then 10 of tier 2) |
| pairs with at least one realized row | **157** |
| realized rows (render + frozen argmax + carrier re-solve, each) | **1,079** |
| screened proposals (render + frozen argmax only) | **3,091** |
| rows inside the MEASURED absolute base band | **165 / 165 carried**, 0 rejected |
| pairs whose best row is negative on the modelled arithmetic at 16.98 bits/token | **34** |

**The per-pair credit histogram**, over the 165 pairs carried into the candidate field — the best
realized credit as a fraction of that pair's own d_pose:

| band | pairs |
|---|---:|
| −100 % … −50 % | **40** |
| −50 % … −25 % | 30 |
| −25 % … −10 % | 30 |
| −10 % … −5 % | 18 |
| −5 % … −2 % | 13 |
| −2 % … 0 % | 14 |
| ≥ 0 % | 20 |

Median **−18.48 %**, best **−99.31 %** — a pair whose resolved pose is very nearly erased.

**And the seg cost the screen buys down:**

| d_cells of the carried edit | pairs |
|---|---:|
| −1 (a seg REPAIR) | **8** |
| 0 | **74** |
| +1 | 57 |
| +2 | 26 |

82 of 165 carried pairs cost zero or negative seg cells; +101 cells across the whole field.

**Two things this search settles that the charter left open.**

1. **K_refine = 8 is what produced the win, and it is attributable.** pd1's own rows on pairs whose
   render did not move were fed into the carry alongside this arm's. **All 165 carried rows and all
   40 admitted rows are this arm's** — every pd1 row was dominated by a deeper refine on the same
   pair, or refused by the base band. pd1 refined the 3 cheapest-on-seg survivors per pair; the best
   POSE credit is regularly not among them.
2. **"A repair consumes local slack, so expect less on the re-searched pairs" is REFUTED on this
   object.** Eight of the 40 admitted pairs are pairs move 50 itself edited, re-searched from their
   NEW renders — and pairs 548 and 591 are among the four largest single finds in the whole arm.

---

## 8. The admission, and the composition re-verified rather than summed

### The composition, MEASURED on one object

The per-pair search realizes each edit alone, so the composed object was RE-MEASURED: a 600-pair
overlay rendered from the candidate field, an n600 pose leg on it, and a carrier re-solve that
starts from the **SHIPPED** codes, never from the search's own.

| leg | per-pair sum | composed, re-verified | realized fraction |
|---|---:|---:|---:|
| pose, all 165 carried | −1.472077e−04 | **−1.472120e−04** | **1.000030** |
| pose, the 40 ADMITTED | −1.001836e−04 | **−1.001964e−04** | **1.000129** |
| seg, the 40 admitted | +13 cells | **+13 cells** | 1.000000 |

Per-pair ratio over the admitted 40: min 0.997862, median 0.999967, max 1.003280. **Falsifier F2
(composition realizes < 0.8 of the sum) does not fire.**

**The two null controls that make those numbers mean something:**

* the **435 unedited pairs** are bit-identical to their base on BOTH legs — max |stale − base| =
  **0.000e+00** and max |resolved − base| = **0.000e+00** — so nothing leaks from an edited pair to
  an untouched one;
* the STALE pose (candidate renders, shipped carrier) is **214.3× the base**. The re-solve recovers
  all of it and then some: composed resolved 4.2035940284384715e−06 against base
  4.448947350080121e−06, **−5.51 % on the population mean from 165 single-token edits.**

### The rate leg, by REAL encode, twins

| field | pairs / tokens | exact archive | Δ vs move 50 | bits/changed token | realized ÷ first-order ideal | twins |
|---|---:|---:|---:|---:|---:|---|
| control (move 50's own field) | 0 / 0 | **179,195 B** sha `1ea274f6…` | 0 | — | — | **byte-identical, 2 processes** |
| full candidate field | 165 / 165 | **179,538 B** sha `110d98b3…` | **+343 B** | **16.630** | 0.99927 | **byte-identical, 2 processes** |
| admitted subset | 40 / 40 | **179,272 B** sha `d4ea52b0…` | **+77 B** | **15.400** | 0.99093 | **byte-identical, 2 processes** |
| the built candidate (subset tail + re-solved carrier) | 40 / 40 | **179,285 B** sha `42e47d0b…` | **+90 B** | — | — | the staged archive reproduces the pricer's repack byte for byte |

The container did not draw a lottery here either: realized ÷ first-order-ideal is 0.99927 on the
full field and 0.99093 on the subset, so the ±34.8 B break term is absent on this object at this
edit shape. That is a measurement on two fields, not a law.

### The three-leg Lagrange admission

40 of 165 pairs kept, on the RESOLVED pose, with the rate leg taken from the MEASURED per-pair bit
ledgers rather than apportioned. Credit fraction among the admitted: median **−52.74 %**, best
−99.31 %; **26 of 40 cost zero or negative seg cells** (3 repairs, 23 neutral).

### The candidate, and its legs on the SHIPPED bytes

| leg | value | how |
|---|---:|---|
| rate | 0.11937852241050849 | **179,285 B EXACT** — the subset's own real encode (+77 B) plus the carrier splice (+13 B over 40 pairs / 416 coordinates) |
| seg | 0.01030502777091059 | **12,148 flipped cells MEASURED on the candidate's own cold parse-back** — the admission predicted 12,148, **zero cells disagreeing** |
| pose | 0.0065436635581153837 | RESOLVED 4.281953276180728e−06, re-verified n600 on the composed object |
| **S projected** | **0.13622721373953445** | |
| **net vs move 50** | **−5.621340e−05** | **2.81 bars** |

Archive sha **`42e47d0bae1b0647d08db8a5061fe3eec6368b2169d89eb62b14f832fdb0978f`**, 179,285 B.

**Proved on the shipped bytes, not on the plan:**

* the candidate's own cold parse-back decodes a token plane **byte-identical to the admitted
  field** (`decoded_field_matches_admitted: true`), raw sha `b46351f8…`, 3,662,409,600 B;
* the seg leg above is measured on THAT decode, not on the overlay;
* the public entrypoint smoke is **symmetric** between candidate and frontier — both reach the CUDA
  gate through `inflate.sh` in 1.88 s and both reach the token decode through the public path inside
  the 240 s bound, with the four native-library exports set as `inflate.sh` sets them;
* the literal census is **CLEAR at rule 118**: 51 files, **three differ from the pointer tree** —
  `archive.zip`, `inflate.py` and the derived `MANIFEST.sha256` — and the ENTIRE `inflate.py` diff is
  the two pins that name the archive:

```
-ARCHIVE_SHA256 = "1ea274f612a26183f31f6d439505d0289bd3751b4da9467343fc156203503cd7"
-ARCHIVE_BYTES = 179195
+ARCHIVE_SHA256 = "42e47d0bae1b0647d08db8a5061fe3eec6368b2169d89eb62b14f832fdb0978f"
+ARCHIVE_BYTES = 179285
```

**And one measurement that bears directly on §9:** the candidate's cold n600 local decode is
**999.008 s** against move 50's **1,124.751 s** on the same host and the same receiver — the
candidate decodes **11.2 % FASTER** than the row it is built on.

---

## 9. The contract route — measured CLOSED, then OPENED by the measurement it named

**Read this section as a sequence, because the verdict changed.** When the candidate was built,
every seal door refused and the cause was move 50's own T4 decode — not this candidate. This arm
executed each door, recorded each typed refusal, and named the one cure that was a MEASUREMENT
rather than an amendment. MAIN then took that cure, and the route reopened. Both halves are kept:
the closure is what made the cure legible, and a reader who only sees the seal would not know which
door had to be opened or why.

### 9a. The state at build time — four doors, each EXECUTED, all four refusing

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

### 9b. The cure is a MEASUREMENT, not an amendment — and it is MAIN's

Mint move 50's `t4_direct` leg by re-measuring its EXACT archive on T4. The class's own measured
decode spread on the same receiver is **1,023.2616 s** (move 48, 179,111 B), **1,106.2192 s**
(move 49, 179,153 B), **1,140.8051 s** (move 46, 180,001 B) and **1,375.7532 s** (move 50,
179,195 B). Move 50 is 42 B larger than move 49 and 806 B SMALLER than move 46, so +269.5 s over
move 49 cannot be payload work — most of it is host variance, and the class straddles the
1,260 s limit. One re-measurement of bytes we already hold would unblock every successor of move
50 through the ordinary inheritance route the pointer has used since move 43.

And there is a second, independent measurement pointing the same way. On THIS host, through the
SAME receiver, this arm's candidate decodes n600 cold in **999.008 s** where move 50 decodes in
**1,124.751 s** — **11.2 % faster**, on 90 MORE archive bytes and 622 more coded bits. Decode work is
not what put move 50 over the limit. This arm fires nothing; the call and the dispatch are MAIN's.

Receipts: `SEAL_ROUTE_CLOSURE.json`, `DOOR_CONTROL_MOVE50_LEG.json`, `TIMING_ROUTE_FINDING.json`.

#### The doors, EXECUTED

| attempt | outcome |
|---|---|
| `make_candidate_seal.py --inherit-decode-wall-clock <move 49 t4_direct leg>` | no seal written; `FATAL: cannot seal this candidate: decode_wall_clock: decode_wall_clock: source measurement is not the pointer archive` |
| `make_candidate_seal.py --decode-wall-clock <move 49 t4_direct leg>` | no seal written; `FATAL: cannot seal this candidate: decode wall-clock refused: decode_wall_clock: candidate receiver differs from measurement` |
| `make_candidate_seal.py --first-fire-intent …` | no intent written; `FIRST_MEASUREMENT_ARGUMENT_REFUSED: intent accepts only frozen CUDA contract flags` |
| `tac.decode_wall_clock.build_t4_direct_leg` on move 50's own harvest | `SealContractError: decode_wall_clock: measured T4 decode 1375.753169s exceeds 1260.0s` |

And the gate that closes the intent route for good, MEASURED rather than argued: the candidate's own
cold n600 decode is sha `b46351f8bee766481a3c1aca9c8d9781a34d11d0377c70aa145944353356202f`, move
50's is `135c9b3a580fcc84d74da8e2ac547f6ba06059b5e2148c8a2086ef843bd72fb7`. `_pf_evidence` requires
them EQUAL. A pose-directed edit changes the rendered frames by construction, so that equality is
unreachable for any distortion move, at any byte count, on any hardware.

Receipt: `REAL_DOOR_CONTROL.json`, and the four refusal transcripts under `door/`.

#### What this arm did NOT do at the time

It did not amend the contract, did not edit `src/tac/candidate_seal.py` or
`src/tac/decode_wall_clock.py`, and did not manufacture a risk object in the one mode still
arithmetically open (`completed_t4_receiver_delta`, whose projection off move 48's 1,023.2616 s
leg would admit a local cost fraction up to 0.2313) — because that mode's own non-timing gate is
the raw-identity gate this candidate cannot pass either. Choosing a route is an adjudication
about the contract, not a measurement. The doors, the refusals, the legs and the arithmetic go
to MAIN; the candidate sits byte-closed and retained, ready for the ordinary inheritance route
the moment move 50 has a leg.


### 9c. The seal, on the NORMAL inheritance path

MAIN re-measured move 50's exact archive on the T4 — call `fc-01M2CC2116RHEAJ3B37C9Z2EGW`, identical
score components (S 0.13628342713679067), **decode 1,123.3287576769999 s ≤ 1,260** — minted its own
`t4_direct` leg with `build_t4_direct_leg`, and adopted it as move 50's sidecar. Committed copy:
`.omx/research/ddm_pd1_packet_inputs_20260913/MOVE50_T4_DIRECT_LEG.decode_wall_clock.json` (commit
`f880398e2`).

**The prediction this arm staked the closure on held.** The claim was that move 50's +269.5 s over
move 49 could not be decode work — 42 B more than move 49 and 806 B LESS than move 46 — and was host
variance. The re-measurement of the SAME bytes came back **252.4 s faster** (1,375.753 → 1,123.329 s),
and the candidate's own local decode had already measured 11.2 % faster than move 50's.

**Re-validated independently before use**, not trusted: `validate_decode_wall_clock` on the sidecar
returns **0 problems** — mode `t4_direct`, completed, 600 pairs, Tesla T4, 1,123.329 s against the
1,260 s limit, on move 50's archive `1ea274f6…`.

**Seal inputs staged from OUTSIDE the candidate tree** (the manifest must not verify itself):

| input | what |
|---|---|
| `seal_inputs/CANDIDATE_MANIFEST.json` | 51 rows regenerated from the bytes on disk through `tac.candidate_seal.measure_runtime_digest`, with the normalized-receiver digest |
| `seal_inputs/MANIFEST_VALIDATION.json` | `all_hashes_passed: true`, `all_runtime_dependencies_listed: true`, plus the Catalog #420 derived-listing re-derivation `tac.decode_wall_clock.validate_receiver_manifest` on the tree |
| `seal_inputs/TWIN_ENCODE.json` + `ENCODER_EXECUTION_{0,1}.json` | two independent encoder PROCESSES, each with its own natively compiled rc64 backend, produced **byte-identical members** (179,172 B, sha `7a2898e7…`) |
| `PUBLIC_SMOKE.json` | the symmetric candidate/frontier entrypoint smoke measured in §8 |
| receiver pins | `inflate.py` (2,735 B, `4e32b448…`) and `inflate.sh` (3,380 B, `0a1820f2…`) |
| retained paths | the whole store, hashed in `RETENTION_MANIFEST.json` |
| falsifiers | the eight pre-registered rows, each carrying its MEASURED outcome |

**The seal**

```
/Volumes/APDataStore/pact/ddm_pd2/SEAL_ddm_pd2_pose_directed_pass2_contest_cuda.json
file sha  e52bb733c9191b3b4bf8a28d8d18c09dfc3adeb66835e74bc466f7976f21d908   (10,590 B)
seal sha  6e4e9131b1434988c71d39085a007f1653aac360e8264a70dc42c01075b9d685
VALIDATED SEAL_VALID, problems []   (validate_seal, require_decode_wall_clock=True)
```

candidate `ddm_pd2_pose_directed_pass2`, axis `contest_cuda`; archive 179,285 B sha `42e47d0b…`;
runtime 51 files / 1,008,644 B, digest `4e0792bc…`; leg mode **inherited**, scope *"identical
normalized receiver code; candidate payload-dependent time not remeasured"*, source the move-50
sidecar sha `8653c407…`; admit bar **net ΔS < −2e−05** against the LIVE pointer move 50
(`1ea274f6…`, 0.13628342713679067) at tolerance 0.

**The candidate was NOT changed to seal it** — the archive, the runtime digest and both pins are the
same objects §8 measured. **NO Modal, NO fire, NO completion, NO packet: MAIN fires.**

---

## 10. What this does NOT claim

1. **No score of any kind.** Every S here is a PROJECTION on measured legs. Only
   `upstream/evaluate.py` on the shipped bytes is a score, and MAIN fires. This arm ran no Modal
   call, wrote no authorization, no completion and no packet.
2. **The pose numbers are `[macOS-CPU advisory]`,** measured on a frozen CPU-torch PoseNet against
   DALI-lineage GT. The instrument's base is 4.448947350080121e−06 against the T4 print 4.45e−06
   (ratio 0.999763) — the tightest agreement this lineage has recorded — but the last several
   packets each measured a local-versus-T4 pose-print class of a few e−06 in the OPTIMISTIC
   direction, and the exact row decides.
3. **The seg leg is carried onto T4 by the same-instrument ratio** (1.0006836845488258), not
   measured there. What IS measured on the shipped bytes is the instrument leg: 12,148 cells.
4. **The container-break lottery is measured, not removed.** The campaign's standing sd is 34.8 B =
   2.32e−05 S ≈ 1.16 bars, and this object drew 0.99093–0.99927 of its first-order ideal on two
   fields. A successor field draws again.
5. **verdict_scope: FORMULATION, on this object.** What is measured is pose-directed single-token
   pre-distortion on move 50's field, proposals ranked by pose saliency masked to argmax-interior
   cells, refined 8-deep, admitted on the resolved pose. Multi-token edits per pair, a clustered
   search (pd1's own §11 lever), and any other carrier parametrisation are untouched.
6. **The population is DERIVED-bounded, not exhausted.** Tier 1 (164 pairs able to pay on pose
   alone at the measured price) was searched to completion; tier 2 (83 pairs that need a repaired
   cell) was searched only 10 deep. The ideal bound on what tier 2 could still hold is small but it
   is NOT zero, and the 341 pairs below both tiers were excluded by the DERIVED bound
   `|credit| ≤ base`, which says they cannot pay at this price — not that no edit there does
   anything.
7. **15.400 bits/token is THIS subset at THIS edit shape**, measured once, as 16.630 is this field's
   and 16.98 was pd1's. None of the three is a law.
8. **The per-bit ranking prices a PAIR, not a proposal.** The rlc1 ledger's `delta_bits` is per
   pair, so two proposals on one pair are charged the same; where they really do cost different
   numbers of bits this ranking cannot see the difference. Said plainly rather than dressed up.
9. **pp1's twelve floor pairs stay excluded** on pp1's measurement. Nothing here reopens them.
10. **The seal is a seal, not a score.** `SEAL_VALID` says the candidate's identity, timing leg,
    smoke, pins and admit bar are coherent; it says nothing about what the T4 will measure. The
    build-time closure in §9a is preserved rather than deleted: it is what named the cure MAIN then
    ran, and its four typed refusals remain true of the state they were measured in.
11. **This arm fired nothing.** No Modal call, no authorization, no completion, no packet. MAIN
    fires.

---

## 11. Custody

Store **`/Volumes/APDataStore/pact/ddm_pd2/`**. APDataStore rather than Vertigo because Vertigo held
**38 GiB** free — below its 40 GiB reserve. **The reserve was never lowered and nothing was written
there.**

**MEASURED: 6,021,453,230 B over 767 files, every one hashed — under the 8 GiB cap.**
`RETENTION_MANIFEST.json` carries bytes and sha256 for each, including the losers: all 1,079 realized
search rows and all 3,091 screened proposals, not only the 40 that shipped.

| path | what |
|---|---|
| `candidate/candidate_archive.zip` · `candidate/candidate_runtime/` | **179,285 B sha `42e47d0b…`** and the 51-file runtime |
| `parseback/0.raw` · `parseback/PARSEBACK_RESULT.json` | the cold decode (3,662,409,600 B sha `b46351f8…`) and its receipt |
| `seg_final/argmax_n600.npy` · `STEP0_RESULT.json` | the seg leg on that decode: 12,148 cells |
| `pose/overlay_pd2/` | the 600-pair odd-frame overlay the pose leg scored (1,831,204,800 B) |
| `pose/pose_stale.npy` · `pose/pose_resolved.npy` · `base/pose_base_move50.npy` | the three n600 pose vectors |
| `refine/refine_rows_*.jsonl` · `refine/codes_resolved.npy` | the composed re-solve, per pair |
| `search/search_b_*.jsonl` · `search/screen_*.jsonl` · `smoke/` | **every realized row and every screened proposal, winners and losers** |
| `assemble/` · `assemble_perbit/` · `admission/` · `stage/` | the candidate field, the per-bit carry ranking, the Lagrange sweep and its trace, the staged tail |
| `rlc1_price/` | the pricer's INPUTS, both control encodes, both full-field encodes, both subset encodes, the per-pair bit ledgers |
| `BIND_RECEIPT.json` · `PREREGISTRATION.json` · `SMOKE_TIMING.json` · `STOP_RULE.json` · `SEARCH_PLAN.json` · `F6_CONTROL_ENCODE.json` · `F_REPRODUCTION_VS_PD1.json` · `F_FLIPS_IDENTITY.json` · `YIELD_HISTOGRAM.json` · `PUBLIC_SMOKE.json` · `LITERAL_CENSUS.json` | the pre-registration and every control receipt |
| `SEAL_ROUTE_CLOSURE.json` · `DOOR_CONTROL_MOVE50_LEG.json` · `REAL_DOOR_CONTROL.json` · `TIMING_ROUTE_FINDING.json` · `door/` | the contract-route finding and every executed refusal |
| `SEAL_ddm_pd2_pose_directed_pass2_contest_cuda.json` | **SEAL_VALID**, file sha `e52bb733…` (10,590 B), seal sha `6e4e9131…` |
| `seal_inputs/` | the candidate manifest regenerated from outside the tree, its validation, the twin-encode receipt and both encoder-execution receipts |
| `PIPELINE_PLAN.md` | the chain as exact commands, written while the search ran |

Producers: `experiments/ddm_pd2_pose_directed_pass2.py` (`bind | base | prereg | run | carry`),
ruff-clean, plus pd1's `ddm_pd1_pose_directed.py search|assemble` and `ddm_pd1_candidate_tree.py
build`, and sj1's `ddm_sj1_rlc1_price.py` and `ddm_sj1_joint_admission.py`, all unchanged. Nothing
under `/Volumes/APDataStore/pact/ddm_pd1`, `/Volumes/VertigoDataTier/pact/ddm_sj1_pass7`,
`ddm_sj1_pass8`, `ddm_pp1`, `ddm_cb1`, `ddm_cr1` or `ddm_so2` was written; every module that imports
from a custody tree sets `sys.dont_write_bytecode = True`.

## 12. What this hands the next arm

1. **The candidate is built, byte-closed and SEALED; the only thing left is the fire.** MAIN minted
   move 50's leg and this candidate inherited it on the ordinary route, because its receiver is
   byte-identical to move 50's.
2. **K_refine is the cheap lever and it is not exhausted.** Every admitted row came from a refine
   pd1 did not run. The screen still throws away survivors past rank 8 on the pairs that have them.
3. **The rate leg is still what decides the size of the win.** 15.4 bits for an isolated token; pd1's
   §11 lever — CLUSTER the edits, or spend a second token on a pair already open — is untouched, and
   pass 8's 8.93 bits/token on a clustered field is the number to beat.
4. **Tier 2 is barely searched** (10 of 83), and its pairs pay only through a seg repair — which this
   arm measured on 8 of 165 carried edits.

<!-- # FORMALIZATION_PENDING: the equations leg is written by tools/pointer_move_packet.py --equations-leg at harvest, and only on an exact row. The score arithmetic used throughout is the registered S = 100*d_seg + sqrt(10*d_pose) + 25*B/37,545,489. -->
