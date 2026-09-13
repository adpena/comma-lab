# ddm_pd3 — pose-directed token pre-distortion, PASS 3 on the move-51 field

Tokens: `[no-triality] [p0-ledger-ok]`

Lane `ddm_pd3_pose_directed_pass3_on_move51_20260913`. Base: **move 51**,
S 0.1362333315680336 @ 179,285 B, archive sha
`42e47d0bae1b0647d08db8a5061fe3eec6368b2169d89eb62b14f832fdb0978f`. Axis
`[macOS-CPU advisory, frozen CPU-torch PoseNet + SegNet, DALI GT lineage, n600]` for the
distortion legs; bytes EXACT through the shipped RLC1 coder and the shipped container.
**No score is claimed. Only `upstream/evaluate.py` on shipped bytes is a score, and MAIN
fires.** No Modal, no authorization, no fire, no completion, no packet.

Pose solver, stated once: every resolved pose number here comes from
`ddm_jg5_pose_resolve_on_edited_renders.refine_pair` (pd1's, pd2's, pass 8's and cr1's same
object at these budgets), driven at PoseNet batch 1 on the moved render; every n600 number
comes from `ddm_up2_shipping_pose_solve.measure_pose` at batch 8, on move 51's own cold
parse-back for the base and on the CANDIDATE's own cold parse-back for the candidate.

Provenance pins: HEAD at charter `59b396a9d`, at landing `eea058849`; pd2 memo
`32b11a4c9c899d4e…`; move-51 packet `60b3686d1fcd1669…`; move-51 leg sidecar
`512b9f9e5ff2a2a3…`.

---

## 0. The headline

**A candidate is built, byte-closed, retained and SEALED: 179,332 B, archive sha
`ae59c5109597968409c4b581337b8e23b5b3c4262fd3710c315cc83913b3b20e`, projected S
0.13620052157076992 — net −3.2810e−05 vs move 51, 1.64 bars, clearing the −2e−05 bar. 26
pairs, 26 changed tokens, 259 re-solved carrier coordinates.**

`/Volumes/APDataStore/pact/ddm_pd3/SEAL_ddm_pd3_pose_directed_pass3_contest_cuda.json` —
file sha `9905cb9238cccf29f738ab71c28a073eb06bc90512e777db509ce49d3c45bf9e`, 10,851 B, seal
sha `c2ef42665b4e0382c7382a36d92a69059db879bc7728196bd50b558baa47ff9a`, **SEAL_VALID with
zero problems**, on the NORMAL route inheriting move 51's own `t4_direct` leg. **NO Modal,
NO fire: MAIN fires.**

| leg | value | how |
|---|---:|---|
| seg | **−8.48288e−07** | 12,148 → **12,147** flipped cells, MEASURED on the candidate's own cold parse-back; the admission predicted 12,147 with **zero cells disagreeing** |
| pose | **−6.32571e−05** | RESOLVED 4.207535785085938e−06 against the base 4.281953276180728e−06, re-verified n600 on the composed object AND then re-measured **bit-identically on the shipped decode** |
| rate | **+3.12954e−05** | **+47 B EXACT** (tail +42 by the subset's own real encode, twins byte-identical; carrier +5 from the splice) |
| **net vs the pointer** | **−3.2810e−05** | **1.64 bars** |

---

## 1. The binding

`ddm_pp1_pose_actuation` and `ddm_sj1_multipass_token_predistortion` pin the live pointer at
**MOVE 49** in module globals evaluated at import time; `ddm_pd1_pose_directed` pins move 49
in its own constants; `ddm_pd2_pose_directed_pass2` pins **MOVE 50**. Move 51 is a real
pointer move, so **every stage this arm runs would otherwise stand on a stale pointer**, and
the failure would be silent: the carrier anchor, the seg baseline, the pose base and the rate
control would each be a predecessor's while every receipt still said "live".

`ddm_pd3_pose_directed_pass3.bind_move51` re-points all four PROCESS-LOCALLY. The shared
modules are not edited — an arm that does not import pd3 is unaffected — and the binder fails
closed:

| checked before binding | result |
|---|---|
| move-51 archive bytes / sha256 | 179,285 B / `42e47d0b…0978f`, MEASURED |
| move-51 argmax sha256 | `b0f35bd5ca12c867…`, MEASURED |
| move-51 token field npz sha256 | `e9c47f0f7c13c710…`, MEASURED |
| move-51 decode bytes / sha256 | 3,662,409,600 B / `b46351f8…202f`, MEASURED (`--verify-raw`) |
| move 51's three components recompute its packet S | 0.1362333315680336 exactly |
| re-asserted THROUGH `pp1.assert_pointer_identity`, `sj1.assert_carrier_is_pointer`, and both `pd1`/`pd2` `seg_s_per_cell()` | all report move 51 |

Receipt: `BIND_RECEIPT.json`.

pd1's constants were added to the binder AFTER the search shards launched. That is recorded
rather than smoothed: the running shards computed their rows' own `dS_modelled_one_token` at
move 49's seg ratio (1.000663 against move 51's 1.000681 — **0.0018 % apart, MEASURED**), and
pd3's `carry` stage recomputes every row's dS from `credit_d_pose` and `d_cells` at move-51
economics before anything is ranked, so the admission never reads the search's own dS.

---

## 2. The base, MEASURED on the bytes the T4 row scored

| quantity | measured |
|---|---:|
| n600 mean d_pose on move 51's own decode | **4.281953276180728e-06** |
| ratio vs the T4 print 4.29e-06 | **0.9981243068020347** |
| median / max | 2.9930223327187533e-07 / 2.1283759625946122e-04 |
| top-12 share | 55.70 % (the same twelve pp1 floor pairs) |
| instrument d_seg | 0.00010297987196180555 (**12,148** flipped cells, DALI leg) |

**The control that makes it load-bearing.** pd2 closed its candidate — the object that became
move 51 — on d_pose **4.281953276180728e-06** (`candidate/CLOSE.json`). This arm measured the
SHIPPED BYTES independently: a cold parse-back, `up2.measure_pose` at batch 8, move 51's own
carrier. The two agree **to all sixteen digits**. pd2's projection is exactly what the shipped
object measures on this instrument.

A second comparison is reported because it is a real gap rather than a match: against pd2's
`pose/pose_resolved.npy` — the vector it measured on its 165-edit OVERLAY, not on the shipped
40-edit object — the maximum absolute per-pair difference is **1.706815446086048e-05**, and
the largest gap sits on a pair neither arm edited. That vector is a different object (165
edits, overlay renders), which is exactly why the base is measured on the DECODE.

Receipts: `base/POSE_BASE_MOVE51.json`, `base/pose_base_move51.npy`, `base/codes_move51.npy`.

**The base band is pd1's, ABSOLUTE, and unchanged because the instrument is unchanged:**
2.2642595483754955e-08 (10× pd1's measured max batch-1-vs-batch-8 gap over 40 pairs spanning
the whole range).

---

## 3. The economics at move 51 — they EXPIRE, so they are re-derived

| term | value |
|---|---:|
| S per archive byte | 6.658589531221714e-07 |
| S per flipped SegNet cell (T4-carried through the instrument ratio) | 8.482877839973658e-07 |
| S per unit of **one pair's** d_pose | **1.2734966061937** |

Three MEASURED real-encode prices exist for this edit shape on this coder: pd1's 41-token
field at **16.98** bits/token, pd2's 165-token field at **16.630**, and pd2's 40-token shipped
subset at **15.400**. This arm SEARCHES at **15.400** — the closest analogue in shape to what
pass 3 ships — and re-prices by its own real encode before anything is admitted.

| price, bits/token | required per-pair credit | non-floor pairs able to pay |
|---|---:|---:|
| 8.93 (pass 8, a different edit shape) | 5.836e-07 | 212 |
| **15.400 (pd2's shipped subset)** | **1.0065e-06** | **161** |
| 16.630 (pd2's full field) | 1.0869e-06 | 153 |
| 16.98 (pd1's 41-token field) | 1.1098e-06 | 151 |

`|credit| ≤ base` is a hard bound, not a heuristic. Two tiers follow, both DERIVED:

* **tier 1 — 161 pairs** whose base exceeds 1.0065e-06 and can pay on pose alone;
* **tier 2 — 113 more** (base > 3.404e-07) that pay only if the same edit also REPAIRS a seg
  cell, which pd2 measured on 8 of its 165 carried pairs.

The pass-3 population is three MEASURED strata, each with pd2's own admitted fraction as its
prior: **R** = the 165 pairs pd2 carried, re-walked at a deeper K (prior 40/165); **U1** = 16
tier-1 pairs pd2 never walked (prior 38/153); **U2** = 102 tier-2 pairs pd2 never walked
(prior 2/12). **283 pairs**, ranked by base d_pose × that prior, round-robin over 8 shards so
a prefix stop keeps a near-uniform prefix of the global ranking.

Receipts: `PREREGISTRATION.json`, `SEARCH_PLAN.json`, `STOP_RULE.json`.

---

## 4. K_refine and the pair budget, both DECLARED from a MEASURED smoke

Eight pairs drawn at ranks 0/40/80/120/160/200/240/280 of the expected-credit order, run as
**8 concurrent shards at 2 threads each, CONCURRENT with the two rlc1 control encodes** — the
fleet condition this arm actually ran in, not an idle machine (pd2's own smoke ran before its
encodes started).

| shard | pair | screened | refined | wall s |
|---:|---:|---:|---:|---:|
| 0 | 57 | 17 | 12 | 577.1 |
| 1 | 407 | 16 | 12 | 496.7 |
| 2 | 201 | 26 | 6 | 313.0 |
| 3 | 373 | 16 | 12 | 456.5 |
| 4 | 209 | 16 | 12 | 420.9 |
| 5 | 336 | 16 | 12 | 294.8 |
| 6 | 0 | 16 | 12 | 404.2 |
| 7 | 309 | 16 | 12 | 332.0 |

| quantity | measured |
|---|---:|
| wall per pair | 294.8 – 577.1 s (mean 411.9, median 412.6) |
| seconds per refine | 24.6 – 52.2 (mean 37.6, median 36.6) |
| fleet throughput at K_refine = 12 | **1.1653 pairs/min** |
| projected: the 283-pair population | **242.9 min = 4.05 h** |

**The linear fit is reported and then set aside.** `seconds = 18.85 × refines + 199.87` has
r² = **0.168**: the per-refine cost is pair-dependent (a larger d_pose needs more Gauss-Newton
work), so the budget is derived from the mean wall per pair, not from a slope. pd2's own
`33.34 × refines + 27.22` was fitted on 8 points too and deserves the same caution.

**K_refine = 12 is DECLARED** on evidence that the depth BINDS: 7 of the 8 smoke pairs hit
the K=12 cap, and on pd2's own retained rows **113 of 165 pairs hit the K=8 cap — including
32 of its 40 ADMITTED pairs**. **The pair budget is DECLARED at the full 283** on the
throughput above. Everything else is pd1's reference form unchanged (cells 16, interior
radius 3, max-cells 2, deltas −1/+1), so the delta is attributable.

REALIZED: 283 pairs walked, slowest shard 11,149.1 s, wall 04:26 → 07:31 UTC = **3.1 h**,
inside the declared envelope. The prefix stop was never needed.

Receipts: `SMOKE_TIMING.json`, `STOP_RULE.json`, `search/SEARCH_*.json`.

---

## 5. The rate control: the pricer reproduces move 51's own archive, not just its stream

`ddm_sj1_rlc1_price.py init` on move 51's tree splits the shipped member:

| section | bytes |
|---|---:|
| header | 14 |
| hpac | 11,629 |
| semantic | 29,862 (BYTE-IDENTICAL to moves 48–51 — the renderer did not move) |
| carrier | 18,465 |
| tail | 119,215 = 96 prefix + 64 rider + **119,055 stream** |

Two independent control encodes then re-encoded move 51's own field:

* both produced a **119,055 B** stream with sha `0d8d237b5970a5cf…` — **byte-identical to the
  shipped stream**;
* both packed an archive with sha `42e47d0bae1b0647…` — **byte-identical to move 51's own
  archive**, which `price` reports as `control_identity_passed: true`;
* realized 119,055 B against a first-order ideal of 119,054.56 B.

**Falsifier F4 does not fire.** Receipts: `rlc1_price/INPUTS.json`, `F4_CONTROL_ENCODE.json`.

A third, unplanned control fell out of the staging step: the rider+stream file this arm built
from move 51's tail has sha `5933fc70674c7f38…`, which is **exactly the sha pd2 recorded for
the rider+stream it spliced**. The two arms' byte chains meet on the same object.

---

## 6. The search

Eight shards, `--threads 2`, K_refine 12, walking the expected-credit order.

| quantity | measured |
|---|---:|
| pairs walked | **283** (165 R + 16 U1 + 102 U2) |
| screened proposals (render + frozen argmax only) | **5,226** |
| realized rows (render + frozen argmax + carrier re-solve, each) | **2,647** |
| rows inside the MEASURED absolute base band | **271 / 271 carried**, 0 rejected |
| pd2's own rows offered to the carry | 1,134, of which **298 refused by the band** |

**The per-pair credit histogram**, over the 271 pairs carried — the best realized credit as a
fraction of that pair's own d_pose:

| band | carried | admitted |
|---|---:|---:|
| −100 % … −50 % | **102** | **15** |
| −50 % … −25 % | 41 | 4 |
| −25 % … −10 % | 30 | 4 |
| −10 % … −5 % | 22 | 2 |
| −5 % … −2 % | 16 | 0 |
| −2 % … 0 % | 17 | 1 |
| ≥ 0 % | 43 | 0 |

Median carried **−27.32 %**, median admitted **−61.59 %**, best **−99.15 %**.

**And the seg cost the screen buys down:**

| d_cells of the carried edit | carried | admitted |
|---|---:|---:|
| −1 (a seg REPAIR) | **14** | **6** |
| 0 | 143 | 15 |
| +1 | 79 | 5 |
| +2 | 35 | 0 |

157 of 271 carried pairs cost zero or negative seg cells; 21 of the 26 admitted do.

**Three things this search settles.**

1. **The deeper refine paid, and it is attributable.** On the **125 pairs both arms walked in
   band**, K=12 is **strictly better on 24, identical on 101, worse on 0**; the summed best
   credit improves from −5.9274e−05 to −7.0843e−05, **+19.5 % from refines 9–12 alone**, with
   a single-pair maximum of **20.34×**. Every one of the 271 carried winners is this arm's
   own K=12 row; not one pd2 row survived the comparison. **Falsifier F8 does not fire.**
   (The three smoke pairs whose render did not move — 57, 201, 373 — returned pd2's best
   credit EXACTLY; each had fewer than 8 survivors, so K never bound on them. That is the
   reproduction control, not evidence against the depth.)
2. **Tier 2 under-delivered against its own prior, and by a lot.** MEASURED admitted fraction
   by stratum: **U1 3/8 = 37.5 %**, **R 19/163 = 11.7 %**, **U2 4/100 = 4.0 %** — against
   pd2's priors of 24.8 %, 24.2 % and 16.7 %. The 102 tier-2 pairs pd2 never walked were the
   largest single block of new work in this arm and returned four admitted pairs.
3. **"A repair consumes local slack" is REFUTED again, harder.** **15 of the 26 admitted
   pairs are pairs move 51 itself edited**, re-searched from their NEW renders — 58 % of the
   admitted set, against pd2's 8 of 40.

Receipts: `assemble_perbit/CARRY_RANKING.json`, `YIELD_HISTOGRAM.json`, `F8_DEPTH_CONTROL.json`,
`F3_BASE_BAND_CONTROL.json`, and every realized row and screened proposal under
`search/` and `smoke/`.

---

## 7. The admission, and the composition re-verified rather than summed

### The composition, MEASURED on one object

The per-pair search realizes each edit alone, so the composed object was RE-MEASURED: a
600-pair overlay rendered from the 271-edit candidate field, an n600 pose leg on it, and a
carrier re-solve that starts from the **SHIPPED** codes, never from the search's own.

| leg | per-pair sum | composed, re-verified | realized fraction |
|---|---:|---:|---:|
| pose, all 271 carried | −1.2002071e−04 | **−1.2001539e−04** | **0.999956** |
| pose, the 26 ADMITTED | −4.4655375e−05 | **−4.4650495e−05** | **0.999891** |
| seg, the 26 admitted | −1 cell | **−1 cell** | 1.000000 |

Per-pair ratio over the admitted 26: min 0.993327, median 1.000168, max 38.75 (one pair whose
predicted credit was near zero realized much more; it is a bonus, not a driver).
**Falsifier F2 does not fire.**

**The two null controls that make those numbers mean something:**

* the **329 unedited pairs** are bit-identical to their base on BOTH legs — max |stale − base|
  = max |resolved − base| = **0.000e+00**;
* the STALE pose (candidate renders, shipped carrier) is **355.8× the base**. The re-solve
  recovers all of it and then some: composed resolved 4.0819276e−06 against base
  4.2819533e−06, **−4.67 % on the population mean from 271 single-token edits.**

### The rate leg, by REAL encode, twins

| field | pairs / tokens | exact archive | Δ vs move 51 | bits/changed token | realized ÷ first-order ideal | twins |
|---|---:|---:|---:|---:|---:|---|
| control (move 51's own field) | 0 / 0 | **179,285 B** sha `42e47d0b…` | 0 | — | — | **byte-identical, 2 processes** |
| full candidate field | 271 / 271 | **179,842 B** sha `511ea68d…` | **+557 B** | **16.443** | 1.00156 | **byte-identical, 2 processes** |
| admitted subset | 26 / 26 | **179,327 B** sha `5f31404f…` | **+42 B** | **12.923** | 1.01256 | **byte-identical, 2 processes** |
| the built candidate (subset tail + re-solved carrier) | 26 / 26 | **179,332 B** sha `ae59c510…` | **+47 B** | — | — | the staged archive reproduces the pricer's repack byte for byte |

The container did not draw a lottery here either: realized ÷ first-order ideal is 1.00156 on
the full field and 1.01256 on the subset, so the ±34.8 B break term is absent on this object
at this edit shape. That is a measurement on two fields, not a law.

### The three-leg Lagrange admission

26 of 271 pairs kept, on the RESOLVED pose, with the rate leg taken from the MEASURED
per-frame bit ledgers rather than apportioned (`rate_leg_is_modelled_not_measured: false`).
The sweep is FLAT near its optimum — 20 to 28 pairs all land within 8e−07 S of each other —
and 26 pairs at λ = 1.2589 is the argmin. The per-pair measured bit cost among the admitted
runs from **−16.53 to +28.06 bits** (median 12.71): some admitted edits SHORTEN the stream.

The ledger's modelled +39.43 B became **+42 B** by the subset's own real encode — the
ledger under-charged by 2.57 B (+0.0014 % of the archive), the same direction and roughly the
same size as sj1's measured +19.6 B on a 370-pair subset.

### The candidate, and its legs on the SHIPPED bytes

| leg | value | how |
|---|---:|---|
| rate | 0.11940981778130523 | **179,332 B EXACT** — the subset's own real encode (+42 B) plus the carrier splice (+5 B over 26 pairs / 259 coordinates) |
| seg | 0.010304151712216 | **12,147 flipped cells MEASURED on the candidate's own cold parse-back** — the admission predicted 12,147, **zero cells disagreeing** |
| pose | 0.006486552077248697 | RESOLVED 4.207535785085938e−06, re-verified n600 on the composed object and **re-measured bit-identically on the shipped decode** |
| **S projected** | **0.13620052157076992** | |
| **net vs move 51** | **−3.2810e−05** | **1.64 bars**; **1.42× the 34.8 B container-break sd** (2.3172e−05 S) |

Archive sha **`ae59c5109597968409c4b581337b8e23b5b3c4262fd3710c315cc83913b3b20e`**, 179,332 B.

**Proved on the shipped bytes, not on the plan:**

* the candidate's own cold parse-back decodes a token plane **byte-identical to the admitted
  field** (`decoded_field_matches_admitted: true`), raw sha `ccb89e3e…`, 3,662,409,600 B;
* the seg leg above is measured on THAT decode, not on the overlay;
* **the pose leg above is ALSO measured on that decode** — `up2.measure_pose` at batch 8 on
  the candidate's own `0.raw` with the candidate's own carrier returns
  **4.207535785085938e-06**, equal to the admission's composed value with a maximum absolute
  per-pair difference of **0.000e+00 over all 600 pairs**. Both scorer legs are in-loop
  numbers measured on the object that ships;
* the public entrypoint smoke is **symmetric** between candidate and frontier — both reach the
  CUDA gate through `inflate.sh` in 1.88 / 1.87 s and both reach the token decode through the
  public path inside the 240 s bound, with the four native-library exports set as
  `inflate.sh` sets them;
* the literal census is **CLEAR at rule 118**: 51 files, **three differ from the pointer tree**
  — `archive.zip`, `inflate.py` and the derived `MANIFEST.sha256` — and the ENTIRE `inflate.py`
  diff is the two pins that name the archive:

```
-ARCHIVE_SHA256 = "42e47d0bae1b0647d08db8a5061fe3eec6368b2169d89eb62b14f832fdb0978f"
-ARCHIVE_BYTES = 179285
+ARCHIVE_SHA256 = "ae59c5109597968409c4b581337b8e23b5b3c4262fd3710c315cc83913b3b20e"
+ARCHIVE_BYTES = 179332
```

**And one measurement that bears on the timing route:** the candidate's cold n600 local decode
is **934.626 s** against move 51's **999.008 s** on the same host and the same receiver — the
candidate decodes **6.4 % FASTER** than the row it is built on, on 47 MORE archive bytes. The
same sign pd2 measured (11.2 % faster on +90 B).

---

## 8. The seal — the NORMAL inheritance route

Move 51's own T4 inflate MEASURED **1,164.896041336 s**, under the 1,260 s `t4_direct` policy
limit, and MAIN minted its leg at harvest. This candidate's receiver is byte-identical in
behaviour to move 51's, so the ordinary route applies and was taken.

**Seal inputs staged from OUTSIDE the candidate tree** (the manifest must not verify itself):

| input | what |
|---|---|
| `seal_inputs/CANDIDATE_MANIFEST.json` | 51 rows regenerated from the bytes on disk through `tac.candidate_seal.measure_runtime_digest`, with the normalized-receiver digest |
| `seal_inputs/MANIFEST_VALIDATION.json` | `all_hashes_passed: true`, `all_runtime_dependencies_listed: true`, plus the Catalog #420 derived-listing re-derivation `tac.decode_wall_clock.validate_receiver_manifest` on the tree |
| `seal_inputs/TWIN_ENCODE.json` + `ENCODER_EXECUTION_{0,1}.json` | two independent encoder PROCESSES, each with its own natively compiled rc64 backend, produced **byte-identical members** (179,227 B, sha `ebe0d40c…`) from distinct paths |
| `PUBLIC_SMOKE.json` | the symmetric candidate/frontier entrypoint smoke measured in §7 |
| `parseback/PARSEBACK_RESULT.json` | the cold decode receipt |
| receiver pins | `inflate.py` and `inflate.sh` |
| retained paths | the whole store, hashed in `RETENTION_MANIFEST.json` |
| falsifiers | the nine pre-registered rows, each carrying its MEASURED outcome |

**The seal**

```
/Volumes/APDataStore/pact/ddm_pd3/SEAL_ddm_pd3_pose_directed_pass3_contest_cuda.json
file sha  9905cb9238cccf29f738ab71c28a073eb06bc90512e777db509ce49d3c45bf9e   (10,851 B)
seal sha  c2ef42665b4e0382c7382a36d92a69059db879bc7728196bd50b558baa47ff9a
VALIDATED SEAL_VALID, problems ()   (validate_seal, require_decode_wall_clock=True)
```

candidate `ddm_pd3_pose_directed_pass3`, axis `contest_cuda`; archive 179,332 B sha
`ae59c510…`; runtime 51 files / 1,008,691 B, digest `08440559…`; leg mode **inherited**, scope
*"identical normalized receiver code; candidate payload-dependent time not remeasured"*,
source the move-51 sidecar sha `512b9f9e…` at 1,164.896 s ≤ 1,260; admit bar **net ΔS <
−2e−05** against the LIVE pointer move 51 (`42e47d0b…`, 0.1362333315680336) at tolerance 0.

**The pr18 behaviour digest matches on both sides: `9f6e71680a13d859…`.** The legacy raw
digests differ (`e854ac98…` against `27b95609…`) exactly as they must — that digest covers
the two archive pins and the derived listing, which move with the archive by construction.
`behavior_digests_equal: true` is the check the inheritance rests on.

**The candidate was NOT changed to seal it** — the archive, the runtime digest and both pins
are the same objects §7 measured. **NO Modal, NO fire, NO completion, NO packet: MAIN fires.**

---

## 9. What this does NOT claim

1. **No score of any kind.** Every S here is a PROJECTION on measured legs. Only
   `upstream/evaluate.py` on the shipped bytes is a score, and MAIN fires. This arm ran no
   Modal call, wrote no authorization, no completion and no packet.
2. **The pose numbers are `[macOS-CPU advisory]`,** measured on a frozen CPU-torch PoseNet
   against DALI-lineage GT. The instrument's base is 4.281953276180728e−06 against the T4
   print 4.29e−06 (ratio 0.998124); the exact row decides.
3. **The seg leg is carried onto T4 by the same-instrument ratio** (1.0006809878169245), not
   measured there. What IS measured on the shipped bytes is the instrument leg: 12,147 cells.
4. **The margin is thinner than pd2's.** −3.2810e−05 is **1.64 bars** and **1.42×** the
   campaign's standing 34.8 B container-break sd. The T4 pose-print class alone moved pd2's
   realized row +6.1e−06 off its projection; at this candidate's resolved 4.2075e−06 the
   print rounds to 4.21e−06, worth about +1.9e−06. Both are smaller than the margin, but the
   margin is not large.
5. **verdict_scope: FORMULATION, on this object.** What is measured is pose-directed single-
   token pre-distortion on move 51's field, proposals ranked by pose saliency masked to
   argmax-interior cells, refined 12-deep, admitted on the resolved pose. Multi-token edits
   per pair, a clustered search, and any other carrier parametrisation are untouched.
6. **The population is DERIVED-bounded, not exhausted.** All 283 pairs of the three strata
   were searched to completion, but the 314 pairs below both tiers were excluded by the
   DERIVED bound `|credit| ≤ base`, which says they cannot pay at 15.400 bits/token — not
   that no edit there does anything.
7. **12.923 bits/token is THIS subset at THIS edit shape**, measured once, as 16.443 is this
   field's. Neither is a law.
8. **The per-bit ranking prices a PAIR, not a proposal.** The rlc1 ledger's `delta_bits` is
   per pair, so two proposals on one pair are charged the same.
9. **pp1's twelve floor pairs stay excluded** on pp1's measurement. Nothing here reopens them.
10. **The seal is a seal, not a score.** `SEAL_VALID` says the candidate's identity, timing
    leg, smoke, pins and admit bar are coherent; it says nothing about what the T4 will
    measure.
11. **This arm fired nothing.** No Modal call, no authorization, no completion, no packet.
    MAIN fires.

---

## 10. Custody

Store **`/Volumes/APDataStore/pact/ddm_pd3/`**. APDataStore rather than Vertigo because
Vertigo held **38 GiB** free — below its 40 GiB reserve. **The reserve was never lowered and
nothing was written there.** APDataStore free space MEASURED at each heavy step: 36 GiB at
launch, 33 GiB after the overlay, **29 GiB at landing**.

**MEASURED: 6,025,760,627 B over 740 files, every one hashed — under the 8 GiB cap.**
`RETENTION_MANIFEST.json` carries bytes and sha256 for each, including the losers: all 2,647
realized search rows and all 5,226 screened proposals, not only the 26 that shipped.

| path | what |
|---|---|
| `candidate/candidate_archive.zip` · `candidate/candidate_runtime/` | **179,332 B sha `ae59c510…`** and the 51-file runtime |
| `parseback/0.raw` · `parseback/PARSEBACK_RESULT.json` | the cold decode (3,662,409,600 B sha `ccb89e3e…`) and its receipt |
| `seg_final/argmax_n600.npy` · `STEP0_RESULT.json` | the seg leg on that decode: 12,147 cells |
| `pose/overlay_pd3/` | the 600-pair odd-frame overlay the pose leg scored |
| `pose/pose_stale.npy` · `pose/pose_resolved.npy` · `pose/pose_on_decode.npy` · `base/pose_base_move51.npy` | the four n600 pose vectors |
| `refine/refine_rows_*.jsonl` · `refine/codes_resolved.npy` | the composed re-solve, per pair |
| `search/` · `smoke/` | **every realized row and every screened proposal, winners and losers** |
| `assemble/` · `assemble_perbit/` · `admission/` · `stage/` | the candidate field, the per-bit carry ranking, the Lagrange sweep and its trace, the staged tail |
| `rlc1_price/` | the pricer's INPUTS, both control encodes, both full-field encodes, both subset encodes, the per-pair bit ledger |
| `BIND_RECEIPT.json` · `PREREGISTRATION.json` · `SEARCH_PLAN.json` · `SMOKE_TIMING.json` · `STOP_RULE.json` · `F2_COMPOSITION_CONTROL.json` · `F3_BASE_BAND_CONTROL.json` · `F4_CONTROL_ENCODE.json` · `F8_DEPTH_CONTROL.json` · `F9_POSE_ON_DECODE.json` · `YIELD_HISTOGRAM.json` · `PUBLIC_SMOKE.json` · `LITERAL_CENSUS.json` | the pre-registration and every control receipt |
| `SEAL_ddm_pd3_pose_directed_pass3_contest_cuda.json` | **SEAL_VALID**, file sha `9905cb92…` (10,851 B) |
| `seal_inputs/` | the candidate manifest regenerated from outside the tree, its validation, the twin-encode receipt and both encoder-execution receipts |
| `PIPELINE_PLAN.md` · `make_retention.py` · `make_census.py` · `make_seal_inputs.py` · `pose_on_decode.py` | the chain as exact commands and the four one-off producers |

Producer: `experiments/ddm_pd3_pose_directed_pass3.py` (`bind | base | prereg | plan | run |
carry`), ruff-clean, plus pd1's `ddm_pd1_pose_directed.py search|assemble` and
`ddm_pd1_candidate_tree.py build`, pd2's `ddm_pd2_pose_directed_pass2.py` helpers, and sj1's
`ddm_sj1_rlc1_price.py` and `ddm_sj1_joint_admission.py`, all unchanged. Nothing under
`/Volumes/APDataStore/pact/ddm_pd1`, `ddm_pd2`, `/Volumes/VertigoDataTier/pact/ddm_sj1_pass7`,
`ddm_sj1_pass8`, `ddm_pp1`, `ddm_cb1`, `ddm_cr1` or `ddm_so2` was written; every module that
imports from a custody tree sets `sys.dont_write_bytecode = True`.

## 11. What this hands the next arm

1. **The candidate is built, byte-closed and SEALED; the only thing left is the fire.** It
   inherits move 51's own measured leg on the ordinary route. A move-52 successor will need
   its OWN leg minted at harvest (pr19: this route inherits ONCE from a measured leg).
2. **The single-token family is visibly draining on this object.** Three passes on the same
   field: pd1 admitted 20 of 41 walked, pd2 40 of 166, pd3 **26 of 283**. The admitted count
   per walked pair fell from 0.49 to 0.24 to **0.09**, and the net per pass from −5.0e−05
   (pd1, realized) to −5.0e−05 (pd2, realized) to **−3.3e−05 projected**. The 314 pairs below
   both tiers cannot pay at 15.4 bits/token with ONE token.
3. **The price is what is left to attack.** 12.923 bits for an isolated token in this subset;
   pass 8's **8.93 bits/token on a CLUSTERED field** is still the number to beat, and
   clustering — two tokens on one pair, or edits packed into one coder context — is the one
   lever both pd1's §11 and pd2's §12 named and neither arm has run.
4. **Tier 2 is now MEASURED, not estimated, and it is poor.** 4 admitted of 100 walked, a
   quarter of its prior. A pass-4 that re-uses this arm's ranking should demote tier 2 and
   spend the budget on depth or on the clustered price instead.
5. **Depth is not exhausted but it is decelerating.** K=8→12 bought +19.5 % of summed credit
   on the shared pairs; 7 of 8 smoke pairs still hit the K=12 cap, so K=16 has something —
   but the yield per refine is falling and the wall per pair is not.

<!-- # FORMALIZATION_PENDING: the equations leg is written by tools/pointer_move_packet.py --equations-leg at harvest, and only on an exact row. The score arithmetic used throughout is the registered S = 100*d_seg + sqrt(10*d_pose) + 25*B/37,545,489. -->
