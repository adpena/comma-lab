# ddm_pd5 — MULTI-TOKEN joint proposals on the move-52 field, priced as a SET

Tokens: `[no-triality] [p0-ledger-ok]`

Lane `ddm_pd5_multitoken_joint_proposals_set_priced_20260916`. Base: **move 52**,
S 0.13620226906030858 @ 179,332 B, archive sha
`ae59c5109597968409c4b581337b8e23b5b3c4262fd3710c315cc83913b3b20e`. Axis
`[macOS-CPU advisory, frozen CPU-torch PoseNet + SegNet, DALI GT lineage, n600]` for the
distortion legs; bytes EXACT through the shipped RLC1 coder and the shipped container.
**No score is claimed. Only `upstream/evaluate.py` on shipped bytes is a score, and MAIN
fires.** No Modal, no authorization, no fire, no completion, no packet.

Pose solver, stated once: every per-pair credit comes from
`ddm_jg5_pose_resolve_on_edited_renders.refine_pair` (pd1's/pd2's/pd3's/pd4's same object at
these budgets), driven at PoseNet batch 1 on the moved render; every n600 number comes from
`ddm_sj1_joint_admission`'s `pose` stage (`ddm_up2_shipping_pose_solve.measure_pose`, batch 8)
on the composed overlay this arm rendered.

---

## 0. The verdict, first

**The price lever hit its target and the bar still did not fall.**

pd4 DERIVED, by bisection on its own pool, that 10.891 bits per changed token would clear the
−2e−05 bar. This arm put THREE candidate sets through their own exact encodes; the cheapest
priced at **8.976 bits per changed token** — better than pd4's derived target, better than
pd4's measured 12.000, and within 0.5 % of pass 8's clustered 8.93 — and the best net any of
them reaches is **−1.766e−05, 0.883 bars**.

| set | pairs / tokens | exact archive | Δ B | bits/token | rate | seg | pose | **net** | **bars** |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| **00** (ledger optimum) | **22 / 41** | **179,378 B** `db78a913…` | **+46** | **8.976** | +3.06295e−05 | +2.54483e−06 | **−5.08317e−05** | **−1.765740e−05** | **0.883** |
| 01 (the λ-sweep's admitted set) | 17 / 34 | 179,373 B `af268709…` | +41 | 9.647 | +2.73002e−05 | +2.54483e−06 | −4.57364e−05 | −1.589134e−05 | 0.795 |
| 02 (the set-price fixed point) | 16 / 33 | 179,371 B `9e186f4f…` | +39 | 9.455 | +2.59685e−05 | +2.54483e−06 | −4.43557e−05 | −1.584236e−05 | 0.792 |

| falsifier | outcome |
|---|---|
| F1 the pricer reproduces move 52's own archive | **does not fire** — 179,332 B sha `ae59c510…`, twins byte-identical in TWO processes, stream 119,097 B |
| F2 the base is move 52's own decode | **does not fire** — sha `ccb89e3e…eced`, re-hashed by this arm |
| F3 the per-proposal price resolves above its noise | **does not fire** — noise median 2.905 bits (106 repeated proposals) against a within-pair range of median 9.555 bits |
| F4 3- and 4-token runs are not cheaper per token | **does not fire** — median bits/token 14.834 → 10.363 → 9.209 → **8.659** by run length |
| F5 the admitted set's real bits/token ≥ 11.0 | **does not fire** — **8.976 / 9.647 / 9.455** across the three sets |
| F6 composition is summed, not realized | **does not fire** — realized fraction **1.0219**; 577 of 578 unedited pairs bit-identical |
| F7 the set re-price differs from the ledger by > 10 % after iteration | **does not fire** — **+8.11 % → −1.08 % → +0.38 %**, fixed point at iteration 3 |
| **F8 the admitted set clears −2e−05** | **FIRES on all three sets — best −1.766e−05, 0.883 bars** |

**F8 firing CLOSES the multi-token joint-proposal formulation on this object**, per the
charter's pre-registered rule. Say it plainly: three waves of search, 1,441 realized rows,
718 real-encode prices and **three** exact archives later, **the pointer did not move and this
arm did not build a candidate.** What the pass buys the campaign is not a row; it is the
knowledge that the price corner is now spent, and WHY.

**And one thing that only the third archive could show.** The set-price iteration SHRANK the
set — 22 → 17 → 16 pairs — and every shrink made the EXACT net WORSE: 0.883 → 0.795 → 0.792
bars. The iteration converges the ledger onto the truth (§6) but its objective is frame-local
bits, and frame-local bits are a LESS complete picture of a SMALLER set: the spill onto
unedited frames runs **10.4 % → 19.2 % → 19.6 %** across the three encodes. So the cure for
pd4's selection bias introduces a new bias of its own at the set-size margin — it over-charges
the marginal pair and drops pairs the container was in fact carrying almost free. **The five
pairs iteration 1 dropped cost 5 bytes and carried −5.10e−06 S of pose.** The lesson is not
"don't iterate"; it is **iterate to converge the ledger, then price the SIZE ladder on real
archives and take the best** — which is what this arm did, and which cost two extra encodes.

---

## 1. The binding, and an inherited decode re-verified rather than re-run

`pd4.bind_move52` re-points pp1, sj1, pd1, pd2 and pd3 — five modules that pin moves 49/50/51
in import-time globals — and re-derives move 52's score from its three components
(0.13620226906030858 exactly). This arm added its own gate on top and then made one
deliberate, declared choice.

**The decode is INHERITED, not re-run.** pd3 and pd4 each parsed the SAME shipped bytes back
cold on their own copies of the runtime tree and produced a bit-identical `0.raw`; pd4
MEASURED max abs per-pair pose difference **0.000e+00** against pd3's vector over all 600
pairs. This arm re-hashed that raw (3,662,409,600 B, sha `ccb89e3e73bf61ac…`) and stands on
it. A third decode would have added 3.66 GB to a store with 22 GiB free and could only have
reproduced the same sha. Receipt: `BIND_RECEIPT.json.raw_reverified`; the choice and its
warrant are recorded in the same file under `decode_provenance`.

**The pricer's control identity passed in this arm's own store, twice.** Two independent
processes re-encoded move 52's OWN field through the shipped RLC1 loop and packed archives at
**179,332 B sha `ae59c5109597968409c4b581337b8e23b5b3c4262fd3710c315cc83913b3b20e`** —
byte-identical to move 52's own archive, `archive_matches_live_pointer: true`, stream 119,097
B, decoded plane sha `117951bbd6800949…`. And the control's per-frame bit ledger is
**bit-identical to pd4's** (max |Δ| = **0.000e+00** over 600 frames), two arms, two stores,
one coder. Receipts: `controls/CONTROL_IDENTITY.json`, `rlc1_price/retained/bits_rlc1_control.npy`.

---

## 2. The prior, read off pd4's own rows before this arm searched anything

Written into `PREREGISTRATION.json` before a single pd5 row existed, from a read-only
re-analysis of pd4's retained `sheets/priced_rows.jsonl` (122 matched cluster/anchor pairs):

| quantity | measured on pd4's rows |
|---|---:|
| isolated-token price, median | 15.420 bits |
| **marginal** bits of the SECOND token, median | **6.213 bits** |
| marginal credit of the second token, median | **+3.504e−08 d_pose — WORSE** |
| fraction of second tokens that IMPROVE the pair's credit | **41.0 %** |

That is the whole shape of the bet in two rows: the coder's context model really does discount
a neighbour by ~2.5×, and the pose credit of that neighbour is a coin flip biased the wrong
way. The pre-registered prediction said the family would land near −1.2e−05 … −1.6e−05 and
the bar would not fall. **It landed at −1.589e−05.**

---

## 3. The timing smoke and the declared budget

Eight one-pair shards at 2 threads each, run CONCURRENTLY with the two RLC1 control encodes
and the price-store init — the fleet condition the waves actually ran in.

| quantity | measured |
|---|---:|
| seconds per refine | **33.86** (pd4 measured 35.0 on its own smoke) |
| shard wall, min / median / mean / max | 20.5 / 291.1 / 254.0 / 466.1 s |
| projected 47-pair 2→3 wave at 5 shards | 39.8 min (mean basis), 73.0 min (slowest-shard) |
| projected 101-pair 1→2 wave at 5 shards | 85.5 min (mean basis), 156.9 min (slowest-shard) |

**K_refine = 12 is DECLARED and held FIXED** — pd3 MEASURED K=12 strictly better than K=8 on
24 of 125 shared pairs and worse on none, and depth is not this arm's declared delta.
Receipt: `SMOKE_TIMING.json`.

---

## 4. The searches: one code path, three waves, no prefix stop

The search is pd4's `cluster-search` with ONE declared generalisation: the anchor is a SEED RUN
of any length, so the same code grows 1→2, 2→3 and 3→4. Every row is realized JOINTLY — one
render, one frozen-argmax, one carrier re-solve — and credited against the PAIR's own base,
never summed from its parts.

| wave | pairs | screened | refined | shard wall (sum) |
|---|---:|---:|---:|---:|
| 1→2 (completes pd4's 45 %-prefix cluster family to 100 %) | 101 | 776 | 569 | 19,524.9 s |
| 2→3 (the rung) | 47 | 565 | 403 | 14,454.1 s |
| 3→4 (the rung) | 44 | 638 | 409 | 11,784.7 s |
| smoke | 8 | 91 | 60 | 2,031.9 s |
| **total** | **200 pair-visits over 146 distinct pairs** | **2,070** | **1,441** | **47,795.6 s** |

**The one semantic change this arm made to a sister's code is LOAD-BEARING, and that was
checked rather than assumed.** pd4's `load_pool` hard-codes `"ddm_pd4" in store.parts` to mean
"measured on move 52's renders"; `load_pool_move52` lifts that marker to a parameter. MEASURED:
**36** of this arm's proposals sit on the 26 pairs move 52 itself edited and would have been
dropped as stale under the hard-coded marker — and **three of them are in the admitted set**,
including **pair 324**, the single largest asset in it (52.51 bits, credit −5.486e−06). The
base band does not carry them: pd4 MEASURED it as a 96.2 %-effective proxy. The admitted set's
row provenance is 7 `ddm_pd5_run` · 5 `ddm_pd4_cluster` · 4 `ddm_pd4` · 1 `ddm_pd3`.
Receipt: `controls/FRESH_STORE_CONTROL.json`.

**Every wave ran to completion. There is no prefix stop in this pass** — pd4's cluster verdict
was scoped to a 45 % prefix of its 140-pair anchor population; this arm walked the remaining
101 pairs, so that scope caveat is now closed.

MEASURED, before any price: the added token improves the pair's own pose credit on **64 of 117
pairs (54.7 %)** when you take the best of up to 12 refined extensions — better than the 41 %
per-proposal rate, because the search gets to choose. But the winning extension usually costs
SegNet cells: of those 117 best rows, **43 cost +2 cells, 35 cost +1**, 35 cost 0 and 4 repaid
one. At 8.483e−07 S per cell, a +2-cell extension spends 1.697e−06 S before it has paid a
single bit — which is most of what a median pair's pose credit is worth.

---

## 5. THE PRICE, by run length — the charter's headline

Eight sheets, 156 pairs, **718 proposals**, one real 600-frame encode each, twins byte-identical
inside every encode. Sheet archives came back at **179,719 – 179,766 B (+387 … +434 B)** over move 52,
carrying 236–257 tokens each, i.e. **12.900 – 14.271 bits per changed token at sheet density**.

| run length | proposals | tokens | **bits/token pooled** | **median** | min | max |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 344 | 344 | 15.485 | **14.834** | −18.277 | 33.608 |
| 2 | 216 | 432 | 11.860 | **10.363** | −3.035 | 30.097 |
| 3 | 84 | 252 | 11.878 | **9.209** | 4.467 | 28.936 |
| 4 | 74 | 296 | 11.401 | **8.659** | 4.409 | 29.034 |

**The price falls monotonically with run length by median — 14.834 → 10.363 → 9.209 → 8.659 —
and the 4-token median is BELOW pass 8's clustered 8.93.** F4 does not fire. The POOLED figures
are flat from 2 to 4 because a handful of expensive outliers dominate the sum; the median is
the number that describes a proposal you would actually pick.

The marginal token stays cheap, and its credit stays a coin flip:

| the added token | marginal bits (median) | marginal bits (mean) | marginal credit (median) | improves credit |
|---|---:|---:|---:|---:|
| 2nd | 5.625 | 7.094 | +1.860e−08 (worse) | 43.3 % |
| 3rd | 6.596 | 9.012 | +1.617e−08 (worse) | 46.1 % |
| 4th | 4.722 | 8.036 | +3.273e−08 (worse) | 37.1 % |

**The context discount compounds; the credit does not.** That asymmetry is the finding of the
pass, and it is why a cheaper price did not become a lower score.

**The price resolves.** Held-fixed proposals measured in ≥2 sheets that differ at other pairs:
spread median **2.905** bits (mean 2.932, max 6.377, 106 proposals), against a within-pair
price range of median **9.555** bits over 128 pairs. Signal 3.3× the noise. F3 does not fire.

---

## 6. SET pricing — the cure for pd4's 36.3 % selection bias, MEASURED

pd4's ledger under-charged its chosen subset by **36.3 %** because the per-bit rule takes the
argmin of a few noisy real prices and the argmin of noisy draws is biased low. The cure is to
re-encode the SELECTED SET as one field and re-run the admission on the set's own real bits.

| iteration | pairs | tokens | ledger bits | real frame-local bits | **residual bias** | exact archive |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 22 | 41 | 308.618 | 333.660 | **+8.11 %** | 179,378 B sha `db78a913…` (+46 B) |
| 1 | 17 | 34 | 268.857 | 265.967 | **−1.08 %** | **179,373 B sha `af268709…` (+41 B)** |
| 2 | 16 | 33 | 248.271 | 249.205 | **+0.38 %** | 179,371 B sha `9e186f4f…` (+39 B) |
| 3 | 16 | 33 | — | — | — | **`set_repeats_previous: true` — FIXED POINT** |

**ONE iteration takes the residual from +8.11 % to −1.08 %** — inside the 2.9-bit noise floor
on a 266-bit set — and the loop reaches a fixed point at iteration 3. F7 does not fire, and
pd4's 36.3 % is not a property of the estimator: it was a property of choosing with it. Two
things did the work: eight sheets instead of six (more independent draws per pair), and the
iteration itself.

**But the converged set is not the best set** (§0). Frame-local spill runs 10.4 % / 19.2 % /
19.6 % across the three encodes — the rail is measurably less complete on a smaller edit set —
so the iteration's own objective over-charges the marginal pair. The honest procedure this
pass arrives at: **converge the ledger, then price the SIZE ladder on real archives.**

Iteration ≥ 1 may only DROP pairs, never swap a pair's proposal. That is enforced in code
(`carry-from-set` pins the candidate space to one iteration's choices) because the realization
renders the SPECIFIC edits it is given: a post-render swap would measure the composed pose leg
on a render that does not exist.

**The independent control that matters most.** `ddm_sj1_joint_admission`'s λ-sweep over the
COMPOSED object (λ = 1.2735, rate leg from the measured per-frame ledgers) and this arm's
set-price iteration reached the **byte-identical admitted field**, sha
`c341107aee7228046b64fb19d2498223d03e79a260c7f8d10728ecfaa3a7cbae` — 17 pairs, two routes, one
object. A second, earlier control: pd5's simple per-pair net rule reproduces pd4's OWN 18-pair
λ-swept set EXACTLY on pd4's own priced rows (`controls/SWEEP_RULE_CONTROL.json`).

---

## 7. The composition, re-verified on one object

A 600-pair overlay rendered from the 22-pair carried field (1,831,204,800 B, 182.2 s), an n600
pose leg on it, and a carrier re-solve over all 600 pairs starting from the SHIPPED codes.

| leg | value |
|---|---:|
| per-pair sum of credits | −3.71334e−05 |
| **composed, re-verified** | **−3.79448e−05** |
| **realized fraction** | **1.0219** |
| stale pose (candidate renders, shipped carrier) | 2.3466e−04 = **55.8× base** |
| unedited pairs bit-identical to base | **577 / 578** |
| coordinates changed vs shipped | 213 |

F6 does not fire. The composition realized **more** than the per-pair sum, and the stale
control shows the carrier re-solve is doing 55.8× of real work. One unedited pair again moved
under a carrier-only re-solve — pd4 saw the same single pair; it is not banked here either.

---

## 8. The three legs, on the admitted subset's OWN exact archive bytes

| field | pairs / tokens | exact archive | Δ vs move 52 | bits/changed token | twins |
|---|---:|---|---:|---:|---|
| control (move 52's own field) | 0 / 0 | **179,332 B** sha `ae59c510…3b20e` | 0 | — | byte-identical, **2 processes** |
| the eight price sheets | 156 / 236–257 | 179,719 – 179,766 B | +387 … +434 | 12.900 – 14.271 | byte-identical within each |
| set 00 | 22 / 41 | 179,378 B sha `db78a913…` | +46 | 8.976 | byte-identical in-process |
| **the ADMITTED subset (set 01)** | **17 / 34** | **179,373 B sha `af26870961d59b3def02d20304a4264d8ed048b75e637813962aedb1cbeb1d96`** | **+41** | **9.647** | byte-identical, **2 processes** |

| leg | value | how |
|---|---:|---|
| rate | **+2.73002e−05** | **+41 B EXACT**, the subset's own real encode |
| seg | **+2.54483e−06** | **+3 flipped cells** — the pose-directed edits COST seg here |
| pose | **−4.57364e−05** | RESOLVED 4.1508575238478585e−06 against the base 4.207535785085938e−06, re-verified n600 on the composed object at 1.0219 of the per-pair sum |
| **S projected** | **0.13618637772083570** | 100·0.00010306544825882935 + √(10·4.1508575238478585e−06) + 25·179,373/37,545,489 |
| **net vs move 52** | **−1.5891339472884347e−05** | **0.795 bars** · **0.686× the 34.8 B container-break sd** |

**F8 FIRES on the admitted set, and on both of its neighbours** — the exact ladder is in §0:
0.883 / 0.795 / 0.792 bars for 22 / 17 / 16 pairs. The best margin, −1.766e−05, is **0.762× the
34.8 B container-break sd**, so the sign of a single re-pack could swallow it. No candidate was
built, no archive staged, no parse-back run, nothing sealed — pass 8's and pd4's precedent,
applied.

The admitted set's composition by run length: **1-token 5 · 2-token 9 · 3-token 1 · 4-token 2**
— 12 of 17 admitted pairs carry a multi-token run, and 3 carry a run of 3 or 4. The longer runs
are not decoration; they are a third of the admitted mass and they are the cheapest rows in it
(pair 14, 4 tokens at 5.55 bits/token; pair 221, 3 tokens at 5.44).

**The tail is MEASURED, not bounded.** Iteration 2's own exact archive is **179,371 B sha
`9e186f4f7121701249cafd86187d9fec53e1b470d4c30d220761c94db99ca3b6`, +39 B**, and its net is
**−1.584236e−05 (0.792 bars)** — the worst of the three, not the best. Iteration 3 repeats
iteration 2's set, so the loop is at a fixed point and the ladder is complete. Receipts:
`SET_LADDER.json`, `setprice/STATE.json`.

---

## 9. What the pass settles

1. **The price lever is now SPENT on this object.** pd4 asked for 10.891 bits/token; pd5
   delivered **8.976** on its cheapest real archive and **8.659** median on 4-token runs, and
   the bar did not fall. pd4's 10.891 was a bisection on a ladder that assumed each pair's
   best proposal keeps its credit at a lower price; the composed, admitted credit is smaller
   than that ladder implies. **Read pd4's threshold as retired: a price target is not a score.**
2. **Longer runs are genuinely cheaper and genuinely no better at buying pose.** Marginal bits
   5.6 / 6.6 / 4.7; marginal credit improves on 43 % / 46 % / 37 %. Every extension is a fair
   coin on the pose axis and a loaded one on the seg axis (+2 cells on 43 of 117 best rows).
   The discount compounds; the credit does not. **That asymmetry, not the price, is the wall.**
3. **Set pricing WORKS as a LEDGER correction and MISLEADS as a selection rule.** The residual
   goes 36.3 % → 8.11 % → −1.08 % → +0.38 % and reaches a fixed point in three iterations — so
   any successor that ranks by a measured per-proposal price must re-encode the chosen SET
   before believing its number. But the converged set is the WORST of the three on exact bytes
   (0.792 bars against 0.883 for the unconverged ledger optimum), because frame-local spill
   grows as the edit set shrinks (10.4 % → 19.2 % → 19.6 %). **Converge the ledger; then price
   the SIZE ladder on real archives and take the best.**
4. **pd4's 45 %-prefix cluster scope is closed.** The remaining 101 pairs were walked; the
   2-token family's verdict is now population-wide.
5. **The pool is not short of credit and it is no longer short of price.** 22 of 156 pairs pay
   at the sheet price and all 22 survive their own exact encode. What is short is the JOINT
   quantity: those 22 pairs together remove **1.50 %** of the n600 pose mean, and that, minus
   46 bytes and 3 seg cells, is 0.883 bars.

**verdict_scope: FORMULATION, on the move-52 field.** What is MEASURED and closed is
pose-directed token pre-distortion in runs of 1–4 tokens grown along the pose-saliency ridge
by 8-neighbourhood extension, masked to the argmax interior at `max-cells 2`, refined K=12 by
jg5, priced by real encode and admitted as a SET on the composed resolved pose. Untouched, and
NOT closed by this pass: runs grown by a PRICE-first generator (this arm still generated by
pose saliency and priced afterwards — pd4's second named lever, still unrun); runs longer than
4; a relaxed `max-cells` with the seg debt paid elsewhere; and cross-PAIR packing into adjacent
coder contexts.

---

## 10. What this does NOT claim

1. **No score of any kind.** Every S here is a PROJECTION on measured legs. Only
   `upstream/evaluate.py` on shipped bytes is a score, and MAIN fires. This arm ran no Modal
   call, wrote no authorization, no completion and no packet.
2. **No candidate, no archive, no seal.** `179,373 B sha af268709…` is the PRICER's archive —
   the object that measured the rate leg — not a staged candidate: it carries move 52's own
   carrier, not the re-solved one, and no runtime tree was built around it. The move-52 leg
   (`SEAL_ddm_pd3_…decode_wall_clock.json`, 1,067.8 s) remains **unconsumed**.
3. **The seg leg was never measured on a decode**, because no candidate was decoded. It is the
   jg1 DALI instrument's count carried onto T4 by the same-instrument ratio 1.0006662543837985.
   pd4's F9 ("seg on the shipped bytes equals the admission's") is again **not reached**.
4. **The pose numbers are `[macOS-CPU advisory]`**, frozen CPU-torch PoseNet against
   DALI-lineage GT. This arm's base is 4.207535785085938e−06 against the T4 print 4.21e−06.
5. **9.647 bits/token is THIS subset at THIS edit shape on THIS field**, measured once, as
   15.485 is this pool's isolated-token pooled price. None is a law.
6. **Every net in §0 and §8 is MEASURED on that set's own real archive.** Nothing in the
   ladder is extrapolated. The pose leg of each set is the composed resolved vector restricted
   to that set, from ONE 600-pair re-solve on the 22-edit overlay; a pair's re-solve depends
   only on its own render, so it transfers across the three sets, but no set other than 22
   was independently re-rendered.
7. **The decode is inherited** (§1), with its warrant stated; this arm did not produce a third
   independent parse-back.
8. **pp1's twelve floor pairs stay excluded** on pp1's measurement. Nothing here reopens them.
9. **The one unedited pair whose carrier re-solve moved is NOT banked** and is counted nowhere.
10. **This arm fired nothing.** MAIN fires.

---

## 11. Custody

Store **`/Volumes/APDataStore/pact/ddm_pd5/`** — APDataStore, because Vertigo held **38 GiB**
free, below its 40 GiB reserve. **The reserve was never lowered and nothing was written there.**
APDataStore free space MEASURED at each heavy step: 22 GiB at start, 22 GiB before the waves,
21 GiB before the sheets, 18 GiB after the overlay.

| path | what |
|---|---|
| `BIND_RECEIPT.json` · `PREREGISTRATION.json` · `SMOKE_TIMING.json` | the binding with the re-verified inherited raw, the pre-registered prediction + falsifiers, the declared budget |
| `plan/` | `SEEDS_{1to2,2to3,3to4}.json`, the walk orders, the shard lists, the smoke picks, the row sources |
| `search/{one_to_two,two_to_three,three_to_four}/` · `smoke/` | **every realized row and every screened proposal** — 1,441 realized, 2,070 screened, winners and losers |
| `sheets/sheet_0{0..7}.npz` · `SHEETS.json` · `priced_rows.jsonl` · `PRICE_MERGE.json` | the eight price sheets, their manifest and the 718 per-proposal REAL prices |
| `rlc1_price/` · `rlc1_bulk/` | the pricer's inputs, both control encodes, all eight sheet encodes, both set encodes, the per-frame bit ledgers |
| `setprice/STATE.json` · `set_0{0,1,2}.npz` | the SET-price iteration, its fields and the measured residual at each step |
| `assemble/` · `admission/` | the carried rows pinned to set 00, the candidate field, the pass rows, the spliced ledger, the λ-sweep and its admitted field |
| `pose/overlay/` · `pose_stale.npy` · `pose_resolved.npy` · `refine/` | the 600-pair overlay, the two n600 vectors, the per-pair re-solve and the merged codes |
| `controls/CONTROL_IDENTITY.json` · `controls/SWEEP_RULE_CONTROL.json` · `controls/ADMITTED_SUBSET_TWINS.json` · `controls/FRESH_STORE_CONTROL.json` | the pricer's identity gate, the selection-rule cross-check against pd4, the admitted subset's cross-process twins, and the proof that the fresh-store parameter is load-bearing |
| `PRICE_BY_RUN_LENGTH.json` · `YIELD.json` · `THREE_LEG.json` · `SET_LADDER.json` | the headline, the admitted histogram, the exact three-leg arithmetic, and the exact ladder over all three sets |
| `PIPELINE_PLAN.md` · `progress.sh` · `three_leg.py` · `launch_joint.sh` | the chain as exact commands and this arm's own producers |

Producer: `experiments/ddm_pd5_multitoken.py` (`bind | prereg | seeds | run-search |
smoke-timing | sheets | price-merge | price-report | setprice | carry-from-set | setabsorb`),
ruff-clean, plus pd4's `carry`/`assemble`/`run`, sj1's `ddm_sj1_rlc1_price` and
`ddm_sj1_joint_admission`, all unchanged. Two functions are rebound PROCESS-LOCALLY for the
duration of one call and restored in `finally` — `pd4.load_pool` (so pd4's rows count as
current for move 52) and `pd4._encode_bits` (so pd4's merge reads `pd5sheet…` in pd5's own
store rather than a `pd4sheet…` name that would be a provenance lie inside a pd5 store). Both
swaps are declared in the module's docstrings. Nothing under `ddm_pd1`–`ddm_pd4`, `ddm_sj1` or
`/Volumes/VertigoDataTier/` was written; every module that imports from a custody tree sets
`sys.dont_write_bytecode = True`.

## 12. What this hands the next arm

1. **Do not buy price on this field again.** 9.647 bits/token is below every target this
   family has set itself and it bought 0.795 bars. The next bar has to come from CREDIT or
   from a different object, not from a cheaper token.
2. **The one price lever pd4 named that is still unrun is the PRICE-FIRST generator** — propose
   from the cheap half of the token plane and check credit afterwards, rather than proposing
   from pose saliency and pricing afterwards. This arm did the latter, like every pass before
   it. It is the last untried thing inside this formulation, and §9's closure is scoped to
   exclude it.
3. **Set pricing is cheap and it is now standard — but price the SIZE ladder too.** One extra
   encode per candidate set takes the ledger error from 36 % to 1 %; two more showed the
   converged set is not the best one. Never believe a chosen subset's ledger, and never
   believe the fixed point without pricing its neighbours.
4. **The seg screen is the binding constraint on extensions, not the coder.** 43 of 117 best
   extensions cost +2 cells. A formulation that pays the seg debt somewhere else — or that
   grows along the argmax interior rather than the pose ridge — sees a different pool.
5. **The composed pose over-realizes.** 1.0219 here, 0.9992 for pd4. Two arms, two fields;
   the per-pair sum is a good estimator of the composed leg on this vehicle, and it is not
   biased optimistic.

<!-- # FORMALIZATION_PENDING: a measurement and a closure verdict; no pointer row is produced, so there is no equations leg for tools/pointer_move_packet.py to write. The score arithmetic used throughout is the registered S = 100*d_seg + sqrt(10*d_pose) + 25*B/37,545,489. -->

Own-vehicle frontier (unchanged by this arm):
**S 0.13620226906030858 @ 179,332 B [contest-CUDA T4 n600]** (move 52).
