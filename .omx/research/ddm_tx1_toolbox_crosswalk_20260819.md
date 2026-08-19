# ddm_tx1 — the toolbox crosswalk: what the shelf is worth now

- **arm** `ddm_tx1` (task #1141 — operator-directed re-pricing of every parked, blocked,
  near-miss and banked item against the machinery that NOW exists)
- **date** 2026-08-19
- **axis** this arm MEASURES nothing new on a scorer. Every number it carries is either
  (a) quoted from a receipt WITH that receipt's own axis label, or (b) arithmetic on the
  pointer's own legs, labelled **DERIVED**, or (c) labelled **MODELLED**.
  `score_claim=false` · `promotable=false`. **No Modal job fired.**
- **cost** $0.
- **status** IN PROGRESS — written incrementally, committed at every stage boundary.

## THE BASE (re-read from `.omx/state/canonical_frontier_pointer.json` at arm start)

| term | value | S contribution |
|---|---:|---:|
| `d_seg` | 0.00030309 | 0.030309 |
| `d_pose` | 7.649246787e-06 | 0.008746 |
| archive | 176,420 B | 0.117471 |
| **S** | | **0.15652626435208142** |

`archive.zip` sha `7ce46fd7a845d5987903a0d85a56581961eb7716a55c38a7361e3b5ecae94b5f`,
lane `lane_ddm_up3_thirteenth_move_t4_20260819`, `[contest-CUDA]`.
**Gap to sub-0.15 = 0.00652626.** My leg model reconstructs S to 14 dp, so the exchange
rates below are exact rather than approximate.

---

## 0. THE EXCHANGE RATES — what "closing the gap" actually costs (DERIVED)

Per-byte S = `25/37,545,489` = **6.658590e-07**. So the gap is worth exactly:

| route | requirement to close 0.00652626 ALONE |
|---|---|
| **rate** | **−9,801 B** (5.56% of the archive) |
| **seg** | `d_seg` 0.00030309 → 0.00023783 (**−21.53%**) |
| **pose** | `d_pose` 7.65e-06 → 4.93e-07 (**15.5× reduction**) |

**MEASURED section budget of the shipped body** (parsed this turn through the candidate's
own receiver, `read_residual_archive` on `ddm_up3/candidate_runtime/archive.zip` — not
quoted from a memo):

| section | bytes | share | note |
|---|---:|---:|---|
| **tail RC64 token stream** | **109,696** | **62.2%** | 117,964,800 tokens @ **0.0074392 bits/token** |
| models (semantic+carrier+hpac, brotli'd) | 66,528 | 37.7% | 76,328 B uncompressed → 0.8716 ratio |
| fixed residual table | 96 | 0.05% | |
| ZIP overhead | 100 | 0.06% | single `ZIP_STORED` member `p` |
| **total** | **176,420** | | |

Uncompressed model split: semantic 36,130 · carrier 22,246 · hpac 17,952.

**The single most useful number this arm produces:** closing the gap on rate alone needs
**8.93% off the tail** — i.e. bits/token `0.0074392` → `0.0067745`. Everything in the
CONVERT-NOW table is priced against that.

**Why the tail and not the coder.** `ddm_jg2` §S1c measured the shipped coder emitting
**555 B against an ideal code length of 554.78 B** from its own probability rows — a coder
tax of **0.04%**. So the tail's 109,696 B is very nearly exactly "what the model believes."
Remaining tail headroom is **MODEL/REPRESENTATION headroom, not coder headroom.** That is
the load-bearing distinction for seed 8 and it is measured, not assumed.

---

## 1. THE TOOLBOX, VERIFIED AT ITS OWN ARTIFACT (not from memo headlines)

Repricing is only honest if the tools are real. Each row was checked this turn:

| tool | verified how | what it actually voids |
|---|---|---|
| **up3 byte-close machinery** | ran `read_residual_archive` on `ddm_up3/candidate_runtime/archive.zip`; section budget above is my own parse | "cannot write the archive" — byte-close is now a function call, minutes not days |
| **un-interleave law + derived offsets** | `residual_archive.py:188` CK2 un-interleave; canonical `carrier_section_from_archive` derives the packed portion from the body's own u24s | the class of **body-specific hardcoded offsets** (`packed[139]`, stale `PACKED_CAP1_SECTION_BYTES`) that blocked up2 |
| **container search** | up3 §3 — CK2-off + brotli q10/lgwin16 recovered **+48 B → 0 B** | a rate tax on any small carrier perturbation |
| **up2 DALI-GT pose instrument** | up3 §4 reproduced the shipping row: base `7.76948e-06` vs T4's `7.77e-06`; candidate predicted-to-printed **exact at 8 dp** | $0 exact pose measurement; no paid row needed to price a pose move |
| **jg1 DALI-lineage seg instrument** | jg1 §ANSWER-3: **0.00030307 vs 0.00030309 = 0.99995×** of the T4 seg leg, byte-exact forward model | $0 contest-axis seg measurement |
| **carrier re-solve** | jg1 §ANSWER-6: recovers `d_pose` to **1.073×** of original (1.01×/1.34×/**0.87×**) at ~0 B, 9–12 of 12 coefficients moved | **"a seg edit destroys pose"** — the ×387 hard negative. The actuators COMPOSE. |
| **decode wall-clock headroom** | up3 §ANSWER-6 measured inflate at **954.5 s of the 1800 s budget** | "a richer decoder model is too slow" — there is **1.89×** measured headroom |
| **coder tax** | jg2 §S1c: 555 B emitted vs **554.78 B ideal** from the model's own rows | "code it better" — the coder is within **0.04%** of its model. Only the MODEL is left. |
| `candidate_seal.v1` + `repin_receiver` | up3 §ANSWER-7 — the seal caught a staged `inflate.py` `ARCHIVE_SHA256` pin that would have made the receiver refuse our own candidate | wasted paid fires |
| `tools/fire_modal_auth_eval.py` · `tools/fire_local_advisory.py` | present on disk | hand-assembled dispatch, the measured error factory |

**Two tools in my charter did NOT survive contact with their receipts — I record both against
my own convenience:**

1. **"The FREE 12-dim carrier re-orientation."** It is **not free.** `ddm_up2` owed-2 states
   the re-fit *"requires pricing the basis section (Huffman-coded 5-bit codes) which is NOT
   free."* jg1 §S3 calls re-orientation "the free move" only relative to basis *enlargement*
   (which costs up to +0.008175 S, more than the entire pose leg). Changing basis VALUES at
   constant dimension is free in the *payload* and **not** in the *archive* — precisely up3
   §ANSWER-5's lesson, where a carrier perturbation cost **+48 B** and a one-coefficient flip
   already costs **+3 B**. Re-orientation must be priced through brotli like everything else.
   This does not kill the item; §2 shows its byte budget is large. It kills the word "free".
2. **"qs5 in-compile Schur compensation makes frame-1 seg edits ~zero pose tax."** The live
   mechanism that discharges the seg→pose coupling on THIS body is jg1's **carrier
   re-solve**, measured at 1.073×. I price the coupling against that receipt, not against a
   qs5 headline measured on an older body.

---

## 2. NOT CHEAPER — the items the toolbox does NOT rescue, with the reason

I priced these first because a crosswalk that only finds treasure is not a crosswalk.

### 2.1 The −0.036 S seg×rate composition (#827) — **NOT RESCUED. The blocker was never the tool.**

Receipt: `.omx/research/ddm_cr1_composition_row_827_20260801.md` `[macOS-CPU advisory]`.
Corrected apples-to-apples the prize is **larger** than advertised — `−0.0867981 S`
(seg −0.0368766 + rate −0.0499214), so the recorded `−0.035996` understates it 2.41×.
Three independent reasons the modern toolbox does not convert it:

1. **The bodies do not exist any more.** Candidate `ep854` is **360,331 B**, baseline
   `gr1_cell_drop50` **359,221 B**, in grammars `ddm_tr1_runtime_archive.v1` and
   `ddm_pfs1_composed_archive.v3_warp`. We ship **176,420 B** of RX1/F24S. These are
   **2.04× our whole archive**; the sections the composition edits have no counterpart in
   the shipped object. `ddm_lv2_terminal_campaign_completeness_20260811.jsonl` rank 19
   already ruled it **SUPERSEDED**: *"#827/#934 banked cell-drop/phase-field compositions
   belong to the tq1c-era parent and do not transfer."*
2. **The pose blocker is a PHOTOMETRIC wall, and the carrier re-solve does not cross it.**
   `window_03/tr1_config.json` carries `w_seg 100.0`, `w_rate 0.05` and **no pose term of
   any kind** — the burn base was trained pose-blind, so its frames carry no pose-legible
   signal. Measured pose cost of the composition: **6.36× worse, 61 of 61 pairs unanimous**,
   ≈ **+3.367 S** — about **39× the prize**. `ddm_cr2r` then measured the re-solve floor at
   **≥ 4.0389 S**, i.e. **46× over break-even even if all 526 unsolved pairs returned exactly
   zero**. jg1's carrier re-solve recovers pose *because the up3 body's frame 2p is a working
   photometric probe with a solved carrier*; re-solving against frames never shaped for pose
   recovers nothing. This is CLAUDE.md's standing law — **only JOINT descent crosses the
   photometric wall** — and the toolbox contains no joint-descent-on-a-dead-base tool.
3. **Cross-vehicle transfer is not automatic and here it plainly fails.** The 1.073× re-solve
   law is proven on the HPAC/CAP1 body. The TR1 body has no CAP1 carrier to re-solve.

**Verdict: NOT-CHEAPER. Blocker stands, restated correctly** — it needs a *pose-carrying
base built by joint in-loop descent*, which is a trained-vehicle deliverable, not a tool.

### 2.2 The lane program / Lane×ANNIHILATE 0.157 S (#934/#920) — **NOT RESCUED, and its premise is now falsified on our body**

Receipt: `.omx/research/ddm_fl2_force_ledger_recovery_20260804.md`. The 0.157 S is a
**description price** (Lane's total flip mass 185,801 × 8.477e-7 S/flip), explicitly *"the
ceiling #934 aims at, not a delta in hand."* Its **realization half is UNMEASURED and ~0 on
every built actuator**; aggregate re-inflation MEASURED **+0.2459 S HARMS**. Its gate `q3x`
is **closed dead** (`codex_arm_queue.jsonl:66`), its research outputs never existed, and the
only measurement on record is an n=2 smoke that FOLDED. Same lv2 rank-19 supersession.

**The re-price that matters, and it is new:** the lane program assumed Lane content is
*annihilated in the stored representation*. On the shipped body that is **false**. jg1
measured the stored tokens **99.9985% identical to the DALI GT argmax — only 1,714 cells
differ in 117,964,800** — and **95.9% of the seg debt is render/re-segment loss**. The lanes
are stored fine; they are lost in the paint→re-segment round trip. **The lane program aims at
a stage that is not where this body loses lanes.** Its energy belongs at the pre-distortion
stage, which is exactly what `ddm_jg2` is already executing.

### 2.3 The banked sub-band pool qs2 / re1 / qs5 — **ALREADY INSIDE THE BODY. There is no bank.**

My charter asked whether these compose onto up3. They do not need to: **they already fired.**
`.omx/research/ddm_bu1_bank_union_compile_20260817.md`: *"It was compiled on 2026-08-14 as
`ddm_mc35`, repaired as `ddm_mc36` Variant C, fired on T4, and PROMOTED at ΔS = −2.068040e-5.
The current hv1 ep0634 frontier carries it."* Subsumption proven at **event ID** — mc36's
runtime parse-back recovers compensation pairs `[7, 96, 105, 176, 178, 517, 523]`. qs5 was
consumed too (mc36 Variant A ran its exact-object DLS/int12 solve). The premise is a
**registered falsified premise** (`falsified_premise_registry.jsonl:14`,
`qs2_re1_bank_union_is_held_and_unfired_20260817`) and `ddm_rv2` §2.2 already REFUTED it
(HIGH): *"There is no bank. There is no pending union."*

**Doubly consumed, and this is the part no prior memo says:** `ddm_up2`/`ddm_up3` then
re-solved the **entire n600 carrier** by coordinate descent to convergence. Any surviving
per-pair compensation object from the cp135 era is superseded by that solve by construction.

Two corrections against my own charter, recorded per the charter-time law: the pool is not
"held below the naming bar" (it is spent), and cp135 (`186,252 B`) was never our current body.

### 2.4 Renderer width distillation (wd2/wd3) — **NOT RESCUED; the miss is 6×, not an instrument artifact**

Receipt: `.omx/research/ddm_wd3_n120_family_disposition_20260816.md`
`[macOS-MPS eval / seeded nonprefix n120 advisory]`. The byte half is genuinely large — the
W0_warm student's quantized semantic packet is **21,807 B (uniform int4, parse-back exact)**,
≈**16 KB** under the incumbent semantic section, which is **0.01065 S = 163% of the whole
gap**. But the seg half projects **+6.6e-4 raw d_seg** against a byte-derived bar of
**1.07e-4** — **~6× over**, and the scorer-aware continuation measured 8.2×. Family verdict
already recorded: fresh-init distillation at this budget is NEGATIVE, `verdict_scope: family`.

Could jg1's better instrument rescue it? **No.** The wd3 miss is a *ratio* on a matched
instrument (student 0.00106 vs base 0.000427 same-instrument); swapping to a 0.99995×
contest-axis instrument moves both sides together. A 6× miss is not an instrument artifact.
**Stays PARKED with its measured reactivation ladder** (warm-lineage-at-w56 · longer-budget
fresh · curriculum-inherited birth). I make no new verdict here — `ddm_na10` owns validity.

### 2.5 The aimed-correction stream (#832, "12.44% of the gap") — **it RAN, and re-anchored to our body its sign INVERTS**

The vacuity question is answered: **the $0 scorer-free test ran.** Receipt
`.omx/research/ddm_dc1_label_price_n600_20260801.json` (generated 2026-08-02T01:07:19Z,
n600, both pre-registered controls PASS, label price **0.2627 B/flip** MEASURED), re-derived
independently by `ddm_wd1`, adjudicated `VERIFIED_CONSUMED`, FOLDED at `ddm_zc1`. Nobody
needs to re-run it.

**But every price in that family is anchored to a gap of `0.7918468` and a base with
`d_seg = 0.00431179` — and hv2 recorded the binding condition: *"any consumer must
re-anchor."* I re-anchored it. This is the crosswalk's job, so I did the arithmetic:**

Position cost is the interpolation-free bound `log2(1/ρ)/8` B/flip, where ρ is the residual
density. **Control first:** on the old base ρ = 4.312e-03 and my formula returns
**0.9822 B/flip** — *exactly* ba31's published uniform bound. The instrument reproduces the
receipt, so the re-anchor is trustworthy.

Our body's residual is **14.2× sparser**: `d_seg 0.00030309` → **35,754 flips**,
ρ = **3.031e-04**, so the position bound rises to **1.4610 B/flip (1.49×)**.

| price basis (our body) | B/flip | bytes | rate +S | seg −S | **NET S** |
|---|---:|---:|---:|---:|---:|
| uniform bound + measured label | 1.7237 | 61,628 | +0.041036 | −0.030309 | **+0.010727 LOSS** |
| coherent (scaled by the measured 0.682 ratio) + label | 1.2596 | 45,035 | +0.029987 | −0.030309 | **−0.000322 ≈ break-even** |
| *the old base's blended 0.8507 B/flip* | 0.8507 | 30,416 | +0.020253 | −0.030309 | *−0.010056 — **invalid**, that price belongs to the old density* |

**The correction-stream family did not get shelved and stay shelved by accident — it died as
the residual got sparse.** Position coding cost grows as `log2(1/ρ)`; the win at
`ρ = 4.3e-3` is break-even at `ρ = 3.0e-4`. And its whole ceiling is the seg axis,
**0.030309 S**, even if corrections were free. Labelled **DERIVED** (from two MEASURED
prices plus a validated control). **NOT-CHEAPER — and now with a mechanism, not a shrug.**

**The cross-check that matters, and it points at the live chain.** jg1's pre-distortion
actuator repairs **1.55 cells per changed token** at a MODELLED **4.718 bits/token** =
**0.3805 B per repaired cell** — **3.31× cheaper** than the correction stream's best
coherent price at our density. Even if jg2's re-encoder finds the real token price is 3×
worse than modelled, the token actuator still ties. **This independently explains why
`ddm_jg2`'s chain is the live one and the correction stream is not.**

### 2.6 The adaptive token map (#869, "−113,555 B") — **NOT RESCUED; its target object does not exist on our body**

Two independent reasons, either fatal:

1. **Already a MEASURED NEGATIVE.** Fired on the fz4 `sub_final` vehicle: `d_seg` +8.4675e-4
   against a pre-registered 7.56e-4 bound, and `d_pose` **0.168** vs baseline 0.00071 — the
   pose bank destroyed, R8 guard FAIL, recomputed ΔS **+1.221 S**. `ddm_lv2` rank 21 is
   explicit: ***"never import −113555 B."***
2. **The mechanism has no target here.** It coarsens tokens onto sub-lattices *"KEEPING the
   container's global levels=16 alphabet."* Our tokens are **5-class labels in `{0..4}`**
   (`cpr1/inflate.py:95`, `NUM_CLASSES = 5`). There is no 16-level alphabet to coarsen.
   The actuator is not expensive here — it is **undefined** here.

Its causal law does transfer as a lesson, and it is the same law jg1 re-measured
independently: *"the pose sections are SOLVED AGAINST the exact token field. Any post-hoc
token mutation invalidates them."* That is jg1's ×387 hard negative, discovered twice.

### 2.7 Lane crop (#939) — **NOT RESCUED, killed by realization, and redundant on our body**

The n32 survivor row (272,869 B projected, 0.911× W) was re-measured at n600 and
**realization killed it**: described 9,591 → fixed 4,768, **collateral 15,994**, survival
**0.497**, sample seg+rate **ΔS +0.254116** — a large loss. `ddm_lv2` rank 37:
**LESSON-ONLY**. On top of that it is now redundant: it ships a *description of GT Lane
geometry*, and our body already stores tokens **99.9985% identical to the GT argmax**. We
are not missing the lanes; we are losing them in the render.

### 2.8 pz4 QAT (2,000 B pre-proof gate) — **gate FAILED by 2,232 B; closed at formulation scope**

`ddm_pz4a`: *"best exact gross joint saving is 500 B and best counted net is −2,232 B"* —
i.e. it went the wrong way against a +2,000 B gate. Recorded REFUTED. The raw int16-width
proxy predicted 8.16 KB where the exact inner coder **grew by 12 B** (a 680× proxy failure
worth remembering). The one live sub-item, **aware in-loop QAT**, is explicitly *"only a
jointly trained representation"* — trained-vehicle line, not this body. **NOT-CHEAPER.**

### 2.9 rate_crush (#949) — **fire-condition NOW met, but the content is subsumed**

Honest status: it **never fired**, has **no measured number anywhere**, authority axis
*"[unmerged source residue; no current-object measurement]"*, disposition
`QUEUED-W-FIRE-ORDER` with fire trigger *"terminal complete candidate is selected."*
**That trigger is now met** — up3 is sealed, byte-closed and inflating. But its Target A
("meta-compression of the counted description bytes") **is** the tail/model axis I rank #1
below, and its Target B ("fit a compact generative model to our own decoded output") is
what this vehicle already does (semantic renderer + token field). I therefore do not raise
it as a separate item; I fold its dual research mandate — day-fresh learned entropy models
**and** oldest-math sleepers (CTW/PPM, Krichevsky–Trofimov, enumerative coding) — into
CONVERT-NOW #1, where it has a real target object. Its retained worktree residue (3 files,
sha-certified) still needs a custody owner.

---

## 3. THE RATE AXIS IS MEASURED SHUT — except in one cell. My seed-8 thesis was WRONG.

I came in expecting the tail's probability model to be the money. It is not, and the
receipts say so plainly. Recording it against my own charter:

| level | status | measured remainder |
|---|---|---:|
| **coder** (choice of coder + byte layout) | **CLOSED** — `ddm_bp1`, byte-exact, 4 mechanisms | **−5 B** total; on the tail every mechanism *costs* (+5 to +63 B) |
| **model** (contexts, mixing, correctors) | **CLOSED** — `ddm_fx2` R6 floors + `ddm_ma1` | **~400 B** hit-event + **~75 B** within-miss |
| **representation** (fewer/smaller symbols) | **OPEN — and one cell is empty** | see §4 |

`ddm_ma1` explicitly **withdrew** the seductive big number: *"77,241.46 B is an entropy,
overwhelmingly irreducible. That framing is a vacuous denominator and I am withdrawing it."*
I will not resurrect it. The honest tail reservoir is **a few hundred bytes, not 77 KB.**
Against a 9,801 B need, `ddm_ck2`'s total credible free-rate inventory is **≈1,900 B ≈ 18%
of the gap.** **Rate alone cannot close the gap with any known machinery.** `ddm_bp1` §4
routes it correctly: *"Rate progress must come from representation — fewer or smaller
symbols — not from coding them better."*

One measured warning for anyone tempted by a static/offline model: fx2's held-out column is
**worse in every row** (114,788.9 vs 109,264.09 online). *"The field is strongly
non-stationary: online adaptation is worth more than 5,500 B."* Any representation proposal
that gives up online adaptation starts 5,500 B in debt.

---

## 4. CONVERT-NOW — ranked, with the tool that voids each blocker

### #1 — THE TOKEN-DROP FAMILY (rc4/td1), re-priced against jg1's carrier re-solve
**Projected NET −0.002929 S = 44.9% of the gap · cost $0 · all instruments exist**

| field | value |
|---|---|
| **why it was shelved** | `ddm_rc4_rung4_token_drop_verdict_20260816`: *"Rung 4 is REFUSED as an uncompensated drop — on the POSE leg, by 517×."* Rate leg exact and favourable, rate+seg gain **−3.243e-3 S** MEASURED, but pose cost **+0.17432 S = 53.8×** the gain. `delta_d_pose` 3.3279e-3 against an allowed 6.431e-6. |
| **which tool voids it** | **jg1's carrier re-solve.** rc4's damage is **435×** our `d_pose`; jg1 measured **387× mean damage recovered to 1.073× of original at ~0 bytes** (1.01× / 1.34× / **0.87×** — one pair ends *better* than shipped), 9–12 of 12 coefficients moved, well inside up2's "±4 on all 7,200 coefficients = +5 B" envelope. **Same damage class, same magnitude, measured recoverable.** The word "uncompensated" in rc4's own verdict is the tell: it refused an *uncompensated* drop, and compensation now exists. |
| **the margin** | Break-even allows the re-solve to leave `d_pose` at **1.9×** current. jg1 delivered **1.073×** — a **1.77× safety margin**. Even at jg1's *worst* pair (1.34×) the move still nets **−0.001865 = 28.6% of the gap**. |
| **projected S** | recover to 1.073× → `dS_pose +0.000314`, NET **−0.002929** (44.9% of gap). At 2× recovery it turns +0.000380 (a loss) — so the re-solve quality is the whole ballgame and must be measured, not assumed. |
| **label** | rate+seg leg **MEASURED** (hv1-era body, same lineage, same 5-class object); recovery factor **MEASURED n=3**; the composition **DERIVED**. **Not a row.** |
| **honest transfer risk** | rc4 measured on the hv1-era body; since then the archive moved 182,759 → 176,420 B, so the tail is already 105 B leaner and part of the rate credit may be spent. Re-measurement on the up3 body is mandatory, not a formality. This is *not* the TR1-era transfer problem — same receiver family, same 5-class map — but it is not free either. |
| **fire order** | (1) re-run rc4's best rung on the up3 token field; (2) score seg with **jg1's $0 instrument** (0.99995× of T4); (3) run the **carrier re-solve** against the dropped field and measure `d_pose` on **up2's DALI-GT instrument** ($0, exact); (4) byte-close through **jg2's re-encoder** (the only honest rate instrument — see the composition warning below); (5) seal, then one T4 row. |

### #2 — THE 12-DIM POSE BASIS RE-ORIENTATION (up2 owed-2 / jg1 §S3) — still unowned
**Byte budget is large and nobody has spent an hour on it · cost $0**

Not free (§1), but the break-even budget is generous, which is the point nobody has computed:

| `d_pose` reduction | new pose leg | ΔS_pose | **bytes affordable at break-even** |
|---:|---:|---:|---:|
| 1.25× | 0.007823 | −0.000923 | **1,387 B** |
| 1.5× | 0.007141 | −0.001605 | **2,410 B** |
| 2× | 0.006184 | −0.002562 | **3,847 B** |
| 4× | 0.004373 | −0.004373 | **6,567 B** |

For scale: up3's whole carrier perturbation cost **+48 B**, and a one-coefficient flip costs
**+3 B**. So even a 1.25× improvement has ~29× the byte headroom a carrier edit has ever
needed. up2 measured the enabling quantity — **6.4× median demanded-step reduction** from
re-fitting the basis to the measured pose-residual subspace — and called it *"the single
highest-value follow-on."* It has sat unowned since.

**What is NOT known and must be measured, not assumed:** the map from "6.4× smaller demanded
step" to an actual `d_pose` factor. The solve already **converged** on the current basis, so
any gain comes from spanning a better subspace. Label: **UNMEASURED**. The break-even table
is **DERIVED**. Cheap to settle — up2's instrument is exact and $0, up3's byte-close is a
function call.

### #3 — THE ONE EMPTY CELL: structure-exploiting representation of the 5-class label map
**cost $0, ~15 min/arm · honest prior UNFAVOURABLE · but the cell is genuinely empty**

`jg1` established **only today (2026-08-19)** that the tail is a plain `(600,384,512)` 5-class
label map. The sweep's explicit finding: **no run-length, connected-component, contour or
chain-code representation has ever been measured on this object.** Every such receipt in the
campaign (`contour_string_flip_coding_n600_20260707`, #307) is on the **flip residual** of the
level-set witness lineage — a different object on a different vehicle.

**And the premise that killed it there does not hold here.** That NO-GO's stated reason was
*"the measured residual is confetti (mean component 3.1 px; 38.5% of flips in ≤3 px
components)"* — the coder was fine, the object was not. The 5-class label map is the
opposite: large coherent regions (Undrivable 49.5%, MyCar 25.4%, Road 23.2% of area) with a
static ego-hood core. A contour/RLE premise that fails on confetti is exactly the premise
that could hold on a region map.

**I will not oversell it.** The current model already codes this field at **0.0074392
bits/token** and sits within **~315 B** of its own conditional-entropy floor — that is a very
strong incumbent, and the floor is hindsight-optimal so the true remainder is smaller. My
prior is that this loses. But it is the only unmeasured cell in an otherwise-complete
coder/model/representation matrix, the object was only identified today, the test is $0, and
`ddm_bp1` itself routes rate progress to exactly this level. **Test it, expect a negative,
and bank the negative as the closure of the rate axis.**

### #4 — RECONCILE THE DECODE WALL-CLOCK BUDGET (blocks fx2's E1 rung, and more)
**cost $0 · worth ~87 B directly, but it un-gates a whole class**

Two margin figures are in circulation and they differ by **7.2×**:

- fx2's `SHIPPED_CONFIG` **freezes D1 and refuses E1** on the grounds that E1 projects
  *"29 s of 1,800 s margin (1.6%)"* against D1's **118 s**, i.e. +87 B for +89 s = 0.98 B/s.
- `ddm_up3` **measured the shipped candidate inflating end to end in 954.5 s of the 1,800 s
  budget** — `returncode=0`, STRICT validation passed, 3,662,409,600 raw bytes — i.e.
  **845.5 s of margin.**

**Caveat, stated plainly:** up3's 954.5 s is a **local macOS** decode; the contest budget is
30 min on a T4 or a 4-core/16 GB CPU, and we have no contest-host decode timing for this
body. So I do **not** claim the E1 refusal is void. I claim the two numbers must be
reconciled before wall-clock is used to refuse *anything* — that reconciliation is $0, and it
gates every "richer decoder model" proposal, not just E1.

---

## 5. THE COMPOSITION WARNING MAIN AND jg2 SHOULD READ FIRST

`ddm_jg2` is spending bytes on token edits to **buy seg**. CONVERT-NOW #1 spends seg to
**buy bytes**. They are opposite directions on the *same actuator*, and **both consume the
same scarce resource: the carrier's pose-recovery budget.**

- The re-solve must absorb the **combined** damage, not each in turn. Break-even tolerates
  `d_pose` ≤ **1.9×** current *in total*; jg1's 1.073× was measured on seg edits alone.
- `ddm_bu1`'s measured law binds here: **"never price a union as the SUM of its legs'
  compensation objects"** — mc36 beat the naive union **3.705×** by fresh-Schur-solving
  compensation *jointly* over the composed object. And `ddm_gx1`: *"qs4 carried a stale
  compensation and paid +2.396e-4 in d_pose."*
- `ddm_jg2` §S1b's finding makes this sharper, not softer: there is **no per-token price** —
  four feedback paths (partially-decoded frame into `sparse.selected_logits`;
  `prepare_frame_context`; `_boundary_buckets`; never-reset KT counters) make one edit's blast
  radius **global through pair 599**. So a drop and an edit on the same field cannot be priced
  separately at all. **One joint re-encode, one joint carrier re-solve, one candidate.**

**Concretely:** if #1 is pursued, it should enter jg2's S2 as an additional coordinate in the
same joint solve — not as a second candidate composed afterwards.

---

*(§6 adjacent/tangential lands as the last sweep returns)*
