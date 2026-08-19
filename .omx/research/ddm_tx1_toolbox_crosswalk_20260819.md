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

---

*(sections 3+ land as the remaining sweeps return)*
