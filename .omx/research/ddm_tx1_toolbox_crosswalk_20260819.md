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

*(sections 2+ land as the sweeps return)*
