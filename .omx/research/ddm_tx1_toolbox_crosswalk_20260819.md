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

*(sections 1+ land as the sweeps return)*
