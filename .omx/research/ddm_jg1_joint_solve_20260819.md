# ddm_jg1 — the full joint solve over every stored section of the shipped object

- **arm** `ddm_jg1` (task #1134 — the js1 joint line reborn as a SOLVE)
- **date** 2026-08-19
- **axis** every number this arm measures is `[macOS-CPU advisory]` unless it carries an
  explicit DALI-lineage tag · `score_claim=false` · `promotable=false`. This arm fires no
  Modal job; MAIN owns the T4 slot.
- **cost** $0.
- **store** `/Volumes/APDataStore/pact/ddm_jg1/`
- **status** IN PROGRESS — this memo is written incrementally, stage by stage, and
  committed at every stage boundary.

STORES CONSULTED: `.omx/state/canonical_frontier_pointer.json` (re-read at start) ·
`.omx/research/ddm_up2_shipping_object_pose_solve_20260819.md` (the method law, §3/§5/§8) ·
`.omx/research/ddm_up3_thirteenth_move_byteclose_20260819.md` (the byte-close machinery,
§1/§3/§5) · memory `pose_gap_was_gt_cache_lineage_not_cuda_20260819` ·
`/Volumes/APDataStore/pact/ddm_up3/retained/**` ·
`/Volumes/APDataStore/pact/ddm_to1/generations/to1_tail_override_r1/**` — the SHIPPED
receiver, read at source (`cpr1/inflate.py`, `runtime/residual_archive.py`,
`runtime/f26_inflate.py`) rather than quoted from memory · `upstream/modules.py`.

---

## THE BASE (re-read from the pointer, not from the charter)

| term | value | S contribution |
|---|---:|---:|
| `d_seg` | 0.00030309 | **0.030309** |
| `d_pose` | 7.649246787e-06 | 0.008746 |
| archive | 176,420 B | 0.117471 |
| **S** | | **0.15652626435208142** |

`archive.zip` sha `7ce46fd7a845d5987903a0d85a56581961eb7716a55c38a7361e3b5ecae94b5f`.
Gap to sub-0.15 = **0.006526**.

---

## S0 — THE COORDINATE MAP

Read out of the receiver's own parse chain (`residual_archive._decode_rx1_models`,
`:159-240`), not guessed. The ZIP holds one `STORED` member `p` of 176,320 B; the RX1
header (`<4sBBBBHHH>`, 14 B) counts three streams and **everything after them is the
tail**:

| # | section | archive B | share | what it is | actuator status |
|---|---|---:|---:|---|---|
| 0 | RX1 header | 14 | 0.0% | framing (`reserved=6`) | no |
| 1 | hpac stream | 13,515 | 7.7% | the **IHS1 probability model** for token coding | model axis, live |
| 2 | semantic stream | 30,856 | 17.5% | `SemanticTokenRenderer` weights (width 96) | weights axis, live |
| 3 | carrier stream | 22,143 | 12.6% | CAP1: basis + 600x12 int12 coefficients | **SOLVED by up2/up3** |
| 4 | **tail** | **109,792** | **62.2%** | **the HPAC-coded token payload** | **THE SEG ACTUATOR** |

(`+100 B` of ZIP container overhead reaches the scored 176,420.)

**The charter's premise — that the seg content is an opaque latent — is wrong, and the
truth is much better.** `cpr1/inflate.py:95` is
`self.token_embed = nn.Embedding(NUM_CLASSES, width)` with `NUM_CLASSES = 5` (`:21`), and
`render_video` (`:318`) calls `semantic(tokens[start:end].long(), indices)`. The tokens
are **5-way class labels on a 384x512 grid, one grid per pair** — the field is
`(600, 384, 512) uint8 in {0..4}`, **117,964,800 cells**, entropy-coded into section 4.
The shipped object is literally a coarse semantic map plus a small net that paints it into
RGB, which SegNet then re-segments. That is the task-lossy ego-scene codec archetype,
observed rather than asserted.

### The two frames are NOT symmetric, and that asymmetry is the whole joint structure

* `cpr1/inflate.py:313-328` renders frame **2p+1** from the tokens.
* `cpr1/inflate.py:336-352` renders frame **2p** as
  `127.5 + CARRIER_AMPLITUDE * (12-dim basis expansion)` — **a photometric probe frame,
  not a picture of the scene.**
* `upstream/modules.py:108` is `x = x[:, -1, ...]` — **SegNet sees only frame 2p+1.**
  PoseNet sees both.

| actuator | frame it writes | reaches `d_seg` | reaches `d_pose` |
|---|---|---|---|
| tokens (section 4) | 2p+1 | **YES** | **YES** |
| carrier coeffs (section 3) | 2p | no | YES |
| renderer weights (section 2) | 2p+1 (all pairs) | YES | YES |

So the carrier is a **pose-only** actuator with `d_seg` obligation exactly zero, and the
tokens are a **joint** actuator. Any token edit that buys `d_seg` and spends `d_pose` can
be met by re-solving the carrier, which is free in bytes (up2 §ANSWER-4) — that is the S2
composition hypothesis, and it is structural, not hopeful.

### The exchange rate that governs the seg axis

Both legs are linear in their own unit, so they can be quoted against each other exactly:

* `d_seg` denominator = 600 x 384 x 512 = 117,964,800 argmax cells, so
  `d_seg = 0.00030309` is **~35,754 disagreeing cells**.
* one fixed cell = `0.030309 / 35754` = **8.477e-07 S**
* one archive byte = `25 / 37,545,489` = **6.658e-07 S**

**One repaired seg cell is worth 1.273 archive bytes.** Section 4 currently spends
109,792 B on 117,964,800 tokens = **0.00745 bits/token**, so the model predicts the token
field extremely well and a token flip against a confident context is expensive. The seg
axis is therefore not a free-actuator problem like pose was: every proposal must be priced
against this exchange rate, and the honest question is not "can we flip cells" but
"can we flip a cell for less than 1.273 bytes".

*(S1/S2/S3 sections follow as they are measured.)*
