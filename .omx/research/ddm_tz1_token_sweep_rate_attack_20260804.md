# ddm_tz1 — archive token-sweep RATE attack, built to READY (byte-only leg landed)

**Arm:** `ddm_tz1` (operator: *"remember the archive token sweep rate attack"* + steer#1 *"adaptive
dynamic quantization and quantization awareness"* + steer#2 *"smaller than int16 where int16 is
unnecessary rate"*) · **Date:** 2026-08-04
**Axis:** apparatus / RATE-attack. `score_claim=false`, `promotion_eligible=false`,
`rank_or_kill_eligible=false`, `ready_for_exact_eval_dispatch=false`.
**Own-vehicle frontier:** `S = 0.7910689 @ 353,805 B [macOS-CPU advisory]` — **UNMOVED**. This arm
runs NO scorer, edits NO receiver / `tac.submission_chain` (sibling `ddm_bz1` owns the one n600
scorer slot + the PROFILES chain). It byte-closes nothing alone; it is the RATE leg that composes
with bz1's seg+pose row (disjoint archive sections) at byte-close time.

Tokens are **99% of the archive bytes** (gk2 §5.2: `tokens.dr7t` = 96.2%–99% of the archive) = **THE
rate axis** (rate = 15.5% of the gap to the PR130 floor). This memo lands EVERY scorer-free (byte-only)
leg of the token-sweep NOW and structures the JOINT waterfill so the d_seg-under-drop verdicts fire
the instant the scorer frees.

## §0 — What this arm is (relative to bs2 / br1 / gk2)

- **bs2** (`ddm_bs2_...20260801`) MEASURED the STATIC global-L rate curve on the **smevr/tr1 form**
  (base 346,478 B; L=14 = −23,655 B) and named the ±1.0 clamp mass (33.30%). It did NOT sweep the
  LIVE ix2 vehicle, the adaptive/per-cell form, the lzma-filter re-race, or the depth axis.
- **br1** (`...20260803`) MEASURED the ix2 unit-drop + alphabet-drop surface (format-free) and proved
  BASE = 341,295 B on the LIVE ix2 form.
- **gk2** (`...20260804`) is the FINDER: it named L (`token_quant_levels`) as unladdered knob #1 and
  cited bs2's −23,655 as its receipt — **but conflated the two vehicle forms** (§1 below).
- **This arm** BUILDS the reusable sweep HARNESS + lands the byte-only leg of ALL six arms + emits the
  READY manifest that fires the scorer verdicts. It is the actuator gk2's finder pointed at.

Harness: `experiments/ddm_tz1_token_sweep_rate_attack.py` (scorer-free; imports `ddm_ix2_archive_container`
[LIVE ix2 form], `ddm_r7_token_coder` [smevr/tr1 form], and reuses pa1b's `margin_coupled_level_map`
recall-at-source). Consolidated byte-only receipt:
`/Volumes/VertigoDataTier/pact/ddm_tz1_20260804/tz1_byte_only_receipt.json`.
Calibration (gk2, current pu2 frontier): gap = 0.7910689 − 0.172141 = **0.6189279**; **1% of gap =
9,295 B**; W = **1.27309 B/flip** (break-even bytes per seg-flip).

## §1 — ARM A: GLOBAL-L (static) sweep + the form-conflation CORRECTION

The global L=14 is the **one-rung special case** of the adaptive waterfill (ARM B). Positive controls
both reproduce EXACTLY on `cx1_tokens.npy`:

| form | base (L=16) | L=14 member | L=14 saved | matches anchor |
|---|---:|---:|---:|---|
| **ix2 (LIVE receiver)** | **341,295** | 316,690 | **24,605** | br1 anchor ✓ |
| smevr / tr1 form | 346,478 | 322,823 | **23,655** | bs2/gk2 anchor ✓ (lossless RT verified) |

**CORRECTION (NO-FAKE):** gk2 §4 cites "L=14 saves 23,655 B" as knob #1's receipt. **23,655 is the
stale smevr/tr1-form number on base 346,478.** The LIVE receiver is ix2 (gk2 §1), and on the ix2
vehicle **L=14 saves 24,605 B** (base 341,295). gk2 conflated the two forms; the live rate saving is
~4% larger than the cited figure. Both are reproduced end-to-end here.

LIVE ix2 global-L curve (each row: pre-registered rate↔distortion break-even):

| L | saved B | ΔS_rate | % of gap | break-even Δd_seg (pays iff below) |
|---:|---:|---:|---:|---:|
| 15 | 17,363 | −0.011560 | 1.87% | 1.156e-4 |
| **14** | **24,605** | −0.016383 | **2.65%** | **1.638e-4** |
| 12 | 50,523 | −0.033641 | 5.44% | 3.364e-4 |
| 10 | 78,691 | −0.052395 | 8.47% | 5.239e-4 |
| 8 | 106,099 | −0.070646 | 11.41% | 7.065e-4 |

Coder winner on the live blocks: **residual = brotli, base = brotli** (bears on ARM E).

## §2 — ARM B: ADAPTIVE per-cell L (#869) — the domination is visible on BYTES ALONE

Operator steer#1: a single global L is the degenerate form. #869's **768-cell × 4-rung** token-by-token
waterfill assigns each (R,C) cell its OWN quant rung. This is **FORMAT-FREE** (encode at global
levels=16; each cell snaps to its own sub-lattice — the br1 alphabet-drop mechanism, per-cell). All
round-trips exact. Two rung MAPS, both priced (steer#1 "price BOTH"):

| map | rung ladder | saved gross | map cost | **saved NET** | % of gap | break-even Δd_seg | level hist (of 768) |
|---|---|---:|---:|---:|---:|---:|---|
| **margin-coupled (QA80 flip-mass)** | [16,12,8,4] | 113,648 | **93 B** (STORED) | **113,555** | **12.22%** | 7.56e-4 | 552·L4 / 87·L8 / 86·L12 / 43·L16 |
| derived (token activity) | [16,12,8,4] | 62,502 | **0 B** (rule-118-free) | 62,502 | 6.72% | 4.16e-4 | 432·L4 / 110·L8 / 148·L12 / 78·L16 |
| margin-coupled | [16,8] | 64,285 | 54 B | 64,231 | 6.91% | 4.28e-4 | 639·L8 / 129·L16 |
| derived | [16,8] | 35,840 | 0 B | 35,840 | 3.86% | 2.39e-4 | 543·L8 / 225·L16 |

**THE DOMINATION (byte-only):** margin-coupled [16,12,8,4] saves **113,555 B net — MORE than global
L=8 (106,099 B)** — while KEEPING 43 high-flip cells at full L16 and 86 at L12. Deep-margin (seg-safe)
cells are **byte-rich**: coarsening them hardest pays most on rate AND (by construction) least on d_seg.
So the adaptive form beats the static form on the rate axis alone; the JOINT (rate × d_seg) domination
is the scorer-gated verdict (READY manifest #1). Global-L is genuinely just one point in this space.

**Quantization-awareness (steer#1):** the margin-coupled map IS `pa1b.margin_coupled_level_map` (rank
transform of the flip-mass order statistic; recall-at-source, not rebuilt) — coarse where SegNet margin
is deep, fine near the separatrix. The QA80 flip-mass field is scorer-derived (NOT decodable), so its
map must be STORED — but it costs only **93 B** (the tier assignment is spatially clustered → crushes).
The **derived** map uses per-cell token activity (decodable from the decoded tokens → **0 counted
bytes**, rule-118-free); whether activity is a good enough proxy for flip-mass is the scorer question.

## §3 — ARM C: rung-map price (STORED vs DERIVED, both priced)

- **STORED** (margin-coupled, scorer-optimal allocation): 4-rung = **93 B**, 2-rung = **54 B**. The map
  is a 768-cell tier index that compresses to near-nothing (spatial clustering).
- **DERIVED** (activity proxy, recomputed at decode): **0 B**.

The 93-B stored margin map buys **+51,053 B** of extra saving over the 0-B derived map (113,555 vs
62,502) — IF the flip-mass allocation's d_seg cost is acceptable. That ~51 KB-for-93-B trade is the
sharpest question in the READY manifest.

## §4 — ARM D: ±1.0 clamp mass (reproduces bs2; refit scorer-gated)

Per-symbol histogram of the shipped codes: **lvl0 = 30.17% + lvl15 = 3.13% = 33.30% of token mass
pinned at the two ±1 bounds** — reproduces bs2's 33.30% exactly. Per-channel extremes: ch1 = 43.6%
(most-clipped), ch0 = 34.4%, ch2 = 38.1%, ch3 = 17.1% (least). The **refit** (widen/narrow the range)
is **SCORER-GATED and needs the continuous pre-clamp tokens** (absent from any artifact — bs2 §5.1:
"the clamped parameters' desired values do not exist in any artifact"). Per steer#1 point 3, the
adaptive/dynamic range is the **per-cell twin** of the static ±1.0 and is already folded into ARM B
(the per-cell sublattice snap re-maps the effective range per cell); a separate global refit is not a
distinct byte lever.

## §5 — ARM E: LZMA-filter re-race — DEAD on the live vehicle

Re-raced lc∈{0..4}/lp∈{0..2}/pb∈{0..2} on the REAL ix2 token payloads:

| block | shipped lzma (lc3lp0pb0) | best variant | best B | variant gain | **brotli (the WINNER)** |
|---|---:|---|---:|---:|---:|
| residual | 348,438 | lc2lp0pb0 | 347,609 | 829 B | **339,970** |
| base | 1,348 | lc0lp1pb0 | 1,248 | 100 B | (brotli wins) |

**VERDICT: no realized rate gain.** brotli (339,970) beats even the best lzma variant (347,609) on the
residual, so lzma is never the `code_block` winner and its filter tuning never ships. Confirms gk2
row-5 ("lzma competes … may not win the bulk"). Closed as a dead knob on the ix2 vehicle.

## §6 — ARM F: DEPTH × CODER (steer#2) — NEGATIVE on the token stream

steer#2: L IS the token bit-depth (4 bits/token @ L=16). Tested a tight `ceil(log2 L)`-bit pack vs the
fixed 4-bit nibble, each through `code_block`:

| L | bits/code | nibble coded | tight coded | tight gain |
|---:|---:|---:|---:|---:|
| 16 | 4 | 339,970 | 339,970 | 0 (identical) |
| 14 | 4 | 315,524 | 315,524 | 0 (ceil(log2 14)=4) |
| 8 | 3 | 234,156 | 274,295 | **−40,139 (WORSE)** |

**VERDICT: sub-nibble bit-packing is a NET LOSS.** At L=8 the 3-bit raw payload shrinks 25%
(921,600→691,200) but the CODED payload GROWS by 40 KB, because tight packing destroys the
byte-alignment brotli/lzma need for the long same-as-before LZ runs. The 4-bit nibble is already
coder-optimal; token bit-depth is fully subsumed by L. steer#2's "smaller than int16" savings must be
sought in OTHER counted sections, not the token stream.

## §7 — ARM G: ST_GRID + other-section depth inventory (steer#2, other turf)

- **ST_GRID** (11 knots): already `encode_exact_table` → **scaled_int, 13 B** — the f16<f32<f64<scaled-int
  depth-ladder already picked the minimal width. Not a depth knob. The support re-race guard
  (ca1 row-7 owed): `comb(11,5)=comb(11,6)=462 ≪ 400,000` cap → passes trivially. The re-race itself is
  SCORER-GATED (re-snapping s_t changes the homography = lossy) and needs the live selector for the
  index-entropy leg.
- **Other counted sections** (selector per-pair `sel`/`beta_idx`/`s_t`-index streams): the depth
  inventory needs the LIVE selector artifact → QUEUED (blocked-by-ARTIFACT, not scorer). The
  pose/dxi stream (k=4) is **bz1's turf** — NOT touched here.

## §8 — READY manifest: scorer measurements queued (bz1 owns the scorer slot)

Fire-order, each with its pre-registered break-even. All are byte-closed here; only the d_seg/d_pose
legs remain.

| # | measurement | byte leg (LANDED) | scorer leg (QUEUED) | break-even | verdict-scope |
|---|---|---:|---|---:|---|
| **1** | **ADAPTIVE margin-coupled [16,12,8,4]** through byte-close, JOINT rate+d_seg | net −113,555 B (map 93 B), 12.22% gap | Δd_seg from the per-cell flip-mass coarsening | **pays iff Δd_seg < 7.56e-4** (17.5% of live d_seg 4.31e-3) | formulation (this ladder+map) |
| **2** | ADAPTIVE derived-activity [16,12,8,4] (0-byte map) | −62,502 B, 6.72% gap | Δd_seg from activity-proxy coarsening | pays iff Δd_seg < 4.16e-4 | formulation |
| **3** | GLOBAL-L {15,14,12,10,8} | curve §1 | Δd_seg per L | L14 iff Δd_seg < 1.64e-4 | measured-config |
| 4 | ±1.0 clamp refit (widen/narrow) | mass 33.30% | Δd_seg | needs continuous tokens (retrain artifact) | formulation |
| 5 | ST_GRID support+knot re-race | table 13 B; guard passes | Δd_seg (lossy re-snap) + index entropy (needs selector) | — | formulation |

**Highest-EV: #1.** A single scorer job (re-quantise per the margin map → render through
`inflate_runner_v4d.py` → score) decides a 12.22%-of-gap rate move whose d_seg cost is DESIGNED to be
sub-break-even (the map coarsens seg-safe cells). If #1 lands under break-even it is the single largest
rate mover measured. #2 is the 0-byte-map fallback if the stored map's d_seg is worse than the activity
proxy's. These are the JOINT remeasure the "coder×drop surface never measured at any drop level" (#869)
was owed — the byte surface is now complete; only the d_seg axis is open.

**Composition:** this RATE leg composes with bz1's seg+pose row (disjoint archive sections — tokens vs
pose/renderer). It fires on the SAME row after bz1's seg+pose byte-closes, reusing bz1's
`submission_chain` PROFILES at byte-close time (not duplicated here).

## §9 — CROSS-REFERENCE (NOT executed here): gk2 #2 window_solve

gk2's unladdered knob #2 (`window_solve` default-OFF, tr1:1382-1421) is MEASURED at **−0.01441 S d_seg,
0 bytes**, blocked solely on **d_pose-under-the-v4d-frame_0-warp**. That blocker is **unblockable by
js1's frame_0 pose-repair mechanism** — flag it for the seg/pose composition (bz1's chain). It is a SEG
lever, NOT part of this rate sweep; noted per the brief, not touched.

## §10 — Boundaries / owed / NOT done (STATE-THE-BOUNDARIES)

- **No score measured.** Every d_seg/d_pose leg is scorer-gated (§8). Every byte number is MEASURED
  (real coder, lossless round-trip verified) — but a rate saving is only REALIZED if the scorer leg
  clears its break-even.
- **The token member is cx1-era.** pu2's frontier delta (0.8265→0.7911) is a POSE search (SEARCH
  bucket, m06/m84), not a token change; the token member (341,295 ix2) is effectively unchanged
  (archive 353,805 pu2 vs 353,808 cx1 = 3 B, pose/selector side). So `cx1_tokens.npy` IS the live token
  rate axis. Should a future token retrain move the lattice, re-run the harness.
- **Adaptive "domination" is byte-true but the JOINT claim is scorer-gated.** Margin > global-L8 on
  bytes is measured; that it also wins on d_seg is the hypothesis the map is designed for, not proven.
- **The full harness receipt run is slow under machine contention** (brotli-q11 × ~57 code_block calls;
  load 6–9 from sibling scorer jobs). Each arm is validated by the lean probes that use the identical
  functions; the consolidated receipt is written directly. The module produces the full JSON uncontended.
- **DERIVED-map fidelity to flip-mass is unmeasured** — activity is a plausible but unproven proxy;
  §8 #1-vs-#2 is exactly that comparison.
- I did NOT edit any receiver / `tac.submission_chain` / carrier-compose file; NO scorer, NO n600, NO
  `upstream/` edit; pointer UNMOVED.

Own-vehicle frontier: **S = 0.7910689 @ 353,805 B [macOS-CPU advisory] — UNMOVED.**
