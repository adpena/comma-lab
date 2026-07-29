# ddm_wr1 — sensitivity-weighted reverse-waterfill on the DR7 token lattice (task #766 / QA06)

**Pointer: `0.1910828242` [contest-CPU] UNMOVED.** Nothing here is a score. Bytes are MEASURED
(lossless, through the landed r7 SMEVR coder). d_seg and d_pose are PREDICTED / TYPED and are
resolved ONLY by the STAGED n600 realized gate (not fired — sb1 owns the slot, QA43 queued ahead).
`score_claim=false · promotion_eligible=false · pointer_moved=false`. Axis:
`[macOS-CPU advisory, rate-only]`.

Status: **BUILD-DONE-GATES-STAGED.**

## 0. The reference row (MEASURED, pfs1 D1 exact-protocol, real evaluator / real bytes)

The "2.256641" row my descent curve is measured against is the pfs1 D1 archive
(`ddm_pfs1_20260729/d1/.../archive.zip`, sha `624ffe57…`):

| term | value | S-contribution | share |
|---|---|---|---|
| SegNet d_seg | 0.00389011 | 0.389011 | 17.2% |
| PoseNet d_pose | 0.22144216 | √(10·d_pose)=1.488076 | **65.9%** |
| rate (569,996 B) | 0.0151815 | 0.379537 | 16.8% |
| **final S** | | **2.256641** | |

Archive member breakdown (569,996 B): **`tokens.dr7t` = 557,253 B (97.8%)** ← my lever;
non-token floor (renderer.sec 3,341 + pose_warp.stp 6,864 + manifest 1,234 + selector 535 +
pose_stub 83) = **12,743 B**. Token grid `[600,24,32,4]` levels 16; SMEVR = base 1,361 B +
delta 555,836 B. Delta occupancy 0.5964 (40% of cells already sit at the temporal mode).

**The reference row is POSE-dominated, not rate-dominated.** The pfs1 warp pose is the SUBOPTIMAL
P3 solve (d_pose 0.221; P3v2 chartered). Every S below is reported three ways so the rung's true
lever value (rate) is never hidden behind the pose term.

## 1. The sensitivity map (analysis-only; Fisher/margin currency, never L2)

Per cell (24×32 = 768, each cell governs a 16×16 SegNet-argmax block; the token grid is 1:1 with
the 384×512 argmax after resize — verified by reproducing ru1's headline exactly):
- **flip mass** = ru1 atlas SegNet flips mapped to the cell (`(y//16, x//16)`). Reproduces ru1
  bit-for-bit: **486 zero-flip cells**, top-100/768 = **83.1%** of flip mass, flips live in argmax
  rows 154–297 (cell rows 9–18, the road/lane mid-band).
- **residual byte mass** = Σ|signed delta| over pairs×channels (decoded grid).
- **pose-region type** (row band → comma10k geometry + warp read): `sky_undriv_top` (rows 0–8,
  above horizon, uniform → low PoseNet content), `road_lane_midband` (rows 9–17, the road plane →
  HIGH ego-motion cue), `mycar_hood_bottom` (rows 18–23, static ego → low motion cue).

Drop ordering = **flip-risk ascending (safest first), ties by residual-mass descending
(fattest-safe first)** → steepest early descent at zero predicted flip cost. κ = 0.0753
logits/quantum (ru1 median-of-medians) anchors the flip-creation model; the 141-edit
`token_quantum_calibration` is the empirical edit anchor (d_bytes, net_fixed, per-cell κ).

## 2. The reverse-waterfill descent curve (bytes MEASURED per tranche; COMPOSED re-pricing)

Every row re-runs the **REAL SMEVR coder on the whole dropped grid** (knee-law compliant; the
8.8%-additivity trap is avoided by construction — never summed). drop-to-mode = zero the cell's
delta across all 600 frames (base/mode retained). Three S readings:
`S_ref_flipfree` (pose 0.221 held, drops assumed flip-free) · `S_ref_ceiling` (+ full current
flip mass of dropped cells as a worst-case seg cost) · `S_if_solved` (seg 0.0152 rp1-q1 + pose
0.0153 PR130-grade + this row's MEASURED rate — the sub-0.15 relevance).

| k cells | archive B | tokens saved | rate | S_ref (flipfree) | S_if_solved | dropped bands (sky/road/hood) |
|---:|---:|---:|---:|---:|---:|---|
| 0 (ref) | 569,996 | 0 | 0.3795 | **2.2566** | — | — |
| 100 | 482,742 | 87,254 | 0.3214 | 2.1985 | 0.3519 | 76 / 20 / 4 |
| 200 | 409,534 | 160,462 | 0.2727 | 2.1498 | 0.3032 | 150 / 25 / 25 |
| 300 | 346,671 | 223,325 | 0.2308 | 2.1079 | 0.2613 | 223 / 27 / 50 |
| 400 | 297,368 | 272,628 | 0.1980 | 2.0751 | 0.2285 | 276 / 28 / 96 |
| **486 (Knee A)** | **274,333** | **295,663** | **0.1827** | **2.0598** | **0.2132** | **288 / 28 / 170** |
| 540 | 227,327 | 342,669 | 0.1514 | 2.0285 | 0.1819 | 288 / 77 / 175 |
| **600 (Knee B)** | **174,578** | **382,675** | **0.1162** | **1.9933** | **0.1467** | **288 / 129 / 183** |
| 660 | 118,245 | 451,751 | 0.0787 | 1.9558 | 0.1092 | 288 / 183 / 189 |
| 730 | 51,128 | 519,125 | 0.0340 | 1.9111 | 0.0645 | 288 / 250 / 192 |
| 768 (all) | 14,303 | 555,693 | 0.0095 | 1.8866 | 0.0400 | 288 / 288 / 192 |

**Budgets (if seg+pose solved to PR130-grade 0.0305 distortion): sub-bar (0.172) ⇒ archive
≤ 212,507 B; sub-0.15 ⇒ archive ≤ 179,467 B.**

## 3. Knee candidates (the two byte-closed, staged points)

**Knee A — the FREE safe floor. k=486, archive 274,333 B** (`wr1_kneeA_safe_274k_archive.zip`,
sha `b6be1691…`). Drops ALL 486 zero-flip cells: **−295,663 B tokens (−0.1968 S of rate) at
PREDICTED ~zero d_seg cost.** ~40% of the delta stream is SegNet-argmax-irrelevant waste
(dynamic sky/hood texture the scorer never reads) — the "spend bytes only on the scorer-relevant
manifold" thesis, measured. Band composition: 288 sky + 170 hood + only 28 road-midband (all
zero-flip stable interior) ⇒ pose-favorable by construction.

**Knee B — the sub-0.15 byte target. k=600, archive 174,578 B**
(`wr1_kneeB_subbar_173k_archive.zip`, sha `8a75ac38…`). First tranche INSIDE the sub-0.15 byte
budget (174,578 < 179,467). `S_if_solved = 0.1467`. Requires dropping 129 road-plane midband
cells (up from 28) — flip-bearing (predicted d_seg band [0.00389, 0.0054]) AND pose-relevant.

Both byte-closed archives roundtrip-decode canonically; Knee A verified to inflate (2-pair
render smoke: f0+f1 at 874×1164 uint8). The pfs1 `inflate_runner` does NOT enforce manifest
`tokens_sha256`, so byte-close is a surgical `tokens.dr7t` member swap — **no source file edited.**

## 4. THE POSE-SAFETY FINDING (the binding coupling)

The token grid renders frame_1; the pfs1 warp base reconstructs frame_0 = warp(frame_1); PoseNet
reads YUV6 of BOTH frames. So token drops can leak into d_pose. Typed risk from geometry +
frozen-scorer factorization (PoseNet = global ego-motion, low-spatial-freq sensitive):

- **k ≤ 486 (Knee A) is pose-favorable**: the safe drops are 288 sky (above horizon, uniform →
  ~zero pose content) + 170 hood (static ego → low motion cue) + only 28 stable road cells. All
  road-plane BOUNDARY detail is retained.
- **Crossing into sub-0.15 bytes (Knee B) requires ~100 road-plane drops** (28→129 midband). The
  road plane IS the primary ego-motion cue. These cells are **simultaneously the highest
  seg-flip-risk AND the highest pose-leakage-risk tranche** — the two risks are co-located.

**In the pose-SOLVED regime the constraint sharpens to a razor:** at d_pose 2.33e-5 the pose
term is 0.0153 and its sensitivity is `d(term)/d(d_pose) ≈ 327 /unit` — a leak of only
+2.3e-5 (a 2× degradation of a solved pose) costs +0.0153 S, comparable to a whole ~57 KB of
rate. Therefore the reverse-waterfill's byte lever and the pose solve **are not independent** —
they are coupled through the road-plane cells. **Implication for MAIN: the sub-0.15 candidate
(Knee B) needs a pose-safety-CONSTRAINED variant** (exclude the highest-leakage road cells at a
small byte cost, or co-solve pose on the dropped base), and the staged gate MUST measure d_pose,
not just d_seg. This is why the gate is the full evaluator (both terms), not a d_seg-only chunk.

## 5. Staged realized gates (BUILT, one-command, DO NOT self-fire)

`experiments/stage_wr1_realized_gate.sh {kneeA|kneeB} [cpu|cuda]` — surgical archive swap into a
copy of the pfs1 D1 submission dir + stock `eval_root/evaluate.sh`. Emits the exact composed row
(SegNet + PoseNet + rate), apples-to-apples with 2.256641. Receipt schema
`wr1_realized_gate_receipt_SCHEMA.json` (`ddm_wr1_realized_gate.v1`).

- **Projected slot time: ~17 min / candidate** on macOS-CPU (pfs1 D1 full n600 was 1006 s), well
  inside the 30-min decode budget. Two candidates ≈ 34 min of idle scorer slot.
- **Break-even (preregistered, gc6 row 6):** accept iff realized ΔS < 25·ΔB/37,545,489.
- Verdicts the gate resolves: (a) does Knee A hold d_seg ≈ 0.00389 and d_pose ≈ 0.221 (pose-safe)?
  (b) does Knee B's flip creation stay within the [0.00389, 0.0054] band AND is its d_pose leak
  small? These are the only unknowns; the bytes are already exact.
- Advisory axis; the contest-CPU authority is the later Modal flight (operator-GO).

## 6. QA07 / QA08 pool overlaps (COMPETE, never sum)

- **QA07 (nested {L16,L8,L4,base} lossy rungs) — SAME POOL as this rung, mutually exclusive
  per cell.** drop-to-mode is the coarsest rung (→ base) of the QA07 ladder. For a given cell you
  do ONE of {keep L16, L16→L8, L8→L4, drop-to-mode} — you cannot bank two. QA07's gentler rungs
  sit BETWEEN "keep" and "drop" on the price frontier ⇒ they **refine the knee granularity**
  (more Pareto points near k≈540–600), they do NOT add independent savings. My curve uses the
  drop-to-mode extreme; QA07 would let the waterfill concede less on the marginal road cells.
- **QA08 (context mixing / o8+prev5 KT, CTW, CM) — a LOSSLESS coder swap, PARTIAL overlap.** It
  lowers bytes for the SAME surviving token values at EVERY drop level (shifts the whole curve
  down) at zero seg cost. It composes with drops ONLY by re-encoding the DROPPED stream with the
  QA08 coder — **never sum** "drop savings + QA08 savings" (pb1 P4: 8.8% additivity). My curve is
  priced with SMEVR; a QA08 re-race on the Knee-A/B dropped streams is a $0 staged follow-on that
  can only improve the byte column (seg unchanged).

## 7. Honest bottom line

- **The rate lever is REAL and LARGE:** −295,663 B (archive 570→274 KB) at PREDICTED-zero seg cost
  from the 486 zero-flip cells alone; the full descent reaches the sub-0.15 byte budget
  (≤179,467 B) at Knee B (174,578 B). Bytes MEASURED through the real coder.
- **But S is NOT rate-bound in the as-measured row — it is POSE-bound** (pose term 1.488 = 66%).
  The rung cannot move the exact pointer by itself; its rate value is unlocked only when the
  suboptimal warp pose (0.221) is solved (P3v2) AND d_seg is solved (fd1/tr1). Composed-if-solved:
  Knee A = 0.213 (over bar), **Knee B = 0.147 (sub-0.15)** — sub-0.15 sits INSIDE the flip-bearing
  road-plane tranche, exactly where seg-flip and pose-leakage risks are co-located.
- **The rung's job is DONE at the build level; the pointer is UNMOVED.** The next move is MAIN
  firing the two staged gates (~34 min slot) to resolve the only two unknowns (Knee B's realized
  d_seg and d_pose), then a pose-safety-constrained Knee-B variant if the road-plane d_pose leaks.

## Receipts (SSD `/Volumes/VertigoDataTier/pact/ddm_wr1_20260729/`)

`wr1_descent_receipt.json` (schema `ddm_wr1_reverse_waterfill.v1`) · `wr1_cell_sensitivity_atlas.npz`
· `wr1_kneeA_safe_274k_archive.zip` (sha `b6be1691…`) · `wr1_kneeB_subbar_173k_archive.zip`
(sha `8a75ac38…`) · `wr1_realized_gate_receipt_SCHEMA.json`. Regenerator:
`experiments/ddm_wr1_reverse_waterfill.py` (deterministic; ruff-clean). Gate runbook:
`experiments/stage_wr1_realized_gate.sh`.
