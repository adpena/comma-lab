# ddm_da1 — TELEMETRY DECOMPOSITION: every fc1/r2s/oc1 aggregate taken apart (bare composite = UNMEASURED)

**Arm:** ddm_da1 (telemetry decomposition). **Base:** worktree off `main@d41cba1b10` (fc1 merged).
**Axis:** `[macOS-CPU advisory]` — every d_seg/flip realized through the frozen CPU-torch SegNet on the
copy(f0) base; every byte a REAL compiled-coder length (LZMA1-x9e-FORMAT_RAW / WebP / plug-in
conditional entropy over decoder-derivable cells); d_pose through the frozen CPU-torch PoseNet.
**NOT** a byte-closed `upstream/evaluate.py` row. **Pointer UNMOVED 0.19108 (contest-CPU)** —
`score_claim=false · promotion_eligible=false · rank_or_kill_eligible=false`. No number here moves it.

**Directive:** `da1_charter.md` (07-28) + two coordinator corrections (D5 pose-slope; D4 bit-depth).
Take every fc1/r2s/oc1 aggregate apart with REAL measurements and reprice the walls.

## STORES CONSULTED
- `.omx/research/ddm_fc1_assembly_capstone_flip_entropy_and_compose_20260728.md` (the aggregates:
  support 421,366 B; frame_0 WebP-Q1 2,695,020 B; H(flip|ctx) 0.32454 b/flip / 41,392 B; compose S).
- `.omx/research/ddm_r2s_stratified_and_sparse_residual_20260728.md` (copy-PREDICT LOCKED; residual
  VALUES 10.06 MB @ 10.3 B/err; per-class flip shares; frame_0 = binding stream #1).
- `.omx/research/ddm_iv4_missing_piece_hunt_20260728.md` (A7 frame_0-crush×support coupling; cb1
  opposite-sign carriers Lane +22.7 / MyCar −0.18 d_pose; Collapse-2 pose blowup).
- fc1 stage JSONs on SSD `/Volumes/VertigoDataTier/pact/ddm_fc1_20260728/` (entropy/stage2/stage3/stage5);
  cached chunks `chunks/ctx_*.npz` (copy_argmax + copy_margin, REUSED — no SegNet re-run for D1/D3).
- `experiments/ddm_fc1_flip_entropy.py` + `ddm_oc1_flip_support_measure.py` (context-feature + SegNet
  machinery reused). `src/tac/boundary_math/range_a_projection.py` (#520 exact P_range(A)).
- R1 banked receipts `/Volumes/VertigoDataTier/pact/comma-lab_latest_20260709/reports/r1_dxi_238/
  n600_shipdxi.json` + run `experiments/results/levelset_n600_R1_storenothing_descent_ev1_20260703T004906Z/
  relaunch.log` (pose descent trajectory).
- MEMORY: `pose_plane_proximity` (D5 ladder), `realization_is_quantization_gated` + #532 (D4 uint8 is
  FRAME-path only), `frozen_scorer_exact_factorization` (range(A) both scorers), `decompose_every_headline`.
  CLAUDE.md class order L80 (0 Road 1 Lane 2 Undrivable 3 Movable 4 MyCar).

---

## THE REPRICED WALL TABLE (fc1 aggregate → decomposed floor → named lever + verdict_scope)

| Wall | fc1 aggregate | DECOMPOSED floor / structure | Named lever + verdict_scope |
|---|---|---|---|
| **D1 support geometry** | 421,366 B LZMA (0.02858 b/px) | Road 213,581 (50.7%) + Lane 145,738 (34.6%) = **85% Road/Lane boundary**; sum-per-class 482,954 (+14.6% separation overhead). Temporal: XOR-delta 582,960 (1.38× WORSE); static-freq predictor empty (68 B). Boundary drifts only **1.41 px median** pair-to-pair (p90 9.06). | Temporal redundancy NOT exploitable at MASK granularity — **INSTANCE/FORMULATION verdict, FALSIFIER TRIGGERED**. Redundancy lives in the CONTOUR (1.4px-coherent boundary curve), not the pixel mask → routes to fc1's contour-support (142 KB UNBUILT), not delta-coding. |
| **D2 frame_0** | 2,695,020 B WebP-Q1 (rate 1.79) | 4-Q curve: pose_term CATASTROPHIC at every rung (**0.75 @ 4.65 MB → 1.04 @ 2.70 MB**, ≥6× banked, ≥4.4× the bar); support grows 1.19–1.53×; **56.5–57.3% of crush error scorer-invisible (ker A)**. The 2.7 MB prices STORED-real-f0; banked pose uses a store-nothing WARP carrier (0.127 @ ~0 B). | WebP-stored frame_0 is pose+rate DOMINATED by the warp carrier — a store-real-f0 FALLBACK, not a floor. range(A) drops ~56% for free but can't rescue pose. **FAMILY scope**. |
| **D3 label H-table** | 41,392 B (0.3248 b/flip, 80 cells) | **73.9% of label bytes in TOP-10 cells, all Road-adjacent-Lane boundary cells**. Determinism from BOUNDARY-DISTANCE (bd0 far-from-boundary = 98% deterministic, 0.14 b/flip), NOT margin (margin-only H stays 1.7-2.0 b/flip; no near-deterministic margin bin). | Label stream already near-optimal (41,358 floor = coded 41,392). Restrict support to the LOW-bdist boundary annulus, DERIVE far-from-boundary labels — INSTANCE verdict; the lever is bdist, not margin (charter's margin-threshold hypothesis FALSIFIED for margin alone). |
| **D4 values** | 10.06 MB @ 10.3 B/err (r2s) | range(A) residual flips 98.87% of sites; minimal uint8 amplitude **median 1.11, p90 7.78; 64% ≤2 steps**. Implied alphabet **~1.7–4 b (+sign)**, NOT int8×3's 24 b. | int8×3 storage is OVER-PRECISION (H0≈8 smooth-signal signature), not incompressibility. Reprice values via amplitude+sign+context (~4-5 b) → low-single-MB; ≤2-step 64% near amplitude-free. verdict_scope INSTANCE (encoding over-precise). |
| **D5 pose** | 0.127 (banked R1 dxi 0.001610) | **STOPPED-DESCENT artifact**: seg plateaued (~0.0046) while d_pose still dropping **−1.26%/epoch (log)**; run killed ep1130. Carrier ladder (proximity law): exact-solve 9.3e-10→0 · PR130 2.33e-5→0.015 · banked 0.00161→0.127 · box 0.0166→0.408. | **~0.095 S left on the table at ZERO added bytes** (~217 epochs of unchanged-byte descent banked→solved-plane). 0.127 is NOT a floor — it is carrier-plane-DISTANCE, and the descent was cut short. FAMILY scope. |

---

## D1 — SUPPORT GEOMETRY 421 KB decomposition (MEASURED)
Driver `experiments/ddm_da1_d1_support_decomp.py`; receipt `d1_support_decomp_n600.json`.
Baseline concat-LZMA **reproduces fc1 exactly: 421,366 B** (self-check ✓).

**(a) PER-CLASS (owned by TRUE label lstars at flip):**
| Class | flips | LZMA B | B/flip | share |
|---|---|---|---|---|
| Road | 464,306 | 213,581 | 0.460 | 45.5% |
| Lane | 279,750 | 145,738 | 0.521 | 27.4% |
| Undrivable | 115,327 | 57,697 | 0.500 | 11.3% |
| Movable | 72,225 | 42,285 | 0.585 | 7.1% |
| MyCar | 87,859 | 23,653 | 0.269 | 8.6% |
Sum-of-per-class = 482,954 (separation overhead **+61,588 / +14.6%** vs joint). **Road+Lane own 85.2%
of the geometry bytes** — the codim-1 Road/Lane separatrix annulus IS the support wall.

**(b) CROSS-PAIR conditional (REAL coder bytes):**
- baseline independent (joint concat) = 421,366 B
- XOR-delta flip[t]^flip[t−1] = **582,960 B (1.38× WORSE)** — consecutive masks are near-independent at
  the pixel level (XOR roughly DOUBLES the support).
- g4-style static-frequency predictor (freq>0.5): residual 421,366 + predmap 68 B = 421,434 (**1.0002×**,
  no gain) — no pixel is a persistent-enough flipper for a static prior.
- **FALSIFIER TRIGGERED**: conditional ≥ baseline ⇒ temporal redundancy NOT exploitable at mask
  granularity. Typed **INSTANCE/FORMULATION** verdict (mask-granularity delta-coding, copy base).

**(c) WORLDSHEET boundary motion:** median-of-pair-medians **1.41 px**, median-of-pair-p90 **9.06 px**
(599 consecutive pairs). The flip-support boundary is CONTOUR-coherent (~1.4 px drift) but not
pixel-exact — which is exactly why (b) fails at mask granularity and why the redundancy is addressable
only in a boundary-CURVE representation (fc1's contour-support 142 KB best-case, UNBUILT).

## D3 — H-TABLE 0.325 b/flip MASS decomposition (MEASURED)
Driver `experiments/ddm_da1_d1c_d3_boundary_htable.py`; receipt `d1c_d3_boundary_htable_n600.json`.
Total label bytes from cells = **41,357.8** (matches fc1's 41,358 floor ✓).

**(a) TOP-10 cells = 73.9% of the 41,358 B.** Every top cell is copy=Road, adj=Lane:
| copy | bdist | adj | flips | b/flip | B | dominant→ |
|---|---|---|---|---|---|---|
| Road | 1 | Lane | 52,386 | 1.553 | 10,169 | Lane 0.61 |
| Road | 6 | Lane | 64,462 | 0.481 | 3,878 | Lane 0.92 |
| Road | 2 | Lane | 22,288 | 1.294 | 3,605 | Lane 0.71 |
| Road | 4 | Lane | 30,635 | 0.607 | 2,323 | Lane 0.90 |
| Road | 3 | Lane | 16,446 | 0.977 | 2,008 | Lane 0.81 |
| Road | 5 | Lane | 47,522 | 0.327 | 1,940 | Lane 0.95 |
| Undriv | 1 | Road | 16,415 | 0.923 | 1,894 | Road 0.69 |
| Movable | 1 | Road | 12,536 | 1.120 | 1,755 | Road 0.51 |
| Road | 0 | MyCar | 73,591 | 0.168 | 1,547 | MyCar 0.98 |
| Road | 0 | Lane | 81,854 | 0.140 | 1,430 | Lane 0.98 |

The cost concentrates at CLOSE-to-boundary (bd1-2) Road↔Lane cells where the relabel is genuinely
ambiguous (b/flip 1.3-1.6, dominant 0.6-0.7). Far-from-boundary (bd0) cells are cheap (0.14-0.17
b/flip, 98% deterministic).

**(b) b/flip vs COPY-MARGIN decile** (H(label | margin-bucket-ALONE)):
low margin [0,0.1) = 1.997 b/flip (15% of flips) … high margin [8,16) = 0.163 (0.1%). Inverse, as
expected — but margin ALONE never gets below ~1.06 b/flip in the mass. Most label predictability comes
from the SPATIAL context (copy_argmax+bdist+adj → 0.325), not margin (~1.8 alone).

**(c) DETERMINISM threshold:** ZERO fraction of flips fall in a near-deterministic margin bin
(H<0.5) under margin-alone context; `margin_threshold_H_below_0p1_bit = null`. **The charter's
"margin threshold above which labels are ~free" is FALSIFIED for margin alone** — determinism is a
BOUNDARY-DISTANCE property (bd0 = 98% deterministic), not a margin property. Lever = restrict support
to low-bdist annulus; derive the far-from-boundary tail.

## D5 — POSE 0.127 decomposition (MEASURED trajectory + carrier ladder)
Driver: R1 receipt + `relaunch.log` parse; receipt `d5_pose_decomp.json`.
Run `levelset_n600_R1_storenothing_descent_ev1_20260703T004906Z`, stopped **ep1130** (killed;
in-flight verdict at ep1127), last completed verdict ep1108.

**Post-collapse pose trajectory (training-implied d_pose, seg FLAT ~0.0046 throughout):**
ep1021 0.003343 → ep1040 0.001842 → ep1054 0.001397 → ep1074 0.001080 → ep1108 0.001012.
**Terminal slope −1.26%/epoch (log), −2.3e-5/epoch (linear)** — monotone descent, NOT converged.

**Carrier ladder via the pose_plane_proximity law (pose cost = distance from solved planes):**
exact-solve 9.3e-10 → 0.0000 (0 B, 96/96) · PR130 12-dim low-freq 2.33e-5 → 0.0153 (23 KB) ·
banked R1 dxi (copy base, STOPPED) 0.001610 → 0.127 (7.2 KB dxi) · box-solve ms2r_r3 0.0166 → 0.408.

**DECISIVE:** the 0.127 is a STOPPED-DESCENT artifact, not an irreducible floor. seg had plateaued
while pose was still descending; extrapolating the terminal −1.26%/epoch, ~**217 more unchanged-BYTE
epochs** carry pose from banked 0.00161 to the solved-plane 1.02e-4 — a pose_term drop of **0.095**
(0.127 → 0.032) at ZERO added bytes. Per-pair tail-vs-uniform is subordinate and NOT reached (packet
deleted; witness regen out of scope) — the aggregate is carrier-plane-DISTANCE driven (the ladder is
the decomposition), so the lever is "finish the descent / move to a closer carrier plane," FAMILY scope.

## D2 — frame_0 2.7 MB ROLE decomposition: the A7 coupling curve (MEASURED n600)
Driver `experiments/ddm_da1_d2_frame0_coupling.py`; receipt `d2_frame0_coupling_n600.json`. WebP
method 6. d_pose = collateral of crushing ONLY frame_0 (PoseNet(crushed_f0, gt_f1) vs PoseNet(gt_f0,
gt_f1)), read through the pose_plane_proximity law; cb1 warns repaint pose sign is class-dependent.

| WebP Q | webp bytes | S_rate | d_seg | ×uncrushed | d_pose | pose_term √(10·) | ker(A) invisible |
|---|---|---|---|---|---|---|---|
| 1 | 2,695,020 | 1.7945 | 0.013222 | 1.53× | 0.10723 | **1.0355** | 0.565 |
| 5 | 3,201,840 | 2.1320 | 0.011614 | 1.34× | 0.08282 | 0.9101 | 0.568 |
| 10 | 3,638,474 | 2.4227 | 0.010958 | 1.27× | 0.06434 | 0.8021 | 0.570 |
| 20 | 4,646,592 | 3.0940 | 0.010241 | 1.19× | 0.05611 | 0.7491 | 0.573 |

(Q1 reproduces fc1's WebP-Q1 2,695,020 B exactly ✓. WebP quality is inverted: low Q = aggressive
crush = fewer bytes; higher Q = less crush = more bytes + better fidelity.)

**(a) SUPPORT growth:** crushing f0 grows the copy-base flip support 1.19×–1.53× (0.00864 → 0.0102–
0.0132). Modest — SegNet argmax is fairly crush-robust.

**(b) d_pose collateral — the HARD wall:** pose_term is CATASTROPHIC at EVERY rung (0.75 at the
gentlest 4.65 MB crush, up to 1.04 at Q1). Even the least-crush rung is **6× the banked 0.127** and
**4.4× the entire 0.172 bar**. This is Collapse-2 made REAL n600: a WebP-stored frame_0 cannot be
crushed to any rate that helps without wrecking pose. (cb1 cross-check: pose sign is class-dependent;
here the aggregate collateral is uniformly large and positive.)

**(c) range(A) split:** **56.5%–57.3% of the crush error is scorer-invisible (ker A), stable across
Q.** More than half of every frame_0 rung's bytes pay for detail NEITHER scorer can resolve. A
range(A)-sufficient carrier (store only P_range(A)(frame_0)) drops that ~56% for FREE — but it does
NOT rescue pose, because the *visible* half is what PoseNet reads and it is already pose-catastrophic.

**REFRAME (the decisive reprice):** the fc1 "frame_0 = 2.7 MB binding stream" prices frame_0 as
**stored real f0 pixels**. But the banked R1 pose used a **store-nothing WARP carrier** (frame_0 =
warp(witness's OWN render, ξ); ~0 marginal bytes) and reached d_pose 0.00161 / pose_term 0.127. So:
(i) for POSE, storing real f0 is DOMINATED by the warp carrier (0.127 vs ≥0.75 at multi-MB); (ii) for
SEG, frame_0's only role is the copy-base, already priced by D1's support geometry. The 2.7 MB WebP
stream is a **store-real-f0 FALLBACK**, pose-dead and rate-dead — NOT a fundamental floor. The real
frame_0 solution is the warp carrier (conditional on the UNBUILT witness realization). **verdict_scope:
FAMILY** (WebP-stored frame_0 carrier is pose+rate dominated; the warp carrier is the live path).

## D4 — VALUES 10 MB repricing: minimal-amplitude line-search (MEASURED)
Driver `experiments/ddm_da1_d4_minimal_amplitude.py`; receipt `d4_minimal_amplitude.json`. 60 evenly-
spaced pairs, 100,596 flip sites. Line-search the range(A)-projected GT residual f0 + round(P_rangeA(α·
(f1−f0))), uint8-clamped, per-site smallest correcting α.

- **range(A) residual is scorer-sufficient**: 98.87% of flip sites flip to the labeled class with the
  full range(A) residual (never_corrected only **1.13%** — those need ker(A) or >1× amplitude).
- **Correction-α curve**: 21% corrected by α≤0.20, 51% by α≤0.50, 78% by α≤0.80, 99% by α=1.0 (most
  sites need a substantial FRACTION of the residual — but the residuals are themselves SMALL at boundary
  flips, so the absolute amplitude is tiny).
- **Minimal uint8 amplitude at correction**: **median 1.11 steps, p25 0.36, p75 3.33, p90 7.78, mean
  2.90**. **64.1% of flips need ≤2 uint8 steps; 78.8% need ≤4.**
- **Implied stored-symbol bit-depth (coordinator D4 extension)**: alphabet = log2(2·N+1)+sign →
  **median 1.69 b, p75 2.94 b, p90 4.05 b (+1 sign)** — NOT the 8 b/channel × 3 = 24 b that r2s's int8×3
  storage spends. **The r2s 10.06 MB @ 10.3 B/err is OVER-PRECISION storage, not incompressibility** —
  the per-#532 rule, uint8 is structurally required only on the FRAME path, not on stored value symbols;
  the value alphabet's true dynamic range is ~4–5 bits (p90) at the boundary annulus. Diagnostic: this
  is the same over-precision signature the coordinator flags (int8 H0≈8 on a smooth signal, cf. ms2r_r3
  plane H0=7.999) — the residual is small-amplitude and boundary-local, not high-entropy.

**Reprice:** the label stream already NAMES the target class (D3); the value's only job is a ~median-1,
p90-8 uint8 step at the site. Coding amplitude(~4-5 b)+sign+context instead of int8×3 reprices the
values from 10.06 MB toward the low-single-MB regime, and the ≤2-step 64% is near amplitude-free
(sign+context). verdict_scope INSTANCE (the value ENCODING is over-precise; support-count × alphabet is
the real floor, not int8×3). NOTE: even repriced, values remain a large stream — but they are DEAD only
in the sense that D2's warp carrier + D1/D3 correction already carry the seg content; values are the
r2s-style stored-residual fallback, and their over-precision is now measured.

---
## HONEST BOUNDARY (decompositions not reached + why)
- **D5 per-pair d_pose distribution** NOT reached: the R1 packet was deleted and regenerating witness
  frames is a full MLX render (out of scope/time). Delivered instead: the pose-vs-step terminal SLOPE
  (the coordinator-flagged decisive quantity) + the carrier ladder via the proximity law. The
  tail-vs-uniform question is subordinate to (and answered by) the carrier-distance framing.
- **D2 d_pose** is the crush collateral of a **WebP-stored** frame_0, not the warp carrier; the warp
  carrier's per-pair pose is the banked aggregate only (same D5 packet-deleted limit).
- All numbers `[macOS-CPU advisory]` — real coders/scorers, NOT byte-closed evaluate.py rows. Pointer
  UNMOVED 0.19108.
