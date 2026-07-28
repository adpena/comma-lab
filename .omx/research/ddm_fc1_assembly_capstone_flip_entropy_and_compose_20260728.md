# ddm_fc1 — THE ASSEMBLY CAPSTONE: STAGE-1 tier-mover H(flip|context) MEASURED + full-codec compose — n600

**Arm:** ddm_fc1 (full-codec assembly). **Base:** worktree off `main@4cdc7517c0` (r2s + iv1–iv4 + cv1
merged). **Axis:** `[macOS-CPU advisory]` — every d_seg/flip realized through the frozen CPU-torch SegNet
on the copy (f0) base; every byte is a REAL compiled-coder output length (constriction Rust range coder /
LZMA1-x9e / WebP). **NOT** a byte-closed `upstream/evaluate.py` row. **Pointer UNMOVED (0.19108 contest-CPU)**
— `score_claim=false · promotion_eligible=false · rank_or_kill_eligible=false`. No number here moves the
pointer.

**Directive:** `fc1_charter.md` (07-28) — build the 5-stage gated codec: STAGE-0 co-measure harness →
STAGE-1 the H(flip|context) tier-mover scalar → STAGE-2 correction coder races → STAGE-3 frame_0 crush →
STAGE-4 pose → STAGE-5 compose+byte-close. Corrected target: box RETIRED; min-S at NEAR-SOLVED distortion,
bar 0.172, aim 0.15.

**NO-FAKE.** The copy-base flip mask reproduces the oc1/r2s n600 aggregate EXACTLY (0.00864213, 1,019,467
sites) as the plumbing self-check (3rd independent reproduction). The STAGE-1 entropy floor is CONFIRMED by
a REAL constriction coder to within 0.05% (41,357 B floor → 41,392 B coded) with a verified lossless
roundtrip — it is an achievable coder rate, NOT an entropy proxy (iv4 A14 warns proxies ≠ coder). The
near-solved d_seg (1.52e-4) in the compose is a BYTE FLOOR (the correction-stream bytes to communicate it
are MEASURED); the REALIZATION operator (label → pixel flip) is UNBUILT (the master-thesis crux) — so no
byte-closed near-solved row is claimed, only the compose arithmetic over measured stream bytes.

## STORES CONSULTED
- `.omx/research/ddm_r2s_stratified_and_sparse_residual_20260728.md` — copy-PREDICT LOCKED (warp family
  CLOSED); 1,019,467 flips per-class Road 50/Lane 25/Undriv 13.5/Movable 6/MyCar 6; support-geom 421 KB;
  residual VALUES 10 MB (DEAD); frame_0 81 MB = binding stream #1; RATE is the wall.
- `.omx/research/ddm_iv4_missing_piece_hunt_20260728.md` — A1 Stream-E-as-Wyner-Ziv-syndrome (H(flip|context)
  the OPEN tier-mover, F6 NULL); A2 co-measure harness MISSING; A5 frame_0 empty predictor; A7 frame_0-crush ×
  support coupling; A14 compiled-coder decode budget; A15 realize the SOLVED distortion not the box;
  Collapse-1 (Stream E incompressible) / Collapse-2 (pose blows up) / §1.5 realization-crux re-proof.
- `.omx/research/ddm_iv3_codec_artist_synergy_bridges_20260728.md` + iv1/iv2 (plug-in inventory).
- `experiments/ddm_oc1_flip_support_measure.py` + `ddm_r2s_stratified_flip_support.py` (SegNet load +
  argmax + flip-vs-lstars machinery — reused, not reinvented). `upstream/modules.py` (SegNet last-frame
  5-class argmax; PoseNet two-frame YUV6 first-6 MSE). `tools/r6cal_byteclose_and_eval.py` +
  `src/tac/optimization/ddm_runtime_exporter.py` (the proven byte-close spine — named, not fired).
- MEMORY: `ms2r_r3_solved_seg_is_box_solve...`, `objective_is_min_S_over_solution_set...`,
  `distortion_byte_economics_are_upper_bounds...`, `meet_it_where_it_is_carry_thing_itself...`,
  `frozen_scorer_exact_factorization...`, `realization_is_quantization_gated...`. CLAUDE.md class order
  L80 (0 Road 1 Lane 2 Undrivable 3 Movable 4 MyCar).

---

## STAGE 0 — CO-MEASURE HARNESS (partial: SegNet-side BUILT + USED; PoseNet-side NAMED)
`experiments/ddm_fc1_context_cache.py` caches the copy-base decoder-side context n600 — copy_argmax
(argmax SegNet(f0)) + copy_margin (top1−top2 logit) — the ONE heavy shared input (174 s, 5 chunks,
resumable). All downstream d_seg / flip / correction measurements flow through the frozen SegNet on this
cache + REAL byte coders (the SegNet-side co-measure IS built). **The PoseNet-side co-measure (composed
d_pose collateral under stream repaint, A2/Collapse-2) is NOT fired** — pose is SETTLED (banked R1 dxi,
stored-target sidecar, not a reconstruction) and the fork is decided by the frame_0 RATE wall
(pose-independent). Named honestly, consistent with r2s's disposition; the A2 PoseNet leg is the named
next instrument if a compact frame carrier ever exists (it does not).

## STAGE 1 — THE TIER-MOVER SCALAR: H(flip-label | free decoder context) — MEASURED n600 (F6 NULL → FILLED)
`experiments/ddm_fc1_flip_entropy.py`. Context = decoder-derivable free features: copy_argmax, copy_margin
bucket, distance-to-copy-argmax-boundary bucket, nearest-adjacent copy class. Two exact plug-in
conditional entropies over decoder-computable cells (the cell partition is decoder-derivable → a
conditional coder using these cell frequencies achieves the rate; the frequency TABLE is a tiny counted
side stream).

| Quantity | MEASURED n600 |
|---|---|
| **H(flip-label \| context)** in-sample plug-in | **0.32454 bits/flip** |
| **H(flip-label \| context)** held-out 2-fold (even/odd pairs, KT-smoothed cross-entropy) | **0.34497 bits/flip** — AGREES (not overfit; 80 occupied cells, ~12.7 K flips/cell) |
| Label floor (all 1,019,467 flips) | **41,358 B** |
| H(flip? \| context) per PIXEL (support / WHERE) | 0.03386 bits/px → **499,248 B** — an OVERESTIMATE (per-pixel model IGNORES the spatial contiguity of the flip contours; the LZMA/contour coders below beat it) |
| Label model table (occupied cells) | 80 cells → ~0.4–0.8 KB |

**Per-class label entropy (mean bits/flip, MEASURED):** Road **0.0896** (50% of support, near-deterministic),
Lane **0.2331** (25%), Undrivable **0.5410** (13.5%), MyCar **0.6180** (8.6%), Movable **0.9532** (7% —
hardest, still < 1). The 75% Road+Lane boundary mass is near-deterministic given local context — this is
the Slepian-Wolf/Wyner-Ziv COSET collapse (A1) MEASURED: the corrected label is almost always the class
across the shifted boundary. **The tier-mover verdict: H = 0.325 bits/flip ≪ the ~1 bit/flip threshold →
the correction VALUES are near-FREE. But the tier-mover is the WHAT, not the WHERE.**

**Concession curve (label bits vs fraction of flips fixed, cheapest-first — the waterfill input):**
fix 50% → 2,158 B · 80% → 4,169 B · 90% → 9,899 B · 95% → 15,767 B · 98.2% (= near-solved 1.52e-4) →
28,606 B · 100% → 36,090 B. Marginal cost stays < 1 bit/flip until ~92% fixed, then rises to ~6 bit/flip
for the last (Movable-island) tail — but even the full fix is only ~36 KB of LABELS.

## STAGE 2 — CORRECTION-STREAM REAL CODERS (compiled cores; decode wall-clock timed) — MEASURED n600
`experiments/ddm_fc1_stage2_coders.py`.

| Stream | REAL coder | bytes | decode | note |
|---|---|---|---|---|
| **LABELS** (1,019,467 corrected labels) | constriction range coder, STAGE-1 context-categorical | **41,392 B** (0.3248 b/flip) | **0.010 s** | **lossless roundtrip TRUE**; == STAGE-1 floor to 0.05% → the tier-mover is coder-REAL |
| label model table | LZMA(occupied cell counts) | 828 B | — | decoder rebuilds identical probs |
| **SUPPORT geometry** (117.9 M binary flip field) | packbits + LZMA1-x9e | **421,366 B** | 0.015 s | reproduces r2s 421,496 B (self-check ✓); per-pixel context-arith (499 KB) LOSES to LZMA — support is SPATIAL |
| **correction stream total** | | **463,586 B** | **0.025 s** | ≪ 1800 s budget (A14 KILLED for this stream) |

**The correction stream is 463.6 KB — already 2.47× the 0.172 bar (187.7 KB) — of which 421 KB is the
WHERE (support geometry) and only 42 KB the WHAT (labels).** The binding SUB-stream within the correction
is the support GEOMETRY. Contour/boundary coding (Road+Lane = 75% are boundary curves; #307 machinery,
`dash_phase_carrier` for Lane) is the NAMED lever to take support 421 KB → ~100–200 KB — UNBUILT here
(and, per the compose below, insufficient regardless).

## STAGE 3 — FRAME_0 CARRIER real crush curve (the #1 binding stream) — MEASURED n600
`experiments/ddm_fc1_stage3_frame0_crush.py`. frame_0 is seg-free (SegNet reads the last frame only); its
only constraint is d_pose. Real WebP crush of the 600 base frames:

| WebP rung | total bytes | B/frame | S rate_term |
|---|---|---|---|
| **Q1 method6 (most aggressive)** | **2,695,020 B** | 4,492 | **1.7945** |
| Q10 | 3,674,044 B | 6,123 | 2.4464 |
| Q30 | 6,053,846 B | 10,090 | 4.0310 |
| Q50 | 9,139,474 B | 15,232 | 6.0856 |
| Q75 | 15,577,530 B | 25,963 | 10.3724 |

**Even the MOST aggressive pose-plausible crush is 2.70 MB → rate_term 1.79 — 10.4× the whole 0.172 bar,
from frame_0 alone.** To fit the ~0.03 rate budget left after banked pose + near-solved seg, frame_0 would
need ~45 KB / 600 = 75 B/frame — ~60× smaller than WebP-Q1. Per-frame image coding cannot; only frame
AMORTIZATION (a trained INR generating all frames from a tiny latent — the BANNED HNeRV/PR lineage) or
backward-prediction (A5, degenerate on the copy base: inverse-warp of f0 by ξ ≈ f0 → residual = the 10 MB
temporal delta). **frame_0 is the binding stream.** d_pose collateral under crush (A7/Collapse-2) NOT
measured (pose banked on the un-crushed base) — named.

## STAGE 4 — POSE (SETTLED, not engineered here)
Baseline `xi_pose_coder(R1 dxi)` = 474–875 B → d_pose 0.001610 → contribution √(10·0.001610) = **0.127**.
The **solved** pose 0.0319 exists only at the 291 MB exact solve (C1_MS1 / charter). The gap between banked
(0.127) and solved (0.0319) pose is **0.108** — it eats most of the sub-bar distortion budget (see compose
E vs D). Naming the demand, not faking a fix.

## STAGE 5 — COMPOSE (arithmetic over MEASURED stream bytes; NO byte-close fired) — the FORK
`experiments/ddm_fc1_stage5_compose.py`. S = 100·d_seg + √(10·d_pose) + 25·B/37,545,489.

| Scenario | d_seg | d_pose | bytes | **S** |
|---|---|---|---|---|
| **A** full measured streams (frame_0 Q1 + correction 463.6 KB + pose) | 1.52e-4 | banked 0.00161 | 3,159,481 | **2.246** |
| B frame_0 FREE + correction measured (463.6 KB) | 1.52e-4 | banked | 464,461 | **0.451** |
| C frame_0 FREE + correction CONTOUR best-case (184 KB) | 1.52e-4 | banked | 185,315 | **0.266** |
| D distortion floor, banked pose, ZERO rate | 1.52e-4 | banked | 0 | **0.142** |
| E distortion floor, SOLVED pose, ZERO rate | 1.52e-4 | solved 1.02e-4 | 0 | **0.047** |

seg_term 0.0152 · pose_term (banked) 0.1269 throughout A–D.

**FORK-3 (S = 2.246 ≫ 0.35): NAME the binding stream + gap arithmetic; typed verdict, FAMILY scope.**

## VERDICT (typed; scope = FAMILY: copy-PREDICT task-lossy-correction codec)
- **STAGE 1 is the win and the whole point:** H(flip-label | free decoder context) = **0.325 bits/flip**
  (held-out 0.345), MEASURED n600, self-check EXACT, coder-CONFIRMED at 42 KB with admissible decode. The
  DSC/Wyner-Ziv coset insight (iv4 A1, the "masters' import") is VALIDATED: the correction VALUES are
  near-free. **F6 is no longer NULL.** But it is NOT the tier-mover the projection needed — because the
  tier that moves S is the WHERE (support geometry) and the FRAME carrier, not the WHAT (labels).
- **Binding stream #1 = frame_0 carrier: 2,695,020 B (WebP-Q1, MEASURED) → rate_term 1.79** (10.4× bar).
  **Binding sub-stream #2 = support geometry: 421 KB (LZMA) / ~184 KB (contour, unbuilt).**
- **The exact gap (why sub-bar is unreachable on this family):** the banked-pose + near-solved-seg
  distortion FLOOR is **0.142** (scenario D), leaving only **0.030 rate budget = 45 KB for ALL streams**.
  The correction stream alone is 463 KB (contour best-case 184 KB); frame_0 is 2.70 MB. Even a FREE frame_0
  with contour-coded support (C) is **0.266** — still **0.094 over bar**, of which **0.127 is pose**. Sub-bar
  requires SIMULTANEOUSLY (a) frame amortization (banned INR lineage), (b) contour support < 45 KB, and
  (c) SOLVED pose 0.0319 (only at a 291 MB solve) — the three-way realization crux, re-proven a third time
  (iv4 §1.5: no artifact is compact AND solved-distortion). **No byte-closed evaluate.py row fired
  (correctly): the near-solved d_seg needs the UNBUILT realization operator, and a copy-base byte-close
  would only reproduce the rate wall (worse than r6cal).**
- **NEXT (routed to MAIN, not this arm):** the ONLY remaining sub-bar path is frame amortization
  (compact generator for the 600 base frames) + compact solved pose — i.e. the realization crux itself, at
  stages 3–4, which every lineage has failed. The correction stream is SOLVED to its floor here; it is not
  the wall.

## PLUG-IN LEDGER (charter assets — USED / RACED / NAMED / N-A)
| Asset | Stage | Disposition | Detail |
|---|---|---|---|
| oc1/r2s SegNet load + flip-vs-lstars machinery | 0/1 | **USED** | reused verbatim (do-not-reinvent); self-check reproduces 1,019,467 sites 3rd time |
| `margin` (top1−top2 logit) | 1 | **USED (built)** | computed in the context cache = the decoder-derivable confidence context; flips median margin 0.39 vs non-flip 5.86 (MEASURED) |
| Wyner-Ziv / DISCUS coset (iv4 A1) | 1/2 | **USED (as the H measurement)** | H(flip-label\|context)=0.325 b/flip IS the Slepian-Wolf H(X\|Y); the coset collapse MEASURED per-class |
| `constriction` (Rust range coder) | 2 | **USED** | real context-categorical label coder, 41,392 B, lossless, decode 0.010 s (kills A14 decode-risk for the label stream) |
| `arith_selfcomp_rate_coders` / LZMA1-x9e | 2 | **USED** | support-geometry packbits+LZMA = 421,366 B (reproduces r2s floor) |
| `context_partition_codec` / `dash_phase_carrier` (#307/#425) contour coding | 2 | **NAMED (unbuilt)** | the support 421 KB → ~100–200 KB lever; insufficient regardless (compose C = 0.266) — not built |
| WebP (compiled image coder) | 3 | **USED** | frame_0 crush curve; Q1 = 2.695 MB → rate 1.79 |
| `keyframe_codec` (#202) / p1 pose-quotient carrier (#715) | 3 | **N-A (dominated)** | frame_0 crush is irreducibly multi-MB at any pose-plausible quality; per-frame carriers can't reach 75 B/frame |
| `xi_pose_coder` (#257) / R1 dxi | 4 | **USED (routed)** | pose SETTLED, 474–875 B, d_pose 0.001610, contrib 0.127 |
| A5 backward-prediction (inverse-warp frame_0) | 3 | **N-A (degenerate on copy base)** | inverse-warp of f0 by ξ ≈ f0 → residual = the 10 MB temporal delta (r2s), not a win |
| A2 PoseNet co-measure (composed d_pose) | 0/4 | **NAMED (not fired)** | pose banked/settled; fork decided by rate wall; the collateral instrument is named for a hypothetical compact carrier |
| `ddm_runtime_exporter` / r6cal byte-close | 5 | **NAMED (not fired)** | near-solved d_seg needs unbuilt realization; copy-base byte-close = rate-wall regime (worse than r6cal), L-cost — correctly not fired per fork-3 |
| `region_merge` MDL (1.273 B/err) | 1/5 | **USED (as concession curve)** | the waterfill water level; concession curve MEASURED (fix-fraction vs label bits) |

## HONESTY + WHAT THIS ARM DID NOT DO
- No score claim; pointer UNMOVED 0.19108. All numbers `[macOS-CPU advisory]`.
- Did NOT build the label→pixel REALIZATION operator (the master-thesis crux) — measured only its byte
  floor. Did NOT run PoseNet (pose banked/settled; fork rate-decided). Did NOT build a contour support
  coder (named; insufficient regardless). Did NOT fire a byte-closed evaluate.py row (correctly — a
  copy-base close is the rate-wall regime; a near-solved close needs the unbuilt realization).
- Artifacts (SSD): `/Volumes/VertigoDataTier/pact/ddm_fc1_20260728/{context_cache_n600.log, chunks/*.npz,
  entropy_n600.json, stage2_coders_n600.json, stage3_frame0_crush_n600.json, stage5_compose_n600.json}`.
- Tools: `experiments/ddm_fc1_{context_cache,flip_entropy,stage2_coders,stage3_frame0_crush,stage5_compose}.py`.
