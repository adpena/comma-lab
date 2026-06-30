# CANONICAL RESEARCH INDEX — the RATE axis (deduplicated, calibrated)

**UTC** 2026-06-29 · **authority** `[$0 CPU research-consolidation / advisory]` · **pointer UNMOVED contest-CPU 0.19110**
**score_claim** false · **promotable** false · **ready_for_exact_eval_dispatch** false · this is a CONSOLIDATION (a MEANS).

**Why this exists** (operator 2026-06-29: "worried about signal loss + rediscovery + STARTING LESS OPTIMAL than
we could with perfect recollection"): exhaustively marshal OUR measured RATE-axis corpus so the next launch +
byte-close starts at TRUE optimal form with nothing left out. Sister index: `canonical_research_index_dseg_20260629.md`
(d_seg axis, subagent `idx-dseg`). The RATE half is **de-risked CHEAP**; the binding sub-0.15 lever lives on the
d_seg axis (the trained generator C7), so this index ends by handing the baton there.

**Score law** (`upstream/evaluate.py:92`): `S = 100·d_seg + √(10·d_pose) + 25·B/N`, `N = 37,545,489`, NO time term.
Only `archive.zip` bytes `B` are counted (`evaluate.py:63`); `inflate.py`/`inflate.sh` code is FREE (rule-118), but
LEARNED / video-derived artifacts (NN weights, lookup tables, per-frame coords) MUST be in `archive.zip` and ARE
counted (README:118, incl. SegNet/PoseNet weights). **Byte price ≈ 6.66e-7 score/byte** (GROUNDED) → 1,500 B ≈ 0.001 S.

**Calibration legend:** `EXACT byte-closed` = a real `archive.zip` stat through `upstream/evaluate.py` (the only
score) · `GROUNDED` = measured artifact cited, but advisory axis (through-R CPU-torch / macOS / 8-pair parity /
small-n) · `ESTIMATE` = derived band · `RETRACTED/SUPERSEDED` = a prior claim corrected (kept for no-signal-loss).

---

## §1. INDEX TABLE (deduplicated)

| # | Finding | Status | Calibration (n / bytes / axis) | Pointer |
|---|---|---|---|---|
| R1 | **0.19110 frontier rate = 0.1185** (177,169 B, PR110/101 recode). The BORROWED frontier (NO-FAKE #7: a recode, not ours-original). | live frontier | EXACT byte-closed, full-600 contest-CPU | `canonical_frontier_pointer.json`; FEED-lb |
| R2 | **Lossless rate on the 0.19110 frontier is EXHAUSTED.** It IS ALREADY the L21–L32 / PR112-L30 recode; every section at entropy floor; finishing-kit `byte_delta=0`. | settled | EXACT (arithmetic proof: 0.19198→0.19110 = −1326 B already-spent L30 recode, decoded-sha identical) | DAG FEED-lb (a27f6ce46) |
| R3 | **finishing-kit "−0.005..−0.008, ~half-day, near-certain sub-0.19" was DOUBLE-COUNTED** — cited the already-spent R1 −1023 B / R2 −317 B as "future" + over-counted S12 (0 bytes on a render substrate storing decoder+latents, no frames). NO-FAKE catch prevented a fake sub-0.19. | RETRACTED | EXACT (verified vs current frontier coding state) | FEED-lb; supersedes FEED-ki/kz/la |
| R4 | **L13 non-RGB witness format = −59% rate** (72,217 B vs 177,169 B), lossless-parity-proven. The custom rendered-witness bytecode (contour/magnitude/ops/interpreter). Pose-carrier ~22.5 KB brotli-q11, d_pose 12.66→**0.006**, survives the uint8 eval round-trip. | format PROVEN (rate half) | GROUNDED, 8-pair parity (NOT 600) | `CAPSTONE_witness_taskspace_roundtrip_byte_floor_20260621`; `custom_witness_format_inflate_interpreter_design_20260623` |
| R5 | **L13 "72 KB lossless-parity sub-0.15" was an OVER-CLAIM** → L13 is S≈0.79 (pose closed by format, **d_seg NOT** closed: L13 d_seg=0.0068, the full-rank island wall). The format packages a given d_seg cheaply; it does not lower d_seg. | SUPERSEDED | GROUNDED correction | `custom_witness_format..._20260623` §1 |
| R6 | **bc20/G3 torch_vehicle = 89,244 B, rate 0.0594** (parity-verified G3 packet, d_pose 0.00034). The cheapest real byte-closeable witness vehicle. | PROVEN byte-close pipeline | EXACT byte-closed; **DUAL exact row [contest-CPU] 0.37797 / [contest-CUDA] 0.39153** | FEED (G3); `_fire_g3_basin_baseline/g3_packet_manifest.json` |
| R7 | **bc20 S-arithmetic (the honest gap):** rate 0.0594 is CHEAP (−0.059 vs frontier) but d_seg-undercapacity dominates → live S 0.489 (Muon-floor projected ~0.31). To beat 0.19110: need d_seg ≤ ~0.00087 (pose-dependent). Rate is NOT the bc20 blocker; d_seg is. | settled (rate cheap / d_seg binds) | EXACT arithmetic on measured d_seg/bytes | DAG L510/L540 |
| R8 | **Deterministic substrate rate is GREEN/CHEAP:** 13 partition keyframes → rate **0.0060** + pose sidecar ~875 B (0.0006) = **0.0066**. 10× conservative reach (130 keyframes) → 0.060. One canonical partition = 809 B (~693 B/pair amortized). | GREEN on rate | GROUNDED through-R, n96 bulk-advisory, ONE 10s highway window | DAG FEED-ll (`screw_reach/reach_n96.json`) |
| R9 | **Store-everything partition rate-WALL = 0.277.** The thing the keyframe+warp substrate (R8) decisively beats (0.0066–0.060 ≪ 0.277 ≪ frontier 0.191). | settled | GROUNDED | FEED-ll; FEED-lk |
| R10 | **BUT the deterministic arm is d_seg-DEAD:** R1-render d_seg floor ≈ 0.0185 bulk / 0.023 full at k=0 ≈ 30–40× the sub-0.15 d_seg budget (~6e-4). A pure-deterministic cheap-rate archive scores S≳2 from d_seg. The reach is carried by partition STABILITY, NOT the warp (d_seg-optimal warp ≈ near-identity). | settled (cheap rate, dead distortion) | GROUNDED, NO-FAKE (k=0 reproduces R1 floor exactly 0.01851==0.0185) | FEED-ll, FEED-lj |
| R11 | **C4 (bulk-jitter) explicit store = rate 0.1185 → S≈0.26 alone** (margin-keyed iid upper bound, 177,926 B). Busts sub-0.15 by itself. → **C4 MUST be folded into the trained generator C7 at training time, not coded post-hoc** (PR95 pattern: jitter rides free with content). The #1 rate decision. | settled (training-time) | GROUNDED (budget_gate M4) | rate playbook §0/§4; budget_gate_overturn M4 |
| R12 | **The PR95 post-hoc coder stack is the FINISHING KIT, all in-tree.** L30 range/arithmetic (R1 −1023 / R2 −317 GROUNDED on FP11) + L31 colex on positions + L25 temporal-delta on pose + S12 resize-null −10–19.5% (if RGB) + L21/22/23/24/26/29 brotli-friendliness + L32 q11. Order ≈ −0.005..−0.008 rate. NOT the breakthrough. | tooling ready; 1 build gap | GROUNDED band (bolton); coders shipped | rate playbook §1–§3; `lossless/range_coder.py`, `codec/pr101_polymorphic.py` |
| R13 | **The ONLY rate build gap = a v2-grammar materializer (~half-day, $0)** to lift the in-tree coders onto the witness container. Coders themselves are shipped (`RangeEncoder`/`Decoder`, `encode_combination_colex`, `encode_huff_length_rank`). | OPEN (low effort) | n/a (engineering) | rate playbook §3 #1 |
| R14 | **int5 score-aware QAT-shrink of the frontier CAPS at S~0.49** (PTQ int5 d_seg collapse; QAT can't recover; byte-closed 118–119 KB rate 0.079). NOT a pointer-mover, NOT sub-0.15. Low-bit-of-frontier is dominated. | settled (capped) | EXACT byte-closed S=0.4828 | `feedback_frontier_int5_score_aware_qat_finetune_path_b_caps_20260618`; `tac.frontier_int5_qat` |
| R15 | **WRQ score-aware per-tensor weight requant on C7** (decoder = ~91% of a NN archive → largest single post-T1 lever). Lossy-in-cell. | OPEN (high ceiling) | UNGROUNDED magnitude (needs own exact sweep) | rate playbook §2; task #69 |
| R16 | **GR rate architecture** (the v2-done-right pipeline): context-tree contour-code the geodesic-lane boundary descriptor (8-dim → hundreds of B) → FREE eikonal-SDF generator → posterior-guided minimal residual (NVRC entropy) → INTEGER-deterministic decode. = rate term E7 (Λ/MDL). | designed | advisory design (E7 in canonical_equations) | `project_gr_unified_action_full_witness_architecture_20260629` |
| R17 | **DM3′ low-rank-GLOBAL additive SDF-correction head** (rank≈16, ~4–10 KB/600 → rate ~0.003–0.007). $0-test FALSIFIED the per-position SPATIAL GRID alternative (ego-motion makes per-pair variation globally low-rank: rank-8 = 95.6%; grid ~8× worse + ~100× bytes). The byte-cheap conditioning carrier. | designed, alt falsified | GROUNDED ($0 rank test) | GR memory; FEED (DM3′) |
| R18 | **Modulation-split FREE-hypernet → INR weights** (COIN++/functa 2201.12904 + D'OH 2403.19163): seed-derived hypernet = ZERO counted bytes; only the low-dim latent counted. Composes with emergent-collapse (latent → intrinsic dim) → smallest-rep BY CONSTRUCTION. The FiLM-alternative for the trained residual. | designed | advisory | FEED-lg; FEED-le |
| R19 | **rule-118 FREE/COUNTED boundary** (the rate game's law): generator ALGORITHM + deterministically-generated tables (Fourier B from seed, rasterizers, solvers, runtime codebooks) = FREE in inflate.py; LEARNED weights + VIDEO-derived payload (per-frame coords, learned residuals) = COUNTED. Forbidden: hide-data-in-code (video-derived table disguised as code). | binding law | n/a (compliance) | CLAUDE.md "inflate.py is a FREE interpreter"; rate playbook §5 |
| R20 | **molt compile-the-free-generator angle** (008 addendum, pact-collab): compile the full decode chain (se3/camera/lane_sdf/levelset/range/coord-INR) to WASM+WebGPU → run the arbitrarily-complex FREE generator inside the 30-min budget; bit-exact-vs-numpy-fp32 + 30-min-T4/CPU contracts. Matures the rule-118 rate-half. | live two-way channel | advisory (molt team building Kernel-A WASM parity) | FEED-lo; `molt_collab_addendum_20260629/008_*` |
| R21 | **Pose rate (solved, near-free):** stored-target sidecar = 6 scalars × 600 = 7,200 B raw / <5,000 B zlib / ~hundreds B via low-rank pose codec rank-2 (#140) → d_pose≈0. Pose carried by STORE, not warp (warp dual-use lossy-REFUTED). | solved | GROUNDED | `src/tac/scorer_targets.py`; FEED-lj; task #140 |
| R22 | **Emergent low-dim collapse = design FOR the smallest-rep:** prior run's blob naturally SHRANK to the intrinsic dim and "fell through." Induce rank/spectral collapse (#110 / A6 nuclear-norm / code-spectral-entropy), store minimal-by-construction. Intrinsic dims: rank-8 lane orbit / eff-rank 4.07 coarse / pose rank-2 / FiLM ~1.2-of-768. Rank FLOOR guard (lane≥8). | design principle | GROUNDED (measured intrinsic dims) | FEED-le; FEED-lf |

---

## §2. OPTIMAL-CONFIG CONTRIBUTION — the measured-optimal RATE path (the marshaled best)

**The one-line answer:** at optimal form the witness rate is **NOT the binding constraint** — it is de-risked CHEAP.
The whole sub-0.15 hinge sits on the d_seg axis (whether the trained generator C7 descends to d_seg~6e-4 at small
bytes). Therefore the optimal-config rate path is to make the rate term *small by construction and recoverable*, then
hand the budget to d_seg. Concretely, in priority order:

**A. TRAINING-TIME (must be in the GPU-run config or LOST — these dominate; cannot be applied post-hoc):**
1. **Fold C4 (bulk-jitter) INTO the trained generator C7** — do NOT ship an explicit dither store (R11). The single
   most important rate decision; an explicit store busts sub-0.15 alone (S≈0.26).
2. **Entropy-penalized / weight-entropy-regularized loss (NVRC/NeuroQuant) on C7** — make decoder weights
   compressible by construction; multiplies every post-hoc coder (R12).
3. **Score-aware QAT** (int5/LSQ/per-channel/outlier #147 + variable-grid #111 + WRQ-as-QAT) — byte-minimal weights
   *for the scorer* (the scorer tolerates far more weight error than reconstruction). Caveat: int5-of-frontier caps
   0.49 (R14) — this is for a from-scratch witness, not for shrinking the borrowed frontier.
4. **Latent-structure regularizer (#110) + emergent-collapse to intrinsic dim** (R22) → smallest-rep by construction.
5. **Modulation-split free-hypernet → INR weights (R18)** for the trained residual — zero counted generator bytes,
   only the tiny collapsed latent counted. NOT vanilla FiLM (rank-1.2 collapse plateau).

**B. THE REPRESENTATION (store only the irreducible video-derived statistic; everything else generated FREE):**
- Geodesic-lane / boundary descriptor: 8-dim → **hundreds of B**, context-tree contour-coded (R16).
- Deterministic backbone (R8): ~13 partition keyframes (**0.0060**) + screw-warp (se3, FREE generator) + pose sidecar
  (R21, ~875 B → 0.0006) = **~0.0066 rate** for the bulk partition + pose.
- Low-rank-global additive SDF-correction head (R17): rank≈16, **~4–10 KB/600** (rate ~0.003–0.007) — the byte-cheap
  conditioning carrier (NOT a spatial grid, NOT FiLM).
- Movables residual: small (~0.75–2.7 KB).
- Trained lane-survival residual C7: **the bulk** and the binding unknown (d_seg-axis question, not rate).

**C. FREE in inflate.py (rule-118, R19/R20):** eikonal-SDF generator, screw-warp/se3, lane-curve rasterizer,
coord-INR forward pass, range decoder, prefix-sum, colex-unrank — all generic algorithm, compiled (molt → WASM) to
run the arbitrarily-complex deterministic program inside the 30-min budget. Counts ZERO bytes.

**D. POST-HOC FINISHING KIT (at byte-close, on the witness container; the LAST fraction, ≈ −0.005..−0.008, NOT the
breakthrough):** L30 range-code every int section → S12 resize-null −10–19.5% *only if the archive ships RGB frames*
(on a render-substrate that stores decoder+latents this yields 0 — the R3 over-count lesson) → L31 colex on position
sets + L25 temporal-delta on pose → L21/22/23/24/26/29 brotli-friendliness bundle (additive on DISJOINT tensors only)
→ L32 brotli-q11 (free) → WRQ on C7 (R15, highest ceiling). Sequencing law: disjoint-section lossless moves SUM;
same-section moves SUBSUME (do NOT double-count — the R3 trap). The only build gap is the v2-grammar materializer (R13).

**Net optimal-form rate budget (advisory):** deterministic backbone ~0.0066 + SDF head ~0.003–0.007 + movables/lane
descriptors ~sub-0.001 + C7 trained weights (the dominant, training-time-minimized term). The witness RD-curve
(R-curve, advisory) says the optimum is NOT at 89 KB — right-size UP: B*≈122 KB → S~0.134 in the optimistic+directional
corner; bare-witness B*≈150 KB → S~0.166–0.186 (clears sub-0.19, sub-0.15 only in the directional-ON corner). See §4
conflict for the honest reconciliation.

---

## §3. OPEN / HEADROOM — high-EV unmeasured rate items (ranked)

1. **v2-grammar materializer (~half-day, $0)** — the ONLY build gap to harvest the entire finishing kit (R13). Lift
   the shipped `range_coder` / `encode_combination_colex` / `encode_huff_length_rank` onto the v2 witness container.
   Highest EV-per-effort; pays the moment a real C7 exists.
2. **C4-fold-into-C7 measurement (training-time, GPU)** — does the trained generator emit the bulk jitter free with
   content (PR95 pattern)? Decisive: explicit store busts sub-0.15 alone (R11). This is a config decision baked at
   training time; losing it dominates everything.
3. **Full-clip screw-reach across turns/traffic** — R8's 13-keyframe / 0.0060 rate was ONE ~10s highway window
   (steady forward driving → partition naturally stable). The 60s drive has ~15 scene-turns; keyframe cost across
   turns is UNTESTED. Rate conclusion is robust to ≥10× reach-shortening (→0.060 ≪ 0.277), but measure the BEST
   turnover representation (screw-trajectory / SDF-evolution / INR — NOT MPEG keyframes; the anti-MPEG family).
4. **WRQ score-aware weight requant on C7 (R15)** — UNGROUNDED magnitude, highest *ceiling* (C7 ≈ 91% of bytes).
   Needs its own exact-authority sweep once C7 has descended near-frontier distortion.
5. **DM3′ low-rank SDF-head byte-close (R17)** — the rank≈16 / 4–10 KB ESTIMATE needs a real byte-close to confirm
   rate ~0.003–0.007 and the rank floor.
6. **molt WASM bit-exact decode (R20)** — closes the rule-118 free-generator path for the contest runtime (30-min,
   bit-exact-vs-numpy-fp32). Read molt report 007 + their live blocker (mutual-elevation, task #187).

---

## §4. CONFLICTS / SUPERSEDED (reconciled — no signal loss)

- **"Rate EXHAUSTED on the frontier" (R2/FEED-lb) vs "rate CHEAP on the witness" (R4/R6/R8)** — NOT contradictory,
  DIFFERENT OBJECTS. The EXHAUSTED finding is about the **0.19110 BORROWED PR-recode frontier** (already at the
  entropy floor; lossless `byte_delta=0`; nothing left to squeeze losslessly). The witness is a **different, SMALLER
  representation** whose rate (deterministic backbone 0.0066 + SDF head + C7) is a separate, much-lower budget. You
  cannot losslessly shrink the borrowed frontier, but a smaller-representation vehicle has its own cheap rate. The
  way to cut rate now = a SMALLER REPRESENTATION (v2-done-right), NOT a coder on the frontier.
- **Witness RD-curve: FEED-cc B*~122 KB→S 0.134 vs the later anchored-curve B*≈150 KB→S 0.166–0.186** — RECONCILE:
  sub-0.15 at B* is reachable ONLY in the optimistic + directional-ON corner (basis −48% directional-Fourier +
  capacity-routing + chroma); the bare-witness measured-anchored RD-optimum clears sub-0.19 but NOT sub-0.15. Both
  AGREE the 89 KB single point was deep in the steep cliff (NOT the optimum) → right-size UP. The fixed-rate 4-review
  gate that measured ONE point (89 KB→S0.216) missed the ~0.025–0.08 curve swing.
- **L13 "72 KB lossless-parity sub-0.15" (SUPERSEDED → R5)** — L13 is S≈0.79 (pose closed, d_seg open). The −59%
  rate WIN stands (R4); only the "sub-0.15" was over-claimed.
- **finishing-kit "−0.005..−0.008 near-certain sub-0.19" (RETRACTED → R3)** — double-counted (R1 −1023 / R2 −317
  already spent into the 0.19110 frontier) + over-counted S12 (0 bytes on a render substrate). NO-FAKE caught it.
- **FEED-kn "rate IMPROVES at n600 via amortization" (RETRACTED, FEED-kz C2)** — for a forward-driving ~60s clip
  (≈1.7 km, ~15 scene-turns), canonical-scene/keyframe cost GROWS with n; n96 (where 0.103 measured) is the
  artificially-cheap low-turnover regime. The n-amortization optimism was the wrong direction.
- **C4 rate 0.062–0.084 (FEED-kd/kf) vs 0.1185 (budget_gate M4)** — RECONCILE: 0.062–0.084 = an earlier *partial*
  waterfill model (only flips clearing break-even / annulus-localized); M4 0.1185 = the *full* margin-keyed store
  (iid upper bound). Both say explicit C4 store is rate-expensive → fold into C7 regardless (R11).

---

## §5. DAG FEED (tight)

**FEED — CANONICAL RATE-AXIS INDEX ($0 consolidation, advisory, pointer UNMOVED 0.19110).** Deduplicated 22 rate
findings into one index. **Optimal-config rate path:** rate is de-risked CHEAP — deterministic backbone (13 keyframes
0.0060 + pose sidecar 0.0006 = 0.0066, GROUNDED through-R n96; beats the 0.277 store-everything wall) + low-rank SDF
head (~4–10 KB ESTIMATE) + C7 trained residual (the binding unknown, a d_seg-axis question). Store ONLY the irreducible
video-derived statistic (8-dim contour-coded → hundreds B + pose store + movables); generate everything else FREE in
inflate.py (rule-118, molt→WASM). The PR95 L21–L32 coder stack is the in-tree FINISHING KIT (≈ −0.005..−0.008, last
fraction, NOT the breakthrough); the only build gap is a ~half-day v2-grammar materializer. **#1 training-time decision:
FOLD C4 into C7** (explicit dither store = 0.1185 → S≈0.26 alone). **Calibration corrections preserved:** lossless rate
on the 0.19110 BORROWED frontier is EXHAUSTED (byte_delta=0, finishing-kit double-count caught by NO-FAKE); L13 −59% is
real but L13-the-vehicle is S≈0.79 (pose closed, d_seg open); the witness RD-optimum is B*≈122–150 KB (right-size UP),
sub-0.15 only in the directional-ON corner; "rate exhausted" (borrowed frontier) and "rate cheap" (witness) are
different objects, reconciled. **Top-3 OPEN:** (1) v2-grammar materializer ~half-day $0; (2) C4-fold measurement at
training time (decisive); (3) full-clip screw-reach across turns. **Hands the baton to the d_seg axis** (sister index
`canonical_research_index_dseg_20260629.md`): rate is not the binding sub-0.15 lever — the trained generator's
d_seg-at-bytes is. Memo: `.omx/research/canonical_research_index_rate_20260629.md`. means≠ends; pointer UNMOVED 0.19110.

**Primary citations:** `witness_rate_attack_playbook_20260629T224719Z.md` (technique×component matrix, GROUNDED/ESTIMATE
deltas); DAG FEED-ll (`screw_reach/reach_n96.json`, deterministic backbone 0.0066) / FEED-lb (frontier lossless
exhausted, arithmetic proof) / FEED-lj (warp dual-use refuted) / FEED-cc (RD-curve B*); `CAPSTONE_witness_taskspace
_roundtrip_byte_floor_20260621` + `custom_witness_format_inflate_interpreter_design_20260623` (L13 −59% format);
`_fire_g3_basin_baseline/g3_packet_manifest.json` (bc20 89,244 B / rate 0.0594 / dual exact row); `feedback_frontier
_int5_score_aware_qat_finetune_path_b_caps_20260618` (int5 cap 0.49); `project_gr_unified_action_full_witness
_architecture_20260629` (E7 rate-Λ, contour→eikonal-SDF→residual→integer-decode, DM3′); `src/tac/lossless/range_coder.py`
+ `src/tac/codec/pr101_polymorphic.py` + `src/tac/scorer_targets.py` (in-tree coders); CLAUDE.md "inflate.py is a FREE
interpreter" + L21–L32.
