# UNIT C — frame_0 seg-freedom pricing + chroma-HF pose-null/seg-lever DOFs (MEASURED) — 2026-07-10

**Task #394** (operator GO *"Pursue all unexploited"* / *"Pursue all those discoveries optimally"*).
Measures the two STRUCTURAL pose-free seg DOFs surfaced in
`upstream_scorer_alldim_reread_20260710.md` (RANK-1 frame_0 seg-freedom, RANK-2 chroma-HF). **$0, CPU,
faithful to `upstream/modules.py` + `frame_utils.py` (no wrapper drift; no upstream edits).**
**Axis `[macOS-CPU advisory]` / `[through-R n600]` — NON-PROMOTABLE. Pointer contest-CPU 0.19110 UNMOVED
(these are MEANS).**

**STORES CONSULTED:** `upstream_scorer_alldim_reread_20260710.md` (the DOF source; RANK-1/RANK-2) ·
`cascade_c_prime_..._20260526.md` (frame-0 seg_delta=0.0 structural) ·
`cascade_c_posenet_null_segnet_region_waterfill_..._landed_20260526.md` (chroma pose-null decile) ·
canonical eq `posenet_luma_chroma_sensitivity_asymmetry_v1` (the 11.1× Jacobian + derivations this
UNIT verifies) · MEMORY L68 (pose banked-as-artifact) / L74 (task-RD) · `docs/operating_manual_craft_handoff.md`
(§4 re-derive-from-primary, §5 label MEASURED/DERIVED). Sibling `tac/through_r/stem_perception.py`
(partial build) NOT touched; store `segnet_texture_perception_20260710` does not exist yet → used simple
band-limited patterns as instructed.

**Method faithfulness (MEASURED sanity):** my direct-upstream forwards reproduce the caches exactly —
PoseNet vs cached `gt_poses` = **2.1e-12**; SegNet argmax vs `lstars` = **0.000** (bit-identical). So
every number below is on the exact scored path.

---

## MEASUREMENT 1 — frame_0 seg-freedom is REAL and PRICED (RANK-1: CONFIRMED)

### 1a. Construction check — frame_0 has ZERO d_seg term (n600 verdict)
SegNet.preprocess_input uses `x[:, -1, ...]` (`modules:108`) → only frame_1 is scored. Passing
**random-noise frame_0 + real frame_1** through the real SegNet:
- n16 subset: argmax **bit-identical** to the clean-f0 argmax (0 pixels differ).
- **n600 [through-R]: d_seg = 8.48e-9 = 1 pixel / 117,964,800.** The single flip is CPU-forward argmax
  tie non-determinism (seg input is byte-identical regardless of f0), **not** an f0 dependence.
- **VERDICT (MEASURED, scope INSTANCE→FORMULATION): frame_0's d_seg obligation is identically 0 at full
  scale.** Half the rendered pixels (all of frame_0) owe NOTHING to d_seg. v8's per-class carriers
  currently pay seg-fidelity on both frames — frame_0 should drop to a pure pose carrier.

### 1b. The priced DOF — bytes-freed vs d_pose-cost curve (n96 real GT pairs, [macOS-CPU advisory])
d_pose = MSE(PoseNet([degraded_f0, gt_f1])[:6], gt_poses); Δ is pure pose cost of the f0 degradation
(seg cost = 0 by 1a). gzip(uint8) = advisory information-content proxy for a codec's bytes:

| frame_0 mode | d_pose | √(10·d_pose) contribution | gzip B/frame | bytes freed |
|---|---|---|---|---|
| full (reference) | 2.5e-12 | 0.0000 | 1,694,036 | — |
| blur_2 | 1.30e-3 | 0.114 | 1,609,304 | 5% |
| blur_4 | 1.84e-2 | 0.429 | 1,458,141 | 14% |
| blur_8 | 2.48e-1 | 1.574 | 1,159,828 | 32% |
| **luma_only (kill chroma)** | **3.24e-3** | **0.180** | **554,183** | **67%** |
| blur_16 | 20.83 | 14.43 | 763,065 | 55% |
| pose_minimal (gray+blur16) | 18.50 | 13.60 | 176,937 | 90% |

**HEADLINE (MEASURED):** the efficient frontier is **luma-only frame_0** — freeing **67%** of the gzip
bytes for a pose cost of **√10·=0.180** (Δd_pose 3.24e-3), with **Δd_seg = 0 exact**. Aggressive spatial
blur (blur_16 / pose_minimal) is DOMINATED: it frees more bytes but pose explodes (√10· ≥ 13). Chroma
removal is the good f0 knob; spatial resolution is the bad one. **These are UPPER bounds** on the cost —
a *trained* f0 pose-carrier hits `gt_poses` directly (the R1 stored-ξ sidecar already reaches d_pose
0.00161), so the residual for a seg-free f0 carrier is far below these naive-degradation numbers.

---

## MEASUREMENT 2 — chroma-HF pose-null: TRUE at the yuv6 plane, does NOT transfer to a camera-res carrier (RANK-2: REFINED)

### 2a. Derivation VERIFIED at the yuv6/384 plane (op-level, exact)
A **luma-null chroma checkerboard at 384-Nyquist** (RGB direction `(1.402, −1.060, 1.772)·(±1)`, luma
component = 0 by construction), fed through `rgb_to_yuv6`:
- RGB mean|Δ| = **11.29** (SegNet WOULD see this texture), but
- yuv6 **chroma** (U_sub,V_sub 2×2 box-avg) mean|Δ| = **3.36e-6**, yuv6 **luma** (y00..y11) mean|Δ| =
  **5.20e-6** → both **≈ 0**.
- **VERDICT (MEASURED): the memo's "chroma above the 2×2 grid is exactly pose-null" is CONFIRMED — but
  it is a property of the yuv6/384 plane.** A grid-aligned, luma-null chroma texture is exactly pose-null
  AND seg-visible *there*.

### 2b. The null does NOT transfer to a naive camera-res chroma dither (CORRECTION)
A witness renders at camera res 874×1164; the scorer bilinear-downsamples 2.28× to 384 **before**
`rgb_to_yuv6`. Injecting a luma-preserving chroma cosine at camera res and pushing it through that
downsample:
- period-2 camera chroma injection → at the yuv6 plane, **luma mean|Δ| = 5.35, chroma mean|Δ| = 4.98
  → luma/chroma = 1.075** (a "chroma" injection that is **50% luma** at the scored plane). The clean
  control (luma injection) stays luma/chroma = **10.8**.
- consequently the camera-res chroma-HF sweep is **NOT pose-free**: Δd_pose (both-frame inject, amp 24,
  n48) = chroma **3.80e-2** vs luma **5.67e-2** at period 2 → only **1.49×** cheaper (not 11×), and
  **tied** at period 8 (chroma 5.74e-3 ≈ luma 5.73e-3). Low-freq is the only regime where chroma clearly
  wins (period 32: chroma Δd_pose 2.7e-2 vs luma **3.41** — luma LF wrecks pose).
- **MECHANISM (DERIVED):** the 2.28× bilinear downsample is a low-pass; a 384-Nyquist checkerboard needs
  camera content at ~2.28× higher freq, which bilinear averages toward zero AND folds camera chroma-HF
  into luma. So the exact-null lever is **inaccessible to a naive camera-res chroma dither**.

### 2c. chroma-HF IS a seg actuator (both planes) — and the IDEAL 384-plane lever is a pose-free seg control channel
- camera-res chroma-HF on frame_1 (n48) moves d_seg: Δd_seg = **3.86e-3** (period 2) down to 1.71e-3
  (period 8) — order of the whole mod32cap baseline d_seg (0.00337). So chroma texture is a genuine seg
  actuator (matches the palette finding: seg is texture-dominated).
- **the IDEAL lever** (luma-null chroma checkerboard at the 384 plane, exactly pose-null by 2a),
  amp-swept through SegNet on n96, **Δd_seg authority**: amp4 → 2.77e-4, amp8 → 6.80e-4, amp16 →
  1.64e-3, amp24 → 2.49e-3, amp32 → **2.73e-3** (saturating). **A monotone, pose-free seg control channel
  with authority ~2.7e-3 argmax-fraction (order of the whole mod32cap d_seg) at ZERO pose cost.**
- **DESIGN IMPLICATION (the corrected RANK-2 lever):** a chroma seg-texture carrier is worthwhile **only
  if band-designed at the scorer's 384 grid** — i.e., the witness must render chroma texture so that,
  after the known 2.28× no-AA bilinear kernel, it lands as a 384-grid-aligned luma-null pattern. A naive
  camera-res chroma dither pays ~luma pose cost and gets no free lunch.

---

## P12 — composition sign with the frame_0 pose carrier: ORTHOGONAL (measured via the two exact-null legs)
- Lever A = frame_0 pose-carrier (any f0 degradation): touches **only** the pose term (d_seg = 0 exact,
  §1a n600 8.5e-9).
- Lever B = 384-plane luma-null chroma seg-texture on frame_1: touches **only** the seg term (d_pose = 0
  exact, §2a op-level chroma Δ 3.4e-6).
- The two DOFs live on **disjoint scorer terms** (A: frame_0 → pose; B: frame_1 chroma-null-band → seg),
  so d_pose(A+B) = d_pose(A) and d_seg(A+B) = d_seg(B) **by construction**. **Composition sign =
  ORTHOGONAL (measured=True via the two exact-null legs; no A/B protocol needed).** They stack additively
  for v8: a seg-free luma-only-or-coarser frame_0 pose carrier + a 384-band-designed chroma seg-texture
  on frame_1.

---

## VERDICT (verdict_scope: FORMULATION — scorer-DOF pricing; NO kill)
- **RANK-1 frame_0 seg-freedom: CONFIRMED + PRICED.** Structurally zero d_seg (n600 8.5e-9); luma-only
  f0 is the efficient operating point (−67% bytes, √10·pose 0.180, seg 0 exact). Directly halves v8's
  seg-carrier obligation on the frame axis. **Feeds the v8 frame_0-pose-only carrier + the seed/condition
  program.**
- **RANK-2 chroma-HF: REFINED (partial).** The exact pose-null seg-texture lever is REAL at the yuv6/384
  plane (authority ~2.7e-3 d_seg at zero pose cost) but a **naive camera-res chroma dither cannot access
  it** (50% luma leak through the 2.28× downsample). v8's chroma seg-carrier must **band-design at the
  384 grid**. **Feeds the v8 chroma carrier design constraint.**
- Neither is a score — the pointer moves only through a byte-closed n600 exact eval. Every number here is
  MEASURED on the exact scored path or DERIVED-from-code. **Pointer 0.19110 UNMOVED (means).**

## Triality legs
- **DAG:** FEED-unitC (this landing).
- **Equations:** two EmpiricalAnchors appended to `posenet_luma_chroma_sensitivity_asymmetry_v1`
  (`frame0_segfree_n600_20260710` residual 8.48e-9 vs predicted 0; `chroma_hf_posenull_transfer_384only_20260710`
  camera-res transfer residual 3.80e-2 vs derived-0 + ideal-lever seg authority 2.73e-3). VERIFIED_VIA_EMPIRICAL_ANCHOR.
- **Verdicts/DSL:** verdict JSON emitted via `tac.verdicts` (rows: frame0 n600 d_seg, luma-only f0 √10
  pose, ideal-lever d_seg authority; Composition ORTHOGONAL). No trainer knob yet — routes #385 build-wave
  (frame_0-pose-only carrier + 384-band-designed chroma seg-texture).

**Pointer 0.19110 UNMOVED (means).**
