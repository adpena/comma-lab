# v10 factor-2 lattice solve — RATE verdict + the composition law (2026-07-19)

Lane `lane_v10_uint8_lattice_20260718` continuation · `research_only=true` ·
`[macOS-CPU advisory n6]` · pointer `0.1910828242 [contest-CPU]` **UNMOVED**.

## What was already true (2026-07-18 receipt, re-read today)

`v10_uint8_lattice_feasibility_receipt_20260718.json`: **exact_uint8_lattice_candidate
d_seg = 0.0** on n6 (all 5 classes zero), 3,538,944/3,538,944 affine blocks
`FEASIBLE_EXACT`, 6/6 frames decoded with exact numerator equality, both confound
controls PASS, 83 s runtime. Verdict scope: frame1/A factor only, no PoseNet, no
receiver, no contest axis.

## NEW MEASURED (2026-07-19): the direct-frame rate is DEAD

brotli-Q11 on the n6 exact-solved uint8 frames (874×1164×3) vs the source frames
(same coder), per pair:

| pair | brotli(solved) | brotli(src) | ratio | delta-vs-src nz | brotli(delta) |
|---|---|---|---|---|---|
| 90  | 1,764,822 | 1,505,570 | 1.17 | 0.760 | 1,697,913 |
| 175 | 1,678,750 | 1,523,429 | 1.10 | 0.664 | 1,549,012 |
| 277 | 1,674,300 | 1,670,817 | 1.00 | 0.645 | 1,572,385 |
| 381 | 1,678,709 | 1,528,439 | 1.10 | 0.677 | 1,551,020 |
| 424 | 1,704,530 | 1,530,889 | 1.11 | 0.695 | 1,590,502 |
| 573 | 1,717,645 | 1,534,219 | 1.12 | 0.707 | 1,608,619 |

Mean 1.70 MB/frame; n600 naive ≈ 1.02 GB ⇒ rate term ≈ 680 (vs whole-frontier
0.118). **verdict_scope: direct-solved-frame-as-payload FORMULATION only** — the
solver's arbitrary/min-norm point in the feasible set is entropy-dense in ker(A);
storing the frame is dominated. The feasibility PARADIGM is untouched (d_seg=0.0
stands).

## The composition law (DERIVED — why the solve still changes everything)

The solve's target is the **scorer-input plane y (384×512×3)** — it inverts the
shared resize A + uint8 realization EXACTLY (`A(frame) = y` to integer-numerator
equality; frozen factorization: `A_seg ≡ A_pose`). Therefore:

1. **Payload = describe ŷ, not the frame.** Receiver: stored ŷ-description →
   free generator/decoder expands ŷ → lattice solve (deterministic code, rule-118
   FREE) → uint8 camera frame with `A(frame) = ŷ` exactly → the frozen scorer
   sees exactly ŷ.
2. **All realization error goes to ZERO by construction**: uint8-STE, R-roundtrip
   survival, AA washout (the measured lever-3 wall) are eliminated — the render
   IS the scorer input, exactly.
3. **The whole remaining game is ONE clean R-D object:**
   `minimize bytes(ŷ-desc) + 100·d_seg(argmax SegNet(ŷ), L*) [+ pose via frame0/6-scalars]`
   at scorer res — no conv inversion, no uint8 confound downstream. This is the
   R-D curve the blocked #536 KKT waterfill needs.
4. **The witness IS a ŷ-generator** (it renders at 384×512). Witness ⊕ lattice
   solve = the witness's render-grid output realized EXACTLY through the scorer.
   The c2-line survival losses become free headroom (magnitude owed: render-grid
   argmax vs through-R argmax on the c2 EMA — a $0 measure).

Feasibility caveat (honest): an arbitrary compressed ŷ need not be in the exact
lattice image; the solve then lands on the NEAREST feasible plane (integer-repair
loop, hard-oracle-gated). The repair gap is a measurement owed at each R-D point,
not an assumption.

## In-flight (2026-07-19)

- **n600 lattice replay RUNNING** (pid 60821, resumable state
  `/Volumes/VertigoDataTier/pact/evidence/v10_uint8_lattice_n600_20260719/`),
  blocker 1 of 5 from the receipt. ETA ~2.3 h.
- Yousfi c2 reducibility n600 render RUNNING (`.omx/research/yousfi_c2_reducibility_n600_20260719/`).
- Next unblocks: #543 receiver-arithmetic DECLARATION (frame-195 tie pixel:
  CPU-Torch f32 IS the contest arithmetic — declare it, rerun) · ŷ-description
  R-D curve rows (feeds #536) · PoseNet/frame0 interaction (receipt blocker 2).

Consumers: #536 (KKT), #543 (receiver), SPEC_v10 §3/§4 (seeds/rate), completeness
matrix factor 2 row. Triality: DAG FEED owed with the n600 result (one FEED, not
two).

## 2026-07-19 ADDENDUM — POSE SOLVED BY THE SAME MECHANISM (MEASURED, n6, real DistortionNet)

**Reading (upstream/modules.py:71-75):** `PoseNet.preprocess_input` = rearrange → **bilinear resize to
(512,384) FIRST** (the same A as SegNet) → `rgb_to_yuv6` AFTER. PoseNet's input is a pure deterministic
function of `A(camera_frame)` — the exact plane the lattice solve pins to numerator equality. The YUV6
clamps therefore act on identical inputs and cannot break the equality.

**MEASURED (full upstream `DistortionNet.compute_distortion`, CPU-torch, pairs (gt_f0, solved_f1) vs
(gt_f0, gt_f1), n6 pairs 90/175/277/381/424/573):**

| pair | d_pose | d_seg |
|---|---|---|
| 90 | 8.78e-10 | 0.0 |
| 175 | 5.46e-10 | 0.0 |
| 277 | 1.01e-11 | 0.0 |
| 381 | 1.19e-10 | 0.0 |
| 424 | 3.69e-09 | 0.0 |
| 573 | 3.50e-10 | 0.0 |

**mean d_pose = 9.3e-10** (34,000× below the 3.2e-5 need; contribution √(10·d_pose)=0.0001 vs R1's
0.127). Residual = fp32 last-bit accumulation only. **Receipt blocker 2 (PoseNet/both-frame interaction)
ANSWERED for the frame1-substitution case: ONE exact realization buys BOTH scorers simultaneously.**
Frame0 substitution (when frame0 is also generated) inherits the same argument — same A, same solve —
but is a separate measurement owed when a compact ŷ₀ exists.

**Consequence for the waterfill (#536):** seg and pose COLLAPSE onto one axis — fidelity of the shared
scorer-input plane ŷ. The 3-axis problem is now bytes(ŷ-desc) vs (d_seg(ŷ), d_pose(ŷ)) with BOTH
distortions zero when ŷ = y exactly, and BOTH degrading together as ŷ compresses. Everything is
reachable; the only question is the rate price (opportunity-pools law, proof instance now includes pose).

**n600 scale-up note:** chunk-00 (pairs 0-11) shows 1 mismatched pixel (d_seg 4.2e-7) with ALL blocks
exact — the fp32 tie-pixel class (same class as #543 frame-195). This is the honest noise floor of
exact-through-fp32; record per-chunk, never round away.
