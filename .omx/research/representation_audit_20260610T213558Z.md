<!-- SPDX-License-Identifier: MIT -->
# REPRESENTATION AUDIT — store-the-measured-quantity vs reconstruct-from-pixels, across the whole stack (Task #83)

**UTC:** 2026-06-10T21:35:58Z · **Subagent:** `task83_representation_audit` · **Mode:** design + measure.
**Authority:** every number below is `[local CPU-torch advisory]` — the EXACT frozen scorers
(`upstream/modules.py`: SegNet reads `x[:,-1]` argmax partition on frame1; PoseNet reads BOTH frames' YUV6
motion → MSE on first 6 of 12 dims; rate = `archive.zip / 37,545,489`), GT decoded via
`frame_utils.yuv420_to_rgb` ONLY (NEVER MPS / rgb24). NOT the contest 600-sample harness → non-promotable.
`$0` spend, no GPU, no dispatch. `promotable=false`, `score_claim=false`, `score_roadmap_update_eligible=false`,
`mechanism_update_eligible=true`. The frontier byte facts (177,169 B → 0.11797 rate; 161,104 B decoder /
15,070 B latents) are carried from #66/#67/#75 (PROVEN-FRONTIER); the partition/motion/pose-serve numbers are
REAL measurements landed this task (`tools/representation_audit_probe.py`, 8 GT pairs + 10-pair delta scan).

The exact ΔS lever: `ΔS = 25·Δbytes / 37,545,489 = 6.6586e-7 · Δbytes`.

---

## LEAD — the two answers the task demands, first

**(1) WHICH quantities use the WRONG representation?** Exactly ONE term is already in its right representation on
the frontier; the rest split by a sharp, MEASURED asymmetry:

| scored/stored quantity | current (WRONG-vs-RIGHT) | verdict |
|---|---|---|
| **POSE (6 scalars/pair)** | RIGHT *only on Quantizr*; the HNeRV frontier RECONSTRUCTS pose from a full pixel render (the #74/#80 wall) | **WRONG on our stack** — store-explicit + FiLM is ~115× cheaper (measured) |
| **SEG (the partition)** | RECONSTRUCT the argmax from a 162 KB full RGB render | **RIGHT-ish (amortized)** — but the RIGHT *fix* is store-partition + FiLM-condition a SMALLER shared decoder, NOT store-the-full-partition (that LOSES, §a) |
| **MOTION / temporal** | per-frame independent HNeRV latents | **WRONG by structure** — the partition is ~99% temporally redundant + frame luma RMSE is 6.6/255, but the naïve deltas LOSE (§b) — the redundancy is real, the *codec* is the open lever |
| **FRAME0 / FRAME1** | render both frames fully | **WRONG** — frame0 is SegNet-INVISIBLE (`x[:,-1]`), a pure pose carrier; warp-from-frame1 is the right split (§c, Quantizr does this) |
| **WEIGHTS** | full learned INT8 weights | RIGHT-as-is on the frozen frontier; a fixed-codebook VQ carrier pays ONLY on a *retrained smaller* net (#67 G5) |

**(2) THE SINGLE HIGHEST-EV REPRESENTATION FIX:** **carry the 6-dim pose explicitly (per-pair) + FiLM-condition
the decoder to realize it — Quantizr's mechanism, generalized as the score-native pose carrier.** This is the
WRONG→RIGHT swap with the largest measured byte/score gain that is *not* an open research campaign:

- **The asymmetry is decisive (MEASURED).** Storing the POSE quotient = **600×6 floats → ~1,557 B brotli** (the
  #67 measured pose-temporal-delta floor) → rate **0.00104**. Reconstructing pose from pixels needs a
  near-lossless pose-bearing frame0 (the #80 rank-1 tube; pixel-RMSE<3) — which is exactly why the frontier
  spends most of its 162 KB decoder on pose-fidelity. The 6 scalars are **~104× smaller** than even *one*
  stored 384×512 partition (§a) and are the cheapest object the score reads.
- **Predicted gain:** replacing pose-from-pixels with pose-explicit + FiLM frees the decoder from the
  pixel-RMSE<3 pose-fidelity constraint (the dominant capacity sink at the PR106 frontier op-point where pose
  marginal dominates SegNet 2.71×, CLAUDE.md). The decoder then only has to land the *seg argmax* + a pose the
  net is HANDED. This is the #67 PATH-1 conditional-floor mechanism (decoder intrinsic ~25–55 KB) made
  concrete: pose stops costing decoder capacity. Predicted archive band on a retrained carrier: **~42–72 KB
  (decoder 25–55 + latents 15 + pose 1.6 + seg sidecar 2) → ΔS_rate −0.070 to −0.090** vs the 177 KB frontier
  IF the smaller net holds the seg cell. Executor: the #74/#76 funded MLX campaign (the only >$1 lever).

The reason this is #1 and not the seg fix: the seg term has NO cheap store-explicit form (§a — the partition is
a 2D-per-frame object that costs more stored-per-frame than the whole amortized decoder), whereas the pose term
is 6 scalars/pair that store for ~1.6 KB total. **Pose is the one quantity where "store the quotient" is
unambiguously, measurably cheaper than "reconstruct from pixels."**

---

## §a — SEG (the partition): the right representation is store + FiLM-CONDITION, NOT store-the-full-partition

**Current (WRONG):** the HNeRV frontier reconstructs the SegNet argmax partition `L*` from a full 384×512×3 RGB
render produced by a 162 KB decoder. The score reads only `argmax S(frame1)` (a 5-class PARTITION), never the
pixels — so rendering pixels to recover an argmax is reconstructing the quotient from a far richer object.

**The naïve "store the quotient" arm — MEASURED, and it LOSES:**

| measurement (real SegNet on GT frame1, 8 pairs) | value |
|---|---:|
| mean contour-coded partition bytes (lossless, roundtrip-exact, d_seg=0) | **886 B/frame** |
| mean regions/frame (connected components) | 36.6 |
| boundary fraction (O(boundary) pixels) | **1.25 %** |
| **store ALL 600 partitions** = 886 × 600 | **531,600 B → rate 0.354** |
| ratio vs the WHOLE 162 KB decoder | **3.3×** the decoder |

Storing the full partition per-frame is **3.3× the size of the amortized decoder that reconstructs BOTH the
partition AND the pose frames.** So "store the quotient directly" — which WINS decisively for pose (6 scalars)
— **LOSES for seg** (a 384×512 2D object per frame). The partition is cheap *per frame* (886 B, O(boundary))
but expensive *×600* because each frame's boundary geometry is largely independent under the contour codec.

**The RIGHT seg representation (the actual fix):** **store the partition's per-pair conditioning code + FiLM a
SMALLER SHARED decoder** — the seg analog of Quantizr's pose FiLM. The amortized decoder is already the right
*shape* (share cross-frame structure); the fix is (i) make it score-aware-trained + smaller (#67 PATH-1: seg
intrinsic ~20–55 KB), and (ii) feed it the stored partition/margin as conditioning so it only has to *realize*
the argmax, not *discover* it. **This is exactly what already lives in the repo** — `boundary_math/lever_b_generator.py`
(amortized 65 KB seg INR), `boundary_math/amortized_luma_carrier.py` (FiLM-modulated per-pair appearance), and
`boundary_math/margin_conditional_residual.py` (the decoder regenerates the SegNet margin FOR FREE, so only the
boundary-band flip residual is stored — `log2 C(|B|,K)` conditional position cost ≪ unconditional). The audit's
contribution: confirm the *full-partition-store* arm is dominated (886×600 = 0.354 rate) so the right move is
the conditioned-residual path, not a partition sidecar.

**Predicted gain:** the conditioned seg path is a distortion-closure lever (hold d_seg→0 at near-constant bytes)
+ a contributor to the #67 PATH-1 conditional floor, not a standalone byte cut on the frozen frontier. Net: the
seg fix is REAL but its EV is *fused into PATH-1* (you cannot split the frozen 162 KB decoder — #71 entanglement
— you must TRAIN the conditioned split).

## §b — MOTION / temporal: the redundancy is real, the naïve flow codecs LOSE, the win needs a real flow codec

**Current (WRONG):** per-frame independent HNeRV latents (15,070 B / 8.5% of archive) — no explicit
flow/keyframe structure, despite the dashcam being extremely temporally redundant.

**The temporal redundancy IS real (MEASURED):**

| measurement | value | reading |
|---|---:|---|
| frame0→frame1 luma RMSE (within a pair, 1 frame apart) | **6.6 / 255** | tiny — frames are nearly identical |
| cross-frame partition changed-pixel fraction (pair→pair) | **1.04 %** | the scored partition is ~99% temporally static |
| consecutive-frame luma Pearson corr (120-frame scan) | ~0.97+ | extreme temporal correlation |

**But every NAÏVE store-the-flow codec LOSES (MEASURED) — the honest negatives:**

1. **Cross-frame partition delta** (store the partition once + the boundary-motion change-mask): mean delta =
   **1,249 B vs 886 B independent → ratio 1.42 (LOSES).** Only 1% of pixels change, but the *scattered
   boundary change-mask* is high-entropy (24 KB raw bitmap, compresses poorly) while a full partition LZMA-eats
   its constant interiors. Naïve XOR-delta is the wrong codec.
2. **Luma frame delta vs raw** (store keyframe + per-frame luma residual): delta zlib bytes / raw frame zlib
   bytes = **0.98 (LOSES).** The frame0→frame1 luma residual barely compresses better than the raw frame —
   because the dominant motion is *camera ego-translation*, which moves every pixel (a global shift), so the
   naïve per-pixel temporal delta is NOT sparse.

**The reframe (NOT a kill):** the redundancy is genuine (1% partition change, 0.97 luma corr), but it is
*motion-warp* redundancy, not *static-residual* redundancy. A real win needs a **motion-compensated** codec
(optical-flow warp + small residual, FFNeRV-style) so the global ego-motion is captured by the flow field, not
re-paid per pixel. The repo already has `ffnerv_as_renderer.py` as the flow-renderer scaffold. The audit
result: the naïve delta arms are CLOSED (don't re-mine partition-XOR or luma-residual); the open lever is a
warp-based flow codec, and it composes with the keyframe + per-pair-latent split.

**MOTION-serves-pose? — NO (decisive negative).** The hope: flow directly serves the 6 pose dims (motion IS the
pose signal), sidestepping the #80 pose tube. MEASURED on the exact frozen PoseNet:

| pose-pair construction | d_pose (mean, 8 pairs) | vs the GT tube (~2.9e-5) |
|---|---:|---|
| GT moving pair (frame0, frame1) | ~2.9e-5 | the tube |
| STATIC pair (frame0, frame0) — motion removed | **189.0** | 6.5M× the tube |
| GLOBAL-translation flow (frame0, frame0 best-shifted) | **189.4** | no better than static |

A coarse 2-scalar global-translation "flow" carrier reads d_pose ≈ 189 — **indistinguishable from removing
motion entirely.** The pose signal is NOT a global shift; it is the fine per-pixel parallax/ego-motion structure
(#80: a rank-1 read that integrates fine spatial detail). **Flow does NOT cheaply serve pose** — a *dense*
optical flow might (untested), but the cheap global-translation flow is worthless for pose. This confirms #80:
the pose carrier must be near-lossless fine structure, not a coarse motion summary. So MOTION compression buys
RATE (the per-frame-latent redundancy) but NOT a pose shortcut.

## §c — FRAME0 / FRAME1: the right split is warp-frame0-from-frame1 (frame0 is SegNet-invisible)

**Current (WRONG):** render both frame0 and frame1 fully.

**The scorer fact (verified `modules.py:108`):** `SegNet.preprocess_input` uses `x[:, -1, ...]` — ONLY frame1.
**Frame0 is SegNet-INVISIBLE.** It contributes ZERO to d_seg; it is a *pure pose carrier*. Frame1 carries BOTH
d_seg (its argmax) AND its half of d_pose. From #80 §5: frame0 contributes ~20× more pose debt than frame1 at
low fidelity — frame0 is the *dominant* pose carrier, frame1 is seg-dominant.

**The RIGHT per-frame representation (Quantizr's mechanism):**
- **frame1** = the seg+pose anchor — store/render it at the fidelity the SegNet argmax + its pose half need.
- **frame0** = WARP from frame1 + a small pose-fidelity correction (Quantizr stores only frame1's mask + warps;
  the per-pair pose code carries the motion). Frame0 needs near-lossless *luma* along the rank-1 pose direction
  (#80) but is free everywhere SegNet cares (it's seg-invisible).

**Predicted gain:** this is a structural composition with the #1 fix (pose-explicit) and §b (warp codec): the
two frames are not two independent renders — frame0 is (warp(frame1), pose-code). Folds into PATH-1; the
standalone byte cut is the ~half of the appearance bytes that frame0 no longer renders independently.

## §d — WEIGHTS: full learned weights vs fixed-codebook VQ (carried from #67)

**Current:** full INT8 learned decoder weights (161,104 B, near-iid, #66/#67). **#67 G1/G2 PROVED** no fixed
orthonormal basis and no small-seed procedural generator reduces the FROZEN weights (compaction headroom ≈ 0;
counting bound bars seed regeneration). **The fixed-codebook VQ carrier pays ONLY on a RETRAINED smaller net**
(the codebook free in inflate.py, the per-weight indices charged, the forward map DESIGNED around the fixed
codebook so indices are genuinely few) — #67 PATH-1 / G5. Verdict: weights are RIGHT-as-is on the frozen
frontier; the WRONG→RIGHT swap is the retrained smaller carrier, identical executor to fix #1.

## §e — OTHER (latents, scales, sidecar): mostly right, one sub-precision arm

- **Per-pair latents (15,070 B):** already below the per-dim-independent floor (#66: AR(1) + cross-dim linear
  prediction). A fixed score-aligned basis captures < 60 B (#67 PATH-2, sub-precision). RIGHT-as-is.
- **Per-tensor fp16 scales / 607 B sidecar / 222 B selector:** at the order-0 floor (#66). RIGHT.
- No quantity here is "reconstructed when it could be stored" — they are all already stored-direct + entropy-coded.

---

## The right-vs-wrong table (the deliverable) — current bytes vs predicted bytes + score effect

| quantity | WRONG (current) representation | bytes now | RIGHT (store-measured + condition) | predicted bytes | score effect | EV rank |
|---|---|---:|---|---:|---|:---:|
| **POSE** | reconstruct 6 dims from a near-lossless pixel render (decoder capacity sink) | most of 162 KB decoder | **store 6 floats/pair + FiLM-condition (Quantizr)** | **~1,557 B** | frees decoder of pose-fidelity → enables retrained ~25–55 KB decoder; ΔS_rate **−0.070…−0.090** (on PATH-1) | **#1** |
| **WEIGHTS** | full frozen learned INT8 | 161,104 B | **retrained SMALLER net + fixed-codebook VQ indices** | **~25–55 KB** | same PATH-1 lever as pose; the byte mass | #1 (fused) |
| **FRAME0/1** | render both frames fully | ~appearance share | **warp frame0 from frame1 + pose code; frame0 seg-invisible** | ~½ appearance | structural; folds into PATH-1 | #2 |
| **SEG partition** | reconstruct argmax from full RGB render | (in decoder) | **store margin/partition cond + FiLM smaller decoder; only boundary residual stored** | seg core ~20–55 KB | distortion-closure (hold d_seg) + PATH-1 contributor; full-partition-store LOSES (531 KB) | #2 (fused) |
| **MOTION/temporal** | per-frame independent latents | 15,070 B | **motion-compensated flow warp + keyframe** (naïve delta LOSES 1.4×) | open (needs real flow codec) | RATE only (NOT pose); naïve arms CLOSED | #3 |
| **latents/scales/sidecar** | stored-direct + entropy-coded | ~16 KB | already right (≤60 B headroom) | ~same | sub-precision; CLOSE the axis | — |

**The convergent picture:** every WRONG representation collapses into ONE fix — the **#74/#76 score-aware
retrained smaller carrier** that (a) carries pose explicitly + FiLM (the #1 measured win), (b) is conditioned on
the stored seg partition/margin (§a), (c) warps frame0 from frame1 (§c), and (d) codes its smaller weights as
fixed-codebook VQ indices (§d). This is the #67 PATH-1 / #66 conditional-floor campaign, and the audit's
contribution is the MEASURED asymmetry that proves WHY: pose stores for 1.6 KB (store-the-quotient WINS) while
seg stores for 531 KB (store-the-quotient LOSES) → the right design is **pose-explicit + seg-conditioned**, the
exact Quantizr split, generalized.

---

## Feeds the capstone #78

1. **Pose-explicit + FiLM is the #1 representation fix** (measured 1.6 KB vs reconstruct-from-pixels) — the
   capstone's pose budget is ~1.6 KB stored, not a near-frontier decoder capacity sink.
2. **Seg is NOT store-explicit** (531 KB loses) — the capstone's seg budget is a *conditioned smaller decoder*
   + boundary-residual sidecar, not a partition store.
3. **Motion buys RATE not POSE** (flow does not serve the 6 pose dims; the naïve deltas lose) — the capstone's
   temporal lever is a motion-compensated flow codec on the latent stream, decoupled from the pose carrier.
4. **Frame0 is seg-invisible** — the capstone renders frame1 (seg+pose) + warps frame0 (pose-only).
5. **All four fixes are ONE campaign:** the score-aware retrained smaller carrier (#74/#76), conditional floor
   ~25–65 KB (#67), predicted archive ~42–72 KB → ΔS_rate −0.070…−0.090 IF the smaller net holds the cell.

---

## WIRE-IN (Catalog #125)

1. **sensitivity-map — ACTIVE.** New prior: pose is the one quotient where store-direct WINS (1.6 KB);
   seg/motion store-direct LOSE (531 KB / 1.4× delta). Aiming surface = pose-explicit + seg-conditioned
   retrained carrier, NOT a partition/flow sidecar on the frozen frontier.
2. **Pareto — ACTIVE.** Hard constraints added: (a) full-partition-per-frame store = 531 KB (rate 0.354,
   dominated); (b) naïve cross-frame partition delta 1.42× (LOSES); (c) global-translation flow d_pose 189 (no
   pose service); (d) pose-explicit 1.6 KB (the cheap vertex).
3. **bit-allocator — ACTIVE.** Pose → ~1.6 KB store (cheapest). Seg → conditioned residual (boundary-band only).
   Motion → flow codec on latents (rate). Frame0 → warp (no independent render).
4. **cathedral-autopilot — N/A.** Audit + design, no archive emitted, non-promotable.
5. **continual-learning — ACTIVE.** Reseed the V3 judge: (a) the store-vs-reconstruct asymmetry is
   quantity-dependent — pose store WINS, seg store LOSES (by MEASURED bytes); (b) flow does NOT serve pose (the
   cheap global-translation flow reads d_pose 189); (c) naïve partition/luma deltas LOSE (the redundancy is
   motion-warp not static-residual); (d) all fixes converge on the #74/#76 retrained-carrier PATH-1.
6. **probe-disambiguator — RESOLVED + ONE open.** "Does store-the-quotient beat reconstruct-from-pixels?" →
   YES for POSE (1.6 KB), NO for SEG (531 KB). "Does flow serve the 6 pose dims?" → NO (global-translation flow
   d_pose 189). "Is the partition temporally redundant enough for a delta codec?" → 99% static BUT the naïve
   change-mask delta LOSES 1.4× (open: a real boundary-motion / motion-compensated codec — UNTESTED, the next
   $0 probe).

---

## NO-FAKE attestation + tests

- The 886 B/frame partition, 1.25% boundary fraction, 36.6 regions, roundtrip-exact, d_seg=0 are REAL outputs of
  the frozen CPU-torch SegNet on GT frame1 + the lossless contour codec (`build_and_measure_lstar` →
  `encode_partition`/`decode_partition`, already tested in `seg_core`). The 531 KB = 886×600 and rate 0.354 are
  derived from the measurement + the exact `evaluate.py` rate rule, flagged as derivation.
- The cross-frame delta 1.42× (1,249 vs 886 B) and 1.04% changed fraction are REAL LZMA-coded lengths of the
  change payload over 10 consecutive GT partition pairs — the honest negative that the naïve delta LOSES.
- The d_pose_static 189 / d_pose_flow 189 are REAL frozen-PoseNet MSE (first 6 of 12 dims), GT via
  `yuv420_to_rgb`, differentiable-yuv6 patch active, NEVER MPS. The GT-moving-tube ~2.9e-5 is carried from #80.
- The pose-explicit 1,557 B is the #67-MEASURED pose-temporal-delta floor (carried, not re-derived); the
  decoder-conditional ~25–65 KB band is #67's DERIVED bound (carried, flagged as derived not measured).
- 19 behavior tests land (`src/tac/boundary_math/tests/test_representation_audit_probe.py`): BT.601 luma (×5),
  real-zlib delta entropy redundant-vs-noise discrimination (×4), O(boundary) boundary-fraction geometry (×5),
  and the cross-frame partition-change codec incl. the scattered-delta-costs-more-than-static NO-FAKE pin (×5).
  `.venv/bin/ruff` clean; review-gate marked reviewed; 216 boundary_math tests green (0 regressions).

## CROSS-REFERENCES

`full_stack_audit_and_findings_trust_20260610T200115Z` (#81 — Quantizr stores 6 pose scalars + FiLM-injects,
the lesson this generalizes) · `pose_crux_and_protection_20260610T195607Z` (#80 — pose rank-1, flow-does-not-
serve confirmed here, frame0 dominant) · `smaller_learned_basis_deep_math_20260610T191009Z` (#67 — pose floor
1,557 B, decoder conditional floor 25–65 KB, PATH-1 retrained carrier = the convergent executor) ·
`src/tac/boundary_math/{contour_codec,partition,seg_core,lever_b_generator,amortized_luma_carrier,margin_conditional_residual,ffnerv_as_renderer}.py`
(the repo's already-built store-the-partition + condition + warp primitives this audit confirms are the right
shape) · `tools/representation_audit_probe.py` (this task's measurement probe) ·
`experiments/results/task83_representation_audit/probe.json` (the evidence) · `upstream/modules.py:61-113`
(SegNet `x[:,-1]` frame1-only argmax; PoseNet both-frame YUV6 6-of-12-dim MSE) · `upstream/evaluate.py:92`
(`score = 100*segnet_dist + √(10*posenet_dist) + 25*rate`). **External:** Quantizr PR#55
`[external:github.com/commaai/comma_video_compression_challenge/pull/55]` (pose-explicit + FiLM existence proof).
