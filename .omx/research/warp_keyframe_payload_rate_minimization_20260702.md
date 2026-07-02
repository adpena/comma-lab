# Warp-real-luma keyframe payload — rate minimization (research/design, #202 byte-close track)

**Date:** 2026-07-02 · **Track:** RATE MINIMIZATION (operator 2026-07-02, background/parallel, NON-BLOCKING).
**Authority:** `[macOS-CPU advisory / CPU-torch research-signal]` ONLY. NOT a contest score. Canonical
frontier pointer **0.19110 UNMOVED**. `score_claim=false, promotable=false`. Every d_pose number is the
FROZEN CPU-torch PoseNet authority (NEVER MPS); every byte number is a real codec on real `gt_f0` frames.
**Means/ends:** this is a MEANS (a #202 byte-close design); the END is a byte-closed `upstream/evaluate.py`
n600 exact row that beats 0.19110. This track does NOT touch the #205 launch/trainer/config.

---

## 0. TL;DR — the honest correction, sharpened

The Phase-3 verdict (`bcfd85b4c` §4) flagged that §7's rate budget OMITS the warp-real-luma pose-carrier
keyframe payload, and used **"13 keyframes ≈ rate-term 0.0060"**. **That 0.0060 is the PARTITION keyframe
cost (a 5-class argmax LABEL map, ~693 B/keyframe compressed) — it was borrowed for the REAL-LUMA pose
carrier, which stores NATURAL IMAGES (`gt_f0`, native 874×1164×3) with 10–40× more entropy.** This ledger
MEASURES the real payload and ranks how to minimize it.

**Measured headline (both axes real, pessimism caveats in §4):**
- A real-luma keyframe that **preserves d_pose** (fixed-ξ / sharp-partner proxy) costs **rate ≈ 0.03–0.07
  for 13 keyframes** (HEVC-video, 384×512), i.e. **5–12× the §7 0.0060 estimate**, and **0.09–0.22 for 40
  keyframes** (if full-clip reach is shorter than the optimistic 47-pair partition reach).
- The payload is dominated by **keyframe RESOLUTION** (≈40× range: native 0.13–2.9 → quarter 0.003), then
  **codec/quality** (≈2–4×). **Temporal (HEVC-video) coding is the single strongest measured codec lever:
  ≈4× cheaper than per-frame WebP at matched d_pose**, because the 13 keyframes are correlated driving frames.
- **Two uncertainties, BOTH pointing CHEAPER, both needing the trained witness (GPU) to resolve** — the
  pivotal one is **residual-compensation** (§4.2): the trained 6-DOF per-pair twist residual `dxi` may absorb
  most keyframe-degradation d_pose (my fixed-ξ probe cannot let it re-optimize). If it does, cheap low-res
  keyframes (rate ~0.006–0.015) work and sub-0.15 stays reachable; if it does not, the d_pose-safe keyframe
  (~0.03–0.07) pushes realistic rate to ~0.09–0.12 and sub-0.15 requires the structural levers in §5.

**One-line #202 recommendation:** the FIRST thing to measure once #205 trains a witness with the pose
carrier + `w_pose>0` is **d_pose with DEGRADED keyframes at the trained residual** (§6 step 1) — it decides
cheap-vs-expensive. In parallel, build **ego-warp-predicted residual keyframe coding** (§5 lever R1, the
rule-118 win) and **verify full-clip pose-reach** (§5 lever C1).

---

## 1. What the payload IS (measured baseline; corrects §7)

The carrier (`src/tac/boundary_math/warp_real_luma_frame0.py`) warps a **native 874×1164×3** source luma
frame by the stored ego-twist ξ. At training the source is `gt_f0`; at decode it must be a **stored REAL
keyframe** (the original video is unavailable). Keyframe schedule (from the d_seg partition reach gate
`experiments/results/screw_reach/reach_n96.json`, reach k*=47 pairs): **13 keyframes** at pairs
{0,47,94,…,564} cover the 600-pair clip — IF the 47-pair reach holds across the whole clip (it is measured
only on the near-start n96; turns/traffic break the constant-plane homography → shorter reach → MORE
keyframes; §5 C1).

**The ξ payload is NOT the problem** — the per-pair 6-DOF twist is **2,424 B (low-rank r2) → rate 0.0016**
(dual-use, already counted for d_pose). **The keyframe payload DWARFS it and is comparable to or larger
than the ENTIRE witness archive (rate 0.055).** That is the finding.

**MEASURED per-keyframe cost of a REAL `gt_f0` frame (13 keyframes; rate = 25·B·13/37,545,489):**

| resolution | codec | B/keyframe | rate (13 kf) | note |
|---|---|---:|---:|---|
| native 874×1164 | PNG (lossless) | 1,079,068 | **9.34** | absurd — upper bound |
| native 874×1164 | WebP q75 | 58,593 | **0.507** | "visually fine" native = fatal |
| native 874×1164 | WebP q50 | 15,259 | **0.132** | still fatal |
| 384×512 | WebP q75 | 12,002 | 0.104 | |
| 384×512 | WebP q30 | 3,884 | 0.034 | |
| 192×256 | WebP q30 | 756 | 0.0065 | ≈ the §7 est — but see §4 (d_pose cliff) |
| 96×128 | WebP q30 | 305 | **0.0026** | cheaper than §7 est — but d_pose CLIFF |

Codec ranking at fixed resolution (secondary lever, ≈2–4×): **HEVC-video (inter) < AVIF ≈ WebP < JPEG2000 <
JPEG < PNG**. Luma-only ≈ 1.3–2× cheaper than RGB (relevant: PoseNet is chroma-null, §5 R3). The 0.006
number is only reachable by **dropping resolution to ≤192×256** — which §4 shows has a d_pose cost.

---

## 2. The rate surface — resolution × quality × codec × count (MEASURED)

Real `gt_f0` keyframes, real codecs, 13 true keyframe indices from `gt_n600`. `rate40` = the pessimistic
40-keyframe (short-reach) case.

**Per-frame still codecs (best-of):**

| config | B/kf | rate13 | rate40 |
|---|---:|---:|---:|
| 96×128 WebP q30 | 305 | 0.0026 | 0.0081 |
| 96×128 HEVC-video crf40 | 336 | 0.0029 | 0.0089 |
| 96×128 AVIF q25 | 496 | 0.0043 | 0.0132 |
| 192×256 WebP q30 | 756 | 0.0065 | 0.0201 |
| 384×512 WebP q30 | 3,884 | 0.0336 | 0.1035 |
| 384×512 WebP q95 | 34,003 | 0.294 | 0.906 |

**Temporal (mini-video, single-GOP inter) codecs — the strong lever:**

| config | tot13 B | rate13 | rate40 |
|---|---:|---:|---:|
| 384×512 HEVC crf28 | 100,064 | 0.0666 | 0.205 |
| 384×512 HEVC crf34 | 32,875 | 0.0219 | 0.067 |
| 256×342 HEVC crf28 | 41,779 | 0.0278 | 0.086 |
| 256×342 HEVC crf40 | 9,120 | 0.0061 | 0.019 |

**Takeaway:** resolution is the ≈40× lever; temporal HEVC is ≈4× over per-frame WebP at matched quality
(measured in §3). AV1-video (libaom) would likely beat HEVC further but failed to encode in this harness
(ffmpeg pipe/pix_fmt issue — a tooling gap, not a method result; retry with yuv420p).

---

## 3. The d_pose coupling — the binding floor (MEASURED, PESSIMISTIC proxy)

**Method:** degrade the keyframe (downsample native→R, optional codec roundtrip, upsample back to native),
warp it by a fixed d_pose-calibrated forward twist (s_t=0.044), and measure d_pose through the frozen
CPU-torch PoseNet on n96 (NO-FAKE self-check `PoseNet(gt pair)==gt_poses` enforced). The **abs increment**
over the native-keyframe baseline (d_pose=2.73) is the keyframe-degradation cost.

**(a) Resolution cliff (uncompressed roundtrip):**

| keyframe source | d_pose | abs incr |
|---|---:|---:|
| native (lossless) | 2.73 | 0 |
| 384×512 | 2.63 | −0.09 (neutral) |
| 192×256 | 2.24 | −0.49 (neutral / slight help) |
| **96×128** | **26.88** | **+24.2 (CLIFF)** |
| 48×64 | 101.3 | +98.6 |

**(b) Quality floor (resolution × codec quality → d_pose, with rate):**

| config | rate13 | d_pose | abs incr |
|---|---:|---:|---:|
| 384×512 WebP q95 | 0.294 | 2.95 | +0.23 (safe) |
| 384×512 WebP q90 | 0.161 | 3.20 | +0.47 (safe) |
| 256×342 WebP q90 | 0.071 | 3.24 | +0.52 (borderline) |
| 192×256 WebP q95 | 0.076 | 3.38 | +0.65 (borderline) |
| 192×256 WebP q80 | 0.016 | 11.85 | +9.1 (cliff) |
| 384×512 HEVC crf28 | 0.067 | 3.76 | +1.03 |
| 256×342 HEVC crf28 | 0.028 | 4.66 | +1.93 |

**Findings (MEASURED, on the proxy):** (1) there is a sharp resolution cliff below ~192×256; (2) lossy
compression at low resolution ALSO catastrophically inflates d_pose (codec artifacts read as spurious
motion — WebP q80 at 192×256 is 10× worse than uncompressed 192×256); (3) at matched d_pose, **HEVC-video
is ≈4× cheaper than WebP-still** (0.067 vs 0.294 for ~+1 vs +0.23 increment). The cheapest configs with a
*small* increment cost **rate 0.03–0.07 for 13 keyframes**.

---

## 4. Why the §3 numbers are a PESSIMISTIC upper bound (two caveats, both → cheaper)

The §3 proxy is not the real decode. Both discrepancies bias the measured cost UPWARD:

**4.1 Sharp-partner pessimism.** At decode PoseNet reads `(warp(keyframe), witness_synthetic_f1)`. The
partner frame1 is the **synthetic witness render** (class-mean colours, low detail at 384×512) — NOT a sharp
real frame. My probe pairs the degraded warp against a SHARP `gt_f0`, so it penalizes the sharp/blurry
appearance MISMATCH. Against the already-blurry synthetic partner, a low-res keyframe likely matches
BETTER → the resolution floor is probably LOWER (cheaper) than §3 shows. (Resolving this needs the trained
witness's synthetic f1 → GPU.)

**4.2 Residual-compensation (the PIVOTAL uncertainty).** The carrier ships a trained per-pair 6-DOF residual
`dxi` (rank-6; d_pose→ξ Jacobian is rank-6). The residual RE-OPTIMIZES the twist per pair to hit PoseNet's
target. My probe uses a FIXED ξ, so it conflates twist-error (residual-closeable) with texture-error. IF a
degraded keyframe still lets PoseNet read a pose that ξ can STEER to the target, the residual closes d_pose
regardless of blur → **cheap low-res keyframes become viable (rate ~0.006–0.015)**. IF the blur saturates
PoseNet into a degenerate readout ξ can't steer, the degradation is irreducible. **This is the single
measurement that decides cheap-vs-expensive** and it requires the trained residual (GPU, #205). §6 step 1.

**Honest statement:** §3's "d_pose-safe keyframe = rate 0.03–0.07" is a PESSIMISTIC upper bound. The true
cost is between ~0.006 (if 4.1+4.2 fully save it) and ~0.07 (if neither does). Do not treat 0.03–0.07 as
final — treat it as "the cost if the residual can't absorb keyframe degradation."

---

## 5. Ranked minimization options (rate-reduction × d_pose-cost × decode-feasibility)

Ranked by expected rate reduction at fixed d_pose ≤ ε (≈0.018 score contribution). `[M]`=measured here,
`[D]`=derived. All keyframe-storage options are **rule-118 clean**: the keyframe bytes are VIDEO-DERIVED →
COUNTED in `archive.zip`; the DECODER (libwebp/libx265/ffmpeg, generic code) is FREE, like brotli. Decode of
≤40 frames is milliseconds ≪ 30-min budget. (Dependency-closure caveat: the codec lib must be in the inflate
runtime tree — the "brotli-missing" bug class; pin it.)

| # | lever | mechanism | expected rate effect | d_pose cost | feasibility | evidence |
|---|---|---|---|---|---|---|
| **R1** | **ego-warp-predicted residual coding** | predict keyframe_k from keyframe_0 warped by the KNOWN cumulative ξ (FREE, rule-118), store only the residual, entropy-code it | **HIGH** — motion is exact/free (unlike generic video block-search), so residual ≪ generic P-frame; plausibly 2–5× under HEVC-video | ≈0 (reconstruction, not a new warp) | med (compose cumulative ξ correctly) | **[D]** strongest structural lever; NOT yet measured |
| **C1** | **reduce keyframe COUNT (verify pose-reach)** | the 13-count assumes the d_SEG partition reach (47 pairs); measure the d_POSE reach of ONE warped keyframe; fewer keyframes = linear rate | **HIGH** — every keyframe dropped is linear; 13→8 = −38% | rises with reach (residual helps) | high ($0 through-R PoseNet sweep) | **[D]** measurable next; partition reach 47 is d_seg-only |
| **R2** | **temporal HEVC/AV1 video** | store the keyframes as one mini-video (I+P), not per-frame | **MED-HIGH** ≈4× vs WebP-still at matched d_pose (0.067 vs 0.294) | matched | high (ffmpeg/x265) | **[M]** §2/§3 |
| **R3** | **luma-primary + subsampled chroma** | PoseNet is chroma-null (YUV6 = 4 luma + 2 heavily-subsampled chroma); store luma at full-kf-res, chroma at ½ res or flat | **MED** ≈1.3–2× | small (measure chroma-drop d_pose) | high | **[M]** luma≈1.3–2× cheaper; d_pose-of-chroma-drop TODO |
| **R4** | **drop resolution as far as d_pose allows** | pick the min resolution above the cliff at the trained residual | **HIGH if 4.2 holds** (192×256→96×128 = ≈8×) | CLIFF below ~192×256 at fixed ξ; UNKNOWN at trained residual | high | **[M]** cliff at 96×128 fixed-ξ; §4.2 caveat |
| **R5** | **optimal keyframe SELECTION** | place keyframes at low-ego-motion / high-texture-stability pairs to maximize reach & minimize residual | LOW-MED | reduces residual | med | **[D]** |
| **R6** | **INR keyframe basis** | fit a small shared still-image INR over the 13 keyframes + per-kf code, then warp its output | LOW-MED, RISKY | risky | low | **[D]** distinct from the amortized-luma-RECON collapse (that reconstructed POSE from luma; this reconstructs a real IMAGE then warps) but shares its failure risk — Pareto-watch |

**Pareto-dominated / avoid:** storing keyframes at NATIVE resolution (rate 0.13–2.9, §1); lossy compression
BELOW ~192×256 (d_pose cliff, §3); reconstructing pose directly from an INR luma carrier (the measured
amortized-luma collapse d_pose 2.67–12.66 — R6 must not regress into it).

---

## 6. Recommended #202 byte-close path toward sub-0.15

1. **[PIVOTAL, needs #205 trained witness — GPU] Measure residual-compensation (§4.2).** Once #205 trains a
   witness with `--pose-carrier` + `w_pose>0`, re-run the §3 degradation sweep BUT let the trained residual
   `dxi` re-optimize per pair (and use the synthetic witness f1 as partner, §4.1). Output: the true
   d_pose-vs-(resolution,quality) surface. This decides the whole branch:
   - **cheap branch** (residual absorbs degradation): store 96–192-wide keyframes, HEVC-video → rate
     **~0.006–0.015** → total rate ~0.061–0.070 → sub-0.15 reachable on the d_seg bet.
   - **expensive branch** (irreducible): d_pose-safe keyframe ~0.03–0.07 → total rate ~0.085–0.125 →
     sub-0.15 needs R1 + C1 + R3 stacked, or a pose-path rethink.
2. **[HIGH-EV, $0 now] Build + measure R1 (ego-warp-predicted residual coding).** Compose cumulative ξ
   across the reach segment, warp keyframe_0 forward, subtract from the real target keyframe, entropy-code
   the residual. Compare rate to R2 at matched decoded-frame d_pose. This is the rule-118 win (free exact
   motion) and is the most likely single lever to make the expensive branch survive.
3. **[HIGH-EV, $0 now] Measure the d_POSE reach (C1).** Warp ONE keyframe forward by cumulative ξ to pairs
   a+k, measure d_pose vs k through PoseNet → the true keyframe count (not the d_seg partition 47). Fixes the
   13-vs-40 uncertainty.
4. **[MED, $0] Measure R3 (luma-primary + chroma-drop) d_pose.** Confirm chroma is droppable for the
   keyframe (PoseNet chroma-null) → ≈1.3–2× at ≈0 d_pose cost.
5. **[byte-close] Once (1) picks a branch:** wire the chosen keyframe codec + count into
   `tools/compose_witness_archive.py` as an explicit `keyframe_blob` line item (its byte accounting currently
   carries the partition keyframes + pose sidecar but NOT the real-luma keyframes), byte-close the 4-section
   archive with the keyframe payload counted, and exact-eval via `tools/levelset_byte_close_and_eval.py` →
   the first HONEST first-row rate incl. keyframes.

**§7/DAG correction to propagate (#219 triality):** restate §7's rate with an explicit keyframe line item;
replace the borrowed "13 keyframes ≈ 0.0060 (partition)" with "real-luma keyframes: rate 0.006–0.07 pending
the §4.2 residual-compensation measurement; partition keyframes (0.0060) are a SEPARATE d_seg payload."

---

## 7. Provenance / reproducibility

- **Codecs:** PIL 12.2 (JPEG/WebP/JPEG2000/AVIF), ffmpeg 8.1 (libx265 works; libaom-av1 video failed via
  raw-pipe — retry with `-pix_fmt yuv420p`), brotli 1.2. **Data:** real `gt_f0` from
  `experiments/results/mlx_fleet_gt_cache/gt_n600.npz` (13 true keyframes) + `gt_n96.npz` (d_pose sensitivity,
  n96). **d_pose authority:** frozen CPU-torch PoseNet via
  `experiments.train_witness_realized_through_R_mlx.cpu_verdict_d_pose_batch` (NEVER MPS), NO-FAKE
  self-check `PoseNet(gt pair)==gt_poses` enforced. **Warp:** `tac.boundary_math.warp_real_luma_frame0`
  (`warp_frame0_uint8_numpy`, s_t=0.044).
- **Method (reproducible):** (probe A) compress real keyframes at resolution×quality×codec, rate=25·B·N/B0;
  (probe B) degrade keyframe (down→[codec]→up native), warp fixed-ξ, d_pose through PoseNet; (probe C) same
  as B via HEVC-video roundtrip, bytes on the true 13 keyframes.
- **Caveats:** n96 for d_pose sensitivity (curve shape; n600 confirmation flagged — the §6 GPU measurement
  should run n600). Fixed-ξ / sharp-partner → PESSIMISTIC (§4). NOT a contest score; pointer **0.19110
  UNMOVED**.
- **Cross-refs:** `warp_real_luma_frame0.py` docstring (byte accounting); `n205_phase3_recursive_adversarial_
  review_verdict_20260702.md` §4 (the S-budget correction this sharpens); `screw_reach/reach_n96.json` (the
  partition reach); `canonical_research_index_rate_20260629.md` R8 (the 0.0060 partition figure).
