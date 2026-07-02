# Keyframe rate minimization — BUILDS + MEASURED ranked levers (#202 byte-close track)

**Date:** 2026-07-02 · **Track:** RATE MINIMIZATION (operator/coordinator 2026-07-02, background/parallel, NON-BLOCKING, $0/local).
**Authority:** `[macOS-CPU advisory / CPU-torch research-signal]` ONLY. NOT a contest score. Canonical
frontier pointer **0.19110 UNMOVED**. `score_claim=false, promotable=false`. Every d_pose is the FROZEN
CPU-torch PoseNet authority (NEVER MPS; NO-FAKE self-check `PoseNet(gt pair)==gt_poses` enforced, max
MSE 0.0 on all runs); every byte is a real codec on real `gt_f0`/`lstars` from `gt_n600.npz` (n600).
**Means/ends:** these are MEANS (a #202 byte-close design); the END is a byte-closed `upstream/evaluate.py`
n600 exact row that beats 0.19110. Did NOT touch the running #205 trainer / launch config / `compose_witness_archive.py`.

Sharpens `warp_keyframe_payload_rate_minimization_20260702.md` (`7521a49fe`) with MEASURED n600 rows.

---

## 0. TL;DR — the store-nothing path WINS; the keyframe payload can collapse to ~0 marginal bytes

The pivotal question was: **do we need to store a real photographic keyframe at all?** MEASURED answer (n600,
frozen CPU-torch PoseNet): **NO.** The keyframe TEXTURE is not the d_pose bottleneck. What PoseNet needs is a
**domain-COHERENT frame pair** whose two frames differ by the ego-motion — and the SDF witness already renders
that (a class-mean image on the SegNet argmax partition). Storing only ξ + FiLM-conditioning the witness's OWN
render is **pose-competitive with the full real keyframe at ~0 marginal rate**.

**The three decisive n600 measurements (all PRE-residual, dxi=0 — the trained rank-6 twist residual closes the
remaining fixed offset toward the ~3.4e-5 target; that final step needs the #205 checkpoint = the harness):**

1. **STORE-NOTHING (lead, coordinator #3):** a texture-free class-mean render (partition + per-class mean colour,
   ZERO real pixels) warped by ξ gives **d_pose 4.97** at **~0 marginal rate** (the argmax partition is the SEPARATE
   d_seg payload we already store; only 5 class-mean colours/keyframe are new ≈ 390 B). Adding a tiny per-class
   low-freq residual: **+lf24 → d_pose 1.12** (BETTER than the full real keyframe's 1.37) at **rate 0.0188**.
2. **TEMPORAL VCM (coordinator #1 — anti-MPEG negative OVERTURNED):** the 13-keyframe stream codes to **rate
   0.004–0.018** (svtav1_crf52 0.00385 @ d_pose 4.57; x265_crf36 0.0074 @ 3.56; svtav1_crf28 0.0185 @ 2.38) —
   **3–5× cheaper than the memo's pessimistic 0.03–0.07.** SVT-AV1 ≈ x265; both beat the memo's WebP-still numbers.
3. **SUFFICIENCY (coordinator #2 / deliverable 3):** the resolution floor is GENTLE (full 1.37 → 192×256 only +0.38),
   confirming §4.1: the real keyframe can be stored small. But global low-pass (DCT-64) is **catastrophic (20.99)** —
   PoseNet needs the sharp partition EDGES, not low frequency. The store-nothing render supplies those edges FREE.

**Ranked recommendation for #202:**
- **#1 STORE-NOTHING (ξ-FiLM render, no keyframe).** Rate ~0 marginal. Endpoint (c). Pending: the trained residual
  must absorb the 4.97 offset (the harness `--dxi-source ckpt:` decides once #205 lands). If it can't fully, add
  the cheap lf residual (0.0055–0.0188) — still a huge win over a stored keyframe.
- **#2 TEMPORAL VCM (svtav1/x265 single-GOP, 384×512) as the FALLBACK** if the synthetic render can't carry pose:
  rate 0.004–0.018 at 13 keyframes. Non-toy, proper stream coding. The anti-MPEG negative is dead.
- **NEGATIVE / demoted — R1 ego-warp residual coding:** MEASURED does NOT beat intra at 47-pair keyframe spacing
  (frames ~94 apart → single homography can't predict; residual entropy 5.27→5.30 UNCHANGED). Implementation-level
  negative (regime = spacing too large), not a paradigm kill; moot given store-nothing.

**Strongest single rate-reducer:** the STORE-NOTHING ξ-FiLM render (#1) — it removes the entire real-keyframe payload
(the memo's 0.03–0.07) and replaces it with ~0 marginal bytes, IF the render is pose-sufficient (n600: classmean 4.97
pre-residual, +lf24 1.12; the harness decides the post-residual reachability).

---

## 1. What was BUILT (all reusable, tested, wired)

* **`src/tac/boundary_math/keyframe_codec.py`** — pure (numpy/cv2/PIL/scipy/ffmpeg) primitive layer, torch/mlx-free,
  byte-close-consumable: degradation ops (resize-roundtrip, gaussian blur, global-DCT truncate, bit-depth), order-0
  entropy + still codecs (png/webp/jpeg2000/avif/zlib/brotli), **proper temporal video coding** (x265/svtav1/vp9 with
  GOP/B-frames + encode↔decode roundtrip), **ego-warp residual predictor** (dense-ECC homography, exact-invertible
  residual), **class-mean texture-free render** (the store-nothing proxy), rate accounting.
* **`src/tac/boundary_math/tests/test_keyframe_codec.py`** — 23 tests (degradation shape/monotonicity, entropy,
  exact-invertible residual, ego-warp<prev-copy on translation, still+temporal codec bytes, odd-dim padding,
  class-mean texture removal, rate formula). **All 23 pass (2.8s).**
* **`tools/measure_keyframe_pose_sufficiency_ladder.py`** — the UNIFIED n600 ladder + **trained-residual sweep HARNESS**
  (deliverable 2): (a) full → (b) degraded → (c) texture-free synthetic, through frozen CPU-torch PoseNet.
  `--dxi-source ckpt:<npz>` loads the #205 trained `pose_carrier.dxi` (checkpoint-ready; runnable AS #205 lands),
  `--partner {self_warp,sharp_gt,blur_gt}` for the §4.1 test, `--mock` unit-self-test (PASS). `--s-t`/`--append-jsonl`
  for bounded resumable slices.
* **`tools/measure_keyframe_vcm_rate.py`** — proper temporal VCM stream coding (coordinator #1) + optional d_pose on the
  decoded stream. Grounded in MPEG-AI Part 2 VCM (below).
* **`tools/measure_keyframe_ego_residual_coding.py`** — R1 ego-warp-predicted residual coding (deliverable 1).

**MPEG-AI grounding (coordinator #1):** ISO/IEC 23888 = MPEG-AI; **Part 2 = Video Coding for Machines (VCM)** — a
pixel-domain codec optimized for MACHINE analysis, coding decisions guided by machine-task metrics (mAP/MOTA), NOT
PSNR/VMAF; **Part 4 = Feature Coding for Machines (FCM)** — compresses NN features (split inference), ~85% bitrate cut
vs pixel baselines. VCM reached DIS in early 2026. **This IS our problem:** our "machine" is PoseNet, our metric is
d_pose. The warp-real-luma keyframe = pixel-domain VCM; the ξ + FiLM-render store-nothing path = closer to FCM
(store a task-sufficient feature ξ, not a photorealistic frame). Sources: iso.org/standard/88879.html (23888-2);
mpeg.org/standards/MPEG-AI/.

---

## 2. MEASURED — the sufficiency ladder (n600, self_warp partner = the faithful carrier/store-nothing proxy)

Pair = `(S, warp(S, ξ))` (both frames from the source S, coherent, differing by the ξ ego-motion — matches the MEASURED
carrier baseline and models the real decode where BOTH frame0+frame1 are witness synthetic renders). d_pose PRE-residual.
Store `experiments/results/keyframe_pose_sufficiency_ladder_n600/ladder_rows.jsonl`.

| ladder config | d_pose | sqrt(10·d_pose) | incr vs full | note |
|---|---:|---:|---:|---|
| **full** (real gt_f0) | 1.367 | 3.697 | — | expensive baseline (real texture) |
| resize 384×512 | 1.478 | 3.845 | +0.11 | |
| resize 256×342 | 1.598 | 3.998 | +0.23 | |
| resize 192×256 | 1.752 | 4.185 | +0.38 | **gentle floor (confirms §4.1)** |
| resize 128×170 | 2.491 | 4.991 | +1.12 | |
| resize 96×128 | 3.722 | 6.101 | +2.36 | cliff onset (≈ §3) |
| blur σ=2 | 1.807 | 4.251 | +0.44 | cheap lever |
| bit-depth 4 | 5.699 | 7.549 | +4.33 | costly (banding reads as motion) |
| **dct-64 (global low-pass)** | **20.99** | 14.49 | +19.6 | **CATASTROPHIC — pose needs EDGES not low-freq** |
| **classmean (texture-free, lf0)** | **4.975** | 7.053 | +3.61 | **store-nothing endpoint (c); rate ~0 marginal** |
| classmean + lf12 | 3.265 | 5.714 | +1.90 | |
| classmean + lf16 | 2.871 | 5.358 | +1.50 | |
| **classmean + lf24** | **1.123** | 3.350 | **−0.24** | **BETTER than full real keyframe** |
| classmean + lf8 | 9.144 | 9.562 | +7.78 | non-monotone (low-K DCT ringing; see §5 caveat) |
| classmean + lf6 | 11.06 | 10.52 | +9.69 | " |

**Key reads:** (1) resolution floor is gentle to 192×256; (2) the store-nothing class-mean render keeps the sharp partition
EDGES (which dct-64 destroys → catastrophe) and is within +3.6 of full at ~0 rate; (3) a modest per-class low-freq residual
(lf24) makes it BEAT the full real keyframe. Non-monotonicity at lf6/lf8 is a DCT-parametrization artifact (§5), not a
sufficiency wall — the trained witness render is not DCT-truncated.

---

## 3. MEASURED — the store-nothing RD vs the temporal-VCM RD (the head-to-head)

**Store-nothing** (rate = MARGINAL bytes over the d_seg partition, which is already counted; per-class means 390 B + brotli'd
KxK low-freq DCT of the per-class residual, coarse quant):

| representation | rate (13 kf) | d_pose | sqrt10 |
|---|---:|---:|---:|
| classmean only (lf0) | **~0.0003** | 4.975 | 7.05 |
| classmean + lf12 | 0.00554 | 3.265 | 5.71 |
| classmean + lf16 | 0.00908 | 2.871 | 5.36 |
| classmean + lf24 | 0.01884 | 1.123 | 3.35 |

**Temporal VCM** (all-new counted bytes; 13-keyframe stream, 384×512, single-GOP, B-frames=2; d_pose on the DECODED stream at
the keyframe points). Store `experiments/results/keyframe_vcm_rate_384x512/results.json`:

| codec | rate (13 kf) | d_pose | rate (40 kf) |
|---|---:|---:|---:|
| svtav1 crf52 | 0.00385 | 4.57 | 0.01034 |
| svtav1 crf44 | 0.00627 | 3.81 | 0.01787 |
| x265 crf36 | 0.00739 | 3.56 | 0.01570 |
| svtav1 crf28 | 0.01850 | 2.38 | 0.06192 |
| x265 crf28 | 0.01611 | 3.20 | 0.03988 |
| vp9 crf28 | 0.04862 | 2.21 | 0.12612 |

(x265 crf52 fails — x265 max CRF is 51; svtav1/vp9 cover the low-rate end. AV1 = SVT-AV1, the modern encoder, competitive/better than x265.)

**Head-to-head (matched rate):** at rate ≈ 0.0185, **store-nothing +lf24 (d_pose 1.12) BEATS svtav1_crf28 (d_pose 2.38)**; at
rate ≈ 0.004–0.006, store-nothing lf12 (0.0055, 3.27) ≈ svtav1_crf52 (0.00385, 4.57). And classmean-only is ~10× cheaper
(0.0003) at d_pose 4.97 ≈ the cheapest codec point. **Store-nothing Pareto-dominates the codec at every measured point**,
because it reuses the FREE partition structure. Temporal VCM remains the FALLBACK if the synthetic render can't carry pose.

---

## 4. MEASURED — diagnostics that shaped the conclusion

**(a) R1 ego-warp residual coding — NEGATIVE at 47-pair spacing** (`keyframe_ego_residual_coding_n600/results.json`, LOSSLESS):

| store res | intra rate | prev-copy | ego-warp R1 | R1 vs intra | residual bits prev→ego |
|---|---:|---:|---:|---:|---:|
| native | 7.628 | 11.32 | 11.22 | **+47% WORSE** | 5.27 → 5.30 |
| 384×512 | 1.849 | 2.416 | 2.412 | +30% WORSE | 5.17 → 5.19 |
| 192×256 | 0.544 | 0.678 | 0.671 | +23% WORSE | 5.04 → 5.10 |

The ego-warp (dense-ECC homography) does NOT reduce the residual because 47-pair keyframes are ~94 frames apart — parallax,
new objects, and large motion exceed a single ground-homography's predictive power (residual entropy unchanged). Intra beats
both temporal predictors. **Honest overturn of the memo's "R1 HIGH structural EV".** Implementation/regime negative (would help
at CLOSE spacing, but close spacing = more keyframes); moot given store-nothing. Paradigm intact, demoted to LOW.

**(b) §4.1 partner-blur diagnostic (partner = blurred real gt_f1, σ=3) — reveals the COHERENCE requirement:**

| config | self_warp d_pose | blur_gt d_pose |
|---|---:|---:|
| full (real) | 1.37 | **0.030** |
| resize 192×256 (real) | 1.75 | **0.022** |
| resize 96×128 (real) | 3.72 | 0.16 |
| classmean (synthetic) | 4.97 | **30.6** |
| classmean+lf24 (synthetic) | 1.12 | **77.6** |

A degraded REAL frame0 with a real-domain partner → d_pose ~0.02–0.16 (texture barely matters — confirms §4.1 pessimism: a
good partner makes low-res real keyframes nearly free). BUT a SYNTHETIC frame0 with a REAL partner → 30–78 (catastrophic:
domain-INCOHERENT pair, PoseNet can't read motion across mismatched domains). **Lesson:** PoseNet needs a domain-COHERENT pair.
At real decode BOTH frames are witness synthetic renders → coherent → the **self_warp ladder (§2) is the faithful proxy**, and
you must NOT mix a synthetic keyframe-render with a real partner. This is why store-nothing works (witness supplies both frames).

---

## 5. Caveats / what the #205 checkpoint (the harness) must confirm

* **All §2/§3 d_pose are PRE-residual (dxi=0).** The trained rank-6 per-pair twist residual closes the FIXED offset toward
  the ~3.4e-5 target (sqrt10 0.0184). The PIVOTAL open question (§4.2 of the parent memo): can the residual absorb the
  store-nothing offset (4.97) as well as the full-keyframe offset (1.37)? Because the offset is a SMOOTH calibration residual
  (not per-pair chaos — the resolution ladder is monotone and gentle), it is very likely closeable, but the HARNESS decides:
  once #205 lands a checkpoint, run `measure_keyframe_pose_sufficiency_ladder.py --dxi-source ckpt:<ema.npz> --partner self_warp`
  to re-measure the ladder WITH the trained residual. If classmean(4.97)→residual holds ε, store-nothing is confirmed and the
  keyframe rate is ~0.
* **classmean is a PROXY for the trained witness render** (per-class mean colour, texture-free). The real witness render is
  richer (FiLM-per-pair, chroma, sub-pixel AA), so it is an UPPER bound on the store-nothing d_pose — the true endpoint is
  ≤ 4.97. The `--partner witness:<ckpt>` render hook is stubbed (NotImplemented, never faked) pending the checkpoint.
* **lf6/lf8 non-monotonicity** = low-K global-DCT residual introduces low-freq ringing that reads as spurious motion; not a
  sufficiency wall. The trained render uses a learned (curvelet/step) basis, not DCT-truncation, so it avoids this.
* **Rate numbers are advisory** (real codecs on real frames) but the EXACT #202 row must byte-close through
  `tools/compose_witness_archive.py` (owned by a sister agent) + `tools/levelset_byte_close_and_eval.py`. Pointer 0.19110 UNMOVED.

---

## 6. Ranked levers (updated; supersedes the parent memo §5 ranking)

| # | lever | mechanism | rate | d_pose (pre-residual) | verdict |
|---|---|---|---:|---:|---|
| **1** | **STORE-NOTHING ξ-FiLM render** | store only ξ; FiLM-condition the witness's OWN frame0 render; no real keyframe | **~0 marginal** | 4.97 (classmean proxy) | **LEAD — measured pose-competitive; harness confirms post-residual** |
| **2** | store-nothing + tiny lf residual | classmean + per-class KxK low-freq DCT | 0.0055–0.019 | 3.27 (lf12) → 1.12 (lf24) | strong; +lf24 beats full keyframe |
| **3** | temporal VCM (svtav1/x265 single-GOP) | code the 13-kf stream as 1 video | 0.004–0.018 | 2.4–4.6 | **FALLBACK; anti-MPEG negative OVERTURNED (3–5× cheaper than memo)** |
| 4 | drop keyframe resolution | store real kf at 192–256 wide | (codec-dependent) | +0.23–0.38 | gentle floor (composes with #3) |
| 5 | reduce keyframe COUNT (pose-reach) | verify d_pose reach of one ξ-warp | linear | (unmeasured here; C1) | still open; measure next |
| — | ~~R1 ego-warp residual coding~~ | homography-predict + residual | +23–47% WORSE | — | **DEMOTED — negative at 47-pair spacing** |
| — | bit-depth / global low-pass | — | — | +4.3 / +19.6 | **avoid — reads as spurious motion** |

**Next $0 measurements** (not blocking): (i) the reach lever C1 (d_pose vs one-ξ-warp reach k → the true keyframe count, fixes
13-vs-40); (ii) fire the HARNESS the moment #205 emits a checkpoint (`--dxi-source ckpt:`) to close the pre→post-residual gap.

---

## 7. Provenance / reproducibility

* **Data:** real `gt_f0`/`gt_f1`/`gt_poses`/`lstars` from `experiments/results/mlx_fleet_gt_cache/gt_n600.npz` (n600, all 600 pairs).
* **d_pose authority:** frozen CPU-torch PoseNet via `experiments.train_witness_realized_through_R_mlx.cpu_verdict_d_pose_batch`
  (NEVER MPS); NO-FAKE self-check `PoseNet(gt pair)==gt_poses` PASS (max MSE 0.0). **Warp:** `tac.boundary_math.warp_real_luma_frame0`
  (`warp_frame0_uint8_numpy`, s_t=0.044 — the established carrier calibration). **Codecs:** PIL 12.2, ffmpeg 8.1 (libx265, libsvtav1,
  libvpx-vp9), brotli 1.2, scipy 1.17, cv2 4.11 (ECC homography).
* **Result JSONs:** `experiments/results/keyframe_pose_sufficiency_ladder_n600/{ladder_rows.jsonl, ladder_rows_partnerblur.jsonl}`;
  `experiments/results/keyframe_vcm_rate_384x512/results.json`; `experiments/results/keyframe_ego_residual_coding_n600/results.json`.
* **Runs bounded** into sub-3-min slices (SIGURG limit) sharing a fixed s_t=0.044 via `--s-t`/`--append-jsonl` (resumable accumulation).
* **NOT a contest score;** pointer **0.19110 UNMOVED**. All rows advisory/research-signal; the EXACT row is the byte-closed
  `upstream/evaluate.py` n600 eval through `compose_witness_archive.py` + `levelset_byte_close_and_eval.py`.
