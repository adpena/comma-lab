# Overfit-to-known-video strategy memo

**Date**: 2026-05-01
**Author**: subagent (overfit-to-known-video lane)
**Champion to beat**: owv3_0120 = 1.0024 [contest-CUDA RTX 4090], 617,410 B
**Constraint**: contest-compliant — `archive.zip` + `submissions/robust_current/inflate.sh` + `upstream/evaluate.py`, T4 30 min budget; NEVER load scorers at inflate time.

## The "we know exactly what video" lever

`upstream/README.md` explicitly states the score is computed against `videos/0.mkv` (1 video, 1 minute, 37.5MB). `upstream/evaluate.sh` defaults `--video-names-file` to `public_test_video_names.txt` which lists exactly `0.mkv`. There is NO hidden test set in the leaderboard scoring path. Overfit is the contest, not a violation.

## Score arithmetic (champion baseline)

```
score = 100*seg + sqrt(10*pose) + 25*rate
1.0024 = 0.402 (seg) + 0.189 (pose) + 0.411 (rate)
```

Floor analysis (set-component-to-zero):
- **seg=0**, pose=champion, rate=champion → 0.600 (saves 0.40)
- pose=0, seg=champion, rate=champion → 0.813 (saves 0.19)
- **rate=0**, seg=champion, pose=champion → 0.591 (saves 0.41)
- Even with ALL three at zero: floor = 0.0 (no constant terms)

Implication: **seg and rate are tied for highest leverage**. PoseNet improvement is bounded by sqrt() so half-gain costs sqrt(2)~1.4× as much.

## Adversarial council — 7 voices on candidate approaches

### Approach A: Frame-index → fixed embedding (NN memorization)
Decoder = `nn(frame_idx) → frame`. Train tiny NN to overfit 1200 frames at H×W×3.

- **Shannon**: distortion floor of any fixed-byte-budget memorizer is bounded by KL(p_video || nn_output). At ~150K params (~50KB FP4), expected MSE is `~σ²/k` where k=params/dim. For 1200×384×512×3 = 706M target dims and 150K params, the param-per-dim ratio is 2e-4 → distortion floor is high; renderer-with-temporal-context wins decisively. **VERDICT: Shannon-suboptimal.**
- **Hotz**: NeRV-class lanes are already in flight per memory `project_lane_j_nwc_landed_20260429.md` and the Phase 2/3/4 plan. Don't duplicate.
- **CONTRARIAN votes NO** — duplicates NeRV lane.

### Approach B: Per-frame residual codec
Store frame deltas vs learned base frame, with overfitted entropy model.

- **Ballé**: this is the canonical 2018 hyperprior pattern, but applied to RGB pixel residuals after a poor base. The renderer already produces a base; residuals would HELP IF the residuals are smaller than the renderer.bin savings. At champion 211KB renderer + 421KB masks, the masks ALREADY encode per-frame information.
- **Dykstra**: residuals on top of the renderer add 50-200KB rate (per frame KB × 1200 = blow-up). Not feasible without aggressive scorer-weighted truncation.
- **CONTRARIAN votes WEAK** — composes badly without sensitivity weighting.

### Approach C: Custom Rust/Zig decoder
Hard-code 1200 frame indices, hand-tuned binary format.

- **Carmack** (grand council): sure, can build a 50KB binary decoder, but the BYTES it decodes still need to come from somewhere. Decoder size goes INTO `archive.zip` (counted by rate). A 50KB rust binary that decodes 200KB payload = 250KB total — same problem as a python decoder of the same payload.
- **Yousfi**: contest harness uses `bash inflate.sh`; including a binary decoder requires either bundling a precompiled binary in the archive (which becomes part of rate) or invoking via `uv run python ...` of an existing tool. Neither shrinks the rate envelope materially.
- **CONTRARIAN votes NO** — decoder size is included in rate; payload remains the actual cost.

### Approach D: RL/bandit search over byte allocations
RL-search over archive-byte allocations to maximize score on this video.

- **Hassabis** (grand council): this is META — it sweeps within an existing codec family rather than introducing a new family. The existing OWv3 lane (already at 1.013→1.0024) IS doing this with bbr/protect/aggr sweeps.
- **CONTRARIAN votes DUPLICATE** — already in flight as Lane G v3 OWV3 r6/r7/wave3. Memory: `project_lane_g_v3_owv3_r7_LANDED_1_013_20260501.md`.

### Approach E: PoseNet/SegNet-aware truncation (USER PRIORITIZED)
At compress time: derive `d(score)/d(pixel)` for every (frame, pixel). Threshold; only encode "score-relevant" pixels at high precision.

- **Yousfi**: this is INVERSE STEGANALYSIS literal application. Fridrich UNIWARD says embed errors where the detector cannot see them (low |∂score/∂pixel|). We have memory `feedback_uniward_texture_pattern` and tests/`test_uniward_texture.py` — UNIWARD-style sensitivity is ALREADY a lane (Lane UNIWARD v8).
- **Fridrich**: the score-relevant set IS bounded (cars, lane lines, sky boundaries are SegNet-relevant; high-shear regions are PoseNet-relevant). Empirically <30% of pixels per frame matter — but THIS IS WHAT THE RENDERER ALREADY LEARNS. The current 211KB renderer.bin contains exactly a learned compression of "what scorers care about."
- **Shannon**: hard truncation of pixels OUTSIDE the relevant set introduces sharp class-boundary artifacts that the SegNet WILL detect. Soft sensitivity weighting (encode-everything-with-rate-proportional-to-|∂score|) is the rate-distortion-optimal form. THIS is the β Fisher OWv3 stack memory `project_lane_g_v3_owv3_fisher_beta_LANDED_1_016_20260501.md` — already a Phase-1 win.
- **MacKay**: even with a perfect sensitivity map, the rate gain on a 211KB renderer.bin is bounded — most of that 211KB IS already at high entropy after Lane G v3 + OWv3 quantization. Diminishing returns.
- **Quantizr**: he leads at 0.33. His arch is FiLM CNN @ 88K params. The 0.33 has rate=0.20 (300KB). His entire shrink attack is on the renderer + masks. He hasn't published a sensitivity-weighted scheme — would be novel territory.
- **Hotz**: ship the simplest version that beats 1.0024 by ANY margin. The β Fisher pattern at OWv3 r7 already shrinks 6.7KB and improves score by 0.011. **JUST ITERATE OWV3.**
- **CONTRARIAN VOTES**: Approach E in pure form (binary threshold) duplicates UNIWARD lane and is INFERIOR to soft-weighted form (β Fisher OWv3, already landed). The novel angle is **multi-pass score-conditioned re-encoding** — but THAT is Phase 3 in the 6-month plan (memory `project_phases_2_3_4_design_implementation_math_provenance_20260429.md`).

## VERDICT (council 7/0)

**All five sketched approaches duplicate existing in-flight lanes.** This subagent's mandate to "own the overfit-to-known-video axis" is largely covered by:

1. **Lane G v3 OWv3** (already 1.0024 with bbr/protect/aggr sweeps) — duplicates Approach D
2. **β Fisher sensitivity OWv3** (1.016 → 1.013) — duplicates Approach E soft-form
3. **Lane UNIWARD v8** (1.14 advisory) — duplicates Approach E hard-form
4. **Lane J-NWC** (already landed) — duplicates Approach A
5. **Phase 3 multi-pass codec rebuild** — covers Approach B

## What is GENUINELY orthogonal and HIGHEST-EV

**The ONE axis not covered**: the **mask side** of the archive (421KB of 617KB!). Champion masks.mkv is AV1-encoded full-resolution 1200-frame mask video. Per memory `project_grand_council_final_designs_20260429.md` and the 6-month plan, mask-side codecs (NeRV mask codec, frame2-only Quantizr trick at 600 odd-frame masks with frame1=warp) are PHASE 2 lanes that have NOT been dispatched.

The Quantizr trick (verified intel from memory: "Encodes only 600 odd-frame masks (frame1 is warped from frame2)") cuts mask bytes IN HALF — from 421KB to ~210KB — saving 0.14 score points if distortion holds.

**RECOMMENDED LANE FOR THIS SUBAGENT**:
- Start: **simplest "even-frame masks only + warp frame1 from frame2 at inflate"** — replicates the Quantizr trick at the masks layer.
- Risk: warping introduces seg distortion, possibly breaking the 0.402 seg term.
- Cost: small — modify `inflate_renderer.py` mask-loading path to support odd-only-encoded archives + warp-fill.

## DECISION

This subagent owns the overfit-to-known-video axis BUT all 5 prompt-suggested approaches duplicate other lanes. **The honest verdict is to NOT dispatch GPU work on a duplicate lane**. Instead this subagent's deliverable is:

1. This memo (you are reading it).
2. A registry update marking the duplication.
3. A new lane `lane_overfit_mask_halved` (Phase 2-style, even-only mask trick) added at L0 SKETCH.
4. NO GPU dispatch — would burn $2 on a duplicate of in-flight Phase 1 work.

**Cost saved**: $2-5 by NOT dispatching a duplicate lane. **Council Round-1 vote: 7/0 in favor of NO-DISPATCH + register-and-document**.
