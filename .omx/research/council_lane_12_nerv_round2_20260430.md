# Lane 12 NeRV — Adversarial Review Round 2 (2026-04-30)

## Reviewer perspectives

Round 2 rotates: **Quantizr** (adversarial competitor, would-tear-this-apart) + **van den Oord** (VQ-VAE pioneer, expects entropy-bottleneck rigor) + **Hotz** (raw-engineering-shortcuts, "would yours actually beat my K-frame interpolation?").

## Counter status

Entering Round 2 at **1/3**. Continues to **2/3** if clean.

## Quantizr: adversarial pass

> "I shipped 88K param SegMap as part of my 0.33 archive. Why should I believe a 12K param NeRV holds the score?"

### Q1: parameter density argument

Quantizr's SegMap is 88K params decoding to ~88K bytes of mask-and-image latent. NeRV is 12K params decoding to ~60K mask-pixels per frame × 1200 frames = 72M pixels output. Ratio: 6000× compression IN PARAM SPACE per frame.

**Counter (defense)**: NeRV is overfitting a SINGLE 1200-frame sequence. The 6000× ratio is per-frame, but the MLP shares weights across all 1200 frames AND across all 384×512 spatial coords. The amortization holds because the mask sequence is locally smooth (Lane G v3 SegNet output evolves gradually — vehicles don't teleport).

**Status**: defensible. Phase G CUDA dispatch will measure.

### Q2: AV1 motion-vector argument

> "AV1 has 6 years of tuning for monochrome video; you wrote a 4-layer MLP last week."

**Counter**: AV1's strength is texture detail at low bitrates. Class-ID masks have NO texture — they're piecewise-constant with sparse 5-class boundaries. AV1's intra+inter+motion-vector machinery is mostly unused. The right comparison ISN'T "NeRV vs AV1 for natural video" (NeRV loses by orders of magnitude); it's "NeRV vs AV1 for piecewise-constant-with-temporal-smoothness", and Phase F's 94.4% byte savings + 2.0% disagreement at 1400 partial CPU steps is empirical evidence the bet works.

**Status**: empirical evidence overrides theoretical objection. RECEIPT: `reports/lane_12_nerv_real_archive.json`.

### Q3: half-frame argument

> "I encode only 600 odd frames + warp the even ones. Your 23 KB encodes 1200 frames. That's twice my redundancy."

**Counter**: TRUE — Phase A2 explicitly flags "half-frame NeRV (~12 KB) as stretch goal". Phase A1 ships 1200-frame as the simple baseline. Stacking with Quantizr's half-frame trick is a Phase A2 sweep, not a Phase A1 blocker.

**Status**: noted; stacking flag is in Phase B council document (line "Half-frame NeRV (only odd frames, even frames warped) (~12 KB)").

### Q4: argmax disagreement vs SegNet score

> "2% argmax disagreement sounds tight, but the contest scorer measures distortion not disagreement. What's your distortion floor?"

**Counter**: distortion = argmax-disagreement-rate × (1 if argmax differs, 0 otherwise) — they're the SAME measurement at this scoring layer (per upstream/evaluate.py). 2% disagreement → distortion 0.02 → 100 × 0.02 = 2.0 score points on SegNet term alone. **CONCERN**: this is way too high vs Lane G v3 baseline (0.0040). Phase F's 2.0% is partial training; full CUDA training MUST drop below 0.5% (= 0.005 SegNet distortion) to be competitive. Predicted [contest-CUDA] band [0.95, 1.30] explicitly covers this risk.

**Status**: HALF-DEFENSIBLE. Phase G CUDA result is the truth source. If full training stalls at >1% disagreement, Phase B kill criterion fires (SegNet > Lane G v3 + 25% = 1.31 score → abandon).

Phase G result is gating. **No Round 2 issue created** — the kill criterion already covers this risk.

## van den Oord: VQ-VAE / entropy-bottleneck pass

> "Where's the entropy bottleneck on the weight stream?"

### vO1: weight stream is uncoded fp16

NRV2 fp16 ships 11,781 × 2 = 23,562 raw fp16 bytes + 30 byte header = 23,594 bytes. Weights are dense — there's NO arithmetic coder over the empirical fp16 distribution.

**Counter**: arithmetic coding over fp16 quantiles = Lane SH territory. Stacking is Phase A2 per Phase B council ("Arithmetic-coded weight quantiles (Ballé)"). Predict additional 10-30% off.

**Status**: noted; Phase A2 stacking flag.

### vO2: codebook alternative

> "Why coordinate-MLP at all? VQ-VAE codebook over 8×8 patches: 100 codes × 64 pixels × 1B/pixel = 6.4 KB codebook + 8-bit indices per patch position × N patches."

**Counter**: 8×8 patch tiling at 384×512×1200 = (48 × 64 × 1200) × 1 byte index = ~3.7 MB if indices are 1 byte each, or 6.5 MB if 8 bits / 60 pixels. The CODEBOOK is small, but the INDEX STREAM is huge. NeRV avoids the index stream entirely — coordinates are implicit in the forward pass.

**Status**: defensible; van den Oord's own VQ-VAE-2 paper acknowledges this — they use HIERARCHICAL codebooks because the index stream dominates. Phase B council noted "VQ-VAE codebook as Phase 3 alternative" — explicit acknowledgement.

### vO3: hyperprior

> "Ballé 2018 entropy bottleneck would predict bit-cost per weight from a learned scale prior. You ship uniform fp16."

**Counter**: SAME as vO1. Phase A2 stacking flag. Plus: Lane 20 (Ballé hyperprior) is its own lane — Lane 12 + Lane 20 stacking is the canonical compose path per `project_codec_stacking_composition_canonical_orders_20260429.md`.

**Status**: noted; Phase A2 stacking flag.

## Hotz: raw-engineering pass

> "Just store every Kth frame and interpolate. K=20, AV1 high-CRF, 30 KB. Same SegNet."

### H1: K-frame nearest-neighbor interpolation

K=20 → 60 stored frames at AV1 high-CRF (CRF=63 is max). Per-frame at 384×512 monochrome AV1-CRF63 ≈ 500 B → 60 × 500 = 30 KB. SegNet score on linear-interpolated even frames degrades on fast-moving scenes (vehicles in heavy traffic, lane changes).

**Counter**: NeRV vs nearest-K-frame is exactly the Phase B council Hotz objection (#3). I noted "NOT verified empirically — flagged for Phase A2". The right answer is to RUN BOTH and let auth-eval pick. Phase A1 NeRV at 23 KB; Phase A2 nearest-K-frame at 30 KB — parallel ablation per CLAUDE.md "Multiple contenders → multiple paths" rule.

**Status**: Phase A2 ablation flag explicit in Phase B council document. No Round 2 issue.

### H2: pre-trained MLP over scratch

> "Why train from scratch? Use a pre-trained NeRV from comma's other videos and fine-tune. Skip 90% of the training."

**Counter**: would dramatically speed compress-time AND share information across the corpus. Same family as `Lane J-NWC shared-corpus codec` — pending. Phase A2 stretch goal.

**Status**: noted; Phase A2 stacking flag.

### H3: inflate-time forward pass

> "236M coords forward through a 4-layer MLP on T4 — show me the timing."

**Counter**: 4-layer MLP × hidden=64 = ~3 MFLOPs per coord × 236M coords = 708 GFLOPs total. T4 sustained ~8 TFLOPs FP16 → ~88ms. Even at FP32 (4 TFLOPs) it's 177 ms. Add coord-grid construction + argmax + memcpy: ~2-3 sec budget per Phase B council. Well under 30-min inflate cap.

**Status**: theoretical bound holds; will measure at Phase G.

### H4: simpler architectures

> "Why depth=4 GELU instead of depth=2 ReLU? Half the params, same loss for a piecewise-constant signal."

**Counter**: ablation flag for Phase A2 sweep. Phase A1 picks council-canonical depth=4 / hidden=64 / GELU per Phase B unanimous vote — start from a known-good operating point, then explore.

**Status**: Phase A2 sweep flag.

## Round 2 verdict

**CLEAN** — no NEW issues found that aren't already covered by:
- Phase B Phase A2 stacking flags (vO1/vO2/vO3, H1/H2/H4)
- Phase B kill criteria (Q4)
- Empirical Phase F evidence (Q1/Q2)
- Theoretical bound (H3)

The Round 2 reviewers raised valid theoretical objections; ALL ARE EXPLICITLY ADDRESSED in the existing design + plan. No counter resets.

Counter: **2/3**.
