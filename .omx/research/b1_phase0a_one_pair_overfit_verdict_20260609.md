# B1 Phase-0A one-pair RGB overfit — verdict (answers the burning question)

UTC 2026-06-09 · claude · `tools/hi_nerv_renderer_sanity_ladder.py one-pair-overfit`
(500ep, lr 1e-2, pure RGB-L2 on source pair 0, NO scorer/rate/QAT/sidecar/Muon).
Artifact: `.omx/research/hi_nerv_one_pair_rgb_overfit_20260609.json`. Frame pack:
`/Volumes/VertigoDataTier/pact/sanity_ladder_overfit_ep500/frame_pack/`. [macOS-CPU advisory].
Model = the EXACT B1 decoder (n_params=192963 = 228903 − 599 fewer pair-latents; decoder identical).

## Burning question (relayed):
## "which exact map first breaks: source decode → raw identity inflate → renderer RGB overfit →
##  uint8 write/read → SegNet frame1 preprocess — and once fixed, does a simple RGB base move
##  the score enough to justify PR95 scorer/rate fine-tuning?"

ANSWER — the first broken map is **renderer RGB overfit**, and it breaks in TWO ways:
1. **B1 never attempted RGB** (no RGB-pixel anchor; distillation-only) → renderer collapsed to 2 fixed
   latent-independent frames. The overfit CONFIRMS the renderer CAN move toward RGB (8→21 dB, latents
   + head receive gradient) — so B1's primary failure was the missing objective.
2. **Even WITH an RGB anchor the renderer PLATEAUS at 21.24 dB** (loss flatlined from ep123→499) — a
   229K decoder memorizing 2 frames should reach 30–40 dB trivially. AND **21 dB is BELOW SegNet's
   cell-entry threshold**: the rendered frame's SegNet hist is `[0,0,196608,0,0]` (100% class 2),
   identical to a black frame.

"Does a simple RGB base move the score?" → **NO, not at this architecture's 21 dB ceiling.** d_seg
stays 0.5069 — and the naive baselines prove it's a hard threshold, not a gradient:

| comp frame → SegNet | d_seg |
|---|---|
| black frame | 0.5069 |
| mean frame | 0.5069 |
| **rendered r1 @ 21 dB** | **0.5069** |  ← no better than black
| source frame0 used as last (real, sharp, adjacent) | 0.0084 |
| source identity | 0.0000 |

SegNet-cell entry is a FIDELITY THRESHOLD, not linear in PSNR. Below threshold, SegNet collapses to
the dominant class (d_seg = non-dominant fraction ≈ 0.50). A real sharp frame is far above (0.008).
PSNR-MSE is the WRONG currency for the seg term — exactly why PR95's curriculum is scorer-aware
(CE/margin/boundary-hinge on the SegNet argmax), not RGB-MSE.

## The two distinct blockers (revising the earlier "just add RGB anchor")
- **Blocker A — objective**: no RGB anchor (confirmed; the renderer learns RGB when given one).
- **Blocker B — architecture/conditioning**: 21 dB single-pair ceiling. Gradient table shows a severe
  coarse-explodes / fine-vanishes imbalance: latents_coarse=7.8e9, blocks.0=5.6e9, latent_embed=1.3e9
  EXPLODE; latents_fine=0.03, head_rgb≈1e-4, blocks.6=8e-5 VANISH. The fine-detail path that SegNet
  boundaries need gets ~no gradient. (AdamW normalizes per-param so it doesn't diverge, but the
  conditioning caps fidelity ≈21 dB — the huge upscale from a tiny grid to 874×1164 + the small
  per-pair fine latent likely limits high-freq detail.)

## What is RULED OUT (cheap rungs paid off)
- source decode, .raw write/read, uint8, SegNet frame1 preprocess, channel order, value range:
  ALL SOUND. source_identity d_seg = 0.0 (scorer path exact); inspection showed real 5-class source
  histograms; range [0,255] ✓; BGR-swap doesn't help. The break is purely renderer-side.

## Verdict + fix path (INSPECT/diagnose, NOT kill)
The HiNeRV lane is viable but needs renderer work BEFORE any scorer fine-tune. Ordered fix:
1. **RGB-reconstruction base** (necessary; add a dominant source-RGB-L2 term — it does not exist today).
2. **Fix the fidelity ceiling (Blocker B)** so the renderer crosses SegNet's cell-entry threshold:
   investigate the coarse-explodes/fine-vanishes imbalance — latent-injection scale, sin-activation,
   bilinear-upsample-from-tiny-grid, per-block normalization, fine-latent capacity. Target ≥30 dB on a
   single pair, then re-measure d_seg (does crossing the fidelity threshold drop d_seg?).
3. **Scorer-aware boundary training** (PR95 CE/margin/hinge) once RGB fidelity is adequate — RGB-MSE
   alone optimizes the wrong objective for the seg term.
4. Only THEN the full 600-pair PR95 curriculum + rate attack.

DO NOT run another long PR95 curriculum, tune optimizers, or add guard losses before Blocker B is
understood. The "RGB base then PR95 fine-tune" path is necessary but BLOCKED on the 21 dB ceiling.

## Parallel implication for the other vehicles
- **V2 SNeRV** may sidestep Blocker B (source-forward basis is more faithful than a tiny-latent NeRV
  decode) — worth prioritizing the SNeRV base in parallel.
- **V3 direct grammar** sidesteps BOTH blockers (encode the SegNet-argmax skeleton directly rather
  than hope a blurry RGB decode enters the cells) — the threshold finding STRENGTHENS the V3 case:
  the evaluator cares about argmax structure, not RGB fidelity, so encode the structure directly.
