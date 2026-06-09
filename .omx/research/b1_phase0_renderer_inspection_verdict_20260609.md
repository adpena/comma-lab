# B1 Phase-0 renderer inspection — DECISIVE verdict (the first broken map)

UTC 2026-06-09 · claude · `tools/hi_nerv_renderer_sanity_ladder.py inspect-renderer-frames`
on the COMPLETED ep3000 archive (`b1_229k_clean_20260609T085348Z/harvest_export_ep3000/archive.zip`).
Artifact: `.omx/research/hi_nerv_renderer_frame_inspection_ep3000_20260609.json`.
Axis: [macOS-CPU advisory] (the scorer math is the EXACT upstream DistortionNet path).

## FIRST BROKEN MAP: source → renderer (the renderer did NOT learn the video)

The chain `source → renderer → uint8/inflate → scorer` breaks at the FIRST map. The
range/channel/inflate/scorer maps are all SOUND; the renderer never learned.

### Evidence (real inflate of the real archive, real DistortionNet)
- **comp_max = 255.0, looks_0_1_range = FALSE** → range map OK (NOT a [0,1] cast bug).
- **bgr_swap_would_help = FALSE** → channel-order map OK (NOT RGB/BGR).
- **frames are NOT near-constant** (std ~106–125) → the renderer outputs *structured* frames.
- **but mean RGB PSNR = 5.48 dB** (good ≥30; this is ~unrelated to source).
- **SegNet argmax COLLAPSES to one class**: renderer frame hist = `[0,0,196608,0,0]` (100% class 2)
  vs source hist `[44k,1.4k,97k,3.6k,51k]` (real 5-class). ⇒ `d_seg = (196608−96943)/196608 = 0.507`
  — MECHANICALLY exactly the flat 0.50.

### The smoking gun: the renderer IGNORES the per-pair latents
Renderer EVEN frames (0,300,600,1198) are all ~identical (mean~61, ch~[63,48,71]); ODD frames
(1,601,1199) all ~identical (mean~132, ch~[96,148,153]). **The renderer emits TWO FIXED frames
regardless of pair index** — the per-pair 28-d latents (PR95's core DOF) are dead. d_seg/d_pose are
~constant across pairs 0/150/300/599 because the OUTPUT is ~constant across pairs.

## Root cause: NO RGB-pixel reconstruction anchor (objective-too-hard)
The trainer argparse has NO raw-RGB-reconstruction loss against the decoded source video. It has
only SCORER-domain terms: `--distillation-weight` (SegNet-KL), `--pose-distillation-weight`
(PoseNet-MSE), `--scorer-domain-bootstrap-rgb-weight`, `segnet_direct_live_class_region_recon_weight`
(scorer-feature/class-region recon, NOT source-pixel recon). The architecture HAS per-pair latents +
`LatentInjector` + `head_rgb` — so the latents EXIST but training never rewarded per-pair RGB
fidelity, so the renderer collapsed into a degenerate latent-independent minimum (2 fixed frames) that
the weak frozen-scorer distillation gradients could not escape. This is the canonical NeRV failure:
a NeRV must MEMORIZE the video via RGB reconstruction FIRST, then scorer-aware fine-tune.

## Hypothesis disposition (vs operator's probability split)
- 35% raw/value/frame/channel/scorer mismatch → **RULED OUT** (range+channel+scorer-path all sound;
  source histograms are real 5-class).
- 10% objective-too-hard without RGB base → **CONFIRMED PRIMARY** (no source-RGB anchor exists).
- 25% renderer/gradient connectivity → **SECONDARY/COUPLED** (latents are dead — but likely because
  the objective never drove them; the one-pair RGB overfit + per-group gradient table separates
  "dead by objective" from "dead by wiring").
- 20% architecture/capacity → tested by 1-pair→16-pair→600-pair overfit ladder.
- 5% export-reads-wrong-state, 5% optimizer → ruled out (output varies by frame-position; stable run).

## Verdict + next (Phase 0A, the operator's burning-question half 2)
INSPECT → diagnose, NOT kill. The decisive separator: **one-pair RGB overfit** (pure RGB L2 on pair 0,
no scorer/rate/QAT/sidecar/Muon) + per-param-group gradient table + a visual frame pack + naive
baselines (black/mean/last-frame-copy d_seg) + frame-index ablation (SegNet reads the LAST frame).
- If 1-pair RGB overfit SUCCEEDS (PSNR↑, latents+head get gradient, d_seg↓): the renderer CAN learn;
  B1's failure was the missing RGB anchor → fix = RGB-recon BASE then PR95 scorer/rate fine-tune.
  (Answers "does a simple RGB base move the score": yes.)
- If it FAILS (PSNR flat, latent/head gradient ~0): deeper latent-injection/gradient/architecture bug
  → fix the renderer wiring before any training.
DO NOT run another long PR95 curriculum, tune optimizers, or add guard losses before Phase 0A.

## Sister: identity baseline (Phase -1) — scorer-path formal confirmation
The inspection already validates the scorer path implicitly (real 5-class source histograms via the
exact DistortionNet). The full `identity-baseline` (source→.raw→evaluate.py, expect d_seg≈0) is the
belt-and-suspenders formal proof; running it confirms the d_seg=0.50 is a real renderer failure, not
a pipeline artifact.
