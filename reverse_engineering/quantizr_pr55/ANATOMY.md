# Quantizr PR #55 ("quantizr (0.33)") — pose-FiLM anatomy (fetched from commaai/comma_video_compression_challenge)

Author: Jimmy (@Quantizr). MERGED 2026-04-19. Score 0.33 = seg 0.00061113 + pose 0.00051010 + rate 0.200 (299,970 B).
**RATE-DOMINATED** (0.200 of 0.33); distortion excellent + STABLE. ffmpeg masks.mkv (half the frames, higher CRF).

## The pose-FiLM (inflate.py) — the stability mechanism our Lever-3 lacks
- `pose_mlp = Linear(6→cond) → SiLU → Linear(cond→cond)` builds cond_emb from the 6 pose scalars.
- `FiLMSepResBlock.forward(x, cond)`: `x = norm2(conv2(conv1(x))); g,b = film_proj(cond).chunk(2); x = x*(1+g)+b; return act(residual + x)`.
  → FiLM modulates ONLY the residual branch; the identity path `residual` anchors the output (bounded perturbation even with unbounded g/b).
- FiLM lives in the **FrameHead** (frame-1 head), NOT the shared stem. d_seg comes from a SEPARATE `SharedMaskDecoder` (mask path). → pose-FiLM variance CANNOT leak into d_seg. FULLY DECOUPLED heads.
- Depthwise-separable convs (SepConv: dw groups=in_ch + pw 1x1), 88K params, FP4Codebook (nibble) quant, 64 KB.

## Contrast with our Lever-3 (src/tac/torch_vehicle/pose_film.py)
- Ours: single FiLM at the 6×8 STEM, `x = γ·x + β` (DIRECT modulation, no residual anchor), on the SHARED x feeding both frames + the seg-relevant render → max leverage + d_seg coupling → d_pose variance (best 0.00043, spikes 0.0046).
- Fix (Lever-3 v2): residual FiLM (`x = x + FiLM(x,cond)`) + later/higher-res injection + pose_mlp cond. Full head-decoupling = v3 (needs a separate mask path; our single-render HNeRV couples by construction).

## Strategic
Quantizr's 0.33 is rate (300 KB masks); OUR base_ch20 rate is already better (89 KB). Adopt their distortion STABILITY (residual+decoupled FiLM) onto our smaller-rate carrier.
