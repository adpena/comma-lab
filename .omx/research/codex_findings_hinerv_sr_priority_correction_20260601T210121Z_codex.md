# Codex Findings - HiNeRV / SR-NeRV Priority Correction

UTC: 2026-06-01T21:01:21Z
Agent: Codex
Scope: score-lowering carrier roadmap correction after HiNeRV advisory review

## Authority

- HiNeRV artifacts reviewed here are `[macOS-CPU advisory]`.
- SR-NeRV mirror artifacts reviewed here are local preprocessing mirrors only.
- `score_claim=false`, `promotion_eligible=false`, and no exact CPU/CUDA claim.
- Receiver-proven archive bytes plus exact CPU/CUDA remain the only promotion
  surface.

## Corrected HiNeRV Verdict

The partner-agent correction is right.

True and independently corroborated:

- G3 dense decoder adjoint is real: HiNeRV latent VJP dot-product residuals are
  machine-close in the landed advisory (`coarse=0.0`, `mid=4.82e-7`,
  `fine=2.07e-7` in the cited smoke; other advisory rows remain around
  `1e-6`). The "HiNeRV re-opens G3" risk is closed for the local analysis path.
- Latent leverage is tiny at the measured local operating point: observed JVP
  norms are about `1e-4` to `1e-7`, so post-hoc per-pair latent allocation is
  second-order. Score bits mostly live in decoder-weight training and QAT.
- `--modelsize` / small model configuration is a real byte knob. HiNeRV can be
  made cheap by construction in a way explicit Z8/HPRC coefficient storage
  cannot.

Overstated and corrected:

- The "36 KB / 5x cheaper" line is a structural projection, not a current
  receiver-proven 600-pair score candidate. The measured advisory archives I
  inspected are local smokes: about `40,491 B` for 2 pairs, about `62 KB` or
  `115 KB` for 6 pairs depending on config, and about `60.7 KB` for 8 pairs.
- Cheap alone is not a score win. The best fixed smoke cited by the local
  findings had `d_seg=0.508`, `d_pose=176.51`, and advisory `S=92.84`, worse
  than the already-bad Z8 advisory comparison. L-inf latent allocation made
  distortion worse than L2 in that fixed smoke because the latents had too
  little leverage.

One-line corrected headline:

**HiNeRV has a structurally interesting rate knob, but the current local
carrier is catastrophically unfit. The next score-lowering work is score-aware
decoder-weight training and coder-aware QAT, not latent polishing.**

## Optimizer / QAT Implication

Optimizer and QAT are first-class rate levers, not final polish:

- PR95-style coder-aware regularization biases decoder weights toward
  compressible byte distributions before Brotli/range coding ever see them.
- Quant-noise / sigma-noise / learned per-group quantization are both fit and
  rate mechanisms: they train the decoder to preserve SegNet/PoseNet under the
  exact quantized weight grammar that will be charged in `archive.zip`.
- Muon or other orthogonalized late optimizers are therefore part of the
  score-lowering stack only if they improve scorer-faithful fit under the byte
  grammar, not because they lower MSE.

Production consequence:

1. PR95 remains the baseline to beat and the control grammar.
2. HiNeRV and SNeRV deserve their own fully optimized stacks because they are
   structurally plausible carriers, but neither can promote until trained
   artifacts, charged archive bytes, receiver proof, and full-video replay pass.
3. The allocator should target decoder weights and learned weight-code grammar
   first, with per-pair latent tweaks reserved for carriers whose JVP norms prove
   latent leverage.

## SR-NeRV Mirror Result

I added a local mirror gate for the resolution-axis hypothesis:

`tools/check_sr_nerv_resolution_axis_mirror.py`

It checks:

`camera RGB -> low internal resolution -> legal 1164x874 output -> scorer resize`

against direct scorer preprocessing. It is false-authority by construction and
only decides whether an SR-style enhancer is worth training.

8-pair real-video artifacts:

- `/Volumes/VertigoDataTier/pact/sr_nerv_resolution_axis_mirror_20260601T205924Z/sr_nerv_resolution_axis_mirror.json`
- `/Volumes/VertigoDataTier/pact/sr_nerv_resolution_axis_mirror_sweep_20260601T210008Z/`

Result:

- Naive `512x384 -> bilinear -> 1164x874 -> scorer` fails the mirror gate:
  SegNet mean drift `0.419`, max drift `45.77`; PoseNet mean drift `0.231`,
  max drift `45.13`.
- Bicubic at `512x384` improves mean drift (`SegNet=0.162`, `PoseNet=0.091`)
  but still has large max spikes (`~18-19`).
- Larger bilinear internal grids did not fix the issue in the 8-pair probe.

Corrected SR headline:

**The resolution-axis dead-zone is still promising, but naive low-res/SR is not
safe enough. SR-NeRV must train with a scorer-preprocess mirror loss or an
exactly matched synthesis kernel, then prove full-video SegNet/PoseNet drift.**

## Queue Consequence

Priority order for score-lowering work:

1. Finish active PACT-NeRV-VQ full600 training/export and price every section by
   full-video scorer value per byte.
2. Run PR95/Stage-8 control and treat it as the byte/fit baseline to beat.
3. Build HiNeRV and SNeRV MLX/portable score-aware trainers with coder-aware QAT
   in the loop; train decoder weights under the charged packet grammar.
4. Add SR only as a trained enhancer after the mirror loss/kernels fix the
   observed preprocessing drift.
5. Use residual/Z8/HPRC tokens only where `delta_nonrate + rate_cost < 0` after
   receiver-proofed replay.

No current HiNeRV or SR-NeRV advisory row is promotable.
