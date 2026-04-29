# logic/problem.md

## Gap definition

Video compression for autonomous-driving fleets is conventionally framed as
rate-distortion optimization on pixel fidelity (PSNR, SSIM, MS-SSIM). But the
*consumer* of compressed driving video is a stack of perception networks
(segmentation, pose estimation, drivable-area, lane detection). Pixel fidelity
is a poor proxy for those networks' loss landscapes.

The comma.ai video compression challenge makes this concrete: the public
scorer is

```
S = 100 * seg_distortion + sqrt(10 * pose_distortion) + 25 * archive_bytes_MB
```

evaluated on a frozen SegNet (EfficientNet-B2 U-Net, 5-class) and a frozen
PoseNet (FastViT-T12 trunk, 6-DOF head). Optimizing PSNR leaves substantial
points on the table because the scorer's marginal sensitivity is anisotropic
in pixel space.

## Key insights that informed our approach

1. **Anisotropic scorer sensitivity.** PoseNet's per-pair Jacobian
   `dPose/dPixel` has effective rank ~1.008/6 with condition number ~399.
   PoseNet's 6-DOF output is effectively 1-dimensional at the operating point
   (`evidence/jacobian/svd_2026-04-08.json`).
2. **Sub-pixel trust radius.** The same Jacobian's linear approximation
   breaks down at perturbations larger than ~1e-4 pixels RMS
   (`evidence/jacobian/trust_region_2026-04-08.json`). Closed-form
   pseudoinverse single-step methods are *mathematically* dead on arrival.
3. **Mid-frequency residual.** A trained learned post-filter places ~90.3% of
   its luma residual energy in the 4-32 cycles/frame mid-band — exactly the
   PoseNet sensitivity bandwidth (`evidence/cnn_residual/karpathy_2026-04-08.json`).
4. **SegNet leverage at the operating point.** At score `1.73`, the partial
   derivative `dS/d(seg) = 100` and `dS/d(pose) ~= 2.74`. SegNet is ~36x more
   leveraged than PoseNet on the marginal axis. By score `1.05`
   (Era 2 floor), the leverage ratio shifts but SegNet still dominates.
5. **MPS-vs-CUDA scorer drift (catastrophic).** PoseNet on Apple MPS produces
   pose distortion 23x larger than CUDA on identical artifacts. ALL "auth"
   scores prior to 2026-04-25 were measurement artifacts. The lab's first
   reproducible-from-saved-artifacts contest-CUDA score is `0.90`, not the
   `2.26` that prior MPS readings suggested.

## What we built

Two distinct paradigms, in chronological order:

- **Era 1 (codec + post-filter, 4.06 -> 1.73 [pre-MPS-fix advisory]).**
  SVT-AV1 with CRF 34 and film-grain synthesis, plus a 3-layer residual CNN
  trained against frozen scorer gradients with QAT (FakeQuant STE), EMA
  weights (decay=0.997), and best-checkpoint int8 selection. Width scaling
  follows a log-linear law: `score = -0.159 ln(h) + 2.382`.

- **Era 2 (neural renderer, 1.73 -> 1.05 [contest-CUDA]).** Abandon the codec.
  A 287K-param dilated-h64 renderer (`AsymmetricPairGenerator`) takes per-pair
  6-DOF embeddings + low-resolution mask channels and emits 384x512 frames.
  AV1 only carries the masks. Compress-time pose TTO warm-started from
  baseline poses (Lane A, 1.15) followed by a small KL-distill nudge on
  SegNet logits at weight=0.002 (Lane G v3, 1.05).

- **Era 3 (Selfcomp paradigm portfolio, live).** Eight Modal lanes
  reverse-engineered from the public Selfcomp 0.38 entry: grayscale-LUT
  masks, single-mask-per-pair + 6-DOF affine duality, analytical pose,
  block-FP self-compression at ~1.017 bpw, and a 94K-param SegMap. NOT
  reported until [contest-CUDA] verified.

## What this artifact does not claim

- We do not claim to beat Quantizr 0.33 or Selfcomp 0.38. Our public floor
  is `1.05`; live work targets sub-0.5 but those numbers are advisory only.
- We do not claim that Era 1 numbers prior to 2026-04-25 are contest-CUDA
  valid. They are explicitly tagged `[MPS-PROXY]` and treated as advisory.

## Forensic bindings

- Methodology rules: `../../../CLAUDE.md`
- Pinned upstream snapshot: `../../../upstream/`
- Era 1 promoted-floor evidence: `../evidence/era1/long1000_h64_authoritative/`
- Era 2 contest-CUDA evidence: `../evidence/era2/lane_g_v3_landed/`
- Detailed claims: `claims.md`
- Verification plan: `experiments.md`
