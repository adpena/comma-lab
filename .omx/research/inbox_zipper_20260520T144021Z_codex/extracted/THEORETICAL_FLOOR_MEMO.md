# Theoretical Floor Memo

This memo articulates a working hypothesis for the lower bound of the contest
score and identifies where uncertainty lies.  It is intended as a planning
reference rather than a definitive statement.  All estimates should be
periodically revisited as new evidence arrives.

## Current best scores

* **PR #101 (gold baseline):** 0.192845 (contest‑CPU)【294212394795766†L355-L357】
  with one zip member and microcodec.
* **PR #110 (FEC6 + fixed‑Huffman):** 0.192051 (contest‑CPU) and 0.226210
  (contest‑CUDA)【294212394795766†L354-L357】.  This is a 0.000794 absolute
  improvement at the cost of 259 extra bytes.

Publicly reported submissions cluster in the 0.192–0.193 band on CPU and
0.226–0.227 on CUDA.  There is no evidence of a submission achieving a score
below 0.192 on CPU at this time.

## Simplified score decomposition

Recall the score formula:

\[\text{score} = 100 \times \text{seg\_distortion} + \sqrt{10 \times \text{pose\_distortion}} + 25 \times \frac{\text{archive\_bytes}}{37\,545\,489}\]

Under typical ranges, the segmentation term dominates (order 0.18–0.19), the
pose term contributes ~0.03–0.04, and the archive term contributes ~0.005–0.01.
Removing 259 bytes from a 178 kB archive changes the score by ~0.00017【294212394795766†L294-L295】.
This suggests that to reduce the score by ≥0.001 you need either a large
reduction in segmentation error or a substantially smaller archive (≥~1.5 kB
savings).

## Lower‑bound scenarios

1. **Perfect segmentation and pose:** In the impossible limit where
   segmentation and pose distortions are zero, the score reduces to the
   archive term alone: \(25\times \text{archive\_bytes}/37\,545\,489\).  To reach
   0.18, one would need an archive of ~270 kB.  Our current archive is 178 kB
   and yields a 0.0001–0.0002 contribution.  Thus, further archive savings
   provide diminishing returns and cannot alone reach much below 0.19.

2. **Zero archive bytes:** If the archive were free, the score would be
   dominated by the distortion terms.  Current segmentation error
   contributes ~0.18; halving it would drop the score by ~0.09.  Achieving a
   50% reduction in segmentation distortion may require radical changes in
   representation and priors (e.g. neural codecs, VQ‑VAE, semantics).  This
   defines an optimistic lower bound near 0.10.

3. **Realistic near‑term improvements:** With existing architectures, a
   5–10% reduction in segmentation error seems plausible through better
   selectors, quantization, pretraining and foveation.  That would reduce the
   score by 0.009–0.018.  Combined with modest archive savings and pose
   improvements, the near‑term floor may lie around 0.180–0.185.  Hitting
   0.18 would require halving segmentation and pose distortions while keeping
   the archive under 200 kB.

4. **Speculative innovations:** If new representations (e.g. SIREN, VQ‑VAE,
   C3) and priors (DreamerV3, V‑JEPA) drastically improve downstream metrics,
   a score around 0.15 might be reachable.  This would require both
   near‑perfect segmentation and pose predictions and efficient entropy
   coders.  Such a leap likely demands months of research and significant
   compute.

## Saturated components and open questions

* **Archive entropy:** Tests on the PR #95–#110 lineage show no meaningful
  reduction in score when recompressing the exported archive bytes with
  general‑purpose compressors【294212394795766†L354-L361】.  This suggests that,
  for the HNeRV family, the archive is near entropy saturation.  Significant
  gains require changing the emitted representation or the entropy model.

* **SegNet sensitivity:** It is unclear whether the segmentation distortion
  term has headroom.  The `tac` post‑filter architecture is designed to
  correct frame artifacts【635462165059268†L147-L163】; similar ideas could be
  integrated into HNeRV variants.  Measuring how segmentation error responds
  to small architectural changes is essential.

* **PoseNet error:** Pose distortion contributes the square root term.  RAFT
  and LA‑Pose priors may reduce this error【58280996536521†L104-L115】.  The
  magnitude of potential improvements is unknown.  Empirical tests are needed.

* **Interaction effects:** Combining multiple innovations may yield
  super‑additive or antagonistic effects.  Without systematic ADMM-style
  composition【58280996536521†L207-L213】, it is difficult to predict the floor.

## Experiments to reduce uncertainty

1. **Run the HNeRV variant sweep (S3) and compute component distances.**
   Understand how segmentation and pose errors change across PR #95–#110.
2. **Byte profiling (S4) and master‑gradient analysis.**  Quantify which
   bytes matter and whether small modifications to selector bits can reduce
   distortion.
3. **Foveation and RAFT smokes (M6, M8).**  These directly target
   segmentation and pose; measure how much headroom exists.
4. **SIREN and VQ‑VAE experiments (M3, M4).**  These change the
   representation and may produce lower‑entropy payloads.  Determine whether
   they outperform HNeRV on small clips.
5. **Predictor calibration (M7).**  Provides better estimates of candidate
   potential, reducing wasted GPU time and focusing resources on promising
   lanes.

## Discussion of alternative views and conclusion

Some researchers in the community believe the true floor could be much lower
than 0.18.  Speculative estimates range from **sub‑0.15** down to **0.1** or
even lower.  These optimistic bounds typically assume that multiple
innovations – neural codecs (e.g. C3, DCVC‑FM), predictive world models
(DreamerV3/V‑JEPA), advanced priors (open‑vocabulary dense segmentation) and
optimised entropy coding – can be combined without interfering.  There is
little empirical evidence yet to support such aggressive targets.  Our
assessment deliberately errs on the conservative side: it anchors the floor
to existing architectures and single‑module improvements.  Nonetheless,
the exploration lanes in the frontier staircase are broad enough to
accommodate the ambitious view; if multiple lanes succeed and compose well,
the score could plausibly approach or fall below 0.15.  A truly sub‑0.1
score would require near‑perfect segmentation and pose predictions plus an
exceedingly compact representation – a feat that would likely constitute a
breakthrough in task‑aware video compression.

In conclusion, we view **0.18–0.185** as a realistic near‑term target and
**0.15** as an optimistic but plausible medium‑term goal.  Hitting **0.1**
would demand innovations across almost every layer of the stack.  The
roadmap provided in the staircase is designed to test these possibilities
systematically.
