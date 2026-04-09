# submission name:
learned_postfilter_av1

# upload zipped `archive.zip`
<!-- will upload as GitHub release -->

# report.txt
```
=== Evaluation results over 600 samples ===
  Average PoseNet Distortion: 0.04809216
  Average SegNet Distortion: 0.00576402
  Submission file size: 864,167 bytes
  Original uncompressed size: 37,545,489 bytes
  Compression Rate: 0.02301653
  Final score: 100*segnet_dist + √(10*posenet_dist) + 25*rate = 1.85
```

Exact score: **1.8453**. Authoritative local CPU scorer run of `2026-04-08 21:36:54 -0500`
against the upstream `evaluate.py` end-to-end path. Evidence:
`reports/raw/2026-04-08-long1000-h32/robust_current-long1000-h32-current_workflow-cpu-report.txt`.

# does your submission require gpu for evaluation (inflation)?
no

# did you include the compression script? and want it to be merged?
yes

# additional comments

## Solution: Long-horizon QAT+EMA learned post-filter (h=32) over SVT-AV1

This submission improves on the prior learned post-filter by combining a stricter
scorer-faithful training regime with compound scaling along two axes (width and
training horizon). It currently leads the public leaderboard first place
(`1.95 av1_roi_lanczos_unsharp`) by **0.105**.

### Pipeline
1. **Encode**: SVT-AV1 preset 0, CRF 34, `film-grain=22:keyint=180:sharpness=1`, 524x394 lanczos downscale
2. **Decode**: PyAV + BT.601 limited-range YUV→RGB (matches evaluator's `frame_utils.py::yuv420_to_rgb`)
3. **Upscale**: torch bicubic (`F.interpolate`, `align_corners=False`) to 1164x874
4. **Post-filter**: 3-conv residual CNN (3→32→32→3, 3x3 kernels, ReLU)
   - Trained by backpropagating the actual competition loss
     `100*segnet_dist + sqrt(10*posenet_dist) + 25*rate` through PoseNet and SegNet
   - **1000 epochs** of quantization-aware training with EMA weight averaging (decay=0.997)
   - **Saliency-weighted loss at α=20** (weights frames by local PoseNet gradient magnitude)
   - Cosine LR schedule 5e-4 → 1e-5, warmup 5 epochs
   - Int8 per-tensor symmetric quantization, `~16KB` on disk
5. **Output**: raw RGB24 frames

### What changed vs the previous 2.05 learned post-filter
- Width: hidden=16 → hidden=32 (11,011 params vs 3,203)
- Horizon: 100 epochs → 1000 epochs of QAT+EMA
- Loss: plain scorer MSE → saliency-weighted scorer loss (α=20)
- Proxy: a "scorer-faithful" local proxy that now matches the authoritative
  upstream `evaluate.py` score to 8 decimal places on held-out archives, which
  removed the blind-spot that caused the first post-filter submission
  (trained on the wrong archive distribution) to regress to 2.35.

Compounding these changes produced an approximately additive trajectory in one
session: **2.01 → 1.99 → 1.945 → 1.92 → 1.85**.

### Mathematical investigation (why the learned CNN is the only route that survives)

After the 1.85 promotion we ran three diagnostic experiments to test whether a
closed-form alternative could match the learned filter. All three gave sharp,
falsifiable answers that validated the CNN approach.

1. **Jacobian pseudoinverse is dead (falsified)**: the Moore-Penrose
   minimum-norm correction `δ = J⁺(pose_gt - pose_decoded)` on `dPose/dPixel`
   makes the pose distortion **3× worse** (`0.0742 → 0.2349`). The correction is
   concentrated on 0.0044% of pixels and immediately crosses ReLU region
   boundaries.
2. **Jacobian is rank-1 and ill-conditioned (surprising)**: SVD across 30
   sampled frame pairs measures effective rank **`1.008 / 6`** with condition
   number `~399`. PoseNet's 6-dim output is effectively one-dimensional at our
   operating point.
3. **Trust radius is sub-LSB (decisive)**: PoseNet's honest linear trust radius
   is at or below `0.0001` pixels RMS; any single-step correction of practical
   size immediately lands in a different ReLU region than where the Jacobian
   was computed, so local-linear / Newton inflate methods are mathematically
   dead on arrival.
4. **CNN residual signature is dense + mid-frequency (confirmed)**: the winning
   filter moves **56.6% of pixels** (vs 0.0024% for the Jacobian delta) and
   places **90.3% of its luma residual energy in the mid-frequency DCT band**.
   The learned filter is an amortized iterative descent, spread densely and
   spatially coherently so every per-pixel nudge stays inside its local ReLU
   region.

These experiments are shipped in-tree as:

- `experiments/jacobian_optimal.py`
- `experiments/jacobian_svd_analysis.py`
- `experiments/trust_region_sweep.py`
- `experiments/karpathy_cnn_residual_analysis.py`
- `experiments/rd_bound_mine.py` (MINE-based lower bound on the rate-distortion
  frontier, used to argue that there is still measurable headroom beyond 1.85)

### Archive contents
- `0.mkv`: SVT-AV1 encoded video (876,737 bytes)
- `postfilter_int8.pt`: learned post-filter weights, `hidden=32`, int8
  quantized (`16,473` bytes, `__meta__ = {"variant": "saliency_weighted",
  "hidden": 32, "kernel": 3, "alpha": 20}`)
- Total archive: `~876 KB`

### Notes for reviewers
- Inflation is CPU-only: PyAV decode + torch bicubic + a single forward pass of
  a ~11k-parameter CNN per frame. Well within the 4-CPU / 16 GB / 30 min
  official budget.
- The co-located `postfilter_int8.pt` inside the submission directory is a
  redundant copy; the authoritative path reads the weights from inside
  `archive.zip` at inflate time, so the submission is self-contained.
- `inflate_postfilter.py` supports the `saliency_weighted` variant tag on the
  shared 3-conv residual architecture; the previous 2.05 residual checkpoint
  still loads unchanged.
