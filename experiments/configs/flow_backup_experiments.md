# Flow Backup Experiments — Council Pre-Registration
## Two contingency plans if flow warmup fails

### Decision gate
Check `flow_magnitude` in the fresh Modal run (asym_v2_fresh_bs4) at:
- Epoch 200 (~20 min): flow_magnitude > 0.05 → warmup working
- Epoch 500 (~45 min): flow_magnitude > 0.1 → flow established
- Epoch 1000 (~90 min): gate_mean < 0.5 → flow surviving ramp

If flow_magnitude < 0.01 at epoch 500: warmup has failed. Deploy one of these.

---

## Experiment A: Quantizr-Fridrich Flow Initialization

**Concept:** Precompute optical flow at compress time using the GT video.
Use this precomputed flow to initialize the MotionPredictor's output layer
bias, so training starts from the RIGHT answer instead of random.

**Architecture change:**
- MotionPredictor unchanged
- New function: `initialize_motion_from_precomputed_flow(motion_predictor, precomputed_flow)`
  - Sets `motion_predictor.head.bias` to the mean precomputed flow per output channel
  - Sets `motion_predictor.head.weight` to a small value (0.01) so the network starts
    near the precomputed flow and fine-tunes from there

**Compress-time computation:**
```python
# For each pair (t, t1), compute dense optical flow using RAFT or Lucas-Kanade
# on the GT RGB frames. Store as (600, 2, 384, 512) float16 tensor.
# Compressed with zlib: ~50KB for 600 pairs.
```

**Rate impact:** ~50KB compressed flow / 37.5MB GT = 0.0013 rate term = 0.033 score
(compared to current 0.100 from 150KB renderer weights)

Wait — this is stored as initialization only. The flow is NOT in the archive.
The MotionPredictor learns to reproduce it during training. At inference,
the MotionPredictor runs normally. No rate impact.

**Training command:**
```bash
python experiments/train_renderer_fridrich.py \
  --pair-mode asymmetric \
  --precomputed-flow experiments/precomputed_flow.pt \
  --flow-warmup-epochs 0 \  # no warmup needed — flow starts correct
  ... (all other council-approved params)
```

**Implementation cost:** ~4 hours (flow computation + initialization function + CLI arg)
**GPU cost:** Same as current training (~$3.25 on Modal T4)
**Risk:** Precomputed flow is from GT-to-GT. The renderer generates from masks,
not GT. The flow learned from GT may not match the flow needed for mask-to-mask
generation. The MotionPredictor needs to adapt from GT-flow to rendered-flow.

**Council question:** Should we store the flow in the archive (50KB rate cost)
or use it only as initialization (0 rate cost but the network must memorize it)?

---

## Experiment B: Hotz Analytical Ego-Flow (Eliminate Motion Predictor)

**Concept:** Compute flow analytically from PoseNet's 6-DOF ego-motion output
and the camera intrinsics. The flow is deterministic and exact for the ego-motion
component. Store 7200 bytes (600 pairs × 6 floats × 2 bytes). At inflate time,
compute dense flow from the stored parameters using the pinhole camera model.

**Architecture change:**
- REMOVE MotionPredictor entirely (saves 30K params = 15KB in archive)
- Add `AnalyticalEgoFlow` module:
  ```python
  class AnalyticalEgoFlow(nn.Module):
      def __init__(self, fx, fy, cx, cy, H, W):
          # Camera intrinsics (known: fx=910, pp=(582,437) at 1164x874)
          # Scaled to scorer resolution: fx=400.3, pp=(256,187) at 512x384
          ...
      
      def forward(self, ego_params):
          # ego_params: (B, 6) — tx,ty,tz,rx,ry,rz from stored targets
          # Returns: (B, 2, H, W) dense flow field from pinhole projection
          # Assumes flat-world depth (z=1 everywhere) as first approximation
          ...
  ```

- AsymmetricPairGenerator becomes:
  ```python
  frame_t1 = renderer(mask_t1)
  flow = analytical_ego_flow(stored_ego_params[pair_idx])
  frame_t = (warp(frame_t1, flow) + gate * residual).clamp(0, 255)
  ```

**Rate impact:**
- Current: 150KB (renderer 125K + motion 30K at FP4)
- New: 135KB (renderer 125K + ego_params 7.2KB + gate/residual ~3K)
- Net: saves ~15KB → rate drops from 0.100 to 0.090 → 0.25 score improvement

**Inflate-time:**
- Load ego_params from archive (7.2KB)
- For each pair, compute dense flow analytically (no neural network needed)
- Apply warp + gate * residual as before

**Training command:**
```bash
python experiments/train_renderer_fridrich.py \
  --pair-mode asymmetric \
  --use-analytical-flow \
  --ego-motion-path experiments/posenet_targets.bin \
  --flow-warmup-epochs 0 \  # no warmup — flow is precomputed
  ... (all other council-approved params)
```

**Implementation cost:** ~8 hours (analytical flow module + training integration +
inflate integration + archive format update)
**GPU cost:** Same as current training
**Risk:** Flat-world depth assumption is wrong for close objects (cars, barriers).
The analytical flow is correct for the ego-motion component but misses parallax
from nearby objects. The gate + residual must compensate for parallax errors.

**Advantage over Experiment A:** No MotionPredictor to train. Flow is exact
(for the ego-motion component). Smaller archive. Deterministic at inflate time.

**Disadvantage vs Experiment A:** Cannot model object-level motion (other cars
moving independently). Only models ego-motion. Close objects have wrong flow.

---

## Council Recommendation

**If flow warmup works (flow_magnitude > 0.05 at epoch 500):**
- Continue with current approach. No backup needed.

**If flow warmup partially works (0.01-0.05 at epoch 500):**
- Try Experiment A first (lower implementation cost, preserves architecture).

**If flow warmup completely fails (< 0.01 at epoch 500):**
- Implement Experiment B (Hotz approach, eliminates the problem entirely).

**The two experiments are complementary, not competing:**
- A preserves the learned motion predictor (can model all motion types)
- B is simpler and cheaper but limited to ego-motion only
- In theory, B's analytical flow + A's fine-tuned MotionPredictor could be combined
