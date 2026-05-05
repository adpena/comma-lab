---
name: Chained Training (Single-Pass Multi-Pass)
description: Train with double forward pass, deploy with single pass. Gets multi-pass quality within time budget.
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## The Idea

Multi-pass inflate (run CNN twice) may improve score but takes ~14 min on M5 Max,
likely exceeding the 30-min contest time limit.

**Solution**: Train with chained loss — apply filter twice during training, compute
loss on the twice-filtered output. At inference, run the filter ONCE. The model
learns deeper corrections within the same architecture and parameter budget.

## Implementation

In fit_lazy, after `_apply_filter_to_pair`:
```python
filtered = self._apply_filter_to_pair(comp_pair)
if cfg.chained_training:
    # Round to uint8 between passes (match inference distribution)
    filtered = filtered.round().clamp(0, 255).to(torch.uint8).float()
    filtered = self._apply_filter_to_pair(filtered)
```

Add `chained_training: bool = Field(False)` to TrainConfig.
At inference, INFLATE_MULTI_PASS stays at 1 (single pass).

## Expected Impact

The model learns corrections calibrated for two applications — which means its
single-pass correction is "deeper." This was a council recommendation from the
shower thoughts session. Doubles per-epoch training compute.

## Queue Priority

After PSD+standard results. If PSD works with standard loss, test:
PSD + standard + chained = the ultimate combination.

## Timing

Training: 2x slower per epoch (two forward+backward passes)
Inference: SAME as single pass (stays within 30-min limit)
