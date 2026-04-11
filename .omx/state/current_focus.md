# Current Focus — 2026-04-10T23:59:00-05:00

## Floor
- **Official score**: 1.33
- **Variant**: dilated_h64
- **Platform**: modal_a10g
- **Epoch**: 905

## Two-Lane Strategy

### CPU Lane (postfilter)
- CRF 35 retrain: running, epoch ~110, current score 1.54
- CRF 36 retrain: running, epoch ~110, current score 1.95
- Both need ~300+ epochs before auth eval is meaningful
- CRF 35 auth eval of old filter: 2.08 (confirms CRF-specific distribution shift)

### GPU Lane (mask renderer)
- Mask renderer: epoch 0, score ~90 (very early, expected)
- MPS optimizations (P0-P4) identified, not yet implemented
- Manual grid_sample: 11.3x speedup over CPU fallback demonstrated
- Next: implement MPS P0-P4, then resume renderer training at speed
- After local MPS works: deploy to Modal A10G for real training

## Competitive Threat
- mask2mask (PR#53 by Quantizr): score 0.60
- Not yet verified by organizers
- Architecture: segment -> compress masks -> neural render (U-Net 36->60->36, FP4, 386KB)
- Validates GPU-lane strategy; we need to execute fast

## Priority Order
1. MPS P0-P4 optimizations (unblock GPU lane training speed)
2. Resume renderer training with MPS optimizations
3. Monitor CRF 35/36 retrains (CPU lane)
4. Auth eval CRF retrains at epoch 300+
5. Modal A10G deployment for renderer (after MPS works locally)
