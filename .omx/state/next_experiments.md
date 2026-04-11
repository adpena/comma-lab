# next experiments

## promoted floor: 1.33 (dilated_h64)

## two-lane strategy

### CPU Lane (postfilter retrains)

| Experiment | Status | Current Score | Target |
|-----------|--------|---------------|--------|
| CRF 35 retrain | running, epoch ~110 | 1.54 | sub-1.33 at epoch 300+ |
| CRF 36 retrain | running, epoch ~110 | 1.95 | sub-1.33 at epoch 300+ |

- Auth eval when epochs reach 300+ (earlier is meaningless given convergence curve)
- If CRF 35 or 36 retrain beats 1.33, promote as new floor
- CRF 36 offers additional rate savings if filter quality matches

### GPU Lane (mask renderer)

| Experiment | Status | Current Score | Target |
|-----------|--------|---------------|--------|
| Mask renderer smoke test | running, epoch 0 | ~90 | sub-1.00 |
| MPS P0-P4 optimizations | next up | N/A | unblock training speed |
| Modal A10G deployment | after MPS works | N/A | real training at scale |
| CLADE/DP-SIMS architecture | queued | N/A | improved synthesis quality |

Priority sequence:
1. **P0**: MPS grid_sample (11.3x speedup already demonstrated)
2. **P1**: MPS batch normalization
3. **P2**: MPS upsampling
4. **P3**: Training loop profiling and bottleneck elimination
5. **P4**: GT scorer cache (40-50% training time savings)
6. Resume renderer training with all optimizations
7. Deploy to Modal A10G once local MPS pipeline validated
8. Explore DP-SIMS (CVPR 2024) architecture for synthesis quality

### Competitive response

- mask2math PR#53 at 0.60 sets urgency for GPU lane
- If verified by organizers, our CPU-only 1.33 drops to #2
- GPU lane must reach sub-0.60 to reclaim #1
