---
name: Weight distribution analysis — 37% carry 1% signal, mixed-precision opportunity
description: Kurtosis 33.6 (heavy-tailed). 27% of weights carry 90% of signal energy. 37% carry only 1%. Optimal is mixed-precision by band + LZMA2. Int4+LZMA2 achieves 2.18 bits/weight on current model.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---

## Weight Distribution (103K param model, epoch ~2150)

Kurtosis: 33.6 (vs Gaussian 3.0) — extremely heavy-tailed Laplacian
Near-zero (<0.01): 6.2% of weights
Mean: 0.007, Std: 0.220

## Signal Energy by Band

| Band | % weights | % signal | Optimal bits |
|------|----------|---------|-------------|
| >0.2 | 18.5% | 83.6% | 6-8 bits |
| 0.05-0.2 | 51.8% | 15.0% | 3-4 bits |
| 0.01-0.05 | 23.4% | 0.5% | 1-2 bits |
| <0.01 | 6.2% | 0.0% | 0 (prune) |

90% of signal in top 27% of weights.
37% of weights carry only 1% of signal.

## Compression Benchmark (actual model)

| Method | Size | bits/weight |
|--------|------|------------|
| Int4+LZMA2 | 27.5KB | 2.18 |
| Log4+LZMA2 | 47.3KB | 3.76 |
| FP4+LZMA2 | 55.9KB | 4.44 |
| Int8+LZMA2 | 79.3KB | 6.29 |

## Optimal Strategy

Mixed-precision by magnitude band + LZMA2:
- High-signal weights (27%): 6-8 bits
- Medium weights (35%): 3 bits
- Low weights (32%): 1 bit
- Near-zero (6%): prune
- Estimated: 1.5-2.0 effective bits/weight after LZMA2

LearnableBitDepth in self_compress.py does this per-channel (automatic).
Combined with ArithmeticCoder or LZMA2, should reach szabolcs's 1.017 territory.
