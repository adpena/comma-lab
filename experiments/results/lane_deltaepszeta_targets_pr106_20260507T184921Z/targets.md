# Delta-Epsilon-Zeta Training Targets - PR106 substrate

**Source**: `experiments/results/lane_per_tensor_shannon_pr106_20260507T173846Z/per_tensor_shannon.json` `[empirical]`
**Substrate**: `experiments/results/sensitivity_map_pr106_20260504_claude/state_dict.pt`
**N tensors**: 28
**Total current H0-H2 gap**: 78,580 B
**Total brotli bytes today**: 170,096 B
**H2/H0 ratio aggregate**: 0.5310

## Top 10 tensors by training-prize bytes

| rank | idx | name | n_symbols | H0 | H2 | headroom (bits) | prize (B) | weight |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 6 | `blocks.2.weight` | 34,992 | 6.3183 | 2.6737 | 3.6446 | 15,942 | 0.2029 |
| 2 | 4 | `blocks.1.weight` | 46,656 | 5.9173 | 3.4619 | 2.4554 | 14,320 | 0.1822 |
| 3 | 2 | `blocks.0.weight` | 46,656 | 5.8286 | 3.5527 | 2.2759 | 13,273 | 0.1689 |
| 4 | 8 | `blocks.3.weight` | 19,440 | 6.1904 | 2.2662 | 3.9242 | 9,536 | 0.1214 |
| 5 | 10 | `blocks.4.weight` | 12,960 | 6.6578 | 1.3211 | 5.3367 | 8,646 | 0.1100 |
| 6 | 0 | `stem.weight` | 48,384 | 5.2221 | 4.0054 | 1.2167 | 7,359 | 0.0936 |
| 7 | 12 | `blocks.5.weight` | 11,664 | 5.2663 | 2.8722 | 2.3941 | 3,491 | 0.0444 |
| 8 | 1 | `stem.bias` | 1,728 | 7.2034 | 0.0956 | 7.1078 | 1,536 | 0.0195 |
| 9 | 22 | `refine.1.weight` | 1,458 | 6.3736 | 0.3409 | 6.0327 | 1,100 | 0.0140 |
| 10 | 20 | `refine.0.weight` | 1,458 | 5.4656 | 0.7820 | 4.6836 | 854 | 0.0109 |

## Interpretation

- The `headroom_bits` column = H0 - H2; this is the current
  per-symbol conditional-entropy gap a context-aware coder could
  exploit (vs current brotli sitting at about 1.015x H0).
- The `prize (B)` column = headroom_bits * n_symbols / 8; this is
  a per-tensor byte-gap weighting signal for delta-epsilon-zeta
  substrate experiments.
- The `weight` column normalizes prizes to sum to 1; use as a
  per-tensor multiplier in the auxiliary loss term.

## How to apply

The example below is a **structured-H2 experiment**: it minimizes H2
on tensors with the largest current H0-H2 gap. It is only useful when
the produced archive uses a context-aware coder that can exploit that
same conditional structure. It is not a direct proof of score or byte
savings.

```python
import torch
from tac.shannon_h2_loss import shannon_h2_loss
import json

targets = json.load(open('targets.json'))
weights = {r['name']: r['loss_weight_normalized'] for r in targets['per_tensor']}

def deltaepszeta_aux_loss(state_dict):
    total = torch.tensor(0.0)
    for name, w_value in state_dict.items():
        if name in weights:
            h2 = shannon_h2_loss(w_value, n_bits=8)
            total = total + weights[name] * h2
    return total
```

Score claims: **none**. This is a training-target derivation; final
score evidence comes from contest-CUDA `archive.zip -> inflate.sh ->
upstream/evaluate.py` on the produced archive bytes.
