# BAT00 surrogate v2 summary

## full labeled set

- rows: `18`
- device: `cuda:0`
- JAX version: `0.9.2`
- leave-one-out MAE: `1.7895286348130968`
- pairwise ranking accuracy: `0.5592105263157895`

Interpretation:
- still too noisy for decision authority
- prototype/outlier points distort the fit badly

## codec-only subset

- rows: `16`
- leave-one-out MAE: `0.1622164398431778`
- pairwise ranking accuracy: `0.6134453781512605`

Interpretation:
- materially better than the full mixed set
- may be useful as a **research-only ranking aid** for cheap codec candidates
- still not authoritative for promotion decisions
