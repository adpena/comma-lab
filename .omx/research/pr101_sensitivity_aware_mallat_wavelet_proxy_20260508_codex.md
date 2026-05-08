# PR101 Sensitivity-Aware Mallat Wavelet Proxy - 2026-05-08

## Scope

This records the CPU byte-proxy sweep for
`tools/pr101_sensitivity_aware_mallat_wavelet.py`, a sister lane to the
falsified Xavier-L2 sensitivity proxy. The proxy uses Mallat/db4 wavelet detail
energy as a per-tensor compression-hardness signal before the existing
per-tensor budget allocator.

Evidence grade: `[byte-anchor; sensitivity_proxy=mallat_wavelet]`.

Score claim: `false`.

Promotion eligible: `false`.

Ready for exact eval dispatch: `false`.

Generated local manifest:
`experiments/results/pr101_sensitivity_aware_mallat_wavelet_codex_20260508Tlocal/build_manifest.json`
(ignored rebuildable artifact).

## Command

```bash
.venv/bin/python tools/pr101_sensitivity_aware_mallat_wavelet.py \
  --output-dir experiments/results/pr101_sensitivity_aware_mallat_wavelet_codex_20260508Tlocal
```

## Results

| average_budget | eta | archive_bytes | rel_err | delta_vs_pr101 | delta_vs_uniform | delta_vs_xavier_l2 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.020 | 0.0 | 176,990 | 0.0019 | -1,154 | 0 | 0 |
| 0.020 | 0.5 | 176,624 | 0.0037 | -1,520 | -366 | -808 |
| 0.020 | 1.0 | 176,700 | 0.0036 | -1,444 | -290 | +69 |
| 0.050 | 0.0 | 156,344 | 0.0386 | -21,800 | 0 | 0 |
| 0.050 | 0.5 | 159,833 | 0.0360 | -18,311 | +3,489 | +1,962 |
| 0.050 | 1.0 | 156,796 | 0.0392 | -21,348 | +452 | -3,183 |

## Verdict

`incremental_improvement_insufficient`.

The Mallat proxy is not a dispatchable score-lowering lane in this
configuration. It beats uniform by 366 B at the conservative 0.020 budget and
beats Xavier-L2 by 808 B in that cell, but it does not clear the 3 KB council
threshold. At the aggressive 0.050 budget, uniform analytical coarsening remains
best.

Reactivation criteria:

- real Hessian-trace / scorer-gradient importance through SegNet and PoseNet;
- empirical compression-hardness proxy from per-tensor K sweeps;
- reuse only as a feature in a learned per-substrate profile registry, not as a
  standalone promotion lane.
