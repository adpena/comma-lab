# PR95 MLX C36 Minimal Drift Scope

Generated: 2026-05-28T12:30Z
Author: Codex
Lane: `pr95_hnerv_mlx_reproduction`
Evidence grade: `[macOS-MLX research-signal]`

## Verdict

For the C36 PR95 public archive GPU parity probe, Kahan accumulation on only
upsample blocks 0 and 1 closes the observed drift to the same no-cliff band as
full `kahan_fp32` and all-block Kahan.

Single-block probes show partial reductions for block 0 and block 1, but no
single block closes the cliff:

| Scope | Max abs | Mean abs | First cliff |
| --- | ---: | ---: | --- |
| block 0 only | 0.00331878662109375 | 0.0003908915095962584 | `rgb_0` |
| block 1 only | 0.00403594970703125 | 0.0004434349248185754 | `rgb_0` |
| block 2 only | 0.00702667236328125 | 0.0006695801275782287 | `rgb_0` |
| block 3 only | 0.00702667236328125 | 0.000669580593239516 | `rgb_0` |
| block 4 only | 0.00702667236328125 | 0.0006695787888020277 | `rgb_0` |
| block 5 only | 0.00702667236328125 | 0.0006695787888020277 | `rgb_0` |
| blocks 0..1 | 0.000030517578125 | 0.0000038458447306766175 | none |

## Canonicalization

Promoted `blocks01_kahan_fp32` as a named
`PR95_MLX_CONV2D_ACCUMULATION_OVERRIDE_PRESETS` entry.

The queue CLI also now accepts repeatable explicit overrides:

```bash
--mlx-gpu-drift-conv2d-override blocks.0.conv=kahan_fp32
--mlx-gpu-drift-conv2d-override blocks.0.skip_conv=kahan_fp32
```

This makes per-layer numerical drift acquisition queue-owned instead of a
one-off code edit.
