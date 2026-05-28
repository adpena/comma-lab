# PR95 MLX Automated Conv2d Drift Scope Search

Generated: 2026-05-28T12:45Z
Author: Codex
Lane: `pr95_hnerv_mlx_reproduction`
Evidence grade: `[macOS-MLX research-signal]`

## What Changed

Added `tools/run_pr95_mlx_conv2d_drift_scope_search.py`, an in-process
acquisition tool that loads the PR95 archive and source decoder once, evaluates
the Conv2d override lattice, ranks candidates, and emits a fail-closed summary.
This replaces manual shell loops for drift-scope probing.

Reusable candidate generation now lives in `tac` via
`pr95_mlx_conv2d_scope_search_candidates(...)`.

## Public PR95 Release Result

Artifact:
`.omx/research/pr95_mlx_conv2d_scope_search_public_pr95_20260528Tlocal/scope_search_summary.json`

On the actual public PR95 release archive:

| Selector | Candidate | Max abs | Mean abs | Overrides | Cliff |
| --- | --- | ---: | ---: | ---: | --- |
| best by raw delta | `preset_blocks_refine_kahan_fp32` | 0.0008697509765625 | 0.00003437255509197712 | 14 | none |
| minimal tolerance-pass | `preset_blocks01_kahan_fp32` | 0.00115966796875 | 0.0000382297184842173 | 4 | `rgb_1` |
| minimal no-cliff | `preset_blocks02_kahan_fp32` | 0.000946044921875 | 0.000037276826333254576 | 6 | none |

## Canonicalization

Promoted `blocks02_kahan_fp32` as the stricter production-facing preset for
public PR95 archive no-cliff MLX GPU parity probes. `blocks01_kahan_fp32`
remains useful as the lower-cost tolerance-passing scope.

All outputs remain `[macOS-MLX research-signal]` and fail closed for score,
promotion, exact-dispatch, and rank/kill authority.
