# macOS-CPU substrate canvas sweep progress (safe relaunch) 2026-05-13

Status: partial progress, safe relaunch in flight
Evidence axis: `[macOS-CPU advisory]` only
Score claim: false
Promotion eligible: false

## Why this memo is partial

The first macOS-CPU canvas sweep produced an early ranking memo, but that run
was interrupted/orphan-prone. This file was canonicalized on 2026-05-13 to
describe the memory-safe relaunch instead of preserving a stale "final" Pareto
ranking.

The active relaunch is supervised by:

```text
tools/relaunch_macos_cpu_canvas_sweep_safe.py --supervise
```

with `concurrency=1`, `timeout=1800`, `min_available_memory_gb=16`,
orphan cleanup enabled, and stale workdir cleanup enabled.

Artifacts:

```text
experiments/results/lane_macos_cpu_substrate_canvas_sweep_20260513_20260513T162636Z/safe_relaunch.log
experiments/results/lane_macos_cpu_substrate_canvas_sweep_20260513_20260513T162636Z/sweep_progress.json
experiments/results/lane_macos_cpu_substrate_canvas_sweep_20260513_20260513T162636Z/canvas_manifest.jsonl
```

## Current progress snapshot

Snapshot time: 2026-05-13 during safe relaunch.

```text
targets: 42
completed/pass: 12
failed: 4
rows finalized: 16
active archive at snapshot: pr084_adaptive_range_mask
```

Current best advisory rows remain above the operator's `do not waste time
above 0.19` threshold:

| Archive ID | Score | Axis | Status |
|---|---:|---|---|
| `pr101_hnerv_ft_microcodec` | 0.192861 | `[macOS-CPU advisory]` | pass |
| `a1_baseline` | 0.192864 | `[macOS-CPU advisory]` | pass |
| `pr103_hnerv_lc_ac` | 0.194865 | `[macOS-CPU advisory]` | pass |
| `pr100_hnerv_lc_v2` | 0.195369 | `[macOS-CPU advisory]` | pass |
| `pr107_apogee` | 0.196640 | `[macOS-CPU advisory]` | pass |
| `pr105_kitchen_sink` | 0.197979 | `[macOS-CPU advisory]` | pass |
| `pr104_qhnerv_ft_best` | 0.198711 | `[macOS-CPU advisory]` | pass |
| `pr081_qzs3_range_mask` | 0.287817 | `[macOS-CPU advisory]` | pass |

Failed rows at snapshot:

```text
pr102_hnerv_lc_v2_scale095_rplus1: FAIL indeterminate_rc_1
pr106_belt_and_suspenders: FAIL indeterminate_rc_1
pr053_mask2mask: FAIL indeterminate_rc_1
pr055_quantizr: FAIL indeterminate_rc_1
```

## Interpretation

This sweep is a routing signal only. It does not rank, promote, or retire
contest candidates. `[macOS-CPU advisory]` can disagree with `[contest-CPU]`
and `[contest-CUDA]`; every candidate that matters still needs same-axis exact
closure.

The immediate score-lowering conclusion is negative but useful: through 16
finalized rows, the sweep has not surfaced a sub-0.19 advisory packet. The
best live advisory remains PR101/A1 around `0.19286`, reinforcing the current
shift toward the HYBRID matrix:

```text
φ1 SABOR boundary-only audit
φ3 S2SBS stride-2-stem byte-stuffing audit
F1 PR95 curriculum reproduction
```

## Rebuild when complete

When `sweep_progress.json` reaches `rows_finalized == total`, rebuild this memo
from the safe relaunch manifest and replace this partial progress note with a
final Pareto ranking. Until then, do not use this file as a final ranking.
