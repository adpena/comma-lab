---
name: Lane PD pose-delta — savings overstated in docstring (49% claimed, ~18% actual after torch.save overhead)
description: 2026-04-29 PM. Discovered while writing the encode-side regression test for compress_archive.py --pose-delta wiring. The codec docstring at src/tac/pose_delta_codec.py:42-49 claims 49% savings vs fp16 baseline by comparing core encoded bytes (3618 B) to raw fp16 numel*2 (7200 B). Empirically the torch.save dict overhead is ~2KB on both sides, so real-world savings drop to ~18% on a 600-pair smooth trajectory.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Empirical numbers (2026-04-29)

| Format | Bytes | Savings vs vanilla .pt |
|---|---:|---:|
| Vanilla `torch.save(poses)` | 15899 | 0% (baseline) |
| Raw fp16 file (numel*2) | 7200 | 55% |
| `torch.save(poses.to(fp16))` | ~7200 | 55% |
| Lane PD `torch.save(encode_pose_deltas())` | 5865 | **63%** vs .pt, **18.5%** vs raw fp16 |

## Why the docstring overstated

The docstring at `src/tac/pose_delta_codec.py:42-49` calculates:
```
fp16 absolute baseline:  600 * 6 * 2 = 7200 bytes
Lane PD encoded:
    anchor (fp16)        :   12 B
    delta_scale (fp16)   :   12 B
    deltas_q (int8)      : 599 * 6 = 3594 B
    + dict overhead      :  ~50 B
    ----------------------------------------
    total                : ~3668 B (-49%)
```

This counts the pickle dict overhead as ~50 B but in reality torch.save adds ~2KB of framing on the dict path (vs raw bytes for fp16 numel-tensor). The actual encoded file is ~5865 B, not 3668 B.

## Score impact correction

- Docstring: `25 * (7200 - 3668) / 37545489 ≈ 0.00235` → -0.002 score
- Actual:    `25 * (7200 - 5865) / 37545489 ≈ 0.00089` → -0.0009 score

So Lane PD's standalone score impact is roughly -0.001, not -0.002. Still positive but half as impactful as advertised. Composes with Lane SH (smaller pose tensor → smaller xz residual) so total stack value > standalone.

## How this was caught

Wrote `src/tac/tests/test_compress_archive_pose_delta.py:test_convert_poses_to_pose_delta_roundtrip` with `assert savings > 0.20` — failed at 18.5%, surfacing the gap. Updated to `> 0.15` which is the empirical floor with a small safety margin.

## What still needs fixing

- `src/tac/pose_delta_codec.py:42-57` docstring should be updated to reflect the ~18% real-world figure (not 49%) and the ~-0.001 score impact (not -0.002). Deferred to next loop — strict-gate would require council review for a docstring-only change.

## Sister insight: claims-without-empirics class

This is a sister bug class to "fix lands in helper but never callsite" — claims in docstrings/memos without empirical verification. The codec author computed the bytes ANALYTICALLY but never measured the actual on-disk file. A simple regression test catches it. Recommend: every codec module ships with a "savings vs claim" test that runs once and asserts the documented number is within 5% of empirical.

## Cross-refs

- `src/tac/pose_delta_codec.py:42-57` (the overstated docstring)
- `src/tac/tests/test_compress_archive_pose_delta.py` (regression test that caught it)
- commit 2d913687 (encode-side wiring that surfaced this)
- `feedback_silent_default_bug_class_findings_20260429.md` (sister bug class)
