---
name: CRITICAL — auth_eval.py used 1199 overlapping pairs, upstream uses 600 non-overlapping
description: compute_score used range(N-1)=1199 overlapping pairs. Upstream uses seq_len=2 non-overlapping (0,1),(2,3),...,(1198,1199) = 600 pairs. ALL eval_checkpoint() scores were wrong.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## The Bug (2026-04-21)

`auth_eval.py` `compute_score()` used `pair_indices = list(range(N - 1))` which
creates 1199 overlapping pairs: (0,1), (1,2), (2,3), ...

Upstream `evaluate.py` uses `seq_len = 2` with non-overlapping batching:
`seq_buf` fills to 2 then clears → pairs are (0,1), (2,3), (4,5), ... = 600 pairs.

This means:
- SegNet distortion was computed on ALL frames (not just odd frames as upstream does)
- PoseNet distortion was averaged over 1199 samples instead of 600
- The numerical distortion values were DIFFERENT from what upstream would compute

**Fix:** Changed to `pair_indices = list(range(0, N - 1, 2))` — 600 non-overlapping pairs.

**How to apply:** ALWAYS verify scoring code matches upstream evaluate.py exactly.
Any new scoring code must be diffed against upstream before deployment.
