---
name: Variable-rate masks + difficulty map built and tested (TDD, 8 tests)
description: Per-frame CRF allocation via difficulty map. Hard frames get CRF20, easy get CRF60. Difficulty from per-pair PoseNet distortion at compress time. AV1 doesn't support per-frame CRF so uses weighted average; primary use is at inflate time via hybrid_inflate.py.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Variable-Rate Mask Encoding (2026-04-23, TDD)

- `src/tac/variable_rate.py`: compute_pair_difficulty, allocate_crf_per_frame,
  encode_variable_rate_masks, decode_masks, save/load_difficulty_map
- `src/tac/tests/test_variable_rate_masks.py`: 8 tests, all pass

## Key insight
AV1 doesn't support per-frame CRF changes within a single encode (temporal
prediction makes it impractical). The difficulty map's primary value is at
INFLATE TIME — telling hybrid_inflate.py which pairs to optimize with
constrained gen vs which to use the fast renderer on.

## How to apply
1. At compress time: run renderer on all pairs, compute PoseNet distortion
2. Save difficulty.pt (2.4KB) in archive
3. At inflate time: hybrid_inflate reads difficulty, allocates CG to hard pairs
4. For mask encoding: use weighted-average CRF (not truly per-frame)
