---
name: tac.parametrize_strip — canonical helper factored out
description: 2026-04-28 factored out 4-call-site duplicated parametrize-strip logic into src/tac/parametrize_strip.py. 8-test regression suite covers Lane I exact crash reproduction. Replaces inline pattern at qat_finetune.py:218, train_joint_pair.py:980, train_distill.py:1288, lane_i Stage 3.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## What changed

Created `src/tac/parametrize_strip.py` with 2 exports:
- `strip_parametrize_hooks(state, drop_internal=True)` — canonical strip
- `has_parametrize_keys(state)` — predicate

Drop-in replacement for the 4 inline implementations across the codebase:
1. `experiments/qat_finetune.py:218-238` (was canonical, slightly more complete)
2. `experiments/train_joint_pair.py:980-982` (subset)
3. `experiments/train_distill.py:1288-1290` (subset)
4. `scripts/remote_lane_i_coolchic_masks.sh` Stage 3 inline (just added 2026-04-28)

## Usage pattern

```python
from tac.parametrize_strip import strip_parametrize_hooks, has_parametrize_keys

ckpt = torch.load(path)
state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
if has_parametrize_keys(state):
    state = strip_parametrize_hooks(state)
model.load_state_dict(state, strict=True)
```

## Why factor

- Lane I lost ~$1.60 + 5.7h GPU + Stage 3 export when its inline strip was missing
- 4 separate copy-paste implementations had drifted slightly (different drop semantics)
- Future Lane F-V5 export will need the same logic for FP8 hardware path

## Tests

`src/tac/tests/test_parametrize_strip.py`:
- `test_no_strip_needed_returns_copy` — plain state passes through
- `test_original_key_renamed_to_plain_weight` — canonical mapping
- `test_codebook_dropped_by_default` — internals dropped
- `test_drop_internal_false_preserves_all` — flag controls
- `test_mixed_state_partially_stripped` — both kinds of keys in same state
- `test_has_parametrize_keys_detects` — predicate correctness
- `test_empty_state` — edge case
- `test_lane_i_crash_exact_reproduction` — verbatim from crash log

All 8 pass.

## Follow-up TODO

Refactor the 3 existing inline implementations (qat_finetune.py, train_joint_pair.py, train_distill.py) to import from tac.parametrize_strip. The Lane I script already uses an inline copy of the helper (mirrored from qat_finetune.py:218); ideally that would import too but inline Python in shell scripts can't easily import from src/tac without env setup.

## Cross-references
- `project_lane_i_crashed_parametrize_strip_20260428` — the motivating crash
- task #121 (commit) — earlier qat_finetune.py fix that was the canonical pattern
