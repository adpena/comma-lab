---
name: parametrize_strip Round 11 codex review — all edge cases hardened
description: 2026-04-28 codex adversarial review caught 3 edge cases in original parametrize_strip helper (root-level keys, multi-original weight_norm, nested chains). Fixed via path-component parsing + warning on multi-original drop + outermost-marker semantics for nested. 17 regression tests cover everything including REAL torch.nn parametrize layout. 0 known edge cases remaining.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Round 11 codex finding (medium-severity)

> "The helper gates on the literal substring `.parametrizations.` and only remaps suffix `original`. A parametrized root module emits keys like `parametrizations.weight.original` with no leading dot, so this function treats it as clean and returns it unmodified. Multi-original parametrizations such as weight_norm emit `original0`/`original1`; nested versions pass the gate but are dropped because the suffix is not exactly `original`."

## Edge cases now handled

### 1. Root-level parametrizations (no leading dot)
**Before**: `parametrizations.weight.original` passed through unchanged → loader-side missing `weight`.
**After**: split on `.` and detect `parametrizations` as a path component (not substring). Path-before may be empty → `weight`.

### 2. Multi-original (weight_norm style)
**Before**: `linear.parametrizations.weight.original0` was dropped silently as if it were a codebook.
**After**: detected as `original\d+`, dropped with `UserWarning` directing caller to use `parametrize.remove_parametrizations(model, 'weight')` on a live module instead — only PyTorch can statically combine `original0`/`original1` into the result.

### 3. Nested parametrize chains
**Before**: outer original would be renamed correctly, but inner chain's `original` keys would also be renamed (creating duplicate `weight` entries).
**After**: outermost (FIRST) `parametrizations` marker is canonical; inner chain dropped as parametrize-internals.

## 17 regression tests

Original 8 (Lane I crash repro + basic) + 9 new edge-case tests:
- `test_root_level_parametrize_key_renamed`
- `test_mixed_root_and_nested`
- `test_multi_original_dropped_with_warning` (uses pytest's `recwarn` fixture)
- `test_multi_original_warning_disabled`
- `test_nested_parametrize_chain`
- `test_has_parametrize_keys_root_level`
- `test_path_with_numeric_components`
- `test_short_parametrize_key_not_a_param`
- `test_pytorch_register_parametrization_real_layout` ← REAL `torch.nn.Linear` + `register_parametrization`, not synthetic strings

All 17 pass. Coverage matches Codex review's recommendations.

## API additions

- `_parse_param_key(key) -> (path, name, suffix)` (internal, but unit-tested via the helpers)
- `_is_canonical_original(suffix) -> bool` (single-original detection)
- `_is_multi_original(suffix) -> bool` (weight_norm-style detection)
- `reset_warning_cache()` — for tests
- `warn_multi_original=True/False` arg on `strip_parametrize_hooks` for opt-out

## Cross-references
- `feedback_parametrize_strip_helper_factored_20260428` — original factor
- `project_lane_i_crashed_parametrize_strip_20260428` — motivating crash
- `.omx/research/codex_adversarial_review_round_11_20260428.md` — review where edge cases surfaced
