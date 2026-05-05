# Codex Adversarial Review — 3 Findings Fixed (2026-04-28)

Codex review verdict: needs-attention (no-ship). 3 findings resolved + regression
tests landed in 3 commits. Lane EC (instance 35781802) + Lane EBR (instance
35781775) were LIVE on Vast.ai when fixes shipped — git-pull at next stage
transition picks up the fixes.

Commits:
- `6de52150` — Codex finding 3 fix: bit allocator bracket growth
- `38f81188` — Codex finding 1 fix: greedy correction allocator preserves grad magnitude + real packed-size cap
- `fed09e10` — Codex finding 2 fix: install entropy bottleneck before EMA + safe key init + strip from export

## Finding 1 [HIGH] — Greedy correction allocator broken

File: `experiments/precompute_gradient_corrections.py:383-443`

### Bug 1: sign-only writes
`greedy_waterfill_correction_map` wrote `±127` for every selected pixel (sign of
the gradient only). Downstream `sparsify_and_quantize` then re-quantized those
at `scale=127`, and `apply_corrections` added `±127 * scale / 127 = ±127` to
each frame pixel — CLAMPING pixels to ±127 instead of nudging them by the
gradient direction. Lane EC would have produced visually destroyed frames.

**Fix**: encode the ORIGINAL gradient values via per-tensor symmetric int8
quantization (`scale = max|grad|`). Both sign and relative magnitude survive.
`sparsify_and_quantize` now pulls original gradient values via
`flat_grads[top_k_indices]` (not the int8-clamped correction_map) so the
downstream quantization scale is correct.

### Bug 2: budget mismatch
`rate_cap_bytes` used a 1-byte/pixel arithmetic model. Real packed format =
~100B JSON header + 4B uint32 index + 3B int8 channels per pixel = 7-13× the
fast model. A `50 KB cap` could materialise as ~350 KB on disk — silently
busting the archive size limit.

**Fix**: added `enforce_packed_byte_cap()` that drops tail entries (lowest-
magnitude first) and repacks until `len(pack_sparse_corrections(...)) <=
rate_cap_bytes`. `sparsify_and_quantize` calls it automatically and records
`packed_bytes` in the output.

### Pattern (memory)
**Whenever a fast-model byte budget feeds into a real packed-format archive,
add a feedback loop that calls the actual packer and drops tail until under
cap.** Never trust a closed-form byte estimate.

## Finding 2 [HIGH] — Entropy bottleneck registration invalidates EMA

File: `src/tac/training.py:487-527`

### Bug
`Trainer.__init__` snapshotted `self.ema = EMA(model, ...)` BEFORE registering
the `entropy_bottleneck` submodule (when `use_entropy_bottleneck=True`). The
first `EMA.update(self.model)` iterates `entropy_bottleneck.*` keys absent
from the EMA shadow → KeyError → Lane EBR crash on first update.

### Fix (3 layers of defense)
1. **Reorder `__init__`**: entropy_bottleneck install moves above the EMA
   construction so the snapshot includes its parameters.
2. **EMA.update hardening**: missing keys seed from live tensor instead of
   KeyError. Defends against ANY future module added post-snapshot
   (RAFT-pose, late SegNet KL distill plug-ins, etc.).
3. **Checkpoint export strip**: training-only `entropy_bottleneck.*` keys are
   filtered from BOTH fp32 and int8 archive dumps. Deployment loader sees no
   phantom entries; int8 archive size doesn't include them (relevant to rate).

### Pattern (memory)
**Any module-registration that happens after EMA/optimizer construction is a
bug class.** The fix is "register first, snapshot second" + "make the snapshot
tolerant of late additions". Both belong in CLAUDE.md / preflight long-term.

## Finding 3 [MEDIUM] — Bit allocator under-spend

File: `src/tac/bit_allocator.py:155-200`

### Bug
`allocate_bits([100, 1], total_bits=12, alpha=0.5, max_bits=8)` returned
`[8, 1]` (sum=9), under-spending by 3 bits. Cause: `c_hi` initialized to
saturate the highest-importance weight at `max_bits`, leaving no room for
the search to push low-importance weights above `min_bits`.

### Fix
Bracket-growth phase before bisection: while `_bits_for_c(c_hi)` sum stays
≤ budget AND not all weights at max_bits, double `c_hi`. Capped at 48 growth
iterations + early-exit when sum stops changing (handles degenerate or
monkeypatched `_bits_for_c`).

Result: same input now yields `[8, 4]` (sum=12) — full budget spent.

### Pattern (memory)
**For monotonic bisection problems, always test the upper bracket first.**
If `f(c_hi)` is on the wrong side of the target, grow the bracket until it
crosses; only then bisect. Reviewing this class of bugs: also applies to
`compute_psd_threshold`, `find_pose_offset_for_target_msg`, etc.

## Test count

73 tests pass across the 4 touched test files:
- `test_bit_allocator.py`: 25 (was 24, +1 anchor)
- `test_lane_ec_v2_greedy.py`: 10 (was 7, +3 anchors)
- `test_training.py`: 25 (was 24, +1 anchor)
- `test_entropy_bottleneck.py`: 13 (unchanged regression coverage)

Broader pytest run: 538 passed, 1 unrelated pydantic literal_error in
`test_hardening.py::test_profile_creates_valid_config[j_jbl_dilated_h64]`
(profile schema mismatch, predates this PR).

## Preflight

`preflight_all()` exits 0. No new warnings.
