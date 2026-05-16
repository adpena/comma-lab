# Time-Traveler L5 AC-State Consumption Hardening

Date: 2026-05-16
Author: codex
Scope: TT5L archive grammar and inflate runtime

## Finding

The paper-fidelity sweep identified a concrete L5-v2 mechanism blocker:
`AC_STATE_BLOB` existed in the TT5L grammar and parser, but `pack_archive()`
rejected non-empty `ac_state` because inflate did not consume those bytes.
That was correct fail-closed behavior, but it left Stage 3 as a placeholder
and prevented byte-closed entropy-state experiments.

## Fix

- `pack_archive()` now accepts non-empty `ac_state`.
- `parse_archive()` already preserved the decompressed state; tests now pin that
  round trip.
- TT5L inflate now consumes non-empty AC-state bytes as narrow residual
  calibration gains over the per-pair side-info sections.
- Archive-level mutation tests prove that changing AC-state bytes changes the
  inflated raw frames when side-info is active.

## Scope Boundary

This is not a completed range/ANS arithmetic decoder and not a score claim.
It is a production hardening step that eliminates dead charged bytes and gives
L5-v2 a byte-closed, mutation-provable Stage-3 state channel. A future entropy
decoder can replace the calibration mapping while preserving the `ac_state_blob`
section contract.

## Verification

```bash
.venv/bin/python -m pytest \
  src/tac/substrates/time_traveler_l5_autonomy/tests/test_time_traveler_archive.py \
  src/tac/substrates/time_traveler_l5_autonomy/tests/test_time_traveler_inflate.py \
  src/tac/substrates/time_traveler_l5_autonomy/tests/test_parse_tt5l_archive_bytes_canonical.py -q
```

Result: `43 passed`.

```bash
.venv/bin/python -m pytest \
  src/tac/substrates/time_traveler_l5_autonomy/tests \
  src/tac/tests/test_l5_staircase_v2.py \
  src/tac/tests/test_substrate_composition_matrix.py -q
```

Result: `180 passed`.
