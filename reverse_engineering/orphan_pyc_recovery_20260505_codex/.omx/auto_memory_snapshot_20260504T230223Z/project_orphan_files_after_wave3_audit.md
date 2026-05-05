# Orphan files after wave-3 audit (2026-04-28)

After auditing the 55 ?? files left by 9 dead codex sessions in the 2026-04-28
mass session, the following 4 files remain GENUINE ORPHANS — they could not be
landed cleanly because either the code is broken (F821) or the tests target
infrastructure that was never authored. They are NOT deleted (per audit rules)
but they require a follow-up engineering pass before they can be committed.

## Orphans

### 1. scripts/launch_lane_on_vastai.py
- **Why orphan**: Two F821 (undefined-name) bugs:
  - `cmd_phase2` referenced at line 811 — the function does not exist (likely renamed
    to `cmd_phase2_wait` / `cmd_phase2_scp` / `cmd_phase2_extract` / `cmd_phase2_launch`
    during the V4 split; the legacy non-split callsite was left behind).
  - `threading` referenced at line 947 without an `import threading` at module top.
- **Suggested action**: Fix both NameErrors. The V4 launcher (`feedback_launcher_v4_split_works_end_to_end_20260428`)
  was the working pattern; this file looks like a partial V3-to-V4 migration that
  did not finish. Either complete the migration or delete the dead callsite + import.
- **Risk if left**: blocks any future commit attempt that stages this file (F821 ruff
  gate fires); also any operator who tries to run it will hit NameError.

### 2. src/tac/uniward_texture.py
- **Why orphan**: Three F821 in the module:
  - `_UNIWARD_ORIGINAL_APPLY` (line 80)
  - `_default_zlib_encoder` (line 120)
  - `_encode_with_inv` (line 121)
- These helper symbols are referenced but never defined. The companion test
  (`test_uniward_texture.py`) only exercises `compute_texture_probability`, so
  the test suite passes (6/6) but ~half the module is dead code that imports
  nothing and references nothing real.
- **Suggested action**: Either (a) implement the missing helpers, or (b) delete
  the broken `apply_saliency_weighted_compression`-style code paths. The Lane
  SI-V3 design intent (UNIWARD-style texture probability masks) is sound; the
  partial implementation is the issue.

### 3. src/tac/tests/test_uniward_texture.py
- **Why orphan**: Pairs with the broken module above. If module #2 is fixed,
  this can land alongside.
- Note: tests pass in isolation, but staging the test alone would orphan the
  test from its module. Holding both for the same fix-and-commit cycle.

### 4. src/tac/tests/test_remote_lane_gp_script.py
- **Why orphan**: Tests target `scripts/remote_lane_gp_gaussian_process_pose.sh`
  which DOES exist (10KB, +x). But 2 of 4 tests fail:
  - `test_no_shell_zip` — script uses raw `zip` shell binary (forbidden per
    `feedback_zip_dep_bootstrap_trap`).
  - `test_no_mps_fallback` — script has an MPS-fallback ternary (forbidden per
    `feedback_default_to_convenience_trap`).
- **Suggested action**: Fix the GP bootstrap script to use `python -c 'import zipfile'`
  for archive creation and remove the MPS fallback (CUDA-required raise). After
  fixing the script, the test will pass and both can land together.
- The other 2 tests (`test_script_has_set_euo_pipefail`, `test_stage_labels_present`)
  also reference path-related properties; verify after script fix.

## Summary

| File | Issue | Fix EV |
|---|---|---|
| `scripts/launch_lane_on_vastai.py` | F821 (cmd_phase2, threading) | medium — V4 already works |
| `src/tac/uniward_texture.py` | F821 (3 missing helpers) | medium — Lane SI-V3 abandoned without UNIWARD path |
| `src/tac/tests/test_uniward_texture.py` | pairs with #2 | n/a — lands with #2 |
| `src/tac/tests/test_remote_lane_gp_script.py` | tests fail; script needs fixing first | low — GP lane is in flight, fix at lane completion |

All other 51 ?? files from the 2026-04-28 wave were classified, tested, and
landed in commits 4f88ac0f .. 1626925e (see git log).
