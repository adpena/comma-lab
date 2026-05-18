# Codex Session Summary - Master-Gradient Projector Contract

Date: 2026-05-18 19:06:08 UTC
Author: Codex

## Landed

- Hardened Ruff configuration with `force-exclude = true` so explicit-path
  manual invocations respect generated custody excludes without needing every
  caller to remember `--force-exclude`.
- Added a typed master-gradient archive `projection_contract` that separates
  xray boundary detection from projector-backed anchor authority.
- Added `--layout-contract-output` so unsupported archive grammars can persist
  a layout/authority manifest before the extractor fails closed.
- Preserved stable partner directives that were present in the worktree:
  v2.5 inbox integration, cheap-probe wave, and Claude memory hermetic export
  channel.

## Verification

- `37 passed` for `src/tac/tests/test_extract_master_gradient.py` and
  `src/tac/tests/test_ci_ruff_scope.py`.
- Focused Ruff passed on the edited files.
- Blocking F821 Ruff passed across `src/ experiments/ submissions/robust_current/
  scripts/ tools/`.
- Real-fixture grammar probes confirmed PR106 format0d, public PR106 packed,
  PR100 HNeRV LC v2, and PR107 Apogee all serialize fail-closed projector
  contracts.

## Next

Continue with the new cheap-probe routing directive after the canonical queue
is refreshed: OP-7 pose-byte hoist is the next concrete Cathedral/autopilot
consumer path once the required master-gradient anchor fields are confirmed.
