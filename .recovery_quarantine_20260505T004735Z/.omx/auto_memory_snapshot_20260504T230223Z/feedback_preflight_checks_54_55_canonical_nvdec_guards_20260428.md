# Preflight Checks 54 + 55: canonical NVDEC workflow guards

**Date**: 2026-04-28 (evening pass)
**Type**: BINDING NON-NEGOTIABLE — preflight as anti-regression armor.
**Status**: STRICT @ 0 live violations (catalog count: 53 → **55**).

## Context

Today wasted ~$10 on 87% NVDEC_BAD Vast.ai 4090 hosts before the
2-layer fix landed:

1. **Layer 1 — DETECTION** (commit 58e55890):
   `scripts/probe_nvdec.sh --lightweight` at `setup_full.sh` Stage 0.5
   dlopens `libnvcuvid.so` + `cuvidGetDecoderCaps` via ctypes — DALI-free,
   ~3s, catches ~95% of NVDEC-missing hosts BEFORE the 5-minute DALI
   install (which used to be the first NVDEC-touching operation).
2. **Layer 2 — ACTION** (commit 5acebb88-ish):
   `scripts/launch_lane_on_vastai.py` `cmd_phase2_launch` Stage 2 polls
   `setup.log` via `_poll_setup_log_for_outcome()` (max ~60s) and
   auto-destroys on `NVDEC_BAD`. Closes the
   "phase2-launch returns success the moment SSH backgrounds the lane"
   silent-failure window (memory:
   `feedback_vastai_launch_returns_success_before_lane_starts`).

User mandate: *"we need to automate and canonicalize and permanently
guard against NVDEC issue."* Without preflight guards, both layers are
"please future engineers remember." With the guards below, both are
**structurally extinct bug classes**.

## What landed

**Check 54** — `check_phase2_launch_polls_setup_log` (`src/tac/preflight.py`):
- AST-scans `scripts/launch_lane_on_vastai.py` for the
  `cmd_phase2_launch` function.
- Asserts BOTH a `_poll_setup_log_for_outcome(...)` call AND a
  `getattr(args, "skip_post_verify", False)` opt-in are present in
  the function body.
- `skip_post_verify` is the explicit fire-and-forget escape hatch.
- Live count at wire-in: **0** → STRICT.
- Sub-fix: added `--skip-post-verify` argparse flag to phase2-launch
  subparser (caught by Check 13 dead-resolver scanner during validation).

**Check 55** — `check_setup_full_probe_before_dali` (`src/tac/preflight.py`):
- Scans `scripts/remote_setup_full.sh` for line N1 = first
  `probe_nvdec.sh --lightweight` (regex matches `bash "$WORKSPACE/.../probe_nvdec.sh" --lightweight` as well).
- Line N2 = first `nvidia-dali-cuda120` install OR `=== Stage 3` marker.
- Asserts N1 < N2. File without either marker is exempt (no DALI
  install ⇒ no savings to defeat).
- Comment-only lines space-padded so doc references don't false-match.
- Live count at wire-in: **0** → STRICT.

Both wired into `preflight_all()` after Check 53.

## Tests

`src/tac/tests/test_preflight_canonical_nvdec_guards.py` — 18 tests, all green:
- Check 54: 7 tests (pass-cases, fail-modes, function-missing, strict
  raises, no-launcher skip, plus a live-codebase sentinel).
- Check 55: 8 tests (probe-before-DALI, probe-after-DALI, neither, probe-only,
  DALI-only, comment-only-doesn't-count, strict raises, no-script skip,
  plus a live-codebase sentinel).
- 1 test: `test_both_checks_wired_into_preflight_all` — source-text
  introspection guards against future refactors silently dropping
  either wiring.
- 2 live sentinels guarantee the actual repo stays 0-violation.

## Meta-pattern: preflight as anti-regression armor

**Anti-arbitrariness work**: each preflight check converts a "we hope
future engineers remember" rule into a structurally extinct bug class.
The pattern:

1. Diagnose a recurring failure ($10 today on NVDEC_BAD hosts).
2. Land the canonical fix in ≥2 layers (detection + action).
3. **Land a preflight check that statically refuses any future code
   that drops a layer.** Without step 3, the fix decays.
4. Wire STRICT into `preflight_all()` once live count = 0.
5. Add live-codebase sentinel tests so the regression class can never
   re-enter via reverting step 2.

**Catalog count**: was 53 STRICT after pass 2 (`check_no_bare_except`,
`check_subprocess_run_checked`, `check_tools_have_argparse` are
warn-only). Now **55** STRICT (Checks 54 + 55).

## Side find: launcher dead-resolver

While validating Check 54, the existing Check 13 (`preflight_dead_resolvers`)
caught that `getattr(args, 'skip_post_verify', False)` had no
matching `--skip-post-verify` argparse flag. Fixed in same commit by
adding the flag to the phase2-launch subparser (and to the launcher's
docstring). This is the same bug class as `pose_dim` /
`uncertainty_loss_floor`. Two preflight checks composing well: one
guards the new pattern, the other catches a real bug exposed by the
new pattern.

## References

- `feedback_canonical_nvdec_workflow_GUARD_20260428` (canonical pattern doc)
- `feedback_vastai_launch_returns_success_before_lane_starts` (parent failure)
- `feedback_vastai_nvdec_host_variation` (host-pool variability)
- `feedback_dead_resolver_violations_20260427` (Check 13 history)
- Commits: 58e55890 (Layer 1), 5acebb88-ish (Layer 2), this commit (Layer 3 — preflight guard).
