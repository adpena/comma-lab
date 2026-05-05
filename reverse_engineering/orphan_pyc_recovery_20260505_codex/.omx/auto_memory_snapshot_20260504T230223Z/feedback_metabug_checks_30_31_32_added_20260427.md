---
name: 3 new strict meta-bug checks added 2026-04-27 — executable-bit + predicted-band + contest-CUDA-tag
description: 30th-32nd preflight checks added today, all from patterns observed in subagent deliveries this session. #30 (executable-bit) STRICT after 6 fixed. #31-32 (predicted_band, contest-CUDA-tag) WARN-only initially, 14 + 10 violations to clean up before flipping strict.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
**Pattern observed today** (2026-04-27): subagents writing remote_*.sh scripts forget repeated DX details. Three patterns, all NOW preflight-protected:

**Check 30 — `check_remote_scripts_executable_bit`** (STRICT):
- Bug: Lane GH script committed with mode 0o100644 → bash dispatch failed.
- Audit subagent caught via `test_script_is_executable`.
- Generalized: scan ALL `scripts/remote_*.sh`, fail-loud if any lack +x bit.
- Live count after fix: 0 (6 historical scripts chmod'd in this commit).

**Check 31 — `check_remote_scripts_record_predicted_band`** (WARN-only):
- Pattern: most lanes today record `predicted_band` in provenance JSON for council calibration. 14 older scripts don't.
- Live count: 14 violations (out of 35 provenance-emitting scripts).
- Promotion plan: sweep older scripts to add predicted_band metadata, then flip to strict in a follow-up commit.

**Check 32 — `check_remote_scripts_tag_contest_cuda_at_completion`** (WARN-only):
- Pattern: per CLAUDE.md FORBIDDEN PATTERNS rule, every score must carry a lane tag. Every recent lane completion log includes `[contest-CUDA]`. 10 older scripts don't.
- Live count: 10 violations (out of 32 contest_auth_eval-invoking scripts).
- Promotion plan: same — sweep, flip strict.

**The promotion pattern** (already documented in commit 7f2740e4 strict-flip of checks 1-11):
1. New check ships `strict=False` (warn-only)
2. Live violation count is fixed across the codebase via a sweep PR
3. Check is flipped `strict=True` in `preflight_all()`
4. Reverting any strict check fails at commit time

**Current strict-mode count: 30** (was 29 before today; check 30 added STRICT immediately because the 6 violations were all fixable in one chmod batch).

**Total preflight checks (strict + warn): 32**.

**Why this matters** (from `feedback_dead_resolver_violations_20260427`):
> Strict-mode preflight checks make bug classes STRUCTURALLY EXTINCT. Without them, every subagent has to remember every DX detail; with them, the codebase enforces the discipline automatically.

**Related memories**:
- `feedback_dead_resolver_violations_20260427` (the canonical promotion pattern)
- `feedback_silent_default_masquerading_as_negative_result` (check 29)
- `feedback_partial_tarball_deploy_traps` (a related deploy DX issue)
