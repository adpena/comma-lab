---
name: NEVER `git reset --hard origin/main` in remote_lane_*.sh — wipes anchors
description: 2026-04-29 5/6 TIER-1 lanes crashed Stage 1 with "missing Lane G v3 anchor archive". Yesterday's "canonical git sync" pattern (`git fetch + git reset --hard origin/main`) was the bug. The launcher tarball already SCP'd local-only anchor files (archive_lane_a.zip, baseline dirs) — `git reset --hard` then DELETED them because they aren't in origin/main. The tarball IS the parity mechanism. Check 66 STRICT prohibits the destructive pattern.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## What happened

2026-04-29 ~01:30 PT. Dispatched 6 TIER-1 lanes (J-NWCS-EC, Ω-Hessian, MAE-V, UNIWARD, CG, HM). 5 of 6 crashed in Stage 1 within 60s. Lane J-NWCS-EC error:
```
[lane-j-nwcs-ec] FATAL: missing Lane G v3 anchor archive: experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip
```

Same pattern in Ω-Hessian (anchor `experiments/results/lane_a_landed/iter_0/`), UNIWARD (anchor `submissions/baseline_dilated_h64_0_90/`), CG/HM (`LANE_A_ARCHIVE`).

~$1.50 wasted, 0 training output across 6 lanes.

## Root cause

Yesterday I added "canonical git sync" pattern to every `remote_lane_*.sh`:
```bash
git fetch origin main && git reset --hard origin/main
```

`git reset --hard origin/main` discards any tracked changes AND removes files NOT in origin/main. The launcher's tarball ships local-only artifacts (`archive_lane_a.zip` 678KB, `archive_lane_g_v3.zip` 678KB, `submissions/baseline_dilated_h64_0_90/`, etc.) — none committed to git.

Sequence per lane:
1. Launcher builds tarball (correctly includes all local-only anchors)
2. SCP to remote
3. Extract on remote (anchors in place)
4. Lane script Stage 1: `git reset --hard origin/main` → ANCHORS DELETED
5. Stage 1 line 123: `[ -f "$ANCHOR_..." ] || { log "FATAL: missing"; exit 1; }`
6. Crash.

## Why prior checks missed it

Check 43 (`check_launcher_tarball_includes_lane_anchors`) validates that the launcher's `build_tarball()` includes referenced anchor paths. It correctly reported "OK: 10 anchor path(s) all in tarball". It does NOT model what the lane script then DELETES via git reset.

Check 57 (`check_lane_scripts_use_canonical_git_sync`) actively REQUIRED the destructive pattern. RETIRED 2026-04-29.

## Permanent fix (committed 2026-04-29)

1. **Strip `git fetch + git reset --hard` from all 11 lane scripts.** Replaced with a comment block explaining tarball-trust. Lane scripts now run `pip install -e .` directly on the SCP'd tarball.

2. **Check 66 STRICT** (`check_no_git_reset_hard_in_remote_lane_scripts`): scans `scripts/remote_lane_*.sh` for executable (non-comment) `git reset --hard` references. Fails preflight with clear message. 7-test regression suite.

3. **Retired Check 57** (`check_lane_scripts_use_canonical_git_sync`): wiring removed from `preflight_all()`. Function still exists for tests but no longer enforced. Inverse of Check 66.

## How to apply

- **Lane scripts MUST NOT run `git reset --hard`.** The launcher's tarball is the ONLY source of truth for what runs on the remote.
- **Code freshness is achieved by rebuilding the tarball on every dispatch.** The launcher reads from local working tree, so `git pull` locally + immediate dispatch = remote runs the latest code.
- **If an old workspace has stale code on remote**: destroy and re-dispatch (don't try to "freshen" via git reset).
- **If you need to test an uncommitted local change**: just dispatch — the launcher tarballs your working tree as-is.

## What we lost / why this hurt

Today (2026-04-28 → 2026-04-29) cascade of regressions:
- Yesterday's NVDEC variability fix (Lane SAUG-V2 batch) introduced the canonical git-sync pattern as collateral.
- Today's 5/6 TIER-1 crash + 5 prior lane crashes (Lane V, Lane I parametrize, Lane M-V2 BUG-1, Lane RM-d 0.mkv, etc.) all share the same meta-pattern: a "fix" introduced a regression that no Check caught.
- Total lanes dispatched today that produced training output: 1 (SZ-Phase2-c moonshot still running). Total dispatched: ~10. Yield: 10%.

## Cross-references

- Commit (TBD) — strip + Check 66
- Check 43 — `check_launcher_tarball_includes_lane_anchors`
- Check 57 (RETIRED) — `check_lane_scripts_use_canonical_git_sync`
- Check 66 — `check_no_git_reset_hard_in_remote_lane_scripts`
- `src/tac/tests/test_no_git_reset_hard_in_lane_scripts.py` — 7-test regression
- Memory `feedback_canonical_remote_bootstraps` — bootstrap script discipline
- Memory `feedback_remote_code_parity_required` — UPDATE: tarball is the parity, not git pull on remote
