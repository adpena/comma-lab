# Canonical git sync pattern for lane scripts

**Date:** 2026-04-28
**Severity:** TIER-1 cost-leak / experiment-blocker
**Permanent guard:** Check 57 STRICT in preflight catalog

## Bug class

Lane scripts trusted `git pull --ff-only` to sync remote code with local
HEAD on Vast.ai. This pattern aborts whenever the remote workspace has
uncommitted local changes — exactly the case when Vast.ai reuses a
container from a prior failed deploy. Stale junk from the previous
attempt blocks the new attempt at the git step before any work begins.

## Cost realised

Lane Q-FAITHFUL (true 1:1 Quantizr replica, predicted [0.40, 0.80] —
the highest-EV candidate, could PASS Quantizr 0.33) crashed at:

> FATAL: git pull failed -- remote has uncommitted/conflicting changes.

Instance was auto-destroyed: ~$0.27/hr × ~37 min = ~$0.17 + a 2nd-attempt
failure on the same fragile pattern. Ongoing exposure across 11 lane
scripts that use the same pattern → multiplies cost-leak per future deploy.

## Fix (canonical pattern)

Replace bare `git pull --ff-only` with:

```bash
# Nuke local junk from prior failed deploys, then sync to origin/main exactly.
git fetch origin main && git reset --hard origin/main
```

The `git -C "$WORKSPACE"` form is also accepted:

```bash
git -C "$WORKSPACE" fetch origin main && git -C "$WORKSPACE" reset --hard origin/main
```

Applied to 11 lane scripts on 2026-04-28 (commit c504a330). The other
58 lane scripts perform NO git sync — they trust the parent launcher to
deploy a clean checkout.

## Permanent guard (Check 57)

`check_lane_scripts_use_canonical_git_sync` in `src/tac/preflight.py`
scans all `scripts/remote_lane_*.sh`:

1. Bare `git pull --ff-only` is FORBIDDEN unless a SAME-LINE
   `# GIT_SYNC_OPT_OUT:<reason>` waiver is present.
2. Any lane script that performs git sync MUST use the canonical
   `git fetch origin main && git reset --hard origin/main` pattern.
3. Lane scripts that do NO git sync are exempt.

Wired STRICT in `preflight_all()`. Live count after Fix 1: 0.

8 unit tests in `src/tac/tests/test_preflight_canonical_git_sync.py`.

## Catalog position

- Check 56 STRICT → Check 57 STRICT (this addition).
- Total: 57 STRICT preflight checks after 2026-04-28.

## Reference commits

- Fix 1: `c504a330` — canonical pattern landed across 11 lane scripts.
- Check 57: `bf961c8b` — STRICT preflight guard + 8 tests.
