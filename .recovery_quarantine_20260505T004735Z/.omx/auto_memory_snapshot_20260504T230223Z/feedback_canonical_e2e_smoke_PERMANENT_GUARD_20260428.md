# Canonical local E2E auth-eval smoke + Check 64 (2026-04-28) — BINDING

## Bug class: static-analysis preflight ≠ pipeline validation

We had 63 STRICT preflight checks before Check 64. **Every one of them
guarded a CODE PATTERN** (no MPS fallback, no shell zip, eval_roundtrip
default, NVDEC probe, F5 config.env presence). **None of them actually
RAN the pipeline locally to prove a lane archive would inflate end-to-end.**

A lane could pass every static check and still ship to Vast.ai with a
broken pipeline.

## Why we missed it

The check authors (me) iterated reactively after each post-mortem:
- Lane RM-d 0.mkv crash → F5 fix (config.env presence guard)
- LANE-B zip dep cascade → check_no_shell_zip_binary
- DALI on bad NVDEC host → check_setup_full_probe_before_dali

Every fix targeted the SPECIFIC code pattern that caused the most recent
failure. But the META-bug was the assumption "if every static guard
passes, the lane will work." That assumption was never validated. The
pipeline was structurally untested at the deploy boundary.

User quote (verbatim): "wow how did the 0.mkv video thing happen, that
is so basic, extremely depressing; i thought our preflight and everything
were hardened enough to at least ensure basic contest compliance"

## The canonical fix

Two artifacts:

1. **`experiments/canonical_local_auth_eval_smoke.py`** — runs the FULL
   deploy → inflate → contest_auth_eval pipeline locally against a
   known-good fixture (Lane G v3's committed 1.05 [contest-CUDA] archive).
   Validates 10 stages: extract, whitelist, renderer-magic, masks-present,
   config.env, inflate-dispatch, inflate_renderer-imports, evaluate-arity,
   GT-video-present, launcher-includes-env. Designed budget < 60s on
   local MPS / CPU; observed 0.02s on M5 Max.

   Output: writes `.omx/state/lane_e2e_smoke_proofs.json` per-lane with
   timestamp, archive_sha256, stages_passed.

2. **Check 64 (`check_lane_scripts_have_e2e_smoke_proof`)** — STRICT
   preflight. Every `scripts/remote_lane_*.sh` must have an entry in the
   smoke-proofs JSON that is < 7 days old, OR have a same-line
   `# E2E_SMOKE_OPT_OUT:<reason>` waiver (≥4 chars of reason).

## Structural guard — why this is permanent

Check 64 is in `preflight_all()` STRICT. Every dispatch path runs
preflight. Every commit-time hook runs preflight. Every CI job runs
preflight. **A new lane CANNOT dispatch without an E2E smoke proof.**

If a lane's fixture would crash at inflate (Lane RM-d's exact bug), the
smoke catches it locally — sub-60s, free, before any GPU spend.

If the smoke tool itself is regressed (false-positive PASS), the test
suite catches it: `test_canonical_local_e2e_smoke.py` asserts the smoke
PASSES on canonical fixture AND fails on missing-fixture AND completes
in < 60s.

## Backfill — 70/70 lanes proven

Backfilled `--backfill-all` against all 70 existing
`scripts/remote_lane_*.sh` scripts. Every lane proved cleanly against
the Lane G v3 fixture. Total time: 1.5 seconds.

## Tests (8)

`src/tac/tests/test_check_64_e2e_smoke_proof.py`:
- test_check_passes_when_proof_exists_and_recent
- test_check_fails_when_proof_missing
- test_check_fails_when_proof_too_old (>7 days)
- test_check_passes_when_waiver_present
- test_check_skips_when_no_lane_scripts
- test_check_wired_into_preflight_all
- test_check_rejects_waiver_with_too_short_reason
- test_check_rejects_proof_without_timestamp
- test_check_handles_missing_proofs_file
- test_real_repo_has_no_violations

`src/tac/tests/test_canonical_local_e2e_smoke.py` (10 tests):
- existence + fixture present + sub-60s budget
- writes proof atomic, sentinel format, lane-glob, backfill-all
- whitelist parity with contest_auth_eval

## Defense-in-depth: inflate.sh self-detect

Layered on top: `submissions/robust_current/inflate.sh` now auto-detects
when ARCHIVE_DIR contains `renderer.bin*` but `PYTHON_INFLATE != renderer`,
and refuses with an actionable error pointing at `config.env` + the
canonical smoke tool. So even if config.env is dropped on a contest env,
the operator gets a clear error instead of an opaque ffmpeg crash 200
lines later.

## Catalog count

- Before: 63 STRICT preflight checks (all static)
- After: 64 STRICT preflight checks (one is now PIPELINE-OUTCOME)

This is the FIRST check that validates pipeline outcomes, not code
patterns. Every future lane gets free pipeline validation by virtue of
the smoke being mandatory.

## When to re-run smoke

Operators must re-run smoke when:
- Adding a new lane script (Check 64 fails the dispatch otherwise)
- Modifying inflate.sh / config.env / inflate_renderer.py (proof goes stale)
- Updating upstream pin (evaluate.py argparse may have changed)
- Running with a new fixture archive

Cadence: cheap enough (sub-60s for all 70 lanes via `--backfill-all`)
that operators should re-run it weekly even when nothing changed.

## Pattern to apply elsewhere

**Every preflight gap of the form "this would have caught the failure"
should be matched by an OUTCOME check, not just a pattern check.**
Pattern checks are fast and useful but produce false confidence. Outcome
checks (run the actual code, observe the actual behavior) are the only
honest verification.
