# Codex session summary — system memory governor measured-growth fix

Date: 2026-07-14  
Lane: `governor_measured_growth_fix`  
Pointer: unchanged; `$0`; control-plane only.

## Landed in the workspace

- Deterministic bounded per-process RSS history at the governor's existing scan edge.
- Pure observed-growth estimator and pure projected-peak consumer.
- PID-reuse protection using kernel start identity.
- Nonzero plateau/burst reserve and max-recent-slope projection capped at legacy +25 GiB.
- Structural requirement that only runtime-throttle-eligible own leaders may receive a relaxed reserve.
- Expanded governor, TOCTOU, tier-floor, and process-parser regressions.
- Typed receipt, findings memo, and isolated DAG FEED; canonical equation registration held for its live owner.

## Measured apparatus result

On the incident snapshot (`used=50`, stable processes `4.4/5.1`, new job `6`, ceiling `72`):

- legacy/no history: `106.0 GiB`, `REFUSE`;
- stable measured history: `59.333333333333336 GiB`, `ADMIT`;
- real rising fixture: `82.66666666666667 GiB`, `REFUSE`.

An unpausable material process still receives +25 GiB. Under critical pressure the eligible stable
fixture is still selected for reversible pause by the unchanged runtime layer.

## Gauntlet

The first review sequence found a contradictory stale safety comment and reset. On the corrected
bundle `2ccdefba59bfb491f0ed6393552e0808e90492f2d5fd83aadca9df68bcd97342`, review-counter rounds
4/5/6 are three consecutive clean passes and the surface is sealed. Each pass reran the relevant
control-plane lens; all four requested suites ended `195 passed, 1 deselected`. The deselected existing
test intentionally SIGKILLs a throwaway process and was excluded to honor the operator's no-kill rule.
Compile, focused Ruff, diff checks, property gauntlet, state concurrency, and review policy are green.

The serializer then received one atomic ten-file allowlist with base and post-edit SHA guards. It
failed closed during `git add` with `rc=128` (`unable to create temporary file: Operation not
permitted`). No direct-Git fallback or override was attempted; no commit landed.

## Honest boundary

The exact incident replay is fixed-snapshot apparatus evidence. The sandbox blocked live `ps`, so no
claim is made about current machine RSS or live PID identities. No signal was sent, no process was
killed, no gate was overridden, and no score/promotion axis was created.

## Inbox

Consumed the per-arm directive through `2026-07-14T16:55:22Z`, including PID reuse, burst reserve,
throttle-coverage, fallback, atomicity, and reset requirements. Consumed fleet broadcast through
`2026-07-14T17:00:15Z`; no later relevant or stop directive was present at receipt creation.

## Recommended owner follow-up

The provenance owner may register `measured_growth_law_v1` from the findings SPEC in the canonical
equation registry. No change to the current governor behavior or witness DSL is needed for that
metadata step.

## HISTORICAL_PROVENANCE

Append-only session anchor. It does not replace the prior 2026-07-11 sub-floor phantom-reservation
fix; it distinguishes and closes the material-but-stable process class.
