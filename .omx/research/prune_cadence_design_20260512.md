# Prune cadence design — dispatch claims ledger (T1-E follow-up)

**Date:** 2026-05-12
**Source:** T1-D `--apply` execution turn (operator-approved cluster wave-up)
**Sister landing:** `feedback_state_hygiene_gc_and_prune_landed_20260512.md` (Cluster 3, commit `92aba3ca`) shipped the `prune` subcommand on `tools/claim_lane_dispatch.py` and pruned the production ledger once (1338→537 rows; 545,900 B → 220,253 B = 60% reduction).
**Scope:** design memo only this turn; implementation deferred until operator routes.

## Status quo (post Cluster 3 commit `92aba3ca`)

The `prune` subcommand is fully built, tested (13 tests), and one-shot ran on the production ledger. Cadence question: how often should `prune` re-run going forward?

| Property | Current value |
| --- | --- |
| Live ledger `.omx/state/active_lane_dispatch_claims.md` size after first prune | 220,253 B (537 rows) |
| Archive `dispatch_claims_2026-05.md` size | 323,898 B (801 rows) |
| Estimated claim rate (≈ derived from 1338 rows in ~30 days) | ~44 rows/day = ~315 rows/week |
| `--terminal-age-days` default | 7 |
| Steady-state live ledger size at 7-day retention | ~315 rows × ~407 B/row ≈ 128 KB (active + recent terminal) |

Without prune, the ledger grows ~13 KB/day; in 30 days it would re-bloat back to ~540 KB. Bounded growth requires a cadence.

## Cadence options

### Option A — On-demand operator command (current state)

**Trigger:** human operator runs `prune` whenever they notice the ledger is large.

- **Pros:** zero new infrastructure; zero failure modes (the operator chooses); aligns with the "every action is operator-routable" CLAUDE.md philosophy; the `prune` rendered the live ledger small enough that "I'll get to it later" is acceptable for weeks.
- **Cons:** prone to forgetfulness — Cluster 3's first prune ran AFTER 30 days because nobody remembered; ledger will silently bloat between operator-attentive windows; bypasses the principle that bounded state is a contract, not a wish.

### Option B — Pre-dispatch hook (run `prune` IF stale > 24h before each `claim` call)

**Trigger:** `tools/claim_lane_dispatch.py::_claim()` consults a sentinel file `.omx/state/.last_prune_utc` at the start of every claim. If `now - last_prune_utc > 24h`, prune first, then proceed with the claim.

- **Pros:** amortizes prune cost over claims (one ~150 ms scan per ~10-50 claims/day = negligible per-claim overhead); no external scheduler; runs ONLY when claims happen anyway (skips weekends with no activity); naturally adaptive to workload — heavy days → daily prunes, idle days → no prunes.
- **Cons:** ~50 LOC patch including the sentinel-file fcntl-lock dance; complicates the `_claim` codepath that is currently very narrow; adds a "hidden" side-effect to claim (operator may be surprised that a claim call also archived 800 rows).
- **Implementation sketch:** new helper `_maybe_prune_if_stale(claims_path, max_age_hours)` called at the top of `_claim()` BEFORE the lock-acquire. The helper reads/writes `<claims_path>.last_prune_utc` under its OWN fcntl lock (separate from the claim lock to avoid deadlock when prune subprocess is invoked). LOC estimate: 30-40 in helper, 5-10 wire-in.

### Option C — Post-completion hook (run `prune` after each terminal-status `claim --force` call)

**Trigger:** every `claim --force --status completed_*` / `failed_*` / `stopped_*` call appends a terminal row AND, if `now - last_prune_utc > 24h`, runs prune at the tail.

- **Pros:** prune amortized over terminal closures (which is exactly the workload that produces stale rows); guarantees post-terminal cleanup; same "no external scheduler" benefit as Option B.
- **Cons:** terminal closures are LESS frequent than claims (every claim has matching terminal eventually, but terminals run later); during long-running dispatch waves, ledger can grow between terminals; same hidden-side-effect surprise as Option B.

### Option D — Scheduled cron / launchd / systemd timer

**Trigger:** OS scheduler fires `python tools/claim_lane_dispatch.py prune` daily/weekly/monthly.

- **Pros:** standard pattern; runs even on idle days; operator can `cron-stop` once and the ledger stays bounded forever; visible in `crontab -l` audit.
- **Cons:** introduces an external dependency (cron/launchd configuration); cron failures are silent unless wired to email/log; the macOS dev machine + remote tailscale fleet machines + Modal/Vast.ai containers all have different scheduler stories — picking ONE consistent surface is non-trivial; operator must remember to migrate the schedule when machines change.
- **Implementation sketch:** add `scripts/setup_prune_cadence.sh` that installs a launchd plist on macOS / cron entry on Linux; document in CLAUDE.md "Tooling — non-negotiable". LOC estimate: 30-50 in the setup script + man-page text.

## Recommendation: **Option B (pre-dispatch hook IF stale > 24h)**

**Rationale:**

1. **No new scheduling surface.** The operator already runs `claim_lane_dispatch.py` before every paid dispatch — that's the natural trigger point.
2. **Cost amortization.** ~150 ms scan per ~10-50 claims/day = sub-1 ms per claim. The Claim-Cluster operator-bound flow is unaffected.
3. **Bounded growth contract.** The 24h-stale-check guarantees the ledger never exceeds ~13 KB of un-archived growth per day. Sister Cluster 3 prune showed this is well within the operator-comfort range.
4. **Self-healing.** If the operator skips dispatches for a week (vacation), the FIRST claim after returning auto-prunes. No reminder needed.
5. **Composable with Option D later.** If a future need for guaranteed-cadence emerges (e.g. CI sentinel monitoring), Option D layers on top of Option B without conflict.

**Rejected alternatives:**

- **Option A (current)** lacks a bounded-growth contract; the 30-day first-prune anti-pattern WILL recur.
- **Option C (post-terminal)** has a longer feedback loop than Option B and doesn't help on long-running waves.
- **Option D (cron)** requires multi-platform scheduler config that the operator doesn't currently maintain.

## Implementation plan (DEFERRED — operator routes)

1. **New helper `_maybe_prune_if_stale(claims_path, max_age_hours=24)`** in `tools/claim_lane_dispatch.py`:
   - Acquires its OWN fcntl lock on `.omx/state/.last_prune_check.lock` (separate from the claim lock).
   - Reads `.omx/state/.last_prune_utc` sentinel.
   - If `now - last_prune_utc > max_age_hours`, calls `_prune(...)` programmatically (NOT subprocess) with `--terminal-age-days 7 --apply`; updates the sentinel.
   - On any exception, swallows + logs to stderr (NEVER fails the claim — prune is a side-effect, not the primary operation).
2. **Wire into `_claim(args)`** at the very top, BEFORE the claim-lock acquisition. Add `--skip-prune-cadence` CLI flag for emergency bypass.
3. **Test** the helper with `tests/test_claim_lane_dispatch_prune_cadence.py`: stale-sentinel triggers prune; fresh-sentinel skips; concurrent-claim race serializes; missing-sentinel triggers prune; corrupt-sentinel-content triggers prune (treat as never-pruned).
4. **Catalog #155 sister-check (optional):** `check_claim_helper_uses_prune_cadence` — refuses any new dispatch helper that calls `claim_lane_dispatch.py claim` without ensuring prune cadence is wired (or carries waiver). Probably overkill; defer unless duplicate-dispatch-helpers appear.

**Estimated LOC:** ~50 production + ~80 tests. Half-day dev. Operator-gated.

## Backout plan

If Option B causes claim slowdown or unexpected side effects, the operator can:
- Set `--skip-prune-cadence` per-call (no LOC change).
- Set env-var `PACT_DISABLE_PRUNE_CADENCE=1` (suggest as part of the implementation).
- Roll back by reverting the `_maybe_prune_if_stale` wire-in line; the helper itself stays available for manual invocation.

## Cross-references

- **Cluster 3 landing:** `feedback_state_hygiene_gc_and_prune_landed_20260512.md`
- **Sister catalog gate:** Catalog #154 (`check_experiments_results_gc_helper_is_canonical`) — established the "canonical helper" pattern Option B's auto-trigger embodies.
- **CLAUDE.md "Operator gates must be wired and used":** Option B is the wire-in for the prune helper into the natural operator flow.
- **CLAUDE.md "Subagent coherence-by-default":** the auto-prune hook is the agent-binding contract that prevents the 30-day-bloat anti-pattern.
