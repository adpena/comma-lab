# Governor phantom-reservation false-refuse — mechanism, fix, control-plane-safety argument (2026-07-11)

**Class:** apparatus / control-plane. **Pointer:** 0.19108282 [contest-CPU] **UNMOVED** — no score
claim. This is MEANS (a governor admission-gate correctness fix), not a witness/score result.

## The observed bug (MEASURED, this session)
An operator-GO'd witness resume was REFUSED by the system admission gate
(`tools/spawn_durable_daemon.py` → `system_memory_governor.live_admission_decision`, the SUM-over-RAM
crash guard) with `active_jobs=9`, `active-growth 200.0 GiB`, projected system-used 270 GiB vs adaptive
ceiling ~91 GiB — while GROUND TRUTH was **30.7 GiB used / ~97 GiB available / ZERO live training
processes** (`ps` for `train_levelset`/`train_witness` returned nothing; `--status` showed every
registered daemon `actual=DEAD`). Across two refused attempts `active_jobs` grew 6→9 and projected
growth 125→200 GiB. `--reconcile` marked 2 dead-registry daemons stopped but did **not** reduce
`active_jobs`. Net: a nearly-idle machine false-refused a legitimate launch from phantom reservations.

## Root-cause mechanism (found, not guessed — reproduced live)
Three independent contributors, isolated by reading the read path (`list_tracked_jobs` →
`resolve_projected_peak_gib` → `sum_active_growth_headroom_gib`) and the registry:

1. **PRIMARY — over-broad ps-pattern matching charged phantom +25 GiB each.** `OUR_JOBS_PATTERN` is a
   BROAD substring regex (`byte_close|inflate\.py|evaluate\.py|train_witness|descent_probe|…`) whose
   real purpose is THROTTLE candidate discovery. `list_tracked_jobs` also unions it into the ADMISSION
   projection, and `resolve_projected_peak_gib` charged every UNREGISTERED ps-match
   `current + UNKNOWN_GROWTH_HEADROOM_GIB` (+25 GiB) of growth headroom. But a match only requires the
   token to appear ANYWHERE in argv — so a `grep`/`ugrep`/`rg` over the source, an editor with the file
   open, a `python -c` mentioning the script, the launch pipeline itself, a short-lived byte-close /
   inflate probe, or a sibling build/measurement agent all match. **Reproduced live 2026-07-11:** a lone
   incidental `ugrep … train_witness…|byte_close|inflate\.py|…` matched the pattern. ~8 such incidental
   matches × 25 GiB = the phantom ~200 GiB. These are neither dead-registry rows nor reservations, so
   `--reconcile` could not touch them — exactly why it "marked 2 dead but active_jobs stayed 9."

2. **The admission path never actually auto-reconciled.** `reconcile_dead_daemons`' docstring claimed
   "the admission path can auto-reconcile before EVERY governed launch decision," but nothing called it
   there (only `witness_memory_preflight`). Dead `running` rows were skipped in projection only
   incidentally (no `ps` sample), but the store was never converged, so `--status`, `owned_pids`, and
   governed-descendant detection stayed stale.

3. **`--reconcile` ignored the pending-reservation store the gate counts.** `reconcile_dead_daemons`
   only marked dead `running` rows; it never swept stale `admitting` reservations (a crashed launcher's
   phantom growth headroom). So the store the gate reads and the store reconcile cleaned were different.

## The fix (three parts; ground-truth, safety-preserving)

### Part A — material-RSS floor for unregistered ps-only matches (`tools/system_memory_governor.py`)
New constant `MATERIAL_UNREGISTERED_RSS_FLOOR_GIB = 2.0` (== `ABS_MIN_SAFETY_FLOOR_GIB`, the
jetsam-avoidance minimum). `resolve_projected_peak_gib` gains a keyword-only `unregistered_ps_only`
flag: an unregistered ps-only match whose CURRENT RSS is below the floor is charged **zero growth**
(projected_peak = current_rss) — identical treatment to the existing protection-infra / governed-
descendant exemptions. `list_tracked_jobs` passes `unregistered_ps_only=(not rec)`. All other paths
(registered running rows, pending reservations, recorded projections, infra, descendants) are
**bit-identical** — the flag defaults `False`.

Why this is ground truth, not a heuristic: a sub-2-GiB process cannot itself drive a 128-GiB SUM crash,
and whatever RSS it does hold is already inside the vm_stat `used` truth the gate anchors on. Charging it
a speculative +25 GiB manufactured the phantom.

### Part B — the admission path auto-reconciles before projecting (`tools/spawn_durable_daemon.py`)
`_do_start` now calls `reconcile_dead_daemons(verbose=False)` inside the held admission `_registry_lock()`
(re-entrant), replacing the standalone stale-pending sweep. The registry the gate counts is converged to
kernel truth before the projection.

### Part C — `reconcile_dead_daemons` converges the FULL counted store, idempotently
It now does TWO reconciliations in ONE locked transaction: (1) mark dead `running` daemons `stopped`
(existing), AND (2) DROP stale `admitting` reservations (crashed launcher: no pid, missing/old
`reserved_ts`), reusing `_sweep_stale_pending_rows` as the single freshness predicate. FRESH in-flight
reservations (concurrent launcher mid-Popen) are preserved — the TOCTOU close is intact. Idempotent:
running it twice reconciles nothing the second time.

## Before / after (synthetic reproduction of the exact observed numbers)
Nearly-idle machine (30.7 GiB used, 128 total), 8 incidental ps-token matches (~0.2 GiB each), a 40-GiB
job, via the PURE admission path:

| | active-growth | projected system-used | verdict |
|---|---|---|---|
| BEFORE (buggy +25 each) | **200.0 GiB** | **270.7 GiB** | REFUSE |
| AFTER (ground-truth fix) | **0.0 GiB** | **70.7 GiB** | **ADMIT** |

The BEFORE row reproduces the operator's observed `active-growth 200.0 GiB` / projected 270 GiB REFUSE.
A refused (and re-refused) launch now also leaves **zero net reservation** (enforce-mode gate returns
before any `admitting` row is written) — no accumulation on retry.

## Control-plane-safety argument (real over-commit STILL refused; #409/#172 respected)
- **The SUM-over-RAM crash guard is NOT weakened.** Every genuine heavy launch goes through the governed
  path and REGISTERS (a `running` row with a `projected_peak`, or a pending reservation) → the recorded
  projection wins → the job is fully charged. Two concurrent heavy jobs summing over the ceiling are
  still REFUSED (`test_admission_refuses_sum_over_ceiling_the_crash` unchanged; new
  `test_registered_heavy_job_low_rss_still_fully_charged`, `test_materially_resident_unregistered_job_still_charged`).
- **The relaxation is scoped to UNREGISTERED, sub-2-GiB, ps-only matches** — by construction not a
  governed heavy job (those register) and not a crash driver. Their live RSS remains in the `used`
  baseline; the runtime WARN/CRITICAL guard-bands + `safe_run` per-arm cap protect any burst DURING a run.
- **No process is ever signalled or killed by this change.** It only alters projection arithmetic and
  reconciles the JSON registry. The control plane cannot be paused/killed by an accounting fix
  (#409/#172 molt CP-kill class not re-introduced).

## Self-protection (two-landing discipline)
Fix + regression tests that refuse re-introduction (the appropriate self-protect form for a numeric
accounting invariant — the STRICT-preflight/AST-scan two-landing is for code-surface bug classes; this
is a pure-function property, so property/regression tests are the gate). **No new catalog # claimed**
(claiming a STRICT-preflight row for a non-scannable arithmetic invariant would be a fake catalog row).
New tests:
- `test_system_memory_governor.py`: `test_resolve_unregistered_ps_only_below_floor_zero_growth`,
  `…_at_or_above_floor_still_charged`, `test_resolve_default_and_registered_paths_unchanged_by_fix`,
  `test_incidental_ps_token_match_charges_zero_phantom_growth` (the reproduced false-positive),
  `test_materially_resident_unregistered_job_still_charged`,
  `test_registered_heavy_job_low_rss_still_fully_charged`.
- `test_spawn_durable_daemon_memguard.py`: `test_reconcile_drops_stale_pending_reservation`,
  `test_reconcile_is_idempotent`, `test_refused_admission_leaves_zero_net_reservation` (retry-safe).
All 83 tests in the three governor/reservation suites pass over 3 deterministic clean passes; ruff-F clean.

## Not a witness triality leg
This is an apparatus invariant (governor accounting), not a witness/score law — it does NOT belong in
`tac.canonical_equations` (that registry is for score-affecting witness laws; a governor-accounting row
would miscategorize). The invariant is encoded in the pure functions + the regression tests + this memo.

## Out of scope (pre-existing, unrelated)
`test_nerv_long_training_campaign_admission::…_accepts_snerv_with_launch_gate` fails on the CLEAN tree
(verified by stash) — a different subsystem (experiment-queue SSD storage admission), touches none of
the code changed here. Not addressed.
