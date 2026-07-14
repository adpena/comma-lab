# FEED-governor-measured-growth-fix-20260714 — control-plane DAG

`research_only=true` · `score_claim=false` · `pointer_moved=false` · `$0 LOCAL` · no process actuation

## Executable dependency graph

```text
memory_guard ps snapshot
  ├─ pid / ppid / pgid / RSS / command
  └─ kernel lstart identity
          ↓
material unregistered candidate discovery
          ↓
fcntl lock → TTL/PID-identity sweep → bounded append → fsync → atomic replace
          ↓
PURE observed-growth estimator
  ├─ insufficient / stale / skew / corrupt / PID reuse ──> UNKNOWN = +25 GiB
  ├─ plateau ──> nonzero poll reserve = 1.6666666667 GiB
  └─ rising ──> max recent subwindow slope × bounded horizon, cap +25 GiB
          ↓
throttle-eligibility safety gate
  ├─ own detached group leader ──> measured reserve may flow
  └─ noneligible / nonleader ──> UNKNOWN = +25 GiB
          ↓
PURE projected-peak resolver
          ↓
Layer-3 admission decision
          ↓ if real pressure rises
unchanged Layer-2 tier-scaled runtime throttle ──> reversible pause/SIGSTOP
```

## Canonical nodes and gates

| Node | State | Producer | Consumer | Fail-closed rule |
|---|---|---|---|---|
| `rss_process_identity_v1` | ACTIVE | `memory_guard.sample_processes` | history edge | missing `lstart` => no relaxation |
| `rss_history_v1` | ACTIVE | governor scan cadence | pure estimator | lock/I/O/corrupt/stale/future => +25 |
| `observed_remaining_growth_v1` | ACTIVE | pure estimator | projected-peak resolver | `<2` samples / `<30s` span => +25 |
| `material_plateau_poll_reserve_v1` | ACTIVE | pure estimator | admission | never zero; `1.6666666667 GiB` |
| `throttle_coverage_precondition_v1` | ACTIVE | `_throttle_eligible` | history-to-resolver edge | noneligible => +25 |
| `runtime_pressure_backstop_v1` | UNCHANGED | pressure classifier | reversible actuator | unchanged tier floors and 2s cadence |
| `measured_growth_law_v1` | HELD_PROVENANCE_OWNER | canonical-equation SPEC | apparatus registry | do not write owned equation surface |

## Receiver evidence

- Legacy/no-history incident projection: `106.0 GiB`, `REFUSE`.
- Stable measured incident projection: `59.333333333333336 GiB`, `ADMIT`.
- Real rising-history incident projection: `82.66666666666667 GiB`, `REFUSE`.
- Non-throttle-eligible material process: `25.0 GiB` growth reserve, unchanged.
- Runtime critical decision: `pause`; verification sent zero signals.

Authority is a deterministic local fixed-snapshot control-plane replay. Live process enumeration was
unavailable in the managed sandbox. No contest score, MPS authority, archive, or promotion claim.

## Triality / consumers

- DSL: none required; this is an existing governor state path, not a new witness lever.
- Equation: `measured_growth_law_v1` HELD for the provenance owner with the SPEC in the findings memo.
- DAG: this FEED is the durable producer/gate/consumer graph.
- Autopilot: safe-run admission directly consumes `list_tracked_jobs` projections.
- Continual learning: typed receipt + regression suite; old sub-floor repair remains a separate valid ancestor.
- Disambiguator: observed bounded trend only on the throttle-backed domain, otherwise +25.
