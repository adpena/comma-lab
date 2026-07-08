# WALL-CLOCK-DEFAULT-ON + PERF-ENV CLASS GUARD (2026-07-08)  [no-triality]

Operator (2026-07-08): *"The wall clock stuff should be default on always and also that shouldn't
have had to be caught manually."* Two structural fixes on top of the compute audit (commit 8d9dabc92,
memo `v7_compute_exploitation_audit_20260708.md`, which built the advisory wall-clock projection +
fixed the v7 perf-env orphan BY HAND). **MEANS, not ends** — nothing here moves the pointer (0.19110);
only a byte-closed n600 exact row < 0.19110 does.

**STORES CONSULTED:** the compute-audit memo + its projection module (`scorer_throughput_gate`); the
launcher gate chain (`_run_throughput_gate`, `verify_perf_env`, the DSL-config-gate migration pattern
it mirrors); `typed_config` (the REQUIRED-field + PERF_ENV_PREFIX SoT); `_build_crucible_v7` /
`_attach_dsl_manifest` (the two TypedWitnessConfig constructions); MEMORY L45 (verify THROUGHPUT +
wall-clock, not flags) · L70 (fused-R) · value-provenance ladder.

## FIX 1 — wall-clock gating is DEFAULT-ON (no opt-in flag)
- **REQUIRED typed field** `TypedWitnessConfig.wall_clock_budget_days: Provenanced` (no default →
  forgetting is a pydantic ValidationError; the schema makes the orphan structurally impossible).
  Validated POSITIVE and NOT `HARDCODED-WITH-WAIVER` (a hand-picked budget defeats the anchor ceiling).
  `budget_days_value()` exposes the float.
- **DERIVED, never hand-picked** — the single SoT is
  `scorer_throughput_gate.derive_wall_clock_budget_days(epochs)`:
  `budget_days = project_wall_clock_days(RUN1_MEASURED_MIN_PER_EP, epochs) × WALL_CLOCK_SLACK_FACTOR`
  `           = (3.1 min/ep × epochs / 1440) × 1.15`.
  At **epochs=3000 ⇒ 6.458 × 1.15 = 7.427 days** (the "~7 days"). SLACK 1.15 tag: 3.65 incl-startup /
  3.1 steady (operator-cited) = 1.18 startup-amortization ceiling; 1.15 keeps a slim thermal/jitter
  headroom so the gate stays a REAL refuse (>15% slower than the run-1 anchor ⇒ REFUSE), not a rubber
  stamp. RE-DERIVE if the anchor changes or a lever batches the forward (micro-batch / τ-advance
  changing the effective per-ep count ⇒ a different min/ep ⇒ a different ceiling).
- **Launcher admission** (`_run_throughput_gate` sub-part 4): projects the MEASURED SegNet bench ×
  epochs and REFUSES **rc=8** when over the resolved budget. Budget resolution
  (`resolve_wall_clock_budget`, pure): `--accept-wall-clock` operator override (stamps
  `wall_clock_accept.txt`, LOUD) > config-declared (typed DERIVED) > **launcher-derived anchor
  fallback** (so a legacy non-declaring WitnessConfig STILL gets a default-on refuse — the gate never
  silently disappears). The advisory PRINT stays.
- **Escape hatch** `--accept-wall-clock <days>` replaces the removed opt-in `--wall-clock-budget-days`.
- **Migration:** `derive_crucible_v7_config` (and the v6 `_attach_dsl_manifest` typed gate) now DECLARE
  the derived budget (in `witness_autoconfig` — see the sibling-absorption note below).

## FIX 2 — perf-env CLASS guard ("shouldn't have been caught manually")
- `typed_config.REQUIRED_PERF_ENV` is DERIVED by parsing `PERF_ENV_PREFIX` (`_parse_perf_env`) — the SoT
  is the prefix, never a duplicate list; a second var added to the prefix is picked up automatically.
- `missing_perf_env_vars(launch_sh_text)` (pure) returns the `NAME=VALUE` assignments absent from the
  emitted launch.sh (a bare NAME or wrong value does NOT satisfy it).
- **Launcher step (b-perf)** asserts the EMITTED launch.sh carries every required var → **REFUSE rc=9**
  naming the missing var. The v7 orphan the audit fixed by hand (a config-path dropping the ~17x
  prefix) is now a STRUCTURAL refuse, not an audit finding. Fail-open (loud) only on helper-import error
  (the post-spawn perf-env LOG check remains the backstop).
- **Drift-IMPOSSIBLE:** `witness_autoconfig.to_command` now CONSUMES the ONE
  `typed_config.PERF_ENV_PREFIX` constant (`wac.PERF_ENV_PREFIX is typed_config.PERF_ENV_PREFIX`) — both
  launch paths share the object, so the prefix cannot drift (was: two literals kept equal by a test).

## FIX 3 — throughput-vs-budget coupling
Framed by `implied_segnet_ms_ceiling(budget_days, epochs)` (inverts `project_min_per_ep`). A bench that
PASSES the 700ms absolute throughput gate (env present, "fast") but is slower than the budget-implied
ceiling STILL projects over-budget ⇒ REFUSE — catching a NON-env perf regression (kernel not loading,
wrong device, thermal throttle). Mathematically this IS the FIX-1 projection>budget refuse; exposed as
an explicit ceiling for legibility + the coupling test.

## TESTS (`test_wallclock_default_on_perfenv_guard.py`, 21) + 2 helper migrations
required-field ValidationError · positive/derived-not-hardcoded validators · budget in config hash ·
derived-budget math == anchor×epochs×slack · v7 declares ~7.4d · resolver priority (accept/declared/
fallback/none) + override stamp · gate rc=8 over-budget (fast-but-over, fix #3) · rc=0 under-budget ·
accept-override stamps + proceeds · implied-ceiling inverts projection · REQUIRED_PERF_ENV derived from
prefix · missing-var named (bare/wrong-value not satisfied) · prefix-parity (same object, both
to_command paths) · emitted v7 command satisfies the guard. Full suite 159 passed.

## SIBLING-ABSORPTION NOTE (coherence)
My `witness_autoconfig` edits (the 2 TypedWitnessConfig budget declarations + the PERF_ENV_PREFIX SoT
import + `to_command` consuming it) were ABSORBED into the REVISIONS-B sibling commit `3563b9c9b`
(`git add`-the-whole-file in the shared worktree) — they are LANDED in HEAD and correct, but attributed
to that commit. I therefore commit ONLY my clean surfaces (typed_config + scorer_throughput_gate +
launcher + tests) and DO NOT re-stage `witness_autoconfig` (it now also carries the τ-advance sibling's
uncommitted `--tau-advance-mode` edits — left for that sibling). No signal lost; recorded here so the
attribution is auditable.
