# System memory governor measured-growth fix — Codex findings

Date: 2026-07-14  
Lane: `governor_measured_growth_fix`  
Receipt: `.omx/research/governor_measured_growth_fix_receipt_20260714.json`  
Classification: `research_only=true`, `score_claim=false`, `$0`, control-plane apparatus  
Pointer: unchanged; no archive, score, GPU launch, paid dispatch, process kill, or gate override.

## Root-cause confirmation

CONFIRMED from current code and predecessor history: the 2026-07-11 repair exempted only
unregistered ps-only matches below the 2 GiB material floor. A material unregistered process with no
recorded projection still reached `resolve_projected_peak_gib` with an unconditional
`current_rss + UNKNOWN_GROWTH_HEADROOM_GIB`, where the unknown headroom was 25 GiB. Two stable
material processes therefore manufactured 50 GiB of active-growth reservation indefinitely.

The defect was a prediction error, not a failure to count resident memory: `system_used_gib` already
contains the processes' current RSS. The old path added 25 GiB of assumed *future* growth for each
process without evidence that either process was growing.

## Landed fix

The impure process-scan edge now persists a deterministic, fcntl-locked, atomically replaced,
TTL-swept rolling RSS history in `.omx/state/system_memory_governor_rss_history.json`. State is
bounded to 128 PIDs and 64 samples per PID. `time.monotonic()` supplies the persisted time base.

The decision functions remain pure. `estimate_observed_remaining_growth_gib` consumes history and
returns either a bounded reserve or `None`; `resolve_projected_peak_gib` consumes that value without
performing I/O. No admissible history, corrupt state, clock reversal/future sample, stale series,
short span, missing kernel start identity, PID reuse, or non-finite value all produce `None` and
therefore preserve the old +25 GiB fallback.

Fresh-eyes constraints received through the live inbox were implemented as safety invariants:

1. **PID reuse:** history binds to PID plus a hash of kernel `ps lstart`, PPID, PGID, and command.
   Identity mismatch drops the old PID series and requires a new evidence window.
2. **Plateau then burst:** projection uses the maximum positive endpoint slope over recent
   subwindows, not a diluting full-window fit. A plateau never receives zero reserve; it retains
   `25 GiB * 2 s / 30 s = 1.6666666667 GiB`, one runtime poll at the fastest modeled allocation rate.
3. **Throttle coverage:** measured relaxation is passed to the pure resolver only when
   `_throttle_eligible` proves an own-group-leader process the runtime daemon can pause. A nonleader
   or otherwise unpausable material process remains at +25 GiB even with flat history.

Recorded projections, governed descendants, protection infrastructure, and the sub-2-GiB path keep
their original branch order and exact results.

## MEASURED fixed-snapshot apparatus replay

This replays the incident inputs exactly; it is not a live-machine RSS claim because the managed
sandbox denies `ps` process inspection.

| Case | Calculation | Projected used | Verdict |
|---|---|---:|---|
| Legacy / no history | `50 + 25 + 25 + 6` | `106.0 GiB` | `REFUSE` |
| Two stable eligible processes | `50 + 1.6667 + 1.6667 + 6` | `59.3333 GiB` | `ADMIT` |
| Real rising fixture (`5.1 -> 8.1 GiB`) | stable reserve + capped rising reserve | `82.6667 GiB` | `REFUSE` |
| Stable but non-throttle-eligible | old reserve retained | `+25.0 GiB` | unchanged |

The rising result is produced by persisted RSS samples and the real estimator, not a mocked growth
return. The history update/sweep occurs before the admission verdict, including on refusal.

## Safety proof

Admission is Layer 3 prediction. Runtime pressure control is Layer 2. Relaxation is legal only on the
intersection of material unregistered processes with the Layer-2 pause domain. Thus any process for
which the prediction is reduced has a structural runtime backstop: at the unchanged 2-second daemon
cadence, live available-memory pressure crosses the unchanged tier-scaled WARN/CRITICAL floor and
`decide_governor_action` selects reversible `pause`/SIGSTOP for the lowest-priority eligible job.

The proof does not claim that prediction is infallible. It makes an underestimate containable:

- observed reserve is clamped to `[G_poll, 25 GiB]`, so the fix can never reserve more than legacy;
- unknown evidence and unpausable processes remain at legacy +25;
- a real rising series can saturate +25 and still refuses in the incident snapshot;
- actual runtime growth is observed independently of the admission history and activates pause;
- no tier floor, runtime throttle, pending reservation, or recorded projection was weakened.

Verification inspected the actuator without sending a signal. The critical-policy decision was
`pause`; `pause_job` contains SIGSTOP and no SIGKILL. The daemon still starts with `govern=True` and
the 2-second cadence matches the modeled poll reserve.

## Three-clean-pass gauntlet

An initial review sequence was reset. Its third pass found a stale constants comment claiming all
material unregistered processes still received +25, contradicting the new throttle-backed exception.
The safety proof and resolver contract were corrected, round 3 was recorded `NOT_CLEAN`, and the
counter restarted on new bundle `2ccdefba59bfb491f0ed6393552e0808e90492f2d5fd83aadca9df68bcd97342`.

Post-reset clean rounds 4/5/6 are sealed as three consecutive clean passes:

1. **Pure-law/control-plane lens:** 25,000 deterministic monotone-safety cases; branch-order locks;
   195 non-actuating tests; compile, selected Ruff, and diff checks clean.
2. **State/TOCTOU lens:** identical state bytes; eight concurrent fcntl writers preserved; atomic
   temp cleanup; PID-reuse reset; TTL sweep; corrupt-state fallback; 195 tests clean.
3. **Runtime-backstop lens:** exact `106 -> 59.333` incident replay; real rising `82.667 REFUSE`;
   noneligible +25; runtime critical `pause`; daemon/floor cadence unchanged; 195 tests clean.

One existing test was deliberately deselected because it SIGKILLs its own temporary `sleep` process;
the operator forbade killing any process. All other tests in the four requested governor/TOCTOU/floor/
sampler suites passed: `195 passed, 1 deselected`. Review policy reports zero violations on all six
modified Python files.

## Triality and held equation registration

- **DSL:** no configuration or witness actuator was introduced; the governor consumes existing live
  process state. A new DSL lever would be an orphan and is intentionally absent.
- **DAG:** `.omx/research/sub015_DAG_governor_measured_growth_fix_20260714.md` lands the producer,
  safety gates, and admission/runtime consumers.
- **Canonical equation SPEC — HELD:** provenance owns `src/tac/canonical_equations/`, so no write was
  made there. Register/refine this law under that owner:

  `G_remaining(p,t) = clamp(max(G_poll, H * max_i((rss_t-rss_i)/(t-t_i), 0)), 0, G_unknown)`

  with `G_unknown=25 GiB`, `G_poll=25*(2/30)=1.6666666667 GiB`, fresh same-start-identity history,
  and admissibility predicate `material_unregistered AND throttle_eligible`; any failed evidence
  predicate maps to `G_unknown`.

## Integration / six-hook disposition

This is control-plane infrastructure rather than a witness representation. Sensitivity-map, Pareto,
and bit-allocator hooks are nonbinding. The cathedral/autopilot consumer is direct: safe-run admission
calls `list_tracked_jobs` and consumes the bounded projection. Continual-learning state is this typed
receipt plus regression suite. The disambiguator is explicit and executable: measured bounded trend
versus fail-conservative +25 fallback, with throttle eligibility deciding whether measurement may be
used. No score or frontier conclusion follows.

## Landing status

The canonical serializer was invoked once with one atomic ten-file `--files` allowlist, exact HEAD
base hashes for the six edited files, `base=new` for the four artifacts, and exact post-edit hashes
for all ten paths. `git add` failed closed with `rc=128`: `unable to create temporary file:
Operation not permitted`. No direct-Git fallback, override, or second staging path was attempted.
The reviewed workspace bytes and standalone DAG remain the handoff authority; no commit landed in
this sandbox.

## HISTORICAL_PROVENANCE

Append-only finding. It refines but does not overwrite the 2026-07-11 sub-floor phantom-reservation
repair. The earlier repair remains valid for sub-2-GiB transient matches; this landing closes the
distinct material-but-stable class.
