# Launch pre-GO worklist — 2026-07-13

Checkpoint ID: `launch_prego`  
Status: **PREPARED / NO LAUNCH / LOCAL $0 ONLY / POINTER DELTA ZERO**  
Verdict scope: the two recompiled n600 tickets at repository HEAD
`413c9e6c5328ba1e3f7096baa734e9cabee547d1` and the live host snapshots recorded below. Dynamic
memory admission must be rerun immediately before any later action.

## Executive gate state

| REFUSE | State | Exact result |
|---|---|---|
| R1 memory | **CLEARED AS AN OPERATOR CHOICE PACKET** | Full: 71.54 GiB/run, historical 119.4 > 97.7 GiB and live 114.5 > 100.1 GiB system REFUSE. Trimmed: 24.48 GiB/run, DERIVED historical-baseline counterfactual 72.28 < 97.7 and live 67.3 < 100.2 GiB admission. |
| R2 B=2 n600 custody | **PREPARED, NOT RUN** | Exact B=1/B=2 read-only-checkpoint foreground command landed. Full variant does not fit; trimmed preview admitted the 24.48 GiB base projection but left only 5.6 GiB live headroom and the 30.78 GiB adjusted projection exceeded that snapshot's 30.1 GiB training budget. R3 also blocks the required SSD output. |
| R3 SSD root | **BLOCKED** | Vertigo parent exists with 827,380,576,256 free bytes, but canonical `--create` returned `mkdir_failed:PermissionError`; APDataStore is absent; local fallback stayed disabled. |
| R4 telemetry producers | **BLOCKED (premise falsified narrowly)** | `pact.causal_manifest.v1` is landed and detected. It does not emit D-A's eight fields or D-B's engagement schema/callbacks. Both launch dry-runs therefore retain four blockers and return rc=11 before spawn. |

Recompiled overall preflight state: **`REFUSE_RC11_NO_SPAWN` for both variants**. The trimmed ticket
passes the memory-admission leg at its recorded snapshot; neither ticket passes the complete launch
contract.

## Containment receipt

- No trainer was launched.
- No durable daemon, remote provider, GPU dispatch, paid action, or eval was attempted.
- The only launcher invocations used `--dry-run`; both ticket invocations stopped at rc=11.
- The R2 command preview also used `--dry-run`; it materialized and validated the bounded command but
  did not read or mutate the source checkpoint.
- No run directory on either SSD was created or mutated. The failed storage attempt wrote only the
  small local machine-readable receipt.
- Sibling work was not absorbed. In particular, `compander_build` remains next-ticket business.

## R1 — waterfill-derived memory variants

The canonical waterfill solver still selects `micro_batch=1`, `verdict_batch=16` for both variants;
the B knob is excluded until an uncontended n600 measurement exists. This is not overridden merely
because the typed treatment requests B=2.

| Leg | Full FreSh/self-orient | Trimmed compliant | Delta |
|---|---:|---:|---:|
| Fixed overhead | 15.00 GiB | 15.00 GiB | 0 |
| Coordinate-feature MLX cache | 47.13 GiB | 0.07 GiB | -47.06 GiB |
| GT | 3.41 GiB | 3.41 GiB | 0 |
| Verdict transient | 6.00 GiB | 6.00 GiB | 0 |
| Projected peak | **71.54 GiB** | **24.48 GiB** | **-47.06 GiB** |
| Realized-spike-adjusted peak | 77.84 GiB | 30.78 GiB | -47.06 GiB |

The sole trim is the typed `FreshFrequencyShift()` lever and its emitted `--fresh-init`,
`--self-orient`, directional-frequency, and FreSh search flags. The V9 parent is the GO'd
self-orient-OFF configuration. The prior owed-16 receiver surface measured the underlying directional
transfer approximately zero while measuring approximately 47 GiB of cache cost. That supports the
memory trim. It does **not** measure the FreSh cold-start treatment effect, so the score impact of the
trim is `UNKNOWN`, not zero.

The two projections required by the operator choice are:

1. **Full stack:** historical ticket snapshot 47.8 + 71.54 = **119.34 GiB**, recorded as 119.4 GiB,
   above the 97.7 GiB adaptive ceiling. Recompiled launcher snapshot: **114.5 > 100.1 GiB**.
2. **Trimmed compliant:** on the same historical baseline, DERIVED 47.8 + 24.48 = **72.28 GiB**,
   below 97.7 GiB. Recompiled launcher snapshot: **67.3 < 100.2 GiB**.

The adaptive ceiling moved between reads as host pressure changed. The ticket preserves both the
historical decision point and exact live dry-run observations instead of smoothing them together.

## R2 — B=2 n600 RSS custody

The preserved source is:

- `experiments/results/v9_cgauge_432_coherent_arm_20260711/levelset_resume_state.npz`
- SHA-256 `17c4b4765370ee39d34919805a1e45f6c88ce8b213c70aaaf18100cbf58881e2`
- `__resume_epoch=275`, optimizer present, self-orient OFF, `in_feat=80`, subpixel-boundary weight 0.

The prepared command uses the checkpoint-compatible `v9_cgauge_432` typed config, not the full FreSh
ticket. It materializes matched B=1 and B=2 epoch-276 commands through `launch_witness_run.py
--dry-run`, then runs each foreground under `safe_run` with `TAC_MEM_PROBE=1`, a 1,200-second bound,
and the source SHA checked before and after. This avoids the ticket-blocker circularity while preserving
the exact n600 treatment surface.

It was not run because the compliant SSD root is unavailable and because the latest command preview
had only 5.6 GiB system headroom after admitting the unadjusted 24.48 GiB projection. At that snapshot,
the waterfill's 30.78 GiB adjusted peak was 0.68 GiB above the 30.1 GiB training budget. Starting an
unknown B=2 treatment there would not be an honest `$0` fit.

Prepared artifact:
`experiments/results/next_launch_all_levers_ticket_trimmed_20260713/r2_b1_b2_n600_measurement_after_GO.sh`.
Expected custody is B=1/B=2 `launch.sh`, `run.log` with `mem_probe` and `safe_run` peak RSS, final
atomic checkpoints, and unchanged source-checkpoint SHA. The resulting measured curve points must be
fed back into `tools/memory_waterfill_config.py`; chat prose cannot clear R2.

## R3 — storage waterfall

The canonical plan checked Vertigo first, then APDataStore, with 1,026,048,000 requested bytes and a
40 GiB reserve:

- `/Volumes/VertigoDataTier/pact` exists and reported 827,380,576,256 free bytes.
- Creating `/Volumes/VertigoDataTier/pact/next_launch_all_levers_20260713` failed with
  `mkdir_failed:PermissionError`; the write probe therefore failed.
- `/Volumes/APDataStore` is not mounted.
- Local-disk fallback was not enabled.

Both ticket directories contain refreshed `storage_preflight.json`. R3 can clear only when an
authorized host context creates and write-probes the selected workload root, after which the same
preflight must be rerun. The available-byte count alone is not a clearance.

## R4 — producer verification and recompile

The ticket compiler now detects producer readiness from the shared trainer rather than assuming it:

- D-A requires `witness_component_wallclock.v1` plus all eight exact field names:
  `teacher_forward_s`, `teacher_backward_s`, `witness_forward_s`, `witness_backward_s`,
  `realized_R_s`, `verdict_s`, `checkpoint_io_s`, and `epoch_total_s`.
- D-B requires `sps_gradient_role_conflict_engagement.v1` plus explicit
  `temporal_screw_engaged` and `phase_advection_engaged` callback surfaces.

Fresh source inspection finds the causal writer default-on at trainer construction and boundary calls
at baseline, resume, verdict/stage, and checkpoint/final surfaces. Its schema is
`pact.causal_manifest.v1`; its rows freeze treatment, data order, state, action, outcome, and artifact
custody. The exact D-A schema/fields and D-B schema/triggers do not occur in the trainer. Existing
`--profile-timing` still fuses forward/backward/optimizer into `step_s`; generic gradient interaction
rows fire at seg-form boundaries, not screw/phase engagement.

Therefore the causal landing clears only the causal dependency. Treating its aggregate boundary rows
as D-A/D-B producers would be a false clearance. Both recompiled manifests retain:

1. `D_A_EXACT_COMPONENT_TIMERS_MISSING`
2. `D_B_EXACT_ENGAGEMENT_HOOK_MISSING`
3. `MEMORY_WATERFILL_B2_UNMEASURED_N600`
4. `SSD_WORKLOAD_ROOT_MISSING`

Full dry-run: 21/21 expected levers, 231/231 real argparse flags, memory REFUSE, rc=11.  
Trimmed dry-run: 20/20 expected levers, 219/219 real argparse flags, memory ADMIT at snapshot, rc=11.

## Muon consistency at HEAD

Muon launcher wiring is consistent with current `tools/launch_witness_run.py` and commit `b6783d45dc`.
Both generated tickets contain the ordinary V9 Muon surface:

- `--muon-lr 0.002`, `--muon-momentum 0.95`, `--muon-ns-steps 5`
- `--muon-warm-start-momentum`, `--muon-lr-final-frac 0.1`
- `--muon-start-event powerlaw_meat` with ep726 as the fail-safe cap

The sibling FiLM polar-chart SPEL finisher remains explicitly excluded from this ticket. Its later
landing must not mutate these receipts; it belongs to a newly compiled next ticket.

## Operator GO brief — the two choices

### A. Full FreSh/self-orient ticket

Measures the FreSh cold-start treatment plus the common V9/Muon trajectory and, after producer
landings, D-A, D-B, and causal boundary rows. It retains the 47.13 GiB feature cache and currently
fails system admission. Choosing A requires all four dependency clearances plus a fresh governor
admission or literal operator override rationale. Its generated `launch.sh` is not executable authority.

### B. Trimmed compliant ticket

Measures the common V9/Muon trajectory and the same telemetry without FreSh/self-orient. It passes the
recorded memory-admission snapshot but still has the same four blockers. This is the preferred substrate
for the B=1/B=2 n600 custody measurement after R3 clears, because its checkpoint geometry matches the
preserved self-orient-OFF source.

Both configs preserve `--stage-checkpoints`, periodic `--ckpt-every 25`, EMA shadow, optimizer,
epoch/stage position, and distinct stage artifacts. Production resume is through typed
`--resume-from <run-dir-or-npz>` only. The runtime root is
`/Volumes/VertigoDataTier/pact/next_launch_all_levers_20260713` for whichever variant is selected;
the two variants must not be launched simultaneously into one root.

Before any later fire: clear exact D-A/D-B producers, harvest R2 into the waterfill ledger, obtain a
green SSD plan, recompile the selected typed variant, require `launch_blockers=[]`, rerun live memory
admission and all preflight gates, then obtain explicit operator GO. No step in this memo is GO.

## Artifacts and triality

- Full ticket: `experiments/results/next_launch_all_levers_ticket_20260713/`
- Trimmed ticket: `experiments/results/next_launch_all_levers_ticket_trimmed_20260713/`
- Each contains typed config, compiled argv, DSL manifest, include/exclude table, launch script,
  constants, storage receipt, preflight receipt, compile receipt, and variant GO brief.
- DSL leg: two named typed variants, with the trim expressed as a typed lever exclusion and explicit
  score-impact provenance.
- DAG leg: producer detection and all four blockers live in the compiled manifest and are enforced by
  the governed launcher's rc=11 stop.
- Equation leg: `M_peak = M_fixed + M_cf + M_gt + M_verdict`; full-to-trim delta is
  `47.13 - 0.07 = 47.06 GiB`; system admission is live baseline plus projected run versus adaptive
  ceiling. The B=2 delta remains an unmeasured term and is not inserted.

## STORES CONSULTED

- `CLAUDE.md`, `AGENTS.md`, `docs/operating_manual_craft_handoff.md`
- `PROGRAM.md`, `docs/vehicle_operating_system.md`
- `.omx/research/t5_crucible/SPEC_v75_optimal_single_trunk_20260708.md`
- `.omx/research/SPEC_v8_perclass_decomposition_20260708.md`
- `.omx/research/next_launch_all_levers_ticket_20260713.md` and its original ticket receipts
- latest Codex findings/session summary, T3 council, V9 design, and last-24-hour directive memos
- `reports/latest.md`, lane registry, subagent progress, modal ledger, cost-band and continual-learning
  ledgers, master gradient anchors, and canonical frontier helpers
- `src/tac/witness_autoconfig.py`, the two next-launch/V9 typed specs, curriculum DSL, and real trainer
- `tools/memory_waterfill_config.py`, `tools/witness_memory_preflight.py`,
  `tools/system_memory_governor.py`, `tools/plan_experiment_storage.py`,
  `tools/compose_next_launch_all_levers_ticket.py`, and `tools/launch_witness_run.py`
- `src/tac/causal_manifest.py`, causal build/DAG memos, and the trainer's live boundary writer/calls
- preserved V9·CGauge epoch-275 checkpoint metadata and SHA-256

Score claim: none. Promotion authority: none. Pointer delta: **0**. Run/eval authority axes: none; all
new results are `[config-preflight] NON-PROMOTABLE` and the R2 preview is `[macOS-MLX advisory/design]`.

Commit custody: serializer landing was attempted for only this lane's four source/test files plus this
memo, with base/post SHA guards. The environment refused Git index mutation with
`unable to create temporary file: Operation not permitted`; nothing was staged. The generated ticket
directories and this memo remain durable workspace artifacts, but the source/memo delta is explicitly
**UNCOMMITTED** pending an authorized writable Git-index context.
