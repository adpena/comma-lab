# Codex findings — DDM CT1 campaign telemetry and anticipation encoding

UTC: 2026-07-25T11:32:13Z
Lane: `lane_ddm_ct1_campaign_telemetry_encode_20260725`
Status: `BUILT_AND_LOCALLY_VERIFIED_MAIN_REVIEW_REQUIRED`
Research only: `true`

## Verdict

The stopped v5 campaign is now a typed, auto-discovered costate-digest source.
The digest filters stable storage roots through
`tac.witness_run_artifacts.is_run_dir`, verifies the DDM run-identity schema,
and selects the maximum directory `mtime_ns`; no transient run-directory name
is compiled into the consumer.

Six authority-separated rows are live in the human and JSON digest:

| row | status | measured or derived value |
|---|---|---|
| latest exact n600 | `MEASURED`, `[macOS-CPU frozen-scorer advisory]` | step 50: `d_seg=0.07051923116048177`, `d_pose=36.622702484220724`; `score_claim=false` |
| accepted cadence | `MEASURED` | 50 accepted checkpoints; 50 telemetry steps; `302.9149270083662 s/step` mean versus sealed `312 s/step` |
| cumulative local trace | `ADVISORY_BATCH_LOCAL` | sum final-minus-initial `delta_d_seg=-0.026585888117551804`, `delta_d_pose=+0.32845163345336914`; explicitly not n600 |
| pose-finish watch | `DERIVED_FROM_SEALED_GATE_CONSTANTS` | 3 exact verdicts observed; sealed 3/3/3 gate at 50-step cadence gives conditional candidate verdict 5 / step 250 and settled verdict 6 / step 300 |
| geometry cures | `MEASURED` | 12 typed cure-event receipts |
| endpoint ETA | `DERIVED_FROM_MEASURED_CADENCE_AND_SEALED_SCHEDULE` | 400 remaining sealed steps, `33.657214112040684 h`; counterfactual because the governor stopped the run |

The campaign verdict is
`BLOCKED_REALIZED_DSEG_REGRESSION`. This is an INSTANCE endpoint for this
campaign trajectory. It is not a formulation, family, paradigm, promotion, or
contest-score rejection. The banked R1 comparator contribution `0.127` is read
from typed ticket/final-receipt custody and remains a non-promoting C1 budget
and harvest signal.

## R6 copied-checkpoint rehearsal

Receipt:
`.omx/research/ddm_ct1_campaign_telemetry_encode_20260725T111500Z/r6_rehearsal_receipt.json`.

The rehearsal copied the completed campaign to the governed SSD before
inspection. Source before/after and copy after all share tree SHA-256
`18526644423949012b29dc1e043c744ee93e5739338b0e98d97f3f1462d8467c`
(126 files, 1,075,654 bytes). The latest accepted copied checkpoint is global
step 50, SHA-256
`043c2a8b3c89688510cc0ff002f37a375a974205a5f8760d93133c47b7cec7c1`.

The launcher `--resume-proof` passed in a fresh process on that copy:

- theta, EMA, Adam first moment, and Adam second moment were bit-identical
  across independent reloads and have recorded array hashes;
- deterministic RNG state and global-step cursor matched;
- `pose_finish_engage_state` restored exactly with verdict steps `[0,1,50]`;
- the receipt cannot authorize execution and explicitly requires MAIN review
  of the modified reader.

R6 then refused at the first honest boundary:
`R6_BLOCKED_E5_MIDCAMP_CHECKPOINT_ADAPTER_ABSENT`. E5 accepts a receiver-closed
WS1 archive plus exactly two typed streams; the campaign checkpoint contains
optimizer/cursor state and an archive identity, not the realized archive
bytes. Therefore E5 export, parse-back, and IC1/IC2 were not run; their
`d_seg`, `d_pose`, and archive-byte fields remain null. Verdict scope is the
mid-campaign E5 adapter FORMULATION only. It is not a family or paradigm
negative, and high first-cut bytes remain an optimization-ladder measurement.
No scorer, campaign launch, live-run mutation, or pointer mutation occurred.

## FEED-603-ct1

```text
typed DDM run roots
  -> wra.is_run_dir + typed identity + max directory mtime
  -> exact n600 / cadence / batch-local / pose / geometry / ETA rows
  -> one read-only costate digest source
       +-> #404 Class-E campaign telemetry: DONE-partial
       +-> #319 outcome-credit input: cadence + typed trajectory available
       +-> co4 post-J8F allocation input: cadence + typed trajectory available
       +-> RIPO metric-radius candidate: note only, no lever or claim

latest accepted checkpoint COPY
  -> fresh-process resume proof: PASS
  -> E5 mid-campaign adapter: BLOCKED_FORMULATION_SCOPED
  -> parse-back / IC1 / IC2: NOT_RUN_UPSTREAM_TYPED_BLOCKER
```

- **#404 Class-E: `DONE-partial`.** Cadence, accepted progress, conditional
  engagement window, geometry events, and counterfactual endpoint ETA are now
  typed and agent-native. A learned convergence-time/stop law is not claimed;
  the run stopped at its existing exact-regression governor.
- **#319 and co4: `FLOWING_TYPED_INPUTS_NO_DELTA_S_PER_HOUR_VERDICT`.** The
  measured time denominator and step-local trajectory now flow through the
  digest. The local objective changes are `ADVISORY_BATCH_LOCAL`, while only
  steps 0, 1, and 50 have exact n600 verdict custody. Consequently this landing
  does not fabricate a campaign-level `deltaS/hour`; #319 outcome credit and
  co4 allocation may consume the typed inputs only after their existing exact
  authority gates.
- **RIPO radius law: note only.** The existing categorical-Fisher,
  metric-scaled trust-radius candidate remains the correct A/B shape. This
  landing adds no radius lever, no constant, no actuation, and no empirical
  RIPO claim.
- **R6 next edge:** build a reviewed adapter that materializes the
  receiver-closed WS1 archive and exactly two typed streams from the copied
  checkpoint state; only then run E5, parse-back, and IC1/IC2.

## Triality

- **DSL/typed leg:** existing hash-sealed DDM ticket and final receipt provide
  schedule, pose gate, evidence axis, and banked comparator custody. This
  read-only landing adds no flag or bypass path.
- **DAG leg:** `FEED-603-ct1` above is the durable consumer route. The central
  hot DAG should receive it only during MAIN landing review.
- **Equation leg:**
  `src/tac/canonical_equations/ddm_pose_finish_engagement_watch_20260725.py`
  owns the conditional 3/3/3 engagement-window derivation and carries no
  actuation authority.

## Resume invariant and verification

Missing optimizer, EMA, RNG, cursor, or pose state is fail-closed for a resume
proof. The costate digest remains warn-only observability: an absent proof
means `WARN_ONLY_NO_RESUME_AUTHORITY`, never permission to resume.

Verification completed:

- default-system-Python human and JSON costate-digest smoke, including all six
  rows despite optional SciPy being unavailable to the broader equation fleet;
- copied-checkpoint fresh-process `--resume-proof` with source/copy
  non-interference hashes;
- focused observability, witness-run-artifact, and DDM joint-descent tests;
- Ruff, Python compilation, diff check, and two review-tracker passes on every
  changed implementation file before serialization. The root-level test file
  was manually reviewed because that directory is outside the tracker scan
  roots.

## STORES CONSULTED

- `CLAUDE.md`, `AGENTS.md`, `docs/operating_manual_craft_handoff.md`;
- v7.5 operating contract and the delegated authority prompt;
- canonical lane registry, lane maturity audit, subagent progress, frontier
  report, and per-arm/broadcast inboxes;
- the auto-discovered v5 campaign run identity, launch manifest, hash-sealed
  ticket, step telemetry, exact verdicts, accepted checkpoints, geometry cure
  receipts, and final receipt;
- existing #404 telemetry, #319 outcome-credit, co4, RIPO, E5, IC1, and IC2
  DAG/findings surfaces;
- the copied SSD campaign tree and fresh-process resume-proof receipt.

## MAIN landing review requirement

MAIN must review the entire branch diff before merge, with special attention to:

1. the broadened name-agnostic `witness_run_artifacts.is_run_dir` predicate;
2. authority separation between exact n600 and `ADVISORY_BATCH_LOCAL` rows;
3. the launcher’s bit-identical-custody-copy allowance being reachable only
   from non-executing `--resume-proof`;
4. the scoped E5 blocker and null IC1/IC2 fields;
5. central-DAG consumption of `FEED-603-ct1`.

Until that review and merge, the landing is local-only, non-promotable, and
cannot authorize any resume or execution.
