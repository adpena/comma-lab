# Codex session summary — DDM v14 receiver-realization fidelity

Date: 2026-07-22  
Lane: `ddm_v14_realization_fidelity`  
MAIN landing review required: true

## Landed

- Extended the v9/v13 receiver in place with a counted 23-byte v14 realization profile, ordered
  874x1164 semantic paint, exact G1 replacement, uint8 amplitude custody, and evaluator-owned R.
- Added exact, parse-back-verified G4 static-cell rules and a research-only companion measurement
  path.
- Added typed, one-stage-per-invocation n64/n600 launchers with immutable batch-16 checkpoints,
  exact archive custody, deterministic replay, storage preflight, and fail-closed mutation probes.
- Added focused tests, canonical equation anchor, DAG FEED, equations note, findings memo, and
  adversarial round-1 record.

## Empirical result

- V14 n600 islands: 133,247 B / d_seg 0.027470296224 / d_pose 163.061327281443.
- Best G4 row, horizon: 133,755 B / d_seg 0.027416720920 / d_pose 163.061458661133;
  joint delta -0.005003006483.
- Gate 0.00116 fails; no compose/dispatch/promotion action.

## Bugs extincted

- Scorer-grid receiver bypass.
- G1 union instead of authoritative replacement.
- Flat inherited Movable RGB under full camera R.
- Uncounted receiver profile/static-rule semantics.
- Launcher import dependence on caller CWD.
- Fake subtraction of future context-stream savings from current archive bytes.

## Pending route

- Direct RGB scorer solve using #559/#549 under the same counted receiver and parse-back contract.
- Physical-BEV AR(1) lane remains `BLOCKED_NO_DECODER_FREE_PHYSICAL_BEV_CUSTODY`.
- MAIN must review target-derived aggregate-rule legality and all verdict scopes before merge.

Pointer: `0.1910828242 [contest-CPU]` — UNMOVED.
