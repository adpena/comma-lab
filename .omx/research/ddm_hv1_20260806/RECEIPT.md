# ddm_hv1 2026-08-06 Receipt

Axis: `[scorer-free harvest/routing overlay]`. `score_claim=false`.
No scorer forward, no `upstream/evaluate.py`, no launch, no paid dispatch, no live run
directory write, no protected-file edit, and no staged-index manipulation occurred.

## Answer First

HV1 drained the recent producer/consumer asymmetry into a central routing overlay:

- Ledger: `.omx/research/ddm_hv1_20260806/HV1_DISPOSITION_LEDGER.jsonl`
- Resume handoff: `.omx/research/ddm_hv1_20260806/NEXT_IF_RESUMED.md`
- Checkpoints: `.omx/research/ddm_hv1_20260806/CHECKPOINTS.md`

Existing owner stores were already better than a fresh broad sweep in two places:

| surface | consumed fact |
|---|---:|
| FO1 rerun | 858 dispositions, 460 queued rows with owners, 0 unowned queued rows |
| QJ1 join | 778 dispositions, 390 queued rows all with owners, unowned queued rows 0 |
| NP1 queue surfaces now | 56 NEXT rows, 79 final-message rows |
| probe outcomes store | 662 rows |

HV1's own overlay now has 29 rows covering the charter row-groups. Before this pass, those
row-groups were scattered across receipts, final messages, task-status rows, and queue stores
with no single HV1 disposition table. After this pass, every HV1 row-group has one of:
`CONSUMED`, `FOLDED`, `QUEUED-WITH-FIRE-ORDER`, `QUEUED-LOW-PRIORITY`,
`MIXED-MEASURED-AND-QUEUED`, or a scoped bounded absence.

`us2` residue status: already committed before HV1. `PREDICTIONS.md` is tracked, clean, has a
final newline, and was already landed at commit `bb9a69bbb4` (`ddm_us2: PREDICTIONS.md EOF
cleanup (arm sandbox git-object write blocked) [no-triality] [p0-ledger-ok]`). HV1 made no edit
there.

## Row Disposition Summary

| HV1 row | disposition | owner / consumer |
|---|---|---|
| NP1 NEXT/final surfaces | CONSUMED | `tools/codex_arm_queue.py`, `tools/costate_digest.py` |
| FO1/QJ1 follow-on ledgers | CONSUMED | ranked follow-on owners |
| probe outcomes / VW1 | FOLDED-INTO-WINNING-STORE | `tac.probe_outcomes_ledger`; VW1 task #936 content |
| AU1 detectors | QUEUED-WITH-FIRE-ORDER | measurement-integrity successor |
| PE3 | QUEUED-WITH-FIRE-ORDER | PE receiver/scorer survival owner |
| PE4 | QUEUED-WITH-FIRE-ORDER | MAIN scorer-slot owner after PE2 harvest |
| OD2/OD3 | QUEUED-WITH-FIRE-ORDER | OD receiver/scorer survival successor |
| OD4-OD7 | QUEUED-WITH-FIRE-ORDER | OD5/generator/worldsheet receiver closure |
| OD8 | FOLDED-AS-ESTIMATE | OD solve-persist successor |
| OD9 | FOLDED-DEAD-FORMULATION | GC19 disposition |
| GC17/GC18 | CONSUMED | GC19/GC20 routable owners |
| GC19/GC20 | CONSUMED | MAIN endpoint consumer plus successor lanes |
| JD/LA1/Q3 | QUEUED-WITH-FIRE-ORDER | JD8-Q3 active lane and MAIN live-board |
| tq1c | QUEUED-LOW-PRIORITY | token-edit owner, resume index 27 |
| WP1 | MIXED-MEASURED-AND-QUEUED | endpoint atlas owner and WP1 successor |
| WL1 | CONSUMED | WL1-LB/WL1-SR/WL1-EIK and listed row owners |
| US2 | MIXED-FOLDED-AND-QUEUED | upstream/runtime/scorer-object maintainers |
| ET1/PH1 | FOLDED-INTO-Q3-ORDERING | Q3/phase-field scorer-slot owner |
| NA4 | CONSUMED-AS-AXIS-LAW | rate-axis projection consumers |
| BO1 | QUEUED-WITH-FIRE-ORDER | mg1/EN1/objective A-B owner |
| BP1 | FOLDED-INTO-EXISTING-RESET-RACE | existing bp1/reset-race lane |
| SV2 | FOLDED-DEAD-FORMULATION | token-rate/content owner |
| SM2 | QUEUED-OPTIONAL-NOT-DEFAULT | rate-model/QA86 successor |
| CA1 | QUEUED-WITH-FIRE-ORDER | cap-default guard / q3x live-risk owners |
| EU2 | QUEUED-CACHED-NO-SCORER | EU2-X1-10K-context-orderer if no conflict |
| KS1 | FOLDED-HISTORICAL | MAIN review / W_seg start selection law |
| AL1 | QUEUED-WITH-FIRE-ORDER | burn-supervisor/confound-gates successor |

## Scoped Absence

These are bounded findings, not global nonexistence claims:

- `vw1`: no standalone `.omx/research` or `.omx/tmp/codex_runs` receipt path was found in
  the searched scopes. The content was found in `.omx/state/canonical_task_status.jsonl` under
  actor `ddm_vw1`, task `#936`.
- `bp1`: no standalone recent 2026-08-05 BP1 receipt was found in the searched scopes. CI1 and
  NA2 route the evidence into the existing reset-race lane if resumed.
- `ks1`: the relevant surface is older, 2026-07-25, and is historical start-state custody. It
  does not compete with current own-vehicle pointer work.

## Boundaries

Measured in HV1: file/receipt existence, ledger rows, queue-surface counts, tracked-clean US2
residue state, and disposition routing over the bounded charter corpus.

Not measured in HV1: any new score, any new d_seg/d_pose, any exact contest CPU/CUDA row, any
scorer output, any full-menu token closure, any OD/PE receiver survival, or any live JD8-Q3
endpoint.

Protected files named by the common contract were not edited:

- `.omx/research/ddm_cr1_composition_row_827_20260801.md`
- `.omx/research/ddm_pu2_pose_tail_floor_probe_20260803.md`
- `src/tac/optimization/direct_description_carrier_compose.py`

## Verification

Planned local verification for this report-only landing:

1. `git diff --check -- .omx/research/ddm_hv1_20260806`
2. `shasum -a 256 .omx/research/ddm_hv1_20260806/*`
3. serializer commit with `--expected-content-sha256` for each HV1 file and tags
   `[no-triality] [p0-ledger-ok]`

Own-vehicle frontier line: `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`;
contest pointer `0.1910828242` borrowed/unmoved.
