---
task: 603
master_task: 578
feeds_task: 613
lane_id: lane_ddm_v4_stratum_structured_members_20260722
research_only: true
main_landing_review_required: true
verdict: MEASURED_STRUCTURED_STRATUM_MEMBERSHIP_POSITIVE_N64
verdict_scope: frozen-SegNet membership and Pose-completeness instrumentation on S4 frames 0:64 only
score_claim: false
d_seg_claim: false
pointer_moved: false
---

# DDM v4 per-stratum structured-member measurement (n64)

## Outcome

The existing v3 entropy-priced member harness now admits deterministic, receiver-closed
per-stratum structured carriers rather than forking a second harness.  The Lane LBND2 carrier is
the decisive positive representation rung: its exact 84,918-byte archive has Lane membership
`0.686570162333` and Pose completeness `1`.  All six measured archives are below both the
approximate 200,000-byte task box and the strict 154,524-byte guard configured for this run.

This is not an evaluator score, a d_seg result, or a candidate archive.  The frozen SegNet was used
only as a local membership instrument.  The frontier pointer remains
`0.1910828242 [contest-CPU]`.

## Exact n64 membership matrix

| member role | exact bytes | Road | Lane | Undrivable | Movable | MyCar | Pose complete |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 46,226 | 0 | 0 | 0.999986797783 | 0 | 0 | 1 |
| Road PXQ1+events | 52,643 | 0 | 0 | 0.995119204750 | 0 | 0.997696884575 | 1 |
| Lane LBND2 | 84,918 | 0.941953307322 | 0.686570162333 | 0.996891360888 | 0.057693374974 | 0.138084564815 | 1 |
| MyCar static hood | 46,601 | 0 | 0 | 1 | 0 | 0 | 1 |
| Undrivable PXQ1 boundary | 50,949 | 0 | 0 | 0.999916117621 | 0 | 0.000685816593 | 1 |
| Movable S4 events | 46,537 | 0 | 0 | 0.999986797783 | 0 | 0 | 1 |

The Road-role carrier does not yield Road membership; it yields MyCar membership
`0.997696884575`.  That is a measured cross-class routing failure, not Road efficacy.  The static
hood and Movable-event roles remain target-class formulation zeros at n64.  The Movable prefix has
zero PCE3 records in the first 64 frames.

Overall / boundary / interior frozen-SegNet agreement is respectively:

- baseline: `0.493605613708 / 0.118697769367 / 0.502205475276`
- Road: `0.746824900309 / 0.188012560160 / 0.759643273784`
- Lane: `0.747565269470 / 0.541132399118 / 0.752300550779`
- MyCar: `0.493612130483 / 0.118697769367 / 0.502212141536`
- UndrivableBoundary: `0.493746439616 / 0.120551314493 / 0.502307013800`
- Movable: `0.493605613708 / 0.118697769367 / 0.502205475276`

## Event-subset price curve

| class | n8 bytes | n16 bytes | n32 bytes | n64 bytes | n64 records | n64 sites |
|---|---:|---:|---:|---:|---:|---:|
| Road | 485 | 841 | 1,503 | 3,082 | 170 | 24,781 |
| Lane | 459 | 689 | 1,149 | 2,117 | 192 | 6,304 |
| Undrivable | 352 | 571 | 1,068 | 2,413 | 151 | 23,646 |
| Movable | 62 | 62 | 62 | 62 | 0 | 0 |

## Custody and resumability

The authoritative receipt is
`ddm_v4_stratum_structured_members_n64_603_613_20260722T054746Z_artifacts/ddm_v4_stratum_structured_members_n64_receipt.json`,
66,855 bytes, SHA-256
`95c164b4e8fde4b9f5437dca68c6c230154554c55958725b46b60d7153700e5e`.
It binds the typed config, S4 source/runtime hashes, target-receipt hash, committed producer source,
semantic argv, exact archive bytes/hashes, scorer custody, storage preflight, and immutable
per-stage candidate checkpoints.  A first pre-commit execution correctly refused receipt
publication because the producer source did not yet exist at its claimed git SHA; the committed
rerun completed in 254.07 seconds.

The refused run's 540 KiB untracked scratch directory was deleted only after all six archive byte
counts and SHA-256 hashes matched the authoritative rerun exactly.  The deleted scratch contained
no final receipt and is deterministically rebuildable from the config/argv below; the surviving
authoritative directory contains all six identical archives, immutable checkpoints, and the final
receipt.  No S4 source or authority artifact was moved or mutated.

Re-derive locally (observed under ten minutes):

```sh
/Users/adpena/Projects/pact/.venv/bin/python tools/run_direct_description_entropy_priced_member.py \
  --config .omx/research/ddm_v4_stratum_structured_members_n64_603_613_20260722T053744Z.config.json \
  --output-dir .omx/research/ddm_v4_stratum_structured_members_n64_603_613_REDERIVE_artifacts \
  --execution-allowed false
```

## Blocker delta and handoff

Positive Lane membership retires the narrow question "can a stratum-specific structured carrier
survive the exact archive/receiver surface at useful membership under the byte box?"  It does not
retire full-tolerance feasibility: no exact evaluator d_seg/d_pose exists, n256/n600 were not run,
Road's class routing is wrong, and MyCar/Movable remain formulation-zero.  Task #603 therefore
stays at register rung `8/19`; the result feeds #613 but authorizes no frontier move or launch.

MAIN landing review must re-run the focused tests, verify every archive/checkpoint against the
receipt, inspect LBND2 parity against the S4 runtime, and decide whether Road palette routing or
MyCar/Movable carrier formulation is the next cheapest blocker.

## STORES CONSULTED

- `CLAUDE.md`; `AGENTS.md`; `PROGRAM.md`; `docs/operating_manual_craft_handoff.md`
- `.omx/state/canonical_task_status.jsonl`; `.omx/state/lane_registry.json`
- `.omx/state/subagent_progress.jsonl`; `.omx/state/probe_outcomes_ledger.jsonl`
- `.omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`
- S4 archive/runtime and target receipt bound by the authoritative JSON receipt
