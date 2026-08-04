---
schema: ddm_r9m_ready_to_fire_memo.v1
date_utc: 2026-08-04T18:39:30Z
arm: ddm_r9m
status: READY_FOR_MAIN_FIRE
axis: "[contest-CPU pending]"
score_claim: false
promotion_eligible: false
pointer_moved: false
---

# ddm_r9m - first own-vehicle contest-CPU row readiness

## Verdict

R9 is ready for main fire, but no Modal dispatch was launched from this
sandbox. The provider probe `.venv/bin/modal app list` did not return within
40 seconds and was interrupted before any dispatch boundary. No call_id exists
and no contest-CPU score was measured.

## Custody

Exact archive selected:

`/Volumes/VertigoDataTier/pact/ddm_fz1_20260804/rowB/sub_final/archive.zip`

Verified bytes and SHA-256:

`358,084 B`

`ad5dd0e4fbe5b13ab53a5995a6d77cc558c25f40b63f894ea50ad336bd50fb66`

The fz2 byte-close receipt reports `run_inflate` rc=0, byte ledger closes true,
residual bytes 0, payload re-encodes identically, one raw output of
`3,662,409,600` bytes, and advisory n600:

| axis | S | bytes | d_seg | d_pose |
|---|---:|---:|---:|---:|
| `[macOS-CPU advisory]` | 0.7541458627 | 358,084 | 0.00431179 | 0.00071459 |

Score terms recomputed from components:

`0.4311790000 + 0.0845334253 + 0.2384334374 = 0.7541458627`.

## Gate State

Local dual-ledger single-flight check:

`MODAL RECONCILE: 0 live ledger call_id(s), 0 active Modal claim(s); OK`.

Dry-run claim passed for:

`lane_ddm_r9m_first_own_vehicle_contest_cpu_20260804`

`ddm_r9m_sub_final_contest_cpu_20260804a`

Runtime upload hash for the exact submission directory:

`3ea13f96213785d6db7751849334b8ceba285560255f3fce6552c6ab1584c523`

## Fire Order

Run the command in:

`.omx/research/ddm_r9m_first_contest_cpu_row_20260804/r9m_ready_to_fire_command.txt`

Then harvest with:

`.venv/bin/python tools/recover_modal_auth_eval.py --output-dir .omx/research/ddm_r9m_first_contest_cpu_row_20260804/modal_exact_eval_results/contest_cpu`

## Boundary

This memo is not a score row. It records custody and readiness only. The
own-vehicle frontier remains `S = 0.7541459 @ 358,084 B [macOS-CPU advisory]`;
the borrowed contest pointer `0.1910828242 [contest-CPU]` is unmoved.
