---
title: Task 602 MDL member carrier eligibility for n600
utc: 2026-07-22T03:23:14Z
task: 603
feeds_task: 613
master_task: 578
source_task: 602
lane_id: lane_ddm_mdl_member_carrier_n600_20260722
verdict: BLOCKED_602_OUTPUT_IS_NOT_A_RECEIVER_CARRIER
verdict_scope: FORMULATION_OUTPUT_INTERFACE
research_only: true
execution_allowed: false
main_landing_review_required: true
---

# Outcome

Task #602 cannot be carried through the proven Task #603 n600 receiver as landed. Its preserved n64
solve selected the canonical member for all 64 pairs, changed zero values, and wrote zero coded-member
payloads. Its 77,651,017-byte figure is zlib-9 over decoded uint8 camera members and is explicitly a
diagnostic, not exact final archive bytes. No honest `(archive_bytes, membership)` point exists, so no
raw source, smooth-chart archive, or n64-to-n600 projection was substituted.

This is a `FORMULATION_OUTPUT_INTERFACE` blocker. It does not close the MDL-member family, the
solution-polytope family, or direct description.

# Curve

| Pairs | Archive bytes | Bytes/pair | Membership | Pose bytes/completeness | Disposition |
|---:|---:|---:|---:|---:|---|
| 64 | not measured | not measured | not measured | not measured | no receiver-carriable #602 payload |
| 600 | not measured | not measured | not measured | not measured | only 64 solver stages exist; no projection |

The only numeric rate datum is deliberately outside the curve:

| Pairs | Diagnostic raw-member zlib-9 | Diagnostic bytes/pair | Versus 440 B/pair | Versus 216 KiB knee |
|---:|---:|---:|---:|---:|
| 64 | 77,651,017 B | 1,213,297.140625 B | 2,757.493501x | 351.069774x total knee |

Those bytes cannot feed #613 because they are neither `len(A(z))` nor a parser-valid receiver
artifact. The 78,969-byte compressed PPCS seed is also not a complete solved-pair archive; #602's
own receipt records that distinction.

# Exact blockers

1. `N600_MEMBER_SOLVE_COVERAGE`: 64 preserved pair stages exist; 600 are required.
2. `RECEIVER_CARRIABLE_CODED_MEMBER_PAYLOAD`: 0/64 stages contain a selected-frame payload. A
   canonical selection refers back to the 3.66 GB read-only source raw.
3. `COUNTED_ARCHIVE_MDL_INSIDE_SOLVE`: the solve admits by array zlib; exact final ZIP bytes are not
   priced inside selection.
4. `PRE_UINT8_MEMBER_STATE`: #602's solver domain and the target receipt are already uint8. There is
   no independent pre-quantization member, so claiming zero #532 realization loss would be fake.
5. `POSE_STREAM_IN_MEMBER_PAYLOAD`: scorer comparison telemetry exists, but no receiver-consumed Pose
   payload section or byte ownership exists in #602.

The existing Task #603 same-artifact apparatus remains green at 2,565,528 B. It was not invoked here
because its payload is the decisively negative smooth chart grammar, not a #602 member description.
No frozen scorer was loaded because no eligible archive reached the scorer gate.

# Durable guard

`MdlMemberCarrierPreflightConfigV1` binds the compact/full #602 receipts, all preserved stages, the
uint8 target receipt, and the exact producer sources by SHA-256. The consumer fails closed on custody
or stage drift and emits an empty curve rather than converting diagnostic bytes to archive authority.

Bounded re-derivation, measured below one second and capped by config at 600 seconds:

```text
/usr/bin/env python3 tools/check_mdl_member_carrier_preflight.py --config .omx/research/ddm_mdl_member_carrier_n600_20260722T032314Z.config.json --output .omx/research/ddm_mdl_member_carrier_n600_20260722T032314Z.rerun.json --execution-allowed false
```

Use a fresh output path because evidence publication is write-once.

# Blocker delta and next admissible edge

- `MEMBER_CARRIER_POINT`: `UNMEASURED -> BLOCKED_WITH_EXACT_INTERFACE_CLASSIFICATION`.
- `N600_SAME_ARTIFACT_ARCHIVE_CLOSURE`: green and unchanged.
- PRIMARY register: `8/19`, unchanged; no synthetic green row was appended.

A successor must persist a decoder-consumed coded member payload and an independent pre-uint8 state,
price exact `len(A(z))` inside admission, carry Pose in the same artifact, and then run frozen-SegNet
batch16 membership through uint8/R. That successor may reuse the proven receiver custody but cannot
call the present #602 outputs carrier-ready.

# Verification and stores consulted

The review follows `docs/operating_manual_craft_handoff.md`: outcome first, primary artifacts rather
than memo trust, explicit evidence labels, and an adversarial attempt to falsify the conclusion.
Focused preflight, predecessor receiver, and #602 tests; Ruff; Python compilation; diff checks; and
review-tracker passes are recorded in the companion review memo and final handoff.

STORES CONSULTED: delegated authority; `CLAUDE.md`; `AGENTS.md`; project memory; `PROGRAM.md`;
`docs/operating_manual_craft_handoff.md`; v7.5/v8 contracts; reports/latest; lane/task/subagent state;
#602 compact/full receipts and 64 preserved stages; #602 solver/producer sources; #603 target receipt,
membership receipt, DAG, and equation note; PRIMARY DDM specification; both live inboxes. SSD inputs
were read-only and no bulk bytes were created, moved, or deleted.

# Pointer honesty

`0.1910828242 [contest-CPU]` — unchanged. No score, candidate, promotion, dispatch, or provider call.
MAIN must review the branch diff and scoped interpretation before merge.
