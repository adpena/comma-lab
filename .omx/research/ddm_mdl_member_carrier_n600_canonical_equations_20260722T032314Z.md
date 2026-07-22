---
schema: canonical_equation_candidate_note.v1
utc: 2026-07-22T03:23:14Z
task: 603
feeds_task: 613
source_task: 602
lane_id: lane_ddm_mdl_member_carrier_n600_20260722
research_only: true
registry_promotion: false
---

# Candidate predicate — member-carrier admissibility

Let `O_602` be the exact persisted output of Task #602, `A` the proven same-artifact archive compiler,
`Q8` the uint8 realization, `R` the frozen resize/evaluator input operator, `S16` the frozen SegNet at
batch16, and `P=600` pairs. A member-carrier row is registerable only if

```text
C_carrier(O_602) =
    1[coverage(O_602) = P]
  * 1[coded_member_payload_i exists for every i in 0..P-1]
  * 1[pre_uint8_member_state_i exists for every i]
  * 1[len(A(O_602)) is priced inside the admission decision]
  * 1[Pose payload has unique byte ownership in A(O_602)]
  * 1[parse(A(O_602)) re-encodes to identical bytes].
```

Only when `C_carrier=1` may the curve contain

```text
(B, M_pre, M_uint8, L_uint8, C_pose)
B       = len(A(O_602))
M_pre   = |sites|^-1 sum 1[argmax S16(R(member_pre)) = target_cell]
M_uint8 = |sites|^-1 sum 1[argmax S16(R(Q8(member_pre))) = target_cell]
L_uint8 = M_pre - M_uint8
C_pose  = receiver-consumed Pose coordinates / required Pose coordinates.
```

For the preserved output, coverage is 64, coded payload rows are 0, the numeric domain is already
uint8, exact archive bytes are not in the solve, and no Pose archive section exists. Thus
`C_carrier=0`; `B`, `M_pre`, `M_uint8`, `L_uint8`, and `C_pose` are undefined, not zero.

The prior smooth-chart observation “aggregate membership approximately equals the Undrivable class
prior” remains a FORMULATION-scoped empirical result from Task #603. It must not be generalized into
a family law. Likewise, the diagnostic quantity

```text
77,651,017 / 64 = 1,213,297.140625 bytes/pair
```

is zlib-9 over decoded uint8 camera arrays, not `B`; it is non-registerable.

No canonical equation or member-carrier byte row is appended in this landing. Promotion requires a
measured `C_carrier=1` tuple plus MAIN review; recording undefined quantities as zero would violate
NO-FAKE and corrupt the #613 knee.

CONSUMED-BY: `ddm_describe_line_rate_distortion_bracket_v1` receiver-admissibility history; registration landing `.omx/research/ddm_structured_carriers_law_registration_20260722T142000Z.md`; MAIN review required.
