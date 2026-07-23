---
title: DDM v19b greedy joint-remeasurement stack DAG feed
date_utc: 2026-07-23
lane_id: ddm_v19b_joint_remeasure_stack
research_only: true
execution_allowed: false
score_claim: false
evidence_axis: "[macOS-CPU frozen-scorer advisory]"
verdict: MULTI_MOVE_JOINT_STACK_ADMITTED_N600_ADVISORY
verdict_scope: "INSTANCE:V19B x ten source-v19 winners x exact common-master replay; no contest-axis, family, score, or promotion verdict"
pointer: "0.1910828242 [contest-CPU]"
pointer_moved: false
main_landing_review_required: true
---

# Verdict

All ten source-v19 winners survive greedy composition when every proposal is
remeasured as one exact receiver archive. The final n600 archive is 137,825 B,
SHA-256 `74ede4194ed0eb6c2716f1033a47569c87a7afdf0c21fb1ee23ac059beff3aae`.
Against the sealed 133,941-byte v15 control it measures:

- `delta_d_seg = -0.0008758714460000011`
- `delta_d_pose = -0.00015067664799062186`
- `delta_archive_bytes = +3,884`
- `delta_S = -0.08501960537266746`
- `103,322` realized net Seg flips

This is `[macOS-CPU frozen-scorer advisory]`, `score_claim=false`. It does not
move the contest pointer.

# Per-move joint table

The first move is the v19 405-flip compact-int8 winner. Each later row is the
incremental delta from the then-current accepted stack, never a sum of
single-step rows. Role means Lane+Movable; residual means
Road+Undrivable+MyCar.

| i | move | admit | delta d_seg | delta d_pose | delta B | delta S | single-step gain survival | role net flips | residual net flips |
|---:|---|:---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | `1x1_rowband_control_solve_02_M_preconditioned_ranked_prefix_r4` | yes | -5.0226847e-05 | 0.000978855354987 | 201 | -0.00476754592343 | 1.000000 | 120 | -41 |
| 1 | `worldsheet_joint_active_x_+1` | yes | -8.6466472e-05 | -0.00180366861099 | -5 | -0.00887348991913 | 0.965752 | -21 | 157 |
| 2 | `worldsheet_joint_active_y_-1` | yes | -0.000386555989 | -0.002182246084 | 3 | -0.0389240303131 | 4.747432 | 121 | 487 |
| 3 | `1x1_rowband_control_solve_03_M_preconditioned_ranked_prefix_r8` | yes | -0.000389099121 | 0.00117129274901 | 30 | -0.0387447867953 | 9.058356 | 520 | 92 |
| 4 | `preuint8_405_scale_q8_256` | yes | -9.1552735e-05 | 0.000557318710975 | 2262 | -0.00758003639845 | 2.232567 | 4 | 140 |
| 5 | `preuint8_405_scale_q8_192` | yes | -6.6757202e-05 | 0.000349224057004 | 0 | -0.00663244365381 | 2.010955 | -20 | 125 |
| 6 | `1x1_rowband_control_solve_01_M_preconditioned_ranked_prefix_r2` | yes | -5.6584676e-05 | 0.000194138998012 | 0 | -0.00563440952572 | 1.869842 | -13 | 102 |
| 7 | `worldsheet_joint_active_x_-1` | yes | -1.5894572e-05 | -0.000584187868014 | 6 | -0.00165785576255 | 0.580810 | 108 | -83 |
| 8 | `preuint8_405_scale_q8_128` | yes | -5.6584676e-05 | 0.000167206421025 | 0 | -0.00563774703627 | 3.499839 | -16 | 105 |
| 9 | `1x1_rowband_control_solve_00_M_preconditioned_ranked_prefix_r1` | yes | -2.7338664e-05 | 9.58490279857e-05 | 0 | -0.00272198859603 | 1.717367 | -12 | 55 |

Zero incremental bytes mean the already-present fixed-count record grew in
amplitude without changing its encoded width. They are measured byte deltas,
not free-data assumptions.

# Non-additivity

For the nine remaining winners, independently measured single-step gains total
`0.03742127019557973` score units. Joint remeasurement preserves
`0.03591006678308081`, degrades `0.001511203412498918`, and adds
`0.08049672121725288` of amplification. Thus 95.9616459% of the original
single-step gain survives before counting amplification. The actual incremental
joint gains telescope to `0.116406787...`; they are not reconstructed from the
source-v19 alternatives.

# Scale ladder and c1 handoff

| rung | archive bytes | delta d_seg | delta d_pose | delta B | delta S | net flips | role | residual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| n64 | 61,087 | -0.001589775085 | -0.000453630121 | 2,344 | -0.157473592822 | 20,004 | 13,280 | 6,724 |
| n600 | 137,825 | -0.000875871446 | -0.000150676648 | 3,884 | -0.085019605373 | 103,322 | 29,377 | 73,945 |

The correction line realizes 26.6019567456 net flips per added byte, or
0.03759121968 B per net flip. It leaves 3,000,367 errors above c1's integer
136,839-error target. The exact n600 output becomes the input to both c1
16,384-byte downstream budgets:

- v18b first exact pricing rung: 137,825 B, `d_seg=0.026594424778`,
  3,137,206 errors; solo full closure would require 183.1278686523 net flips/B.
- J3 xi/template/worldsheet finish: same exact input; its required credit is
  conditional on the preceding exact v18b replay. Independent deltas remain
  non-additive.

The measured credit is mostly residual-bucket, so c1 must not classify the
stack as Lane+Movable-only.

# Atom-order gauge

The shared six-template raw payload is 140 B as emitted and 140 B under
canonical atom ordering: delta 0 B. The current format is fixed-width and
ZIP-stored, so ordering has no rate actuator yet. The order-matching lever is
routed to c1 CODE; a future order-sensitive delta/entropy coder must remap
template placement indices and prove exact camera-byte identity. No frozen
scorer permutation or training-landscape claim is imported.

# Executable DAG

```text
SHA-bound v19 config + v19 receipt
  |
  +-> reconstruct v17 problem, v15 n64/n600 carriers, frozen scorer custody
  |
  +-> common-master compiler
  |     carrier worldsheet grammar
  |       -> compact int8 coupled-margin templates/sparse records
  |         -> summed Q8 correction before one final uint8
  |
  +-> force 405 winner
  |
  +-> remaining winners sorted by source single-step delta S
  |     -> compile {stack + candidate}
  |     -> exact receiver + SegNet + PoseNet + archive length
  |     -> admit iff incremental delta S < 0
  |     -> immutable candidate checkpoint
  |
  +-> n64 four-batch replay
  |     -> strict negative delta S gate
  |
  +-> n600 38-batch replay
        -> exact per-class buckets + c1 handoff + final receipt
```

# Triality and custody

- DSL: `.omx/research/configs/ddm_v19b_joint_remeasure_stack_20260723.json`
- DAG: this file and the immutable stage receipts
- equations: `.omx/research/ddm_v19b_joint_remeasure_stack_canonical_equations_20260723.md`
- final receipt:
  `.omx/research/ddm_v19b_joint_remeasure_stack_20260723T051914Z/ddm_v19b_joint_remeasure_stack_receipt.json`,
  SHA-256 `4bb5d6b4b793b667c7cbe15e37cbf9a27f6c0e75451374839fb5df8ca1c1b8e8`
- n64 batch chain:
  `1ea23cc6d4c60c8bf080564d8973f7e12a785e3c0df5dc4ad7cce20aaab15aec`
- n600 batch chain:
  `49248868df850d0ce55eb17f0ff0fd386798906db433a3b02ef4d7605eadeb24`

No paid dispatch, remote execution, GPU run, live vehicle actuation, or
contest-axis evaluation occurred.

# MAIN landing review

MAIN must independently verify:

1. source v19 config SHA `61138fa71ce2adb388e619a5773ad49df9f5d83c7637241e5f7b9ee6eed9abed`
   and receipt SHA `ec6d49b5ba89c352d1c76bbf8e4e1783374a36d61db6aa2262099a59d52294db`;
2. exact reproduction of the 405 archive/cells before accepting composition;
3. per-track feasible-subset grammar merge, additive int8 deltas with wire
   clipping, and Q8 summation before one final uint8;
4. all ten incremental `delta_S < 0` decisions from exact joint replays;
5. four n64 and 38 n600 batch checkpoints, archive identity, and digest chains;
6. c1 role/residual arithmetic, 16,384-byte handoff formulas, false-authority
   labels, pointer immobility, and no inferred contest score.

