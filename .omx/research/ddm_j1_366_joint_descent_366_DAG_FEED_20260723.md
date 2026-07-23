---
title: FEED-366 DDM J1 joint-descent sealed prep ticket
utc: 2026-07-23T00:32:10Z
tasks: [366, 578, 603, 613]
status: PREP_COMPLETE_EXECUTION_BLOCKED
verdict: REFUSE_NO_JOINT_CONSUMER_OR_REAL_MEMORY_RECEIPT
verdict_scope: build and launch readiness of the proposed J1 vehicle; grammar-family efficacy remains open
execution_allowed: false
research_only: true
main_landing_review_required: true
---

# FEED-366-joint-description-descent-trigger-fired

The v15 trigger is settled and #366 moves from fallback to the required next formulation. V12
exhausted post-solve obligations at d_seg 0.034004; v14 exposed a mask-to-through-R projection gap;
v15 admitted zero zero-collateral shared-template steps with minimum off-target harm 13 Movable /
23 Lane. R1 independently proved that joint descent can cross the pose photometric wall and reached
byte-close d_pose 0.001610. Therefore train the compact description with both evaluator legs in the
objective; do not extend post-hoc correction.

## Node state

| node | state | evidence / obligation |
|---|---|---|
| v15 receiver-closed warm start | GREEN_LANDED_ADVISORY | 133,941 B; SHA `759e2833...d6df`; d_seg 0.027470296224 |
| counted worldsheet/lane/template grammar | GREEN_RECEIVER / RED_OPTIMIZER | 29,810 B G1 + six templates; explicit track/knot and lane-program/knot counts are zero; no parameter-lift adapter |
| joint Seg/Pose action | SPECIFIED | exact-R SegNet + official-YUV6 PoseNet + archive rate |
| amber #378 | GREEN_LAW / RED_J1_COMPILE | launch-blocking stability contract not yet emitted by a J1 compiler |
| repaired pose gate #383 | GREEN_LANDED / RED_J1_WIRE | commit `b617cef526`; J1 consumer wiring absent |
| #549 target feasibility | GREEN_ADVISORY | joint RGB target exists; compact grammar reachability unmeasured |
| typed semantic ticket | GREEN_HASHABLE_PREP | proposed schema; executable compiler absent |
| governed launcher route | RED | named config not accepted by launcher |
| real memory receipt | RED | current preflight models another consumer; historical 67.6 GiB is surrogate only |
| timing smoke | RED | provisional 17–30 h derived band only |
| compute baseline | SPECIFIED | MLX-GPU custom grouped backward + fused diff-R; kernels are speed-only |
| launch | REFUSE | prep-only authority; no operator GO; no consumer |
| pointer | UNMOVED | `0.1910828242 [contest-CPU]` |

## Executable successor edge

`v15 receiver archive -> typed parameter adapter -> byte-identical stage-00 replay -> exact-R joint
Seg/Pose descent -> immutable EMA stage checkpoints -> same-artifact 0.00116/200KB fork -> exact CPU
and CUDA replay`.

MAIN should open one build node, not a research branch:

`ddm_j1_366_joint_descent_consumer_adapter`

Exit requires the five build receipts named in the SPEC: typed compile/hash, v15 load/replay,
governed launcher route, real memory-model projection, and bounded n600 step/checkpoint/resume.
Only then may the sealed fire argv become syntactically and operationally valid.

## Blocker delta versus #603 / #366

#603's scorer obligations and v14/v15 receiver now supply a compact, parse-back-closed actuator and
a measured starting field. The open #366 problem is narrower than “invent a description”: consume
that existing description as trainable low-dimensional state and put official Pose inside the
objective. The blocker is apparatus, not a claimed mathematical impossibility: missing adapter,
typed compiler route, and real resource model. This prep landing closes the ticket/spec ambiguity;
it does not close implementation or efficacy.

## Triality

- DSL: hash-sealed proposed DDM typed config; executable compile leg owed by the successor node.
- DAG: this feed and successor edge.
- Equations: contest joint action + Fisher/margin reverse-waterfill stop law; no new empirical law
  from prep-only work.

## STORES CONSULTED

`docs/operating_manual_craft_handoff.md`; v7.5/v8 operating specs; #366 and #603 specs; R1 recipe;
#378, #383, and #549 artifacts; v14/v15 configs/receipts/code/findings; curriculum DSL/LawRefs;
launcher and memory-preflight code; canonical lane/task/frontier surfaces; 2026-07-19 operator
Fisher/reverse-waterfill directives; 2026-07-23 operator compute mandate.

MAIN landing review is required.
