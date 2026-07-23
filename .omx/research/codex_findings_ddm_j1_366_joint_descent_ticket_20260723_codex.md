---
title: Codex findings — DDM J1 #366 joint-descent sealed ticket
utc: 2026-07-23T00:32:10Z
tasks: [366, 578, 603, 613]
verdict: PREP_COMPLETE_EXECUTION_BLOCKED_BY_MISSING_CONSUMER_AND_REAL_PREFLIGHT
verdict_scope: launch-ticket preparation for the proposed v15 grammar-parametrized joint-descent vehicle
research_only: true
execution_allowed: false
score_claim: false
main_landing_review_required: true
---

# Outcome

Landed the #366 finishing SPEC, hash-sealed proposed DDM typed ticket, explicit stage/fork ladder,
memory observation, fail-closed checklist, DAG FEED, and round-1 review. No launch, dry start, GPU
job, or paid dispatch occurred.

The important finding is adversarial: v15 is receiver-loadable but not optimizer-loadable. Its
strict parser/receiver and receipt close the counted archive, yet the repository has no governed
consumer that maps the v15 worldsheet/lane/template streams into trainable state. The level-set
memory model cannot honestly authorize a different J1 consumer. The ticket is therefore sealed
`execution_allowed=false`; MAIN must land the adapter/compiler/resource-model build before the
governor + memory + operator-GO sequence is meaningful.

The receipt sharpens the adapter debt: current v15 has a 29,810-byte G1 worldsheet payload and six
template records, but zero explicit worldsheet track/knot and lane-program/knot records. The build
must lift the G1 payload into equivalent trainable coordinates and define a counted lane zero-state
or seed; those parameters cannot be claimed present merely because the receiver schema supports
them.

## Sealed inputs

- Authority prompt: 6,807 bytes, SHA
  `274937bc20f08b1e2492631ba57d747620fe91b79459c996f8c6fc304174c5cf`.
- Source commit: `968e499a99640f811fd13da8e30531b2cf127425`.
- V15 archive: 133,941 bytes, SHA
  `759e28332ce1ea2d4cabba731e4b7b2b21c191fef1bd2b104fab18805388d6df`.
- V15 receipt: SHA
  `5ed6f830b3749a51e0d300a9104fda9a77e86bbeb3b81428a20e1ec0d3dcfcb8`.
- Current receiver row: d_seg 0.027470296224 / Movable 0.291615222639 / Lane
  0.435195521828 / d_pose 163.061327281443.

## Memory and wall clock

Read-only host observation: 128.0 GiB total, 96.2245 GiB available, 29.8948 GiB used. Historical
R1 projected 67.6 GiB standalone, leaving 28.6245 GiB against that instantaneous free-memory row.
This is explicitly a surrogate, not a J1 admission receipt. A real J1 projection is unavailable
until the adapter exposes its tensor/cache/verdict geometry. R1 history yields a DERIVED idealized
17.28 hours at eval-every=5; the ticket preregisters 17–30 hours pending a real timing smoke. The
baseline is MLX-GPU custom grouped backward plus fused differentiable-R. Conditional fused
grammar/R and YUV6/Pose kernels are named with SPECULATIVE gains, but activate only if the real
component smoke binds; they are speed-only and cannot mint evaluator authority.

## Round-1 review disposition

1. **Payload boundary — PASS as a specification.** Only grammar/template/xi parameters are counted;
   scorer weights, GT tables, decoded planes, and post-hoc pose are forbidden.
2. **Warm-start loadability — FAIL for execution.** Receiver parse/render custody is landed; no
   optimizer adapter exists, explicit track/knot and lane-program records are zero, and a typed
   parameter-lift/counting contract is owed. This blocks fire.
3. **DSL/typed hash — PASS for declarative custody, FAIL for executable compile.** The semantic
   program is canonical-hashed; the proposed type is not in the current compiler/launcher.
4. **Resumability — PASS as a binding contract, UNMEASURED in J1.** Required state and immutable EMA
   stage checkpoints are specified; no J1 crash/resume receipt exists.
5. **Memory — FAIL admission.** The available 67.6-GiB R1 model targets another consumer. Recording
   it as J1 green would recreate the surrogate-green bug class.
6. **Targets — preregistered, not predicted.** Fraction-ladder values are DERIVED stop thresholds;
   per-stratum final watches are SPECULATIVE; 0.00116/200KB is the settled same-artifact fork.
7. **Authority — PASS.** Ticket remains $0/prep-only and requires MAIN landing review plus separate
   operator GO.

## Blocker delta versus #603 / #366

The compact actuator and starting archive now exist; “build a representation” is no longer the
immediate blocker. The concrete delta is a missing v15-to-trainable-state adapter, typed compile
route, governed launcher registration, and real memory/timing model. Once those apparatus receipts
are green, the vehicle can test the still-open efficacy question. Nothing here claims the vehicle
will reach the box.

## STORES CONSULTED

`CLAUDE.md`; `AGENTS.md`; `PROGRAM.md`; `docs/operating_manual_craft_handoff.md`; v7.5/v8 specs;
#366/#603 specs; R1, #378, #383, #549 artifacts; v14/v15 configs, receipts, code, and prior Codex
findings; curriculum DSL and LawRefs; launcher/memory preflight; lane/task/progress/frontier state;
2026-07-19 Fisher/reverse-waterfill directives and the 2026-07-23 compute mandate.

Pointer `0.1910828242 [contest-CPU]` unchanged. MAIN landing review is required.
