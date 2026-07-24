---
feed_id: "FEED-DDM-DC1-SCORE-QUOTIENT-20260724"
date_utc: "2026-07-24T13:18:00Z"
lane_id: "lane_ddm_dc1_score_quotient_functional_contract_20260724"
research_only: true
execution_allowed: false
score_claim: false
promotion_eligible: false
main_review_required: true
pointer: "0.1910828242 [contest-CPU] UNMOVED"
---

# DDM DC1 score-quotient functional DAG FEED

## Current state

`BINDING_INCOMPLETE_FIT_OWED`

Missing stream:
`FIT_RESULT_RECEIVER_CLOSED_V14_OR_BETTER`.

The feed binds a build-only contract. It does not authorize a fit, launch,
scorer run, exact eval, candidate archive, or frontier update.

## Producer → gate → consumer graph

```text
MAIN-reviewed named base bytes
  + DC1 parameters [SKELETON x L1]
  + DC1 temporal latents / xi [CONNECTION x L2]
  + DM1 externally priced 25 records [FIBER x L3 transport]
  + at-risk exceptions [RESIDUAL x L4]
    |
    v
canonical compiler
  - zlib9 / raw-LZMA1 / passthrough exact byte arbitration
  - section CRC32 + outer CRC32
  - canonical address and stream-tag validation
  - typed-section byte conservation, including packet framing
  - inactive streams => exact named-base bytes
    |
    v
strict parser + exact recompile equality
    |
    v
score-plane receiver
  - two uint8 384x512 RGB planes
  - six xi pose targets
  - mandatory external DM1 applier when records exist
  - decoded-value SHA-256 validation
    |
    v
explicit camera preimage -> real R -> uint8 equality
    |
    v
FUTURE: frozen SegNet argmax + PoseNet YUV6
    |
    v
S = tac.contest_score.compute_contest_score(
      d_seg, d_pose, exact total counted bytes)
    |
    +--> reverse-waterfill: stop below 25/37545489 score units/byte
    |
    v
v14 binding falsifier
  d_seg <= 0.027470296224
  bytes <= 133247
  receiver_closed == true
    |
    +--> today: INCOMPLETE
    |
    v
MAIN independent review
    |
    +--> only then may a separate authority enable DDMEventContinuationV1 fit
```

## Triality

| Leg | Durable surface | State |
|---|---|---|
| typed DSL/contract | `tac.optimization.ddm_score_quotient_functional_contract` | built, 7 contract tests pass |
| DAG | this feed | built, execution disabled |
| equations | `ddm_score_quotient_functional_v1` plus `populate_ddm_score_quotient_functional_v1` | design-only row; no empirical anchors |

## System hooks

| Hook | Binding |
|---|---|
| sensitivity | consume registered Fisher/margin + corrected-inner-Jacobian + resize/necessity laws; no duplicate metric |
| Pareto | exact contest S over receiver-closed Seg/Pose/rate |
| bit allocation | exact per-section emitted byte receipts and `25/N` break-even |
| autopilot | refused; fit request is `INTERFACE_ONLY_NOT_EXECUTABLE` |
| continual learning | equation/findings preserve the structural law and exact blocker; no fake empirical posterior |
| disambiguation | exact shortest real coder wins internally; exact-S future fit arbitrates representation families |

## Ownership boundaries

- DM1 owns decoded values, #669c home adjudication, and real coder prices for
  the 25 rows.
- DC1 owns outer typed transport, strict parser, scorer-plane receiver,
  exact-byte objective, capacity interface, and v14 falsifier.
- SCHED1/MAIN own whether an executable `DDMEventContinuationV1` engine ever
  exists or runs.
- Frozen scorers and exact contest evaluation retain their existing authority;
  a stored `xi` target is not a PoseNet measurement.

## Capacity edge

```text
Seg head: 5 classes -> K-1 = 4 exact
Lane orbit: ~8 hint -> NULL until realized-through-R rank certificate
Pose xi: upstream first 6 coordinates -> 6 exact
DM1 demand set: 25 exact rows
Total: NULL until lane-orbit rank certificate
```

No component substitutes the approximate `8` as an exact dimension.

## Verification edge

Targeted local suite:

- 7 packet/receiver/objective/capacity/falsifier tests;
- 2 canonical-equation/locked-registry tests;
- result: `9 passed in 0.63s`;
- axis: `[macOS-CPU frozen-scorer advisory]`;
- score claim: false;
- pointer delta: zero.

The receiver proof is a constant-plane `n=24` hard-tail-first structural
fixture through the real R/uint8 chain. Real-video n600 frozen-scorer closure
remains owed.

## Promotion blockers

1. MAIN landing review and serializer SHA.
2. DM1 25-row decoded values plus exact real coder-price records.
3. Exact lane-orbit rank certificate or retained `NULL`.
4. Executable, resumable, per-stage-checkpointed `DDMEventContinuationV1`
   implementation under separate authority.
5. Receiver-closed real-video n600 frozen SegNet/PoseNet result.
6. v14-or-better falsifier pass.
7. Exact archive parse-back plus contest-CPU/CUDA custody before any score or
   promotion claim.
