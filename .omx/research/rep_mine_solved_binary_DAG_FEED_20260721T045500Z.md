# DAG FEED — solved-binary native representation

**UTC:** 2026-07-21T04:55:00Z  
**Lane:** `lane_rep_mine_solved_binary_20260721`  
**Axis:** `[macOS-CPU advisory]` · no score/candidate/promotion  
**Pointer:** `0.1910828242 [contest-CPU]` UNMOVED

## Feed payload

```text
FEED-rep-mine-solved-binary-596
  upstream custody
    M2 exact raw/archive
      archive 1,717,172,741 B, d_seg=0, d_pose=0
      raw sha a7192f93..., archive sha 0fee1b74...
    gt_n600 scorer planes cf8d8360...
    teacher logits 41d3ef53... (fp16 tie caveat)
    #580 resize projector
    #519 gauge premise (re-measured here, not copied)
    G1 transport + G3 17,926 flip identities
    r1b3/r2b realized sparse stream

  measured invariances
    resize
      80.6742% dimension DERIVED
      45.1668% actual M2 energy in ker(A) MEASURED
      22.6969% implemented blind coordinates MEASURED
    logits
      31.1071% gauge energy MEASURED
      rank 4, fifth centered direction 0 MEASURED
    partition
      99.7115% horizontal / 98.9119% vertical equal adjacency
      21,304 digital connected cells
      83.1564% quotient energy explained by cell constants
    dynamics
      0.0150857 bit/cell position+adjacency+pose-proxy entropy
        optimistic 600-f1 empirical model; Pose side information excluded
        not authoritative sequential xi transport and not an information lower bound
      17,926 flip identities -> 2,724.873 B ideal target symbols
      colex sites -> 31,653.132 B ideal

  representation decision
    values are wrong primary object
      exact reconstructive owned-seed payload estimate = 1,571,792,105 B
        new manifest/container framing owed; not an archive size
      valid full-archive upper bound remains M2 = 1,717,172,741 B
      exact resize numerators = 2,161,746,454 B
    constraints/cells/dynamics are correct primary object
      context entropy ideal estimate = 222,447.027 B
      this model is 6,225.027 B over 216,222 B before Pose/value/receiver
      a better model may be smaller; no impossibility inference
    compose next
      rank 1: causal constraint seed + deterministic predict/project
      rank 2: MS 1-skeleton + persistence vineyard
      rank 6: causal appearance phase response + true exceptions

  consumers
    S2-compose #595
      consume ranked M4 table and exact lower/upper bounds
      do not copy ideal event bytes as a standalone witness estimate
    v10 spec
      define typed counted seed / free interpreter boundary
      add receiver parse-back and 30-minute decode gate
    #535 / #30
      owns joint rollouts and commutator-aware interaction matrix
      this lane must not fork that work
    #557
      arithmetic coder consumer for causal partition stream
    G3/r1b5
      colex event-site representation; Fisher payload custody guard

  blockers
    B1 RGB/Pose realization from constraint seed absent
    B2 finite coder, model/header, archive framing absent from 222,447 B ideal
    B3 classical MS critical points/separatrices/persistence/vineyard absent
    B4 R2 appearance chart exact equal-fidelity stream absent
    B5 per-stream RD curves absent
    B6 all off-diagonal receiver interactions unmeasured

  authority routing
    build authority: none
    launch authority: none
    score authority: none
    promotion authority: none
    equation registration: FORMALIZATION_PENDING
```

## Required next-node contract

1. Emit one deterministic counted constraint seed. A video-specific table, basis, contour, ranking,
   persistence order, chart parameter, or exception is counted. A generic decoder that remains correct
   for a different dashcam seed is free.
2. Decode by causal context plus constrained predict/project. Preserve integer resize numerators,
   argmax inequalities, Pose constraints, and deterministic tie behavior.
3. Parse back the emitted seed and measure exact archive bytes, d_seg, d_pose, and decoder wall clock.
4. Add the MS/vineyard and appearance-chart streams only through #535 joint rollouts. Measure both
   stream orders and the commutator residual; do not add isolated marginal gains.
5. Reject any stream whose joint marginal is below
   `lambda*=25/37,545,489 = 6.6585895e-7 score/B`. The isolated 27,213 B r2b stream currently returns
   only 6.806% of that price and is eaten absent a positive interaction.

## Triality

- **DSL:** eventual typed modes: `constraint_seed`, `ms_vineyard`, `appearance_chart`, and
  `exact_exceptions`; no invented flags and no launch from this feed.
- **DAG:** #595 is the composition consumer; #535 owns interactions; #557 owns entropy coding.
- **Equations:** kernel energy, gauge energy, context entropy, and colex ideal length remain
  `FORMALIZATION_PENDING` because they are empirical/ideal anchors without general receiver evaluators.

## Durable evidence

- `.omx/research/rep_mine_solved_binary_20260721T045500Z.md`
- `.omx/research/rep_mine_solved_binary_20260721T045500Z.json`
- `tools/measure_rep_mine_solved_binary.py`
- `/Volumes/VertigoDataTier/pact/evidence/rep_mine_20260721/full_n600/receipt.json`

**MAIN landing review required:** review the Python measurement semantics, the non-additivity verdict,
the counted-seed/free-interpreter classifier, and the lane/state changes before merge.
