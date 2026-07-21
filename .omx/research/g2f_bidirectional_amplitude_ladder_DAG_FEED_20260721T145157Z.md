---
schema: g2f_bidirectional_amplitude_ladder_dag_feed.v1
task_id: "578"
lane_id: lane_g2f_bidirectional_amplitude_ladder_578_20260721
research_only: true
status: MEASURED_QP_NO_ADMISSION_N64_FAMILY_OPEN
authority: "[macOS-CPU advisory]"
score_claim: false
promotion_eligible: false
pointer: "0.1910828242 [contest-CPU] UNMOVED"
main_landing_review_required: true
---

# DAG FEED: G2f bidirectional amplitude ladder

## Measured path

```text
G2e 64 signed-branch priors (hash-custodied, no remeasurement)
  + exact factor-2 R gain
  -> derived ladder [0.5, 1, 2, 4, 8, 16]
  -> paired +a/-a receiver responses
  -> odd/even decomposition + class/bucket/stratum trust
  -> amplitude 1.0 knee
       209/462 usable trust rows
       76/254 usable pair-directions
  -> best usable rung per effective direction
  -> 5/64 pairs reach receiver-closed active-set QP
       5 QP_INFEASIBLE
       59 TRUST_REGION_REFUSED
  -> 0 admitted corrections / 0 correction bytes
  -> D4 NOT RUN
```

## Fail-closed routing

```text
0 receiver admissions at n64
  -> refuse n600 continuation
  -> refuse +95,094-byte headroom spend
  -> do not route a candidate stream to #598 r5
  -> keep rank-4 quotient and integer-lattice families OPEN
  -> queue n16 exact integer-lattice/NFS-style preimage disambiguator
```

## Six-hook wire-in

1. Sensitivity map: expose the measured rung response field, with amplitude
   `1.0` as the knee and per-class/per-margin/per-stratum trust counts. Never
   replace these rows with a pooled proxy.
2. Pareto constraint: correction allocation remains pinned to zero because
   no receiver-closed correction exists. The base is `121,128` bytes and the
   registered rate threshold is unchanged.
3. Bit allocator: consume only `ADMITTED_RECEIVER_CLOSED` packets. Empty,
   trust-refused, and QP-infeasible packets contribute no marginal value.
4. Cathedral/autopilot dispatch: rank the n16 exact-integer-lattice
   disambiguator above any n600 local-linear replay. No launch authority is
   emitted by this feed.
5. Continual-learning posterior: append-only canonical equation anchor
   `realization_g2f_bidirectional_amplitude_n64_20260721` records the knee,
   five infeasible QPs, zero admissions, receipt hashes, and scope.
6. Probe disambiguator: compare preserved bidirectional local-linear charts
   against direct integer-lattice/NFS-style exact preimages on the same n16
   states. Require hard receiver closure before scaling.

## Triality delta

- DSL/code: bidirectional branch, odd/even, rung trust/selection, and strict
  receipt contracts extend `realized_secant_custody.py`; the existing G2
  runner owns resumable measurement and hard admission.
- DAG: the quantization knee and joint-QP incompatibility are separate nodes;
  a nonempty local trust region no longer impersonates receiver admission.
- Equations: `predict_project_realization_admissibility_v1` is unchanged and
  receives one empirical anchor with `accepted=false`.

## Custody

- receipt path:
  `/Volumes/VertigoDataTier/pact/evidence/g2f_amplitude_20260721/receipt.json`
- file SHA-256:
  `0a09b7b5022ff64eebc54d086f00c89378d7eb7091c5963cf1056120469bc38e`
- canonical receipt SHA-256:
  `3ddd1a51b2e238fe9f20e85f0f6b293df5cbbffa5f2007eb71e85cedecfc9ce1`
- config SHA-256:
  `a8a2393e266da0aa629898b3d935e70b678244e1c48940af109c428906ee2a3e`
- verdict scope: exact contiguous n64 openpilot-base local-linear charts only;
  no n600, contest-axis, score, promotion, quotient-family, or
  integer-lattice-family claim.

MAIN landing review is required before this DAG feed becomes canonical.
