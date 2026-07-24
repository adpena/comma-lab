---
schema: dag_feed.v1
date_utc: 2026-07-24T20:37:08Z
feed_id: FEED-DDM-CO2-CAMPAIGN-EVIDENCE-JOIN-20260724
lane_id: ddm_ev1_campaign_evidence_joins
research_only: true
execution_allowed: false
score_claim: false
main_landing_review_required: true
---

# DDM co2 campaign evidence-join elevation

This append-only feed supersedes only the evidence status in
`ddm_co2_costate_campaign_sense_DAG_FEED_20260724T190025Z.md`.  It does not
erase that historical pre-join snapshot.

## Schema-authority delta

`ddm_ev1_campaign_evidence_join_receipt.v1` is now a required, content-hash
verified co2 source.  `validate_campaign_evidence_join` is the single schema
authority used before any consumer sees the evidence.

- V19 blocker 2: `8/600, 592 owed` → exact receiver-closed `600/600`, with the
  archive-rate delta counted once in a global shared home and never fabricated
  per pair.
- RD1 evidence blocker 3: missing typed homes/histograms → `162/162` exclusive
  G4-amortized homes plus `162/162` exact uint8 receiver-step histograms.
- RD1 price status remains `0/162 actionable`.  This is not an evidence
  blocker: the tolerance-capped dual solve is exclusively owned by ms2r.

## Consumer fanout

The shared campaign state carries, for every RD1 row:

```text
byte_home_scope
byte_home_k and exact numerator/denominator
amortized_bytes_per_frame
exclusive byte range
exact receiver uint8 histogram and moments
null ms2r-owned price.
```

The digest and dashboard expose the measured V19 and RD1 evidence statuses.
The duty queue no longer requests either evidence join; its remaining RD1
duty is exactly `MS2R_TOLERANCE_CAPPED_DIMENSION_PRICING`.  The activation nag
retains the independent J8F measurement blocker.

## Authority boundary

The co2 state is advisory, `actuation=NONE`, `score_claim=false`, and
`promotion_eligible=false`.  Fresh RD1 endpoint replay is local
`[macOS-CPU frozen-scorer advisory]` measurement-harness evidence, not contest
archive authority.  Pointer remains `0.1910828242 [contest-CPU]`.

MAIN landing review is required before this schema source or its consumer
status can be treated as landed repository truth.
