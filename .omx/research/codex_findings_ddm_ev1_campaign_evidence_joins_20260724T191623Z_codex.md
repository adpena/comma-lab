---
schema: codex_findings.v1
date_utc: 2026-07-24T21:09:58Z
lane_id: ddm_ev1_campaign_evidence_joins
research_only: true
score_claim: false
main_landing_review_required: true
---

# Codex findings — DDM EV1 campaign evidence joins

## Verdict

`MEASURED_ADVISORY_EVIDENCE_COMPLETE_PRICING_PENDING_MS2R`.

- V19 is receiver-closed for exact pair ids `0..599`; the measured 1,588-byte
  candidate archive delta has one global home and no invented per-pair share.
- RD1 has 162/162 exclusive byte homes and 162/162 exact uint8 histograms.
  Of the homes, 108 use a shared-k G4 class-level prior and 54 are per-frame.
  No aggregate is falsely labeled `shared_clip`.
- Real Brotli Q11 output is 14,730 bytes,
  SHA-256 `19e539de179818de148a4a5cdd41b74fb1bed1844a0778041ac68aee3017f91a`,
  with byte-identical parse-back.
- The only metric is
  `exact_composite_R_rank4_margin_fisher_plus_pose6_quadratic`; Euclidean-naive
  rows are refused.  All 162 prices remain null and ms2r-owned.

The final EV1 receipt is 668,997 bytes, SHA-256
`28af280a6c40216616eb4cbd90b283b68822d3c5f9a604efbed2a638b46d9c0a`.

## Adversarial findings and fixes

1. Prior MENU1 score batches could not be joined to current receiver cameras:
   their camera hashes drifted.  The producer refused the mixed custody and
   freshly rescored every RD1 endpoint.  This is local measurement-harness
   payload evidence, not contest archive authority.
2. The first completed receipt left the G4 reuse span looking edge-specific.
   Round 1 forced a regenerated receipt: every home now says the k is a
   measured G4 class-level prior, and transient mass excludes the two ξ-proxy
   events (`47,880/47,880`).  G4 ξ remains a metric proxy, not physical BEV.
3. Generic decoder, solver, and transport interpreter code is zero-rate; only
   irreducible video-derived statistics are counted.  ξ video parameters are
   not silently declared free.

## co2 and remaining authority

co2 consumes the validated receipt as one schema authority.  Its deterministic
campaign state digest is
`1482921f0bbba8880012c9a678a68cf747abac6cb7166e959401ed6a299ea0ec`;
the lambda join is `N600_EXACT_COMPLETE` with 600 pair and 3,000 site rows.
The historical V19-592 and RD1-home evidence blockers are removed.

Remaining exact duties:

- `BLOCKED_J8F_REALIZED_VERDICT_TELEMETRY` remains an independent co2 blocker.
- `MS2R_TOLERANCE_CAPPED_DIMENSION_PRICING` remains a duty, not an EV1
  evidence blocker.  A superseding hash-bound handoff was sent to the ms2r
  inbox.
- Contest CPU and contest CUDA are unmeasured.  Pointer
  `0.1910828242 [contest-CPU]` is unchanged.
- Global lane validation still reports 110 historical missing-evidence paths
  outside this lane; no unrelated lane was altered.

Focused tests: 46 passed.  Ruff and diff checks are clean.  MAIN landing review
is required.

