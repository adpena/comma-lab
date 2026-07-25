---
title: Codex session summary — DDM CC2 coder races
utc: 2026-07-25T04:12:00Z
lane_id: ddm_cc2_coder_races
commit: 7b28ff69ba
research_only: true
score_claim: false
pointer_moved: false
main_landing_review_required: true
---

# TIER-0 outcome

Landed the exact Race 2 quantizer proxy comparison and Race 3 per-counted-stream five-arm coder pricing apparatus on isolated branch `codexwt/ddm_cc2_coder_races_20260725T030606Z`.

- Race 2: Cool-Chic-v5 terminal proxy wins the bounded fixed-theta instance at equal `139538 B`, with `delta d_seg=-2.4668375651071273e-06`, `delta d_pose=-0.006400773500303103`, and `delta advisory action=-0.001924772136713493`.
- Race 3: 27 leaves, 135 exact coder frames, and a derived mixed-archive estimate of `-3422 B`; G1 and PC1 remain raw.
- Exact boundary: Race 3 is price-table-only until the generic context-frame interpreter is wired into the receiver and an exact mixed archive is reconstructed.
- Verification: 49 passed, 1 optional skip; 57 randomized round-trips; 114 malformed-frame rejections; two clean review passes per modified Python file; byte-identical resume replay.

# Durable anchors

- Findings: `.omx/research/codex_findings_ddm_cc2_coder_races_20260725T035900Z_codex.md`
- Receipt: `.omx/research/ddm_cc2_coder_races_receipt_20260725T035900Z.json`
- DAG/feed: `.omx/research/ddm_cc2_coder_races_DAG_FEED_20260725T035900Z.md`
- Done marker: `.omx/research/ddm_cc2_coder_races_20260725T035900Z.done`
- Full SSD receipt: `/Volumes/VertigoDataTier/pact/experiments/results/ddm_cc2_coder_races_20260725T030606Z/ddm_cc2_coder_races_receipt.json`

# Pending MAIN decision

MAIN must review commit `7b28ff69ba` before landing. If accepted, the next highest-EV build is restricted to receiver integration for the eight negative-delta Race 3 leaves, followed by exact composition parse-back. A true v5 adoption trial requires multi-seed or schedule-retraining evidence; this proxy result alone is not promotion authority.

Pointer `0.1910828242 [contest-CPU]` remains unchanged.
