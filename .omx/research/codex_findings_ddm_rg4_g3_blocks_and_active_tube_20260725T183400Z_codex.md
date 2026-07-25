---
title: DDM RG4 RG3 block closure and candidate-local PC1 Pose6 active-tube findings
date_utc: 2026-07-25T18:34:00Z
lane_id: ddm_rg4_g3_blocks_and_active_tube
research_only: true
score_claim: false
promotion_eligible: false
pointer_moved: false
main_landing_review_required: true
verdict: FALSIFIED_25_POSITIVE_RG3_CLOSURES; TWO_CANDIDATE_LOCAL_DESCENTS_MEASURED_OUTSIDE_ACTIVE_TUBE
---

# Outcome

The preregistered RG3 falsifier fired. The terminal production RG1/RG2/RG3
checkpoint corpus already measured both signs and every admissible magnitude
for all 25 missing exact pair/bucket blocks. None changed a target-bucket
event, so this arm did not rerun settled scorers or mislabel the rows as
positive causal coverage:

- `coverage_proven=false`
- `missing_block_count=25`
- `positive_closure_count=0`
- `typed_exclusion_count=25`
- `producer_rerun_eligible=false`

The exact per-instance blocker is
`NO_TARGET_BUCKET_EVENT_CHANGED_BY_ANY_COUNTED_RG3_MAGNITUDE_OR_SIGN`.
The typed successor split is:

- 10 `WORLDSHEET_EVENT_INDEXED_TYPED_INTERFACE_ARC`
- 9 `FISHER_MARGIN_SITE_LOCAL_PER_STRATUM_CODEBOOK`
- 6 `CURVELET_OR_SHEARLET_BOUNDARY_ARC_CODEBOOK`

The rebuilt aggregate passed the unmodified current ms2r_r3 strict source
validator with the original false-coverage/25-missing semantics. The MS5 loader
table was emitted byte-identically at SHA-256
`57d3954bc4661f5da48aae943433a7c5f611639b2d5a24854a01d658fd52aebd`.

# Corrected-inner-Jacobian boundary

The global #583 bank is not complete. Its SHA-bound status record says
`first_order_vjp=MEASURED_REAL_N600`,
`realized_backbone_secants=ABSENT`,
`qp_receiver_closure=ABSENT`, and `formalization=FORMALIZATION_PENDING`.
Accordingly, no #583 completion claim is made. The 25 obstruction verdicts
rest on the actual receiver/composite-R/uint8 signed-quantum checkpoint rows,
not on an inferred corrected-J bank.

# Candidate-local PC1 receiver

The old active-home PC1 receiver was not used as a replacement for either
candidate. This arm lands and measures the source-preserving receiver

`C(q;W) = clip_u8(W + P(q;W) - P(0;W))`,

where `W` is the exact SHA-pinned candidate and `P` is the admitted PC1
receiver. Actual W_joint and W_seg smoke proves `C(0;W) == W` byte-for-byte.
The generic adapter is free receiver code; the nested parent plus PC1 packet is
the counted archive. Each nonzero proposal uses `q=256`, exactly one physical
PC1 quantum.

Each candidate received the same deterministic budget: 32 bit-reversal knot
stages, six Pose axes, both signs, 384 proposals, a cap of eight accepted
steps, a preserved checkpoint after every stage, and source/final n600
batch-32 frozen-scorer verdicts. A `psutil` threshold check of at least 20 GiB
preceded every scorer call.

# Measured n600 results

All rows below are `[macOS-CPU frozen-scorer advisory]`, `score_claim=false`.

| Candidate | Base SHA | Source B | Final SHA | Final B | delta d_seg | delta d_pose | delta S |
|---|---|---:|---|---:|---:|---:|---:|
| c1/composed-line current | `2a2c0367150f...` | 138,813 | `d86710793f77...` | 139,685 | -0.007830166286892364 | -4.218779488253894 | -1.9373849668656398 |
| ws2 W_seg | `264a09abb8f6...` | 138,031 | `7fb0a4e63acd...` | 138,897 | +0.00045073615180121415 | -27.644693661060387 | -3.756216491086043 |

The c1 final has `d_seg=0.061912604437934025`,
`d_pose=31.281041321337113`; W_seg final has
`d_seg=0.024575246175130207`, `d_pose=118.72023879381737`.

Both states remain outside the full per-pair active tube: all 600 pairs are
outside in each source and final state. The equal-rank-share diagnostic marks
exactly one of six Pose outputs active in both finals: `tx`. This dimension
label is diagnostic only; membership authority remains the full six-dimensional
quadratic on every pair. Final equal-share slacks are carried in the
candidate-local receipts.

# Interpretation

The source-local PC1 line is a real measured descent direction at both
operating points, but it does not place either candidate inside the sealed
Pose validity tube. Therefore these rows do not activate the Task #701
homotopy and do not promote PC1, RG3, or either final archive. The useful
system signal is sharper:

1. RG3 needs one of three named coordinate-family extensions; more magnitude
   or sign sweeps of the existing production alphabet are settled duplication.
2. PC1's measured benefit is overwhelmingly through the `tx` Pose output even
   though accepted counted coordinates are rotational. Future solve-local
   work must treat that as a candidate-specific realized coupling, not equate
   parameter axes with Pose output axes.
3. W_seg trades a small Seg regression for a larger nonlinear Pose-term gain;
   c1 improves both distortion components. The two operating points must
   remain separate.

# Verification and custody

- RG3 aggregate semantic rebuild from four checkpoint roots: pass.
- Current ms2r_r3 strict validator replay: pass, no semantic weakening.
- Source metrics reproduce settled W_joint and W_seg batch-32 values.
- Two candidates x 33 contiguous search checkpoints: pass.
- Two candidates x two states x 19 n600 chunks: pass.
- Chunk sums reproduce receipt errors and Pose SSE: pass.
- Every chunk memory receipt admits at least 20 GiB: pass.
- Final archive parse-back/hash: pass.
- Fresh complete-run resume after landing: pass.
- SSD output: 151 files, 2,211,638 bytes, digest-chain SHA-256
  `84e15687a5e13e4a9f959365534fd48c4ddd79ff1522eb3f746df2b9a4dff597`.

# Durable artifacts

- Coverage receipt:
  `.omx/research/ddm_rg4_g3_blocks_and_active_tube_20260725T165958Z/rg3_25_block_coverage_receipt.json`
- Enriched aggregate:
  `.omx/research/ddm_rg4_g3_blocks_and_active_tube_20260725T165958Z/ddm_rg4_receiver_support_summary.json`
- Candidate receipts:
  `.omx/research/ddm_rg4_g3_blocks_and_active_tube_20260725T165958Z/c1_composed_line_current_active_tube_receipt.json`
  and
  `.omx/research/ddm_rg4_g3_blocks_and_active_tube_20260725T165958Z/ws2_w_seg_138031_active_tube_receipt.json`
- Complete run receipt:
  `.omx/research/ddm_rg4_g3_blocks_and_active_tube_20260725T165958Z/run_receipt.json`
- Resumable external stages:
  `/Volumes/VertigoDataTier/pact/experiments/results/ddm_rg4_g3_blocks_and_active_tube_20260725T165958Z`

# STORES CONSULTED

`CLAUDE.md`; `AGENTS.md`; craft handoff; v7.5 operating contract; delegated
authority and both watched inboxes; lane/subagent/frontier state; terminal
RG3/MS6 summary, assignments, MS5 table, and four SSD checkpoint roots; #580
receiver support implementation; #583 status record; MS4D Pose metric; old
PC1 admission; current ms2r_r3 strict join; WS2/WS4 receipts; Menu1 scorer
custody; exact n600 target cache. Pointer `0.1910828242 [contest-CPU]`
unchanged.

# Lane-maturity validation scope

This lane is internally L2 and all four evidence paths recorded for
`impl_complete`, `real_archive_empirical`, `strict_preflight`, and
`three_clean_review` exist. The repository-global
`tools/lane_maturity.py validate` remains nonzero on **110 pre-existing missing
legacy evidence paths** outside this lane. That unrelated registry debt is not
silently reclassified as a failure of this landing and remains owed by MAIN's
registry owners.

# MAIN landing review required

MAIN must independently verify the 25-row falsifier and scope, the explicit
#583 incomplete-bank boundary, source-local zero identity, both base/final
archive hashes, all n600 chunk sums, full-quadratic tube non-membership, the
diagnostic-only status of per-dimension activity, false-authority labels, and
that no old 07-24 PC1 receipt or live j12 state was mutated.
