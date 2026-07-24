# DDM PC1 Pose Stream Admission — Build Specification

`research_only=true` · `score_claim=false` · `pointer_moved=false` ·
`main_review_required=true`

## Authority and stores consulted

- delegated authority:
  `/Users/adpena/Projects/pact/.omx/tmp/codex_runs/ddm_pc1_pose_stream_admission_20260724T191805Z.wrapped.prompt.txt`,
  SHA-256 `f54aab6f763ea6a0c5cd09a837f379495f3871c15207e054d123af0a0a851842`,
  7,111 bytes;
- `CLAUDE.md`, `AGENTS.md`, `docs/llm_handoff.md`, the canonical v7.5 and v8
  specifications, the DDM dimension-completeness contract, and the latest DDM
  IC2/E2/PA1/MS4d/MS6/J8e findings;
- exact settled W_seg and W_joint archive custody in
  `.omx/research/ddm_ws2_warm_start_custody_producer_receipt_20260724.json`;
- exact MS4d n600/batch32 pose metric, SHA-256
  `5e06cc78711a6ca6984c907600a25816cdecc6239903f782d85bcf9473a8f1bc`;
- task inbox through `2026-07-24T19:50:42Z` and broadcast through
  `2026-07-24T17:39:13Z`.

R1 `dxi` is harvest-signal-only. This build does not open, parse, hash, load,
copy, compose, or anchor any R1 payload or weights.

## Built object

`pose/pc1.ddp` is the sole counted owner for two effects:

1. a 32-knot, six-axis int16 twist-control curve decoded at all 600 pair
   positions by deterministic linear interpolation; and
2. a 32-knot, four-phase int8 luma-residual home aligned with the scorer stem
   lattice.

The zero counted home is 40 bytes. The separately generated 349-byte nonzero
geometry-quantum packet is probe-only and is not included in either candidate.
The parameter map exposes 320 stable descent coordinates to #366.

The receiver derives a continuous ground-plane depth field from camera
intrinsics and height, substitutes a contact-depth stratum on the decoded
Movable mask, and computes

`frame_0 = W_{-xi/2,D}(source) + residual`, then
`frame_1 = W_{xi,D}(frame_0)`.

Inactive decode returns the exact parent bytes before any resize or transform.
For descent, #366 receives the exact solved-plane target derived for free from
the decoded W parent by the frozen evaluator resize and BT.601 YUV6 polyphase
map. No solved-plane values are stored in the packet.

## Execution and storage

All task packages live in the owned SSD venv:

`/Volumes/VertigoDataTier/pact/.venvs/ddm_pc1_pose_stream_admission_20260724T191805Z`

All measurement checkpoints and candidate archives live under:

`/Volumes/VertigoDataTier/pact/experiments/results/ddm_pc1_pose_stream_admission_20260724T191805Z_curve32`

The builder preflights 20 GiB free space, writes every batch checkpoint
atomically, refuses changed immutable checkpoints, and reuses all 19 stages per
parent on resume. Parent raw stages remain external certified inputs. No
temporary bulk survives success.

## Admission fence

Admission requires all of:

- canonical packet and complete composition parse/re-emission;
- exact inactive output byte identity;
- nonzero-q causal support through the landed MS6 composite-R detector, using
  the active zero-q home as the baseline;
- exact replay of both parent archive byte strings;
- unique #417 effect ownership;
- fresh n600/batch32 frozen-scorer measurement;
- direct, non-telescoping `S(W + PC1) - S(W)` for both parents.

Admission does not imply descent success, tube membership, promotion, a contest
score, or pointer movement.
