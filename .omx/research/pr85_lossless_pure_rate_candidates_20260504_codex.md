# PR85 lossless pure-rate candidate lane - Codex - 2026-05-04

## Scope

Lane is disjoint from STBM, PR91/HPM1 transfer, and Lightning CLI hardening.
It is local-only and targets exact pure-rate improvements over the verified
public PR85 frontier without scorer load, CUDA, or remote dispatch.

Focus area: exact lossless repack / non-mask payload self-compression /
pose-stream shrink without decoded PR85 replay semantic drift.

## Source frontier

- Source archive: `experiments/results/public_pr85_intake_20260503_codex/archive.zip`
- Archive bytes: `236328`
- Archive SHA-256: `eb18df2f1b364e513f36933116ec0c1011c6d3bbf8022e16495e0508c6258f5e`
- Single member: `x`
- Member bytes: `236228`
- Member SHA-256: `53bc78effa78cc7850d08a9ddc5488665b93136e9843549d917c17df729a1c50`
- Exact eval artifact: `experiments/results/lightning_batch/exact_eval_public_pr85_adaptive_masking_joint_frame_model_t4_20260503T2332Z/contest_auth_eval.json`
- Recomputed score: `0.25806611029397786`
- Rate contribution: `0.157361`
- Seg contribution: `0.05718500000000001`
- Pose contribution: `0.043520110293977884`
- Evidence grade for score context: existing contest-CUDA/T4 exact eval artifact.

## Public anatomy context

- Public PR86 local archive: `207579` bytes,
  SHA-256 `e67b7c22240dbe33853c19d049b0044a5df16ce5f751ba8f1021cab8ceb03cef`,
  five stored members (`master.pt.gz`, `slave.pt.gz`, `hpac.pt.ppmd`,
  `tokens.bin`, `meta.pt`).
- Public PR90 local archive: `218080` bytes,
  SHA-256 `608ea0355e60faad97b046c27644205d05120ac85ab3e8a99543a75a4ab2dd2d`,
  one stored member `p`.
- Public PR91 local archive: `222404` bytes,
  SHA-256 `4c16d04c746c981feb902e4dd508ffadaf3615e532d351993c3d2f6eccda1b4f`,
  one stored member `x`. This lane did not use HPM1/PR91 transfer.

## Local artifact

- Tool: `experiments/build_pr85_lossless_pure_rate_candidates.py`
- Tests: `src/tac/tests/test_build_pr85_lossless_pure_rate_candidates.py`
- Result summary: `experiments/results/pr85_lossless_pure_rate_candidates_20260504_codex/candidate_summary.json`
- Human summary: `experiments/results/pr85_lossless_pure_rate_candidates_20260504_codex/pr85_lossless_pure_rate_candidates.md`

The tool screens:

1. deterministic strict ZIP repacks of unchanged `x`;
2. Brotli recodes of every Brotli-decodable non-mask segment;
3. P1D1 pose canonical/reordered streams with decoded pose semantic SHA
   equality.

It writes candidate archives only when a transform is strictly byte-negative
and locally decoded-identical. It records `score_claim=false` and
`remote_gpu_dispatch_performed=false`.

## Result

No byte-negative candidate archive was built.

Reason: the best local lossless screen was byte-neutral. PR85 already uses the
strict one-member ZIP minimum overhead (`100` bytes for member name `x`), and
every decoded-identical non-mask/P1D1 recode found by the grid is no smaller
than the source segment.

Best screened rows:

- `bias_brq10_lg16`: `0` bytes delta, pure-rate score delta `0.0`
- `frac2_brq10_lg16`: `0` bytes delta, pure-rate score delta `0.0`
- `frac3_brq10_lg16`: `0` bytes delta, pure-rate score delta `0.0`
- `frac_brq3_lg16`: `0` bytes delta, pure-rate score delta `0.0`
- `model_brq10_lg11`: `0` bytes delta, pure-rate score delta `0.0`
- `pose_brq10_lg16`: `0` bytes delta, pure-rate score delta `0.0`
- `pose_p1d1_order_0_2_brq10_lg16`: `0` bytes delta, pure-rate score delta `0.0`
- `pose_p1d1_order_2_0_brq10_lg16`: `0` bytes delta, pure-rate score delta `0.0`
- `post_brq10_lg16`: `0` bytes delta, pure-rate score delta `0.0`
- `randmulti_brq10_lg11`: `0` bytes delta, pure-rate score delta `0.0`
- `region_brq10_lg16`: `0` bytes delta, pure-rate score delta `0.0`
- `shift_brq10_lg16`: `0` bytes delta, pure-rate score delta `0.0`

ZIP deflate repack was also screened and is byte-negative false: deflating
the single `x` member increases the archive by `75` bytes.

## Dispatch gate

No dispatch is allowed from the current summary because
`built_candidate_count=0`.

If a future run produces a byte-negative manifest, exact-eval dispatch requires:

1. claim a non-conflicting lane with `tools/claim_lane_dispatch.py claim ...`;
2. verify source and candidate archive SHA fields still match current files;
3. rerun the decoded-semantic parity proof;
4. run PR85 replay/fixed-runtime local preflight against the exact candidate
   archive;
5. run `experiments/contest_auth_eval.py --device cuda` through the canonical
   `archive.zip -> inflate.sh -> upstream/evaluate.py` path;
6. adjudicate structured `contest_auth_eval.json` and recompute score from
   components before making any score claim.

