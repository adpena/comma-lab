# Codex Findings: SNeRV Current Limitations And Candidate Feedback Scope

Created: 2026-06-02T12:45:42Z

Axis: local advisory only. No score, rank, kill, promotion, CPU-auth, or CUDA-auth claim.

## Verdict

The statement "SNeRV is still rate-blocked on stored LF unless representation changes" is directionally true for the current full600 SNeRV artifact, but it is not a proven fundamental limitation of SNeRV as a design family.

What is proven:

- The current full600 SNeRV package is rate-fatal: `archive.zip` is `10,057,021` bytes, with `0.bin` compressed to `10,023,339` bytes.
- The LF payload dominates the binary: the full600 LF predictor profile measured current LF payload at `9,996,235` bytes.
- Simple deterministic lossless LF predictors do not collapse it. The best measured candidate, raster delta, saved only `355` bytes versus current LF payload.
- Post-export materializers are useful reusable chain pieces, but they cannot rescue a 10 MB explicit LF payload into the 178 KB frontier regime.

What is not proven:

- It is not proven that a fully optimized SNeRV family is fundamentally uncompetitive.
- It is not proven that native MLX score-aware training, learned LF/HF generation, SR low-res carrier structure, symbolic/shared residual grammar, or decoder-weight scorer fitting cannot move the rate-distortion operating point.
- It is not proven by full600 local scorer replay; current full600 SNeRV package ingest is blocked on full-video MLX prefilter/local CPU replay.

## Current Evidence

Partial SNeRV advisory smokes:

- 1-pair archive packets are around `72,962` to `73,001` bytes before package overhead and around `100,251` to `106,895` bytes as package archives.
- Advisory distortion is nontrivial: typical 1-pair `score_linf` around `1.18`, `d_seg_mean_linf` around `0.00896`, `d_pose_mean_linf` around `0.00566`.
- These are useful local signals only, not promotion evidence.

Full600 SNeRV package:

- Archive: `/Volumes/VertigoDataTier/pact/snerv_lf_delta_sharedshape_bits2p0_l2_affine_lastframe_package_full600_metadataonly_20260602T032545Z/archive.zip`
- SHA-256: `13db5e6e659bc4b0aa6a6925119673749e2b72a8cf1a2e6e943603d4821608ca`
- Bytes: `10,057,021`
- Dominant member: `0.bin`, compressed `10,023,339` bytes, uncompressed `10,030,524` bytes.

Full600 LF profile:

- Source memo: `.omx/research/codex_findings_snerv_full600_lf_predictor_profile_20260602T111906Z_codex.md`
- Current LF payload: `9,996,235` bytes.
- Best simple predictor: raster delta, `9,995,880` bytes, `-355` bytes.
- Verdict: `simple_lossless_lf_predictors_do_not_collapse_full600_lf_payload`.

Scorer-loop decoder-QAT evidence:

- Strict 1-pair smoke did not improve under strict SegNet gate.
- Pose-hard, SegNet-slack smoke improved 1-pair score from `1.1817133779961255` to `1.1718591965053646`, mainly via PoseNet improvement, while slightly worsening SegNet.
- That is a real local direction signal, but it remains blocked by full600 coverage, CPU/CUDA authority, and byte-optimized decoder grammar.

## Candidate Feedback Scope Fix

The planner previously risked treating partial-pair byte measurements as ready feedback for full600 modelsize candidates. That was false authority.

This landing changes candidate byte feedback to carry:

- `candidate_num_pairs`
- `measured_num_pairs`
- `feedback_scope`
- `scope_matches_candidate`
- `feedback_ready`

Partial-pair runs now remain harvestable but fail closed with `partial_pair_byte_feedback_only`.

Real SNeRV smoke proof:

- Report: `/Volumes/VertigoDataTier/pact/experiments/results/codex_snerv_candidate_feedback_scope_smoke_20260602T124616Z/compact_renderer_mlx_spine_runner_report.json`
- Candidate: `snerv_np600_lv5_lfb1p5_stepb0p5_int8_symmetric_ceil216000`
- Candidate pairs: `600`
- Measured pairs: `2`
- Measured archive bytes: `43,296`
- Measured packet bytes: `10,259`
- Feedback scope: `partial_pair_advisory`
- Feedback ready: `false`
- Blocker: `partial_pair_byte_feedback_only`

## Consequence

The correct next SNeRV work is not "more lossless LF predictor craft" and not demotion. It is representation-changing, score-aware work:

1. Native MLX SNeRV train/export/archive adapter.
2. Full-video MLX prefilter and local CPU replay for byte-closed SNeRV packages.
3. Learned or symbolic LF/HF generator that stores fewer explicit LF coefficients.
4. SR low-res carrier path: encode below scorer input resolution and protect PoseNet geometry explicitly.
5. Pose-hard, score-primary decoder fit with explicit SegNet slack sweeps.
6. Wavelet-group P18/P19 saliency binding so rate is spent only on scorer-causal atoms.
7. Mixed-precision decoder grammar and byte-optimized receiver payload for accepted learned decoder updates.

SNeRV should stay alive as a top-priority carrier/enhancer lane, but current full600 LF-store SNeRV is not close to frontier until the explicit LF payload collapses or is replaced.

