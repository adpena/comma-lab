# PR75 Next Candidate After QP1-Fixed Wave - 2026-05-03

Scope: local planning only. No remote GPU, Lightning, Modal, Vast.ai, or other
dispatch was performed. This note ranks the next PR75/QZS3 packer candidates
after the active QP1-fixed exact-eval wave already claimed in
`.omx/state/active_lane_dispatch_claims.md`.

Planning artifact:
`experiments/results/pr75_next_candidate_after_qp1_wave_20260503_codex/candidate_priority.json`

## Active Wave Excluded

The newest active claims at `2026-05-03T06:07:44Z` already cover:

- public PR75 QP1 replay on T4;
- `c067_pr75_actions_lag_eval_top67_p6`;
- `c067_pr75_actions_lag_eval_pose2_top67_p6`;
- `c067_pr75_actions_beam_pose2_top55_p3`.

Those are not counted as next candidates here.

## Read

The best exact T4 parent in hand is still `c067_pr75_actions_top40_p3`:
score `0.3155226919767294`, bytes `276386`, SHA-256
`9feef7ffaa254f9e5408996a122682757a054144a3000539553786c5292b7d0a`.

The active wave should answer whether the current QP1-fixed runtime preserves
public QP1 pose precision and whether P6 delta-varint action payloads are
exact-eval safe under the new runtime tree. Until then, the next queue should
prefer candidates that either preserve decoded streams or have an exact P3
parent.

## Top Next Candidates

1. `c082_p6_delta_varint_actions_stream_resweep`
   - Archive:
     `experiments/results/c082_pr75_lossless_repack_20260503_worker/c082_p6_delta_varint_actions_stream_resweep/archive.zip`
   - Bytes/SHA: `276394`,
     `9b78333dd39c12c986ca7fc02a4bb35a6a61ef5a1cc0c9c9c840820eb840058a`
   - Expected score: `0.3154887970644586`
   - Rationale: exact T4 `actions_only_p3` score minus the 66-byte rate delta.
     Decoded masks, renderer, QP1 poses, and action records are preserved; this
     is the highest-confidence post-wave byte win and not a no-op.

2. `c067_pr75_actions_top40_p6`
   - Archive:
     `experiments/results/c067_pr75_tile_action_compiler_p6_trace_ev_20260503_codex/c067_pr75_actions_top40_p6/archive.zip`
   - Bytes/SHA: `276342`,
     `0ec53e5b871149ed6eea56c0b9bcca3baec998d5bfad4f371979e0c90e62cea8`
   - Expected score: `0.3154933941827920`
   - Rationale: exact T4 `top40_p3` parent score minus the 44-byte P6 rate
     delta. Same selected runtime action records as the current A++ top40 P3
     parent; only the action wire coding changes.

3. `c067_pr75_actions_lag_eval_pose4_top67_p6`
   - Archive:
     `experiments/results/c067_pr75_tile_action_compiler_p6_trace_ev_20260503_codex/c067_pr75_actions_lag_eval_pose4_top67_p6/archive.zip`
   - Bytes/SHA: `276338`,
     `de7d549ec4437b3ed9508c1f62f8a79bef88440ba3db0f4a7de5a2c83fe160ef`
   - Expected score: `0.3154606501519654` planning-only
   - Rationale: next non-active P6 pose-weight sensitivity check. It has the
     best remaining nominal trace estimate after the active lag-eval top67 and
     pose2 P6 claims, but it should promote only if the active P6/QP1-fixed
     wave preserves the trace ordering.

Near miss: `c067_pr75_actions_top49_p6` is a reasonable fallback
(`276368` bytes, trace estimate `0.3154680901896367`) but has less exact-parent
confidence than `top40_p6` and less orthogonality than `lag_eval_pose4_top67_p6`.

## Promotion Boundary

All three remain non-promotable until exact CUDA auth eval runs on the exact
archive bytes under the QP1-fixed runtime tree. If the active wave shows P6
runtime drift or QP1 pose precision does not close the public replay gap, stop
and repair the runtime contract before adding more packer candidates.
