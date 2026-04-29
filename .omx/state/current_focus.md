# Current Focus -- 2026-04-29

## Score State

Current best measured contest-CUDA artifact is **Lane G v3: 1.05 [contest-CUDA]**.

- Evidence: `experiments/results/lane_g_v3_landed/contest_auth_eval.json`
- Archive: `experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip`
- Archive bytes: 694,074
- Components: SegNet 0.00400846, PoseNet 0.00345458, rate 0.01848622
- Recomputed score: 1.0488665

Lane A remains the clean fallback floor at **1.15 [contest-CUDA]**.

- Evidence: `experiments/results/lane_a_landed/contest_auth_eval.json`
- Archive bytes: 694,045
- Components: SegNet 0.00460724, PoseNet 0.00496876, rate 0.01848544
- Recomputed score: 1.1457672

Do not resurrect stale floors from AGENTS.md or older reports without explicit evidence. The 1.33 / 0.90 / 2.01 / 2.26 numbers are historical context, not the current promoted floor.

## Verification State

- Local toolchain doctor is OK for git/python/curl/ffmpeg/git-lfs/node/npm/uv/omx.
- Canonical E2E smoke passed for Lane G v3:
  `.venv/bin/python experiments/canonical_local_auth_eval_smoke.py --lane g_v3_corrected_kl_weight --quiet`
- Focused tests passed:
  `.venv/bin/python -m pytest src/tac/tests/test_canonical_local_e2e_smoke.py src/tac/tests/test_check_64_e2e_smoke_proof.py src/tac/tests/test_contest_auth_eval.py -q`
  -> 34 passed.
- Check 64: 0 violations.
- Check 65: 0 violations after 2026-04-29 lane-class proof backfill. Several entries are explicitly `canonical-local-smoke` plumbing proofs, not score claims.

## Snapshot Caveat

The upstream snapshot state is not cleanly reinitialized yet.

- `comma-lab status` still reports snapshot commit `ec82c291ffeae5212e9a38253791d58995518a80`, last verified 2026-04-03.
- Live `workspace/upstream/comma_video_compression_challenge` is at `cd64c68b740ffbe90c0132ca560a9cefc9d78ac5` and has installed submission-state changes.
- Root `upstream/` is at `11ad728f563d8970929e8947a1cf6124ee6303e4` with many local modifications.

Do not claim upstream snapshot freshness until the snapshot is deliberately rebootstraped and the resulting file/hash changes are reviewed.

## Active Failures / Triage

- `modal_auth_eval_8e331354a6b5.json`: inflate failed because `optimized_poses.pt` had shape `(600, 6)` but the archive/renderer expected pose_dim 1. Treat this as a pose-dimension export mismatch, not a score.
- Recovered SZ phase2 produced a 3.3KB renderer-only archive but canonical local smoke failed at `masks_present`: no mask file in archive, so the current inflator would fall back to non-compliant SegNet extraction. This lane is blocked until the no-mask paradigm has an explicit compliant inflator path.
- 2026-04-29 Modal first-wave failures:
  - MAE-V: missing `pydantic` in remote runtime before training.
  - Omega Hessian: CUDA device-side assert during Hessian profiling.
  - Uniward: missing `submissions/baseline_dilated_h64_0_90/renderer.bin` in remote path.

## Current Posture

Protect Lane G v3 and Lane A artifacts. Spend next cycles on deployment correctness and cheap rate reductions only when they have a local smoke proof and a clear contest-CUDA auth path.
