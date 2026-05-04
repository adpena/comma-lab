# PR79/S2 Mask Body Reduction Worker - 2026-05-03

Scope: local-only PR79/S2 mask-stream and full-archive byte screening. No
remote dispatch was performed.

Tooling:

- `experiments/profile_pr79_mask_body_reduction_candidates.py`
- `src/tac/tests/test_profile_pr79_mask_body_reduction_candidates.py`
- Artifact root:
  `experiments/results/pr79_mask_body_reduction_20260503_worker/`

Source archive:

- Path:
  `experiments/results/top_submission_reverse_engineering_20260503_pr79/archive.zip`
- Bytes: `277388`
- SHA-256:
  `01dc02badf851d99108fd92c570271f36f74cc5424c6d2a8f1b499cb4d1c3446`
- Payload format: `public_pr75_qzs3_qp1_segactions_fixed_slices`
- Raw segment bytes:
  `masks.mkv=219472`, `renderer.bin=55756`,
  `seg_tile_actions.bin=1162`, `optimized_poses.qp1=898`

Finite neighborhood tested:

- One lossless PR79 mask rebrotli/P3 control.
- Sixteen existing protected mask reencode streams whose source mask SHA
  matched PR79 decoded `masks.mkv`.
- Candidate wrappers were strict single-member `p` archives:
  source-style OBU masks used P3; MKV reencode masks used RPK1 with PR79
  renderer bytes, decoded PR79 action records, and PR79 QP1 pose bytes.

Recommendation:

- Decision: `recommend_exact_cuda_eval_after_lane_claim`
- Candidate:
  `protected_c067_micro_mask_reencode_threshold_ladder_20260503_save05k_crf53`
- Archive:
  `experiments/results/pr79_mask_body_reduction_20260503_worker/protected_c067_micro_mask_reencode_threshold_ladder_20260503_save05k_crf53/archive.zip`
- Bytes: `260866`
- SHA-256:
  `5e49a972ab5d77d8a8eeedc9dd36a309bf8533ad2d2e2b9b96ad30448691d338`
- Delta vs PR79: `-16522` bytes
- Manifest:
  `experiments/results/pr79_mask_body_reduction_20260503_worker/protected_c067_micro_mask_reencode_threshold_ladder_20260503_save05k_crf53/manifest.json`
- Matrix:
  `experiments/results/pr79_mask_body_reduction_20260503_worker/candidate_matrix.json`
- Recommendation JSON:
  `experiments/results/pr79_mask_body_reduction_20260503_worker/recommendation.json`

Best-candidate member deltas:

- `masks.mkv`: `196984` charged bytes, delta `-22488` vs PR79 mask slice.
- `renderer.bin`: `59288` raw RPK1 bytes, decoded PR79 renderer preserved.
- `seg_tile_actions.bin`: `2688` raw decoded action bytes, decoded PR79
  action semantics preserved.
- `optimized_poses.qp1`: `1140` raw QP1 bytes, PR79 QP1 payload preserved.

Validation:

- `validate_archive_seg_tile_actions_payloads(...) == []`
- Runtime payload parser:
  `submissions/robust_current/unpack_renderer_payload.py`
- Parser OK: `true`
- Candidate mask SHA matched expected:
  `49eedf22801b610603275d21dc5b1ae3668cd4decc9349d083ad8cef933837ef`
- Non-mask runtime member checks:
  `renderer_decoded_preserved=true`, `actions_decoded_preserved=true`,
  `pose_decoded_preserved=true`
- Plausibility gate:
  overall argmax agreement `0.997173487345`, protected argmax agreement
  `0.992559116516`; both clear local thresholds `0.997` and `0.99`.

Dispatch rule:

- No remote dispatch was performed.
- Before any exact CUDA eval, claim a non-conflicting lane with
  `tools/claim_lane_dispatch.py claim ...`.
- Exact score truth remains
  `archive.zip -> inflate.sh -> upstream/evaluate.py` via
  `experiments/contest_auth_eval.py --device cuda` on identical archive bytes.
