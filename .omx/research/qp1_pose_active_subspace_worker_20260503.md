# QP1 Pose Active-Subspace Worker - 2026-05-03

Scope: Worker B owns QP1 pose active-subspace candidates only. No remote GPU
jobs were dispatched and no active lane dispatch claims were edited.

Implementation target:

- Builder: `experiments/build_qp1_pose_active_subspace_candidates.py`
- Tests: `src/tac/tests/test_build_qp1_pose_active_subspace_candidates.py`
- Artifacts: `experiments/results/qp1_pose_active_subspace_worker_20260503/`

Contract:

- Source default is C-089 `c067_pr75_qp1_top40_p6` exact T4 archive.
- The builder preserves `masks.mkv`, `renderer.bin`, PR75 action payload, and
  optional action dictionary slices byte-for-byte.
- The only archive stream changed by candidates is the Brotli-compressed QP1
  pose stream.
- Manifests record source/candidate archive bytes and SHA-256, QP1 raw bytes
  and SHA-256, uint16 word custody, decoded float32 semantic SHA-256, changed
  pair indices, and local parser/roundtrip gates.
- Candidate rankings are empirical policy screens only. CUDA exact auth eval is
  required before any score claim.

Verification plan:

- `py_compile` on the new builder.
- Focused pytest on the new tests.
- `git diff --check`.

Local artifact run:

Command:

```bash
.venv/bin/python experiments/build_qp1_pose_active_subspace_candidates.py \
  --output-dir experiments/results/qp1_pose_active_subspace_worker_20260503 \
  --force
```

Output summary:

- Summary JSON:
  `experiments/results/qp1_pose_active_subspace_worker_20260503/candidate_summary.json`
- Candidate count: `5`
- All candidates preserve the source C-089 P6 mask, renderer, and action
  slices byte-for-byte and rewrite only `optimized_poses.qp1`.
- All candidates have local parser/roundtrip gates passed.
- No score is claimed; ranking is an empirical trace/byte policy screen only.

Top local screens:

| rank | candidate | bytes | archive delta vs C-089 | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `ref_active_combined_top32_s0125` | `276398` | `+56` | `944c2ba5af9c2d9c5897e4913f9c476d12d12884d964cdd8531716cf4ec92dc1` |
| 2 | `ref_active_pose_top16_s025` | `276376` | `+34` | `089195580123c25e4dcedc77639d79858a02d2a6052b7ce43876731f139dcd55` |
| 3 | `neighbor_combined_top20_a020` | `276383` | `+41` | `669206c2695d0ecd5029cccbbf7466ac5374cf1059062bfc6ab1ec9644b5cc8e` |

Exact eval command templates are present in each manifest only because local
parser/roundtrip gates passed. They are templates only; this worker did not
dispatch remote GPU work.

Verification results:

- `.venv/bin/python -m py_compile experiments/build_qp1_pose_active_subspace_candidates.py`
- `.venv/bin/python -m pytest src/tac/tests/test_build_qp1_pose_active_subspace_candidates.py -q`
  passed with `2 passed`.
