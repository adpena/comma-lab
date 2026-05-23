# Codex Findings - IAS1 Runtime Inflate Parity Empirical Closure

UTC: 2026-05-23T19:19:04Z
Lane: `lane_inverse_scorer_inflate_parity_20260523`
Authority axis: local full-runtime parity / false authority

## Finding

The current IAS1 inverse-scorer candidate can pass the strict actual-runtime
inflate parity gate when run through an IAS1-aware DQS1 runtime. This clears the
local full-frame parity blocker for this candidate chain and leaves only
`exact_auth_eval_required_before_score_claim`.

## Evidence

- Rebuilt IAS1-aware DQS1 runtime:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/selective_runtime_candidate_append_tail_top32_ias1aware_20260523T1930Z/submission_dir`
- Chain manifest:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/inverse_scorer_cell_chain_ias1_runtime_parity_20260523T1930Z/inverse_scorer_cell_candidate_chain_manifest.json`
- Parity probe:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/inverse_scorer_cell_chain_ias1_runtime_parity_20260523T1930Z/inflate_parity_probe.json`

## Measured Closure

- `receiver_contract_satisfied=true`
- `inflate_parity_satisfied=true`
- `full_frame_inflate_output_parity_claim=true`
- Source inflate return code: `0`
- Candidate inflate return code: `0`
- File list: `0.mkv`
- Source output tree: `file_count=1`, `total_bytes=3662409600`,
  `tree_sha256=7a5e8d1ac8faad3b25d5236e0c09cbed027537e5f5d7426b8ea093e8f7a047ef`
- Candidate output tree: `file_count=1`, `total_bytes=3662409600`,
  `tree_sha256=7a5e8d1ac8faad3b25d5236e0c09cbed027537e5f5d7426b8ea093e8f7a047ef`
- `output_contract_paths_match=true`
- `output_contract_nonempty=true`
- `output_bytes_identical=true`
- `work_dir_retained=false`; raw inflate outputs were intentionally treated as
  rebuildable bulk after compact tree custody was recorded.

## Candidate

- Candidate archive SHA-256:
  `2d0850789483e17c7ee68ae8bfe1e33489d1981416f71266cf8a66b19a87e549`
- Candidate archive bytes: `181232`
- Candidate member SHA-256:
  `61535249fabd1d21835c56e171e1e437e0d329393b8c4ffd69359d15ba2b70a0`

## Lane Update

Marked `lane_inverse_scorer_inflate_parity_20260523.real_archive_empirical`
using the runtime-parity chain manifest. The lane is now L2.

## Authority

No score is claimed. This is not contest CPU/CUDA auth eval. The candidate,
runtime parity probe, and chain remain non-promotional and
`ready_for_exact_eval_dispatch=false` until exact auth gates and dispatch
claims are satisfied.
