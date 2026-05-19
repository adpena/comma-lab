# Codex Findings: PR101 Pose-Axis Score-Response Matrix

Task:
`codex_routing_directive_v2_synthesis_followup_null_space_plus_hash_seed_plus_cross_stack_20260518::ITEM_7`.

Context:
the prior PR101 OP-7 raw-delta builder produced a same-size component-moving
candidate archive and a runtime-consumption proof, but it still left the
score-response surface as prose. This landing adds the missing matrix builder:
paired baseline/candidate auth-eval commands plus score-response probe commands,
with score and dispatch authority still fail-closed.

Implemented:

- `src/tac/master_gradient_pr101_score_response_matrix.py`
  - Builds a `pr101_pose_axis_score_response_matrix_v1` artifact.
  - Reuses canonical `tac.auth_eval_roundtrip_matrix` target rows.
  - Reuses `tools/probe_substrate_score_response.py` command grammar for each
    matched baseline/candidate target.
  - Requires a component-moving PR101 raw-delta candidate plus green
    `tac_runtime_consumption_proof_v1`.
  - Emits explicit false authority flags:
    `score_claim`, `score_claim_valid`, `promotion_eligible`,
    `rank_or_kill_eligible`, `ready_for_provider_dispatch`,
    `ready_for_exact_eval_dispatch`, `dispatch_attempted`,
    `raw_archive_byte_coordinates_allowed`, and
    `candidate_specs_are_dispatchable`.
- `tools/build_pr101_pose_axis_score_response_matrix.py`
  - Thin operator CLI that loads the PR101 source runtime manifest, writes JSON
    and optional Markdown, and performs no dispatch.
- `src/tac/tests/test_master_gradient_pr101_score_response_matrix.py`
  - Covers contest-axis target pairing, raw-equivalent/non-component-moving
    blocking, authority-leak blocking, Markdown labeling, and CLI output.

Local PR101 matrix artifact:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/build_pr101_pose_axis_score_response_matrix.py \
  --source-archive experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip \
  --source-submission-dir experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/source/submissions/hnerv_ft_microcodec \
  --operator-manifest experiments/results/pr101_pose_axis_operator_candidate_raw_delta_20260519T084439Z_codex/operator_manifest.json \
  --candidate-manifest experiments/results/pr101_pose_axis_operator_candidate_raw_delta_20260519T084439Z_codex/candidate_manifest.json \
  --runtime-proof experiments/results/pr101_pose_axis_operator_candidate_raw_delta_20260519T084439Z_codex/runtime_consumption_proof.json \
  --label pr101-op7-rank1-raw-byte-delta-same-length \
  --lane-id pr101_pose_axis_op7_rank1_raw_delta_score_response \
  --output-root experiments/results/pr101_pose_axis_score_response_matrix_20260519T092500Z_codex \
  --json-out experiments/results/pr101_pose_axis_score_response_matrix_20260519T092500Z_codex/score_response_matrix.json \
  --md-out experiments/results/pr101_pose_axis_score_response_matrix_20260519T092500Z_codex/score_response_matrix.md
```

Artifact custody:

- matrix JSON:
  `experiments/results/pr101_pose_axis_score_response_matrix_20260519T092500Z_codex/score_response_matrix.json`
- matrix JSON SHA-256:
  `80db2ddb08fb6fae2011fdadadf2f9b1b08c6b99efde5f9b856971af70853026`
- matrix Markdown:
  `experiments/results/pr101_pose_axis_score_response_matrix_20260519T092500Z_codex/score_response_matrix.md`
- matrix Markdown SHA-256:
  `eb5c28e9b754e55db9814123d0091603e64fccd47774c91db0560620860502a8`
- candidate archive:
  `experiments/results/pr101_pose_axis_operator_candidate_raw_delta_20260519T084439Z_codex/archive.zip`
- candidate archive bytes:
  `178258`
- candidate archive SHA-256:
  `30826b37093ee3af9512a1b46bd0b569fecbc4ccf75b8ff2dd746de113a5144a`

Matrix readiness:

- `authority_blockers`: `[]`
- `ready_for_score_response_probe`: `false`
- `ready_for_score_response_probe_after_exact_eval`: `true`
- `ready_for_score_response_probe_after_exact_eval_and_lane_claim`: `false`
- score-response blockers:
  - `modal_contest_cpu_linux_x86_auto_baseline_exact_eval_json_missing`
  - `modal_contest_cpu_linux_x86_auto_candidate_exact_eval_json_missing`
  - `modal_contest_cuda_t4_auto_baseline_exact_eval_json_missing`
  - `modal_contest_cuda_t4_auto_candidate_exact_eval_json_missing`
- dispatch blockers:
  - `modal_contest_cpu_linux_x86_auto_active_lane_claim_missing`
  - `modal_contest_cuda_t4_auto_active_lane_claim_missing`
  - `modal_contest_cpu_linux_x86_auto_exact_eval_missing`
  - `modal_contest_cuda_t4_auto_exact_eval_missing`
  - `full_inflate_output_manifest_missing_until_auth_eval`
  - `paired_contest_cuda_cpu_exact_eval_missing`
  - `paired_contest_cuda_and_cpu_result_review_missing`

Authority interpretation:

This is not a score claim and not an exact-eval dispatch authorization. It is
the command/control artifact that makes the next OP-7 step unambiguous: claim
paired contest CUDA/CPU lanes, run baseline and candidate auth evals on matching
runtime, then run the generated score-response probes. Local macOS CPU remains
advisory only.

Verification:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  src/tac/tests/test_master_gradient_pr101_score_response_matrix.py
```

Result: `5 passed`.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m ruff check \
  src/tac/master_gradient_pr101_score_response_matrix.py \
  tools/build_pr101_pose_axis_score_response_matrix.py \
  src/tac/tests/test_master_gradient_pr101_score_response_matrix.py
```

Result: passed.
