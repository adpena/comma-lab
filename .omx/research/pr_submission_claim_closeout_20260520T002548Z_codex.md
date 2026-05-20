# PR Submission Claim Closeout - Codex Round

UTC: 2026-05-20T00:25:48Z
Lane: `lane_pr_submission_public_surface_convergence_20260519`
Verdict: `NOT_SAFE_TO_PR`

## Artifact Landed

Closed the stale active paired Modal auth-eval claims for the FEC6 packet using
`tools/claim_lane_dispatch.py claim`.

CUDA terminal row:

- lane: `lane_pr101_fec6_paired_pre_submission_20260519_contest_cuda`
- job: `pr101_fec6_k16_clean_paired_pre_submission_20260519_paired_modal_auth_20260519T212331Z_cuda`
- status: `completed_contest_cuda_modal_auth_eval_recovered`
- score: `0.22621002169349796` `[contest-CUDA]`
- archive SHA-256: `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`
- auth runtime tree: `ca7f4f323d57a346739532c74c95af4f0a82fedf400a9c6f9e201eb5124f1e61`

CPU terminal row:

- lane: `lane_pr101_fec6_paired_pre_submission_20260519_contest_cpu`
- job: `pr101_fec6_k16_clean_paired_pre_submission_20260519_paired_modal_auth_20260519T212331Z_cpu`
- status: `completed_contest_cpu_modal_auth_eval_recovered`
- score: `0.1920513168811056` `[contest-CPU]`
- archive SHA-256: `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`
- auth runtime tree: `e0d7fd8dc86545a273f163b7c9b35d186705f22623429d36d7ce75ed9e6b9130`

Both terminal rows also bind the proof-backed split submission runtime:

`fd4b36b0114789ffd25c6169f529bca70b20da8f70e4ee1336dad9fd64971a09`

Runtime-equivalence proof:

`.omx/research/pr101_fec6_runtime_equivalence_proof_20260520T001500Z_codex.json`

## Strict Gate Rerun

Command added the selected CPU lane/job:

```bash
.venv/bin/python scripts/pre_submission_compliance_check.py \
  --contest-final \
  --strict \
  --submission-dir experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir \
  --archive experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/archive.zip \
  --auth-eval-json experiments/results/modal_auth_eval_paired_20260519/cuda/contest_auth_eval.json \
  --contest-cpu-auth-eval-json experiments/results/modal_auth_eval_paired_20260519/cpu/contest_auth_eval.json \
  --archive-manifest-json experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/archive_manifest.json \
  --submission-score-axis contest_cpu \
  --expected-archive-sha256 6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf \
  --expected-archive-size-bytes 178517 \
  --expected-runtime-tree-sha256 fd4b36b0114789ffd25c6169f529bca70b20da8f70e4ee1336dad9fd64971a09 \
  --runtime-equivalence-proof-json .omx/research/pr101_fec6_runtime_equivalence_proof_20260520T001500Z_codex.json \
  --expect-single-member x \
  --competitive-or-innovative-statement-file .omx/research/pr_submission_check_in_package_20260519/PR_BODY_UPSTREAM_TEMPLATE_CONFORMANT.md \
  --public-scan-path .omx/research/pr_submission_check_in_package_20260519/PR_BODY_UPSTREAM_TEMPLATE_CONFORMANT.md \
  --expected-lane-id lane_pr101_fec6_paired_pre_submission_20260519_contest_cpu \
  --expected-job-id pr101_fec6_k16_clean_paired_pre_submission_20260519_paired_modal_auth_20260519T212331Z_cpu
```

Result: exit code `1`, `passed=false`.

Newly green:

- `contest_final_expected_lane_id_supplied`
- `contest_final_expected_job_id_supplied`
- `dispatch_claim_terminal_row`
- `dispatch_claim_successful_exact_eval_terminal_row`
- `dispatch_claim_terminal_archive_sha_bound`
- `dispatch_claim_terminal_runtime_tree_sha_bound`
- `dispatch_claim_prior_active_row`

Remaining blockers:

- `contest_cpu_auth_eval_score_at_or_below_submission_threshold`: strict gate threshold is `0.192`; CPU score is `0.1920513168811056`.
- `public_source_pinned_revision_present`: packet still needs a public commit/tag containing the current runtime.

## Packet Text Update

Updated the PR body draft, submission README, and archive manifest blocker text
to remove terminal lane/job evidence from the open blocker list. The PR remains
blocked on CPU-threshold policy and public source pinning. No PR was opened.
