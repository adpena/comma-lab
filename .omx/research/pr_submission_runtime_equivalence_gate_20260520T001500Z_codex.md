# PR Submission Runtime Equivalence Gate - Codex Round

UTC: 2026-05-20T00:15:00Z
Lane: `lane_pr_submission_public_surface_convergence_20260519`
Verdict: `NOT_SAFE_TO_PR`

## Artifact landed

Runtime-equivalence proof:

`.omx/research/pr101_fec6_runtime_equivalence_proof_20260520T001500Z_codex.json`

The proof ties:

- archive SHA-256 `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`
- existing CUDA auth-eval runtime tree `ca7f4f323d57a346739532c74c95af4f0a82fedf400a9c6f9e201eb5124f1e61`
- current split submission runtime tree `fd4b36b0114789ffd25c6169f529bca70b20da8f70e4ee1336dad9fd64971a09`
- full local split-runtime inflate output `0.raw` SHA-256 `d1afc583b01ff4a7aaa844d4f03ece3ed381d56763a06cb2c5e011526e5f868c`

Fresh local split-runtime inflate result:

```text
saved 1200 frames
d1afc583b01ff4a7aaa844d4f03ece3ed381d56763a06cb2c5e011526e5f868c  0.raw
3662409600 bytes
```

Prior auth-eval/monolithic baseline evidence:

- `.omx/research/codex_codec_py_refactor_verification_20260519T200658Z.md`
- `.omx/research/codex_codec_py_refactor_verification_20260519T214807Z.md`

## Gate change

`scripts/pre_submission_compliance_check.py` now accepts an opt-in:

```bash
--runtime-equivalence-proof-json <path>
```

This is fail-closed. A runtime mismatch is accepted only when the proof:

- has schema `pre_submission_runtime_equivalence_proof_v1`;
- matches exact archive SHA and bytes;
- matches one auth-eval runtime candidate;
- matches the current submission runtime candidate;
- records `full_inflate_output_byte_identity`;
- records equal baseline/candidate output hashes and `diff_bytes=0`.

Regression coverage:

```bash
.venv/bin/python -m pytest src/tac/tests/test_pre_submission_compliance_check.py -q
```

Result:

```text
42 passed in 3.08s
```

The new tests prove a strict runtime-equivalence proof can clear the runtime mismatch, malformed proof archive bytes fail closed instead of throwing, and the existing runtime-mismatch test still rejects a naked mismatch.

## Strict gate rerun

Command used:

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
  --public-scan-path .omx/research/pr_submission_check_in_package_20260519/PR_BODY_UPSTREAM_TEMPLATE_CONFORMANT.md
```

Result: exit code `1`, `passed=false`.

Newly green:

- `auth_eval_runtime_tree_expected_match`
- `runtime_equivalence_proof_exists`
- `runtime_equivalence_proof_json_object`
- `runtime_equivalence_proof_schema`
- `runtime_equivalence_proof_archive_matches`
- `runtime_equivalence_proof_auth_runtime_matches`
- `runtime_equivalence_proof_submission_runtime_matches`
- `runtime_equivalence_proof_full_output_identity`
- `submission_runtime_tree_matches_auth_eval`

Remaining blockers:

- `contest_cpu_auth_eval_score_at_or_below_submission_threshold`: strict gate threshold is `0.192`; CPU score is `0.1920513168811056`.
- `public_source_pinned_revision_present`: packet still has `<PINNED_COMMIT>` and needs a public commit/tag containing the current runtime.
- `contest_final_expected_lane_id_supplied`: final gate needs explicit lane id.
- `contest_final_expected_job_id_supplied`: final gate needs explicit job id and matching terminal claim evidence.

## PR status

Do not open a PR yet. Runtime custody is now proof-backed, but final packet submission remains blocked on threshold policy, source publication, and terminal lane/job evidence.
