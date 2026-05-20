# PR Submission Public Surface Convergence - Codex Round

UTC: 2026-05-20T00:08:27Z
Lane: `lane_pr_submission_public_surface_convergence_20260519`
Verdict: `NOT_SAFE_TO_PR`

## Files changed

- `.omx/research/pr_submission_check_in_package_20260519/PR_BODY_UPSTREAM_TEMPLATE_CONFORMANT.md`
- `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/README.md`
- `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/archive_manifest.json`
- `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/report.txt` (local ignored packet file)

## Corrections landed

- Corrected public lineage: PR #95 @AaronLeslie138, PR #98 @EthanYangTW, PR #100 @BradyMeighan, PR #101 @SajayR, PR #102 @EthanYangTW, PR #103 @rem2.
- Removed the false `0.226210 [contest-CUDA T4]` label form and now use the canonical `[contest-CUDA]` token with Modal Tesla T4 as host metadata.
- Removed raw provider call-id exposure from public PR body text.
- Removed the double-counted `-0.000622` net claim; the CPU delta is now stated as `-0.000794` total, already including the +259-byte rate cost.
- Corrected the FEC6/Brotli boundary: selector payload is appended outside the PR101 Brotli-coded source payload and is not additionally Brotli-compressed.
- Removed the hallucinated latent-residual codec claim from the manifest.
- Added the gate-consumable top-level `members` table for ZIP member `x`.
- Added archive SHA-256 and raw size footer to `report.txt`.
- Fixed the README full-score reproduction path so it stages runtime files separately from `archive.zip`; the ZIP extracts only member `x`.

## Verification

Text/JSON checks:

- `archive_manifest.json` parses with `.venv/bin/python -m json.tool`.
- Bad-string scan is clean for `[contest-CUDA T4]`, raw Modal call-id pattern, `0.000622`, `innovation_4`, `0.bin`, old Brotli-wrapper wording, and `Vast.ai`.
- `.venv/bin/python tools/lane_maturity.py validate` passes: `1033 lane(s) validated cleanly`.

Strict gate rerun:

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
  --expected-runtime-tree-sha256 fb4ba11f998ec8c0137dffaa7f567db416e5c8790d155aa9365a0aa0b3580dbb \
  --expect-single-member x \
  --competitive-or-innovative-statement-file .omx/research/pr_submission_check_in_package_20260519/PR_BODY_UPSTREAM_TEMPLATE_CONFORMANT.md \
  --public-scan-path .omx/research/pr_submission_check_in_package_20260519/PR_BODY_UPSTREAM_TEMPLATE_CONFORMANT.md
```

Result: exit code `1`, `passed=false`.

Newly green versus the prior audit:

- `archive_manifest_members_present`
- `archive_manifest_member_count_matches`
- all five member field checks for member `x`
- `report_mentions_archive_sha256`
- `report_mentions_archive_size_bytes`
- `public_source_reproduce_command_or_sha_binding_present`
- `public_evidence_contest_cuda_label_present`
- `public_scan_has_no_private_surface`

Remaining blockers:

- `submission_runtime_tree_matches_auth_eval`: local split runtime (`fd4b36b0114789ffd25c6169f529bca70b20da8f70e4ee1336dad9fd64971a09`; portable no-custody `ee53871c8766718b9ab289c4a12a59501795ce7fbf71bfbb42728b5669884e8c`) does not match existing CUDA auth-eval runtime (`ca7f4f323d57a346739532c74c95af4f0a82fedf400a9c6f9e201eb5124f1e61`; portable no-custody `b94638718572c12300886a6ce786810bfa5d9305d216753feefbcaa463011c49`).
- `auth_eval_runtime_tree_expected_match`: the rerun still used stale expected SHA `fb4ba11f998ec8c0137dffaa7f567db416e5c8790d155aa9365a0aa0b3580dbb`; final validation must use the selected runtime SHA after the runtime/auth-eval decision is resolved.
- `contest_cpu_auth_eval_score_at_or_below_submission_threshold`: current hard threshold is `0.192`; CPU score is `0.1920513168811056`.
- `public_source_pinned_revision_present`: packet still has `<PINNED_COMMIT>` by design; needs a real public source-sync commit or release tag.
- `contest_final_expected_lane_id_supplied` and `contest_final_expected_job_id_supplied`: final gate still needs lane/job arguments and corresponding terminal claim evidence.

## Next concrete path

Do not open a PR yet. The remaining decision is runtime custody: either rerun paired auth eval on the split `src/codec.py` + `src/codec_sidecar.py` runtime, revert public runtime to the monolithic runtime that produced the existing auth eval, or land an explicit same-output equivalence proof that the compliance gate consumes. After that, pin source commit, attach terminal lane/job evidence, and re-run strict gate with the correct expected runtime SHA.

## Round Extension - Axis Label Scan Surface

UTC: 2026-05-20T00:56:45Z

`inspect_public_evidence_axis_labels()` now scans the same public packet surface
set as the hosted-archive and public-source checks: submission public files plus
any `--public-scan-path` inputs. This closes the case where the PR body is the
only public text carrying `[contest-CUDA]` / `[contest-CPU]` labels.

Regression added:

```text
test_contest_final_accepts_public_axis_labels_in_public_scan_path
```

Focused checks:

```text
.venv/bin/python -m pytest src/tac/tests/test_pre_submission_compliance_check.py -q -k "axis_labels"
2 passed in 0.78s

.venv/bin/python -m pytest src/tac/tests/test_pre_submission_compliance_check.py -q -k "public_source or hosted"
6 passed in 1.05s
```

Full checker file:

```text
.venv/bin/python -m pytest src/tac/tests/test_pre_submission_compliance_check.py -q
50 passed in 3.56s
```

Real strict gate still exits `1` / `passed=false`, with the same four
publication blockers:

```text
hosted_archive_manifest_supplied
hosted_archive_public_text_has_no_placeholder
public_source_pinned_revision_present
public_source_pinned_revision_publicly_visible
```

Axis-label section now confirms both labels are present and that the PR body
draft is included in the scan:

```text
labels={"[contest-CUDA]": true, "[contest-CPU]": true}
sources=[
  "experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/README.md",
  "experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/report.txt",
  "experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/archive_manifest.json",
  ".omx/research/pr_submission_check_in_package_20260519/PR_BODY_UPSTREAM_TEMPLATE_CONFORMANT.md"
]
```

## Round Extension - Generic Template Placeholder Hygiene

UTC: 2026-05-20T02:48:40Z

Added a contest-final public placeholder hygiene section:

```text
public_text_has_no_unresolved_template_placeholders
```

It scans the merged public packet surface (`submission_dir` public files plus
`--public-scan-path`) and fails on unresolved uppercase template placeholders
such as `<HOSTED_URL_PLACEHOLDER>`, `<PINNED_COMMIT>`, and `<REPLACE_ME>`.
Legitimate upstream inflate signature placeholders remain allowed:

```text
<archive_dir>
<output_dir>
<file_list>
```

Regression coverage:

```text
test_contest_final_rejects_generic_public_template_placeholder
test_contest_final_allows_upstream_inflate_signature_placeholders
```

Checks:

```text
.venv/bin/python -m pytest src/tac/tests/test_pre_submission_compliance_check.py -q
53 passed in 3.83s

.venv/bin/python -m pytest src/tac/tests/test_pre_submission_compliance_check.py -q -k "public_source or source_pin or hosted or axis_labels"
10 passed in 1.39s
```

Real strict gate still exits `1` / `passed=false`. The new generic placeholder
section reports:

```text
placeholders=['<HOSTED_URL_PLACEHOLDER>', '<PINNED_COMMIT>']
locations=[
  '.omx/research/pr_submission_check_in_package_20260519/PR_BODY_UPSTREAM_TEMPLATE_CONFORMANT.md:3:<HOSTED_URL_PLACEHOLDER>',
  '.omx/research/pr_submission_check_in_package_20260519/PR_BODY_UPSTREAM_TEMPLATE_CONFORMANT.md:68:<PINNED_COMMIT>',
  '.omx/research/pr_submission_check_in_package_20260519/PR_BODY_UPSTREAM_TEMPLATE_CONFORMANT.md:71:<PINNED_COMMIT>',
  'experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/README.md:71:<PINNED_COMMIT>',
  'experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/README.md:95:<PINNED_COMMIT>',
  'experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/README.md:99:<HOSTED_URL_PLACEHOLDER>'
]
```

The same rerun also surfaced fresh runtime-custody drift in the dirty worktree:

```text
runtime_equivalence_proof_submission_runtime_matches
submission_runtime_tree_matches_auth_eval
```

The runtime-equivalence proof still points at the earlier split-runtime hashes
`fd4b36...` / `ee5387...`, but the current submission runtime now hashes to
`cd76c8...` / `0c7019...`. Treat the existing proof as stale until refreshed
or until the runtime changes are intentionally incorporated into a new
same-output proof.
