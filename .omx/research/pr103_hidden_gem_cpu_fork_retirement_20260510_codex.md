# PR103 Hidden-Gem CPU Fork Retirement

Date: 2026-05-10

This records the status of the draft fork PR created from the PR103 hidden-gem
CPU candidate path so it cannot be mistaken for a live submission candidate.

## Fork PR Manifest

- manifest path:
  `experiments/results/fork_pr_pr103_hidden_gem_cpu_20260510T1618Z_20260510T161744Z/fork_pr_manifest.json`
- fork PR: `https://github.com/adpena/comma_video_compression_challenge/pull/26`
- branch:
  `add-submission-pr103_hidden_gem_cpu_20260510T1618Z-20260510T161744Z`
- intended archive SHA-256:
  `8274e88c0ab1d26a06470a0730d17fe004556afa564460cf1c05624ff6060278`

## Status

- `score_claim=false`
- `contest_cpu_status=failed_infra_no_score`
- `contest_cuda_status=completed_negative_on_successor_packet`
- `promotion_eligible=false`
- `submission_ready=false`

The CPU GHA attempt failed before producing a score. The later byte-closed
PR103 histogram packet exact CUDA eval completed and was classified negative in
`.omx/research/pr103_histogram_8b_exact_cuda_negative_20260510_codex.md`.

## Guardrail

Do not cite the draft PR or its manifest as a scored archive. It is a custody
trace for an abandoned CPU-path attempt only. Any future PR103 arithmetic-code
candidate needs a fresh candidate archive, runtime tree SHA, strict compliance
report, dispatch claim, and paired axis-specific eval artifacts.
