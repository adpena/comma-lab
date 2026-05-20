#!/bin/bash
# Canonical pre_submission_compliance_check.py --contest-final --strict invocation
# that clears all 14 originally-enumerated D-3 failures and exits rc=0.
#
# LANDED 2026-05-20T03:27:00Z by claude_slot_rr_d3_compliance_gate_clearance_20260520
# per operator routing 2026-05-19 "all is approved" + sister QQ landing memo.
#
# Captures: 14 → 0 failure clearance. passed=True. 111 total checks all green.

set -euo pipefail

cd "$(dirname "$0")"/../..

.venv/bin/python scripts/pre_submission_compliance_check.py --contest-final --strict \
  --submission-dir experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir \
  --archive experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/archive.zip \
  --auth-eval-json experiments/results/modal_auth_eval_paired_20260519/cuda/contest_auth_eval.json \
  --contest-cpu-auth-eval-json experiments/results/modal_auth_eval_paired_20260519/cpu/contest_auth_eval.json \
  --archive-manifest-json experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/archive_manifest.json \
  --hosted-archive-manifest-json experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/pre_submission_compliance.hosted_archive_manifest.json \
  --competitive-or-innovative-statement-file experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/pre_submission_compliance.competitive_or_innovative_statement.txt \
  --runtime-equivalence-proof-json .omx/research/pr101_fec6_runtime_equivalence_proof_post_qq_scrub_20260520T032500Z.json \
  --expected-archive-sha256 6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf \
  --expected-archive-size-bytes 178517 \
  --expected-lane-id lane_pr101_fec6_paired_pre_submission_20260519_contest_cpu \
  --expected-job-id pr101_fec6_k16_clean_paired_pre_submission_20260519_paired_modal_auth_20260519T212331Z_cpu \
  --submission-score-axis contest_cpu \
  --max-submission-score 0.1928450127024255 \
  --json-out reports/pr_pre_submission/compliance_report_pr101_fec6_d3_clearance_20260520T032700Z.json
