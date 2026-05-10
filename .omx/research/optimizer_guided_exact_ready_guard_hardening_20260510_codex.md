# Optimizer-guided exact-ready guard hardening (2026-05-10)

## Scope

This tranche hardened the optimizer-candidate exact-readiness path while keeping
score authority fail-closed. The immediate score-lowering object is
`bias_refine_cmaes_style_stdlib_0127_pr101_proxy_runtime_packet`, a PR101/A1
same-archive runtime-bias candidate selected from the optimizer-guided local
queue.

## Candidate custody

- Source queue:
  `.omx/research/optimizer_guided_candidate_queues_20260510_codex/bias_refine_0127_runtime_packet_planning_queue.json`
- Exact-ready queue:
  `.omx/research/optimizer_guided_candidate_queues_20260510_codex/bias_refine_0127_exact_ready_queue.json`
- Exact-ready report:
  `.omx/research/optimizer_guided_candidate_queues_20260510_codex/bias_refine_0127_exact_ready_report.json`
- Candidate id:
  `bias_refine_cmaes_style_stdlib_0127_pr101_proxy_runtime_packet`
- Lane id:
  `pr101_kaggle_proxy_runtime_packet_exact_eval`
- Archive SHA-256:
  `b83bf3488625dbd73adeddff91712994197ab53098e578e91327a0c6e49efb3e`
- Archive bytes: `178258`
- Runtime tree SHA-256:
  `40def9339343b9fcb50bce211fb47a22f19332edd7e3ff7090733b9d15247ff2`
- Runtime proof SHA-256:
  `7b404db7b77b4afc51fe370dbb377be61e5a4601cf28a2b0f12cfbb749e6e4ba`

The exact-readiness gate returned `ready_for_exact_eval_dispatch=true` with
`score_claim=false`, `exact_cuda_auth_eval=false`, and
`dispatch_claim_required_before_gpu_or_remote_eval=true`.

## Adversarial classification

This is a byte-closed dispatchable packet, not a score claim. It should not
preempt the active T1 Ballé Modal run. Prior exact CUDA evidence on the same
PR101 archive family produced negatives at `0.22650343150032118` and
`0.22688160652506983`, so this candidate is useful as a custody/probe artifact
but not the highest-EV score-lowering dispatch unless the queue is explicitly
selected as a CUDA-drift calibration point.

## Guard fixes

- `tools/promote_optimizer_candidate_for_exact_eval.py` now writes the
  structured `--report-output` even when the candidate fails the readiness
  gate, as long as the report is outside the submission runtime tree.
- Exact-ready queue discovery now scans multiple roots. The default CLI and
  operator briefing scan both `experiments/results` and `.omx/research`, so
  research-ledger exact-ready queues are not invisible to terminal-evidence
  audits.

## Verification

```bash
.venv/bin/python -m pytest \
  tests/test_audit_exact_ready_queues_cli.py \
  src/tac/tests/test_optimizer_exact_ready_audit.py \
  tests/test_promote_optimizer_candidate_for_exact_eval_cli.py -q
# 29 passed

.venv/bin/python tools/audit_exact_ready_queues.py \
  --format json \
  --suppression-manifest .omx/research/exact_ready_queue_retraction_manifest_20260510_codex.json \
  --warn-only
# passed=true; queue_count=6; stale_ready_row_count=0; raw_stale_ready_row_count=5; suppressed_ready_row_count=5
```

## Next action

Keep this packet available for exact-eval calibration, but do not dispatch it
while T1 is active unless the operator explicitly chooses a PR101 CUDA-drift
probe. The highest-EV score-lowering action remains harvesting and classifying
the active T1 end-to-end Ballé run, then moving to train-time substrate work if
it returns a valid packet or a concrete export/runtime blocker.
