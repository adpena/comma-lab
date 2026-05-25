# Codex Findings: Frontier Many-Receiver Portfolio Execution

- utc: 2026-05-25T23:16:50Z
- lane_id: codex_frontier_many_receiver_portfolio_execution
- status: local-queue-executed; exact-readiness blocked as designed
- authority: local materializer signal only; no score, promotion, rank/kill, dispatch, or exact-eval authority

## Why

The previous many-materializer portfolio still stopped one layer too early: it ranked registry families, but concrete frontier/archive context was not bound broadly enough, so rows remained blocked and manual. The next useful tranche was to widen the queue-owned final-rate bootstrap from two packet-member leaves into a receiver-aware, multi-operation local actuator.

## Implemented

- `frontier_rate_attack_bootstrap` now defaults to `packet_member_zip_header_elide_v1`, `packet_member_recompress_v1`, `packet_member_merge_v1`, and `renderer_payload_dfl1_v1`.
- Multi-member archives now expand `packet_member_recompress_v1` across every member instead of requiring a hand-picked member.
- `packet_member_merge_v1` is compiled per eligible archive with a derived all-member merge contract.
- `renderer_payload_dfl1_v1` is compiled per archive only when the fixed source-runtime-native `renderer.bin`, `masks.mkv`, `optimized_poses.pt` deflated-member grammar is present.
- Unsupported archives emit typed omissions instead of freezing the portfolio.
- Sweep-scoped exact-readiness follow-up harvests no longer scan the entire work queue, which fixed the parallel-execution race where sibling sweep manifests were missing while still running.
- Pure byte-shaving signal-surface materializer campaigns now plan directly from the surface instead of passing through inverse-action water-bucket selection when no scorer/acquisition/feedback inputs are present. This preserves singleton registry materializer rows as queueable combinations.

## Proof Artifact

Executed:

```bash
.venv/bin/python tools/build_frontier_final_rate_attack_queue.py \
  --no-current-frontier \
  --archive robust_current_correct=submissions/robust_current/archive_correct.zip \
  --output-dir .omx/research/frontier_final_rate_attack_many_receiver_portfolio_20260525T231411Z \
  --results-root experiments/results/frontier_final_rate_attack \
  --queue-id frontier_final_rate_attack_many_receiver_portfolio_20260525T231411Z \
  --derive-section-manifests \
  --include-exact-readiness-followup \
  --execute \
  --max-steps 32 \
  --max-parallel 0 \
  --local-cpu-concurrency 2
```

Artifacts:

- `.omx/research/frontier_final_rate_attack_many_receiver_portfolio_20260525T231411Z/frontier_rate_attack_bootstrap.json`
- `.omx/research/frontier_final_rate_attack_many_receiver_portfolio_20260525T231411Z/materializer_work_queue.json`
- `.omx/research/frontier_final_rate_attack_many_receiver_portfolio_20260525T231411Z/experiment_queue.json`
- `.omx/research/frontier_final_rate_attack_many_receiver_portfolio_20260525T231411Z/execution_report.json`
- `.omx/research/frontier_final_rate_attack_many_receiver_portfolio_20260525T231411Z/derived_packet_member_merge_contract.json`

## Results

- Bootstrap emitted 6 executable materializer experiments and 18 queue steps.
- Queue execution completed with `status_counts={"succeeded": 18}` and zero failed/queued steps.
- Positive local byte deltas:
  - `renderer_payload_dfl1_v1`: +380 saved bytes; receiver/exact-readiness blocked pending full-frame inflate parity.
  - `packet_member_merge_v1`: +258 saved bytes; receiver/exact-readiness blocked pending byte-closed runtime adapter consumption proof.
  - `packet_member_zip_header_elide_v1`: +156 saved bytes; local proof only.
- `packet_member_recompress_v1` across `renderer.bin`, `masks.mkv`, and `optimized_poses.pt` was not rate-positive on this archive (`-556` bytes each), and that negative is preserved as materializer feedback.
- Every harvest/exact-readiness handoff emitted a source queue and bridge report with `source_queue_dispatch_ready_count=0`.

## Guardrails

- All artifacts keep score/promotion/rank/dispatch authority false.
- Local parser/reconstruction proof remains separated from runtime adapter readiness.
- Exact-readiness is an explicit follow-up edge and remains blocked until per-candidate proofs satisfy the contest-runtime contract.
- Section/tensor targets remain typed omissions when concrete section/tensor manifests are missing.
- Singleton materializer-registry portfolio units now survive into PacketIR and campaign queue planning instead of requiring an arbitrary second unit to become actionable.

## Next Gate

Wire the harvested many-operation observations back into the inverse action-functional acquisition rule so the next queue wave automatically prioritizes DFL1, merge-runtime-adapter proof, and non-recompress alternatives instead of re-running known-negative recompress rows on the same archive.
