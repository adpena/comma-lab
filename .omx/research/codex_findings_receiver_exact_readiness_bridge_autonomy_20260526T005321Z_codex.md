# Receiver Exact-Readiness Bridge Autonomy - 2026-05-26T00:53:21Z

## Verdict

The receiver/materializer handoff was still losing signal at the exact-readiness
boundary. Family-agnostic runtime proofs were allowed to prove a receiver
consumed changed charged bytes, but the exact-readiness gate treated truthy
`charged_bits_changed` / `score_affecting_payload_changed` inside those proofs
as false-authority violations. That made receiver-positive materializers look
like authority bugs instead of the real remaining blocker: byte-closed
submission/runtime closure before exact eval.

## Landed

- Split family-agnostic runtime proof change evidence from score/dispatch
  authority. Truthy `charged_bits_changed` and
  `score_affecting_payload_changed` are now accepted only as change evidence for
  family materializer receiver proofs.
- Kept PR101 runtime proof authority checks strict and kept score, promotion,
  rank/kill, dispatch, GPU launch, and ready-for-exact-eval claims fail-closed.
- Added `tools/run_materializer_exact_readiness_bridge.py` as a queue-callable
  bridge from harvested materializer source queues to per-candidate exact
  readiness reports.
- Added `--force-recompute` so queue steps can refresh exact-readiness bridge
  reports after compiler/gate semantics change instead of reusing stale reports.
- Tightened the non-forced idempotent skip so an existing bridge report is reused
  only when the source queue, exact-readiness output dir, and candidate filter
  match. Different filters recompute or fail closed instead of quietly reusing
  stale signal.
- Wired receiver repair queues to run the exact-readiness bridge after work
  order emission, with false-authority postconditions.
- Kept tensor-factorize receiver readiness on the compiled receiver-runtime proof
  path. The scheduler harvest regression now proves the positive case through
  `build_tensor_factorize_receiver_runtime(...)` plus
  `build_tensor_factorize_runtime_consumption_proof(...)`, not through the
  parser-only factorization proof.

## New Artifacts

- `.omx/research/frontier_operation_portfolio_20260526T005231Z/feedback_refresh_report.json`
- `.omx/research/frontier_operation_portfolio_20260526T005231Z/operation_portfolio.json`
- `.omx/research/frontier_operation_portfolio_20260526T005231Z/receiver_repair_backlog.json`
- `.omx/research/frontier_operation_portfolio_20260526T005231Z/receiver_repair_queue.json`
- `.omx/research/frontier_operation_portfolio_20260526T005231Z/dqs1_followup_queue.json`

The refreshed portfolio consumed 6 materializer feedback payloads, preserved 32
operation rows, kept 5 queue-executable operations, and carried 14 follow-up
signals. Receiver repair backlog moved from stale false-authority work toward
real receiver/runtime closure: 170 rows, 115 queue-actionable rows, and no
`runtime_consumption_proof_false_authority_violation:charged_bits_changed`
entries.

The receiver repair queue selected four source-diverse rows:

- `chain_dfl1_merge_header_elide_minimal_envelope`
- `materializer_renderer_payload_dfl1_v1`
- `materializer_packet_member_merge_v1`
- `materializer_packet_member_zip_header_elide_v1`

One selected repair now has a queue-owned exact-readiness bridge step. The
others remain work-order first because their current blocker class is parity or
runtime-proof construction, not bridge recomputation.

## Verification

- `ruff` passed on the touched exact-readiness, feedback, bridge tool, and test
  files.
- Focused bridge/feedback/readiness regression slice: 130 passed.
- New `.omx/research/frontier_operation_portfolio_20260526T005231Z/receiver_repair_queue.json`
  validates cleanly.
- New `.omx/research/frontier_operation_portfolio_20260526T005231Z/dqs1_followup_queue.json`
  validates cleanly.
- Bounded receiver repair queue smoke executed all 5 steps successfully with
  zero failures and zero orphaned steps.

## Remaining Work

The system is less leaf/manual now, but not finished. The next high-EV slice is
to make the top receiver-repair rows execute the actual runtime/manifest closure
instead of only producing work orders and bridge reports:

- build byte-closed submission packets for receiver-positive materializers;
- emit `inflate.sh`, `report.txt`, `archive_manifest.json`, runtime tree SHA,
  and runtime content tree SHA automatically;
- re-run the exact-readiness bridge from the receiver repair queue;
- only then hand an exact-ready queue to the materializer exact-eval consumer.

Freed bytes remain planning signal for targeted SegNet/PoseNet correction
budget only. They are not score, promotion, rank/kill, or dispatch authority.
