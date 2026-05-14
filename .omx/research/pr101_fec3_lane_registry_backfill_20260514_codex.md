# PR101 FEC3 lane-registry backfill - 2026-05-14

## Scope

The PR101/FEC3 compact selector near-miss had dispatch-claim custody but no
matching lane-registry row. This was an integration/governance defect, not a
score invalidation.

## Evidence

Dispatch claims:

- `.omx/state/active_lane_dispatch_claims.md`
- lane id:
  `lane_pr101_frame_exploit_selector_fec3_compact_exact_k8_cpu_overlay_20260514`
- terminal result:
  `completed_contest_cpu_modal_auth_eval_recovered`
- archive sha256:
  `8866ebb655e96ccf0ffcd84feae08c131734cba8c402bfb8c661a29f289ce409`
- score:
  `0.19209788683213053`
- axis:
  `[contest-CPU]`

Authoritative result:

- `experiments/results/modal_auth_eval_cpu/archive_8866ebb655e9/contest_auth_eval.json`
- `score_axis=contest_cpu`
- `exact_cuda_eval_complete=false`
- `score_claim=false`
- `promotion_eligible=false`

## Backfill

Local live-state row added to ignored `.omx/state/lane_registry.json`:

- id:
  `lane_pr101_frame_exploit_selector_fec3_compact_exact_k8_cpu_overlay_20260514`
- level: `2`
- `contest_cpu.status=true`
- `contest_cuda.status=false`
- `real_archive_empirical.status=true`
- `strict_preflight.status=true` only for axis/promotion guard, not for
  submission readiness

Because `.omx/state/lane_registry.json` is ignored by policy, this committed
ledger preserves the backfill signal for future analysis and recovery.

## Non-claim

This backfill does not promote the artifact. It explicitly preserves:

- no CPU-to-CUDA conversion
- no score claim
- no promotion eligibility
- no rank/kill eligibility from the CPU result

## Next concrete action

The remaining CPU-axis gap to `<0.192` is about `9.8e-5`. With unchanged
components, that requires saving at least `148` charged archive bytes from the
`178517` byte FEC3 archive, or an equivalent component improvement. The next
byte-closed work item is a smaller selector grammar, selector entropy coding,
or independent PR101 source-payload saving, followed by separate exact Linux
CPU and exact CUDA evals.
