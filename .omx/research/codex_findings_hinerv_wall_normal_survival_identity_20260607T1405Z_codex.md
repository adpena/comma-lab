# HiNeRV Wall-Normal Survival Identity Replay

UTC: 2026-06-07T14:05Z
Author: Codex
Axis: macOS-MLX/local receiver evidence, non-promotional

## Artifact Paths

- Same-action parseback + inflate survival row:
  `/Volumes/VertigoDataTier/pact/experiments/results/hinerv_target_region_action_survival_identity_replay_20260607Tcodex_v2_inflate/hi_nerv_target_region_action_parseback_survival.json`
- Sidecar/backend comparison report:
  `/Volumes/VertigoDataTier/pact/experiments/results/hinerv_target_region_action_comparison_20260607Tcodex_identity_inflate_closed_v1/hinerv_target_region_action_sidecar_backend_comparison.json`
- ActionEffect rows:
  `/Volumes/VertigoDataTier/pact/experiments/results/hinerv_target_region_action_comparison_20260607Tcodex_identity_inflate_closed_v1/hinerv_target_region_action_sidecar_backend_action_effects.jsonl`
- Advisory long-run gate verdict:
  `/Volumes/VertigoDataTier/pact/experiments/results/hinerv_target_region_action_comparison_20260607Tcodex_identity_inflate_closed_v1/long_run_gate_advisory.json`

## Result

The selected v43 target-region sidecar action now has a same-action receiver
survival replay:

- `action_id`: `08fdb60f677cf324fa2a8769ce84a4d189442e5bc545ffdbe05e53f40ad30510`
- `support_sha256`: `744bc2096422e1cbf2aa6113898746998f4d327502921e65a45d39ef92d0f81d`
- fakequant survived: true
- parseback survived: true
- inflate survived: true
- blockers: []

The comparison report now moves the first failing surface from
`target_region_action_survival_action_id_missing` to `support_identity_mismatch`.
That is the correct current blocker for this older v43 artifact: the survived
sidecar support is archive-executable action support, while the direct teacher in
the old runner report only exposes bool-mask support.

## Non-Promotion Boundary

No score claim or promotion claim is made. The advisory long-run gate remains
fail-closed with:

- `archive_parseback_selection_contract_missing`
- `real_video_birth_receipt_missing`
- `source_qualified_metrics_missing`

## Next Concrete Work

Rerun the short HiNeRV wall-normal smoke on current main so the direct teacher
emits `archive_executable_support_sha256` from the same target-region action
coordinates. If direct teacher support then matches the survived sidecar support,
the next blocker should become backend realization or sidecar grammar economics
instead of support identity.
