# L5 v2 TT5L side-info dispatch-plan visibility - 2026-05-17

Scope: preserve the already materialized TT5L side-info effect-curve dispatch
queue inside the architecture lock/no-lock packet, so the next byte-closed
measurement surface is visible without reading a separate plan by hand.

Evidence axis: planning and custody visibility only. This is not a score claim,
not promotion evidence, and not provider dispatch authorization.

## Change

`src/tac/optimization/l5_staircase_v2.py` now validates and surfaces
`.omx/research/l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan_20260517_codex.json`
inside `tt5l_campaign_readiness.sideinfo_effect_curve_dispatch_plan_status`.

The rendered architecture packet now includes `## TT5L Sideinfo Dispatch Plan`
with:

- dispatch-plan artifact validity;
- 5/5 structurally ready side-info work units;
- required variants: `zero`, `random_lsb`, `shuffled`, `trained`, `ablated`;
- archive SHA-256, byte count, and pair-group id per variant;
- explicit `score_claim=false`, `promotion_eligible=false`,
  `ready_for_provider_dispatch=false`, and `dispatch_attempted=false`.

## Why

The live side-info effect curve remains invalid because it has only one
observed cell (`contest_cuda/trained`) and that cell has all-zero side-info
liveness. The already generated dispatch plan describes the missing 5 paired
variant jobs, but the lock packet did not show it. That made the packet look
blocked without preserving the concrete byte-closed queue that resolves the
missing cells.

## Current status

- `sideinfo_effect_curve_artifact_valid=false`
- `sideinfo_effect_curve_dispatch_plan_status.artifact_valid=true`
- `sideinfo_effect_curve_dispatch_plan_status.ready_work_unit_count=5`
- `sideinfo_effect_curve_dispatch_plan_status.work_unit_count=5`
- `ready_for_provider_dispatch=false`

Provider execution still requires resolving the Modal blocker or making the
Lightning provider path executable-current with identity, workspace, machine,
source manifest, remote CUDA probe, lane claims, and harvest flow.
