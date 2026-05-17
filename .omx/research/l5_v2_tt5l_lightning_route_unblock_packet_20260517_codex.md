# L5 v2 TT5L Lightning route unblock packet

**Generated:** 2026-05-17T11:22:48Z
**Commit:** `c5755a4a3e0ecdf39be4ebbf8a09d6ffd2aa8af5`

This packet is generated from live artifact hashes. It turns the current TT5L Lightning blocker into an executable operator checklist. It is not a dispatch, score claim, or promotion artifact.

## Verdict

The TT5L method path is not blocked here. The route is blocked on provider configuration:

- Lightning credits or quota not checked
- LIGHTNING_SDK_USER or LIGHTNING_ORG missing
- LIGHTNING_SSH_TARGET missing
- LIGHTNING_TEAMSPACE missing
- active dispatch claims not created for non-dry-run cells
- Lightning machine inventory not checked
- source manifest not staged to remote Lightning workspace
- remote CUDA runtime not probed

## Current Evidence

- Provider readiness refresh: `.omx/research/l5_v2_provider_readiness_refresh_20260517_codex.json`
  - exists: `True`
  - SHA-256: `9ae5433a2c769c1f39a9c9996c86cabe970225b5070c2ad3e620ee45a7459d65`
- 10-cell sideinfo execution preflight: `.omx/research/l5_v2_tt5l_sideinfo_lightning_execution_preflight_20260517_codex.json`
  - exists: `True`
  - SHA-256: `4b6031bd226ab4feb1d866866a8eceacdb2d37a6e492cad24fac401079010e36`
- 10-cell sideinfo execution bundle: `.omx/research/l5_v2_tt5l_sideinfo_lightning_execution_bundle_20260517_codex.json`
  - exists: `True`
  - SHA-256: `d80166e34b8b60e015fb3f9d50edede5592c81133e622d6dd8dea7138ab4ee52`
- 10-cell dry-run verification: `.omx/research/l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run_verification_20260517_codex.json`
  - exists: `True`
  - SHA-256: `9316f0a8de767ab862efa40c491bd1e211759fd304b8c501a7a9ddf2e069ca4c`
  - all dry-runs passed: `True`
  - cells passed: `10`/`10`
- 10-cell paired-axis plan: `.omx/research/l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan_20260517_codex.json`
  - exists: `True`
  - SHA-256: `f2dd83cd43b5ef8904f0e4a693782cdec4b3c1500509652747485c3b151bd9ab`
  - source commit: `d369f6b6e2f86b602749ffd44f55c01f118efda3`
  - source-relevant paths match current HEAD: `True`
- Sideinfo harvest cells: `.omx/research/l5_v2_tt5l_sideinfo_effect_curve_harvest_cells_20260517_codex.json`
  - exists: `True`
  - SHA-256: `3483e1e4f71147eeefaf001f219f9f312b1457d9c56823db4b02149c5b74276f`
  - harvested exact-eval artifacts: `0`
  - missing exact-eval artifacts: `10`
- Sideinfo effect curve: `.omx/research/l5_v2_tt5l_sideinfo_effect_curve_20260516_codex.json`
  - exists: `True`
  - SHA-256: `6b3284be6d5c24cf69d604b7c58324fc4669b0dc3f8190f633c767d2b2aad1b4`
  - predicate passed: `False`
- Architecture lock packet: `.omx/research/l5_v2_architecture_lock_packet_20260516_codex.json`
  - exists: `True`
  - SHA-256: `5f1c19d356d3d882cdec997bf2c14366b8f9fde6660ac405b8fc27333062249b`
  - architecture lock allowed: `False`

The refreshed bundle embeds the T4/g4dn exact-eval runtime pins required by `scripts/launch_lightning_batch_job.py`: `INFLATE_TORCH_SPEC=torch==2.5.1+cu124`, `INFLATE_TORCHVISION_SPEC=torchvision==0.20.1+cu124`, `UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124`, `UV_INDEX_STRATEGY=unsafe-best-match`.

Dry-run scope: `local parser and queue-spec custody only; no provider job, no score`.

Bundle axes preserved: `[contest-CPU]`, `[contest-CUDA]`

## Command Order

1. configure_lightning_route

```bash
export LIGHTNING_TEAMSPACE='<lightning-teamspace>'; export LIGHTNING_SSH_TARGET='<lightning-ssh-target>'; export LIGHTNING_SDK_USER='<lightning-user>'  # or export LIGHTNING_ORG='<lightning-org>'
```

2. required_doctor

```bash
.venv/bin/python scripts/launch_lightning_batch_job.py doctor --json-out .omx/research/l5_v2_tt5l_lightning_required_doctor_20260517_codex.json --run-id l5_v2_tt5l_lightning_required_doctor_20260517 --strict --ssh-target "$LIGHTNING_SSH_TARGET" --require-ssh --remote-supply-chain --require-remote-supply-chain --repo-dir /teamspace/studios/this_studio/pact --python-bin .venv/bin/python --teamspace "$LIGHTNING_TEAMSPACE" <--user-or---org> '<lightning-user-or-org>' --machine-inventory --require-machine-inventory --machine T4 --gpu-only
```

3. restage_source_manifest_per_cell
   - command source: `.omx/research/l5_v2_tt5l_sideinfo_lightning_execution_bundle_20260517_codex.json:cells[*].stage_source_manifest_command_template`
4. claim_each_lane_before_non_dry_run
   - command source: `.omx/research/l5_v2_tt5l_sideinfo_lightning_execution_bundle_20260517_codex.json:cells[*].claim_command`
5. submit_non_dry_run_after_placeholders_removed
   - command source: `.omx/research/l5_v2_tt5l_sideinfo_lightning_execution_bundle_20260517_codex.json:cells[*].non_dry_run_submit_command_template`
6. harvest_and_close_claims
   - command source: `.omx/research/l5_v2_tt5l_sideinfo_lightning_execution_bundle_20260517_codex.json:cells[*].harvest_probe_command_template plus terminal_success_claim_template or terminal_failure_claim_template`
7. refresh_effect_curve_and_architecture_packet

```bash
.venv/bin/python tools/build_l5_v2_tt5l_sideinfo_effect_curve_cells_from_lightning_plan.py --lightning-plan-json .omx/research/l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan_20260517_codex.json --output-json .omx/research/l5_v2_tt5l_sideinfo_effect_curve_harvest_cells_20260517_codex.json --repo-root . && .venv/bin/python tools/build_l5_v2_sideinfo_effect_curve.py --cell-json .omx/research/l5_v2_tt5l_sideinfo_effect_curve_harvest_cells_20260517_codex.json --output-json .omx/research/l5_v2_tt5l_sideinfo_effect_curve_20260516_codex.json --repo-root . && .venv/bin/python tools/build_l5_v2_architecture_lock_packet.py --repo-root .
```

## Authority

- `planning_only=true`
- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `ready_for_provider_dispatch=false`
- `dispatch_attempted=false`
- `provider_spend_attempted=false`

No CPU/CUDA axis may be promoted from this packet. Promotion requires harvested contest-auth-eval artifacts with custody, adjudication, and terminal lane claims.

Blockers: `[]`
