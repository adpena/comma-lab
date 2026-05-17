# L5 v2 TT5L Lightning route unblock packet

**Generated:** 2026-05-17T15:40:02Z
**Commit:** `9b926ab6e099585d64a76726db91c8af6be0f181`

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
  - SHA-256: `b558a46ab1908587bca658f49b3dda0a8888c214179788cb6b3e8f040596e716`
- 10-cell sideinfo execution bundle: `.omx/research/l5_v2_tt5l_sideinfo_lightning_execution_bundle_20260517_codex.json`
  - exists: `True`
  - SHA-256: `3c79806149add6780146538c8aed7615153a8c62ca028a660e65cd9b06155e79`
- 10-cell dry-run verification: `.omx/research/l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run_verification_20260517_codex.json`
  - exists: `True`
  - SHA-256: `e1b8c26fce04996a8e26b68c8a0d2bc63541a894ba6515f05466acbd07257ea8`
  - all dry-runs passed: `True`
  - cells passed: `10`/`10`
- 10-cell paired-axis plan: `.omx/research/l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan_20260517_codex.json`
  - exists: `True`
  - SHA-256: `acc247e499a9f3a855d97623099bbb95d59807eb5371a0d31f6ade3f8b145414`
  - source commit: `9b926ab6e099585d64a76726db91c8af6be0f181`
  - source-relevant paths match current HEAD: `True`
- Sideinfo harvest cells: `.omx/research/l5_v2_tt5l_sideinfo_effect_curve_harvest_cells_20260517_codex.json`
  - exists: `True`
  - SHA-256: `44024ea8019dcea3186b77d93c25fc4c970c1392357066dd9ed025c6d633f394`
  - harvested exact-eval artifacts: `0`
  - missing exact-eval artifacts: `10`
- Sideinfo effect curve: `.omx/research/l5_v2_tt5l_sideinfo_effect_curve_20260516_codex.json`
  - exists: `True`
  - SHA-256: `d0e38af5388a625381c1d9eba4949ca85a64ea3ed0c3601f481d2f4fb5f6216e`
  - predicate passed: `False`

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
