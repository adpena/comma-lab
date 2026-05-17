# L5 v2 TT5L Lightning route unblock packet

**Generated:** 2026-05-17T11:08:18Z
**Commit:** `d369f6b6e2f86b602749ffd44f55c01f118efda3`

This packet turns the current TT5L Lightning blocker into an executable
operator checklist. It is not a dispatch, score claim, or promotion artifact.

## Verdict

The TT5L method path is not blocked here. The route is blocked on provider
configuration:

- `LIGHTNING_TEAMSPACE` missing
- `LIGHTNING_SDK_USER` or `LIGHTNING_ORG` missing
- `LIGHTNING_SSH_TARGET` missing
- Lightning machine inventory not checked
- source manifest not staged to the remote workspace
- remote CUDA runtime not probed
- active dispatch claims not created for non-dry-run cells

## Current Evidence

- Provider readiness refresh:
  `.omx/research/l5_v2_provider_readiness_refresh_20260517_codex.json`
  - SHA-256:
    `9ae5433a2c769c1f39a9c9996c86cabe970225b5070c2ad3e620ee45a7459d65`
- 10-cell sideinfo execution bundle:
  `.omx/research/l5_v2_tt5l_sideinfo_lightning_execution_bundle_20260517_codex.json`
  - SHA-256:
    `d80166e34b8b60e015fb3f9d50edede5592c81133e622d6dd8dea7138ab4ee52`
- 10-cell dry-run verification:
  `.omx/research/l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run_verification_20260517_codex.json`
  - SHA-256:
    `9316f0a8de767ab862efa40c491bd1e211759fd304b8c501a7a9ddf2e069ca4c`
  - all dry-runs passed: `true`
  - cells passed: `10/10`
  - every cell's launcher stdout record matches the latest persisted
    `--state-path` record across `dry_run`, `queue`, and `spec`
  - every cell's local `--inflate-sh` runtime exists and is executable
- 10-cell paired-axis plan:
  `.omx/research/l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan_20260517_codex.json`
  - SHA-256:
    `f2dd83cd43b5ef8904f0e4a693782cdec4b3c1500509652747485c3b151bd9ab`
  - source commit:
    `d369f6b6e2f86b602749ffd44f55c01f118efda3`
  - source-relevant paths match current HEAD: `true`
- Sideinfo harvest cells:
  `.omx/research/l5_v2_tt5l_sideinfo_effect_curve_harvest_cells_20260517_codex.json`
  - SHA-256:
    `3483e1e4f71147eeefaf001f219f9f312b1457d9c56823db4b02149c5b74276f`
  - harvested exact-eval artifacts: `0`
  - missing exact-eval artifacts: `10`
- Sideinfo effect curve:
  `.omx/research/l5_v2_tt5l_sideinfo_effect_curve_20260516_codex.json`
  - SHA-256:
    `6b3284be6d5c24cf69d604b7c58324fc4669b0dc3f8190f633c767d2b2aad1b4`
  - predicate passed: `false`

The refreshed bundle embeds the T4/g4dn exact-eval runtime pins required by
`scripts/launch_lightning_batch_job.py`: `INFLATE_TORCH_SPEC=torch==2.5.1+cu124`,
`INFLATE_TORCHVISION_SPEC=torchvision==0.20.1+cu124`,
`UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124`, and
`UV_INDEX_STRATEGY=unsafe-best-match`.

The older `.omx/research/l5_v2_tt5l_lightning_alt_provider_plan_20260517_codex.json`
is preserved as historical single-cell provider-blocker evidence. The 10-cell
bundle is now the sideinfo effect-curve command authority.

## Command Order

1. Configure Lightning route:

```bash
export LIGHTNING_TEAMSPACE='<lightning-teamspace>'
export LIGHTNING_SSH_TARGET='<lightning-ssh-target>'
export LIGHTNING_SDK_USER='<lightning-user>'  # or export LIGHTNING_ORG='<lightning-org>'
export LIGHTNING_STUDIO='<lightning-studio>'
```

2. Run required doctor:

```bash
.venv/bin/python scripts/launch_lightning_batch_job.py doctor \
  --json-out .omx/research/l5_v2_tt5l_lightning_required_doctor_20260517_codex.json \
  --run-id l5_v2_tt5l_lightning_required_doctor_20260517 \
  --strict \
  --ssh-target "$LIGHTNING_SSH_TARGET" \
  --require-ssh \
  --remote-supply-chain \
  --require-remote-supply-chain \
  --repo-dir /teamspace/studios/this_studio/pact \
  --python-bin .venv/bin/python \
  --teamspace "$LIGHTNING_TEAMSPACE" \
  <--user-or---org> '<lightning-user-or-org>' \
  --machine-inventory \
  --require-machine-inventory \
  --machine T4 \
  --gpu-only
```

3. For each cell in the 10-cell bundle, run:

- `cells[*].stage_source_manifest_command_template`
- `cells[*].claim_command`
- `cells[*].non_dry_run_submit_command_template` after replacing all placeholders
- `cells[*].harvest_probe_command_template`
- `terminal_success_claim_template` or `terminal_failure_claim_template`

4. Refresh derived TT5L surfaces:

```bash
.venv/bin/python tools/build_l5_v2_tt5l_sideinfo_effect_curve_cells_from_lightning_plan.py \
  --lightning-plan-json .omx/research/l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan_20260517_codex.json \
  --output-json .omx/research/l5_v2_tt5l_sideinfo_effect_curve_harvest_cells_20260517_codex.json \
  --repo-root .

.venv/bin/python tools/build_l5_v2_sideinfo_effect_curve.py \
  --cell-json .omx/research/l5_v2_tt5l_sideinfo_effect_curve_harvest_cells_20260517_codex.json \
  --output-json .omx/research/l5_v2_tt5l_sideinfo_effect_curve_20260516_codex.json \
  --repo-root .

.venv/bin/python tools/build_l5_v2_architecture_lock_packet.py --repo-root .
```

## Authority

- `planning_only=true`
- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `ready_for_provider_dispatch=false`
- `dispatch_attempted=false`
- `provider_spend_attempted=false`

No CPU/CUDA axis may be promoted from this packet. Promotion requires harvested
contest-auth-eval artifacts with custody, adjudication, and terminal lane
claims.
