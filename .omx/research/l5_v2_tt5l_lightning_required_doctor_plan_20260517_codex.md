# L5 v2 TT5L Lightning required doctor plan

Generated: 2026-05-17T14:00:37Z
Commit: `4b6ff1a6110fd78a0a79e6817a2d64369780f322`

This generated plan converts the TT5L route blockers into the exact Lightning doctor commands and JSON pass predicates. It does not run Lightning, submit jobs, claim score movement, or create dispatch claims.

## Authority

- `planning_only=true`
- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `ready_for_provider_dispatch=false`
- `dispatch_attempted=false`
- `provider_spend_attempted=false`

## Source

- Route packet: `.omx/research/l5_v2_tt5l_lightning_route_unblock_packet_20260517_codex.json`
- Route packet SHA-256: `ebcf5fee8c51b3e03364e3a1fefa8cd01df910c5e39ec7448b8b18bc29d9c090`
- ready_for_operator_doctor: `True`
- blockers: `[]`

## Doctor Commands

### user

- required env: `['LIGHTNING_TEAMSPACE', 'LIGHTNING_SSH_TARGET', 'LIGHTNING_SDK_USER']`
- forbidden env: `['LIGHTNING_ORG']`

```bash
.venv/bin/python scripts/launch_lightning_batch_job.py doctor --json-out .omx/research/l5_v2_tt5l_lightning_required_doctor_20260517_codex.json --run-id l5_v2_tt5l_lightning_required_doctor_20260517 --strict --ssh-target "$LIGHTNING_SSH_TARGET" --require-ssh --remote-supply-chain --require-remote-supply-chain --repo-dir /teamspace/studios/this_studio/pact --python-bin .venv/bin/python --teamspace "$LIGHTNING_TEAMSPACE" --user "$LIGHTNING_SDK_USER" --machine-inventory --require-machine-inventory --machine T4 --gpu-only
```

### org

- required env: `['LIGHTNING_TEAMSPACE', 'LIGHTNING_SSH_TARGET', 'LIGHTNING_ORG']`
- forbidden env: `['LIGHTNING_SDK_USER']`

```bash
.venv/bin/python scripts/launch_lightning_batch_job.py doctor --json-out .omx/research/l5_v2_tt5l_lightning_required_doctor_20260517_codex.json --run-id l5_v2_tt5l_lightning_required_doctor_20260517 --strict --ssh-target "$LIGHTNING_SSH_TARGET" --require-ssh --remote-supply-chain --require-remote-supply-chain --repo-dir /teamspace/studios/this_studio/pact --python-bin .venv/bin/python --teamspace "$LIGHTNING_TEAMSPACE" --org "$LIGHTNING_ORG" --machine-inventory --require-machine-inventory --machine T4 --gpu-only
```

## Required Checks

- doctor output: `.omx/research/l5_v2_tt5l_lightning_required_doctor_20260517_codex.json`
- expected `status`: `OK`
- required checks: `local_supply_chain`, `ssh_auth`, `remote_supply_chain`, `machine_inventory`

## Remaining Route Work

- Run per-cell stage_source_manifest_command_template from the execution bundle.
- Run per-cell claim_command before every non-dry-run submit.
- Replace all placeholders in non_dry_run_submit_command_template.
- Submit only after doctor JSON status is OK and all required checks pass.
- Harvest contest_auth_eval artifacts and terminal claims before any score claim.
