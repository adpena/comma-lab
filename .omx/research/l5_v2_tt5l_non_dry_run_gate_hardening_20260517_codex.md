# L5 v2 TT5L non-dry-run gate hardening - 2026-05-17

Lane: `l5_v2_tt5l_sideinfo_effect_curve`

Axis policy: paired `[contest-CPU]` and `[contest-CUDA]`

Scope: L5 v2 staircase / TT5L Lightning exact-eval control plane.

## Why this landed

The TT5L Lightning route already had a dry-run execution bundle and a required
doctor plan, but no single fail-closed artifact consumed all non-dry-run
evidence together. That left a retread-prone gap: a stale bundle, missing doctor
JSON, unstaged source manifest, unclaimed lane, or unreplaced Lightning
placeholder could be diagnosed in one surface while another surface still looked
operator-actionable.

This patch adds the missing conjunction gate:

`ready_for_non_dry_run_submit = doctor_ok AND all_source_manifests_staged AND all_stage_receipts_remote_sha_verified AND all_active_claims_present AND all_templates_concrete AND paired_axes_complete AND current_head_matched`

The gate is spend-readiness only. It never creates a score claim, promotion
claim, rank/kill claim, or dispatch claim.

## New artifacts

- `src/tac/optimization/l5_v2_tt5l_lightning_non_dry_run_gate.py`
- `tools/check_l5_v2_tt5l_lightning_non_dry_run_gate.py`
- `src/tac/tests/test_l5_v2_tt5l_lightning_non_dry_run_gate.py`
- `.omx/research/l5_v2_tt5l_lightning_non_dry_run_gate_20260517_codex.json`
- `.omx/research/l5_v2_tt5l_lightning_non_dry_run_gate_20260517_codex.md`

## Live gate result

Command:

```bash
.venv/bin/python tools/check_l5_v2_tt5l_lightning_non_dry_run_gate.py --repo-root .
```

Result:

- `ready_for_non_dry_run_submit=false`
- `ready_for_provider_dispatch=false`
- `ready_cell_count=0/10`
- `blocker_count=163` after regenerating the TT5L paired-axis plan,
  execution preflight, execution bundle, dry-run verification, route-unblock
  packet, doctor plan, and non-dry-run gate on commit
  `1f1f0140789f9f12a7e9a140ed41802e99fcdaa1`
- `score_claim=false`
- `promotion_eligible=false`
- `dispatch_attempted=false`

Primary live blockers:

- Required doctor output is absent:
  `.omx/research/l5_v2_tt5l_lightning_required_doctor_20260517_codex.json`
- Per-cell source manifests are absent under
  `experiments/results/lightning_batch/<job_name>/source_manifest.json`
- Per-cell remote-verified staging receipts are absent under
  `experiments/results/lightning_batch/<job_name>/source_manifest_receipt.json`
- Non-dry-run submit templates still contain Lightning placeholders:
  `<lightning-studio>`, `<lightning-teamspace>`, `<--user-or---org>`,
  `<lightning-user-or-org>`, `<lightning-ssh-target>`
- No active Lightning dispatch claim rows exist for the 10 TT5L side-info cells.
- Source-manifest git-head blockers remain only because the manifests do not
  exist yet. The stale source-bundle/current-head mismatch was removed by the
  follow-up regeneration.

## Adversarial review

The important bug class is not a scoring bug. It is a provider-spend false
authority bug: multiple individually sensible artifacts could be mistaken for a
complete non-dry-run route. The gate prevents that by requiring the exact
conjunction and recording every missing piece in one JSON artifact.

2026-05-17 follow-up: a sharper false-authority hole was found after the first
landing. `scripts/lightning_repro_workspace.py --dry-run` can write a local
`source_manifest.json`; without a separate transfer receipt, that local dry-run
manifest could masquerade as staged provider evidence. The hardened route now
requires `source_manifest_receipt.json` per cell with
`status=OK`, `dry_run=false`, and `remote_sha256_verified=true`, plus matching
`run_id`, manifest path, manifest SHA-256, file count, and byte count. Dry-run
receipts are explicit negative evidence and block non-dry-run submission.

This is also a non-retread action for the L5 v2 staircase. It does not polish
old PR106/FEC6 work and does not add council prose. It moves the TT5L paired
measurement toward a real launchable packet by converting remaining blockers
into executable prerequisites.

## Verification

```bash
.venv/bin/python -m pytest src/tac/tests/test_l5_v2_tt5l_lightning_non_dry_run_gate.py -q
.venv/bin/ruff check src/tac/optimization/l5_v2_tt5l_lightning_non_dry_run_gate.py tools/check_l5_v2_tt5l_lightning_non_dry_run_gate.py src/tac/tests/test_l5_v2_tt5l_lightning_non_dry_run_gate.py
.venv/bin/python -m pytest src/tac/tests/test_l5_v2_tt5l_sideinfo_lightning_execution_bundle.py src/tac/tests/test_l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run.py src/tac/tests/test_l5_v2_tt5l_lightning_non_dry_run_gate.py -q
.venv/bin/python -m pytest src/tac/tests/test_l5_v2_tt5l_sideinfo_lightning_execution_bundle.py src/tac/tests/test_l5_v2_tt5l_lightning_non_dry_run_gate.py src/tac/tests/test_lightning_repro_workspace.py -q
```

Observed:

- `6 passed`
- `ruff: All checks passed`
- `23 passed`
- `34 passed`

## Next route-unblock actions

1. Regenerate the TT5L execution bundle on the current head.
2. Run the required Lightning doctor command from the doctor plan and save the
   doctor JSON at the canonical output path.
3. Stage all ten source manifests with the bundle's
   `stage_source_manifest_command_template` values and preserve each generated
   `source_manifest_receipt.json`.
4. Claim all ten lanes with `claim_command`, preserving `score_claim=false`,
   archive SHA, variant, and axis in notes.
5. Replace non-dry-run placeholders with concrete Lightning identity/workspace
   values, rerun the gate with `--strict-ready`, then and only then submit.
