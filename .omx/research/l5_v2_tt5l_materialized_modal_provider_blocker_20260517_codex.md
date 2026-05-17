# L5 v2 TT5L materialized Modal provider blocker

- date: 2026-05-17
- scope: TT5L `random_lsb` materialized paired CPU/CUDA measurement
- provider: Modal
- failure class: `modal_workspace_billing_cycle_spend_limit_reached`
- score claim: `false`
- promotion eligible: `false`

## Current Work Unit Scope

The blocker is refreshed onto the current materialized TT5L work unit in
fail-closed mode. No current-archive Modal job was spawned; the active status is
a provider-level carry-forward from the earlier Modal workspace billing failure.

- archive: `experiments/results/time_traveler_l5_v2/tt5l_sideinfo_variant_packets_current_code_fullshape_advisory_20260517T052719Z/random_lsb/archive.zip`
- archive sha256: `ccce77aaf1907d6e70d8cba498261708b241ac7d78a9bf22978aa459cb6b7fd1`
- archive bytes: `38681`
- run id: `l5_v2_measure_tt5l_autonomy_paired_exact_paired_measurement_random_lsb_ccce77aaf190`
- pair group: `pair_l5_v2_measure_tt5l_autonomy_paired_exact_cpu_cuda`
- refresh class: `provider_level_blocker_carried_forward_to_current_work_unit`
- live refresh attempted: `false`

## Source Attempt Preserved

- archive: `experiments/results/time_traveler_l5_v2/tt5l_sideinfo_variant_packets_20260517_codex/random_lsb/archive.zip`
- archive sha256: `b6a5b63c0ea8acd582d8f273a1ee9e00f74becc9d1993a2f3085f2f89d64b1c7`
- archive bytes: `38911`
- run id: `l5_v2_measure_tt5l_autonomy_paired_exact_paired_measurement_random_lsb_b6a5b63c0ea8`
- canonical tool: `tools/dispatch_modal_paired_auth_eval.py`

## Result

The earlier paired dispatch plan validated the archive SHA and produced
distinct CPU/CUDA command plans for archive-specific output directories. Modal
then failed before provider job spawn:

`App creation failed: workspace billing cycle spend limit reached`

No paired score job was created for the source attempt. The blocker is not TT5L
method evidence; it is provider-capacity evidence. Because the failure is at the
workspace/provider level, the current TT5L work unit remains fail-closed until
Modal billing is positively resolved or an alternate contest-compliant provider
route is executed.

## Current Routing

`src/tac/optimization/l5_staircase_v2.py` now reads the JSON blocker artifact and changes the TT5L next action from retrying the same Modal dispatch to:

`resolve_l5_v2_tt5l_modal_provider_blocker_or_dispatch_alternate_provider`

The byte-closed TT5L work unit remains valid; only the Modal provider path is
blocked until the billing limit is resolved or an alternate contest-compliant
provider path is selected.

## Execute Leak Hardening

2026-05-17 follow-up: the provider-blocked next action no longer carries any
recursive `--execute` string, including inside the embedded materialized work
unit status. Earlier hardening removed the top-level execute command from the
provider-blocked action; this pass also suppresses
`operator_execute_command_template_after_review` at the work-unit-status layer
whenever the Modal blocker artifact is active or invalid.

Verification receipt:

- focused regressions:
  `.venv/bin/pytest -q src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_tt5l_materialized_work_unit_surfaces_modal_billing_blocker src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_tt5l_stale_modal_blocker_blocks_execute_command src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_tt5l_modal_blocker_surfaces_lightning_alternate_plan`
- result: `3 passed`
- live recursive scan:
  `next_action=resolve_l5_v2_tt5l_modal_provider_blocker_or_dispatch_alternate_provider`,
  `recursive_execute_present=False`, `nested_keys=[]`

## Lightning Source Manifest Probe Refresh

2026-05-17 follow-up: reran the Lightning repro workspace source-manifest probe
against the current `ccce77aaf1907d6e70d8cba498261708b241ac7d78a9bf22978aa459cb6b7fd1`
archive and current runtime path. The no-spend dry run still fails before any
remote staging because no Lightning SSH identity is configured locally:

```text
set --remote or LIGHTNING_SSH_TARGET before staging to Lightning
```

This refresh converts the older inherited `not_rerun_after_archive_refresh`
note into current evidence:
`rerun_after_archive_refresh_against_current_archive_and_runtime`. It does not
change the method status, score claim, promotion eligibility, or exact-dispatch
readiness.

2026-05-17 validator follow-up: the L5 v2 alternate-provider status now treats
that refreshed source-manifest probe as required evidence. A Lightning alternate
plan is not artifact-valid unless `source_manifest_probe` is tied to the same
current archive path and runtime path, has an `observed_at_utc`, and records the
`rerun_after_archive_refresh_against_current_archive_and_runtime` basis. This
prevents stale source-manifest probes from surviving as valid alternate-provider
evidence after TT5L archive/runtime refreshes.

Verification receipt:

- focused regressions:
  `.venv/bin/pytest -q src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_tt5l_modal_blocker_surfaces_lightning_alternate_plan src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_tt5l_lightning_alternate_plan_rejects_stale_source_manifest_probe`
- result: `2 passed`
- live status:
  `alt_valid=True`, `source_manifest_probe_current=True`, `execution_ready=False`

The paired Lightning alternate-provider plan has also been refreshed onto the
same current archive/runtime metadata. It is artifact-valid, but dispatch is
still blocked on real Lightning environment prerequisites:

- `missing_lightning_ssh_target`
- `missing_lightning_teamspace`
- `machine_inventory_not_checked`
- `source_manifest_not_staged`
- `remote_cuda_runtime_not_probed`
