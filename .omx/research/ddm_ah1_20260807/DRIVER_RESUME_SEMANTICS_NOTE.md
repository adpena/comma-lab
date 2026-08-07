# Driver Resume Semantics Note

status: DESIGN-ONLY
charter_item: H5

No driver code was changed for H5 in AH1.

## Required Semantics

A resumable driver should compute remaining work from checkpoint metadata, not from a caller-side loop counter that resets on resume.

Required checkpoint fields:

- `global_epoch_completed`
- `stage_name`
- `stage_epoch_completed`
- `planned_total_epochs`
- `planned_stage_epochs`
- `checkpoint_kind`
- `config_sha256`
- `dataset_or_cache_sha256`

Resume rule:

`remaining_stage_epochs = planned_stage_epochs[stage_name] - stage_epoch_completed`

`remaining_global_epochs = planned_total_epochs - global_epoch_completed`

The driver must refuse when:

- checkpoint config hash does not match the launch config,
- checkpoint stage is not in the launch plan,
- checkpoint claims more completed epochs than the plan,
- the driver cannot distinguish a loop-end checkpoint from a stage-boundary checkpoint.

## Acceptance Test Shape

Use a synthetic two-stage plan and three checkpoints:

- resume from middle of stage A,
- resume from exact stage boundary into stage B,
- refuse stale checkpoint with mismatched config hash.

The test should assert that no resumed run repeats completed epochs and no resumed run skips owed epochs.

