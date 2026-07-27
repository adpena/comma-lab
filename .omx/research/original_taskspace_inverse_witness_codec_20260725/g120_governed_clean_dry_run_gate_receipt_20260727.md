# G120 governed clean dry-run gate receipt

Date: 2026-07-27  
Task: `pact-g120-governed-clean-dry-run-gate-20260727`  
Axis: `[macOS-CPU orchestration dry-run; no scorer or score authority]`  
Producer:
`/Volumes/VertigoDataTier/pact/g111_batch16_v9_semantic_base_dry_start_v6_20260727`  
Status: gate closed for the exact producer launch and current source bytes

## Adversarial audit verdict

The pre-fix production path did not satisfy the G120 acceptance contract:

1. G120-v2 persisted immutable per-batch prediction arrays and receipts but
   never reopened them before invoking the scorer. A crash therefore repeated
   every already-complete scorer batch.
2. The G121 live monitor had no structurally required receipt proving the
   production wrapper, physical SSD custody, storage preflight, and
   crash-resume behavior before it could append G120 work.
3. The production invocation therefore could not prove the exact gate named by
   `SPEC_g120_parsed_stage_production_authority_20260727.md`.

The closure makes the real production batch path receipt-resumable and makes
the exact dry-run receipt a required input to every live-monitor launch epoch.

## Landed behavior

- G120-v2 now stable-reopens each canonical batch receipt and bound NPY before
  any scorer invocation. Exact completed batches skip the scorer; malformed,
  mismatched, or receipt-without-array states fail closed.
- The G120 measurement records freshly measured and physically resumed batch
  counts. Reopening requires the two counts to equal the full production batch
  census and requires direct scorer calls to be compatible with the fresh
  count.
- `tools/run_g120_governed_clean_dry_run.py` supplies a two-process
  `checkpoint`/`resume` CLI. It uses the same production persist/reopen
  primitives with one batch-16, 384x512 prediction artifact, but supplies no
  scorer callable and emits no semantic measurement.
- `tools/run_taskspace_g121_live_stage_harvest.py` now requires the exact dry-run
  receipt path and externally supplied SHA-256. It reopens the receipt before
  monitor binding, before registering a launch epoch, and again when adopting
  the latest launch epoch.

## Exact executed dry-run

Both phases used the same arguments and ran in separate shell processes:

```bash
.venv/bin/python tools/run_g120_governed_clean_dry_run.py checkpoint \
  --producer-run-dir /Volumes/VertigoDataTier/pact/g111_batch16_v9_semantic_base_dry_start_v6_20260727 \
  --expected-launch-manifest-sha256 3041badae1329b188bce1ef52bb99754e4430e41aa24e80d076e5f3dfbdbe786 \
  --monitor-output-dir /Volumes/VertigoDataTier/pact/g111_batch16_v9_semantic_base_dry_start_v6_20260727/g121_harvest \
  --monitor-progress-dir /Volumes/VertigoDataTier/pact/g111_batch16_v9_semantic_base_dry_start_v6_20260727/g121_progress \
  --measurement-cache-dir /Volumes/VertigoDataTier/pact/g111_batch16_v9_semantic_base_dry_start_v6_20260727/g121_measurement_cache \
  --gate-dir /Volumes/VertigoDataTier/pact/g111_batch16_v9_semantic_base_dry_start_v6_20260727/g120_dry_run_gate

.venv/bin/python tools/run_g120_governed_clean_dry_run.py resume \
  --producer-run-dir /Volumes/VertigoDataTier/pact/g111_batch16_v9_semantic_base_dry_start_v6_20260727 \
  --expected-launch-manifest-sha256 3041badae1329b188bce1ef52bb99754e4430e41aa24e80d076e5f3dfbdbe786 \
  --monitor-output-dir /Volumes/VertigoDataTier/pact/g111_batch16_v9_semantic_base_dry_start_v6_20260727/g121_harvest \
  --monitor-progress-dir /Volumes/VertigoDataTier/pact/g111_batch16_v9_semantic_base_dry_start_v6_20260727/g121_progress \
  --measurement-cache-dir /Volumes/VertigoDataTier/pact/g111_batch16_v9_semantic_base_dry_start_v6_20260727/g121_measurement_cache \
  --gate-dir /Volumes/VertigoDataTier/pact/g111_batch16_v9_semantic_base_dry_start_v6_20260727/g120_dry_run_gate
```

Bound producer manifest:

- path:
  `/Volumes/VertigoDataTier/pact/g111_batch16_v9_semantic_base_dry_start_v6_20260727/launch_manifest.json`
- SHA-256:
  `3041badae1329b188bce1ef52bb99754e4430e41aa24e80d076e5f3dfbdbe786`

Checkpoint:

- path:
  `/Volumes/VertigoDataTier/pact/g111_batch16_v9_semantic_base_dry_start_v6_20260727/g120_dry_run_gate/g120_governed_clean_dry_run_checkpoint.json`
- SHA-256:
  `49d8ec4439a6b680b105f6c4227959ca5d5b1905312d6a913667c0c8da4ea9b6`
- PID: `24055`

Completion:

- path:
  `/Volumes/VertigoDataTier/pact/g111_batch16_v9_semantic_base_dry_start_v6_20260727/g120_dry_run_gate/g120_governed_clean_dry_run_receipt.json`
- SHA-256:
  `178907cd677d2d6a2e2b6a3a394e110778a2d3c524365e5db38db786882ffc65`
- PID: `24339`
- `clean_dry_run_complete=true`
- `completed_batch_reopened=true`
- `scorer_callable_supplied=false`
- `scorer_calls=0`

The checkpoint storage preflight observed `347411787776` free bytes and the
resume preflight observed `347408625664` free bytes at every bound path. Both
exceeded:

`B_min = 2*N*384*512 + 4*16*(384*512*3 + 874*1164*3)`

with `N=600`, or exactly `469006848` bytes.

## Authorized production monitor invocation

This is the literal v6-bound invocation admitted by the receipt:

```bash
.venv/bin/python tools/run_taskspace_g121_live_stage_harvest.py \
  --producer-run-dir /Volumes/VertigoDataTier/pact/g111_batch16_v9_semantic_base_dry_start_v6_20260727 \
  --expected-launch-manifest-sha256 3041badae1329b188bce1ef52bb99754e4430e41aa24e80d076e5f3dfbdbe786 \
  --g120-dry-run-receipt /Volumes/VertigoDataTier/pact/g111_batch16_v9_semantic_base_dry_start_v6_20260727/g120_dry_run_gate/g120_governed_clean_dry_run_receipt.json \
  --expected-g120-dry-run-receipt-sha256 178907cd677d2d6a2e2b6a3a394e110778a2d3c524365e5db38db786882ffc65 \
  --output-dir /Volumes/VertigoDataTier/pact/g111_batch16_v9_semantic_base_dry_start_v6_20260727/g121_harvest \
  --progress-dir /Volumes/VertigoDataTier/pact/g111_batch16_v9_semantic_base_dry_start_v6_20260727/g121_progress \
  --poll-seconds 30
```

The monitor was not launched by this unit because a live monitor may invoke the
heavy G120 scorer when a preserved G111 stage becomes eligible. This unit was
authorized to close the gate, not to launch that scorer.

## Triality

DSL:

`LaunchManifest x PhysicalPaths x SourceBindings x BatchResumeProof
-> G120DryRunAdmission`.

DAG:

`G111 launch -> gate checkpoint process -> immutable production batch
-> distinct resume process -> exact receipt -> G121 launch epoch -> G120-v2`.

Equation:

`complete_b(E) = receipt_b exists AND array_b exists AND
reopen(receipt_b, array_b, target_b, scorer_hash_b, camera_hash_b) == exact`;

`scorer_calls_b = 0` when `complete_b(E)`, otherwise `1`.

## Authority and verification

- `ruff check` passed on all seven changed Python and focused-test files.
- `27 passed` across the G120-v2, dry-run-gate, and G121 live-monitor tests.
- This receipt is apparatus evidence only. It launched no SegNet scorer, emitted
  no semantic measurement, ran no exact evaluation, produced no candidate, and
  did not move the frontier pointer.
