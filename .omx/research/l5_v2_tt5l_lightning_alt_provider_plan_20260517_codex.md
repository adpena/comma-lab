# L5 v2 TT5L Lightning Alternate Provider Plan - 2026-05-17

## Verdict

Modal failed before job spawn on the TT5L random_lsb paired measurement because
the workspace billing cycle spend limit was reached. This is a provider-capacity
blocker, not a TT5L method result.

Lightning is a plausible alternate exact-CUDA route, but it is not currently
execution-ready from this checkout. The local Lightning SDK/supply-chain doctor
is green, and an exact-CUDA dry-run job spec exists for the same archive,
runtime, axis, and pair group. Execution is blocked on provider identity and
capacity evidence.

## Packet

- Archive: `experiments/results/time_traveler_l5_v2/tt5l_sideinfo_variant_packets_20260517_codex/random_lsb/archive.zip`
- SHA-256: `b6a5b63c0ea8acd582d8f273a1ee9e00f74becc9d1993a2f3085f2f89d64b1c7`
- Bytes: `38911`
- Axis: `[contest-CUDA]`
- Pair group: `pair_l5_v2_measure_tt5l_autonomy_paired_exact_cpu_cuda`
- Runtime: `experiments/results/time_traveler_recovered_exact_eval_20260514_codex/runtime`
- Expected CUDA runtime tree SHA-256: `5518199a6f95f43366aa37bfb7452ae76892abdda73ff489f1fffaaf1dd27583`

## Evidence

- `launch_lightning_batch_job.py doctor` returned `status=OK`.
- Local Lightning supply-chain scan passed with `lightning-sdk==2026.4.23`.
- SSH auth and machine inventory were skipped because no `LIGHTNING_SSH_TARGET`
  or `LIGHTNING_TEAMSPACE` was configured.
- `lightning_repro_workspace.py --dry-run` refused to stage because the remote
  target is missing: `set --remote or LIGHTNING_SSH_TARGET before staging to Lightning`.
- `launch_lightning_batch_job.py exact-eval --dry-run` emitted an exact-CUDA
  job spec for `l5-v2-tt5l-random-lsb-cuda-20260517`.
- Dry-run command SHA-256: `72b693724b3e13eb06c066239362a8058231f7a81072244ae981041d7f68568d`.

## Execution Blockers

- `missing_lightning_ssh_target`
- `missing_lightning_teamspace`
- `machine_inventory_not_checked`
- `source_manifest_not_staged`
- `remote_cuda_runtime_not_probed`

## Next Action

After Lightning identity/teamspace is configured, run the doctor with SSH and
machine inventory, stage the source manifest including the TT5L archive and
runtime, claim `lane_l5_v2_measure_tt5l_autonomy_paired_exact_contest_cuda`,
then submit the exact-CUDA job with the dry-run command template in the JSON
artifact.

This artifact carries no score claim, no promotion eligibility, and no exact
dispatch readiness.
