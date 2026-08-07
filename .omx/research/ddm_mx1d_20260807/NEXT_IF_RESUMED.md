# ddm_mx1d Next If Resumed

Date: 2026-08-07
Tokens: [no-triality] [p0-ledger-ok]

## Current State

Use `.omx/research/ddm_mx1d_20260807/launch_ticket_v3_fire_guarded.json` as the
current Row-1 guarded ticket. The old mx1c ticket is superseded for fire protocol.

The local precheck verdict at
`.omx/research/ddm_mx1d_20260807/row1_v2_two_arm/launch_arm_cap/n32_metal/fire_guard_verdict.json`
is expected to be `status=failed`, `reason_code=mem_probe_receipt_missing`. That failed
verdict is safe: `mlx-train --device gpu` will refuse it.

## MAIN Fire Sequence

For the first CAP n32 fire, follow the ticket's `main_fire_sequence` exactly:

1. `guard_precheck`
   - Run `fire_guard_commands.argv_n32_arm_cap`.
   - Before the Metal mem-probe, expected rc is nonzero unless a valid keyed receipt already exists.
2. `probe`
   - Run `mem_probe_commands.argv_n32_arm_cap` on the MAIN Metal host.
   - It must write
     `.omx/research/ddm_mx1d_20260807/row1_v2_two_arm/launch_arm_cap/n32_metal/mem_probe/mem_probe_receipt.json`.
   - Required receipt: `schema=ddm_mx1_load_phase_peak_receipt.v1`, `status=passed`,
     `metal_fire_clearance=true`, final `after_train_step_000003` sample with MLX telemetry,
     and `memory_limits.hard_limit_satisfied=true` unless `--allow-soft-mem-limit` was
     explicitly passed.
3. `gate`
   - Rerun `fire_guard_commands.argv_n32_arm_cap`.
   - It must write
     `.omx/research/ddm_mx1d_20260807/row1_v2_two_arm/launch_arm_cap/n32_metal/fire_guard_verdict.json`
     with `status=passed`.
4. `fire`
   - Only then run `argv_n32_arm_cap`.
   - The trainer itself re-reads `--fire-guard-verdict`; absent/failed/malformed verdict exits 9
     before MLX setup.

For VEH n32, do not reuse the CAP receipt. Run `mem_probe_commands.argv_n32_arm_veh`, then
`fire_guard_commands.argv_n32_arm_veh`, then `argv_n32_arm_veh`. The keyed probe exists because
VEH uses the tq1c input cache while CAP uses the GT input cache.

Do not fire either n120 arm until the two n32 CPU-torch verdicts select the scaled arm. The n120
keys also have their own keyed mem-probe commands and guard verdict paths.

## Boundaries

- No scorer slot is owned by this arm.
- No Metal training, scorer, archive build, remote dispatch, or `upstream/evaluate.py` was run here.
- Local sandbox MLX probe remains blocked by inaccessible Metal; the MAIN host must produce the
  passed receipt.
- Own-vehicle frontier remains `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`.
