# Codex Findings: HPRC Archive-Runnable Loop + PR95 MLX Drift Probe

UTC: 2026-06-01T00:58:55Z

## Landed Engineering

- `tools/run_hprc_compact_receiver_training.py` now accepts real contest video input via the canonical `decode_real_pairs` helper, downsamples deterministically to a low-resolution HPRC training tensor, records source video bytes/SHA and decoded-frame SHA, and can write an explicit `--output-manifest` for queue postconditions.
- `tools/build_hprc_compact_receiver_training_queue.py` builds a storage-waterfall-backed `experiment_queue.v1` campaign for the HPRC compact receiver train/export path. The selected bulky output tier is SSD-first and the queue owns execution.
- `comma_lab.scheduler.local_training_queue` now accepts `hprc_compact_receiver_training_plan.v1` and correctly treats an explicit `--output-manifest` as authoritative when a command also takes `--output-dir`.
- `HprcCompactReceiverLongTrainingAdapter.export_archive` now emits the full canonical HPRC false-authority set in its export manifest. The queue runner caught this gap before the fix.

## Live Run

Queue:
`.omx/research/hprc_compact_receiver_real_lowres_queue_20260601T005227Z.json`

Plan:
`.omx/research/hprc_compact_receiver_real_lowres_plan_20260601T005227Z.json`

SSD output:
`/Volumes/VertigoDataTier/pact/experiments/results/hprc_compact_receiver_training_queue/hprc_real_lowres_20260601T005227Z`

Result:

- Source: `upstream/videos/0.mkv`
- Source SHA-256: `2611f5f3e186f3529777749f97bd4cce3a208d6b3559e137bd45d256980d2fa9`
- Decoded frames: `4 x 48 x 64 x 3`
- Epochs: `2`
- Archive bytes: `27,150`
- Archive SHA-256: `01b8e6a110ef967a6a921f50c8503cdc1aa7dcba492354dc87375ce998ae0aff`
- Queue state: `succeeded`
- Exact dispatch: refused, as intended, until local replay and contest-axis gates clear.

## PR95 MLX Drift Probe

Probe:
`.omx/research/pr95_full_frame_inflate_parity_kahan_fp32_20260601T005314Z.json`

Result:

- Accumulation mode: `kahan_fp32`
- Public PR95 archive SHA-256: `e976acd5fe565c94fb9a8c62e5200c949919f76150e84599f268d6a58588440a`
- Byte exact: `false`
- Changed bytes: `101,503,942`
- Changed fraction: `0.027715070974038514`
- Max abs uint8 delta: `3`
- Mean abs uint8 delta: `0.02771587427031646`

Compared with the optimized baseline proof (`101,504,006` changed bytes), Kahan fp32 improves only `64` bytes over the whole full-frame output. This is a demotion signal for global Kahan as a primary PR95 MLX drift fix. It can remain as a calibration knob, but it is not the missing parity mechanism.

## Next 12-Week Tranche

1. Make the HPRC queue emit local replay gates by default: receiver proof, bounded raw output, component replay, and exact-readiness refusal in the same queue.
2. Scale HPRC from 2-pair low-res smoke to controlled 32/128/600-pair campaigns with SSD checkpoint retention, no local-disk bulk, and exact byte accounting.
3. Bind the Z8 residual sidecar and full-video P18/P19 allocator to HPRC outputs as queue follow-up steps, not standalone probes.
4. Continue PR95 MLX drift work with targeted layer/preset probes; demote global Kahan unless a layer-local version shows materially better full-frame parity.
5. Promote only byte-closed local winners to CPU exact auth, then CUDA exact auth if CPU clears.
