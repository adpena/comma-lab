# Score-lowering checkpoint: preflight wall clock + Kaggle CUDA classification (2026-05-11)

## Scope

This checkpoint records two operator-critical facts from the 2026-05-11 Codex
turn:

1. the all-lanes preflight wall-clock guard is currently fast enough for normal
   development use while preserving full gate coverage; and
2. the Kaggle-generated PR101 proxy runtime packet is classified as terminal
   negative on the contest-CUDA axis, not as a frontier or submission candidate.

This is not a new score claim and not a new dispatch. It is a custody and
decision checkpoint so future agents do not redispatch stale candidates or
weaken preflight protection for speed.

## Preflight wall-clock proof

Command:

```bash
PREFLIGHT_SECONDS=2 .venv/bin/python tools/all_lanes_preflight.py \
  --timings-json reports/preflight_latest_timing.json
```

Result:

- status: `ALL 29 PREFLIGHT CHECKS PASSED`
- wall elapsed: `2.374314 s`
- serial elapsed: `11.952758 s`
- estimated parallel speedup: `5.034195x`
- failed steps: `0`
- slow steps: `9`
- slowest recorded step: `2.338769 s`

The generated timing JSON was intentionally not committed because it is a
rebuildable report artifact. Durable policy already lives in
`tools/all_lanes_preflight.py`: default 30 s all-lanes budget, explicit
`--allow-slow-preflight` override, cooperative cancellation, subprocess timeout
capping, and a hard watchdog that exits `124` after budget plus grace if a gate
does not return cooperatively. Focused tests passed:

```bash
.venv/bin/python -m pytest -q \
  src/tac/tests/test_all_lanes_preflight_timing_profile.py \
  src/tac/tests/test_preflight_cli_timeout.py
# 24 passed
```

## Kaggle/PR101 exact-CUDA classification

The completed Kaggle PR101 proxy/bias work produced a byte-closed runtime
packet that was promoted through Modal T4 exact CUDA eval. Current terminal
artifact:

- path:
  `experiments/results/modal_auth_eval/pr101_kaggle_proxy_runtime_packet_exact_cuda_modal_20260510T194142Z/contest_auth_eval.json`
- evidence axis: `[contest-CUDA]`
- canonical score: `0.22688160652506983`
- rounded final score: `0.23`
- SegNet distance: `0.00066894`
- PoseNet distance: `0.00017051`
- samples: `600`
- archive bytes: `178258`
- archive SHA-256:
  `b83bf3488625dbd73adeddff91712994197ab53098e578e91327a0c6e49efb3e`
- promotion eligible: `false`

Apples-to-apples comparison is against the A1 contest-CUDA anchor, not A1
contest-CPU or any macOS/MPS proxy result:

- A1 `[contest-CUDA]`: `0.2263520234784395`, `178262 B`,
  archive SHA-256
  `87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5`
- Kaggle proxy runtime packet `[contest-CUDA]`: `0.22688160652506983`,
  `178258 B`
- delta vs A1 CUDA: `+0.00052958304663033` worse despite `4 B` fewer archive
  bytes.

Verdict: measured-config negative on the contest-CUDA axis. Keep the artifact
as runtime-consumption proof and a CPU/proxy-to-CUDA drift calibration point.
Do not redispatch this same archive/runtime identity as score-lowering work.
The exact-ready audit and retraction manifest already suppress stale queue rows
for this lane.

## Current active score-lowering lane

Dispatch claim summary still has exactly one active lane:

- lane: `t1_balle_128k_endtoend`
- job: `t1_balle_modal_phase1_ab2d0f6_20260510T1437Z`
- platform: Modal
- status: `active_dispatching`

Recover check:

```bash
.venv/bin/python experiments/modal_t1_balle_endtoend.py recover \
  --label t1_balle_modal_phase1_ab2d0f6_20260510T1437Z
```

Result: `NOT READY: call_id=fc-01KR955JSYQAVTTYZA48VAV7WJ still queued or running`.

Do not launch a duplicate T1 job. Next score-lowering action is to harvest and
classify this T1 run when Modal returns an artifact, or to move to a different
claimed lane with a fresh charged-byte/runtime identity.

## Next engineering implications

- Keep all normal preflight paths under the 30 s wall-clock budget. If a future
  check pushes the normal command past budget, treat it as a DX regression and
  split/cache/vectorize that check before dispatch.
- Keep Kaggle, MPS, and macOS CPU as proxy/config-discovery substrates only.
  They can nominate byte-closed candidates, but only contest-CPU or
  contest-CUDA evidence changes lane status.
- Avoid the stale PR101 bias/proxy local basin. Future score lowering should
  prioritize train-time substrate work, HNeRV parity discipline, runtime
  compiler/codec transforms with actual charged-byte changes, and CUDA/CPU drift
  mechanism probes that emit reusable evidence rather than another same-archive
  coordinate sweep.
