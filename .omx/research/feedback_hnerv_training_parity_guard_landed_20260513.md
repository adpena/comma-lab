# HNeRV Training Parity Guard Landed - 2026-05-13

## Scope

Objective: prevent PR95/PR100/PR101 HNeRV parity regressions and false
readiness in trainer surfaces without touching PacketIR compiler code or
substrate inventory.

## Landing

- Added `src/tac/hnerv_training_parity_guard.py`.
- Added focused tests in `src/tac/tests/test_hnerv_training_parity_guard.py`.

The guard statically verifies that HNeRV-family trainers preserve:

- differentiable `rgb_to_yuv6` patching before scorer construction;
- `apply_eval_roundtrip=True` in the score-aware training path;
- EMA update, validation/export application, and EMA shadow save;
- archive build-in-loop via `pack_archive`, `_write_runtime`, and
  `_build_archive_zip`;
- exact contest runtime signature:
  `inflate.sh <archive_dir> <output_dir> <file_list>` and `inflate.py`
  consuming `sys.argv[3]` plus `archive_dir/0.bin`;
- scorer-free and network-free emitted inflate runtime templates.

## Evidence

Focused test command:

```bash
.venv/bin/python -m pytest src/tac/tests/test_hnerv_training_parity_guard.py -q
```

Result at landing:

```text
5 passed in 0.13s
```

## Unified Solver Wire-In

- Sensitivity-map contribution: N/A - static guard only; no empirical anchor or
  tensor importance update.
- Pareto constraint: N/A - rejects false readiness; does not change a packet
  frontier.
- Bit-allocator hook: N/A - no allocation policy or stream weights changed.
- Cathedral autopilot dispatch hook: N/A - guard is a pre-dispatch readiness
  filter and does not enqueue work.
- Continual-learning posterior update: N/A - no score result observed.
- Probe-disambiguator: N/A - no competing design interpretation; this is a
  fail-closed contract check.

research_only=true for solver-stack hook purposes; this landing is a reusable
guard/test artifact, not a score claim or dispatchable lane.
