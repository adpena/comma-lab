# Codex Subagent Signal Recovery

Timestamp UTC: 2026-05-24T15:53:56Z

## Pool Reconciliation

All six inherited subagents completed read-only work and were closed to free the
agent pool:

- `019e59ba-6805-7fa2-8a87-d23aa91d71c3` / Epicurus: pixelshuffle public-intake
  audit.
- `019e59bf-4868-7903-b32b-0cb4b1a1dccf` / Curie: MLX dynamic sweep batch-root
  API audit.
- `019e59ff-0dea-7432-91ef-d36aee0cc4be` / Goodall: MLX learned-sweep runtime
  telemetry audit.
- `019e5a0f-4ee6-7cc1-99d9-1b3bb5e36191` / Carson: pixelshuffle mechanism audit.
- `019e5a74-8732-78f0-9d96-1c2ad56d4c3b` / Arendt: final-byte automation
  integration audit.
- `019e5a74-bf8d-7350-a75d-30e20e401b9f` / Bacon: PR95 MLX/source-faithfulness
  audit.

A previously refused inverse-water-bucket audit was respawned after cleanup:
`019e5ab1-4916-72e2-a223-2cb5d07008f5` / Wegener.

## Recovered Signals

1. PixelShuffle should not become a standalone public-intake lane. Both audits
   found no public submission literally named pixelshuffle; the relevant signal
   is an operator primitive inside HNeRV/NeRV/BoostNeRV lowering and older weak
   proxy postfilter experiments.

2. The highest-leverage final-byte gap is a canonical materializer-context
   compiler. Current planners can select water-bucket and rate-attack units, but
   candidate-producing execution still depends on manually supplied
   `--materializer-contexts`.

3. MLX dynamic sweep batching needs a safe root-selection artifact that selects
   one executable local-MLX root per optimization pass until row-specific
   actuator filtering exists. It must refuse exact-eval/CPU-advisory/non-ready
   rows and require explicit false-authority fields.

4. MLX learned-sweep runtime telemetry helpers exist, but the public builder and
   CLI do not yet consume runtime telemetry for adaptive root grouping. Tests
   should prove observed seconds alter grouping and mark aggregate multi-row
   telemetry as approximate or refuse it.

5. PR95 MLX remains synthetic/timing-only, not source-faithful PR95
   reproduction. Remaining source-faithfulness blockers are real video loading,
   source scorer/eval roundtrip loss, source hparams/schedule, QAT/resume,
   portable NumPy inflate parity, strict archive parsing, runtime custody, and
   exact CPU/CUDA auth eval.

6. PR95 archive parsing and exact-readiness blocker recomputation should be
   hardened before any source-faithful or rate-attack promotion path trusts the
   rebuilt artifacts.

## No-Signal-Loss Actions

- Closed completed subagents only after collecting their final outputs.
- Persisted the non-duplicative signals in this memo for subsequent agents.
- Respawned the missing inverse-water-bucket audit after the pool was cleared.
- Kept score/dispatch authority unchanged: all recovered signals are planning
  and hardening guidance until byte-closed archives plus exact auth artifacts
  exist.
