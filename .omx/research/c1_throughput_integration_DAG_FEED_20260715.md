# C1 throughput/convergence integration DAG FEED — 2026-07-15

`research_only=true`; pointer unchanged. This is a collision-safe FEED for concurrent-append then
union; MAIN may append it to the shared DAG only after branch review.

## Graph

```text
C1a v9_cgauge_ideal_mod19_sR (commit bdbbf5da175a46c11393ebbe56f53653828fb765)
  -> S_R consumer requires --micro-batch-pairs 1
  -> C1 strict speed identity wall
       -> ON: fixed-order fused R [per-chip startup parity]
       -> ON: constant GT skeleton cache [exact constant reuse]
       -> ON: explicit-order persistence pool [max_abs_delta=0]
       -> ON: one-thread frozen training scorer [adopted training standard]
       -> ON: observational async verdict + exact verdict chunking
       -> ON: component timers [measurement-only]
       -> REFERENCE: custom grouped VJP [primary proof is cosine/roundoff only]
       -> REFERENCE: safe compile [host certificate absent]
       -> EXCLUDE: B>1 [S_R consumer absent + fp-reduction identity wall]
       -> EXCLUDE: whole-step megakernel [fp reorder + marginal/negative wall]
  -> typed DSL program v9_cgauge_ideal_mod19_sR_c1_throughput
  -> content-bound benchmark gate
       -> requires real GT SHA + S_R SHA + argv SHA + typed hash
       -> requires sec/ep + peak RSS + identity PASS
       -> BLOCKED_INPUT_CUSTODY in this worktree
  -> C1b launch REFUSE
```

## Canonical equations

- `witness_fp_reorder_transform_bit_identity_wall_v1`: whole-step compile and B>1 fp-reorder
  exclusions; also forces the reference grouped VJP under C1's strict contract.
- `mlx_gpu_crossprocess_nondeterminism_v1`: fixed-order fused-R graph, 0/28 divergent tensors and
  25.35 s -> 23.44 s in the scoped smoke.
- `segnet_exact_forward_cpu_thread_control_v1` and
  `segnet_exact_forward_cpu_thread_static_process_v2`: training scorer thread selection.

## Consumer and apparatus edges

- DSL/launcher: `src/tac/witness_dsl/spec_c1_throughput_20260715.py` and
  `tools/launch_witness_run.py`.
- Runtime consumer: `experiments/train_levelset_witness_realized_through_R_mlx.py`.
- Lever registry: all named C1 factories are AST-discovered and zero-argument composable; all emitted
  trainer flags are MAPPED. Runtime-kernel choices are typed Lever `runtime_environment` fields.
- Activation ledger: each new factory appears in `duty_to_measure()` until a governed launch records
  fired/measured events. The blocked receipt does not falsely mark them measured.
- Costate digest: run on 2026-07-15; it remained CLEAN and exposed 97 registered duties before launch.
- Sensitivity/Pareto/bit allocator: non-binding because this integration changes wall-clock MEANS and
  emits no archive or score row.
- Autopilot: `launch_blockers != []` is a hard REFUSE; benchmark input mutation invalidates admission by
  SHA-256.

## Scoped negatives and reopen triggers

- Whole-step megakernel: `FORMULATION`, not the compile/fusion family. Reopen on strict-order semantics
  plus exact eager parity and whole-step gain.
- Custom grouped backward: `FORMULATION ON C1 STRICT-IDENTITY VEHICLE`. Reopen on exact tensor-byte
  equality against the reference VJP on the actual composed path; its measured 17.96x/5.5x wall wins
  remain valid functional-parity evidence.
- Micro-batch B>1: `CURRENT C1 S_R/BATCHED TWIN`. Reopen when a batched S_R consumer and fixed-order
  serial-identical scorer/reduction path both exist.
- Safe compile: `CURRENT HOST CERTIFICATE`. Reopen with a fingerprint-fresh per-chip manifest.
- Costate/forward-kill/ANE negatives are formulation- or execution-substrate-scoped; families remain
  open as recorded in the source audit.
- Current benchmark negative is `INPUT/RUNTIME CUSTODY ONLY`; it says nothing about expected C1 speed.
