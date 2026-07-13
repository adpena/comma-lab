# DAG FEED — throughput fresh-eyes truthing — 2026-07-13

**research_only=true · score_claim=false · pointer_moved=false**  
**Axis:** `[macOS local wall-clock advisory] NON-PROMOTABLE`

## FEED-TF-1 — async overlap is not contention identity

```text
MEASURED_INPUTS := {T_train_solo, T_train_concurrent, T_async}
T_window := max(T_train_concurrent, T_async)
delta_contention := T_train_concurrent / T_train_solo - 1
T_exposed_tail := max(0, T_async - T_train_concurrent)
```

An async-only log may prove `cadence_miss_count == 0`; it cannot bind `delta_contention`.

- equation: `async_overlap_and_inclusive_vjp_throughput_v1`
- DSL disambiguator: `throughput_component_timer_async_20260713` versus
  `throughput_component_timer_solo_20260713`
- verdict_scope: matched host/config wall-clock MEANS only; no contest axis, score, archive, fidelity, or
  promotion transfer
- req-R: re-run matched solo/concurrent when host load, frozen-scorer build, thread binding, MLX build,
  batch geometry, or verdict cadence changes

## FEED-TF-2 — inclusive backward subtraction

```text
T_teacher_vjp_incremental := T_teacher_backward_inclusive - T_teacher_forward
T_witness_vjp_incremental := T_witness_backward_inclusive - T_witness_forward
```

Reject negative differences as unresolved order/cache/noise, and reject any additive epoch accounting that
sums inclusive backward with its forward. D-A is a one-pair, update-free probe; multiplication by 25 is
`n24-linear-extrapolation`, never n600 measured.

## FEED-TF-3 — campaign routing gate

```text
if measured_teacher_vjp_share >= 0.60:
    route := backward/costate campaign
elif measured_teacher_forward_share >= 0.60:
    route := forward campaign
else:
    route := mixed whole-teacher campaign
```

The 0.60 thresholds are the pre-registered decision bands from the held GO packet, not score laws. The
autopilot must remain `NEEDS_MEASUREMENT` if the timer receipt is missing, incomplete, or lacks the inclusive
subtraction.

## Six-hook wire-in

1. Sensitivity map: none; time is a means-only axis.
2. Pareto constraint: score/archive bytes unchanged; any fidelity delta refuses the timing treatment.
3. Bit allocator: non-binding.
4. Cathedral/autopilot: block forward/backward campaign routing until FEED-TF-2 is measured.
5. Continual learning: append the matched JSON receipt and supersede, never overwrite, this prior.
6. Probe-disambiguator: matched async/solo typed programs; no single-mode guess.

Canonical registry append is deferred because `.omx/state/canonical_equations_registry.jsonl` is currently
dirty under a sibling lane. The executable equation and this feed are the durable non-colliding handoff.
