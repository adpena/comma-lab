# FEED — custom Metal frozen-SegNet convolutions — 2026-07-13

`research_only=true` · `score_claim=false` · `pointer_moved=false` ·
lane `lane_custom_metal_conv_20260713`

## Executable readiness DAG

```text
pinned frozen model + real receiver custody
  -> real Conv2d shape/MAC inventory                         DONE
  -> NumPy-fp32 references + deterministic packets           DONE
  -> pointwise fp16/int8/int4 Metal source                    BUILT-NEVER-FIRED
  -> depthwise fp16 3x3/5x5 Metal source                      BUILT-NEVER-FIRED
  -> default-OFF frozen-forward adapter                       DONE
  -> static tests                                             23-PASS / 5-DEVICE-SKIP
  -> evaluated Metal allocation                               BLOCKED-NOT-MEASURED
       -> device compilation                                  REFUSED-UPSTREAM-BLOCKER
       -> real-shape pointwise/depthwise timing                REFUSED-INCOMPLETE-MEASUREMENTS
       -> Metal-vs-NumPy parity + real-frame argmax             REFUSED-INCOMPLETE-MEASUREMENTS
       -> sibling im2col comparison                            DEFERRED-NONOVERLAPPING-GEOMETRY
       -> measured composed Amdahl                             REFUSED-INCOMPLETE-MEASUREMENTS
       -> deterministic VJP                                    NEEDS-BUILD
       -> custom-metal-segnet-conv-n600-fidelity-gate          NEEDS-MEASUREMENT
       -> governed training opt-in                             REFUSE
```

Current blocker: MLX reports a GPU device label here, but an evaluated allocation raises
`[metal::load_device] No Metal device available`. The sibling im2col probe first reached the same
blocker, then obtained a separate Metal-visible main-local receipt for the 3x3 stride-2 stem. Its
B=1 arms range from `1.0068950x` to `1.3478409x`; all B=8 arms lose (`0.4160392x` to
`0.5533905x`). Because it contains no 1x1 row, the requested pointwise comparison remains deferred,
not tied or won. The fallback `xcrun metal --version` compile probe here also returns 1 because the
optional Metal Toolchain is absent. These are environment-scoped edge cuts, not a kernel-family
negative.

## Canonical node statuses

| node | status | evidence / next gate |
|---|---|---|
| `custom-metal-pointwise-fp16-int8-int4` | `built-never-fired` | source + static tests; Metal compile/timing owed |
| `custom-metal-depthwise-fp16` | `built-never-fired` | source + static tests; bandwidth verdict owed |
| `custom-metal-device-execution` | `blocked-not-measured` | exact receipt blocker |
| `custom-metal-real-shape-timing` | `refused-incomplete-measurements` | zero timing rows |
| `custom-metal-device-parity` | `refused-incomplete-measurements` | NumPy-only static evidence is not Metal parity |
| `custom-metal-composed-amdahl` | `refused-incomplete-measurements` | structural ceiling only |
| `custom-metal-deterministic-vjp` | `needs-build` | forward-only source fails closed |
| `custom-metal-segnet-conv-n600-fidelity-gate` | `needs-measurement` | n600 paired latency/logits/argmax/gradients/replay |
| `custom-metal-training-opt-in` | `refused-upstream-gates` | default remains OFF |

## Six-hook wire-in

1. Sensitivity map: n600 final-logit/argmax and per-pair input-gradient drift.
2. Pareto constraint: matched full-forward wall-clock versus training-signal fidelity; any losing
   real geometry remains visible.
3. Bit allocator: non-binding because this is teacher compute and changes no archive bytes.
4. Cathedral/autopilot: candidate pool row is `built-never-fired`; dispatch refuses on the open
   device/VJP/n600 gates.
5. Continual learning: receipt canonicalizes the source inventory and exact environment blocker;
   later measurements supersede nulls without erasing this evidence.
6. Probe disambiguator: the sibling stem result preregisters batching as a decisive axis; matched
   custom-pointwise versus im2col 1x1 and custom-depthwise versus native measurements decide the two
   still-live interpretations.

## Pointer delta and triality

- Pointer delta: none.
- DSL: default-OFF environment selectors only; governed typed training policy remains owed.
- DAG: this file.
- Equation: `1 / 0.7542635826186354 = 1.3257964762506795x` is a DERIVED MAC-share ceiling,
  explicitly not an empirical law.

The durable source of truth for this FEED is
`.omx/research/custom_metal_conv_build_20260713.md`; the machine-readable receipt is
`experiments/results/custom_metal_conv_20260713/receipt.json`.
