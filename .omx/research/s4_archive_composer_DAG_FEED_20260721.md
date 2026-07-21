# S4 archive composer DAG FEED

Status: research-only apparatus; MAIN review required; pointer unchanged.

```text
PPCS seed bytes -------------------\
BASE PBASE3 bytes ------------------+--> versioned section registry
causal PCR3 bytes ------------------+          |
event PCE3 bytes -------------------+          v
admitted PCOMP3 bytes --------------/      monolithic 0.bin
                                                |
                                                v
                                  strict standalone receiver
                                  | parse/hash/exact-consume
                                  | xi + causal replay
                                  | event/component decode
                                  | lane lattice + factor-2 R
                                  v
                          atomic 1200-frame RGB raw stream
                                  |                  |
                                  v                  v
                         repo-native parity      upstream evaluator
                         n16 -> n64 -> n600      macOS-CPU advisory
                                  |                  |
                                  +--------+---------+
                                           v
                                R6 pre-dispatch admission gate
```

## FEED edges

| Producer | Consumer seam | Gate |
|---|---|---|
| `realization_g2d` merged section bytes | replace registry-bound realization/component bytes | strict manifest/hash parse-back, then n16/n64/n600 parity |
| `predictor_r4` merged winning streams | replace a registered terminal with raw/LZMA/Brotli/#557 `range_static_v1` bytes | exact decoded length + section hash + double decode |
| S4 parity receipt | R6 | all three prefix rows exact; no scorer invocation during decode |
| S4 advisory receipt | R6 pre-Modal decision | decompose d_seg/d_pose/bytes/wall; never promote the macOS axis |

## Fail-closed rules

- Unknown section, codec, version, length, hash, order, or trailing byte refuses.
- Any video-derived value outside `0.bin` refuses the authority claim.
- Any scorer/GT/repository dependency in inflate refuses the standalone claim.
- A parity failure stops before evaluator consumption.
- A local advisory row cannot move the contest pointer.

## Unified-system wire-in

This landing is explicitly `research_only=true`, so it does not create a new
sensitivity map, Pareto constraint, bit allocator, or autopilot promotion edge.
Its reusable consumer is the R6 receiver-closure gate. Once a hot-swapped section
is measured, its producer owns the sensitivity/rate posterior; S4 supplies exact
archive bytes, runtime custody, parity, and evaluator receipt as the admission
surface.
