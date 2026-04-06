# track b: robust patched world

This is the real codec lane.

## Mainline bet

1. establish an interframe codec floor with x265
2. add task-aware pre/post processing
3. add sparse residuals only after the floor is healthy
4. keep any neural or INR work as a side bet until it clearly earns its bytes and runtime

## Why x265 first

- decode support is reliable
- packaging is simple
- the published baseline uses x265 in an intentionally weak all-intra configuration
- long-GOP search is the fastest route to an improved robust baseline

## Metric-aware implications

The published score cares about two fixed downstream models, not about human visual quality:

- PoseNet works on pairs of frames after resize and YUV-like conversion.
- SegNet works on only the last frame in each pair after resize.

That means:
- preserve downstream semantics first
- preserve motion cues second
- ignore texture that does not move the score

## Next evolutions after the starter pack

- teacher cache
- sparse ROI residual stream
- micro-restorer
- optional AV1 feasibility lane
- optional HNeRV / HiNeRV side bet
