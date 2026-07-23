# DDM E2 compact-pose closure premise falsification — 2026-07-23

`research_only=true` · `score_claim=false` · pointer unchanged · MAIN landing review required.

## Premise tested

The delegated brief proposed that the 3,600-byte `receiver.pose6_codes` object was a small solved pose stream that only needed to be wired through the exporter.

## Fresh result

That production premise is **FALSIFIED at the E1/E2 packet boundary**:

- The counted E1 packet contains only `manifest.json`, `base/chart.ddb`, and `semantic/composed.dds`.
- `receiver.pose6_codes` is a `uint8[600,6]` nested-state target with SHA-256 `d23df2fa23f2aa6ef3cce63e5ab87f29b4de9c19687c9dd3fbeb645dc0b6f25e`.
- It is consumed by the pre-export inter-pair worldsheet construction and then stripped. There is no counted pose member.
- #417 counted-pose→output is therefore not applicable at zero pose bytes; output-pose-effect→single-owner fails because no compact code-to-photometry inverse exists.

The correct classification is `ABSENT_FROM_COMPOSED_PACKET`, not `COUNTED_BUT_INERT`.

## What remains true

The exact two-plane lattice control remains a valid feasibility witness (`d_pose_n64=0.000060022091887905524`), but its archive is 409,526,925 bytes. It proves the family is open; it does not supply the missing compact inverse. The bounded E2 frame-home repair improves `d_pose` from `163.05291748` to `162.58094788`, not to the requested tube.

## Verdict scope

Negative only for the current compact E1/E2 composed-export formulation. It is not a negative on DDM, the exact solve, ξ factorization, or future learned/analytic compact pose inverses.
