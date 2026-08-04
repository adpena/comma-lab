---
arm: ddm_sg4
task: 920
axis: "[macOS-CPU frozen-scorer advisory]"
score_claim: false
promotion_eligible: false
pointer: "0.1910828242 [contest-CPU] UNMOVED"
own_vehicle_frontier: "S = 0.7541459 @ 358,084 B [macOS-CPU advisory] n600"
---

# ddm_sg4 -- SEG-axis protection primitives

## HEADLINE

**Road<->Lane phase has a real seg leg, but this receiver is pose-blocked.** On the current fz2
vehicle, the legal deterministic block16 RGB-translation receiver recovers 48,504 SegNet flips
at n600 (`eta_det = 0.2279366`), enough to beat the measured 57,809-byte LZMA1 offset price by
`-0.0026247 S` on seg+rate alone. The same receiver raises mean d_pose from `0.00071459` to
`0.66376308`, adding `+2.4918265 S` in the pose term. The combined diagnostic row is therefore
`+2.4892017 S` worse than the base under LZMA1 bytes. It is not adoptable and cannot move the
pointer.

**Lane x ANNIHILATE was not newly scored here.** The existence hinge is built and wired, but it is
a trainer-side A/B, not a post-hoc receiver pass. Firing it requires a governed TR1 launch ticket
with a matched pose-carrying control; a seg-only or unsealed shortcut would be a false mechanism.

## Evidence

Artifacts are on the SSD tier:

| artifact | sha256 |
|---|---:|
| `/Volumes/VertigoDataTier/pact/ddm_sg4_20260804/sg4_phase_rgb_translate_n600_aggregate.json` | `1d61026d143c6eef9633d3fe602661e72d26741fe4f9e015ee754f6414ee916a` |
| `/Volumes/VertigoDataTier/pact/ddm_sg4_20260804/sg4_phase_pose_damage_n600_aggregate.json` | `99dbc197b52f7dae66e5c2f4f74e7d58ec4280fe1358f175343537fc317fb577` |
| `/Volumes/VertigoDataTier/pact/ddm_et1_20260803/et1_ph1_block16_our_vehicle.json` | `8b9da08d420bf2b6bb37d4a97bbac5ab27d9ed029385be92f82bc34f5db4a1c1` |
| `/Volumes/VertigoDataTier/pact/ddm_fz1_20260804/rowB/sub_final/fz2_byteclose_receipt.json` | `d7285e918b359597f57f613c543f8d4a615aebf5680577fc26122588e6c097ce` |

The run used 5 chunks of 120 pairs, respecting the scorer-slot chunk limit. The receiver was:
deterministic scorer-lattice block16 translation of the rendered frame_1 RGB, upsampled to camera,
rounded to uint8, then re-scored by the frozen CPU SegNet/PoseNet. No MPS, no upstream mutation,
no persisted `/tmp` evidence.

## Phase Numbers

| quantity | value |
|---|---:|
| baseline flips | 508,640 |
| label-space described flips | 212,796 |
| label-space seg ceiling | `0.1803894 S` |
| deterministic RGB gain | 48,504 flips |
| population `eta_det` | `0.2279366` |
| deterministic seg gain | `0.04111735 S` |
| negative-eta pairs | 89 / 600 |
| LZMA1 offset bytes | 57,809 B |
| SMEVR projected bytes | 46,247 B |
| break-even eta, LZMA1 | `0.2133864` |
| break-even eta, projected | `0.1707084` |
| seg+rate net, LZMA1 | `-0.0026247 S` |
| seg+rate net, projected | `-0.0103234 S` |

Chunk etas: `0.1564`, `0.2395`, `0.2967`, `0.3193`, `0.1544`. This is a population-mixed
receiver, not a uniform transform.

## Pose Gate

| quantity | value |
|---|---:|
| mean d_pose before | `0.0007145917` |
| mean d_pose after | `0.6637630764` |
| pose score before | `0.08453353` |
| pose score after | `2.57635998` |
| pose score delta | `+2.49182646 S` |
| median per-pair pose ratio | `447.43x` |
| max per-pair pose ratio | `87201.02x` |
| pose-improved pairs | 2 / 600 |
| combined diagnostic score, LZMA1 bytes | `3.24334818` |
| delta vs base, LZMA1 bytes | `+2.48920175 S` |

Verdict scope: **FORMULATION-CLOSED for deterministic block16 RGB translation of rendered frame_1
as a standalone receiver on this fz2 vehicle.** This does not kill Road<->Lane positional DOF:
the seg mass is real. It says the direct deterministic RGB translation buys that mass by breaking
PoseNet correspondence, so any successor must be joint pose-preserving, frame0-compensated,
selective/gated, or a different renderer.

## Lane x ANNIHILATE

No new scorer A/B was run for the existence hinge in sg4. The relevant already-built surface is
`--existence-hinge-weight` with `--existence-hinge-classes lane` and `lane,movable`, documented in
`ddm_p4x_lane_existence_birth_matrix_20260803.md`. The fz2 base is pose-carrying, so the prior
blocker is plausibly cleared, but the primitive needs a governed TR1 A/B, not this receiver-only
scorer pass.

Fire order:

1. `control {}` versus `A_lane_only --existence-hinge-weight 0.1 --existence-hinge-classes lane`
   on a seeded random sample with matched pose control.
2. If A is live, run `B_birth_matrix --existence-hinge-weight 0.1 --existence-hinge-classes lane,movable`.
3. Then sweep weight and grammar/weight-policy; do not call a null without the `at_risk` telemetry.

Typed outcome until that fires: **QUEUED-WITH-FIRE-ORDER / NOT-MEASURED**, not a negative.

## Frontier

Own-vehicle frontier remains `S = 0.7541459 @ 358,084 B [macOS-CPU advisory] n600`. Contest pointer
remains `0.1910828242 [contest-CPU] UNMOVED`. The sg4 phase row is diagnostic only:
`score_claim=false`, `promotion_eligible=false`.
