# Codex findings — DDM CC3 mixed-coder receiver integration

Date: 2026-07-25
Axis: `[macOS-CPU frozen-scorer advisory]`
Verdict: `RECEIVER_CLOSED_LOSSLESS_RATE_GAIN_MEASURED;CC2_REUSED_SCORE_PREMISE_FALSIFIED_INSTANCE`

## Receiver-closed result

CC2's price table is now a real counted archive. The exact source
composition (`139538 B`) becomes `136116 B`, a measured `-3422 B` delta
with zero archive-level integration overhead. The candidate contains
exactly one G4 context frame and seven Bellard KT-mixing frames. The other
19 physical leaves remain byte-identical.

Restoration checks every selected frame's exact final-byte consumption,
raw length, SHA-256, and deterministic re-encode; preserves every nested
ZIP member and the W_joint trailing receiver suffix; restores all 27
physical leaves; and requires exact PC1 parse/re-emit.

CC3 also freshly rebuilt all five CC2 coder frames for each current leaf:
`135/135` parse backs passed and reproduced the settled `-3422 B` table.
The replay core SHA-256 is
`b392dab369ec2a257a29a1ed9beeb59c4773e35df2af7a76eeb6a4e38538a107`.

## Full receiver proof

The candidate ran through the declared locked venv and the existing
E3/E4/E5 launcher in 489.727193 seconds. All 19 runtime stages were
preserved. An independent raw-leaf control rendered from the original
source composition also preserved 19 stages. Both final outputs are
3,662,409,600 bytes with SHA-256
`5094e277dc4c736ad1ab50aead9f49630319bb6e3d42c48e9777fbdd09c215f3`.

Thus the integration itself has exactly zero distortion delta. The rate
costate is `-0.0022785693375840703`, and the preregistered overhead
falsifier passes.

## Premise falsification

Fresh n600 batch32 scoring of the full composition produced
`d_seg=0.024731920030381944` and `d_pose=163.0492342914382`.
CC2 had reused `d_seg=0.0702156745062934` and
`d_pose=36.37587755493872` from the parent-only endpoint under
`IDENTITY_ACTIVE_ALL_QUANTIZED_COORDINATES_ZERO`.

Those values are not equal. That scorer-reuse premise is falsified for
this exact active-zero PC1 instance. This does not weaken the CC3
losslessness result: the candidate and unrecoded composition remain
byte-identical at the full receiver output. It also does not close the PC1
family.

## Consumer guidance

- R6 consumers may inherit the measured `-3422 B` only when using this
  exact receiver-closed recursive transform or remeasuring another stream
  shape.
- LP1's coordinated post-integration budget is `130789 B`, labeled
  `DERIVED_COORDINATED_BUDGET`, not a new physical archive measurement.
- Consumers must not reuse CC2's parent-only d_seg/d_pose as the active-PC1
  composition endpoint.
- Pointer `0.1910828242 [contest-CPU]` remains unchanged.

Full SSD receipt:
`/Volumes/VertigoDataTier/pact/experiments/results/ddm_cc3_mixed_coder_receiver_integration_20260725T041134Z/ddm_cc3_mixed_coder_receiver_receipt.json`
SHA-256:
`f821c63f82183abd947635d88e04eb660742b0ab60dff8b1eb27d6f9f5a52b82`.

MAIN review is required before landing.
