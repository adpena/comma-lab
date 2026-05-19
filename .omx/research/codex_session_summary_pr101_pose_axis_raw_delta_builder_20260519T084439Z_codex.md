# Codex Session Summary: PR101 Pose-Axis Raw-Delta Builder

Date: 2026-05-19T08:44:39Z

Codex continued OP-7 from packet-mechanics closure to the first component-moving
PR101 local candidate mode. The default builder still emits score-neutral
same-length Brotli recompressions. The new opt-in `raw_byte_delta` mode mutates
one decompressed decoder-stream byte, recompresses the stream to the exact same
compressed length, rebuilds the monolithic archive, and preserves the no-score
authority fence.

Local artifact:

- `experiments/results/pr101_pose_axis_operator_candidate_raw_delta_20260519T084439Z_codex/archive.zip`
- archive SHA-256:
  `30826b37093ee3af9512a1b46bd0b569fecbc4ccf75b8ff2dd746de113a5144a`
- runtime-consumption proof SHA-256:
  `3f5899da99c47e0c2987634a050721cf92dbf94b621c023e4732c9d24426550d`
- raw mutation: split-Brotli stream `2`, raw offset `33803`, value `19 -> 18`
- archive byte delta: `0`
- `score_claim=false`, `promotion_eligible=false`,
  `ready_for_exact_eval_dispatch=false`

Next best OP-7 closure is a score-response harness for this raw-delta candidate:
inflate/output proof first, then paired component deltas, then exact-eval only
behind a lane claim if the response matrix is worth dispatch.

