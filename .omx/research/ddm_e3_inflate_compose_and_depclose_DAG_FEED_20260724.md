# DDM E3 inflate composition and dependency closure — DAG FEED — 2026-07-24

`research_only=true` · `score_claim=false` ·
`[macOS-CPU frozen-scorer advisory]` · pointer unchanged.

## Closed feed edge

`E2 archive 8891012e… / raw 4871b1c1…`
→ stdlib raw-LZMA1 decode
→ preserved 16-pair base stages
→ literal PoseNet resize + YUV6 moment reduction
→ scorer-weight-only target constants
→ decode-derived float32 gain/bias
→ frame-0-only camera residual
→ preserved corrected stages
→ `0.raw 4c553508…`
→ locked upstream `evaluate.sh`
→ `{d_pose=147.49104309, d_seg=0.02861482, archive=439303}`.

The authoritative transfer edge is stronger than printed metric precision:
all 38 source/corrected camera batch hashes match the PA1 checkpoints, so the
full-precision advisory row is `147.49104204339514 /
0.02861480712890625`. Frame 1 remains byte-identical.

## Rate homes

- COUNTED: exact archive 439,303 bytes, SHA
  `dd8fc5fed6ff11e532765dfe6104f02b3b97171b824123312a3ab469c1be6cbe`.
- FREE: `ddm_pa1_scorer_only_bn_inverse_frame0_receiver_v1`, zero payload
  bytes. Constants derive only from scorer weights/BN; dynamic affine derives
  from decoded counted content.
- NULL: D2 and D5, zero bytes and no receiver effect.
- Runtime Python is not counted; dependency inventory is Torch plus stdlib.

## Consumers

`menu1:pose_amplitude` may replace the prior
`COMPOSITION_OWED` marker with `RECEIVER_SURVIVAL_PASS_ADVISORY`. It must not
promote or infer contest-CPU/CUDA authority. The next admissible action requires
new operator authority for a contest-axis replay; no such dispatch occurred.

## Triality

- DSL: typed E3 exporter and harness configs seal paths, hashes, storage floor,
  local-only execution, and false-authority flags.
- DAG: the closed feed edge above, with base/moment/corrected stage persistence.
- Equations: `ddm_e3_pa1_receiver_survival_v1`,
  `ddm_pa1_free_null_counted_target_partition_v1`, and
  `ddm_runtime_export_identity_receiver_closed_v1`.

Canonical receipt:
`.omx/research/ddm_e3_inflate_compose_and_depclose_20260723/ddm_e3_receiver_survival_receipt.json`.

MAIN landing review is required.
