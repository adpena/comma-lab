# PR65/Henosis Stream Anatomy And C091 Transfer Candidates - 2026-05-03

## Scope

Local-only PR65/Henosis anatomy and deterministic candidate-builder slice.
No GPU jobs were dispatched. No Lightning state or `.omx/state` dispatch
records were edited.

Artifacts:

- Planner: `experiments/plan_pr65_henosis_stream_transfer.py`
- Candidate matrix: `experiments/results/pr65_henosis_stream_transfer_20260503_codex/candidate_matrix.json`
- Candidate archives/manifests: `experiments/results/pr65_henosis_stream_transfer_20260503_codex/*/{archive.zip,manifest.json}`
- Tests: `src/tac/tests/test_plan_pr65_henosis_stream_transfer.py`

## Source Anatomy

PR65/Henosis archive custody:

- Archive bytes: `284425`
- Archive SHA-256: `b331cb4f6df9d8929db966b943b8c73624cdf3b6db71acbde361570852e59e68`
- Strict ZIP shape: one stored member `x`, central/local names match
- `x` bytes: `284325`; ZIP overhead: `100`
- Henosis `x` layout: 30-byte 24-bit length header, then mask/model/pose/post/shift/frac/frac2/frac3/bias/region, then `randmulti`
- Core encoded bytes: `278033`
- PR65 qpost-family encoded bytes: `6262`
- PR65 mask encoded SHA matches C091/C089 mask SHA: `d1ae0d39c848e5715b74bb0122269066a4a1ab60ba9e12f34e70fd62ac136d87`

C091 PR75/minp anchor:

- Archive bytes/SHA: `276481`, `03a2afd5fe92c93a9b7b7e43625158a73b455f0cfbca82d278008a728db78746`
- Exact T4 score: `0.31516575028285976`
- PoseNet/SegNet: `0.00049371`, `0.00060804`
- Encoded streams: mask `219472`, renderer `55756`, actions `255` SG2/fixed, pose `898`

C089 P6 action source:

- Archive bytes/SHA: `276342`, `0ec53e5b871149ed6eea56c0b9bcca3baec998d5bfad4f371979e0c90e62cea8`
- Encoded streams: mask `219472`, renderer `55965`, actions `116` P6 delta-varint, pose `677`

## Break-Even Math

Using the user-supplied PR65 official fields:

- PR65 official PoseNet/SegNet/bytes: `0.00049291`, `0.00060138`, `284425`
- Recomputed PR65 score from those fields: `0.3197324821933817`
- PR65 component gain vs C091: `0.0007229508312516508`
- PR65 rate penalty vs C091: `0.00528958352360253`
- Full PR65 archive is therefore `+0.004566731910521926` worse than C091 despite the better aggregate component basin.

Interpretation: PR65 is useful as a component-basin signal only if a small
charged subset transfers. Whole-archive PR65 bytes cannot be justified.

## Candidate Matrix

All rows below are local byte/planning artifacts only (`score_claim=false`).
Break-even is against C091 exact score `0.31516575028285976`.

| Candidate | bytes | SHA-256 | delta bytes vs C091 | rate delta | component gain needed for sub-0.314 | trace positive combined bound | guard |
|---|---:|---|---:|---:|---:|---:|---|
| `c091_renderer_pose_c089_actions_p6_control` | 276353 | `c5d2d83cdfc128cafe4ae59278ba872cc9ef423d5dd32a6ab31b95648343439a` | -128 | -0.00008522994599963794 | 0.0010805203368601185 | 0 | exact already exists; not new |
| `c091_pr65_pose_qp1_c089_actions_p6` | 276346 | `d3913ec75bd1917f16f2bca5e672313f66f81da5446405fc8eccbba757eed79d` | -135 | -0.00008989095867149314 | 0.0010758593241882634 | 0 | possible future pose-only screen |
| `c091_pr65_bias_segadv_top032` | 276669 | `ae2da0835959064e3ba35eba51d65c43fb9e8df162ea74c2f52653864cc0b8e9` | +188 | +0.00012518148318696823 | 0.0012909317660467247 | 0.00016224682170900764 | non-dispatchable qpost |
| `c091_pr65_bias_combined_top064` | 276713 | `be1ac7fd5364fc16232cdc7ea94c4bb429adf4848ae2402e762616d52fa79832` | +232 | +0.00015447927712434377 | 0.0013202295599841003 | 0.002608738086984667 | non-dispatchable qpost |
| `c091_pr65_bias_region_segadv_top032` | 276713 | `b2361bf7b322d0d98107ef3f74f7b0e537e2bfef4884ec0c77a6c6fc24c11789` | +232 | +0.00015447927712434377 | 0.0013202295599841003 | 0.00016224682170900764 | non-dispatchable non-bias qpost |
| `c091_pr65_post_bias_segadv_top016` | 276797 | `72a47f3e7cad156e9321f846b49a7c34d9269c8a3ce3f7f15eb4f73e5502392e` | +316 | +0.00021041142918660615 | 0.0013761617120463627 | 0.0000615575630176132 | non-dispatchable post qpost |
| `c091_pr65_pose_qp1_c089_actions_p6_bias_segadv_top032` | 276534 | `2c1b939be22d53cc5238307b4cd90c75fb7e778a3a92c22e71bc84b8681fcdb8` | +53 | +0.00003529052451547508 | 0.0012010408073752316 | 0.00016224682170900764 | non-dispatchable qpost |

Important trace caveat:

- The available PR65 component trace used for pair ranking is
  `experiments/results/vast_harvest/public_external_component_trace_20260502T0642Z/pr65_torch25_compat_adapter/component_trace.json`.
- It does not show positive SegNet transfer against C091 for the top
  SegNet-ranked qpost candidates (`selected_positive_pair_count=0`).
- The aggregate user-supplied official PR65 fields do show a better SegNet
  basin. This disagreement makes qpost pair selections diagnostic only.

## Exact-Eval Recommendations

Do not dispatch any qpost candidate by default. Existing exact T4 qpost screens
near this family are negative:

- `exact_eval_pr65_qpost_bias_poseadv_top032_t4_20260503T0756Z`: score `0.31562772378789217`
- `exact_eval_pr65_qpost_bias_poseadv_top064_t4_20260503T0759Z`: score `0.3156849465853906`
- `exact_eval_pr75_qpost_microstack_bias032_t4_20260503T1050Z`: score `0.3156217237878922`

The only plausible future exact screen from this slice is
`c091_pr65_pose_qp1_c089_actions_p6`, and only after an explicit lane claim.
Its closest exact control is already known:

- `c091_renderer_pose_c089_actions_p6_control`
- SHA `c5d2d83cdfc128cafe4ae59278ba872cc9ef423d5dd32a6ab31b95648343439a`
- Existing exact T4 result `exact_eval_pr75_minp_p6_public_renderer_pose_t4_20260503T1108Z`
- Score `0.3153126722810012`, PoseNet `0.0004936`, SegNet `0.00061044`

Therefore the PR65 pose candidate must improve that control by about
`0.000142` score just to beat C091, and by about `0.001308` to reach sub-0.314
from the control basin. This is possible only as a PoseNet/SegNet interaction
hypothesis, not as a byte-only win.

## Non-Dispatchable Guards

- `score_claim=false` on every generated manifest.
- `promotion_eligible=false` on every generated manifest.
- `remote_dispatch.dispatched=false` and `lightning_state_touched=false`.
- Component traces are used only for atom ranking; they are not promotable
  score evidence.
- Any future exact eval must first claim a lane via
  `tools/claim_lane_dispatch.py claim ...`.
- Any future score claim must use exact CUDA auth eval:
  `archive.zip -> inflate.sh -> upstream/evaluate.py`.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_plan_pr65_henosis_stream_transfer.py -q`
- `.venv/bin/python experiments/plan_pr65_henosis_stream_transfer.py --output-dir experiments/results/pr65_henosis_stream_transfer_20260503_codex`
