# Codex Findings: MLX Decoder-Q Surface-Guided Waterbucket

Generated: 2026-05-22T03:56:36Z

## Verdict

PROCEED_TO_COMPONENT_RESPONSE_SMOKE for the fixed-length surface-guided decoder-q candidates.

This is not a score claim. The artifacts remain non-promotional until advisory component response and exact CUDA auth eval are measured.

## Inputs

- Response surface: `experiments/results/mlx_same_axis_decoderq_d1f1e56e042692f2_20260522T031205Z/mlx_decoderq_response_surface_plan.json`
- Waterbucket plan: `experiments/results/mlx_same_axis_decoderq_d1f1e56e042692f2_20260522T031205Z/decoder_q_surface_guided_waterbucket_plan.json`
- Candidate root: `experiments/results/mlx_same_axis_decoderq_d1f1e56e042692f2_20260522T031205Z/decoder_q_surface_guided_waterbucket_candidates`
- Inflate controls: `experiments/results/mlx_same_axis_decoderq_d1f1e56e042692f2_20260522T031205Z/decoder_q_surface_guided_inflate_controls/summary.json`
- Runtime: `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/runtime_hfv1`
- Baseline raw SHA-256: `d1afc583b01ff4a7aaa844d4f03ece3ed381d56763a06cb2c5e011526e5f868c`

## Objective

The decoder-q response surface showed candidate regressions dominating preserved gains:

- Matched windows: 300
- Preserve/improvement windows: 76
- Suppress/invert regression windows: 224
- Preserve gain sum: 0.028526397509450963
- Suppress harm sum: 0.1535767647957981
- Harm/gain ratio: 5.383671904064199
- Dominant axis: `seg` (216 seg-dominant windows, 84 pose-dominant windows)

The new objective therefore ranks decoder-q atoms as `suppress_or_invert_regressions_first`, with a seg-dominant proxy used only for planning. The helper explicitly carries `score_claim=false`, `promotion_eligible=false`, `ready_for_exact_eval_dispatch=false`, and `rank_or_kill_eligible=false`.

## Waterbucket Output

Plan summary:

- Candidates: 25
- Fixed-length candidates: 16
- Non-fixed candidates: 9
- Response-surface-ranked atom count: 75
- Surface-guided fixed candidates selected for official inflate controls: 3

The surface-guided fixed candidates were:

| candidate_id | edit budget | archive bytes | surface proxy priority |
| --- | ---: | ---: | ---: |
| `a2f90a216aac4184` | 1 | 178517 | 0.47923087385670265 |
| `a9b04920db67ec71` | 2 | 178517 | 0.9584617477134053 |
| `8f3a33e49b9b7906` | 4 | 178517 | 1.9169234954268106 |

## Official Inflate Controls

All three fixed-length surface-guided candidates pass the official inflate/raw visibility control:

| candidate_id | member SHA-256 | raw SHA-256 | changed frames | changed bytes | byte L1 total |
| --- | --- | --- | ---: | ---: | ---: |
| `a2f90a216aac4184` | `c4b353b9c6667ebae47729f1e0dad7bcbea9535e8f2bd487da367e2ce9d9c106` | `387571bfc54d33939e4db50d1e53f69555711cf122823e1a35a7dfbab5644df5` | 600 | 35132514 | 35132514 |
| `a9b04920db67ec71` | `c2130a9797ed25b14867439356f7aae4c2e1423879a9edc02070e13d82640b74` | `959f5fa43ed48e40191488274b7c2f37d76a27128b6409859352e91d823c8459` | 600 | 46453851 | 46453851 |
| `8f3a33e49b9b7906` | `d6564ab441310ad89a5d5458714bec9a01a4b6ae5c940d7e995bd4956d0eb29e` | `b23798df89aa4b7d7e67f94b641ababec3f7221bfc0595cf4221e15b9dabec55` | 600 | 62258822 | 62287265 |

Control summary:

- Candidate count: 3
- Visible-change count: 3
- No-visible-change count: 0
- Candidate raw outputs were deleted after SHA capture to conserve disk.

## Authority Boundaries

Remaining blockers:

- `advisory_component_response_not_measured`
- `exact_cuda_auth_eval_missing`

This lane should not enter promotion, ranking, or kill decisions until at least a small component-response smoke measures SegNet/PoseNet deltas on the three fixed candidates. If that smoke keeps the suppress-first hypothesis alive, dispatch only the best candidate to exact CUDA auth eval under the normal lane-claim lifecycle.
