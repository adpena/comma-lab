# Renderer Parity Shrink Search - Worker - 2026-05-03

Scope: local renderer-byte shrink search only. No Lightning, Modal, Vast.ai, or
other remote dispatch was performed. No dispatch claim was opened because this
work did not launch training/eval/remote GPU jobs.

## Source Custody

- Source archive:
  `experiments/results/lightning_batch/exact_eval_c067_pr75_qp1_lag_eval_top67_p6_t4_20260503T0608Z/archive.zip`
- Source bytes: `276352`
- Source SHA-256:
  `d73f0957cf2d8da0526c1786613443331ccb913f5648131cfaaac5e6f7eae972`
- Source exact evidence:
  `experiments/results/lightning_batch/exact_eval_c067_pr75_qp1_lag_eval_top67_p6_t4_20260503T0608Z/contest_auth_eval.json`
- Exact T4 score: `0.3154979650614253`
- Components: SegNet `0.00061045`, PoseNet `0.0004962`, archive bytes
  `276352`, `n_samples=600`

Byte-only target math: to cross `0.314` with unchanged components, the archive
must save at least `2250` bytes. Smaller safe candidates are useful boundary
signal but not dispatch candidates by themselves.

## Tooling Added

- `experiments/search_renderer_parity_shrink_candidate.py`
  - Preserves PR75 `p` slices and every non-renderer logical member.
  - Applies structured QZS3 FP4 threshold-zero transforms by prefix.
  - Rebuilds deterministic single-member archives.
  - Runs `experiments/preflight_renderer_transplant_pose_safety.py` locally.
  - Fails closed: `score_claim=false`, `promotion_eligible=false`, and no
    remote dispatch command is emitted.
- `src/tac/tests/test_search_renderer_parity_shrink_candidate.py`
  - Covers transform parsing, targeted FP4 zeroing, PR75 non-renderer
    preservation, and fail-closed behavior when preflight is skipped.

Implementation note: a local MQZ1/per-prefix block-size variant was considered
but was removed from the default search. The current PR75 unpacker validates
the decoded renderer slice as `QZS3`; an MQZ1 renderer inside this PR75 payload
is therefore not contest-faithful without a runtime change.

## Search Command

```bash
.venv/bin/python experiments/search_renderer_parity_shrink_candidate.py \
  --output-dir experiments/results/renderer_parity_shrink_search_20260503_worker \
  --force \
  --max-preflight-candidates 30 \
  --preflight-max-pairs 5 \
  --transform zero-fp4-prefix:all_fp4:0.03 \
  --transform zero-fp4-prefix:all_fp4:0.04 \
  --transform zero-fp4-prefix:all_fp4:0.05 \
  --transform zero-fp4-prefix:all_fp4:0.075 \
  --transform zero-fp4-prefix:all_fp4:0.1 \
  --transform zero-fp4-prefix:shared_trunk:0.03 \
  --transform zero-fp4-prefix:shared_trunk:0.04 \
  --transform zero-fp4-prefix:shared_trunk:0.05 \
  --transform zero-fp4-prefix:shared_trunk:0.075 \
  --transform zero-fp4-prefix:shared_trunk:0.1 \
  --transform zero-fp4-prefix:frame2_head:0.03 \
  --transform zero-fp4-prefix:frame2_head:0.04 \
  --transform zero-fp4-prefix:frame2_head:0.05 \
  --transform zero-fp4-prefix:frame2_head:0.075 \
  --transform zero-fp4-prefix:frame2_head:0.1 \
  --transform zero-fp4-prefix:frame2_head.block2:0.03 \
  --transform zero-fp4-prefix:frame2_head.block2:0.04 \
  --transform zero-fp4-prefix:frame2_head.block2:0.05 \
  --transform zero-fp4-prefix:frame2_head.block2:0.075 \
  --transform zero-fp4-prefix:frame2_head.block2:0.1 \
  --transform zero-fp4-prefix:frame2_head.pre:0.03 \
  --transform zero-fp4-prefix:frame2_head.pre:0.04 \
  --transform zero-fp4-prefix:frame2_head.pre:0.05 \
  --transform zero-fp4-prefix:frame2_head.pre:0.075 \
  --transform zero-fp4-prefix:frame2_head.pre:0.1 \
  --transform zero-fp4-prefix:frame1_head:0.05 \
  --transform zero-fp4-prefix:frame1_head:0.075 \
  --transform zero-fp4-prefix:frame1_head:0.1
```

Artifacts:

- `experiments/results/renderer_parity_shrink_search_20260503_worker/summary.json`
- `experiments/results/renderer_parity_shrink_search_20260503_worker/dispatch_recommendation.json`
- Per-candidate `archive.zip`, `build_manifest.json`, and
  `pose_safety_preflight.json` where a byte-saving candidate was preflighted.

Preflight thresholds: `max_pairs=5`, sampled pairs `[0, 150, 300, 449, 599]`,
`mean_abs_delta <= 3.0`, `rms_delta <= 8.0`, `max_abs_delta <= 80.0`.

## Candidate Results

Candidates that would cross `0.314` by byte-only rate math all failed local
pose-safety:

| Candidate | Bytes | Delta | SHA-256 | Byte-only score | mean_abs | rms | max_abs | Safe |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| `zero_fp4_all_fp4_0.1` | `272069` | `-4283` | `623e764ae1feeb11aba31d3be8c825ecf4e74948c8a5f5bb756ac92e8f91905a` | `0.31264609116520303` | `8.58510971069336` | `13.47567592192445` | `231.69529724121094` | false |
| `zero_fp4_all_fp4_0.075` | `273722` | `-2630` | `8ae527985b36f92c8d0e409844fb8a4901662ce8dfbb65497349e5e9f23942d4` | `0.31374675601471397` | `7.168813228607178` | `11.267663609865963` | `197.95152282714844` | false |
| `zero_fp4_shared_trunk_0.1` | `273951` | `-2401` | `9cf86ac92f3d7a97190a0abbb86b7d65277e3333b6f168ff547cd934c38c7ce9` | `0.31389923771497896` | `6.819228649139404` | `11.356843312058524` | `224.2159881591797` | false |

Largest pose-safe byte saver:

| Candidate | Bytes | Delta | SHA-256 | Byte-only score | mean_abs | rms | max_abs | Safe |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| `zero_fp4_frame1_head_0.1` | `275900` | `-452` | `bb8d1835303478183465e15b28fd9af8776b4ea1c07cdd6cfe0895032a267b64` | `0.3151969968146141` | `1.236077904701233` | `2.478089352287593` | `45.092079162597656` | true |

Other safe boundary signals:

- `zero_fp4_frame2_head_0.05`: `276222` bytes, delta `-130`,
  SHA-256 `efccff1fbfb5f9dcc38c05a79e1ab5ebe2f90d13fcae5fdc0e0f8465e79b6ed6`,
  mean_abs `0.9484355449676514`, rms `2.0316283827260455`, max_abs
  `75.78392028808594`.
- `zero_fp4_shared_trunk_0.04`: `276282` bytes, delta `-70`,
  SHA-256 `074c38d67e1ba927800c8ccdd735b027045fe60401e724c702ea6c9d4ba0f62d`,
  mean_abs `0.9723898768424988`, rms `1.5254677571077875`, max_abs
  `72.1895980834961`.
- `zero_fp4_all_fp4_0.03`: `276300` bytes, delta `-52`,
  SHA-256 `30e8c2051c977742cbd617aea990400cec53d2c502fd12a515590ad25faef044`,
  mean_abs `0.337311714887619`, rms `0.6211091285577061`, max_abs
  `24.794830322265625`.

## Decision

Do not dispatch any candidate from this search. The only candidates large
enough to plausibly move the exact T4 frontier below `0.314` failed local
renderer output parity before exact CUDA auth eval. The largest safe candidate
saves only `452` bytes and has a byte-only projected score
`0.3151969968146141`, which does not cross the target.

Dispatch recommendation artifact:
`experiments/results/renderer_parity_shrink_search_20260503_worker/dispatch_recommendation.json`
reports `do_not_dispatch_yet_safe_but_too_small`.

## Verification

```bash
.venv/bin/python -m pytest src/tac/tests/test_search_renderer_parity_shrink_candidate.py
```

Result: `3 passed`.
