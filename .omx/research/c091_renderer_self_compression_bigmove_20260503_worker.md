# C091 Renderer Self-Compression Big-Move Screen - 2026-05-03 Worker

Scope: local-only C091-native renderer/QZS3 byte-shrink planning. No Lightning,
Modal, Vast.ai, T4, or other remote GPU dispatch was performed. No dispatch
claim was opened.

## Anchor

- Frontier: C091 PR75/minp public replay exact T4.
- Archive:
  `experiments/results/lightning_batch/exact_eval_pr75_minp_public_replay_t4_20260503T1049Z/archive.zip`
- Eval JSON:
  `experiments/results/lightning_batch/exact_eval_pr75_minp_public_replay_t4_20260503T1049Z/contest_auth_eval.adjudicated.json`
- Score: `0.31516575028285976`
- Bytes: `276481`
- SHA-256:
  `03a2afd5fe92c93a9b7b7e43625158a73b455f0cfbca82d278008a728db78746`
- Components: SegNet `0.00060804`, PoseNet `0.00049371`, `n_samples=600`,
  device `cuda`, GPU `Tesla T4`.
- Byte-only requirement for strict `<0.314` at unchanged components: `1751`
  bytes.

## Tooling

- Added `experiments/plan_c091_renderer_self_compression_bigmove.py`.
- Added `src/tac/tests/test_plan_c091_renderer_self_compression_bigmove.py`.
- Result root:
  `experiments/results/c091_renderer_self_compression_bigmove_20260503_worker/`
- Primary artifacts:
  - `plan.json`
  - `dispatch_recommendation.json`
  - `qzs3_reblock_candidates/summary.json`

The helper reuses reviewed runtime extraction and pose-safety preflight, but
does not reuse the older generic PR75 fixed-slice fallback for C091/minp. The
first attempted generic build exposed a scoped parser gap: the older fallback
uses `model_len=56034` and `actions_len=236`, while C091/minp is
`model_len=55756` and `actions_len=255`. The C091 helper therefore uses an
explicit C091 fixed-slice parser and emits self-describing `P3` payloads with
the original mask, SG2 action, and QP1 pose streams preserved.

## Command

```bash
.venv/bin/python experiments/plan_c091_renderer_self_compression_bigmove.py \
  --output-dir experiments/results/c091_renderer_self_compression_bigmove_20260503_worker \
  --force \
  --qzs3-block-sizes 32,48,64,96,128,192,256,512 \
  --max-preflight-candidates 5 \
  --preflight-max-pairs 5
```

Focused verification:

```bash
.venv/bin/python -m py_compile \
  experiments/plan_c091_renderer_self_compression_bigmove.py \
  src/tac/tests/test_plan_c091_renderer_self_compression_bigmove.py

.venv/bin/python -m pytest \
  src/tac/tests/test_plan_c091_renderer_self_compression_bigmove.py -q
```

Result: `4 passed`.

## Candidate Screen

All rows below preserve decoded `masks.mkv`, `optimized_poses.qp1`, and
`seg_tile_actions.bin` byte-for-byte and pass local runtime unpack closure.
The five byte-sufficient rows were preflighted with sampled pairs
`[0, 150, 300, 449, 599]` and strict thresholds mean `<=0.05`, RMS `<=0.08`,
max `<=1.5`.

| candidate | bytes | delta vs C091 | SHA-256 | byte-only score | pose-safe | mean | RMS | max |
| --- | ---: | ---: | --- | ---: | --- | ---: | ---: | ---: |
| `qzs3_b0512_c091_p3_preserved_minp_slices` | `272340` | `-4141` | `7960d7abfb53750eddfc1658e183335a37d19cb639414247a1ea9a4af601efa1` | `0.3124084283579808` | false | `11.231759071350098` | `17.11295267959514` | `236.19015502929688` |
| `qzs3_b0256_c091_p3_preserved_minp_slices` | `272894` | `-3587` | `5b82f1558f3f591f8ae72e0ceb78ed05aa4b2b76b552b81adbc410f76889f9f2` | `0.3127773142180105` | false | `11.436809539794922` | `16.618909135955363` | `239.05215454101562` |
| `qzs3_b0192_c091_p3_preserved_minp_slices` | `273140` | `-3341` | `ad3accc0fd789700500b9eb1ee2a3126c011cb2db1f603333856ce16628c9d9e` | `0.3129411155204786` | false | `10.04934310913086` | `15.30472189649657` | `244.8638916015625` |
| `qzs3_b0128_c091_p3_preserved_minp_slices` | `273704` | `-2777` | `4eabd1c8b4462530a53e4a57f2f91afead41cc956d9d3ad0ff0abbc8a15c270d` | `0.3133166599700395` | false | `8.731392860412598` | `14.015757820580243` | `224.42857360839844` |
| `qzs3_b0096_c091_p3_preserved_minp_slices` | `274166` | `-2315` | `711af4b78976f9302d1d01c5ef0359faab9316f2a1df20e9e240a0170061c516` | `0.31362428680638194` | false | `8.704894065856934` | `14.295000730970163` | `231.69911193847656` |
| `qzs3_b0064_c091_p3_preserved_minp_slices` | `274840` | `-1641` | `1af9c5db6e9990e368c600c8246111e53de81aff9b72c3feb0a94b01b86c896b` | `0.3140730757407863` | not run | - | - | - |
| `qzs3_b0048_c091_p3_preserved_minp_slices` | `275562` | `-919` | `17603b13eaaffa58b6411d5b3d9ed9b49ddc035d0dd9bf36b19485a4010e54f8` | `0.31455382590494047` | not run | - | - | - |
| `qzs3_b0032_c091_p3_preserved_minp_slices` | `276492` | `+11` | `aeacdc7e31fcec319eaa6f98f424246863cc3906d4e2ca52ef9858dc9a6701a7` | `0.3151730747313441` | not run | - | - | - |

## Decision

Do not dispatch. No candidate archive is exact-eval recommended.

Reason: every C091-native renderer reblock that plausibly saves at least
`1.8KB` and crosses `<0.314` by byte-only math fails local renderer
pose-safety by a wide margin before CUDA auth eval. The largest non-preflighted
local byte saver, block size `64`, saves only `1641` bytes and remains above
`0.314` even if components were unchanged.

This is scoped negative evidence against naive C091 QZS3 block-size
self-compression as a sub-`0.314` route. It does not kill trained/fixed
renderer exports, parity-aware encoders, or a semantic renderer replacement;
those still require their own byte-closed archive and pose-safety gate.
