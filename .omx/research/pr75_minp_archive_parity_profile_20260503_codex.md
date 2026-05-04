# PR75/minp Archive Grammar And Local Parity Profile - 2026-05-03

Status: local forensic support only. No remote dispatch. No score claim.

## Inputs

- Public archive:
  `experiments/results/top_submission_reverse_engineering_20260503_pr75_minp/archive.zip`
- Public archive bytes: `276481`
- Public archive SHA-256:
  `03a2afd5fe92c93a9b7b7e43625158a73b455f0cfbca82d278008a728db78746`
- PR75 source clone:
  `/tmp/pr75-minp`, branch `submission/qpose14-r55-segactions-minp`,
  commit `deeef56eeb915ddb3f20ee22fd947002606a8b9c`
- Public inflate source SHA-256:
  `ea490dec83e36788dd64913650d1c265dbef1386320a5f550b5df063a71838cc`
- Comparison archive:
  `experiments/results/lightning_batch/exact_eval_c067_pr75_qp1_top40_p6_t4_awsfix1_20260503T0630Z/archive.zip`
- Comparison archive bytes: `276342`
- Comparison archive SHA-256:
  `0ec53e5b871149ed6eea56c0b9bcca3baec998d5bfad4f371979e0c90e62cea8`

## Artifact Outputs

- Grammar profile:
  `experiments/results/top_submission_reverse_engineering_20260503_pr75_minp/pr75_minp_grammar_profile.json`
- Skip-render parity:
  `experiments/results/top_submission_reverse_engineering_20260503_pr75_minp/raw_output_parity_skip_render/pr75_raw_output_parity.json`
- Selected-pair CPU parity:
  `experiments/results/top_submission_reverse_engineering_20260503_pr75_minp/raw_output_parity_pairs_33_104_598_cpu/pr75_raw_output_parity.json`

## Fixed-Slice Grammar

The current PR75/minp archive is a strict single-member stored ZIP with member
`p`. The inner payload is `276381` bytes and splits as:

| stream | charged bytes | decoded bytes | decoded SHA-256 |
|---|---:|---:|---|
| `masks.mkv.br` | `219472` | `223385` | `a5c2b89c110d75220cd09b2f27f2e92844626ae7ed0d2c797290dcf43c7068eb` |
| `renderer.bin.br` | `55756` | `59288` | `30159b6ace27a4013d1516c340d58f6d683e6847429fd3d6303a2c650aa2abef` |
| `seg_tile_actions.br` | `255` | `281` SG2 wire bytes / `432` runtime bytes | runtime `5af557cdf4c8c4c3747b06c1daabfe34581b62cb9f317d41593b836c6727427a` |
| `optimized_poses.qp1.br` | `898` | `1140` | `8d77f6a39d1a84eca78fbe8fa5ddc31f6ada29814b49e32daeb800ea84a015cc` |

Current `submissions/robust_current/unpack_renderer_payload.py` parses this
payload in the live tree and reports
`public_pr75_qzs3_qp1_segactions_fixed_slices`. The SG2 wire stream is
converted to ordinary runtime raw4 action records before render-time use.

## Action Delta

Public PR75/minp actions:

- wire kind: `SG2_grouped_tile_frame_delta_varint`
- SG2 raw bytes: `281`
- charged Brotli bytes: `255`
- runtime records: `108`
- runtime bytes: `432`
- unique pairs: `106`
- pair range: `33..598`
- unique tiles: `21`
- unique actions: `60`

C089 comparison actions:

- wire/runtime kind: `public_pr75_qzs3_qp1_segactions_p6_delta_varint`
- runtime records: `40`
- runtime bytes: `160`
- runtime SHA-256:
  `50e997db4fd1b7642d9a2cd67b55802a4f7f9bd7a6603d0ca2eaf42e7fe7e320`

The mask stream is byte-identical after decode between public PR75/minp and
C089. Renderer, QP1 pose, and tile-action streams differ. The public archive
is `+139` bytes versus C089 but carries `+68` action records, a different
renderer, and a different QP1 pose stream.

## Local Runtime Parity

Skip-render parity confirms public decoded streams and robust unpacked streams
agree on masks, renderer, pose, and runtime action records for the public
archive. Selected CPU render parity on action-bearing pairs `33`, `104`, and
`598` is exact:

- native pair tensors after actions: exact equal
- selected raw RGB after actions: exact equal
- robust-current pose path equals public QP1 float32 pose path

The all-600 CPU parity pass was launched with `--all-pairs --chunk-size 16
--fast-fail` and stopped after about seven minutes with no mismatch output and
no completed JSON. This is an interrupted local guard, not negative evidence.
Selected-pair parity remains the completed local semantic proof in this slice.

## Commands

```bash
.venv/bin/python -m py_compile \
  experiments/profile_pr75_minp_archive.py \
  experiments/pr75_raw_output_parity.py \
  src/tac/tests/test_profile_pr75_minp_archive.py \
  src/tac/tests/test_pr75_raw_output_parity.py

.venv/bin/python -m pytest \
  src/tac/tests/test_profile_pr75_minp_archive.py \
  src/tac/tests/test_pr75_raw_output_parity.py \
  src/tac/tests/test_unpack_renderer_payload_fixedslice.py -q

.venv/bin/python experiments/profile_pr75_minp_archive.py \
  --archive experiments/results/top_submission_reverse_engineering_20260503_pr75_minp/archive.zip \
  --compare-archive experiments/results/lightning_batch/exact_eval_c067_pr75_qp1_top40_p6_t4_awsfix1_20260503T0630Z/archive.zip \
  --pr75-source-root /tmp/pr75-minp \
  --output-json experiments/results/top_submission_reverse_engineering_20260503_pr75_minp/pr75_minp_grammar_profile.json

.venv/bin/python experiments/pr75_raw_output_parity.py \
  --public-archive experiments/results/top_submission_reverse_engineering_20260503_pr75_minp/archive.zip \
  --public-inflate-py /tmp/pr75-minp/submissions/qpose14_r55_segactions_minp/inflate.py \
  --output-dir experiments/results/top_submission_reverse_engineering_20260503_pr75_minp/raw_output_parity_pairs_33_104_598_cpu \
  --pair-indices 33,104,598 \
  --device cpu \
  --force
```

## Next Implementation Actions

1. Finish/record all-600 local CPU raw-output parity. If exact, current
   robust runtime is semantically faithful to public PR75/minp for this
   archive.
2. Build deterministic stream ablations against the C089/robust grammar:
   actions-only, renderer-only, pose-only, renderer+pose, actions+pose, and
   full public stream stack. These are local archive candidates only until
   claimed and exact CUDA auth-evaluated.
3. If full public stack exactly reproduces the public PR75/minp runtime, queue
   a T4 exact replay only after dispatch claim. If replay matches public
   metadata, use the component trace to rank which of the public streams
   actually moved score.
4. The most likely sub-0.314 path is not more top-40 action tweaking. It is
   public-minp parity plus targeted byte savings or component improvement on
   top of the 108-action / 55756-renderer / 898-pose basin.
