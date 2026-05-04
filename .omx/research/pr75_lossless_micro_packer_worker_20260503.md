# PR75 Lossless Micro Packer Worker - 2026-05-03

Worker: Packer

Scope owned:
- `experiments/build_pr75_minp_lossless_micro_candidates.py`
- `src/tac/tests/test_build_pr75_minp_lossless_micro_candidates.py`
- `.omx/research/pr75_lossless_micro_packer_worker_20260503.md`
- `experiments/results/pr75_lossless_micro_packer_worker_20260503/`

No remote jobs were dispatched.

## Inputs

- C089/C067 PR75 QP1 top40 P6 frontier:
  - archive: `experiments/results/lightning_batch/exact_eval_c067_pr75_qp1_top40_p6_t4_awsfix1_20260503T0630Z/archive.zip`
  - bytes: `276342`
  - SHA-256: `0ec53e5b871149ed6eea56c0b9bcca3baec998d5bfad4f371979e0c90e62cea8`
  - exact A++ score from provided custody: `0.3154707273953505`
- Public PR75/minp:
  - archive: `experiments/results/top_submission_reverse_engineering_20260503_pr75_minp/archive.zip`
  - bytes: `276481`
  - SHA-256: `03a2afd5fe92c93a9b7b7e43625158a73b455f0cfbca82d278008a728db78746`

## Tooling Added

Added `experiments/build_pr75_minp_lossless_micro_candidates.py`.

The tool:
- reads one ZIP member `p` only and rejects hidden, duplicate, or zip-slip members;
- uses `submissions/robust_current/unpack_renderer_payload.py` as the parse/decode truth;
- emits deterministic single-member stored ZIPs with fixed timestamp, permission, and member name;
- writes per-candidate manifests and `candidate_matrix.json`;
- marks no-op and speculative wire-form probes non-dispatchable;
- keeps `score_claim=false` and `promotion_eligible=false` for all outputs.

Focused tests:
- `src/tac/tests/test_build_pr75_minp_lossless_micro_candidates.py`

## Candidate Matrix

All deltas are byte deltas versus C089 `276342` bytes. Formula-only score deltas are rate-term-only diagnostics and are not score claims.

| candidate | bytes | SHA-256 | delta vs C089 | semantic contract | dispatch gate |
| --- | ---: | --- | ---: | --- | --- |
| `public_renderer_c089_p6_lossless_stream_resweep` | `276124` | `337b04040bee1316375bae8b2cfc2f08acddc235e4ee02d37ecfd62c0c831d95` | `-218` | Renderer transplant: decoded public PR75/minp renderer, C089 decoded masks/poses/actions byte-identical; C089 streams losslessly re-Brotlied. | `experiments/preflight_renderer_transplant_pose_safety.py` against exact C089 SHA and candidate SHA, then claim lane, then exact CUDA auth eval. |
| `c089_raw_no_header_fixedslice_probe` | `276321` | `da92eea8b59ba1e377bc83bdc5483abd9d89bd6393783338e51b0da718230bd4` | `-21` | Invalid current-runtime raw fixed-slice header-removal probe. Parser rejects this C089 length tuple. | Non-dispatchable; requires runtime support and tests before any exact eval. |
| `c089_p6_lossless_stream_resweep` | `276333` | `3de0d1546c909404df2f9b40a9ab8218100be36650b6bfcb3132bac50400ec7f` | `-9` | Strict decoded byte parity vs C089 for masks, renderer, QP1 poses, and tile actions. | Claim lane, then exact CUDA auth eval. |
| `c089_p6_action_resweep` | `276341` | `428757d21a6237d284e80f2a0305e0aff99ef0523c3cbcbd476254c84948787b` | `-1` | Strict decoded byte parity vs C089; only P6 action Brotli stream changed. | Claim lane, then exact CUDA auth eval. |
| `c089_zip_rewrite_noop` | `276342` | `0ec53e5b871149ed6eea56c0b9bcca3baec998d5bfad4f371979e0c90e62cea8` | `0` | Byte-identical no-op control. | Do not dispatch. |
| `public_renderer_pose_c089_p6_lossless_stream_resweep` | `276346` | `6f9cc5406f4b2d57409b940699b0d06f21154ae69382caa08d90f31f14cf4b18` | `+4` | Public renderer and public pose transplant with C089 mask/actions. | Non-dispatchable byte regression; local runtime parity would be required before exact CUDA. |
| `c089_p3_raw_actions_probe` | `276371` | `4723425564d8310deea4fae7234298c8b84ed7e1ead4d3560ccb94ef3de3778c` | `+29` | Strict decoded byte parity vs C089 using P3 raw action Brotli. | Do not dispatch unless a byte win appears; exact CUDA would still be required. |
| `public_minp_p6_sorted_actions_probe` | `276490` | `1cdbfd3d9963e1eeaa6bb1914b93f3652b6631b03b5ee5711456fb1178949e47` | `+148` | Non-dispatchable public action ordering probe; sorting enables P6 but changes decoded action order. | Local runtime raw-output parity required; duplicate pair/tile prevents assuming commutation. |
| `c089_p5_action_dict_probe` | `276519` | `6b4eba54366d62631efe6462b80cd091f8b951784f82dd90e28300c5caf595f6` | `+177` | Non-dispatchable P5 action-dictionary remap probe. | Local runtime raw-output parity required; byte regression. |
| `public_minp_p3_raw_actions_probe` | `276541` | `911a0e8547dccc12281576b35c11267dcf71ca239e493adc3f645265418f30a0` | `+199` | Strict decoded byte parity vs public PR75/minp using P3 raw action Brotli. | Do not dispatch; byte regression versus C089 and public fixed-slice source. |

Matrix artifact:
- `experiments/results/pr75_lossless_micro_packer_worker_20260503/candidate_matrix.json`

## Findings

1. Strict C089 lossless resweep saves `9` bytes:
   - `masks.mkv.br`: `219472 -> 219465` using Brotli `quality=11, mode=0, lgwin=19, lgblock=17`
   - `optimized_poses.qp1.br`: `677 -> 676` using Brotli `quality=11, mode=0, lgwin=16, lgblock=0`
   - `seg_tile_actions.delta_varint.br`: `116 -> 115` using Brotli `quality=9, mode=0, lgwin=10, lgblock=0`
   - `renderer.bin.br`: unchanged at `55965`
   - decoded members are byte-identical to C089.

2. Public-renderer-only candidate improves the already queued `276132`-byte candidate by `8` bytes:
   - new archive: `276124`
   - candidate SHA-256: `337b04040bee1316375bae8b2cfc2f08acddc235e4ee02d37ecfd62c0c831d95`
   - decoded changes vs C089: only `renderer.bin`
   - exact next gate: renderer-transplant pose-safety preflight against exact source/candidate SHAs, then lane claim, then exact CUDA auth eval.

3. ZIP overhead is already at the single-member floor for this contract:
   - `c089_zip_rewrite_noop` rewrites to exactly the same archive SHA and bytes.
   - member name `p` and stored ZIP overhead remain `100` bytes.

4. Dropping the P6 self-describing header would show `21` bytes of apparent room after stream resweep, but it is invalid today:
   - `c089_raw_no_header_fixedslice_probe` is `276321` bytes.
   - robust runtime parser rejects it with bad payload magic because no fixed-slice table supports the C089 stream-length tuple.
   - this is not contest-faithful and must not dispatch without runtime support, parser tests, and exact eval.

5. P3/P5 action alternatives are negative:
   - C089 P3 raw action wire is decoded-byte preserving but `+29` bytes.
   - C089 P5 3-byte packed actions need a charged dictionary and become `+177` bytes.
   - Public/minp P3 raw action self-describing payload is `+199` bytes vs C089.
   - Public/minp sorted P6 action probe is `+148` bytes and changes decoded action order; one duplicate pair/tile means commutation is not safe to assume.

## Verification

Commands run:

```bash
.venv/bin/python -m pytest src/tac/tests/test_build_pr75_minp_lossless_micro_candidates.py -q
.venv/bin/python experiments/build_pr75_minp_lossless_micro_candidates.py --force
```

Results:
- focused tests: `2 passed`
- real candidate build: `10` candidates emitted
- no remote dispatch performed

## Dispatch Notes

Do not dispatch from this ledger directly. Any exact eval dispatch must first:

1. claim the lane with `tools/claim_lane_dispatch.py claim ...`;
2. for `public_renderer_c089_p6_lossless_stream_resweep`, pass `experiments/preflight_renderer_transplant_pose_safety.py` with exact source/candidate SHA matching;
3. run exact CUDA auth eval through `archive.zip -> inflate.sh -> upstream/evaluate.py` using `experiments/contest_auth_eval.py --device cuda`;
4. preserve `contest_auth_eval.json`, logs, runtime tree hash, archive SHA, bytes, and recomputed score.
