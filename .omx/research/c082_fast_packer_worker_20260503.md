# C082 Fast Packer Worker - 2026-05-03

## Scope

Task: search for contest-faithful byte-only archive/packer/self-compression
candidates around the current C082/C067 PR75/QZS3/QP1 basin without retraining
and without remote GPU dispatch.

Remote GPU dispatch performed: false.
Score claim: false.
Promotion eligible: false until exact CUDA/T4 auth eval on exact archive bytes.

Current A++ T4 frontier supplied at start:

- candidate: `c082_qp1_p6_delta_varint_actions_stream_resweep`
- score: `0.3154889937553647`
- bytes: `276394`
- sha256: `9b78333dd39c12c986ca7fc02a4bb35a6a61ef5a1cc0c9c9c840820eb840058a`

Already in flight at start:

- smaller C082/top40 lossless repack: `276333` bytes,
  sha256 `30932c684c6b09a7bd2bd248e17fd66a4bc448ed2fe5747f92e74bcceec66681`
- renderer shrink candidate: `275900` bytes,
  sha256 `bb8d1835303478183465e15b28fd9af8776b4ea1c07cdd6cfe0895032a267b64`

## Method

Added result-local screen:

- `experiments/results/c082_fast_packer_worker_20260503/search_p6_fast_repack.py`

The tool reads single-member `p` archives, requires a P6
`public_pr75_qzs3_qp1_segactions_p6_delta_varint` payload, recompresses only
existing logical streams, validates decoded stream parity through
`submissions/robust_current/unpack_renderer_payload.py`, and emits deterministic
ZIP_STORED archives plus manifests. It also records negative container checks
for outer Brotli and ZIP_DEFLATED member packing.

## Results

Primary candidate:

| rank | candidate | bytes | sha256 | source | local evidence | dispatch status |
| --- | --- | ---: | --- | --- | --- | --- |
| 1 | `shrink_queued_bb8d_p6_stream_resweep` | `275890` | `002b1d0681a895aac6dbf2eb1194c9d765debdee5e3b3101e034173e21295bac` | lossless resweep of `bb8d1835303478183465e15b28fd9af8776b4ea1c07cdd6cfe0895032a267b64` | decoded stream parity true; source pose-safety preflight true | recommended T4 exact eval after lane claim |

Byte deltas versus the queued `bb8d` source:

- `masks.mkv`: `-7` bytes by reusing the known smaller encoding for the same raw mask stream.
- `optimized_poses.qp1`: `-1` byte by reusing the known smaller encoding for the same raw QP1 stream.
- `seg_tile_actions.delta_varint`: `-2` bytes from local Brotli params
  `quality=9, mode=0, lgwin=16, lgblock=0`.
- `renderer.bin`: `0` bytes; the source renderer Brotli stream stayed best in the local screen.

The candidate is `10` bytes smaller than `bb8d` and `504` bytes smaller than
the current A++ frontier. This is only formula-only rate pressure
(`-0.000006658589531221714` vs `bb8d`, `-0.0003355929123735744` vs current
frontier if components were equal), not a score claim.

Ranked fallback candidates:

| rank | candidate | bytes | sha256 | source sha256 | note |
| --- | --- | ---: | --- | --- | --- |
| 2 | `shrink_frame1_head_0075_p6_stream_resweep` | `276084` | `befe0e312377485d322735eb634fdbdad13fca9320dd66b0a8f7494ed4f69ba1` | `7814f529ae964b5aa787134d114eca7734fbcaff3c85fcc6e2ed23b50edd072d` | pose-safe source; `-142` bytes from stream resweep |
| 3 | `shrink_frame2_head_005_p6_stream_resweep` | `276211` | `331e2d2fa54029cd9c1da860e7399197a6f94780752f484e44666c7327efea23` | `efccff1fbfb5f9dcc38c05a79e1ab5ebe2f90d13fcae5fdc0e0f8465e79b6ed6` | pose-safe source; `-11` bytes from stream resweep |
| 4 | `shrink_shared_trunk_004_p6_stream_resweep` | `276272` | `9a00553a2ea2646ea0336686df6cedab18c10018b651c9995f46cef22ca0b140` | `074c38d67e1ba927800c8ccdd735b027045fe60401e724c702ea6c9d4ba0f62d` | pose-safe source; `-10` bytes from stream resweep |

No-op and negative checks:

- Current A++ C082 `9b78333d...` is already locally optimal under this P6 stream screen; emitted archive is byte-identical/no-op.
- Already-running smaller C082 `30932c68...` is already locally optimal under this P6 stream screen; emitted archive is byte-identical/no-op.
- Outer Brotli did not improve any screened P6 payload.
- ZIP_DEFLATED member packing was consistently `+85` archive bytes versus deterministic ZIP_STORED.
- More aggressive renderer block-size shrink candidates below these byte counts remain dispatch-blocked by renderer transplant pose-safety failures and should not be T4-dispatched from this evidence alone.

## Artifacts

- Screen summary:
  `experiments/results/c082_fast_packer_worker_20260503/p6_stream_resweep/p6_repack_screen_summary.json`
- Dispatch recommendation:
  `experiments/results/c082_fast_packer_worker_20260503/dispatch_recommendation.json`
- Best archive:
  `experiments/results/c082_fast_packer_worker_20260503/p6_stream_resweep/shrink_queued_bb8d_p6_stream_resweep/archive.zip`
- Best manifest:
  `experiments/results/c082_fast_packer_worker_20260503/p6_stream_resweep/shrink_queued_bb8d_p6_stream_resweep/manifest.json`
- Best local unpack:
  `experiments/results/c082_fast_packer_worker_20260503/smoke_unpack_best/renderer_payload_unpack_summary.json`
- Source local unpack for parity:
  `experiments/results/c082_fast_packer_worker_20260503/smoke_unpack_source_bb8d/renderer_payload_unpack_summary.json`

## Local Validation

Commands run:

```bash
unzip -t experiments/results/c082_fast_packer_worker_20260503/p6_stream_resweep/shrink_queued_bb8d_p6_stream_resweep/archive.zip
rm -rf experiments/results/c082_fast_packer_worker_20260503/smoke_unpack_best
mkdir -p experiments/results/c082_fast_packer_worker_20260503/smoke_unpack_best
unzip -q experiments/results/c082_fast_packer_worker_20260503/p6_stream_resweep/shrink_queued_bb8d_p6_stream_resweep/archive.zip -d experiments/results/c082_fast_packer_worker_20260503/smoke_unpack_best
.venv/bin/python submissions/robust_current/unpack_renderer_payload.py experiments/results/c082_fast_packer_worker_20260503/smoke_unpack_best --summary-json experiments/results/c082_fast_packer_worker_20260503/smoke_unpack_best/renderer_payload_unpack_summary.json
```

Result:

- ZIP integrity: passed.
- Runtime payload parse: unpacked `renderer.bin`, `masks.mkv`, `optimized_poses.qp1`, and `seg_tile_actions.bin`.
- Source-vs-candidate decoded member parity: `cmp=0` for all four members.
- Candidate unpack summary:
  - payload format: `public_pr75_qzs3_qp1_segactions_p6_delta_varint`
  - payload bytes: `275790`
  - renderer SHA: `e77225bcd1ab1aecef8d2ee35f84b7314cb5779183316889e545a82e70258629`
  - mask SHA: `a5c2b89c110d75220cd09b2f27f2e92844626ae7ed0d2c797290dcf43c7068eb`
  - QP1 pose SHA: `1d2a6c31e836aa138bd09b3448db7e066f29f0cfcbf71b00e13357242655b583`
  - action records SHA: `6551a4bee3a9741000dbd8e8c6de8c0ffeb6ca03b644c3ff8a47aa3078ae752b`

Focused code checks:

```bash
.venv/bin/python -m py_compile experiments/results/c082_fast_packer_worker_20260503/search_p6_fast_repack.py experiments/build_pr75_lossless_repack_candidates.py submissions/robust_current/unpack_renderer_payload.py
.venv/bin/python -m pytest src/tac/tests/test_build_pr75_lossless_repack_candidates.py src/tac/tests/test_unpack_renderer_payload_fixedslice.py -q
```

Result: `11 passed in 1.28s`.

## Exact T4 Eval Recommendation

Do not claim score from local evidence. Before any remote T4 job, claim the
lane with `tools/claim_lane_dispatch.py claim ...` and use the exact archive
SHA/bytes below in the notes.

Recommended T4 exact eval command once running on a claimed T4/equivalent CUDA
worker:

```bash
.venv/bin/python -u experiments/contest_auth_eval.py \
  --archive experiments/results/c082_fast_packer_worker_20260503/p6_stream_resweep/shrink_queued_bb8d_p6_stream_resweep/archive.zip \
  --inflate-sh submissions/robust_current/inflate.sh \
  --upstream-dir upstream \
  --device cuda \
  --keep-work-dir \
  --work-dir experiments/results/c082_fast_packer_worker_20260503/t4_exact_eval_shrink_queued_bb8d_p6_stream_resweep
```

Expected archive custody:

- bytes: `275890`
- sha256: `002b1d0681a895aac6dbf2eb1194c9d765debdee5e3b3101e034173e21295bac`
- evidence grade before CUDA: `empirical_lossless_byte_transform`

Dispatch criteria:

1. If the in-flight `bb8d...` exact eval has not closed and queue capacity is
   available, dispatch this candidate instead of another copy of `bb8d...`.
2. If `bb8d...` returns component-clean, this exact-byte resweep is the direct
   10-byte better custody candidate and should receive T4 confirmation.
3. If `bb8d...` fails component gates, do not promote this candidate; classify
   with the same renderer-shrink failure family unless exact eval shows a
   runtime-custody difference.

## Supersession - 2026-05-03T07:22Z

The in-flight `bb8d...` source exact eval has now closed as A-negative:

- Source archive SHA-256:
  `bb8d1835303478183465e15b28fd9af8776b4ea1c07cdd6cfe0895032a267b64`
- T4 score: `0.5066348615388583`
- T4 PoseNet: `0.00685808`
- Failure class: PoseNet component-gate collapse.

Therefore `shrink_queued_bb8d_p6_stream_resweep` is blocked from dispatch.
Its decoded logical streams are byte-identical to the failed source, so the
10-byte wrapper improvement cannot change the component failure. This is a
useful lossless-packer result for future clean sources, but not a valid current
T4 spend.

The machine-readable recommendation was updated with
`dispatch_status_current=blocked_inherited_a_negative_source`, and the best
candidate manifest now carries a `remote_dispatch_block` entry.
