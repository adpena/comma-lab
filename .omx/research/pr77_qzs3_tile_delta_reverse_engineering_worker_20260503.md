# PR77 QZS3 Tile Delta Reverse Engineering - Worker PR77 - 2026-05-03

Scope: ingest public PR77 `qzs3_tile_delta_r147` only. PR78
`qzs3_script_payload_r147` is a withdrawn rules-relocation submission and was
ignored except as a compliance negative.

## Source Custody

- PR77: <https://github.com/commaai/comma_video_compression_challenge/pull/77>
  head `e9512c284dcd233dc5c6a7ed3362a943f8f5e340`, state `open`.
- PR77 public report: PoseNet `0.00049314`, SegNet `0.00060631`, archive
  bytes `276551`, recomputed score from the PR body components
  `0.3149988868909903`.
- PR78: <https://github.com/commaai/comma_video_compression_challenge/pull/78>
  state `closed`; comments record: "Withdrawing this one as a
  rules-interpretation payload relocation submission. Leaving #77 as my
  legitimate final submission."

Primary artifacts:

- Archive:
  `experiments/results/top_submission_reverse_engineering_20260503_pr77/archive.zip`
- Download mirror:
  `experiments/results/top_submission_reverse_engineering_20260503_pr77/downloads/pr77_qzs3_tile_delta_r147_archive.zip`
- PR77 source:
  `experiments/results/top_submission_reverse_engineering_20260503_pr77/sources/inflate.py`
  and `inflate.sh`
- Byte profile:
  `experiments/results/top_submission_reverse_engineering_20260503_pr77/byte_profile.json`
- Decoded-member summary:
  `experiments/results/top_submission_reverse_engineering_20260503_pr77/unpack_summary.json`
- Stream comparison:
  `experiments/results/top_submission_reverse_engineering_20260503_pr77/comparisons/pr77_vs_pr75_c089_streams.json`

## Archive Profile

- ZIP bytes: `276551`
- ZIP SHA-256:
  `f90880383c95e14d82704f99db9b20944786ae6452a844348638b06c439972af`
- ZIP member: single stored member `p`
- Payload bytes: `276451`
- Payload SHA-256:
  `3be6a58673133db2dd14d9f1f0903d528e452bd2e930d57fb4adb02bf264f8ec`
- Local archive validator: passed central/local filename integrity and member
  whitelist (`p`).

Payload fixed-slice grammar recovered from PR77 `inflate.py`:

| segment | encoded bytes | encoded sha256 | decoded bytes | decoded sha256 |
|---|---:|---|---:|---|
| `masks.mkv` | `219472` | `d1ae0d39c848e5715b74bb0122269066a4a1ab60ba9e12f34e70fd62ac136d87` | `223385` | `a5c2b89c110d75220cd09b2f27f2e92844626ae7ed0d2c797290dcf43c7068eb` |
| `renderer.bin` | `55756` | `e892539adec2406f87c824accc0effc80911f160ca8d324429c5d2bac175f2cf` | `59288` | `30159b6ace27a4013d1516c340d58f6d683e6847429fd3d6303a2c650aa2abef` |
| `seg_tile_actions.bin` | `325` | `d8c75e4f3725bbcf608434f0a78f5b37a9ce86bd8177c71092fd727d7e2af75a` | `588` runtime bytes | `8ac9a01caad973096c58b42daf2b1a8e476ad68cf285d443baa4ac94fdb42255` |
| `optimized_poses.qp1` | `898` | `7d7c35f4e7b0eb7022e56aaa76cad111b6c2e536b68080f10a536b2cb418a082` | `1140` QP1 bytes | `8d77f6a39d1a84eca78fbe8fa5ddc31f6ada29814b49e32daeb800ea84a015cc` |

Action stream:

- Wire kind: PR77 grouped tile/frame-delta varint without the `SG2` magic.
- Encoded Brotli bytes: `325`
- Decompressed wire bytes: `371`
- Runtime raw4 bytes: `588`
- Runtime records: `147`
- Unique pairs: `121`; pair range `11..599`
- Unique tiles: `24`; tile range `82..140`
- Unique action IDs: `73`; action range `2..107`

## Stream Comparison

Against PR75/minp
`experiments/results/top_submission_reverse_engineering_20260503_pr75_minp/archive.zip`:

- PR77 is `+70` archive bytes and `+70` payload bytes.
- `masks.mkv`, `renderer.bin`, and `optimized_poses.qp1` are byte-identical.
- Only `seg_tile_actions.bin` changes: PR77 runtime action bytes `588` versus
  PR75/minp `432`, records `147` versus `108`.
- Action relation is not a pure superset: exact record overlap `56`, pair/tile
  overlap `67`.

Against C089 staged top25 ampminus1 P3
(`bdc966be526bb8f5ddcd433eaff2e3708779fd291eb40deea5539df5a7bc2386`):

- PR77 is `+223` archive bytes and `+223` payload bytes.
- `masks.mkv` is byte-identical.
- `renderer.bin` has same decoded length but different bytes
  (`1948` changed prefix bytes; first difference at offset `38044`).
- `seg_tile_actions.bin` differs completely at exact-record level: PR77 has
  `147` records, C089 has `25`, exact overlap `0`, pair/tile overlap `13`.
- PR77 pose equals PR75/minp QP1; the local C089 comparison artifact stores
  fp16 decoded pose, so direct QP1-vs-QP1 byte comparison was not made here.

## Compliance And Replay Risk

Safe for exact T4 replay, with caveats:

- Use PR77's own `inflate.sh` and `inflate.py`; do not route this archive
  through `submissions/robust_current` unless the shared parser is deliberately
  updated for the 276451-byte fixed-slice variant.
- The archive itself passes local ZIP integrity and member whitelist checks.
- `inflate.py` compiles under `.venv/bin/python -m py_compile`.
- Runtime dependencies are public Python packages already expected by this
  family (`av`, `brotli`, `einops`, `torch`, `tqdm`). The exact replay runner
  must verify they exist in the T4 environment.
- PR77 `inflate.py` contains a broad `torch.load` fallback for unknown renderer
  payloads, but this archive's renderer decodes to `QZS3`, so the fallback is
  not exercised for this exact archive.
- No score claim is made from this worker. The PR body report is external until
  replayed through `archive.zip -> inflate.sh -> upstream/evaluate.py` on CUDA.

Recommended claim before any remote/T4 dispatch:

```bash
.venv/bin/python tools/claim_lane_dispatch.py claim \
  --lane-id pr77_qzs3_tile_delta_r147_t4_replay \
  --platform lightning \
  --instance-job-id exact_eval_pr77_qzs3_tile_delta_r147_t4_20260503T1130Z \
  --agent codex:worker-pr77 \
  --predicted-eta-utc 2026-05-03T12:30Z \
  --status eval \
  --notes "PR77 public qzs3_tile_delta_r147 exact T4 replay; archive f9088038; external report recomputes 0.3149988869"
```

Recommended exact replay command after the claim, on a T4/CUDA runner:

```bash
.venv/bin/python experiments/contest_auth_eval.py \
  --archive experiments/results/top_submission_reverse_engineering_20260503_pr77/archive.zip \
  --inflate-sh experiments/results/top_submission_reverse_engineering_20260503_pr77/sources/inflate.sh \
  --device cuda \
  --work-dir experiments/results/lightning_batch/exact_eval_pr77_qzs3_tile_delta_r147_t4_20260503T1130Z \
  --keep-work-dir
```

No remote job was dispatched by this worker.
