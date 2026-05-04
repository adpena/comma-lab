# Block-FP C067 Local Builder Addendum

Date: 2026-05-02

Scope: local byte-screen builder for QBF1/block-FP renderer transplants on the
C067 public-floor packed archive. No remote job was launched. No score claim is
made.

## Contract

Builder:

```text
experiments/build_blockfp_c067_archive.py
```

Output evidence grade before CUDA auth eval:

```text
empirical_archive_candidate_until_exact_cuda
score_claim=false
promotion_eligible=false
```

The builder extracts a packed C067/public-floor renderer payload through the
runtime unpacker, decodes supported JointFrameGenerator renderer wire formats
(`QZS3`, `MQZ1`, or `QBF1`) into a state dict, repacks the renderer as QBF1
block-FP bytes, and writes deterministic single-member archives through the
existing packed-payload writer. It fails closed for unsupported renderer magic
instead of falling through to `torch.load()`.

Runtime contract:

```text
archive.zip -> p -> unpack_renderer_payload.py -> renderer.bin(QBF1)
inflate_renderer.py::_load_renderer QBF1 branch -> tac.qbf1_renderer_codec.load_qbf1
```

No scorer import is required at inflate time. All score-affecting bits remain
inside `archive.zip`.

## Local Source Probe

Measured source archive:

```text
path=experiments/results/c067_breakthrough_candidate_matrix_20260502T1030Z/line_search_source_c067_fixedslice/archive.zip
archive_bytes=276214
archive_sha256=226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a
zip_member=p
p_bytes=276114
p_sha256=1f0aaaa9d06967552df6f22d39905a48349239dee4731b093067951a20adc2af
payload_format=public_pr67_qzs3_qp1_fixed_slices
```

Logical runtime members after runtime parser:

```text
renderer.bin: bytes=59288 sha256=5657593aec0bf380... magic=QZS3
masks.mkv: bytes=223385 sha256=a5c2b89c110d7522...
optimized_poses.bin: bytes=7200 sha256=5236cf75cc95f6a9...
```

## Initial QBF1 Byte Screen

Decoded C067 QZS3 state was repacked locally as deterministic QBF1. Raw
renderer payload bytes are larger than the current QZS3 source for every block
size checked:

```text
block=16   qbf1_bytes=143301
block=24   qbf1_bytes=135979
block=32   qbf1_bytes=132129
block=48   qbf1_bytes=128667
block=64   qbf1_bytes=126584
block=96   qbf1_bytes=124889
block=128  qbf1_bytes=124081
block=256  qbf1_bytes=122745
block=512  qbf1_bytes=122074
block=1024 qbf1_bytes=121891
```

Interpretation: the QBF1 path is structurally usable and runtime-consumable,
but the current QBF1-v1 int8+float32-scale+JSON metadata format is a
rate-negative transplant on the C067 QZS3 frontier. It should not be promoted
or exact-eval dispatched unless a later local layout beats the source archive
bytes by meaningful margin or needs exact CUDA only as a bounded diagnostic.

## Built Local Candidate

Command:

```text
.venv/bin/python experiments/build_blockfp_c067_archive.py --source-archive experiments/results/c067_breakthrough_candidate_matrix_20260502T1030Z/line_search_source_c067_fixedslice/archive.zip --output-dir experiments/results/blockfp_c067_candidate_20260502T_local_qbf1_b1024 --block-sizes 1024
```

Artifacts:

```text
summary=experiments/results/blockfp_c067_candidate_20260502T_local_qbf1_b1024/blockfp_c067_summary.json
manifest=experiments/results/blockfp_c067_candidate_20260502T_local_qbf1_b1024/qbf1_b1024/build_manifest.json
archive=experiments/results/blockfp_c067_candidate_20260502T_local_qbf1_b1024/qbf1_b1024/archive.zip
```

Byte result:

```text
source_archive_bytes=276214
source_archive_sha256=226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a
qbf1_block_size=1024
transformed_renderer_bytes=121891
transformed_renderer_sha256=82d275e87929bfcb0c2f99c947d0d694db2229e7004379c509d3f267ac3ef2a7
output_archive_bytes=283869
output_archive_sha256=6d331e479d961df22a2baa8b3f09722394ece7d0c194821c80c6aa354cb1449b
delta_bytes_vs_source_archive=+7655
formula_only_rate_delta_vs_source_archive=+0.0050971502861502215
local_archive_byte_win=false
score_claim=false
promotion_eligible=false
```

The runtime unpack check reconstructed the QBF1 renderer from the output
archive's `p` member and verified the transformed renderer SHA. This is a
runtime-consumable charged payload, but it is byte-negative and remains
non-promotable without exact CUDA auth eval.

## Verification Added

Focused test:

```text
src/tac/tests/test_build_blockfp_c067_archive.py
```

The test covers packed public mask-first source extraction, QZS3-to-QBF1
transplant, deterministic archive output, runtime unpacker consumption, manifest
fields, `score_claim=false`, `promotion_eligible=false`, and fail-closed
behavior for unsupported renderer magic.
