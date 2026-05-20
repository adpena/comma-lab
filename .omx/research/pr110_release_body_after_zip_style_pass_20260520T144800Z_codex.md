# FEC6 selector submission - `0.192051 [contest-CPU]` / `0.226210 [contest-CUDA T4]`

Submitted by Alejandro Peña <adpena@gmail.com>.

This release hosts the `archive.zip` for PR #110, a FEC6 selector bolt-on submission to the comma.ai video compression challenge. It improves the immediate PR #101 CPU anchor by `-0.0007936958213199 [contest-CPU]` (reported as `-0.000794`).

## Archive facts

- **SHA-256**: `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`
- **Size**: 178,517 bytes
- **Layout**: single ZIP member `x` (178,417 bytes, stored uncompressed at `compression_type=0` / `ZIP_STORED`). Member `x` wraps `FP11 + source_len + source_pr101_payload + selector_len + selector_payload`: PR #101's source payload (HNeRV state-dict at FP11 + latent sidecar, both inside PR #101's Brotli envelope) plus the locally appended FEC6 selector (fixed-Huffman bitstream, not additionally Brotli-coded). The ZIP member `x` itself is stored uncompressed.

## Measured scores (axis-separated)

| Axis | Score | Hardware | Notes |
|---|---:|---|---|
| `[contest-CPU]` | `0.1920513168811056` | Modal Linux x86_64 (Ubuntu, no GPU) | Upstream report records `num_threads: 2` |
| `[contest-CUDA T4]` | `0.22621002169349796` | Modal NVIDIA Tesla T4 | Same `archive.zip` SHA; same `inflate.sh` runtime tree |

Both were evaluated with the upstream evaluator on the exact `archive.zip` bytes. This is not an A100 eval and not a Vast.ai eval. CPU and CUDA are reported as separate observations per dual-axis discipline.

Rate term: `25 * 178517 / 37545489 = 0.11886714273451067`.

vs PR #101 GOLD (archive SHA `b83bf3488625dbd73adeddff91712994197ab53098e578e91327a0c6e49efb3e`, 178,258 bytes, CPU `0.1928450127024255`): byte delta `+259`; total CPU-axis delta `-0.0007936958213199`, already including the +259-byte rate cost.

## What's new in this submission

FEC6 selector bolt-on around the public HNeRV lineage:

1. **FEC6 31-mode frame-exploit selector** (K=16 active palette). PR #101 has no per-pair selector. The deterministic per-frame-pair transform space includes identity, luma/RGB biases, blue-chroma amplification, and 1-pixel rolls; offline scorer-targeted search picks one of K=16 transforms per pair.
2. **Fixed-Huffman k=16 codebook on selector indices**. This is sister to PR #101's canonical Huffman for the latent sidecar, but applied to a new selector-index layer with a fixed code, so no per-archive header bytes are spent declaring the code table. The 600-pair selector compacts to a 249-byte wire payload.

## Synergy boundary

Brotli (RFC 7932) operates only inside PR #101's source-payload region (HNeRV state-dict + sidecar), not at the ZIP layer and not over the appended FEC6 selector bitstream. The ZIP member `x` itself is stored uncompressed (`compression_type=0` / `ZIP_STORED`).

## Attribution chain

- HNeRV decoder (`src/model.py`) originates in PR [#95](https://github.com/commaai/comma_video_compression_challenge/pull/95) by `@AaronLeslie138` (`hnerv_muon`). It is byte-identical across PR #95 / PR #98 / PR #101 / PR #110.
- Immediate byte substrate is PR [#101](https://github.com/commaai/comma_video_compression_challenge/pull/101) by `@SajayR` (`hnerv_ft_microcodec`).
- Prior lineage: PR [#98](https://github.com/commaai/comma_video_compression_challenge/pull/98) by `@EthanYangTW`; PR [#100](https://github.com/commaai/comma_video_compression_challenge/pull/100) by `@BradyMeighan`; PR [#102](https://github.com/commaai/comma_video_compression_challenge/pull/102) by `@EthanYangTW`; PR [#103](https://github.com/commaai/comma_video_compression_challenge/pull/103) by `@rem2`. This submission does not import PR #103's `constriction` range coder.

## Reproducibility smoke

60-second CPU inflate smoke against the PR #110 runtime:

```bash
git clone https://github.com/adpena/comma_video_compression_challenge.git pr110-runtime
cd pr110-runtime
git checkout ec6cc7f98c16b6ad2db8bc7cde65757bb7993004
SUB=submissions/hnerv_fec6_fixed_huffman_k16
python -m venv .venv
.venv/bin/pip install --quiet torch numpy brotli
mkdir -p /tmp/pr110-data /tmp/pr110-out
curl -L -o /tmp/pr110-archive.zip \
  https://github.com/adpena/comma_video_compression_challenge/releases/download/fec6-frontier-submission-20260520/archive.zip
unzip -oq /tmp/pr110-archive.zip -d /tmp/pr110-data
printf '0.mkv\n' > /tmp/pr110-list.txt
PACT_PYTHON_BIN="$PWD/.venv/bin/python" bash "$SUB/inflate.sh" /tmp/pr110-data /tmp/pr110-out /tmp/pr110-list.txt
shasum -a 256 /tmp/pr110-out/0.raw
# expect: d1afc583b01ff4a7aaa844d4f03ece3ed381d56763a06cb2c5e011526e5f868c
```

Dependency closure: `torch + numpy + brotli` (`lzma` is stdlib). The entry-point contract is `inflate.sh <archive_dir> <output_dir> <file_list>`. The rate term is fully accounted for by `archive.zip` bytes; no out-of-archive sidecars; no scorer weights loaded at inflate time.

## Submission

PR #110 is open at https://github.com/commaai/comma_video_compression_challenge/pull/110 and references this release URL for `archive.zip` retrieval.
