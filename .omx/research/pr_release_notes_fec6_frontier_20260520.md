# FEC6 frontier — `0.192051 [contest-CPU]` / `0.226210 [contest-CUDA T4]`

Submitted by Alejandro Peña <adpena@gmail.com>.

This release hosts the `archive.zip` for a FEC6 selector bolt-on submission to the comma.ai video compression challenge. Beats PR #101 GOLD by `-0.000794 [contest-CPU]`.

## Archive facts

- **SHA-256**: `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`
- **Size**: 178,517 bytes
- **Layout**: single ZIP member `x` (178,417 bytes, stored uncompressed at `compression_type=0` / `ZIP_STORED`). Member `x` wraps `FP11 + source_len + source_pr101_payload + selector_len + selector_payload` — i.e., PR #101's source payload (HNeRV state-dict at FP11 + latent sidecar, both inside PR #101's Brotli envelope) plus the locally appended FEC6 selector (fixed-Huffman bitstream, NOT additionally Brotli-coded). The ZIP member `x` itself is stored uncompressed.

## Measured scores (axis-separated)

| Axis | Score | Hardware | Notes |
|---|---|---|---|
| `[contest-CPU]` | `0.1920513168811056` | Modal Linux x86_64 (Ubuntu, single-thread, no GPU) | Matches upstream `ubuntu-latest` GHA runner family |
| `[contest-CUDA T4]` | `0.22621002169349796` | Modal NVIDIA Tesla T4 | Same `archive.zip` SHA; same `inflate.sh` runtime tree |

Both evaluated with `upstream/evaluate.py` on the exact `archive.zip` bytes. Not an A100 eval; not a Vast.ai eval. CPU and CUDA reported as separate observations per dual-axis discipline.

Rate term: `25 * 178517 / 37545489 ≈ 0.118867` (exact Decimal: `0.11886714273451066…`).

vs PR #101 GOLD (sha `b83bf3488625dbd73adeddff91712994197ab53098e578e91327a0c6e49efb3e`, 178,258 bytes, public CPU `0.1928450127024255`): byte delta `+259`; total CPU-axis delta `-0.0007936958213199` (reported `-0.000794`); this already includes the +259-byte rate cost.

## What's new in this submission

FEC6 selector bolt-on around the public HNeRV lineage:

1. **FEC6 31-mode frame-exploit selector** (K=16 active palette) — NEW BOLT-ON. PR #101 has no per-pair selector. Deterministic per-frame-pair transform space (identity / luma + RGB biases / blue-chroma amp / 1-pixel rolls); offline scorer-targeted search picks one of K=16 transforms per pair.
2. **Fixed-Huffman k=16 codebook on selector indices** — NEW BOLT-ON (sister to PR #101's canonical Huffman for the latent sidecar but applied to a NEW layer with a FIXED code, so no per-archive header bytes for code declaration). 600-pair selector compacts to a 249-byte wire payload (3.32 bits/pair).

## Synergy boundary

Brotli (RFC 7932) operates only inside PR #101's source-payload region (HNeRV state-dict + sidecar), NOT at the ZIP layer and NOT over the appended FEC6 selector bitstream. The ZIP member `x` itself is stored uncompressed (`compression_type=0` / `ZIP_STORED`).

## Attribution chain

- HNeRV decoder (`src/model.py`) — originates in PR [#95](https://github.com/commaai/comma_video_compression_challenge/pull/95) by `@AaronLeslie138` (`hnerv_muon`). Byte-identical across PR #95 / PR #98 / PR #101 / this packet.
- Immediate byte substrate — PR [#101](https://github.com/commaai/comma_video_compression_challenge/pull/101) by `@SajayR` (`hnerv_ft_microcodec`).
- Prior lineage — PR [#98](https://github.com/commaai/comma_video_compression_challenge/pull/98) by `@EthanYangTW`; PR [#100](https://github.com/commaai/comma_video_compression_challenge/pull/100) by `@BradyMeighan` (sister `hnerv_lc_v2` sidecar/schema pattern; not directly inherited by this packet); PR [#102](https://github.com/commaai/comma_video_compression_challenge/pull/102) by `@EthanYangTW`; PR [#103](https://github.com/commaai/comma_video_compression_challenge/pull/103) by `@rem2` (does NOT inherit `constriction` range coder).

## Reproducibility

60-second CPU smoke (replace `<PINNED_COMMIT>` with the source-sync commit):

```bash
git clone https://github.com/adpena/comma-lab.git && cd comma-lab && git checkout <PINNED_COMMIT> && \
  cd experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir && \
  python -m venv .venv && .venv/bin/pip install --quiet torch numpy brotli && \
  mkdir -p /tmp/data /tmp/out && unzip -oq archive.zip -d /tmp/data && echo "0.mkv" > /tmp/list.txt && \
  PACT_PYTHON_BIN=.venv/bin/python bash inflate.sh /tmp/data /tmp/out /tmp/list.txt && \
  shasum -a 256 /tmp/out/0.raw
# expect: d1afc583b01ff4a7aaa844d4f03ece3ed381d56763a06cb2c5e011526e5f868c
```

Dependency closure: `torch + numpy + brotli` (`lzma` is stdlib). The entry-point contract is the canonical `inflate.sh <archive_dir> <output_dir> <file_list>`. The rate term is fully accounted for by `archive.zip` bytes; no out-of-archive sidecars; no scorer weights loaded at inflate time.

## Submission

The companion PR will be opened against `commaai/comma_video_compression_challenge` referencing this release URL for `curl -L` archive retrieval per the upstream template requirement.
