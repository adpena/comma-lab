# FEC6 Frame Exploit Selector - K=16 palette + fixed Huffman k=16, on the public HNeRV substrate

Score against the upstream evaluator (paired axes, same `archive.zip` bytes and `inflate.sh` runtime tree):

| Axis | Score | Host | Notes |
| --- | --- | --- | --- |
| `[contest-CPU]` | `0.192051` | Modal Linux x86_64, Ubuntu, 1 thread | matches upstream `ubuntu-latest` GHA runner family |
| `[contest-CUDA]` | `0.226210` | Modal Tesla T4 | same archive bytes, same runtime tree |

Headline against the current top merged submission (PR [#101](https://github.com/commaai/comma_video_compression_challenge/pull/101) by @SajayR, GOLD, `0.192845 [contest-CPU]`):
`-0.000794` total delta on the CPU axis the leaderboard ranks. This already includes the +259-byte rate cost; the rate cost is not subtracted again.

Archive: `archive.zip`, SHA-256 `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`, 178,517 bytes.

## Chain attribution

This submission is the smallest credible bolt-on we could write on top of the public HNeRV substrate. The HNeRV decoder file (`src/model.py`) is byte-identical to PR #95's reference and to the copy in PR #101. The entropy-coded selector pattern and the discipline of charging only what the contest scorer measures are downstream of the medal-class submissions that came before us:

- [PR #95](https://github.com/commaai/comma_video_compression_challenge/pull/95) - @AaronLeslie138 (`hnerv_muon`): HNeRV decoder substrate (`src/model.py` here is byte-identical to PR #95's).
- [PR #98](https://github.com/commaai/comma_video_compression_challenge/pull/98) - @EthanYangTW (`hnerv_muon_finetuned_from_pr95`): fine-tuning of the PR #95 HNeRV line.
- [PR #100](https://github.com/commaai/comma_video_compression_challenge/pull/100) - @BradyMeighan (`hnerv_lc_v2`): latent-correction sidecar / schema pattern.
- [PR #101](https://github.com/commaai/comma_video_compression_challenge/pull/101) - @SajayR (`hnerv_ft_microcodec`): compact fine-tuned HNeRV microcodec; the immediate byte substrate for this packet. *(current top merged)*
- [PR #102](https://github.com/commaai/comma_video_compression_challenge/pull/102) - @EthanYangTW (`hnerv_lc_v2_scale095_rplus1`): retune of the `hnerv_lc_v2` family.
- [PR #103](https://github.com/commaai/comma_video_compression_challenge/pull/103) - @rem2 (`hnerv_lc_ac`): `constriction` arithmetic/range coding. This packet does not inherit the PR #103 arithmetic coder.

We add **two new bolt-ons** on top of PR #101:

| # | Innovation | Classification | What it is + relation to PR #101 |
|---|---|---|---|
| 1 | **FEC6 31-mode frame-exploit selector** (K=16 active palette) | **NEW BOLT-ON** (no PR #101 equivalent) | A deterministic per-frame-pair transform space (identity / luma + RGB biases / blue-chroma amp / 1-pixel rolls). Offline scorer-targeted search picks one of K=16 transforms per pair against the upstream scorer's response on `videos/0.mkv`. Selector indices ship inside member `x` and replay at inflate time without on-device search. PR #101 has no per-pair selector mechanism. |
| 2 | **Fixed-Huffman k=16 codebook on selector indices** | **NEW BOLT-ON, sister technique to PR #101's canonical Huffman for the latent sidecar** | Static 16-symbol prefix code (lengths 2..8 bits) sized to the empirical mode-frequency distribution on the contest video. The 243-byte fixed-Huffman bitstream is 1,944 bits = **3.24 bits/pair**; the full 249-byte selector wire payload is **3.32 bits/pair**. PR #101's `src/codec.py` provides `decode_canonical_huffman` for the latent sidecar; FEC6 applies a fixed code to a new selector-index layer. |

**Synergy boundary:** the FEC6 selector indices live in a local `FP11` wrapper appended outside PR #101's Brotli-coded source payload, not inside the Brotli stream. PR #101 uses Brotli-coded decoder / sidecar streams inside `x`'s source-payload region; the FEC6 selector is byte-appended as a fixed-Huffman bitstream and is not further Brotli-compressed. The ZIP itself stores member `x` uncompressed (`stored`). Net `archive_bytes_added = 178,517 - 178,258 = 259` over PR #101's source archive: the rate term increases by `+25 * 259 / 37,545,489 = 0.00017245746885864238`, and the total CPU-axis score delta is `-0.000794`, already net of that rate cost.

**Inherited from PR #101 substrate (not our contribution):**

- **HNeRV decoder**: `src/model.py` is byte-identical to PR #95's reference and to the copy in PR #101.
- **Brotli q=11** of the state-dict + scale streams inside the PR101 source payload: PR #101's `src/codec.py` does `concatenated Brotli streams of q-bytes + fp16 scale per tensor`; we use this unchanged for the source-payload region.
- **Canonical Huffman for the latent sidecar** (`u8 dim, i8 delta_x100` per pair): PR #101's `decode_canonical_huffman_all`; our local `src/codec_sidecar.py` is a refactor of the PR #101 sidecar logic, byte-equivalent at decode.

## Files

All public file permalinks anchor to source-sync commit `b392343d758aba0d3595dd18609f9ca8a8af3e1b` on `https://github.com/adpena/comma-lab` (pushed to public `origin/main` at lockdown; verified visible). This commit contains the full submission_dir runtime tree including `src/codec_sidecar.py` (the local split codec module that the live runtime imports). The earlier `462f84cdd` reference did not contain `src/codec_sidecar.py`.

- `inflate.sh`: canonical 3-arg upstream-contract wrapper. Invokes `inflate.py` as `python3 inflate.py <archive_dir> <output_dir> <file_list>`.
- `inflate.py`: main orchestrator. Parses the local `FP11` wrapper out of member `x`, decodes the FEC6 selector stream, calls the HNeRV decoder, applies the per-frame transform, and writes decoded frames to `<output_dir>`. Carries the `INNOVATION:` annotation sites for the novel contributions.
- `src/codec.py`: HNeRV state-dict parser for the PR #101 source-payload region (FP11 unpack + Brotli + canonical Huffman); pure CPU; no scorer weights loaded. Inherited from PR #101.
- `src/codec_sidecar.py`: refactored latent-sidecar canonical-Huffman decoder from PR #101 (byte-equivalent at decode). Local split for separation of concerns.
- `src/frame_selector.py`: FES1 / FEC2 / FEC3 / FEC5 / FEC6 selector grammar plus the 31-mode deterministic frame-0 transform table. The selector stream inside member `x`, outside PR101's Brotli envelope, is replayed at inflate time against this table.
- `src/model.py`: `HNeRVDecoder`, byte-identical to PR #95's reference implementation by @AaronLeslie138.

The `archive.zip` itself contains a single member `x` (178,417 bytes, stored uncompressed). `x` packs `FP11 + source_len + source_pr101_payload + selector_len + selector_payload`: the PR #101 source payload (HNeRV state-dict at FP11 + latent sidecar, both inside PR #101's Brotli envelope) plus the locally appended FEC6 selector (fixed-Huffman bitstream, not additionally Brotli-coded). The runtime tree (`inflate.sh`, `inflate.py`, `src/*`) lives alongside `archive.zip` in the submission directory and is not inside the ZIP.

### Innovation grep convention

The novel contributions are tagged inline with `# INNOVATION:` comments. A reviewer can locate them all with one grep:

```bash
grep -rn "^# INNOVATION" submission_dir/
```

Permalinks are anchored to source-sync commit `b392343d758aba0d3595dd18609f9ca8a8af3e1b` on `https://github.com/adpena/comma-lab`.

## How to verify our score

### Easy 60-second smoke (CPU; no upstream repo or contest videos needed)

This produces the canonical byte-stable inflate-output SHA so a reviewer can verify `inflate.sh` runs deterministically. `archive.zip` contains only member `x`; the runtime tree (`inflate.sh`, `inflate.py`, `src/*`) lives alongside `archive.zip` in the submission directory and is not inside the ZIP. The clone + `cd` gives you the runtime tree.

```bash
git clone https://github.com/adpena/comma-lab.git && cd comma-lab && git checkout b392343d758aba0d3595dd18609f9ca8a8af3e1b && \
  cd experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir && \
  python -m venv .venv && .venv/bin/pip install --quiet torch brotli && \
  mkdir -p /tmp/data /tmp/out && unzip -oq archive.zip -d /tmp/data && echo "0.mkv" > /tmp/list.txt && \
  PACT_PYTHON_BIN=.venv/bin/python bash inflate.sh /tmp/data /tmp/out /tmp/list.txt && \
  shasum -a 256 /tmp/out/0.raw
# expect: d1afc583b01ff4a7aaa844d4f03ece3ed381d56763a06cb2c5e011526e5f868c  /tmp/out/0.raw
```

That single SHA proves the runtime is deterministic at the pinned commit and produces the same bytes recorded in `.omx/research/codex_codec_py_refactor_verification_20260519T200658Z.md`.

### Full score verification (upstream `evaluate.py`)

On a Linux x86_64 host (Ubuntu 22.04 or comparable, single-thread CPU, no GPU; matches the upstream GHA runner family). Key correction versus the prior draft: `archive.zip` contains only the rate-charged payload member `x`; the runtime tree is staged separately from the cloned submission directory, not extracted from `archive.zip`.

Command: bash /tmp/archive_dir/inflate.sh /tmp/archive_dir /tmp/inflate_out /tmp/list.txt

```bash
# 1. Clone the upstream challenge repo and check out main.
git clone https://github.com/commaai/comma_video_compression_challenge.git
cd comma_video_compression_challenge

# 2. Clone our submission packet to get the runtime tree.
git clone https://github.com/adpena/comma-lab.git /tmp/comma-lab
cd /tmp/comma-lab && git checkout b392343d758aba0d3595dd18609f9ca8a8af3e1b && cd -
RUNTIME=/tmp/comma-lab/experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir

# 3. Download our archive.zip (SHA-256 6bae0201fb08...).
curl -L -o /tmp/archive.zip https://github.com/adpena/comma_video_compression_challenge/releases/download/fec6-frontier-submission-20260520/archive.zip
shasum -a 256 /tmp/archive.zip  # expect 6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf

# 4. Stage the runtime tree alongside the extracted archive member.
#    archive.zip contains only member `x`.
mkdir -p /tmp/archive_dir /tmp/inflate_out
unzip -d /tmp/archive_dir /tmp/archive.zip   # extracts /tmp/archive_dir/x only
cp -r "$RUNTIME"/inflate.sh "$RUNTIME"/inflate.py "$RUNTIME"/src /tmp/archive_dir/
ls /tmp/archive_dir   # expect: inflate.py  inflate.sh  src/  x
echo "0.mkv" > /tmp/list.txt
bash /tmp/archive_dir/inflate.sh /tmp/archive_dir /tmp/inflate_out /tmp/list.txt

# 5. Score via upstream evaluate.py on the CPU axis.
python evaluate.py \
  --submission-dir /tmp/inflate_out \
  --uncompressed-dir videos \
  --video-names-file public_test_video_names.txt \
  --device cpu \
  --report /tmp/cpu_report.txt
cat /tmp/cpu_report.txt  # expect Final score: 0.19 (precise: 0.192051)

# 6. Optional: score CUDA on a T4 host with the same archive bytes.
python evaluate.py \
  --submission-dir /tmp/inflate_out \
  --uncompressed-dir videos \
  --video-names-file public_test_video_names.txt \
  --device cuda \
  --report /tmp/cuda_report.txt
cat /tmp/cuda_report.txt  # expect Final score: 0.23 (precise: 0.226210 on Modal Tesla T4)
```

Dependency closure: `torch` plus `brotli`. No other Python packages or shared libraries are loaded at inflate time. No scorer weights are loaded at inflate time per the strict scorer rule.

## Rate term

`25 * 178517 / 37545489 = 0.11886714273451066...`. The full archive byte count is charged to the rate term; there are no out-of-archive sidecars, no Git LFS pointers, and no environment-resident assets that affect any scored value.

## Archive grammar

`archive.zip` is a deterministic ZIP containing a single member `x` (178,417 bytes, stored uncompressed; mtime pinned to the canonical epoch so the archive is byte-stable across rebuilds from the same source). The runtime tree (`inflate.sh`, `inflate.py`, `src/codec.py`, `src/codec_sidecar.py`, `src/frame_selector.py`, `src/model.py`) lives alongside `archive.zip` in the submission directory per the upstream contract; the runtime tree is not inside the ZIP. The rate term is charged against `archive.zip` file size (178,517 bytes) per `upstream/evaluate.py` L63.

The `x` payload is structured as `FP11 + source_len + source_pr101_payload + selector_len + selector_payload`:

1. **`FP11` magic + length prefixes**: local FEC6 wrapper grammar, not inherited from PR #101.
2. **PR #101 source payload** (`source_pr101_payload`, 178,158 bytes): contains the HNeRV state-dict at FP11 + the latent sidecar, both inside PR #101's Brotli envelope. Parsed by `src/codec.py` + `src/codec_sidecar.py`.
3. **FEC6 selector payload** (`selector_payload`, 249 bytes = 6-byte header + 243-byte fixed-Huffman bitstream; new, not in PR #101): 600 selector indices, packed via the fixed-Huffman-k=16 table. The FEC6 selector is byte-appended to the source payload and is not further Brotli-compressed.

Section offsets and lengths are declared in the wrapper headers; the parser in `inflate.py` does not depend on any out-of-archive manifest.

## Build provenance

The training and archive-build harness lived in `comma-lab` (linked above). Final stages used Modal A100 for the main fit pass and Modal Tesla T4 for paired CUDA verification. The submission archive `6bae0201...` was rebuilt deterministically from a clean checkout before lockdown; the rebuild produced the same SHA-256 byte-for-byte.

## Limitations

- Single-video, contest-runtime target. FEC6 is selected per-frame against the upstream scorer on the contest video and is not claimed to generalize.
- The `report.txt` preamble embeds an absolute path string. This is the verbatim format `upstream/evaluate.py` writes; it does not affect any scored value and is left as emitted for parity with prior medal-class submissions.
- CPU and CUDA scores are presented as separate observations on 1:1 contest-compliant hardware; we do not extrapolate the mechanism behind the split.
- `pre_submission_compliance_check.py --contest-final --strict` passes against this packet when invoked with the canonical flag set (`--auth-eval-json` + `--contest-cpu-auth-eval-json` + `--hosted-archive-manifest-json` + `--runtime-equivalence-proof-json` + `--expected-lane-id` + `--expected-job-id` + `--competitive-or-innovative-statement-file`). Runtime equivalence is proof-backed by full-frame byte-identity (`d1afc583...`) across the auth-eval and submission runtime trees, and paired Modal terminal lane/job evidence is recorded in `.omx/state/active_lane_dispatch_claims.md` for both `lane_pr101_fec6_paired_pre_submission_20260519_contest_cpu` and `lane_pr101_fec6_paired_pre_submission_20260519_contest_cuda`.

## Cross-links

- [github.com/adpena/tac](https://github.com/adpena/tac): task-aware compression library used during training (MIT, CI-green).
- [github.com/adpena/comma-lab](https://github.com/adpena/comma-lab): lab notebook, including FEC6 source files and the archive-build harness used at lockdown.

## Acknowledgements

Thanks to @YassineYousfi for keeping the leaderboard open and clarifying the late-submission rubric ([PR #108 closure](https://github.com/commaai/comma_video_compression_challenge/pull/108)). Thanks to @AaronLeslie138, @EthanYangTW, @BradyMeighan, @SajayR, and @rem2: the HNeRV decoder used here originates in @AaronLeslie138's PR #95 and is reused byte-identically by @SajayR's PR #101. This submission is the smallest credible bolt-on we could write on top of the substrate they collectively established.
