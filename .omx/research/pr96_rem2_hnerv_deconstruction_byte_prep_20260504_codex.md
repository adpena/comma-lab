# PR96 rem2_HNeRV Deconstruction And Byte-Optimization Prep - 2026-05-04

## Scope

Ownership for this pass was PR96 deconstruction and byte-optimization prep only.
No remote GPU job was launched. No score claim is made here; all candidate
archives below are byte-only artifacts until raw-output parity and exact CUDA
auth eval prove the scored behavior.

## Inputs

- Source archive:
  `experiments/results/leaderboard_intel_20260504_codex/pr96_archive.zip`
- Source bytes: `186631`
- Source SHA-256:
  `2ecbd2118bebdb5566f719ed538a89c4608ccab19c9edc7ae7a6de778bd42b46`
- Runtime:
  `experiments/results/leaderboard_intel_20260504_codex/pr96_runtime/inflate.py`
- Runtime SHA-256:
  `940e57ce0b6532db4aeb38b644bb7de75ccefdb29b92d00f8ddacaa3603829b2`
- Runtime shell wrapper:
  `experiments/results/leaderboard_intel_20260504_codex/pr96_runtime/inflate.sh`
- Runtime wrapper SHA-256:
  `202e9d917c1a2f56ece729193de92c32f11a23ec341d3a44cf5cac1d4a8b19a7`

## Archive Contract

`zipinfo -v` and the PR96 profiler agree on three source members:

| member | method | compressed | uncompressed | SHA-256 |
|---|---:|---:|---:|---|
| `decoder.bin` | deflated | `169272` | `169242` | `940d2328ad5384c99b51872cc5f87b6153befa20c10c1c2564b08ba469c97868` |
| `latents.bin` | deflated | `16133` | `16920` | `be4ed381db136baf114701fe6a9bba4f604dac18e79244c9f657383578a6ff71` |
| `p` | stored | `930` | `930` | `bf04a4e2dd69ca32e3b1bd1a3c64481d7f6930096b552d49d175eec8768d1c43` |

Static AST read-set extraction on `inflate.py` found only:

- `decoder.bin`
- `latents.bin`

The visible runtime does not read `p`, does not enumerate `archive_dir`, and has
no dynamic archive read detected by the profiler. It chooses CUDA if available,
loads `decoder.bin`/`latents.bin`, decodes `600` latent rows with `latent_dim=28`,
bicubic-upsamples to the camera size, and writes raw RGB.

## Decoder And Latent Structure

`decoder.bin` is `pr96_rem2_hnerv_decoder_v1`:

- header: `br_len=45658`, `hist_len=805`, `meta_len=102`,
  `lengths_len=20`, `comp_id=2`
- brotli decoder records: `24`
- brotli-record quantized bytes: `52270`
- range-coded records: `4`
- range-coded names and coded bytes:
  - `stem.weight`: `31920`
  - `blocks.0.weight`: `32420`
  - `blocks.1.weight`: `32632`
  - `blocks.2.weight`: `25668`
- histogram codec: brotli
- histogram raw bytes: `2048`
- range-coded payload bytes consumed: `122640`

`latents.bin` is `pr96_rem2_hnerv_latents_v1`:

- rows: `600`
- dims: `28`
- fp16 mins bytes: `56`
- fp16 scales bytes: `56`
- quantized bytes: `16800`

## Tooling Added

- `experiments/profile_pr96_rem2_hnerv_packing.py`
- `src/tac/tests/test_profile_pr96_rem2_hnerv_packing.py`

The tool:

- validates ZIP names and duplicate-member absence
- validates local-header and central-directory member name agreement
- parses PR96 decoder and latent payload structure without scorer loads
- extracts the runtime archive read set from AST
- chooses deterministic ZIP storage per member
- can raw-copy an existing source deflate stream when it is smaller than local
  recompression
- emits byte-only candidate archives and a JSON manifest
- records `score_claim=false` and `requires_exact_cuda_eval=true`

## Emitted Artifacts

Profiler command:

```bash
.venv/bin/python experiments/profile_pr96_rem2_hnerv_packing.py \
  --archive experiments/results/leaderboard_intel_20260504_codex/pr96_archive.zip \
  --runtime-py experiments/results/leaderboard_intel_20260504_codex/pr96_runtime/inflate.py \
  --output-dir experiments/results/pr96_rem2_hnerv_packing_profile_20260504_codex
```

Primary manifest:

- `experiments/results/pr96_rem2_hnerv_packing_profile_20260504_codex/profile_pr96_rem2_hnerv_packing.json`
- SHA-256:
  `075dfd7a3a780d51f848539d27e590e75f61ec5e38f312addd4e97d175adee61`

Candidate 1, member-preserving lossless ZIP repack:

- path:
  `experiments/results/pr96_rem2_hnerv_packing_profile_20260504_codex/archive.pr96_member_preserving_repack.zip`
- bytes: `185682`
- SHA-256:
  `615b5125314ee5c8c55f34a21766c7e6835767796ff490fd4c59a91586fb2f9b`
- byte delta vs source: `-949`
- plan:
  - `decoder.bin`: store uncompressed, `169242` bytes
  - `latents.bin`: source raw deflate copy, `16133` bytes
  - `p`: deflate level 4, `11` bytes
- kept all source members and verified extracted member payload identity.

Candidate 2, drop statically unused `p`:

- path:
  `experiments/results/pr96_rem2_hnerv_packing_profile_20260504_codex/archive.pr96_drop_unused_repack.zip`
- bytes: `185593`
- SHA-256:
  `b1d8ee040f462d5623d77ed5941a01933c0d88c93a5f3664bf25a7363a85b779`
- byte delta vs source: `-1038`
- plan:
  - `decoder.bin`: store uncompressed, `169242` bytes
  - `latents.bin`: source raw deflate copy, `16133` bytes
  - removed `p`
- Kept-member payload identity was verified, but this candidate still requires
  raw-output parity or exact CUDA auth eval before treating it as behavior
  preserving.

Generic byte profile artifacts:

- `experiments/results/pr96_rem2_hnerv_packing_profile_20260504_codex/archive_byte_profile.json`
- `experiments/results/pr96_rem2_hnerv_packing_profile_20260504_codex/archive_byte_profile.md`

## Verification

Commands run:

```bash
.venv/bin/python -m pytest src/tac/tests/test_profile_pr96_rem2_hnerv_packing.py -q
.venv/bin/python -m py_compile experiments/profile_pr96_rem2_hnerv_packing.py
unzip -t experiments/results/pr96_rem2_hnerv_packing_profile_20260504_codex/archive.pr96_drop_unused_repack.zip
unzip -t experiments/results/pr96_rem2_hnerv_packing_profile_20260504_codex/archive.pr96_member_preserving_repack.zip
.venv/bin/python experiments/profile_archive_bytes.py \
  experiments/results/leaderboard_intel_20260504_codex/pr96_archive.zip \
  experiments/results/pr96_rem2_hnerv_packing_profile_20260504_codex/archive.pr96_drop_unused_repack.zip \
  experiments/results/pr96_rem2_hnerv_packing_profile_20260504_codex/archive.pr96_member_preserving_repack.zip \
  --json-out experiments/results/pr96_rem2_hnerv_packing_profile_20260504_codex/archive_byte_profile.json \
  --markdown-out experiments/results/pr96_rem2_hnerv_packing_profile_20260504_codex/archive_byte_profile.md
```

Results:

- `3 passed in 0.09s`
- both candidate ZIPs passed `unzip -t`
- generic archive byte profile reported all three archives valid, no duplicate
  member names, and no duplicate payload hashes inside each archive
- no scorer load, no local exact eval, and no remote dispatch occurred

## Exact Next Action

If PR96 should be exact-evaled, first let the active public PR96 baseline replay
finish so the repack has an exact same-runtime baseline. Then run a PR96 repack
exact eval on the drop-unused candidate, because it is the smallest byte-closed
candidate:

- archive:
  `experiments/results/pr96_rem2_hnerv_packing_profile_20260504_codex/archive.pr96_drop_unused_repack.zip`
- expected bytes: `185593`
- expected SHA-256:
  `b1d8ee040f462d5623d77ed5941a01933c0d88c93a5f3664bf25a7363a85b779`
- inflate:
  `experiments/results/leaderboard_intel_20260504_codex/pr96_runtime/inflate.sh`
- suggested lane id:
  `public_pr96_rem2_hnerv_repack_t4_20260504`
- suggested job id:
  `exact_eval_public_pr96_rem2_hnerv_repack_t4_20260504TBDZ`

Before submit, claim the lane:

```bash
.venv/bin/python tools/claim_lane_dispatch.py claim \
  --lane-id public_pr96_rem2_hnerv_repack_t4_20260504 \
  --platform lightning \
  --instance-job-id exact_eval_public_pr96_rem2_hnerv_repack_t4_20260504TBDZ \
  --agent codex \
  --predicted-eta-utc <UTC_ISO8601> \
  --status active \
  --notes "PR96 drop-unused-p byte-only repack exact eval; archive b1d8ee040f462d5623d77ed5941a01933c0d88c93a5f3664bf25a7363a85b779 bytes 185593"
```

Then submit through the existing exact-eval Lightning path with the same lane id
and the unchanged external PR96 inflate script. A valid submit must include the
expected archive SHA/bytes and the dispatch lane id; the exact command should
use the operator's current Lightning staging parameters, not hard-coded local
credentials.
