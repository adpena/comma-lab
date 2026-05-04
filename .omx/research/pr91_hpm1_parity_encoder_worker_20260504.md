# PR91 HPM1 Parity And Encoder Worker - 2026-05-04

Scope: PR91/HPM1 parity and encoder/re-encoder reconstruction only. No remote
GPU dispatch, no training, and no exact eval submission were performed.

## Source Artifacts

- PR91 Codex archive:
  `experiments/results/public_pr91_intake_20260504_codex/archive.zip`
- PR91 worker archive:
  `experiments/results/public_pr91_intake_20260504_worker/archive.zip`
- Archive bytes: `222404`
- Archive SHA-256:
  `4c16d04c746c981feb902e4dd508ffadaf3615e532d351993c3d2f6eccda1b4f`
- Worker standalone token stream:
  `experiments/results/public_pr91_intake_20260504_worker/pr91_hpm1_tokens.bin`
- Worker standalone HPAC model:
  `experiments/results/public_pr91_intake_20260504_worker/pr91_hpm1_hpac.pt.ppmd`
- Local preflight artifact:
  `experiments/results/public_pr91_intake_20260504_codex/pr91_hpm1_local_preflight_20260504_codex.json`

## Local Reproduction

Command:

```text
.venv/bin/python experiments/replay_pr91_hpm1_mask.py --archive experiments/results/public_pr91_intake_20260504_codex/archive.zip --max-frames 1 --json-out experiments/results/public_pr91_intake_20260504_codex/pr91_hpm1_local_preflight_20260504_codex.json
```

Result: `failed_closed`, `score_claim=false`, `dispatch_performed=false`.

- Failure stage: `submitted_tokens_decode`
- Failure reason: `hpac_entropy_decode_contract_mismatch`
- Failure error: `AssertionError: Tried to decode from compressed data that is invalid for the employed entropy model.`
- Failed at: frame `0`, group `10`, symbol in group `191`
- Decoded symbols before failure: `5951`
- Blocker class:
  `real_invalid_entropy_or_probability_model_contract_mismatch`

## Contract Findings

Static HPM1 parsing passes:

- Bundle format: `pr85_v5_micro_24bit_lengths_fixed_bias_region`
- HPM1 mask bytes: `145087`
- HPM1 mask SHA-256:
  `a4ed57ff0af1d8c914f004de165aeead50ec8dd61e99b0afdfbfa2d1e7fd9fcc`
- Dimensions: `N=600`, `H=384`, `W=512`
- HPAC params: `P=32`, `delta=2`, `ch=64`, `use_spm=1`,
  `hpac_d_film=8`, `ppmd_order=4`
- Token stream bytes: `116796`
- Token stream SHA-256:
  `541016d83852a5bb3e0738caa3b44d7b2b0f7372f1841085cf9554f039c6cf6b`
- Token stream is uint32-aligned: `29199` words
- HPAC model bytes: `28243`
- HPAC model SHA-256:
  `de7638c531c9dafa06148602cf784bf3ae9997f326f85cc25b9f3646b536abdd`
- HPM1 tail bytes: `0`

Local dependency contract passes and matches the PR86 recorded stack exactly:

- Python: `3.12.13`
- torch: `2.11.0`
- numpy: `2.4.4`
- constriction: `0.4.2`
- pyppmd: `1.3.1`
- constriction queue same-order round-trip self-test: passed

The PR91 pyproject in the worker source still declares `brotli` but not
`constriction` or `pyppmd`; this is an OSS/replay dependency hygiene issue, but
it is not the local decode blocker because the local environment has the pinned
packages and their behavior self-test passes.

## Classification

- Caller misuse: not the root blocker. The earlier `N=1` bounded smoke was
  caller misuse because the submitted model has `frame_embed.weight` for 600
  frames. The corrected local preflight instantiates `N=600` and still fails
  during frame-0 entropy decode.
- Dependency/version mismatch: not supported locally. Installed versions match
  the recorded PR86 dependency contract and the constriction queue self-test
  passes.
- Stream slicing: not supported by current evidence. The PR91 Codex and worker
  archives have identical SHA-256; the extracted token/model lengths and SHAs
  match the worker standalone `pr91_hpm1_tokens.bin` and
  `pr91_hpm1_hpac.pt.ppmd`; HPM1 declared lengths exactly consume the segment.
- Real invalid entropy data or unrecovered probability/model contract:
  currently most likely. The model PPMd decompresses, torch-loads, reconstructs
  into `HPACMini`, and loads with no missing/unexpected keys before constriction
  rejects the submitted token stream under the source PR86
  `float64 + Categorical(..., perfect=False)` contract.

## Encoder/Re-Encoder Investigation

PR91 submitted code contains the HPM1 inflater branch and bundled
`pr86_hpac.py` decoder, but no PR91-local compressor/build recipe for producing
the submitted `HPM1` token stream from PR85 masks.

PR86 source does contain enough encoder shape to reconstruct a local encoder:

- `experiments/results/public_pr86_intake_20260504_merged_refresh/training/hpac.py::encode_frame`
  defines the group-wise constriction encoder.
- `experiments/results/public_pr86_intake_20260504_merged_refresh/training/archive.py::write_tokens`
  writes `tokens.bin` from raw SegNet token maps and an HPAC model.

Implemented local-only prototype surfaces:

- `src/tac/pr91_hpm1_codec.py`
  - parses PR91 `HPM1` as a typed mask segment
  - validates archive/header/token/model/dependency contracts
  - runs prefix/full HPAC decode without scorer loads
  - builds HPM1 segments from explicit token/model bytes
  - provides `prototype_reencode_hpm1_from_raw_tokens(...)`
- `experiments/replay_pr91_hpm1_mask.py`
  - writes deterministic JSON preflight reports
  - can run a local-only raw-token HPM1 re-encode prototype when provided a
    decoded uint8 `N,H,W` token file

The prototype is intentionally not dispatchable. Because PR91's submitted HPM1
stream does not decode under the recovered source contract, byte-exact
decode->re-encode parity and a trustworthy fewer-than-`145087` mask-byte target
remain blocked. A future implementation can use the prototype only after the
entropy contract is recovered or a new locally verified HPAC model/token source
exists.

## Verification

```text
.venv/bin/python -m py_compile src/tac/pr91_hpm1_codec.py experiments/replay_pr91_hpm1_mask.py
.venv/bin/python -m pytest src/tac/tests/test_pr91_hpm1_codec.py
.venv/bin/python -m pytest src/tac/tests/test_pr85_bundle.py src/tac/tests/test_pr91_hpm1_codec.py
```

Results:

- `src/tac/tests/test_pr91_hpm1_codec.py`: `6 passed`
- Combined PR85 bundle + PR91 HPM1 tests: `16 passed`

## Next Dispatch Recommendation

Do not dispatch PR91-derived HPM1 candidates or PR91+side-channel stacks from
local custody yet. The next unblocked local action is probability/model
contract recovery: test full decode and byte-exact re-encode only after a
candidate contract passes the deterministic frame-0 failure point and then the
full 600-frame stream. If exact T4 replay succeeds while this local preflight
continues to fail, classify PR91 as a hardware/runtime contract divergence and
diff the T4 runtime dependency path before promoting any local HPM1-derived
work.

