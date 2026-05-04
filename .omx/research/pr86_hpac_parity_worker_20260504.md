# PR86 HPAC Parity Worker - 2026-05-04

Scope: local-only intake hardening for public PR86 HPAC replay/parity. No
remote dispatch, no GPU dispatch, and no score promotion were performed.

## Inputs

- Public PR: `https://github.com/commaai/comma_video_compression_challenge/pull/86`
- Local intake dir:
  `experiments/results/public_pr86_intake_20260504_codex`
- Source archive:
  `experiments/results/public_pr86_intake_20260504_codex/archive.zip`
- Existing forensic artifacts:
  - `pr86_hpac_token_anatomy_forensics.json`
  - `pr86_hpac_full_decode_reencode_gate_20260504_codex.json`
  - `pr86_hpac_pr85_qma9_parity_probe.json`
  - `pr86_view.json`

## Local Custody Facts

- Archive bytes: `207579`
- Archive SHA-256:
  `e67b7c22240dbe33853c19d049b0044a5df16ce5f751ba8f1021cab8ceb03cef`
- Member anatomy:
  - `master.pt.gz`: `31144` bytes,
    SHA-256 `3f3ee2b19ba5cf97017559750c0d64bc422c3f84fedaed1877741ee6c6bd5236`
  - `slave.pt.gz`: `32287` bytes,
    SHA-256 `817294dea0d940a8ef62c190bf96338f5a756930882f0f0d7f4d7c7eb87a82a8`
  - `hpac.pt.ppmd`: `28243` bytes,
    SHA-256 `de7638c531c9dafa06148602cf784bf3ae9997f326f85cc25b9f3646b536abdd`
  - `tokens.bin`: `113900` bytes,
    SHA-256 `14144bde496631f89a02646496bc2e66306bba6da149ddca37e21d85d175f225`
  - `meta.pt`: `1499` bytes,
    SHA-256 `848381e2da1b0f307670174f135a3925c43d8cdc73576b4bf05fadf833de4a08`

All required members are present, ZIP-stored, and the saved forensic artifacts
refer to the same archive identity. This converts PR86 archive custody and
container anatomy into local reproducible evidence.

## Fail-Closed Blocker

Blocker class: `hpac_entropy_decode_contract_mismatch`.

The full PR86 decode/reencode gate fails closed before byte parity:

- Status: `failed_closed`
- Error type: `AssertionError`
- Error:
  `Tried to decode from compressed data that is invalid for the employed entropy model.`
- Failed at: frame `0`, HPAC group `10`, symbol-in-group `191`
- Decoded symbols before failure: `5760`
- Source `tokens.bin`: `113900` bytes,
  SHA-256 `14144bde496631f89a02646496bc2e66306bba6da149ddca37e21d85d175f225`

The PR85/QMA9 HPAC parity probe failed with the same constriction invalid-model
assertion. Therefore PR86 is design evidence and archive-custody evidence, but
not local score evidence or a direct PR85 transplant template.

## Dependency And Code Contract

Observed dependency contract:

- `constriction==0.4.2` in the replay artifact; repo range is
  `constriction>=0.4,<0.5`.
- `pyppmd==1.3.1` in the replay artifact; repo range is `pyppmd>=1.3,<2.0`.
- Replay artifact also records Python `3.12.13`, Torch `2.11.0`, and NumPy
  `2.4.4`.

Faithful replay or port requires:

- `master.pt.gz` and `slave.pt.gz` decoded as gzip-wrapped torch state dicts.
- `hpac.pt.ppmd` decoded with
  `pyppmd.decompress(..., max_order=4, mem_size=16<<20)`, then `torch.load`.
- `tokens.bin` interpreted as same-order little-endian `uint32` constriction
  queue words.
- HPAC decoding with `constriction.stream.queue.RangeDecoder` and
  `constriction.stream.model.Categorical(..., perfect=False)`.
- Probability generation from HPACMini logits, softmax, float probability
  arrays, clip `1e-7`, and renormalize per symbol.
- Submitted archive token semantics treated as raw class tokens, not the
  residual-token training objective, unless a separate byte-identical replay
  proves otherwise.
- Full 600-frame decode and byte-exact decode->encode `tokens.bin` parity
  before any local score claim, PR85 transfer, or dispatch.

## New Guard

Added `experiments/diagnose_pr86_hpac_parity.py`, a read-only-by-default
adjudicator that consumes the existing PR86 artifacts and prints a JSON report.
It does not run inflate, contest eval, GPU work, or remote dispatch.

The focused tests in `src/tac/tests/test_diagnose_pr86_hpac_parity.py` cover:

- Default report classifies the archive as blocked by
  `hpac_entropy_decode_contract_mismatch`.
- Exact archive bytes/SHA and five-member anatomy are preserved.
- Stale artifact identity is detected.
- Unsafe, duplicate, and byte-mismatched archive members fail closed.
- CLI writes only to an explicit `--json-out` path.

Verification run:

- `.venv/bin/python -m py_compile experiments/diagnose_pr86_hpac_parity.py`
  passed.
- `.venv/bin/python -m pytest src/tac/tests/test_diagnose_pr86_hpac_parity.py -q`
  passed: `6 passed`.
- `.venv/bin/python experiments/diagnose_pr86_hpac_parity.py` reported
  `blocked 207579 e67b7c22240dbe33853c19d049b0044a5df16ce5f751ba8f1021cab8ceb03cef hpac_entropy_decode_contract_mismatch`.

## Next Implementation Patch

Build an owned PR86 HPAC replay shim or module with a byte-exact full-stream
parity gate. The patch should extract the HPACMini loader/decode loop,
pin/verify `constriction` and `pyppmd` behavior, and require
`RangeEncoder.get_compressed().tobytes()` to match the submitted `tokens.bin`
SHA before PR86 is used as local score evidence or as a PR85 mask entropy-coder
template.
