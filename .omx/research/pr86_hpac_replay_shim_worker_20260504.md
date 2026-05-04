# PR86 HPAC Replay Shim Worker - 2026-05-04

## Scope

Implemented a local-only, fail-closed PR86 HPAC replay/parity shim. This work
does not run remote dispatch, does not run contest eval, and does not make a
score claim.

Owned write surfaces:

- `src/tac/pr86_hpac_codec.py`
- `experiments/replay_pr86_hpac_tokens.py`
- `src/tac/tests/test_pr86_hpac_codec.py`
- `.omx/research/pr86_hpac_replay_shim_worker_20260504.md`

## Inputs

- Archive: `experiments/results/public_pr86_intake_20260504_codex/archive.zip`
- Archive bytes: `207579`
- Archive SHA-256: `e67b7c22240dbe33853c19d049b0044a5df16ce5f751ba8f1021cab8ceb03cef`
- Submitted `tokens.bin` SHA-256: `14144bde496631f89a02646496bc2e66306bba6da149ddca37e21d85d175f225`
- Intake JSONs:
  - `pr86_hpac_full_decode_reencode_gate_20260504_codex.json`
  - `pr86_hpac_token_anatomy_forensics.json`
  - `pr86_hpac_pr85_qma9_parity_probe.json`
  - `pr86_view.json`

## Implementation

`src/tac/pr86_hpac_codec.py` now provides reusable helpers for:

- strict PR86 archive custody and member validation;
- duplicate member, zip-slip, unknown member, non-stored member, and expected
  SHA/byte fail-closed checks;
- gzip plus `torch.load` decode of `master.pt.gz` and `slave.pt.gz`;
- `meta.pt` decode and HPAC config extraction;
- PPMd decode of `hpac.pt.ppmd` with `max_order=4`, `mem_size=16 << 20`;
- HPACMini state reconstruction from PR86 packed `*.weight_q` and
  `*.weight_scale` tensors;
- `constriction.stream.queue.RangeEncoder/RangeDecoder` behavior self-test;
- full HPAC token decode with frame/group/symbol failure coordinates;
- decode-to-reencode byte parity via
  `RangeEncoder.get_compressed().tobytes()`.

`experiments/replay_pr86_hpac_tokens.py` is a thin JSON CLI around the module.
By default it loads the PR86 archive and the four intake JSON artifacts. It
prints JSON and can optionally write it with `--json-out`.

## Verification

Commands run:

```bash
.venv/bin/python -m py_compile src/tac/pr86_hpac_codec.py experiments/replay_pr86_hpac_tokens.py src/tac/tests/test_pr86_hpac_codec.py
.venv/bin/python -m pytest src/tac/tests/test_pr86_hpac_codec.py -q
.venv/bin/python experiments/replay_pr86_hpac_tokens.py
```

Focused pytest result:

- `6 passed, 1 warning in 7.44s`
- Warning was Python `zipfile` reporting the intentionally duplicated
  synthetic `tokens.bin` member in a fail-closed test fixture.

## Real PR86 Replay Result

The default CLI replay validated:

- archive identity matches expected bytes and SHA;
- exact five-member archive contract passed;
- installed dependencies match the recorded intake versions:
  - Python `3.12.13`
  - torch `2.11.0`
  - numpy `2.4.4`
  - constriction `0.4.2`
  - pyppmd `1.3.1`
- constriction queue self-test passed with `uint32` compressed words;
- `master.pt.gz`, `slave.pt.gz`, `meta.pt`, and `hpac.pt.ppmd` decoded;
- HPAC state reconstruction loaded with no missing or unexpected keys.

The submitted `tokens.bin` decode still failed closed before byte parity:

```json
{
  "failure_stage": "submitted_tokens_decode",
  "failure_reason": "hpac_entropy_decode_contract_mismatch",
  "failed_at": {
    "frame": 0,
    "group": 10,
    "symbol_in_group": 191
  },
  "decoded_symbol_count_before_failure_group": 5760,
  "decoded_symbol_count_before_failure": 5951,
  "source_tokens_bytes": 113900,
  "source_tokens_sha256": "14144bde496631f89a02646496bc2e66306bba6da149ddca37e21d85d175f225"
}
```

`byte_parity_achieved=false` and `dispatch_unlocked=false`.

## Next Blocker

Full 600-frame HPAC decode does not reach reencode parity because the submitted
stream asserts under the local PR86 dependency/runtime contract at frame 0,
group 10, symbol 191. The next unblocker is resolving the entropy-model or
runtime-contract mismatch before any PR86-derived dispatch, PR85 transfer, or
score claim can be considered.

