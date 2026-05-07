# Categorical HPM1 Structural Inventory - 2026-05-06

Evidence grade: HPM1 structural byte inventory
Score claim: false
Dispatch attempted: false
Ready for exact eval dispatch: false

## Context

The byte-closed categorical/openpilot candidate already packaged the public
PR91 `HPM1` mask segment as `categorical_payload.bin`, but its parity blockers
were generic. This tranche adds a deterministic structural inventory for that
payload so the next decode/reencode proof starts from exact section custody
instead of prose.

## Artifact

Output directory:
`experiments/results/categorical_openpilot_payload_candidate_20260506_codex/`

New artifact:

- `hpm1_structural_inventory.json`, 12612 bytes,
  SHA-256 `985d5b944e61208020dbadf3d5a43233a0a6d494bf156010d3f16c09b45a8bba`

Regenerated manifest/readiness artifacts:

- `candidate.json` records `hpm1_structural_decode_inventory`
- `readiness.json` accepts that inventory as structural evidence while keeping
  decode/reencode dispatch blockers active
- `summary.json` includes the structural inventory path and tool-run input file

Archive custody did not change:

- `archive.zip`: 152235 bytes,
  SHA-256 `106af3ed6917d6115586463ef35e43119add5db002fd19f3da7cd8065a63eb8d`
- `categorical_payload.bin`: 145087 bytes,
  SHA-256 `a4ed57ff0af1d8c914f004de165aeead50ec8dd61e99b0afdfbfa2d1e7fd9fcc`

## Structural Decode Inventory

HPM1 section custody:

| section | offset | bytes | sha256 |
|---|---:|---:|---|
| header | 0 | 48 | `e22642f557e19923041914bd2d2de62d9bd29dd667c98480cb92bf6fa53df7e4` |
| tokens | 48 | 116796 | `541016d83852a5bb3e0738caa3b44d7b2b0f7372f1841085cf9554f039c6cf6b` |
| hpac_ppmd_model | 116844 | 28243 | `de7638c531c9dafa06148602cf784bf3ae9997f326f85cc25b9f3646b536abdd` |

Header contract:

- `N=600`, `H=384`, `W=512`, `P=32`, `delta=2`, `ch=64`
- `use_spm=1`, `hpac_d_film=8`, `ppmd_order=4`
- token stream is uint32-aligned: 29199 little-endian words
- decoded geometry target remains `600 x 384 x 512`, 117964800 symbols
- patch grid is `12 x 16`, 94 groups per frame

Structural re-emit proof:

- `structural_reencode.matches_source_segment=true`
- re-emitted HPM1 segment SHA-256:
  `a4ed57ff0af1d8c914f004de165aeead50ec8dd61e99b0afdfbfa2d1e7fd9fcc`
- common prefix with source segment: 145087 bytes

This is only header plus opaque token/model section re-packing. It is not
semantic decode/reencode parity.

## Blocker Manifest

The machine-readable blocker manifest names these unsupported wire constructs:

- `hpac_autoregressive_probability_rows`
- `constriction_range_decoder_uint32_queue_replay`
- `hpac_context_update_order`
- `range_encoder_uint32_reemit`
- `contest_runtime_sidecar_free_hpm1_loader`

Current readiness remains blocked on:

- `no_op_control_not_passed:decode_reencode_identity_control`
- `no_op_control_not_passed:label_permutation_fail_closed_control`
- `no_op_control_not_passed:runtime_consumes_conditioning_control`
- `runtime_loader_parity_not_passed`
- `decode_reencode_parity_not_passed`
- `decode_reencode_full_decode_not_proven`
- `decode_reencode_byte_exact_reencode_not_proven`

## Next Proof

The next rigorous step is not another archive wrapper. It is a CPU-only HPM1
semantic replay proof that loads the embedded HPAC model, decodes all 600
frames from the exact token stream, records the decoded token tensor SHA-256,
then range-encodes those semantic tokens back to the exact token stream and
HPM1 segment SHA-256. Runtime sidecar-free loading remains a separate gate.

## Verification

Commands run before this ledger entry:

```text
.venv/bin/python -m py_compile src/tac/hpm1_payload_structure.py src/tac/categorical_payload_candidate.py src/tac/categorical_candidate_readiness.py tools/build_categorical_candidate_payload.py tools/audit_categorical_candidate_readiness.py
.venv/bin/python -m pytest src/tac/tests/test_hpm1_payload_structure.py src/tac/tests/test_build_categorical_candidate_payload.py src/tac/tests/test_categorical_candidate_readiness.py -q
.venv/bin/python -m ruff check src/tac/hpm1_payload_structure.py src/tac/categorical_payload_candidate.py src/tac/categorical_candidate_readiness.py src/tac/tests/test_hpm1_payload_structure.py src/tac/tests/test_build_categorical_candidate_payload.py src/tac/tests/test_categorical_candidate_readiness.py tools/build_categorical_candidate_payload.py tools/audit_categorical_candidate_readiness.py tools/materialize_comma_lab_public_export.py src/tac/tests/test_materialize_comma_lab_public_export.py
.venv/bin/python tools/build_categorical_candidate_payload.py --out-dir experiments/results/categorical_openpilot_payload_candidate_20260506_codex --payload-source pr91-hpm1-mask
```

Results:

- focused pytest: 27 passed, 1 expected duplicate-ZIP warning
- ruff: passed
- regenerated candidate: `ready_for_exact_eval_dispatch=false`,
  `score_claim=false`, `dispatch_attempted=false`

No GPU, remote dispatch, lane claim, or score claim happened.

## Worker B Supersession Note - 2026-05-07

The HPM1 payload bytes and semantic blockers are unchanged, but the candidate
archive was regenerated after adding the charged label-prior payload manifest.
The structural inventory was therefore re-emitted with the new candidate
archive reference.

Updated artifact custody:

- `archive.zip`: 160400 bytes, SHA-256
  `3455c82708b1d628e17fb21cf3ccb334a4375e023a80217681c10912224881ac`
- `hpm1_structural_inventory.json`: 12612 bytes, SHA-256
  `453095759e19defd3e8f0b1011495c9b50d6b60dffe904b3c31fe10348d15e37`
- `categorical_payload.bin`: 145087 bytes, SHA-256
  `a4ed57ff0af1d8c914f004de165aeead50ec8dd61e99b0afdfbfa2d1e7fd9fcc`

The inventory remains structural only: it does not prove full semantic
decode/re-encode parity, runtime output parity, exact CUDA auth eval readiness,
or any score claim.
