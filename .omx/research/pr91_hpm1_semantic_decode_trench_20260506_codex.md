# PR91 HPM1 Semantic Decode Trench - 2026-05-06

Scope: PR91/HPM1 semantic decode recovery only. No GPU work, no remote
dispatch, no lane claim, no score claim.

## Artifact

Machine-readable report:
`experiments/results/pr91_hpm1_semantic_decode_trench_20260506_codex/semantic_decode_trench.json`

Report SHA-256:
`1409494e25b0fef56b0e1ed0c043f295fa7fcebc4eaa5c2c71b563cdcb65f7da`

Command:

```text
.venv/bin/python tools/audit_pr91_hpm1_semantic_decode_trench.py --json-out experiments/results/pr91_hpm1_semantic_decode_trench_20260506_codex/semantic_decode_trench.json
```

## Finding

The embedded PR91 HPAC model is now semantically inventoried, not just treated
as opaque bytes:

- Compressed HPM1 HPAC bytes: `28243`
- Compressed HPM1 HPAC SHA-256:
  `de7638c531c9dafa06148602cf784bf3ae9997f326f85cc25b9f3646b536abdd`
- PPMd-decompressed torch state bytes: `55856`
- PPMd-decompressed torch state SHA-256:
  `0710789f72200a640c1f3b2f6bc0d2f037affaa086f73b2c9e5034e006e80ac1`
- Packed state tensor count: `34`
- Reconstructed `HPACMini` state tensor count: `25`
- First probability-row probe: `passed_probability_row_inventory`
- First raw softmax row SHA-256:
  `c670f106abff43976aa5e102858d39248158c00371a57f0478d09bb176280534`
- Source-contract normalized row SHA-256:
  `e0d10a91f0b9b42283aebbb3bde4d618950d8145c056c6e9d5801b10e0359cc9`

This is useful semantic progress but not decode parity. The local CPU prefix
decode still fails closed:

- Failure: `hpac_entropy_decode_contract_mismatch`
- Location: frame `0`, group `10`, symbol-in-group `191`
- Decoded symbols before failure: `5951`
- Group start decoded symbols: `5760`
- Probability variant: `source_float64_perfect_false`

## Dispatch Boundary

`semantic_decode_trench.json` records:

- `score_claim=false`
- `dispatch_allowed=false`
- `dispatch_performed=false`
- `ready_for_exact_eval_dispatch=false`
- `gpu_or_remote_work=false`

Remaining blockers:

- `full_hpm1_decode_600_frames_not_proven`
- `byte_exact_hpm1_reencode_not_proven`
- `runtime_hpm1_loader_sidecar_free_not_proven`
- `prefix_decode_failed:hpac_entropy_decode_contract_mismatch`
- `exact_cuda_auth_eval_not_allowed_without_local_parity`

## Verification

Focused verification:

```text
.venv/bin/python -m pytest src/tac/tests/test_pr91_hpm1_codec.py -q
.venv/bin/ruff check src/tac/pr91_hpm1_codec.py src/tac/tests/test_pr91_hpm1_codec.py tools/audit_pr91_hpm1_semantic_decode_trench.py
.venv/bin/python -m py_compile src/tac/pr91_hpm1_codec.py src/tac/tests/test_pr91_hpm1_codec.py tools/audit_pr91_hpm1_semantic_decode_trench.py
```

Results:

- `13 passed`
- Ruff passed
- Py compile passed

## Next Required Work

- Repair the HPAC probability/range contract past the frame-0 group-10 failure.
- Decode all 600 HPM1 frames from the exact PR91 token stream on CPU.
- Record decoded mask tensor SHA-256 and context-window traces.
- Re-encode the decoded symbols back to the exact token-stream SHA-256.
- Prove charged runtime HPM1 loading without sidecars or fallback before any
  lane claim or exact CUDA auth eval.
