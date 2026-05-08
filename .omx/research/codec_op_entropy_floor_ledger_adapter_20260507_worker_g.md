# CodecOp Entropy-Floor Ledger Adapter - 2026-05-07

## Scope

This tranche promotes the Worker D/G adapter into a durable, repo-local
planning surface. It converts PR101 CodecOp entropy-floor reports into
meta-Lagrangian atom rows while preserving the hard boundary that entropy
floors are not candidate archives, score claims, or dispatch-ready evidence.

Implementation:

- `tools/codec_op_entropy_floor_to_ledger.py`
- `src/tac/tests/test_codec_op_entropy_floor_to_ledger.py`
- artifacts under
  `experiments/results/codec_op_entropy_floor_ledger_20260507_worker_g/`

## Inputs

- `reports/pr101_provable_optimal_floor.json`
  - schema: `pr101_compression_floor_ladder.v2`
  - SHA-256 prefix: `42e4f458157d26d8`
  - atoms: `5`
  - negative byte-delta atoms: `4`
- `reports/pr101_context_transform_floor_probe.json`
  - schema: `pr101_context_transform_floor_probe.v1`
  - SHA-256 prefix: `7053fdb71aec6e69`
  - atoms: `21`
  - negative byte-delta atoms: `11`

## Output

- `codec_op_entropy_floor_atoms.jsonl`: `26` planning-only atom rows
- `codec_op_entropy_floor_atoms.json`: JSON-list copy for cross-paradigm tools
- `codec_op_entropy_floor_summary.json`: fail-closed policy summary
- `codec_op_entropy_floor_ledger.md`: human-readable run ledger

Policy summary:

- planning-only rows: `26`
- proxy rows: `26`
- score-claim rows: `0`
- dispatchable rows: `0`
- exact-eval-ready rows: `0`
- promotion-eligible rows: `0`
- fail-closed: `True`

No training, eval, remote GPU job, lane claim, or archive substitution happened
in this tranche.

## Best Planning Rows

The largest negative byte-delta rows are useful as design targets only:

| atom | byte_delta | target archive estimate | status |
| --- | ---: | ---: | --- |
| `delta_mod255:markov2` | `-107115` | `71029` | planning-only/proxy |
| `identity:markov2` | `-64037` | `114107` | planning-only/proxy |
| `signed_zigzag:markov2` | `-64037` | `114107` | planning-only/proxy |
| `markov2_per_tensor` | `-64037` | `114107` | planning-only/proxy |
| `zero_mask_nonzero_value:markov2` | `-62438` | `115706` | planning-only/proxy |

These rows are lower-bound/router signals. They cannot be dispatched until a
real CodecOp bitstream, charged table/model bytes, decoder roundtrip proof,
byte-closed archive manifest, and exact CUDA auth eval exist.

## Verification

```bash
uv run ruff check tools/codec_op_entropy_floor_to_ledger.py src/tac/tests/test_codec_op_entropy_floor_to_ledger.py
uv run --with pytest python -m pytest src/tac/tests/test_codec_op_entropy_floor_to_ledger.py -q
uv run python tools/codec_op_entropy_floor_to_ledger.py --input reports/pr101_provable_optimal_floor.json --input reports/pr101_context_transform_floor_probe.json --output experiments/results/codec_op_entropy_floor_ledger_20260507_worker_g/codec_op_entropy_floor_atoms.jsonl --atoms-json-output experiments/results/codec_op_entropy_floor_ledger_20260507_worker_g/codec_op_entropy_floor_atoms.json --summary-output experiments/results/codec_op_entropy_floor_ledger_20260507_worker_g/codec_op_entropy_floor_summary.json --ledger-md-output experiments/results/codec_op_entropy_floor_ledger_20260507_worker_g/codec_op_entropy_floor_ledger.md
```

Observed:

- ruff: passed
- pytest: `4 passed`
- adapter run: `26` rows, `15` negative byte-delta rows
