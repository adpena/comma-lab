# CodecOp Entropy-Floor Ledger Adapter - Worker D - 2026-05-07

Supersession note: Worker G subsequently hardened the same adapter with
explicit `score_affecting_payload_changed=false`,
`charged_bits_changed=false`, generated artifact custody, and
cross-paradigm ledger output. See
`.omx/research/codec_op_entropy_floor_ledger_adapter_20260507_worker_g.md`
and commit `1f1ef027`.

## Scope

Worker D owned the Joint-ADMM <-> CodecOp adapter / meta-Lagrangian integration
slice. The goal was to keep entropy-floor evidence from becoming an orphaned
lane by converting it into explicit planning atoms while preserving the
contest-custody rule that CPU/model-class floors are not exact-eval dispatch
readiness.

## Implementation

Added `tools/codec_op_entropy_floor_to_ledger.py`.

Supported inputs:

- `reports/pr101_provable_optimal_floor.json`
  - schema `pr101_compression_floor_ladder.v2`
  - emits one atom per `provable_floors[]` row.
- `reports/pr101_context_transform_floor_probe.json`
  - schema `pr101_context_transform_floor_probe.v1`
  - emits one atom per transform and entropy model (`iid`, `markov1`,
    `markov2`).

Each atom carries:

- `family`
- `family_group=joint_admm_codec_op_entropy_floor`
- `pareto_scope`
- `byte_delta`
- `confidence`
- `evidence_grade=<source_grade>_planning`
- `proxy_row=true`
- `score_claim=false`
- `dispatchable=false`
- `ready_for_exact_eval_dispatch=false`
- `dispatch_blockers` including exact CUDA, byte-closed manifest,
  materialized CodecOp, and no-op/round-trip proof requirements.

The adapter deliberately refuses source readiness flags. If a source report
claims `ready_for_exact_eval_dispatch=true`, `score_claim=true`,
`charged_bits_changed=true`, or `score_affecting_payload_changed=true`, the
emitted atom remains planning-only and records an explicit refusal blocker.

## Smoke Evidence

Real-report smoke command:

```bash
.venv/bin/python tools/codec_op_entropy_floor_to_ledger.py \
  --input reports/pr101_provable_optimal_floor.json \
  --input reports/pr101_context_transform_floor_probe.json \
  --output /tmp/pact_worker_d_codec_op_entropy_floor_atoms.jsonl \
  --atoms-json-output /tmp/pact_worker_d_codec_op_entropy_floor_atoms.json \
  --summary-output /tmp/pact_worker_d_codec_op_entropy_floor_summary.json
```

Result:

- 26 planning atom rows emitted.
- 15 rows have negative `byte_delta`.
- Summary has `ready_for_exact_eval_dispatch=false`.

Cross-paradigm/meta-Lagrangian smoke command:

```bash
.venv/bin/python tools/build_cross_paradigm_atom_ledger.py \
  --base-pose-dist 0.000003389640351909648 \
  --source worker_d_codec_op_entropy_floor_tmp_smoke \
  --atoms-json /tmp/pact_worker_d_codec_op_entropy_floor_atoms.json \
  --json-out /tmp/pact_worker_d_codec_op_entropy_floor_cross_paradigm_ledger.json
```

Result:

- `atom_count=26`
- `proxy_row_count=26`
- `pareto_eligible_count=0`
- `ready_for_exact_eval_dispatch=false`

Largest negative planning row observed in the smoke:

- `codec_op_entropy_floor:pr101_context_transform_floor_probe:delta_mod255:markov2`
- `byte_delta=-107115`
- `proxy_row=true`
- `pareto_eligible=false`
- `ready_for_exact_eval_dispatch=false`

This is a high-EV planning target, not a dispatch target. The next materialized
work is a CodecOp bitstream and decoder contract that actually consumes the
delta/context stream and proves byte-closed archive custody.

## Tests

Focused test added:

- `src/tac/tests/test_codec_op_entropy_floor_to_ledger.py`

Checks:

- PR101 provable-floor rows carry required atom fields and negative byte deltas.
- Context-transform rows stay dispatch-refused even if the source report
  incorrectly sets readiness or score-claim flags.
- CLI writes JSONL, JSON-list, and summary outputs.
- Rows survive `build_atom_ledger()` with `proxy_row=true`,
  `pareto_eligible=false`, `dispatchable=false`, and exact-dispatch blockers.

Verification run:

```bash
ruff check \
  tools/codec_op_entropy_floor_to_ledger.py \
  src/tac/tests/test_codec_op_entropy_floor_to_ledger.py

PYTHONPATH=src UV_CACHE_DIR=/tmp/uv-cache-pact-worker-d \
  uv run --no-project --python .venv/bin/python \
  --with pytest --with pytest-timeout \
  python -m pytest src/tac/tests/test_codec_op_entropy_floor_to_ledger.py -q
```

Observed:

- Ruff: all checks passed.
- Pytest: `3 passed in 0.07s`.

The repo `.venv` lacked `pytest` and `ruff`; the focused pytest run used an
isolated uv environment with the repo's Python 3.12 interpreter to avoid the
project-level `constriction` wheel blocker on Python 3.14 freethreaded.

## Next Patch Plan

1. Materialize the best planning row (`delta_mod255:markov2`) as a real
   CodecOp bitstream with charged model/table bytes.
2. Add a no-op/round-trip decoder proof and byte-closed archive manifest.
3. Re-run the adapter with the materialized report, then allow Pareto eligibility
   only after the proxy/planning blockers are replaced by manifest and exact
   CUDA evidence.
