# CodecOp ADMM Adapter - Worker J - 2026-05-07

## Scope

Worker J implemented a planning-only bridge between the canonical CodecOp
encode/decode surface and the Joint-ADMM `StreamProximalCodec` surface.

Owned write scope:

- `src/tac/codec_op_admm_adapter.py`
- `src/tac/tests/test_codec_op_admm_adapter.py`
- `.omx/research/codec_op_admm_adapter_20260507_worker_j.md`

No PR101/PR106 probe files were edited. No GPU dispatch was attempted.

## Contract

The adapter exposes one CodecOp operating point as a Joint-ADMM proximal stream:

- runs CodecOp `encode()` once and caches the result;
- runs CodecOp `decode()` and requires every input tensor key to decode as a
  tensor with the same shape and dtype;
- records deterministic tensor contract entries with key, shape, dtype, numel,
  bytes, and tensor SHA-256;
- records blob SHA-256, op_state SHA-256, op params, op identity, context keys,
  and reconstruction RMS;
- returns a `ProximalStepResult` with the caller-supplied cached
  `score_delta` and `marginal`;
- places the planning row in `ProximalStepResult.state`.

All rows are fail-closed:

- `score_claim=false`
- `dispatchable=false`
- `ready_for_exact_eval_dispatch=false`
- `field_selection_ready_for_exact_eval_dispatch=false`
- `promotion_eligible=false`
- `score_affecting_payload_changed=false`
- `charged_bits_changed=false`
- `exact_cuda_auth_eval=false`
- `archive_sha256=null`
- `archive_bytes=null`

Default blockers include missing byte-closed archive manifest, no archive
substitution, no score-affecting payload-change proof, and missing exact CUDA
auth eval.

## Tests

Focused coverage in `src/tac/tests/test_codec_op_admm_adapter.py` verifies:

- adapter satisfies `StreamProximalCodec`;
- `proximal_step()` returns deterministic CodecOp byte custody in the state row;
- planning rows keep all dispatch and score flags false;
- tensor contract ordering and SHA custody are stable;
- helper `codec_op_to_admm_planning_row()` emits the same fail-closed schema;
- missing decode keys fail closed;
- shape and dtype mismatches fail closed;
- negative cached marginals are rejected before ADMM use;
- CodecOp `validate()` failures fail closed.

## Blockers

This bridge is not a score candidate. Promotion still requires a materialized
byte-closed archive whose scored bytes actually changed, a fresh archive
manifest, and exact CUDA auth eval on the final archive path.
