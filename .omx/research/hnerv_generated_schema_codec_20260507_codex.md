# HNeRV Generated-Schema Codec - 2026-05-07 Codex

## Evidence Boundary

- `score_claim=false`.
- `ready_for_exact_eval_dispatch=false`.
- This is a deterministic codec/roundtrip artifact for generated HNeRV schemas,
  not a contest archive or score claim.
- The generated `.hngs` blob is rebuildable binary output and is intentionally
  not promoted as a score artifact. The tracked manifest records byte count and
  SHA-256.

## Code Landed

- `src/tac/hnerv_generated_schema_codec.py`
  - `HNGS` envelope: magic + JSON header + per-tensor Brotli streams.
  - Explicit generated schema fingerprint and per-stream SHA-256 checks.
  - Symmetric int8 quantization with deterministic decode to quantized tensors.
  - Fail-closed parsing for bad magic, truncated header, stream hash mismatch,
    trailing bytes, and shape/schema mismatch.
- `tools/materialize_hnerv_generated_schema_codec.py`
  - Materializes a generated-schema state dict into an `HNGS` blob and compact
    manifest.
  - Runs decode immediately and records roundtrip schema fingerprint.
- `src/tac/tests/test_hnerv_generated_schema_codec.py`
  - Determinism, fail-closed mismatch/tamper, and CLI materialization tests.

## Stage-D Materialization

Command:

```bash
.venv/bin/python tools/materialize_hnerv_generated_schema_codec.py --state-dict experiments/results/hnerv_arch_shrink_training_driver_stage_d_20260507_codex/initial_state_dict.pt --schema-json experiments/results/hnerv_arch_shrink_training_driver_stage_d_20260507_codex/generated_schema.json --output-blob experiments/results/hnerv_arch_shrink_training_driver_stage_d_20260507_codex/generated_schema_codec.hngs --output-manifest experiments/results/hnerv_arch_shrink_training_driver_stage_d_20260507_codex/generated_schema_codec_manifest.json --brotli-quality 11
```

Manifest:

```text
experiments/results/hnerv_arch_shrink_training_driver_stage_d_20260507_codex/generated_schema_codec_manifest.json
```

Measured byte accounting:

- blob bytes: `76,263`
- blob SHA-256:
  `fcd7c6f9556f625f107df80ca39f128c3889450d1653ec5b8552aa11c25a01b9`
- payload bytes: `68,802`
- header bytes: `7,453`
- streams: `28`
- schema fingerprint:
  `4c67e69d5cc4a2324e8429465deaccf13dc4077e490335d1b3756680d76ba220`

## Interpretation

The Stage-D overlap-initialized generated schema is now byte-serializable and
roundtrip-validated at the codec level. This removes the previous
`target_codec_export_for_generated_schema_not_implemented` blocker for local
codec artifacts, but does not remove contest blockers:

- no submission runtime loader for `HNGS`;
- no local inflate output parity;
- no strict packet preflight;
- no lane claim;
- no exact CUDA auth eval.

The byte count is not a performance result because the checkpoint is not
trained. It is a payload-contract proof for the architecture-shrink path.

## Verification

```bash
.venv/bin/python -m pytest src/tac/tests/test_hnerv_generated_schema_codec.py -q
```

Result: `4 passed`.

```bash
.venv/bin/python -m ruff check src/tac/hnerv_generated_schema_codec.py tools/materialize_hnerv_generated_schema_codec.py src/tac/tests/test_hnerv_generated_schema_codec.py
```

Result: `All checks passed`.

## Next Tranche

1. Add a tiny `HNGS` runtime-loader fixture and local inflate parity check.
2. Teach the packet compiler to include an `HNGS` payload as a typed renderer
   section in a non-dispatch smoke packet.
3. Train a generated-schema checkpoint before treating `76,263` bytes as
   anything other than a contract/proof artifact.
