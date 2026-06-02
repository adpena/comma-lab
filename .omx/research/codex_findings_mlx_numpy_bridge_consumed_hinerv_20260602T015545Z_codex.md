# Codex Findings: MLX NumPy Bridge Consumed By HiNeRV Export

UTC: 2026-06-02T01:55:45Z

Scope: MLX-first substrate export portability for HiNeRV/SNeRV/HPRC-class train/export/archive lanes.

## Finding

The previous HiNeRV MLX portability contract correctly recorded that the exported state was NumPy-array backed, but the archive exporter still passed the in-memory `model.export_state_dict()` directly into the HIV1 packer. That left the canonical `state_dict -> npz -> numpy primitives` bridge as metadata rather than the actually consumed export path.

## Landing

- Added manifest-backed NPZ bridge helpers to `tac.framework_agnostic.helpers`:
  - sorted tensor names;
  - numeric, object-free, finite tensor validation;
  - artifact SHA-256;
  - per-tensor shape, dtype, byte count, SHA-256, and finite status;
  - fail-closed false-authority flags.
- Wired `export_hi_nerv_mlx_archive()` to:
  - write `hi_nerv_mlx_exported_state.npz`;
  - write `hi_nerv_mlx_exported_state_npz_manifest.json`;
  - reload the NPZ through `npz_to_numpy_primitives()`;
  - pack the HIV1 archive from the reloaded NumPy bridge, not from the in-memory state.
- Updated HiNeRV runtime/candidate manifests to carry the consumed NPZ bridge SHA.
- Updated the HiNeRV portability contract from "NumPy export exists but canonical bridge unused" to "canonical bridge used; Torch receiver still blocks pure NumPy inflate."

## Authority

This improves deterministic custody and portability only. It does not make HiNeRV a contest score claim and does not make the current receiver pure NumPy. The remaining portability blocker is explicit: `inflate_runtime_not_pure_numpy`.

## Verification

Focused verification passed:

```text
61 passed, 2 skipped
```

Suite:

```text
src/tac/framework_agnostic/tests/test_framework_agnostic.py
src/tac/local_acceleration/tests/test_mlx_numpy_portability_contract.py
src/tac/substrates/hi_nerv/tests/test_hi_nerv_mlx_renderer_and_archive_candidate.py
```

## Next Engineering Hooks

- Move SNeRV export through the same NPZ bridge before archive packing.
- Add a pure NumPy HiNeRV receiver only if the byte/runtime budget beats the Torch receiver contract.
- Use this bridge manifest in queue promotion gates so MLX train/export/archive lanes cannot claim portability without consumed artifact SHA identity.
