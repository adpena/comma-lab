# A1 Sidecar Runtime-Output No-Op Probe - 2026-05-09

<!-- generated_at: 2026-05-10T04:48:00Z -->
<!-- evidence_grade: guardrail_hardening; no score claim; no dispatch -->

## Scope

Codex added a bounded runtime-output no-op probe to the A1 sidecar builder.

No remote job, GPU job, exact eval, or dispatch claim was launched.

## Change

The builder can now compare baseline A1 and candidate decoded output hashes on
the same bounded smoke-pair subset. The probe records:

- baseline archive SHA-256;
- candidate archive SHA-256;
- baseline output SHA-256;
- candidate output SHA-256;
- `runtime_output_changed`;
- nested baseline/candidate runtime-smoke evidence.

The existing import-smoke path now explicitly declares
`runtime_surface=inflate_py_import_smoke`. That proof is sufficient to close
the "bytes changed but output may not have changed" no-op class, but it is not
the exact contest runtime surface. `ready_for_exact_eval_dispatch` still
requires separate `runtime_surface=inflate_sh_exact_signature` evidence.

## Verification

```bash
.venv/bin/python -m py_compile \
  tools/build_a1_per_pair_latent_correction_sidecar.py \
  src/tac/tests/test_build_a1_per_pair_latent_correction_sidecar.py
.venv/bin/python -m pytest \
  src/tac/tests/test_build_a1_per_pair_latent_correction_sidecar.py \
  src/tac/tests/test_a1_sidecar_builder_hardening.py -q
```

Observed: `38 passed in 0.74s`.

## Status

The active A1 sidecar recheck process was started before this patch. Its
terminal manifest should be classified post-hoc with current code, then a
future builder pass can materialize the new runtime-output no-op fields.
