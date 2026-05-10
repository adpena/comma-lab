# A1 Sidecar Dispatch-Readiness Hardening - 2026-05-09

<!-- generated_at: 2026-05-10T03:38:00Z -->
<!-- evidence_grade: guardrail_hardening; no score claim; no dispatch -->

## Scope

Codex patched the A1 per-pair latent sidecar builder after fresh read-only
adversarial review found that a locally complete manifest could theoretically
self-promote if textual dispatch blockers were absent.

No remote job, GPU job, exact eval, or dispatch claim was launched.

## Bug Classes Closed

1. **Structured dispatch custody required.** `ready_for_exact_eval_dispatch`
   now requires machine-readable `dispatch_claim` and `exact_eval_preflight`
   records bound to the candidate archive SHA-256 and runtime-tree SHA-256.
   Textual blocker strings are no longer the only dispatch-custody surface.

2. **Exact inflate.sh smoke required.** Runtime smoke evidence must declare
   `runtime_surface=inflate_sh_exact_signature`. The current fast import-only
   smoke remains useful as local sanity evidence, but cannot satisfy exact-eval
   dispatch readiness.

3. **No-op proof must include runtime-consumed output change.** The no-op
   detector still proves sidecar byte change, but readiness now also requires
   baseline/candidate runtime-output SHA-256s and
   `runtime_output_changed=true`.

## Tests

```bash
.venv/bin/python -m py_compile \
  tools/build_a1_per_pair_latent_correction_sidecar.py \
  src/tac/tests/test_build_a1_per_pair_latent_correction_sidecar.py
.venv/bin/python -m pytest \
  src/tac/tests/test_build_a1_per_pair_latent_correction_sidecar.py \
  src/tac/tests/test_a1_sidecar_builder_hardening.py -q
```

Observed: `37 passed in 0.72s`.

## Operational Status

The local resumable sidecar artifact remains non-dispatchable. At the time of
this patch, the local search was continuing toward `528/600`; that process was
started before this stricter readiness code and must be classified by terminal
manifest plus post-hoc guard checks.

Even after `600/600`, exact-eval dispatch remains blocked until legacy pairs
`0..335` are rechecked or otherwise gain scalar-equivalent per-pair provenance.
