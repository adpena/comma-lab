# Categorical/HPM1 refresh - 2026-05-07

Scope: local-only rebuild and readiness refresh for the categorical/OpenPilot/HPM1
stacking lane. No GPU dispatch, no lane claim, and no score claim.

## Why This Refresh Was Needed

The older candidate at
`experiments/results/categorical_openpilot_payload_candidate_20260506_codex/`
was byte-closed, but a fresh readiness audit against current `main` reported
stale-artifact blockers:

- `label_prior_payload_manifest_typed_label_atoms_mismatch`
- `candidate_construction_plan_typed_label_atoms_mismatch`
- `runtime_loader_parity_runtime_consumer_sha256_mismatch`
- `runtime_loader_parity_source_loader_sha256_mismatch`

Those were artifact drift, not the real scientific blocker. I rebuilt into a
new dated directory instead of mutating the older result.

## New Current-Code Candidate

Builder:

```text
.venv/bin/python tools/build_categorical_candidate_payload.py
  --out-dir experiments/results/categorical_openpilot_payload_candidate_refresh_20260507_codex
  --payload-source pr91-hpm1-mask
  --source-archive experiments/results/public_pr91_intake_20260504_codex/archive.zip
```

Artifacts:

- Summary:
  `experiments/results/categorical_openpilot_payload_candidate_refresh_20260507_codex/summary.json`
- Readiness audit:
  `experiments/results/categorical_openpilot_payload_candidate_refresh_20260507_codex/readiness_audit_refresh.json`
- Archive bytes: `179979`
- Archive SHA-256:
  `9bfea530158ab498a55ec626804c5e9eb1bb80da14a2f2d21d7262c1841bc2fe`
- Charged HPM1 payload SHA-256:
  `a4ed57ff0af1d8c914f004de165aeead50ec8dd61e99b0afdfbfa2d1e7fd9fcc`
- Runtime consumer SHA-256:
  `a4e45905e6555ae41f3c322779a5e0c5dc40c39fef02d47903421225bb4887b3`

The archive is below the active local rate-only A++ floor of `185578` bytes,
but this is not a score claim because the stream changes semantics and has no
full decode/reencode proof.

## Remaining Real Blockers

Fresh readiness blockers on the rebuilt candidate:

- `no_op_control_not_passed:decode_reencode_identity_control`
- `runtime_loader_parity_semantic_runtime_output_parity_not_proven`
- `runtime_execution_proof_hpm1_full_decode_reencode_parity_not_proven`
- `decode_reencode_parity_not_passed`
- `decode_reencode_full_decode_not_proven`
- `decode_reencode_byte_exact_reencode_not_proven`
- `decode_reencode_independent_proof_full_decode_not_proven`
- `decode_reencode_independent_proof_byte_exact_reencode_not_proven`
- `exact_eval_dispatch_requirements_missing`

PR91/HPM1 readiness with the entropy-failure probe attached still blocks on:

- `full_hpm1_decode_600_frames`
- `byte_exact_hpm1_reencode`
- `runtime_hpm1_loader_without_sidecars`
- `exact_cuda_auth_eval_after_parity`

The concrete HPM1 failure row remains:

```text
frame=0, group=10, symbol_in_group=191,
decoded_symbol_count_before_failure=5951
```

The next implementation target is therefore not archive packing. It is the
semantic HPAC probability/range grammar that maps the exact HPM1 uint32 token
queue to all `600 x 384 x 512` class tokens and re-emits the same token bytes.

## Dispatch Boundary

Do not exact-eval this categorical packet yet. It needs:

1. Full 600-frame HPM1 semantic decode.
2. Byte-exact HPM1 reencode to the original HPM1 payload SHA.
3. Sidecar-free runtime loader proof.
4. Active Level-2 dispatch claim.
5. Exact CUDA auth eval through `archive.zip -> inflate.sh -> upstream/evaluate.py`.

## Hardened Readiness Refresh

After the initial refresh, the categorical readiness gate was tightened so
runtime-loader parity requires explicit semantic runtime output parity, not
just sidecar-free execution of the archived runtime consumer. Decode/reencode
independent proofs also have to match the decoded-mask SHA they claim to prove.

Rebuilt hardened artifacts:

```text
.venv/bin/python tools/build_categorical_candidate_payload.py
  --out-dir experiments/results/categorical_openpilot_payload_candidate_hardened_20260507_codex
  --payload-source pr91-hpm1-mask
  --source-archive experiments/results/public_pr91_intake_20260504_codex/archive.zip
```

Hardened audit artifacts:

- `experiments/results/categorical_openpilot_payload_candidate_hardened_20260507_codex/readiness_audit_hardened.json`
- `experiments/results/categorical_hpm1_gate_hardened_20260507_codex/categorical_readiness_hardened.json`
- `experiments/results/categorical_hpm1_gate_hardened_20260507_codex/pr91_hpm1_readiness_with_failure_probe.json`

The archive remains byte-identical to the refreshed build:

- Archive bytes: `179979`
- Archive SHA-256:
  `9bfea530158ab498a55ec626804c5e9eb1bb80da14a2f2d21d7262c1841bc2fe`
- Runtime output SHA-256:
  `b5deeb218b4a96e51f005d15aea430fad7ff4bd69095efa12511bc375ed663c8`

The stricter readiness result is still `ready_for_exact_eval_dispatch=false`.
This is the intended fail-closed state: categorical/HPM1 remains a promising
rate-axis packet, but it is blocked until semantic HPM1 decode/reencode parity
is proven and wired into the runtime without sidecars.
