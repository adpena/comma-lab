# Z3 T4 Smoke Manifest Contract And Byte Result (2026-05-15)

## Dispatch

- lane_id: `lane_z3_balle_hyperprior_bolton_recover_20260514`
- instance_job_id: `substrate_z3_balle_hyperprior_bolton_modal_t4_dispatch_20260515T040905Z__smoke__20ep`
- Modal call_id: `fc-01KRMX6A5DPHYXG77JYQZ6RBPQ`
- mounted git head: `71c7f7740407be532d0ddb31b49e2efebd4ffec7`
- GPU: T4
- elapsed: 11s
- terminal claim: `completed_modal_training_recovered_no_score_claim`
- score_claim: false
- promotion_eligible: false

## Result

The remote training process completed with returncode 0 and emitted a
research-only archive:

- archive member bytes: `178897`
- archive.zip bytes: `178997`
- archive.zip SHA-256: `d513007a8cc2f77da1a8bc31c37cb09c65fa9defca405b4b09857f33e2bc2109`
- final proxy loss: `0.0002648475638125092`
- final rate bits: `3182.02587890625`
- Z3HP1 sidecar bytes: `735`

The archive is not score-lowering. Its own contract reports
`byte_saving=false` because the v1 layout appends a Z3HP1 diagnostic trailer
to A1:

```text
append-only Z3HP1 trailer adds bytes to A1 and cannot realize predicted byte
savings; replace A1 latent_blob with a Z3-coded latent section before marking
this archive byte-saving or exact-eval-ready
```

## Validator Finding

`tools/run_modal_smoke_before_full.py` rejected the smoke under
`training_artifact_v1` because the Z3 trainer wrote `stats.json` but not the
required current-run `manifest.json`.

Classification:

- training infrastructure: green
- smoke validator: correct red
- score movement: none
- next engineering action: emit `manifest.json` and rerun only if testing the
  gate; for score movement, prioritize Z3 v2 latent replacement.

## Fix

`experiments/train_substrate_z3_balle_hyperprior_bolton.py` now writes
`manifest.json` next to `stats.json` and `archive.zip` with:

- `schema=training_artifact_v1`
- `training_mode=smoke|full`
- `research_only=true` for smoke and false-authority full outputs
- all score/promotion/rank/readiness flags false
- archive member and zip bytes/SHA matching the emitted archive

Focused tests:

```bash
.venv/bin/python -m pytest \
  src/tac/substrates/z3_balle_hyperprior_bolton/tests/test_z3_substrate.py::test_smoke_main_emits_training_artifact_manifest \
  src/tac/substrates/z3_balle_hyperprior_bolton/tests/test_z3_substrate.py::test_full_main_cpu_builds_a1_fallback_runtime_without_auth_eval \
  src/tac/tests/test_run_modal_smoke_before_full.py::test_training_artifact_contract_accepts_current_false_authority_manifest -q
```

Result: `3 passed`.

