# Modal PR106 Q10 Runtime Closure Failure (2026-05-10)

## Scope

This ledger records the first Modal T4 exact auth-eval attempt for
`pr106_q10_151byte_brotli` on 2026-05-10. The run failed before
`contest_auth_eval.json` was produced because the uploaded runtime tree was not
closed over its delegated PR106 adapter. This is an infrastructure/runtime
packaging failure, not score evidence and not a method negative.

## Custody

- Lane claim: `pr106_q10_151byte_brotli`
- Failed dispatch job: `pr106_q10_exact_cuda_modal_20260510T173000Z`
- Modal run: `ap-nEk25Ev3rElErAxgnXjnBA`
- Archive:
  `experiments/results/hnerv_lowlevel_repack_pr106_q10_packet_20260507_codex/release_surface/archive.zip`
- Archive bytes: `186088`
- Archive SHA-256:
  `626b1c76d318eaed45198dc26aea7ee98c8a05f685b840356cf5b621bcddeea7`
- Modal artifacts:
  `experiments/results/modal_auth_eval/pr106_q10_exact_cuda_modal_20260510T173000Z/`

## Failure

The local command uploaded only the static release surface as `--submission-dir`.
That release surface contains a wrapper `inflate.sh` which delegates to the
reviewed PR106 adapter:

```text
experiments/public_runtime_adapters/pr106_belt_and_suspenders_adapter/inflate.sh
```

Inside the uploaded Modal runtime tree, the wrapper was no longer inside a repo
checkout, so its fallback `REPO_ROOT` resolved to `/`. Inflate failed before
decode:

```text
FATAL: delegated inflate.sh is missing or not executable:
//experiments/public_runtime_adapters/pr106_belt_and_suspenders_adapter/inflate.sh
```

Wrapper validation correctly reported:

- `passed=false`
- `score_claim=false`
- `promotion_eligible=false`
- `validation_errors=["contest_auth_eval.json was not produced"]`

## Fix

The Modal CUDA image now mounts the canonical runtime dependencies needed by
the PR106 adapter:

- `experiments/public_runtime_adapters`
- `experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/source`

The corrected invocation should not upload the release-surface wrapper. It
should upload only the archive and use the repo-mounted adapter directly:

```bash
.venv/bin/modal run experiments/modal_auth_eval.py \
  --archive experiments/results/hnerv_lowlevel_repack_pr106_q10_packet_20260507_codex/release_surface/archive.zip \
  --inflate-sh experiments/public_runtime_adapters/pr106_belt_and_suspenders_adapter/inflate.sh \
  --output-dir experiments/results/modal_auth_eval/pr106_q10_exact_cuda_modal_20260510T173900Z \
  --gpu T4
```

## Score-Lowering Consequence

Do not classify `pr106_q10_151byte_brotli` until the corrected Modal run
produces an exact CUDA result. The candidate remains a rate-only byte-repack
anchor with local raw-equivalence proof: decoder Brotli raw bytes match source,
latents/sidecar raw bytes match source, archive bytes decrease by `151`, and
expected rate-only score delta is `-0.00010054470192144788`.
