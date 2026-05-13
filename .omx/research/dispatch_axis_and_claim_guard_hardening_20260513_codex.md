# Dispatch Axis + Claim Guard Hardening - Codex 2026-05-13

## Scope

Recursive adversarial bug-hunter follow-up for two P0 integration findings from
the 2026-05-13 frontier substrate landings:

- exact-CUDA evidence parser divergence between `experiments/contest_auth_eval.py`
  and `tools/auth_eval_records.py`;
- malformed lane IDs admitted by `tools/claim_lane_dispatch.py`.

## Fixes

1. `tac.device_axis_eval.is_contest_cuda_equivalent_gpu(...)` is now the shared
   GPU-family classifier for contest-CUDA-equivalent auth evals. The accepted
   CUDA evidence family is T4, A100, 4090, H100, A10G, and L40S. The helper
   deliberately classifies only GPU family; callers still own device,
   sample-count, and platform checks.

2. `experiments/contest_auth_eval.py` and `tools/auth_eval_records.py` now use
   the same GPU-family helper. This closes the silent demotion where A100/4090/
   H100 exact-CUDA evidence could be emitted as `[contest-CUDA]` by auth eval but
   later parsed as non-contest CUDA by downstream records tooling.

3. `tools/claim_lane_dispatch.py` now rejects placeholder lane IDs such as `0`,
   pure numerics, `modal`, `unknown`, `none`, and other non-canonical values
   before any row is written. This prevents a repeat of the live 2026-05-13
   malformed terminal row with `lane_id=0`.

4. Summary mode no longer pulls the repository default dispatch-claim archive
   when a caller passes a non-default `--claims-path`. That keeps test and
   temporary ledgers isolated while preserving default live+archive summary
   behavior for the real repository ledger.

## Verification

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_device_axis_eval.py \
  src/tac/tests/test_auth_eval_records.py \
  src/tac/tests/test_contest_auth_eval.py \
  src/tac/tests/test_claim_lane_dispatch.py -q
```

Result: `92 passed`.

## Status

No remote GPU dispatch was launched by this hardening pass. These are guardrail
and evidence-custody fixes only.
