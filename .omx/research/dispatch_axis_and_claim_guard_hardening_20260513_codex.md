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

Follow-up P1 exact-readiness guard:

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_optimizer_exact_readiness.py \
  tests/test_parallel_dispatch_top_k_exact_ready_audit.py -q
```

Result: `24 passed`.

## P1 Follow-Up: Non-Promotional HLM1 Reference

HLM1 exact CUDA remains preserved as a reviewed score-lowering reference:

- score: `0.20638030907530963`
- label: `hnerv_hlm1_fixed_latent_recode_modal_t4_enforced_20260513`
- status: non-promotional result-review input, not an unqualified active
  dispatch frontier

The active promotable score frontier used by exact-readiness and
parallel-dispatch defaults is now HDM4:

- score: `0.20642625334307507`
- label: `pr106_r2_lowlevel_hdm4_candidate_pr101_runtime_cuda_20260513_codex`

This prevents non-promotional HLM1 evidence from silently blocking dispatch of
future candidates that could be promotable between HDM4 and HLM1, while keeping
HLM1 visible as exact-CUDA review signal.

## Status

No remote GPU dispatch was launched by this hardening pass. These are guardrail
and evidence-custody fixes only.
