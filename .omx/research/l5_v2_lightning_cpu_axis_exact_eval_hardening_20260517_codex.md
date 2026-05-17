# L5 v2 Lightning CPU-axis exact-eval hardening

Recorded: 2026-05-17T06:20:40Z

## Context

The TT5L side-info effect curve requires paired `[contest-CPU]` and
`[contest-CUDA]` exact-eval cells for each variant. The existing Lightning
alternate-provider path was CUDA-only: `launch_lightning_batch_job.py
exact-eval` had no explicit eval-axis flag, `make_exact_eval_spec()` always
created `exact_cuda_eval`, the generated command always passed `--device cuda`
and `INFLATE_REQUIRE_CUDA=1`, and adjudication always emitted
`contest_cuda_*` / `[contest-CUDA]` fields.

That was a real no-signal-loss risk: relaxing only the required device would
let a CPU artifact carry CUDA-shaped adjudication fields. This patch makes the
axis explicit instead.

## Landed Changes

- Added explicit Lightning exact-eval axis support via `--eval-device {cuda,cpu}`
  on `scripts/launch_lightning_batch_job.py exact-eval`; default remains
  `cuda`.
- Preserved the existing CUDA path as `exact_cuda_eval`, including CUDA runner
  preflight, DALI preflight, hash-pinned DALI bootstrap, `INFLATE_REQUIRE_CUDA=1`,
  and CUDA-only component trace.
- Added a separate `exact_cpu_eval` role that runs `experiments/contest_auth_eval.py
  --device cpu`, requires a Linux x86_64 CPU runner preflight, omits DALI/CUDA
  runner preflights, and rejects `component_trace`.
- Made `scripts/adjudicate_contest_auth_eval.py` axis-aware:
  - CUDA remains `contest_cuda_*` / `[contest-CUDA]`.
  - CPU emits `contest_cpu_*` / `[contest-CPU]`.
  - CPU adjudication is never `promotion_eligible` or rank/kill eligible.
- Made Lightning artifact validation accept CPU exact-eval artifacts only when
  they are labeled `contest_cpu`, carry CPU adjudication fields, omit DALI
  requirements, and remain non-promotional.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_lightning_batch_jobs.py src/tac/tests/test_adjudicate_contest_auth_eval_policy.py -q`
  - Result: `135 passed`.
- `.venv/bin/python -m pytest src/tac/tests/test_contest_auth_eval.py src/tac/tests/test_auth_eval_records.py -q`
  - Result: `64 passed`.
- `.venv/bin/python -m ruff check scripts/adjudicate_contest_auth_eval.py scripts/launch_lightning_batch_job.py src/tac/deploy/lightning/batch_jobs.py src/tac/tests/test_lightning_batch_jobs.py src/tac/tests/test_adjudicate_contest_auth_eval_policy.py`
  - Result: `All checks passed!`.
- `.venv/bin/python -m py_compile src/tac/deploy/lightning/batch_jobs.py scripts/launch_lightning_batch_job.py scripts/adjudicate_contest_auth_eval.py src/tac/tests/test_lightning_batch_jobs.py src/tac/tests/test_adjudicate_contest_auth_eval_policy.py`
  - Result: pass.

## Score And Dispatch Status

No provider job was launched by this patch. No score movement is claimed.

This is dispatch plumbing hardening for the TT5L/L5-v2 staircase: the next
non-dry-run Lightning exact-eval wave can now create explicit CPU-axis jobs
without laundering CPU results into CUDA-shaped custody.

