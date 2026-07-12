# V9 CGauge Modal money-safety implementation spec (2026-07-12)

Status: implementation contract for the pre-paid-launch audit. No provider
mutation or paid dispatch is authorized by this document.

## Scope and ownership

The audited money path is exactly:

1. `tools/launch_witness_cloud.py`
2. `src/tac/deploy/witness_cloud_launcher.py`
3. `scripts/remote_v9_cgauge_cuda.sh`
4. `experiments/modal_train_lane.py`
5. `experiments/train_levelset_witness_realized_through_R_torch.py`

Tests may be added or changed only for these contracts. The live MLX run at
`experiments/results/v9_cgauge_432_coherent_arm_20260711` is out of scope and
must not be read, written, signalled, or used as evidence.

## Required behavior

### Plan and pre-spend boundary

- The V9 timing-smoke plan keeps the typed scientific horizon at 3000 epochs
  and carries a separate runtime stop of exactly three additional completed
  epochs. The runtime stop must not enter the typed DSL/config hash.
- The child lane timeout is 1500 seconds: an explicitly conservative,
  unmeasured budget hypothesis of `3 * 300 s/epoch + 600 s startup/compile`.
  It is a spending backstop, not a throughput estimate.
- The Modal invocation-wide hard timeout is 1800 seconds, reserving 300 seconds
  for timeout handling and volume/receipt finalization. The shell gives the
  trainer 1440 seconds before TERM and 30 seconds after TERM before KILL; the
  remaining wrapper time is for fail-closed validation and receipt handling.
  The local `modal run` process is also bounded to the same 1800-second plan
  envelope.
- Planning records a conservative H100 GPU rate ceiling of $5/hour, hard
  request/limits of four CPU cores and 32 GiB memory, the current listed
  CPU/memory unit rates, a bounded 600-second same-image CPU preflight, and a
  $0.50 image/staging allowance. The resulting plan ceiling is $3.296256 and
  must fit an explicit plan-bound maximum of at most $5.00. The Task 381 $20
  envelope is not treated as permission to spend $20 on this smoke.
- Every provider workload derives its resource ceiling from GPU, hard CPU/RAM
  requests, and timeout. The declaration must cover that derived amount and
  both values must be at most $5. Modal's exact benchmark request `H100!`
  (which disables automatic H200 substitution) is reserved to the canonical
  Task438 V9 driver; plain `H100`, generic, and constructed-wrapper requests
  cannot produce this timing smoke.
- The plan rejects unsupported GPU names, unsafe labels, non-finite/nonpositive
  timeouts or budgets, stop counts outside `1..3`, invalid SHA-256 values, and
  resume paths outside normalized `/modal_results/**` custody.
- Labels are strict slugs and cannot contain traversal, comma, equals, slash,
  whitespace, or control characters. Staged GT assets are SHA-addressed.
- `execution_allowed` is false without exact GT SHA custody. Execution requires
  the expected plan SHA, a clean main worktree, exact local GT hash agreement,
  and all validation before asset staging.
- The destination results volume is looked up with `create_if_missing=False`.
  The SHA-addressed GT object is reused only after a read-only size/name check;
  execution refuses without provider mutation when it is absent. Asset staging
  is a separate operator-authorized lifecycle and is never implicit in the GPU
  fire command. Asset lookup ambiguity or provider read failure refuses.
  Forced/ignored image-cache environment modes refuse rather than creating an
  unplanned rebuild.
- The Modal dispatch carries `--require-clean-head` and a reviewed sentinel
  list for the remote driver, Torch trainer, V9 DSL, CUDA throughput/controller,
  and CUDA model runtime. Unsupported/stale source must refuse before spawn.
- Before the H100 claim/spawn, the same image, mounts, driver, staged GT asset,
  and trainer import/argparse surface run through a bounded CPU-only preflight.
  It validates GT SHA/schema/count/geometry, imports, arguments, paths, and
  resume intent without creating the training output. A non-green preflight
  refuses before any GPU allocation.
- Before any provider read or 4.7-GiB staging attempt, a zero-provider local
  preflight validates the actual Modal SDK/app-definition AST, trainer parser,
  GT schema/geometry, scorer hashes, CPU scorer construction, epoch-window
  receipt, output intent, and source sentinels. The provider-side CPU preflight
  is the sole statically exported endpoint; it has no GPU and a 600-second hard
  timeout. The GPU resource is attached dynamically only after local checks,
  provider CPU receipt, and the lane-global claim succeed.
- A pre-dispatch harvest field is an exact-call-ID template, never the broad
  `--from-ledger --execute` command that can touch unrelated calls.

### Remote and trainer fail-fast behavior

- The remote shell validates integer epoch/pair/stop values, explicit resume
  existence, fresh-output versus explicit-resume intent, GT SHA, and durable
  normalized paths before invoking the trainer.
- Resume custody includes the exact checkpoint SHA-256. A populated output may
  resume only from a checkpoint inside that same output lineage; a cross-label
  resume must target a fresh output. Full storage preflight precedes hashing,
  and the large GT object is hashed only once on the GPU path.
- Every shell-emitted trainer flag exists in trainer argparse; a regression
  compares the two surfaces.
- The trainer validates arguments, GT-cache keys/count/geometry, and explicit
  resume-path existence before CUDA model construction or compile/autotune.
- CUDA forward parity failure and compile-probe non-adoptability raise
  immediately. They cannot continue into structured prefit, scorer loading, or
  regional compilation.
- A fresh timing smoke executes epochs 1 through 3. A resumed smoke executes at
  most three additional epochs, bounded by the unchanged 3000-epoch typed
  horizon. A completed checkpoint refuses instead of emitting a false completed
  zero-work receipt.
- Every successfully completed epoch in a runtime-limited smoke atomically
  refreshes the canonical full resume checkpoint before the next epoch starts.
  Epoch-complete throughput/controller receipts are emitted only after that
  faithful post-controller checkpoint and its EMA companion have landed. The
  runtime-stop epoch also writes a preserved stage checkpoint. The final result says
  `runtime_epoch_budget_reached` when the typed horizon is not complete.
- Throughput rows distinguish attempted from successful optimizer updates and
  use successful updates for productive updates/second. The rows remain
  durable, advisory, and non-promotable.

### Detach custody and duplicate-spend behavior

- An already-active claim with the same lane and label refuses a second spawn.
- The active-claim and local execution locks are lane-global, not bypassable by
  changing the output label.
- Invalid timeout values refuse rather than silently becoming a paid 60-second
  call.
- `.spawn()` is retained. The invocation-wide Modal timeout is the hard billing
  backstop if the local CLI dies.
- Because provider preemption can restart an input and provider timeouts are
  per attempt, one absolute billing deadline is computed before `.spawn()` and
  threaded into every worker attempt. A restarted attempt refuses after that
  deadline and caps its child runtime to the remaining wall-clock envelope;
  retries are explicitly zero. The 30-minute budget cannot reset per attempt.
- Call-ID extraction and canonical ledger registration occur immediately after
  spawn, before sentinel/metadata writes. If call-ID extraction or registration
  fails, the just-created call is cancelled best-effort and the process fails
  loudly; otherwise the canonical ledger is already sufficient to harvest.
- Post-spawn output prints exact call-ID-scoped recover and harvest commands.
- The provider-account ability to define a different paid function is an
  external IAM/budget boundary. Repository-native dispatches fail closed, but
  source code cannot make deliberately authorized arbitrary SDK calls
  impossible; provider budget configuration remains an operator gate.

## Acceptance tests

- No test or verification command may contact Modal or stage provider assets.
- Focused pytest covers plan determinism, unsafe-input refusal, cost/timeout
  arithmetic, SHA-addressed custody, stop-after resume math, compile fail-fast,
  zero-work refusal, flag parity, duplicate-claim refusal, dynamic hard timeout,
  and call-ID registration ordering.
- `bash -n scripts/remote_v9_cgauge_cuda.sh`, focused pytest, `ruff check` on
  touched Python, `python -m py_compile` on touched Python, and `git diff
  --check` pass.
- After fixes, perform three consecutive recursive source-review passes. Any
  new finding resets the clean-pass counter.
- Every Python file is marked reviewed before serializer commits. Commits use
  post-edit content SHA values and do not use `REVIEW_GATE_OVERRIDE`.

## Verdict rule

`PROCEED-TO-FIRE` is allowed only if all acceptance tests and three clean passes
hold, the repository is clean on `main`, and the exact plan hash/custody digest
are supplied. Otherwise the receipt verdict is `DEFER`; a fail-closed command
that would refuse before spending is not itself a successful fire clearance.
