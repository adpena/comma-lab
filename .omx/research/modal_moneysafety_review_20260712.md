# V9 CGauge CUDA Modal money-safety recursive review (2026-07-12)

Status: **SOURCE PATH PASS; PAID FIRE DEFERRED**

Authority: pre-paid-launch money-safety review for Task 438 / Task 381. This
receipt is training-advisory and makes no contest score claim. No `--execute`,
`modal run`, `modal volume put`, GPU dispatch, or provider mutation was run.
The live MLX run at
`experiments/results/v9_cgauge_432_coherent_arm_20260711` was not touched.

Implementation commit: `213b117ba6` (`fix(modal): hard-cap V9 CGauge timing smoke spend`)

Pre-implementation contract commit: `57d62beca1`

Pre-existing volume-name correction: `b601e1a749`

## Executive verdict

The audited repository path now refuses any V9 CGauge timing smoke that is not
an exact `H100!`, typed as a 3000-epoch curriculum with a separate stop after
at most three completed epochs, enclosed by a 1500-second child budget and an
1800-second invocation-wide hard timeout. Its conservative source plan is
`$3.296256`, with an internal `$5.00` refusal threshold, versus the Task 381
`$20` envelope.

This does **not** make `$5` a provider-enforced all-in budget. Image-cache
state, provider account/budget/IAM, exact remote asset presence, and persistent
Volume retention remain external state. The repository is also currently dirty
from unrelated in-flight work and the deployment version/wheel checklist is not
sealed. Therefore the paid-fire verdict is **DEFER**.

## Ranked findings and fixes

| Rank | Finding and original exposure | Money-risk estimate | Fix | Commit |
|---:|---|---:|---|---|
| 1 | The V9 child accepted `--timeout-hours 10`; the Modal module also exposed a static 14-hour H100 endpoint. A hang, import regression, or detached orphan could consume either full ceiling. | 10-hour H100+4 CPU+32 GiB: **$43.935840** at current listed unit rates. Static 14-hour exposure: **$61.510176**. Both exceed the $20 envelope. | V9 child fixed at 1500 s; invocation, local CLI, and absolute cross-preemption deadline fixed at 1800 s; provider preflight fixed at CPU-only 600 s; retries explicitly zero; hard CPU/RAM resource tuples; no static GPU endpoint remains. | `213b117ba6` |
| 2 | GPU selection and spend gates were too late and duplicate labels could evade label-scoped exclusion. A bad parser/config/cache or concurrent launcher could allocate one or more H100s before useful work. | Up to the old **$43.94 per accidental 10-hour call**, multiplied by duplicate calls. | Zero-provider local parser/scorer/GT/app-definition preflight; bounded same-image CPU preflight; lane-global outer lock and active-claim gate; dynamic GPU attachment only after the CPU receipt and claim; fresh-output and resume-lineage gates. | `213b117ba6` |
| 3 | Execution could implicitly stage the 5,078,017,610-byte GT cache with `modal volume put`, coupling egress/storage mutation to the fire path without a staging-specific timeout, retention policy, or authorization. | 4.729272 GiB retained at $0.09/GiB-month is about **$0.425635/month** before any free-tier effect; egress/quota state was unmeasured. The larger risk was paying GPU startup after an unreviewed staging lifecycle. | The plan may display `asset_stage_argv` as advisory, but execution never runs it. Missing exact SHA-addressed bytes now fail closed; staging is a separate operator-authorized transaction. Volume lookup uses `create_if_missing=False`. | `213b117ba6` |
| 4 | Plain Modal `H100` permits automatic H200 substitution, invalidating the requested H100 timing evidence even if the dollar model remained conservative. | Up to the full smoke cost for scientifically unusable hardware evidence; under the new repository cap, at most **$3.296256 planned**. | Exact provider selector `H100!` is required end to end; plain `H100`, aliases, and generic constructed V9 calls refuse. | `213b117ba6` |
| 5 | `WITNESS_EPOCHS=3` was duration-honest but scientific-config dishonest: it ran three epochs, yet changed the typed curriculum horizon/config hash from V9's 3000-epoch vehicle. Resume comparisons could become unusable. | Not an unbounded-runtime bug: the old path did **not** secretly run 3000 epochs. Risk was paying for a non-comparable three-epoch receipt. | Preserve `--epochs 3000` as typed horizon and thread `--stop-after-epochs 3` as runtime control. Fresh runs execute epochs 1-3; resumed runs execute at most the next three, never beyond 3000. Zero-work resume refuses. | `213b117ba6` |
| 6 | Detached `.spawn()` custody had a process-death window and broad harvesting could miss or touch the wrong call. | Orphan exposure was the old hard timeout; post-fix residual exposure is bounded by **1800 s**. | Extract and register the exact call ID immediately after spawn; cancel best-effort if extraction or ledger registration fails; print exact call-ID-scoped harvest and emergency-cancel commands. The invocation timeout remains the billing backstop if the local CLI dies. | `213b117ba6` |
| 7 | A timing smoke could finish optimizer work without leaving a faithful checkpoint before its throughput receipt, making paid timing evidence non-resumable or ambiguous. | At most the full smoke cost for a receipt that could not be reproduced. | Every stop-limited completed epoch atomically refreshes the canonical resume checkpoint and EMA/Polyak companion before buffered `training_throughput_epoch` telemetry is flushed; throughput distinguishes attempted and successful updates. | `213b117ba6` |

No dead trainer flag was found. Every flag emitted by
`scripts/remote_v9_cgauge_cuda.sh` exists in the trainer parser, including
`--gt-cache`, `--num-pairs`, `--epochs`, `--out-dir`, `--device`,
`--compile-probe`, `--resume-from`, and the new strict custody/runtime flags.
Regression coverage compares the shell and parser surfaces.

## Timeout ceiling and cost math

There is no admissible measured H100 V9 seconds-per-epoch receipt yet. A
performance-derived timeout would therefore be false precision. The fixed
1500-second child ceiling is explicitly a **preregistered, unmeasured budget
hypothesis**, not a throughput estimate:

```text
3 epochs * 300 s/epoch acceptance budget + 600 s startup/compile = 1500 s child
1500 s child + 300 s kill/finalization margin = 1800 s invocation
```

The shell gives the trainer 1440 seconds before `TERM` and 30 seconds after
`TERM` before `KILL`; remaining time is reserved for validation and durable
receipt handling. After the first real receipt, a later plan should replace
the epoch acceptance budget with measured high-quantile timing plus measured
compile/checkpoint headroom. This audit does not fabricate that measurement.

At Modal's listed H100 rate of `$0.001097/s`, CPU rate of
`$0.0000131/core/s`, and memory rate of `$0.00000222/GiB/s`:

```text
H100, 1800 s                              $1.974600
4 CPU + 32 GiB, 1800 s                    $0.222192
CPU-only preflight, 4 CPU + 32 GiB, 600 s $0.074064
listed function-compute maximum            $2.270856

conservative H100 ceiling, $5/h * 0.5 h   $2.500000
CPU/RAM for GPU invocation                 $0.222192
CPU/RAM preflight                          $0.074064
conservative function compute              $2.796256
budgeted image/staging allowance           $0.500000
conservative plan total                    $3.296256
internal repository refusal maximum        $5.000000
```

The `$3.296256` plan total is the post-fix maximum represented by this source
path. The allowance is not measured or enforced by Modal. Image construction,
Volume retention, provider budget/IAM, and deliberate credentialed calls are
outside that source-only ceiling; this is why the final verdict remains DEFER.

## Epoch-honesty verdict

**Old path:** `WITNESS_EPOCHS=3` flowed to shell `--epochs 3`, trainer
argparse, and the trainer loop. It meant three epochs, not the full 3000. The
catastrophic epoch-ignore hypothesis is falsified.

**Problem:** `--epochs 3` also changed V9's typed scientific horizon and config
identity. It did not provide a faithful three-epoch slice of the 3000-epoch
vehicle.

**New path:** environment carries `WITNESS_EPOCHS=3000` and
`WITNESS_STOP_AFTER_EPOCHS=3`; shell carries `--epochs 3000
--stop-after-epochs 3`; trainer validates the runtime window before CUDA. A
fresh run completes exactly epochs 1, 2, and 3. An explicit, SHA-bound resume
completes at most three additional epochs. The runtime stop is not inserted
into the scientific config hash.

## Fail-fast and receipt verdict

**PASS, with one unavoidable bounded GPU boundary.** Before provider contact,
the launcher validates source custody, app-definition/resource structure,
trainer argparse, all real flags, GT SHA/schema/count/geometry, scorer hashes
and CPU construction, runtime epoch window, throughput-receipt path, and the
absence of output mutation. Before H100 allocation, a 600-second same-image
CPU function rechecks imports, mounts, remote GT custody, trainer preflight,
and dispatch nonce/receipt. Output/resume collisions and duplicate claims
refuse before spawn.

CUDA driver discovery, CUDA forward parity, and Triton/compile adoptability
necessarily require a GPU. They now run immediately after allocation, before
structured prefit/training, and fail loudly inside the 1800-second backstop.
The trainer emits durable `training_throughput_epoch` rows, but only after the
corresponding faithful checkpoint is on disk.

## Detach, harvest, and kill verdict

`.spawn()` is retained because the run must survive local CLI disconnect.
Immediately after spawn, the code obtains the call ID and registers it in the
canonical ledger before convenience metadata writes. Registration/extraction
failure triggers best-effort cancellation. The spawned invocation has retries
zero, a provider timeout of 1800 seconds, and an immutable absolute deadline
passed into restarted/preempted attempts so the billing window cannot reset.

The remaining process-death sliver between provider spawn and call-ID
registration cannot be made atomic across the provider/local process boundary.
Its hard backstop is 1800 seconds. After a successful spawn, use only the exact
commands printed by `modal_train_lane.py`:

```text
.venv/bin/python tools/harvest_modal_calls.py --from-ledger --call-id <exact-call-id> --execute
.venv/bin/python -c "import modal; modal.FunctionCall.from_id('<exact-call-id>').cancel(terminate_containers=True)"
```

Harvest is required within 24 hours per the repository HARVEST-OR-LOSE
contract. The timeout is the automatic hang backstop; the exact cancel command
is the operator kill path.

## Recursive clean-pass seal

Any finding reset the counter. The final stable source hashes were independently
re-read after the exact-H100 and no-implicit-staging fixes.

1. Clean pass 1: integrated focused suite **224 passed**; Ruff, `bash -n`,
   `py_compile`, and `git diff --check` passed.
2. Clean pass 2: the same integrated focused suite **224 passed** with no new
   finding.
3. Clean pass 3: independent adversarial source/cost/alias/receipt/checkpoint/
   cancellation harness passed; its scoped suite reported **136 passed** and
   stable hashes. No new repository-native money path was found.

Two broader-suite failures were classified as unrelated stale guard tests: a
Modal event-enum test omits `pre_spawn_fatal`, and a Catalog 245 scanner test
self-matches its own example. Neither touches this five-surface money path.

## Residual blockers before any paid fire

1. Verify the authenticated Modal workspace, current rates, remaining budget
   or provider hard limit, and least-privilege identity. Source metadata is not
   provider IAM.
2. Verify a cache hit/build receipt for the exact image. A natural cache miss
   is outside function timeout, and the image still includes mutable installer
   inputs such as `latest`.
3. Verify the exact remote asset exists at
   `/modal_results/assets/v9_cgauge/gt_cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6.npz`
   with 5,078,017,610 bytes. If absent, keep DEFER: staging needs a separate
   operator-authorized, costed, timeout-bounded lifecycle. Do not fold
   `modal volume put` into the fire command.
4. Establish an output/checkpoint retention or cleanup receipt; persistent
   Volume size and TTL are not yet measured.
5. Make `main` clean after sibling work lands. The launcher refuses a dirty
   tree.
6. Resolve the deployment-version checklist: `pyproject.toml` remains
   `0.2.0rc1` and no rebuilt-wheel receipt accompanies the changed `src/tac`,
   or obtain an explicit source-mount exception from the operator.

Provider references checked 2026-07-12: Modal
[pricing](https://modal.com/pricing),
[resource requests and limits](https://modal.com/docs/guide/resources),
[timeouts](https://modal.com/docs/guide/timeouts),
[GPU exact-selection semantics](https://modal.com/docs/guide/gpu),
[image caching](https://modal.com/docs/guide/images),
[Volumes](https://modal.com/docs/guide/volumes), and
[preemption](https://modal.com/docs/guide/preemption). These references support
the unit rates and provider-boundary cautions above; repository tests remain the
authority for the implemented refusal path.

## Exact conditional safe-fire command

**Do not run this while the DEFER blockers above remain.** It performs a local
plan-only compile first, binds execution to that exact plan hash, then invokes
the paid route. It deliberately contains no asset-staging command.

```bash
GT_SHA=cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6
LABEL=v9-cgauge-cuda-438-smoke-20260712-r2
PLAN_FILE="$(mktemp)"

PYTHONPATH=src .venv/bin/python tools/launch_witness_cloud.py \
  --provider modal \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --gt-cache-sha256 "$GT_SHA" \
  --label "$LABEL" \
  --gpu 'H100!' \
  --epochs 3000 \
  --num-pairs 600 \
  --output "$PLAN_FILE"

PLAN_SHA="$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["plan_sha256"])' "$PLAN_FILE")"
rm -f "$PLAN_FILE"

PYTHONPATH=src .venv/bin/python tools/launch_witness_cloud.py \
  --provider modal \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --gt-cache-sha256 "$GT_SHA" \
  --label "$LABEL" \
  --gpu 'H100!' \
  --epochs 3000 \
  --num-pairs 600 \
  --expected-plan-sha256 "$PLAN_SHA" \
  --execute \
  --operator-go-token GO-CLOUD-381
```

Final decision: **DEFER.** The source path is money-safe enough for a
conditional fire after the six external/deployment gates close; it is not
authorized to fire in the current state.
