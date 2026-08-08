# ddm_wc1 findings

## Verdict

WC1 completed CPU-only prep for the lifted PR130 semantic-renderer wall-clock bench. No Metal job,
scorer slot, archive build, or live-run mutation was performed by this arm.

| deliverable | status | evidence |
|---|---|---|
| Throughput audit | DONE | table below; scoped to `experiments/ddm_mx1_pr130_semantic_renderer.py` and `src/tac/pr130_lift/mlx_semantic_renderer.py` |
| Default-off trainer levers | BUILT | `--perf-thread-pin`, `--train-compute-dtype {fp32,bf16,fp16}`, `--compile-train-loss`; existing `--microbatch-pairs` used for the batched variant |
| Fire-guard binding | BUILT | `tools/mx1_fire_guard.py` now compares the throughput flags against the mem-probe receipt, defaulting old tickets to off/fp32 |
| Resume mem-probe | FIXED | `mem-probe` now advances from `resume_step + mem_probe_steps`, so resume benches collect post-resume telemetry |
| Bench script | READY, NOT FIRED | `experiments/ddm_wc1_wallclock_bench.py`; plan-only run wrote ticket + planned receipts |
| Fast CPU tests | PASSED | `12 passed in 1.07s` for WC1 + mx1 fire-guard focused tests |

## Throughput Audit

| lever | present before WC1? | applicable to lifted conv vehicle? | expected multiplier / receipt basis | risk to gradient fidelity | WC1 action |
|---|---|---|---|---|---|
| 1-thread training pin | NO in this driver argv/receipt | YES; process env + torch thread pool only | 2.96x MEASURED on the witness line per charter recall; must be re-measured here | Low math risk; may change CPU scheduling and host contention, not decode bytes | Added `--perf-thread-pin one`; bench variant also sets OMP/MKL/OpenBLAS/vecLib/NumExpr/MLX env to 1 |
| Batched pair forward | PARTIAL: `--microbatch-pairs` existed, GPU default was 4-pair serial accumulation | YES; n32 full-batch variant is the real throughput question | 2-4x banked receipt class per charter; current live projection says 21 GiB vs SSD/host headroom, but still bench-gated | Medium: full-batch vs serial accumulation can differ by reduction order; bench records d_seg-batch sanity | Uses existing `--microbatch-pairs 32` in the bench variant; no default change |
| `mx.compile` / fused loss | NO for this lifted trainer | PROVISIONAL: compile-safe regions were certified elsewhere, not for this PR130 loss closure | #356/#357 safe-compile lessons say only certified regions are load-bearing; no multiplier promoted here | High: MLX compile may reject or alter the closure; bench must treat failure as a result, not a launch blocker | Added default-off `--compile-train-loss`; fire-guard binds it to the mem-probe |
| fp16/bf16 training numerics | NO seam in this lifted trainer | YES for training gradient path only; decode/byte-close untouched | Prior wall-clock burndown estimated 1.5-1.8x ceiling, explicitly estimate-only until QC | High: numerics can degrade d_seg; bench records d_seg-batch sanity and no promotion without later scorer evidence | Added `--train-compute-dtype`; fp32 default is off/identity, fp16 is in the one-shot bench |
| Async/subprocess verdict reclaim | N/A in this training loop | NOT YET: lifted MLX train has in-loop MLX `d_seg_batch`, not blocking CPU-torch verdict submission | #330/#495 apply to verdict-side wall-clock when CPU verdicts are in-loop | None for current train loop; adding subprocesses would create new custody surface | No code built; recorded N/A rather than adding an unused mechanism |

## One-Shot Bench

Plan-only command run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/ddm_wc1_wallclock_bench.py
```

Outputs:

| artifact | status | sha256 |
|---|---|---|
| `.omx/research/ddm_wc1_20260807/wc1_bench_ticket.json` | PLANNED ticket, not fired | `fdfa462b2f95ad98227a570542554ef53b4a1a7d3536d041a47823b128cd1793` |
| `.omx/research/ddm_wc1_20260807/wc1_bench_receipts.jsonl` | 5 planned receipt rows | `5457e4f0a41a8c766b7f662098c5b486fb847f14e9eeb63cdbffb9d72dda909c` |

The planned variants are `{baseline, threads, batched, compile, fp16-train}`. The ticket derives from
`.omx/research/ddm_mx1e_20260807/regen2/probe_result.json` / `argv_n32_arm_cap` and resumes from
`.omx/research/ddm_mx1e_20260807/regen2/launch_arm_cap/n32_metal/mlx.latest.npz` at step 4500,
targeting step 4505 for the short bench. Each variant has its own SSD run dir under
`/Volumes/VertigoDataTier/pact/ddm_wc1_20260807/wallclock_bench/...`, own mem-probe receipt path,
fire-guard verdict path, safe_run status receipt, child pidfile, and result path. Execute only with
`--execute` during the MAIN-owned Metal gap.

Receipt schema: `ddm_wc1_wallclock_bench_receipt.v1`. A measured row, after `--execute`, carries:
`variant`, `env_overrides`, `resume_step`, `step_horizon`, `mem_probe_run`, `fire_guard_run`,
`train_run`, `seconds_per_step`, `d_seg_batch_sanity`, result path/hash, and status. Planned rows keep
`seconds_per_step=null` and `d_seg_batch_sanity=null`.

## DERIVED n120 Step Count

Source measurement: `.omx/research/ddm_mx1t_20260807/MX1T_FINDINGS.md` and
`.omx/research/ddm_mx1t_20260807/mx1t_facets_result.json`, axis
`[macOS-CPU advisory torch upstream SegNet]`, n32 ARM-CAP checkpoint-series only.

| interval | d_seg start | d_seg end | delta d_seg / 1000 steps | read |
|---|---:|---:|---:|---|
| 250 -> 1250 | 0.001051108042 | 0.001070181529 | +0.000019073486 | worse |
| 1250 -> 2250 | 0.001070181529 | 0.001071453094 | +0.000001271566 | flat/worse |
| 2250 -> 3250 | 0.001071453094 | 0.001073201497 | +0.000001748403 | flat/worse |
| final 3250 -> avg-K=8 | 0.001073201497 | 0.001067320506 | -0.000005880992 | tail-average helps |

DERIVED recommendation: do not promote the inherited 6000 as the n120 default from these facets.
Use an initial n120 horizon of 3250 steps with checkpoint grid and avg-K=8 tail selection wired
symmetrically to ARM-CAP/ARM-VEH/n120. Continue beyond 3250 only if the n120 (or fresher n32) facet
table shows negative marginal d_seg per 1000 steps or another measured selection rule beats avg-K=8.

This is a step-count recommendation, not a score claim. The near-flip fraction rose
0.398003931650 -> 0.402251184834 and far-margin mismatch fell 0.063813700287 -> 0.057760663507,
with low churn median 0.035615937141, so the family is not killed; the measured aggregate d_seg
trajectory just does not justify blind 6000-step n120 spend.

## RECALL EVIDENCE

| scope | query / source | found beyond charter seeds | changed plan |
|---|---|---|---|
| Governing contract | `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md` | ARM-CAP owns Metal; WC1 owns CPU prep only; live own frontier is `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]` | No Metal/scorer run; bench script is dry-run by default and uses SSD paths |
| MX1T facets | `.omx/research/ddm_mx1t_20260807/MX1T_FINDINGS.md`, `mx1t_facets_result.json`, `mx1t_provenance_addendum.json` | Step-1500 anchor reproduced exactly; cache binding is gt->gt; avg-K=8 beat final by `-5.880991617838397e-06` | n120 recommendation uses 3250 + avg-K=8, not copied 6000 |
| Safe compile | `rg "mx.compile|safe_compile" .omx/research/mlx_safe_compile_v2_finish_20260708.md .omx/research/safe_compile_gpu_bitcert_20260708.md` | Compile safety is region-scoped; PR130 loss closure has no current cert | Built `--compile-train-loss` as bench-only/default-off with d_seg sanity, not as a promoted default |
| Precision | `rg "fp16|bf16" .omx/research/wallclock_burndown_build_20260715.md .omx/research/sub015_DAG_cheapen_real95_tilehalo_fp16_20260713.md` | Prior precision speedup is estimate/blocked until QC; no seam existed in this lifted trainer | Built fp32-master/train-forward cast flag with fp32 default and fire-guard binding |
| Fire guard / resume | source inspection of `run_mlx_train`, `_mem_probe_args`, `tools/mx1_fire_guard.py` | Resume mem-probe horizon was zero-step-shaped when `resume_step > mem_probe_steps`; new flags were not guard-compared | Fixed resume horizon and added throughput flags to guard config comparison |

## Boundaries

- No Metal command was executed by WC1.
- No scorer slot, n600 job, archive build, or `upstream/evaluate.py` run was performed.
- Live run directories were read for ticket/checkpoint derivation only.
- Planned bench receipts are not measurements; their `seconds_per_step` and `d_seg_batch_sanity` are null until MAIN runs `--execute`.
- Score claim is false. Contest pointer remains borrowed/unmoved.

## Serializer Status

Serializer commit was attempted with post-edit `--expected-content-sha256` for the seven intended
files and message tags `[no-triality] [p0-ledger-ok]`, with no co-author trailer. It failed at
`git add` with rc=128:

```text
error: unable to create temporary file: Operation not permitted
error: experiments/ddm_mx1_pr130_semantic_renderer.py: failed to insert into database
error: unable to index file 'experiments/ddm_mx1_pr130_semantic_renderer.py'
fatal: updating files failed
```

`git diff --cached --name-only` was empty after the failure. This is the managed-sandbox Git object
write blocker, not a test, review, or artifact failure.

Own-vehicle frontier remains `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`; contest pointer remains borrowed/unmoved.
