# C1 throughput/convergence integration audit — 2026-07-15

## Launch verdict

**BLOCKED — DO NOT FIRE C1b.** The typed C1 treatment is compiled, mapped, and structurally tested,
but the required real composed-path benchmark was not executable in this isolated worktree. No
`gt_n600.npz`, adjacent S_R sidecar/admitted in-cache S_R receipt, or gitignored upstream scorer runtime
exists here or on either authorized SSD search surface. Therefore seconds/epoch, peak RSS, and the
composed-path bit-identity spot check are **NOT MEASURED**. The fail-closed receipt is
`c1_throughput_composed_bench_20260715.json`; no zero or inferred metrics substitute for the missing run.

This is an input/runtime-custody blocker, not a C1 formulation negative. Pointer remains
`0.19108282419209976 [contest-CPU]`; the separate borrowed defensive bank remains
`0.1880443979880752 [contest-CPU, non-submission]`.

## Exact config and argv

- Config id: `v9_cgauge_ideal_mod19_sR_c1_throughput`
- Parent: committed C1a `v9_cgauge_ideal_mod19_sR`, commit
  `bdbbf5da175a46c11393ebbe56f53653828fb765`. This isolated branch reconstructs that exact named
  S_R delta because its base predates C1a; after merge, the compiler consumes the official function.
- Scientific trainer argv SHA-256:
  `ea9e8a3f22ef9121dedcb689a7debc26171eee2ce613224824d09db052135d78`
- Typed-config hash:
  `4f1b9d1cc23383e134ce6213a8e6f43e53800a64d5174b7f2c60262cb453d1e6`
- Runtime environment:
  `TAC_MLX_CUSTOM_GROUPED_BACKWARD=0 TAC_MLX_CUSTOM_PERSISTENCE_POOL=1`
- Exact governed launcher argv:

```text
.venv/bin/python tools/launch_witness_run.py --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz --num-pairs 600 --config v9_cgauge_ideal_mod19_sR_c1_throughput --out-dir experiments/results/v9_cgauge_ideal_mod19_sR_c1_throughput_20260715 --no-dashboard
```

The named compiler owns the full trainer argv. Its non-inherited C1 actuation subset is exactly:

```text
--margin-saliency-reachability
--micro-batch-pairs 1
--safe-compile-regions none
--fused-r-kernel
--cache-gt-skeleton
--training-torch-threads 1
--async-verdict
--verdict-batch 32
--verdict-pairs 0
--component-wallclock-telemetry
--component-wallclock-probe-every 1
--profile-timing
```

The complete deterministic argv is emitted by
`compile_c1_throughput_launch_config().to_program().compile_trainer_argv()` and is content-bound by the
scientific argv hash above. The launcher refuses if the benchmark receipt does not bind the same argv,
typed hash, GT bytes, and S_R bytes.

## Stores consulted

- `docs/operating_manual_craft_handoff.md`, `CLAUDE.md`, `AGENTS.md`, C1a commit object
- `.omx/state/canonical_task_status.jsonl`, canonical equation registry, lane registry, activation ledger
- `tools/costate_digest.py --json` on 2026-07-15: CLEAN; 97 registered duty rows before launch
- `.omx/research/throughput_fresh_eyes_20260713.md` and its machine measurement JSON
- `.omx/research/next_launch_all_levers_ticket_20260713.md`
- `.omx/research/whole_step_megakernel_356_20260711.md`
- `.omx/research/microbatch_bit_identity_smoke_n600_20260710.md`
- `.omx/research/deterministic_gpu_accum_348_20260707.md`
- `.omx/research/mlx_custom_grouped_backward_kernel_makes_mlx_gpu_fast_20260612.md`
- `.omx/research/d16_metal_kernels_20260708.md`
- `.omx/research/codex_findings_frozen_segnet_exact_forward_20260713_codex.md`
- `.omx/research/frozen_segnet_necessity_optimality_alternatives_20260712.md`
- `.omx/research/codex_findings_throughput_nogo_naive_rescope_audit_20260714_codex.md`
- `.omx/research/throughput_authority_ladder_20260714T031002Z.md` and static receipt

## Per-optimization disposition

`W/I/T/M/ON` means wired / integrated in this typed config / tested at the stated authority / measured
at the stated scope / active in C1. “Exact” is never inferred across hardware, batch geometry, or config.

| Finding / task cluster | Verdict | W | I | T | M | ON | Evidence and scope | Equation |
|---|---|:---:|:---:|:---:|:---:|:---:|---|---|
| Whole-step megakernel #356 | EXCLUDED, measured formulation NO-GO | Y | excluded | Y | Y | N | CPU 0.788–0.831x; GPU closure 1.117–1.210x, about 5% e2e; grad max-abs 2.31e-7…2.267e-5, deterministic but different | `witness_fp_reorder_transform_bit_identity_wall_v1` |
| Kernel-stack #443 custom grouped VJP | EXCLUDED on strict C1; reference explicitly selected | Y | Y-reference | functional only | Y | N | 17.96x scorer backward, 5.5x n8 e2e; primary proof is cosine 0.99999775/0.99999921 plus fp32 roundoff, not tensor-byte identity | fp-reorder wall; verdict_scope=current Metal formulation on strict C1 |
| Fused R #348 / L70 | KEEP | Y | Y | Y, per-chip gate | Y, scoped | Y | fixed-order graph 0/28 divergent tensors; 25.35→23.44 s in n1 200-ep smoke = 1.081x; n600 composition is still owed and therefore part of the launch blocker | `mlx_gpu_crossprocess_nondeterminism_v1` |
| Persistence-pool kernel #443 | KEEP | Y | Y via typed env | Y exact | Y | Y | max-abs 0 vs NumPy; full-loss flag A/B exact; N=5 cross-process exact; soft-skeleton 8.377→2.139 ms = 3.92x; live persistence total measured about 2.8 s/ep at K=1 | `# NO_EQUATION_NEEDED: explicit-order max/min/mean equivalence and timing receipt` |
| GT skeleton constant cache | KEEP | Y | Y | Y exact | partial | Y | caches an epoch-invariant gradient-free GT constant; exact reuse is proved; no independent C1 whole-epoch credit is claimed | `# NO_EQUATION_NEEDED: constant recomputation elision` |
| Micro-batch #261/#313/#410/#447, B>1 | EXCLUDED; B1 pinned | Y | Y-reference | Y | Y | N | current S_R consumer refuses B>1; B2 has scorer/reduction drift and historical faithful n24 is only 1.036x epoch / 1.001x step; B4/B8 regress | `witness_fp_reorder_transform_bit_identity_wall_v1` |
| One-thread frozen SegNet #449/#451/#456 | KEEP, training path only | Y | Y | Y internal replay | Y | Y | n64 936.312→312.677 ms = 2.995x with zero argmax flips; n600 two-build ABBA 2.9562855x/2.9970427x, all arms internally stable; 15/600 cross-thread ties are accepted only for training, not evaluator authority | `segnet_exact_forward_cpu_thread_control_v1`; `...static_process_v2` |
| Async verdict | KEEP inherited observational path; no new speed credit | Y | Y | Y no optimizer readback | Y timeliness only | Y | real n600 logs measured zero cadence skips; worker service is 30–34% of completion window; matched whole-trainer reclaim/contention remains unmeasured and is not claimed | `# NO_EQUATION_NEEDED: observational worker isolation; overlap law remains measurement telemetry` |
| Verdict chunking #240 | KEEP | Y | Y | Y | Y operational | Y | batch 32, all pairs (`pairs=0`); bounded exact verdict path avoids historical OOM without changing the evaluator cell | `# NO_EQUATION_NEEDED: exact partition of observational verdict work` |
| Component timer/curriculum #492 | KEEP for measurement | Y | Y | Y read-only | instrument, not speed | Y | exact component schema plus coarse profile are ON at cadence 1; no causal speedup is assigned | `async_overlap_and_inclusive_vjp_throughput_v1` |
| FreSh initialization #448 | HELD, not a measured GO | Y | N | build tests only | N real epochs-to-target | N | prior execution was host/governor-blocked; fixed warm-start result was an instance, not a FreSh test | `# NO_EQUATION_NEEDED: no admitted empirical treatment yet` |
| Muon finishing #269/#272 | KEEP inherited convergence schedule | Y | Y | Y inherited | Y historical schedule | Y | existing C1 parent emits warm-start momentum, final LR fraction, event start and preserved stage checkpoints; this audit adds no multiplicative speed claim | existing Muon schedule laws in parent manifest |
| Safe compile / HOSC | HELD on this host; reference explicitly selected | Y | Y-reference | certificate gate exists | N on this host | N | current per-chip manifest is absent; no certificate transfer. Parent `hosc_activation` is overridden DSL-natively to `none` | `witness_fp_reorder_transform_bit_identity_wall_v1` |
| ANE / CoreML #482/#490 | EXCLUDED current training path | Y advisory | N | forward fidelity only | Y forward | N | fidelity-passing fp32 forward 3.609x, but no differentiable VJP/training placement; weight-only W8 measured 38.291% aggregate and 88.775% worst-pair flips | `# NO_EQUATION_NEEDED: execution-substrate/custody exclusion` |
| Raw costate reuse #454 | EXCLUDED measured formulation; family open | Y research | N | Y | Y | N | raw ZOH K2: 456 accepts, 67 guard fallbacks, only 308/456 accepted rows non-worse; no admitted in-loop economics | verdict_scope=raw-input ZOH formulation only |
| Sparse/grouped adjoint #486/#487/#488 | EXCLUDED current formulation; host-kernel family open | Y research | N | Y | Y | N | dense execution realized 1.0x despite 2.2086x derived sparse ceiling; masks/basis fail fidelity; exact K2/native sparse provider remains separate | verdict_scope=current masks+dense execution |
| Forward-kill #455/#456/#465 | EXCLUDED measured formulations; nonlinear teacher family open | Y research | N | Y | Y | N | current nonlinear surrogate/reuse/gating variants did not clear held exact descent plus economics; no “95% killed” production receipt exists | verdict_scope=measured variants only |
| PoseVerdictGate named surface | RETIRED | Y | N | Y fail-closed | Y | N | no payload-bound banked pose cache; live PoseNet remains required. Async live verdict is the compatible path | `# NO_EQUATION_NEEDED: missing content-bound substitution authority` |
| Transient label #495 | HELD identity-blocked | N/A | N/A | N/A | N/A | N | canonical task/source lookup found no exact object; no technical verdict is attached to the number | verdict_scope=task identity only |
| Maximum-throughput fixed-point/int R #445/#494 and integer-R follow-on | HELD | Y policy/probes | N | partial | partial | N | precision/R-adjoint ladders have static/local receipts, but no admitted current-host n600 training-loop parity+determinism+wall receipt | `mlx_gpu_crossprocess_nondeterminism_v1` plus authority-ladder laws |

## S_R and micro-batch reconciliation

The S_R treatment and B>1 are not composable today: the trainer fails closed because the batched twin
does not consume S_R. Independently, B>1 changes scorer/reduction order and fails the strict identity
wall. C1 therefore pins B1 in the DSL and does not claim the historical isolated 1.56x/1.75x scorer
anchors or the disproved “2–4x in-loop” premise.

The measured faithful full-step comparison makes the current B1 tax versus B2 only about 0.1% on step
time in that n24 historical instance; B2 epoch was 3.6% faster, while B4/B8 were slower. These do not
transfer into a C1 n600 projection. The custom grouped VJP would be the much larger historical speed
lever, but C1's strict bit-identity requirement excludes it until exact tensor-byte parity exists. No
compatible replacement is fabricated. C1 pays both the B1 constraint and reference-VJP cost; the actual
combined sec/ep must be measured on the exact composed config.

Compatible speed/measurement apparatus retained at B1 is fixed-order fused R, exact persistence pool,
constant GT cache, the one-thread training scorer, async observational verdict, exact verdict chunking,
and component telemetry. Safe compile is held pending a fresh host certificate. FreSh is held pending a
real epochs-to-target receipt.

## Composed-config measurement

| Required result | Actual |
|---|---|
| real path / n | **NOT RUN**; intended n600 exact config |
| sec/epoch | **NOT MEASURED** |
| peak RSS | **NOT MEASURED** |
| bit-identity spot check | **NOT RUN** |
| reason | GT cache, S_R custody, and upstream runtime absent in this isolated worktree/authorized SSD search |
| durable receipt | `c1_throughput_composed_bench_20260715.json`, status `BLOCKED_INPUT_CUSTODY` |

The compiler validates an admitted receipt only if status is `MEASURED_PASS`, B1 and S_R were actually
exercised, sec/ep and RSS are positive finite measurements, identity is `PASS`, scientific argv and typed
hash match, and the GT/S_R SHA-256 values match the bytes now present. Mutation after measurement is a
tested hard refusal.

## Own round-1 recursive review

- Every active mathematical speed path has an exactness/isolation basis; no compiled-whole-step,
  B>1, uncertified safe-compile, or cosine-only grouped-VJP path is promoted under the strict C1 rule.
- Every excluded technical formulation carries a measured wall/fidelity reason and a narrow
  `verdict_scope`; transient #495 is identity-blocked rather than guessed.
- The S_R/B1 conflict is explicit in both flags and manifest, and the historical B2/B4/B8 wall evidence
  is not promoted into a C1 projection.
- The composed config did **not** run. The audit and compiler remain launch-blocked instead of calling
  structural tests a throughput measurement.
- Regression tests cover exact flags/environment, registry mapping, activation duty, benchmark
  admission, and post-measurement input mutation refusal.

## Remaining owed gate and exact operator action

MAIN must review/merge this branch together with C1a, restore the content-addressed real cache/S_R and
upstream runtime in an authorized environment, then run the exact governed argv above. Replace the
blocked receipt only with the schema-bound real measurement. Require `launch_blockers=[]`, rerun the
focused tests, and inspect the emitted component rows before C1b authorization. No paid dispatch is
needed. Until then: **C1b launch REFUSE; pointer unchanged.**
