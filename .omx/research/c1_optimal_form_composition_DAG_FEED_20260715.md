# DAG FEED — #507 C1 OPTIMAL-FORM COMPOSITION (three legs → ONE DSL config)

FEED id: `FEED-507-c1-optimal-form`
Date: 2026-07-15 · Research-only: `true` · Pointer moved: `false` (0.19108 submittable / 0.18804 banked-borrowed UNMOVED — a config is MEANS)
Directive basis: #507 brief + operator 2026-07-15 update (relaxed identity — "we don't care about drift
as long as gradient is good"; objective = JOINT wall-clock-to-target = epochs × sec/ep; pose unchanged;
fold PoseBlindComputeGate).

## The composed object

`tac.witness_dsl.spec_c1_optimal_form_20260715.compile_c1_optimal_form_launch_config`
→ launcher `--config c1_optimal_form` (registered in `tools/launch_witness_run.py`; fail-loud branch,
run-identity tuple, argparse choices). 22 levers = the official leg-A parent (19 + S_R) + 3 additions;
emitted-argv delta vs parent is EXACTLY the 7 addition flags (fail-closed delta-contract in the factory).

### Per-leg consumption (verified on emitted argv, not intent)

- **Leg A (S_R reachability)** — CONSUMED via the official factory
  `spec_v9_cgauge.compile_v9_cgauge_ideal_mod19_sR_launch_config` (`--margin-saliency-reachability`
  emitted; `--margin-saliency-weight 1.0`; mod-dim 19). The throughput arm's compatibility
  reconstruction is retired; `spec_c1_throughput_20260715.compile_c1_sr_parent_launch_config` now
  delegates to the official factory.
- **Leg B (throughput)** — the parent ALREADY emits the speed core (`--fused-r-kernel`,
  `--cache-gt-skeleton`, `--async-verdict`, `--verdict-batch 32 --verdict-pairs 0`,
  `--safe-compile-regions hosc_activation`, `--micro-batch-pairs 1`); the ~17× custom
  grouped-backward + persistence-pool kernels ride `PERF_ENV_PREFIX`
  (`tac/witness_dsl/typed_config.py:98`, launcher perf-env gate enforces). The only genuine argv
  delta from the leg: the folded `c1_component_wallclock_telemetry` lever
  (`--component-wallclock-telemetry` + `--component-wallclock-probe-every 1` + `--profile-timing`;
  score-neutral observability, defaults-ON per the off-is-a-tracked-queue law).
- **Leg C (deep math)** — FOLDED where a trainer consumer exists: `PoseBlindComputeGate`
  (`--pose-training-compute-gate` + `--verdict-pose-gate`, task #495 trunk-phase compute saver;
  pose PHYSICS unchanged — the R1 two-phase finisher, pose_finish at sigma_min_plateau, no parallel
  pose thread) + `HeadOffsetSolver(mode="flip_median")` (`--head-offset-solver flip_median`, the #386
  Hamming-optimal advisory decode-time arbiter; consumed at the trainer's EMA-verdict call site;
  NEVER mutates shipped/EMA/resumed weights ⇒ no flicker pathology by construction; its
  realized-through-R delta rows ARE the owed n600 A/B instrument).

### Typed SLOTS (directive gate (b) consumed-not-inert, the #417 proof — could NOT fold honestly)

| surface | blocker (cited custody) | unlock |
|---|---|---|
| Bregman #504 | its own DAG FEED (`bregman_all_surfaces_504_DAG_FEED_20260715.md`): "DSL: OWED — no real trainer-consumed swept Bregman/centroid/sigma actuator is evidenced" | a trainer flag consuming the Bregman step |
| Fisher trust-region | `FisherNaturalSolverPolicy`: argv-inert, `activation='built_not_activated_measurement_owed'`, `research_only=True` | the owed measured A/B + a trainer consumer |
| Hessian-precond #423 | `laguerre_logit_offset._newton_step_from_cov(precondition=…)` is opt-in; the trainer's `solve_head_offsets` call passes NO `precondition` kwarg — not argv-reachable | a `--head-offset-precondition` trainer flag |
| Curvelet basis | operator 2026-07-15: fold only when `curvelet_optimal_form_crux` lands its optimal form (+ no-Fourier-basis gate: opt-in, never a default flip) | `curvelet_optimal_form_receipt=<file>` (typed kwarg; fail-closed on a missing file; folds `WindowedCurveletBasis`) |

Folding any of the first three today would be the counted-but-inert #417 fake class; the directive's own
inclusion gate (b) forbids it. The slots are typed, manifest-visible, and each names its exact unlock.

## Reconciliation verdicts (the two the throughput arm deferred)

1. **S_R ↔ micro-batch**: resolved by TRAINER CODE, not assumption —
   `train_levelset_witness_realized_through_R_mlx.py` fail-closes
   "`--margin-saliency-reachability` is not supported with `--micro-batch-pairs>1` (the batched
   LEVER-4 twin does not consume S_R yet)". Under the 2026-07-15 joint-wall-clock criterion B>1 is
   now admissible IN PRINCIPLE (the batched twin's functional-tolerance contract IS the gradient
   bar; strict bit-identity no longer bars it), but for ANY S_R config it stays CODE-BLOCKED.
   **Named fallen-crack: the batched LEVER-4 twin's missing S_R consumer** — that consumer is the
   unlock for B>1 on this vehicle. The factory pins + VERIFIES `--micro-batch-pairs 1` and refuses
   parent drift (tested).
2. **Grouped-VJP exclusion**: **REFUTED under the new criterion.** The throughput leg excluded the
   custom grouped/depthwise Metal VJP on STRICT identity grounds ("17.96× backward / 5.5× n8 e2e,
   but primary proof is cosine + fp32 roundoff, not bit identity"). Operator 2026-07-15 makes
   functional gradient quality the bar — exactly what that proof establishes — and it is the single
   largest measured wall-clock lever. It stays ON via `PERF_ENV_PREFIX
   TAC_MLX_CUSTOM_GROUPED_BACKWARD=1` (the SoT; the launcher's perf-env gate refuses a launch
   without it). The whole-step megakernel stays EXCLUDED on MEASURED ECONOMICS (CPU 0.79–0.83×,
   GPU 1.12–1.21× — `witness_fp_reorder_transform_bit_identity_wall_v1`), not identity.
3. **FrozenScorerOneThread / `--training-torch-threads`**: the flag never landed and is
   UNNECESSARY — the trainer hard-wires `torch.set_num_threads(SELECTED_THREADS)` from
   `canonical_equations.segnet_exact_forward_cpu_thread_law_20260713` at startup (auth-eval
   untouched by construction). Recorded in `C1_SPEED_DISPOSITIONS`.

## Repair of the import-broken throughput module

`spec_c1_throughput_20260715.py` landed import-broken on main (its 10 lever factories +
`runtime_environment` typed-config fields were never harvested from the isolated worktree; its test
file could not even collect). REWRITTEN as the reconciled surface: every historical factory mapped in
`HISTORICAL_FACTORY_RECONCILIATION` onto the live main-tree owner (parent-carried flag / PERF_ENV
carrier / trainer-native law / folded telemetry lever); the strict-identity `EXCLUDED_OR_HELD` rows
preserved verbatim with explicit `superseded_20260715` annotations; compile entry points now delegate
(official C1a parent; composed #507 config with a supersession stamp). Its bench receipt
(`c1_throughput_composed_bench_20260715.json`, status `BLOCKED_INPUT_CUSTODY`) is carried forward as
launch-blocker `C1_COMPOSED_BENCH_NOT_MEASURED` on the composed config, alongside
`C1_SR_SIDECAR_CUSTODY` (the `sR` cache member / sidecar must exist; the trainer fails closed).

## Triality

- **DSL**: `spec_c1_optimal_form_20260715.py` (the factory IS the leg) + the reconciled
  `spec_c1_throughput_20260715.py` + launcher registration.
- **DAG**: this FEED.
- **Equations**: NO new equation registered — no new measured law emerged; the composition cites
  existing ids (`margin_saliency_reachability_replaces_texture_proxy_v1`,
  `witness_fp_reorder_transform_bit_identity_wall_v1`, `laguerre_ot_head_offset_20260709`,
  `segnet_exact_forward_cpu_thread_law_20260713`, `cgauge_whitney_moddim_v1`). The re-adjudications
  are POLICY changes under an operator directive, recorded as manifest dispositions, not laws.

## State

HELD / `operator_go_required` / PREPARED_NOT_FIRED. Launch blockers: composed-path bench receipt
(measure sec/epoch + peak RSS on the real cache) + sR sidecar custody. No launch, no paid dispatch,
no evaluator call, no archive mutation. Pointer UNMOVED.
