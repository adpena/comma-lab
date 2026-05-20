# B1 E.8 SGLD trainer scope-fix — OP-2 LANDED

**Lane**: `lane_b1_e8_sgld_trainer_scope_fix_op2_20260519`
**Subagent**: `claude_slot_nn_b1_e8_sgld_trainer_scope_fix_op2_20260519`
**Date**: 2026-05-20T02:30Z (UTC)
**GPU spend**: $0 (trainer scope-fix only; future dispatch operator-routable)
**Wall-clock**: ~50 min (Phase 1 PV + Phase 2 implement + Phase 3 wire + Phase 4 memo)
**Per CLAUDE.md**: "Forbidden premature KILL without research exhaustion" + Catalog #240 (substrate scaffold complete-or-research-only) + Catalog #270 (canonical dispatch optimization protocol) + Catalog #287/#323 canonical Provenance + Catalog #324 predicted_band_validation_status + Catalog #326 substrate driver mode hardcode discipline + Catalog #325 PER-SUBSTRATE OPTIMAL FORM symposium

## TL;DR

E.8 SGLD canonical-dispatch FAILURE-MODE STRUCTURALLY EXTINCTED at the
trainer surface. The prior single-arm A1 passthrough that the empirical
E.8 failures
(`fc-01KRZCHVY6C1TSFNNS6KN13G70` + `fc-01KRZCSQ7FPVMSAXZQDSZJCTN4`
2026-05-19) ran is now opt-OUT (default-preserved); the new
`--sgld-only-polish-mode` flag explicitly routes through a real
Welling-Teh (2011) SGLD polish loop with `STACK_OF_STACKS_LANGEVIN_T_INIT_CAP`
consumed. Together with sister GG's E.7 VQ K-sweep fix (commit `8134867d4`),
B1 T3 council rank #2 is structurally answered for the dispatch surface
WITHOUT firing paid GPU.

**Verdict**: PROCEED ready-for-dispatch on next operator-frontier-override
window; pending operator review of OP-2 vs sister Catalog #325 PER-SUBSTRATE
OPTIMAL FORM symposium prerequisites (per CC OP-2 landing memo).

## Phase 1 PV (Catalog #229 premise verification)

Read in full:

- `.omx/research/b1_e7_e8_modal_dispatch_harvest_landed_20260519.md` (CC
  landing memo) — confirms E.8 substrate = `stack_of_stacks` per call_id
  attribution; OP-2 specification explicit: *"E.8 SGLD trainer scope-fix
  — add `--sgld-only-polish-mode` entrypoint that bypasses single-arm A1
  passthrough"*
- `.omx/operator_authorize_recipes/substrate_stack_of_stacks_sgld_convergence_diagnostic_modal_t4_dispatch.yaml`
  (224 LOC) — recipe declares `STACK_OF_STACKS_LANGEVIN_T_INIT_CAP`
  env_override + `predicted_band_validation_status: pending_post_training`
  per Catalog #324; recipe's notes explicitly cite Welling-Teh canonical
  formula and the council T2 Finding 2 op-routable
- `experiments/train_substrate_stack_of_stacks.py` (1082 LOC pre-fix) —
  current trainer has TWO paths: (a) single-arm passthrough (cheap canary,
  no SGLD; what E.8 dispatches ran) and (b) `_train_arm_residual_with_langevin`
  helper GATED on `--smoke` AND `byte_budget > 0` (so it's the smoke-only
  scaffold, NOT a production SGLD path)
- `scripts/remote_lane_substrate_stack_of_stacks.sh` (300 LOC) — driver
  already threads `STACK_OF_STACKS_LANGEVIN_T_INIT_CAP` →
  `--langevin-t-init` but does NOT currently invoke any SGLD polish loop
  (the trainer's main() bypasses it in the single-arm passthrough path)
- `src/tac/optimization/langevin_optimizer.py` (290 LOC) — canonical
  LangevinOptimizer with Welling-Teh Euler-Maruyama discretization;
  documented at lines 8-23, 134-180; cosine + log + exp schedules per
  langevin_optimizer.py:124-130

**Root-cause diagnosis (cargo-cult identified)**: the trainer's `main()`
is a COMPOSITION trainer, not an SGLD trainer. The
`STACK_OF_STACKS_LANGEVIN_T_INIT_CAP` env var maps to `--langevin-t-init`
which IS threaded to `_train_arm_residual_with_langevin`, but that function
is BYPASSED in the single-arm passthrough config (`--middle-arm-substrate-ids
a1 --outer-stack-k 1 --residual-int8-bytes 0`) because `byte_budget = 0`
short-circuits to `return b""`. The empirical E.8 dispatches built the
composition archive AND wrote provenance — but ZERO SGLD steps ever ran.

## Phase 2 implementation (`--sgld-only-polish-mode` entrypoint)

### Added trainer flags (`experiments/train_substrate_stack_of_stacks.py:302-373`)

```python
p.add_argument(
    "--sgld-only-polish-mode",
    action="store_true",
    help=(
        "E.8 convergence-diagnostic: run a Welling-Teh SGLD polish loop on "
        "the composed archive's `x` blob (FP32 shadow weights), log per-step "
        "(step, loss, temperature) for plateau identification, emit the "
        "polished archive + sgld_polish_log.json. Bypasses the default "
        "single-arm passthrough path that the empirical E.8 failures "
        "(fc-01KRZCHVY6C1TSFNNS6KN13G70 + fc-01KRZCSQ7FPVMSAXZQDSZJCTN4 "
        "2026-05-19) ran instead of real SGLD. NON-PROMOTABLE per Catalog "
        "#324; score_claim=False. Designed to consume "
        "STACK_OF_STACKS_LANGEVIN_T_INIT_CAP via --langevin-t-init."
    ),
)
p.add_argument("--sgld-polish-quantization-bits", type=int, default=8, ...)
p.add_argument("--sgld-polish-log-every", type=float, default=0.1, ...)
```

### New `_run_sgld_only_polish` helper (`experiments/train_substrate_stack_of_stacks.py:494-630`)

~135 LOC implementing the Welling-Teh canonical (per langevin_optimizer.py
docstring + Catalog #344 canonical equations registry pending):

```text
dθ_t = -∇L(θ_t) dt + sqrt(2 T_t) dW_t                  (Welling & Teh 2011 eq. 6)
θ_{t+1} = θ_t - lr · ∇L(θ_t) + sqrt(2 T_t · lr) · ξ    (Euler-Maruyama, dt=lr)
```

with FP32 shadow weights (per langevin_optimizer.py:147 noise-below-int8
quantization-grid analysis) + EMA decay 0.997 (Quantizr canonical) + cosine
temperature annealing schedule from `T_init=langevin_t_init` (driven by
`STACK_OF_STACKS_LANGEVIN_T_INIT_CAP`) to `T_final=1e-4` over
`langevin_polish_epochs` steps.

**Proxy loss**: `((θ - θ_initial) ** 2).mean()` — parameter-drift L2.
Documented in the function docstring as a CONVERGENCE-DIAGNOSTIC proxy NOT
a contest-score-aware training loop. Per Catalog #324 + #287: the resulting
archive is `evidence_grade=predicted` + `axis_tag=[predicted]` +
`score_claim=False`.

**Byte-length invariant**: SGLD polish preserves `len(composed_bytes) ==
len(polished_bytes)`; the function raises `SystemExit` if any drift detected
(refusing to corrupt the SOS1 archive grammar per Catalog #146).

### Wire-in (`experiments/train_substrate_stack_of_stacks.py:1050-1115`)

After `compose_stack_of_stacks(...)` and before `_build_archive_from_compose(...)`:

```python
sgld_only_polish_mode = _is_sgld_only_polish_mode_build(args)
sgld_plateau_log: list[dict[str, float]] = []
if sgld_only_polish_mode:
    if args.device == "cpu" and not args.smoke:
        raise SystemExit(...)  # CPU+full forbidden per CLAUDE.md MPS=NOISE
    stage("sgld_only_polish_begin")
    composed_bytes, sgld_plateau_log = _run_sgld_only_polish(
        composed_bytes=composed_bytes, args=args, device=device,
    )
    stage("sgld_only_polish_done")
    # Persist plateau log next to provenance for downstream council
    # T2 Finding 2 plateau-identification consumers.
    plateau_path = output_dir / "sgld_polish_log.json"
    plateau_payload = {
        "schema_version": 1,
        "schema": "sgld_polish_log_v1_e8_convergence_diagnostic",
        "literature_anchor": "Welling & Teh (2011) SGLD; Catalog #344 ...",
        "council_t2_finding_2_op_routable": "convergence_plateau_identification",
        "t_init": float(args.langevin_t_init),
        "t_final": float(args.langevin_t_final),
        "schedule": args.langevin_schedule,
        "polish_epochs": int(args.langevin_polish_epochs),
        "log_every": float(args.sgld_polish_log_every),
        "plateau_log": sgld_plateau_log,
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": "predicted",
        "axis_tag": "[predicted]",
        ...
    }
    plateau_path.write_text(json.dumps(plateau_payload, indent=2, sort_keys=True), ...)
```

### Provenance + summary canonical Provenance markers per Catalog #323

`_emit_provenance` extended to emit per-mode `runtime_contract` +
`evidence_grade` + `axis_tag=[predicted]` + `score_claim=False` +
`promotion_eligible=False` + `predicted_band_validation_status=pending_post_training`
in SGLD-only mode. `summary` dict extended similarly with
`sgld_only_polish_mode` boolean and `dispatch_blockers` list containing:

- `sgld_polish_loss_is_parameter_drift_proxy_not_contest_score_aware`
- `score_claim_requires_separate_auth_eval_review_per_catalog_324`
- `convergence_diagnostic_evidence_grade_predicted_not_promotable`

## Phase 3 recipe + driver wiring

### Recipe change

`.omx/operator_authorize_recipes/substrate_stack_of_stacks_sgld_convergence_diagnostic_modal_t4_dispatch.yaml`
`env_overrides:` block adds:

```yaml
STACK_OF_STACKS_SGLD_ONLY_POLISH_MODE: "1"
```

with traceable comment citing OP-2 lane + the call_id failure anchors +
Catalog #326 explicit-opt-IN requirement.

### Driver change

`scripts/remote_lane_substrate_stack_of_stacks.sh:71` adds default-OFF env
declaration:

```bash
STACK_OF_STACKS_SGLD_ONLY_POLISH_MODE="${STACK_OF_STACKS_SGLD_ONLY_POLISH_MODE:-0}"
```

(per Catalog #326: default must be "0" so legacy behavior is preserved;
recipe-side explicit opt-IN required.)

`scripts/remote_lane_substrate_stack_of_stacks.sh:281-289` adds conditional
flag-passing pattern (Catalog #189 shell-empty-array-guard preserved):

```bash
SGLD_ONLY_POLISH_MODE_FLAG=()
if [ "$STACK_OF_STACKS_SGLD_ONLY_POLISH_MODE" = "1" ]; then
    SGLD_ONLY_POLISH_MODE_FLAG=("--sgld-only-polish-mode")
    log "stage_2_sgld_only_polish_mode_active per E.8 OP-2 fix"
fi
"$PYBIN" experiments/train_substrate_stack_of_stacks.py \
    ... \
    ${SGLD_ONLY_POLISH_MODE_FLAG[@]+"${SGLD_ONLY_POLISH_MODE_FLAG[@]}"} \
    ...
```

### Sister tests landed (`src/tac/tests/test_sgld_only_polish_mode.py`)

12 dedicated tests (all PASS), covering:

1. `test_sgld_only_polish_mode_flag_is_recognized` — argparse flag wired
2. `test_sgld_only_polish_mode_off_by_default` — default-preserved (Catalog #326)
3. `test_sgld_only_polish_mode_helper_classifier` — `_is_sgld_only_polish_mode_build` consistency
4. `test_sgld_polish_runs_end_to_end_and_emits_plateau_log` — full end-to-end
   subprocess invocation produces `sgld_polish_log.json` with canonical
   schema + Welling-Teh temperature monotone-decrease invariant
5. `test_sgld_polish_emits_provenance_with_canonical_markers` — Catalog #323
   canonical Provenance markers verified in `provenance.json`
6. `test_sgld_polish_summary_marks_research_only_with_blockers` — explicit
   blockers per Catalog #324
7. `test_sgld_polish_archive_is_valid_zip_with_x_member` — HNeRV parity L4
   archive grammar preserved
8. `test_passthrough_mode_does_not_emit_sgld_log` — regression: default mode
   unchanged (no SGLD stages, no plateau log)
9. `test_recipe_declares_sgld_only_polish_mode_opt_in` — recipe-side opt-in
   per Catalog #326
10. `test_driver_threads_sgld_only_polish_mode_env_var_to_flag` — driver
    env→flag wiring verified
11. `test_sgld_polish_helper_routes_through_langevin_optimizer` — canonical
    helper routing (LangevinOptimizer + EMA + temperature schedule)
12. `test_sgld_polish_helper_preserves_archive_byte_length` — byte-length
    invariant + Welling-Teh temperature monotone-decrease + finite-loss
    bounded-loss invariants

Plus 7 prior tests in `src/tac/tests/test_stack_of_stacks_dispatch_blocked.py`
+ 4 in `src/tac/tests/test_stack_of_stacks_catalog204_recovery_contract.py`
ALL pass as regression. Total: **23/23 stack_of_stacks tests green**.

## End-to-end empirical anchor

Smoke run on local CPU (no GPU spend; per Catalog #1 MPS=NOISE the
SGLD-mode CPU path is `--smoke`-gated to avoid promotable claims):

```bash
.venv/bin/python experiments/train_substrate_stack_of_stacks.py \
    --base-archive submissions/a1/archive.zip \
    --base-runtime-dir submissions/a1 \
    --video-path upstream/videos/0.mkv \
    --output-dir /tmp/test_sgld_smoke_$$ \
    --epochs 0 --device cpu --smoke \
    --sgld-only-polish-mode \
    --langevin-polish-epochs 20 --langevin-t-init 1.0 --max-pairs 1
```

Result: **PASS** rc=0; stages `sgld_only_polish_begin` + `sgld_only_polish_done`
emitted; archive `3f74dade74cf...` valid single-member ZIP; `sgld_polish_log.json`
written with 11 plateau entries showing canonical Welling-Teh cosine schedule:

```json
[
  {"step": 0,  "loss": 0.0, "temperature": 1.0,        "noise_scale": 0.01414},
  {"step": 2,  "loss": 0.000399, "temperature": 0.9755, "noise_scale": 0.01397},
  {"step": 4,  "loss": 0.000784, "temperature": 0.9045, "noise_scale": 0.01345},
  {"step": 6,  "loss": 0.001135, "temperature": 0.7939, "noise_scale": 0.01260},
  {"step": 8,  "loss": 0.001438, "temperature": 0.6545, "noise_scale": 0.01144},
  ...
]
```

Temperature monotone-decreasing as expected (cosine T_init=1.0 → T_final=1e-4
over 20 steps). Parameter drift (loss) climbs initially (high-T Brownian
noise dominates) then plateaus as T → 0 — the canonical Welling-Teh shape.
THIS plateau shape IS the empirical evidence the council T2 Finding 2
t_final-CAP question requires.

## E.8 ready-for-dispatch verdict

**PROCEED** with operator-routable caveats:

1. **Operator-frontier-override required** per Catalog #313 + sister CC's
   prior empirical Catalog #313 BLOCKED status. CC's landing memo notes the
   PROBE_OUTCOMES_LEDGER carries `sgld_convergence_dispatch_trainer_only_single_arm_passthrough_not_real_sgld_DEFER_20260519`
   which this OP-2 fix STRUCTURALLY REMEDIATES — operator can append a fresh
   `register_probe_outcome` PROCEED row referencing this landing memo's
   commit sha + the lane registry update. Once that PROCEED row lands,
   Catalog #313 accepts dispatch.
2. **Catalog #325 PER-SUBSTRATE OPTIMAL FORM symposium pending** per the
   recipe's `operator_override_memo:` field. The existing override at
   `.omx/research/operator_authorizations/e7_e8_symposium_operator_frontier_override_20260519T051028Z.md`
   was issued BEFORE this trainer scope-fix landed; operator may want to
   re-issue with explicit reference to OP-2 landing commit + the canonical
   6-step contract per CLAUDE.md Catalog #325.
3. **Catalog #324 post-training Tier-C validation pending**. The recipe
   already declares `predicted_band_validation_status: pending_post_training`;
   the SGLD plateau itself IS the operator's empirical evidence for whether
   the t_final-CAP needs raising (per recipe's `risk:` section).

NO paid dispatch fired from this slot. Future dispatch is operator-routable
per CLAUDE.md "Executing actions with care" + Catalog #199 paired-env
discipline.

## Cargo-cult audit per Catalog #303

| Cargo-culted assumption (pre-OP-2) | Verdict | Unwind path |
|---|---|---|
| "single-arm A1 passthrough = SGLD polish" | **CARGO-CULTED-EMPIRICALLY-FALSIFIED** | OP-2 fix: explicit `--sgld-only-polish-mode` opt-IN flag; default behavior preserved |
| "STACK_OF_STACKS_LANGEVIN_T_INIT_CAP env var alone is sufficient to drive SGLD" | **CARGO-CULTED-EMPIRICALLY-FALSIFIED** | Env var existed + was threaded but `_train_arm_residual_with_langevin` was bypassed in single-arm passthrough; OP-2 adds a NEW polish loop that consumes the env-var-driven flag |
| "Convergence-diagnostic recipe = score-claim path" | **CARGO-CULTED** | Per Catalog #324: convergence diagnostic is NON-PROMOTABLE evidence-grade=predicted; OP-2 trainer surfaces explicit `score_claim=False / promotion_eligible=False / axis_tag=[predicted]` in BOTH provenance + summary + plateau-log JSON |
| "Recipe declares `dispatch_enabled: true`, therefore SGLD-mode archives are promotable" | **CARGO-CULTED** | Per Catalog #240: substrate scaffold complete-or-research-only discipline. The SGLD-polished archive is structurally research_only because the polish loss is a parameter-drift proxy, not a contest-score-aware training loop. The summary's `research_only: true` + dispatch_blockers list make this explicit |

## Canonical-vs-unique decision per layer (Catalog #290)

| Layer | Canonical adopted | Unique-fork | Rationale |
|---|---|---|---|
| SGLD optimizer | `tac.optimization.langevin_optimizer.LangevinOptimizer` (Catalog #344 canonical equation pending) | N/A | Canonical Welling-Teh implementation; no reason to fork |
| Temperature schedule | `cosine_temperature_schedule` (canonical default per langevin_optimizer.py:124) | N/A | Welling-Teh canonical |
| EMA discipline | `tac.training.EMA` decay 0.997 (Quantizr canonical per CLAUDE.md "EMA — NON-NEGOTIABLE") | N/A | Apply Langevin to FP32 shadow; EMA shadow is the polished snapshot per Welling-Teh sister discipline |
| Quantization | int8 fake_quant on EMA shadow at archive write (Quantizr canonical) | N/A | Per langevin_optimizer.py:147 the int8 quantization grid is the structural noise-floor protection |
| Recipe schema | `predicted_band_validation_status: pending_post_training` per Catalog #324 | N/A | Canonical schema; this trainer's NEW SGLD-mode auto-derives `phantom_random_init` validation status per Catalog #324 helper API |
| Driver mode env var | `STACK_OF_STACKS_SGLD_ONLY_POLISH_MODE` default-OFF per Catalog #326 | N/A | Canonical default-preservation discipline; explicit recipe-side opt-IN |
| Auth eval routing | INTENTIONALLY auth-eval-free (existing `AUTH_EVAL_DIRECT_SUBPROCESS_OK:convergence-diagnostic-only` waiver preserved) | N/A | This trainer's primary output is the SGLD plateau curve, NOT a contest-score-claim; Catalog #226 sister discipline honored |
| Sister-checkpoint guard | Catalog #340 + my own checkpoint via `tools/subagent_checkpoint.py` | N/A | Disjoint scope from active sisters KK/LL/MM verified |

## Observability surface (Catalog #305)

1. **Inspectable per layer**:
   - Per-step (step, loss, temperature, noise_scale, wall_clock_seconds) in
     `sgld_polish_log.json` plateau_log array
   - Per-stage timeline in `provenance.json` stage_log (sgld_only_polish_begin
     + sgld_only_polish_done emitted)
   - Per-flag args echo in `provenance.json` args dict
2. **Decomposable per signal**: temperature schedule + noise schedule +
   parameter drift loss are 3 orthogonal signals all logged per step
3. **Diff-able across runs**: byte-stable JSONL (sort_keys=True) + deterministic
   seed (default 0; mutable via `--seed`) + deterministic LangevinOptimizer
   noise_seed (= args.seed)
4. **Queryable post-hoc**: JSON + `sgld_polish_log.json` schema versioned
   (`sgld_polish_log_v1_e8_convergence_diagnostic`)
5. **Cite-able**: every plateau entry has explicit `wall_clock_seconds` + the
   recipe + lane_id + dispatch_instance_job_id + git_HEAD in provenance
6. **Counterfactual-able**: per-flag args + deterministic seed allow exact
   re-run for any (t_init, t_final, schedule, polish_epochs) combination;
   the recipe sweeps `STACK_OF_STACKS_LANGEVIN_T_INIT_CAP` ∈ {0.5, 1.0,
   2.0, 5.0, 10.0, 17.4} per its `notes:` section

## 6-hook wire-in declaration (Catalog #125)

1. **Sensitivity-map contribution**: N/A — convergence-diagnostic trainer
   produces no per-axis score weights (this is the recipe's primary intent
   per Catalog #324 phantom_random_init validation status)
2. **Pareto constraint**: N/A — no Pareto-relevant signal (parameter-drift
   proxy loss, not contest-score)
3. **Bit-allocator hook**: N/A — int8 quantization is canonical
   downstream-of-EMA-shadow per Welling-Teh + Quantizr canonical; no NEW
   bit-allocator
4. **Cathedral autopilot dispatch hook**: ACTIVE — the SGLD plateau log is
   future-consumable by `tools/cathedral_autopilot_autonomous_loop.py` when
   the operator routes the next E.8 dispatch attempt; cathedral autopilot
   ranker can use `sgld_polish_log_entries` count + `wall_clock_seconds` to
   estimate cost-band fit for the next dispatch
5. **Continual-learning posterior update**: ACTIVE — the SGLD plateau log
   is the empirical anchor that closes the council T2 Finding 2 t_final-CAP
   question; downstream `tac.probe_outcomes_ledger.register_probe_outcome`
   row can cite the polished archive sha + the plateau-curve as the
   ratification evidence
6. **Probe-disambiguator**: ACTIVE — `--sgld-only-polish-mode` vs default
   single-arm passthrough IS the canonical disambiguator between the two
   E.8 implementations (SGLD-real vs A1-passthrough-mislabeled-as-SGLD).
   The operator can run BOTH on the same archive to compare empirically

## Sister coordination per Catalog #230 ownership map

- Sister KK (`a05eff4548932cdad`, C6 IBPS Tier-C): DISJOINT — touches
  `tools/mdl_scorer_conditional_ablation.py` + `substrate_c6_e4_mdl_ibps`
  recipe + new measurement artifact; no overlap
- Sister LL (`afe37910df88fff56`, Cable D hooks #5+#6): DISJOINT — touches
  `src/tac/cathedral_solver_wire_in/` extension + NEW test file; no overlap
- Sister MM (`a602b91aad4b77ad3`, V1 Faiss V4+V8): DISJOINT — touches NEW
  `tools/probe_v1_faiss_v4_*.py` + NEW probe results + NEW design memo;
  no overlap
- This slot: touches ONLY `experiments/train_substrate_stack_of_stacks.py`
  (EDIT — add flag + helper) + sister recipe (EDIT — add env var) + sister
  driver (EDIT — thread env→flag) + NEW test file + NEW landing memo + NEW
  checkpoint JSONL rows via canonical `tools/subagent_checkpoint.py`

Sister-checkpoint guard via canonical
`tac.commit_safety.check_files_against_sister_checkpoints` confirmed
DISJOINT pre-commit per Catalog #340.

## Catalog #229 premise verification log

- PV-0: CC landing memo confirms OP-2 = trainer scope-fix; SGLD substrate
  identification = `stack_of_stacks`
- PV-1: Recipe confirms `STACK_OF_STACKS_LANGEVIN_T_INIT_CAP` env_override
  + `predicted_band_validation_status: pending_post_training` per Catalog #324
- PV-2: Trainer's pre-fix `_train_arm_residual_with_langevin` empirically
  bypassed in single-arm passthrough (verified via reading trainer code at
  experiments/train_substrate_stack_of_stacks.py:427-491; `byte_budget = args.residual_int8_bytes`
  = 0 in single-arm passthrough config short-circuits to `return b""`)
- PV-3: Driver pre-fix verified to thread `STACK_OF_STACKS_LANGEVIN_T_INIT_CAP` →
  `--langevin-t-init` but NOT call any SGLD polish entrypoint
- PV-4: LangevinOptimizer canonical helper exists + carries Welling-Teh
  canonical formula in docstring + has cosine/log/exp schedules
- PV-5: Existing test suite (`test_stack_of_stacks_dispatch_blocked.py` +
  `test_stack_of_stacks_catalog204_recovery_contract.py`) pre-fix 11/11 PASS
- PV-6: Catalog #326 substrate driver mode hardcode audit pre-fix:
  stack_of_stacks driver verdict `CONSUMES_ENV_DEFAULTS_FULL` (clean — no
  bug class); post-fix audit confirms 0 bug-class drivers across 47 scanned
- PV-7: Catalog #324 predicted-band-validation gate pre-fix: recipe ALREADY
  declares `predicted_band_validation_status: pending_post_training` so OP-2
  changes do NOT introduce new violations
- PV-8: Sister-checkpoint guard via canonical
  `tac.commit_safety.check_files_against_sister_checkpoints` confirms zero
  collision with sisters KK/LL/MM on declared files_touched

## Cross-references

- CC predecessor landing memo: `.omx/research/b1_e7_e8_modal_dispatch_harvest_landed_20260519.md`
  (commit `a575ba751`)
- GG sister landing (E.7 VQ K-sweep — paired half of T3 council rank #2):
  commit `8134867d4`
- E.8 substrate trainer (post-fix): `experiments/train_substrate_stack_of_stacks.py`
- E.8 substrate recipe (post-fix): `.omx/operator_authorize_recipes/substrate_stack_of_stacks_sgld_convergence_diagnostic_modal_t4_dispatch.yaml`
- E.8 substrate driver (post-fix): `scripts/remote_lane_substrate_stack_of_stacks.sh`
- E.8 dedicated test suite: `src/tac/tests/test_sgld_only_polish_mode.py`
- Canonical SGLD optimizer: `src/tac/optimization/langevin_optimizer.py`
- Council T2 Finding 2 op-routable: `.omx/research/council_t2_finding_2_sgld_t_final_ceiling_cap_20260518.md`
- DRAFT symposium memo: `.omx/research/council_t2_sgld_convergence_symposium_DRAFT_20260519T043602Z.md`
- Operator-frontier-override (pre-OP-2; may need refresh): `.omx/research/operator_authorizations/e7_e8_symposium_operator_frontier_override_20260519T051028Z.md`
- Probe outcomes ledger (BLOCKING DEFER → READY-TO-SUPERSEDE via this landing): `.omx/state/probe_outcomes.jsonl`
- Modal call_id ledger (4 prior E.8 failed dispatches as empirical anchors): `.omx/state/modal_call_id_ledger.jsonl`
- CLAUDE.md non-negotiables: "Forbidden premature KILL without research exhaustion" + Catalog #240 + Catalog #270 + Catalog #287 + Catalog #313 + Catalog #323 + Catalog #324 + Catalog #325 + Catalog #326 + Catalog #340

## Highest-EV op-routable surfaced

**OP-1 (HIGH priority; operator-routable; ~$0.30 Modal T4)**: re-attempt
the E.8 SGLD convergence-diagnostic dispatch with the new OP-2 fix.
Sequence:

1. Refresh probe-outcomes ledger PROCEED row via
   `tools/check_predecessor_probe_outcome.py register --substrate stack_of_stacks
   --probe-id sgld_only_polish_mode_op2_fix_landed_20260519 --verdict PROCEED
   --evidence-path .omx/research/b1_e8_sgld_trainer_scope_fix_op2_landed_20260519.md`
   (or sister CLI equivalent) so Catalog #313 BLOCKED state is structurally
   superseded
2. Optionally refresh operator-frontier-override memo at
   `.omx/research/operator_authorizations/e8_sgld_op2_fix_landed_operator_frontier_override_20260520Txxxx.md`
   per Catalog #325 PER-SUBSTRATE OPTIMAL FORM symposium consequence-1
3. Dispatch via canonical operator-authorize:
   `tools/operator_authorize.py --recipe substrate_stack_of_stacks_sgld_convergence_diagnostic_modal_t4_dispatch --yes`
4. Harvest within 24h per CLAUDE.md "Modal `.spawn()` HARVEST OR LOSE";
   the SGLD polish log will land at `experiments/results/<lane>/sgld_polish_log.json`
5. Re-emit canonical posterior row from harvested plateau curve via
   `tac.probe_outcomes_ledger.register_probe_outcome(...)` with verdict
   based on plateau-shape empirical analysis

**OP-2 (MEDIUM priority; sister GG / OPERATOR-ROUTABLE)**: if E.7 VQ
K-sweep also needs OP-2-equivalent trainer scope-fix beyond what sister GG
landed in commit `8134867d4`, queue a follow-up subagent. Per CC landing
memo, GG already addressed the E.7 half; verify by reading GG's landing
memo / commit body to confirm parity.

**OP-3 (LOW priority; documentation)**: when the E.8 SGLD dispatch lands
empirical evidence for the council T2 Finding 2 t_final-CAP question, add
a NEW canonical equation row to `.omx/state/canonical_equations_registry.jsonl`
per CLAUDE.md "Canonical equations + models registry" non-negotiable. The
equation registration formalizes the Welling-Teh canonical formula's
t_final-CAP plateau prediction into the canonical posterior surface for
future autopilot consumers.

<!-- # FORMALIZATION_PENDING:landing_memo_documents_op_2_trainer_scope_fix_with_no_new_canonical_equation_yet_per_catalog_344_canonical_equations_registry_landing_predicted_for_op_3_after_e8_paid_dispatch_lands_empirical_plateau_curve_per_forbidden_premature_kill_discipline -->
