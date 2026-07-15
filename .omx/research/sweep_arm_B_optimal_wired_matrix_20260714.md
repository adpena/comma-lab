# Sweep Arm B — OPTIMAL×WIRED CONFIG + DSL-WIRING audit (2026-07-14)

**MEANS, not ends.** Exact pointer UNMOVED (0.19108 submittable / 0.18804 borrowed bank). This is a
$0 config-wiring audit + fold pass; no run, no score, no launch. Verdicts labeled
MEASURED / DERIVED / INFERRED / GUESSED-NEEDS-DERIVATION per the operating manual §5.

## Headline
- **Registry unmapped 102 → 83** (coverage 0.744 → 0.792), **stale = 0**, via **6 genuine-orphaned-lever
  folds** into `curriculum_dsl.py` (19 flags), each verified through the REAL trainer argparse
  (`test_every_composable_lever_parses_through_real_trainer_argparse` green) with every weight/iters
  labeled RUN-GATED (no asserted optimum). Commits `2b3f18a0f5` + `a45b83b4b5`.
- **Config-generator ↔ launch.sh flag-level reconciliation: PASS.** `stale == []` — the DSL emits
  ZERO flag the trainer argparse rejects (the #422-class "generator lags launch.sh" byte-identity check
  at the flag surface). No never-invent-flags drift.
- **The 83 remainder is triaged with reasons** (partition below sums to 83); NONE is a silent orphan.
- **META-WIRING FINDING (route):** `lever_registry.completeness()` scans ONLY `curriculum_dsl.py`, so
  **32 flags the DSL genuinely HOLDS in sibling modules** (`spec_v9_cgauge`, `gauge`, `campaign`,
  `optimal_basis_20260714`, `spec_throughput_component_timer`, `spec_next_launch_all_levers`) report as
  "unmapped." The coverage number UNDER-STATES real coverage → BUILD-TICKET B1.

## Config-generator ↔ launch.sh reconciliation (surface 1 crosscheck)
| Check | Method | Result |
|---|---|---|
| DSL-emitted flag ⊆ trainer argparse | `completeness().stale` | **PASS** (stale=[]) — no invented/typo/dead flag |
| every composable lever parses on sealed config | pytest parse-test (real `build_real_trainer_parser`) | **PASS** (16/16, incl. 6 new folds) |
| sealed base argv parses | parse-test control | **PASS** |
Byte-identity of the compiled config vs a live launch.sh was NOT run (no launch this turn, per boundary);
the flag-level reconciliation above is the $0 surface that catches the #422 class.

## OPTIMAL×WIRED matrix — surfaces 1–5
Legend: OPTIMAL = {DERIVED · MEASURED · INFERRED-lit · GUESSED→needs-derivation}; WIRED = {dsl-lever+composable ·
held-elsewhere(registry-blind) · aggregate · default-off-correct · default-on-correct · retired · orphan→FOLDED}.

### 1. THROUGHPUT / SOLVE-TIME
| Item | OPTIMAL | WIRED | Action |
|---|---|---|---|
| megakernel #356 | DERIVED **NO-GO** (FP-reorder wall: compile-fusion not bit-identical, MLX fp32; MEMORY `fp_reorder_transform_bit_identity_wall`) | `--mx-compile` **default-off-CORRECT** (gated bit-identity check) | record default-off-with-reason; not a lever to turn on |
| fused-R kernel #348 | MEASURED (0/28 cross-proc, ~8% faster, bit-exact) | `FusedRKernel` **held+composable** | none — winner ON path exists |
| grouped-backward ~17× | MEASURED | env `TAC_MLX_CUSTOM_GROUPED_BACKWARD` (NOT a flag) | record: env-gated, outside flag registry (BUILD-TICKET B2 to surface in DSL) |
| safe-compile regions | INFERRED | `SafeCompileRegions` **held+composable** (`--safe-compile-manifest/-regions`) | none |
| micro-batch batched twin #313/#447 | GUESSED (batch size run-gated) | `MicroBatch` **held+composable** (`--micro-batch-pairs`) | run-gated: measure optimal pairs |
| verdict-batch #495 | MEASURED (32 = OOM-safe, bit-identical; #205 fix) | `--verdict-pairs` mapped; `--verdict-batch` mapped | none — memory-safe default derived |
| pose-verdict-gate #494 | — | `PoseVerdictGate`/`PoseVerdictGateDryStart` held (code-gated, empty flags); `--verdict-pose-gate` **RETIRED** in V9 | record retired |
| frozen-SegNet fwd/bwd #449/#455/#456 | — | `scorer_gradient_policy` module (registry-blind) | route B1 |

### 2. CONVERGENCE
| Item | OPTIMAL | WIRED | Action |
|---|---|---|---|
| DE-derived viscous-HJ #318 + adaptive-ε #320 | DERIVED (ε(t)=clamp(|c_a|√(ηλ/8)(1+m),floor,upper); floor 0.3 MEASURED FEED-05v; upper 0.7 biharmonic; margin 0.5 CFL) | `EikonalViscosity` **held+composable** (`--eikonal-viscosity`,`-adaptive`); bounds `--eikonal-visco-eps-{floor,upper,margin-factor}` **held-elsewhere gauge.py** + `-ca-band/-ca-pairs/-anneal` aggregate | BUILD-TICKET B3: extend `EikonalViscosity` factory to EMIT the derived bounds (currently ride argparse defaults) |
| eikonal stabilization StEik #317 | GUESSED (weight run-gated) | **ORPHAN → FOLDED this turn** (`EikonalStEik`) | run-gated A/B owed |
| eikonal junction relax θ*-STRETCH-1 | GUESSED | **ORPHAN → FOLDED** (`EikonalJunctionRelax`) | run-gated A/B owed |
| closed-loop eikonal control | INFERRED | `ClosedLoopEikonalControl` **held+composable**; plateau-detector thresholds `--annulus-plateau-*` held-elsewhere | route B1 |
| curriculum hand-offs CE→τ→l7→Muon #315 | PARTIAL — event-gating DERIVED; schedule still PR95-echo (per #302) | `EventTriggeredCurriculum`,`TauAdvanceEvent`,`Muon`,`MuonAtCheckpointBoundary` held | l7 = MEASURED DEFECT (demote); route #302 for witness-native anneal |

### 3. LOSS (math-optimal)
| Item | OPTIMAL | WIRED | Action |
|---|---|---|---|
| focal-γ #301 | INFERRED-lit (γ=2.0 canonical focal) | `SegFocalGamma` **held+composable** | none |
| boundary-distance #301 | GUESSED (weight run-gated) | `BoundaryDistance` **held+composable** | run-gated |
| anisotropic σ_cc' #382 | DERIVED | `LengthSigma` held (`--length-sigma-matrix`) | none |
| margin-band-satisficing #459/#360 | DERIVED (hinge from level-set energy) | `MarginBandSatisficing` **held+composable** | none |
| DsegAwareTaper | GUESSED | `DsegAwareTaper` **held+composable** | run-gated |
| flicker-aware seg-CE reweight (L85) | GUESSED | **ORPHAN → FOLDED** (`SegSpikeReweight`) | run-gated A/B owed |
| low-rank code nuclear θ*-MUST-2 | GUESSED | **ORPHAN → FOLDED** (`CodeNuclearNorm`) | run-gated A/B owed |
| optimal-metric unification #500 (g=∇²F categorical-Fisher) | DERIVED but SQUARED-HESSIAN caveat (MEMORY `dual_metric_no_solve_is_squared_hessian`: no-solve dual ≠ Fisher-natural) | `bregman_dual_metric_guard` + `spec_v9_cgauge.policy_bindings.optimal_metric` (registry-blind); canonical metric `argmax_native_vjp_fidelity_v1` bound in `lever_registry._CANONICAL_METRICS` | route: metric IS bound as `policy_bindings.optimal_metric`; the H⁻¹-solve gap (#500/#501/#504) is a run/design blocker, not a wiring gap |

### 4. OPTIMIZER (math-optimal) — the #272 base-vs-levelset gap
| Item | OPTIMAL | WIRED | Action |
|---|---|---|---|
| Muon warm-start-momentum + LR-anneal #269/#272 | MEASURED (−32% d_seg vs AdamW; anneal is the finishing cure) | `MuonWarmStart` **held+composable** (`--muon-lr-final-frac`,`--muon-warm-start-momentum`) | none — anneal schedule wired |
| Muon base LR `--muon-lr` (=0.1×lr) | **GUESSED — INHERITED-PR95** (per #302 not witness-native) | held-elsewhere; NOT in `Muon` factory | **NEEDS-DERIVATION → route #302** |
| Muon momentum/ns-steps/weight-decay | INFERRED-lit (Keller-Jordan 0.95/5; AdamW wd) | held-elsewhere/argparse defaults | GENERIC_KNOB (no swept intent); BUILD-TICKET B4 folds them into `Muon` factory |
| MD-decoupling #175/#195 | DERIVED | `WitnessStability` held (`--grad-*`,`--stability-preset`) | none |
| β₂-from-n #222/#223 | DERIVED | `AdamBeta2`,`Beta2WindowRewarmup` **held+composable** | none |

### 5. CURRICULUM
| Item | OPTIMAL | WIRED | Action |
|---|---|---|---|
| state-gated coherent schedule #430 | DERIVED | `EventTriggeredCurriculum`,`CurriculumReanchorLevers` held; `TailCycles` held+composable | none |
| event-triggered hand-offs #315 | DERIVED | event levers held; `--muon-start-event`/`--lane-band-start-event`/`--seg-chroma-boundary-start-event` aggregate | none (aggregate to their levers) |
| Schedule/Curriculum first-class DSL #334/#339 | — | `schedule_readback.py` + `curriculum_candidate_pool.py` exist | curriculum-candidate-pool P0 orphan-class (MEMORY `curriculum_candidate_pool_p0_orphan_class`): full sweep owed at config-finalization → route |

## Unmapped triage — the 83 remainder (partition sums to 83, MEASURED via classifier)
| Bucket | Count | Meaning / representative action |
|---|---:|---|
| **FOLDED this turn** | (19) | 6 new `Lever` factories → 102→83 |
| HELD_ELSEWHERE (registry-blind) | 32 | DSL holds them in sibling modules; registry scans only `curriculum_dsl` → BUILD-TICKET B1 (broaden scan OR migrate canonical holder) |
| AGGREGATE_HELD | 18 | sub-params of a held curriculum lever (AmplifyIsland shape, PersistenceTopology clDice, EikonalViscosity ca-band/anneal, StoreNothingPoseCarrier geom, LanePrior BUILD-2, AA self-orient cache, LogitAdjust classes, seg-loss stage) — ride the parent; record |
| GENERIC_KNOB | 11 | literature/neutral scalar, no swept intent (`--weight-decay`,`--warmup-epochs`,`--n-hidden`,`--wire-s0/w0`,`--hinge-weight`,`--pose-eps`,`--margin-target-end`,`--muon-momentum/-ns-steps/-weight-decay`) — record, NOT a lever |
| RESUME_INFRA | 8 | resume/warm-start control-flow + memory/throughput infra (`--resume-*`,`--warm-start-weights-only`,`--mlx-cache-clear-accum`=#205-OOM default-on,`--verdict-subprocess`,`--gpu-reorient`,`--freeze-decoder-fit-codes`) — not swept levers |
| MODE_BUILD_TICKET | 5 | `--residual-mode`+`--residual-target-npz` (v8 rate MODE, B5); `--score-domain-loss` (default-on guardrail, HNeRV-L6); `--per-group-grad-clip` (C4 confound-fix, FOLD-candidate B6); `--mx-compile` (default-off-correct #356) |
| NEEDS_DERIVATION | 3 | `--muon-lr` (INHERITED-PR95 → #302); `--l7-mult`,`--l7-threshold` (l7 measured-DEFECT, values moot until witness-native) |
| RETIRED | 3 | `--verdict-pose-gate`,`--verdict-pose-canary-every` (task-494 retired in V9), `--unselected-r1-advisory-dpose` (deprecated/refused) |
| OBSERVABILITY_ON | 3 | `--mod-dim-dynamics/-ablation/-ablation-k` DEFAULT-ON score-neutral (correct per "observability defaults on") |
**New unmapped count: 83** (was 102). Folded 19 / aggregate 18 / held-elsewhere 32 / knob 11 / infra 8 / mode 5 / needs-deriv 3 / retired 3 / obs-on 3.

## Build-tickets (terminal designs — buildable as-is by the owning arm)
- **B1 (registry scope, MY domain — deferred to a dedicated turn, NOT $0-blind):** broaden
  `dsl_referenced_flags()`/`lever_factories()` to scan the whole `tac.witness_dsl` package (or a
  curated policy-module allowlist) so the 32 sibling-held flags count as mapped. RISK: sibling modules
  mention flags in prose/docstrings → false coverage; must gate on real Lever/override emission, and
  update `#332` provenance-bijection + `test_lever_registry` invariants. Deliverable: a `dsl_source_files()`
  helper + per-module AST union + a test that the 32 named flags move unmapped→mapped WITHOUT admitting
  docstring-only mentions. Measurement gate: `completeness().stale` stays 0 and the parse-test stays green.
- **B2 (env-gated throughput → DSL):** surface `TAC_MLX_CUSTOM_GROUPED_BACKWARD` (and sibling env
  toggles) as observability rows in the DSL/activation-ledger so env-gated winners are not orphaned from
  the "off is a tracked queue" apparatus. Owner: witness_dsl + throughput_authority_policy.
- **B3 (EikonalViscosity emits derived bounds):** extend the `EikonalViscosity` factory to emit
  `--eikonal-visco-eps-floor 0.3` / `-eps-upper 0.7` / `-margin-factor 0.5` (the #320 DERIVED bounds,
  currently riding argparse defaults + held only in gauge.py) so the derived optimum is DSL-explicit.
  $0, MY domain — deferred only to keep this turn's fold set reviewable; buildable next turn.
- **B4 (Muon base hyperparams into factory):** extend the non-composable `Muon(start_epoch)` factory to
  accept+emit `--muon-lr/-momentum/-ns-steps/-weight-decay`; route `--muon-lr` to #302 witness-native
  derivation (do NOT hardcode 0.1×lr as optimal — it is INHERITED-PR95). Trainer-side: none (flags exist).
- **B5 (residual-only MODE lever):** `--residual-mode`+`--residual-target-npz`+`--freeze-decoder-fit-codes`
  compose the v8 rate machinery (fixed deterministic bulk + learned residual). Design a `ResidualOnlyMode`
  Lever; needs a `--residual-target-npz` artifact (run-gated input) → route-run-gated.
- **B6 (fold PerGroupGradClip next turn):** `--per-group-grad-clip` is the C4 confound-fix stability
  lever (store-bool `val` type — verify render before fold; not folded this turn to avoid the ambiguous
  `val`-type render risk). $0 fold candidate.

## Run-gated blockers (exact: what needs a run/operator-GO to MEASURE the optimum)
Each FOLDED lever's WEIGHT/ITERS optimum is RUN-GATED (owed per-lever A/B through the real n600 verdict,
byte-closed): `EikonalStEik.weight`, `EikonalJunctionRelax.relax`, `CodeNuclearNorm.weight`,
`SegSpikeReweight.downweight`, `LambdaPreProbe.iters`, `SpikeGuardRollback.{frac,lr_cut,window,max}`.
`MicroBatch.pairs`, `BoundaryDistance.weight`, `DsegAwareTaper` — same. `--muon-lr` witness-native value
is a #302 derivation blocked on a witness-native anneal design (not a run). The #500 optimal-metric
H⁻¹-solve gap (#500/#501/#504) is a design+run blocker, not a wiring gap (the metric IS bound as
`policy_bindings.optimal_metric` + `argmax_native_vjp_fidelity_v1`).

## Triality legs
- **DSL leg:** 6 `Lever` factories landed in `curriculum_dsl.py` (commits 2b3f18a0f5, a45b83b4b5);
  registry auto-discovers them (AST); `[consumers-generic]` (Levers render generically via
  `compile_trainer_argv`).
- **DAG leg:** this report is the FEED row (Arm B OPTIMAL×WIRED matrix + 102→83 fold + B1–B6 tickets).
- **equations leg:** no new law measured (this is a wiring pass); the FP-reorder-wall NO-GO, #320
  adaptive-ε bounds, and #500 squared-Hessian caveat are ALREADY-registered anchors (no owed equation).
  The RUN-GATED lever optima, once measured, are owed as `EmpiricalAnchor` rows on their levers.
