# ddm_lt1 receipt - lever_registry triage

Captured: 2026-08-05/06 local session, path date retained from charter.

Scope: scorer-free mechanical triage of `tac.witness_dsl.lever_registry`
factories plus `completeness()` unmapped/stale rows. No scorer, no launch, no
`upstream/evaluate.py`, no trainer edit, no live run-dir edit.

Note: this directory already contained `LT1_RECEIPT.md` for a different PE3/OD9
accounting receipt. I left it intact and wrote the charter-required
`RECEIPT.md` and `NEXT_IF_RESUMED.md` separately.

Frontier line carried from the live board, not moved here:

`S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory] n600`; contest pointer
`0.1910828242 [contest-CPU]` is borrowed and unmoved.

## Answer first

Denominator: `198` package-wide lever factories from
`lever_registry.build_completeness()`.

| bucket | count | meaning |
|---|---:|---|
| TR1-portable | 73 / 198 | Live-TR1 factories plus old witness factories with a named TR1 consumer or fire order. |
| vehicle-agnostic | 19 / 198 | Portable laws, telemetry/control helpers, and policy rows; not a launch by themselves. |
| witness-scoped | 106 / 198 | Retired levelset/witness vehicle rows; archive as historical unless a future charter names a consumer. |

Build-grade facts:

| fact | value |
|---|---:|
| modules globbed | 178 |
| modules with factories | 20 |
| factories bound to retired levelset/witness trainers | 141 |
| factories bound to live TR1 trainer | 57 |
| designed/measured stubs | 10 |
| silent stubs | 2 |
| label-drift rows | 2 |
| undeclared-trainer factories | 17 |
| verdict-relevant undeclared | 0 |

The 10 stubs are not fireable rows. The two silent label-drift rows are
`WeightNormTelemetryRow` and `IntegerPlaneEmitter`.

## Recall evidence

| query / source | finding beyond charter seeds | plan impact |
|---|---|---|
| `.omx/state/main_hot_state.md` | `lt1` is listed live; `wp1` is in flight; `wl1`, `la1`, `dy2` have landed; own-vehicle line is `0.7539807296911207 @ 357,836 B`. | Kept this arm scorer-free and deduped against live owners. |
| `.omx/research/ddm_wl1_20260805/TRANSFER_TABLE.md` | WL1 already queued LB, S_R, EIK and folded tp1/wp1/la1/dy2-adjacent surfaces; WL1 also recorded old `completeness()` as retired-vehicle coverage. | Rows that match WL1 fire orders are marked FOLDED/owned, not re-fired by LT1. |
| `.omx/research/ddm_tp1_20260805/TP1_PACKET.md` | TP1 sealed BI1 tickets and at that time refused PE3/cheapdct4 as missing consumers. | BI1 and telemetry are already owned; old TP1 caveat had to be refreshed because TK1 later landed the missing consumers. |
| `.omx/research/ddm_tk1_20260805/TK1_RECEIPT.md` | TK1 implemented default-off PE3 conditioning and cheapdct4 accounting consumers, no score/launch. | PE3/cheapdct4 are no longer "missing consumer" rows, but still need ticket recompile and no score claim. |
| `.omx/research/ddm_la1_20260805/RECEIPT.md` | LA1 implemented `lever_jd1_lr_anneal`, default-off, no scorer/launch; boundary A/B owed. | LR anneal is already ported; LT1 does not duplicate. |
| `.omx/research/ddm_dy2_20260805/RECEIPT.md` | DY2 implemented `lever_jd1_plateau_tail_average_ema`, registered the equation, no scorer/launch. | Tail EMA is already ported; LT1 does not duplicate. |
| `.omx/research/ddm_wp1_20260805/row4_g3_transfer_tr1_endpoint_delta.json` plus live board | WP1 owns Muon/MC-finisher/readback style work; available WP1 artifact is cached advisory read only, score_claim=false. | Muon and readback rows are FOLDED to WP1, not requeued here. |
| `src/tac/witness_dsl/lever_registry.py` | `build_completeness()` is the package-wide factory census; legacy `completeness()` remains a single-file DSL/trainer flag coverage surface. | Factory denominator uses package-wide `198`; old single-file `completeness()` is recorded but not used as live TR1 anti-orphan proof. |
| `.venv/bin/python tools/list_canonical_equations.py --json` | Registry is available and includes recent dy2 equation rows; the raw registry is broad. | Used for recall coverage only; no new equation or score claim. |
| `rg ... MEMORY.md` | Prior memory says Pact frontier work must claim lanes before dispatch, keep lanes separate, and avoid treating advisory/proxy rows as pointer moves. | Reinforced no-launch/no-score boundary. |

Scoped negative: I did not find a newer LT1 lever-registry receipt in
`.omx/research/ddm_lt1_20260805/` before this write. I found only the older
PE3/OD9 `LT1_RECEIPT.md`, which is not this charter.

## Commands / measured facts

Registry census:

```text
.venv/bin/python - <<'PY'
from pathlib import Path
from tac.witness_dsl import lever_registry as lr
build = lr.build_completeness()
legacy = lr.completeness()
tr1 = lr.completeness(Path("experiments/train_tr1_partition_renderer_mlx.py"))
print(build.to_dict())
print(legacy.trainer_total, len(legacy.mapped), len(legacy.unmapped), len(legacy.stale))
print(tr1.trainer_total, len(tr1.mapped), len(tr1.unmapped), len(tr1.stale))
PY
```

Observed current facts:

```text
build_completeness total_factories: 198
factories_by_trainer:
  experiments/train_levelset_witness_realized_through_R_mlx.py, experiments/train_witness_realized_through_R_mlx.py: 141
  experiments/train_tr1_partition_renderer_mlx.py: 57
stub_count: 10
silent_stub_count: 2
label_drift_count: 2
undeclared_trainer_factory_count: 17
verdict_relevant_undeclared_count: 0

legacy completeness() [RETIRED vehicle: train_levelset_witness_realized_through_R_mlx.py]:
  trainer_total 443
  mapped 363
  unmapped 80
  stale 3

completeness(Path("experiments/train_tr1_partition_renderer_mlx.py")):
  trainer_total 116
  mapped 19
  unmapped 97
  stale 306
```

The live-TR1 `completeness(Path(...))` row is diagnostic only: the legacy
function compares the live trainer against only `curriculum_dsl.py`, so sibling
TR1 factories in `spec_tr1_renderer_20260728.py`, `p4x_*`, `pt2_*`, `bi1_*`,
and `tk1_*` appear unmapped even when package-wide factory grading sees them.
Use `build_completeness()` for package-wide factory anti-orphan coverage.

`spec_v10_status(Path("."))`:

```text
clear False
post_merge_resolved 5 / 5
gate_present 0 / 6
seed_present 1 / 2
blocker_count 7
```

Blockers were unchanged from WL1:

```text
gate:v9c2_completion
gate:p0_497_curvelet_ab_verdict
gate:warmup_8v27_ab_verdict
gate:probe_p1_n600_band_and_terminal_decomp
gate:probe_p2_mirror_transport_rate
gate:probe_p3_chroma_plane_jacobian
seed:hood_tex_seed
```

No dated v10 addendum was created because the blocker set did not change.

## Factory bucket ledger

### TR1-portable: 73 / 198

These are either already live-TR1 factories or retired-witness factories with a
specific TR1 consumer/fire order. This bucket is not a claim that all rows should
fire now.

- `ax1_derived_levers_20260730.py` (1): `Ax1Frame0CarriedWarp` STUB.
- `bi1_birth_seed_levers_20260805.py` (1): `lever_tr1_birth_seed_amplify`.
- `curriculum_dsl.py` (16): `AnalyticLaneBandTraining`, `AnalyticLaneRenderBand`, `EikonalStEik`, `EikonalViscosity`, `FinerBiasInit`, `HorizonWeightedMargin`, `MarginSaliency`, `MarginSaliencyReachability`, `PhaseAdvectionConsistency`, `PoseFinisherLiveGap`, `RangeAProjection`, `SegChromaBoundary`, `SegFormUnifyTau`, `StepNativeActivation`, `VerdictLiveGap`, `WeightEntropyPenaltyMLX`.
- `fh1_adapted_force_levers_20260731.py` (5): `BirthPlateauKneeConjunct` STUB, `ErfBirthContextCoadapt` STUB, `MarginSatisficeCap` STUB, `TieLocusEdgeWeighted` STUB, `XiAdvectedTokenBase` STUB.
- `p4x_existence_levers_20260803.py` (6): `lever_existence_beta`, `lever_existence_birth_matrix`, `lever_existence_grammar`, `lever_existence_target`, `lever_existence_weight_policy`, `lever_lane_existence_hinge`.
- `ph3_s10_frontloaded_levers_20260731.py` (2): `Qa80MarginBoundedPhotometric` STUB, `Qa81LaneCarrierComposite` STUB.
- `pt2_ported_levers_20260803.py` (4): `lever_fisher_density`, `lever_head_natural_grad`, `lever_seg_focal_gamma`, `lever_tau_softplus_tau`.
- `spec_tr1_renderer_20260728.py` (36): `lever_a1_gate`, `lever_basin_handoff`, `lever_boundary_probe`, `lever_byte_ledger_coder`, `lever_composed_s_verdict`, `lever_delta_group_sparsity`, `lever_desc_level_roundtrip`, `lever_ema_decay`, `lever_head_range_relax`, `lever_jd1_joint_pose_finish`, `lever_jd1_lr_anneal`, `lever_jd1_muon_finisher`, `lever_jd1_plateau_tail_average_ema`, `lever_lane_guard_born`, `lever_lane_guard_lambda`, `lever_lane_guard_margin_floor`, `lever_lane_guard_ratchet`, `lever_lotto`, `lever_rate_in_loss`, `lever_renderer_capacity`, `lever_renderer_head`, `lever_reset_operator`, `lever_seg_grad_q3_project`, `lever_seg_margin_weight`, `lever_seg_physics`, `lever_solve_frame_distill`, `lever_telemetry_v9_port`, `lever_token_cell_mask`, `lever_token_grid`, `lever_token_init`, `lever_token_quant_anneal`, `lever_token_quant_margin_coupling`, `lever_token_rowband`, `lever_token_temporal`, `lever_variant`, `lever_window`.
- `tk1_pe3_conditioning_levers_20260805.py` (2): `lever_tk1_cheapdct4_pose_accounting`, `lever_tk1_pe3_conditioning`.

### Vehicle-agnostic: 19 / 198

These are portable as laws, campaign/control helpers, or telemetry patterns.
They are not direct TR1 scorer actions unless a later charter supplies a
consumer.

- `campaign.py` (7): `advance_to_l7`, `expand_cycles`, `extend_stage`, `rerun_stage_new_config`, `scale_progression`, `select_synergistic_combos`, `select_winners`.
- `constants_telemetry_build_wave_20260715.py` (8): `DerivedAdamBeta2`, `DerivedEmaDecay`, `DerivedEvalEvery`, `DerivedWPoseAtEngage`, `ModDimDynamicsOn`, `RateRollingTelemetry`, `VerdictBatch64`, `WeightNormTelemetryRow` STUB.
- `exact_costate_reuse_policy.py` (1): `exact_costate_reuse_k2_lever`.
- `terminal_costate_skip_policy.py` (1): `terminal_exact_metric_costate_skip_lever`.
- `whole_teacher_distilled_student_policy.py` (1): `whole_teacher_distilled_student_lever`.
- `windowed_curvelet_basis_lever_20260714.py` (1): `windowed_curvelet_basis_lever`.

### Witness-scoped: 106 / 198

These are bound to the retired levelset/witness trainer surface or old capstone
launch specs. Archive them as historical unless a future charter names a live
consumer.

- `curriculum_dsl.py` (100): `AACoverageRender`, `AdamBeta2`, `AdamWReferenceSemantics`, `AdaptiveGradClip`, `AmplifyIsland`, `AreaConstraintBirth`, `Beta2WindowRewarmup`, `BirthCompletionEvent`, `BoundaryDistance`, `CacheGtSkeleton`, `ClosedLoopEikonalControl`, `CodeNuclearNorm`, `CodeSpectralEntropy`, `CompactShearletBasis`, `ComputeDtype`, `ComputeDtypeBf16QCGate`, `CorrectedWeightDecay`, `CurriculumReanchorLevers`, `DM1Minimal`, `DashComb`, `DecoupledField`, `DirectionalBasis`, `DirectionalBasisRebalance`, `DsegAwareTaper`, `EikonalJunctionRelax`, `EmaDecayCalibrated`, `EmaDecayFinisher`, `EventTriggeredCurriculum`, `FiLMFix`, `FilmPolarChartSPELManifoldMuon`, `FisherDensityWeight`, `ForkEmaClearance`, `ForkHeadSolve`, `FreShFixedQualitySlice`, `FreShInitControl`, `FreshFrequencyShift`, `FusedRKernel`, `GradNormalizeNone`, `GroundFrameChart`, `HardnessOversample`, `HeadGeometry`, `HeadNaturalGradient`, `HeadOffsetSolver`, `IntegerPlaneEmitter` STUB, `LadderIslandHomotopy`, `LambdaPreProbe`, `LaneBandStaticCache`, `LanePrior`, `LaneSkipBand`, `LegacyFourierABControl`, `LengthSigma`, `LiteralPolarCurveletBasis`, `LogitAdjust`, `LrAnnealPin`, `MarginBandSatisficing`, `MarginCompandedGroundChart`, `MarginFieldHead`, `MarginStepCap`, `MicroBatch`, `Mod32SegOnlyControlBase`, `Muon`, `MuonAtCheckpointBoundary`, `MuonWarmStart`, `OutTexHidden`, `PersistenceTopology`, `PolyakFinisher`, `PoseBlindComputeGate`, `PoseDecouple`, `PoseEngageWPoseRamp`, `PoseFinishBetaAnnealCoupling`, `PoseFinishConditioningGate`, `PoseFinisherFilmReadbackArm`, `PoseMarginalWeightLaw`, `PoseVerdictGate`, `PoseVerdictGateDryStart`, `ResumeLRWarmup`, `SafeCompileRegions`, `SeedIslandBirth`, `SeedIslandEased`, `SegFocalGamma`, `SegSpikeReweight`, `SoftBoundary`, `SpikeGuardRollback`, `StiefelW`, `StoreNothingPoseCarrier`, `TailCycles`, `TauAdvanceEvent`, `TauFrozen`, `TemporalScrewConsistency`, `TerminalPoseFinish`, `TextureTrunk`, `TieLocusDisplacement`, `UniWARD`, `VerdictDevice`, `VerdictParallelWorkers`, `WarmStartRestoreBoundaryState`, `WarpRealLumaFrame0`, `WindowedCurveletBasis`, `WitnessStability`, `YhatNativeGenerator`.
- `spec_c1_optimal_form_20260715.py` (2): `_phase_tail_label_floor_lever`, `_telemetry_lever`.
- `spec_c2_surgical_20260716.py` (1): `compile_c2_surgical_warm_launch_config`.
- `spec_next_launch_all_levers_20260713.py` (1): `_observer_telemetry`.
- `spec_v9_cgauge.py` (1): `compile_v9_cgauge_ideal_launch_config`.
- `spec_v9c3_duty_ab_20260719.py` (1): `compile_v9c3_duty_ab_config`.

## TR1-portable rows and deduped consumers

| row | consumer / owner | fire order |
|---|---|---|
| `lever_telemetry_v9_port` | TP1 already consumes this as read-only telemetry. | FOLDED to TP1, no LT1 action. |
| `lever_tr1_birth_seed_amplify` | BI1/TP1 birth ON/OFF packet. | FOLDED to TP1 ticket path; no duplicate. |
| `lever_tk1_pe3_conditioning`, `lever_tk1_cheapdct4_pose_accounting` | TK1 landed after TP1. | Recompile TP1/full crossed tickets before use; no score claim. |
| `lever_jd1_lr_anneal` | LA1. | FOLDED to LA1 jd7-or-Case-B boundary A/B. |
| `lever_jd1_plateau_tail_average_ema` | DY2. | FOLDED to DY2 anchor-selected boundary A/B. |
| `lever_jd1_muon_finisher`; retired `Muon`, `MuonAtCheckpointBoundary`, `MuonWarmStart`, `FilmPolarChartSPELManifoldMuon`, `PolyakFinisher` | WP1 Muon/MC-finisher/readback lane. | FOLDED to WP1; LT1 does not re-own. |
| `AnalyticLaneBandTraining`, `AnalyticLaneRenderBand` | WL1-LB, JD/TR1 fragile-lane training lever. | QUEUED-WITH-FIRE-ORDER #1 after scorer slot free and clean stage boundary. |
| `MarginSaliencyReachability` | WL1-SR, exact through-R `S_R` reachability weighting. | QUEUED-WITH-FIRE-ORDER #2; first revalidate `gt_n600_sR.npz` custody. |
| `EikonalViscosity`, `EikonalStEik` | WL1-EIK fair fixed-guard reopen. | QUEUED-WITH-FIRE-ORDER #3 after clean checkpoint provenance and scorer slot. |
| `SegFormUnifyTau` | TR1/JD schedule-structure A/B or next witness restart baseline. | RACE after top three WL1 rows; build live TR1 consumer before launch. |
| `FinerBiasInit`, `StepNativeActivation` | Activation family reopen. | RACE after current JD boundary; own-optimum A/B only. |
| `SegChromaBoundary` | Grammar-v2/carrier race; chroma as d_seg lever. | RACE after carrier route opens; no pose-rescue claim. |
| `HorizonWeightedMargin`, `MarginSaliency` | Local annulus saliency reweighting. | QUEUE after WL1-SR or if SR custody blocks; local A/B, not global taper resurrection. |
| `PhaseAdvectionConsistency` | TR1/JD phase/coherence queue. | Matched ON/OFF only; do not cite v9 phase as d_seg win. |
| `RangeAProjection` | v10/endgame range(A) gate/dashboard. | HOLD until v10 blockers clear; no config from prose constants. |
| `PoseFinisherLiveGap`, `VerdictLiveGap` | Costed live/EMA readback when pose divergence matters. | QUEUE only with explicit scorer/readback budget; TP1 did not port by default. |
| `WeightEntropyPenaltyMLX` | Structured code-table / weight-entropy probe. | $0 locality probe before any hyperprior or TR1 training build. |
| `pt2_ported_levers_20260803.py` four factories | TR1 seg-force duty-to-measure set. | Already ported; schedule one-at-a-time matched A/B only after current higher-ranked boundary items. |
| `p4x_existence_levers_20260803.py` six factories | #924 existence-hinge A/B. | TP1 queued after BI1 smoke or frame0-repaired base boundary. |
| `Ax1Frame0CarriedWarp`, `fh1_*`, `ph3_*` STUB factories | TR1-targeted but missing trainer flags. | BUILD-CONSUMER-FIRST; no launch or A/B until `missing_flags == ()`. |
| `spec_tr1_renderer_20260728.py` native factories not named above | Current TR1 DSL surface. | Use through existing governed launcher/ticket path; no new LT1 fire order. |

## Unmapped/stale row triage

Legacy `lever_registry.completeness()` is scoped to the retired
`train_levelset_witness_realized_through_R_mlx.py` trainer. It currently reports
`80` unmapped flags and `3` stale emitted flags.

Exclusive unmapped triage:

| bucket | count | rows |
|---|---:|---|
| TR1-portable or already-owned candidate | 11 | `--eikonal-visco-ca-band`, `--eikonal-visco-ca-pairs`, `--eikonal-viscosity-anneal`, `--lane-band-start-event`, `--lane-band-would-fire-telemetry`, `--muon-adamw-lr`, `--muon-momentum`, `--muon-ns-steps`, `--muon-start-event`, `--muon-weight-decay`, `--seg-chroma-boundary-start-event` |
| vehicle-agnostic utility/custody | 32 | `--ckpt-retain-per-stage`, `--containment-damp`, `--containment-mode`, `--freeze-decoder-fit-codes`, `--fresh-lineage-parent-receipt`, `--fresh-lineage-parent-receipt-sha256`, `--fresh-producer`, `--gpu-reorient`, `--literal-chart-fine-factor`, `--logit-adjust-classes`, `--mlx-cache-clear-accum`, `--mod-dim-dynamics`, `--mx-compile`, `--n-hidden`, `--per-group-grad-clip`, `--profile-timing`, `--rate-rolling-telemetry`, `--residual-mode`, `--residual-target-npz`, `--resume-allow-lever-drift`, `--resume-clear-spike-guard`, `--resume-model-from`, `--seed-lr`, `--skip-boot-baseline-verdict`, `--tail-live-mq`, `--training-target-capsule`, `--training-target-capsule-sha256`, `--unselected-r1-advisory-dpose`, `--verdict-pose-canary-every`, `--verdict-subprocess`, `--warmup-epochs`, `--weight-decay` |
| witness-scoped | 37 | `--aa-self-orient-fine-cache-cap`, `--aa-self-orient-fine-mode`, `--amplify-form`, `--amplify-margin-target`, `--amplify-persist`, `--annulus-plateau-dwell-windows`, `--annulus-plateau-min-epochs`, `--annulus-plateau-rel-eps`, `--cldice-iters`, `--hinge-weight`, `--island-dilate-px`, `--lane-prior-phi1-bias-scale`, `--lane-prior-phi1-source-pair`, `--margin-target-end`, `--mod-dim-ablation`, `--mod-dim-ablation-k`, `--persistence-classes`, `--persistence-recall-weight`, `--pose-carrier-fit-pairs`, `--pose-carrier-pitch`, `--pose-carrier-residual-scale`, `--pose-carrier-s-r`, `--pose-carrier-s-t`, `--pose-eps`, `--score-domain-loss`, `--seed-anneal-epochs`, `--seed-anneal-shape`, `--seed-blend`, `--seg-loss`, `--structured-init-lr`, `--structured-init-sdf-clip`, `--structured-init-steps`, `--structured-init-subsample`, `--structured-init-thresh`, `--warm-start-weights-only`, `--wire-s0`, `--wire-w0` |

Stale emitted flags:

| bucket | count | rows | disposition |
|---|---:|---|---|
| witness-scoped stale | 3 | `--integer-plane-emitter-basis`, `--integer-plane-emitter-mode`, `--integer-plane-emitter-policy-sha256` | `IntegerPlaneEmitter` is a non-fireable stale factory until a real trainer/receiver consumer is named. |

## Boundaries

Measured by LT1:

- `build_completeness()` factory denominator and build-grade counts.
- `completeness()` retired-trainer unmapped/stale rows.
- `completeness(Path("experiments/train_tr1_partition_renderer_mlx.py"))` diagnostic counts.
- `spec_v10_status(Path("."))` blocker state.
- Deduplication against WL1, TP1, TK1, LA1, DY2, and WP1 surfaces read in this turn.

Not measured by LT1:

- No scorer forward.
- No `upstream/evaluate.py`.
- No archive bytes or byte-closed candidate.
- No contest-CPU/CUDA row.
- No MLX launch or run-dir mutation.
- No review of unrelated dirty worktree files.

## Follow-on disposition

FIRED:

- Wrote this typed triage with denominator `198`.
- Re-ran current `spec_v10_status(Path("."))`.
- Classified legacy unmapped/stale rows without claiming live TR1 anti-orphan closure from the old single-file `completeness()` surface.

FOLDED:

- TP1 telemetry and BI1 packet work.
- TK1 PE3/cheapdct4 consumer build.
- LA1 LR-anneal build.
- DY2 plateau-tail EMA build.
- WP1 Muon/MC-finisher/readback ownership.
- WL1 LB/S_R/EIK top fire orders.

QUEUED-WITH-FIRE-ORDER:

1. WL1-LB analytic lane-band training-lever A/B.
2. WL1-SR exact `S_R` reachability weighting A/B.
3. WL1-EIK fixed-guard eikonal/viscosity fair reopen.
4. PT2 seg-force duty-to-measure set, one-at-a-time after higher-ranked boundary items.
5. #924 existence-hinge A/B after BI1 smoke or frame0-repaired base boundary.
6. Stub consumer-build wave for AX1/FH1/PH3 only if MAIN names those missing flags as current.

Pointer delta: none. Own-vehicle frontier remains
`S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory] n600`; contest pointer
borrowed/unmoved.
