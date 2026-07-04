# Sweep B — TRIALITY (DAG · equations · DSL gauges) + MEMORIES lever ledger

**2026-07-04. MAX-SIGNAL PARANOIA SWEEP, surface B of 3.** Exhaustive per-lever status table +
ORPHAN / UNDOCUMENTED / triality-drift flags, to gate the **fresh seeded run**. `$0` research,
READ-ONLY, NO heavy/paid/GPU, #205 (pid 29129) untouched.
**Pointer contest-CPU 0.19110 UNMOVED — every lever below is MEANS.** NO-FAKE: no GO/NOGO is
fabricated; each cites the artifact/number it came from, or is honestly marked
CONJECTURE / A-B-owed / needs-BUILD / HELD / PENDING.

Status legend: `MEASURED-GO` · `MEASURED-NOGO` · `A-B-owed` (built, load-bearing number not yet
measured through the real target) · `needs-BUILD` (no working flag/code) · `needs-CALIBRATE` ·
`CONJECTURE` · `SUPERSEDED` · `HELD` (measurement deliberately deferred).

## 0. Anchors used for this sweep (the cross-check substrate)

- **Live #205 baseline (pid 29129, READ-ONLY):** the byte-identical baseline the fresh-run A/Bs layer
  on. Already runs: `--curriculum --tau-softplus-start-epoch 300 --tau-softplus-tau 0.3
  --l7-start-epoch 1000` (l7 OFF) `--muon-start-epoch 726 --muon-lr 0.002 --muon-momentum 0.95
  --muon-ns-steps 5 --stage-transition-rewarmup-epochs 8 --stage-transition-rewarmup-floor 0.1
  --stage-transition-rewarmup-shape linear --stage-transition-reset-moments --w-seg 100 --w-pose 1.0
  --score-domain-loss --pose-carrier --pose-carrier-residual-mode table --pose-carrier-source generated
  --mod-dim 32 --hidden-dim 96 --n-hidden 4 --activation hosc --hosc-beta 1.0 --hosc-beta-end 4.0
  --hosc-beta-anneal linear --siren-init --softmax-temp-start 1.0 --softmax-temp-end 0.05
  --tau-anneal-shape cosine --self-orient --n-dir-freqs 2 --freq-across 32 --freq-along 4
  --reorient-every 50 --max-bank-freq 64 --chroma --palette-anchor --eikonal-weight 0.01
  --length-weight 0.001 --render-aa none --lane-render-band --lane-band-start-epoch 300
  --lane-band-* --persistence-loss-weight 1.0 --persistence-recall-weight 1.0 --cldice-iters 5
  --persistence-warmup-epochs 300 --amplify-weight 1.0 --amplify-form hinge --island-dilate-px 1
  --structured-init --structured-init-include-lane --lane-prior-phi1 --lane-prior-phi1-mode replace
  --lane-prior-phi1-dash-gate --accum-pairs 8 --ema-decay 0.997 --lr 1e-3 --lr-end 1e-4
  --stage-checkpoints`.
  **Note:** #205 ALREADY runs partial transition-easing (rewarmup 8 / linear); the fresh-run config
  plan *refines* it to 20 / cosine + deconflict band@350. And #205 ALREADY runs store-nothing-xi
  (`--pose-carrier-source generated`) — the 07-02 ledger orphan #2 (store-nothing) is now WIRED into #205.
- **Fresh-run config plan** = `.omx/research/deepmath_converged_next_run_config_20260704.md` (task #285):
  Tier-1 (flags exist) transition-easing + Γ-τ-eikonal; Tier-2 live #270 muon warm-start@726;
  Tier-3 (needs build) NTK-whitening + geometry-native solvers.
- **Trainer flag surfaces (grep-verified):** LEVELSET launch-path trainer
  `experiments/train_levelset_witness_realized_through_R_mlx.py` (~189 flags) — the launch path;
  BASE `experiments/train_witness_realized_through_R_mlx.py` (~87 flags) — shared primitives.
- **Orphan discipline:** Catalog #396 (`check_measured_win_findings_are_wired_or_research_only`,
  WARN-ONLY) + `tools/audit_orphaned_measured_wins.py`.

---

## 1. Deep-math laws (9 objects) + 5 scaling-law facets → lever → built/measured/included

Source: `deepmath_amortizing_argmax_paper_draft_20260704.md` §2 + `src/tac/canonical_equations/
deepmath_amortizing_argmax_laws_20260704.py` (`ALL_DEEPMATH_AMORTIZING_ARGMAX_BUILDERS`) +
`scaling_law_facet{1..5}_*_20260704.md`. **Enumeration nuance (a real triality-drift):** the paper §2
8-law list and the registry 8-builder list **differ by one member** — the registry DROPS
`se3_screw` (registered externally as `store_nothing_pose_carrier_rate_collapse_vs_dpose_v1`) and ADDS
`annulus_anisotropy_magnitude_disputed_v1` → **9 distinct law-objects.** The "5 facets" also grew 4→5
(facet-5 dynamic-control added later; facet1/2/3 headers still say "of 4").

### Table 1A — the 9 deep-math laws

| law_id | implied lever | status | cited number | flag / ORPHAN |
|---|---|---|---|---|
| `maslov_dequantization_bound_v1` | τ-anneal curriculum | needs-CALIBRATE | bound `[0, τ·lnK]`, `τ_end 0.05·ln5=0.0805 nats` (derived) | `--softmax-temp-* --tau-anneal-shape` |
| `fisher_curvature_equals_categorical_fisher_trace_caustic_v1` | margin-saliency (margin=byte-faithful Fisher) | **MEASURED-GO** | Pearson **0.978** band / Spearman 0.908 / min top1−top2 diff **0.0 over 118M px** | `--margin-saliency-*` |
| `ce_softmax_mirror_descent_natural_gradient_v1` | CE-on-softmax loss; ANTI-lever: do NOT build output-space F⁻¹ | CONJECTURE (theorem derived; trajectory-IS-NG unregistered) | matches CE→τ −21.6% | loss/curriculum default (anti-lever correctly ORPHAN) |
| `shearlet_nterm_upper_bounds_task_rate_v1` | `--self-orient` directional (shearlet) basis | **MEASURED-GO (constant)** / A-B-owed (exponent) | **−48% all-class** d_seg (n96 advisory); −48% is a POINT not a SLOPE | `--self-orient --n-dir-freqs --freq-across --bank-n-scales` |
| `tau_eps_hbar_one_dequantization_two_scales_v1` | raise `--eikonal-weight` + τ floor | A-B-owed | derived Γ-coincidence; raise 0.01→0.05 measurement-gated | `--eikonal-weight --softmax-temp-end --tau-anneal-shape` |
| `multiphase_modica_mortola_perimeter_gamma_limit_v1` | `--length-weight` (coarea perimeter) small | needs-CALIBRATE | theorem; length shipped-small (0.001) as MCF-erosion driver | `--length-weight --eikonal-junction-relax` |
| `mcf_minority_erasure_inevitability_v1` | per-class area/volume constraint (auction-MBO) | needs-BUILD (surrogate only) | MBO probe **95.7% of smoothing cost = Lane** (the #205 failure mechanism) | surrogate `--lane-thin-weight --persistence-*`; **hard auction-MBO = ORPHAN(no-flag)** |
| `annulus_anisotropy_magnitude_disputed_v1` | directional/self-orient basis | needs-CALIBRATE | anisotropy **9.56:1** grad-proj / **37.8:1** struct-tensor MEASURED; **198:1 DISPUTED — do NOT quote** | `--self-orient --freq-across --n-dir-freqs` |
| `se3_screw…` (ext. `store_nothing_pose_carrier_rate_collapse_vs_dpose_v1`) | store-nothing pose carrier (ξ) | **MEASURED-GO (rate)** / A-B-owed + HELD (d_pose) | rate **13.9× cut, bit-exact** (#257); **witness d_pose OPEN/HELD** (3.4e-5 is ABANDONED ancestor, never witness-validated) | `--pose-carrier --pose-carrier-source/-residual-mode/-s-r/-s-t` |

### Table 1B — the 5 scaling-law facets

| facet | implied lever | status | cited number | flag / ORPHAN |
|---|---|---|---|---|
| F1 metric preconditioning | Muon (constant) **+ NTK/feature-Gram whitening (exponent)** | MEASURED-GO (Muon) / **needs-BUILD** (whiten) | Muon **−32%** d_seg constant; exponent unmeasured | Muon `--muon-*`; **NTK `--whiten` = ORPHAN(no-flag), grep-confirmed absent** |
| F2 intrinsic-manifold param | `--mod-dim 19` (rate) + bank + self-orient + **parabolic-shearlet front-end** | MEASURED-GO (mod-dim rate) / A-B-owed (exponent) / needs-BUILD (front-end) | m≈8 measured (AE-knee 8/MLE 13/TwoNN 11) → mod-dim 32→19 ≈41% fewer code DOF | `--mod-dim --bank-* --self-orient`; **parabolic front-end = ORPHAN/needs-BUILD** |
| F3 separatrix/asymmetry seed | **`--lane-prior-phi1-mode paint`** (paint-then-SDF) + structured-init | **needs-BUILD** (paint mode); replace-mode = MEASURED-NOGO | paint-then-SDF lane_FN **0.0058→0.0019 (3×)** n96; `--mode replace` measured **no-op** | flag exists but `choices=[replace,bias]` only (verified L4700) → `paint` = **needs-BUILD** |
| F4 adiabatic schedule + nucleation | `--tau-anneal-shape geometric` + τ floor + seed + eikonal 0.05 + length small + area force | A-B-owed | schedule $0-confirmed (info/octave CV≈0.39); survival knee σ=0.8→94% σ=1.5→49% +2px→98%; net-S CONJECTURE | BUILT `--tau-anneal-shape --seed-islands --island-dilate-px --eikonal-weight --length-weight --lane-thin-* --persistence-* --stage-transition-*`; hard area-MBO = ORPHAN |
| F5 dynamic control / self-convergence | `tools/witness_control_monitor.py` creep-detector + `--margin-saliency-reachability` (S_R costate) | **needs-BUILD** (monitor) / needs-CALIBRATE (thresholds) | #205 τ-creep MEASURED `d_seg 0.004752→0.006568` (ep300→400) while loss falls = diverging branch | costate flags BUILT `--margin-saliency-reachability --async-verdict --stage-transition-*`; **`witness_control_monitor.py` = ORPHAN/needs-BUILD** (sensor `tools/render_witness_trajectory_dynamics.py` EXISTS but un-consumed) |

**Coverage bottom line:** the two strongest *measured* laws (Fisher=caustic 0.978; shearlet −48%) are
built + MEASURED-GO. The decisive *unbuilt* items cluster on **nucleation/erasure** — the exact
measured cause of the #205 lane creep: L7 hard per-class-area cure (ORPHAN), F3 `paint` seed mode
(needs-BUILD, measured 3× FN drop), F5 creep-detector (needs-BUILD). The biggest *unmeasured* claim is
the **EXPONENT** story (L4/F1/F2): everything measured so far is a CONSTANT win; whether any lever
changes the scaling *exponent* is A-B-owed behind ONE $0 log-log-slope pre-metric that has not run.

---

## 2. DSL gauges (`src/tac/witness_dsl/gauge.py`, 26 gauge Enums) → argv → cost → status

The gauge layer = 3 surfaces: **STORE charts** (`GaugeChoice`/`CANONICAL_GAUGE`-selectable, carry
archive bytes) · **accessor-only loss/optimizer facets** (0 bytes) · **design-stage** (accessor RAISES,
never-invent-flags). No live invented-flag drift: every argv a gauge accessor emits exists in the
189-flag levelset trainer.

### Table 2A — STORE-component gauges (`GaugeChoice`-selectable, carry bytes)

| chart | argv | cost | status | source |
|---|---|---|---|---|
| WarpGauge.SCREW_TWIST | code wire-in (0B) | bytes=0; through-R d_seg≈0.00477 n96 | MEASURED-GO (CANONICAL) | FEED-jk/jj |
| WarpGauge.PER_CLASS_HOMOGRAPHY | — | bytes=6600 | MEASURED-NOGO (~85% clip-overfit) | FEED-ja |
| WarpGauge.LEARNED | — | None | needs-CALIBRATE | — |
| CarrierGauge.SINGLE_SDF | render carrier | d_seg_R **0.0319** | MEASURED-GO (CANONICAL) | F1 FEED-iw |
| CarrierGauge.HARD_BITMAP | — | d_seg_R **0.166** (5.2×) | MEASURED-NOGO (Gibbs) | F1 |
| CarrierGauge.MSDF | — | None | needs-CALIBRATE (probe running) | — |
| ResidualGauge.CONDITIONAL_ON_LANE_PRIOR | Wyner-Ziv head-start | bytes=5000; d_seg_R **0.00214** base floor | MEASURED-GO (base) / A-B-owed (learned on top) | 389f84f6f |
| ResidualGauge.DIRECT_LEARNED | — | None | needs-CALIBRATE (**THE binding move**, GPU-pending, target ≤1.23e-3) | — |
| ResidualGauge.{ALARD_LUPTON, PERSISTENCE_EVENTS} | — | None | needs-CALIBRATE | — |
| PoseGauge.RANGE_DELTA | sidecar codec | bytes=**875** | MEASURED-GO (CANONICAL pose) | F4 |
| PoseGauge.SCALAR_STORE | — | bytes=4800 | MEASURED-NOGO (dominated) | F4 |
| PoseGauge.LOW_RANK | — | None | needs-CALIBRATE (#140) | — |
| PoseGauge.WARP_REAL_LUMA | frame0-warp | **NO COST CELL → validate() RAISES** | **DRIFT** + SUPERSEDED (xi now pure-pose, dual-use refuted for lanes) | lane_band_source_reparam_measured_resolution_v1 |
| PoseGauge.STORE_NOTHING_XI | `--pose-carrier-source generated` (byte-close) | bytes=7200 (proxy); BIT-EXACT max_abs=0 | A-B-owed (d_pose; classmean proxy 4.97 pre-residual; #205-gated) | 18927a1ae — **now WIRED in live #205** |
| MovablesGauge.STORE | — | bytes=2700; d_seg_R **0.0** | MEASURED-GO (STORE not predict) | F3 FEED-je |
| MovablesGauge.WARP_PREDICT | — | d_seg_R 0.00082 | MEASURED-NOGO (dominated) | F3 |
| GenerationGauge.DETERMINISTIC_FREE | rule-118 inflate.py (0B) | bytes=0 | MEASURED-GO (CANONICAL) | README:118 |
| GenerationGauge.LEARNED_COUNTED | — | None; determinism gate fails | needs-CALIBRATE | rule-118 |
| RenderAAGauge.NONE | `--render-aa none` | baseline | **IS the default launch** | #224/#220 |
| RenderAAGauge.SUPERSAMPLE_2X | `--render-aa supersample --aa-supersample 2` | d_seg_R 0.00086 `[advisory NON-PROMOTABLE]` | **MEASURED-NOGO for witness** (real-frame CEILING; brute supersample HURTS witness −49%; n600 OOM) | FEED-ly/ma; aa_feasibility_reconciliation |
| RenderAAGauge.{SUPERSAMPLE_3X, IPE} | `--render-aa {supersample 3, ipe}` | None | needs-CALIBRATE | — |
| LaneGauge.BAND_RENDER_AUTHORITY | `--lane-render-band` (+all `--lane-band-*`) | None (measured=False; adv base d_seg~0.00087) | A-B-owed (through-R GPU-pending; **IS in all-levers launch**) | FEED-dv #203/213/215 |
| HeadGeometryGauge.SOFTMAX | `--head softmax` | baseline | baseline | default |
| HeadGeometryGauge.ETF | `--head etf` | None | needs-CALIBRATE (rate-win; d_seg pending) | #218 |
| HeadGeometryGauge.ADDITIVE_MARGIN | `--head additive-margin --additive-margin 0.5` | None | needs-CALIBRATE | #218 |
| HeadGeometryGauge.MENON_LOGIT_ADJUST | `--logit-adjust-per-class` | None | needs-CALIBRATE | #218 |

### Table 2B — accessor-only loss/optimizer facets (0 archive bytes; NOT `GaugeChoice`-selectable)

| chart | argv | status | source |
|---|---|---|---|
| MarginSaliencyGauge.TEXTURE_PROXY (UniWARD) | `--margin-saliency-uniward` | **MEASURED-NOGO (INERT)** — texture ⊥ through-R reachability (Pearson −0.033 vs S_R); mildly misdirects | f99a3863a |
| MarginSaliencyGauge.THROUGH_R_REACHABILITY | `--margin-saliency-reachability` | A-B-owed (S_R 3.0× on fragile band; SUPERSEDES inert texture; #205-gated) | f99a3863a; precompute_sR_reachability.py |
| MuonMomentumGauge.WARM_START | `--muon-warm-start-momentum` | A-B-owed (removes +0.000357 cold-start spike; arm #270); NG-attribution=CONJECTURE | cba2e4375 |
| MuonLRGauge.ANNEAL_LR | `--muon-lr-final-frac 0.1` | A-B-owed (cosine 0.002→2e-4; arm #270) | cba2e4375 |
| AlongTangentFrequencyGauge.N_DIR_FREQS_4 | `--n-dir-freqs 4` | A-B-owed (**#1-ranked ep300+ lever**; #205-live baseline is the `2` DEFICIT, 3.2× along-tangent deficit MEASURED) | FEED-03t |
| VectorFieldMarginSaliencyGauge.VECTOR_T_SUBPIXEL | `--seg-subpix-boundary-weight 5.0` | A-B-owed (LEVER-4b BUILT, probe GREEN A==B; net owed) | separatrix_asymmetry_t_..._v1 |
| ChromaBoundaryGauge.CHROMA_ACTIVE | `--chroma` (default ON) | MEASURED-GO baseline (proven independent d_seg DOF) | a3e9f0bd |
| ChromaBoundaryGauge.LUMA_ONLY | `--no-chroma` | MEASURED-NOGO (ablation flips 7.54% Lane→Road) | a3e9f0bd |
| ChromaBoundaryGauge.ANNULUS_CHROMA_SHARPEN | `--seg-chroma-boundary-weight 0.05` | A-B-owed (LEVER-4c BUILT, GREEN A==B; net owed) | a3e9f0bd |
| FlickerTreatmentGauge.DOWNWEIGHT_IRREDUCIBLE | `--seg-spike-reweight --seg-spike-downweight 0.25` | A-B-owed (BUILT #274; STANDING seg play if Lever-D NO-GOes) | #274 |
| FlickerTreatmentGauge.REPLICATE_PREDICTABLE | accessor RAISES | needs-BUILD / NOT-WARRANTED (only 11.4% predictable, ego r=0.16) | a949ff63 |
| FlickerTreatmentGauge.STORE_REGIONAL_LEVERD | accessor RAISES (`--seg-flip-residual` unbuilt) | needs-BUILD (#279; even optimistic net-S −0.35 leaves S~0.40 ≈2× pointer — NOT a pointer move) | leverd_flicker_..._economics_v1 |
| TripleJunctionMarginGauge.TOP1_TOP2_SCALAR | `()` | MEASURED-GO (n600: scalar IS exact distance-to-flip) | a4c66f2f |
| TripleJunctionMarginGauge.MULTICLASS_SIMPLEX | accessor RAISES | MEASURED-NOGO (adds no flip-onset DOF; BANKED) | a4c66f2f |
| GammaTauEikonalGauge.GEOMETRIC_TAU_FLOOR_EIKONAL | `--tau-anneal-shape geometric --softmax-temp-end 1.0 --eikonal-weight 0.05` | needs-CALIBRATE (config-B #285; UNMEASURED; #205 A/B; GO-gated) | tau_eps_hbar + modica_mortola |
| StageTransitionEasingGauge.DECONFLICT_REWARMUP | `--lane-band-start-epoch 350 --stage-transition-rewarmup-epochs 20 --stage-transition-rewarmup-floor 0.1 --stage-transition-rewarmup-shape cosine` | needs-CALIBRATE (motivated by MEASURED FEED-ft ep300 bump 0.0056→0.020; config UNMEASURED; GO-gated) | ce_softmax + muon_finisher |
| PoseTrainingGauge.{H1,D1,E1} | accessor RAISES; intended `--pose-xi-warmstart / --pose-disjoint-frame / --pose-kkt-tube` | **needs-BUILD** (design-stage; survey: NONE beats each alone, all complements) | pose_in_training_lever_survey_verdict_v1 |

---

## 3. DAG FEED-* levers (`sub015_DAG`, 273 FEED blocks; special-focus tail FEED-03t…04c)

Pointer 0.19110 UNMOVED across all 273 blocks; no lever has produced a byte-closed exact row below it.
All d_seg/d_pose are `[advisory · NON-PROMOTABLE]` unless tagged byte-closed.

### Tail (2026-07-02..04) — the fresh-run drivers

| lever / finding | FEED | status | number | flag / ORPHAN |
|---|---|---|---|---|
| Along-tangent freq deficit = THE lane-dash root | 03t | MEASURED-GO root / A-B-owed | 3.2× short (freq_along ≤8 vs dash ~25 cyc); #205 runs `--n-dir-freqs 2`=deficit | `--n-dir-freqs 2→4 --freq-along --bank-n-scales 4→5` |
| R is ALL-PASS, not the wall | 03t | MEASURED-NOGO (kills AA spend) | MTF 0.997@10px, 0.00 only @2px; noR 0.00424≈R 0.00423 | retires `--render-aa`/`--aa-*` spend (ORPHAN) |
| Gibbs ABSENT (hosc working) | 03t | MEASURED | top-prob min 0.318 > K=5 floor | `--hosc-* --activation` (keep annealed) |
| SDF decoupled from score (RGB/tex carries) | 03t | MEASURED (guard) | SDF-argmax 0.103 vs realized 0.005 (20×) | design constraint (structured-init won't move score alone) |
| vector-t separatrix-asymmetry saliency | 03t | A-B-owed (GREEN) | self-consistency +0.56/+0.85; eq `separatrix_asymmetry_t_subpixel…v1` | `--seg-subpix-boundary-weight` (VECTOR-t built LEVER-4b) |
| chroma decides lane+movable at annulus | 03t | A-B-owed (GREEN) | const-luma flips 7.54% Lane→Road + 4.38% Movable→Undriv; eq `chroma_decides…v1` | `--chroma --seg-chroma-boundary-*` |
| Lever-D flicker-residual reactivation | 03u/v/w/x | MEASURED-NOGO | net-S −0.35 opt/−0.048 exp/+0.117 pess; Stage-0 coder floor min(b)=0.876 B/flip > 0.65 GO; eq `leverd_flicker…economics_v1` | ORPHAN (residual coder); `--seg-spike-*` for downweight only |
| 8 deep-math laws registered | 03y/03z | PROVEN (equations leg) | 8/8 JSONL commit e4524c94d; 24+301 tests | ORPHAN (laws) |
| NTK/multiscale band-pass whitening | 03y | needs-BUILD | ~3-10× speed, up to −3e-4 d_seg | ORPHAN (no `--ntk`/`--whiten`; excluded from #285 argv) |
| Ch.4 geometric-τ + float τ_end 0.05→~1.0 + eikonal 0.01→0.05 (COUPLED) | 03y/04c | A-B-owed | 0.05=40× sub-grid aliasing waste; makes τ real interface width | `--tau-anneal-shape geometric --softmax-temp-end --eikonal-weight` |
| Ch.6 transition-easing (deconflict ep300 + rewarmup) | 03y/03z | A-B-owed (BUILT default-off) | ep300 collide CE→tau+band → d_seg 0.0056→0.020 (3.4×) | `--lane-band-start-epoch 350 --stage-transition-rewarmup-*` |
| Ch.6 L7 MD-Decoupling optimizer | 03y/03k | MEASURED (BUILT; arXiv 2606.25971) | stage-transitions stable-by-construction | **ORPHAN `--optimizer md` NOT in levelset surface = DRIFT** |
| damped-Newton semi-discrete OT head-offset b* | 04a | needs-A/B (BUILT, $0-gate HELD) | KKT 1e-11; `damped_newton_ot_offsets` (8bc91449c) byte-free | ORPHAN (`apply_offset_to_sdf_bias`; ≈`--logit-adjust-per-class` unwired) |
| solver queue: auction-MBO, Airy profile, RKMK ξ-transport, FISTA shearlet | 04a | needs-BUILD | each $0-gated | ORPHAN |
| **Lane NUCLEATION failure = d_seg-creep root** | 04b | MEASURED (decisive) | lane+movable seeded ZERO; d_seg 0.004752@ep300→0.006568@ep400 (+38%); MCF critical-nucleus | `--structured-init-include-lane --seed-islands --island-dilate-px` |
| SEED wider +2px above nucleus | 04b/04c | CALIBRATED ($0 n24 probe) | σ=1.5 native 44.6%→+1px 90.0%→**+2px 98.3%**; MOVABLE native 99.5% (no dilation) | `--island-dilate-px 2 --structured-init-* --seed-islands` |
| **Facet-2 mod-dim: use 19 not 32** | 04c | MEASURED-GO (config) | intrinsic m≈8→Whitney 17-19; 32=−41% code DOF at equal d_seg | `--mod-dim 19 --bank-n-scales 6` |
| Facet-4 geometric-τ = unique adiabatic schedule; length small IS MCF-erosion driver | 04c | derived-GO | g_ττ=Var/τ⁴, τ_c=0.242·m; keep length 0.001 + eikonal 0.05 + dilate +2px = inversion of #205 erosion | `--tau-anneal-shape --length-weight 0.001 --eikonal-weight --island-dilate-px` |
| Facet-5 costate / τ-creep detector | 04c | needs-BUILD (facet-1/5 landed) | Lyapunov Tier-A OT-dual gap; `witness_control_monitor.py` emits decisions only | ORPHAN (tooling #247) |
| **openpilot lane prior through R = at goal** | 04c | MEASURED-GO (#138) | d_seg ≈ **0.00087 = the 0.00092 goal**; witness fails only by seeding lane 0 + eroding | `--lane-prior-phi1* --lane-render-band --structured-init` |

### Body (2026-06-25..07-01) — foundational + realization

| lever / finding | FEED | status | number | flag / ORPHAN |
|---|---|---|---|---|
| **All-class directional (self-orient) basis = THE d_seg lever** | 25t | MEASURED-GO | **−48%** all-class (lane-only −8%); 0-byte, compiles into inflate.py FREE | `--self-orient --n-dir-freqs` |
| Capacity-AFTER-basis-match | 25t/u | MEASURED-GO | relu-cap **0.002447 (−70%)**; capacity-alone +6% HURTS/diverges | `--hidden-dim --mod-dim --n-hidden` (ORDER law) |
| KD soft-logit c1 (kd_w 0.3, T2.0) | 25x | MEASURED-GO (marginal) | 0.002423 (−1.0%); KD-only diverges | ORPHAN (KD flags in smoke tool, not levelset surface) |
| Hard-pixel error-boost {3,9} | 25w | MEASURED-NOGO | +3-5× worse; redundant w/ margin-hinge | `--hardness-weighted/-power/-source` (NOGO) |
| gauss/step activation (as-impl at relu-lr) | 25v | MEASURED-NOGO (not paradigm kill) | 0.004656 then diverged 0.0091 | `--activation gauss` |
| #257 store-nothing pose carrier byte-close | 03b/snx | MEASURED-GO | rate 0.0347→**0.00250 (13.9×)**, bit-exact, d_pose-invariant; section 1049 B vs warp 697941 B | `--pose-carrier-source generated --pose-carrier-mode store_nothing` |
| Wave-F LBND2 lane-band RD codec | wfs1 | MEASURED-GO (rate) | 156,340→**41,526 B (3.76×)**; Shannon floor 26,179 B; eq `lane_band_camera_frame_rd_rate_v1` | `--lane-render-band --lane-band-*` (byte-close tool) |
| OOM verdict-batch chunk | oom | MEASURED-GO (score-neutral) | +66→6 GiB; d_seg bit-identical; eq `oom_verdict_batch_spike_peak_rss_v1` | `--verdict-batch 32` |
| sg-cache clDice bit-identical free win | 03g | MEASURED-GO (free) | 339→275ms (1.23×); max_abs_diff 0.000 / 31 arrays | `--cache-gt-skeleton` |
| Muon KEEP + tune finishing schedule | 03o | MEASURED-GO | −32% d_seg vs AdamW same fork; κ≈19; successors=scale-plays | `--muon-lr/-momentum/-ns-steps/-weight-decay` |
| Muon finishing levers (LR-decay+warm-start) BUILT | 03q/s | A-B-owed (byte-identical off; GO to fire #270) | cold buffer +0.000357 spike; #270 armed ep726 | `--muon-lr-final-frac --muon-warm-start-momentum` |
| S_R reachability lever BUILT+WIRED | 03p | A-B-owed | sR 3.0× on fragile band (0.35 vs 0.12); active byte-identical-off | `--margin-saliency-reachability` |
| LEVER-4 msal_uni texture proxy INERT | 03n | MEASURED-NOGO (texture) | Pearson −0.033 (at chance); Jaccard 0.024 vs 0.026 | `--margin-saliency-uniward` (inert) → S_R |
| mx-compile REJECTED | 03b | MEASURED-NOGO | fp-contraction flips argmax across uint8-STE; 1.11× only | `--mx-compile` (rejected) |
| #252 fused-R kernel | 03b/c | MEASURED (parity GO; whole-run ~1.02× Amdahl) | 4.69× fwd+bwd on R, bit-identical | `--fused-r-kernel` |
| L13 task-space format rate | af | MEASURED-GO (rate) | **72,217 B (−59%** vs 177,169), lossless-parity | ORPHAN (byte-close tool) |
| POWERPLAY S-isomorphism | pp | MEASURED-identity | `powerplay_cost(x).S == compute_contest_score(x)` residual 0.0 | ORPHAN (`tac.witness_dsl.powerplay`) |
| task R(D) < reconstruction R(D) | rdd | DERIVED-framing | arXiv 2602.12866; eq `task_rd_dominates_reconstruction_rd_v1` | ORPHAN |
| lane↔road boundary = 57% of ALL flips | perclass | MEASURED (n600) | 447,622 px; recomputed d_seg 0.006655 EXACT | `--lane-render-band` |
| Movable 93.5% / Lane 49.9% self-grown → #208 seed DOWNGRADED | perclass | MEASURED — **TENSION w/ 04b nucleation (see §8 D9)** | recall 0.9346 / 0.4986 | `--seed-islands` |

### DAG MEASURED-GO shortlist (highest confidence)
self-orient −48% · capacity-after-basis −70% · openpilot lane prior **0.00087=goal** · mod-dim 19 (−41% DOF) · #257 pose 13.9× · LBND2 3.76× · verdict-batch chunk · sg-cache clDice · Muon KEEP −32% · render-aa none (supersample −49%).

---

## 4. Canonical equations (221 JSONL + 3 py-only) → implied lever

`tools/list_canonical_equations.py` reads `.omx/state/canonical_equations_registry.jsonl` (221 ids /
472 rows). Of 221, **~35 imply a live witness trainer flag**; the rest are encoder/rate levers,
MLX/composition/apparatus infra, or literature-inferred framing. All witness-equation flags were verified
present in the levelset trainer.

### Witness θ* trainer-lever equations — MEASURED-GO (well-calibrated, includable)

| equation_id | lever → flag | cited number |
|---|---|---|
| `fisher_curvature_equals_categorical_fisher_trace_caustic_v1` (215) | margin=Fisher → `--margin-saliency-* --margin-field-head-weight` | Pearson **0.978**, res 0.0 |
| `shearlet_nterm_upper_bounds_task_rate_v1` (217) | self-orient=shearlet → `--self-orient --n-dir-freqs` | D1 **−48%** all-class VERIFIED (tightness ASSUMED) |
| `mcf_minority_erasure_inevitability_v1` (220) | protect thin-Lane → `--persistence-loss-weight --lane-thin-* --seed-islands` | MBO erases minority first |
| `chroma_decides_lane_and_movable_at_annulus_v1` **(PY-ONLY)** | chroma→annulus → `--chroma --seg-chroma-boundary-weight` | const-luma flips 4.38% Movable→Undriv |
| `theta_star_eikonal_length_boundary_energy_v1` (176) | `--eikonal-weight 0.01 --length-weight 0.001` | live |
| `step_basis_stability_vs_hosc_saturation_v1` (194) + `hosc_activation_saturation_trainability_v1` (198) | `--activation step_basis` / `--hosc-beta-anneal` NEVER fixed β=4 | fixed-β saturates→vanishing-grad |
| `dm1_stiefel_isometry_rank_preservation_v1` (168) | byte-free rank cure → `--code-spectral-entropy-weight --code-nuclear-* --wire-w0` | PR collapse **3.34→1.19** |
| `residual_manifold_intrinsic_dim_whitney_v1` (185) | `--mod-dim` 17-19 (mod-16 under-embeds) | intrinsic ~8 |
| `persistence_topology_cldice_betti_island_recall_v1` (186) + `island_finest_scale_protection_survival_v1` (187) | `--persistence-* --seed-islands --island-dilate-px --amplify-*` | 1 anchor each OK 0.0 |
| `warp_real_luma_frame0_pose_carrier_dpose_v1` (188) + `store_nothing_pose_carrier_rate_collapse_vs_dpose_v1` (212) + `pose_ego_screw_twist_identifiable_up_to_affine_v1` (180) + `pose_sqrt_concave_coupling_sidecar_v1` (175) | `--pose-carrier-source generated --pose-carrier-mode store_nothing --dxi-source --pose-eps` | store_nothing frame0 **bit-exact max_abs=0**; d_pose A-B-owed |
| `detector_informed_recon_weight_d_seg_savings_v1` (162) | `--margin-saliency-uniward --hardness-weighted` | **13 anchors** OK 0.0 |
| `dseg_stretched_exponential_anneal_trajectory_v1` (165) | `--anneal-epochs --tau-anneal-shape` | res 0.0246/0.0174 |
| `argmax_of_sdf_is_additively_weighted_power_diagram_v1` (197) + `aa_sdf_observation_footprint_render_dseg_v1` (199) | `--head --render-aa --structured-init*` | res 0.0013 / 0.0 |
| `oom_verdict_batch_spike_peak_rss_v1` (210) | `--verdict-batch 32` (score-neutral) | +66→+6 GiB |

### A-B-owed (measured grounding; net is #205-class byte-closed A/B)
`along_tangent_freq_deficit*` **(PY-ONLY, #1 ranked ep300+ lever)** → `--n-dir-freqs 2→4` ·
`curvelet_directional_basis_dseg_reduction_v1` (193, synthetic-GT caveat, #185 ladder A/B) ·
`margin_saliency_reachability_replaces_texture_proxy_v1` **(PY-ONLY)** → `--margin-saliency-reachability`
(texture proxy MEASURED-NOGO, align 0.214≈chance) · Muon-finisher law **(PY-ONLY)** →
`--muon-warm-start-momentum --muon-lr-final-frac` (#270) · `analytic_lane_band_dseg_recon_floor_v1`
(192, dash-gate binding) · `pose_in_training_lever_survey_verdict_v1` (204, seg-pose cos 6e-5) ·
`openpilot_unified_physical_prior_both_scored_axes_v1` (208, pose −99% GO / lane build #234).

### MEASURED-NOGO (equations that KILL a lever)
`r_transfer_function_near_all_pass_negative_v1` (196, kills R-deconv) · `l7_linf_sharpening_defect…v1`
(195, DROP l7) · `hosc_activation_saturation…` fixed-β · `aa_supersample_lane_recall_lift_v1` (191,
witness supersample HURTS −49%) · `analytic_lane_render_band_fp_reduction_v1` (189, naive band +0.00082
HURTS through R → reactivate only fine-tuned WITH `--lane-render-band`) ·
`l28_channel_offset_does_not_transfer_to_levelset_witness_v1` (200, kills borrowed PR95 lever) ·
`index_permutation_discontinuity_defeats_temporal_model_v1` (205, corresp-first not predictor) ·
`waterfill_annulus_through_r_store_realization_vs_witness_capacity_v1` (201, STORE-side paste negative) ·
`leverd_flicker_residual_reactivation_economics_v1` **(PY-ONLY)** (min(b)=0.99>0.65 GO → NO-GO).

### CONJECTURE / needs-CALIBRATE
INFERRED-from-literature (anchor-OK but no empirical measurement): `maslov_dequantization_bound_v1` ·
`ce_softmax_mirror_descent_natural_gradient_v1` · `tau_eps_hbar…` · `multiphase_modica_mortola…` ·
`task_rd_dominates_reconstruction_rd_v1` · `independent_jitter_dseg_floor*` · `dm3_natural_gradient…v1`
(0 anchors). **0-anchor registered laws** (~35): score_marginal_lagrange, per_pair_loss_weighting,
ema_decay_substrate_stage_aware, frozen_scorer_fisher_pullback_metric, indirect_rd_logloss=IB,
rate_mdl_cosmological_constant, se3_bspline, movables_out_of_inr, categorical_blahut_arimoto, etc.
**DRIFT residuals** (well-calibrated=False): mps_drift_architecture_class (30.0), per_byte_leverage_uniform
(5.4), triple_substrate_composition_alpha (92.5), procedural_codebook_from_seed (86.08), markov_context
(4-64), pose_axis_score_direction_matching (multiple).

### ORPHAN equations (imply a lever/byte-mover but map to NO trainer flag)
- **Encoder/rate-side (real byte movers, byte-close tool not trainer):** `lane_band_camera_frame_rd_rate_v1`
  (202, LBND2 3.76×) · `lane_band_ego_factorization_source_reparam_v1` (203) ·
  `lane_band_source_reparam_measured_resolution_v1` (206, smoothing win15 −42% LOSSY; ξ pure-pose) ·
  `correspondence_first_lane_coding_optimal_pipeline_v1` (207, #234 unbuilt) · procedural_codebook /
  procedural_predictor / fec8/fec10 Markov / daubechies-wavelet / null_space_byte_fraction / wyner-ziv density.
- **Framing/meta ("NOT a contest lever"):** `task_rd_dominates_reconstruction_rd_v1` (211) ·
  `powerplay_variant_ii_cost_isomorphism_v1` (209) · `witness_unified_action_fixed_fisher_background_v1` (166).
- **Governor / launch-safety:** `adaptive_ceiling_admission_control_v1` (213).
- **Separatrix-t vector saliency:** explicitly UNBUILT (scalar `--margin-saliency-*` exists; vector-t does not).

> **⚠ REGISTRY DRIFT (VERIFIED — the headline flag, see §8 D1):** the 6 freshest **2026-07-03 lever laws**
> (`along_tangent_freq_deficit`, `separatrix_asymmetry_t`, `chroma_decides_lane_and_movable_at_annulus`,
> `margin_saliency_reachability_replaces_texture_proxy`, `leverd_flicker_residual_reactivation_economics`,
> `independent_jitter_dseg_floor`) are **code-live** (imported in `canonical_equations/__init__.py` = 9
> hits, consumed by `witness_dsl/gauge.py` = 8 hits) but **JSONL=0** (grep-verified) — invisible to
> `list_canonical_equations.py` and every cathedral/DSL JSONL consumer. The 07-04 deepmath laws (214-221)
> ARE flushed. So the equation leg of the triality is **blind to the #1 ranked ep300+ lever** and to
> margin-saliency-reachability + Muon-finisher — the exact levers the fresh run turns on.

---

## 5. Memory-file levers (2026-06/07) — the honest MEASURED / HELD state

Every memory holds pointer 0.19110 UNMOVED; witness implied-S ~0.67–0.75 is far above pointer (the
capstone is a long bet). All d_seg/pose numbers are partition-level or through-R ADVISORY until a
byte-closed exact row.

### MEASURED-GO (include in fresh seeded run)
- `--muon-*` KEEP (−32% vs AdamW, same fork) + `--muon-warm-start-momentum` + `--muon-lr-final-frac 0.1`
  (both measured gaps; #270, operator GO) + `--weight-decay 1e-4` decoupled ON.
- `--verdict-batch 32` (OOM fix, bit-identical) + `witness_memory_preflight` refuse >0.70×RAM (BUILT).
- `--render-aa none + --lane-render-band` (analytic coverage AA MEASURED to help; SIGNAL-A ceiling
  0.00086 proves representation floor is BELOW target = gap is TRAINING not representation).
- `--self-orient` directional/shearlet basis BEFORE capacity (−48%; capacity on isotropic basis +6% HURTS).
- `--hosc-*` annealed / step_basis (Gibbs ABSENT, working) — NEVER fixed β=4.
- `--length-weight 0.001 / --eikonal-weight 0.01` LIVE (keep; raise eikonal for nucleation).
- `--lane-band-* source = coherent_slot_none` (lossless 0.5% rate, SAFE default; bounded-K slotting).
- `seg⊥pose FREE` (render-space cos median 5.9e-5, 99.95% pose-null) — enables independent pose arm.
- `--cache-gt-skeleton` (sg-cache clDice, bit-identical free 1.23×).
- `--seg-spike-downweight / --seg-coherent-upweight` BUILT #274 (byte-identical at defaults; benefit A-B-owed).

### MEASURED-NOGO / EXCLUDE
- `--render-aa supersample / --aa-supersample` — witness brute supersample HURTS **−49%** + 41min>30min budget.
- naive warp-only pose carrier — d_pose 1.37–10.53 catastrophic (each contribution exceeds S target).
- ξ-coding of LANES (advection/predictive) — REFUTED (LBND3 1.04–1.34× worse); ξ = pure-pose only.
- P-E per-pair frame0 inverse-solve (1e-8 = existence-proof, per-pair table = NO-FAKE #6/#8) + P-F
  counted-δ coder (n600 rate 2.70–8.56 ≫ R1's 0.105) — EXCLUDE.
- raw UNIWARD as `--margin-saliency-uniward` (at chance −0.033 vs S_R; route via Fisher instead).
- Predictable-REPLICATE flicker (only 11.4% predictable) · optimizer SWITCH (Dion/SOAP/…scale-plays) ·
  l7 in default curriculum (defect) · `--lane-prior-phi1` init as band's only role (CE collapses lane_px=0) ·
  R-deconvolution / output-space F⁻¹ NG / sub-2px basis refine (≤+1.25dB, below stem Nyquist).

### A-B-owed / needs-BUILD (high-EV, pointer-gated) — the fresh-run queue
- **SEED EARLIER dilated +2px** (`--seed-islands --structured-init* --island-dilate-px 2 --seed-blend`) —
  nucleation_failure (07-04) physics-required; **requires a FRESH run (cannot retrofit #205 by resume)**.
  Probe: σ=1.5 native 44.6%→+1px 90.0%→+2px 98.3%; MOVABLE native 99.5% (no dilation).
- **RAISE eikonal 0.01→0.05 COUPLED with geometric-τ** + **per-class area constraint / auction-MBO** (pin
  lane mass ≠0) + **Ch.6 easing** (`--lane-band-start-epoch 350 --stage-transition-rewarmup-epochs 20`)
  to deconflict the MEASURED ep300 τ+band collision (harm 3.4×).
- **DASH-GATE, range-dependent** (`--lane-band-* --lane-thin-*`) — kills the 90%-dominant FP; through-R
  re-confirm of the 0.00087 band owed.
- **`--n-dir-freqs 2→4` + `--freq-across 8`** (Nyquist cap) — along-tangent dash deficit at its root (~0 byte).
- **exact S_R reachability weight** (`--margin-saliency-reachability`) — replace inert texture; MODEST, UNPROVEN.
- **#217 post-Muon leap-residual reheat micro-stage** — muon deep-dive SINGLE HIGHEST-EV; **ORPHAN (no flag)**.
- **P-B FiLM stored-target read-back** (`--film-* --pose-carrier-residual-mode film`) — pose CRUX (90:1
  costate); read-back UNMEASURED. Pose is OPEN/HELD on the witness (see below).
- clDice persistence loss (`--persistence-*` wired) + logit-adjust (`--logit-adjust-*`) + ETF head (ORPHAN)
  + `--code-nuclear-*` low-rank code (rate) + L3 NTK band-pass whitening (~3-10× speed, **ORPHAN**).

### HELD / CONSTRAINTS (preserve the honesty)
- **⛔ Pose OPEN + UNMEASURED on the witness** — warp alone d_pose 1.37–10.53; 3.4e-5 is the ABANDONED
  ANCESTOR (never witness-validated); #205 died OOM before measuring. Do NOT cite pose SOLVED / 3.4e-5.
  R1 store-nothing through-R d_pose 62→0.0011 (contrib 0.105, ~6× ancestor) is ADVISORY; byte-close #238 owed.
- **MLX-GPU byte-close NOT bit-identical cross-process** (28/28 tensors diverge) → CPU verdict authority.
- **‡ SEEDING TENSION (resolve for the run):** #209 (07-01, n600) DOWNGRADED early-seed to "accelerant"
  (Lane self-grew to 49.9%, Movable 93.5% under CE); the NEWER nucleation_failure (07-04) RE-ELEVATES
  seed to **physics-required** for the τ-stage d_seg CREEP (CE grows lane, then τ's MCF erodes the
  sub-critical thin lane back). Consistent once scoped: seed keeps lane ABOVE the critical nucleus so MCF
  grows not erases. The 07-04 nucleation read is the decisive newer one for a fresh run.

### ORPHAN levers with no flag (from memories)
#217 leap-residual micro-stage · Lever-D margin-conditional temporal flip-residual coder (#72/#226,
store-side) · ETF/additive-margin head · L3 NTK/multiscale band-pass whitening · render-at-874 / mod-dim
reduction (config knobs, `--mod-dim` derived-optimum not a lever) · per-class area constraint (auction-MBO,
`damped_newton_ot_offsets` unwired) · `--hardness-*` (appears in flag list but **NO cited number in any
06/07 memory → CONJECTURE/unsupported**).

---

## 6. ORPHAN LIST (highest-value output — levers at risk of being FORGOTTEN)

Two distinct orphan classes, kept separate:

### 6A. LEVER-level orphans (a real, often MEASURED, lever with NO trainer flag / unwired) — the paranoia core

| # | orphan lever | evidence it matters | why orphan | disposition |
|---|---|---|---|---|
| O1 | **Hard per-class-area / auction-MBO constraint** (L7/F4 cure) | MBO probe **95.7% smoothing cost = Lane** = the MEASURED #205 lane-creep mechanism | NO trainer flag; `damped_newton_ot_offsets` exists at `laguerre_logit_offset.py:177` but is NOT wired as a loss; only the persistence/lane-thin SURROGATE is wired | needs-BUILD; the principled cure to the measured erasure |
| O2 | **`--lane-prior-phi1-mode paint`** (F3 paint-then-SDF seed) | measured lane_FN **0.0058→0.0019 (3×)** n96; the current `replace` mode is a measured **no-op** | flag exists but `choices=["replace","bias"]` only (verified L4700); `paint`≈10-LOC build | needs-BUILD; single decisive measured seed fix |
| O3 | **NTK / feature-Gram per-scale whitening** (F1 exponent lever) | the ONLY lever that could change the scaling EXPONENT (highest theoretical ceiling) | no `--whiten`/`--ntk` flag (grep-confirmed 0 in both trainers) | needs-BUILD (Tier-3, config plan) |
| O4 | **`tools/witness_control_monitor.py`** (F5 creep-detector) | #205 τ-creep MEASURED `d_seg 0.004752→0.006568` while loss falls = diverging branch | tool ABSENT; plant sensor `tools/render_witness_trajectory_dynamics.py` EXISTS but is un-consumed | needs-BUILD (design-only) |
| O5 | **Parabolic-shearlet front-end** (F2 `width≈length²` spatial support) | shearlet-rate law's proper representation (curvelet bank has angular law but no parabolic window) | no front-end module / no flag | needs-BUILD |
| O6 | **Geometry-native SOLVERS** (config-plan §Tier-3): damped-Newton semi-discrete OT head-offset · auction-MBO volume-preserving flow · Airy caustic asymmetry profile · RKMK Lie-group ξ-transport | named theory-optimal cures in the deep-math lenses | NONE has a trainer flag, gauge, or build (grep-confirmed absent in trainer + gauge) | CONJECTURE / needs-BUILD; each $0-gated before wiring |
| O7 | **PoseTrainingGauge H1/D1/E1** (in-loop d_pose training) | witness d_pose is OPEN/HELD; these are the survey's complement levers | gauge accessor RAISES NotImplementedError; intended flags `--pose-xi-warmstart/--pose-disjoint-frame/--pose-kkt-tube` unbuilt | needs-BUILD (design-stage, correctly fail-closed) |
| O8 | **MBO decode-regularizer / `curvature_ranks_segnet_margin`** (inflate-side) | curvature↔SegNet-margin **10–40× separation** MEASURED; σ=1.0 removes 11.4% boundary length @ d_seg 0.00087 | CANDIDATE eqn + CANDIDATE DSL lever `mbo_decode_regularizer`, **neither registered** (verified absent today); inflate/decode-side, not a trainer flag | needs-BUILD/CALIBRATE; genuine remaining orphan from the #396 sweep |

### 6B. MEMO-level orphan backlog (Catalog #396 discipline signal — heuristic, over-counts theory memos)

`tools/audit_orphaned_measured_wins.py`: **42 ORPHAN memos since 2026-07-01, 105 since 2026-06-25**
(the classifier flags any memo co-mentioning measured+win tokens without an eqn ref + config pointer).
Many are *theory* memos (deepmath_lens_*, scaling_law_facet*) that are correctly design-only — NOT
lever orphans. **The curated real mechanism-win orphans** (from
`orphaned_measured_win_..._20260702T224934Z.md`), with today's burn-down status:

- **Wave-F LBND2 rate codec (3.76× rate, 0.1041→0.02765):** equation `lane_band_camera_frame_rd_rate_v1`
  **IS now registered** + wired in `tools/levelset_byte_close_and_eval.py` (`serialize_lane_band_rd`) →
  largely burned down on the RATE/byte-close axis (correctly not a trainer flag).
- **Store-nothing keyframe (d_pose 1.12<1.37, ~0 marginal rate):** now WIRED into live #205
  (`--pose-carrier-source generated`) + `serialize_pose_carrier_store_nothing` in byte-close → burned down.
- **Analytic lane band:** eqn `analytic_lane_render_band_fp_reduction_v1` registered; default-OFF in
  `levelset_byte_close_and_eval.py` → MED (needs #205 trained-in d_seg A/B).
- **MBO/curvature (O8 above):** still ORPHAN (candidate eqn/DSL unregistered).
- **Wave-F unified-ξ (−42% rate):** RESEARCH_ONLY (compliant, not an orphan).
- **Sig-proc levers:** NOT-A-WIN (measured-negative R near all-pass) — honest, no action.

---

## 7. UNDOCUMENTED list (config-plan items / running flags not backed by a triality record)

- The fresh-run config plan Tier-1/Tier-2 are FULLY backed: Tier-1 (1) transition-easing →
  `StageTransitionEasingGauge` + eqns `ce_softmax_mirror_descent_natural_gradient_v1` +
  `muon_finisher_schedule_warmstart_and_lr_anneal_v1`; Tier-1 (2) Γ-τ-eikonal → `GammaTauEikonalGauge`
  + `tau_eps_hbar…` + `multiphase_modica_mortola…`; Tier-2 muon warm-start → `MuonMomentumGauge`/`MuonLRGauge`.
  **No UNDOCUMENTED config items in the fresh-run plan.**
- **Base-only reachability gap (launch-path orphans):** the LEVELSET launch-path trainer does NOT expose
  ~37 BASE-only flags. Most are SUPERSEDED equivalents (levelset uses `--self-orient`/`--bank-*` not
  `--basis`/`--n-fourier`/`--fourier-sigma`; `--softmax-temp-*`+`--tau-anneal-shape` not
  `--tau-anneal-start/end`; `--muon-start-epoch`/`--muon-adamw-lr` not
  `--muon-finisher-start-epoch`/`--muon-adam-lr`). **Notable non-superseded base-only levers to confirm
  are not silently lost:** `--md-base`/`--md-gain-lr-scale` (MD-decoupling), the `--margin-weighted-loss`
  + `--margin-weight-fn/temp/start-epoch/anneal` family, and `--plateau-*` (plateau LR scheduler).
  These carry findings but are unreachable from the launch path — verify against DAG/memory tables (§3/§5).
- Live-#205 flags to confirm have a triality record (checked in §3/§5): `--palette-anchor`,
  `--length-weight`, `--lane-prior-phi1*`, `--amplify-*`, `--island-dilate-px`, `--structured-init*`.

---

## 8. TRIALITY-DRIFT flags

- **D1 — ⚠ HEADLINE: JSONL registry drift — the 6 freshest 07-03 lever laws are code-live but
  registry-INVISIBLE (VERIFIED).** `along_tangent_freq_deficit` (the #1 ranked ep300+ lever),
  `separatrix_asymmetry_t`, `chroma_decides_lane_and_movable_at_annulus`,
  `margin_saliency_reachability_replaces_texture_proxy`, `leverd_flicker_residual_reactivation_economics`,
  `independent_jitter_dseg_floor` — all grep **JSONL=0** in `canonical_equations_registry.jsonl`, yet
  imported in `canonical_equations/__init__.py` (9 hits) + consumed by `witness_dsl/gauge.py` (8 hits).
  The 07-04 deepmath laws (214-221) ARE flushed. The JSONL was last written for 07-04 but SKIPPED the
  07-03 batch → `list_canonical_equations.py` + every cathedral/DSL JSONL consumer are BLIND to the exact
  levers the fresh run turns on. **Fix: flush the 07-03 laws to the JSONL (register) before the run so
  the equation leg agrees with DSL+DAG.** (Sister of the #396 orphaned-measured-win discipline.)
- **D2 — mod-dim 32-vs-19 UNRECONCILED conflict.** FEED-205p2 / live #205 ship `--mod-dim 32` ("protect
  binding d_seg, 19-neutrality UNMEASURED"); FEED-04c facet-2 + eqn `residual_manifold_intrinsic_dim_whitney_v1`
  MEASURE 19 optimal (32 = −41% code DOF waste at equal d_seg). The sealed config and the latest
  scaling-law finding actively contradict — **not reconciled in the DAG.** A fresh-run decision (keep 32
  for d_seg safety vs drop to 19 for rate) must be made explicitly, not inherited.
- **D3 — 8-law list mismatch (paper §2 vs registry):** registry drops `se3_screw`, adds
  `annulus_anisotropy_magnitude_disputed_v1` → 9 objects. se3-screw registered externally. Reconcile the
  "8 laws" count across paper/registry.
- **D4 — 5-facet header drift:** facet1/2/3 headers say "of 4"; facet4/5 say "of 5" (pass grew 4→5).
  Also: the 5 scaling facets are a DIFFERENT enumeration from the CLAUDE.md §OPERATOR-PRIORITY physics
  facets (distortion/representation/curriculum/dimensionality/temporal-pose/compute) — do not conflate.
- **D5 — Muon GAP-flag doc-drift:** `MuonMomentumGauge`/`MuonLRGauge` docstrings + accessor docstrings
  assert `--muon-warm-start-momentum`/`--muon-lr-final-frac` are "BASE-only, levelset wire-in owed."
  **Verified false** — both ARE in the levelset trainer (L4647/L4658). Stale comment (not functional).
- **D6 — `PoseGauge.WARP_REAL_LUMA` has NO cost cell** in `default_cost_table` →
  `GaugeChoice(pose=WARP_REAL_LUMA).validate()` would RAISE. A real byte-close-proven carrier with no
  ledger row (its sibling STORE_NOTHING_XI has one).
- **D7 — `TopologyLossGauge` + `IslandProtectionGauge` have NO `*_trainer_flags` accessor and NO cost
  cell** — their chart↔flag mapping (`--persistence-loss-weight`/`--seed-islands`) lives only in the
  docstring; the actual wiring is done directly in `witness_autoconfig`, BYPASSING the gauge layer.
- **D8 — accessor-only gauge layer is NOT on the live launch path:** `witness_autoconfig` sets the FEED-03t
  loss/optimizer flag values DIRECTLY; no production launch-config caller consumes the gauge `*_trainer_flags()`
  accessors (callers found = tests + canonical_equations + triality_drift_detector/dashboard/preflight).
  So the accessor gauge layer is currently DSL-symbolic/observability, not the source of truth for the run.
- **D9 — `--no-chroma` regex latent gotcha:** `real_trainer_flags()` regex captures `--chroma` but not
  the `BooleanOptionalAction`-generated `--no-chroma`; dormant (gauge accessors bypass the validator) but
  any `--no-*` routed through `WitnessProgram.validate()` would be falsely flagged as invented.
- **D10 — Catalog #344 live-count drift 0→480** (strict gate whose backlog silently decayed;
  `preflight_all(strict=True)` currently red on #344) + Catalog #396 flags 15 measured-win ORPHANs since
  2026-07-02. Strict-gate-backlog-rot is itself the meta-bug the #396 landing named. Sister of D1.
- **D11 — `--optimizer md` (MD-Decoupling) is base-only, unreachable from launch path.** Cited BUILT +
  confirmed (FEED-03k/03y-L7; arXiv 2606.25971) but the LEVELSET launch-path trainer exposes no
  `--optimizer` / `--md-base` / `--md-gain-lr-scale` (those are BASE-only). If MD-decoupling is a live
  lever it is orphaned from the launch path — reconcile (either wire into levelset or confirm superseded
  by the `--muon-*` finisher).
- **D12 — corrfirst equation vs its own memory disagree.** `correspondence_first_lane_coding_optimal_pipeline_v1`
  (207) + FEED-corrfirst predict lane-band rate ~0.007–0.012; the CORRECTING memory
  (`project_lane_band_rate_crux_corrected…`) says that was fit-JITTER not swaps (correspondence = 0.5%
  lossless). Registered prediction and its correction disagree; #234 build unbuilt.

<!-- Sections 3/4/5 tables appended from the DAG / equations / memory extraction sweep. -->

## 9. FRESH-RUN GATE SUMMARY (what this ledger recommends)

**INCLUDE (MEASURED-GO, already in or ready for the seeded run):** `--self-orient` (−48%), capacity-after-
basis, `--render-aa none + --lane-render-band`, `--hosc-*` annealed / step_basis, `--eikonal 0.01 /
--length 0.001`, `--muon-*` KEEP, `--verdict-batch 32`, `--cache-gt-skeleton`, `--pose-carrier-source
generated` (store-nothing, bit-exact), `--chroma`, `--persistence-* / --seed-islands / --island-dilate-px`,
`--margin-saliency-*` (margin=Fisher 0.978).

**FRESH-RUN-SPECIFIC (physics-required, cannot retrofit #205 by resume):** seed lane+movable dilated
**+2px** above the critical nucleus (nucleation fix) + `--tau-anneal-shape geometric` + raise
`--eikonal-weight 0.05` + Ch.6 easing (`--lane-band-start-epoch 350 --stage-transition-rewarmup-epochs 20
--shape cosine`) to deconflict the ep300 collision + `--mod-dim` decision (D2 conflict).

**A/B-OWED ARMS (isolated, byte-close-gated, operator-GO):** `--n-dir-freqs 2→4` (#1 ep300+ lever),
`--margin-saliency-reachability` (replaces inert texture), `--muon-warm-start-momentum --muon-lr-final-frac 0.1`
(#270, live), `--seg-subpix-boundary-weight` (vector-t), `--seg-chroma-boundary-weight`.

**EXCLUDE (MEASURED-NOGO):** `--render-aa supersample`, naive warp-only pose, ξ-lane-coding, raw-UNIWARD
margin proxy, l7-in-curriculum, fixed-β hosc, `--lane-prior-phi1` as band's only role, predictable-replicate
flicker, R-deconvolution.

**HELD:** witness d_pose (never validated on the witness; 3.4e-5 is abandoned ancestor) — byte-close #238.

**BEFORE THE RUN, fix the triality (D1/D2):** (a) flush the 07-03 lever laws to the JSONL registry so the
equation leg is not blind to the #1 lever; (b) make the mod-dim 32-vs-19 decision explicit.

**MEANS reminder:** pointer 0.19110 UNMOVED; witness implied-S ~0.67–0.75; nothing here is a score until a
byte-closed `upstream/evaluate.py` n600 row beats 0.19110.
