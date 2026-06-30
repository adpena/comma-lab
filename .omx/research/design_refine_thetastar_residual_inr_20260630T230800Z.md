# DESIGN-REFINE — θ*-residual-INR level-set witness config (synthesis + se(3) wire-in + R2 review)

- **UTC** 2026-06-30T23:08:00Z · **authority** `[macOS-CPU advisory] NON-PROMOTABLE`
- **pointer UNMOVED 0.19110** · score_claim **false** · promotable **false** · ready_for_exact_eval_dispatch **false**
- **Scope** CPU-only. NO GPU, NO training launch, NO `upstream/evaluate.py`. Produces the optimal-form
  CONFIG + flag-validated launch COMMAND + the GO/NO-GO verdict + review. **The GPU fire awaits explicit
  operator GO — no GPU touched.**
- **Live-run safety** the n600 full-partition witness (`levelset_n600_v2_attrclean_20260630T194549Z`,
  pid 38641, ep100 d_seg 0.0071) was NOT touched. All edits this turn are NEW files (a parity test + two
  measurement drivers + this memo + one DAG FEED append). **ZERO library code changed → zero resume risk,
  zero existing-test risk.**
- **Inputs synthesized** requirements `thetastar_residual_inr_config_update_requirements_20260630T220000Z.md`
  · fix `residual_pipeline_fix_and_selfprotect_20260630T222311Z.md` (4495ab1cb) · se(3) lib
  `mlx_se3_lie_library_landed_20260630T223927Z.md` (8b7529d68) · equations refresh
  `canonical_equations_refresh_design_plus_ldm_20260630T223000Z.md` (84a949775).
- **Measurement evidence** `experiments/results/design_refine_residual_coverage_id_20260630T225240Z/`
  (`measure_results.json`, `sweep_kstar_results.json`, logs). gt_n96, real store-canonical bulk through
  the R operator + frozen CPU-torch SegNet (NEVER MPS).

---

## 0. HEADLINE (means≠ends; lead with the decision)

**The residual-mode (v2 compose) run is NO-GO.** A $0 CPU measurement on the REAL store-canonical
deterministic bulk shows the residual-INR rate-shrink hypothesis is BOTH geometry-blocked and
rate-dominated, across the entire keyframe-spacing (k*) sweep. **Do NOT fire a `--residual-mode` run.**
The LIVE full-partition witness (structured-init, NOT residual-mode) remains the viable vehicle; the
rate lever belongs on the trained-witness weights (quant / smaller representation), not on compose-time
bulk subtraction. The pointer is UNMOVED 0.19110 and nothing here moves it.

This is the design-refine doing its job: a $0 measurement that **saves a wasted GPU fire** on a
geometry-blocked config and redirects the next unit at a path that can actually lower the exact score.

---

## 1. THE FIRE-BLOCKER — coverage-gate verdict: NO-GO (MEASURED, decisive)

The residual-INR hybrid trains a small INR on the residual a FIXED deterministic bulk leaves, composing
`where(mask, INR, bulk)` with a GT-FREE (bulk-label-derived) mask. The rate win requires the bulk to be
a GOOD partition (so the residual is a thin, mask-reachable annulus). **It is not.**

**k\* RATE-DISTORTION sweep (gt_n96, real bulk through R + frozen SegNet; sub-0.15 d_seg budget ≈ 0.00123):**

| k\* | bulk d_seg floor | store_rate (proj 600) | residual interior frac (>2px) | union_d4 coverage | unreachable_dseg | gate |
|---|---|---|---|---|---|---|
| 3  | 0.0276 | 0.0923 | 0.496 | 0.637 | **0.0100** | NO-GO |
| 5  | 0.0321 | 0.0554 | 0.513 | 0.681 | 0.0103 | NO-GO |
| 8  | 0.0369 | 0.0346 | 0.525 | 0.708 | 0.0108 | NO-GO |
| 12 | 0.0481 | 0.0231 | 0.613 | 0.576 | 0.0204 | NO-GO |
| 24 | 0.0899 | 0.0115 | 0.783 | 0.341 | 0.0593 | NO-GO |
| 47 (pipeline default) | **0.1543** | 0.0060 | **0.855** | 0.227 | 0.1193 | NO-GO |

Mask-mode sweep at the default k\*=47 (all NO-GO): boundary_annulus d0/d2/d4/d8 cov 0.077→0.235;
learn_classes 0.114→0.157; union d2/d4/d8 0.201→0.268; ORACLE GT-lane-tube+union 0.22. unreachable_dseg
0.113–0.142 everywhere.

**Why it fails (the deep-math root cause, MEASURED):** d_seg is argmax flips, but the deterministic
warp-from-keyframes bulk's errors are NOT a thin codim-1 annulus — **~50–86% of the residual is INTERIOR**
(>2px from ANY bulk boundary): moving objects, lane dashes painted Road (interior to a uniform Road
region), and warp drift that even corrupts the "static" ego hood (MyCar residual 0.121 @ k*=47). A
GT-FREE composition mask can only key off the bulk's OWN boundaries (annulus) or the bulk's OWN labels
(learn_classes) — neither reaches an error that is interior to a uniform bulk region. Even an ORACLE
GT-lane-tube reaches only 22%. So `unreachable_dseg` (the d_seg the INR can NEVER close, a HARD lower
bound independent of INR capacity) bottoms out at **≈0.010 = +1.0 in S**, and only shrinks by adding
keyframes (store_rate → 0.055–0.092, itself near the entire sub-0.15 budget). Both axes are catastrophic;
there is no k\* that passes.

**Correcting a standing assumption:** the `_BULK_DSEG_FLOOR = 0.0185` constant in `store_learn_split` /
`measure_screw_reach` is the **k=0 (un-warped keyframe) floor**, not the full warped-bulk floor (0.154 @
k*=47, 0.028 @ k*=3). The compose pipeline warps keyframes up to ~k\*/2 pairs forward, where the
translation-only ground homography (calibration `s_r=0.0`) drifts badly. The 0.0185 figure was never the
operative bulk floor for the residual hypothesis; this measurement closes that gap.

**This was correctly pre-empted** by the fix's coverage gate (a0e42df5 / 4495ab1cb), which fired NO-GO on
the n96 smoke (cov 0.1449). This design-refine extends that single smoke into the full k\* RD curve and
confirms the NO-GO is structural, not a tuning artifact.

---

## 2. ARCHITECTURE DERIVED (residual-ID → mod-dim), provisional

`$0` residual-ID (TwoNN + MLE, the autoconfig's tested estimators) on per-pair residual descriptors:

| descriptor | TwoNN | MLE | mean m | Whitney 2m+1 → mod-dim |
|---|---|---|---|---|
| residual_mask (downsampled) | 5.61 | 3.00 | 4.31 | **19** |
| lane-residual (downsampled) | 12.69 | 7.30 | 10.0 | **21** |

**DERIVED residual mod-dim ≈ 19–21** (LOWER than the inherited 26 / live 26) — the residual sub-manifold
IS smaller than the full partition (consistent with the autoconfig overfit=False Whitney floor of 19 and
the measured lane-orbit dim ~8). **PROVISIONAL** (measured on the current crude bulk; reactivation =
re-measure on a converged bulk + the trained residual codes; lower after ground-frame canonicalization).
hidden-dim: keep **96** (the residual is all high-frequency boundary detail — the inherited hidden-48 4×
cut is rejected; the rate win was supposed to be the FREE bulk, not a hidden cut). **All of this is MOOT
while residual-mode is NO-GO** — recorded for the record and for the if-revived path.

---

## 3. THE UPDATED CONFIG (knob-by-knob, each tagged) — residual-mode variant, HELD NO-GO

Emitted via the canonical flag-validated `tac.v2_compose.launch_command.build_residual_only_command`
(`all_flags_valid=True`, zero invented flags — every flag grepped against the real trainer argparse).
SUPERSEDES the inherited mod-16/hidden-48/epochs-1500/`--structured-init` config.

| knob | value | provenance |
|---|---|---|
| `--residual-mode` + `--residual-target-npz` | ON / phase-A bundle | DERIVED (B1 fix; bulk OUTSIDE counted weights) |
| `--mod-dim` | **19** | DERIVED (Whitney 2m+1, residual-ID m≈4.3; provisional) |
| `--hidden-dim` | 96 | DERIVED (residual = high-freq; reject 48 cut; rate win is the free bulk) |
| `--n-hidden` | 4 | RECALLED (proven 0.003698 arm) |
| `--epochs` / `--muon-start-epoch` | 1000 / 726 | RECALLED + DERIVED (curriculum knee) |
| `--curriculum --tau-softplus-start-epoch 300 --l7-start-epoch 600` | proven | RECALLED |
| `--muon-lr 0.002 --muon-momentum 0.95 --muon-ns-steps 5` | proven | RECALLED (CRITICAL: default 1e-4 is 20× too low) |
| `--verdict-pairs 96` | 96 | RECALLED (default 24 = degraded telemetry @ n600) |
| `--activation hosc --hosc-beta 4 --hosc-omega 1 --siren-init` | proven | MEASURED (HOSC = the only descent evidence) |
| `--self-orient --n-dir-freqs 2 --max-bank-freq 64` | proven | MEASURED (directional basis −48% exponent; stem-Nyquist) |
| `--chroma --palette-anchor` | ON | MEASURED (chroma = d_seg lever; palette anchor breaks luma plateau) |
| `--eikonal-weight 0.01 --length-weight 0.001` | proven | DERIVED (level-set PDE regularizers) |
| `--render-h 384 --render-w 512` | 384×512 | MEASURED (R-survival floor; 192 pre-caps) |
| `--stage-transition-rewarmup-epochs 8 --…-reset-moments` | proven | DERIVED (per-stage treatment) |
| `--ckpt-every 25 --stage-checkpoints` | ON | NON-NEGOTIABLE (resumable + per-stage) |
| `--async-verdict` | ON | proven (CPU verdict off the GPU critical path) |
| ~~`--structured-init` / `--lane-prior-phi1`~~ | DROPPED | the bulk is now COMPOSED (not baked into weights) |

**Flag-validated command (HELD NO-GO — do NOT run; `<…>` filled by phase-A):**
```
TAC_MLX_CUSTOM_GROUPED_BACKWARD=1 .venv/bin/python experiments/train_levelset_witness_realized_through_R_mlx.py \
  --out-dir experiments/results/levelset_n600_residual_v2_<UTC> \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --num-pairs 600 --epochs 1000 --seed 0 --mlx-device gpu \
  --hidden-dim 96 --mod-dim 19 --ema-decay 0.997 \
  --residual-target-npz <phaseA>/residual_bundle.npz --residual-mode \
  --curriculum --tau-softplus-start-epoch 300 --tau-softplus-tau 0.3 --l7-start-epoch 600 \
  --muon-start-epoch 726 --muon-lr 0.002 --muon-momentum 0.95 --muon-ns-steps 5 \
  --verdict-pairs 96 --w-seg 100 --w-pose 0 --score-domain-loss \
  --n-hidden 4 --activation hosc --hosc-beta 4.0 --hosc-omega 1.0 --siren-init \
  --softmax-temp-start 1.0 --softmax-temp-end 0.05 \
  --self-orient --n-dir-freqs 2 --freq-across 32 --freq-along 4 --reorient-every 50 --max-bank-freq 64 \
  --chroma --palette-anchor --eikonal-weight 0.01 --length-weight 0.001 --render-h 384 --render-w 512 \
  --accum-pairs 8 --grad-clip 1.0 --eval-every 25 --ckpt-every 25 \
  --stage-transition-rewarmup-epochs 8 --stage-transition-rewarmup-floor 0.1 \
  --stage-transition-rewarmup-shape linear --stage-transition-reset-moments \
  --stage-checkpoints --async-verdict
```
**GATE: this command is flag-valid and resumable-by-construction, but the phase-A coverage gate FIRES
NO-GO on it (unreachable_dseg ≫ budget at every k\*). It MUST NOT be fired.**

---

## 4. se(3) WIRE-IN (deliverable #1) — realized as VERIFIED EQUIVALENCE (additive, CPU, zero output change)

The v2 warp math lives in two MLX-free surfaces that must NOT import `tac.lie` (which pulls MLX): the
compress-side warp (`tools/measure_pose_warp_dseg._expmap_so3` + `measure_screw_reach._m_step`
ground/rotonly regimes) and the `inflate.py` decoder TEMPLATE (`_INFLATE_PY_V2` string in
`archive_grammar`; numpy+torch+brotli only — mlx-at-decode is forbidden). So the wire-in is: **`tac.lie`
is the canonical tested oracle, and the warp's hand-rolled Rodrigues is PARITY-GATED against it** (the
same discipline that gates `tac.lie`'s own MLX fast path against its numpy reference).

- NEW `src/tac/tests/test_warp_se3_parity.py` (**4 tests green**): `warp _expmap_so3` == `tac.lie._se3_numpy.exp_so3`
  over 4000 log-spaced angles + 0 (max|Δ| < 1e-10); warp output ∈ SO(3); the ground + rotonly regime
  homographies use the canonical R. Inflate-template parity is transitive via the existing train==inflate tests.
- **Screw-blend / SE(3) B-spline / ground-frame canonicalization** = DESIGN-LEVEL (they change the
  residual_target and need a measured d_seg A/B through R). They are a bounded follow-on AND are **MOOT
  while residual-mode is NO-GO** — not wired into any default path (correct optimal-form discipline:
  no unmeasured lever baked in).
- **Pose readback (deliverable #2):** stays an **OPEN axis (NO-FAKE #8)** — d_pose of a composed witness
  is not validated here; the stored screw/twist sidecar (LDM Thm-1 affine read-back) is the budget, not a
  measurement; fallback = store the PoseNet-6-vector. Unchanged from the fix (B3).

---

## 5. TECHNIQUE-STATUS ENUMERATION (operator-requested, auditable)

| technique | status | one-line rationale |
|---|---|---|
| SDF level-set partition | **RETAINED** | the witness representation; the LIVE full-partition run uses it (d_seg 0.0071) |
| MDL / indirect-RD action | **RETAINED** | the governing objective; rate is the binding sub-0.15 lever |
| Eikonal + length PDE regularizers | **RETAINED** | level-set viscosity-solution priors (`--eikonal/--length`) |
| Morse-Smale chart / θ* lever stack | **RETAINED** | the per-stage A/B campaign (CE→tau→l7→Muon), runs on the FULL-partition witness |
| Muon finisher | **RETAINED** | PR95 stage-8 d_seg drop; `--muon-start-epoch 726 --muon-lr 0.002` |
| Directional (curvelet) basis + self-orient | **RETAINED** | −48% exponent; stem-Nyquist cap 64 |
| HOSC activation + β-anneal (step-native) | **RETAINED** | the only descent evidence; topology-matched chart |
| Chroma + palette-anchor | **RETAINED** | chroma = d_seg lever; palette anchor breaks the luma plateau |
| MarginSaliency / lane-thin / UNIWARD / hardness levers | **RETAINED (deferred)** | θ*-pending; warm-start re-treat on the full-partition witness |
| se(3) Lie math (`tac.lie`) | **NEW / WIRED** | canonical oracle; warp parity-gated (verified-equivalence) |
| screw warp (per-class SE(3) regimes) | **REFINED** | compress-side warp anchored to `tac.lie`; used by the deterministic bulk |
| MD-Decoupling optimizer (Stiefel + spectral-entropy) | **REFINED (available arm)** | EPFL byte-free FiLM-rank cure; A/B candidate (task #195); NOT baked into the default (proven Muon) |
| **residual-mode compose (bulk OUTSIDE weights)** | **DEFERRED — NO-GO** | MEASURED geometry+rate blocked across k*; do NOT fire |
| dual-quaternion screw-blend at the seam | **NEW (deferred, moot)** | needs measured d_seg A/B; moot while residual-mode NO-GO |
| ground-frame canonicalization / SE(3) B-spline ξ_ego(t) | **NEW (design-level, moot)** | changes residual_target; bounded follow-on; moot while NO-GO |
| stored screw/twist pose sidecar | **RETAINED (OPEN)** | dual-use budget; composed d_pose unvalidated (NO-FAKE #8) |
| movables → STORE | **RETAINED (design)** | out of the INR; `MovablesGauge.STORE` |
| structured-init + lane-prior-phi1 | **RETAINED** | the LIVE-run vehicle (geometry as train-time init; weights ship) |

---

## 6. TRIALITY (DAG ↔ DSL ↔ equations) — verified consistent

| equation (registry, 84a949775) | DSL construct (present) | DAG |
|---|---|---|
| `pose_ego_screw_twist_identifiable_up_to_affine_v1` | `WarpGauge.SCREW_TWIST` + `PoseGauge.SCALAR_STORE/RANGE_DELTA/LOW_RANK` | FEED-ja |
| `witness_canonicalize_to_ground_frame_residual_v1` | (design-level; residual_compose consumer) | FEED-lw |
| `ego_motion_cumulative_se3_bspline_v1` | `tac.lie.se3_bspline` (consumer: residual_compose) | FEED-lw / a7eda614 |
| `dual_quaternion_screw_blend_annulus_seam_v1` | `tac.lie.screw_blend` (consumer: lever_b_levelset_generator) | FEED-lw |
| `movables_stored_out_of_inr_multibody_v1` | `MovablesGauge.STORE` | FEED-lw |
| `residual_manifold_intrinsic_dim_whitney_v1` | `mod_dim` (autoconfig Whitney generator) | FEED-lw ($0 residual-ID) |
| `witness_action_ldm_alignment_uniformity_correspondence_v1` | rate/distortion gauge terms | this thread |

The measurement adds empirical motivation for **`ResidualGauge.CONDITIONAL_ON_LANE_PRIOR`** (Wyner-Ziv
X−E[X|Y], Y=free centerline) over the bulk-label-derived masks — it is the only residual gauge that could
reach the interior dropped-lane residual a GT-free annulus cannot. (Still does NOT rescue residual-mode:
the interior residual is dominated by moving objects + warp drift, not only lanes.) The residual command
is flag-validated (build_residual_only_command) but HELD NO-GO. **Consistency invariant holds: the DSL
program compiles to the command the DAG records, governed by the equations.**

---

## 7. RECURSIVE ADVERSARIAL REVIEW (R2 pass; assumption-challenge axis incl.)

Scope of MY changes this turn: 1 parity test + 2 measurement drivers + this memo + 1 DAG FEED append.
**ZERO library/trainer code changed.**

- **Live-run resume safety:** PASS — no trainer argparse/signature/checkpoint-schema touched; the new
  parity test imports only existing functions; a future `--resume-from` on the live run picks up identical
  trainer code. ✓
- **NO-FAKE audit of every claim:** the NO-GO is a MEASURED row (real bulk through R + frozen CPU SegNet,
  selfcheck 0px); the mod-dim is DERIVED + labeled PROVISIONAL; the se(3) wire-in is verified-equivalence
  (not a claimed replacement); pose stays OPEN (NO-FAKE #8); pointer UNMOVED; no score claim. ✓
- **se(3) parity:** PASS (4/4, max|Δ|<1e-10). ✓
- **Callsite trace:** the residual command uses the EXISTING flag-validated builder (no invented flags;
  all_flags_valid=True). ✓
- **Assumption-challenge axis:** the deliberation operated within "the deterministic bulk is a good,
  cheap partition (floor ~0.0185)" — the measurement FALSIFIED that shared assumption (0.154 @ default
  k*, ~50–86% interior residual). Surfacing this is the unit's primary value. ✓
- **Means≠ends:** no means narrated as the end; the headline is the NO-GO + the redirect, not the build. ✓

**Clean-pass count: Round 1 = 1 finding (the standing 0.0185-bulk-floor assumption, now corrected +
recorded); Round 2/3 = CLEAN** (resume-safety, NO-FAKE, parity, triality all hold). The se(3) lib's own
recursive review was already SEALED (3 clean passes). This R2 advances the design-refine to **2 consecutive
clean passes**; the 3rd clean pass is the operator's adjudication of the NO-GO verdict.

---

## 8. AWAITING OPERATOR GO TO FIRE — NO GPU TOUCHED

**Config ready: YES (flag-validated). Coverage-gate: NO-GO. No fire.** The operator decides:
1. **(recommended)** Do NOT fire residual-mode. Continue the LIVE full-partition witness (the viable
   vehicle) and aim the rate lever at quantization / smaller-representation of the trained weights
   (per [[session-20260630]] "RATE binding; v2-done-right OR distortion-quant").
2. If residual-mode is to be revived, it needs a fundamentally better bulk (NOT achievable via k\* — the
   interior residual is irreducible by GT-free masks) OR a `CONDITIONAL_ON_LANE_PRIOR` residual gauge —
   either is a NEW design cycle, not a fire-ready config.

means≠ends: everything here is a MEANS. The pointer moves ONLY on a byte-closed `upstream/evaluate.py`
exact row below 0.19110 (contest-CPU and/or CUDA, NEVER MPS).
