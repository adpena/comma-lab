# Capstone level-set witness — DEEP-MATH-OPTIMAL launch config (#205, all-levers from-scratch)

**Advisory / build-only. NO GPU fired. Pointer 0.19110 UNMOVED (moves ONLY via a byte-closed
n600 `upstream/evaluate.py` exact row < 0.19110 — this is a MEANS).** Tag `[macOS-MLX advisory /
design]`, `score_claim=false`, `promotable=false`.

Produced for the parent's final GO gate: recursive adversarial review + deep math + optimal-config.
Every flag verified to EXIST in the LEVELSET trainer argparse
(`experiments/train_levelset_witness_realized_through_R_mlx.py`, grepped `add_argument`) and every
lever verified WIRED (not one of the 3 fail-closed raisers). Values grounded in the proven-base
(autoconfig `_proven_base`, recalled verbatim from the 0.003698 arm), the calibration protocol
(`feedback_calibrate_parametrization_fisher_geometry_common_unit_math_first_tiered_20260701`), the
build-state (`project_1b_capstone_build_state_20260702`), the canonical equations, and the DAG.

## THE EXACT ARGV (single command; resumable + per-stage checkpointed)

```bash
TAC_MLX_CUSTOM_GROUPED_BACKWARD=1 \
.venv/bin/python experiments/train_levelset_witness_realized_through_R_mlx.py \
  --out-dir experiments/results/levelset_n600_witness_capstone_<UTC> \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --num-pairs 600 \
  --mlx-device gpu --seed 0 \
  --epochs 1000 --eval-every 25 \
  --verdict-pairs 0 --async-verdict \
  --curriculum \
  --tau-softplus-start-epoch 300 --tau-softplus-tau 0.3 \
  --l7-start-epoch 1000 \
  --muon-start-epoch 726 --muon-lr 0.002 --muon-momentum 0.95 --muon-ns-steps 5 \
  --stage-transition-rewarmup-epochs 8 --stage-transition-rewarmup-floor 0.1 \
  --stage-transition-rewarmup-shape linear --stage-transition-reset-moments \
  --w-seg 100 --w-pose 0 --score-domain-loss \
  --mod-dim 19 --hidden-dim 96 --n-hidden 4 \
  --activation hosc --hosc-beta 1.0 --hosc-beta-end 4.0 --hosc-beta-anneal linear \
  --hosc-omega 1.0 --siren-init \
  --softmax-temp-start 1.0 --softmax-temp-end 0.05 --tau-anneal-shape cosine \
  --self-orient --n-dir-freqs 2 --freq-across 32 --freq-along 4 --reorient-every 50 \
  --max-bank-freq 64 \
  --chroma --palette-anchor \
  --eikonal-weight 0.01 --length-weight 0.001 \
  --render-h 384 --render-w 512 \
  --render-aa none \
  --lane-render-band --lane-band-start-epoch 300 --lane-band-uncertainty-source witness \
  --lane-band-tau 0.85 --lane-band-eps 0.35 --lane-band-softness 1.0 \
  --lane-band-dash-forward-max-m 55.0 --lane-band-weight 1.0 \
  --persistence-loss-weight 1.0 --persistence-recall-weight 1.0 --cldice-iters 5 \
  --persistence-warmup-epochs 300 --persistence-classes auto \
  --amplify-weight 1.0 --amplify-form hinge --amplify-margin-target 1.0 \
  --amplify-persist inverse_thickness --island-dilate-px 1 \
  --structured-init --structured-init-include-lane \
  --lane-prior-phi1 --lane-prior-phi1-mode replace --lane-prior-phi1-dash-gate \
  --accum-pairs 8 --grad-clip 1.0 --ema-decay 0.997 \
  --lr 1e-3 --lr-end 1e-4 --weight-decay 1e-4 --adam-beta2 0.9999999 \
  --ckpt-every 25 --stage-checkpoints
```

**Durable launch (the parent fires — NOT this pass):** wrap in `spawn_durable_daemon.py`
(`--min-free-gb 10 --rss-cap-mb 90000`) via a `launch.sh` script so the child argv is `bash
launch.sh` (no word-split). `TAC_MLX_CUSTOM_GROUPED_BACKWARD=1` = the ONLY real perf env (the
~17× grouped-backward fast path; verify `custom_grouped_backward active=true` in the log). The
island/persistence Metal kernels (`TAC_MLX_CUSTOM_ISLAND_BIRTH`, `TAC_MLX_CUSTOM_PERSISTENCE_POOL`)
are NOT-YET-BUILT flag signatures — do NOT set them (mx.compile'd path is the authority).

## Per-flag justification table

| flag | value | grounding | confidence |
|---|---|---|---|
| `--num-pairs` | 600 | n600 pointer-mover (allergic-to-non-n600); real gt_n600 | HIGH |
| `--epochs` | 1000 | proven arm budget; autoconfig default; calibrated schedule fractions | HIGH |
| `--mod-dim` | **19** | Whitney floor for MEASURED intrinsic m~9 (`whitney_mod_dim(9)=clip(2·9+1,19,26)=19`); task-specified 17-19; #223 T0-DERIVE-free; rate-saving (rate is binding sub-0.15 term). Alt: 26 = autoconfig `overfit` headroom for composite m~13 (lane-orbit8+screw6) — NOT needed here (w-pose=0 → code carries only the d_seg manifold m~9). | HIGH (see caveat #1) |
| `--hidden-dim` | 96 | proven arm = review RD-optimum ~122KB; 19/96 is a rate WIN vs 26/120 | HIGH |
| `--n-hidden` | 4 | proven_base | HIGH |
| `--activation hosc` + `--hosc-beta 1.0` + `--hosc-beta-end 4.0` + `--hosc-beta-anneal linear` + `--hosc-omega 1.0` + `--siren-init` | β-anneal 1.0→4.0 | build-state ACTIVATION-DRIFT-RESOLVED (`17f8fc663`): from-scratch launch = hosc + siren-init + β-anneal 1.0→4.0 (belt+suspenders; each independently cures the tanh(β·sin) saturation the CLAUDE.md caveat forbids as fixed-β4). β sharpens toward step-native (no-Gibbs) as the SDF pins. eq `hosc_activation_saturation_trainability_v1` | HIGH (start/end); MED (linear shape = default, no measured shape optimum) |
| `--curriculum` + `--tau-softplus-start-epoch 300` + `--tau-softplus-tau 0.3` | CE→tau | proven fractions (tau@0.300·epochs); tau_softplus = "THE primary d_seg drop" | HIGH |
| `--l7-start-epoch 1000` | **l7 DEMOTED** | l7 = MEASURED DEFECT (eq `l7_linf_sharpening_defect`: L∞ sharpening inside a viscosity flow = d_seg-decoupling). Set = epochs to collapse l7 to ≤1 trailing epoch (curriculum guard requires `tau<l7<=epochs`; can't exceed). CE→tau→Muon-on-tau. | HIGH mechanism (see caveat #2) |
| `--muon-start-epoch 726` + `--muon-lr 0.002` + `--muon-momentum 0.95` + `--muon-ns-steps 5` | proven fraction 0.726 → 274-ep Muon finisher on tau loss | proven arm (`run_muon.log`); muon-lr 0.002 CRITICAL (default None→0.1·lr=1e-4 is 20× too low). Benign WARN fires (muon<l7 after demote) — allowed operator-freedom | HIGH |
| `--stage-transition-rewarmup-epochs 8` + `--floor 0.1` + `--shape linear` + `--reset-moments` | proven_base | re-treats AdamW→AdamW stage boundaries ("different stages need different treatment") | HIGH |
| `--w-seg 100` / `--w-pose 0` | proven_base | w-pose=0 = pose-blind-BY-DESIGN (pose-carrier fail-closed; d_seg is the binding controllable term) | HIGH |
| `--score-domain-loss` | on | proven_base default | HIGH |
| `--self-orient` + `--n-dir-freqs 2` + `--freq-across 32` + `--freq-along 4` + `--reorient-every 50` | directional basis (#1 lever, −48%) | proven_base; basis-BEFORE-capacity (calibration DAG order) | HIGH (self-orient); MED (n-dir-freqs 2 = proven-arm, not independently swept vs default 6) |
| `--max-bank-freq 64` | stem Nyquist | SEG_W/(4·stem_stride)=512/8=64; LEVER-2 anti-alias cap (calibration T0 boundary-Nyquist) | HIGH |
| `--render-h 384` / `--render-w 512` | R-survival floor | eq `oracle_r_dseg_floor_by_render_grid`: g384 oracle floor ~0.00086 < first-row target 0.00118 (representation adequate; gap = training). render-192 pre-caps → blocks sub-0.15 | HIGH |
| `--render-aa none` + `--lane-render-band` | analytic coverage-integrated AA (the contest-feasible OPTIMAL) | **Wave D AA CORRECTION (2026-07-02, supersedes Wave C FIX-2)** per `aa_feasibility_reconciliation_20260702.md`: brute `--render-aa supersample` is DISQUALIFIED on TWO independent grounds — (1) it HURTS the witness **−49%** (MEASURED; the 0.00086 floor is a REAL-FRAME *ceiling* SIGNAL-A, NOT the witness-realized SIGNAL-B, which supersampling an already-smooth softmax-of-SDF partition cannot improve), AND (2) the fp64 decode is **41.3 min > the 30-min budget** (4-core CPU) AND neither shipped inflate even *applies* supersample (both render base-grid point-sample) → a train/decode observation MISMATCH. `--render-aa none` + the analytic coverage-integrated `--lane-render-band` (O(1)/pixel, base-grid, `mx.compile`-friendly, decode IN budget ~10 min fp64/4.4 min fp32) IS the contest-feasible optimal AA and composes with `--self-orient` (no fine dir-feat cache). This removes the 14 GB fine-dir-feat cache, the ss² decode risk, the −49% witness harm, and the train/decode mismatch in ONE edit. The supersample code path stays BUILT + fail-closeable in the trainer (OUT of the launch config); `--render-aa ipe` (O(1), decode-safe) is the documented alt if a full-partition AA is ever wanted — NEVER supersample. | HIGH (contest-feasible; MEASURED to help; decode-in-budget) |
| `--lane-render-band` + band params | trained-in class-1 render authority | build-state: naive band HURTS +0.000622 → witness-uncertainty gate kills 98% FP → net-win NEEDS training-in (3× confirmed). uncertainty-source=witness, start@300 (=tau; witness partition formed), dash-forward-max-m 55.0 (SegNet-Nyquist #215). eq `analytic_lane_render_band_fp_reduction_v1` | MED (all defaults; trained-in per build-state) |
| `--persistence-loss-weight 1.0` + recall 1.0 + cldice-iters 5 + warmup 300 + classes auto | soft-clDice + persistence-recall on shared seg forward | wave-1 landed `122e59ba8` (111× more erasure-sensitive than CE). weight 1.0 = ENGAGE value (T2 knob per calibration — optimum only exists in-training); warmup 300 = coarse→fine (ramp over CE stage). eq `persistence_topology_cldice_betti_island_recall_v1` | LOW weight (T2-calibration start, no measured optimum — labeled) |
| `--amplify-weight 1.0` + form hinge + margin-target 1.0 + persist inverse_thickness | island-birth on shared LEVER-4 `_signed` | wave-1 landed `0f013e17a`. AMPLIFY_ONLY path is WIRED (seed/containment is fail-closed). weight 1.0 = ENGAGE (T2 knob). eq `island_finest_scale_protection_survival_v1` | LOW weight (T2-calibration start — labeled) |
| `--chroma` / `--palette-anchor` | on | proven_base; chroma = d_seg lever (operator); palette-anchor breaks the ~0.51 luma-ramp plateau | HIGH |
| `--eikonal-weight 0.01` / `--length-weight 0.001` | θ* lever stack | proven_base level-set regularizers (topology bias) | HIGH |
| `--structured-init` + `--structured-init-include-lane` + `--lane-prior-phi1` (+mode replace, dash-gate) | static-core phi init + openpilot deg-3 centerline (Road↔Lane separatrix, residual 1.9e-5) | proven_base active geometric priors; rule-118 FREE (ships 0 bytes). lane-prior REQUIRES structured-init (guard) | MED (proven_base; see caveat #3 fragility) |
| `--verdict-pairs 0` + `--async-verdict` | ALL 600, background | n600-allergy rule: override the proven-arm 96-subset → all 600 for decision-informing telemetry; async self-throttles (non-blocking, bit-identical training) | HIGH |
| `--accum-pairs 8` / `--grad-clip 1.0` / `--ema-decay 0.997` | proven_base | EMA 0.997 = Quantizr; EMA shadow ships | HIGH |
| `--lr 1e-3` / `--lr-end 1e-4` / `--weight-decay 1e-4` | trainer defaults (proven arm did not override) | proven_base-default | MED (proven-default, not swept) |
| `--ckpt-every 25` / `--stage-checkpoints` | resumability + per-stage | MANDATORY per-stage ckpt non-negotiable (CE/tau/Muon boundaries; EMA shadow; atomic) | HIGH |

## Flags left at proven_base default (no better-grounded value; honest)

- `--lr / --lr-end / --weight-decay` (1e-3 / 1e-4 / 1e-4): trainer defaults; the proven arm did not
  override them. NOT independently swept → proven-default, not a derived optimum.
- `--n-dir-freqs 2`, `--freq-across 32`, `--freq-along 4`: proven-arm values (autoconfig `_proven_base`);
  not independently re-swept for this from-scratch config.
- `--aa-ipe-footprint 1.0`, `--lane-band-*` (tau 0.85 / eps 0.35 / softness 1.0 / weight 1.0):
  trainer defaults; no measured per-lever optimum → T1 ($0 frozen-ckpt gate) calibratable.
- `--persistence-loss-weight 1.0`, `--amplify-weight 1.0`: **ENGAGE values, NOT measured optima**
  (T2 knobs per the calibration protocol — the optimum only exists inside the descent; warm-started
  sweep off per-stage ckpts calibrates them). Set non-zero to honor operator "levers TRAINED-IN".

## Fail-closed levers EXCLUDED (verified NotImplementedError/ValueError raisers)

1. `--pose-carrier` → `NotImplementedError` (line 1443): frame0 real-luma warp render_fn + residual
   co-grad wire-in is the #224 follow-up. ⇒ `--w-pose 0` (pose-blind-by-design; d_seg-first).
2. `--seed-islands` → `NotImplementedError` (line 1631): protected seed-residual PARAM + grad-shield
   restructure not wired. ⇒ use `--amplify-weight` (AMPLIFY_ONLY, WIRED, rides `_signed`) instead.
3. ~~`--render-aa supersample`~~ **NOT in the launch config — Wave D AA CORRECTION (2026-07-02, supersedes
   Wave C FIX-2)** per `aa_feasibility_reconciliation_20260702.md`: supersample is DISQUALIFIED (HURTS
   the witness −49% [SIGNAL-A ceiling ≠ SIGNAL-B realized] AND fp64 decode 41min > 30min budget AND
   neither shipped inflate applies ss → train/decode mismatch). The launch AA is `--render-aa none` +
   the analytic coverage-integrated `--lane-render-band`. The supersample code path (`--aa-supersample`
   / `--aa-self-orient-fine-mode`) stays BUILT + fail-closeable in the trainer but is NOT emitted by
   the all-levers config; re-open ONLY if a witness-realized (SIGNAL-B byte-closed through-R) n600
   measurement shows it HELPS AND decode is moved to fp32-torch or T4-GPU inflate (<18 min).

Also OFF (parent's authorized list excludes them; kept for a later warm-start re-treatment):
- **LEVER-4 margin-saliency** (`--margin-saliency-weight 0`): calibration's #2 lever (KKT waterfill on
  the annulus). Strong DEFERRED lever; amplify rides the SAME `_signed` mechanism WITHOUT it (verified
  `_seg_levers_on`, line 1654 — amplify alone triggers the shared forward; NO silent no-op).
- LEVER-3 lane-edge, LEVER-B thin-lane, DM1 film-stiefel/code-spectral, code-nuclear, margin-field head
  (`--margin-field-head-weight 0`, head=softmax): margin-field is WIRED but net-positivity is UNCONFIRMED
  (in-flight #218 Laguerre/ETF sweep) → OFF per parent's "if net-positive, else off".

## Resolved gaps (was "CANNOT SET" — now actuated)

- **#222 Adam β₂ — RESOLVED (Wave C FIX-1 + Wave D)**: the trainer NOW exposes `--adam-beta2` (added
  Wave C; `experiments/train_levelset_witness_realized_through_R_mlx.py:3273` + sister base trainer).
  DERIVED optimum (calibration T0, arXiv 2603.02092): with β₁=0.9, n=75 (=P/accum_pairs=600/8),
  `1−β₂* ≲ (1−β₁⁵)/n^3.5 = (1−0.59049)/75^3.5 = 0.40951/3.653e6 = 1.12e-7` ⇒ **β₂* ≈ 0.9999999**
  (≈ 1−1.1e-7; the default 0.999 → 1−β₂=1e-3 is ~4 orders ABOVE the floor = under-smoothed for n=75).
  The all-levers argv above emits `--adam-beta2 0.9999999`; the trainer gates bias-correction ON off
  the 0.999 default (`_adam_bias_correction_for`; without it the high β₂ = ~100× step-1 LR blowup) and
  Wave D threads the SAME β₂ + gate into the Muon-finisher AdamW rest-group (ep726-1000) so the whole
  run is CONSISTENT. Default 0.999 stays == the MLX default (byte-identical-off).

## Caveats / watch-items (for the recursive + senior review)

1. **mod-dim 19 vs 26**: 19 = Whitney floor for the MEASURED d_seg-manifold m~9 (this run is d_seg-only,
   w-pose=0, so no screw in the code manifold). 26 = autoconfig `overfit` headroom for the composite m~13
   (adds the se(3) screw for when pose is later composed). Rate has slack (~0.055 crushed), so 19 is the
   rate-optimal embedding floor; if a training capacity ceiling is hit (intrinsic-dim TwoNN upper ~11 →
   Whitney ~23), bump toward 23-26. **Task-specified band = 17-19 → 19 chosen.**
2. **l7 demotion is flag-level (≤1 trailing epoch)**: the curriculum guard forbids `l7_start > epochs`,
   so l7 fires for exactly the final epoch (ep=1000, under Muon, EMA 0.997 ⇒ negligible). TRUE l7 removal
   is the #224 wire-in code change to `_seg_form_for_epoch` (pending). Negligible for the first row.
3. **structured-init is hosc/siren-FRAGILE** (trainer WARNs loudly if the pretrain stalls; "no epoch-0
   realized win, texture-gated"). Included per proven_base + it enables the strong lane-prior-phi1 DOMAIN
   prior. If it stalls at launch → fallback `--no-structured-init --no-lane-prior-phi1`.
4. **Confound**: this stacks β-anneal + l7-demote + lane-band + ipe + persistence + amplify simultaneously
   (operator chose 1B all-levers-trained-in over attribution-clean-first). Per-lever attribution needs the
   per-stage-ckpt A/B the `--stage-checkpoints` enable. Documented tradeoff, not a hidden one.

## Compose / triality note (means != ends)

DAG: #205 from-scratch launch → #202 byte-close → first exact n600 row. DSL: this argv IS the compiled
`tac.witness_dsl` program (flag-validated against the real argparse). Equations: 9 anchored (persistence,
island, warp, analytic-lane, l7-defect, R-all-pass, hosc-stability, oracle-R-floor, power-diagram).
The pointer (0.19110) moves ONLY through the byte-closed exact row — this config is a MEANS.
