# witness_autoconfig LANDED — the clip → witness_config actuator (dogfooded on n600)

**UTC** 20260630T184951Z · **tag** `[macOS-MLX advisory · design/tooling · NON-PROMOTABLE]` · **pointer 0.19110 UNMOVED.**
$0 CPU-only. NO GPU touched, NO training launched, NO score claimed. Files committed via serializer.

## means ≠ ends (lead with it)
This is **config-derivation tooling**, a MEANS. It produces + flag-validates a launch command. The only END is a
**byte-closed n600 exact row < 0.19110** from `upstream/evaluate.py` (contest-CPU and/or CUDA, NEVER MPS). The
pointer is **UNMOVED at 0.19110** and stays UNMOVED until that exact row lands. The n600 **LAUNCH awaits operator GO**
(one GPU; containment/protect/preservation — no autonomous heavy launch).

## What landed
- `src/tac/witness_autoconfig.py` — the reusable pure-numpy/CPU actuator (`derive_config(gt_cache, num_pairs, overfit) → WitnessConfig`).
- `src/tac/tests/test_witness_autoconfig.py` — **23 tests, all green** (`.venv/bin/python -m pytest … -q` → 23 passed).
- `tools/witness_autoconfig.py` — thin CLI; `--emit-command` emits the GO-ready n600 command AND flag-validates it (55/55 PASS).

This dogfoods the guiding principle (`feedback_overfit_contest_but_build_generalizable_auto_value_generators`): overfit
the contest objective HARD, but the machinery that produces the overfit is a clip-agnostic AUTOMATED value-GENERATOR
(not ad-hoc constants), so the same investment is both the best n600 submission AND the durable 10-yr comma.ai asset.

## The 7 value-generators (what each MEASURES → derives)
| Generator | Knob | Derivation | Fallback (NO-FAKE) |
|---|---|---|---|
| `intrinsic_dim` + `whitney_mod_dim` → `mod_dim_generator` | `--mod-dim` | TwoNN+MLE nonlinear intrinsic dim `m` → Whitney embed `clamp(2m+1, 19, 26)`; overfit ships **26** (headroom ceiling; composite m~13→27→26), aggressive ships the floor (**19** for m~9) | no live code matrix → recalled measured `m=9` flagged `source="fallback_constant"` |
| `hidden_dim_generator` | `--hidden-dim` | RD-min of a byte-close sweep `{hidden→bytes}` (RATE is the binding sub-0.15 lever) | no sweep → proven/review RD-optimum **96** (NOT 120: 26/96 ≈ −0.004 S vs 26/120 +0.010 S) |
| `curriculum_schedule` | tau/l7/muon start | proven fractions 0.300/0.600/0.726 of the epoch budget; exact **300/600/726** at epochs=1000, proportional otherwise | proven schedule constants |
| `muon_lr_generator` | `--muon-lr` | recalled proven finisher **0.002** (run_muon.log + muon_finisher_switch JSON); default None→1e-4 is 20× too low | — (recalled_proven) |
| `verdict_pairs_generator` | `--verdict-pairs` | proven **96** (default 24 = degraded telemetry at n600) | — |
| `lever_priors` | surgical levers | per-stage attribution (Road=STUCK/causal-warp, Lane=PRIMED/trained-residual) → **all surgical levers + DM1 OFF** for the attribution-clean FIRST launch; queued ON as a shape-compatible warm-start re-treatment | measured default attribution |
| `warp_priors` | (v2 design) | screw-fit per-class: Road=ground_homography, sky=rotation_only, hood=identity, Lane=learned residual — **design-level, NOT yet a trainer flag** (v2 vehicle), so NOT emitted into the current INR command | — |

Each value carries a `ProvenancedValue(value, source, provenance, portability, tag)`. The CLI prints the full
generator→value→provenance trail so a reviewer audits every number.

## The portable / clip-specific split (the comma.ai corpus-generalization path)
The production-path core: for a NEW clip (or the whole corpus), re-run the **INSTANCE** generators; ship the rest as defaults.
- **SCORER_FIXED** (invariant across every contest video — frozen SegNet/PoseNet, R, 37.5M normalizer): `muon_lr, verdict_pairs, w_seg, w_pose, render_h, render_w`.
- **DOMAIN_FUNDAMENTAL** (any dashcam; FORM transfers, coefficients re-fit): `directional_basis, lane_prior_phi1, warp_priors, activation, chroma, palette_anchor`.
- **INSTANCE_CONDITIONED** (this clip's numbers; RE-MEASURE per clip): `mod_dim, hidden_dim, tau/l7/muon start, epochs, structured_init`.

CORPUS path (10-yr asset): run `derive_config` across comma2k19/comma10k → response-surface (cheap clip stats → optimal
config) → meta-model predicting config from cheap stats. The held-out n400 probe already validated the split is portable.

## The dogfood output (the flag-validated GO-ready command)
`.venv/bin/python tools/witness_autoconfig.py --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz --num-pairs 600 --emit-command`
→ derives mod-26 / hidden-96 / muon-lr-0.002 / verdict-96 / levers+DM1 OFF and emits **FLAG VALIDATION: 55/55 PASS — no invented flag** (exit 0).
The 4 binding recursive-review revisions are all applied; the command is attribution-clean (all 5 surgical levers + DM1 omitted
= their 0.0/False argparse defaults = OFF, verified — matches the proven 0.003698 arm exactly), from-scratch (no `--resume-from`),
epochs 1000, `--gt-cache gt_n600.npz --num-pairs 600`. Cross-checked verbatim against the proven n200 muon command (run_muon.log):

```bash
TAC_MLX_CUSTOM_GROUPED_BACKWARD=1 \
  .venv/bin/python experiments/train_levelset_witness_realized_through_R_mlx.py \
  --out-dir experiments/results/levelset_n600_v2_attrclean_<utc> \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz --num-pairs 600 \
  --mlx-device gpu --seed 0 --async-verdict \
  --epochs 1000 --eval-every 25 --verdict-pairs 96 \
  --curriculum --tau-softplus-start-epoch 300 --tau-softplus-tau 0.3 --l7-start-epoch 600 \
  --muon-start-epoch 726 --muon-lr 0.002 --muon-momentum 0.95 --muon-ns-steps 5 \
  --stage-transition-rewarmup-epochs 8 --stage-transition-rewarmup-floor 0.1 \
  --stage-transition-rewarmup-shape linear --stage-transition-reset-moments \
  --w-seg 100 --w-pose 0 --score-domain-loss \
  --mod-dim 26 --hidden-dim 96 --n-hidden 4 \
  --activation hosc --hosc-beta 4.0 --hosc-omega 1.0 --siren-init \
  --softmax-temp-start 1.0 --softmax-temp-end 0.05 \
  --self-orient --n-dir-freqs 2 --freq-across 32 --freq-along 4 --reorient-every 50 \
  --max-bank-freq 64 --chroma --palette-anchor \
  --eikonal-weight 0.01 --length-weight 0.001 --render-h 384 --render-w 512 \
  --accum-pairs 8 --grad-clip 1.0 --ema-decay 0.997 \
  --structured-init --structured-init-include-lane \
  --lane-prior-phi1 --lane-prior-phi1-mode replace --lane-prior-phi1-dash-gate \
  --ckpt-every 25 --stage-checkpoints
```
`gt_n600.npz` exists on disk (5.08 GB) — no regen note needed. At launch, replace `<utc>` and honor the pre-launch
discipline checklist (perf-env `active=True` log line, whole-run archiver, memory guard `--min-free-gb 10`, one-GPU-await-go).

## NO-FAKE honesty
- A generator with its heavy input absent returns the recalled measured CONSTANT and flags `source="fallback_constant"` —
  it NEVER fabricates a measurement. (mod-dim's m=9 and hidden-96 are currently fallbacks; the CLI prints `[fallback!]`.)
- `intrinsic_dim` ACTUALLY computes TwoNN+MLE when given a code matrix (test `…measures_known_low_dim_manifold` recovers
  m≈2 on a 2-D plane in 12-D) — the test would fail if the body were a constant.
- Flag-validation is structural (parses the real argparse) — the suite would fail if any emitted flag were invented.
- `mod-26` provenance is honestly "held_out_corrected headroom"; the underlying intrinsic m=9 is flagged fallback within
  the same provenance string (the "~26 is a LINEAR PR overcount; nonlinear ID ~9" recursive-review finding is preserved).
- warp_priors is explicitly `status="design_v2_not_a_trainer_flag"` and is NOT emitted into the command.

## Wire-in / next
- This is `tac` lib code (reusable) + thin CLI per the repo discipline. Consumers: the n600 launch (dogfood), and the
  corpus meta-model (phase-4). The witness DSL (#189) is the sister declarative front-end; this actuator is the generator side.
- Follow-on (NOT blockers): byte-close sweep → `hidden_dim_generator(byte_close_result=…)` measured (vs fallback);
  live code-matrix → `derive_config(code_matrix=…)` measured intrinsic dim (vs fallback); corpus response-surface meta-model.

Cross-refs: `n600_v2_launch_ready_design_20260630T180947Z.md`, `n600_v2_recursive_review_20260630T182612Z.md`,
`post_muon_application_plan_optimal_form_20260630T1710Z.md`, `feedback_overfit_contest_but_build_generalizable_auto_value_generators`,
`feedback_per_stage_per_pixel_annulus_attribution_surgical_repair_toolbox`. Pointer 0.19110 UNMOVED.
