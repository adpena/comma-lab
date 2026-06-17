# Track-A bind-all CANONICAL config — FINAL reconciliation (all memos informing each other)

**Operator 2026-06-17:** "read all memos and reach a final conclusion, all should be informed by
each other" + "adjust and reconcile to what is optimal" + "we had previously unbundled before you
introduced a regression, don't do that again" + "close gaps". This memo is the SINGLE source of
truth for the canonical Track-A small-basis run config, so the next session inherits the reconciled
optimum instead of re-drifting from memory (the drift that caused the regression). All
`[contest-CPU advisory]`; pointer 0.19110 UNMOVED until a byte-closed dual exact eval.

## The regression that motivated this (the bug class to never repeat)
The MPS-pose UNBUNDLE (`cfg.pose_grad_on_train_device`) was already built + wired (driver `4f2ef0321`
`_split_by_head_backward` L1909-1911 `pose_dev = train_device if on_train else device`; launcher flag
+ guards). The regression was **reconstructing the launch command from memory** and OMITTING
`--pose-grad-on-train-device` → every arm_b/corrected run silently took the 7.3× CPU-pose path
(133 vs 18 s/ep). Sister near-miss: a draft full-MPS launch *dropped* `--split-by-head` (losing the
equimarginal/Mahalanobis levers) and another draft *re-added* `--pose-film-trunk-stopgrad` (the pose
bug). **Root cause: config from memory, not from the canonical doc. This memo is the fix.**

## The FINAL canonical config (every lever, every conflict resolved by cross-memo synthesis)
| lever | FINAL | resolved by (memos informing each other) |
|---|---|---|
| base / taper | base_ch20, solved taper `[22,16,15,14,15,14,10]` | P4 byte-neutrality (−0.548%); roadmap d_seg-aware realloc |
| KD-warm | `--kd-warm-epochs 300` | P4 canonical prime; the unbundle removed the wall-clock wall that forced kd=80 |
| total budget | `--total-epoch-budget 3200` (stage-0=324 ≥ kd 300) | kd-300 needs stage-0≥300; unbundle makes 3200 ep ~22-36h |
| pose carrier | `--pose-film-v2` (pose trains the trunk) | film_v2 memo |
| **trunk-stopgrad** | **OFF** | film_v2 memo §5: A/B-gated on "confirm d_pose HELD"; live arm_b d_pose=0.19 FAILED that gate; structural f1-no-pose-signal. P4's "seal" was code-correctness, not the d_pose A/B. |
| pose synergy | **equimarginal λ_pose + per-dim Mahalanobis** (`--pose-equimarginal --pose-dim-weights-auto`) | sophisticated-pose memo Lever A/C: restores oomph↔pose balance. (The trunk-stopgrad orthogonality synergy is INCOMPLETE; equimarginal is the working one.) |
| pose cadence | **every epoch (APGC OFF)** | score>time memory (throttle is score-negative); the unbundle dissolves APGC's reason to exist (pose now cheap) |
| **pose device** | **MPS unbundle** `--split-by-head --pose-grad-on-train-device --train-device mps` | the "previously unbundled" lever; basin proved MPS pose gradient faithful (d_pose 0.00034); authority eval CPU (`--device cpu` default) — MPS NEVER scores |
| oomph | **1.0** (`--oomph-seg-weight-mult 1.0`) | iso (track_a roadmap): sw1.0 d_seg 0.002790 ≈ sw1.5 0.002786 (equal within noise); 1.5 ONLY adds pose drift. P4 read the SAME iso as "1.5 fine, equimarginal restores"; tie-break = 1.5 buys 0 d_seg → don't pay the drift. **1.0 wins.** |
| **seg surrogate** | **margin_hinge THROUGHOUT** (`--seg-margin-hinge-throughout`) | ACCEL #1 probe (2026-06-17, real frozen SegNet CPU): flip-targeting margin_hinge BEAT soft_cosine (the prior refinement lever, the WORST arm at 0.00407 residual vs margin_hinge 0.00120) AND CE (0.00142), and bent the d_seg power-law exponent +0.18 over CE. soft_cosine's gradient VANISHES on confident flips (1.9e-22) — it cannot push the boundary residual where d_seg lives. Validated PLAIN config = margin_target 1.0 + road_lane 1.0 (OFF; the 2.0 emphasis arm HURT) + NO Lever-5 margin-weight (margin_hinge is intrinsically flip-targeted). Applied to ALL stages (not just refinement) because at 50k the refinement phase starts at ep17,960 (~2.3 days); the first ~18k epochs are CE-family where the descent actively happens (operator 2026-06-17: "throughout (all stages)"). The oomph seg_weight crank stays refinement-only (orthogonal knob). Memo: `accel1_margin_hinge_flip_targeting_dseg_exponent_20260617.md`. NOTE: probe is a small-slice (6-8 pair) overfit; the RELATIVE exponent bend is the trustworthy signal; the restarted 600-pair seed run IS the at-scale validation (vs the logged CE trajectory). |
| rate-attack | `--rate-attack` (variable-level codec, L4 spine) | production-readiness; measured −8,037 B |
| always-on | eval_roundtrip + `--ema-warmup` + PR95 8-stage (QAT/Muon/C1a/σ) | curriculum.py; production-readiness |
| eval | `--async-eval --eval-every 10 --seed 0` | arm_b timing memo: async mandatory (eval 458s non-blocking; CPU authority) |

Canonical launch = `.omx/tmp/arm_b_canonical/supervise.sh` (the flags above). Running 2026-06-17T03:44Z.

## Deterministic reproducibility status (operator ask)
- **SCORE reproducibility = YES.** The authority eval is CPU (`--device cpu`), deterministic, on the
  byte-closed archive. The contest score of the produced artifact is reproducible regardless of the
  training device. MPS is GRADIENT-only, NEVER authority (the launcher AUTHORITY-INVARIANT guard +
  driver `__post_init__` both refuse `device='mps'`).
- **Training-trajectory bit-identical reproducibility = UNVERIFIED on the MPS-pose path.** The 47
  determinism tests + "two same-seed runs bit-identical incl. Muon" were established on the CPU-pose
  path (predate the unbundle `4f2ef0321`); NO determinism test sets `pose_grad_on_train_device=True`.
  MPS kernels can be non-deterministic. **OPEN GAP** (tracked): a same-seed×2 MPS-pose check + a
  determinism regression test. Pragmatic stance: the seed is pinned (0) + the run is resumable + the
  RESULT (archive + CPU score) is the reproducible artifact; only the exact training path may vary.

## Open A/B contingencies (not blockers; the run proceeds on the FINAL config)
1. **oomph 1.0 vs 1.5 WITH equimarginal** — the iso was without the equimarginal restore; a paired
   A/B could confirm 1.0 still dominates once equimarginal is active. Default 1.0; 1.5 is the contingency.
2. **kd 300 vs "the right N"** — 300 is the safe prime; a KD-descent probe (frame-MSE-to-basin) would
   find the minimal N. Non-blocker (P4).
3. **additive-KD (#129)** — the clean fix so kd is independent of stage-0 (vs the budget-bump workaround).

## Cross-refs
P4 `p4_recursive_review_bind_all_20260616T232458Z.md` · film_v2 `film_v2_trunk_decoupling_completion_20260616T200919Z.md` ·
pose `sophisticated_pose_treatment_design_20260616T222900Z.md` · timing `arm_b_training_time_efficiency_20260616T225717Z.md` ·
roadmap `track_a_long_train_then_taper_capacity_roadmap_20260616.md` · production `production_readiness_bind_all_ingredients_20260616.md` ·
memory `score-over-training-time-always-pose-throttle-is-score-negative` + `small-basis-rate-headroom-is-the-sub015-asset`.
