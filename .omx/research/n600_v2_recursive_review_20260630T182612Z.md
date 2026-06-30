# n600 v2 launch-ready design — RECURSIVE ADVERSARIAL REVIEW (independent reviewer)

**UTC** 20260630T182612Z · `[macOS-numpy/MLX advisory · review artifact · NON-PROMOTABLE]` · **pointer 0.19110 UNMOVED.**
Reviewer = INDEPENDENT (not the design author). $0 CPU-only, NO GPU, READ-ONLY on caches/ckpts, NOT committed, NOT launched.
Reviews `.omx/research/n600_v2_launch_ready_design_20260630T180947Z.md` + its 5 inputs + the trainer + the level-set core
+ the coordinator-relayed sister doc `factorized_4d_kplanes_observability_20260630T180753Z.md`.

## VERDICT: **PROCEED-WITH-REVISIONS** (4 binding revisions; 1 is launch-blocking-critical)

The design is sound in substrate, flag-validated (all 116 flags real, no invention), and disciplined (resumable / per-stage
ckpt / EMA-shadow / atomic / best-preservation / perf-env-default-on all verified present). But it ships **one launch-blocking
NO-FAKE error** (a guessed flag value that contradicts the recalled proven config) and **three optimal-form/rate revisions**.
The deepest stated risk (capacity crux #1) **substantially DISSOLVES under a measurement I ran this review.**

---

## Pass history (3-clean-pass recursive discipline)

- **Pass 1** — flag-validation + provenance + capacity crux. Found: (P1-a) `--muon-lr` guessed wrong; (P1-b) crux-#1
  capacity test runnable $0; (P1-c) byte-close runnable $0; (P1-d) `--film-per-layer/--film-concat-code` exist as the
  challenge-#1 additive fixes but are unused.
- **Pass 2** — re-examination found NEW issues: (P2-a) `--verdict-pairs` defaults to 24 (proven used 96) → degraded n600
  telemetry; (P2-b) the proven arm had **all 5 surgical levers OFF** → the design's full command stacks 5 unproven levers +
  arch change + scale change = 3-way confound; (P2-c) `--hardness-oversample 0.5 --hardness-source realized` is a ~1.5×
  wall-clock tax on the baseline; (P2-d) margin-saliency (all-class) + lane-thin (lane) both fire at ep300 with guessed
  weight 1.0 → interacting double-pressure on the lane annulus.
- **Pass 3** — CLEAN. No new issues. All Pass-1/2 items reduce to the 4 revisions below; remaining flags trace to the
  recalled proven cfg or to measured numbers.

---

## The 5 headline challenges — resolved with MEASURED evidence

### #1 THE CAPACITY CRUX — **substantially DISSOLVES (measured this review).**
The design fears the per-pair manifold needs ~26 dims but FiLM PR(M) caps at 1.19/4.57 ≪ 26. **I ran TwoNN + Levina-Bickel
MLE nonlinear-ID (pure numpy, $0) on the two objects that matter:**

| object | LINEAR PR (eff-dim) | TwoNN nonlinear ID | MLE nonlinear ID |
|---|---:|---:|---:|
| proven-arm per-pair **code** (400×32, the thing that hit d_seg 0.003698) | **1.32** | **10.93** | **8.30** |
| **GT partition manifold** (n600, 24×32 one-hot) | 18.77 | **9.79** | **9.27** |

**The "~26" is a LINEAR participation-ratio OVERCOUNT of a curved manifold** (exactly the sister-doc hypothesis, now
MEASURED two independent ways that agree at **~9**). The proven code has linear PR **1.32** yet nonlinearly spans **~9–11
dims** — and the GT manifold's true intrinsic dim is **~9**. So the FiLM multiplicative bottleneck collapses the *linear*
rank but the trunk nonlinearity unfolds the low-linear-rank code into a ~9-dim *nonlinear* manifold that **covers the
~9-dim target.** The proven arm reaching 0.003698 with linear-PR-1.32 code is the existence proof.

**Consequences:** (a) mod-26 has huge headroom (≥9 nonlinear), so the "must be ≥ 26" rationale is built on a
misinterpreted number — harmless choice, wrong justification; (b) **DM1 (`--film-stiefel --code-spectral-entropy-weight`)
is likely solving a NON-PROBLEM** — it raises the *linear* PR, but the *nonlinear* capacity is already adequate. Drop it
from the first launch (the design flags it unproven anyway). (c) The real wall is the **lane-survival/representation
residual** (lane ~39% flip through-R, warp-unexplainable per `screw_warp_through_R_gap2`; PH⁰-dim 0.83, R-recoverable),
NOT per-pair code capacity. Spend on lane-targeted levers + the v2 screw-warp, not on mod/hidden/DM1.

> Caveat: my GT estimate uses a coarse 24×32 one-hot downsample; the code estimate is exact on the trained latent. Both
> are advisory `[macOS-numpy]`. They AGREE at ~9 and both undercut "~26 needed", which is the load-bearing direction.

### #2 RATE (the binding sub-0.15 lever) — **the design's arch is rate-NEGATIVE (measured this review).**
Byte-close via `quantize_levelset_blob` (random-gaussian at trained-per-tensor-std; the **delta** is the signal, the
absolute is upper-ish), n_pairs=600, in_feat=88:

| arch | n_params | base int8+brotli | code int8+brotli | TOTAL | rate_term 25·B/37.5M | ΔS vs proven |
|---|---:|---:|---:|---:|---:|---:|
| **PROVEN 32/96** | 110327 | 63275 | 33553 | 96828 | 0.06447 | — |
| **DESIGN 26/120** | 126863 | 84556 | 27346 | 111902 | 0.07451 | **+0.0100** |
| 32/120 | 139823 | 89697 | 33380 | 123077 | 0.08195 | +0.0175 |
| **26/96 (recommended)** | — | ~63275 | ~27346 | **~90621** | **~0.0603** | **−0.0042** |

`hidden 96→120` is the rate-costly half (+21KB base; trunk params scale hidden²·4 = 36864→57600, **+56%**, robust to
compressibility). `mod 32→26` only saves ~6KB code. Given (a) RATE is the binding lever and (b) crux #1 shows capacity is
already adequate at hidden-96, **the hidden bump costs +0.010 S to add trunk capacity the ~9-dim manifold does not need.**
The RD-optimal direction is **26/96**: a rate WIN (~−0.004 S) at ~zero capacity risk. Design's own Open-Q1 (hidden-128 →
161KB, +0.026 S) already pointed here.

### #3 LAUNCH SEQUENCING — **attribution-clean FIRST (confirmed; the proven arm proves it).**
The run log shows the proven 0.003698 used **`--lane-edge-weight 0`, NO margin-saliency, NO lane-thin, NO hardness, NO
film-stiefel, NO code-spectral-entropy** — i.e. NONE of the 5 surgical levers. So the design's *full* command stacks 5
unproven levers + arch change + scale change (n200→n600) = a 3-way confound with no clean attribution. The
attribution-clean variant (proven base + arch) isolates scale+arch; the θ*-filled levers then land as a **shape-compatible
warm-start re-treatment** (all 5 are loss/projection-only — no new params — so they resume the attribution-clean ckpt
cleanly). **Launch attribution-clean first.** Lower risk (guessed weight-1.0 levers may hurt; margin-saliency+lane-thin
double-fire on the lane annulus at ep300), wall-clock-cheaper (no +50% hardness oversample tax), cleanest attribution.

### #4 DISCIPLINES — **all present and verified.**
- Resumable: `--resume-from` restores live+EMA-shadow+optimizer+epoch (line 1623+). From-scratch n600 correctly omits it. ✓
- Per-stage ckpts: `_STAGE_TAGS` {CE,Tau,L7,Hinge} + `_do_checkpoint(stage_tag=…)` + proven arm emitted
  `stageMuonStart_ep726` / `stageL7_muon_ep1000`. `--stage-checkpoints` default ON. `--ckpt-every 25` rolling. ✓
- EMA-shadow (not live) + best-preservation: `_maybe_preserve_best` → `levelset_witness_ema_BEST.npz` + `levelset_best.json`,
  Stiefel-projected shadow that matches the verdict, `_verdict_lock` thread-safe, atomic. ✓
- Atomic writes: `_atomic_savez` / `_atomic_write_json` = tmp + `os.replace`, refuse /tmp. ✓
- Perf-env: `_custom_metal_backward_status` default `os.environ.get(...,"1")` = ON; `_log_custom_backward_decision_once`
  (mlx_scorer_adapters.py:1173) imported by trainer (line 818). The "MUST verify `active=True`" launch-log step is a valid
  runtime gate. ✓
- Config validators fire pre-GPU: curriculum `0<300<600≤1000` ✓; muon `726≥600` ✓ (lines 2556, 2636). ✓
- min-free-gb 10 / contained out-dir / one-GPU-await-go: present in the design checklist, consistent with standing rules. ✓

### #5 NO-FAKE SWEEP
- ✅ **Design's 2 self-catches STAND:** `--margin-saliency-target` IS a float (default 0.5) — the source recipe's
  `--margin-saliency-target lane` would crash; the design correctly does not propagate it.
- 🔴 **NEW CRITICAL CATCH — `--muon-lr` is guessed and WRONG.** Design omits it → trainer default `0.1·lr = 1e-4`, claiming
  that's "the value the proven n200 arm most likely used." **The run log + `muon_finisher_switch` JSON show the proven arm
  used `--muon-lr 0.002`** (2e-3, 20× higher; also inside the help's optimal-form range 1e-3–5e-3). Omitting it runs the
  Muon finisher at 1/20th LR and would NOT reproduce the 0.003698 descent. This is the "flag value guessed vs derived"
  NO-FAKE class. **Must add `--muon-lr 0.002`.**
- 🟡 **mod-dim↔eff-dim provenance is real-but-misinterpreted.** eff-dim 26.33 is genuinely in the genprobe (line 61) but it
  is a LINEAR PR; the capacity-relevant nonlinear ID is ~9 (measured above). The number is sound; its use as "required
  capacity" is not.
- 🟡 **structured-init benefit unproven but de-risked:** the proven arm used `--structured-init --structured-init-include-lane`
  + hosc + siren-init and did NOT stall (pretrain disagree 0.00313) — so the Open-Q5 fragility is real but there is an
  existence proof it works in this exact config. (Note: `lane_static_mask_px=0` in the proven run → the static-lane band is
  a no-op; lane is fully learned, as intended.)
- 🟡 **`--tau-softplus-tau` omitted → default 0.3 = proven value** (proven set it explicitly to 0.3). No action, but it is
  load-bearing and silently default-matched.

---

## THE REVISIONS (apply before launch)

1. **[CRITICAL] add `--muon-lr 0.002`** — recalled proven value; the omitted default 1e-4 is 20× too low and would not
   reproduce the measured descent.
2. **Launch the ATTRIBUTION-CLEAN variant first** — drop `--margin-saliency-*`, `--lane-thin-*`, `--hardness-*`,
   `--film-stiefel`, `--code-spectral-entropy-weight`, `--dm1-telemetry` (keep `--dm1-telemetry` only if you want the PR(M)
   baseline row; it's log-only). These were ALL OFF in the proven 0.003698. Re-treat them as a shape-compatible warm-start
   after #183 fills θ*.
3. **Set arch to `--mod-dim 26 --hidden-dim 96`** (NOT 120) — the byte-close shows 26/96 is a rate WIN (~−0.004 S) vs proven
   while 26/120 is +0.010 S; the nonlinear-ID test shows hidden-96 capacity is already adequate. If the residual-capacity
   hypothesis must be tested, do it as a SEPARATE RD-curve arm, not the baseline. (Pure-control alternative: 32/96 to
   reproduce proven exactly first; 26/96 is the higher-EV single launch.)
4. **Add `--verdict-pairs 96`** (or ~120 for n600) — design's default 24 is a degraded, non-apples-to-apples realized-d_seg
   verdict at 600 pairs (telemetry-accuracy discipline; the proven arm verdicted on 96).

### Revised FIRST-launch command (attribution-clean, rate-positive, optimal-form)
```bash
TAC_MLX_CUSTOM_GROUPED_BACKWARD=1 \
.venv/bin/python experiments/train_levelset_witness_realized_through_R_mlx.py \
  --out-dir experiments/results/levelset_n600_v2_attrclean_<utc> \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz --num-pairs 600 \
  --mlx-device gpu --seed 0 --async-verdict \
  --epochs 1000 --eval-every 25 --verdict-pairs 96 \
  --curriculum --tau-softplus-start-epoch 300 --l7-start-epoch 600 \
  --muon-start-epoch 726 --muon-lr 0.002 --muon-momentum 0.95 --muon-ns-steps 5 \
  --stage-transition-rewarmup-epochs 8 --stage-transition-rewarmup-floor 0.1 \
  --stage-transition-rewarmup-shape linear --stage-transition-reset-moments \
  --w-seg 100 --w-pose 0 --score-domain-loss \
  --mod-dim 26 --hidden-dim 96 --n-hidden 4 \
  --activation hosc --hosc-beta 4.0 --hosc-omega 1.0 --siren-init \
  --softmax-temp-start 1.0 --softmax-temp-end 0.05 \
  --self-orient --n-dir-freqs 2 --freq-across 32 --freq-along 4 --reorient-every 50 \
  --max-bank-freq 64 --chroma --palette-anchor \
  --eikonal-weight 0.01 --length-weight 0.001 \
  --render-h 384 --render-w 512 \
  --structured-init --structured-init-include-lane \
  --lane-prior-phi1 --lane-prior-phi1-mode replace --lane-prior-phi1-dash-gate \
  --ckpt-every 25
```
(Then warm-start re-treat with the θ*-filled surgical levers — they add no params, so the resume is shape-compatible.)

---

## On the sister doc (factorized-4D / S²WL) — folded into sequencing, NOT this launch
My nonlinear-ID ~9 is CONSISTENT with the sister's "~6 screw + ~8 lane-orbit, shared" decomposition (it CONFIRMS ~26 is a
linear artifact). But S²WL is a v2 RATE-half BUILD (screw temporal factor + canonical-frame coding + poly-SDF lane +
persistence residual), not a trainer-flag config — and `screw_warp_through_R_gap2` already showed previous-frame bulk-warp
is ~4× over budget (the decisive next $0 step is the **clean stored-canonical warp**, not a launch). The INR witness launch
does NOT block on it and correctly scopes it to v2. **Recommendation:** launch the attribution-clean INR now (it is the
lane+movables learned residual either way); pursue S²WL's screw factor + the Step-0 canonical-warp measurement in PARALLEL
as the rate-half v2. They compose; neither blocks the other.

## Honest provenance
- MEASURED this review (advisory `[macOS-numpy]`): nonlinear-ID (TwoNN/MLE) on code + GT manifold; byte-close arch deltas;
  recall of `--muon-lr 0.002` + all-levers-OFF from the proven run log; all 116 flags validated against argparse.
- NOT a score: pointer **0.19110 UNMOVED**. realized d_seg is the surrogate; the only END is a byte-closed n600 exact row
  < 0.19110 (CPU/CUDA, never MPS). Every recommendation here is a MEANS.
