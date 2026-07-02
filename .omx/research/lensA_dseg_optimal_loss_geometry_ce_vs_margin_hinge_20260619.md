# LENS A — d_seg-optimal LOSS GEOMETRY: is the vendored CE-curriculum the d_seg lever, or does a boundary-weighted MARGIN-HINGE descend the argmax-flip rate faster/lower? (2026-06-19)

**Mandate:** RECURSIVE MATH-OPT + ADVERSARIAL REVIEW, Lens A (loss geometry & d_seg-finishing).
Council seats channeled: **Shannon LEAD** (score-domain Lagrangian / R(D)), Rudin (interpretable
surrogate), Yousfi+Fridrich (boundary/steganalysis), Mallat (multiscale).
**Authority:** every number here is `[macOS-MPS research-signal] / [contest-CPU advisory]` NON-PROMOTABLE
(real frozen SegNet, seg-isolated, small-n; a SLOPE/ranking diagnostic, NOT a score claim). The exact
frontier pointer is UNMOVED (`0.19110` contest-CPU). NO paid dispatch. $0 local MPS.

---

## 0. Score-domain framing (Shannon LEAD) — why the loss is the highest-leverage lens

Basin (`torch_vehicle_full_mps_basin_bc20_n600`, advisory): d_seg=0.00260, d_pose=0.000342, rate=0.00237.
`S = 100·d_seg + √(10·d_pose) + 25·B/N` → **seg term = 0.260 = 81% of S** (pose 0.0585, rate 0.0024).
- To beat the borrowed frontier 0.19110 (keeping the small-basis rate+pose floor 0.1178): need **d_seg ≤ 0.000733** (basin is 3.5× too high).
- For sub-0.15: need **d_seg ≤ 0.000322** (basin is 8× too high).
- `∂S/∂d_seg = 100` (constant); `∂S/∂d_pose = 5/√(10·d_pose) = 85.5` at basin pose = **86% of d_seg's marginal** (the pose-destabilization caveat is real and quantified — §5).

d_seg is the binding term; the seg loss is where it is shaped. **This lens is on the critical path.**

---

## 1. The derived d_seg-optimal loss geometry (bare math)

The contest d_seg is EXACT (`upstream/modules.py:112`): `(SegNet(out1).argmax != SegNet(out2).argmax).float().mean()`
on the last frame — a per-pixel **hard argmax-flip indicator**. The training loss is a smooth surrogate for it.

Let pred logits `z ∈ R^K` (K=5), GT class `g`, and on a flip pixel the winning wrong class `m` with
**logit margin Δ = z_m − z_g > 0**. The push that fixes a flip raises `z_g` above `z_m`.

**(a) Vendored CE** (`stages.stage1_v328_ce.ce_seg_loss = F.cross_entropy`): GT-logit gradient
`∂/∂z_g = p_g − 1 ≈ −1` for ALL wrong pixels (p_g→0) — and ALSO `p_g − 1 ≠ 0` on confident-correct
interior pixels. **Robust but UNTARGETED**: spends gradient magnitude on pixels the argmax already wins.

**(b) Soft-cosine** `1 − softmax(z/T)_g` (the current `from0` lever): GT-logit gradient
`(1/T)·p_g(1−p_g) ≈ (1/T)·e^{−Δ/T}` on a flip. Calculus: maximized at the **resonance T*=Δ**; below it
dies super-exponentially (`e^{−Δ/T}→0`), above it fades ~1/T. **Targeted but margin-matched** — needs
T≈Δ, and on deep flips (Δ≫T) the pull VANISHES (the saturated-softmax escape-geometry trap, anneal-memo §4).

**(c) Margin-hinge** `L = relu(m_target − (z_g − max_{c≠g} z_c))` (`segnet_margin_hinge_per_pixel`):
- ZERO gradient when correct-with-margin (`z_g − max_{c≠g} ≥ m_target`) → **no wasted interior gradient**;
- CONSTANT slope −1 pull on EVERY flip / near-flip, **independent of flip depth Δ** → no resonance to
  mistune, no dead zone. Defined on the **raw-logit surface where the argmax lives** — no temperature.

**Empirically verified GT-logit gradient on a confident flip** (`tac.losses.core`, this session):

| margin Δ | margin-hinge | soft_cosine T=0.3 | soft_cosine T=1.0 |
|---:|---:|---:|---:|
| 0.3 | **−1.000** | −0.422 | −0.152 |
| 1.0 | **−1.000** | −0.101 | −0.127 |
| 3.0 | **−1.000** | −1.5e-4 (dead) | −0.040 |
| 6.0 | **−1.000** | −6.9e-9 (dead) | −0.0024 |

**The d_seg-optimal loss is the margin-hinge**: it is the only one whose gradient is (i) ZERO on
correct-interior (no wasted capacity, unlike CE) AND (ii) NON-VANISHING on every flip regardless of depth
(unlike soft-cosine). It is the Crammer-Singer multiclass hinge restricted to "GT vs its top competitor" —
exactly the binary margin the contest argmax flips on. (Rudin: it is also the most interpretable — a
piecewise-linear contract on the exact decision surface, not a temperature-distilled proxy.)

---

## 2. MEASURED A/B — CE vs margin_hinge vs soft_cosine, REAL frozen SegNet, from the converged basin

`experiments/probe_lensA_ce_vs_margin_dseg_slope.py` (new). Forkpoint EMA shadow latents
(`basin_bc20_20260612T121523Z`, the actual converged operating point d_seg≈0.0033 on the slice),
n48, seg-isolated (pose_weight=0), MPS train device, deepcopy-per-arm bit-identical init, AdamW lr=1e-3,
seg_weight=100, same seed. The seg-loss FUNCTION is the only variable. d_seg measured EXACTLY on the
same MPS SegNet for every arm (apples-to-apples ranking).

**Floor regime (forkpoint, 150 steps), real frozen SegNet:**

| arm | d_seg 0→end | Δ (%) | final ÷ CE | descent slope b | grad-norm@1 | median flip-margin 0→end |
|-----|------------:|------:|-----------:|----------------:|------------:|-------------------------:|
| **CE (vendored)** | 0.003339→0.002277 | +31.8% | 1.000 | −0.166 | 0.637 | 0.416→0.380 |
| **margin_hinge m=1.0** | 0.003339→0.001906 | +42.9% | **0.837** | −0.197 | 0.945 | 0.416→0.286 |
| **margin_hinge m=0.5** | 0.003339→0.001463 | +56.2% | **0.643** | **−0.313** | 1.481 | 0.416→**0.173** |

(CE reproduced at 100 steps on a relaunch: 0.002342, slope −0.171 — consistent.)

**The mechanism is directly visible:** margin_hinge m=0.5 drives the **median flip margin 0.416→0.173**
(it is actively closing the flip margins — raising z_g on exactly the flip set), while CE barely moves it
(0.416→0.380, because it spreads gradient onto correct-interior pixels). The hinge's grad-norm GROWS
(1.48 vs CE 0.64) — the non-vanishing flip pull — and its descent slope is ~2× CE's.

**Repair regime (random init, the from-0 dynamic) — corroborating LANDED evidence** (independent probe
`accel1_margin_hinge_exponent_random_20260617.json`, real frozen SegNet, CPU): margin_hinge reached the
LOWEST residual d_seg (0.00120 vs CE 0.00142 vs soft_cosine **0.00407**) with the steepest LATE power-law
exponent (p_late 0.787 vs CE 0.608 vs soft_cosine 0.646, a **+0.18 bend over CE**) and a non-collapsed
grad-norm (125 ≈ CE's 111; soft_cosine collapsed to 23). **soft_cosine was the WORST arm** (3× CE's
residual) — its gradient vanishes on the flips, exactly as §1 predicts.

**soft_cosine at the floor** (`lever2_softcosine_vs_ce_flipfix_20260613T002902Z.json`): modestly beats CE
(Δd_seg 0.000503 vs 0.000200 at T=0.3) but is a clear DOWNGRADE on random init (LEVER-2-DOWNGRADE). It is
regime-fragile; margin_hinge wins in BOTH regimes.

---

## 3. RANKED loss / d_seg-finishing optimizations (each with the exact config change + evidence)

1. **REPLACE the seg surrogate with `margin_hinge` (the #1 lever).** Measured ≥16% lower residual d_seg
   at the floor (0.837–0.643× CE) AND lowest residual + steepest exponent in the repair regime; wins in
   BOTH regimes where soft_cosine is regime-fragile. PR95-faithfulness: the vendored stage-1 loss is
   plain `F.cross_entropy` — a margin-hinge is a deviation, but it EARNS its place by measured d_seg on
   the exact contest metric, and the hinge is the tighter surrogate of the argmax-flip the curriculum was
   already chasing. The loss + wiring are production-grade (15 passing tests in
   `test_segnet_margin_hinge_loss.py`; default path byte-identical).
   - **`experiments/launch_from0_lever_stack_ab.py:59`** `_SEG_SURROGATE = "soft_cosine"` → `"margin_hinge"`
     (the from0 decisive-run launcher still uses soft_cosine — the WORST arm).
   - OR use the already-wired campaign flag **`experiments/launch_bind_all_taper_ab.py --seg-margin-hinge-throughout`**
     (the `_apply_margin_hinge_curriculum` path exists, default OFF at `seg_margin_hinge_throughout=False`).

2. **Set `seg_margin_hinge_target = 0.5`, NOT the validated 1.0 (NEW finding this lens).** At the
   converged floor the flip margins have collapsed to ~0.2–0.4 (measured median 0.38); m=0.5 reached
   0.643× CE vs m=1.0's 0.837× — a further ~23% d_seg cut with a ~2× steeper slope. The campaign's
   "validated PLAIN config = margin_target 1.0" was tuned in the repair regime; at the floor the lower
   target is better (it concentrates the constant pull on the reachable near-flips). **A margin-target
   ANNEAL 1.0→0.5 across the curriculum** (large early when margins are broad, small late when they
   collapse) is the principled schedule — the hinge analogue of the soft-cosine T-anneal, but on the
   raw-logit surface (no dead zone). Config: `StageSpec.seg_margin_hinge_target` per stage (the field
   exists, `curriculum.py:151`); a `seg_margin_hinge_target_end` anneal hook is the small driver addition.

3. **DROP Lever-5 `margin_weight_tau` when using the hinge.** The hinge is intrinsically flip-targeted
   (zero gradient on correct-with-margin pixels) — the `exp(−margin/τ)` detached weight is redundant and
   can only SCALE DOWN gradient. The accel1 finding + this lens agree (the from0 `_MARGIN_WEIGHT_TAU=1.0`
   is for soft_cosine). Set `margin_weight_tau=None` (the StageSpec default) on the hinge path.

4. **Keep `road_lane_emphasis = 1.0` (OFF).** Probe E found 64% of flips are road↔lane, but accel1
   measured the 2.0 emphasis HURT the bare hinge (p_late 0.687 < 0.787) — over-weighting traded away the
   other 36% real flips. Leave it at the default; revisit only as a scale-time sweep.

5. **Keep FiLM-v2 trunk-stopgrad ON when the hinge is cranked (pose protection).** `pose_film_version=2`
   + `pose_film_trunk_stopgrad` decouples ∂d_seg/∂(pose-objective)=0 so a stronger seg loss cannot drag
   pose via the shared trunk. The from0 `stack`/`bind_all_taper` arms already set v2 (the §5 mitigation).

---

## 4. ADVERSARIAL VERDICT — is the loss the lever? **NOT-CLEAN on "CE is optimal" → the loss IS a lever.**

The mandate asked for a clean/NOT-clean verdict on whether the loss is the lever. **Verdict: the vendored
CE-curriculum seg loss is NOT d_seg-optimal — it is the UNTARGETED member of the family, and a flip-targeting
margin-hinge measurably descends the exact d_seg ~16–36% faster/lower on the real frozen scorer, in BOTH
the floor and repair regimes.** The loss is a real, measured lever (not a tied/noise difference): the
ranking is monotone, reproduced, mechanistically explained (the gradient table §1 + the median-margin
collapse §2), and the slope bend is ~2× at the floor.

**Adversarial point #4 (is it loss or capacity/optimizer?) — HONEST PARTIAL:** the basin's measured d_seg
power-law exponent is **b ≈ −0.45** over the full run (−0.21 is a late slice). The hinge BENDS this
exponent (accel1 p_late +0.18 over CE; this lens slope −0.31 vs −0.17 at the floor) — so the shallow slope
is PARTLY a loss issue, and the loss lever is real. BUT carrying the bend onto the canonical 600-pair late
exponent (≈0.19, per Probe C) gives ≈0.24–0.25 — which moves the d_seg(50k) projection TOWARD but does NOT
by itself clear sub-0.15. **The honest claim: the hinge is a real accelerant of the d_seg descent that
soft_cosine (the current lever) is not, but it is NOT a proven standalone path to sub-0.15.** The capacity
question (whether base_ch20 can even reach d_seg 0.000322) is a SISTER lens (HiNeRV-grid N1 probe / taper
realloc) — this lens establishes the loss is not the BOTTLENECK it was suspected to be, but capacity may
still be the ultimate wall.

**Adversarial point: does margin-hinge destabilize pose?** The probe isolates seg (pose_weight=0) so this
lens cannot measure it directly. Reasoned bound: the hinge gradient is BOUNDED (constant magnitude ≤
seg_weight per flip pixel, ReLU-gated to the flip set) and is ZERO on correct-interior — it is actually
GENTLER on the shared trunk than CE in the interior while stronger on the flip boundary. The pose-drift
risk (∂S/∂d_pose=86% of d_seg's) is the SAME for any cranked seg loss and is structurally mitigated by
FiLM-v2 trunk-stopgrad (rec #5), orthogonal to the CE-vs-hinge choice. **The decisive next measurement is
the 600-pair from-basin A/B (hinge vs the live soft_cosine lever) on the FULL driver (seg+pose) measuring
BOTH the late d_seg exponent AND d_pose stability** — exactly the campaign gate `--seg-margin-hinge-throughout`
unlocks. This lens establishes the loss-geometry case for it.

---

## Artifacts
- `experiments/probe_lensA_ce_vs_margin_dseg_slope.py` (new; the CE/margin_hinge/soft_cosine slope A/B).
- `.omx/tmp/lensA_slope_20260619T224145Z.log` (the complete 3-arm floor-regime A/B).
- Gradient-table verification (this session, `tac.losses.core` inline).
- Corroborating LANDED: `accel1_margin_hinge_exponent_random_20260617.json` +
  `accel1_margin_hinge_flip_targeting_dseg_exponent_20260617.md` (repair regime, power-law) +
  `lever2_softcosine_vs_ce_flipfix_20260613T002902Z.json` (soft_cosine floor/random) +
  `anneal_optimal_math_geometry_calculus_20260613.md` (the soft_cosine resonance calculus).
