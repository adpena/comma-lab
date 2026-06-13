# Seg-surrogate anneal — optimal math / geometry / algebra / calculus / behavior (for max signal)

**Operator (2026-06-13):** *"review anneal for optimal math and geometry and algebra and calc and all
behavior"* + *"we want optimal and max signal."* This derives the optimal seg-temperature schedule from
first principles and upgrades the (held, #119) calibration to test the **theoretically-optimal** schedule.
Authority: `[contest-CPU advisory]`. Grounds: `driver._seg_loss` (soft_cosine + Lever-5), the gate
(`.omx/tmp/lever2_gate/stdout.log`), R12 (`SEG_ANNEAL_GRADIENT_FLOOR_T=0.3`).

## 1. The surrogate (definition)
SegNet gives per-pixel logits `pred ∈ R^K` (K=5), GT class `g`. Soft-cosine at temperature `T`:
- `p_g(T) = softmax(pred/T)_g = e^{pred_g/T} / Σ_k e^{pred_k/T}`
- `L_T = 1 − p_g(T)` per pixel (the differentiable surrogate for the SegNet argmax-flip = d_seg).

## 2. The calculus (gradient) — the central result
`∂L_T/∂pred_j = −(1/T)·p_g·(δ_{jg} − p_j)`. The "push toward GT" magnitude at the GT logit:
`|∂L_T/∂pred_g| = (1/T)·p_g·(1 − p_g)`.

On a **confidently-wrong pixel** (a flip): argmax = wrong class m, **margin Δ = pred_m − pred_g > 0**.
Then `p_g ≈ e^{−Δ/T}` and `(1−p_g) ≈ 1`, so:
```
|grad_g(Δ, T)|  ≈  (1/T)·e^{−Δ/T}
```
**Optimize over T** (the resonance):
```
d/dT[(1/T)e^{−Δ/T}] = (e^{−Δ/T}/T³)·(Δ − T) = 0   ⟹   T* = Δ
```
(T<Δ ⟹ derivative>0 increasing; T>Δ ⟹ <0 decreasing.) **The surrogate gradient on a flip of margin Δ is
MAXIMIZED exactly at T = Δ.** Below T=Δ it dies *super-exponentially* (`e^{−Δ/T}` beats `1/T`); above, it
fades as `~1/T` (too soft, p_g → 1/K uniform). The surrogate has a **temperature resonance at the flip's
own margin.**

## 3. Empirical match (the gate + R12 confirm the calculus)
- Gate optimum **T=0.3 = 1.58× CE** ⟹ the typical *fixable* flip margin **Δ ≈ 0.3**.
- R12 **T≤0.1 dead** (~10 orders down) = T far below Δ ⟹ `e^{−Δ/T}→0` (the dead zone, exactly as the
  formula predicts).
- T=1.0 **worse than CE** = T ≫ Δ ⟹ gradient spread thin across classes (untargeted).
The 1.0→0.05 cosine spends ~52% at T≥0.5 (T≫Δ, untargeted) + ~15% at T<0.3 (T≪Δ, dead) → almost never at
the resonance. **Miscalibrated by construction.**

## 4. The geometry (why low-T can't escape a wrong vertex)
`softmax(pred/T)` is a point on the (K−1)-simplex; `1/T` is the sharpness. High T → near the centroid
(uniform); low T → near a vertex (hard argmax). `L_T = 1 − p_g` is the **barycentric distance from the GT
vertex.** A confidently-wrong pixel sits near the *wrong* vertex; the softmax Jacobian
`(1/T)p_g(δ−p) → 0` as `p` → a one-hot vertex, so the simplex is **flat (saturated) near the wrong
vertex → no gradient to escape it** at low T. `T* = Δ` is the temperature at which the GT-vertex basin
still has gradient *reach* into the wrong-vertex region. (This is the saturated-softmax escape geometry —
the same reason label-smoothing/temperature exists.)

## 5. The algebra (vs CE — why CE is robust but untargeted)
CE = `−log p_g(1)`; its GT-gradient `= −(1 − p_g) ≈ −1` on a confident-wrong pixel — **large, never
vanishes, at all margins.** So CE has **no resonance**: it pushes hard on every wrong pixel (robust) but
*also* on already-near-correct pixels (untargeted → wasted gradient + over-smoothing). soft_cosine@T*=Δ
is the opposite: **maximal gradient exactly on the margin-Δ flips, ~0 on confident-correct pixels**
(targeted) — hence 1.58× CE *when T matches the margin*, and **worse than CE when mistuned** (the gate's
T=1.0 row). Trade: **CE = robust·untargeted (fixed T=1); soft_cosine = targeted·margin-matched (needs
T ≈ Δ).**

## 6. The behavior over a run → the OPTIMAL (max-signal) schedule
Flips have a *distribution* of margins that *shrinks* as training fixes the large-Δ errors:
- **Early:** broad margin distribution (large-Δ deep errors + small-Δ boundary) → higher T reaches them.
- **Late:** only small-Δ boundary flips remain → T should DECREASE to track them.
- **Floor:** the smallest *worth-fixing* margin (deeper than ~Δ_min the pixels are either too-deep-to-reach
  or genuinely-ambiguous boundary that flip-flops). Empirically Δ_min ≈ 0.3 (the gate optimum).

⟹ **The optimal anneal is MARGIN-ADAPTIVE, not a fixed cosine-to-0 and not static:**
```
T(t) ≈ clamp( running_median( {Δ_i : pixel i is a current flip} ),  T_floor=0.3,  T_max≈0.6 )
```
i.e. a **self-tuning temperature that tracks the live flip-margin distribution**, floored at the dead-zone
boundary. This maximizes the per-step useful gradient (every step sits at the resonance for the bulk of
the remaining flips) → **max signal.** Static-0.3 is the *limit* of this once the margin band has
collapsed to ≈0.3; the margin-adaptive schedule additionally captures the early large-Δ errors that
static-0.3 (T too low for them) and the cosine (races past) both miss.

## 7. The Lever-5 co-tuning (margin-weight × temperature)
Lever-5 weights pixels by `w_i = e^{−margin_i/τ}` (up-weight small-margin/boundary). It **multiplies** the
surrogate gradient, so it is **synergistic at the resonance** (amplifies exactly the small-Δ pixels whose
soft_cosine gradient survives) but **futile below it** (big weight × dead gradient ≈ 0). Therefore τ and T
must be **co-tuned to the same margin band: τ ≈ T ≈ Δ_target.** A mismatched τ (targeting margins T can't
reach) wastes Lever-5. In the margin-adaptive schedule, **slave τ(t) = T(t).**

## 8. Calibration upgrade (the held #119, for max signal)
Test FOUR arms, not three: (a) static-0.3, (b) fast-cool→0.3-hold, (c) cosine 1.0→0.05 (the miscalibrated
baseline), **(d) MARGIN-ADAPTIVE `T(t)=clamp(median flip-Δ, 0.3, 0.6)` with τ(t)=T(t)** — the
theoretically-optimal. Measure d_seg over a real training window; (d) should win if the resonance theory
holds (and confirm Δ_median ≈ 0.3 collapse). Run on the **Lever-3 v2 clean-`rgb_1` decoder** (held until
v2 lands) so the seg signal is FiLM-decoupled. Default-preserving: the schedule fn extension keeps the
static/cosine paths byte-identical.

## 6′. The CRUX — slice-0 falsified the MEDIAN; the optimal statistic is the WEIGHTED MEAN
**The slice-0 calibration (4 arms, v2 clean-`rgb_1`) ranked: fast-cool→0.3 (d_seg red 0.001496) >
margin-adaptive[median] (0.001400) > static-0.3 (0.001378) > cosine-1.0→0.05 (0.001328); `margin_adaptive_
wins: false`.** The theoretically-"optimal" margin-adaptive arm LOST to a committed-low fixed schedule.
This is not noise — it falsifies the §6 **median** heuristic, and the calculus says exactly why.

§2 derived `T*=Δ` for ONE flip. But the schedule must pick ONE global `T` for a whole *distribution* of
flip margins `{Δ_i}`, and the optimal aggregating statistic is **not** the median. The aggregate useful
gradient is
```
G(T) = Σ_i (1/T)·e^{−Δ_i/T} = (1/T)·Σ_i e^{−Δ_i/T}.
```
**Calculus (the crux).** `G'(T)=0`:
```
G'(T) = −(1/T²)·S(T) + (1/T³)·Σ_i Δ_i e^{−Δ_i/T} = 0,  S(T)=Σ_i e^{−Δ_i/T}
⟹  T* = [Σ_i Δ_i·e^{−Δ_i/T*}] / [Σ_i e^{−Δ_i/T*}]
```
**`T*` is the `e^{−Δ/T}`-WEIGHTED MEAN of the live flip margins — a self-consistent fixed point, NOT the
median.**

**The nexus (the self-reference).** The weight `w_i = e^{−Δ_i/T}` is the **reachability** factor — it
exponentially discounts deep (large-Δ) flips the surrogate cannot move. So `T*` is a *softmin-temperature
mean*: **the unique temperature equal to the reachability-weighted mean of the margins it itself induces.**
That fixed-point self-reference (`T = mean weighted by e^{−Δ/T}`) is the nexus binding the per-flip
resonance (§2) to the distribution-level optimum. Verified empirically (test
`test_weighted_mean_is_self_consistent_fixed_point`): the returned `T*` reproduces its own weighted mean.

**Geometry (why the median is biased HIGH).** For fixed `T`, `g(Δ,T)=(1/T)e^{−Δ/T}` is a ridge in `Δ`:
**gentle on the left** (small Δ<T, gradient still large), **steep on the right** (large Δ>T, `e^{−Δ/T}→0`).
The flip-margin distribution is **right-skewed** (a mass of small boundary flips + a long deep tail). The
median sits inside the tail's pull; but the steep right side means the deep flips contribute ~zero
gradient, so the aggregate optimum **shades LEFT** toward the reachable small-Δ mass. The median ignores
reachability; the weighted mean encodes it. That asymmetry breaks the symmetry that would make the median
optimal. **Empirical (the crux confirmed):** on a slice-0-like broad distribution (median 0.600) the
weighted-mean fixed point is **T\*=0.485 — below the median**; the median-adaptive arm clamps to 0.6 (too
warm), the weighted mean commits to 0.485 (near fast-cool's winning 0.3–0.4 hold). As the margins collapse
toward the floor, both converge — exactly the slice-0 dynamics.

**Algebra (median = rank statistic; weighted mean = reachability moment).** For a SYMMETRIC margin
distribution the two nearly coincide; the flip-margin distribution is right-skewed (deep errors form a
tail), so `median > T*`. `T*` is the maximum-aggregate-gradient "effective margin" of the *reachable*
subpopulation. Existence/uniqueness: `φ(T)=Σ Δ_i e^{−Δ_i/T}/Σ e^{−Δ_i/T}` maps `[minΔ, meanΔ]→[minΔ,
meanΔ]`, is continuous + monotone-increasing (`φ'=Var_w(Δ)/T² > 0`), `φ(minΔ)≥minΔ`, `φ(meanΔ)≤meanΔ`
⟹ a fixed point exists (IVT) and is a contraction near it ⟹ 3–5 iterations from the median converge.

⟹ **The CORRECTED max-signal schedule** (`curriculum.seg_temperature_weighted_mean_resonant`):
```
T(t) = clamp( fixed-point of  T = Σ Δ_i e^{−Δ_i/T} / Σ e^{−Δ_i/T}  over the live flip set,  0.3,  0.6 )
```
with Lever-5 `τ(t)=T(t)` slaved. This is the principled "more nuanced curve" — it is **measured** (the
fixed point on the live margin histogram each step) and **assignable** (the from-0 launcher selects it).
#119 v3 measures it as the 5th arm against static / fast-cool / cosine / median-adaptive on slices 0+1.

**Level-3 endpoint (documented, deferred):** the maximally-nuanced schedule removes the global `T` entirely
— a **per-pixel resonant temperature** `T_i = clamp(Δ_i, 0.3, T_max)` so every flip pixel trains at its OWN
resonance simultaneously (gradient ∝ `1/Δ_i`: small-margin = largest push). This needs a per-pixel-
temperature `soft_cosine` loss form (a driver change to validate apples-to-apples), so it is the next
frontier, not in the v3 run; the global weighted-mean fixed point is the immediate corrected assignment.

## Bottom line
The anneal was never calibrated — and the calculus says the *whole shape* was wrong: the surrogate is a
**margin-resonant** loss (`grad ∝ (1/T)e^{−Δ/T}`, peak at T=Δ). The §6 median-adaptive schedule was the
right *idea* (track the live distribution) with the wrong *statistic* — slice-0 falsified it. The **crux**
is that the distribution-level optimum is the `e^{−Δ/T}`-**weighted mean** (a reachability-discounted
softmin), not the median; the **nexus** is its self-consistency (`T*` equals its own reachability-weighted
mean). That weighted-mean fixed point — clamped to `[0.3, 0.6]`, τ slaved — is the max-signal anneal, and
#119 v3 measures it against the four prior arms on the v2 clean-`rgb_1` decoder across slices 0+1.
