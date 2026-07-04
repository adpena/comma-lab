# Scaling-law FACET 2 — the intrinsic-manifold parametrization as a scaling-law lever

**Facet 2 of the 4-facet "geometry-optimal scaling-law engineering" pass (task #284/#285 lineage).**
Operator frame: *"engineer to change scaling laws optimal based on differentiable geometry, bridged with
deep math and engineering."* MY facet = **the intrinsic-manifold parametrization → the effective DOF →
the d_seg-vs-compute SCALING EXPONENT.** $0 research, no heavy/paid/GPU, #205 SACRED READ-ONLY.
**MEANS, not ends** — pointer contest-CPU **0.19110 UNMOVED**; this memo names the geometry of a lever,
it does not move the score. Governing discipline: NO-FAKE (no fabricated number/citation), means/ends
firewall. Every number below is tagged MEASURED / DERIVED / CONJECTURE / HONEST-REFRAME.

Sisters: `deepmath_amortizing_argmax_paper_draft_20260704.md` (law #4 shearlet-N-term, law #8 MCF) ·
`capstone_witness_launch_config_deepmath_optimal_20260702.md` (mod-dim 19 already chosen) ·
`canonical_equations_refresh_design_plus_ldm_20260630T223000Z.md` (eq#5 Whitney) · code:
`src/tac/boundary_math/lever_b_levelset_generator.py` (`curvelet_directional_B`,
`stem_nyquist_max_freq_cycles_per_unit`) · `src/tac/boundary_math/lever_b_generator.py` (self-orient).

---

## 0. TL;DR — the exponent-vs-constant verdict (the headline)

> **Parametrizing the witness on its intrinsic ~8-dim manifold in the shearlet/curvelet basis is,
> HONESTLY, TWO different levers on TWO orthogonal DOF axes, with TWO different answers:**
>
> - **SPATIAL axis (per-frame partition = the 2-D cartoon):** the shearlet basis genuinely owns a
>   STEEPER approximation-error EXPONENT than isotropic Fourier — `ε²~N⁻²(logN)³` (curvelet) vs `N⁻¹`
>   (wavelet) vs `N⁻¹ᐟ²` (Fourier), a THEOREM (Candès–Donoho 2004), and it is *why* the `--self-orient`
>   lever measured −48%. **BUT the exponent only bites in the fine-scale (large-N) limit**; at our
>   operating bank size (N≈40–90 atoms) the `(logN)³` prefactor is order-60–180 (§1.4 arithmetic),
>   i.e. we sit *at or below the curvelet↔wavelet crossover-N* — so today's realized win is the
>   **CONSTANT** (the −48% is a single operating-point, not a measured slope), and the exponent is
>   an ASPIRATIONAL tail we have not yet reached.
> - **CONDITIONING/CODE axis (per-pair FiLM code = the ~8-manifold):** the effective DOF is the
>   intrinsic manifold dim `m≈8`, so the Whitney embedding `mod_dim = 2m+1 ≈ 17–19` is the
>   rate-optimal DOF; today's `--mod-dim 32` is pure **over-embedding waste** → this is a clean
>   **RATE-CONSTANT** win (fewer counted code DOF at equal d_seg), NOT an exponent change.
>
> **Net verdict: it is a CONSTANT lever now (rate + approximation), an EXPONENT lever only
> asymptotically — and the $0 GT-margin N-term log-log slope (§3.2) is the exact measurement that
> promotes "exponent" from conjecture to fact BEFORE any #205 dispatch.**

---

## 1. Effective-DOF → scaling-exponent theory (rigorous)

### 1.1 The two DOF axes are orthogonal (the key decomposition)

The witness (`experiments/train_levelset_witness_realized_through_R_mlx.py`;
`lever_b_levelset_generator.py`) factorizes into two DOF sources that scale *independently*:

| DOF axis | what it is | the target it approximates | the relevant scaling theory |
|---|---|---|---|
| **A. spatial** | curvelet/Fourier front-end `B` (bank size `N` atoms) → WIRE/HOSC trunk (`hidden_dim`, `n_hidden`) | the per-frame K-SDF field whose argmax = the 2-D **cartoon** partition | nonlinear N-term approximation of the cartoon class |
| **B. conditioning** | per-pair FiLM `code` (`mod_dim`) | the point on the ~8-dim lane-orbit manifold that selects THIS pair's geometry | Whitney embedding of the intrinsic manifold |

They are orthogonal because A sets *what shape is cheap to draw* (shared across all 600 pairs) and B
sets *which shape this pair is* (per-pair). The scaling law of the whole is the product; the intrinsic-
manifold reparametrization touches BOTH but with different mechanisms — §1.2 (A, exponent) and §1.5 (B,
rate).

### 1.2 The cartoon N-term rates (axis A — this IS a real exponent)

The scored object is a **cartoon**: piecewise-C² with C² edge curves (the frozen SegNet argmax is a
locally-curved Laguerre tessellation — paper §1, PROVEN-local). For the best **N-term** (nonlinear)
approximation of the cartoon class, the squared-L² error obeys (DeVore nonlinear-approximation theory;
Candès–Donoho 2004 curvelets — REAL, cited in the paper draft §2.4/law#4):

```
Fourier / adaptive plane waves :  ‖f − f_N‖²₂  ≍  N^(−1/2)          (edge → 1/k coeff decay)
orthonormal wavelets           :  ‖f − f_N‖²₂  ≍  N^(−1)
curvelets / shearlets          :  ‖f − f_N‖²₂  ≤  C · N^(−2) (log N)³
```

The **EXPONENT** (−1/2 → −1 → −2 in squared error) is a property of the *frame's ability to sparsify a
codim-1 edge singularity*: a curved edge is an anisotropic object, and only an anisotropic multi-scale
frame (parabolic `width ≈ length²` support) achieves the optimal `N⁻²` rate. **This is a theorem, not a
heuristic — it is the rigorous reason the directional basis is the right chart** (paper law#4:
`shearlet_nterm_upper_bounds_task_rate_v1`, PROVABLE, tightness CONJECTURED).

### 1.3 The bridge to d_seg (why the cartoon exponent is the RIGHT exponent for us)

d_seg is NOT the field L²-error; it is the **Hamming (symmetric-difference) rate of the argmax** = a
**boundary-DISPLACEMENT functional**. By the implicit-function theorem, a field error `e` near the
zero level set produces a normal boundary displacement `δ = e/|∇φ|`, and the symmetric-difference area
is `∫_∂ δ ds = ∫_∂ e/|∇φ| ds`. So:

```
d_seg  ∝  ∫_boundary ( field_error / |∇margin| ) ds     ← edge-localized, Fisher/margin-weighted L¹
```

Two consequences, both rigorous: (a) the relevant error is the **edge-localized** field error — exactly
the regime where curvelet ≫ wavelet ≫ Fourier is *largest* (curvelets put their whole budget at the
edge); (b) the weight `1/|∇margin|` is the Fisher/caustic metric (paper law#2, the measured 0.978
identity). So the **Fisher-weighted shearlet N-term count is a proven UPPER BOUND on the task rate
`R_X(D_Y)`** (paper law#4). The cartoon exponent IS the d_seg exponent — modulo tightness (CONJECTURE).

### 1.4 The honest finite-N caveat (why it is the CONSTANT today)

The exponent is asymptotic. Three honesty gates, in decreasing strength:

1. **The `(logN)³` prefactor swamps the exponent at our N.** Curvelet beats wavelet in *absolute*
   squared error only when `N⁻²(logN)³ < N⁻¹`, i.e. `N > (logN)³`. Arithmetic (natural log, hidden
   constant `C=1`): `N=50 → (ln50)³≈59.8 > 50` (curvelet NOT yet ahead); `N≈90–100` is the crossover;
   `N=256 → 170 < 256` (ahead). **Our default bank is N≈40–90 atoms (§2.2) — i.e. AT OR BELOW the
   crossover.** [DERIVED arithmetic on the theorem prefactor; `C` unknown → order-of-magnitude only,
   which is *exactly* why §3.2's measured slope, not this armchair estimate, is the authority.]
2. **The witness is a NONLINEAR INR, not a linear N-term expansion in a fixed frame.** The "effective
   DOF" is not the bank size `N` — it is the reachable-manifold / Jacobian rank at the solution (the
   MEASURED collapse: code rank-8, FiLM rank ~1.2-of-768 — R22). A WIRE/HOSC MLP on *isotropic* Fourier
   features is already a universal approximator of oriented edges; the curvelet front-end does not
   change *representability*, it changes the **cheapness** (fewer active atoms, better-conditioned
   gradient) → it shifts the d_seg-vs-params curve DOWN/LEFT (constant) far more reliably than it tilts
   the slope (exponent). The exponent is a property of the representable *class*; the network can
   already reach the class.
3. **−48% is a POINT, not a SLOPE.** The measured −48% all-class d_seg (canonical index D1; MLX-rs, n96)
   is ONE bank config vs the isotropic control — a single point on the d_seg-vs-N curve. **Claiming an
   exponent change requires d_seg swept vs N (bank size) and a log-log slope**, which has NOT been
   measured. Treating −48% as an exponent is a NO-FAKE overstatement; it is a constant-level win.
   [HONEST-REFRAME of MEASURED data.]

**Verdict (axis A):** the shearlet basis *genuinely owns* the steeper cartoon exponent (theorem), and
that exponent *is* the d_seg exponent (§1.3, modulo tightness), **but the realized-at-#205 win is the
CONSTANT + the conditioning**; the exponent becomes operative only as we push the bank toward the
stem-Nyquist fine scale (§2.2), and even there the `(logN)³` prefactor blunts it until N≫100.

### 1.5 The intrinsic-manifold DOF (axis B — a RATE constant, not an exponent)

The per-pair conditioning DOF should equal the **effective** DOF of the conditioning problem, which is
the intrinsic dimension of the lane-orbit manifold: **MEASURED `m≈8`** (AE-knee 8 / MLE 13 / TwoNN
upper ~11; canonical index R22, `dseg` index "Manifold ~8-dim NONLINEAR lane-orbit"). The Whitney
embedding theorem: a generic smooth `m`-manifold embeds injectively in `R^(2m+1)`; `2m` (weak) is the
boundary that risks self-intersection. So:

```
mod_dim*  =  2m + 1  =  2·8 + 1  =  17     (Whitney floor; band 17–19 for m∈{8,9})
mod_dim = 16 = 2m     →  UNDER-embeds (non-injective; the CLAUDE.md "mod-16 under-embeds" note)
mod_dim = 32 (default) →  OVER-embeds  →  ~13 wasted per-pair DOF = pure counted-rate waste
```

[DERIVED-from-MEASURED: Whitney theorem + measured `m≈8`; `whitney_mod_dim(9)=clip(2·9+1,19,26)=19`,
already the choice in `capstone_witness_launch_config_deepmath_optimal_20260702.md` §Per-flag.] This is
a **RATE-CONSTANT** lever: `mod_dim 32→19` is `(32−19)/32 ≈ 41%` fewer per-pair *code* DOF at equal
d_seg (rate is the binding sub-0.15 term). It does NOT change the d_seg exponent — it removes DOF that
were either overfitting or being pruned by the rank lever anyway. The `--code-spectral-entropy-weight`
+ `--film-stiefel` levers are the *soft* version of the same idea (induce the code onto the intrinsic
8-manifold); the mod-dim floor is the *hard* version (never allocate the wasted dims).

---

## 2. The concrete reparametrization (what's present, what's MISSING)

### 2.1 What is ALREADY a discrete shearlet (do not re-invent)

`curvelet_directional_B(cfg)` in `lever_b_levelset_generator.py` (verified by reading the source):
- **Parabolic ANGULAR sampling — PRESENT.** `l_j = n_orient0 · 2**(j//2)` (line 157) = the curvelet
  "more orientations at finer scales" law (angular width ~2^(−j/2)).
- **Multi-scale radial octaves — PRESENT.** `f_j = f0 · base**j`.
- **Wavefront-adaptive orientation — PRESENT.** `--self-orient` orients feats to the boundary tangent
  (`self_orientation_directional_feats`, byte-closeable via a decoder-reproducible partition; mean
  |cos| 0.89–0.91 to GT tangent, above the 0.85 bar — MEASURED, canonical index).

### 2.2 What is MISSING — the anisotropic parabolic SPATIAL support (the real gap)

The current atoms are **global plane waves**: `[sin(2π X@B), cos(2π X@B)]` (`curvelet_feats`, line
176) — a sinusoid over the *whole* image, with NO spatial window. A genuine curvelet/shearlet frame
element is a plane wave **times an anisotropic, scale-coupled spatial envelope** with
`support_across ≈ 2^(−j)`, `support_along ≈ 2^(−j/2)` (the parabolic `width ≈ length²` law). Today that
localization is delegated to the WIRE Gabor activation `cos(w0 u)·exp(−(s0 u)²)` — but WIRE's envelope
is **two global scalars** `--wire-w0 20 --wire-s0 10` (verified in argparse): it is **isotropic AND
single-scale**, NOT the per-scale anisotropic parabolic support. **So the "curvelet" front-end is, in
truth, an oriented plane-wave dictionary with the *angular* curvelet law but *without* the parabolic
spatial support** — which (a) leaves Gibbs/leakage on the trunk to fight and (b) forfeits the tight-
frame `N⁻²` sparsity constant that the theorem's parabolic scaling provides.

**The missing piece, concretely = an anisotropic-WIRE (parabolic-shearlet) front-end atom:**

```
ψ_{j,ℓ}(x) = cos( 2π f_j · (n̂_θ · x) ) · exp( − (u_across / w_j)² − (u_along / ℓ_j)² )
      with  f_j = f0·base^j ,  θ = πℓ/L_j ,  L_j = L0·2^(j//2)  (present) ,
      and the NEW parabolic envelope:  w_j ∝ 2^(−j)  (across) ,  ℓ_j ∝ 2^(−j/2)  (along) ,
      u_across = n̂_θ·(x−x_c) ,  u_along = t̂_θ·(x−x_c) .
```

This is the buildable delta: give each oriented atom a per-scale anisotropic Gaussian window (still a
deterministic function of the `(J, L0, f0, base)` scalars + the wavefront orientation ⇒ **still
byte-closeable, 0 archive bytes, no GT leak**). It is a NEW front-end module, not a flag flip (it
touches `curvelet_feats`/the MLX front-end), so it is a facet-2 *design proposal* gated by the §3.2
pre-metric, not an auto-launch.

### 2.3 The optimal bank size — the #223 Whitney/Nyquist numbers (DERIVED)

**Nyquist bank-freq (LEVER-2, in-code DERIVED).** `stem_nyquist_max_freq_cycles_per_unit(512,2) =
512/(4·2) = 64` cycles/unit (EfficientNet-B2 stride-2 stem → post-stem Nyquist). To span up to
`f_max=64` with `f0=2, base=2`: `2·2^j ≤ 64 ⇒ j ≤ 5 ⇒ n_scales = 6` (radial freqs {2,4,8,16,32,64}).
**Current default `--bank-n-scales 4` → max f = 16 = f_max/4 — only 2 of 6 available octaves used
(2 octaves of unused fine-scale headroom).** [DERIVED arithmetic.]

**Parabolic angular count.** With `n_scales=6, n_orient0=6`, the curvelet law gives orientations
`L_j = 6·2^(⌊j/2⌋) = {6,6,12,12,24,24}` → 84 oriented + `n_iso=4` = **88 atoms** (feature dim 176) at
full Nyquist, vs the default `n_scales=4` → `{6,6,12,12}=36 + 4 = 40 atoms` (feat 80). [DERIVED
arithmetic — the two laws jointly fix the bank size; whether the finer octaves PAY in d_seg is the
§3.2 measurement, NOT asserted here.]

**Whitney mod-dim.** `mod_dim = 2m+1 = 17–19` for `m≈8–9` (§1.5). [DERIVED-from-MEASURED.]

**Honest caution:** the in-code note (line 124) records that at the DEFAULT config the bank max-freq is
already 4× below Nyquist, so *capping* is a no-op there; the over-Nyquist waste today lives in the
`--self-orient` directional feats (`freq_across=32, n_dir_freqs=6` → up to 1024 cycles/unit, 16× over
Nyquist → aliases under R). So the Nyquist LEVER cuts two ways: **raise `n_scales` to use the budget in
the isotropic-ish curvelet bank, AND cap the self-orient directional freqs to `n_dir_freqs≤2` @
`freq_across=32` (or `freq_across=8, n_dir_freqs=4`) to stop aliasing.** [DERIVED, in-code Lever-2.]

---

## 3. The MLX-first lever + the $0 pre-metric

### 3.1 The near-term lever (real flags only — grepped, never invented)

Config-only, warm-start-safe VALUE changes on the existing bank (no new module) — all flags verified in
`train_levelset_witness_realized_through_R_mlx.py` argparse:
- `--bank-n-scales 6` (up from 4 — use the Nyquist radial budget) · keep `--bank-n-orient0 6` ·
  `--max-bank-freq 64` (cap at stem-Nyquist; drops aliasing atoms, shrinks `in_proj`).
- `--self-orient` (wavefront orientation, byte-closeable) with Nyquist-safe `--n-dir-freqs 2`
  `--freq-across 32 --freq-along 4` (per the in-code Nyquist note — NOT the default n_dir_freqs=6).
- `--mod-dim 19` (Whitney floor; **already the deepmath-optimal-config choice** — a RATE win vs 32).
- `--film-stiefel` + `--code-spectral-entropy-weight <β>` (soft-induce the code onto the 8-manifold).

NOTE: `--bank-n-scales`, `--bank-n-orient0`, `--mod-dim`, `--hidden-dim` are SHAPE-changing → they
force a FRESH arm (not warm-startable), per the canonical adaptive-stacking note. `--max-bank-freq`,
`--self-orient`, `--n-dir-freqs` are value/feature toggles. **CONTAINMENT: this is a config proposal for
the next fresh run — operator GO gates any dispatch; #205 SACRED READ-ONLY.**

The genuine build (facet-2 deliverable, gated by §3.2): the **anisotropic-WIRE parabolic-shearlet
front-end** (§2.2) as a new `front_end="shearlet"` option with a numpy reference + MLX forward +
argmax-parity test, byte-closeable from the same 5 scalars.

### 3.2 The $0 pre-metric — the decisive measurement (the tool, NOT the result)

**This is the measurement that promotes "exponent" from CONJECTURE to FACT with zero GPU, zero #205
contact.** Everything upstream is theory + arithmetic; this is real data on the real target.

Build `tools/pre_metric_nterm_basis_slope.py` (new, pure numpy/scipy, reuses
`all_class_boundary_proximity_and_tangent`, `curvelet_directional_B`, `isotropic_fourier_B`, and the
cached GT — NO witness, NO training):

1. Load `experiments/results/mlx_fleet_gt_cache/gt_n96.npz` → `lstars` (96,384,512) argmax +
   `margins` (96,384,512). Compute the signed all-class-boundary distance field (scipy EDT, already
   wired) = the GT margin/SDF the witness's SDF must match.
2. For each basis ∈ {isotropic Fourier `isotropic_fourier_B`, current plane-wave curvelet
   `curvelet_directional_B`, parabolic-WINDOWED shearlet (§2.2 envelope)}: compute the best **N-term
   linear** reconstruction (greedy/thresholded coefficients) of the GT margin field for
   `N ∈ {8,16,32,64,128,256}`.
3. Record TWO error curves per basis: (a) field squared-L² `ε²(N)`; (b) the **boundary-displacement /
   d_seg-proxy** error = symmetric-difference rate of `argmax` after reconstruction (the functional in
   §1.3 — this is the one that matters).
4. Fit the log–log SLOPE of each → the measured EXPONENT per basis, and the measured crossover-N.
   Aggregate over the 96 pairs (n96 lower bound; n600 if cheap — allergic-to-non-n600 for any *verdict*,
   but n96 is admissible as a $0 *pre-metric slope estimate* explicitly labeled provisional).

**Decision rule (pre-registered):** if the parabolic-shearlet log-log slope is *measurably steeper*
than isotropic Fourier on the **d_seg-proxy** curve AND the crossover-N is below the Nyquist bank size
(≈88), the exponent claim is REAL at our operating point → build the §2.2 module. If the slopes are
equal and only the intercept differs, the honest verdict stands: it is a CONSTANT lever → ship the
config-only §3.1 changes (Nyquist n_scales + Whitney mod-dim) and do NOT invest in the new module.
**The result is NOT asserted here — it is the deliverable of running the tool.** [Pre-metric SPEC;
result MEASURED-PENDING.]

---

## 4. The honest boundary — where this helps vs where it's neutral

| surface | facet-2 effect | tag |
|---|---|---|
| **d_seg approximation (constant)** | HELPS — oriented basis makes edges cheap → the −48% point win | MEASURED (point) |
| **d_seg exponent (slope)** | HELPS only ASYMPTOTICALLY (N→Nyquist); blunted by `(logN)³` at N≈40–90 | DERIVED + CONJECTURE (needs §3.2) |
| **RATE (counted bytes)** | HELPS — Whitney `mod_dim 32→19` ≈ 41% fewer per-pair code DOF at equal d_seg | DERIVED-from-MEASURED |
| **conditioning (Fisher metric / whitening / Stiefel)** | NEUTRAL — that is **FACET 1's** job; the basis change *improves conditioning as a side-effect* (oriented atoms → better Jacobian) but the dedicated lever is facet 1 | boundary |
| **pose / temporal screw** | NEUTRAL — `w_pose=0` this run; the se(3) screw is a separate facet | boundary |
| **the R round-trip / aliasing** | HELPS if paired with the Nyquist cap (`--max-bank-freq 64`, `n_dir_freqs≤2`) — else the fine octaves ALIAS and HURT | DERIVED (Lever-2) |

The cleanest one-line boundary: **facet 2 owns the two CONSTANTS (approximation + rate) and the
asymptotic exponent tail; facet 1 owns the conditioning that determines whether the finite-N fit
actually reaches that tail.** They compose: a steeper representable exponent is worthless if the
optimizer can't descend to it (facet 1), and perfect conditioning is worthless on a basis that can't
sparsify the edge (facet 2).

---

## 5. THE ONE CONTRIBUTED SYNTHESIS CLAIM

> **The witness's d_seg-vs-DOF law has TWO SEPARABLE exponents on TWO ORTHOGONAL DOF axes, and the
> intrinsic-manifold reparametrization is a CONSTANT lever on both today, an EXPONENT lever on only one
> and only asymptotically:** on the SPATIAL axis the shearlet basis genuinely *owns* the steeper cartoon
> N-term exponent (`N⁻²(logN)³` vs Fourier `N⁻¹ᐟ²`, a theorem, and the reason `--self-orient` measured
> −48%), but at our bank size N≈40–90 we sit *at or below the curvelet↔wavelet crossover-N ≈ (logN)³*, so
> the realized win is the *constant* (−48% is a point, not a slope) until we push the bank to the
> stem-Nyquist `f_max=64` (`n_scales 4→6`, ≈88 atoms) — where the parabolic **spatial** support
> (`width≈length²`), which the current global-plane-wave front-end LACKS, is the missing piece that would
> actually realize the exponent; and on the CONDITIONING axis the effective DOF is the *measured*
> intrinsic dim `m≈8`, so `mod_dim = 2m+1 ≈ 17–19` (Whitney) is the rate-optimal DOF and today's `32` is
> pure over-embedding waste — a clean rate-constant win, NOT an exponent change.
> **The $0 best-N-term log-log SLOPE on the cached GT `lstars`/`margins` (§3.2) is the single
> measurement that decides whether the exponent is real at our operating point BEFORE any #205 spend.**

Tag: SPATIAL-exponent-owned = **DERIVED (theorem)**; realized-as-constant-at-finite-N =
**HONEST-REFRAME of MEASURED −48% + DERIVED (logN)³ crossover arithmetic**; that-we-reach-the-exponent-
at-Nyquist = **CONJECTURE (the §3.2 pre-metric resolves it)**; Whitney `mod_dim≈17–19` rate-constant =
**DERIVED-from-MEASURED (`m≈8`)**; parabolic-spatial-support-is-the-missing-piece = **DERIVED (frame
theory) — code-verified absent**.

**Exponent-vs-constant verdict: CONSTANT now (rate + approximation), EXPONENT only asymptotically —
and the exponent is a MEASURED-PENDING claim, not a fact, until the $0 §3.2 slope is run.**

---

*MEANS, not ends. Pointer 0.19110 UNMOVED. No pointer moves until a byte-closed `upstream/evaluate.py`
n600 row < 0.19110. Everything above is the geometry of a lever.*
