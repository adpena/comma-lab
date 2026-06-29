# Witness v2 conditioning redesign — DM2 (oriented byte-free basis) + DM3 (spatial conditioning)

`[research / deep-math / design / advisory]` · 2026-06-29T17:32:50Z · score_claim=false ·
promotion_eligible=false · pointer UNMOVED (the END is a byte-closed exact row below 0.19110).
Online + OSS authority exercised (standing deep-math rule). NO GPU launch — design + $0
numpy/CPU measurement only; the build + first GPU arm AWAIT operator steer, then a 3-clean
recursive (online+OSS) review.

> **TL;DR (the means→ends).** The $0 test the prompt asked for **falsifies the literal DM3
> premise** (a per-*position*, per-*pair* spatial latent grid) and in doing so **sharpens the
> binding axis**. The cross-pair SegNet-argmax-partition variation is **globally LOW-RANK**
> (rank-8 = 95.6%, rank-16 = 98.6% of variance — ego-motion coherence), so a per-pair spatial
> grid mis-allocates bytes: at ~60 floats/pair the best global per-pair code reaches annulus
> d_seg **0.026** while a (block OR LSQ-optimal) grid sits at **0.21 / 0.20** (~8× worse). The
> witness's measured FiLM rank-collapse is therefore a **conditioning-MECHANISM** failure, not
> evidence that a global code is too weak. **v2 = DM2 (maximize the free oriented basis) + DM3′
> (a per-pair LOW-RANK code over a SHARED, spatially-resolved oriented dictionary — the SVD
> structure made buildable), NOT a per-pair grid.** First GPU arm: DM2-max from scratch + the
> low-rank-additive code head. Byte cost of the counted per-pair payload stays tiny (~5–15 KB /
> 600, rate ≪ 0.01).

---

## 0. Ground: what v2 keeps, what it replaces

Read: `experiments/train_levelset_witness_realized_through_R_mlx.py` (the coord-INR forward, the
FiLM `_trunk`, the WIRE/HOSC `_act`, the curriculum), `src/tac/boundary_math/lever_b_levelset_generator.py`
(the witness module + numpy ONE-CODEPATH `levelset_rgb_forward_numpy`, the curvelet bank
`curvelet_directional_B`, the Nyquist cap `stem_nyquist_max_freq_cycles_per_unit`,
`quantize_levelset_blob`), `tools/measure_eikonal_sdf_dseg_recovery.py` (the eikonal-SDF rate-half),
`src/tac/boundary_math/lever_b_generator.py::self_orientation_directional_feats` (the byte-closeable
self-orient basis), `src/tac/witness_dsl/curriculum_dsl.py` (the `Lever` pattern).

**KEEP (the strengths):** the softmax-of-SDF level-set head (1-Lipschitz margin → R-survival);
the eikonal (0.01) + Chan-Vese length (0.001) live PDE regularizers; the pose-legal
palette+texture RGB; the stored-pose Quantizr sidecar (pose is solved); the
realized-through-R authority (frozen CPU-torch SegNet argmax verdict, NEVER MLX/MPS); the int8 +
brotli byte-close with the deterministic bank EXCLUDED (rule 118); the numpy ONE-CODEPATH.

**REPLACE / MAXIMIZE:** (DM2) the oriented byte-free basis — currently the default curvelet bank
(`n_scales=4, n_orient0=6, f0=2, base=2` → max 16 cycles/unit, **4× below the stem-Nyquist of 64**
→ headroom unused). (DM3) the GLOBAL per-(pair,frame) FiLM `code` (mod_dim=32) whose participation
ratio collapses to ~1.2 (measured) — the conditioning mechanism, not the byte allocation.

---

## 1. The decisive $0 finding (run, not asserted) — DM3-as-grid is FALSIFIED

**Tool (promoted, reusable):** `tools/measure_dm3_spatial_grid_vs_global_code.py`
(numpy/CPU, $0). **Data:** `experiments/results/mlx_fleet_gt_cache/gt_n96.npz` (`lstars` = the
real frozen-SegNet argmax, 96 pairs). **Method (closed-form, no INR training):** per pair, build
the ideal per-class SDF `phi*_p` via the REAL scipy EDT (`signed_distance_fields`; asserted
argmax(phi*)==L*, 0 px mismatch). Split `phi*_p = S + R_p` where `S = mean_p phi*` (what the
shared decoder + static-core init give for free) and `R_p` is the per-pair residual (the counted
payload's job). Reconstruct `R_p` two ways at MATCHED per-pair floats, add back `S`, argmax,
measure d_seg vs `L*` FULL and in the 2px ANNULUS (`d_seg_reference`, the canonical authority):

* **GLOBAL-CODE CEILING** = best rank-D factorization of `R` across pairs (truncated SVD on the
  96×96 pair Gram): `R_p ≈ code_p(D) @ Dict(D, N·K)`. This is the **strongest possible** global
  per-pair code of dim D (a free dense per-pixel dictionary; the witness FiLM is strictly *below*
  it). Per-pair latent = D floats.
* **SPATIAL GRID** = per-pair per-class block-mean (and a steelman **LSQ-optimal** projection) of
  `R_p` onto a (Gh,Gw) grid + bilinear up. Per-pair latent = Gh·Gw·K floats, integer-decodable.

**Pre-registered falsification:** if a coarse grid does NOT beat the global-code ceiling on the
annulus at matched bytes, the literal DM3 premise is wrong and the variation is globally low-rank.

### Result (n96, 10 s, `[macOS advisory/research-signal]`)

Floor (shared `S`, 0 per-pair bytes): full **0.0244**, annulus **0.2473**.
Eigenspectrum cumulative variance: rank-1 **0.642** · rank-2 **0.816** · rank-4 **0.897** ·
rank-8 **0.956** · rank-16 **0.986** · rank-32 **0.996**.

| method | floats/pair | d_seg full | d_seg **annulus** |
|---|---:|---:|---:|
| global-code ceiling D=4 | 4 | 0.0116 | 0.187 |
| global-code ceiling D=8 | 8 | 0.0104 | 0.173 |
| global-code ceiling D=16 | 16 | 0.0086 | 0.149 |
| global-code ceiling D=32 | 32 | 0.0059 | **0.103** |
| global-code ceiling D=64 | 64 | 0.0015 | **0.026** |
| grid 3×4 (block) | 60 | 0.0191 | 0.210 |
| grid 6×8 (block) | 240 | 0.0143 | 0.194 |
| grid 12×16 (block) | 960 | 0.0104 | 0.175 |
| grid 6×8 (LSQ-optimal) | 240 | 0.0130 | 0.200 |
| grid 12×16 (LSQ-optimal) | 960 | 0.0108 | 0.180 |

**Verdict: FALSIFIED, robustly.** At ~60 floats the global ceiling hits annulus **0.026**; the
grid (block OR LSQ-optimal) at 60–240 floats is **0.18–0.21** — ~7–8× worse. Even the
LSQ-optimal grid at **960** floats (0.180) loses to the global code at **16** floats (0.149). The
grid only out-performs a *rank-1-collapsed* FiLM, and only by spending ~100× the bytes a working
rank-8 global code (8 floats, annulus 0.173) uses.

### What this PROVES vs what it does NOT (honesty firewall)

* **PROVEN ($0, linear, pre-R):** the cross-pair partition variation is **globally low-rank** —
  ~8–16 coherent modes carry 96–99% of it. This is the signature of **ego-motion**: as the RAV4
  drives, the whole partition translates/scales coherently frame-to-frame, NOT
  spatially-independently. A per-*pair* per-*position* grid is the wrong byte allocation for a
  low-rank-global signal.
* **PROVEN:** the per-pair residual is concentrated in the annulus (band frac 0.056, but annulus
  d_seg 0.247 ≫ full 0.024) — the prompt's "10.5× cross-pair enrichment in the annulus" is
  reproduced — and a LOW-RANK GLOBAL code drives that annulus residual down efficiently.
* **NOT proven (necessary-not-sufficient caveats):** (a) pre-R — no uint8 round-trip aliasing;
  (b) ideal-EDT SDF targets, not the witness's learned SDFs; (c) the global "ceiling" uses a free
  dense dictionary no FiLM realizes (an upper bound); (d) the grid tested is a per-pair
  *independent* grid (a *shared* grid + per-pair code is a different, better object — it IS the
  global code). So this redirects the BUILD; it does not replace the EXACT realized-through-R
  verdict. It is consistent with the operator's DAG FEED-ip finding that **DM1 (rank penalty on
  the collapsed FiLM) does not help** — because the fix is a richer conditioning *mechanism*, not
  a rank penalty, and not a grid.

### Reconciliation with the operator axis verdict

The DAG FEED-ip verdict named **DM2 + DM3** binding and **DM1 wrong**. This $0 test refines, not
contradicts: **DM2 stays fully valid** (below); **DM3 is reframed** from "per-position per-pair
grid" → "per-pair LOW-RANK code over a SHARED spatially-resolved dictionary" (the SVD structure —
the spatial detail lives in the cheap/shared decoder, the per-pair payload stays low-rank). The
operator's spatial intuition is honored: the *dictionary* is spatially-resolved (per-position),
but it is **shared/amortized**, not a per-pair grid. **DM1 (rank penalty) is wrong** because
FiLM-scale/shift-on-a-shared-trunk cannot realize a high-effective-rank per-pair spatial
dictionary regardless of penalty — the cure is the mechanism in §3, not the penalty.

---

## 2. DM2 — maximize the oriented BYTE-FREE basis (the free, theory-backed lever)

The bank is deterministic → compiled into inflate.py → **0 counted bytes** (rule 118; it is the
measured −48% d_seg directional lever). Curvelet/shearlet theory (Candès–Donoho 2004 CPA;
Kutyniok–Lim arXiv:1106.1325) gives the optimal **‖f−f_m‖₂² ≤ C·m⁻²(log m)³** decay for a C²
object with a C² edge — vs wavelets m⁻¹, Fourier m⁻½ — via **parabolic scaling (width ≈ length²)**
and **orientations doubling every other scale (~2^{j/2})**. A curvelet's aspect ratio *equals* its
orientation count = 2^{j/2}; the measured annulus anisotropy **9.56 ⇒ 2^{j/2}≈9.56 ⇒ j≈6.5**,
which sizes the bank.

**Concrete v2 basis (all via EXISTING trainer flags + one anisotropic-envelope addition):**

1. **Angular coverage to match the anisotropy:** `--bank-n-orient0 16` (was 6). With the parabolic
   doubling `L_j = L0·2^{j//2}` already in `curvelet_directional_B`, the finest scale carries ≥16
   orientations over [0,π) (Δθ ≈ 11.25°). Theory: under-covering angle is the dominant error term
   and orientations are free → round UP to 16 (PROVEN rate theorem; INFERRED the exact 16).
2. **More scales, Nyquist-capped:** `--bank-n-scales 6`, `--bank-f0 2`, `--bank-base 2`,
   `--max-bank-freq 64` (= `stem_nyquist_max_freq_cycles_per_unit(512,2)`). The cap DROPS atoms
   above the EfficientNet-B2 stride-2 stem Nyquist (which only alias into off-boundary flips under
   R) AND shrinks `in_proj` (a counted param). Default bank is 4× under Nyquist → 6 scales × 16
   orientations stays legal and uses the headroom.
3. **Anisotropic WIRE envelope (the one code addition):** today `--activation hosc` (β-annealed
   step) or isotropic WIRE. Add an *oriented, anisotropic* Gabor atom: rotate to the local
   tangent frame and use `exp(−(s∥·u_t)² − (s⊥·u_n)²)·cos(ω₀·u_n)` with the oscillation ω₀ ACROSS
   the boundary (normal u_n, where the partition flips) and the envelope elongated ALONG it
   (s⊥/s∥ = 9.56). Recommended (PROVEN WIRE nominal ω₀=20, s₀=10; INFERRED the anisotropy split):
   **s⊥ ≈ 10, s∥ ≈ 1.0, ω₀ ≈ 20 (→ 30–40 at the finest scale)**. This makes each atom a true
   curvelet ridge (pool evidence along the edge, resolve the class transition across it). The
   tangent for the anisotropic envelope is byte-closeable via the existing
   `self_orientation_directional_feats` (cos 0.89–0.91 vs GT, > the 0.85 bar) — the witness's own
   argmax fixed point, regenerated at decode, 0 GT leak.
4. **Self-orient at Nyquist-legal config:** the current `--freq-across 32, --n-dir-freqs 6` runs to
   1024 cycles/unit (16× over Nyquist → aliasing). Use `--freq-across 8 --n-dir-freqs 4` (4
   octaves all ≤ 64) per the code's own note — finer angular coverage, fewer params, no aliasing.

**Why DM2 matters MORE given §1:** the low-rank finding means the per-pair payload is tiny, so the
spatial expressivity MUST live in the SHARED decoder — and the richest free shared spatial
dictionary is exactly the maximized oriented basis. DM2 is the substrate DM3′ rides on.

OSS adopted: WIRE (arXiv:2301.05187) — Gabor activation formula + ω₀=20/s₀=10; curvelet optimality
(Candès–Donoho CPA 2004) — parabolic scaling + m⁻² rate; shearlets (arXiv:1106.1325) —
shear-parametrized orientation is the *digitization-faithful* cousin of rotation (use shear if the
rotated-envelope grid sampling shows artifacts); AFPE (arXiv:2509.02488) — diagonal-anisotropy
Fourier features (we need the full-covariance/rotated generalization, an open slot — INFERRED).

---

## 3. DM3′ — the binding conditioning fix: low-rank per-pair code over a SHARED dictionary

The $0 test says the per-pair correction is `code_p(D) @ Dict(D, ·)` with D≈8–16 and a
**spatially-resolved SHARED** `Dict`. The witness's multiplicative FiLM (per-channel scale/shift
on a shared trunk) cannot express a per-pair *selection among spatial modes* — hence the PR
collapse. The fix gives the per-pair code authority over a shared spatial dictionary directly.

### DM3′-A (primary, minimal, buildable): per-pair LOW-RANK ADDITIVE SDF-correction head

Augment the LINEAR `out_sdf` with a per-pair additive term over a shared learned dictionary of
oriented-basis spatial modes:

```
phi_k(x, p) = out_sdf_k( trunk(feats(x)) )                         # shared cartoon (DM2 basis)
            + Σ_{d=1..D} code_{p,d} · G_{d,k}(x)                    # per-pair low-rank correction
G_{d,k}(x)  = (A · feats(x))_{d,k}      # A: shared (D·K, F) readout of the SAME free oriented basis
```

* The per-pair COUNTED payload is `code` ∈ ℝ^{(2·num_pairs)×D} (D≈12–16) — exactly the
  low-rank global code the SVD shows is optimal (rank-16 = 98.6% variance). `A` and the trunk are
  SHARED (counted once, amortized over 600). This realizes `code_p @ Dict` directly; FiLM's PR
  collapse is bypassed by construction (the correction rank is D, not the trunk's degenerate
  modulation rank).
* It composes with the SDF level-set head (the correction is in SDF space → the 1-Lipschitz
  margin / eikonal / length / R-survival machinery is unchanged) and with the pose-legal RGB.
* This is `mod_dim`-like but *additive over the free basis* rather than *multiplicative on the
  trunk* — the architecturally-minimal change that gives the code real spatial authority.

### DM3′-B (secondary, if A underfits the fixed structure): SHARED multi-res latent grid

Add ONE **shared-across-all-pairs** coarse latent grid (spatial-functa structure, but shared not
per-pair → amortized → ≈ free per frame) injected into the trunk to supply per-position structure
the coord-basis misses near the hood/horizon. Sample with **LIIF local-ensemble bilinear**
(arXiv:2012.09161 Eq. 4 — the 4-corner area-weighted blend; nearest-neighbor seams would
masquerade as false class boundaries in the argmax) and re-inject at EVERY trunk layer
(SPADE lesson, arXiv:1903.07291 — depth washes out an input-only code). Per-pair variation still
rides the §3-A low-rank code. Grid e.g. 24×16×16 (≈21 px cells) shared = 6 KB once ≈ 10 B/frame.

OSS adopted: Cool-Chic (arXiv:2212.05458 / 2403.11651) — the **integer-latent + range-coder**
half is the bit-exact counted statistic (see §4); the synthesis MLP is the free deterministic
generator. **But we do NOT import Cool-Chic's per-image multi-res latent pyramid as the per-pair
payload** — Cool-Chic codes INDEPENDENT images (spatially-distributed detail); our 600 pairs are a
COHERENT video (low-rank ego-motion), so the per-pair payload is a low-rank code, not a grid.
Spatial Functa (arXiv:2302.03130) — the global-FiLM→grid analysis + the perturbation argument
(a global code entangles all positions) explains the FiLM cap; we apply its SHARED-grid form in
DM3′-B. LIIF — local-ensemble sampling. Instant-NGP (arXiv:2201.05989) — deliberately NOT used
(the hash-grid's documented grainy-SDF collision artifact, §6 of that paper, is an unforced error
on an SDF output; at our tiny budget a dense small grid suffices).

---

## 4. Entropy model + byte budget + deterministic decode

**Counted payload = the per-pair `code` only** (DM2 bank free; trunk/`A`/shared-grid amortized
int8 weights). The code is **temporally smooth** (ego-motion → low-rank → adjacent pairs nearly
collinear), so:

1. **Quantize:** int8 symmetric per-tensor (the existing `_int8_symmetric` in
   `quantize_levelset_blob`) — the eval-roundtrip-for-weights so the verdict renders deploy
   weights.
2. **Decorrelate:** temporal-delta along the pair axis (PR95 L25 prefix-sum form) — a low-rank
   smooth code delta-codes to near-zero.
3. **Entropy-code:** brotli q=11 (deterministic, already in `quantize_levelset_blob`) as the
   default; OR a **range coder over a Laplacian/Gaussian model** (Cool-Chic uses `constriction`;
   Ballé E11) for the bit-exact integer path. Either is **integer-decodable → bit-identical**
   reproduction (the int8 codes are the integer statistic; the decoder math is the free generator,
   pinned to the numpy ONE-CODEPATH fp64-accumulation argmax-stable forward already in tree).

**Budget (n=600 pairs, 2 frames each):** D=16 code → 16×2×600 = 19,200 int8 = **18.75 KB raw**;
temporal-delta + brotli on a rank-16 smooth code → estimate **~4–10 KB**. rate term
`25·bytes/37,545,489`: 18.75 KB → 0.0125 (raw upper bound); ~7 KB → **~0.005**. Break-even is
3.995e-6 Δd_seg/byte, so every KB must buy ≥ ~4e-3 d_seg — the global-code ceiling (annulus
0.247→0.10 at D=32 ≈ Δfull 0.018 for ≤19 KB) clears it with margin. DM2 + shared-grid add ~0 and
~6 KB-once respectively. **Total counted ≪ the PR95 frontier rate; the witness is d_seg-bound, not
rate-bound** (consistent with the eikonal-SDF rate-half finding).

**Disk hygiene:** the $0 tool writes one small JSON to
`experiments/results/dm3_spatial_grid_val_run/` (gitignored, rebuildable). No bulk; no /tmp.

---

## 5. DSL lever shapes (mirror `curriculum_dsl.py::Lever`; real flags; default-off byte-identity)

`OrientedBasisMax` uses **only existing trainer flags** (validate()-passing today). The DM3′ levers
require NEW trainer flags (don't exist yet) → spec'd here as the BUILD contract; they MUST NOT be
added to the DSL until the trainer argparse has them (the DSL's `real_trainer_flags()` guard would
reject invented flags — the never-invent-flags non-negotiable).

```python
def OrientedBasisMax(n_orient0=16, n_scales=6, max_freq=64.0,          # noqa: N802 — DM2 (REAL flags)
                     freq_across=8.0, n_dir_freqs=4, self_orient=True,
                     window=300) -> Lever:
    """DM2: maximize the FREE oriented curvelet bank to the stem-Nyquist budget (16 orientations
    matching the 9.56 annulus anisotropy; 6 Nyquist-capped scales) + Nyquist-legal self-orient.
    Changes in_feat -> in_proj shape -> NOT warm-startable (FROM-SCRATCH arm; resume_from=None)."""
    ov = {"--bank-n-orient0": n_orient0, "--bank-n-scales": n_scales,
          "--max-bank-freq": max_freq, "--freq-across": freq_across, "--n-dir-freqs": n_dir_freqs}
    if self_orient:
        ov["--self-orient"] = True       # BooleanOptionalAction
    return Lever("DM2_oriented_basis_max", overrides=ov, epochs_delta=window,
                 notes="free oriented basis maxed to Nyquist (FROM-SCRATCH: in_proj shape changes)")

# DM3' (BUILD CONTRACT — needs NEW trainer flags before it can be a real DSL Lever):
def LowRankCodeDict(rank=16, window=300) -> "LeverSpec":   # DM3'-A — NEW flag --lowrank-code-dict-rank
    """Per-pair low-rank ADDITIVE SDF correction over the shared oriented basis (the SVD structure;
    bypasses FiLM PR collapse). Default-off (rank<=0 => byte-identical multiplicative-FiLM path)."""
    return {"--lowrank-code-dict-rank": rank}             # + --lowrank-code-dict (BooleanOptional)

def SharedLatentGrid(gh=24, gw=16, ch=16, window=300) -> "LeverSpec":  # DM3'-B — NEW flags
    """ONE shared (amortized) coarse latent grid, LIIF local-ensemble bilinear, re-injected per
    layer. Default-off (ch<=0 => byte-identical). NOT a per-pair grid (the $0-falsified form)."""
    return {"--shared-latent-grid": True, "--slg-gh": gh, "--slg-gw": gw, "--slg-ch": ch}
```

Default-off byte-identity is preserved exactly as the existing LEVER-A/B routes do (zero-init new
submodules; absent flags → the numpy ONE-CODEPATH auto-detects key absence → byte-identical). The
faithful A/B harness (commit 667ea0cbc / `tools/dm1_smoke_verdict.py`) tests `OrientedBasisMax`
today; DM3′ after the flags land.

---

## 6. Means→ends: predicted impact, what to build first

* **Predicted d_seg-long-tail impact (advisory):** the annulus floor (shared `S`) is 0.247; an
  achievable rank-16 global code targets annulus ~0.10–0.15 PRE-R (full ~0.006–0.009). DM2 lifts
  the SHARED dictionary so a lower rank reaches the same annulus, and (the part the $0 test can't
  see) DM2 + SDF + eikonal are the R-SURVIVAL machinery that keeps the annulus gain after the
  uint8 round-trip. Net target: realized-through-R d_seg toward the ~0.001–0.002 regime at the
  current tiny rate → S below 0.19110.
* **Byte cost:** counted per-pair code ~4–10 KB (rate ~0.003–0.007); DM2 free; shared grid ~6 KB
  once. Rate stays ≪ frontier — the witness is d_seg-bound.
* **Build FIRST (minimal first GPU arm, awaiting steer):** `OrientedBasisMax` (free, existing
  flags, FROM-SCRATCH) **+** DM3′-A `--lowrank-code-dict-rank 16` (the one new head). This is the
  smallest change that (a) uses the free basis headroom and (b) replaces the collapsing FiLM with
  the SVD-optimal low-rank-additive structure. Measure realized-through-R d_seg + the annulus
  decomposition; then byte-close (`tools/levelset_byte_close_and_eval.py`) for the exact row.
  DM3′-B (shared grid) is the second arm only if A underfits fixed hood/horizon structure.
* **Do NOT build:** a per-pair spatial latent grid (the $0-falsified form); a DM1 rank penalty
  (the verdict + §1 both say it's the wrong lever).

---

## 7. Honesty firewall (proven / inferred / speculation; NO-FAKE)

* **PROVEN (measured $0, `tools/measure_dm3_spatial_grid_vs_global_code.py`, n96, advisory):**
  cross-pair variation is globally low-rank (rank-8 95.6% / rank-16 98.6%); the per-pair grid
  (block + LSQ-optimal) is ~7–8× worse than the global-code ceiling on the annulus at matched
  bytes; the literal DM3 per-pair-grid premise is FALSIFIED. phi* round-trip asserted (0 px);
  d_seg via the canonical `d_seg_reference`.
* **PROVEN (literature, cited):** WIRE Gabor activation + ω₀=20/s₀=10 (arXiv:2301.05187);
  curvelet m⁻²(log m)³ optimality + parabolic scaling + 2^{j/2} orientations (Candès–Donoho CPA
  2004; shearlet arXiv:1106.1325); Cool-Chic integer-latent + range-coder bit-exactness
  (arXiv:2403.11651 §III-B) with the explicit caveat that the float synthesis MLP is NOT formalized
  as fixed-point (we pin it to our numpy ONE-CODEPATH); LIIF local-ensemble (arXiv:2012.09161
  Eq. 4); SPADE per-layer re-injection (arXiv:1903.07291); Spatial Functa global-vs-grid
  perturbation argument (arXiv:2302.03130); Instant-NGP grainy-SDF collision caveat
  (arXiv:2201.05989 §6).
* **INFERRED (theory-grounded extrapolation, not measured):** the exact basis numbers (16
  orientations, 6 scales, s⊥/s∥=9.56, the rotated-anisotropic WIRE envelope); the rank-16 target;
  the byte estimates (temporal-delta compression ratio); the predicted realized-through-R d_seg.
  All become MEASURED only on the GPU arm + byte-close.
* **SPECULATION (flagged):** that DM3′-A alone (no shared grid) suffices; that the full-covariance
  rotated curvelet-INR front-end (an open literature slot per AFPE) is the right anisotropic form.
* **Deterministic decode:** confirmed in principle — the int8 code is the integer counted
  statistic; brotli q=11 (or a range coder) is deterministic; the decoder forward is the numpy
  ONE-CODEPATH (fp64 accumulation, argmax-stable). Bit-identical reproduction holds as long as the
  new DM3′ heads are added to the ONE-CODEPATH (a build requirement, asserted in the byte-close
  parity test before any score claim).
* **NO-FAKE / borrowed-substrate:** BORROWED = curvelets/shearlets/WIRE/LIIF/SPADE/Cool-Chic/
  spatial-functa, the realized-through-R pipeline, frozen scorers + CPU authority. OURS-ORIGINAL =
  the $0 low-rank diagnosis that redirects DM3 from a per-pair grid to a low-rank-additive code
  over the shared oriented dictionary for the SegNet-argmax task-space witness. This is a DESIGN +
  $0 advisory row; the pointer is UNMOVED; the END is a byte-closed exact row below 0.19110.

**Next:** 3-clean recursive (online+OSS) review of this memo → then the minimal GPU arm
(`OrientedBasisMax` + DM3′-A) on operator steer → realized-through-R measure → byte-close → exact.
