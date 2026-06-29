# Witness config arbitrariness + dependency/synergy audit (v2 witness + GPU-run curriculum + θ* levers)

**UTC:** 20260629T224737Z · **Axis:** `[$0 CPU design-audit / advisory]` · **Pointer UNMOVED: contest-CPU 0.19110.**
**This is MEANS, not ends.** It de-risks the optimal-form GPU-run config before launch; it does NOT move the
exact score. The END is a byte-closed exact row < 0.19110. NO score claim here.

**Operator directive (2026-06-29):** *"do a pass for arbitrariness and config that should be solved or derived
or learned or swept and what is synergistic and might depend on others configs."* This is the optimal-form audit
BEFORE the v2 witness GPU run.

**SEAM:** this memo owns **CONFIG-CLASSIFICATION** (every knob → SOLVED / DERIVED / LEARNED / SWEPT /
ARBITRARY-CARGO-CULTED + dependencies + synergies), **including the make-compressible training configs treated AS
configs**. A sister agent owns the rate-attack **TECHNIQUE** playbook (FEED-kg/kh THREAD 2: which technique
attacks which witness component). Where a rate technique is a training-time config (entropy penalty, QAT, pose
codec bits) it appears HERE as a config; the technique-detail mapping is the sister's.

**Classification key:**
- **SOLVED** — closed-form / first-principles optimum (e.g. KKT waterline from the score arithmetic; R-survival
  render floor from the resample math). The value is *derived from an equation*, not chosen.
- **DERIVED** — from measured geometry / contest structure (e.g. directional basis from the boundary tangent
  field; margin=Fisher surrogate; w-seg=100 = the literal score weight). FORM is principled; magnitude may still
  be SWEPT.
- **LEARNED** — trained (FiLM codes, decoder weights, lane-survival residual).
- **SWEPT** — optimum genuinely unknown; needs grid/search/per-lever tune (optimal-form-before-dispatch binding).
- **ARBITRARY-CARGO-CULTED** — a bare default with no comment/derivation, OR a default that contradicts a
  documented solve/measurement. **These are the fix-targets.**

---

## 0. TOP-LINE FINDINGS (read first)

1. **The trainer on `main` is the LANE-SURVIVAL-RESIDUAL witness (v2 section S4 only).** The v2 deterministic
   store-canonical + per-class-pose-warp + integer-decode (S0–S3, S5; FEED-jb/F5 SDS-TSC design) is **build-gated
   DESIGN** (`--canonical-pose-warp/--warp-class-mask/--stratified-warp/--movables-residual` do NOT exist;
   confirmed both agents). So "the GPU run" = the S4 lane-residual run; the v2 6-section codec is not yet wired.
2. **The θ* TIER-2 additive levers (A1 adaptive-τ, A3 SWA/finisher-EMA, A6 nuclear-norm code penalty, A7
   junction-eikonal-relax) are NOT on `main`** — they live on worktree branch `6b4c0b962`
   (FEED-gm, subagent abc89d4fb), unmerged. The flags `--tau-anneal-shape`, `--ema-decay-finisher`,
   `--code-nuclear-weight`, `--eikonal-junction-relax` are absent from the main trainer. **They must be merged
   before the A/B campaign can run them.** (A9 scale-curriculum, A10 perspective-chart, A11 early-stop = TIER-3,
   never built.)
3. **Two MEASURED-decisive d_seg levers are OFF by default:** capacity-routing
   (`--margin-saliency-weight 0.0`, `--hardness-oversample 0.0`) and the Muon finisher (`--muon-start-epoch None`).
   These are the #1-after-basis lever (n96 −64% combined) and "the drop" (FEED-fk). The default config does NOT
   run the levers the whole crux points at.
4. **Defaults are off the SOLVED RD-optimum.** `--mod-dim 32` (comment claims "RD-optimum ~122KB") but FEED-fl/fq
   refined mod to the **SVD floor 21** and hidden to the **waterfilled ~120** (current `--hidden-dim 96`). The
   joint waterfill (FEED-fq) is closed-form; the GPU run must override h=120 / mod=21, not use the stale defaults.
5. **The effective SDF decision-band width is UNCONTROLLED** (no margin-band regularizer flag). R-survival
   physics (FEED-iw, MEASURED) requires lane ramp half-width ≳5px (slope ≲24/px) at render ≥320; the current band
   is an *emergent* product of `--softmax-temp-end 0.05` × `--eikonal-weight 0.01` × `--render-h 384`. This is the
   binding-class (Lane) survival knob and it is not a first-class config.
6. **`--l7-start-epoch` default = 800 (trainer) vs 900 (DSL baseline `curriculum_dsl.py:345` + the documented
   curriculum).** A real config inconsistency — pick one and reconcile.
7. **Uncapped basis frequency** (`--max-bank-freq None`): the curvelet bank emits up to ~1024 cycles/unit, **16×
   above the stride-2 stem-Nyquist (64)** — capacity spent on frequencies SegNet structurally cannot see, plus R
   aliasing risk. The agent flagged it; FEED-cu (stem-Nyquist free-bytes) says cap at 64.
8. **DM1 conditioning levers (`--film-stiefel`, `--code-spectral-entropy-weight`, `--film-rank-floor-*`) are
   DEEP-MATH-DEMOTED** (GR memory CORRECTION 2026-06-29, FEED-ip: DM1 is second-order — PR collapses 2.6× while
   d_seg improves 1.9×). They remain in the trainer (default-OFF) but should NOT be priority A/B arms; the real
   conditioning axis is DM2 (oriented basis, #7 above) + DM3′ (low-rank additive head, unbuilt).

---

## 1. CONFIG LEDGER

### 1A. THE LAW (the score function — fixed by the contest, NOT knobs)
| name | location | value | class | provenance / note |
|---|---|---|---|---|
| seg weight | `boundary_solver.py:80`, `margin_conditional_residual.py:58`, `dykstra_legal_frame.py:60`; trainer `--w-seg` (2070) | 100.0 | **SOLVED (fixed)** | literal score coefficient `100·d_seg`. HARD-EARNED. Duplicated in 4 files (consolidation hygiene, not arbitrariness). |
| pose ten | `boundary_solver.py:81` etc | 10.0 | **SOLVED (fixed)** | `√(10·d_pose)`. |
| rate coef | `boundary_solver.py:82` etc | 25.0 | **SOLVED (fixed)** | `25·bytes/N`. |
| contest total bytes (N) | 4 files | 37_545_489 | **SOLVED (fixed)** | the rate denominator (`evaluate.py:63`). |
| N scored/frame | `margin_conditional_residual.py:55`, `boundary_solver.py:84` | 196,608 = 384×512 | **SOLVED (fixed)** | scorer resolution. |

### 1B. CURRICULUM (CE → tau_softplus → l7_softplus → Muon)
| name | location | value | class | provenance / dependencies / synergies |
|---|---|---|---|---|
| epochs | trainer `--epochs` (2019) | 1500 | **SWEPT** | PR95-flavored budget. Coupled to stage boundaries (guard 2413-2420: `0<tau<l7≤epochs`). |
| anneal-epochs | `--anneal-epochs` (2021) | None→epochs | **DERIVED** | cosine denominator for tau/β/LR anneal, decoupled from epochs. **DEP: tau-schedule + hosc-β-anneal + LR-schedule ALL read this.** |
| tau-softplus-start-epoch | `--tau-softplus-start-epoch` (2145) | 300 | **SWEPT→should be A11 early-stop** | FEED-gn: tau productive 300→675, then DEAD 675→899 (~200ep waste). DEP: stage-transition fires here. |
| l7-start-epoch | `--l7-start-epoch` (2146) | **800** | **⚠ ARBITRARY (inconsistent)** | DSL baseline `curriculum_dsl.py:345` = **900**. RECONCILE. FEED-gn: l7 declares in ~50ep. |
| tau end value | `--softmax-temp-end` (2063) | 0.05 | **SWEPT** | argmax-hardness target; cosine `_softmax_temp_for_epoch` 658-674. SYNERGY: sets effective ramp band → R-survival (1E). |
| tau start value | `--softmax-temp-start` (2062) | 1.0 | **DERIVED** | soft start (gradients flow). FORM (cosine homotopy toward argmax) DERIVED; endpoints SWEPT. |
| tau-softplus-tau | `--tau-softplus-tau` (2147) | 0.3 | **SWEPT** | bare. |
| l7-mult / l7-threshold | `--l7-mult` (2148) / `--l7-threshold` (2149) | 4.0 / 1.0 | **SWEPT** | bare; the worst-pixel (L∞) finisher knobs. NB: literal softplus `p` lives in imported `make_loss_fn` (NOT exposed — see 0/§5). |
| seg-loss form | `--seg-loss` (2143) | "ce" | **DERIVED** | switched by `_seg_form_for_epoch` 625-632 at the boundaries above. |

### 1C. OPTIMIZER + EMA + stability
| name | location | value | class | provenance / dependencies |
|---|---|---|---|---|
| lr / lr-end | `--lr` (2064) / `--lr-end` (2065) | 1e-3 / 1e-4 | **SWEPT** | warmup→cosine 1799-1808. **Muon LRs default to 0.1× of these.** |
| weight-decay | `--weight-decay` (2066) | 1e-4 | **SWEPT** | Muon-wd defaults to this. |
| warmup-epochs | `--warmup-epochs` (2069) | 1 | **ARBITRARY-CARGO-CULTED** | bare. |
| ema-decay | `--ema-decay` (2067) | 0.997 | **DERIVED** | CLAUDE.md "EMA — Quantizr decay 0.997" non-negotiable (PR101 + Quantizr-0.33 anchor). HARD-EARNED. Deploy ships EMA shadow. |
| muon-start-epoch | `--muon-start-epoch` (2334) | **None (OFF)** | **DERIVED-but-OFF** | "the drop" (FEED-fk). Guard: must be `[1,epochs]`, WARN if `<l7-start`. **DEP: softmax-temp + hosc-β FROZEN at muon-start value (1771-93).** |
| muon-lr | `--muon-lr` (2338) | None → 0.1·lr = 1e-4 | **SWEPT (optimal-form-pending)** | PR95 0.1× finetune. CURRENT_STATE flags muon-lr ~1e-4 may be ~150-300× too small vs working ~0.03 → FEED-gl step-3 sweep mandate. |
| muon-momentum | `--muon-momentum` (2346) | 0.95 | **ARBITRARY-CARGO-CULTED** | bare. |
| muon-ns-steps | `--muon-ns-steps` (2349) | 5 | **DERIVED** | Keller-Jordan NS5 default (literature). |
| muon param-split | code 2330-33 | 2-D hidden→Muon; bias/code/heads→AdamW | **DERIVED** | PR95 "Muon on hidden only". |
| accum-pairs | `--accum-pairs` (2078) | 8 | **ARBITRARY-CARGO-CULTED** | bare batch knob. |
| grad-clip | `--grad-clip` (2079) | 1.0 | **DERIVED** | bounds the update (FEED-cw: clipped every step; the real spike control). |
| spike-factor | `--spike-factor` (2080) | 5.0 | **ARBITRARY-CARGO-CULTED** | skip-batch > factor·median. |
| num-pairs / verdict-pairs / eval-every | `--num-pairs` (2018) / `--verdict-pairs` (2081) / `--eval-every` (2028) | 24 / 24 / 25 | **SWEPT** | n600 is the real scale (defaults are smoke). verdict freq is the throughput lever (FEED-cw), not the MLP device. |

### 1D. ARCH / CAPACITY (RD-optimum)
| name | location | value | class | provenance / dependencies |
|---|---|---|---|---|
| hidden-dim | `--hidden-dim` (2056); `LevelSetConfig` `lever_b_levelset_generator.py:227` | 96 | **SOLVED-but-default-mismatch** | FEED-fq joint waterfill → **~120** (the cheap d_seg-productive knob). Default 96 ≠ solve. |
| mod-dim (FiLM code) | `--mod-dim` (2060); `:229` | 32 | **⚠ ARBITRARY (stale comment)** | flag comment claims "RD-optimum ~122KB"; FEED-fl/fq refined to **SVD floor 21** (32 is 1.5× over-wide). |
| n-hidden | `--n-hidden` (2057); `:228` | 4 | **SWEPT** | bare. |
| n-classes | hardcoded everywhere; `:230` | 5 | **SOLVED (fixed)** | contest scorer. |
| render-h / render-w | `--render-h` (2054) / `--render-w` (2055) | 384 / 512 | **SOLVED** | R-survival floor (FEED-iw: lane needs ≥320; 384=scorer res, safe). Comment notes 192 pre-caps d_seg. |
| FiLM code shape | `:967` | (2·num_pairs, mod_dim) | **LEARNED** | per-pair f0/f1 codes. |
| FiLM linear | `:969` | Linear(mod_dim, 2·hidden·n_hidden) | **LEARNED** | (scale,shift) per layer; identity-at-init (zero w+b, 365-382 = the "warm-up", not a flag). |

### 1E. BASIS / FRONT-END / ACTIVATION (the #1 measured d_seg lever)
| name | location | value | class | provenance / dependencies / synergies |
|---|---|---|---|---|
| front_end | `LevelSetConfig:236` | "curvelet" | **DERIVED** | oriented basis = the −48% d_seg lever (MEASURED). |
| self-orient | `--self-orient` (2109) | False | **DERIVED-but-OFF** | self-orientation directional feats; the directional orientation IS the boundary-tangent match (DERIVED from the 9.56:1 anisotropy). |
| freq-across / freq-along | `--freq-across` (2119) / `--freq-along` (2120) | 32.0 / 4.0 | **DERIVED (magnitude SWEPT)** | high-across / low-along = anisotropy-matched; ratio DERIVED, magnitudes SWEPT. |
| n-dir-freqs | `--n-dir-freqs` (2111) | 6 | **SWEPT** | `dir_w = 4·n_dir_freqs`. |
| reorient-every | `--reorient-every` (2112) | 50 | **ARBITRARY-CARGO-CULTED** | bare cadence. |
| **max-bank-freq** | `--max-bank-freq` (2106) | **None (uncapped)** | **⚠ ARBITRARY (wastes capacity)** | bank emits ~1024 cyc/unit, 16× over stem-Nyquist=64 (`lever_b_levelset_generator.py:107-130`). Cap at 64 (FEED-cu). |
| bank-n-scales / n-orient0 / f0 / base / n-iso | `--bank-*` (2097-2101); `:100-104` | 4 / 6 / 2.0 / 2.0 / 4 | **DERIVED (curvelet) / SWEPT (counts)** | parabolic curvelet geometry; the FORM is Daubechies-grounded, the counts SWEPT. |
| activation | `--activation` (2124); `:231` | "hosc" | **DERIVED** | step-native (argmax IS a step fn; hosc=tanh(β·sin), no Gibbs). |
| hosc-beta / hosc-omega | `--hosc-beta` (2127) / `--hosc-omega` (2139) | 4.0 / 1.0 | **⚠ SWEPT (optimal-form-pending)** | CURRENT_STATE: "hosc_β=4.0, ω=1.0 untuned" → verdict not load-bearing until tuned. |
| hosc-beta-end (anneal) | `--hosc-beta-end` (2135) | None | **DERIVED (β→∞ L∞ limit; schedule SWEPT)** | step-native sharpening; DEP: frozen at muon-start. |
| wire-w0 / wire-s0 | `--wire-w0` (2125) / `--wire-s0` (2126) | 20.0 / 10.0 | **SWEPT** | WIRE Gabor (only if activation=wire). |
| iso fallback (control) | `:237-238` | iso_n_fourier 48 / iso_sigma 8.0 | **DERIVED (control arm)** | isotropic baseline; the thing directional beats. |
| chroma | `--chroma` (2092) | True | **DERIVED** | operator "Chroma too"; SegNet reads RGB → chroma flips boundary. |
| palette-anchor | `--palette-anchor` (2093) | True | **DERIVED** | init palette to per-class mean GT RGB. |

### 1F. LOSS LEVERS / CAPACITY-ROUTING (residual-aware-CAPACITY; MEASURED #1-after-basis)
| name | location | value | class | provenance / synergies |
|---|---|---|---|---|
| margin-saliency-weight | `--margin-saliency-weight` (2247) | **0.0 (OFF)** | **DERIVED-but-OFF** | the capacity-routing/waterfill loss; **SYNERGY/ORDERING: pays ONLY after basis-match; on isotropic basis HURTS +6%** (MEASURED). Co-set with self-orient ON. |
| margin-saliency-tau | `--margin-saliency-tau` (2250) | 0.5 | **DERIVED** | `sal=exp(−gt_margin/τ)`; the annulus selector (margin<0.5 = ~91% of boundary px, `margin_polytope.py:109`). |
| margin-saliency-uniward / β | `--margin-saliency-uniward` (2260) / (2265) | False / 4.0 | **DERIVED (Yousfi UNIWARD) / SWEPT (β)** | down-weight textured regions (flips concentrate on smooth curves, FEED-cu). |
| hardness-oversample / power / band / source / weighted | `--hardness-*` (2279-2291) | 0.0 / 1.0 / 0.5 / "margin" / False | **DERIVED-but-OFF** | LEVER-5 per-pair "waterfill" (extra steps on hard pairs). band 0.5 = flip-prone margin. |
| lane-edge-weight / class / target | `--lane-edge-*` (2159-2166) | 0.0 / 1 / 0.5 | **DERIVED-but-OFF** | class=1 (Lane, comma10k canonical ✓); the binding-class hinge. |
| lane-thin-weight / class / radius / target | `--lane-thin-*` (2220-2232) | 0.0 / 1 / 4 / 0.5 | **DERIVED-but-OFF** | thin-lane density routing (the ~8-dim long-tail). |
| KKT waterline | `margin_conditional_residual.py:60-62` | **1.27 B/flip** | **SOLVED** | closed-form = SEG_VALUE_PER_FLIP(8.48e-7)·BYTES_PER_SCORE(N/25). The byte-close-time capacity water level (NOT a trainer flag). |
| margin/free split | `margin_polytope.py:62`, `boundary_solver.py:230` | quantile 0.5 / slack 0.5 | **DERIVED** | tight-boundary set = top1−top2<0.5. |

### 1G. LIVE REGULARIZERS (the variational PDE constraints)
| name | location | value | class | provenance / synergies |
|---|---|---|---|---|
| eikonal-weight | `--eikonal-weight` (2295) | 0.01 | **DERIVED form / ⚠ SWEPT magnitude** | |∇φ|=1 (eikonal HJ PDE); on decision margin `_eikonal_length_mlx` 474-497. **SYNERGY: with softmax-temp-end + render-res sets the effective ramp band (1E) → R-survival.** Magnitude 0.01 bare. |
| length-weight | `--length-weight` (2296) | 0.001 | **DERIVED form / ⚠ SWEPT magnitude** | Chan-Vese ∫ds geodesic-active-contour; smoothed-Heaviside eps=1.0 (`:343`). Magnitude bare. |
| boundary-band radius | `lever_b_levelset_generator.py:386,416` | 2 (px) | **DERIVED** | the 2px annulus (MEASURED 2.26% px / 96.3% flip-mass). |
| (effective ramp half-width) | EMERGENT — no flag | ~implied | **⚠ ARBITRARY (uncontrolled)** | FEED-iw: lane needs ≳5px / slope ≲24/px. No `--margin-band-width` regularizer exists → ADD or derive temp-end from it. |

### 1H. DM1 CONDITIONING (DEEP-MATH-DEMOTED — keep OFF; not priority arms)
| name | location | value | class | provenance |
|---|---|---|---|---|
| film-stiefel | `--film-stiefel` (2194) | False | **DERIVED-but-DEMOTED** | DM1a Stiefel WᵀW=I (byte-free isometry, PROVEN). FEED-ip: DM1 second-order, NOT binding d_seg. |
| code-spectral-entropy-weight | `--code-spectral-entropy-weight` (2201) | 0.0 | **DERIVED-but-DEMOTED** | DM1b −β·log PR(cov code) capacity penalty. |
| film-rank-floor-weight / target | `--film-rank-floor-*` (2183-2187) | 0.0 / 4.0 | **SWEPT-but-DEMOTED** | soft PR floor. |
| film-per-layer / film-concat-code | `--film-per-layer` (2175) / `--film-concat-code` (2179) | False / False | **SWEPT** | FiLM mechanism variants (the collapse-to-rank-1 cause; DM3′ bypasses BY CONSTRUCTION, unbuilt). |

### 1I. WITNESS VEHICLE / R-SURVIVAL / per-class warp / IPM (v2 design-stage + structured priors)
| name | location | value | class | provenance / note |
|---|---|---|---|---|
| SDF int8 quant | `lever_b_levelset_generator.py:695-698` | symmetric, /127 | **DERIVED** | byte-close quant. |
| uint8/bicubic R | `:377-382` | 255 affine, bicubic↑874 / round / bilinear↓512, align_corners=False | **SOLVED (fixed)** | contest-exact R; SDF survives BECAUSE R is interp-exact on the 1-Lipschitz ramp (FEED-iw). |
| camera fx/fy | `lane_sdf_component.py:66-67`; trainer comment 896/2375 `fx=910·512/1164=400.3` | 400.3 / 399.5 | **DERIVED** | EON intrinsics scaled to scorer res. |
| principal pt cx / **cy** | `lane_sdf_component.py:68` | 256.0 / **MISSING** | **⚠ ARBITRARY (cy unmodeled)** | no `_CY`; principal-point-y never modeled. |
| v_horizon | `lane_sdf_component.py:70` | **174** (comment: 188 IPM-optimal) | **⚠ ARBITRARY (stale)** | 188 noted optimal; 174 in use. |
| cam height | `lane_sdf_component.py:69` | 1.2 m | **DERIVED** | RAV4 mount. |
| dash-period search | `lane_sdf_component.py:182-208` | 3-25m / 64-grid / accept 0.25 | **DERIVED (ground-frame) / SWEPT (grid)** | dashes constant-length in world meters (FEED — the decisive ground-vs-image win). |
| centerline-deg / half-width | `lane_sdf_component.py:214,234,243-250` | deg 3 / 3.0px / pctl90 clip[0.5,20] | **DERIVED** | openpilot polynomial; lane SHAPE captured (false-neg d_seg 0.00046<target). |
| per-class warp (Road=homography / hood=identity / sky=KRK⁻¹) | **NOT IN CODE** (FEED-jb design) | n/a | **DERIVED-but-UNBUILT** | depth×rigidity gradient MEASURED (grok FEED-ja, Road+15%). v2 S2/S3 flags ungated/unbuilt. |
| pose sidecar | `scorer_targets.py:15,150-155` | 600×6 fp16 zlib-9 (<5KB) | **⚠ cargo-culted dtype** | fp16 default; FEED-db measured per-column bits [11,5,5,4,4,5]→2.3KB BETTER-than-fp16 (−0.0030 S). SWEEP (sister rate-attack owns technique). |
| low-rank pose codec #140 | NOT a literal (`dykstra.rank=24` is unrelated SVD trunc) | rank-2 planned | **DERIVED-but-UNBUILT** | rank-1 PoseNet structure (dim0=99.8% var, `pose_from_embedding.py`) → rank-2 codec DERIVED, ~2.7× over scalar. |
| structured-init | `--structured-init*` (2312-2325) | False; lr 5e-3 / steps 600 / clip 20 | **DERIVED-but-OFF / SWEPT(lr)** | pretrain φ to static-core partition (comment: 5e-3 converges, 8e-3 stalls). |
| lane-prior-phi1 | `--lane-prior-phi1*` (2380-2392) | False | **DERIVED-but-OFF** | init lane φ1 to openpilot deg-3 centerline SDF. |
| static-region clamp | `hood_static_component.py`, `road_horizon_component.py` | bottom/top_frac 0.25, vote 0.5 | **DERIVED (self-detecting)** | sky+hood 14-byte clamp frees 63% px (FEED a6ee15c5). Class self-detected, NOT hardcoded ✓. |
| seeds | `_LEVELSET_SEED 0`, `_FOURIER_SEED 0`, `_LUMA_FOURIER_SEED 7`, `--seed 0` | 0/0/**7**/0 | **⚠ ARBITRARY (inconsistent)** | luma seed=7 unexplained vs 0 elsewhere. Determinism hygiene. |
| gauge chart-per-component | `witness_dsl/gauge.py` `fix_gauge` | min-S rank over measured cost table | **SOLVED (mechanism)** | chart (SCREW vs PER_CLASS_HOMOGRAPHY vs LOW_RANK) selected by min S-contribution — arbitrariness already solved by ranking. `POLY_BASE_DSEG_FLOOR=0.00214`. |

### 1J. STAGE-TRANSITION TREATMENT + rate/byte-close
| name | location | value | class | provenance |
|---|---|---|---|---|
| stage-rewarmup epochs / floor / shape | `--stage-transition-rewarmup-*` (2357-2364) | 0 (OFF) / 0.1 / linear | **DERIVED-but-OFF** | "different stages need different treatment" (CLAUDE.md). Flush stale moments into new loss landscape. |
| stage-reset-moments | `--stage-transition-reset-moments` (2366) | False | **DERIVED-but-OFF** | rebuild AdamW m/v at boundary. |
| brotli quality | many (`:903`, contour/context codecs) | 11 | **DERIVED** | CLAUDE.md L32 (max compression, offline-free). |
| LZMA filters | `contour_codec.py:43` | preset 9-extreme, lc0/lp0/pb0 | **DERIVED** | small-alphabet label tuning (L24). |
| context codec template | `context_partition_codec.py:215` | "temporal" (5³=125 ctx) | **DERIVED** | spatial 5²=25 / temporal 5³=125. |
| (entropy-penalized loss / QAT-in-train / weight-entropy) | **NOT IN TRAINER** | n/a | **⚠ MISSING (training-time rate)** | FEED-kg THREAD 2: these are training-time "make-compressible" configs that MUST enter the GPU-run config or the rate gain is lost post-hoc. Sister owns technique; flagged here as missing configs. |

---

## 2. DEPENDENCY + SYNERGY GRAPH (which knobs are coupled)

**Dependency edges (A→B means B's value/effect depends on A):**
- `--anneal-epochs` → {tau schedule, hosc-β anneal, LR cosine}. One denominator drives three anneals.
- `--lr`, `--weight-decay` → {muon-lr, muon-adamw-lr, muon-wd} (all default to 0.1× / 1× of base).
- {tau-start, l7-start, muon-start} → stage-transition rewarmup/reset firing points (boundary-detected).
- `--muon-start-epoch` → freezes {softmax-temp, hosc-β, LR-schedule} at their muon-start value (1771-93). So
  **tau-end and Muon-start INTERACT**: Muon finishes at whatever τ the schedule reached by muon-start.
- guard: `muon-start ≥ l7-start` (WARN). `0 < tau-start < l7-start ≤ epochs`.
- KKT waterline 1.27 B/flip ← {seg weight 100, N, rate coef 25} (the SOLVED score arithmetic).
- per-class warp (v2) ← stored pose stream (FREE dual-use) ← EON intrinsics {fx,fy,cx,(cy),cam-h,v_horizon}.

**Synergy / co-set constraints (must be tuned TOGETHER, ordering matters):**
1. **basis-match PRIOR to capacity (ORDERING, MEASURED):** capacity-routing
   (`--margin-saliency-weight`, `--hardness-*`) on an ISOTROPIC basis HURTS +6%; on a DIRECTIONAL basis
   (`--self-orient` ON, `--max-bank-freq` capped) PAYS (n96 −64% combined). **Never enable capacity-routing
   without directional basis ON.** This is the single most important ordering constraint.
2. **effective ramp band = f(`--softmax-temp-end`, `--eikonal-weight`, `--render-h`):** these three JOINTLY set
   the decision-band half-width that must clear R-survival ≳5px / render≥320 (FEED-iw, binding Lane class). Tune
   as a TRIPLE, not independently. Currently uncontrolled (no margin-band regularizer).
3. **RD capacity pair (`--hidden-dim`, `--mod-dim`):** jointly waterfilled (FEED-fq) — mod pinned at SVD floor
   21 (per-frame, expensive), hidden waterfilled to ~120 (d_seg-productive, 1.72× cheaper/byte). Co-set h120/mod21.
4. **Muon + stage-re-treat (`--muon-start-epoch`, `--stage-transition-reset-moments`):** Muon needs fresh
   moments at the AdamW→Muon boundary (DSL `Muon` lever sets reset-moments=True together). Co-enable.
5. **tau adaptive-shape ⊗ SWA (A1 ⊗ A3):** A1 (slow late-τ anneal) may make A3 (finisher averaging) redundant
   (FEED-gk: "adaptive-τ may make SWA redundant"). A/B isolate before stacking.
6. **junction-eikonal-relax ⊗ sub-pixel (A7 ⊗ A8):** overlap (A8 RESOLVED already-optimal, FEED-gm; A7 down-
   weights |∇φ|=1 at triple junctions where SDF can't be smooth). A7 stands alone now.
7. **pose-decouple (`--w-pose 0.0`) ⊗ d_seg convergence:** decoupling deletes the MEASURED +0.70 seg-pose shared-
   decoder coupling (FEED-gi) → frees capacity for d_seg. Default already w-pose=0 ✓ (the synergy is realized).

**ASCII coupling sketch:**
```
            anneal-epochs ──► tau-sched ─┐
                          ──► hosc-β     ├─(frozen at)─► muon-start ──(needs)──► stage-reset-moments
        lr / wd ──(0.1×)──► muon-lr/wd ──┘                  │
                                                            └─► tau-end (interacts: Muon finishes at this τ)

  EON intrinsics{fx,fy,cx,(cy),cam-h,v_horizon} ──► per-class warp(v2) ──► pose stream(FREE dual-use)

  [self-orient ON + max-bank-freq≤64]  ──REQUIRED-BEFORE──►  [margin-saliency / hardness capacity-routing]
        (DIRECTIONAL BASIS = the gate; capacity on isotropic HURTS +6%)

  {softmax-temp-end ⊗ eikonal-weight ⊗ render-h}  ──set──►  effective ramp band  ──must clear──►  R-survival ≳5px (Lane)

  {hidden-dim ⊗ mod-dim}  ──joint waterfill──►  B* (RD-optimum h120/mod21)
```

---

## 3. RANKED "FIX-THE-ARBITRARINESS" LIST (highest exact-score leverage first)

Binding axis is **d_seg** (FEED-jb: rate has slack, pose solved). Rank by d_seg-leverage × confidence.

1. **Cap `--max-bank-freq` at stem-Nyquist 64 (SOLVED).** Currently None → ~1024 cyc/unit (16× over what the
   stride-2 stem sees). Wastes the #1 basis lever's capacity + R-aliasing risk. One-line set; FEED-cu grounds it.
2. **Enable capacity-routing WITH directional basis (DERIVED, ordering-gated).** Turn ON `--self-orient` +
   `--margin-saliency-weight`/`--hardness-*` TOGETHER (never capacity-without-basis: isotropic HURTS +6%). This is
   the MEASURED #1-after-basis lever (n96 −64%) and it is OFF by default.
3. **Control the SDF ramp band for Lane R-survival (currently UNCONTROLLED).** Add a margin-band regularizer OR
   derive `--softmax-temp-end` from the FEED-iw ≳5px / slope≲24/px target; co-tune with `--eikonal-weight` +
   `--render-h`(≥320 for lane). Binding-class survival knob.
4. **Set defaults to the SOLVED RD-optimum: `--hidden-dim 120`, `--mod-dim 21` (FEED-fq closed-form).** Stale
   defaults (96 / 32) and the stale "RD-optimum 122KB" comment on mod-dim are off the joint waterfill.
5. **Turn Muon ON + sweep `--muon-lr` (DERIVED-but-OFF; lr SWEPT).** muon-start=None disables "the drop";
   muon-lr=0.1·lr=1e-4 is flagged possibly ~100-300× too small (CURRENT_STATE). Co-set reset-moments=True
   (synergy 4). Per FEED-gl step-3 optimal-form mandate.
6. **Sweep the activation per-lever: `--hosc-beta`, `--hosc-omega` (SWEPT, optimal-form-pending).** 4.0/1.0 are
   untuned; CURRENT_STATE: no activation verdict is load-bearing until tuned to its own optimum.
7. **Reconcile `--l7-start-epoch` 800 (trainer) vs 900 (DSL/curriculum) + add A11 early-stop-on-plateau.** Fix the
   discrepancy; FEED-gn measured ~200ep dead tau-tail → early-stop saves wall-clock and starts l7 sooner.
8. **Merge the θ* TIER-2 levers off worktree `6b4c0b962` (A1 adaptive-τ / A3 SWA / A6 nuclear-norm / A7 junction-
   eikonal) to main (process/infra).** They are bit-identical-when-off; the A/B campaign cannot run them otherwise.
9. **Derive the SWEPT magnitudes of the live regularizers `--eikonal-weight 0.01` / `--length-weight 0.001`.**
   Forms are PDE-derived; magnitudes are bare. Sweep small, or tie to the ramp-band target (synergy 2).
10. **Pose-stream rate config: sweep per-column bit-alloc vs fp16 + wire low-rank codec #140 (rate, sister-owned
    technique).** FEED-db: [11,5,5,4,4,5] → 2.3KB, −0.0030 S free, BETTER-than-fp16. Cheap rate win.
11. **Reconcile EON intrinsics: set `_V_HORIZON` to the IPM-optimal 188 (or sweep) + model `_CY` (v2-warp-stage).**
    Affects Road/Lane ground-plane warp accuracy; relevant once S2/S3 are built.
12. **Determinism hygiene: unify seeds (`_LUMA_FOURIER_SEED 7`→0 or document).** Low score-leverage, but the
    deterministic-reproducibility non-negotiable wants seeds principled+consistent.
13. **Bare training-stability defaults (`--warmup-epochs 1`, `--muon-momentum 0.95`, `--accum-pairs 8`,
    `--spike-factor 5.0`, `--reorient-every 50`, `--hinge-weight 4.0`, `--tau-softplus-tau 0.3`):** low individual
    leverage; sweep opportunistically or accept with a one-line rationale to clear the cargo-culted tag.

**Keep OFF / DEPRIORITIZE (deep-math-settled):** DM1 levers (`--film-stiefel`, `--code-spectral-entropy-weight`,
`--film-rank-floor-*`) — FEED-ip demoted DM1 to second-order (PR collapses while d_seg improves). Not priority
A/B arms; the conditioning axis is DM2 (basis, #1-3) + DM3′ (low-rank additive head, unbuilt). A8 sub-pixel
RESOLVED already-optimal (FEED-gm) — no arm.

---

## 4. DAG-FEED SUMMARY

**FEED-kh ACTION (B): ARBITRARINESS/CONFIG audit — DONE ($0 CPU design, advisory; pointer 0.19110 UNMOVED).**
Classified every v2/GPU-run/lever knob across the trainer (116 args, `train_levelset_witness_realized_through_R_mlx.py:2017-2393`) + 14 boundary_math modules + the DSL, as SOLVED/DERIVED/LEARNED/SWEPT/ARBITRARY-CARGO-CULTED + a dependency+synergy graph. Memo `.omx/research/witness_config_arbitrariness_audit_20260629T224737Z.md`. Headline findings: **(1)** the trainer on main is the S4 lane-survival witness; the v2 6-section store-canonical+per-class-warp+integer-decode (S0-S3,S5) is build-gated DESIGN (unbuilt flags). **(2)** θ* TIER-2 levers (A1/A3/A6/A7) are UNMERGED (worktree 6b4c0b962). **(3)** two MEASURED-decisive d_seg levers are OFF by default — capacity-routing (margin-saliency/hardness=0.0) and Muon (start=None). **(4)** defaults off the SOLVED RD-optimum (mod-dim 32 vs solved 21, hidden 96 vs ~120). **(5)** the SDF decision-band width (binding Lane R-survival) is UNCONTROLLED (no margin-band regularizer; emergent from temp-end×eikonal×render). **(6)** `--l7-start-epoch` 800(trainer)≠900(DSL). **(7)** `--max-bank-freq None` = 16× over stem-Nyquist 64 (wasted basis). **(8)** DM1 levers deep-math-demoted → keep OFF. **Top ordering synergy (MEASURED, binding):** basis-match is PRIOR to capacity — capacity-routing on isotropic basis HURTS +6%, on directional PAYS; never co-enable capacity without `--self-orient` + capped bank freq. **Top fix:** the ranked list §3 — cap bank-freq, enable basis+capacity together, control the ramp band, set h120/mod21, Muon-on+lr-sweep, BEFORE the GPU run. MEANS≠ends; SEAM: config-classification (me) vs rate-attack technique playbook (sister FEED-kg/THREAD-2). Cross-refs: FEED-gj/gk/gm/gn (θ* levers), FEED-iw (R-survival/ramp), FEED-fl/fq (RD-optimum h/mod), FEED-ip (DM1 demote), FEED-jb (v2 6-section design), FEED-cu (stem-Nyquist), FEED-db (pose bits). pointer 0.19110.
