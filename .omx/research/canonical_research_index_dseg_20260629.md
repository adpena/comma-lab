# CANONICAL RESEARCH INDEX — d_seg LEVERS + FLOORS/CAPS slice (2026-06-29)

**Purpose (operator 2026-06-29):** marshal EVERYTHING we have MEASURED/BUILT/SOLVED on the **d_seg axis**
into one deduplicated, calibrated index so the next witness launch starts at TRUE optimal form with nothing
measured left on the table. Sister of the proactive-recall non-negotiable
([[proactive-recall-consult-own-research-before-concluding-20260630]]): consult THIS before concluding /
designing / killing on the d_seg axis. **$0 consolidation, no score moved — pointer UNMOVED contest-CPU 0.19110.**

## Calibration legend (every row carries one)
- **Authority:** `EXACT` = upstream/evaluate.py byte-closed (contest-CPU/CUDA) · `CPU-adv` = frozen CPU-torch
  SegNet argmax on cached `lstars` (advisory, NON-PROMOTABLE) · `MLX-rs` = `[macOS-MLX research-signal]`
  (advisory, NON-PROMOTABLE) · `DERIVED` = math/theory, not a measured row.
- **Axis:** `direct` = witness logits→argmax vs L* (the symbolic partition) · `through-R` = realized
  d_seg = argmax(SegNet(R(rendered RGB))) vs L*, R = bicubic↑384→874 → uint8-STE → bilinear↓512×384.
  **THE CRITICAL DISTINCTION: direct ≪ through-R (direct 0.0022 → realized 0.0064 on the palette store).**
- **n:** pairs scored (n6/n24/n48/n96/n120/n200/n600). Per FEED-kn, n CHANGES the outcome, not just the CI.
- Baselines used below (same-vehicle, MLX-rs n600 unless noted): lever-B control **0.008257**; SIREN
  n600 **0.003051**; FINER n600 **0.002915**.
- **THE NEED (the gates, from the S arithmetic):** bc20-standalone beats 0.19110 at d_seg ≤ **~0.00087**
  (pose-held); at the L13 72KB rate, d_seg < **9.2e-4 → ~0.162 (sub-0.19)**, < **3.2e-4 → ~0.110 (sub-0.15)**.
  Realized-axis gate (L13 0.0481 + stored-pose 0.0184 floor 0.0665): realized d_seg < **1.25e-3 → sub-0.19**,
  < **8.3e-4 → sub-0.15**.

---

## 1. INDEX TABLE — d_seg LOWERING LEVERS (deduplicated)

| # | Lever | Status | Magnitude / measured value (calibration) | Pointer |
|---|---|---|---|---|
| L1 | **ALL-CLASS DIRECTIONAL (anisotropic/curvelet) Fourier basis** — orient feats to the all-class boundary tangent field | **MEASURED — THE decisive lever, ~0 byte** | **−48% all-class** vs −8% lane-only (MLX-rs, direct, n96). n600: 0.008257→0.005697 (−31% alone). **CAVEAT (HIGH): circular — built from `gt.lstars`, NOT byte-closeable for a pure-RGB witness** → self-orientation fixed-point (iso→own argmax→tangent→dir) proposed but realized-axis UNVERIFIED; the RGB-witness gate fell back to ISOTROPIC. | FEED-25t/r DAG:629-641,633; boundary_routing.py 71bd86646; FEED-bi DAG:1118 |
| L2 | **Capacity-routing (KKT/hp-FEM waterfill on margin-saliency)** — fill DOF to the 0.72% lane band; `boundary_routing.BoundaryFiLM` | **MEASURED — dominant FLOOR lever, pays ONLY after basis-match** | basis+cap n600 best **0.002447 (−70%)**; n96 dir+cap −64%. **Capacity ALONE on isotropic basis HURTS +6%** → STRICT order basis-before-capacity. KKT: pour capacity until ∂d_seg/∂c(x) equalizes = reverse-waterfill (#157) on the saliency field. | FEED-25u DAG:643-648; FEED-25b/f/o/p DAG:479-599; boundary_routing.py (20 tests, lane frac 0.00639) |
| L3 | **Margin-hinge seg loss (lensA)** — `relu(margin_target − (z_GT − max_other))`, all gradient on flip set | **MEASURED — saturates the loss-reweight axis** | grad on confident flip = **1.0** (constant) vs soft-cosine **1.9e-22** (vanished) vs CE ~1.0 (but wastes grad on interior). Realized **−16–36% vs CE** (lensA). Annealed target 1.0→0.5. = the witness's PR74/PR62 hard-pixel routing, keyed on GT-margin. | accel1_margin_hinge_..._20260617.md; FEED DAG:1003; tac.losses.core.segnet_margin_hinge_per_pixel |
| L4 | **KD from frontier teacher (soft-logit, CE-anchor + light-KD-aux)** — kd_w=0.3, T=2.0, blend=1 | **MEASURED — real but small, genuinely-distinct lever** | c1 **0.002447→0.002423 (−1.0%)** clean monotone; **long900 → 0.002176 (−10.2%, did NOT saturate at 450ep)**. KD is NOT redundant with margin-hinge (adds dense distribution-match); pure-KD-only DIVERGES (0.0055). T=2>T=1 confirmed. | FEED-25w/ab DAG:686-696,748-753; kd_{a,b,c1,c2} JSONs |
| L5 | **Longer curriculum (900ep)** — same c1 config, extend budget | **MEASURED — the cheapest real floor-mover; already-running** | **0.002176** (long900 ep800), the best DIRECT-partition d_seg on record beating everything; ep450 0.002423→ep800 0.002176, still descending. Pairs with KD. | FEED-25ab DAG:748-753 |
| L6 | **SDF level-set witness + hosc (the level-set chart itself)** | **MEASURED — BEST witness chart on record** | **n96 hosc 0.00124 @ep950** (through-R-faithful surrogate), beats relu-cap direct 0.002447. transfer-probe: r_added_segnet ~9.5e-5 (R-survival TRANSFERS for SDF). Converged-n600 FEED-ey TARGET ~5.2–6.5e-4 at ≤120KB (projected, unmeasured). | DAG:2924; levelset_transfer_probe.json DAG:1688; FEED-ey/fh |
| L7 | **Step-native activation family (step_basis vs hosc vs FINER vs SIREN)** | **MEASURED — speed+small-bandwidth knob at capacity limit, NOT a floor lever** | FINER **−18.7% n100 → −4.5% n600** vs SIREN (capacity-regime decay). step_basis k=8/k16 healthy (K-knee −2–3% residual floor edge). **hosc β=4/8 FAILS standalone** (optimizer saturation: `d/dx tanh(β·sin)≈0 a.e.` → AdamW random-walk; β=8 strictly worse). step_basis (learnable slopes gₖ) is the stable carrier of the step-native paradigm. | FEED-25..h DAG:452-551; activation lens DAG:473 |
| L8 | **UNIWARD LEVER-4 (texture down-weight, Fridrich inverse-steg)** — `sal/=(1+β·tex)`, β=4.0; concentrate on smooth boundary | **BUILT — smoke-verified, convergence-time A/B DEFERRED** | `--margin-saliency-uniward --margin-saliency-uniward-beta 4.0`; src/tac/uniward_texture.py + uniward_delta.py. No converged d_seg A/B yet. Add as LATE-STAGE (l7/Muon) lever + synergy map. | CURRENT-STATE memory FEED-lf; combined_tier_1_wave_3_uniward_multi_scale_..._20260525.md |
| L9 | **Chroma (SegNet RGB-slack → argmax-flip lever; load-bearing for BOTH seg+pose)** | **OPEN — lives in legal-frame realization, UNMEASURED on realized axis** | SegNet argmax on RGB ⇒ chroma carries argmax-relevant signal at the codim-1 annulus; PoseNet reads YUV6 ⇒ luma-only carrier provably lossy (the pose-collapse 2.67–12.66 cause). Has NO surface in the direct-partition form; a TRAIN lever on a fresh witness (positive), though frontier decode-side chroma perturb HURT (trained optimum). Every pre-chroma verdict is PROVISIONAL. | FEED-25q DAG:601-606; antagonism map DAG:1195 |
| L10 | **Openpilot lane-prior φ1 / structured-init (road-plane SDF lane channel)** | **MEASURED (isolation) + BUILT — 0-byte train-time prior (rule-118 free)** | road-plane (inverse-homography) SDF lane-attributable **0.000439** vs image-coords **0.000858** (~2× worse, road-plane supplies CONTAINMENT). Separatrix residual 1.9e-5 (centerline IS the Road↔Lane separatrix). `--structured-init` ships 0 bytes (self-detect roles, NEVER luma-hardcode); ep0 win texture-dominated (no epoch-0 realized win, init absorbed by trained weights). | FEED-di DAG:1821-1845; lane_sdf_component.py (19 tests); structured_init FEED-ef DAG:2281-2294; FEED-fs |
| L11 | **Polynomial-fill / lane-geometry prior (#144/#145)** — deg-3 ground-frame centerline + width + dash | **MEASURED — captures lane SHAPE perfectly; residual = DASH** | recon false-NEG (shape) **0.00046 < target 0.00087**; false-POS (band fills dash gaps) 0.00396 = 90% of recon d_seg. Lane = ~35 floats/frame (centerline 2-3 + width 2 + dash period/phase 2, ~4.5 lines) → ~1-2KB counted, rate ~0.001. Ground-frame dashes = 2-param (const period in world-meters) vs image chirp. | FEED-di DAG:1829-1841; comma_openpilot_crossref_polynomial_geometry_20260619.md |
| L12 | **Ego-hood static-clamp (#139)** | **MEASURED — FREE 0-byte, negligible standalone** | ~19 flips in 25% of frame → clamp saves ~0 standalone; the VALUE is freeing capacity (static core, self-detect). MyCar class-4 IoU 0.994 static. | DAG:208,482,1145; hood_static_component.py |
| L13 | **Sub-pixel boundary placement (#149) / oriented 1-Lipschitz SDF ramp = area-coverage AA** | **MEASURED (advisory) — the binding R-survival cure** | the (C) wall = sub-pixel boundary COLOR-MIXING in eval bilinear downsample (boundary band 2.25–4.58% px, ~24% flip; flat-GT realized 0.00811, "texture barely helps; resize is the cause"). #149 measured 12× boundary-band collapse advisory. Cure = sub-pixel placement, NOT texture. ker(R)=80.67% scorer-invisible → control in R's 19.33% row-space ∩ small-margin annulus. **NOT yet built on the realized trainer.** | FEED-ku/kq DAG:6668; segnet_argmax_control_clues_20260629.md |
| L14 | **Round-trip-in-loop survival (R_surv) / train-through-R** | **MEASURED (decomposition) — targets the disjoint R_surv flip set** | every flip ∈ R_cap (high-res argmax already wrong = capacity, routing fixes) XOR R_surv (high-res correct, post-R aliased = survival, only round-trip/sub-pixel fixes). texture-survival wall **~16%** of boundary px flip regardless (measured SINE-only; step-native prediction UNVERIFIED). R_cap/R_total = routing's leverage ceiling (measure queued). | FEED-25c/f DAG:493-521; #149 |
| L15 | **NCA continuous-texture witness (#146)** | **MEASURED — AMBER, the strongest d_seg-core the generative campaign produced, training-fragile** | realized **0.00337 (1.31× frontier)**, interior_flip 0.0, **boundary_band_flip 0.079 (HALF the polynomial wall)**, rate 0.019, proj S 0.415. #143 flat-partition NCA = RED 0.0162 (fuzzier boundary). Dominated by SDF witness 0.00124. | FEED-fh DAG:3100-3119,3152-3166 |
| L16 | **Morse-Smale critical-structure / Kuramoto / coupled-oscillator partition** | **DEFERRED — fundamental alternative, measured-dominated** | Kuramoto ≡ #146's class (AMBER 0.00337), dominated by SDF 0.00124; binding open question = TRAINING STABILITY, not representation. REACTIVATE iff the SDF witness WALLS on the Gibbs/boundary-sharpness limit. | FEED-fh DAG:3030-3166; #180 |

---

## 2. d_seg FLOORS / CAPS (the hard numbers — don't re-derive, don't awfulize)

| Floor/cap | Value | Calibration | Meaning |
|---|---|---|---|
| **Label-noise confident-GT cap** | **ΔS ≈ 0.012** (margin≥0.5) | **EXACT contest-CPU advisory, n600** | 93.9% of flips at GT-margin <0.5 (median flip-margin 0.122 vs non-flip 5.89, ~48×). The seg axis is THIN — even a PERFECT confident-GT fool caps ~0.012 ΔS. **dseg_reducibility "IRREDUCIBLE" = OUR CURRENT decoder's flip set is label-noise, NOT an absolute axis floor.** |
| **d_seg AXIS capacity headroom** | reachable ~**0.00016–0.0003** | EXACT (frontier existence) | frontier vehicle achieves d_seg ~**0.0003 at 177KB** (BELOW the τ=0.137 label-noise proxy floor 0.00123) = existence-proof the axis is reducible 13× below our 0.0021. RECONCILES with the cap above: current decoder near its flip-floor; AXIS has headroom. |
| **384+uint8 pipeline floor** | ~**1.6e-4** | DERIVED/measured | ~11× below our 0.0021 → the resize/uint8 pipeline is NOT the wall. |
| **Deterministic-render floor (R1, k=0, store+warp, NO trained generator)** | ~**0.0185 bulk / 0.023 full** | CPU-adv, through-R, n96 bulk | the pure-deterministic materializer is d_seg-DEAD (~15–40× the budget) → the trained amortized-residual generator is REQUIRED for the d_seg floor. NO-FAKE: k=0 reproduces the ladder R1 floor exactly (0.01851==0.0185). |
| **Realized palette / exact-L* store through R** | **0.0064** (seg_term 0.64) | CPU-adv, through-R, n4-24 | the "d_seg=0" direct store realizes to 0.0064 (boundary band ~24% flip) = the realization gap that kills the pure-symbolic route. |
| **Best DIRECT witness d_seg** | **0.002176** (long900) / **0.00124** (SDF level-set hosc n96) | MLX-rs / CPU-adv, direct/surrogate | the record. SDF level-set 0.00124 is the best chart. |
| **Best byte-closeable RGB-witness d_seg** | **0.004445** (all-class-dir+cap) | MLX-rs, direct, n600 | the directional −48% lever, but at the ISOTROPIC fallback the byte-closeable ceiling is higher (the circular-GT caveat). |
| **Manifold / flip structure** | ~**8-dim NONLINEAR** lane-orbit (AE-knee 8 / MLE 13); **rank 53/60 full-rank LINEAR** | EXACT/measured | inherently compressible (low intrinsic dim) BUT a linear "store-the-flips" sidecar is NO-GO ×3 — compressibility is NONLINEAR (trained chart). |
| **Flip-mass distribution** | **50% class-0 Road / 19% class-1 Lane / 13% class-2 Undriv** | CPU-adv, n600 | binding residual = union of ALL inter-class edges (NOT just lanes) → orient directional capacity to ALL boundaries (the −48% vs −8% proof). Class order canonical [Road,Lane,Undriv,Movable,MyCar]. |
| **Seg-only best-case S** | **≈0.184** (still > 0.15) | DERIVED from caps | even a perfect seg lever caps at ~0.184 → **sub-0.15 REQUIRES the rate attack (0.118→~0.08), seg is necessary not sufficient.** |

---

## 3. OPTIMAL-CONFIG CONTRIBUTION — the measured-best d_seg lever STACK (each at its OWN optimum)

For a **from-scratch, openpilot-seeded, trained-THROUGH-R witness** (the [[session-...-CURRENT]] neural arm).
Ordered by the measured dependency structure (basis-before-capacity is STRICT). Each lever at its OWN measured
optimum; synergies/antagonisms noted.

**A. Representation / basis (do FIRST — basis-match is PRIOR to capacity):**
1. **SDF level-set chart** (L6) — the measured-best chart (n96 0.00124 ≪ relu-cap 0.002447); R-survival transfers.
2. **ALL-CLASS DIRECTIONAL Fourier basis** (L1, −48%) — oriented to the all-class tangent field. **Must be made byte-closeable** via the self-orientation fixed-point (iso→own-argmax→tangent→dir, 0-byte) — the circular-GT version is NOT shippable. Multi-scale octaves NEUTRAL (don't add >6).
3. **step_basis activation** (L7, learnable slopes gₖ) — the stable step-native carrier (NOT hosc β-fixed: saturates). FINER as the sine-family fallback baseline.

**B. Capacity (do SECOND — pays ONLY after basis-match; isotropic-alone HURTS +6%):**
4. **KKT capacity-routing / boundary_routing.BoundaryFiLM** (L2) keyed on the boundary-distance map → −70% combined. Do NOT scale total capacity blindly (bigcap overfits); route it.

**C. Loss (the saturated axis — use the right surrogate, don't double-stack):**
5. **margin-hinge seg loss** (L3, target anneal 1.0→0.5) = the d_seg-optimal surrogate (grad 1.0 on flips). This IS the witness's hard-pixel routing — do NOT add dynamic error-boost on top (L3⊗error-boost = redundant double-weight, 3-5× WORSE).
6. **KD soft-logit aux** (L4, CE-anchor kd_w=0.3 T=2.0) — the ONE genuinely-distinct loss add atop margin-hinge (+dense distribution-match; −1% short, −10% at long budget).
7. **UNIWARD texture down-weight** (L8, β=4.0) as a LATE-STAGE (l7/Muon) lever — concentrate gradient on the smooth boundary, off SegNet-blind texture.

**D. Priors (0-byte, rule-118 free — compile into inflate.py):**
8. **openpilot lane-prior φ1 / structured-init** (L10) — road-plane (inverse-homography) SDF, ~2× better containment than image-coords; separatrix residual 1.9e-5; ships 0 bytes.
9. **polynomial-fill lane geometry** (L11) — deg-3 ground-frame centerline captures shape (FN 0.00046 < target); residual = dash (2-param ground model). ~1-2KB counted.
10. **ego-hood static-clamp** (L12) + the static-core partition (free capacity for the boundary).

**E. Realization / R-survival (the binding wall — the OPEN frontier):**
11. **train-through-R in-loop** (L14) + **sub-pixel boundary placement** (L13, oriented 1-Lipschitz SDF ramp = area-coverage AA) — targets the R_surv flip set (~16% sine wall) the routing/capacity levers CANNOT touch.
12. **chroma active** (L9) — load-bearing for the legal-frame realization (SegNet RGB-slack) AND pose (YUV6); every pre-chroma verdict is provisional.

**Budget:** d_seg 0.508→ measured stack reaches **~0.00124 (SDF n96)** / **~0.002176 (long900 direct)**; converged-n600
TARGET ~5.2–6.5e-4. **Curriculum: skip the smooth_disagreement stage (RAISES d_seg, transient), de-weight rate-reg
stages, Muon-finish on d_seg** (the 3rd lever, ~+7% from one buggy stage). Per-stage treatment + reheat 0.1×/8ep.

---

## 4. OPEN / HEADROOM — top high-EV unmeasured/deferred d_seg items (the named path to ~0.001 → ~3e-4)

1. **[HIGHEST-EV] Realized-axis verdict for the full lever stack** — every big d_seg number above is DIRECT or
   MLX-rs; the binding wall (R-survival, the gate from 0.0024 direct → realized) is UNMEASURED for
   directional+step_basis+chroma+sub-pixel TOGETHER. The genuine open capstone problem.
2. **Sub-pixel boundary placement built on the realized trainer (L13)** — the #1 R-survival lever, measured 12×
   collapse advisory but NOT yet built into `train_*_through_R_mlx.py`. Tests whether step-native R_surv ≪ the
   sine 16% wall (the pre-registered prediction).
3. **Byte-closeable directional basis (self-orientation fixed-point)** — the −48% lever is circular-GT (not
   shippable for RGB); resolve the iso→own-argmax→tangent→dir fixed-point and RE-MEASURE realized −48%.
4. **Chroma as a TRAIN lever on a fresh witness (L9)** — never measured on the realized axis; load-bearing for
   both seg AND pose; provisional-flips every pre-chroma verdict.
5. **Converged-n600 SDF level-set** — the best chart (0.00124 @ n96) at n600 to the FEED-ey target 5.2–6.5e-4;
   pair with longer-curriculum (L5, the cheapest floor-mover) + KD.
6. **NCA residual-SHARPENER on the annulus (L15)** — reactivate IFF the SDF witness walls on the boundary band;
   witness-seeded + Lipschitz-normed + train-through-R (breaks the 0.079 boundary-band wall the from-scratch NCA could not).
7. **Curriculum-fix verification (MUONJUMP same-config test)** — do rate-reg stages 6-7 raise the FINAL floor that
   Muon could otherwise reach? (queued same-config test; ~31% hypothesis unverified).

---

## 5. CONFLICTS / SUPERSEDED — contradictory or retired d_seg findings + the LATEST verdict

| Claim | Latest verdict |
|---|---|
| "error-boost (hard-pixel dynamic) lowers d_seg" | **NEGATIVE (impl-level, NOT paradigm)** — eb3/eb9 both 3-5× WORSE; redundant with the margin-hinge weight (double-weighting destabilizes). The margin-hinge already IS the hard-pixel routing. Reactivate only if it REPLACES (not multiplies) the weight. (DAG:678-679) |
| "gauss/step-native activation lowers the floor" | **gauss NEGATIVE (diverges, div-by-zero, not optimal-form); hosc β-fixed FAILS (optimizer saturation).** step_basis (learnable slopes) carries the paradigm. Activation = speed+small bandwidth edge at capacity limit, NOT the floor lever — floor lever is capacity+routing. (DAG:473,526,654) |
| "multi-scale (more octaves) lowers d_seg" | **NEUTRAL** — 6→10 octaves ≈ 0.002447 (basis already matched at 6). Clean negative. (DAG:672) |
| "bigger capacity lowers d_seg" | **bigcap OVERFITS** (h160/mod64 destabilizes late); capacity-ALONE on isotropic HURTS +6%. Capacity pays only ROUTED, AFTER basis-match. (DAG:752,632) |
| "the smooth-disagreement curriculum stage helps" | **RAISES d_seg** (0.00396→0.00423), TRANSIENT (c1a recovers). Skip/repair it; CE+softplus+Muon LOWER, smooth+rate-reg RAISE. (DAG:533-539) |
| "store-the-flips linear sparse sidecar" | **NO-GO ×3** — residual full-rank in every linear basis (rank 53/60); compressibility is NONLINEAR (trained chart only). (DAG:585) |
| "d_seg is IRREDUCIBLE (our flips are label-noise)" vs "d_seg is CAPACITY-LIMITED (11–13× headroom)" | **RECONCILED (framing, not contradiction):** OUR CURRENT decoder's flip set is near its label-noise flip-floor; the d_seg AXIS is capacity-reachable to ~13× lower (frontier hits 0.0003). Lead both memos with the SAME composite verdict. (adversarial_review_all_results_20260623.md:66) |
| "0-byte decode-side levers (#139/#149/#169/PR98) give a free sub-0.19" | **NO 0-byte sub-0.19 row** — frontier decoder is trained-through-R → sits at a trained optimum → generic decode-side perturbations move AWAY (all measured WORSE, CPU-torch exact). #128/PR98/FECa/DQS1 ALREADY in 0.19110. Levers are TRAIN-time on a fresh witness, not decode-time on the frontier. (DAG:1145) |
| "lossless rate recode banks a free sub-0.19" | **EXHAUSTED** — frontier IS already the L21-L32/PR112-L30 recode (finishing-kit byte_delta=0); the FEED-ki estimate was double-counted (NO-FAKE catch). (CURRENT-STATE FEED-lb) |
| "free dual-use warp lowers BOTH d_seg and d_pose" | **REFUTED for the lossy arm** — d_seg and d_pose want OPPOSITE homography scales; pose stays on the STORED sidecar (d_pose ~3.4e-5), warp's role = residual predictor. (CURRENT-STATE FEED-lj) |
| "pose is a binding d_seg-competing lever" | **POSE RULED OUT as a lever** — d_pose descends FOR FREE with training (FiLM/low-rank/stored sidecar already solve it); d_seg is the SOLE binding controllable crux. (DAG:541) |

---

**NO-FAKE ledger:** every value cited carries its calibration (EXACT / CPU-adv / MLX-rs / DERIVED; direct vs
through-R; n). No score moved by this index — it is a MEANS (the marshaled toolbox); the END is a byte-closed
exact-eval row below 0.19110. Levers with no measured row are tagged OPEN, not MEASURED. Pointer UNMOVED **0.19110**.
