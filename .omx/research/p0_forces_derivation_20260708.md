# P0 FORCES DERIVATION — the four in-trunk missing forces (task #360 phase 1)

**Author:** P0 FORCES DERIVATION agent · **Date:** 2026-07-08 · **Axis:** all numbers `[macOS-CPU/numpy advisory · NON-PROMOTABLE]` · **Pointer 0.19110 UNMOVED (means).**

**Scope / constraint.** This is the phase-1 DESIGN + DERIVATION deliverable. It contains: per-force derivation · measured constant(s) with provenance · exact loss-term formulation · DSL `Lever` spec (name/params/defaults/gating/telemetry) · pre-registered A/B acceptance criterion · the cross-cutting interaction matrix. **Phase 2 (the code wave) implements this spec mechanically** — it edits `experiments/train_levelset_witness_realized_through_R_mlx.py`, `src/tac/witness_dsl/*`, `src/tac/witness_autoconfig.py`, `src/tac/witness_control/*`. This memo touches NONE of those (v7.5 counter-force builder is concurrently in them). New standalone artifact: `tools/measure_delta_R_noise_floor.py` (+ `reports/delta_R_noise_floor.json`).

**Source ranking:** DAG FEED-missingforces / FEED-PA / FEED-v8risks (2026-07-08) · run-1 telemetry (`experiments/results/levelset_n600_crucible_v6_run1_20260708T095730Z`, read-only) · #333 annulus (`tac.witness_annulus_metrics`) · #141 margin (`tac.margin_saliency_map`) · #275 subpix (trainer L4559-4576) · #149 (`curve_core_gate_*`, `dseg_side_feasibility_*`) · `tac.lie` + `tac.boundary_math.warp_real_luma_frame0` · L67/L68/L4.

---

## 0. Measured substrate (the facts every force is calibrated against)

**run-1 (v6 crucible, birth-arm; best ep150):** d_seg 0.1286, d_pose 1.796, blob 87675 B, ep_loss 478. Loss terms at CE stage: `seg≈4.3` (dominant), `pose≈0.29–0.42`, `eikonal≈0.01`, `length≈1e-4`, `island_amplify≈0.6`, `persistence≈0.35–0.66`, `weight_entropy≈0.87`, total≈6.7. Default-off (value 0.0 all epochs): `boundary_distance, lane_edge, margin_saliency, subpix, chroma_boundary, thin_lane, margin_field_head, code_nuclear, rankfloor, code_spectral`.

**Annulus (#333, band=2.0 GT-margin):** area_frac 0.0483; ep150 annulus_flip_frac 0.557, interior_flip_frac 0.107, annulus_flip_mass_share 0.209. GT annulus margin **p10=0.172, p50=0.897**. NOTE: run-1 is the *birth-arm over-paint regime* (Road interior theft, FEED-roadfloor) → its 0.209 annulus-mass-share is NOT the converged number.

**Converged clean regime (FEED-PA, archive bf1ee1fa8, n600, reproduces 0.000910@384 exactly):** every class flips ONLY at its Road separatrix, **ZERO interior flips**; **100% of the achievable floor is BOUNDARY PLACEMENT**. Flip-mass shares: Road 43.7% / Lane 16.3% / Undriv 18.2% / Movable 10.4% / MyCar 11.4%. Within-class flip: Road 0.17% / Lane 2.5% / Undriv 0.03% / Movable 0.76% / MyCar 0.04%. Highest-leverage ~0-byte lever: **Road↔Lane tie calibration (41% of Road's flips)**.

**#149 (measured, exact R):** eval R = raw uint8 camera-res 874 → **bilinear-down→384 ONLY** (modules.py:113); the bicubic-UP to 874 is the *decoder's* choice for a sub-camera render, not part of D. 3-way decomposition: resize +0.00005 (0.9%) · flat-colour interior-texture **+0.00562 (99%) = the wall** · camera-res sub-pixel placement collapses the 1px-band flip **12× (0.176→0.022)** and (grad through D+SegNet, 1200 it) reaches d_seg **0.00094 = 1.7× frontier** — the boundary-placement floor.

**L67:** #205 CE-floor d_seg 0.00496; the CE-residual is temporal flicker; **44% of CE-residual spikes = temporal flicker, LANE-dominated.**

**The R chain (contest-exact, `apply_contest_faithful_roundtrip_nhwc`):** render(base) → bicubic↑ CAMERA(874×1164) → **uint8 STE @ CAMERA** (`clip(0,255); round; clip + stopgrad(round−clip)`) → bilinear↓ SEG(384×512, float) → SegNet. The uint8 round at camera res is the ONLY non-smooth, uncontrollable step.

---

## FORCE 1 — TEMPORAL SCREW-CONSISTENCY

### 1.1 Derivation
Each scored pair is `(f0=t, f1=t+1)`. The witness renders BOTH frames; **SegNet reads ONLY f1** (`x=x[:,-1]`, modules.py:108 — f0 is *seg-free*, confirmed `warp_real_luma_frame0` docstring). The per-class witness field φ_c (logits / the `_signed` GT-class margin) therefore exists for both frames but is only *scored* on f1. Under ego-motion, the ground-plane scene classes (Road, Lane, Undrivable) transform by a **plane-induced homography** `H(ξ)=K(R−t·nᵀ/d)K⁻¹` where ξ∈se(3) is the ego screw. So the temporally-consistent constraint is:

  **φ(x, f1) ≈ Warp_ξ( φ(·, f0) )(x)**  on the ground-plane classes.

The 44%-flicker residual (L67, lane-dominated) is exactly the failure of this constraint: the witness partition jitters frame-to-frame instead of moving rigidly with ξ. Enforcing it kills that residual fraction and is zero-byte (ξ already stored for pose; the warp is a generic algorithm, rule-118 free).

### 1.2 Loss-term formulation
```
L_temp = w_t · Σ_{x∈annulus, c∈GROUND}  m_ann(x) · || φ_c(x, f1) − Warp_ξ(φ_c(·, f0))(x) ||²  / |annulus|
```
- **φ_c**: witness per-class softmax prob (5-vec) at 384×512, the through-R realized field (so the term is consistent with what SegNet scores). Prob-space (not logit) so the scale is bounded [0,1] and comparable across classes.
- **Warp_ξ**: `warp_real_luma_frame0`'s MLX differentiable homography-warp path (`.mlx()`, bit-checked vs the numpy oracle). Warps the f0 field forward by ξ into the f1 frame. Bilinear grid-sample → differentiable w.r.t. the field always.
- **GROUND = {0 Road, 1 Lane, 2 Undrivable}** ONLY. Movable(3)/MyCar(4) are NOT ground-plane → the homography is wrong for them → **masked out** (else L_temp injects a systematically wrong target on moving objects). This matches the lane-dominated flicker signature.
- **m_ann**: the #333 annulus mask on f1 (`|GT margin| < band`, band=2.0), a θ-independent constant (stop-grad weight field) — restrict to the boundary band where d_seg lives (converges with Force 2: don't spend gradient on stable interiors).

### 1.3 ξ source + honest pose coupling (per L68)
**Two arms; SAFE default is stop-grad GT-ξ.**
- **DEFAULT (`ground_gt`, stop-grad ξ):** ξ = per-pair GT screw from the cached `gt_poses` via `xi_from_pose_calibration` (the same calibration the pose carrier uses). ξ is a FIXED, correct warp; gradient flows ONLY to the field φ. L_temp is then a **pure seg-consistency regularizer with a correct warp — ZERO coupling to the (open) pose facet.** This is the confound-safe cold start.
- **DUAL-USE ARM (`carrier_live`, grad-to-ξ):** ξ = the pose carrier's LIVE co-adapted twist, gradient flows to ξ. L_temp then *teaches ξ a seg-consistency signal* — the "seg face" of the unified screw (the unification's claim: the ξ that aligns the partition IS the ego motion that minimizes d_pose). **HONEST COUPLING (L68):** pose is OPEN on the witness (warp d_pose 3.7–10.3, NOT solved; 3.4e-5 is the borrowed ancestor). The seg-derived ξ-gradient and the d_pose-derived ξ-gradient *should* agree if geometry is consistent, but that is an EMPIRICAL claim, not proven. The dual-use arm therefore carries a **d_pose tripwire**: if the verdict d_pose rises > X% over the arm's window, revert to `ground_gt`. Do NOT default to dual-use — it bets the (fragile) pose optimum on the unification holding at the implementation level.

### 1.4 w_t calibration arithmetic (show the work)
**Goal (prompt):** w_t so the term's gradient share is comparable to the seg term *at the measured flicker amplitude*.

**Flicker amplitude f_t = 0.44** (L67: 44% of the CE-residual spikes are temporal flicker). Corroboration from run-1 per-class annulus flip-frac: Lane is the unstable orbit (IoU 0.263, CLAUDE.md; ep150 Lane annulus_flip_frac 0.038 on tiny 0.59%-area support → high *relative* instability) — consistent with L67's lane-dominated flicker.

**The target:** give L_temp a gradient share ≈ f_t of the seg term's annulus gradient, so it can actually fix the flicker fraction rather than be drowned by CE:
```
||∇L_temp||  ≈  f_t · ||∇L_seg||_annulus   with f_t = 0.44
```
**Why we cannot set w_t in closed form now:** the trainer logs per-term VALUES + a GLOBAL gnorm, but NOT per-term gnorm. So the rigorous weight is a one-time stage-boundary measurement, not an a-priori constant. The honest procedure:
1. **Cold-start `w_t = 0.1`** (confound-safe). Order-of-magnitude check: raw L_temp (prob-space MSE on the annulus) ≈ (flickering annulus fraction ≈ f_t·annulus_flip_frac ≈ 0.44·0.56 ≈ 0.25) × (per-flicker-px ‖Δprob‖² ≈ 0.3) ≈ **0.075**. Then w_t·L_temp ≈ 0.0075 = **0.1% of total loss 6.7** → far under the 40% `term_domination` alarm (L4) → safe to introduce without destabilizing.
2. **Duty-to-measure:** phase-2 emits per-term gnorm at each verdict (cheap — the per-term grads are already computed inside `value_and_grad`; log `‖∂L_temp/∂θ‖` and `‖∂L_seg/∂θ‖`).
3. **Ramp `w_t` at STAGE BOUNDARIES ONLY** toward `w_t* = w_t · (0.44·‖∇L_seg‖_annulus / ‖∇L_temp‖)` measured. **Never per-step** (the GradNorm-canary warning, L4/L5: per-step adaptive weighting muted the confound canary). Expected band: w_t ∈ [0.1 cold, ~1–10 target] — the order the arithmetic implies (0.075 raw vs a ~0.66 target contribution ⇒ w_t≈9 upper); the measurement pins it, the cap (≤~15% loss share) bounds it.

### 1.5 DSL Lever spec
```
Lever name:  temporal_screw_consistency        (NEW factory)
compiles argv: --seg-temporal-screw-weight <w_t=0.0 default-OFF>
               --seg-temporal-screw-start-epoch <l7_start>     # partition must be formed first
               --seg-temporal-screw-xi-source {ground_gt|carrier_live}   default ground_gt
               --seg-temporal-screw-classes 0,1,2               # GROUND only
               --seg-temporal-screw-band 2.0                    # annulus GT-margin band
params/defaults: w_t=0.1 (when activated), start=l7_start, xi_source=ground_gt,
                 classes={0,1,2}, band=2.0
gating: default-OFF (score-affecting lever, registered + duty-to-measure per L31).
        start_epoch ≥ l7 (needs a formed partition to warp).
        carrier_live arm gated on a d_pose tripwire.
telemetry rows: {stage:"temporal_screw", epoch, w_t, xi_source, classes,
                 raw_L_temp, gnorm_L_temp, gnorm_L_seg, gnorm_ratio,
                 flicker_annulus_frac, d_pose_guard (carrier_live only)}
```
`.validate()`: xi_source∈{ground_gt,carrier_live}; classes⊆{0,1,2}; start≥l7; fail-loud if carrier_live without a live pose carrier.

### 1.6 Pre-registered A/B acceptance
Arm = `ground_gt`, w_t ramped to gradient-share ≈0.44, start=l7, vs the v7.5 baseline. **ACCEPT if:** verdict d_seg drops by ≥ (0.44 × annulus flicker mass) at matched epoch AND the per-class annulus_flip_frac for Lane(1) drops (the flicker is lane-dominated) AND d_pose does NOT rise. **KILL-verdict scope = formulation** (this L_temp composition), NOT the temporal-consistency paradigm.

---

## FORCE 2 — MARGIN-BAND SATISFICING (hinge at R-survival δ_R)

### 2.1 Measured constant δ_R (provenance)
**δ_R = 0.0196** — measured by `tools/measure_delta_R_noise_floor.py` (n96 GT frames, `gt_n96.npz`, frozen CPU-torch SegNet authority, `reports/delta_R_noise_floor.json`).
- **Definition:** p95 of `|margin(uint8-round(x_c)) − margin(x_c)|` over annulus pixels (`|GT margin|<1.0`), where `x_c = bicubic(bilinear(gt_f1→384)→874)` is a witness-reachable *continuous* camera frame and margin = top1−top2 logit. This isolates the **uint8-at-camera** perturbation — the one uncontrollable step in R.
- **Quantiles (uint8-isolation, annulus):** mean 0.0067, p50 0.0046, p90 0.015, **p95 0.0196**, p99 0.0317, max 0.086. Stable across n (n24 p95=0.0179, n96 p95=0.0196).
- **Cross-check (full-R vs GT-direct, annulus):** p95 0.0371 (includes bicubic up/down resize; upper bound — but resize is deterministic and the witness CAN place edges to survive it via Force 3, so the *uncontrollable* floor is the uint8-only 0.0196).
- **Why p95 (not max/p50):** the hinge must stop pushing once a pixel is safe against the *typical worst-case* R-noise. p95 ⇒ "R-noise exceeds this only 5% of the time" ⇒ a pixel above δ_R·(1+headroom) is R-robustly safe. max (0.086) is a rare-tail artifact; p50 (0.005) under-protects.
- **Scale context:** δ_R=0.0196 is ~11% of the GT annulus margin p10 (0.172). So the noise floor is SMALL — the hinge saturates at a LOW margin, meaning almost all gradient is freed from the large-margin interior.

### 2.2 Formulation — one-sided saturation, and REPLACE-vs-MASK derivation
```
L_sat = w_s · mean_{x∈annulus}  relu( m_safe − m_wit(x) ) ,   m_safe = δ_R·(1+headroom)
```
- **m_wit** = the witness GT-class signed margin (the `_signed` field, #141 `_topk_margin` = top1−top2, differentiable). Zero gradient where m_wit ≥ m_safe (already R-safe) → gradient reallocates to the band BY CONSTRUCTION.
- **m_safe default = 3·δ_R ≈ 0.06** (headroom≈2): safely above both the uint8-p95 (0.020) and the full-R-p95 (0.037), so a pixel at m_safe survives both uint8 AND resize noise. (Expose headroom as a param; A/B over {2·δ_R, 3·δ_R, full-R-p95=0.037}.)

**REPLACE vs MASK — derived, this preserves the τ-anneal:** the existing seg loss at the sharpening stage is `tau_softplus`, which is a *smooth top-2 margin reduction* of CE (FEED-v7seal: "tau_softplus = top-2 reduction of multiclass L_τ; bit-exact CE at τ=1"). So the incumbent seg loss is ALREADY a temperature-τ margin loss; **the satisficing hinge is its τ→0 hard limit with an explicit ceiling m_safe.** Therefore:
- Do **NOT** REPLACE CE globally — early (CE/low-τ) the partition is still FORMING; an interior ceiling would starve region formation (the area-Lagrange / island-birth stack depends on interior-forming pressure).
- **MASK-BY-STAGE (the derived-correct composition):** keep the incumbent CE/τ loss unchanged; ADD `L_sat` that **anneals IN at the l7 sharpening stage boundary** (partition formed → now satisfice). This preserves the τ-anneal (early formation intact) AND reallocates the sharpening-stage gradient off safe interiors onto the band. Equivalently: at l7, down-weight the interior CE contribution and let L_sat own the band. Default `start_epoch = l7_start` (matches seg_chroma_boundary start=300, lane_band start=350).

### 2.3 Gradient-reallocation prediction (from #333/FEED-PA)
In the converged regime ~89–100% of d_seg is in the annulus (2.3–4.8% of pixels; band-1.0 annulus area_frac = 0.0256 measured). CE currently spends gradient on the ~95% interior that already has margin p50≈0.897 ≫ m_safe=0.06. Capping at m_safe removes gradient from every pixel with margin > 0.06 (≈ the entire interior + most of the annulus that's already safe). **Prediction:** the band's share of the seg-gradient budget rises from ~(annulus flip-mass share) toward ~100%; the interior 95%-of-pixels budget is reallocated to the 2.6%-area band. This is the UNIWARD/Fisher satisficing reading (spend the code where the detector's margin is fragile).

### 2.4 DSL Lever spec
```
Lever name:  margin_band_satisficing         (NEW factory; distinct from the existing
             --seg-loss margin_hinge which has NO R-derived ceiling)
compiles argv: --seg-margin-satisfice-weight <w_s=0.0 default-OFF>
               --seg-margin-satisfice-msafe <0.06>          # = headroom·δ_R
               --seg-margin-satisfice-delta-r 0.0196        # provenance-stamped measured floor
               --seg-margin-satisfice-headroom 2.0
               --seg-margin-satisfice-start-epoch <l7_start>
               --seg-margin-satisfice-band 2.0              # annulus mask
params/defaults: w_s=0.2 (when active), m_safe=0.06, delta_r=0.0196, headroom=2.0, start=l7
gating: default-OFF; start≥l7 (partition-formed, preserves τ-anneal); m_safe stamped
        with δ_R provenance (reports/delta_R_noise_floor.json).
telemetry rows: {stage:"margin_satisfice", epoch, w_s, m_safe, delta_r,
                 frac_px_below_msafe, gradient_share_band, gradient_share_interior}
```
`.validate()`: m_safe ≥ δ_R (else the hinge is inside the noise floor = pointless); start≥l7; delta_r matches the reports/ artifact (fail-loud on drift).

### 2.5 Pre-registered A/B acceptance
Arm = margin_band_satisficing (w_s=0.2, m_safe=0.06, start=l7) vs baseline. **ACCEPT if:** `frac_px_below_msafe` shrinks over the window AND verdict d_seg improves OR is neutral while `gradient_share_band` rises (the reallocation is the mechanism; a neutral d_seg with freed budget still helps when composed with Force 3). **WATCH:** interior d_seg must NOT rise (τ-anneal preserved). Scope = formulation.

---

## FORCE 3 — TIE-LOCUS NORMAL-DISPLACEMENT

### 3.1 Derivation + status: THE MACHINERY IS ALREADY BUILT (the `subpix` term, trainer L4559-4576)
The d_seg currency IS boundary displacement: a flip is the argmax tie-locus landing on the wrong side of a pixel center. FEED-PA proves 100% of the achievable floor is boundary placement. The existing `subpix` term already trains this:
- Per genuine inter-class straddle pixel, witness sub-pixel ratio `t_wit = M_w/(M_w+M_q+ε)` where `M_w=max(_signed,0)` (witness GT-class margin) and `M_q` = the dominant cross-edge partner margin (a pure shift of the shared `M_w` in the RIGHT/DOWN direction).
- `subpix_term = Σ (t_wit − t_ref)²·active / Σactive`, `t_ref` = the GT sub-pixel ratio `M_GT[p]/(M_GT[p]+M_GT[q])`. **Fully differentiable** through `_signed` (witness through-R logits). Providers (`_subpix_t_prov`, `_subpix_dir_prov`) are pure numpy from cached `gt.margins/gt.lstars` (θ-independent, ~940MB@n600), keep GENUINE-V straddles (lstar differs AND both GT margins < v-band=1.0 → ~1.12% px active). Flags exist: `--seg-subpix-boundary-weight/-start-epoch/-v-band`.
- **δn = |t_wit − t_ref|** IS the sub-pixel normal-displacement error. So the "soft δn" is already the differentiable margin-ratio; the #275 localizer machinery IS this loss.

**What is MISSING to make it the full force:** the term is currently **uniform over straddles**. FEED-PA says the flips are NOT uniform — they concentrate on Road-adjacent edges (Road hub; Road↔Lane = 41% of Road's flips).

### 3.2 The missing piece — flip-density edge weighting w_e
Weight each straddle by its adjacency-graph edge's flip-mass share (FEED-PA destination matrix). Build a 5×5 symmetric edge-weight matrix `W_e[c_a,c_b]` from the P-A per-edge flip mass; the providers already classify each straddle by its class-pair (lstar[p], lstar[q]) → look up `W_e` → per-straddle weight. Concrete seed weights (∝ P-A flip mass, Road-hub dominant, normalized to mean 1.0): Road↔Lane heaviest, then Road↔Undrivable, Road↔Movable, Road↔MyCar; non-Road edges light. Exact matrix = increment-1 reads from the P-A artifact (`bf1ee1fa8`); do NOT hardcode a guess — stamp it from the measured destination matrix.
```
L_tie = w_tie · Σ_straddles  W_e[c_a,c_b] · (t_wit − t_ref)² · active / Σ (W_e·active)
```

### 3.3 Non-duplication (vs boundary_distance / focal / annulus telemetry)
- **vs `boundary_distance` (default-off):** penalizes distance-to-boundary = a field-*regularity/localization* prior. It shapes WHERE the level-set is smooth, NOT the sub-pixel argmax crossing. Complementary; different quantity.
- **vs `margin_saliency`/focal (default-off):** reweights CE by the detector's ∂margin/∂input sensitivity = a *first-order importance weight*. It sharpens the existing loss on sensitive pixels; it does NOT set a displacement target. Complementary.
- **Force 3 is the ONLY term whose target is the sub-pixel tie POSITION** (the d_seg currency itself). The annulus telemetry (#333) is *observability*; Force 3 is the *actuator* on the same locus.

### 3.4 DSL Lever spec
```
Lever name:  tie_locus_displacement        (wrap the EXISTING subpix term + add w_e)
compiles argv: --seg-subpix-boundary-weight <w_tie=0.0 default-OFF>   # EXISTS
               --seg-subpix-boundary-start-epoch <l7_start>            # EXISTS
               --seg-subpix-boundary-v-band 1.0                        # EXISTS
               --seg-subpix-edge-weight-source {uniform|pa_flipmass}   # NEW: default pa_flipmass
               --seg-subpix-edge-weight-path <reports/pa_edge_weights.json>  # NEW
params/defaults: w_tie=0.3 (when active), start=l7, v_band=1.0,
                 edge_weight_source=pa_flipmass (falls back to uniform if artifact absent)
gating: default-OFF; start≥l7; w_e stamped from the P-A artifact (fail-loud if pa
        source selected but artifact missing → uniform + WARN, never silent).
telemetry rows: {stage:"tie_locus", epoch, w_tie, v_band, edge_weight_source,
                 n_active_straddles, mean_delta_n=|t_wit-t_ref|,
                 per_edge_delta_n (Road↔Lane etc.), frac_active_road_adjacent}
```
`.validate()`: edge_weight_source∈{uniform,pa_flipmass}; if pa_flipmass, path must exist or downgrade-with-WARN; v_band>0.

### 3.5 Pre-registered A/B acceptance
Arm A = tie_locus (uniform w_e), Arm B = tie_locus (pa_flipmass w_e), vs baseline. **ACCEPT the force if:** `mean_delta_n` drops AND verdict d_seg improves; **ACCEPT the w_e refinement (B over A) if:** the Road↔Lane `per_edge_delta_n` drops more than uniform AND Road d_seg (the 43.7% flip-mass hub) improves. Scope = formulation. This force is the direct **precision counter-force** the FEED-roadfloor bug named as missing (recall-without-precision) — it is the highest-EV of the four per FEED-PA (aims at "what the oracle says is EVERYTHING").

---

## FORCE 4 — R-PHASE ALIGNMENT → **FOLDS INTO FORCE 3** (verdict: do NOT build a second term)

### 4.1 The verdict and its derivation
**#4 is NOT a separate DSL lever. It is (a) captured in TRAINING by Force 3 being computed on the through-R `_signed` field, and (b) realized at DECODE by the existing subpix Consumer-B render-placement target.** Reasons:

1. **#149's mechanism = place the boundary at the sub-pixel location at 874-res so that D (bilinear-down→384) lands the boundary pixel on the intended side.** That target IS Force 3's `t_ref`. The subpix term's own code comment (trainer L4603+, "CONSUMER B") states the same t/dir maps ARE "the #149 R-phase bridge / decode-time RENDER-PLACEMENT target." Two names, one mechanism.
2. **The R-phase is already handled by training-through-R.** Force 3's `_signed` is the witness margin AFTER the exact R (bicubic↑→uint8→bilinear↓). So the sub-pixel target is already expressed in the post-D (384) decision domain; the gradient flows through D. The measured #149 gain (12× band collapse, d_seg 0.176→0.022→0.00094 via "grad through D+SegNet") is EXACTLY what a through-R subpix loss produces — the trainer already trains through the exact R. There is no additional training signal a separate "R-phase" term would add.
3. **The only genuinely distinct #149 artifact is the DECODE-time render placement** (the analytic lane band / AA-SDF placed at 874 by the closed-form) — that is a RENDER change, already speced as subpix Consumer-B, NOT a loss term. The ½-pixel phase offset of the area/bilinear downsample is a *provider detail*: whether `t_ref` is computed at 384 (current) or at 874 mapped through D's sampling phase.

### 4.2 The ONLY addition #4 motivates (a sub-option of Force 3, not a new lever)
Give Force 3's providers an OPTIONAL flag to compute `t_ref` at camera-res (874) mapped through D's sampling phase, for the render-placement consumer:
```
add to tie_locus_displacement lever:
  --seg-subpix-ref-domain {seg384|camera874_dphase}   default seg384
```
- `seg384` (default): current behaviour, `t_ref` from `gt.margins` at 384 — correct for the TRAINING loss (already post-R).
- `camera874_dphase`: compute `t_ref` at 874 and fold the area-downsample phase into the sign threshold — for the decode-time analytic-band render-placement (Consumer B). This is the #149 closed-form realized as a *provider option*, not a second force.

**Net: four forces → three DSL levers** (temporal_screw_consistency, margin_band_satisficing, tie_locus_displacement) + one sub-option on the third. No duplicate mechanism.

---

## CROSS-CUTTING

### 5.1 Interaction matrix (the four + v7.5's area-Lagrange + completion event)
| pair | relation | note / ordering |
|---|---|---|
| **#1 temporal ⊗ #2 satisfice** | **SYNERGY** | both act on the annulus; #1 owns the 44% flicker sub-part, #2 owns the static-margin sub-part → **disjoint residual fractions, ADD**. |
| **#1 temporal ⊗ #3 tie-locus** | SYNERGY, ordered | #3 places the tie (spatial), #1 stabilizes it (temporal) → mostly disjoint. **Order: #3 first (placement), then #1 (stabilize).** Mild antagonism at MOVING edges → #1 masks Movable/MyCar (already speced). |
| **#2 satisfice ⊗ #3 tie-locus** | **STRONG SYNERGY** | #2 frees the interior gradient budget → #3 (the actuator) spends it on the band. #2 = budget-freer, #3 = actuator. **The core pairing.** |
| **#2 satisfice ⊗ v7.5 area-Lagrange** | **WATCH (orthogonal quantities, sequencing-critical)** | area-Lagrange constrains per-class AREA (fixes Road over-paint theft); #2 caps MARGIN. Orthogonal quantities → no direct conflict, BUT #2 must not remove the interior-forming pressure area-Lagrange needs. **Resolution: #2 start_epoch ≥ region-formation (l7); area-Lagrange forms regions first.** |
| **#3 tie-locus ⊗ v7.5 completion event** | **STRONG SYNERGY** | completion fills birthed islands (recall); #3 places their boundaries (precision). FEED-roadfloor bug = recall-without-precision; **#3 IS the precision counter-force.** |
| **#1 ⊗ area-Lagrange** | mild synergy | temporal coherence steadies per-class area. |
| **#3 ⊗ #2 ⊗ completion** | compound | the sub-0.15 boundary-placement path: completion (recall) + #3 (precision placement) + #2 (budget) — the three that target the FEED-PA "boundary placement is everything" finding. |

### 5.2 Total gradient-share budget sanity
- **Hard rule (L4 confound):** `term_domination` alarm fires at any single term > 40% of total loss. seg≈4.3 must stay dominant during formation. The three new terms must SUM well under the seg budget.
- **Cold-start all default-OFF (0).** Activate ONE per crucible increment (scope discipline, FEED-missingforces: "composition decisions ride the crucible, NOT folded into v7.5 mid-seal").
- **Recommended caps:** each new term ≤ ~15% loss share; the three sum ≤ ~40%. Weights set at STAGE BOUNDARIES ONLY. **NEVER per-step adaptive re-weighting** (L4/L5: per-step GradNorm-style adaptation muted the confound canary in the frozen-run incident — stage-boundary weights only).

### 5.3 Compute cost (score-neutral speed discipline)
- **#1 temporal:** ONE extra differentiable homography-warp (`warp_real_luma_frame0.mlx`) + a prob-space MSE on the annulus (~2.6–4.8% of px). ξ from cache = free. The single new forward; bounded by annulus size → small. `ground_gt` arm has NO ξ-grad (cheaper).
- **#2 satisfice:** ONE relu on the already-computed `_signed` field. No new forward. ~free.
- **#3 tie-locus:** ALREADY in the trainer (subpix); providers are a one-time numpy build (~940MB@n600 memory, already budgeted); per-step is a masked MSE. w_e adds a 5×5 lookup. ~free incremental.
- **#4:** folded → zero cost.
All score-neutral (speed is lexicographic-secondary, L59; none trades score).

### 5.4 Event vs always-on
- **#2, #1, #3 are CONTINUOUS FIELDS → always-on from their start_epoch** (standing gates, annealed in at l7). None is a natural EVENT.
- **The v7 EVENT machinery (completion/retraction) is birth/death** — a different mechanism (discrete islands). The only EVENT candidate among the *missing forces* is persistence-PRUNING (death; FEED-missingforces #4-weak) — **out of scope for these four.**
- #4's decode-time render-placement (Consumer B) could be a one-shot recompute at a stage boundary, but that is a render step, not a loss event.

---

## 6. Summary for the phase-2 code wave
- **3 new/extended DSL `Lever` factories** (all default-OFF, registered + duty-to-measure per L31): `temporal_screw_consistency` (NEW), `margin_band_satisficing` (NEW), `tie_locus_displacement` (WRAP existing subpix + add w_e + ref-domain sub-option).
- **#4 R-phase is a sub-option of #3** (`--seg-subpix-ref-domain`), NOT a lever.
- **δ_R = 0.0196** (measured, `reports/delta_R_noise_floor.json`) → m_safe default 0.06.
- **w_t** cold-start 0.1 + stage-boundary ramp to gradient-share≈0.44 (per-term gnorm telemetry is the missing observability phase-2 must add).
- **Highest-EV force = #3 tie-locus** (FEED-PA: boundary placement is 100% of the floor; the precision counter-force the roadfloor bug named).
- Activate ONE per crucible increment; caps ≤15% each / ≤40% sum; stage-boundary weights only; never per-step.
- **Pointer 0.19110 UNMOVED — all of this is MEANS until a byte-closed `upstream/evaluate.py` n600 row moves it.**
