# FLICKER × TRANSFORM-CHAIN × LEVEL-SET GEOMETRY — mechanism attribution + the correcting term (design)

**Agent:** Fable flicker deep-dive · **Date:** 2026-07-10 · **Axis:** all fresh numbers `[macOS-CPU advisory · NON-PROMOTABLE]`, $0, NO scorer forward (cached argmax/margins + the exact R ops only), MPS never. **Pointer 0.19108282 UNMOVED** — this designs a term; nothing here moves the score until built + byte-closed through `upstream/evaluate.py`.

**Operator directive (verbatim):** *"deep dive on the flicker and interaction with rescaling and bicubic and all transforms and source against our level set and witness and morse smale … we can likely solve as well and integrate as a term or operator at the proper place and step and level and strength and dimension and frames vs pairs."* Coordinator addenda: ground in the authoritative upstream code (line-referenced), and treat GIBBS RINGING as a first-class candidate mechanism.

**STORES CONSULTED:** graph_memory_recall ("flicker temporal coherence R-phase…") · DAG FEED-03t/03u/03w/03x/lq/ma/PA/missingforces · `.omx/research/p0_forces_derivation_20260708.md` (#360 phase-1) · eq `independent_flicker_jitter_dseg_floor_smooth_optimal_v1` · eq `contest_r_operator_mtf_allpass_to_2px_v1` · #149 (`p0_forces` §0) · L65/L66/L67/L68 · trainer `--seg-spike-reweight` (#274, BUILT never-fired) · `reports/delta_R_noise_floor.json` (δ_R=0.0196) · lane geometric factorization `fcfc02309` · movable carrier `87493d947` · ground-plane verdict `692489e3b`.

---

## 1. The AUTHORITATIVE transform chain (line-referenced; paraphrase corrections)

Read from the pinned upstream snapshot (READ-ONLY) + the live trainer:

| step | op | source |
|---|---|---|
| pairing | `seq_len = 2`, consecutive frames accumulated into NON-OVERLAPPING pairs (0-1, 2-3, …; 1200 frames → 600 pairs) | `upstream/frame_utils.py:10`; `AVVideoDataset.__iter__` L199-208; `TensorVideoDataset.__iter__` L238-245; asserted `evaluate.py:78` |
| seg input | **`x = x[:, -1, ...]` — ONLY frame_1 of each pair is ever seg-scored** | `upstream/modules.py:111` |
| seg resize (D) | `F.interpolate(x, size=(384,512-grid), mode='bilinear')` — **`antialias` defaults False, `align_corners` default** — camera (874×1164) → (384×512) | `upstream/modules.py:113` (`segnet_model_input_size=(512,384)` `frame_utils.py:13`) |
| d_seg | `(out1.argmax(1) != out2.argmax(1)).float().mean()` — per-pair argmax disagreement | `upstream/modules.py:116-117` |
| pose input | BOTH frames; bilinear to (384,512); `rgb_to_yuv6` (`@torch.no_grad()`, in-place clamps, 2×2 box chroma subsample) | `upstream/modules.py:76-80`; `frame_utils.py:51-72` |
| d_pose | MSE on first 6 of 12 pose dims | `upstream/modules.py:88-90` |
| score | `100*segnet_dist + sqrt(posenet_dist*10) + 25*rate`; rate = `archive.zip` stat / uncompressed | `upstream/evaluate.py:96`, `:64-66` |
| frames | comp side loaded as raw **uint8** camera-res tensors; GT side decoded `yuv420_to_rgb` → uint8 | `frame_utils.py:218-245`, `:200` |
| witness up-path (OURS, not upstream) | `F.interpolate(mode="bicubic", align_corners=False)` render→camera, then `clamp(round(x),0,255)` uint8 | trainer `_torch_R_to_camera_uint8` L1415-1429 |

**Paraphrase corrections that matter (real code wins):** (a) the "R chain" `bicubic↑ → uint8 → bilinear↓` is HALF upstream, half ours — upstream applies ONLY `bilinear↓ noAA` (D) + consumes uint8 frames; the **bicubic↑ is the decoder's choice** (already recorded in `p0_forces` §0 via #149; re-confirmed here at source). Upstream contains **zero bicubic** anywhere. (b) d_seg's temporal support is the **stride-2 scored-frame sequence** {f1 of pair t} = odd frames only; f0 is structurally seg-free. (c) The score is a SUM of independent per-pair terms — **no cross-pair coupling in the objective**; temporal coherence enters only through what it does to each pair's own error.

---

## 2. Fresh measurements (all $0, cached data + exact ops; scripts in session scratchpad, results inline)

### 2.1 Part A — synthetic phase-transfer function of the exact chain (torch CPU, 64-phase sweeps, contrast 60 LSB)

Sweep a boundary's sub-pixel phase over one pixel; measure the recovered post-D (384-grid) edge-position error's PHASE-VARYING part (p-p, px@384 — the constant offset is learnable; the phase-varying part flickers as ego-motion advances phase) and Gibbs overshoot.

**Witness side** (edge rendered at 384-grid → bicubic↑874 → uint8 → bilinear↓384):

| edge width @384 | full chain (noAA) p-p | no-uint8 p-p | AA p-p | bilinear-up p-p | Gibbs overshoot @874 → post-D |
|---|---|---|---|---|---|
| hard (w=0) | 0.969 px | 0.969 | 0.988 | 0.969 | **12.38 LSB → ≤1.14 LSB** |
| w=0.5 px | 0.075 px | 0.055 | 0.063 | 0.073 | ~0 → 0 |
| w=1.0 px | 0.064 px | **0.016** | 0.054 | 0.062 | ~0 → 0 |

- The hard-edge ~1px p-p is the RENDER's own grid quantization (a binary-pixel witness can't encode sub-pixel position), not the chain: identical with bilinear-up, identical without uint8. This is the #149 mechanism seen from the other side (soft sub-pixel edges collapse it 13×).
- For the soft edges the witness actually paints: total phase noise ≤0.075 px p-p, **uint8-dominated** (w=1: 0.016→0.064 when uint8 added), bicubic-vs-bilinear-up ≈ no difference, AA ≈ no difference.
- **GIBBS (coordinator directive): measured ≈ NON-BINDING through this chain.** Bicubic's negative lobes ring 12.4 LSB (~10%/side, textbook) at 874 on a HARD step, but upstream's bilinear-D averages it to ≤1.1 LSB at the SegNet grid; for soft (≥0.5px) edges the ring is ~0 even at 874; and the phase-error is bit-identical between bicubic-up and bilinear-up. Consistent with eq `contest_r_operator_mtf_allpass_to_2px_v1` (R all-pass; AA-supersample even HURT −49% at ep200) and FEED-03t "Gibbs ABSENT (hosc working)". The representation-side Gibbs (sine basis) is already owned by the hosc/step-native anneal (L13 lever #5, β 1.0→3.177 in the live config) — **no new anti-ring term is warranted; verdict scope: this chain + soft-edge witness renders** (a future hard-step render family would need re-measurement).

**GT side** (camera-native uint8 edge → bilinear↓, phase advancing at CAMERA scale = one full cycle per camera-px of boundary motion):

| camera blur | noAA p-p @384 | AA p-p @384 |
|---|---|---|
| hard (0) | **0.426 px** | 0.426 px |
| 0.7 px | 0.114 px | 0.085 px |
| 1.5 px | 0.059 px | 0.055 px |

The GT-side boundary position at the SegNet grid carries **0.09–0.43 px p-p deterministic phase jitter** as edges advect; for hard edges AA removes NOTHING (it is sampling information loss, not filterable aliasing). Boundary-pixel VALUE modulation: 18–22 LSB p-p at contrast 60 (30%+ of edge contrast oscillates at the boundary pixel per camera-px of motion). D's sampling-phase pattern is spatially quasi-periodic (scale 1164/512 = 291/128 → phase pattern period 128 output px) — a fixed Moiré comb the advecting boundary sweeps through.

### 2.2 Part B — GT argmax churn on the SCORED sequence (n600 cached `lstars`/`margins`, 599 transitions)

- Churn (lstars[t]≠lstars[t+1]): **1.245%/transition**; best global shift = **(0,0) on 94.8% of transitions** (mean |shift| 0.053 px) → between scored frames (2 frames, ~0.1 s) the partition's gross motion is SUB-PIXEL at 384; the churn is boundary-phase jitter, **not** gross motion. Shift compensation removes ~nothing (1.2456→1.2425%).
- Concentration: 64.0% of churn inside the |GT margin|<1 annulus (2.67% area → **24×**); flip margin p50 0.647, 63.2% below 1.0.
- Class structure: flip share Road 45.5% / Lane 28.0% / Undriv 11.4%; per-px flip RATE Lane **59.5%** per transition (the unstable orbit); transition matrix **Road↔Lane = 55.5%** of all churn (matches FEED-03t "lane↔road = 55%").
- **Blink-back fraction 41.8%** — flips that revert by the next scored frame (single-frame flicker), n600 all-class confirmation of FEED-lq's 40% (n96 Road↔Lane).
- q̂ field: 27.9% of px ever flip; top-1% px carry 21.9% of flip mass.

### 2.3 Part C — real-frame temporal INPUT jitter at the SegNet grid (50 transitions, exact D, luma)

| where | noAA mean | noAA p95 | AA mean | noAA/AA |
|---|---|---|---|---|
| annulus (\|margin\|<1) | **9.03 LSB** | 40.0 LSB | 8.48 | 1.065 |
| interior (\|margin\|>5) | 2.66 LSB | 7.1 LSB | 2.37 | 1.123 |

- Boundary input jitter is **3.4×** the interior sensor/scene/codec floor (2.66 LSB — the direct measurement of the source-noise bucket).
- **The AA-removable (classical aliasing) share is only ~6%** of annulus temporal jitter — the jitter is dominated by REAL sub-pixel advection of camera-blurred edges through the sampling comb, which no receiver-side filter removes.
- Exact static-per-pixel-mode Bayes floor, n600 all-class: **0.02650** (a witness with no per-pair conditioning cannot beat 2.65% d_seg; predict-stay floor = churn = 0.01242; the witness's 0.00496 already beats both → per-pair conditioning works, per FEED-lq).

### 2.4 Part D — spike structure (the #274 quantity, n600)

- **GT single-frame SPIKE rate (label differs from BOTH stride-2 neighbors): 0.005318 of the frame** (1046 px/pair); coherent-unstable px 2807/pair.
- **97.7% of spikes are "repairable"** (both neighbors agree on the same label — the temporal-majority target is well-defined).
- 67.7% of spikes inside the annulus; spike GT-margin p50 **0.555** (vs GT annulus p50 0.897 — spikes live in the LOW-persistence tail of the margin field).
- Spike class share: **Lane 43.6%** / Road 35.5% / MyCar 8.2% / Undriv 7.4% / Movable 5.4% — the L67 "44% of CE-residual spikes = LANE" is the same object measured from the target side.

### 2.5 THE IDENTIFICATION (the load-bearing new fact)

**The witness's converged residual (d_seg 0.00496–0.0052, the #205 popout floor, L67/FEED-03u) is numerically THE GT spike rate (0.00532).** A temporal-majority oracle (predict the repaired label) scores exactly the spike rate; the witness has converged to that oracle's performance (slightly below — it already fits ~10–20% of spikes via per-pair conditioning). Two corollaries:

1. **The sub-0.15 d_seg need band (0.00077–0.00118) is 4.5–7× BELOW the smooth-label flicker floor.** By eq `independent_flicker_jitter_dseg_floor_smooth_optimal_v1` (cost q at r=0), NO temporally-smooth-in-LABEL-space witness reaches the need. The flicker is not a nuisance term — it is the remaining wall.
2. **The floor is pierceable only by APPEARANCE-PHASE faithfulness:** the spikes are deterministic functions of the per-pair camera frame (SegNet is deterministic); a witness that reproduces the boundary-pixel intensity PHASE inherits the spikes for free (fully correlated jitter → d_seg→0 at those px). Existence proof already measured: FEED-ma SIGNAL A (real-frame content through R) reaches d_seg **0.00086 < 0.00532**. The information is per-pair-storable (the witness has per-pair codes) and low-dim (a sub-pixel phase per boundary element, advected by ξ); the binding constraint is the basis's along-tangent capacity (L65, 3.2× deficit) and chroma — the training/representation walls already on the DAG, NOT a new information wall.

### 2.6 Mechanism attribution — the jitter budget (annulus, between scored frames)

| mechanism | measured size | share/verdict |
|---|---|---|
| **(a1) sub-pixel ADVECTION PHASE through noAA-D sampling comb** (deterministic given scene+ξ) | input 9.0 LSB mean / 40 p95; position 0.09–0.43 px p-p per camera-px motion | **DOMINANT source** (≈70% of annulus input jitter after removing the sensor floor) |
| (a2) classical aliasing (AA-removable) | 6.5% of annulus jitter (0.55 LSB) | minor; unfixable anyway (upstream D is pinned) |
| (b) uint8 quantization phase | 0.5 LSB p-p input; margin-referred δ_R p95 = 0.0196 vs flip-margin p50 0.65 | minor direct contributor (~11% of annulus margin p10); witness-side it is the largest CONTROLLABLE chain term (Part A w=1: 4× the resize part) — already owned by Force 2's m_safe=3·δ_R hinge |
| (c) source/sensor/codec noise | 2.66 LSB interior floor (≈30% of annulus mean); GT label-noise ~1px floor tiny (ground-plane verdict `692489e3b`, 97.5% coherent) | secondary source; the truly aleatoric core of the spikes |
| (d) SegNet argmax instability at small margin | flips at margin p50 0.65, spikes p50 0.555, margin AUC-for-flip 0.91 (FEED-lq), 24× annulus concentration | the AMPLIFIER (converts (a)+(b)+(c) into label flips), not a source |
| (e) GIBBS ringing (bicubic lobes / sine basis) | 12.4 LSB @874 hard-step → **≤1.1 LSB post-D**; ~0 for soft edges; phase-error identical bicubic-vs-bilinear-up | **NON-BINDING through this chain** (§2.1); representation side already owned by hosc anneal |
| (f) witness-side chain phase noise | ≤0.075 px p-p (soft edges), uint8-dominated | small, DETERMINISTIC, and already learned through-R (#149/Force 3) |

**Mechanism sentence:** camera-blurred class boundaries advect sub-pixel under the ego-screw; upstream's antialias-False bilinear-D + uint8 sample that motion through a fixed 291/128 phase comb, modulating boundary-pixel intensities by ~9 LSB (p95 40); SegNet's low-persistence margin tail (p50 0.56) amplifies the modulation into single-frame label spikes (0.53% of frame, Lane-dominated); the witness, trained by hard-CE to the spiked labels, has converged to exactly the temporal-majority floor those spikes define.

---

## 3. The geometry connection (Morse-Smale / eikonal / ξ)

- **Morse-Smale, made precise:** spikes are the **persistence tail of the SPACE-TIME margin field below one resolution cell** — spatial persistence p50 0.56 (below the annulus median 0.897), temporal persistence exactly 1 scored frame (below the witness's temporal resolution of 1 pair). They are birth-death pairs of the argmax partition flickering across the resample lattice; the flicker IS the L66 annulus read in time. Under the unified level-set flow this is the same object as the L65 lane-dash erasure (error ∝ 1/persistence), now on the TEMPORAL axis.
- **Eikonal/level-set:** the predictable part of the flicker is a pure **transport term** — the phase of the separatrix advected by the ground-plane homography of ξ: ∂t/∂τ + v_ξ·∇t = 0 along the boundary (t = the #275/Force-3 sub-pixel tie coordinate). The unpredictable part enters the FIDELITY term of the variational flow, where marginalizing over boundary-position noise = normal-direction blur of the target indicator = an anisotropic viscosity on the data term (NOT a new PDE term on φ).
- **ξ-residual:** the ground-plane verdict's split (97.5% coherent / 0.8px incoherent) is the same advective/aleatoric split measured here at label level (58% coherent-unstable / 42% blink-back). Advect-then-treat: transport by ξ explains the coherent 58%; the term below treats the two parts differently.

---

## 4. THE CORRECTING TERM — two branches at one locus, fully specified

The measurement splits the flicker into a **predictable phase channel** (advective, per-pair-storable) and an **aleatoric residue** (sensor-driven single-frame spikes). The correct operator treats each on its own branch, both defined on the same locus (the annulus tie-field of the SCORED frame). Neither is a Gibbs/anti-ring term and neither is an R-side surrogate change (both measured non-binding, §2.1).

### T1 — CROSS-PAIR PHASE-ADVECTION CONSISTENCY (new lever; the genuinely missing operator)

**WHAT/WHY:** Force 3 (`subpix`) targets the per-pair sub-pixel tie position t_wit→t_ref, but each pair's t_ref is measured through the jittering chain — per-pair phase noise σ ≈ 0.09–0.43 px p-p (§2.1). Force 1 transports the FIELD φ within a pair (f0→f1; f0 is seg-free, so it is a pure regularizer). **Nothing ties the SCORED phase sequence together across pairs**, even though the phase advances deterministically with ξ (advection, §3) and the per-pair FiLM codes are free to jump. T1 is Force 1's transport applied to Force 3's quantity on the stride-2 scored sequence: the optimal shrinkage of the noisy per-pair phase targets toward their ξ-advected trajectory. This is how the witness fits the PREDICTABLE flicker channel (the path below the 0.0053 floor) without storing it: the phase trajectory is low-dim and mostly generated by ξ.

```
L_phase = w_p · Σ_{x∈straddle(f1,pi)∩GROUND} m_ann(x) ·
          ( t_wit(x, pi+1) − t_wit(A_ξ(x), pi) − Δt_ξ(x, pi) )² / Σ m_ann
```
- **t_wit** = the EXISTING Force-3 differentiable sub-pixel ratio `M_w/(M_w+M_q+ε)` on the through-R `_signed` field (trainer L4559+; providers already built).
- **A_ξ** = ground-plane homography advection of the straddle locus from scored frame 2pi+1 to 2pi+3; **Δt_ξ** = the closed-form phase advance of the tie locus under the same homography (the fractional part of the boundary's normal displacement on the 384 grid — pure geometry, θ-independent, precomputable per pair from cached poses). The pair-gap screw (frame 2pi+1→2pi+2 is not in `gt_poses`) is interpolated `ξ_gap ≈ ½(ξ_pi+ξ_pi+1)` (train-time regularizer target only — approximation error tolerated; A/B may substitute an offline gap-homography estimated from the cached `gt_f1[pi]`/`gt_f0[pi+1]` frames, θ-independent).
- **GROUND = {0,1,2} only** (homography wrong on Movable/MyCar — same mask as Force 1).
- **WHERE (stage):** loss term, annealed in at the **l7/sharpening boundary** (partition + tie-field must exist; same gate as Forces 1–3). It is a TRAINING prior only — decode is untouched (no inflate dependency, no rate change).
- **LEVEL/DIMENSION:** boundary level — a scalar phase per straddle pixel (the ~1.1% active straddle set), i.e. the codim-1 separatrix's normal coordinate — NOT pixel-RGB, NOT the full field.
- **STRENGTH (derived, #360-style):** the flicker fraction of the residual is 0.42–0.44 (blink-back 41.8% §2.2; L67 44%). Target gradient share ≈ **0.4 of the subpix term's** (not of total seg): cold-start `w_p = 0.5·w_tie`, ramp at stage boundaries only toward measured `‖∇L_phase‖ ≈ 0.4·‖∇L_subpix‖` via the per-term gnorm telemetry #360 phase-2 adds; cap ≤10% of total loss (under the 40% domination alarm, L4).
- **FRAMES vs PAIRS (exact):** defined on **f1-to-f1 across CONSECUTIVE PAIRS** (the stride-2 scored sequence — the only support d_seg has, modules.py:111 + frame_utils.py:10). f0 enters nowhere (Force 1 owns f0↔f1). The score has no cross-pair coupling (§1) — T1 is a PRIOR that reduces each pair's own phase-target variance, not a scored quantity.
- **DSL Lever spec (what SHOULD exist — flags do NOT exist yet; build-wave #377/#386 must add trainer flags + `Lever` factory TOGETHER):**
```
Lever name: phase_advection_consistency            (NEW factory)
argv: --seg-phase-advect-weight <w_p=0.0 default-OFF>
      --seg-phase-advect-start-epoch <l7_start>
      --seg-phase-advect-classes 0,1,2
      --seg-phase-advect-gap-xi {interp|offline_homography}   default interp
      --seg-phase-advect-band 2.0
gating: default-OFF, registered + duty-to-measure (L31); start ≥ l7;
        requires the subpix providers (reuse; fail-loud if absent);
        micro-batch: same serial-path constraint as #274 unless the batched twin
        learns per-pixel weights first.
telemetry: {stage:"phase_advect", epoch, w_p, gap_xi, raw_L_phase, gnorm_ratio_vs_subpix,
            mean_phase_residual_px, blink_fit_frac}
```
- **Pre-registered A/B acceptance:** vs the Forces-1+3 baseline at matched epoch: verdict d_seg drops toward the sub-floor regime AND the witness's own scored-sequence spike rate (witness argmax differing from both its own temporal neighbors) RISES toward the GT's 0.0053 **in the correlated direction** (blink_fit_frac ↑ — it must fit the GT's spikes, not add its own; the two are distinguished by per-px correlation with the GT spike mask) AND d_pose non-rising. KILL scope = **formulation**.
- **New or #360 re-parameterization?** NEW lever, but a **composition, not an invention**: T1 = (Force 1's ξ-transport) ∘ (Force 3's tie coordinate) on the support neither covers (cross-pair × scored-frames). It reuses Force 1's warp path + Force 3's providers; ~zero new machinery. #360's Force-4 verdict ("R-phase folds into Force 3") stands for WITHIN-pair phase; T1 is the CROSS-pair phase #360 did not treat.

### T2 — FIRE #274 WITH DERIVED VALUES (built lever; the aleatoric branch)

**WHAT/WHY:** the residue T1 cannot capture (sensor-driven spikes; ~30% of annulus jitter is the 2.66-LSB sensor floor, §2.3) must not be CHASED: hard-CE to a spiked label injects a wrong-target gradient at exactly the highest-loss pixels (44% of CE-residual spikes, L67). The lever is BUILT (`--seg-spike-reweight --seg-spike-downweight --seg-coherent-upweight`, trainer L5690-5727, default-inert 1.0/1.0, never fired = orphaned signal, L31). Derived firing values:
- **WHERE:** CE fidelity term, per-pixel weight, active from **ep0** (θ-independent provider; the spike map is pure GT preprocessing) — unlike T1 it needs no formed partition; it only stops noise injection.
- **LEVEL:** pixel level, on the spike mask (0.53% of px; 67.7% in-annulus).
- **STRENGTH (derived, not tuned):** the Bayes-consistent treatment of an identified single-event label noise is the soft target = neighbor-majority with weight (1−ρ) on the spiked label, where ρ = the fraction of spike mass the phase channel (T1) is expected to capture. Bounded honest range: ρ∈[0.11 (measured ego-predictable, FEED-03w), 0.7 (if T1's appearance channel performs)] → **A/B `--seg-spike-downweight ∈ {0.0, 0.25}`** (w=0 = pure don't-chase; the CE mass removed is ≤0.53% of px — domination-safe by construction), `--seg-coherent-upweight 1.0` (the coherent branch is owned by T1/Force 3 — do NOT double-weight, interaction discipline §5.2 of #360).
- **FRAMES vs PAIRS:** the spike mask is defined on the stride-2 scored sequence (lstar[pi] vs lstar[pi±1] — the trainer already does exactly this, L5709-5711); f0 untouched.
- **Sequencing with T1:** T2 first (it is built — one flag flip + values), T1 next increment (one lever per crucible increment, #360 §5.2). **T2 caps the witness at the 0.0053 floor if run FOREVER alone** — it is the mid-game noise-hygiene lever; end-game descent below the floor belongs to T1 + the appearance/chroma/along-tangent levers. Record this on the lever ledger so T2's success is never read as "flicker solved."

### T3 — measured NON-designs (scope-honest negatives)

- **R-side anti-ring / monotone-resample surrogate: NOT warranted.** Gibbs ≤1.1 LSB post-D, ~0 on soft edges, phase-error identical under bilinear-up (§2.1). Training already goes through the exact R (#149/Force 3). Scope: this chain + soft-edge renders.
- **AA-target / AA-render terms: NOT warranted for flicker.** AA-removable share 6.5% (§2.3); upstream D is pinned noAA; witness-side AA measured HURTING at ep200 (FEED-03t). (FEED-ma's AA-SDF render remains a REPRESENTATION lever on other grounds — unaffected.)
- **Store/replicate the flicker: already MEASURED NO-GO** (FEED-03x: b=0.876>0.65, r_admit 0.198≪0.688) — T1 differs by generating the phase from ξ+low-dim trajectory, not storing events.

---

## 5. Candidate equations (MEASURED here, NOT yet registered — flag for the equations leg)

1. `gt_scoredframe_spike_rate_equals_witness_flicker_floor_v1` — GT stride-2 spike rate **0.005318** (n600; repairable 0.977; Lane 0.436/Road 0.355; annulus share 0.677; margin p50 0.555) = the smooth-label witness floor; witness converged residual 0.00496–0.0052 ≈ AT it; sub-0.15 need 0.00077–0.00118 is 4.5–7× BELOW it; pierceable only by appearance-phase correlation (existence proof FEED-ma 0.00086). Anchors: Part B/D JSONs + L67 + FEED-03u.
2. `transform_chain_phase_noise_partition_v1` — the chain's phase-noise budget: witness-side soft-edge p-p ≤0.075 px (uint8-dominated, Gibbs ≤1.1 LSB post-D, bicubic≈bilinear-up); GT-side 0.09–0.43 px p-p per camera-px advected (AA-irremovable); input-referred annulus jitter 9.0 LSB mean / 40 p95 vs sensor floor 2.66 LSB; AA-removable 6.5%. Anchors: Part A/C JSONs; sisters `contest_r_operator_mtf_allpass_to_2px_v1` (amplitude), δ_R (uint8 margin-noise).

Verdict scopes: (1) is n600-full-support (the entire scored population, cached authority argmax) — FAMILY-level for smooth-label witnesses on THIS video/scorer; (2) is synthetic-plus-50-transition sampled — FORMULATION-level, adequate for term design, not for score claims.

## 6. Honesty block

- $0, CPU-only, no SegNet/PoseNet forward anywhere (cached `lstars`/`margins` are the frozen CPU-torch authority's outputs; the R ops run on synthetic strips + cached uint8 frames). Live run pid 88030 and the scorer probe untouched.
- All fresh numbers `[macOS-CPU advisory · NON-PROMOTABLE]`; nothing here is a score. **Pointer 0.19108282 UNMOVED.** The terms are MEANS; they move the pointer only via a trained arm → byte-close → `upstream/evaluate.py` exact row.
- The "witness residual = spike rate" identification is a numerical coincidence-with-mechanism at n600; the witness-side decomposition of ITS residual into spike/coherent parts needs the witness argmax (first telemetry row of the T2 arm — pre-registered above).

---

## Observability surface

*(OBSERVABILITY-ADDENDUM 2026-08-25 — APPEND-ONLY per Catalog #110/#113. This
section is an INDEX into this memo's own content per Catalog #305's 6 facets;
it adds no new claim. Facets with no counterpart in this memo say so plainly.)*

1. **Per-layer inspection** — §1 "The AUTHORITATIVE transform chain (line-referenced; paraphrase corrections)" walks the chain stage by stage with line references, which is the per-layer surface this design attributes against.
2. **Per-signal decomposition** — §2 is a four-part decomposition measured separately: Part A synthetic phase-transfer function of the exact chain, Part B GT argmax churn on the scored sequence (n600, 599 transitions), Part C real-frame temporal input jitter at the SegNet grid, Part D spike structure (n600). §2.6 decomposes the jitter budget.
3. **Run-to-run diff** — §4's terms are default-OFF trainer flags (`--seg-phase-advect-band`, `--seg-phase-advect-classes`, `--seg-phase-advect-gap-xi`, `--seg-phase-advect-start-epoch`, `--seg-coherent-upweight`), so an ON arm diffs against a byte-identical OFF arm.
4. **Post-hoc query** — `reports/delta_R_noise_floor.json` is the retained noise-floor artifact; the authority is `upstream/evaluate.py` on `archive.zip`; the chain surfaces are `frame_utils.py` / `modules.py`.
5. **Cite-chain** — §2's measurements are labelled with their cached inputs (n600 `lstars` / `margins`); §5 "Candidate equations (MEASURED here, NOT yet registered — flag for the equations leg)" keeps the equation debt visible; §6 is the honesty block.
6. **Counterfactual hooks** — §4 specifies two branches at one locus (T1 cross-pair phase-advection consistency, T2 fire #274 with derived values) plus §4's T3 "measured NON-designs (scope-honest negatives)" — the explicit did-not-work counterfactuals.
