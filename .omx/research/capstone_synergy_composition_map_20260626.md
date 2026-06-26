# Capstone Synergy + Composition Map — BIND ALL ingredients into the NON-RGB Task-Space Step Witness (2026-06-26)

**Trigger:** operator "remember all the synergy and all of our findings and research and stuff too" + the
PR95 win condition (a winner BINDS ALL ingredients into ONE coherent vehicle; we kept building ingredients
that never bound). This is the durable **no-signal-loss synergy map**: the full accumulated body inventoried,
classified by STATE, mapped by how each COMPOSES into the capstone, with the ANTAGONISMS made explicit and
the binding ORDER fixed. $0 CPU research-only; NO GPU arm (the decisive `witness_refound_n600_iso` owns the
single slot; fleet HARD-BLOCKED #173). Authority: every number `[advisory]`/`[macOS-MLX research-signal]`
unless tagged contest-CPU/CUDA. Pointer UNMOVED **contest-CPU 0.19110** (means != ends; no exact row here).

**Score law (JOINT, ingredients FUSE not add):** `S = 100·d_seg + sqrt(10·d_pose) + 25·B/37,545,489`.
**Binding term = SEG.** Frontier split: d_seg 0.00056 (0.0560) + d_pose 2.94e-5 (0.0172) + rate 0.118.
**base_ch20 existence proof [contest-CPU dual-axis]:** 89,244 B, d_seg 0.00260, d_pose 0.00034, **S 0.378**
→ rate 0.0594 grounded (half the frontier), binding wall = d_seg. **Sub-0.15 target: d_seg < 7.3e-4 at
rate 0.0594 + stored-pose 0.017.** Clean achievable **S ≈ 0.10–0.14** (reaudit R4).

This memo EXTENDS (does not duplicate) the canonical binding spec
`capstone_taskspace_step_witness_reorientation_and_binding_20260626.md` (the 7-step launch order + the
NO-FAKE pose-blind-palette resolution) with the full inventory table, the antagonism map, the joint-S, the
top-5, and the orphan-signal-loss list.

---

## ## INVENTORY (full table: ingredient / state / composes-how / expected-ΔS / byte-vs-0-byte / antagonisms)

Legend STATE: BUILT (code+tests, runnable) · PARTIAL (core built, gap noted) · RESEARCH (memo/claim only) ·
THEORY (proof, no impl) · MEASURED (number is real, axis tagged).

### POSE — SOLVED (composes orthogonally; frees ALL witness capacity to d_seg)
| # | Ingredient | Path | State | Composes-how | ΔS / number | byte vs 0-byte | Antagonism |
|---|---|---|---|---|---|---|---|
|1|Stored-pose sidecar|`src/tac/scorer_targets.py`|BUILT|stores 600×6 PoseNet targets; render supervised-conditioned to hit them|pose_term **0.017 SOLVED**|~5KB (7.2KB raw / <5KB zlib)|needs a REAL-RGB frame with texture DOF to hit targets (see anti-3)|
|2|Low-rank pose codec #140|`src/tac/codec_pipeline_kl_pose.py`|PARTIAL (KL built; poly unbuilt)|shrinks the sidecar via ~1-2 DOF trajectory|~hundreds B vs 5KB|byte (tiny)|none|
|3|pose_from_embedding MLP|`src/tac/pose_from_embedding.py`|BUILT|predicts 6-DOF from mask feats at inflate (scorer-free)|net −13KB vs `optimized_poses.pt`|~1-10KB|alt to #1, not stack|
|4|amortized-luma carrier #57/#163|`src/tac/boundary_math/amortized_luma_carrier.py`|BUILT|frame0 real-pose TEXTURE, SegNet-invisible → gives PoseNet pair texture|fixes palette d_pose collapse 12.66|byte (brotli)|the frame0 half of anti-3 fix|
|5|comma2k19 ego-motion GT #158|HF `commaai/comma2k19/global_pos/`|RESEARCH (downloadable)|grounds pose trajectory + base_seg warp from KNOWN source|±0.6m/±0.2°|0-byte prior|not integrated (orphan)|

### D_SEG LEVERS — the BINDING term (basis is PRIOR to capacity; STRICT ORDER)
| # | Ingredient | Path | State | Composes-how | ΔS / number | byte vs 0-byte | Antagonism |
|---|---|---|---|---|---|---|---|
|6|step-native / hosc basis|`train_witness_…_mlx.py --activation hosc`; `lever_b_generator.py`|BUILT (PARTIAL on realized)|topology-matched to piecewise-const argmax; no Gibbs → survives R|proxy best 0.004445|0-byte (train prior)|hosc tanh(β·sin) saturates → β-anneal (fixed)|
|7|chroma|`--chroma` (RGB witness)|BUILT|SegNet-argmax DOF (reads RGB) AND PoseNet texture (resolves anti-3)|−0.01..−0.03 rate (claim)|0-byte (train)|frontier decode-side chroma perturb HURT; it is a TRAIN lever not a decode perturb (anti-4)|
|8|all-class directional Fourier|`lever_b_generator.py all_class_boundary_*`|BUILT but **PROXY+circular**|orient feats to boundary tangent (curvelet-optimal)|**−48% n96 / −46% n600 PROXY**|0-byte IF tangent decode-available|tangent uses GT argmax → NOT byte-closeable for RGB witness; needs self-orientation fixed-point (anti-1)|
|9|capacity-routing KKT waterfill|`src/tac/torch_vehicle/boundary_routing.py`|BUILT (identity-at-init)|concentrate capacity on the ~0.72% boundary band|n96 dir+cap **−64%**|0-byte (train)|**ALONE on isotropic HURTS +6%**; pays ONLY after basis-match (anti-2)|
|10|margin-saliency map #141|`src/tac/boundary_math/posenet_jacobian_saliency.py`|BUILT (pose-side)|drives the waterfill (#9)|exact Jacobian field|0-byte|is POSE saliency; a SEG-margin saliency for d_seg routing is the gap|
|11|boundary-math seg core #52|`src/tac/boundary_math/seg_core.py` (+partition/contour_codec/bitmask/margin_polytope/region_merge)|BUILT|encodes base_seg geometry (the ~8-dim/edge statistic)|d_seg **0.0** lossless; **895.7 B/frame**|byte (~10-20KB/600)|none (it IS base_seg)|
|12|R_surv round-trip survival|`src/tac/torch_vehicle/lever_d_selective.py`|BUILT/MEASURED|R in-loop during train (already in B3 harness)|σ*=**0.7737**; crude σ=0.464 NO-GO|0-byte|crude (all-flips) below break-even; selective only|
|13|Lever-D margin-conditional residual #72|`src/tac/boundary_math/margin_conditional_residual.py`|BUILT/MEASURED|sparse seg-repair on decoder-KNOWN low-margin set B|waterline **1.27 B/flip**|sparse byte|net-negative on frozen frontier; EASIER on less-converged base|
|14|finishing-kit #105 (PR98/T10/S12)|`src/tac/torch_vehicle/distortion_finishing_kit.py`|BUILT/MEASURED|0-byte decode-side channel bias post-convergence|converged **−0.00291** (mid-basin −0.058 was artifact)|0-byte (PR98/T10); S12 cert|already in FRONTIER inflate (re-fit per base)|
|15|horizon-band margin #169|`src/tac/research/hardware_exploitation.py`|RESEARCH|sky/road band saliency boost|−5.48e-4 claimed (VIDEO-derived)|needs sidecar → dead rate|NO-GO standalone unless 0-byte prior|

### FROZEN-INSTANCE 0-BYTE PRIORS — source IS comma2k19 RAV4, GT public → FREE in inflate.py
| # | Ingredient | Path | State | Composes-how | ΔS / number | byte | Antagonism |
|---|---|---|---|---|---|---|---|
|16|openpilot lane poly + ground-plane homography #138/#145|`src/tac/openpilot_seeding.py` (V2)|BUILT (compress-time pose seed)|deterministic base_seg/pose geometry from known source|pose warm-start (no d_seg yet)|0-byte/tiny|currently pose-seed only; as decode-prior unbuilt|
|17|comma10k label conventions #156|`src/tac/losses/core.py`|RESEARCH (cited)|5-class argmax priors|—|0-byte|not a distinct module|
|18|ego-hood static clamp #139|(geometry_deliberation refs)|RESEARCH (not built)|free interior where SegNet never sees hood|~19 flips (negligible)|0-byte|tiny|
|19|homography static-class margin-map #158|`src/tac/segmap_renderer.py`|PARTIAL (road-plane)|geometric consistency on static class|rate −5..15% claim|0-byte|render-time; seg ΔS unmeasured|
|20|camera-res sub-pixel placement #149|(not found)|RESEARCH|boundary-band precision via full-res luma|12× boundary-band collapse claim|0-byte if deterministic|video-derived-as-sidecar = dead rate (anti-6)|
|21|keyframe + tiny-warp temporal amortization #148|`src/tac/lossless/rgb_semantic_labels.py`|PARTIAL (keyframe sel; warp unbuilt)|amortize seg across keyframes|every-20th (60 kf)|byte (reduces)|the seg-side of seg=pose fusion|

### REPRESENTATION / THEORY — the home of the task-space witness
| # | Ingredient | Path | State | Composes-how | number | Antagonism |
|---|---|---|---|---|---|---|
|22|level-set/fiber QUOTIENT codec #155|capstone binding memo|THEORY|scorer-quotient orbit = task-sufficient statistic (indirect-RD home)|S_floor 0.118 (loose LB)|pose CANNOT be coded separately <400B → MUST fuse (drives #24)|
|23|VCM / indirect-RD / CEO floor #150-152|`.omx/research/vcm_theory_primitive_layer_20260619T033429Z.md`|THEORY+MEASURED|the contest IS remote-RD coding-for-machines|S_floor **0.11797** (rate at 8b/byte entropy ceiling, 62% frontier)|rate axis already at ceiling → attack d_seg, not rate|
|24|**seg=pose FUSION** (deepest synergy)|capstone memo §4 + reaudit R4|THEORY+PARTIAL|per-frame partition = warp(base_seg, pose) + sparse object residual; **pose reused AS the seg-warp = 0-byte**|rate+pose → **15-40KB (rate 0.01-0.027)**|NOT trained yet; the one genuinely-new build|
|25|legal-frame variational bridge #56/#73|`src/tac/boundary_math/legal_frame_bridge.py` + `dykstra_legal_frame.py`|BUILT + MEASURED NEGATIVE|logit→legal-RGB inflate (compress-time target/boundary tool)|**palette POSE-BLIND S=11.65** (d_pose 2.67-12.66)|RESOLVED: legal frame must be REAL RGB (luma+chroma), NOT logit-derived (anti-3)|
|26|exact-sensitivity KKT reverse-waterfill #157/#54|`src/tac/optimization/cross_pair_waterfilled_corrector.py`|BUILT/MEASURED|bit allocation by exact scorer sensitivity|frontier already optimal (0/30 pairs; NET 0.0)|frontier-locally-optimal ≠ witness-optimal → RE-RUN on witness|

### OPTIMIZER / RATE-FINISHING
| # | Ingredient | Path | State | Composes-how | number | Antagonism |
|---|---|---|---|---|---|---|
|27|MD-Decoupling #B2|`--optimizer md`|BUILT (12/12 tests, review CLEAN)|anti-collapse + LR-transfer-across-width (the measured "capacity walls" were optimizer bugs — R2)|toy MD 0.00150 vs Adam 0.00243|orthogonal swap-in; default path byte-identical|
|28|validated realized harness #B3|`tools/validate_realized_harness_vs_oracle.py`; `src/tac/measurement_integrity.py`|BUILT|trainer verdict == contest oracle **to 11 decimals**|3.88e-11|the foundation (Catalog #392 STRICT)|
|29|byte-close pipeline #B1|`tools/witness_byte_close_and_eval.py`|BUILT|witness ckpt → int8+brotli archive.zip → MLX-free inflate → realized d_seg/d_pose|realized==verdict|the converge→row path|
|30|lossless rate stack (R1/R2/T1/S12/WRQ)|`bolton_inventory_and_stacking_plan_20260612.md`|BUILT (FP11-tied)|post-byte-close rate harvest|−0.005..−0.008 + WRQ|**grammar-specific** → needs witness-grammar materializer (anti-5)|

---

## ## SEG=POSE FUSION (the deepest synergy — the joint rate+pose collapse)

**The insight (reaudit R4 + #155 quotient):** the argmax partition's temporal evolution IS the ego-motion
that pose encodes — **seg and pose are the SAME information.** Coded separately we pay TWICE: a seg
representation (base_seg + per-frame deformation) AND a pose representation. Coded JOINTLY:

> `partition_t = warp(base_seg, pose_t) + sparse_object_residual_t`  — pose_t is reused AS the seg-warp.

**Quantified collapse:** base_seg (boundary geometry, #11 seg_core, **~10-20KB**) + pose trajectory (#1/#5,
~5KB → hundreds B via #2) + **per-frame fusion warp = 0 EXTRA bytes** (the warp IS pose, already stored) +
sparse object residual (moving cars not on the ego-warp). Task statistic **~15-40KB ⇒ rate 0.01-0.027**
vs frontier 0.118 and base_ch20 0.0594. **Pose term → ~0** (fused, not a separate sqrt term to pay). This
RELAXES the d_seg target from <7.3e-4 (corridor A) toward <1.1-1.3e-3 (corridor B) because the rate budget
freed by fusion can be spent on d_seg OR the d_seg term has more slack under S<0.15.

**Why #155/#22 forces it:** pose cannot be coded standalone below ~400B (it is 6 floats × 600 with ~2-3
effective DOF); the quotient orbit is task-sufficient ONLY when seg+pose share the carrier. State:
THEORY+PARTIAL — seg_core/partition/contour_codec BUILT; the warp-fusion training is the un-built capstone
representation (the highest-ceiling, highest-effort build; sister to bolton **T2** warp-residual-frame0).

---

## ## BINDING PLAN (composition order + dependency graph + joint expected S)

**Dependency graph (→ = requires):**
```
FOUNDATION (DONE):  B3 harness#28 ─┐   B1 byte-close#29 ─┐   B4 bc20 grounding ─┐
                                   └──────────── all downstream verdicts/rows ──┘
L0 priors (free):   comma2k19 GT#5/#16/#156 ──→ base_seg seed + pose trajectory
L1 pose (solved):   stored sidecar#1 (→ #2 shrink) ──────────────┐  (orthogonal)
L2 base_seg:        seg_core#11  ← needs L0 seed                  │
L3 FUSION#24:       warp(base_seg#11, pose#1)  ← needs L1 + L2    │  (rate win)
L4 witness (d_seg, BINDING):  realized RGB coord-INR renders L3 partition
        4a step-native#6  →  4b chroma#7  →  4c directional#8 (self-orient fixed-point)
                              →  4d capacity#9 (STRICT: AFTER basis) ← driven by 4e seg-saliency(#10 gap)
                              ;  4f R_surv#12 (in-loop)  ;  4g curriculum-fix
        optimizer: MD-Decoupling#27 (ablation if iso plateaus)
L5 frame0:          amortized-luma carrier#4  ← pose texture (PoseNet pair survives)
L6 finishing (0-byte): PR98#14 → LeverD#13 (R_surv-gated) → reverse-waterfill#26 (re-run)
L7 rate stack#30:   materializer port → R1/R2/T1/S12/WRQ
TERMINAL:           byte-close#29 → contest-CPU exact eval → pointer move
```

**Joint expected S (all bound):**
- **Corridor A (existence-grounded, fastest banker):** rate 0.0594 + pose 0.017 + d_seg<7.3e-4 (0.073) =
  **S ≈ 0.149**; sub-0.19 banks the moment d_seg<1.1e-3.
- **Corridor B (fusion rate-win):** rate 0.01-0.027 (#24) + pose ~0 (fused) + d_seg ~1.1-1.3e-3 →
  **S ≈ 0.12-0.14**.
- **Clean achievable S ≈ 0.10-0.14, binding term = SEG** (reaudit R4). Every wall except pose's √-flatness
  is EMPIRICAL/optimizer-bug, not fundamental (R2: 0/11 capacity walls proven fundamental).

---

## ## TOP-5 SYNERGIES TO BIND FIRST (highest ΔS-per-effort)

1. **Stored-pose sidecar (#1, BUILT, ~5KB)** — compose at byte-close. Instantly makes pose 0.017 SOLVED
   and frees 100% of witness capacity to d_seg. Zero training cost; pure compose. **Do first, always.**
2. **step-native#6 + chroma#7 on the realized RGB witness** — both flags ALREADY wired
   (`--activation hosc --chroma`); ONE training run on the B3 harness. Topology-matched (survives R) +
   chroma gives BOTH d_seg DOF and PoseNet texture (resolves the palette pose-blindness anti-3). This IS
   the decisive gate; the in-flight iso run is the degenerate (no-hosc, basis-isotropic) version.
3. **directional self-orientation fixed-point (#8 byte-closeable form)** — the $0 numpy de-risk on the
   existing `generator_n600.npz`: iso pass → own argmax → tangent → directional pass (tangent DERIVED,
   0-byte). If stable, unlocks the ~8× d_seg lever byte-closeably. **Decisive $0 gate BEFORE any GPU.**
4. **seg=pose FUSION (#24, THEORY+PARTIAL)** — the deepest synergy; collapses rate+pose to 15-40KB and
   relaxes the d_seg target. Highest ceiling, highest effort (the one new representation build); sequence
   after the iso banker proves the realized witness descends.
5. **MD-Decoupling (#27, BUILT --optimizer md)** — the principled fix for the measured "capacity walls"
   (R2: they were collapse / no-LR-transfer / warmup bugs). Swap-in ablation if the iso/hosc arm plateaus;
   removes the per-bc-size LR retune that was itself an artifact source.

---

## ## ORPHANED HIGH-EV FINDINGS (signal-loss risk — NOT first-class in the in-flight plan)

1. **seg=pose FUSION (#24)** — the DEEPEST synergy and highest ceiling, but the in-flight decisive run is a
   plain iso-RGB witness; fusion is in NO current GPU arm. At risk of being forgotten while the banker runs.
   **Fold as Layer 3 explicitly** (this memo + the binding spec are the durable anchor).
2. **T5 null-space-as-TRAINING-constraint** (bolton "strongest synergy", −0.01..−0.04 compounding) — train
   the witness to put error in the certified resize-null → free error + lower-entropy residual. NOT in the
   witness curriculum. ORPHAN.
3. **T2 warp-residual frame0 head** (−0.01..−0.03; frame0 is SegNet-blind) — regenerate frame0 from
   frame1+pose; the same temporal-amortization as #24, applied to frame0. Related to #4 but the warp-residual
   form is unbuilt. ORPHAN.
4. **Frozen-instance 0-byte priors compiled into inflate.py** (#16/#19/#20/#21/#149/#158) — measured-advisory
   / research, NEVER byte-closed onto a witness. The inflate.py-is-a-free-interpreter rule makes them 0-byte
   DETERMINISTIC priors from the KNOWN public comma2k19 source (NOT video-derived sidecars → not dead rate).
   High-EV, free, orphaned.
5. **exact-sensitivity KKT reverse-waterfill (#26, BUILT)** — measured optimal on the FRONTIER, but the
   witness sits at a different point; re-run on the witness archive may find real gains. ORPHAN for witness.
6. **comma2k19 ego-motion GT #5/#158** (downloadable) — grounds base_seg + pose at 0-byte; research-only,
   not integrated. ORPHAN.
7. **low-rank pose codec #140 (#2, PARTIAL)** — drops the sidecar ~5KB → hundreds B; small free rate.

---

## NO-FAKE / authority notes
- Every "BUILT" = code exists and is runnable; "MEASURED" numbers are `[advisory]`/`[macOS-MLX
  research-signal]` (or the one tagged `[contest-CPU]` for bc20) — non-promotable until byte-closed +
  contest-CPU exact eval. The directional −48% is PROXY (generator-argmax, no R, no SegNet-reseg) →
  realized-axis UNVERIFIED; its realized ΔS is the #3 $0 gate, not a banked win.
- Pointer UNMOVED **0.19110**. This memo BINDS the ingredients; it does not move the score. The pointer
  moves only when the bound witness byte-closes (#29) and a contest-CPU exact eval returns a lower row.

**Cross:** capstone binding spec `capstone_taskspace_step_witness_reorientation_and_binding_20260626.md` ·
`reaudit_refounding_and_md_decoupling_20260626.md` (R4 ranking) · `bolton_inventory_and_stacking_plan_20260612.md`
(T1/T2/T5/WRQ/S12 + grammar antagonism) · `vcm_theory_primitive_layer_20260619T033429Z.md` (#23) ·
`witness_capstone_deepmath_levers_20260625.md` (proxy levers) · DAG FEED-bo below.
