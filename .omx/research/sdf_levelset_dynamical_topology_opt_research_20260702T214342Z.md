# SDF / level-set + dynamical systems + topology optimization — scorer-grounded research + $0 n600 measurement (2026-07-02)

**Authority** `[macOS-CPU advisory] NON-PROMOTABLE` · **score_claim** false · **promotable** false ·
**ready_for_exact_eval_dispatch** false. Pointer UNMOVED: contest-CPU **0.19109982** / contest-CUDA
**0.20533003** (`.omx/state/canonical_frontier_pointer.json`). This is a MEANS (research + geometry
characterization), not a score. git `08c967bcd`. Operator directive 2026-07-02: *"Research SDF level set
and the dynamical system and topology optimization and all related and adjacent while designing and testing
and measuring optimal. And upstream modules.py and evaluate.py."*

Sisters: `[[unified-variational-levelset-flow-everything-is-facets]]` ·
`[[project_gr_unified_action_full_witness_architecture]]` · `[[analytic-lane-band-primary-authority-decomposition]]`
· `[[proactive-recall-consult-own-research-before-concluding]]` ·
`[[papers-checked-not-relevant-or-watch-item-ledger]]`.

---

## §1. SCORER GROUNDING (the authority — re-read line-by-line 2026-07-02)

`upstream/evaluate.py` + `upstream/modules.py` + `upstream/frame_utils.py`. The score is the ONLY thing a
lever must move; a lever that does not change a scored quantity is noise.

**The score (`evaluate.py:92`):**
`S = 100·segnet_dist + √(10·posenet_dist) + 25·rate`, where
`rate = compressed_size / uncompressed_size`, `compressed_size = (submission_dir/'archive.zip').stat().st_size`
(`evaluate.py:63` — ONLY the zip bytes; inflate.py/inflate.sh are NOT sized; NO time term, only the 30-min
budget), `uncompressed_size = 37_545_489` (sum of `videos/`). No `d_pose`/`d_seg` clamp; both averaged over
the 600 samples (`evaluate.py:90-91`).

**SegNet (`modules.py:103-113`) — the d_seg axis:**
- `smp.Unet('tu-efficientnet_b2', classes=5, activation=None, encoder_weights=None)`.
- Input framing: `x[:, -1, ...]` (`modules.py:108`) — **ONLY the LAST frame of the 2-frame seq is scored by
  SegNet.** Then bilinear `interpolate` to `segnet_model_input_size=(512,384)` i.e. `(H,W)=(384,512)`.
- `d_seg = (out1.argmax(1) != out2.argmax(1)).float().mean(spatial)` (`modules.py:112`) — the **argmax
  disagreement RATE** = the normalized AREA of the symmetric difference of two K=5 partitions. This is a
  **set/partition distance**: a hard, purely-combinatorial functional of the two argmax label maps. **Logit
  magnitudes are invariant** — only which class wins per pixel matters (the RGB-slack: any pixel value that
  keeps the same argmax is free).
- **Class-index order CONFIRMED (measured on the cached argmax, NOT luma-sorted)** —
  `gt_n96.npz['lstars']` per-class area / vertical-centroid: `0=Road (22.9%, vc0.617)`,
  `1=Lane (0.59%, vc0.614, THIN)`, `2=Undrivable/sky (49.3%, vc0.257 TOP)`, `3=Movable (1.24-1.56%, vc0.497
  mid)`, `4=MyCar/hood (25.4-25.6%, vc0.869 BOTTOM)`. Comma10k canonical order; the luma-sort
  `[Road,Lane,MyCar,Undriv,Movable]` is WRONG and bit us 3×.

**PoseNet (`modules.py:61-84`) — the d_pose axis:**
- FastViT-T12, `in_chans=12` = 2 frames × YUV6-6. `preprocess_input`: bilinear resize to (384,512), then
  `rgb_to_yuv6` (`frame_utils.py:51-78`): per frame → `[y00,y10,y01,y11,U_sub,V_sub]` (4 luma quincunx
  sub-samples + 2×2-box-averaged chroma). Both frames feed PoseNet (temporal). Normalize by
  `_mean=127.5, _std=63.75`.
- `d_pose = MSE(out1['pose'][:6], out2['pose'][:6])` (`modules.py:82-84`) — first 6 of the 12-dim pose head.

**Properties that constrain / enable an SDF/level-set representation (the design surface):**
1. **d_seg is a partition-set distance → the level-set / cartoon representation is the NATURAL chart.** The
   target is a piecewise-constant K=5 label field; its only content is the codim-1 class-boundary curves.
   `argmax_k φ_k` of K SDF fields is the topology-matched representation (already shipped:
   `lever_b_levelset_generator`).
2. **RGB-slack + argmax-only ⇒ huge null space.** Only the boundary location matters, and only to ±1 argmax
   flip. The witness is free everywhere the argmax is uncontested (the flat-interior "dark" region of the
   Fisher metric — `[[unified-variational-levelset-flow]]`).
3. **Stride-2 EfficientNet stem ⇒ sub-256 blindness**, but the scored argmax is at (384,512). The boundary
   that matters is the argmax boundary at the OUTPUT resolution; the round-trip R (bicubic↑874×1164 → uint8
   → bilinear↓512×384) low-passes it. SDFs are 1-Lipschitz so the zero-crossing shifts only O(blur),
   monotonically (the R-survival argument, `lever_b_levelset_generator`).
4. **Only frame-1 (last) is d_seg-scored** ⇒ the witness's d_seg job is a SINGLE-FRAME partition render;
   temporal coupling is pose-only (the se(3) screw ξ, `[[pose-solved-screw-twist-dual-use]]`).
5. **The measured binding residual is the ERASURE tail** — the thin/small/low-persistence features (class-1
   Lane dashes, class-3 Movable) are dropped below the argmax margin (error ∝ 1/persistence). This is a
   **topology-preservation** problem, which is exactly what the level-set/persistence/threshold-dynamics
   literature is built for.

---

## §2. THE MAPPED LEVER TABLE (facet ↔ method ↔ actionable? ↔ measured/DERIVE)

Facets per `[[unified-variational-levelset-flow]]`: **distortion**=boundary-geometry/Fisher-margin ·
**representation**=curvelet=rate · **curriculum**=annealing-flow=persistence-order · **pose**=se(3) screw ·
**compute**=MLX+Metal.

| # | Method (literature) | Facet | Already ship? | Actionable? | Measured Δd_seg/byte or DERIVE |
|---|---|---|---|---|---|
| L1 | **Multiphase MBO / threshold-dynamics** (Merriman-Bence-Osher; Esedoglu-Otto arbitrary surface tensions; ICTM 1904.10917) as a DECODE-TIME byte-free partition regularizer, class-pair-weighted σ_ij | distortion×rate | NO (new) | **YES** | **MEASURED label-space §3**: σ=1.0 removes 11.4% boundary at d_seg 0.00087, 95.7% Lane. High-margin (Road/Undriv/MyCar mutual) boundaries smooth ~free. DERIVE→needs through-R + byte-closed to be a MOVER. |
| L2 | **Curvature-as-margin surrogate** (from L1): boundary curvature magnitude ranks SegNet margin | distortion | NO (new) | **YES** | **MEASURED §3**: m_flip 0.13–0.55 ≪ m_keep 5.6, monotone in σ ⇒ a BYTE-FREE, SCORER-FREE geometric fragility prior. Candidate canonical equation. |
| L3 | **Volume-constrained / auction MBO** (Esedoglu auction dynamics, Jacobs-Merkurjev-Esedoglu) — enforce class-area at decode to stop minority erasure | distortion | NO (new) | YES (couples L1) | DERIVE. Directly cures the Lane(1)/Movable(3) area collapse measured in §3. |
| L4 | **Spatial-aware persistent feature matching** (Wen et al. 2412.02076, 2024) + **Betti matching** (Stucki-Bürgin-Paetzold-Bauer 2407.04683, 2024) — match birth-death pairs by SPATIAL location, not persistence value | curriculum/distortion | PARTIAL (`persistence_topology_loss` = soft-clDice+Betti-count) | **YES (upgrade)** | DERIVE. Exact thin-structure (lane-dash/movable) preservation; sharper than clDice/Betti-count. Training-time loss. |
| L5 | **α-curvelet / shearlet cartoon N-term** (Grohs-Kutyniok α-curvelets 1404.1043; Kutyniok-Lim shearlets 1002.2661) — provably optimal sparse basis for cartoon (piecewise-const, C² boundary) images | representation=rate | YES (curvelet front-end) | refinement | DERIVE. CONFIRMS shipped basis; only refinement = sweep anisotropy α to the measured boundary smoothness β (parabolic α=1/2 optimal for C² boundary). |
| L6 | **Phase-field / Modica-Mortola / Allen-Cahn** (multi-material phase field 1312.2356) — Ginzburg-Landau ε\|∇u\|²+(1/ε)W(u) perimeter penalty, ε→0 Γ-converges to sharp partition | curriculum=annealing×rate | implicit (CE→tau→l7→Muon IS this anneal) | validating lens | DERIVE. The ε-anneal = our curriculum; the perimeter term is a candidate TRAIN-TIME rate regularizer (perimeter = curvelet N-term = archive bytes). |
| L7 | **Level-set topology optimization** (Osher-Sethian; SIMP; Hamilton's-principle LSTO 2504.14892) — nucleation/merging of regions via level-set advection | distortion topology | N/A | LOW | DERIVE. Our target topology is GIVEN (the GT partition); we don't optimize a shape functional. Only the merge/split machinery (region birth) maps to the "target-region class birth" HiNeRV blocker — low EV vs L1-L4. |
| L8 | **Fast-marching / eikonal / narrow-band / reinitialization** (Sethian; Osher) — maintain \|∇φ\|=1 efficiently | compute × distortion | YES (eikonal loss + AA-SDF) | compute-only | DERIVE. Narrow-band = decode speed, not d_seg. Eikonal already in the θ* lever stack. |
| L9 | **DeepSDF / SIREN-SDF / neural implicit surface** (Park 2019; Sitzmann 2020) | representation | YES (coord-INR IS this) | — | The K=5 softmax-of-SDF witness IS a neural implicit multi-region SDF. Not new; the INR-NTK spectral bias is the low-pass we already characterized (`[[sig-proc-filter-chain-measured-R-allpass]]`). |

**Adversarial / settled (do NOT re-research):** curvelet/shearlet optimality (L5) CONFIRMS the shipped basis
(not a new lever); DeepSDF/SIREN (L9) IS the witness; phase-field (L6) IS the curriculum (a lens, not a
knob); LSTO (L7) optimizes an UNKNOWN shape — ours is GIVEN — so it is dominated by L1-L4 for our fixed
target. The genuinely NEW, actionable surface is **L1-L4** (threshold-dynamics decode regularizer +
curvature-margin prior + volume-constraint + spatial-aware persistence).

---

## §3. THE $0 MEASUREMENT — MBO curvature-flow RD probe on the cached n600 GT partition

**What.** Esedoglu-Otto multiphase MBO single step = one-hot the K=5 GT argmax `L*` → heat-diffuse each
indicator with a Gaussian (time τ=σ²/2) → argmax (the "threshold"). Sweeping σ traces the RATE
(boundary-length, a curvelet-N-term / chain-code proxy) vs DISTORTION (d_seg) curve of representing the
partition with a curvature-bounded (smooth) boundary. Cross-checked against the cached SegNet margin field.
Tool: scratchpad `mbo_levelset_rd_probe.py` (deterministic; scipy `gaussian_filter`; n=600, all pairs).

**Authority.** `[macOS-CPU advisory] NON-PROMOTABLE`, **LABEL-SPACE DIRECT** = a LOWER bound on the
through-R d_seg (the canonical direct≪through-R gap is ~2-3×; direct 0.0022→realized 0.0064 on the exact-L*
store) AND a rate PROXY (boundary length ≠ archive bytes). A geometry / dynamical-systems characterization,
NOT a score claim. Baseline: mean boundary-length/frame = 2700 edges; class-area-frac
`[0.232, 0.0059, 0.495, 0.0124, 0.254]`.

| σ (px) | d_seg (direct/label) | boundary-len reduction (rate proxy) | Lane(c1) area retention | Movable(c3) retention | mean margin FLIPPED px | mean margin KEPT px | flip-from-class dist |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.5 | 0.00000 | 0.000 | 1.000 | 1.000 | — | 5.612 | — |
| 0.75 | 0.00021 | 0.034 | 0.966 | 1.000 | 0.137 | 5.613 | **98.4% Lane** |
| 1.0 | 0.00087 | 0.114 | 0.859 | 0.999 | 0.322 | 5.616 | **95.7% Lane** |
| 1.5 | 0.00325 | 0.323 | 0.476 | 0.994 | 0.398 | 5.629 | 94.7% Lane |
| 2.0 | 0.00438 | 0.404 | 0.313 | 0.988 | 0.452 | 5.634 | 92.0% Lane |
| 3.0 | 0.00599 | 0.476 | 0.125 | 0.969 | 0.552 | 5.642 | 85.7% Lane |

**Findings (deep-math, cross-checked, triality-consistent):**

1. **Curvature flow reproduces the ERASURE crux via an INDEPENDENT (dynamical-systems) route.** The d_seg
   cost of smoothing is ~entirely class-1 Lane (85-98% of flips at every σ). Curvature flow annihilates the
   thin/high-curvature/high-perimeter-to-area Lane structures first — a quantitative confirmation of
   `error ∝ 1/persistence` (`[[birth-death-persistence-dseg]]`) from pure geometry, not the persistence
   filtration. Lane retention collapses 1.00→0.86→0.48→0.13 with σ; Movable (compact blobs) survives
   (1.00→0.97). **Isotropic smoothing = an erasure operator selective for exactly our binding residual.**

2. **CURVATURE MAGNITUDE RANKS THE SegNet MARGIN (the new cross-check, candidate canonical equation).** The
   pixels curvature-flow flips first have mean margin **0.13** (σ=0.75) rising monotonically to **0.55**
   (σ=3.0), vs kept-pixel margin **~5.6** — a **10-40× separation**, monotone in σ. Low-σ MBO touches the
   LOWEST-margin (most d_seg-fragile) boundary first. Since curvature is a BYTE-FREE, SCORER-FREE geometric
   quantity, this means **the witness can allocate its finest-scale capacity to the fragile boundary from
   pure boundary geometry — no stored margin field required.** (Adversarial caveat: boundary pixels are
   trivially lower-margin than interior; the sharper, defended claim is that curvature *magnitude* orders
   margin *within* the boundary — supported by the monotone m_flip(σ) 0.13→0.55.)

3. **The high-margin boundaries are SMOOTHABLE ~free ⇒ a class-pair-weighted (anisotropic) MBO is a real
   byte-free rate lever.** At σ=1.0, 11.4% of the total boundary length is removed while 95.7% of the d_seg
   cost is Lane. Set surface tensions σ_ij HIGH on the Road/Undrivable/MyCar mutual boundaries (m~5.6, big
   stable regions, high persistence → smoothing costs ≈0 through-R d_seg, saves rate/curvelet-N-term) and
   σ_ij ≈ 0 on Lane↔Road & Movable (preserve the fragile tail). This is a DECODE-TIME, byte-free operation
   (rule 118) the witness could apply after render. **Must be validated through R + byte-closed before it is
   a MOVER — the boundary-length reduction is a rate proxy, not archive bytes, and the direct d_seg is a
   lower bound.**

4. **Volume-constrained (auction) MBO is the decode-time cure for the minority erasure** — the Lane-area
   collapse in the table is exactly what Esedoglu's auction dynamics (area-preserving MBO) prevents by
   construction. Couples with L1.

5. **Scale coincidence worth noting:** the σ=1.0 label-space d_seg **0.00087** lands on the same ~1e-3 lane
   band as the openpilot analytic-lane-band floor (0.00087) and the AA-SDF real-frame ceiling (0.00086) —
   three independent routes agree the Lane residual sits at ~9e-4, consistent with the sub-0.19 gate need
   band (direct <9.2e-4→sub-0.19).

---

## §4. TRIALITY-CONSISTENT NOTES (candidate rows — for the sister agents / next DAG feed)

Nothing here moves the pointer (all DERIVE / label-space / rate-proxy). Candidate rows earned:

- **Candidate canonical equation `curvature_ranks_segnet_margin_v1`** (`tac.canonical_equations`):
  *"boundary mean-curvature magnitude is a byte-free, scorer-free surrogate for the frozen-SegNet top1-top2
  margin (fragility): pixels removed by MBO curvature flow at diffusion σ have mean margin monotone in σ
  (0.13@σ0.75 → 0.55@σ3.0) vs 5.6 for kept pixels."* Producers: this ledger + `mbo_levelset_rd_probe.py`.
  Consumers: the θ* finest-scale capacity allocator (allocate to high-curvature boundary),
  `laguerre_logit_offset` (per-class geometry). Sister of the measured Fisher-curvature↔(−margin) Pearson
  0.978 (`[[unified-variational-levelset-flow]]`) — this is the same correspondence read through the level-
  set curvature route.

- **Candidate DSL lever `mbo_decode_regularizer`** (`tac.witness_dsl`, DECODE-time, byte-free): a
  class-pair-weighted multiphase MBO/ICTM smoothing pass applied to the rendered witness partition inside
  inflate.py — surface tensions σ_ij high on {Road,Undrivable,MyCar} mutual boundaries, ≈0 on Lane↔Road and
  Movable; optional area-constraint (auction) on class-1/3. GATED: only a MOVER after a through-R +
  byte-closed A/B via `tools/levelset_byte_close_and_eval.py` (#202) — the net-S is #205-gated like the
  Wave-F lane band. Complements the shipped `aa_sdf_observation_render` (AA render) and
  `analytic_lane_render_band`.

- **Candidate persistence-loss upgrade** (`persistence_topology_loss.py`): swap soft-clDice+Betti-count for
  **spatial-aware persistent feature matching** (Wen 2412.02076) / **Betti matching** (Stucki 2407.04683) —
  match birth-death pairs by spatial location so the lane-dash / small-movable pairs get an exact
  preservation gradient. Training-time; feeds the l7/curvelet-finest stage of the curriculum.

- **DAG feed (`sub015_DAG_*`):** FEED — *MBO/threshold-dynamics (Esedoglu-Otto) mapped as a decode-time
  byte-free d_seg×rate lever; curvature↔margin correspondence measured (n600 label-space); L1-L4 are the
  new actionable surface, L5-L9 confirm/are-dominated. All DERIVE/label-space — pointer 0.19110 UNMOVED;
  next real step = through-R + byte-closed A/B of the class-pair-weighted MBO decode pass, #205/#202-gated.*

---

## §5. HONEST STATE

Pointer UNMOVED (0.19110). This unit produced: (a) a fresh line-by-line scorer grounding (the authority);
(b) an adversarial literature sweep (SDF/level-set, threshold-dynamics, topology-opt, phase-field,
persistence, cartoon-curvelet) mapped to our facets, separating the NEW actionable surface (L1-L4:
threshold-dynamics decode regularizer + curvature-margin prior + volume-constraint + spatial-aware
persistence) from the confirmed/dominated (L5-L9); (c) a $0 n600 measurement (MBO curvature-flow RD probe)
that independently reproduces the erasure crux, establishes the curvature↔margin correspondence, and
quantifies the class-pair surface-tension prior. **No lever here is a measured MOVER** — all are label-space
DIRECT (a lower bound) and/or rate PROXY (boundary-length ≠ archive bytes). The one that could become a
mover is L1 (class-pair-weighted MBO decode regularizer), gated on a through-R + byte-closed A/B (#202/#205,
owned by sister agents). Next unit aimed DIRECTLY at that exact-relevant row, not more characterization.

**Sources (research sweep):** MBO/threshold-dynamics — Merriman-Bence-Osher; Esedoglu-Otto "Threshold
Dynamics for Networks with Arbitrary Surface Tensions"; ICTM (arXiv 1904.10917); auction dynamics
(ScienceDirect S0021999117308033). Persistence — Wen et al. "Topology-Preserving Image Segmentation with
Spatial-Aware Persistent Feature Matching" (arXiv 2412.02076, 2024); Stucki-Bürgin-Paetzold-Bauer "Efficient
Betti Matching" (arXiv 2407.04683, 2024); Shit et al. clDice (CVPR 2021). Cartoon/curvelet — Grohs-Kutyniok
α-curvelets (arXiv 1404.1043); Kutyniok-Lim compactly-supported shearlets optimally sparse (arXiv 1002.2661).
Phase-field/topology-opt — multi-material phase field (arXiv 1312.2356); Hamilton's-principle LSTO (arXiv
2504.14892). SDF — DeepSDF (Park 2019); SIREN (Sitzmann 2020).
