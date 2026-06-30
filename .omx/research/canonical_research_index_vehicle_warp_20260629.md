# Canonical research index — the VEHICLE / WARP / GEOMETRY / SCENE-STRUCTURE axis (v2 task-space witness)

**Purpose (operator 2026-06-29, anti-signal-loss / proactive-recall):** a deduplicated, calibration-tagged
index of every MEASURED/BUILT/SOLVED/DEFERRED/OPEN finding on the v2 vehicle axis, so the deploy + the v2 codec
start at TRUE optimal form (perfect recollection), not rediscovered-from-scratch. This is a **MEANS** (a
consolidation memo). Pointer UNMOVED at contest-CPU **0.19110**; nothing here is a score until a byte-closed
exact row beats it.

**Scope:** the level-set φ witness / SDF carrier / coord-INR+Fourier+FiLM; the screw/twist SE(3) warp; the
stratified per-class homography; the screw-reach gate; the GR unified action (DM1/DM2/DM3); FiLM-collapse +
alternatives; the openpilot lane prior; the Morse-Smale partition codec; the grok depth×staticness warp; the
gauge meta-layer; the originality/borrowed-substrate accounting.

**Calibration legend (CRITICAL on this axis):**
- **pre-R** = label-space numpy probe, NO contest R operator (bicubic↑874→uint8→bilinear↓384) → a LOWER bound, GAP-2 blind.
- **through-R** = warp/render → exact ladder R → frozen **CPU-torch** SegNet/PoseNet argmax (the authority, NEVER MPS).
- **advisory** = macOS-CPU/MLX research-signal, `score_claim=false`/`promotable=false`, small-n (n6/n24/n96/n200), NOT 600-comparable.
- **exact** = real `inflate.sh → upstream/evaluate.py` byte-closed (still advisory unless contest-CPU/CUDA n600).
- **n=** the sample size — per FEED-kn, n CHANGES the outcome, not just the CI.
- Sub-0.15 d_seg pass-line (F4-grounded): **d_seg ≤ 1.23e-3** at the v2 byte budget (FEED-jc/jd).

---

## 1. INDEX TABLE (deduplicated; each finding = its canonical latest verdict)

### A. The WARP — screw/twist, stratified homography, through-R reality

| # | Finding | Status | Calibration | Pointer (FEED / tool / commit) |
|---|---|---|---|---|
| W1 | **Screw/twist SE(3) warp (Chasles) = ~0-byte WIN on the physical classes.** Reuses the already-stored 6-DOF pose; O(10) static params/clip vs ~6,600 per-pair per-class-homography params. Road reproduced EXACTLY by the shared twist; hood→identity, sky→rotation-only KRK⁻¹. | MEASURED-WIN (gauge cell `WarpGauge.SCREW_TWIST`) | pre-R advisory, n96≈n200 | FEED-jj (a513372a / tool `tools/measure_screw_warp_through_R.py` / `src/tac/se3.py`) |
| W2 | **Stratified per-class warp (the correct model):** Road=ground-homography(pose) [+15–17% d_seg, calibration CLOSES via EON intrinsics fx=fy=910 cx=582 cy=437 h=1.22m] · hood/MyCar=IDENTITY (static core #139) · sky/Undriv=rotation-only KRK⁻¹ (depth→∞) · Lane/Movables=learned residual. A SINGLE global homography is WRONG for the full image. | MEASURED (the depth×rigidity gradient) | advisory/pre-R, n96≈n200 | FEED-iz, FEED-ja (2f83e0b9e / `tools/measure_pose_warp_dseg.py`) |
| W3 | **Sky-divergence-null ablation:** adding the t-term back to sky HURTS (more under forward motion) → CONFIRMS depth-stratification (translation flow vanishes at ∞, LHP). | MEASURED (sign-of-effect) | pre-R advisory | FEED-jj |
| W4 | **Screw-warp THROUGH R = bulk is NOT free-via-warp.** Warping a neighbor inherits the inter-frame SegNet boundary-JITTER floor (~0.008). Bulk (Road+sky+hood) through-R d_seg = 0.0048/0.0051 = ~4× the 1.23e-3 budget. | MEASURED-NEGATIVE (robust) | **through-R** advisory, n96+n200 | FEED-jq (a23062c4 / 10f1dc908) |
| W5 | **Clean-canonical warp budget gate = NEGATIVE but reframed "wall → open rate question".** Clean canonical bulk through-R = 0.00291 (2.4×)/0.00427 (3.5×); removes only 15–37% of jitter. Bulk floor is REPRODUCIBLE-BY-STORAGE (PR95 reaches ~6e-4 by storing per-frame) → a RATE question, not a hard wall. | MEASURED-NEGATIVE (9-axis audited) | through-R advisory, n96/n200 disagree (pose-proxy inflates) | FEED-jz (6b13e65e9 / memo `clean_canonical_warp_budget_gate_20260629T203717Z.md`) |
| W6 | **EXACT-pose (comma2k19 GT) A9 overturn = negative CONFIRMED ROBUST.** Exact poses do NOT beat the proxy (0.00251≈0.00256); static-hood (warp-free) = 32% of bulk flips → floor is INTRINSIC texture-dependent per-frame SegNet jitter, NOT warp error. Naive margin-keyed dither = 177,926 B/600 = rate 0.118 (PR95-scale). "Free deterministic bulk + tiny trained lane" thesis DEAD for warp+naive-store. | MEASURED-NEGATIVE (linchpin: comma2k19 GT pose validated, rel_err 1e-4) | through-R advisory, n96 | FEED-kb (54c6f8287) |
| W7 | **Warp-carries-POSE gate: d_pose 190 was a UNITS BUG (zero-motion null), NOT a wall.** Real homography warp H=K(R−t·nᵀ/d)K⁻¹ carries pose 190→12.6 (−93%) at the d_pose-optimal calib (s_t≈+0.16). | MEASURED (bug fixed) | through-R advisory, n6≈n24 | FEED-lj (eb7d0c968 / `tools/measure_warp_dpose_through_R.py` / `experiments/v2_witness_byteclose_smoke.py`) |
| W8 | **DEEP CRUX: d_seg & d_pose demand OPPOSITE warp scales → lossy dual-use REFUTED.** d_seg-optimal s_t≈−0.0014 near-identity (argmax is flow-ROBUST, doesn't carry pose); d_pose-optimal s_t≈+0.16 WRECKS d_seg (0.0071→0.050, 7×). Best pure-warp net contribution ~16.2 = ~100× from sub-0.15, DOMINATED. → **pose stays on the STORED sidecar**; warp's REAL job = a RESIDUAL PREDICTOR for v2_det (calibrate to MINIMIZE residual ≈ geometric scale, NOT d_seg). | MEASURED-VERDICT | through-R advisory, n6≈n24 | FEED-lj |
| W9 | **Screw-REACH gate: the BULK partition is intrinsically STABLE for 47+ pairs (94 frames).** persist (NO warp) bulk d_seg stays ~0.006→0.022 across the whole window tested. The reach is carried by partition STABILITY, not the warp (d_seg-optimal warp ≈ near-identity, re-confirms W8). | MEASURED | through-R advisory, n96, ONE ~10s highway window | FEED-ll (7def780da / `tools/measure_screw_reach_through_R.py`) |
| W10 | **RATE consequence of W9: partition store is GREEN/cheap.** k*=47 → 13 keyframes → partition rate **0.0060** (+pose 875B = 0.0066). Even 10× conservative reach (130 keyframes) → 0.060. ALL ≪ store-everything partition wall 0.277 and ≪ frontier 0.191. One canonical partition = 809 B measured (~693 B/pair amortized). | MEASURED-WIN (rate axis) | through-R advisory, n96 | FEED-ll |
| W11 | **BUT the binding wall is the deterministic-render d_seg FLOOR, not rate.** R1 ≈ 0.0185 bulk / 0.023 full at k=0 = ~30–40× the sub-0.15 d_seg budget (~6e-4). A pure-deterministic materializer = cheap-rate but d_seg-DEAD (S≳2). → REDIRECT to the TRAINED amortized-residual generator for the d_seg floor, COMPOSING with the now-rate-de-risked deterministic substrate = the HYBRID. | MEASURED-VERDICT (NO-FAKE: k=0 reproduces R1 floor exactly 0.01851==0.0185) | through-R advisory, n96 bulk | FEED-ll, FEED-lk |

### B. The CARRIER / SDF / boundary-render

| # | Finding | Status | Calibration | Pointer |
|---|---|---|---|---|
| C1 | **Single-SDF carrier VALIDATED through R below threshold.** 1-Lipschitz SDF ramp: lane d_seg 5.9e-4 @192 / 1e-5 @320 — CLEARS ≤1.23e-3 at both render res. The carrier + R are PROVEN. | MEASURED-WIN (gauge cell `CarrierGauge.SINGLE_SDF`) | through-R advisory, n96 | FEED-jk (5290c937b) |
| C2 | **MSDF (multi-channel SDF) carrier FALSIFIED/dominated.** ~3.6× worse @192, ~76× worse @320 vs single-SDF (3 pseudo-distance channels non-1-Lipschitz → worse bicubic-R survival; lane is THIN/sub-Nyquist, not a corner problem → MSDF the corner-tool has no target). msdfgen port real (validated on synthetic corners, convex-corner bug fixed). | MEASURED-FALSIFIED | through-R advisory, n96, slopes {48,24,12} robust | FEED-jk |
| C3 | **Wide-SDF ramp (σ=1.0) is a FREE d_seg cure (0 extra bytes).** R0 flat-fill 0.0273/0.0242 → R1 ramp 0.0230/0.0185 = −16% full / −24% bulk at ZERO bytes. Texture (per-class mean+std+seed) HURTS full d_seg → it is boundary PLACEMENT (shape), not texture. The deterministic ramp DOMINATES the learned residual. | MEASURED-WIN | through-R advisory, n96 | FEED-lk, FEED-ku (segnet-fooling mechanism: sub-pixel BOUNDARY PLACEMENT, not texture/low-pass) |
| C4 | **"Fool SegNet" mechanism RESOLVED = sub-pixel boundary placement (the SDF).** Cheap synthetic renders FAIL; EOT-δ R-survival confirmed but per-frame δ expensive+low-yield. Magnitude bounded + unproven (seg = thin axis). | MEASURED (mechanism) | through-R advisory, n96/R3-n24 | FEED-ku, FEED-kv |
| C5 | **SDF carrier existence-proof cross-check** (graphics-AA ↔ astronomy deblending ↔ 4D-Hodge ↔ level-set-PDE): persistence≡deblending identity; ~0-byte warp parameterization win. | research-signal | macOS advisory | FEED-jg (a1da84c / 389f84f6f) |

### C. The LANE residual (the binding learned term)

| # | Finding | Status | Calibration | Pointer |
|---|---|---|---|---|
| L1 | **Road↔Lane (lane-marking) boundary IS the binding sub-0.15 residual (98–99% of flip mass).** openpilot deg-3 centerline IS the Road↔Lane separatrix (residual 1.9e-5) → a FREE geometric prior for φ1 (lane channel). | MEASURED | static-GT advisory, n96 | FEED-fr, FEED-fs (`scratchpad/feed_fs_roadlane*.py`, `src/tac/boundary_math/lane_sdf_component.py`) |
| L2 | **The FP over-paint residual = a HIGH-FREQUENCY per-row width residual** (centerline perfect, smooth-width exhausted, oracle ceiling 0.000019 = 30–47× headroom). The exact-ragged single-SDF contour (5.9e-4) beats the smooth-polynomial floor (0.00214) — the gap = the ragged ±1px detail = the trained generator's job. | MEASURED (diagnosis) | static-GT advisory, n96 | FEED-fs, FEED-jk reconciliation |
| L3 | **openpilot lane HEAD-START built + HONEST rate correction.** From-scratch 0.00586 → conditioned base 0.00207 (64.7% recovered); residual target ≤1.23e-3. **⚠️ base ~65 KB/600 image-space iid (rate 0.043), NOT 0.5–5KB** — adjacent-frame lane IoU only 0.284 (ego-motion moves the image-space lane every frame). The lane wants the GROUND-FRAME + screw-warp home (static lane in bird's-eye + warp by stored pose). | BUILT + MEASURED-NEGATIVE (rate) | $0 CPU, n96 + full-600 | FEED-jm (e68cf6a2f / `src/tac/boundary_math/lane_headstart.py` / `tools/build_lane_headstart.py`) |
| L4 | **openpilot world-model prior does NOT collapse the residual (3rd confirm: lane needs a TRAINED generator)** — gives a ~64% Wyner-Ziv head start. supercombo.onnx (49.1MB, sha verified) loads $0 CPU; lane-AGREEMENT number BLOCKED (3 named steps: EON medmodel warp, v0.9.7 offsets, back-projection) — no number guessed (NO-FAKE). | MEASURED-NEGATIVE-that-sharpens | advisory, n96 | FEED-jh (a99f41f0) |

### D. The GR unified action + DM1/DM2/DM3 conditioning

| # | Finding | Status | Calibration | Pointer |
|---|---|---|---|---|
| G1 | **The contest = ONE variational action S_τ = 100·d_seg + √(10·d_pose) + 25·rate**, stationarity δS/δφ=0 in the FIXED frozen-scorer Fisher metric G=E[JᵀFJ] = "matter on a FIXED curved background" (QFT-on-fixed-background, NOT full GR — no back-reaction → well-posed + measurable). Formalized E0–E12 into `tac.canonical_equations`. | SOLVED (framework, formalized) | derivation + 3 off-the-shelf theorems | FEED-ia, FEED-if |
| G2 | **Co-location CONFIRMED ×3: Fisher curvature ↔ (−margin) Pearson 0.978 (Spearman 0.908); boundary anisotropy 9.56:1; 96.8% flip-mass in a 2px band.** → the cheap top1−top2 MARGIN field is a byte-faithful surrogate for the Fisher metric. v2 loss geometry PINNED. | MEASURED | byte-faithful, b0bee924e | FEED-id |
| G3 | **DM1 (per-pair FiLM RANK / Stiefel + spectral-entropy) is byte-FREE BY CONSTRUCTION** (WᵀW=I isometry → PR(M=code@Wᵀ)=PR(cov(code)) exactly, PROVEN; Muon NS5 does NOT orthonormalize columns → proper cubic polar projection built). BUILT 07dd971d8 (default-OFF byte-identical, 513 tests). MEASURED +6.5× effR. | BUILT (byte-free) | unit-tested + measured PR | FEED-ht, FEED-ic, FEED-ih |
| G4 | **DM1 DEMOTED to SECOND-ORDER (NOT the binding d_seg lever).** EXACT $0 on per-stage ckpts: PR(M) collapses 2.6× (3.08→1.18) across CE→tau→l7 WHILE d_seg IMPROVES 1.9× (0.00593→0.00316); witness uses ~1.2 of 768 FiLM DOF. Algebra: per-pair FiLM only re-weights ≤192 FIXED channel patterns — cannot synthesize localized support → the moving lane annulus needs the spatial BASIS, not rank. DM1 decisive smoke MOOT. | MEASURED-VERDICT (deep-math settled) | EXACT $0 per-stage ckpts | FEED-ip (aa05397bb) |
| G5 | **v2 conditioning axis (corrected): DM2 (oriented byte-free curvelet/WIRE basis, the −48% d_seg lever, Candès-Donoho cartoon-optimal) + a low-rank GLOBAL additive code, NOT DM1, NOT a spatial grid.** | MEASURED-DESIGN | EXACT $0 | FEED-ip, FEED-ir |
| G6 | **DM3 per-position SPATIAL LATENT GRID FALSIFIED.** Cross-pair variation is globally LOW-RANK (rank-8 = 95.6%, ego-motion coherent) → a per-position grid is ~8× worse + ~100× bytes vs a working low-rank global code. (refined v2 = DM2 shared dict + per-pair low-rank-additive SDF-correction head, rank≈16, ~4-10KB/600 — but see G7.) | MEASURED-FALSIFIED ($0) | $0 necessary-not-sufficient (linear/pre-R) | FEED-ir, FEED-is |
| G7 | **The per-pair conditioning is NOT a learned head for the BULK — it's the DETERMINISTIC STRATIFIED POSE-WARP** (W2, grok-confirmed). GAP3 SETTLED: bulk needs NO trained INR. The DM3′-additive-head + spatial-grid are SUPERSEDED for the bulk; the trained INR shrinks to the Lane-survival + small movables residual. | MEASURED-VERDICT | $0 grok-test FEED-ja | FEED-ja (supersedes G6's additive-head for bulk) |
| G8 | **The annulus is a HOMOGRAPHY ORBIT; rank-8 = ground-plane homography 8-DOF.** class×distance×staticness = the rank decomposition; the lane long-tail SPLITS 3 ways. The modulation is a COORDINATE WARP, not amplitude-FiLM; FiLM failed by GROUP-ACTION MISMATCH; canonicalization unifies modulation+staticness. | DEEP-MATH (derivation) | derivation, connects to comma2k19 GT prior | FEED-it, FEED-iu |
| G9 | **THE GROK: the contest collapses to ONE object (ego-pose trajectory × static world); d_seg & d_pose are two readouts of the SAME sufficient statistic** → the stored pose IS a free d_seg modulation. (FIRST-ORDER; complete object = dynamic structure-from-motion w/ 3 named gaps.) Refined by W2/W8 (stratified; lossy dual-use refuted). | DEEP-MATH (grok), refined-down by measurement | derivation + $0 grok-test | FEED-iv, FEED-iw, refined by FEED-iz/lj |

### E. FiLM collapse + alternatives, emergent collapse

| # | Finding | Status | Calibration | Pointer |
|---|---|---|---|---|
| F1 | **FiLM rank-1.2 collapse = the MEASURED (2×-confirmed) d_seg-plateau cause** (multiplicative resonance). M2 launch-config measured: PR(raw codes)=4.57, PR(M)=1.19, PR(filmW)=7.44. → NEVER vanilla FiLM. | MEASURED | advisory, M2 measurement | FEED-lg, FEED-lm, FEED-ht |
| F2 | **Superior alternatives to FiLM (the #1 = the warp itself):** spatial-warp conditioning + modulation-split (COIN++/functa/D'OH free hypernet + tiny latent) + spectral-entropy/SinkGD; DM1 Stiefel-orthonormal W (byte-free, +6.5× effR) as the rank lever NOT capacity (A1/A2 per-layer/concat). | MEASURED-DESIGN | advisory | FEED-lg, FEED-lm |
| F3 | **EMERGENT LOW-DIM COLLAPSE = the smallest-representation IDEAL materializing.** Induce rank/spectral collapse to intrinsic dim (#110 / A6-nuclear-norm / code-spectral-entropy), store minimal-by-construction. Intrinsic dims: lane orbit rank-8 / coarse eff-rank 4.07 / pose rank-2 / FiLM ~1.2-of-768. Rank FLOOR guard (lane≥8). | DESIGN-PRINCIPLE | measured intrinsic dims | FEED-le |

### F. Morse-Smale / dynamical partition codec

| # | Finding | Status | Calibration | Pointer |
|---|---|---|---|---|
| M1 | **The witness argmax IS a soft Morse-Smale complex; the separatrix = margin-zero set (AUC 0.9987 separatrix-DETECTION).** argmax of 5 φ_k = additively-weighted power diagram in R⁵; 1-skeleton = Morse-Smale graph. Codim strata: codim-1 boundary (Lane+Movable annulus) / codim-2 triple points / codim-3 quad birth-death. | MEASURED (structure) | advisory, gt_n96 | FEED-fh, FEED-fj |
| M2 | **Morse-Smale + Neural-CA standalones DEFER (measured-dominated by the SDF witness); COMPOSE as a rate-lever + residual-sharpener.** AMBER NCA 0.00337 > SDF 0.00124 standalone. The margin-zero set is a FREE separatrix locator (0 bytes, rule-118) but separatrix-ARC coding never beats the trunk. | MEASURED-VERDICT (DEFER standalone) | advisory, through-R | FEED-fh, FEED-fl(ii) |
| M3 | **The one un-covered chart stratum = codim-2/3 Movable junction** (medial-axis ridge). Relabeled "irreducible" → "single-field + eikonal-tension, fixable by K>5 OR local-eikonal-relax." But FIX-B (local eikonal-relax at medial axis) DE-CONFIRMED: Movable INTERIOR-pixel miss = 0.0000 at every bandwidth → the eikonal tension is real but IRRELEVANT to d_seg (lives in blob interior, never flips argmax). | MEASURED (both framings WRONG) | advisory, gt_n96 | FEED-fp, FEED-fr (the actual binding residual = Road↔Lane, not Movable) |

### G. The gauge meta-layer + the vehicle synthesis

| # | Finding | Status | Calibration | Pointer |
|---|---|---|---|---|
| V1 | **Gauge meta-layer BUILT, TESTED, WIRED** (`src/tac/witness_dsl/gauge.py`, 30 tests, 76 green). Operationalizes the quotient codec #155: gauge-invariant base = scorer-equivalence quotient, gauge = cheapest fiber representative, coding = picking min-cost gauge. `fix_gauge → CANONICAL_GAUGE = warp:SCREW_TWIST · carrier:SINGLE_SDF · residual:CONDITIONAL_ON_LANE_PRIOR · pose:RANGE_DELTA · movables:STORE · generation:DETERMINISTIC_FREE`. Rejects non-compliant/non-deterministic charts BY CONSTRUCTION. | BUILT | $0 decision/observability infra | FEED-ji, FEED-jl (c1a878cec / 5cdd21c2b) |
| V2 | **The gauge layer, queried, names the ONE remaining binding probe = residual DIRECT_LEARNED (the trained-through-R lane residual = THE GPU run).** Everything else is selected/measured. | BUILT (the pointer-mover named) | $0 | FEED-jl |
| V3 | **SDS-TSC capstone vehicle (the synthesis):** 6 typed sections + integer decode (S0 calib header FREE · S1 canonical IPM scene in DM2 curvelet+eikonal-SDF L13 −59% ~8–25KB · S2 ego-pose stream FREE dual-use ~6.4KB · S3 per-class warp-type mask ~0.2–1KB · S4 Lane-survival residual through R = THE binding learned term ~6–20KB · S5 movables ~0.5–2KB). Total ~21–55KB (vs frontier 177KB / capstone 97KB). Predicted band S∈[0.12,0.17] (straddles sub-0.15), Dykstra-feasible. | DESIGN (build-gated, do NOT launch) | `pending_post_training`; predicted band | FEED-jb (memo `stratified_dynamic_sfm_taskspace_codec_design_20260629T182602Z.md`) |
| V4 | **F4 byte-budget: RATE HALF CLOSED (~0.0021); the contest collapses to ONE number — sub-0.15 ⟺ d_seg ≤ 1.23e-3.** K_machine(witness) ≈ 3.2 KB (131× < lossless). (⚠️ L3 corrects the lane-base figure up to ~65KB image-space; ground-frame coding restores the comfortable budget — OPEN.) | MEASURED (budget arithmetic) | advisory | FEED-jc, FEED-jd |
| V5 | **Movables (GAP1): multi-body = STORE-not-PREDICT, ~0.0008 d_seg, ~750B / store 2700B.** | MEASURED-BOUNDED | advisory, n96 | FEED-je, gauge cell movables:STORE |
| V6 | **3 REAL byte-closed exact-eval rows (apparatus WORKS, the `--batch-size n` $0 exact-eval trick).** store_raw S=53.91 (d=0/0, validates pipeline); v2_warp S=73.23 d_pose=190 (the W7 bug, now fixed); store_jpeg q40 S=1.38 d_seg=0.0021 (scorer robust to lossy content). SIMPLE per-pair forms RATE-DEAD 150–450× over 0.19, NOT PR95-adjacent. | MEASURED-EXACT (small-n advisory) | exact inflate.sh→evaluate.py, $0 CPU, n24 | FEED-kx (f1ead4fd5 etc.) |

### H. Originality / borrowed-substrate accounting (NO-FAKE #7)

| # | Finding | Status | Calibration | Pointer |
|---|---|---|---|---|
| O1 | **v2 = the UNOCCUPIED INTERSECTION of (A) driving-scene recon-with-warp (PSNR objective) + (B) codecs-for-machines (black-box, no geometry).** Closest single neighbor = "Implicit Neural Video Compression" (arXiv 2112.11312) = our skeleton but pixel-fidelity/learned-warp/no-task/no-SDF. **HONEST CLAIM: "a novel COMPOSITION of known prior art," NOT a new primitive.** UNVALIDATED design-originality (MEANS not ends). | RECORDED (provisional) | $0 lit sweep, 7 families ~50 arXiv IDs | FEED-js/jt/ju (981e07f4e / memos `related_codec_sweep_v2_*`, `v2_originality_provenance_synergy_citations_*`) |
| O2 | **The 5 genuinely-OURS elements:** (1) distortion = exact frozen-oracle hard argmax CELL (indirect-RD), not PSNR; (2) physical per-class SE(3) screw warp (depth-stratified, ~0-byte from stored pose); (3) SDF carrier validated by SCORER-SURVIVAL not rendering fidelity; (4) warp→SDF→openpilot-Wyner-Ziv residual chain w/ stored-jitter/generated-structure SPLIT; (5) gauge-as-codec-canonicalization for MINIMUM DESCRIPTION LENGTH (literature canonicalizes for accuracy; for RATE appears new). | RECORDED | provisional | FEED-ju memo |
| O3 | **Overfit/generalize split (operator directive):** the WITNESS (counted payload) OVERFITS this clip (fine/PREFERRED); the IMPLEMENTATION (free generic algorithm in inflate.py) GENERALIZES. = rule-118; FORBIDDEN = smuggle clip-payload into "code" (NO-FAKE #6). | RECORDED (binding) | design principle | FEED-ka, FEED-js |

---

## 2. ⭐ OPTIMAL-CONFIG CONTRIBUTION — the measured-optimal VEHICLE COMPOSITION

This is the marshaled best — the HYBRID Pareto solution the whole arc converges to, every measured winner
included, every measured loser excluded. It is the canonical gauge (V1) made concrete with the latest verdicts
(W8/W9/W11/G4/G7 corrections folded in). **It is a DESIGN at optimal form; not yet a score — the binding GPU run
(S4) is the only unmeasured cell on the canonical path (V2).**

**THE VEHICLE = deterministic substrate (rate de-risked) + ONE trained residual (the d_seg floor), composed:**

- **S0 — calibration header (FREE/tiny).** EON intrinsics (fx=fy=910, cx=582, cy=437, h=1.22m) + ~tens of B
  fitted globals (`src/tac/camera.py`). Generic algorithm in inflate.py; per-clip globals counted-tiny.
- **S1 — ONE canonical static scene partition (~8–25 KB counted).** Stored as the **SINGLE-SDF carrier (C1, WIN)**
  with the **wide-SDF ramp σ=1.0 (C3, FREE −24% bulk d_seg)** — NOT MSDF (C2, falsified). Rendered in the
  **DM2 oriented byte-free curvelet/WIRE basis (G5, the −48% lever)** + eikonal-SDF L13 −59% format.
- **S2 — ego-pose stream on the STORED SIDECAR (~875 B, d_pose≈0; W8/FEED-lj verdict).** Pose is NOT carried
  lossily by the warp (dual-use refuted W8); it is stored (Quantizr-style 6-scalar target, `src/tac/scorer_targets.py`,
  low-rank rank-2 #140). RANGE_DELTA gauge cell. It IS still dual-use in the FREE sense (the same stored pose
  drives the deterministic warp at decode for d_seg, at zero extra bytes).
- **S3 — per-class warp-type mask + STRATIFIED screw-warp (W1/W2, ~0-byte + ~0.2–1 KB).** At decode, warp S1's
  canonical partition forward via the **stratified per-class SE(3) screw** (`src/tac/se3.py`,
  `tools/measure_screw_warp_through_R.py`): Road=ground-homography(pose), hood/MyCar=IDENTITY (#139 static core),
  sky/Undriv=rotation-only KRK⁻¹. Calibrated to MINIMIZE the residual (≈ the geometric/d_pose scale), NOT d_seg
  (W8). **Reach is carried by partition STABILITY (W9): k*=47 → ~13 keyframes → partition rate 0.0060** (W10).
- **S4 — the Lane-survival residual through R (THE BINDING LEARNED TERM, ~6–20 KB).** The ONLY real trained
  payload = the amortized level-set residual generator (`train_levelset_witness_realized_through_R_mlx`) that
  emits the ragged ±1px Road↔Lane contour detail (L1/L2) the deterministic prior can't (smooth base 0.00207 →
  target ≤1.23e-3). Conditioned on the **openpilot deg-3 centerline prior (L1/L3, FREE φ1 seed, separatrix
  residual 1.9e-5)** but coded in the GROUND-FRAME + screw-warp home (L3 rate correction). FiLM is REPLACED by
  spatial-warp conditioning + DM1 Stiefel-W rank lever (byte-free, F1/F2/G3), NEVER vanilla FiLM (collapse).
- **S5 — movables residual (STORE-not-predict, ~0.5–2 KB, d_seg ~0.0008; V5).**
- **Decode** = deterministic ANS/Ballé integer networks, bit-identical CPU/CUDA, NO scorer weights in archive
  (compliance + determinism non-negotiables).

**Why this is the optimum (the 3 corrections that make it optimal-form, not naive):**
1. **RATE is de-risked CHEAP (W10): do NOT build a pure-deterministic materializer** — it is d_seg-DEAD (W11,
   R1 floor ~0.0185 = 30–40× budget, S≳2). The deterministic substrate's JOB is to make the rate axis free so
   the trained budget concentrates on S4.
2. **Pose on the sidecar, warp as residual-predictor (W8): lossy dual-use is refuted** — a single global warp
   can't serve both d_seg and d_pose. The warp's value is shrinking the v2_det residual, not carrying pose.
3. **DM2 + low-rank-global + warp, NOT DM1/FiLM/spatial-grid (G4/G6/G7):** the bulk needs NO trained INR
   (the stratified pose-warp IS the per-pair conditioning); the trained INR shrinks to S4 only.

**The Pareto SET the operator mandated (not one point — FEED-km/kj/kk):**
- **DETERMINISTIC corner** = S0–S3+S5, no S4 trained residual. Cheap rate, fast prep, browser-portable; but
  d_seg-DEAD (S≳2). Value = the rate-de-risk substrate + the fast-prep / generalization / reviewability corner.
- **HYBRID** = the full stack above (deterministic substrate + S4 annulus neural residual). The min-S candidate.
- **NEURAL** = the level-set witness alone (the paused GR-unified-action capstone). Highest prep cost.
- The gauge layer (V1) picks the chart per objective; `compose_pareto_frontier` traces the curve. **MAP THE
  CURVE, not one point** — the witness RD-curve (advisory/through-R) projects ~89KB→S0.216, B*~122KB
  optimal-form+directional → S0.134 (sub-0.15); the optimum is NOT the current point.

**Measured anchor (the calibration row, pointer unmoved):** g3 torch_vehicle bc20 byte-closed DUAL exact row
89244 B → [contest-CPU] 0.37797 / [contest-CUDA] 0.39153 (`g3_torch_vehicle_bc20_first_exact_row_*`).

---

## 3. OPEN / HEADROOM (the unmeasured cells, ranked by EV-to-sub-0.15)

1. **S4 — the trained-through-R Lane-survival residual = THE GPU run (V2, the named pointer-mover).** Does the
   amortized generator emit the ragged ±1px contour (target ≤1.23e-3) at a cheap conditioned code that beats
   PR95's rate? UNTESTED. The single binding gate. (DEPLOY AUTHORIZED operator 2026-06-29 after 3 gates clear:
   indep 3-clean review / deep-math ✅ / config pass; from-scratch openpilot-seeded; full config in
   `[[session-20260630-review-warpfix-lossless-exhausted-CURRENT]]`.)
2. **Is the per-frame SegNet JITTER predictable/compressible from local content (W6 open door)?** If predictable
   → a conditioned generator emits it cheaply (door open). If ~white SegNet decision-noise → even a trained
   generator can't (door mostly closed). The precursor $0 probe to S4. (FEED-kb spawned it.)
3. **GROUND-FRAME (bird's-eye) lane coding rate (L3): does static-lane + screw-twist warp hit 0.5–5 KB** (vs the
   measured 65 KB image-space iid)? The rate-half closure for the lane. Plus the v2_det residual BYTES under the
   geometric-calib warp (W8 next gate) + the FEED-li full-drive screw-reach (W9 caveat: only ONE 10s window
   tested; turns/traffic untested).

---

## 4. CONFLICTS / SUPERSEDED (cite the LATEST verdict to avoid re-litigating a dead branch)

- **"Free deterministic bulk via warp" → REFUTED ROBUST** (W4→W5→W6). The bulk carries the same per-frame
  texture-jitter wall (~2–4× budget) as the lane. Latest: FEED-kb. The thesis survivor = warp/SDF/openpilot
  PRIOR conditioning a TRAINED generator (closer to the hybrid).
- **Grok "stored-pose = FREE LOSSY dual-use d_seg+d_pose carrier" → REFUTED for the lossy v2_warp arm** (W8).
  d_seg & d_pose want opposite warp scales. Pose stays on the STORED sidecar; warp = residual predictor. (The
  grok survives in the FREE sense: stored pose drives the deterministic warp at 0 extra bytes.) Latest: FEED-lj.
- **DM1 (per-pair FiLM rank / Stiefel) as the binding d_seg lever → DEMOTED to second-order** (G3 built but G4
  demoted). PR collapses while d_seg improves. The DM1 decisive smoke is MOOT. Latest: FEED-ip.
- **DM3 per-position SPATIAL LATENT GRID → FALSIFIED** (G6). Variation is globally low-rank (rank-8=95.6%).
  Refined to DM2 + low-rank-global; then SUPERSEDED again for the bulk by the deterministic stratified warp (G7).
  Latest: FEED-is → FEED-ja.
- **DM3′ per-pair low-rank-additive SDF head for the BULK → SUPERSEDED** by the deterministic stratified
  pose-warp (G7; bulk needs NO trained INR). Latest: FEED-ja. (The additive head may still apply to the residual
  long-tail — OPEN probe in the FEED-is note.)
- **MSDF carrier → FALSIFIED/dominated** (C2) by the single-SDF (C1). Latest: FEED-jk.
- **Morse-Smale / Neural-CA / Kuramoto STANDALONES → DEFER (measured-dominated)**; compose only as rate-lever +
  residual-sharpener (M2). Latest: FEED-fh/fl.
- **"Movable medial-axis is the IRREDUCIBLE chart gap / the final wall" → OVER-LABELED** (M3). The actual
  binding residual is the Road↔Lane separatrix (L1, 98–99%), not Movable. FIX-B (eikonal-relax) DE-CONFIRMED.
  Latest: FEED-fp/fr.
- **"Lane base rate ~0.5–5KB / rate-half solved 0.0021" → CORRECTED UP to ~65KB image-space iid** (L3/V4);
  ground-frame coding is the open restoration. Latest: FEED-jm.
- **"Free bulk + tiny trained lane budget ~21–55KB closes sub-0.15" (V3 SDS-TSC band [0.12,0.17])** is the
  DESIGN band, NOT a measured row — and the "free bulk" premise it rested on is REFUTED (W6). The vehicle is
  really deterministic-substrate (rate-free) + a TRAINED generator for the d_seg floor across the annulus.
  Latest: FEED-kb consequence + the §2 optimal-config corrections.
- **Calibration cautions (own misses, OWNED):** the "near-certain sub-0.19 finishing-kit" was a DOUBLE-COUNTED
  estimate (lossless rate on 0.19110 is EXHAUSTED, byte_delta=0 — FEED-lb); n-amortization optimism RETRACTED
  (keyframe cost GROWS with n on a driving clip — FEED-kn). RULE: verify a load-bearing estimate against the
  CURRENT state before acting on it.

---

**Cross-refs:** `[[gr-unified-action-full-witness-architecture-20260629]]` ·
`[[v2-novel-contribution-originality-accounting-20260629]]` · `[[witness-dsl-and-dag-dsl-equations-triality-20260629]]` ·
`[[session-20260630-review-warpfix-lossless-exhausted-CURRENT]]` · `[[proactive-recall-consult-own-research-before-concluding-20260630]]` ·
DAG FEED chain fh→ll in `.omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`.
**MEANS≠ENDS — pointer UNMOVED contest-CPU 0.19110; this index is a MEANS, only a byte-closed exact row below it is the end.**
