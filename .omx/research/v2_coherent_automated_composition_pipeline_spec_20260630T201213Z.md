# v2 Witness — Coherent, Optimal, Automated STORE / LEARN / POSE-SIDECAR Composition Pipeline Spec

**Date:** 2026-06-30T20:12:13Z · **Author:** senior research engineer (CPU-only design pass) ·
**DAG:** sister of FEED-iz / FEED-ja / FEED-lj / FEED-ll / FEED-lf (the warp + reach + budget thread).

**Authority / status (NO-FAKE supreme rule + means≠ends).** This is a **SPEC** — a design artifact
(a MEANS). It moves nothing. The frontier pointer is **contest-CPU 0.19110 and UNMOVED**; it moves ONLY
on a byte-closed `archive.zip` exact row from `upstream/evaluate.py` (BOTH `--device cpu` AND `--device
cuda` on contest-compliant hardware, NEVER MPS). Every architectural claim below is tagged with the
MEASURED artifact it rests on, and every number is `[macOS advisory / research-signal] NON-PROMOTABLE`.
This pass touched no GPU, launched no training, and did not disturb the live n600 run (pid 38641) or the
dashboard.

**Operator question this answers (2026-06-30):** *"at what point are we deterministic to store versus what
to learn and at what point is pose sidecar built[?] all must be coherent and optimal and automated."*

---

## 0. The grounded architecture (the resolved STORE / GENERATE / LEARN / POSE boundary)

The session's measurements RESOLVED the three tensions that had made the boundary ambiguous. Stating the
resolved facts up front, because every pipeline step depends on them:

**FACT 1 — RATE is GREEN; it is NOT the binding wall (FEED-ll, `experiments/results/screw_reach/reach_n96.json`).**
The bulk SegNet partition is intrinsically *stable*: one stored canonical partition keyframe persists
(through R, frozen CPU-torch SegNet, n96) for **k\* = 47 pairs (≈94 frames)** before bulk d_seg crosses 2× the
R1 floor. → **13 partition keyframes** for the 600-pair clip → partition rate **0.0060**; + pose sidecar
~875 B = **0.0066 total**. Even a 10× conservative reach (130 keyframes) → 0.060. Both ≪ the store-everything
partition wall (0.277) and ≪ the frontier (0.191). The deterministic store-canonical + warp substrate is
**cheap**.

**FACT 2 — the BINDING WALL is the deterministic-render d_seg FLOOR (FEED-ll / FEED-lk, `segnet_fooling_ladder/ladder_n96.json`).**
The FREE wide-SDF-ramp (R1, σ=1.0, 0 bytes) cures boundary placement to **d_seg ≈ 0.0185 bulk / 0.023 full**.
That floor is **~30–40× the sub-0.15 d_seg budget (~6e-4 … 1.4e-3, `break_even_d_seg`)**. A pure-deterministic
materializer is cheap-rate but d_seg-DEAD (S ≳ 2). → **the trained residual generator is REQUIRED for d_seg**,
but ONLY for the residual the deterministic bulk leaves, NOT the whole partition.

**FACT 3 — pose stays on the STORED SIDECAR; the warp is dual-use only as a d_seg RESIDUAL PREDICTOR, not a
lossless pose carrier (FEED-lj, `warp_dpose_through_R_n{6,24}/results.json`).** d_seg and d_pose demand
**OPPOSITE warp scales of the same homography**: the d_pose-optimal scale (s_t≈+0.16) drives d_pose 190→12.6
but WRECKS d_seg (7×); the d_seg-optimal scale is near-identity. A single lossy global warp cannot serve both
→ the "free dual-use warp" grok is REFUTED for the lossy arm. **Pose → store 6 scalars/pair (d_pose ≈ 3.4e-5).
The SAME stored pose then drives the per-class warp for the d_seg residual at ~0 extra bytes** (the warp is
a FREE generic algorithm; only calib scalars + a per-class warp-type mask are counted). "Dual-use" survives
in this weaker, real sense: one stored pose, two free read-outs (direct d_pose; warp-as-d_seg-predictor),
calibrated independently.

**FACT 4 — the warp is STRATIFIED PER-CLASS (FEED-iz / FEED-ja, `grok_pose_warp_dseg_*`).** A single global
homography is wrong. Measured per-class (advisory/pre-R, EON intrinsics fx=fy=910 cx=582 cy=437 h=1.22m):
- **Road** = ground-homography(pose): **+15%** d_seg vs persist (calibration CLOSES) — warp HELPS.
- **MyCar/hood** = **identity** (static core #139; ground-warp DESTROYS it −525%).
- **Undrivable/sky** = rotation-only `KRK⁻¹` (depth→∞; ground-warp mis-warps −9…−43%).
- **Lane** = learned survival residual (warp can't help, −1%; the binding wall, R-survival = open GAP2).
- **Movables** = small learned residual (~0.0008 area, independent motion, not warp-reducible).

**The boundary is KNOWN deep-math, not discovered (the framing that matters).** The STORE-vs-LEARN split is
ALREADY DETERMINED by the convergent measured deep-math (GROK REFINEMENT-2 + warp-through-R + indirect-RD/MDL
+ intrinsic-dim ~9 + GAP3-settled "the bulk needs NO INR"). It is an **established optimum**, parameterized by
ONE generalizable per-class question: **is this class geometrically/causally RECOVERABLE from the stored pose
(the depth×rigidity gradient, MEASURED)?** → if yes, STORE/GENERATE; if no (not pose-recoverable AND doesn't
survive R), LEARN. This rule is **clip-agnostic**: the per-class recoverability is the measured warp-through-R
gradient, so it transfers to any dashcam clip/corpus (re-measure the per-class recoverability, the split FORM
is fixed). The current n600 full-partition INR is **NOT the optimal vehicle** — the **HYBRID (deterministic
bulk + small residual INR + dual-use pose) IS THE optimal vehicle and the rate-win path to sub-0.15**. The
baseline's only roles: (a) the d_seg distortion FLOOR (the upper bound the hybrid must match/beat), (b) a
warm-start checkpoint, (c) empirical CONFIRMATION of the known split. The split is not learned FROM it.

**The boundary, stated once (the answer to the operator's question):**

| Tier | What | Where the bytes go | Mechanism |
|---|---|---|---|
| **STORE** (counted, tiny) | 13 canonical partition keyframes · per-class warp-type mask · 3 calib scalars · 6 pose scalars/pair | `archive.zip` (~9–10 KB partition + ~0.9–5 KB pose) | contour-codec (LZMA label map) + scorer_targets |
| **GENERATE** (FREE, rule-118) | per-class stratified warp (homography from pose+calib) · SDF rasterizer · R1 σ=1.0 ramp · openpilot lane centerline geometry · deterministic Fourier basis | `inflate.py` code (0 counted bytes) | se3/camera/lane_sdf/contour_codec forwards |
| **LEARN** (counted, the open size) | the residual ONLY: Lane-survival annulus (R-survival) + small movables | `archive.zip` (small INR weights, ≪ the 90 KB full-partition INR) | shrunk level-set INR, residual target |
| **POSE-SIDECAR** (counted, tiny) | 6 PoseNet scalars/pair (→ d_pose directly AND drives the warp) | `archive.zip` (~0.9–5 KB; ~2 KB at rank-2 #140) | `scorer_targets` + `qp1_pose_codec`/`pose_from_embedding` |

This is the coherent decomposition: **the bulk (Road/sky/hood = the rank-8 homography orbit + static
classes, the large-area majority) is STORED-once + GENERATED-per-frame deterministically; the binding residual
(Lane-survival + movables) is LEARNED; pose is STORED and re-used for free.** The single open quantity is the
LEARN tier's byte cost — settled only by the GPU residual run.

---

## 1. STEP-BY-STEP PIPELINE (6 steps; INPUT → OPERATION → OUTPUT [FREE vs COUNTED] → BUILT vs NEEDS-WIRING → automation)

### Step 1 — Baseline (the d_seg FLOOR + warm-start + CONFIRMATION of the known split)  ·  **BUILT**

**Role (demoted, per the framing above):** the baseline is NOT the source of the store/learn boundary (that
is known deep-math, §0). Its three roles are (a) the **d_seg distortion FLOOR** the hybrid must match/beat
(the upper bound on what LEARN must achieve), (b) a **warm-start checkpoint** for the residual INR, (c)
**empirical CONFIRMATION** of the known split (the per-stage attribution should show exactly that the bulk
(Road/sky/hood) is stage-stable/pose-recoverable while the Lane annulus + movables are the persistent
residual — confirming, not deciding).

- **INPUT:** the current n600 run dir (per-stage EMA-shadow checkpoints CE→tau→l7→Muon) + the clip GT cache
  (`experiments/results/mlx_fleet_gt_cache/gt_n600.npz`).
- **OPERATION:** render each per-stage checkpoint through the EXACT R operator (bicubic↑874 → uint8 →
  bilinear↓384) + the frozen CPU-torch SegNet argmax; diff argmax across stages per pixel / per class /
  per region.
- **OUTPUT (analysis only, 0 archive bytes):** the d_seg floor; WHERE each stage fixes d_seg; the residual
  no stage corrects (expected = Lane annulus + movables, CONFIRMING the known LEARN tier); per persistent-wrong
  residual, a recommended repair ∈ {LEARN / STORE / DETERMINISTIC / UNIWARD} — used to CONFIRM/calibrate the
  known split, not to derive it.
- **BUILT:** `tools/witness_per_stage_annulus_attribution.py` (real render-through-R + real CPU-torch SegNet,
  canonical class order, NO-FAKE selfchecks). The *observability* substrate that confirms the split + sizes
  the residual target.
- **Automation:** `witness_per_stage_annulus_attribution.py --run-dir <baseline> --gt-cache <npz>` → JSON.

### Step 2 — ENCODE the KNOWN store-vs-learn split (NOT a discovery step)  ·  **NEEDS-WIRING** (inputs all BUILT)

**This is NOT a decision/discovery step.** The split is the established deep-math optimum (§0). Step 2 is the
deterministic, generalizable **ENCODING** of that known split as a per-class recoverability rule, parameterized
by the MEASURED warp-through-R recoverability so it transfers to any clip/corpus. The function returns the
known optimal assignment; the attribution (Step 1) only CONFIRMS it + sizes the residual target.

- **INPUT:** (a) the per-class warp-through-R recoverability deltas (FEED-iz/ja: Road +15%, Lane −1%,
  sky/Undriv −9…−43%, hood −525%, movables ~0.0008) from `tools/measure_pose_warp_dseg.py` +
  `tools/measure_screw_warp_through_R.py` — the GENERALIZATION parameter; (b) the reach k\* from
  `tools/measure_screw_reach_through_R.py` (keyframe count = ceil(600/k\*)); (c) the Step-1 attribution
  (CONFIRMATION + residual sizing only).
- **OPERATION — the concrete automated function (encodes the known split, parameterized for any clip):**
  ```python
  # proposed: src/tac/v2_compose/store_learn_split.py
  def encode_known_split(
      warp_through_R: dict,         # per-class persist→warp recoverability (the generalization param)
      reach_kstar: int,            # screw-reach k* (FEED-ll) → keyframe budget
      attribution: dict | None = None,  # CONFIRMATION + residual sizing ONLY (not the decision source)
      uncompressed_size: int = 37_545_489,
  ) -> StoreLearnAssignment:
      """Emit the KNOWN-OPTIMAL per-class {STORE, GENERATE, LEARN} assignment + predicted bytes.

      The split is established deep-math (GAP3-settled). The ONLY per-class question is
      pose-recoverability (the depth x rigidity gradient, MEASURED) -- clip-agnostic:

        recoverable_from_pose(class)  ->  GENERATE  (deterministic warp; 0 counted bytes beyond calib)
          * Road           : ground-homography(pose)            (recoverable, +15%)
          * hood/MyCar     : identity (static, temporal-IoU>=0.99)
          * sky/Undrivable : rotation-only KRK^-1 (depth->inf)
        bulk partition stable for k* pairs ->  STORE ceil(600/k*) keyframes (contour-codec)
        NOT pose-recoverable AND R-fragile ->  LEARN  (small residual INR)
          * Lane-survival annulus (~8-dim orbit / per-dash high-rank, R-survival wall)
          * small movables (~0.0008, independent motion)

      Predicted bytes per tier via contest_score.rate_term. If `attribution` is given,
      ASSERT it confirms the known split (the bulk is stage-stable, the residual is
      Lane+movables) and use it only to SIZE the residual target -- never to override
      the known assignment (a divergence is a finding, not a re-decision).
      """
  ```
  Returns: `{class -> (decision, predicted_bytes, rate_contribution)}` + the residual-target spec for Step 4
  (which pixels/classes the residual INR must carry) + the assembled rate budget (Σ over STORE+LEARN+POSE).
- **OUTPUT:** the known-optimal per-class assignment + predicted archive-byte budget (0 bytes; a plan).
- **BUILT:** all INPUT generators (the warp-dseg tools, the reach tool, the attribution tool);
  `contest_score.rate_term` is the canonical byte→score helper.
- **NEEDS-WIRING:** the encoding FUNCTION itself — a single callable that emits the known split parameterized
  by the measured per-class recoverability + predicted bytes. **This is the keystone seam** (composition glue,
  not new science: the split is already known).
- **Automation:** `encode_known_split(...)` called inside the PHASE-A entry point (§4).

### Step 3 — Deterministic bulk generation (STORE + GENERATE; rule-118 boundary)  ·  **components BUILT, composition NEEDS-WIRING**

- **INPUT:** the 13 canonical partition keyframes (chosen at the reach-determined spacing) + the stored
  6-DOF pose/pair + the 3 calib scalars + the per-class warp-type mask.
- **OPERATION (the FREE generic algorithm, runs in inflate.py):** for each target pair, compose the
  cumulative ego-motion to the nearest keyframe → per-class stratified warp of the keyframe label map
  (Road = `H = K(R − t·nᵀ/d)K⁻¹`; hood = identity; sky = rotation-only `KRK⁻¹`; movables routed to LEARN) →
  rasterize the warped label map to SDFs → R1 σ=1.0 ramp → class-mean RGB → the deterministic bulk frame.
- **OUTPUT — the rule-118 FREE/COUNTED split (NO-FAKE boundary, FACT 1 + CLAUDE.md "compile the generator"):**
  - **COUNTED in `archive.zip`:** the 13 keyframe label maps (contour-coded, ~9 KB), the pose stream, the
    calib (3 f64), the per-class warp-type mask (tens of bytes). All video-DERIVED.
  - **FREE in `inflate.py`:** the warp algorithm, the SDF rasterizer, the R1 ramp, the openpilot lane
    centerline geometry, the Fourier basis. All GENERIC. **FORBIDDEN:** smuggling a per-frame learned table
    into inflate "code" disguised as generic (NO-FAKE #6 hide-data-in-code; rule 118).
- **BUILT (the pieces):** `src/tac/se3.py` (SE(3)/screw compose), `src/tac/camera.py` (EON intrinsics +
  homography), `src/tac/boundary_math/contour_codec.py` (partition ↔ label-map LZMA, bit-exact),
  `lane_sdf_component.py` / `hood_static_component.py` / `road_horizon_component.py` (structured SDF
  components + self-detecting class roles), and the warp+render+R path proven inside
  `tools/measure_screw_reach_through_R.py` (k=0 reproduces the R1 floor EXACTLY — the NO-FAKE faithfulness
  anchor).
- **NEEDS-WIRING:** these live in MEASUREMENT tools, not a production decoder. The v2 6-section codec is
  DESIGN-not-BUILT (session CRITICAL #2: the byteclose smoke's "store-canonical" is actually per-pair raw f0).
  Wiring = lift the warp+rasterize+ramp from the reach tool into a reusable `tac.v2_compose.bulk_generator`
  + the matching inflate-side numpy/torch decoder (MLX-free, per the contest-CPU Linux-x86_64 constraint that
  `witness_byte_close_and_eval.py` already enforces).
- **Automation:** `bulk_generator.generate_bulk(keyframes, pose, calib, warp_mask) -> bulk_frames` (compress
  side) + the byte-identical inflate-side mirror.

### Step 4 — Small residual INR (LEARN; the d_seg-floor closer)  ·  **NEEDS-WIRING** (substrate + structured-init BUILT)

- **INPUT:** the deterministic bulk frames (Step 3) + the GT SegNet argmax (`lstars`). The **residual target =
  the argmax cells the deterministic bulk gets WRONG** — i.e. the Lane-survival annulus + small movables,
  NOT the whole partition.
- **OPERATION:** train an INR (the level-set witness substrate, `lever_b_levelset_generator`) whose loss is
  the score-domain d_seg on `(deterministic_bulk ⊕ INR_residual)` — the INR only has to flip the residual
  cells. Because it carries the residual not the partition, it can be **much smaller** than the full
  mod-26/hidden-96 INR (the rate win). Warm-start: from-scratch with the openpilot lane-prior seed (in-basin,
  FEED-fs separatrix 1.9e-5) is preferred over resuming the PR95-curriculum ckpts (the anti-pattern).
- **OUTPUT:** the trained residual-INR EMA-shadow weights (int8+brotli) = the COUNTED LEARN-tier bytes
  (size OPEN — the single unmeasured quantity, settled only by the GPU run).
- **BUILT (substrate, partial):** the level-set trainer
  `experiments/train_levelset_witness_realized_through_R_mlx.py` already has `--structured-init` +
  `--lane-prior-phi1` where it "LEARNS only the residual (lane wall + Movable)" (trainer line ~2443) — BUT
  that is a TRAINING-TIME init prior: the trained INR weights still encode the whole partition (the bulk SDFs
  ship inside the INR weights). It accelerates convergence; it does NOT shrink the rate.
- **NEEDS-WIRING — the rate-bearing difference:** a trainer mode where (a) the deterministic bulk is GENERATED
  at decode (outside the counted INR weights) and SUBTRACTED from the target, (b) the INR is SIZED for the
  residual only (smaller mod/hidden), and (c) the inflate composes `bulk ⊕ INR_residual`. Today's structured-
  init keeps the bulk inside the INR; v2-done-right moves the bulk OUT (deterministic) and shrinks the INR.
- **Automation:** emit a flag-validated residual-INR launch command (the `witness_autoconfig` dogfood pattern:
  derive → validate every flag vs the real argparse → HOLD for operator GO; one GPU; containment/per-stage
  ckpt/EMA-shadow/single-seed). The composition entry point emits this command; it does NOT launch.

### Step 5 — Dual-use pose sidecar (STORE; d_pose direct + d_seg-warp driver)  ·  **BUILT**

- **INPUT:** the GT frames + the frozen PoseNet.
- **OPERATION:** extract 6 PoseNet scalars/pair (`extract_posenet_targets`) → store fp16 + zlib
  (`save_posenet_targets`), or compress further via the low-rank pose codec (`qp1_pose_codec` /
  `pose_from_embedding` MLP, #140, rank-2 ~2 KB).
- **OUTPUT (COUNTED, ~0.9–5 KB):** `posenet_targets.bin`. **Consumed TWICE:** (1) at inflate, the render is
  supervised to hit these targets → d_pose ≈ 3.4e-5 directly; (2) the SAME stored pose drives the Step-3
  per-class stratified warp for the d_seg bulk — at ~0 extra bytes (FACT 3 dual-use, the real weak sense).
- **BUILT:** `src/tac/scorer_targets.py` (PNTG format, extract/save/load, CLI). Low-rank codec `qp1_pose_codec.py`
  + `pose_from_embedding.py` present (#140).
- **Automation:** `python -m tac.scorer_targets --gt-video <mkv> --posenet <st> --output posenet_targets.bin`,
  then optional rank-2 recompress; the composition entry point calls this + folds the bytes into the budget.

### Step 6 — Byte-close + dual CPU/CUDA exact eval (the ONLY end)  ·  **partially BUILT (full-witness path; 6-section composition NEEDS-WIRING)**

- **INPUT:** the four COUNTED sections — (1) pose sidecar (Step 5), (2) 13 keyframes + warp mask + calib
  (Step 3 STORE), (3) residual-INR weights (Step 4), and the FREE generic inflate code.
- **OPERATION:** assemble `archive.zip` (the four counted sections, each entropy-coded) + `inflate.py`
  (MLX-free numpy bulk-generate + residual-INR forward + torch R) + `inflate.sh`; the rate term =
  `archive.zip` st_size (`upstream/evaluate.py:63`). Run inflate → realized d_seg/d_pose on the inflated
  frames (frozen CPU-torch, the trainer-authority mirror) → emit the STAGED dual exact-eval command (CPU
  Linux-x86_64 + CUDA T4, on the SAME bytes).
- **OUTPUT:** `S = 100·d_seg + √(10·d_pose) + 25·archive_bytes/37_545_489` (`tac.contest_score.compute_contest_score`).
  Advisory until `upstream/evaluate.py` runs the SAME packet on contest-compliant hardware (the only authority).
- **BUILT:** `tools/witness_byte_close_and_eval.py` does exactly this for the SINGLE full-witness blob
  (int8+brotli, full-output inflate, realized d_seg/d_pose, staged exact-eval command, false-authority block).
- **NEEDS-WIRING:** extend its archive grammar from 1 section (witness blob) to the **4-section v2 grammar**
  (pose sidecar + keyframe-store + warp-mask/calib + residual-INR) and the matching multi-section inflate.py
  that runs bulk-generate ⊕ residual ⊕ pose-supervise. The byte-accounting discipline + MLX-free constraint +
  staged-eval scaffolding are already there to reuse.
- **Automation:** the composition entry point's PHASE-B (§3c) calls the extended byte-close after the residual
  INR is trained.

---

## 2. (a) COHERENCE AUDIT — does each step feed the next without manual glue? (the seams)

The pipeline is **coherent in data-flow** (every step's output is a typed input to the next) but has **four
seams that need wiring** before it runs end-to-end without a human stitching artifacts:

| Seam | From → To | State | What's missing |
|---|---|---|---|
| **S1 (keystone)** | warp/reach measurements → Step 2 encoding of the KNOWN split | **OPEN** | `encode_known_split(...)` does not exist. The split is known deep-math; what's missing is the typed callable that EMITS it parameterized by the measured per-class recoverability + predicted bytes (not a discovery — an encoding). |
| **S2** | Step 2 assignment → Step 3 bulk generator | **OPEN** | the bulk generator lives in a measurement tool, not a reusable `tac.v2_compose.bulk_generator`; no production inflate-side mirror. |
| **S3** | Step 3 bulk → Step 4 residual target | **OPEN** | no trainer mode that subtracts the deterministic bulk and sizes the INR for the residual only (current structured-init keeps the bulk in the INR weights). |
| **S4** | Steps 3+4+5 → Step 6 archive grammar | **OPEN** | `witness_byte_close_and_eval.py` knows a 1-section grammar; needs the 4-section v2 grammar + multi-section inflate. |

**Coherent by construction (no seam):** Step 1 (built tool), Step 5 (built tool), the canonical score helper
(Step 6 math), the FREE/COUNTED rule-118 boundary (consistent across Steps 3–6), the class order (canonical
[Road,Lane,Undrivable,Movable,MyCar], self-detected everywhere — never hardcoded), and the MLX-free inflate
constraint (already enforced). The geometry substrate (se3/camera/contour_codec/lane_sdf/hood_static) is
built and self-consistent; the seams are *composition* glue, not new science.

**The honest seam summary:** ~70% of the pipeline is built as *components*; the missing 30% is the
**composition layer** (`tac.v2_compose` package: the decision function + bulk generator + residual-target mode
+ 4-section grammar). None of the seams requires GPU or new measurement to wire — they are CPU/numpy plumbing
of already-measured mechanisms, except S3 which needs the GPU residual run to produce its weights.

---

## 3. (b) OPTIMALITY NOTE — the rate math (the predicted win toward sub-0.15)

**The HYBRID (deterministic bulk + small residual INR + dual-use pose) is THE optimal vehicle** — not the
full-partition INR. The optimality argument is a **rate decomposition**: the hybrid spends bytes only on the
genuinely-irreducible LEARN residual, moving the large-area pose-recoverable bulk off the counted ledger
(deterministic-generated, FREE). The full-partition INR is the vehicle being SUPERSEDED; it serves only as the
floor/warm-start/confirmation (Step 1).

**The vehicle being superseded (full-partition level-set INR, mod-26/hidden-96):** ~90 KB → rate ≈ **0.060**
(review byte-close; the proven g3 bc20 DUAL exact row is 89,244 B → rate 0.0594 → [contest-CPU] 0.37797,
d_seg-dominated). The INR weights encode the ENTIRE argmax partition — most of which (the pose-recoverable
bulk) is deterministically generatable and therefore wasted counted bytes. THAT is the rate the hybrid recovers.

**v2 composition budget (the decomposition):**

| Section | Bytes | Rate contribution | Source / status |
|---|---:|---:|---|
| Pose sidecar (6 scalars/pair, rank-2 #140) | ~875 B – 5 KB | 0.0006 – 0.003 | **MEASURED** (scorer_targets; FEED-ll 875 B) |
| Partition keyframes (13, contour-coded) + warp mask + calib | ~9 KB | **0.0060** | **MEASURED** (FEED-ll reach k\*=47) |
| Residual INR (Lane-survival + movables ONLY) | **OPEN** | **OPEN** | the single unmeasured quantity — the GPU run |
| **FREE-generated structure** (warp, SDF, R1 ramp, lane geom, Fourier) | 0 | 0 | rule-118 generic algorithm |

**The win (rate axis, GREEN per FACT 1):** the bulk goes from ~90 KB-of-INR-weights to ~9 KB-of-keyframes +
0 KB-of-free-warp. The residual INR replaces the full INR; since it carries only the thin Lane annulus +
small movables (not the rank-8 bulk + static classes), it should be **far smaller than 90 KB**. If the
residual INR lands at, say, 10–30 KB, total rate ≈ 0.006 + 0.001 + (0.007–0.020) ≈ **0.014–0.027** vs the
baseline 0.060 — a rate cut that, combined with the d_seg the residual buys, is the credible sub-0.15 path.

**The binding constraint (FACT 2, the HONEST flag):** rate being cheap is necessary, NOT sufficient. The
deterministic bulk alone sits at d_seg ≈ 0.0185 (S ≳ 2). **The whole sub-0.15 case rests on the residual INR
closing d_seg from 0.0185 → ~6e-4 … 1.4e-3 at a SMALL byte cost.** Whether a small residual INR can do that
is the single OPEN quantity — and the lane-survival R-survival physics (GAP2) is the deepest unknown (the
Lane 0.58 pre-R is a lower bound; the through-R survival cost is unmeasured). The rate math says "the budget
closes IF the residual is small and effective"; only the GPU residual run + byte-closed exact eval proves it.
The `break_even_d_seg(0.19110, d_pose=3.4e-5, archive_bytes≈0.014·N/25)` arithmetic is the live target the
composition entry point should print at plan time.

---

## 4. (c) AUTOMATION PLAN — the single entry point

**Proposed: `tools/compose_witness_archive.py`** (thin CLI) over a new **`src/tac/v2_compose/`** package
(the `tac` stays clean discipline: implementation in `tac`, CLI delegates). It runs **steps 2–6 from a
finished baseline run dir**, split into two phases around the one unavoidable GPU step:

**PHASE-A (CPU/$0, no GPU, no launch) — plan + deterministic build + residual launch command:**
1. Load the baseline run dir + GT cache; run / read Step-1 attribution.
2. **Step 2:** `encode_known_split(warp_through_R, reach_kstar, attribution)` → the KNOWN-OPTIMAL per-class
   STORE/GENERATE/LEARN plan + predicted byte budget + the break-even d_seg target (printed); attribution
   only CONFIRMS the split + sizes the residual.
3. **Step 3:** select the 13 keyframes, build the deterministic bulk (compress-side), contour-code the
   keyframes, compute the residual target.
4. **Step 5:** build the pose sidecar (extract + rank-2 compress); fold bytes into the budget.
5. Emit the **flag-validated residual-INR launch command** (the `witness_autoconfig` dogfood: parse the real
   trainer argparse, assert every flag exists, print PASS/FAIL, HOLD for operator GO — one GPU,
   containment/per-stage-ckpt/EMA-shadow/single-seed, perf env `TAC_MLX_CUSTOM_GROUPED_BACKWARD=1`).

**[operator GO → the residual INR trains on the one GPU, resumable/per-stage; NOT this tool's job]**

**PHASE-B (CPU/$0) — byte-close + exact-eval staging:**
6. **Step 6:** assemble the 4-section `archive.zip` (pose + keyframes + warp-mask/calib + residual-INR) +
   MLX-free inflate.py; run inflate → realized d_seg/d_pose (advisory) → `compute_contest_score` →
   emit the STAGED dual CPU(Linux-x86_64)/CUDA(T4) `upstream/evaluate.py` command on the SAME bytes.

**Design constraints (inherited, non-negotiable):** deterministic + seeded; resumable (PHASE-A re-runnable,
PHASE-B resumes from the trained residual); every emitted flag validated against the real argparse (never
invent a flag); every number false-authority-tagged; the FREE/COUNTED rule-118 boundary enforced at archive
build (a NO-FAKE check that the inflate.py carries no video-derived table); reuses the built tools rather than
reimplementing (attribution, warp/reach tools, scorer_targets, contour_codec, se3/camera, byte_close_and_eval).

`witness_autoconfig.py` is the right pattern sibling but is scoped to the *full-partition INR* launch; the v2
composition is a DIFFERENT actuator (it consumes a finished run + builds the 4-section archive). Recommend a
NEW `tools/compose_witness_archive.py` + `src/tac/v2_compose/` rather than overloading `witness_autoconfig`.

---

## 5. (d) HONEST STATUS

- This is a **SPEC (a MEANS)**. The pointer is **0.19110 and UNMOVED**. Nothing here is a score, a frontier,
  a promotion, or a kill. It moves only on a byte-closed `archive.zip` exact row from `upstream/evaluate.py`
  (CPU + CUDA, never MPS).
- **~70% built (components), ~30% needs-wiring (the `tac.v2_compose` composition layer + 4-section grammar +
  residual-target trainer mode).** None of the wiring is new science; it is CPU/numpy plumbing of measured
  mechanisms — EXCEPT the residual INR weights (Step 4), which need the one GPU run.
- **The single open quantity** that decides v2's fate is the **LEARN-tier byte cost AND its d_seg efficacy**:
  can a SMALL residual INR close d_seg from the 0.0185 deterministic floor to ~6e-4…1.4e-3? The rate axis is
  GREEN (FACT 1); the binding wall is this residual + the Lane R-survival physics (GAP2). The composition is
  designed so that this one GPU run is the decisive measurement — everything else is $0/CPU and built or
  wireable now.
- **NO-FAKE boundary held throughout:** generic algorithm FREE in inflate.py; video-derived payload COUNTED
  in archive.zip; the warp is dual-use only in the weak (one-store / two-free-readouts) sense — the lossless
  dual-use grok was REFUTED (FACT 3) and is NOT relied on.

**Cross-refs:** `[[gr-unified-action-full-witness-architecture-20260629]]` ·
`[[v2-novel-contribution-originality-accounting-20260629]]` ·
`[[session-20260630-review-warpfix-lossless-exhausted-CURRENT]]` · DAG FEED-iz / ja / lj / lk / ll / lf ·
CLAUDE.md "Evaluator-Equivalent Witness Compiler" + "Pose is SOLVED" + rule-118.
