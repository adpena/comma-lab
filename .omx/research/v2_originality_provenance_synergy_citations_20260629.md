# v2 Task-Space Witness Codec — Originality, Provenance, Synergy & Citations

**Operator directive 2026-06-29:** *"It should include links and citations and notes on synergy and how it
fell through or was solved or discovered or adapted."* This is the paper-grade companion to the memory record
[[v2-novel-contribution-originality-accounting-20260629]] (DAG FEED-jt/js). Source verdict: related-codec sweep
`related_codec_sweep_v2_20260629T201707Z.md` (FEED-js, commit `981e07f4e`).

> **⚠️ STATUS — UNVALIDATED DESIGN-ORIGINALITY CLAIM, MEANS NOT ENDS.** Pointer UNMOVED contest-CPU **0.19110**.
> Every element below is a *design* claim; it is a *contribution* only when a byte-closed exact row beats 0.19110.
> An elegant unoccupied intersection that does not move the score is a MISS per THE GOAL, not a contribution.
> NO-FAKE #7: arXiv IDs marked `[unverified]` came from the sweep agent and are NOT independently confirmed —
> verify before any external use.

## The verdict — the unoccupied intersection

Two mature fields flank our problem; neither occupies it:
- **(A) driving-scene reconstruction with canonical-scene + warp** — GS-Lucas-Kanade (ICLR'25) `[unverified arXiv]`,
  WorldSplat `[unverified arXiv]` — but optimizes *rendering PSNR*.
- **(B) codecs-for-machines / distributed source coding** — right (task) objective, but *black-box decoders, no
  geometry*.
- **Closest single neighbor:** *Implicit Neural Video Compression*, Zhang et al. (arXiv **2112.11312**,
  https://arxiv.org/abs/2112.11312) = our literal skeleton (an implicit net modulates coordinates for motion
  compensation + a small residual net for P-frames) **but** pixel-fidelity / learned warp / no task-space / no SDF.

## Motivation — why this design exists (the "why," for the record)

**1. HNeRV was both SLOW and a domain/info-space MISFIT (the push off the incumbent).**
- *Training cost:* the PR95/HNeRV winning recipe is an 8-stage, **29,650-epoch** curriculum (canonical eq
  `pr95_family_l14`) — a multi-DAY burn on one GPU, and loop-end-only saving loses everything on a crash (our
  resumability + per-stage-checkpoint non-negotiable was born from exactly this pain). Our faithful reproduction
  stuck at d_seg≈0.50 (the "PR95 elephant," task #75) with an inert score-aware loop (#76): long, fragile, and
  never cleanly at 0.193 in our hands.
- *Domain/info-space misfit* (cross-ref [[why-hnerv-blackbox-misfit-vs-levelset-taskspace-fit-20260627]]): a
  smooth black-box INR fights the piecewise-constant argmax target (spectral bias → Gibbs); full-RGB
  reconstruction spends capacity on channels SegNet ignores (channel arbitrariness); ~94% of bytes are decoder
  weights, not task-saliency (allocation arbitrariness); and the objective is reconstruction-first when the
  contest is **indirect-RD task-space**. bc20 was under-capacity; full-RGB is wasteful under `evaluate.py`
  (#171). HNeRV is a *parity bank*, not the vehicle.
- **→ the pivot:** stop reconstructing RGB; code the task-sufficient statistic directly (the witness). That seed
  is the root of every "ours" element below.

**2. The meta-innovation: the whole research program runs on a SINGLE MacBook Pro, at $0.**
- Substrate: M5 Max, 128 GB unified memory; MLX-first; the **MPS-as-training-GRADIENT-device** unlock (the frozen
  scorer runs ~**104× faster** on the Apple GPU at fp32 after one BatchNorm-backward stride fix,
  `tac.torch_mps_compat.patch_scorer_for_mps`), with **numpy-fp32 as the bit-identical verdict authority** and
  CPU/CUDA the only score axes (MPS never an authority).
- Consequence: this entire session's discovery — the grok, the screw-warp, the SDF carrier, the gauge layer, the
  openpilot head-start, the jitter wall, the literature sweep, this very ledger — was produced by a FLEET of $0
  CPU/MLX probes on one laptop. A contest usually decided by GPU budget was **reframed as a deep-math +
  $0-measurement problem**; the ONE heavy step (the lane-survival GPU run) is deliberately minimized and gated.
  This is itself a methodological contribution: indirect-RD task-space codec design is tractable on commodity
  hardware *because the expensive object — the witness's irreducible statistic — is tiny* (F4: ~KB, not MB).

**3. The underlying math is genuinely beautiful — and the beauty is a DESIGN SIGNAL, not decoration.**
- *One variational object:* the witness is the viscosity solution of a level-set PDE; `S_τ` is the indirect-RD
  action; the SDF is φ, the screw-warp is the transport V, the residual is births/deaths (Osher-Sethian).
  Geometry (SDF) + motion (Chasles screw) + topology (persistence) + the task (frozen-scorer argmax) are ONE
  system.
- *Deep classical math throughout:* Chasles' screw theorem (every rigid motion is one screw); Longuet-Higgins–
  Prazdny depth-stratified flow; Fisher-information geometry (the annulus = a homography orbit; rank-K−1 ceiling;
  curvature ↔ −margin Pearson 0.978); canonicalization / gauge-fixing for MDL; flow-matching on Lie groups.
- *The cross-disciplinary convergence is the tell:* fluids (Helmholtz–Hodge), graphics anti-aliasing (Valve SDF),
  astronomy difference-imaging (Alard–Lupton), Lie theory (flow-matching), even Buddhist phenomenology
  (dependent-origination = code the relational statistic, not the inherent pixel) ALL land on the *same* vehicle.
  When that many independent lenses converge, the design is probably "right" in the deep sense — the elegance is
  *evidence*. And per the operator: the deep-math IS the joy; this has been genuinely fun, which is part of why it
  went far.

> Honest caveat (unchanged): motivation + beauty are MEANS. They are the *why*, not the *result* — they do not
> move the pointer; only a byte-closed exact row below 0.19110 does. Recorded because the why is real signal.

## The level-set system EUREKA — the unifying mathematical object

THE eureka (the "Crystal Cathedral" moment): the pieces — SDF carrier, screw warp, residual, gauge,
indirect-RD objective — are **not a pile of tricks; they are terms of ONE variational object.** The witness is the
**viscosity solution of a level-set PDE**, and *minimizing the indirect-RD action S_τ over level-set fields φ IS
the codec.* This is *why* the pieces compose with zero friction (the synergy map) — they were always one object.
The five facets the operator named:

- **Manifolds.** Three nested manifolds: (a) the scorer-**equivalence quotient** (all witnesses with the same
  argmax = one fiber; coding = pick the cheapest representative = the gauge/quotient codec #155); (b) the **task
  manifold** — the small-margin boundary annulus is a **homography orbit** (~8-dim lane orbit; the ego-motion
  group acting on the boundary); (c) the **Fisher manifold** — the fixed curved background metric G=E[JᵀFJ] the
  witness is stationary in (matter on a fixed background).
- **Dimensionality.** The witness is small because the *intrinsic* dimension is small and **measured**: the
  **rank-K−1 ceiling** (K=5 classes resolve ≤4 directions; pose eff-dim 4.08 sits on it, F2); the ~21-dim off-pose
  residual; **86% of the Fisher nullspace is discardable** (F2); the lane orbit ~8-dim. Store the intrinsic
  dimensions, discard the nullspace — **dimensionality IS the rate.**
- **Derivatives.** The system is differential: the **eikonal |∇φ|=1** (the SDF's unit-gradient — the live
  derivative constraint that makes it interpolation-exact and R-surviving); the **Fisher metric** G (built from J,
  the scorer's derivative); the **transport** ∂φ/∂t + V·∇φ = 0 (V = the ego-flow, a derivative of the stored
  pose); natural-gradient / geodesic flow toward θ*.
- **Integrals.** The system is variational: **S_τ is an integral functional** being minimized; the **length**
  (∮ along the boundary) + eikonal terms are integral regularizers (the live integral constraints, θ* memory); the
  **rate term is the integral of the description length** (MDL); the distortion energy is integrated over the
  boundary annulus (where it concentrates — co-location 0.978).
- **Topology.** The deepest facet: the partition's **topology** (births/deaths of components under a sublevel-set
  filtration) **IS the irreducible information** (Morse-Smale codec #180; persistence ≡ astronomy deblending ≡ the
  Kelvin invariant — the FEED-jg identity). The level-set method's signature strength is handling **topology
  changes (merge/split) natively** — which is *exactly why* it fits a driving partition whose connectivity changes
  as the car moves (new lanes appear, objects occlude). Topology is conserved except at bifurcations; **the
  bifurcations (the events) are the COUNTED residual.**

*Provenance:* emerged across the session — the θ* variational-levelset frame (2026-06-27,
[[theta-star-witness-lever-stack-and-variational-levelset-frame-20260627]]); the GR-unified-action
(2026-06-29, [[gr-unified-action-full-witness-architecture-20260629]]); the Osher-Sethian unification (FEED-jg,
from the graphics/astronomy sweep). *Borrowed:* the level-set method (Osher–Sethian 1988); viscosity solutions
(Crandall–Lions); persistent homology / Morse theory; Fisher-information geometry (Amari); variational
segmentation (Mumford–Shah, Chan–Vese). **OURS:** assembling them so the level-set field *is the compressed
witness* and the action *is the contest score* (indirect-RD task-space) — the unifying frame is the insight that
makes the "novel composition" coherent. Honest: the eureka is the unifying FRAME (MEANS); it is validated when a
byte-closed exact row beats 0.19110.

## OUR five elements — what, citation, provenance (discovered/solved/adapted/fell-through), synergy role

### 1. Distortion = the exact frozen-oracle argmax **CELL** (indirect-RD), not a proxy/PSNR
- **Nearest prior art / adapted from:** indirect rate-distortion / CEO problem (Berger; Dubois "Lossy Compression
  for Lossless Prediction," arXiv **2106.10800**, https://arxiv.org/abs/2106.10800); coding-for-machines / VCM.
- **Provenance:** DISCOVERED 2026-06-19 (contest = indirect-RD, task-space coding) → formalized as the action
  S_τ (FEED gz→ii; canonical equations E0–E12). **Adapted** the indirect-RD *theory* to the *exact hard-argmax
  cell* of a specific frozen scorer (not a soft proxy) — this is the sharpening that is ours.
- **Synergy:** defines the objective every other element optimizes; makes the Fisher metric (curvature↔−margin
  0.978, FEED b0bee924e) the geometry the warp/SDF/gauge all live in.

### 2. Physical per-class **SE(3) screw warp** (Chasles twist, depth-stratified)
- **Adapted from:** screw theory (Ball 1900; Chasles 1830); Longuet-Higgins–Prazdny optical-flow equations;
  warp-coordinate INR motion comp (2112.11312, CWRNN-INVR arXiv **2604.06564** `[unverified]`).
- **Provenance:** DISCOVERED via the SFM/grok lens (FEED-ja, 2f83e0b9e — stratified per-class warp, $0-confirmed)
  → SOLVED as the ~0-byte gauge (FEED-jj, a513372a: screw reproduces Road exactly, fixes hood/sky; the per-class
  homography's edge was 85% non-physical overfit). **FELL-THROUGH/corrected:** through R it does NOT get the bulk
  under budget (FEED-jq, a23062c4: bulk 0.0048 = 4× budget; the per-frame SegNet *jitter* floor a neighbor-warp
  can't predict) → opened the clean-canonical test (a95b0ad6, running).
- **Synergy:** reuses the stored pose (dual-use with d_pose); the lane's cheap encoding lives in *its* ground
  frame; it is the transport field V of the level-set PDE.

### 3. SDF carrier validated by indirect **scorer-survival** (not rendering fidelity)
- **Adapted from:** SDF text rendering (Green/Valve, SIGGRAPH 2007 — no arXiv); msdfgen (Chlumský 2018,
  https://github.com/Chlumsky/msdfgen); quantized neural displacement fields (Pentapati, arXiv **2504.01027**
  `[unverified]`).
- **Provenance:** DISCOVERED (F1, FEED-iw: an SDF survives R far better than a bitmap). **Corrected** the
  *mechanism* mid-stream — the fluid/scale-space lens (operator daydream) FALSIFIED my "heat-stability" claim and
  replaced it with the truer one: R is bicubic *interpolation*, exact on the 1-Lipschitz ramp (the Valve result).
  **SOLVED/validated:** single-SDF lane d_seg 5.9e-4 @render-192, 1e-5 @320 (FEED-jk, a1d5682964 — both clear the
  1.23e-3 threshold). **FELL-THROUGH:** MSDF (the corner-recovery variant) was FALSIFIED — our flips are thin
  (sub-Nyquist), not corners; MSDF *hurts* (non-1-Lipschitz channels survive R worse). Live lever = render
  resolution, not corners.
- **Synergy:** the carrier the warp moves; survives R *because* of #1's interpolation; cheap in #2's ground frame.

### 4. The 3-stage prior chain **warp → SDF → openpilot-Wyner-Ziv residual** with the stored-jitter / generated-structure split
- **Adapted from:** Wyner-Ziv coding with decoder side-info (Whang arXiv **2106.02797**,
  https://arxiv.org/abs/2106.02797; Özyılkan–Ballé arXiv **2310.16961**, https://arxiv.org/abs/2310.16961);
  difference imaging (Alard–Lupton 1998; ZOGY, Zackay-Ofek-Gal-Yam arXiv **1601.02655**); reward/target smoothing
  (DreamSmooth, arXiv **2311.01450**); margin-keyed allocation (PICM-Net arXiv **2512.20070** `[unverified]`).
- **Provenance:** the openpilot lane prior was DISCOVERED to be a FREE positional prior, NOT a residual-collapser
  (FEED-jh, a99f41f0: the polynomial floor 0.00214 > threshold → the lane *needs a trained generator*; but the
  centerline recovers ~64% as Wyner-Ziv side-info). BUILT as the conditional-residual pipeline (FEED-jm,
  a5b83c730: bit-exact X−E[X|Y]). **FELL-THROUGH/corrected:** the centerline byte cost was 65 KB image-space, NOT
  the 0.5–5 KB I'd claimed (FEED-jm corrects FEED-jd) → fix = ground-frame coding (= element #2's home), the next
  $0 rate gate. The stored-jitter/generated-structure split was DISCOVERED from the jitter wall (FEED-jq) +
  ADAPTED from DreamSmooth (smooth the learning target, store the irreducible jitter as margin-keyed dither).
- **Synergy:** binds #2 (warp base) + #3 (SDF carrier) + the openpilot prior into one Wyner-Ziv chain; the
  jitter-split is what makes the trained generator's job small + stable.

### 5. **Gauge-as-codec-canonicalization** — canonicalizing for minimum description length over a task-equivalence class
- **Adapted from / differs from:** learned canonicalization (Kaba et al., arXiv **2211.06489**,
  https://arxiv.org/abs/2211.06489); frame averaging (Puny et al., arXiv **2110.03336**); a canonicalization
  perspective (arXiv **2405.18378**); the impossibility of *continuous* canonicalization (Dym et al., arXiv
  **2402.16077**, ICML 2024). **The difference that is ours:** the whole literature canonicalizes for downstream
  *accuracy*; we canonicalize for **rate** (pick the minimum-description-length representative of the
  scorer-equivalence class). That = the operationalization of the level-set/fiber **quotient codec** (#155,
  Dubois 2106.10800).
- **Provenance:** DISCOVERED via the operator's gauge insight (FEED-ji: "the gauge fits into the DSL + a new meta
  layer") → BUILT (FEED-jl, a89b620b: gauge.py + fix_gauge + 30 tests). **FELL-THROUGH/correction pending:**
  `fix_gauge` uses *hard* argmin; the Dym impossibility theorem says that can be *discontinuous* → must become
  **soft-weighted** (FEED-js follow-up). This also rhymes with the diffusion paper's "factorized but discontinuous
  manifold" (FEED-jo, arXiv 2408.13256).
- **Synergy:** the meta-glue — picks the cheapest representative for *every* component (#2 warp, #3 carrier, #4
  residual, pose); the soft-weighting keeps the whole composition continuous (so #4's generator has a continuous
  target).

## Borrowed substrate (explicitly NOT ours) — citation + how adapted

| Borrowed | Cite | How adapted into v2 |
|---|---|---|
| Flow-matching generator | LieFlow 2512.20043; GNVC-VD **2512.05016**; OT-NFM **2604.06413** `[unverified]`; CoD-Lite **2604.12525** `[unverified]` | residual *correction* from the warp+SDF prior (not from-noise); few-step deterministic ODE for budget+bit-identical |
| Conditional VQ-VAE / neural Wyner-Ziv | Whang 2106.02797; Özyılkan-Ballé 2310.16961 | the openpilot-conditioned lane residual codec |
| Warp-coords + residual INR skeleton | 2112.11312; NIRVANA **2212.14593**; TINC **2211.06689** | the architectural skeleton (we add task-space + SDF + physical warp) |
| Canonicalization framework | Kaba 2211.06489; Puny 2110.03336; Dym 2402.16077 | re-aimed at rate (MDL) instead of accuracy |
| SDF survives interpolation | Green/Valve 2007; msdfgen | the lane carrier; MSDF tried + falsified |
| Difference-imaging residual | Alard-Lupton 1998; ZOGY 1601.02655 | code residual vs the pose-warped canonical (we skip PSF-matching — R is interpolation-exact) |
| Target smoothing | DreamSmooth 2311.01450 | smooth the generator's learning target; store the jitter separately |
| Metadata conditioning | FINO **2606.05107** (confirmed via screenshot + WebSearch; one sweep-agent locate-fail) | condition the generator on the full free metadata (pose/K/calibration/class) |
| Margin-keyed bit allocation | PICM-Net 2512.20070 `[unverified]` | spend dither bytes only in the thin-margin annulus |
| Controlled-stochasticity knob | GVCC **2603.26571** `[unverified]` | ODE→SDE if a deterministic generator under-fits the jitter |
| Diffusion factorization/composition | 2408.13256 | the few-example factorized-rep motivation + the discontinuity caveat |
| Contest substrate | PR95/HNeRV; comma2k19; openpilot; comma10k | the frozen scorers, the source clip, the lane/pose priors |

## Synergy map — why the composition coheres (not just a pile of parts)

- **One stored quantity, three uses:** the ego-pose/twist is stored once → pays d_pose (element #1 objective),
  drives the per-class warp (#2), AND anchors the lane base (#4). Dual/triple-use is the rate win.
- **The ground frame is the universal home:** the warp (#2), the cheap SDF lane coding (#3), and the residual
  base (#4) all live in the bird's-eye ground frame — the same representation makes the warp physical AND the
  bytes small. (This is *why* FEED-jm's 65KB→ground-frame correction strengthened rather than broke the design.)
- **The level-set PDE unifies #1–#4:** φ = the SDF carrier (#3), V = the screw-warp transport (#2), the residual
  = births/deaths (#4), the action = the indirect-RD objective (#1) — one Osher-Sethian system (FEED-jg).
- **The gauge (#5) is the meta-glue:** it selects the cheapest representative for each of #2–#4 and keeps the
  whole thing continuous (soft-weighted, per Dym). It is the codec-level form of the quotient (#155).
- **The four recent papers compose on the one open piece (the generator):** generator class = flow-matching
  (LieFlow, deterministic+few-step) + conditioning = full metadata (FINO) + training signal = smooth-target +
  stored-jitter (DreamSmooth) + flicker-fix = sequence-level refinement (GNVC-VD) + annulus dither (PICM-Net).

## The generate-vs-store partition (the FREE/COUNTED boundary, decided EMPIRICALLY per component)

A core part of the method — and itself novel: most codecs fix the generate-vs-store split *by architecture*. We
**decide it per component by a $0 measured analysis of its score-cost** (the gauge layer's `GenerationGauge` axis;
`fix_gauge` = hard-gates → min-S → the cheaper side wins). This is the operationalization of CLAUDE.md's "compile
the maximal deterministic GENERIC structure into inflate.py (FREE), store only the irreducible video-derived
statistic (COUNTED)" — with the *decision rule* measured, not asserted (canonicalize-for-MDL).

| Component | Decision | The measured analysis that decided it (DAG FEED) | rule-118 |
|---|---|---|---|
| Ego-pose / SE(3) twist | **STORE** (tiny, dual-use) | range-codes to 474–875 B (F4); rank-1 / forward-speed dominant (a99f41f0/FEED-jh) | COUNTED ~875 B |
| Bulk warp (Road/sky/hood) | **GENERATE** (deterministic LHP from the stored pose) | screw reproduces Road exactly at ~0 marginal bytes; per-class-homography's edge 85% non-physical overfit (a513372a/FEED-jj) | FREE algorithm |
| — bulk per-frame jitter | **OPEN** (generate-via-clean-canonical vs store-per-frame) | per-frame-warp through R = 4× budget, the SegNet jitter floor (a23062c4/FEED-jq); **the clean-canonical budget gate a95b0ad6 is resolving this exact cell** | TBD by a95b0ad6 |
| Canonical scene + lane geometry | **STORE** descriptor (ground-frame) + **GENERATE** the rasterization | openpilot centerline = free positional prior, recovers 64% (a99f41f0/FEED-jh); ground-frame coding is the open rate gate, 0.5–5 KB target vs 65 KB image-space (FEED-jm) | descriptor COUNTED-tiny / rasterizer FREE |
| Lane SDF carrier | **GENERATE** (FREE SDF rasterizer; survives R) | single-SDF lane d_seg 5.9e-4 @192 / 1e-5 @320 clears R; MSDF falsified (a1d5682964/FEED-jk) | FREE algorithm |
| Lane ragged residual | **GENERATE** (trained flow-matching from the prior) + **STORE** the irreducible jitter as margin-keyed dither | polynomial can't collapse it → needs a trained generator (a99f41f0); flow-matching residual-correction + margin-keyed annulus dither (FEED-js: GNVC-VD/OT-NFM/PICM-Net) | trained weights COUNTED + dither COUNTED-tiny (annulus only) |
| Movables (class-3) | **STORE** (templates + low-rank trajectories) | warp-predict floor 0.00082 = 67% of the budget; store → ~0 at ~0.9–2.7 KB (F3/FEED-je) | COUNTED ~1–3 KB |

**The meta-point (why this is part of the contribution):** the FREE/COUNTED line is the rate term's whole game,
and we set it by *measuring each component's score-cost on both sides and keeping the cheaper* — pose/movables
proved cheaper to STORE, bulk-warp/SDF-carrier cheaper to GENERATE, the lane residual a generate+store hybrid,
and the bulk-jitter cell is the one still being measured (a95b0ad6). Deciding generate-vs-store by measured
minimum-description-length over a task-equivalence class is the same novelty as element #5 (gauge-canonicalization
for MDL), applied at the codec-structure level.

## Overfit the witness, generalize the implementation (operator design directive 2026-06-29)

Operator: *"Overfitting to the contest video is fine and preferred if optimal for min score but also want
generalizable implementations."* A TWO-LEVEL split the v2 already honors — and it sharpens what the contribution is:

- **WITNESS = the COUNTED payload (archive.zip) → OVERFIT freely, it's PREFERRED.** The contest scores ONE video;
  the optimal stored bytes are maximally clip-specific: the pose/twist stream, lane spline coeffs, bulk-jitter
  dither, the trained residual-generator *weights*, movables templates+trajectories, the canonical scene
  descriptor. No generalization penalty on the payload.
- **IMPLEMENTATION = the FREE generic algorithm (inflate.py) + the codec/tools → GENERALIZES** (works on any
  contest-shaped dashcam clip): the screw-warp (LHP homography from pose), the SDF carrier + rasterizer, the
  flow-matching generator *architecture + recipe*, the gauge/canonicalization-for-MDL decision rule, the
  level-set PDE, the curvelet basis *oriented-to-the-clip's-MEASURED-boundary* (orientation computed at
  compress-time, NOT hardcoded to this clip), the openpilot/comma priors + EON intrinsics, the DSL/campaign/tools.
  inflate.py is a generic decoder, not a clip-specific lookup.
- **Compliance boundary (rule-118 + NO-FAKE #6/hide-data-in-code):** the generic ALGORITHM is FREE in inflate.py;
  the overfit PAYLOAD is COUNTED in archive.zip. FORBIDDEN: smuggling the clip-specific payload into inflate.py
  "code" disguised as a generic algorithm. = CLAUDE.md "Contest vs production target modes"
  (`contest_one_video_replay` payload + `contest_generalized`/`production_generalized` algorithm).
- **Why it strengthens the contribution:** the originality is a GENERALIZABLE task-space codec METHOD that
  optimally overfits its payload per-instance (like any codec: general algorithm, specific compressed file) — more
  valuable and more honestly novel than a one-clip stunt; it's what makes the five ours-elements a *reusable codec*.

REVIEW CHECK (folded into the recursive review): every "generalizable" piece must be a REAL generic algorithm
(computes from the clip at compress-time, no hardcoded this-clip constants in inflate.py); every "overfit" piece
must live in the COUNTED payload. Confirms the generate-vs-store partition above from the algorithm-vs-payload side.

## Conditioning basis & training curriculum (curvelets + the scale curriculum)

Two design elements that are ADAPTATIONS (borrowed primitive + our task-specific, byte-free application), and that
materially shrink the witness:

**Curvelets / oriented directional basis — the ~0-byte d_seg lever (DM2).**
- *What:* orient the carrier/conditioning's basis to the MEASURED all-class boundary tangent field. Curvelets
  (Candès–Donoho, 2004) are the optimal sparse representation for *curved edge singularities* — O(N⁻²)
  approximation for a C² curve vs wavelets' O(N⁻¹) and isotropic Fourier's O(N⁻¹) — exactly matching the lane's
  curved boundary. The boundary is a 1-D curve in a 2-D field; an anisotropic, oriented basis is information-
  theoretically the right chart.
- *Provenance — DISCOVERED + MEASURED as THE decisive lever:* orienting the Fourier/curvelet features to the
  all-class boundary tangent field measured **−48% d_seg at ~0 byte** (vs lane-only −8%); basis-match is PRIOR to
  capacity (CLAUDE.md frontier §; `[macOS-MLX research-signal]`). It is DM2 in the v2 conditioning axis
  ([[gr-unified-action-full-witness-architecture-20260629]]).
- *Borrowed:* curvelets (Candès–Donoho 2004); shearlets (Kutyniok–Labate); WIRE wavelet-INR (arXiv **2301.05187**
  `[verify]`); BACON band-limited coordinate nets (arXiv **2112.04645** `[verify]`). **OURS:** orienting it to the
  *measured task-boundary tangent field* as a deterministic ~0-byte TRAIN-TIME prior that compiles into inflate.py
  for FREE (rule-118: a generic oriented basis generated from a seed = FREE; element of the generate-vs-store
  table's GENERATE side).
- *Synergy:* the FREE oriented basis the SDF carrier (#3) and the residual generator (#4) ride; "basis-match
  before capacity" is a core reason the witness can be small.

**Scale curriculum — coarse→fine, the spectral-bias / Nyquist climb.**
- *What:* train the witness coarse-to-fine across scales / render resolutions — climb the spectral-bias wall
  progressively, matched to the R Nyquist cliff (F1: a 2px lane needs render ≥192, ideally 320, to survive R).
- *Provenance:* the level-set curriculum is already a homotopy of relaxations (CE→tau→l7→Muon, θ* memory); the
  SCALE axis adds coarse→fine on top. Grounded in Daubechies multiresolution ("hierarchical coarse-gates-fine,"
  the inner-council CO-LEAD discipline) + the scale-space lens (F1) + render-resolution as the lane's live lever
  (FEED-jk). DreamSmooth-aligned: learn the smooth/coarse structure first, sharpen later.
- *Borrowed:* multiresolution analysis (Daubechies); progressive growing; coarse-to-fine INR training (BACON,
  band-limited nets). **OURS:** tying the scale curriculum to the *R-survival Nyquist cliff* + the level-set
  homotopy for a task-space (indirect-RD) witness.
- *Synergy:* the curriculum's stages ↔ the curvelet basis's scales (one multi-scale object); lets the
  flow-matching generator (#4) learn smooth structure before the ragged detail; directly attacks the spectral
  bias that sank the smooth black-box INR (the HNeRV misfit in §Motivation).

> Honest: both are MEANS — the curvelet −48% is a measured *advisory* d_seg lever (not an exact row); the scale
> curriculum is a training-design choice pending the GPU run. arXiv IDs `[verify]` recalled from memory, confirm
> before external use (NO-FAKE).

## The corrections / fell-through journey (the honest lineage — no signal loss)

1. **DM1 (per-pair FiLM rank) as the d_seg lever → DEMOTED** (FEED-ip): PR collapsed *while* d_seg improved; FiLM
   re-weights fixed channel-patterns, can't localize. → moved to spatial basis.
2. **Per-position spatial latent grid (DM3) → $0-FALSIFIED** (FEED-is): cross-pair variation is globally low-rank
   (ego-motion coherent); grid ~8× worse. → low-rank-global, then the screw-warp.
3. **F4 "rate solved ~3.2 KB" → CORRECTED** (FEED-jm): image-space lane is 65 KB; the 0.5–5 KB needs ground-frame
   coding (= element #2). → ground-frame rate gate queued.
4. **"bulk near-free via warp" → CORRECTED** (FEED-jq): per-frame warp through R is 4× budget; the per-frame
   SegNet jitter floor is the real wall. → the clean-canonical budget gate (a95b0ad6, running).
5. **SDF "heat-stability" mechanism → CORRECTED to interpolation-exactness** (F1 + the fluid-lens daydream test):
   the truer mechanism (and it's stronger — it's the Valve result).
6. **MSDF carrier → FALSIFIED** (FEED-jk): a corner tool with no corner target (our flips are thin/sub-Nyquist).
7. **openpilot polynomial lane → PARTIAL** (FEED-jh): can't collapse the residual (floor 0.00214 > threshold), but
   became the FREE 64% Wyner-Ziv head-start.
8. **diffusion generator caveats (determinism/budget) → SOLVED by flow-matching** (FEED-jp/js: LieFlow + OT-NFM).
9. **gauge hard-argmin → CORRECTION PENDING to soft-weighted** (FEED-js, Dym 2402.16077 impossibility theorem).

## Update triggers + cross-refs

Re-evaluate after: (a) the clean-canonical budget gate (a95b0ad6) — may resize element #4; (b) the lane-survival
GPU run — validates/falsifies each element against a measured exact row. Cross-refs:
[[v2-novel-contribution-originality-accounting-20260629]] · [[gr-unified-action-full-witness-architecture-20260629]]
· [[witness-dsl-and-dag-dsl-equations-triality-20260629]] · DAG FEEDs ja–jt · CLAUDE.md NO-FAKE #7
borrowed-substrate-accounting + the Innovation Gate. All `[research-signal]`; pointer 0.19110; means≠ends.
## UPDATE 2026-06-30 (b) — the screw & twist (19th-c kinematics) × coding-for-machines (21st-c) + the math-fusion thesis (operator: "add screw and twist + the combo driving and coding for machines")
**The screw & twist — explicit (the temporal factor + the deepest ours×borrowed line).** The ego-vehicle's motion between the two scored frames is, by **Chasles' theorem (Michel Chasles, 1830)**, exactly a SCREW: every rigid-body displacement = a rotation about a unique axis + a translation along it. We encode it as a **twist** ξ∈se(3) (Lie algebra of SE(3)); the worldline = exp(t·ξ) (a one-parameter subgroup / geodesic). The screw AXIS is a line in ℙ³ → **Plücker line coordinates (Julius Plücker, 1865)**. Per-class warps fall out of the ONE twist: Road = ground-plane projective **homography** (19th-c projective geometry, Möbius 1827 / Poncelet 1822), sky = rotation-only (the twist's rotational part), hood = identity (the twist acts trivially on the ego-static body). Whole temporal codec = ONE twist (~6 DOF) × per-class depth → ~24–48 floats/clip. A car driving down a road literally executes a Ball screw — **Sir Robert Stawell Ball, "A Treatise on the Theory of Screws" (1876)** is the native language of the ego-motion factor; we repurpose it as a CODEC.
**The combo — "driving-scene-recon-with-warp" × "coding-for-machines" (the unoccupied intersection, named by its two parents).** Two fields never crossed: (A) **driving-scene reconstruction with ego-motion warp** (Nerfies / D-NeRF / NSFF / 4DGS — reconstruct RGB, warp by motion, FOR humans/sensors); (B) **coding for machines / VCM** (compress for a downstream task net, not the eye). v2 = A∩B: compress the *task-space* (the frozen scorer's argmax readout) of a *driving* scene using the *physical ego-screw* as the temporal codec. A-people reconstruct pixels; B-people compress generic features with no physical ego-warp. Driving-physics-warp ON a frozen-scorer task manifold = empty cell (closest neighbor still INVC arXiv 2112.11312, generic-machine not driving-screw). This is the honest "novel COMPOSITION, not new primitive" verdict (NO-FAKE #7), now named by its two parents.
**The math-fusion thesis (operator's observation — accurate history).** The kinematic/geometric scaffold is **19th-century**: Chasles screws (1830), projective homography (Möbius 1827 / Poncelet 1822 / Plücker 1865), Lie groups & algebras (Sophus Lie, 1870s), Riemann metric & geodesics (Riemann 1854 → our frozen-scorer Fisher metric). It is FUSED with **20th–21st-century** machinery: level-set PDEs (Osher–Sethian 1988), Morse–Smale topology (Morse 1934 / Smale 1960s), persistent homology (Edelsbrunner et al. 2002), neural fields / INR (SIREN, NeRF 2020), the rate-distortion topological transition (Agmon–Tishby arXiv:2103.02646, 2021), coding-for-machines (~2020+). Applied to a domain that **could not have existed in the 1800s**: the argmax partition of a frozen convolutional scorer over dashcam video, compressed for machine perception under a contest's exact byte-counted oracle. Chasles would recognize the screw on sight; he could not have imagined the SegNet whose level sets it warps. That fusion-in-a-new-domain IS the originality thesis — and it stays UNVALIDATED (means≠ends) until a byte-closed exact row beats 0.19110.
