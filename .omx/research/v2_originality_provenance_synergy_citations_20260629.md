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
