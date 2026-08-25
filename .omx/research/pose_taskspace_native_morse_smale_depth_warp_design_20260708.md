# DESIGN — Task-space-native pose: Morse-Smale-stratified parallax warp (2026-07-08)

STORES CONSULTED: FEED-poseresearch/FEED-posehard/FEED-sdfresearch (sub015 DAG, 2026-07-08) ·
pose_representation_deepresearch_20260708 (560b16634) · pose_sidecar_reuse_assessment_20260708
(f8f8f4479) · `src/tac/boundary_math/xi_pose_coder.py` (the store-nothing screw = ground homography,
_CAMERA_HEIGHT_M=1.22, `H=K(R−t nᵀ/d)K⁻¹`, n=[0,−cos p,−sin p], d=1.22) · `src/tac/lie/` (se3/so3/
screw_blend/se3_bspline) · `src/tac/boundary_math/warp_real_luma_frame0.py` · `src/tac/depth_motion.py`
· openpilot geometry reconciliation #325/#326/#327 · comma2k19 ego-GT #158 · tasks #194 (per-class
warp / canonicalize-to-ground), #238 (real pose MEASURED), #248 (pose-carrier ladder). Author: main
(operator design riff 2026-07-08). Pointer **0.19110 UNMOVED** — this is a DESIGN + a $0 measurement
plan; nothing here is a score until byte-closed through upstream/evaluate.py.

## 0. THE CLAIM (operator riff, worked through)
There is a pose carrier NATIVE to the SDF-level-set-witness + Morse-Smale framing that is **better than
Quantizr/HNeRV on rate** (~1–2 KB vs their ~270 KB) at equal-or-better d_pose, because **the depth field
the parallax warp needs is already stratified by the exact argmax partition the witness computes for
d_seg.** The store-nothing screw was not wrong — it was a plane-only warp that is EXACT for the ground
plane and sets depth-parallax γ(p)≡0 everywhere else. The fix is surgical and free-per-region.

## 0b. REVIEW-HARDENED DELTA (fresh-eyes adversarial pass, review memo 5711a4fdf; SUPERSEDES the specific claims below where they conflict — original derivation kept intact per append-only provenance)
Verdict: PARADIGM SOUND, four FORMULATION-level defects, all fixable, none a paradigm kill. Corrections:
- **D1 — depth is a FIELD, not a scalar.** "One inverse-depth scalar per island / handful of scalars"
  (§3) IS the sister-research's DOMINATED Option-D (piecewise-planar "does NOT close ~1.8–2.5"). Correct
  form: the partition defines the SUPPORT of a per-cell **affine-in-inverse-depth field** (~2–3 params/
  cell) + a small learned residual on OFF-PLANE cells for intra-class depth jumps. Revise the headline
  equation to name the depth FIELD, not the scalar.
- **D2 — "undrivable" is 49.3% area (MEASURED L80), the BIGGEST region, and it is under-modeled.** It
  conflates true sky (∞, rotation-OK) with finite vertical structure (parallax-heavy). §3's "sky=R(ξ),
  0 params" applied to the whole class is wrong. DROP the brittle horizon hard-split; a per-cell inverse-
  depth field fits sky→≈0 AUTOMATICALLY. RE-DERIVED: a 20 m car roof projects to row 424, ABOVE the
  horizon (437) where the plane model is degenerate; off-plane residual ~13 px (city) → ~40 px (highway)
  vs PoseNet's ~192–256 px grid — i.e. real and worth modeling.
- **D3 — rate does NOT bind; drop the "~1–2 KB / ~100× win" framing.** ξ alone is ~2.7 KB MEASURED, so
  honest total ~3–10 KB (±2×). Byte affordance is ~6.3 MB (3 orders of magnitude of headroom). VALUE is
  NOT "100× rate win" — it is "reaches low d_pose while PRESERVING store-nothing/witness (does NOT
  collapse into a codec like Option B)." §4's rate argument is the non-binding axis.
- **D4 — "occlusion = subset of RAG edges" is a good-but-INCOMPLETE prior.** Intra-class occlusions
  (car-behind-car, near-vs-far building) are missed. Keep as a prior, not a complete labeling. Also: the
  ~2.5 cap may bundle a confound (planar-warp defect AND warped-real-f0/task-space-f1 appearance
  mismatch) — attribute to parallax ONLY after the ladder separates them.
- **M-LADDER METHODOLOGY FIX (the important one).** evaluate.py:74–79: ONE decoded video feeds BOTH
  nets, so deployment d_pose = `MSE(PoseNet([decoded_f0, WITNESS_f1]), PoseNet([real_f0, real_f1]))`.
  §6's ladder measures `PoseNet([real_f0, warp(real_f0)])` — an all-real-luma pair whose f1 is a warp —
  matching NEITHER deployment arch. CORRECTED ladder: n600, real gt, through the real R operator; the
  task-space arm's PoseNet frame1 = the ACTUAL witness render; a 2-axis grid (warp-model A0-plane → A5
  real-flow × appearance B-real/B-witness); PLUS the missing highest-EV rung — derive f0 FREE by
  inverse-depth-warping the witness f1 (zero stored keyframe = the TRUE store-nothing win); depth stored
  per-keyframe + ξ-propagated (rule-118 free). Step 0 = the free-companion d_pose-vs-|t_forward| scatter
  ($0) to bound the achievable win up front. M2/M3 report GRADED d_pose (an RD frontier), never a
  threshold verdict.
- **MISSING FORCES:** rolling shutter (irreducible geometric-warp floor); **lens distortion /
  rectification — CHECK whether PoseNet frames are openpilot-rectified pinhole before trusting
  peripheral A2–A5 geometry**; the scored 6-of-12 PoseNet dims' semantics are ASSUMED (generic Linear
  head — verify); independent object motion is likely NON-fatal (PoseNet is ego-robust by training →
  movable's 1.56% may be droppable, a simplification not a risk). Full detail: review memo 5711a4fdf.

## 0c. MEASUREMENT CONFIRMATION — the fork is RESOLVED (pose_carrier_arms_measured_20260708, 16030e6bf)
n8 direction-only, real byte-close, frozen CPU-torch PoseNet, positive control valid (gt-pair d_pose
1.2e-12). Same-pairs controlled contrast (robust; absolute n600 owed):
- **The ~2.5 cap is PAIR-CONSISTENCY-bound, NOT source-dependent.** PoseNet reads the flow BETWEEN the
  two frames — indifferent to photorealism. store-nothing GENERATED pair (warped witness-f0 + witness-f1)
  = **d_pose 1.995** (≈ run-1's 1.79). A perfect REAL f0 paired with the cartoon witness-f1 = **10.42**
  (WORSE — mixing real+cartoon breaks consistency) + rate_term **≈573** (860 MB keyframes). **Option B
  (store real frame) is DOUBLY DOMINATED → DEAD.**
- **The binding wall is the rank-6 HOMOGRAPHY FLOW MODEL** (caps a *consistent* generated pair at
  ~1.8–2.0; run-1's trained dxi is only an ~11% refinement over the deterministic 1.995). Breaking sub-2
  requires the generated pair to reproduce TRUE optical flow — KEEP the generated source, UPGRADE the
  flow model. That is EXACTLY this design (per-region depth-stratified flow). The fork collapses to ONE
  path: this one. 3.4e-5 stays ancestor-borrowed (ancestor read a consistent PHOTOMETRIC pair).
- **Corrects §6's ladder framing:** do NOT "warp real_f0"; the deployment-faithful arm is the CONSISTENT
  generated pair (derive f0 by depth-warping witness f1). The A0-plane control should reproduce ~1.995.
  n600 confirmation + a dense-flow floor are owed before any promotable pose number.

## 1. WHY store-nothing was RIGHT-for-the-ground-plane, wrong off-plane (measured mechanism)
`xi_pose_coder.py` warps by the ground homography `H(ξ)=K(R−t·nᵀ/d)K⁻¹`, d=1.22 m. Irani–Anandan–
Weinshall: real flow `u(p)=u_planar(H;p)+γ(p)·e`, γ(p)=depth-scaled parallax. A homography reproduces
`u_planar` EXACTLY and sets γ≡0. So the screw is **exact for pixels ON the ground plane** (road, lane
markings — they lie on the plane at d=1.22 geometry) and **exact for pixels at infinity** (sky: γ→0
because parallax vanishes at ∞, only rotation remains). It is WRONG only for OFF-PLANE 3-D structure:
movable vehicles (finite depth ≠ ground) and vertical structure (buildings). PoseNet loses the parallax
cue that separates forward-translation from rotation → d_pose caps ~2.5 (MEASURED, FEED-posehard).
The defect is spatially CONCENTRATED in exactly the regions the partition already isolates.

## 2. THE UNIFICATION (rigorous, not a decorative identity chain)
- The image partition `P(x)=argmax_c(φ_c(x)+b_c)` has separatrices = tie loci (tropical/RAG; CONFIRMED).
- The 3-D scene projects to this partition; depth Z(p) is a function on the image.
- **Depth is smooth WITHIN a class-region and jumps only at OCCLUSION separatrices** — a LABELED SUBSET
  of the RAG edges (near-class abutting far-class). Coplanar boundaries (road↔lane) carry NO depth jump;
  occlusion boundaries (road↔movable, ground↔sky-at-horizon, hood↔road) DO. The witness's tie loci give
  the CANDIDATE depth-edge set for free; which edges are occlusion vs coplanar is a per-edge bit.
- Therefore depth is a PER-MORSE-CELL model, and the cells are FREE (already computed for d_seg). This
  is the task-space-native structure: **one partition serves both d_seg (the labels) and d_pose (the
  per-cell depth-warp).** The SAME twist ξ warps the partition (d_seg) and IS the pose (d_pose) — the
  dual-use screw (CLAUDE.md unified-flow; tac.lie).

## 3. THE PER-REGION WARP LAW (what falls out)
Per Morse cell, warp frame0→frame1 under ego-twist ξ by the depth-appropriate model:
- **Road + Lane (coplanar ground):** `H_ground(ξ)` — CLOSED-FORM from openpilot calibration (d=1.22,
  pitch), ZERO stored depth params. This is the existing `xi_pose_coder` path, now correctly SCOPED to
  ground pixels only. Dominant image area → near-free.
- **Sky / Undrivable-at-∞:** rotation-only `R(ξ)` (translational parallax vanishes). Zero params.
- **MyCar / hood:** identity / fixed near-field (IoU 0.994 static, #139). Zero params.
- **Movable islands (1.56% area):** each island gets ONE stored inverse-depth scalar → parallax
  `(1/Z)·K·t`. openpilot's model outputs lead-vehicle distance; comma2k19 has it (#158). ~a few bytes.
- **Vertical structure (the off-plane part of "undrivable"):** one coarse depth model (e.g. depth ∝
  1/row via horizon, or a low-order per-region field). A handful of scalars.
Full flow: `u(p)=K[ω]×K⁻¹p + (1/Z(p))K·t`, with Z(p) supplied PIECEWISE by the partition. This is
plane+parallax with the parallax structure handed over by the segmentation — the thing openpilot +
our witness already produce.

## 4. BYTE ACCOUNTING (why it beats Quantizr/HNeRV)
Quantizr/HNeRV get low pose as a BYPRODUCT of full-frame RGB reconstruction (~270 KB archive; the pose
blob is a FiLM input welded to a ~214 KB mask + ~56 KB generator — pose_sidecar_reuse_assessment).
This carrier stores: ξ trajectory (store-nothing, derive-H free per #257) + per-edge occlusion bits +
sparse off-plane inverse-depths (movable islands + coarse structure) ≈ **~1–2 KB**. Same twist, same
partition (both already paid for d_seg). If d_pose lands low, this DOMINATES Quantizr on rate by ~100×
at equal pose — the "better than Quantizr" the operator sees. It is Option A (depth-consistent warp)
made TASK-SPACE-NATIVE: depth free-per-region (partition-keyed + openpilot-calibrated) instead of a
generic learned dense depth net.

## 5. RISKS (honest; each has a measurement)
- **R1 — PoseNet appearance vs motion sensitivity (THE make-or-break):** does PoseNet read correct
  motion from a geometrically-correct warp of a TASK-SPACE (non-photometric) f0, or does it need real
  luma appearance? If motion suffices → fully store-nothing-class (warp a cheap f0). If appearance
  needed → store real f0 (Option B bytes) but STILL derive f1 by geometric warp (cheaper than storing
  both). Decided by M3 below.
- **R2 — movable-depth accuracy:** movable is sparse (1.56%); PoseNet is a global 6-DOF regressor. Is
  the off-plane parallax mass large enough to matter, and is a per-island scalar accurate enough?
  Decided by the |t_forward| scatter + M2.
- **R3 — occlusion holes:** the warp exposes disocclusion at depth edges. These coincide with the
  separatrix (Morse-Smale) → localized, inpaintable from the partition. Measure hole area; likely small.

## 6. THE DECISIVE $0 MEASUREMENT LADDER (read-only through frozen CPU-torch PoseNet; no GPU/paid)
- **M1 (control):** ground-homography-only warp of real_f0 → PoseNet d_pose. Should REPRODUCE the ~2.5
  cap (confirms the plane-only defect is the whole story).
- **M2 (the test):** per-region warp of real_f0 — ground `H(ξ)` + sky `R(ξ)` + hood id + movable
  stored-inverse-depth + coarse structure → PoseNet d_pose. **Does it collapse ≪ 2.5?** If yes, the
  parallax stratification is the fix and Option A is viable at ~1–2 KB.
- **M3 (store-nothing viability):** same per-region warp of the TASK-SPACE witness f0 (not real luma) →
  d_pose. Decides R1: store-nothing-class vs store-real-f0 (Option B bytes).
- **Free companion (0 compute):** scatter per-pair d_pose vs |t_forward| from stored ξ + fraction of
  parallax mass that is off-plane — confirms the root cause and bounds R2 before any warp is built.
Ladder is byte-honest: M1/M2/M3 measure d_pose only; the RATE claim (~1–2 KB) is byte-closed separately
through the L13 format before any exact row.

## 7. RELATION TO EXISTING WORK (build-on, don't rebuild)
This COMPLETES #194 (per-class warp / canonicalize-to-ground / dual-quat blend — the per-region law is
its missing depth stratification), consumes #325–327 (calibrated ground homography), #158 (comma2k19
ego-GT + lead depth), tac.lie (ξ), `warp_real_luma_frame0.py` (the warp op M2/M3 extend), and
`depth_motion.py` (existing depth-motion surface — audit for reuse). It is the top rung of the #248
pose-carrier ladder and the concrete form of Option A from FEED-poseresearch. Canonical equation
`morse_smale_stratified_parallax_dpose_v1` COUNCIL-FLAGGED (anchor owed to M2/M3 + byte-close).

---

## Observability surface

*(OBSERVABILITY-ADDENDUM 2026-08-25 — APPEND-ONLY per Catalog #110/#113. This
section is an INDEX into this memo's own content per Catalog #305's 6 facets;
it adds no new claim. Facets with no counterpart in this memo say so plainly.)*

1. **Per-layer inspection** — §3 "THE PER-REGION WARP LAW (what falls out)" is per-region by construction, so each stratum's warp is inspectable alone; §2 "THE UNIFICATION" states which layers are being identified.
2. **Per-signal decomposition** — §4 "BYTE ACCOUNTING (why it beats Quantizr/HNeRV)" decomposes the byte cost per component; §1 decomposes the store-nothing result into its right-for-the-ground-plane and wrong-off-plane parts.
3. **Run-to-run diff** — §0c "MEASUREMENT CONFIRMATION — the fork is RESOLVED (`pose_carrier_arms_measured_20260708`, `16030e6bf`)" is a multi-arm comparison; the arms are what make two builds diffable.
4. **Post-hoc query** — named surfaces are `src/tac/boundary_math/warp_real_luma_frame0.py`, `src/tac/boundary_math/xi_pose_coder.py`, `src/tac/depth_motion.py`; the authority is `upstream/evaluate.py` through the frozen CPU-torch PoseNet.
5. **Cite-chain** — §0b "REVIEW-HARDENED DELTA (fresh-eyes adversarial pass, review memo `5711a4fdf`)" supersedes conflicting claims while keeping the original derivation intact per append-only provenance; §7 "RELATION TO EXISTING WORK (build-on, don't rebuild)" attributes the reused surfaces.
6. **Counterfactual hooks** — §5 "RISKS (honest; each has a measurement)" pairs every risk with the measurement that would fire it; §6 "THE DECISIVE $0 MEASUREMENT LADDER (read-only through frozen CPU-torch PoseNet; no GPU/paid)" is the ordered ablation ladder.
