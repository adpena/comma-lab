# FRESH-EYES ADVERSARIAL REVIEW — task-space-native Morse-Smale parallax pose carrier (2026-07-08)

**Reviewer:** independent (author≠reviewer). **Design under review:**
`pose_taskspace_native_morse_smale_depth_warp_design_20260708.md` (4e2a11855) + DAG FEED-posetaskspace.
**Axis:** `[deep-math re-derivation + code-trace of primary artifacts + tiny local numpy sanity; n600
target; read-only]`. pid 63069 + run dirs UNTOUCHED; NO GPU/paid/heavy. **Pointer contest-CPU 0.19110
UNMOVED — everything here is MEANS.** Every claim labeled MEASURED / RE-DERIVED / INFERRED / ASSUMED.
STORES CONSULTED: the design memo · `pose_representation_deepresearch_20260708` · `xi_pose_coder.py` ·
`warp_real_luma_frame0.py` · `depth_motion.py` · `upstream/modules.py` + `frame_utils.py` +
`evaluate.py` (re-read this turn) · FEED-posehard/-poseresearch/-sdfresearch/-posetaskspace · CLAUDE.md
class-order block (L80).

VERDICT IN ONE LINE: the paradigm (partition-keyed depth-stratified warp of a seg-free frame0) is
**SOUND and worth measuring**, but the design as written has **four load-bearing defects** — an
under-modeled dominant off-plane region, a rate framing that argues the non-binding axis, a
per-region-*scalar* depth model that is the very Option-D the sister deep-research memo registered as
DOMINATED, and an **M-ladder that measures a pair deployment never feeds.** All four are fixable; none
is a paradigm kill. Hardened spec below.

---

## PART 1 — RE-DERIVATION OF THE LOAD-BEARING CLAIMS (verify or overturn)

### 1.1 "Homography is a plane-only warp; γ(p)≡0 off-plane" — RE-DERIVED TRUE, and quantitatively large
The Irani–Anandan–Weinshall split `u(p)=u_planar(H;p)+γ(p)·e` with our `H=K(R−t nᵀ/d)K⁻¹` reproducing
`u_planar` and setting γ≡0 is correct (RE-DERIVED from `warp_real_luma_frame0.homography_from_Rt`
+ `xi_pose_coder.homographies_from_xi`, op-for-op). **New quantitative teeth (RE-DERIVED, tiny numpy
this turn):** with f=910, cy=437, h=1.22, pitch≈0, horizon row v_h≈437. Ground depth profile: row
874→2.5 m, 537→11 m, 492→20 m, 460→48 m. A car at 20 m has its **base at row 493** (ground-contact →
plane warp EXACT there) but its **roof at row 424 — ABOVE the horizon**, where the plane model is
degenerate (v−v_h<0 → behind-camera/∞). The residual parallax flow for off-plane pixels is
`f·t_z·(1/Z−1/Z_plane)·(radial FOE offset)`: for a car-body pixel (trueZ=20, planeZ=48) it is **≈13 px
at city t_z=0.5 m/frame and ≈40 px at highway t_z=1.5 m/frame**, times the normalized offset from the
focus-of-expansion. Against PoseNet's effective ~192–256 px working grid this is **enormous**. So the
plane-only defect is real, spatially concentrated **above/near the horizon and in the image periphery**
(near the FOE it vanishes even off-plane), and O(10–40 px) — consistent with an O(1) d_pose gap.
**Root cause CONFIRMED at the mechanism level.**

### 1.2 "Depth jumps ONLY at occlusion separatrices = a labeled SUBSET of RAG edges" — RE-DERIVED IMPRECISE (overstated)
Depth discontinuities are the object-silhouette set. The RAG (SegNet argmax) edge set is **neither a
superset nor a subset** of it: (a) coplanar semantic edges (road↔lane) carry NO depth jump — correctly
excluded by the design ✓; but (b) **intra-class occlusions are MISSED** — one movable occluding another
(both class 3), a near building edge against a far building (both class 2 "undrivable"), the road
surface receding behind a crest — none is a RAG edge, yet all are depth jumps. So the partition's tie
loci are a **correlated-but-imperfect proxy** for occlusion edges, capturing the largest
(object-vs-background) jumps that carry most parallax-residual mass while missing intra-class jumps. The
design's "labeled subset" phrasing is **optimistic**; corrected claim: *the RAG edges upper-bound the
inter-class occlusion set and are a good but incomplete prior for the true depth-discontinuity set;
intra-class jumps need a within-cell depth field, not a within-cell constant.*

### 1.3 "Sky = rotation-only R(ξ), 0 params" — RE-DERIVED WELL-POSED ONLY FOR TRUE SKY; the split is not clean
γ→0 as Z→∞ is correct: true sky needs only rotation. **But "undrivable" (class 2, 49.3% area — MEASURED,
CLAUDE.md L80) conflates sky (∞) with finite vertical structure (buildings, poles, tree-lines, overpasses)**,
and §1.1 shows structure routinely projects ABOVE the horizon. A hard horizon-line split does NOT
separate sky from structure semantically (buildings extend above v_h). **This is the single largest
defect: the design assigns rotation-only (0 depth params) to the biggest region while a large,
uncounted fraction of it is finite-depth parallax-heavy structure.** RESOLUTION (non-brittle): do NOT
hard-code sky=rotation; let a **per-cell low-order inverse-depth field** fit it — sky pixels fit
inverse-depth≈0 (→ rotation-only automatically), structure pixels fit finite inverse-depth. One
mechanism, no horizon hard-split, and it self-detects sky.

### 1.4 "Ground homography is EXACT for road+lane" — RE-DERIVED TRUE-TO-FIRST-ORDER; small systematic mid-field bias
A homography maps plane→plane exactly under any camera motion (projective), so for pixels truly on a
planar road it is exact. Real ground is not perfectly planar (slope changes, ~2% crown, curbs, banking).
Flat-plane height error δh gives residual ≈ `f·t_z·δh/(h·Z)`: for δh=0.1 m, Z=20 m, t_z=1.5 m → ~5.6 px
(RE-DERIVED). This **vanishes near-field** (road genuinely planar close up) **and near horizon**
(parallax→0), peaking mid-field. It is a systematic bias a per-cell **affine-inverse-depth** (letting the
"ground" cell be a slightly non-horizontal plane) absorbs — so the road cell should ALSO carry a tiny
depth correction, not be assumed perfectly planar. Minor but free to fix inside the same per-cell model.

### 1.5 The ~2.5 cap — RE-DERIVED as a FORMULATION cap, with a CONFOUND flag on its attribution
The measured 2.562 (build self-fit) → 1.793 (trained residual) plateau (MEASURED, run-1, FEED-posehard)
is real and the "`dxi` stays in the 6-DOF planar family → projects target onto planar manifold →
saturates" argument is RE-DERIVED SOUND. **BUT** (confound discipline): the run-1 carrier
(`warp_real_luma_frame0`) pairs a **warped real-luma f0** with a **task-space witness f1** — two frames
from *different sources* (real luma vs task-space render). PoseNet may read part of the ~2.5 as
**appearance mismatch between the two frames**, not purely the planar-parallax defect. So the ~2.5 cap
plausibly **bundles two confounds**: (i) planar-warp defect (§1.1) AND (ii) f0/f1 source-appearance
mismatch. Attributing 100% to planar-parallax is an INFERRED-not-isolated step. The corrected ladder
(Part 2) separates them; the design should not treat "planar warp is the whole story" as settled.

### 1.6 What the M-ladder actually measures vs what deployment feeds — THE METHODOLOGY DEFECT (MEASURED from evaluate.py)
`evaluate.py:74–79` (re-read): `compute_distortion(batch_gt, batch_comp)` feeds the **single decoded
video** `batch_comp` (shape `[B, seq_len=2, H, W, 3]`) to BOTH nets. SegNet reads `x[:,-1]` = frame1;
PoseNet reads the pair. **There is ONE decoded video — you cannot give the two nets different frame1.**
Therefore at DEPLOYMENT: `d_pose = MSE(PoseNet([decoded_f0, witness_f1]), PoseNet([real_f0, real_f1]))`,
where `witness_f1` = the task-space d_seg render (fixed by d_seg) and `decoded_f0` = whatever we
synthesize (seg-free). **The M-ladder in BOTH memos measures `PoseNet([real_f0, warp(real_f0)])` — an
all-real-luma pair whose frame1 is a warp, not the witness render.** That pair matches NEITHER
deployment arch faithfully. This conflates the appearance confound (§1.5) with the warp-model axis and
does not test the actual seg-free-f0 / task-space-f1 constraint. **This is the corrected ladder's
reason to exist (Part 2).**

### 1.7 "~1–2 KB total, ~100× smaller than Quantizr" — RE-DERIVED INCONSISTENT + arguing the non-binding axis
The stored ξ payload ALONE is **~2.7 KB (delta_res) / ~3.2 KB (delta_ar) at n600** (MEASURED, per
`xi_pose_coder`/`xi_spline_residual_coder` docstrings; ~0.0021 coded rate at the #205 precision). So
"~1–2 KB TOTAL including occlusion bits + off-plane depths" **undercounts ξ itself.** Honest estimate
(RE-DERIVED, with error bars): ξ ~1–3 KB (precision-dependent) + occlusion-edge bits (≤10 RAG edges,
temporally coherent, shareable → ~hundreds B) + per-clip depth field (the real variable: per-cell
affine-inverse-depth ≈ 2 params × N_cells × N_keyframes ~ hundreds B–low KB; dense low-rank/INR
~0.5–5 KB/keyframe amortized) ≈ **~3–10 KB total**, error bars ±factor-2. That is still **~30–90×**
smaller than Quantizr's ~270 KB, not 100×. **More importantly the rate framing is a RED HERRING:** the
sister deep-research memo MEASURED the byte affordance to reach ancestor d_pose at **~6.3 MB** — rate is
NOT the binding constraint by three orders of magnitude. Selling this carrier on "beats Quantizr on
rate" argues the axis that doesn't bind. **The real prize is that it reaches low d_pose WHILE PRESERVING
the store-nothing/witness paradigm — it does NOT collapse into a photometric codec (Option B's fatal
side-effect: a real f1 also ≈solves d_seg → the witness degenerates into a plain video codec).** Reframe
the value proposition to paradigm-preservation, not byte count.

### 1.8 Per-region SCALAR depth = Option D in disguise (the design regresses from the deep-research conclusion)
The sister deep-research memo (§2, §5) is explicit: the sufficient statistic is a **per-pixel depth field
D** (SfMLearner `K·T(ξ)·D·K⁻¹`), and Option D ("richer twist / per-region twists / piecewise-planar")
is **DOMINATED — "does NOT close ~1.8–2.5"** because piecewise-planar is only a coarse multi-plane
proxy. The design's §3 "movable islands = ONE stored inverse-depth scalar each" and "vertical structure
= coarse depth (a handful of scalars)" is **precisely that piecewise-constant/piecewise-planar proxy** —
i.e., the design proposes the representation its own sister memo registered as insufficient. A car spans
a depth RANGE (fronto-parallel-plane approximation error); a building/tree-line spans near→far; one
scalar cannot represent either. **Corrected representation:** the partition defines the **support** of a
piecewise **low-order depth FIELD** — each cell carries a small parametric inverse-depth surface (affine
in inverse-depth = a 3-D plane per cell, ~2–3 params/cell), C0-discontinuous only across occlusion RAG
edges, with an optional small learned residual-depth on the off-plane cells for intra-class jumps
(§1.2). This is strictly richer than Option D (continuous within a cell, not rigid) and strictly cheaper
than dense D (parametric, partition-supported). It is the honest middle rung — and the M-ladder must
trace the whole rung sequence, not assert one.

---

## PART 2 — CORRECTED M-LADDER (n600, through-R, deployment-geometry-faithful)

Binding rules for EVERY rung (else it is a toy, not evidence):
- **n600** — all 600 pairs, real `gt` for the d_pose target `PoseNet([real_f0, real_f1])` per pair
  (MSE first-6, `cpu_verdict_d_pose_batch`, frozen CPU-torch PoseNet, NEVER MPS).
- **Through the REAL R operator** — every synthesized frame goes bicubic↑874→uint8@camera→bilinear↓384×512
  (`apply_contest_faithful_roundtrip_nhwc`), exactly as inflate/eval; no clean-float shortcut.
- **Deployment-faithful pairing** — PoseNet's frame1 must be the **actual witness task-space render**
  (pulled from the live checkpoint's render, or a faithful task-space render), NOT `real_f1`, NOT
  `warp(real_f0)`, for the task-space arm. This is the whole point of §1.6.

Run it as a **2-axis grid** (warp-model complexity × frame1-appearance), reporting the graded
d_pose(bytes) FRONTIER — not a go/no-go threshold:

**Axis A — warp-model complexity (the depth ladder):**
- A0 plane-only homography `H(ξ)` (control; must reproduce the ~2.5 cap → validates apparatus).
- A1 partition + rotation-only sky / id hood / plane ground + per-movable-island **scalar** (the design as written).
- A2 partition + per-cell **affine-inverse-depth field** (the §1.8 correction), sky fits inverse-depth≈0.
- A3 + small learned residual-depth on off-plane cells only (intra-class jumps, §1.2).
- A4 dense monocular-depth D (off-the-shelf net, one-shot) → SfMLearner warp = the deep-research L2
  UPPER BOUND on any depth-warp.
- A5 real dense flow (RAFT/classical) warp = the achievable floor of ANY warp carrier (only
  disocclusion holes remain).

**Axis B — frame1 appearance (the store-nothing decider = R1/M3):**
- B-real: PoseNet frame1 = warp/real luma (isolates the pure warp-model axis, no appearance confound).
- B-witness: PoseNet frame1 = the **actual task-space witness render** (the deployment truth). The gap
  B-witness − B-real at fixed warp model = the **task-space-appearance penalty** (does PoseNet need
  photometric texture, or just correct geometry?). THIS is the axis both memos leave implicit.

**The missing rung both memos omit (highest-EV store-nothing test):** **A-derive-f0-free** — synthesize
frame0 by inverse-depth-warping the **witness f1 itself** (backward by ξ+D) instead of storing a
keyframe. If `PoseNet([depth-unwarp(witness_f1), witness_f1]) ≈ target`, then **frame0 costs ZERO stored
bytes** (derived free from the render we already have + already-stored ξ + the depth field) — strictly
better than Option A's stored keyframe, and the true task-space-native store-nothing win. Add it.

**Depth is per-KEYFRAME, propagated FREE (store-nothing extension the design should state):** the static
scene's depth at frame t is deterministic given a keyframe depth + the ego trajectory ξ (already
stored). So store D once per keyframe and DERIVE per-frame depth via the stored ξ (rule-118 free) —
depth is NOT re-stored per pair.

**Free companion (genuinely ~$0, read-only):** scatter per-pair d_pose vs per-pair `|t_forward|` (= ρ_z
of the dequantized stored ξ). `cpu_verdict_d_pose_batch` returns per-pair values (RE-DERIVED from the
base trainer signature) and ξ is the stored payload, so this is one forward pass, not a log-scrape (mild
correction to the memo's "0 compute from existing telemetry" — it is ~$0 but a recompute). **This
BOUNDS the win before any warp is built:** near-static pairs already sit ~0 (no parallax → homography
suffices); the 1.793 mean is carried by the fast-forward tail, so the depth fix can only recover that
tail's mass. Run this FIRST.

---

## PART 3 — MISSING FORCES / TERMS (deep-math completeness)

- **Rolling shutter (INFERRED, floor-contributor):** the comma sensor has rolling shutter; the real
  target `PoseNet([real_f0,real_f1])` bakes it in, but a global-shutter geometric warp cannot reproduce
  row-dependent readout distortion → a small irreducible residual on ANY pure-geometric warp (A2–A4).
  Bounds the achievable minimum; A5 (real flow) will still beat it because flow captures the actual
  displacement. Flag, don't block.
- **Lens distortion / rectification (ASSUMED-unverified — CHECK before A2):** `xi_pose_coder` uses a
  pinhole K (fx=fy=910, pp centered). If the eval/keyframe frames are RAW (distorted) rather than the
  openpilot-rectified "medmodel" pinhole frame, the warp geometry is wrong at the periphery — exactly
  where §1.1 parallax residual is largest. **Verify against openpilot #325–327 whether the frames
  PoseNet sees are rectified;** if raw, the warp needs the distortion model or peripheral geometry is
  off. Cheap to check, potentially confounds A2–A5.
- **Chroma subsampling (RE-DERIVED, mild POSITIVE):** `rgb_to_yuv6` = 4 full-res luma channels
  (space-to-depth of the 2×2 Y block — luma is NOT downsampled) + 2 chroma at 4:2:0. Motion/parallax
  cues live in luma edges (full res); chroma is 2× coarse and forgiving. So the warp must get **luma
  geometry** right; chroma imprecision is tolerated → mildly favorable for the task-space arm.
- **Which 6 of 12 PoseNet dims are scored (ASSUMED):** `compute_distortion` uses `out['pose'][...,:6]`
  of a generic `Linear(32,12)` head — the semantics (translation+rotation vs something else) are NOT
  knowable from code. The design's "warp must get translation+rotation right" and the "d_pose ∝
  |t_forward|" corollary ASSUME the scored 6 are the 6-DOF ego pose. Plausible (openpilot pose head) but
  UNVERIFIED; the free companion's scatter empirically tests the corollary regardless of semantics.
- **FOE structure (RE-DERIVED nuance):** forward driving puts the focus-of-expansion near image center,
  so central off-plane structure (lead car dead ahead) has SMALL parallax residual while PERIPHERAL
  structure (side buildings, side cars) dominates — and peripheral parallax is exactly what
  disambiguates rotation from translation for a global regressor. Reinforces §1.3 (periphery matters
  most) and §1.4 (near-field/near-horizon vanish).
- **Independent object motion (RE-DERIVED, likely NON-fatal — the non-pessimistic overturn of seeded
  weak-point #2):** the warp models ego-parallax only; moving vehicles have flow = ego + own-motion.
  BUT PoseNet regresses EGO pose and is trained on real driving saturated with moving cars → it is
  ego-robust by construction (down-weights independent motion). Since `d_pose = MSE(PoseNet(gen),
  PoseNet(orig))` and PoseNet ignores moving-car regions for its ego estimate in BOTH inputs, a WRONG
  warp on movable (1.56% area — MEASURED L80) perturbs the output little. **Prediction: movable's depth
  scalar may be droppable entirely** (persist/rotation-warp movable, save its bytes). Testable as an A1
  ablation. This turns seeded weak-point #2 from a risk into a potential simplification.
- **Disocclusion holes (design R3, RE-DERIVED small):** localized to occlusion RAG edges; the existing
  persist-fallback (`warp_frame0_native_numpy`) fills them; PoseNet's coarse readout washes small holes
  out. Measure hole area as a diagnostic; not expected to bind.

---

## PART 4 — GRADED VIABILITY (verdict-scope ladder held: INSTANCE < FORMULATION < FAMILY < PARADIGM)

Not a binary. As a rate-distortion object the design occupies a real, unoccupied, paradigm-preserving
point; its viability is a **frontier of reachable d_pose vs representation richness**, decided by the
Part-2 grid:

- **PARADIGM (partition-keyed depth-stratified warp of a seg-free frame0, dual-use ξ):** VIABLE and
  under-explored. Not challenged by anything here; it is the correct task-space-native structure and the
  free companion + A0 will confirm the root cause for ~$0. **Do not let any negative below escalate to
  this level.**
- **FAMILY (depth-warp carriers, Option A):** the deep-research memo already ranks this highest-EV and
  store-nothing-preserving; this review strengthens that (deployment-faithful ladder + free-f0 rung +
  keyframe-depth propagation). Reachable d_pose is PREDICTED ≪1, owed to A2–A4.
- **FORMULATION (the design's SPECIFIC per-region-*scalar* depth + hard sky-split + all-real-luma
  ladder):** **this is where the defects bite** — §1.3 under-models 49.3% of the image, §1.8 is the
  DOMINATED Option-D proxy, §1.6 measures the wrong pair. Expected outcome: A1 (design as written)
  partially closes but likely does NOT reach ≪1 on the fast-forward tail; A2 (affine-inverse-depth
  field) is the formulation that should. **If A1 disappoints, that falsifies the SCALAR formulation, NOT
  the depth-warp family** — the pre-registered next reformulations are A2 → A3 → A4 (already laddered),
  so no verdict escalates.
- **INSTANCE:** any single rung's number is an instance; report the frontier.

Non-pessimism enumerated (what WOULD make it work, pre-registered): (1) per-cell affine-inverse-depth
(A2); (2) small learned residual-depth on off-plane cells (A3); (3) off-the-shelf dense depth as the
upper bound (A4); (4) derive-f0-free-from-f1 for true store-nothing; (5) keyframe-depth + ξ propagation;
(6) drop movable depth (ego-robust PoseNet). Rate is affordable at every rung (6.3 MB affordance), so
the ONLY binding question is reachable d_pose in the deployment geometry — and every fallback is cheaper
than the affordance ceiling.

**Net graded assessment: PROCEED to the corrected Part-2 ladder, free companion + A0 first.** The design
is a real contribution once (a) the depth model is upgraded from per-region scalar to per-cell
low-order field, (b) the rate claim is reframed as paradigm-preservation not byte-count, (c) the ladder
is made deployment-faithful (witness-f1, through-R, n600) with the free-f0 rung added.

---

## PART 5 — HARDENED-DESIGN DELTA (fold into the design memo)

1. **§1/§3 (depth model):** replace "one inverse-depth scalar per movable island / a handful of scalars
   for structure" with **a piecewise low-order (affine-in-inverse-depth) depth FIELD supported on the
   Morse cells**, C0-jumps only across occlusion RAG edges, sky fitting inverse-depth≈0 (removes the
   brittle horizon hard-split), optional small learned residual-depth on off-plane cells for
   intra-class jumps. State explicitly that this is Option A (per-pixel-depth family), NOT Option D
   (piecewise-planar), which the sister memo registered as dominated.
2. **§1.3 correction (the biggest defect):** "undrivable" (49.3%) is NOT sky — treat it as
   finite-depth structure via the per-cell field; do not assign it rotation-only.
3. **§4 (byte accounting):** correct "~1–2 KB total" → **~3–10 KB (±factor-2), ξ alone ~2.7 KB
   MEASURED**; and **reframe the value proposition from "~100× rate win" (non-binding; 6.3 MB
   affordance) to "reaches low d_pose while PRESERVING store-nothing/witness (does not collapse into a
   codec like Option B)."**
4. **§6 (the ladder) — replace wholesale with the Part-2 grid:** n600, through real R, PoseNet frame1 =
   actual witness render for the task-space arm; 2-axis grid (warp-model A0–A5 × appearance B-real /
   B-witness); ADD the derive-f0-free-from-witness-f1 rung; state depth is per-keyframe + ξ-propagated
   (rule-118 free), not per-pair.
5. **§1.5 confound:** confirm whether run-1's ~2.5 fed (warped-real-f0, witness-f1) mismatched sources;
   attribute the cap to planar-parallax ONLY after B-real vs B-witness separates the appearance
   confound.
6. **New pre-launch check (§3/§5):** verify frames PoseNet sees are openpilot-rectified pinhole (else
   add lens distortion to K) before trusting A2–A5 peripheral geometry.
7. **Add the free companion as STEP 0** (per-pair d_pose vs |t_forward| scatter) — it bounds the
   achievable win and predicts which pairs the depth fix helps, for ~$0.
8. **Verdict-scope guard:** annotate that A1-disappointment falsifies the scalar FORMULATION only; A2–A4
   are the pre-registered reformulations.

---

## PART 6 — COUNCIL-FLAGGED-EQUATION STATUS

Confirmed NOT registered (flagged only; grep of `src/tac/canonical_equations/` +
`canonical_equations_registry.jsonl` returns nothing — correct, anchors owed):
- `morse_smale_stratified_parallax_dpose_v1` (design memo) — anchor owed to the Part-2 ladder + byte-close.
- `posenet_planar_parallax_dpose_floor_v1` (deep-research) — the d_pose ∝ translational-parallax-energy
  law; anchor = the free-companion scatter (L0) + A0. Its per-pair |t| scaling rests on the ASSUMED
  scored-6-dims semantics (§Part 3) — register only after the scatter confirms it.
- `pose_sufficient_statistic_depth_pose_v1` (deep-research) — {compact per-clip D + stored ξ} drives
  d_pose→0; anchor owed to A2/A4.

**Review recommendation:** keep all three COUNCIL-FLAGGED, do NOT register until the corrected
deployment-faithful ladder (A2 through-R, n600, witness-f1) produces the anchor. The design's headline
equation should be revised to name the depth FIELD (not per-region scalar) as the generator, so the
registered form matches the hardened representation, not the dominated one.

---

## FINAL STATE
$0 read-only (pid 63069 + run dirs UNTOUCHED; NO launch/train/GPU). Pointer contest-CPU 0.19110
UNMOVED — MEANS. Paradigm SOUND; four FORMULATION-level defects (under-modeled 49.3% off-plane region;
non-binding rate framing; per-region-scalar = dominated Option-D; deployment-infaithful ladder) each
fixed above without paradigm escalation. Highest-EV next unit: the free companion (Step 0) then A0/A2
on the corrected grid.
