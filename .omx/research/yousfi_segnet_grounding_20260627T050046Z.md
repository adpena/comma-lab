# Yousfi SegNet / comma10k Grounding — for the Witness SDF Partition + seg→pose Solve

Landed 2026-06-27T05:00:46Z. Research-only memo (web + read-only; NO GPU touched — the
d_seg level-set descent (pid 72600, FEED-cv/cw) ran untouched throughout).

`research_only: true` · score_claim=false · promotable=false · authority=research-signal.
Pointer UNMOVED contest-CPU 0.19110.

**Source-tagging discipline (NO-FAKE):** every claim is tagged **[CONFIRMED:<src>]**
(I read the file/line) or **[INFERRED]** (derived/design) or **[MEASURED:<ours>]** (our
own prior measurement, cited from the task prompt / DAG). Web sources are cited inline.

---

## 0. WHO + the contest scorer (CONFIRMED baseline)

- Yassine Yousfi — comma.ai; Fridrich's PhD student, Binghamton DDE Lab (steganalysis);
  ALASKA-#1 winner. The contest is **inverse steganalysis**: the scorer is a *detector*;
  our witness must embed the task-relevant signal *undetectably* (= keep the scorer's
  argmax/pose identical) while spending minimum bytes. **[CONFIRMED:** local
  `.omx/research/alaska_yousfi_repo_deep_research_landed_20260530.md` +
  `council_yousfi_voice_..._20260530.md`; his bio yassineyousfi.github.io.]
- Contest **SegNet** = `smp.Unet('tu-efficientnet_b2', classes=5, activation=None,
  encoder_weights=None)`; `preprocess_input` takes **last frame only** `x[:,-1,...]`,
  bilinear-resizes to **(W=512, H=384)**; `compute_distortion` =
  `mean( argmax(out1) != argmax(out2) )` — unweighted 5-class **argmax-disagreement**
  (pixel-accuracy style). **[CONFIRMED:** `upstream/modules.py:103-113`, read this session.]
- Local intake already had the inverse-steganalysis seg framing + stride-2 blind-spot +
  square-root-law + detector-informed-embedding canon. This memo ADDS the precise
  comma10k **class semantics, argmax index ordering, and per-class seg→pose roles**, which
  were NOT in local intake.

---

## A. SEG class semantics + argmax index structure

### A.1 The five classes + RGB (CONFIRMED, commaai/comma10k README)
comma10k masks are P/8-bit PNGs with **anti-aliasing DISABLED** (hard, exact-color
boundaries — no soft transition pixels). **[CONFIRMED:** github.com/commaai/comma10k
README.] Road camera uses 5 classes (a 6th, `#00ccff` "movable in my car", is
driver-camera `imgsd`-only and does NOT apply to the contest 0.mkv road camera):

| RGB hex | RGB | class | comma10k labeling note |
|---|---|---|---|
| `#402020` | (64,32,32) | **Road** | "anywhere nobody would look at you funny for driving" |
| `#ff0000` | (255,0,0) | **Lane markings** | EXCLUDES turn arrows / crosswalks (thin lines only) |
| `#808060` | (128,128,96) | **Undrivable** | obstacles + buildings + **sky/horizon** |
| `#00ff66` | (0,255,102) | **Movable** | vehicles, pedestrians, animals |
| `#cc00ff` | (204,0,255) | **My car (ego)** | hood + anything mounted inside; no reflections |

comma.ai's own grouping by motion (this is the seg→pose backbone, §C):
**moves-with-scene** = {Road, Lane, Undrivable+sky}; **moves-itself** = {Movable};
**moves-with-you** = {My car}. **[CONFIRMED:** comma.ai crowdsourced-segnet blog +
comma10k README.]

### A.2 THE ARGMAX INDEX ORDERING — the load-bearing finding
Yousfi's baseline (`LitModel.py`, branch `main`) declares
`class_values: List[int] = [41, 76, 90, 124, 161, 0]` and builds the label by
`np.stack([(mask==v) for v in class_values])`. **[CONFIRMED:** raw
LitModel.py + retriever.py, read this session.] Those integers are the **PIL "L"-mode
luma** reductions `L=(299R+587G+114B)//1000` of the 5 colors — I verified all five
exactly (this session, `.venv/bin/python`):

```
Road  #402020 -> 41    Lane #ff0000 -> 76    MyCar #cc00ff -> 90
Undrivable #808060 -> 124   Movable #00ff66 -> 161   (pad/black -> 0)
```

So the baseline channel order [41,76,90,124,161,0] is literally
**[Road, Lane, MyCar, Undrivable, Movable, padding]**. Dropping the padding channel for
the contest's `classes=5` gives the **most-likely contest argmax index map**:

| argmax idx | class | **[CONFIRMED baseline / INFERRED contest]** |
|---|---|---|
| **0** | Road | CONFIRMED baseline ordering; INFERRED identical in contest |
| **1** | Lane markings | " |
| **2** | **My car (ego hood)** | " — NOTE: idx 2 is MY-CAR, *not* undrivable |
| **3** | Undrivable (+sky) | " |
| **4** | Movable | " |

**Caveat (NO-FAKE):** the contest SegNet is a *separately trained* model (b2, no-pretrain).
Its index order is INFERRED to match the public baseline (same author, same comma10k loader
lineage) but is NOT byte-confirmed. **$0 disambiguation (do this before relying on idx→class):**
dump the contest scorer's GT argmax over 0.mkv and label each index by spatial signature —
largest center-bottom region = Road(0); thin bright lines = Lane(1); fixed bottom-hood
crescent = MyCar(2); top/sides+sky = Undrivable(3); compact blobs = Movable(4). Our own GT
argmax is the authority; this table just predicts what we'll see.

### A.3 Does this explain our measured flip distribution?
**[MEASURED:ours]** flip-prone pixels ≈ **50% class-0 / 19% class-1 / 13% class-2**. Under
the A.2 map this reads **50% Road / 19% Lane / 13% MyCar** — and it lines up with class
geometry:
- **Road (0) = 50%** — Road is the single largest region, so it shares the longest total
  perimeter with every neighbor (lane, undrivable, hood, cars). Most boundary pixels touch
  road. **[INFERRED, consistent with A.1 area dominance.]**
- **Lane (1) = 19%** — lane markings are **thin lines ≈ all-boundary, ~0 interior** (lit.
  anchor: lane-marking classes are ~2% of pixels but ~100× boundary-to-area ratio
  [arxiv 2505.12206 / 2110.11867]). High flip share per unit area. **[INFERRED+lit.]**
- **MyCar (2) = 13%** — the ego hood is a large fixed region with a long, fairly smooth
  road/hood boundary at the frame bottom. **[INFERRED.]** (If the contest order instead put
  undrivable at idx 2, 13% would be the sky/horizon + building edges; the $0 disambiguation
  decides — but either way idx-2 is a coherent single long boundary family.)

**Crux refinement vs the DAG:** the binding residual = **union of ALL inter-class edges**
**[MEASURED:ours, DAG 2026-06-25]**; class semantics say those edges are dominated by
**Road↔{Lane,Undrivable,Hood}** with the **Lane double-edge** (each thin marking is TWO
road↔lane edges a few px apart) as the hardest long-tail (the ~8-dim lane-orbit manifold).

---

## B. SEG argmax FRAGILITY — the steganalysis lens (where d_seg flips live)

The Fridrich/Yousfi inverse-steganalysis cost model says **errors in textured/high-variance
regions are undetectable; errors at smooth/low-variance regions are detectable** (UNIWARD).
**[CONFIRMED:** local alaska deep-research + council_yousfi_voice memo §"square root law" /
"UNIWARD".] Translated to argmax-disagreement:

- **The margin is THIN (fragile → our d_seg battleground) exactly at the class boundaries**,
  and those boundaries in comma10k are **SMOOTH, geometric, low-texture** curves:
  road↔lane edges, road↔hood crescent, road/undrivable↔**sky horizon**. A piecewise-constant
  argmax target has all its entropy on a **codim-1 boundary annulus**; the softmax margin
  →0 there by construction. **[INFERRED from A + UNIWARD.]**
- **Anti-aliasing-OFF labels** ⇒ the GT boundary is a **hard 1-pixel step**, so the SegNet's
  learned boundary is a steep but finite ramp; sub-pixel RGB nudges flip argmax there
  cheaply. **[CONFIRMED label convention + INFERRED effect.]**
- **No class weighting in training** (`CrossEntropyLoss()`, no weights — **[CONFIRMED:**
  LitModel.py]) ⇒ the dominant Road class dominates the loss; **rare classes (lane markings)
  are relatively under-fit ⇒ their boundaries carry the thinnest, noisiest margins ⇒ MOST
  fragile.** This is *why* lane edges are the hard long-tail, and it is a *direct
  consequence of Yousfi's recipe*, not an accident. **[INFERRED from CONFIRMED recipe.]**
- **Robust (thick-margin, ~free to be sloppy):** region INTERIORS — mid-road, mid-sky,
  mid-hood, large building faces. The witness SDF can be coarse/cheap there.

**Implication for the directional/curvelet basis:** orient the Fourier/curvelet features to
the **boundary tangent field** of the smooth low-texture edges (lane lines, road/hood
crescent, sky horizon). This is exactly the **−48% all-class directional lever**
**[MEASURED:ours, DAG]** — Yousfi's class structure explains *why* it works: the flips live
on a small number of smooth, orientable curves, so a tangent-aligned basis matches the
target's actual support. Basis-match is PRIOR to capacity (DAG lever #1).

---

## C. seg→pose DERIVATION SUPPORT — per-class role (the live insight)

Live insight: **pose is solvable FROM the seg-partition flow + ground geometry; the reverse
(pose→seg) is not.** Mechanism: lane markings + road lie on the **ground plane**; under ego
motion their image displacement between two frames is a **ground-plane homography**
H(ego_6dof) — a clean, low-DOF, over-determined constraint. openpilot does exactly this:
online extrinsic calibration → **inverse-perspective-mapping to a ground/road frame**
(`ground_from_medmodel_frame`), lane lines fit as **polynomials in meters**, image warped to
the calibrated (pitch/yaw-normalized) frame. **[CONFIRMED:** comma.ai openpilot-in-2021
blog + openpilot calibration discussion (thomasfermi Algorithms-for-Automated-Driving);
the homography/IPM math is standard MVG — **[INFERRED]** for our specific solve.]

### Per-class seg→pose role table

| argmax idx | class | on ground plane? | seg→pose role | what it constrains |
|---|---|---|---|---|
| 0 | **Road** | YES (planar) | dense ground-plane support; large-area homography anchor | translation + pitch (texture-poor, so edges carry it) |
| 1 | **Lane markings** | YES (planar, high-contrast) | **PRIMARY** ego-motion features — sharp, oriented, sit ON the plane | forward translation + yaw (vanishing-point) — the cleanest constraint |
| 2 | **My car (ego hood)** | NO (rigid w/ camera) | **静 reference / mask-out** — fixed in image, zero parallax | calibration sanity; exclude from flow (no motion ⇒ would bias the solve) |
| 3 | **Undrivable (+sky/horizon)** | NO (far field) | **rotation reference** — horizon line + distant structure ≈ pure-rotation cues | pitch + roll + yaw (rotation), via horizon/vanishing geometry |
| 4 | **Movable (cars/peds)** | independently moving | **OUTLIERS — REJECT** | violate static-scene homography; must be RANSAC-rejected before the solve |

Reading: **the seg partition pre-segments the scene into exactly the four motion-geometry
roles a robust visual-odometry pose solve needs** — ground-plane inliers (Road+Lane), a
rotation reference (Undrivable/horizon), a zero-parallax static reference to remove (Hood),
and independent-motion outliers to reject (Movable). This is why a faithful seg witness is a
*sufficient statistic for pose* (and why pose rides the already-built stored-target sidecar
**[ours, DAG: pose SOLVED, d_pose 3.4e-5]** rather than a separate carrier): the seg flow +
ground homography already determines the 6-DOF, so the witness's binding controllable job is
**d_seg**, and pose composes at byte-close. **[INFERRED design synthesis.]**

comma10k labels confirm the classes needed for this are distinct and isolable: Lane and Road
are SEPARATE classes (so lane markings can be extracted for the homography), Movable is its
own class (so outliers are pre-flagged), MyCar is its own class (so the static hood is
maskable). **[CONFIRMED:** A.1.]

---

## D. BLIND SPOTS to exploit (where the SDF can be sloppy = free)

- **EfficientNet-B2 stride-2 stem:** the stem is a single stride-2 conv, so the first
  feature map is at **half input resolution**. Contest input is bilinear-resized to (512,384)
  → first features at **256×192**. **Detail finer than ~256×192 is invisible to the SegNet
  argmax.** **[CONFIRMED:** EfficientNet-B2 standard stem stride-2 + `upstream/modules.py`
  preprocess (512,384); framing already canon in `council_yousfi_voice_...` op-routable #4.]
- **Consequence for the witness SDF:** the partition must be **SHARP only at the boundary
  annulus at/above 256×192 effective resolution**; everywhere else (interiors, AND any
  spatial frequency below the stem's Nyquist) it can be **coarse / quantized / byte-cheap**
  with zero d_seg cost. The byte budget belongs on the **oriented boundary curves at the
  visible scale**, not on interior fidelity or sub-stem-Nyquist texture.
- **Last-frame-only:** SegNet scores ONLY `x[:,-1,...]`. **[CONFIRMED:** modules.py.] The
  witness need only get the **final frame's** argmax right for d_seg (pose uses the pair) —
  the seg-frame and pose-frame budgets are separable.
- **Chroma is in-scope:** SegNet reads RGB → argmax depends on chroma; comma10k's defining
  colors are chroma-saturated (lane=pure red, movable=green, hood=magenta), so **chroma
  carries argmax-relevant signal at the boundaries** — a genuine d_seg lever, consistent
  with the operator "Chroma too" directive. **[CONFIRMED color saturation + ours-DAG.]**

---

## SDF partition design implications (synthesis)

1. **Index map matters:** build the witness against the **[Road0, Lane1, MyCar2, Undriv3,
   Movable4]** ordering (after the $0 GT-argmax disambiguation confirms it). The level-set
   palette-anchor (per-class mean RGB) should use the comma10k canonical colors so the
   witness argmax lands on the right class. **[FEED-cv already uses per-class mean RGB —
   verify the class→color binding matches this ordering.]**
2. **Boundary-tangent (directional/curvelet) basis is the #1 lever** because Yousfi's class
   geometry concentrates ALL flips on a few smooth oriented curves (lane double-edges, road/
   hood crescent, sky horizon). Orient features to that tangent field; self-orient on the
   decoder's own argmax tangent (the −48% leg, already ON in FEED-cv).
3. **Spend bytes by class fragility, not by area:** Lane double-edges (thinnest margins, no
   class weighting in training ⇒ most under-fit) get the finest capacity; Road/sky interiors
   get the coarsest. A class-fragility-weighted bit allocation = the KKT waterfill on
   margin-saliency (DAG lever #2), now with a *semantic prior* on where margin is thin.
4. **Resolution cap is free money:** clamp SDF sharpness to the **256×192 stem Nyquist** —
   anything finer is invisible. Coarsen the interior + sub-Nyquist bands to ~0 bytes.
5. **Pose composes, not carries:** the per-class seg→pose role table shows the seg partition
   is a sufficient statistic for ego-motion; keep w_pose handled by the stored-target
   sidecar and let the witness optimize d_seg only.

## Argmax-fragility map for the directional basis (one-glance)

```
FRAGILE (thin margin, sharp SDF, most bytes)         ROBUST (thick margin, coarse SDF, ~free)
  Lane double-edges (road↔lane, ×2 per marking)        mid-Road interior
  Road↔Undrivable + sky/horizon line                   mid-Sky / mid-Undrivable faces
  Road↔MyCar hood crescent                             mid-Hood interior
  Movable↔Road edges (but Movable=pose-outlier)        anything below 256×192 stem Nyquist
  -> all are SMOOTH, ORIENTABLE curves -> tangent-aligned curvelet basis is the match
```

## Open / next (research-only; for the running descent + capstone)
- **$0:** dump contest-scorer GT argmax on 0.mkv, label idx→class by spatial signature,
  CONFIRM the A.2 ordering (turns INFERRED→CONFIRMED). Feeds palette-anchor binding + the
  per-class fragility weights.
- **$0:** measure per-class boundary-perimeter share of flips on OUR GT (not just class-of-
  flip-pixel) to confirm Road↔Lane / Road↔Hood / horizon are the binding edges.
- These sharpen the directional basis + the waterfill prior already live in the FEED-cw run.

## Sources
- github.com/commaai/comma10k (README: 5 classes, RGB, anti-alias-off, lane-marking rule)
- github.com/YassineYousfi/comma10k-baseline `LitModel.py` + `retriever.py` (CONFIRMED
  `class_values=[41,76,90,124,161,0]`, CrossEntropyLoss no-weight, Adam 1e-4/wd1e-3 cosine,
  efficientnet-b0 [README b4], two-stage 437×582→874×1164, pixel-accuracy, 0.044 val loss)
- comma.ai "Crowdsourced Segnet" blog (motion grouping)
- comma.ai "How openpilot works in 2021" + thomasfermi Algorithms-for-Automated-Driving
  (calibrated frame, IPM ground-plane, lane polynomials in meters)
- local: `.omx/research/alaska_yousfi_repo_deep_research_landed_20260530.md`,
  `council_yousfi_voice_canonical_inverse_steganalysis_review_..._20260530.md`,
  `upstream/modules.py:103-113`, DAG `sub015_DAG_...` (FEED-cv/cw, crux 2026-06-25)
