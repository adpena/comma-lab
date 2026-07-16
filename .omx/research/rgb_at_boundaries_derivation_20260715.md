# RGB AT BOUNDARIES — the per-class-pair chroma-NECESSITY derivation (frozen scorer, n600, $0) — 2026-07-15

`research_only=true` · **means != ends**: pointer contest-CPU **0.19108 UNMOVED** — this is a deep-math +
measured derivation feeding the c2 battery; no launch, no dispatch, the live run untouched.

**Operator directive (verbatim):** *"We might need RGB at boundaries regardless. Or some boundaries. Deep
math and geometry and frozen contest information space should reveal."* + coordinator refinements: ground in
the EXACT upstream `modules.py`/`evaluate.py`/`frame_utils.py`; compose with R; check the PoseNet chroma
path; the BLIND complement is the rate side of the same equation.

**Verdict in one line:** #508's "RGB = finisher-only at edges/annulus/long-tail" was DIRECTIONALLY RIGHT but
TOO COARSE in two ways the frozen scorer reveals: (1) chroma necessity is strongly CLASS-PAIR-STRUCTURED —
the Movable (car) edges and the Lane|Road paint edges are the chroma-decided boundaries, the Undrivable
(sky/horizon) bulk edges are luma-decided; (2) "RGB only AT the boundary" is measurably INSUFFICIENT —
SegNet's stride-2 region reading means REGION-CONSISTENT chroma (a per-class palette in the interiors)
carries most of the signal, with per-pixel chroma needed only at the sensitive-pair annulus. The minimal-RGB
answer is palette-chroma interiors (≈0 bytes, generic) + per-pixel chroma ONLY on the chroma-sensitive
annulus subset (counted, tiny).

## STORES CONSULTED (proactive recall — none of this re-derived)

- **#276 chroma-DOF probe `a3e9f0bd`** (eq `chroma_decides_lane_and_movable_at_annulus_v1`, n96): desat
  flips 7.54% Lane→Road + 4.38% Movable→Undrivable, 93.4% of chroma flips inside the margin<1 annulus;
  margin-gradient energy 78.8% luma / 21.2% chroma (aggregate). THIS memo extends it to per-class-pair ×
  n600 × three region-scoped ablations + the distance-to-flip law. The n96 aggregate is CONFIRMED, not
  re-opened.
- **#333 annulus telemetry** (L66): ~97% of d_seg lives in the ~4.7%-area margin<1 annulus — the band this
  derivation restricts to.
- **#141 margin-saliency** (∂margin/∂input on the 512×384 grid): the object whose BT.601-chroma-plane
  projection IS the RGB-at-boundary map; this memo computes exactly that projection per class-pair.
- **#391 / FEED-scorerfactor** (`frozen_scorer_exact_factorization_20260715.md`): SegNet and PoseNet share
  ONE bilinear resize A to (512,384) (modules.py:109 == :73); §8 blind set B1/B2/B4/B5.
- **#401 blind-coordinate** (22.70% camera-frame zero-weight), **#508 doctrine memo**
  (`rgb_only_finisher_edges_annulus_longtail_cargocult_sweep_20260715`), **chroma rung**
  (`chroma_rung_design_20260710.md`, LEVER-4c `SegChromaBoundary` DSL lever already built),
  **ADVISORY_evaluator_video_geometry** (Pose Jacobian energy 95.97% luma / 4.03% chroma).

## 1. The exact frozen forward (no idealized SegNet)

- `modules.py:103-113`: SegNet = smp.Unet('tu-efficientnet_b2', classes=5), consumes **raw 0–255 RGB, no
  normalization**; `preprocess_input` = last frame only (L108) → ONE bilinear resize A to (512,384) (L109);
  d_seg = per-pixel **argmax disagreement** (L112). The argmax flips exactly where the top1−top2 margin
  m(x) crosses 0 ⇒ the sensitivity object is the margin Jacobian g(p) = ∂m/∂x(p) ∈ ℝ³ per pixel of the
  (384,512) grid — the #141 map.
- `frame_utils.py:60-63` fixes the pipeline's ONLY color decomposition — BT.601:
  `Y = k·rgb, k = (0.299, 0.587, 0.114)`; `U ∝ B−Y`, `V ∝ R−Y` (exactly linear pre-clamp).
- **The exact color factorization (per pixel):** `ker(U,V) = span{(1,1,1)}` (the achromatic line — the only
  direction invisible to both chroma channels) and `ker(Y) = {δ : k·δ = 0}` (the Y-preserving 2-plane that
  the U,V coordinates parametrize). `ℝ³ = span{1} ⊕ ker(Y)` — a direct (oblique) sum since k·(1,1,1)=1.
  A luma-faithful-but-chroma-wrong witness errs exactly inside `ker(Y)`.
- **Chroma necessity of a boundary pixel** = the frozen margin's response to Y-preserving error:
  `S_ch(p) = max_{δ∈ker Y, ‖δ‖=1} g(p)·δ = ‖g − (g·k̂)k̂‖` and `S_lu(p) = |g·k̂|`, an ORTHOGONAL split
  (`S_ch² + S_lu² = ‖g‖²`). Linearized **distance-to-flip along the chroma plane** = `m(p)/S_ch(p)` in
  0–255 RGB units (and `m/S_lu` for luma). A boundary is *chroma-decided* where the chroma flip distance is
  smaller/comparable AND within the scene's actual chroma contrast.
- EfficientNet-B2's stride-2 stem reads REGIONS: the flip decision integrates chroma over the receptive
  field, so boundary-LOCAL chroma and CONTEXT chroma must be separated by construction — hence the three
  region-scoped exact ablations below (not gradient-only).

## 2. What was measured (all through the real frozen CPU-torch scorers, fp32, n600 real-gt)

Tool: `tools/rgb_at_boundaries_chroma_jacobian_n600.py`; rows
`experiments/results/rgb_at_boundaries_chroma_jacobian_20260715/rows.jsonl`; per-frame NO-FAKE sanity: the
baseline forward reproduces the cached exact L* (0 mismatched px, frame 0 gate).

Per pair (frame1), per class-pair boundary (a,b) of the exact L*:
- **A1 exact desat ablations** (ONE batched SegNet forward, 3 variants): `desat_full` (all chroma → BT.601
  Y replicated), `desat_annulus` (chroma removed ONLY in the margin<1 annulus = the lever's failure mode),
  `keep_annulus` (chroma ONLY in the annulus = the naive "RGB only at boundaries" sufficiency test). Flips
  vs the exact L*, attributed to the nearest class-pair boundary.
- **A2 chroma-Jacobian split** (1 fwd + 1 bwd per frame): g = ∂(Σ boundary margins)/∂(SegNet input),
  split per §1; energy aggregated over the ≤8px near-boundary band per pair; distance-to-flip percentiles
  at the boundary pixels (aggregation caveat: g at a pixel sums nearby same-edge margin terms — the
  chroma/luma RATIO is robust because same-edge pixels share sensitivity direction).
- **B scene geometry**: across-edge Δrgb at the (384,512) grid, chroma/luma split — which boundaries are
  chroma-DEFINED in the frozen video.
- **C PoseNet constraint** (modules.py L73-74 rgb_to_yuv6 path, every frame): d_pose of (f0, f1_variant)
  vs the cached exact GT pose for `desat_annulus` and `keep_annulus`.
- **R-composition**: R = bilinear↓ ∘ uint8 ∘ bicubic↑ is channel-diagonal with the SAME spatial kernel per
  channel ⇒ commutes with every fixed color-space map ⇒ the split of §1 is preserved by R exactly, up to
  the uint8 floor. MEASURED floor (`r_survival_check.json`, 8 frames × ±both signs): annulus-band chroma
  transfer gain through the REAL R = **0.98–1.00 at 0.5/1/2/4 LSB** — even sub-LSB boundary chroma
  survives (the band spans many camera pixels; rounding dithers out). **R is NOT a barrier for boundary
  chroma.**

## 3. RESULTS — n600, MEASURED through the frozen CPU-torch scorers

### 3.1 Global chroma worth (d_seg-equivalent = argmax disagreement vs the exact L*, mean over 600 pairs)

| ablation | d_seg-equiv | reading |
|---|---|---|
| `desat_full` (no chroma anywhere) | **0.005384** | total chroma worth = **6.2× the whole d_seg need** (0.00087) — chroma is TRUNK-level, not garnish |
| `desat_annulus` (no chroma in the margin<1 band) | **0.002972** | annulus-local chroma alone decides ~0.003 d_seg |
| `keep_annulus` (chroma ONLY in the band, grey context) | **0.006293** | **WORSE than removing all chroma** — the naive "RGB only at boundaries" is REFUTED |
| annulus area fraction | 0.0267 | the seen band (B2 dual: ~97% of frame is d_seg-null bulk) |

The keep_annulus > desat_full inversion is the structural finding: EfficientNet-B2's stride-2 stem reads
REGIONS, so REGION-CONSISTENT chroma (the class palette in the interiors) is load-bearing; a chroma
discontinuity at the band edge hurts more than uniform greyness. Chroma correctness at the annulus and
chroma CONSISTENCY over the region are jointly necessary — strongly non-additive
(0.00297 + 0.00629 vs 0.00538).

### 3.2 Per-class-pair boundary table (n600; flip = argmax change at the pair's boundary pixels; dtf = linearized distance-to-flip in 0–255 RGB units; edge contrasts = scene across-edge Δrgb split)

| pair | boundary px (n600) | flip desat_full | flip desat_ann | flip keep_ann | grad chroma-frac | frac chroma-stronger | dtf_ch med | dtf_lu med | frac dtf_ch≤8 | edge chroma | edge luma | chroma-NECESSARY? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Undrivable\|Movable | 146,089 | **0.363** | 0.280 | 0.376 | 0.091 | 0.273 | 55 | 29 | 0.090 | **20.7** | 22.8 | **YES — strongest** (car-vs-background; scene chroma contrast ≈ luma) |
| Road\|Movable | 138,664 | **0.303** | 0.204 | 0.275 | 0.081 | 0.215 | 57 | 24 | 0.086 | 9.4 | 10.0 | **YES** (car-vs-road) |
| Road\|Undrivable | 507,461 | 0.242 | 0.207 | 0.274 | 0.090 | 0.191 | 51 | 20 | 0.094 | 4.5 | 4.4 | **YES** (horizon/roadside; biggest absolute flip mass: 188k/126k flips) |
| Road\|MyCar | 605,253 | 0.220 | 0.100 | 0.243 | 0.077 | 0.168 | 71 | 24 | 0.063 | 4.4 | 5.4 | YES but half CONTEXT-driven (annulus-scoped flips drop to 0.100; scene chroma contrast tiny — flips come from regional chroma, the static hood #139 core) |
| Road\|Lane | 1,142,616 | 0.082 | 0.056 | 0.104 | 0.072 | 0.250 | 63 | 30 | 0.081 | **12.3** | 20.9 | YES-moderate (lowest per-px rate BUT largest boundary mass → 106k absolute flips; lane paint has real chroma contrast; 25% of its boundary px are chroma-dominant) |
| Lane\|Undrivable | 885 | 0.292 | 0.319 | 0.411 | 0.094 | **0.330** | 28 | 20 | **0.181** | 8.0 | 12.8 | YES (rare; highest chroma-dominance + shortest chroma dtf) |
| Lane\|MyCar | 5,399 | 0.089 | 0.029 | 0.086 | 0.096 | 0.257 | 110 | 53 | 0.037 | 13.4 | 24.2 | weak-NO at the annulus (large margins, dtf ~110) |
| Movable\|MyCar / Lane\|Movable | <200 | 0.32/0.32 | — | — | — | — | — | — | — | — | — | support too thin (≤5 frames) — no verdict |

Flip-mass ranking (absolute flips near the pair boundary, desat_full → desat_annulus): Road|Undriv
187,760→125,986 · Road|MyCar 152,209→61,981 · Road|Lane 105,635→67,123 · Undriv|Movable 103,130→56,635 ·
Road|Movable 68,210→36,087.

### 3.3 The sensitivity law (the #141 map's chroma-plane projection, n600)

- Chroma gradient-energy fraction is **7–12% per pair** in the ≤8px boundary band (aggregate confirms
  #276's whole-image 21.2% order) — luma is the DOMINANT axis on average, BUT:
- **17–33% of boundary pixels are chroma-DOMINANT** (S_ch > S_lu) — chroma is the DECIDING axis at a
  fifth-to-third of boundary pixels, pair-structured (Lane|Undriv 33%, Undriv|Movable 27%, Lane 25%,
  MyCar 17%).
- Median linearized chroma distance-to-flip is 50–70 (0–255 units) vs luma 20–30; **6–9% of boundary
  pixels flip within an 8-LSB chroma move, ~2% within 2 LSB** — the knife-edge chroma set.
- R-composition: measured chroma transfer gain through the REAL R = **0.98–1.00 at 0.5/1/2/4 LSB** — the
  split commutes with R (channel-diagonal kernels) and the uint8 floor dithers out over the band. Every
  chroma-flippable pixel above is REACHABLE through the byte-closed decode.

### 3.4 PoseNet constraint (n600, modules.py L73-74 rgb_to_yuv6 path)

- Wrong (removed) chroma at the annulus of f1: Δd_pose mean **4.26e-4** (median ~1e-4, heavy-tailed) —
  NOT free vs the banked 1.61e-3 (+26% if additive) — BUT this is the risk of WRONG boundary chroma;
  the lever carries chroma TOWARD GT, so its pose effect has the opposite sign (moves the pair toward
  the GT pair PoseNet scored).
- Grey context / chroma-only-at-annulus: Δd_pose **9.57e-3** — regional chroma is strongly pose-visible
  (PoseNet's 2×2 box-averaged U/V see region means). B4's "fine chroma is pose-null" holds only BELOW
  2px@(512,384); annulus-band-scale chroma is partially pose-visible. Pose-safety of a boundary-chroma
  carrier therefore comes from CORRECTNESS (match GT), not from invisibility; the strict pose-null
  projection (zero-sum chroma per 2×2 block) remains available if a counted carrier ever needs it.

## 4. The refined #508 scope + the RGB-AT-BOUNDARY lever

### 4.1 Was #508 "RGB = finisher-only at edges/annulus/long-tail" too coarse? YES — two corrections

1. **Chroma is TRUNK-level at the REGION scale.** Removing regional chroma costs 0.0054–0.0063 d_seg
   (6–7× the entire d_seg need). The witness's per-class near-constant palette is not a rendering habit
   to tolerate — it is a LOAD-BEARING generic prior (0 bytes, rule-118-free: 5 classes × 3 numbers).
   "RGB as finisher-only" is correct ONLY about *per-pixel* RGB; *per-class palette* chroma must be in
   the trunk everywhere. (`verdict_scope: formulation` — the #508 slogan read as "no chroma outside the
   annulus"; the doctrine's own clause (d) "another formulation PROVEN optimal by measurement" is this
   memo.)
2. **"Some boundaries" is exactly right and now has the map.** Chroma necessity is strongly
   pair-structured: Movable (car) edges are the most chroma-decided (0.30–0.36 flip under desat),
   then horizon (0.24), hood (0.22, half context-driven), lane paint (0.08 per-px but the largest
   absolute mass). NO major boundary is chroma-free — the weakest verdict on real support is
   Lane|MyCar (dtf ~110, flip 0.09).

### 4.2 The lever (c2 battery) — minimal RGB, dual-side derived

**SEEN side (distortion):** carry per-pixel chroma ONLY where the frozen Jacobian pays — the annulus ×
chroma-plane of the sensitive pairs — ON TOP of the region-consistent class palette (never instead of it;
§3.1 inversion). **BLIND side (rate):** everywhere else chroma is palette/generic (B2 interior ~95% of
frame d_seg-null; B1 camera-res kernel filled free #401; B4 chroma-HF pose-null; B5 only the argmax sign
needs to survive).

The rate ladder (fire in order):
- **Rung 0 — 0 bytes (FIREABLE NOW):** `SegChromaBoundary` (LEVER-4c, DSL-built, NEVER-FIRED,
  duty-to-measure `seg_chroma_boundary_276`) — train-time chroma-match on the margin<1 annulus; ships
  nothing. THIS table is its aim + ceiling: recovering annulus-chroma flips is worth UP TO ~0.0030 d_seg
  (the desat_annulus worth; the realized gain is the palette→GT fraction of it — S5-N10 worth≠gain
  discipline). The measured pair structure predicts WHERE its wins land (Movable edges + horizon +
  lane); the pre-registered A/B in `chroma_rung_design_20260710.md` is unchanged and now has a
  per-pair expected-signature to check.
- **Rung 1 — ~8 KB counted:** per-frame per-class-pair chroma palette-DELTA along the boundary band
  (9 pairs × 2 chroma DOF × ~6 bit ≈ 14 B/frame ≈ 8.4 KB / 0.0056 rate) — captures the mean boundary
  chroma shift the constant palette misses. Worth candidate only if rung 0's A/B shows a residual
  mean-shift signature.
- **Rung 2 — sparse per-pixel chroma sidecar (likely DOMINATED, do NOT build first):** even scoped to
  the chroma-knife-edge set (dtf_ch≤8: 6–9% of boundary px ≈ 300–500 px/frame), ~180–300 KB ≈ 0.12–0.2
  rate buys ≤0.3 d_seg upper-bound — marginal at the bound, dominated after the trained head takes its
  share. Reformulation queue, fire only on a measured rung-0/1 residual.
- **Pair-scoped weights extension** (per-pair `w_k` on the chroma match, aimed by this table) = a DSL
  lever + trainer wiring change — OWED, not half-wired here (this arm is no-trainer-edit; SPEC §8 A).

### 4.3 Owed / reformulation queue

1. Fire rung 0 (`SegChromaBoundary` ON/OFF warm-start A/B per the pre-registered plan) at the next
   machine window — the ONLY step that converts this derivation into a d_seg row.
2. Pair-scoped chroma-match weights (`w_k` per class-pair) as a DSL lever + trainer wiring (owed).
3. If rung 0 washes: engage chroma EARLIER in the curriculum (before the palette hardens) — the
   §3.1 region-consistency finding says the palette forms first and per-pixel chroma refines it.
4. Counted rungs 1–2 only on measured residual signatures.

## 5. Artifacts + honesty

- Tool: `tools/rgb_at_boundaries_chroma_jacobian_n600.py` (resumable JSONL; NO-FAKE gate: baseline forward
  reproduces the cached exact L* px-exact before any row).
- Rows/summary: `experiments/results/rgb_at_boundaries_chroma_jacobian_20260715/{rows.jsonl,summary.json,r_survival_check.json}`
  (600/600 pairs; d_pose n=600).
- Measurement axis: **[macOS-CPU advisory]** through the frozen CPU-torch scorers on the exact cached GT
  (the same authority the trainer verdict uses); NOT a score claim; the ablation worths are d_seg-EQUIVALENT
  ablation costs, not achieved moves. Equation leg: `rgb_chroma_necessity_per_boundary_pair_v1`
  (`src/tac/canonical_equations/rgb_at_boundaries_20260715.py`).
- **Pointer 0.19108 UNMOVED** — this is MEANS (a derivation + aim for a never-fired 0-byte lever), not a row.
