# Static ego-hood level-set component — PRECISE + structurally CONTAINED at ~0 bytes — FEED-du (the HOOD-STATIC component the FEED-dt seal gate named)

**UTC:** 2026-06-27T07:11:50Z
**Lane:** `hood_static_component_FEED_du`
**Authority:** `[macOS-CPU advisory]` research-signal — `score_claim=false`, `promotable=false`,
`ready_for_exact_eval_dispatch=false`. $0 CPU-only. NOT a byte-closed row, NOT a trained-witness
output. Frozen CPU-torch SegNet argmax partition (cached `lstars` in `gt_n96.npz`, bit-exact per
FEED-db; NO surrogate, NO new scorer pass, NO GPU/MPS). **GPU UNTOUCHED** (levelset descent pid
72600/72602 + pose subagent a1116e516f ran concurrently; this task read ONLY `lstars` ~150 MB, n96).

Operator 2026-06-27 verbatim: *"other levers likely deserve and need this same attention."* This is the
NEXT per-lever optimal-form treatment after the VALIDATED manifold-aware lane-SDF (FEED-dm/ds,
`src/tac/boundary_math/lane_sdf_component.py`), applying the SAME isolation template (build structured
field → inject as phi_k → decompose realized argmax) to the EGO-HOOD class. It also directly ANSWERS the
load-bearing invariant the FEED-dt review+seal gate flagged for this component: *"the STATIC assumption
itself (hood IoU ~0.99 across ALL 600 or does it DRIFT on turns/bumps? a drifting 'static' mask = silent
d_seg leak)."* Additive, default-off; does NOT disrupt the critical path.

<!-- FORMALIZATION_PENDING: the canonical equation "ego-hood class = SINGLE frame-shared SDF-to-static-
majority-mask level-set field, contained by SDF local support, ~0 video-derived bytes" is queued for
tac.canonical_equations registration once a byte-closed witness row confirms the predicted d_seg benefit. -->

---

## 0. TL;DR (the decisive answer)

**YES — the static ego-hood is captured PRECISELY by a SINGLE frame-shared majority-vote mask SDF, is
structurally CONTAINED (it touches ONLY road, leak 0.000102, other-class disturbance exactly 0), costs
~0 bytes (56 B for the WHOLE mask over 600 frames = 0.093 B/frame), and R-survives.** Measured n96, $0,
frozen CPU-torch L*, injecting `phi_hood = SDF-to-(single static mask)` into the K=5 level set with the
OTHER classes' ideal per-frame SDFs (the lane-SDF isolation test the operator validated):

| variant (phi_hood =) | total d_seg | hood_fn (SHAPE) | containment leak | other-class d_seg | post-R d_seg |
|---|---|---|---|---|---|
| **ideal SDF** (baseline; argmax==L*) | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000010 |
| **majority static mask** (recommended) | **0.000737** | **0.000634** | **0.000102** | **0.000000** | **0.000677** |
| intersection static mask (max-contain) | 0.003294 | 0.003294 | 0.000000 | 0.000000 | — |

- **NEAR-STATIC: YES (answers the seal-gate invariant).** The majority mask's per-frame IoU is
  **mean 0.9944, min 0.9833** across n96 (intersection-mask IoU 0.9788). The hood occupies 0.2558 of the
  frame, rows **[282,383]** — exactly the #139 ego_hood static core (rows 284-383). It does NOT meaningfully
  drift over n96. **HONEST CAVEAT:** n96, and the comma2k19 RAV4 segment is mostly straight (FEED-dj) — turns/
  bumps are under-sampled; n600 confirmation is owed to the seal (see §5).
- **PRECISE: YES.** Majority mask gives the hood to total **0.000737** (shape FN **0.000634**) — **below the
  witness target 0.00087**. The FN is purely the slowly-moving TOP edge of the hood (the static mask cannot
  track the few-pixel jitter at the hood/road boundary); the static *core* is captured exactly.
- **CONTAINED: YES, and CLEANER than the lane.** Containment leak (non-hood→hood) **0.000102**, and
  **other-class d_seg is EXACTLY 0** — the hood region borders ONLY road, so it cannot disturb lane/top/
  movable. This is even tighter than the lane-SDF's class-0 leak 0.000193. Containment is STRUCTURAL (the
  deep ideal road SDF `phi_0` dominates the locally-supported `phi_hood` just outside the mask).
- **~0 BYTES.** The ENTIRE static mask compresses to **56 bytes** (per-row hood-top span table → brotli),
  ONE mask amortized over 600 frames = **0.093 B/frame** → rate-score contribution **0.0000373** (negligible).
  Could even be a frozen camera-mount prior (0 video-derived bytes). The EDT rasterizer is FREE (rule 118).
- **R-SURVIVES:** majority-static post-R **0.000677** (sub-target), vs ideal 0.000010 — the big solid region
  is R-robust; its only R-cost is the thin top boundary.
- **The intersection mask is the over-gating analog of the lane dash gate:** leak→0 but FN explodes
  0.000634 → 0.003294 (it discards the moving top edge entirely). **Majority is optimal — do NOT use
  intersection as default** (mirrors "continuous beats dash" in the lane SDF).

---

## 0b. NO-FAKE class-index correction (the first, mandatory finding)

The task framed the ego-hood as **class 2** per the FEED-dn canonical order
`[Road0, Lane1, MyCar2, Undrivable3, Movable4]`. **That order does NOT match the cached SegNet argmax
ordering in `gt_n96.npz`.** `identify_static_hood_class` MEASURED, per-class, on the real `lstars` (n96):

| cls | static IoU | frac of frame | bottom-25% share | majority row span | → identity |
|---|---|---|---|---|---|
| 0 | 0.330 | 0.224 | 0.003 | [175,291] | road |
| 1 | 0.000 | 0.006 | 0.001 | [189,223] | lane markings |
| 2 | 0.907 | 0.494 | **0.000** | **[0,218]** | static **TOP** (sky/undrivable-upper) — NOT a hood |
| 3 | 0.000 | 0.015 | 0.000 | [173,218] | movable/other |
| 4 | **0.963** | 0.256 | **0.975** | **[282,383]** | **the static ego-hood** ✅ |

The real static ego-hood is **class 4** (static IoU 0.963, 97.5% of its pixels in the bottom rows, majority
span [282,383] = the #139 static core). Class 2 is the LARGE static TOP region (49% of frame, rows [0,218]) —
also near-static but it is the sky/undrivable-upper, NOT a hood. Building a "hood" component on class 2 would
have been a FAKE (wrong region). **The module never hardcodes the index** — `identify_static_hood_class`
detects it from the data (`bottom_share × static_iou`), so the correction is structural, and the same code is
correct if a future cache permutes the labels. The operator's INTENT ("MyCar-hood STATIC", the #139 ego-hood
at rows 284-383) is unambiguous and is exactly class 4.

---

## 1. The design — phi_hood as ONE frame-shared SDF on the static ego-hood mask

The K=5 level-set witness (`lever_b_levelset_generator.py`,
`train_levelset_witness_realized_through_R_mlx.py`) represents the SegNet argmax partition as
`argmax_k phi_k` of K=5 SDF fields. The ego car hood is FIXED relative to the camera (fixed RAV4 mount), so
its argmax region is near-static across all frames — the SIMPLEST structured component: not a per-frame
polynomial (the lane) but a SINGLE constant mask shared by all 600 frames.

**OPTIMAL FORM (this memo):** `phi_hood = signed-distance-to-(single majority-vote static hood mask)`:

- the canonical mask = pixels that are class-4 in ≥50% of frames (majority vote over `lstars`),
- `phi_hood = +EDT inside the mask, −EDT outside` (scipy EDT — the SAME 1-Lipschitz construction as
  `signed_distance_fields` / `lane_signed_distance`; reused, not duplicated),
- the SAME `phi_hood` injected as channel `k=4` into EVERY frame's level set (no per-frame fit, no per-frame
  floats — the hood is static).

Why this is the optimal form (the three properties, MEASURED not asserted):

1. **PRECISE** — the SDF is 1-Lipschitz; the static core is captured exactly, and only the few-pixel-jitter
   top edge is FN. Measured shape FN 0.000634 < target 0.00087.
2. **CONTAINED** — the SDF is locally supported: `phi_hood > 0` ONLY inside the mask, decaying negative
   outside → just past the mask the deep ideal road SDF `phi_0` wins → road preserved by construction; the
   hood borders no other class → other-class d_seg = 0. Measured leak 0.000102.
3. **STATIC / ~0-DOF** — DOF = ONE mask (56 B) shared by 600 frames, not H·W·600 pixels and not even a
   per-frame coefficient vector. This is the rate end-state: the most byte-efficient possible structured
   component.

This generalizes FEED-dm's lane treatment to the hood and confirms the FEED-do per-lever optimal-form thesis:
the structured-manifold decomposition GIVES the simple, stable classes (lane shape, static hood) to the
witness for free, freeing capacity for the hard all-class boundary annulus (81% of flips, FEED-dj).

---

## 2. The $0 mechanism (decisive isolation test) — what was actually measured

Per frame (n96, frozen CPU-torch L* = `lstars[i]`, 384×512):
1. `hood_cls, ev = identify_static_hood_class(lstars)` → 4 (the NO-FAKE detection; §0b).
2. `maj = compute_static_hood_mask(lstars, hood_cls=4, agg="majority")` → ONE mask (computed once).
3. `phi_hood = build_static_hood_sdf(maj.mask)` → ONE SDF field (computed once, shared by all frames).
4. per frame: `phi_ideal = signed_distance_fields(L, 5)` (argmax==L exactly → containment baseline 0);
   `pred = inject_hood_sdf(phi_ideal, phi_hood, lane_cls=4, mode="replace").argmax(-1)` (substitute ONLY
   channel 4, keep the other classes ideal → isolates the hood component);
   `decompose_argmax_disagreement(pred, L, lane_cls=4, road_cls=0)` → hood FN (shape) / containment leak /
   other.

This is the operator's exact lane-SDF test transplanted to the hood: *inject it as phi_k and re-measure the
realized argmax d_seg — does it give the region precisely AND leave other classes unchanged (containment)?*
Answer in §0.

NO-FAKE: real majority vote over the real cached L*; real scipy EDT on the real mask; real argmax recomputed;
real disagreement vs the real cached L* (bit-exact). No stub, no surrogate. The hood class is DETECTED from
the data, not assumed. Scripts: `experiments/measure_hood_static_containment.py` (delegates to
`src/tac/boundary_math/hood_static_component.py`, which REUSES `lane_sdf_component`'s class-agnostic
`lane_signed_distance` / `inject_lane_sdf` / `decompose_argmax_disagreement`); JSON
`experiments/results/hood_static_containment_FEED-du/n96.json`; 17 tests in
`src/tac/boundary_math/tests/test_hood_static_component.py`.

---

## 3. Byte cost + integration spec into the witness (for the NEXT iteration)

**Byte cost (rule-118 boundary):**
- **COUNTED (archive.zip):** the SINGLE static hood mask — best **56 bytes** (per-row hood-top span table
  `[u_lo,u_hi]` per row, brotli-q11; bitmap-brotli alternative is 90 B). Amortized over 600 frames =
  **0.093 B/frame**; rate-score `25·56/37_545_489 ≈ 0.0000373` (negligible). The mask is fit from L*
  (video-derived → legitimately counted), but it is ONE mask, so per-frame cost ≈ 0. (Stronger option: the
  ego hood is a fixed camera-mount geometry → it could be a frozen-instance prior shipped in inflate.py =
  0 video-derived bytes; but the conservative NO-FAKE accounting counts the 56 B.)
- **FREE (inflate.py):** re-expand the span table → bool mask → `phi_hood` via EDT (generic deterministic
  algorithm, rule 118 / "inflate.py is a FREE interpreter").

**Integration** — additive / default-off. New flags (proposed) on
`experiments/train_levelset_witness_realized_through_R_mlx.py`: `--hood-static-component` (bool, default off),
`--hood-static-mode {bias,replace}` (default `bias`), `--hood-static-bias-scale` (float, default 1.0).
Consume `tac.boundary_math.hood_static_component`:

- **Train time:** compute `phi_hood` ONCE from `gt.lstars` (`compute_static_hood_mask` → `build_static_hood_sdf`),
  cache it (a single (H,W) field, not per-pair). In `total_loss_fn`, after `phi = model.sdf(cf, c0)`, add to the
  hood channel:
  - **`mode="bias"` (recommended start):** `phi[...,4] += scale · phi_hood` — keeps the LEARNED hood head and
    PULLS it toward the static mask (a structural prior); the witness still renders all classes.
  - **`mode="replace"` (strongest, ~0-DOF):** render class-4 DIRECTLY from `phi_hood` (FN 0.000634 < target)
    and let the witness model ONLY non-hood classes + the all-class boundary residual → DIRECTLY reallocates
    bytes/capacity off the hood.
  - No FiLM conditioning needed (unlike the lane) — `phi_hood` is frame-invariant.
- **Inflate time:** the stored 56-B span table → bool mask → EDT → `phi_hood`, injected (FREE).

**How it COMPOSES with the lane-SDF (the multi-component codec):** lane (phi_1, ~1–2 KB, per-frame poly) +
hood (phi_4, ~56 B, static) are BOTH structured class components injected into the ONE argmax. The witness then
models only the REMAINING classes — road region (class 0), the static top region (class 2, also a candidate for
this same static-mask treatment — IoU 0.907, the queued next lever), and Movable + the all-class boundary
residual. Because each component is locally supported, their containment is per-component STRUCTURAL — but the
JOINT containment (lane vs hood vs road competing in one argmax) is the highest-risk seal item flagged by
FEED-dt and must be measured on the wired codec, not inferred from the per-component isolation (§5).

---

## 4. Supersede / compose verdict

**The static-hood SDF is the OPTIMAL FORM of the ego-hood lever and should be ADOPTED (not stacked with any
per-frame hood lever).** It dominates any learned/per-frame hood representation on all three axes: precise
(0.000737 < target), structurally contained (leak 0.000102, other-class 0), and ~0 bytes (56 B static). It is
strictly simpler than the lane-SDF (no per-frame fit, no FiLM, no dash question) and strictly cheaper (56 B vs
~1–2 KB). Net plan for the capstone: GIVE the witness the lane-SDF (phi_1) AND the hood-static (phi_4) as
structured components; the witness's remaining job is the road/top regions + Movable + the all-class boundary
annulus — i.e. the actual hard part of the d_seg gate.

---

## 5. Honest caveats / NO-FAKE
- **Isolation test** (other classes IDEAL). The real witness LEARNS the other classes; the measured
  containment (leak 0.000102) assumes `phi_0` (road) is the DEEP ideal field. With a learned, shallower
  `phi_0` the static `phi_hood` could leak more — the Eikonal regularizer already in the witness
  (`--eikonal-weight 0.01`) maintains the deep-field property the containment relies on (a second
  justification, same as the lane). `mode="bias"` is the safer first integration.
- **n96, not n600** (the seal gate asked "across ALL 600 or on turns/bumps"). Measured min per-frame IoU
  0.9833 over n96 is strong, but the RAV4 segment is mostly straight (FEED-dj) → turns/bumps under-sampled.
  **Owed to the FEED-dt seal:** an n600 IoU + isolation re-measure to confirm the static assumption does not
  break on the curved/bumpy frames (cheap $0 CPU; re-run with `--n 600` once a 600-frame `lstars` cache exists,
  or extend the cache).
- **JOINT containment is NOT yet measured.** Per-component isolation says lane and hood are each contained;
  the wired multi-component codec (lane + hood + road all competing in one argmax) is the real test and the
  FEED-dt seal's highest-risk item — required before the byte-closed row.
- This is a PRIOR / structural lever at the representation level — the predicted exact-eval benefit requires a
  byte-closed trained-witness row (the next step, not this memo). **Pointer UNMOVED 0.19110.**

## 6. Observability surface
- Per-variant decomposition (`ContainmentDecomp`, reused): total / hood_fn / leak←road / leak←other /
  other_dseg — diff-able across variants (ideal / majority / intersection), decomposable per term, queryable
  from the JSON.
- Per-class detection evidence (`HoodClassEvidence`): static_iou / frac / bottom_share / maj_row_span — the
  NO-FAKE class-index audit surface (cite-able: the table in §0b is regenerated by the script).
- Static-mask diagnostics (`StaticHoodMask`): mean/min per-frame IoU, px, frac, row_span — the staticity
  signal the seal gate consumes.
- Byte cost (`hood_mask_byte_cost`): raw / row-span / brotli / amortized / rate-contribution.
- Reproducible CPU $0: `experiments/measure_hood_static_containment.py --n 96 --r-survival`; inputs:
  `gt_n96.npz` `lstars` member only. Module: `src/tac/boundary_math/hood_static_component.py` (17 tests).

## 7. Wire-in (6-hook)
1. **sensitivity-map:** the ego-hood = ~13% of d_seg flips (task framing) / 7.36% of witness-basin flips
   (#139 ever_hood); the static-mask SDF is its ~0-byte structural solver.
2. **Pareto:** hood payload 56 B (rate utterly non-binding) → frees the rate budget for the binding terms.
3. **bit-allocator:** the static mask = a new COUNTED section, ONE 56-B blob (no temporal/per-frame coding).
4. **cathedral:** the inflate.py span-table→EDT hood rasterizer = a FREE generic interpreter path.
5. **continual-learning:** this memo + DAG FEED-du + `hood_static_containment_FEED-du/n96.json`; answers the
   FEED-dt seal-gate HOOD-STATIC invariant (static IoU mean 0.9944 / min 0.9833 over n96).
6. **probe-disambiguator:** N/A (single measured verdict; majority-vs-intersection arbitrated by the
   measurement: majority wins — intersection over-gates exactly like the lane dash gate).

## 8. Borrowed-substrate accounting
- **BORROWED (cited):** scipy EDT (same primitive as `signed_distance_fields` / `lane_signed_distance`); the
  lane-SDF injection/decomposition template (FEED-dm, `lane_sdf_component`, REUSED class-agnostically); the
  #139 ego_hood static-core measurement (`ego_hood_static_dseg_20260617.json`, rows 284-383).
- **OURS-ORIGINAL:** parametrizing the SegNet argmax EGO-HOOD class as a SINGLE frame-shared signed-distance
  field to a majority-vote static mask, injected as phi_k of the softmax-of-SDF level set so the hood is given
  STRUCTURALLY at ~0 bytes (contained by SDF local support, other-class disturbance exactly 0); the data-driven
  hood-class identification (NO-FAKE correction of the FEED-dn index label: hood = class 4, not class 2).
