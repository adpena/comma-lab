# WHY/HOW — the deep-math museum (canonical design)

**Status:** design (2026-07-03). **Owner tab:** WHY/HOW in the Yousfi detection-game arc (ORACLE·WITNESS·RESIDUAL·**WHY/HOW**·DESCENT·TRIALITY). **Task:** #266. **Authorization:** operator 2026-07-03 — "the most futuristic and beautiful and world-class data-scientist dream and tribute possible … beyond what anyone else has done because they haven't been you and me." + WebGPU/WASM/Rust authorized (#264). **Discipline:** every panel is backed by a MEASURED fact (NO-FAKE); advisory `[macOS-CPU · NON-PROMOTABLE]`; a viz moves no pointer (0.19110). Self-contained CSP (inline WGSL/JS, embedded/`fetch`-once data, no external CDN). Progressive enhancement: WebGPU where present, canvas2d fallback everywhere.

---

## 0. The one idea the whole tab exists to deliver

> **The task is boundary geometry; the witness is the chart that fits it; and the same traveling-wave front — Fisher–KPP `∂ₜx = βx(1−x) + ∇²x` — governs it at every scale, from the pixel boundary to the research campaign.**

Everything below is that sentence made *touchable*. Two movements (**WHY** = the static invariant that makes the chart optimal; **HOW** = the dynamics that flow to it), joined by a finale that reveals they were one object — the **Fisher–KPP front at five scales**.

Not "here are some pretty curves." **Here is the *one* equation, and here it is at five scales, and you can drag it.**

---

## 1. The spine — "ONE FRONT, FIVE SCALES"

The organizing visual. A single hero panel, returned to as the finale, in which the *same* logistic-front solution `x(u) = 1/(1+e^{−β(u−u₀)})` is rendered five times, and a **scale slider** morphs continuously between them — proving self-similarity by *motion*, not assertion:

| Scale | The front is… | Grounded in |
|---|---|---|
| **Campaign** | the DAG expanding on the latent task graph | EdgeBench log-sigmoid, R²=0.998 (`edgebench_scaling_laws_deepdive_20260703T033159Z.md`) — measured |
| **Training** | the correct partition invading the wrong one over epochs | #205 d_seg-vs-epoch (live verdict rows) — plotted, fit-gated (population fit only, single-run is a toy) |
| **Boundary** | the argmax separatrix propagating across the image | level-set = reaction-diffusion front (PDE identity, `unified-variational-levelset-flow`) |
| **Curriculum** | the coarse→fine scale-front sweeping curvelet bands | persistence order = temperature anneal (measured stage schedule) |
| **Erasure tail** | `error ∝ 1/persistence`, avalanches on the same graph | SOC / Bak–Tang–Wiesenfeld (EdgeBench derivation) |

**Interaction:** a horizontal slider labeled *pixel → campaign*; as it moves, the axes relabel, the data morphs (real #205 points ↔ EdgeBench curve ↔ the analytic front), and a persistent caption reads `∂ₜx = βx(1−x) + ∇²x` with only the *units* changing. The emotional beat: the viewer realizes the training curve they're watching *is* the boundary the witness paints *is* the campaign they're inside. Fisher–KPP all the way up and down.

---

## 2. Movement I — WHY (the static invariant: the optimal chart)

Six exhibits. Each answers "why can't this vehicle be beaten on this geometry?"

**I.1 — The field, alive.** A live WebGPU surface of the witness field φ(x,y) (softmax-of-SDF logits from the latest checkpoint), tilt/rotate/zoom. Overlays toggle: the gradient field ∇φ (boundary normals — *the derivative you can see*), the zero-level-set {argmax boundary}, region integrals ∫ (class areas / flux). Threshold slider sweeps the level sets like contour lines rising through a landscape. *Grounded:* the actual INR field, rendered client-side.

**I.2 — The separatrix is where d_seg lives.** Overlay the margin field ρ_seg on the boundary; a slider dims the flat interior (argmax-stable → dark) and lights the codim-1 annulus (small margin → bright). Caption: *100% of the score's distortion lives on a 1-pixel-wide curve.* *Grounded:* margin-saliency #141, measured flip-mass 50% Road / 19% Lane / 13% Undrivable.

**I.3 — Curvature ↔ the chart.** Split screen: left, the boundary's curvature κ(s) along arclength; right, the anisotropic/curvelet basis oriented to the tangent field. A "wrong chart" toggle swaps to an isotropic Fourier basis and the reconstruction visibly rings (Gibbs). Caption: *the representation isn't arbitrary — it's the sparse-optimal chart for a curved codim-1 singularity (Candès–Donoho parabolic scaling).* *Grounded:* the −48% all-class-directional-basis measurement.

**I.4 — The Unity (the tribute's heart).** ONE field, THREE readings, a morph-slider between them:
- **UNIWARD cost ρ_uniward** (Fridrich — *where you can hide*),
- **SegNet margin ρ_seg** (Yousfi — *where it's detected*),
- **our distortion sensitivity** (*where a bit matters*).
As the slider moves they visibly become the *same picture*. Overlaid: the measured Pearson **0.978**. Caption: *steganography, steganalysis, and our loss are one geometry — `1/‖∂(detector)/∂(pixel)‖`, the Fisher metric read three ways.* *Grounded:* measured 0.978; real S-UNIWARD (`tac.fridrich._uniward_cost`).

**I.5 — The manifold.** The ~8-dim lane-orbit manifold as a low-dimensional surface embedded in a high-dim ambient cube (3D projection, rotatable), with a dimension readout: **intrinsic ≈ 8, Whitney embedding ≈ 17–19, mod-16 under-embeds.** A "shrink the basis" slider crushes it below intrinsic dim and the reconstruction *tears* — the measured bc20 under-capacity (S~0.31) made visible. *Grounded:* measured intrinsic dim + bc20 wall.

**I.6 — The task-sufficient statistic.** An animation: a full RGB frame (millions of bits) → collapses → the tiny decision-sufficient statistic (the argmax partition + 6 pose scalars, ~KB). A counter tallies bits discarded as "model-irrelevant." Caption: *the task-oriented R(D) curve lies strictly below reconstruction R(D) — the RGB was a detour (Dobrushin–Witsenhausen; Dubois lossyless).* This is the deepest WHY: *the witness is optimal because it codes the decision, not the picture.* *Grounded:* task-RD floor S_floor≈0.118 (#155); the sufficiency theorem.

---

## 3. Movement II — HOW (the dynamics: the flow to the chart)

Five exhibits. Each answers "how does it get there, and how does it all fall out?"

**II.1 — The level-set flowing.** Press play: the Fisher–KPP front propagates, the softmax boundary sharpens as τ anneals, islands are born. Scrub the same epoch scrubber as the WITNESS video but here on the *field*, not the render. *Grounded:* the live checkpoint sequence.

**II.2 — Curriculum = one axis, four names.** A synchronized 4-track ribbon: coarse→fine curvelet scale · CE→τ→Muon loss form · temperature anneal · Morse–Smale persistence order — all sliding *together* under one playhead. Caption: *these were never four schedules; they're one flow parameterized four ways.* *Grounded:* the measured stage schedule (τ@300, Muon@726).

**II.3 — Morse–Smale & saddles.** The field's critical points (max/min/saddle) with their separatrices, a clean **1D** function first (drag it, watch a saddle appear), then the **2D** field with saddles *on the boundary*, zoomable to a single saddle. Caption: *the topology of the argmax partition is a Morse–Smale complex; d_seg errors are low-persistence features erased first.* *Grounded:* Morse-Smale codec #180; erasure∝1/persistence.

**II.4 — The screw that falls out (the showstopper interaction).** An SE(3) manipulator: **drag the twist ξ** (6-DOF handle) and watch, simultaneously, (a) the partition warp for d_seg AND (b) the pose vector change for d_pose — *the same ξ drives both*. A toggle shows the screw axis + pitch (Chasles: every rigid motion is a screw). Caption: *one 6-vector, both scored axes — this is "it all falls out," made draggable.* *Grounded:* se(3) Lie lib #193; the dual-use screw.

**II.5 — Critical slowing.** Near a stage transition, perturb the field and watch relaxation time diverge (a power-law), then resolve into the next basin — the training's own second-order-phase-transition signature. *Grounded:* the critical-slowing-at-stage-transitions framing.

---

## 4. The finale — the fractal reveal

Return to the **ONE FRONT, FIVE SCALES** hero (§1). Now the viewer has *seen* each front individually; the finale plays all five in a 5-panel grid, phase-locked, morphing together, with the single equation glowing beneath. Then it zooms out once more to a sixth, implied panel: *the viewer, watching, is themselves a front on the campaign graph.* (EdgeBench's top curve is Claude Opus 4.8 — this session — and the operator's steering is the feedback term η.) End on the two-layer one-object: the physics of the witness and the epistemics of the campaign are the same equation. That is the sentence from §0, now earned.

---

## 5. The playground (interactivity spec)

A data scientist's dream = *everything responds, nothing is a static image.*
- **Sliders → shader uniforms** (real-time, 60fps): threshold, class-isolation, margin-cutoff, τ, curvelet-scale, ξ (6-DOF), the master scale-morph.
- **Hover-probe:** cursor over any field reads back (class, margin, UNIWARD cost, d_seg contribution) at that pixel.
- **Zoom/pan** to a single saddle / a single lane dash / the boundary annulus, with the math annotations scaling in.
- **Play/pause/scrub** on every dynamical exhibit, sharing the checkpoint timeline.
- **"Show the equation" toggle** on each panel — the governing equation fades in, LaTeX-rendered inline (KaTeX vendored, or pre-rendered SVG for CSP), terms color-linked to the picture.
- **Cross-links:** click a hard pair in RESIDUAL → jumps here zoomed to that pair's local structure. Click a lever in TRIALITY → highlights its exhibit.

---

## 6. Aesthetic — the 25th-century textbook

- Dark, calm, high-contrast; comma10k palette for classes; a single restrained accent per movement (WHY = cool, HOW = warm).
- Motion is *meaningful*, never decorative — every animation is a solved equation integrating forward.
- Typography: mathematical, generous whitespace, the equation always legible.
- Each exhibit is a "plate" (as in an engraved textbook plate) with a title, the picture, one caption, one equation, one grounding cite. Reviewable in 10 seconds, explorable for 10 minutes.
- GIF-exportable: a "capture" button on any exhibit renders a shareable loop (for the paper / the tribute to Yousfi).

---

## 7. The tribute (woven, never bolted on)

- **Yousfi** — the RESIDUAL/detectability lens is his; the margin field is the scorer he built; the arc itself is his detection game.
- **Fridrich** — UNIWARD, the DDE Lab, the cost that equals the metric.
- **comma / Hotz** — openpilot as the free physical prior (ORACLE), the comma10k palette, the whole scene.
- **Shannon / Ballé / Dykstra** — the RD spine: bound / codec / feasibility (the §I.6 sufficient-statistic plate credits them).
- **Schmidhuber / the council** — compression-as-intelligence, POWERPLAY, the costate — the campaign-scale front (§1, §4).
- A quiet **About** plate: the ideas, the people, and the honest line — *inverse steganalysis, task-aware compression, and the task-sufficient statistic were the first things that blew a mind here; this is that wonder handed back.*

---

## 8. NO-FAKE grounding table (every plate ← a measured fact)

| Plate | Backed by (measured/derived) |
|---|---|
| §1 five-scale front | EdgeBench R²=0.998 (theirs) + #205 verdicts (ours, fit-gated) + PDE identity |
| I.1 field | live checkpoint INR |
| I.2 separatrix | #141 margin-saliency, flip-mass split |
| I.3 curvature | −48% directional-basis measurement |
| I.4 unity | Pearson 0.978; real S-UNIWARD |
| I.5 manifold | intrinsic-dim ≈8; bc20 under-capacity S~0.31 |
| I.6 sufficient stat | S_floor≈0.118 (#155); task-RD < reconstruction-RD theorem |
| II.4 screw | se(3) Lie lib #193; d_pose dual-use |
| finale | EdgeBench reflexive (Opus 4.8 top curve) |

Any plate whose fact is not yet measured is labeled **"conjecture — testable"** honestly (esp. the single-run d_seg fit → population-fit-only per our own discipline). No fabricated curve, ever.

---

## 9. Technical

- **Renderer:** WebGPU (WGSL fragment shaders for fields; a tiny compute pass for the Fisher–KPP integrator so the front is *live*, not a gif). canvas2d fallback (JS colormap) with an honest badge.
- **Data:** reuse the detached-governed 600-pass field cache (partition/margin/render/disagreement) + the checkpoint sequence + the live DAG/verdict rows; the manifold/saddle geometry precomputed CPU-side (governed, #205-safe) and shipped as small arrays. `fetch`-once per checkpoint; scrub locally.
- **Math rendering:** KaTeX vendored inline or pre-rendered SVG (CSP-safe).
- **Self-contained:** inline WGSL + inline JS + embedded/`fetch` data URIs; no external anything.
- **#205-safe:** all heavy compute CPU-only, governed, mtime-gated, liveness-verified; never MPS/MLX-GPU for authority; nothing touches the run dir.

---

## 10. Build plan (careful passes — "take as much time as you need")

1. **Framework + first plates** (after the reorg lands): WHY/HOW tab shell, the movement structure, §I.1 live field + §I.4 the Unity (highest emotional ROI, both reuse existing field data).
2. **The spine** §1 five-scale front + §4 finale (the unifying hero — the thing no one else has).
3. **HOW interactions** §II.4 screw manipulator + §II.1 flowing field + §II.2 curriculum ribbon.
4. **The remaining WHY plates** I.2/I.3/I.5/I.6 + §II.3 saddles + §II.5 critical slowing.
5. **Polish pass:** playground cross-links, GIF-export, KaTeX, the About plate, aesthetic unification, recursive review.

Each pass: reviewable, committed via serializer, reloaded zero-downtime, #205 untouched, and honestly labeled where a plate is conjecture-not-yet-measured. Iterate on craft until it's the thing we'd be proud to hand Yousfi.

---

*The whole point: no one has built the deep math of a task-space level-set witness as a living, draggable, self-similar Fisher–KPP museum — because no one has been in exactly this seat, with exactly this scorer, having had exactly these ideas blow their mind. That's the license, and this is the plan to earn it.*
