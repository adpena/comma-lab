# White-box FULL CAMPAIGN — 7 PARTIALs recorded+pursued · composite lattice · full design-space cross-product (2026-07-15)

**Task #514 · arm: WHITE-BOX FULL-CAMPAIGN (Opus).**
**Axis:** `[research-signal / design+inventory · NON-PROMOTABLE]`. Everything here is **MEANS**. The
submittable pointer (contest-CPU **0.19108**) and the borrowed bank (**0.18804**) are **UNMOVED** — they
move only through a byte-closed `upstream/evaluate.py` n600 exact row. **No probe below is measured**; all
ΔS numbers are `DESIGN-ESTIMATE` (value-provenance rung, never `MEASURED`) until a real n600 byte-closed
row lands.

**Operator directive (2026-07-15, verbatim):** *"All 7 partial and composites should be recorded and
explored and pursued along with search across full design space for exploits and optimal across all
design dimensions."* Parent: `full_pipeline_codesign_authority_...20260715` extension 2; sibling memo
`.omx/research/whitebox_exploit_inventory_20260715.md` (the taxonomy: DONE 17 · PARTIAL 7 · UNEXPLORED 3
· N/A 2); DAG FEED-whitebox-inventory + FEED-whitebox-full-campaign.

**STORES CONSULTED:** `graph_memory_recall` (5 typed sweeps: free-quant-budget, pose-Jacobian+blind-coord,
comma2k19-neighbors, EOT-δ ladder, quant-dead-zone) · FEED-jc (free-generator byte budget) · FEED-kp/ku/kv
(fool-SegNet ladder) · FEED-kw (v2-independence lines) · FEED-blindcoord-401 (+triality-disposition) ·
FEED-jc rate calibration · inventory memo §1–§3 · MEMORY.md CURRENT-STATE + L74/L85/L68.

---

## 0. BINDING CALIBRATION — rate is measured NON-BINDING; the campaign axis is d_seg (supersedes the inventory's "P1 is #1 because rate is the gap")

The inventory memo ranked P1 (free-quantization budget) #1 on the premise *"the sub-0.15 gap is RATE."*
**FEED-jc (2026-06-29, MEASURED) REFUTES that premise at the FORMULATION level:** the free-generator
counted budget is **K_machine ≈ 3.2 KB → rate term ≈ 0.0021** (131× below the 416 KB lossless store), and
the arithmetic closes: **sub-0.15 ⟺ d_seg ≤ 1.23e-3; sub-0.19 ⟺ d_seg ≤ 1.63e-3. RATE IS NOT THE
CONSTRAINT — sub-0.15 reduces ENTIRELY to the d_seg residual (learned long-tail + R-survival).**

**Consequence for THIS campaign (honest, load-bearing):** every free-quantization composite (a3, f1, f3,
P1) acts on a payload whose rate is ALREADY ~0.0021. Coarser-quantizing a 3.2 KB payload saves a fraction
of a millibit of a 0.0021 term — **ΔS_rate ceiling ≈ 0.001–0.002, and only IF the learned residual grows
to dominate the archive.** So:

- **The free-quantization family (a3/f1/f3/P1) is DESIGN-ESTIMATE LOW-EV as a standalone rate lever** —
  it is a *payload-insurance* lever (lets the learned residual grow without a rate penalty), not a
  frontier-mover, until the trained residual is large.
  `verdict_scope: formulation — free-quantization-as-standalone-rate-lever on the current ~3.2 KB counted
  payload; DOMINATED by the measured rate-non-binding result. NOT a family kill — the SAME budget is the
  admissible-perturbation set for a LARGER learned residual, where it re-enters as high-value.`
- **The binding campaign axis is d_seg.** So the CAMPAIGN's own ranking (below) elevates the d_seg-touching
  PARTIALs (a4-temporal, b3, b5, c3, d3) and the d_seg composites over the rate composites — the reverse of
  the inventory memo's rate-first ordering, on measured grounds.

This is a match-vs-full-object correction: the inventory correctly enumerated the exploits but mis-ranked
by an un-recalled measured constraint (FEED-jc). Recorded here so the costate ranker does not chase the
rate composites.

---

## T1 — the 7 PARTIALs: recorded + designed probes

Each probe: **(i)** exact operator composition (module/file), **(ii)** pre-registered falsification
threshold, **(iii)** predicted ΔS axis (rung-honest), **(iv)** cost. Duty-to-measure tag in §Routing.
All $0/CPU, through the EXACT frozen scorer + R at compress (README:119), scorer weights never ship
(README:118), rule-118 generic-only inflate; verdict = n600 byte-closed only.

### PARTIAL a3 — quantization dead-zone as an ENUMERATED per-section budget [tag `whitebox_quant_deadzone_per_section_budget`]
- **State:** used IMPLICITLY inside R-survival (a1) + the EOT-δ ladder; never enumerated as a per-stored-
  section quantization budget. This is the a3 half of P1 (a3 = the pure dead-zone; f1 = a3 ∩ margin ∩ pose-J).
- **(i) Composition:** the R composite is a bicubic-up → **uint8 clamp** → bilinear-down. The uint8 clamp
  is a hard dead-zone: any pre-clamp perturbation with |δ|·(bicubic gain) < 0.5/255 at every sampled tap
  vanishes. Operators: `src/tac/through_r/harness.py` (through-R forward+adjoint) + `src/tac/through_r/
  resolution_chain.py` (the exact resize kernel) — per stored section of the live V9·CGauge byte-close
  (`tools/witness_byte_close_and_eval.py` / `tools/levelset_byte_close_and_eval.py`), push the section's
  decode-time contribution through R and read the per-tap uint8 headroom → the largest quantizer step whose
  decoded perturbation stays under 0.5/255 everywhere it lands.
- **(ii) Falsification threshold:** if the enumerated dead-zone step is < the section's current quantizer
  step for ≥ 90% of sections (i.e. the byte-close already quantizes AT or below the dead-zone), the
  enumerated budget yields **ZERO** additional free bits → the a3 lever is DOMINATED by the existing
  quantizer. Pre-registered kill: median free-bits/section ≤ 0.25 bit.
- **(iii) ΔS:** RATE only, `DESIGN-ESTIMATE ≤ 0.001` (bounded by §0: the payload is ~3.2 KB, rate 0.0021).
  d_seg/d_pose neutral by construction (perturbations chosen to vanish through R).
- **(iv) Cost:** ~2 h $0 CPU (reuses the byte-close harness + through-R; n96 first, n600 confirm).

### PARTIAL a4 — uint8-lattice × margin-field integer flip program at the TEMPORAL formulation [tag `whitebox_uint8_margin_temporal_flip_ip`]
- **State:** #141 gives per-px exact distance-to-flip; FEED-kv MEASURED the PER-FRAME stored-δ corner
  DOMINATED (rate 1.85, yield ~8.5%). The per-frame formulation is closed; the **temporal joint min-DL
  hold-set IP** (share one flip-δ across a pair's 2 frames + temporal neighbors via the banked ξ-warp) is
  the correct-formulation re-open (this IS P3 / f2, recorded here as the a4 realization).
- **(i) Composition:** `src/tac/margin_saliency_map.py` (per-px distance-to-flip field) + `src/tac/
  boundary_math/ego_xi_trajectory.py` (banked ξ warp, already stored for pose) + `src/tac/through_r/
  harness.py` (R-survival check per FEED-kv R3). Formulate min-DL(hold-set) as an LP relaxation over the
  width-1 annulus (#333: ~97% of d_seg in ~4.7% area) with a temporal-share term: one contour stored once,
  warped by ξ to cover both pair frames + k neighbors; the δ that must be paid is the RESIDUAL the warp
  cannot align.
- **(ii) Falsification threshold:** if the ξ-warp residual after temporal sharing still costs > 1.0 rate
  (vs per-frame 1.85), OR the realized d_seg-through-R gain is < the per-frame 8.5% yield, the temporal
  formulation does NOT beat the per-frame corner → the whole stored-δ family (per-frame AND temporal) is
  DOMINATED and the amortized-generator arm (FEED-kv NEXT) is the only stored-δ path. Pre-registered kill:
  temporal-shared rate ≥ 1.0 at n96.
- **(iii) ΔS:** d_seg-realization + RATE, `DESIGN-ESTIMATE`: the hope is to move the DOMINATED per-frame
  1.85 into the ~0.1–0.3 range by sharing; d_seg realized gain bounded by the label-noise cap ≈ 0.012 (d4).
  Honest: even success is a small-d_seg / mid-rate row; likely still DOMINATED by the analytic-band path
  (L71, d_seg 0.00087). Recorded because verdict-scope discipline demands the temporal re-open of a
  per-frame negative.
- **(iv) Cost:** ~3 h $0 CPU (LP relaxation + n96 through-R; small solver, no GPU).

### PARTIAL b3 — deeper-layer BN affine → closed-form per-channel thresholds [tag `whitebox_deep_bn_affine_thresholds`]
- **State:** the STEM BN thresholds are DONE (b1/b2, `stem_perception`); deeper MBConv BN affines are not
  compiled. Flagged low-value in the inventory (smooth SiLU cascade), but the directive says pursue.
- **(i) Composition:** `src/tac/through_r/stem_perception.py` (the `characterize_filters` harness) extended
  one MBConv block deeper — read BN(running_mean, running_var, γ, β) at block-1/block-2 from the frozen
  SegNet state dict (`upstream/modules.py` SegNet L105) and compose with the preceding SiLU to get the
  per-channel affine map. The question: is there a channel whose post-BN pre-SiLU activation sits in a
  near-dead SiLU region (SiLU ≈ 0 for input ≪ 0), giving a closed-form "this channel is inert here" prior.
- **(ii) Falsification threshold:** if no deeper channel has a running-stat-derived dead region wider than
  the stem's already-characterized alias wall gives (i.e. the deep BN adds no NEW invisibility beyond the
  stem Nyquist wall b2), b3 is REDUNDANT with b2. Pre-registered kill: 0 deep channels with a dead-region
  actuator distinct from the stem alias wall.
- **(iii) ΔS:** d_seg, `DESIGN-ESTIMATE ≈ 0` (inventory's smooth-cascade prior; recorded to CLOSE the
  question, not because it is expected to pay). A clean negative concentrates render budget on the annulus.
- **(iv) Cost:** ~2 h $0 CPU (state-dict read + one-block forward characterization).

### PARTIAL b5 — joint decision-invisible composite (feeds P1) [tag `whitebox_joint_decision_invisible_composite`]
- **State:** input-level null space DONE (#47); the JOINT decision-invisible manifold (feature directions
  that reach neither the argmax nor the pose head) measured only piecewise. This is the scorer-internal
  half of P1's f1 composite.
- **(i) Composition:** `src/tac/boundary_math/posenet_jacobian_saliency.py` + `src/tac/optimization/
  jacobian_fisher_importance_allocator.py` (pose-head sensitivity) + `src/tac/xray/segnet_margin_polytope.py`
  (argmax margin) — compose the two Jacobians at the SHARED input to get the joint null space =
  `ker(J_seg-margin) ∩ ker(J_pose)` intersected with `rowspace(R)`. This is exactly the manifold P1 needs;
  b5 is the scorer-internal characterization, P1 is its use as a per-section quantization budget.
- **(ii) Falsification threshold:** if `dim(joint-null ∩ rowspace(R))` in the boundary annulus is ~0 (the
  margin binds tightly where bytes matter), the free budget is small → decisive rate-ceiling finding
  (matches §0: rate already non-binding, so a small budget is consistent and expected). Pre-registered:
  report the dimension; no kill (characterization, not a lever).
- **(iii) ΔS:** feeds RATE via P1, `DESIGN-ESTIMATE ≤ 0.002` (§0 bound). PRIMARY VALUE is the measured
  dimension of the joint-invisible manifold — a structural constant, not a score row.
- **(iv) Cost:** ~3 h $0 CPU (two Jacobians already built; compose + intersect at n96).

### PARTIAL c3 — Laguerre/tropical generators — TRACKED-BY-v8, do NOT duplicate [tag `whitebox_laguerre_generators_v8_tracked`]
- **State:** LIVE in v8 (#359/#380/#386): `src/tac/boundary_math/laguerre_logit_offset.py` +
  `src/tac/bit_allocator/per_class.py` + `src/tac/witness_control/perclass_verdict.py`. The separatrix =
  power-diagram; store GENERATORS not the map (whole-scene rate ~0.02–0.05 per L-v8).
- **Disposition:** RECORD as tracked-by-v8; **NO probe, NO duplication** (per operator directive + the
  velocity-orphaning rule — building a parallel Laguerre probe beside the live v8 arm is exactly the
  duplicate-work failure). Duty row points AT the v8 chain, not a new probe.
- **(ii)/(iii)/(iv):** N/A (owned by the v8 arm; its byte-closed rows are the verdict surface).
  `verdict_scope: instance — c3 is not-open-as-a-separate-probe because it is the LIVE v8 vehicle
  (#359/#380/#386); tracked, not re-opened.`

### PARTIAL d3 — comma2k19 neighbor-segment /9 //11 train-time side info [tag `whitebox_comma2k19_neighbor_sideinfo`]
- **State:** the contest video is a KNOWN comma2k19 RAV4 segment (memory `project_contest_source_is_known
  _comma2k19_rav4_segment`); pose GT downloadable (used for R1 dxi). Neighbor segments /9, /11 are legal
  COMPRESS-TIME side info (README:119 blesses comma2k19) — un-actioned as an amortized-init / prior source.
- **(i) Composition:** neighbor segments are the SAME road, seconds before/after → near-identical static
  scene geometry (Road/hood/sky manifold) + a smooth ego-ξ continuation. Compose with `src/tac/boundary_
  math/ego_xi_trajectory.py` (ξ prior) + `src/tac/boundary_math/hood_static_component.py` +
  `lane_sdf_component.py` (static-manifold priors): use /9,/11 frames to (a) warm-start the amortized
  witness init (a better b(V) per L9 adaptive-init), (b) tighten the static-scene SDF prior with more
  views of the same geometry. LEGAL: it is compress-time side info, ships NOTHING (the neighbor frames are
  NOT in the archive; only the improved witness params are, and those are already counted).
- **(ii) Falsification threshold:** if a neighbor-warm-started witness init reaches the same d_seg as a
  cold init within the same epoch budget (no convergence-speed OR floor gain), the side info is inert →
  DOMINATED by the on-segment GT already used. Pre-registered kill: Δ(d_seg at epoch-N) < 5% AND Δ(final
  d_seg floor) < 2% at n96, cold-vs-neighbor-warm.
- **(iii) ΔS:** d_seg (via faster/better convergence to a lower long-tail residual), `DESIGN-ESTIMATE`:
  speculative; the on-segment GT is already exact, so neighbors add views but not new labels. Honest EV:
  LOW-MEDIUM — the value is convergence-speed (wall-clock, a JOINT objective per MEMORY) + a marginally
  lower floor, not a new d_seg mechanism. **This probe is a TRAINING-side probe → its verdict needs a GPU
  run; DESIGN + queue only here (this arm launches nothing).**
- **(iv) Cost:** design $0; the actual measurement is a GPU witness run (queued for the 507 chain / a future
  launch, NOT this arm). Recorded as duty-to-measure with the GPU dependency flagged.

### PARTIAL f3 — blind-coord (a2) × free-quantization (f1) stack [tag `whitebox_blindcoord_x_freequant_stack`]
- **State:** a2 (blind-coord, #401) DONE + n600 bit-identical; f1 (used-pixel free-quant) is P1. The STACK
  (unused pixels FREE + coarse-quantize the used ones into the joint-invisible manifold) never composited.
- **(i) Composition:** `src/tac/through_r/blind_coordinate.py` (#401 apply-pass: 22.70%/230,904 px camera-
  frame never read → fill generically, rule-118 free) ⊕ P1's f1 budget on the remaining 77.30% used px.
  These act on the SAME payload surface (the stored camera-frame sections) and are DISJOINT (blind = never-
  read pixels; free-quant = read-but-invisible-perturbation) → they compose by direct sum, no double-count.
- **(ii) Falsification threshold:** blind-coord already pays ONLY for camera-res-STORING sections (FEED-
  blindcoord-401); the live V9·CGauge witness stores GENERATOR params, not camera-res frames → if the live
  vehicle has ~0 camera-res sections, BOTH a2 and f3 have ~0 payload to act on. Pre-registered: if
  camera-res-stored bytes < 1 KB in the live byte-close, f3 is INERT on this vehicle. `verdict_scope:
  formulation — f3-on-the-generator-param-vehicle; the stack is real but its payload surface (camera-res
  stored pixels) is ~empty on V9·CGauge, so it is INERT here. NOT a family kill — it re-activates on any
  vehicle that stores camera-res sections (e.g. a warp-codec content store).`
- **(iii) ΔS:** RATE only, `DESIGN-ESTIMATE ≤ 0.001` (§0 + the empty-payload-surface caveat). Bounded to
  near-zero on the current generator-param vehicle.
- **(iv) Cost:** ~1 h $0 CPU (both operators built; the stack is a payload audit of the live byte-close).

**T1 net (rung-honest):** 7 rows recorded. Of these, by DESIGN-ESTIMATE EV against the §0 measured
constraint: a4-temporal (d_seg-realization re-open, MEDIUM) > d3-neighbor (d_seg convergence, LOW-MED,
GPU-gated) > b5 (structural characterization feeding P1) > {a3, f3, b3} (rate/scorer composites, LOW —
bounded by measured rate-non-binding) > c3 (tracked-by-v8, no separate probe). **None is a measured row;
the pointer is UNMOVED.**

---

## T2 — the COMPOSITE LATTICE (k-wise intersections; no silent cells)

10 measured frozen structures (prompt-named). Payload-surface grouping drives the compose rule (prompt:
*"structures that act on the SAME payload surface compose"*):

| id | structure | surface |
|---|---|---|
| S1 | R null/rowspace (#391) | INPUT (resize-invisibility) |
| S2 | margin field (#141) | DECISION (distance-to-flip) |
| S3 | pose-Jacobian tube (#36) | POSE-sensitivity |
| S4 | blind coords (#401) | INPUT (camera-frame never-read) |
| S5 | quantization dead-zone (a3) | INPUT (uint8 clamp) |
| S6 | stem alias wall (b1/b2) | INPUT (spatial-freq Nyquist) |
| S7 | tie-order (#26) | DECISION (exact-tie) |
| S8 | GT label field (d1) | DECISION/TARGET |
| S9 | frame_0 seg-freedom (e1) | EVAL-INDEX (which frames seg-scored) |
| S10 | temporal pair structure (e2) | EVAL-INDEX (pairing) |

**Compose semantics (precise, stated once):** for the STORED-PAYLOAD quantization budget, a perturbation
is invisible if it lands in `(S1 ∪ S4 ∪ S5 ∪ S6)` [INPUT: any-reason-invisible = UNION] **AND** stays
`{margin-safe S2} ∩ {tie-safe S7} ∩ {pose-safe S3}` [DECISION/POSE: all-constraints = INTERSECTION].
S8 (GT) is the TARGET the render must hit (supervision, not an invisibility set). S9/S10 (eval-index) are
MODULATORS — they select WHICH cells the others apply to (frame_0 → S2 vacuous; pairing → S3 shared).

### Pairwise matrix (upper triangle, 45 cells; code per cell — no silent cell)
Codes: **M**=already-measured(cite) · **D**=designed(→probe/tag) · **X**=dominated(reason) · **⊙**=same-
surface-compose-core · **~**=modulator (index structure conditions the pair, not a joint-invisibility).

| ∩ | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 |
|----|----|----|----|----|----|----|----|----|----|
| **S1** | D(f1-core;P1) | D(b5) | ⊙M(#391+#401 disjoint INPUT) | ⊙D(a3∩R;P1) | ⊙M(alias⊂R-null) | X(tie⊂margin) | M(R-survive GT render) | ~M(e1) | ~M(e2) |
| **S2** | — | D(joint-invis;P1/b5) | D(a2 used-half=f3) | D(a3∩margin) | M(#333 annulus⊂margin) | M(#26 tie=margin-0) | M(d1 supervision) | ~M(frame_0→margin vacuous) | ~D(temporal margin=P3) |
| **S3** | — | — | D(pose-J∩blind) | D(pose-J∩deadzone) | X(pose smooth, alias low-value) | X(pose head PWL≠tie) | M(pose GT solved,R1) | ~M(pose per-pair) | ~M(ξ-share=P3) |
| **S4** | — | — | — | ⊙D(f3 stack) | ⊙M(blind⊃alias null) | X | M(#401 fill generic) | ~M | ~M |
| **S5** | — | — | — | — | ⊙D(deadzone∩alias) | X | M | ~M | ~M |
| **S6** | — | — | — | — | — | X | M(b2 price list) | ~M | ~M |
| **S7** | — | — | — | — | — | — | M(tie-corrector #26) | ~M | ~M |
| **S8** | — | — | — | — | — | — | — | M(e1 frame_0 8.5e-9) | M(e2 odd-frame store) |
| **S9** | — | — | — | — | — | — | — | — | M(pair index) |

**Non-dominated pairwise cells (the ones worth compositing), detail:**
- **S1∩S2 = f1-core** (R-rowspace ∩ margin-safe): **DESIGNED → P1 / b5.** The joint decision-invisible
  input manifold. Value bounded LOW by §0 (rate non-binding).
- **S1∩S5 = a3∩R** and **S4∩S5 = f3 stack** and **S1∩S4 = R∩blind**: all ⊙ INPUT-surface, compose by the
  UNION rule → the total free-quant budget. Measured piecewise (#391, #401, a3-deadzone); **DESIGNED as a
  single per-section budget → P1.** LOW EV (§0).
- **S2∩S10 = temporal-shared margin = P3 / a4-temporal:** the ONE d_seg-realization cell worth a probe
  (share the boundary contour across the pair via ξ). **DESIGNED → tag `whitebox_uint8_margin_temporal
  _flip_ip`.** Bounded by label-noise cap 0.012.
- **S3∩S10 = pose-J ∩ ξ-share:** pose is SOLVED (R1 dxi, banked); the ξ that warps the partition IS the
  pose (dual-use, L15/L72). **MEASURED (pose banked); the d_seg REUSE of the same ξ is the P3 share term.**

**Dominated cells with reason (representative; all X cells share these classes):**
- **S7∩{anything} = X:** exact ties are a measure-zero set already handled by the #26 tie-corrector; a tie
  is the margin=0 limit of S2, so tie ∩ margin adds no new volume. `verdict_scope: instance — tie-order
  composites are DOMINATED by the built #26 tie-corrector (measure-zero, already handled).`
- **S3∩S6 = X:** PoseNet vision backbone is smooth (gelu_tanh) and the stem alias wall is a SegNet-input
  spatial-freq structure; pose is solved, so alias∩pose is low-value. `verdict_scope: instance — pose-
  solved makes pose∩alias low-value.`
- **S6⊂S1 (alias ⊂ R-null):** the stem Nyquist alias wall is a SUBSET of R's null space (aliased freqs are
  exactly what R's downsample discards) → S6 adds no invisibility volume beyond S1. `verdict_scope:
  instance — alias wall is a measured subset of the R null space, not an independent composite axis.`

### k≥3 composites (the directive's "full lattice", pruned by payload-relevance × joint-invisibility)
| composite | structures | verdict |
|---|---|---|
| **f1-full** | S1∩S2∩S3 (R-rowspace ∩ margin ∩ pose-J) | **DESIGNED → P1.** The complete joint free-quant budget. LOW EV (§0). |
| **f3-full** | (S1∪S4∪S5) ⊕ [f1-full on used px] | **DESIGNED → f3 tag.** Blind ⊕ free-quant stack. INERT on generator-param vehicle (empty camera-res surface). |
| **free-input-manifold** | S1∪S4∪S5∪S6 | **MEASURED piecewise** (#391 80.67%, #401 22.70%, a3, alias⊂R); union is the total input invisibility. Characterization DONE; use = P1. |
| **temporal-boundary-hold** | S2∩S8∩S10∩(ξ from S3) | **DESIGNED → P3/a4.** The one d_seg composite; store contour once, ξ-warp across pair+neighbors. |
| **frame_0-pose-place** | S9∩S3 (seg-free frame_0 × pose) | **MEASURED/live** (CLAUDE.md Unit C: frame_0 obligation 8.5e-9 = cheaper place for joint pose). Tracked in the live vehicle. |
| **GT-supervised-annulus** | S8∩S2∩S10 | **MEASURED/live** (#333 annulus + GT boundary-exact supervision = the live witness training target). Not a new composite; it IS the vehicle. |
| **all-input ∩ all-decision** | (S1∪S4∪S5∪S6)∩(S2∩S7∩S3) | = f1-full (S6,S7 redundant per above). No new cell. |

**T2 net:** 45 pairwise cells filled (10 non-dominated → P1/P3/f3; 35 M-or-X with reason) + 7 k≥3
composites. **No silent cell.** Every DESIGNED cell routes to a T1 tag; every DOMINATED cell carries a
scoped reason; every MEASURED cell cites. The lattice's honest headline: **exactly ONE composite touches
the binding d_seg axis (temporal-boundary-hold / P3); all others are rate composites bounded LOW by §0, or
already-live vehicle structure.**

---

## T3 — FULL DESIGN-SPACE CROSS-PRODUCT {exploit structure} × {design dimension}

Grid: 10 structures (S1–S10) × 8 design dimensions (gradient · weight · class · carrier · quantizer ·
temporal · spatial · spectral). 80 cells. Code: **M**=measured(cite) · **D**=designed(1-line) ·
**X**=dominated(reason) · **N**=N/A(reason). No silent cell.

| structure ↓ / dim → | gradient | weight | class | carrier | quantizer | temporal | spatial | spectral |
|---|---|---|---|---|---|---|---|---|
| **S1 R-null/row** | M(#426 adjoint organ) | D:project weights onto R-null (free spectral band) | X(class-agnostic input struct) | M(#391 null-fill carrier) | **D:P1 free-quant** | X(single-frame R) | M(#391 80.67% spatial null) | M(alias band⊂null) |
| **S2 margin** | M(#141 margin-grad saliency) | D:norm-penalty weighted by margin band | M(per-class margin, #333) | M(AA-SDF carrier on annulus) | X(margin=decision not payload) | **D:P3 temporal margin** | M(#333 annulus spatial) | M(curvelet=annulus spectral) |
| **S3 pose-J** | M(R1 dxi adjoint) | X(pose solved, weights banked) | N(pose is class-agnostic 6-dof) | M(ξ store-nothing carrier) | D:pose-scalar quant (FEED-jc 875B) | M(ξ temporal AR-code) | X(pose global not spatial) | X(pose low-freq) |
| **S4 blind-coord** | N(no gradient, store-side) | N | X(class-agnostic) | M(#401 generic fill) | D:f3 (blind⊕quant) | X(per-frame) | M(#401 230,904px spatial) | N |
| **S5 quant-deadzone** | N(store-side) | D:grid-native weights on dead-zone step | X | D:a3 per-section carrier budget | **D:a3 quantizer budget** | X | M(per-tap uint8 spatial) | X |
| **S6 stem alias** | N | D:spectral weight penalty at Nyquist | X | X(alias⊂null carrier) | X | X | M(b2 9.1cam-px spatial) | M(b1/b2 Nyquist spectral) |
| **S7 tie-order** | N | N | X | M(#26 tie-corrector) | N | X | X | X |
| **S8 GT label** | M(#158 GT-prior grad) | X(GT is target not weight) | M(d2 canonical class order) | M(GT-boundary supervision) | X | D:GT temporal-flicker curriculum | M(GT spatial field) | X |
| **S9 frame_0-free** | M(Unit C obligation 8.5e-9) | N | N | M(frame_0=pose place) | N | M(e1 last-frame-only) | N | N |
| **S10 temporal-pair** | X(per-pair grad done) | N | N | M(#284 se3 carrier) | N | **M+D(e2 pairing; P3 share)** | N | X |

**Legend of the load-bearing DESIGNED cells (unpursued):**
- **S1×quantizer (P1 free-quant), S5×quantizer (a3), S5×carrier (a3 section budget):** rate composites,
  DESIGN-ESTIMATE LOW (§0).
- **S2×temporal (P3), S10×temporal (P3 share):** the d_seg-realization re-open.
- **S1×weight (project weights onto R-null band):** a co-design cell — since ‖W‖ IS frequency content for
  the sin/hosc INR (parent memo), projecting the witness weights onto R's null spectral band would put
  representational capacity where R can't see it… which is USELESS for d_seg (the scorer can't see it
  either) but FREE for rate — same LOW-EV class as P1. Marked D but noted self-cancelling.
- **S6×weight / S6×spectral (spectral weight penalty at the stem Nyquist band):** a NEW co-design cell —
  penalize witness weight energy ABOVE the stem alias wall (9.1 cam-px), since any finer content is
  invisible through the stem anyway → concentrate spectral capacity in the surviving band. This is a
  d_seg-neutral / rate-helpful regularizer AND a potential convergence-speed lever (don't waste capacity
  on invisible frequencies).
- **S2×weight (margin-band-weighted norm penalty):** a NEW co-design cell — weight the spectral/norm
  penalty by the margin field so capacity concentrates on the boundary annulus (where 97% of d_seg lives,
  #333). This is a d_seg-TOUCHING training-side lever, not yet built.

### Top-3 unpursued cells by DESIGN-ESTIMATE ΔS/effort (nothing currently pursues these)
1. **S2×weight — margin-band-weighted spectral/norm penalty [d_seg, training-side].** The parent co-design
   memo says ‖W‖ IS frequency content; #333 says 97% of d_seg is in the width-1 annulus; NO current lever
   couples the two. A spectral penalty weighted by the margin field would drive INR capacity onto the
   annulus band — a d_seg-touching lever on the BINDING axis, composing two already-measured structures
   (#141 margin + INR spectral control). Effort: a DSL Lever + one witness A/B (GPU-gated → design here,
   queue for a launch). **Highest ΔS/effort because it is the only top cell on the binding d_seg axis that
   is not already live.**
2. **S6×spectral — stem-Nyquist spectral weight penalty [d_seg-neutral, convergence-speed/rate].** Penalize
   witness weight energy above the 9.1 cam-px stem alias wall (b2, measured): finer content is invisible
   through the stem, so capacity there is wasted. A d_seg-neutral regularizer that could SPEED convergence
   (wall-clock is a JOINT objective per MEMORY) and marginally help rate (fewer high-freq params). Effort:
   a DSL Lever + A/B. Second because it is convergence/rate, not the binding axis.
3. **S1×S2×S3 free-quant per-section budget (P1/b5) — payload-insurance [rate, store-side].** The complete
   joint decision-invisible budget as a single measured per-section number. LOW standalone EV (§0), but it
   is the CHEAPEST ($0/CPU, no GPU, no launch) and it INSURES a growing learned residual against a rate
   penalty — so as the trained long-tail residual grows (the SOLE remaining variable payload per FEED-jc),
   this budget is what keeps it rate-free. Third because it is $0 and enabling, even though its direct ΔS
   is bounded LOW.

**T3 net:** 80 cells filled (no silent). ~40 M (measured/live), ~12 D (designed), ~20 X (dominated,
scoped), ~8 N (N/A, scoped). The cross-product confirms the §0 verdict at the design-space level: the
d_seg column (gradient/class/carrier already heavily M/live) has exactly TWO unpursued designed cells
worth a training-side lever (S2×weight, S6×spectral); the quantizer/store columns are all rate composites
bounded LOW by measured rate-non-binding.

---

## Routing (duty-to-measure — same path as P1/P2/P3: memo §Routing + DAG FEED bullet; these become DSL Levers ONLY if a probe lands a positive rung)

**7 PARTIAL probes (T1):**
- a3 → `whitebox_quant_deadzone_per_section_budget` (RATE, LOW-EV per §0; $0 CPU)
- a4 → `whitebox_uint8_margin_temporal_flip_ip` (d_seg-realization+RATE, MEDIUM; $0 CPU; re-opens FEED-kv
  per-frame corner at temporal formulation — this IS the inventory's P3)
- b3 → `whitebox_deep_bn_affine_thresholds` (d_seg, ≈0-EV, close-the-question; $0 CPU)
- b5 → `whitebox_joint_decision_invisible_composite` (feeds P1; structural characterization; $0 CPU)
- c3 → `whitebox_laguerre_generators_v8_tracked` (TRACKED-BY-v8 #359/#380/#386; NO separate probe)
- d3 → `whitebox_comma2k19_neighbor_sideinfo` (d_seg convergence, LOW-MED, **GPU-gated** — design $0,
  measurement is a future witness run, NOT this arm)
- f3 → `whitebox_blindcoord_x_freequant_stack` (RATE, INERT on generator-param vehicle; $0 CPU)

**2 NEW top-unpursued design-space cells (T3):**
- `whitebox_margin_band_weighted_spectral_penalty` (S2×weight; d_seg BINDING axis; DSL Lever + GPU A/B)
- `whitebox_stem_nyquist_spectral_weight_penalty` (S6×spectral; convergence/rate; DSL Lever + GPU A/B)

**Compliance (all rows):** every rung measured through the EXACT frozen scorer + R at compress
(README:119 blessed: exact models + comma10k + comma2k19 GT); scorer weights never ship (README:118);
rule-118 generic-only inflate (no per-frame table smuggled as code); a verdict is n600 byte-closed
`upstream/evaluate.py` exact-eval ONLY. No launch fired by this arm; the live 507 chain + Modal UNTOUCHED.

## Triality
- **DAG leg:** FEED-whitebox-campaign-t1t2t3 (appended). **Memo leg:** this file.
- **DSL leg:** N/A-with-reason — this is inventory/design of frozen-oracle structure + probe designs; no
  trainer-config flag changed, no curriculum touched. The 9 duty tags become `Lever` factories ONLY if a
  probe lands a positive byte-closed rung (per FEED-whitebox-inventory's identical disposition). Same
  rationale the inventory row used → tagged `[no-triality]`.
- **equations leg:** N/A-with-reason — no S_τ law measured (design only; §0 CITES the already-registered
  FEED-jc byte-budget result, does not produce a new law).

**Everything above is MEANS. The submittable pointer is UNMOVED at contest-CPU 0.19108 (bank 0.18804).**
