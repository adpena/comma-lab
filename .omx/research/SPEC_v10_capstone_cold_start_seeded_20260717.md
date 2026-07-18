# SPEC_v10 — THE CAPSTONE: cold start on a fully seeded, Kolmogorov-optimal program (2026-07-17)

**Charter (operator verbatim, naming SoT `.omx/research/vehicle_naming_resolution_v10_capstone_20260717.md`):**
*"We should advance to v10 as our capstone frontier and clear up naming resolutions and be very clear that
after the current run and any outstanding A/B we are doing cold start on a fully seeded Kolmogorov optimal
program."* Headline principle (operator verbatim, same directive family): *"Appearance and texture phase only
for that which is absolutely necessary according to deep math and geometry and flattening and expansion and
reformulation and solving and factorization and differentiation and integration and such of upstream
modules.py."*

**Task:** #521 · P0 ledger `p0_v10_capstone_cold_start_seeded_20260717` · branch
`claude/p0_521_spec_v10_capstone_20260717`.
**Subordination:** NO-FAKE supreme rule > THE GOAL (sub-0.15) > this SPEC. SPEC_v75 §8 OPERATING CONTRACT
binds verbatim (resumability P0 · already-settled table · no-stray rules · execution guardrails · cathedral
invariant). **Pointer honesty: contest-CPU pointer 0.19108 UNMOVED — everything in this SPEC is MEANS**; the
only success definition is a lower byte-closed `upstream/evaluate.py` n600 exact row. Bank 0.18804 (PR128
splice) is NON-SUBMISSION borrowed substrate, never a v10 input.
**Axis honesty:** every number below is tagged with its source; d_seg numbers are frozen CPU-torch fp32
SegNet through the exact contest R on bit-exact cached GT (`gt_n600.npz`) unless noted —
`[macOS-CPU advisory]`, `score_claim=false`, `promotable=false`.

---

## §0 Vehicle identity

Per the canonical version ledger (naming SoT): **v10 = fresh-init vehicle, projection-native from birth.**
Run configs are tagged `v10cN` (never bare cN). v10 is NOT a warm start: v9c2's measured results become
LAWS and SEEDS, never weights. Sequencing (operator-binding): v9c2 completion → outstanding A/Bs
(p0_497 curvelet matched-bytes; #518 8-vs-27 warm-up short arm) → v10 launch (operator GO).

## §1 The operating frame — the quadrilateral

v10 is the first vehicle DESIGNED under all four legs at once (memories
`train_least_surgical_kolmogorov_projection_realization_doctrine_20260716` +
`completeness_fourth_leg_kolmogorov_projection_realization_20260716` +
`.omx/research/projection_unification_and_eight_lenses_20260715.md`):

- **Kolmogorov (what to count):** witness = fixed point `w = G(w; seed)`; `G` (the projection composition)
  compiles FREE into inflate.py (rule 118); rate = `|program| + Σ|seed|`, `|program|` free. Standing test
  on every carrier: *"is this the fixed point of a shorter program?"* K/H = 0.47 MEASURED (necessity
  solver) — half the "data" was geometry not yet written as a generator.
- **Projection (how to produce):** geometry is solved — rank-4 head (`segnet_head_rank4_linear_flipdist_v1`),
  Laguerre/Morse-Smale partition (#284), Fisher metric (#500, margin↔Fisher Pearson 0.978), exact scorer
  factorization — so produce by Dykstra-projection / closed-form solve / seeded generator; train only the
  irreducible.
- **Realization (where difficulty lives):** d_seg is realization-limited, not gradient-limited (closed-form
  flip distances are sub-LSB at saddles); the question is which few DOF realize the target.
- **Completeness (whether G converges to the right point):** every force the MEASURED dynamics demand,
  none the dynamics forbid — the per-config COMPLETENESS TABLE (§5) certifies the fixed-point equation.

**Train-least at full force:** every training stage in any compiled v10 config MUST carry a per-stage
Kolmogorov justification (*"no solve/seed/projection can produce this"*). A stage without it is cargo-cult
training budget and the DSL factory refuses it.

## §2 The 8 pillars (enumeration source: naming SoT §"The v10 commitment"; each realized with EXISTING machinery)

### P1 — Seeded static classes (structured init as the default)
- **WHAT:** hood (MyCar) and sky/horizon (Undrivable) born from the measured GT masks; lane born from
  openpilot polynomial priors + per-dash anchors. No epoch is spent learning what a mask already states.
- **MEASURED BASIS:** hood/sky static IoU 0.993/0.976 [MEASURED, naming SoT P1 ← mask-stability analysis];
  hood static core IoU 0.994 n96 (CLAUDE.md scorer section); static hood-tex seed cuts d_seg
  0.04538 → 0.01328 at ε=0 for 1,759 counted bytes (min-S 1.613 knee)
  [MEASURED, `necessity_dseg_calibration_20260715.md`]; dash comb REFUTED — spacing CV 0.41–0.78 ⇒
  per-dash anchors are the sufficient statistic [MEASURED, `hard_frame_mechanism_atlas_20260716.md`].
- **MACHINERY (in-tree):** `src/tac/boundary_math/hood_static_component.py` +
  `lane_sdf_component.py` (class-SELF-DETECTING — never hardcode class index);
  openpilot polynomial + homography rasterizer discipline (CLAUDE.md rate-lever table);
  the necessity calibration hood-tex seed builder (`tools/necessity_inverse_factorization_solver.py`
  family + `necessity_dseg_calibration` artifacts); illumination-cone gate `g(x)` (P-seed, §4).
- **STATUS:** machinery DONE (hood/lane components, hood-tex seed measured); the *structured-init-as-default*
  wiring into the trainer init path is the v10 build item (Lever factory, §9).

### P2 — Head born SOLVED + gauge-fixed
- **WHAT:** the rank-4 output head is solved in closed form at init (not descended into existence) and
  sum-zero gauge-canonicalized from birth.
- **MEASURED BASIS:** frozen SegNet head is EXACT rank-4 linear; flip distance d = |m|/‖Δw‖ closed form
  [MEASURED, `segnet_recursive_fractal_factorization_20260715.md`, eq `segnet_head_rank4_linear_flipdist_v1`].
  Gauge (out_sdf class-mean) = 52.4% of head norm but RATE-NEUTRAL under dense int8 (shape-priced coder);
  it is a PRECISION lever: canonicalizing pre-quant gives a 22.3% finer int8 scale at identical bytes and
  3.3× lower palette dequant error [MEASURED, `.omx/research/null_subspace_rate_measure_20260717.md` (#519)].
- **MACHINERY:** `ForkHeadSolve` (#518 — **post-merge dependency**, see §9); `HeadOffsetSolver` (wired
  advisory arbiter in `curriculum_dsl`); #341 full-P GN head solve (solve-don't-train inventory #342);
  gauge canonicalizer per #519 (pre-quant projection), n600 confirm owed in **#406**.
- **STATUS:** solve machinery exists (HeadOffsetSolver advisory; #341 build owed per c2 memo);
  `ForkHeadSolve` is on the #518 branch — this SPEC marks it POST-MERGE, the DSL factory fail-closes on it.

### P3 — range(A)-restricted render targets (#520)
- **WHAT:** render in the scorer's sigma-algebra — spend zero capacity on the measured scorer-invisible
  complement of the shared resize `A` (modules.py:109 ≡ :73).
- **MEASURED BASIS:** ker(A) blind rows: 106/874 × 140/1164 → 230,904 px = 22.6969% of camera rows exactly
  zero under Aᵀ; witness-diff energy in ker(A) ≈ 52.4% raw / 52.9% mean-removed — over HALF of rendered
  output energy is scorer-invisible; ~50–53% of marginal output-layer weight effect likewise
  [MEASURED, `null_subspace_rate_measure_20260717.md` + `blind_coordinate.py` #391/#401].
- **MACHINERY:** `src/tac/through_r/blind_coordinate.py` (exact resize-kernel machinery #391/#401);
  `resize_null_preimage_compiler` (#49); the exact A factorization
  (`frozen_scorer_exact_factorization_20260715.md` §2/§6.1).
- **STATUS:** measurement DONE; the range(A)-restricted render-target projection layer is task **#520**
  (build owed — named launch-gate input, not aspiration: the DSL factory carries it as a blocker until
  the projection layer exists and is A/B'd at $0).
- **COMPOSITION (amendment 2026-07-17):** range(A)-restriction and the cell-generator description (§3.0)
  COMPOSE — both are projections onto the scorer's σ-algebra, of which the Laguerre cell complex is the
  atom structure: ker(A) removes what the scorer cannot RESOLVE, the generator description removes what
  the argmax cannot DISTINGUISH within an atom, so the render target is "cells in range(A)", not pixels.

### P4 — Content-priced coder (fix the #519 Kolmogorov violation)
- **WHAT:** the archive coder prices CONTENT, not SHAPE — dense int8 pricing (shape-priced) is the measured
  Kolmogorov violation: a 52%-of-norm gauge component costs nothing to remove and nothing to keep, while
  score-relevant deviation pays 25% higher int8 error under the uncanonicalized scale.
- **MEASURED BASIS:** #519 table — gauge projection rate-neutral (+11 B), Δd_seg −4.8e-6; scale
  0.028416 → 0.022088 (22.3% finer) [MEASURED, `null_subspace_rate_measure_20260717.md`].
- **MACHINERY:** #336 sensitivity bit-allocator hooks; #461 cross-tensor structure; entropy coding stack
  (brotli-cascade / raw-LZMA / range-coder primitives, canonical-equations L21–L32 lineage); the
  byte-close tool `tools/levelset_byte_close_and_eval.py` as the measurement surface.
- **STATUS:** primitives exist; the composed content-priced coder for v10's weight sections is a build
  item measured ONLY via byte-closed archive.zip stat (never asserted).

### P5 — Boundary laws ON from birth (the #518 set as defaults, not levers)
- **WHAT:** the resume/fork boundary laws become birth defaults: beta2-derived LR warm-up
  (`ResumeLRWarmup`), engage-boundary registration, w_pose ramp (`PoseEngageWPoseRamp`), EMA clearance
  (`ForkEmaClearance`), margin step cap (`MarginStepCap`), state persistence.
- **BASIS:** warm-start law + stepped-epoch sec/ep + admission-REFUSE physics
  [memory `warm_start_derived_schedule_provenance_and_admission_physics_20260716`]; the #518 8-vs-27
  warm-up A/B (short arm) decides the warm-up-length law [OPEN — OQ-3].
- **MACHINERY:** all five named levers are **#518-branch post-merge dependencies** — referenced here as
  such, never fake-resolved (§9 fail-closes). Existing: `stage-transition-rewarmup-*` trainer flags
  (already DSL-held) are the ancestor form.
- **STATUS:** POST-MERGE. On merge, each becomes a default-ON birth law with a LawRef; until then the DSL
  factory refuses to compile a launchable config.

### P6 — Store-nothing pose restored (#314) + joint descent
- **WHAT:** pose rides the store-nothing arm (#314 operator decision, fresh-arm boundary) with JOINT
  descent — the ONLY mechanism that crosses the photometric wall.
- **MEASURED BASIS:** post-hoc/stored pose is DEAD on the witness (5 formulations, verdict_scope:
  formulation — photometric wall) [MEASURED, L68 + CLAUDE.md §Pose CLARIFICATION 2026-07-10]; live banked
  result R1: d_pose 0.001610 through byte-close, contribution 0.127, 7.2 KB dxi section (fallback bank);
  frame_0 is structurally seg-free (d_seg obligation 8.5e-9) — the cheap PLACE for joint-trained pose
  output [MEASURED, Unit C / frozen factorization §6.2]; PoseNet chroma is 2×2-box-averaged → fine
  boundary-RGB is pose-safe by construction [DERIVED-from-source, frozen factorization §5/§6.5].
- **MACHINERY:** `PoseFinishConditioningGate` + `PoseBlindComputeGate` (wired, c2); store-nothing JSONL
  arm (#314 lineage); banked R1 dxi as fail-safe section.
- **STATUS:** machinery DONE; the v9c2 pose-window OUTCOME (OQ-1) decides the v10 pose stage's form
  (joint-finish engage law + whether the dxi fallback ships).

### P7 — Per-class carriers as built (v8 increment-1 kit #386) + basis-cure per p0_497
- **WHAT:** the archive carriers are the v8 per-stratum kit: cell → `decoupled_field` /
  `road_undriv_bulk_field`; edge → `curve_relative_offset_coder` δ(s) (→ openpilot-poly + dash-phase
  generator swap — 61% of the edge H-floor is Road-Lane, the generator swap is the measured cheaper
  program); saddle → per-saddle precision annotation (NO byte section — 29.2% of saddle flips are
  sub-LSB ⇒ the currency is precision, not bytes) [MEASURED,
  `necessity_solver_inverse_factorization_20260715.md`].
- **BASIS-CURE:** trunk basis decided by the p0_497 curvelet matched-bytes A/B (OQ-2); the no-more-Fourier
  ban gate (WARN-ONLY) stands; curvelet is opt-in until the A/B verdicts
  [memory `no_fourier_basis_DAG_FEED_20260715`].
- **MACHINERY:** SPEC_v8 #359/#380/#386 carrier modules; `curve_relative_offset_coder`;
  one-sided per-pair band generator terms (0 counted B, measured −29.5% on the palette-vehicle subset)
  [MEASURED, `c2_perclass_stratum_carrier_taxonomy_20260716.md` §4 — subset-ranked, n600 re-measure owed].
- **GENERATORS-FIRST (amendment 2026-07-17, binding):** P7 carriers are explicitly generators-first per
  §3.0(2) — for every cell-interior stratum the carrier ships Laguerre GENERATORS (sites/weights) and
  NEVER a field over pixels (rate ~0.02–0.05 vs 0.118 [ANCHOR]); fields are admitted only where a §3
  texture certificate exists (annulus luma, hood-tex, dash phase).
- **STATUS:** kit built (v8 increment-1); composition into the v10 archive grammar is the byte-close item.

### P8 — Seeds from v9c2's terminal state where Kolmogorov-optimal
- **WHAT:** harvest v9c2's TERMINAL artifacts as COUNTED seeds/sections — never as warm weights:
  converged self-orient field, per-dash anchors, phase carriers (#425 sections).
- **RULE:** each harvested seed passes the standing Kolmogorov test (*"fixed point of a shorter
  program?"*) before it ships; anything regenerable deterministically at decode moves to the free
  generator side. Full inventory: §4.
- **STATUS:** blocked by v9c2 completion (OQ-0) by construction.

## §3 THE NECESSITY-CERTIFICATE TABLE (the headline principle, operationalized)

**Rule:** appearance/texture/phase content is admitted to the v10 program ONLY with a certificate DERIVED
from the exact scorer factorization (`frozen_scorer_exact_factorization_20260715.md`) + the measured
inversions (necessity solver, cure-driver VJP, null-subspace, night-wet atlas). **Texture WITHOUT a
certificate = geometry-only (ξ,R + partition + generated fill).** Where a certificate needs a measurement
that does not exist, the row names the exact $0 probe — never a guess.

### §3.0 Within-class structure: flat cells (the DERIVATION of every GEOM-ONLY certificate)

**Operator amendment 2026-07-17 (verbatim: "But also remember our findings about within class structure
and flat and cells").** The GEOM-ONLY interior certificates below are not row-by-row empirical accidents —
they are ONE derivation, a three-way convergence of independent established findings, and they bind
**per-CELL, not per-class**:

1. **Fisher-flat interiors (the metric).** The frozen-scorer Fisher metric is FLAT inside class
   interiors — argmax stable ⇒ zero first-order score sensitivity — with curvature concentrated on the
   codim-1 separatrix; the margin field IS the Fisher surrogate (Pearson 0.978, MEASURED)
   [#500 `optimal_metric_unification_v1`; CLAUDE.md §unified level-set flow].
2. **Cell structure = the rate answer (the geometry).** The argmax partition is a Laguerre/power-diagram
   cell complex (#284; the τ→0 witness IS tropical — #311 TropNNC is cell-aware). The Kolmogorov-minimal
   description of an interior is its cell **GENERATORS** (sites/weights), never a field over its pixels:
   rate ~0.02–0.05 vs 0.118 [ANCHOR, v8 parsimony measure L-v8/#284]. The parsimony win IS this
   generators-not-fields swap.
3. **Measured flat-amplitude exhaustion (the dynamics).** The v9c2-line witness-own residual
   decomposition: 90.6% of residual is 1px edge-FLICKER (Road-Lane 66%), and flat-amplitude carriers are
   EXHAUSTED — post-hoc RGB on interiors is DEAD [MEASURED,
   `witness_own_residual_decomposition_v1` / `c2_witness_own_decomp_20260716`] — the empirical proof
   that interiors carry no recoverable score signal on the trained trunk.

**Binding consequences for the table:** (a) every GEOM-ONLY certificate is a **per-cell** certificate —
the certified object is the CELL (generators + border profile), transferred to pixels by cell MEMBERSHIP,
not class label (a class with many transient cells, e.g. Movable, is certified cell-by-cell); (b) an
interior's necessary content is its `|generators|` seed bytes + its border profile — never per-pixel
appearance/fields; (c) **the structural EXCEPTION is Lane: necessity-by-inversion shows Lane has NO safe
interior** [MEASURED, `realization_necessity_preimage_per_stratum_v1` family] — every Lane pixel is
effectively boundary/annulus (thin-structure cells are all-border), so no Lane row may EVER inherit an
interior/flat-cell exemption. Metric ∧ geometry ∧ dynamics agree; that convergence, not any single
measurement, is the derivation behind every GEOM-ONLY verdict below.

Verdict legend: **TEX-CERT** = texture/appearance certified necessary (scoped) · **GEOM-ONLY** = certified
no-texture · **PRECISION-ONLY** · **PROBE** = named $0 probe owed before seal.

| # | stratum | measured basis (source) | verdict | certificate / named probe |
|---|---------|-------------------------|---------|---------------------------|
| 1 | Road + Undrivable bulk interiors | ∂d_seg/∂px = 0 off-annulus; ~95% of frame d_seg-null (B2 blind, #333: ~97% of d_seg in the annulus); region identity is border-driven (2px band re-classifies interiors) [frozen factorization §8-B2; carrier taxonomy §4 region-from-boundary law] | **GEOM-ONLY** | CERT: generated palette/flat fill; interior texture spends bytes on a blind subspace |
| 2 | Movable interiors | 83% of Movable\|far cured by a 0-B one-sided 2px border band (subset −29.5% total); persist 0.865 ⇒ ξ-transportable object border [carrier taxonomy §1/§4] | **GEOM-ONLY** (ξ-tracked border profile + blur-width param) | PROBE **P-1**: n600 re-measure of the β=2 one-sided band + witness-own bucket decomp on v9c2 TERMINAL frames (bucket weights are vehicle-specific; §6 caveat of the taxonomy memo) |
| 3 | Hood interior (MyCar) | static hood-tex seed: d_seg 0.04538→0.01328 at ε=0, 1,759 B, min-S 1.613 [necessity calibration] | **TEX-CERT** (static, counted, 1.7 KB) | CERT measured; the ONE decisive static texture buy |
| 4 | Hood rim (MyCar\|edge, specular) | wet-hood specular flicker on a static boundary, visually confirmed + measured; demands light-anchored MIRROR transport (~2× angular rate), not surface ξ [night-wet atlas] | **PROBE** | PROBE **P-2**: $0 rate-vs-residual measure of a mirror-transport term vs static rim residual (0.000140 contrib) on v9c2 terminal frames |
| 5 | Boundary annuli, all class-pairs | cure gradient is 92–94% LUMA, flat-shift-ORTHOGONAL (coherence ≤0.24), non-local — the required signal is sign-alternating spatial structure = texture, on the annulus, one-sided per the measured per-pair slope field (Road-Undriv 2.78× Road-shallow; Lane shallow on both its pairs; Movable pairs symmetric-fragile) [carrier taxonomy §2/§3] | **TEX-CERT** (annulus-scoped, LUMA, one-sided) | CERT measured; deep-side/symmetric pushes FORBIDDEN (+98%/+73% measured harm); blurry low-freq realism FORBIDDEN (tex_global +78%) |
| 6 | Lane dashes | per-dash anchors (comb REFUTED: spacing CV 0.41–0.78); appearance = g(x)·φ(x−ξt) (illumination-cone gate); GT SELF-flicker floor: 38–54% of lane residual sits on GT self-flicker pixels — caps the recoverable core; the ONE flat-coherent bucket ("brighten lane side", β≤0.5) [night-wet atlas; carrier taxonomy §3] | **TEX-CERT** (phase + gate + small-β one-sided luma; amplitude sub-LSB context) | CERT measured for form; per-dash anchor BYTES (~0.9–1.8 KB DERIVED estimate) confirmed only at byte-close |
| 7 | Saddles (triple junctions) | 29.2% of saddle flips sub-LSB (uint8∩preimage TIGHT); ~0 marginal bytes given edges [necessity solver] | **PRECISION-ONLY** | CERT: precision annotation on edge carriers (tie-locus #360); NEVER texture, NEVER new byte sections |
| 8 | frame_0 (vs frame_1) | S1 discards frame_0 for SegNet — d_seg obligation 8.5e-9; pose-legible photometric content is demanded ONLY by joint descent (photometric wall) [frozen factorization §6.2; L68] | **GEOM-ONLY for seg; pose-scoped TEX as OQ-1 demands** | Consumption point: v9c2 pose-window outcome (OQ-1) sets the frame_0 photometric budget |
| 9 | Chroma vs luma | cure gradient 6–8% chroma at the flat render; PoseNet chroma 2×2-box → <2px chroma invisible (pose-safe); per-cell palette is the chroma TRUNK (6.2× worth), per-separatrix chroma the finisher [carrier taxonomy §3; frozen factorization §5/§6.4-6.5; rgb_at_boundaries] | **GEOM-ONLY + palette constants; separatrix chroma = PROBE** | PROBE **P-3**: per-pair chroma-plane margin-Jacobian projection (∂(top1−top2)/∂RGB onto span{U,V}) at n600 — certifies WHICH pairs are chroma-decided before any chroma texture ships |
| 10 | ker(A) camera-res complement | 22.70% of camera rows exactly zero under Aᵀ; ~52% of render energy scorer-invisible [null-subspace #519] | **GEOM-ONLY (generic fill, FREE)** | CERT: never a byte, never a texture; render range(A)-restricted (P3) |
| 11 | Partition geometry (cell membership + edges) | strict-necessary camera support 1.66% of pixels; bytes→edges (Road-Lane = 61% of the edge H-floor; K/H = 0.47); generator swap (openpilot-poly + phase) is the cheaper program [necessity solver] | **GEOM (counted SEEDS)** | CERT: geometry seeds counted per §4; the generator rides free |

**Row count: 11. Certificates measured: 8 (rows 1,3,5,6,7,8*,10,11 — row 8 seg-side certified, pose-side
gated on OQ-1). Named $0 probes owed: 3 (P-1, P-2, P-3).** Row 2 is certified at subset scale and OWNED by
P-1 for n600 confirmation — counted as probe-gated, not sealed.

### §3.1 Per-row re-check under the flat-cell lens (§3.0) — amendment 2026-07-17

No row CHANGED certificate class; annotations below record how each row reads under the lens:

- **Row 1 (Road/Undriv interiors): STRENGTHENED.** Was justified empirically (B2 blind + border-driven);
  now DERIVED (Fisher-flat ∧ cell-interior ∧ exhaustion). Certificate re-scoped per-CELL: content =
  Laguerre generators + palette constant per cell, not a per-class fill rule.
- **Row 2 (Movable interiors): STRENGTHENED + probe-interaction noted.** Movable is many small TRANSIENT
  cells (persist 0.865, ξ-transportable) — the per-cell binding matters most here: each object cell
  carries its own generators + one-sided border profile under ξ-transport; brief cell births/deaths are
  event handling (§5), not texture. **P-1 interaction:** the n600 re-measure + v9c2-terminal decomp must
  report PER-CELL (per-object) buckets, not class aggregates, so a failing transient cell cannot be
  misread as "Movable interiors need texture" (that would violate the cell-membership binding).
- **Row 3 (hood interior): UNCHANGED (the one certified interior-texture buy).** Consistent with §3.0:
  the hood is a single STATIC cell whose measured cure (1,759 B, −71%) is border+texture signature the
  generators cannot produce — the exception that carries its own measured certificate, exactly as the
  headline principle demands.
- **Rows 4/5 (rims + annuli): UNCHANGED-CONSISTENT.** Curvature (and hence certified texture) lives on
  the separatrix — the lens PREDICTS these are the texture-bearing strata; the slope-field one-sidedness
  is the anisotropic boundary metric read as a carrier rule.
- **Row 6 (Lane dashes): EXCEPTION MADE EXPLICIT.** Lane has NO safe interior (necessity-by-inversion)
  — every Lane pixel is boundary/annulus; the row can NEVER inherit an interior/flat-cell exemption.
  Its phase/gate/small-β certificate is annulus-class content by construction; dash cells are pure
  birth-death boundary objects (per-dash anchors ARE their generators).
- **Row 7 (saddles): STRENGTHENED.** Saddles = the cell complex's 0-cells; precision-only follows from
  the same geometry (tie-locus points are generator-coincidence loci, not content).
- **Row 8 (frame_0): UNCHANGED** (S1 selection argument is upstream of the cell lens; pose-side still
  OQ-1).
- **Row 9 (chroma): STRENGTHENED.** "Per-cell palette is the chroma TRUNK" is §3.0(2) verbatim — the
  palette constant IS a cell-generator attribute; P-3 decides which SEPARATRICES additionally need
  chroma.
- **Row 10 (ker(A)): UNCHANGED-COMPOSING.** See P3 amendment — ker(A) restriction and cell-generator
  description compose as projections onto the same σ-algebra.
- **Row 11 (partition geometry): STRENGTHENED.** "Geometry seeds counted" now reads precisely as: the
  counted seeds ARE the cell generators + edge coefficients; the Laguerre expansion rides free.

## §4 Seeds inventory (COUNTED vs FREE; rule-118 boundary)

FREE (generic deterministic program in inflate.py — rule 118): the Dykstra projection composition G,
Laguerre/palette cell generator, openpilot polynomial + homography lane rasterizer, dir/Fourier feature
bank expanded from a recorded RNG seed, the exact A/R operators, per-pair one-sided band generator
(the CODE), illumination-cone gate FUNCTIONAL FORM, decode program itself.

COUNTED (video-derived / learned; ship in archive.zip; sizes measured at byte-close only):

| seed / section | est. bytes | provenance of estimate | harvest source |
|---|---:|---|---|
| trained trunk weights (irreducible residual after all solves/seeds) | THE main item — content-priced (P4) | rate-law ladder anchor: D37 net 384,637 B (ceiling to beat; D38 #483 owed) [ANCHOR] | v10 training (cold start) |
| static hood-tex seed | 1,759 B | MEASURED (necessity calibration) | necessity frame builder |
| per-dash lane anchors | ~0.9–1.8 KB | DERIVED (L86 estimate) — byte-close confirms | v9c2 terminal (P8) or GT analysis |
| converged self-orient field | TBD — low-order compressible | field is video-derived → COUNTED if the decode-time forward needs it; Kolmogorov test at harvest: if regenerable from shipped witness geometry, move to FREE | v9c2 terminal (P8) |
| phase carriers (#425 sections) | TBD | byte-close tool `--phase-carrier` (WIRED) | v9c2 terminal (P8) |
| illumination-cone gate g(x) coefficients | ~tens of B (low-order) | DERIVED (night-wet atlas "near-free low-order seed") | static fit to GT luma field |
| one-sided per-pair band constants (side, β, kpx) | ~0 B (a few constants) | MEASURED form (taxonomy §4) | slope-field table |
| pose: store-nothing arm; banked R1 dxi fallback | 0 B (store-nothing) / 7.2 KB (fallback) | MEASURED (R1) | #314 arm / bank |
| saddle precision annotations | ~0 new B (rides edge seeds) | MEASURED (necessity solver) | tie-locus #360 |
| edge geometry coefficients (curves as ~8 coefficients + phase) | ≪ 303,047 B chain-coding floor (the generator swap IS the saving) | MEASURED H-floor; generator-swap size at byte-close | curve_relative_offset_coder → generator |

**No projected total-S is stated** — per NO-FAKE #8 a rate claim exists only as a byte-closed archive.zip
stat; the ladder anchors above are ceilings/floors, not predictions.

## §5 COMPLETENESS TABLE (the 4th leg — forces demanded by MEASURED dynamics × v10 disposition)

| force demanded (measured dynamics) | evidence | v10 disposition |
|---|---|---|
| per-event island birth/death handling (churn ~9.4 births + 9.5 deaths/step) | degraded-lane-markings audit | PRESENT: Force-3 `TieLocusDisplacement` + event-fallback phase supervision (form per v9c2 amendment outcome, OQ-4); birth-completion class-level form is a FORMULATION-MISMATCH to per-island events — reformulation queue |
| sub-pixel advection phase (dominant flicker mode; GT sub-pixel advection) | L85/L86; temporal ξ-transport jitter-bound | PRESENT: #424 `PhaseAdvectionConsistency` + #425 phase carriers |
| illumination-cone gain modulation g(x) | night-wet atlas (cone-edge margin modulation) | PRESENT (NEW in v10): static gate seed — without it the phase stack wastes rate attributing cone modulation to per-dash appearance |
| light-anchored mirror transport (wet road/hood specular) | night-wet atlas (streak columns mimic paint; hood rim) | PRESENT probe-gated (P-2): second transport law |
| one-sided separatrix restoring force (per-pair side from slope field) | taxonomy §2/§4 (−29.5% at 0 B) | PRESENT: per-pair one-sided band generator term |
| joint pose descent (photometric wall) | L68 (5 formulations dead) | PRESENT: `PoseFinishConditioningGate` + store-nothing #314; dxi fallback |
| resume/fork boundary physics (LR warm-up, EMA clearance, moment reset) | #518 warm-start law + admission physics | PRESENT post-merge (P5) |
| **FORBIDDEN:** persistence-hold force | would fight GT's genuine deaths (completeness = also NOT adding) | EXCLUDED |
| **FORBIDDEN:** symmetric / deep-side boundary pushes | +73% / +98% measured harm | EXCLUDED |
| **FORBIDDEN:** l7 stage | measured defect (L∞ sharpening in a viscosity flow) | EXCLUDED (l7-start = epochs+1, the TRUE never-runs form) |
| **FORBIDDEN:** smooth-stage | measured d_seg RAISE | EXCLUDED from curriculum |
| **FORBIDDEN:** fixed-β hosc | diverges (tanh saturation) | EXCLUDED; annealed hosc (β 1.0→4.0) or step_basis only |
| **FORBIDDEN:** blurry low-frequency realism / interior texture paste | tex_global +78%, tex_band +87% | EXCLUDED (instance-scoped negatives; family "trained blended band" stays open) |

## §6 Launch-gating chain (in order; each gate = an artifact, not a vibe)

1. **v9c2 completion** (run `levelset_n600_witness_20260717T113932Z` reaches its governed stop) +
   terminal harvest: full-facet read (per-class d_seg vs anchors · island birth · d_pose vs need · rate)
   + P8 seed harvest + P-1 terminal-frame decomp.
2. **p0_497 curvelet matched-bytes A/B** verdict artifact → P7 basis choice (`curvelet_through_R_dseg_ab`).
3. **#518 8-vs-27 warm-up A/B** (short arm) + #518 branch MERGE → P5 defaults become importable Lever
   factories (the DSL factory stops refusing).
4. **$0 probes P-1/P-2/P-3** landed (necessity table rows 2/4/9 sealed or re-verdicted).
5. **DSL compile clean:** `spec_v10_capstone_20260717.compile_v10_capstone_launch_config()` returns a
   config with ZERO blockers (post-merge deps resolved, gate artifacts present, seeds inventoried).
6. **Governed launcher + memory preflight** (tools/launch_witness_run.py registration owed at GO time;
   projected peak RSS at the REAL config; resumable + per-stage checkpoints P0).
7. **Operator GO** (CONTAINMENT: this SPEC and its factory never launch anything).

## §7 OPEN QUESTIONS (honest — only v9c2 + the A/Bs can answer; each with its consumption point)

| id | question | decided by | consumed at |
|---|---|---|---|
| OQ-0 | v9c2 terminal facet state (does the warm surgical arm validate the phase stack at all?) | v9c2 completion | gates 1/4; necessity rows 2/4 re-weight (bucket weights are vehicle-specific) |
| OQ-1 | pose window outcome — does the #383 conditioning gate engage and does joint descent land d_pose ≤ need on THIS vehicle class? | v9c2 pose stage | P6 form; necessity row 8 frame_0 photometric budget; §4 dxi-fallback ship/no-ship |
| OQ-2 | curvelet verdict (matched-bytes, through-R) | p0_497 A/B | P7 trunk basis; no-more-Fourier gate disposition |
| OQ-3 | warm-up law: 8 vs 27 (beta2-derived) | #518 short arm | P5 `ResumeLRWarmup` default constant (LawRef) |
| OQ-4 | event-fallback phase supervision amendment (pre-ep700) — does per-event coverage close the 26.3% straddle gap? | v9c2 mid-run amendment | §5 event row; the v10 phase-stage form |
| OQ-5 | tie-locus Force-3 efficacy (@ep700) | v9c2 | §3 row 7 precision mechanism; `TieLocusDisplacement` default |
| OQ-6 | #406 n600 gauge-canonicalization confirm | #406 run | P2 seal (currently measured on donor witness) |

## §8 DSL module (deliverable 2)

`src/tac/witness_dsl/spec_v10_capstone_20260717.py` — mirrors the `spec_c2_surgical_20260716.py` pattern:
module-level pillar/gate metadata, a `compile_v10_capstone_launch_config()` entry point, $0/pure,
CONTAINMENT (never launches). **Fail-closed by design:** the factory REFUSES (raises
`SpecV10CompileBlocked` with the full blocker list) while (a) any #518 post-merge lever
(`ResumeLRWarmup`, `ForkHeadSolve`, `MarginStepCap`, `PoseEngageWPoseRamp`, `ForkEmaClearance`) is not
importable from `tac.witness_dsl.curriculum_dsl` — deps are NEVER fake-resolved; (b) any §6 gate artifact
is missing; (c) the seeds inventory artifacts are absent. Every constant it carries is a
(value, provenance, cite) triple on the value-provenance ladder — no bare constants. The full
flag-for-flag base config is deliberately NOT emitted in this landing (that would be a fake config for an
unresolved composition); it compiles only after gates 1–4 clear.

## §9 Value-provenance note

Every number in this SPEC is labeled at point of use: MEASURED (with source artifact/memo), DERIVED
(from pinned upstream source or closed form), ANCHOR (historical ladder rung), or OPEN (OQ). No constant
is bare; the DSL skeleton encodes the same triples. LawRef registration for the birth-default constants
(P5) is owed at post-merge fold time and named as such.

## §10 DAG FEED block (ready to paste into `sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`)

```
FEED-521-v10spec (2026-07-17) — SPEC_v10 THE CAPSTONE assembled (task #521, branch
claude/p0_521_spec_v10_capstone_20260717). v10 = COLD START on a FULLY SEEDED KOLMOGOROV-OPTIMAL
program (naming SoT vehicle_naming_resolution_v10_capstone_20260717.md). SPEC:
.omx/research/SPEC_v10_capstone_cold_start_seeded_20260717.md; DSL skeleton
src/tac/witness_dsl/spec_v10_capstone_20260717.py (fail-closed on #518 post-merge levers
ResumeLRWarmup/ForkHeadSolve/MarginStepCap/PoseEngageWPoseRamp/ForkEmaClearance + gate artifacts;
CONTAINMENT, $0). 8 pillars realized with existing machinery (hood/lane seeds · rank-4 head solve +
#519 gauge precision lever · range(A)-restricted render #520 · content-priced coder #336/#461 ·
#518 boundary laws · store-nothing pose #314 + joint descent · v8 #386 carriers + p0_497 basis-cure ·
v9c2 terminal seeds). NECESSITY-CERTIFICATE TABLE: 11 strata, 8 measured certificates
(texture admitted ONLY on: hood-tex 1.7KB static + boundary-annulus one-sided LUMA + lane dash
phase·g(x)·small-β; everything else GEOM-ONLY/PRECISION-ONLY/FREE), 3 named $0 probes owed
(P-1 n600 band+terminal decomp · P-2 mirror-transport rate · P-3 per-pair chroma-plane Jacobian).
COMPLETENESS table: 7 forces present, 6 forbidden-and-excluded. Launch chain: v9c2 completion →
p0_497 → #518 A/B+merge → probes → clean DSL compile → governed launcher → operator GO.
OQ-0..6 recorded with per-question consumption points (pose window → P6/frame_0 budget;
tie-locus → row-7; curvelet → P7). OPERATOR AMENDMENT folded same-day (§3.0/§3.1): within-class
flat-cell derivation — GEOM-ONLY certificates are DERIVED (Fisher-flat #500 ∧ Laguerre-cell #284/#311
∧ flat-amplitude-exhaustion) and bind PER-CELL not per-class; interiors described by GENERATORS not
fields (~0.02-0.05 vs 0.118 rate ANCHOR); Lane carries the no-safe-interior exception; range(A)+
cell-generators compose (σ-algebra atoms, P3); P7 generators-first binding; P-1 upgraded to per-cell
reporting. All 11 rows re-checked: 0 certificate-class changes.
Pointer 0.19108 UNMOVED (means/apparatus — SPEC only).
```

## §11 Round-1 adversarial review (own attack — findings + fixes, landed in this document)

1. **F1 (fake-config risk):** an earlier draft emitted a full flag-for-flag v10 base config. That would be
   a FAKE (the composition depends on 7 open questions). FIX: the DSL factory compiles BLOCKERS
   fail-closed; the base config is explicitly deferred to post-gate fold (§8). No launchable argv is
   claimed to exist.
2. **F2 (borrowed-number risk):** the taxonomy's −29.5% one-sided band and the movable-interior
   certificate are SUBSET (stride-5, 120-frame) rankings on the PALETTE vehicle, and bucket weights are
   vehicle-specific. FIX: row 2 downgraded from CERT to probe-gated (P-1: n600 + v9c2-terminal decomp);
   the number is labeled subset-ranked everywhere it appears.
3. **F3 (self-orient seed mislabeling):** the converged self-orient field is video-derived; calling it
   free would be the hide-data-in-code fake. FIX: §4 marks it COUNTED-if-shipped with the harvest-time
   Kolmogorov test named; disposition at byte-close, never asserted.
4. **F4 (rate-projection temptation):** citing D37 (384,637 B) as a v10 rate prediction would be a
   surrogate score claim. FIX: §4 states no projected S; ladder rungs labeled ANCHOR (ceiling), rate
   claims only via archive.zip stat.
5. **F5 (flicker-floor over-claim):** the 0.005318 GT-oracle floor is FORMULATION-scoped (binds only
   smoother-than-GT witnesses; counter-proofs 0.00086/6e-4) — an earlier draft used it as a v10 d_seg
   floor. FIX: removed as a floor; retained only as context in the GT self-flicker cap on lane-dash
   recovery (38–54%, which upper-bounds label noise and includes 1-frame advection).
6. **F6 (pillar-5 dead-flag risk):** naming #518 levers in a compiled argv before merge would violate
   never-invent-flags. FIX: post-merge deps are import-probed at compile time and converted to typed
   blockers; the factory refuses, it never guesses flag names.
7. **F7 (chroma blanket claim):** "chroma is a d_seg lever" (CLAUDE.md) vs "cure is 92–94% luma"
   (taxonomy) could be read as contradiction. RESOLUTION recorded: the luma number is measured AT THE
   FLAT PALETTE RENDER; chroma-decides applies at GT-textured inputs and per-pair — hence P-3 measures
   the per-pair chroma-plane Jacobian before any chroma texture ships (no blanket verdict either way).
8. **F8 (gate-artifact path contracts):** the skeleton's gate/seed artifact paths (e.g.
   `RUN_COMPLETE.json`, probe `verdict.json` paths, `hood_tex_seed.npz`) are DECLARED contracts, not
   existing files — the producers bind to them at harvest/fold time (or the path constants are updated
   in the same fold commit). This is safe BECAUSE the factory is fail-closed either way: a wrong path
   can only over-block, never under-block. Verified live: `spec_v10_status(".")` returns 13 blockers
   (5 post-merge levers + 6 gates + 2 seeds) and `compile_v10_capstone_launch_config` raises
   `SpecV10CompileBlocked` — the compile surface is present and honest, not a marker.

## §12 Triality + stores consulted

- **DAG leg:** FEED-521-v10spec (§10, ready-to-paste; appended by the main-session integrator at merge —
  this worktree does not mutate the shared DAG file to avoid clobbering the live session's appends).
- **DSL leg:** `spec_v10_capstone_20260717.py` skeleton (this landing); full lever fold owed post-#518-merge.
- **equations leg:** consumed (not new law): `realization_necessity_preimage_per_stratum_v1` ·
  `segnet_head_rank4_linear_flipdist_v1` · `necessity_generator_seed_dseg_calibration_v1` ·
  `perclass_stratum_residual_carrier_taxonomy_v1` · `witness_own_residual_decomposition_v1` ·
  `rate_law_ladder` family · `optimal_metric_unification_v1`. New-law registration is owed only when a
  v10 MEASUREMENT lands (none here — this is a SPEC; `# FORMALIZATION_PENDING:spec-only-no-new-measured-finding`).
- **STORES CONSULTED:** vehicle_naming_resolution_v10 (charter) · SPEC_v75 §8 · SPEC_v8 ·
  projection_unification_and_eight_lenses · frozen_scorer_exact_factorization ·
  necessity_solver_inverse_factorization + necessity_dseg_calibration · null_subspace_rate_measure (#519) ·
  c2_perclass_stratum_carrier_taxonomy · night_wet_video atlas · degraded_lane_markings ·
  c2_witness_own_decomp · train-least + completeness doctrine memos · warm-start law memo ·
  no_fourier ban · L65/L68/L85/L86 · graph_memory_recall (necessity/frozen-scorer/train-least/projection).

**Pointer 0.19108 UNMOVED — this SPEC is MEANS.** The v10 program exists to land a byte-closed
`upstream/evaluate.py` n600 exact row below it, then toward sub-0.15.

## §13 SESSION FOLD 2026-07-17B — triggers/forces review · Fisher actuation · EMA law · dual-metric discipline (SSoT per operator: "All of this should be reflected in the v10 spec we need a single source of truth")

Source authority: operator directives 2026-07-17 (triggers/forces review "all p0 regardless of severity" ·
"We are not fully leveraging fisher" · "EMA calibration is p0 too" · dual-metric "both are informative stop
forgetting that" · "Build all unbuilt now"). P0 ledger rows: p0_triggers_forces_review_all_findings_20260717 ·
p0_fisher_full_leverage_20260717 · p0_ema_calibration_20260717 · p0_boundary_merge_queue_post_v9c2_20260717.
Tasks #524/#525. Measured basis: live c2 run 20260717T113932Z telemetry (per-class verdicts ep650–800,
sps_gradient_role ep701, jacobian_basin ep786–798, ema_warmup_updates=667) + launch.sh flag extraction +
FEED-lane-gain (event-fallback missing) + segnet_recursive_fractal (skip-limited) + FEED-we-conflict
(Euclid-vs-Fisher sign flip). Everything below is MEASURED or DERIVED unless marked.

### §13.1 Completeness-table amendments (extends §5 — forces the measured dynamics demand)
| force | demanded by (measured) | v10 disposition |
|---|---|---|
| event-fallback phase supervision | 26.3% straddle sites uncovered; phase stack transport-only, birth-SILENT (FEED-lane-gain); Lane per-class FLAT ~0.237 through Force-3 while flip-share fell | BUILD (wave arm B), default-OFF DSL lever; Δd_seg duty-to-measure at the c2 per-stage A/B |
| Fisher-density seg-loss weight w(x)=sech²(m/2) | the EXACT registered Fisher law (tr g=½sech²(m/2), ρ0.978); current satisfice/subpix use empirical flip-mass proxies | BUILD (wave arm A), default-OFF; $0 A/B on cached ckpt |
| per-class-pair σ_cc′ anisotropic tension (#382, BUILT) | Γ-limit demands per-pair σ; live scalar length-weight 0.001 is the MCF-lane-erasure term; Lane 0.2368 stuck consistent | COMPOSE into the v10 config (no rebuild); scalar length term demoted where σ_cc′ active |
| Lane stride-2 skip lever | Lane flips 77% limited by the 16-ch stride-2 skip path (fractal factorization); verified UNBUILT 2026-07-17 | BUILD (wave arm C, #524), default-OFF; targets the dominant per-class residual |

### §13.2 Trigger laws — constants dissolve into event-continuation (extends §2/§6)
- **Force-entry triggers**: fixed-epoch co-engagement (c2's phase_advect+satisfice+subpix all @ep700) is a
  stage-skeleton constant; it confounded attribution (ep750 Road-concentrated +50% tax spans BOTH Force-3
  settling AND ep726 cold-Muon entry). v10: per-force event triggers (d_seg slope-flatten OR λ-critical via
  the #344 NCDE hit→solve detector — currently fire=unavailable, MUST be wired), staggered engagement for
  attribution.
- **Pose-finish gate**: sigma_min_plateau is the right event CLASS (and reads LIVE weights — verified immune
  to EMA lag); v10 adds the fire-on-crest alternative (σ_min slope sign-change = conditioning peak; live c2
  measured σ_min 0.0010→0.0068 still climbing +15%/ep at ep798) and puts flat_rel_band 0.0003 / hysteresis 3
  / settle 3 on the provenance ladder (currently bare).
- **Coupling not coincidence**: anneal-epochs(1000) == pose-finish-start(1000) is two constants agreeing;
  v10 expresses it as the event coupling β-anneal-complete → pose-finish-eligible.
- **Trigger observable**: convergence/engagement events read Fisher-mass-in-annulus (the decision-geometry
  observable), not raw Euclidean d_seg slope.

### §13.3 Constants → LawRefs (extends §9)
- **ema_decay**: MEASURED — 0.997/update × 1 update/epoch (full-batch accum) ⇒ τ≈333 EPOCHS; declared
  warmup 2/(1−d)=667 updates ≈ ep1318 on a 1400-ep run (the shadow spends the whole run inside warmup;
  ~64% warm-start seed @ep800; EMA−live −0.00095 @ep775). The 0.997 provenance (Quantizr per-step minibatch)
  does NOT transfer to the deterministic full-batch regime (noise-averaging rationale vanishes). v10 LawRef:
  decay derived from (updates_per_run, measured update noise); finisher horizon via the BUILT never-fired
  --ema-decay-finisher lever (SWA-style). Mechanics verified SOUND — this is calibration, not a bug.
- **w_pose**: 1.0 → w_pose(t) = 5/√(10·d_pose(t)) — the exact score marginal (the score's own derivative).
- **Muon entry**: --muon-warm-start-momentum / --muon-lr-final-frac default-OFF cold entry is the measured
  #269 gap; v10 default = warm-start momentum + LR-anneal ON, pending the #270 per-stage A/B verdict.

### §13.4 Fisher actuation — measurement → actuator across 6 surfaces (p0_fisher_full_leverage)
(1) TRAINING FORCE: the §13.1 Fisher-density weight (arm A). (2) OPTIMIZER: head-space natural gradient is
CLOSED-FORM CHEAP because the head is exact rank-4 linear — composes with #423 Hessian-preconditioned
head-offset + #518 fork head-SOLVE (arm A). (3) TELEMETRY: dual-metric read-backs — every load-bearing
gradient alignment/conflict claim reports Euclidean cosine AND Fisher cosine AND rel-norm (the decisive
quantity); measured anchor: sign FLIP on the same pair (−0.00105 Euclid vs +0.0435 Fisher, FEED-we-conflict);
the sps global_cosine is EUCLIDEAN-only; Fisher read-back of phase_advect OWED (arm A harness).
(4) RATE/PRECISION: Fisher-sensitivity bit-alloc (#157/#336) + ker(A)/gauge 22% int8-scale coarsening (#519)
— BUILT, applied in the #406 post-run batch. (5) TRIGGERS: §13.2 Fisher-mass-in-annulus observable.
(6) AVERAGING/SELECTION: decision-geometry-aware checkpoint averaging/selection, joint with the §13.3 EMA law.

### §13.5 Terminal measurement gates (extends §6 — each gate = an artifact)
- **Shadow-vs-live byte-close A/B** at terminal + per-stage checkpoints (arm C comparator) — ship the winner;
  decides the §13.3 EMA question empirically.
- **Fisher read-back of phase_advect** on a cached checkpoint (arm A harness) — closes the owed dual-metric gap.
- **Boundary A/B set**: Muon warm-start (#270) · event-fallback Δd_seg · the duty-queue top-3
  (DsegAwareTaper 78.9% / HorizonWeightedMargin 47.3% / StepNativeActivation 34.2% of remaining descent —
  the apparatus's own top-ranked unfired levers, fired BEFORE lower-ranked new work).

### §13.6 Realization — the 2026-07-17 build wave (all default-OFF, live run untouched, merge at post-v9c2 boundary)
Arm A `p0_build_fisher_actuation_20260717`: Fisher-density weight lever + rank-4 head natural-gradient lever +
dual-metric read-back harness. Arm B `p0_build_forces_triggers_20260717`: event-fallback phase supervision +
fire-on-crest gate option + event-coupling triggers + w_pose(t) law. Arm C `p0_build_skiplever_ema_20260717`:
stride-2 skip lever + EMA decay LawRef + shadow-vs-live comparator + ema_decay_finisher duty registration.
Every lever lands as a DSL Lever factory (never a hand trainer flag), tested, with equations-leg registration;
recursive adversarial review to 3 clean passes gates merge eligibility. This section is the SSoT these arms
implement against; drift between an arm's landing and this section is a triality violation.

### §13.7 CURVELET / FOURIER-REPLACEMENT — takeover fold (operator 2026-07-17 "Don't forget curvelets")
The basis question is a FIRST-CLASS member of the §13.5 boundary A/B set, not an afterthought. State:
branch claude/p0_497_curvelet_matched_bytes_ab_20260717 holds the sealed receiver paths (gap-a
charted_grid_bilinear_v1 counted receiver-program · gap-b post-render supersample A_s, both
bit-exact-gated); four codex basis findings memos (curvelet_throughR_p0 · genuine_curvelet_shearlet ·
no_fourier_basis_sweep · optimal_basis_beyond_fourier) are reconciled by ARM D
(.omx/tmp/build_wave_20260717/ARM_D_CURVELET.md), which VERIFIED the matched-COUNTED-bytes curvelet-vs-Fourier through-R d_seg A/B FIREABLE (2026-07-17 landing, commit 50d12d3382 + memo curvelet_takeover_fireable_20260717.md: fire tool PRE-EXISTED on the branch — the earlier 'missing fire tool' premise was WRONG, absent from main only pending post-c2 merge; both arms governed dry-run PASS rc=0, config diff = exactly the basis lever, receiver custody rule-118-clean, 142 tests green; fire sequence + pre-registered verdict criteria in the memo §5, sequential-only per the memory guard). Standing law: Fourier ban-gate
WARN-ONLY, curvelet opt-in; the measured 3.2× along-tangent deficit (#497) is the quantity at stake;
the A/B verdict decides the v10 basis (curvelet adoption vs Fourier retention) — pre-registered criteria
in curvelet_takeover_fireable_20260717.md. FIRE = operator-GO sequential at the post-v9c2 boundary,
ordered per the OLD-FINDINGS-FIRST rule (p0_SUPREME row) alongside the duty-queue top-3. P0 rows:
p0_curvelet_fourier_replacement_takeover_20260717 + p0_497_basis_cure_decisive_ab.

### §13.8 Arm A landing (2026-07-17): Fisher actuation BUILT + the owed dual-metric gap CLOSED with a NEW measured fact
Branch `p0_build_fisher_actuation_20260717` (5 commits, 59 tests): Fisher-density seg weight (exact
sech²(m/2) law, model/gt sources, DERIVED model=Fisher-natural) · rank-4 head natural-gradient
(forward-identity/backward-g⁺, bitwise OFF-identity proven) · dual-metric harness
`tools/dual_metric_readback.py` · Fisher-annulus observable — all default-OFF DSL Levers, resume-safe.
**MEASURED (the §13.4(3) owed read-back, live-c2 BEST ep725, n96, [macOS advisory]): phase_advect vs
armed seg base = Euclid cos −0.149 · Fisher cos −0.118 · rel-norm 0.627 Euclid / 0.478 Fisher — the
phase stack is a LARGE (≈½ the seg force), mildly ANTAGONISTIC term at ep725, in BOTH metrics.**
Reconciliation with the ep701 sps reading (+0.238 Euclid, no-conflict): different epoch + partition —
at engagement (ep701) the phase force pulled TOWARD the new optimum; by ep725 boundaries had
repositioned and the force mildly opposes further seg descent. Consistent with the ep750 regression +
the measured σ_min-erosion coupling (§13.2). v10 CONSEQUENCE (feeds §13.1/§13.2): the phase weight
(c2 constant 0.4) is a LATE-PHASE annealing/event-gating candidate — hold phase hard while building
conditioning, relax as seg re-descends; decide by the per-stage A/B, not by constant. Muon×NG
double-preconditioning flagged open. Sister risk recorded: focal/Fisher-density DISAGREE on
confidently-wrong pixels — compose at most one of the two per config.

### §13.9 Arm B landing (2026-07-17): forces + triggers BUILT — the missing force exists, the #344 gap is root-caused, the clamp is DERIVED
Branch `p0_build_forces_triggers_20260717` (5 commits, 57 new tests + regressions green, all default-OFF,
OFF-path behavior-identical): **(1) event-fallback phase supervision** (§13.1 row 1 CLOSED as a build:
memo-exact t_ref fallback to own-GT tie at straddle sites, stateless, no persistence-hold per anti-scope)
— crux-3's missing force now EXISTS; Δd_seg = duty-to-measure #1 at the boundary per-stage A/B.
**(2) fire-on-crest gate** (`--pose-finish-engage-on sigma_min_crest`) with the LIVE c2 crest (peak
~ep802) folded in as a regression test anchor. **(3) event couplings**: β-anneal-complete →
pose-finish-eligible (the §13.2 coincidence dissolved; eligibility-constant measured-suboptimal stated
in the DSL) + the #344 NCDE event-entry consumer with THREE measured false-negative fixes found
in-build, and the **fire=unavailable ROOT CAUSE MEASURED**: <8 verdict rows ⇒ silent None omission —
fixed observer-side (always-structured probe), verified read-only against the live run. λ-critical
entry NOT built (needs a trainer per-class-λ stream — honest routing, not half-wired). **(4) w_pose(t)
= 5/√(10·d_pose) law** with the clamp DERIVED (= seg marginal 100 at the crossover d_pose 2.5e-4 — the
score's own geometry, not a chosen constant), verdict-cadence piecewise-constant, pose-finish-only
consumption. Phase-weight relaxation deliberately NOT wired (pa_w closure-captured in the compiled
loss; half-wiring forbidden) — routed to the boundary owner with the ep725 antagonism anchor. Open
(round-2): detector mode not resume-persisted; ncde BASIN window sensitivity; D4 live/EMA d_pose
source second-order. P0 consequence: triggers/forces F2+F4+F6 → BUILT-awaiting-measure; lane-crux-3
TRAIN side complete pending the A/B; #425 STORE leg remains the unbuilt half.

### §13.10 Arm C landing (2026-07-17): skip lever BUILT+BINDS with a MEASURED 10× deficit · EMA law VALIDATED · shadow-vs-live FIRST DATA
Branch `p0_build_skiplever_ema_20260717` (5 commits, 77 tests). **(1) Lane skip-band lever (#524, crux-1):**
numpy-fp32 reference SB = D2 − U2(D4) on BT.601 luma (the stride-2 skip passband derived from the
factorization), FD-verified closed-form adjoint, default-0.0 byte-identical, DSL `LaneSkipBand`.
**Bindingness MEASURED (n24 real mod32cap render): BINDS (term 1.55e-3, grad>0) and the witness carries
only 1.68e-4 skip-band lane energy vs GT 1.70e-3 — a live ~10× DEFICIT**, quantifying crux-1's mechanism
on OUR render: the witness under-supplies exactly the band SegNet's decisive Lane path reads. Δd_seg =
boundary A/B. **(2) EMA decay law (§13.3 → VALIDATED):** `ema_decay_run_geometry_v1` REGISTERED with 2
anchors — the warmup-667 executable cross-check AND seed-fraction 0.6391@ep800 vs the SPEC's derived
~64% (residual 9e-4: the §13.3 arithmetic is now an anchored equation, not prose). `EmaDecayCalibrated`
LawRef-resolves --ema-decay; `EmaDecayFinisher` duty REGISTERED on the main ledger (orphan cured).
**(3) Shadow-vs-live comparator + FIRST MEASURED ROWS (mod32cap ep1000, n96, [macOS advisory]):**
d_seg EMA 0.003976 vs live 0.004884 — **EMA wins by 18.6%** on the meaningful axis at a converged-basin
checkpoint (composite-S ranking pose-blind on that w_pose=0 ckpt — tool banners it; caveat recorded).
First real datum for the §13.5 ship-the-winner gate; the c2 terminal comparison remains the decisive one.
Open (round-2): micro-batch twin routing for the lever (fail-closed, live family B=1 unaffected);
1-epoch ON smoke; feature-space (16-ch stem) upgrade path if the band form under-delivers; pose-aware
comparator ranking; idempotent duty re-registration post-merge. Wave status: build arms A/B/C ALL LANDED;
G (#425 store leg) building; boundary merge queue += this branch.

### §13.11 Arm G landing (2026-07-17): #425 phase-carrier STORE leg BUILT + n600 MEASURED — object-domain coding pays, the event stream is the new rate crux
Branch `p0_build_phase_carrier_425_20260717` (5 commits, 49 tests, equation
`dash_phase_carrier_rate_blinkback_prior_divergence_v1` REGISTERED). Curve-domain per-dash δ(s) codec:
ξ-advected world tracks, prior-derived canonical Huffman, explicit birth/death/REBIRTH events with a
dormant world-anchor pool, byte-close `--dash-phase-carrier` section, NO-FAKE bit-identity +
every-seed-byte-consumed refusal. **MEASURED (n600, [macOS-CPU advisory]):**
- **Rate 29,958 B excl-ξ = 11.3× UNDER the naive lane raster (338,523 B)** — the OBJECT-DOMAIN
  reactivation of the raster jitter-bound negative (amortization 0.71<1) is CONFIRMED: coding dashes as
  world objects beats coding pixels. BUT **16.6× OVER the 0.9–1.8 KB per-dash-anchor budget** — the gap
  is the alive/rebirth EVENT STREAM; that budget implicitly assumed a FREE visibility generator
  (persistence-class prediction of which dashes are visible per frame). The visibility generator is now
  THE named rate crux for this carrier (rule-118-free candidate: deterministic persistence classes).
- **Blink-back 0.787** — first measurement of the lane-memo §4 open item (79% of reappearances re-anchor
  dormant world tracks): the rebirth/dormant-pool design is justified by the data.
- **Prior-transfer NEGATIVE (honest, formulation-scoped):** the site-level jitter prior does NOT
  transfer to dash centroids (pre-registered 4.53 bits/dash → realized 9.58; iid prior code loses to
  zlib9 by 20%). Cures named in the equation's reactivation criteria: dash-level measured prior in the
  header + per-track context coding.
- **Calibration DISCOVERY with cure-adjacency:** raw s_t=1 pose→ξ (the #359 convention) mis-advects;
  fitted (s_t=−0.00322, pitch=−0.01) lifts transport coverage 52%→87% and cuts the section 24%.
  **Sister flag: #359's ξ_amort=1.041 is plausibly the SAME bug — re-measure owed** (a per-class-carrier
  economics number may improve for free).
- **Recovery (label-space, honestly scoped; through-R d_seg on c2 ep725 EMA = OWED at the boundary):**
  phase-correct centroids 0.38 px mean (100% ≤1px); lane XOR 0.749 vs persist 1.129 (−34%);
  transport-only is WORSE than persist — **ξ's value here is WORLD IDENTITY for events, not the warp**
  (refines lens-5: transport codes identity, the phase seed codes position).
Crux-3 status: TRAIN legs (B) + STORE leg (G) both BUILT; the joint through-R A/B at the boundary is the
remaining decisive measurement. Wave COMPLETE: arms A/B/C/D/E/F/G all landed.

### §13.12 Fisher TRAJECTORY force-map (2026-07-17, Arm A commit 18eea72738) — the per-epoch evidence, and the finish-compression go/no-go ANSWERED
16-row dual-metric table across {seed-ep650, ep725-BEST, ep726-stage, ep800} × {phase_advect, margin_satisfice,
subpix_boundary, weight_entropy-λ15} vs the armed seg base (trainer-faithful Force-2/3 replicas incl. pa_flipmass;
[macOS advisory, n96, EMA-shadow states, NON-PROMOTABLE). MEASURED:
- **Phase sign-flip at engagement CONFIRMED across the trajectory:** Fisher cos +0.17 → −0.12, rel-norm
  2.0 → 0.48 → 0.29 — the force is large-and-aligned at engagement, then flips antagonistic and shrinks as
  boundaries settle. Pins the §13.8 single-point read to a trajectory (answers "when did it flip" = at/just-after
  engagement). Late-phase anneal/event-gate (§13.2/§13.9) is the correct v10 form.
- **weight_entropy finish-crossover ANSWERED (the rate-question go/no-go):** rel-norm at ep800 = 0.153 Euclid /
  0.061 Fisher. Euclid has CROSSED the ~0.1 binding threshold; **Fisher has NOT** → per the dual-metric
  discipline, GATE finish-phase compression ON THE FISHER read (still sub-threshold ⇒ do NOT turn weight_entropy
  on yet; re-check at the c2 terminal/finish where d_seg is nearer the floor). This is the measured answer to the
  rate-forcing soft-spot.
- **subpix_boundary** is ≈half the seg force and the MOST Fisher-antagonistic term — flag for the boundary A/B
  (its byte-free d_seg value must beat its antagonism cost).
- **margin_satisfice** subdominant, with a standing Euclid/Fisher SIGN FLIP (the exact both-metrics discipline
  case — neither cosine alone is the verdict).
- **Fisher noise floor MEASURED:** ep725≡ep726 are bit-identical checkpoints, so their |Δcos| ≤ 0.036 is the
  harness noise floor — any trajectory cosine change below that is not signal (a calibration the completeness
  table must respect).
This IS the completeness-table "demanded-by-measured-dynamics" column for the force terms. Standing per-run
instrument (memory dual-metric-readback) — every long run gets this table.

### §13.13 SOL-ultra TRUE-FINAL-FORM review fold (2026-07-17, rc=0) — the SPEC's "true-final" claim is DOWNGRADED pending 5 fixes
Source: `.omx/research/sol_ultra_v10_true_final_form_review_20260717.md` (first successful SOL-ultra run of the
session — validates the reaper/PTY fix; read-only, no launch/paid/pointer authority). Verdict:
**`NOT_TRUE_FINAL / NOT_LAUNCH_CERTIFYING / SPEC_AND_COMPILER_REWRITE_REQUIRED`**. The strategic direction
(task-space witnesses · generator-first rate · explicit boundary geometry · train-least · typed gates · exact
evaluator closure) is AFFIRMED as a paradigm; four premises are unclosed. **None reject a paradigm — all are
FORMULATION / IMPLEMENTATION / CUSTODY scope.** v10 is UNLAUNCHED ⇒ no live risk; these are pre-launch owed.

Five launch-blocking fixes, each a routed task (consumption-gated per the disposition):
1. **[#528 launch-safety] `w_pose` SQUARES the contest marginal under `--score-domain-loss`.** ADJUDICATED vs
   Arm B's built `PoseMarginalWeightLaw` (§13.9): **SOL is mathematically correct.** The inner loss already
   carries `pose_term=√(10·pose_l)` (`levelset_micro_batch_loss.py:326-329`); multiplying by
   `w_pose=5/√(10·d_pose)` gives `dL/d(pose_l) ≈ 2.5/d_pose`, not the contest marginal `5/√(10·d_pose)` (which
   is what you multiply RAW `d_pose` by). Arm B's law is admissible ONLY when the pose term is raw `d_pose`
   (weight-domain loss); under score-domain loss `w_pose=1` IS the exact objective. Arm B's round-1 tests
   fake-green'd it (finite-diff of the STANDALONE score derivative + string-search; never the composed loss).
   FIX: compiler/launcher REFUSE `PoseMarginalWeightLaw AND score_domain_loss`; add a composed-gradient
   regression test. verdict_scope=COMPOSITION.
2. **[#529] The v10 "compiler" is a presence-checker, not a compiler** — `hasattr`/`Path.exists` probes fake-pass
   on empty/stale/adverse/wrong-hash/wrong-axis artifacts; `compile_v10_capstone_launch_config` has NO success
   path (unconditionally raises `post_gate_fold_owed`), contradicting SPEC:311-324. FIX: emit typed
   `WitnessProgram`, resolve every LawRef, compile via the canonical constants compiler, parse with the REAL
   trainer parser, return `(argv, manifest, config_hash)`; every receipt validator REOPENS bytes +
   SHA/schema/provenance/axis/verdict/coverage/producer-consumer identity (= #332 bijection applied to v10).
   verdict_scope=COMPILER IMPLEMENTATION.
3. **[#530] Fresh-init cannot use fork/resume-only birth laws** — SPEC:24-29 says cold init + "never weights",
   yet P2/P5 depend on `ForkHeadSolve` + `ForkEmaClearance` which REQUIRE `--resume-from` (#518 memo). FIX:
   split `InitHeadSolve` (cold seeds + fresh head) from the fork-exclusive laws. verdict_scope=VEHICLE COMPOSITION.
4. **[#531 HIGHEST-IMPACT delta] `T` must be an explicit class-/cell-conditioned quotient residual.** MEASURED:
   flat realization d_seg=**0.0416** vs textured **0.0048** (8.7×) ⇒ sufficient statistic is at least
   `W=(G,ξ,T)`, NOT `G` alone (the generic flat-cell/GEOM-ONLY theorem is FALSIFIED at generic-flat-cell scope;
   nonlocal VJPs move remote decisions). v10 leaves `T` implicit in an undefined counted trunk ⇒ pays TWICE for
   "solved" geometry. FIX: unique-custody `T`, train ONLY the quotient residual after deterministic
   `G,ξ,seed,solve,projection` — this IS train-least/Kolmogorov made concrete. verdict_scope=FORMULATION.
5. **[#532] Two MEASURED probes (real n600 cache, no scorer forward):** (a) the "exact range(A)" consumer is
   exact ONLY over reals — real `P_A x` Δ=1.71e-13 but **valid uint8 `clip(round)` Δ=62.74** (0.059% out-of-gamut)
   ⇒ the ker(A)/gauge carrier (p0_null_subspace) needs a lattice-feasible realization proof + scorer-equality
   receipt before the n600 gauge sign-confirm; the fp64 projector theorem is INTACT, only "realized exactness"
   falsified. (b) `identify_static_hood_class` returns class 4 (hood, correct) but CANNOT be the sky detector,
   and its `frac_of_frame` (139.4, 3.51, …) is mis-scaled by frame-count ⇒ structured-init readiness is
   fake-green; fix normalization + build a distinct sky/static-top detector. verdict_scope=IMPLEMENTATION/CUSTODY.

Recursive-round bonus (independent CONFIRMATION of this session's own corrections): SOL round 6 FALSIFIED the
post-ep725 "Force-3 + clean pose-crest" reading — a no-Force-3 control has the SAME cold-Muon fingerprint, and
later σ_min telemetry oscillates/rebounds. This matches the session's F1-confound → cold-Muon attribution and
the crest-is-oscillation correction — two independent adversarial arms converging is the strongest evidence we
have that those reads are right.

## §14 SESSION FOLD 2026-07-18 — naming lock · curriculum-ORDER law · v9c2 disposition · organ-B localization · decision framework

**Operator SSoT directive stands: all of this lives in THIS doc.** Pointer `0.19108` UNMOVED — design only.

### §14.1 VEHICLE NAMING LOCK (operator 2026-07-18 — extends the naming charter)
- **v9c2** = the live warm-start run (`levelset_n600_witness_20260717T113932Z`).
- **v9c3** = a restart from v9c2's ep725 best WITH corrected events (#270 warm-Muon + #518 resume-warmup
  geometry). STILL the warm-start lineage — NOT v10. This is the cheap de-risk of the resume-event confound.
- **v10** = the FINAL capstone vehicle, RESERVED. **Its first run is the from-scratch cold-start** (this doc).
  No warm-start ever wears the v10 name.

### §14.2 CURRICULUM-ORDER LAW: island-birth BEFORE the phase stack (operator 2026-07-18)
The v10 stage order is **cold-start → seed (P1) → trunk conditioning → ISLAND-BIRTH (lane curve-prior #291 +
movable dilation-GO #323, per-class-λ homotopy #300/#323) → PHASE-STACK flicker-conditioning (#424/#425 +
illumination-cone two-transport) → pose joint finish (P6) → terminal SOLVE on realization-locked pockets.**
DERIVED law: **you cannot condition the flicker of an island that is not yet born** — the phase carrier aimed at a
stratum needs that stratum to exist. Born first, then match its GT flicker. This makes the flicker floor
(0.005318) engineerable, not hard (it binds only a smoother-than-GT witness). Extends §2 (P5/P7) with the
explicit ORDER; the phase stack is a curriculum STAGE, not a bolt-on.

### §14.3 v9c2 disposition = DIAGNOSTIC + defensive bank (do NOT grind to ep1400)
MEASURED this session: v9c2 has a STRUCTURAL ceiling — Movable 3.4% is unborn BY DESIGN (mod32cap island-birth
OFF), Lane 22.5% was Muon-kicked at ep726 (the resume-event confound: absolute-epoch Muon fired 75ep after a
cold-AdamW resume). **EMA-masking finding:** the shipped EMA reads 0.0039 but LIVE weights are 0.0048 and NOT
recovering — the EMA is flattered by its ~333-ep window still remembering the pre-kick ep700-725 state
(`ema−live` widened −0.00026@725 → −0.00089@900). Grinding to ep1400 is negative-EV (EMA drifts UP as it forgets
ep725). ACTION: snapshot v9c2's best EMA as a lossless defensive bank; v9c2 is the diagnostic that FEEDS v10,
not the optimal vehicle. Law: `[[warm-start-resume-must-adapt-events-to-resume-epoch-and-geometry]]`.

### §14.4 Organ-B realization localization (MEASURED, merged to main — the phase-stack aiming input)
The factorized costate organ (A+B+C) landed on main (real range(A) tap-table max-abs 2.5e-14; ker(A) zero-marginal
a tested theorem). Module B on the live EMA ep900: **sub-LSB fraction 0.3617 → regime MIXED** (terminal-solve not
yet forced; ~64% amplitude-open on the EMA — but re-read on LIVE per §14.3). Per-stratum: `Road→Lane` is
amplitude-OPEN (trainable — the phase-stack target), `Lane→Road`/`Undrivable→Road` are realization-locked pockets
(→ terminal SOLVE). ker(A) 22.6% of residual energy is FREE unexploited carrier (P3/#520). B is the localizer that
AIMS §14.2's phase stack. Equations `witness_realization_lsb_regime_v1` + `factorized_duty_marginal_projected_v1`.

### §14.5 DECISION FRAMEWORK (how to proceed — not a foregone v10 launch)
Order, each gated: **(a)** snapshot v9c2 best (defensive bank) → **(b)** cheap de-risking measurements: the
DECISIVE gate is the **$0 phase-stack-efficacy probe** (fire the phase carrier at B's `Road→Lane` strata on the
v9c2 EMA, through-R n600, measure Δd_seg — does conditioning the flicker actually move d_seg?); optional v9c3
(ep725-restart-fixed-events, tests resume-event recovery); island-birth movable-birth cost → **(c)** complete v10
design (#529 compiler real-success + #530 Init≠Fork + #531 T-quotient-residual + the resume-event self-protect
gate) → **(d)** fire v10 cold-start ONLY when (b) confirms the levers pay AND (c) is complete AND compute is free.
If the phase-stack probe comes back FLAT, that is decisive too — redesign §14.2 before the expensive v10 commit.
This is MVP-first: measure v10's assumed levers on the cheap substrate before the multi-day cold-start.

### §14.6 CONFIRMED CORRECTIONS (2026-07-18 — fold as we confirm, operator SoT discipline)
- **Phase-stack efficacy (the §14.5(b) gate) RETURNED — but MIS-TESTED (operator caught).** The probe applied a
  PER-PIXEL-INDEPENDENT min-norm post-hoc displacement (neither #424 conditioning-in-loop nor the real #425
  coherent codec). Its "+56.6%→+161% coupling wall" is SUBSTANTIALLY a strawman artifact of the wrong
  (per-pixel-independent) formulation. VALID: the diagnosis (96.7% Road→Lane flips ARE phase errors on the GT
  band; stratum amplitude-open). RETRACTED: any efficacy verdict — conditioning + real coherent codec UNTESTED.
  **§14.2 phase-stage is train-side/constrained-solve ONLY; the #425 store-and-apply RATE plan is RULED OUT**
  (post-hoc-stored corrections dead; only JOINT descent or constrained SOLVE crosses — pose L68 + phase, one law).
  Cure = FiLM-conditioned joint-trained sidecar (Quantizr-pose-analog). Memo `phase_stack_efficacy_probe_v10_gate_20260718.md`.
- **`_dev`/`_prod` maturity axis** (orthogonal to vehicle): dev iterates freely + is non-pointer-promotable by a
  structural guard; prod exact rows only move the frontier. v9c3=dev signal-harvest, v10=prod capstone. (Merged
  to main; helper `tac.checkpoint_maturity`.)

### §14.7 CHANNEL/HYPERPLANE-NATIVE WITNESS (design direction — operator 2026-07-18, empirically gated on the running arm)
Operator Q "should we bake Channels or Hyperplane into the witness?" → **BOTH, two layers of one object**, with
the render-legality constraint (the contest scores a DECODED IMAGE through the real frozen scorer, so the witness
must still EMIT a legal render — bake into internal geometry + seed + generator, NEVER skip the scorer):
- **CHANNELS → the witness HEAD + FiLM conditioning + LOSS.** Trainable/control coordinates = the scorer's
  per-class-weight channel basis (move along `W_c−W_c′`); loss = hyperplane-margin. Train + steer in the scorer's
  OWN decision coordinates ⇒ conditioning is collateral-free (FiLM = channel op, cannot per-pixel-overshoot).
- **HYPERPLANE → the witness REPRESENTATION + SEED.** The argmax partition IS a rank-4 hyperplane arrangement =
  tropical/Laguerre power diagram (#284/#311). Make that the witness's NATIVE representation (its cells = the
  scorer's cells); the SEED = the cell-membership/partition (video-derived, COUNTED); the render = the free
  generator decode (rule-118). This IS the non-RGB task-space thesis: spend bytes on the partition, not the pixels.
- **Falls out of {task-space level-set witness × channels × upstream modules}** (operator 2026-07-18): the witness
  is the scorer's argmax partition re-expressed in the channels the frozen modules define, counting only the seed.
- **STATUS: design direction, NOT yet MEASURED-to-pay.** Empirical gate = the collateral-coupling/FiLM channel-space
  arm (is channel/hyperplane control collateral-free? how much Road→Lane d_seg is reachable coherently?). Fold the
  MEASURED verdict here on the arm's return. Memories `[[solve-the-right-problem-is-right-coordinates-at-every-level-kolmogorov]]`
  + `[[post-hoc-stored-corrections-dead-joint-descent-or-constrained-solve-required]]`. Sisters #531 (T=quotient
  residual), #524 (16-ch stride-2 skip), #503 (recursive-fractal-optimal), #500 (Fisher metric).
