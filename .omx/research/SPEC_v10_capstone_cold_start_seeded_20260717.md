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
tie-locus → row-7; curvelet → P7). Pointer 0.19108 UNMOVED (means/apparatus — SPEC only).
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
