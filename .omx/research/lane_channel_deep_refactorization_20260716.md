# Lane-channel deep re-factorization — the composed gain chain vs the degraded-marking dynamics (2026-07-16)

**Operator P0 (verbatim):** *"Remember the additional weight on the lane channel too we discovered in
modules.py might be interesting to see flatten and factorize again with deeper analysis against lane and
road lane dynamics."* + the same-day Morse-Smale capstone: *"All of this is directly related to Morse
smale and brief births and deaths and completeness as well of all necessary terms in the system of
equations."* This is the analytic companion to the visual atlas for the Road-Lane 66% flicker bucket
(`c2_witness_own_decomp_20260716.md` §4 rank-1).

**Authority / axis:** `[macOS-CPU advisory]` — frozen CPU-torch fp32 SegNet (real weights,
sha256 `68956e32…6991b6`), exact upstream preprocess, bit-exact cached GT argmax+margins
(`gt_n96.npz` / `gt_n600.npz`), witness flips = the frozen mod32cap ep650 packbits masks from
`c2_witness_own_decomp_20260716` (no re-render). GT birth/death matching uses the EXACT T1 machinery
(`tac.boundary_math.phase_primitives` + the ground homography + `gt_poses` calibration).
`research_only=true; score_claim=false; promotable=false`. **Pointer 0.19108 UNMOVED — MEANS.**

**Artifacts (all MEASURED numbers below):** `experiments/results/lane_channel_refactor_20260716/`
— s1_gain_chain.json `e7680f9f…`, s2_margins_flicker.json `75db5c08…`, s3_dash_geometry.json
`241f17ea…`, s4_events_t1_audit.json `df0de020…` + the four generating scripts (deterministic from the
frozen weights + caches). Scopes per section: s1 = 4 frames of n96, 5 boundary px/pair/frame; s2/s3/s4 =
full n600 (T1 audit = every 10th pair, 60 pairs).

## 0. ALREADY KNOWN (net of this)

Head rank-4 linear, Lane pair normals LARGEST 3.75–4.01, feature-space flip d=|m|/‖Δw‖, Lane 77%
skip-limited (fractal memo + eq `segnet_head_rank4_linear_flipdist_v1`); Road-Lane = 66% of the witness
residual, FLICKER character, flat-amplitude exhaustion (c2 witness-own decomp); L65 error∝1/persistence;
L85 GT sub-pixel advection phase; L71/L73 openpilot lane priors; #287 dash comb.

## 1. NEW №1 — the composed gain chain INVERTS the head amplifier: Road-Lane is decision-amplified but CONTROL-ATTENUATED (MEASURED)

Composed input→pair-margin gain `G_cc′ = ‖∂(z_c−z_c′)/∂input‖₂` through the FULL frozen net at real GT
boundary pixels (s1, 4 frames × 5 px/pair):

| pair (at its boundary px) | G med | G skip-detached | skip ratio | luma share | \|m\| med | input flip-dist med |
|---|---:|---:|---:|---:|---:|---:|
| **Road-Lane** | **0.0212 (LOWEST)** | 0.0174 | **1.344 (LARGEST)** | 90.4% | 0.324 | **16.0 (LARGEST)** |
| Road-Undrivable | 0.0404 | 0.0387 | 1.052 | 89.1% | 0.417 | 8.8 |
| Road-Movable | 0.0471 | 0.0444 | 1.082 | 89.8% | 0.141 | 3.4 |
| Road-MyCar | 0.0537 | 0.0523 | 1.027 | 89.8% | 0.410 | 8.7 |
| Undrivable-Movable | 0.0415 | 0.0403 | 1.044 | 85.8% | 0.212 | 5.9 |
| Lane-Movable / Lane-MyCar / Lane-Undriv (n=8/5/2) | 0.058–0.101 | — | 1.03–1.30 | 87–91% | 1.0–1.5 | 15–21 |

- **The head-norm amplifier (Lane rows largest) does NOT survive composition**: in pixel space the
  Road-Lane margin has the LOWEST gain — 1.9–2.5× below every other pair. The frozen net is a
  **high-decision-gain head reading a LOW-pixel-gain lane-evidence channel** (comma10k pressure: faint
  paint ⇒ weak pixel coupling, amplified at the decision). The "additional weight on the lane channel"
  is real at the head; the chain inverts it at the input.
- **Consequence (the sharp new law): Road-Lane is flicker-prone AND amplitude-cure-resistant.**
  Feature-space noise flips Lane cheapest (d_feat = |m|/‖Δw‖, Lane norms largest — the fractal law);
  pixel-space DELIBERATE amplitude correction is hardest (largest input flip-dist 16.0 L2 units,
  optimally aimed). The only efficient control coordinate is **sub-pixel geometry/phase** (placement
  moves features without fighting the attenuated amplitude channel). This DERIVES the c2 measured
  flat-amplitude exhaustion (§3 of the decomp) from the frozen weights.
- **The stride-2 skip is the Road-Lane gain carrier**: detaching it drops RL gain 26% (ratio 1.344) vs
  3–10% for every other pair — the gradient-path confirmation of the 77% skip-ablation finding.
- **Luma share ~90%** of the input-gradient energy at ALL boundaries (BT.601 projection) — matches the
  c2 luma-cure-driver 0.85.

### 1b. Skip-channel factorization: there is NO private "lane channel" in the skip (MEASURED)

Per-channel |grad| profile of the RL margin over the 16 stride-2-skip channels: top-5 = ch {10, 1, 11,
14, 9} = 55% of mass — but the Road-Undrivable profile is near-IDENTICAL (same channels, same order).
Per-channel ablation (channel→its mean, 4 frames): max-damage channels ch7 (975 flips/634 RL), ch1, ch5,
ch6; RL share of ablation flips ≈ 55–74% in EVERY channel ≈ the RL boundary-length share. **The lane
bottleneck is the WHOLE 16-ch skip, not a dedicated channel** — generic boundary-placement channels
serve all pairs; Lane is just the pair that depends on placement most. One notable structure: **ch10 —
the single largest RL-grad channel (0.148) — is CHROMA-tuned** (its effective input filter is 99.6%
chroma-plane; ch1/11/14 are 95–97% luma; footprints r≈4.5–5.4 px). The chroma d_seg lever has a named
carrier at the skip.

## 2. NEW №2 — the margin × length × churn composition of the 66% bucket (MEASURED, n600)

Operator thesis test ("faint paint ⇒ small margins along the whole lane boundary"):

- **Per-pixel Road-Lane margins are NOT uniquely low** — boundary median 0.365 = Road-Undrivable 0.365,
  ABOVE Road-Movable 0.212 / Undrivable-Movable 0.185. REFINED, not refuted: the degradation shows up as
  (i) **LENGTH — Road-Lane = 50.3% of ALL boundary samples** (1.63M of 3.24M; the ragged fragmented
  boundary is long), (ii) a **fat low tail** (37% of RL boundary below 0.25, p10 = 0.063), (iii)
  **island-level near-thresholdness** (§3 — the persistence story, where "faint paint" actually lives).
- **Witness flips live in each pair's low-margin tail**: RL flip-pixel margin median 0.212 vs boundary
  0.365; 57.6% of RL flips below 0.25 (vs 37% of the boundary). Composition CONFIRMED at the tail level.
- **Bucket-share prediction** (static: length × P(|m|<0.25)): predicts RL 46.9% vs observed flip share
  57.0% (Road-Undriv 18.4→13.5, Road-MyCar 16.5→8.4). The static predictor under-predicts RL by ~10 pts
  and over-predicts the STATIC hood — **the excess is exactly the temporal churn** (§4: the birth/death
  events are Lane-specific; the hood boundary is frozen). Length × margin-tail × event-churn is the
  complete decomposition; per-pixel "high gain" is NOT a factor (§1 measured the opposite).

## 3. NEW №3 — the dash islands ARE the low-persistence Morse-Smale stratum; measured geometry of the degraded paint (MEASURED, n600)

Lane connected components (8-conn, area≥3): **20.6 islands/frame** (+3.3 sub-3px specks), area median
12 px (p90 194), aspect median 6.7, orientation broad/oblique (major axis median ≈ 18° from horizontal,
p10 ≈ 38° — the operator's "tilted irregular ellipses", now with numbers; feeds the §6-fractal
axis-aligned-filter-gap story), isoperimetric raggedness p90 6.9 (large islands are 7× rougher than a
disk; small-island discrete-perimeter bias makes the median uninformative).

**Persistence per island** (max GT margin inside the island = prominence above the argmax threshold =
saddle-node distance):

| class | islands | pers median | frac < 0.5 | frac < 1.0 | frac < 2.0 |
|---|---:|---:|---:|---:|---:|
| **Lane** | 14,323 | **0.625** | **0.42** | **0.72** | 0.95 |
| Movable | 2,197 | 2.63 | 0.12 | 0.19 | 0.35 |
| MyCar | 600 | 9.13 | 0.00 | 0.00 | 0.00 |

The Lane dash layer sits ON the argmax threshold — 4.2× lower median prominence than Movable, 15× lower
than the hood. **This is L65 (error ∝ 1/persistence) instantiated with this video's histogram, and the
operator's "brief births and deaths" given its mechanism: near-zero-persistence islands are saddle-node
events waiting on sub-pixel appearance changes.**

## 4. NEW №4 — GT birth/death event rates: the dash layer CHURNS ~50% per scored step (MEASURED, n600, T1 cadence)

ξ-advected island matching (EXACT T1 machinery: `cross_scored_frame_xi_interp` + ground-homography
advection; interior islands area≥3, 6px border excluded; birth/death = <5% overlap):

- **9.43 births + 9.50 deaths per frame-step on 19.1 interior islands/frame** — about HALF the dash
  layer is involved in an event every scored step. Birth persistence median 0.647 ≈ persisting islands'
  0.765: births are not anomalies, the whole layer hovers at threshold (median event area 7 px).
- **Caveat (stated):** <5%-overlap matching on median-7px islands with the interp gap-screw inflates
  event counts by advection mismatch — treat 9.4/step as an upper bound of genuine events at this
  tolerance. Independent floor: the T1 audit (below) finds **354 lane-adjacent straddle px/frame that
  are >3px from ANY advected reference** — genuine new lane structure at a substantial rate however you
  match.
- **Kolmogorov/world-frame reading:** lstars = SegNet(real video) — GT births are the SCORER's own
  saddle-node crossings on world-STATIC paint under ego-motion + sub-pixel resampling. The generative
  sufficient statistic is a **ground-frame anchor field + ξ**, not per-frame events (§5 rate estimate).

### 4b. T1 phase-advection formulation verdict (MEASURED on the exact trainer construction, 60 pairs)

T1's weight = `annulus(p) ∧ ground(p) ∧ advected_active(p−1)` (trainer L7584-7654; ~1,368 weight
px/frame):

1. **T1 does NOT fight GT births — it is birth-SILENT.** At a birth site there is no advected reference
   ⇒ weight = 0 by construction. The "pure advection penalty penalizes correct births" concern is
   **REFUTED at the weight level** (verdict_scope: THIS formulation — gt_advected ref mode, band 2.0).
2. **Death contamination is small: 5.1% of weight pixels** (70 px/frame) sit >3px from any genuine
   current-pair straddle — a stale 1-step pull toward dead boundaries, bounded by pa_w × 0.051 share.
   Not the missing term.
3. **The REAL gap is COVERAGE, not mis-force: 26.3% of candidate straddle pixels (601/frame; 354
   lane-adjacent) receive NO T1 supervision** (genuine straddle, no advected reference — births +
   fast-moved structure). And **the c2 config emits T1 (`--seg-phase-advect-weight 0.4` @ ep700) but
   does NOT emit the Force-3 subpix term** — so in the c2 phase stage, the churning lane subset (the
   exact 66%-bucket target) is **phase-unsupervised at birth sites**.

**⇒ OWED c2 AMENDMENT (LOUD — the dry-start is in flight; phase stage engages at ep700; there is
time):** *event-fallback phase weight* — in the T1 provider:
`t_ref := where(ref_active, advected_prev_tie, own_gt_tie)`;
`weight := ann ∧ ground ∧ (ref_active ∨ a_p)`.
Both primitives already exist in `phase_primitives` (`gt_tie_targets_numpy` supplies the own-tie
fallback); this is "advect-where-persistent, target-where-born" implemented as a fallback, NOT as
gating advection off. Equivalent alternative: also emit Force-3 subpix in the c2 phase stage.
θ-independent target, per-pair-local, zero batching change — same containment as T1 itself. Lands as a
DSL lever amendment (`--seg-phase-advect-ref gt_advected_with_own_tie_fallback` or a subpix
co-emission in `spec_c2_surgical`), never a hand-added flag.

## 5. Comb vs anchors verdict + the rate side (MEASURED geometry, DERIVED bytes)

- **Ground-plane spacing** (EON K, h=1.22m, pitch −0.01; island centroids clustered into lane lines,
  gaps in meters): median 7.7m, **CV 0.80**; within-line constant-period residual **median 35%, p90
  58%** (76 sequences, 156 gaps). **The periodic dash comb (#287) is REFUTED as the generator for THIS
  video** (verdict_scope: formulation — constant-period comb per lane line on ground-plane centroid
  spacing). The measured spacing sequence is the operator's "inconsistently spaced", quantified.
- **Byte estimates (DERIVED, assumption-labelled):**
  - comb phase: ~1–2 KB but mismodels 35% of gaps — invalid.
  - naive per-frame anchors: 20.6 islands × ~3.3 B × 600 ≈ 41 KB — dominated.
  - per-event stream: ~5.6k events × 2–3 B ≈ 12–17 KB — dominated.
  - **world-frame static anchor field + ξ-transport (the Kolmogorov answer): ≈ 220 dash cycles/line
    (60 s × ~25–30 m/s ÷ 7.7 m) × 2–4 lines × ~2 B (Δs entropy-coded + extent/tilt/persistence-class)
    ≈ 0.9–1.8 KB TOTAL** + spline/registration overhead. The generator (ξ ground-warp + rasterize +
    per-anchor visibility/persistence threshold reproducing the scorer's saddle-node births) is FREE
    rule-118 code. Speed 25–30 m/s is ASSUMED (not measured here); anchors-fold into the L71 analytic
    lane band + #425 ξ-residual codec design, replacing any comb-phase section.
- **Priority honesty:** these are RATE-side design inputs for the c2/anchor carriers; per the c2 decomp,
  the trained-witness residual is cured by TRAINING-side phase levers first; anchors matter for the
  analytic/necessity vehicle and the eventual archive shape.

## 6. Control authority + the per-pair λ scaling law (MEASURED → DERIVED)

- **Measured control authority** (Δmargin per unit L2 of a coherent lane-side 5×5 BT.601-luma bump at
  RL boundary pixels): **0.0029** — comparable to other pairs (0.0018–0.0081), NOT elevated. Flipping a
  median RL pixel by local amplitude alone needs ≈ 0.324/0.0029 ≈ 112 L2 units ≈ 22 uint8 levels/px
  over 5×5 — LOCAL FLAT BRIGHTEN IS A WEAK ACTUATOR (derives the c2 flat-amplitude exhaustion +
  non-local cure r36 0.31–0.72 from the frozen chain). The Lane-brighten flat-coherent finding survives
  ONLY as a training-time prior direction, as the c2 memo already scoped.
- **Per-pair λ law (DERIVED, recorded for the phase stack):** per unit loss-gradient, the render moves
  the RL margin with gain G_RL ≈ half the other pairs' — the phase stage's effective step size on
  Road-Lane is ~2× SMALLER per unit pressure, not larger. First-order equalization of margin-space
  progress ⇒ **λ_cc′ ∝ 1/G_cc′: λ_RL ≈ 1.9–2.5× the other pairs'** (G_med ratios from §1). The
  measured per-class λ (#433 flip-temp) is the empirical class-level compensation; this is its
  pair-level derived prior. Directionally consistent with σ_cc′ Young's σ[Road-Lane] = 0.377 (LESS
  length-penalty erosion on lane perimeter): both push protection/pressure toward Lane. NOT wired
  anywhere yet — recorded as the derivation the next per-pair-λ sweep should seed from (duty-to-measure).

## 7. COMPLETENESS TABLE — the system of equations vs the measured dynamics (energy-vs-forces)

| term | status | formulation (as built) | measured demand (this memo) | verdict |
|---|---|---|---|---|
| σ_cc′ length term (#382, Young's σ[RL]=0.377) | PRESENT | static per-pair surface tension on {m=0} | anti-erosion on the longest boundary (50.3%) | MATCHED for its role (static) |
| birth-completion event (v7.5 Lever-2) + rare-class birth stack | PRESENT | per-CLASS latched persistence/area hand-off (recall→precision) | per-ISLAND births at 9.4/step | **FORMULATION MISMATCH** — handles "did Lane nucleate", not per-dash events; not a per-island force |
| critical-nucleus guard (#315/#302) | PRESENT | curriculum hand-off admissibility per class | same | MATCHED for its role (curriculum, not dynamics) |
| persistence/topology loss (#218, clDice) | PRESENT | per-frame soft-skeleton recall on tail classes | per-frame thin-structure existence | PARTIAL — a static per-frame birth force, no event timing; whether it suffices at the measured churn = OPEN (unmeasured) |
| T1 phase-advection (#424) | PRESENT | transport of persisting phase; birth-SILENT; death-stale 5.1% | transport (matched) + supervise the churning 26.3% (NOT covered) | **PRESENT-BUT-INCOMPLETE — the event-fallback weight (§4b) is the missing completion** |
| Force-3 subpix (within-pair own-tie) | BUILT, **not emitted by c2** | per-pair GT tie target | exactly the birth-site supervision T1 lacks | **CONFIG GAP in c2 (§4b)** |
| per-class λ (#433) | PRESENT | empirical class-level flip-temp weights | pair-level gain equalization λ∝1/G (§6) | PARTIAL — derived pair-level prior recorded, unswept |
| per-dash anchor generators (rate side) | **MISSING** | — | world-frame anchor field ≈ 0.9–1.8 KB (§5); comb REFUTED | OWED archive-shape item (folds into L71/#425); not a training force |

**Net completeness verdict:** the system is complete for TRANSPORT and CLASS-level birth; the measured
dynamics demand two additions — (i) the **event-fallback phase weight** (training force; c2-routable
before ep700), (ii) the **world-frame per-dash anchor field** (rate carrier; replaces the comb). The
"persistence floor per island" force is NOT added as a row of its own: at 42%-below-0.5 GT prominence,
matching GT means matching its births/deaths (the event-fallback does this through the tie target), not
holding islands artificially persistent — a hold force would FIGHT GT exactly as feared of T1.

## 8. Round-1 adversarial review (own attack) + verdict scopes

- **s1 sample sizes are small** (20 px/pair for the 5 major pairs; 2–8 for rare Lane-X pairs): medians
  stable across frames for the majors; Lane-Undrivable (n=2) is indicative only. The RL-lowest-gain
  finding is a 1.9–2.5× median gap — robust at n=20; a fuller sweep is cheap if a lever consumes the
  exact ratio.
- **Gain is measured at (384,512) scorer-input space**, not through R (bicubic↑→uint8→bilinear↓); the
  through-R composition attenuates all pairs together but could reorder — UNMEASURED; the §1 law is
  about the frozen scorer chain proper.
- **Flip attribution in s2** = nearest-different-GT-neighbor (r≤3), not the decomp's stratum machinery:
  RL share 57.0% here vs 66.0% pair-side there — methodology difference (their edge+near strata double
  over both sides), directions agree; 7,192 interior flips (2%) unattributed.
- **Event rates are upper bounds** at the <5%-overlap/interp-gap tolerance (§4 caveat); the 354
  lane-adjacent no-reference px/frame is the tolerance-independent floor.
- **Ground-plane spacing** assumes flat road + pitch −0.01 + the pinned K; slope/curvature bends the
  meter scale but not the CV-0.8/35%-residual conclusion (relative measures).
- **NOT measured:** through-R gain re-ordering; blink-back fraction of births (re-births of the same
  world anchor); whether clDice at its current weight already supplies enough birth force (the §7 OPEN);
  the event-fallback's actual Δd_seg (it is a spec, duty-to-measure at the c2 per-stage A/B); vehicle
  speed (assumed 25–30 m/s for §5 bytes).
- Naive negatives here (comb refutation, flat-brighten weakness) are FORMULATION-scoped as labelled;
  neither kills its family (irregular combs / trained band profiles remain open).

## 9. Triality + routing

- **DAG:** FEED-lane-gain appended to `sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`.
- **equations:** `lane_gain_chain_composed_v1` REGISTERED
  (`tac.canonical_equations.lane_gain_chain_composed_20260716`) — sibling of
  `segnet_head_rank4_linear_flipdist_v1`: head-amplified ∧ chain-attenuated ⇒ phase-only efficient
  control; anchors = the §1 gain table + the §3 persistence histogram + the §4 event rates.
- **DSL:** the event-fallback amendment recorded OWED (§4b) — routes to `spec_c2_surgical_20260716`
  (as a `--seg-phase-advect-ref` mode or Force-3 co-emission) BEFORE the ep700 engage; the per-pair λ
  prior (§6) seeds the next λ sweep; the anchor-field carrier folds into the L71/#425 archive shape.
  Nothing hand-wired here.
- **memory:** measured-verdict addendum appended to
  `degraded_lane_markings_x_lane_head_gain_flicker_mechanism_20260716.md` (the topic file; no new
  MEMORY.md line — the law lives in the equation + topic file).

## STORES CONSULTED

CLAUDE.md; AGENTS.md; operating manual; segnet_recursive_fractal_factorization_20260715.md;
frozen_scorer_exact_factorization_20260715.md; c2_witness_own_decomp_20260716.md;
degraded_lane_markings_x_lane_head_gain_flicker_mechanism_20260716.md (incl. the Morse-Smale extension);
phase_primitives.py + trainer T1 block (read, not edited); spec_c2_surgical_20260716.py;
FEED-sigma-ccprime (#382); birth_completion.py; graph-memory recall ("lane channel weight skip
stride-2"); L65/L67/L68/L71/L73/L80/L85/L86; dash_comb_probe artifacts; gt_n96/gt_n600 caches.

**Pointer 0.19108 UNMOVED — MEANS.**
