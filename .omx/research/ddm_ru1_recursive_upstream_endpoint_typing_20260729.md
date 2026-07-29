---
council_tier: T3
council_attendees: [Schmidhuber-LEAD, Shannon, Dykstra, Rudin, Daubechies, Yousfi, Fridrich, Contrarian, Assumption-Adversary, NumericalAnalyst-consultee, MicrolocalAnalyst-consultee, Statistician-consultee, Tao, Mallat]
council_quorum_met: true
council_verdict: PROCEED
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
council_dissent:
  - member: Contrarian
    verbatim: "the -0.138 dS 'GN band' projection assumes the joint solve reaches every m_def<0.25 flip without net collateral; the measured evidence is 18 hotspot cells of best-of-8 single edits — treat -0.046 (strict one-quantum band) as the defensible number until pb1's P2 lands a full-population row."
  - member: Assumption-Adversary
    verbatim: "shared assumption: 'the token lattice is the only actuation surface at this endpoint.' The renderer section (3,341 B) and selector are ALSO archive-counted editable surfaces; a rank-1 renderer-bias edit moves a class boundary GLOBALLY across all 600 pairs at ~0 marginal bytes. Untested. Queue as a probe before declaring the tier-2 band token-only."
council_assumption_adversary_verdict:
  - assumption: "token lattice = sole actuation surface"
    classification: CARGO-CULTED
    rationale: "inherited from the QDBS framing; renderer/selector sections never edit-probed at this endpoint"
  - assumption: "1.2731 B/err water price"
    classification: HARD-EARNED
    rationale: "exact score arithmetic: (100/(600*196608))/(25/37545489) = 1.2731"
council_decisions_recorded:
  - "op-routable 1: pb1 P2 aims solves at the typed cell targets (atlas_flat.npz) — top-100 cells, channel-sign directions, m_def<0.25 first"
  - "op-routable 2: pose terminal solve to <=0.018-class contribution is a HARD prerequisite for the 0.172 bar (0.127 line cannot cross at any rate)"
  - "op-routable 3: renderer-section rank-1 edit probe (Assumption-Adversary) before tier-2 is declared token-only"
related_deliberation_ids: [ddm_lv1_t3_reseal_20260728, box_retired_min_s_target_20260728]
---

# ddm_ru1 — recursive upstream typing of the 0.0038892 endpoint residual (atlas · typing · floor-as-price · convocation)

**Pointer honesty first: 0.1910828242 [contest-CPU] UNMOVED.** Everything below is
`[macOS-CPU advisory]`, `score_claim=false`, `promotion_eligible=false`. Operating point:
the t3 burn endpoint (`ddm_lv1_20260728/t3_long_burn_lotto_v2`, ckpt
`stage_seg_trunk_tau_final.npz` sha `33776302e4fa…`, config_hash `53ac33ce…`), byte-closed by
pb1 into archive `85d575bed157…` (768,689 B) and receiver-realized at d_seg **0.0038887702**
(trainer fp32 EMA full-confirm 0.0038892195 — realization gap 4e-7, nil). Operating S on the
deployed bytes = 28.86 (pose stub inert; pose is pb1's terminal solve, out of ru1 scope).

**Doctrine (operator 2026-07-29, binding): "nothing is truly unreachable."** The frozen scorer
is deterministic — every remaining flip is flippable by SOME input change; only PRICE varies.
This memo has NO unreachable bucket and NO hard floor: three price tiers
{free-solve · priced-in-budget · priced-above-water-with-named-trigger}; every row ends with a
price and a re-pricing trigger.

## §0 Instruments REUSED (recall-first; nothing rebuilt)

pb1 receiver (`tac.optimization.ddm_tr1_runtime.parse_archive` + `render_frame1_camera_uint8`)
· frozen CPU-torch SegNet (`tac.boundary_math.seg_core.load_real_segnet`) · trainer verdict
batch (`cpu_verdict_d_seg_argmax_batch`) · gt_n600 cache (lstars + GT margins, sha `cf8d8360…`)
· pb1 P1 chunks (per-pair dsegs + 24×32 cell flip maps) · rank-4 head law
(`segnet_head_rank4_linear_flipdist_v1`) · g4 stationarity frame (cell/stationarity axis) ·
#141 margin-saliency lineage (old-vehicle floor numbers re-DERIVED here, never transferred).
NEW artifacts (the previously-missing realized-side instrument): the per-flip realized-logit
atlas + the token-quantum currency calibration (tools below).

**Apparatus-validity positive control:** my per-pair d_seg over all 600 pairs is
**bit-identical to pb1's landed verdict (max |diff| = 0.0)**; total flips 458,738 exactly
equal. The atlas and pb1's P1 measure the same object.

Artifacts (SSD, durable): `/Volumes/VertigoDataTier/pact/ddm_ru1_20260729/`
`atlas_chunks/atlas_0000..0600.npz` · `atlas_flat.npz` (458,738 per-flip records:
pair,y,x,gt_class,realized_class,m_def,gap12,gt_margin,dist_bin,gt_flicker) ·
`atlas_analysis_receipt.json` · `token_quantum_calibration.json` (141 real single-token edits)
· `gt_flicker_receipt.json` · `pb1_chunk_mining_receipt.json` · `tier_crosstabs_receipt.json`.
Tools (this landing): `tools/ru1_endpoint_residual_atlas.py`,
`tools/ru1_token_quantum_calibration.py`.

## §1 DELIVERABLE 1 — the endpoint residual atlas (MEASURED, n600, realized-through-receiver)

The 458,738 remaining flips (d_seg 0.0038892) decompose:

**Geometry (dist-to-GT-boundary):** 93.9% ON a GT inter-class boundary pixel · 6.1% within
3 px · **0.06% (264 px total) interior/island**. Island/topology error is OVER at this
endpoint; the residual is pure codim-1 boundary-placement jitter. (Level-set frame confirmed
at the endpoint; the 565-of-985 erased GT Lane components live inside the boundary mass as
erasure-of-thin-structures, not as isolated islands.)

**Two-class structure:** in **98.8%** of flips the GT class is the realized runner-up
(m_def == gap12). Every flip is a TWO-class contest → rank-1 directional logit nudges suffice;
no multi-class reshuffle anywhere. (Aim statement for GN solves.)

**THE decisive histogram — logit deficit m_def = z[realized]−z[gt], in measured token-quanta.**
Currency (Tier-B, 141 REAL single-token ±1-level edits through the receiver + frozen SegNet,
6 pairs × 3 hotspot cells × 8 channel/sign arms): **κ = 0.0753 logits per token quantum**
(median of per-edit medians at in-cell flip pixels; IQR 0.044–0.110). One quantum = 1 of 16
levels = 2/15 of the renderer's [-1,1] token range.

| band (logits) | ≈ quanta | flip mass | cumulative | reading |
|---|---|---|---|---|
| m_def < 0.075 (κ) | ≤1 | **11.9%** | 11.9% | strict single-token single-quantum prize |
| m_def < 0.25 | ≤3.3 | **35.4%** | 35.4% | joint-GN band (4 ch × neighbor cells ≫ 3.3κ DOF) |
| 0.25 ≤ m_def < 1.0 | 3.3–13 | 47.8% | 83.1% | deep-solve band |
| m_def ≥ 1.0 | >13 | 16.9% | 100% | tail (64% flicker-targets, 65% Lane) |

Percentiles: p10 0.063 · p50 0.389 · p90 1.330 · p99 3.22 · max 12.97.

**Class-pair structure:** Lane→Road 176,733 (38.5%, median deficit 0.55 — the DEEPEST bulk
class: decisive erasure, matching Lane's largest head normals ‖Δw‖ 3.95–4.01) · Road→Lane
49,107 · Undriv→Road 42,275 · Road→MyCar 39,353 (GT margin median 0.105 — hood-rim fragility)
· Movable→Undriv 37,772. MyCar total only 2.4%.

**Spatial concentration (pb1 cell maps, n600):** 486/768 token cells have ZERO flips across
all 600 pairs; **top-100 cells hold 83.1%** of all flips; top-32 hold 41.2%; hotspots = rows
11–12 (horizon band) — cell (11,17) alone 12,910 flips. Consistent with g4 (top-1%/5%/10%
pixels = 21/69/90% of flip mass).

**Per-pair distribution is THIN-tailed:** top-60 pairs hold 13.4%, bottom-300 hold 44.1%
(range 0.00267–0.00678; worst block = contiguous scene pairs ~507–573). Pair selection does
NOT pay; **cell selection is the strategy** (the solve surface is ~100 cells, not 600 pairs).

**Single-edit response (Tier-B, MEASURED):** blind edits median net −1 flips (the endpoint is
a genuine local optimum vs undirected edits; 65% of arms net-negative — ERF ≫ cell ⇒ real
collateral); best-of-8 per cell **positive in 17/18 cells, up to +24 flips from ONE quantum**;
positive edits fix on average 7.5% of their cell's flips each. Byte price of a token edit:
brotli section delta −850…+2624 B per edit, centered near 0 (noise, ≤0.35% of section) —
**token edits are byte-neutral at first order**; a solved field's true Δbytes = one re-encode
measurement (owed by the consumer, not extrapolated here).

**Aim gain (quantified, deliverable-2 requirement):** aimed (best-of-8, atlas-targeted) edits
flip the sign of the yield distribution: median −1 blind → +3..+24 aimed at hotspot cells.
An unaimed solve wastes ~65% of its edit budget on net-negative moves; the typed table below
is what converts pb1's GN iterations from median-negative to positive-yield. Lower bound
already banked: Σ best single-edit fixes over the 18 sampled cells = +151 flips at ~0 B.

## §2 DELIVERABLE 2 — recursive upstream typing (every row: price + re-pricing trigger; NO walls)

Deepest-cheap-address per bucket, through the scorer's own factorization (head rank-4 → decoder
feature → input pixel through composite-R → token lattice = this vehicle's actuator):

| # | bucket | mass | deepest cheap address | measured price NOW | re-pricing trigger (parked-with-trigger, never closed) |
|---|---|---|---|---|---|
| 1 | m_def<κ, hotspot cells | 11.9% (54.6k flips, ΔS −0.046) | token lattice, single ±1 quantum, channel-sign per atlas | **~0 B** (byte-neutral edits); measured +yield in 17/18 cells | falsifier: pb1 P2 full-population solve fixes < the +151 single-edit lower bound |
| 2 | m_def<0.25 joint band | 35.4% cum (162.5k, ΔS −0.138 ceiling) | joint GN over 4 ch × ~4 neighbor cells (≥16 DOF vs 3.3κ deficit); 98.8% rank-1 two-class targets | ~0 B; collateral is the binding cost (median blind edit −1) → full-pair objective mandatory | Contrarian bound: book −0.046 until measured; trigger: P2 row on top-100 cells |
| 3 | 0.25≤m_def<1.0 deep band | 47.8% (219k) | multi-quantum token solve; declining yield/quantum; residual via pair-local corrections | tokens ~0 B + corrections AT water (1.2731 B/err exactly ⇒ marginal) | (a) finer quanta at hotspot cells (16→32 levels, cost = +1 bit × hot cells only); (b) clustered-correction coding (errors sit in 100 cells ⇒ joint sparse/colex coding prices B/err BELOW water); (c) renderer rank-1 edit probe (Assumption-Adversary) |
| 4 | tail m_def≥1.0 | 16.9% (77.4k); 65% Lane, 64% flicker | Lane thin-structure erasure — the stride-2-skip-limited pair (77% of skip-detail flips are Road-Lane); positional carrier needed, not amplitude | PRICED-ABOVE-CURRENT-WATER at this grid: 16px cells cannot place 1-2px dashes; corrections at water for 77k flips = 98KB ≈ break-even, not profitable | (a) lane-corridor parametric sub-carrier (g4's lane_corridor_wedge amortization row; ~8-dim/frame AR-coded = hundreds of B for the whole corridor); (b) per-cell code_width bump on the ~30 lane cells; (c) box re-derivation absorbs what stays |
| 5 | flicker∩fragile core (GT flickers AND GT margin<0.1) | 16.6% overall; 12.1% also deep (≥0.25) | per-pair token deltas (the vehicle IS per-pair — flicker is trackable, just unamortizable) | per-pair-specific precision, zero cross-pair sharing; overshoot risk is measured (two-sided fragile boundaries drive the −1 median) | (a) flicker-aware per-pair boundary-phase channel; (b) hysteresis-style asymmetric targets (aim mid-margin, not GT-tie); (c) leave-in-place → absorbed into the corrected box (§3) |

Consumers named per row: **pb1 P2/QDBS** (rows 1–3: `atlas_flat.npz` = the target list; cell,
y,x, channel-sign, m_def, two-class direction) · **r7 context structure** (rows 4–5: the
lane-corridor + flicker-phase carriers are CONTEXT design inputs) · **the composed candidate**
(§3 arithmetic = its budget sheet) · **organ co9** (κ, water, tier masses as SENSE-layer
constants w/ provenance to the receipts).

## §3 DELIVERABLE 3 — the floor question, re-specified as PRICE (verdict-scoped)

**Measured GT-side facts at THIS endpoint** (cache-only + atlas intersections; the old #141
bc20 numbers were NOT transferred): adjacent-pair GT argmax flicker = 1.246% of all pixels
(3.2× the whole residual; 45% Road / 28% Lane); GT near-tie mass <0.1 logits = 0.28% of all
pixels. AT the flips: 49.5% sit on flicker sites; 60.0% where GT margin < 0.25; 30.1% < 0.1.
GT margin median at boundary flips 0.176 (vs 0.150 on the old bc20 vehicle — same order,
re-derived).

**Verdict (scope: INSTANCE — this checkpoint, this archive, this scorer):** there is NO
abandoned mass. The former "floor-class" = bucket 5 (16.6%) + the tail's flicker share —
concrete argmax comparisons, each flippable at a MEASURED cost class (per-pair precision, no
amortization, overshoot-prone). Its price today ≈ water (1.27 B/err via corrections) or ~0 B
via per-pair token precision paid in solve effort + collateral risk; its re-pricing triggers
are named in rows 4–5.

**Corrected box arithmetic (vs the retired #613 box's 0.00116 corner):** the box memo already
showed 100×0.00116 + 0.127 banked pose = 0.243 > 0.172 at ZERO rate. Endpoint-corrected, with
tier masses (all-scenario table in `tier_crosstabs_receipt.json`):
- Pose line first: **at pose contribution 0.127, NO d_seg (even 0) crosses 0.172 at any rate
  ≥ 45KB — the pose terminal solve to ≤0.018-class is a HARD prerequisite.**
- At lv1-truncated rate (130KB tokens → rate 0.090) + pose 0.018: d_seg 0.00116 → S 0.224;
  **the bar needs d_seg ≈ 6e-4** (S 0.168). That is 3.35e-3 of residual to remove ≈ 395k of
  458.7k flips: ALL of tiers 1–2 (−1.38e-3, to 2.51e-3) + most of tier 3 + roughly half the
  tail. Equivalently: the correction band alone cannot buy it (395k × 1.27 B = 503KB ≫
  budget); **the bar is crossed by solve mass + the row-4/5 re-pricing triggers, or by more
  rate reduction (total ≤ ~110KB moves the seg box back up toward 1.0e-3).**
- Honest sensitivity: every +10KB of archive tightens the seg box by 6.7e-5; every 0.01 of
  pose contribution tightens it by 1.0e-4.

## §4 DELIVERABLE 4 — convocation synthesis (T3, Schmidhuber LEAD; ranked last-mile allocation)

**Ranked allocation:** ① free-solve mass 35.4% ceiling / 11.9% strict (rows 1–2, ~0 B, pb1 P2
NOW, aimed by the atlas) → ② priced-in-budget: clustered corrections + hot-cell quantum
refinement (row 3 triggers a/b — engineered to price BELOW 1.2731 water via clustering) →
③ priced-above-water-with-triggers: lane-corridor parametric carrier + flicker-phase channel
(rows 4–5; r7's context brief) — three tiers, no graveyard.

Per-master, grounded in the rows above:
- **Schmidhuber (LEAD):** the residual is now 94% boundary-phase information about a
  temporally jittering separatrix. The shortest program: re-aim bytes ALREADY PAID FOR (the
  per-pair token field) — the solve is compression, zero new rate. The tail's erased lane
  dashes are the vehicle's low-persistence casualties; they need a POSITIONAL sub-program
  (row-4 trigger a), not amplitude.
- **Numerical analyst (head-solve conditioning):** the head is exactly rank-4 and
  well-conditioned (σ1/σ4 = 1.74); with 98.8% two-class contests the per-flip residual is
  rank-1 — GN on tokens is well-posed AT THE HEAD. The conditioning risk is the COLLATERAL
  operator: ERF r90 ~300px ≫ 16px cell makes the token→flip Jacobian non-diagonal (measured:
  blind median −1). Directive: solve with the FULL-PAIR objective, never cell-local; accept
  steps only on net pair d_seg.
- **Microlocal analyst:** the wavefront set of the residual concentrates on near-horizontal
  boundaries in rows 11–12 at 16px scale — exactly the token lattice's resolution: actuation
  scale matches for the bulk. The exception is the Lane diagonal fine structure (tail), where
  the deep net has almost no diagonal-tuned filters (axis-aligned orientation gap) and
  integrates position ally — hence positional carrier, not HF amplitude (agrees with LEAD).
- **Statistician (vs g3's registry):** per-PAIR tails are thin (top-60 = 13.4%) — retire
  pair-ranked measure-first for THIS endpoint; per-CELL tails are extreme (top-100/768 = 83%).
  Re-key the hard-registry to hard-CELLS; the g3 pair registry stays valid for pose/rate axes.
- **Yousfi (scorer-blind-spot read of the ≤1-quantum mass):** 60% of flips sit where the
  scorer's OWN margin is < 0.25 — our carrier disagrees with the scorer by less than the
  scorer's own decision fragility. Sub-quantum-scale, texture-embedded steering (UNIWARD
  discipline) suffices for the κ band; the two-sidedness of fragile boundaries is why blind
  amplitude backfires (measured). Aim = the atlas's signed two-class directions.
- **Dissent recorded verbatim in frontmatter** (Contrarian: book −0.046 not −0.138 until P2
  measures; Assumption-Adversary: probe the renderer section before calling tier-2 token-only).

**What each consumer does DIFFERENTLY given the typing:** pb1 P2 — aim GN/QDBS at the top-100
cells with per-flip (direction, m_def) targets from `atlas_flat.npz`, accept on full-pair
d_seg, book the strict band first; measure the solved field's Δbytes by ONE re-encode. r7 —
context brief gains two named carriers (lane-corridor parametric ~8-dim/frame; flicker
boundary-phase per-pair). Composed candidate — budget sheet per §3 (pose ≤0.018-class
prerequisite; tokens ≤130KB; seg box ~6e-4 with the trigger ladder). co9 — ingest κ=0.0753,
water=1.2731, tier masses {11.9/35.4/47.8/16.9} as provenance-carrying SENSE constants.

## §5 Honesty labels, falsifiers, scopes

- MEASURED: everything in §1 (n600, realized-through-receiver, frozen CPU-torch scorer,
  positive control bit-exact vs pb1); κ and byte-neutrality (141 real edits); GT flicker/tie
  masses (cache). DERIVED: water 1.2731; §3 arithmetic (exact score formula). PROJECTED
  (labeled): the −0.138 tier-2 ceiling (Contrarian bound −0.046 measured-backed).
- verdict_scope: INSTANCE throughout (this checkpoint 33776302…, this archive 85d575be…, this
  scorer 68956e32…). No family/paradigm claims. All rows advisory; pointer unmoved.
- Falsifiers: row-1/2 — pb1 P2 population row under the single-edit lower bound (+151);
  κ — a full-cell joint calibration displacing the per-edit median by >2× IQR; §3 — any exact
  row whose realized components contradict the receiver-realized decomposition.
- 6-hook wire-in: sensitivity-map = ACTIVE (per-cell/per-class deficit masses); Pareto =
  ACTIVE (§3 budget sheet); bit-allocator = ACTIVE (hot-cell quantum refinement row);
  cathedral autopilot = N/A (no dispatchable archive from ru1; pb1 owns the candidate);
  continual-learning = ACTIVE (receipts + this memo); probe-disambiguator = ACTIVE (renderer
  rank-1 edit probe queued, op-routable 3). research_only=false for the atlas artifacts (they
  are pb1 P2 inputs); no score claims.

## DAG FEED — ddm_ru1 (2026-07-29)

FEED-ru1: endpoint residual typed through the scorer's own factorization at n600
realized-through-receiver (positive control bit-exact vs pb1 P1). 458,738 flips = 94%
GT-boundary jitter, 98.8% two-class rank-1 contests, 83% in 100/768 token cells, thin per-pair
tails. Currency measured: κ=0.0753 logit/token-quantum (141 real edits), token edits
byte-neutral; strict free-solve band 11.9% (ΔS −0.046), GN ceiling 35.4% (−0.138, Contrarian
bound applies); deep band 47.8%; tail 16.9% (65% Lane erasure, 64% flicker-targets). Floor
RE-SPECIFIED as price per operator doctrine: flicker∩fragile core 16.6% is per-pair-precision
mass with named re-pricing triggers (flicker-phase channel, hot-cell finer quanta, clustered
corrections, lane-corridor parametric) — zero walls. Corrected box: pose ≤0.018-class is a
HARD prerequisite (0.127 line cannot cross 0.172 at any rate); at 130KB tokens the seg box is
~6e-4, not 0.00116. Consumers: pb1 P2 (atlas_flat.npz targets), r7 (two named carriers),
composed candidate (budget sheet), co9 (SENSE constants). Pointer 0.1910828242 [contest-CPU]
UNMOVED; all advisory. [no-triality] [p0-ledger-ok]

## STORES CONSULTED

CLAUDE.md; AGENTS.md; docs/operating_manual_craft_handoff.md; MEMORY.md (box_retired /
objective_is_min_S / pose_is_terminal / meet_it_where_it_is / realization_is_quantization_gated
/ verdict_scope ladder); segnet_recursive_fractal_factorization_20260715.md;
frozen_scorer_exact_factorization_20260715.md; label_noise_floor_and_margin_saliency_20260618;
ddm_g3/g4 receipts; at1/at1x receipts; #580 resize_full_kernel;
pb1 live receipts (/Volumes/VertigoDataTier/pact/ddm_pb1_20260729/ + worktree tool);
t3_long_burn_lotto_v2 run dir (receipt/telemetry/manifest); gt_n600 cache; instrument-inventory
subagent report (paths/interfaces verified against committed artifacts).

## §AMENDMENT (2026-07-29, MAIN — #404 relative-significance cure for the tier-table price language)

The tier table's row-2 fragment ("collateral is the binding constraint") and sibling
priced-band rows use binding/limiting language — here is the relative arithmetic at the
CURRENT operating point (milestone row S=20.27 advisory: pose 19.51 / seg 0.389 / rate 0.376;
bars 0.19108/0.172):

- Row 2 (m_def<0.25 joint-GN band, ΔS −0.138 ceiling) is a POSITIVE-reach row, not a
  dismissal: −0.138 = **42% of the seg-axis gap** (seg contribution 0.389 vs seg-budget-at-bar
  ~0.061 → axis gap 0.328) and 0.69% of the total gap (pose-dominated until P3v2 resolves).
- "Collateral is the binding constraint" is MEASURED, not eyeball: blind single-token edits
  land median **−1 net flips (65% net-negative)** [this memo's aim-gain measurement] — the
  collateral cost that caps naive actuation — while aimed best-of-8 edits flip positive in
  17/18 hotspot cells (+24/quantum). Exit criterion (standing falsifier): aimed joint-GN
  realized rows measuring net-negative at scale would refute the reach ceiling downward;
  an ES/MC family row beating aimed edits at matched evals would refute the aim-currency
  upward. verdict_scope: INSTANCE (this endpoint, this actuation alphabet).
