# Necessity solver — inverse of the flattened factorization, per Morse-Smale stratum (2026-07-15)

**Operator (P0-of-all-P0 crown):** *"Pursue the unification as p0 of all p0. Can't we determine what is
necessary for realization by inverse of the flattened factorization?"* + *"And per edge and saddle."* +
the Kolmogorov guiding meta-question (*"shortest PROGRAM whose fixed point is the witness ... Kolmogorov,
not entropy"*) + *"v9 cgauge is v8 but with even more optimal carriers per class."*

**Pointer 0.19108 UNMOVED.** Everything here is MEANS: the necessity/free split + per-stratum seeds are
the carrier-content spec for V9·CGauge; the exact byte-closed `evaluate.py` row remains the only authority.

**Solver:** `tools/necessity_inverse_factorization_solver.py` (standalone, cached-artifact only; no trainer
edit, no dispatch). **Artifacts:** `experiments/results/necessity_solver_20260715/{strat,kladder,asupport,vjp,floors}.json`
(floors sha256 `e5e17120cd45…`). **Equation:** `realization_necessity_preimage_per_stratum_v1` (registered).
**Parent frame:** `projection_unification_and_eight_lenses_20260715.md` §2 (the crown);
`frozen_scorer_exact_factorization_20260715.md` (the chain being inverted).

**Axes / scope (honest):** stratification, margins, and rate floors = **n600 (ALL scored pairs)** on the
bit-exact cached GT argmax/margin fields (`gt_n600.npz`; margin parity vs the real forward == 0.0 per
segnet-fractal stage_b1). A⁻¹ support = **exact closed-form matrices**, stride-25 subset (24 frames).
N⁻¹/uint8 fragility = full-chain VJP through the REAL frozen CPU-torch SegNet **including the exact
resize**, stride-75 × sampled pixels (n=120). All `[macOS-CPU advisory]`, research_only, score_claim=false.
Floors are geometric/codelength quantities at (384,512) — **NOT d_seg claims** (the measured D-ladder rows
`rate_law_ladder_v2_measured` are the sibling codelength surface; not conflated).

## 1. The inversion, step by step (what each inverse yielded)

| step | inverse | measured yield |
|---|---|---|
| decision⁻¹ | rank-4 polytope margins (cached, bit-exact; pair normals from `segnet_head_rank4_linear_flipdist_v1` — READ, not re-derived) | per-stratum margin fields §2; **runner-up == neighbor-label 80/80** at edges (validates keying the cached top1−top2 margin by the adjacent-class pair) |
| N⁻¹ | full-chain VJP min-norm displacement `m/‖∂m/∂x_cam‖` (first-order; through the real frozen net) | fragility ladder §4: interior 90.8 ≫ edge 14.2 > saddle 12.2 (median LSB); the fragile TAIL lives at saddles (p10 0.89) |
| A⁻¹ | exact support pullback `(R_hᵀ M R_w) > 0` (closed-form separable kernel, #47/#49; verified bit-exact vs einsum) | camera-res necessity split §3: strict 1.66%, loose 75.6%, certified-free 22.7% |
| ∩ uint8 | max per-coordinate amplitude of min-norm δ* vs 0.5 LSB | sub-LSB (realization-limited) fractions: **saddle 29.2% > edge 12.5% > interior 0%** — the uint8∩preimage intersection is TIGHT exactly at saddles |

## 2. The stratification (n600, exact on the cached argmax)

- **Cells (2-cells):** interiors are 97.8% of scorer pixels. Interior margins med 5.6–6.1 for
  Road/Undrivable/MyCar — deep polytope interior, membership-only necessity (blind B2). **EXCEPTION:
  Lane interiors are fragile everywhere** (margin med 1.54, p10 0.97 — 4–6× lower than other classes):
  the Lane class has effectively NO safe interior; treat ALL of Lane as annulus.
- **Edges (1-cells), per pair (cracks/frame · components/frame · p_corner · H_turn bits/step · margin med):**

| pair | cracks/f | comp/f | p_corner | H_turn | m_med | flipdist_feat med |
|---|---|---|---|---|---|---|
| Road-Lane | 1356.8 | **21.42** | 0.517 | **1.516** | 0.39 | 0.0987 |
| Road-MyCar | 529.5 | 1.94 | **0.092** | **0.534** | 0.42 | 0.155 |
| Road-Undrivable | 483.6 | 4.15 | 0.237 | 1.028 | 0.35 | 0.135 |
| Undrivable-Movable | 165.9 | 3.50 | 0.497 | 1.497 | 0.19 | 0.065 |
| Road-Movable | 150.5 | 3.64 | 0.435 | 1.422 | 0.23 | 0.078 |
| (Lane-MyCar / Lane-Undriv / Lane-Movable / Movable-MyCar) | ≤8.5 | ≤1 | — | — | — | — |

  Road-Lane is ~50% of all cracks, has the HIGHEST turn entropy AND 21.4 components/frame — **the dash
  long-tail measured as fragmentation** (each dash = its own curve component paying start-point overhead).
  Road-MyCar (hood) is the anti-pole: near-straight (p_corner 0.092), 2 components, and static (#139 IoU 0.994).
- **Saddles (0-cells):** 11.2 junction vertices/frame (6,703 total; only ONE 4-way in all of n600).
  Triple census: Road-Undrivable-Movable 3,344 · Road-Lane-Movable 1,348 · Road-Lane-MyCar 1,114 ·
  Road-Lane-Undrivable 875 (Lane participates in 3 of the top 4). Saddle-flank margins med 0.26
  (vs edge ~0.39) — saddles are the most fragile 1-D-measure-zero set of the partition.

## 3. NECESSARY vs FREE at camera res (exact A⁻¹ pullback)

| tier | camera-res fraction | necessity class |
|---|---|---|
| saddle support | 0.016% | strictly necessary, PRECISION-critical |
| edge support | 1.646% | strictly necessary (values, chroma per rgb_at_boundaries) |
| cell support | 75.64% | LOOSE: membership-only (any in-polytope fill; generic/free content) |
| zero-weight (ker A rows/cols) | 22.70% | certified FREE (#401/#49; fill generically) |

**Strict-necessary camera-res support = 1.66% of pixels.** 98.3% of the camera frame is free-or-loose —
the quantitative form of "put bytes on the seen sufficient statistic and none on the blind complement."

## 4. Fragility / uint8 tightness (full-chain VJP subset, n=120)

| stratum | flipdist med (LSB) | p10 | sub-LSB frac (min-norm δ* max-coord < 0.5) |
|---|---|---|---|
| saddle | 12.2 | **0.89** | **29.2%** |
| edge | 14.2 | 4.06 | 12.5% |
| interior | 90.8 | 50.8 | 0% |

**Answer to "do saddles dominate?": saddles dominate NECESSARY PRECISION, not necessary BYTES.** They are
where the uint8-lattice ∩ preimage intersection is tight (29.2% of samples realization-limited: the min-norm
flipping displacement is sub-LSB, so lattice-realizable displacements must exceed min-norm — corrections
there need structured multi-pixel placement, not amplitude). In bytes they are ~0-marginal given edges:
every saddle is a crack-graph vertex of degree ≥ 3, IMPLIED by the intersection of coded curves.

## 5. Rate floors — H-ladder vs K-ladder (Kolmogorov supersedes entropy)

Both floors are **spatial-only** (per-frame independent; NO temporal/ξ-advection prediction — that is the
named headroom, lens 5). Models explicit in the artifacts.

- **H-ladder** (entropy of the necessary preimage; cracks×H_turn + components×(log2(HW)+2)):
  **303,047 bytes n600** (505.1 B/frame). Per-pair: Road-Lane 309.6 B/f (**61%**) · Road-Undriv 72.3 ·
  Road-MyCar 40.1 · Undriv-Movable 39.6 · Road-Movable 35.7 · rest ≤3.1.
- **K-ladder** (generator+seed; generator = polygon rasterizer + region fill, FREE in inflate.py per rule
  118; seed = DP-simplified boundary vertex chains, int16-delta + brotli-q11, MEASURED bytes):
  eps=0.5px → 232,605 B · **eps=1px → 143,552 B** (239.3 B/frame) · eps=2px → 97,865 B (shared-edge /2
  adjusted; border arcs make the adjustment slightly under-correct). **K/H = 0.47 MEASURED** — the
  generator reduction already halves the entropy floor at equal spatial tolerance, before any temporal
  or per-pair generator specialization.
- **Cells seed:** ~15 B/video (5 palettes) given edges; membership implied by the edge complex.
  Chroma trunk = per-cell palette (6.2× worth, `rgb_at_boundaries_derivation_20260715`).
- **Saddles seed:** ~0 marginal bytes given edges (§4); their necessity is a PRECISION annotation
  (variable-eps: tighter DP tolerance near junctions — tie-locus #360), not a byte section.
- **Context (not conflated):** current archive = 83,430 B total; the spatial-only K-floor for edges alone
  (143.5 KB) EXCEEDS it ⟹ the live vehicle already amortizes temporally below per-frame spatial coding.
  The inversion's remaining wins are therefore per-stratum GENERATOR specializations (§6), not raw coding.
- **Kolmogorov reduction ledger (the standing test "fixed point of a shorter program?"):**
  Road-Lane → openpilot-polynomial + dash-phase-ξ generator (~8-dim/frame vs 310 B/frame H-floor; L71
  poly d_seg 0.00087 + L73 dash phase ≈ ego-ξ near-free) — **the single largest identified reduction**;
  Road-MyCar → static one-time seed (hood IoU 0.994; per-frame cost → ~0);
  Undrivable-Movable / Road-Movable → per-object Laguerre-cell generators (v8 #284) + ξ-transport.

## 6. V9·CGauge carrier map (v9 = v8++ = per-STRATUM optimal carriers)

| stratum | carrier (generator, seed) | what the inversion changes / makes cheaper |
|---|---|---|
| cell bulk | `road_undriv_bulk_field` / `decoupled_field` + palette | **CHEAPER**: content requirement is polytope MEMBERSHIP + palette constant only (interiors margin ≥ 2.5 p10 for Road/Undriv/MyCar); drop all value content beyond argmax-correct fill — the bulk field needs sign(margin) not values |
| Lane "cell" | — (reclassify) | **Lane has no safe interior** (margin med 1.54): fold the whole class into the edge/annulus carrier; do not spend a bulk-field on Lane |
| edge: Road-Lane | `curve_relative_offset_coder` δ(s) → polynomial+dash-phase generator | **CHEAPER by generator swap**: 61% of the H-floor; highest H_turn + 21.4 comp/f (dashes) ⟹ per-frame chain coding is the WRONG program; the openpilot-poly + phase-ξ generator carries dashes as phase, curves as ~8 coefficients |
| edge: Road-MyCar | one-time static seed + identity transport | **CHEAPER**: H_turn 0.534, near-static ⟹ code once per video, ξ≈0 transport; per-frame cost ~0 |
| edge: others | curve-relative δ(s), per-pair tuned (H_turn measured per pair) | per-pair floors now measured (table §2) — coder budgets per edge, not shared |
| saddle | NO byte section: variable-eps precision annotation on edge curves near junctions | **NEW (completes the complex)**: saddles = where 29.2% of flips are sub-LSB realization-limited; carrier = tighter local tolerance + structured multi-pixel placement (tie-locus #360), costing precision on EXISTING edge seeds, not new bytes |

## 7. Negatives / caveats (verdict_scope)

- "Saddles dominate the necessary rate" — **REFUTED at FORMULATION scope** (bytes currency, given coded
  edges, on this partition): saddle marginal bytes ≈ 0. The operator intuition survives in the PRECISION
  currency (29.2% sub-LSB) — the correct carrier is precision, not bytes. Scope: n600 stratification +
  n=120 VJP subset; INSTANCE→FORMULATION only, not paradigm.
- The K/H=0.47 and all floors are spatial-only geometric tolerances (eps px @ (384,512)), not d_seg-verified;
  the d_seg-verified sibling is the D-ladder (`rate_law_ladder_v2_measured`). A DP-eps → d_seg calibration
  A/B through the real byte-closed decode is the named next measurement before any carrier ships.
- N⁻¹ is first-order (J_f⁺ local); exactness holds only in penultimate-patch space (rank-4 head law).

## 8. Triality + stores consulted

- **DAG leg:** FEED-necessity appended (`sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`).
- **equations leg:** `realization_necessity_preimage_per_stratum_v1` registered (3 anchors: A-support
  split · rate floors · saddle fragility).
- **DSL leg:** N/A-with-rationale — this is a measurement/inversion surface, not a trainer lever; its
  consumers are the V9 carrier builders (`decoupled_field` / `road_undriv_bulk_field` /
  `curve_relative_offset_coder`), which hold their own DSL surfaces.
- **STORES CONSULTED:** frozen_scorer_exact_factorization (chain) · segnet_recursive_fractal +
  `segnet_head_rank4_linear_flipdist_v1` (head, NOT re-derived) · evaluator_invisibility_basis #47/#49
  (exact resize, REUSED) · blind_coordinate #401 · rate_law_ladder_v2_measured (D36/D37 context) ·
  rgb_at_boundaries_derivation (chroma palette trunk) · SPEC_v8 carriers · #333 annulus · #360 tie-locus ·
  L71/L72/L73 (openpilot poly + ξ + dash phase) · projection_unification_and_eight_lenses (frame).

## 9. Next actions (feeding an exact row)

1. DP-eps → d_seg calibration through the real byte-closed decode (converts the K-ladder from geometric
   to score-verified; picks the shipping eps per pair, tighter near saddles).
2. Road-Lane generator swap A/B: polynomial+dash-phase seed vs chain-coded δ(s) on the same target curves.
3. Road-MyCar one-time-seed carrier (static hood) — smallest, most certain byte win of the map.
