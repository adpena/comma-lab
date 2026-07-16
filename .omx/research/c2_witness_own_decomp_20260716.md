# c2 WITNESS-OWN residual decomposition + law-transfer verdicts (2026-07-16)

**Source:** #515 FINAL-OPTIMAL c2 witness-design campaign — the arm that closes the c2
taxonomy's **#1 caveat** (`c2_perclass_stratum_carrier_taxonomy_20260716.md` §6: the
taxonomy decomposed the PALETTE vehicle, which DILATES the fine classes; the trained
witness ERODES them, so the taxonomy's bucket WEIGHTS were known not to transfer).
This arm runs the SAME decomposition on the best available TRAINED witness and
re-verifies each taxonomy law on it. **Pointer 0.19108 UNMOVED — everything here is
MEANS** (c2 design inputs); the exact `upstream/evaluate.py` row is the only authority.

**Vehicle:** the FROZEN **mod32cap EMA-best ep650** checkpoint — the best measured
witness d_seg we hold (`levelset_best.json` 0.003366 verdict; **0.003146 n600
re-measured here through the exact contest R + frozen CPU SegNet**, matching the prior
dash-probe baseline 0.0031463 to 4 decimals — cross-checked). The live c1 run
(`levelset_n600_witness_20260716T014623Z`) has only dry-start state, no usable frames —
SAID SO and fell back per tasking. The V9·CGauge coherent arm (ep150, d_seg 0.0348) and
v5-preserved (0.0252) are strictly worse checkpoints.

**Axis / honesty:** `[macOS-CPU advisory]` — frozen CPU-torch fp32 SegNet, bit-exact
cached GT argmax (`gt_n600.npz`). n600 unless labelled; smokes are stride-5 subsets
(120 frames, ranking-only). `research_only; score_claim=false; promotable=false`.

**Tool:** `tools/c2_witness_own_decomp.py` (stages decomp/temporal/sens/smoke; resumable,
chunked-foreground; reuses the dash-probe frozen renderer read-only). **Artifacts:**
`experiments/results/c2_witness_own_decomp_20260716/` (decomp_rows.jsonl · temporal.json
sha 77f760af · sens.json · smoke_*.json); frame cache (1.75 GB, rebuildable) on the SSD
tier `/Volumes/VertigoDataTier/pact/c2_witness_own_decomp_20260716/` with
REBUILD_MANIFEST.json.

---

## 1. The witness-own decomposition (n600, MEASURED) — the table c2 actually targets

Witness d_seg through R = **0.003146** (n600). Same stratum priority as the taxonomy
(saddle > edge(1px) > near(≤3px) > far; class = pixel's own GT class).

| bucket | d_seg contrib | % of residual | persist_next | occupancy |
|---|---:|---:|---:|---:|
| **Lane\|edge** | 0.001197 | **38.0%** | 0.135 | 0.048 |
| **Road\|edge** | 0.000917 | **29.2%** | 0.110 | 0.034 |
| Undrivable\|edge | 0.000308 | 9.8% | 0.113 | 0.027 |
| Movable\|edge | 0.000301 | 9.6% | 0.200 | 0.027 |
| MyCar\|edge | 0.000126 | 4.0% | 0.054 | 0.039 |
| Road\|near | 0.000118 | 3.8% | 0.190 | 0.055 |
| all remaining (near/far/saddle) | 0.000179 | 5.6% | ≤0.25 | ≤0.07 |

- **Edge (1px) stratum = 90.6% of the witness residual.** The palette's #2 bucket
  (Movable|far, 24.3%) is **ABSENT** on the witness (<100 px total over n600) — the
  trained witness has essentially NO interior/region misses left, only boundary jitter.
- **Pair-side (edge+near): Road-Lane = 66.0%** of the residual (Lane side 0.1432 +
  Road side 0.0643, ×100 units); Road-Undrivable 15.1%; Movable pairs 21.3% combined;
  Road-MyCar 9.6%.
- **Erosion CONFIRMED (the predicted mirror):** edge|Lane→Road **165,026** vs
  edge|Road→Lane 62,459 = **2.64:1 on the Lane side** (palette: 240k Road-side vs 123k
  = dilation). Vehicle-dependent flip side, exactly as the taxonomy §2 predicted.
- **Temporal character = FLICKER, not static and not object-tracking:** persist_next
  ≤ 0.251 and occupancy ≤ 0.067 in EVERY bucket (palette: persist up to 0.865,
  occ up to 0.26). The witness residual is per-frame boundary jitter — consistent with
  L85 (flicker = GT sub-pixel advection phase) and the evasion-arm realization floor
  ("~8e-4 Lane-dominated, sub-pixel geometry not amplitude").

## 2. Law-transfer verdicts (each taxonomy law re-verified on witness frames)

| taxonomy law | witness verdict | evidence |
|---|---|---|
| (b) one-sided dominance ORDERING (correct-side < symmetric < deep-side) | **HOLDS** | β=0.5: oneside_movable +35.8% < oneside_lane +62.9% < symmetric +123.8% < deep +141.6% (subset) |
| (b) per-pair slope field | **TRANSFERS BY CONSTRUCTION** | measured on cached GT margins (vehicle-agnostic); not re-measured |
| (c) luma-BT.601 = cure driver | **HOLDS (softer)** | median luma cure energy 0.85 (buckets 0.83–0.87) vs 0.92–0.94 palette — chroma share doubles on this chroma-active vehicle, luma still dominant |
| (c) non-local cure | **HOLDS** | r4 ≤ 0.26, r36 0.31–0.72 (sens, 120 VJP samples) |
| (c) flat-orthogonal + Lane flat-coherent brighten exception | **HOLDS, exception REPLICATED** | coh ≤ 0.07 all buckets except Lane (cohGT 0.28 edge / 0.37 saddle, sign +9/3 = brighten lane side) — now on the **#1** bucket |
| (d) crisp > blurry | **HOLDS** | blur σ=2 → +20.3% |
| (a) region-from-boundary | **MECHANISM HOLDS, exploitable side EMPTY** | β=2 Movable band perturbs non-Movable buckets non-locally (Road\|near +5.4×, Lane\|edge +41% px on subset) — border DOES drive region identity; but the witness has no interior deficit left to cure |
| **carrier VALUES** | **DO NOT TRANSFER** | the palette winner (oneside_movable β=2, −29.5% @ 0B) is **+80.1%** on the witness |

## 3. NEW LAW — flat-amplitude EXHAUSTION on the trained witness (formulation-scoped)

Every flat palette-delta band variant is **net-negative** on the witness (subset
baseline 0.0031068): oneside_lane β=0.1 **+1.2%** (best ≈ neutral), β=0.25 +23.8%,
β=0.5 +62.9%; oneside_movable β=0.5 +35.8%, β=2.0 +80.1%; symmetric +123.8%; deep
+141.6%; blur +20.3%. β-monotone WORSENING (palette: monotone improving).

**Mechanism (MEASURED):** oneside_lane β=0.25 cures its target (Lane|edge −19% px) but
buys Road|edge **+70%** collateral — at the trained optimum, the ~30×-more-numerous
correctly-classified boundary pixels sit near the margin too, and any finite flat push
crosses more of them than it cures. The trained witness **already sits at the flat-basis
optimum of its own residual**; what remains is sub-pixel geometry / temporal phase.

**Verdict scope: FORMULATION** (flat palette-delta band, kpx=2, post-hoc composite,
5 sides × β 0.1–2.0). Trained band profiles, sub-pixel band placement, and joint-trained
carriers remain OPEN — but the β→0 bracket bounds the flat family's ceiling at ≈0.

**Echo of L68:** this is the d_seg analogue of the pose photometric wall — post-hoc
render-surface corrections fail on the trained vehicle; only JOINT descent (or sub-pixel
geometry the render itself carries) crosses. The taxonomy's flat carriers are for the
UNTRAINED/analytic (palette/necessity) vehicle.

## 4. Ranked witness-specific c2 targets (value = S units of a full cure, 100·contrib)

| rank | target | % of witness residual | S value | carrier form (from measured laws) |
|---|---|---:|---:|---|
| 1 | **Road-Lane boundary flicker** (Lane\|edge + Road\|edge at the Road-Lane pair) | 66.0% | **0.208** | joint-trained sub-pixel boundary geometry + appearance-phase (ξ,R) carriers (L85/L86 endgame #424/#425/#360); analytic lane band + dash-phase priors (L71/L73) as geometry, NOT flat amplitude; the Lane flat-coherent brighten survives only as a training-time prior direction |
| 2 | Movable border flicker (both Movable pairs, edge+near) | 21.3% | 0.067 | ξ-tracked per-object border carriers — border PROFILE only (region-from-boundary); witness already tracks the objects (no interior bucket; persist 0.20–0.23) |
| 3 | Road-Undrivable horizon flicker | 15.1% | 0.047 | horizon-band sub-pixel geometry; Road side is the vehicle-agnostic shallow side (2.78×) — precision goes to the Road side |
| 4 | Road-MyCar hood rim | 9.6% | 0.030 | extend the static hood seed to a JOINT-trained rim profile (witness rim persist 0.054 = flicker, so a static rim band is NOT enough on this vehicle) |
| 5 | saddles (all) | 0.7% | 0.002 | no bytes; precision annotation on edge carriers |

(Pair-side %s overlap the bucket %s; both sum over the same residual.) **d_seg/byte
honesty:** NO positive-Δ carrier was measurable on the witness in this $0 post-hoc
scope — that is itself the headline (the flat 0-byte family measures 0-at-best here,
vs −29.5% on the palette). The witness targets are TRAINING-side levers; their
d_seg/byte must come from per-stage A/Bs of the c2 run, not post-hoc composites.

## 5. OWED DSL Lever SPEC — `oneside_border_band_carrier` (recorded, NOT hand-wired)

- **Surface:** `tac.witness_dsl` archive-build/generator seed `Lever` (inflate-time
  band rasterized from the decoded partition = FREE generic code, rule 118; the per-pair
  constants are video-derived → COUNTED).
- **Typed params:** `pairs: dict[pair_code, BandTerm]` with
  `BandTerm = {side: class_id|None, beta: float, kpx: int}`; defaults from the measured
  slope field (`carrier_side_for_pair`): Movable pairs side=Movable β=2.0 kpx=2;
  Road-Lane side=Lane β≤0.5.
- **Vehicle gate (from THIS memo):** `enabled_vehicles = {palette_necessity: candidate
  (pending the §6 A/B), trained_witness: OFF — measured net-negative (flat-amplitude
  exhaustion, §3)}`.
- **Counted bytes:** ~2 B/pair (side 3b + β u8 + kpx u4) ≈ 12–20 B total →
  ΔS_rate ≈ +1.1e-5 (negligible).
- **Registry state:** never-fired; duty-to-measure = the §6 A/B. Recorded OWED here +
  DAG; lands as a `Lever` factory when the carrier ships (never a hand-added flag).

## 6. Decisive validation A/B (DESIGNED, NOT RUN — flagged OPERATOR-GO)

Converts the taxonomy's advisory −29.5% into a real n600 d_seg row on the vehicle where
it pays (the **palette/necessity** vehicle — NOT the witness, per §3):

- **Arms:** necessity knee archive (ε=0 lossless partition + palette + ds16 hood seed,
  parent d_seg_real 0.01328) **vs** same + `oneside_movable β=2.0 kpx=2` band composited
  at inflate time.
- **d_seg leg (byte-closed through-R):** run the existing taxonomy smoke path at
  `--stride 1` (full n600): `.venv/bin/python tools/c2_perclass_stratum_carrier_analysis.py
  --stage smoke --variant oneside_movable --beta 2.0 --stride 1` (≈50 min local, $0),
  PLUS the band constants appended to the archive build so the decode is byte-closed
  (extend the necessity archive assembler with the §5 BandTerm section; ~16 B).
- **Expected (advisory, from the stride-5 ranking):** d_seg 0.01328 → ≈0.00936,
  ΔS_seg ≈ **−0.392**; ΔS_rate ≈ +1.1e-5.
- **Pose-safety gate (REQUIRED before GO):** the band's palette-delta has chroma
  components, but PoseNet chroma is 2×2-box-averaged (<2px invisible per
  `frozen_scorer_exact_factorization` §6); the luma component IS PoseNet-visible at 2px.
  Measure n600 `d_pose = MSE(PoseNet(YUV6 pair)[:6] vs GT targets)` band-ON vs band-OFF;
  gate: Δ√(10·d_pose) < 0.005 against the banked 0.127 contribution (R1 dxi, L68).
- **Authority:** this A/B is the advisory gate; only an exact `upstream/evaluate.py` row
  on the assembled archive is a score. Validates the CARRIER for c2 composition on the
  analytic vehicle; it does not move the pointer by itself.

## 7. Round-1 adversarial review (own attack) + boundaries

- **Checkpoint caveat:** the decomposition is of mod32cap ep650 (luma-witness lineage,
  chroma=1, w_pose=0) — the best AVAILABLE witness, not the c2 vehicle itself. If c2's
  trunk differs structurally (per-class carriers, pose-conditioned), its weights shift
  again; the FLICKER character and the exhaustion mechanism are expected to persist
  (they follow from "trained to its own flat optimum"), but that is INFERRED, not
  measured, for c2.
- **Smokes are stride-5 rankings** (120 frames); the exhaustion NULL is supported by 8
  variants all ≥ baseline with monotone β-trends — but each individual number is
  subset-labelled. The β→0 neutrality bracket is the strongest evidence.
- **Post-hoc only:** the smokes composite at camera res AFTER R's bicubic^ (same surface
  as the taxonomy smokes — consistent comparison). A band composited at the render grid
  BEFORE R would see different resize physics; unmeasured (open).
- **NOT measured:** the c2/V9 vehicle's own decomposition (checkpoint too immature,
  ep150 d_seg 0.0348); trained/optimized band profiles (formulation boundary of §3);
  pose interaction of any variant (design §6); n600 versions of the smoke variants;
  sub-pixel band placement.
- **What would change my mind on §3:** a single flat-band variant with per-pair tuned
  β ≤ 0.1 and sub-pixel (pre-R) placement beating baseline by >2σ of the subset noise
  would re-open the flat family on trained vehicles at INSTANCE level.

## 8. Triality + stores consulted

- **DAG:** FEED-c2w-witness-own appended to `sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`.
- **equations:** `witness_own_residual_decomposition_v1` REGISTERED (3 anchors:
  decomposition · cure-driver transfer · flat-amplitude-exhaustion smoke). Sibling of
  `perclass_stratum_residual_carrier_taxonomy_v1` (closes its #1 vehicle caveat).
- **DSL:** the §5 Lever spec recorded OWED (vehicle-gated); not hand-wired.
- **STORES CONSULTED:** c2_perclass_stratum_carrier_taxonomy (parent) ·
  necessity_dseg_calibration · dash_comb_probe (renderer + baseline cross-check) ·
  frozen_scorer_exact_factorization · adversarial_evasion_fisher_null (realization
  floor) · L65/L68/L85/L86 · levelset_best.json lineage (mod32cap / v5 / V9-coherent).

**Pointer 0.19108 UNMOVED — MEANS.**
