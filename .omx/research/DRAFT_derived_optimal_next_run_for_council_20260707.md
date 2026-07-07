# DRAFT — derived-optimal next-run proposal (INPUT to the grand-council symposium)

**Status: DRAFT ONLY — binds nothing.** Prepared per operator directive for the council "to
review and edit and tear apart and use or not however it wishes; the council will be the one to
ultimately design and sign off." Every number below carries its source artifact + an
epistemic label per the canonical vocabulary (MEASURED / DERIVED / INFERRED / ASSUMED / UNKNOWN).
Axis discipline: everything herein is `[macOS-MLX/CPU research-signal]` advisory unless tagged
otherwise. **Pointer contest-CPU 0.19110 UNMOVED** (`.omx/state/canonical_frontier_pointer.json`)
— this entire document is MEANS; the pointer moves only through a byte-closed
`upstream/evaluate.py` n600 exact row.

Inputs assembled (the full information space): DAG FEED-07a/07b/07c + FEED-247\*
(`.omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`), islands T3 symposium
(`.omx/research/council_t3_symposium_islands_treatment_arm_20260706.md`), #302 curriculum
derivation T3 (`.omx/research/council_grand_symposium_curriculum_derivation_20260705.md`),
the $0 probe (`experiments/results/levelset_n600_witness_mod32cap_20260706T115554Z/lane_share_probe_ep225_n600.json`),
live telemetry (`annulus_live_narration.txt`, `levelset_best.json`, `costate_shadow.jsonl`,
`lever_ledger.json`, `launch.sh` in the same run dir), the composable-lever registry
(`tac.witness_dsl.lever_registry.name_composable_levers()`, 27 levers), and the canonical
equations registry (`.omx/state/canonical_equations_registry.jsonl`).

---

## 1. STATE — the measured board

**Rate is beaten; d_seg is the wall; pose is held (seg-only arm).**

| Quantity | Value | Label | Source |
|---|---|---|---|
| Exact frontier | **0.19110** `[contest-CPU]` | MEASURED | `.omx/state/canonical_frontier_pointer.json` (pointer-only SoT) |
| PR95 rate term | **0.1188** (178,417 B) | DERIVED (25·178417/37,545,489; archive bytes from two intake copies) | `docs/operating_manual_craft_handoff.md` §4; DAG FEED-07a |
| Current witness rate | **0.05499** | MEASURED (byte-closed archive stat) | DAG FEED-07a (line "Rate context") |
| Live baseline BEST d_seg | **0.0036364 @ ep425** | MEASURED n600 verdict `[macOS-MLX research-signal]` | `.../levelset_n600_witness_mod32cap_20260706T115554Z/levelset_best.json` (ts 2026-07-07T02:37:38Z) |
| Baseline trajectory | ep225 0.004869 → ep250 0.004751 → ep275 0.004682 → ep300 0.004571 → ep350 0.003953 → ep425 0.0036364 | MEASURED | `annulus_live_narration.txt` + `costate_shadow.jsonl` (ep350 row, `NO_STALL`/converging) + `levelset_best.json` |
| Residual character | 97.0% of flip mass in the boundary annulus; interior_flip_frac 1.5e-4; Lane(cls1) dominant stuck boundary (annulus flip-frac 0.387) | MEASURED (16-pair strided ADVISORY subset) | `annulus_live_narration.txt` |
| tau erosion in baseline | **ABSENT** (d_seg descends cleanly through tau@300) | MEASURED | FEED-07c + `annulus_live_narration.txt`; contrast #205's creep 0.004752@300→0.006568@400 (#302 symposium §B.2, MEASURED) |
| Muon stage | not yet fired (ep726 scheduled) | — | `launch.sh` `--muon-start-epoch 726` |

**Budget arithmetic (the target band).** Score law
`S = 100·d_seg + √(10·d_pose) + 25·bytes/37,545,489` (`upstream/evaluate.py`; costate row cites
the same law). With rate = 0.05499 (MEASURED above) and a pose term p:

- beat-frontier break-even: `d_seg = (0.19110 − 0.05499 − p)/100`
- sub-0.15: `d_seg = (0.15 − 0.05499 − p)/100`

With **p ≈ 0.018** — the ancestor stored-target sidecar's √(10·3.4e-5) — break-even d_seg ≈
**0.00118** and sub-0.15 needs ≈ **0.00077**. ⚠ **p = 0.018 is ASSUMED (borrowed-hypothesis):
d_pose is OPEN + UNMEASURED on the witness vehicle** (memory
`project_pose_solved_screw_twist_dual_use_film_conditioned_sidecar_20260701.md`; 3.4e-5 is the
RGB ancestor's number). The live baseline runs `--w-pose 0` (pose-blind by design; measured
operating d_pose 125.833 per `costate_shadow.jsonl` — transparency, not the deploy plan). The
pose mover is the SEPARATE store-nothing ξ carrier (#257 derive-H, memory
`pose_mover_is_store_nothing_xi_derive_h_not_warp_real_luma_20260706.md`), not this run.
FEED-07c states the same band independently: full island birth ⇒ d_seg 0.0037 → ~0.0013, seg
term 0.37 → 0.13 "≈ the break-even band."

**Probe verdict (the branch gate) — PROCEED-class.** `lane_share_probe_ep225_n600.json`
(n600, ep225 EMA-BEST, 6.2h, `[macOS-CPU/MLX advisory] NON-PROMOTABLE`):

- Islands region = **63.9% of ALL flips on 1.82% of px** (within-region flip-frac 90.2%);
  bulk interior = 9.3% of flips on 93.4% of px. MEASURED.
- Movable(3): within-flip **93.1% un-born**, share_of_d_seg **44.8%** (single biggest
  contributor). Lane(1): within-flip **83.9% un-born**, share_of_d_seg **19.1%**. Road 20.3% /
  Undrivable 12.7% / MyCar 3.1%. MEASURED.
- The islands symposium's ASSUMED premise ("does plain CE birth islands?") RESOLVED: **NO** —
  both rare classes essentially un-born (confirms the deliberate baseline design; refutes the
  DEFER "98% bulk" branch; branch criterion lane ≥ ~10% un-born SATISFIED at 19.1%).
- **BINDING CAVEAT (travels with every ΔS claim):** shares measured on the **WITNESS-ALONE**
  surface (probe `d_seg_subset` 0.0257 vs live composed verdict 0.0049 at the same epoch;
  seeds not persisted in ckpts ⇒ composed surface unreconstructable). The probe self-flags its
  shares as an **UPPER BOUND** on the composed-surface share. Within-class un-born fractions
  transfer (live `part_frac` lane=movable=0 independently confirms), but the **composed-surface
  ΔS ceiling arithmetic is OWED to this council before any launch-sizing claim** (FEED-07c).
- Focal-γ analytic island-weight share (the #301 calibration target surface): γ=0.5→0.284,
  γ=1→0.506, γ=2→0.599, γ=3→0.633. MEASURED. Note the tension logged in §6-Q4.

**Islands arm status.** T3 symposium verdict REVISE, gated on the probe → probe landed
PROCEED-class (FEED-07c). Warm-start verdict: **FROM-SCRATCH** (CE-converged basin has ≈0
island gradient; islands symposium §Warm-start). Launch remains GO-gated on: memory-safety
CRITICAL fixes mid-flight + the composed-surface ceiling arithmetic (FEED-07c) + operator GO for
heavy GPU (containment non-negotiable).

**The mod32cap control is a COUNCIL-DESIGNED clean baseline, not missing research** (memory
`mod32cap_is_council_designed_clean_baseline_not_missing_research_20260706.md`): eik-off,
fixed-epoch schedule, no focal, islands un-born are DELIBERATE controls; the derived curriculum
and island treatments are the arms this draft composes, to be A/B'd against it at matched
epochs.

---

## 2. PRIMARY ARMS (per FEED-07a operator reframe: basis-match + rule-118 are PRIMARY; capacity ladder = PR95-echo local minimum)

All compositions below are **already proven to compile + parse through the real trainer
argparse**: `test_feed07_dsl_wirein.py` validates sealed base + DirectionalBasisRebalance +
AnalyticLaneRenderBand + SeedIslandEased + MuonWarmStart → `validate()==[]` (FEED-07c #335).
Every lever is default-off / byte-identical when unfired and auto-surfaces in the activation
ledger's `duty_to_measure()` for the #247 SENSE (no orphaned "off").

### Arm A — anisotropy-REBALANCED basis (`DirectionalBasisRebalance`)

- **Rationale:** all-class DIRECTIONAL basis = **−48% d_seg at ~0 bytes** (MEASURED,
  `[macOS-MLX research-signal]`, n600 baseline d_seg 0.008257, CLAUDE.md §capstone lever
  ranking, FEED 2026-06-25) — "THE decisive lever, basis-match PRIOR to capacity." The live
  control spends `--freq-across 32 / --freq-along 8` — BACKWARDS vs the MEASURED along-tangent
  **3.2× frequency deficit** (4-lens root cause, memory
  `lane_dash_residual_root_is_along_tangent_freq_deficit_R_allpass_20260703.md`; FEED-07a).
- **DSL:** `--dsl-lever DirectionalBasisRebalance` — two DERIVED regimes
  (equation `anisotropic_basis_two_regime_allocation_v1`, registered):
  - `regime="lane_offloaded"` (band ON): along = max(4, round(√across)) = **6** @ across=32
    (Candès–Donoho parabolic scaling — DERIVED; the √-optimum anchor is
    ASSUMED_AWAITING_VERIFICATION per the registry row; this run's A/B is the owed anchor).
  - `regime="lane_carried"` (band OFF): along = min(across, round(8·3.2)) = **26** (MEASURED
    dash-comb deficit).
- **Mechanism:** orient/allocate the Fourier budget to the all-class boundary tangent field so
  the basis matches the codim-1 separatrix anisotropy; capacity pays only AFTER basis-match
  (capacity alone on isotropic basis +6%, MEASURED, CLAUDE.md lever ranking).
- **Owed measurements:** per-regime n600 A/B vs control at matched epochs (the equation's
  missing anchor); composed with Arm B to select the regime.

### Arm B — rule-118 analytic lane band (`AnalyticLaneRenderBand`)

- **Rationale:** the lane class (19.1% of d_seg, probe) is the anisotropic long-tail; the
  render-time analytic band achieves **lane d_seg 0.00087 at ~0 counted bytes** (MEASURED,
  FEED-07a; memory `analytic_lane_band_primary_authority_decomposition_20260701.md`) — a
  representation win at zero d_seg-cost, the CHEAPEST lane lever (islands symposium Lens-A
  correction). Compiles into inflate.py for FREE under rule 118 (deterministic generic
  generator; openpilot polynomial prior).
- **DSL:** `--dsl-lever AnalyticLaneRenderBand` (start epoch 350 — DERIVED deconflict; the
  ep300 collision bump was MEASURED 0.0056→0.020, 3.4×, persistent 75+ ep; #302 row 22).
- **Mechanism:** lane leaves the witness's carried class set → the witness stops needing high
  along-tangent frequency → Arm A flips to `lane_offloaded` (small along) → smooth-class
  remainder needs only a SMALL mod-dim. This composition is what dissolves the capacity
  question (FEED-07a).
- **Owed measurements:** **net-S A/B is OWED** (band's own residual + interaction with the
  witness where band and witness overlap; `lever_ledger.json` records it as a treatment arm).
  Split vs training-birth for lane: §6-Q2.

### Arm C — LADDER islands (per the probe PROCEED verdict)

- **Rationale:** islands = 63.9% of flips (UPPER-BOUND caveat above); movable alone is 44.8% of
  d_seg and 93.1% un-born. The islands symposium: soft-gated island loss is MEASURED absorbing
  (lane within-flip −45% while total descended 0.162→0.122); UNIFORM amplification is the
  measured net-negative (full-stack 0.121, paint-seed 0.026) — margin-GATED support is
  net-positive by construction (Lens A, deep-math). The PAIR (birth pressure + nucleus-guarded
  hand-off) is the only config where birth pays (Lens B). Movable = SDF-dilation homotopy:
  **PROVEN transfer, GO independent of the probe**; lane = VP-tangent along-tangent widening
  (manifold-preserving; NOT under the isotropic NO-GO) (islands symposium §#323, wiring
  commit 705afea84).
- **DSL:** `--dsl-lever SeedIslandEased --dsl-lever AmplifyIsland --dsl-lever
  EventTriggeredCurriculum` (+ `SegFocalGamma` per §6-Q4; + `BoundaryDistance` (#301) as the
  council prefers). `EventTriggeredCurriculum` emits `--curriculum-event-triggered
  --curriculum-nucleus-guard` — the #315 pair, verified end-to-end wired (islands symposium
  Lens B), byte-identical when unfired.
- **Config discipline:** FROM-SCRATCH (not warm-start), seg-only (`--w-seg 100 --w-pose 0`,
  per the islands symposium reactivation criteria), A/B vs the live baseline at matched epochs.
- **Owed measurements:** composed-surface ceiling arithmetic FIRST (§6-Q1); then the arm's
  n600 trajectory + per-class within-flip (the `handoff_readiness` telemetry harvests it).

### Arm D — byte-free head (zero-byte rare-class cure; composes with C)

- **Rationale:** textbook rare-class treatment at ZERO bytes (#218; FEED-07b lever 3);
  partially subsumes/composes with LADDER.
- **DSL now:** `--dsl-lever SegFocalGamma` (γ per the MEASURED probe table; choice is §6-Q4)
  + optionally `--dsl-lever MarginFieldHead` / `--dsl-lever PersistenceTopology` (flags EXIST —
  the earlier build-needed binning was a grep false-negative, corrected in FEED-07c #335).
- **BUILD-needed (named task, NOT a stub lever — never-invent-flags):** logit-adjustment
  per-class offset (`LogitAdjust`) — the equations-leg law is registerable now (FEED-07b);
  include in this run ONLY if the small build lands + passes review before launch.
- **Owed measurements:** focal-γ A/B against the analytic-vs-realized-grad-share tension
  (§6-Q4); logit-adjust first measured row.

### Arm E — train-big-compress-small POST-PASS (not a trainer lever)

- **Rationale:** FEED-07a arm D — train mod-48/64, then compress the COUNTED weights at
  Δd_seg=0 through R. Tools exist: #157 sensitivity bit-alloc (COMPLETE, never pointed at our
  checkpoint), #242 flat-minima, #311 TropNNC, KD.
- **Disposition:** post-pass on whatever checkpoint the arms above produce; the deliverable is
  a byte-closed re-measure (rate is already 0.05499, so this pays only if the bigger model's
  d_seg gain survives compression — MEASURED verdict owed, no prediction offered).
- **Mod-dim = SECONDARY, exactly 2 points, NOT a ladder** (FEED-07a binding): {32 = control,
  48 = second point}. The 48 choice is ASSUMED (FEED-07a names "mod-48/64"; council picks).

---

## 3. SCHEDULE / CURRICULUM — the derived (non-PR95-echo) design

Per #302: "the curriculum's remaining PR95-ness is its CLOCK, not its physics." Stage ORDER
CE→tau→Muon is HARD-EARNED (independently re-derived: CE=mirror-descent/NG, tau=Maslov/Γ
dequantization/MCF, Muon=κ-buster finisher that measurably cannot nucleate); the fixed EPOCHS
300/726 are CARGO (cross-run transfers of one trajectory's knees; #302 audit row 3).

**DSL sketch (the #334 first-class object; field↔emission drift-gated by
`verify_schedule_consistency`):**

```python
program = sealed_205_program(...base...)                 # sealed base (control-identical)
curr = sealed_205_curriculum(cfg, handoff="event")       # Curriculum(handoff="event")
levers = [
    EventTriggeredCurriculum(),          # CE→tau on readiness: plateau(rel-eps 1e-4, window 25,
                                         #   min-stage 250 — MEASURED C1 recalibration) AND
                                         #   per-class nucleus predicate π₁=w/σ≳5 (MEASURED knee);
                                         #   fixed epochs become CAPS (never hangs, byte-identical unfired)
    MuonWarmStart(lr_final_frac=0.1),    # kills the MEASURED +8% cold-Muon transient (#270/#272)
    DirectionalBasisRebalance(regime=...),  # Arm A
    AnalyticLaneRenderBand(...),            # Arm B (start 350, boundary-relative when event mode fires)
    SeedIslandEased(), AmplifyIsland(), SegFocalGamma(gamma=...),  # Arms C/D
]
```

Concrete schedule elements (each labeled):

1. **CE→tau hand-off = event trigger + nucleus guard** (MEASURED constants above; #302 §C.ii-1;
   #315 wired per islands symposium). The C1 anchor: eps 1e-3 fires ep151 mid-descent = 15%
   CE-floor loss; 1e-4 separates ep275+ from ep150 (MEASURED, #205 CE trace).
2. **Stagger the ep300 3-way collision** (works even under fixed clock): `--seed-anneal-epochs
   275 --persistence-warmup-epochs 275` — one homotopy parameter per epoch neighborhood
   (DERIVED, #302 §C.ii-3; collision harm MEASURED 3.4×).
3. **Boundary-relative re-anchoring** of persistence-warmup / seed-anneal / β-anneal / band
   start to the FIRED boundary (#302 §C.ii-2) — was a ~40–80 LOC build at symposium time;
   council must confirm current build status before relying on it (do not launch event-mode
   with wall-clock-anchored followers un-re-anchored).
4. **Muon: keep, warm-started, annealed** — Muon −32% vs AdamW (MEASURED fork A/B 2026-06-22);
   `--muon-lr 0.002` (MEASURED), momentum 0.95 / ns 5 (literature-settled, NVIDIA 2606.00371);
   `MuonWarmStart` + `--muon-lr-final-frac 0.1` (DERIVED from the MEASURED +8% transient).
   Muon-as-event-trigger (fire on tau-plateau + nucleus-complete, 726→CAP) was a BUILD (~80
   LOC, #302 §C.ii-5) — same build-status confirmation owed; ep726-fixed is admissible as the
   cap-fallback.
5. **hosc-β anneal:** endpoints DERIVED (β at Muon-fire = 4.00 exactly — run-2 used 1.0→5.134
   for ep726; if Muon becomes event-triggered the endpoint must be recomputed boundary-relative,
   same law). Shape: GEOMETRIC is the DERIVED shape (equal epochs per octave, τ=ε=ħ; #302 B.1);
   `geometric` was NOT in the trainer choices (linear|cosine) — ~10 LOC BUILD; linear is the
   fallback with the derivation gap on record.
6. **Finisher EMA:** `--ema-decay-finisher 0.9995` A/B arm (DERIVED window: ρ_fin ∈
   [0.9995, 0.9998] ≈ 0.1–0.3× finisher steps; 0.997 = 333-step window averages only the last
   ~1.6% of a 274-ep finisher; MEASURED receipt: 78× early shadow lag; #302 B.6). House 0.997
   stays default until a byte-closed A/B (§6-Q5).
7. **Rewarmup 20 ep / floor 0.1 / cosine + reset-moments:** window DERIVED-satisfied
   (1500 steps ≥ 1/(1−β₂)=1000 AdamW moment memory); shape + floor ASSUMED (no derivation;
   #302 row 17) — keep, don't churn blind.
8. **Unchanged derived keeps:** length 0.001 (MCF-erosion driver held small, DERIVED);
   eikonal ramp 0.05→0.10 (DERIVED π_int ≳ 1) — NOTE the control runs eik 0 deliberately; the
   council must decide whether the treatment arm re-engages the ramp (+ `EikonalViscosity`
   adaptive-ε, equation `adaptive_eps_cfl_edge_tracking_v1`) or inherits eik-0; l7 stays
   demoted (measured defect); lr 1e-3→1e-4 cosine (PR95-ECHO acknowledged; needs the sharpness
   probe before churning — #302 row 5).

---

## 4. KNOBS — derived-or-measured values only

| Knob | Proposed | Label | Source |
|---|---|---|---|
| `--w-seg 100` | 100 | DERIVED (exact score coefficient) | score law; #302 row 24 |
| `--w-pose 0` | 0 (seg-only arm) | DERIVED (pose held; store-nothing ξ mover is separate) | islands symposium reactivation criteria; memory `pose_mover_...20260706` |
| `--freq-across / --freq-along` | 32 / 6 (band ON) or 32 / 26 (band OFF) | DERIVED / MEASURED (see Arm A) | `anisotropic_basis_two_regime_allocation_v1`; FEED-07c #335 |
| `--curriculum-plateau-rel-eps / windows / min-stage` | 1e-4 / 25 / 250 | MEASURED (C1 recalibration on #205 CE trace) | #302 §B.2, §C.ii-1 |
| nucleus predicate | π₁ = w/σ ≳ 5 | MEASURED (knee, resolution-portable) | #302 §B.2 |
| `--seed-anneal-epochs / --persistence-warmup-epochs` | 275 / 275 | DERIVED (stagger; collision MEASURED 3.4×) | #302 §C.ii-3 |
| `--muon-lr / momentum / ns` | 0.002 / 0.95 / 5 | MEASURED / literature-settled | #302 rows 19–20 |
| `--muon-warm-start-momentum --muon-lr-final-frac` | on / 0.1 | DERIVED from MEASURED +8% transient | #302 row 21; FEED-07b lever 7 |
| `--muon-start-epoch` | event-trigger if built; else 726 as CAP | DERIVED criterion / CARGO fallback | #302 §B.5, §C.ii-5 |
| `--hosc-beta` endpoints | 1.0 → β(Muon-fire)=4.00 | DERIVED endpoint / shape geometric DERIVED (BUILD) else linear | #302 row 10, B.1 |
| `--ema-decay` / `--ema-decay-finisher` | 0.997 / A/B 0.9995 | QTZ-INH default / DERIVED window (A/B owed) | #302 B.6 |
| `--stage-transition-rewarmup-*` | 20 ep, floor 0.1, cosine, reset-moments | DERIVED-window / ASSUMED-shape+floor | #302 row 17 |
| `--length-weight` | 0.001 | DERIVED (keep small) | #302 row 14 |
| eikonal | council decision: 0 (control-inherit) vs 0.05→0.10 ramp (+adaptive-ε) | DERIVED ramp law; engagement UNKNOWN for this arm | #302 row 13; §3-8 |
| `--seg-focal-gamma` | from MEASURED table (γ=1→0.506 … γ=3→0.633 analytic island share) — choice open §6-Q4 | MEASURED surface, ASSUMED pick | probe JSON `focal_weight_island_share_analytic` |
| `--mod-dim` | {32, 48} — SECONDARY 2-point axis only | control MEASURED / 48 ASSUMED (FEED-07a names 48/64) | FEED-07a binding |
| `--render-h/-w` | 384×512 (grid ≥384 per #220) | MEASURED (renderer requires 384×512-class grids) | FEED-07b lever 1; CLAUDE.md mask-resolution lesson |
| `--adam-beta2` | 0.9999 as a SECONDARY A/B arm only | DERIVED (help-text small-n law), un-A/B'd | #302 row 7, §C.ii-7 |
| `--epochs` | 1000 as CAP (event mode makes epochs caps) | MEAS-weak budget; cap semantics DERIVED | #302 rows 3–4 |
| seed / determinism | `--seed 0`, resumable + per-stage checkpoints, EMA-shadow saves | non-negotiable | CLAUDE.md determinism spine; `launch.sh` control |

No knob above is a bare guess; where a value is ASSUMED it is flagged and routed to §6.

---

## 5. FULL composable-lever inventory (27; `name_composable_levers()`, 2026-07-07)

Recommendation column is the DRAFT's, for the council to overturn per lever.

| Lever | Signal in hand | Draft rec + why |
|---|---|---|
| `AACoverageRender` | "#220 the gate's #1 MEASURED islands lever" (gate verdict, FEED-07b); AA×seed/band/residual INCOMPATIBLE until compose-after-downsample lands (#224 guard raises cleanly) | **SEPARATE ARM only** — cannot compose with Arms B/C today (§6-Q3) |
| `AdamBeta2` | DERIVED small-n law in trainer help (~0.9999999; #222), un-A/B'd | INCLUDE as SECONDARY A/B (0.9999), not primary |
| `AmplifyIsland` | margin-GATED support net-positive by construction (Lens A deep-math); UNIFORM amplify MEASURED net-negative (0.121 / 0.026 arms) | **INCLUDE (Arm C)** — gated form only |
| `AnalyticLaneRenderBand` | lane d_seg 0.00087 MEASURED @ ~0 counted bytes; net-S A/B OWED | **INCLUDE (Arm B)** — cheapest lane lever |
| `BoundaryDistance` | #301 loss-geometry (annulus concentration); designed, part of the non-starving island fix; no standalone n600 number in this draft's evidence set | OPTIONAL in Arm C (council call) |
| `CacheGtSkeleton` | #260 speed lever, bit-identical | INCLUDE (free speed, score-neutral) |
| `CodeSpectralEntropy` | DM1 rate-side conditioning; rate already beaten (0.05499) | EXCLUDE this run — rate not the wall |
| `DirectionalBasis` | lane-edge LOSS lever; lane-only directional −8% MEASURED (vs −48% all-class) | EXCLUDE — dominated by Rebalance + band composition |
| `DirectionalBasisRebalance` | DERIVED two-regime law; −48% all-class anchor VERIFIED; √-optimum ASSUMED_AWAITING | **INCLUDE (Arm A)** — the A/B is the owed anchor |
| `EikonalViscosity` | adaptive-ε law DERIVED (`adaptive_eps_cfl_edge_tracking_v1`); confound-hunt fixes landed; control runs eik 0 deliberately | COUNCIL DECISION (§3-8) — engage only with the ramp |
| `EventTriggeredCurriculum` | #315 wired end-to-end (nucleus guard + plateau, MEASURED constants); byte-identical unfired | **INCLUDE (Arm C / §3-1)** |
| `FiLMFix` | in-code arch lever; no measured number in this draft's evidence set | EXCLUDE (UNKNOWN-to-draft; council may consult registry) |
| `LanePrior` | lane paint prior; paint-seed family MEASURED net-negative (0.026 arm) under uniform support | EXCLUDE — superseded by band + eased birth |
| `MarginFieldHead` | #218/#224 flag EXISTS (`--margin-field-head-weight`); byte-free head partial | OPTIONAL in Arm D |
| `MarginSaliency` | LEVER-4 texture proxy MEASURED INERT (at chance vs through-R reachability); #268 exact-S_R owed | **EXCLUDE until #268 lands** |
| `MicroBatch` | speed A/B, separate per `lever_ledger.json` | EXCLUDE from the score arm (separate speed A/B) |
| `MuonWarmStart` | MEASURED +8% cold-switch transient; flags wired | **INCLUDE (free win)** |
| `PersistenceTopology` | flag EXISTS; persistence loss = erasure-axis cure (designed); warmup must be boundary-relative in event mode | OPTIONAL in Arm C/D (with §3-3 re-anchor) |
| `PoseDecouple` | seg⊥pose measured FREE (memory) | N/A in a w_pose=0 arm; keep for the pose arm |
| `SeedIslandEased` | #323 wired (705afea84); movable SDF-dilation transfer PROVEN; lane VP-tangent not under NO-GO | **INCLUDE (Arm C)** |
| `SegFocalGamma` | MEASURED γ-table on probe surface; tension w/ realized grad share (§6-Q4) | INCLUDE (Arm D), γ per council |
| `SoftBoundary` | no measured number in this draft's evidence set | EXCLUDE (UNKNOWN-to-draft) |
| `StepNativeActivation` | capstone lever #5 UNSWEPT; real-flag route = β-anneal 4→8 (step_basis choice BUILD-NEEDED); non-step best bounds MEASURED (0.004445 / ep450 0.002447) | DEFER to its own sweep arm — do not confound the primary A/Bs |
| `StiefelW` | DM1a conditioning axis (GR unified action), derived; A/B owed | EXCLUDE this run (conditioning not the binding wall) |
| `TauFrozen` | tau-softplus-tau 0.3 MEAS-weak (low-priority A/B, #302 row 12) | EXCLUDE (low priority) |
| `UniWARD` | msal_uni proxy = the same INERT texture surrogate (memory `msal_uni_texture_proxy_inert...`) | **EXCLUDE until #268 exact S_R** |
| `WarpRealLumaFrame0` | byte-close adds bytes, NO d_pose drop (MEASURED); pose mover is store-nothing derive-H | **EXCLUDE** (wrong pose mover) |

Non-composable factories (`Muon(start_epoch)`, `DM1Minimal`) are held by the Curriculum object,
not the lever list.

---

## 5A. Rate/mod-dim scaling analysis (operator-flagged, council to adjudicate)

Presented as the outgoing operator's analysis FOR COUNCIL REVIEW — use or reject.

**MEASURED state.** Current rate **0.05499** (~82.6 KB counted; DAG FEED-07a, byte-closed
archive stat) vs PR95's **0.1188** (178,417 B; DERIVED 25·178417/37,545,489 from the actual
submitted archive, `docs/operating_manual_craft_handoff.md` §4) — **rate beaten at ~46% of
PR95's bytes; d_seg is the wall** (seg term ≈ 0.37 at current best ~0.0036,
`levelset_best.json`). Budget: 0.19110 − 0.055 rate − 0.018 pose(ASSUMED, §1) ≈ **0.118 for
seg** → break-even d_seg ≈ **0.00118**; sub-0.15 needs ≈ **0.00077**. **~0.026 of SPARE rate
budget exists to buy capacity** — OPERATOR-STATED; this draft could not uniquely reproduce
the 0.026 from the score-law arithmetic above (0.1188 − 0.055 ≈ 0.064 is the headroom to
PR95's own rate; other decompositions give 0.018–0.043 depending on the assumed seg landing
point) — **derivation OWED; council to re-derive before spending it** (a number without its
derivation must not become load-bearing).

**HYPOTHESIS (labeled, UNMEASURED): the two scaling curves DECOUPLE.** rate(mod_dim) grows
SLOWLY — mod_dim only widens the Fourier→hidden projection; hidden+head dominate counted
bytes — while d_seg(mod_dim) falls FAST then saturates: mod-19 was frequency-starved on the
along-tangent axis (MEASURED 3.2× deficit, 4-lens memory), and mod-32 clears more of that
band. If true, capacity is nearly-free d_seg. **The council should rule whether to measure
this (the 2-point SECONDARY axis per FEED-07a) or dismiss it.**

**The Whitney tension.** Intrinsic dim ~8 → Whitney ~17–19 bounds **DIMENSIONS**, not
**FREQUENCY RESOLUTION** (unified-flow memory). "mod-32 beats mod-19" is therefore likely
frequency-resolution-limited, not dimension-limited — exactly the open question the 2-point
check answers.

**The FOUR levers for carrying big-mod-dim signal at ~mod-19 bytes** (all in FEED-07a/b;
cross-references only, nothing new invented here):

1. **Basis-match / anisotropy-rebalance shrinks the NEEDED mod_dim** (curvelet
   cartoon-optimality; the live config's `--freq-across 32 / --freq-along 8` is BACKWARDS vs
   the measured 3.2× along-tangent deficit — Arm A).
2. **LADDER = capacity-EFFICIENCY, not capacity-add** (difficulty-gradient spends the limited
   coefficient budget on the hard residual — Arm C; memory
   `ladder_costate_optimal_difficulty_gradient_lane_movable_20260706.md`).
3. **Rule-118 offload** (lane → free band removes the one class demanding high along-tangent
   frequency → the witness carries only the smooth remainder → a small mod_dim suffices —
   "the hard class leaves the building"; Arm B, FEED-07a composition).
4. **Train-big-compress-small** (train mod-48/64 → compress counted weights at Δd_seg=0
   through R: sensitivity bit-alloc #157/#336, flat-minima #242, TropNNC #311, KD — Arm E) —
   decouples train-time capacity from ship-time bytes.

**Irreducible-floor caveat:** part of the residual is TEMPORAL FLICKER that no mod_dim buys
(MEASURED on #205: CE-residual = flicker, 44% of spikes = lane; memory
`witness_converged_to_flicker_floor_leverD_is_path_below_20260703.md`).

## 5B. Pose: ON or OFF in the next run? (operator question, council to decide)

Presented as the outgoing operator's decision frame + recommendation FOR COUNCIL REVIEW.

**FACTS (labeled):**

- The live baseline and the planned islands A/B are **seg-only `w_pose=0` BY DESIGN** (clean
  attribution; `launch.sh` MEASURED; islands symposium reactivation criteria).
- Pose-blind renders have realized **d_pose ~O(100)** (MEASURED operating point 125.833,
  `costate_shadow.jsonl`) → pose term √(10·d_pose) ≈ **31** — catastrophic on any submitted
  row.
- A STORED sidecar does **NOT** lower realized d_pose on a pose-blind render — the scorer
  reads FRAMES (the byte-close tool's own docstring establishes this; DERIVED-from-source).
  Pose cannot be patched on at byte-close; it must be **TRAINED-IN** via a FiLM-conditioned
  render that CONSUMES ξ.
- The store-nothing ξ carrier (derive-H **#257**) is **PROVEN rate-optimal** (~1–2 KB class;
  memory `pose_mover_is_store_nothing_xi_derive_h_not_warp_real_luma_20260706.md`), but
  realized d_pose through a witness FiLM render is **UNMEASURED** (#238/#248 open; the
  ancestor 3.4e-5 does NOT transfer — memory `project_pose_solved_screw_twist...20260701.md`).
- **seg⊥pose MEASURED FREE** (decoupled gradients) — interference risk is low, but the FiLM
  machinery itself is a CONFOUND for A/B attribution.

**THE RECOMMENDATION (outgoing operator's; council to accept/reject): TWO-TRACK.**
Diagnostic A/B arms stay **seg-only** (attribution purity). But the **derived-optimal RUN**
— the one intended to produce the pointer-moving exact row — **MUST train pose ON**
(`w_pose > 0` + FiLM conditioning + store-nothing ξ), because a seg-only run structurally
CANNOT produce a competitive S row (the ≈31 pose term swamps everything).

**Open sub-questions for the council** (also listed in §6): w_pose warmup schedule (pose from
ep0 vs staged-on); whether **#227** (seg⊥pose decoupling MLX port) gates it; whether **#248**
(P-B FiLM read-back, the decisive pose measurement) should run as a BOUNDED PROBE before the
full run.

## 6. OPEN QUESTIONS — the council must rule

1. **Composed-surface ceiling (Q1, blocking for launch-sizing).** The probe's 63.9%/44.8%/19.1%
   shares are an UPPER BOUND (witness-alone surface, d_seg_subset 0.0257 vs composed 0.0049).
   The symposium arithmetic converting within-class un-born fractions (which DO transfer) into a
   composed-surface ΔS ceiling is OWED. Without it, "islands = THE measured d_seg target" is a
   direction, not a size.
2. **Lane: band vs training-birth split (Q2).** Rule-118 band delivers lane 0.00087 at render
   time; training birth (SeedIslandEased lane channel) treats what the band doesn't cover. What
   is the residual lane mass under band-ON, and is lane training-birth then worth its
   interaction risk? Draft default: band ON + movable-only birth first; lane birth as the
   follow-on arm. Council decides.
3. **AA-render compatibility (Q3).** `AACoverageRender` is the #220 "#1 measured islands lever"
   but is INCOMPATIBLE with seed/band/residual until compose-after-downsample lands. Priority
   call: fund the compose-after-downsample build, or run AA as its own arm?
4. **Focal-γ choice (Q4).** The ANALYTIC island weight share rises with γ (γ=1→0.506,
   γ=3→0.633) but the probe's REALIZED grad share on the ep225 surface FALLS under focal
   (current island 0.1268 → focal_g1 0.1044 → focal_g3 0.0973; `grad_share` block). Which
   surface is the calibration authority, and does focal belong in a birth arm at all (focal
   re-weights EXISTING errors; un-born islands may need seed/amplify, not focal)?
5. **EMA π-group (Q5).** Adopt `--ema-decay-finisher 0.9995` in this run's A/B, or hold the
   Quantizr 0.997 house value everywhere until a byte-closed A/B? (Deployed-checkpoint authority
   unchanged either way.)
6. **Eikonal engagement (Q6).** Control is eik-0 by design; the derived ramp (0.05→0.10,
   π_int ≳ 1) + adaptive-ε exists. Does the treatment arm re-engage it (interface protection for
   born islands under tau) or stay eik-0 to preserve A/B cleanliness?
7. **Build-status confirmations (Q7).** Boundary-relative re-anchoring (§3-3), Muon event
   trigger (§3-4), geometric β (§3-5), LogitAdjust (Arm D): each was a named BUILD at symposium
   time; the council must confirm what has since landed (the islands symposium already upgraded
   #315 to fully-wired) and refuse any launch that depends on an unlanded build.
8. **Concurrency + machine safety (Q8).** How many arms run concurrently under the memory
   governor (P0: concurrent >128GB crashed the box; memory
   `machine_crashing_risk_is_P0_hard_gate_...`)? Draft assumes: treatment arm waits for
   operator GO; nothing fires autonomously (containment non-negotiable).
9. **Rate/mod-dim decoupling (Q9, §5A).** Measure the decoupling hypothesis via the 2-point
   secondary axis (mod-32 vs 48, tracking counted bytes AND d_seg per point), or dismiss it
   as dominated by the four capacity-efficiency levers? If measured: same-arm or separate?
10. **Whitney vs frequency resolution (Q10, §5A).** Does the council accept the framing that
    the mod-19→32 gain was frequency-resolution-limited (not dimension-limited)? The 2-point
    check + Arm A's rebalance A/B jointly answer it — confirm the attribution design.
11. **Pose ON/OFF (Q11, §5B — the operator question).** Accept/reject the TWO-TRACK
    recommendation: seg-only for diagnostic A/Bs, pose ON (w_pose>0 + FiLM + store-nothing ξ)
    for the derived-optimal pointer-moving run.
12. **Pose staging (Q12, §5B).** If pose ON: w_pose from ep0 vs staged-on (warmup schedule);
    and does the fixed-linear w_pose under-weighting near d_pose→0 (#302 row 24: the score's
    local gradient 5/√(10·d_pose) diverges) need a derived schedule in the same run?
13. **Pose gates (Q13, §5B).** Does #227 (seg⊥pose decoupling MLX port) gate the pose-ON run,
    and should #248 (P-B FiLM read-back — the decisive realized-d_pose measurement) fire as a
    BOUNDED PROBE before committing the full run?

---

## 7. EXPLICIT DEFERENCE

This draft binds nothing. It is assembled evidence + one candidate composition, produced so the
symposium has a concrete object to attack. The council designs the run, sets every knob, rules
on §6, and signs off; the operator gates any heavy launch. Where this draft and a council
verdict differ, the council verdict wins. Where a cited number's source artifact disagrees with
this draft, the artifact wins — recompute from it, don't trust the transit copy.

*Round-1 adversarial self-review performed before commit (internal consistency, citation
resolvability, no unlabeled/borrowed numbers); catches folded in: (a) the pose term p=0.018 in
the break-even arithmetic was initially unlabeled — now explicitly ASSUMED/borrowed-hypothesis
with the witness-pose-OPEN caveat welded on; (b) `EventTriggeredCurriculum`'s nucleus guard was
a BUILD in #302 but fully-wired per the later islands symposium — dated both, kept the later
with the earlier on record (Q7 generalizes the class: confirm every build's live status);
(c) the focal-γ analytic-vs-realized-grad-share contradiction inside the probe JSON was
initially reported one-sided (analytic only) — now both, promoted to Q4; (d) in the
operator-directed §5A, the "~0.026 spare rate budget" could not be uniquely re-derived from
the score-law arithmetic — labeled OPERATOR-STATED with the derivation explicitly OWED
(Q9-adjacent) rather than silently laundered into a DERIVED number.*

## §14 OPERATOR DIRECTIVE (2026-07-07, BINDING on this symposium) — design the SCHEDULE, not just the lever set

Operator verbatim: *"It will be very important for the grand council symposium to consider this
naive inherited stack when designing the next run, not just which levers are active but when they
are activated at which levels and how the levels should change over time and which stages when
and which should be repeated and what needs priming and how to drive max convergence across the
run in shortest wall clock prioritizing leaving no meat left in the bone even if the run takes
long."*

Objective ordering this fixes: **completeness-of-convergence DOMINATES wall-clock** (leave no
meat, even if long); wall-clock is minimized SUBJECT TO that (consistent with the lexicographic
training-time law, sharpened). The design space the council must cover explicitly, with the
naive-inheritance item each replaces:

1. **ACTIVATION TIMING per lever** (naive: on-at-ep0 or off): when each lever ENTERS — epoch vs
   EVENT (#315 machinery is built). Evidence to consult: basis-match is prior-to-capacity (from
   ep0); island seeding wants CE-stage nucleation (#300, paint-then-SDF #291); focal-γ has a
   pre-registered fire criterion (ep50-100 slope flattens, #301); Muon-from-CONDITIONING (#302:
   enter when tau's conditioning stalls, not at inherited ep726 — the live control just measured
   tau saturating ~70 epochs BEFORE the fixed boundary, i.e. the naive schedule left ~70 wasted
   epochs on the table).
2. **LEVELS AS PATHS λ(t), not constants** (naive: fixed weights): per-class-λ homotopy (LADDER),
   eikonal ramp/adaptive-ε (#320 DE law), Γ-optimal geometric τ-anneal (#286), amplify-island
   anneal, EMA decay per stage (the .997 constant is itself PR95-inherited; #302 flags the
   π-group). Per the different-stages-different-treatment law, every level should have a declared
   path per stage, even if the path is constant — constancy must be a DECISION, not a default.
3. **STAGE SET, ORDER, REPETITION** (naive: one-shot CE→tau→Muon at 300/726/1000): the deep-math
   frame (curriculum = coarse-to-fine persistence/annealing, #284) admits CYCLES — e.g. re-enter
   tau after Muon flattens, or repeat a Muon+leap-residual (#217) finishing block until measured
   exhaustion. PR95's one-shot ladder is the inheritance; repetition-until-dry is the no-meat
   version. Council: rule per stage.
4. **PRIMING** (naive: cold everything): Muon warm-start momentum (#269 — the control is paying
   the cold-start cost live right now via 8-ep rewarmup); structured/seeded init; FINER++
   bias-init (activation priming); warm-start-from-mod32cap-ckpt vs from-scratch (the islands
   symposium's warm-start verdict); the amortized pre-seeding question (#211) as the long-horizon
   version.
5. **MAX-CONVERGENCE TERMINATION** (naive: fixed 1000 epochs): per-stage EVENT exits (plateau
   classifiers exist: SC1'/#315; the costate shadow's slope detectors) + a run-end criterion of
   MEASURED exhaustion of every active lever (each lever's marginal Δd_seg/epoch below a
   pre-registered floor) rather than an epoch count. "No meat left" = every stage exits on
   evidence, not on schedule.
6. **WALL-CLOCK, SUBJECT TO 1-5**: per-lever s/ep costs are MEASURED (#306 audit; new datapoint:
   Muon NS-5 adds ~0 s/ep — 107.3 vs ~106 tau baseline, mtime-measured on the live control), so
   the shortest-wall-clock composition under completeness is a solvable allocation, not a guess.
   Verdict cadence (+16 s/ep every other block) is itself a schedulable knob.

Q14 for the council: which of 1-6 rides the DSL's EventTriggeredCurriculum now vs which needs a
new Schedule primitive (the DSL holds Curriculum/Schedule as first-class objects, #334 — the
council's design should compile through them, not through hand-set epochs).
