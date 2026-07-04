# MASTER LEVER LEDGER — the fresh seeded run (synthesis of sweep A/B/C + the 5-facet scaling-law pass)

**2026-07-04. Operator: "max signal and paranoia sweep of all triality and codebase and omx and task
and catalogue index and everything for all levers... insanely detail oriented and meticulous."** Merges
`sweep_{A,B,C}_*_lever_ledger_20260704.md` + `scaling_law_facet{1..5}_*_20260704.md` +
`lane_nucleation_failure_seed_above_critical_nucleus_20260704.md`. Every GO/NOGO cites its artifact.
**Pointer contest-CPU 0.19110 UNMOVED; witness implied-S ~0.67-0.75. ALL MEANS** — no lever has produced
a byte-closed exact row below pointer. The fresh run is the vehicle to change that; this ledger is its
optimal-form gate.

## 0. The three-surface convergence (the paranoia paid off)
Sweep A (189 trainer flags), sweep B (triality/221 equations/26 gauges/memories), sweep C (tasks/orphans)
**independently converged** on: (a) #205 launched at the SUBOPTIMAL value on EVERY fresh-run lever; (b)
the lane is seeded at part_frac=0 (nucleation failure); (c) `--length-weight` must stay SMALL (it IS the
MCF erosion term); (d) the acceptance gate is MEASURED `part_frac[lane]>0` at ep0, not flag presence.
**Reconciliation applied:** facet-3 (landed after A+B+C) OVERRIDES all three sweeps' stale "+2px dilation"
with **paint-then-SDF** (+2px measured FP-costly ~15:1; `--mode replace` measured NO-OP).

## 1. THE CALIBRATED FRESH-RUN CONFIG — PRIMARY levers (one coherent seeded run)
Per sweep-A's discipline (clean attribution): the fresh run carries ONLY the seed-fix + the 4 geometry
primaries; secondaries run as subsequent ISOLATED A/Bs (§3). All flags grep-verified in the levelset trainer.

| lever | #205 → fresh | class | grounding |
|---|---|---|---|
| `--seed-islands` | OFF → **ON** | INCLUDE | the nucleation fix; amplify was a no-op without it (sweep-A #1) |
| **SEED mechanism** | replace → **paint-then-SDF (#291 BUILD ~10 LOC)** | BUILD→INCLUDE | facet-3: `replace` NO-OP; paint → lane-FN 0.0058→0.0019 (3×) |
| `--island-dilate-px` | 1 → **KEEP 1 (NOT 2)** | INCLUDE | facet-3: +2px FP-costly ~15:1 — survival gained < FP lost |
| `--eikonal-weight` | 0.01 → **0.05** | INCLUDE | facet-4 interface-width control; holds the thin edge sharp at τ/2 |
| `--length-weight` | 0.001 → **KEEP 0.001** | INCLUDE | facet-4 + sweep-A/B/C: it IS the MCF surface-tension erosion term — do NOT raise |
| per-class AREA constraint (auction-MBO) | — → **BUILD** | BUILD→INCLUDE | the principled hold vs the 95.7%-Lane MCF erosion (sweep-B orphan) |
| `--tau-anneal-shape` + `--softmax-temp-end` | cosine/0.05 → **geometric / 1.0** | INCLUDE ✅ | facet-4 adiabatic Fisher-Rao geodesic; **τ_end=1.0 MEASURED the resolution floor** (0.05 = 20× sub-pixel aliasing) |
| `--mod-dim` | 32 → **19** | ✅ **MEASURED-SAFE** | m≈7.15 (TwoNN) = scene-PR 7.29; Whitney 2m+1=15.3 < 19 → no cliff; 32 = waste |
| `--bank-n-scales` | 4 → **6** | INCLUDE (slope-gated) | facet-2: use full stem-Nyquist f_max=64 |
| `--film-stiefel` | OFF → **ON** | INCLUDE | facet-2: byte-free rank fix, PR 1.19→4.57 |
| `--muon-warm-start-momentum` + `--muon-lr-final-frac 0.1` | OFF/1.0 → **ON/0.1** | INCLUDE | #270/facet-1: kills the cold-start spike + escapes flat-LR plateau |
| `--lane-band-start-epoch` + rewarmup | 300/8-linear → **350/20-cosine** | INCLUDE | Ch.6: deconflict the MEASURED ep300 collision (3.4× harm) |

**KEEP from #205 (correctly optimal):** render 384×512 · `--hosc` ANNEALED β1→4 (NEVER fixed) ·
`--self-orient --n-dir-freqs 2` · `--render-aa none + --lane-render-band` · palette-anchor · persistence
1.0/warmup 300 · `--verdict-batch 32` · `--cache-gt-skeleton` · muon@726 · pose-carrier generated (store-nothing).

**ACCEPTANCE GATE (sweep-A #2, binding):** the run is only valid if the ep0 structured_init log shows
**`part_frac[lane] > 0` (≈0.006)** — measured, NOT inferred from flag presence. Abort+fix if lane_px=0.

## 2. THE CONTROLLER (facet-5) — the self-converging safety net ✅ BUILT (3db114735)
`tools/witness_control_monitor.py` (✅ BUILT + 7 tests; de-orphans the trajectory instrument #188/#216):
follows the verdict log, computes the **τ-creep detector** (`r̂≥+δ ∧ net_Δd_seg>0 ∧ ep_loss↓` = the #205
signature) + a **Lyapunov early-termination** (OT-dual gap V_OT PROVEN tier / EWMA descent-rate MEASURED
tier), classifies 4-value (converging/plateau/diverging-ERASING/volatile), and **EMITS decisions +
config-diffs ONLY** — never launches (CONTAINMENT + P0 governor). Training stays BIT-IDENTICAL. This is
the mechanism that would have caught #205's erosion at ep325 instead of ep425.

## 3. SECONDARY levers — ISOLATED A/Bs (NOT stacked in the primary run)
Each MEASURED-GO or ready, folded as its own clean arm post-primary: **chroma** (#227 GREEN, verdict-BLOCKING
— the whole d_seg ledger is provisional until A/B'd) · `--n-dir-freqs 2→4 --freq-across 8` (#277, the #1
along-tangent lever, Nyquist-capped) · **OT head-offset b\*** (#288, built; $0-gate vs Menon) ·
`--margin-saliency-reachability` (#268, replaces the INERT texture proxy; needs sR gt_n600 cache) ·
`--seg-spike-downweight` (#274) · vector-t + chroma-boundary sub-pixel.

## 4. THE EXPONENT BET (conditional — gated by ONE $0 probe)
Facets 1+2 agree: everything deployable changes the CONSTANT; the only EXPONENT levers are **NTK
feature-Gram whitening** (facet-1) + **parabolic-spatial-shearlet front-end** (facet-2), and they only pay
if the lane-dash tail is spectrum-limited (not the STE flicker floor). **DECISIVE $0 gate first:** the
GT-margin N-term log-log SLOPE probe (`pre_metric_nterm_basis_slope.py` SPEC) + the persistence/margin
histogram. Build the exponent levers ONLY if the slope is power-law over the surviving flip mass.

## 5. PRE-LAUNCH CHECKLIST (gates the governed stop-and-launch)
**BUILD:** (1) `--lane-prior-phi1-mode paint` #291 — ✅ **DONE 4f1580d0c** (real-GT smoke: part_frac[lane]
0→0.0064, lane_FN 0.00713→0.00211 3.4×; 5 unit tests; byte-identical default) · (3) `tools/witness_control_monitor.py`
— ✅ **DONE 3db114735** (flags DIVERGING_ERASING on the live #205 log; 7 tests; decision-only). · (2) per-class
area constraint (auction-MBO) — ✅ **DEFERRED, MEASURED-justified (not signal loss):** the $0 MCF-survival
pre-check (gt_n6, per-channel σ-smoothing of the paint-seed phi + re-argmax) shows **native paint retains 93%
of the nucleated lane at σ0.8 (the eikonal-0.05-sharpened regime) vs only 52% at σ1.5 (raw MCF, no eikonal)** —
so **paint + eikonal-0.05 holds the lane at native mass with no FP inflation**; +2px retains 98% but at 2.7× the
mass (the FP cost). The eikonal is the critical survival partner; the area-constraint is deferrable. Caveat
(NO-FAKE): σ0.8↔eikonal-0.05 is a directional PROXY, not an exact prediction — the real run + the monitor (3)
are authority; build the area-constraint IF the fresh run's tau stage erodes despite paint+eikonal.
**PRE-LAUNCH BUILD QUEUE CLOSED: 2 built (paint, monitor) + 1 measured-deferred (area-constraint).**
**MEASURE-first ($0, gate the config values) — ✅ ALL DONE:** (a) `part_frac[lane]>0` acceptance smoke on
the paint-seed ✅ (0.0064, nucleated) · (b) mod-dim intrinsic-dim gate ✅ **MEASURED: combined nonlinear
m≈7.15 (TwoNN) = scene linear-PR 7.29; Whitney 2m+1=15.3 < 19 → mod-dim 19 SAFE, NO capacity cliff, mod-32
= waste** · (c) `--softmax-temp-end` floor ✅ **MEASURED: τ_end=1.0 is the resolution floor (only τ where the
discrete EDT-SDF boundary enters a resolvable soft transition, 2.83% band); τ≤0.5 = hard sub-pixel step
(0% band, aliasing) → use 1.0, NOT the 0.05 default or 0.25** · (d) paint-seed init d_seg ✅ (lane_FN 0.00211).
**APPARATUS hygiene (sweep-B, fix so the run's records are trustworthy):** (I) **flush the 6 code-live
2026-07-03 lever laws to the JSONL registry** (currently JSONL=0 → the equations leg is blind to the
fresh-run levers) · (II) reconcile the mod-32-vs-19 record · (III) note the gauge accessor layer is
DSL-symbolic not launch-path (witness_autoconfig sets flags directly) · (IV) the Catalog #344 strict-gate
backlog (0→480) is a separate hygiene debt, non-blocking for the run.

## 6. STALE / SUPERSEDED — do NOT re-open (frees attention)
Lever-D flicker coder (#280 MEASURED NO-GO b=0.876>0.65) · deconv/pre-emphasis/matched-filter/brute-AA
(measured-negative) · β₂ mis-anchor (0.9999999→0.999) · 198:1 anisotropy (disputed→9.56/37.8) · "level-set
has no exact-eval path" (RESOLVED — `levelset_byte_close_and_eval.py` exists) · #248 P-F pose optimum
(RETRACTED false-positive) · l7-as-default (DEMOTED) · mx-compile (flips argmax, fails closed) ·
margin-saliency-UNIWARD (INERT) · film-per-layer/concat (dominated by stiefel).

## 7. COMPLETENESS CRITIC (what the 3 sweeps + 5 facets could still have missed)
- **POSE is OPEN on the witness** (sweep-B #5): warp-alone d_pose 1.37-10.53; 3.4e-5 is the abandoned
  ANCESTOR, never witness-validated. The fresh run's pose rides the store-nothing carrier (generated), but
  the WITNESS d_pose through byte-close is UNMEASURED (#238 owed). Do NOT claim pose solved; the run's S is
  d_seg-primary, pose-provisional.
- **The exact-eval loop is the ONLY authority** — every lever above is advisory/field-level/$0 until the
  byte-closed `upstream/evaluate.py` n600 row. The fresh run's FIRST milestone after convergence is a
  byte-closed row, not another verdict.
- **Stacking risk:** even the 4 primaries interact (seed × schedule × eikonal × dim). The `witness_control_monitor`
  τ-creep detector + the per-stage checkpoints are the guardrail; if the primary run creeps, the monitor
  flags it early and we isolate.
- **The seed's smooth-class debt (0.023, 87%)** is trainable-away via the ξ per-pair warp (seg-seed =
  pose-seed, #194) — but that warp integration (#194) is IN-PROGRESS, not landed; the fresh run may carry
  the smooth debt as a transient the flow removes, or wait for #194.

## 8. THE LAUNCH SEQUENCE (operator-approved "after A/B/calibration")
BUILD (§5) → MEASURE-first $0 gates (§5) + APPARATUS flush (§5.I) → **10-second final config-confirm** →
memory-preflight through the governor → PRESERVE + cleanly stop #205 (CE-best kept) → LAUNCH the primary
seeded run via the governed launcher (resumable, per-stage, `witness_control_monitor` attached) →
byte-closed exact row → secondary isolated A/Bs (§3). NO autonomous heavy launch; operator GO + P0 governor
gate the actuation. **HARD GATE: pointer 0.19110 UNMOVED until a byte-closed row proves otherwise.**
