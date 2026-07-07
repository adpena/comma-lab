# Adversarial review of ALL load-bearing negative findings — 2026-07-07

**Reviewer:** fresh-eyes adversarial reviewer, REPORT-ONLY (operator directive verbatim: *"Need
adversarial review against all negative findings"*). Live #205 run untouched (read-only; pid alive
throughout; verdicts read from the log symlink target). Every number below re-derived from PRIMARY
artifacts (JSONs / tool code / run.log), never from memo summaries. Axis: everything here is
`[macOS-CPU advisory]` NON-PROMOTABLE analysis of advisory rows. **Pointer contest-CPU 0.19110
UNMOVED — this review is MEANS.**

**Discipline applied per negative (L3 verdict-clearance, Catalog #307, kill-last-resort):**
(a) apparatus-validity — was the instrument measuring? (b) implementation-vs-paradigm scope;
(c) untried canonical fixes; (d) over-generalization check; (e) arithmetic re-derivation.

## Verdict table (ranked by council impact)

| # | negative | FINAL verdict |
|---|---|---|
| 1 | τ-crossover trainflow FLAT-H (513ea349d) | **OVERTURNED-AS-INSTRUMENTED** — the H index has ≈zero dynamic range (GT-H control); correct verdict is the pre-registered INDETERMINATE |
| 6 | Muon cold-start finishing shortfall (projection) | **CONFIRMED** (arithmetically sound; will not beat ep650 by ep1000) + one borrowed number in the council draft (+8% should be the live-measured ≈+29%) |
| 2 | #341 subset-solve NO-GO | **CONFIRMED at the measured point; scope WEAKENED** — implementation-level for the un-regularized K=8 tool form; "only full-P" prose is stronger than the evidence (and stronger than the equation's own reactivation clause) |
| 4 | Lever-D #280/#307 flip-residual NO-GO | **CONFIRMED** — survives all four attack surfaces; correctly scoped implementation-level with a training-outcome reactivation criterion |
| 3 | LEVER-4 msal_uni texture proxy INERT | **CONFIRMED (strengthened)** — inversion attack fails; sharper mechanism identified (scale-free normalization cancels smooth within-band reweightings) |
| 5 | Pose warp-real-luma byte-close negative | **CONFIRMED-AS-SCOPED** — carrier-level; no wrong-object citation found in the council draft; store-nothing start is separately MEASURED (1.095 @ep2, descending) |

---

## 1. τ-crossover trainflow FLAT-H — OVERTURNED-AS-INSTRUMENTED

**Claim as stated** (memo `.omx/research/tau_crossover_trainflow_probe_20260707.md`; JSON
`experiments/results/tau_crossover_trainflow_20260707/tau_crossover_trainflow_n600_20260707.json`):
H = P(realized==lane|comb-gap)/P(realized==lane|comb-mark) FLAT at 0.666–0.677 across τ 0.806→0.216,
per-band frozen (0.66/0.55 near, ≈1.0/0.9 mid-far) ⇒ *"SUPPORTS-R-Nyquist-bound + pinned-interface;
NO τ-crossover"* ⇒ operational: *"no reachable τ-anneal buys dash resolution; the #287 comb is the
only live repair path."* Appended as an anchor on `dash_erasure_homogenization_v1`.

**Apparatus-validity check — FAILED (the missing calibration).** The mark/gap regions are purely
analytic (per-pair line fits × global ego-phase comb) and are NEVER conditioned on GT
(`tools/tau_crossover_trainflow_probe_n600.py:222-224,242-243`: `gap_lane_ct = (r_lane & gap)`).
The probe therefore lacks the one positive control that validates the INDEX (its registered
"internal positive control" only bit-matched a prior arm = reproducibility, not validity). I ran
that control at $0 (n600, READ-ONLY, identical machinery/regions, peak RSS < 3 GiB): **H computed
on the GT labels themselves** — the perfect-witness limit.

**GT-H control result** (`experiments/results/tau_crossover_trainflow_20260707/gt_h_control_n600.json`,
rebuildable via `gt_h_control.py` beside it):

| | aggregate | band0 (δ 110 px) | band1 (19.8 px) | band2 (5.4 px) | band3 (1.9 px) |
|---|---|---|---|---|---|
| **GT-H** (realized ≡ GT) | **0.7015** | **0.669** | **0.593** | **0.994** | **1.040** |
| witness H (all 4 ckpts) | 0.666–0.677 | 0.663–0.671 | 0.550–0.559 | 0.975–1.018 | 0.830–0.923 |
| GT r_mark / r_gap (band0) | | 0.629 / 0.421 | | | |
| witness ep925 r_mark/r_gap (band0) | | 0.612 / 0.411 | | | |

**A perfect dash-resolving render would read the SAME "flat" H the witness reads.** The measurable
range of the index in band0 is [0.67 (=GT), 1.0 (solid)] and the witness sits at the GT end at
every τ (band1 slightly BELOW GT); bands 2/3 have **zero instrument range** (GT-H ≈ 0.99/1.04).
Consequences, each overturning a reading of the memo:

- *"Near bands hold PARTIAL dash contrast... pinned"* — misread of an uncalibrated index: on the
  calibrated scale the witness has **FULL (GT-level) dash contrast** in bands 0/1 at every
  checkpoint, ep299 included.
- *"Mid/far bands fully homogenized = the pure R-Nyquist bound"* — vacuous on this instrument
  (GT-H is also ≈1 there; the comb/footprint cannot separate marks from gaps at that scale even
  for GT). R-Nyquist retains only its independent arithmetic support (δ_along vs pixel pitch).
- *"NO τ-crossover ⇒ τ-anneal buys zero dash resolution"* — the index had nothing left to buy
  from ep299 onward; flatness carries **no information about τ** in either direction. The
  pre-registered vocabulary contained the correct verdict: **INDETERMINATE-at-this-resolution**
  (instrument-bounded).

**Why GT-H ≈ 0.67 (either reading suffices for the overturn):** (a) comb/footprint
misregistration (global ego-phase fit + 1 px softness leaves 42% GT-lane inside nominal "gaps" in
band0), and/or (b) the GT LABEL field itself is substantially solid (SegNet's stride-2/receptive
field paints lane across physical paint gaps). Under (b), matching GT labels — the actual d_seg
objective — does not require resolving physical dashes where GT doesn't, which further undercuts
the operational claim.

**Implementation-vs-paradigm:** instrument-level only. The `dash_erasure_homogenization_v1` LAW's
five HELD anchors (dash-gap FP 0.00396, MBO 95.7%, annulus pinning, 3.2× along-tangent deficit,
ego-phase) are separate rows and are NOT touched. The raw GT-conditioned `dash_gap_fp` rows in the
probe JSON remain valid data (0.000191→0.000124→0.000190, amplitude-tracking, as pre-registered
amplitude-confounded).

**Untried canonical fixes (the repaired instrument):** GT-conditioned mark/gap masks (dash runs /
gaps extracted from the GT lane raster along the fitted centerline, not the analytic comb);
per-pair comb-phase refit; report the calibrated statistic `H_witness − H_GT` (currently ≈0
everywhere) or a GT-conditioned contrast. Then re-run — the frozen snapshots are already on disk.

**Collateral flag (owed before the n287_dash_comb arm fires):** the same GT-H numbers say the
ego-phase comb only weakly separates GT marks from gaps (0.63 vs 0.42 in band0). A decode/train-
time comb gate mis-phased at that level risks suppressing lane where GT HAS it. The comb-probe's
own measured mechanism row ("comb removes 86% of the solid band's ADDED dash-gap FP") is a
GT-conditioned statistic and is NOT overturned — but a GT-conditioned comb-registration audit
should precede any in-training comb activation.

**Registry correction spec (append-only; do NOT mutate the existing anchor):** append a
supersession anchor to `dash_erasure_homogenization_v1`: `anchor_id=
"tau_crossover_anchor_instrument_bounded_supersession_20260707"`, empirical_output = the GT-H
table above + "the 2026-07-07 tau-crossover anchor's FLAT-H verdict is re-classified
INDETERMINATE-instrument-bounded (H has ≈zero headroom: witness ≡ GT on the index at every τ/band);
the τ-crossover leg of the law returns to OWED pending a GT-conditioned index + the fixed-τ
control arm", source_artifact = this memo + `gt_h_control_n600.json`, verification =
VERIFIED_VIA_EMPIRICAL_ANCHOR. Council-draft §19 consequences 1–2 lose this anchor (see the
addendum appended to the draft); consequence 3 (fixed-τ control arm) SURVIVES and is now the
primary discriminator.

## 2. #341 subset-solve NO-GO — CONFIRMED at the point, scope WEAKENED

**Re-derived:** transfer law (8/600)(−3.36%) + (592/600)(+5.17%) = **+5.06%** ✓ = the measured
net (`reports/basin_finisher_probe_20260707.json`). In-subset proxy→verdict 1:1 through int8
deploy (−3.3% → −3.4%) ✓ — quantization and surrogate excluded as mechanisms; 546/600 worse =
genuine held-out damage. Apparatus liveness + positive control (in-subset improvement) present.

**The attack that lands:** the solve was Levenberg-damped (λI on H per CG step) — damping
stabilizes the STEP, it does not bound the total displacement from θ0. The canonical small-K
overfit cures were NOT tried: (a) proximal/ridge term toward θ0, (b) held-out-pair early
stopping, (c) any K other than 8 (one subset, one seed). One un-regularized K=8 point cannot
identify in(K)/out(K), so *"the only admissible solve is FULL-P"* (memo/JSON prose) is not
evidence-backed — and is stricter than the registered equation's own reactivation clause, which
already allows *"a measured-generalizing K"*
(`src/tac/canonical_equations/quadratic_head_chart_subset_solve_gap_20260707.py:150`).

**Narrower true claim:** the un-regularized K=8 head-solve overfits (+5.2% held-out); the
post-run CPU subset TOOL AS DESIGNED is NO-GO; intermediate-K + proximal-regularized forms are
UNMEASURED. Named cheap follow-up (not owed for the council decision — the draft already routes
priming in-trainer): ridge-to-θ0 K∈{32,128} ladder, ~1–11 h CPU (or minutes on the 17× GPU path)
using the same probe tool. Positive sub-findings (ρ≈0.85 quadratic chart, 1:1 transfer,
0.7%-reconstructible self-orient state) verified as stated.

## 3. LEVER-4 msal_uni texture proxy INERT — CONFIRMED (strengthened)

**Evidence verified on two independent surfaces** (`.omx/research/sr_reachability_weight_build_20260705.md`
+ trainer source `experiments/train_levelset_witness_realized_through_R_mlx.py:4005-4013`):
map-correlation vs cached through-R sR: Pearson −0.044 ± 0.030 (n24; replicates the original
−0.033), and ep100 total-gradient shares: tex ≡ plain (island/boundary/interior identical).

**Inversion attack (the dashboard-reciprocal concern) — FAILS:** the measurement compared the
trainer's ACTUAL multiplier `sal/(1+β·tex)` (same convention as the code; no display-layer
inversion in the measurement path), and an inverted cost map only flips the correlation's sign —
|r| ≈ 0.04 either way = chance. Same-field/resolution check passes (tex is computed from the same
rendered `_f1` on the SegNet grid the flips live on).

**Sharper mechanism (INFERRED from source, labeled as such):** the term is a sal-weighted MEAN
(`sum(hmap)/sum(sal)`) — scale-free — so any multiplier that is approximately constant WITHIN the
hinge-active fragile band cancels by construction. The max-normalized luma-gradient tex is smooth
within the boundary band → near-cancellation; sR acts precisely because it VARIES within-band
(3.02× band concentration). Consequence: "better texture normalizations" are predicted inert too;
only within-band-varying weights can act. The broader sentence "texture carries no reachability
signal" is not established (wavelet-residual UNIWARD cost proper untried) — but nothing
load-bearing rests on it; the sR replacement stands.

## 4. Lever-D #280 (+ #307 re-confirmation) — CONFIRMED

All four attack surfaces from the review charter fail against the primary artifacts (DAG
FEED-03v/03w/03x + `.omx/research/contour_string_flip_coding_n600_20260707.md`):

- **"Naive coder?"** No — FOUR coders measured: bz2 BWT+MTF+RLE 0.876, H_k context arithmetic
  (margin-decile × causal-spatial × temporal-prev-1/2) 0.925, colex 1.384 (#280, n32 gate), and
  the stronger contour chain-code + range coder **0.820 B/flip at full n600, decode-verified
  bit-exact** (#307). The published 1–1.5 bit/contour-px floors assume long coherent strings; the
  measured residual is confetti (mean component 3.1 px; 38.5% of flips in ≤3 px components;
  anchors alone 0.37 B/flip) — the premise, not the coder, fails.
- **"Stale baseline?"** No — #307 re-measured at the CURRENT residual (ep425 snapshot, byte-close
  render authority d_seg 0.003741), both surfaces.
- **"Survival de-conflated?"** Yes — #280 measured recovery separately from rate: r_global −0.106
  (collateral-dominated), EDT-attributed fair-admission r_admit +0.198 vs break-even 0.688 at
  b=0.876. Rate AND recovery fail independently.
- **"Where does the 0.65 bar come from?"** Re-derived: bar = σ_eff × WATERLINE = 0.51 × 1.273108
  ≈ 0.65 — it embeds the MEASURED best-decile induce survival, not an arbitrary margin. (Raw
  waterline check: b=0.820 < 1.273 would be net-positive ONLY at ≈100% recovery — measured false.)

Residual untried surface: the variant-B coded-RGB-delta induce; with r_admit 0.198 ≪ 0.688 even a
2–3× better induce does not close, so the NO-GO scope is sound. Classification as stated
(implementation-level; reactivation = the residual becoming ~3× more coherent, a TRAINING
outcome) is correct and stands.

## 5. Pose warp-real-luma byte-close negative — CONFIRMED-AS-SCOPED

The catastrophic START d_pose (2.562–12.66) belongs to the **warp-real-luma PCAR carrier**
(carrier-level negative, already reclassified; the byte-close tool's own docstring records it).
The store-nothing ξ path has its OWN measured start — axis-9 smoke 2026-07-03
(`reports/sn205_axis9_byteclose.json` + `witness_205_axis9_smoke_result_20260703.md`): d_pose
1.887 (untrained v0) → **1.095 (ep2, descending)**, frame0 decode bit-exact, warp ceiling 1.367,
w_pose=1.0 active. **Wrong-object citation audit of the council draft: CLEAN** — the draft labels
p=0.018 ASSUMED/borrowed-hypothesis, states d_pose OPEN+UNMEASURED through a trained FiLM render,
and explicitly warns off citing the PCAR negative against store-nothing (§ lines 45–56, 375–405).
Minor hygiene: the MEMORY.md index phrase "warp 3.7–10.3" is the wrong-carrier's number; the
honest current store-nothing start is the measured 1.095@ep2 (still ~5 orders above the ancestor
target — pose remains genuinely OPEN; the negative's operative content stands).

## 6. Muon cold-start finishing shortfall — CONFIRMED (projection sound) + one stale number

**Re-derived from run.log** (live run at ~ep962 at review time; final ep1000 verdict not yet
landed): best ep650 **0.003366** (tau_softplus era); pre-Muon tail flat (ep675 0.003376 / ep700
0.003407 / ep725 0.003414); Muon fires ep726 → ep750 **0.004351** (+27.4% vs ep725, +29.3% vs
best) → monotone decelerating recovery to ep950 **0.003818** (recent slope ≈ −4.3e-5/25 ep).
Reaching 0.003366 needs ≳250 further epochs at the current decaying rate; the run ends at ep1000
⇒ **it will not beat ep650**. The projection is arithmetically sound; verify the literal ep1000
number when it lands (expected ~0.0037–0.0038).

**Alternative-mechanism attacks:** (a) schedule confound EXCLUDED BY DESIGN — τ and hosc-β are
intentionally FROZEN during the Muon finisher (trainer line 6453 "FEED-fm FIX-2"; telemetry
confirms τ 0.2157 / β 3.1772 constant ep726→962), so the ONLY thing that changed at ep726 is the
optimizer; (b) EMA-lag excluded — window 1/(1−0.997) ≈ 333 steps ≈ 4.4 epochs at 75 accepted
batches/ep, cannot produce a 200-epoch recovery shape; (c) verdict-cadence/liveness clean
(9 consistent async points, accepted_frac 1.0, weights_stepped true). "Cold-momentum transient"
vs "Muon LR 0.002 too high for this basin" is NOT distinguishable from this data — but both are
answered by the same already-designed #270 warm-start + lr-final-frac arm, so the inference (A/B
needed) stands either way.

**Borrowed-number flag for the council draft (operating-manual §8.5):** lines 203/224/260 justify
`MuonWarmStart` with "MEASURED +8% cold-Muon transient" — that is the OLDER fork's anchor; the
LIVE run's measured transient is **≈+29%** (0.003366→0.004351, cited correctly at draft line 561).
Cite the live number; the warm-start case is ~3.6× stronger than the rationale text says.

**Cross-path caveat (recorded, not blocking):** the τ-probe's independent render authority orders
ep726 ckpt (0.003033) BETTER than ep650 EMA-BEST (0.003146), while trainer verdicts order them
opposite (0.003414 vs 0.003366) — a ~5%, state-dependent disagreement between the self-orient-
reconstructed probe path and the in-run co-state path (the basin probe measured the gap at +4.3%
on ep650). It does not change this verdict (13% shortfall > 5% path noise), but the warm-start
resume ckpt choice ("resume pre-switch" per #270) is supported: the ep726 checkpoint is at least
as good as ep650 on the independent authority.

## Sweep of other recent negatives the council draft leans on

- "viscosity NO-GO" — previously adversarially cleared by the #321 confound hunt; no new
  evidence; not re-litigated.
- l7 = measured defect (demoted from default curriculum) — no new evidence found; stands.
- Hyperprior twice-ruled-out at our byte scale (§20) — out of this review's scope; its two legs
  are measured+derived and nothing here touches them.
- Islands/mod32cap "zero lane/movable islands" — council-designed control, not a negative
  (memory `mod32cap_is_council_designed_clean_baseline...`); no action.

## $0 probes run (read-only, all from existing artifacts)

1. **GT-H control, n600** (the #1 overturn): `experiments/results/tau_crossover_trainflow_20260707/
   gt_h_control_n600.json` (+ `gt_h_control.py`), peak RSS < 3 GiB, live run untouched.
2. Transfer-law re-derivation (#341): (8/600)(−3.36%)+(592/600)(+5.17%)=+5.06% ✓.
3. H re-derivation from raw r_gap/r_mark in the probe JSON (0.2739/0.4113=0.666 ✓ etc.).
4. Full verdict-trajectory extraction from run.log (negative 6 table above).
5. 0.65-bar re-derivation (0.51×1.273108≈0.65 ✓) and waterline cross-check.

**Owed (named, NOT launched):** GT-conditioned H re-instrument + re-run on the 4 frozen snapshots
($0, ~2–3 h CPU); GT-conditioned comb-registration audit before any n287_dash_comb activation;
fixed-τ control arm (operator-GO, already a §19 design item); optional ridge-to-θ0 K-ladder for
#341; the ep1000 final-verdict read-back.

means ≠ ends: pointer 0.19110 UNMOVED; nothing here is a score row.
