---
doc_type: t5_crucible_seal_round1_v6_verdict
role: SEAL ROUND 1 of 3 (counter reset at the v6 fold) — both lenses on the launch candidate:
  LENS A recursive adversarial verify (arithmetic · provenance · buildability · probe-fold
  faithfulness) + LENS B deep-math meat hunt (missed meat · naive assumptions · synergies)
date: 2026-07-08
target: DRAFT_OPTIMAL_STACK_v6_20260708.md (feec6e7af+b5771b531 per ledger fold)
verdict: CLEAN (counter 1/3 on v6) — zero BLOCKER, zero MAJOR; 4 MINOR/nits listed with per-item
  bar tests; none changes a decision, a load-bearing number a consumer reads, or a build item.
  Per NO-OPEN-GATES: the FINAL round still waits on P-TAU2 + P-DITHER resolution (ledger-bound;
  not a v6 defect — the launch constant 0.31 and B19 default-off stand regardless by design).
axis: all numbers [macOS-CPU/MLX advisory]; pointer contest-CPU 0.19110 UNMOVED — this verdict
  is MEANS.
review_status: fresh-eyes-reviewed(1) — this verifier authored none of v1..v6, the probe waves,
  τ-confirm, #348, or the prior verdicts; every load-bearing number below RE-EXECUTED
  ([re-executed], .venv python full precision / live argparse grep / primary artifact JSON reads)
  or re-read against the primary on-disk artifact ([verified-by-inspection]).
---

STORES CONSULTED: ORCHESTRATION_LEDGER.md (full — reqs A–S; operator pins J/K/L/M/N/P/R as audit
angles; the v6/τ-confirm/wave-A/wave-B/#348 landing folds = the fold spec) ·
DRAFT_OPTIMAL_STACK_v6_20260708.md (full) · probe_waveA_ct_schedule_20260708.md (full) ·
probe_waveB_geometry_class_20260708.md (full) · probe_tau_confirm_ep1000_20260708.md (full) ·
deterministic_gpu_accum_348_20260707.md (full) · seal_round1_v5_verdict_20260707.md (full — the
9 minors; regression base) · seal_round2_verdict_20260707.md (§1–§2 — the 9 round-2 findings;
regression base) · DRAFT_OPTIMAL_STACK_v5_20260707.md (inherited-row spot-checks: §0.0a M1/M2/M3,
SC-3/SC-16/SC-18 rows, §11 rows 4–6, §5 rates, §1.1 sketch) · trainer argparse LIVE RE-GREP this
session (experiments/train_levelset_witness_realized_through_R_mlx.py — all 22 flag families v6
names counted; `--fused-r-kernel` add_argument + the `--mlx-device gpu` ValueError requirement
read in source; dither = 0 hits confirmed) · PRIMARY ARTIFACTS re-read: pdz_deadzone_census.json
(census substrate + full-precision totals) · trace_probes_...mod32cap...json (ν = 0.012653403634932212,
s* = 6.897090681181217e-6, settle 237.09, cycle 387.09) · pcon_conley_backtest.json path +
tools/conley_persistence_certifier.py SOURCE (margin-axis provenance) · tools/uint8_deadzone_census.py
SOURCE (margin-axis provenance) · experiments/results/witness_per_stage_attribution/summary.json
(census-vehicle identity) · artifacts dir ls (all cited artifact files exist; NO v5.1 errata file
exists — v6's premise confirmed). NOT consulted: durable-state files (stale per sweep); CT-1/CT-2
full re-reads (carried via v5 per round-1 v5's fold-faithfulness audit). $0: reading + arithmetic
+ grep only; no training launched, no run dir written.

# SEAL ROUND 1 (v6) — CLEAN. Counter 1/3.

## §1 LENS A — RECURSIVE ADVERSARIAL VERIFY

### 1.1 Crossing chain re-executed unrounded, independently [re-executed]

Fresh python, no reference to v6's chain:
- pose √(10·3e-5) = 0.017320508 → **0.0173205** ✓ · central 93,092×25/37,545,489 = 0.061986142
  → **0.0619861** ✓ · win9 81,032 → 0.053955883 → **0.0539559** ✓
- S rows (g_dec = 1.0427e-4): (0.0011, central) = **0.1997336**, over by 0.0086336 = 4.85×0.00178 ✓
  · (0.0010, central) = 0.189733650, margin 0.001366350 → **0.1897336 / 0.0013664** ✓ ·
  (0.0010, win9) = 0.181703391, margin 0.009396609 → **0.1817034 / 0.0093966** ✓ ·
  (0.0011, win9) = 0.1917034, over by 0.0006034 ✓
- bars: central **0.0010136635 → 0.0010137** ✓ · win9 **0.0010939661 → 0.0010940** ✓ · ILC
  0.0011 − 1.0427e-4 = **9.9573e-4** ✓ (row (c) holds)
- fold 12's claim "no probe fold enters this arithmetic" verified structurally: τ*/ν/B17/B16/
  #149/P-MP touch schedule/curriculum/telemetry/decode-lever surfaces only; rate/pose/g_dec
  inputs unchanged from v5 (round-1-v5-verified) ✓

### 1.2 The NEW asymptote arithmetic (fold 8 / §0.3) [re-executed]

- decomposition: 0.0036146 − 1.5795495775e-3 = **2.0350504225e-3 → "≈ 2.0351e-3"** ✓; ratio
  1.5795495775e-3/0.0036146 = **0.4369915 → "0.4370"** ✓; subset frac 1.5795495775e-3/0.0041170120
  = 0.38366407 → "38.366%" ✓
- smooth-only floor: 100×1.5795495775e-3 = 0.157954958; + 0.017320508 + 0.061986142 =
  **0.237261608 → "≥ 0.2372616"** ✓; win9 + 0.053955883 = **0.229231349 → "0.2292313"** ✓
- **composition CORRECT — and the charter's suggested form 100·(g_dec + locked) would be WRONG:**
  the census counts flips on the (near-)decoded/through-R state, so the locked mass is a subset of
  the DECODED residual; the floor is 100·d_seg_decoded ≥ 100·locked (+pose+rate). Adding g_dec on
  top would double-count the train→decode gap (g_dec is already inside decoded d_seg). v6 got the
  composition right ✓
- "the bar is 1.6× BELOW the locked mass": 1.5795e-3/0.0010 = 1.58; vs 0.0010137 → 1.558 → "1.6×" ✓
- "cannot cross" robustness probed: even at pose→0 the win9 floor is 0.15795+0.05396 = 0.2119 >
  0.19110 — the claim holds at the design rates independent of the pose term ✓ (it is rate-family-
  scoped, which is how §9 states it)
- "smooth-perturbative-only" defined honestly: "INR corrections through R, no large-amplitude
  analytic levers, no dither" + the floor explicitly CONDITIONAL ("if the census transfers to the
  design endpoint") + self-attack 2's 2×-either-way robustness ✓. One labeling gap → MINOR-1.
- yield denominations: 1% census = 1.5795e-3 S/0.00178 = **0.887 → "≈ 0.89×"** ✓; full census
  0.158 S ✓; P-DZ bands 5.34e-6/1.78e-5 = 0.3×/1× of 0.00178 ✓; 88.74× / 295.8× ✓

### 1.3 τ* fold faithfulness (fold 5 / §1.4a) [re-executed]

- table reproduces the probe bit-for-bit (END/BEST/ep300/anchor rows) ✓; τ*(q) arithmetic (ln5 =
  1.6094379): 0.44646/ln5 = **0.2774012 → 0.27740** ✓ · 0.65607/ln5 = **0.4076392 → 0.40764** ✓ ·
  0.49485/ln5 = **0.3074676 → 0.30747** ✓ · 0.74347/ln5 = **0.4619439 → 0.46194** ✓ · q50 0.11149/
  0.12304 → "0.11–0.12" ✓
- launch value: ep650 ckpt τ = 0.3098 ∈ [0.27740, 0.40764] ✓ (cosine cross-check: τ(650; den
  1000, end 0.05) = 0.310 ✓; freeze-at-726 τ = 0.216 ✓ — the M2 freeze narrative internally
  consistent); `--softmax-temp-end` EXISTS (1 argparse hit) and takes the constant — buildable ✓
- fixed-point convention: mass(m < τ*·ln5) = f_target, f_target = conversion rate, correctly
  identified as NOT derivable from a static snapshot → P-TAU2 named with disposition ("reporting
  probe; live-law promotes ONLY after f_target; fail-safe constant stands regardless") ✓;
  SC-3 promotion path = would-be row per verdict cadence + run-2 config promotion — no new flag
  needed, q̂ = 0.85 stated as the bracket midpoint convention ✓
- the "anneal TRUNCATED → over-descent" INVERSION: tagged [INFERRED, epoch-confounded] with the
  causal adjudicator named (Q2-τ + SC-3) ✓; the β-leg of M2 explicitly preserved ✓; the tautology
  (signed witness-margin-toward-GT ≤ 0 at flips; bit-reproduced 0.7644972239) verified against the
  probe's provenance chain ✓. Honest, not convenient: v6 also concedes v2's 0.2 sat inside the
  defensible q50 band (per the probe) via the cascade rather than claiming novelty.
- completion guarantee: raising the endpoint 0.062→0.31 shortens descent distance under the same
  `--anneal-epochs 600` event-margin law — guarantee strengthens ✓ (sound)

### 1.4 ν-law recomputes (fold 2 / §2.2g) [re-executed vs the artifact]

Artifact unrounded ν(tau) = 0.012653403634932212:
- settle 3/ν = 237.0904 → **237.1** ✓ · cycle floor 387.0904 → **387.1** ✓ · dwell ≥ 237.1 ✓
- s* = ν·forfeit: artifact **6.897090681e-6 → 6.8971e-6** ✓ exactly; 4-dp-ν recompute 6.89687e-6
  → v6's own "6.8969e-6 display-precision" footnote ✓
- k_max = floor(2350/387.09) = floor(6.071) = **6** ✓ · dwell switching: ln(1.275)/0.012653 =
  19.20 ep; 250/19.2 = 13.0 → "13× margin" ✓ · LPV ratio (1/ν)/20 = 3.952 → **3.95×** (old 1.908
  → 1.9×) ✓ · V-demand 237.1/25 = 9.48 → "V ≈ 10–11" ✓
- **V=5 retention: SOUND, not a rationalization.** The chain is measured→derived, not convenient:
  (i) MEASURED — the V=4-form estimator fired ep675, EMA-best-at-fire = the true stage best,
  forfeit exactly 0, on this trace; (ii) MEASURED — fire epoch invariant across s* ∈ [6.9e-6,
  1.42e-5] (slope crosses zero in (650, 675)); (iii) DERIVED — a zero-crossing/threshold detector
  near exhaustion does not need its window to cover the ν settle time; settle-coverage is a
  ν-ESTIMATION requirement, and v6 correctly re-binds it to the F3/SC-9 estimation consumers
  (windows ≥ 237) instead of silently dropping it; (iv) scope-tagged FORMULATION/this-trace;
  (v) residual risk (false fire on run-1's different schedule) is triple-covered: anneal-complete
  PRECONDITION + fail-safe cap 726 + B-INJ owed pre-GO. This REVERSES the CT-1 import's rationale
  while keeping its conservative artifact (V=5 > the bit-reproduced V=4) — the reversal is
  measured-grounded and the law is re-derived, not deleted. ✓
- P-CT3 fold verbatim-faithful (ep675 ∈ [670,700]; +5.450779e-4 S recovery; fire@625-as-written
  +2.666897e-3; fire@650/675 forfeit exactly 0 — all match the probe/artifact) ✓; promotion
  contingency honest (armed-with-fallback until B-INJ; self-attack 3 addresses the req-B question
  correctly — backtest leg done, injection leg named pre-GO) ✓
- P-CT2/cadence: 5/41 vs band 12–17, kill not triggered, ×1–×6 floor sweep never reaches band ✓;
  the antagonism conjunct is the probe's fix (a), with fix (b) dropped WITH the probe's own reason
  carried ("restore-best selection needs the verdict") ✓

### 1.5 B17 fitted bar (fold 6 / §3.4) [re-executed]

- Tau: s×τ×ln5 = 21.75 × 0.0500065329 × 1.6094379 = 1.75049242 → **1.7504924172** ✓ (equivalently
  21.75 × raw-thr 0.0804824100; sub-9th-decimal input-rounding artifacts only) · MuonBest: 3.75 ×
  0.2156894835 × 1.6094379 = **1.3017706202** ✓ exact
- "near τ-INDEPENDENT": threshold ratio 1.7505/1.3018 = 1.3447 → "1.35×" while τ ratio
  0.21569/0.05001 = 4.313 → "4.3×" ✓ · raw-form kill numbers 0.4408189379/0.5635593220 vs band
  ≥0.95 ✓ · cert-vs-uncert 5.16×/5.10× → "5.2×/5.1×" ✓
- **provenance CRITICAL CHECK PASSED:** tools/conley_persistence_certifier.py takes the margin
  axis from `z["margins"]` of the GT cache (gt_strided_n200.npz) — the TRUE frozen-SegNet
  top1-top2 field — NOT the corrupted maps-npz `gt_margin` key (maps npz supplies only witness
  argmax + τ). B17's fitted bars do NOT inherit the τ-confirm tautology. Same check on
  tools/uint8_deadzone_census.py: margins from the GT cache ✓ (its GT-geometry form limitation is
  stated in its own docstring and carried in v6 §0.3). The one law family that consumed the
  corrupted axis (τ*) is exactly the one v6 re-derived. ✓

### 1.6 Flag-reality (angle 7) [re-executed — live argparse]

All 22 flag families v6 names counted in the live trainer: `--fused-r-kernel` 1 (BooleanOptionalAction
default False, source-read; the `--mlx-device gpu` requirement is a real ValueError raise, and
`--mlx-device` defaults to gpu — the launch posture satisfies it) · `--softmax-temp-end` ·
`--anneal-epochs` · `--persistence-warmup-epochs` · `--persistence-classes` ·
`--persistence-loss-weight` · `--amplify-weight` · `--amplify-persist` · `--island-dilate-px` ·
`--seed-island-eased` · `--seed-anneal-*` (2) · `--margin-saliency-*` (7) ·
`--eikonal-visco-eps-floor` · `--eikonal-visco-ca-band` · `--weight-entropy-penalty-lambda` ·
`--muon-start-epoch` · `--stage-transition-rewarmup-*` (3) · `--verdict-batch` · `--render-aa` ·
`--curriculum-plateau-windows` — every one ≥1 hit; **dither flags = 0 hits** confirming B19 is
correctly PROPOSED-NEW/decode-side ✓. Line refs (7577/6088) drift-adjacent to current HEAD —
v6's own "counts, not line numbers" rule applies ✓. **Zero invented flags.**

### 1.7 Fold-charter completeness + probe-fold faithfulness (angles 2–5, 10)

All 14 §0.1 rows checked 1:1 against the four probe memos + #348 + the round-1 minors: every
number cited in v6 reproduces in its source (Q1: 0.1242/0.1116/0.1147/+0.1346/−0.1027/0.0808/
−0.24…−0.38 ✓ · P-DZ: 1.5795495775e-3/1.4556778802e-3/rows 176–224/24.7%/77.5%/67–85% ✓ · P-MP:
~0.41 plateau/oracle 0.0761–0.1306/envelope 2.30–58.18/2,304,000 B = 1.5341 S = 862× ✓ · F-DET:
0/28 N=10, Muon 0/28 N=5, 25.35→23.44s = −8%, 25/25 parity ✓ · comb 79.33: 1−0.0003600/0.0017417
= 0.79331 ✓, cited as 79.33 with the 86% quarantined + step-0 named ✓ · FEED-08l correction +
compensated-trade structure carried ✓). B-DET is #348's own reactivation clause, honestly
non-blocking either way ✓. B19 rule-118 reasoning verified: seeded ordered/blue-noise dither =
generic deterministic algorithm + one seed constant; no per-frame/video-derived table enters the
archive; OFF-identical; byte-close-SELECTABLE path named (decoded-KKT §5.0) ✓. P-DITHER band/kill
pre-registered (fire ≤ −1e-5 = 0.562× the 0.00178 margin ✓; kill ≥ 0 scoped to THIS decode-side
zeroth-order form with the trained-with reformulation named) — the (−1e-5, 0) middle zone resolves
to the safe default (stays unadmitted) by the default-off construction ✓.

### 1.8 Verdict-scope substance (angle 6, req R)

Every negative v6 consumes carries a scope tag at the narrowest supported level, with reformulation
queues where required: τ* 0.062 = INSTANCE (law form untouched) ✓ · ν registered-value = FORMULATION
with the AIC-preferred-form statement + F3/window-origin reformulations in the consumed probe ✓ ·
muon_fin window laws = FORMULATION with fail-safe caps carrying ✓ · window-covers-settle =
FORMULATION-of-the-window-law/this-trace, and the requirement is RE-BOUND to estimation consumers
rather than deleted (exemplary narrow scoping) ✓ · P-CON = FORMULATION (raw τ·ln5) with 5-item
queue ✓ · P-MP = FORMULATION with RANKED queue + measured "larger K is NOT the fix" ✓ · Q1 =
FORMULATION non-fire with 4-item queue, S_R weight-of-choice withdrawal scoped to point-predictor
role only (attenuation-BOUND consumer untouched) ✓ · §12 new rows tagged ✓. **No over-scoped kill
found** (the BLOCKER bar is not approached).

### 1.9 Regression tables

**Round-2 findings (all 9) stay fixed in v6:** AA ipe in the sketch ✓ · no "0.011" regression
(grep: zero stale hits; every 0.062/1.4154e-5/115/265 occurrence is a strike/was-context) ✓ ·
B13 per-class weights inherited (§3.1–3.3, w 1.0/0.28) ✓ · `--softmax-temp-end` ✓ · hood
5.32688e-6 in §5 ✓ · adaptive-ε saturation ALARM kept at the relaxed endpoint (cascade row 1) ✓ ·
recon-gap ranking / AA×island row / feedforward-ε_ff all inherited-v5 (round-1-v5-verified) ✓.
**Round-1-v5's 9 minors all incorporated** (§13 table checked item-by-item; the corrected
persistence spellings verified REAL by live grep; A2's x pinned computed-from-SC-18; B3's 10th
c(τ) row present; R-1 two-τ conjunct restored in §1.3; R-2 §12 tags present; R-3 disposition
wording in §7c; F12→Q2-τ label split done) ✓. The lane-anisotropy scope pin honored (no
anisotropic lever demoted on the u_min-isotropic negative; B16's non-fire is Q1-measured, a
different instrument) ✓.

## §2 PROVENANCE AUDIT (L81 — load-bearing measured claims NEW or CHANGED in v6)

| claim | anchor | review status | form limitations (carried in v6?) |
|---|---|---|---|
| τ tautology + true m_q table (END 0.74347 / BEST 0.65607 / anchor 0.818) | tau_mq_confirm_{end,cached}_20260708.json + tools/witness_tau_mq_confirm.py (bit-for-bit wave-A cross-check) | fresh-eyes-measured(1); instrument reviewed, 4 tests | 16-pair advisory subset, NOT n600 — carried (SC-3 = authority path) ✓ |
| ν = 0.012653 / settle 237.1 / cycle 387.1 / s* 6.8971e-6 | trace_probes_...json (unrounded values read this round) | fresh-eyes-measured(1); 17 tests + bit-for-bit v2-estimator anchor | one trace, one vehicle — carried (F3 refits on run-1) ✓ |
| forfeit-arm fire ep675 / +5.450779e-4 S | same artifact + trace | fresh-eyes-measured(1) | backtest leg only; B-INJ owed pre-GO — carried ✓ |
| P-CON fitted bars 1.7504924172 / 1.3017706202 | pcon_conley_backtest.json + .ledger.npz (3454×2) | fresh-eyes-measured(1); **margin axis verified TRUE gt-cache field this round (source-read)** | 96-frame subset; attribution-run vehicle; Δ_dec^logit=0 until SC-7 — subset carried ✓, vehicle → MINOR-1 |
| P-DZ census 1.5795495775e-3 / 38.366% | pdz_deadzone_census.json (+.gi_hists.npz = SC-16 seed, exists on disk) | fresh-eyes-measured(1); **margin axis verified TRUE gt-cache field (source-read)** | GT-geometry estimator form + state-dependent carried ✓; census vehicle = the θ* per-stage-attribution run (summary.json 2026-06-30), NOT mod32cap → MINOR-1 |
| Q1 ρ table / sR-at-chance (SC-20) | q1_signed_asymmetry.json | fresh-eyes-measured(1); inherited-instrument defect found+fixed at review (missing-side kill block) | sR sentinel failed; admitted WITH CAVEAT via the passing margin control — carried verbatim in v6 §1.3 ✓ |
| F-DET 0/28 N=10 / −8% / 25/25 parity | #348 memo + tools/mlx_gpu_determinism_probe.py + tests | reviewed landing (2 review passes per its memo) | smoke scale; n600/self-orient composite check OWED = B-DET, carried as SC-21 + pre-GO ✓ |
| wave-B instruments + memo recovery chain | wave-B header: instruments inherited from credit-killed predecessor, **reviewed-as-own with one defect found+fixed**; verdicts written fresh-eyes this side | chain verified: recovery-inherited-then-reviewed, compliant with L81 (not recovery-written-UNREVIEWED) ✓ | |
| FEED-08l corrected verdict | freq_along_ladder JSON re-derived by wave-B fresh eyes | the recovery-written original superseded by the fresh-eyes re-review — the L81 queue item DISCHARGED | oracle-injection form only — carried ✓ |

## §3 LENS B — DEEP-MATH MEAT HUNT

**Survived (certified for rounds 2–3):** the smooth-only floor's composition (locked ⊂ decoded ⇒
no g_dec term — v6 avoided the double-count a naive reading invites) · the V=5 re-derivation
(trigger-vs-estimator consumer split is the mathematically right factoring of "window law") · the
fixed-point τ* form (conversion-rate f_target is the correct Maslov-budget operationalization; a
static-snapshot f_target would have been the naive shape) · consistency row (d) as the honest
replacement of struck row (a) — two independent probes converging on "absolute logit scale, not
τ·ln5" is genuine cross-instrument physics · B19's mechanism honesty ("plausibly a small
fraction") · the M1-census → three-binding-constraints honesty upgrade in §0.2 · the 21-row ledger
(dedupe exact: 19 + SC-20 + SC-21; both new rows carry real named consumers; SC-16's seed data
verified on disk) · req-I seams: the B16×AmplifyIsland precedence law, the cadence×forfeit-arm
conjunct, and the fused-R train-path/decode-path separation (byte-close decode is the numpy
reference path — no fused-R×B19 interaction exists; dither position pinned at final uint8
quantization, after band/clamp render) · meat sweep vs the probe memos found NO silently dropped
ranked item (P-CT2 fix (b) dropped WITH reason; τ-confirm's restricted-m_q reformulation is
subsumed by the sharper conversion fixed-point; the 20260630 margin-axis re-read flag is
discharged for every v6 consumer — τ* re-derived, Conley re-measured on the true axis).

### Findings (all MINOR/nit; per-item bar test printed)

**[MINOR-1] The census/certificate VEHICLE is unnamed in §0.3 — cross-vehicle transfer is real
but implicit.** P-DZ and P-CON ran on `witness_per_stage_attribution` maps = the θ* per-stage-
attribution run (summary.json generated 2026-06-30, MuonBest @ep900, with an l7 stage mod32cap
does not have) — NOT mod32cap. v6 §0.3's decomposition row subtracts that vehicle's subset-absolute
census (1.5795e-3) from mod32cap's n600 decoded residual (0.0036146) in one equation, and the
"43.70% of the n600 decoded residual" gloss reads as same-object. Mitigations already in v6: the
floor is explicitly conditional ("if the census transfers"), the row is tagged [MEASURED-at-state,
subset], self-attack 2 claims 2×-either-way robustness, and τ-confirm independently measured the
two vehicles' margin fields as "statistically the same shape". Bar test: under the alternative
FRACTION-transfer convention (0.38366 × 0.0036146 = 1.3868e-3 locked), the smooth-only floor is
0.2180 central — still above 0.19110, the bar-vs-locked ratio is still >1 (1.37×), and every
downstream disposition (three binding constraints; B19 gated; #149 duty queue; family band
conditional) is UNCHANGED; no launch knob consumes the census (it re-computes from run-1 SC-16).
**Changes no decision, number-a-consumer-reads, or build item.** Owed: one clause naming the
vehicle + the transfer convention (absolute vs fraction) in §0.3.

**[MINOR-2] δ_τ width printed 0.4986 is τ=0.3098's value, labeled "at τ_end = 0.31".**
0.31×ln5 = 0.49893; 0.3098×ln5 = 0.49860 — v6 §1.4a-cascade row 3 and §11 row 16 print 0.4986
under the 0.31 label (the measured-anchor τ leaked into the launch-constant slot). Consumer check:
cascade row 5 (island gate margin ∝ τ·ln5) "updates with row 3" — a 0.07% slip, far below any
band/kill resolution; no other consumer. Changes nothing; two-token fix (0.4989, or relabel).

**[nit-3] §0.3 prints "0.3836640693 of subset flip mass"; the artifact's value is 0.3836640671**
(deadzone_frac_of_flips_HR). v6's 10-digit figure reproduces from the ROUNDED numerator/denominator
(1.5795495775e-3/0.0041170120) — false precision at digits 9–10 (req-J cosmetic; "38.366%" is
correct everywhere it is used). No consumer at that resolution.

**[nit-4] P-DITHER measures the dither's marginal on the CLEAN control decode** (mod32cap ep650:
no band/clamp/islands), while run-1's byte-close would select it composed with band+clamp on
exactly the census's concentration regions (far-lane rows / hood) — the gate's Δ could over- or
under-state the composed marginal. Already structurally covered: admission is via decoded-KKT
selection on run-1's OWN measured composed decode (§5.0), so the final admit is apples-to-apples;
the gate only decides build/ship. One sentence in §7c acknowledging composed-marginal re-measure
at selection would close the req-I grain. Changes no decision (gate semantics and selection law
both stand as written).

## §4 VERDICT + COUNTER

**CLEAN.** Zero BLOCKER, zero MAJOR. Four MINOR/nit items, each with its bar test printed: none
changes a decision, a load-bearing number any consumer reads, or a build item (MINOR-1 is a
labeling clause whose both-conventions outcome preserves every disposition; MINOR-2 is a 0.07%
label slip below all bands; nit-3 is display precision; nit-4 is already structurally covered by
the decoded-KKT selection). Stated plainly per the charter: these are nits that change nothing —
CLEAN, and I am not manufacturing findings to look rigorous, nor suppressing them (all four bind
to the P7 editorial fold).

The strongest things that SURVIVED adversarial re-derivation this round: (1) the crossing chain +
the new asymptote arithmetic — every digit reproduces, and the floor's composition is correct
where the naive form (adding g_dec) would have been wrong; (2) the B17/P-DZ margin-axis provenance
— I source-read both instruments specifically hunting a second tautology inheritance, and both
take the TRUE gt-cache field (the corrupted-axis blast radius is exactly the one law v6 already
re-derived); (3) the V=5 retention — a reversal of a CT-1 import that could have been convenience,
verified instead as measured-grounded with the settle-coverage requirement re-bound (not deleted)
to the estimation consumers.

**Counter: 1 of 3 on v6.** Sequencing per the ledger's NO-OPEN-GATES rule: rounds 2–3 may
proceed; the FINAL certifying round waits for P-TAU2 + P-DITHER resolution (both fail-safe:
τ_end 0.31 constant and B19 default-off stand regardless). Round 2 delta-scope suggestion: the
four editorial items above if a v6.1/v7 lands; fresh eyes on the CT-1/CT-2 internal derivations
(still round-1-unreviewed; v5/v6 verified their FOLDS and plug-consistency, not every internal
step); and the B-INJ + B-DET pre-GO artifacts when they land.

Pointer contest-CPU 0.19110 UNMOVED — this verdict is MEANS.
