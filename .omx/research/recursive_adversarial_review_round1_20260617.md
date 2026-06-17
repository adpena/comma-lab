# Recursive adversarial review — ROUND 1 (operator: "review for math/engineering/algebra/calculus/geometry/OSS optimal and adversarial recursively review all")

**Date 2026-06-17. Three independent adversaries over this session's work. Verdict: NOT-CLEAN
(clean-pass counter RESET to 0). R2 CLEAN; R1 + R3 NOT-CLEAN with real findings, including one
CRITICAL math error I own. All `[contest-CPU advisory]`; exact pointer UNMOVED at 0.19110.**

## Panel verdicts
- **R2 (engineering / OSS / NO-FAKE): CLEAN** — launcher `--seg-margin-hinge-throughout` verified real
  end-to-end (validated-PLAIN-config = StageSpec defaults; byte-identical default; driver wire-in real,
  no dead-flag; tests verify behavior not constants; 29 pass). Ego-hood probe NO-FAKE compliant. Restart
  hygiene sound. 2 LOW nits (probe SCORE_DROP correctly tagged a projection; test4 `/tmp/x` stylistic).
- **R1 (math / calculus / algebra / geometry): NOT-CLEAN** — 1 CRITICAL, 3 MEDIUM, 2 LOW.
- **R3 (assumption-adversary / strategy): NOT-CLEAN** — 2 CRITICAL, 3 MEDIUM, 1 LOW.

## CRITICAL findings + dispositions

### CR-A (R1-CRITICAL-1 + R3-M1) — the #1 pose-low-rank FALSIFICATION was UNSOUND. FIXED + REOPENED.
My inline falsification compared low-rank vs the iid codec at the **over-provisioned** recon MSE 2.9e-5,
when the contest only needs **MSE ≲ d_pose = 3.4168e-4** (storage MSE just ADDS to d_pose; 12× headroom).
At the correct fidelity, low-rank WINS: **rank-2 SVD @ 254 levels = 1142 B vs iid 3088 B = 2.70× smaller**
at MSE 2.66e-4 ≤ d_pose (durable: `.omx/research/pose_lowrank_CORRECTED_fidelity_20260617.json`). Saving
1946 B ≈ **0.0013 score**, lossless-relative-to-d_pose. Mechanism: pose dim-0 std ≈175× dims 1-5, ~99.8%
one shared SVD temporal mode; iid wastes bytes on 6 independent columns. **Disposition: REOPEN #1** as a
real fold-able rate win on the FiLM-STORE pose section (the running run uses `--pose-film-v2` → has a pose
section this shrinks). Wire-in queued (a low-rank `encode_pose_section` variant). My earlier "falsified"
claim is RETRACTED.

### CR-B (R3-CRITICAL-1) — margin_hinge-THROUGHOUT restart had NO revert trigger / NO registered control. FIXED.
Killed a still-descending CE run (ep3700 d_seg 0.00237, monotone) for an unvalidated lever resting on a
6-pair overfit probe, with the CE arm DELETED. FIX (this memo): the preserved CE log
`run_CE_baseline_ep3700.log` is now the **REGISTERED CONTROL**; both are from-the-same-basin so matched-epoch
comparison is valid. **REVERT TRIGGER:** the margin_hinge run MUST be clearly below the CE control d_seg at
matched epochs; if margin_hinge d_seg ≥ CE at ep~2000, REVERT to the basin and re-run the proper 600-pair
hinge-vs-soft_cosine A/B the accel-1 memo specified. CE control milestones (from-basin):
| ep | 500 | 1000 | 1500 | 2000 | 2500 | 3000 | 3500 | 3700 |
|----|----|----|----|----|----|----|----|----|
| CE d_seg | 0.004783 | 0.004355 | 0.003778 | 0.003251 | 0.003000 | 0.002703 | 0.002439 | 0.002370 |
margin_hinge live @ ep150 = 0.00551 (too early; decisive read at ep1000/2000). Run rides to the trigger.

### CR-C (R3-CRITICAL-2 + R3-L1 + R1-M1) — means-vs-ends drift; the nearest exact row (G3) is unrun. OPERATOR-ROUTABLE.
Pointer moved 0.00000 this session; everything is `[advisory]`. The small-basis vehicle has NEVER had a
byte-closed EXACT eval row — the advisory→exact gap is unmeasured. **The highest-EV next action is G3:
byte-close the current 600-pair basin (best_archive.bin, 89,136 B, advisory S≈0.39) and buy its FIRST dual
CPU/CUDA exact row** (CPU is $0 local; CUDA is a Modal <$5 dispatch). That single row tells us whether the
vehicle is anywhere near T_1 and calibrates every advisory projection. Surfaced to operator.

## MEDIUM/LOW dispositions
- **R1-M1 (power-law mis-specified):** the d_seg log-log slope is monotonically STEEPENING (0.24→0.64), not
  constant — no single exponent exists. The S(50k) projection is a 13.5× extrapolation off 0.87 decades and
  swings 0.15–0.26. FIX: projections are reported as a RANGE with the "slope still climbing" caveat (mildly
  optimistic), NOT point estimates 0.177/0.188/0.226. My earlier point projections are SUPERSEDED by this range.
- **R1-M2 (floor uses 0.bin not archive.zip):** rate term should use archive.zip = 89,274 B (incl. 138 B zip
  overhead), so rate 0.05944, floor **0.11790** (not 0.11781); sub-0.15 d_seg target ~0.000321. Memo floors
  corrected to 0.1179.
- **R1-M3 (additive vs multiplicative exponent carry):** algebra-ambiguous but the accel-1 memo honestly
  disowns the absolute projection → flag, not fake. No code change.
- **R3-M2 (pose is an intermittent catastrophe, 428× spread):** VERIFIED the artifact is protected —
  `driver.py:2596 is_best = score < best_score` selects on the full canonical S, so a pose-spike epoch is
  never shipped. Residual risk (pose variance wastes wall-clock; FiLM-v2 trunk-stopgrad OFF) noted, not a bug.
- **R3-M3 / R1-L2 (ego-hood small):** VINDICATED by the actual probe (0.038% of flips in static core,
  0.0001 score). #139 falsified-as-free-lever; folds into #137.
- **R1-L1 (d_pose floor optimistic):** the 0.000342 floor is below the live run's min/median d_pose; the
  pose-term half of any projection is optimistic. Noted in the range caveat.
- **R2-L2 (test4 /tmp/x):** trivial nit; cleaned.

## Round-2 gate
Round 1 is NOT-CLEAN; counter reset. Code/doc fixes (CR-A retraction+reopen, CR-B trigger, floor,
projection reframe, test nit) land this batch. The two OPERATOR-ROUTABLE strategy items (CR-C: fire G3?
+ CR-B: let margin_hinge ride to the ep2000 trigger vs revert now) gate the next clean pass — round 2
re-reviews after the operator's call + the fixes.
