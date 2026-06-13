# Recursive adversarial review — ROUND 13 of the 5 Layer-2 levers (2026-06-13)

**Reviewer:** Partner-A2 (author ≠ reviewer). The SEAL requires 3 FRESH consecutive clean rounds.
Prior FRESH count: R9 CLEAN → 1/3; R10 NOT-CLEAN (cold-anneal floor) → reset 0/3; R11 CLEAN → 1/3;
**R12 NOT-CLEAN (miscalibrated `SEG_ANNEAL_GRADIENT_FLOOR_T` 0.1→0.3) → reset 1/3 → 0/3.** So R13 began
the FRESH count at **0/3**.

**R13 has the THIRTEENTH, distinct lens: the all-5-levers-on END-TO-END MULTI-STEP DESCENT on the REAL
frozen scorer + byte-close + parse-back + finite real-score components.** No prior round measured whether
the FIVE COMPOSED levers, integrated, actually DESCEND the combined loss over a real optimizer TRAJECTORY:
R10 measured the single-STEP gradient SIGN; R5 measured determinism; R7/R11 measured resume bit-identity;
the synthetic `test_compose_all_five_levers_end_to_end` only BYTE-CLOSES (it never checks the loss goes
DOWN). A set of levers can each point downhill at step 0 yet fail to descend over a trajectory
(oscillation, the seg term going cold-dead and stalling, the rate/QAT terms cumulatively fighting the
score terms). R13 RUNS a real all-5-on Adam trajectory on the real scorer and asserts the combined loss
genuinely decreases, the archive byte-closes + parses back, and the real-score components are finite.

## CLEAN-PASS VERDICT: **CLEAN → fresh counter ADVANCES 0/3 → 1/3.**

R13 found **NO defect in the lever code.** The five composed levers form a genuine, strong training signal
on the real scorer. The headline measurement: at the FAITHFUL daemon LR (1e-3), the all-5-on combined loss
descends **47.04 → 6.21** monotonically over 20 steps (a −40.65 / −87% reduction), the seg term drops
0.0846 → 0.0491, the pose term drops 37.97 → 0.70, every step is finite, the archive byte-closes, the
Lever-3 pose section round-trips, the decoder/latent section parses back, and the BEST real-score components
are finite + non-negative. **`R13_CLEAN: true`.**

### THE R10-LESSON DISCIPLINE — a first-run oscillation was correctly NOT declared a finding

The FIRST R13 probe run used `lr=5e-2` (50× the daemon's faithful 1e-3) and the loss OSCILLATED
(47→58→57→49→55→52…), giving `descended: false`. Per the recursive-review protocol AND the R10
test-instrument lesson (the prior R10 round's instrument bug, recorded in
`test_r10_levers_compose_not_fight_on_real_scorer`'s docstring), I did **NOT** declare this a lever finding.
Instead I separated instrument from substrate by consulting the GROUND TRUTH — the live distortion arm
daemon (`experiments/results/distortion_arm_l235_20260612T205102Z/torch_vehicle_trajectory.jsonl`), which
trains all the levers at `adamw_lr≈1e-3` and descends loss **0.684 (ep427) → 0.521 (ep676)** monotonically
with `pose_mse` 0.0039 → 0.0019. The 50× overshoot was a pure INSTRUMENT artifact (Adam diverges from step
1 at 50× the stable LR), NOT a lever defect. Re-running the probe at the faithful LR confirmed clean descent.
This is the canonical correct application of "Forbidden premature KILL without research exhaustion" at the
review surface: a mis-tuned instrument must not be allowed to fabricate a paradigm-level finding.

---

## A. THE R13 MEASUREMENT (MEASURED on the REAL frozen scorer).

`experiments/probe_r13_all5_end_to_end_descent.py` builds the real frozen scorer (RealScorerContext →
EfficientNet-B2 SegNet + FastViT PoseNet) on 8 real `0.mkv` pairs, a FiLM (Lever-3) base_ch=20 decoder, an
all-5-on stage (Lever-1 rate `rate_lambda_w=0.05`/`rate_lambda_lat=0.02` + Lever-2 surrogate `soft_cosine`
+ the anneal 1.0→0.3 floor + Lever-4 score-aware QAT + Lever-5 margin `τ=2.0`), and runs a real Adam
trajectory at the faithful `lr=1e-3`. **MEASURED (`scorer_class: RealScorerContext`, `R13_CLEAN: true`):**

| step | combined loss | seg term | pose term | annealed T |
|------|---------------|----------|-----------|------------|
| 0  | 47.043 | 0.0846 | 37.97 | 1.000 |
| 4  | 32.945 | 0.0774 | 24.60 | 0.926 |
| 9  | 14.928 | 0.0607 |  8.26 | 0.679 |
| 14 |  6.892 | 0.0483 |  1.46 | 0.413 |
| 19 |  6.207 | 0.0491 |  0.70 | 0.300 (floor) |

- **(1) DESCENT:** windowed tail-mean (last quartile) **6.39 << start 47.04** — the composed levers train.
- **(2) BYTE-CLOSE + PARSE-BACK:** `archive_byte_closed=true`, `lever3_pose_roundtrip_ok=true` (pose
  shape (8,6)), `decoder_section_parse_ok=true`.
- **(3) FINITE REAL-SCORE COMPONENTS:** `real_score_components_finite=true` (the BEST score is finite +
  non-negative — no NaN/Inf from any lever's contribution to the trained weights).

The seg term stays gradient-ALIVE the whole trajectory because the anneal lands AT the R12-corrected floor
(0.3), validating the R12 fix in the integrated path: the seg term's gradient does its work (0.0846→0.0491)
and is still alive at the floor, exactly as the R12 floor finding predicted.

## B. STANDARD CLEAN-CHECK (R13 lens) — default byte-identity preserved; no live-arm touched.

- The probe + the 2 new regression tests touch ONLY the lever surrogate/rate/QAT/pose paths through the
  driver's OWN loss assembly (`_seg_loss_for_spec` + pose + `_weight_regularizers`) — no driver code was
  changed this round (R13 is a pure measurement + test-add round, no fix needed). The default
  byte-identity guards (`test_all_default_driver_run_is_deterministic_and_byte_identical`,
  `test_default_train_epoch_matches_vendored_only_reference`) are unaffected.
- The live distortion-arm daemon (`distortion_arm_l235_20260612T205102Z`) was READ ONLY (its trajectory is
  the descent ground truth); it was NOT touched. The probe writes only `.omx/tmp/r13_*`.
- No `curriculum.py` constant changed (the R12 floor 0.3 is consumed correctly by the anneal).

## C. FRESH-EYES "QUESTION EVERYTHING"

1. **Is `descended=false` (first run) a lever finding?** NO — it was `lr=5e-2`, 50× the faithful LR. At
   lr=1e-3 the loss descends 47→6 monotonically (and the live daemon descends 0.68→0.52). Instrument, not
   substrate.
2. **Do the levers FIGHT cumulatively over a trajectory?** NO — both the seg term (0.0846→0.0491) AND the
   pose term (37.97→0.70) descend together; the rate (Lever-1) + QAT (Lever-4) regularizers do not stall
   the descent.
3. **Does the seg term go cold-dead before the argmax converges (the R12 risk)?** NO — the anneal lands AT
   the corrected floor 0.3 (gradient-alive the whole way); the seg term keeps descending until ~step 11
   then plateaus at its converged value while the temperature is still ≥ floor. The R12 fix holds in the
   integrated path.
4. **Does any lever produce NaN/Inf in the trained weights?** NO — all 20 loss steps finite; the
   byte-closed archive's real-score components are finite + non-negative.
5. **Is the byte-close a no-op (does the archive actually capture the trained model)?** The Lever-3 pose
   section round-trips to shape (8,6) and the decoder/latent section parses back to the right shapes —
   the archive grammar survives the all-5-on compose (consistent with the synthetic
   `test_compose_all_five_levers_end_to_end`, now also confirmed on the real scorer).

## Findings by severity

- **HIGH:** NONE.
- **MEDIUM:** NONE.
- **LOW:** NONE.
- (The first-run `descended=false` was an instrument-LR artifact, NOT a lever finding — correctly diagnosed
  via the live-daemon ground truth + a faithful-LR re-run, per the R10 lesson. It is documented here for the
  audit trail, not counted as a finding.)

## New regression tests (2 — the R13 mechanism guards)

1. `test_r13_all5_multi_step_descent_at_faithful_lr_synthetic` (FAST, base_ch=8, 8 steps, synthetic scorer)
   — asserts the all-5-on combined loss descends at the faithful lr=1e-3 (windowed tail-mean < start) AND
   the step-0 composed loss DIFFERS from the all-default loss (the levers are genuinely active — a no-op
   set would make the descent test vacuous). NO-FAKE: real Adam trajectory through the driver's own loss
   assembly; a constant lever set is rejected by the distinctness-from-default assertion.
2. `test_r13_all5_descent_byteclose_parseback_on_real_scorer` (REAL scorer, 16 steps + a full `run()`) —
   the headline lens: windowed descent on the real scorer + a full all-5-on `run()` that byte-closes an
   archive whose decoder + Lever-3 pose section parse back + the BEST real-score components are finite +
   non-negative. RESEARCH-ONLY tiny slice ⇒ [contest-CPU advisory] NON-PROMOTABLE.

## Test-run count

- Fast synthetic R13 descent test (`test_r13_all5_multi_step_descent_at_faithful_lr_synthetic`):
  **1 passed in 104.87s** (isolated).
- R13 real-scorer probe (`experiments/probe_r13_all5_end_to_end_descent.py`, 20 steps, lr=1e-3):
  `R13_CLEAN=true`, loss 47.04→6.21 (−87%), byte-close + parse-back + finite components all PASS.
- Full lever suite: see the trailer (run detached, SIGURG-proof; pinned at memo-write time).

## Wire-in / provenance

6-hook (Catalog #125): #6 probe-disambiguator ACTIVE (`probe_r13_all5_end_to_end_descent.py` is the
all-5-on descent + byte-close disambiguator); #1/#2/#3/#4/#5 N/A (review-round + 2 regression tests, no
new score-claim surface; no driver code changed). Mission contribution: `frontier_protecting` (R13 proves
the five composed levers descend the real combined loss + export a well-formed archive — the integrated
training signal a from-scratch run rests on — and locks it with 2 regression guards; the END remains a
lower exact score, frontier UNMOVED `0.19109982419209975` contest-CPU). Authority: all numbers
`[contest-CPU advisory]` real-frozen-scorer-but-tiny-slice NON-PROMOTABLE (descent-direction + finiteness
claim, not a score claim). No GPU launched, no daemon touched (read-only ground-truth consult), no
Cool-Chic touched, no archive-build region touched (Partner B's surface untouched).

**VERDICT: CLEAN (zero findings; the all-5-on combined loss descends 47→6 on the real scorer at the
faithful LR + byte-closes + parses back + finite components; the first-run oscillation was a 50×-LR
instrument artifact correctly NOT declared a finding) → fresh counter ADVANCES 0/3 → 1/3.** R14 + R15 must
each be a FRESH consecutive clean round; the SEAL requires TWO more consecutive clean rounds after this.
