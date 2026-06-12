# Recursive adversarial review — ROUND 12 of the 5 Layer-2 levers (2026-06-12)

**Reviewer:** Partner-A2 (author ≠ reviewer). The SEAL requires 3 FRESH consecutive clean rounds.
Prior FRESH count: R9 CLEAN → 1/3; R10 NOT-CLEAN (cold-anneal seg-gradient floor finding, fixed) → reset
0/3; R11 CLEAN → 1/3. So R12 began at **1/3**.

**R12 has the TWELFTH, distinct lens: the COMBINED (Lever-5 margin × Lever-2 surrogate) gradient floor +
the live-arm config-waste quantification.** R10 measured the SURROGATE-alone gradient floor and set a guard
constant `SEG_ANNEAL_GRADIENT_FLOOR_T`. But the live distortion arm runs Lever-5 ON (`margin_weight_tau=2.0`),
which MULTIPLIES the per-pixel surrogate by a DETACHED weight `exp(−margin/τ) ∈ (0,1]`. R12 asks two new
questions no prior round measured: (Q1) does Lever-5's up-weighting RESCUE the cold-dead surrogate gradient
(the "multiply-a-dead-gradient" case)? and (Q2 — the operator's embedded analysis) the live arm uses
`seg_temperature_end=0.05`, below R10's floor — QUANTIFY the seg-lever waste at end=0.05 vs end=0.1 on the
REAL frozen scorer.

## CLEAN-PASS VERDICT: **NOT-CLEAN → fresh counter RESETS 1/3 → 0/3.**

R12 found **ONE genuine finding IN THE LEVER CODE** (a MEDIUM-grade miscalibration of the R10 floor guard):
**`SEG_ANNEAL_GRADIENT_FLOOR_T` was set to 0.1, but the measured combined gradient at T=0.1 is ~10 orders of
magnitude below the warm-T norm — effectively DEAD. R10's OWN measured table labeled T=0.1 "≈ dead"
(|grad|≈5e-12), yet the guard `seg_anneal_temperature_is_gradient_alive` classified 0.1 (and 0.12, 0.15, 0.2)
as "alive".** The guard mislabeled a deep dead-zone as the usable boundary. A tuner consulting it to choose
`seg_temperature_end` would be told end=0.10–0.20 is "alive" when the seg lever there produces a 5–10-order-
below-warm rounding-error gradient. Per the recursive-review protocol ("the counter resets whenever a round
finds any issue"), R12 is **NOT a clean pass** — fresh count resets **1/3 → 0/3**. FIXED this round with the
2-landing pattern (constant corrected to 0.3 + 3 regression tests + docstring).

**The Q1 multiply-a-dead-gradient question + Q2 config-waste quantification are ANSWERED CLEAN** (no defect):
Lever-5 CANNOT rescue the cold-dead gradient (its detached weight ≤ 1 only scales DOWN; MEASURED margin-ON ≤
margin-OFF at every T), and the config-waste is now quantified for the operator. The ONLY finding is the
miscalibrated floor constant, surfaced WHILE measuring the combined floor.

---

## A. THE OPERATOR'S EMBEDDED ANALYSIS — live-arm config-waste, MEASURED on the REAL scorer.

`experiments/probe_r12_combined_seg_gradient_floor.py` builds the real frozen scorer (RealScorerContext →
EfficientNet-B2 SegNet + FastViT PoseNet) on 8 real `0.mkv` pairs, a FiLM (Lever-3) base_ch=20 decoder, and
measures the COMBINED (Lever-5 margin × Lever-2 surrogate) param/latent gradient norm across a fine T-sweep,
normalized to the warm-T=1.0 norm. **MEASURED (`scorer_class: RealScorerContext`):**

| T | combined grad-norm | ratio vs warm (1.0) | corrected guard (floor 0.3) |
|------|--------------------|---------------------|------------------------------|
| 1.00 | 1.5459e+00 | 1.00 | ALIVE |
| 0.50 | 1.0004e-01 | 6.5e-2 | ALIVE |
| 0.40 | 2.5615e-02 | 1.7e-2 | ALIVE |
| **0.30 (floor)** | 2.6616e-03 | **1.7e-3** | **ALIVE (boundary)** |
| 0.25 | 4.4873e-04 | 2.9e-4 | DEAD |
| 0.20 | 3.4526e-05 | 2.2e-5 | DEAD |
| 0.15 | 7.5298e-07 | 4.9e-7 | DEAD |
| 0.12 | 1.5526e-08 | 1.0e-8 | DEAD |
| **0.10 (OLD floor)** | 3.0020e-10 | **1.9e-10** | **DEAD (was wrongly ALIVE)** |
| 0.08 | 8.9028e-13 | 5.8e-13 | DEAD |
| 0.05 (live tail) | 6.2672e-20 | 4.1e-20 | DEAD |

### THE OPERATOR NUMBER (for the pending end=0.05-vs-0.10 decision)

The waste depends on the floor used. At the **R12-corrected floor (0.3 — the measured usable knee)**, over a
per-stage cosine anneal (1.0 → end over a stage):

| seg_temperature_end | dead fraction of each stage (at floor 0.3) | tail combined-grad lost vs warm |
|---------------------|--------------------------------------------|----------------------------------|
| **0.05 (LIVE ARM)** | **34%** (66/100 alive) | tail grad ≈ 4.1e-20 of warm |
| 0.10 (a proposal)   | **31%** (69/100 alive) | tail grad ≈ 1.9e-10 of warm |
| 0.30 (the alive floor) | **0%** (100/100 alive) | tail grad ≈ 1.7e-3 of warm |

**The crisp operator one-liner:** *end=0.05 wastes the seg lever for ~34% of each stage (the cold-dead tail),
and loses ~4.79e+09× combined seg-gradient on the flip pixels at the tail vs end=0.10 — but **end=0.10 is NOT
the fix**: at the true usable floor (0.3) end=0.10 STILL wastes ~31% of the stage (the old 0.1 floor falsely
reported end=0.10 as "0% wasted"). To keep the seg lever ALIVE for the whole stage the end must be ≥ 0.30, OR
the cold tail must be CE-blended (a non-temperature-sharpened seg term that keeps a usable gradient at the
tail).* This is the operator's decision; **R12 does NOT change the live arm config** (the daemon out-dir
`distortion_arm_l235_20260612T205102Z` is untouched).

Note the original R10 memo's "wastes 15% at end=0.05 / 0% at end=0.10" figures were computed against the
MISCALIBRATED 0.1 floor — they UNDERSTATED the waste. The corrected figures (34% / 31% / 0%) are the honest
ones the operator should use.

## B. THE R12 FINDING (MEDIUM — the miscalibrated floor) + the 2-LANDING FIX.

### B.0 The finding (MEASURED on the real scorer)

R10 set `SEG_ANNEAL_GRADIENT_FLOOR_T = 0.1` and the guard `seg_anneal_temperature_is_gradient_alive(T) ⇔ T ≥
0.1`. But the gradient at T=0.1 is **1.9e-10 of warm** (R12 real-scorer) — and R10's OWN documented table
already labeled T=0.1 "≈ dead" (|grad_lat|≈5.45e-12). The guard therefore classified a 10-orders-below-warm
dead-zone (T=0.1, 0.12, 0.15, 0.2) as "alive". The gradient is still USABLE (within ~2 orders of warm) only
down to **T ≈ 0.3** (ratio 1.7e-3); below 0.3 it craters (T=0.2 already 2.2e-5 ≈ 5 orders down). So the floor
should be **0.3**, the measured usable knee.

### B.1 Why it is a MEDIUM (a real guard miscalibration) and not a LOW

- It is the EXACT class of bug the R10 guard was added to PREVENT: a tuner who sets `seg_temperature_end`
  consulting the guard would be told 0.10–0.20 is "alive" and silently waste the seg lever for that entire
  range. The guard's whole job is to give the tuner the right boundary — and it gave the wrong one (off by
  ~3× in T, ~8 orders in gradient).
- It is INTERNALLY CONTRADICTORY in the lever code: R10's own docstring table called T=0.1 "≈ dead" while
  R10's own constant called 0.1 the alive boundary. R12 resolved the contradiction toward the measured knee.
- It is NOT a HIGH because the DRIVER does not gate on the guard (the intended late-cold lock-in is correct;
  the daemon's `seg_temperature_for_epoch` return values are byte-unchanged) — so no live run is corrupted.
  It is an OBSERVABILITY/schedule-design correctness defect, not a training-correctness defect.

### B.2 The 2-LANDING FIX (per "Bugs must be permanently fixed AND self-protected against")

**Landing 1 (the fix — `curriculum.py`):**
- `SEG_ANNEAL_GRADIENT_FLOOR_T` corrected `0.1 → 0.3` (the measured usable knee on the real scorer).
- Updated the `seg_temperature_for_epoch` docstring with the FULL R12 ratio table (T=1.0→0.05), the rationale
  for 0.3 (within ~2 orders of warm only down to 0.3), and the explicit note that R10's 0.1 mislabeled a
  dead-zone edge. Added the Lever-5 no-rescue note (margin × surrogate = scaled-down dead gradient).
- Updated `seg_anneal_temperature_is_gradient_alive`'s docstring to the corrected floor + the combined-floor
  rationale.
- **DEFAULT-PRESERVING + DAEMON-SAFE:** `seg_temperature_for_epoch`'s RETURN VALUE is byte-unchanged (only
  the floor CONSTANT + the separate GUARD changed; the driver/daemon calls `seg_temperature_for_epoch`, NOT
  the guard). Verified: annealed temps `[1.0, 0.858201, 0.517464, 0.181226, 0.05]` at (0,25,50,75,99) are
  identical pre/post fix; default end=None still returns static 1.0. The live daemon (loaded code at launch
  HEAD, no .py reload) is unaffected; a resume is bit-identical.

**Landing 2 (the strict regression guards — `test_all_layer2_levers.py`):**
- `test_seg_anneal_gradient_alive_guard_matches_floor_and_live_schedule` (R10 test, R12-CORRECTED): asserts
  `SEG_ANNEAL_GRADIENT_FLOOR_T == 0.3`, that the deep-dead temps the OLD floor mislabeled (0.2/0.15/0.12/0.1/
  0.08/0.05) ALL read DEAD now, and that the live 1.0→0.05 schedule has the corrected `alive == 66/100`.
- `test_r12_floor_is_at_usable_knee_not_dead_zone_edge` — measures the ACTUAL gradient ratio at the floor
  (0.3) vs the old floor (0.1) on saturated logits; asserts the 0.1 grad is < 1e-3 of the 0.3 grad (the
  correction separates usable from dead). NO-FAKE: constant surrogate → 0 grad (rejected); T-invariant →
  ratio 1 (rejected).
- `test_r12_margin_lever5_cannot_rescue_cold_dead_seg_gradient` — the multiply-a-dead-gradient guard:
  margin-ON cold grad ≤ margin-OFF (Lever-5 cannot revive), AND margin-ON ≠ margin-OFF at warm (the weight is
  NOT a no-op — a constant-1 weight would fail the warm inequality, so the cold test is non-vacuous).
- `test_r12_combined_seg_gradient_floor_on_real_scorer` — the headline real-scorer lens: combined grad
  collapses warm→cold (cold < 1e-3 of warm), the floor 0.3 sits >> the 0.05 tail (>1e3×), and margin-ON ≤
  margin-OFF at the tail. RESEARCH-ONLY tiny slice → [contest-CPU advisory] NON-PROMOTABLE.

## C. STANDARD CLEAN-CHECK (R12 lens) — default byte-identity preserved; no external consumers.

`grep` confirms `SEG_ANNEAL_GRADIENT_FLOOR_T` + `seg_anneal_temperature_is_gradient_alive` have NO consumers
outside `curriculum.py` (definition) + the test file — so raising the floor is fully self-contained and the
driver's loss path is unchanged. The `seg_temperature_for_epoch` body is byte-unchanged. Full suite result
(below).

## D. FRESH-EYES "QUESTION EVERYTHING"

1. **Can Lever-5 rescue the cold-dead surrogate gradient?** No — its detached `exp(−margin/τ) ∈ (0,1]` weight
   only scales DOWN. MEASURED margin-ON ≤ margin-OFF at every T (6.3e-20 ≤ 1.8e-19 at T=0.05). And the weight
   is SMALLEST on the confidently-wrong large-margin flips — it down-weights exactly the unfixable pixels.
2. **Is the combined floor a surrogate-only artifact?** No — the COMBINED (margin × surrogate) gradient also
   collapses ~4e-20 cold/warm on the real scorer. The floor holds for the live arm's actual loss.
3. **Was the R10 floor 0.1 actually the usable boundary?** No (the finding) — 0.1 is ~10 orders below warm;
   R10's own table called it "≈ dead". The usable knee is 0.3.
4. **Does raising the floor break the daemon?** No — the daemon calls `seg_temperature_for_epoch` (return
   values unchanged), never the guard. Resume bit-identical.
5. **Is end=0.10 the fix for the live arm?** No — at the corrected floor 0.3, end=0.10 still wastes 31% of
   the stage. The operator's real options are end≥0.30 OR a CE-blended cold tail. (Operator decision; not
   changed here.)

## Findings by severity

- **HIGH:** NONE.
- **MEDIUM-R12-1 (FIXED this round — lever code):** `SEG_ANNEAL_GRADIENT_FLOOR_T` was 0.1, mislabeling a
  ~10-orders-below-warm dead-zone (T=0.1/0.12/0.15/0.2) as "alive" — the exact mis-tuning the R10 guard was
  built to prevent, and internally contradictory with R10's own "T=0.1 ≈ dead" table. Corrected to 0.3 (the
  measured usable knee) + 3 regression tests + docstring. **This is the counter-resetting finding (1/3 → 0/3).**
- **LOW:** NONE.
- (The Q1 multiply-a-dead-gradient property + Q2 config-waste quantification are CLEAN — no defect; Lever-5
  cannot rescue, and the waste is now honestly quantified for the operator.)

## Test-run count

- Fast R12 + corrected-R10 structural tests (`gradient_alive_guard` + `r12_floor_is_at_usable` +
  `r12_margin_lever5` + `gradient_floor_collapses`): **4 passed in 0.98s.**
- R12 real-scorer combined-floor test (`r12_combined_seg_gradient_floor_on_real_scorer`): **1 passed in 59.31s.**
- Full lever suite: see the trailer (run detached, SIGURG-proof; pinned at memo-write time).
- R12 probe (`experiments/probe_r12_combined_seg_gradient_floor.py`): `R12_CLEAN=true`, real scorer,
  combined collapse 4.05e-20 cold/warm, margin-no-rescue confirmed, operator waste numbers emitted.

## Wire-in / provenance

6-hook (Catalog #125): #6 probe-disambiguator ACTIVE (`probe_r12_combined_seg_gradient_floor.py` is the
combined-floor + config-waste disambiguator); #1/#2/#3/#4/#5 N/A (review-round + a constant/guard/doc fix, no
new score-claim surface). Mission contribution: `frontier_protecting` (the corrected floor stops a future
tuner from silently wasting the seg lever in the 0.1–0.3 range the old guard mislabeled; the config-waste
quantification informs the operator's live-arm decision; the END remains a lower exact score, frontier UNMOVED
`0.19109982419209975` contest-CPU). Authority: all numbers `[contest-CPU advisory]` real-frozen-scorer-but-
tiny-slice NON-PROMOTABLE. No GPU launched, no daemon touched (distortion arm out-dir separate + untouched;
`seg_temperature_for_epoch` return values byte-identical so a resume is bit-identical), no Cool-Chic touched,
no archive-build region touched (Partner B's surface untouched).

**VERDICT: NOT-CLEAN (1 MEDIUM miscalibrated-floor finding in the lever code, FIXED via 2-landing) → fresh
counter RESETS 1/3 → 0/3.** The multiply-a-dead-gradient question is answered CLEAN (Lever-5 cannot rescue the
cold surrogate), and the operator's config-waste analysis is delivered (end=0.05 wastes ~34% of each stage at
the corrected floor; end=0.10 still wastes ~31% — the floor 0.1 had falsely reported 15%/0%). R13 must begin a
FRESH clean-pass count (0/3 → 1/3); the SEAL now requires THREE more consecutive clean rounds after this reset.
