# ddm_gs3 — the gestalt after submission (MAIN, 2026-09-03): where sub-0.12 lives, priced from the receipts, with the doors I re-checked and did NOT re-derive

Owner: MAIN. Axis: recall + S-arithmetic only; no scorer, no archive, no pointer move.
Tokens: `[no-triality] [p0-ledger-ok]`. Context: PR #140 posted (`ddm_pr140_submission_posted_20260903.md`);
operator standing GO "as long as it takes … be creative and weird … break walls … old n<600 that might
not hold up … near wins … synergies … recursive fractal optimizations."

## 1. The archive, decomposed (afr1, 180,002 B) and the measured floor of each section

| section | bytes | share | measured state of its rate axis | receipt |
|---|---:|---:|---|---|
| RC64 token stream | 113,411 | 63.0% | drained on this body: coder axis 0 B (jt23), model axis ≤211 B held-out (mi1), reorder 0 B within the fixed model (rr9), oracle on 21 taps 144,167 B (dc1), generic adaptive CM 3× worse (ef1) | `ddm_mi1_indicator_model_axis_20260824.md`, `ddm_dc1_decode_budget_conditional_coding_20260816.md`, `ddm_ef1_token_entropy_floor_20260822.md` |
| HPAC integer model | 13,515 | 7.5% | shipped stack already carries a multi-stage ADAPTIVE corrector (mi1 §"heavily engineered"); a paid probability model misses break-even 47× | `ddm_mi1_…` |
| semantic renderer | 30,856 | 17.1% | lossy quantization worse at every depth; distillation's pose cost tens of × the byte credit | PR #140 body; `ddm_wd2_ep60_advisory_refusal_verdict_20260815.md` |
| pose carrier (CPR1 low-rank luma basis × per-pair coefficients, 12-bit Rice) | 22,010 | 12.2% | lossless recode ceiling **−18 B** (dx1); rank/precision cuts damage pose 104.6–822.7× (jg1) | `ddm_dx1_dxi_recode_and_fruit_sweep_20260820.md` |
| container + residual | 217 | 0.1% | — | afc1 anatomy |

Demand: −42,016 B at held distortion (23.3%), or a different object. The optimistic SUM of every
secondary lever above is well under a quarter of the demand. **The current archetype cannot reach
sub-0.12 by rate work; the corpus's "sub-0.12 needs a DIFFERENT OBJECT" (m144) is re-confirmed
from the section table, not from a slogan.**

## 2. Doors I opened this turn and closed by RECALL (not by re-deriving)

- **Temporal / inter-pair context for the token coder** — already shipped: `hpac_integer.py`
  `prepare_frame_context(idx, previous_raw)` conditions on the previous pair's field; pose-WARPED
  context lost +12,262 B (xi1, `ddm_na6_arc_negative_audit_20260811.md` S1). Not a virgin door.
- **Sub-pixel antialiased Lane compositing into RGB, fitted through SegNet** — `ddm_lp1_lane_program_20260803.md`
  §2: post-hoc pixel compositing is NET-NEGATIVE and 100% of the collateral is receiver physics
  (SegNet's ~85 px ERF). Re-render, never composite. Closed at FORMULATION scope.
- **Decode-time online-adaptive context model (free-algorithm clause)** — the shipped corrector is
  already adaptive (mi1); generic adaptive CM/PPMd are 3× worse (ef1); the 21-tap oracle floor is
  above the shipped stream (dc1). A "tiny init + online adaptation" variant remains UNMEASURED but
  its ceiling is ≤13,515 B (32% of demand) — secondary; listed for rn1, not chartered.
- **Pose-carrier predictive lossless recode** — dx1 measured the ceiling at −18 B on the shipping
  body. Closed. (Recall saved a build here.)
- **Win-win edit cone as a 24th pointer move** — wwc1: FCD3's pose-screened subset saved 2,940 B
  but realized d_seg rose 0.000347→0.000387 (advisory), net **+0.00194 S**. Not a move.

## 3. The arithmetic of the only open door — born-generator accuracy

- bz2 (lb1's own renderer + pose carrier + a 47,779 B GT-fit generator): rate 0.067160, **37,124 B
  under cap**; d_seg 0.01299522 = 1.157× its 1.12% token error (bz2d). Sub-0.12 needs ~99.95%
  token accuracy uncorrected; with corrections at the measured 0.2909 B/site the cap room
  (≈16–37 KB) buys ≈55K–127K corrected sites ⇒ the generator must be wrong on ≤ ~0.05–0.1% of
  sites. Today: 1.12%. **Required: 10–20× fewer generator errors.**
- One argmax site = 100/117,964,800 = 8.477e-7 S = **1.273 B-equivalent**; a correction costs
  0.29 B. Corrections always pay (4.4×); tolerating errors never does. The sharp-optimum law is
  this ratio.
- r7→r10 doublings: d_seg_hat 0.0131 → 0.0046 → 0.0031 → 0.0025 ⇒ error ∝ steps^−0.44 with a
  receding target (r10 memo). Brute force to 6× lower error ≈ 64× more steps — infeasible. The
  CONFIG cure (#1091 in `ddm_qbz1_descent_rate_configuration_20260829.md`: 81% of LR budget on the
  worst-aligned objective; re-aim = 13.6×) has never run to endpoint. **qbr1 is the discriminator**
  (capacity vs optimization), invalidated by WC2-F1, being re-sealed by arm ddm_wc3.
- gf1's 5.09× capacity ceiling is "target-independent" but was measured on the 47,779 B packet
  (`ddm_gf1_generator_form_capacity_verdict_20260830.md`); bz2's decomposition leaves ~37 KB of
  cap room ⇒ a 1.5–1.8× larger generator is an OPEN question for rn1 to adjudicate at source.
- DDS1 measured that the born geometry carries only ~613 B of information the HPAC causal context
  does not already have — so "generator + HPAC-context corrections" ≈ HPAC alone unless the
  generator is standalone-accurate. The route is accuracy, not conditioning.

## 4. Critical path to a sub-0.12 exact row (as long as it takes)

wc3 re-seal (hours) → MAIN fires QBR1 six cells on the Metal slot (~17.8 h) → if
`OPTIMIZATION_LIVE_DISTORTION_ROUTE`: same-object n600 realization → corrections layer under cap →
receiver-closed archive ≤137,986 B → T4 row (Modal ~$0.30). Parallel: scm2 (SCMDL G/M refit,
rate corner), fpc1 (from-raw-video pipeline = the substrate every successor needs), sp9 (distinct
receiver contract), rn1 (the operator's re-open sweep). MAIN's scorer/Metal lane idles BY STATE
until one of these hands it a valid order.

## GESTALT-DELTA

Sub-0.12 is a **generator-accuracy** problem (10–20× fewer wrong sites at ≈50 KB), not a coder
problem. Every coder door on the current object is measured shut within 1.3–1.6× of its floor.
The walls that could still be fake are the ones WW1 §3 names — and the first (born-object
capacity-vs-optimization) is exactly the experiment now on the critical path.

Own-vehicle frontier: **afr1 S 0.14797617125559104 @ 180,002 B [contest-CUDA T4 n600]** — unmoved.

## ADDENDUM 2026-09-03 ~21:5xZ — gc1 closes CAPACITY on the GF1 form; the residual is the wall, not the count

`ddm_gc1_generator_capacity_control_20260903.md` (commit 226a7fecf; strict full-n600, 365 retained files,
all four residual-closed fields reproduce the exact 117,964,800-byte target):

| packet | mismatches | ACTUAL residual (domain-matched coder) | combined | vs 85,020 B cap |
|---:|---:|---:|---:|---:|
| 47,971 B | 1,334,939 | 359,280 B | 407,251 B | 4.79× |
| 53,277 B | 985,100 | 348,260 B | **401,537 B** | 4.72× |
| 65,093 B | 822,610 | 343,128 B | 408,221 B | 4.80× |
| 76,113 B | 725,965 | 340,552 B | 416,665 B | 4.90× |

Three facts the section table did not have:
1. **Capacity buys little:** 1.59× bytes → 1.84× fewer mismatches (local exponent −1.228); the crossing at
   46,804 mismatches extrapolates to a **686,618 B packet = 9.62× the cap**. CAPACITY-CLOSED for this form.
2. **The residual is priced by its HARD sites, not its count:** removing 608,974 mismatches saved only
   18,728 B of actual residual. The generic 0.2909 B/site law UNDERPRICES the endpoint by 129,369 B — it was
   calibrated on a different residual population (m111/m118 genus: price the ceiling on the real object).
3. **Generator + honest residual = ~400 KB vs the HPAC's 113 KB for the same exact field.** The context mixer
   beats "generate then correct" by 3.5× even at 1.6× generator capacity, because the generator's residual is
   the same SegNet-jitter "wrong half" the mixer already prices best, spread over MORE sites.

**Consequence for the small-body route:** the born object can reach sub-0.12 ONLY if its OWN field is nearly
exact without corrections (≤ ~0.05% wrong sites) — i.e., by OPTIMIZATION (qbr1's zero-native treatment, burning)
or by a form whose atoms match the classes that fail (gc1: square atoms barely improve Lane and add ~39,000
Movable errors → class-protected anisotropic/curve atoms; that is gf2's static/dynamic split with sparse
Lane/Movable events). Adding capacity to the existing form is closed. GESTALT unchanged in kind, sharpened in
mechanism: sub-0.12 = born-field accuracy by optimization or by class-matched form, never by size.
