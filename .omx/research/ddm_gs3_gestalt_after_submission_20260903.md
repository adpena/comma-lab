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

## ADDENDUM 2 — 2026-09-03 ~20:2xZ — gf2 closes the static/dynamic form at its static ceiling (10.5×)

`ddm_gf2_static_dynamic_generator_form_20260903.md` (MAIN adjudication): one shared static field + converged rigid
per-pair alignment leaves 3,072,488 mismatches (2.605%) vs 292,264 plausibly repairable with the whole packet as
dynamic — the mismatch mass is per-pair boundary MOTION, not sparse events; per-GOP fields are closed by the same
measured motion (DERIVED). With gc1 (capacity, 9.62×) and gf2 (form, 10.5×) both closed, **the burn is the last
open door**: sub-0.12 on the small-body route = OPTIMIZATION_LIVE, or the Pareto-shelf conjecture becomes a measured
family verdict.

## ADDENDUM 3 — 2026-09-03 ~20:5xZ — qn1 re-derives the born target: the QXR1 falsifier was 73× too loose

`ddm_qn1_qbr1_n600_realization_ticket_20260903.md` (Opus arm; a42e0fa5f/2d7241b11): at the falsifier's pose corner
(d_pose ≤ 1.25e-4) S = **1.1064**, not < 0.12; the d_seg that actually clears 0.12 at that pose and ≤ 137,986 B
is **1.3646784205e-4** — i.e. **≈0.014% wrong sites (~16K of 117.96M)**. The born field is at 1.12–1.4% → the burn
must show a **~100× accuracy jump**, not 10–20×. Restated gestalt bar: sub-0.12 on the small body = a born field
wrong on ≲16,000 sites at ≤ 137,986 B with pose ≤ 1.25e-4. The n600 realization ticket is one command
(`experiments/ddm_qn1_qbr1_n600_realization_ticket.py ticket --scorer-claim-id …`), dry-run bound to cell 1 step
2000 (archive 106,626 B, receiver bit-identical both ways, 14/14 refusals fire). Read the eventual row against
1.3647e-4, never the falsifier.

**Fold-back program opened (`ddm_fb1_foldback_program_20260903.md`):** post-hoc found the laws, training is where
they pay. Fold #1 = ft1 (Opus, live): fine-tune the shipped 30,856 B renderer from its own weights with the realized
aligned loss + pose@0 — same bytes, tokens untouched; each 1e-5 of d_seg = −0.001 S on the frontier object.

## ADDENDUM 4 — 2026-09-03 ~21:4xZ — ft1's corrections (ERRATA to §1 and to the ft1 charter; append-only)

`ddm_ft1_shipped_renderer_aligned_finetune` (Opus; commits 77af37116…af7440962, 40 tests) measured at source:
1. **ERRATUM §1 table:** afr1's `semantic_renderer` section is **36,130 B, sha 17e0fd0b…, SM3R v1 MODE_ROW_PRUNE_MIXED**
   (width 96, keep_percent 1, per-tensor depths {3,4}; FiLM weights keep 2 of 192 rows) — NOT the 30,856 B /
   39d1be52… uniform-int4 section, which belongs to the superseded gb1-generation container. The 30,856 / 22,010
   split in §1 was carried from bz2's lb1-era anatomy; afc1's 53,076 B "framing" is the authoritative total.
   Export size is value-independent, so a fine-tune re-exports at exactly 36,130 B by construction.
2. **`experiments/results/mlx_fleet_gt_cache/gt_n600.npz` is PyAV-lineage** (`lstars` differs from `gt_cache_av.pt`
   at 2 of 117,964,800 sites); DALI-vs-PyAV differ at **20,671 argmax sites** (pose MSE fork 1.4061e-4 = rf1's
   additive fork). Training a renderer or a born field against it aims at 20,671 sites the contest does not
   score — 87% of d_seg's entire 23,757-flip budget. The T4-scored table is `gt_cache_dali.pt` (sha a91d9825…);
   `ddm_up2` already labels the npz `DEFAULT_AV_GT`. Step-0 n600 on DALI: d_seg 2.0387e-4 vs T4 2.0139e-4 (1.23%).
3. **Same-object promotion pose ceiling ≈ 1.694e-5, not 1.25e-4.** The m110 "absolute budget" is the sub-0.12
   allowance for a new object with the full distortion room; for a move on afr1, `d_pose ≤ 1.25e-4` would cost
   +0.02737 S = 5.44× the credit of a 25% seg cut. Promote iff exact S < 0.14797617125559104, period.
4. **Pre-registered ceiling for this actuator family:** msr1's flow-balance bound (learned renderer weights
   included) = 2,123 px = **8.94% of d_seg** — below the charter's 10% falsifier; the arm expects the falsifier to
   fire. Its run (`aligned_dali_lr2e5_s1800`, detached, 2.6–3.3 h, no pose term ⇒ a COUPLING measurement,
   mechanism-reduced, no family verdict) exists to measure Δd_pose/Δd_seg coupling; FO-1..3 in its FIRE_ORDER.sh.
5. **Cost:** F 12.07 s · 3.63 s/step · 1.14 s/pair-eval on CPU beside the burn ⇒ 36 min/epoch; w96b's 65-epoch
   window = 51.7 h on CPU (not deliverable while Metal is owned).
Gestalt consequence: unchanged in kind — and every training-side arm from now on targets `gt_cache_dali.pt`,
never `gt_n600.npz`; MAIN is checking whether the QBR1 seal itself carries the PyAV table.

## ADDENDUM 5 — 2026-09-03 ~23:1xZ — fold #1's first rung: falsifier FIRED (+31%), INSTANCE scope; the next rung is defined

ft1 step 600 (`aligned_dali_lr2e5_s1800`, seg-only, lr 2e-5, 1,800-step cosine; DALI target; EMA shadow; n600
advisory): d_seg 2.0387e-4 → **2.6753e-4 (+31.23%, +7,510 flips = 12.4× CE1's 605-flip A/A floor)**; +0.006366 S on
the seg leg. Not one evaluated point improved on the shipped renderer (commit f505674dd). Three transferred
constants (LR from CE1's plateau on a different init — the object's own PR130 tail LR is 2e-7; τ 21.7× faster per
step than w96b's 39,000-step law; DALI asks the renderer to override 9,179 token sites) — INSTANCE, not family.
Coupling Δd_pose/Δd_seg on the step-600 checkpoint (n200 seeded random) in flight — the number that decides whether
pose-in-loop is mandatory at this size.
**Next rung (FOLD-AFTER-BURN, Metal ≈ 104× faster than CPU):** init = shipped 36,130 B SM3R weights; lr 2e-7 (the
object's own tail; BS16: −3.03% seg in 30 steps); pose term at step 0 (w96b/qbr1 law); τ at w96b's per-step rate
over its full window; DALI target; EMA per epoch; per-epoch B/H/W. Read against the same-object pose ceiling
1.69e-5 and promote iff S < the pointer. Until the coupling number lands, fold #1 is a wrong-LR probe, not a verdict.

## ADDENDUM 6 — 2026-09-03 ~23:3xZ — fold #1 rung 1 CLOSED by the coupling number: Δd_pose/Δd_seg = 217

`retained/verdict_ft1_step600.json` (n200 seeded random, DALI lineage, `[macOS-CPU advisory]`): base d_seg 2.0030e-4 /
d_pose 9.00e-6; step-600 export realized d_seg 2.6962e-4 (+34.6%) / **d_pose 0.015074 (+1,674×; pose term 0.388)**.
**coupling Δd_pose/Δd_seg = 217.30** — above rf1's 166.8 ⇒ a seg-only renderer change closes by arithmetic at this
size: every unit of seg movement drags ~217 units of pose. Realization gap: the trained float weights read d_pose
0.0837 / d_seg 4.45e-4 while the int4 row-prune export realizes 0.0151 / 2.70e-4 — the export path REPAIRS most of
what training broke (the "manufactured" mechanism, now measured on this actuator). tv1/tv2's co-location law
(seg slack and pose damage live in the same pixels) is confirmed on the renderer: they are not separable by a
seg-only loss.
**Consequence:** fold #1 survives only as a JOINT rung — pose term at step 0 with a weight derived from this
coupling (≈217× the seg weight in S units), lr 2e-7, full τ window, on Metal after the burn — and its expected value
is LOWER than fb1 assumed: the renderer's 2e-4 seg residual may be inseparable from its pose behaviour at 36 KB.
Read it as: the shipped renderer is at a joint optimum the post-hoc chain already found; the fold-back's real
targets are the born trainer (qbr1's config) and the population pipeline, where pose is already in the loop.

## ADDENDUM 7 (2026-09-04) — two post-hoc doors on trained objects closed the same way

ft1 (renderer weights, coupling 217) and ar1 (render sampling, 0.76× with sign reversed) close by the SAME mechanism:
a trained object has adapted to every deterministic choice made in its loop (its export, its sampling lattice, its
LR), and a post-hoc change of any of them is paid in full at the scorer. The accuracy half of sub-0.12 therefore has
exactly one open door — **changes made INSIDE the born trainer's loop**, raced one lever at a time from the same warm
start (vr1 rows 1/3/4/7 first, AA-in-loop last). The burn (QBR1 → qn1 n600 realization) remains the only vehicle.
Calibration side-benefit from ar1: the MPS↔CPU axis gap on this vehicle is 0.06% (d_seg) / 0.14% (d_pose) — the
burn's own MPS readings are trustworthy to that bound (MPS is still never authority).
