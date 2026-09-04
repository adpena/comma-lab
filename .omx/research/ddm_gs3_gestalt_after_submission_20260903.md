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

## ADDENDUM 8 (2026-09-04) — the born trainer's excursion has a mechanism, and the loss could not see it

Two arms in one night: ng1 (the QBR1 transition is cold on ONE axis — AdamW moments, first step 6.46× larger; no LR
schedule exists in this trainer) and sd1 (the loss's fall was the τ anneal deflating the surrogate by 40.5% on a frozen
field; at fixed τ it tracks the exact term; the excursion is rare-class over-paint peaking at step 2,000, mass-
conserving, and the trainer has no area cap). The born field did not get harder to fit — the optimizer started cold,
painted too much Lane and Movable, and the objective's own clock hid it. The accuracy half of sub-0.12 now has an
ORDERED race inside the loop: warm transition (sealed) → area cap (chartered) → margin weight / τ at δ_R scale. Every
prior "the surrogate is miscalibrated" reading in this corpus should be re-read against the τ-schedule identity first.

## ADDENDUM 9 (2026-09-04 05:40Z) — the post-submission wave, in one page

**Pointer:** afr1 S 0.14797617125559104 @ 180,002 B [contest-CUDA T4 n600] — UNMOVED since 08-31; PR #140 posted
09-03 (pointer field now records it). No exact row was bought this wave; every unit below is MEANS toward the one
door the accuracy half still has.

**What the fifteen arms measured (all $0, all committed, all in the three legs):**
- ft1 / ar1: on a TRAINED object, changing any in-loop choice post hoc is paid at the scorer — renderer weights
  (coupling |Δd_pose|/|Δd_seg| = 217, rf1 166.8; law `renderer_seg_pose_coupling_shipped_object_v1`) and render
  sampling (footprint render 0.76× d_seg, 32/32 worse; the 6.39× law is an achievable-signal upper bound, domain now
  says so). → Only in-loop levers remain.
- QBR1 (seeds 1–2 read; 3 running): BOTH cells open a +22% excursion from a cold optimizer transition (AdamW moments
  are the only cold axis — no LR schedule exists; 2e-4 is r10's own terminal LR) and end +6.6–9.3% above the warm start.
  The native-interface-OFF treatment loses both seeds → adjudication decided NEGATIVE at second order.
- sd1: the loss never saw it — the τ anneal deflates the surrogate 40.5% on a frozen field (exact identity); at fixed τ
  the surrogate is faithful. The excursion is rare-class OVER-PAINT (Lane ×1.09, Movable ×1.06 at step 2k, mass-
  conserving) and the trainer has no area cap.
- gm1: 77.7% of the seg gradient is WASTE (correct pixels outside m_safe); the τ band [2δ_R, δ_R] removes 46–97% of it;
  the pixel-weight lever's headline setting is exactly INERT; levers couple through Lane.
- dr1 / eq1 / ql2 / ql3: δ_R n600 = 0.021882 (n96 prefix 11.7% low, annulus-specific — a new detector law); the
  equations gate is on the commit path (29 → 0, 55% of the backlog was `ratified`⊂`stratified`); two harnesses were
  deciding R-safety with the retired constant (anti-conservative) — census by VALUE, not by comment.

**The ordered race, sealed and staged (each a twin of the measured cold control, one lever, pre-registered):**
1. **ng1 warm transition** (r10 AdamW state carried; cold first step 6.46× warm) — fires at the chain's end (≈11:45Z)
   with one reviewed command; falsifier S_hat(5k) < 0.398768 and below control at every milestone.
2. **ng2 area cap** (λ_Lane 2799.8, λ_Movable 7587.4 from the trainer's own bincount; cap gradient only 1.25% of the
   recall term — prior lowered honestly) — re-seal as a WARM twin if 1 wins.
3. **ng3 τ band** [2δ_R → δ_R] (schedule-leg deflation measured 4.76× smaller at $0; peak RSS 41.5 GiB).
Then winners pairwise (m164). Every cell ≈3 h Metal. If a cell ends below 0.398768 the qn1 n600 realization ticket
is the byte-close path to an exact row.

**Equations leg (`tac.canonical_equations`):** `renderer_seg_pose_coupling_shipped_object_v1` · `aa_sdf_observation_footprint_render_dseg_v1`
(domain refined) · `annulus_restricted_prefix_bias_detector_v1` · `margin_band_satisficing_threshold_v1` (n600 anchor) ·
`muon_finisher_schedule_warmstart_and_lr_anneal_v1` (ng1) · `chan_vese_area_constraint_birth_balance_v1` (ng2) ·
`scalar_top1_top2_margin_is_exact_distance_to_flip_v1` (sd1/gm1); the τ-schedule deflation identity awaits ng3's cell as its
second anchor.

**What this wave did NOT do:** move the pointer. Said plainly, per the means/ends firewall.

## ADDENDUM 10 (2026-09-04 14:40Z) — md1: the micro→macro bridge closes the accuracy corner on THIS vehicle

MEASURED (ddm_md1 456c74551; 71 checkpoints × live+shadow × 2 cells × 32 pairs; calibration 0 in integers; law
`checkpoint_trajectory_error_partition_v1`, two anchors): **the cold optimizer's damage is complete after 16 updates**
(live d_seg_hat 0.0025556 → 0.0060335; Lane painted 1.296× GT; 24,336 sites born wrong, 45.7% Road→Lane) — the
milestone record "peaks at 2,000" only because the EMA shadow is a 1,086-update low-pass; the live field never returns
to the init's quality. The run creates 2.21× the error it repairs (removed 4.233% of the inherited error). Warm moments
REDIRECT rather than damp (prediction 2 falsified 20/20; Jaccard 0.20–0.37 at identical steps).

**The bridge:** PERSISTENT sites = **62.0%** of the terminal d_seg on the shadow (the object the archive re-encodes);
**delete every optimizer-reachable site and the born field is still 12.75× the sub-0.12 accuracy corner** (persistent
floor 0.00174 vs the 1.3647e-4 target). The combined credit ceiling of EVERY schedule/optimizer/objective lever is
**1.61×** on d_seg against **20.57×** needed. The persistent set is named: 11,842 sites, 64.8% on a Lane edge, GT-Lane
enriched 51.5×, 33.7% deeper than 25 δ_R (the inherited error is DEEP; the error the run creates is shallow, 99.8%
within 25 δ_R).

**Consequence (gestalt):** with capacity (gc1), form (gf2) and now OPTIMIZATION (md1) closed, the accuracy half of
sub-0.12 on the small born body is closed on this vehicle by three independent instruments. ng2 (area cap) and ng3 (τ
band) still measure the schedule's share of the 1.61× — worth having, no longer a path to the target. The qn1 n600
realization ticket is moot for sub-0.12 on this vehicle. What the persistent set demands is a different
REPRESENTATION at the Lane-edge sites, and the rate corner (−42,016 B at held distortion; CLAUDE.md banner) remains
the live demand the corpus already named. Free byte found on the way: the QBF1 pose head is never read by the
objective (1,836 params, zero gradient across three sealed cells) = **2,014 B = 1.89%** of the cell archive — a
packet-ABI cut on the born vehicle, not on afr1. Micro lever for the next born generation (md1): clip/ramp the first
~64 updates rather than substitute another run's moments.
Equations leg (`tac.canonical_equations`): `checkpoint_trajectory_error_partition_v1` (md1) ·
`muon_finisher_schedule_warmstart_and_lr_anneal_v1` (negative anchor, ng1) · `chan_vese_area_constraint_birth_balance_v1` (ng2, live).

## ADDENDUM 11 (2026-09-04 15:05Z) — lb1: the fourth instrument; the accuracy half of sub-0.12 on the small born body is CLOSED

MEASURED (ddm_lb1 1cb05f03d; prereg ced026cdd BEFORE the numbers; md1's argmax reproduced bit-for-bit, d_seg_hat gap 0.0):
a **perfect-Lane ORACLE** (exact Lane authority both directions) removes 63.12% of md1's persistent set with zero harm
and still leaves the born field at **8.94× the sub-0.12 corner** (4.70× jointly with perfect optimization). The landed
lane-band carrier, 162 configurations, improves d_seg in NONE (best +0.008481 S): it recalls 93.5% of the lane but
its precision ceiling is 0.565 against a break-even of **0.909**, because P(born wrong | GT=Lane) = 0.0996 — the born
field is already 90% right about the lane; its largest casualty is Lane itself. Bytes were never the constraint
(2,832 B). Corrections: vr1 row 10's "Lane band d_seg 0.00087" is the witness TARGET (FN 0.00046; FP as full authority
0.00396; gauge `measured=False`); the carrier's coder is `serialize_lane_band_rd` (LBND2), not the residual sidecar.

**State of the accuracy corner (× the 1.3647e-4 target):** born terminal 20.57× → persistent floor 12.75× (md1) →
perfect Lane carrier 8.94× → + schedule ceiling 5.55× → perfect carrier AND perfect optimization 4.70×. Four
independent instruments — capacity (gc1), form (gf2), optimization (md1), class-matched carriers at their ceiling
(lb1) — close the accuracy half of sub-0.12 on the small born body. The rate corner (−42,016 B at held distortion) is
the only arithmetic door left, and no lever reaches it. **NOT authorized:** a born trainer with Lane held in-loop.

**Transferable law (lb1):** an authority-substitution lever is priced by the INCUMBENT's per-class accuracy, not by
the lever's own fidelity — break-even precision = P(inc correct | ¬C) / (P(inc correct | ¬C) + P(inc wrong | C)) is one
line, computable before anything is built; lb1's NEXT #2 wires it into the charter lint. Also closed: a silent
`centerline_deg>3` truncation in the band coder (23.33 m lateral error) — guard + tests, live count 0.
Equations leg (`tac.canonical_equations`): `v8_geometric_rate_decomposition_v1` (Lane carrier, new vehicle) ·
`checkpoint_trajectory_error_partition_v1` (the object scored against).

## ADDENDUM 12 (2026-09-04 15:40Z) — pr1: the re-solve measured; the renderer axis closes on SEG; a −1.03e-4 S candidate

MEASURED (ddm_pr1 c7b537053, n600, jg5's GN solver — up2's ±2 radius would have measured the solver): the terminal
pose re-solve on ft1's renderer-change candidate recovers **16.42×** (598/600 pairs; jg5's 8× was a token-edit number)
→ k_post **13.82** (k_pre 228.45); re-solved d_pose 9.43e-4 is still **41.5× over** the payable bar; the carrier
re-solve costs **+125 B**. The premise the whole closing arithmetic rested on — local linearity — is FALSE: the
reflected step raises d_seg 21.55× more than the forward step. **The seg-only renderer axis is closed because the seg
gain is unreachable along the only exportable axis, not because pose is unpayable.** Residue = the int12 carrier's
representation limit (100% of pairs want a GN step beyond ±2; 9.67% beyond the lattice; ten pairs own 69% of the
post-solve mean).

**Pointer path found on the LIVE afr1 object:** the receiver's per-pair frame-0 selector — 39/600 pairs beat their
shipped mode by >1% (pair 85's shipped op is actively harmful); priced through the receiver's own blob formula:
**+36 B for net −1.032e-4 S** → projected **0.14787295862740366** `[macOS-CPU advisory projection]`. Needs: an encoder
for the selector op (the runtime is decode-only), a batch-8 re-measure, a byte-closed splice, and a T4 row; promote iff
exact S < 0.14797617125559104. Small, real, and the first exact-row candidate of the wave (ddm_fs1).
Equations leg (`tac.canonical_equations`): `renderer_seg_pose_coupling_shipped_object_v1` (domain PRE/POST re-solve, 3
anchors).

## ADDENDUM 13 (2026-09-04 16:20Z) — bh1: the operator was right; the bugs, ranked by what they corrupt

MEASURED (ddm_bh1 0dca1e3b2; 15 findings, 2 fixed, closures recomputed independently — no verdict moves):
1. **The born trainer trains AND scores against the PyAV table, pinned by sha** (`ddm_qbt1_qbflow_trainer.py:123,:246,
   :2067-2073`; milestones via the burn prep :615). Bound: 20,671 sites = 0.017523 S = ≤7% of today's born d_seg_hat
   but **1.28× the entire sub-0.12 d_seg budget**. Every cell verdict's SIGN survives (same target both arms); the
   absolute levels and the falsifiers (0.425149 / 0.485677) do NOT transfer to DALI or to the 0.12 target. $0 cure:
   `_retain_eval_outputs` keeps `segnet_argmax_u8` per pair per milestone — a DALI d_seg_hat for every existing
   milestone is a re-read of retained bytes (MAIN, below).
2. **The lineage preflight gate is blind to `.npz`** (`src/tac/preflight.py:2466-2469`, regexes require .npy/.pt) and
   WARN-ONLY (:2863) — the detector built for this harm cannot see its largest live consumer; widening lights ~372
   historical consumers → needs a live-count plan (chartered by bh1, not patched).
3. **The dual constraint measures an UNWEIGHTED within-class error and penalises an HT-weighted one**
   (`qbt1:598-612`): heavy-stratum share gap 1.60× Lane / 1.39× Movable — the 8 heavy pairs are 40% of the population
   and get 24–25% of the driver; propagates into ng2's derived λ. Owned by the next trainer generation.
4. **The memory guard over-trusts inactive pages** (free + ALL inactive counted reclaimable; `tools/mem_basis.py` was
   written to refuse exactly that): 1.204× over-trust now, 4.2× under load per its own anchor; never subtracts the live
   cell's footprint → gv1's admission function must use the mem_basis basis and subtract live cells.
5. A pin that can never fail (`verify_pins` synthesizes `wd3_reference` when absent); four lineage-unlabelled axis strings.
Fixed: the reseal tool's receipt attested the OUTPUT sha as the input (17 tests); the δ_R PRODUCER still defaulted to
the n96 prefix that made the retired constant (3 tests; law anchor `producer_default_reinfects_cured_constant`).
Prose overstatements corrected: md1's 51.5× mixes n32/n600 and its Movable 1.62× is 1.864×; gm1's "EXACTLY inert" is
0.042%; lb1's 5.55× applies the born-field 1.61× to a post-oracle field. Verified clean: tau_for_step, both STEs,
build_initial_state (EMA into live AND shadow), balanced schedule, milestone ema_scope, float32 argmax before the f16
cast, upstream preprocessing order, HT estimators, ng3's band at step 0, ng2's cap caveat. EMA τ = 1,086.24 updates
confirmed (md1's low-pass): a milestone's shadow still carries 39.8/15.8/6.3/1.0% of the init at 1k/2k/3k/5k.
Equations leg (`tac.canonical_equations`): `annulus_restricted_prefix_bias_detector_v1` (bh1 anchor) ·
`checkpoint_trajectory_error_partition_v1` (recomputed 62.0107%).
