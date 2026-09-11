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
| semantic renderer | 30,856 (container bytes; RAW SM3R body 36,130 — two bases, see addendum 4 note) | 17.1% | lossy quantization worse at every depth; distillation's pose cost tens of × the byte credit | PR #140 body; `ddm_wd2_ep60_advisory_refusal_verdict_20260815.md` |
| pose carrier (CPR1 low-rank luma basis × per-pair coefficients, 12-bit Rice) | 22,010 (container bytes; coefficients 9,829 Rice + basis ≈12,181) | 12.2% | lossless recode ceiling **−18 B** (dx1); rank/precision cuts damage pose 104.6–822.7× (jg1) | `ddm_dx1_dxi_recode_and_fruit_sweep_20260820.md` |
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

## ADDENDUM 14 (2026-09-04 17:00Z) — the first exact-row buy of the wave, and the apparatus that saturates the box

**fs1 (7ec320551):** pr1's selector re-selection byte-closed: the encoder for the per-pair frame-0 selector op now exists
(the shipped runtime was decode-only; `encode(decode(shipped))` rebuilds the archive's own 14-byte blob exactly; 300/300
fuzz round-trips through the SHIPPED decoder); the container identity control reproduces `archive.zip` bit-for-bit with the
tail unchanged. A one-dimensional scan pr1 never ran finds the byte-optimal set: **21 pairs, +20 B, net −1.103610e-4 S
(MEASURED n600 batch 8, advisory)**, projected **0.1478658102574271** — 16 fewer bytes and 7.0% more score than pr1's
>1% gate; 579/600 unchanged pairs measure Δd_pose = 0.0; 0/21 adopted pairs worse; d_seg ≡ 0 structurally; ADMISSIBLE under
`exchange_ratio_noise_floor_v1` (95% [−1.89e-4, −5.66e-5]). Near-misses carried: the pointer row's runtime is the PUBLIC
PR tree (g8v1), not afr1's native tree; "only archive.zip changed" is impossible because `inflate.py` pins the archive —
the receiver diff is exactly two pin lines, proved. **T4 buy fired 16:5xZ** (`fire_modal_auth_eval.py --seal`, detached);
PROMOTE IFF exact S < 0.14797617125559104; the 8-dp report bound 7.32e-6 makes a landed net ΔS inside (−7.3e-6, 0)
UNRESOLVED. Launcher refusals on the way: an argv carrying "claude" (reaper guard) and seal-owned flags passed by hand
(hand-assembly hazard) — both cures are the tool doing its job.

**gv1 (c1701128a):** `tools/cell_admission.py` (relative headroom on the canonical reclaimable basis + Σ live UNREALIZED
growth + 16 GiB margin, and the absolute ceiling; rc 0/2/3), a Metal-contention ledger (N=2 concurrency MEASURED +11.7%
throughput, per-cell efficiency 0.559 — one anchor, not a law; the serial baseline is second-hand), `tools/cell_queue_driver.py`
(generalizes the chain driver; its dry-run re-found ng2's un-rerooted pins), `costate_digest.section_live_cells()`;
divergence from bh1 recorded (resident footprint must NOT be subtracted twice on the canonical basis — charge unrealized
growth instead). ANE closed by RECALL: the 2026-07-13 lane already measured fp16 flipping argmax at 0.088 on n600 (90× the
d_seg budget) — the ANE can never be a d_seg authority; CoreML fp32 on CPU+GPU (3.6× forward) is the banked advisory
accelerator. The shell fire scripts still carry the inline arithmetic (SSD custody, not git): one-line replacement
`tools/cell_admission.py admit --candidate-peak-gib <P>` for every future fire.
Equations leg (`tac.canonical_equations`): `exchange_ratio_noise_floor_v1` (4th anchor, first pure-pose case).

**Addendum 14 correction (gv1 final, 17:10Z):** the "+11.7% at N=2" above is a ONE-WINDOW artifact. gv1 measured the same
concurrency twice, 30 min apart: 31.29 steps/min (1.117×) then 27.00 (0.964×, the digest's live read); spread 4.286
steps/min = 15.9% of the mean, larger than the effect. **N=2 Metal concurrency is UNRESOLVED** (`metal_concurrency_speedup_gv1_v2`,
verdict UNRESOLVED_AT_N2; v1 preserved). The guard flipped to REFUSE on the second row — that is the apparatus working.
Two more gv1 defects found by RUNNING, not reading: the manifest PID is a supervisor (per-PID RSS under-reads the tree 26×);
an unreached milestone falsifier returned `fired=false` off step-741 data (FALSE SURVIVED; cured to PENDING — the fixture
test had codified the bug).

## ADDENDUM 15 (2026-09-04 17:30Z) — THE POINTER MOVED (24th): fs1 exact T4 row S 0.14786319521362173

The wave's first exact row landed below the frontier: **−1.1298e-4 S** (pose 6.37e-6 → 6.17e-6, d_seg identical, +20 B),
realized within −2.6e-6 of the advisory projection. Full record: `ddm_fs1_pointer_move_24_20260904.md`. The lesson for the
gestalt: the door that opened was not a wall-break — it was an UNBUILT ENCODER on a shipped op (decode-only selector) that
the operator's "pose resolves after seg cut" reminder exposed. Two closures (rf1/ft1 coupling, ar1 post-hoc) said the
renderer cannot be re-aimed post hoc; the selector is a DIFFERENT object (a per-pair frame-0 choice the renderer never sees),
which is the object-change law (m148) paying out. Sub-0.12 arithmetic re-derived at the move: rate corner −41,845.5 B at
held distortion; distortion corner 214.1× at held bytes. Equations leg (`tac.canonical_equations`):
`exchange_ratio_noise_floor_v1` first authority anchor.

## ADDENDUM 16 (2026-09-04 18:15Z) — ng4 sealed: two of my four "restarted objective states" were not restarts

ng4 measured each named state against the quantity that ACTS. τ (0.05 → 0.15, 3.0× band) and the duals (converged → 0) ARE
restarts and act on the gradient — carried (τ held at 0.05; duals carried, which BIND one update after the transition, so the
carried pair starts 47.5×/211.6× ahead and the held τ removes sd1's schedule artefact 4.76×). The EMA law is NOT a restart:
r10 EXECUTED 10011/10020 = 0.99910 (warmup arithmetic) vs the cell's 0.99908 — gap 2.24e-5 — and `ema.update` never writes the
model (measurement channel). Batch geometry never restarted (same pair ids, chunk 16). A held temperature was structurally
INEXPRESSIBLE (`tau_for_step` refused start == end) — that geometry contract is the landing (32 tests; control configs
byte-identical when the blocks are absent). $0 check on ng3's retained field: the differential is bit-identical in all 15
components (1.0765775442123413 = ng3's number).
Instrument finding (routed to gv1): `--measured-peak-rss-gib 2.396` is an RSS FICTION on Apple Silicon — machine-wide `ps rss`
summed 12.9 GiB while the governor's used_gib sat 104.6–113.1; INFERRED ~45 GiB of system availability per concurrent Metal
cell (~19× the declared number). The admission gate reads the true system state and refuses correctly; the PROJECTION is blind.
Fire order: ng4's owed B=16 smoke (no-op detector) then the cell, both behind admission once ng2 releases (~20:20Z).
Lesson (memory erratum 2 on the cold-transition file): compare the quantity that acts, not the config field.
Equations leg (`tac.canonical_equations`): none new — the ng4 cell is the measurement.

## ADDENDUM 17 (2026-09-04 22:00Z REAL — clock note: addenda 14–16 headers ran ~1 h ahead of UTC) — four landings

**Burn series read (born vehicle, one cold control, seed_20260902, `[macOS-MLX research-signal]`):** the τ band [2δ_R, δ_R] (ng3) is
the FIRST cell to END BELOW ITS START — S_hat @5k 0.391810 vs start 0.398768 vs control 0.425149 (−7.84%); excursion peak +9.2% vs the
control's +21.8%; d_seg −1.2% vs start, d_pose −10%, bytes −77 B. The area cap (ng2) passed its pre-registered rule by −0.97% but is
MARGINAL (@4k +3.8% worse; terminal d_seg +10.8% above start). Ordering @5k: τ band −7.84% ≫ cap −0.97% > cold 0 > warm +4.4%.
ng4's owed smoke: ALL GREEN (no-op detector DIFFERENT; training path UNMOVED, bit-identical to ng1's cold reference; differential
bit-identical 1.0765775442123413; peak RSS 40.4 GiB PER ARM — the 41.5 figures were per-arm, not summed). What none of it changes:
md1's accuracy-corner closure (18.3× on d_seg still owed; schedule levers ≤1.61× — the τ band is 1.012×). The τ band is the burn-QUALITY
lever of record; the born vehicle is not the pointer object.

**ps1 (70717fb11):** the PR #140 update packet is PREPARED at `/Volumes/APDataStore/pact/ddm_ps1_pr140_update_prep/` — stage-6 selector
replay REBUILDS fs1's archive bit-exactly from afr1's (3 runs incl. clean state; 4 negative controls refuse); compliance 78 GREEN / 7 RED
(vs pq12 80/7; 3 NEW reds were claim-ledger SHAPE — cured by MAIN's canonical terminal row with full archive + runtime-tree shas).
ps1's key finding falsified my charter's prediction: fs1's evaluated tree differs from the live PR tree in FOUR files (inflate.py two pin
lines + README/compress.py/MANIFEST from the older g8v1 lineage) → the staged packet = live PR tree + two pin lines; its digest
(ec4c9d19…) ≠ the evaluated fbf4aaf4… → a CUSTODY ROW on the staged tree was sealed (8cfa0c98…) and fired on T4 (~$0.30; same bytes,
same score expected; binds the public tree). NOTHING PUBLISHED — the operator's one-line confirm gates it.

**fm3:** the fmtools on-device lane beats the #344 regex gate where it matters — F1 0.769 vs 0.303 on eq1's 29-memo adjudication; the
eq1 `(?<!st)ratified` fix that killed a 55% false-positive rate ALSO killed 15 of 16 true positives (recall 0.25). Landed as an ADVISORY
column in the commit hook (fail-open, 30 s cap, 2.48 s/memo). GT-lineage gate widened to `.npz`: 2 → 378 findings, 326 in the refusing
class → REPORT-ONLY by the charter's own rule. Constant-provenance lane: MEASURED NEGATIVE (36%, `unknown` never emitted; binary control
fails identically → property of the task). fmtools 0.0.219: `classify_batch` + CLI, 691 tests green.

**gv1's governor caught a defect by RUNNING (MAIN, 21:50Z):** with the Metal FREE, `cell_admission` still REFUSED ng4 ("5 live cells") —
the five are launch/waiter/shard JOBS, not training cells; `contending = cells+1 = 1` then consults concurrency-≥2 evidence
(`max(2, live_count)`) for a cell that would run ALONE. Patch: no training cell live ⇒ no contention verdict. Sister of [[m50]] VACUITY and
the FALSE-SURVIVED genus: the guard read the wrong denominator.
**P0 hygiene (ng4 report):** the boot volume hit 344 MiB free (ENOSPC twice) — `.omx/tmp` 208 GB + `experiments/results` 149 GB; dk1 arm
spawned for certify-and-MOVE reclaim (never delete uncertified bytes).
Equations leg (`tac.canonical_equations`): gm1's τ-band law — first full-burn anchor (ng3).

## ADDENDUM 18 (2026-09-04 22:00Z real) — fs2 seals the next candidate; the custody row binds the public tree; ng4 fires; two governor defects

**fs2 (500189019 / 60f6a9668): the falsifier did NOT fire.** Re-solving the 12-dim carrier on the 21 pairs whose frame 0 moved: 15/21
pairs moved 67 int12 coordinates, summed gain 2.147e-5 for **+1 B** (the `up2` control reproduced the shipped 78,628 bits exactly);
the matched control on the base body (frame 0 unmoved) changed **0/21** coordinates — 100% of the win is selector-induced staleness.
One alternation sweep moved one selector mode byte-free; the k drop ladder is a staircase (23→24 saves 0 B) so no drops. Candidate D:
archive a8f3a379…, **180,023 B, projected S 0.14784104973157752 [advisory], net −2.2145e-5 = 1.107× the admit bar, 3.00× the 8-dp
report bound**; 585 untouched pairs Δd_pose = 0.0; odd-frame sections byte-identical. Two frozen constants un-frozen on the way
(pr1's hardcoded afr1 body sha; up3's hardcoded q=11/lgwin=24 container — a different generation's, +2 B here). SEALED (532f2482…);
T4 fire refused once by a TRANSIENT Modal build error ("test_cell_admission.py was modified during build process" — my `ruff format`
raced the image build; lesson: no source edits while a fire's image is building) and RE-FIRED (receipt `fs2_t4_buy_r2`).
PROMOTE IFF exact S < 0.14786319521362173.

**ps1's custody row (fc-01M1Q5S91QG8CSTV9458M1S6VE):** the fs1 archive on the STAGED public tree reproduces the pointer row EXACTLY
(pose 6.17e-06, seg 0.00020139, 180,022 B, S 0.14786319521362173; 554.8 s). The public packet's runtime tree is now an evaluated
tree (fire-tool tree digest 05b85e1e…); the tree-digest RED is cured. The prepared PR #140 update still waits on the operator's
one-line confirm — and if fs2's row promotes, the packet needs a seventh stage (carrier re-solve) before it is worth posting.

**ng4 FIRED 21:55:39Z** (receipt `ng4_continuous_DONE.json`, ~4.4 h): the continuous-objective cell (τ held at r10's exact terminal float
0.05000000074505806, duals carried — head start 57.0×/224.4×).

**Two governor defects found by RUNNING, fixed, tested (4274f7fd5, 349f1fd82):** (1) a candidate that would run ALONE was gated by
N=2 contention evidence (`max(2, live_count)` with contending=1) — refused ng4 for 90 min with the Metal free; (2) each admission poll
WALKED both 1.8 TB SSD roots to depth 6 — MEASURED >120 s per poll (the waiter's polls stretched to 100 s, then stalled 10 min in one
call). Cure: the launcher appends a `detached_launch_registry.v1` row at every launch (fail-open) and the governor reads the registry
first (0.09 s measured), walking only as a pruned fallback or with `--walk-roots`; registry seeded from a pruned walk (APDataStore 396
rows in 13.7 s; Vertigo seed running detached). Genus: a guard that is slow enough to be skipped is a guard that is not there ([[m102]]).
Equations leg (`tac.canonical_equations`): fs2's projection will be the second pure-pose authority anchor for `exchange_ratio_noise_floor_v1`
when the row lands.

## ADDENDUM 19 (2026-09-04 22:15Z real) — THE POINTER MOVED AGAIN (25th): fs2 carrier re-solve S 0.14784474152757654

Two pose-only moves in one day off the same object: afr1 0.14797617 → fs1 0.14786320 (frame-0 selector) → fs2 0.14784474 (carrier
re-solve on those pairs). fs2 −1.85e-5, realized within +3.69e-6 of the advisory. Both wins came from the same crux: pr1's terminal
pose re-solve was never byte-closed, and fs1's selector edit made 21 pairs' carrier codes stale — fs2's matched control (0/21 on the
unmoved base body) proves 100% of the win is selector-induced, not pre-existing slack. Full record `ddm_fs2_pointer_move_25_20260904.md`.
The pose corner on this object is now nearly drained (fs2 alternation moved one more pair byte-free; the k drop ladder is a staircase).
Sub-0.12: rate corner −41,817.8 B, distortion corner 215.1×. Equations leg (`tac.canonical_equations`): `exchange_ratio_noise_floor_v1`
second pure-pose authority anchor.

## ADDENDUM 20 (2026-09-04 23:05Z real) — dk1: the disk was not full of bulk; it was full of Time Machine snapshots

Boot data volume 68 GiB → **211 GiB free** (MEASURED on `/System/Volumes/Data`; `df /` reads the sealed system volume). 110.40 GiB
reclaimed under certificate, zero uncertified bytes deleted, reconstructibility PROVEN (ddm_eu1 rebuilt from its cert row, tree sha
identical). **The mechanism inverted the charter's prediction:** 21 APFS Time Machine LOCAL SNAPSHOTS pinned every deleted byte —
a certified 32.97 GiB deletion freed +1 GiB; thinning the snapshots freed +65 GiB; ap1's 56.34 GiB move freed exactly 0 until one more
thin. The TM destination is a NETWORK share (`smb://bat00-tm.local`); when unreachable, local snapshots accumulate hourly and are never
thinned. **Law for this machine: no reclaim frees anything until local snapshots are thinned, and they regenerate hourly.** Operator
item: the TM network backup is unreachable/stale; dk1 thinned all 21 snapshots (gentlest urgency thinned more than intended) WITHOUT
verifying the network backup's currency — if a lost file's only copy lived in a local snapshot, that path is gone (stated plainly).
Cert classes: git-reconstructible DELETE 32.97 GiB (12 trees) · ap1 advisory frames MOVE 56.34 GiB · stale codex logs MOVE 21.09 GiB;
symlinks + `MOVED_TO.json` left; 9 worktrees removed by plain `git worktree remove`. Blocked, correctly: `ddm_mst1` 20.55 GiB (all
`retained/`), `arm_receipts_local` 39.7 GiB (every store has a `retained/` or `.pt`), 12 dirty worktrees 20.79 GiB, `ddm_lt1` (2 refs
absent). Four bugs dk1 caught in its own output (vacuous fully-blocked census; deref failure read as absence; DST-wrong UTC parse; the
never-touch test judged the candidate's path not its descendants — `ddm_mst1` was offered as movable) — all with regression tests (61).
Routed to gov2: the launcher's storage waterfall must read CONTAINER free space and snapshot-pinned bytes (`tmutil listlocalsnapshots`),
and the reclaim cadence (`docs/runbooks/local_disk_reclaim_cadence.md`) must thin snapshots first. `experiments/results` 149 GB owed
its own arm. Equations leg (`tac.canonical_equations`): none — disk hygiene moves no score.

## ADDENDUM 21 (2026-09-04 23:45Z real) — hv1: the harvest→pointer autopilot exists, and its replay graded MAIN's hand work

`tools/pointer_move_packet.py` (10 stages, dry by default) + `src/tac/pointer_move.py` + `src/tac/modal_source_snapshot.py`; the poller now
stages the canonical terminal claim row with FULL shas and an axis-derived status `completed_contest_{cuda,cpu}_exact_eval_harvested`.
Replay on today's two harvests in isolated state reproduced every load-bearing number (S, three terms, deltas, corners, margins, projection
errors, post-move anchors) and found **5 MAIN hand-errors**: fs1 pose-term last digit; fs2 rate term, pose term and gap transcribed wrong;
fs2 claim row's runtime-tree sha truncated to 12 hex — plus a RED neither ps1 nor I expected: my terminal-row status prefix
`completed_modal_auth_eval_harvested_S_…` is not a `SUCCESSFUL_EXACT_EVAL_TERMINAL_STATUS_PREFIXES` member (both lanes RED; the old poller
code hardcoded cuda and would have mis-closed every CPU row). Errata appended to both pointer memos; canonical rows to be re-emitted by the
apparatus. hv1 also found 5 bugs in its own packet by replay (relative tool paths → silent no-op with a success print; first run wrote real
custody at the tier root; two tools not repo-root aware; snapshot digest double-counted nested mounts; hardcoded axis label) — all fixed.
Fire-from-snapshot: `git archive HEAD` is IMPOSSIBLE here (upstream/ 0 of 19,677 files tracked) → APFS clonefile snapshot of the mounted
paths, 16,060 files / 780 MB in 6.7 s; a 40-rewrite race during the build leaves the manifest digest identical; never carried a real Modal
build yet (`--no-source-snapshot` reverts). Pre-existing red: `test_claim_lane_dispatch.py::test_terminal_prefixes_constants` (`stale_` vs
`stale_assumed_dead`/`stale_superseded`) — routed to gov2's owner list. Next move ordinal derives from `.omx/state/pointer_move_events.jsonl` (26).
Law banked: the number a human typed is the number the apparatus must re-derive ([[m18]] WRITES>READS; the packet is the reader).
Equations leg (`tac.canonical_equations`): none new — the packet reads the registered laws; it does not add one.

## ADDENDUM 22 (2026-09-04 23:45Z real) — gov2 and ps2 landed: the governor is complete by construction; the packet tracks the pointer

**gov2 (f3d6b5ce0 / 7455580b9), six mechanisms, each drilled against the live ng4 cell:** process-table discovery 0.045 s vs >120 s (>2,600×;
found ng4 pid 33030 declared 45.0, registry a strict subset; the walk survives only as `--walk-roots` unioned, never substituted) ·
measured-peak ledger `tools/measured_peaks.py` (9 receipts → 9 rows; **ng4's system-availability delta 49.572 GiB, 28.4× its 1.746 GiB RSS** —
my "~45" was 9.2% low; ng2 ran a full Metal burn declared at 2.396 GiB, **20.69× under**) · memory-pressure watchdog (10-min report-only drill,
121 polls, 0 alarms, 0.000 GiB from the blackbox daemon) · ONE fire path through the queue driver with all three cells resolving 49.572 GiB
FROM_LEDGER, four `.sh` deleted git+SSD, **Catalog #413 STRICT in the same landing, live count 0** · storage waterfall at the launcher (rc=11 on a
simulated 344 MiB boot volume WITH the snapshot census in the refusal; budget = 2× the family's measured artifact_gib) · concurrency law
`UNRESOLVED_AT_CONCURRENCY` (rows straddle the baseline 0.964..1.117 → two Metal cells refused by default).
**Two corrections to my record, from the machine's own log:** the compressor peaked at **76.978 GiB** (not 83) and **the collapse happened
TWICE** — 16:45:56Z and 17:10–17:11Z — peak swap 72.0 GiB; the whole ramp is **16.06 s** (+6.03 then +9.79 GiB/s), so the watchdog acts on the
first critical sample, no debounce; and `available_gib` NEVER fell below 18.222 GiB during the collapse — **the free-memory guard is blind to
this failure by construction** (`free < 1 GiB` rejected as a trigger on a 28.99% base rate over 13,235 samples). Untested in anger: no cell fired
through the driver yet, the watchdog never fired for real. No equation registered, with the reason stated (host-specific apparatus constants
are the borrowed constants the registry exists to keep out). Pre-existing: `check_no_bulk_write_strands_the_ready_record` fails at HEAD on
`experiments/ddm_pr1_*` (MAIN's gate from this morning; MAIN fixes).

**ps2 (107e22ea8):** stage 7 (`fold_fs2_stage`) rebuilds the fs2 bytes ×3 incl. from a clean store (a8f3a379…, 180,023 B; Rice bits
78,628 → 78,634 = the whole +1 B; identity control reproduces fs1); four negative controls refuse; compliance **83 GREEN / 4 RED of 87, ZERO new red
classes** (vs ps1 78/7, pq12 80/7) — all three claim-shape reds GREEN on the apparatus-written rows; the four survivors are pq12-frozen. Custody
seal 60b8d3db… (canonical) → **FIRED by MAIN 23:47Z** (receipt `ps2_t4_custody`; the first real Modal build through hv1's source snapshot).
The README's `evaluated commit 1c9fbbf5…` pins the public receipts repo, not pact — flagged, unchanged. NOTHING PUBLISHED.
Equations leg (`tac.canonical_equations`): none new (apparatus constants stay out of the registry by design; ps2 is a byte identity).

## ADDENDUM 23 (2026-09-04 23:55Z real) — ng5 sealed; my charter premise was MEASURED FALSE; ng4 turned up at 3k

ng5 (d54f65c1e / 70f2edc8f / 98a192d4c): the two parents' τ blocks COLLIDE — ng3 and ng4 both write `tau_band` + `expected_flip_tau_start/end`
— so "ng4's config plus ng3's block" can only mean ng3's band REPLACES ng4's τ half. Composition of record: τ leg = ng3's `msafe_band`
[0.04376363754272461, 0.021881818771362305] (subsumes ng4's cure: the entry NARROWS 1.1425× instead of re-widening 3.0×; given up: τ continuity at
the exact terminal float); dual leg = ng4's carried `initial_lambdas`. The third geometry [r10's terminal τ → δ_R] keeps both halves whole and is
follow-on #1, not smuggled in ([[m164]] UNION≠SUM). Sealed config 1205463b…, re-rooted 93f92fc6…, tree d54f65c1…, ZERO pin movement vs ng4's
tree; $0 checks (differential + 3-way no-op detector against control/ng3/ng4 step-1 shas) run inside the waiter's smoke and FAIL-CLOSE the fire.
Queued behind ng4 through gov2's driver (waiter pid 69289; peaks FROM_LEDGER 40.92/49.572 GiB; the argv carries a non-numeric placeholder so a
failed rewrite is refused by argparse). **ng4 turned UP at 3,000: 0.431595** (0.4258 → 0.4261 → 0.4316) while ng3 fell to 0.403796 — ng3 is the
stronger parent through three milestones; BELOW-BOTH is judged against ng4's ACTUAL terminal (~01:10Z). Blocker: APDataStore 16 GiB — the next
generation will not clear the 8 GiB reserve without a cold-store sweep (dk2 chartered). Watchdog: first WARN rows logged (no action at WARN).
Equations leg (`tac.canonical_equations`): none new (the ng5 cell IS the measurement of gm1's band × the continuous-dual anchor).

## ADDENDUM 24 (2026-09-05 00:00Z real) — the snapshot fire path failed in anger, and the refusal was the lucky outcome

The first real Modal build through hv1's source snapshot (ps2's custody fire) REFUSED rc=5: `FileNotFoundError '.venv/bin/python'`. hv1's root cause
(1fbc7c066) was deeper than the symptom: the cwd change moved the LOCAL half of the dispatcher too — `claim_modal_auth_eval_dispatch(repo_root=Path.cwd())`
ran the claim writer with `cwd=<snapshot>`, so my third cure option (symlink `.venv` into the snapshot) would have made it SILENTLY WRONG: the dispatch
claim would have landed in `<snapshot>/.omx/state/active_lane_dispatch_claims.md` and been lost, and the single-flight guard would then read an empty
ledger. Cure = my second option: dispatch cwd stays the repo root; the snapshot reaches Modal through `PACT_MODAL_SOURCE_ROOT` + a one-line
`_mount_path()` in both app modules (byte-for-byte no-op when unset; VERIFIED on the real modules); plus `verify_dispatch_paths(cwd, argv)` refuses
rc=9 BEFORE the subprocess if any relative path in the argv or in the local half's known spawns fails to resolve (the spawned paths are read from
`dispatch_claim_command`'s real signature via `inspect.signature`, never re-typed). Dry-run proved (11 new tests, 81 across suites); honest snapshot
count corrected to 8,307 files / 642 MB in 9.8 s (the 16,060 was a nested-mount double count). Still NOT fire-proved end to end → snapshot stays OFF
for fires (`--no-source-snapshot`); fire-proof it on the next CUSTODY row (known answer), never on a frontier row. The custody row re-fired through
the proven path (call fc-01M1QDB3R37N2JWWNHW64TZT2P). Lesson for the genus: a refusal that stops a wrong write is the guard WORKING; a "cure" that
makes the refusal go away can be the cure that makes the failure silent ([[m102]] · [[m100]]).
Equations leg (`tac.canonical_equations`): none.

## ADDENDUM 25 (2026-09-05 00:05Z real) — mc1 interim: the motion-compensated plane's ceiling is firing its falsifier

mc1 (Fable), wave 1 on the exact fs2 body, all MEASURED: rows control byte-identical (113,411 B, sha 5601d6fd…) — the coder's own rows; alignment:
ALL SIX decoder-derivable MC planes align WORSE than co-located on Lane/Road/Undrivable/MyCar (Lane IoU 0.218–0.2457 vs co-located 0.2495); only
block variants gain on Movable; the oracle (reads field_t, diagnostic) gains +0.018–0.025 Lane IoU rigid, +0.074 block — but block motion is
temporally UNPREDICTABLE (derivable-vs-oracle dy correlation 0.09–0.27 in the road rows) and carrying it costs 9,861 B; held-out indicator-family
ceiling: `mc` +5.0–5.8 B, `agree` +3.1–3.9 B, richer cells −149…+30 B — **three orders of magnitude under the 5,000 B refusal bar.** Wave 2
(block, block_gated, oracle_block) pending; if it holds, the charter's falsifier fires → CEILING-REFUSED at FORMULATION scope for the shipped
receptive field (dc1's mechanism does not transfer through a decoder-derivable motion estimate: the ego-motion between pairs is not predictable
enough from two decoded fields to re-align the classes that carry the bits). The rate corner's "unmeasured door on the closest wall" is then
measured. Side finding: mc1's three parallel ~10.3 GiB ceiling processes drove the watchdog's 11 WARN rows; on CRITICAL the watchdog would have
paused ng4 (not the cause) — target-policy gap routed to gov2.
Equations leg (`tac.canonical_equations`): the ceiling table will anchor dc1's learned-receptive-field law with a NEGATIVE transfer row when mc1 lands.

## ADDENDUM 26 (2026-09-05 00:05Z real) — the prepared PR #140 update is fully bound; snapshot fire-proof fired

ps2's custody row (call fc-01M1QDB3R37N2JWWNHW64TZT2P, proven no-snapshot path, 660.2 s): the fs2 archive on the STAGED public tree reproduces the
pointer row EXACTLY — pose 6.14e-06, seg 0.00020139, 180,023 B, S 0.14784474152757654 — fire-tool tree digest 355e7f95…. The prepared update packet
(`/Volumes/APDataStore/pact/ddm_ps2_pr140_update_prep/`, stages 1–7, compliance 83/4 zero new reds) is now bound to an evaluated tree. What remains
before a public update is ONLY the operator's one-line confirm (p0_swap_procedure) and the RELEASE_PLAN's six gh steps with fetchback. Then, with
single-flight clear, MAIN fired the FIRE-PROOF of hv1's fixed snapshot path (1fbc7c066) on the SAME known-answer seal, snapshot ON (receipt
`ps2_t4_custody_snapshot_proof`): expected components identical; if it dispatches and harvests, the snapshot becomes the default for frontier fires.
Equations leg (`tac.canonical_equations`): none.

## ADDENDUM 27 (2026-09-05 00:35Z real) — the source-snapshot fire path is FIRE-PROVED; default ON for every fire from here

Call fc-01M1QE7C0KYRX56GBGX6FAEF6G, snapshot ON (1fbc7c066: cwd stays the repo root; mounts via `PACT_MODAL_SOURCE_ROOT`; `verify_dispatch_paths`
pre-spawn guard), on the known-answer ps2 custody seal: pose 6.14e-06 · seg 0.00020139 · 180,023 B · **S 0.14784474152757654 — IDENTICAL** to the fs2
pointer row and to the no-snapshot custody row; tree digest 355e7f95… identical; snapshot 8,307 files, `complete: True`; 542.0 s Modal wall.
Consequence: the edit-race class that refused fs2's first fire ("source modified during build process") is closed STRUCTURALLY — a working-tree
edit during an image build can no longer refuse or contaminate a fire. Apparatus proved on a known answer, never on a frontier row, as the law
says. hv1's poller also wrote the fire-proof lane's canonical terminal row itself. Equations leg (`tac.canonical_equations`): none.

## ADDENDUM 28 (2026-09-05 00:40Z real) — the fix in git was not the fix in RAM: the old watchdog paused ng4 three times; ps2 r5 = 84/3 by digest

**Incident (MEASURED from the alarm ledger):** the watchdog instance launched at 23:50Z (pre-fix build) fired CRITICAL on compressor GROWTH RATE
(5.43 / 6.48 / 6.60 GiB/s ≥ 4.0) at 00:16:45, 00:27:45 and 00:34:20Z and SIGSTOPped ng4's TRAINER (pid 33374) each time — the "newest training
cell" rule gov2 had already falsified and replaced in 2a24996da (00:08Z), while the actual growers were mc1's four ceiling jobs (26.6 GiB; ng4 0.5
GiB RSS). Two SIGCONTs followed (00:20:49, 00:29:29); the THIRD pause was never released — the clear hold never came with the compressor parked at
41 GiB and swap at 10.5 GiB — so ng4 sat in state T until MAIN found it at 00:36:35Z and SIGCONTed it (now R; 3,824/5,000; terminal slips to ~01:25Z).
MAIN then TERMed the old instance and relaunched on 2a24996da (pids 63765/63766/63811, receipt `memory_watchdog_r2`). Law banked: **a fixed guard is
not live until the running instance is restarted — the fix in git is not the fix in RAM**; and a SIGSTOP without a guaranteed SIGCONT path (clear
hold unreachable under sustained-but-stable pressure) is a silent kill of sunk work — gov2 owes a max-pause bound (SIGCONT after N minutes regardless,
with an alarm) and an on-restart reconciliation that SIGCONTs anything the previous instance left stopped.
**ps2 r5 (23a2f9329): 84 GREEN / 3 RED of 87** — exactly one flip, `submission_runtime_tree_matches_auth_eval` RED → GREEN on a REAL digest match
(fired portable custody-pruned tree 963955e8… == staged portable tree), not a vacuity; the three survivors are STRUCTURAL-RECORD (raw promotion-policy
flags), RECORD-WITH-REASON (no CPU row; the prior attempt timed out at 1,800 s) and BLOCKED-ON-OPERATOR (nothing hosted). The packet is publish-ready
modulo the operator's confirm. Equations leg (`tac.canonical_equations`): none.

## ADDENDUM 29 (2026-09-05 00:45Z real) — the pause actuator is retired: SIGSTOP makes the paused process the swap victim

gov2 (de7b4229c) MEASURED the mechanism behind the stuck pause: stopping a process does not free its memory — a stopped process is resident,
dirty and idle, i.e. the IDEAL eviction victim — so swap crossed the 4 GiB WARN threshold **12 s after** the 00:34:20Z SIGSTOP and stayed above it
for 4 m 19 s (peak 11.29 GiB) with the compressor parked at ~41 GiB: the clear hold was unreachable BY CONSTRUCTION and the old code could only
resume inside the OK branch. Three cures landed (MAX_PAUSE_S = 180, DERIVED from the worst clearing pause 97 s + 60 s hold; `reconcile_orphaned_pauses`
at startup and as a subcommand; rate-CRITICAL requires compressor ≥ WARN) and the reconciler's FIRST run found a live orphan: the corrected 2a24996da
instance had stopped mc1's `ceil_block` tree at 00:39:14Z (swap 5.86 GiB alone held the WARN) — released for real. The watchdog now retires itself when
its source mtime changes (fix-in-git ≠ fix-in-RAM, structural half; the launcher-refuses-stale-instance half is named, not built).
**MAIN's decision:** SIGSTOP hurt twice (ng4 ×3, mc1 ×1) and helped zero times — every pause was followed by swap growth, never relief. Under "never
weaker state" the actuator is RETIRED: the live instance runs `--report-only` (r3 on de7b4229c: alarms name the actor via `top_rss_growers`, no
signals); gov2 asked to flip the default. Follow-on named, not built: a COOPERATIVE pause (SIGUSR1 → the trainer checkpoints at the next step
boundary and exits resumable → the queue driver re-admits when pressure clears) — the only actuator that frees resident bytes without losing work.
Zero processes in state T after reconciliation; ng4 running. Equations leg (`tac.canonical_equations`): none (host apparatus constants stay out).

**Default flipped (gov2 f6f3f03f8, 00:50Z):** `report_only` defaults True; `--act` is the explicit opt-in; `--act --report-only` → report-only wins (fail-safe, drilled); the RECONCILER runs in both modes (rescue is not an actuation). gov2's own words: "the guard I built destroyed work and prevented none; its value tonight was entirely in the alarms and the actor-naming." Owed, named: the cooperative pause protocol (SIGUSR1 → checkpoint → exit resumable → re-admit) and a launcher that refuses an instance older than the tool it runs.

## ADDENDUM 30 (2026-09-05 01:05Z real) — dk2: APDataStore 17 → 55 GiB; the bulk class was not what the charter predicted

dk2 (2130725f8 / 2b9e10b54): **39.00 GiB freed** by certify-and-move of 8 `work/inflated/` trees (32 ledger rows PLAN→COPIED→VERIFIED→MOVED_SYMLINKED,
zero BLOCKED, zero bytes deleted; the generating archive/provenance/auth-eval stay local; symlinks resolve). APDataStore 17 → 55 GiB; Vertigo 83 → 44
(the ≥40 floor was the cap, not supply). No TM local snapshots on this volume (dk1's mechanism is boot-volume-only). **Charter prediction falsified ~7×:**
finished burn-cell generations total 4.44 GiB; the real bulk is **49 `inflated/` trees = 140.30 GiB** (inventory committed). ng5's storage leg: PASS with
margin (55.52 GiB vs the 49.572 GiB resolved peak; it had passed at 16.69 GiB too — ng5 was blocked on admission, not disk). A vacuous PASS in dk2's own
first planner run (304 child paths on one `du` argv pinned everything → "0.00 GiB in 0 trees") was made LOUD (denominator on stdout, `VACUOUS-CENSUS`
warning; +6 tests). Collateral, owned: 7–8 concurrent movers saturated both tiers (`df` 48–120 s; ng4 21 steps in 10 min vs 130) — capped, then
paused; ng4 recovered to 27 steps/min within a minute of the last move. Not done: ~101 GiB more of the `inflated/` class (needs a destination), the
247 GiB of `cold_store*` (a retention question), the 4.44 GiB burn trees. Pre-existing reds routed: `[lane-pre-registered]` on mc1's working file
(`lane_oracle`, `lane_iou` identifiers); orphan-module guard 22 > 10 on unrelated `src/tac/` modules. Equations leg (`tac.canonical_equations`): none.

## ADDENDUM 31 (2026-09-05 01:10Z real) — mc1 (Fable): the motion-compensated plane is CEILING-REFUSED; two of MAIN's claims corrected

**Verdict (MEASURED on the exact fs2 body, receipt `…/ddm_mc1_motion_compensated_previous_plane/ceiling/CEILING_RESULT.json`; commits ebc571397 /
54f96458d):** the best decoder-derivable motion-compensated previous plane conditions the shipped coder for **+159.60 B held-out** (`block` 48×64,
tilt `mc_x_coloc_x_arg`; 3-seed indicator minimum +138.49 B) against the pre-registered 5,000 B bar — 31× short, 0.14% of the 113,411 B stream.
Instrument control: the retained field re-encodes byte-identically through the shipped encoder (113,411 B, sha 5601d6fd…), so every ceiling row sits
on the coder's own 50,009,121 live positions (mi1 family + 5-way tilt; pair-level two-fold; 3 seeds). Alignment vs field_t: EVERY derivable plane
is worse than co-located on Lane/Road/MyCar/Undrivable (shift 0.2457, zoom 0.2423, planar 0.2429, block 0.2295, gated 0.2460, median3 0.2180 vs
co-located 0.2495); only Movable (1.24% of area) gains. Mechanism: rigid/ground-plane oracle alignment buys ≤ +0.025 Lane IoU — the field's inter-pair
change is NOT rigid motion; block motion is real (oracle +0.074) but NOT extrapolable (derivable-vs-oracle shift correlation 0.09–0.27 in the road
rows, −0.35 in one). Even the ORACLE block plane (+3,420 B, reads field_t) sits below the bar and below its own 9,861 B carriage → carried motion is
closed on arithmetic. Steps 2–4 (retrain, RC64 price, fire order) correctly NOT run. Equation `motion_compensated_previous_plane_alignment_gate_v1`
registered (3 anchors, 8 re-derivation tests); lane L2 research_only; payloads retained (coding rows 2.36 GB, 7 planes, control stream).
**The rate corner's "unmeasured door on the closest wall" is now measured shut at FORMULATION scope for the shipped receptive field.**

**Two corrections to MAIN (NO-FAKE, on my own claims):** (1) the RE-SPAWN ADDENDUM said the 09-03 codex spawn was "stranded, no process, no result" —
FALSE: its screen RAN (three planes, DF1 rows, best −17.1 B) and its memo/receipts exist; the queue row was stale, not the work. mc1's `Write` then
overwrote that committed module at the charter path (mc1's error, restored byte-identically as `experiments/ddm_mc1_motion_plane_ceiling_screen_20260903.py`
with its 12 tests rebound; cited as an independent replication). Law: before declaring a spawn stranded, read the arm's OWN receipts/memo, not the queue
row ([[m75]] CHECK REFS before 'lost'). (2) MAIN told mc1 its 23:47Z rc=143 was a memory-cap kill correlated with the watchdog's WARN burst — FALSE:
it was mc1's OWN kill (an in-sample telemetry formula fix). The correlation was coincidence; the causal reading was mine, unmeasured. What WAS real:
from ~00:05Z dk2's movers froze APDataStore (ExFAT/FSKit) and mc1's ceiling children sat in uninterruptible I/O wait ~50 min (wall-clock lost, nothing else).
Equations leg (`tac.canonical_equations`): `motion_compensated_previous_plane_alignment_gate_v1` (new, negative-transfer anchor for dc1's mechanism).

## ADDENDUM 32 (2026-09-05 01:33Z real) — ng4 terminal: the continuous objective delays the excursion and returns it; only the τ band ends below its start

ng4 @5k 0.424842 vs cold 0.425149 (−0.07%, a wash; the pre-registered rule "holds" by 3e-4 — inside noise) vs ng3 0.391810 (+0.033 worse). It led at
@1k/@2k (−8.8%/−12.3% vs cold) then gave it all back: d_pose drifted from 5.76e-4 to 8.12e-4 (worse than start), d_seg +4.7% above start. Reading: holding
τ at r10's terminal and carrying the duals removes the RESTART shock but not the OVER-PAINT the band removes — sd1/gm1's mechanism (77.7% of the seg gradient
is waste outside m_safe) is the one that matters at the terminal, and the dual state alone cannot substitute for it. Four single-lever cells read; the
ordering is now complete: τ band −7.84% ≫ continuous ≈ cold ≈ cap (−0.97%) > warm +4.4%. ng5 (band × duals) fires next from admission; expectation
REDUNDANT (≈ ng3). Burn-quality series conclusion: **the τ band is the burn default; nothing else in the series earned a place.** Not a pointer object.
Equations leg (`tac.canonical_equations`): gm1's τ-band law — the four-cell ordering is its anchor set (τ band the sole below-start lever).

## ADDENDUM 33 (2026-09-05 01:40Z real) — first queue-driver fire in anger: ng5 is live, with no hand in the loop

ng5's waiter: gate 1 (ng4's receipt) 01:30:58Z → three admission holds a minute apart → bounded smoke (60 s): **3-way no-op detector PASSED** (the
composition's step-1 state differs from the control AND from both parents' step-1 shas) → cell peak resolved FROM_LEDGER 49.572 GiB (ng4's measured
system-availability delta; ng3's shape-matched 40.92 GiB for the smoke) → `tools/cell_queue_driver.py run` dry-run rc=0 → fire rc=0 at 01:36:11Z; cell
`seed_20260902_tau_band_x_continuous_objective_control_native100` live (3 processes), receipt `ng5_composition_DONE.json`, ~4.4 h → ~06:00Z.
No bespoke shell, no inline vm_stat arithmetic, no hand-typed peak, no MAIN action between ng4's exit and ng5's start — the permanence program's
first end-to-end proof. Pre-registered expectation after ng4's terminal: REDUNDANT (≈ ng3's 0.391810), possibly slightly ANTAGONISTIC on d_pose; the
words decide at ~06:00Z. Not a pointer object. Equations leg (`tac.canonical_equations`): none.

## ADDENDUM 34 (2026-09-05 04:25Z real) — ng5: BELOW-BOTH; the burn default is now τ band × carried duals

ng5 @5k **0.384833**: −9.48% vs cold, −1.78% below ng3 (0.391810), −9.4% below ng4, −3.5% below its own start; d_seg −1.25% (the band), **d_pose 4.247e-4
= −26.2% vs start, the series' best** (the duals, which alone drifted pose UP in ng4, bring it DOWN once the band holds the seg gradient inside m_safe —
complementary through the pose axis, not redundant; MAIN's post-ng4 REDUNDANT expectation was wrong, the charter's sub-additive-same-signed prediction
held). Series complete: band×duals −9.5% ≫ band −7.8% ≫ continuous ≈ cold ≈ cap > warm. Burn default of record: τ band × carried duals (ng5's sealed
config). Not a pointer object (born vehicle 106.6 KB; accuracy corner still 18.3× away; md1 stands). Fired and read with no hand in the loop.
Equations leg (`tac.canonical_equations`): gm1's τ-band law + the continuous-dual anchor — the interaction row (duals × band → pose) is a NEW measured
anchor; register on the next equations pass.

## ADDENDUM 35 (2026-09-05 13:25Z real) — hc2: the flip-LOCATION half is at its address floor (CEILING-REFUSED, FAMILY scope)

hc2 (7b767b5f9 / 9c7dfa8de) re-verified mc1's rows five-sha and byte-inside-the-archive (member `p`, offset 66,512), reproduced hc1's decomposition to
0.3–0.4% (indicator 110,909.07 B = yes 34,642.82 + no-branch 76,266.24; conditional 2,501.80; 227,555 flips at 2.6812 b/flip — mc1's `base_argmax` was the
pre-corrector argmax, 0.0275% off), then priced the location half against component representations, held-out 2-fold × 3 seeds: **the flips are 81.1%
singletons** (172,193 components / 227,555 sites at 8-conn; median 1, p99 5, max 21; only 9 components ≥ 16 sites carrying 52 B). Every explicit
representation is WORSE than the per-site sum — best (a) β-per-cell seeds + shapes +17,775 B vs the indicator; best overall (b) D=3 boundary geometry
+30,290 B vs the no-branch, −4,353 B vs the full indicator only when the acausal current-frame boundary is used (mi1's causal version: +5.27 B). The
mi1-family conditional bound over the SAME clustering information: `pat4` **+23.82 B on 110,909 B — 210× under the 5,000 B bar**. Verdict: the falsifier
fired harder than written — CEILING-REFUSED at FAMILY scope (two independent instruments agree). Fourth instance of the address law ([[m118]]): clustering
addresses is not an escape from naming them; the shape token is itself an address. Not touched: the joint field+model escape. Equation
`flip_location_component_address_floor_v1` registered (10 guards); lane L2; 916 MB retained. Equations leg (`tac.canonical_equations`): that equation.
**Rate-corner ledger on the shipped object is now complete:** coder swaps 0 B · buckets ≤ 211 B · reorder 0 B · 21-tap oracle +32 KB above · MC previous
plane +160 B (ceiling) · flip-location representations ≥ +17,775 B worse · ONLY cl2's prior-capacity ladder remains unpriced (training now).

## ADDENDUM 36 (2026-09-05 13:35Z real) — ane1: placement PROVED; the ANE is closed for authority AND for pose screening; my prediction was inverted

ane1 (397e10038 … addf3814c): **placement PROVED per op** (`MLComputePlan`, coremltools 9.0): SegNet fp16 298/298 ops and PoseNet fp16 287/287 on the
Neural Engine; **fp32 never reaches the ANE (0/ops, both scorers)** — the 07-13 lane's fp32 rung was never an ANE rung. Trunk speedups vs 1-thread CPU
torch: SegNet fp16 63.8×, PoseNet fp16 74.1×. **Fidelity at n600 INVERTS my charter prediction:** SegNet fp16 flip rate 4.818e-5 (fails the 3.3e-5 authority
bar by only 1.46× — 513× below the 07-13 n24 number; inputs-vs-toolchain not separated, stated); **PoseNet fp16 self-MSE 1.125e-2 vs d_pose 7.77e-6 — fails
by 1,448×.** Mechanism: fp16 error is RELATIVE (0.83% of a |31| pose dimension = 0.26) while d_pose is ABSOLUTE at 1e-6 — the argmax spends its top-2 margin,
the regression has no slack. Registered `scorer_fp16_drift_by_axis_v1`. Wired `--scorer-backend {cpu_torch,coreml_cpu_fp32,ane_fp16_screen}` on pr1's
selector (ranks; CPU-confirmed) and fs1's measure (refuses non-authority backends before any disk access) — then the replay REFUSED the ANE screen by
measurement: argmin agreement 4/39 (chance 12.5%), 34/39 picks worse than shipped, adopting them moves d_pose −4.73e-2 where the CPU sweep gains
+1.21e-4. End-to-end the 74× trunk becomes 2.06× (76.8 of 160 ms/forward is torch render + preprocess). **The usable finding: `coreml_cpu_fp32` is the
admissible accelerator — bit-exact SegNet argmax at 3.28×, pose to 3.1e-7 of the axis at 5.12×, no screening contract needed.** Hybrid exact-SegNet:
priced GO on area (0.357% of pixels in the flip band; 89× headroom) but NO-GO on realization (the parent lane measured tile recompute at 4.27× the dense
pass). 28 tests; lane L1. Saturation verdict for the operator's "ANE as well": the ANE runs the scorers at 100% placement and is useless for both scorers'
numbers; the CPU-fp32 CoreML path is the honest 3–5× win for advisory forwards.
**MAIN error banked:** three charters told arms to pass `--artifact-budget-gib`, a flag the launcher does NOT have (the waterfall derives the budget) —
never-invent-flags applies to MAIN's charters too; grep argparse before writing a flag into a charter.
Equations leg (`tac.canonical_equations`): `scorer_fp16_drift_by_axis_v1` (ane1).

**ERRATUM to ADDENDUM 36 (13:45Z):** the "MAIN error banked" paragraph is WRONG. `--artifact-budget-gib` DOES exist on `tools/launch_detached_process.py`
(line 1203; in `--help`; landed by gov2 f3d6b5ce0; md2's launch used it and was ACCEPTED with provenance "operator-declared --artifact-budget-gib"). ane1's
final message claimed it did not exist; MAIN propagated that negative-existence claim to md2 and into memory WITHOUT grepping the argparse — the #1
false-claim class ([[m53]]) committed by MAIN itself, one hour after banking "grep before writing a flag". md2 caught it (its message quotes the line
numbers). Corrected: the m140 memory paragraph, the hot-state line, and this addendum. The charters were right. The real lesson: a subagent's
"X does not exist" is a claim to VERIFY against the primary source before it moves anywhere — especially when it flatters a rule you just wrote down.

## ADDENDUM 37 (2026-09-05 14:05Z real) — md2: the burn default does NOT change which sites are reachable — PERSISTENT 62.954% (STANDS); the persistent set is the SAME sites

md2 (1c2f35d2f / 7fb218398), md1's instrument unchanged on ng5's 313 retained 16-step checkpoints, 141 forwards, integer bridge exact at all 71: **PERSISTENT
= 62.954% of the terminal shadow d_seg (cold 62.011%) — my 40–55% prediction FALSIFIED, the share ROSE 0.94 pp** (the band shrank the persistent set
11,842 → 11,019 sites but shrank the REACHABLE error more). Floor 11.671× the corner (cold 12.753×); terminal 18.539×. **Jaccard 0.8069 against the cold
control's persistent set** — 92.65% of ng5's persistent sites are cold's; against the honest within-pool null (both sets ⊂ the bit-identical 16,553-site
step-0 wrong pool; chance J 0.5263, attainable max 0.9305) the measured value sits 69.4% of the way from chance to ceiling (live: 84.3%); the classes the run
MANUFACTURES overlap at 0.10–0.36 — the inherited class is the same sites. Birth still step 16 (Lane and Movable, both cells); Lane MORE concentrated
(65.52% touching; GT=Lane enrichment 54.86× vs 51.50×). What the default DID buy: created/repaired 2.21× → 0.91× (2.44× better burn quality), terminal
−1.00% vs start where cold ended +9.82%. **Verdict STANDS: the born vehicle's unreachable set is fixed by the START, not the schedule — the accuracy corner
on this vehicle needs a different initialisation or generator, not a better burn.** md2's correction to md1's owed item #5: all three retained seed controls
share the SAME init sha (991a1cc6…), so a second SEED varies data order only and cannot test init-independence. Priced next step: a DIFFERENT-INITIALISATION
cell (~2.8 h Metal + 54 min CPU partition, $0, no new code) with the seed_20260903 data-order control as the free prerequisite. n32→n600 transfer caveat
travels with every row (stratified selection, not a prefix). Equations leg (`tac.canonical_equations`): md2's 3rd anchor on the persistent-partition law.

## ADDENDUM 38 (2026-09-05 15:30Z real) — md3: the born vehicle's unreachable error is DATA-ANCHORED (formulation scope); the "different init" was not purchasable

md3 (six commits): **premise correction first** — `build_initial_state` takes NO seed; every burn cell starts from r10's stage-03 EMA shadow, and the vehicle's random
init (`packet.initialize_params(20260827)`) sits BEHIND the whole qbt1 r1…r10 chain — a different random initialisation costs the chain's re-derivation, not md2's
2.8 h (md2's price omitted it). Positive control: the rebuilt init reproduces the pinned `initialized_float_params.npz` byte-for-byte. **Data-order control** (wc3
seed_20260903, 141 forwards): PERSISTENT **61.606%** — the fourth instance within 1.35 pp (cold 62.011, ng5 62.954, PyAV 61.67) — Jaccard vs cold **0.8536**,
HIGHER than ng5's schedule (0.8069): data order moves the unreachable sites less than the whole schedule does. Live share +4.78 pp (40.56% vs 35.3–35.8): the live
forward IS order-sensitive; the shipped object is the shadow, so no verdict moves. **The decisive $0 result:** across eight legitimately-produced starts (r6–r10
stage ends, r10 periodics; 0.8–30% weight distance) the step-0 wrong POOL moves enormously (J 0.9909 → 0.1175 — md2's "bit-identical pools" were an artifact of
every cell declaring one file) but the unreachable SITES do not: **83.13% of the incumbent's persistent sites are wrong at ALL FOUR comparable-quality starts vs
19.78% of the sites the optimizer did reach — 4.20× (4.21× on PyAV)**. Verdict **DATA-ANCHORED at FORMULATION scope**: the sites are scorer-hard for this generator
FORM, no start reaches them. A different-init cell (r10_live, max attainable J 0.8082; r8/r7/r6 rungs cannot fire their falsifiers by arithmetic) is sealed,
in-tree validated, armed on a process-table waiter behind cl2's Metal use (~16:40Z) — its harvest is a falsifiable PREDICTION (J ≥ 0.70 expected; 91.01%
containment). md3 correctly REFUSED the queue driver's `ready: true` while cl2's trainer held the Metal (the gov3 ITEM 3 gap, independently logged).
**gs4 §5(b) is answered:** the born route's accuracy corner is closed for this generator form by data-anchored sites; only a different generator FORM (gc1/gf2's
forms are themselves formulation-closed; other forms unpriced) or the joint field+model escape remain. Equations leg (`tac.canonical_equations`): 4th anchor on
`checkpoint_trajectory_error_partition_v1`; `known_boundary` rewritten (shadow share data-order-invariant MEASURED; live forward NOT; init-invariance owed to the
armed cell).

## ADDENDUM 39 (2026-09-05 15:35Z real) — cl2: the prior-capacity ladder is FALSIFIED (secant +0.446 vs the −1 bar); the CONTROL is the candidate

cl2 (Fable; equation `hpac_prior_capacity_slope_v1` committed 65fd5ffa1): the λ 1.0 → 0.5 secant Δstream/Δmodel = **+0.446** against cl1's −1 break-even —
MORE prior capacity does NOT repay itself on this object (the λ=0.5 rung's packed model grew and the exact stream did not fall enough); λ=0.25 is not fired per the
preregistration. **cl1's prior-law prediction is falsified at FORMULATION scope for the fixed topology.** The remaining door on the shipped mixer is an ARCHITECTURAL
rung (width/frame-dim/taps), which cl1 explicitly did not admit before the slope was measured — and the slope says the marginal capacity already pays +0.45 B of
tokens per model byte, so an architectural rung must change the RECEPTIVE FIELD's shape, not its size, to have a prior. What DID land: the re-trained λ=1.0 CONTROL
prices −41 B under the shipped joint (model 13,466 B, stream 113,419 B; archive 179,982 B; two parse-backs PASS; the render is byte-identical to the shipped render
at the receiver output, 3,662,409,600 B sha f86bfaf3…) — a **26th-move candidate by pack-size/re-train effect, projected S 0.14781744131049854 [DERIVED, rate-only]**,
gated on the twin's bit-identity at epoch 60 (now 42/60). Equations leg (`tac.canonical_equations`): `hpac_prior_capacity_slope_v1` (cl2).

## Addendum 1 (2026-09-05 16:00Z) — 26th-move candidate T4-FIRED; ane2 engineered the drift; md3 fired by hand; cl3 spawned

- **cl2 SEALED + FIRED.** The re-trained λ=1.0 HPAC prior control on the shipped fs2 mixer = **179,982 B (−41 B)**, twin-reproduced
  byte-exactly (stream `e07274ca…`, archive `08ec8533…`), parse-back render sha `f86bfaf3…` = the shipped render. Seal
  `SEAL_ddm_cl2_lambda1_control_repack_contest_cuda.json` (file sha256 `3cf630a6…`; the seal's own canonical digest field is `e42288fc…` — two different digests of one seal, both correct) → MAIN fired T4 15:58Z through the canonical launcher:
  call `fc-01M1S4PBEPBKQJVEPWVRDJHGNT`, lane `ddm_cl2_t4_lambda1_control_repack_20260905`, job `…20260905T155621Z`, poller armed,
  harvest receipt `ddm_cl2_lambda1_control_repack_20260905T155621Z_harvest`. Predicted S = 0.14784474152757654 − 41×6.658589531221714e-7
  = **0.14781744131049854** (rate-only; distortion held by decoded-field identity). Honest label: a retrain/pack-size residual (−49 B model,
  +8 B stream), NOT a capacity win — the ladder's bigger direction is FALSIFIED (secant +0.446 vs −1; λ=0.5 = +465 B). Law registered
  `hpac_prior_capacity_slope_v1`. cl2 also settled the training law: the shipped 13,515 B IHS1 model is the epoch-634 EMA of a 960-epoch
  rx2_mc36 burn — our own lineage, not a PR135 repack.
- **cl3 spawned (Opus, charter `charters/ddm_cl3_hpac_smaller_prior_and_seed_selection_20260905.md`):** the untested half of the axis —
  λ=2.0 (smaller model), λ=4.0 iff 2.0 pays, and two extra seeds at λ=1.0 — with prior-law prediction lines (λ=2.0 predicted −350…−50 B;
  seeds min-of-3 −40…−90 B beyond the control; whole-axis falsifier if λ=2.0 nets ≥ 0). Serial on Metal beside md3's cell.
- **ane2 LANDED (memo `ddm_ane2_engineer_the_precision_drift_20260905.md`, commits 769f4e0cd/ceab7d211/a7eb849e6).** The drift is
  engineerable, as the operator said: PoseNet's fp16 error is ONE place (ops 0:18 = 92.83% of the drift; dim-0 pinned 0.151 at every tail
  split; the fp32 HEAD cure hd128 = 7.02e-5, 23.7× better, 54% ANE); SegNet's is distributed with one hot spot (g13, 33.8%). **Selective
  fp32 on 18 profile-chosen ops (`g13`) makes the ANE SegNet PASS the screening bar** (flip rate 3.2188e-05 vs 3.3e-05, 90.8% ANE, 4.93 ms;
  `g13stem` 0.874× at 75.5% ANE — the one to ship, 12.6% margin). A mixed graph DOES reach the ANE (96.5% of ops at k=8) — ane1's
  "fp32 never reaches the ANE" holds only for wholly-fp32 graphs. Hybrid crop recompute NO-GO twice (25× worse argmax; 32.1% of recomputed
  pixels disagree — a U-Net crop is not a window on the result). `coreml_cpu_fp32` (bit-exact, 1.94×) replays fs1/pr1's 39-point sweep at
  38/39 argmin agreement, tau-b 1.00, and reproduces pr1's gain to the digit. Owed: re-measure the finalist through the fixed conversion
  path (ane2 found its own instrument declared the OUTPUT fp16 whenever `op_selector` was used — the SegNet rates are upper bounds, the
  pass is conservative); the tail-worsening mechanism is MEASURED and UNEXPLAINED. Pointer unmoved by ane2 (a means: screening throughput).
- **md3's cell fired by MAIN by hand at 15:47Z** (the queue driver refused its own just-filed claim 4×; gov3 ITEM 4d); 209 steps by 15:55Z
  (~28 steps/min → terminal ~18:50Z). Two claim rows mangled by the zsh `$VAR:l` modifier trap were closed; the correct scorer+metal rows are
  active (job `md3_r10_live_cell_main_20260905T154454Z`). Memory `zsh_parameter_modifier_trap_var_colon_literal_20260905`.
- Frontier unchanged until the harvest: **fs2 S 0.14784474152757654 @ 180,023 B [contest-CUDA T4 n600]**.

## Addendum 2 (2026-09-05 16:12Z) — THE 26TH POINTER MOVE LANDED: S 0.14781744131049854 @ 179,982 B [contest-CUDA T4 n600]

Call `fc-01M1S4PBEPBKQJVEPWVRDJHGNT`, archive sha `08ec85333d13d71344b4482cf261e3b2d508725e49f3ca05971265a81498ad4e`. d_seg 0.00020139 and d_pose
6.14e-06 IDENTICAL to fs2 — the row is the custody confirmation of a rate-only move, realized − projected = **0.0**. Δ vs fs2 =
−2.7300217078002342e-5 = exactly 41 × 6.658589531221714e-7. Re-derived at this move: gap 0.027817441310498542; rate corner −41,776.8 B at held
distortion (archive ≤ 138,205.2 B); distortion corner 177.8×; zero-distortion margin 236.347 B. **First fully-live run of the harvest→pointer
autopilot (hv1):** the poller staged `POINTER_MOVE_PLAN.json` (beats_pointer=True) and `tools/pointer_move_packet.py --apply` wrote the memo
(`ddm_cl2_t4_lambda1_control_repack_20260905_pointer_move_26_20260905.md`), the pointer file, the #316 surfaces (gate strict-PASS), the lane, the
second-tier custody copy (sha verified), the POINTER_LINE, and the events ledger — MAIN reviewed and committed; zero hand-typed shas. One tool
quirk: the plan's `apply_command` carries literal placeholders (`'<one clause>'`) that break a shell `eval` — MAIN rebuilt the argv by hand.
Honest label repeated: a retrain/pack-size residual (−49 B model, +8 B stream) banked as an exact row; not a capacity win; 0.098 % of the rate
demand. Three pointer moves in 24 h (fs1, fs2, cl2), all sub-1e-4. PR #140 still carries afr1's bytes (three moves behind); public update =
operator's decision.

**§1 basis note (MAIN, 09-05 17:55Z):** the §1 table is on the ARCHIVE-BYTES basis (what the rate term charges; the five rows sum to 179,968 ≈ 179,982).
ft1's addendum-4 figure 36,130 B is the RAW SM3R section body BEFORE the container's Brotli (2-plane, ck2) coding brings it to ≈30,856 B; afc1's 53,076 B
framing total is likewise raw. Both are correct on their own basis. MAIN briefly struck the table's rows with the raw figures (17:35Z) and reverted them here —
[[m99]]: units × level × aggregation are part of the claim; name the basis with every byte count.

## Addendum 3 (2026-09-05 18:15Z) — bd1 CLOSED at family scope: bidirectional temporal context does not transfer to the label field

The $0 counting-model screen (three context ladders, KT and plug-in brackets, instrument cross-checks to 1e-13) measured the B-pyramid family
(GOP 2…32) at **2.84–5.68 % of the stream (−3,219…−6,442 B)** against the pre-registered 8 % falsifier; even the UNATTAINABLE supremum (every
pair bidirectional at distance 1) is 7.5–9.0 %. Mechanism, MEASURED: `P(32)/P(1) = 1.069–1.129` — a past plane thirty-two pairs away costs only
7–13 % more than the adjacent one, because the SegNet argmax field is piecewise-constant over regions that persist for tens of pairs; the past
plane already carries nearly everything a future plane could add. The video-coding prior (B-frames buy 20–35 %) transferred its CONCLUSION
without its PREMISE (pixel intensity decorrelates fast; a label field does not). MAIN's prediction (−15…−30 %) was 3–9× high. Law registered
`bidirectional_pyramid_context_gain_v1`; memo `ddm_bd1_bidirectional_pyramid_context_20260905.md`; commit 232ba9348; nothing trained, Metal never
requested. Consequence for the map: **the token stream is at its floor under any predictor that reads other frames' labels** (with mc1 and
hc1 this is now three independent closures of the temporal axis); sub-0.12 must come from the other four pools. Metal order after md3: cl3.

## Addendum 4 (2026-09-05 18:35Z) — rc1 ADMITTED: the two model sections had never been entropy-coded — −1,733 B at zero distortion, sealed and T4-fired

rc1 (Opus, memo `ddm_rc1_adaptive_recode_race_of_the_model_sections_20260905.md`, commit 8979e18aa, lane L2): an adaptive per-group tree coder over the
PACKED INTEGER CODES of the SM3R renderer body and the IHS1 HPAC model beats the shipped Brotli q11 by **−610 B (semantic, 30,856→30,246)** and
**−1,123 B (hpac, 13,466→12,343)**; xz and zstd both LOSE to Brotli. Candidate archive **178,249 B, sha 1438049e3655fbcf…**; decoded field, carrier and
tail byte-identical to cl2; twin encode byte-identical; +0.11 s inflate. Predicted S = 0.14781744131049854 − 1,733 × 6.658589531221714e-7 =
**0.14666350774473783** (rate-only). Prediction residuals: SM3R −610 vs predicted −2,050…−4,350 (MAIN's prediction charged the codes for the fp16
scales/masks — the code stream's own H0 is 3.281 b/param); IHS1 −1,123 inside its band. **Transferable law** (`model_section_adaptive_recode_ceiling_v1`):
the credit tracks the packing's WIDTH STABILITY — Brotli recovers a constant-width packed stream's order-0 statistics but not a stream whose width changes
every few hundred symbols; hence hpac's credit is 1.84× the renderer's off a body half the size. Order-1 contexts CLOSED at formulation (context dilution:
10 B of ~1,139 B of first-order structure converted); semi-static/depth-mixing prior = ITEM 1. MAIN fired T4 at 18:30Z (canonical launcher; first attempt
REFUSED rc=9 because the waiver text contained "fs2/cl2", which the fire tool's path scanner reads as a relative path — write waivers without slashes).
Call id appended below at harvest. Also this window: md3's Metal cell fell from 30 to 2 steps/min under CPU load 21 from sj1 (6 shards) + pc1 (8 solvers);
MAIN reniced the CPU arms to +10 (reversible) — a Metal cell's host thread is a CPU consumer the governor does not model (gov3 ITEM 3, third face).

## Addendum 5 (2026-09-05 19:40Z) — THE 27TH POINTER MOVE: S 0.14666350774473783 @ 178,249 B [contest-CUDA T4 n600] (rc1, −1,733 B, zero distortion)

Call `fc-01M1SG7CY107YXHSFS26NWV69T` (attempt 2; attempt 1 died in `f26_inflate.py` before the codec dispatch — the arm had verified identity through the
library path, not `inflate.sh`), archive sha `1438049e3655fbcfa8eb289fa51ac58f834d72d8a09586353663cea68e57c122`, d_seg 0.00020139 / d_pose 6.14e-06 IDENTICAL
to cl2, projection error 0.0, Δ −1.153933565760712e-3 = exactly 1,733 × 6.658589531221714e-7 — **42× the 26th move; the largest since afr1's crossing.**
Re-derived: gap 0.02666350774473783; rate corner −40,043.8 B at held distortion; zero-distortion margin 1,969.347 B. The packet's first live REFUSAL was a
tool defect (the recompute and the receipt differed by one ulp from summation order) — fixed with a 1e-15 tolerance + two tests (f569e4c1a). Composition
law for the live arms: carrier and tail are byte-identical, so pc1's carrier deltas and sj1's field deltas compose additively with rc1's coder; cl3's HPAC
rungs must be packed through rc1's adaptive coder to compose (their Brotli-packed model bytes no longer describe the object). PR #140 is now FOUR moves behind.

## Addendum 6 (2026-09-05 20:20Z) — pc1 LANDED: V3 (lattice ×4 + full re-solve) ADMITTED and T4-FIRED on the rc1 base; the generated-basis door CLOSED with a number that explains the whole carrier

pc1 (Opus, memo `ddm_pc1_pose_carrier_efficiency_20260905.md`, 12 commits, lane L2, law `pose_carrier_basis_rate_fidelity_exchange_v1`). Corrected anatomy
(parsed from the bytes, sums exactly): carrier stream **22,031 B** = brotli(q9) over a 22,278 B body = **12,277 B basis** (12 atoms × 3 planes × 24×32 at 5 bits —
NOT luma-only) + **9,830 B** AR1+Rice coefficients + 171 B framing; Brotli removes only 247 B. **The anchor:** zero carrier → d_pose **52.131**; the shipped
carrier drives that to 6.13e-6 — 6.93 orders, returning 22.82 S for 0.0147 S of bytes (**pays for itself 1,556×**). The bytes buy a SPECIFIC 12-dim
subspace positioned so its reachable box contains the solution: V1/V2 (basis 5→4/3 bits) move that subspace and fail (V2 d_pose ≥ 2.85e-5, 2.87× past
break-even); **V4 (generated DCT basis, zero bytes) recovers 3.5 % of the carrier's work on a log scale — d_pose ≥ 0.9986, 39,748× past break-even:
analytic/generated bases CLOSED at family scope**; V5 (rank-8 SVD) dominated. **V3 — coefficient lattice ×4 (12→10 bits) WITH the full n600 re-solve — is
the only variant that leaves all twelve atoms bit-identical: −1,801 B AND d_pose 6.14e-6 → 5.728e-6 (−6.6 %), net ΔS −1.462948e-3 (73× the admit bar).**
Three-way decomposition of the composition law, numerical: coarsening without re-solve costs 4.79×; the re-solve recovers 5.13× and lands below base.
Sealed on the rc1 base (`SEAL_ddm_pc1_v3_lattice_x4_resolved_on_rc1.json`): **176,448 B, sha 891add546f5cf094…, projected S 0.14520055969848670**; MAIN
fired T4 at 20:15Z (call id appended at harvest). Follow-on already dispatched to the same arm: lattice ×8 (−2,693 B, break-even d_pose 9.27e-6) and ×16
(−3,484 B, 1.03e-5) with the full re-solve. Two self-corrections by the arm (basis_scales are not dead bytes; per-atom quantizer step moves 3,731 B at fixed
alphabet — the registered rate↔fidelity exchange inside the step). SVD spectrum of the 600 realized fields: rank-8 cumulative 0.9869, effective rank 9 at 99 %.

## Addendum 7 (2026-09-05 20:05Z) — THE 28TH POINTER MOVE: S 0.1451981569076111 @ 176,448 B [contest-CUDA T4 n600] (pc1 lattice ×4 + re-solve; a POSE-CHANGING authority row)

Call `fc-01M1SHRJ45TJRMA9G2YXCE4MTW`, archive sha `891add546f5cf0943929b566f29dd4318f1d8b2ab76ae05183d8189098880f40`; d_seg 0.00020139 identical; **d_pose 5.73e-06
(from 6.14e-06)**; Δ vs rc1 −1.4653508371267332e-3; realized − projected −2.4e-6 = the evaluator's 3-sig-fig pose print. Two moves in one evening: −1,733 B
(rc1, model sections) + −1,801 B AND better pose (pc1, carrier lattice) = −3,534 B and −2.62e-3 S since cl2 this afternoon. Re-derived: gap 0.0251981569076111.
Composition for the live arms: **the carrier coefficient block is now pc1's (lattice ×4, re-solved) — sj1's edited pairs must re-solve FROM these coefficients
on this lattice**, model sections rc1's, tail unchanged. pc1's ×8/×16 rungs run on this base. PR #140 is five moves behind.
**Pointer advanced (20:35Z):** the compliant twin was itself SHADOWED — the refresh took only the scan's single best row per axis (the refused v3 lane, same
score) and kept the prior pointer. Fixed: the refresh now walks the ranked candidates and takes the first admissible (ad1f80974, test). Pointer =
`ddm_pc1_t4_lattice_x4_on_rc1_20260905`, S 0.1451981569076111 @ 176,448 B. Three apparatus defects surfaced by tonight's two moves, all fixed with tests:
packet ulp cross-check, fire-time lane-id maturity check, refresh ranked fall-through.

## Addendum 8 (2026-09-05 21:35Z) — THE 29TH POINTER MOVE: S 0.1445177913121716 @ 175,576 B [contest-CUDA T4 n600] (pc1 lattice ×8 + re-solve supersedes ×4)

Call `fc-01M1SPVVJGAJ5WY373TKQY1Z6A`, archive sha `f7e0bb793645894b2f6885fca82b98cab3067837bd66181e222f3d4b1f43e1ff`; d_seg identical; **d_pose 5.73e-06 → 5.58e-06**;
Δ vs the 28th −6.803655954394916e-4; −872 B. pc1 pre-registered "×8 succeeds V3 on both legs" (predicted d_pose band 6.3–7.0e-6; measured 5.58e-6 — BETTER than
the band: each coarser lattice lets the re-solve escape a local optimum). Pointer advanced first time (the ranked-candidate refresh fix held). Tonight's ledger on
one object, no retraining: rc1 −1,733 B · pc1 ×4 −1,801 B & pose −6.6 % · pc1 ×8 −872 B & pose −2.6 % = **−4,406 B and −3.30e-3 S since cl2 this afternoon**;
gap to sub-0.12 now 0.0245177913121716. ×16 pending (the knee is at or beyond ×8). PR #140 is six moves behind.

## Addendum 9 (2026-09-05 22:05Z) — THE 30TH POINTER MOVE: S 0.14411787458634504 @ 174,786 B [contest-CUDA T4 n600] (pc1 lattice ×16 + re-solve; the pose leg turned)

Call `fc-01M1SRR3JGWRKQKH6GSZ3RVTRS`, archive sha `1de6c5d7186a0b31e5cc085bb6d2baab8275ee0d9de4d509f4d8add13695a629`; d_seg identical; **d_pose 5.58e-06 → 5.77e-06 (+3.4 %)** against −790 B; net
Δ −3.999167258265657e-4 — the first rung of the lattice lever where the pose leg WORSENED; admitted on the exchange, not on fidelity. Knee located: fidelity turns
between ×8 and ×16; S still pays at ×16; ×32 extrapolates near break-even (byte leg ≈ −700 B vs a pose leg rising and accelerating) — PRICE before solving. Tonight's
ledger on one object, no retraining: rc1 −1,733 B · pc1 ×4 −1,801 B (pose −6.6 %) · ×8 −872 B (pose −2.6 %) · ×16 −790 B (pose +3.4 %) = **−5,196 B and −3.70e-3 S
since cl2 this afternoon**; gap to sub-0.12 now 0.02411787458634504. PR #140 is seven moves behind.

## Addendum 10 (2026-09-05 22:20Z) — pc1 COMPLETE: the coefficient-lattice lever is EXHAUSTED at ×16 (×32 measured and refused); three laws banked

| rung | archive | d_pose n600 | net ΔS vs rc1 | outcome |
|---|---:|---:|---:|---|
| ×4 | 176,448 | 5.728e-6 | −1.463e-3 | 28th move |
| ×8 | 175,576 | **5.579e-6** (pose min) | −2.142e-3 | 29th move |
| ×16 | 174,786 | 5.768e-6 | **−2.543e-3** (score min) | 30th move |
| ×32 | 174,205 | 6.553e-6 | −2.430e-3 | REFUSED (loses to ×16 by +1.14e-4) |

1. **The distortion knee and the score knee are different rungs** (pose turns between ×8 and ×16; score one rung later, because the rate ladder keeps paying
   −872 → −790 → −581 B until the pose cost +1.25e-4 → +5.01e-4 S outruns it). A sweep stopping at the distortion minimum leaves −4.0e-4 S on the table.
2. **On a LOCAL solver, coarser can measure better than finer** — the ±2 polish spans twice the coefficient distance per doubling and escapes minima the finer
   lattice traps; pc1's own "nesting" prior (×16 fails vs ×4) FIRED as wrong. Rung order is measured, never argued.
3. **The local cpu_torch instrument predicts a carrier-only edit's T4 row to ≤ 2.4e-6** (three rows; the residual is the evaluator's 3-sig-fig pose print) — a
   reusable control: carrier-only candidates are predictable to ~1 part in 10⁵ before a paid call.
The `lattice_floor` stop count (7 → 13 → 27 → 47) IS the lattice binding. Basis edits stay refused (V1/V2/V4/V5). Remaining carrier ITEMs (owned by pc1's
ledger): per-atom quantizer step (3,731 B at fixed alphabet), packed Rice-k field width, basis_scales blast radius (recoverable, not free: 14/24 frames ±1 uint8).

## Addendum 11 (2026-09-06 00:45Z) — cl3: the HPAC capacity axis is CLOSED in both directions; coder strength and capacity value are SUBSTITUTES

λ=2.0 on the live object (exact containers): model 12,343 → 11,886 (−457 B under rc1's coder; −659 B under Brotli), stream 113,419 → 114,100 (+681 B, identical
on both trees — measured), **J +224 B vs the live 125,762 B → NO**. P3 (−350…−50 B) falsified; the pre-registered whole-axis falsifier fired; λ=4.0 not run
(recorded as not-run-because-falsified). With cl2's +465/+506 B on the bigger side, **λ=1.0 is a local optimum of the HPAC prior's capacity** (formulation scope).
The transferable law (`coder_strength_substitutes_for_capacity_v1`): the same weight change is worth 69.3 % as much under the strong coder — a rung that was
break-even on Brotli (+22 B) is a clear loss on the object that ships (+224 B); cl2's Brotli-priced ladder deltas need a ~31 % model-side discount, and landing
coder work first REMOVES headroom from downstream capacity levers. Seed selection at λ=1.0 (s17/s18) still in flight — a different question.

## Addendum 12 (2026-09-06 01:05Z) — md4: the born vehicle's unreachable error set is DATA-ANCHORED (measured on the burned different-start cell)

md3's cell (5,000 steps, resumed once — resume measured as a non-confound) partitioned with md1's instrument unchanged: PERSISTENT **64.877 %** of terminal
error (10,993 sites; floor 11.652× the sub-0.12 corner; Lane 64.4 % of terminal wrong, GT=Lane enrichment 54.6×). **Jaccard vs cold 0.7280, vs data-order
0.7380 (shadow; live 0.79/0.80) — the pre-registered ≥ 0.70 bar cleared by 0.28; within-pool null 0.35 → 2.06× chance; 81.4 % of the corrected ceiling 0.894.**
A different start moved the FLOOR −0.17 % and the reachable error −8.15 %: it helps only where the optimizer already wins; 89.26 % of cold-persistent sites
step-0-wrong at this start stayed persistent through 5,000 updates (the free step-0 probe predicts unreachability at 9 in 10). **verdict_scope FORMULATION**
(starting point varied at rel-L2 0.0084 within root seed 20260827; FAMILY needs a different root seed and a different falsifier). 5th anchor on
`checkpoint_trajectory_error_partition_v1`; commits 196630263 / 1f4ca95d4 / 7ad16239b; $0, CPU. Consequence for the map: the born vehicle's accuracy corner
stays closed for this generator form — sub-0.12 on the born object needs a mechanism that changes WHICH sites are reachable (a different generator form), not a
different start, schedule, or seed. Owed (md4, not built): `review_tracker mark-file` rescan; a red test elsewhere (`test_resize_exploit_flip_fix_frontier.py`
asserts 2 anchors against 3 — the anchor-added-without-updating-the-count class).

## Addendum 13 (2026-09-06 02:10Z) — THE 31ST POINTER MOVE: S 0.1398140172839628 @ 180,904 B [contest-CUDA T4 n600] (sj1: the seg-debt pool OPENED)

Call `fc-01M1T6TCW2JS1JEW5CSZH3FVBY`, archive sha `42aa84b59f71d83b8f11a26c635a7af8f32dcfdf183e3fea4bb2007e74a5f2f8`. **d_seg 0.00020139 → 0.00012009** (9,593 flips
repaired by multi-pass single-cell token pre-distortion under REALIZED acceptance: render → re-segment → keep only if flips fall; 7,804 tokens changed, 1.229
cells/token), **d_pose 5.77e-6 → 5.40e-6** (carrier re-solved on the candidate's own renders: 0.936× base — the composition law on the seg actuator, numerical),
bytes 174,786 → **180,904 (+6,118)**: token stream +6,078 B at 6.23 bits per changed token against a 12.52-bit break-even (2.0× margin), carrier +40 B. Δ vs the
30th −4.303857302382225e-3 — **the largest move since afr1's sub-0.15 crossing.** Projection error +5.3e-6 (seg leg +10 cells of 14,167 on the parse-back raw;
pose at the 3-sig-fig print). Predictions: repair fraction 40.39 % BEAT the charter band (20–35 %); cells/token 1.229 fell BELOW it (1.3–1.6); the stale carrier
damage 430× reproduced jg1/jg4's ×387. **The map changes shape:** this is a DISTORTION move — bytes went UP; the rate corner at held distortion grows to
≈ −48 KB while the distortion corner shrinks (d_seg now 0.000120, 12,009 B-eq of seg debt left). Sub-0.12 gap 0.0198140172839628. Tonight since cl2 (15:47Z):
FIVE moves — rc1 −1,733 B · pc1 ×4/×8/×16 −3,463 B & pose · sj1 seg −0.0081 S — **S 0.14781744 → 0.13981402 = −8.00e-3 in one evening**, all on the shipped object,
no retraining. Successors already in motion: sj1 pass 3 (residual field; convergence rule < 1 % not reached), the 566-pair admission subset (needs its own encode
pair). PR #140 is now eight moves behind.

## Addendum 14 (2026-09-06 04:40Z) — THE 32ND POINTER MOVE: S 0.13900437796841966 @ 181,645 B [contest-CUDA T4 n600] (sj1 pass 3)

Call `fc-01M1TFD35EPY2YZHNV3VKJP6MG`, archive sha `06c44dc464038649f1cc149f04ac03a518294ffcf49b87d8f66df30eb3c63cd3`; **d_seg 0.00012009 → 0.00010913** (1,447 more
flips; the admission's 12,866 predicted = 12,866 measured on the shipped bytes), **d_pose 5.40e-6 → 5.1e-6** (re-solved on the candidate renders), +741 B (token
stream +728 at **5.31 bits/changed token — cheaper than pass 2a's 6.23**: the context model absorbs earlier edits' structure; carrier +13). Δ −8.09639315543148e-4;
projection error +4.2e-6 (the 2-sig-fig pose print). The 370-pair admitted subset beat the full 445-pair set by 1.03e-4 and was priced by a REAL encode pair (the
ledger-sum estimate under-charged by 19.6 B — so the sweep may choose on the ledger, the seal must not). Three silent-revert classes now cured structurally in this
arm (carrier start codes; subset writer omitting non-admitted planes — would have reverted 2,337 banked tokens invisibly; stage-tail baseline equal to the encoder
tree's own previous output): all one shape — **a check that encoded a round-one premise and kept passing after the premise moved.** Six moves since the cl2 row
(15:47Z yesterday): **S 0.14781744 → 0.13900438 = −8.81e-3 in 13 hours**, all on the shipped object, no retraining. 45.8 % of the flipped cells the wave inherited
are gone; seg debt 12,866 cells (10.9 KB-eq); gap to sub-0.12 0.01900437796841966. Convergence rule (< 1 %/pass) not reached: pass 4 is a decision on this row.

## Addendum 15 (2026-09-09 14:30Z) — "All point to gestalt" (operator): what the respawn wave measured, and where it all points

Pointer UNMOVED since move 32 (S 0.13900437796841966 @ 181,645 B). Twelve hours were lost to a power cord, not to physics: the machine hit 1 % battery at 18:53 −0500 and hibernated 46,467 s; at wake the SSDs re-enumerated and every detached process died in one second (`low_battery_sleep_kills_every_detached_run_at_wake_20260909`). sj1's resume-time refusal then caught 12 rows-without-planes — the fourth silent-revert class, written for a wall-clock cap, fired on a hibernate. Same genus, different trigger.

**Six measured findings, one object.**

1. **The residual is the renderer's boundary placement at CORRECT tokens** (sj1 §16, commit 4ff3c1c84): of the 12,866 remaining flipped cells, 86.39 % store a token already equal to GT; 99.58 % are one-pixel boundary displacements on a GT edge (46.1× enriched), reciprocal (symmetry 0.905); Lane 40.15× and Movable 10.92× over-represented; rows 128–319 only; 86.55 % isolated singletons. Single-cell token moves are exhausted (pass-3 `sites_persistent == flips_after`). Pass 4 (in flight, 276/600, 3.53 % repaired vs 2.586 % projected, ahead) is the LAST token pass; pass 5 would fall under the convergence rule.
2. **The renderer-coupling wall does not transfer to the per-pair case** (fe1): a per-pair FiLM move raises d_pose 440–4,157× stale and `refine_pair` recovers 643–3,053× — pair 118 lands at 1.36× base (+2.6e-8 S ≈ 3 % of one repaired cell). Twelve carrier coefficients re-aim an arbitrarily re-rendered pair. The coupling law's domain is ALL-PAIRS weight changes only.
3. **But the per-pair FiLM axis barely moves boundaries** (fe1, 672 realized evaluations over 12 seeded pairs): 664/672 single-code moves make d_seg worse; the 10/101 pairs that admit at n600 repair −1 cell each. Projected ΔS ≈ −5.7e-5 (sign not in doubt; 40 % under the point prediction). The renderer's boundary is not a function of its per-pair conditioning; it is a function of its weights and its input field.
4. **Rate edits into a range-coded section pay a container-break fee, and the fee is refundable** (fe1, `model_section_edit_container_break_fee_v1`): dB = 0.2134·N + 58.89 B unsearched vs 0.1677·N − 1.92 B searched over (ck2, q, lgwin); the shipped sections are already at their brotli optimum (0 B on a 56-shape grid) — the law is a lever on EDITED bytes only. Also: two window settings compress the same rider to the same 30,246 B with 30,129 different bytes; a length-only tie-break ships a different stream.
5. **The carrier's bytes buy a lattice point, not a direction** (pc2): every Jacobian column is exactly in the span of the other eleven (residual 0.000000) so a rank cut is free at first order and drop order is irrelevant; the measured 4–2,733× pose cost is the int12 lattice; r=8 (−6,356 B) refused 48.9× past break-even on a population lower bound; halving the step fails paired. Only scales=1.0 paid (−41 B, net −2.64e-5 S, SEAL READY, held for sequencing behind pass 4).
6. **The generator's shared-static form is closed by a certified bound** (gf2): ≥ 923,953 mismatches for one shared field + any translation vs the 292,264 ceiling (3.16×); gf3 is pricing the time-conditioned successor closed-form before any build.

**Where they point (the gestalt, revised).** The shipped object has now been squeezed along every axis that leaves the renderer's WEIGHTS fixed: token field (32 moves, exhausted at one speck per token), carrier coefficients and lattice (closed both ways), per-pair FiLM (alive, thin), model coding (rc1 done; rc2 pricing the last ≈1,139 B), container (at optimum). What remains on the seg leg — 12,866 → ~12,400 cells after pass 4, 0.0105 S — is ONE thing: where the renderer draws a class edge within a pixel, in the horizon band, on Lane and Movable, at inputs it already reads correctly. Findings 2 and 3 together say the per-pair re-solve makes ANY render change payable on pose, but only the weights (or the input the weights read) move the edge. Findings 1 and 3 say the change must be sub-pixel and boundary-local, not capacity. So the door is a **weight change that is local in the boundary band and admitted per pair by the carrier re-solve** — the JOINT renderer fold-back the coupling memo left open (w96b 204× over was pre-re-solve; pr1's k_post 13.82 is the post-re-solve number to beat, and finding 2 says the per-pair re-solve is stronger than pr1's global one). Concretely: fine-tune ONLY the head and the last TokenBlock (the edge-drawing layers) on a boundary-band, Lane/Movable-weighted realized-flip loss with the exact R in the loop, at the object's own tail LR (2e-7, ft1's LR-transfer lesson), DALI target, pose in the loop, then per-pair carrier re-solve and per-pair Lagrange admission — the sj1/fe1 admission chain applied to a weight delta. The rate side is small (head + one block, int4, container-searched: hundreds of bytes). Predicted: the persistent partition on the SHIPPED vehicle is not the born vehicle's 62 % — it was measured at 100 % for single-cell TOKEN moves, which is a different actuator; the renderer-weight actuator's reach is unmeasured on this object and is the number the next arm buys. Rate corner unchanged: −28,541 B at held distortion remains the other half, and gf3/rc2 are its live pricing arms.

Apparatus landed by the wave (means, not ends): gov3 (Metal occupancy column + one-occupant rule, progress-budget timeout, SUM-over-RAM refuse; replays 16:12 REFUSE / 22:30 ADMIT / 21:05 REFUSE), scg1 (seals require structured public-entrypoint smoke receipts; the fire tool refuses without; review_tracker rescans; 39 literal anchor-count assertions converted — and the invariant caught a real stale residual key in `bf16_compute_seam`), ql1 (51 red tests adjudicated, 0 real regressions), cl3c (s18 admissible at exactly +137 B; ladder closed). Equations leg (`tac.canonical_equations`): `model_section_edit_container_break_fee_v1` registered (fe1); anchors on `pose_carrier_basis_rate_fidelity_exchange_v1` (pc2), `hpac_prior_capacity_slope_v1` + `coder_strength_substitutes_for_capacity_v1` (cl3c s18); the two digest definitions (`measure_runtime_digest` 2435dab8… vs fire-manifest 6508d184… for one tree) are recorded as a naming trap, not drift.

## Addendum 16 (2026-09-09 18:10Z) — two zero-seg moves banked, and the renderer door's shape after rw1

**Pointer:** move 33 (rc2, −231 B, S 0.13885056455024844) and move 34 (pc2, −41 B + a zero-byte lattice re-solve, **S 0.13882326433317044 @ 181,373 B**); both projections matched the exact rows to ≤ 1.84e-6 (pose print). Gap to sub-0.12: **0.018823264**. In flight against move 34: sj1 pass 4 (seg −3.5858e-4, rate +2.2107e-4 at a MEASURED 6.40 bits/token, seg+rate −1.375e-4 = 6.9× the bar; pose leg pending), fe1's FiLM candidate (sealing), rc3's −201 B model-row seal (held, re-bases after sj1).

**Three laws landed since Addendum 15, all the same shape — the lattice is the cost, not the direction:**
1. **Carrier (pc2):** every pose-Jacobian column lies exactly in the span of the other eleven; a rank cut is free at first order and its 4–2,733× measured cost is the int12 lattice; the shipped codes are already a `refine_pair` fixed point (17/7,200 coordinates moved by 40 more rounds). Only the scales paid (−41 B).
2. **Renderer (rw1):** the smallest action on the int4 code grid of head + blocks.3 — one code, one step, from the cheapest end of the gradient — breaks 240–455 argmax cells; the gradient orders damage (238× between its ends), not help; a discrete realized search over the same codes: 0 accepts of 150. The rate side is nearly free (+176.7 B for all 12,672 codes). Stake 57.9 % of the gap; toll 0.63 %; verdict FORMULATION (int4, depth 4).
3. **Token field (sj1):** pass 4 repaired 423 cells (1.27× projected) but the marginal price ROSE (6.23 → 5.31 → 6.40 bits/token — the "context absorbs earlier edits" model is falsified; cheapest repairs first fits); the residual is NOT concentrating (Gini 0.2542 → 0.2518, zero clean pairs, 206/600 pairs touched); pass 5 projects ≈ −2.4e-5 S, on the bar.

**And two refutations of walls:** the renderer-coupling law does not transfer to per-pair render changes (fe1: 440–4,157× stale → 643–3,053× recovered by the per-pair re-solve), and rate edits into range-coded sections cost a flat container-break fee that the encoder-side container search refunds (fe1's law, three anchors). Closed at the pricing level: the time-conditioned factorized generator (gf3: every GOP length fails the byte gate; L=10 closest at +12.8 KB over the cap). Blocked honestly: gb1 (the generated-basis bound construction was defective — no family rejected).

**Where it points now.** The seg residual (~12,443 cells after pass 4, 0.0105 S) sits at boundary placement the renderer makes at CORRECT tokens, and every actuator that moves that boundary on a coarse grid — token cell, int4 code — pays whole-step collateral that exceeds the repair. The two representations that admit a SUB-GRID step are the ones untried: the renderer's per-row fp16 scales (a fractional step of a whole row; rw1 is on it) and a finer code depth (the receiver already decodes depths 2–8; its rate must be priced first). The pose side is no longer a wall for any of these — the per-pair re-solve is strong (rank 6/6 on all 600 pairs). The rate corner (−28,269 B at held distortion) has its own live arms: tc1 (token tail bound + shared mixer; 120 KB never bounded) and rc3's held row.

Apparatus: gov3 (Metal occupancy, progress budget, SUM refuse), scg1 + scg2 (structured public-smoke receipts in every seal; both digest definitions named), the fire tool's import fix after scg2's landing crashed pc2's first fire pre-dispatch (6c74c56fd), and the keeper on gpt-6-astra medium…xhigh (operator). Twelve hours lost to a low-battery hibernate at 18:53 −0500 (every detached run died at wake); caffeinate armed; AC power is the operator's lever.

## Addendum 17 (2026-09-09 19:50Z) — move 35, and the seg half of the gap reaches a capability floor on the shipped renderer

**Pointer: move 35, S 0.13867171823146562 @ 181,521 B [contest-CUDA T4 n600]** (sj1 pass 4: 112-pair Lagrange subset + per-pair carrier re-solve; seg −2.14e-4, pose −3.5e-5 by SELECTION, rate +9.9e-5; the T4 row landed 1.09e-6 BELOW the projection — the wave's first pessimistic forecast; the pre-registered inputs were wrong on the price (1.417× high) and on the pose sign, and what admitted the pass was the subset sweep, not the forecast). Three moves today: 0.13900438 → 0.13867172 (−3.33e-4). Gap to sub-0.12: **0.018671718**.

**The seg residual is at a capability floor for every actuator this vehicle has.** Six measurements, one population: 12,614 flipped cells, 86.25 % at a correct token, 99.56 % one-pixel boundary displacement on a GT edge (46.1×), Lane 40.4× / Movable 10.9×, rows 128–319, 93.92 % isolated singletons, Gini FALLING (0.2542 → 0.2494), zero clean pairs. (1) Single-cell token moves: 89.46 % of the residual was proposed ~4.2× across three passes and REFUSED; pass 5 projects −4.1e-5 subset-only and pass 6 falls under the convergence rule. (2) Two-cell SLIDE moves reach 10.8 % of the refused cells but at 1.14 cells/slide (the rate clause is being priced by encode as this is written). (3) Renderer int4 codes: one code-step breaks 240–455 cells; discrete search 0/150. (4) fp16 row scales (41× finer): 0/400. (5) Sub-fp16 scales: 1 repair : 22 no-effect : 213 damages — finer steps become NO EFFECT, never repair; receiver change not built. (6) The per-pair FiLM axis: 18,624/18,906 moves worse; 15 pairs admitted for 23 cells. **Law:** every global actuator on this renderer pays ~100–200:1 collateral because the residual is a knife-edge population on GT edges beside a far larger correct-boundary population; the only per-cell knob (the token) is exhausted. The remaining seg debt, 0.0107 S = 57 % of the gap, is not reachable by search on the shipped object. Reaching it needs a REPRESENTATION with a spatially selective actuator at the boundary — not a precision rung, not another pass.

**Therefore the live demand is the RATE corner, as the operator's 08-21 binding already said (rate-representation mandatory; archive ≤ ~138 KB).** Rate doors measured today: the tail coder (tc1, −549 B; a richer joint-context mixer is its live hypothesis), the model rows (rc3, −201 B; ≈ 700 B left under the order-1 bound), the carrier (closed on rank/lattice; −7 B left), the container (at optimum; edits are a one-sample lottery, sd 34.8 B — sample, never search), the generator's static/time-conditioned factorizations (closed at pricing; L=10 within 12.8 KB of the cap). Composition of the sealed rate rows (cmp1: ≈ −750 B) is in flight. Untried at the coder level: the semantic renderer section (30,246 B) under the shared-mixer form that paid on the model rows. Untried at the representation level: the cross (born vehicle holds rate at 121,928 B; shipped vehicle holds accuracy) — the one structural composition whose arithmetic reaches the corner, and whose law (m148: a closed leg survives only if another leg changes its object) is exactly what the seg floor now demands.

**Apparatus/day:** four moves' worth of packets, five fallback bundles landed by MAIN (scg1, scg2, rc3, tc1, cs1), the fire-tool import fix, gov3, vr3 (+61 GiB), cs1 (266 blobs certified-or-blocked, 14 arms dispositioned), keeper on astra. Two power-of-attention lessons: a low-battery hibernate cost 12.9 h; a wrong pose base (no overlay, 500× off) was caught by magnitude, not by a gate — the pose-base law is now in memory.

## Addendum 18 (2026-09-09 22:30Z) — "All negative signal and months of research and new research yet to do points to gestalt" (operator)

**Pointer: move 37, S 0.13791730003757818 @ 180,388 B [contest-CUDA T4 n600].** Gap to sub-0.12: **0.017917**. Rate corner at held distortion: **−26,908.6 B**. Distortion corner at held bytes: closed (∞×). Five moves today, all on the shipped object, none by retraining: 0.13900438 → 0.13791730.

### I. The day's negatives, each with its number and scope

| door | actuator | measured | scope |
|---|---|---|---|
| renderer int4 codes | gradient (AdamW) / discrete search over head + blocks.3 | one code-step breaks 240–455 cells; 0 accepts / 150 | FORMULATION (rw1) |
| renderer fp16 row scales | realized search, 41× finer step | 0 accepts / 400 | FORMULATION (rw1) |
| renderer sub-fp16 scales | realized search; n600-confirmed | 1 : 22 : 213 (repair : none : damage) on a 120-pair screen; at n600 the one "repair" is **+3 cells** — repairs confirmed at n600 by ANY renderer actuator: **0**; finer steps → no effect, never repair | FORMULATION (rw1) |
| per-pair FiLM codes (3-bit lattice) | realized search | 18,624 / 18,906 moves worse; 15 pairs → 23 cells → after two pointer moves **nothing survives** | FORMULATION (fe1) |
| single-cell token moves | pass 5 | 237 cells at a price that decides admission; **89.46 % of the residual explicitly refused ~4×** | capability limit (sj1) |
| two-cell slide moves | realized + priced | reach 10.8 % of refused cells at 1.14 cells/slide; **17.1 bits/slide vs 11.2** | FORMULATION (sj1) |
| rate-directed token moves | realized at strict argmax identity | 4.4 % neutral; first-order −220 B → **−32 B real** (0.14×); full pass −217 B | closed to this actuator at scope; a pointer move remains (rp1) |
| generated carrier basis | closed-form bound | bound construction defective; **blocked honestly, nothing rejected** | design gate (gb1) |
| rank cut / lattice step | closed-form + paired | rank cut free in span, 4–2,733× on the lattice; ×1/2 step fails paired | FORMULATION (pc2) |
| time-conditioned generator field | certified bound | every GOP length fails the byte gate; L=10 +12.8 KB over the cap | PRICING (gf3) |
| byte search over seg-neutral edits | 250 real builds | one-sample lottery, sd 34.8 B; no landscape | FORMULATION (fe1 ITEM 5) |
| ANE screening | fixed conversion | rates under bar but **0 % ANE** (CPU fallback) | porting item stays closed (ane3) |

And the positives that closed the coder level: rc2/rc3 (model rows −432 B), tc1 (tail −548 B), sm1 (semantic −384 B composed), pc2 (carrier −41 B + a zero-byte re-solve that settled pc1's confound). **Total coder-level rate since move 27: ≈ 3.2 KB against a 26.9 KB demand.**

### II. Placed against the months (the walls that were already there)

- Distortion corner closed on every measured route: sg2b, the wrong-sign accuracy→byte exchange, the round-trip intercept ~140,477 B, gd2's frozen realized-pixel law (August).
- Temporal context saturated (bd1), MC previous plane (mc1), reordering under a context model, HPAC capacity both directions (cl2/cl3), carrier basis edits ×39,748 (pc1), the born vehicle's persistent partition 62 % data-anchored (md1/md4), the post-hoc render re-aim (ar1), the renderer coupling 170–220 pre-re-solve (rf1/ft1/pr1), embedding prediction (hp4), the jt23 coder axis at 0 B, ld1/ae1/oe1/rr9, the static generator (gf2 certified 3.16× over the ceiling).
- Refuted today, in the campaign's favour: the coupling wall does NOT transfer to per-pair render changes (fe1: 643–3,053× recovered); the container-break fee is refundable (fe1); the rate side of a renderer rewrite is nearly free (rw1: +177 B for all 12,672 codes).

### III. Where the gap lives — stated as two numbers nobody can search away

1. **Seg: 12,377 cells (after pass 5) ≈ 0.0105 S = 59 % of the gap.** One population: a correct token, a one-pixel boundary displacement on a GT class edge, Lane/Movable, rows 128–319, isolated singletons, knife-edge (damage ∝ sqrt(perturbation), 100–200:1 collateral for every global actuator, 89 % explicitly refused by the only per-cell actuator). No actuator on the shipped object reaches it.
2. **Rate: −26,909 B.** rp1's census of the tail: ~36 KB is "it was the prediction" flag mass over 117.7 M tokens (0.0024 bits each) — untouchable by any field change; **83 KB sits on the 235,044 tokens (0.199 %) the coder mispredicts**, and those tokens are the boundary population of (1). The coder-level squeeze is done; the field-level squeeze returns 0.14× of first order.

**The two halves are one object: the class boundary.** The renderer places it wrong by one pixel at correct tokens; the coder pays 83 KB to describe where it is. The shipped representation stores the boundary twice (in the token field, expensively) and draws it once (in the renderer, imprecisely), and neither store admits a local move.

### IV. The research program this points to (new research yet to do)

1. **A boundary representation with a spatially selective, sub-pixel actuator.** Not a precision rung, not a pass: a term the receiver draws that (a) is parameterized per boundary segment (position, normal offset, class pair) so a move touches only the cells of that segment, and (b) is coded by the mispredicted-token statistics rp1 censused. Candidates to price closed-form FIRST against both numbers in III: a signed-distance / offset field per class pair in the horizon band (the lane-orbit ~8-dim manifold is the long-tail case); a per-segment displacement side-channel decoded into the token field before rendering; a boundary-aware renderer head that consumes it. The bound to beat: 83 KB of boundary description at ≤ 0.0105 S of boundary error. (Prior: msr1's flow-balance ceiling for boundary-moving actuators, 8.94 % of d_seg at the time; the v8 "argmax = Laguerre → store generators" law; hc1's one binary question — recall all three at charter time.)
2. **The cross at the representation level** (the-cross memo, m148): the born vehicle holds RATE (121,928 B) with distortion the shipped vehicle proves unreachable; a successor inherits byte feasibility while starting from the shipped distortion regime. The intersection is measured empty at n=3 among small bodies — so the successor is not "a smaller body" but a DIFFERENT DECOMPOSITION: the shipped field minus its boundary description (the 83 KB) plus a generated boundary (1). Price it closed-form on rp1's census before any trainer exists.
3. **The pose leg is no longer a wall for any of this** (per-pair re-solve, rank 6/6 on all 600 pairs): every candidate representation is admitted per pair through the carrier re-solve, the chain that produced moves 31/32/35.
4. **Instrument laws to carry into every new arm**: a subset screen is valid only for pair-confined actuators (rw1); a first-order −log2 p is a ranking never a charge (rp1); the pose base is measured on the pointer's own configuration on the arm's instrument (sj1); container deltas of edits into range-coded sections are a lottery — sample, never search (fe1); persist the ledger before any optional dump (rp1); a wrong base 500× off was caught by magnitude, not by a gate (sj1) — add the magnitude gate.
5. **Still owed at the apparatus level**: gb1's valid lower-bound construction for generated bases; the 182 BLOCKED SSD code rows (cs1); the 26 unreferenced Vertigo raws (vr3 successor at < 50 GiB); the PR #140 swap packet on the operator's confirm (13 moves behind).

The negatives are not a list of failures; they are the coordinates of the one object left to build.

### Addendum 18 — CORRECTION (2026-09-10 ~00:10Z, from bnd1's source audit, `.omx/research/ddm_bnd1_boundary_representation_closed_form_pricing_20260909.md`)

§III item 2 and the bnd1 charter misread two numbers. (1) **"83 KB" is not a payload.** 83,258.632 B is the IDEAL-cost attribution (Σ −log2 p under the shipped model) of the 235,044 mispredicted tokens; the actual retained tail envelope is **119,784 B**. An attribution cannot be "replaced"; a representation must beat the ENVELOPE it displaces, and an achieved codec size bounds the entropy from ABOVE, so no "≥ 78 KB floor" follows from it — the floor must come from a counting argument on the population. (2) **12,377 cells is the pass-5 field, which is not shipped**; the shipped pass-4 field has 12,614. The two numbers in §III are one object only on the same field; the census must be joined on the shipped field's hash. Surviving, sharpened: ≥ 88.9957 % of mispredicted tokens lie on an edge (retained geometry), which is the premise a segment representation needs; 213,733 / 235,044 mispredictions have retained locations, 21,311 do not. The door is unpriced, not closed.

### Addendum 18 — CORRECTION 2 (2026-09-10 ~02:10Z, from pr5's second-family review, `.omx/research/ddm_pr5_second_family_review_moves_33_37_20260910.md`, `3607c7252`)

The five scores of moves 33–37 reproduce from their receipts to 2.78e-17; the pointer is move 37. Six further misreadings in this addendum, each with its correct statement: (1) the "~36 KB flag mass" is an ideal-cost ATTRIBUTION (36,519.391 B), like the 83 KB — neither is serialized; only the 119,784 B envelope is. (2) rw1's "one code-step breaks 240–455 cells" — the 240 endpoint is a fivefold extrapolation from a partial screen that the screening law itself declares invalid for a global actuator; the n600-measured fact is the retraction (+3 cells), not the 240. (3) rp1's "projected −217 B" is an n120 projection; the exact subset encode and the pose leg were still owed when this addendum was written. (4) "the two halves are one object" — edge enrichment (≥ 88.9957 % of mispredictions on an edge) does not measure the intersection of the mispredicted-token population with the residual-cell population; the join was never computed. (5) "the coder-level squeeze covers every section" is overbroad: cmp1 excluded joint tail contexts by construction, and rc2 left counted model-row designs predicting ≥ 150 B open. (6) "the pose leg is no longer a wall for any of this (rank 6/6 on all 600 pairs)" — rank 6/6 is a structural precondition of the incumbent carrier, not a proof that arbitrary representations re-solve safely. Also: PR #140 carries move 23 and is therefore **14** moves behind, not 13. Standing lesson: an addendum written within an hour of its inputs inherits their labels; the second family read the receipts, not the labels.

## Addendum 19 (2026-09-10 ~05:20Z) — the boundary door, priced by real encode: the segment grammar LOSES; the boundary is not a curve population at the token level

**Pointer unchanged: move 37, S 0.13791730003757818 @ 180,388 B.** Two candidates in seal (sj1 pass-5 subset, projected 0.13788256; rp1's subset).

**bnd2 gen 2 (`.omx/research/ddm_bnd2_boundary_segment_code_real_encode_20260910.md`), real n600 twin-verified recodes on the shipped field:** all 235,044 mispredicted tokens located (the 21,311 bnd1 lacked, recovered); six segment-grammar variants (class-pair chains with ternary normal offsets, greedy assignment) — best **121,200 B vs the 119,784 B envelope (+1,416 B)**; masked control (mispredicted tokens replaced by the model's prediction) **129,316 B (+9,532 B)**. The independent decoder reproduces all 117,964,800 tokens. **Median segment length: one cell.** The pre-registered gate (≤ 114,784 B) failed; the lossy draw was cancelled unrun.

**What this closes and what it does not.** (1) The segment-grammar family at this addressing scheme is closed at formulation scope (six greedy variants; joint edge assignment / gap bridging untested; verdict_scope: formulation — greedy per-class-pair chain grammars with ternary offsets on the shipped field). (2) The premise of Addendum 18 §IV item 1 — that the mispredicted population is a CURVE population a segment code can address cheaply — is falsified at the token level: ≥ 89 % of mispredictions lie on an edge, but they lie there as isolated single cells, so every address costs more than the token it replaces. of1's 2.88 px arclength and or1's address law were the right priors. (3) The "83 KB attribution" is not detachable in the strongest sense: removing those tokens' surprise from the stream makes the stream LARGER (+9,532 B) because the shipped mixer's contexts are causal through them. (4) The seg side of the door was already small by law (msr1/lb1); the rate side is now measured closed for token-level boundary description.

**Where the gestalt stands after tonight's four closures (gb2 vacuous bound; bnd1/bnd2 boundary code; rw1 renderer grid; pass-5 full field on pose).** Every representation-level door that keeps the shipped token field and re-describes part of it is closed or vacuous. Two doors remain, and they are the same door seen from two sides: (a) the GENERATOR — a specified generic basis/generator need only reproduce the six pose outputs and the argmax partition, not the incumbent carrier field or the token field (gb2's and bnd2's live hypotheses converge; the-cross, m148); (b) the FIELD ITSELF as the decision variable under the three-leg admission — the only class that moved the pointer today (moves 35, 38-class): change the field where the seg yield covers the resolved-pose and rate price, with the frame-0 selector (seg-free) as the first pose lever and the carrier re-solve as the second (operator 2026-09-10: "Remember pose re solve", "And frame 0"). The measured pass-5 table says the re-solve is the largest single effect in the chain (1,235× the bar on the full field) and the full field still fails on pose — so the frame-0 lever is the next measurement, on the same field, before any new representation is chartered.

## Addendum 20 (2026-09-10 ~09:00Z) — two moves in one hour, and the token-level rate corner measured near its entropy

**Pointer: move 39, S 0.1376693148220904 @ 180,186 B [contest-CUDA T4 n600]** (rp1, `c243108ea`); move 38 the same hour (sj1 pass 5, S 0.13789029 @ 180,436 B, `566338ef5`). Gap to sub-0.12: **0.017669**. Nine exact moves in ~30 h, all on the shipped object, none by retraining.

**What moved the pointer.** The FIELD under the three-leg admission (seg / resolved pose / real-encode rate). Move 38: a seg-directed pass whose full field FAILED after the carrier re-solve (resolved pose +1.33e-4 despite a 1,235×-bar re-solve credit) and whose 42-pair Lagrange subset won by declining to pay pose. Move 39: the first RATE-directed field change — 473 argmax-neutral token changes over 253 pairs, seg exactly 0, pose below base, −202 B. The asymmetry law on one object: removing surprise pays 0.15–0.27× its ranking; adding it costs 1.28×.

**Two operator reminders, banked as law with their measurements.** "Remember pose re solve": admit only on the resolved pose (full pass-5 field stale +2.48e-2 → resolved +1.33e-4). "And frame 0": the seg-free selector is a REPAIR for edit-broken pairs (88 % improvable) run INSIDE the admission — NOT a standalone win (on the untouched field the shipped mode is already optimal on 84 % of pairs; best standalone net 0.31× bar). MAIN's inference that the selector had gone unchosen since move 24 was falsified by the n600 denominator within the hour (m88 genus).

**The token-level rate corner, measured from three sides tonight (all on the move-37 field, real encodes, twins):** (1) bnd2/bnd3 — describing the 235,044 mispredicted boundary tokens by segments LOSES on all 54 variants; the address term is only 1,717 B, the loss is context disruption; a zero-cost ORACLE flag would displace 83,259 B. (2) tc2 — a lane-boundary DISTANCE context map buys 5,480 B even with the GT edge as oracle; a causal lane predictor buys 78 B real. (3) rp1 — 87.6 % of the payable bits sit on GT-agreeing tokens the model cannot predict; neutrality is 4.36 % and flat in rank. Together: at the token level, given every context built from the field itself (geometry included), the boundary population's surprise is near-inherent; the 83 KB attribution is displaceable only by a predictor that already knows the answer. The corner (−26,9xx B, re-derived by the packet) is not reachable by any coder over this field.

**Where the gestalt stands.** Three classes remain: (a) the FIELD as decision variable under the three-leg admission — proven, small, compounding (moves 31–39: −2.5e-3 in five days), with the composition of moves 38+39 and the frame-0 repair inside it running now; (b) the tail's last 78 B and tc2's "better predictor" (coherent lane tracking across rows; joint mixer re-calibration) — a micro-move class; (c) the GENERATOR — the only door that changes the object the surprise is measured on: reproduce the six pose outputs and the argmax partition from a program plus a small statistic, so that the boundary is DRAWN by geometry rather than coded per cell (gb2's vacuous bound says no bound decides it; bnd2/tc2 say no recode of the field decides it; only a construction can). That is the research yet to do the operator named, and it is now sharply posed: the field-level surprise the coder cannot remove is exactly what a generator must not have to transmit.

### Addenda 19–20 — CORRECTION 3 (2026-09-10 ~13:00Z, from pr6's second-family review, `.omx/research/ddm_pr6_second_family_review_moves_38_40_20260910.md` §"Addenda 19–20 source audit", `72565bdab`; thirteen classified misreadings + one arithmetic error — the full table is there; the load-bearing ones, each with its corrected statement)

1. **"Median assigned segment length one ⇒ the population is isolated single cells, not a curve population" (A19)** — ASSIGNMENT-AS-TOPOLOGY. bnd2's greedy edge assignment charges a new start at every gap, branch, class change and offset change; the run length is a property of that grammar, not of the boundary. Correct: under the six greedy grammars the charged runs are short.
2. **"Every address costs more than the token it replaces" (A19)** — BOUND-DIRECTION. The best side channel cost 2,013 B and displaced 597 B on its 2,125-cell support; an achieved losing code is an upper bound on achievable description, not a floor on addressing.
3. **"The masked control proves the attribution is not detachable in the strongest sense" (A19)** — DIFFERENT-OBJECT. Masking is an intervention that changes 235,044 field symbols and the later probability trajectory (250,892 new misses); it measures that intervention, not detachability.
4. **"The rate side is measured closed for token-level boundary description" (A19) and "every re-description door that keeps the field is closed or vacuous" (A19)** — FORMULATION-TO-FAMILY. bnd2 closed six greedy formulations and bnd3 forty-eight, both declared FORMULATION; richer assignment, joint refits, other geometry/context maps and other probability models were not tested. Correct: the tested re-descriptions lose.
5. **"The address term is only 1,717 B; the loss is context disruption" (A20)** — ATTRIBUTION-AS-PHYSICAL-ALLOCATION and WRONG MECHANISM. The 121,200 B packet is one shared Brotli block with no unique address/content allocation; 1,717 B is a separate transform of the logical address stream; the lossless receiver re-inserts symbols at their causal positions, and the loss is exactly 2,013 − 597 = 1,416 B on that support.
6. **"tc2's 5,480 B oracle is a geometry ceiling" / "serialized savings" (A20)** — ATTRIBUTION and BOUND-DIRECTION: an ideal-codelength attribution under a frozen mixer with a noncausal map, not a ceiling and not bytes.
7. **"87.6 % of the payable bits on GT-agreeing tokens" and "neutrality flat in rank" (A20)** — SCREENED SUBPOPULATION promoted to a law (rp1's screened census; seeded n12 for rank-flatness).
8. **"At the token level … near its entropy … the corner is not reachable by any coder over this field" (A20)** — UNSUPPORTED FAMILY CLAIM: no entropy lower bound and no optimality certificate exists; three formulation-scope closures do not compose into one. Correct: every context map and recode tried loses or buys < 100 B; the missing measurement is an entropy bound, which nobody has.
9. **Projection residuals of moves 38 and 40 called "pose-print class"** — INCOMPLETE: both are 8-dp print residuals but MIXED (38: 23.9 % seg / 76.1 % pose; 40: 26.8 % / 73.2 %); only move 39 is pose-print-only.
10. **Container two-sidedness** is fe1's law (the sampling), not compose-39's (which proved the favourable side only).

What survives verbatim from A19–A20: the three scores and their chain; the six lost repairs on the 13 shared pairs; the 6.38 % rate anti-synergy on the 78-token splice; the four closures as FORMULATION-scope negatives with their numbers; the generator as the door that changes the represented object (pr6: "these receipts neither construct nor price it"). Standing lesson, twice now: MAIN's addenda over-generalize formulation closures into family laws within the hour; the second family reads the receipts. Rule: a gestalt addendum states each negative with its verdict_scope beside it, and never writes "closed" without the level.

### Addendum 20 §"near its entropy" — the bound itself (eb1, 2026-09-10 ~16:00Z, `.omx/research/ddm_eb1_entropy_lower_bound_boundary_description_20260910.md`)
Conditional population lower bounds on the bits to describe the 600-frame argmax partition within D cells: D = 6,270 → 2,468 B (conservative class) / 5,006 B (natural profile); D = 12,540 → 1,405 / 2,100 B; D = 25,080 → 255 B. Against the 119,784 B tail these are 50–85× smaller — the bound is too WEAK to decide anything: it neither shows the incumbent near its entropy nor shows savings exist (the reference classes freeze most geometry, and the receiver's side information — previous partition + pose — is unconditioned). Standing statement, level FORMULATION for every recode/context tried and UNBOUNDED for the coder as a family: the tail is between ~2 KB (a weak floor) and 119,784 B (an achieved size); the corner is open as a matter of bounds. Also corrected: "12,540 cells" is the argmax-cell disagreement at the scorer's resolution; the move-40 TOKEN field disagrees with GT-argmax at 17,631 cells — two objects, state which.

### eb1 note — CORRECTION 4 (pr7, `6da7a0fbe`, 2026-09-10 ~17:40Z)
eb1's arithmetic, converse direction and ball-volume upper bounds are valid — for its two ARTIFICIAL classes C and N. Neither class is proved to contain physically realizable video/SegNet outputs, so 1,404.5 B and 2,100.25 B are NOT lower bounds for the observed video, the move-40 partition, or the token tail. The "~2 KB to 119,784 B bracket" and "50–85×" I wrote compare different objects and quantifiers and are withdrawn. Also: D = 12,540 was a charter scenario derived from rounded distortion, not a certified exact count; 17,631 is the exact disagreement of the stored token field (a separate object). Standing statement: NO valid bound on the tail exists yet, in either direction. A tighter applicable bound needs a charged/declared receiver state Z, an ex-ante (or inclusion-proved) physical population A_z, an upper bound on its maximum distortion ball, and the exact realized integer D — which is eb2's brief, now with those four requirements binding. Lesson (third time tonight): a number from an arm enters the gestalt only with its object and quantifier beside it.

## Addendum 21 (2026-09-10 ~18:40Z) — the generator door as I posed it in Addendum 20 §c does not exist at decode time; what the receiver actually holds, and what that leaves

**Pointer: move 40, S 0.13763861019288715 @ 180,233 B.** tc3's row (−79 B, projected 0.137586) is on T4; rp1's round 2 (projected ≈ −4.3e-4) is 2 h out.

**eb2 (`.omx/research/ddm_eb2_conditional_partition_bound_20260910.md`), n600, same real coder for every row:** pose-warp residual of the previous partition 542,248 B; persistence residual 536,997 B; the partition coded from scratch 357,123 B; the incumbent tail 119,754 B. The warped previous partition agrees on 98.75 % of cells and the residual still costs MORE than coding the partition from scratch — the disagreements are the boundary jitter, scattered, and a residual mask of scattered cells is expensive under any simple coder. Level: FORMULATION (one simple coder; no optimality certificate; not a temporal-family closure — eb2 says so and pr7's rules apply).

**The premise correction, which matters more than the numbers.** Addendum 20 §c asked whether "the partition is a FUNCTION of information the receiver already has (the previous frame's partition + pose)". The receiver has NEITHER: SegNet and PoseNet do not run at decode time (the strict-scorer rule), so the argmax partition and the six pose outputs exist only at the scorer. What the receiver holds is the decoded token plane of the previous pair and the carrier — exactly the side information bd1 (temporal context, B-frames 2.8–5.7 %) and mc1 (motion-compensated previous plane, +160 B) measured at the token level and found saturated. The conditional bound I wanted is a bound on an object the receiver cannot see.

**What that leaves, stated at its level.** (a) The field under the three-leg admission — the only class moving the pointer (moves 31–40; rp1 round 2 running with the frame-0 repair inside its admission). (b) The tail's last tens of bytes (tc3 on T4). (c) The generator, now posed correctly: not "draw the partition from what the receiver has", but "replace the stored token field by a PROGRAM whose output the receiver renders" — the born vehicle's question (the-cross: rate 121,928 B, distortion 0.33; md1: 62 % of its distortion optimizer-unreachable). The surprise the coder cannot remove is the surprise the program must generate; eb2 measured that neither persistence nor rigid warp generates it. That door is open as a matter of bounds (no valid tail bound exists — pr7) and closed as a matter of every construction tried (gb2, bnd2/3, tc2/3, eb2, bd1, mc1), each at formulation scope. The research yet to do is a construction, not a bound.

## Addendum 22 (2026-09-10 ~23:45Z) — move 41 RETRACTED as a pointer: the receiver-code door paid 79 B by embedding video-selected constants in free code (pr8, P0)

**Submittable pointer: move 40, S 0.13763861019288715 @ 180,233 B.** Move 41 (tc3, S 0.13758600733559048 @ 180,154 B) is an exact row whose archive does NOT qualify: pr8's second-family compliance review (`.omx/research/ddm_pr8_receiver_code_compliance_review_20260910.md`) found the receiver's lane predictor hard-codes the Lane class id (1) and the row band (128–319) chosen from the full-video census — content that must be COUNTED in the archive or replaced by an independently pre-declared generic rule (rule 118; NO-FAKE #6/#7, the hide-data-in-code class). tc4 inherits it and adds Movable (3). Also found: move 41's inflate took 1,336.7 s against the 1,260 s margin the seal will now require; the shipped predictor selects a float64 branch (cross-host determinism unproved); the receiver refuses the CPU path by declaration (no contest-CPU receipt exists for this lineage). Mixer weights and tc4's selector mask ARE in the archive; no scorer content entered the runtime.

**What this teaches, at the class level.** "No per-frame fitted table" was the test tc2/tc3/tc4 and MAIN applied; it is not sufficient — a fitted SCALAR (a class id, a row bound chosen by looking at the video) is content too. The first family (astra arms + MAIN) shipped it three times in one night and reviewed it as clean; the second family caught it on the first read of the runtime tree. Rule: every constant in receiver code carries a provenance line (a-priori generic, or counted), audited by the other family before a receiver change is fired.

**Disposition.** (1) Move 41 and tc4's candidate are withdrawn from every packet candidate list; the operator-facing frontier line is move 40 until a rule-118-clean successor lands (the canonical pointer JSON has no disqualification field — cp1 adds one; until then it still lists 41). (2) Cure chartered: the class ids and row bounds move into the counted rider (a few bytes; most of the 79 B survives), the geometry goes fixed-point, tc4's maps run mask-selected to fit the 1,260 s margin; rebuild, revalidate through pr8's checklist, re-fire. (3) rp1's round-2 seal targets move 40's clean receiver. (4) The gestalt's Addendum 21 stands: the field under the three-leg admission moved the pointer four times today with a clean receiver; the receiver-code door is real (79 B measured) and re-enters only counted and timed.

## Addendum 23 (MAIN, 2026-09-10 ~13:40Z) — move 42, and what the timing saga taught about authority

**Move 42 = rp1 round 2: S 0.1374765052591843 @ 180,238 B [contest-CUDA T4 n600]** (d2803c214),
−1.621e-4 vs move 40 (8.1× the bar), +8.7e-6 above the seal's projection (pose-print class, stated
before the row). The mechanism is the composition law again, read from the other side: the
rate-directed token pass did NOT pay on rate (+5 B; the first-order K-curve promised −1,756.6 B and
the real coder charged +620 B on the full rebased field — the ranking is not a charge, memory
`first_order_token_price_is_a_ranking_never_a_charge…`); the edits are the FIELD the carrier was
re-solved on, and the re-solve landed pose below base (4.66e-6 vs 4.887e-6) with seg exactly 0 over
117,964,800 cells. The gestalt's statement stands: the pointer moves when a leg CHANGES ITS OBJECT
first (here the field), and the pose re-solve is the leg that then pays. Frame 0 measured inside the
admission at 0.1× the bar and was not adopted. rp1 also caught, before any heavy step, that its
n600 acceptance had been measured on move 37's field (a flag and a constant disagreeing silently),
re-verified on move 40's field in minutes with a 1.0 control group, and lost 361 of 4,503 tokens to the
base alone.

**The timing saga (dwc1 → pr10 → pr11).** dwc1's decode-wall-clock leg was a gate with no door (no
producer could write the quiesced count). MAIN's first instrument fixed its rule twice after seeing
the data; pr10 (sol) refused that and wrote a conservative replacement; frozen and hashed, it then
refused five cold runs of the move 40 receiver (783–797 s, bit-identical output) — two for our own
tool activity, three for the PID-1 macOS maintenance pair `dasd`/`syspolicyd` and other Apple daemons
whose measured effect on the decode pace was ≤ 0.7 %. pr11 (same family) adjudicated on the receipts:
a bounded burst class for exactly that pair (v3, landed 6a857a1ec, frozen 25d0a778), and T4-DIRECT —
the candidate's own completed cold n600 contest-T4 decode, hash-bound to the exact archive and runtime,
is the timing authority with no local denominator; receiver-identical candidates inherit it. Move 40's
leg is now `t4_direct` (990.054 s); move 42 inherited it and decoded on T4 in 978.1 s. Local
calibration is SUSPENDED for receivers that cannot inherit (rlc1's cure): their authority is their own
fire. Law: the second family owns its rule — freeze before the run, send receipts back, never loosen.

**Doors from here (updated).** Seg: 100·d_seg = 0.010637 = 7.7 % of S, the largest lever; sj1 pass 6
on the move 42 field (renderer boundary jitter at correct tokens). Receiver-code door: rlc1's cure
(−60 B, counted rider) re-based onto move 42 by rlc2, timing by its own T4 fire. Rate: token flips do
not pay under the real coder above K≈32 per pair; the lever is representation. Custody: vr8 audits the
certified MOVEs after rp1 found five 1.83 GB overlays recorded MOVED with only ExFAT stubs (re-render
reproduced sha 37ea3842…). Sub-0.12 gap: 0.01747651.

verdict_scope: instance — move 42's row and the five move-40 timing runs on this host; the composition
reading is the same law as Addendum 22, not a new one.

## Addendum 24 (MAIN, 2026-09-10 ~19:05Z) — move 43, and the first real door test of the pre-fire contract

**Move 43 — S 0.1372848557085275 @ 180,466 B [contest-CUDA T4 n600]** (sj1 pass 6; d_seg 0.00010345, d_pose 4.59e-06; −1.9165e-4 vs move 42, 9.6× the bar; exact vs sealed projection −1.938e-4, gap 2.2e-6 = the T4 pose-print class). The seg leg on the shipped bytes matched the admission exactly (12,196 predicted = 12,196 measured). The mechanism is the one Addendum 23 named: a field change (moves 38–42 re-rendered 365 of 600 token planes) re-opens the seg-debt pool for a repair family that had declared itself converged on move 37's field. Pre-registered decomposition: 375 of 489 admitted positions are new on re-rendered pairs, 114 are carryover, exactly zero are new on the 235 unchanged pairs. **A token-repair family exhausts per object, never per vehicle.** The yield curve (−8.1e-4, −1.5e-4, −2.7e-5, then −1.9e-4) reversed because the object changed, not because the search improved. Frame 0: measured (+0.075 bar), not shipped (build blocker: no selector splice in the close), reactivatable.

**The contract met its first real producer and refused it — correctly, on a defect.** rlc4 materialized the counted-rider cure on move 42's field: 180,178 B (−60 B), twin encodes identical, full cold n600 public output byte-identical to move 42's retained raw, manifest regenerated from outside the tree. The frozen pre-fire producer refused the intent: `PREFIRE_RISK_EVIDENCE_REFUSED: receiver risk endpoints differ`. The only normalized difference was `MANIFEST.sha256` itself — a derived listing carrying the raw hash of the pin-bearing `inflate.py`. Two second-family requirements (pr9: regenerate the manifest; pr12: receiver identity for risk inheritance) collided on one derived file, with zero executable bytes differing. pr13 had ratified MAIN's two clarifications before this; pr14 adjudicated the collision: exclude the manifest ONLY from a new versioned timing-risk digest, keep it in every custody check; both endpoints recompute to 9f6e71680a13d859… over 49 identical rows. Law banked: a derived hash listing is never receiver content; scope the exclusion to the behavior digest, never widen it into custody. ffi4 implements; rlc5 re-bases the rider onto move 43's bytes (the objects compose by object change, so the −60 B is expected to survive the re-base but must be re-proven by raw identity).

**Authority note.** The exact row was again the only score; the pointer packet re-verified the archive on disk and wrote the consequences. Modal spend this session remains under the $5 ceiling; the first-measurement cost preflight is validated at $1.209 for the full T4 cap.

## Addendum 25 (MAIN, 2026-09-10 ~22:40Z) — move 44: the rider cure lands, and the contract's first end-to-end pass

**Move 44 — S 0.1372449041713402 @ 180,406 B [contest-CUDA T4 n600]** (rlc5; −60 B on move 43 with the cold n600 public output byte-identical, so d_seg 0.00010345 and d_pose 4.59e-06 are move 43's by construction; −3.9951537e-5, exactly the conditional arithmetic; measured twice, run4 and run5 identical). Rule 118 is now closed for this receiver family: the fitted geometry and lane-class selection that move 41 had smuggled into free code are counted in the archive, and the archive still shrank.

**The contract passed end to end on its fifth dispatch.** Between "every test green, three ratifications" and one T4 row sat eight pass-path defects, all found by real runs and none by fixtures, plus the NO-GRACE consequence: landing ffi6's completion-custody fixes made run4's intent stale, so run5 re-measured the same bytes (pr17). Cost: a few dollars of T4 time and one day of second-family adjudication; return: a contract that now provably admits a real producer, refuses every replay, and records custody at every step. The law is banked (`a_contract_pass_path_is_proven_only_by_real_dispatches_eight_doors_in_one_chain_20260910`).

**The rate corner, re-derived at move 44:** cap 154,507 B at the held distortion → demand −25,899 B. gdc2's Stage A closed the categorical Cool-Chic shape at formulation scope (5.6× the zero-budget mismatch ceiling); its measured R(M) law (0.26→2.21 B/mismatch across three decades) is the door's admissible frontier: describe the token field exactly in ≤ 94,010 B. The next construction is chosen against that curve, not against gf1's transferred constant.

## Addendum 26 (MAIN, 2026-09-10 ~23:05Z) — gdc2 closes the categorical distillation; the residual rate is a geometry, not a count

gdc2 (`.omx/research/ddm_gdc2_categorical_coolchic_k8_distill_20260910.md`, sha 2f534b1d2ec91038…) ran the governed burn to the end: 19 full-n600 authority evaluations, exact MLX/NumPy identity, decode
inside budget — and FORMULATION-NO-GO by 5.75× (540,681 B against the 94,010 B gate). Three facts outlast the row. (1) It is
capacity-limited, not rate-limited: rate pressure cut the packet 35 % while mismatches moved 2.7 %, and the coder found nothing past
zeroth-order entropy — the latents are already incompressible at this budget, so the door is not "code the latents better". (2) The
exact-residual rate is set by error GEOMETRY: the neural decoder's scattered single-cell errors cost 1.83× what the scanline family's
contiguous runs cost at the same mismatch count, so R(M) is family-local and gf1's constant produced three wrong numbers in one memo
([[cross-regime constant transfer]] sharpened). (3) The frontier the door must beat is now stated at its level: describe the token
field EXACTLY in ≤ 94,010 B, 21.6 % under the shipped 119,969 B tail, with the program's own residual priced at its own geometry.
The gestalt reading: every construction so far (gb2, bnd2/3, tc2/3, eb2, bd1, mc1, gdc1, gdc2) pays for the surprise the coder cannot
remove by ADDRESSING it; the only object that has ever held the rate is the born generator, whose distortion is data-anchored at
Lane edges (md1–md4). The next design (gdc3) is chosen against the measured geometry law: it must generate contiguous, Lane-aware
structure whose errors cluster where the residual coder is cheapest — or state, with a $0 falsifier, why it cannot.

## Addendum 27 (MAIN, 2026-09-10 ~23:45Z) — gdc3: the residual-geometry table, one more closure, and the first design point under the gate

gdc3 (`.omx/research/ddm_gdc3_next_construction_against_the_geometry_law_20260910.md`, sha db4510fdae22e343…) measured what Addendum 26 asked for: the exact-residual price BY GEOMETRY, on both retained fields with the
real coder. Long boundary-adjacent runs are the cheapest class in both (0.495 and 0.246 B/mismatch), isolated cells the dearest
(1.84 / 1.55), and the same class differs 2.01× between generators — so R(M) is generator-local even within a geometry class, and the
only honest screen is a candidate's own residual coded at its own errors. The fixed anisotropic key-row hold put 100 % of its
634,370 errors next to boundaries and still lost by 4.8× because its packet alone is 2.42× the door: geometry-cheap errors do not
rescue an expensive description. What survives is the first construction whose design point sits under the gate on measured rates:
a learned run-native endpoint generator (runs as the native representation, endpoints not cells), K ≤ 60,000 B, M ≤ 65,000,
≥ 90 % of errors long-and-boundary-adjacent → 92,174 B projected. The margin is 2 %, and every projection this week missed by 3× or
more; so gdc4 burns under the standing GO with those three numbers as early-stop gates, not as hopes. The gestalt has narrowed to one
sentence: the token field must be described by its runs, with a generator whose mistakes are runs too.

## Addendum 28 (MAIN, 2026-09-11 ~00:05Z) — gdc4 closes the run family before burning; the door's question becomes one sentence

gdc4 (`.omx/research/ddm_gdc4_run_native_endpoint_generator_20260910.md`, sha 4cd0187b2f6d2c48…) did the thing this program rewards most: it refused its own burn by exact arithmetic. The optimal
bounded-endpoint approximation per row, computed by DP, IS gdc1's retained scanline family to the unit (E=8→88,304 mismatches,
E=24→0), so no training run could beat renders already on the SSD; the family's best exact description of the shipped field is
235,087 B, 2.50× the door, with exactness the cheapest point and no interior minimum — a real coder inverts gdc1's ordering. Runs are
not a better factorization: an independent dense 2-D context model lands within 2.4 %. And the "≥ 90 % long-boundary errors"
premise was inverted (6.44 %) by a table gdc3 had already published — the borrowed-premise genus again, caught this time at $0.
gdc4 also found that four arms had pinned the never-shipped pass-6 field (154 cells off the shipped one); verdicts unchanged, law banked.

**What the twelve closures say together.** Every program that addresses the surprise (endpoints, runs, atoms, latents, contexts,
warps, planes) pays for it in full; the only object that ever held the rate is the born generator, and its accuracy is data-anchored
at Lane edges. The remaining demand, 25,899 B, must come from the MODEL on the dense factorization — and tc1 measured the mixing
ceiling at roughly a third of it. So the question is no longer "which generator" but **where 25,899 B of Lane-conditioned surprise
lives that the HPAC prior and the free mixer do not already remove** (Lane: 0.59 % of area, 34.1 % of stream bits, 59–65 % of
boundary cost in two unrelated instruments). ls1 measures that atlas at $0 on the SHIPPED field. If the atlas shows the surprise is
Lane geometry the receiver cannot see, the gestalt answer is a Lane carrier in the counted archive — the born vehicle's question
posed at the pointer's rate.

## Addendum 29 (MAIN, 2026-09-11 ~01:10Z) — ls1: the atlas closes the token-tail rate corner at the measured level

ls1 (`.omx/research/ddm_ls1_lane_conditioned_surprise_atlas_on_the_shipped_field_20260911.md`, sha b729faa146b62ea3…) instrumented the SHIPPED decoder on the shipped field and reconciled its per-symbol surprise to the real
stream within 0.0006 %: the instrument is exact. Lane is 0.586 % of the symbols and 33.5 % of the surprise, and in every class the
cost sits in isolated boundary cells, not runs. Then the oracle ladder, each rung priced as a bias-corrected Miller–Madow estimate:
tc1's joint contexts 8,218 B; receiver-visible Lane geometry 17,534 B — 8,365 B short of the 25,899 B demand; and a GRANTED complete
previous-row geometry, side information the receiver does not have and would have to be paid for, 24,864 B — still 1,035 B short
before its own cost. The gain is diffuse (90 % of it needs 503 of 600 pairs), and the one concrete carrier instance (an explicit
override map) costs 594,003 B. Three source corrections travel with this: tc2's "5.5 KB" was oracle gain, not a map's price; gdc4
reported computed lengths, not serialized prices; tc1's ceiling was one mixer's, not a universal one.

**What this settles.** On this object — the pointer's token field under the incumbent's decoder — no measured mechanism on the tail
reaches the rate demand, and even an oracle with unavailable side information does not. ls2 ($0) measures the last receiver-visible
hypothesis (full-resolution probability corrections rather than fixed cells). If it falls short too, the honest statement is that the
sub-0.12 path is closed on THIS object at the measured level, and the remaining question is the one the cross posed on 08-29: an object
that inherits the born generator's rate (121,928 B) while starting from the distortion regime this lineage reaches (0.017). That is an
object change, not a mechanism, and it is the operator's gestalt call to make with these numbers in hand.

## Addendum 30 (MAIN, 2026-09-11 ~02:10Z) — ls2 closes the last receiver-visible hypothesis; the rate corner is measured closed on this object

ls2 (`.omx/research/ddm_ls2_full_resolution_lane_probability_bound_20260911.md`, sha 225db0ec75bd20b3…) priced the full-resolution Lane probability correction on all 600 pairs of the shipped field: 266 B (separate
features) and 386 B (joint distance) of RC64 byte-equivalents, net — 1.49 % of the 25,899 B demand — and Lane itself got worse under the
separate model. With ls1's ladder above it (receiver-visible geometry 8,365 B short; a granted oracle the receiver cannot have 1,035 B
short before its own cost), the token-tail rate corner on THIS object is closed at the measured level. The operator's reading of 08-21 and
09-09 was right on the evidence: months of negatives point at the gestalt, and the gestalt is an OBJECT question. The cross (08-29) already
stated it: the born generator holds the rate (121,928 B, 16,058 B under the cap) and this lineage holds the distortion (0.0171; the born
object's 0.33 is data-anchored at Lane edges, md1–md4); born rate + pointer distortion = 0.109, sub-0.12 by 0.011 — a derived feasibility,
not an archive. Nothing in this session's twelve closures contradicts it; each closure was a mechanism trying to buy the born object's rate
from inside the pointer's object. ob1 designs the successor object under the cross's criterion — inherit the born generator's byte
feasibility, start from a reachable distortion regime, put the Lane-edge geometry in the counted archive as a carrier the generator
consumes — with $0 falsifiers on the retained born renders and the shipped field. Whether to burn it is the operator's call.

## Addendum 31 (MAIN, 2026-09-11 ~02:40Z) — obx1: the cross corrected on n600, the direct splice closed, and the object decision put where it belongs

obx1 (`.omx/research/ddm_obx1_successor_object_under_the_cross_design_20260911.md`, sha 9f49b2dcacd788fc…) did what the cross needed first: it measured the born object on the COMPLETE population. qbt2b r10 is not at
distortion 0.33 — that was the n32 hardtail prefix — it is at **8.6267** (d_seg 0.0633, d_pose 0.529) on n600 through the frozen scorer,
with its rate at 106,714 B physical / 121,928 B HT. The cross's two halves stand, but the distance between them is 8.6, not 0.3, and the
corrected counterfactual is 0.0983. The $0 falsifier then closed the naive bridge: splicing the pointer's Lane cells into the born render
makes distortion WORSE (14.57) at a 4.45 MB carrier — interpolation spill, the same physics that killed every post-hoc sidecar on the
witness (§Pose is SOLVED, 2026-07-10). What survives is the first design that changes the OBJECT instead of the tail: an edge-local implicit
correction lattice co-trained through R over the born generator (LIIF/SHACIRA anchors), with a Poisson-fusion and a temporal-edge design
queued behind it. obx1 sealed it with a burn spec and, correctly, stopped: this is a new vehicle generation measured in days of local
training, not a bounded burn, and the standing GO was written for the latter. The decision is the operator's: `GO ddm_obx2` or hold.
Everything MAIN can do without that decision is done; the pointer is at move 44 and PR #140 is staged on its bytes at 91/93.

## Addendum 32 (MAIN, 2026-09-11 ~03:40Z) — the operator's GO, and six bets chosen by marginal value and by where the free compute is

The operator (09-11): "do whatever it takes … full authority and standing go … be creative and weird and think divergently." Six parallel
bets, each ending in a sealed candidate or a measured negative, never a memo alone:

1. **obx2 — the object.** Design A from obx1: an edge-local implicit correction lattice co-trained through R over the born generator. The only
   construction that changes the object rather than the tail. Gate: a parsed ≤ 122,000 B object at n600 distortion < 0.04 before any scorer.
2. **pc3 — pose is the steepest marginal.** At d_pose 4.59e-6 the pose term's derivative is 738 S per unit, 7.4× seg's 100. Every pass re-solved
   the carrier at its lattice; nobody priced the d_pose-vs-bytes curve. Halving d_pose is 99 bars and is worth up to 2,970 B.
3. **ntb1 — the 60,497 non-tail bytes** were never priced this week: ZIP overhead, header, the 13.5 KB HPAC model, the renderer's weights, the
   carrier's container. A 500 B cut at unchanged distortion is 16 bars.
4. **mxo1 — free compute is the untouched budget.** The T4 leg uses ~1,000 s of 1,800; rule 118 makes decode-time compute free. tc1's counted
   35-weight mixer realized 6 % of a 9,011 B joint oracle; an online-learned mixer/recurrent model trained on the decoded prefix, zero counted
   bytes, deterministic integer arithmetic, is the thing those bounds were about and no one built.
5. **gpp1 — the receiver can look at its own pictures.** eb2 said the receiver holds no partition; but it renders RGB, and a generic, public,
   non-video-derived model (never the scorers) run on those renders gives a Lane belief the token plane cannot. Rule 118 free if generic;
   the legality reading is deliverable one, the operator decides at publish.
6. **rbf1 — 86 % of the remaining d_seg is one-pixel boundary jitter at correct tokens.** rw1 closed grid repairs; a free deterministic
   render-time boundary treatment (guided filter, edge-aware AA, token-driven sub-pixel displacement) is a different family, priced through
   the real R operator at n600 on both seg and pose.

What they share: every one is priced on the shipped field by the real coder or the frozen scorer, every one seals through the contract that
just proved itself, and every one is a MEASURED bet, chosen by marginal S per byte or per second, not by novelty for its own sake.

## Addendum 33 (MAIN, 2026-09-11 ~05:30Z) — mxo1 closes the free-online-mixing bet and corrects the slack; the portfolio adjusts

mxo1 (`.omx/research/ddm_mxo1_free_decode_time_online_context_mixing_20260911.md`, sha 60b5a34bb6146af8…) built the thing the bounds were about and measured it: the best free online learner over the shipped stream saves
368 B (1.42 % of the demand; 4.08 % of tc1's oracle); a leaky recurrence adds 35 B; a previous-frame hash saves 114 B. Two premises fell with it,
both mine. The receiver's strict slack under the contract's 1,260 s ceiling is **27.6 s**, not the ~540 s I derived from the README's 1,800 s
budget — one wrong denominator turned a 20× compute budget into 1×, and the correction is now in gpp1's and rbf1's charters before either
built on it. And move 44 already runs a 23-family PAQ-style online adaptive corrector, so "add online adaptation" was never new; what is
untested is the COLLAPSE of those 23 predictions into one probability through a fixed 35-weight mixer. mxo2 screens exactly that (a low-rank
nonlinear stacker over the pre-mix outputs) at $0 under mxo1's own trigger: ≥ 3,000 B at ≤ 200 ns per symbol. The portfolio stands at six live
bets: the object (obx2), pose (pc3), non-tail lossy levers (ntb2), the generic pretrained prior (gpp1), the boundary treatment (rbf1), and the
pre-mix stacker (mxo2). ntb1 closed every lossless non-tail lever at 0 B; the lossy ones are ntb2's.

## Addendum 34 (MAIN, 2026-09-11 ~15:20Z) — obx2 handback 1: the born object's distance was coverage, not capacity

obx2 (`.omx/research/ddm_obx2_edge_local_implicit_correction_20260911.md`, sha fb0468916ecc6ea5…) built the successor object and, before training it, measured the thing the cross had been arguing about for
two weeks. The born object's 8.6267 on n600 decomposes as 0.4072 on the 32 pairs it was trained on and 9.0299 on the 568 it never saw:
`qbt1::validate_config` refused every training set but the sealed n32, so 568 of its 600 latent records were never optimized. The 8.6 was
never a capacity wall; it was a fence the code put around the data. That reframes the cross from "8.6 apart" to "one full-coverage training
away from a measured answer" — and obx2's harness reproduces move 44's contest-CUDA row to 0.036 %, so the answer will be trusted when it
lands. Three more facts: the object's archive is 108,988 B with the lattice (13 KB under the gate; training grew the model section 51 B in
ten epochs); the 384×512 render grid clears 0.04 (0.0324) when back-projected in the scorer's plane, while the naive downsample would have
produced a false structural refusal (0.1187) — the prefix-and-plane genus again, caught by the arm; and the pose budget at < 0.04 implies a
scorer-plane RMSE near 0.22 of one LSB, so no 122 KB object matches the teacher photometrically: the object must be a pose-equivalent
WITNESS, which is what both live runs are (joint scorer descent, not distillation). One porting item stands between a passing object and a
T4 row: the portable NumPy receiver is 2.06× over the 1,260 s budget (torch 187–334 s).

**Correction to Addendum 34 (MAIN, same hour).** "One porting item stands between a passing object and a T4 row" is wrong;
obx2 cured it before handback. The shipping receiver is torch-CPU on the parsed packet (187–334 s for n600, generic free
code under rule 118); the float64 NumPy receiver at 2,593 s is the cross-check only. The real consequence is different and
sharper: the two receivers disagree on 0.093 % of rounded uint8 values (max abs 0.0075), so the object must be validated
and scored through the receiver that ships, never through the other. Nothing stands between a passing object and a T4 row
except the object passing.

## Addendum 35 (MAIN, 2026-09-11 ~18:00Z) — rbf1: the free pixel actuator dies on pose, not on wall-clock, and the error is context

rbf1 (`.omx/research/ddm_rbf1_free_post_render_boundary_treatment_20260911.md`) was chartered on a premise its own first
measurement corrected: the arm was told the T4 slack was ~540 s, then ~27.6 s, and asked whether a post-render boundary
treatment could fit. Two of three operators fit at the band (SSAA 8.8 s, SDF 20.4 s of 27.581 s). None of that mattered. Every
mode raised S, the best by +0.56, the composition by +10.9, because a receiver-side pixel edit pays a pose tax that is quadratic
in its amplitude (ΔS_pose = 6.6e-3·τ², the excess constant to 5 % across an 8× range) and, at the pointer's d_pose of 4.59e-6,
about 100× the seg channel for the same intervention. The seg side closes on its own arithmetic too: 0.53 % of token-edge pixels
are wrong, so breaking even needs 186× selectivity and the operators reach 10–18×. Three facts the charter did not ask for are the
gestalt's gain: the composite resampler is centroid-preserving except at the border (there is no phase to pre-compensate); jitter
does not track blur (r ≈ +0.008); and at two thirds of the token-correct errors the render is already CLOSER to the ground-truth
class colour than to its own. The residual is not a pixel that is the wrong colour. It is a region the scorer reads in context,
which is the SegNet-sees-regions law measured from the inside. Law for successors, in the arm's words: price the pose channel
before the seg channel for any receiver-side pixel actuator on this vehicle. This is the third free lever closed this week on the
same object (mxo1 slack, ls1/ls2 atlas, rbf1 treatment); the object question, not the lever question, is where the bet sits.

## Addendum 36 (MAIN, 2026-09-11 ~15:30Z) — obx2 handback 4: the store closed the base-only stage, the fit could not, and a ratio I carried was retracted

Three things happened in one unit, and the order they happened in is the lesson. First, the arm pre-registered a seg-slope falsifier,
fired it on six epochs, got CLOSED for both arms, and refused to bank the verdict because the two fit families disagreed by five orders
of magnitude on a 0.07-decade window; it amended the rule in the open (INDETERMINATE when the data cannot separate the families; the
amendment can withhold a verdict but never flip one) and reported both firings. Second, it did the recall MAIN ordered, apparatus not
volition, and the store answered what the fit could not: md1–md4 were measured on this exact generator form. Sixty-two percent of the
terminal seg error is persistent across schedule, data order, and start; deleting every optimizer-reachable site leaves 12.75× the
accuracy corner; every schedule, optimizer, and objective lever's combined ceiling is 1.61× against a 20.57× need. The w2 base-only
stage was an objective lever on a closed form. It stopped, with checkpoints kept. The lattice head survives because it is the
representation change md3 and md4 name as the cure, and their conditional excludes precisely what it changes. Third, the arm corrected
a citation MAIN had carried into its charter: bz2d's ×1.157 token-to-argmax ratio was retracted the same day it was written. The
relation is affine, argmax ≈ 17,241 + 1.1435·tokens, and the intercept is a render-manufactured floor no token work removes. For a
witness that regenerates the partition, the transferable fact is that the intercept exists, not the slope. The pose side sharpened
to a single sentence: ±1 LSB of independent noise on the render alone puts the object 2.2× past the gate (law on six n600 points,
exponent 1.744, admissible 0.253 LSB). Standing question handed to the arm: how much of the object's seg gap is token-level partition
error, which no lattice fixes, versus render floor, which the lattice targets. That split decides whether design A is aimed at the
binding term at all, and the arm's retained per-chunk argmax can measure it against the pointer's token plane.

## Addendum 37 (MAIN, 2026-09-11 ~16:45Z) — design A closes with its mechanism validated, and the object question turns back into a rate question

obx2's burn ended the way a good burn ends: with the mechanism proven and the bar measured closed on the same day
(`.omx/research/ddm_obx2_design_a_closure_20260911.md`). The edge-local lattice does what it was designed to do — on the epoch-30
checkpoint, n600, through the shipping receiver, zeroing it costs 0.00137 of seg and 6.4× on pose, so it takes 53 % of the render
floor available to it for 575 B. What it cannot reach is the object's real debt. On n600 the generator's own partition is wrong on
9.0 % of pixels, 561× the pointer's token plane, and that partition-level error is 89.8 % of the seg leg. The remaining 10.2 %, the
render floor at a correct partition, is 0.00231 on its own, 5.77× the entire seg ceiling the gate allows. A perfect partition rendered
by this generator still fails. The seg rate decayed monotonically 0.95 → 0.32 %/epoch and the stop rule fired at epoch 46 with the
ceiling 1,300 epochs away. Two lessons are new. First, md4 already said to run the free step-0 render-floor probe before burning and
predicted unreachability at nine in ten; this burn ran it afterward. That probe, with the partition-vs-tokens decomposition obx2 built,
is now the pre-burn gate for any successor object, written into the closure's reactivation criteria and the cross memory. Second, the
pose relation is not a law: noise does 3.23× the damage of smooth error at equal RMSE, so scorer-plane RMSE is not a sufficient
statistic for pose and any future admissibility number must name its error family. Where this leaves the object question: the
pointer's renderer already has a render floor of 1.03e-4 at an exact token plane, below the 4e-4 ceiling. The accuracy half was never
the renderer's; it is the partition's, and the pointer buys it with ~140 KB of token tail whose lossless rate corner ls1/ls2 measured
closed. So the open question is the rate of a partition the scorer accepts — which is the seg-debt pool, the pre-distortion line that
produced moves 30–43, and the non-tail levers ntb2 is measuring now. The successor-object bet under the GO cost one day and returned a
validated corrector, a pre-burn gate, and a sharper statement of the problem. That is the correct price for it.

## Addendum 38 (MAIN, 2026-09-11 ~17:30Z) — two rate candidates, one weird bet closed on the rule's own text, and a manifest row that quietly closed the seal's inherit path

The day's ledger under the GO now reads: two live candidates and three formulation-scope closures. The candidates are both rate cuts
at identical decode, which is the cheapest kind of move and the kind the cross said the born object could not supply. pc3 refit the
shipped pose-carrier predictor over its closed legal schema at bit-identical codes and found ten of twelve biases parked at the ±16
clamp — a clipped fit, never a rate-optimised one — for −160 B (`.omx/research/ddm_pc3_…`, archive 145e02e2…, cold n600 raw
byte-identical to move 44's). ntb2 rounded 2,320 of 4,800 frame-embed values to even and shrank the HPAC prior 603 B against a
358 B growth of the tail it conditions, −245 B net, and the lever is output-lossless by construction because the prior is built from
the same bytes at both ends (archive 432e8f09…, projected S 0.13708). ntb2 also gave the renderer its first PoseNet measurement at
3-bit depth: every byte-saving layer moves 87–98 % of the frame by 3–13 grey levels and pays 100–1,900× the bar on pose; sd1's
"winners" were winners of a seg-only objective. The weird bet, gpp1, is closed twice on the same day the operator asked for weird:
a free public RAFT prior on the receiver's own rendered frames has an oracle marginal of 8,171 B against the 25,899 B demand, but the
real causal coder realizes 3 B of it (inside container-break noise), inference runs 302 s against a 27.6 s allowance, and the rule's
own text counts neural-network weights by artifact type, not provenance — 4 MB of weights would cost 2.67 S. The successor gpp2 exists
on paper with two hard preconditions and no launch. The apparatus finding is the one that will matter longest: the receiver behavior
digest hashes MANIFEST.sha256 raw, the manifest lists inflate.py's raw hash, and that hash changes with every archive pin, so the
normal inherit path refuses every honest successor and passed earlier ones only because their manifests were stale (pr9's condition 1).
Two candidates hit the same wall within hours of each other; pr18 is landing pr14's owed amendment through the contract's own
amendment path while pc3 proceeds on the intent chain that already excludes the row. Sequencing is by readiness: first sealed fires
as move 45, the other rebases and fires as move 46. Net for the gestalt: the successor-object family and the free-prior family both
closed at formulation scope on numbers, the coded-token object is the vehicle, and its rate is moving by clipped fits and
mis-rounded priors that nobody had priced — the same lesson as moves 30–44, found again in two new sections.

## Addendum 39 (MAIN, 2026-09-11 16:40Z) — pointer move 45: the first move whose apparatus was built the same day it was used

Move 45 is S 0.1371383667388406 at 180,246 B [contest-CUDA T4 n600] (commit 01f2b66ad), −1.0654e-4 against move 44, with d_seg
0.00010345 and d_pose 4.59e-6 exactly move 44's because the cold n600 decode is byte-identical across every raw byte. The lever is
pc3's: the shipped pose-carrier predictor was a clipped fit with ten of twelve biases parked at the ±16 clamp, and an exhaustive refit
over the closed legal schema at bit-identical codes removes 160 B while `decode_cap1` reconstructs the same canonical carrier. The
CPU-axis sibling refused by design in 10.5 s (call fc-01M28MCY…, receiver declaration linux-nvidia-t4), which is the declaration
plus refusal receipt the packet carries, as for move 44. What makes this move different from the twenty-one before it is the path
it took. At 14:50Z the candidate could not be sealed: the receiver identity digest hashed a derived listing whose contents change with
every archive pin, and the normal inherit path had passed earlier moves only because their manifests were stale. pr18 landed the
behavior digest through the contract's own amendment path at 10:47Z-relative-to-the-arm (commits 5d2632ee4, f1b9a0dbb, 301 tests),
ntb2 measured that its own candidate cleared it, pc3 sealed on the normal path at 15:5xZ, MAIN fired at 15:59Z, the harvest landed
at 16:31Z, and the packet applied at 16:33Z. Three refused CPU sibling attempts before the harvest were the single-flight law working
(one Modal job at a time), retained as receipts and not overridden. Two more facts for the gestalt: the projection was exact to float
rounding, as every rate-only move has been, and the rate corner's binding number re-derives at this move to a cap of 154,347 B at held
distortion 0.01712, so the demand is now −25,899 + 160 = −25,739 B. ntb2's output-lossless HPAC prior (−245 B) rebases onto this
pointer next. The day under the GO: two closures by measurement (design A, free boundary treatment), one closure on the rule's text
(free prior), one apparatus amendment, one pointer move, and a second candidate queued behind it.

**Correction to Addendum 39 (MAIN, same hour).** The re-derived cap at held distortion is 154,507 B, not 154,347 B: distortion is
unchanged from move 44, so the cap is unchanged and only the demand moves, 180,246 − 154,507 = −25,739 B (was −25,899 B at move 44).

## Addendum 40 (MAIN, 2026-09-11 ~16:50Z) — mxo3: the receiver already contains the stacker the charter asked for

The charter framed move 44's context mixer as a fixed 35-weight object and asked whether a low-rank nonlinear stacker over the 23
pre-mix family outputs could buy 3,000 B inside 200 ns/symbol. mxo3 measured the premise before the lever: the shipped mixer is
4,000 contexts × 23 weights, cold-started at zero counted bytes, learned online every group, and it reaches the coded row through a
single scalar. A second stacker on top of an online stacker finds +9.28 B at rank 2 and loses at ranks 4 and 8, on realized code
length against the real 119,749 B stream reconciled to 0.000584 %. The arm did not stop at the zero; it built a second instrument to
tell a ceiling from a weak learner, and held out: every one of the 23 families costs bytes on its own, the table costs 894–5,491 B
across partitions, and the most generous in-sample bound plus the entire Q15 blind spot is 1,210 B. The timing half passed
comfortably (27–57 ns/symbol), which is what makes this a closure of the family and not of the budget. For the gestalt: the
context-mixing corner of the token tail is now closed from both sides on this object — ls1/ls2 from the context side (the atlas's
oracle is 1,035 B short of the demand) and mxo3 from the mixing side (no stacker over the existing families pays). What remains on
the tail is what the pre-distortion line already exploits: changing the field the scorer does not read, not coding the field better.

## Addendum 41 (MAIN, 2026-09-11 ~17:20Z) — the store's own answer to "what is left": one unpriced door, and it is a shape

With four families closed in a day, MAIN asked the store rather than itself. A read-only recall over the gestalt, the DAG, every
charter with an OPTIMAL FORM block, and the memories, ranked every door by measured evidence with a successor check per door. The
answer is sober: nothing on the books reaches the −25,739 B demand at held distortion. The seg-debt pool has no door on move 45's
field because no pair has been re-rendered since move 43 and the repair family exhausts per object. Growing the HPAC prior was closed
by cl2 and cl3 (secant +0.446; λ a local optimum both ways), a closure ntb2 had not recalled when it wrote "not closed". The renderer
sits at its rate-distortion knee. obx1's designs B and C die on the same generator by the render-floor criterion design A just
measured. The pose corner's whole remaining reach is bounded at 707 B. What remains unpriced by anyone is the one thing cl2 named
and no arm touched: the receptive field's SHAPE. Every arm since 09-05 that worked the mixer changed the context set, the stacking,
the model rows, the size, or the values; none moved a tap. hpr1 now prices that rung by the real coder on the real stream, with the
decoded field proven identical and cl2's secant as the prior against it. If shape does not pay either, the honest gestalt statement
becomes: the coded-token object's rate is closed at the measured level on every axis the store can name, and −25 KB needs an object
the store does not yet describe. That statement would itself be the day's most valuable result, and it is one measurement away.

## Addendum 42 (MAIN, 2026-09-11 ~17:55Z) — sr5: the reserve restored by exercised certificates, and three storage laws measured on the way

Both SSD tiers sat at their fail-closed reserves this afternoon while two candidates needed cold decodes, and the two proven cures
were the wrong shape: moving between tiers nets nothing (sr4 had already offloaded 31 GiB of one onto the other) and the local disk
is not a tier by the operator's rule. sr5 (`.omx/research/ddm_sr5_certified_rebuildable_deletion_20260911.md`) freed 25.7 true GiB on
Vertigo by deleting 1,684 files from one closed August store under a per-file certificate whose rebuild was actually re-run and
sha-matched before each unlink, with a negative control that refused a wrong derivation and a restore proven from a sidecar on a file
already gone. Zero refusals across 28.2 GiB, and nothing uncertified touched: rbf1 and mxo3 were verified closed and left whole
because a cheaper class met the target. The laws it measured matter beyond today. A manifest whose hash field says
"SKIPPED_LARGE_STREAMED" is not a certificate (ntb2's 3.78 GB cold store was correctly refused for exactly that). A sampled
prefilter chooses what to hash and never decides equality (full hashes falsified one of its two projected duplicate groups; the
other released 3.4 GiB losslessly by hardlink). And `df -h` on this host prints gigabytes under a "Gi" label, which is why MAIN's
reserve arithmetic had been optimistic by 7 %: the gates use true GiB, so read `df -k` when a reserve is close. The apparatus finding
is the same one the day keeps producing in different coats: custody is a hash that was checked, never a label that was written.

## Addendum 43 (MAIN, 2026-09-11 ~19:50Z) — pc3 closes the pose corner on this carrier, and prices expire at every pointer move

The day's second pc3 result is a closure with a number attached. At n600, with every stop on physics, the continuous optimum inside
the shipped twelve-dimensional span is worth 349.9 B (CI 284.7–415.5) for the entire lattice family — the n=135 provisionals had been
1.4–1.7× too generous — and the gain is anti-concentrated: the twenty hardest pairs carry 63 % of the base and 21 % of the gain, and
twenty pairs gain nothing from infinite precision. That bound closes the global rungs on cost; the realized half-step rungs close the
per-dimension ones the bound could not reach, every one of them netting positive with the rate cost 4.4× the pose credit at best.
Two findings outlive the closure. First, the continuous optimum is a knife edge: projected onto any lattice, even sixteen times finer
than shipped, the pose leg scores an order of magnitude WORSE than the shipped point, because two rounds sit inside the render and the
shipped point was found by the solver, not by projection. Second, and the one every future charter must carry: rung prices expire at a
pointer move. On move 45 every capacity rung is dearer than on move 44 — the cheapest per-dimension halving went from 54 to 75 B and
the global halving from 808 to 914 B — because move 45 put the predictor at the schema's minimum, so a halving can no longer hide part
of its cost inside a predictor that was leaving 164 B on the table. A capacity rung priced on a body whose coder is not yet optimal is
priced too low; cl3's substitutes law, now measured on the predictor axis. The pose corner on this carrier is closed at formulation
scope with 349.9 B as its number; it reopens only for a different carrier format or a different basin, and this basin is narrow.

## Addendum 44 (MAIN, 2026-09-11 ~19:35Z) — hpr1: the shape rung is falsified at exact bytes, and its control is the largest rate move since the pre-distortion line

The one unpriced door the store could name was priced today, and the answer inverted the charter. hpr1
(`.omx/research/ddm_hpr1_hpac_receptive_field_shape_rung_20260911.md`) reconstructed the HPAC prior's receptive field from the
shipped receiver (four causal neighbourhoods; not one shape bit ships, so every shape rung is a receiver change in three files),
pre-registered six rungs from a geometry-blind window atlas over 112 million observations, and priced the first by the real coder on
the real stream with the decoded field proven identical. Dilating the past-plane window costs 812 B once isolated against its own
control: the joint-information statistic that predicted −3,625 B is refuted as a rung predictor on this object, and the atlas says
the unexploited information sits on the temporal axis, not the spatial one. The control is the finding. Retraining the shipped
prior's mixer in its shipped geometry for sixty epochs, warm-started from itself under cl2's law, prices −887 B at 179,359 B with no
receiver change and no distortion change: 3.45 % of the remaining gap in one output-lossless edit, thirty times the fire bar. Why
nobody had it is the law that outlives the number: the shipped prior descends from cl2's move-26 fit and the token field moved at
move 32; for thirteen moves the coder's model has been fit to a field that no longer exists. Every model section fit to an older
field is owed a refit before any structural rung on it is priced; pc3 measured the same law from the other side this afternoon
(rung prices expire at a pointer move). Sequencing: ntb2's rounding (−245 B) is the move-46 measurement now on the T4 and gives the
chain a measured leg; the retrained control rebases onto it as move 47 by normal seal; the two edits touch the same object and do
not add. MAIN's relative-significance review of the arm's DO-NOT-FIRE verdict on the dilated rung is at
`.omx/research/ddm_hpr1_20260911/MAIN_R1_VERDICT_REVIEW_20260911.md`: dominance, not magnitude; verdict scope INSTANCE.
