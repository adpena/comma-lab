# ddm_fb1 — THE FOLD-BACK PROGRAM (MAIN, 2026-09-03): every post-training discovery, restated as the training objective it should have been

Operator 2026-09-03: *"Seems like all of the stuff we have discovered and chain after training points to improvements
to the training and other steps themselves"* + *"believe in yourself … be creative and weird … think divergently"*.
Tokens: `[no-triality] [p0-ledger-ok]`. Axis: recall + S-arithmetic; no scorer, no pointer move.

## The receipts that say the operator is right

| post-hoc discovery (receipt) | what it bought post-hoc | what it says about TRAINING |
|---|---:|---|
| five lossless stages after freeze (pq12) | −454 B | post-hoc re-partitions what training decided; the object is set upstream |
| joint edit admission (jg5: 455/573; the sub-0.15 crossing) then the win-win cone drained (fcd1/wwc1: ~3.7 KB gross, net +0.0019 S) | one crossing, then nothing | the render↔judge equivalence class is thin near GT: the field should be trained for the judge, not edited toward it |
| residual priced by HARD sites, not count (gc1: −608,974 mismatches saved 18,728 B) | closed capacity 9.62× | the loss must weight sites by their coded price; "kill hard sites" is the objective, not "fewer wrong sites" |
| 95% of seg error is MANUFACTURED by the render path (td1/rt1; mst1 78.71% at the native render) | — | the renderer, not the labels, owns d_seg's residual; retrain the renderer against the realized argmax |
| pose = terminal solve, compensation in-compile (qs5/up2/jg5); large edits break the pose gate (fcd2: 26,710× miss) | pose held at 6.37e-6 | pose belongs IN the training loop at step zero (w96b/qbr1 law), not as a post-hoc rescue |
| Lane = 0.59% area, 33.56% of bits; square atoms barely touch Lane (gc1) | — | capacity/atoms must be routed by class at training (Lane/Movable), not uniformly |
| SegNet's boundary jitter is the irreducible information (gs3) | every coder door shut | only a field that is RIGHT WITHOUT CORRECTIONS moves rate — trained, not fitted post-hoc |

## The map: discovery → training-time lever (status)

1. **Realized expected-flip margin through R → uint8 → SegNet argmax vs the DALI GT** — the CE1 law; live in the
   born trainer (qbt1) and the w96b aligned branch; NEVER applied to the SHIPPED 30,856 B semantic renderer at
   its own size from its own weights (rj1 left realized deltas UNMEASURED; wd2 was a smaller student; w96a/b are
   a from-birth lineage at d_seg 8e-4). → **ft1 (chartered now): fine-tune the shipped renderer, same bytes.**
2. **Pose at step zero with the terminal re-solve law** — live in qbr1's config; folded into ft1's loss.
3. **Hard-site (coded-price) weighting** — a per-site weight from the coder's own −log2 p under the shipped HPAC
   prior (or the residual coder's price map): unmeasured as a training weight anywhere. → queued lever
   (`ddm_fb2_hard_site_weighted_margin`) for the born trainer after the burn's adjudication (do not confound
   the running discriminator).
4. **Rate-in-the-loop for the field** (differentiable codelength surrogate under the frozen HPAC prior) — the
   SCMDL/jf1 lineage; bounded by the thin equivalence class (wwc1); low priority.
5. **Class-routed capacity/atoms** — gf2/gc1 closed the static and dyadic forms; a class-matched generator is
   still only plausible as a parametric-motion form (ltg1 priced Lane at 233 KB): parked.
6. **The full-pipeline fold** (fpc1–3): train → lossless tail, with 1–4 inside the trainer instead of as stages.
   The 58 h n600 ticket is the vehicle; its objective should be the fold-back objective, not the PR130 curriculum.

## Divergent ideas I opened and closed at the arithmetic (so nobody re-opens them silently)
- Transmit a boundary-band GT texture code so the render reproduces the judge's jitter: the code must carry at
  least the jitter's entropy (data-processing), so it cannot be cheaper than coding the jitter. CLOSED.
- Ship the judge's bottleneck activations instead of the argmax (the NLA two-hop): inverting them needs scorer
  weights at inflate time. CLOSED by the strict scorer rule.
- Smooth-boundary field + tolerate jitter mismatches: 1 site = 1.273 B-equiv vs ~0.1 B to code (12×). CLOSED.

## Gestalt line
Post-hoc found the laws; training is where they pay. The first fold with a measured exchange rate on the
FRONTIER object is ft1: d_seg 2.01e-4 is manufactured by a renderer never trained against the realized argmax
at its own size — each 1e-5 recovered is −0.001 S at zero bytes (pose must be held: ≤ 1.25e-4 absolute, re-solved).
Own-vehicle frontier: **afr1 S 0.14797617125559104 @ 180,002 B [contest-CUDA T4 n600]** — unmoved.

## ADDENDUM 2 (2026-09-03, MAIN) — vr1's v7–v11 fold table consumed (a582c6019)

vr1 measured what the two live doors import from the v7–v11 substrate: ONE law (`ema_decay_run_geometry_v1`)
plus a memmap utility and the activation bridge. Zero loss terms, allocators, render kernels or basis frames
reach the born trainer (qbt1) or the fold-back (ft1) by import. Its 7 FOLD-NOW rows are training levers for
the BORN trainer, which is where fb1's fold-back belongs after ft1 closed the renderer door (coupling 217):

| fb1 lever | vr1 row | landed code → qbt1 site | measured receipt | order |
|---|---|---|---|---|
| render sampling (fb1 OMITTED it) | 2 | `aa_sdf_observation_render` → `forward:477-479` | 6.389× at g384 on achievable signal; born field 2.37× above the point bound | **ar1 fired** ($0 price on the born field, n32) |
| hard-site margin weight (fb2) | 1 | `_live_margin_weight` (mean-1, stop-grad) → `expected_flip_margin_loss:538` | annulus 5.71% holds 98.23% of flips (17.2×); 3.55× fixed-byte downshift | next burn generation |
| area cap (NEW to fb1) | 3 | Chan-Vese one-sided `relu(A_c−A_c^GT)²` beside `:593` | Lane 13.76× GT, Movable 4.58× GT ⇒ Road floored 0.398 | next burn generation |
| per-edge τ (NEW) | 4 | `tau·‖w_c−w_c'‖` from the frozen rank-4 head at `:538/:564` | flipdist spread 2.185× across edges | next burn generation |
| satisficing cap (NEW) | 6 | `MarginBandSatisficing` on the allocator | δ_R 0.01959, m_safe 0.03918 (n96 — re-run at n600 first) | after n600 δ_R |
| Lane thin-structure (NEW) | 7 | `persistence_topology_loss` torch port → Lane term | clDice 110× CE sensitivity; island recall +0.443; bulk Δ 0 | next burn generation |
| chroma-first routing | 5 | `chroma_boundary_match` → any renderer-touching loss | 22.5× pose-cheaper in the annulus; but Lane is the class chroma decides LEAST (0.08) | with any renderer rung only |

Corrections vr1 made to fb1 (accepted): item 3 SPLITS — the UNIWARD half is measured dead (texture ≈ chance,
Jaccard 0.024 vs 0.026) and the "Fisher" half needs no Fisher (margin↔curvature Pearson 0.978; margin weight AUC
0.991 vs S_R 0.767); item 3's "unmeasured as a training weight" is FALSE — `band_objective_weight` fired α=1.0
on 08-17 and was measured 08-22 (step rotated 66.3°, best exact-seg unmoved, judge underpowered, MPS axis) →
status RE-RACE; item 5 has landed code + n600 receipts; item 6 has a backtested schedule (−27.4% ∫d_seg).
Composition caution (m164): rows 1/3/4/6 act on the same per-pixel budget — ONE lever per race, B/H/W reported.
Rows 8–11 (costate schedule, along-tangent ladder, v8 atoms, warm-start law) are FOLD-AFTER-BURN so the running
discriminator is not confounded.

## ADDENDUM 3 (2026-09-04, MAIN) — ar1 verdict: the post-hoc footprint render REVERSES sign on the born field

MEASURED (ddm_ar1, memo 6fa21dce8; sealed QBR1 control checkpoint step 5,000; n600 with the trained n32 selection read
separately; DALI authority; frozen CPU-torch scorers): ss=2 footprint render vs the trainer's point render →
d_seg 0.002857 → 0.003753 (ratio **0.7612**, 32/32 pairs worse), d_pose +1.563e-3 (156× the +1e-5 bound), ΔS_HT
**+0.161213** = +242,113 B-equivalent (2.27× the whole cell archive). B/H/W: AA breaks 2.67 sites per site fixed; Lane
net −8,578. All three pre-registered falsifier clauses fired. The module's `build_supersampled_coords` lattice is
misregistered by 0.2497 coarse px at ss=2 (a real defect) but the centred lattice recovers only 1.6% — the cost is
footprint averaging itself. Dose-response: trained pairs lose 1.314×, unfitted 1.118× → the field LEARNED its own
point sampling. My charter carried two expired premises (m143 genus, ×2 more): the 6.39× law is an achievable-signal
upper bound that does not transfer to a learned field, and the burn trains n32 (`SELECTION_IDS`), not n600.

Consequence for the fold-back: **the render's sampling is part of the born field's identity; no post-hoc render swap
survives on a trained field** (sister of ft1's "training and export fight each other"). vr1 row 2 is DOWNGRADED from
FOLD-NOW to FOLD-AFTER-BURN as an in-loop RACE only (ss=2 costs 1.78× wall/pair ≈ 5.2 h per matched cell; falsifier:
AA-trained d_seg must beat the point-trained control at equal steps and bytes or the family closes at FORMULATION+1),
queued BEHIND the loss levers (rows 1/3/4/7), whose prior is untouched by this result. Owed and landed by MAIN: the
equation's `domain_of_validity` gains a `signal` key (real_frame_achievable IN-DOMAIN; point_trained_learned_field
EXCLUDED). Owed later, named trigger: re-anchor the law on the centred lattice only if AA-in-loop is ever raced.

## ADDENDUM 4 (2026-09-04, MAIN) — sd1: the surrogate mis-priced the CLOCK, and the excursion is rare-class over-paint

MEASURED (ddm_sd1, retained QBR1 seed-20260902 milestones, arithmetic exact to 0.0 at all 384 pair-milestones):
`seg_expected_flip_realized` −37.66% = (1 − 0.405364)·(1 + 0.048453) − 1 — a **−40.54% τ-schedule leg** (field frozen,
τ 0.15→0.05) times a **+4.85% field leg** (τ frozen at 0.05); residual 0.000e+00; both cells to four decimals. At
fixed τ the surrogate is FAITHFUL: it peaks at step 2,000 with the exact term, net sign matches in 5/5 windows, per-edge
sign agreement 97.2–99.98% of the excursion mass. The functional form is sound; **the schedule is the defect, and none
of vr1 rows 1/3/4 touches the schedule.** Pre-registered row-4 prediction FALSIFIED IN DIRECTION: the named edges
(Road→Lane 1.64, Undriv→Movable 1.44) are the LEAST mis-priced; the ≥2× edges are the long majority boundaries
(MyCar↔Road 3.8/3.5, Movable→Road 3.2, Road↔Undriv 2.7/2.5) carrying only 3–7% of d_seg each; spread 1.78–2.02× at
τ=0.15 collapsing to 1.19–1.23× at τ=0.05 (fires the 1.3× falsifier at the terminal τ).

Two facts that rank the rows: (a) **67–85% of the seg gradient sits on already-correct pixels** (τ=0.15 is 6.86 δ_R,
gradient half-max at 12.08 δ_R — the soft band is 4–12× wider than the undecided band); (b) **the excursion is
rare-class OVER-PAINT**: Lane predicted/GT area 1.0334→1.0929, Movable 1.0259→1.0580, both maximal at step 2,000,
mass-conserving (rare +7.550e-4 of the frame, majority −7.550e-4); qbt1's dual ascent is recall-only with no area cap
(`:593`) — vr1 row 3's precondition MEASURED present. 91.9–95.3% of the excursion lies within |margin| < 25 δ_R (2.0%
of pixels); the δ_R annulus itself holds 1.5% — DECIDED error, not roundtrip noise.

DERIVED order for the next burn generation (one lever per race, from the same warm start):
0. **FREE, now:** log `seg_expected_flip_realized` at a FIXED reference τ (0.05) beside the annealed value — the
   excursion becomes visible live (telemetry-always law, no training change).
1. **ng1 warm transition** (already sealed) — tests whether cold AdamW moments CAUSE the over-paint.
2. **ng2 = row 3, one-sided area cap** — covers 75.9–82.6% of the excursion mass, acts on the MECHANISM; λ_c from
   `derive_balanced_class_weights`' own bincount (no hand-typed areas).
3. **Row 1 margin weight** (reach 91.9% on 2.0% of pixels, 36× enrichment; attacks the 85%-on-correct-pixels waste)
   and **τ band at δ_R scale** (row 6 + this measurement: start τ near δ_R, not 6.86 δ_R) — raced after 1–2.
4. **Row 4 per-edge τ** — demoted: real but ≤1.23× at the terminal τ; second-order.
Composition caution m164: 1–3 overlap on the same pixels; race singly, then the winning pair.
Owed (sd1, structural): the live-vs-EMA-shadow half of the run's decoupling is UNMEASURED (milestones retain only the
shadow forward); unfitted pairs are structurally unavailable at milestones (`_evaluate_milestone` materializes only
`SELECTION_IDS`). Equations leg (`tac.canonical_equations`): `scalar_top1_top2_margin_is_exact_distance_to_flip_v1` (anchored by sd1,
038f2d81c), `chan_vese_area_constraint_birth_balance_v1` (row 3, ng2's lever), `muon_finisher_schedule_warmstart_and_
lr_anneal_v1` (ng1's lever); the τ-schedule deflation identity is a candidate law
(`expected_flip_surrogate_tau_schedule_deflation_v1`) to be registered when ng2's cell gives it a second anchor.

## ADDENDUM 5 (2026-09-04, MAIN) — gm1: the τ band is the third race; row 1's headline setting is inert

MEASURED (ddm_gm1 b828ce103, sd1's retained logits, 6 milestones × 2 cells × n32 × 41 τ, d_seg_hat reproduced to
0.0 at all 12 milestone-cells; `[macOS-CPU advisory · NON-PROMOTABLE]`): sd1's "85% of the seg gradient on correct
pixels" splits **77.7% WASTE (outside m_safe, R cannot undo) / 7.7% legitimate defence**; the all-correct share has a
FLOOR at 52.2% (the field's near-boundary correct:wrong ratio — no τ moves it), the WASTED share crosses <50% at
2.5–3.0 δ_R and <25% at 1.25–1.40 δ_R. **The τ band dominates row 1**: τ alone removes 45.6% (2 δ_R) → 77.7% (1 δ_R) →
96.8% (0.5 δ_R) of the waste from two default arguments (`ddm_qbt1_qbflow_trainer.py:643`); row 1's best tested
setting removes 37.3%, its shipped `inverse@1.0` 3.7%, and its headline **`bottom-k@0.05` is EXACTLY inert (0.0%)** — a
mean-1 mask whose selected set already contains the gradient support is a uniform rescale. The pre-registered "≥50%"
prediction FAILED under both readings (19.9% all-correct / 45.6% wasted). Levers compose near-multiplicatively (≤2.5 pp)
and are COUPLED through Lane: lowering τ drops Lane's gradient share 1.6–2.1× and raises a global cap's Lane over-push
2.76% → 17.45% — so the per-class m_safe_c cap is a dependent leg of the τ band, not an independent row-6 variant.
Corrections carried: sd1's 85.10% was unweighted (HT 85.385%); hg1's "GT is runner-up on 98.0% of flips" is 90.1% on
this vehicle (does not transfer).

DERIVED third race (**ng3**): `tau_for_step(start = 2·δ_R = 0.04376363754272461, end = δ_R = 0.021881818771362305)`
replacing [6.86, 2.29] δ_R — waste 77.7% → 42.3% at start, 44.4% → 16.2% at end; bonus: cuts sd1's schedule-leg
deflation 4.8× (−41.3% → −8.6%). One lever: the cold control has NO satisficing cap, so the Lane over-push caveat
binds only when a cap is later added (then co-fire per-class m_safe_c or hold τ_end at 1.5 δ_R). Row 1 is DEMOTED to
"re-race only with a non-inert setting, after ng3". RACE ORDER NOW: ng1 warm → ng2 area cap → ng3 τ band → (winners
composed pairwise, m164). Equations leg (`tac.canonical_equations`): `margin_band_satisficing_threshold_v1` (m_safe
resolved through the law at runtime, n600, fallback False) and `scalar_top1_top2_margin_is_exact_distance_to_flip_v1`;
the waste-floor identity is FORMALIZATION-owed to the τ-schedule law named in addendum 4.
