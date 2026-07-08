---
doc_type: t5_crucible_deepmath_meat_hunt_round1
role: DEEP-MATH MEAT-HUNT reviewer (second lens of the recursive adversarial review; parallel to P5 verify)
date: 2026-07-07
target: DRAFT_OPTIMAL_STACK_v2_20260707.md
round: 1 (findings reset the 3-clean-pass counter)
verdict: NOT CLEAN — 1 BLOCKER · 5 MAJOR · 12 MINOR · 10 MEAT items
axis: all numbers [macOS-CPU/MLX advisory] unless noted; pointer contest-CPU 0.19110 UNMOVED — this file is MEANS.
---

STORES CONSULTED: DRAFT_OPTIMAL_STACK_v2 (full) · DRAFT_OPTIMAL_STACK v1 (targeted §1.2/§3.2/§4 rows) ·
pursuit_chainA_spectrum_solve (full) · ORCHESTRATION_LEDGER (full, incl. reqs H+I + log) ·
CONTEXT_COMPENDIUM (targeted: FEED-07g lines, #139/#207 rows) · mod32cap
`levelset_train_result.json` (41-row trace RE-READ: ep550-700 d_seg extracted, backtest re-verified) ·
trainer `train_levelset_witness_realized_through_R_mlx.py` (anneal denominator functions READ at
L2325-2400; --render-aa/--logit-adjust fail-close comments L807-2790; eikonal fns L1198-1371) ·
corpus_query ×4 (#207 sig-proc levers · #139 hood clamp · openpilot tangent/self-orient · measured
lever inventory) · memos surfaced: `signal_processing_filter_levers_derived_20260701`,
`project_sig_proc_filter_chain_measured_R_allpass_L3_ntk_20260701`, `hood_static_component_20260627`,
`frozen_source_0byte_dseg_priors_design_20260626`, `measured_lever_inventory_for_synergy_pass_20260701`,
`capstone_synergy_composition_map_20260626`, `openpilot_world_model_lane_alignment_plan_20260706`,
CANONICAL_RESEARCH_INDEX. NOT consulted: P3 verdict full text (its F1-F17 carried via v2 §0.1);
position_S1/S4/S6 full (targeted via ledger log); durable-state files (stale per sweep).

# ROUND-1 FINDINGS (ranked; every claim below re-derived or tagged)

## §A HUNT-1: LAW RE-DERIVATIONS — what CERTIFIES and what FAILS

**CERTIFIED (recomputed by hand, all pass):**
- **§0.2 crossing table — every row re-verified.** 100·0.0011+√(3.0e-4)+0.0620 = 0.110+0.017320+0.0620
  = **0.18932** ✓ (margin 0.00178); knife-edge row 0.105+0.024083+0.0620 = 0.191083 ✓; S6 triple
  0.092+0.038859+0.0602 = 0.191059 ✓; v1-bug row 0.22073 ✓; central 0.26073 ✓. Budget legs
  0.1291/0.1338/0.1442/0.1143/0.1056 ✓. λ_bytes 25/37,545,489 = 6.6586e-7 ✓.
- **§5.1 byte sums**: central 60,000−3,108+30,892+4,500+800 = **93,084** ✓ → rate 25·93,084/37,545,489
  = 0.061981 ✓. Independent band [70,392, 103,513] ✓ → [0.046871, 0.068925] ✓. Waterfill-fail 115,277
  → 0.076758 ✓. (Worst joint tail: see A-7.)
- **§0.3 per-class sums**: 0.4396/0.1226/0.4378 × 0.0034 = 0.001495/0.000417/0.001489 ✓ (sum 0.0034);
  design band [0.0010, 0.0023] ✓; islands-only floor 0.0034×0.4378 = 0.001489 ≈ 0.0015 ✓. §9.1 ladder
  edges: lower 0.10+0.0173+0.0469 = 0.1642 ✓; upper 0.28+0.105+0.0855 = 0.4705 ✓.
- **`--anneal-epochs 600` completion guarantee — CERTIFIED with the counterexample search done.**
  Trainer READ (L2332-2390): `_ae = args.anneal_epochs or args.epochs`; progress = (ep−1)/(\_ae−1)
  with **ABSOLUTE 1-based epoch** for BOTH `_softmax_temp_for_epoch` and `_hosc_beta_for_epoch`.
  ⇒ τ=τ_end ∧ β=β_end at ep600 REGARDLESS of stage-exit path: early CE co-predicate fire, CE
  cap-fire at 300, TAU fire waiting on anneal-complete (predicate includes it), and the 726 cap all
  satisfy completion-before-consumer since every consumer fires ≥600. No counterexample exists under
  absolute-epoch semantics. Re-derivation cross-check: control (denominator 1000, τ 1.0→0.05 cosine)
  at ep726 gives τ = 0.05+0.95·0.5(1+cos(0.7257π)) = **0.216** — exactly M-S2-2's measured truncation
  value. The law is sound. (But see MAJOR-3 for what "completion" now means.)
- **Eased-dilation 1-Lipschitz (v1 L140/L294)**: |dr/dt| = r₀/275 px/ep ≤ 1 for any r₀ ≤ 275 px —
  trivially satisfied at movable dilation radii (px-scale). The load-bearing clause is
  completion-gates-CE-exit: 275 ≤ CE cap 300 ⇒ worst truncation 25 ep ✓ as stated. CERTIFIED.
- **Trust-region ½λ̂_max‖Δθ‖²**: damping at the measured stiffest curvature (λ_max ≈ +117-139,
  K-stable per chain-A) upper-bounds the quadratic model error along any step — standard and
  consistent with the measured spectrum. CERTIFIED (Euclidean-metric caveat → §D-1).
- **Co-predicate backtest arithmetic re-verified on the trace**: at ep625 trailing-V=4 rel slope =
  (0.0033929−0.0034069)/(3·25ep·0.0034069-normalized) = −1.37e-3 ✓ matches the printed value; at
  ep600 the window slope ≈ −5.2e-3 < −5e-3 ⇒ correctly no fire ✓. The table is real.

**FAILED / INCOMPLETE laws:**

**[BLOCKER-1] §0.3 vs inherited §3.2: TWO CONTRADICTORY per-class share tables drive the design.**
v2 §0.3 (from S5-R5 composed-ceiling, ep300 surface): **lane 0.4396 · movable 0.1226 · big-3 0.4378**.
v2's §3 inherits v1 §3.2 UNCHANGED, whose per-class table reads: **Movable 44.8% of flips · Lane
19.1% · Big-3 36.1%** (S5's un-born flip-share split, 44.8+19.1 = 63.9%). The island-dominant CLASS is
swapped between the two sections. §0.3 computes the crossing design residuals (lane leg biggest);
§3.2 computes the curriculum/λ priorities (movable leg biggest, movable-first mechanism emphasis).
Both may be true on their own surfaces (composed d_seg share at ep300 vs terminal un-born flip
share on mod32cap-final) — but v2 uses them interchangeably with NO bridging law, and the headline
"crossing engineered" (§0.2/§0.3) rests on the lane-dominant table while the lever priorities rest
on the movable-dominant one. Req-H apples-to-apples violation inside one document. **Fix:** print
both measurements WITH their surfaces, derive the bridge (composed-share(t) evolves as births land;
flip-share = composed-share at the un-treated endpoint), and re-check that §3.2's λ-gate priorities
and §0.3's residual bands are consistent under ONE model. P6 (share stability) covers
control→arm transfer, NOT this internal contradiction. review_status: fresh-eyes-round-1.

**[MAJOR-1] The ξ q-law threshold (0.002) EXCEEDS the crossing margin (0.0018) and is not
byte-balanced — a per-section threshold where a joint KKT law is required.** F10's fixed law
("smallest q with Δ(pose-term) < 0.002") can accept a move that costs up to 0.002 of pose term to
save ~2-2.5 KB ≈ 0.0013-0.0017 of rate — **net-NEGATIVE S and larger than the entire §0.2 crossing
margin 0.00178**. Same disease at the section level (binary-collapse item 8): base bits ride
waterfill, band rides P8 net-ΔS, pose rides a fixed 0.002 — three different admission rules, no
equalized marginal. **The law:** one joint condition at byte-close — for every section, shed bytes
while marginal Δ(distortion terms)/Δrate < 1, i.e. pick q* = argmin_q [√(10·d_pose(q)) +
25·bytes(q)/37,545,489]; identical form for band umask/giveback and waterfill depth. v2 already owns
λ_bytes = 6.659e-7 S/B; use it everywhere. Cost: $0 (byte-close-time selection over the same P9
sweep). review_status: fresh-eyes-round-1, arithmetic above.

**[MAJOR-2] TAU→FIN event exit: transition-from-stage-best is UNSPECIFIED, and the backtested fire
FORFEITS measured d_seg that v2 does not print.** From the trace: fire ep625 lands on θ with d_seg
**0.0033929** — WORSE than ep600 (0.0033716) and worse than the cap-path best ep650 (**0.0033662**).
If FIN warm-starts from current θ (v2 nowhere says otherwise) and the regression guard's
restore-best is FIN-internal, the event path enters the finisher from a noise bump and, on
FIN-fails-to-beat-entry, finishes **+2.67e-5 d_seg = +2.7e-3 S** worse than the cap path — 1.5× the
crossing margin. Even with transition-from-TAU-stage-best (ep600), the never-trained ep650 point
costs +5.4e-6 d_seg = **+5.4e-4 S**. The §2.2b framing "one cadence before the actual best" presents
a quantifiable cost as a validation. **Fixes (both cheap):** (i) SPECIFY the boundary law: TAU→FIN
restores the TAU-stage EMA-best before Muon warm-start (one line; consistent with per-stage-ckpt
mandate); (ii) print the forfeited-Δ column in §2.2b and fold ~5e-4 S into the event-path leg of
§9.1 (the bet that warm-Muon-on-completed-anneal recovers it is fine — but it is a BET, currently
invisible). Related: the forfeit is partly a 25-ep cadence artifact (binary item 9 → §C-9).
review_status: fresh-eyes-round-1, numbers re-read from the on-disk JSON this session.

**[MAJOR-3] The "big-3 anneal-completion recovery ~4-9e-4" band is UNDERIVED and partially
inconsistent with v2's own config.** v2 completes the anneal to **τ_end = 0.2** — but the control's
TRUNCATION point was τ = 0.216. The τ leg of "completion" is Δτ = 0.016, ~nil by design; the entire
recovery must ride β 3.177→4.0 plus a MID-RUN REGIME CHANGE v2 never names: compressing the
denominator 1000→600 moves τ(ep300) from 0.804 (control) to 0.601 (v2) — the arm trains the whole
TAU stage at materially sharper τ. So the named binding constraint's in-flight instrument will
measure completion ∧ compression ∧ path-shape CONFOUNDED. And no source derives 4-9e-4 (it appears
first in v2 §0; S2's M2 measured the truncation, not the recovery). **Fix:** re-state the constraint
honestly as "β-completion + τ-path-compression recovery, jointly"; either derive the band from the
measured d(d_seg)/dβ near ep650 (the trace's late-TAU slope gives a first-order anchor) or tag the
4-9e-4 DPR. Optional clean arm: τ-shape `cosine_hold` with hold at 0.2 matching the control's early
path then completing β — isolates β-completion. review_status: fresh-eyes-round-1; τ values
recomputed from the trainer's own cosine formula.

**[MAJOR-4] THE OPENPILOT GEM — deterministic tangent orientation is unadjudicated while v2 pays
for the polynomial anyway.** v2 stores the openpilot lane polynomial (LBND4, 30,892 B booked) and
SEPARATELY learns orientation via self-orient — paying the +4.3% checkpoint-reconstruction gap
(chain-A's own caveat on every probe), the GFC×self-orient fail-close seam (trainer L2783), and the
AA fine-mode cost (~P fine-EDTs/ep ≈ +29 s/ep@n600, trainer L2727-2731 — the dominant P11 risk).
Measured (coordinator-supplied, DAG ~3440): openpilot centerline-perpendicular alignment to lane
boundary normals median |cos| = **0.966** vs learned self-orient transfer |cos| **0.893-0.909** —
the FREE deterministic field is MORE aligned on the lane than the learned one. Honest scope: the
−48% anchor is ALL-CLASS directional; the polynomial gives the LANE field only — full replacement
is NOT justified. But a HYBRID is: lane-annulus orientation from the stored polynomial (already
paid; rule-118 free at decode → nothing to persist for the lane field) + static rows for
hood/horizon + self-orient for the movable/road residual; plus the VP-radial global chart
(`normalize(VP−centroid)`, #325 memo) as a $0 oracle arm. **This is the req-I geometry-sharing
headline: ONE stored polynomial should serve band + comb phase (dash phase = ego-distance = screw ξ,
L73) + lane basis orientation; v2 currently serves only the band from it.** Demand: a $0 oracle
probe (swap the lane-annulus orientation field, frozen scorer, ep650 ckpt) added to §7 pre-GO or
first-recess; adjudicate with the 0.966-vs-0.909 arithmetic. review_status: alignment numbers
coordinator-supplied — verify the DAG row at probe time.

**[MAJOR-5] Fractal/req-H mis-tuning: uniform weights where per-class optima are derivable.**
AmplifyIsland weight=1.0 and PersistenceTopology weight=1.0 are shared across lane and movable —
two objects whose shares differ 3.6× (whichever table from BLOCKER-1 wins) and whose failure laws
differ (dash erasure ∝ 1/persistence vs blob non-birth). At equal marginal-ΔS the per-class weights
are λ_c ∝ share_c / (current per-class residual sensitivity), both measurable from the per-class
F-rows v2 already builds. Same pattern: LogitAdjust τ=1.0 shared (Menon τ is per-class tunable);
one global EMA decay 0.997 across trunk/head/film groups. Not launch-blocking (per-class λ gates
exist in §3.2), but the amplify/persistence weights are the two places where the physics is
per-class and the knob is pooled. **Fix:** per-class weight law from the F-row shares, or an
explicit DPR tag per knob. review_status: fresh-eyes-round-1.

## §B MINOR findings (each with the arithmetic)

1. **§0.2 "band lower 0.0573" is v1's F11-condemned band edge** (≈ 25·86,054/37,545,489): §5.1
   replaced [86K, 99K] as underivable, yet §0.2 still prints its lower edge as a budget leg. Delete
   or re-derive from §5.1 components.
2. **Worst-joint-tail 128,376 B is a THREE-event tail labeled as two**: 82,193−3,108+41,562+800 =
   121,447 + pose **6,929** (upper tail, not central 4,500) = 128,376. Rate 0.0855 ✓ for the number;
   the label must add "∧ pose-q upper" (or the number becomes 125,947 → 0.0839).
3. **Model A product is 1.6-6.4%, printed "2-6%"**; and **Model B "8-15%" carries NO arithmetic at
   all** — no per-axis repair probabilities are stated, so the number is unfalsifiable. Print the
   per-axis repair-success estimates (even coarse) or tag Model B's band DPR.
4. **§8 "5-27%" is not reproducible**: with FIN duration a shared random variable, savings = 101 ep
   fixed → 10.1-12.2%; comparing event-fast vs cap-slow endpoints gives (1000−725)/1000 = 27.5% and
   (1000−899)/1000 = 10.1% → honest range ≈ **10-27%**, and the 5% lower edge is underived.
5. **"within ~15% of its optimistic edge" is actually ≤10%**: 0.0011/0.0010 = 1.10.
6. **LogitAdjust prior source mismatch**: −5.13 = ln(0.0059) but §3.1's GT lane mass 0.00577 gives
   ln = −5.155; −4.39 implies movable prior 0.0124 vs n96's 0.0156. Constants are the right FORM
   (log-priors, Menon, sign consistent with the trainer's loss-side wiring) but the two sections
   cite different prior measurements — pin ONE n600 artifact (req-H apples-to-apples).
7. **"waste bounded ≤ ~50 ep" (§1.2 anneal row) is underived**: if the arm's co-predicate readies at
   ep450, the anneal-complete wait is 150 ep. The real bound is 600 − (earliest arm fire), argued
   only by "arm exhausts later" — tag the 50 as an expectation, not a bound.
8. **λ=0 twin contract incomplete**: same-seed pinning and cadence/stage-boundary matching are not
   stated. Event exits are data-driven ⇒ twin/primary stage boundaries can diverge; "matched epochs"
   needs a stated convention (absolute epoch, both post-CE-cap). One sentence fixes it.
9. **Co-predicate cadence generalization**: eps_rel = 5e-3 is calibrated per-25-ep. The B1 in-trainer
   build MUST normalize slope per-epoch (or re-derive eps_rel) so a cadence change doesn't silently
   recalibrate the trigger. (Binary item 9: an event-adaptive cadence — finer near predicted
   exhaustion, cadence ∝ 1/|slope| with a floor — would also shrink MAJOR-2's forfeit; cost = extra
   n600 verdicts; RECESS.)
10. **Adaptive-ε #318/#320 "self-deriving CFL law" is cited but never STATED** in either draft — the
    one law in the stack a reviewer cannot re-derive from the document (formula + floor/upper clamps
    + this config's lr/λ_eik values absent). Print it in §1.2 or it fails the control-law contract
    it claims. (Trainer has `_visco_eps_for_epoch` linear-decay + `len/norm_eps=1e-2` constants; the
    CFL row lives in the equations registry — cite the row id and its clamp values.)
11. **LogitAdjust ∧ micro-batch fail-close**: trainer L890 refuses `--logit-adjust-loss-tau` with
    `--micro-batch-pairs>1`. S5 retired MicroBatch so ARM-PRIMARY is serial — but add this pair to
    the P12 config-consistency checklist so no throughput fix re-introduces it.
12. **ChromaBoundarySharpen(0.1, margin_band=1.0) are bare constants with no law row and no cite of
    the LEVER-4c measured-GREEN config**; and no chroma×pose antagonism guard (sharpened chroma
    feeds PoseNet's YUV6 on scored pairs — the F11 pose watch should stamp chroma-engage epoch for
    attribution). DPR-tag + one telemetry line.
13. **Comb engage lacks the measured deconflict ramp**: v1's band row carries "20-ep cosine rewarmup
    on its weight — measured 3.4× collision harm at hard engage"; v2's comb engages at band-fire+25
    with NO ramp law. Same physics, same fix: comb weight inherits the 20-ep cosine engage. (Binary
    item 6 confirmed; the P1 gate itself staying binary is fine.)

## §C MEAT HUNT + operator-addenda dispositions (per lever, with receipts)

- **#207 pre-emphasis + deconvolution — the charter's premise is WRONG; exclusion is JUSTIFIED.**
  Measured (memo f206231a4 / L-index `project_sig_proc_filter_chain_measured_R_allpass_L3_ntk`):
  R is near ALL-PASS (|H_R| 1.0→0.842 at render-Nyquist; Wiener ceiling +1.25 dB) ⇒ R-deconvolution
  and pre-emphasis-OF-R are DEAD, "don't build". Matched-filter + brute-AA also negative
  (compendium line 310-region). OUT-with-receipt. The SURVIVING sibling is the **L2 phase lever**
  (sub-pixel placement, $0 inflate-side, targets set train-time) — which is exactly the #149 class
  v2 defers with build-spec; AA ss=2 partially covers it in run-1. Disposition sound.
- **#139 ego-hood static clamp — quantified: exclusion defensible on ΔS, but it is the 8-BYTE
  req-H answer for the hood class.** Measured: negligible standalone (~19 flips; hood IoU 0.994)
  per `capstone_synergy_composition_map` row 18 + `measured_lever_inventory` D12. BUT the
  frozen-source design (`frozen_source_0byte_dseg_priors_design_20260626`) prices the sky+hood
  row-threshold clamp at **≈8 bytes clamping 63% of pixels** — a no-regret decode-side option.
  Verdict: **IN-at-byte-close** as a paired-verdict-gated byte-close-selectable (same class as AA's
  ship-better-of rule) — it costs 8 bytes, can only be kept if the paired verdict improves, gives
  hood its own law (req H), and composes with #158's homography margin prior (run-2). MEAT-minor.
- **Chroma-at-annulus (LEVER-4c)**: IN ✓ (ChromaBoundarySharpen) — see MINOR-12 for the missing law
  row/anchor cite.
- **L3 NTK band-pass / whitening preconditioner — REAL MEAT, S-NEUTRAL SPEED.** The same sig-proc
  memo's ranked-#1 lever: ~3-10× convergence speed on the finest addressable band (the deep-math
  reason curvelet-scale curricula work). v2's §8 wall-clock plan never mentions it. Per L59
  (training-time = lexicographic secondary, S-neutral speed wins FREE): if the preconditioner is
  built (check `--` flags / DSL), it belongs in P11's throughput smoke as a candidate; if unbuilt,
  name it in §9.4 with a build-spec. MEAT-check (status unverified this pass).
- **Orbit-coding beyond rev-2k**: permutation slack MEASURED NO (best arm −8 B; most HURT) —
  OUT-with-receipt ✓ (compendium §C correction). rev-2k −3,108 measured stands.
- **Persistence birth ORDER**: stagger 275/275/boundary+50 is mechanism-proven-first, not
  persistence-derived. The Morse-Smale reading (curriculum = persistence order) would order births
  by feature persistence — a fractal refinement, not a defect; note in §9.4.
- **#288 semi-discrete OT (head offsets)**: stays HELD; RECESS-with-band only. No run-1 claim.
- **Openpilot harvest audit (operator addendum, 9 items):**
  (1) deterministic-tangent GEM → **MAJOR-4** above. (2) **v_h=174 / cam_h reconciliation (#327)**:
  ledger says MEASURED-OPTIMAL, but neither draft PINS v_h=174 in the band config rows — pin it
  (MINOR; apples-to-apples with the LBND4 measurement). (3) **#158 homography static-class margin
  prior**: unconsumed; ΔS unmeasured; run-2/recess with the hood clamp as its cheap sibling.
  (4) **comma2k19 GT ego-motion as ξ INIT** (compress-time, legal): unconsumed; cheap pose-track
  init that de-risks the 3e-5 bar — MEAT-minor, fold into P9. (5) **bicycle-model kinematic prior
  on ξ**: the store-nothing section is a smooth dynamical trajectory; AR/dynamics-constrained
  residual coding pushes pose bytes toward the 2,700 tail AND regularizes the carrier — derivable
  law (highway ego-motion ≈ 2-3 effective DOF/step), MEAT-minor, fold into P9's q-sweep (synergy
  with MAJOR-1's joint byte law). (6) **MUTCD dash-period prior**: comb period SET (measured once
  at compress time / standard), phase from ξ (L73: dash phase = ego-distance) — directly shrinks
  the L65 mis-phase risk P1 exists for; MEAT-minor, fold into P1's audit as the null hypothesis.
  (7) per-clip VP: CONSUMED ✓ (VP-tangent eased seed). (8) **#203 free lane-raster through-R**:
  subsumed in run-1 by band-trained-with; keep the task cross-linked. (9) **#191 openpilot-seeded
  ep0**: OUT-with-reason — GT-paint-then-SDF (§11 row 10) dominates any openpilot-derived seed at
  compress time; openpilot's value is at DECODE (rule-118), not at ep0.

## §C2 Binary-collapse audit (operator follow-up, 10 candidates graded)

| # | candidate | grade | law / reason |
|---|---|---|---|
| 1 | pairs pooled | FINDING-minor | hardness-graded pair oversampling with unbiased reweighting (importance sampling; variance ∝ per-pair flip mass) — RECESS ($0 design + trainer sampler); naive oversampling CHANGES the objective — the law must keep E[grad] fixed |
| 2 | frame asymmetry | OUT-with-reason (verify-cheap) | SegNet scores x[:,-1] only; seg-only stages should render/backprop the scored frame only — believed true by construction; add a one-line assert to P12 rather than a law |
| 3 | annulus-binary vs margin-graded | FINDING-minor | weight ∝ flip-proximity IS our Fisher law (margin↔Fisher 0.978); note S5's DEFER killed the msal_uni TEXTURE proxy (L76: at chance), NOT margin-weighting itself — a margin-graded annulus weight is un-tried and deep-math-grounded; $0 oracle probe, run-2/recess |
| 4 | one global τ (+ one EMA decay) | FINDING-minor | per-class τ_c ≤ margin_q(c)/ln5 (see §D-3); per-scale τ via parabolic scaling; trainer lacks per-class τ ⇒ run-2 build; run-1: SET τ_end from the law (D-3), keep global |
| 5 | step-function kills | OUT-with-reason | W1b score-matched w_pose IS the graded law; kills are ship-gates (inherently discrete decisions), correctly discrete |
| 6 | comb-vs-along binary | FINDING → MINOR-13 | ramp the comb ENGAGE (measured 3.4× hard-engage harm on the band — same physics); the P1 pass/fail gate itself stays binary, fine |
| 7 | one trajectory vs branch points | FINDING-minor/MEAT | per-stage EMA ckpts are free branch points; formalize a run-1.5 branch protocol in §9.4 (tau-boundary branches: comb-on/off arm, mid-λ point, openpilot-tangent arm) — sequential fine-tunes at fraction of full-run cost |
| 8 | section-level rate split | FINDING → **MAJOR-1** | joint marginal law at λ_bytes across {base bits, band, pose q, grammar} |
| 9 | fixed 25-ep cadence | FINDING-minor → §B-9 | cadence ∝ 1/\|slope\| with floor; shrinks MAJOR-2's forfeit; costs verdict wall-clock — RECESS |
| 10 | λ=0 twin 2-point | OUT-with-reason | sequential wall-clock binds; DPR λ*∈{5,15,30} unswept stands; name a short tau-boundary-only mid-λ arm in §9.4 |

## §C3 Req-H per-class + req-I synergy scorecard

**Per-class (H):** lane ✓ (band/comb/VP-seed/LengthSigma/own λ + ep150 alarm) · movable ✓ (dilation
homotopy/logit-adjust/part_frac abort) · **hood ✗ — pooled into big-3, no own law/telemetry/kill;
the 8-byte clamp (§C #139) is the cheap fix** · road/undrivable ✓ (basis C² + τ jitter curriculum).
Pooled comparisons: BLOCKER-1 (share tables), MINOR-6 (prior sources), and the RUN-level kill
(d_seg > control, pooled) — the last is a DELIBERATE F7 choice (stack-level kills); per-class
F-alarms carry the surgical layer; tension acknowledged, acceptable.
**Synergy (I):** anneal×warm-Muon ✓ (precondition law) · WeightEntropy×waterfill ✓ (Class-D×B
recess) · entropy-as-regularizer ✓ (twin measures it) · AA×island-survival — coupling is the
RATIONALE but has no attribution row; add per-class AA paired deltas at the stage boundaries
(lane vs movable split of the AA delta, same 4 verdicts, zero extra renders beyond the paired pass)
· **geometry-sharing ✗ — the headline gap (MAJOR-4): one polynomial should serve band + comb-phase
+ lane orientation; ξ dual-use serves warp+pose ✓ but not comb phase** · chroma dual-use: d_seg ✓,
pose-side guard missing (MINOR-12) · telemetry-sharing: per-class λ feeds exits+amplify ✓ (v1
§3.2(iv)).

## §D HUNT-4: fresh deep-math (each IN-with-law / RECESS / OUT)

1. **GN/Fisher-metric trust region for SOLVE — IN-with-law (run-2/SOLVE stage).** Chain-A's Krylov-TR
   solves ½yᵀTy+‖g‖e₁ᵀy s.t. ‖y‖≤r in the EUCLIDEAN θ-metric; the natural metric for a
   frozen-scorer argmax objective is the GN/Fisher pullback (margin field = Fisher surrogate,
   0.978). Same Lanczos machinery on the GN-preconditioned operator; changes WHICH directions the
   radius admits. Does not touch run-1 (SOLVE is gated off at ep650 anyway); write into §2.3's
   SOLVE spec. Cost ~0.
2. **Persistence-weighted island amplification — IN-with-law pending a $0 probe.** Measured law:
   error ∝ 1/persistence (dash-erasure). Replace AmplifyIsland's uniform hinge weight with
   w_i ∝ 1/pers_i (clamped), computable from the CacheGtSkeleton persistence pairs v2 already
   builds. Targets the measured failure law directly instead of amplifying all islands equally
   (which over-spends on already-safe high-persistence blobs). $0 oracle probe on the ep300
   composed surface decides run-1 vs run-2. (Also resolves half of MAJOR-5.)
3. **Semiclassical τ_end law (τ=ε=ħ; Maslov err ≤ τ·ln5) — IN-with-law, $0, SOLVES §11 row 14.**
   The argmax-vs-smoothed-partition error is bounded by τ·ln5 in logit units; the run is safe when
   that bound sits below the annulus margin quantile: **τ_end* = m_q / ln5** with m_q the q-th
   percentile of the |margin| field on the flip annulus (cached; $0). If τ_end* ≈ 0.2 the DPR is
   RATIFIED with a derivation; if it lands materially lower, v2's τ_end=0.2 is leaving sharpening
   on the table exactly where MAJOR-3 says the recovery story is thinnest. Run pre-launch.
4. **Semi-discrete OT for head offsets — RECESS-with-band** (held #288); nothing new measured;
   no run-1 claim.

## §E Flag/config spot-checks (hunt 3 residue)

`--anneal-epochs` ✓ (semantics READ, absolute-epoch — the load-bearing verification) ·
`--tau-anneal-shape`/`--tau-hold-frac` ✓ · `--render-aa`/`--aa-supersample`/`--aa-self-orient-fine-mode`
✓ referenced with the fail-close comment (L2727: supersample+self-orient fail-closed WITHOUT
fine-mode; fine-mode cost ~P fine-EDTs/ep ≈ +29 s/ep@n600 — fold this NUMBER into P11's predicted
band: base 107 + AA-render ≈ 4×px + 29 s/ep EDT ⇒ the 1.5× gate is TIGHT, state it) ·
`--logit-adjust-loss-tau` ✓ (with the micro-batch fail-close, MINOR-11) · `--ground-frame-chart`×AA
fail-close ✓ declared (L2783) · AA×band ordering: band is analytically coverage-integrated and
composed AFTER downsample (L2672 + FEED-07g byte-identity proofs) ⇒ no double-AA seam, ordering
defined — CERTIFIED. Memory: P11 must also print the projected cf_mx_cache under fine-mode (the
~41 GiB n600 self-orient cache must NOT scale ×4 with supersampled px; if it does, the preflight
REFUSES — that is the gate working, but print the projection so the mode choice is visible pre-GO).

---
**Round-1 verdict: NOT CLEAN — 1 BLOCKER · 5 MAJOR · 13 MINOR · MEAT as listed. Counter resets.**
All fixes are document/law-level or $0 probes; nothing found invalidates the vehicle, the crossing
triple's arithmetic (re-verified ✓), or the anneal-completion mechanism (certified ✓ as scheduled —
its RECOVERY story is the confounded part). Pointer 0.19110 UNMOVED — this review is MEANS.
