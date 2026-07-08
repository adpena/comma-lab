---
doc_type: t5_crucible_seal_round2_verdict
role: SEAL ROUND 2 (both lenses: disposition-verify + deep-math meat hunt) on DRAFT v3
date: 2026-07-07
target: DRAFT_OPTIMAL_STACK_v3_20260707.md (chunks e65cfdf76 · 1097a49e7 · 83224c801; landed 11017be0e)
round: 2 of the 3-consecutive-clean seal (round 1 = 1 BLOCKER · 5 MAJOR · 13 MINOR on v2; counter 0)
verdict: NOT CLEAN — 1 BLOCKER-grade · 2 MAJOR · 6 MINOR (counter RESETS to 0)
operator_pins_folded: req J (precision + margin-denominated significance) + req K (native format +
  adequate capacity per technique) + req M (capacity humility · self-deriving/feedforward control ·
  response surfaces as DECIDE models) — all three arrived DURING this round; J/K/M findings are v4
  fold items, graded against the now-binding ledger text.
axis: all numbers [macOS-CPU/MLX advisory] unless tagged; pointer contest-CPU 0.19110 UNMOVED —
  this verdict is MEANS.
review_status: fresh-eyes round-2 (this verifier authored none of v1/v2/v3/P5/meat-hunt/chain-A);
  every load-bearing number below RE-EXECUTED against the primary on-disk artifact, tagged
  [re-executed]; text readings tagged [verified-by-inspection].
---

STORES CONSULTED: DRAFT v3 (full) · deepmath_meat_hunt_v2 (full) · P5_second_redteam_verdict (full)
· pursuit_chainA_spectrum_solve (CURRENT on-disk incl. the 20:36 TERMINUS commit 42fa00812) ·
ORCHESTRATION_LEDGER (reqs A-K incl. the NEW J+K rows) · recess_wave1_R1_R3_R6 (LANDED: R0/R3
647201389, R1 f4cdf00c5 — both PRE-DATE the v3 chunks) · DRAFT v2 + v1 §10 (regression base) ·
PRIMARY ARTIFACTS re-read: `lane_share_probe_ep225_n600.json` (Surface A) ·
`islands_composed_ceiling_arithmetic_20260707.md` (Surface B) · `levelset_train_result.json`
(41-row trace; ep550-700 extracted) · `birth_death_persistence_dseg_20260630T172510Z.md` (flip-
support margin bins) · canonical_equations_registry (`adaptive_eps_cfl_edge_tracking_v1` full row)
· `witness_config_differential_equations_derivation_20260705.md` (CFL two-sided window + c_a(τ)
coupling) · trainer argparse + island_amplify block (L3780-3820, L8284-8290, L8368) ·
`island_protection.island_persistence_weight` source · `curriculum_dsl.py` (AmplifyIsland L2095,
PersistenceTopology L2308, DirectionalBasisRebalance, sealed_205_curriculum L2469 + gauge L515-518)
· `dash_comb.py` header · git log (commit ordering v3 vs recess wave-1 vs chain-A terminus).

# SEAL ROUND 2 VERDICT — NOT CLEAN (findings ranked; dispositions verified first)

## §1 LENS 1 — ROUND-1 DISPOSITION VERIFY (re-executed, per item)

| item | verdict | evidence (what I re-did) |
|---|---|---|
| (a) two-surface share resolution (BLOCKER-1) | **HOLDS** | [re-executed] Surface A from `lane_share_probe_ep225_n600.json`: lane share_of_d_seg 0.19089, movable 0.44808 ✓ (v3 prints 0.1909/0.4481). Surface B from the composed memo/`maps_BEST_ep300.npz` table: lane 0.4396, movable 0.1226, big-3 = 1−0.4396−0.1226 = 0.4378 ✓. Bridging-law numbers: movable part_frac 0.01083/0.01109 = **97.7% ≈ "~98% birthed"** ✓; within_flip 0.05288 = 5.3% ✓; lane 36.478% GT px flip ✓; 22.5% mass deficit ✓. §0.0 residuals re-multiplied: 0.4396/0.1226/0.4378 × 0.0034 = 0.0014946/0.0004168/0.0014885, sum 0.0034 ✓. §3.2 lane-first follows: ratio 0.4396/0.1226 = **3.586 ≈ 3.59** ✓; w_movable = 0.1226/0.4396 = **0.2789 → 0.28** ✓. Pinned priors: ln(0.00577) = **−5.1552** ✓, ln(0.01109) = **−4.5017 → −4.50** ✓ (both from the same Surface-B artifact — req-H clean). |
| (b) τ* = m_q/ln5 derivation (MAJOR-3) | **HOLDS** | [re-executed] flip-support anchor verified in `birth_death_persistence_dseg_20260630T172510Z.md` L134/L196: per-pixel flip-rate **0.7645** for GT-margin ∈ [0, 0.10), ~0.000 every higher bin ⇒ m_q = 0.10 ✓. 0.10/1.60944 = **0.06213 → 0.062** ✓. Safe-median cross-check 0.416/1.60944 = **0.25848 → 0.258** ✓ (v2's 0.2 < 0.258 = safe-at-median, ratified). τ(ep300) path shape recomputed from the trainer's cosine (progress = (ep−1)/(_ae−1)): control (den 1000, end 0.05) = 0.05+0.95·0.5·(1+cos(0.2993π)) = **0.8052 → 0.805** ✓; v2 (den 600, end 0.2) = **0.6014 → 0.601** ✓; v3 (den 600, end 0.062) = 0.062+0.938·0.50125 = **0.5322 → 0.532** ✓. τ-confirm $0 probe present §7 ✓. Recovery split honest (β-leg DPR demotion + 3 confounds named + cosine_hold arm) ✓. |
| (c) TAU-window EMA-best restore law (MAJOR-2) | **HOLDS** | [re-executed] trace `levelset_train_result.json`: ep600 d_seg **0.0033716414**, ep625 **0.0033928596**, ep650 **0.0033661906**. Forfeits: ep625−ep650 = 2.6669e-5 → **+2.667e-3 S ≈ +2.7e-3 (1.5× margin)** ✓; ep600−ep650 = 5.4508e-6 → **+5.451e-4 S ≈ +5.4e-4** ✓. Law specified (restore θ*_TAU, momentum never-reset) ✓; cost folded into §9.1 event leg ✓; bet stated as a bet ✓. (Req-J precision: the table carries d_seg to 7 decimals ✓.) |
| (d) single KKT byte law (MAJOR-1) | **HOLDS with one arithmetic slip → MINOR-R2-5** | [re-executed] λ_bytes = 25/37,545,489 = **6.65860e-7 S/B** ✓ (v3 6.6586e-7 ✓). "0.002 S ≈ 3,004 B": 0.002/6.6586e-7 = 3,003.6 ✓. Old 0.002 pose threshold GONE from every operative row (grep: survives only inside DELETED-explanations §1.2/§5.0/self-review — correct usage) ✓. BUT the hood exemplar mis-computes: "the clamp's 8 bytes need only Δd_seg > 5.3e-9 to pay" — correct is 8×6.6586e-7 = 5.3269e-6 S ⇒ **Δd_seg > 5.33e-8** (v3 is 10× low). See MINOR-R2-5. |
| (e) chain-A fold faithful | **PARTIAL → MAJOR-R2-2** | [verified-by-inspection + git] TerminalSolve OUT of the stage graph ✓; sensor K≥32 + K-trend ✓; HOLD_STAGE_NEGATIVE_CURVATURE disarmed ✓; lane-anisotropy scope sentence §2.3(6) ✓; instrument bounds carried ✓; run-2 SOLVE spec + Fisher-metric TR ✓. BUT the fold is faithful to the RECOVERED LINK-5, whose K=128 number was superseded by the measured TERMINUS (commit 42fa00812, 20:36 — 5 min after v3 completed): final ratio **0.163** (strict kill <0.1 NOT reached; middle zone; persist >0.5 decisively excluded; full-P extrapolation ≈0.08 DERIVED). v3 cites "K=128 ratio 0.011 ≪ kill band" in FOUR places incl. the I-6 registry spec. The terminal DISPOSITION survives (the conjunction — 1/√K collapse + holdout ±1.2 transfer + isotropy + every step measured non-descending — is the dispositive evidence, as the terminus itself adjudicates), but the numeric basis is falsified and I-6 as written would register a wrong number. |
| (f) six P5 PARTIALs amended | **5 of 6 HOLD; F4/P11′ → BLOCKER-R2-1** | P5-2 anneal-speed confound NAMED §2.2d + B9 PREFERRED + B1-contingency tag §8 ✓. P5-3 three-leg label + two-leg row 125,955/0.0839 printed ✓ ([re-executed] all §5.1 sums with +8: 93,092 → 0.061986; 128,384 → 0.085482; 125,955 → 0.083865; 115,285 → 0.076763; [70,400, 103,521] → [0.046877, 0.068930] — every printed 4-decimal rate ✓). P5-4 twin mirror-schedule clause ✓ §6. P5-5 scope + registry tranche-2 routing ✓. P5-6 all 4 nits ✓ (rung-1 upper 0.0028 = 0.0023+0.0005 derived; comb 0-byte receipt; P7 concurrency; GPU-reorient caution). P5-1 (P11′ AA-aware gate): the REFERENCE is realized (a0b82ba6c) — but the referenced lane's MEASURED R3 verdict landed BEFORE v3 chunk 1 and REFUSES v3's shipped AA config; v3 did not fold it. See BLOCKER-R2-1. |
| (g) hood clamp +8 B · MUTCD comb period · hybrid orientation | **HOLDS** | +8 B booked in every §5.1 row (re-summed ✓, rates unchanged at 4 decimals ✓); paired-verdict gate + own req-H law + hood F-row §4 ✓. MUTCD period + ξ phase + 20-ep cosine engage ✓ §1.2 (comb module verified: ground-meter period/duty/phase, `--lane-band-dash-comb` + softness flag exist; period pinning is compress-time fit machinery — the MUTCD value is P1's null hypothesis, coherent). Hybrid orientation: R12 $0 probe §7 + fallback law (Δd_seg ≥ 0 OR poly-fit residual ⇒ self-orient retained) + honest lane-field-only scope ✓; 0.966-vs-0.893-0.909 flagged coordinator-supplied pending DAG verify ✓ (§9.2). |

**MINOR-1..13 + meat adoptions spot-verified:** 0.0573 leg deleted (§0.2 prints 0.0469 band edge
only) ✓ · Model A **1.6-6.4%** = (0.10·0.2·0.8, 0.20·0.4·0.8) [re-executed] ✓ · Model B per-axis
printed, product 0.85·0.15·0.25 = **3.19%** to 0.95·0.25·0.45 = **10.69%** → "3-11%" ✓, 8-15%
retained ONLY as the labeled run-1.5-branches judgment ✓ · §8 range 10-27% with derivation ✓ ·
≤10% edge (0.0011/0.0010 = 1.10) ✓ · anneal-wait restated as expectation ✓ · per-epoch slope
normalization ✓ · adaptive-ε law PRINTED with clamps + registry id (registry row re-read: form
matches `clamp(|c_a|·sqrt(eta·lambda_eik/8)·(1+margin), floor, upper)`, c_a measured 11.16 at ep1
gt_n6 — the "8" constant honestly FORMALIZATION_PENDING in the registry) ✓ · MINOR-11 checklist ✓
· chroma DPR-tag + pose-side stamp ✓ · NTK verified UNBUILT → §9.4 build-spec ✓ · kinematic-ξ +
comma2k19 init → P9 ✓ · #191/#203/#207/orbit receipts ✓ · L25 temporal-delta DROP ✓.

## §2 LENS 2 — FRESH HUNT ON THE v3 DELTAS

**(a) τ_end=0.062 × anneal-600 × adaptive-ε:** completion mechanics are τ_end-INDEPENDENT
(absolute-epoch progress; certified round 1) — no break. The CFL interaction is subtler: the
derivation memo's OWN coupling (L162/L178: β/τ descent pushes the annulus into the flat a<1
regime ⇒ **|c_a|(t) GROWS ⇒ ε_lower = |c_a|·√(η·λ_eik/8) RISES** — "lowers BOTH CFL ceilings
exactly as |c_a| rises") means v3's printed "known behavior: with |c_a| ~ 10 the clamp binds
(adaptive rarely fires)" is calibrated on the OLD τ path and is stale under a 3.2×-sharper
endpoint. [re-executed] with |c_a|=11.16, λ_eik=0.10, η~2e-3: raw ε = 11.16·√(2.5e-5)·1.5 =
0.0837 → FLOOR-clamped 0.3, stable (0.3 > ε_lower 0.0558). The self-deriving form protects the
run up to |c_a| ≈ 0.7/(√(η·λ_eik/8)·1.5) ≈ **93** (upper-clamp saturation ⇒ π_eik > 1 ⇒
tangential instability); beyond that only the spike/liveness guards catch it. Not a design break
(the adaptive law is exactly the right machine for this) — but the row's "rarely fires" sentence
should be replaced by the τ-coupled statement + an upper-clamp-saturation ALARM row (req-F class).
→ MINOR-R2-6.

**(b) lane-first rebuild vs round-1-certified laws:** CLEAN. [verified-by-inspection] eased
dilation 1-Lipschitz + 275-completion-gates-CE-exit kept verbatim (movable row); stagger
275/275/boundary+50 inherited; ≤2 new loss-geometry levers; per-class λ gates + interference
guards (comb OFF until P1, chroma at tau-fire, nucleus π₁≳5, ep150 lane ALARM, ep0 movable ABORT)
all present; band start=350 > CE cap 300 with 20-ep cosine follower — no ordering violation. No
certified item touched (crossing digits, §5.1 sums, absolute-epoch semantics, AA ordering — all
re-verified above).

**(c) Model A/B arithmetic:** ✓ (see §1 spot-verify). Labels honest.

**(d) NEW naive-collapse/toy-shapes introduced by the revision:** ONE found — the MAJOR-5 fix
specifies lever forms that DO NOT EXIST on any build surface (MAJOR-R2-3 below). Otherwise the
v3 deltas are law-shaped (derived endpoint, restore law, joint KKT, gated hybrid) — no new
binary-collapse instances (the comb engage ramp, per-epoch slope norm, and per-class veto each
REMOVE one).

**(e) requirement sweep A-K on the v3 text:**
- A ✓ (§11 rows 11-13 terminal — but row 13's "K=128 ratio 0.011" number rides MAJOR-R2-2).
- B ✓ (per-class veto + per-epoch norm + B9-PREFERRED + honest per-trigger status + B1-contingency).
- C ✓ (LADDER kept; lane-first re-founding preserves every mechanism).
- D ✓ (concurrency sentence; PowerPlay-consistent ordering).
- E ✓ (receipts: orbit −8 B best-arm, #207 R-all-pass with the L2-phase surviving sibling, L25 +64%).
- F ✓ new rows (hood F-row, chroma-engage stamp, K-trend spectrum discipline) — but req-F #6
  CKPT FIDELITY (persist self-orient) remains buried in the pooled "F1-F12 ~475 LOC" row at
  "strongly-wanted", NOT margin-critical → req-J(3) finding (MINOR-R2-7).
- G ✓ (I-6 updated set — but carries the falsified 0.011; fold with MAJOR-R2-2).
- H ✓ (ONE share model + surfaces printed; hood own law; priors pinned to one artifact).
- I **PARTIAL**: geometry-sharing headline realized (ONE polynomial → band + comb phase + lane
  orientation; ξ → warp + pose + comb + dash phase) ✓; but the meat-hunt §C3 scorecard's named gap
  "AA×island-survival attribution row (per-class AA paired deltas at stage boundaries)" was NOT
  added (grep: absent) — the one §C3 item the disposition table did not carry (MINOR-R2-8).
- J (NEW, binding): precision audit PASSES on the main chains (d_seg 7dp in §2.2c; bytes to the
  byte; S at 5dp in §0.2; no rounded intermediate feeds a downstream number — §0.3's 0.00150-class
  roundings are display-only, §0.0 carries the exact chain). Violations: the hood 5.3e-9 slip
  (MINOR-R2-5) + the un-ranked reconstruction-gap fix (MINOR-R2-7). "Δτ=0.016 ≈ nil" is a τ-unit
  input claim (acceptable); chain-A "below the gap, hence unattributable" is stated as ATTRIBUTION
  honesty, not insignificance — compliant with J(3) as written.
- K (NEW, binding) — per-technique audit: comb NATIVE+ADEQUATE (modulation carrier; ground-meter
  period, RANGE-DEPENDENT by construction; 3-scalar+phase capacity is the structural optimum;
  softness ramp) · screw-ξ pose NATIVE+ADEQUATE (q swept at its OWN optimum under the joint law;
  kinematic 2-3-DOF prior arm) · polynomial band NATIVE+ADEQUATE (v_h pinned; LBND2/4/smoothed
  swept, R1-measured) · hood clamp NATIVE+ADEQUATE (8 B static) · curvelet bulk NATIVE, capacity
  CONDITIONALLY adequate: along=8 is justified ONLY in the lane_offloaded regime (band trained-with
  ⇒ Candès-Donoho along-optimum ≈ 6; the DSL Rebalance factory itself brands across=32/along=8
  "BACKWARDS" in the lane-carried regime) — v3 never STATES the regime justification; one sentence
  owed (fold into MINOR-R2-8's req-K sweep) · islands **APPARATUS-FINDING** (MAJOR-R2-3: per-class
  weight has no format to live in) · AA **CAPACITY/APPARATUS-FINDING** (BLOCKER-R2-1: the shipped
  form is refused by its own gate; ipe is the surviving native form) · self-orient
  **APPARATUS-FINDING** (state not checkpoint-persisted, +4.3% gap = req-K(3)'s named violation;
  = MINOR-R2-7's build).

**(f) launch-readiness:** flags spot-checked [re-executed] against the trainer argparse + DSL:
`--anneal-epochs` ✓ · `--softmax-temp-end` ✓ (the REAL τ_end flag; DSL gauge L516 emits it) ·
`--tau-anneal-shape`/`--tau-hold-frac` ✓ · `--muon-warm-start-momentum`/`--muon-lr-final-frac` ✓ ·
`--lane-render-band`/`--lane-band-start-epoch` ✓ · `--lane-band-dash-comb`(+softness) ✓ ·
`--render-aa`/`--aa-supersample` ✓ · `--logit-adjust-loss-tau` (+L890 micro-batch fail-close) ✓ ·
`--eikonal-visco-ca-pairs` ✓ · `--amplify-weight`/`--amplify-persist`/`--persistence-loss-weight`
✓ exist but are SINGLE-FLOAT/POOLED (MAJOR-R2-3) · **`--tau-anneal-end` DOES NOT EXIST**
(MINOR-R2-4). Governed launcher + memory-preflight + per-stage EMA ckpts in the §7 RUN row ✓
(inherited, P5-evidenced) — but the memory-preflight VERDICT at the real config is REFUSE
(BLOCKER-R2-1). Req-B per-trigger honesty ✓ (B1 owed + contingency tag; T-1..3 owed; caps).

**(g) req-M audit (operator addendum, arrived mid-round — v4 fold items):**
- **M(1) capacity humility: LARGELY SATISFIED, one flag.** Capacity-bearing choices are
  measured-anchored (mod32 = the control-PROVEN train-side capacity; band coeffs LBND2/4/smoothed
  SWEPT + R1-measured; pose q SWEPT under the joint law; waterfill = measured compress-small on a
  generous train-side base — the train-big-compress-small hedge is structurally present via
  §5.0/P4 even though v3 never names the principle). Knobs with single values carry measure paths:
  amplify w=0.28 recalibrates in-flight from the first F-row readout (class (e)); mod-dim 2-point
  named run-2; chroma 0.1 DPR-tagged with anchor-cite plan. The flag: §0.3's per-class residual
  bands are derivation/judgment capacity claims (honestly labeled, kill-thresholded) and the
  along=8 justification is derivation-only in exactly the sense M(1) warns about — already
  MINOR-R2-8(ii); the DSL factory's own ASSUMED_AWAITING_VERIFICATION on the √-optimum is the
  right humility, v3 just doesn't surface it.
- **M(2) self-deriving/feedforward control: TWO purely-reactive laws where a forecast model
  exists.** (i) adaptive-ε feeds back on CURRENT |c_a| only, yet the dominant c_a driver is the
  KNOWN τ(t)/β(t) schedule — a feedforward term ε_ff(t) = ε(ĉ_a(τ(t))) is derivable offline from
  the derivation memo's own coupling (fold with MINOR-R2-6: the τ-coupled restatement + the
  saturation alarm ARE the MPC term). (ii) the TAU→FIN co-predicate fires on a TRAILING slope
  (reactive) while the F3 online-meat AIC mixture fit (req-F #3, LB) is precisely a remaining-meat
  FORECAST — a forecast-consuming co-predicate (fire when predicted remaining meat < the forfeit
  it would recover) would shrink the §2.2c forfeit the same way the event-adaptive cadence does,
  at zero verdict cost. B5's powerlaw-meat run-end exit shows the forecast form is already the
  house style; the TAU→FIN trigger just doesn't consume it. → MINOR-R2-9.
- **M(3) response surfaces as DECIDE-layer models: SEEDS PRESENT, not named.** P9's q-sweep, the
  waterfill marginal curve (the KKT stop rule IS a response surface read), the mid-λ arm, and the
  F-row in-flight recalibration are exactly the #170 minimal seeds — v3 presents them as gates/
  diagnostics, never as persistent DECIDE-layer models the controller keeps. One labeling
  sentence + routing the fitted (bytes, d_pose)(q) and (bits, d_seg) waterfill curves into the
  costate SENSE store closes it (fold into MINOR-R2-9).

## §3 FINDINGS (ranked)

**[BLOCKER-R2-1] v3 ships an AA config its OWN launch gate measurably REFUSES — the landed recess
R3 verdict is unfolded.** Commit order: R3 (647201389, incl. the completed P11′ gate a0b82ba6c)
landed BEFORE v3 chunk 1 (e65cfdf76), yet v3 §1.1 ships `AACoverageRender(ss=2)` from ep0 and §7
treats P11′ as pending. R3 MEASURED [re-read]: ARM-PRIMARY worst case (ndf=4, fine-mode full)
projects **105.90 GiB > 89.6 REFUSE (rc=3)**; ndf=2/full = 87.91 GiB — SAFE by only **1.7 GiB <
the 10 GiB assumed-margin ⇒ honest margin-REFUSE**; batch mode memory-SAFE but wall-clock-killed
(~29 s/ep EDT thrash); the surviving AA form is **`--render-aa ipe`** (~0 memory/compute, wired).
R3 also surfaces the trainer Wave-D header: supersample is train-only + decode-budget-disqualified
+ **measured −49% witness-harm** — a measured NEGATIVE against the shipped lever, un-cited by v3.
Consequences: the launch plan as written is not launchable (the P0 machine-crash gate fires by
design); the lane/big-3 "AA sub-px" rows in §0.3's design residual bands cite a lever whose
shipped form is dead. Fix (v4): re-adjudicate AA-IN against ipe (and/or ndf=2 after the reconcile
ledger measures a smaller p95 spike margin) at recess close, per R3's own pre-registered kill law
("REFUSE ⇒ AA → run-2 with measured cost written" — the cost IS now written); re-scope §0.3's AA
citations and §8's s/ep base to the surviving form. Also fold R1: §5.1 still prints LBND2 41,562
(measured 41,526 — F16ii resolved) and band tail 18,000 vs the MEASURED coder-min 18,832
(win9, roundtrip-exact; 832 B = 5.5e-4 S = **31% of the crossing margin** on that tail edge —
req-J denominated).

**[MAJOR-R2-2] Four stale "K=128 ratio 0.011" citations vs the measured chain-A TERMINUS
(0.163).** The terminus (42fa00812, 5 min after v3 completed) replaces the recovery-written 0.011
with the converged measurement: ratio **0.163** — strict kill (<0.1) NOT formally reached, middle
zone; persist (>0.5) decisively excluded; full-P=600 extrapolation ≈0.08 (labeled DERIVED). v3
§2.3(2), §2.3(7), §11 row 13, and the I-6 registry spec all carry 0.011; I-6 as written would
register a falsified number into `tac.canonical_equations` (the exact recovery-written-verdict
class L81 exists for — the recovered LINK-5 was flagged fresh-eyes-unreviewed, and indeed its
number did not survive). The terminal DISPOSITION (TerminalSolve OUT; wall = basis; sensor
K≥32-disciplined; HOLD disarmed) SURVIVES on the conjunction evidence — v3's architecture is
unchanged; the numeric basis + registry spec must be corrected to the terminus framing
("kill-middle-zone; conjunction-dispositive; extrapolation DERIVED") before tranche-2.

**[MAJOR-R2-3] MAJOR-5's fix specifies lever forms with NO build surface — per-class amplify
weights and per-island 1/pers weights are unbuildable as written.** [re-executed] DSL
`AmplifyIsland(weight: float)` → single `--amplify-weight` float; trainer applies ONE pooled
scalar × a per-PIXEL `island_persistence_weight` over the COMBINED any_mask (kinds:
uniform | inverse_thickness only — 1/(1+EDT-depth), mean-1 normalized; NOT the per-island
w_i ∝ 1/pers_i clamped [0.25,4] from CacheGtSkeleton persistence PAIRS that §0.3b specifies).
`PersistenceTopology(weight: float)` → single `--persistence-loss-weight`. v3 §1.1 passes
`weight={"lane": 1.0, "movable": 0.28}` (a dict into a float param — `prog.validate()` cannot
hold it) and §0.3b's fallback "ship per-CLASS weights only" is ALSO unbuilt. §10 has NO build item
for either form (B9-B12/I-6 don't cover it). Net: run-1 as buildable ships the POOLED weight —
exactly the MAJOR-5 defect the disposition claims fixed (movable over-spend at 0.1226 share; the
w=0.28 law exists on paper only). Fix (v4, small): a B13 build item (~30-60 LOC: per-class
`--amplify-weight-lane/--amplify-weight-movable` or dict flag + factory params + the R13-gated
per-island kind as a third `--amplify-persist` kind), OR the honest re-disposition (pooled w=1.0
run-1 with the per-class law as a NAMED run-2 build — matching how per-class τ_c was handled).
This is the config-orphan / never-invent-lever-params class (triality: the DSL must HOLD the
lever before the draft may ship it).

**[MINOR-R2-4] `--tau-anneal-end` is an invented flag name.** The trainer flag is
`--softmax-temp-end` (argparse L7365; the DSL gauge emits it at L516). v3 §1.2's ★★ row header
names a nonexistent flag. The DSL compile path is correct, so this is doc-surface only — but a
by-hand launcher would die on it. One-token fix.

**[MINOR-R2-5] Hood-clamp pay threshold off by 10×.** §5.0: "the clamp's 8 bytes need only
Δd_seg > 5.3e-9 to pay" — correct: 8 × 6.6586e-7 = 5.3269e-6 S ⇒ **Δd_seg > 5.33e-8** (or state
it as 5.33e-6 S = 0.30% of the margin — req-J(4) same-currency). No wrong admission results (the
operative gate is the stricter paired-verdict), but the number is the joint law's own exemplar.

**[MINOR-R2-6] Adaptive-ε "clamp binds / rarely fires" is stale under τ_end=0.062.** The
derivation memo's own coupling says |c_a| GROWS as τ descends ⇒ the adaptive path is MORE likely
to fire late — that is the law working, not a defect; but v3 asserts the old-path behavior while
adopting a 3.2×-sharper endpoint in the same revision. Replace the sentence with the τ-coupled
statement + add an upper-clamp-saturation alarm (ε_raw > 0.7 sustained ⇒ π_eik > 1 risk; with
current constants that is |c_a| ≳ 93 vs measured 11.2) as a req-F row.

**[MINOR-R2-7] (req J(3)/K(3)) The +4.3% reconstruction-gap fix (persist self-orient state,
req-F #6) is not ranked margin-critical.** It sits inside the pooled "F1-F12 ~475 LOC /
strongly-wanted" row. Per the now-binding req J: it gates ATTRIBUTION of every win smaller than
~0.015 S — i.e. every win that matters in a 0.00178-margin campaign (the chain-A terminus itself
could not attribute sub-gap effects). Promote to a named LB-class build (or explicitly rank it
with a reason it can wait past run-1's byte-close attribution needs).

**[MINOR-R2-8] Req-I/K completeness residue:** (i) the meat §C3 "AA×island-survival per-class
paired-delta attribution row" was never added to §4 (the one scorecard item the disposition table
missed); (ii) the along=8 capacity choice lacks its one-sentence lane_offloaded-regime
justification (the DSL's own Rebalance factory brands along=8 BACKWARDS in the lane-carried
regime — v3 should state why band-trained-with puts run-1 in the offloaded regime, and what
flips if P1 fails).

**[MINOR-R2-9] (req M(2)/(3)) Two reactive-only control laws where a forecast model exists, and
response-surface seeds unnamed as DECIDE models.** (i) TAU→FIN fires on a trailing slope while
the LB F3 online-meat forecast exists — add the forecast-consuming co-predicate form (or name it
run-2 with the forfeit-shrink rationale); (ii) adaptive-ε lacks its derivable τ(t)-feedforward
term (fold with MINOR-R2-6); (iii) label P9's q-curve + the waterfill marginal curve + the F-row
recalibration as persistent DECIDE-layer response surfaces (costate SENSE store), not one-shot
gates. All small; see §2(g).

## §4 WHAT CERTIFIES (carried forward — do not re-litigate in round 3)

Two-surface resolution + bridging law (artifact-exact) · τ*-law arithmetic + path-shape triple
(0.805/0.601/0.532) · transition-law forfeit table (trace-exact to 7dp) · λ_bytes + every §5.1
sum/rate at 4 decimals (+8 rows re-summed) · crossing table all 5 rows · Model A/B products ·
§9.1 rung-1 upper derivation · lane-first rebuild preserves every certified curriculum law ·
per-class veto + per-epoch norm + B9-PREFERRED · twin mirror-schedule · hybrid-orientation gate +
fallback · MUTCD comb law + 0-byte receipt · pinned priors (recomputed) · 18/18 inherited flags +
9/11 new-row flags real (the 2 exceptions = MINOR-R2-4 + MAJOR-R2-3).

## §5 VERDICT + COUNTER

**NOT CLEAN — 1 BLOCKER-grade (unfolded measured R3 refusal of the shipped AA form) · 2 MAJOR
(stale 0.011 → registry poisoning risk; MAJOR-5 fix unbuildable as written) · 6 MINOR (R2-4..9).**
Counter: **0 of 3** (reset). Nothing found invalidates the vehicle, the two-surface model, the
τ*-law, the transition law, the joint KKT law, or the crossing arithmetic — every fix is a
document/law/build-item-level fold plus one config re-adjudication (AA→ipe or measured-margin
ndf=2) that R3's own pre-registered kill law already prescribes. v4 should be a small revision.

Pointer 0.19110 UNMOVED — this verdict is MEANS.
