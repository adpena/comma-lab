---
doc_type: t5_crucible_synthesis_draft_v6_launch_candidate
role: v6 SYNTHESIZER (T5 crucible; ALL probe gates resolved — this fold turns the v5 design into
  the launch-candidate draft; NOT a redesign: v5's architecture, crossing-arithmetic FORM,
  lane-first curriculum, 19-row signal ledger, and all v4/v5 fold resolutions are carried forward
  WHOLE except where a resolved gate or measured amendment below explicitly amends them)
date: 2026-07-08
supersedes: DRAFT_OPTIMAL_STACK_v5_20260707.md (b241cf466) — v5 preserved append-only. v6 folds
  (1:1, §0.1): probe_waveA_ct_schedule_20260708.md (P-CT3 PASS · P-CT1 ν refit · P-CT2 band-fail
  + seam · τ-confirm PARTIAL) + probe_tau_confirm_ep1000_20260708.md (the m_q=0.10 anchor =
  apparatus artifact; τ*_end 0.062 REVOKED) + probe_waveB_geometry_class_20260708.md (Q1 BETWEEN ·
  P-CON KILL@FORMULATION → fitted bar · P-DZ FIRES · P-MP KILL@FORMULATION · FEED-08l
  upheld-with-correction · comb 79.33) + deterministic_gpu_accum_348_20260707.md (F-DET GO) +
  seal_round1_v5_verdict_20260707.md (9 editorial minors — no v5.1 errata file exists; all 9
  incorporated here).
epistemic_contract: unchanged — every knob carries a CONTROL LAW class {(a) CONSTANT · (b)
  RAMP/ANNEAL+completion guarantee · (c) SELF-DERIVING · (d) EVENT-CONDITIONED · (e)
  FRACTIONAL/PARTIAL} + a tag {V-S · V-A · D · DPR}. Every load-bearing number labeled
  MEASURED / DERIVED / INFERRED / ASSUMED. Nothing unmeasured asserted as measured (NO-FAKE).
  Precision per req J: d_seg ≥ 5 dp, d_pose ≥ 6 dp, bytes exact, S ≥ 6 dp in chains; smallness
  claims denominated in the 0.00178 S crossing margin.
axis: all numbers [macOS-CPU/MLX advisory] unless tagged; archive bytes exact (zip stat).
  Pointer contest-CPU 0.19110 UNMOVED — this whole file is MEANS.
review_status: pre-registered-only (v6.2; seal counter 0/3 after the round-2 NOT-CLEAN reset — rounds restart on THIS revision; the round-3 re-verify re-runs the launcher dry-run against --config crucible_v6)
verdict_scope_discipline: every negative below carries its req-R scope tag
v6.1_errata: seal-r1 MINOR-1 (census vehicle named + 0.2180 alternative floor) + MINOR-2 (width label 0.4989) applied — no decision, consumer-read number, or build item changed
v6.2_fold: seal-round-2 verdict (1 BLOCKER + 2 MAJOR + 3 MINOR) + the P-TAU2/P-DITHER gate resolutions folded — SUBSTANTIVE (build items + consumer-read claims change); see §14 "v6.2 changelog". The BLOCKER fix is CODE (src/tac/witness_autoconfig.py derive_crucible_v6_config + tools/launch_witness_run.py --config crucible_v6), PROVEN by re-running the round-2 dry-run at n600/3000ep against the new config (launch.sh tokens verified; test-guarded in src/tac/tests/test_witness_autoconfig.py::test_crucible_v6_schedule_matches_design_doc)
verdict_scope: formulation — the negatives THIS draft asserts are all formulation-scoped at their
  source probes (P-CON kill = pers>τ·ln5 AS-FORMULATED, fitted bar is the live form; P-MP dead =
  K≤64 concave-max selection AT the annulus, capacity passes; Q1 "at chance" = the POOLED-unsigned
  statistic, per-direction mechanism confirmed real; τ*=0.062 struck = INSTANCE, corrupted input).
  Family-level claims appear ONLY in the asymptote section with their deciding measurements named.
  Remaining uppercase tokens quote source-scoped probe verdicts. # VERDICT_SCOPE_OK:quotes-of-source-scoped-verdicts-plus-the-per-item-tags-below
  {INSTANCE | FORMULATION | FAMILY | PARADIGM}; kills enumerate reformulation queues.
---

STORES CONSULTED: ORCHESTRATION_LEDGER.md (full — reqs A–S binding; ALL landing folds from
"#348 GO" downward = the fold spec, resolved 1:1 in §0.1) · DRAFT_OPTIMAL_STACK_v5_20260707.md
(full — the base, carried whole) · seal_round1_v5_verdict_20260707.md (full — the 9 minors;
verified NO v5.1 errata file exists in the crucible dir, so all 9 fold here) ·
probe_waveA_ct_schedule_20260708.md (full) · probe_waveB_geometry_class_20260708.md (full) ·
probe_tau_confirm_ep1000_20260708.md (full) · .omx/research/deterministic_gpu_accum_348_20260707.md
(full) · .omx/research/dseg_side_feasibility_corners_verdict_20260619.md (the #149 build-state
check — corpus_query "sub-pixel boundary placement camera resolution 149 closed-form") · trainer
argparse RE-GREPPED this session (experiments/train_levelset_witness_realized_through_R_mlx.py:
`--fused-r-kernel` L7577 verified + its `--mlx-device gpu` requirement L6088; per-flag counts for
every flag named below, incl. the corrected persistence spellings; NO dither flag exists — 0
matches) · arithmetic re-executed unrounded this session (crossing chain, ν laws, τ* table, M1
term — printed in §0.2/§0.3). NOT consulted: durable-state files (stale per sweep); CT-1/CT-2
full re-reads (carried via v5; their derivations remain fresh-research-round-1 pending seal
lens-B); no training launched, no live config touched ($0 reading + arithmetic + grep only).

# T5 CRUCIBLE — DRAFT v6: THE LAUNCH-CANDIDATE STACK (all gates resolved: forfeit arm FIRES · ν laws recomputed · τ* re-derived from intent · B17 fitted bar · #149 graduates with a run-1 gated lever · F-DET determinism in-config)

## §0.1 — THE FOLD, 1:1 WITH THE 14-ITEM CHARTER LIST

| # | fold item | resolution in v6 |
|---|---|---|
| 1 | **F-DET** `--fused-r-kernel` in launch config | §1.1 + B-DET: flag VERIFIED (trainer L7577, BooleanOptionalAction default False; REQUIRES `--mlx-device gpu`, L6088 ValueError). MEASURED: 0/28 tensors diverged cross-process N=10 (Muon arm 0/28 N=5), **−8% wall-clock** (25.35s→23.44s 200-ep smoke — determinism is FASTER), 25/25 numpy-authority parity. Pre-GO verification = **B-DET**: n600/self-orient COMPOSITE determinism check riding the launch preflight (2 short resumed segments at the EXACT launch config, cross-process hash compare; #348 memo's own reactivation clause). DSL leg: existing #252 lever, already DSL-held — no new factory. CPU (numpy-fp32) stays the verdict/score authority regardless (L53) |
| 2 | **ν-law amendments** (P-CT1) | §2.2g: registered ν = 0.026210 NOT REPRODUCIBLE from the trace (suspected rel-vs-abs units mix at the "3.3e-3" input — measured single-cadence slope at ep350 = 1.4812e-3, not 3.3e-3). Adopted per-stage MEASURED fits: **ν(CE) = 0.019955 · ν(tau) = 0.012653 (binding) · ν(muon_fin) = 0.003289** /ep. Every 3/ν constant recomputed (§2.2g table): settle **237.1 ep** (was 115) · TAIL cycle floor **387.1 ep** (was 265) · dwell_TAIL ≥ **237.1** · s\* = ν·forfeit = **6.8971e-6 S/ep** (artifact full-precision; recompute at 4-dp ν gives 6.8969e-6 — display-precision only) · k_max = floor(2350/387.1) = **6** (turnpike "3–7" survives at its edge). V/window law: RE-DERIVED, not V-grown — see §2.2g(c). P-CT3's promotion SURVIVES the ν dispute: fire epoch INVARIANT across s\* ∈ [6.9e-6, 1.42e-5] because the slope crosses ZERO between ep650 and ep675 (probe §1, cited) |
| 3 | **P-CT3 promotion executes** | §2.2f: PASS (first sustained fire ep675 ∈ band [670,700] both estimators; EMA-best-at-fire = ep650 = the TAU-stage true best, forfeit EXACTLY 0; +5.450779e-4 S recovery vs the shipped ep625 fire VERIFIED full-precision). The forfeit-matched arm is now **the FIRING arm**; remaining req-B bind = the INJECTION test (B-INJ, pre-GO — not a $0 trace probe); the shipped slope arm stays the tested FALLBACK; fail-safe cap 726 unchanged |
| 4 | **Cadence antagonism** | §2.5: B-CT3 stays UNBUILT (P-CT2 band-fail: 5/41 skipped vs band 12–17; kill not triggered; no floor_S in ×1–×6 reaches the band — the savings materialize only in exhausted-tail regimes, run-2's TAIL_k). BINDING CONJUNCT registered on any future cadence law: **cadence never stretches while any exit co-predicate is within one decision band of firing** (the measured seam: composed with the promoted forfeit arm, the replayed cadence law skips ep650 and hands back the ENTIRE +5.450779e-4 S) |
| 5 | **τ\* RE-DERIVATION** (the big one) | §1.4a: 0.062 STRUCK (the m_q = 0.10 anchor was TAUTOLOGICAL — binned on the maps-npz `gt_margin` key = SIGNED witness margin-toward-GT, ≤ 0 at flips by definition, max −1.12e-5; bit-reproduced 0.7644972239; apparatus-vs-anchor, scope INSTANCE, law form τ\* = m_q/ln5 untouched). Consistency row (a) STRUCK as CIRCULAR (§0.4). Convention DERIVED from the law's original intent (Maslov error budget) → FIXED-POINT form with f_target DEFERRED to a named probe; launch value = CONSTANT 0.31 (the measured-best anchor) + live-law promotion path. Anneal-"TRUNCATED" narrative re-examined honestly: the τ-leg INVERTS (evidence supports over-descent past the optimum); the β-leg stands. Full cascade (9+1 c(τ) rows, adaptive-ε, TAIL ladder τ_k = max(τ_{k−1}/2, τ\*_k-from-live-m_q), F12 τ-samples re-centered) in §1.4a-cascade |
| 6 | **B17 fitted form** | §3.4: certificate = **ABSOLUTE persistence bar** (fitted s: thresholds **1.7504924172 logit** @Tau-stage maps / **1.3017706202 logit** @MuonBest — near τ-INDEPENDENT while τ varied 4.3×; raw pers > τ·ln5 certifies only 0.4408/0.5636, KILL scope=FORMULATION, band ≥ 0.95). Death alarm semantics unchanged. COHERENCE NOTE: P-CON's absolute bar + τ-CONFIRM's τ·ln5-anchor collapse are the SAME finding from two independent probes — the τ·ln5 scaling does not govern the real margin/persistence field; absolute logit scale does |
| 7 | **B16** | §1.3: stays DEFAULT-OFF (Q1 BETWEEN: max \|ρ\| = 0.1242 < 0.3 fire bar; 3 sides ≥ 0.1 block robust-dead). REGROUNDED on the measured opposite-signs mechanism (raw-texture ρ: Lane→Road erasure **+0.1346** high-texture · Movable→Road **−0.1027** low-texture — pooled −0.033 averaged away opposite signs exactly as the shape-gradient theorem predicted) at **40% of the fire bar**; the struck circular row-(a) grounding REMOVED. Duty-to-measure kept with the sharpened re-test spec (hinge / per-range / rank-ρ / winner-side-logit reformulations). NEW PHYSICS ROW (SC-20): annulus flips are **MARGIN-driven, not reachability-driven** (S_R positive control AT CHANCE max \|ρ_sR\| = 0.0808 while the margin control passes −0.24…−0.38 all sides) — consumers named in §4c |
| 8 | **#149 GRADUATION** (P-DZ fired 88.7×) | §0.3 + §5/B19: deadzone flip mass = **1.5795495775e-3 d_seg-equivalent** (H1 variant 1.4556778802e-3), **38.366% of subset flip mass** (= 43.70% of the n600 decoded residual 0.0036146) sub-quantum — far-range lane rows 176–224, horizon shadows, hood boundary. Build-state checked (corpus): #149 = a COMPLETED closed-form $0 probe on the #155 flat-paint vehicle (probe_dseg_side_feasibility_corners.py; mechanism REAL — 12× boundary-flip win — RED only as a #155 replacement); NO render-side lever exists in the witness trainer (grep: 0 dither flags). **DECISION: RUN-1, gated** — a decode-side deterministic dither lever is ~small LOC and $0-gateable (B19 + P-DITHER, §7c). Family-asymptote M1 term computed in §0.3; smooth-reachable vs quantum-locked decomposition enters the d_seg budget |
| 9 | **P-MP** | §11 row 16 + §12: KILL scope=FORMULATION (max-of-K≤64-concave-quadratics at the annulus: agreement plateaus ~0.41 vs band 0.95). SHARPENED AUTOPSY: **oracle-SELECTION capacity PASSES** (K=64 rms 0.0761–0.1306 logit ≈ the 0.0998-class coupling bound) — the max-ENVELOPE selection mechanism binds (envelope rms 2.30–58.18 logit), NOT representation richness; **K=64 payload = 2,304,000 B = 1.5341 S — rate-dead by 862× the margin regardless**. Max-plus stays in solve-inventory (band-residual K=1 specials UNAFFECTED) with the RANKED reformulation queue: log-sum-exp at finite τ · tropical rational · per-class; larger K is NOT the fix |
| 10 | **Corrections** | comb gap-FP removal = **79.3283%** (0.0017417→0.0003600), NOT 86% — cited as 79.33 everywhere in v6; the 86-vs-79 discrepancy sits in the SOLID-BASELINE construction between probes (cCOMB's own gap_FP reproduces FEED-08c c3 EXACTLY at 0.000360) → NAMED CHECK: the comb-registration audit's step-0 pins ONE solid baseline before the number is cited again (§7c). FEED-08l = UPHELD-WITH-EVIDENCE-CORRECTION: the scoreability column was mis-mapped (range bands, not freq rungs) — ALL FIVE rungs d_seg-scoreable at n600 ⇒ the flat-ladder claim is now 5-point = STRONGER; the flatness is a COMPENSATED TRADE (contrast-closure and gap-FP improve monotonically with f_along, recall pays it back) — req-K confirmation: the dash wants the comb's REGISTRATION, not its spectrum |
| 11 | **Signal ledger updates** | §4c: SC-3 extended (quantile-convention OWNER: the f_target fixed-point + q̂ decision live here) · SC-16 status → **SEEDED** (g_I edge-contrast histograms EMITTED: experiments/results/t5_probe_waveB_20260708/pdz_deadzone_census.json + .gi_hists.npz, 128 bins/pair) · NEW SC-20 (margin-vs-reachability flip-driver row) · NEW SC-21 (GPU-determinism composite check row). Ledger = **21 rows**; zero no-signal gap terms RE-VERIFIED (§4c-end) |
| 12 | **Crossing arithmetic** | §0.2: re-executed unrounded this session — NO probe fold moves rate/pose/g_dec or any crossing number; central S = **0.1897336** (margin 0.0013664), win9 S = **0.1817034** (margin 0.0093966); train bars 0.0010137/0.0010940; ILC chain 0.0011 − Δ̂ = **9.9573e-4** unchanged. What CHANGES: the d_seg TRAJECTORY expectations (settle 237 not 115; fire band ep675; TAIL floor 387) and the honesty layer — the M1 census places the train bar BELOW the measured quantum-locked mass (§0.3): crossing explicitly REQUIRES the large-amplitude/dither levers, not smooth-INR descent alone |
| 13 | **9 round-1 minors + flag-reality pass** | §13: all nine incorporated (A1 persistence spellings · A2 M3 x% pinned computed-from-SC-18 · B1 B-CT2 disposition line · B2 B16×AmplifyIsland seam law · B3 10th c(τ) row ca-band · R-1 two-τ retirement conjunct · R-2 §12 scope tags · R-3 P-MP disposition wording · nits incl. the F12→Q2-τ label split). Every flag named in v6 re-grepped this session against the live argparse — zero invented flags; `--fused-r-kernel` spelling verified at L7577 |
| 14 | **Requirement-N section** | §0.3 (asymptote with the measured M1 term) + §9 (run-1 instrument framing + the honest statement of what this family can and cannot claim) |

## §0.2 — THE CROSSING ARITHMETIC (fold 12: re-executed unrounded this session; NO gate resolution moves it)

Condition (v4/v5 inherited): **100·(d_seg_train + g_dec) + √(10·d_pose) + rate < 0.19110**,
g_dec = +1.0427e-4 d_seg [MEASURED, R6 ep650/mod32cap; per-stage SC-7 rows re-measure].

- pose term: √(10 × 3e-5) = **0.0173205**.
- central rate: 93,092 B × 25/37,545,489 = **0.0619861**. win9 rate: 81,032 B → **0.0539559**.

| triple (d_seg_train, d_pose, rate) | S decoded | crosses 0.19110? |
|---|---|---|
| v3 triple (0.0011, 3e-5, central) | **0.1997336** | NO — over by 0.0086336 = 4.85× margin (stated plainly) |
| **v6 central (0.0010, 3e-5, central)** | **0.1897336** | **YES — margin 0.0013664** |
| **v6 + win9 arm (0.0010, 3e-5, win9)** | **0.1817034** | **YES — margin 0.0093966** |
| win9 at the old target (0.0011, 3e-5, win9) | 0.1917034 | NO — over by 0.0006034 |

Required train-side bars (re-derived): central ≤ **0.0010137**; win9 ≤ **0.0010940**. ILC-formal
bar chain UNCHANGED: 0.0011 − Δ̂₀ = 0.0011 − 1.0427e-4 = **9.9573e-4 ≈ 0.0010** (consistency
row (c), holds). None of folds 1–11 enters this arithmetic: τ\*/ν/B17/B16/#149/P-MP are
schedule/curriculum/telemetry/lever surfaces — they move WHEN and HOW the trajectory reaches the
bar, not the bar. **What crosses, stated plainly:** the engineered gated tail (central needs every
class within ≤ 1.4% of its optimistic edge simultaneously; win9 admitted restores ≈ 9.4% headroom);
run-1's central expectation (≈ 0.26, v5 §9 inherited) does NOT cross. NEW binding-constraint
honesty from fold 8: see §0.3 — the 0.0010 bar sits BELOW the measured quantum-locked mass, so the
crossing tail now names THREE binding constraints: big-3 anneal-completion/endpoint-placement
recovery · lane composed-band efficacy · **locked-mass coverage by large-amplitude/dither levers**.

## §0.3 — REQUIREMENT-N: THE MEASURED M1 TERM + THE FAMILY ASYMPTOTE, REVISED (folds 8 + 14)

**The census (MEASURED, P-DZ, 96-frame subset @MuonBest, HR = 0.842 conservative form):**
deadzone (sub-quantum) flip mass = **1.5795495775e-3 d_seg-equivalent** (H1 variant
1.4556778802e-3) = **0.3836640693 of subset flip mass** (subset flip d_seg 0.0041170120)
= **0.4370 of the n600 decoded residual** (0.0036146, R6). Concentration: far-range lane rows
176–224 (Lane→Road 24.7% of pair flips) · horizon/shadow edges (Road→Undrivable 77.5%) · hood
boundary rows 224–384 (Road↔MyCar 67–85% — #139 clamp territory). Honest boundary carried from the
probe: this is an in-principle unreachability census on the CURRENT state (state-dependent;
GT-geometry estimator form; not a measured fix-yield).

**The decomposition row (enters the d_seg budget):** decoded residual 0.0036146 =
**smooth-reachable ≈ 2.0351e-3** + **quantum-locked ≈ 1.5795e-3** [MEASURED-at-state, subset; VEHICLE NOTE (seal-r1 MINOR-1): the P-DZ/P-CON census vehicle is the θ* per-stage-attribution run (2026-06-30), not mod32cap — under the alternative fraction-transfer the smooth-only floor is 0.2180, still > 0.19110, so every disposition stands; run-1's F-rows re-measure the fraction on its own vehicle].

**The M1 asymptote term (computed, as the fold demands):** if the census transfers to the design
endpoint, the SMOOTH-PERTURBATIVE-ONLY witness (INR corrections through R, no large-amplitude
analytic levers, no dither) has d_seg floor ≥ 1.5795e-3 ⇒ d_seg term ≥ 0.1579550 ⇒
**S_asymptote(smooth-perturbative-only) ≥ 0.2372616 central-rate / 0.2292313 win9** — ABOVE the
0.19110 frontier. Stated plainly: **the smooth-INR-correction mechanism ALONE cannot reach the
0.0010 train bar** (the bar is 1.6× BELOW the locked mass). This is the third independent
confirmation of "the wall is BASIS/representation" (chain-A curvature · FEED-08l format · M1
quantum-lock). It does NOT invalidate the crossing case, because the stack's crossing was never
smooth-only: the large-amplitude levers (analytic lane band · #139 hood clamp · island birth ·
comb) act at multi-quantum amplitude on MOST of the census's concentration regions — but NOT all
of them (the v6.1 "EXACTLY the concentration regions" sentence OVERSTATED horizon coverage;
corrected per seal-round-2 MAJOR-B1 with the $0-computed table below).

**◆ v6.2 (MAJOR-B1) — the locked-mass per-lever coverage table ($0, [re-executed] from
`experiments/results/t5_probe_waveB_20260708/pdz_deadzone_census.json`, HR form, fractions of the
1.5795e-3 locked mass):**

| lever family | class-pairs | locked share | locked d_seg |
|---|---|---:|---:|
| lane band + islands + comb | Lane↔Road + Lane↔\* | **42.7%** | 6.749e-4 |
| #139 hood clamp | Road↔MyCar (+MyCar↔\*) | **19.3%** | 3.042e-4 |
| islands w_movable | Movable↔Road/Undrivable | **12.9%** | 2.032e-4 |
| **NO run-1 lever** | **Road↔Undrivable (horizon/shadow)** | **25.3%** | **4.001e-4** |

The horizon/shadow quarter = **0.0400 S ≈ 22.5× the crossing margin**. ◆ v6.2 with P-DITHER's
KILL (§7c): B19 zeroth-order decode-side dither — previously the only (gated) lever nominally
attacking this quarter — is DEAD-AS-FORMULATED (measured Δd_seg = **+2.1277533637e-6 ≥ 0**, kill
bar fired ~10σ robust; churn ratio ≈ 0.98 = the locked sub-quantum residual carries ~zero net GT
information at this checkpoint), so the 25.3% is now **strictly leverless in run-1** (attacked
only by ChromaBoundarySharpen's small-amplitude g_I-raising mechanism, weight 0.1 — a
locked-mass lever via the HR·g_I·m/|∇m| predicate the earlier enumeration under-credited, but
NOT large-amplitude), **and the run-1-leverless share is ≥ 25.3%**: the "dither attacks whatever
they don't cover" backstop for the other 74.7% is gone too, so every SC-16-unmeasured uncovered
fraction of the band/clamp/island support masks adds to it. **Named follow-up (duty queue, ranks
ABOVE several queued reformulations at 22.5× margin):** an explicit Undrivable-boundary /
horizon-shadow lever question for run-2 (candidates: render-informed placement per #149's real
mechanism · trained-with dither · a horizon-band analytic prior), plus the SC-16 overlap
computation deciding the covered fractions.

**◆ v6.2 (MAJOR-B1) — crossing-case sensitivity (full precision, [re-executed]): does the
engineered case survive if the leverless 25.3% converts NOTHING?** Leverless non-conversion
consumes 4.001e-4 of the train-side d_seg budget (= 0.040010 S = 29.28× the central margin /
4.258× the win9 margin if it had to be recovered elsewhere — it cannot be). Remaining train
budget: central 1.0136635e-3 − 4.0010e-4 = **6.135635e-4** (60.5% of the bar); win9
1.0939661e-3 − 4.0010e-4 = **6.938661e-4** (63.4% of the bar). **Verdict: the engineered win9
case SURVIVES leverless-converts-nothing IFF the entire remaining residual (smooth descent +
covered-locked conversion by band/clamp/islands/comb) lands within 6.9387e-4** — a strictly
harder tail than v6.1 stated (the covered 74.7% must now convert through the large-amplitude
levers ALONE, no dither backstop). The central arm's equivalent bound is 6.1356e-4. Crossing
DISPOSITION unchanged (the tail was already conditional on coverage — this measures the
condition and tightens it).

**The composed-family asymptote estimate S ≈ 0.165, band [0.154, 0.181] (v5 §0) is RETAINED,
with its lower edge explicitly CONDITIONAL on locked-mass coverage — a condition now carrying a
measured NEGATIVE prior for its cheapest lever class (unbiased decode-side dither) and a
strictly-leverless 25.3% quarter:** the run-1 signal that decides it is the locked-mass ×
lever-support overlap (SC-16 g_I histograms × the band/clamp/island support masks — a named $0
computation on run-1 artifacts, consumer: run-2 asymptote row). T_3 = 0.15 remains at/beyond the
family's optimistic edge; the family step (quotient codec #155 / compress-half #336 /
0.0005-regime) remains named run-2+ work. Yield denomination (req J): even **1% recovery of the
census ≈ 1.5795e-3 S ≈ 0.89× the crossing margin**; full census = 0.158 S of headroom, two orders
above the margin.

**#149 run-1 disposition (fold 8 decision, ◆ v6.2 SUPERSEDED by the measured P-DITHER verdict):**
the 2026-06-19 probe proved the sub-pixel mechanism REAL (boundary flip 0.18→0.04, 12×) but its
#155 form stored camera-res band residuals (rate-dead) — that kill is scope=FORMULATION
(flat-paint vehicle), and the witness inherits only the MECHANISM. v6.0 shipped B19 (decode-side
ordered dither at the uint8 quantization; rule-118 free; byte-close-selectable) gated on P-DITHER.
◆ **P-DITHER has now RUN and the pre-registered KILL bar FIRED** (probe_tau2_dither_20260708.md
§B, n600 full-scale on the REAL byte-close decode, instrument-validated bit-for-bit against R6):
Δd_seg = **+2.1277533637e-6 ≥ 0** (fire bar −1e-5 not reached; ~10σ from firing under the
churn-null — no seed of this form plausibly fires). Mechanism: the dither REACHES the census
geometry (59% of its 20,421-pixel churn in the far-range-lane band) but is DIRECTION-BLIND
(fixed/created ratio ≈ 0.95–0.99 every low/mid margin band) — the locked sub-quantum residual
carries ~zero net GT information at this checkpoint, so unbiased decode-side noise randomizes
placement instead of recovering it. ⇒ **B19 is DEAD-AS-FORMULATED (scope=FORMULATION) and leaves
the run-1 build list**; the built instrument (`tac.witness_control.decode_dither` +
`tools/witness_dither_decode_ab.py`) stays in the toolbelt. NOT killed: the #149 render-informed
placement MECHANISM and trained-with dither — run-2 queue, now with measured priors AGAINST every
unbiased decode-side variant (amplitude/pattern/mask sweeps: per-band ratios ≤ 1 in every band a
mask would select) and FOR the render-informed forms. No launch-blocking dependency existed
either way.

## §0.4 — CROSS-FIELD CONSISTENCY ROWS (amended: row (a) STRUCK)

| row | status | content |
|---|---|---|
| (a) | **STRUCK — CIRCULAR** (fold 5) | "τ_end·ln5 = 0.0998 ≈ measured 0.10 flip-support edge" — BOTH sides consumed the same corrupted apparatus output (the tautological witness-margin binning). Not two fields agreeing; one artifact cited twice. The δ_τ = τ·ln5 width LAW survives as a law-form; its VALUE re-binds after the §1.4a convention decision |
| (b) | HOLDS (recomputed) | PMP ε_stop = 0.00178/25 = 7.12e-5 S/ep ≈ shipped eps_rel 6.8e-5 at the exhaustion operating point (within 5%; operating-point caveat carried) |
| (c) | HOLDS (recomputed) | ILC bar 0.0011 − 1.0427e-4 = 9.9573e-4 ≈ v4's independently-chosen 0.0010 |
| (d) | **NEW** | P-CON's fitted ABSOLUTE persistence bar (1.30–1.75 logit, τ-independent while τ varies 4.3×) + τ-CONFIRM's τ·ln5-anchor collapse = the SAME finding from two independent probes: the real field's survival/flip structure lives on an absolute logit scale, not on τ·ln5. One coherent physics statement replacing the struck row (a) |

---

## §1 — THE WitnessProgram (v6 deltas marked ◆; all v5 ★★★★ / v4 / v3 rows inherited unless amended)

### §1.0 FLAG-VERIFICATION AMENDMENTS (fold 13; re-grepped this session)

- ◆ CORRECTED SPELLINGS (MINOR-A1): the real flags are **`--persistence-warmup-epochs`** and
  **`--persistence-classes`** (grep-verified, 1 hit each); `--persistence-loss-weight` is real.
  v5 §0.1 row 10's "zero invented flags remain" amended accordingly; DSL factory already emits
  the true spellings — no launch artifact changed.
- ◆ VERIFIED NEW-TO-CONFIG: **`--fused-r-kernel`** (L7577, BooleanOptionalAction, default False —
  must be set explicitly) with its hard requirement **`--mlx-device gpu`** (L6088 raises
  ValueError otherwise). Launch posture: MLX-GPU training + CPU-authority verdicts (L53), so the
  requirement is satisfied by the existing config.
- ◆ NO dither flag exists (0 grep hits) — B19 is PROPOSED-NEW (decode-side, not a trainer flag).
- All other §1.0 verified-existing rows re-confirmed (spot-recount this session: `--anneal-epochs`
  · `--softmax-temp-end` · `--amplify-weight/-persist` · `--island-dilate-px` ·
  `--seed-island-eased` · `--seed-anneal-*` · `--margin-saliency-*` · `--eikonal-visco-eps-floor`
  / `-ca-band` · `--weight-entropy-penalty-lambda` · `--muon-start-epoch` ·
  `--stage-transition-rewarmup-*` · `--verdict-batch` · `--render-aa` ·
  `--curriculum-plateau-windows`). Line refs drift with file growth (round-1 nit ii) — counts,
  not line numbers, are the check.
- ◆ **v6.2 (BLOCKER-1) THE LAUNCH ROUTE IS NOW A NAMED CONFIG: `--config crucible_v6`**
  (`tools/launch_witness_run.py`; `src/tac/witness_autoconfig.py::derive_crucible_v6_config`,
  built on the store-nothing #205 base per the `store_nothing_variant` pattern). It pins AT THE
  SOURCE: `--softmax-temp-end 0.31` (no extras-route C13 collision — the round-2 REFUSAL is
  structurally gone) · ABSOLUTE stage anchors `--tau-softplus-start-epoch 300` /
  `--muon-start-epoch 726` (NOT family-scaled 0.726×epochs = 2178, the measured wrong emission) ·
  the EXPLICIT τ denominator `--anneal-epochs 3000` + `--tau-anneal-shape cosine_hold` +
  `--tau-hold-frac 0.2` (⇒ descent completes at ABSOLUTE ep600 = 0.2×3000 and HOLDS τ = 0.31
  through the fire band and the Muon freeze: τ(675) = τ(726) = 0.31 EXACTLY, simulated against
  the trainer's own law and cross-checked on mod32cap's measured τ(650) = 0.3098) ·
  `--fused-r-kernel` (F-DET) · the pinned pose block (next bullet) · the §1.1 lever pins
  (V=5 co-predicate, ChromaBoundarySharpen 0.1/1.0/start-300, AA ipe, band 350, persistence
  warmup 275, and the composable DSL levers). **Materialization note (derived, not guessed):
  the literal "`--anneal-epochs 600` + cosine" reading of the v6.1 prose does NOT realize the
  design — the trainer's cosine is UNCLAMPED past the denominator (prog_t = (ep−1)/(ae−1)), so
  τ REBOUNDS to 0.3363 at ep675 and freezes at 0.3826 at the 726 cap; `cosine_hold` at
  hold-frac 0.2 of an explicit den-3000 is the correct token set** (guard: the
  config-materialization test `test_crucible_v6_schedule_matches_design_doc` asserts the
  emitted-tokens→τ-law equivalence, with the round-2 wrong emissions pinned as anti-targets).
  PROOF: the round-2 dry-run invocation re-executed at n600/3000ep with `--config crucible_v6`
  → flag validation 103/103, NO C13 refusal, memory preflight 67.61 GiB PASS, launch.sh carries
  every token above.
- ◆ **v6.2 (MAJOR-A2) THE POSE BLOCK IS PINNED AT THE CONFIG SURFACE** (it was absent from
  v3→v6 flag tables — an inheritance five documents deep): the launch config emits
  `--w-pose 1.0 --pose-carrier --pose-carrier-residual-mode table --pose-carrier-source
  generated` (the store-nothing ξ carrier, Track 1 — the mover the crossing books at
  √(10·3e-5) = 0.0173205; `real_keyframe` is the EXCLUDED wrong mover, and the trainer default
  is `real_keyframe` + `w_pose 0.0` = pose-blind). **#314 (DAG DRIFT-D2, pose-carrier-source
  inheritance drift) is the named risk this pin mitigates**: crucible_v6 inherits the block
  STRUCTURALLY from `derive_store_nothing_205_config` and the regression test
  `test_crucible_v6_pose_block_pinned` turns any future inheritance revert into a test failure,
  not a silent pose-blind run.
- ◆ **v6.2 (MINOR-3) launcher-injected knob row**: the real launch argv ALSO carries
  `--per-group-grad-clip` — the launcher's C4 emit-side injection (landed f1dd0cc2f;
  score-affecting; trainer default OFF for byte-identity). Knob classification per the epistemic
  contract: **class (a) CONSTANT, tag V-S**; opt-out = `--no-per-group-grad-clip` on the
  launcher. Every knob table below is to be read WITH this row.

### 1.1 Program sketch (v6: two deltas — F-DET on; τ_end constant re-anchored)

```python
prog = WitnessProgram(
    purpose="T5 crucible ARM-PRIMARY v6: v5 + fused-R determinism + tau_end 0.31 + firing forfeit arm",
    base=Mod32SegOnlyControlBase(),
    curriculum=sealed_205_curriculum(cfg, handoff="event"),  # anneal-epochs 600;
                                                             # ◆ --softmax-temp-end 0.31 (§1.4a; was 0.062)
    levers=[
        FusedRKernel(),                                      # ◆ F-DET: --fused-r-kernel (requires mlx-device gpu);
                                                             #   0/28 cross-proc N=10 MEASURED; −8% wall-clock;
                                                             #   pre-GO gate = B-DET composite check
        SeedIslandBirth(), SeedIslandEased(release="r_star"),
        EventTriggeredCurriculum(),                          # B1 spec: V=5 (retained — §2.2g(c))
        LogitAdjust(tau=1.0),
        AmplifyIsland(form="hinge", weight={"lane": 1.0, "movable": 0.28}, gated="witness_alone"),
        PersistenceTopology(weight="1/pers clamped [0.25,4]", warmup=275),
        ConleyCertificate(threshold="fitted_absolute"),      # ◆ B17: s_fit per stage (1.750/1.302 logit)
                                                             #   + Delta_dec^logit; NOT tau*ln5 (fold 6)
        SignedBoundaryWeight(gated="Q1-sharpened"),          # B16: DEFAULT-OFF (Q1 BETWEEN); §1.3 amended
        CacheGtSkeleton(), LengthSigma("fitted-20260707"),
        AACoverageRender(mode="ipe"),
        AnalyticLaneRenderBand(start=350, boundary_relative=True, v_h=174),
        ChromaBoundarySharpen(weight=0.1, margin_band=1.0, start="tau_fire"),
        MuonWarmStart(lr_final_frac=0.1),                    # entry = TAU-window EMA-best; ◆ FIRING arm =
                                                             #   forfeit-matched s* (§2.2f); slope arm = fallback
        WeightEntropyPenaltyMLX(lam=15),                     # twin lam=0, mirror-schedule
        GNSpectrumProbe(k_pairs=">=32 + K-trend row"),
        # ◆ v6.2: B19 DitherDecodeSide REMOVED (P-DITHER kill fired: Δd_seg = +2.1278e-6 ≥ 0;
        #   dead-as-formulated, §0.3/§7c; render-informed forms = run-2 queue)
    ],
)
prog.validate()
```

◆ **v6.2 — this program MATERIALIZES as `--config crucible_v6`** (BLOCKER-1 fix; §1.0 bullet):
the pose block (store-nothing ξ carrier) + τ/Muon absolute schedule + F-DET + the 1:1 lever pins
are emitted by `derive_crucible_v6_config`; the composable §1.1 levers ride the DSL merge
(`SeedIslandBirth · SeedIslandEased · EventTriggeredCurriculum · LogitAdjust · AmplifyIsland ·
PersistenceTopology · CacheGtSkeleton · LengthSigma · MuonWarmStart · WeightEntropyPenaltyMLX`);
SignedBoundaryWeight stays DEFAULT-OFF (Q1 BETWEEN), ConleyCertificate-fitted (B17′) +
GNSpectrumProbe are telemetry/build surfaces, not launch flags. **Named OPEN base-delta question
for the seal round (stated, not silently decided):** `Mod32SegOnlyControlBase()` as a DSL factory
carries 8 control-vehicle deltas; the variant inherits 3 of them via the sealed base (mod-32 ·
verdict-pairs 0 · annealed hosc 1→4) and does NOT auto-compose the rest (eikonal-0 · freq-along-8
· n-dir-freqs-4 · lane-paint-off · l7@1001 — the last MISFIRES at 3000 ep: 1001 < 3000 would RUN
the demoted l7 defect stage; the others change the vehicle the ν/τ trace laws were measured on
vs the P3-sealed lever pins — a genuine authority conflict this fixer surfaces rather than
adjudicates). The one-line A/B lever exists: `--dsl-lever Mod32SegOnlyControlBase` composes them
at launch (minus an l7 re-pin).

### 1.3 ◆ B16 REGROUNDED (fold 7 + MINOR-R-1 + MINOR-B2)

- **Gate outcome:** Q1 = BETWEEN (no-fire: max \|ρ\| = 0.1242 < 0.3; no-kill: Lane→Road 0.1116 ·
  Movable→Road 0.1147 · Movable→Undrivable 0.1242 all ≥ 0.1). B16 stays **DEFAULT-OFF**,
  activation-ledger state: never-fired → Q1-adjudicated-no-fire. It does NOT enter the duty queue
  with a live prior; measured prior = ~0.12/0.3 = **40% of the fire bar**.
- **Grounding amended:** the row-(a) citation (τ_end·ln5 ≈ 0.10 edge) is REMOVED (struck,
  circular). The lever's grounding is now the MEASURED mechanism itself: per-direction effects
  with OPPOSITE SIGNS (Lane→Road erasure +0.1346 high-texture · Movable→Road −0.1027 low-texture)
  — the pooled −0.033 averaged away opposite-signed weak effects, exactly as the Hadamard
  shape-gradient theorem predicted. Real but weak at this formulation.
- **Weight-of-choice amended (SC-20 consequence):** v5's "S_R is the realized weight of choice"
  is WITHDRAWN — the S_R positive control is AT CHANCE within the annulus (max \|ρ_sR\| = 0.0808)
  while the margin control passes strongly (−0.24…−0.38, all sides): annulus flips are
  MARGIN-driven, not reachability-driven. The realized weighting keeps the δ_τ(m)·\|∇m\| margin
  factor (the Fisher surrogate, L1) and DROPS S_R as a flip-predictive multiplier. Scope:
  FORMULATION (S_R as point-predictor of realized flips within the annulus; S_R's role as a
  through-R attenuation BOUND is a different consumer, untouched). Retro-validates LEVER-4
  msal_uni inertness (L76).
- **Sharpened re-test spec (duty-to-measure):** reformulation queue, ranked: winner-side per-class
  LOGIT conditioning (needs SC-16's full form — most headroom) · hinge (one-sided) response ·
  per-range conditioning (the census showed per-range structure) · rank-ρ (Spearman). S_R-composited
  cost fields DEPRIORITIZED (SC-20).
- **MINOR-R-1 fixed:** retirement clause now reads: "retire-with-reason ONLY if \|ρ\| < 0.1 both
  sides every major pair at BOTH the coarse AND the fine-τ checkpoint (v4 §7b two-point protocol)."
  Q1's coarse-point BETWEEN triggers neither retirement nor fire.
- **MINOR-B2 fixed (seam law):** B16 and AmplifyIsland share the LEVER-4 signed-margin machinery.
  Composition law declared: if Q1's successor ever fires B16 while the island hinge is active,
  B16's σ_ij,dir applies ONLY OUTSIDE the island-support mask (island support takes precedence;
  no pixel carries two margin-band weights). The pre-registered A/B (signed-hinge vs unsigned
  margin-gate) covers the seam's attribution.

### 1.4a ◆ THE τ\* RE-DERIVATION (fold 5 — the big one)

**What died (scope=INSTANCE; the law form τ\* = m_q/ln5 untouched):** τ\*_end = 0.062 rested on
m_q = 0.10 from `birth_death_persistence_dseg_20260630`, which binned flips on the maps-npz
`gt_margin` key = the SIGNED witness margin-toward-GT (≤ 0 at every flip BY DEFINITION, measured
max over flips −1.12e-5) — so "all flip mass below 0.10" was a TAUTOLOGY true of ANY vehicle
(bit-reproduced 0.7644972239). Confound class: DEFAULT-HARMFUL × SILENT × MEASUREMENT-CORRUPTING
(misnamed npz key). The instrument now takes the margin axis from the GT cache by construction
(tools/witness_tau_mq_confirm.py); the npz-key rename is a named follow-up.

**The TRUE GT-margin field (MEASURED, 16-pair advisory, full table in the probe):**

| leg | τ(ckpt) | mass<0.10 | m_q50 | m_q80 | m_q90 |
|---|---|---|---|---|---|
| END ep1000 | 0.2157 | 0.298 | 0.19803 | 0.49485 | **0.74347** |
| BEST ep650 (EMA) | 0.3098 | 0.322 | 0.17943 | 0.44646 | **0.65607** |

Heavy-tailed to q99 ≈ 2.3–3.4 on EVERY leg (both vehicles, three epochs) — **the compact
"support edge" the law bound to does not exist on the real field**; the quantile convention is a
design decision, not a measurement.

**Derivation from the law's ORIGINAL INTENT (Maslov error budget):** softmax-τ approximates the
argmax with per-pixel selection error bounded by τ·ln5 (5 classes). A flip at GT-margin m is
τ-ATTRIBUTABLE (fixable by descent) only if m ≲ τ·ln5; flips at m ≫ τ·ln5 are structural
(capacity/geometry/quantum-locked — the M1 census lives in the SAME population). The anneal
endpoint should therefore sit where **further τ-descent stops converting flip mass**:

  **FIXED-POINT FORM: τ\* solves mass(m < τ\*·ln5) = f_target**, f_target = the τ-attributable
  fraction of flip mass.

f_target is NOT derivable from a static snapshot (it is a CONVERSION rate — how much of the
sub-τ·ln5 mass actually flips back under descent). **DEFERRED with a named probe** (P-TAU2, §7c):
measure converted flip mass between two live τ-samples on run-1 (the Q2-τ/F12 samples + SC-3 live
rows make this free); f_target = converted/(mass in the swept band).

**The launch value (V-A anchor, class (a) constant + promotion path):** the only MEASURED optimum
on this vehicle is the control's best checkpoint at **τ = 0.3098** — which sits INSIDE
[τ\*(q80), τ\*(q90)] of its OWN live field (0.27740–0.40764). After ep650 the control kept
descending (τ → 0.2157, BELOW its own q80 endpoint 0.30747) and d_seg ERODED (slope negative from
ep675; never re-beat ep650). ⇒ **`--softmax-temp-end 0.31`** ships as the anchored constant
(replaces 0.062), with the live-law promotion path: SC-3 emits τ\*(q̂, live) per verdict cadence
(q̂ = 0.85, midpoint of the measured-best bracket [q80, q90]) as a WOULD-BE row; run-2 promotes the
live law once P-TAU2 pins the LIVE f_target. Fail-safe/cap semantics unchanged (the descent-600
event-margin law untouched — a HIGHER endpoint only shortens the descent distance, so the
anneal-completion guarantee strengthens; ◆ v6.2 token materialization: `--anneal-epochs 3000` ×
`--tau-hold-frac 0.2` cosine_hold = descent completes ep600 ABSOLUTE + HOLDS 0.31, per the §1.0
BLOCKER-1 bullet — plain cosine at den 600 rebounds and is NOT the law). ◆ **v6.2 P-TAU2
CORROBORATION (measured 2026-07-08, probe_tau2_dither_20260708.md §A):** the knee-derived static
f_target = **0.861663 (ep650) / 0.862512 (ep1000)** — leg-stable to 3 dp and landing ON the
q̂ = 0.85 convention (independent derivation, Kneedle marginal-return-collapse criterion);
**0.31 STANDS** inside the knee band [0.190724, 0.542937] and the ep650 primary sensitivity
interval [0.288437, 0.431976]. The knee is NOT sharply localized (heavy tail, no crisp static
elbow) — independently CONFIRMING this section's decision that f_target must be the run-1 LIVE
conversion measurement (SC-3 as-is); the static knee mildly favors a slightly HIGHER endpoint
(primary knees 0.344–0.385), recorded as a directional note for SC-3's live law.

**The "anneal TRUNCATED" narrative, re-examined honestly (fold-5 charter question):** the control's
end τ 0.216 sits BELOW the new τ\*(q90) band 0.41–0.46 AND below its own-state q80 endpoint. The
ep650-best + post-650 erosion evidence supports **the τ-leg INVERTED**: the control did not stop
short of the τ-optimum — it descended THROUGH it (best at τ ≈ 0.31) and past it; the M2
"truncation" at 0.216 (vs the 0.05 target) likely SAVED it from worse. [INFERRED — the post-650
erosion is epoch-confounded (exhaustion/noise vs τ-causal); the Q2-τ samples + SC-3 adjudicate
causally on run-1.] The β-leg of M2 STANDS untouched (β frozen 3.177/4.00 = a genuinely incomplete
anneal on a separate axis), and M2's structural law — anneal-completion as a consumer PRECONDITION
with event-bound denominators — stands in full. What dies is only the assumption that completing
the τ-descent to a small endpoint was the withheld recovery.

**§1.4a-cascade (every τ-adjacent constant re-anchored; fold-5 cascade):**

| # | constant | v6 state |
|---|---|---|
| 1 | adaptive-ε FLOOR law | form unchanged; the fine-τ saturation concern RELAXES at τ_end 0.31 (the \|c_a(τ)\| growth that drove it is milder); saturation ALARM (ε_raw > 0.7 sustained) KEPT |
| 2 | adaptive-ε UPPER 0.7 | unchanged (measured-anchor); Q3 clamp-binding check now runs at the RE-CENTERED τ-samples |
| 3 | δ_τ width = τ·ln5 | law-form kept (exponent 1); at τ_end = 0.31: width = **0.4989 logit** (0.31·ln5; the 0.4986 print was 0.3098·ln5) (was 0.0998); no longer grounded by row (a) — grounded by the Maslov bound alone |
| 4 | island release r\*(t) = 0.95·σ_eff(t) | unchanged (its anchors are σ/dilation-knee measurements, not the m_q artifact) |
| 5 | island gate margin ∝ τ·ln5 | kept; value at τ_end updates with row 3 |
| 6 | c_cond | unchanged (first-run-measures) |
| 7 | ChromaBoundarySharpen margin_band 1.0 | unchanged constant; exponent decision at the re-centered samples |
| 8 | Conley threshold | **REPLACED by the fitted ABSOLUTE bar** (fold 6, §3.4) — no longer τ-indexed; the τ-scaling FAILED measurement |
| 9 | TAIL τ\*_k | **τ_k = max(τ_{k−1}/2, τ\*_k-from-LIVE-m_q(q̂))** — the live form is now the SAFER form given the frozen-anchor failure; q̂ inherits the §1.4a convention (0.85) until P-TAU2 pins f_target |
| 10 | ◆ `--eikonal-visco-ca-band` (MINOR-B3) | 10th enumeration row: default 0.0 (interior mean) = the symposium exact launch formula, measured-anchored at coarse τ; declared status = MEASURED-FLAT-PENDING at the re-centered τ-samples (interior-mean vs annulus-restricted c_a diverge as τ descends); SC-18 + the row-1 alarm instrument the consequence |
| — | Q2-τ/F12 dash-contrast samples | RE-CENTERED: **{0.31, 0.216, 0.12}** (was {0.216, 0.12, 0.062}) — spans the now-defensible band incl. the launch endpoint and the control's realized end; ALSO the label split (round-1 nit iv): the τ-sample probe is renamed **Q2-τ**; F12 = stage wall-clock only |

---

## §2 — THE SCHEDULE (v6: the firing forfeit arm; recomputed ν laws; cadence conjunct)

### 2.2f ◆ THE FORFEIT-MATCHED TAU→FIN ARM — PROMOTED TO FIRING ARM (fold 3)

P-CT3 PASS (both estimator forms agree): first sustained fire **ep675** ∈ band [670, 700], no
flapping; EMA-best-at-fire = **ep650 = the TAU-stage true best exactly** (forfeit 0); recovery vs
the shipped ep625 fire = **+5.450779e-4 S**, VERIFIED at full precision from the trace (fire@625
v2-as-written forfeits +2.666897e-3; with the v3 restore-EMA-best law +5.450779e-4; fire@650/675
forfeits EXACTLY 0). **The arm now FIRES** (entry = TAU-window EMA-best restore, unchanged), with:
- s\* = ν·forfeit at the AMENDED ν(tau) = 0.012653 ⇒ **s\* = 6.8971e-6 S/ep** (artifact precision).
- **Fire epoch INVARIANT to the ν dispute** (probe §1, cited per the fold charter): the trace
  slope crosses ZERO between ep650 and ep675, so any s\* ∈ [6.9e-6, 1.42e-5] fires at ep675.
- **Remaining req-B bind: the INJECTION test** (B-INJ, pre-GO build: synthetic
  fires-when-it-should + stays-silent-when-it-shouldn't through the LIVE witness_control wiring).
  Until B-INJ passes, the launch config carries the arm armed-with-fallback:
- **The shipped slope arm stays the tested FALLBACK** (not deleted); fail-safe cap 726 unchanged;
  anneal-completion precondition unchanged (fires only post-anneal-complete).

### 2.2g ◆ THE ν-LAW AMENDMENTS (fold 2)

(a) **Registered ν = 0.026210 is DEAD** — not reproducible from the trace (its "3.3e-3 S/ep at
ep350" input does not exist there; measured 1.4812e-3; the 3.3e-3 magnitude matches the costate
RELATIVE-slope rows ⇒ suspected rel-vs-abs units mix — the MINOR-9 bug class, audit-grade
provenance flag carried). Scope: FORMULATION (single-ν exponential per stage remains
AIC-preferred in all three stages; the VALUE moved). Adopted per-stage fits:
**ν(CE) = 0.019955 · ν(tau_softplus) = 0.012653 · ν(muon_fin) = 0.003289** /ep.

(b) **Every 3/ν constant recomputed** (at the binding ν(tau) = 0.012653):

| law | v5 value | **v6 value** |
|---|---|---|
| settle 3/ν | 115 ep | **237.1 ep** |
| TAIL cycle floor (settle + 150) | 265 ep | **387.1 ep** |
| dwell_TAIL ≥ | 115 ep | **237.1 ep** |
| s\* = ν·forfeit | 1.4154e-5 S/ep | **6.8971e-6 S/ep** |
| k_max after ep650 in 3000 ep | 3–7 | ◆ v6.2 (seal-r2 MINOR-1): **≤ 6 gross = floor(2350/387.1) / ≤ 5 net of FIN dwell = floor((2350 − 250 FIN min-stage)/387.1)** — gross-6 booked ZERO epochs to FIN between fire (~675) and TAIL_1; turnpike "3–7" still survives; TAIL stays τ\*-limited not budget-limited; crossing books no TAIL gain — no decision changes |
| LPV ramp floor | ≥ 115 ep | **≥ 237 ep** (the 20-ep band-engage ramp's rate ratio (1/ν)/20 moves 1.9× → **3.95×** the physics time constant — MORE marginal; flagged, measured-good at the coarse point, unchanged pending run-1 F3 per-class ν_c fits) |

(c) **The co-predicate window: V = 5 RETAINED; the window LAW re-derived (the fold's "state
which").** The v5 rationale (window ≥ settle, two-timescale) would now demand V ≈ 10–11
(237.1/25 = 9.48 cadences) — a 250-ep trailing window on a ~425-ep stage, absurdly sluggish. The
MEASUREMENT says the requirement was wrong: P-CT3's fire epoch is INVARIANT across the full s\*
dispute range because the trigger is a **zero-crossing detector near exhaustion, not a ν
estimator** — window-covers-settle is REFUTED-AS-NECESSARY for the FIRING trigger
[scope=FORMULATION of the window law; measured on this trace]. Re-scoped law: **V = 5 stands for
the trigger** (margin over the bit-reproduced V=4 behavior); the settle-coverage requirement
binds only the ν-ESTIMATION consumers (F3/SC-9 tail fits use windows ≥ 237 ep; B1 spec carries
both, no silent recalibration).

(d) Trajectory expectations (fold 12): TAU-stage meat has a LONGER tail than v5 assumed (settle
237 vs 115) — the event exits, not the clock, absorb this (the design is already event-bound);
TAIL cycles budget 387-ep floors with k_max ≤ 5 net of FIN dwell (v6.2); muon_fin's own ν = 0.003289 < 0.01 KILLS its
window laws (scope=FORMULATION) — FIN-stage exit laws ride the measured F3 fits on run-1, with
the fail-safe caps carrying until then.

### 2.5 ◆ CADENCE: B-CT3 STAYS UNBUILT + THE ANTAGONISM CONJUNCT (fold 4)

P-CT2 BAND-FAIL (5/41 skipped vs band 12–17; kill NOT triggered — zero missed bests; no floor_S
×1–×6 reaches the band: this trace's Muon stage RESTARTS descent so the 25-ep floor re-binds).
B-CT3 stays unbuilt; re-probe on run-1's trace once TAIL cycles exist. **BINDING on any future
cadence law (the seam finding, req I):** composed with the now-FIRING forfeit arm, the replayed
cadence law skips ep650 (the TAU best) and hands back the ENTIRE +5.450779e-4 S the arm exists to
recover — antagonistic as formulated. Registered conjunct: **cadence never stretches while any
exit co-predicate is within one decision band of firing** (fix (a) from the probe — the clean fix,
since restore-best selection NEEDS the verdict).

§2.1 / 2.2e / 2.4 / 2.6 / 2.3 inherited from v5 with the §2.2g value substitutions (TAIL budget
law now 387.1/237.1/k_max ≤ 6 gross · ≤ 5 net of FIN dwell (v6.2); dwell check for switching:
worst μ = 1.275 ⇒ τ_d > ln(1.275)/0.012653 ≈ 19.2 ep at the amended ν; shipped min-stage 250 =
13× margin — still SATISFIED).

---

## §3 — CURRICULUM (v6: B17 ships the FITTED ABSOLUTE bar)

### 3.4 ◆ THE CONLEY CERTIFICATE — FITTED FORM (fold 6)

P-CON verdict: **KILL, scope=FORMULATION** — the raw threshold pers > τ_k·ln5 certifies only
**0.4408189379** (Tau maps, τ = 0.0500065329) / **0.5635593220** (MuonBest, τ = 0.2156894835)
survival vs band ≥ 0.95. The discriminative signal is REAL (cert-vs-uncert separates 5.2×/5.1×;
pixel-weighted certified survival 0.9998 — the failures are tiny lane-dash islands; **Lane is the
entire failure**: P(s\|cert) 0.3056/0.4081 on 2691 lane islands, all other classes 0.81–1.00).
**B17 ships the FITTED ABSOLUTE form:**

    island I survives stage k AND decode ⟸ pers(I) > s_fit(stage) + Δ_dec^logit
    s_fit = 1.7504924172 logit (Tau-stage fit, s = 21.75, P(s|cert) = 0.9573901465)
          / 1.3017706202 logit (MuonBest fit, s = 3.75, P(s|cert) = 0.9505783386)

— near **τ-INDEPENDENT** (thresholds move 1.35× while τ varies 4.3×): survival behaves as an
ABSOLUTE persistence bar ~1.3–1.75 logit. The τ-SCALING of the law failed, not just its constant
(consistency row (d)). Consequences: (i) DEATH-ALARM semantics unchanged (certified-death =
controller/instrument failure) but computed on the fitted bar — the raw τ·ln5 form would
false-alarm ~50% of lane islands; (ii) NEW per-class row: lane islands get their own certificate
column (per-class s is first in the reformulation queue: fitted s per stage · per-class
thresholds · Δ_dec^logit once SC-7 measures it · size-weighted); (iii) born-to-die accounting +
release coupling (B18) unchanged in form, thresholds re-pointed at s_fit. The B17 backtest
artifact (3454-island ledger ×2 stages, pcon_conley_backtest.json + .ledger.npz) IS the SC-17 row
format, already emitted once.

§3.1–3.3, 3.5 inherited unchanged (per-class weights w_lane 1.0 / w_movable 0.28; birth-scheduler
record-only; LPV ramp floor updated per §2.2g).

---

## §4 — COSTATE + TELEMETRY (fold 11)

§4a inherited; SC-9's shadow arms updated: the forfeit-matched arm is now the FIRING arm (its
would-fire row becomes the fire-audit row); slope arm + forecast arm + PMP ε_stop remain shadows.

### 4c ◆ SIGNAL-COMPLETENESS LEDGER — v6 amendments (21 rows)

| ID | v6 change |
|---|---|
| SC-3 ★ext | now the **quantile-convention OWNER**: emits live m_q(q̂ = 0.85) + the full quantile vector per verdict cadence; consumers add: τ_end live-law promotion (§1.4a) + P-TAU2's f_target fixed-point measurement + TAIL τ\*_k (live form, §1.4a row 9) |
| SC-7 | unchanged form; Δ_dec^logit now ALSO feeds B17's fitted bar (s_fit + Δ_dec^logit) |
| SC-16 ★SEEDED | status "once built" → **seed data EXISTS**: per-pair g_I edge-contrast histograms (128 bins) EMITTED at experiments/results/t5_probe_waveB_20260708/pdz_deadzone_census.json + .gi_hists.npz; consumers add: the locked-mass × lever-support overlap computation (§0.3) + P-DITHER band setting |
| SC-17 | B17 row format now = the emitted pcon backtest ledger (3454 islands × 2 stages); fitted-bar column added |
| ◆ SC-20 NEW | **annulus flip-driver decomposition** (margin-vs-reachability-vs-texture, per-side): the measured row = S_R at chance (0.0808) / margin passes (−0.24…−0.38) / texture opposite-signed (+0.1346/−0.1027). Generated: $0 cached-field re-run per stage boundary (the Q1 correlator instrument, committed). Consumers: margin-saliency weight-of-choice (§1.3 — margin factor kept, S_R dropped) · B16 reformulation adjudication · L76 retro-validation ledger |
| ◆ SC-21 NEW | **GPU-determinism composite check**: 2-process cross-hash at the exact launch config (n600/self-orient ON), recorded in the launch-preflight artifact. Consumers: B-DET GO gate (fused-R reliance for bit-exact proof paths) · crash-resume/byte-close proof routing (CPU-locked vs GPU-with-fused-R per the relaxed discipline) · any future lever adding scatter/gather-VJP re-triggers the probe |

**Zero-gap re-check (fold 11):** every §0.0a bound still mapped (M1→SC-16 now SEEDED, M2→Q2-τ,
M3→SC-18 with x pinned computed-from-SC-18 per MINOR-A2, M4→SC-10, M5→SC-19); d_seg/d_pose/rate/
decode/schedule/capacity/basis/noise axes unchanged from v5's check; the two NEW rows carry named
consumers (no write-only telemetry). **Zero gap terms without a generated-recorded-consumed row.**

---

## §5 — RATE PLAN — inherited UNCHANGED (central 93,092 B → 0.0619861 · win9 81,032 B → 0.0539559
· hood clamp gate 5.32688e-6 S · decoded-KKT selection · Δ_dec^logit emission). One addition:

◆ **B19 — decode-side deterministic dither — v6.2: DEAD-AS-FORMULATED (P-DITHER kill fired).**
The v6.0 spec (seeded ordered dither at the uint8 quantization; 0 archive bytes; byte-close-
selectable; gated on P-DITHER) was probed at n600 full-scale on the REAL byte-close decode
(probe_tau2_dither_20260708.md §B): **Δd_seg = +2.1277533637e-6 ≥ 0** — the pre-registered kill
bar fires, ~10σ robust to seed under the churn null. The mechanism answer is itself the measured
census input: churn REACHES the locked geometry (59% in the far-range-lane band) but is
DIRECTION-BLIND (fixed/created ≈ 0.98) — the sub-quantum residual carries ~zero net GT
information at this checkpoint, so unbiased decode-side perturbation cannot mine it. B19 leaves
the run-1 byte-close-selectable set; the instrument stays in the toolbelt
(`tac.witness_control.decode_dither`, `tools/witness_dither_decode_ab.py` — OFF-identical
proven). Run-2 queue (measured priors attached): render-informed placement (#149's real 12×
mechanism) + trained-with dither FOR; every unbiased decode-side variant (amplitude / pattern /
locked-geometry-masked) AGAINST (per-band ratios ≤ 1 in every band a mask would select).

---

## §7 — MEASUREMENT PLAN (v6: gates RESOLVED; new probes)

### §7c — probe ledger: RESOLVED verdicts + the v6 additions

| probe | verdict (scope) | disposition executed in v6 |
|---|---|---|
| P-CT3 | **PASS** | forfeit arm = FIRING arm (§2.2f); B-INJ injection test = pre-GO |
| P-CT1 | BAND-FAIL (CE marginal, tau) / **KILL muon_fin window laws** (FORMULATION) | ν laws recomputed (§2.2g); registered 0.026210 dead; F3 per-class ν_c fits re-derive on run-1 |
| P-CT2 | BAND-FAIL, kill not triggered (FORMULATION) | B-CT3 unbuilt; stage-best-protection conjunct registered (§2.5) |
| τ-CONFIRM (+ ep1000 follow-up) | 0.062 REVOKED (INSTANCE; anchor = apparatus artifact) | τ_end = 0.31 anchored constant + live-law path (§1.4a); row (a) struck |
| P-CON | **KILL raw τ·ln5 form** (FORMULATION) | B17 ships fitted absolute bar (§3.4) |
| P-DZ | **FIRES** 88.7× band (census, no kill) | #149 → duty queue + B19/P-DITHER run-1 gated lever; M1 term computed (§0.3) |
| Q1 | **BETWEEN** (FORMULATION non-fire) | B16 default-off, regrounded, sharpened re-test (§1.3); SC-20 row born |
| P-MP | **KILL K≤64 concave-max form** (FORMULATION — this expansion form stays unadmitted; richer tropical forms re-enter only with their own probe [MINOR-R-3 wording]) | §11 row 16 autopsy sharpened; band-residual K=1 specials unaffected |

### ◆ New pre-registered probes (all $0-class)

| # | probe | band · kill |
|---|---|---|
| P-TAU2 | ◆ v6.2 STATIC HALF RESOLVED (probe_tau2_dither_20260708.md §A): knee-derived static f_target = **0.861663/0.862512** (both legs) ≈ q̂ = 0.85 — **0.31 STANDS** (inside knee band [0.1907, 0.5429] + the ep650 primary interval); no crisp static elbow ⇒ the LIVE conversion-rate measurement (Q2-τ samples {0.31, 0.216, 0.12} / SC-3 rows) remains the run-1 law source | reporting probe (derives a constant); τ_end live-law promotes ONLY after the LIVE f_target lands; fail-safe constant 0.31 stands regardless |
| P-DITHER | ◆ v6.2 RESOLVED — **KILL fired** (probe §B, n600 real byte-close decode, instrument-validated bit-for-bit vs R6): Δd_seg = **+2.1277533637e-6 ≥ 0**, ~10σ from the fire bar; churn direction-blind (ratio ≈ 0.98) | fire bar −1e-5 NOT reached; kill bar Δ ≥ 0 FIRED ⇒ B19 dead-as-formulated (scope=FORMULATION); trained-with dither + render-informed placement = run-2 reformulations with measured priors |
| B-DET check | 2-process composite determinism at the exact launch config (n600, self-orient ON, fused-R ON) | pass: 0 diverged tensors ⇒ GPU bit-identity usable for proof paths; fail: fused-R stays (throughput + parity-gated) but proofs remain CPU-locked; launch NOT blocked either way |
| comb-reg step-0 | pin ONE solid baseline (re-render solid under the ladder-probe construction), then the phase-sweep + registration score (4–20m bands only) | pre-comb-inclusion gate (the comb is P1-conditional anyway); until it passes, the citable comb gap-FP number is **79.33%** (fold 10) |

Q2-τ / Q3 / R12 / R13 / P8/F8 / P11′ inherited (Q2-τ samples re-centered per §1.4a-cascade).

---

## §9 — PREDICTED S LADDER + WHAT THIS FAMILY CAN CLAIM (fold 14)

Inherited: central ≈ 0.26 does NOT cross (stated plainly); crossing = the engineered gated tail
(§0.2); dual probability model (independent 2–6% / with-repair 8–15% incl. run-1.5 branch).
◆ Amendments: (i) the three named binding constraints now include **locked-mass coverage** (§0.3);
(ii) the anneal-recovery constraint's τ-leg re-signs per §1.4a (recovery is END-PLACEMENT, not
completion-to-small-τ; the β-leg unchanged); (iii) run-1 REMAINS primarily an instrument (EVSI
≈ 0.05 S of decision value, pose row 0.044 dominant — v5 §0.0c unchanged), now with THREE
gate-resolved instruments already banked pre-launch (trace probes · margin-quantile · deadzone
census — req Q paying rent before the run). **Honest family statement (◆ v6.2, post P-DITHER):**
this composed-lever family (mod32 witness + band/clamp/islands/comb; the dither-class decode
repair leg is DEAD-as-formulated and re-pointed at RENDER-INFORMED forms, run-2) claims the
[0.154, 0.181] asymptote band CONDITIONAL on locked-mass coverage — a condition now measured
tighter: the horizon/shadow 25.3% (4.001e-4 = 22.5× margin) is strictly leverless in run-1 and
the covered 74.7% has no dither backstop (§0.3 table + sensitivity); smooth-perturbative-only is
floored at ≥ 0.237 by the measured census (§0.3) and cannot cross; T_3 = 0.15 requires the family
step (quotient codec / compress-half / 0.0005-regime) regardless of run-1's outcome.

---

## §10 — BUILD LIST (v6 deltas; all v5 rows stand unless amended)

| id | build | ~LOC | status/route |
|---|---|---:|---|
| ◆◆ **B-CFG (v6.2, seal-r2 BLOCKER-1)** | the **crucible-v6 named config**: `witness_autoconfig.derive_crucible_v6_config` (store-nothing base + τ_end 0.31 + ABSOLUTE anchors tau@300/Muon-726 + explicit anneal den 3000 × hold 0.2 + F-DET + pose block + §1.1 lever pins + DSL-lever composition) + launcher `--config crucible_v6` choices entry + append-not-clobber `--dsl-lever` + activation-ledger rows for config-composed levers | **~140 built** (was claimed "0 (config)" — the round-2 under-scope: NO existing config surface held the v6 values) | **LANDED + PROVEN**: round-2 dry-run re-executed at n600/3000ep → 103/103 flags, no C13 refusal, mem-preflight 67.61 GiB PASS, all tokens in launch.sh; class-guarded by `test_crucible_v6_*` (schedule-law equivalence + pose pin + no-dup + byte-stability of the sister configs) |
| ◆ **F-DET** | `--fused-r-kernel` in the launch config (+ `--mlx-device gpu` requirement satisfied) | 0 (rides B-CFG) | measured-dominant (0/28 N=10, −8% wall-clock, 25/25 parity); DSL #252 lever already held; ◆ v6.2: emitted by crucible_v6 |
| ◆ **B-DET** | n600/self-orient composite determinism check riding launch preflight (2 short resumed segments, cross-process hash) | ~15 | pre-GO; SC-21 row; non-blocking either way (§7c) |
| ◆ **B-INJ** | injection test for the FIRING forfeit arm through live witness_control wiring (fires-when-should + silent-when-shouldn't) | ~20 | pre-GO (the remaining req-B bind on fold 3); until pass, slope-arm fallback carries (crucible_v6: cap 726 + V=5 event co-predicate) |
| ◆ **B17′** | Conley certificate FITTED-BAR amendment (s_fit per stage + per-class lane column; death alarm on fitted bar) | ~10 on B17 | P-CON-resolved; backtest ledger already emitted |
| ◆◆ **B19** | ~~decode-side deterministic dither~~ — **v6.2: REMOVED from run-1 (P-DITHER kill fired: Δd_seg +2.1278e-6 ≥ 0, §0.3/§5/§7c)** | built (instrument stays: decode_dither + witness_dither_decode_ab) | DEAD-with-autopsy (scope=FORMULATION); render-informed forms = run-2 with measured priors |
| ◆◆ **F26 (v6.2, seal-r2 MINOR-2) SC-3-ext live-m_q route** | in-trainer per-verdict-cadence wrap of `tac.witness_annulus_metrics.flip_margin_quantiles` in the levelset trainer's verdict block (emit m_q{50,80,90} + τ\*(q̂) WOULD-BE rows into the verdict telemetry; SC-20's instrument-row pattern) | ~15 | named route for the SC-3-ext "live m_q per verdict cadence" claim; UNTIL it lands, run-1 is honestly CHECKPOINT-granularity via the committed offline instrument (`tools/witness_tau_mq_confirm.py` — the pattern P-TAU2 just exercised); fail-safes (constant 0.31; τ_{k−1}/2) carry regardless |
| B-CT1→FIRING | forfeit-matched arm s\* = 6.8971e-6 S/ep (ν-amended) | done (~10) | PROMOTED (P-CT3 PASS); B-INJ owed |
| B-CT3 | self-triggered cadence | ~15 | STAYS UNBUILT (P-CT2); future spec carries the §2.5 conjunct |
| B16 | signed σ_ij,dir slots | ~25 | stays gated (Q1 BETWEEN); sharpened re-test spec §1.3 |
| ◆ B-CT2 disposition (MINOR-B1) | evaluate adaptive-ε at τ(t+H): **SUBSUMED by the inherited ε_ff(t) = ε(ĉ_a(τ(t))) internal-model form; residual one-horizon delta queued BEHIND Q3** (one line, recorded) | 0 | resolved |
| F18–F23 + ◆ F24/F25 | the six v5 SC rows + SC-20 (flip-driver decomposition, instrument committed) + SC-21 (determinism check row) | ~15 add | default-ON observability |

## §11/§12 — SOLVE INVENTORY + DEAD LEDGER (folds 9 + 13/R-2)

- Row 16 (max-plus) AMENDED with the measured autopsy: selection-mechanism binds (envelope rms
  2.30–58.18 logit), capacity passes (oracle rms 0.0761–0.1306), K=64 payload 1.5341 S rate-dead
  (862× margin); the τ_end coupling constant in the row re-binds after §1.4a (0.4989 at τ_end
  0.31, law-form unchanged). Band-residual K=1 specials (band/clamp/comb) UNAFFECTED — still the
  surviving essence. Reformulation queue ranked per wave-B.
- ◆ §12 rows now carry explicit scope tags (MINOR-R-2): backstepping = **family-dead-for-this-
  plant** (legs (b) actuator-through-loss + (c) descent-already-Lyapunov-stable carry the family
  scope; leg (a) no-1-D-causality kills only the kernel FORM) · LQR/HJB = family-dead-for-this-
  plant (derivation) · ES-dither = formulation-dead (campaign-timescale FD-ES survives) · Hajek =
  **import-dead** (the schedule-import formulation; the THEOREM lives on as the M4 bound) ·
  Griewank = DEFER-with-reason · ◆ NEW ROW: **max-plus K≤64 concave-max annulus fit =
  formulation-dead** (P-MP; queue in §11 row 16) · ◆ NEW ROW: **raw τ·ln5 Conley threshold =
  formulation-dead** (P-CON; fitted bar ships) · ◆◆ v6.2 NEW ROW: **decode-side zeroth-order
  seeded dither (B19) = formulation-dead** (P-DITHER kill: Δd_seg +2.1277533637e-6 ≥ 0, ~10σ;
  direction-blind churn ratio ≈ 0.98; unbiased decode-side variants carry measured priors
  AGAINST; render-informed placement + trained-with dither stay run-2 OPEN with priors FOR).

## §13 — THE NINE ROUND-1 MINORS (fold 13; no v5.1 errata existed — all folded here)

| minor | disposition in v6 |
|---|---|
| A1 flag spellings | §1.0 corrected: `--persistence-warmup-epochs` / `--persistence-classes` |
| A2 M3 "x%" unpinned | pinned: x = computed-from-SC-18's measured clamp-binding distribution (run-2 asymptote computation input; no run-1 knob consumes it) |
| B1 B-CT2 no disposition | §10: subsumed-by-ε_ff; residual queued behind Q3 |
| B2 B16×AmplifyIsland seam | §1.3: composition law declared (island support takes precedence; no double-weighting) |
| B3 ca-band outside c(τ) | §1.4a-cascade row 10 added |
| R-1 Q1 retirement clause | §1.3: two-τ-point conjunct restored |
| R-2 §12 scope tags | §12: tagged {formulation-dead \| family-dead-for-this-plant \| import-dead}; backstepping legs + Hajek scoping stated |
| R-3 P-MP kill wording | §7c: disposition language ("this expansion form stays unadmitted"), not truth claim |
| nits | B1-spec LOC noted as "spec + a few advisory LOC"; §1.0 line refs = counts-not-lines; EVSI "~10×" gloss = conservative (every consistent reading gives MORE); F12 label split done (Q2-τ vs F12 wall-clock) |

## SELF-ATTACK (v6-specific; v5's stand)

1. **τ_end = 0.31 is anchored on ONE checkpoint of ONE run** (the mod32cap ep650 best) and the
   over-descent reading is epoch-confounded [INFERRED, flagged in §1.4a]. Mitigation: it is the
   only MEASURED optimum we possess; the q̂ = 0.85 convention brackets it from its own live field;
   P-TAU2 + SC-3 make the endpoint self-correcting by run-2; the fail-safe is a CONSTANT (no
   silent adaptivity). The alternative (keep 0.062) is worse: it rests on a proven tautology.
2. **The M1 census could over-count** (state-dependent; GT-geometry estimator; |H_R| conservative;
   96-frame subset): the smooth-only floor 0.237 is a CONDITIONAL statement, labeled, and nothing
   in the launch config consumes it as a constant — it re-computes from run-1's SC-16 rows. It
   could also UNDER-count (H1 variant is 8% lower, not higher risk). Either way the design
   response (large-amplitude levers + gated dither) is robust to the census being 2× off in
   either direction.
3. **Did promoting the forfeit arm before its injection test violate req B?** No — the promotion
   is CONTINGENT (armed-with-fallback until B-INJ passes, §2.2f); the backtest leg is done, the
   fail-safe cap is untouched, and the injection test is a named pre-GO build. A GO with B-INJ
   unfired would be the violation; the launch checklist carries it.
4. **Comb numbers:** v6 cites 79.33% only; the 86% is quarantined until comb-reg step-0 pins the
   solid baseline. No v6 decision consumes either number (comb stays P1-conditional).
5. **Precision spot-audit (req J):** s\* printed at artifact precision 6.8971e-6 (4-dp-ν recompute
   6.8969e-6 — 0.003% display delta, no consumer at that resolution); crossing chain digits
   reproduce v5's exactly; the forfeit 5.450779e-4 S = 30.6% of the crossing margin — the arm's
   value is margin-denominated, not "small".

## §14 — v6.2 CHANGELOG (seal-round-2 fold, 2026-07-08; counter was RESET 0/3 by the NOT-CLEAN verdict — rounds restart on THIS revision)

Folds `seal_round2_v6_verdict_20260708.md` (1 BLOCKER + 2 MAJOR + 3 MINOR, 1:1) + the two
resolved gate probes (`probe_tau2_dither_20260708.md`). SUBSTANTIVE: build items + consumer-read
claims changed. Per-item:

1. **BLOCKER-1 (launch-route materialization) → FIXED IN CODE + PROVEN.** New named config
   `--config crucible_v6` (`src/tac/witness_autoconfig.py::derive_crucible_v6_config`, ~140 LOC
   incl. launcher route + tests, store-nothing-variant pattern). Pins at the source:
   τ_end 0.31 (extras-route C13 collision structurally gone) · ABSOLUTE tau@300 / Muon-cap 726
   (the family's 0.726×epochs = 2178 scaling was the measured wrong emission) · EXPLICIT τ
   denominator. **Derived-not-guessed τ tokens:** the trainer cosine is UNCLAMPED past its
   denominator, so the literal "--anneal-epochs 600 + cosine" REBOUNDS (τ(675) = 0.3363,
   τ(726-freeze) = 0.3826 ≠ 0.31); the correct materialization is `--anneal-epochs 3000` +
   `--tau-anneal-shape cosine_hold` + `--tau-hold-frac 0.2` ⇒ descent completes at ABSOLUTE
   ep600 and HOLDS: τ(675) = τ(726) = 0.31 exactly (law replica cross-checked against
   mod32cap's measured τ(650) = 0.3098 / τ(726) = 0.2157). PROOF: the round-2 dry-run
   invocation re-executed with the new config — 103/103 flags, NO C13 refusal, mem-preflight
   67.61 GiB PASS, every token verified in the emitted launch.sh; the design-schedule ==
   emitted-schedule equivalence is a permanent test
   (`test_crucible_v6_schedule_matches_design_doc`, wrong emissions pinned as anti-targets).
   §10 F-DET "0 (config)" under-scope corrected (B-CFG row). Named honest residual: the
   hosc-β anneal shares the den-3000 denominator ⇒ β(726-freeze) ≈ 1.41 (vs the control's
   3.177) — the M2 β-leg stays unresolved (v6 names no β pin); + the §1.1 base-delta OPEN
   question (Mod32SegOnlyControlBase's eikonal-0/freq-along-8/n-dir-4/lane-paint-off not
   auto-composed; l7@1001 would MISFIRE at 3000 ep) surfaced for the seal round.
2. **MAJOR-A2 (pose leg unpinned; #314 unnamed) → PINNED.** The launch config emits
   `--w-pose 1.0 --pose-carrier --pose-carrier-residual-mode table --pose-carrier-source
   generated` (inherited STRUCTURALLY from store_nothing_205; regression test
   `test_crucible_v6_pose_block_pinned`). **#314 / DAG DRIFT-D2** (pose-carrier-source
   inheritance drift) is now NAMED in §1.0 with the variant + test as its mitigation.
3. **MAJOR-B1 (locked-mass coverage) → §0.3 corrected + quantified.** "EXACTLY the
   concentration regions" overstatement replaced by the $0-computed per-lever table (42.7% lane
   / 19.3% hood / 12.9% movable / **25.3% Road↔Undrivable = 4.001e-4 = 22.5× margin with NO
   run-1 lever**); post-P-DITHER the leverless share is **≥ 25.3%** (dither backstop gone;
   ChromaBoundarySharpen g_I-raising credited as the only — small-amplitude — attacker).
   Named follow-up: Undrivable-boundary lever question ranked into the duty queue.
   **Crossing sensitivity (full precision): leverless-converts-nothing leaves remaining train
   budget 6.938661e-4 (win9, 63.4% of the 1.0939661e-3 bar) / 6.135635e-4 (central, 60.5%) —
   the engineered win9 case SURVIVES iff all other residual fits there; disposition unchanged,
   condition measured + tightened.**
4. **MINOR-1 (k_max) → recomputed:** ≤ 6 gross / **≤ 5 net of FIN dwell** =
   floor((2350 − 250)/387.1) (§2.2g(b) + §2 TAIL budget lines). Turnpike 3–7 survives.
5. **MINOR-2 (SC-3-ext route) → named:** §10 F26 row — in-trainer ~15-LOC per-verdict-cadence
   wrap of `flip_margin_quantiles` (currently ZERO trainer callers); until it lands run-1 is
   honestly CHECKPOINT-granularity via the committed offline instrument.
6. **MINOR-3 (C4 knob) → knob row added:** `--per-group-grad-clip` (launcher-injected,
   score-affecting, class (a) constant, V-S, `--no-` opt-out named) — §1.0 bullet; both
   crucible dry-runs carried it.
7. **Gate probes folded (mid-fold, sibling-measured):** P-TAU2 static half — knee f_target
   0.861663/0.862512 ≈ q̂ 0.85, **0.31 STANDS** (§1.4a corroboration; SC-3 live law unchanged);
   P-DITHER — **B19 KILL-as-formulated** (Δd_seg +2.1277533637e-6 ≥ 0, ~10σ): removed from the
   run-1 build list/§1.1/§5, §12 dead row added, §0.3/§9 coverage + family claims re-pointed at
   render-informed forms (run-2, measured priors).

Code surfaces landed with this fold: `src/tac/witness_autoconfig.py` (crucible_v6 variant) ·
`tools/launch_witness_run.py` (choices entry + config family + append-not-clobber `--dsl-lever`
+ activation-ledger rows for config-composed levers) ·
`src/tac/tests/test_witness_autoconfig.py` (7 new tests; 51/51 green; launcher suite 39/39).
All numbers [macOS advisory]; the config is MEANS.

Pointer contest-CPU 0.19110 UNMOVED — this draft is MEANS until the exact-eval row lands.
