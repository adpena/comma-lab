# v7.5 D_SEG LEVER + OPTIMAL-COMBINATION AUDIT — 2026-07-08

[no-triality] (read-only audit; the triality legs it touches = DSL config surface + DAG measured rows + the counter-force equations, all cited, none mutated)

STORES CONSULTED: `SPEC_v75_optimal_single_trunk_20260708.md` · live v6 run `experiments/results/levelset_n600_crucible_v6_run1_20260708T095730Z/launch.sh` · `src/tac/witness_autoconfig.py` (`compile_crucible_v7_config` / `diff_crucible_v6_to_v7` / `_CRUCIBLE_V7_DSL_LEVERS`) — the v6→v7 argv diff was COMPUTED, not read from a memo · DAG FEED 2026-06-25t (the −48% all-class directional decisive result) · `freq_along_ladder_probe_verdict_20260707.md` (the FLAT freq_along ladder) · `basis_integration_v7_20260708.md` · `counterforce_insufficiency_deepmath_20260708.md` · `council_symposium_clean_config_20260705.md` (lane-prior replace no-op #291) · memory L25/L65/L71/L76/L78.

**Authority:** every Δd_seg below is `[macOS-MLX research-signal]` / `[macOS-CPU advisory]`, NON-PROMOTABLE — measured on the frozen CPU-torch SegNet argmax, NOT byte-closed exact-eval. **Pointer contest-CPU 0.19110 UNMOVED — everything here is MEANS.** The END is a byte-closed `upstream/evaluate.py` n600 row < 0.19110.

---

## 0. AUDIT BASELINE — the EXACT v6→v7 delta (COMPUTED from `diff_crucible_v6_to_v7`, not asserted)

Live run is **crucible_v6** (pid 63069, 104 emitted flags). v7.5 = **crucible_v7** compiled by the DSL (148 flags). The semantic delta:

**v7.5 ADDS (d_seg-relevant):** Chan-Vese area constraint (`--area-constraint-birth` classes **1,3** force 1.0 tol 0.25) · birth-completion event (classes 1,3, ramp 50ep, post-level 0.2, τ-persist 0.8) · n323 ladder island homotopy (`--ladder-*`: lane birth 80ep/anneal 260, movable birth 60/anneal 200, λ-gate 0.0) · `--logit-adjust-classes 3` (Movable) · `--seg-form-unify-tau` (continuous L_τ, removes last PR95 stage bone) · tail warm-restart cycles (k_max 2) · Polyak finisher (start 2546) · `--per-group-grad-clip` · directional-basis rebalance (`--n-dir-freqs 2→4`, `--freq-along 4→6`) · `--tau-advance-mode event`.

**v7.5 CHANGES:** `--hosc-beta-end 10.0→3.177` (BLOCKER fix) · `--persistence-classes auto→3` · `--tau-anneal-shape cosine_hold→geometric` · `--lane-band-start-epoch 350→500` (+ `--lane-band-start-event lane_nucleus`) · `--seg-chroma-boundary-start-epoch 300→450` (+ event `annulus_plateau`) · `--muon-start-event powerlaw_meat`.

**v7.5 REMOVES:** `--l7-start-epoch` (l7 = measured DEFECT — correct drop) · `--tau-softplus-start-epoch` (dissolved into unify) · `--tau-hold-frac`.

**CARRIED UNCHANGED from v6 (already ON):** `--self-orient` · `--chroma --palette-anchor` · `--structured-init --structured-init-include-lane` · `--persistence-loss-weight 1.0 --persistence-recall-weight 1.0 --cldice-iters 5` · `--amplify-weight 1.0 --amplify-form hinge --amplify-persist inverse_thickness` · `--lane-render-band` · `--eikonal-weight 0.01 --length-weight 0.001` · `--island-dilate-px 1` · `--muon-*` (−32% lever) · **`--lane-prior-phi1 --lane-prior-phi1-mode replace`** (⚠ measured NO-OP, carried over — see §2).

---

## 1. RANKED MEASURED-EV d_seg LEVER LEDGER (with IN-v7.5 status)

| # | lever | MEASURED Δd_seg (n + artifact) | mechanism | IN-v7.5? |
|---|---|---|---|---|
| 1 | **ALL-CLASS DIRECTIONAL / anisotropic Fourier basis** | **−48%** (all-class) vs −8% (lane-only); n600 lever-B baseline 0.008257 → dir 0.005697 (−31%) → dir+cap 0.004445 (ep100) → **fuller basis+cap 0.002447 (−70% n600)**. ⚠ probe vehicle is CIRCULAR (built from `gt.lstars`); the REALIZED −48% via production `--self-orient` is UNVERIFIED. DAG FEED 2026-06-25t (subagent a922483dfc636ccc3), CANONICAL_RESEARCH_INDEX D1/D2 | orient Fourier features to the all-class boundary tangent field; 0-byte deterministic prior | **PARTIAL-ON** — `--self-orient` ON (was ON in v6 too); v7 adds `--n-dir-freqs 2→4` (in_feat 88→96) capacity on the directional basis. NOT the full probe vehicle. |
| 2 | **Capacity-routing (mod-dim / n-dir-freqs)** — pays ONLY after basis-match | **basis+cap n600 = 0.002447 (−70%)**; n96 −64% COMBINED; ALONE on isotropic = **+6% (HURTS)**, diverges @lr3e-3. FEED 2026-06-25t / CANONICAL D2 | KKT waterfill on margin-saliency; capacity is dominated until basis matches geometry | **ON** — n-dir-freqs 2→4 rides the directional basis (correct order); mod-dim 32 held (mod-dim ladder = PR95-echo, SECONDARY per L25) |
| 3 | **Muon optimizer (vs AdamW)** | **−32%** d_seg (memory L78; SPEC §B; DAG). THE drop in the curriculum-fix ablation | orthogonalized momentum SGD via Newton-Schulz | **ON** — fires ep726 (event `powerlaw_meat`, backstop 726); warm-start momentum + lr-final-frac |
| 4 | **Analytic lane band (openpilot poly, rule-118 free prior)** | lane d_seg **0.00087** (memory L71; `basis_integration_v7`) — the FREE lane carrier | render-time analytic band carries lane class off the learned basis | **ON but GATED to ep500** (event lane_nucleus / backstop 500, was 350) |
| 5 | **Curriculum-fix (drop d_seg-harmful stages)** | MEASURED MLX-port n600: CE 0.01045→0.00643 → **τ_softplus 0.00396 (THE primary drop, −38%)** → **smooth 0.00423 (RAISES +6.8% — DROP)** → l7 0.00369. ⚠ TENSION: SPEC/CLAUDE.md call l7 a measured DEFECT (L∞ decouples from viscosity flow) → v7 removes it; the ledger's l7-stage 0.00369 is CUMULATIVE-stage not l7's marginal effect. SETTLED per SPEC §B, flagged not reopened. CANONICAL cur-C2 | temperature/relaxation homotopy of the score functional; smooth + l7 decouple from the viscosity flow | **ON (best form)** — l7 REMOVED; τ_softplus dissolved into continuous `--seg-form-unify-tau`; geometric τ-anneal; no discrete smooth stage |
| 6 | **Persistence / clDice recall loss** | INFERRED-large: persistence collapse → movable ~0.001 d_seg (94% handled), DAG line 6220; no isolated n600 Δ | topology-preserving recall on thin structures (birth-death pairs) | **ON** — persistence-loss/recall 1.0, cldice-iters 5, persistence-classes=3 (Movable) |
| 7 | **Margin amplify (hinge / inverse-thickness)** | INFERRED — margin field IS the Fisher surrogate (−margin↔Fisher r 0.978, memory L57/#141); no isolated Δd_seg | amplify small-margin (flip-prone) pixels toward the target margin | **ON** — amplify-weight 1.0, hinge, margin-target 1.0, persist inverse_thickness |
| 8 | **Chroma (as d_seg lever)** | INFERRED (mechanism-grounded, NOT a measured Δd_seg): SegNet argmax reads RGB → RGB-slack in chroma flips the boundary annulus. DAG FEED (§CHROMA deep-math), memory | route capacity into chroma where it flips codim-1 boundary | **ON** — `--chroma --palette-anchor`; chroma-boundary sharpener gated ep450 (event annulus_plateau) |
| 9 | **Structured-init (road-plane SDF)** | **MEASURED road-plane SDF 0.000439** vs image-coords 0.000858 (0 bytes). ⚠ BUT `--structured-init-include-lane` = **effective NO-OP** (lane_px=0, lane isn't a static-maskable class) — see GAP #3. CANONICAL D10 | 0-byte train-time SDF prior for static classes; lane not static → paints nothing | **ON** (road-plane helps; include-lane is a NO-OP) |
| 10 | **n323 ladder island homotopy (NEW v7.5)** | INFERRED/derived — per-class λ-gated island-birth homotopy; addresses lane/movable birth (mod32cap islands-unborn was DELIBERATE, memory L2/L3) | homotopy grows islands per-class (lane 80ep, movable 60ep) | **ON (new)** — replaces the isotropic-dilation-only birth (proven NO-GO) |
| 11 | **Chan-Vese area constraint (NEW v7.5 counter-force)** | INFERRED/PREDICTED — Road within_flip ~0.38→**≈0.018** (0.015 area residual + placement); NOT measured end-to-end. `counterforce_insufficiency_deepmath` | area-Lagrange returns birth-arm over-painted area (Lane 13.8×GT, Movable 4.6×GT in run-1) | **ON (new)** — classes 1,3, λ DERIVED-LIVE (lane 683.8 / movable 322.6) |
| 12 | **Birth-completion event (NEW v7.5)** | INFERRED — Morse-Smale birth→boundary hand-off; DETECTOR live, loss-ramp landed | completes birthed islands (recall) before boundary placement | **ON (new)** — classes 1,3, ramp 50ep |
| 13 | **eikonal + length regularizers** | θ*-lever stack (memory L13): eikonal 0.01 + length 0.001 = viscosity-solution SDF regularizers | keep φ a signed-distance function; smooth boundary | **ON** |
| 14 | **logit-adjust (rare-class)** | INFERRED — Movable (class 3) logit adjustment for the rare-class-birth long tail | rare-class margin boost | **ON** — logit-adjust-classes 3 |

---

## 2. GAP LIST — best-measured levers OFF/sub-optimal in v7.5 (ranked by measured Δd_seg)

1. **[MED–HIGH] The −48% directional lever is only PARTIALLY realized.** The −48% (n600) was measured on the `lever_b_generator` PROBE vehicle (`ImprovedSegGenerator`), scorer-only-trained. The production trainer's `--self-orient` was ALREADY ON in v6; v7.5's `FEED_07a` lever adds only n-dir-freqs 2→4 + freq-along 4→6. **The full −48%/−64% gap between production self-orient and the probe's all-class-directional-basis is UNQUANTIFIED on the production trainer.** Whether v7.5's production basis realizes the probe's −48% is the run's own A/B — do NOT assume it does. (Defensible-as-configured, but flag: the decisive lever's transfer to production is unverified.)

2. **[MED] freq-along 4→6 does NOT close the 3.2× along-tangent deficit — and the real fix (COMB) is NOT in v7.5.** MEASURED: `freq_along_ladder_probe_verdict_20260707` (n600, oracle-capacity form) shows the freq_along ladder is **FLAT** (0→8 rungs 0.00731→0.00756, ≈ noise; ≥16 rungs INDETERMINATE below the 0.05 GT-separation floor). The **COMB (modulation carrier / 2nd-order scattering) is the best condition (0.00695)**, favored for the dash/along-tangent residual — and it is NOT wired in v7.5 (the in-training comb A/B FEED-08c remains the arbiter, un-fired). **Mitigant:** `basis_integration_v7` argues lane is OFFLOADED to the free analytic band (0.00087), so the witness carries only C²-cartoon edges → parabolic along=√32≈6 is the DERIVED optimum, and the FLAT ladder is consistent with lane_offloaded. So freq-along 6 is DEFENSIBLE **IF** the analytic band actually carries lane — but the band is gated to ep500, so for ep0–500 the witness DOES carry the lane at along=6 (under-budgeted vs the 3.2× carried deficit). GAP = the comb lever, un-activated.

3. **[MED — ELEVATED] Lane seeding is DOUBLY no-op'd; the MEASURED 3× lane-FN fix (paint-then-SDF) is NOT wired.** TWO of the three lane-seed surfaces in v7.5 are measured no-ops: (a) `--lane-prior-phi1-mode replace` (MEASURED no-op: replace-mode shallow lane SDF loses argmax to the deep road core, lane FN 0.00583→0.00538; #291, `council_symposium_clean_config` L15/63/90 + `lane_nucleation_failure` memo); (b) `--structured-init-include-lane` (MEASURED effective no-op: lane_px=0, lane isn't a static-maskable class; CANONICAL D10). The MEASURED FIX is **paint-then-SDF mode = lane FN 0.0058→0.0019 (3×)** — NOT wired in v7.5. So lane birth in v7.5 rests ENTIRELY on the NEW n323 ladder island homotopy (lane birth 80ep) + Chan-Vese area constraint (class 1). That's a plausible replacement, but **the measured 3× paint-mode win is left on the table**, and the two no-op flags should be dropped (cleanliness + remove the false signal that lane is prior-seeded, + add a `part_frac[lane]>0` post-init assert per the council rec). Caveat: seed dilation is FP-costly (see §3) — the ladder's eased/gated form is the reason raw paint-seed is NOT used.

4. **[LOW] mod-dim held at 32 (mod-dim ladder SECONDARY).** Per L25 the mod-dim ladder is a PR95-echo, SECONDARY to basis-match; not a gap, but the capacity headroom on mod-dim is unexplored. Deliberately deferred, not orphaned.

**NOTE — NOT gaps (already optimal):** l7 correctly REMOVED (measured defect); τ_softplus correctly dissolved (unify); Muon ON; persistence/clDice/amplify/chroma/structured-init/eikonal all ON; capacity ordered AFTER basis (correct).

---

## 3. ANTAGONISM / MEASURED-HARMFUL LIST (must stay off / be fixed)

| item | measured status | v7.5 handling | verdict |
|---|---|---|---|
| **l7 (L∞ sharpening)** | measured DEFECT — RAISES d_seg (decouples from viscosity flow) | REMOVED (`--l7-start-epoch` deleted; inert under unify) | ✅ CORRECTLY OFF |
| **fixed-β hosc divergence** | MEASURED DIVERGES (bc20 0.0119→0.01357; β8→0.03996) — but CONFOUND: only WITHOUT siren-init; RESOLVED by siren-init + β-anneal | `--hosc-beta 1.0→3.177` ANNEALED linear + `--siren-init` ON, bounded [1,3.177] ≤4 divergence bound (BLOCKER fix from inherited 10.0) | ✅ CORRECTLY AVOIDED (siren-init + anneal) |
| **isotropic capacity BEFORE basis-match** | +6% (HURTS), diverges @lr3e-3 (n96) | n-dir-freqs bump rides the directional/self-orient basis (capacity AFTER basis) | ✅ CORRECT ORDER |
| **τ-anneal erodes sub-critical lane (τ ⟂ nucleation)** | MEASURED: τ MCF erosion creep 0.00475→0.00657 ep300→400; τ_softplus realized −21.6% overall BUT erodes the 6px sub-critical lane (`lane_nucleation` memo). Rule: "never anneal τ below the dash period unless the comb is active" | geometric τ-anneal ON; COUNTERED by seed/eikonal(0.01)/Chan-Vese area constraint + ladder lane-birth | ⚠ COUNTER-FORCES PRESENT (area constraint + eikonal + ladder are the designed antidotes); comb NOT active (GAP #2) |
| **raw paint-seed / +2px island dilation** | MEASURED net-negative: paint-seed PLATEAUS d_seg ~0.026 (starves); +2px seed FP-costly ~15:1 (FP 0.0017→0.0114 > 0.0058 FN); #300/#323 | v7.5 uses the EASED/λ-gated n323 ladder, NOT raw dilation | ✅ CORRECT FORM (ladder not raw seed) |
| **smooth-stage** | RAISES d_seg +6.8% (0.00396→0.00423, MEASURED) | continuous unify-τ + geometric anneal (no discrete smooth stage) | ✅ CORRECTLY OFF |
| **lane-prior-phi1 `replace`** | MEASURED NO-OP (#291) | STILL EMITTED (carried from v6) | ⚠ HARMLESS-BUT-DROP (see GAP #3) |
| **GradNorm / per-step loss reweighting** | would MUTE the canary (SPEC §C: loss weights adapt at STAGE BOUNDARIES only) | v7.5 uses stage-boundary weighting only; no per-step GradNorm | ✅ CORRECTLY OFF |
| **micro-batch** | scorer forward batch-DEPENDENT (GPU 2.26e-2 drift, 11 argmax flips) — bit-identity-at-speedup IMPOSSIBLE | OFF (verdict-batch 32 is the chunked ADVISORY path, not a training micro-batch) | ✅ CORRECTLY OFF |

**SYNERGY structure (the optimal COMBINATION, not the union):**
- **Basis-match is PRIOR to capacity** (−48% basis alone; capacity alone +6% HURTS; combined −64%). v7.5 orders it correctly: self-orient/directional basis carries the geometry, n-dir-freqs adds capacity on top.
- **Margin-saliency is the shared driver** of the boundary-family levers (amplify + persistence-recall + the annulus/chroma sharpener) — all target the same small-margin (Fisher-heavy) annulus where ~97% of d_seg lives (#333). These compose, not conflict.
- **Chan-Vese (area) ⟂ tie-locus (placement) are ORTHOGONAL and SYNERGISTIC** (`counterforce_insufficiency`): area constraint returns stolen birth-arm area; it CANNOT place the separatrix. The placement residual (~0.0017 oracle floor) is P0 Force-3's domain — **NOT activated in v7.5's first launch by design** (§9: counter-force ONLY; P0 forces one-per-increment). So Road is expected to floor at the PLACEMENT floor (~0.018–0.035), and the NEXT lever after v7.5 is Force-3, not more area constraint.
- **Curriculum = coarse-to-fine flow** (memory L6/L57): the geometric τ-octave ladder = Morse-Smale persistence order = curvelet coarse-to-fine = temperature anneal. v7.5's event-driven octave advance is the level-set-native schedule (the last PR95 bone removed via unify-τ). ⚠ NOTE the open concern (memory L6): curriculum was the last PR95 inheritance; the geometric schedule is derived-shape but the per-rung dwell/order is still being validated (T3 #302).

---

## 4. OPTIMAL-COMBINATION VERDICT

**v7.5 turns on the best-MEASURED d_seg levers in a substantially correct, synergy-aware order.** The two decisively-measured levers — the directional basis (−48%) and Muon (−32%) — are ON; capacity is correctly ordered AFTER basis-match (avoiding the +6% isotropic-capacity harm); every measured-HARMFUL item (l7, fixed-β, smooth-stage, GradNorm, micro-batch) is correctly off/avoided; the new counter-force stack (Chan-Vese + birth-completion + n323 ladder) is the derived area-Lagrange fix for run-1's measured birth-arm over-paint.

**The four real caveats (none is a launch blocker; all are the run's own A/B to resolve):**
1. The **−48% directional lever's transfer from the (circular, gt.lstars-built) probe vehicle to the production trainer is UNVERIFIED** — v7.5's basis rebalance (self-orient + n-dir-freqs 4) is the plausible-but-unmeasured production realization; the run itself is the A/B.
2. The **along-tangent 3.2× deficit's real fix (the COMB / 2nd-order scattering) is NOT wired** — freq-along 6 is the DERIVED lane_offloaded optimum (defensible), but the FLAT freq_along ladder means the comb A/B (FEED-08c) is the true next along-tangent lever, un-activated.
3. **Lane seeding is doubly no-op'd (lane-prior `replace` + structured-init-include-lane), and the MEASURED 3× paint-then-SDF fix is NOT wired** — lane birth rests entirely on the new n323 ladder + Chan-Vese; the 3× measured win is left on the table, and the two no-op flags should be dropped.
4. **l7-removal tension** — SPEC/CLAUDE.md authority says l7 is a measured DEFECT (removed, correct); a curriculum-port row shows l7-stage 0.00369 (cumulative, not marginal). SETTLED per SPEC §B; flagged for honesty, not reopened.

**The DOMINANT unaddressed term is NOT d_seg — it is POSE.** Per SPEC §1 the store-nothing-ξ pose carrier is H-capped ~2.5; run-1 measured d_pose ≈ 1.79 flat ⇒ √(10·1.79) ≈ **4.24 of S from pose alone**. Every d_seg lever above is real MEANS, but **v7.5 as-configured cannot reach sub-0.19** — the pose representation (paid keyframe-pair / depth-warp carrier, break-even re-derived with MEASURED d_pose) is the binding blocker. The d_seg lever stack is optimal-enough; pose is the wall.

**Pointer contest-CPU 0.19110 UNMOVED.** No d_seg lever moves it until a byte-closed `upstream/evaluate.py` n600 row lands — and pose gates that row.
