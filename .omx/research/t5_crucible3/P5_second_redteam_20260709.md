# T5 CRUCIBLE-3 — P5 SECOND RED-TEAM (attack the post-recess design) — 2026-07-09

**Phase:** P5 (SECOND RED-TEAM → v3-revision decision). **Targets:** `SYNTHESIS_v2_v8_20260709.md` (P3b) +
`P4_recess_20260709.md` + the #386 builder memos (`residual_kit_deshare_curverel_build`, `inc1a_harness_build`;
`flip_weighted_bc_build_and_gate` **NOT landed**). **Surface:** `crucible3_v8`. `[no-triality]` (P6 seal owns
the legs) · **$0 · no GPU · no training · run dirs READ-ONLY** · #205 STOPPED/untouched. Pointer contest-CPU
**0.19110 UNMOVED** — everything here is `[macOS-CPU advisory · research-signal · NON-PROMOTABLE]` MEANS.
Remaining gap to sub-0.15 = **0.0411 S**; every magnitude quoted ÷0.0411.

**Fresh-eyes rule honored (§4 re-derive, not recognize).** I did NOT trust v2 or P4. Every FINDING below is
RE-DERIVED from primary artifact — `road_undriv_bulk_field.py::_horizon_profile` (L448, read: single-valued
"topmost Road row whose pixel-above is Undrivable … not the full multi-branch boundary"), the inc1a harness
smoke row (Undriv MASK d_seg 0.162 / 79.7% flip share, MEASURED n600), `decoupling_screen.py` L160 (`delta_mask
= DELTA_R_PROXY` STILL HARDCODED), `residual_kit_measured_20260709.json` (de-share 0.004399, dilate=2 footprint),
and the eightfold SEAL standing checks (`crucible_standing_checks_eightfold_20260709.md`). P3b's revisions AND
P4's measurements are themselves UNREVIEWED — I attacked both. `review_status: self-executed, fresh-eyes-UNREVIEWED`
(P6 treats every disposition here as a finding-producing round).

**STORES CONSULTED:** `docs/operating_manual_craft_handoff.md` (§3 blast-radius · §4 re-derive · §5 label ·
§6 attack-own · §8.4 plausible-summary/§8.5 borrowed-number) · `SYNTHESIS_v2_v8` (all 562 lines: §0 dispositions
· §A.1–A.7 · §B config · §C 6 risks · §D owed · §E R1–R7 · §F provenance · §G dedup · §H consumed · §I rep table)
· `P4_recess` (P5-facing summary · R3/R4/R6/R7 ledger · integration §1–4 · eightfold self-application) ·
`residual_kit_deshare_curverel_build` + `.json` · `inc1a_harness_build` · `crucible2/P5_second_redteam` (the
pattern) · `crucible_standing_checks_eightfold_20260709.md` (P2/P5/P6/P7/P8 SEAL checks) · **PRIMARY CODE
re-read:** `road_undriv_bulk_field.py` L448 · `src/tac/inc1a_harness/decoupling_screen.py` L26-222 ·
`residual_kit_measured_20260709.json` · grep-confirmed `flip_weighted_bc_build_and_gate_20260709.md` ABSENT +
`design_philosophies_eightfold_*` ABSENT at cited path (the checks live in `crucible_standing_checks_eightfold`).

**verdict_scope discipline:** every negative carries `verdict_scope: INSTANCE|FORMULATION|FAMILY` (default
narrowest); every magnitude carries ÷0.0411 relative-significance.

---

## HEADLINE (answer-first)

**VERDICT: A v3 REVISION IS NEEDED — ONE near-break (F-P5-1) forces a §B carrier-spec + §I representation-table
change; two REVISE (F-P5-2 δ_mask-still-proxy-in-code, F-P5-4 no-temporal-section) are edit-grade; three
CARRY-AS-RISK; one DISSOLVED. The launch DIRECTION (edge-centric decoupling) survives; increment-1a's CARRIER
does not survive as-configured.**

**The single WORST finding (F-P5-1): T1's three supports are ALL rate-side, and P4's R6 just measured a d_seg
hole that T1 never owned — and increment-1a §B ships EXACTLY the carrier that measured into the hole.** The
Road↔Undriv carrier is the single-valued horizon poly (`_horizon_profile` L448, re-read). P4 R6 MEASURED that
this carrier gives Undrivable MASK d_seg **0.162** with **97.5% of the under-coverage = UNSUPPORTED-COLUMN
lateral/side undrivable** the single-valued curve structurally CANNOT represent. T1's demotion of the bulk SDF
field was justified on rate (cheaper 0.0032 · dedup-correct · min-dim) — all three are RATE arguments. The bulk
field was the ONLY §I representation that could carry lateral undriv, and it is DEMOTED. **⇒ lateral undriv
(97.5% of the Undriv d_seg under-coverage) is now UN-HOMED in §I** (I1 = top-arc only; I10 = demoted). v2 §B
`road_undriv_carrier: mode: horizon_poly_xi, horizon_profile_scope: single_valued` ships precisely this. So the
increment-1a decoupled arm is either **hobbled** (byte-closed to single-valued → 0.162 Undriv floor → a KILL <!-- # VERDICT_SCOPE_OK: hypothetical confounded-kill SCENARIO the finding prevents, not a rendered verdict; untested formulations / alternatives: per-column undriv-extent curves (the P5b fix) · multi-branch horizon profile · retained bulk-field fallback (P-C-gated). -->
verdict is a CARRIER artifact, not a decoupling verdict → confounded falsifier, eightfold-P7 violation) or
**measured pre-byte-close** (overstates the shippable d_seg — the horizon poly cannot reproduce the trained
field's lateral undriv). P4 flagged "carrier must not be single-valued" but did NOT revise §B or §I; v2 predates
P4 so §B still says single-valued. **The seam between v2's carrier decision and P4's d_seg measurement is
UNOWNED — the classic §2 seam.** verdict_scope: **FORMULATION** (the single-valued carrier formulation; the
edge-centric decomposition FAMILY survives). Relative-significance: 0.162 Undriv on 79.7% of flips is the
DOMINANT d_seg term of the whole increment-1a screen — this is not a nit, it is the object.

**Second-worst is a LIVE-CODE proxy (F-P5-P9-1): the inc1a harness `evaluate_kill` STILL hardcodes
`delta_mask = DELTA_R_PROXY = 0.0196` (`decoupling_screen.py:160`), ~5600× LARGER than R7's measured 3.5e-6.**
With a 0.0196 kill margin, DECOUPLING-CONFIRMED requires a >0.0196 improvement (≈20% of the 0.100 agg d_seg —
enormous); realistic decoupling improvements (0.001–0.01) return **INCONCLUSIVE-below-floor**. **The kill
machinery is DECISION-INERT as-shipped** — it cannot fire CONFIRMED or KILLED for any realistic result. R7 <!-- # VERDICT_SCOPE_OK: evaluator output-vocabulary (CONFIRMED/KILLED are the harness's labels); the finding is the inert-proxy default, fixed by P5b's delta_mask swap. -->
measured the replacement; it is not swapped in. This is the P9 "proxy jams the verdict" failure in live code.

| # | finding | verdict | scope |
|---|---|---|---|
| **F-P5-1** | T1 rate/d_seg conflation → un-homed lateral undriv (97.5%, MEASURED) → increment-1a §B carrier hobbled; §I incomplete | **REVISE (near-break) — forces v3** | FORMULATION |
| **F-P5-P9-1** | `evaluate_kill` δ_R=0.0196 proxy STILL in live code (`decoupling_screen.py:160`); ~5600× too strict → screen decision-inert; R7's 3.5e-6 not swapped | **REVISE (worst proxy)** | FORMULATION |
| **F-P5-2** | δ_mask operative floor = max(3.5e-6, in-run seed spread); seed component needs ≥3 seed replicates/arm the 1a build does NOT specify → kill can fire on within-seed noise | **REVISE** | FORMULATION |
| **F-P5-3** | rate: shippable (pre-P-C) increment-1 = 0.135 = WASH; sub-frontier win rides the flip-weighted #226 waterfill whose r\* is P-C-gated (UNMEASURABLE in increment-1) | **CARRY-AS-RISK** | INSTANCE |
| **F-P5-4** | NO temporal section for the 1a d_seg screen (eightfold-P6 SEAL FAIL); decoupled per-class fields may flicker independently at the tie | **REVISE** | FORMULATION |
| **F-P5-5** | lane-generator coverage (53% of enemy) = FAST-FOLLOW not increment-1 scope; but the rate wash is PARTLY a weak-lane-generator artifact, not v8's ceiling | **CARRY (recommendation)** | FORMULATION |
| **F-P5-P9-2** | de-share 0.0044 rides a dilate=2 Movable-footprint PROXY for the bbox carrier's realized coverage | **CARRY-AS-RISK (minor proxy)** | INSTANCE |
| **F-P5-6** | provenance: R7/R6/triple-arithmetic CLEAN; the 0.00277-vs-0.0032 memo-vs-code nit must be pinned for P8 number hygiene | **CARRY (minor)** | INSTANCE |
| **F-P5-7** (welded from F7) | "#226-admitted flips" vs "uncovered residual px": SAME pool, but #226 admits a d_seg-ranked SUBSET at r\*<1 (equal only at r=1) | **DISSOLVED-with-clarification** | — |

**Net: v3 revision warranted.** F-P5-1 alone forces a §B carrier change + a new §I row + pinning 1a to
byte-closed measurement. F-P5-P9-1/F-P5-2/F-P5-4 are edit-grade REVISEs. The rate/proxy/scope items are CARRY.

---

## THE ATTACK SURFACE, worked (the task's five items)

### 1. The P4→v2 INTEGRATION DEBT (v2 written BEFORE P4 measured)

**(a) Rate policy survival — F-P5-3, CARRY-AS-RISK.** T2's v1 "complete-with-de-share+curve-relative" default
is DEAD: curve-relative REFUTED (P4/residual_kit: horizon 0.99×, lane 0.90×). The complete-lossless DEFAULT
degrades 0.140 → **0.135** (MEASURED: 0.140 − de-share 0.00440 − triple-point 0.00102 = 0.13458; arithmetic
CLEAN, re-checked). The SOLE remaining rate-closure path is the flip-weighted #226 waterfill (`WATERLINE ==
1.27 B/flip`, F7 real KKT). **Is r\* boundable before 1b/P-C? NO** (P4 confirms: per-flip through-R net_value is
P-C-owed). ⇒ **increment-1's SHIPPABLE (pre-P-C) rate is 0.135 = a WASH-with-frontier, NOT a win.** The sub-0.118
win is UNMEASURABLE until a governed-heavy P-C event. This survives ONLY IF the P8 brief states the shippable
rate is 0.135 (wash) with the sub-frontier win explicitly P-C-gated, NEVER blending r\* into the increment-1
shippable claim. v2/P4 do carry this honestly (0.061 never quoted alone; complete degraded). So NOT a break —
but the P8 brief carries a genuinely UNMEASURABLE r\* row; it must be a labeled RANGE [0.061, 0.135] with the F8
uncertainty, never a point. **The brief must not present increment-1 as a rate win.** ÷0.0411: the residual enemy
is now 0.074 = 180% of the gap.

**(b) The δ_mask kill floor has an unmeasurable-at-$0 component — is the kill PRE-REGISTERED in any meaningful
sense? — F-P5-2, REVISE (defensible pre-registration, but a build-spec gap).** The δ_R proxy is RETIRED
(δ_mask = 3.5e-6 MEASURED, ~5600× category error). P4's operative floor = max(3.5e-6, control-seed spread), and
the seed spread is NOT $0-measurable. **Attack (P7-falsifier philosophy):** a kill whose floor has an
unmeasurable component is STILL meaningfully pre-registered IF the DECISION RULE is fixed before the verdict is
read AND the data-dependent component is measured from the control arm's OWN replicates (standard practice for a
data-dependent noise floor). **BUT** — the load-bearing gap: v2 §A.4/§B does NOT specify that 1a runs ≥3 SEED
REPLICATES PER ARM. v2 §B says "matched compute, same seed budget" (singular seed). With ONE seed per arm there
are NO replicates → the seed-variance component (P4's DOMINANT floor component) is UNMEASURED → δ_mask collapses
to 3.5e-6 (frame-sampling only) → the kill can fire DECOUPLING-CONFIRMED on a difference that is entirely # MAGNITUDE_DISMISSAL_OK: this passage DEMANDS the missing measurement (seed-variance replicates) rather than dismissing by magnitude — it is the P2 noise-floor law applied, the opposite of a magnitude dismissal;
within-seed noise. **⇒ the kill is pre-registrable but currently under-specified: it needs (i) ≥3 seed replicates
per arm added to the 1a build, and (ii) the δ_mask formula wired as max(3.5e-6, measured seed spread), never a
fixed constant.** This is exactly the eightfold-P2 "SEED-VARIANCE honesty: single-seed Δ is INSTANCE-level until
the floor is measured." verdict_scope: FORMULATION.

**(c) THE SHARPEST — R6's 97.5%-unsupported-column: did T1 conflate RATE with D_SEG? — F-P5-1, the WORST
finding, detailed in the HEADLINE.** Confirming the conflation precisely: T1's triple support is
{cheaper 0.0032 · dedup-correct clause-A · min-dim clause-B} — I re-read all three in v2 §A.1; **all three are
RATE-side** (they argue the horizon poly is a cheaper/dedup-correct/lower-dim ENCODING of the Road↔Undriv
boundary). NONE is a d_seg-generation argument. P4 R6 then MEASURED the d_seg side: the single-valued horizon,
used as a GENERATOR, floors Undriv at 0.162 with 97.5% lateral. **T1 demoted the bulk SDF field on rate grounds
that R3 confirmed are code-real, but the bulk field was the only §I representation carrying lateral undriv.** So
the demotion is rate-correct AND d_seg-lossy, and v2 never reconciled the two because §A.1 argued rate while §A.4
config'd the carrier and P4 measured d_seg — three documents, one unowned seam. **The fix is LOW-DIM-ACHIEVABLE
(good news): left/right per-column undriv-extent curves are themselves low-order = GEOMETRIC-MINIMAL — the
single-valued `_horizon_profile` is not a min-dim NECESSITY, it is an under-complete generator.** So v3 must:
(i) replace §B `road_undriv_carrier` with a lateral-capable representation (multi-valued/per-column undriv-extent,
or a paired left/right undriv-boundary generator); (ii) ADD its §I row in GEOMETRIC-MINIMAL mode with a
derivation; (iii) PIN the 1a screen to measure the BYTE-CLOSED composite argmax (not the pre-byte-close trained
fields), so a 1a PASS predicts the shippable d_seg.

### 2. PROVENANCE RE-AUDIT OF P4 ITSELF (F-P5-6, CARRY-minor)

- **R7 floor method** (3.46e-6 SEM from per-frame margin-perturbation flip fraction, SEM=σ/√600): VALID as a
  frame-sampling LOWER BOUND; correctly labeled LOWER-BOUND; correctly flags seed variance as the real
  (unmeasured) floor. The number is near-decorative (P4 says so). CLEAN.
- **R6 97.5% decomposition** (reproduced harness composite Undriv 0.1616 ≈ 0.162 ✓; attributed to
  unsupported-column/below-arc/above-arc): CLEAN, and correctly scoped verdict_scope FORMULATION (this analytic
  generator). This is the MEASUREMENT that anchors F-P5-1.
- **Triple arithmetic** 0.140 − 0.00440 − 0.00102 = 0.13458 → "0.135": re-checked CLEAN; P4 conservatively uses
  the LOW end of the de-share range (0.0044, not 0.0104) → gives the HIGHER complete number (honest, no
  over-claim). de-share 0.004399 re-derived from `residual_kit_measured.json` ✓.
- **0.00277-vs-0.0032 nit:** the horizon dominant re-derives at S 0.00277 (4167 B, R3) but the ledger/T1 headline
  quotes 0.0032 (~14% apart). IMMATERIAL to T1's DIRECTION (0.00277 is even cheaper → T1's rate argument is
  STRONGER), ÷0.0411 = 1.1% of the gap. But it is a memo-vs-code discrepancy on a load-bearing headline number —
  **pin the P8 brief's T1 number to the code-emitted 0.00277 or annotate the amortization-method delta** (also
  P9-3 below).

### 3. LANE-GENERATOR COVERAGE (53% of the enemy) — F-P5-5, FAST-FOLLOW not increment-1 scope

P4: "the generator-coverage lever DOMINATES the coder lever" — the 40% off-curve lane residual (|n|mean 96 px,
828,048 px in the residual_kit json) is a LANE-GENERATOR coverage gap, not a coder gap. **Is it an increment-1
scope change NOW or a fast-follow?** FAST-FOLLOW. A better lane generator is a real BUILD (new module +
byte-close + n600 A/B) orthogonal to the decoupling thesis 1a tests; folding it into increment-1 BLOATS the
cheapest-falsifiable row and DEFERS the decoupling screen (§8.2 capacity-sweep reflex — do not displace the
measured screen with a build). Philosophy clauses engaged: **clause B / §I** (any new lane-generator params must
be GEOMETRIC-MINIMAL-derived or KKT-waterfilled — an unjustified lane-capacity bump is a §I FINDING) and
**P8 floor-first** (state the lane residual's floor + gap before optimizing). **Cost of NOT flagging it:** the
P8 brief's "increment-1 rate = 0.135 wash" would read as v8's CEILING when it is partly the CURRENT ANALYTIC
LANE GENERATOR's ceiling. **Connection to F-P5-1:** the analytic lane band is ALSO a weak generator (harness Lane
MASK d_seg 0.362) — so "weak analytic generators floor the decoupled arm" spans BOTH Undriv AND Lane. The
decoupled-arm carriers in increment-1a lean on two weak analytic generators; F-P5-1's lateral-undriv fix and
this lane-coverage fast-follow are the same disease (under-complete analytic generators) at two edges.

### 4. EIGHTFOLD SEAL PRE-CHECK (applied to v2+P4 now — cheaper than a P6 reset)

Using `crucible_standing_checks_eightfold_20260709.md` P2/P5/P6/P7/P8:

- **P2 (every Δ carries its noise floor):** PARTIAL. de-share 0.0044 has range floor ✓; ledger rows deterministic
  + INSTANCE-scoped ✓; δ_mask 3.5e-6 stated ✓ BUT its seed component is unmeasured (F-P5-2); the **r\* range has
  NO floor** (unmeasured — F-P5-3). Two load-bearing Δs (the 1a kill margin, the r\* operating point) lack a
  complete floor. → **P2 partial-fail, closed by F-P5-2/F-P5-3.**
- **P5 (no arm without in-run control):** PASS. 1a has a matched-compute control; `evaluate_kill` REFUSES
  borrowed baselines (P4 confirmed the baseline is measured in-run, never run-1's 0.312). Caveat: the
  seed-replicate spec is missing (F-P5-2); COMPUTE-match (not just ±5% param-match) is asserted but per-step FLOPs
  of a 5-field decoupled arm vs a 1-head control may differ — WATCH.
- **P6 (temporal section OR explicit temporal-N/A):** **FAIL — F-P5-4.** The 1a d_seg screen is per-frame-averaged
  mask d_seg with NO temporal-consistency derivation and NO explicit temporal-N/A. v2 §A.6 carries a
  temporal-screw companion in the POSE/risk-5 section, but the d_seg SCREEN is silent. The eightfold doc ITSELF
  flags "First application owed: the v8 temporal section." Decoupled per-class fields can jitter INDEPENDENTLY at
  the argmax tie (a shared head cannot) → the decoupling could WIN static per-frame d_seg while LOSING temporal
  flicker (MEMORY L67: #205 CE-residual = temporal flicker, 44% spikes = LANE). **v3 must add a 1a temporal
  section or an explicit temporal-N/A derivation.**
- **P7 (falsifier before build, threshold vs MEASURED baseline):** PARTIAL. 1a has a pre-registered kill ✓ but
  (i) its live threshold is the δ_R PROXY not a measured baseline (F-P5-P9-1), and (ii) the seed component is
  unmeasured (F-P5-2), and (iii) F-P5-1 makes the kill CONFOUNDED (a KILL could be the carrier hobble, not the <!-- # VERDICT_SCOPE_OK: confound warning about a FUTURE kill's validity preconditions, not a verdict; the reformulation queue is the P5b fix-set (carrier re-home + delta_mask swap + seed replicates). -->
  decoupling). → **P7 partial-fail, closed by F-P5-1/F-P5-2/F-P5-P9-1.**
- **P8 (floor-first):** PASS. rate floor 0.118 / complete 0.135 / dominant 0.061 ✓; d_seg analytic floor 0.162
  Undriv / 0.100 agg stated ✓; gaps stated ✓.

**SEAL PRE-CHECK verdict: P6 FAIL + P2/P7 partial-fail → NOT seal-ready; v3 required.**

### 5. BLIND-SPOT SWEEP (what v2 + P4 + the builders ALL miss)

1. **The unowned seam (F-P5-1)** — v2 argued T1 on rate, P4 measured the d_seg hole, NEITHER connected them into
   "the §B carrier contradicts the R6 finding." Nobody owns the carrier↔measurement seam.
2. **Temporal (F-P5-4)** — all of v2, P4, and both builders measure static per-frame d_seg; zero temporal-flicker
   coverage for the composite partition.
3. **The highest-leverage lever is UNRESOLVED at seal (b_c)** — `flip_weighted_bc_build_and_gate` has NOT landed
   (grep-confirmed absent). b_c is a 5.2×-gap-leverage move (N-1 span 0.00215 = 5.2× the gap). The seal ships with
   the single highest-leverage ~0-byte lever unmeasured (safe default no_offset — honest, but unresolved). P4
   tracks this (R4 owed); not a break, a WATCH.
4. **#384 wall-clock applicability to the 5-carrier decomposition** — 1a is PAINT-FREE (no through-R) → NO
   +66 GiB verdict-batch spike (the #205 OOM cause) → 1a is memory-CHEAP. GOOD. BUT the 1a A/B trains TWO arms
   (5-field decoupled + 1-head control), and the memory-preflight (`launch_witness_run.py`, calibrated for the
   single trunk) has NOT been re-validated for the per-class-field decoupled arm's TRAINING memory profile.
   1b/P-C (through-R) inherits the OOM risk and v2 routes it through the preflight ✓, but the 1a decoupled-arm
   training memory is un-characterized. Minor (1a is small/paint-free) — WATCH: re-validate the preflight for the
   5-carrier decoupled arm before 1a launch.

---

## P9 "PROXIES ARE POISON — USE THE THING ITSELF" — dedicated attack pass (coordinator-mandated, P9 maxim)

Every proxy in v2+P4+the builders, verdicted {has-thing-itself-path / FINDING}. Calibration receipts:
δ_R-as-δ_mask ~5600× · MPS 23× · proxy-loss 350× · LEVER-4 at-chance.

| # | proxy | where | thing-itself replacement path | verdict |
|---|---|---|---|---|
| **P9-1** | `delta_mask = DELTA_R_PROXY = 0.0196` STILL default in live code | `decoupling_screen.py:160` (RE-READ) | R7's MEASURED mask floor 3.5e-6 + in-run seed spread; swap the default, delete the constant as a default | **FINDING (worst proxy) — kill machinery decision-inert; the measurement EXISTS and is un-swapped** |
| **P9-2** | de-share magnitude rides a **dilate=2 Movable-footprint** proxy for what the bbox carrier actually holds | `residual_kit_measured.json` (dilate=2, tol=2); memo caveat | rasterize the ACTUAL G3 bbox carrier realized coverage (post-byte-close) and attribute the true intersection | **FINDING (minor) — has a named path; bounded by the 0.0044–0.0104 range; owed** |
| **P9-3** | ledger/T1 headline **0.0032** vs code-emitted **0.00277** | v2 §A.1 / rate ledger vs R3 | pin the brief to the code-emitted 0.00277 or annotate the amortization delta | **FINDING (minor) — memo-vs-code; immaterial to direction, pin for number hygiene** |
| **P9-4** | b_c SAFE-DEFAULT no_offset 0.00272 is a MASK/paint-free number used until the gate | v2 §A.3 | the #386 **realized-through-R** A/B gate is the SOLE authority; no_offset is a proxy PLACEHOLDER, never a verdict | **DISSOLVED-mostly — v2 ranks #386 through-R as authority; v3 must state through-R = SOLE authority explicitly** |
| **P9-5** | the ENTIRE 1a MASK d_seg (0.100 agg) is a PROXY for through-R shipped d_seg | v2 §A.4 (labeled NECESSARY-CONDITION PROXY) | 1b through-R (governed) is the thing itself, NAMED | **DISSOLVED for the generic case — BUT F-P5-1 is the SPECIAL case with NO thing-itself path (see below)** |

**P9-5 SHARPENED by F-P5-1 (the intersection of the two worst findings):** for the single-valued Road↔Undriv
carrier, the 1a mask d_seg is a proxy whose thing-itself path is **STRUCTURALLY BLOCKED** — the horizon poly
CANNOT represent lateral undriv at any capacity, so no amount of 1b through-R recovers it (the generator is
incapable, not merely un-tuned). A proxy with a blocked thing-itself path is the worst P9 class. This is why
F-P5-1 forces a carrier change, not merely a measurement upgrade: you cannot "use the thing itself" for lateral
undriv until the carrier can represent it.

**P9 net: 3 FINDINGs (P9-1 severe, P9-2/P9-3 minor) + 1 dissolved-mostly + 1 dissolved-but-F-P5-1-sharpened.**
The worst is P9-1 (a measured replacement sitting un-swapped in live verdict code) — it merges with F-P5-2 (both
are the δ_mask kill floor). Together: v3 must (i) swap the 3.5e-6 default into `decoupling_screen.py`, (ii) add
seed replicates so the seed component is measured, (iii) wire δ_mask = max(3.5e-6, in-run seed spread).

---

## OWED INTO v3 (the revision the findings force)

1. **[§B + §I, F-P5-1, LOAD-BEARING] Replace the single-valued `road_undriv_carrier`** with a lateral-capable
   representation (per-column undriv-extent / multi-valued horizon / paired L-R undriv-boundary generator); ADD
   its §I row (GEOMETRIC-MINIMAL, with derivation); PIN 1a to measure the BYTE-CLOSED composite. Without this the
   1a decoupled arm is hobbled and its kill is confounded.
2. **[code + §B, F-P5-P9-1 + F-P5-2] Swap the δ_mask default** in `decoupling_screen.py:160` from DELTA_R_PROXY to
   3.5e-6; wire `delta_mask = max(3.5e-6, in_run_control_seed_spread)`; ADD ≥3 seed replicates/arm to the 1a build
   so the seed component is measured. Update v2 §B `measure_1a.delta_mask` (still says `delta_R_proxy_0.0196`).
3. **[§A.4, F-P5-4] Add a 1a TEMPORAL section** (composite-partition temporal-flicker meter across frames) OR an
   explicit temporal-N/A derivation — eightfold-P6 SEAL requirement.
4. **[P8 brief, F-P5-3] State increment-1's shippable rate = 0.135 (WASH)** with the sub-0.118 win explicitly
   P-C-gated; carry r\* as a labeled RANGE [0.061, 0.135] with F8 uncertainty, never a point; note the wash is
   partly a weak-lane-generator artifact (F-P5-5), not v8's ceiling.
5. **[roadmap, F-P5-5] Name lane-generator coverage a FAST-FOLLOW** (post-1a), NOT increment-1 scope; its params
   ride §I clause-B.
6. **[P8 brief hygiene, F-P5-6/P9-3] Pin T1's 0.00277** (or annotate); tag de-share 0.0044 as `[dilate=2 proxy;
   thing-itself = bbox realized coverage, owed]` (P9-2).
7. **[§A.3, P9-4] State the #386 realized-through-R gate is the SOLE b_c authority**; no_offset is a proxy
   placeholder.
8. **[WATCH] Re-validate the memory-preflight** for the 5-carrier decoupled-arm training profile before 1a launch.

---

## VERDICT

**A v3 REVISION IS NEEDED.** ONE near-break (F-P5-1: T1's rate-only supports left the MEASURED 97.5% lateral-undriv
d_seg hole un-homed, and increment-1a §B ships exactly that hobbled carrier) forces a §B carrier-spec + §I
representation-row change + a 1a byte-close-measurement pinning. Two edit-grade REVISEs (F-P5-P9-1 the live-code
δ_R proxy that jams the kill machinery; F-P5-4 the missing temporal section — an eightfold-P6 SEAL FAIL) plus
F-P5-2 (seed-replicate build-spec gap) must land in v3. Three CARRY-AS-RISK (rate-wash honesty, dilate=2 de-share
proxy, provenance nits) and one DISSOLVED (F7 flip-set clarification). The launch DIRECTION (edge-centric
decoupling) and the RATE PROVENANCE (F9) survive fresh eyes; what does not survive is increment-1a's Road↔Undriv
CARRIER as-configured. **The worst finding is F-P5-1**; the worst PROXY is P9-1; they intersect (the single-valued
carrier is a proxy with a structurally-blocked thing-itself path).

**Pointer 0.19110 UNMOVED — this red-team is MEANS. Only a byte-closed `upstream/evaluate.py` n600 row < 0.19110
moves it.** Every finding above is unreviewed new work; P5b/v3 must RE-DERIVE from the primary artifacts, not
trust this memo. `[no-triality]` — P6 seal owns leg propagation.

## STORES CONSULTED (line)
`docs/operating_manual_craft_handoff.md` · `SYNTHESIS_v2_v8_20260709.md` (562 lines) · `P4_recess_20260709.md` ·
`residual_kit_deshare_curverel_build_20260709.md` + `residual_kit_measured_20260709.json` ·
`inc1a_harness_build_20260709.md` · `crucible2/P5_second_redteam_20260709.md` (pattern) ·
`crucible_standing_checks_eightfold_20260709.md` (P2/P5/P6/P7/P8) · **PRIMARY CODE re-read:**
`road_undriv_bulk_field.py::_horizon_profile` L448 (single-valued, confirmed) ·
`src/tac/inc1a_harness/decoupling_screen.py` L26-222 (`DELTA_R_PROXY` default confirmed) ·
`residual_kit_measured_20260709.json` (de-share 0.004399, dilate=2) · grep: `flip_weighted_bc_build_and_gate`
ABSENT (R4 gate owed). $0, no GPU, no training, run dirs READ-ONLY, #205 STOPPED, `[no-triality]`.
