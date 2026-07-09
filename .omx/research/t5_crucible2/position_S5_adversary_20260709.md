# T5 CRUCIBLE-2 — SEAT S5 (ADVERSARY, pre-mortem seeded) — 2026-07-09

Agent: P1 SEAT S5 · INDEPENDENT (did NOT read position_S1/S2/S3). Attacks the EVIDENCE PACK
(DELTA_GROUNDING) + the incumbent apparatus, not the sibling seats. Pointer contest-CPU **0.19110
UNMOVED** — everything here is [macOS-MLX/CPU advisory] research-signal, MEANS. This is a PRE-MORTEM:
it is 2026-07-12 and the v7.5.2 launch FAILED or produced unattributable results. Below are the most
probable failure narratives, each {mechanism · earliest signal · sensor-that-catches-it OR the GAP ·
pre-staged response}, ranked by probability × blast-radius × SILENCE.

STORES CONSULTED: CONVENING_20260709 · DELTA_GROUNDING_20260709 (all rows) · ORCHESTRATION_LEDGER
(seat charters + OPERATOR POSE-GATE CONSTRAINT lines 103-111) · `docs/operating_manual_craft_handoff.md`
(the 10 competence-lookalikes; §attack-your-own-conclusion) · prior crucible exemplar
`negatives_scale_validity_review_20260707.md` (§0 scale-bound lens) · PRIMARY CODE re-derived, not
memo-trusted: `src/tac/witness_control/shadow_controller.py` (~L300-330 the two false-green overlays) ·
`src/tac/witness_control/verdict_trend_alarm.py` (L28-66 thresholds, L131/263-267 per-class path) ·
`src/tac/witness_control/jacobian_basin.py` (L248-272 basin criterion + `plateau_est` = running-MAX) ·
`src/tac/canonical_equations/pose_jacobian_basin_conditioning_20260709.py` (L89-127 `would_fire_basin`,
basin_frac_ce=1.0 anchor) · `src/tac/witness_dsl/curriculum_dsl.py` (L3413-3448 amber preset) ·
`src/tac/witness_control/perclass_verdict.py` (d_seg_by_class IS emitted). NOT consulted: S1/S2/S3
positions (off-limits); no training, no n600 spend, #205 UNTOUCHED (reading + arithmetic only).

review_status: self-executed, fresh-eyes-UNREVIEWED (P6 treats any adopted disposition as a
finding-producing round). Every failure narrative below is a PREDICTED risk (verdict_scope stated
inline), NOT a MEASURED negative — the disambiguating probe that would confirm/refute each is named.

---

## §0 THE ONE STRUCTURAL INDICTMENT (frames all 11 narratives)

The pack's own headline — *"the frontier gap to sub-0.19 is now ENTIRELY d_seg"* (A/§C) — is **half
true and dangerously framed.** Pose is BANKED **as an artifact of run R1** (P-1), but P-5's HONEST FLAG
says v7.5's terminal pose-finish efficacy is **UNVALIDATED**, and the mid-P1 operator constraint
(ledger L103) adds a **conditioning GATE that has never fired in production and whose threshold is
provisional-by-construction** (§N1 below). So the true statement is: *"d_seg is the gap IF the σ_min
gate correctly re-ships the R1-class dxi from v7.5.2's own basin."* The pack sells the "entirely d_seg"
framing as settled; it is a **compound bet** with an un-instrumented second leg. Every seat that treats
pose as done inherits this. This is the operating-manual "borrowed numbers" mistake at campaign scale:
R1's 0.127 is borrowed, and the vehicle that must re-earn it is different.

---

## §1 FAILURE NARRATIVES (ranked)

### N1 — σ_min POSE-GATE FALSE-GREEN / NEVER-FIRES → the banked 0.127 never ships (TOP: high P × high blast × MAX silence)
**Mechanism.** The operator binding (L103) makes pose a conditioning-gated EVENT: fire iff
`median σ_min ≥ f_basin·σ_min^plateau AND basin_frac ≥ q` (`jacobian_basin.py` L269). Two independent
defects make this gate lie:
(a) **`σ_min^plateau` is `plateau_est` = the running MAX of median σ_min observed so far**
(`jacobian_basin.py` L270-272, verbatim: *"the plateau is only KNOWN offline at run end, so this live
answer is PROVISIONAL"*). With `f_basin=1.0` (the "current TERMINAL policy", equation L94) the criterion
reduces to *"current σ_min ≥ every σ_min ever seen"* — a **ratchet that is True only transiently at each
new all-time high**, never a stable plateau. It either fires spuriously on an early rising-σ_min
transient (pose engages on an ill-conditioned trunk = the EXACT R1 blocker, pose 1.8→4.35 of S) or,
if σ_min is still climbing at run-end, **never latches** → pose never fires → dxi never ships → frontier
stays pose-limited while the run reads *"correctly waiting for conditioning."*
(b) **`basin_frac` is a dead conjunct.** The canonical anchor records `basin_frac_ce = 1.0` at CE ep1
(`pose_jacobian_basin_conditioning_*.py` L127). If basin_frac saturates to 1.0 early, `basin_frac ≥ q`
is trivially True from the start and the AND collapses to the single ratchet sensor (a). The gate
LOOKS like a two-sensor conjunction; it is effectively one fragile sensor.
**Earliest signal.** Dashboard pose-readiness `would_fire` flips True on a single σ_min spike well
before per-class d_seg conditions (spurious-fire arm), OR stays False through terminal with the run
"complete" and shipped archive byte-identical to the pose-blind incumbent (never-fire arm).
**Sensor.** GAP. `would_fire_basin` is OBSERVER-ONLY and its own doc labels the answer PROVISIONAL;
nothing cross-checks the ratchet against a stable plateau, and no alarm fires when a "successful d_seg
run" ships with pose disengaged (0.19110 UNMOVED reads as success).
**Pre-staged response.** (1) Synthesis MUST replace the running-MAX ratchet with a **derived** plateau:
fit σ_min's relaxation curve (the P-6 conditioning↔coherence law) and gate on `σ_min ≥ σ_min(∞)·(1−δ)`
with δ derived, hysteresis specified, never-reached fallback = **terminal forced pose-finish with a
LOUD provisional flag** (never silently skip the banked contribution). (2) Add a **positive-control
sentinel**: the R1 basin is known to condition (σ_min→ the R1-from-converged regime); replay one R1
verdict through the gate and assert it fires. If the canary doesn't fire, the gate is untrusted → no
pose disposition is admissible (CLAUDE.md L3 verdict-clearance). (3) Fix `basin_frac_ce=1.0`: if
saturated, DROP it from the AND or re-derive `sigma_floor` so it is informative.
**verdict_scope:** FORMULATION (this σ_min-plateau formulation of the gate). The pose-conditioning
PARADIGM is sound; this instrument is not.

### N2 — REL-SIG RANKS (73% / 43.8% / 31.6%) ARE NON-ADDITIVE, MIXED-EVIDENCE-GRADE → compose-all fantasy (high P × high blast × high silence)
**Mechanism.** The Q1 composition menu is ordered by `rel-sig = ΔS/(0.19110−0.15)`. Three defects:
(a) **They sum to 148%** (73+43.8+31.6). These are SINGLE-lever ceilings measured in ISOLATION on
DIFFERENT baselines; the metric silently invites *"compose all three → 148% → we overshoot sub-0.15."*
That is the Dykstra non-additivity trap wearing a percentage sign — independent rate/d_seg savings are
conditional, not additive, and the pack itself elsewhere (V8-1 de-sharing double-count) knows this.
(b) **Grade laundering.** 73% (L-1 taper) is an ESTIMATED ΔS~0.03 from a re-interpreted under-converged
run; 43.8% (L-2 horizon) is an ORACLE CEILING (best case if every band-flip were fixed FREE, through-
training UNMEASURED); 31.6% (L-3 StepNative) is a real −4.5% n600. Three evidence grades (estimated /
oracle / measured) share one ranked column and read as commensurable.
(c) **RANK-1 rests on a flip.** L-1's decisive claim is "+18% NO-GO [verdict_scope: INSTANCE — one under-converged run, scoped at source] was ONE under-converged run;
converged anchors FLIP to −8%." That flip is a RE-INTERPRETATION, not a new byte-closed measurement.
The RANK-1 lever — the one everything composes around first — is the least-measured.
**Earliest signal.** The P2 synthesis draft composes ≥3 top levers ON in launch-1 citing rel-sig, OR a
post-launch A/B shows the composed ΔS is a fraction of the summed ranks.
**Sensor.** GAP. No apparatus flags that a rel-sig column mixes oracle/estimated/measured grades or that
the ranks are being read additively.
**Pre-staged response.** Treat rel-sig as a **priority hint, never a ΔS budget.** Re-label every row
with its grade (ORACLE / ESTIMATED / MEASURED-n600). L-1's −8% flip must be RE-VALIDATED byte-closed
before it earns RANK-1 composition; until then it is a candidate arm, not the anchor. Compose by
SYNERGY-ORDER (basis-before-capacity) with per-arm isolation, NOT by summing percentages.
**verdict_scope:** INSTANCE (this ranked-column presentation) — the levers themselves are real.

### N3 — LAUNCH-PATH ≠ CONFIG-TESTS RECURRENCE, now with 6 NEW composed gate-keys (high P × high blast × high silence)
**Mechanism.** F-1 JUST happened: B.4's `seg_temporal_screw` EventBackstopGate key prefix was omitted
from `GATE_KEY_PREFIXES` → EVERY launch would crash at `run_train` startup, and *"221 tests green"
tested CONFIG COMPILATION, never the LAUNCH PATH.* v7.5.2 composes ≥6 new levers, an amber preset, a
temporal-screw EVENT, a pose-finish TypedStage, AND a NEW σ_min conditioning event — **each a fresh
GATE_KEY_PREFIXES / event-wiring surface.** The seal's "397 tests green" is the SAME false comfort F-1
exposed.
**Earliest signal.** `run_train` crashes within seconds of the operator-GO — AFTER the launch window
opens, burning it.
**Sensor.** PARTIAL. SYNTHESIS HARD REQ B mandates INJECTION TESTS through the LIVE trainer path for
every event trigger — but only if the synthesis actually writes them for ALL new keys, and F-1 shows
the reflex is to test compilation.
**Pre-staged response.** Pre-register (BEFORE launch) a **full-config `run_train` dry-start smoke** that
exercises the actual `witness_control` wiring with EVERY composed lever's gate key present — a
fires-when-should + silent-when-shouldn't injection per HARD-REQ-B, NOT a unit stub. Gate the P8 GO on
this smoke passing, not on the test count.
**verdict_scope:** FORMULATION (the launcher-surface test gap; structural, orthogonal to any lever).

### N4 — AMBER COLLAPSE-FIX un-A/B'd, ON re-writes EVERY term's optimizer dynamics, may double-normalize Muon (high P × high blast × high silence)
**Mechanism.** Q3 wants `--stability-preset amber` composed ON by default as the joint-descent enabler
+ precondition for arming the sharpeners. Amber = grad-clip 0.5 + pose-grad-coeff-max 25 (eps floor
4e-2) + **per-group-grad-clip** (`curriculum_dsl.py` L3413-3414). Two silent risks: (a) **Muon already
orthogonalizes** the seg params (−32% d_seg vs AdamW, L78); layering per-group grad-clip / normalize ON
Muon can DOUBLE-normalize → flatten the very seg gradient doing the work → d_seg descent stalls while
the run looks *stable* (collapse-fix reads as "converged calmly"). (b) The pose-eps floor 4e-2 caps the
`5/√(10·pose+ε)` coefficient at 25 — correct for run-1's blowup, but it also **caps the terminal pose
descent** the D.9 finish needs; a too-aggressive floor throttles the pose-finish that must re-earn R1's
0.127. Amber is **un-A/B'd** (Q3 admits it).
**Earliest signal.** With amber ON, d_seg slope flattens EARLIER than the amber-OFF arm at equal epochs,
or terminal d_pose fails to reach the R1 regime.
**Sensor.** PARTIAL. The gnorm_hijack alarm + spike-guard catch BLOWUP, not SUPPRESSION; a normalized-
flat gradient is invisible to them.
**Pre-staged response.** Amber is a **parallel arm, NOT a silent launch default.** Run amber-ON vs
amber-OFF at bounded n600 (the collapse regime is reproducible per L-7's diagnosis). Instrument
per-term effective LR post-normalization; assert Muon's seg-update magnitude is not crushed. Do NOT
arm the sharpeners on the amber arm until amber alone is shown non-suppressive.
**verdict_scope:** FORMULATION (amber's interaction with Muon + pose-eps floor).

### N5 — SCALAR d_seg VERDICT MASKS the area-constraint's MASS-CONSERVING CLASS SWAP → false-green (high P × high blast × MAX silence)
**Mechanism.** R-3's Road-floor CURE (Chan-Vese area constraint) works by **rebalancing mass between
classes** — birth over-paints Lane 13.8×/Movable 4.6× INTO Road, mass-conserved (0.1191≈0.1189). A cure
that MOVES mass can produce a trajectory where scalar d_seg is FLAT or descending while ONE class (Lane)
regresses and another (Road) improves — the exact signature that reads GREEN. The new verdict_trend
alarm has a `per_class_alarms` path, BUT (`verdict_trend_alarm.py` L263-267) it is *"never fabricated
from scalar d_seg"* — it needs the verdict row to carry `d_seg_by_class`, AND the classification
OVERRIDE that turns a green not-green is driven by `vt.fired()` which keys on the scalar rise. If a
class swap keeps SCALAR flat (no scalar rise), the run stays classified in a converging/plateau class
and the per-class rise is at best a data field no controller acts on.
**Earliest signal.** Per-class verdict rows show Lane d_seg rising while scalar is flat — but only if
the n600 verdict pipeline emits `d_seg_by_class` on the SAME cadence AND a reader looks at it.
**Sensor.** GAP (conditional). `perclass_verdict.py` DOES emit `d_seg_by_class` — but the FALSE-GREEN
classification override is scalar-gated. A per-class-only regression under scalar-flat is not promoted
to a not-green classification.
**Pre-staged response.** Make the verdict-trend alarm's `fired()` include **per-class rise even when
scalar is flat** (the area-constraint failure mode is class-swap, not scalar-rise). Add a hard
watch-item: EVERY check-in reads per-class d_seg vs anchors (the watch-items-are-facets law, memory) —
never the scalar composite. Pre-register a class-swap alarm: `Road↓ AND (Lane↑ OR Movable↑)` with
scalar |slope| < flat-gate.
**verdict_scope:** FORMULATION (the scalar-gated false-green class, sibling of R-4).

### N6 — VERDICT-BATCH BIT-IDENTITY unverified for the NEW render paths → cross-arm A/B numbers not comparable (medium-high P × high blast × MAX silence)
**Mechanism.** The n600 verdict runs chunked `--verdict-batch 32`, claimed bit-identical because
eval-mode BN uses running stats (batch-independent). That invariant was validated for the CURRENT
render. The composition menu changes the RENDER: #276 chroma (chroma := rgb − BT.601-luma at the
annulus), #220 AA-supersample at grid≥384, StepNative activation — each alters the R-operator / render
resolution / value path. If ANY new lever perturbs the render→R→SegNet numerical path, the
verdict-batch bit-identity that makes ON/OFF arms comparable may silently NOT hold, and the A/B
Δd_seg becomes render-path noise + lever effect, indistinguishable.
**Earliest signal.** An ON/OFF A/B where the OFF arm's verdict d_seg drifts vs its own historical value
at the same ckpt (should be identical) — the tell that the instrument moved, not the lever.
**Sensor.** GAP. #313 established batch-DEPENDENCE for the scorer forward (2.26e-2 drift / 11 flips) but
that guard is about micro-batch training, not about verdict-batch stability under render changes.
**Pre-staged response.** Before any composition A/B, re-run the OFF arm's verdict at the SAME ckpt with
`--verdict-batch 0` (single-batch authority) vs 32 and assert bit-identity FOR THE NEW RENDER. If it
breaks, the A/B must use verdict-batch 0 (slower) or the comparison is void.
**verdict_scope:** INSTANCE (per new-render lever) — needs the $0 single-vs-chunked check each.

### N7 — WARM-START INHERITS THE ROAD-FLOORED BASIN; area-constraint can't dislodge a committed over-paint (high P × medium-high blast × medium silence)
**Mechanism.** Q2's operator rec (re-resume run-1 to stage-1, ~10h, then warm-start) inherits run-1's
PRE-actuation basin where birth over-painted Lane/Movable INTO Road (R-3), Road d_seg FLOORED ~0.40.
Warm-starting asks the counter-force (area constraint) to UNLEARN a committed over-paint — and unlearning
a saturated argmax commitment is generally HARDER than never committing (the boundary has to cross back
through the same low-margin annulus it already flipped). The warm arm may plateau ABOVE the fresh arm.
The FRESH arm has the mirror risk (N8).
**Earliest signal.** Warm arm's Road d_seg descends slower than fresh at equal post-warm epochs.
**Sensor.** PARTIAL — per-class Road d_seg trend, IF read per-class (see N5).
**Pre-staged response.** Do NOT spend 10h re-resuming to a "clean stage-1 decision point" that
FEED-205stop says is already PAST (ep257 CE→tau ckpt exists, ep325 stopped — the DELTA flags this timing
contradiction). Run the warm-vs-fresh decision as a SHORT bounded A/B from the EXISTING ckpts using the
per-class island-birth verdict (operator's own evidence choice, correct BECAUSE train-loss decoupled),
NOT a 10h re-resume. Pre-register the EXIT: warm wins only if Road d_seg crosses below the fresh arm
within N epochs.
**verdict_scope:** FORMULATION (warm-from-floored-basin) — the disposition, not a lever.

### N8 — FRESH-WITH-COUNTER-FORCE: area-constraint STRANGLES island birth before nucleation (medium-high P × high blast × high silence)
**Mechanism.** Fresh-from-ep0 composes the Chan-Vese area constraint (λ_lane 683.8 / λ_movable 322.6)
WITH the birth stack from the start. The area constraint penalizes over-paint; but an island NUCLEATES
by transiently over-painting before it refines. A strong area penalty applied during nucleation can
SUPPRESS the birth it is meant to balance → the island never forms → the 5-island-born state run-1
achieved is LOST, and the run stalls with unborn Lane/Movable (the mod32cap failure mode, L2/L3, where
lane-not-static→init NO-OP + growth-losses-OFF left islands unborn — here it would be growth-losses-ON
but area-STRANGLED).
**Earliest signal.** Lane/Movable part_frac stays ~0 past the birth-completion event window; the
Morse-Smale persistence event never fires → downstream stages gated on it never start (see N9).
**Sensor.** PARTIAL. Island part_frac is telemetered; but "event hasn't fired yet" reads as healthy
waiting, not as strangled birth.
**Pre-staged response.** The area constraint must be **RAMPED with a birth-completion PRECONDITION**:
zero (or floor) area penalty until the island's persistence crosses the birth threshold, THEN engage
the equilibrium 1.25×GT constraint. Pre-register: if part_frac < ε at the birth fail-safe epoch, the
area penalty ramp is too early → back it off. This couples N8 to the event-schedule (N9).
**verdict_scope:** FORMULATION (birth × area-constraint interaction at fresh start).

### N9 — EVENT-GATED CURES SILENTLY NEVER ENGAGE (temporal-screw, birth-completion, pose-finish co-fire) (medium P × medium-high blast × MAX silence)
**Mechanism.** v7.5 is ~80% event-driven (S-1). The Undriv-sky 0.082 cure (temporal-screw L-4) fires on
`annulus_plateau FORMED-boundary`; pose-finish co-fires with `_muon_gate.fired`; #341 head-solve fires
at terminal tau-best. A chain of events where each downstream cure depends on an upstream event FORMING
has a failure mode: if the sharpeners (taper/chroma/AA) keep the annulus boundary MOVING, it never
plateaus → temporal-screw never fires → Undriv-sky cure absent → but every telemetry row reads "healthy,
event pending." Same for birth-completion gating the tau stage (N8). Cascading silent non-engagement.
**Earliest signal.** An event's precondition sensor (annulus_plateau, Morse-Smale persistence,
_muon_gate) reads False for an anomalously long window while train loss descends.
**Sensor.** GAP for the CASCADE. Individual epoch fail-safes exist (min-stage 250, muon cap 726) but a
fail-safe that FORCES a downstream stage does not mean the CURE it gated actually did its job — it just
un-stalls the schedule; the cure may fire into an unready state (temporal-screw on an unformed boundary).
**Pre-staged response.** Per HARD-REQ-B, every event ships a FAIL-SAFE cap — but the cap must ALSO
record "fired-by-fallback-not-by-precondition" so the post-hoc attribution knows the cure engaged in a
degraded regime. Add a **stuck-precondition alarm**: event precondition False for > 2× its expected
window = LOUD (a cousin of the binding-term-stall overlay). Never let a fail-safe firing read as a
healthy event.
**verdict_scope:** FORMULATION (event-cascade non-engagement).

### N10 — ABLATION ≠ ADD-BACK: chroma/AA arms gain near-zero, confound the composition (medium P × medium blast × medium silence)
**Mechanism.** L-5 (chroma) and L-6 (AA) are the two UNMEASURED (duty-to-ESTIMATE) levers. L-5's basis
is a REMOVAL ablation: constant-luma FLIPS 7.54% Lane→Road. That measures what chroma information is
WORTH, not what ADDING a chroma-boundary lever GAINS — the witness may already exploit most of it
(the pack notes it "converges to a near-constant palette," i.e., under-exploits, which cuts BOTH ways:
big headroom OR the lever can't move a palette-collapsed render). Composing chroma ON expecting ~7.5%
and getting ~0.5% both wastes the arm AND confounds any co-composed lever's attribution.
**Earliest signal.** Chroma-ON vs OFF Δd_seg ≪ the ablation magnitude.
**Sensor.** OK — the DELTA explicitly flags "ablation ≠ add-back" (low silence, this one is called out).
**Pre-staged response.** Chroma/AA are ISOLATED first arms with an ADD-BACK ESTIMATE probe BEFORE
composition (the duty-to-ESTIMATE the pack already assigns). Do NOT compose them with another unmeasured
lever in the same arm.
**verdict_scope:** FORMULATION (add-back efficacy) — flagged, lower risk.

### N11 — TORCH-PARITY TWIN: mid-training lever A/Bs don't transfer MLX→torch (parity-safety is only at the ARCHIVE) (low-medium P × medium blast × high silence)
**Mechanism.** Q6/M-2 floats a torch twin to fan out ON/OFF lever A/Bs on paid Modal within the $20 cap.
The pack's parity argument — *"the byte-closed ARCHIVE is the shared invariant, so exact-eval is
parity-safe"* — is TRUE only for the FINAL archive. A **mid-training lever A/B** compares TRAINED arms;
if the torch twin TRAINS them, MLX↔torch training divergence (different optimizer numerics, different R
kernel) means the torch-twin's ON/OFF verdict does not transfer to the MLX production run. The parity
holds at the archive, not at the comparison the fan-out is buying.
**Earliest signal.** A torch-twin A/B verdict contradicts the local MLX arm at the same lever.
**Sensor.** GAP. Nothing asserts MLX↔torch training-trajectory parity (only archive exact-eval parity).
**Pre-staged response.** Use the $20 cap for what parity-safety DOES cover: **exact-eval of byte-closed
archives** (the promotion authority) + the owed CPU-torch n600 VERDICT on FINAL candidates. Do NOT build
a torch twin to A/B lever ARMS — run the lever A/Bs on the MLX production substrate (serial local
byte-close), reserve Modal for the archive-level exact rows.
**verdict_scope:** INSTANCE (the torch-twin-for-A/B-fan-out proposal specifically).

---

## §2 SENSOR GAPS FOUND (the apparatus's blind spots, consolidated)

1. **σ_min pose-gate has no stable-plateau reference** — `plateau_est` is a running-MAX ratchet, its own
   doc calls the live answer PROVISIONAL, `f_basin=1.0` makes it "≥ all-time-high"; `basin_frac` is a
   dead conjunct (saturates 1.0 at CE ep1). No canary asserts the gate fires on a KNOWN-conditioned (R1)
   basin. → the single highest-value missing sensor (N1).
2. **No alarm when a "successful d_seg run" ships pose-DISENGAGED** — 0.19110 UNMOVED reads as success;
   the never-fire arm of the pose gate is silent (N1).
3. **False-green classification is SCALAR-gated** — the verdict_trend override keys on scalar rise; a
   mass-conserving class swap (the area-constraint cure's own mechanism) stays classified green while a
   class regresses (N5). `d_seg_by_class` is emitted but not promoted to a not-green classification.
4. **Launch-path startup is not gated on a full-config live dry-start** — "397 tests green" = config
   compilation, the exact F-1 false comfort, now × 6 new gate keys (N3).
5. **Collapse-fix suppression is invisible** — gnorm_hijack/spike-guard catch blowup, not a
   normalized-flat gradient that stalls d_seg while looking stable (N4).
6. **Verdict-batch bit-identity is unverified for the NEW render paths** — cross-arm A/B comparability
   is assumed, not checked, after chroma/AA/StepNative change the render (N6).
7. **Event-cascade fail-safes un-stall the schedule but don't verify the CURE engaged in a READY state**
   — a fail-safe firing reads as a healthy event; a stuck-precondition alarm is missing (N9).
8. **rel-sig ranks carry no evidence-grade label and are read additively** — no guard against summing
   oracle+estimated+measured percentages (N2).

## §3 THE THREE I'D BET ON (probability × blast × silence)

1. **N1 (σ_min gate)** — it is the NEW operator constraint, the least-instrumented surface, and a
   never-fire failure ships a byte-identical incumbent that reads as success. Fix the gate BEFORE launch.
2. **N3 (launch-path ≠ tests)** — F-1 proves the reflex is alive; 6 new gate-keys × the same false
   comfort = a startup crash after the operator-GO. A full-config dry-start smoke is cheap insurance.
3. **N5 (scalar masks class swap)** — the cure's own mechanism (mass conservation) is the false-green's
   mechanism; the alarm that patched R-4 is scalar-gated and won't catch it.

**Meta pre-mortem (attack my own pre-mortem):** the biggest risk to THIS document is that it reads as a
checklist and the synthesis "addresses" each with a tag rather than a control-law fix — the
means-as-ends mistake. N1 and N3 are LAUNCH-BLOCKING (a run that ships pose-disengaged or crashes at
startup is not "a caught risk," it is a dead launch); the rest are ATTRIBUTION-protecting. Every "fix"
above is unreviewed new code/config — the P3 red-team must re-derive them, not trust them.

---
Adversarial self-check: attacked the pack (N2 rel-sig laundering, N6 instrument-under-render), the
false-green class beyond R-4 (N1 pose-gate, N5 class-swap), unmeasured interactions (N4 amber×Muon,
N9 event-cascade, N10 add-back), warm-vs-fresh BOTH ways (N7 warm-inherits-floor, N8 fresh-strangles-
birth), the σ_min gate self-false-green (N1), and EMA/verdict instrumentation (N5/N6, EMA-lag folded
into N7's disposition-evidence caveat). Did NOT read sibling seats. $0, #205 UNTOUCHED. [no-triality].
