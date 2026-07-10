# T5 CRUCIBLE-3 — SEAT S5 (ADVERSARY, pre-mortem seeded) — 2026-07-09

Agent: P1 SEAT S5 · INDEPENDENT. Attacks the EVIDENCE PACK (DELTA_GROUNDING) + the incumbent SPEC_v8
+ the on-disk scaffold — NOT the sibling seats. Pointer contest-CPU **0.19110 UNMOVED** — everything
here is [macOS-CPU advisory · research-signal · NON-PROMOTABLE], MEANS. This is a PRE-MORTEM: it is
2026-07-14 and the v8 increment-1 either shipped ABOVE the frontier rate, crashed at launch, or produced
an unattributable d_seg result. Below are the most probable failure narratives, each {mechanism · earliest
signal · sensor-that-catches-it OR the GAP · pre-staged response · verdict_scope}, ranked by
probability × blast-radius × SILENCE. Positive contribution = every narrative names the control-law the
P2 draft MUST carry to survive me.

STORES CONSULTED: CONVENING_20260709 · DELTA_GROUNDING_20260709 (all sections A–M + SETTLED + 6 Qs) ·
ORCHESTRATION_LEDGER (seat charters + operator bindings) · `docs/operating_manual_craft_handoff.md`
(the 10 competence-lookalikes; §4 re-derive-don't-recognize; §6 attack-your-own-conclusion) ·
crucible-2 `position_S5_adversary_20260709.md` (PATTERN ONLY — different crucible) · **PRIMARY CODE
re-derived, not memo-trusted:** `src/tac/boundary_math/road_undriv_bulk_field.py` (module header L1-63 ·
`bulk_boundary_byte_cost` L385-445 · `horizon_poly_xi_byte_cost` L470-555 · `road_component_stats`
L138-165 · b_c-out-of-loop guard L48-51) · `SPEC_v8_perclass_decomposition_20260708.md` (§1-§8, the
incumbent I attack) · grep-verified: SPEC_v8 contains NO "41 edges" (only "41% of Road's oracle flips");
`residual_sidecar_owed=True` present in the horizon coder; C(5,2)=10. NOT CONSULTED (sibling
independence): crucible-3 S1/S2/S3 positions (landed, off-limits), S4/S6 (not yet landed). $0, read-only,
no GPU, run dirs untouched (reading + arithmetic + grep only).

review_status: self-executed, fresh-eyes-UNREVIEWED (P3/P6 treat any adopted disposition as a
finding-producing round; every "fix" below is unreviewed new code/config — re-derive, don't trust it).
Every failure narrative is a PREDICTED risk (verdict_scope inline), NOT a MEASURED negative unless it
cites one (N-1..N-4 carry their measured scope). Remaining-gap-to-target for relative-significance =
0.19110 − 0.15 = **0.0411 S**.

---

## §0 THE ONE STRUCTURAL INDICTMENT (frames all 9 narratives)

**The pack's headline — "dominant 0.061 is 1.9× BELOW the 0.118 frontier, thesis CONFIRMED" (§A/§B) — is
the RESIDUAL-CODER-OPTIMISM trap wearing a MEASURED tag.** Re-derived from the scaffold's own code:
`horizon_poly_xi_byte_cost` (the generator of the 0.0032 horizon number) sets `residual_sidecar_owed=True`
and carries an in-code warning verbatim: *"Do NOT quote this as the complete Road↔Undriv rate without the
sidecar."* The 0.061 dominant total is a SUM of five per-edge DOMINANT-ONLY numbers, EACH of which the
generator flags as incomplete. **The archive ships the COMPLETE partition — the uncovered residual px are
not decoration, they are argmax-determining (a missing residual px is a wrong-class px = a d_seg penalty),
so the shipped operating point is the COMPLETE 0.140 (1.2× ABOVE frontier), not the dominant 0.061.** The
0.079 S gap between them is closed by TWO levers the pack itself labels un-built: curve-relative residual
coder = *"NOT built"* (§E-2); de-share double-count = *"uncounted … currently INFLATING the complete
number"* (§E-1). So the honest state is: **the v8 rate thesis is CONFIRMED on dominant structure and
UNPROVEN on the complete number — and only the complete number ships.**

**Relative-significance of the named enemy:** the residual-coder gap is 0.079 S. Against the remaining
distance to target (0.0411 S), that is **192% of the entire remaining journey to sub-0.15 (0.079/0.0411).**
The named enemy is BIGGER than the whole race that's left. This is not a rounding term to defer; it is the
decisive rate question, and it rides on two unbuilt coders. Any seat or draft that carries 0.061 as the v8
operating point is doing the operating-manual "borrowed number stripped of its caveat" at campaign scale.

---

## §1 FAILURE NARRATIVES (ranked by P × blast × silence)

### V1 — RESIDUAL-CODER-OPTIMISM: v8 ships at ~0.140 (ABOVE frontier), not 0.061 (TOP: high P × high blast × high silence)
**Mechanism.** The composite must reproduce the FULL argmax partition to avoid d_seg penalty; the
generators cover only 70–83% of each edge's boundary (§B: Road/Lane 72.5%, horizon 74.3%, Movable 70%).
The uncovered residual px are argmax-determining. Today's residual coder costs ~0.4–0.6 B/px (§E, MEASURED,
scattered short fragments near coordinate entropy) → residual S: Movable 0.0174 + horizon 0.0189 + Road/Lane
0.0420 = 0.078. Both headroom levers that would deflate it are UNBUILT. So increment-1's first honest
byte-close lands the COMPLETE number ~0.14, and the "1.9× below frontier" thesis evaporates on the number
that ships.
**Earliest signal.** The increment-1 rate row reads ~0.14 while the memo/DAG still quotes 0.061; OR a seat
composes the v8 rate advantage into the S-path projection using the dominant number.
**Sensor.** PARTIAL — the CODE flags `residual_sidecar_owed=True` per edge, but the ROLLUP (§B TOTAL row)
presents 0.061 as the un-caveated headline and the pack's §A one-line state-change leads with it.
**Pre-staged response.** Every v8 rate quote is a TRIPLE {dominant | complete | operating-point} with the
residual coder's BUILD-STATUS attached; the P2 draft's rate section is the COMPLETE number (0.140) as the
default operating point, the dominant (0.061) labeled "floor IF both headroom levers land + MEASURE," and
the comparison brief (P8) carries the COMPLETE number vs v7.5.2's ~0.060. Never the dominant as the S-path
input.
**verdict_scope:** FORMULATION (the dominant-as-operating-point presentation) — the geometry win over the
0.339 bitmap (5.5×) is REAL; the sub-frontier CLAIM is the un-earned part.

### V2 — THE REAL SDF-FIELD BYTE COST IS MEASURED BY NEITHER on-disk function → the increment-1 rate number does NOT exist yet (high P × high blast × MAX silence)
**Mechanism (re-derived from code).** The increment-1 build SHIPS a signed SDF field `phi_bulk`
(coarse-grid + INR-annulus, `lever_b_levelset_generator.signed_distance_fields`, lifted to two channels —
module header L14-24). But: (a) `bulk_boundary_byte_cost` (L385) measures brotli of the **Road-MASK
temporal bitmap** — a PROXY for the region SIGN, NOT the SDF field, and the code's own header (L41-46)
calls the 20-50 KB band *"CONJECTURED (a GUESS, ~1 order-of-magnitude; NO RD curve `d_bulk(B)` fitted)"*;
(b) `horizon_poly_xi_byte_cost` (L470) measures the dominant **arc** (residual owed). **NO on-disk function
measures the cost of the thing the build ships.** The DELTA §D "20-50 KB DERIVED" and the SPEC §2 net-stack
"−50..75% ≈ 0.049 S" both PRESUPPOSE this un-measured number. The P8 dual-chain comparison brief needs an
increment-1 byte number that, today, is a GUESS the code labels a GUESS.
**Earliest signal.** The increment-1 config's byte-close call site reports `bulk_boundary_byte_cost`
(mask-bitmap ≈ high-KB) or `horizon_poly_xi_byte_cost` (arc, residual-owed) as "the increment-1 rate" —
either way, not the SDF-field cost; multi-blob Road (37.2% of frames, `road_component_stats`) pushes the
real field cost to the CONJECTURED HIGH end.
**Sensor.** GAP. No apparatus asserts that the byte number quoted for increment-1 is the cost of the SDF
FIELD that inflate.py actually decodes, vs a mask/arc proxy.
**Pre-staged response.** Increment-1's FIRST deliverable is an RD curve `d_bulk(B)` of the REAL SDF-field
parametrization (coarse-grid resolution × INR-annulus coeff count → measured brotli of THAT payload,
bit-exact roundtrip), not a mask or arc proxy. Until it lands, the byte number in every brief is tagged
CONJECTURED (20-50 KB) with the multi-blob HIGH-end caveat welded on. This is the operating-manual §4
re-derive discipline applied to the one number the whole v8-vs-v7.5.2 comparison turns on.

### V3 — LAUNCH-PATH ≠ CONFIG-TESTS on the new scaffold: THREE byte surfaces, the DELTA §K "fix" is STALE, one wire gives the naive number (high P × high blast × high silence)
**Mechanism (re-derived from code).** DELTA §K says the scaffold bug is that `bulk_boundary_byte_cost`
"measured the FULL Road perimeter (2228 px) … needs a `mode='horizon_poly_xi'`." Re-deriving: (a) the
function measures the Road **MASK bitmap/RLE**, not a "2228-px perimeter" — the memo's px-scope framing does
not match the code; (b) the horizon path already exists as a **SEPARATE FUNCTION** (`horizon_poly_xi_byte_
cost`), NOT a `mode=` on the same function — so the DELTA §K prescribed fix does not match the code either.
Net: the scaffold exposes THREE byte surfaces (full-mask proxy / dominant-arc proxy / the unbuilt real SDF
cost) with confusable names. A config-compilation test passes calling ANY of them. This is F-1 exactly
(crucible-2: 221 tests green while every launch crashed on an unregistered gate key) — config tests are not
the launch path, and here the failure is silent-wrong-number rather than crash.
**Earliest signal.** Increment-1 reports a byte number that is one function's output while the build ships
another's payload; nobody notices because all three "compiled."
**Sensor.** PARTIAL. SYNTHESIS HARD-REQ-B mandates injection tests through the LIVE path, but only if the
draft writes them for the byte-close call site specifically, and F-1 shows the reflex is to test
compilation.
**Pre-staged response.** The increment-1 config PINS the byte-close call site to the SDF-field RD-curve
function BY NAME (V2's deliverable) with an assertion that its output is the field cost, and a full
dry-run byte-close that measures the ACTUAL shipped payload end-to-end. The DELTA §K "mode=" prescription
is corrected to "call the field-cost function"; the two proxy functions are labeled proxies in-code so no
config can quote them as the increment-1 rate.
**verdict_scope:** FORMULATION (the launcher/byte-close-surface confusion; structural, orthogonal to the
geometry).

### V4 — THE "41-EDGE-vs-5-FIELD" ASPIRATIONAL-RESIDUE FRAMING IS BUILT ON A PHANTOM NUMBER (medium P × medium blast × high silence — a pack-integrity catch)
**Mechanism (grep-verified).** The CONVENING §7 + DELTA §G + the ledger's S6 constraint all cite *"SPEC §1
'one field per adjacency EDGE' (41 edges)"* as the prime aspirational-residue to hunt. **SPEC_v8 contains
NO "41 edges."** The only "41" in the entire spec is *"the Road↔Lane tie calibration = 41% of Road's oracle
flips"* (§1 L17 — a PERCENTAGE, a flip-share). The pack conflated a flip-percentage with an edge COUNT and
attributed a nonexistent count to SPEC §1. The arithmetic: 5 classes → complete graph C(5,2) = **10** edges;
the REAL region-adjacency graph (Road = hub, P-A destination matrix: every class flips ONLY at its Road
separatrix, ZERO interior flips) has ~**4** live edges (Road↔{Lane, Undriv, Movable, MyCar}). So the true
tension is not "41 edges vs 5 fields" — it is "**~4 Road-hub edges vs 5 per-class fields vs 1 edge de-shared
in increment-1**," and because Road is the shared hub, the per-class table and the edge-centric graph
NEARLY COINCIDE (4 Road-edges ≈ 5 classes with Road as the shared spoke). The edge-centric-vs-class-naive
gap is MUCH SMALLER than the 41-vs-5 framing implies; increment-1 de-shares exactly ONE of ~4 Road edges,
leaving the other three per-class — which is fine, not a spec violation.
**Earliest signal.** The synthesis inherits "41 edges," reasons about a decomposition scale that does not
exist, and S6's blind-derivation is graded against a phantom target.
**Sensor.** GAP — the pack's OWN standing check #1 (numbered-cross-ref drift) did not catch the pack's own
drift; the failure-mode check fired on a fabricated instance of the very class it names.
**Pre-staged response.** Correct the figure everywhere to ~4 Road-hub adjacency edges (state the C(5,2)=10
complete / ~4 live derivation); re-frame the aspirational residue as "increment-1 de-shares 1-of-~4 Road
edges, the other 3 stay per-class by design — the SPEC §8(2) 'class-naive 5-field build is a spec violation'
line over-reaches, since per-class ≈ edge-centric on a hub graph." Let S6's blind derivation land against
the REAL adjacency graph; agreement is then a genuine vindication, not a match to a phantom.
**verdict_scope:** INSTANCE (this pack's 41-edge citation) — the edge-centric PRINCIPLE (don't pay for the
shared Road/Undriv curve twice) is sound; the COUNT is fabricated.

### V5 — P-C IS THE BLOCKING PRECONDITION, is GOVERNED-HEAVY + UNRUN, and reveals a COUNTED SEED FLOOR that moves the rate a THIRD time (high P × high blast × high silence)
**Mechanism.** Risk-3's P-C (flat/procedural-fill paint floor) is a BLOCKING PRECONDITION before the paint
stage is designed (SPEC §6(b); DELTA §G-3). But P-C is a heavy n600-through-R SegNet forward — the exact
full-P batched scorer path that OOM'd #205 at +66 GiB — so it is GOVERNED, memory-gated, NOT $0, UNRUN. And
increment-1 §9's sub-finding (surfaced in DELTA §G-3): flat-paint FAILS (0.0064 floor) ⇒ adequate texture
needs class-typical statistics ⇒ those are VIDEO-DERIVED = a nonzero **COUNTED seed floor** that NEITHER
P-A (oracle UPPER bound with real-frame texture) NOR P-C has isolated. So even the "interiors near-free"
dominant claim sits on an UNMEASURED counted floor that RAISES the rate a third time (after V1's residual
and V2's SDF-field cost). The rate ledger is a THREE-term sum where only one term (dominant generators) is
measured.
**Earliest signal.** P-C runs and reports a non-trivial counted-seed-floor byte number; OR increment-1's
paint stage is designed BEFORE P-C (violating the blocking precondition) because P-C is memory-gated-hard.
**Sensor.** OK-but-UNRUN (the counted-seed-floor is flagged in DELTA §G-3 as MEASURED-owed by P-C). The
silence risk is that P-C's memory-gate defers it and the paint stage proceeds on the P-A oracle bound.
**Pre-staged response.** Increment-1 cannot start its paint stage before P-C AND P-C's deliverable is the
counted-seed-floor byte number, folded into the rate ledger as an explicit THIRD term (dominant + residual
+ seed-floor). The P8 comparison brief lists P-C as an OWED-GATE with its memory-gated status, so the
v8-vs-v7.5.2 pick is never made pretending the rate is a one-term number.
**verdict_scope:** the P-A "interiors near-free" is CONDITIONAL on P-C (DELTA §F states this); my addition
is that P-C itself has a THIRD unmeasured term (the counted seed floor) beneath its own floor.

### V6 — THE THEFT GUARD IS STRUCTURALLY CORRECT BUT THE b_c VALUE IT COMPUTES IS ON THE MEASURED-WORSE SIDE OF N-1 (medium-high P × high blast × MAX silence)
**Mechanism (re-derived from code).** Risk-2's cure is staged training + b_c calibrated OUTSIDE the
scorer-gradient loop. The scaffold DOES this (L48-51: b_c closed-form via `damped_newton_ot_offsets`, no
scorer gradient) — the DECOUPLING guard is real. BUT the b_c rides `laguerre_logit_offset.damped_newton_
ot_offsets`, and N-1 (MEASURED, eq `laguerre_ot_head_offset` REGISTERED) proved that OT area-mass-matching
head offsets HURT on BOTH arms: no_offset 0.00272 < menon 0.00293 < ot_newton 0.00487 (WORSE). The
scaffold's b_c is AREA-matched (matches argmax mass to raw GT class frequencies); N-1's only open
reformulation — FLIP-weighted target masses (match to where flips are, not raw area) — is UNBUILT. So the
theft channel is closed (no gradient coupling) while the b_c the guard installs is the measured-worse OT
value.
**Earliest signal.** Increment-1's b_c calibration reproduces the N-1 OT regression (Road/Undriv tie bias
over-predicts the rare class → SegNet penalizes → d_seg worse than no_offset).
**Sensor.** GAP. Nothing cross-checks the scaffold's b_c against the N-1 no_offset < ot_newton ordering;
the guard verifies "no gradient coupling," not "the value is on the good side of N-1."
**Pre-staged response.** b_c MUST be FLIP-WEIGHTED (N-1's open reformulation), not area-matched: the
`damped_newton_ot_offsets` target masses are set to the per-edge FLIP masses (from run-1 per-class flip
data), not raw GT frequencies. If flip-weighting is unbuilt at increment-1, b_c defaults to **no_offset
(0.00272)**, NOT ot_newton (0.00487) — i.e., the safe default is OFF, and the OT/flip-weighted arm is a
measured A/B, never a silent default. This inherits SETTLED row "OT head offsets HURT — only FLIP-WEIGHTED
is open."
**verdict_scope:** N-1 is FORMULATION (area-matching as a d_seg surrogate); my addition is that the
scaffold ships the FORMULATION-falsified value as its default.

### V7 — CHROMA-FIRST ROUTING IS A PER-EDGE PROPERTY SOLD AS A GLOBAL DEFAULT; the real guard is "MEASURE pose," not the routing (medium P × medium blast × medium silence)
**Mechanism.** §F: chroma-first/luma-reserved is sound ONLY for Road/Undriv (chroma-separable grey/green/
blue). Road/Lane (41% of Road's flips) is LUMA-separable (bright lines on dark road) and CANNOT be
chroma-repaired. Increment-1 is safe because Lane is a separate ANALYTIC carrier (not paint-repaired) — TRUE
for increment-1. The RISK is the label graduating: the FULL v8 (later increments) paints more edges, the
routing's triangularity assumption is FALSE (§F caveat: NOT structurally triangular — low-freq chroma
recolors pass into pose at near-full strength), and "chroma-first" read as a global default silently harms
pose on a luma-separable edge.
**Earliest signal.** A later increment paints a luma-separable edge, chroma-first does nothing (or low-freq
chroma edits leak into d_pose).
**Sensor.** OK for increment-1 (§F explicitly scopes chroma-first to Road/Undriv paint + flags the
triangularity caveat). The silence risk is at the label's inheritance into full-v8.
**Pre-staged response.** The routing is a PER-EDGE property (chroma-separable vs luma-separable), computed
per painted edge, NEVER a global default; the REAL guard is ALWAYS "MEASURE d_pose on the composite +
Dykstra alternating projection to the fixed point" (§F step 4), the routing is only a warm-start. The P2
draft states this as the invariant, not the routing heuristic.
**verdict_scope:** FORMULATION (chroma-first as a global vs per-edge property) — flagged, increment-1-safe.

### V8 — THE POSE-CONDITIONING GATE READS "d_seg-sufficiency" AMBIGUOUSLY ON A DECOMPOSED 5-FIELD TRUNK (medium-high P × high blast × MAX silence)
**Mechanism.** The operator pose-engagement gate binds v8: pose fires only when d_seg is "sufficiently
conditioned" (a conditioning-gated EVENT, never epoch; DELTA §I P-5 + operator binding 2). On the v8
DECOMPOSED trunk, d_seg-sufficiency is PER-CLASS — five independent fields (Road/Lane/Undriv/Movable/MyCar)
condition at DIFFERENT rates (Road multi-blob + hub; Movable sparse late-born). An AGGREGATE conditioning
quantity masks a lagging class: pose can fire on an aggregate-conditioned trunk while Movable is still
unborn, engaging pose on an ill-conditioned per-class basin — the exact regime P-5 flags UNVALIDATED
(memory refutes cheap post-hoc/stored carriers; only JOINT descent from a coherent render crosses the
photometric wall). Store-nothing (~1 KB MANDATE, §I P-7) is a FRESH arm with no borrowed conditioning.
**Earliest signal.** Pose fires (conditioning-gate True) while a per-class d_seg (Movable/Lane) is still
descending steeply; the terminal pose-finish fails to reach the R1 regime.
**Sensor.** GAP — the pack does not specify whether the v8 pose-conditioning quantity is aggregate or
worst-class; the gate control law was designed for the single-trunk.
**Pre-staged response.** The v8 pose-conditioning gate reads WORST-CLASS d_seg-sufficiency (the gate fires
only when the LAGGING per-class field is conditioned, not the aggregate); threshold WITH derived provenance;
hysteresis specified; never-reached fallback = ship the BANKED R1 dxi (0.127 / 7.2 KB, DELTA §I P-1). The
gate is a conditioning-gated EVENT, never an epoch. Pose ⊥ d_seg EXACTLY (§I P-2: ∂d_seg/∂ξ ≡ 0) so a wrong
gate cannot corrupt d_seg — but it CAN ship a bad dxi or none; the fallback makes "none" = the banked
0.127, never a silent skip.
**verdict_scope:** FORMULATION (aggregate-vs-worst-class conditioning quantity on the decomposed trunk).

### V9 — OPPORTUNITY COST: v8's rate advantage is a WASH; its ONLY decisive delta (d_seg decoupling) is UNPROVEN, and v7.5.2's counter-force may already tame the theft (structural, low-silence — the risk-6 gate)
**Mechanism.** v8's dominant rate (0.061) vs v7.5.2's (~0.060) is a WASH (DELTA §J). v8's real bet is the
d_seg DECOUPLING: ∂φ_c/∂θ_{c'}=0 kills the MEASURED Lane 13.8× / Movable 4.6× theft into Road BY
CONSTRUCTION (SPEC §1). But that is UNPROVEN through R until increment-1 trains + byte-closes. Meanwhile
v7.5.2 is SEALED and launch-ready, and its remaining fight is the SAME d_seg blocker (§J: seg 0.455 is the
entire v7.5.2 fight) — and its Chan-Vese area-constraint counter-force is DESIGNED to rebalance exactly the
Lane/Movable→Road over-paint. If the counter-force ALSO floors the theft, v8's whole reason-to-exist (a NEW
scaffold + P-C + staged training + an unbuilt residual coder = apparatus×5) is dominated by the incumbent.
**Earliest signal.** v7.5.2 launch-1's per-class Road d_seg descends under the counter-force to a level v8's
decoupling was supposed to uniquely reach — v8 increments then never fire (risk-6 as-designed, the correct
outcome).
**Sensor.** OK — the dual-chain wall (P8) IS this gate; the risk is only if the comparison brief lets v8's
DOMINANT rate (V1) or CONJECTURED byte number (V2) stand in for a real S-path.
**Pre-staged response.** The P8 comparison brief states v8's DECISIVE-DELTA hypothesis explicitly (the
construction-level decoupling that v7.5.2's shared head CANNOT achieve) AND the SINGLE measurement that
would prove it: increment-1's Road d_seg-through-R under the decoupled field vs v7.5.2's Road d_seg under
the counter-force, at matched compute. If v7.5.2 already floors the Road theft, v8 is DOMINATED and the
honest brief says so — the dual-chain wall exists precisely so the operator picks on measured S-paths, not
on the decoupling's a-priori elegance.
**verdict_scope:** the decoupling is a construction-level PROPERTY (real); its d_seg PAYOFF vs the
counter-force is UNMEASURED (the increment-1 A/B is the arbiter).

---

## §2 SENSOR GAPS FOUND (the v8 apparatus's blind spots, consolidated)

1. **The rate ROLLUP presents the dominant (0.061) as the headline while the CODE flags every term
   `residual_sidecar_owed`** — no guard stops a seat quoting the incomplete number as the operating point
   (V1). Highest-value: the shipped number is 0.140, above frontier.
2. **No on-disk function measures the SDF-field cost the build ships** — two proxy functions (mask bitmap,
   dominant arc) exist; the real `d_bulk(B)` RD curve is unbuilt; the 20-50 KB is a code-labeled GUESS the
   comparison brief must not launder into a derived number (V2).
3. **THREE confusable byte surfaces + a STALE DELTA §K fix prescription** — config-compilation passes on
   any of them; the launch-path≠config-tests class recurs as a silent-wrong-number rather than a crash (V3).
4. **The pack's "41 edges" is a phantom** (SPEC §1 has no such number; C(5,2)=10, ~4 live Road-hub edges) —
   the numbered-cross-ref-drift check fired on the pack's own fabricated instance (V4).
5. **P-C (blocking precondition) is memory-gated-heavy + UNRUN + hides a THIRD rate term** (the counted seed
   floor beneath the P-A oracle bound) — the rate is a three-term sum with two terms unmeasured (V5).
6. **The theft guard verifies "no gradient coupling" but not "b_c is on the good side of N-1"** — the
   scaffold's default b_c is area-matched = the N-1-falsified value; flip-weighted is the open reformulation
   and no-offset (0.00272) is the safe default (V6).
7. **The pose-conditioning gate's "d_seg-sufficiency" is aggregate-vs-worst-class-ambiguous on the 5-field
   decomposed trunk** — aggregate masks a lagging per-class field; the single-trunk gate law does not
   transfer (V8).
8. **Chroma-first routing is a per-edge property at risk of graduating to a global default** in full-v8
   (increment-1-safe; the real guard is MEASURE-pose, not the routing) (V7).

## §3 THE THREE I'D BET ON (probability × blast × silence)

1. **V1 (residual-coder-optimism)** — the shipped number is 0.140 (above frontier), the code itself flags
   the incompleteness, and the 0.079 gap is 192% of the remaining journey to target while both closing
   levers are unbuilt. If the draft carries 0.061 as the operating point, v8 sells a sub-frontier rate it
   does not have. **The named enemy; the draft's rate row must be the COMPLETE number.**
2. **V2 (real SDF-field cost measured by neither function)** — the ONE number the dual-chain comparison
   brief turns on is a code-labeled GUESS; the P8 operator pick would be made on a conjectured byte number
   dressed as measured. **Increment-1's first deliverable must be the real `d_bulk(B)` RD curve.**
3. **V4 (the 41-edge phantom)** — cheap to catch, high integrity value: the pack's central
   aspirational-residue framing (that S6 is told to blind-hunt) is built on a fabricated count; the real
   tension (~4 Road-hub edges ≈ 5 per-class fields) is much smaller and increment-1's 1-edge de-share is
   fine, not a spec violation. **Correct the figure before the synthesis inherits it.**

**Meta pre-mortem (attack my own pre-mortem, operating-manual §6).** The biggest risk to THIS document is
that the P2 draft "addresses" each Vk with a tag rather than a control-law fix — the means-as-ends mistake.
V1 and V2 are DECISION-BLOCKING for the comparison brief (a brief built on the dominant number or the
conjectured byte cost makes the operator's which-to-run pick on sand); V4 is INTEGRITY-blocking (a phantom
count corrupts S6's blind derivation); the rest are ATTRIBUTION-protecting. Every "fix" above is unreviewed
new code/config — the P3 red-team must re-derive them, not trust them. Second self-attack: I re-derived V2
and V3 from the code and found the DELTA §K memo's framing (2228-px perimeter; mode='horizon_poly_xi')
does not match the actual functions — but I did NOT execute the functions (no GPU/heavy path), so my
"neither measures the SDF field" claim is SOURCE-INSPECTION verified, not runtime-verified; a P4 read-only
run of both functions on `gt_n600.npz` would confirm the proxy-vs-field gap empirically (it is $0). Third:
V6's "b_c defaults to area-matched" is inferred from the scaffold calling `damped_newton_ot_offsets` — I did
not trace whether increment-1's config sets its target masses to flips or areas; if it already flip-weights,
V6 downgrades to a standing check. Flag for P3.

---
Adversarial self-check: attacked the pack's headline (V1 residual-optimism), the one number the comparison
turns on (V2 SDF-field cost), the launcher surface (V3), the pack's own cross-ref integrity (V4 phantom
41-edge, grep-verified), the blocking precondition + its hidden third term (V5), the theft guard's
falsified default value (V6 vs N-1), the routing's label-graduation (V7), the pose gate on a decomposed
trunk (V8), and the opportunity-cost gate (V9). Re-derived V2/V3/V4/V6 from PRIMARY CODE + grep, not memo.
Did NOT read sibling seats (S1/S2/S3 landed but off-limits; S4/S6 not yet landed). $0, read-only, run dirs
untouched. [no-triality] (position doc; P2 synthesis owns leg propagation).
