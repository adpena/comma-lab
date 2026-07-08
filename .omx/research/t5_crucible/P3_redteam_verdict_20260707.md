---
doc_type: t5_crucible_p3_redteam_verdict
role: P3 RED-TEAM (operator-convened T5 crucible)
date: 2026-07-07
target: DRAFT_OPTIMAL_STACK_20260707.md (467daadd2, 565 lines)
passes: PASS1 provenance audit · PASS2 design attack · PASS3 deep-math grounding audit (operator
  critique adjudication, added mid-task by coordinator)
axis: all numbers [macOS-CPU/MLX advisory] unless noted; pointer contest-CPU 0.19110 UNMOVED —
  this verdict is MEANS.
---

STORES CONSULTED: the draft (full) · ORCHESTRATION_LEDGER (charter, reqs A–G, recess queue,
pinned flags, operator PASS-3 addition) · GROUNDING_PACKET · CONTEXT_COMPENDIUM (targeted:
AWAITING cluster, grad-clip confound, pose error bars) · positions S1–S6 (all read in full) ·
pursuit_chainA log (LINK 0/1, landed after the draft) · FEED-08l verdict memo + durable JSON dir ·
DAG sub015 (FEED-07c/07g/08k rows; #207 line 8249/8256; #149 lines 213–381; paintseedON 8366;
band-post-hoc 9054) · trainer argparse (31-flag spot-check, all found) · curriculum_dsl factories
(13/15 found; 2 declared-NEW confirmed absent) · canonical_equations_registry (AWAITING grep) ·
S6 byte-close JSONs ×3 (bit_exact=true, bytes 83,430/125,267/90,393 re-read) · LBND4
lane_band_res_coder_n600_measured.json · lane_share_probe_ep225_n600.json · S3 spectrum artifacts
(K8 + gpu-K8/K32 states) · mod32cap + 20260703T120444Z launch.sh · CLAUDE.md non-negotiables ·
docs/operating_manual_craft_handoff.md.

# P3 RED-TEAM VERDICT — REVISE-THEN-RECESS

**One-line:** the per-lever deep math is real and the provenance discipline is genuinely good —
but the draft's own S ladder contains a load-bearing arithmetic error (its stated "crossing tail"
does NOT cross the pointer), its schedule design quietly nullifies its own headline event-exit,
its two strongest measured d_seg levers sit in run-2 on a stale seam claim and an asymmetric
standard, and requirement A(ii) is claimed-satisfied but not satisfied. Fixable in one revision
pass + a handful of $0 probes. Not a structural reject.

---

## PASS 1 — PROVENANCE AUDIT (mandatory first pass)

Verdict: **largely CLEAN — the draft's tags are honest and the anchors are real.** I re-verified
by execution/inspection, not by trust:

| load-bearing claim | anchor | verified | review_status / form-limits | draft handles? |
|---|---|---|---|---|
| S6 byte rows ×3 (83,430 / 125,267 / 90,393 B; rates 0.0556/0.0834/0.0602) | reports/t5_s6_byteclose_mod32cap_ep650_*.json | ✅ re-read: `bit_exact_roundtrip_gate.bit_exact=true` all 3, archive bytes + sha256 present | 2-pair gate only (parity, not n600 realized d_seg — that is G5/P7 by design) | YES (P7 owed before run) |
| LBND4 30,892 B / LBND2 41,562, decode-reencode bit-identical | lane_band_res_coder_n600_measured.json | ✅ exists, [macOS-CPU advisory] labeled | decode NOT inlined (G2/B6) | YES (B6 LB at byte-close) |
| FEED-08l ladder (flat; comb best 0.00695) | freq_along_ladder_probe_verdict_20260707.md + durable JSON dir | ✅ both exist | **recovery-written post-credit-death, NEVER fresh-eyes-reviewed; only 2 scoreable rungs (0,8); ORACLE-form only; form-a retrain UNSUPPORTED** — the memo self-declares all of this | YES — P2 review owed; ordering guard makes along=8 primary regardless; lane_carried demotion PROVISIONAL |
| S3 ep650 spectrum (indefinite, |λ₋|=2.65×λ_max) | t5_s3_hvp_lanczos_20260707/spectrum_ep650_K8_s0.json + npz | ✅ artifacts exist (plus gpu K8/K32 states from chain-A) | PROVISIONAL-K=8; EMA-shadow state; surrogate-loss units; **NEW (post-draft): chain-A LINK 0 measured the HVP operator itself deviates ~35% from FD-true curvature along g (STE-smoothed vs real rounding landscape)** | PARTIAL — §9.2 tags K=8→full-P INFERRED + P3 gate ✅; the LINK-0 instrument caveat is NOT yet folded (finding F8) |
| Islands necessity (63.9% flips = un-born; big-3-only floor 0.00215) | lane_share_probe_ep225_n600.json | ✅ exists | **oracle-side, ep225, witness-alone upper bounds** (probe self-flags); flip-share ep225→ep650 stability UNMEASURED | PARTIAL — tagged ASSUMED + P6 flip-share gate ✅; but S5's R5 composed-surface ceiling ($0, ~1h, explicitly OWED by S5) was DROPPED from §7 (finding F5) |
| Muon finisher facts (+27.5% quench; anneal truncation 3.177/4.00, 0.216/0.05; τ_e=305) | S2 M-S2-1..5 on the run log; anneal freeze mechanism VERIFIED-VIA-SOURCE by S2 | ✅ (log-derived; mechanism source-verified) | **M-S2-5's τ_e=305 is an 11-point EXTRAPOLATION that S2 itself routed to RECESS-4 as "not load-bearing"** — the draft cites it as "the measured failure" anchoring the finisher budget law (§2.4) | NO — provenance upgrade-in-transit (finding F16); regression guard B4 mitigates |
| 251-flag argparse verification / zero invented flags | trainer argparse | ✅ spot-checked 31 draft flags incl. every law-carrying one — all exist | — | YES |
| DSL factories in §1.1 | curriculum_dsl.py | ✅ 13/13 existing factories found; GNSpectrumProbe + ChromaBoundarySharpen confirmed absent and declared NEW (I-4/I-5, LB) | "launchable as written" is contingent on I-4/I-5 landing — correctly marked LB | YES |
| mod32cap ground truth (--freq-along 8, --per-group-grad-clip ON, rewarmup 8, muon 726, epochs 1000) | launch.sh | ✅ re-read | — | YES (F10 row + grad-clip ON matches control) |
| band settings tau .85/eps .35/weight 1.0 "#205 measured settings" | 20260703T120444Z launch.sh | ✅ re-read | they are the #205 CONFIGURED values, not per-value swept optima — wording nit | acceptable |
| "15 registry equations AWAITING relied on" | canonical_equations_registry.jsonl | grep = 22 `ASSUMED_AWAITING_VERIFICATION` occurrences (rows vs fields — count drift, likely benign) | — | verify exact row count at P5 |

**Pass-1 net:** no claim was found treated as STRONGER than its provenance supports except the
two flagged upgrades (τ_e=305 anchor; chain-A instrument caveat not yet folded — the latter is
post-draft information, not a draft sin). The mandatory hot spots (FEED-08l, S3 K=8, islands
ep225, S6 byte rows, comb audit) are all either verified or correctly gated.

---

## PASS 2 + PASS 3 — RANKED FINDINGS

### F1 — BLOCKER — the §9.1 crossing arithmetic is FALSE and the band lower edge is inconsistent with its own components

Draft §9.1: *"Crossing requires the joint favorable tail: island birth near its ceiling (d_seg
≲ 0.0012) ∧ pose ≤ ~1.5e-4 ∧ waterfill-pass rate ≤ ~0.062."* Compute it:
100·0.0012 + √(10·1.5e-4) + 0.062 = 0.12 + 0.0387 + 0.062 = **0.2207 > 0.19110. The stated
joint tail does NOT cross.** At the draft's own band-lower d_seg 0.0012 and band-lower rate
0.0573, crossing needs pose term ≤ 0.0138 → d_pose ≤ 1.9e-5 — **~8× BELOW the 1.5e-4 kill
threshold**, i.e. essentially the borrowed-ancestor 3.4e-5 regime. Corollary: the S-band lower
edge **0.186 is not reproducible from the printed component bands** (0.12 + 0.018 + 0.0573 =
0.1953; 0.186 is reachable only by ALSO taking the waterfill-lower 52 KB byte tail, which
contradicts the printed rate-lower 0.0573) — and it leans on the pose term 0.018, which §6
itself declares BORROWED-ancestor and promises not to use ("the plan's numbers use only the
measured thresholds"). Internal contradiction.

**What this means:** under the draft's own measured ceilings (FEED-07c islands upper bound
~0.0013; rate floor ~0.057), run-1 crosses 0.19110 **only if the never-fired L3 pose mechanism
lands near ~2e-5** — not merely "survives the 1.5e-4 kill." The draft's honesty statement
("central does not cross") is correct but materially UNDERSELLS the miss: the honest run-1 best
case at the pose kill edge is ≈ 0.220.

**Fix:** recompute §9.1 end-to-end; restate the crossing condition as the actual frontier
(e.g. the S6 arithmetic: d_seg ≤ 0.00092 ∧ d_pose ≤ 1.51e-4 ∧ rate ≤ 0.0602, or any equivalent
triple); restate the band lower edge from consistent component tails; and then face the design
consequence — either (a) elevate the pose null-texture build W2 + the d_seg-ceiling levers (F4)
into run-1, or (b) explicitly re-scope run-1 as the two-wall measurement run whose pointer-
crossing probability is small-but-instrumented. Option (b) is legitimate under measurement-first;
selling (b) with (a)'s band is not.

### F2 — MAJOR — requirement A(ii) claimed satisfied, not satisfied

The ledger's req A demands *"the FULL #342 solve-don't-train inventory folded — every block
solvable by linear/quadratic/KKT/closed-form is SOLVED not trained, with where/when/conditions
stated."* The draft's self-review claims A is satisfied via the chain-A branch + B1/B5 + the
gated SOLVE stage. That covers A(i) (TerminalSolve) only. There is no #342 inventory disposition
anywhere in the draft. Ironically the draft already CONTAINS the solve instances (LengthSigma =
solved Young's-law fit; logit-adjust = solved priors; analytic band = solved openpilot fit;
waterfill = solved KKT; ξ derive-H = closed-form) — it just never presents them against the
#342 inventory, so unfolded blocks are invisible. **Fix:** add the per-block table (solved /
trained-with-reason / not-solvable-with-proof) — a ~1h documentation pass, but the requirement
says the stack fails synthesis without it.

### F3 — MAJOR — `--anneal-epochs 726` structurally nullifies the TAU event exit; §8 wall-clock is internally inconsistent

§2.2 makes anneal-complete (β=4.00 ∧ τ=τ_end) a TAU→FIN fire precondition AND sets the anneal
denominator = 726 (absolute epochs; trainer source confirms `--anneal-epochs` is the cosine/
geometric denominator decoupled from `--epochs`). Consequence: **the earliest admissible
TAU→FIN fire is exactly ep726 = the cap — the B1 event trigger is vacuous by construction for
run-1.** The control's measured 76–125 τ-stage epochs past meat exhaustion (M-S2-4, ~ep600→726)
are locked back in by design. This contradicts (i) the draft's own §2.2 law ("no denominator is
a free-running clock detached from its consumer" — 726 IS a fixed clock), (ii) the §8 projection
"CE ~275 + TAU ~350–400 + FIN ~150–250 ≈ 775–925 ep" (FIN cannot start before 726, so the event
path is ≥ 876–976 ep ≈ 26–29 h), and (iii) the headline "~35% wall-clock from event exits" (only
the FIN regression guard + run-end exit can actually save time). **Fix (adjudicate at P3b):**
either accept-and-state (cap-only TAU→FIN in run-1; correct §8 and the savings claim), or design
the genuinely event-bound denominator (on TAU meat-exhaustion, ACCELERATE the anneal tails to
completion over the rewarmup window, then fire — an anneal re-anchor law, small build, and the
only form that satisfies both M-S2-2 and M-S2-4 simultaneously).

### F4 — MAJOR (PASS 3) — AACoverageRender excluded on a STALE seam; asymmetric standard vs the band

The draft excludes AA from ARM-PRIMARY citing "seam + shipping blocker G3." **The seam is
resolved:** DAG FEED-07g (landed 2026-07-07, build a2f4acee7) — compose-after-downsample BUILT
with byte-identity proofs; AA now composes with lane-band/seed/residual by construction. S5
marked AA **FIRE** and quotes its status as "the gate's #1 MEASURED islands lever." The two
remaining gates are (a) the self-orient×supersample fine-mode MEMORY/wall-clock gate — a $0
pre-launch measurement, and (b) the AA decode at byte-close (B-3) — **exactly the same blocker
class as the band's B6, which the draft happily made launch-blocking-at-byte-close.** On an
islands-first arm, deferring the #1 measured islands lever on a resolved seam plus a blocker
class accepted elsewhere is convenience, not derivation. The one honest quantitative counter-
argument (ss=2 ⇒ ~4× render px ⇒ wall-clock/memory; S5's kill is >1.5× control) is nowhere
stated as the reason. **Fix:** run the $0 fine-mode memory+throughput gate in the probe wave;
if it passes within the wall-clock budget law, AA rides ARM-PRIMARY with B-3 LB-at-byte-close;
if it fails, the exclusion becomes DERIVED and gets written down with the measured cost.

### F5 — MAJOR — the headline d_seg band leans on ceiling arithmetic the draft dropped from its own probe wave

Rung-1 grounding cites "FEED-07c ceiling ~0.0013 upper-bound-caveat" — an ORACLE-side, ep225,
witness-alone bound. S5 explicitly flagged the composed-surface ceiling memo as OWED and
proposed it as R5 ($0, ~1h): DAG 8951 already shows the "fully unborn on the composed surface"
premise is FALSE at ep300 (part_frac 77%/98% of GT mass on the composed surface). The draft's
§7 carries S5-R2/R3/R4 (as P6) but **silently drops S5-R5** — the one probe that directly
tightens or breaks the rung-1 band. **Fix:** add S5-R5 to the probe wave; recompute rung-1
after it and P6 land.

### F6 — MAJOR — req-B letter violated: the CE→TAU verdict co-predicate ships with no backtest; the M3/M4 telemetry rows demoted below LB while triggers ship armed

Req B: every armed trigger ships ONLY with backtest + injection + cap. The draft backtested the
plateau half (fires ep251 → hence the co-predicate) but the co-predicate itself — the part that
does the work — has a hand-set threshold ("rel-eps **~**5e-3/25ep") and only an injection test
(T-1) named. The backtest is $0: the mod32cap history has 41 verdict rows; run the co-predicate
over them and report its fire epoch before arming. Separately: F4 (trigger would-fire audit =
the M3 fix) and F3 (online meat = the M4 fix) are demoted to "strongly-wanted" non-LB while B2
ships armed — contradicting req F's *"anything found by archaeology must be observable in
flight"* for exactly the two archaeology classes that motivated it. **Fix:** $0 co-predicate
backtest; promote telemetry F3/F4 to LB whenever any event trigger ships armed (they are ~55
LOC combined).

### F7 — MAJOR — attribution design double-books the one run; WeightEntropy λ=15 admission violates two seats' declared seams

S2 (interface, binding seam): "WeightEntropyPenaltyMLX… its λ A/B must not share a run with the
schedule A/Bs (attribution)." S4's admission design was a λ∈{0,15} PAIRED rider. The draft puts
λ=15 into the single pointer-aimed run with NO λ=0 comparator anywhere (the twin also carries
λ=15 — it is "ARM-PRIMARY minus pose"), and its stated kill ("twin-lag telemetry") cannot
attribute entropy-lever d_seg harm because the twin shares the lever. More broadly: ARM-PRIMARY
differs from mod32cap in ~15 simultaneous dimensions (λ=15, logit-adjust, LengthSigma, seeds,
eased, persistence, band, τ-shape geometric, anneal-epochs, rewarmup 20, w_pose, …), so the
per-stage kill "d_seg > control at ep100-matched (islands)" attributes to *islands* a delta
that belongs to the *stack*; F8's paired-delta rows only cover mid-run activations, not the
ep0-engaged levers. **Fix:** (i) restate every per-stage kill as STACK-level (kill the RUN, not
a lever); (ii) adjudicate at P3b which single attribution run-1 buys — pose (twin as designed)
or entropy-λ (twin at λ=0) — and route the other to the in-run Class-D×B recess with the
confound named in the row. Composing everything into one arm is the operator's charter; kill
criteria that pretend per-lever attribution inside it are not honest.

### F8 — MAJOR (post-draft evidence) — chain-A LINK 0 requires a measured-acceptance clause on SOLVE and an instrument caveat on the basin predicate

Chain-A (landed after the draft) measured the analytic HVP vs central-FD of the gradient:
rel err ~0.35, cos ~0.94 — the STE-smoothed operator disagrees with the true local (rounding-
jump) landscape by ~35% along g. Its own implication 1: *"any solve step must be accepted by
MEASURED loss/verdict, never by predicted quadratic reduction alone."* The draft's §2.3
quadratic_basin predicate (PD ∧ Newton decrement ∧ no usable λ₋) and the SOLVE stage carry no
measured-acceptance clause, and P3's kill bands don't carry the instrument gap. **Fix:** add
lm_accept-style measured-verdict acceptance as a HARD condition of any SOLVE step; annotate the
basin predicate as advisory-on-a-smoothed-operator (it already is advisory in run-1 — make the
reason durable); fold LINK 0/1 results into P3's pre-registered bands.

### F9 — MAJOR — the islands ep0 abort gate is calibrated against behavior the only measured seed run did NOT produce

§3.1 hard acceptance gate: "ep0 `part_frac[lane] > 0` (≈0.006) measured… launch aborts if the
seed did not take." The only measured seed run (paintseedON, DAG 8366) produced ep0 init d_seg
−36% **but `part_frac[lane]` STILL 0 at init** — the paint biases the rendered frame's
seg-agreement, not the witness φ-argmax partition; and L3/DAG 8420 diagnose lane init as a
measured NO-OP mechanism class. Unless the +include-lane / VP-tangent-eased variant measurably
differs (never measured at ep0), the abort gate fires on a healthy launch — or the mechanism
doesn't do what the gate assumes. **Fix:** $0 one-epoch init probe of the EXACT ARM-PRIMARY
seed config before GO; recalibrate the gate to what the mechanism measurably produces (possibly
part_frac[movable] > 0 ∧ lane-seed-present-in-φ, not lane part_frac).

### F10 — MINOR — the ξ q-levels control law is stated backwards

§1.2: "law: largest q with Δ(pose-term) < 0.002 (sweep 4096/1024/256)." As written this always
selects q=4096 (most levels = least penalty = most bytes), making the sweep vacuous. Intended:
**smallest** q with Δ(pose-term) < 0.002. A literal DSL implementation ships the wrong law.
(S6's M4 has the same wording bug — inherited, not caught.)

### F11 — MINOR — §5.1 byte-band inconsistencies

The archive band [86,000, 99,000] is not the sum of its printed component bands (component sum
= [~70.4K, ~103.5K]); and the joint failure tail (waterfill-fail ∧ B6-slip ⇒ LBND2) ≈ 125.9 KB
→ rate 0.0839 is unrepresented in either printed fallback band. State how the band was narrowed
(correlated tails?) or print the component-consistent band.

### F12 — MINOR — ChromaBoundarySharpen mislabeled score-neutral in the §1.1 sketch

The program sketch groups `ChromaBoundarySharpen(weight=0.1, …)` under "# OBSERVABILITY (req F;
default-ON, score-neutral)". It is a score-affecting LOSS lever (its own §1.2 row: class d+e,
DPR) — under the default-off-is-a-tracked-queue rule it must not inherit observability's
default-ON rationale. Fix the comment/grouping before someone compiles the sketch literally.

### F13 — MINOR — wall-clock throughput is borrowed from the control for a heavier stack

107 s/ep is mod32cap-class. ARM-PRIMARY adds persistence (clDice — S5's own kill is wall-clock
>1.3×), amplify, chroma-boundary, entropy penalty, band raster, pose carrier (a measured
SAVER). No throughput smoke exists in the plan; the paintseedON run measured ~8× slow-down from
the seed co-gradient class (partially mitigated since). **Fix:** a 5-epoch governed throughput
smoke at the exact ARM-PRIMARY config (also feeds the memory preflight) before §8's numbers are
believed.

### F14 — MINOR — launch-safety surfaces implied, not named

Resumability, per-stage EMA checkpoints, seed/determinism, self-orient persistence (F6-build)
are all in §1.2 ✅. But §7's RUN row does not name `tools/launch_witness_run.py` (governed
launcher, raw-python FORBIDDEN) or `witness_memory_preflight` at the REAL config (the #205 OOM
lesson: B=8 throughput gates hand out false greens; the stack's memory profile ≠ control's).
Name both in the launch plan.

### F15 — MINOR — req D PowerPlay ordering not honored by the printed §7 order

P5 ($0, ~1 min) sits behind P3/P4 (hours); P7 — the "FIX before ANY run" decode-integrity gate
that de-risks every later row (S6 put it first) — sits 7th. Either reorder cheapest-decisive-
first or state the concurrency plan that makes the printed order irrelevant.

### F16 — MINOR — provenance upgrades-in-transit

(i) Finisher budget law anchored on M-S2-5's τ_e=305 — an 11-point extrapolation S2 routed to
RECESS-4 as "not load-bearing"; the draft cites it as "the measured failure." Tag it INFERRED
and let B4's guard carry the weight. (ii) LBND2 bytes: 41,562 (S6/draft) vs 41,526 (S4/S5) —
36 B unresolved between seats; trivial but resolve at P5. (iii) "15 AWAITING equations" vs 22
registry grep hits — verify the row count.

---

## PASS 3 — DEEP-MATH GROUNDING AUDIT (operator critique: "lazy and naive design not grounded in deep math")

**Adjudication: the critique is WRONG at the per-lever layer, PARTIALLY RIGHT at the
composition/ceiling layer.**

**(a) Classification of major choices:**

| choice | class |
|---|---|
| basis (along=8 guard, Nyquist cap 64, hosc β-anneal, siren-init) | DERIVED (Candès–Donoho parabolic scaling, stem-Nyquist sampling theory, measured divergence class) |
| islands core (seed→eased→nucleus-guard→persistence) | DERIVED (Allen-Cahn critical nucleus; 1/persistence erasure law; measured collision 3.4×) |
| LengthSigma σ_ij | DERIVED (frozen-scorer Young's-law junction fit — the anti-cargo exemplar) |
| LogitAdjust per-class | DERIVED (Menon log-prior; measured n600 priors) |
| τ/β geometric shapes; anneal-completion-as-precondition | DERIVED (τ=ε=ħ equal-epochs-per-octave; Fisher-Rao constant velocity; M-S2-2 measured defect) — but see F3 for the denominator |
| rate plan | DERIVED (measured order-0 entropy floor ⇒ only symbol-count/symbol-entropy movers; KKT waterfill convexity dominance; λ_bytes exact-analytic) |
| Muon finisher laws (warm-start, lr-final-frac, budget law) | MIXED (measured quench/truncation receipts; budget-law anchor is extrapolated — F16) |
| WeightEntropy λ=15 magnitude | VERDICT-COMPOSED (torch mechanism-proof; λ* unswept; admission violates declared seams — F7) |
| **the ARM composition itself** (which levers ride together) | **VERDICT-COMPOSED** — S5's FIRE list + coupling heuristics (stagger, family-cap ≤2, one-homotopy-per-neighborhood). There is no interference calculus and the composed-surface ceiling is OWED (F5). This is the layer where the operator's critique lands. |

**(b) Ceiling-or-design-harder, per S-term:**
- **rate 0.0620:** ceiling DERIVED and binding (base at order-0 floor is a measured wall; the
  waterfill + LBND4-smoothed probes already chase the remaining slack). PASS.
- **d_seg 0.0019 central:** NOT derived. The binding constraint offered is an oracle-side ep225
  upper bound with the composed-surface arithmetic owed (F5), and two measured levers that
  raise the ceiling — AA (the #1 measured islands lever, seam RESOLVED per FEED-07g — F4) and
  the in-training comb (the FEED-08l-best dash carrier) — sit in run-2. The comb's gate (P1
  registration audit; measured mis-phase risk L65; render-composite measured net-negative
  +0.0038 ⇒ must be in-training) is DERIVED as a gate — but the draft gives no conditional
  inclusion law for P1-PASS: if the $0 audit passes pre-launch, keeping the comb out of the
  event-gated lever set (it rides the band, engages boundary-relative, F8 gives it an
  attribution row like chroma) is schedule convenience. **FAIL — name the headroom, revise.**
- **pose 0.039 central:** ceiling unmeasurable-by-design (the L3 mechanism is the never-fired
  decision variable; M5 IS the measurement). Acceptable under measurement-first — PROVIDED F1's
  honesty fix lands, because crossing hinges on pose reaching ~2e-5, not on surviving 1.5e-4.
  The draft demotes W2 (null-texture, the mechanism's full form) to "gated, NOT launch-blocking"
  while its own crossing math (once corrected) makes W2's mechanism the single highest-leverage
  unknown in the stack. Adjudicate W2's priority at P3b with the corrected arithmetic on the table.
- **Verdict for (b): REVISE** — the d_seg ceiling must either be derived (composed-surface memo
  + AA/comb interference statements) or the deferred levers admitted.

**(c) Strongest deep-math levers — consumed or excluded-with-reason?**

| lever | status in draft | adjudication |
|---|---|---|
| max-plus/comb dash carrier | OFF, gated on P1 registration audit; run-2 in-training A/B | gate DERIVED (L65 + measured render-composite negative); **no P1-pass inclusion law — under-designed (F4-sibling)** |
| chroma-at-annulus (GREEN, 93.4% of chroma-flips in-annulus) | CONSUMED — ChromaBoundarySharpen 0.1, event-anchored at tau-fire, annulus-gated by construction | ✅ (fix F12 labeling) |
| quadratic-basin solve | CONSUMED, gated on chain-A + full-P-only (subset-overfit honored) | ✅ but needs F8's measured-acceptance clause |
| matched-filter / pre-emphasis (#207) | absent | **CORRECTLY absent — the coordinator's premise is stale: DAG 8249/8256 records #207 deconv/matched MEASURED NEGATIVE (R all-pass) and the premise measured-CLOSED; only the unbuilt NTK band-pass survives, exponent-bet-gated.** Excluded-with-receipt exists; draft could cite it in §5.2-style form for completeness |
| orbit-coding rate framing | CONSUMED (FOLD-as-realized: derive-H + rule-118; I-2 equation registration) | ✅ |
| Morse-Smale persistence staging | CONSUMED (PersistenceTopology + skeleton cache + stagger; coarse-to-fine = the τ/β continuation) | ✅ |
| #149 sub-pixel/camera-res closed-form boundary placement | ~absent (1 oblique mention via the survival wall) | PARTIALLY represented by AA (sub-pixel coverage — deferred, F4) and StepNative β 4→8 (deferred with a stated mechanism rationale: τ_end pins sharpness, R low-passes, meat above β=4 unknown — and R6 rides the SAME run's Muon-fire checkpoint as a fork, which is defensible sequencing). The #149 closed-form placement itself (set the facet at 874 before R) is neither consumed nor excluded-with-reason — **name it in the run-2 headroom list with a disposition** |

---

## OPERATOR COUNTER-FRAME ADJUDICATION ("you are being pessimistic; this is a sandbox")

Weighed with teeth, not deference. Verdict: **the counter-frame is 2/3 RIGHT and it AGREES with
this red-team's findings more than it disagrees — it makes F1/F4/F5 MORE urgent, not less.**

**(1) The optimist's arithmetic — CORRECT as arithmetic, and it lands exactly on F4/F5.**
Rate IS a measured structural advantage (0.056–0.062 vs the frontier's ~0.118): the crossing
condition is 100·d_seg + √(10·d_pose) < 0.19110 − rate ≈ **0.129–0.134** (waterfill-pass).
At ancestor-class pose (term 0.018) that gives d_seg ≤ **0.00111–0.00116** — sitting EXACTLY at
the edge of the measured islands ceiling (~0.0012–0.0013, oracle-form, composed-surface OWED).
So in the optimist frame the crossing hinges on (a) pose reaching ancestor-class through the
never-fired L3 mechanism and (b) d_seg beating the CURRENT measured ceiling by a hair — which is
precisely why F5 (do the composed-ceiling arithmetic) and F4 (admit the ceiling-RAISING levers:
AA, comb-on-P1-pass) are the crossing enablers, not pessimism hygiene. The "7× pose gap" is
indeed a PRE-MEASUREMENT number (R1 ran w_pose=0 — the null was never pushed); the existence
proof (3.4e-5 on this same frozen scorer, different vehicle) makes ~2–3e-5 credible-not-proven.
Guard held: per L68 the ancestor number is NEVER cited as witness-solved — optimism about the
MECHANISM, rigor about the CLAIM. F1 stands unchanged: the draft's PRINTED crossing condition is
arithmetically false and its band lower edge inconsistent — the fix is to print THIS conditional
frontier, which is more optimistic AND more honest than what the draft printed.

**(2) The probability-model critique — RIGHT, and it becomes revision item F17.**
The draft's central 0.29 multiplies conservative independent per-rung priors — a one-shot
lottery model of a run that is actually instrumented for SEQUENTIAL DESCENT WITH REPAIR
(per-stage EMA checkpoints + event exits + F-alarms + kills-with-fallbacks + restore-and-
continue). **F17 (MAJOR, revision): the synthesis must present BOTH bands** — the joint-
independent-tail band AND a sequential-with-repair band with the repair mechanism NAMED per
lever: pose (F11 watch → W1b shadow law → L1 Jacobian fallback), band (umask/LBND2 giveback at
byte-close — fully post-run repairable), rate (waterfill/grammar — fully post-run repairable),
finisher (regression guard → restore-best → DECIDE), islands amplify (bounded closed-loop +
soft-gate), schedule (cap fail-safes). And it must name the NON-repairable-in-run failures,
which bound the repair band's optimism: seed-not-taken at ep0 (the F9 gate — abort, not repair),
basis/regime choice, ep0-engaged-lever harm hiding in the stack (F7 — WeightEntropy has no
in-run comparator). The honest central lies between the two models; which one headlines is a
P3b adjudication with both on the table.

**(3) The sandbox demand — RIGHT as a prior, already the direction of F4/F5/PASS-3(b).**
In a sandbox we own, iterate-ability is high and the marginal cost of a DERISKED lever in run-1
is low — provided it ships event-gated with an F8 attribution row and a kill (the chroma
pattern), not as a silent ep0 rider (the F7 anti-pattern). Applied: AA in (post the $0 gate),
comb in conditionally (post P1), #149 gets a named disposition, persistence-staged birth and
orbit-coding are already consumed. The sandbox prior does NOT license: citing ancestor pose as
witness-solved (L68), skipping the co-predicate backtest (req B), or attributing stack deltas
to single levers (F7). Include-with-instrumentation, never defer-by-default AND never
compose-blind — that is the synthesis's needle.

---

## OVERALL VERDICT: **REVISE-THEN-RECESS**

The draft is not lazy — 251-flag verification, honest DPR tags, real control laws, and the best
provenance hygiene of any synthesis doc in this campaign. But it may not proceed to recess
as-is: F1 (its own crossing arithmetic is false), F2 (req-A letter unmet), F3 (its schedule
nullifies its own event exit and its wall-clock table), and F4/F5 (the d_seg ceiling leans on
owed arithmetic while the #1 measured islands lever sits out on a stale seam) each change what
the recess should measure or what the run contains.

**Revision list for P3b (designer):** F1 arithmetic + the conditional crossing frontier
(counter-frame §1 form: 100·d_seg + pose-term < 0.129–0.134) · F17 dual-band presentation
(independent-tail AND sequential-with-repair, repair mechanisms named per lever, non-repairables
named) · F2 #342 fold · F3 anneal-denominator adjudication (accept-and-state vs re-anchor law)
+ §8 correction · F4 AA admission decision on the $0 memory/throughput gate · comb P1-pass
conditional inclusion law · F6 co-predicate backtest + F3/F4-telemetry promotion · F7
attribution restatement + twin-λ adjudication · F8 SOLVE measured-acceptance · F9 gate
recalibration · F10 q-law direction · F12 sketch label · #149 named disposition.

**Recess items now decisive (updated queue):**
1. **P1 comb-registration audit** ($0, ~1h) — now decides a run-1 lever (conditional inclusion), not just run-2.
2. **NEW: AA fine-mode memory+throughput gate** ($0) — decides F4 (AA in/out of ARM-PRIMARY).
3. **NEW: islands ep0 init probe** ($0, one epoch) — decides F9 (the abort gate's calibration).
4. **NEW: co-predicate backtest on the mod32cap verdict trace** ($0, minutes) — req-B compliance for the only armed trigger.
5. **P3 full-P HVP ladder** (chain-A, running) — with the LINK-0 instrument caveat folded into its kill bands.
6. **S5-R5 composed-surface ceiling** ($0, ~1h) — re-grounds rung-1; P6 flip-share rides with it.
7. **P4 waterfill + P5 LBND4-smoothed + P7 n600 parity row** — as drafted (P7 earlier, per F15).

Pointer 0.19110 UNMOVED — this verdict is MEANS until the revised stack's §7 ROW lands.
