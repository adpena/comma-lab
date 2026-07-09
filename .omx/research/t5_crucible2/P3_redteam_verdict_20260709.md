# T5 CRUCIBLE-2 — P3 RED-TEAM VERDICT vs THE DRAFT (v7.5.2) — 2026-07-09

**Phase:** P3 (RED-TEAM + MANDATORY PROVENANCE AUDIT). **Target:** `SYNTHESIS_DRAFT_v752_20260709.md`
(commit 5ce62314c). **Surface:** `crucible2_v752`. `[no-triality]` · `$0` · no GPU · #205 UNTOUCHED.
Pointer contest-CPU **0.19110 UNMOVED** — everything here is `[macOS-MLX/CPU advisory]` MEANS.

**Fresh-eyes rule honored:** no seat position is trusted because the synthesis adopted it. Every
load-bearing number below was RE-DERIVED from primary artifact (code / reports JSON / DAG), not
memo-trusted. `review_status: self-executed, fresh-eyes-UNREVIEWED` (P5/P6 treat any adopted
disposition as a finding-producing round).

**STORES CONSULTED:** the DRAFT (all 566 lines) · ORCHESTRATION_LEDGER (incl. §110-118 POSE GATE +
the mid-run OPERATOR WALL-CLOCK ADDENDUM) · positions S1/S5 (re-read for the two most load-bearing
derivations: σ* + the unification) · PRIMARY CODE re-derived, not trusted:
`src/tac/witness_control/jacobian_basin.py` (L246-272 `aggregate_conditioning` + `would_have_fired`
running-MAX ratchet) · `src/tac/canonical_equations/pose_jacobian_basin_conditioning_20260709.py`
(L89-127 `would_fire_basin` f_basin=1.0; `median_sigma_min_ce_ep1=0.0633`, `basin_frac_ce=1.0`) ·
`src/tac/witness_control/resume_registry.py` (L67-99 `GATE_KEY_PREFIXES` — F-1 temporal-screw fix
LANDED; registration is UNCONDITIONAL at startup) · `experiments/train_levelset_witness_realized_
through_R_mlx.py` (argparse: `--jacobian-basin-f-basin` default 1.0 @L10242; `--pose-finish-start-epoch`
@L9977; `--seg-horizon-margin-*` @L10808-10833; `_signed` #141 shared field @L4214/4361; speed flags
`--fused-r-kernel/--micro-batch-pairs/--safe-compile-*/--async-verdict` EXIST; **NO `--pose-finish-
engage-on`**) · `src/tac/boundary_math/length_sigma.py` (EXISTS) + `curriculum_dsl.py:3468` `LengthSigma`
(trainer flag default `all-ones`=byte-identical) · `reports/delta_R_noise_floor.json` (δ_R=0.019590) ·
`reports/levelset_pose_gate_l7best_20260701.partial.jsonl` (real σ_top6 = [0.063, …, 1.2e-4]) ·
`docs/operating_manual_craft_handoff.md`. **Not a store:** no new measurement taken; #205 STOPPED,
box untouched, reading+arithmetic only.

**verdict_scope discipline:** every negative carries `verdict_scope: INSTANCE|FORMULATION` and, where a
finding is a magnitude call, a measurement cite (never eyeball).

---

## HEADLINE (answer-first)

**Attacks run: 21. BROKEN 1 · WEAKENED 10 · HELD 10.**

**THE SINGLE MOST DANGEROUS FINDING (A-1, BROKEN):** the repaired pose-gate's *"DERIVED stable floor"*
σ* = √(C/(δ_seg·λ_min(F))) is **numerically unreachable by 224×–1900× on the ONLY measured σ_min
anchors**, and the owed λ_min(F) probe CANNOT rescue it because λ_min(F)≤0.25 is a hard spectral bound
(F=diag(p)−ppᵀ is a categorical covariance). With C=25, δ_seg=0.5, and the largest physically possible
λ_min(F)=0.25, **σ* ≥ √(50/0.25) = 14.14**; measured σ_max(J_ξ)=0.063 and σ_min(J_ξ)=1.2e-4
(`levelset_pose_gate_l7best_20260701.partial.jsonl` + the ce ep1 median 0.0633 in the canonical anchor).
The identifiability condition σ_min ≥ σ* is UNSATISFIABLE by 2–6 orders of magnitude regardless of the
owed probe. Consequences that break the draft as written:
1. The σ* gate **structurally never fires** → pose always ships banked R1 → the entire in-basin
   pose-finish apparatus (A.4c, D8 sequencing, the "σ_min amplifier" head-solve) is **decorative**.
2. The **R1 positive-control canary structurally FAILS** (R1's σ_min is also O(1e-4) ≪ 14) → A.4 Repair 2
   declares the gate UNTRUSTED → "NO pose disposition admissible."
3. **A.4 is internally CONTRADICTORY on the canary-fail branch:** Repair 2 says untrusted-gate ⇒ no pose
   disposition (reads launch-BLOCKING), while the never-reached fallback says ship banked R1 (reads
   PROCEED). Owed-item #1 is LAUNCH-BLOCKING + requires the canary to fire pre-launch — which it cannot.
   So as written, **the launch-blocking gate cannot pass its own pre-launch assertion.**

The fix is available IN the draft and must be PROMOTED: make the **RELATIVE relaxation-fit plateau**
(σ_min ≥ σ_min(∞)·(1−δ), scale-free — S5-N1(1) / A.4 "cross-check") the **PRIMARY** trigger and DEMOTE
the absolute σ* to (at most) an advisory sanity note. A.4 Repair 1 currently has this exactly BACKWARDS
(*"Primary threshold = S1's σ*; the relaxation-curve fit is the cross-check"*). Until reordered, D3 is
BROKEN and pose is de-facto banked-R1-only — which is FINE as a floor but must be stated plainly, and the
σ* build must NOT gate the launch.

---

## §1 PROVENANCE AUDIT (load-bearing claims → primary artifact)

| claim (draft) | traced to | verdict |
|---|---|---|
| σ_cc′ built OPTIMAL-FORM (3571e5b65), σ≡1 byte-identical, wired main-loss+micro-batch twin | `length_sigma.py` EXISTS; `curriculum_dsl.py:3468 LengthSigma`; trainer @L1423/1454/4269 (`sigma[top1,top2]` on the length term); flag default `all-ones` | **HELD** (build real) |
| δ_R=0.0196 ⇒ m_safe=3·δ_R=0.06 (the "only DERIVED threshold") | `reports/delta_R_noise_floor.json:27` δ_R=0.019590 → 3× = 0.0588 | **HELD** |
| Force-2 & #169 ride the SAME #141 `_signed` field | trainer L4214 (horizon, "SHARED realized _signed") + L4361 (mfh, "on the SHARED _signed") | **HELD** (math); realization WEAKENED (W-2) |
| taper +18%→−8% flip is a re-interpretation, not a new measurement | draft L90 labels it **ESTIMATED / RANK-1-rests-on-flip / verdict_scope INSTANCE**; S5-N2c same | **HELD** (handled HONESTLY — not laundered) |
| rel-sig 73/43.8/31.6 non-additive (148% trap) | draft A.3 adopts S5-N2 verbatim ("PRIORITY HINT, NEVER a ΔS budget", re-graded, "DO NOT SUM") | **HELD** |
| ep257 CE→tau ckpt preserved (FRESH+read decision) | DAG L11303 `levelset_resume_stageTau_ep257.npz`; `v75_dynamic_curriculum_audit` CE→tau fired ep257/plateau | **HELD** (leg); caveat W-6 |
| σ* = √(C/(δ_seg·λ_min(F))) "DERIVED stable floor" that will fire | C=25/δ_seg=0.5 from amber; σ_min telemetry 0.063→1.2e-4; λ_min(F)≤0.25 hard bound ⇒ σ*≥14.14 | **BROKEN** (A-1) |
| λ_lane 683.8 / λ_movable 322.6 "DERIVED-LIVE" | formula W_birth/(δ·A_GT_c) in draft; **computing artifact NOT locatable** by grep | **WEAKENED** (W-7) |
| S2 compose-safety "from the compose-guard CODE (term_domination alarm keys on loss-share)" | `term_domination` appears ONLY as config COMMENTS (`curriculum_dsl.py:2944`, `witness_autoconfig.py:2628`); **NO runtime alarm in `confound_gates.py`/`witness_control`** | **WEAKENED** (W-3) |
| speed stack composed in launch-1 | flags EXIST; **draft §B composes NONE** (fused-R/grouped only in head-solve context) | **WEAKENED** (W-9/W-10) |

Grade-labeling audit: the draft's evidence-grade tags (MEASURED / DERIVED / ESTIMATED / ORACLE-CEILING)
are, where I could trace them, **honestly applied** — S5's N2 grade-laundering indictment was ABSORBED,
not papered over. The one systematic mislabel is the σ* "DERIVED" tag (A-1): the FORM is derived; the
numeric plug-in is uncalibrated-and-unreachable, which "DERIVED" reads as settled.

---

## §2 DECISION ATTACKS (the prompt-directed set + D-table)

### A-1 — Repaired pose-gate σ* floor — **BROKEN** (see HEADLINE; the most dangerous finding)
σ*≥14.14 vs σ_min≤0.063 ⇒ never-fires; canary structurally fails; A.4 canary-fail branch is
block-vs-ship-R1 CONTRADICTORY; owed-item-#1 launch-blocking gate cannot pass its own assertion.
`verdict_scope: FORMULATION` (this absolute-σ* formulation of the conditioning gate — the RELATIVE
relaxation-fit formulation is viable and already half-written in A.4). Fix: promote the scale-free
relaxation-fit to PRIMARY, demote σ* to advisory, resolve the canary-fail branch to ship-R1 (NOT
launch-block), and DEMOTE the σ* build off the launch-blocking path. Measurement cite:
`reports/levelset_pose_gate_l7best_20260701.partial.jsonl` σ_top6 + canonical anchor median 0.0633 +
the λ_min(F)≤0.25 spectral bound. `# MAGNITUDE_DISMISSAL_OK: measured σ_top6 + hard categorical-cov bound`.

### W-1 — "all four Class-A levers carry ZERO seg-gradient-share" — **WEAKENED** (D1 / D7)
Self-orient, #121 taper, AA-ipe are genuine representation (0 loss-share — HELD). **σ_cc′ is NOT:** the
code re-weights the **length LOSS term's** per-pixel `sigma[top1,top2]` gradient (trainer L1454/4269).
"ZERO NEW loss-share" (no new term) is TRUE; "zero loss-share / out of §9's domain / Class-A" is a
**synthesis RECLASSIFICATION that reverses the originating seat** — S1 filed σ_cc′ as an ENERGY TERM /
REGULARIZER (position table row 5 / missing-term row 5) and explicitly assigned it *"its own increment
A/B"* (S1 §4). σ_cc′ acts on the SAME Road↔Lane annulus boundary as the area-Lagrange and the margin
hinge ⇒ it **can confound their attribution** (not term-dominate — length weight 0.001 is tiny — but
confound). D1's "orthogonal-in-G, ×3 independent" is therefore ×2-for-σ_cc′ (S1 said isolate; S6 is about
representation-not-stage which σ_cc′ is not). `verdict_scope: FORMULATION` (σ_cc′'s launch-1 composition
status). Owed: either isolate σ_cc′ to its own increment (S1's call) OR add an attribution guard that the
σ≡1-vs-fitted A/B is run BEFORE any other length-touching lever moves.

### W-2 — Force-2 ∪ #169 = ONE hinge — math **HELD**, realization **WEAKENED** (D4)
The `relu(m_safe−m_wit)`-on-shared-`_signed` identity is CODE-CONFIRMED (both branches use `_signed`).
BUT they are **two distinct code branches** (`hz_w` horizon @L4214 vs `mfh_w` satisfice @L4361) with
DIFFERENT spatial masks (horizon rows vs annulus) and defaults (`--seg-horizon-margin-lo` default 0.3,
not 0.06). "ONE hinge, spatial = annulus∩horizon" needs a wrap NO single existing flag provides — the
draft admits this (A.7 OWED-CHECK). So: unify the MATH now, but the launch-1 realization is either
horizon-with-lo=0.06 (loses annulus) or mfh-with-annulus (loses horizon) until the intersection wrap
lands. `verdict_scope: N/A` (positive unification; the caveat is realizability, owed-item #9).

### W-3 — S2's "compose-guard code" provenance — **WEAKENED** (D1 / A.1)
The `term_domination` runtime alarm the draft leans on ("a lever with no loss term cannot term-dominate
→ §9's term_domination leg is vacuous") **is not implemented** as a runtime guard (grep: only config
comments; `confound_gates.py` has no such alarm; CLAUDE.md lists it as an INTENDED L1 alarm). Compose
SAFETY rests on the DERIVATION (representation adds no term), which is sound for pure-representation
levers and FALSE for σ_cc′ (W-1). D1's third independent leg (S2-from-code) is really S2-from-design-
concept. `verdict_scope: FORMULATION` (the code-provenance of the term_domination leg, not the
representation-adds-no-term math, which HELDS on its own).

### W-4 — Dry-start gate: crash-class **HELD**, "injection-per-event" **WEAKENED** (N3 / owed #2)
GOOD NEWS the draft/S5 under-credit: the F-1 crash is a `KeyError` at `run_train` STARTUP inside
`build_gate_resume_registry` (gates registered **unconditionally at startup** — `resume_registry.py`
comment). So ANY dry-start that reaches ep1 **already caught it** — the 6-gate-key startup-crash class is
covered by a short live dry-start (HELD). The **WEAKENED** part: the draft conflates that startup smoke
with *"fires-when-should + silent-when-shouldn't injection per event"* — the BEHAVIORAL injection (does
the pose-finish/σ_min/decoupling event fire correctly) is NOT delivered by a short run and is aspirational
in owed-item #2. Second gap: the OWED `--pose-finish-engage-on` conditioning gate MUST be registered
UNCONDITIONALLY at startup (like temporal-screw) or a short dry-start misses its prefix and F-1 recurs at
the terminal boundary. `verdict_scope: FORMULATION` (the injection-completeness claim; the startup-crash
gate is real). Owed: (a) state the dry-start is a STARTUP + all-gate-registration completeness gate; (b)
require the new engage-on gate to register at startup; (c) keep the per-event injection as a SEPARATE
owed test, not folded into the smoke.

### W-5 — Amber launch-blocking, Muon-exemption + C value — **WEAKENED** (D2)
`per_group_grad_clip_exempts_muon` is an **OWED build/config** (tagged `[OWED build/config]` in §B L380;
not in code). The eps_floor(C)=(5/C)² throttle-vs-blowup tension (S5-N4b) is a legitimate un-A/B'd A/B the
draft folds as an instrumented arm — honest, but the design-fix is UNBUILT and the "amber is DERIVED not
optional" claim (S1 §2) is sound for the CAUSTIC but does not license the specific constants. `verdict_
scope: FORMULATION` (amber×Muon interaction + C). Owed: the design-fix must LAND before the first joint
arm, and the instrumented per-term-effective-LR arm is genuinely launch-blocking (a normalized-flat Muon
update is invisible to gnorm_hijack/spike-guard).

### W-6 — FRESH default + ep257-read equivalence — **WEAKENED** (D5)
ep257 CE→tau ckpt EXISTS and reading its part_frac IS the stage-1 mass verdict (HELD leg). But: (a) the
mass-conservation "warm = wrong basin" argument is **DERIVED, not measured** — both fresh (N8 strangle-
birth) and warm (N7 inherit-floor) have UNMEASURED symmetric risks; the draft picks fresh on the DERIVED
argument + the F2-resume-refused-basis argument (the latter is the STRONGER, code-grounded leg). (b)
**Process inversion:** the operator asked to *read the verdict THEN decide* warm-vs-fresh; the draft
pre-decides FRESH and demotes the read to a confirm — defensible given the F2-refuse leg, but it should be
named as a departure from the operator's stated sequence. (c) ep257 is the PLATEAU-triggered boundary of a
run STOPPED for false-green — its birth may be UNDER-completed, so "the stage-1 birth verdict" inherits
that incompleteness. `verdict_scope: FORMULATION` (warm-from-floored-basin disposition). Owed: state the
fresh default rests on the F2-resume-refuse leg (measured) PRIMARILY, mass-conservation SECONDARILY.

### W-7 — λ_lane 683.8 / λ_movable 322.6 "DERIVED-LIVE" — **WEAKENED** (A.5 provenance)
The equilibrium formula is stated but the computing artifact (the DERIVED-LIVE value at config from loaded
GT) was **not locatable** by grep. Likely real (matches the 1.25×GT / ~96% deficit story) but "DERIVED-
LIVE" is the strongest tag and needs its artifact path. `verdict_scope: INSTANCE` (this provenance cite).
Owed: cite the FEED/report that computes 683.8/322.6, or downgrade to DERIVED-AT-CONFIG.

### W-8 — Head-solve "σ_min amplifier" for pose — **WEAKENED** (D8)
"The head solve sharpens the boundary → σ_min↑ = a CONDITIONING AMPLIFIER for pose" is **INFERRED, not
measured** — no anchor shows the ~791-param affine solve raises σ_min(J_ξ). Given A-1 (σ_min must climb
~2 orders to reach any usable gate), this claim is load-bearing for the pose story yet unmeasured. Low
BLAST (ρ-gated, terminal, solve-fallbacks-to-train) but the causal claim should be labeled INFERRED.
`verdict_scope: FORMULATION` (the amplifier mechanism). Owed: measure σ_min before/after the solve on one
ckpt (cheap, reuses the jacobian_basin probe).

### W-9 — WALL-CLOCK (operator axis a): speed stack NOT composed with neutrality proofs — **WEAKENED**
The built score-neutral speed flags ALL EXIST (`--fused-r-kernel`, `--micro-batch-pairs`,
`--safe-compile-manifest/-regions`, `--async-verdict`) but **§B composes NONE of them** — fused-R and
grouped-backward appear ONLY in the terminal head-solve context (L237/460/461), the "micro-batch twin"
mention is the σ_cc′ correctness path (not the #313 speed lever). The lexicographic rule BINDS: each must
land with a bit-exact / measured-neutral receipt. **Critical caveat:** `--micro-batch-pairs` is NOT a free
neutral lever — #313 established batch-DEPENDENCE (2.26e-2 drift / 11 flips; S5-N6) — so it needs an
explicit neutrality proof for the NEW render or it is a SCORE-AFFECTING change masquerading as speed.
`verdict_scope: FORMULATION` (the launch-1 config's speed composition). Owed: add a `speed:` block to §B
composing {fused-R [bit-exact, L70], grouped-backward [~17×, L45], safe-compile [fingerprint-verified],
async-verdict} EACH with its neutrality receipt; micro-batch-pairs ONLY with a re-proven bit-identity
(else OFF).

### W-10 — WALL-CLOCK (operator axis c): no per-stage wall-clock budget — **WEAKENED**
The draft has **no ~42s/ep baseline × epochs → hours** estimate anywhere. The min_stage floors clamp
[150,400] × ~5 stages × ~42s/ep ≈ **8.7–23 h in FLOORS alone**, plus ~3 h CPU head-solve — a multi-day run
whose wall-clock is emergent, not designed. `verdict_scope: INSTANCE` (the missing budget). Owed: a
per-stage wall-clock budget (floor-epochs × 42s/ep + measured lever throughput deltas) so total wall-clock
is a designed quantity and the byte-close cadence (A.10) is costed.

### HELD set (survived)
- **H-6 (N3 core):** dry-start catches the F-1 startup-KeyError class (registration unconditional). HELD.
- **H-8 (D10):** S6's BLIND 6-block skeleton converging to the incumbent spine with zero PR95 stage-echo
  is a genuine seal signal (the anti-cargo-cult LAW's own test passed). HELD.
- **H-9 (D9):** torch-twin NO-GO is sound — a torch-trained arm ≠ MLX-trained arm measures the ENGINE
  (NO-FAKE #8); archive-level exact-eval is the parity-safe boundary (S5-N11 agrees). HELD. (Minor: the
  "~$0.25/paired-row" is an ESTIMATE not a quote — owed a live provider quote before the $20 spend.)
- **H-3/H-4/H-5/H-7** per §1 table.
- **Operator axis b (structural wall-clock): HELD-partial** — the design DOES exploit event-exits vs
  fixed floors (CE→tau fired ep257 vs cap 300 = 43 ep saved) and solve-replaces-train (J5 head solve
  deletes a terminal fine-tune stage). NOT exploited: verdict/telemetry cadence tuned to information value
  (byte-close cadence is promotion-driven, not compute-cost-driven). Partial credit.

---

## §3 THE THREE MOST DANGEROUS (probability × blast × silence)

1. **A-1 (σ* unreachable → launch-blocking gate that cannot pass).** New operator constraint, least-
   instrumented surface, and BOTH failure arms are silent: never-fire ships byte-identical-to-incumbent as
   "success," and the pre-launch canary that is supposed to catch it ALSO structurally fails. Fix the
   ordering (relative-plateau PRIMARY) + resolve the canary-fail branch (ship-R1, not block) BEFORE launch.
2. **W-1 (σ_cc′ mis-classified Class-A).** It is IN launch-1 as "zero loss-share" but modifies the length
   loss term on the exact Road↔Lane annulus the area-Lagrange and margin hinge also touch — silent
   attribution confound the composed launch cannot separate; reverses S1's own "isolate" call.
3. **W-9 (micro-batch-pairs as a "free" speed lever).** Under the lexicographic rule, composing
   micro-batch-pairs for wall-clock WITHOUT re-proving bit-identity for the new render (batch-DEPENDENCE
   is MEASURED, #313) trades score for speed — the exact thing the operator binding forbids.

**Meta (attack my own pass):** A-1 rests on the assumption that "median_sigma_min" in the basin telemetry
is on the same scale as σ_min(J_ξ) in the l7best JSONL — they ARE (both O(0.06) at ce), and the λ_min(F)≤
0.25 bound makes σ*≥14 robust to ANY probe outcome, so A-1 does not depend on a scale-matching assumption.
The one place I could be wrong: if the intended σ-comparison rescales σ_min (e.g. σ_min/σ_max, a condition
number ~1/8000), the numbers shift — but the draft's A.4 writes the RAW σ_min ≥ σ* comparison, so on the
spec as written A-1 stands. Every "owed" above is unreviewed new work; P5 must re-derive, not trust.

**Pointer 0.19110 UNMOVED — this red-team is MEANS. Only a byte-closed `upstream/evaluate.py` n600 row
< 0.19110 moves it. Disposition to P4/P5: A-1 is a launch-blocking BROKEN that has a fix already latent in
the draft (relative-plateau primary); the 10 WEAKENED each name a specific owed addition; the draft is
NOT a dead launch, but it ships pose-blind-with-a-decorative-gate and wall-clock-blind unless A-1 is
re-ordered and a costed speed block + budget land.**

## STORES CONSULTED (line)
DRAFT SYNTHESIS_DRAFT_v752 · ORCHESTRATION_LEDGER (§110-118 pose-gate + wall-clock addendum) · positions
S1_deepmath / S5_adversary · PRIMARY CODE (jacobian_basin.py · pose_jacobian_basin_conditioning_20260709.py
· resume_registry.py GATE_KEY_PREFIXES · train_levelset_witness_realized_through_R_mlx.py argparse+_signed+
length-sigma+speed flags · length_sigma.py · curriculum_dsl.py LengthSigma) · reports/delta_R_noise_floor.json
· reports/levelset_pose_gate_l7best_20260701.partial.jsonl · DAG sub015 (ep257 ckpt) · v75_dynamic_curriculum_audit
· docs/operating_manual_craft_handoff.md. $0, no GPU, #205 UNTOUCHED, [no-triality].
