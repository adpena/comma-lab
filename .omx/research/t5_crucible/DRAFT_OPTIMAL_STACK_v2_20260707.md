---
doc_type: t5_crucible_p3b_revised_synthesis_draft_v2
role: P3b REVISION DESIGNER (operator-convened T5 crucible)
date: 2026-07-07
supersedes: DRAFT_OPTIMAL_STACK_20260707.md (v1, 467daadd2) — v1 preserved append-only; v2 is the
  launch-candidate synthesis. Every P3 finding F1-F17 has a disposition (§0.1).
inputs: P3_redteam_verdict_20260707.md (ALL F1-F17 + PASS-3 + operator counter-frame adjudication)
  · v1 draft · ORCHESTRATION_LEDGER (reqs A-G + counter-frame entry + continuation queue) ·
  pursuit_chainA (CURRENT state: LINK 0/1/2/2b/2c/3b LANDED; K=128 eigen + LINK-3 full protocol
  branch carried) · islands_composed_ceiling_arithmetic_20260707.md (S5-R5 — LANDED, GATE PASSES)
  · CONTEXT_COMPENDIUM (FEED-07g compose-after-downsample; FEED-08k) · $0 co-predicate BACKTEST
  run THIS SESSION on the mod32cap 41-row verdict trace (results §2.2b).
epistemic_contract: unchanged from v1 — every knob carries a CONTROL LAW class {(a) CONSTANT ·
  (b) RAMP/ANNEAL+completion guarantee · (c) SELF-DERIVING · (d) EVENT-CONDITIONED (req B) ·
  (e) FRACTIONAL/PARTIAL} and a tag {V-S · V-A · D · DPR}. Nothing unmeasured asserted as measured.
axis: all numbers [macOS-CPU/MLX advisory] unless tagged; archive bytes exact (zip stat).
  Pointer contest-CPU 0.19110 UNMOVED — this whole file is MEANS.
---

STORES CONSULTED: P3 verdict (full, F1-F17 + PASS-3 + counter-frame) · v1 draft (full) ·
ORCHESTRATION_LEDGER (reqs A-G, counter-frame entry, continuation queue) · pursuit_chainA log
(current on-disk state) + its artifact dir (gradstep_cross_subset.json, rescreen_negcurvK32_p02.json,
krylov_step_screen_curvtransfer_K32.json, spectrum_ep650gpu_K{8,32}_s0.json; K128 lanczos-state npz
present, spectrum json NOT yet — branch carried) · islands_composed_ceiling_arithmetic_20260707.md
(read in full) · CONTEXT_COMPENDIUM (stores 2/3 targeted: FEED-07g L8991 AA compose-after-downsample
BUILT; FEED-08k grammar; composed-ceiling row) · position_S5 (AA row 6 + R5) · mod32cap run dir
(levelset_train_result.json 41-row history — co-predicate backtest EXECUTED this session;
costate_shadow.jsonl schema) · canonical_equations_registry (AWAITING recount: 22 rows / 21 unique
ids) · trainer argparse spot-checks (--render-aa/--aa-supersample/--aa-self-orient-fine-mode exist;
AACoverageRender factory exists in curriculum_dsl L2179) · CLAUDE.md non-negotiables ·
docs/operating_manual_craft_handoff.md. NOT consulted: durable-state files (3-8wk STALE per sweep).

# T5 CRUCIBLE — P3b DRAFT v2: THE OPTIMAL FULL STACK (launchable as written; crossing made honest AND engineered)

## §0 — HEADLINE + WHAT CHANGED

**The stack in one paragraph (v2).** One pointer-aimed run (ARM-PRIMARY) over
`Mod32SegOnlyControlBase`: control basis at **along=8**, **islands-first curriculum** (S5 FIRE core
under per-class LADDER laws), **AACoverageRender IN from ep0** (F4 — the stale-seam exclusion is
dead; FEED-07g compose-after-downsample BUILT with byte-identity proofs; two launch gates §7 P11),
**analytic lane band trained-with** (LBND4 booked), **comb conditionally IN on P1-pass** (inclusion
law §1.2), **pose ON two-track with restructured thresholds** (success bar d_pose ≤ 3e-5 — the value
that arithmetically crosses; 1.5e-4 demoted to mid-run milestone/hard-kill), **event-exit schedule
with a REAL TAU→FIN event** (F3 — `--anneal-epochs 600` event-margin law; co-predicate BACKTESTED
this session: first sustained fire ep625 on the mod32cap trace, one cadence before the actual ep650
best), **rate compress-half** (WeightEntropy λ=15; **twin re-purposed to λ=0** for single-dim
attribution — F7), **SOLVE gated on measured-acceptance only** (F8 — chain-A's landed verdict: no
cheap descent exists at ep650; TerminalSolve-from-ep650 measured NO-GO), byte-close → contest-CPU.

**The honest crossing story (F1, the one-paragraph version).** The crossing condition, computed
from the measured rate moat: **100·d_seg + √(10·d_pose) < 0.19110 − rate ≈ 0.129 (central rate
0.0620) to 0.134 (rate lower tail)**. v2 engineers the central case toward it and prints the
frontier honestly: the verified crossing triple is **d_seg ≤ 0.0011 ∧ d_pose ≤ 3e-5 ∧ rate ≤ 0.062
→ S ≤ 0.1893 < 0.19110** (arithmetic in §0.2). v2's central (0.0016, 1.5e-4, 0.062 → S ≈ 0.26)
still does NOT cross — but v2 exhibits the TWO binding constraints that gate the crossing case and
instruments both in-flight: **(i) the big-3 anneal-completion recovery** (~4-9e-4 of d_seg;
UNMEASURED — the control's best sat on a truncated anneal 0.216/3.177, v2 completes 0.2/4.0 by
ep600) and **(ii) lane composed-band efficacy trained-with** (post-hoc measured NEUTRAL; trained-with
never measured — P8 + F8 attribution). Crossing is a small-but-instrumented tail under the
independent model and a materially better one under the sequential-with-repair model (§9.3, both
printed per F17). This is option (a)+(b) of F1's fix: ceiling-raising levers IN, and the run
re-scoped as the two-wall measurement run whose crossing probability is real, printed, and steered.

### 0.1 Finding-by-finding disposition (F1-F17 — the revision contract)

| finding | disposition in v2 |
|---|---|
| F1 BLOCKER crossing arithmetic false | FIXED §0.2: frontier recomputed from measured rate; pose success bar moved to 3e-5 (1.5e-4 → milestone/hard-kill); d_seg design target = islands ceiling WITH AA+band+comb in; band lower edge recomputed from consistent tails (§9.1) |
| F2 req-A(ii) #342 inventory | PRODUCED §11 (per-block solved / trained-with-reason / not-solvable-with-proof) |
| F3 vacuous TAU event exit | FIXED §2.2: `--anneal-epochs 600` (event-margin law — anneal completes BEFORE earliest measured exhaustion ep625); §8 recomputed arithmetically consistent; 35% claim corrected to 5-27%; optional B9 re-anchor law named |
| F4 AA stale-seam exclusion | AA IN ARM-PRIMARY from ep0 (§1.2); two gates get the B6 treatment: P11 $0 memory+throughput gate (launch-blocking probe) + BA AA-decode inline (LB at byte-close); composed-ceiling arithmetic re-run with AA (§0.3) |
| F5 dropped S5-R5 | MOOT-BY-LANDING: `islands_composed_ceiling_arithmetic_20260707.md` LANDED — GATE PASSES, band [0.02, 0.26] ΔS, island share 0.562 (lane .44/movable .12); rung-1 recomputed on it (§0.3, §9.1); P6 flip-share still rides |
| F6 co-predicate no backtest; M3/M4 rows non-LB | BACKTEST RUN this session (§2.2b): 5e-3/V=4 first-fire ep625; CE window never fires <300 (conservative — CE exit is cap-fired in practice, now STATED); F3/F4 telemetry rows PROMOTED to LB (§4.1) |
| F7 attribution double-booking; λ=15 seams | Twin RE-PURPOSED to λ=0 (pose ON) — single-dim λ attribution + clean Class-D×B bytes recess; pose-interference attribution routed to run-2 with confound named; every per-stage kill restated STACK-level (§7 RUN row) |
| F8 chain-A instrument caveat; SOLVE acceptance | FOLDED §2.3: SOLVE requires measured-verdict acceptance (HARD); basin predicate annotated advisory-on-STE-smoothed-operator; chain-A landed results folded (K-ladder collapse, transfer test, isotropic u_min, no-cheap-descent) — the K=8 indefiniteness verdict is OVERTURNED as subset-idiosyncratic |
| F9 ep0 abort gate miscalibrated | FIXED §3.1: gate re-predicated on mechanical seed application + init-d_seg effect + movable part_frac; lane part_frac moves to an ep150 ALARM milestone; P12 $0 one-epoch init probe of the EXACT seed config pre-GO |
| F10 q-law backwards | FIXED: **smallest** q with Δ(pose-term) < 0.002 (§1.2) |
| F11 byte-band inconsistency | FIXED §5.1: component-consistent band [70.4K, 103.5K] printed; narrowed central band restated as correlated-tails judgment; worst joint tail (waterfill-fail ∧ B6-slip) 128,376 B → rate 0.0855 printed |
| F12 chroma mislabel | FIXED §1.1: ChromaBoundarySharpen grouped under score-affecting loss levers |
| F13 borrowed throughput | 5-ep governed throughput smoke at the EXACT ARM-PRIMARY config folded into P11; §8 numbers tagged pending-P11 |
| F14 launch surfaces unnamed | §7 RUN row names `tools/launch_witness_run.py` + `witness_memory_preflight` at the REAL config |
| F15 ordering | §7 reordered cheapest-decisive-first (P5 1-min first; P7 decode-integrity second) |
| F16 provenance upgrades | τ_e=305 re-tagged INFERRED (11-pt extrapolation; B4 guard carries the weight); LBND2 36 B discrepancy resolved at P5; AWAITING recount DONE: 22 rows / 21 unique ids (v1's "15" corrected) |
| F17 dual probability model | §9.3: BOTH bands printed (joint-independent-tail AND sequential-with-repair, per-lever repair mechanisms + non-repairables named); honest central adjudicated between them |
| PASS-3 #149 disposition | NAMED §9.4 run-2 list: closed-form facet placement at 874-before-R — DEFER-with-build-spec (partially represented by AA's sub-pixel coverage in run-1) |
| PASS-3 comb inclusion law | §1.2: P1-PASS ⇒ comb rides ARM-PRIMARY in-training (engage band-fire+25, boundary-relative, F8 paired row, kill law); P1-FAIL/unrun ⇒ OFF |

### 0.2 THE CROSSING ARITHMETIC (F1 fix — recomputed end-to-end, every point verified)

Budget form (the counter-frame's frontier): `100·d_seg + √(10·d_pose) < 0.19110 − rate`.

| rate leg | budget for (seg+pose) |
|---|---|
| central 0.0620 (waterfill-pass) | **0.1291** |
| band lower 0.0573 | 0.1338 |
| component-consistent lower 0.0469 (§5.1) | 0.1442 |
| waterfill-fail 0.0768 | 0.1143 |
| worst joint tail 0.0855 | 0.1056 |

Verified crossing points at central rate 0.0620 (all computed, not asserted):

| d_seg | d_pose | 100·d_seg + √(10·d_pose) + 0.0620 | crosses 0.19110? |
|---|---|---|---|
| 0.0011 | 3.0e-5 | 0.110 + 0.01732 + 0.0620 = **0.18932** | **YES** (margin 0.0018) |
| 0.00105 | 5.8e-5 | 0.105 + 0.02408 + 0.0620 = 0.19108 | YES (knife-edge) |
| 0.00092 | 1.51e-4 | 0.092 + 0.03886 + 0.0602 = 0.19106 | YES (the S6 triple, at rate 0.0602) |
| 0.0012 | 1.5e-4 | 0.120 + 0.03873 + 0.0620 = 0.22073 | **NO — v1's printed "crossing tail" (the F1 bug)** |
| 0.0016 (v2 central) | 1.5e-4 (milestone) | 0.160 + 0.03873 + 0.0620 = 0.26073 | NO (v2 central, stated plainly) |

**Consequences engineered into v2 (not just stated):**
1. **Pose success bar = d_pose ≤ 3e-5** (term 0.0173). At that bar, d_seg ≤ 0.0011 suffices at
   central rate. The 1.5e-4 level is DEMOTED to (i) a mid-run milestone (d_pose < 1.5e-4 by
   tau-end, ALARM on miss) and (ii) the hard kill (converged > 1.5e-4 ⇒ pose-as-designed KILLED →
   L1 Jacobian fallback). Between 3e-5 and 1.5e-4: pose bytes stay (they still pay per §6), but the
   run-1 pointer-crossing claim is dead and stated so. Existence proof for the 3e-5 class: 3.4e-5
   measured on THIS frozen scorer by a different vehicle (NEVER cited as witness-solved, per L68 —
   optimism about the mechanism, rigor about the claim).
2. **d_seg design target ≤ 0.0011 = the islands-ceiling region WITH the ceiling-raising levers
   in** (AA + band trained-with + comb-on-P1-pass) — arithmetic §0.3.
3. **Rate ≤ 0.062 = waterfill-pass** — hence P4 is launch-relevant, unchanged.

### 0.3 The d_seg design arithmetic (S5-R5 LANDED + AA in — rung-1 re-grounded)

From `islands_composed_ceiling_arithmetic_20260707.md` (MEASURED composed per-class, ep300 surface,
16-pair advisory subset +4.7% vs n600; shares ASSUMED stable → P6 rides): islands = **0.5622** of
composed d_seg (lane 0.4396, movable 0.1226); big-3 jitter = 0.4378. Ceiling band for island
treatment [0.02, 0.26] ΔS — **GATE PASSES** (lower edge ≥4× the 0.005 stop gate).

Per-class design table on the control-best basis d_seg 0.0034 (shares transferred — ASSUMED):

| class | component | run-1 lever set | design residual band |
|---|---:|---|---|
| lane | 0.00150 | band trained-with (isolated analytic bound 0.00087-class) + VP-tangent eased seed + AA sub-px + comb (P1-cond.) | [0.0004, 0.0009] |
| movable | 0.00042 | eased SDF-dilation (PROVEN transfer) + AA + logit-adjust −4.39 | [0.0001, 0.0003] |
| big-3 | 0.00149 | **τ/β anneal COMPLETION (M2 recovery — UNMEASURED, the binding constraint)** + AA sub-px + chroma-at-annulus + eikonal ramp | [0.0005, 0.0011] |
| **sum** | 0.0034 | | **[0.0010, 0.0023], central ≈ 0.0016** |

**The binding constraint, exhibited (the operator's derive-or-cross demand):** crossing needs the
sum ≤ 0.0011 — every class within ~15% of its optimistic edge simultaneously. The least-instrumented
leg is the **big-3 anneal-completion recovery**: mod32cap's best (0.0033662) sat on a truncated
anneal (τ 0.216/0.05, β 3.177/4.00 — M-S2-2); no probe has measured what completing it recovers.
Second: **lane composed-band efficacy trained-with** (post-hoc = NEUTRAL measured, FEED-dv;
trained-with unmeasured — P8 + F8). v2 instruments BOTH per-class in-flight (F-rows); the run IS
their measurement. Islands-only arithmetic honesty: full island fix alone floors at
0.0034×(1−0.562) ≈ 0.0015 — **below 0.0015 is reachable ONLY through the big-3/anneal leg**, which
is why the F3 fix (a real, completed anneal) is a crossing enabler and not schedule hygiene.

---

## §1 — THE EXACT WitnessProgram (v2 deltas from v1 marked ★; everything else inherited verbatim)

Compiled through `tac.witness_dsl.curriculum_dsl` over `Mod32SegOnlyControlBase`. All v1 flag
verifications stand (251-flag extraction, zero invented); v2 additions verified this session:
`--render-aa` / `--aa-supersample` / `--aa-self-orient-fine-mode` exist in the trainer;
`AACoverageRender(ss=2, …)` factory exists (curriculum_dsl L2179).

### 1.1 Program sketch (★ = changed vs v1)

```python
prog = WitnessProgram(
    purpose="T5 crucible ARM-PRIMARY v2: pointer-aimed islands+AA+band+pose full stack",
    base=Mod32SegOnlyControlBase(),
    curriculum=sealed_205_curriculum(cfg, handoff="event"),   # + laws in §2 (anneal-epochs 600 ★)
    levers=[
        # ISLANDS core (S5 FIRE set; laws §3)
        SeedIslandBirth(), SeedIslandEased(), EventTriggeredCurriculum(),
        LogitAdjust(tau=1.0), AmplifyIsland(form="hinge", weight=1.0, gated="witness_alone"),
        PersistenceTopology(weight=1.0, warmup=275), CacheGtSkeleton(), LengthSigma("fitted-20260707"),
        # RENDER SUBSTRATE ★ (F4: seam RESOLVED per FEED-07g; gates P11 + BA)
        AACoverageRender(ss=2),                       # ep0-engaged; attribution law §1.2
        # BAND (trained-with)
        AnalyticLaneRenderBand(start=350, boundary_relative=True),   # LBND4 at byte-close (B6)
        # SCORE-AFFECTING LOSS LEVERS ★ (F12: moved OUT of the observability group)
        ChromaBoundarySharpen(weight=0.1, margin_band=1.0, start="tau_fire"),  # NEW stub (I-4)
        # FINISHER
        MuonWarmStart(lr_final_frac=0.1),
        # RATE Class-D
        WeightEntropyPenaltyMLX(lam=15),              # twin runs lam=0 ★ (F7)
        # POSE (two-track; §6) — pose flags via base overrides
        # OBSERVABILITY (req F; default-ON, score-neutral — READ-ONLY rows only)
        GNSpectrumProbe(),                            # NEW factory (build I-5)
    ],
)
prog.validate()
```

### 1.2 Knob → control-law table (v2 DELTA rows only; all unlisted v1 rows inherited unchanged)

| knob | value/law | class | tag |
|---|---|---|---|
| ★ `--render-aa supersample --aa-supersample 2` | **ON from ep0** (render-side half of island birth; zero archive bytes — code-only). **Gates:** (i) P11 $0 fine-mode memory+throughput gate PASSES pre-launch (self-orient×supersample: ss=2 ⇒ ~4× render px; kill = preflight REFUSE or wall-clock >1.5× control — S5's own kill); (ii) BA AA-decode inline = LB at byte-close (same class as the band's B6). **Attribution law (F7/F8-compatible):** at each stage-boundary verdict, render the checkpoint TWICE (aa=supersample vs aa=none) through the same frozen scorer → paired with/without delta at fixed θ (~2× verdict cost, 4 boundaries only). Render-side ⇒ also BYTE-CLOSE-SELECTABLE: ship whichever verdict measures better (a named F17 repair). If P11 FAILS: AA drops to run-2 and the exclusion is DERIVED with the measured cost written (F4's demand either way) | a | V-A (S5 row 6 FIRE, "#1 measured islands lever"; FEED-07g compose-after-downsample BUILT with byte-identity proofs) |
| ★ `--lane-band-dash-comb` | **CONDITIONAL inclusion law (P1-PASS ⇒ IN):** if the $0 comb-REGISTRATION audit (P1) passes pre-launch → comb rides ARM-PRIMARY **in-training** (render-composite is measured net-negative +0.0038 ⇒ in-training is the only admissible form), engaging boundary-relative at band-fire+25 (one-homotopy-per-neighborhood stagger), with an F8 paired-delta row at engage and kill = paired delta > 0 for 2 consecutive windows ⇒ disable + restore-best. P1 FAIL or unrun at GO ⇒ OFF (default) and the run-2 A/B is removed (P1-fail branch) | d | V-A gate (L65 mis-phase risk; FEED-08c/08l comb-best 0.00695 ORACLE-form — P2 fresh-eyes review still owed) |
| ★ ξ coder q_levels (byte-close) | law: **smallest** q with Δ(pose-term) < 0.002 (sweep 256/1024/4096) — F10 direction fix (v1/S6-M4 wording selected max-bytes vacuously) | c | DPR (S6-M4) |
| ★ `--weight-entropy-penalty-lambda` | 15 constant from ep0 (unchanged) — **attribution now real: the twin runs λ=0** (single-dim delta; §6 track 2). Kill: twin-vs-primary d_seg gap > +5% at matched epochs sustained 2 verdict windows ⇒ λ=15 implicated ⇒ twin PROMOTES to primary (a named F17 repair; sequential cost stated §8) | a | DPR (λ*∈{5,15,30} unswept; torch −19.6% mechanism-proof) |
| ★ `--anneal-epochs` | **600** — the event-margin law (F3 fix): denominator = earliest MEASURED tau-exhaustion (co-predicate backtest first-fire ep625, §2.2b) minus one verdict cadence, floored to 600. Anneal (τ→0.2, β→4.0) provably completes BEFORE any admissible TAU→FIN fire (≥600) ⇒ the anneal-complete precondition is satisfiable at an EVENT, not only at the cap — the TAU→FIN exit becomes real. Cap 726 unchanged. Worst case (arm exhausts before 600 — unlikely: the arm engages MORE mid-TAU levers than the backtested control): fire waits for 600, waste bounded ≤ ~50 ep, alarmed by F12 epochs-past-meat | a (binding the b-laws) | D (backtest-anchored) + V-A (M-S2-2 truncation) |
| ★ pose thresholds | success bar d_pose ≤ **3e-5** (crossing-enabling, §0.2) · milestone < 1.5e-4 by tau-end (ALARM) · hard kill converged > 1.5e-4 ⇒ L1 Jacobian fallback (§6) | d | D (score-law arithmetic) |

All other v1 §1.2 rows (basis, bank, hosc β-ramp shape, τ-shape, chroma, mod-dim, islands core,
schedule/optimizer, band settings, w_pose staged law, pose carrier) are inherited UNCHANGED —
including the ordering guard (along=8 primary) and every V-S/V-A anchor the red-team re-verified.

---

## §2 — THE DERIVED SCHEDULE (v2: the TAU→FIN event is now real; chain-A folded)

### 2.1 Stage graph — unchanged from v1 (DAG with DECIDE-selected transitions; cap-path fail-safe)

```
P(prime, ep0) → CE → TAU → FIN(warm-Muon) → END
                         ↘ SOLVE (measured-acceptance gated; §2.3) → END
   FIN --regression-guard-trip--> RESTORE-BEST --DECIDE--> { TAU-continue | SOLVE(if PD∧accept) | END }
   TAU/FIN --per-class-meat-exhausted (all classes)--> END   (build B5)
```

### 2.2 Event exits (req B; v2 status column updated)

| transition | fire predicate | cap | req-B status (v2) |
|---|---|---|---|
| CE→TAU | plateau(rel-eps 1e-4, W=25, min-stage 250) ∧ verdict co-predicate (V=4, rel-eps 5e-3/25ep) ∧ nucleus guard ∧ seed-anneal-complete | 300 | plateau BACKTESTED (ep251); **co-predicate BACKTESTED §2.2b: never fires < ep300 on the control trace ⇒ CE exit is expected CAP-FIRED in practice — stated, not hidden (accept-and-state). The trigger is conservative in the safe direction (cannot truncate CE early). Injection test T-1 still owed before arming** |
| TAU→FIN | tau-plateau (ep_loss ∧ verdict co-predicate) ∧ anneal-complete (β=4.0 ∧ τ=0.2 — reached ep600 by the §1.2 denominator law) ∧ quadratic_basin ADVISORY (§2.3) | 726 | **co-predicate BACKTESTED §2.2b: first sustained fire ep625 — one verdict cadence before the actual measured best (ep650). The event exit is REAL: on the control trace it fires 101 ep before the cap.** B1 in-trainer trigger still owed; run-1 admissible without B1 (out-of-process advisory + cap-fire, anneal complete either way) |
| FIN→END | regression guard: no new best within 100 ep of fire ⇒ restore-best + exit | budget law §2.4 (τ_e anchor re-tagged INFERRED per F16 — the guard carries the weight) | B4 + T-2 owed |
| run-end | per-class remaining-meat < floor for EVERY class | 1000 | B5 + T-3 owed; backtest on mod32cap trace rides the B5 build |

**Optional build B9 (NOT LB):** the anneal re-anchor law — if plateau ∧ co-predicate fire BEFORE
anneal-complete, compress the remaining τ/β path over the 20-ep rewarmup window, then fire (the
red-team's genuinely-event-bound alternative). The 600 denominator makes the event real WITHOUT B9;
B9 removes the residual ≤50-ep wait branch. ~30 LOC + injection test if built.

### 2.2b ★ THE CO-PREDICATE BACKTEST (F6 — run this session, $0, on-disk mod32cap 41-row verdict trace)

Predicate: trailing V=4 n600-verdict rows; rel d_seg slope per 25 ep > −eps_rel ⇒ fire.
Source: `levelset_n600_witness_mod32cap_20260706T115554Z/levelset_train_result.json` history.

| V | eps_rel | first fire | first fire ≥ ep250 | n fires |
|---|---|---|---|---|
| 4 | **5e-3 (shipped)** | **ep625** (rel slope −1.37e-3) | ep625 | 8 |
| 4 | 1e-2 | ep575 | ep575 | 10 |
| 4 | 2e-2 | ep275 | ep275 | 23 |
| 5 | 5e-3 | ep625 | ep625 | 9 |

Reading: at the shipped 5e-3 the co-predicate detects GENUINE exhaustion (fires ep625; measured best
ep650; d_seg was still descending −1.9%/25ep at ep275 and the trigger correctly held). CE→TAU:
cap-fires at 300 (conservative — cannot truncate a descending CE). TAU→FIN: a real event, worth
~101 ep vs the 726 cap on the control trace. The threshold stays 5e-3 (the 2e-2 variant would fire
CE at ep275 — rejected: 25 ep saved is not worth premature-truncation risk on the arm's richer
curriculum). Caveat: this is the CONTROL's trace; the arm engages more mid-TAU levers ⇒ exhaustion
shifts LATER, which the event (unlike a clock) tracks automatically. Backtest artifact: this table
(reproducible one-liner against the on-disk JSON).

### 2.3 Finisher fire + SOLVE (chain-A FOLDED — the landed results change this section materially)

**Chain-A landed verdicts (artifact dir `experiments/results/t5_pursuit_chainA_20260707/`):**
1. **LINK 0 (instrument):** analytic HVP deviates ~35% from FD-true curvature along g (STE-smoothed
   vs real rounding landscape) ⇒ **any SOLVE step is accepted ONLY on MEASURED loss/verdict
   improvement through the real load path (lm_accept-style), NEVER on predicted quadratic
   reduction — now a HARD condition of the SOLVE stage.** The quadratic_basin predicate is annotated
   ADVISORY-on-a-smoothed-operator (durable reason, not just run-1 convenience).
2. **LINK 1 (K-ladder):** λ₋ collapses ~1/K (K=8: ratio 1.3-2.7 → K=32: **0.28**); λ_max K-stable.
   Pre-registered K=128 prediction: ratio ∈ [0.04, 0.14] — at/below the kill band. K=128 Lanczos
   state is on disk; eigen-extraction pending — **branch carried:** if ratio < 0.1 ⇒ the S3 "ep650
   not 2nd-order exhausted" verdict is KILLED at full-P too; if > 0.5 ⇒ indefiniteness is real and
   §2.3 reverts toward v1's framing. Current evidence strongly favors the former.
3. **LINK 2b (transfer):** the K=8 λ₋=−175 direction has |curvature| ≈ 1 on independent holdouts
   (150× smaller, sign-unstable) ⇒ the indefiniteness is SUBSET-IDIOSYNCRATIC.
4. **LINK 2c (structure):** u_min's basis-coupling mass is **ISOTROPIC — decisively** (along-tangent
   enrichment 1.00-1.05 vs expected 1.0). **The honest negative, stated per the red-team's ask: the
   Hessian at ep650 provides NO mechanism-identified grounding for the basis arm.** Arm A's case
   rests entirely on its own measured −48% directional anchor (unchanged, and sufficient); the hoped
   Lanczos shortcut is dead.
5. **LINK 3/3b (steps):** cross-subset gradient steps WORSEN holdout at every η (best +5.3e-5 at
   η=0.003); the one negative-curvature candidate that improved int8-deploy (−6.1e-4) was REJECTED
   on rescreen against a second holdout (+5.8e-4). **No cheap first- or second-order descent exists
   at ep650-EMA at the frozen schedule point ⇒ TerminalSolve-from-ep650 = measured NO-GO; the wall
   is capacity/basis/schedule — which is exactly what ARM-PRIMARY attacks.** Chain-A's own protocol
   (holdout screen → second-holdout rescreen → deploy-params screen) is the SOLVE acceptance
   template, adopted verbatim.
6. **Incidental measured fact:** int8-dequant round-trip alone costs +5.2% surrogate loss ⇒ every
   solve/step/rung claim is screened at deploy params (folded into rung-2 §9.1 and the SOLVE gate).

**v2 SOLVE stage law:** enabled ONLY when (i) quadratic_basin advisory is PD at a NEW basin (a
post-ARM-PRIMARY checkpoint — ep650-mod32cap is measured-exhausted), (ii) full-P in-trainer GPU
solve (subset solves NO-GO per #341 +5.1%), (iii) **measured-acceptance: the stepped θ must beat the
entry best on the n600 verdict through the real load path AND at int8-deploy params, else REJECT and
restore** — the chain-A template. P3's kill bands carry the ~35% instrument gap + the ~2× fp32-path
|λ₋| magnitude fragility (order-of-magnitude only).

### 2.4 Self-deriving schedule parameters — v1 inherited; finisher budget anchor re-tagged:
cap_fin = clamp(1.5·τ̂_e, 150, 350) with τ̂_e ONLINE (F3 row); the τ_e=305 anchor is **INFERRED**
(11-point extrapolation S2 routed to RECESS-4) — the B4 regression guard, not the anchor, carries
the failure class. Interim before first online estimate: 274.

---

## §3 — THE CURRICULUM (v1 inherited; §3.1 gate recalibrated per F9)

### 3.1 Priming (ep0) — ★ acceptance gate FIXED

Measured reality (paintseedON, DAG 8366 + L3/DAG 8420): the paint biases init d_seg (−36%) but
`part_frac[lane]` stays 0 at init — the v1 gate would have ABORTED a healthy launch. v2 gate:

- **Abort-class checks (mechanical + measured, ep0):** (i) the seed op REPORTS painted-px > 0 for
  BOTH classes (mechanical application proof); (ii) init d_seg ≤ 0.8× the unseeded control init
  (the measured −36% class, with margin); (iii) `part_frac[movable] > 0` (movable transfer is the
  PROVEN mechanism). Any failure ⇒ governed abort.
- **Milestone-class check (ALARM, not abort):** `part_frac[lane] ≥ 0.003` (≈50% of GT lane mass
  0.00577) by **ep150** — calibrated from the composed-ceiling memo's measured ep300 value (77% of
  GT mass). Miss ⇒ ALARM + costate advisory (lane birth failing → the lane leg of §0.3 is dying).
- **P12 ($0, one epoch, pre-GO):** init probe of the EXACT ARM-PRIMARY seed config
  (+include-lane / VP-tangent-eased — never measured at ep0); recalibrate (ii)/(iii) constants to
  what the mechanism measurably produces before arming.

§3.2 per-class LADDER laws and §3.3 LevelPaths: inherited from v1 unchanged (denominator updates
per §1.2 flow through automatically since every path is event/cap-bound).

---

## §4 — COSTATE + TELEMETRY (v1 inherited; two promotions per F6)

§4.1 F-rows F1-F12 inherited. ★ **F3 (online meat/τ̂_e) and F4 (trigger would-fire audit) are
PROMOTED to LB whenever any event trigger ships armed** (they are the M3/M4 archaeology classes that
motivated req F; ~55 LOC combined). LB set is now: F1, F2, F3, F4, F9, F10, F11.
§4.2 DECIDE laws inherited; the curvature-aware exhaustion predicate carries the chain-A instrument
caveat (advisory-on-smoothed-operator). §4.3 ACT boundary unchanged (CONTAINMENT).

---

## §5 — THE RATE PLAN (v2: bands made component-consistent per F11)

### 5.1 Byte budget (λ_bytes = 6.659e-7 S/B exact)

Components (central | independent tails): base+code post-waterfill 60,000 [52,000, 68,000] ·
grammar rev-2k −3,108 (measured) · band LBND4 30,892 [18,000 (P5), 30,892] · pose ξ 4,500
[2,700, 6,929] · manifest ~800. AA adds **0 bytes** (render-side code).

| scenario | archive bytes | rate |
|---|---:|---:|
| central (correlated-tails judgment: waterfill mid ∧ LBND4 as-measured ∧ q per law) | **93,084** | **0.0620** |
| component-consistent independent band | [70,392, 103,513] | [0.0469, 0.0689] |
| waterfill-fail (base 82,193), LBND4 holds | 115,277 | 0.0768 |
| **worst joint tail: waterfill-fail ∧ B6-slip (LBND2 41,562)** | **128,376** | **0.0855** |

v1's [86K, 99K] was a correlated-tails narrowing that was not derivable from its printed components
(F11); v2 prints the independent band AND the named correlation judgment (waterfill success
correlates with LBND4-on-smoothed success — both ride the same smoothed-source probe P5). The
crossing arithmetic (§0.2) uses only the central and the printed tails. LBND2 byte discrepancy
(41,562 vs 41,526 — 36 B between seats) resolves at P5 (F16ii).

### 5.2 Requirement-E dispositions — inherited from v1 unchanged (FOLD/DEFER/DEAD table).

---

## §6 — THE POSE PLAN (v2: thresholds restructured per F1; twin re-purposed per F7)

- **Track 1 (ARM-PRIMARY): pose ON** — carrier/staging/W1/W1b laws inherited from v1 verbatim.
  **Threshold ladder (v2):**
  - **Success bar (crossing-enabling): converged d_pose ≤ 3e-5** (term ≤ 0.0173). This is the bar
    the crossing arithmetic actually requires (§0.2) — v1's "survives 1.5e-4" oversold by ~8×.
  - **Mid-run milestone: d_pose < 1.5e-4 by tau-end** ⇒ else ALARM + L1-fallback prep (run
    continues; its d_seg value stands).
  - **Hard kill: converged d_pose > 1.5e-4** ⇒ pose-as-designed KILLED for this vehicle → L1
    Jacobian-coefficient $0 gate (fallback carrier). Between 3e-5 and 1.5e-4: pose bytes ship
    (still S-positive per the byte law), pointer-crossing claim for run-1 is dead and stated.
- **Track 2 (twin) ★ RE-PURPOSED (F7 adjudication):** twin = **ARM-PRIMARY with λ_entropy=0, pose
  ON** — a single-dimension comparator. Rationale: λ=15 is the only ep0-engaged, never-swept,
  score-affecting lever with NO independent measurement path (pose has direct d_pose measurement +
  a 3-step fallback ladder; islands have per-class F-rows + the control trajectory). The twin
  yields (i) the λ d_seg-attribution (kill law §1.2) and (ii) the clean Class-D×B bytes recess
  (waterfill BOTH twins' tau-boundary checkpoints — a single-dim byte-axis comparison, replacing
  v1's confounded vs-mod32cap version). **Pose-interference attribution is thereby routed to
  run-2/recess with its confound NAMED:** primary-vs-control d_seg trajectory comparison is
  15-dim-confounded (F10 row stamps the delta); the M5 pose read relies on d_pose direct + F11,
  not on a seg-twin. Twin capped at the tau stage (~450-625 ep), sequential after primary
  stability (memory governor).
- Byte side: q-levels law DIRECTION FIXED (smallest q, §1.2); derive-H live; error-bar honesty
  inherited (0.018 = BORROWED-ancestor, never used in v2's arithmetic — §0.2 uses only 3e-5/1.5e-4
  derived from the score law + measured rate).

---

## §7 — THE MEASUREMENT PLAN (F15: cheapest-decisive-first; ★ = new/changed vs v1)

| # | probe | cost | predicted band (grounding) | kill/proceed |
|---|---|---|---|---|
| P5 | LBND4-on-smoothed source | $0, ~1 min | 18-22 KB | ≥24,149 B ⇒ no gain; also resolves the 36-B LBND2 discrepancy (F16ii) |
| ★ BT | co-predicate backtest | **DONE this session** (§2.2b) | fires ep625 | DONE — trigger ships armed pending T-1 |
| ★ P12 | islands ep0 init probe (EXACT ARM-PRIMARY seed config, 1 epoch) | $0 | seed applies mechanically; init d_seg ≤0.8× control | recalibrates the §3.1 gate constants (F9) |
| ★ P11 | **AA fine-mode memory+throughput gate**: `witness_memory_preflight` at the REAL config + 5-ep governed throughput smoke at the EXACT ARM-PRIMARY config (AA ON — doubles as the F13 stack-throughput smoke) | $0, ~15 min | peak RSS within preflight SAFE; s/ep ≤ 1.5× control | FAIL ⇒ AA → run-2 with measured cost written (F4 derived-exclusion); §8 re-projected on the measured s/ep |
| P1 | comb-REGISTRATION audit | $0, ~1 h | comb separates marks/gaps ≥ GT-sep floor | PASS ⇒ comb IN per §1.2 law; FAIL ⇒ OFF + run-2 A/B removed |
| P2 | FEED-08l fresh-eyes review | $0, reading | verdict survives its limits | FAIL ⇒ lane_carried demotion reverts OPEN |
| P6 | flip-share stability ep650 + FiLM-PR + γ recalibration + per-class meat split (+ composed-share stability, riding the landed R5 memo) | $0, ~2-3 h | island share 55-70% stable | share <35% ⇒ big-3 levers re-rank first |
| P7 | n600 realized-parity row on ep650 (S6-M1) — **early per F15: the decode-integrity gate that de-risks every later row** | ~30-60 min | realized d_seg 0.0034±3e-4; inflate ≤20 min | Δ>+5e-4 ⇒ decode defect — FIX before ANY run |
| P4 | #336 waterfill on mod32cap ep650 | $0, 30-90 min | [52,68] KB @ Δd_seg ≤+5e-5 | kill >+2e-4 ⇒ §5.1 fallback row |
| P3 | full-P HVP ladder (chain-A; K=128 eigen pending) | $0, running | ratio ∈[0.04,0.14] pre-registered | kill bands carry the LINK-0 ~35% instrument gap (F8) |
| P8 | band ROI numerator | ~30-60 min | net ΔS <0 at LBND2 pricing | ≥0 ⇒ band defers to LBND4 re-price; trained-with arm still fires |
| P9 | pose chain validation + ξ q-sweep (smallest-q law) | ~1-2 h | q1024 ≈2-4 KB @ Δterm <0.002 | frame0 not bit-exact ⇒ fix first |
| P10 | exact-eval leg dry-run | ~2-3 h | recomputed-S delta <1e-5 | failure ⇒ no pointer-row plan until it fires |
| T-1/2/3 | req-B injection tests (LIVE trainer path) | $0, ~2 h | fires-when-should + silent-when-shouldn't | failing trigger ⇒ cap-only mode |
| **RUN** | **ARM-PRIMARY launch — via `tools/launch_witness_run.py` (governed; raw-python FORBIDDEN) with `witness_memory_preflight` at the REAL config (F14)** + per-stage byte-closes + stage-boundary AA paired verdicts | ~24 h (pending P11 s/ep) | §9 ladder | ★ **per-stage kills are STACK-level (F7): kill/restore the RUN on d_seg > control at matched epochs — never attributed to a single lever in-flight**; F-alarms advisory |
| TWIN | λ=0 twin to tau boundary (sequential) | ~13-18 h | primary within ±5% of twin | breach ⇒ λ=15 implicated ⇒ twin promotes (named repair) |
| **ROW** | byte-close (B6+B7+B8+**BA** in; waterfill; grammar) → Linux x86_64 contest-CPU `upstream/evaluate.py` | operator GO, ~$1-2 | §9 | THE success definition |

---

## §8 — WALL-CLOCK PLAN (F3/F13: recomputed, arithmetically consistent with the config)

Base s/ep: 107 (control-class) — **PENDING P11's measured stack s/ep** (AA ss=2 ≈ 4× render px;
persistence/chroma/entropy riders; pose carrier is a measured SAVER). All hour figures below scale
by (measured/107).

Event path (consistent with §2.2): CE = 300 (cap-fired; co-predicate conservative — §2.2b) +
TAU fire ≈ ep625 (backtested event; anneal complete ep600) + FIN ∈ [100 (guard exit), 274 (budget
interim)] ⇒ **end ≈ ep725-899 ≈ 21.6-26.7 h**. Cap path: 1000 ep ≈ 29.7 h.
**Event exits are worth ~5-27% vs the cap path** (corrects v1's "~35%", which required the
structurally-impossible pre-726 FIN fire — F3). Twin: ~450-625 ep ≈ 13-18 h sequential. $0 probe
wave ≈ 1.5 days. Total to the exact row: **~4-5 days**, one governed heavy launch + one paid CPU eval.

---

## §9 — PREDICTED S LADDER v2 + DUAL PROBABILITY MODEL (F17) + ASSUMPTIONS

### 9.1 The ladder (component-consistent; Dykstra-grounded)

| rung | central | band | grounding |
|---|---|---|---|
| training-side d_seg @ matched ep | 0.0016 | [0.0010, 0.0028] | §0.3 per-class design table on the LANDED composed-ceiling shares (lane .44/movable .12/big-3 .44); lower edge = every class at its optimistic edge; kill > 0.0030 |
| byte-closed realized d_seg | +0..+1e-4 | — | int8+R prior; **+ the chain-A measured +5.2% int8-deploy surrogate-loss gap ⇒ every rung screened at deploy params** |
| rate | 0.0620 | [0.0469, 0.0689] pass / 0.0768 fail / 0.0855 worst joint | §5.1 component-consistent |
| pose term | 0.0387 (milestone 1.5e-4) | [0.0173 (success bar 3e-5), 0.105 (R1 floor)] | thresholds derived §0.2; NO borrowed-ancestor number used |
| **S [advisory → contest-CPU]** | **≈ 0.26** | **[0.164, 0.47]** | sum of rungs (lower edge = consistent joint tail: 0.10+0.0173+0.0469) |

**Plain statement (means/ends firewall):** v2's central (≈0.26) does NOT cross 0.19110. The
crossing case is now arithmetically TRUE (§0.2: d_seg ≤0.0011 ∧ d_pose ≤3e-5 ∧ rate ≤0.062 →
0.1893) and ENGINEERED (AA in, band trained-with, comb-on-P1, pose bar at 3e-5, anneal completed) —
with the two binding constraints named (§0.3) and instrumented. T_3 (0.15) is NOT in run-1's band.

### 9.2 Load-bearing assumptions (#363) — v1 table inherited, plus:
composed shares transfer control→arm surface (ASSUMED; P6) · AA Δd_seg −1e-4..−4e-4 transfers from
the S5 band (V-A-band, never-fired composed — stage-boundary paired verdicts measure it) · K=128
ratio lands in [0.04, 0.14] (pre-registered; branch §2.3) · AWAITING registry reliance corrected:
**22 rows / 21 unique ids** (not v1's "15").

### 9.3 ★ THE DUAL PROBABILITY MODEL (F17 — both printed; central adjudicated)

**Model A — joint-independent-tail (one-shot lottery):** P(cross) = P(d_seg ≤0.0011) ×
P(d_pose ≤3e-5) × P(rate ≤0.062) ≈ (0.10-0.20) × (0.2-0.4) × (~0.8) ≈ **2-6%**. Assumes no
mid-run steering and independent tails — both false by construction (the run has per-stage EMA
checkpoints, event exits, F-alarms, kills-with-fallbacks).

**Model B — sequential-descent-with-repair:** the byte-close COMPOSES the per-axis best, so only
the two TRAINED legs are one-shot; every other axis carries a named repair:

| axis | repair mechanism (named, per lever) | repairable when |
|---|---|---|
| rate | waterfill/grammar/q-law: choose measured min at byte-close | post-run |
| band | umask / LBND2 giveback / LBND4-vs-smoothed min | post-run (byte-close) |
| AA | render-side ⇒ byte-close-selectable (ship better of aa=on/off verdicts, measured at stage boundaries) | post-run |
| comb | paired-delta kill ⇒ disable + restore-best | in-run |
| pose | F11 watch → W1b shadow law → L1 Jacobian fallback carrier | in-run watch; fallback at byte-close |
| finisher | regression guard → restore-best → DECIDE | in-run |
| islands amplify | bounded closed-loop + witness-alone soft-gate + restore | in-run |
| schedule | caps + F1/F2/F12 alarms | in-run |
| λ=15 harm | λ=0 twin detects ⇒ twin PROMOTES to primary | sequential re-run cost (~13-18 h), not in-run |

**Non-repairables (bound Model B's optimism):** seed-not-taken at ep0 (§3.1 abort, not repair) ·
the along=8 basis/regime choice (guarded at the control value — minimal regime risk) · the two
trained legs (d_seg, d_pose) within a single run (repair = restore-best + twin-promotion + run-2,
i.e. iteration, not in-run rescue). Model B: P(cross by the end of the run-1 CAMPAIGN including
byte-close selection and the twin) ≈ **8-15%**.

**Honest central: between the models, nearer B for rate/band/AA (selection is certain) and nearer A
for the two trained legs.** Headline: the run is a two-wall measurement instrument with a real,
printed, steered crossing tail — not a lottery ticket and not a promised crossing.

### 9.4 Run-2 headroom (named, with dispositions)
In-training comb if P1-fails-late · StepNative β 4→8 finisher fork · mod-dim 2-point · TerminalSolve
at a NEW basin (chain-A template, measured-acceptance) · BoundaryDistance w*=0.2 · per-class hard
λ-gate (#268) · **#149 closed-form facet placement at 874-before-R: DEFER-with-build-spec** (set the
argmax facet at camera resolution before R; partially represented in run-1 by AA's sub-pixel
coverage; build = render-side facet snapping + byte-identity proof — the named disposition PASS-3
demanded) · pose-interference seg-twin (the attribution v2 traded away — confound named §6).

---

## §10 — BUILD LIST = INTEGRATION MAP (v1 inherited; ★ v2 deltas)

| id | build | ~LOC | status/route |
|---|---|---:|---|
| B1-B8, W1/W1b/W2, I-1..I-5, F-rows, T-1..3 | inherited from v1 unchanged | — | v1 §10 table stands |
| ★ BA | AA decode inline into `_INFLATE_PY` (the B-3 build; same class as B6) | small | **LB at byte-close** (F4) |
| ★ B9 | anneal re-anchor law (accelerate-then-fire) | ~30 | NOT LB (600 denominator suffices); optional |
| ★ F3/F4 rows | online-meat + trigger would-fire audit | ~55 | **PROMOTED to LB** (F6) whenever any trigger ships armed |
| ★ P11/P12 | AA gate + init probe | probes | launch-blocking probes (pre-GO) |
| ★ I-6 | equation registrations from the crucible's NEW measured rows: co-predicate backtest law (`stage_exit_verdict_copredicate_backtest_v1`), chain-A subset-idiosyncratic-curvature law (`hessian_negative_curvature_subset_artifact_v1`), int8-deploy +5.2% gap row | 0 (registry) | rides the triality landing |

Supersessions inherited (#183/#124/#285). All commits via serializer with post-edit shas.
Costate code-landing recap unchanged. Triality: this v2 → DAG FEED entry + DSL program file +
equation rows in the P7 landing (crucible-local until then, per req G).

---

## §11 — REQUIREMENT A(ii): THE #342 SOLVE-DON'T-TRAIN INVENTORY (F2 — the table v1 claimed but never produced)

Per block: SOLVED (where/when/conditions) · TRAINED-with-reason · NOT-SOLVABLE-with-proof.

| # | block | disposition | where/when/conditions |
|---|---|---|---|
| 1 | LengthSigma σ_ij junction weights | **SOLVED** (Young's-law fit, closed-form from the frozen scorer's own junction geometry) | offline, pre-launch, DONE (`fitted-20260707`) |
| 2 | LogitAdjust per-class priors | **SOLVED** (Menon log-prior; measured n600 class priors) | offline, DONE (lane −5.13 / movable −4.39) |
| 3 | Analytic lane band geometry | **SOLVED** (openpilot poly + homography fit; rule-118 free generator) | render-side, per-frame, live |
| 4 | Pose H at decode (derive-H) | **CLOSED-FORM** (store-nothing; H_bytes=0) | decode-time, live |
| 5 | Bit-depth allocation | **SOLVED** (KKT waterfill #336, convexity-dominant) | post-training pass, gated P4 |
| 6 | λ ladder (λ_seg/λ_pose/λ_bytes) | **EXACT-ANALYTIC** (score-law derivatives) | costate, live |
| 7 | w_pose schedule | **CLOSED-FORM** score-matched law (W1b) | shadow-first in-run |
| 8 | Eikonal ε | **SELF-DERIVING** CFL law (#318/#320) | in-training, ON |
| 9 | Rewarmup length | **SOLVED** (moment-memory bound n_rw = ⌈1.5/((1−β₂)·steps/ep)⌉ = 20) | launch-time formula |
| 10 | Seed placement (ep0) | **SOLVED-geometric** (paint-then-SDF from GT masks — closed-form init, no training) | ep0 priming |
| 11 | Terminal head/full solve | **NOT-SOLVABLE-NOW, with measured proof:** chain-A — no cheap 1st/2nd-order descent at ep650 (gradstep worse at every η; negcurv winner rejected on rescreen); subset solves overfit (+5.1% #341) | admissible ONLY as full-P in-trainer GPU solve from a PD basin with measured-acceptance (§2.3) — run-2/SOLVE stage |
| 12 | FiLM head least-squares (quadratic chart L77) | same as 11 (chart CONFIRMED ρ 0.85-0.87; subset-solve gap measured) | full-P in-trainer only |
| 13 | Trunk/basis weights | **NOT-SOLVABLE, with proof:** nonconvex composition (coord-INR → through-R render → uint8-STE → frozen-scorer argmax); no closed form exists for the argmax partition objective; chain-A confirms even local quadratic structure is subset-noise at the optimum | TRAINED — the reserved case |
| 14 | τ_end | NOT YET SOLVED (resolution-floor candidate law τ_pix) | recess P-τ; interim 0.2 DPR |
| 15 | Coder/grammar choices | measured SELECTION (enumerate + pick measured min — honestly a search, not a solve) | byte-close, per P5/P4 receipts |

Training is reserved for #11-13 exactly — everything solvable is solved, with its site named.

---

*Round-1 self-review before commit: (1) every F1-F17 finding has a §0.1 disposition realized in a
numbered section, not just claimed; (2) the §0.2 crossing table was computed by hand and each row
re-verified (v1's false tail printed WITH its arithmetic as the negative example); (3) the §2.2b
backtest was RUN, not promised — numbers from the on-disk mod32cap JSON; (4) chain-A folded from its
CURRENT on-disk state with the K=128 branch explicitly carried (eigen pending — not asserted);
(5) new flags/factories verified against trainer argparse + curriculum_dsl this session (AA trio +
AACoverageRender L2179); (6) §8 arithmetic is consistent with §2.2 (FIN cannot start before ep600
anneal-complete + ep625 backtested fire; no pre-726-FIN claim survives anywhere); (7) the pose
success bar is used consistently (§0.2/§6/§9.1 all carry 3e-5; no borrowed 0.018 anywhere in v2's
arithmetic); (8) reqs A-G: A §11+§2.3 · B §2.2/2.2b/T-1..3 · C §3.2(inherited) · D §7 ordering ·
E §5.2(inherited) · F §4.1(+2 LB promotions) · G §10.*

Pointer 0.19110 UNMOVED — this draft is MEANS until the §7 ROW lands.
