# T5 CRUCIBLE — SEAT S2 POSITION — SCHEDULE + CURRICULUM (witness-native derivation)

Seat: S2 (schedule/curriculum face). Written blind (no other position_S*.md read).
Date: 2026-07-07. All numbers `[macOS-MLX/CPU research-signal]` advisory unless noted;
pointer contest-CPU **0.19110 UNMOVED** — everything here is MEANS.

Evidence base honored per the operator correction: **mod32cap
(`experiments/results/levelset_n600_witness_mod32cap_20260706T115554Z/`) is the
COUNCIL-DESIGNED CLEAN BASELINE** (`.omx/research/council_symposium_clean_config_20260705.md`,
T3): eik-0, no seeding/band/prior/birth, verdict-pairs 0 (n600), FRESH-from-scratch — its
trajectory is a PURE read of the schedule/curriculum itself. The exclusions are design, not gaps.

**NEW $0 MEASURED rows produced by this seat (inline probes on the mod32cap log + tac code, this
turn — commands in §2):**

- **M-S2-1 (finisher diagnosis).** Full n600-verdict trajectory (`levelset_train_result.json`
  history, 41 rows): CE 0.7439→0.004571@ep300 (still descending −2.4%/25ep at the fixed boundary);
  τ-stage descends to **best 0.0033662@ep650**, then flat/noise to ep725 (0.0034139); Muon fires
  ep726 → **+27.5% transient** (0.0034139@725 → 0.0043514@750); recovery incomplete: **ep1000 =
  0.0037373 = +11% ABOVE the ep650 best**. The `muon_finisher_switch` log row confirms the control
  ran `muon_warm_start_momentum=False`, `muon_lr_final_frac=1.0` (flat 0.002), AdamW-LR frozen 1e-4.
- **M-S2-2 (anneal truncation).** `loss_terms` rows at ep726 and ep1000 both read
  **hosc_beta=3.177, softmax_temp=0.2157** — the Muon freeze (FEED-fm; `_softmax_temp_for_epoch`
  extraction comment) froze BOTH anneals at 72.6% of their paths: β reached 3.18 of its DERIVED
  4.00 endpoint; τ reached 0.216 of its configured 0.05. The anneal denominators anchor to ep1000
  while the freeze anchors to ep726 — **the sealed schedule's effective endpoints were never the
  derived ones.**
- **M-S2-3 (event-trigger backtest, ep_loss).** Re-implementing the trainer's plateau rule
  (rel-eps 1e-4, W=25 trailing ep_loss, min-stage 250) on the control's per-epoch `loss_terms`:
  CE→tau would have fired at **ep251** (rel slope −1.8e-5) — the FIRST eligible epoch — while the
  n600 d_seg was still descending 2%/25ep, and the trailing-25 rel slope at ep300 is −9.2e-3
  (descent RESUMED after the ep250 self-orient reorientation; `--reorient-every 50`). I.e. on this
  trajectory the ep_loss-only plateau test is nearly vacuous (min-stage is the binding constant)
  AND is confounded by the reorient cycle. The τ-stage plateau fires at **ep559**
  (β=2.676, τ=0.438 there).
- **M-S2-4 (meat / exit).** `tac.witness_control.powerlaw_exit.powerlaw_meat_exit` on the τ-stage
  n600 d_seg (ep300–725): preferred model EXPONENTIAL (ΔAIC −47), **asymptote a=0.003377,
  τ_e=79 ep, remaining meat @+300ep = 5.5e-6 → EXHAUSTED**. The τ stage was done by ~ep600–650;
  ~76–125 epochs ran past saturation before the fixed Muon boundary.
- **M-S2-5 (Muon recovery fit).** Exponential fit of the Muon stage (ep750–1000): asymptote
  **a=0.003236, τ_e=305 ep** — Muon-as-configured is (extrapolated, ADVISORY) net-positive
  *asymptotically* (~−4% below the τ asymptote) but its cold-switch quench + flat LR gives a
  recovery time constant (305 ep) LONGER than the remaining budget (274 ep). **The control's
  finisher failed by TRANSIENT × BUDGET, not by paradigm.**

---

## 1. Position

### 1.0 The frame (derived, not inherited)

The witness trains ONE continuously-annealed energy
`E_τ[φ] = data(softmax_τ) + λ_eik(t)(|∇φ|−1)² + ν|∂argmax| + …` rendered through R. "Stages" are
discretization artifacts of the continuation path (τ(t), β(t), λ_eik(t), η(t), optimizer). The
schedule's physics (paths) is largely DERIVED already (#302 B.1–B.7); what remained PR95 is the
**clock** (fixed 300/726/1000), the **optimizer constants**, and — NEW from this seat — the
**anneal-denominator/freeze mis-composition (M-S2-2)** and the **finisher-budget law (M-S2-5)**.
My position: keep the re-derived stage ORDER, convert every boundary to an event with the fixed
epoch as CAP, make anneal-completion a PRECONDITION of the finisher, fire the two built-unfired
finisher levers, and add a measured EXIT so no epoch runs past exhaustion. On the control this
would have saved ~350 of 1000 epochs (35% wall-clock) at ZERO score cost and plausibly ended
BELOW 0.0033662 instead of at 0.00374 (M-S2-1/4/5).

### 1.1 The derived stage sequence (the level-set energy's own dictates)

**Stage P — PRIMING (ep0):** structured-init (+include-lane) as in the control (kept); island
arms add SeedIslandEased HERE (Allen-Cahn: classes must be born ABOVE critical nucleus BEFORE any
MCF — birth belongs in CE, never after τ fires). FINER bias-init (`--finer-bias-init`, built,
never-fired) is an activation-priming A/B for the vehicle seat, not primary.

**Stage 1 — CE / FORMATION (mirror-descent/NG encoding of the partition).**
Active: basis levers from ep0 (basis-match is PRIOR to capacity), seeds (island arms), w_seg 100.
τ=τ_start, β=β_start, eik per-arm (0 control / ramp-armed treatment), LR cosine top.
**Exit (CE→tau) = event, cap 300:** `--curriculum-event-triggered --curriculum-nucleus-guard
--curriculum-plateau-rel-eps 1e-4 --curriculum-plateau-windows 25 --curriculum-min-stage-epochs 250`
(#315/#292/#302, ALL BUILT) **plus two derived hardenings from M-S2-3:**
(a) **reorient-aware window** — the plateau window must not span a `--reorient-every` boundary
(or W ≥ 2× the reorient period), else the pre-reorient flat shelf misfires the trigger;
(b) **verdict co-predicate (BUILD-SMALL ~40 LOC)** — require the trailing n600-verdict d_seg
slope to ALSO satisfy the plateau (V=4 verdicts, rel-eps ~5e-3/25ep), because ep_loss-only fired
at first-eligible-epoch while d_seg still paid 2%/25ep. Until (b) lands, event mode on a
mod32cap-like trajectory degenerates to "fixed at min-stage+1" — admissible but mislabeled.
Nucleus guard semantics (interface-critical): in a NO-BIRTH control the guard can never satisfy
(lane/movable part_frac=0) → the cap fires = fixed mode in disguise. Guard ON only in island arms.

**Stage 2 — τ/SHARPENING (Γ-dequantization + MCF; the partition anneal).**
Levels (paths, each a DECISION per §14 axis 2):
- τ: geometric shape — **`--tau-anneal-shape geometric` EXISTS** (THETA* MUST-1; verified
  trainer:7369) — the Γ-optimal constant-Fisher-Rao-velocity law (facet-4, CV≈0.39 $0-confirmed).
  The control's cosine was the unexamined default.
- β_hosc: 1.0→4.00 with **the 4.00 reached AT the finisher fire** (M-S2-2 fix below). Geometric
  shape = BUILD (~10 LOC; `--hosc-beta-anneal` choices are linear|cosine only, verified
  trainer:7632); linear is the fallback with the derivation gap on record.
- **THE M-S2-2 FIX (flag-level, $0, available NOW): set `--anneal-epochs` = the Muon CAP** so
  both anneals COMPLETE exactly at the finisher boundary instead of being silently truncated at
  72.6% of path. Robust form once the Muon event trigger lands: **anneal-complete is a
  PRECONDITION of the finisher fire** (fire ⟺ plateau ∧ nucleus ∧ β=β_end ∧ τ=τ_end), because an
  EARLY event fire otherwise re-creates the truncation (M-S2-3: firing at ep559 would freeze
  β=2.68 — softer than the control's own defect).
- λ_eik: control-inherit 0 in the control replicate; island/eik arms step 0.05→0.10 at the FIRED
  boundary (π_int ≳ 1; `_scheduled_eikonal_weight` already re-anchors; `--curriculum-reanchor-levers`
  BUILT, verified trainer:7696) + adaptive-ε (#318/#320 flags exist) + ca-band 0.5 (§15.2
  Froese–Oberman filtered scheme) in the eikonal arm only.
- ν (length) 0.001 held small (MCF-erosion driver, DERIVED keep); WD held ≠0 (§15.5 selection).
- Island-arm followers staggered + boundary-relative: seed-anneal 275, persistence-warmup 275
  (one homotopy parameter per epoch neighborhood; collision harm MEASURED 3.4×), band engage
  boundary+50 (all re-anchored by the built flag).
**Exit (tau→finisher) = event, cap 726 — THE ONE SCHEDULE BUILD I RANK FIRST (~80 LOC; Muon is
verifiably not event-fireable today):** fire ⟺ [τ-stage plateau: ep_loss ∧ verdict co-predicate]
∧ [nucleus-complete (island arms)] ∧ [anneal-complete]. Measured basis: the control saturated by
~ep600–650 (M-S2-4) and then burned 76–125 epochs; the backtest trigger (ep559 on ep_loss alone)
shows the co-predicates are what make the fire epoch honest.

**Stage 3 — FINISHER (Muon, κ-buster on the FORMED partition).**
Keep Muon — conditionally. The −32% fork A/B (2026-06-22) is the keep anchor, but it was measured
at a higher residual on a different arm (borrowed-number risk, flagged), and the control's own
finisher was NET-NEGATIVE AT BUDGET (M-S2-1/5). Keeping Muon is JUSTIFIED **only with the built
levers ON**: `--muon-warm-start-momentum` (kills the +27.5% quench; #269) +
`--muon-lr-final-frac 0.1` (river-valley: Newton-Schulz fixes magnitude, flat LR cannot
self-reduce; #270) + `--muon-lr 0.002 --muon-momentum 0.95 --muon-ns-steps 5` (measured /
literature-settled). EMA π-group: `--ema-decay-finisher 0.9995` A/B (built; 0.997 = 4.4-ep window
averages 1.6% of a 274-ep finisher; house default stays until a byte-closed A/B).
**Finisher exits (both new):**
(a) **regression guard (BUILD-SMALL ~30 LOC or costate-advisory):** pre-register — the finisher
must produce a NEW BEST within 100 epochs of fire, else EXIT-AND-RESTORE-BEST (per-stage EMA
checkpoints make restore free). On the control this alone caps the damage at ~100 wasted epochs
instead of 274 + a worse final.
(b) **meat exit:** per-class power-law/exponential remaining-meat (`powerlaw_meat_exit` exists as
costate SENSE; `ExitEvent(criterion="powerlaw_meat")` exists in the DSL but is a
TrainerSupportGap — the trainer's only run-end lever is the closed-loop early-stop). Wire the
meat exit into the existing clean early-stop arming path (reuse, not new machinery). Terminal
time becomes free (T is a decision variable; fixed `--epochs` forfeits it — keep 1000 as CAP).

**Stage 4 — TERMINAL SOLVE (§16.1; `TerminalSolve` DSL object exists, trainer build absent).**
The exact (not asymptotic) "no meat left" mechanism: in-basin → GN/CG-solve → verify at full n600
through the real verdict. GATED on RECESS-2 below; the control provides the cheapest possible bar
(iteration from ep650 actually went BACKWARDS, so "beat 0.0033662" is the whole test).

**Repetition (§14 axis 3):** admit cycles ONLY as event-triggered restore-and-continue — on
regression-guard trip: restore best, then either resume the τ-continuation under AdamW at frozen
levels (the #270 restart pattern) or terminal-solve — and ONLY when the per-class meat detector
says meat remains. No blind re-entry: M-S2-4 says the control's τ stage was genuinely exhausted;
a cycle there would have been theater.

### 1.2 The concrete Curriculum object (verified DSL surface) + run-3 flag diff

Verified class surface (`tac.witness_dsl.curriculum_dsl`): `Curriculum(stages, temp, regularizers,
hosc, tau, transition, handoff∈{fixed,event}, level_paths, operational, terminal_solve)` ·
`Stage/StageSpec` · `ExitEvent(criterion∈{plateau, nucleus_guarded_plateau, marginal_dseg_floor,
lever_exhaustion, powerlaw_meat})` (last three = TrainerSupportGap) · `LevelPath(quantity,
segments, …)` with duplicate-emitter protection vs `Curriculum.temp/hosc` · `TerminalSolve`
(compiles to no argv; surfaces as gap) · `sealed_205_curriculum(cfg, handoff=)` ·
`verify_schedule_consistency` · `CURRICULUM_OWNED_FLAGS` (schedule flags MUST be sourced from the
object — no hand-set epochs).

```python
curr = sealed_205_curriculum(cfg, handoff="event")   # emits --curriculum-event-triggered
                                                     #       --curriculum-nucleus-guard
# + LevelPath("softmax_temp", geometric)  -> --tau-anneal-shape geometric  (EXISTS)
# + LevelPath("hosc_beta", geometric)     -> BUILD (~10 LOC) else linear-with-gap-on-record
# + anneal_epochs = MUON_CAP (726)        -> the M-S2-2 truncation fix (EXISTS: --anneal-epochs)
# + Transition(rewarmup 20ep, floor 0.1, cosine, reset_moments=True)   # 20ep=1500 steps ≥ 1000
# + ExitEvent("nucleus_guarded_plateau", windows=25, min_stage=250, rel_eps=1e-4)   # CE->tau
# + ExitEvent("powerlaw_meat", floor=1e-4/eq-band, min_points>=8)  # gap today -> wire to early-stop
# + terminal_solve=TerminalSolve(...)     # gap; gated on RECESS-2
```

Flag-level diff vs the sealed control (everything below EXISTS unless marked BUILD):

| change | flag(s) | basis |
|---|---|---|
| event hand-off + nucleus guard | `--curriculum-event-triggered --curriculum-nucleus-guard --curriculum-plateau-windows 25 --curriculum-min-stage-epochs 250` | #315 built; C1 recalibration; M-S2-3 caveat on record |
| boundary-relative followers | `--curriculum-reanchor-levers` | #302 M1, built |
| readiness telemetry ALWAYS ON | `--handoff-readiness-telemetry` | score-neutral read-only ⇒ default-ON per "off is orphaned signal" |
| anneal-complete at the finisher | `--anneal-epochs 726` (= Muon cap) | M-S2-2 (measured truncation) |
| τ shape | `--tau-anneal-shape geometric` | facet-4 Γ/Fisher-Rao (DERIVED) |
| Muon warm-start + LR anneal | `--muon-warm-start-momentum --muon-lr-final-frac 0.1` | M-S2-1/5; #269/#270 built-unfired |
| finisher EMA A/B | `--ema-decay-finisher 0.9995` | π_ema violation (B.6); A/B owed |
| rewarmup window | `--stage-transition-rewarmup-epochs 20` (control ran 8 = 600 steps < the 1000-step β₂ bound) | derived window bound |
| β₂ secondary A/B | `--adam-beta2 0.9999` | help-text small-n law, un-A/B'd |
| BUILD rank 1: Muon event trigger (~80 LOC, cap 726) | — | M-S2-4 (76–125 wasted ep) + anneal-complete precondition |
| BUILD rank 2: verdict co-predicate in the plateau test (~40 LOC) | — | M-S2-3 (ep_loss-only fires at first eligible epoch) |
| BUILD rank 3: geometric hosc-β (~10 LOC) | — | τ=ε=ħ equal-epochs-per-octave |
| BUILD rank 4: finisher regression guard (~30 LOC) | — | M-S2-1 (274-ep worse-than-entry finisher) |
| BUILD rank 5: powerlaw_meat run-end wire into the clean early-stop | — | M-S2-4; §15.4; frees terminal time |

Per-stage lever/level table (control-arm-compatible; island/eik/pose engagement is per-arm and
always AT a boundary, never mid-τ-descent — IB rule B.3):

| stage | active levers | τ | β | λ_eik | LR | optimizer | exit |
|---|---|---|---|---|---|---|---|
| P/CE | basis@ep0, seeds (island arms) | τ_start hold→geometric | 1.0→ | 0 (arm: armed) | cosine top | AdamW | event(plateau∧nucleus∧min-250), cap 300 |
| τ | + band@fire+50, persistence/seed-anneal fire−25 (arms) | geometric → τ_end AT finisher | → 4.00 AT finisher | step 0.05→0.10 @fire (arms) | cosine + 20ep rewarmup @boundary | AdamW (moments reset) | event(plateau∧anneal-complete), cap 726 |
| finisher | frozen levels | frozen τ_end | frozen 4.00 | frozen | muon-lr 0.002 →×0.1 cosine | Muon warm-started | new-best-within-100ep guard ∧ meat exit, cap 1000 |
| solve (gated) | — | frozen | frozen | — | — | GN/CG | full-n600 verify beats entry best |

### 1.3 The mod32cap finisher diagnosis (the asked deliverable, compact)

Why did Muon (ep726+) not beat ep650? Decomposition, each part labeled:
1. **MEASURED — cold-switch quench:** +27.5% at the switch with `warm_start=False` (the built
   #269 lever off; §21 corrected the equivalent vs-best number to ≈+29%).
2. **MEASURED — flat finisher LR:** `muon_lr_final_frac=1.0`; recovery τ_e=305 ep > 274-ep
   remaining budget (M-S2-5) — the step never settles.
3. **MEASURED — under-sharpened entry:** β frozen at 3.177 (not the derived 4.00), τ at 0.216
   (not the configured 0.05) — the finisher polished a NOT-fully-dequantized energy (M-S2-2).
4. **MEASURED — late entry:** τ-stage exhausted by ~ep600–650 (asymptote 0.003377, meat 5.5e-6);
   entry at 726 wasted 76–125 ep of finisher budget (M-S2-4).
5. **INFERRED (extrapolated fit, advisory):** Muon's own asymptote 0.003236 < the τ asymptote —
   the κ-buster premise is not falsified by this run; the SCHEDULE around it was the failure.
   The GN-spectrum probe (RECESS-1) is the direct test of the premise.
The deployed-checkpoint authority (best-EMA @ep650) bounded the damage — but 35% of the run's
wall-clock was spent making the result worse. That is the exact inversion of "no meat left."

---

## 2. Derivations + assumption tags (#363)

Probe commands (deterministic, $0, foreground): parsed
`.omx/tmp/levelset_mod32cap_20260706T115614Z.log` (per-epoch `loss_terms` rows + 41 `verdict`
rows + the `muon_finisher_switch` / `curriculum_transition` rows) with a ~60-line python script
re-implementing the trainer's trailing-window relative-LS-slope plateau rule, plus
`tac.witness_control.powerlaw_exit.{powerlaw_meat_exit, fit_tail_models}`.

- Stage order CE→sharpen→finisher forced by the math — VERIFIED-VIA-ANCHOR (#302 A row 2 /
  #284: Muon cannot nucleate, measured facet-4 §2.1).
- Fixed 300/726/1000 = cross-run epoch transfer (CARGO) — VERIFIED-VIA-ANCHOR (#302 row 3; cert
  A7 verbatim "PR95 stage-8 placement") + M-S2-1 (CE still descending at 300).
- M-S2-1..5 — VERIFIED-VIA-ANCHOR (this seat's probes on the run log + train_result history;
  fits are deterministic per powerlaw_exit's own contract). The Muon-asymptote comparison
  (M-S2-5 conclusion) is INFERRED (11-point extrapolation) — routed to RECESS-4, not load-bearing.
- Anneal truncation mechanism (freeze at muon-start; denominator = --anneal-epochs) —
  VERIFIED-VIA-SOURCE (`train_levelset_witness_realized_through_R_mlx.py:2318-2400`
  `_hosc_beta_for_epoch` / `_softmax_temp_for_epoch` + the `loss_terms` ep726/ep1000 values).
- Event/nucleus/re-anchor/closed-loop/warm-start/final-frac/finisher-EMA/tau-geometric flags all
  EXIST — VERIFIED-VIA-SOURCE (trainer argparse lines 7364-8204, grep-verified this turn;
  never-invent-flags honored). Muon event trigger does NOT exist — VERIFIED-VIA-SOURCE
  (`_evt_reanchor_epoch` docstring: "stays a fixed cap until the Muon-event-trigger BUILD").
- Geometric τ = Γ-optimal constant Fisher-Rao velocity — VERIFIED-VIA-ANCHOR (facet-4 memo via
  #302 B.1; CV≈0.39 $0-confirmed there). Geometric β transfer — INFERRED (same law, different
  scalar; the code path is a BUILD).
- powerlaw_meat_exit is costate-SENSE only (not a trainer exit) — VERIFIED-VIA-SOURCE
  (`src/tac/witness_control/powerlaw_exit.py` module docstring "NOT wired here").
- σ-noise never built on the witness; uint8-STE live & un-disableable — VERIFIED-VIA-ANCHOR
  (dossier §22(3) source-inspection).
- C1a stacking net-negative (`supersedes_c1a=True`) — VERIFIED-VIA-ANCHOR (§22(2) caveat chain).
- "−32% Muon vs AdamW" — VERIFIED-VIA-ANCHOR (fork A/B 2026-06-22) but ASSUMED-transferable to
  the current residual regime (borrowed-number risk explicitly flagged; RECESS-1/4 measure it).
- Plateau-constants fragility / reorient confound — VERIFIED-VIA-ANCHOR (M-S2-3) for THIS
  trajectory; the reorient causal mechanism is INFERRED (temporal coincidence at ep250; cheap to
  confirm from reorient rows — noted, not load-bearing).
- p=0.018 pose term in any break-even arithmetic — ASSUMED (borrowed ancestor; pose OPEN on the
  witness). This seat takes no pose-value position; only the ENGAGEMENT rule (§5).

## 3. PR95 cargo-cult audit (my face — every inherited element)

Tags: **D** = DERIVED-FROM-WITNESS-MATH (kept, with derivation) · **J** = JUSTIFIED-KEPT
(measured evidence) · **X** = DROP/REPLACE (with replacement).

| # | element | tag | derivation / evidence / replacement |
|---|---|---|---|
| 1 | stage ORDER CE→sharpen→finisher | **D** | #284: CE=mirror-descent/NG; τ=Maslov/Γ-dequantization (MCF); finisher outside the τ-continuum cannot nucleate (measured) ⇒ nucleate→sharpen→polish is forced. Inherited, then independently re-derived. |
| 2 | CE stage | **D** | partition FORMATION under NG geometry; also the only stage where island birth is admissible (Allen-Cahn critical nucleus). |
| 3 | fixed boundaries 300/726/1000 | **X** | cross-run transfers of another trajectory's knees (cert A7 verbatim "PR95 stage-8 placement"). REPLACE: event triggers with the fixed epochs as CAPS (#315 built for CE→tau; Muon trigger = BUILD rank 1) + anneal-complete precondition (M-S2-2/3). |
| 4 | stage LENGTHS (epoch counts) | **X** | PR95's 29,650-ep ladder; PR95Author: "the numbers were never meant to transfer." REPLACE: min-stage floors + event exits + caps; run length = CAP with meat exit (M-S2-4: 350 of 1000 control epochs were past-exhaustion). |
| 5 | tau_softplus loss form (T=0.3) | **J** | cert A5 proven value; dominant temperature lever is elsewhere; low-priority A/B (TauFrozen lever exists for it). |
| 6 | τ anneal endpoints 1.0→0.05 | **J-PROVISIONAL** | τ_end's dash-role is UNMEASURED (§21 GT-H overturn); roles that remain: margin sharpening + pixel-pitch floor (π_τ→1). Fixed-τ control arm (§19-cons-3) is the discriminator — operator-GO arm, not run-3 primary. |
| 7 | τ anneal shape cosine | **X** | un-derived default. REPLACE: `--tau-anneal-shape geometric` (EXISTS) — constant Fisher-Rao velocity / adiabatic dwell (facet-4, $0-confirmed CV≈0.39). |
| 8 | anneal denominator = run length | **X** | M-S2-2: freeze at Muon truncated β to 3.177/4.00 and τ to 0.216/0.05 — silent un-derived effective endpoints. REPLACE: `--anneal-epochs` = finisher CAP now; anneal-complete as finisher-fire precondition once event-Muon lands. |
| 9 | hosc β anneal 1→4 (annealed) | **D** | fixed-β=4 DIVERGES (measured); annealed-hosc stable; β_end=4.00 AT the finisher = the derived freeze value. |
| 10 | β anneal shape linear | **X** | β is the activation's own dequantization scalar; τ=ε=ħ ⇒ GEOMETRIC (equal epochs per octave). BUILD ~10 LOC (choices are linear|cosine — verified); linear fallback with gap on record. |
| 11 | l7 stage | **X** | measured DEFECT (L∞ sharpening inside a viscosity flow); already demoted to 1001 (parked). Keep parked; guard already landed (l7 never converge-fires when demoted). |
| 12 | PR95 smooth stage | **X** | measured RAISES d_seg (CLAUDE.md lever-4). Stays dropped. |
| 13 | PR95 QAT stage (L14 s4) | **X** | REPLACE: uint8-STE is LIVE and un-disableable in every R path (§22(3) source-audit) — quantization-awareness is structural, not a stage. |
| 14 | PR95 σ-noise schedule 0.2→0.1 (L17) | **X** | NEVER BUILT on the witness (no noise flag exists — §22(3)); not an orphaned default-off but a design question for the counted-weights arm. Do not cargo it back. |
| 15 | PR95 C1a λ 0.01→0.02 stages (L16) | **X** | C1a stacking measured NET-NEGATIVE vs the weight-entropy lever (`supersedes_c1a=True`, §22(2)). |
| 16 | Muon presence (finisher) | **J-conditional** | −32% fork A/B keep-anchor (2026-06-22) BUT control finisher net-negative at budget (M-S2-1/5). Keep ONLY with warm-start + LR-anneal ON + event entry + regression guard; premise test = RECESS-1. |
| 17 | Muon placement last | **D** | κ-buster on the formed partition; measurably cannot nucleate; κ grows as the interface sharpens (B.5). |
| 18 | Muon start 726 | **X** | REPLACE: event (plateau ∧ nucleus ∧ anneal-complete), cap 726. M-S2-4: control saturated ~ep600–650. |
| 19 | Muon lr 0.002 flat | **J/X** | 0.002 MEASURED (A8). FLAT is the cargo: REPLACE with `--muon-lr-final-frac 0.1` (built; river-valley law; M-S2-5 τ_e=305ep is the receipt). |
| 20 | Muon momentum 0.95 / ns 5 | **J** | literature-settled (NVIDIA 2606.00371: k3-vs-k5 ≤0.15pp; polar accuracy not the lever). |
| 21 | Muon cold momentum start | **X** | REPLACE: `--muon-warm-start-momentum` (built). Control's +27.5% quench is the measured cost (M-S2-1). |
| 22 | LR 1e-3→1e-4 cosine | **J-PROVISIONAL (PR95-ECHO acknowledged)** | no witness-native derivation of the endpoints; do NOT churn blind — the missing input is a measured per-stage sharpness trace = RECESS-1 (HVP-Lanczos). §15.6: LR is the true temperature; spend escape budget before the freeze; never re-raise LR in the selection regime without re-raising τ. |
| 23 | weight decay 1e-4 | **J** | role now DERIVED (§15.5 DFIZ vanishing-discount selection/uniqueness ⇒ keep WD finite LATE); the VALUE stays unexamined (honest UNEX, low priority). |
| 24 | adam-β₂ 0.999 | **J + A/B** | MLX default; the trainer's own help cites the derived small-n law (~0.9999999). Secondary A/B 0.9999 only — never primary-confounding. |
| 25 | EMA 0.997 everywhere | **X (finisher) / J (default)** | Quantizr π-group violation: 333-step=4.4-ep window averages 1.6% of a 274-ep finisher; measured 78× early shadow lag. REPLACE in finisher: `--ema-decay-finisher 0.9995` A/B (built). House 0.997 stays deployed-authority until a byte-closed A/B. |
| 26 | rewarmup 8 ep (control) | **X** | 8ep=600 steps < 1/(1−β₂)=1000 AdamW moment memory — UNDER the derived bound. REPLACE: 20 ep (=1500 steps, bound-satisfied). Shape cosine/floor 0.1 ASSUMED — keep, don't churn (#302 row 17). |
| 27 | reset-moments at boundaries | **J** | measured stale-moment root cause (FEED-ft#3). |
| 28 | spike-guard rollback default | **J** | confound-hunt L1/L2 landings; legacy median-freeze = the measured ep103-114 deadlock class (#397/#398). Control ran rollback, spike_skipped_rate 0.00. |
| 29 | eval/ckpt cadence 25 | **J** | post-hoc derived: control lag 1 eval ≪ ~100-ep erosion timescale (4× margin); also > the 4.4-ep EMA window so verdicts read a settled shadow. |
| 30 | seed-anneal / persistence-warmup @300 (island arms) | **X** | 3-way ep300 collision (measured harm 3.4× on the band's sister collision). REPLACE: 275/275 stagger + `--curriculum-reanchor-levers` (built). |
| 31 | eikonal 0 (control) | **J (per-arm)** | deliberate T3 clean-config design (scorer reads the zero level set; Ballé dissent = the eik-0 vs 0.05→0.10 same-seed A/B stands as the arbiter). Treatment arms re-engage the DERIVED ramp + adaptive-ε + ca-band. |
| 32 | epochs 1000 | **X** | budget never π_train-checked; REPLACE: cap + meat exit (free terminal time). |
| 33 | w_seg 100 / w_pose 0 | **D (control)** | 100 = the score's own coefficient; w_pose 0 = deliberate seg-only attribution arm. Pose engagement rule in §5. |
| 34 | reorient-every 50 (schedule interaction) | **X-harden** | NEW (M-S2-3): the plateau window spanning a reorient boundary misfires. Harden the trigger to reorient-aware windows (BUILD rank 2 includes it). |

## 4. RECESS measurement proposals

1. **HVP-Lanczos GN/Fisher spectrum on the preserved mod32cap stage checkpoints** (ep299 CE /
   ep650 best / ep726 MuonStart / ep1000 final). $0, no training; chunked pair-subsampled HVPs
   (each ≈2 forwards), foreground, <8 GiB if chunked, est 15–60 min → RECESS not inline.
   Pre-registered bands: κ(ep650) > κ(ep299) by ≥3× (interface sharpening grows boundary-Hessian
   anisotropy; the basin κ≈19 anchor); top spectrum at ep650 positive with a gap (quadratic
   basin); κ(ep1000) ≈ κ(ep650) (Muon didn't change the basin). KILL: if κ does NOT grow CE→τ,
   the Muon-κ-buster premise is falsified ON THIS VEHICLE → Muon demoted to A/B arm and the
   finisher defaults to terminal-solve-or-stop. PROCEED: basin-quadratic ⇒ fire RECESS-2. This
   probe is simultaneously the #302 B.7 missing sharpness trace (unblocks an LR derivation, row
   22) and the D-3/4/5 costate 2nd-order SENSE first measurement — one probe, three consumers.
2. **GN/CG terminal-solve from ep650** (§16.1's natural A/B). Bounded compute (hundreds of CG
   iters on subsampled pairs + ONE full-n600 verify through the real verdict), governor-gated.
   Pre-registered: beat 0.0033662 (the realized alternative DID NOT — ep1000 = 0.0037373, so any
   improvement wins the A/B). KILL: CG-solve ≤ no improvement at full-n600 verify → TerminalSolve
   deferred, quadratic-finisher claim stays PREDICTION. PROCEED: adopt as run-terminal stage.
3. **Per-class meat-exit calibration** ($0, minutes): fit per-class tails on the available
   per-class traces (annulus_live lane flip-frac; probe pairs JSONL) — pre-registered: lane tail
   power-law α<1 while the TOTAL is exponential (M-S2-4 measured the total; weak-KAM predicts the
   split). If confirmed: the run-end meat exit MUST be per-class (a total-only detector exits
   while the binding class still pays) and the verdict row grows per-class d_seg telemetry
   (BUILD-SMALL, score-neutral, default-ON). KILL of the per-class requirement: lane tail also
   exponential. |
4. **Finisher-schedule A/B in anger** (bounded n600, ~150 ep ≈ 4.5 h at the measured ~107 s/ep,
   operator-GO): resume ep650 BEST → Muon with warm-start + final-frac 0.1 (the #270 restart
   pattern). Pre-registered: switch transient ≤ noise band (vs +27.5% control) AND new best <
   0.0033662 within 100 ep. KILL: still spikes >10% → warm-start mechanism falsified as the
   transient's cause; if no new best in 100 ep → regression guard fires by design and Muon is
   demoted for this residual regime (composes with RECESS-1's verdict).
5. **Fixed-τ control arm** (τ frozen 0.8, identical seed/config; §19-cons-3, survives §21 as the
   primary τ-dash discriminator): full-run cost, operator-GO, NOT run-3-blocking. Read: d_seg + a
   GT-conditioned dash index at matched epochs vs the annealed run's per-stage checkpoints.

## 5. Interfaces

- **From the vehicle/basis face (S1):** reorient cadence (`--reorient-every`) feeds my trigger
  windows (M-S2-3 hardening); band/comb engage epochs are boundary-relative (+50 after the fired
  τ boundary — the re-anchor flag handles it); the dash-comb may NOT be scheduled to engage until
  the §21 GT-conditioned comb-registration audit passes; FEED-08l (freq_along ladder
  INDETERMINATE; comb favored) means my schedule carries NO freq_along-dependent timing.
- **To the islands face:** nucleus guard ON only in birth arms (in a no-birth arm it silently
  degrades event mode to cap-fixed — by construction, lane/movable part_frac=0 never satisfies);
  island birth belongs in CE (Allen-Cahn); stagger 275/275; the eikonal ramp is the born-island
  MCF-protection lever (π_int ≳ 1) — engage it in island arms even though the control is eik-0.
- **To the costate face:** I provide the SENSE surfaces (handoff_readiness default-ON; per-class
  verdict telemetry BUILD-SMALL; loss_terms already on) and pre-registered DECIDE thresholds
  (plateau constants, meat floors, regression-guard window); the GN/Fisher 2nd-order state
  (RECESS-1) is the Muon-entry + TerminalSolve-entry mechanistic signal; ACT stays
  advisory-autonomous except the already-built bounded closed-loop actions + the event flags.
- **To the pose face:** engagement rule only (no pose-value claim): pose joins at a STAGE
  BOUNDARY or ep0, never mid-τ-descent (IB rule B.3); if pose-ON is the pointer-run verdict,
  prefer ep0 (no mid-run objective change) with the transition machinery (rewarmup +
  reset-moments) at any staged engagement.
- **To the rate face:** `WeightEntropyPenaltyMLX` (never-fired) engages from ep0 if included
  (objective stationarity); its λ A/B must not share a run with the schedule A/Bs (attribution).
- **From the DSL face:** I need trainer support for `ExitEvent(powerlaw_meat / marginal_dseg_floor)`
  (today TrainerSupportGap), the Muon event trigger, geometric hosc-β, and the plateau verdict
  co-predicate — ranked in §1.2; everything else I use is verified-existing.

**Wall-clock corollary (score-first honored):** every proposal above is score-side; the wall-clock
win (~35% of the control run was past-exhaustion or regression) falls out of the exits for free —
the §14/§18 objective realized structurally, no speed-for-score trade anywhere.

*Round-1 self-review before commit: attacked (a) "Muon net-negative" → corrected to
net-negative-AT-BUDGET with the asymptote fit (M-S2-5) and routed the extrapolation to RECESS-4;
(b) "event trigger ready" → weakened per M-S2-3 (ep_loss-only is nearly vacuous on this
trajectory; co-predicate BUILD named); (c) checked every recommended flag against the live
argparse (never-invent-flags); (d) verified the anneal-freeze mechanism in source before calling
M-S2-2 a defect; (e) confirmed the clean-baseline framing per the operator correction — no
exclusion treated as a gap.*
