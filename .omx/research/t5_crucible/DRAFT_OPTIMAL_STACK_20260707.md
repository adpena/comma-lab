---
doc_type: t5_crucible_p2_positive_synthesis_draft
role: CHIEF DESIGNER (P2), operator-convened T5 crucible
date: 2026-07-07
inputs: convening record §1-§4 · ORCHESTRATION_LEDGER (requirements A-G + ordering guard) ·
  CONTEXT_COMPENDIUM (all 20 stores) · positions S1-S6 · dossier §1-§24
chain_A_status: pursuit_chainA_spectrum_solve_20260707.md ABSENT at draft time — INBOUND;
  §2.4 carries the explicit branch on its outcome (TerminalSolve GO vs warm-Muon-only).
epistemic_contract: every knob carries a CONTROL LAW class {(a) CONSTANT · (b) RAMP/ANNEAL with
  completion guarantee · (c) SELF-DERIVING · (d) EVENT-CONDITIONED (backtest+injection+cap per
  req B) · (e) FRACTIONAL/PARTIAL} and a tag {VERIFIED-VIA-SOURCE · VERIFIED-VIA-ANCHOR ·
  DERIVED · DEFAULT-PENDING-RECESS}. Proposing-optimal-with-tags; nothing unmeasured asserted
  as measured. Pointer contest-CPU 0.19110 UNMOVED — this whole file is MEANS.
axis: all numbers [macOS-CPU/MLX advisory] unless tagged; archive bytes are exact (zip stat).
---

# T5 CRUCIBLE — P2 DRAFT: THE OPTIMAL FULL STACK (launchable as written)

## §0 — HEADLINE + THE SIX CROSS-SEAT CONFLICT RESOLUTIONS (decided, not deferred)

**The stack in one paragraph.** One pointer-aimed run (ARM-PRIMARY) over the
`Mod32SegOnlyControlBase` composition base: control basis at **along=8** (ordering guard),
**islands-first curriculum** (seed-pair + eased + logit-adjust + soft-gated amplify + persistence
+ LengthSigma + eikonal-ramp — the S5 FIRE core under per-class LADDER laws), **analytic lane
band trained-with** (LBND4 cost booked, not "near-zero"), **pose ON two-track** (store-nothing ξ
carrier, w_pose staged law, 1.5e-4 kill), **event-exit schedule with caps** (anneal-complete as
finisher precondition — the M2 fix; warm-started LR-annealed Muon; curvature-aware fire), **rate
compress-half** (WeightEntropyPenaltyMLX λ=15 in-training + #336 waterfill post-pass + grammar
rev-2k), all compiled through a DSL `WitnessProgram`, sensed/decided by the costate controller
with the 12 requirement-F telemetry rows, terminating in byte-close (G1 fix + LBND4 inline) →
`upstream/evaluate.py` contest-CPU on Linux x86_64.

**Conflict resolutions:**

1. **Band byte-cost (S4 24–31 KB vs S6 41.5 KB): BOTH are real — different coders.** LBND2 (the
   only decode inlined today) = 41,562 B measured (S6). LBND4 (varint/rice, decode-reencode
   bit-identical, decode NOT inlined) = 30,892 B measured (S4/FEED-08h). **Decision: the band arm
   ships at LBND4; build B6 (G2 LBND4 inflate inline) is LAUNCH-BLOCKING for the byte-close (not
   for training start); run the $0 LBND4-on-smoothed probe (P5, predicted 18–22 KB) first and take
   the min measured.** Fallback if B6 slips at byte-close time: LBND2 at +0.02767 with the −0.0071
   delta booked as a known giveback.
2. **Budget (110–117 KB > ~105 KB headroom): the compress-half is IN — resolved per Class-D
   lever.** `WeightEntropyPenaltyMLX` **IN** (λ=15 constant from ep0; the only measured-mechanism
   mover of a base stream that is AT its order-0 entropy floor). Flat-minima **#242 OUT**
   (DEFER: designed-unbuilt + would stack two unmeasured Class-D levers — attribution). Variable-
   grid **QAT-at-target-depth OUT** for run-1 (uint8-STE already live; post-hoc waterfill covers
   bit-depth; the Class-D×B interaction RECESS decides run-2). **#336 waterfill post-pass IN**,
   gated on its own $0 probe (predicted [52,68] KB @ Δd_seg ≤ +5e-5). Budget lands ~86–99 KB
   central (§5) — under the 105 KB sub-0.15 headroom line.
3. **along<8 ordering guard: honored structurally.** ARM-PRIMARY carries `--freq-along 8`
   (the T3-designed control point). `DirectionalBasisRebalance(32,"lane_offloaded")` (along=6) is
   the R2 A/B arm and may become primary ONLY after (a) the comb-registration audit passes AND
   (b) the band survives through-R at n600 (S6 M2). The along-tangent axis is the measured starved
   dimension (3.2×) — never re-starved on an ungated premise.
4. **Pose: ON, two-track, staged at the tau boundary, kill 1.5e-4** (S6 accepted in full). The
   seg-only twin arm supplies attribution; w_pose carries a staged + score-matched law (§6); the
   L1 Jacobian-coefficient fallback activates on kill.
5. **Muon finisher: event-fire = quadratic_basin ∧ anneal-complete ∧ plateau-with-verdict-
   co-predicate; cap 726; warm-start + lr-final-frac 0.1 always-on; regression guard.** Chain-A
   branch: TerminalSolve GO ⇒ the SOLVE stage is enabled after the finisher (or replaces its
   tail); NO-GO/absent ⇒ warm-Muon only. S3's measured ep650 indefiniteness means the basin
   predicate may never fire — the cap + warm-start + guard is the fail-safe path (req B).
6. **Islands: the S5 FIRE core composed per the treatment-arm memo + requirement-C LADDER
   per-class-λ laws** (§3): SeedIslandBirth pair + Eased + EventTriggered/nucleus-guard +
   LogitAdjust + soft-gated Amplify + Persistence(+skeleton cache) + LengthSigma.
   SegFocalGamma OMITTED pending its per-checkpoint recalibration (γ staged at measured γ*∈{0,1}
   only if grad-share gain ≥ +0.5pp). Hard costate margin-gate stays DESIGNED-ONLY behind #268.

---

## §1 — THE EXACT WitnessProgram (every lever + its control law)

Compiled through `tac.witness_dsl.curriculum_dsl` over the `Mod32SegOnlyControlBase` factory
(S5 row 13 — the config-generator SoT; no hand-edited launch.sh). Every flag below verified
against the live levelset-trainer argparse this session (251 quoted flags; zero invented).
`DirectionalBasisRebalance` hardcodes `--n-dir-freqs 4` — the program must not double-set it
(S1 interface note); ARM-PRIMARY does not invoke that factory (guard §0-3), it inherits the
control basis from the base.

### 1.1 Program sketch

```python
prog = WitnessProgram(
    purpose="T5 crucible ARM-PRIMARY: pointer-aimed islands+band+pose full stack",
    base=Mod32SegOnlyControlBase(),          # S5 row 13; mod32cap-identical unless overridden
    curriculum=sealed_205_curriculum(cfg, handoff="event"),   # + laws in §2
    levers=[
        # ISLANDS core (S5 FIRE set; laws §3)
        SeedIslandBirth(), SeedIslandEased(), EventTriggeredCurriculum(),
        LogitAdjust(tau=1.0), AmplifyIsland(form="hinge", weight=1.0, gated="witness_alone"),
        PersistenceTopology(weight=1.0, warmup=275), CacheGtSkeleton(), LengthSigma("fitted-20260707"),
        # BAND (trained-with; §0-1)
        AnalyticLaneRenderBand(start=350, boundary_relative=True),   # LBND4 at byte-close (B6)
        # FINISHER
        MuonWarmStart(lr_final_frac=0.1),
        # RATE Class-D
        WeightEntropyPenaltyMLX(lam=15),
        # POSE (two-track; §6)  — pose flags via base overrides (see argv)
        # OBSERVABILITY (req F; default-ON, score-neutral)
        GNSpectrumProbe(),                    # NEW factory (build I-5)
        ChromaBoundarySharpen(weight=0.1, margin_band=1.0, start="tau_fire"),  # NEW stub (I-4)
    ],
)
prog.validate()   # fail-closed on any invented flag
```

### 1.2 Knob → control-law table (the binding contract)

Legend: law class (a)–(e); tags V-S=VERIFIED-VIA-SOURCE, V-A=VERIFIED-VIA-ANCHOR, D=DERIVED,
DPR=DEFAULT-PENDING-RECESS(named recess).

**Basis / representation**

| knob | value/law | class | tag |
|---|---|---|---|
| `--self-orient` | ON | a | V-A (−48% all-class directional carrier) |
| `--reorient-every` | 50 (also feeds the trigger-window law §2.2) | a | V-A (control-proven) |
| `--n-dir-freqs / --freq-across / --freq-along` | 4 / 32 / **8** (ordering guard §0-3; along=6 and across=8 are R2 arms) | a | V-S (mod32cap launch.sh) + guard |
| `--max-bank-freq` | 64 (stem-Nyquist cap) | a | V-S (lever_b_levelset_generator derivation) |
| bank flags | defaults (n-scales 4, n-orient0 6, f0 2.0, base 2.0, n-iso 4) | a | V-A (control-proven; bank-6 REFUSED rc=4) |
| `--activation hosc --siren-init` | ON | a | V-A (hosc 0.221 vs wire 0.265; siren-init = trainability) |
| `--hosc-beta → --hosc-beta-end` | **RAMP 1.0→4.00, shape geometric** (build B3 ~10 LOC; fallback `linear` with gap on record). Denominator = `--anneal-epochs` (event-bound, §2.1). **Completion guarantee:** anneal-epochs = Muon CAP AND anneal-complete ∈ finisher-fire precondition ⇒ β=4.00 provably reached before its consumer (the finisher) fires; cap-fire completes it by construction. | b | D (τ=ε=ħ equal-epochs-per-octave; endpoint β(fire)=4.00 #302) — the anti-M2 exemplar |
| `--softmax-temp-start → -end` | **RAMP 1.0→0.2 geometric** (`--tau-anneal-shape geometric`, exists). Same denominator + completion guarantee as β. τ_end LAW = resolution floor τ_end = max(τ_pix, τ*): the master-ledger measured resolution-floor claim ("0.05 = 20× sub-pixel aliasing") conflicts with mod32cap's configured 0.05; interim 0.2 sits inside the measured best window (best @τ≈0.31, freeze 0.216, flat below). | b | DPR (recess P-τ: τ_end ladder {0.4,0.2,0.1,0.05} oracle re-score on stage ckpts + the fixed-τ control arm as final discriminator) |
| `--chroma --palette-anchor` | ON | a | V-A (control-proven; chroma = measured d_seg lever) |
| `--seg-chroma-boundary-weight/-margin-band/-start-epoch` | 0.1 / 1.0 / **event-anchored at tau-fire** (boundary must exist before chroma-matching pays) | d+e (annulus-gated by construction) | DPR (S1-R5; n96 mechanism measured, n600 never fired) |
| `--mod-dim / --hidden-dim` | 32 / 96 (mod48 = separate 2-point secondary arm, NOT primary) | a | V-A (basis PRIOR to capacity; +0.0135 rate per step costed) |

**Islands (details + per-class laws in §3)**

| knob | value/law | class | tag |
|---|---|---|---|
| `--seed-islands` + paint-then-SDF structured init (+include-lane) | ON at ep0 (priming); **acceptance gate: ep0 `part_frac[lane] > 0` (≈0.006) measured, not flag-presence** | a | V-A (#291 lane_FN 0.00713→0.00211) |
| `--seed-island-eased` | eased homotopy over `--seed-anneal-epochs 275` (shape per #323). **Completion guarantee:** seed-anneal-complete ∈ CE-exit precondition; CE cap 300 ≥ 275 ⇒ worst-case truncation ≤ 25 ep, event-safe (movable dilation is 1-Lipschitz — truncation degrades smoothly). | b | V-A (movable transfer PROVEN) |
| `--island-dilate-px` | 1 (2 is measured FP-costly ~15:1) | a | V-A |
| `--witness-alone-island-loss` | ON — the SOFT margin-gate (support = witness-alone errors) | e | V-A (#300: lane within-flip −45% while total descended) |
| `--amplify-weight/-form` | 1.0 / hinge, gated by witness-alone soft-gate; hard costate margin-gate (Ω = big-3 margin preserved) DESIGNED-ONLY behind #268 exact-S_R | e | V-A (uniform amplify measured net-negative; gated form net-positive by construction) |
| `--logit-adjust-loss-tau` (+ per-class) | 1.0; per-class Menon log-prior shift (lane −5.13 / movable −4.39, measured n600 priors); zero-byte | e | V-A (priors measured; byte-identity boundary proven) |
| `--persistence-loss-weight / --persistence-warmup-epochs` | 1.0 / 275 (boundary-relative via `--curriculum-reanchor-levers`) | b | V-A (stagger; ep300 collision harm 3.4×) |
| `--cache-gt-skeleton` | ON (bit-identical speed, #260) | a | V-A |
| `--length-weight` + `--length-sigma-matrix` | 0.001 + `fitted-20260707` σ_ij (per-class-pair; σ[Road-Lane]=0.377 — uniform over-penalizes lane 2.7×) | a+e | D (scorer's own Young-angle fit — the anti-cargo exemplar) |
| `--seg-focal-gamma` | **OMIT** (γ=0); staged re-entry ONLY at measured per-ckpt γ*∈{0,1} with grad-share gain ≥ +0.5pp | d | DPR (recess P6 = S5-R3 per-ckpt recalibration; staged γ=2.0 was cargo) |
| `--eikonal-weight → --eikonal-weight-end` | **RAMP 0.05→0.10**, step at the FIRED tau boundary (`--curriculum-reanchor-levers`); adaptive-ε flags ON (#318/#320 — self-deriving CFL law) | b+c | D (π_int ≳ 1 island MCF-protection; paint+eik retains 93% vs 52% raw) — engaged because ARM-PRIMARY is an islands arm (the eik-0 control comparison lives in mod32cap) |
| `--eikonal-viscosity(-anneal)` | default-OFF, **fire-on-creep** event law (predicate: eikonal-term share > 40% of total loss for 2 consecutive windows OR gnorm-hijack alarm) with lr×0.1 + rollback response; caps: ≤2 firings/run | d | DPR (litsweep guard-redesign; viscosity never fairly tested post-confound — req B triple-test before ship, else the lever stays OFF and only the alarm ships) |
| `--boundary-distance-weight` / MarginFieldHead / UniWARD / MarginSaliency / GFC / FiLM-family / AACoverageRender / lane_carried / MicroBatch / SoftBoundary | **NOT in ARM-PRIMARY** — dispositions per S5 (DEFER/RETIRE with named reasons; AA additionally G3 decode-blocked) | — | V-A (S5 table rows 6*, 17, 18, 24–31, 32–33) |

*AACoverageRender is S5-FIRE but as its OWN arm after the B-3 AA-decode build + fine-mode memory
gate — it does not ride ARM-PRIMARY (seam + shipping blocker G3).

**Schedule / optimizer (laws in §2)**

| knob | value/law | class | tag |
|---|---|---|---|
| `--curriculum-event-triggered --curriculum-nucleus-guard` | ON; plateau rel-eps 1e-4 / windows 25 / min-stage 250 + **verdict co-predicate** (build B2) + reorient-aware window | d | V-S wired (#315) + M-S2-3 hardening |
| `--curriculum-reanchor-levers` | ON (all followers boundary-relative) | a | V-S (#302 M1) |
| `--handoff-readiness-telemetry` | ON (score-neutral ⇒ default-ON) | a | V-S |
| `--anneal-epochs` | **= Muon CAP (726)** — the M-S2-2 truncation fix; both anneals complete AT the finisher boundary | a (binding the b-laws) | V-A (measured truncation 3.177/4.00, 0.216/0.05) |
| `--muon-start-epoch` | 726 as **CAP**; fire = event (§2.3) | d | V-A (cold ep726 fire = measured counterexample) |
| `--muon-warm-start-momentum` | ON | a | V-A (+27.5% cold quench measured) |
| `--muon-lr / --muon-lr-final-frac` | 0.002 / **0.1 cosine over the finisher** (river-valley: NS fixes direction, flat LR cannot settle; τ_e=305 ep receipt) | b | V-A/D |
| `--muon-momentum / --muon-ns-steps` | 0.95 / 5 | a | V-A (literature-settled) |
| `--ema-decay / --ema-decay-finisher` | 0.997 / **0.9995** (π_ema window law: ρ_fin ≈ 0.1–0.3× finisher steps; 0.997 averages 1.6% of a 274-ep finisher) | a | D (window law) + DPR (byte-closed A/B rides this run; regression guard is the fail-safe) |
| rewarmup | `--stage-transition-rewarmup-epochs 20 --stage-transition-rewarmup-floor 0.1` shape linear (keep, don't churn) + `--stage-transition-reset-moments` at BOUNDARIES; **NEVER reset moments on RESUME** (measured: restored moments damp spikes 6.7× vs 25.3×) | c | D: n_rw = ⌈k·(1/((1−β₂)·steps_per_ep))⌉, k=1.5, β₂=0.999, 75 steps/ep ⇒ 20 |
| `--lr / --lr-end / --lr-schedule` | 1e-3 / 1e-4 / cosine | a | V-A J-PROVISIONAL (PR95-echo acknowledged; the HVP sharpness trace feeds the run-2 law η_max = c/λ_max, c≈38 edge-of-stability) |
| `--adam-beta2` | 0.999 (0.9999 = secondary A/B arm only, never primary-confounding) | a | V-S default |
| `--per-group-grad-clip` | **ON** (closes the measured baseline-vs-stack confound; telemetry F10 stamps the delta) | a | V-A (store-11 confound) |
| `--epochs` | 1000 as CAP; run-end = per-class meat exit (build B5) | d | D (terminal time is a decision variable) |
| `--closed-loop-control` | ON with built bounded caps (eikonal bump/max/stop-after) | d | V-S (built; no longer bit-identical when it acts — accepted, S4 supersession) |
| spike guard | rollback mode + lr-cut (NEVER median-freeze) | a | V-A (confound L1/L2) |
| `--verdict-pairs / --verdict-batch / --accum-pairs` | 0 (all-600) / 64 / 8 | a | V-A (n600 non-negotiable; 64 SAFE never-slower) |
| `--seed` + determinism | 0; per-stage EMA-shadow checkpoints; resumable; self-orient state persisted (build F6) | a | non-negotiable |

**Rate / pose (laws in §5/§6)**

| knob | value/law | class | tag |
|---|---|---|---|
| `--weight-entropy-penalty-lambda` | 15 constant from ep0 (objective stationarity) | a | DPR (λ*∈{5,15,30}; torch −19.6% is mechanism-proof only; kill via twin-lag telemetry) |
| `--lane-render-band` + `--lane-band-*` | ON; start 350 → boundary-relative (tau-fire+50); tau .85 / eps .35 / weight 1.0 (the #205 measured settings); umask at byte-close | a+d | V-A (fired #205 ep300+; value ONLY trained-with) |
| `--lane-band-dash-comb` | **OFF** until the comb-REGISTRATION audit passes (recess P1); then in-training A/B only | d | V-A (L65: mis-phase risks real lane; render-composite measured net-negative) |
| `--w-pose` | staged law §6: 0 until tau-fire, ramp→1.0 over the 20-ep rewarmup window (build W1; interim if W1 unlanded: 1.0 from ep0 — the #205 measured pattern); self-deriving score-matched extension w = r(t)·clip(5/√(10·max(d̂_pose, 2.5e-4)), 1, 100) shadow-first | b→c | D (score law) + DPR (M5 bounded smoke) |
| `--pose-carrier --pose-carrier-source` | ON / `generated` (store-nothing; `real_keyframe` warp-real-luma EXCLUDED — wrong mover) | a | V-S/V-A |
| ξ coder q_levels (byte-close) | law: largest q with Δ(pose-term) < 0.002 (sweep 4096/1024/256) | c | DPR (S6-M4) |

---

## §2 — THE DERIVED SCHEDULE (schedule-as-policy, per the operator extension)

### 2.1 The stage graph IS the policy; the fixed sequence is the fail-safe law

The schedule is a **stage DAG with DECIDE-selected transitions**, not a hardcoded list. ONE
default path is the fail-safe cap-path (req B): every event trigger degrades to its epoch CAP,
so a dead/vacuous trigger yields the capped fixed schedule — never an unbounded or truncated run.

```
P(prime, ep0) → CE → TAU → FIN(warm-Muon) → END
                         ↘ SOLVE (chain-A GO branch) → END
   FIN --regression-guard-trip--> RESTORE-BEST --DECIDE--> { TAU-continue(AdamW, frozen levels)
                                                            | SOLVE (if basin-PD) | END }
   TAU/FIN --per-class-meat-exhausted (all classes)--> END   (run-end meat exit, build B5)
```

Repetition is admitted ONLY as event-triggered restore-and-continue (never blind re-entry: the
control's tau stage was measurably exhausted; a cycle there is theater). Transition selection is
costate-DECIDE (advisory-autonomous through the built trainer event flags + bounded closed-loop;
anything else operator-GO).

**Blended (fractional) transitions, per the operator extension:** the CE→TAU hand-off is NOT a
step switch — it is a **blend law**: τ/β continue their geometric paths across the boundary
(the energy is one continuation), the LR re-warms over 20 ep (law §1.2), moments reset at the
boundary, and followers (band +50, chroma-boundary, w_pose ramp, eikonal step) engage
boundary-relative with their own ramps. The only BINARY switches kept are (i) the optimizer swap
AdamW→Muon (blended in effect via warm-start momentum + LR rewarmup — the measured cure for the
quench) and (ii) loss-term engagement where blending is measured harmful or meaningless
(l7: dropped defect; smooth stage: dropped, measured raises d_seg).

### 2.2 Event exits — the tested-trigger contract (req B: backtest + injection + cap, else NOT in the launch config)

| transition | fire predicate | cap (fail-safe) | req-B status |
|---|---|---|---|
| CE→TAU | plateau(ep_loss rel-eps 1e-4, W=25, min-stage 250) ∧ **verdict co-predicate** (trailing V=4 n600-verdict d_seg slope rel-eps ~5e-3/25ep — the honesty fix for M-S2-3's first-eligible-epoch vacuity) ∧ nucleus guard (π₁=w/σ≳5, island arms) ∧ **seed-anneal-complete** | 300 | plateau: BACKTESTED (fires ep251 alone — hence co-predicate); co-predicate: build B2 + injection test T-1 owed BEFORE launch; window reorient-aware (never spans a `--reorient-every` boundary) |
| TAU→FIN | tau-plateau (ep_loss ∧ verdict co-predicate) ∧ **anneal-complete (β=β_end ∧ τ=τ_end — the anti-M2 precondition)** ∧ quadratic_basin advisory (§2.3) | 726 (`--anneal-epochs` = this cap ⇒ anneal complete at cap by construction) | Muon event trigger = build B1 (~80 LOC); run-1 admissible WITHOUT B1: out-of-process spectrum advisory + cap-fire (still anneal-complete by the flag identity) |
| FIN→END | regression guard: NO new best within 100 ep of fire ⇒ restore-best + exit | budget law §2.4 | build B4 (~30 LOC) + injection test T-2; fail-safe = epochs cap 1000 |
| run-end | per-class powerlaw/exponential remaining-meat < floor for EVERY class (a total-only detector exits while the binding class still pays — recess P6b calibrates the per-class split) | 1000 | build B5 (wire into the existing clean early-stop arming); backtest on mod32cap trace (M-S2-4: would have saved 76–125 ep); injection test T-3 |

**Every anneal is a precondition of its consumer** (the generalized anti-M2 law, registered as
a new equation, §10): CE-scoped anneals (seed) gate the CE exit; run-scoped anneals (τ, β) have
denominator = the finisher cap and gate the finisher fire. No denominator is a free-running
clock detached from its consumer.

### 2.3 The finisher fire predicate (curvature-aware, S3)

`quadratic_basin` := (gradient-Krylov T_k is PD ∧ Newton decrement < floor ∧ no usable negative
curvature |λ₋|·Δ²/2 < floor). MEASURED at ep650-EMA (K=8): **strongly indefinite** — the basin
predicate would correctly have HELD where the control cold-fired. Run-1 realization: the
`GNSpectrumProbe` runs out-of-process on the per-stage checkpoints (every 25 ep at stage
boundaries); the controller emits fire/hold ADVISORY; the trainer fires at cap regardless
(fail-safe). Full in-trainer `ExitEvent(criterion="quadratic_basin")` is build B1.

**Chain-A branch (inbound):** if chain-A returns TerminalSolve GO (full-P GN/CG from the basin
beats the entry best at n600 verify), enable the SOLVE stage: fire it when quadratic_basin is PD
(after the finisher, or replacing its tail on DECIDE). If NO-GO or unreturned at launch: SOLVE
stays a DSL TrainerSupportGap (compiles to nothing), warm-Muon is the sole finisher. Either way
the launch config is identical — SOLVE is additive and gated.

### 2.4 Self-deriving schedule parameters (operator extension #4)

- **Finisher budget law:** cap_fin = clamp(k·τ̂_e, 150, 350) epochs, k=1.5, with τ̂_e (the
  recovery time-constant) re-estimated ONLINE from the running exponential tail fit (telemetry
  F3). Anchor: control τ_e=305 ep > its 274-ep budget = the measured failure. Interim (before
  first online estimate): 274 (the cap-1000 remainder). [D]
- **Rewarmup law:** n_rw = ⌈1.5/((1−β₂)·steps_per_epoch)⌉ (moment-memory bound) — 20 ep at
  75 steps/ep; recomputed automatically if β₂ or batch geometry changes. [D]
- **Trigger-window law:** plateau windows never span a reorient boundary (reorient-aware
  segmentation); verdict co-predicate horizon V·cadence = 100 ep ≥ 2× reorient period. [D from
  M-S2-3]
- **LR ceiling law (run-2, fed by this run's spectra):** η_max = c/λ̂_max per stage, c≈38
  (edge-of-stability). Run-1 keeps the measured 1e-3→1e-4 cosine (don't churn blind). [DPR]

### 2.5 PR95 cargo-cult audit — adopted verbatim from S2 §3 (34 rows) + S1/S4/S5/S6 face tables

The consolidated audit is the union of the five seat tables; the draft adds one row: **the
"anneal denominator = run length" convention is X/DROPPED everywhere** (M-S2-2) — replaced by
the §2.2 consumer-precondition law. No element of the launch config is PR95-inherited without a
derivation or a measured keep: the stage ORDER is re-derived (mirror-descent → Γ-dequantization
→ κ-buster), the CLOCK is events-with-caps, the optimizer constants carry receipts, l7/smooth/
QAT-stage/σ-noise/C1a are dropped with receipts, τ/β shapes are geometric-derived, and the two
acknowledged PR95-echo survivors (LR endpoints, β₂) are tagged J-PROVISIONAL with their
measurement path named.

---

## §3 — THE CURRICULUM (levels, priming, LADDER homotopies, per-class laws)

### 3.1 Priming (ep0)

Structured init paint-then-SDF (+include-lane); FiLM identity-safe init; hosc siren-init;
**acceptance gate (hard):** ep0 log shows `part_frac[lane] > 0` (≈0.006) — measured, not
flag-presence; launch aborts (governed) if the seed did not take. FinerBiasInit NOT in
ARM-PRIMARY (init-confound; own cheap A/B, S1-R4).

### 3.2 Per-class sub-curricula (operator extension #3 + requirement C)

The LADDER (easier-surrogate → real-target under per-class-λ gates; L56: LADDER ⊂ our costate,
per-class-λ generalization):

| class | birth law | homotopy (surrogate→real) | λ gate law (fractional) | interference guard |
|---|---|---|---|---|
| **Movable** (44.8% of flips) | seed at ep0, **eased SDF-dilation** radius r(t): 1-Lipschitz anneal over 275 ep (class-b ramp; completion gates CE exit) | dilation-GO (proven transfer) | logit-adjust prior −4.39 (constant per-class shift) + amplify hinge gated on witness-alone support | nucleus guard holds TAU until π₁≳5; persistence loss protects births from MCF |
| **Lane** (19.1%) | **render-side authority = the analytic band** (trained-with, start tau-fire+50); training birth = curve-prior VP-tangent eased seed (manifold-preserving); barrier is AREA/MARGIN, NOT dash-bridging (measured — no oriented-tangent gap-bridging law) | band engages as a ramped follower (20-ep cosine rewarmup on its weight — the deconflict law, measured 3.4× collision harm at hard engage) | logit-adjust −5.13 + LengthSigma σ[Road-Lane]=0.377 (removes the 2.7× uniform-length lane tax) + witness-alone soft-gate | dash-comb OFF until registration audit; chroma-boundary joins at tau-fire (annulus-gated) |
| **Big-3** (Road/Undrivable/MyCar jitter, 36.1%) | no birth needed — boundary-position curriculum | τ geometric anneal = the jitter curriculum (margin sharpening); eikonal ramp 0.05→0.10 at tau-fire protects interfaces | uniform (λ=1); chroma-boundary sharpener at the annulus | LengthSigma prevents over-penalizing lane while big-3 smooth |

**Coupling guards:** (i) one homotopy parameter per epoch neighborhood (stagger 275/275/boundary
+50 — the measured-collision law); (ii) loss-geometry family cap ≤ 2 new levers per arm
(LogitAdjust + LengthSigma are the two; BoundaryDistance/MarginFieldHead round-2); (iii) island
support is FRACTIONAL by construction (witness-alone soft-gate; the hard Ω-gate waits for #268);
(iv) per-class λ telemetry (F-rows) monitors that lane share collapses after band engage — if it
does NOT, the lane_offloaded premise is wrong in-flight (S1's live monitor).

### 3.3 Levels (the LevelPath objects)

τ and β are `LevelPath`s (geometric; §1.2 laws, duplicate-emitter protection vs Curriculum
fields). Eikonal is a stepped path (0.05→0.10 at the fired boundary, re-anchored). w_pose is a
staged ramp (§6). Seed/persistence warmups are boundary-relative followers. LR is
cosine-with-rewarmup (per-stage). Every path's denominator is an EVENT (stage fire or cap),
never a detached clock — the §2.2 completion law.

---

## §4 — THE COSTATE CONFIG (SENSE → DECIDE → ACT) + requirement-F telemetry

### 4.1 SENSE inventory (existing wired + new; score-neutral read-only rows default-ON)

Existing (verified, S3 §1.1): λ ladder (λ_seg=100 exact · λ_pose=5/√(10·d_pose) ·
λ_bytes=6.659e-7 S/B exact) · shadow controller + trajectory classification · powerlaw_meat_exit
(1st-order, curvature-blind — kept as the cheap in-run fit) · annulus telemetry (#333) ·
per-class verdict sensors (#315 family) · liveness/skip/spike alarms · duty-to-measure +
failure ledger + sensitivity map producers.

**NEW: `gn_spectrum.checkpoint_lanczos` producer** (S3; harness LANDED at
`experiments/t5_s3_hvp_lanczos_probe.py`): per stage-boundary checkpoint, the tuple
{λ_max, λ₋, n_neg, Newton decrement, grad_norm, k_pairs} → `producer_bridge` ProducerSignal +
costate_digest section. Cadence: every stage boundary + every 100 ep in FIN (out-of-process,
off the critical path).

**Requirement-F telemetry rows (all 12; forensics → live rows + alarms; alarms obey req B):**

| F# | row (from the forensic finding) | alarm law | build (~LOC) |
|---|---|---|---|
| F1 | per-epoch EFFECTIVE anneal state (β_eff, τ_eff, % of path) | consumer-precondition alarm: finisher eligible while anneal < 100% ⇒ ALARM (M2 class) | ~30 |
| F2 | stage-transition optimizer health (loss/gnorm 25-ep pre/post window) | quench detector: post-switch loss > 1.1× pre ⇒ ALARM + regression-guard arm (M1) | ~40 |
| F3 | online remaining-meat + τ̂_e (per-class tail fits, every verdict) | feeds the finisher budget law + run-end exit (M4) | ~30 (reuse powerlaw_exit) |
| F4 | trigger would-fire audit rows (every trigger, every epoch: predicate values + would-fire) | vacuity alarm: trigger would-fire at first-eligible-epoch ⇒ flag (M3) | ~25 |
| F5 | checkpoint-cadence Lanczos SENSE row | basin/indefiniteness state to DECIDE (S3) | ~40 (wrap landed harness) |
| F6 | **persist self-orient state in checkpoints** + save-time reconstruction-gap check | gap > 1% ⇒ ALARM (the measured +4.3% ckpt-probe gap) | ~60 |
| F7 | launch-time argv→lever ENGAGED predicates → activation ledger (value ≠ argparse default) | closes ledger≠truth (S5 R1) | ~80 (+backfill tool ~60) |
| F8 | single-lever activation attribution (paired with/without delta at each activation epoch, from the verdict stream) | attribution row per lever engage | ~50 |
| F9 | effective-config provenance fail-loud (ckpt `__cfg_*` vs CLI; the G1 freq_along class) | mismatch ⇒ REFUSE (L2 confound gate) | ~40 (byte-close side = build B7) |
| F10 | config-delta-vs-named-baseline row at launch (the grad-clip confound class) | any silent delta vs declared baseline ⇒ printed + stamped | ~30 |
| F11 | pose wall watch (once w_pose>0): d_pose advisory + FiLM read-back vs the 1.5e-4 kill | d_pose not < 0.01 by tau-end ⇒ ALARM (L1 fallback advisory) | ~30 |
| F12 | per-stage wall-clock + epochs-past-meat | epochs-past-meat > 50 ⇒ ALARM (M-S2-4 class) | ~20 |

### 4.2 DECIDE laws

- **Arbitration spine:** rank by expected ΔS/cost via the λ ladder with propagated stderr;
  POWERPLAY never-regress refusal (central predicted ΔS ≥ 0 ⇒ refused); duty-to-measure queue
  re-ranked from S5's FIRE/DEFER/RETIRE table with the diversity floor f=1/4 for never-fired.
- **Curvature-aware exhaustion (replaces scalar-only):** stage exhausted ⟺ 1st-order meat <
  floor ∧ T_k PD ∧ decrement < floor ∧ no usable λ₋. Shadow-first (advisory) in run-1.
- **Spectrum-rate mixture forecast** replaces linear-λ extrapolation (the ep450-miss fix);
  recalibrated at each stage-boundary spectrum row.
- **Trust region:** accepted-step discipline ½·λ̂_max·‖Δθ‖² ≤ accepted surrogate-risk;
  rollback-to-best enforcement on per-stage EMA checkpoints.
- **Prohibitions (binding, litsweep):** no GradNorm/PCGrad per-step multi-term balancing (the
  eikonal is the canary, not the underdog); no hypergradient LR; never reset moments on resume.
- **Pose weight law (shadow-first):** w_pose* = clip(5/√(10·max(d̂_pose, 2.5e-4)), 1, 100) —
  emitted as advisory until build W1b lands; the 2.5e-4 floor = the marginal crossover
  (effort-law: never chase pose below it).

### 4.3 ACT boundary (CONTAINMENT unchanged)

Autonomous = advisory only: shadow rows, recommendation sets, $0 checkpoint probes, event-flag
CONDITION INPUTS, RECOMMENDED-CONFIG emission compiled/validated through the DSL (never raw flag
edits). In-run actuation = the built bounded closed-loop exclusively. Heavy/paid launches, run
stops, live-config changes = operator-GO. No emitted recommendation may target
evaluator/permission surfaces.

---

## §5 — THE RATE PLAN (budget to the byte + requirement-E class dispositions)

### 5.1 Byte budget (ARM-PRIMARY at byte-close; λ_bytes = 6.659e-7 S/B exact)

| section | bytes (central) | band | source/law |
|---|---:|---|---|
| base+code int8+brotli (mod32/h96) | 82,193 | measured | S4 measured ep650 |
| → after #336 waterfill post-pass (gated) | **60,000** | [52,000, 68,000] | pre-registered; gate Δd_seg ≤ +5e-5 (probe P4) |
| → −FEED-08k grammar rev-2k (parity-deinterleave + per-dim delta + col-major, bit-identical) | −3,108 | measured | FEED-08k; fold into byte-close (build B8) |
| lane band LBND4 (build B6) | 30,892 | [18,000 (smoothed, probe P5), 30,892] | measured coder; LBND2 fallback 41,562 |
| pose ξ (store-nothing, derive-H, q per M4 law) | 4,500 | [2,700, 6,929] | measured endpoints |
| manifest (brotli'd) + zip STORED 1-char | ~800 | measured components | S4 P5 |
| **archive.zip central** | **≈ 93,100** | **[86,000, 99,000] (waterfill-pass) / [112,000, 120,000] (waterfill-fail fallback)** | rate central **0.0620** [0.0573, 0.0659] / fallback [0.0746, 0.0799] |

Budget verdict: central lands UNDER the ~105 KB sub-0.15 headroom line; the waterfill gate is
what pays the band+pose+capacity budget back — hence Class B/D are launch-relevant, not optional.

### 5.2 Requirement-E disposition — every rate class (FOLD / DEFER-with-reason / DEAD-with-receipt)

| class | lever | disposition |
|---|---|---|
| **D in-training** (⚠ launch-blocking decisions) | WeightEntropyPenaltyMLX | **FOLD** (λ=15 ep0; §0-2) |
| | flat-minima #242 | DEFER (designed-unbuilt; two unmeasured Class-D levers = confound) |
| | #154 entropy-penalized / #110 latent-structure | DEFER (queue behind the λ A/B verdict; code stream is where structure remains) |
| | #111 variable-grid QAT / QAT-at-target-depth | DEFER pending the **Class-D×B RECESS** (waterfill an entropy-shaped stage-ckpt of THIS run vs unshaped ep650 — $0, in-run) |
| | PR95-L16 C1a | **DEAD-with-receipt** (measured net-negative; `supersedes_c1a=True`) |
| **B bit-depth** | #336 KKT waterfill | **FOLD-gated** (probe P4; [52,68] KB @ Δd_seg ≤ +5e-5; kill > +2e-4) |
| | uniform int6/5/4 | receipt curve only (54.2/41.3/29.0 KB) — dominated by waterfill by convexity |
| **C structural** | TropNNC #311 / KD #74 / low-rank #71 | DEFER (post-run; run-2 candidates on the landed basin) |
| **E invariance/orbit** | weight-permutation canonicalization | **DEAD-with-receipt** (best arm −8 B; most arms HURT +72/+251/+339 B; group-theory §C header "MEASURED NO") |
| | orbit-coding principle (H(orbit)=H(rep)+H(g)) | FOLD-as-realized (derive-H + rule-118 ARE it); register the candidate equation (§10) |
| **A post-hoc coding** | grammar rev-2k (FEED-08k) + brotli-manifest + STORED-1-char | **FOLD** (measured −3,108 B + −436 B) |
| | PR95-L25 temporal-delta on code | **DEAD-with-receipt** (+64% measured) |
| | base-weight coder migration (lzma/range) | DEAD-with-receipt (base AT order-0 floor; lzma +1%) |
| | hyperprior entropy models | DEFER-with-reason (twice-ruled-out at our payload scale) |
| **F payload-specific** | pose derive-H | DONE (−43.2 KB, live) |
| | LBND4 band-coeff coding (+smoothed) | **FOLD** (B6 + probe P5) |
| | #307 contour-string flip coding | **DEAD-with-receipt** (0.820 B/flip > 0.65 bar; "fragmented confetti") |
| | latent-AR constriction | DEFER (PTC1 dominated brotli — receipt) |
| **G scorer-invariance** | #153 P-SUFF findings | FOLD-findings (dominated-rung ablations consumed in the band/τ choices); null-space compiler #47 DEFER (post-run) |

---

## §6 — THE POSE PLAN (S6 two-track, integrated)

- **Track 1 (ARM-PRIMARY): pose ON.** `--pose-carrier --pose-carrier-source generated
  --w-pose <staged law>`. Engagement: **staged at the tau-fire boundary** with a ramp over the
  20-ep rewarmup window (build W1; class-b law, completion guaranteed inside the rewarmup whose
  denominator is the boundary event). **Interim if W1 unlanded at GO: w_pose 1.0 from ep0** —
  the #205 measured pattern (d_pose descended to ~0.0019–0.0023 advisory while d_seg reached
  0.004752@ep300); tagged DPR. The self-deriving score-matched law (§4.2) runs shadow-first.
- **Track 2 (twin): seg-only attribution arm** — ARM-PRIMARY minus pose (w_pose 0, no carrier),
  capped at the tau stage (~450 ep), launched sequentially after the primary is confirmed stable
  (memory governor: no blind concurrency). Twin supplies the M5 co-predicate: primary d_seg
  within +5% of twin at matched epochs, else pose-interference ALARM (F11).
- **Kill law:** converged d_pose > **1.5e-4** (T_1-infeasibility threshold, S6 derivation) ⇒
  pose-ON-as-designed KILLED for this vehicle → activate the symposium L1 Jacobian-coefficient
  $0 gate (fallback carrier). Mid-run watch: d_pose not < 0.01 by tau-end ⇒ ALARM + operator
  advisory (run continues — its d_seg value stands).
- **Null-texture enhancement (#206 L3 full form):** BUILD W2, gated — NOT launch-blocking; the
  bounded M5 read (first ~300 ep of ARM-PRIMARY, per-stage byte-closes) measures whether the
  base store-nothing mechanism suffices before W2 is funded.
- **Byte side:** ξ q-levels law (§1.2); derive-H live (H_bytes=0); the pose section is the best
  S/byte buy in the stack (~1e-2 S/B if M5 passes) — and dead bytes if it fails, which is why
  the kill threshold ships with the config.
- **Error-bar honesty:** the 0.018 pose-term target is BORROWED-ancestor; 0.026 "spare" is
  OPERATOR-STATED — neither is measured on this vehicle (compendium gap #10). The plan's numbers
  use only the measured thresholds (1.5e-4 / 3.2e-5 derived from the score law + measured rate).

---

## §7 — THE MEASUREMENT PLAN (PowerPlay-ordered: cheapest NEW decisive first; every row pre-registered)

Named recess items consumed: HVP full-P (P3), comb-registration audit (P1), FEED-08l fresh-eyes
review (P2), Class-D×B interaction (in-run recess), waterfill probe (P4).

| # | probe | cost | predicted band (grounding) | kill/proceed |
|---|---|---|---|---|
| P0 | S6 byte-close rows (3 compositions) | DONE | rate 0.0556/0.0834/0.0602 | landed |
| P1 | **comb-REGISTRATION audit** (GT-conditioned mark/gap phase) | $0, ~1 h | comb separates marks/gaps with margin ≥ the GT-sep floor | FAIL ⇒ `--lane-band-dash-comb` stays OFF (already default); comb A/B removed from run-2 queue |
| P2 | **FEED-08l fresh-eyes review** (recovery-written verdict + durable JSON) | $0, reading | verdict survives with its 2-scoreable-rung + oracle-form limits | FAIL ⇒ S1 lane_carried demotion reverts to OPEN; R2 arm set gains along=26 |
| P3 | **full-P HVP-Lanczos ladder** (ep650-EMA/live, ep726, ep1000; K∈{8,32,128}) | $0, hrs CPU / ~min MLX-GPU (CPU spot-verify) | ep650 indefiniteness persists at K=128 (\|λ₋\|/λ_max > 0.5) | shrinks < 0.1 ⇒ wall is capacity/basis ⇒ Arm A case STRENGTHENS; basin predicate recalibrated either way |
| P4 | **#336 waterfill** on mod32cap ep650 (exact CLI in S4-R1) | $0, 30–90 min | base+code ∈ [52,68] KB @ Δd_seg ≤ +5e-5 | kill > +2e-4 @ mean-bits 6 ⇒ compress-half dominated ⇒ budget falls back to §5.1 fallback row |
| P5 | LBND4-on-smoothed source | $0, ~1 min | 18–22 KB | ≥ 24,149 B ⇒ no gain over smoothed LBND2 |
| P6 | SegFocalGamma per-ckpt recalibration (S5-R3) + FiLM-PR telemetry (S5-R4) + flip-share stability ep650 (S5-R2) + per-class meat-tail split (S2-R3) | $0, ~2–3 h total | γ*∈{0,1}; island share 55–70% stable; lane tail power-law | γ gain < +0.5pp ⇒ γ stays OMIT; island share <35% ⇒ big-3 levers re-rank first |
| P7 | **n600 realized-parity row** on ep650 (S6-M1; `--so-freq-along 8`) | ~30–60 min | realized d_seg 0.0034 ± 3e-4; inflate ≤ 20 min | Δ > +5e-4 ⇒ decode/quant defect — FIX before ANY run; > 25 min ⇒ budget risk |
| P8 | band ROI numerator (S6-M2: P7 + band) | ~30–60 min | net ΔS < 0 at LBND2 pricing | ≥ 0 ⇒ band DEFERS to LBND4 (B6) re-price; trained-with arm still fires (post-hoc ≠ trained-with, measured) |
| P9 | pose chain validation + ξ q-sweep (S6-M3/M4 on the 0703 store-nothing ckpt) | ~1–2 h | realized d_pose within ~2× training-side; q1024 ≈ 2–4 KB @ Δterm < 0.002 | frame0 not bit-exact ⇒ #239-class regression — fix first |
| P10 | exact-eval leg dry-run (S6-M6, `--run-exact-eval`, advisory) | ~2–3 h | evaluate.py parses; recomputed-S delta < 1e-5 | any failure ⇒ NO pointer-row plan until the leg fires |
| T-1/2/3 | **req-B injection tests** for co-predicate / regression guard / meat exit through the LIVE trainer path | $0, ~2 h | fires-when-should + silent-when-shouldn't | any trigger failing = that trigger NOT in launch config (cap-only mode) |
| **RUN** | **ARM-PRIMARY launch (operator GO)** + per-stage byte-closes (each stage ckpt = an early row) + in-run Class-D×B recess (waterfill the λ=15 tau-boundary ckpt vs unshaped ep650) | ~24 h train | §9 ladder | per-stage kills: d_seg > control at ep100-matched (islands); pose watch F11; anneal/quench alarms F1/F2 |
| TWIN | seg-only twin to ~ep450 (sequential) | ~13 h | primary d_seg within +5% at matched epochs | breach ⇒ pose-interference finding (run continues; attribution recorded) |
| **ROW** | byte-close (B6+B7+B8 in; waterfill pass; grammar rev-2k) → **Linux x86_64 contest-CPU `upstream/evaluate.py`** (Modal CPU) | operator GO, ~$1–2 | §9 | THE success definition; CUDA axis paired per non-negotiable when PR-bound |

---

## §8 — WALL-CLOCK PLAN (score-first; lexicographic)

Measured base: ~107 s/ep (mod32cap class). Event exits alone are worth ~35% of the control run
(M-S2-4/5: 76–125 ep past-exhaustion + a 274-ep worse-than-entry finisher) at ZERO score cost.
Projection for ARM-PRIMARY (event path): CE ~275 + TAU ~350–400 + FIN ~150–250 (budget law) ≈
**775–925 ep ≈ 23–27 h**; cap-path worst case 1000 ep ≈ 30 h. Twin ~450 ep ≈ 13 h sequential.
$0 probe wave (P1–P10, T-1..3): ~1.5 days wall, mostly foreground CPU, governed. Total to the
exact row: **~4–5 days**, one governed heavy launch + one paid CPU eval. Score-neutral speed
riders: CacheGtSkeleton, verdict-batch 64, pose-carrier (measured speed-SAVER — never drop pose
for time). GPU-reorient only if its parity probe passes (never a bit-exact surface).

---

## §9 — PREDICTED S LADDER (honest bands) + #363 ASSUMPTION TABLE

### 9.1 The ladder (pre-registered; Dykstra-grounded: no band exceeds its measured flip-mass ceiling)

| rung | central | band | grounding |
|---|---|---|---|
| training-side d_seg @ matched ep650 | 0.0019 | [0.0012, 0.0028] | islands composed Δ −0.10..−0.25 seg-units on 0.0034 (S5 arm bands + FEED-07c ceiling ~0.0013 upper-bound-caveat); kill > 0.0030 |
| byte-closed realized d_seg | +0..+1e-4 over training-side | — | int8+R delta prior; waterfill gate ≤ +5e-5 |
| rate | 0.0620 | [0.0573, 0.0659] (waterfill-pass) / [0.0746, 0.0799] (fail) | §5.1 measured components |
| pose term | 0.039 (at the kill edge) | [0.018, 0.105] | 0.018 = BORROWED-ancestor target; 0.105 = R1 measured floor; M5 decides |
| **S [macOS advisory → contest-CPU]** | **≈ 0.29** | **[0.186, 0.47]** | sum of rungs |

**Plain statement (means/ends firewall):** the run-1 CENTRAL prediction does NOT cross 0.19110.
Crossing requires the joint favorable tail: island birth near its ceiling (d_seg ≲ 0.0012) ∧
pose ≤ ~1.5e-4 ∧ waterfill-pass rate ≤ ~0.062. That joint event is exactly what the per-stage
byte-closes, event exits, and F-alarms are built to detect early and steer toward — and the
plan's kills fire long before a wasted full run. **T_3 (0.15) is NOT in run-1's band**; the
named run-2 headroom items are: in-training comb (post P1), AACoverageRender arm (post B-3),
StepNative β 4→8 finisher fork, mod-dim 2-point + Arm-E deep pass, TerminalSolve (chain-A),
BoundaryDistance w*=0.2, per-class λ hard gate (#268). This is stated per NO-FAKE: a launch-ready
optimal stack with honest bands beats an optimistic fake band every time.

### 9.2 Load-bearing assumptions (#363)

| assumption | tag | gate |
|---|---|---|
| ep225 flip-share transfers to ep650 (islands necessity arithmetic) | ASSUMED | P6 flip-share probe |
| islands composed net-positive at n600 TRAINING (ceiling arithmetic is oracle-side) | INFERRED | RUN per-stage kills (ep100/ep-matched) |
| store-nothing pose beats R1's 0.0011 floor by ~7× via w_pose>0 | ASSUMED-with-mechanism | M5 (F11 watch + 1.5e-4 kill) |
| waterfill Δd_seg ≤ +5e-5 at [52,68] KB | DERIVED-convexity, Δd_seg side ASSUMED | P4 |
| band net-positive at conservative FEED-07d edge | NOT guaranteed (S4) | P8 + trained-with attribution (F8) |
| lane_offloaded regime (band+comb carry lane/dash) | PROVISIONAL | P1 + P2 + ordering guard (along=8 primary) |
| τ_end = 0.2 interim (resolution-floor law unresolved vs master ledger) | DPR | recess P-τ + fixed-τ discriminator arm |
| WeightEntropy λ=15 transfers mechanism (not magnitude) from torch | ASSUMED-mechanism | in-run bytes row + twin-lag kill |
| Muon −32% keep-anchor transfers to this residual regime | ASSUMED (borrowed-number flagged) | P3 spectra + FIN regression guard |
| K=8 spectrum → full-P transfer (ep650 "not exhausted") | INFERRED | P3 (kill band pre-registered) |
| 15 registry equations AWAITING_VERIFICATION relied on (two-regime, step-native, adaptive-ε, chroma-at-annulus, …) | PROVISIONAL per #363 | anchors land from this run's A/Bs (§10 map) |

---

## §10 — BUILD LIST = INTEGRATION MAP (requirement G: DSL · equations · DAG · tasks · code sites)

**Supersessions (mark at land):** #183 per-lever A/B campaign → superseded by §7; #124 REORDER
DAG → superseded by §2 stage-DAG; #285 converged next-run config → superseded by this draft.
Reconcile with #337 BUILD-WAVE before starting any item (no duplicate work).

Launch-blocking = LB. All commits via serializer with `--base-content-sha256` (live absorption
class, store 16).

| id | build | ~LOC | DSL integration | equation registration | code site | task |
|---|---|---:|---|---|---|---|
| B1 | Muon event trigger (fire predicate §2.2; cap fallback) | ~80 | `ExitEvent("quadratic_basin")` gap-kind → real | `muon_switch_conditioning_criterion_v1` anchor-update | trainer `_evt_*` family | TaskCreate; not LB (cap path admissible) |
| B2 | plateau verdict co-predicate + reorient-aware window | ~40 | ExitEvent plateau params | new `stage_exit_event_with_cap_failsafe_v1` | trainer `_evt_resolve_seg_form` | LB for event-mode honesty (else cap-only mode) |
| B3 | geometric hosc-β anneal choice | ~10 | LevelPath("hosc_beta","geometric") | `hosc_activation_saturation_trainability_v1` anchor | trainer `_hosc_beta_for_epoch` | not LB (linear fallback, gap on record) |
| B4 | finisher regression guard (new-best-in-100 else restore+exit) | ~30 | StageSpec exit | new `finisher_budget_tau_e_law_v1` (with §2.4 law) | trainer finisher loop | LB (the measured 274-ep failure class) |
| B5 | per-class powerlaw_meat run-end wire into clean early-stop | ~40 | `ExitEvent("powerlaw_meat")` gap→real | `weak_kam_powerlaw_tail_exit_v1` anchor | `witness_control.powerlaw_exit` ↔ trainer early-stop arm | not LB (cap 1000) |
| B6 | LBND4 decode inline into `_INFLATE_PY` (G2) | small | — | `lane_band_res_entropy_stage_v1` anchor | `tools/levelset_byte_close_and_eval.py` | LB at byte-close (−0.0071 S free) |
| B7 | byte-close consumes ckpt `__cfg_freq_*`, CLI override-only + mismatch REFUSAL (G1) | ~40 | — | — (L2 confound gate per Confound non-negotiable) | same tool `_load_levelset_ckpt` | LB at byte-close |
| B8 | grammar rev-2k fold (FEED-08k + brotli-manifest + STORED-1-char) | ~80 | — | new `code_stream_grammar_rev2k_v1` (measured −3,108/−436 B) | same tool + inflate | LB at byte-close |
| W1 | w_pose stage-gating + ramp (+W1b: shadow score-matched law) | ~40 (+40) | Lever `PoseStagedOn` (emit_stub_lever) | new `w_pose_score_matched_schedule_v1` | trainer loss assembly + witness_control shadow | not LB (interim ep0/1.0 tagged DPR) |
| W2 | ξ-consistent null-texture (#206 L3 full form) | TBD-spec | Lever stub | — | trainer pose-carrier block | NOT LB; gated on M5 read |
| I-1 | anti-M2 law registration | 0 (registry) | — | new `anneal_completion_is_consumer_precondition_v1` (M-S2-2 anchor) | canonical_equations | TaskCreate |
| I-2 | orbit-coding candidate registration | 0 | — | `rule118_orbit_coding_free_action_counted_coords_v1` (G-C4) | canonical_equations | TaskCreate |
| I-3 | S5 R1 ledger reverse-map + backfill + byte-close drain hook | ~140 | activation_ledger ENGAGED predicates | — | `tools/launch_witness_run.py` + new backfill tool | LB for F7 (duty-to-measure realism) |
| I-4 | `ChromaBoundarySharpen` DSL stub-fold (flags exist, DSL doesn't hold) | ~30 | emit_stub_lever → factory | `chroma_decides_lane_and_movable_at_annulus_v1` anchor path | `witness_dsl/curriculum_dsl.py` | LB (lever in ARM-PRIMARY) |
| I-5 | `GNSpectrumProbe` observability Lever + `gn_spectrum.checkpoint_lanczos` producer + digest section | ~90 | new factory, default-ON score-neutral | — | `witness_control/producer_bridge.py` (+~L258), `shadow_controller._recommendations` (HOLD_STAGE_NEGATIVE_CURVATURE, FIRE_FINISHER_BASIN_ENTERED), `tools/costate_digest.py` | LB for F5 |
| F1–F12 | requirement-F telemetry rows (§4.1 table; ~475 LOC total) | ~475 | GNSpectrumProbe + telemetry default-ON levers | — | trainer telemetry emitters + witness_control consumers | F1,F2,F9,F10,F11 LB; rest strongly-wanted |
| T-1..3 | req-B injection tests (co-predicate, guard, meat exit) through the LIVE trainer path | ~120 test | — | — | tests beside trainer | LB for any trigger that ships armed |
| DAG/DSL/eq landing | this draft → DAG FEED entry + DSL program file + equation rows, same commit batch (triality) | — | WitnessProgram checked in as the launch SoT | rows above | `.omx/research/sub015_DAG_*` FEED + `witness_dsl` | P7 deliverable |

**Costate code-landing recap (req G):** gn_spectrum producer (I-5) · curvature-aware DECIDE laws
shadow-first (shadow_controller) · quadratic_basin ExitEvent as DSL gap-kind now, real via B1 ·
activation-ledger argv→lever ENGAGED ingest (I-3) · duty-to-measure re-ranked from S5's
FIRE/DEFER/RETIRE table (producer_bridge queue). Actuation boundary unchanged.

---

*Round-1 self-review before commit: (1) checked every launch flag against the live argparse
(251-flag extraction; the two seat-cited flags that are NOT trainer flags — `--pose-carrier-mode`
(byte-close tool) and the DSL-only lever names — are used only in their own surfaces); (2) the
predicted-S central honestly does NOT cross the pointer — stated plainly rather than tuned;
(3) the ordering guard is honored structurally (along=8 primary; Rebalance factory absent from
ARM-PRIMARY's lever list); (4) every anneal has a consumer-precondition completion guarantee or
an event-safe truncation argument; (5) requirement A is satisfied via chain-A branch + B1/B5 +
the SOLVE gated stage (training reserved where solving is unproven: the full-P in-trainer solve
is the only admissible form per the measured subset-overfit); (6) requirements B–G each have a
named section (B §2.2/T-1..3 · C §3.2 · D §7 ordering · E §5.2 · F §4.1 · G §10).*

Pointer 0.19110 UNMOVED — this draft is MEANS until the §7 ROW lands.
