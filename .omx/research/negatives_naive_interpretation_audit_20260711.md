# Negatives / naive-interpretation audit — this session's #433–#440 results (task #440, 2026-07-11)

**Auditor:** rigor-auditor subagent. **Scope:** re-grade this session's negative/mixed results on the
verdict-scope ladder (INSTANCE < FORMULATION < FAMILY < PARADIGM); flag any naive/toy/binary read;
sweep this session's artifacts for toy-cited-as-real. **$0 · no GPU · no pid disturbed · read-only.**
**Pointer 0.19108282 [contest-CPU] UNMOVED — this audit is MEANS; it moves nothing, it prevents a
mis-read from mis-directing the campaign.**

**STORES CONSULTED:** CLAUDE.md (verdict-scope ladder · Forbidden-premature-KILL · non-n600 allergy ·
NO-FAKE #3 synthetic/#8 surrogate) · `docs/operating_manual_craft_handoff.md` (the ten
competence-lookalikes: §means-as-ends, plausible-summary, borrowed-number, round-finished≠clean-pass) ·
`verdict_scope_ladder_formulation_level_one_failure_not_family_dead_20260708.md` ·
`.omx/research/negative_findings_register_20260709/{auditor_A,auditor_B}.md` (the #390 register — this
audit EXTENDS it with the #433–#440 set, does not duplicate its #390 corpus rows) · the 6 target memos +
their DAG FEED / commit / equation-anchor surfaces · `costate_organ_capabilities_limits_envelope_20260711.md`
(§3 plateau/transient thesis) · `.omx/state/canonical_frontier_pointer.json` (live pointer verified).

---

## HEADLINE (MEASURED, cross-surface)

**Every one of the six target results is ALREADY correctly scoped — not only in its memo, but in its
compressed DAG FEED, commit message, and equation-anchor surfaces.** I checked each result at all four
surfaces (memo body → DAG FEED one-liner → git commit subject → registered equation anchor) and the
p-values, `provisional-until-accrual`, `INSTANCE`/`FORMULATION` labels, `n=1`, and `[NOT PROVABLE $0]`
qualifiers SURVIVE the compression. This is the interpretation-hygiene discipline working. The audit's
value is therefore (a) pinning the TRUE scope so a future reader cannot drift it, (b) five genuine
residual risks (all downstream-compression / cross-doc, none a live over-claim), (c) a clean
toy-cited-as-real sweep. **No result in this session is mis-read as more than it is; two carry a
downstream-compression risk worth a guard-line (#434 label, #439 PROCEED).**

---

## PER-TARGET RE-GRADE

### #434 Transient Forge — "HONEST NEGATIVE"
- **Measured:** forge prior-mean arm WF MAE 0.003882 marginally beats real-only ridge 0.003902 but LOSES
  to persistence 0.002792 and incumbent E_prototype_bregman 0.002839 → ADOPTED=FALSE. 6/7 folds pick λ=0.
  #433 aniso acid P(aniso) 0.003880 vs Q(iso) 0.003902, separation +2.3e-5 within noise. sim2real gap
  large (synthetic-fold ~1e-4 vs real 0.0039). tier-2 (bias-free) NOT fired.
- **TRUE verdict_scope: INSTANCE / TEST-REGIME-scoped negative** — NOT "synthetic data is useless"
  (FAMILY) and NOT "the engine is broken." The negative says: *on an n=1 PLATEAU-dominated test
  trajectory, manufactured-transient synthetic data does not confer walk-forward skill.* The memo §3(c)
  states this precisely ("the adoption TEST is still n=1… unprovable on a plateau-only test set").
- **Naive/binary risk (LOW, downstream-compression):** the two-word label "HONEST NEGATIVE" traveling
  WITHOUT its "on a plateau-only test set, sim2real gap unclosed" qualifier could be read as
  "synthetic-data path is dead." It is not — the ENGINE + honest gate + the diversity/regret/influence
  machinery all BANK, and the aniso-separation "within noise" is explicitly "coupling absent OR simulator
  can't express it — indistinguishable at 0 tier-2 runs."
- **Honest reading:** a correctly-built engine met an honest gate on the WRONG test regime; the value is
  unprovable until the TEST set contains transients (the plateau cannot exercise transient-forecasting).
- **Reactivation:** tier-2 micro-runs (bias-free referee, also the ≥3-record graduation data) OR a
  transient-rich real test trajectory; then re-run the same gate. Synthetic-fold wins are NEVER adoption
  evidence (NO-FAKE #3) — correctly held.

### #436 self-dispatch — "beats both persistence AND global-single-best"
- **Measured:** dispatcher WF 0.001596 (meta-λ ON) / 0.001738 (OFF) beats persistence 0.002792 AND
  global-single-best T_gp 0.001852; routes 5/7 folds to oracle. **But:** margin over GSB 13.8%/6.2% at
  n=7; differs from GSB on 4 folds = 3 wins/1 loss, sign-test **p≈0.31 NOT significant**; policy +
  thresholds IN-SAMPLE-DERIVED (GP memo §4 analyzed these exact folds).
- **TRUE verdict_scope: INSTANCE, provisional-until-accrual (≥2 trajectories).** Correctly a
  BEATS-BOTH-BUT-PROVISIONAL, explicitly "NOT an honest-negative, NOT an adoption."
- **Naive/binary risk (NONE at any surface):** the memo §4 pre-empts it verbatim ("I refuse to narrate
  it as one [adoption]"); the DAG FEED carries "p≈0.31 — NOT significant" and "IN-SAMPLE-DERIVED… owed at
  ≥2 trajectories." The task's worry ("read as the organ knows what to use when — DONE") does NOT
  materialize on any surface. The MECHANISM (past-only classify→route, leak-free) is sound and banks; the
  POLICY's transfer is unmeasured.
- **Honest reading:** a parameter-free structural prior confirms CONSISTENT-WITH (not out-of-sample-of)
  its motivating data; nominal robust win, significance not cleared.
- **Reactivation:** register `regime_conditional_dispatch_v1` only when ≥5 records show per-state dispatch
  beats GSB out-of-noise AND per-fold significance clears; owed 3-arm plateau→prototype extension.

### GP-costate (#439 sibling, Warnick 2025) — "first to beat persistence WF"
- **Measured:** GP arm WF 0.001852 beats persistence 0.002792 (−34%, first arm to) AND dominates every
  trained surrogate. **But** per-fold 4/7, **sign-test p=0.50** (coin-flip in the plateau tail); win
  CONCENTRATED in early transient folds (ep50–100); **per-class MAE WORSE** (0.0405 vs 0.0117); produces
  NO lever field (binding gate N/A). Forecast leg is EXACT-Warnick (FD parity rel 2.85e-9); the map to the
  true nonlinear/discrete d_seg costate is a stated SURROGATE (no Fréchet-linearized posterior built).
- **TRUE verdict_scope: INSTANCE, PARTIAL-NOT-DECISIVE.** A win on the score-relevant AGGREGATE mean WF;
  a loss on per-fold significance, plateau tail, per-class decomposition, and lever field.
- **Naive/binary risk (LOW):** "first to beat persistence WF" is TRUE but must carry "mean-aggregate, on
  the 7-fold sealed trajectory, transient-concentrated, p=0.50 per-fold." The memo's own verdict line
  ("NOT an n=1 miracle; the honest best-forecaster-so-far with a calibrated covariance") is the correct
  compression.
- **Honest reading:** CONFIRMS-and-sharpens the envelope thesis (edge = transient forecasting), does not
  overturn the plateau→persistence law; best deployed as the base-drift forecaster with meta-λ still
  selecting persistence in the pure plateau.
- **Reactivation:** ≥2 transient-rich trajectories to clear per-fold significance; build the
  Fréchet-linearized nonlinear posterior if a true-costate solve (not a scalar forecast) is wanted.

### #433 aniso per-class λ — formulation wins, aniso DIRECTION forecast-neutral
- **Measured:** the physics prior-mean FORMULATION beats the isotropic-independent baseline A by −18% WF
  / −18% early-fold and is the first ridge-family arm to beat persistence at EARLY folds (0.004024 vs
  0.004715) — the n=1-fragility regime. **But** the anisotropic-coupled DIRECTION is forecast-neutral:
  the M0=I ablation Q is equal-or-better (Δ 0.000115 WF, ~3.7%, within 30× fold-variance noise). The
  whole family still LOSES to persistence at n=9 (plateau posture).
- **TRUE verdict_scope: split — the FORMULATION win is real (score-law-pinned κ cure, beats the old L
  failure 4–5× early); the aniso DIRECTION is INSTANCE-scoped forecast-neutral** (1 plateau-dominated
  trajectory; a transient-rich window is the discriminating data). C_phys (Lane→Road 0.494) remains a
  MEASURED STRUCTURAL readout the #430 composer consumes — physics-as-structure, not yet physics-as-
  forecast-edge.
- **Naive/binary risk (LOW — under-claim risk, not over-claim):** the danger is reading "aniso
  forecast-neutral" as "anisotropy is wrong / the P0 directive failed." It is NOT — the coupling is a
  measured fact independent of the forecast gate; only its FORECAST edge is unproven at this n. The memo
  splits this cleanly ("physics as structure, not yet as forecast edge").
- **Honest reading:** the directive's physics is correctly modeled and STRUCTURALLY consumed; its
  forecast payoff is undecidable on a plateau at n=9.
- **Reactivation:** ≥3 organ-ledger records / a regime-rich interval to separate aniso-direction from the
  iso ablation.

### comma10k regime (thread inside #433) — 2 formulations INERT/worse
- **Measured:** the SCALE cure works (arm R early-fold 0.00527 vs old L 0.0222) but R ≈ plain ridge and is
  WORSE than the direction-free ablation Q ⇒ comma10k rarity DIRECTION adds nothing to forecasting here.
- **TRUE verdict_scope: FORMULATION ×2 (φ-rescale absorbed; EB-κ) + DIRECTION ×1 (score-law-pinned) —
  all measured; FAMILY OPEN.** Correctly "CLOSED for forecast use at this n on this vehicle," NOT killed:
  the rarity prior stays alive as a SENSOR/duty-queue instrument (its Lane 3.5× converges with the flip
  crux).
- **Naive/binary risk: NONE** — the memo uses the ladder verbatim and keeps the sensor role.
- **Reactivation:** ≥3 records / regime-rich intervals.

### organ family loses WF to persistence at the plateau
- **Measured (envelope §3 + ledger `passed_walkforward:false` all arms on the 8-interval #426 record;
  GP/dispatch 7-fold):** on the plateau every learned arm loses to persistence (fold ep150→175: heuristic
  0.0007 vs models 0.0017–0.0030).
- **TRUE reading: this is the ENVELOPE THESIS CONFIRMED, not "the organ fails."** The organ's designed
  edge is TRANSIENT-regime forecasting; on the plateau the meta-λ self-monitor CORRECTLY flags
  prefer_persistence and the organ defers ("knowing when to use nothing"). The naive read ("organ
  fails") inverts a correct-by-design deferral into a failure.
- **⚠ Cross-doc subtlety (interpretation-hygiene flag):** the envelope (8-interval #426 record) reports
  the prototype family BEATING persistence (WF 0.002501–0.002513 vs 0.002896), while the GP/dispatch
  backtests (9-interval / 7-fold SEALED trajectory) report the prototype LOSING to persistence
  (E_prototype_bregman 0.002839, +1.7%). These are **DIFFERENT fold sets on different interval counts,
  not a contradiction** — each is correct within its set. A compressed reader who sees only one could
  wrongly conclude the other is stale/wrong. Any statement about "prototype vs persistence" MUST name its
  fold set.

### #439 T1 SEAL — GATE VERDICT PROCEED  ⚠ SHARPEST
- **Measured:** SEAL = 3 consecutive clean passes, zero blocking defects (correct, byte-identical-when-OFF,
  deterministic/resume-safe, fused-R-safe, cannot-destabilize). The $0 n600 measurement is a
  TARGET-VALIDITY / PREMISE check: ξ-advected GT_tie[p-1] predicts pair-p GT tie at wMSE 0.10864 (RMS
  0.330 px, in the predicted 0.09–0.43 px band) vs no-warp 0.11659 = **−6.8%, wins 70.6% of 599 pairs,
  ~6–7% of cross-frame tie variance is ξ-predictable (MODEST)**.
- **TRUE verdict_scope: the SEAL clears the STABILITY gate ONLY; the $0 measurement clears the PREMISE
  gate ONLY (and is n600-authority — the highest-authority $0 result in the batch). d_seg-EFFICACY is
  UNMEASURED and unprovable $0** — the tie-residual→d_seg map goes through SegNet argmax (nonlinear) and
  needs a phase-ON vs matched-phase-OFF training A/B.
- **Naive/binary risk (MODERATE — the one to guard):** "T1 SEALED → PROCEED" in isolation reads like "T1
  helps d_seg / T1 is adopted." It does NOT mean that. The memo's "Honesty on 'helps'" paragraph is
  impeccable ("the d_seg-EFFICACY claim… is UNPROVEN and unprovable $0… delivered BY the run"), the DAG
  FEED carries "[NOT PROVABLE $0]," and the equation anchor
  (`t1_tie_coord_xi_transport_positive_modest_measured_20260711` on
  `lane_groundframe_xi_transport_no_collapse_v1`) correctly registers only the PREMISE (ξ-transport pays
  modestly on the un-canonicalized tie chart) and explicitly notes "d_seg-crossing needs the phase-ON/OFF
  training A/B." **All four surfaces hold. The residual risk is purely a future reader compressing
  "PROCEED" → "helps."** Guard-line for downstream: PROCEED = "safe to FIRE + premise validated," never
  "lowers d_seg."
- **Honest reading:** a stability+premise gate correctly cleared; efficacy is the run's job, cleanly
  attributable ONLY with the matched phase-OFF control (`--seg-phase-advect-weight 0.0`), else the arm
  confounds T1 with the rest of the #432 cascade.
- **Reactivation / owed:** fire ON + matched phase-OFF control; watch pre-registered T1 telemetry
  (`blink_fit_frac`↑ / correlated spike direction / d_pose non-rising); the byte-closed d_seg delta is
  the registrable law.

### governor fix — the caught mis-diagnosis
- **Measured:** the initial "reservation-leak" hypothesis was WRONG; the reproduced-live PRIMARY cause is
  an over-broad `OUR_JOBS_PATTERN` ps-substring regex charging +25 GiB phantom growth to incidental
  matches (a lone `ugrep` over the source matched it). Fix = material-RSS floor for unregistered ps-only
  sub-2-GiB matches + admission-path auto-reconcile + reservation-store sweep.
- **TRUE reading: a POSITIVE example of re-derive-from-primary** — the mechanism was FOUND by reading the
  read path and reproduced live, NOT guessed; the wrong first hypothesis was overturned by the primary
  artifact. This is the operating-manual's "attack your own conclusion" working.
- **⚠ Naming-drift flag (LOW):** the memo FILENAME `governor_reservation_leak_fix_20260711.md` preserves
  the DISPROVEN hypothesis label. The body + DAG FEED correct it ("over-broad ps-pattern match — NOT a
  reservation leak"), but a future grep-by-filename could re-anchor the wrong cause. Cosmetic; noting for
  the record, not a re-name request (append-only provenance).

---

## CODEBASE / SESSION SWEEP — toy-cited-as-real (non-n600 allergy)

**Result: CLEAN for this session's 29 artifacts (dated 2026-07-11).** Every memo carrying a score-like
claim also carries an axis/scope label (`[macOS advisory] NON-PROMOTABLE` / `n=1` / `verdict_scope` /
`MEANS` / `[contest-CPU]`). Specific checks:

- **All 6 organ/witness memos** labeled; the organ trajectory ledger records `passed_walkforward:false`
  honestly per arm with `[macOS advisory] NON-PROMOTABLE`.
- **#433 uses `gt_n96.npz` cached margins/lstars twice** — for PHYSICS STRUCTURE (flip-temperature
  ε_flip=1.048, annulus 0.577 in 5.6% area, per-class bulk/boundary split), used as a DESIGN-prior
  direction, NOT as a score claim; it CROSS-CHECKS against the n600 #333 annulus anchor (0.57). Acceptable
  and labeled — but note for the record: **these physics numbers are n96-cached, not n600**; the forecast
  gate itself is on the real trajectory. No violation, but do not cite the n96 annulus/flip-temperature as
  n600.
- **T1 SEAL is n600** (599 scored transitions) — the highest-authority $0 measurement in the batch; the
  lone "smoke" token is a DENIAL ("it is not a smoke-only pass"), used correctly.
- **The two files with zero scope-label tokens** (`paper_harvest_leg_disposition`,
  `papers_checked_knowledge_distillation_sota`) are paper-survey / triality-disposition ledgers; their
  "beats" tokens describe OTHER papers' claims (e.g. "plain logit+feature KD beats CWD/CIRKD" is paper
  2604.25530's claim), not our measurements. Correct — literature ledgers, not score claims.
- **Cross-doc pointer staleness (note, not a target mis-read):** the #390 register (auditor A/B,
  2026-07-09) cites the pointer as **0.19110**; the LIVE `canonical_frontier_pointer.json` is
  **0.19108282** (moved 2026-07-10 via `lane_clickpolish_pr110_frontier`). This session's memos correctly
  cite 0.19108282. Read the #390 register's pointer as of its date.

---

## INTERPRETATION-HYGIENE SUMMARY

1. **This session's negatives are the MATURE case, not the failure case:** every result is scoped on the
   ladder at all four surfaces (memo / FEED / commit / equation). The discipline held.
2. **Two downstream-compression guard-lines** (the residual risk is future readers, not current text):
   - **#434 "HONEST NEGATIVE"** = INSTANCE/test-regime-scoped (plateau-only, sim2real unclosed), NOT
     "synthetic-data path dead." Engine + gate BANK.
   - **#439 "PROCEED"** = "safe to FIRE + premise validated," NOT "T1 lowers d_seg." d_seg efficacy is
     the run's job (phase-ON/OFF A/B).
3. **One cross-doc subtlety:** "prototype beats persistence" (envelope, 8-interval) vs "prototype loses"
   (GP/dispatch, 7-fold) — different fold sets, both correct; always name the fold set.
4. **Two under-claim guards:** "organ loses at plateau" = envelope thesis CONFIRMED + correct deferral,
   NOT "organ fails"; "#433 aniso forecast-neutral" = direction's forecast edge unproven at n=9, NOT
   "anisotropy is wrong" (C_phys is a measured structural fact).
5. **One positive-example lesson:** the governor mis-diagnosis was overturned by primary-artifact
   re-derivation (found + reproduced, not guessed) — the "attack your own conclusion" craft working; the
   filename retains the disproven label (cosmetic).
6. **No new toy-cited-as-real violation this session.** The #390 register remains the corpus-level record;
   this audit extends it with the #433–#440 set.

**MEANS. Pointer 0.19108282 [contest-CPU] UNMOVED.** The value of this audit is that no result above can
mis-direct the campaign: #434 does not close the synthetic path, #439 PROCEED does not claim a d_seg win,
#436/GP/#433 do not claim adoption, and the plateau losses are the thesis, not a failure — the sub-0.15
gap is still ENTIRELY d_seg and the live levers (T1 A/B, V9·CGauge, the organ at ≥2 trajectories) are the
next real rows.
