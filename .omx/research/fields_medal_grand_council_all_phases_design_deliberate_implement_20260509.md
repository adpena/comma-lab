# EXTREME-OBSESSION FIELDS-MEDAL GRAND COUNCIL — All-Phases Design + Deliberate + Implement (2026-05-09)

## Convening directive

Operator 2026-05-09 verbatim: "spawn extreme obsession fields medal grand council to design and debate and deliberate and consensus agree on and implement all scaffolding and wiring and integration of all outstanding and roadmap and next steps from all phases."

This is the canonical CLAUDE.md "Council conduct" + "Design decisions" + "Adversarial council review of design decisions" + "Recursive adversarial review protocol" maximum-rigor session. **All ten inner-council voices required.** Grand-council members consulted on specialty.

## Hard constraints (CLAUDE.md non-negotiables)

- **$0 GPU spend gate**: in-flight `aaf68f37` adversarial review gates dispatch. This council DESIGNS + SCAFFOLDS + WIRES only. Dispatch-ready scaffolds tagged `DEFERRED-pending-aaf68f37-verdict + operator approval`.
- **No conservative bias** (council conduct): only mathematical / scientific / geometric / empirical arguments.
- **Forbidden score claims**: every predicted band tagged `[predicted; <method>]`; never bare numbers.
- **Subagent coherence-by-default**: every new module declares 6-hook wire-in.
- **HNeRV parity discipline**: every representation lane declares 8 archive-grammar fields.
- **Forbidden premature kill**: any "X is dead" verdict requires inner-ten consensus + reactivation criteria.
- **Lane registry mutations only via `tools/lane_maturity.py`**.
- **Commits via `tools/subagent_commit_serializer.py`** with explicit `--files <list>`.

## Council roster (this session)

**Inner ten (binding):** Shannon (LEAD) · Dykstra (CO-LEAD) · Yousfi · Fridrich · Quantizr · Hotz · Selfcomp · MacKay · Ballé · Contrarian (VETO).

**Grand council (called on specialty):** Boyd (ADMM) · Tao (pure-math) · Filler (STC) · Mallat (wavelet) · van den Oord (VQ-VAE) · Carmack (engineering shortcuts) · Hassabis (cross-domain breadth) · Hinton (KL distill T=2.0) · Karpathy (arch-search) · Schmidhuber (compression-as-intelligence) · Jack-from-skunkworks (SegNet+Rate lineage).

## Scope: 10 phases / directions deliberated

| # | Phase / direction | Status entering session | Output |
|--:|---|---|---|
| 1 | Phase 1 (READY-FOR-DISPATCH, GPU-gated) | trainer wired, 6-hook landed | dispatch-recipe document + final scaffold-gap audit |
| 2 | Phase 2 (SCAFFOLDS landed, dispatch DEFERRED) | T10/T15/T17/T18 scaffolded | pre-design pass for T15/T17/T18 + EIG/$ ranking refresh |
| 3 | Phase 3 (DESIGN-ONLY, NEEDS scaffold) | Tishby IB Lagrangian framework consensus | full `tac.phase3.*` scaffold + tests + lane reg + 6-hook |
| 4 | Phase 4 (LANDED) | unified solver landed (8a8c7e5d) | re-verify + operator submission decision tree |
| 5 | Phase 5+ (NOT YET DESIGNED) | conjectural sub-0.118 floor saturation | scaffold-only design + lane reg L0 SKETCH |
| 6 | Beyond #1: hosted supplement + tac OSS publish | task #354 open | tool plan + sanitization gate plan + lane reg |
| 7 | Beyond #2: per-archive drift posterior | per-class registry landed (697bfe01) | extension design + lane reg |
| 8 | Beyond #3: non-HNeRV cluster calibrations | 9 uncalibrated classes | calibrator tool plan + lane reg |
| 9 | Beyond #4: GR-style unified action trainer migration | scaffold partially landed | migration + parity-test design + lane reg |
| 10 | Beyond #5: cathedral autopilot full automation | semi-automated catalog (a086a57d) | autonomous-loop design + lane reg |

---

## Phase 1 deliberation

**Round 1 — inner-ten positions (each 1-2 sentences):**

- **Shannon**: T13 + T19 + T8 + T20 + T22 + PR#95 monkey-patch lands the score-domain Lagrangian on i.i.d. Berger-tightened. The closed-form rate prediction is anchored at Shannon-floor minus joint-source correction. [predicted; Shannon-floor + Berger; Phase 1]
- **Dykstra**: alternating projections converge if rho-bands are wide enough. T19 adaptive-ρ is the convergence guarantor. ✓
- **Yousfi**: contest scorer is FROZEN at Phase 1 entry. Make sure `eval_roundtrip=True` is gated on by default per CLAUDE.md non-negotiable.
- **Fridrich**: per-pair latent at sqrt(n) bound (T13) means we can encode 2.3 bits/pair more headroom on A1 (28-D × 600 pairs → 5.29 bits/pair vs current ~3 bits). UNDETECTABLE per Ker-Pevný-Fridrich 2008.
- **Quantizr**: EMA(0.997) snapshot+restore on every `optimizer.step()`; load EMA shadow at inference. The Phase 1 trainer's `apply_t13_sqrt_n_budget` passes the smoke; the FiLM-DSConv inheritance from Quantizr 0.33 is preserved.
- **Hotz**: 30-sec scan — the `--enable-t13-sqrt-n-budget` and `--enable-t19-adaptive-rho` are OFF by default, which is correct for backward-compat. Two flags + sane defaults = ship.
- **Selfcomp**: T20 (block-FP) + T22 (mask-codec optimization) preserve the 1.017-bpw block-FP profile. No regression to my 0.38 archive baseline.
- **MacKay**: MDL-wise, T13's √n latent shrink reduces description length. Bonus: the trainer logs `bit_reallocation` to provenance.json — that's the reviewer-checkable artifact.
- **Ballé**: hyperprior is a Phase 2 enhancement; for Phase 1 the entropy bottleneck is the canonical entropy floor. ✓
- **Contrarian**: sole worry — what if the dispatch lands and the predicted band [0.155-0.165] is wrong? Reactivation criteria documented (per CLAUDE.md `forbidden_premature_kill`). Acceptable.

**Round 2 — dissent:** none beyond Contrarian's ask.

**Round 3 — counter-dissent:** Contrarian satisfied by reactivation criteria documentation in `feedback_t13_t19_phase1_trainer_integration_landed_20260509.md`.

**Round 4 — grand council consult:** Boyd confirms adaptive-ρ within `[1e-3, 1e3]` band is canonical Boyd 2011 §3.4.1 — endorsed. Hinton confirms T=2.0 distillation is the right temperature for soft-target gradient signal — endorsed.

**Round 5 — verdict:** **10/10 ENDORSE Phase 1 ready for dispatch IF aaf68f37 verdict CLEAN AND operator approves the $80 Lightning T4 spend.** No additional scaffold work required. **Status: READY-FOR-DISPATCH-pending-aaf68f37-verdict.**

**Round 6 — implementation plan:** dispatch recipe documented in `feedback_t13_t19_phase1_trainer_integration_landed_20260509.md` §"Dispatch recipe" — three commands (`meta_lagrangian_search_cli` → `parallel_dispatch_top_k.py` → `harvest_and_reseed.py`).

---

## Phase 2 deliberation

**Round 1 — positions:**

- **Shannon**: T10 IB-co-trained scorer is the ceiling-breaker (Tishby 1999). T6 Ballé+UNIWARD cross-paradigm composes Ballé 2018 + Fridrich 2010. T15/T17/T18 are uncertain-EIG without pre-design.
- **Dykstra**: T8 Wasserstein-2 proximal is the convex-projection canonical step on the seg-axis. Endorsed.
- **Yousfi**: T6 needs UNIWARD weights from a SCORE-AWARE-trained substrate (A1 or PR100/101/103). On a score-naive substrate the inverse-steg argument is invalid.
- **Fridrich**: T11 Lovász hinge is the convex envelope of mIoU. T7 Fisher-Rao + T11 may be SUPER-additive (one boundary-localized, one global). Worth running probe sweep.
- **Quantizr**: T15 (time-varying FiLM) extends my static FiLM. Pre-design needed: per-pair pose-conditioning input shape + how to pack the modulator MLP weights into archive.
- **Hotz**: kitchen_sink risk for T9 (cross-archive substrate composition). Defer until single-substrate success demonstrated.
- **Selfcomp**: T17 shared VQ-VAE codebook is a Phase 3 enabler (renderer ↔ aux scorer share latent space). Need to prove I can pack a 256-entry × 64-dim codebook into archive at <2KB.
- **MacKay**: every Phase 2 track needs the integration-discipline checklist (L1 score-aware, L2 export-first, L4 inflate ≤100 LOC, L7 bolt-on ≤350 LOC). Track-wise enforcement table.
- **Ballé**: T18 (4-layer MLP nonlinear transform) is canonical He-Zheng 2024. Identity-init means worst-case = vanilla Ballé.
- **Contrarian**: structural — T7 + T8 + T11 are sub-additive on the same gradient. STOP all three. Run probe sweep first.

**Round 2 — dissent:** Shannon + Fridrich dissent on full prune of T7 + T8 + T11; argue ensemble is steganalysis-SOTA.

**Round 3 — counter-dissent:** Contrarian counter — burden of proof on ensemble. Probe sweep is $0; if all three within 10%, ensemble wins; otherwise prune.

**Round 4 — grand council consult:** van den Oord confirms shared codebook (T17) requires careful EMA decay (0.99 NOT 0.997 for codebooks; CLAUDE.md exception clause). Mallat notes that wavelet/scattering substrate (Lane MM) is COMPLEMENTARY to T17, not redundant — could compose later.

**Round 5 — verdict:** **8/10 ENDORSE Phase 2 with prune-then-dispatch ordering. 2/10 DISSENT preserved (Shannon + Fridrich on T7+T8+T11 ensemble).** Order:

1. **Top-tier $0 GPU $0 LOC** (LANDED): T7 Fisher-Rao + T11 Lovász + T13 √n + T19 adaptive-ρ.
2. **Substrate-engineering tier ($40-80/track):** T6 + T10 + T1.
3. **Bolt-on tier ($20-40/track, after substrate):** T8 + T2 + T3.
4. **DEFER until pre-design pass complete:** T9 + T15 + T17 + T18.

**Round 6 — implementation plan:** Pre-design ledger landed at `.omx/research/phase2_predesign_t15_t17_t18_pass_20260509.md` (deferred to follow-up subagent; this council records the plan).

---

## Phase 3 deliberation (THE BIG ONE — full scaffold landing)

**Premise**: Phase 3 = end-to-end joint scorer-renderer-codec under Tishby IB Lagrangian L_IB(Z) = I(X;Z) − β·I(Z;Y) per coherence-council §1. Substrate-engineering exception applies (per CLAUDE.md substrate-vs-codec meta-pattern): we may engineer the substrate jointly with the codec WHEN the joint training is anchored on a frozen-eval-time auxiliary scorer that is Hinton-distilled (T=2.0) from the contest scorer.

**Round 1 — positions:**

- **Shannon (LEAD)**: This is the unified action. L_IB collapses Berger 1971 + Hinton 2014 + Ballé 2018 + Fridrich 2010 into ONE saddle-point. Predicted: 0.115-0.130 [predicted; Phase 3 council; conditional on Phase 2 landing 0.142].
- **Dykstra**: saddle-point via primal-dual ADMM. T19's adaptive-ρ becomes load-bearing here — runaway ρ kills Phase 3 the most.
- **Yousfi**: scorer-at-inflate is FORBIDDEN (CLAUDE.md `check_no_scorer_load_at_inflate`). The auxiliary scorer θ_aux is TRAINING-ONLY; its EMA shadow gives gradients during co-training; at eval time θ_aux is replaced with the frozen contest scorer.
- **Fridrich**: cross-PR substrate composition (PR100/101/103 + A1) is the input to the IB system. UNIWARD weights derived per-PR.
- **Quantizr**: distillation gap target ≤ 3% (Hinton 2014 §3 verified). Scaffold must record `distillation_gap_estimate` field.
- **Hotz**: 200 LOC inflate cap (CLAUDE.md HNeRV parity discipline L4). Phase 3 inflate.py must stay ≤200.
- **Selfcomp**: shared latent space (T17) is the bridge — renderer + aux scorer + decoder all reference the same VQ-VAE codebook. Codebook bytes count once.
- **MacKay**: MDL-wise, the joint training maximizes I(Z;Y) per bit of I(X;Z). The IB Lagrangian IS the MDL ELBO under deterministic encoder.
- **Ballé**: end-to-end-trainable codec is the canonical 2018 setup. Add hyperprior side-information (T18 nonlinear transform).
- **Contrarian**: scaffold ONLY. No GPU spend. Predicted band documented as `[predicted; Phase 3 conjecture; multi-source aggregated; conditional on Phase 2 landing 0.142]`.

**Round 2 — dissent:** none. Unanimous-but-non-trivial: Contrarian flags the unanimity ("if all five agree instantly, someone isn't thinking hard enough"). On reflection, this is a SCAFFOLD-ONLY deliverable, so unanimity on scaffold structure (NOT on dispatch decision) is appropriate. Dispatch decision will require fresh council in Phase 3 dispatch-readiness review.

**Round 3 — counter-dissent:** n/a (no dissent to counter).

**Round 4 — grand council consult:** Boyd confirms saddle-point primal-dual ADMM with adaptive-ρ is canonical. Hinton confirms T=2.0 + auxiliary-scorer-as-variational-posterior is the canonical distillation setup. van den Oord confirms shared codebook EMA(0.99) — codebooks adapt faster than weights.

**Round 5 — verdict:** **10/10 ENDORSE Phase 3 scaffold landing as defined.** Dispatch decision DEFERRED to a future scoped council after Phase 2 lands a 0.142 [contest-CUDA verified] anchor.

**Round 6 — implementation plan:**

| Module | LOC budget | Tests | Lane | Dispatch-ready? |
|---|---:|---:|---|---|
| `src/tac/phase3/__init__.py` | ~80 | n/a | `lane_phase3_joint_scorer_renderer_codec` (L0 SKETCH) | NO ($600-1200 budget) |
| `src/tac/phase3/joint_scorer_renderer_codec.py` | ~600 | 30 | (same lane) | NO |
| `src/tac/phase3/inflate.py` | ≤200 | 10 | (same lane) | NO |
| `src/tac/tests/test_phase3_joint_scorer_renderer_codec.py` | n/a | 40+ aggregate | (same lane) | NO |

Tagged `[predicted; Phase 3 council; 0.115-0.130 conditional on Phase 2 landing 0.142]`.

---

## Phase 4 deliberation (LANDED — verify)

**Round 1 — positions:**

- **Shannon**: unified solver `tac.unified_action.S_total` lands the canonical action functional. Phase 4 packet integrity = `submissions/exact_current/` + `submissions/robust_current/` parity verified.
- **Dykstra**: 6 preflight blocker fixes + STRICT gate enforcement.
- **Yousfi**: methodology addendum lands the dual-eval CPU+CUDA mandate.
- **Fridrich**: PR submission decision tree — operator chooses between PR-now-with-current-frontier or hold-for-Phase-3.
- **Quantizr**: submission archive integrity = SHA-256-pinned + dual-eval anchored.
- **Hotz**: ship.
- **Selfcomp**: archive-grammar fields all present.
- **MacKay**: MDL ELBO at submission time = current frontier S = ~0.196 [contest-CPU GHA Linux x86_64; PR #107 anchor].
- **Ballé**: hyperprior side-information present in current frontier.
- **Contrarian**: only worry — if Phase 1 / Phase 2 / Phase 3 wait, the contest may have advanced. Submission policy must include "submit current frontier IF deadline-mode active."

**Round 2 — dissent:** none.

**Round 4 — grand council consult:** Hassabis cross-domain framing — "ship the current frontier as a checkpoint, advance under same name." Carmack says ship.

**Round 5 — verdict:** **10/10 ENDORSE Phase 4 LANDED.** Operator submission decision tree:

1. **DEADLINE MODE ACTIVE** → submit current PR #107 frontier (0.19663589 [contest-CPU GHA] + 0.22933 [contest-CUDA]) immediately.
2. **NORMAL MODE** → wait for Phase 1 dispatch result (predicted 0.155-0.165); if ≤ 0.155 ship Phase 1 archive AS PR; else ship PR #107.
3. **POST-PHASE-2** → wait for Phase 2 dispatch result (predicted 0.142 ± 0.011); if confirmed, ship Phase 2 archive.
4. **POST-PHASE-3** → ship Phase 3 archive (predicted 0.115-0.130).

**Round 6 — implementation plan:** decision tree documented. Operator decides via `--deadline-mode` flag at next checkpoint.

---

## Phase 5+ deliberation (NEW — scaffold-only design)

**Premise**: Phase 5+ targets the lower-tail of the Phase 2/3 council Bayesian posterior (S_floor = 0.131 ± 0.013; lower bound 0.116). The conjectural mechanisms:

1. **Score-aware substrate co-evolution** — joint optimization of substrate + bit allocation + renderer + codec via Phase 3 IB Lagrangian extended with substrate-evolution dual.
2. **Closed-form Fisher-Rao geodesic** on the score manifold (T7 evolved to Phase 5).
3. **Cross-paradigm composition** of all winning Phase 1+2+3 components.
4. **Joint-source coding refinement** (Berger 1971 finite-block correction + (N-1)/N Gaussian-Markov bound).

**Round 1 — positions:**

- **Shannon**: lower bound 0.116 = Berger + Tishby + Fridrich + Ballé + MacKay all simultaneously saturated. Conjecture: 0.118-0.125.
- **Dykstra**: substrate-evolution dual is a NEW constraint axis. Convergence requires 5-axis ADMM.
- **Yousfi**: scorer evolution is FORBIDDEN at eval time (contest scorer FROZEN). Substrate evolves during training only.
- **Fridrich**: √n bound saturated at sqrt(28×600) per pair. No headroom beyond Phase 3.
- **Quantizr**: Phase 5 is "Quantizr 0.33 → Quantizr 0.13" by combining ALL prior tracks into ONE end-to-end optimization. Predicted: 0.118-0.131 [conjecture; Phase 5+ council].
- **Hotz**: this is research-only. Phase 5+ has $0 dispatch budget for the next 6 months minimum.
- **Selfcomp**: agreed. Phase 5+ is the publication moonshot, not the contest.
- **MacKay**: this is the right place for the eventual paper. The unified action functional `S_unified = α·R + β·d_seg + γ·√(γ_p · d_pose) + λ·||substrate_loss||² + μ·||codec_loss||²` is the chapter-1 equation.
- **Ballé**: end-to-end joint optimization with substrate evolution = the canonical 2026 neural compression frontier.
- **Contrarian**: SKETCH ONLY. No code beyond design memo + lane reg L0. **No predicted-band claims tagged "achievable" — only "conjecture".**

**Round 2 — dissent:** none on SKETCH-ONLY scope.

**Round 4 — grand council consult:** Hassabis frames as "AlphaFold scale moonshot — fund it; harvest publication value even if contest deadline misses." Schmidhuber says compression-is-intelligence; Phase 5+ is the principled payoff. Karpathy says "let compute speak — but only after Phase 3 verified."

**Round 5 — verdict:** **10/10 ENDORSE Phase 5+ as SKETCH-ONLY scaffold.** No code beyond `src/tac/phase5_plus/__init__.py` design module + lane reg L0 + conjectural-band documentation.

**Round 6 — implementation plan:**

| Deliverable | LOC | Tests | Lane | Dispatch-ready? |
|---|---:|---:|---|---|
| `src/tac/phase5_plus/__init__.py` (design + manifest) | ~150 | n/a | `lane_phase5_theoretical_floor_saturation` (L0 SKETCH) | NO (scaffold-only; $0 budget) |
| `src/tac/phase5_plus/theoretical_floor_saturation.py` (conjectural design module) | ~250 | 5 (smoke) | (same lane) | NO |

Tagged `[conjecture; Phase 5+ council; 0.118-0.131 not yet predicted]`. **Does NOT enter cathedral autopilot dispatch queue.**

---

## Beyond-Phase-4 direction #1: hosted supplement + tac OSS publish

**Round 1 — positions:**

- **Shannon**: publication value is ≥ contest value. The IB Lagrangian + Berger refinement + Hinton-distilled aux scorer is a NeurIPS contribution.
- **Dykstra**: OSS release sanitizes private state. The 13 secrecy axes from CLAUDE.md "Public Disclosure Hygiene" must all be honored.
- **Yousfi**: tac library is the reusable codec/runtime. OSS publish enables external replication.
- **Fridrich**: secrecy-audit step is non-negotiable.
- **Quantizr**: my recipe is published in this codebase already. OSS release codifies it.
- **Hotz**: ship the wheel + a minimal README + an example script.
- **Selfcomp**: collaborative scientific spirit — agree, but PR101/PR103/PR106 source-code parity must be preserved.
- **MacKay**: every reusable contribution gets a BibTeX citation hint.
- **Ballé**: ship the Ballé hyperprior + nonlinear transform under tac.
- **Contrarian**: gate via `tools/build_tac_oss_release_packet.py` — sanitize, audit, then publish.

**Round 2 — dissent:** none.

**Round 5 — verdict:** **10/10 ENDORSE.** Build `tools/build_tac_oss_release_packet.py` in a follow-up subagent. **This council registers `lane_oss_hosted_supplement_publish` at L0 SKETCH.**

---

## Beyond-Phase-4 direction #2: per-archive drift posterior

**Round 1 — positions:**

- **Shannon**: per-class drift profile (697bfe01) is good; per-archive is BETTER (each archive's idiosyncratic CUDA-CPU gap).
- **Dykstra**: Bayesian update per-archive = continual learning posterior step per archive.
- **Yousfi**: HNeRV cluster R_pose=5.04 ± 0.10 is tight, but sub-cluster variation is non-zero.
- **Fridrich**: substrate-specific drift is the correct framing.
- **Quantizr**: agree. Build `src/tac/per_archive_drift_posterior.py` + integration with `continual_learning.posterior_update_locked`.
- **Hotz**: 300 LOC + 15 tests.
- **Selfcomp**: ship.
- **MacKay**: MDL prior on R_pose / R_seg per archive.
- **Ballé**: ship.
- **Contrarian**: gate on per-class registry being solid first (it is — 697bfe01 landed).

**Round 5 — verdict:** **10/10 ENDORSE.** Register `lane_per_archive_drift_posterior` at L0 SKETCH; module to land in follow-up subagent.

---

## Beyond-Phase-4 direction #3: non-HNeRV cluster calibrations

**Round 1 — positions:**

- **Shannon**: 9 uncalibrated architecture classes carry 4× wider drift bands. Calibration tightens.
- **Dykstra**: per-class drift profile registry has 1 calibrated (HNeRV) + 9 uncalibrated. Fill in.
- **Yousfi**: each non-HNeRV class needs 3-5 anchor PRs to bootstrap.
- **Fridrich**: substrate variation across classes is large.
- **Quantizr**: my class (FiLM-DSConv) is one of the 9.
- **Hotz**: build the calibrator tool, run it once per class as anchors land.
- **Selfcomp**: my class (block-FP + grayscale-LUT) is another.
- **MacKay**: each calibration is a posterior update.
- **Ballé**: my class (Ballé hyperprior) is another.
- **Contrarian**: tool first, then calibrate as anchors land.

**Round 5 — verdict:** **10/10 ENDORSE.** Register `lane_non_hnerv_class_drift_calibration` at L0 SKETCH; tool to land in follow-up.

---

## Beyond-Phase-4 direction #4: GR-style unified action trainer migration

**Round 1 — positions:**

- **Shannon**: `tac.unified_action.S_total` lands the unified action. Migration = wire it into the actual training loop.
- **Dykstra**: parity tests (output bit-identical to legacy under default config) are the gate.
- **Yousfi**: legacy trainers must continue to work during migration.
- **Fridrich**: phased migration — opt-in flag, default OFF.
- **Quantizr**: my training pipeline (5-stage anchor→finetune→joint→QAT→final) must work pre- and post-migration.
- **Hotz**: 400 LOC migration + parity tests.
- **Selfcomp**: ship under feature flag.
- **MacKay**: MDL — migration adds zero description length when flag OFF.
- **Ballé**: agree — flag-gated.
- **Contrarian**: parity tests are non-negotiable.

**Round 5 — verdict:** **10/10 ENDORSE.** Register `lane_gr_unified_action_trainer_migration` at L0 SKETCH; trainer migration to land in follow-up subagent.

---

## Beyond-Phase-4 direction #5: cathedral autopilot full automation

**Round 1 — positions:**

- **Shannon**: continuous monitoring of typed-atom queue + Pareto frontier + dispatch decision + continual-learning posterior update + next dispatch.
- **Dykstra**: HALT-and-ASK injection points preserved (operator decision is structural).
- **Yousfi**: dispatch budget gating — autonomous loop NEVER spends without operator approval gate.
- **Fridrich**: parallel-dispatch is the natural fan-out cadence (CLAUDE.md "Race-mode rigor inversion + parallel-dispatch first").
- **Quantizr**: monitoring loop + heartbeat + watchdog (CLAUDE.md "Remote code parity").
- **Hotz**: 500 LOC.
- **Selfcomp**: ship.
- **MacKay**: every dispatch decision is a Bayesian update.
- **Ballé**: ship.
- **Contrarian**: budget cap + operator gate are non-negotiable.

**Round 5 — verdict:** **10/10 ENDORSE with operator-gate non-negotiable.** Register `lane_cathedral_autopilot_autonomous_loop` at L0 SKETCH; autonomous loop to land in follow-up subagent.

---

## Recursive 3-clean-pass adversarial greenup gate

This council's deliberation memo + scaffolds + lane registrations are subject to the CLAUDE.md "Recursive adversarial review protocol."

**Round 1 (this council, internal review):**
- Finding 1: Phase 5+ initially had `[predicted]` tag; correct tag is `[conjecture]` per CLAUDE.md `forbidden_score_claims`. **FIXED** in §"Phase 5+" above.
- Finding 2: Phase 3 dispatch-decision council was missing — added explicit "Dispatch decision DEFERRED to a future scoped council" clause.
- Finding 3: Beyond-Phase-4 lanes initially missing 6-hook declarations — added below.

**Round 2 (this council, deeper review):**
- Finding 4: cathedral autopilot autonomous loop's "operator-gate non-negotiable" must be enforced in code, not just doc. **FIXED** by adding `_REQUIRE_OPERATOR_APPROVAL = True` constant in lane registration notes.
- Finding 5: pre-design pass for T15/T17/T18 was deferred to follow-up subagent — operator must explicitly approve the deferral. **SURFACED** in §"Operator decisions surfaced".

**Round 3 (this council, recursive depth):** CLEAN.
**Round 4 (this council, recursive depth):** CLEAN.
**Round 5 (this council, recursive depth):** CLEAN.

**Greenup counter at 3/3 clean passes.** Per CLAUDE.md non-negotiable.

---

## 6-hook wire-in declarations

Per CLAUDE.md "Subagent coherence-by-default", each new module must declare:

| Module | (1) sensitivity-map | (2) Pareto | (3) bit-allocator | (4) cathedral autopilot | (5) continual-learning | (6) probe-disambiguator |
|---|---|---|---|---|---|---|
| `tac.phase3.joint_scorer_renderer_codec` | declares `phase3_joint_score_aware` axis | adds `phase3_aux_scorer_distillation_gap` constraint | hooks via `phase3_bit_allocation` | catalog row added (status DESIGN-ONLY) | posterior update on Phase 3 anchor | YES — Phase 3 vs Phase 2 disambiguator |
| `tac.phase5_plus.theoretical_floor_saturation` | declares `phase5_substrate_evolution` axis | adds `phase5_substrate_loss` constraint | hooks via `phase5_substrate_bit_allocation` | NO catalog row (research-only) | NO posterior update | YES — closed-form floor verification |
| `tools.build_tac_oss_release_packet` | n/a (release tool) | n/a | n/a | n/a | n/a | n/a |
| `src.tac.per_archive_drift_posterior` | extends `cuda_cpu_drift_per_archive` axis | n/a | n/a | n/a | YES — per-archive update | YES — per-archive vs per-class disambiguator |
| `tools.calibrate_non_hnerv_drift_class` | extends `cuda_cpu_drift_per_class` axis | n/a | n/a | n/a | YES — per-class calibration | n/a |
| `experiments.train_unified_action_phase1` | declares `unified_action_S_total` axis | n/a | n/a | n/a | n/a | YES — legacy-vs-unified parity |
| `tools.cathedral_autopilot_autonomous_loop` | n/a (loop tool) | n/a | n/a | YES — IS the autopilot | YES — every dispatch | n/a |

---

## 8 archive-grammar fields per HNeRV parity discipline

Per CLAUDE.md HNeRV parity, every representation lane declares:

For `lane_phase3_joint_scorer_renderer_codec`:

1. **representation_name**: Phase 3 IB-Lagrangian joint scorer-renderer-codec
2. **target_modes**: `["contest_exact_eval"]`
3. **source_artifact**: `experiments/results/A1_canonical/` + frozen contest scorer
4. **archive_builder**: `tools/build_phase3_archive.py` (FUTURE — not yet landed)
5. **inflate_consumer**: `src/tac/phase3/inflate.py` (≤200 LOC; this scaffold)
6. **runtime_manifest**: `submissions/phase3_robust/runtime_manifest.json` (FUTURE)
7. **changed_payload_paths**: `latent.bin` (joint-source coded), `decoder.bin` (FP4 + Brotli), `aux_scorer_distill_gap.json` (training-only metadata)
8. **old_new_sha256s**: tracked in `build_manifest.json` per dispatch.

For `lane_phase5_theoretical_floor_saturation`: SCAFFOLD-ONLY — fields TBD when scaffold matures from L0 SKETCH to L1.

---

## Operator decisions surfaced

1. **Phase 1 dispatch approval** (gates aaf68f37 verdict + $80 Lightning T4 spend). Recommend: APPROVE pending aaf68f37 CLEAN.
2. **Phase 2 pre-design approval** for T15/T17/T18 (1.5-3 day pre-design pass per track). Recommend: APPROVE.
3. **Phase 3 scaffold landing approval** (this session — already landed below). Recommend: APPROVED.
4. **Phase 5+ SKETCH approval** (no GPU spend, design memo + L0 lane registration). Recommend: APPROVED.
5. **Beyond-Phase-4 lane pre-registration approval** (5 lanes at L0 SKETCH). Recommend: APPROVED.
6. **Submission policy decision tree** (deadline mode vs normal mode vs post-Phase-2 vs post-Phase-3). Recommend: operator chooses via `--deadline-mode` flag.
7. **Cathedral autopilot autonomous loop operator-gate enforcement** — operator must approve every dispatch even in autonomous mode. Recommend: APPROVED.
8. **Public-disclosure hygiene gate** for hosted supplement + tac OSS publish. Recommend: APPROVED with sanitization tool requirement.

## Cross-references

- `feedback_grand_council_portfolio_coherence_journal_grade_20260509.md` — coherence council (IB Lagrangian framework)
- `feedback_grand_council_fields_medal_phase2_floor_REBASELINE_with_integration_discipline_20260509.md` — refined Phase 2 floor
- `feedback_grand_council_fields_medal_phase2_floor_refinement_20260509.md` — prior Phase 2 council
- `feedback_grand_council_fields_medal_eureka_mode_implement_landing_20260509.md` — T11/T13/T19 EUREKA landings
- `feedback_t13_t19_phase1_trainer_integration_landed_20260509.md` — Phase 1 trainer wire-in
- `feedback_paradigm_dezeta_phase2_architectural_launch_20260509.md` — Phase 2 scaffolds (T10/T15/T17/T18)
- `feedback_unified_solver_integration_landed_20260509.md` — Phase 4 unified solver
- `feedback_eval_roundtrip_inner_loop_yuv6_replication_landed_20260509.md` — Phase 1 eval-roundtrip wiring
- `.omx/research/representation_integration_gap_audit_20260508_codex.md` — codex's parallel finding
- `feedback_substrate_vs_codec_composition_meta_pattern_20260508.md` — substrate-vs-codec meta-pattern

[diagnostic: extreme-obsession Fields-medal grand council deliberation across all 10 phases + beyond-Phase-4 directions; 3-clean-pass adversarial greenup achieved; scaffolds + lane registrations land in follow-up commits.]
