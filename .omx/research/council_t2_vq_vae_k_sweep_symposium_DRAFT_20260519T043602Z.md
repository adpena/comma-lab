---
council_tier: T2
council_attendees: [Shannon, Dykstra, Yousfi, Fridrich, Contrarian, Assumption-Adversary, vanDenOord, MacKay, Boyd, Tao, Selfcomp]
council_quorum_met: false
council_verdict: DRAFT_PENDING_CONVOCATION
council_dissent: []
council_assumption_adversary_verdict: []
council_decisions_recorded:
  - "DRAFT: per-substrate symposium memo prepared per Catalog #325 6-step contract; awaits operator convocation OR inner-quintet ratification"
council_predicted_mission_contribution: rigor_overhead
council_override_invoked: false
council_override_rationale: null
deferred_substrate_id: vq_vae
deferred_substrate_retrospective_due_utc: "2026-06-19T04:36:02Z"
predicted_mission_contribution: rigor_overhead
finding_action_class: research
finding_followup_dispatch_envelope_usd: 2.40
finding_canonical_path: per_substrate_symposium_draft
---

# VQ-VAE K-sweep per-substrate symposium DRAFT (E.7)

**Status**: DRAFT — NOT CONVENED. Awaits operator approval to convene the full T2 symposium OR ratification by inner-quintet pact (Shannon + Dykstra + Yousfi + Fridrich + Contrarian + Assumption-Adversary).

**Substrate**: `vq_vae` (van den Oord persistent-codebook substrate)
**Variant**: K-sweep diagnostic smoke at lambda=1.0 across K in {2, 4, 8, 16, 32, 64, 128, 256}
**Dispatch envelope**: $2.40 Modal T4 (8 K-values * $0.30) per `substrate_vq_vae_k_sweep_modal_t4_dispatch.yaml`
**Predecessor council**: T3 Finding 1 PROCEED_WITH_REVISIONS op-routable #1 (`.omx/research/council_t3_finding_1_vq_codebook_anti_pareto_20260518.md`)
**Catalog #325 6-step contract compliance**: declared below

---

## Canonical 6-step contract per Catalog #325

### Step 1: Cargo-cult audit per Catalog #303

#### Cargo-cult assumption #1: "K=2 is the substrate-VQ Pareto-optimum at lambda=1.0"

**Source**: Wave 2A `8b987215a` row #2+#3 analytical R-D solve.
**Hard-earned-vs-cargo-culted**: **CARGO-CULTED** per T3 Finding 1 Assumption-Adversary verdict.
**Rationale**: The analytical R-D bound assumes IID-symbol entropy with squared-error distortion. Substrate-VQ operates on score-aware loss in BTCHW residual space where distortion is the contest scorer's PoseNet+SegNet gradient response, NOT squared error. The pole-in-Pareto-frontier finding (K=64 + K=256 are dominated) may be REAL but the K=2 OPTIMAL claim requires per-substrate empirical validation.
**Unwind-test plan**: This K-sweep dispatch IS the unwind test. Empirical Pareto-frontier at substrate-VQ regime; falsify or confirm K=2 optimum.

#### Cargo-cult assumption #2: "K=64 + K=256 are ANTI-PARETO and should be retired across 14 substrates"

**Source**: Wave 2A analytical conclusion.
**Hard-earned-vs-cargo-culted**: **CARGO-CULTED-EMPIRICALLY-FALSIFIABLE**.
**Rationale**: vanDenOord council dissent: "VQ-VAE codebook usage in practice (WaveNet+image generation) shows hugely-utilized large codebooks (K=512+); the R-D bound is achievable only when training signal is rich enough to populate the codebook." K=2 may collapse to mode-averaged outputs at the substrate level. If K=2 underperforms K=64/K=256 empirically, the analytical Pareto-pole IS the cargo-cult; if K=2 wins, the wire-in wave proceeds.
**Unwind-test plan**: Same as #1.

#### Cargo-cult assumption #3: "100-epoch smoke is sufficient to determine K-optimum"

**Source**: Predecessor's $2.40 envelope assumption.
**Hard-earned-vs-cargo-culted**: **CARGO-CULTED-PENDING-EMPIRICAL**.
**Rationale**: vanDenOord 2017 paper trains VQ-VAE for 1000s of epochs to populate large K. At 100ep, K=256 may show underutilization NOT because the architecture is dominated but because training has not converged. The smoke is a DIAGNOSTIC anchor for the SHAPE of the K-vs-score curve, not a definitive K-optimum verdict.
**Unwind-test plan**: If K=256 shows clearly-dominated pattern even at 100ep (e.g. score worse than K=16 by significant margin), the diagnostic is informative even pre-convergence. If K=256 is near-tied with K=2/K=16, follow-on 1000ep dispatch needed.

#### Cargo-cult assumption #4: "lambda=1.0 is the canonical lambda for the K-sweep"

**Source**: Wave 2A row #2+#3 reported R-D pole specifically at lambda=1.0.
**Hard-earned-vs-cargo-culted**: **HARD-EARNED-FROM-WAVE-2A**.
**Rationale**: Wave 2A is the canonical analytical-R-D-solve artifact; lambda=1.0 is the operating point the analytical claim was made at. Sweeping lambda is a separate follow-on lane (not E.7 scope).
**Unwind-test plan**: N/A; this assumption is the K-sweep's design boundary.

---

### Step 2: 9-dimension success checklist evidence per Catalog #294

## 9-dimension success checklist evidence

1. **UNIQUENESS**: K-sweep dispatch is a DIAGNOSTIC for an analytical hypothesis NOT a new substrate. The VQ-VAE substrate itself is canonical (van den Oord 2017). UNIQUE aspect: empirical-anchor on the substrate-VQ regime at lambda=1.0.
2. **BEAUTY + ELEGANCE**: 8 K-values, 100ep each, identical config except K. 8 paired-comparison anchors emit per-K (rate, distortion, score) triples for direct Pareto-frontier plotting.
3. **DISTINCTNESS**: Distinct from sister K-sweeps on D4 or sane_hnerv substrates (which would be follow-on lanes). The VQ-VAE substrate-of-canonical-design IS the right test surface.
4. **RIGOR**: Catalog #229 premise verification confirmed `--codebook-size` flag EXISTS (line 310, default=512); predecessor's audit FALSIFIED on this point. Catalog #324 `predicted_band_validation_status: pending_post_training` declared. Catalog #325 6-step contract honored in this memo.
5. **OPTIMIZATION PER TECHNIQUE**: VQ-VAE substrate (covered by Catalog #290 canonical-vs-unique decision in substrate's own design memo); this dispatch inherits.
6. **STACK-OF-STACKS-COMPOSABILITY**: K-optimum derived here feeds the 14-substrate VQ wire-in wave per T3 Finding 1 op-routable #3. Each substrate's K-decision composes additively (per-substrate codebook is local).
7. **DETERMINISTIC REPRODUCIBILITY**: Seed-pinned per `_pin_seeds(args.seed)` in trainer; identical config across 8 K dispatches except `VQ_VAE_CODEBOOK_SIZE` env var.
8. **EXTREME OPTIMIZATION + PERFORMANCE**: T4 100ep at K<=256 fits within $0.30/dispatch budget; total $2.40 for 8 dispatches.
9. **OPTIMAL MINIMAL CONTEST SCORE**: K-optimum feeds the substrate's full 2000ep production dispatch which targets sub-0.193 [contest-CUDA] per the canonical VQ-VAE recipe.

---

### Step 3: Observability surface declaration per Catalog #305

## Observability surface

1. **Inspectable per layer**: VQ-VAE substrate exposes encoder + codebook + decoder activation hooks via `tac.substrates.vq_vae.architecture.VqVaeSubstrate.forward()`; intermediate quantization-state tensors capturable.
2. **Decomposable per signal**: Score-aware loss decomposes into (rate, seg_distortion, pose_distortion, commitment) per `tac.substrates.vq_vae.score_aware_loss.VqVaeScoreAwareLoss`; per-K dispatch records all 4 components.
3. **Diff-able across runs**: Identical seed + config across K runs; per-K diff = K only. Byte-level archive diff per K dispatchable via `tools/verify_distinguishing_feature_byte_mutation.py`.
4. **Queryable post-hoc**: All 8 K dispatches emit canonical Modal call_id ledger rows per Catalog #245; harvested-artifact JSON includes per-K final score + components.
5. **Cite-able**: Anchor = (substrate=vq_vae, K=<K>, lambda=1.0, commit=<git_HEAD>, call_id=<modal_call_id>, config_seed=<seed>); 8 anchors emit to `tac.master_gradient_anchors.append_anchor` per Catalog #245 4-layer pattern.
6. **Counterfactual-able**: Byte-mutation discipline (Catalog #139): per-K archive bytes mutate codebook index by 1; downstream auth-eval delta verifies the K bit is actually consumed at inflate time.

---

### Step 4: Sextet pact deliberation (DRAFT — pending convocation)

#### Shannon LEAD (operating-within: R-D bound IS the canonical floor)
[DRAFT POSITION]: At lambda=1.0, the R-D bound R(D) = R*(D) is achievable only at the substrate-VQ regime if score-aware-loss-distortion approximates squared-error. The K-sweep empirically measures the gap between IID-R-D-prediction and substrate-VQ-realized; if K=2 wins, IID-R-D is a faithful proxy; if K=64/K=256 win, the substrate-VQ regime is dominated by training-signal-richness NOT R-D theoretic optimum. Either outcome IS canonical signal.

#### Dykstra CO-LEAD (operating-within: alternating-projections Pareto-feasibility)
[DRAFT POSITION]: The Pareto-frontier intersection of (rate-feasibility, distortion-feasibility) is the substrate K-feasibility set. Wave 2A's analytical R-D-pole IS the Pareto-feasibility-vertex under IID assumptions; substrate empirical may exhibit DIFFERENT Pareto-vertex due to NON-IID training-signal regime. Diagnostic resolves the IID-vs-substrate-regime ambiguity.

#### Yousfi (operating-within: contest-scorer-as-distortion)
[DRAFT POSITION]: SegNet stride-2 stem aliases below 256x192; small-K VQ codebooks may be invisible to SegNet (substrate redundancy filtered by scorer). K=2 may win because BOTH K=2 and K=256 produce same-class SegNet output; only rate differs. Empirical anchor needed.

#### Fridrich (operating-within: inverse-steganalysis on codebook indices)
[DRAFT POSITION]: VQ codebook indices are discrete tokens; inverse-steganalysis lens suggests small-K (K=2 = bit-stream) is maximally compressible via STC/UNIWARD-style entropy coding. Large-K (K=256 = byte-stream) competes with entropy coder overhead. K-sweep IS the rate-axis Pareto.

#### Contrarian (operating-within: training-signal-poverty masking)
[DRAFT POSITION] (echoes T3 Finding 1 dissent): If 100-epoch smoke is below convergence threshold for K=256, the "K=256 anti-Pareto" finding is a CARGO-CULTED interpretation of training-time-poverty NOT architectural-dominance. 1000ep follow-on dispatch needed before retirement-decision on K=64/K=256.

#### Assumption-Adversary (operating-within: HARD-EARNED-vs-CARGO-CULTED classification)
[DRAFT POSITION]:
- IID-R-D-pole-at-K=2: **CARGO-CULTED** (already classified at T3 Finding 1)
- 100ep-smoke-sufficient-for-K-optimum: **CARGO-CULTED-PENDING-EMPIRICAL**
- Per-substrate-wire-in-yields-composite-[-0.070,-0.014]: **HARD-EARNED-FROM-CATALOG-#233**
- Lambda=1.0-IS-canonical: **HARD-EARNED-FROM-WAVE-2A**

#### vanDenOord (operating-within: canonical-VQ-substrate author)
[DRAFT POSITION] (echoes T3 Finding 1 dissent): WaveNet+image-generation shows K=512+ populated; K=2 collapse risk is REAL. Recommend 1000-epoch follow-on if 100-epoch smoke shows large-K underutilization without clear collapse evidence.

#### MacKay (operating-within: MDL + Bayesian inference)
[DRAFT POSITION]: MDL favors smaller K when codebook-entropy + index-stream dominates; favors larger K when training-signal richness dominates. The K-sweep IS the MDL diagnostic; per-K (codebook-bits + index-stream-entropy) MDL is queryable post-hoc.

#### Boyd (operating-within: convex-feasibility via alternating projections)
[DRAFT POSITION]: Per-K (rate, distortion) is a convex-feasibility set; Pareto-frontier IS the lower-left boundary. K-sweep maps this boundary at 8 K-values; the K-optimum is the convex-hull vertex. Open-vs-closed boundary at K=2 / K=256 matters: Boyd's BOYD-2 R2 LOW catch (Catalog #239) is relevant for the cost-class boundary if K=2 dispatches to long_burn (it does not — all K-sweep dispatches are `smoke` class).

#### Tao (operating-within: harmonic-analysis on discrete signal regime)
[DRAFT POSITION]: K-stream is a finite-alphabet discrete signal; finite-alphabet R-D theory IS Shannon's canonical regime. The score-aware-loss distortion measure may shift the R-D pole but does NOT change the Shannon-axis interpretability. K-sweep is harmonic-analytically clean.

#### Selfcomp (operating-within: byte-level archive grammar)
[DRAFT POSITION]: K=2 = 1 bit/cell = 768 B/video at grid 48x64; K=256 = 8 bit/cell = 6144 B/video. Plus codebook overhead (32 B vs 4096 B). Per-K archive bytes are computable in advance; rate-axis Pareto is analytically derivable.

---

### Step 5: Per-substrate reactivation criteria pinned per CLAUDE.md "Forbidden premature KILL"

**Reactivation paths (priority order)**:

1. **OPERATOR FRONTIER OVERRIDE** (Catalog #300 Mission Alignment): Operator-verbatim quote in `council_override_rationale` frontmatter authorizes immediate K-sweep dispatch bypassing 14-day symposium requirement. Cost: $2.40. Predicted ΔS impact: empirical confirmation OR falsification of Wave 2A R-D pole.

2. **INNER-QUINTET RATIFICATION** (~30min): 5-of-6 inner-quintet pact (Shannon + Dykstra + Yousfi + Fridrich + Contrarian + Assumption-Adversary) ratify this DRAFT via comment in the memo body + emit canonical posterior anchor via `tac.council_continual_learning.append_council_anchor`. Cost: $0 deliberation time.

3. **FULL T2 SYMPOSIUM CONVOCATION** (~2h): All 11 listed attendees (including vanDenOord + MacKay + Boyd + Tao + Selfcomp grand-council seats) deliberate full per Catalog #292 explicit-assumption-statement discipline. Output: PROCEED / PROCEED_WITH_REVISIONS / DEFER / REFUSE verdict + posterior anchor. Cost: $0 deliberation time.

4. **DEFER UNTIL SISTER PROBE** (multi-week): Wait for sister D4 or sane_hnerv K-sweep to land first; ratify this VQ-VAE-substrate K-sweep AFTER sister-substrate K-sweep verdict establishes the substrate-K-Pareto pattern.

---

### Step 6: Catalog #324 post-training Tier-C validation discipline

**Predicted_band_validation_status**: `pending_post_training` (declared in recipe).
**Reactivation criterion**: Post-training Tier-C density re-measurement via `tools/mdl_scorer_conditional_ablation.py --tier c` on each landed K-dispatch archive AFTER smoke completion. The analytical Wave 2A predicted_band [0.180, 0.300] is derived from IID-R-D under canonical assumptions; substrate-empirical Tier-C density may shift the band.

---

## Sister coordination per Catalog #230 ownership map

- Sister 1 (`phase_b_mps_gap_experiment_infrastructure_build_20260518`): owns NEW `src/tac/mps_gap_experiment/` namespace; disjoint scope.
- Sister 3 (`phantom_api_backfill_wave_1_20260518`): owns ~20 EXISTING `.omx/research/*.md` memos for phantom-API backfill; disjoint from this DRAFT memo (new file, not phantom-API-flagged).
- This DRAFT memo: only touches NEW path (`.omx/research/council_t2_vq_vae_k_sweep_symposium_DRAFT_20260519T043602Z.md`) + new variant recipe + memory entry.

---

## Catalog #229 premise verification log

- PV-0: Canonical helpers verified (tac.substrates.vq_vae / tac.scorer / tac.deploy.modal.call_id_ledger / tac.probe_outcomes_ledger / tac.council_continual_learning) — all importable.
- PV-1: **PREMISE FALSIFICATION**: Predecessor audit (`.omx/research/3_smoke_paid_gpu_reactivation_wave_blocker_audit_20260519T042301Z.md`) PV-4 claimed "VQ-VAE trainer hardcodes codebook_size=16; no --codebook-size CLI flag" — FALSIFIED. The `--codebook-size` flag EXISTS at `experiments/train_substrate_vq_vae.py:310` with default=512. Predecessor's `grep "add_argument.*codebook"` regex missed it because argparse spans 2 lines. The `codebook_size=16` at line 730 is in `_smoke_main` (smoke-hardcoded) NOT `_full_main` (which reads `args.codebook_size`).
- PV-2: VQ-VAE recipe `substrate_vq_vae_modal_a100_dispatch.yaml` confirmed as the production 2000ep dispatch (not K-sweep variant); separate variant recipe is correct approach.
- PV-3: T3 Finding 1 council memo verified at `.omx/research/council_t3_finding_1_vq_codebook_anti_pareto_20260518.md`.
- PV-4: Sister subagents in flight understood via `.omx/state/subagent_progress.jsonl`.

---

## 6-hook wire-in declaration per Catalog #125

1. **Sensitivity-map contribution**: ACTIVE — per-K (rate, distortion) anchors feed `tac.sensitivity_map.*` for substrate-K-sensitivity ranking
2. **Pareto constraint**: ACTIVE — empirical Pareto-frontier IS the Pareto-feasibility constraint for substrate-VQ wire-in wave
3. **Bit-allocator hook**: ACTIVE — per-K bit allocation (log2(K)) feeds substrate-VQ bit-allocator design
4. **Cathedral autopilot dispatch hook**: ACTIVE — K-sweep verdict feeds autopilot's substrate-K-canvas decision per T3 Finding 1 op-routable #3 (14-substrate wire-in wave)
5. **Continual-learning posterior update**: ACTIVE — symposium ratification anchor + per-K dispatch anchor both via `tac.council_continual_learning.append_council_anchor`
6. **Probe-disambiguator**: ACTIVE — this symposium IS the K=2-vs-K=256-vs-intermediate disambiguator per T3 Finding 1 op-routable #1

---

## Cross-references

- T3 Finding 1 council memo: `.omx/research/council_t3_finding_1_vq_codebook_anti_pareto_20260518.md`
- Variant recipe: `.omx/operator_authorize_recipes/substrate_vq_vae_k_sweep_modal_t4_dispatch.yaml`
- Production recipe (sister, NOT this variant): `.omx/operator_authorize_recipes/substrate_vq_vae_modal_a100_dispatch.yaml`
- VQ-VAE trainer: `experiments/train_substrate_vq_vae.py` (--codebook-size at line 310)
- Predecessor blocker audit: `.omx/research/3_smoke_paid_gpu_reactivation_wave_blocker_audit_20260519T042301Z.md`
- Wave 2A analytical Pareto solve: `.omx/state/8b987215a` rows #2+#3
- CLAUDE.md non-negotiables: Catalog #313 / #324 / #325 / #270 / #167 / #294 / #303 / #305 / #296 / #229 / #220 / #272 / #292 / #300


<!-- # FORMALIZATION_PENDING:pre_framework_memo_dated_2026-05-19_predates_canonical_equations_birthday_registry_population_in_progress_appended_by_strict_flip_enablers_per_operator_blanket_approval_per_claude_md_forbidden_premature_kill_without_research_exhaustion_this_is_DEFER_pending_canonical_equation_backfill_NOT_kill -->
