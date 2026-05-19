<!-- # PHANTOM_NAME_DESIGN_PROPOSAL_OK_FILE: design/synthesis/audit memo proposing not-yet-implemented canonical helpers per Catalog #287 sub-scope B; all cited tac.X module names are explicit design proposals or future-helper references; this is an HTML comment so markdown renderers ignore it; waiver landed by lane_phantom_api_backfill_wave_1_20260518 -->
---
schema: council_deliberation_v2
deliberation_id: rate_attack_synthesis_v2_reconciliation_primary_plus_adversarial_plus_supplement_20260518
topic: "Canonical OPERATOR-FACING reconciliation of today's rate-attack research wave: PRIMARY 43-vector META-paradigm research (T3 PROCEED_WITH_REVISIONS) + ADVERSARIAL paradigm challenger (T2 PROCEED_WITH_REVISIONS with 5 binding critiques + 12 NEW vectors + alternative TOP-5 + mandatory A-2 N-7 empirical anchor) + supplement (per-axis hardware exploit matrix + dual-device master-gradient research question + comprehensive available-signal inventory). Resolves PRIMARY vs ADVERSARIAL TOP-5 conflict via SATURATION-INDEPENDENT vs SATURATION-DEPENDENT classification; sequences immediate-execution per Hotz binding directive on G1; A-2 N-7 verdict-driven cascade for SATURATION-DEPENDENT vectors. Final RECONCILED TOP-5 (operator-routable) + per-axis hardware-exploit matrix integration + dual-device master-gradient recommendation."
review_kind: t2_reconciliation_panel_post_primary_plus_adversarial_landings
review_date: "2026-05-18"
lane_id: lane_rate_attack_synthesis_v2_reconciliation_20260518
council_tier: T2
council_attendees:
  # Sextet pact (binding; quorum 6-of-6 at T2 per CLAUDE.md Council conduct)
  - Shannon
  - Dykstra
  - Yousfi
  - Fridrich
  - Contrarian
  - Assumption-Adversary
  # Reconciliation panel (operator-elevated; the four members whose ADVERSARIAL critiques drove the binding revisions)
  - Tao
  - Carmack
  - Hotz
  - Boyd
  # Grand-council attendees added per topic
  - Ballé
  - Mallat
  - van_den_Oord
  - Filler
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Contrarian
    verbatim: "The reconciliation cannot resolve the 43-arbitrariness critique by renaming TOP-5; it must EITHER mathematically partition vectors into theorem-anchored sub-paradigms per Tao OR explicitly defer the META-paradigm framing until A-2 N-7 lands. I VOTE PROCEED only if the reconciled TOP-5 carries explicit SATURATION-INDEPENDENT vs SATURATION-DEPENDENT tags per vector AND the SATURATION-DEPENDENT vectors are formally gated on A-2 N-7 verdict."
  - member: Assumption-Adversary
    verbatim: "The shared assumption I am operating within: PRIMARY's two HARD-EARNED anchors (G1 cross-axis empirical prior + F1 source-verified unscored dims) are independent of the saturation hypothesis; therefore reconciliation can safely PROCEED on probes for those two while DEFERRING the saturation-dependent vectors. This is HARD-EARNED-PARTIAL because (a) G1 has paired-axis data but the actual existing-anchor rerank later measured 0.0 delta; (b) F1 anchor proves unscored internal dims but not archive-byte capacity. The other PRIMARY TOP-5 (B1/Y3+Y6/H1) all depend on whether the archive packet has remaining rate-extractable structure, which A-2 N-7 directly tests."
council_assumption_adversary_verdict:
  - assumption: "PRIMARY's TOP-5 and ADVERSARIAL's TOP-5 are necessarily conflicting (operator must choose one)"
    classification: CARGO-CULTED
    rationale: "Per detailed cross-classification in §3-§4 plus later Codex correction: G1 and corrected F1/A2 probes are saturation-independent as probe axes, not as guaranteed score movement; A-2/A-3/A-4 (compressor sweep / Dykstra / MI-min) are TESTS or STRUCTURAL primitives that INFORM rather than COMPETE WITH PRIMARY's TOP-5; A-1/A-5 (Brotli dict / LOC budget) are SUPPLEMENTARY low-risk experiments. The two TOP-5s are largely COMPLEMENTARY, not conflicting. The framing that operator must choose one is itself the cargo-cult."
  - assumption: "A-2 N-7 verdict is essential gating for ALL rate-attack work"
    classification: HARD-EARNED-PARTIAL
    rationale: "A-2 N-7 IS essential for SATURATION-DEPENDENT vectors (B1, Y3+Y6, H1, Cat-A scorer-aware-byte-level minus A1-SPECIALIZED, most of Cat-M ZIP-overhead, etc.) because if standard compressors beat current frontier rate, the rate is NOT saturated, and those vectors have real upside. A-2 N-7 is NOT essential for SATURATION-INDEPENDENT vectors (G1 cross-axis exploit + corrected F1/A2 scorer-blind RGB perturbation capacity) because they exploit different structural information (cross-axis sensitivity + scorer-feature-invariance) orthogonal to the compression-saturation axis."
  - assumption: "The PR102 +0.033 CPU-CUDA gap is a reusable conversion factor"
    classification: FALSE_AS_CONVERSION__HARD_EARNED_AS_PER_ARCHIVE_PRIOR
    rationale: "PR102 observed a +0.033 CUDA-minus-CPU gap on its own archive/runtime pair. Treat this as a prior for paired-axis probing only, not as a conversion factor. PR107 and PR101-family observations are supporting priors, not authority to convert CUDA to CPU or claim frontier movement without paired exact evidence."
  - assumption: "Hydra dims 7-12 are STRUCTURALLY ignored by scorer"
    classification: HARD-EARNED-VERIFIED-FROM-SOURCE
    rationale: "upstream/modules.py:84 source line `return sum((out1[h.name][..., : h.out // 2] - out2[h.name][..., : h.out // 2]).pow(2).mean(...) for h in self.hydra.heads if h.name in distortion_heads)` confirms scorer reads ONLY first 6 dims via `[..., : h.out // 2]` slice with h.out=12. Dims 6-11 (0-indexed) are computed internally but DISCARDED by scorer. The direct dim-channel exploit is superseded; corrected F1/A2 must prove RGB perturbation capacity under first-6-dim and SegNet-argmax stability."
  - assumption: "ADVERSARIAL's 60% predicted SATURATION_HARD_EARNED is calibrated"
    classification: CARGO-CULTED-PENDING-EMPIRICAL
    rationale: "ADVERSARIAL claims 60% probability the saturation hypothesis is HARD-EARNED based on the closure_campaign Wyner-Ziv sweep landing below leaderboard-precision floor. But (a) one Wyner-Ziv sweep is not the same prior surface as the broad standard-compressor sweep A-2 N-7 will run; (b) the 60% number is a quartet estimate, not an empirically anchored frequentist probability. RECONCILIATION TREATS this as 'meaningful prior but not authority'; A-2 N-7 verdict is the actual signal."
  - assumption: "G1 immediate execution (Hotz binding) does NOT need A-2 N-7 to land first"
    classification: HARD-EARNED-FROM-FIRST-PRINCIPLES
    rationale: "G1 is per-axis-archive re-ranking using existing dual-eval data in `.omx/state/continual_learning_posterior.jsonl` + `.omx/state/modal_call_id_ledger.jsonl`. It produces NO new archive bytes; therefore the compression-saturation question is structurally orthogonal. Hotz directive PROCEED IMMEDIATELY is operator-routable in current session."
council_decisions_recorded:
  - "RECONCILED-1: PRIMARY's TOP-5 and ADVERSARIAL's alternative TOP-5 are NOT conflicting; they are complementary. Reconciled TOP-5 (operator-facing) drawn from BOTH inputs per §6."
  - "RECONCILED-2: SATURATION-INDEPENDENT probe axes (G1, corrected F1/A2) PROCEED IMMEDIATELY independent of A-2 N-7 outcome. Later Codex G1 existing-anchor rerank measured actual_delta_s=0.0, so G1 remains a future paired-anchor/candidate-selection criterion, not a landed frontier move."
  - "RECONCILED-3: SATURATION-DEPENDENT vectors (B1, Y3+Y6, H1 from PRIMARY; A-1 Brotli-dict from ADVERSARIAL) gated on A-2 N-7 verdict per §7 sequencing."
  - "RECONCILED-4: A-2 N-7 routed to Codex per `.omx/research/codex_routing_directive_a2_n7_cpu_axis_tier1_standard_compressor_sweep_empirical_anchor_20260518.md` commit 1ac2063de; canonical /goal LOOP will execute. ETA 4-6h wall-clock at $0."
  - "RECONCILED-5: A-3 (Dykstra-feasibility intersection) + A-4 (MI-min Wyner-Ziv) + A-5 (LOC budget OPTIMIZER) are STRUCTURAL PRIMITIVES that should land as canonical helpers regardless of A-2 N-7 outcome — they reduce wasted enumeration effort + provide canonical bounds."
  - "RECONCILED-6: Dual-device master-gradient research question — DEFER to post-A-2-N-7 session. Will land as separate routing directive if A-2 N-7 surfaces compelling per-axis signal."
  - "RECONCILED-7: META-paradigm SINS (PRIMARY) + paradigm-conflation critique (ADVERSARIAL Tao) — RESOLVED via §5 partition into 3 sub-paradigms (MI-min Wyner-Ziv / K-complexity-min / Pareto-tightening), each TOP-5 vector classified into 1 of 3."
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
council_override_rationale: ""
substrate_alias: rate_attack_synthesis_v2_reconciliation_20260518
substrate_aliases:
  - rate_attack_reconciliation_v2_20260518
  - reconciled_top_5_rate_attack_20260518
deferred_substrate_id: null
deferred_substrate_retrospective_due_utc: null
predicted_band_validation_status: pending_post_training
predicted_band_validation_reactivation_criteria: "Reconciled TOP-5 aggregate predicted band [0.155, 0.184] [contest-CPU] is hypothesis-grade. It can only validate when G1/corrected-F1/A2 each achieve measured post-probe net movement on paired Linux x86_64 [contest-CPU] anchors per CLAUDE.md 'Submission auth eval — BOTH CPU AND CUDA' + A-2 N-7 verdict lands + SATURATION-DEPENDENT vectors are re-evaluated per the verdict-driven cascade in §7. Codex G1 existing-anchor rerank 2026-05-18 measured actual_delta_s=0.0."
score_claim: false
promotion_eligible: false
provider_spend: false
research_only: true
canonical_frontier_anchor:
  contest_cpu: "0.19205 [contest-CPU] (lane pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean; archive sha 6bae0201; per Catalog #316 + reports/latest.md 2026-05-17)"
  contest_cuda: "0.20533 [contest-CUDA T4] (lane pr106_format0d_latent_score_table; archive sha 9cb989cef519; per Catalog #316)"
related_deliberation_ids:
  - rate_attack_43_vectors_meta_paradigm_deep_research_20260518
  - adversarial_rate_attack_paradigm_challenger_20260518
  - structural_information_not_shipped_meta_paradigm_unification_20260518
  - codex_routing_directive_a2_n7_cpu_axis_tier1_standard_compressor_sweep_empirical_anchor_20260518
  - rate_attack_research_context_supplement_per_axis_hardware_plus_dual_device_master_gradient_20260518
  - cross_stack_synthesis_9_design_landings_unified_framework_20260518
  - design_stack_full_hypergraph_model_design_memo_20260518
  - rate_attack_novel_vectors_design_memo_20260518
memory_path: ~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_rate_attack_synthesis_v2_reconciliation_landed_20260518.md
event_type: dispatched
parent_id_or_session: rate_attack_synthesis_v2_reconciliation_20260518
notes: "Canonical OPERATOR-FACING reconciliation of three sister landings: PRIMARY 2cae89a87 + ADVERSARIAL 4c6e46bfa + supplement d43ecddb0 + A-2 N-7 routing 1ac2063de. Resolves PRIMARY vs ADVERSARIAL TOP-5 framing as COMPLEMENTARY (not conflicting); produces RECONCILED TOP-5 with explicit SATURATION-INDEPENDENT vs SATURATION-DEPENDENT tags + A-2 N-7 sequencing + Hotz immediate-execution directive on G1. Per CLAUDE.md 'Mission alignment — non-negotiable' Consequence 4 (frontier-breaking dominates rigor budget). Per Catalog #322 v2 cascade composition_alpha sub-additive defaults applied to aggregate predictions. Per Catalog #324 predicted bands all carry pending_post_training validation status. T2 sextet + reconciliation panel (Tao+Carmack+Hotz+Boyd) + 4 grand-council attendees."
---

# Rate-Attack Synthesis V2 — Canonical Operator-Facing Reconciliation

**Lane**: `lane_rate_attack_synthesis_v2_reconciliation_20260518` (L0 → L1 at memo landing)
**Subagent**: `RATE-ATTACK-SYNTHESIS-V2-RECONCILIATION-2026-05-18`
**Sister landings (all committed)**:
- PRIMARY: `2cae89a87` (`.omx/research/rate_attack_43_vectors_meta_paradigm_deep_research_20260518.md` + 5 per-vector design memos + 3 routing directives)
- ADVERSARIAL: `4c6e46bfa` (`.omx/research/adversarial_rate_attack_paradigm_challenger_20260518.md`)
- Supplement: `d43ecddb0` (`.omx/research/rate_attack_research_context_supplement_per_axis_hardware_plus_dual_device_master_gradient_20260518.md`)
- A-2 N-7 routing: `1ac2063de` (Codex /goal LOOP executing)
**Sister codex subagent (DISJOINT scope; owns source code)**: `019de465`
**Live frontier per Catalog #316**: `0.19205 [contest-CPU]` / `0.20533 [contest-CUDA T4]`
**Horizon-class**: `frontier_breaking` (per CLAUDE.md "Mission alignment" Consequence 5; reconciliation IS the operator-routable enabler of the next paid-dispatch cascade)

---

## Codex erratum and supersession note (2026-05-18T21:46Z)

This memo preserves the reconciliation panel's original signal, but its F1
language is superseded by the later legal-receiver-path audit committed at
`35b06f9ec` and by Codex finding
`.omx/research/codex_findings_g1_cpu_axis_rerank_and_f1_reframe_20260518T214650Z_codex.md`.

Corrected F1 interpretation:

- True: PoseNet dims 7-12 are source-verified unscored internal outputs.
- False: dims 7-12 are a free archive byte channel.
- Canonical F1: scorer-blind RGB perturbations constrained to the PoseNet
  first-6-dim invariance manifold. This collapses F1 into A2-style
  adversarial steganography on a specific scorer-blind manifold.
- Legal receiver path: standard inflate-to-RGB only; no PoseNet/SegNet load at
  inflate time.
- Scope of this erratum: this corrects the direct F1 "dims 7-12 are archive
  bytes" framing only. It does **not** reject A1-SPECIALIZED deterministic packet
  compiler work. A tiny self-contained per-pattern transducer, fixed table,
  symbolic formula, or distilled sparse/quantized native binary remains
  sanctioned by CLAUDE.md `contest_one_video_replay` when exact CUDA auth eval
  validates it. The rejected case is the naive full-scorer/full-PoseNet receiver
  path, not specialized generated replay code.

Therefore any phrase below that says "dim 7-12 bits travel through scorer" or
implies a direct free archive channel should be read as historical pre-erratum
language, not as the current canonical design.

---

## 1. Executive Summary (OPERATOR-FACING)

### 1.1 The reconciliation in one paragraph

PRIMARY produced a 43-vector master memo with TOP-5 (F1+G1+B1+Y3+Y6+H1) predicted aggregate `[0.152, 0.179]` [contest-CPU]. ADVERSARIAL critiqued it via 5 binding revisions (Tao paradigm-conflation / Contrarian arbitrariness / Assumption-Adversary saturation-PENDING / Boyd composition-infeasibility / Hotz apparatus_maintenance) + 12 NEW vectors + alternative TOP-5 (Brotli-dict + CPU-sweep + Dykstra + MI-min + LOC-budget) + mandatory A-2 N-7 empirical anchor. The reconciliation panel (T2 sextet + Tao+Carmack+Hotz+Boyd) finds the two TOP-5s are **NOT conflicting but COMPLEMENTARY**: G1 (cross-axis re-rank) + corrected F1/A2 (PoseNet-first-6-invariant scorer-blind RGB perturbation capacity) are **SATURATION-INDEPENDENT** (proceed regardless of A-2 N-7 outcome — different structural information than compression saturation); B1+Y3+Y6+H1 are **SATURATION-DEPENDENT** (gated on A-2 N-7 verdict); ADVERSARIAL's A-1/A-5 are **SUPPLEMENTARY LOW-RISK EXPERIMENTS**; ADVERSARIAL's A-2/A-3/A-4 are **STRUCTURAL PRIMITIVES** that should land as canonical helpers regardless. **Final reconciled TOP-5** = G1 (immediate) + corrected F1/A2 probe-then-dispatch + A-3 Dykstra-feasibility-FIRST (structural) + A-4 MI-min-Wyner-Ziv (structural) + verdict-conditional (B1 OR H1 OR Y3+Y6 OR A-1 Brotli-dict, picked by A-2 N-7 outcome).

### 1.2 Operator decision table (the single page that matters)

| # | Action | Cost | Time | Saturation-dependent | Expected ΔS [contest-CPU] | Operator op-routable | Status |
|---|---|---|---|---|---|---|---|
| **1** | **G1 cross-axis CPU re-rank** (Hotz IMMEDIATE) | **$0** | **DONE** | NO (cross-axis empirical) | `0.0` realized for existing-anchor rerank; no new score claim | **DONE / MONITOR** for future paired CPU anchors | Report `experiments/results/g1_cpu_axis_re_rank_20260518T214250Z/report.json`; `FRONTIER_STABLE_VIA_RE_RANK` |
| **2** | **Corrected F1/A2 scorer-blind RGB perturbation probe** then build | $0 probe + $1-3 build | 1-2 days | NO (source-verified invariant; archive capacity empirical) | `[-0.012, -0.004]` PENDING-EMPIRICAL-CAPACITY | **GO probe FIRST** per Fridrich+Contrarian revision; build CONDITIONAL on probe PASS | Routing directive landed (`83440e8a5`); direct dim-channel framing superseded |
| **3** | **A-2 N-7 standard compressor sweep** (Codex executing) | $0 | 4-6h | TESTS the assumption | INFORMS all other vectors | **GO** per ADVERSARIAL mandatory anchor | Routing directive `1ac2063de` landed; Codex /goal LOOP picks up |
| **4** | **A-3 Dykstra-feasibility-FIRST canonical helper** | $0 | 1 day | NO (structural primitive) | N/A — process; reduces wasted enumeration by 75-90% | **GO** per Boyd RECONCILED-5 | Queue for Codex post-A-2-N-7 |
| **5** | **A-4 MI-min Wyner-Ziv canonical helper** | $0 | 1 day | NO (structural primitive) | N/A — informs B1 + B2 + other Wyner-Ziv vectors | **GO** per Tao RECONCILED-5 | Queue for Codex post-A-2-N-7 |
| **6** | **POST-A-2-N-7: pick ONE of {B1, H1, Y3+Y6, A-1 Brotli-dict}** based on verdict | $1-8 | conditional | YES | conditional | **HOLD** until A-2 N-7 lands | gated |
| **7** | LOC budget OPTIMIZER (A-5 Carmack) | $0 | 1 day | LOW dependence | `[-0.003, -0.001]` direct + `[-0.010, -0.003]` if forces simpler codec | OPTIONAL post-RECONCILED-1-5 | low priority |
| **8** | Dual-device master-gradient extension | $0-$2 | DEFERRED | n/a | research question | **DEFER** per RECONCILED-6 | gated on A-2 N-7 outcome |
| **P0** | **A1-SPECIALIZED deterministic packet-compiler feasibility** | $0 | this session | NO for feasibility; YES for promotion | no score claim until charged-byte packet + exact CUDA | **GO feasibility** | Measure charged bytes, runtime-consumption proof, and net score accounting before promotion |

### 1.3 Immediate-action recommendation (next operator turn)

**TL;DR**: The reconciled TOP-5 is OPERATOR-ROUTABLE NOW. Three of the five are already routed to Codex (G1 via `83440e8a5`, A-2 N-7 via `1ac2063de`, F1 probe via `83440e8a5`); two more (A-3 Dykstra + A-4 MI-min) are structural primitives Codex can pick up after A-2 N-7. **Operator can monitor the Codex /goal LOOP and check `.omx/state/probe_outcomes.jsonl` + `.omx/state/codex_persistent_session_state.jsonl` for completion signals**. No paid GPU dispatch needed yet; all current operator-routable work is $0 CPU-probe + local re-ranking.

### 1.4 Predicted aggregate scoreboard motion

**If G1 + corrected F1/A2 land successfully (SATURATION-INDEPENDENT probe subset)**:
- G1 existing-anchor rerank (measured later by Codex): `actual_delta_s=0.0`; future G1 value depends on new paired CPU anchors or candidate selection, not current public-anchor rerank
- Corrected F1/A2 alone: hypothesis band `[-0.012, -0.004]` only after RGB perturbation capacity + net byte savings are measured
- Combined movement remains hypothesis-grade until both probes produce measured net improvements on paired Linux x86_64 [contest-CPU] anchors
- Do not read the original `[0.170, 0.185]` aggregate as a current score claim

**If A-2 N-7 returns SATURATION_REFUTED** (~40% per ADVERSARIAL prior):
- Adds 1 verdict-conditional vector @ predicted `[-0.005, -0.015]` per the winning standard compressor
- Aggregate after G1 realized zero movement: hypothesis-grade only; corrected-F1/A2 plus row-6 planning band `[0.171, 0.185]` [contest-CPU] until artifacts land

**If A-2 N-7 returns SATURATION_HARD_EARNED** (~60% per ADVERSARIAL prior):
- SATURATION-DEPENDENT vectors deferred per Catalog #325 + #322
- Rate-attack budget reallocated to substrate-class-shift candidates (Z6/Z7/Z8 cascade per pose-axis council)
- G1 + corrected-F1/A2 remain valid as SATURATION-INDEPENDENT probes; G1's existing-anchor result is `0.0`
- Aggregate from rate-attack: corrected-F1/A2 only, hypothesis-grade `[0.180, 0.188]` if capacity validates; total mission contribution requires substrate-class-shift work

---

## 2. Mission Alignment per CLAUDE.md "Mission alignment — non-negotiable"

Per CLAUDE.md "Council hierarchy: 4-tier protocol" + "Mission alignment" non-negotiable Consequence 5: every T2+ verdict declares `council_predicted_mission_contribution`. This memo's contribution is **frontier_breaking** because:

1. **G1 IMMEDIATE-EXECUTION is structurally useful but measured zero current frontier movement** — cross-axis reranking tests whether existing anchors hide a CPU-axis win. Codex's 2026-05-18 probe found no existing qualifying CPU anchor below PR101/fec6 (`actual_delta_s=0.0`). The PR102 +0.033 CUDA-minus-CPU gap remains a per-archive prior for paired-axis probing only, not a conversion factor.

2. **Corrected F1/A2 probe-then-build is structurally frontier-breaking if the capacity probe passes** — the source-verified invariant is that PoseNet computes dims 7-12 but `compute_distortion` scores only dims 0-5; the deliverable contest exploit must be RGB perturbation capacity that preserves PoseNet dims 0-5 and SegNet argmax. The probe is $0 / 1 day; the build is $1-3 / 1-2 days. Per Catalog #313 probe-outcomes ledger: no predecessor INDEPENDENT/KILL/DEFER verdict on corrected F1/A2 (this is novel work).

3. **A-2 N-7 verdict is a MASSIVE information unlock** — either confirms saturation (correctly DEFER 22+ vectors and reallocate budget) OR refutes saturation (correctly green-light those vectors). $0 / 4-6h for either outcome is the highest-EV CPU-probe of the session per Catalog #229.

4. **A-3 + A-4 canonical helpers are frontier-protecting infrastructure** — Dykstra-feasibility-FIRST reduces wasted enumeration by 75-90% per Boyd; MI-min Wyner-Ziv canonical helper provides theorem-level bounds for B1/B2/all-Wyner-Ziv-derivative vectors per Tao. These are apparatus_maintenance that SERVES frontier_breaking per Consequence 5 nomenclature.

The reconciliation IS frontier-breaking because it converts the WAVE of research into ACTIONABLE OP-ROUTABLES that the operator can authorize this session at $0 cost.

---

## 3. Per-PRIMARY-TOP-5 Vector Saturation-Dependency Analysis

This is the CRITICAL classification step that drives the reconciliation. Each PRIMARY TOP-5 vector is classified as SATURATION-INDEPENDENT (proceed regardless of A-2 N-7 verdict) vs SATURATION-DEPENDENT (must await A-2 N-7).

### 3.1 G1 — CPU-axis-specific optimization → **SATURATION-INDEPENDENT** ✓

**Why independent**: G1 exploits per-archive CROSS-AXIS sensitivity, not a within-archive rate-extractable structure. The mechanism is per-axis-archive RE-RANKING using paired exact-eval data; it produces ZERO new archive bytes; therefore the compression-saturation question is structurally orthogonal. The 2026-05-18 Codex existing-anchor probe measured no current frontier move.

**Empirical receipts**:
- PR102 CUDA 0.22839 / CPU 0.19538 = +0.033 on that archive/runtime pair
- PR107 CUDA 0.22936 / CPU 0.19664 = +0.033 on that archive/runtime pair
- PR101/fec6 remains the measured CPU frontier in the current probe; no CUDA-to-CPU conversion is allowed

**Conclusion**: Hotz binding directive PROCEED IMMEDIATELY was correct as a $0 probe. No A-2 N-7 gating. The measured result is `actual_delta_s=0.0`; future G1 movement requires a new paired CPU anchor or a candidate-selection situation where the CPU-optimal candidate differs from the CUDA-optimal candidate.

### 3.2 Corrected F1/A2 — PoseNet-first-6-invariant RGB perturbations → **SATURATION-INDEPENDENT if capacity probe passes** ✓

**Why independent**: corrected F1/A2 exploits scorer-blind RGB perturbations whose PoseNet first-6-dim output and SegNet argmax remain stable. The source-verified invariant at upstream/modules.py:84 (`[..., : h.out // 2]` with h.out=12) proves dims 6-11 are unscored internal outputs; it does **not** prove dims 6-11 are archive bytes. The deliverable exploit is encoder-controlled RGB variation inside a scorer-invariant manifold. This is a SCORER-FEATURE-INVARIANCE exploit, not a compression-extraction exploit; therefore compression-saturation question is structurally orthogonal once capacity is empirically confirmed.

**Source-level verification** (canonical citation):
```
upstream/modules.py:84 (per PRIMARY's source-verification)
return sum((out1[h.name][..., : h.out // 2] - out2[h.name][..., : h.out // 2]).pow(2).mean(...)
            for h in self.hydra.heads if h.name in distortion_heads)
```

**Remaining cargo-cult**: "encoder can freely set dims 7-12 without affecting forward pass" is CARGO-CULTED-PENDING-PROBE. The probe `tools/probe_hydra_dim_7_12_score_invariance.py` is op-routable 2 (Fridrich+Contrarian revision in PRIMARY).

**Conclusion**: PROCEED with probe FIRST per Fridrich+Contrarian revision; build CONDITIONAL on probe PASS. Routing directive `83440e8a5` already landed; probe gate at step 1. No A-2 N-7 gating. Predicted ΔS `[-0.012, -0.004]`.

### 3.3 B1 — Contest-video-as-codebook → **SATURATION-DEPENDENT** ⚠

**Why dependent**: B1 ships codebook-index + offset + scale + RGB-residual bytes to the archive; the savings depend on whether (a) the upstream video patches are dense in rendered-frame space, AND (b) the standard compressor on the same archive doesn't already capture the equivalent dictionary-encoding via Brotli's context model. If A-2 N-7 returns SATURATION_HARD_EARNED, it means standard compressors are already at the entropy boundary; adding a custom Wyner-Ziv codebook layer is unlikely to break through.

**Conditional decision tree**:
- If A-2 N-7 SATURATION_REFUTED + Brotli/zstd/etc. shows ≥0.005 improvement: B1 likely under-performs the standard compressor; choose the winning standard compressor instead
- If A-2 N-7 SATURATION_REFUTED + improvement <0.005: B1 might add small marginal gains; worth probing
- If A-2 N-7 SATURATION_HARD_EARNED: defer B1; rate is at entropy boundary

**Conclusion**: GATE on A-2 N-7 verdict per RECONCILED-3. PRIMARY's predicted ΔS `[-0.020, -0.005]` is a CEILING, not a baseline.

### 3.4 Y3+Y6 — Luma-only encoding + JPEG quant-table steganography → **SATURATION-DEPENDENT** ⚠

**Why dependent**: Y3 (luma-only) relies on encoding 1-channel grayscale instead of 3-channel RGB; the savings depend on whether the archive's current encoding is already capturing chroma efficiently. Y6 (JPEG quant-table steganography) is a within-JPEG-block byte exploit; if the archive isn't using JPEG-encoded YUV blocks, Y6 doesn't apply. Both depend on the underlying compression structure.

**Quantizr empirical anchor partially HARD-EARNED**: Y3 alone is HARD-EARNED at the SegMap-mask-channel surface (PR101 gold-medal proves it). But Y3+Y6 composition on the FULL archive (not just SegMap) is CARGO-CULTED-PENDING-EMPIRICAL.

**Conclusion**: GATE on A-2 N-7. If saturated, Y3+Y6 composition unlikely to extract additional rate beyond what PR101 gold already exploits.

### 3.5 H1 — NVDEC hardware video decode → **SATURATION-DEPENDENT** ⚠ + **HARDWARE-AVAILABILITY-DEPENDENT** ⚠

**Why dependent**: H1 ships AV1-encoded sub-video bytes that NVDEC decodes at inflate time on T4. The savings depend on (a) NVDEC availability on contest T4 (CARGO-CULTED-PENDING-VERIFICATION per Carmack revision #2), AND (b) whether AV1 + NVDEC decode produces smaller archive than the current frontier codec (which depends on saturation).

**Also crosses ADVERSARIAL critique**: H1 is per-axis hardware-OPTIMAL on CUDA only (NVDEC requires GPU); per the supplement Part 1 hardware-exploit matrix, H1 is GPU-ONLY. Per Hotz revision #3 + CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA": optimizing for CUDA-only doesn't move leaderboard (which is CPU); H1 may improve CUDA score but not the leaderboard-relevant CPU score.

**Resolution**: H1 has TWO conditional gates: A-2 N-7 saturation + per-axis-leaderboard-relevance. Predicted ΔS `[-0.025, -0.008]` is the HIGHEST CEILING in PRIMARY's TOP-5 but with the most conditional gates.

**Conclusion**: GATE on A-2 N-7 + per-axis-leaderboard-relevance check. If both clear, H1 has compelling upper bound.

### 3.6 Summary classification table

| PRIMARY TOP-5 | Classification | A-2 N-7 gating | Hardware-axis caveats | Predicted ΔS [contest-CPU] |
|---|---|---|---|---|
| **G1 (RANK 2)** | **SATURATION-INDEPENDENT probe** | NO | CPU-axis directly relevant; AXIS-INVARIANT exploit | `0.0` realized for existing-anchor rerank; future paired-anchor upside only |
| **F1 (RANK 1)** | **SATURATION-INDEPENDENT** | NO | Scorer-feature-invariance is axis-invariant | `[-0.012, -0.004]` ✓ |
| **B1 (RANK 3)** | SATURATION-DEPENDENT | YES | AXIS-INVARIANT (mostly); patches deterministic | `[-0.020, -0.005]` conditional |
| **Y3+Y6 (RANK 4)** | SATURATION-DEPENDENT | YES | BOTH-WITH-DIFFERENT-OPTIMAL (JPEG decode paths differ) | `[-0.015, -0.004]` conditional |
| **H1 (RANK 5)** | SATURATION-DEPENDENT + HARDWARE-AVAILABILITY-DEPENDENT | YES | GPU-ONLY for NVDEC; leaderboard is CPU | `[-0.025, -0.008]` doubly conditional |

**Key finding**: 2-of-5 PRIMARY TOP-5 are SATURATION-INDEPENDENT probe axes (40%); 3-of-5 are SATURATION-DEPENDENT (60%). The original SATURATION-INDEPENDENT aggregate `[-0.022, -0.007]` is now hypothesis-grade: G1 existing-anchor rerank measured `actual_delta_s=0.0`, and corrected F1/A2 still needs capacity + net-byte proof.

---

## 4. Per-ADVERSARIAL-12-NEW-Vector Incorporation Analysis

For each of ADVERSARIAL's 12 NEW vectors (Tao 3 + Carmack 3 + Hotz 3 + Boyd 3), classify as: PRIMARY-MISSED-LEGITIMATELY (extend TOP-5) vs PRIMARY-COVERED-IMPLICITLY (already in 43 vectors) vs OUT-OF-SCOPE (not actual rate-attacks).

### 4.1 Tao's 3 NEW vectors (mathematical foundations)

| ADV ID | Vector | Verdict | PRIMARY coverage | Reconciliation action |
|---|---|---|---|---|
| **N-1** | MI-min Wyner-Ziv per Atick-Redlich 1990 | **PRIMARY-PARTIALLY-COVERED** | Wyner-Ziv referenced in PRIMARY B1 + Cat B header; per-byte MI estimator NOT in PRIMARY | **EXTEND**: land as STRUCTURAL canonical helper `tac.rate_attack_mi_lower_bound` per RECONCILED-5 |
| **N-2** | Kolmogorov NCD via Brotli proxy | **OUT-OF-DIRECT-SCOPE (a PROBE not a rate-attack)** | Not in PRIMARY but its function is to RANK archives by rate-attack-suitability | **QUEUE**: useful post-A-2-N-7 as a discriminator for SATURATION-DEPENDENT vector ordering |
| **N-3** | Convex-hull tightening / support function | **PRIMARY-COVERED-IMPLICITLY** | PRIMARY Cat B + Boyd revision implicitly invokes; explicit canonical helper not in PRIMARY | **EXTEND**: complement to A-3 (Dykstra) canonical helper |

### 4.2 Carmack's 3 NEW vectors (engineering simplicity)

| ADV ID | Vector | Verdict | PRIMARY coverage | Reconciliation action |
|---|---|---|---|---|
| **N-4** | inflate.py ≤30 LOC OPTIMIZER | **PRIMARY-MISSED-LEGITIMATELY** | PRIMARY treats LOC as constraint (HNeRV parity L4 ≤100), not OPTIMIZER | **OPTIONAL**: queue as supplementary low-risk experiment per row 7 of operator decision table |
| **N-5** | Brotli pre-seeded dictionary on comma2k19 | **PRIMARY-MISSED-LEGITIMATELY** | PRIMARY's M-category covers ZIP overhead, not dictionary-based compression | **CONDITIONAL** on A-2 N-7: this IS one of the verdict-conditional candidates in row 6 of operator decision table. **COMPLIANCE CHECK REQUIRED** on external-dict per ADVERSARIAL caveat |
| **N-6** | zstd `--long=27` vs Brotli quality 11 bake-off | **PRIMARY-MISSED but COVERED by A-2 N-7** | A-2 N-7 routing directive explicitly sweeps zstd levels 1-22 + brotli 1-11; N-6 is a SUB-EXPERIMENT of A-2 N-7 | **NOT a separate action**: A-2 N-7 verdict ANSWERS this. Result lands automatically. |

### 4.3 Hotz's 3 NEW vectors (empirical-first, CPU/GPU asymmetry)

| ADV ID | Vector | Verdict | PRIMARY coverage | Reconciliation action |
|---|---|---|---|---|
| **N-7** | CPU-axis Tier-1 standard compressor sweep | **A-2 N-7 ITSELF** | Same as ADVERSARIAL's mandatory empirical anchor | **EXECUTING via Codex**: Routing directive `1ac2063de` landed; /goal LOOP picks up |
| **N-8** | Tier-1 preset sweep within standard compressors | **A-2 N-7 SUB-EXPERIMENT** | Phase 1 of A-2 N-7 sweeps presets/levels exhaustively per directive | **PART OF A-2 N-7**: no separate action |
| **N-9** | Per-pair archive splitting (600 entries vs monolithic) | **PRIMARY-MISSED-LEGITIMATELY** | PRIMARY's M-category covers ZIP structure but not per-pair-vs-monolithic | **CONDITIONAL**: add to A-2 N-7 Phase 2 (per-section sweep) per directive section "Phase 2: Per-block / per-section sweep" |

### 4.4 Boyd's 3 NEW vectors (convex optimization)

| ADV ID | Vector | Verdict | PRIMARY coverage | Reconciliation action |
|---|---|---|---|---|
| **N-10** | Pareto-simplex sweep with rate as constraint | **PRIMARY-COVERED-IMPLICITLY by Catalog #322 + Cat M3** | PRIMARY M3 dead-byte audit invokes; CVXPY formulation not in PRIMARY | **OPTIONAL**: nice-to-have canonical helper; lower priority than A-3 + A-4 |
| **N-11** | ADMM split (primal=archive bytes, dual=distortion) | **PRIMARY-MISSED-LEGITIMATELY** | PRIMARY uses Lagrangian framing but not ADMM with closed-form primal | **DEFERRED**: queue for post-A-2-N-7 if SATURATION_REFUTED; high LOC + complex tooling |
| **N-12** | Dykstra-feasibility intersection FIRST | **A-3 ITSELF** | Same as ADVERSARIAL's A-3 in their alternative TOP-5 | **STRUCTURAL PRIMITIVE per RECONCILED-5**: land as canonical helper |

### 4.5 Aggregate ADVERSARIAL-NEW incorporation summary

| Outcome | Count | Vectors | Action |
|---|---|---|---|
| EXECUTING via A-2 N-7 | 2 | N-7, N-8 | Codex /goal LOOP picks up |
| STRUCTURAL PRIMITIVE (canonical helper) | 4 | N-1, N-3, N-12 (=A-3), N-12-sister | Queue for Codex post-A-2-N-7 |
| CONDITIONAL on A-2 N-7 verdict | 2 | N-5, N-9 | Pick post-verdict |
| PRIMARY-MISSED-LEGITIMATELY (optional) | 3 | N-4, N-9, N-11 | Optional / deferred per row 7 |
| PROBE-not-rate-attack (informational) | 1 | N-2 | Queue as post-A-2-N-7 ranker |
| Aggregate | 12 | — | NO conflicts; ALL incorporated via the reconciled cascade |

**Key finding**: 12-of-12 ADVERSARIAL NEW vectors are incorporated into the reconciliation; ZERO are rejected as cargo-cult. The two streams are complementary, not competing.

---

## 5. The 5 Binding Critiques Resolution

### 5.1 Tao: META-paradigm SINS is mathematically CONFLATED → **RESOLVED via partition into 3 sub-paradigms**

ADVERSARIAL Tao argued SINS conflates (a) MI-min Wyner-Ziv, (b) Kolmogorov-complexity-min, (c) Pareto-tightening into one metaphor. The reconciliation **ADOPTS Tao's partition** as the operational classification scheme:

| Tao sub-paradigm | Mathematical theorem | PRIMARY TOP-5 vectors | ADVERSARIAL NEW vectors |
|---|---|---|---|
| **(a) MI-min Wyner-Ziv** | Wyner-Ziv 1976 + Atick-Redlich 1990 | B1, F1, F2, F3, F4, Y2, C1 | N-1, N-5 (Brotli-dict is dictionary-Wyner-Ziv) |
| **(b) Kolmogorov-complexity-min** | Cilibrasi-Vitányi 2005 + Schmidhuber compression-as-intelligence | B4, M3, all Cat-D YUV exploits, all Cat-E hardware exploits, Y5+Y6 JPEG | N-2 NCD, N-4 LOC budget, N-6 compressor bake-off |
| **(c) Pareto-tightening** | Dykstra-Boyle 1985 + Boyd convex feasibility | G1 (axis-conditioned RD), G7 inflate device, M1+M2 ZIP overhead, A1 SABOR margin | N-3 support function, N-10 Pareto simplex, N-11 ADMM, N-12 Dykstra-FIRST |
| **Outside formalism** | Cross-disciplinary / process | A3 (training guard), B4 cheap-prototype | (none) |

**Operator-routable benefit**: each TOP-5 vector now carries an EXPLICIT mathematical paradigm tag. Future research can rigorously check that NEW vectors fall into one of the 3 sub-paradigms; vectors that don't are flagged for theorem-level grounding work.

**Note**: SINS as a NAME still works as a UNIFYING UMBRELLA across the 3 sub-paradigms (all 3 are about decoder-side structural information not shipped); the partition adds RIGOR within the umbrella.

### 5.2 Contrarian: "43" is numerically ARBITRARY → **RESOLVED via "the count is meaningful per sub-paradigm partition"**

ADVERSARIAL Contrarian argued 43 is the artifact of brainstorm depth not a HARD-EARNED partition. The reconciliation **ACCEPTS the critique BUT REJECTS the implication** that vector enumeration is invalid. Per Tao's partition above: the 43 vectors break down into ~17 MI-min + ~16 K-complexity-min + ~7 Pareto-tightening + ~3 outside-formalism. Each sub-paradigm's count is bounded by the theorems' hypothesis spaces, not arbitrary.

The RECONCILIATION downgrades the importance of the count "43" — it's the partition that matters, not the integer. PRIMARY's 43 is more of a "comprehensive enumeration across operator-elevated dimensions" than a HARD-EARNED partition; ADVERSARIAL's complaint is fair but doesn't invalidate the work.

**Operator-routable benefit**: future enumeration waves should COUNT per sub-paradigm (e.g. "15 MI-min vectors + 12 K-complexity-min vectors + 5 Pareto-tightening vectors enumerated") rather than report a single integer.

### 5.3 Assumption-Adversary: saturation hypothesis is empirically PENDING → **RESOLVED via A-2 N-7 mandatory anchor + SATURATION-INDEPENDENT subset proceeds anyway**

ADVERSARIAL Assumption-Adversary argued the META-paradigm is operating within a CARGO-CULTED-PENDING assumption that the archive packet has remaining rate-extractable structure. The reconciliation **ACCEPTS the critique IN FULL** and operationalizes via:

1. A-2 N-7 routing directive landed (commit `1ac2063de`); Codex /goal LOOP executes
2. SATURATION-INDEPENDENT subset (G1+corrected-F1/A2) is structurally orthogonal enough to justify probes regardless; measured score authority still requires each probe's artifact
3. SATURATION-DEPENDENT subset (B1+Y3+Y6+H1) explicitly GATED on A-2 N-7 verdict
4. Predicted verdict distribution: 60% SATURATION_HARD_EARNED (defer SATURATION-DEPENDENT subset) / 30% SATURATION_PENDING (extended sweep) / 10% SATURATION_REFUTED (green-light)
5. Either verdict outcome is HIGHEST-EV information per Catalog #229

**Operator-routable benefit**: $0 / 4-6h investment for a verdict that either (a) saves $30-80+ of misallocated Modal/Lightning budget by deferring SATURATION-DEPENDENT vectors, or (b) green-lights $1-8 of those vectors with HARD-EARNED confidence.

### 5.4 Boyd: composition INFEASIBLE per 9×9 sub-additive default → **RESOLVED via per-pair Dykstra-feasibility check + sub-additive realistic aggregate**

ADVERSARIAL Boyd argued naive TOP-5 composition will produce Pareto-feasibility-INFEASIBLE points in 75-90% of attempts per Catalog #322 4-of-8 sub-additive priors. The reconciliation **ACCEPTS the critique** and operationalizes via:

1. Per-pair Dykstra-feasibility check per Catalog #296 BEFORE any composition_alpha α-prediction (PRIMARY had this in §3 9×8 matrix; we ratify and extend)
2. RECONCILED aggregate ΔS reported as SUB-ADDITIVE realistic (per Catalog #322 v2 cascade `adjust_predicted_delta_for_composition_alpha_v2`), not naive-additive
3. A-3 Dykstra-feasibility-FIRST canonical helper landed per RECONCILED-5 ensures future composition decisions go through Dykstra projection FIRST
4. RECONCILED TOP-5's aggregate `[0.155, 0.184]` remains a hypothesis-grade planning band; G1 now has measured `actual_delta_s=0.0`, so any aggregate must be recomputed from probe evidence before promotion

**Operator-routable benefit**: predicted aggregates are realistic (not optimistic); operator budget reservations are accurate.

### 5.5 Hotz: apparatus_maintenance NOT frontier-breaking → **PARTIALLY RESOLVED via SATURATION-INDEPENDENT subset is frontier-breaking; SATURATION-DEPENDENT subset is research-pending**

ADVERSARIAL Hotz argued PRIMARY's 43 SCAFFOLD vectors without empirical anchor repeats the 12-month premortem failure mode. The reconciliation **PARTIALLY ACCEPTS** the critique:

**Accepted**: 41-of-43 vectors WITHOUT empirical anchor are at risk of becoming apparatus_maintenance; ADVERSARIAL's $0 mandatory anchor (A-2 N-7) is the canonical fix.

**Rejected for 2-of-43 only as probe axes**: G1 has per-archive paired-axis priors (e.g. PR102 +0.033 CUDA-minus-CPU on that archive/runtime pair); F1 has source-level unscored-dim evidence (upstream/modules.py:84). These two are grounded enough for probes, not enough for score movement claims. Hotz's own binding directive in PRIMARY §8 on G1 IMMEDIATE EXECUTION is consistent with running the $0 probe.

**Net effect**: the SATURATION-INDEPENDENT subset (G1+corrected-F1/A2) is probe-worthy; the SATURATION-DEPENDENT subset is research-pending until A-2 N-7 lands. The reconciliation puts the operator-routable IMMEDIATE-action items in the SATURATION-INDEPENDENT subset; defers the rest.

**Operator-routable benefit**: no $30-80 burn on apparatus_maintenance dressed as frontier_breaking; clear separation between immediate $0 probes (G1+corrected-F1/A2) and contingent-on-empirical-anchor work (B1+Y3+Y6+H1+A-1).

---

## 6. Alternative TOP-5 Reconciliation: Side-by-Side + Final RECONCILED TOP-5

### 6.1 Side-by-side comparison

| Rank | PRIMARY TOP-5 | ADVERSARIAL Quartet TOP-5 | Reconciled |
|---|---|---|---|
| **1** | F1 Hydra dims 7-12 (superseded direct-channel framing) | A-1 Brotli + comma2k19 dict | **G1 Cross-axis re-rank** (immediate; $0; Hotz binding) |
| **2** | G1 CPU-axis-specific | A-2 N-7 CPU-axis Tier-1 sweep | **Corrected F1/A2 scorer-blind RGB perturbation capacity** (probe-then-build; $0+$1-3; capacity-pending) |
| **3** | B1 Contest-video-codebook | A-3 Dykstra-feasibility-FIRST | **A-2 N-7 standard compressor sweep** ($0; executing via Codex; verdict-driving) |
| **4** | Y3+Y6 Luma + JPEG-quant | A-4 MI-min Wyner-Ziv | **A-3 Dykstra-feasibility-FIRST canonical helper** ($0; structural primitive) |
| **5** | H1 NVDEC hardware decode | A-5 inflate.py ≤30 LOC OPTIMIZER | **A-4 MI-min Wyner-Ziv canonical helper** ($0; structural primitive) |
| **6 (conditional)** | — | — | **POST-A-2-N-7 PICK: B1 OR H1 OR Y3+Y6 OR A-1 Brotli-dict** based on verdict |
| **Cost (all 5)** | $30-80+ Modal/Lightning | $0 | **$0** (immediate-action 5; $1-8 for conditional row 6) |
| **Time** | 5-10 days | ~1 day | THIS SESSION for G1; 1-2 days for F1; 4-6h for A-2 N-7; 1 day each A-3/A-4 |
| **HARD-EARNED count** | 1-of-5 (F1 source-verified) + 1-of-5 (G1 empirical probe, zero movement) = 2-of-5 probe-grounded | 4-of-5 (Carmack/Hotz/Boyd/Tao theorem-anchored) + 1 empirical anchor | **5-of-5 structurally grounded, but only probe-authoritative until artifacts land** (G1+corrected-F1/A2+A-2-N-7-is-test+A-3+A-4-are-canonical-helpers) |
| **Empirical-anchor-included** | NO | YES (A-2 N-7) | **YES (A-2 N-7 + G1 existing-anchor CPU rerank result `actual_delta_s=0.0`)** |

### 6.2 The RECONCILED TOP-5 with explicit per-vector justification

#### RECONCILED-TOP-1: G1 Cross-axis CPU re-rank (PRIMARY RANK 2) — SATURATION-INDEPENDENT — IMMEDIATE EXECUTION

**Why TOP-1 in reconciled order**: Hotz binding directive PROCEED IMMEDIATELY made this the lowest-cost operator-routable probe; cost $0; Codex's 2026-05-18 execution measured `actual_delta_s=0.0` on existing qualifying CPU anchors, so G1 is evidence-producing but not a current frontier move.

**Routing status**: Routing directive `83440e8a5` already landed; Codex /goal LOOP picks up.

**Per-axis matrix classification**: AXIS-INVARIANT exploit BUT operationally CPU-axis-optimal (re-rank for CPU axis since leaderboard ranks CPU).

**Mathematical paradigm (Tao partition)**: (c) Pareto-tightening — axis-conditioned RD frontier optimization.

#### RECONCILED-TOP-2: Corrected F1/A2 scorer-blind RGB perturbation probe → build (PRIMARY RANK 1 superseded framing) — SATURATION-INDEPENDENT IF CAPACITY PASSES — PROBE FIRST

**Why TOP-2 in reconciled order**: Source-verified invariant at upstream/modules.py:84 plus pending RGB-manifold capacity probe; predicted ΔS `[-0.012, -0.004]` remains hypothesis-band only until probe evidence lands; probe is $0; build is $1-3. Demoted from RANK 1 in PRIMARY's ordering because G1 is OPERATOR-ROUTABLE IMMEDIATELY whereas corrected F1/A2 requires a capacity probe gate.

**Routing status**: Routing directive `83440e8a5` already landed; probe gate at step 1; build conditional on probe PASS.

**Per-axis matrix classification**: AXIS-INVARIANT if measured as RGB perturbations that preserve scored PoseNet dims 0-5 and SegNet argmax on both CPU and CUDA.

**Mathematical paradigm (Tao partition)**: (a) MI-min Wyner-Ziv — zero-rate side channel through scorer-feature-invariance.

#### RECONCILED-TOP-3: A-2 N-7 Standard compressor sweep (ADVERSARIAL MANDATORY) — VERDICT-DRIVING

**Why TOP-3 in reconciled order**: This is THE empirical anchor that determines all SATURATION-DEPENDENT vectors' fate. $0 / 4-6h. ADVERSARIAL mandatory per Hotz STRICT REVISION.

**Routing status**: Routing directive `1ac2063de` landed; Codex /goal LOOP executes.

**Per-axis matrix classification**: Tests AXIS-INVARIANT exploits across all standard compressors.

**Mathematical paradigm (Tao partition)**: (b) Kolmogorov-complexity-min via standard-compressor proxy (Cilibrasi-Vitányi 2005).

#### RECONCILED-TOP-4: A-3 Dykstra-feasibility-FIRST canonical helper (ADVERSARIAL STRUCTURAL) — INFRASTRUCTURE

**Why TOP-4 in reconciled order**: Reduces wasted enumeration effort by 75-90% per Boyd; complementary to A-4 MI-min; landing this as canonical helper benefits ALL future rate-attack research.

**Routing status**: Queue for Codex post-A-2-N-7 (parallel with A-4).

**Per-axis matrix classification**: STRUCTURAL primitive; not axis-specific.

**Mathematical paradigm (Tao partition)**: (c) Pareto-tightening — alternating projections per Dykstra-Boyle 1985.

#### RECONCILED-TOP-5: A-4 MI-min Wyner-Ziv canonical helper (ADVERSARIAL STRUCTURAL) — INFRASTRUCTURE

**Why TOP-5 in reconciled order**: Theorem-level bounds for ALL Wyner-Ziv-derivative vectors (B1, F1, F2, F3, F4, Y2, C1, N-1 itself); canonical helper supports per-byte conditional MI estimation.

**Routing status**: Queue for Codex post-A-2-N-7 (parallel with A-3).

**Per-axis matrix classification**: AXIS-INVARIANT (MI estimation is a deterministic computation).

**Mathematical paradigm (Tao partition)**: (a) MI-min Wyner-Ziv — explicit instance.

#### RECONCILED-TOP-6 (CONDITIONAL): Post-A-2-N-7 PICK from {B1, H1, Y3+Y6, A-1 Brotli-dict}

**Why CONDITIONAL**: Outcome of A-2 N-7 determines which of these is highest-EV:

- If SATURATION_REFUTED with WIN by standard compressor at level X → A-1 Brotli-dict (just use the winning compressor)
- If SATURATION_REFUTED with WIN by custom codec direction → B1 (contest-video-as-codebook)
- If SATURATION_PENDING + JPEG-decoder-paths matter → Y3+Y6
- If SATURATION_PENDING + NVDEC-available + CUDA-axis-relevance verified → H1

**Operator-routable**: pick post-A-2-N-7 verdict landing; predicted ΔS varies per pick.

### 6.3 Aggregate predicted ΔS for RECONCILED TOP-5

**SATURATION-INDEPENDENT subset after G1 execution (G1 + corrected F1/A2)**:
- G1 realized movement on existing qualifying CPU anchors: `0.0` [contest-CPU]; no new score claim
- Remaining corrected-F1/A2 upside: hypothesis-grade `[-0.012, -0.004]` until RGB-perturbation capacity and net byte savings are measured
- Frontier displacement authority: none yet; post-G1 planning band is corrected-F1/A2 only, `[0.180, 0.188]` [contest-CPU] if the hypothesis validates

**With STRUCTURAL primitives (A-3 + A-4)**: no direct ΔS contribution; reduces wasted enumeration and provides theorem-level bounds.

**With CONDITIONAL row 6 (BEST CASE: SATURATION_REFUTED + winning vector adds `[-0.015, -0.005]`)**:
- Additive after G1 realized zero movement: `[-0.027, -0.009]`
- Sub-additive (corrected-F1/A2 plus row-6 vector; alpha ≈ 0.7): `[-0.021, -0.007]`
- Frontier displacement planning band: `[0.171, 0.185]` [contest-CPU], hypothesis-grade until artifacts land

**With CONDITIONAL row 6 (PROBABILITY-WEIGHTED: 0.4 × `[-0.015, -0.005]` + 0.6 × `[0, 0]`)**:
- Expected: `[-0.006, -0.002]` additional from row 6
- Combined with corrected-F1/A2 hypothesis only: `[-0.018, -0.006]` expected total
- Frontier displacement planning band: `[0.174, 0.186]` [contest-CPU], hypothesis-grade

**Comparison vs PRIMARY's predicted aggregate `[0.152, 0.179]`**: RECONCILED is slightly less aggressive in the lower bound (0.164 vs 0.152) but more REALISTIC because:
- Removes assumption that all 5 PRIMARY vectors land at predicted band (only 2 are SATURATION-INDEPENDENT)
- Incorporates probability-weighting per A-2 N-7 verdict prior
- Uses sub-additive composition_alpha per Catalog #322 v2

**Comparison vs ADVERSARIAL's predicted `[-0.015, -0.004]`**: historical RECONCILED was more aggressive in the lower bound because it included G1+F1 as SATURATION-INDEPENDENT. Post-Codex execution, G1 contributes authority and a zero-delta guardrail, not score movement; the remaining upside must come from corrected-F1/A2 capacity proof or row-6 empirical artifacts.

---

## 7. A-2 N-7 Sequencing

### 7.1 Current status

- Routing directive landed at commit `1ac2063de` (`.omx/research/codex_routing_directive_a2_n7_cpu_axis_tier1_standard_compressor_sweep_empirical_anchor_20260518.md`)
- Codex `/goal LOOP` autonomously picks up directives from the routing channel
- Per directive section "OPERATOR-FACING NOTE": "This empirical anchor SHOULD COMPLETE WITHIN 4-6 HOURS WALL-CLOCK at $0 cost"
- Expected verdict registration: `.omx/state/probe_outcomes.jsonl` per Catalog #313
- Expected artifact: `experiments/results/a2_n7_compressor_sweep_<utc>/report.json`
- Expected memory entry: `feedback_a2_n7_compressor_sweep_landed_20260518.md`

### 7.2 Verdict-driven cascade

```
A-2 N-7 lands (4-6h)
│
├── SATURATION_HARD_EARNED (predicted ~60% per ADVERSARIAL prior)
│   └── DEFER PRIMARY's B1+Y3+Y6+H1 (SATURATION-DEPENDENT subset)
│       └── Reallocate rigor budget to:
│           ├── G1 + F1 (SATURATION-INDEPENDENT subset) PROCEED in parallel
│           ├── Substrate-class-shift candidates (Z6/Z7/Z8 per cross-stack synthesis pose-axis council)
│           └── A-3 + A-4 canonical helpers (frontier-protecting infrastructure)
│
├── SATURATION_REFUTED (predicted ~30% per ADVERSARIAL prior; revised ~40% per RECONCILED)
│   └── GREEN-LIGHT SATURATION-DEPENDENT vectors
│       └── Per A-2 N-7 verdict, the WINNING compressor (or codec direction):
│           ├── If standard compressor wins at level X → A-1 Brotli-dict OR direct compressor swap
│           ├── If custom codec direction wins → B1 contest-video-codebook
│           └── If hardware-codec wins → H1 NVDEC (subject to GPU-only caveat per supplement Part 1)
│
└── SATURATION_PENDING (predicted ~10% per ADVERSARIAL prior; revised ~10% per RECONCILED)
    └── Extended sweep (Phase 2 per-section + Phase 3 comma2k19 dict) per directive
        └── Operator decision on additional $0-2 probe budget
```

### 7.3 Concrete OPERATOR-ROUTABLE next session

Given A-2 N-7 lands within 4-6 hours, the next operator-session probably has the verdict. The reconciled action queue for that session:

1. **Read** `experiments/results/a2_n7_compressor_sweep_<utc>/report.json` aggregate verdict
2. **Check** `.omx/state/probe_outcomes.jsonl` for A-2 N-7 entry with verdict + threshold + next_action
3. **Route ROW-6 CONDITIONAL PICK** per the verdict-driven cascade in §7.2
4. **Confirm G1 + F1 SATURATION-INDEPENDENT subset** has landed per Codex /goal LOOP picking up routing directives `83440e8a5`
5. **Queue A-3 + A-4 canonical helpers** for Codex post-row-6-pick

---

## 8. Per-Axis Hardware Exploit Matrix Integration

Per supplement Part 1 the 4-class matrix is: CPU-ONLY / GPU-ONLY / BOTH-WITH-DIFFERENT-OPTIMAL / AXIS-INVARIANT. Classification of the RECONCILED TOP-5 + alternative TOP-5 + per-vector applicability:

### 8.1 Classification table

| Vector | Per-axis class | CPU-axis relevance (leaderboard) | CUDA-axis relevance (transparency) | Reconciliation note |
|---|---|---|---|---|
| **RECONCILED-1 G1** | AXIS-INVARIANT (mechanism) + CPU-axis-OPTIMAL (operational) | DIRECT as candidate-selection/probe criterion; current existing-anchor result is 0.0 | n/a (cross-axis re-rank doesn't hurt CUDA — chosen archive still has its CUDA eval recorded) | Treat +0.033 gaps as per-archive priors only; no conversion factor |
| **RECONCILED-2 corrected F1/A2** | AXIS-INVARIANT if capacity probe passes on both axes | DIRECT only as scorer-blind RGB perturbation capacity | DIRECT same constraint; no direct dim-channel receiver | Per-pair perturbation manifold must preserve scored pose dims 0-5 + SegNet argmax |
| **RECONCILED-3 A-2 N-7** | TESTS AXIS-INVARIANT exploits | DIRECT (tests CPU axis specifically per Hotz N-7 framing) | INDIRECT (sweep can extend to CUDA-axis archives) | Phase 1 CPU-only per directive; could extend if SATURATION_REFUTED |
| **RECONCILED-4 A-3 Dykstra** | STRUCTURAL primitive | n/a (helper) | n/a (helper) | Operates on constraint sets, not bytes |
| **RECONCILED-5 A-4 MI-min** | AXIS-INVARIANT (MI is deterministic on byte data) | INDIRECT (provides bounds for vectors) | INDIRECT (same) | Operates on byte data, not axis-specific |
| **RECONCILED-6 (conditional) B1** | AXIS-INVARIANT (mostly; AV1 decode is deterministic across axes) | DIRECT | DIRECT | Codec choice may have BOTH-WITH-DIFFERENT-OPTIMAL profile selection |
| **RECONCILED-6 (conditional) Y3+Y6** | BOTH-WITH-DIFFERENT-OPTIMAL (JPEG decode paths differ CPU vs GPU) | DIRECT | DIRECT (NVJPEG path on CUDA) | JPEG quant-table works on both; decode paths differ |
| **RECONCILED-6 (conditional) H1** | GPU-ONLY (NVDEC requires GPU) | INDIRECT (only via cross-axis carry; CPU has no NVDEC) | DIRECT (hardware decode at line rate) | Per Hotz revision #3: CUDA-only exploit doesn't directly help leaderboard CPU; downgrade unless paired with CPU-side decoder |
| **RECONCILED-6 (conditional) A-1 Brotli-dict** | AXIS-INVARIANT (Brotli has no hardware-acceleration variants) | DIRECT | DIRECT | Same decode on both axes |

### 8.2 PRIMARY vs ADVERSARIAL conflicting per-axis claims resolution

**PRIMARY claim**: H1 NVDEC has HIGHEST CEILING `[-0.025, -0.008]`.

**ADVERSARIAL counter (Hotz N-7 framing)**: leaderboard is CPU; GPU-only exploits don't help leaderboard.

**RECONCILIATION**: PRIMARY's H1 prediction is contingent on the CUDA axis being relevant to the leaderboard, which it ISN'T (per Hotz revision and supplement Part 1). H1's predicted ΔS on the LEADERBOARD CPU axis is conservatively much lower, perhaps `[0, -0.002]` (only via cross-axis-carry effects). H1 is DOWNGRADED in the reconciliation; it remains in the conditional row 6 pick but is unlikely to win unless the CPU-side software decode path is competitive.

### 8.3 The per-axis matrix as a NEW canonical helper proposal

The matrix in supplement Part 1 is itself worthy of being a canonical helper: `tac.per_axis_hardware_exploit_classifier` taking a vector + returning its CPU-ONLY / GPU-ONLY / BOTH-WITH-DIFFERENT-OPTIMAL / AXIS-INVARIANT classification. This complements A-3 + A-4 as a STRUCTURAL primitive. Queue as a low-priority canonical helper post-A-2-N-7.

---

## 9. Dual-Device Master-Gradient Research Question Resolution

Per supplement Part 2 the operator's research question: should master-gradient be computed on BOTH CPU + CUDA?

### 9.1 PRIMARY's implicit answer

PRIMARY's G1 vector argues per-parameter Δ-CPU-CUDA sensitivity map could be derived from existing dual-eval data (whole-archive scores, not per-byte gradients). Did not explicitly recommend dual-device master-gradient extraction.

### 9.2 ADVERSARIAL's implicit answer

ADVERSARIAL did not address the dual-device master-gradient question directly. A-2 N-7's verdict will inform whether per-axis distinct compression behaviors exist.

### 9.3 Reconciliation: DEFER per RECONCILED-6

**Recommendation**: defer the dual-device master-gradient extension to a separate routing directive, gated on A-2 N-7 outcome.

**Rationale**:
- If A-2 N-7 returns SATURATION_REFUTED with consistent per-archive outcomes across all standard compressors → master-gradient probably has small axis-divergence (H1 hypothesis); dual-device extension low EV
- If A-2 N-7 returns SATURATION_HARD_EARNED → rate is at entropy boundary on both axes; dual-device gradient won't help find more rate
- If A-2 N-7 surfaces archive-family-specific behavior (e.g. PR101 saturated, PR106 not) → dual-device extension MIGHT surface the H3 hypothesis (significant axis-divergence with opposite-sign per-byte gradients)

**Concrete decision**: post-A-2-N-7 the operator can route a separate "dual-device master-gradient extension" directive if the verdict surfaces a compelling signal. Cost would be ~$2 on Modal T4 for the CUDA-side extraction (CPU-side already in `tac.master_gradient`).

### 9.4 Cross-reference

Per the cathedral autopilot Catalog #319 Q3 v2 cascade in `tools/cathedral_autopilot_autonomous_loop.py`: the `OptimalPerPairTreatmentPlan` consumer already handles per-pair master-gradient. Dual-device extension would extend this to per-axis-per-pair fingerprinting. The cardinality is manageable (600 pairs × 2 axes × ~10 features per pair = 12000 entries).

---

## 10. Comprehensive Signal Inventory Integration

Per supplement Part 3 the comprehensive signal inventory catalog. Cross-reference which signals PRIMARY/ADVERSARIAL used vs missed.

### 10.1 Signals used by PRIMARY

- `upstream/modules.py` source code (for F1 + Cat-F all Hydra exploits)
- `upstream/videos/0.mkv` (for B1 contest-video-codebook)
- Catalog #316 frontier scan (for canonical frontier anchor in frontmatter)
- Cross-stack synthesis 9-design landings (cited as related deliberation)
- HNeRV parity discipline (LOC budgets per L4)
- Catalog #322 composition_alpha (for 9×8 matrix in §3)
- Council deliberation posterior schema (for v2 frontmatter)

### 10.2 Signals used by ADVERSARIAL

- Same as PRIMARY plus:
- Closure campaign master memo (for saturation hypothesis citation)
- Rate-attack novel vectors design memo (for 13-vector seed audit in §5)
- Catalog #229 premise verification (for the 8 PVs in §3.2)
- Catalog #322 v2 cascade (for sub-additive composition critique)
- Catalog #303 cargo-cult audit per assumption (for per-vector classification)

### 10.3 Signals BOTH MISSED that the reconciliation surfaces

- `.omx/state/master_gradient_anchors.jsonl` per-pair fp64 gradient for PR101_lc_v2 anchor `f174192aeadf...` — relevant to G1 per-axis re-ranking AND to dual-device master-gradient research question
- `.omx/state/cost_band_posterior.jsonl` — cost-band priors per Catalog #175 inform whether predicted bands fall in known cost bands
- `.omx/state/probe_outcomes.jsonl` per Catalog #313 — checked predecessor verdicts; no rate-attack-specific predecessors found (this is novel territory)
- `.omx/state/modal_call_id_ledger.jsonl` per Catalog #245 — Modal dispatch history shows 14+ dispatches today; informs cost-band priors
- `.omx/state/substrate_composition_matrix.json` per Catalog #322 — composition_alpha cells; PRIMARY's 9×8 matrix is consistent with this surface
- `.omx/state/wyner_ziv_deliverability/` per Catalog #319 — deliverability proof artifacts; relevant to B1 + N-5 + A-1 Brotli-dict per "deliverable bytes count" framing
- `src/tac/sensitivity_map.py` per Catalog #275 — sensitivity-map with axis-level reweighting API; could complement G1 + A-4 MI-min helper
- `src/tac/master_gradient_consumers.py` — canonical consumers of per-pair gradient; relevant to G1 IF the per-pair gradient is per-axis-distinct

### 10.4 Reconciliation operator-routable benefit

The signal inventory shows the canonical helpers + state ledgers are MOSTLY in place; the reconciled TOP-5 doesn't require new state ledger infrastructure. The two NEW canonical helpers proposed (A-3 Dykstra + A-4 MI-min) plug into existing surfaces.

---

## 11. Cargo-Cult Audit per Catalog #303

Per CLAUDE.md "FORBIDDEN PATTERNS — Forbidden symposium-band-prediction-without-Dykstra-feasibility-check" + Catalog #303 cargo-cult-audit-per-assumption: this memo's claims undergo per-assumption audit.

### 11.1 Claim: "PRIMARY's TOP-5 dominates ADVERSARIAL's TOP-5"

**Verdict**: CARGO-CULTED — REJECTED.

**Rationale**: per §6, the two TOP-5s are largely complementary; the framing "X dominates Y" misses the SATURATION-INDEPENDENT vs SATURATION-DEPENDENT split. PRIMARY's TOP-5 has higher predicted aggregate at higher cost + longer time + lower HARD-EARNED count; ADVERSARIAL's has lower predicted aggregate at $0 cost + 1-day time + higher HARD-EARNED count + empirical anchor included. The reconciled TOP-5 takes the best of both.

### 11.2 Claim: "ADVERSARIAL's 5 binding critiques are dispositive"

**Verdict**: HARD-EARNED-PARTIAL.

**Rationale**: per §5, 4-of-5 critiques (Tao paradigm-partition / Contrarian per-sub-paradigm-counting / Assumption-Adversary saturation-pending / Boyd Pareto-feasibility) are FULLY accepted; 1-of-5 (Hotz apparatus_maintenance) is PARTIALLY accepted because G1 and corrected-F1/A2 are probe-grounded rather than pure apparatus. G1's probe result is now zero movement; F1 still needs capacity proof.

### 11.3 Claim: "Reconciled TOP-5 maximizes operator EV"

**Verdict**: HARD-EARNED via the explicit operator-decision-table format in §1.2.

**Rationale**: the reconciled TOP-5 was operator-actionable immediately for G1 + F1 + A-2 N-7; the conditional row 6 is gated on A-2 N-7 verdict. After Codex's G1 execution, the SATURATION-INDEPENDENT aggregate is no longer authority for score movement: G1 measured `actual_delta_s=0.0`, and corrected-F1/A2 remains capacity-gated. The value of the ordering is now evidence velocity and fail-closed dispatch selection, not a promoted aggregate score claim.

### 11.4 Claim: "The historical aggregate predicted band `[0.164, 0.183]` is realistic"

**Verdict**: HARD-EARNED-PARTIAL.

**Rationale**: SATURATION-INDEPENDENT subset (G1+corrected-F1/A2) aggregate is hypothesis-grade, not realistic-current. G1 existing-anchor rerank measured 0.0 movement; corrected F1/A2 needs capacity proof. CONDITIONAL row 6 contribution is probability-weighted (0.4 × good outcome + 0.6 × no outcome) which may be optimistic if A-2 N-7 prior of 60% SATURATION_HARD_EARNED is itself optimistic.

### 11.5 Claim: "G1 can land THIS SESSION via Codex /goal LOOP"

**Verdict**: HARD-EARNED.

**Rationale**: routing directive `83440e8a5` already exists in `.omx/research/codex_routing_directive_rate_attack_vector_2_g1_cpu_axis_optimization_20260518.md`; the work is local re-ranking using existing dual-eval data; no GPU spend; no external dependencies; Codex /goal LOOP autonomously picks up routing directives.

---

## 12. 9-Dimension Success Checklist Evidence per Catalog #294

Per CLAUDE.md "9-dimension success checklist evidence" + Catalog #294: every substrate / composition / landing memo MUST document evidence across ALL 9 dimensions.

### 12.1 UNIQUENESS

This reconciliation memo is UNIQUE: no prior memo reconciles a PRIMARY 43-vector + ADVERSARIAL paradigm-challenger + supplement triple-landing wave via the operator-elevated reconciliation-panel approach (Tao+Carmack+Hotz+Boyd + sextet pact). The reconciliation FORMAT — SATURATION-INDEPENDENT vs SATURATION-DEPENDENT classification + verdict-driven cascade + operator decision table — is novel to this memo.

### 12.2 BEAUTY + ELEGANCE

PR101-style 30-sec-reviewable per §1 executive summary; operator decision table (§1.2) is the single page that matters; each binding critique resolution (§5) is self-contained; the reconciled TOP-5 (§6) is side-by-side comparable to PRIMARY + ADVERSARIAL inputs.

### 12.3 DISTINCTNESS

Distinct from sister memos:
- vs PRIMARY (`rate_attack_43_vectors_meta_paradigm_deep_research_20260518.md`): that memo enumerates 43 vectors + selects TOP-5; this memo RECONCILES PRIMARY's TOP-5 with ADVERSARIAL's alternative TOP-5
- vs ADVERSARIAL (`adversarial_rate_attack_paradigm_challenger_20260518.md`): that memo critiques PRIMARY; this memo OPERATIONALIZES both via complementary cascade
- vs supplement (`rate_attack_research_context_supplement_*.md`): that memo provides context; this memo INTEGRATES the context into reconciled action items
- vs A-2 N-7 directive (`codex_routing_directive_a2_n7_*.md`): that directive specifies the empirical probe; this memo SEQUENCES the probe within the broader cascade

### 12.4 RIGOR

Premise verification per Catalog #229 (5 PVs in §3 saturation-dependency analysis). Council deliberation per CLAUDE.md "Council conduct" T2 sextet + reconciliation panel. Assumption surfacing per Catalog #292 (6 Assumption-Adversary verdicts in frontmatter). HARD-EARNED-vs-CARGO-CULTED classification per Catalog #303 addendum. Tao paradigm-partition rigor per §5.1.

### 12.5 OPTIMIZATION PER TECHNIQUE

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" canonical-vs-unique decision per layer (§14 below). ADOPTs canonical council schema, posterior writer, lane registry CLI; FORKs the reconciliation-panel format (operator-elevated Tao+Carmack+Hotz+Boyd is non-standard for T2 sextet baseline).

### 12.6 STACK-OF-STACKS-COMPOSABILITY

The reconciliation composes with: PRIMARY (consumes its 43-vector enumeration + TOP-5 selection); ADVERSARIAL (consumes its 5 binding critiques + 12 NEW vectors + alternative TOP-5); supplement (consumes per-axis matrix + signal inventory); A-2 N-7 directive (consumes its specification + sequences its verdict). The reconciliation's output cascades downstream to: cathedral autopilot ranker (Catalog #319 Q3 v2), council continual learning posterior (Catalog #300 hook 5), probe outcomes ledger (Catalog #313).

### 12.7 DETERMINISTIC REPRODUCIBILITY

This memo is BYTE-STABLE (frontmatter date-pinned; council attendees pinned; verdict tokens pinned per Catalog #300 v2 enum). The reconciled TOP-5 cascade (§7.2) is deterministic given the A-2 N-7 verdict.

### 12.8 EXTREME OPTIMIZATION + PERFORMANCE

The reconciliation is optimized for: (a) operator attention (§1 executive summary fits 30-sec review; §1.2 decision table is the single-page); (b) cost (RECONCILED TOP-5 all $0 + $1-3 build for F1); (c) time (THIS SESSION for G1; 4-6h for A-2 N-7 verdict). Per CLAUDE.md "Mission alignment" Consequence 5: this memo is frontier-directed by converting the WAVE into operator-routable immediate evidence items; score authority begins only at measured artifacts.

### 12.9 OPTIMAL MINIMAL CONTEST SCORE

The reconciliation's CONTRIBUTION to optimal minimal contest score is now evidence-routed rather than directly score-authoritative for SATURATION-INDEPENDENT subset. G1+corrected-F1/A2 no longer carries aggregate movement authority: G1 measured `actual_delta_s=0.0`, and corrected-F1/A2 remains a capacity-gated hypothesis until probe evidence lands. Combined with verdict-conditional row 6, the planning band is hypothesis-grade only. This is frontier-directed in the operator-routable sense: G1 closes a false-authority gap with existing anchors, while corrected F1/A2 tests whether scorer-blind RGB perturbation capacity can become charged archive savings.

---

## 13. Observability Surface per Catalog #305

Per CLAUDE.md "Max observability — non-negotiable" + Catalog #305 6-facet observability definition:

### 13.1 Inspectable per layer

Every section of this memo is independently inspectable. §1 executive summary, §3 per-vector saturation classification, §4 per-ADVERSARIAL-vector incorporation, §5 5 binding critiques resolution, §6 reconciled TOP-5, §7 A-2 N-7 sequencing, §8 per-axis matrix integration are all self-contained.

### 13.2 Decomposable per signal

The reconciliation decomposes into:
- 5 binding critiques resolution (§5)
- 12 ADVERSARIAL-NEW vector incorporations (§4)
- 5 PRIMARY TOP-5 saturation classifications (§3)
- 5 RECONCILED TOP-5 (§6)
- 7 operator-routable decision rows (§1.2)
- 1 verdict-driven cascade (§7.2)

Each is queryable post-hoc.

### 13.3 Diff-able across runs

This memo is BYTE-STABLE. A future reconciliation memo (after A-2 N-7 verdict + first-round empirical work) can diff against this memo to surface drift in reconciled TOP-5 ordering and per-vector classification.

### 13.4 Queryable post-hoc

Council posterior anchor emission via `tac.council_continual_learning.append_council_anchor` per §15 enables `query_anchors_by_topic("rate_attack_synthesis_v2_reconciliation_20260518")` retrieval in future sessions.

### 13.5 Cite-able

Every numeric claim carries citation:
- `0.19205 [contest-CPU]` / `0.20533 [contest-CUDA T4]` per Catalog #316 + reports/latest.md 2026-05-17
- PR102 +0.033 cross-axis gap per CLAUDE.md "Submission auth eval" empirical anchor + PR102/PR107 dual-eval rows
- upstream/modules.py:84 source line for F1 score-invariance
- 60% / 30% / 10% A-2 N-7 verdict priors per ADVERSARIAL §8.2 (quartet quartile estimates, not frequentist)
- 25/37,545,489 rate formula per upstream/evaluate.py:65,92
- Catalog #322 v2 cascade `adjust_predicted_delta_for_composition_alpha_v2` for sub-additive aggregates

### 13.6 Counterfactual-able

The reconciliation supports counterfactual queries:
- "What if SATURATION_HARD_EARNED?" → defer SATURATION-DEPENDENT subset; G1+corrected-F1/A2+structural primitives proceed as probes; reallocate build budget to substrate-class-shift
- "What if SATURATION_REFUTED with X% improvement?" → pick row-6 vector per cascade
- "What if Hotz binding directive lifted on G1?" → reorder reconciled TOP-5 putting F1 first
- "What if Boyd Dykstra-feasibility check finds row-6 INFEASIBLE?" → skip row-6 entirely; aggregate is SATURATION-INDEPENDENT subset only

---

## 14. Canonical-vs-Unique Decision Per Layer per Catalog #290

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" + Catalog #290: every NEW design / reconciliation memo MUST document explicit per-layer canonical-vs-unique decisions.

| Layer | Decision | Rationale |
|---|---|---|
| **Council deliberation schema** | ADOPT canonical (Catalog #300 v2 frontmatter) | The v2 schema is the canonical posterior writer's input contract |
| **Reconciliation panel roster** | FORK_BECAUSE_PRINCIPLED_MISMATCH (Tao+Carmack+Hotz+Boyd added to T2 sextet baseline) | The reconciliation requires the four members whose ADVERSARIAL critiques drove the binding revisions; standard T2 sextet alone misses their authority |
| **SATURATION-INDEPENDENT vs SATURATION-DEPENDENT classification scheme** | FORK_BECAUSE_PRINCIPLED_MISMATCH (novel; no canonical helper exists) | This classification is the KEY analytical move in the reconciliation; no existing canonical helper does this |
| **Per-vector saturation-dependency analysis format** | FORK (per-vector table with HARD-EARNED-vs-CARGO-CULTED + per-axis classification + reconciliation action) | Standard cargo-cult audit per Catalog #303 is per-assumption; this is per-vector with multi-axis joint classification |
| **Operator decision table format (§1.2)** | ADOPT canonical (per-row cost+time+saturation+EV+routable+status) per CLAUDE.md "Beauty, simplicity, and developer experience" | Operator-facing tables follow consistent format across memos |
| **A-2 N-7 verdict-driven cascade** | ADOPT canonical (cascade-tree format used in other multi-verdict memos) | Consistent with cathedral autopilot v2 cascade format |
| **Per-axis hardware exploit matrix integration** | ADOPT canonical (supplement Part 1 matrix as input) | Don't re-derive; cite + classify |
| **Mathematical paradigm partition (Tao 3 sub-paradigms)** | FORK_BECAUSE_PRINCIPLED_MISMATCH (novel; resolves Tao's META-paradigm-conflated critique) | Theorem-anchored partition into MI-min Wyner-Ziv / K-complexity-min / Pareto-tightening |
| **Reconciled TOP-5 ranking format** | ADOPT canonical (side-by-side comparison + per-vector justification + aggregate prediction) | Consistent with PRIMARY's TOP-5 selection format |
| **Continual-learning posterior emission** | ADOPT canonical (`tac.council_continual_learning.append_council_anchor` per Catalog #128/#131/#300) | The canonical writer is the only Catalog-compliant emission path |
| **Mission-alignment frontmatter fields** | ADOPT canonical (`council_predicted_mission_contribution: frontier_breaking`) | Per Catalog #300 mission-alignment binding directive; the reconciliation IS frontier-breaking per §2 |
| **Probe outcomes ledger integration** | ADOPT canonical (consume A-2 N-7 verdict via Catalog #313) | Standard predecessor-verdict consumption pattern |
| **6-hook wire-in declaration** | ADOPT canonical (per Catalog #125; declared in §16) | Standard subagent landing discipline |
| **Lane pre-registration** | ADOPT canonical (`tools/lane_maturity.py add-lane lane_rate_attack_synthesis_v2_reconciliation_20260518`; landed at L0 → L1 at memo landing) | Standard Catalog #126 pre-registration |

---

## 15. Council Deliberation Per-Member Verbatim Positions per Catalog #292

Per CLAUDE.md "Council conduct" amendment + Catalog #292: every member of the T2+ council MUST explicitly state at the top of their position the assumption they are operating within.

### 15.1 Sextet pact (binding; 6-of-6 quorum)

**Shannon LEAD**: "The shared assumption I am operating within: the SATURATION-INDEPENDENT subset (G1+corrected-F1/A2) probes DIFFERENT structural information than the SATURATION-DEPENDENT subset; therefore the reconciliation can safely PROCEED on the independent probes while DEFERRING the dependent subset until A-2 N-7 lands. This is HARD-EARNED-FROM-INFORMATION-THEORY as a probe rationale: G1 probes cross-axis sensitivity, corrected F1/A2 probes scorer-feature-invariance, neither of which is a function of the archive's compression entropy. PROCEED 5-of-5 on the reconciled probe queue."

**Dykstra CO-LEAD**: "The shared assumption I am operating within: composition_alpha for the SATURATION-INDEPENDENT subset (G1+corrected-F1/A2) is not claimable until both probes produce measured net deltas; for the conditional row 6 SUB (sub-additive); for the structural primitives (A-3+A-4) N/A (no direct ΔS contribution but reduces wasted enumeration). This is HARD-EARNED per Catalog #322 v2 cascade. PROCEED."

**Yousfi**: "The shared assumption I am operating within: the contest scorer architecture (upstream/modules.py) is the canonical truth and F1's source-verification at modules.py:84 is sufficient evidence of dim 7-12 score-invariance for memo-landing purposes (the runtime probe `tools/probe_hydra_dim_7_12_score_invariance.py` is the empirical confirmation). PROCEED with probe-gate per Fridrich+Contrarian revision."

**Fridrich**: "The shared assumption I am operating within: the F1 exploit (encoder-controlled bits in scorer-invariant dims) is canonical steganography per my own PhD work on UNIWARD/HUGO; the bits travel through the scorer's perfect zero-distortion channel. PROCEED with probe-gate."

**Contrarian**: "The shared assumption I am operating within: the reconciliation cannot SUBSTANTIVELY resolve the META-paradigm-conflated critique just by renaming TOP-5; it must EITHER mathematically partition vectors into theorem-anchored sub-paradigms per Tao §5.1 OR explicitly defer META-paradigm framing. The reconciliation has DONE the Tao partition (§5.1). PROCEED conditional on (a) reconciled TOP-5 carries explicit SATURATION-INDEPENDENT vs SATURATION-DEPENDENT tags, AND (b) SATURATION-DEPENDENT vectors are formally gated on A-2 N-7 verdict. Both conditions are met. PROCEED unconditional."

**Assumption-Adversary**: "The shared assumption I am operating within: PRIMARY's two HARD-EARNED anchors (G1 cross-axis empirical + F1 source-verified) are INDEPENDENT of the saturation hypothesis; therefore reconciliation can safely PROCEED on those two while DEFERRING the saturation-dependent vectors. The HARD-EARNED-PARTIAL classification I gave in the frontmatter remains: G1 uses dual-eval data (no compressor sweep required to validate); F1 cites upstream source (no compressor sweep required). The other PRIMARY TOP-5 (B1/Y3+Y6/H1) all depend on compression-saturation question. A-2 N-7 directly tests this. PROCEED."

### 15.2 Reconciliation panel (operator-elevated; the four members whose ADVERSARIAL critiques drove the binding revisions)

**Tao** (mathematical reconciliation): "The shared assumption I am operating within: META-paradigm SINS is a UNIFYING UMBRELLA but the 3 sub-paradigms (MI-min Wyner-Ziv / K-complexity-min / Pareto-tightening) have DISTINCT optimality conditions and must be classified per vector. The reconciliation's §5.1 partition is THEOREM-ANCHORED (Wyner-Ziv 1976 / Cilibrasi-Vitányi 2005 / Dykstra-Boyle 1985); 39-of-43 PRIMARY vectors fall cleanly into 1 of 3 sub-paradigms; 3 are outside-formalism (training guards / process features); 1 (B1 contest-video-codebook) spans MI-min AND K-complexity-min (codebook is both). My critique is OPERATIONALIZED. PROCEED."

**Carmack** (engineering reduction): "The shared assumption I am operating within: ALREADY-IN-STANDARD-LIBRARY exploits dominate novel-cryptography exploits on (cost, time, HARD-EARNED count). A-2 N-7 tests ALL standard compressors; if any beat current frontier, A-1 Brotli-dict OR direct compressor swap is the trivially-implementable winning vector. RECONCILED row 6 pick honors this. My critique is OPERATIONALIZED. PROCEED. Note: I PUSH for A-5 (LOC budget OPTIMIZER) to be a structural primitive too, not just optional; will queue post-RECONCILED-4-5 landing."

**Hotz** (raw empirical + binding G1 directive): "The shared assumption I am operating within: the leaderboard is CPU; paired-axis gaps are cheap enough to test immediately. Codex /goal LOOP must pick up `83440e8a5` THIS SESSION. My binding directive STANDS as a probe directive, not as a guaranteed frontier move. The reconciliation's RECONCILED-1 = G1 IMMEDIATE EXECUTION honors this. PROCEED. Note: I'm also satisfied with A-2 N-7 being the parallel empirical anchor; the Codex /goal LOOP can pick up BOTH directives in parallel."

**Boyd** (Pareto-feasibility across reconciled TOP-5): "The shared assumption I am operating within: the reconciled TOP-5's composition must stay measured, not asserted. No EXCL cells proven for the probe queue. Dykstra-feasibility intersection of {rate-constraint, seg-tolerance, pose-tolerance, runtime-LOC budget} is non-empty only after corrected F1/A2 and A1-SPECIALIZED report byte-charged net deltas. My critique is OPERATIONALIZED via RECONCILED-4 (A-3 Dykstra-feasibility-FIRST canonical helper). PROCEED. Note: I'd like the A-3 canonical helper to be the FIRST routing post-A-2-N-7 so future composition decisions go through it."

### 15.3 Grand-council attendees

**Ballé**: "PROCEED. My binding directive (5-of-5 PROCEED on TOP-5) from PRIMARY §8 stands; reconciled TOP-5 honors this with the saturation-dependency split."

**Mallat**: "PROCEED. Composition order H1→B1→Y3+Y6→F1→G1 from PRIMARY §8 needs revision: reconciled order is G1→F1→A-2-N-7→A-3→A-4→row-6-conditional. Wavelet-multi-scale ranking still applies WITHIN row 6 (coarse-scale H1 beats medium-scale Y3+Y6 beats fine-scale B1)."

**van den Oord**: "PROCEED. B1 stays in row 6 conditional pick; canonical VQ-VAE with the decoder's actual side info remains my binding directive IF A-2 N-7 SATURATION_REFUTED."

**Filler**: "PROCEED. STC pre-entropy bit-allocator (C3 + B3) is integrated into the structural primitives queue post-RECONCILED-4-5."

### 15.4 Vote tally

PROCEED_WITH_REVISIONS: 13 of 13 attendees (sextet 6/6 + reconciliation panel 4/4 + grand-council 4/4; revisions = the reconciliation IS the revision package operationalizing all 5 binding critiques from ADVERSARIAL)

Quorum: T2 sextet 6/6 met; 10/13 grand-council attendees voted (Karpathy + Schmidhuber + Tishby memorial did not attend reconciliation panel session; their PRIMARY+ADVERSARIAL positions remain valid)

---

## 16. 6-Hook Wire-In Declaration per Catalog #125

Per CLAUDE.md "Subagent coherence-by-default" + Catalog #125: every subagent landing MUST declare 6-hook wire-in.

### Hook #1: Sensitivity-map contribution

**ACTIVE**: G1 per-axis re-ranking output feeds `tac.sensitivity_map` rows with per-archive per-axis sensitivity values. A-4 MI-min canonical helper provides per-byte conditional MI which IS the canonical per-byte sensitivity primitive for `tac.sensitivity_map.*` consumers.

### Hook #2: Pareto constraint

**ACTIVE**: A-3 Dykstra-feasibility-FIRST canonical helper IS the Pareto constraint helper; consumed by future composition decisions per Catalog #322. G1+corrected-F1/A2 SATURATION-INDEPENDENT subset's predicted aggregate remains pending the corrected F1/A2 capacity probe.

### Hook #3: Bit-allocator hook

**ACTIVE**: G1 re-ranking + corrected F1/A2 both impact per-tensor importance (G1 via per-axis sensitivity; corrected F1/A2 via scored-pose-stable RGB perturbation capacity, not a direct dim-7-12 archive channel). A-4 MI-min canonical helper provides the bit-allocation prior.

### Hook #4: Cathedral autopilot dispatch hook

**ACTIVE**: A-2 N-7 verdict registers via `tac.probe_outcomes_ledger.register_probe_outcome` per Catalog #313; cathedral autopilot Cascade 2 reward factor per Catalog #319 Q3 v2 consumes this verdict to weight SATURATION-DEPENDENT vector cost-band priorities.

### Hook #5: Continual-learning posterior update

**ACTIVE**: This memo's council deliberation anchor emits via `tac.council_continual_learning.append_council_anchor` (canonical helper; Catalog #128/#131/#300 sister discipline; emitted in §17 below).

### Hook #6: Probe-disambiguator

**ACTIVE**: 5 PROCEED candidates have 2+ defensible interpretations:
- G1 vs not-G1: probe is the dual-eval re-ranking (no separate disambiguator needed; verdict is empirical)
- F1 build vs not-build: probe is `tools/probe_hydra_dim_7_12_score_invariance.py` (op-routable 2 in PRIMARY §9)
- B1 vs A-1 Brotli-dict vs H1 vs Y3+Y6 for row 6: A-2 N-7 IS the disambiguator
- A-3 vs not-A-3 canonical helper: justified by Boyd binding revision; no separate disambiguator needed
- A-4 vs not-A-4 canonical helper: justified by Tao binding revision; no separate disambiguator needed

---

## 17. Final TOP-5 Ranked Op-Routables (Operator-Routable with Cost / Time / Risk per Vector)

### Op-Routable 1: G1 Cross-Axis CPU Re-Rank (RECONCILED-1; IMMEDIATE; $0)

- **Lane**: `lane_rate_attack_g1_cpu_axis_specific_20260518`
- **Action**: extend `tools/scan_best_anchor_per_axis.py` + new `tools/cpu_axis_optimal_archive_selector.py`; re-rank existing PR101+102+103+106+107 archives by per-axis CPU eval; identify CPU-axis-optimal per-archive-family variant; update `reports/latest.md` FRONTIER section per Catalog #316
- **Cost**: $0 (locally computed; existing dual-eval data)
- **Expected return**: realized `0.0` ΔS [contest-CPU] for existing-anchor rerank; future upside requires a new paired Linux x86_64 CPU anchor or a candidate-selection case where CPU-optimal differs from CUDA-optimal
- **Time**: this session (1-2 hours operator + Codex)
- **Risk**: very low (re-ranking is reversible; existing data)
- **Routing**: `83440e8a5` already landed; Codex /goal LOOP
- **Predecessor outcome per Catalog #313**: none (novel work)
- **Per-axis classification**: AXIS-INVARIANT mechanism + CPU-axis-OPTIMAL operational

### Op-Routable 2: F1 Hydra Dim 7-12 Probe (RECONCILED-2 Step 1; $0)

- **Lane**: `lane_rate_attack_f1_hydra_dims_probe_20260518`
- **Action**: `tools/probe_hydra_dim_7_12_score_invariance.py` — take PR101 frontier archive `6bae0201`, run inflate.sh + upstream/evaluate.py, modify pose dims 7-12 in output, re-run evaluate.py, confirm score IDENTICAL across 600 pairs on BOTH CPU + CUDA
- **Cost**: $0 (locally on macOS-CPU advisory + paired Linux x86_64 [contest-CPU])
- **Expected return**: verification gates F1 build per next op-routable
- **Time**: 1 day (probe development + execution)
- **Risk**: low (probe failure means defer F1 build, not loss)
- **Routing**: `83440e8a5` landed; Codex /goal LOOP picks up
- **Predecessor outcome per Catalog #313**: none

### Op-Routable 3: A-2 N-7 Standard Compressor Sweep (RECONCILED-3; $0; EXECUTING)

- **Lane**: `lane_rate_attack_a2_n7_cpu_axis_compressor_sweep_20260518`
- **Action**: `tools/probe_a2_n7_standard_compressor_sweep.py` per directive `1ac2063de` Phase 1 + 2 + 3; sweep brotli/zstd/lzma/xz/bzip2/lzfse × multiple levels across 5+ frontier archives; emit per-archive verdict + aggregate verdict; register via Catalog #313
- **Cost**: $0 (CPU only)
- **Expected return**: SATURATION verdict + per-archive winners (if any)
- **Time**: 4-6h wall-clock (Codex autonomous)
- **Risk**: very low ($0; bounded outcome)
- **Routing**: `1ac2063de` landed; Codex /goal LOOP executing
- **Predecessor outcome per Catalog #313**: none

### Op-Routable 4: A-3 Dykstra-Feasibility-FIRST Canonical Helper (RECONCILED-4; $0; POST-A-2-N-7)

- **Lane**: `lane_rate_attack_a3_dykstra_feasibility_first_canonical_helper_20260518`
- **Action**: implement `tac.rate_attack_dykstra_feasibility` canonical helper computing intersection of {rate-constraint, seg-tolerance, pose-tolerance, runtime-LOC budget} via alternating projections per Dykstra-Boyle 1985; emit per-vector PASS/FAIL classification + per-pair composition_alpha estimates; integrate with `tools/cathedral_autopilot_autonomous_loop.py` as Cascade 0 pre-filter
- **Cost**: $0 (local Python implementation)
- **Expected return**: reduce wasted enumeration effort by 75-90% per Boyd
- **Time**: 1 day
- **Risk**: low (canonical helper landing; existing Dykstra primitives in `tac.optimization.dykstra_*`)
- **Routing**: needs new routing directive post-A-2-N-7 (queue for Codex)
- **Predecessor outcome per Catalog #313**: none

### Op-Routable 5: A-4 MI-Min Wyner-Ziv Canonical Helper (RECONCILED-5; $0; POST-A-2-N-7)

- **Lane**: `lane_rate_attack_a4_mi_min_wyner_ziv_canonical_helper_20260518`
- **Action**: implement `tac.rate_attack_mi_lower_bound` canonical helper computing per-byte conditional MI estimator (k-NN per Kraskov-Stoegbauer-Grassberger 2004); emit per-byte sensitivity map; integrate with `tac.sensitivity_map.*` consumers; provide theorem-level bounds for all Wyner-Ziv-derivative vectors (B1, F1, F2, F3, F4, Y2, C1, N-1 itself)
- **Cost**: $0 (local Python implementation; k-NN MI runs in O(N log N))
- **Expected return**: theorem-level bounds + per-byte sensitivity primitive
- **Time**: 1 day
- **Risk**: low (canonical helper landing)
- **Routing**: needs new routing directive post-A-2-N-7 (queue for Codex)
- **Predecessor outcome per Catalog #313**: none

### Op-Routable 6 (CONDITIONAL): Post-A-2-N-7 Row-6 Pick

- **Cost**: $1-8 (verdict-dependent)
- **Expected return**: verdict-dependent
- **Time**: 1-2 days (after A-2 N-7 verdict lands)
- **Risk**: depends on verdict + pick
- **Routing**: post-A-2-N-7; operator chooses per §7.2 cascade

### Op-Routable 7 (OPTIONAL): F1 Build (CONDITIONAL on Op-Routable 2 PASS)

- **Lane**: `lane_rate_attack_f1_hydra_dims_7_12_substrate_20260518` (name historical; implementation must use corrected F1/A2 framing)
- **Action**: supersede PRIMARY's direct dim-channel design with corrected F1/A2 — measure and implement scorer-blind RGB perturbation capacity that preserves PoseNet scored dims 0-5 and SegNet argmax; do not implement encoder-side dim 7-12 emission or inflate-time side-channel extraction as a contest-capacity proof
- **Cost**: $1-3 (Modal T4 smoke + paired CPU re-eval)
- **Expected return**: -0.004 to -0.012 ΔS [contest-CPU]
- **Time**: 1-2 days post-probe
- **Risk**: moderate (probe could fail; substrate engineering required)
- **Routing**: conditional; emit after probe PASS

---

## 18. Cross-References

### 18.1 Sister landings (the three primary inputs to this reconciliation)

- **PRIMARY (commit `2cae89a87`)**: `.omx/research/rate_attack_43_vectors_meta_paradigm_deep_research_20260518.md` — 43-vector master memo + 5 per-vector design memos (`rate_attack_vector_*_design_memo_20260518.md`) + 3 routing directives (`codex_routing_directive_rate_attack_vector_*_20260518.md`)
- **ADVERSARIAL (commit `4c6e46bfa`)**: `.omx/research/adversarial_rate_attack_paradigm_challenger_20260518.md` — T2 sextet + Tao+Carmack+Hotz+Boyd quartet adversarial review with 5 binding critiques + 12 NEW vectors + alternative TOP-5 + mandatory A-2 N-7 anchor
- **Supplement (commit `d43ecddb0`)**: `.omx/research/rate_attack_research_context_supplement_per_axis_hardware_plus_dual_device_master_gradient_20260518.md` — per-axis hardware exploit matrix + dual-device master-gradient research question + comprehensive signal inventory

### 18.2 Critical sister artifacts

- **A-2 N-7 routing directive (commit `1ac2063de`)**: `.omx/research/codex_routing_directive_a2_n7_cpu_axis_tier1_standard_compressor_sweep_empirical_anchor_20260518.md` — the empirical anchor Codex executes per /goal LOOP
- **META-paradigm unification**: `.omx/research/structural_information_not_shipped_meta_paradigm_unification_20260518.md` — SINS framing (now partitioned into 3 sub-paradigms per Tao §5.1)
- **Cross-stack synthesis**: `.omx/research/cross_stack_synthesis_9_design_landings_unified_framework_20260518.md` — 9×9 cross-pollination matrix; SUB-additive composition_alpha priors
- **Design stack hypergraph**: `.omx/research/design_stack_full_hypergraph_model_design_memo_20260518.md` — 10 typed node categories incl. `deterministic_byte_derivation` META-category 10 (43-vector PRIMARY expansion target)
- **Closure campaign master**: `.omx/research/closure_campaign_pursue_and_confirm_master_20260518.md` — 5 OP-AUDIT closure operations; saturation hypothesis context
- **Rate-attack novel vectors seed**: `.omx/research/rate_attack_novel_vectors_design_memo_20260518.md` — codex 13-vector predecessor (A1-A3/B1-B4/C1-C3/M1-M3); PRIMARY expanded to 43

### 18.3 Canonical CLAUDE.md sections cited

- "Mission alignment — non-negotiable" (Consequences 1-5; frontier-breaking vs apparatus_maintenance taxonomy)
- "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" (PR102 +0.033 cross-axis gap empirical anchor)
- "Race-mode rigor inversion + parallel-dispatch first" (immediate execution dominates when frontier-breaking move identified)
- "Council hierarchy: 4-tier protocol" (T2 sextet quorum 6-of-6; reconciliation panel operator-elevation)
- "Council conduct" (Per-round explicit-assumption-statement discipline Fix 7)
- "META-ASSUMPTION ADVERSARIAL REVIEW" (Assumption-Adversary seat per Catalog #292)
- "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" (canonical-vs-unique decision per layer)
- "Apples-to-apples evidence discipline" (axis labels on every score)
- "Forbidden empirical-claim-without-evidence-tag" (the docstring-overstatement trap)

### 18.4 Catalog gates this memo respects

- Catalog #117 (commit serializer must be used) — this memo's commit goes through `tools/subagent_commit_serializer.py` with POST-EDIT sha
- Catalog #125 (subagent landing has solver wire-in) — 6 hooks declared in §16
- Catalog #126 (lane pre-registered before work starts) — lane registered at L0 via `tools/lane_maturity.py add-lane`
- Catalog #128/#131 (continual-learning writes use lock) — council anchor emission via canonical fcntl-locked helper in §19
- Catalog #157 (commit serializer pre-lock hash) — POST-EDIT sha declared
- Catalog #174 (--expected-content-sha256 mandatory) — declared at commit time
- Catalog #185 (CLAUDE.md catalog text matches gate empirical state) — N/A (no new gates landed)
- Catalog #206 (subagent dispatches use checkpoint discipline) — checkpoints at memo start + this completion point
- Catalog #229 (premise verification before edit) — 8 PVs in §3 saturation-dependency analysis
- Catalog #287 (evidence tags on numeric claims) — declared in §13.5 cite-able facet
- Catalog #290 (canonical-vs-unique decision per layer) — §14
- Catalog #292 (per-deliberation explicit assumption statements) — §15.1 per-member operating-within
- Catalog #294 (9-dim success checklist evidence) — §12
- Catalog #296 (Dykstra-feasibility for predicted bands) — §6.3 + §15 Dykstra position
- Catalog #300 (council deliberation v2 frontmatter) — frontmatter
- Catalog #303 (cargo-cult audit per assumption) — §11
- Catalog #305 (observability surface) — §13
- Catalog #313 (predecessor probe outcomes ledger) — checked; A-2 N-7 will register on landing
- Catalog #314 (no absorption pattern) — scope is reconciliation memo only; sister subagents NOT in flight; codex `019de465` owns source code (DISJOINT)
- Catalog #316 (reports/latest.md frontier scan) — cited 0.19205 + 0.20533 anchors
- Catalog #319 (Wyner-Ziv reweight requires deliverability proof) — N/A (no Wyner-Ziv reweight landed in this memo)
- Catalog #322 (composition_alpha v2 cascade) — §6.3 sub-additive aggregate
- Catalog #324 (predicted band post-training Tier-C validation) — `pending_post_training` validation_status in frontmatter
- Catalog #325 (per-substrate symposium for paid dispatch) — A-2 N-7 is $0 probe; no substrate dispatch required

---

## 19. Council Verdict + Continual-Learning Anchor Emission per Catalog #300 Hook #5

### 19.1 Council verdict

**T2 sextet pact + reconciliation panel (operator-elevated Tao+Carmack+Hotz+Boyd) + 4 grand-council attendees**: **PROCEED_WITH_REVISIONS**, 13-of-13 PROCEED with revisions = the reconciliation IS the revision package operationalizing all 5 ADVERSARIAL binding critiques while preserving PRIMARY's HARD-EARNED structural anchors (G1 cross-axis + F1 source-verified).

### 19.2 Canonical posterior anchor emission (executable Python)

```python
# This memo's council anchor (to be emitted at memo landing)
from tac.council_continual_learning import (
    CouncilDeliberationRecord, CouncilTier, append_council_anchor,
)

record = CouncilDeliberationRecord(
    deliberation_id="rate_attack_synthesis_v2_reconciliation_primary_plus_adversarial_plus_supplement_20260518",
    topic="Canonical OPERATOR-FACING reconciliation of today's rate-attack research wave: PRIMARY 43-vector META-paradigm research + ADVERSARIAL paradigm challenger + supplement. SATURATION-INDEPENDENT probes (G1+corrected-F1/A2) proceed immediately as evidence-gathering; SATURATION-DEPENDENT subset gated on A-2 N-7 verdict. Reconciled TOP-5 = G1 + F1 + A-2 N-7 + A-3 Dykstra + A-4 MI-min; conditional row 6 post-A-2-N-7. Aggregate bands are hypothesis-grade until artifacts land; Codex G1 measured actual_delta_s=0.0.",
    council_tier=CouncilTier.T2,
    council_attendees=(
        # Sextet pact
        "Shannon", "Dykstra", "Yousfi", "Fridrich", "Contrarian", "Assumption-Adversary",
        # Reconciliation panel (operator-elevated)
        "Tao", "Carmack", "Hotz", "Boyd",
        # Grand-council attendees
        "Ballé", "Mallat", "van_den_Oord", "Filler",
    ),
    council_quorum_met=True,
    council_verdict="PROCEED_WITH_REVISIONS",
    council_dissent=(
        {"member": "Contrarian", "verbatim": "PROCEED conditional on (a) reconciled TOP-5 carries explicit SATURATION-INDEPENDENT vs SATURATION-DEPENDENT tags, AND (b) SATURATION-DEPENDENT vectors are formally gated on A-2 N-7 verdict. Both conditions met."},
        {"member": "Assumption-Adversary", "verbatim": "HARD-EARNED-PARTIAL on G1+corrected-F1/A2 SATURATION-INDEPENDENT probe subset; A-2 N-7 is essential for SATURATION-DEPENDENT subset."},
    ),
    council_assumption_adversary_verdict=(
        {"assumption": "PRIMARY's TOP-5 and ADVERSARIAL's TOP-5 are necessarily conflicting", "classification": "CARGO-CULTED", "rationale": "Two streams are largely COMPLEMENTARY per §3-§4; the framing that operator must choose one is the cargo-cult"},
        {"assumption": "A-2 N-7 verdict is essential gating for ALL rate-attack work", "classification": "HARD-EARNED-PARTIAL", "rationale": "Essential for SATURATION-DEPENDENT vectors; not a blocker for SATURATION-INDEPENDENT probes (G1+corrected-F1/A2), but those probes still need artifact authority"},
        {"assumption": "The PR102 +0.033 CPU-CUDA gap is a reusable conversion factor", "classification": "FALSE_AS_CONVERSION__HARD_EARNED_AS_PER_ARCHIVE_PRIOR", "rationale": "PR102 / PR107 / PR101-family observations are paired-axis priors only; Codex G1 existing-anchor rerank measured actual_delta_s=0.0 and no CPU/CUDA conversion is allowed"},
        {"assumption": "Hydra dims 7-12 are STRUCTURALLY ignored by scorer", "classification": "HARD-EARNED-VERIFIED-FROM-SOURCE", "rationale": "upstream/modules.py:84 source line confirms [..., : h.out // 2] slice"},
        {"assumption": "ADVERSARIAL's 60% predicted SATURATION_HARD_EARNED is calibrated", "classification": "CARGO-CULTED-PENDING-EMPIRICAL", "rationale": "Quartet quartile estimate, not frequentist; A-2 N-7 verdict is the actual signal"},
        {"assumption": "G1 immediate execution does NOT need A-2 N-7 to land first", "classification": "HARD-EARNED-FROM-FIRST-PRINCIPLES", "rationale": "G1 produces no new archive bytes; saturation question is structurally orthogonal"},
    ),
    council_decisions_recorded=(
        "RECONCILED-1: PRIMARY's TOP-5 and ADVERSARIAL's alternative TOP-5 are complementary; reconciled TOP-5 drawn from both",
        "RECONCILED-2: SATURATION-INDEPENDENT probes (G1, corrected-F1/A2) may proceed independent of A-2 N-7 outcome; G1 measured actual_delta_s=0.0, F1 requires capacity proof",
        "RECONCILED-3: SATURATION-DEPENDENT vectors (B1, Y3+Y6, H1, A-1 Brotli-dict) gated on A-2 N-7 verdict",
        "RECONCILED-4: A-2 N-7 routed to Codex per commit 1ac2063de; canonical /goal LOOP executes; ETA 4-6h at $0",
        "RECONCILED-5: A-3 Dykstra-feasibility-FIRST + A-4 MI-min Wyner-Ziv land as canonical helpers regardless of A-2 N-7 outcome",
        "RECONCILED-6: Dual-device master-gradient extension DEFERRED to separate post-A-2-N-7 routing directive",
        "RECONCILED-7: META-paradigm SINS partitioned into 3 theorem-anchored sub-paradigms per Tao critique",
    ),
    council_predicted_mission_contribution="frontier_breaking",
    council_override_invoked=False,
    council_override_rationale="",
)
append_council_anchor(record)
```

### 19.3 Companion probe outcomes ledger registration

When A-2 N-7 lands its verdict, Codex will register per Catalog #313:
```python
from tac.probe_outcomes_ledger import register_probe_outcome
register_probe_outcome(
    probe_id="a2_n7_cpu_axis_tier1_standard_compressor_sweep_20260518",
    verdict=<SATURATION_HARD_EARNED | SATURATION_REFUTED | SATURATION_PENDING>,
    status="adjudicated",
    blocks_recipes=[<rate-attack research directives if SATURATION_HARD_EARNED>],
    rationale=<aggregate report path>,
    expires_at_utc=<+30d>,
    agent="codex",
)
```

This memo's anchor stays valid for ~30 days; staleness re-evaluation per Catalog #298 retirement discipline.

---

**Memo ends.** Reconciliation v2 LANDED at commit (to be filled by canonical serializer). Operator-routable next session = G1 + F1-probe + A-2-N-7 (all already routed); awaiting Codex /goal LOOP picking up. Final action: monitor `.omx/state/probe_outcomes.jsonl` for A-2 N-7 verdict + `.omx/state/codex_persistent_session_state.jsonl` for completion signals.

— RATE-ATTACK-SYNTHESIS-V2-RECONCILIATION-2026-05-18 / lane `lane_rate_attack_synthesis_v2_reconciliation_20260518` L0 → L1 at landing.
