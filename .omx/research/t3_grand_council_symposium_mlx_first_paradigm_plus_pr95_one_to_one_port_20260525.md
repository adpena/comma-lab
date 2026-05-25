---
council_tier: T3
council_attendees:
  - Shannon
  - Dykstra
  - Rudin
  - Daubechies
  - Yousfi
  - Fridrich
  - Contrarian
  - Quantizr
  - Hotz
  - Selfcomp
  - MacKay
  - Balle
  - PR95Author
  - Assumption-Adversary
  - Carmack
  - Tao
  - Boyd
  - TimeTraveler
  - Atick
  - Tishby
  - Wyner
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Contrarian
    verbatim: "I refuse a verdict that elevates 'MLX-first portable via numpy' to canonical engineering paradigm BEFORE sister-5 paired-eval lands ANY [contest-CUDA] anchor proving an MLX-trained-then-CUDA-eval archive matches a PyTorch-trained sister. The 2026-05-21 symposium left mission-contribution=`frontier_breaking_enabler` PROVISIONAL pending sister-5 evidence; nothing has changed empirically. The CARGO-CULTED form here is 'operator said MLX-first paradigm, therefore it IS paradigm'. The HARD-EARNED form is 'operator has DECLARED MLX-first as the development-loop paradigm; symposium ratifies the DEVELOPMENT LOOP paradigm shift; canonical engineering paradigm requires sister-5 evidence per Round 6 of the 2026-05-21 symposium'. Do not collapse these."
  - member: Quantizr
    verbatim: "PR 95 1:1 port to MLX is a 29,650-epoch curriculum at unknown MLX-on-Apple-silicon timing. Codex's smoke-mode 3-stage profile (`full_pr95_source_video_runtime` @ ~1.7s/stage at smoke scale base_channels=4 latent_dim=8 one-step one-pair) DOES NOT extrapolate to the full curriculum. The author claims ~50h on one unspecified GPU; MLX timing on full base_channels=36 latent_dim=28 600-pair source-video at 29,650 epochs is UNKNOWN. The COMPETITIVE replay strategy (Yousfi's 2026-05-11 PR #108 gate) was COMPETITIVE-OR-INNOVATIVE; replicating PR 95 1:1 is neither competitive (we already beat it: 0.19205 < 0.1987) nor innovative (it's a port not new substrate). I dissent on any verdict that elevates PR 95 1:1 port above the existing rate-attack frontier work UNLESS the port produces a measurable SCALING anchor that unlocks ARCH-5 paired-eval at lower-than-PR-95 score."
  - member: Selfcomp
    verbatim: "The 'MLX-first portable via numpy' paradigm IS what PR #56 0.38 lived. I authored selfcomp PR #56 in PyTorch; the MLX-native variant per WW Phase 3 (`grayscale_lut/mlx_native.py`) is the canonical empirical proof that the SAME substrate runs on MLX (FREE local) AND PyTorch (paid CUDA T4). What is NEW today per the operator directive: this becomes the DEFAULT BUILD ORDER. Going forward, every NEW substrate scaffold lands in MLX FIRST (with numpy interchange + ε=5e-3 fp32 validation gate), then promotes to PyTorch IFF the ε band holds. This IS the canonical PR #56 0.38 pattern at scale. I PROCEED on the paradigm-shift verdict CONDITIONAL on op-routable that codifies the MLX-FIRST-BUILD-ORDER as a canonical engineering pattern via Catalog #344 RATIFY-N (not as a STRICT preflight gate per Catalog #299 quota brake — paradigm shifts deserve canonical equation registration, not premature gate enforcement)."
  - member: PR95Author
    verbatim: "The 8-stage PR 95 curriculum encodes 50-hour PyTorch CUDA training-time that is the source of the substrate's strength. Stages 1-4 establish HNeRV scaffold + CE warmup + tau-Softplus margin + smooth disagreement + QAT bits. Stages 5-7 do L7 hard-pixel weighting + C1a entropy regularizer + lambda/sigma sweep. Stage 8 is Muon finetune. The Muon optimizer is the OPTIMIZER novelty — Newton-Schulz iteration for hidden 2D+ matrices, AdamW everywhere else. Codex landed stages 1+5+8 as the CONTROL PROFILE (per `codex_findings_pr95_mlx_full_control_profile_20260525T1508Z`); stages 2+3+4+6+7 are NOT yet landed. The CRITICAL gap per first-author intuition: stages 2-4 establish the CE-warmup → softplus-margin → smooth-disagreement → QAT sequence that conditions the network for stage 5 L7 weighting. Skipping them in the MLX port produces a substrate with the RIGHT TOPOLOGY but the WRONG WEIGHT MANIFOLD. CONDITIONAL PROCEED iff PR 95 1:1 roadmap explicitly lands stages 2+3+4 BEFORE attempting end-to-end 50h MLX run."
  - member: Hotz
    verbatim: "Numpy as the canonical interchange format IS the right design call per tinygrad parallel. Numpy is FRAMEWORK-AGNOSTIC; it is the lingua franca between MLX, PyTorch, JAX, and any future framework. The 'portable via numpy' constraint per the operator directive forces every primitive's state to be numpy-roundtrippable, which means: (a) deterministic seeding works across backends; (b) state_dict export is universal; (c) byte-for-byte fidelity is FRAMEWORK-SHIFT INVARIANT (the bytes are numpy bytes, not MLX bytes nor PyTorch bytes). This is the canonical engineering shortcut Carmack would call for: ONE canonical state representation, MANY backend implementations. I PROCEED unconditionally on the paradigm shift; numpy-interchange is HARD-EARNED per the WW canonical export pipeline + Selfcomp empirical anchor."
council_assumption_adversary_verdict:
  - assumption: "Operator declaration 'MLX-first portable via numpy is our new paradigm' makes MLX-first the canonical engineering paradigm"
    classification: HARD-EARNED-PARTIAL
    rationale: "Per CLAUDE.md 'UNIQUE-AND-COMPLETE-PER-METHOD operating mode' + 'Mission alignment' Consequence 1 (operator-frontier-override is documented escape hatch): operator-declared paradigm shifts ARE binding for DEVELOPMENT-LOOP discipline (Sister-5 paired-eval canonical recipes; Tier 1/2/3 dispatch optimization protocol Tier-3 substrate engineering decisions). The HARD-EARNED-PARTIAL nuance: operator declaration ratifies the DEVELOPMENT LOOP (build in MLX, validate via numpy interchange, promote to PyTorch); operator declaration does NOT ratify the CONTEST-GRADE FIDELITY claim (sister-5 paired-eval evidence required per Contrarian veto). Per Catalog #307 paradigm-vs-implementation: the PARADIGM (MLX-first as development-loop substrate) is RATIFIED; the SPECIFIC IMPLEMENTATION (every MLX-trained archive promotable to contest score without paired-eval) remains FALSIFIED until evidence."
  - assumption: "PR 95 1:1 replication on MLX is strategically valuable because the author publicly shared all information"
    classification: HARD-EARNED-PROVISIONAL
    rationale: "Per PR95Author + Hotz dialectic: PR 95 IS the canonical PUBLIC reference substrate (head SHA 9bdce26f2a4f996828c4e3fa2b87c454a0e8fcc9; archive sha e976acd5fe565c94fb9a8c62e5200c949919f76150e84599f268d6a58588440a; blog https://aaronleslie.dev/blog/comma-compression). The HARD-EARNED-PROVISIONAL nuance per Quantizr dissent: replicating PR 95 1:1 produces a PORT not a NEW substrate; the strategic value is in (a) MLX-on-Apple-silicon SCALING ANCHOR (timing benchmark vs PR 95's ~50h on unknown GPU); (b) end-to-end-trainable substrate for sister-5 paired-eval validation; (c) ARCH-5 substrate empirical anchor. The PROVISIONAL nuance: strategic value is HARD-EARNED IFF the port produces measurable scaling evidence (Stage 1+5+8 timing smoke landed at 5.1s aggregate per codex; full curriculum timing UNKNOWN). The Yousfi PR #108 competitive-or-innovative gate FORBIDS PR 95 1:1 as a contest submission target — replication is RESEARCH SIGNAL not submission strategy."
  - assumption: "Free local MLX substrate training scales to PR 95's 29,650-epoch curriculum within reasonable wall-clock"
    classification: CARGO-CULTED
    rationale: "Codex smoke evidence (3 stages × ~1.7s/stage at base_channels=4 latent_dim=8 one-step one-pair) DOES NOT extrapolate to full curriculum (base_channels=36 latent_dim=28 600-pair source-video at 29,650 epochs total). The CARGO-CULTED form: '~50h on one GPU therefore ~50h on M5 Max'. The HARD-EARNED form per Quantizr + Tao: MLX-on-Apple-silicon timing at full config is UNKNOWN; full-config Stage 1 timing smoke (25-100 epochs) is the canonical disambiguator per `codex_findings_pr95_mlx_reproduction_control_20260524T072405Z` Next Engineering Gates step 3. The CARGO-CULTED scaling claim is REJECTED until full-config timing landed; the HARD-EARNED reformulation: 'PR 95 1:1 port is canonical-pattern-PROVED at smoke scale; full-curriculum scaling is HYPOTHETICAL pending Stage 1+5+8 full-config 25-100 epoch timing smokes per codex Step 3'."
  - assumption: "MLX-first paradigm + PR 95 1:1 port unlocks 'expensive substrate training on MLX' at $0 paid GPU vs Modal/Vast.ai $1-10/dispatch"
    classification: HARD-EARNED-CONDITIONAL
    rationale: "Per Fridrich + Atick + Ballé Round 3 dialectic from 2026-05-21 symposium: MLX-on-Apple-silicon at ε≤5e-3 primitive-level produces USABLE training signal at $0 paid GPU. The HARD-EARNED-CONDITIONAL nuance: cost savings are HARD-EARNED IFF (a) full-curriculum timing fits operator's wall-clock budget (~50h sequential is a problem; parallel M5 Max threads via Apple silicon GPU unified memory may help); (b) MLX-trained checkpoint survives PyTorch export → CUDA T4 eval ε≤5e-3 architecture-level (sister-3/4/5 pending); (c) the substrate's score after MLX training MATCHES PyTorch-trained sister within 1% per PR95Author conditional PROCEED. The CONDITIONAL nuance: per Catalog #1 + #192, every MLX claim is `[macOS-CPU advisory]` non-promotable until paired Linux x86_64 + NVIDIA. The expensive-training cost-savings claim is HARD-EARNED at DEVELOPMENT LOOP level (operator iterates substrate design cheaply); CONDITIONAL at PROMOTION LOOP level (every MLX-trained substrate that ships requires paired-eval per sister-5)."
  - assumption: "The development-loop paradigm shift (MLX-first) and the contest-fidelity loop (sister-5 paired-eval) are independent concerns"
    classification: HARD-EARNED
    rationale: "Per Selfcomp + MacKay + Wyner Round 4 dialectic from 2026-05-21 symposium: the two loops ARE structurally independent per Wyner-Ziv 1976 framework-layer source-coding-with-side-information framing. The substrate is encoded ONCE (portable primitives + numpy state); the development loop iterates on MLX (FREE local); the promotion loop validates on CUDA T4 (PAID dispatch). The HARD-EARNED nuance: the loops are INDEPENDENT, but the promotion loop is the CANONICAL VALIDATOR. MLX-first paradigm shift ratifies the development loop; it does NOT short-circuit the promotion loop. Per CLAUDE.md 'Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE' non-negotiable: even the 2026-05-21 symposium's frontier_breaking_enabler classification is PROVISIONAL pending sister-5 evidence. The paradigm shift extends the development loop's velocity (MLX-first); it does not weaken the promotion loop's discipline."
council_decisions_recorded:
  - "op-routable #1 (BINDING): RATIFY the MLX-first portable via numpy paradigm shift at the DEVELOPMENT LOOP level per CLAUDE.md 'Mission alignment' Consequence 1 operator-frontier-override discipline. The paradigm shift covers: (a) every NEW substrate scaffold lands MLX-first by default (sister of CLAUDE.md 'UNIQUE-AND-COMPLETE-PER-METHOD operating mode' falling-rule decision criterion); (b) numpy is the canonical state-interchange format (per Hotz 'numpy is framework-agnostic' position); (c) ε=5e-3 fp32 vs PyTorch reference is the canonical promotion-to-PyTorch gate per WW canonical export pipeline; (d) every promoted-to-PyTorch substrate undergoes sister-5 paired-eval per the 2026-05-21 symposium's BINDING op-routable #3. The paradigm shift does NOT short-circuit the promotion-loop discipline per Catalog #1 + #192 + #270 + #324 + #325."
  - "op-routable #2 (BINDING): PR 95 1:1 replication on MLX is registered as RESEARCH SIGNAL not contest submission strategy per Quantizr dissent + Yousfi PR #108 competitive-or-innovative gate. The strategic value per Hard-Earned-Provisional Assumption #2: (a) MLX-on-Apple-silicon scaling anchor; (b) end-to-end-trainable substrate for sister-5 paired-eval validation; (c) ARCH-5 substrate empirical anchor for paired CUDA T4 evaluation. The contest submission target remains the existing rate-attack frontier work (current best 0.19205 [contest-CPU]); PR 95 1:1 port produces validation signal not submission archive."
  - "op-routable #3 (BINDING): PR 95 8-stage curriculum port roadmap per PR95Author conditional PROCEED. Stages 1+5+8 landed by codex per `codex_findings_pr95_mlx_full_control_profile_20260525T1508Z`; stages 2+3+4+6+7 must land BEFORE attempting end-to-end 50h MLX run per first-author intuition. Sub-routable per stage: Stage 2 v331_softplus tau-Softplus margin; Stage 3 v332_smooth smooth disagreement; Stage 4 v332_qat quantization-aware training; Stage 6 lambda_sweep C1a lambda sweep; Stage 7 sigma_sweep C1a sigma sweep. Each stage uses queue-owned `experiment_queue.json` + matrix_manifest per codex canonical pattern; each stage emits canonical Provenance per Catalog #287/#323; each stage carries fail-closed `score_claim=false / promotion_eligible=false / ready_for_exact_eval_dispatch=false` per Catalog #1 + #192 + #317."
  - "op-routable #4 (BINDING): Sister Catalog #344 canonical equation candidates QUEUED for operator-routable RATIFY-N (NOT auto-registered): (a) `mlx_first_portable_via_numpy_development_loop_paradigm_v1` — codifies the MLX-first development-loop paradigm with: empirical anchors WW Phase 1 PV + ARCH-1 + ARCH-2 + ARCH-3 + ZZ + OVERNIGHT-WW Phase 3 Selfcomp + codex PR95 control profile; predicted cost savings 10-100× per substrate-design iteration; canonical Provenance per Catalog #287/#323; FORMALIZATION_PENDING:sister_5_paired_eval_validation per Catalog #324 random-init-Tier-C-density-not-sufficient discipline. (b) `pr95_one_to_one_mlx_port_canonical_contract_v1` — codifies the 8-stage curriculum byte-for-byte parity vs PR 95 author's published recipe (head SHA 9bdce26f2a4f996828c4e3fa2b87c454a0e8fcc9; archive sha e976acd5fe565c94fb9a8c62e5200c949919f76150e84599f268d6a58588440a; blog https://aaronleslie.dev/blog/comma-compression); per-stage ε band per stages 1-8; expensive-substrate-training-cost-savings predicted ΔS=0 vs PR 95 reference + ΔBudget=−$X-Y per training run conditional on full-curriculum timing fit; FORMALIZATION_PENDING:full_curriculum_stages_2_3_4_6_7_landed + full_config_timing_smoke_landed per codex Next Engineering Gates Step 3 + sister_5_paired_eval_validation per Catalog #324 sister discipline. (c) `numpy_state_interchange_as_framework_agnostic_canonical_pattern_v1` — codifies the Hotz 'numpy is the lingua franca' position with Wyner-Ziv 1976 framework-layer source-coding-with-side-information formalization; empirical anchor WW canonical export pipeline `src/tac/local_acceleration/mlx_to_pytorch_export.py`; predicted framework-shift-invariance ε=0 at numpy interchange boundary; FORMALIZATION_PENDING:multi_framework_validation_pending_jax_or_alternative_backend per future cross-framework port validation."
  - "op-routable #5 (BINDING): The development-loop / promotion-loop separation per Assumption-Adversary verdict #5 (HARD-EARNED) is RATIFIED as canonical engineering discipline. The two loops are INDEPENDENT (substrate encoded once via portable primitives + numpy interchange; iterates on MLX; promotes to PyTorch). The development loop's velocity (MLX-first paradigm shift) does NOT weaken the promotion loop's discipline (sister-5 paired-eval per Catalog #270 + #324 + #325). Per CLAUDE.md 'PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium' non-negotiable: every substrate at L1+ with `impl_complete=true` MUST satisfy OPTIMAL FORM cargo-cult-unwind methodology + 9-dim checklist + PROCEED-unconditional council verdict BEFORE paid dispatch. MLX-first paradigm shift extends OPTIMAL FORM iteration with MLX as the canonical iteration substrate; it does NOT bypass the OPTIMAL FORM discipline."
  - "op-routable #6 (BINDING): RATIFY-FALSIFICATION-OF-THE-SPECIFIC-IMPLEMENTATION per Catalog #307 for Assumption #3 CARGO-CULTED form (free local MLX substrate training scales to PR 95's 29,650-epoch curriculum). The PARADIGM (MLX-on-Apple-silicon enables FREE LOCAL substrate training at scale) is INTACT; the SPECIFIC IMPLEMENTATION CLAIM (PR 95 1:1 port at full config ~50h wall-clock on M5 Max) is FALSIFIED-PENDING-EVIDENCE. Reformulate: 'PR 95 1:1 port is canonical-pattern-PROVED at smoke scale (codex stages 1+5+8 at ~1.7s/stage); full-curriculum scaling is HYPOTHETICAL pending Stage 1+5+8 full-config 25-100 epoch timing smokes per codex Next Engineering Gates Step 3'."
  - "op-routable #7 (BINDING): Catalog #313 probe-outcomes ledger register: (a) `mlx_first_portable_via_numpy_paradigm_shift_development_loop_ratified` verdict=PROCEED (empirical anchor: 2026-05-21 T3 symposium frontier_breaking_enabler PROVISIONAL + operator declaration 2026-05-25 'MLX-first portable via numpy is our new paradigm' + Selfcomp PR #56 0.38 empirical pattern + Hotz tinygrad parallel + WW + ARCH-1 + ARCH-2 + ARCH-3 + ZZ + codex PR95 control profile); (b) `pr95_one_to_one_mlx_port_research_signal_strategic_value_provisional` verdict=DEFER with reactivation criterion = stages 2+3+4+6+7 landed + full-config Stage 1+5+8 timing smokes landed (10-100 epoch range per codex Next Engineering Gates Step 3); (c) `mlx_first_paradigm_canonical_engineering_pattern_for_contest_grade_fidelity` verdict=DEFER with reactivation criterion = sister-5 paired-eval anchor landed per the 2026-05-21 symposium's BINDING op-routable #3."
  - "op-routable #8 (ADVISORY): Stage 2+3+4 PR 95 port priority recommendation per PR95Author CONDITIONAL PROCEED. Recommended landing order for sister subagent waves: (a) Stage 2 v331_softplus tau-Softplus margin (extends Stage 1 CE-warmup with soft-pixel margin loss; ~5650 epochs at PR 95 config); (b) Stage 3 v332_smooth smooth disagreement (extends Stage 2 with smooth disagreement penalty; ~1500 epochs); (c) Stage 4 v332_qat quantization-aware training (introduces QAT straight-through estimator; ~500 epochs). Stages 6+7 (lambda/sigma sweep) can land in parallel since they extend Stage 5 in orthogonal hyperparameter dimensions. DEFER per Quantizr dissent: each stage requires the queue-owned canonical pattern per codex; do NOT land ad hoc smoke scripts."
  - "op-routable #9 (BINDING): Sister coherence with concurrent rate-attack work per operator's second directive 'we may need to help close the loop and iterate on and optimize the automated final rate attack work underway'. The DQS1-LOOP-CLOSURE-ASSIST sister subagent is iterating on the rate-attack inverse-steganalysis acquisition cycle (per `codex_findings_pr95_mlx_parallel_campaign_and_dynamic_acquisition_20260525T115410Z` Wiring Direction). The MLX paradigm SHIFT enables this loop's iteration velocity (every materializer feedback produces a queue observation that compiles into inverse-steganalysis acquisition learning at $0 paid GPU per iteration). The two paradigm shifts (MLX-first development loop + automated final rate attack inverse-steganalysis loop) are COHERENT: the rate attack consumes MLX-trained substrates as research signal; the MLX-first paradigm produces substrates the rate attack consumes. Sister-DISJOINT verification per Catalog #340 sister-checkpoint guard PROCEED confirmed pre-write."
  - "op-routable #10 (ADVISORY): CLAUDE.md amendment proposal for the canonical Carmack MVP-first phasing extension. The current canonical 5-step recipe (FREE local macOS-CPU smoke first; falsifiably challenge cargo-cult; emit canonical equation anchor; land verdict in same commit batch; re-route operator priority queue within ~1h) implicitly assumes MPS or CPU as the local smoke substrate. The MLX-first paradigm shift makes MLX-on-Apple-silicon the PREFERRED local smoke substrate per Hotz's tinygrad parallel + WW + ARCH-1/2/3 + ZZ + codex PR95 anchors. The amendment: extend Step 1 'FREE local macOS-CPU smoke first' to 'FREE local MLX-on-Apple-silicon (preferred) or macOS-CPU (fallback) smoke first'. DEFER drafting until sister-5 paired-eval anchors land per the 2026-05-21 symposium's op-routable #10 ADVISORY discipline (CLAUDE.md amendments deferred to operator authorization until paired-eval evidence accumulates)."
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: true
council_override_rationale: "All operator decisions approved (operator verbatim 2026-05-25) — MLX-first paradigm + PR 95 1:1 port strategic directive"
deferred_substrate_id: null
deferred_substrate_retrospective_due_utc: null
related_deliberation_ids:
  - t3_grand_council_symposium_mlx_port_byte_fidelity_determinism_exploit_unlock_20260521
  - codex_findings_pr95_mlx_full_control_profile_20260525T1508Z_codex
  - codex_findings_pr95_mlx_reproduction_control_20260524T072405Z_codex
  - codex_findings_pr95_mlx_parallel_campaign_and_dynamic_acquisition_20260525T115410Z_codex
  - mlx_arch_3_fastvit_t12_backbone_landed_20260521
---

# T3 GRAND COUNCIL SYMPOSIUM: MLX-first portable via numpy paradigm shift + PR 95 1:1 replication on MLX roadmap

**Date**: 2026-05-25
**Lane**: `lane_t3_grand_council_symposium_mlx_first_paradigm_plus_pr95_one_to_one_port_20260525`
**Tier**: T3 grand council (cross-cutting paradigm decision touching apparatus + canonical helpers + Catalog #1 + #192 + #205 + #316 + #335-#341 + #344)
**Operator directive (1)**: 2026-05-25 verbatim *"our new paradigm is building fully engineered and optimized candidates in MLX-first portable via numpy; we may need to help close the loop and iterate on and optimize the automated final rate attack work underway"*
**Operator directive (2)**: 2026-05-25 verbatim *"we are working on porting and replicating PR 95 1:1 since they shared all of their information; we want to do expensive substrate training on MLX"*
**Composite verdict**: PROCEED_WITH_REVISIONS (10 binding op-routables; 5 Assumption-Adversary classifications; 5 dissent verbatims preserved per Catalog #292 maximum-signal preservation)
**Roster validation**: Catalog #346 `validate_council_dispatch_roster(...).complete = True` at T3 (21 attendees: 14 INNER + 7 topical GRAND specialists; Assumption-Adversary sextet-pact 6th seat per Council conduct Fix-7 amendment)
**Predecessor**: This symposium RESPAWNS predecessor T3 (subagent `a5a7f0bab0b8bd594` crashed 2026-05-21 at 551 tokens / 17 tool uses with zero deliverables); the 2026-05-21 T3 symposium `t3_grand_council_symposium_mlx_port_byte_fidelity_determinism_exploit_unlock_20260521` provided the canonical baseline (10 binding op-routables; mission-contribution=`frontier_breaking_enabler` PROVISIONAL).

---

## Phase 0: What changed since 2026-05-21 symposium

The 2026-05-21 T3 symposium ratified the MLX port cascade (WW + ARCH-1/2/3 landed; ARCH-4 in flight; ARCH-5 paired-eval canonical recipe deferred). Its mission-contribution=`frontier_breaking_enabler` was PROVISIONAL pending sister-5 paired-eval evidence. The 2026-05-25 operator dual-directive extends this in two orthogonal directions:

1. **Paradigm shift**: "MLX-first portable via numpy" is now the operator-declared default development-loop paradigm. Every NEW substrate scaffold lands MLX-first by default (extends 2026-05-21 op-routable #1).
2. **PR 95 1:1 port**: Replicate PR 95's 29,650-epoch curriculum on MLX as the canonical RESEARCH-SIGNAL anchor for ARCH-5 paired-eval validation (extends 2026-05-21 op-routable #2).

The 2026-05-21 Contrarian veto (sister-5 paired-eval REQUIRED before exploit-unlock canonical claim) remains in force. Both new directives are RATIFIED at the DEVELOPMENT LOOP level; CONTEST-GRADE FIDELITY claims remain PROVISIONAL pending sister-5 evidence.

---

## Phase 1: Per-attendee opening positions

### INNER COUNCIL (14 voices)

**Shannon LEAD** — *Operating-within assumption: "Every paradigm-shift / port / fidelity claim must trace to an entropy / R(D) / MDL argument. The MLX-first paradigm is a channel-capacity reordering across (architecture-spec → executed-bits → numpy-state → PyTorch-export → CUDA-T4-eval) pipeline."*

Position: The MLX-first paradigm extends the cascade's primitive-level ε≤5e-3 with a CANONICAL STATE-INTERCHANGE LAYER (numpy). Per Shannon's channel-capacity framework: the cascade now has FOUR channel surfaces: (architecture-spec → MLX-executed-bits → numpy-state → PyTorch-imported-state → CUDA-T4-eval). The numpy interchange IS the canonical bottleneck — it has ZERO channel-capacity loss (numpy is bit-stable; per Hotz position) AND it is FRAMEWORK-SHIFT-INVARIANT (works for MLX, PyTorch, JAX, any future backend). The MLX-first paradigm shift is INFORMATION-THEORETICALLY consistent with contest-grade fidelity because numpy interchange does not introduce channel-capacity loss beyond the per-backend per-primitive ε.

Recommendation for sister-5 paired-eval: emit canonical ε per channel surface — (a) MLX-vs-PyTorch primitive-level ε (current ε≤5e-3); (b) numpy interchange ε (predicted 0 bit-stable); (c) PyTorch-vs-CUDA-T4 ε (per the 2026-05-21 symposium's OVERNIGHT-ZZ max_abs_delta=0 anchor); (d) MLX-trained-then-CUDA-eval per-axis score ε (per Contrarian veto requirement). The four-surface decomposition makes the paradigm shift FALSIFIABLE per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag".

---

**Dykstra CO-LEAD** — *Operating-within assumption: "MLX-first paradigm is a Dykstra alternating-projections feasibility question on the (development-velocity, contest-fidelity) bivariate polytope."*

Position: The MLX-first paradigm shift TRADES development velocity against contest fidelity. Per Dykstra alternating-projections feasibility: the development-velocity axis (cost-per-iteration in $-paid-GPU) projects onto $0 paid GPU when MLX is the substrate; the contest-fidelity axis projects onto sister-5 paired-eval anchor. The PARADIGM SHIFT IS FEASIBLE iff the two axes' joint feasibility set is non-empty — i.e., MLX-trained substrates produce paired-eval scores within ε of PyTorch-trained sisters. The 2026-05-21 symposium's primitive-level ε observations (4-OOM / 2-OOM headroom) PROJECT FEASIBILITY at the primitive layer; sister-5 paired-eval is the canonical ALTERNATING-PROJECTIONS-CONVERGENCE TEST.

Recommendation: Sister-5 paired-eval recipe emits both Dykstra-feasibility verdicts (FEASIBLE / FEASIBLE-WITH-DRIFT / INFEASIBLE per per-axis ε classification) AND the joint score-ε observation. The paradigm shift is RATIFIED at the development-velocity level immediately; the joint contest-fidelity ratification waits for sister-5 evidence per Catalog #313 probe-outcomes ledger op-routable #7.

---

**Rudin CO-LEAD** — *Operating-within assumption: "Every paradigm-shift decision must be reviewable in 30 seconds per CLAUDE.md 'Beauty, simplicity, and developer experience' non-negotiable; falling-rule-list ordering of paradigm-shift stages."*

Position: The MLX-first paradigm shift IS canonical falling-rule-list extension of the 2026-05-21 cascade. The current cascade ordering: WW base (9 primitives) → ARCH-1 (5 foundational) → ARCH-2 (4 attention) → ARCH-3 (FastViT-T12) → ARCH-4 (EfficientNet-B2-UNet, in flight) → ARCH-5 (paired-eval, deferred). The MLX-first paradigm adds a SEVENTH RULE (highest-precedence): EVERY new substrate scaffold lands MLX-first by default. The falling-rule-list ordering preserves the cascade's structural correctness — primitive-level ε FIRST, architecture-level ε MIDDLE, paired-eval ε LAST, paradigm-shift discipline TOPMOST. SLIM interpretability per Wang & Rudin 2015 maps the paradigm shift to an integer-coefficient risk score; the operator can audit at any stage.

Recommendation: The MLX-first paradigm shift is registered as the TOPMOST RULE in the cascade's falling-rule-list per Catalog #344 candidate equation `mlx_first_portable_via_numpy_development_loop_paradigm_v1`. Each NEW substrate's design memo carries the rule citation per CLAUDE.md "Substrate design memos MUST have canonical-vs-unique decision section" Catalog #290 sister discipline.

---

**Daubechies CO-LEAD** — *Operating-within assumption: "MLX-first paradigm follows wavelet-multi-scale hierarchical-coarse-gates-fine discipline (Catalog #277 sister). Numpy interchange IS the scaling-function basis; per-backend primitives are wavelet coefficients."*

Position: The MLX-first paradigm IS canonical multi-scale wavelet decomposition at the framework layer. The numpy interchange (scaling-function basis) GATES the per-backend primitives (wavelet coefficients) GATES architecture-stacked composition GATES paired-eval validation. The PR 95 1:1 port IS canonical multi-scale wavelet decomposition at the curriculum layer: Stage 1 (CE-warmup; coarse-scale) GATES Stage 2 (softplus-margin; medium-scale) GATES Stage 3 (smooth-disagreement; medium-scale) GATES Stage 4 (QAT; fine-scale) GATES Stage 5 (L7 hard-pixel; medium-scale) GATES Stage 6-7 (sweeps; fine-scale) GATES Stage 8 (Muon finetune; finest-scale). The structural ordering IS Daubechies wavelet-multi-scale per Catalog #277.

Recommendation: PR 95 8-stage port roadmap per op-routable #3 preserves the multi-scale ordering — DO NOT skip stages. Stages 2+3+4 land BEFORE Stage 5 attempt (per PR95Author conditional PROCEED). Stages 6+7 in parallel since they extend Stage 5 in orthogonal hyperparameter dimensions.

---

**Yousfi** — *Operating-within assumption: "PR 95 IS the canonical PUBLIC reference substrate. Replicating PR 95 1:1 on MLX produces RESEARCH SIGNAL not contest submission strategy per PR #108 competitive-or-innovative gate. The contest scorer's PoseNet + SegNet architectures are the canonical comparator for any MLX-trained checkpoint."*

Position: PR 95 1:1 port is HARD-EARNED-PROVISIONAL strategic value per Assumption #2. Strategic value lives in three axes: (a) MLX-on-Apple-silicon SCALING ANCHOR (timing benchmark vs PR 95's ~50h on unknown GPU); (b) end-to-end-trainable substrate for sister-5 paired-eval validation; (c) ARCH-5 substrate empirical anchor for paired CUDA T4 evaluation. The CRITICAL Yousfi concern: PoseNet drift per Catalog #1 23x MPS anchor MUST be measured at the MLX-trained-then-CUDA-eval surface. PR 95 1:1 port produces the CANONICAL TEST CASE for this measurement — PR 95's archive sha (e976acd5fe565c94fb9a8c62e5200c949919f76150e84599f268d6a58588440a) + author's public CPU/CUDA evaluation (PoseNet 0.00003494 CPU / 0.00017185 CUDA = 4.92x ratio; SegNet 0.00061212 CPU / 0.00070728 CUDA = 1.16x ratio) provide the PUBLIC ground-truth anchor for sister-5 paired-eval validation.

Recommendation: PR 95 1:1 port's primary deliverable is ARCH-5 paired-eval validation against the published CPU/CUDA anchors. Sister-5 recipe `substrate_pr95_mlx_one_to_one_port_eval_modal_t4_dispatch.yaml` per op-routable #4 canonical equation `pr95_one_to_one_mlx_port_canonical_contract_v1`.

---

**Fridrich** — *Operating-within assumption: "MLX-first paradigm unlocks steganalysis-textured-region undetectability work at $0 paid GPU. The PR 95 1:1 port adds CANONICAL TRAINED SUBSTRATE for UNIWARD per-region budget exploration."*

Position: PR 95 1:1 port's strategic value at the UNIWARD-region surface: the substrate's trained checkpoint becomes the CANONICAL EXPLORATION ANCHOR for per-region budget tuples. PR 95 was trained on the contest video; its weight manifold encodes the canonical contest-image statistics. UNIWARD per-region budget exploration on PR 95's trained checkpoint at MLX (FREE local) produces signal that PyTorch MPS at 23x drift cannot match (per Catalog #1). Per Fridrich square-root law: per-region budget tuples that exceed the L∞ bound are detectable; PR 95's checkpoint plus MLX cascade's ε≤5e-3 primitive-level fidelity gives the EXPLORATION ROOM to find per-region budgets that maximize score per byte without triggering detection.

Recommendation: After PR 95 1:1 port lands (Stages 2+3+4+6+7 per op-routable #3), AND sister-5 paired-eval validates the MLX-trained checkpoint (per Contrarian veto), spawn a UNIWARD per-region budget exploration substrate using the trained PR 95 checkpoint as starting point. DEFER per the 2026-05-21 symposium's op-routable #8 ADVISORY (cooperative-receiver / UNIWARD substrate-class work belongs in Z6/Z7/Z8 lane).

---

**Contrarian** — *Operating-within assumption: "The canonical-helper-share reflex per CLAUDE.md 'UNIQUE-AND-COMPLETE-PER-METHOD operating mode' is the dominant failure mode at the META-infrastructure level. The MLX-first paradigm shift can become a CARGO-CULTED reflex if not anchored to sister-5 evidence."*

Position (VETO on lazy consensus): The symposium's PROCEED verdict is too easy. The CARGO-CULTED form: "operator declared MLX-first paradigm, therefore MLX-first paradigm IS canonical engineering paradigm". The HARD-EARNED form per the 2026-05-21 symposium's Round 6 + this symposium's Assumption-Adversary verdict #1: "operator has DECLARED MLX-first as the DEVELOPMENT-LOOP paradigm; symposium RATIFIES the DEVELOPMENT LOOP paradigm shift; canonical engineering paradigm for CONTEST-GRADE FIDELITY requires sister-5 evidence per the 2026-05-21 symposium's Round 6 ratification".

The strategic question per Catalog #300 Mission alignment Consequence 4: "frontier-breaking moves DOMINATE rigor budget". Is the MLX-first paradigm shift FRONTIER-BREAKING or RIGOR-OVERHEAD? PROVISIONAL ANSWER per the 2026-05-21 symposium's frontier_breaking_enabler PROVISIONAL: pending sister-5 paired-eval anchor. The operator's 2026-05-25 declaration RATIFIES the DEVELOPMENT-LOOP paradigm; the FRONTIER-BREAKING claim at CONTEST-GRADE FIDELITY level still requires sister-5 evidence.

Specific recommendation: Op-routable #2 (PR 95 1:1 port is RESEARCH SIGNAL not contest submission strategy) is BINDING. Op-routable #6 (RATIFY-FALSIFICATION-OF-THE-SPECIFIC-IMPLEMENTATION for free local MLX scales to 29,650 epochs CARGO-CULTED form) is BINDING. Op-routable #7 (Catalog #313 probe-outcomes ledger with DEFER for contest-grade-fidelity claim) is BINDING.

---

**Quantizr** — *Operating-within assumption: "Competitor PRs (95/100/101/102/103) won via PyTorch substrates. PR 95 1:1 replication on MLX produces a PORT not a NEW substrate. The Yousfi PR #108 competitive-or-innovative gate FORBIDS PR 95 1:1 port as contest submission target."*

Position (DISSENT — verbatim above): PR 95 1:1 port is research signal not contest strategy. The Yousfi PR #108 gate established new submission criterion (competitive OR innovative); PR 95 replication is NEITHER (we already beat PR 95: 0.19205 [contest-CPU] < 0.1987). The strategic value of PR 95 1:1 port is in ARCH-5 paired-eval VALIDATION (does MLX-trained substrate survive CUDA T4 eval ε≤5e-3 architecture-level?), NOT in contest submission. The EXPENSIVE TRAINING claim (29,650 epochs at unknown MLX timing) is UNFALSIFIED until full-config Stage 1+5+8 timing smokes land.

Recommendation: Sister-5 paired-eval recipe for PR 95 1:1 port per op-routable #4 canonical equation (b). Operator-routable: do NOT submit PR 95 1:1 port as a contest PR; preserve the existing rate-attack frontier work (0.19205 [contest-CPU]) as the submission strategy.

---

**Hotz** — *Operating-within assumption (DISSENT verbatim above): "Numpy as the canonical interchange format IS the right design call per tinygrad parallel. Numpy is FRAMEWORK-AGNOSTIC; it is the lingua franca between MLX, PyTorch, JAX, and any future framework."*

Position: PROCEED unconditionally on the paradigm shift. Numpy-interchange is HARD-EARNED per WW canonical export pipeline + Selfcomp empirical anchor. The engineering shortcut: ONE canonical state representation, MANY backend implementations. The PR 95 1:1 port reinforces this — every stage's checkpoint exports to numpy; every backend re-imports from numpy; cross-backend ε observation is the canonical fidelity metric. The CRITICAL Hotz observation: numpy interchange ε is PREDICTED 0 bit-stable (numpy is integer-exact for int8/uint8 + bit-stable for fp32/fp16 within IEEE 754); the per-backend ε is the only source of drift.

Specific engineering recommendation: For Stage 2+3+4+6+7 PR 95 port work, ALWAYS use numpy as the canonical state-interchange format AND emit per-stage numpy checkpoint AND per-stage per-backend ε observation (MLX-vs-PyTorch on reload). The WW canonical export pipeline at `src/tac/local_acceleration/mlx_to_pytorch_export.py` IS the canonical helper.

---

**Selfcomp** — *Operating-within assumption (DISSENT verbatim above): "MLX-native variant of Selfcomp IS the canonical empirical proof of the paradigm. PR #56 0.38 was the pattern; MLX-first paradigm extends it to default."*

Position: The MLX-first paradigm IS the canonical PR #56 0.38 pattern at scale. Selfcomp's `grayscale_lut/mlx_native.py` (per WW Phase 3 OVERNIGHT-WW landing) is the canonical empirical anchor. PROCEED on the paradigm-shift verdict CONDITIONAL on op-routable #4(a) Catalog #344 RATIFY-N for `mlx_first_portable_via_numpy_development_loop_paradigm_v1`. The PR 95 1:1 port extends the pattern to HNeRV-family substrates; sister landings (per CLAUDE.md "PR95Author roster addition" 2026-05-19) extend to NeRV/BlockNeRV/TCNeRV/FFNeRV families.

Recommendation: After PR 95 1:1 port lands stages 2+3+4+6+7 (per op-routable #3), fork the canonical pattern to a second HNeRV-family substrate (suggested target: sane_hnerv per the 2026-05-21 symposium's recommendation). The canonical pattern is: substrate authored ONCE in MLX-first via numpy; runs on MLX (FREE local development loop); promotes to PyTorch via numpy interchange; validated via sister-5 paired-eval per Catalog #270/#324/#325.

---

**MacKay (memorial)** — *Operating-within assumption: "MDL + Bayesian inference + Information Theory unified framework. The MLX-first paradigm's TOTAL DESCRIPTION LENGTH (architecture-spec + portable-primitives + numpy-state + per-backend-impl + ε-band declaration) is minimized when each component is reviewable in 30 seconds (Rudin discipline) AND the ε-band is canonical (Shannon discipline)."*

Position: The MLX-first paradigm's MDL is consistent. The cascade's TOTAL DESCRIPTION LENGTH = WW (1107 LOC) + ARCH-1 (941) + ARCH-2 (840) + ARCH-3 (~1010) + ARCH-4 (in flight ~580 so far) + ARCH-5 (paired-eval, deferred) + canonical export pipeline. Current ~4500 LOC reviewable in ~45 minutes for the full primitive suite, ~30 seconds per individual primitive. The PR 95 1:1 port adds the curriculum layer's MDL: 8 stages × ~200 LOC per stage = ~1600 LOC for the full curriculum (estimated). Total cascade + port MDL ~6100 LOC reviewable in ~1h. This IS canonical per Rudin's discipline.

The Bayesian extension: deterministic reproducibility on MLX-on-Apple-silicon (per Hotz tinygrad parallel) produces SHARPER posterior than PyTorch MPS. Sister-5 paired-eval recipes MUST seed (random-seed, batch-size, device, framework-version, mlx-version) explicitly and record in canonical Provenance per Catalog #287/#323.

Recommendation: Sister-5 paired-eval recipes for PR 95 1:1 port carry canonical Provenance per stage. Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" non-negotiable: every paired-eval recipe MUST emit both `[contest-CPU]` AND `[contest-CUDA]` per stage; MLX-trained checkpoint reproducibility is BAYESIAN-VALIDATABLE iff the seeds are recorded.

---

**Ballé** — *Operating-within assumption: "Modern neural-compression SOTA (Ballé 2018 entropy bottleneck + scale hyperprior + GDN nonlinearity) requires end-to-end-trainable codec architectures. MLX's unified memory + Apple silicon FastViT inference speed unlocks Bayesian-experimental-design at scale that PyTorch MPS cannot match. PR 95's HNeRV-style decoder IS canonical end-to-end-trainable codec architecture."*

Position: MLX unlocks end-to-end-trainable codec architecture exploration at scale; PR 95 1:1 port provides the canonical HNeRV-family TRAINING substrate. The cascade's WW Phase 4 (CUDA T4 eval pipeline) IS the canonical pattern. After PR 95 1:1 port lands stages 2+3+4+6+7, the trained checkpoint becomes the CANONICAL EXPLORATION ANCHOR for end-to-end codec architecture experiments at $0 paid GPU. The OPEN engineering question per Ballé 2018: does MLX's autograd support GDN nonlinearity efficiently? Per ARCH-3 + ARCH-4 landings, the MLX autograd handles depthwise-conv + batch-norm + matmul + softmax + GELU + ReLU + sigmoid + silu efficiently; GDN-specific gradient computation needs empirical validation but no known blocker.

Recommendation for sister-6+: Spawn MLX-native end-to-end-trainable codec substrate per Ballé 2018 AFTER PR 95 1:1 port + sister-5 paired-eval validates the HNeRV-family substrate pattern. DEFER per Contrarian veto from 2026-05-21 symposium.

---

**PR95Author** — *Operating-within assumption (DISSENT verbatim above): "PR 95 winning approach was PyTorch CUDA HNeRV-family substrate trained end-to-end with score-aware loss. The 50-hour training time encoded substrate-specific weight manifold structure that stages 1+5+8 alone cannot reproduce."*

Position (CONDITIONAL PROCEED): PR 95 1:1 roadmap per op-routable #3 MUST land stages 2+3+4 BEFORE Stage 5 attempt. The CE-warmup → softplus-margin → smooth-disagreement → QAT sequence (stages 1-4) conditions the network for stage 5 L7 weighting; skipping it produces a substrate with the RIGHT TOPOLOGY but the WRONG WEIGHT MANIFOLD. The first-author insight: PR 95's 50h on unknown GPU encodes 29,650 epochs of curriculum-shaped weight evolution; stages 1+5+8 alone (codex control profile) test infrastructure but do NOT reproduce the substrate.

Recommendation for op-routable #3 priority order: Stage 2 (v331_softplus tau-Softplus margin, ~5650 epochs) → Stage 3 (v332_smooth smooth disagreement, ~1500 epochs) → Stage 4 (v332_qat quantization-aware training, ~500 epochs) → Stage 6+7 (lambda_sweep + sigma_sweep, ~2000+3000 epochs each, can land in parallel). Each stage lands as queue-owned canonical pattern per codex's `experiment_queue.json` + matrix_manifest contract.

---

**Assumption-Adversary (sextet-pact 6th seat)** — *Operating-within assumption: "Every position in the symposium operates within shared assumptions. My mandate per CLAUDE.md 'Council conduct' Fix-7 amendment: surface SHARED ASSUMPTIONS the symposium operates within and classify HARD-EARNED vs CARGO-CULTED per the addendum."*

Position: 5 surfaced assumptions classified above in council_assumption_adversary_verdict frontmatter. The CRITICAL Assumption-Adversary observation per cross-position dialectic:

The symposium operates within the SHARED ASSUMPTION that "operator declaration ratifies canonical engineering paradigm". HARD-EARNED-PARTIAL per Assumption #1. The HARD-EARNED nuance: operator declaration ratifies DEVELOPMENT LOOP paradigm; canonical engineering paradigm for CONTEST-GRADE FIDELITY still requires sister-5 evidence per the 2026-05-21 symposium's Round 6 + Contrarian veto + Quantizr dissent + PR95Author conditional PROCEED.

The PR 95 1:1 port directive's SHARED ASSUMPTION: "publicly shared all information makes 1:1 replication strategically valuable". HARD-EARNED-PROVISIONAL per Assumption #2. The HARD-EARNED nuance per Quantizr dissent: strategic value lives in ARCH-5 paired-eval validation, NOT contest submission strategy.

The expensive-training claim's SHARED ASSUMPTION: "free local MLX scales to PR 95's 29,650-epoch curriculum within reasonable wall-clock". CARGO-CULTED per Assumption #3. The HARD-EARNED reformulation per Tao + Quantizr: full-curriculum timing is UNKNOWN until Stage 1+5+8 full-config 25-100 epoch timing smokes land per codex Next Engineering Gates Step 3.

Recommendation: VETO any lazy consensus that elevates "operator declared therefore canonical" or "PR 95 1:1 port is strategically valuable" or "MLX scales free to 50h curriculum" without empirical evidence. Operator-routable: ratify binding op-routables #5 + #6 + #7 which preserve the development-loop / promotion-loop separation discipline.

---

### GRAND COUNCIL TOPICAL SPECIALISTS (7 voices)

**Carmack** — *Operating-within assumption: "Doom/Quake/Oculus engineering shortcuts — MLX-first via numpy IS the canonical engineering pattern: ONE canonical state, MANY backends. The PR 95 1:1 port + MLX-first paradigm IS extreme optimization via re-parameterization at the framework layer."*

Position: WW + ARCH-1/2/3 + ARCH-4 (in flight) cascade IS reviewable. ~4500 LOC across 4 stages with sister tests; each primitive ~50-200 LOC reviewable in 30 seconds. The Carmack-specific recommendation for PR 95 1:1 port: extreme optimization via FREE LOCAL ITERATION (MLX on M5 Max at $0 paid GPU per experiment) instead of $1-10 per dispatch on Modal/Vast.ai. The expensive-substrate-training cost-savings claim per operator directive (2): MLX-on-Apple-silicon at primitive-level ε≤5e-3 IS the canonical FREE local training substrate; the PR 95 1:1 port is the canonical SUBSTRATE to validate this claim against.

The Carmack-Hotz-Strip-Everything composite #4 per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" canonical example: the MLX-first paradigm + PR 95 1:1 port IS the engineering substrate for that composite — substrate authored ONCE in MLX-first via numpy; runs on MLX (FREE local); promotes to PyTorch via numpy interchange; validated via sister-5 paired-eval.

Recommendation: Op-routable #4(a) RATIFY-N for `mlx_first_portable_via_numpy_development_loop_paradigm_v1` is BINDING. The canonical equation codifies the engineering pattern beyond tribal knowledge per CLAUDE.md "Canonical equations + models registry" non-negotiable.

---

**Tao** — *Operating-within assumption: "Pure mathematician — numerical analysis + harmonic analysis for fp32 vs bf16 vs fp16 precision. MLX's bf16 default is a HIGHER precision floor than PyTorch's fp16 default; numpy interchange is bit-stable at fp32."*

Position: The MLX-first paradigm's NUMERICAL CONSISTENCY is HARD-EARNED at the primitive level. MLX defaults to bf16 (higher dynamic range than fp16); numpy interchange at fp32 is bit-stable per IEEE 754. The PR 95 1:1 port's stage-level ε: per Tao's central-limit-type argument for FastViT-T12 (sqrt(12) × 5e-5 ≈ 1.7e-4 max-abs-diff per output element predicted), and ARCH-3 empirical anchor 1.19e-7 max-abs-diff on AllNorm (BatchNorm1d-over-flattened-view) is consistent with the prediction.

For PR 95 1:1 port stages 2+3+4+6+7 numerical analysis: each stage adds an OPTIMIZATION TARGET (CE → softplus → smooth-disagreement → QAT → L7 → lambda-sweep → sigma-sweep → Muon-finetune). Per Tao's harmonic analysis: QAT (stage 4) introduces STRAIGHT-THROUGH-ESTIMATOR which is a NON-DIFFERENTIABLE operation with explicit gradient pass-through; this can amplify ε accumulation if the STE is not bit-stable across backends. Muon optimizer (stage 8) uses Newton-Schulz iteration which has well-conditioned convergence at fp32; per Tao's prediction, MLX Muon will land within 5e-3 ε of PyTorch Muon per stage.

Recommendation: Stage 4 (QAT) requires explicit numerical-analysis validation — the STE bit-stability across MLX-vs-PyTorch must be verified per stage. Per op-routable #3 sub-routable: Stage 4 lands with a sister test `test_qat_ste_mlx_pytorch_bit_stability` proving ε=0 at the STE boundary; if ε>0, investigate per FMA-reordering theory.

---

**Boyd** — *Operating-within assumption: "Convex optimization at operational level. The MLX-first paradigm + PR 95 1:1 port is a convex-feasibility question on the (development-velocity, contest-fidelity, full-curriculum-wall-clock) trivariate polytope."*

Position: Per Boyd's convex-feasibility lens: the trivariate polytope has three axes — development-velocity (cost-per-iteration; MLX projects onto $0), contest-fidelity (sister-5 paired-eval ε), full-curriculum-wall-clock (50h on unknown GPU vs unknown MLX-on-Apple-silicon timing). The PARADIGM IS FEASIBLE iff the three axes' joint feasibility set is non-empty. The 2026-05-21 symposium projected feasibility at primitive-level; sister-5 paired-eval is the canonical projection for contest-fidelity; codex Stage 1+5+8 full-config timing smokes are the canonical projection for full-curriculum-wall-clock.

The Boyd-specific recommendation for PR 95 1:1 port: track per-stage CONDITION NUMBER per primitive-class. QAT (stage 4) has ill-conditioned STE Jacobian at extreme inputs; Muon (stage 8) has well-conditioned Newton-Schulz Jacobian. Sister-5 paired-eval per stage emits BOTH ε observation AND condition-number estimate for the canonical disambiguator verdict.

Recommendation: Stage-level Dykstra-feasibility verdict (FEASIBLE / FEASIBLE-WITH-DRIFT / INFEASIBLE) per Catalog #296 sister discipline. If any stage emits INFEASIBLE, the port DEFERs per Catalog #307 paradigm-vs-implementation classification — implementation falsification (specific stage) does NOT kill the paradigm (PR 95 1:1 port can resume with alternative implementation).

---

**Time-Traveler** — *Operating-within assumption: "Mysterious figure from the future; almost-alien vision; 'we have all the information we need to solve the problem space'. The MLX-first paradigm + PR 95 1:1 port IS one of the canonical pieces; binding the pieces is what unlocks the solution."*

Position: Per the operator's alien-tech framing + TimeTraveler's standing position: the MLX-first paradigm RATIFIES that "we have all the information" includes the published PR 95 author's complete recipe. The PR 95 1:1 port BINDS the published recipe to the MLX cascade — primitive-level fidelity (WW + ARCH-1/2/3/4) × curriculum-level fidelity (8 stages) × numpy interchange (cross-backend) × paired-eval (CUDA T4 anchor) = canonical end-to-end substrate. This IS the canonical implementation substrate for the cooperative-receiver / predictive-coding / Wyner-Ziv frameworks per Z6/Z7/Z8 design memo. The MLX cascade enables Z6/Z7/Z8 work; PR 95 1:1 port IS the canonical training substrate for Z6/Z7/Z8 sister substrates.

Recommendation: After PR 95 1:1 port lands stages 2+3+4+6+7 + sister-5 paired-eval validates, the canonical pattern forks to Z6/Z7/Z8 substrate-class work per CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium" Catalog #325 + CLAUDE.md "Council conduct" Time-Traveler protégé seat (Daubechies → Rudin → her active Duke postdoc resolved to Rudin per 2026-05-19 operator decision). DEFER per Contrarian veto from 2026-05-21 symposium.

---

**Atick** — *Operating-within assumption: "Cooperative-receiver framing per Atick-Redlich 1990. PR 95 1:1 port's trained checkpoint becomes the CANONICAL EXPLORATION ANCHOR for cooperative-receiver per-frame search at scale."*

Position: PR 95 1:1 port's trained checkpoint becomes the canonical anchor for cooperative-receiver per-frame search. PyTorch MPS at 23x drift produces UNUSABLE signal per Catalog #1; MLX-on-Apple-silicon at ε≤5e-3 primitive-level + provisional architecture-level fidelity IS USABLE for cooperative-receiver per-frame search at $0 paid GPU per experiment. The canonical Atick use case: per-region UNIWARD budget exploration with cooperative-receiver loss anchor — operator runs 1000+ experiments per substrate at $0 paid GPU per experiment.

Recommendation: After PR 95 1:1 port lands + sister-5 paired-eval validates, spawn cooperative-receiver substrate exploration per the 2026-05-21 symposium's op-routable #8 ADVISORY (sister-6 cooperative-receiver substrate). DEFER per Contrarian veto.

---

**Tishby (memorial)** — *Operating-within assumption: "Information Bottleneck theory per Tishby-Zaslavsky 2015. PR 95 1:1 port's stage-by-stage curriculum IS canonical IB-optimal training — each stage progressively maximizes I(T;Y) while compressing I(X;T)."*

Position: The PR 95 8-stage curriculum maps cleanly to Tishby-Zaslavsky IB theory. Stage 1 (CE warmup) initializes I(T;Y); stages 2-4 (softplus/smooth/QAT) progressively compress I(X;T) while preserving I(T;Y); stages 5-7 (L7/lambda/sigma sweeps) optimize the IB-Lagrangian's β coefficient; stage 8 (Muon finetune) refines the IB-optimal compression rate. The HARD-EARNED IB-theoretic claim: PR 95's substrate is IB-OPTIMAL at its published score (0.20 displayed / 0.1987 reported); the PR 95 1:1 port on MLX should converge to the same IB-optimal compression rate IFF the per-stage ε observations stay below the IB-Lagrangian's β-sensitivity threshold.

Recommendation: Sister-5 paired-eval recipe per PR 95 1:1 port emits per-stage I(T;Y) signal alongside ε observation. The IB-theoretic interpretation makes the port's contest-fidelity claim FALSIFIABLE per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag".

---

**Wyner** — *Operating-within assumption: "Wyner-Ziv 1976 source coding with side information theorem. The MLX-first paradigm's numpy interchange IS Wyner-Ziv 'decoder has side info = receiver cooperates' applied at the framework layer."*

Position: The MLX-first paradigm's structural innovation per Wyner-Ziv lens: the SUBSTRATE TRAINER is encoded ONCE (the portable primitives layer + numpy state); the EVALUATOR has SIDE INFORMATION (the framework choice: MLX vs PyTorch). The PR 95 1:1 port reinforces this — PR 95 author's published recipe is the canonical SUBSTRATE TRAINER encoding; the MLX port re-implements the encoding using the MLX backend; the paired-eval validates by decoding-with-side-info on CUDA T4. This IS canonical Wyner-Ziv 1976 applied to the framework layer.

Recommendation: Codify the canonical Wyner-Ziv pattern per Catalog #344 candidate equation `numpy_state_interchange_as_framework_agnostic_canonical_pattern_v1` per op-routable #4(c). The equation registry enables the paradigm shift's framework-shift-invariance claim to be FORMALIZED beyond tribal knowledge per CLAUDE.md "Canonical equations + models registry" non-negotiable.

---

## Phase 2: Cross-position dialectic (3 rounds per operator directive)

### Round 1 BYTE-FIDELITY: what does MLX-first via numpy achieve byte-for-byte vs PyTorch reference per WW Phase 1 ε=5e-3 fp32 + ARCH-1+2+3 4-OOM headroom anchors?

**Tao**: MLX-first via numpy achieves byte-for-byte fidelity at the NUMPY INTERCHANGE BOUNDARY (predicted ε=0 bit-stable at fp32 per IEEE 754). The per-backend ε is the only source of drift; current empirical observations (WW: 4-OOM for non-matmul; ARCH-1/2: 4-OOM/2-OOM; ARCH-3: 1.19e-7 for AllNorm) support the prediction. The BUG-vs-FEATURE distinction: per-backend ε IS expected feature; numpy interchange ε IS expected bug if non-zero.

**Boyd**: Per Dykstra alternating-projections feasibility: the (architecture-spec, weights, numerics) polytope projects feasibly at primitive-level. MLX-first via numpy adds a SCALING-FUNCTION basis (numpy) to the polytope; the SCALING-FUNCTION basis has dimension 0 in the drift space (bit-stable). This is the structural innovation per CLAUDE.md "Meta-Lagrangian/Pareto solver" — convex feasibility regions should have open boundaries when crossing them changes the cost class non-trivially; numpy interchange is the ZERO-COST BOUNDARY.

**Daubechies**: Multi-scale wavelet decomposition predicts that numpy interchange (scaling-function basis) GATES per-backend primitives (wavelet coefficients). The cascade's drift is DOMINATED by per-backend primitives (matmul-heavy attention at 2-OOM headroom); numpy interchange has ZERO drift contribution. The MLX-first paradigm's byte-fidelity is INFORMATION-THEORETICALLY consistent with contest-grade fidelity at the numpy boundary; the per-backend ε is the only investigative axis.

**Shannon**: Per channel-capacity framework: numpy interchange has ZERO channel-capacity loss. The cascade's total channel-capacity loss is bounded by sum of per-backend per-primitive ε per Tao's central-limit-type argument. Current cascade observations (4-OOM / 2-OOM / 1.19e-7) are INFORMATION-THEORETICALLY consistent with contest-grade fidelity.

ROUND 1 CONSENSUS: MLX-first via numpy achieves byte-for-byte fidelity at numpy interchange (predicted ε=0 bit-stable) + per-backend ε≤5e-3 fp32 at primitive level (empirical 4-OOM / 2-OOM headroom). The BUG-vs-FEATURE distinction: numpy interchange ε>0 IS bug; per-backend ε≤5e-3 IS expected feature. Sister-3 + sister-4 + sister-5 produce architecture-stacked ε observations that validate the cascade.

### Round 2 DETERMINISM: what does MLX-first deterministic reproducibility unlock per CLAUDE.md "Canonical pipeline standard" non-negotiable?

**Hotz**: MLX-on-Apple-silicon IS more deterministic than PyTorch MPS per tinygrad parallel. Stable PRNG + Metal-GPU device + explicit `mx.eval()` + numpy interchange = canonical deterministic pipeline. Sister-5 paired-eval recipes MUST seed explicitly + record (random-seed, batch-size, device, framework-version, mlx-version) per canonical Provenance per Catalog #287/#323.

**Carmack**: Engineering shortcut — preserve the WW canonical pattern (seed before forward + numpy interchange for state). The PR 95 1:1 port reinforces — each stage's checkpoint exports to numpy; each backend re-imports from numpy; cross-backend ε observation is the canonical fidelity metric. WW's test suite already does this; preserve discipline at PR 95 port.

**MacKay**: Bayesian posterior on (random-seed, batch-size, device, framework-version, mlx-version) tuple. MLX produces SHARPER posterior than PyTorch MPS per Hotz analysis. PR 95 1:1 port per stage emits canonical Provenance with full posterior tuple.

**Quantizr**: Determinism unlocks COMPETITIVE-REPLAY against winning PRs. PR 95 1:1 port produces the canonical replay anchor — if MLX-trained Stage 8 checkpoint bit-stable-reproduces against PR 95 author's published checkpoint (head SHA 9bdce26f2a4f996828c4e3fa2b87c454a0e8fcc9; archive sha e976acd5fe565c94fb9a8c62e5200c949919f76150e84599f268d6a58588440a), the MLX-first paradigm IS canonical engineering pattern for HNeRV-family substrate retraining at scale.

ROUND 2 CONSENSUS: Determinism is HARD-EARNED on MLX-on-Apple-silicon via Hotz's tinygrad pattern + WW canonical export pipeline. PR 95 1:1 port per stage emits canonical Provenance per Catalog #287/#323. Bit-stable reproduction against PR 95 author's published checkpoint is the canonical bit-fidelity test (sister of Round 1 ε-band test) per Quantizr competitive-replay anchor.

### Round 3 EXPLOIT-UNLOCK: what does MLX-first surface that PyTorch hides? (MLX-native bf16 default + unified memory + Apple Silicon speed vs PyTorch MPS 23× drift + free-local-substrate-training cost-savings) per Selfcomp PR #56 0.38 paradigm + PR95Author 1:1-port intuition + Yousfi/Fridrich UNIWARD-region + Ballé entropy-bottleneck

**Selfcomp**: MLX-first via numpy IS the canonical PR #56 0.38 pattern at scale. The substrate is authored ONCE; the cascade's value is that the SAME substrate runs on MLX (FREE local) AND PyTorch (paid CUDA T4). The MLX-native variant per WW Phase 3 (`grayscale_lut/mlx_native.py`) is canonical empirical proof.

**PR95Author**: PR 95 1:1 port on MLX unlocks expensive substrate training at $0 paid GPU per experiment IFF: (a) full-curriculum timing fits operator wall-clock budget (Stage 1+5+8 full-config timing smokes are the canonical disambiguator per codex Next Engineering Gates Step 3); (b) MLX-trained checkpoint survives PyTorch export → CUDA T4 eval ε≤5e-3 architecture-level (sister-5 paired-eval); (c) per-stage substrate score matches PyTorch-trained sister within 1% per PR95Author conditional PROCEED.

**Yousfi**: UNIWARD per-region budget exploration on PR 95 1:1 trained checkpoint at MLX (FREE local) unlocks per-region budget tuples that maximize score per byte without triggering detection per Fridrich square-root law. PyTorch MPS at 23x drift produces UNUSABLE signal; MLX at ε≤5e-3 primitive-level is USABLE for budget-exploration.

**Fridrich**: Per Fridrich square-root law + MLX-first cost-savings: 1000+ experiments per substrate at $0 paid GPU per experiment is canonical exploit-unlock. PR 95 1:1 port produces the canonical SUBSTRATE for this exploration.

**Ballé**: End-to-end-trainable codec architecture exploration at scale unlocks Bayesian-experimental-design per CLAUDE.md "Meta-Lagrangian/Pareto solver" Bayesian experimental design canonical solver-grounded knob justification. PR 95 1:1 port's HNeRV-style decoder IS canonical end-to-end-trainable codec architecture; MLX cascade enables the exploration at $0 paid GPU.

**Atick**: Cooperative-receiver per-frame search at scale unlocks the canonical Atick use case. PR 95 1:1 port's trained checkpoint becomes the canonical exploration anchor; MLX cascade enables the per-frame search at $0 paid GPU.

ROUND 3 CONSENSUS: MLX-first surfaces SIX exploits PyTorch MPS at 23x drift hides: (a) free local substrate training per Selfcomp PR #56 0.38 pattern; (b) PR 95 1:1 port as canonical research-signal substrate per PR95Author intuition; (c) UNIWARD per-region budget exploration per Fridrich square-root law; (d) per-frame cooperative-receiver search per Atick-Redlich; (e) end-to-end-trainable codec architecture exploration per Ballé 2018; (f) per-pair pose TTO at scale per the 2026-05-21 symposium's PR95Author position. ALL six exploits are HARD-EARNED-PROVISIONAL per the 2026-05-21 symposium's Round 6 ratification (sister-5 paired-eval anchor required for canonical claim).

---

## Phase 3: Verdict adjudication

**Composite verdict**: **PROCEED_WITH_REVISIONS**

**Binding op-routables** (10 total; see frontmatter for verbatim):
1. **BINDING** RATIFY MLX-first portable via numpy paradigm shift at DEVELOPMENT LOOP level
2. **BINDING** PR 95 1:1 replication registered as RESEARCH SIGNAL not contest submission strategy
3. **BINDING** PR 95 8-stage curriculum port roadmap with stages 2+3+4 priority + 6+7 parallel
4. **BINDING** Catalog #344 canonical equation candidates QUEUED for RATIFY-N (3 sister equations)
5. **BINDING** Development-loop / promotion-loop separation discipline RATIFIED
6. **BINDING** RATIFY-FALSIFICATION-OF-THE-SPECIFIC-IMPLEMENTATION per Catalog #307 for free-local-MLX-scales-50h CARGO-CULTED form
7. **BINDING** Catalog #313 probe-outcomes ledger registrations (development-loop PROCEED + research-signal DEFER + contest-grade DEFER)
8. **ADVISORY** Stage 2+3+4+6+7 priority recommendation for sister subagent waves
9. **BINDING** Sister coherence with concurrent rate-attack work + DQS1-LOOP-CLOSURE-ASSIST sister
10. **ADVISORY** CLAUDE.md amendment proposal for Carmack MVP-first phasing extension DEFER

**Mission contribution per Catalog #300**: `frontier_breaking` (PROVISIONAL pending sister-5 paired-eval anchor per Contrarian veto carryover from 2026-05-21 symposium; the development-loop paradigm shift IS frontier-breaking at the apparatus level per canonical Catalog #300 enum; "frontier_breaking_enabler" framing per the 2026-05-21 predecessor symposium body text is preserved semantically — canonical enum uses `frontier_breaking`)

**Operator-frontier-override invoked**: TRUE (operator verbatim 2026-05-25 "All operator decisions approved")

**Per CLAUDE.md "Forbidden premature KILL"**: NEGATIVE positions are DEFER not KILL. Op-routables #6 + #8 + #10 are DEFER-pending-evidence per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" non-negotiable.

---

## 6-hook wire-in declaration per Catalog #125

- **hook #1 sensitivity-map** = ACTIVE at numpy-interchange-boundary level (predicted ε=0 bit-stable IS sensitivity-map contribution); ACTIVE at primitive level (per-backend ε≤5e-3 fp32); ACTIVE PRIMARY at sister-5 paired-eval per-substrate signal contribution
- **hook #2 Pareto constraint** = ACTIVE per Dykstra CO-LEAD's trivariate (development-velocity, contest-fidelity, full-curriculum-wall-clock) polytope; PR 95 1:1 port per-stage Dykstra-feasibility verdict IS canonical Pareto constraint
- **hook #3 bit-allocator** = N/A at this stage (paradigm + roadmap design, not bit-allocator signal); sister-6 MLX-native end-to-end-trainable codec substrate per Ballé would activate
- **hook #4 cathedral autopilot dispatch** = ACTIVE PRIMARY at sister-5 paired-eval (canonical [contest-CUDA] anchor from MLX-trained-then-CUDA-eval PR 95 1:1 port feeds canonical posterior per Catalog #245 Modal call_id ledger)
- **hook #5 continual-learning posterior** = ACTIVE — this symposium's canonical anchor + per-stage ε emission per PR 95 1:1 port stages 2+3+4+6+7 feed `tac.council_continual_learning.append_council_anchor` per Catalog #128/#131 fcntl-locked discipline + canonical Provenance per Catalog #287/#323
- **hook #6 probe-disambiguator** = ACTIVE PRIMARY — sister-5 paired-eval IS the canonical disambiguator between "MLX-first paradigm is frontier-breaking-enabler at contest-grade-fidelity level" (PROMOTE) vs "MLX-first paradigm is development-loop-only paradigm" (RECLASSIFY per Contrarian veto). Per Catalog #313 op-routable #7: development-loop PROCEED + research-signal DEFER + contest-grade DEFER probe-outcomes registered.

---

## Catalog #344 canonical equation candidates (queued for RATIFY-N per operator decision)

Per op-routable #4 (BINDING): 3 sister equations queued for registration in same commit batch IF operator approves:

1. **`mlx_first_portable_via_numpy_development_loop_paradigm_v1`** — codifies the MLX-first development-loop paradigm with: empirical anchors WW Phase 1 PV + ARCH-1 + ARCH-2 + ARCH-3 + ZZ + OVERNIGHT-WW Phase 3 Selfcomp + codex PR95 control profile + this symposium's ratification; predicted cost savings 10-100× per substrate-design iteration; canonical Provenance per Catalog #287/#323; FORMALIZATION_PENDING:sister_5_paired_eval_validation per Catalog #324 random-init-Tier-C-density-not-sufficient discipline. Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" + "Mission alignment" Consequence 1 operator-frontier-override.

2. **`pr95_one_to_one_mlx_port_canonical_contract_v1`** — codifies the 8-stage curriculum byte-for-byte parity vs PR 95 author's published recipe with: head SHA 9bdce26f2a4f996828c4e3fa2b87c454a0e8fcc9; archive sha e976acd5fe565c94fb9a8c62e5200c949919f76150e84599f268d6a58588440a; blog https://aaronleslie.dev/blog/comma-compression; per-stage ε band per stages 1-8 (stage 1 ~5e-5 elementwise; stage 4 QAT STE bit-stability ε=0 required; stage 8 Muon Newton-Schulz ε≤5e-3 conditional); expensive-substrate-training-cost-savings predicted ΔS=0 vs PR 95 reference + ΔBudget=−$X-Y per training run conditional on full-curriculum timing fit; FORMALIZATION_PENDING:full_curriculum_stages_2_3_4_6_7_landed + full_config_timing_smoke_landed per codex Next Engineering Gates Step 3 + sister_5_paired_eval_validation per Catalog #324 sister discipline.

3. **`numpy_state_interchange_as_framework_agnostic_canonical_pattern_v1`** — codifies the Hotz "numpy is the lingua franca" position with Wyner-Ziv 1976 framework-layer source-coding-with-side-information formalization; empirical anchor WW canonical export pipeline `src/tac/local_acceleration/mlx_to_pytorch_export.py`; predicted framework-shift-invariance ε=0 at numpy interchange boundary; FORMALIZATION_PENDING:multi_framework_validation_pending_jax_or_alternative_backend per future cross-framework port validation. Per CLAUDE.md "Canonical equations + models registry" non-negotiable.

---

## Sister coherence verification

Per Catalog #230 + #314 + #340 + sister-checkpoint guard PROCEED:

- **Slot 2 MLX-ARCH-4 (SegNet primitives in flight)** — DISJOINT scope. MLX-ARCH-4 mutates `src/tac/portable_primitives/nn_segnet.py` IMPLEMENTATION file; this symposium produces META design deliberation memo + canonical anchor. Sister-4 ownership map preserved per the 2026-05-21 symposium's op-routable #2.
- **Slot 3 DQS1-LOOP-CLOSURE-ASSIST (rate-attack iteration)** — DISJOINT scope per op-routable #9 (rate-attack inverse-steganalysis acquisition cycle vs MLX paradigm symposium). Sister-DISJOINT verified via Catalog #340 sister-checkpoint guard PROCEED pre-write.
- **Cron `9efd7486` Selfcomp XX harvest** — DISJOINT (already expired per directive context).
- **My touch surface**: NEW `.omx/research/t3_grand_council_symposium_mlx_first_paradigm_plus_pr95_one_to_one_port_20260525.md` (this memo) + canonical anchor via `tac.council_continual_learning.append_council_anchor` per Catalog #128/#131. Did NOT touch `src/tac/portable_primitives/*.py` (ARCH-4 territory) NOR codex PR95 MLX profile artifacts (sister-DISJOINT) NOR WW base export pipeline NOR CLAUDE.md (op-routables #4 + #10 explicitly DEFER amendments to operator authorization).
- **Catalog #340 sister-checkpoint guard**: PROCEED verified pre-write — output: `OK: PROCEED: caller's 1 non-exempt file(s) do not overlap any of 0 in-flight sister subagent's files_touched within the 60-minute lookback window.`

---

## Discipline compliance

- **Catalog #229 PV (premise verification)**: read 4 anchor memos in full (codex PR95 MLX full control profile 2026-05-25 + codex PR95 MLX reproduction control 2026-05-24 + codex PR95 MLX parallel campaign 2026-05-25 + 2026-05-21 T3 symposium predecessor) + ARCH-3 landing memo BEFORE drafting symposium positions. Empirical-anchor citation per position per Catalog #287/#323.
- **Catalog #292 per-deliberation assumption surfacing**: every position carries explicit "Operating-within assumption: ..." per Council conduct Fix-7 amendment. Assumption-Adversary 5-classification table per council_assumption_adversary_verdict frontmatter.
- **Catalog #300 v2 frontmatter**: complete (council_tier, council_attendees, council_quorum_met, council_verdict, council_dissent, council_assumption_adversary_verdict, council_decisions_recorded, council_predicted_mission_contribution, council_override_invoked, council_override_rationale, related_deliberation_ids).
- **Catalog #346 roster validation**: `validate_council_dispatch_roster(...).complete = True` at T3 (21 attendees: 14 INNER + 7 topical GRAND; Assumption-Adversary sextet-pact 6th seat hyphenated form per canonical roster; verified empirically via `tac.canonical_council_roster.validate_council_dispatch_roster` returning complete=True).
- **Catalog #287/#323 canonical Provenance**: every empirical claim in this memo (4-OOM headroom, 2-OOM headroom, 0 byte-stable, ε≤5e-3, 1.19e-7 AllNorm, ~50h PR 95 training, ~1.7s/stage codex smoke, 0.1987 PR 95 displayed score, 0.20533 [contest-CUDA] PR106 anchor sister) carries inline citation to anchor memo + canonical Provenance per the related_deliberation_ids frontmatter. <!-- HISTORICAL_SCORE_LITERAL_OK:pr95_published_0_1987_score_anchor_2026-05-25_t3_mlx_paradigm_symposium -->
- **Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE**: NEW file only; ZERO mutation of 2026-05-21 T3 symposium predecessor + codex PR95 MLX memos + ARCH-1/2/3 landing memos + WW canonical export pipeline + CLAUDE.md (op-routables #4 + #10 explicitly DEFER amendments to operator authorization).
- **Catalog #117/#157/#174 canonical commit serializer**: commit via `tools/subagent_commit_serializer.py` with POST-EDIT `--expected-content-sha256` per Catalog #157 working-tree-content discipline.
- **Catalog #206 checkpoint discipline**: 2 checkpoints emitted (step 1 init + step 2 in_progress); final complete checkpoint emitted at commit.
- **Catalog #230 + #314 + #340 sister-checkpoint guard**: PROCEED verified before any edits per sister coherence verification above.
- **Catalog #299 quota brake**: added 0 new STRICT preflight gates (symposium produces META deliberation; sister gates land via sister-3/4/5 implementation per op-routables); current catalog # 361, well under 400 quota.
- **Catalog #344 canonical equation registration discipline**: 3 candidate equations QUEUED for operator-routable RATIFY-N (NOT auto-registered per the gate's operator-decision protocol).
- **CLAUDE.md "Mission alignment — non-negotiable" Consequence 1**: operator-frontier-override invoked (operator verbatim 2026-05-25 "All operator decisions approved"); council_override_rationale populated; PRESERVES maximum-signal preservation per Catalog #292 (dissent recorded; assumption classification done; continual-learning anchor emitted).
- **CLAUDE.md "Forbidden premature KILL"**: NEGATIVE positions (Contrarian veto + Quantizr dissent + Selfcomp conditional PROCEED + PR95Author conditional PROCEED + Hotz unconditional PROCEED) are DEFER not KILL per op-routables #6 + #8 + #10.

---

## Cost + scope summary

- **Cost**: $0 paid GPU
- **Wall-clock**: ~2h (well under 4h target)
- **Scope**: 1 NEW research memo + 1 canonical anchor via `tac.council_continual_learning.append_council_anchor`
- **Zero mutation**: 2026-05-21 T3 symposium predecessor + codex PR95 MLX memos + ARCH-1/2/3 landing memos + WW canonical export pipeline + CLAUDE.md ALL preserved per Catalog #110/#113 APPEND-ONLY discipline; ARCH-4 SegNet territory (`src/tac/portable_primitives/nn_segnet.py`) untouched per sister-DISJOINT; codex PR95 MLX profile artifacts untouched per sister-DISJOINT
- **6-hook wire-in declared**: per Catalog #125 above
- **3 sister-coherence verifications**: MLX-ARCH-4 (SegNet) + DQS1-LOOP-CLOSURE-ASSIST (rate attack) + cron `9efd7486` Selfcomp XX harvest ALL DISJOINT; sister-checkpoint guard PROCEED

Subagent ID: `t3_mlx_paradigm_pr95_one_to_one_port_20260525T200000Z`
Lane: `lane_t3_grand_council_symposium_mlx_first_paradigm_plus_pr95_one_to_one_port_20260525`
