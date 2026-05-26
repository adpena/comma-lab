---
schema_version: path_3_canonical_substrate_development_cascade_doctrine_v1
created_utc: 2026-05-26T23:00:00Z
adopted_as_canonical: true
binding_for: all_path_3_substrates_present_and_future
council_tier: T2
council_attendees: [Shannon, Dykstra, Rudin, Daubechies, Carmack, Hotz, Quantizr, Selfcomp, MacKay, AssumptionAdversary, PR95Author]
council_quorum_met: true
council_verdict: PROCEED
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
council_dissent: []
horizon_class: frontier_pursuit
related_deliberation_ids:
  - path_3_candidate_inventory_for_next_wave_spawning_20260526
  - path_3_d_z6_l1_promotion_landed_20260526
  - path_3_optimization_tooling_audit_and_wire_in_roadmap_20260526
  - mlx_candidate_contest_equivalence_gate_landed_20260526
council_decisions_recorded:
  - "op-routable #1: HOLD Tier 3 L0 SCAFFOLD spawns (L/M/N/O queued) until existing 11 substrates have L2 training results"
  - "op-routable #2: Pivot to L1-PROMOTION-CASCADE for next 4-5 substrates per D=Z6 reference pattern"
  - "op-routable #3: Open L2 LONG-TRAINING-CASCADE concurrent on M-series Apple Silicon (3-5 substrates parallel)"
  - "op-routable #4: Reserve paid CUDA strictly for bridge calibration per substrate-class + final submission auth eval per CLAUDE.md non-negotiable"
  - "op-routable #5: META work (Wave #1 posterior_emission + CONSOLIDATE-OP-1 + R2-COMBINED + future) continues in parallel; benefits all cascade levels"
---

# Path 3 Canonical Substrate Development Cascade Doctrine

**Status**: CANONICAL adopted 2026-05-26 per operator binding directive *"Document that and adopt as canonical"*.

**Purpose**: define the canonical 6-level cascade every Path 3 substrate traverses from L0 SCAFFOLD to L6 CONVERGED CANDIDATE; establish pacing discipline (HOLD Tier 3 L0 spawns; pivot to L1-promotion + L2-long-training cascade); narrow paid CUDA scope per operator's *"All training to be done on MLX and sweeps on MLX"* directive.

## Operator binding directives (verbatim, 2026-05-26)

1. *"The MLX first requirement might also force us out of the issue we were having before where we had great ideas but we're building them as Boltons to the same substrates over and over again; we want to design the substrate and curriculum and then optimize the design the whole stack around it for extreme optimization and performance and optimal score lowering"*
2. *"Never simply extend unless a rigorous adversarial cargo cult pass has been done first"*
3. *"we also need adversarial review against all landing recursive for math and scientific and engineering rigor and for MLX drift minimization and portability via numpy"*
4. *"Keep feeding the queue but we need to be mindful not to outpace session rate limits"*
5. *"All may likely benefit from our optimization tooling and tech which is quite extensive and likely underutilized"*
6. *"Great work, let's keep optimizing and iterating and building and getting stuff training and running on MLX"*
7. *"All training to be done on MLX and sweeps on MLX. What do you want to use paid CUDA for"*
8. *"Don't we need to do long extensive training runs and optimization and iteration first"*
9. *"Yes. Document that and adopt as canonical"*

## Canonical 6-level cascade

Every Path 3 substrate traverses L0 → L6. ALL of L0-L5 is MLX-local ($0 Apple Silicon). Paid CUDA enters ONLY at L6 → submission boundary.

| Level | Name | Duration (per substrate) | Gate (advance criterion) | Canonical Catalog references |
|---|---|---|---|---|
| **L0** | SCAFFOLD | ~2-5h | Design memo with all 7 required sections (Catalog #290/#294/#296/#303/#305/#309 + frontmatter Catalog #300 v2) + scaffold package + smoke trainer (≤5ep ≤8pairs synthetic) + `_full_main raises NotImplementedError` per Catalog #240(c) + Cargo-cult-pass FIRST per binding directive #2 (or 2-phase if FRESH design) + 3-axis evidence per directive #3 + non-promotable markers per Catalog #127/#192/#317/#341 | #240, #220, #272, #303, #294, #290, #305, #296, #309, #310, #311, #312, #324, #325, #335, #341, #357 |
| **L1** | INFRASTRUCTURE-CONVERGENCE | ~1-2h | Real GT loader via `tac.data.decode_video` + canonical score-aware loss per Catalog #164 + canonical EMA decay=0.997 per Catalog #2 non-negotiable + differentiable rgb_to_yuv6 + eval_roundtrip per CLAUDE.md non-negotiable + multi-epoch ≤100ep at decoder resolution + per-substrate symposium per Catalog #325 (PROCEED) + smoke convergence proof + lane registry promotion to L1 | #164, #2, #325, #324, #128 (posterior_update_locked), #1265 (REFUSED-PENDING-SISTER-GATE expected; sister gate per substrate-grammar) |
| **L2** | LONG TRAINING | ~12-48h MLX wall-clock | Multi-thousand-epoch training (3,000-29,650 per PR95 reference) at contest resolution (384×512 decoder OR 874×1164 camera) with 600 frame pairs + curriculum stages (warm-up → joint → QAT-prep → finetune mirroring PR95 8-stage) + checkpoint+resume capability + canonical posterior anchor per arm via `tac.continual_learning.posterior_update_locked` per Catalog #128 | #128, #131, #138, #245 (canonical 4-layer ledger), PR95 8-stage curriculum reference |
| **L3** | HYPERPARAMETER SWEEPS | ~24-96h MLX wall-clock; parallel arms | Architectural sweep axes per substrate (e.g. MOD_DIM ∈ {16, 32, 64, 128, 256} for K=COIN++; β_ib ∈ {1e-5..1e-2} for J=MDL-IBPS; latent_dim {48, 96, 192} for C6/J; num_levels sweep for F=Z8; K×G categorical configs for A=DreamerV3 + J) + curriculum stage hyperparams (lr schedule, batch_size, warmup_epochs) + parallel arm dispatch with canonical posterior anchor per arm + cross-arm diff per `tac.cathedral_consumers.master_gradient_xray_consumer` | #128, #245, master_gradient_consumers (8-exploit BUNDLE per Catalog #354), #220 byte-mutation no_op_proof per arm |
| **L4** | ARCHITECTURAL ITERATION | ~24-48h | Cargo-cult unwind v2 based on L3 sweep observations (training-time signal often reveals assumptions invisible at design-time per NSCS06 v6→v7 = 44% reduction canonical pattern) + Catalog #303 audit refresh + Phase 2 substrate-design decision per Catalog #290 (canonical-vs-unique re-evaluation per layer) + sister-substrate cross-pollination | #303, #290, #292 (per-deliberation assumption surfacing), NSCS06 v6→v7 canonical pattern, #325 per-substrate symposium re-convene |
| **L5** | OPTIMIZATION | ~12-24h | Quantization-aware training (FP4/INT8 per PR95 reference + Quantizr 0.33 anchor) + EMA decay sweep (final tuning around 0.997 baseline per CLAUDE.md non-negotiable) + decoder compression (brotli grammar tightening + section-byte minimization) + canonical bit allocator wire-in per `tac.bit_allocator` (per-byte / per-pair / per-class / per-axis / pareto_dual) + canonical equation #26 + canonical procedural codebook savings if applicable per Catalog #344 + #359 misapplication self-protect | #2 EMA, #122 quantization, `tac.bit_allocator`, #344 canonical equations, #359 equation #26 misapplication self-protect, #319 deliverability proof if Wyner-Ziv pipeline-stage applicable |
| **L6** | CONVERGED CANDIDATE | Byte-stable archive + #1265 gate PASS | Canonical `tools/gate_mlx_candidate_contest_equivalence.py` (sister-gate parameterized per substrate-class grammar) returns PASS verdict (max_abs < 0.001 contest-units; 90× margin over empirical anchor 0.000011 per #1265 landing memo) + canonical Provenance per Catalog #323 with `evidence_grade="macOS-MLX-research-signal"` + per-substrate symposium PROCEED-unconditional per Catalog #325 + R3 3/3 clean-pass counter per CLAUDE.md "Recursive adversarial review protocol" | #1265 gate, #323 canonical Provenance, #325 symposium, recursive adversarial review 3-clean-pass, `tac.cathedral_consumers.*` auto-discovery per Catalog #335 |

**After L6 → bridge calibration → final submission** (FIRST paid CUDA boundary):
- **Bridge calibration** (paid CUDA; one-time per substrate-class; ~$0.50-2 per class): characterize MLX↔CUDA drift bound for substrate-class (HNeRV / predictive-coding / IB-class / cooperative-receiver / iterative-boosting / hierarchical-residual / categorical-RSSM / meta-INR / Faiss-PQ / etc.). After per-class calibration, ALL future same-class substrates trust MLX-local without re-calibration.
- **Final submission auth eval** (paid CUDA; per submission; ~$0.20-1): BOTH contest-CUDA on T4-equivalent + contest-CPU on Linux x86_64 per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" non-negotiable. Contest-CPU = leaderboard score; contest-CUDA = Yousfi bot comment.

**Total paid CUDA forecast**: ~$5-30 for entire 11-substrate Path 3 wave (vs prior "per-substrate per-iteration" framing which would have been $100-500+).

## Pacing discipline (binding)

### DO (canonical cadence)

1. **Pivot to L1-PROMOTION-CASCADE** for the next 4-5 substrates per D=Z6 reference pattern (`8833b9db5`). Spawn order recommendation per R1+R1' CLEAN status + 3-axis evidence at landing:
   - **L1-PROMOTION-E-BOOSTNERV** (residual-against-PR110 direct stacking; highest EV-per-dollar; predicted ΔS bracket [-0.010, +0.0045])
   - **L1-PROMOTION-G-NIRVANA** (3-axis FULLY evidenced at landing; numpy reference complete; sister-canonical for portability pattern)
   - **L1-PROMOTION-C-PRIME-NSCS06** (180 tests + cargo-cult #5 empirically confirmed; needs cls_stream wire-in at L0 inflate BEFORE L1 per cargo-cult #5 remediation)
   - **L1-PROMOTION-B-PRIME-Z7-MAMBA-2-V2** (Path c FRESH; 40 tests; Mamba-2 paradigm distinct from sister NeRV-family)
   - **L1-PROMOTION-J-MDL-IBPS** (DISCRETE-CATEGORICAL-MINE-HYBRID; 39 tests; β_ib + κ sweep candidate for L3)
2. **Open L2 LONG-TRAINING-CASCADE** concurrent on M-series Apple Silicon — 3-5 substrates parallel at any time (limited by Apple Silicon shared GPU memory; small-batch training viable concurrent)
3. **Continue META work in parallel** — Wave #1 posterior_emission + CONSOLIDATE-OP-1 + R2-COMBINED + future audits remain valuable infrastructure regardless of cascade level
4. **L3 HYPERPARAMETER SWEEPS** on L2-converged substrates (results dictate L3 → L4 advancement)
5. **L4 ARCHITECTURAL ITERATION** on L3-sweep-winning substrates (cargo-cult unwind v2 informed by training-time signal)
6. **L5 OPTIMIZATION** on L4-iterated substrates (quantization + compression + bit allocator)
7. **L6 CONVERGED CANDIDATE** + sister #1265 gate PASS = bridge calibration eligibility
8. **Bridge calibration → final submission** = paid CUDA boundary

### DO NOT (canonical anti-cadence)

1. **DO NOT spawn more L0 SCAFFOLDs** (Tier 3 L=TT5L / M=Wyner-Ziv pipeline / N=NSCS06 v8 Path B / O=Z6-v2 currently QUEUED per task #1280-#1283) — 11 substrates already covers sufficient paradigm diversity (categorical-RSSM / Mamba-SSM / chroma-LUT / predictive-coding / iterative-boosting / hierarchical-quadruple / hierarchical-residual / cooperative-receiver-Atick / IVF-PQ / IB-MINE / meta-INR). Reactivation criterion: ≥3 of 11 LANDED substrates pass L3 with paradigm-distinct results AND we've exhausted the 11-paradigm space.
2. **DO NOT use paid CUDA for training, sweeps, smoke, iteration, cargo-cult, review, L1-promotion verification, or any intermediate work** — all MLX-local per binding directive #7. Paid CUDA STRICTLY for bridge calibration (one-time per class) + final submission auth eval (per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" non-negotiable).
3. **DO NOT skip levels** — L0 → L1 → L2 → ... → L6 sequential. Each level's gate is a precondition for the next. Skipping levels was the bug class that produced 22× miss on C6 IBPS (predicted band from random-init Tier-C density per Catalog #324 anchor).
4. **DO NOT spawn more than ~5 concurrent subagents** per Catalog #302 sister-subagent-scope-overlap protection + operator pacing directive #4 mindful-of-rate-limits.

## Per-substrate cascade status (current 2026-05-26 23:00 PDT)

| Substrate | Level | Commit | Next gate to advance |
|---|---|---|---|
| A=DreamerV3 RSSM | L0 + FIX-WAVE-R1 | `69253a1cc` + `e1b101888` | R2-COMBINED CLEAN counter advance; then L1 promotion |
| B'=Z7-Mamba-2-v2 cargo-cult-first | L0 | `7a103fdbb` | L1 promotion |
| C'=NSCS06 v8 chroma_lut cargo-cult-first | L0 + cls_stream wire-in pending | `f59c8401b` | L0 cls_stream remediation per cargo-cult #5; then L1 promotion |
| **D=Z6 predictive coding** | **L1** | `83b9ee3e2` + `8833b9db5` | **Sister #1265 gate parameterization for Z6PCWM1 grammar → L2 long-training (multi-thousand-epoch at 384×512 / 600 pairs)** |
| E=BoostNeRV against PR110 | L0 + FIX-WAVE-R1 docs | `83910e54e` + `e1b101888` | L1 promotion (highest-EV next per residual-against-PR110 stacking) |
| F=Z8 canonical-quadruple | L0 + FIX-WAVE-R1' | `5ff5d2ab9` + `4684dbbab` | R2-COMBINED CLEAN counter advance; then L1 promotion |
| G=NIRVANA cascading NeRV | L0 + FIX-WAVE-R1' | `f7d2e86fe` + `4684dbbab` | R2-COMBINED CLEAN counter advance; then L1 promotion (3-axis fully evidenced at landing; preferred-L1 candidate) |
| H=ATW V2 cooperative-receiver v2 | L0 | `06ea98483`+`683878854`+`98484a08b`+sister commits | R1'' review; then L1 promotion |
| I=V1 Faiss IVF-PQ residual | L0 | `c4d8bbae8` | R1'' review; then L1 promotion |
| J=MDL-IBPS DISCRETE-CATEGORICAL-MINE-HYBRID | L0 | `4506e2333` | R1'' review; then L1 promotion |
| K=COIN++ implicit neural representation | L0 | `eadee66ae` | R1'' review; then L1 promotion |

## Tier 3 L0 SCAFFOLD queue: DEFERRED

Per pacing discipline rule #1 above:
- L=TT5L foveation + LAPose (task #1280): DEFERRED
- M=Wyner-Ziv pipeline-stage codec primitive (task #1281): DEFERRED
- N=NSCS06 v8 Path B alternative class-shift (task #1282): DEFERRED
- O=Z6-v2 cargo-cult-unwind redesign (task #1283): DEFERRED

**Reactivation criteria**: ≥3 of 11 LANDED substrates pass L3 HYPERPARAMETER SWEEPS with paradigm-distinct results AND we have empirical evidence that the 11-paradigm space exhausts contest-CUDA frontier-breaking potential. Until then, Tier 3 is research INPUT only (sister memos referenced in L1+ work for cargo-cult-pass enrichment).

## L2 LONG-TRAINING INFRASTRUCTURE — CANONICAL + REUSABLE + COMPOSABLE + PRODUCTION-HARDENED (binding amendment per operator directive 2026-05-26)

**Operator binding directive verbatim**: *"Also the long training infrastructure, ensure reusable composable beautiful elegant creative expressive cimposable abstractions and production hardened OSS and no duplicative code"*

L2 LONG TRAINING advancement REQUIRES landed CANONICAL long-training infrastructure as a prerequisite. No per-substrate ad-hoc training code per CLAUDE.md "Beauty, simplicity, and developer experience" non-negotiable + Catalog #299 gate consolidation discipline + #335 cathedral consumer canonical contract pattern.

### Canonical surface

`tac.substrates._shared.trainer_skeleton.long_training_canonical` (or sister module under canonical naming) extending the existing canonical skeleton per Catalog #178 + #190 with:

1. **`run_long_training(substrate, config: LongTrainingConfig) -> TrainingArtifact`** — canonical entry-point; takes substrate-conforming object (per `tac.substrate_registry.contract.SubstrateContract`) + canonical config dataclass; emits canonical TrainingArtifact with EMA shadow checkpoint + canonical Provenance + canonical posterior anchor
2. **`LongTrainingConfig` frozen dataclass** — canonical config schema: `epochs: int` / `batch_pair_indices_per_step: int` / `curriculum_stages: tuple[CurriculumStage, ...]` / `ema_decay: float = 0.997` (per Catalog #2 non-negotiable) / `checkpoint_interval_epochs: int` / `early_stopping_patience: int` / `score_aware_loss_kwargs: Mapping` / `optimizer_class: str` / `lr_schedule: str` / `seed: int` / `reproducibility_seed_pinning_canonical: bool = True`
3. **`CurriculumStage` frozen dataclass** — canonical curriculum stage: `name: str` / `start_epoch: int` / `end_epoch: int` / `loss_weights: Mapping[str, float]` / `lr_scale: float` / `freeze_layers: tuple[str, ...]` / `enable_qat: bool`. Mirrors PR95 8-stage curriculum + extensible per substrate-class
4. **Checkpoint+resume** — canonical interruption-safe pattern; resume-on-crash via fcntl-locked checkpoint state per Catalog #131; sister of `tac.subagent_checkpoint` per Catalog #206 crash-resume discipline
5. **Per-arm canonical Provenance + posterior anchor** — every long-training run emits canonical Provenance per Catalog #323 + `tac.continual_learning.posterior_update_locked` per Catalog #128 with non-promotable markers per Catalog #127/#192/#317/#341 ([macOS-MLX research-signal] for MLX-local; [contest-CUDA-T4] for paid CUDA boundary only)
6. **Differentiable-eval-roundtrip + EMA-apply-at-eval canonical wrappers** — per CLAUDE.md non-negotiables (eval_roundtrip / EMA / differentiable yuv6); inherit via canonical helper not per-substrate boilerplate
7. **Multi-arm parallel dispatch** — single canonical helper handles N-concurrent arms on M-series shared GPU memory; per-arm canonical posterior anchor + cross-arm `tac.master_gradient_consumers.master_gradient_xray_consumer` invocation
8. **Crash-recovery + OOM-safe** — canonical patterns for Apple Silicon memory pressure; per-batch retry with batch_size halving on OOM; checkpoint preservation across crashes
9. **Observability surface per Catalog #305** — per-epoch loss curve emission + score-aware d_seg + d_pose + rate per-epoch + EMA-drift-per-tensor + canonical metrics flushed to `.omx/state/` (queryable post-hoc) and TensorBoard-style event files (cite-able + diff-able across runs)
10. **OSS-clean public API** — narrow `__all__` per Catalog #335 canonical contract + canonical docstrings + zero `/Users/adpena/...` paths per Catalog #208 + canonical SPDX-License-Identifier: MIT headers

### Production-hardening contract

Per CLAUDE.md "Beauty, simplicity, and developer experience" non-negotiable + Catalog #299 gate consolidation + #335 canonical contract pattern:

- **Tests**: comprehensive test suite at `src/tac/substrates/_shared/tests/test_long_training_canonical.py` — config dataclass validation + canonical entry-point happy path + checkpoint+resume integrity + OOM-safe batch halving + multi-arm parallel + canonical Provenance emission + canonical posterior anchor emission + EMA shadow correctness + curriculum stage transition correctness + reproducibility seed-pinning byte-stable across runs
- **Conformance tests for substrate-conformance**: every substrate's `SubstrateContract` validated against canonical helper requirements (canonical scorer-loss helper present + differentiable yuv6 present + canonical EMA wrapper present + canonical checkpoint interface present)
- **Documentation**: canonical usage memo at `docs/canonical_long_training_infrastructure.md` (operator-facing + sister-subagent-facing) with worked example end-to-end on D=Z6 substrate
- **NO duplicative code**: each substrate's L2 trainer is ≤30 LOC of substrate-specific config + ONE canonical helper invocation. Existing per-substrate trainer boilerplate REMOVED + replaced with canonical invocation pattern
- **Composable with sister tooling**: integrates with cathedral autopilot (`tools/cathedral_autopilot_autonomous_loop.py`) for canonical posterior anchor ingestion + with `tac.canonical_equations.update_equation_with_empirical_anchor` for canonical equation calibration per Catalog #344 + with `tac.bit_allocator` for per-pair / per-class / per-axis allocation surfacing + with `tac.findings_lagrangian` for 4-term Lagrangian invocation per Catalog #355 (`invoke_meta_lagrangian_on_candidates`)
- **Beautiful + Elegant + Creative + Expressive**: 30-second-reviewable canonical entry-point + minimal substrate-side LOC + canonical primitive composition (curriculum × EMA × checkpoint × score-aware-loss × differentiable-eval) emerges from substrate-conformance not boilerplate

### Spawn order

L2 LONG-TRAINING-INFRASTRUCTURE BUILD wave MUST land BEFORE any substrate enters L2 LONG TRAINING. Sequencing:

1. **L2-INFRA-BUILD subagent** (~6-10h; operator-routable) — extracts canonical long-training infrastructure to `tac.substrates._shared.trainer_skeleton.long_training_canonical` (or sister canonical module); production-hardened tests; OSS-clean API; canonical docstrings; D=Z6 reference migration as proof-of-pattern
2. **D=Z6 L2 first long-training run** — proof-of-canonical-helper end-to-end; canonical Provenance + posterior anchor lands; cathedral consumer cascade observes
3. **L1-PROMOTION-CASCADE per non-L2-ready substrates** (E + G + C' + B' + J + others) — each L1 promotion uses the canonical helper as the L2-prep target
4. **L2 LONG-TRAINING-CASCADE concurrent on M-series** — 3-5 substrates parallel; each canonical-helper-invocation thin (~30 LOC substrate-side)

### Anti-patterns (DO NOT)

- Per-substrate long-training code: each substrate writes its own training loop, checkpoint logic, EMA wrapping, score-aware loss invocation, etc. → DUPLICATIVE CODE that violates operator directive + Catalog #299
- Substrate-coupled trainer infrastructure: training helper hard-codes substrate-specific architecture details (e.g. `dreamer_v3_specific_training_loop()`) → BREAKS COMPOSABILITY
- Skip-L2-infra-and-spawn-L2-training: each substrate gets ad-hoc training code → DUPLICATIVE + non-production-hardened + non-OSS-clean

## META work continues in parallel (regardless of cascade level)

These work streams remain canonical at ALL cascade levels:
- **Wave #1 posterior_emission_canonical_wire_in** (task #1290, agent `ae7d6276a7902bdf5`, in-flight) — lifts all substrates into 62-cathedral-consumer cascade
- **CONSOLIDATE-OP-1 canonical MLX primitive extraction** (task #1292, agent `a087828641070c158`, in-flight) — prevents drift bug class recurrence
- **R2-COMBINED 7-substrate 3-axis review** (task #1293, agent `acb5b7dae5bd09896`, in-flight) — recursive adversarial review per binding directive #3
- **Future R1''** (covers H+I+J+K NEW landings; not yet spawned)
- **Future FIX-WAVE-R2-COMBINED** (conditional on R2-COMBINED findings)
- **Future R3+R3'** (3-clean-pass gate per CLAUDE.md "Recursive adversarial review protocol")
- **Future audit waves** (Wave #2 substrate_contract / Wave #3 canonical_equation_registry / Wave #4 probe-disambiguator / Wave #5 from AUDIT roadmap `e757bb74c`)

## 6-hook wire-in declaration (Catalog #125)

- hook #1 sensitivity-map = ACTIVE (L3 sweep arms emit per-arm sensitivity contributions per `tac.sensitivity_map`)
- hook #2 Pareto constraint = ACTIVE (L3 sweep arms feed Pareto polytope via per-axis decomposition per Catalog #356)
- hook #3 bit-allocator = ACTIVE (L5 OPTIMIZATION leverages `tac.bit_allocator` per Catalog #1068)
- hook #4 cathedral autopilot dispatch = ACTIVE (per-arm canonical posterior anchors per Wave #1 wire-in → 62 cathedral consumers consume)
- hook #5 continual-learning posterior = ACTIVE (canonical anchor per L1/L2/L3/L4/L5/L6 advance via `tac.continual_learning.posterior_update_locked` per Catalog #128)
- hook #6 probe-disambiguator = ACTIVE (L4 cargo-cult unwind v2 IS the canonical disambiguator per substrate-class trajectory)

## Mission contribution per Catalog #300

`frontier_breaking_enabler` + `frontier_protecting` — establishes the canonical substrate-development trajectory that:
- (a) MAXIMIZES research throughput via MLX-local $0 ($5-30 total Path 3 paid spend vs prior $100-500+ framing)
- (b) STRUCTURALLY PREVENTS skip-level bug classes (e.g. C6 IBPS 22× miss from skipping L2 post-training Tier-C validation)
- (c) FORCES each substrate to fully exhaust MLX-local search space (L2 long training + L3 sweeps + L4 iteration + L5 optimization) BEFORE bridge calibration + submission
- (d) ENABLES parallel concurrent substrate development on M-series Apple Silicon (3-5 substrates training concurrently)
- (e) ESTABLISHES canonical reference for future Path 3 + Path 4+ substrates

## Discipline applied

Catalog #229 PV + #117/#157/#174 canonical serializer with POST-EDIT `--expected-content-sha256` + #119 Co-Authored-By + #110/#113 APPEND-ONLY (NEW canonical doctrine file; zero mutation of existing memos) + #208 docs/local-paths + #230 sister-subagent ownership map (zero file collision; doctrine doc + inventory amendment) + #287 placeholder-rationale rejection + #287 + #299 quota brake (no new STRICT gates introduced; uses existing Catalog references throughout) + #300 v2 frontmatter + #340 sister-checkpoint guard (passes structurally).

## Cross-references

- `path_3_candidate_inventory_for_next_wave_spawning_20260526.md` (sister inventory; will be amended to reference this canonical doctrine + flag Tier 3 as DEFERRED)
- `mlx_candidate_contest_equivalence_gate_landed_20260526.md` (#1265 gate canonical reference for L6 advancement)
- `path_3_d_z6_l1_promotion_landed_20260526.md` (D=Z6 L1 reference pattern; canonical L1-promotion shape)
- `path_3_optimization_tooling_audit_and_wire_in_roadmap_20260526.md` (AUDIT META findings + Wave #1-5 roadmap; META work parallel to cascade)
- CLAUDE.md "MLX portable-local-substrate authority" (canonical non-promotable markers per Catalog #127/#192/#317/#341)
- CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" (paid CUDA boundary at L6+ only)
- CLAUDE.md "Recursive adversarial review protocol — close paths" (3-clean-pass gate before paid CUDA boundary)
- CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" L1+L2+L7 (canonical scorer-aware + export-first + substrate-engineering-unique-ifies-bolt-ons-share)
- CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium" Catalog #325 (per-substrate symposium 6-step contract; required at L1 gate)
- CLAUDE.md "Substrate MUST be at OPTIMAL FORM before paid empirical dispatch" Catalog #315 (council verdict gating; required at L6 → submission boundary)
- Catalog #355 META-LAGRANGIAN-WIRE-1 Phase 1 (cathedral_autopilot consumes per-arm Lagrangian per L3 sweep)
- Catalog #356 per-axis decomposition (per-arm contribution per L3 sweep)
- Catalog #357 dual-tier consumer architecture (Tier A observability-only at L1+; Tier B score-contributing at L6+)

EOF

---

## L6 gate 3-verdict map (T3 grand council `7d04474cb` op-routable #7)

**APPEND-ONLY** per Catalog #110/#113 HISTORICAL_PROVENANCE. This section
appends the canonical 3-verdict map for the L6 cascade gate per T3 grand
council deliberation `7d04474cb` (PROCEED_WITH_REVISIONS verdict; 24-of-26
council attendees + 3 binding revisions captured). The original doctrine
body above remains unchanged; this section adds the L6 gate verdict
taxonomy that ALL future Path 3 (and Path N) substrate cascades MUST
honor at the L6 → bridge calibration boundary.

### Canonical empirical anchor (per CLAUDE.md "Apples-to-apples evidence discipline")

The T3 grand council reasoned with PRE-DRIFT-CHAR n=2 empirical (300ep +
1000ep extrapolated; α~1.5 super-linear assumed). The canonical empirical
anchor is now the **DRIFT-VS-DEPTH-CHAR-D-Z6 n=5 fit** (commit `60a9de751`
landed 2026-05-26):

| epochs | drift | margin to 0.001 | safety factor |
|-------:|------:|----------------:|--------------:|
| 300 | 0.000253 | 0.000747 | 3.95× |
| 500 | 0.000358 | 0.000642 | 2.79× |
| 1000 | 0.000458 | 0.000542 | 2.18× |
| 2000 | 0.000721 | 0.000279 | 1.39× |
| 3000 | 0.000725 | 0.000275 | 1.38× |
| ~4973 (predicted crossing) | 0.001000 | 0.000000 | 1.00× |

**Fit**: `drift = 1.8105e-5 * epochs^0.4713`; R²=0.971; **sub-linear** with
saturation observed at 2000→3000ep (+0.5% drift growth for 50% more
training; consistent with EMA equilibrium + per-pair gradient noise floor
combining to bound drift asymptotically). The council's stale n=2
extrapolation (~1000ep threshold-crossing) is FALSIFIED at n=5; actual
extrapolated threshold-crossing is ~4973ep — ~5× headroom over the L3
sweep operating point (500-1500ep).

Canonical equation registered per Catalog #344:
`mlx_pytorch_drift_vs_training_depth_z6_v1`
(`.omx/state/canonical_equations_registry.jsonl`).
Empirical artifact paths per anchor:
`[empirical:experiments/results/z6_drift_vs_depth_{300,500,1000,2000,3000}ep_20260526T*/gate_1265_verdict.json]`.

### Canonical 3-verdict taxonomy (extends cascade doctrine §L6)

Per T3 council Decision 4 + sister R1''-K canonical equation
`mlx_matmul_drift_m_series_canonical_floor_v1` (Catalog #344;
commit `2d59283d4`) 3-verdict taxonomy + Catalog #341 Tier A
non-promotable markers, the canonical L6 gate verdict map is:

| Verdict | Trigger condition | Cascade-progression rule | Per-class example |
|---|---|---|---|
| **BIT_EXACT_LIKE_SINUSOIDAL** | substrate decoder is structurally byte-identical to PyTorch reference (max_abs ~1e-7 floor of FP32 ULP); typical of structural-decoder primitives (no matmul accumulation in critical path) | **PROCEED to L6 paid CUDA WITHOUT bridge calibration**; substrate-class is trivially-CUDA-faithful by construction; bridge calibration would be a $0 measurement that confirms what the structural property already guarantees | I=Faiss IVF-PQ residual decoder (commit `1f929127a`; FIX-WAVE-R1''-I empirically measured max_abs=0.0 at canonical-config dims via `mlx_pq_codebook_gather` + `mlx_pq_reconstruct_tile_vectors` + sister numpy reference; no accumulation in decoder critical path) |
| **WITHIN_CANONICAL_FLOOR** | substrate decoder drift sits within R1''-K canonical floor (abs_max ≤ 6e-2 + rms ≤ 1.5e-2; sister #1265 gate's 0.001 threshold provides ≥1× safety factor at training-depth operating point per drift-vs-depth empirical) | **PROCEED to L6 paid CUDA WITH bridge calibration**; one-time per substrate-class ~$2-8 (see MLX-first doctrine amendment per OP #8 below); after calibration lands, ALL future same-class substrates trust MLX-local routing without re-calibration | D=Z6 predictive coding world model (commit `60a9de751`; n=5 drift fit sub-linear sat ~2000ep; current 1000ep operating point has 2.18× safety factor over 0.001 threshold; bridge calibration codifies the per-class MLX→PyTorch decoder parity bound) |
| **ABOVE_CANONICAL_FLOOR_NEEDS_MITIGATION** | substrate decoder drift exceeds R1''-K canonical floor (abs_max > 6e-2 OR rms > 1.5e-2) OR exceeds sister #1265 threshold even with maximum training-depth budget AND Class 1-SCOPED Kahan-EMA mitigation already applied | **DEFER substrate to Tier A PROXY-grade** per Catalog #341 non-promotable markers UNTIL canonical primitive hardening lands per T3 Class 1-SCOPED Kahan-EMA on `PolyakEMAShadow.update()` per L2-INFRA-BUILD canonical helper (in-flight `a075fe299ca54fe3a`); rerun L2 at extended depth budget post-mitigation; re-evaluate L6 verdict | K=COIN++ implicit neural representation (commit `2d59283d4` FIX-WAVE-R1''-K; per-matmul drift at K-typical (64,256)@(256,64) dims is 4.60e-2 abs / 1.24e-2 rms; if K's per-pair forward composes >50 matmul ops the accumulated drift can exceed sister #1265 threshold; Kahan-EMA mitigation pending T3 Class 1-SCOPED landing) |

### Sister #1265 gate parameterization implication

Per T3 council Decision 3 + DRIFT-VS-DEPTH-CHAR empirical refutation of
the n=2 extrapolation:

- **Current 0.001 threshold is APPROPRIATE for current Path 3 ceiling
  ~1000ep operating point** per the DRIFT empirical (2.18× safety factor
  at 1000ep; ~5× headroom over predicted crossing at 4973ep)
- **No immediate parameterization change needed**; canonical L3 sweep
  operating point (500-1500ep) safely under threshold
- **Deferred `--gate-threshold-epoch-aware` flag** per T3 Tier 2
  op-routable #5 priority: lower-priority canonical flag wiring for the
  Z6 sister gate (`tools/gate_mlx_candidate_contest_equivalence_z6.py`)
  consuming `mlx_pytorch_drift_vs_training_depth_z6_v1` equation predictor
  for epoch-aware threshold; defaults FALSE; activates depth-aware
  threshold for L3 sweep arms that explicitly opt in; sister cathedral
  consumer auto-discovers per Catalog #335
- **PR95 canonical gate (`tools/gate_mlx_candidate_contest_equivalence.py`)
  remains at hard-coded 0.001** per PR95Author dissent: PR95 anchor
  0.000011 is static-decoder reference (untrained state_dict-equivalent),
  not trained-weight drift; the 0.001 threshold defines the SAFETY margin
  for the static bridge calibration; trained-weight drift is a DIFFERENT
  axis with per-substrate-class empirical anchors per the DRIFT subagent's
  cross-substrate impact section

### Per-substrate cascade L6 verdict assignment (current 2026-05-26)

Each Path 3 substrate's L6 verdict (predicted; subject to per-substrate
empirical confirmation at L6 advance):

| Substrate | Predicted L6 verdict | Rationale |
|---|---|---|
| D=Z6 predictive coding | WITHIN_CANONICAL_FLOOR | n=5 empirical at sub-linear sat fit; 2.18× safety factor @ 1000ep |
| I=V1 Faiss IVF-PQ residual | BIT_EXACT_LIKE_SINUSOIDAL | structural-decoder no accumulation; FIX-WAVE-R1''-I max_abs=0.0 anchor |
| K=COIN++ implicit neural representation | ABOVE_CANONICAL_FLOOR_NEEDS_MITIGATION | per-matmul drift exceeds floor at K-typical dims; pending Kahan-EMA Class 1-SCOPED + per-class characterization |
| A=DreamerV3 RSSM | WITHIN_CANONICAL_FLOOR (predicted) | matmul-bound RSSM transformer-class; HNeRV-sister-class drift expected |
| E=BoostNeRV against PR110 | WITHIN_CANONICAL_FLOOR (predicted) | residual-against-PR110 stacking; HNeRV-sister-class |
| G=NIRVANA cascading NeRV | WITHIN_CANONICAL_FLOOR (predicted) | cascading NeRV variant; HNeRV-sister-class |
| F=Z8 canonical-quadruple | WITHIN_CANONICAL_FLOOR (predicted) | quadruple-hierarchy; predictive-coding-sister-class |
| C'=NSCS06 v8 chroma_lut | WITHIN_CANONICAL_FLOOR (predicted) | chroma-LUT replacement; structural primitive class |
| B'=Z7-Mamba-2-v2 | WITHIN_CANONICAL_FLOOR (predicted) | Mamba-2 SSM; HNeRV-sister-class matmul-bound |
| H=ATW V2 cooperative-receiver v2 | WITHIN_CANONICAL_FLOOR (predicted) | cooperative-receiver; matmul-bound |
| J=MDL-IBPS DISCRETE-CATEGORICAL | WITHIN_CANONICAL_FLOOR (predicted) | DISCRETE-CATEGORICAL-MINE-HYBRID; predictive-coding-sister-class |

**Per-substrate empirical L6 verdict TBD at L6 advance** — each substrate
runs its own drift-vs-depth characterization sister landing per the D=Z6
canonical reference pattern (commit `60a9de751`) before L6 verdict
assignment. The table above is the CURRENT PREDICTED VERDICT per substrate-class
analogy; empirical verification at L6 advance is non-skippable.

### Cross-references (this amendment)

- T3 grand council deliberation `7d04474cb`
  (`.omx/research/t3_grand_council_mlx_pytorch_drift_accumulation_source_and_engineer_away_20260526.md`;
  PROCEED_WITH_REVISIONS verdict; 24-of-26 attendees; 3 binding revisions)
- DRIFT-VS-DEPTH-CHAR-D-Z6 landing `60a9de751`
  (`.omx/research/path_3_d_z6_drift_vs_training_depth_characterization_landed_20260526T125130Z.md`;
  n=5 empirical fit α=0.47 sub-linear sat ~2000ep; canonical empirical
  anchor for this amendment)
- R1''-K canonical floor `2d59283d4`
  (`.omx/research/mlx_first_everywhere_canonical_doctrine_20260526.md` §M-series
  MPS fp32 hardware floor canonical anchor; canonical equation
  `mlx_matmul_drift_m_series_canonical_floor_v1` per Catalog #344;
  3-verdict taxonomy that THIS amendment's L6 verdict map inherits)
- FIX-WAVE-R1''-I byte-identical anchor `1f929127a`
  (drift-free structural-decoder class exemplar for BIT_EXACT_LIKE_SINUSOIDAL
  verdict; max_abs=0.0 empirical at canonical-config dims)
- Catalog #341 Tier A non-promotable markers (`tac.cathedral_consumers.*`
  canonical-routing markers; ABOVE_CANONICAL_FLOOR_NEEDS_MITIGATION
  verdict routes here)
- Catalog #344 canonical equation registry (sister equations
  `mlx_pytorch_drift_vs_training_depth_z6_v1` + `mlx_matmul_drift_m_series_canonical_floor_v1`)
- Catalog #1265 sister gate (`tools/gate_mlx_candidate_contest_equivalence.py`
  PR95 canonical + `tools/gate_mlx_candidate_contest_equivalence_z6.py`
  Z6 sister; threshold parameterization deferred per Tier 2 priority)
- CLAUDE.md "Apples-to-apples evidence discipline" (n=5 empirical anchor
  supersedes n=2 stale extrapolation in council's pre-DRIFT reasoning)
- CLAUDE.md "Forbidden premature KILL without research exhaustion"
  (ABOVE_CANONICAL_FLOOR verdict is DEFER not KILL; routes to Class 1-SCOPED
  Kahan-EMA mitigation per T3 op-routables #2+#3)

EOF
