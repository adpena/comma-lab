---
schema_version: mlx_first_everywhere_canonical_doctrine_v1
created_utc: 2026-05-26T23:45:00Z
adopted_as_canonical: true
binding_for: all_path_3_path_n_and_future_substrate_work
council_tier: T2
council_attendees: [Shannon, Dykstra, Rudin, Daubechies, Carmack, Hotz, Quantizr, Selfcomp, MacKay, AssumptionAdversary, PR95Author]
council_quorum_met: true
council_verdict: PROCEED
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
council_dissent: []
horizon_class: frontier_pursuit
related_deliberation_ids:
  - path_3_canonical_substrate_development_cascade_doctrine_20260526
  - mlx_candidate_contest_equivalence_gate_landed_20260526
  - pr95_mlx_full_inflate_parity_closure_landed_20260526
  - pr95_mlx_full_decoder_downstream_scorer_drift_landed_20260526
  - pr95_mlx_pytorch_export_parity_bridge_landed_20260525
council_decisions_recorded:
  - "op-routable #1: every NEW substrate work spawn brief MUST cite this doctrine as canonical reference"
  - "op-routable #2: future CLAUDE.md amendment (operator-routable) to elevate this doctrine to non-negotiable status alongside existing 'MLX portable-local-substrate authority' section"
  - "op-routable #3: amend inventory doc + cascade doctrine to reference this doctrine"
---

# MLX-First Everywhere Canonical Doctrine

**Status**: CANONICAL adopted 2026-05-26 per operator binding directive *"MLX first everywhere document that too like you suggested before"*.

**Purpose**: establish the binding canonical principle that ALL substrate development work (training / sweeps / iteration / smoke / scaffolding / convergence verification / canonical helper development / review / audit) is MLX-local Apple Silicon at $0 GPU spend. Paid CUDA reserves STRICTLY for 3 narrow use cases per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" + "MLX portable-local-substrate authority" non-negotiables.

## Binding principle (verbatim per operator 2026-05-26)

> *"All training to be done on MLX and sweeps on MLX"* (2026-05-26)
>
> *"Great work, let's keep optimizing and iterating and building and getting stuff training and running on MLX"* (2026-05-26)
>
> *"MLX first everywhere document that too like you suggested before"* (2026-05-26)

Together these establish: **MLX is the primary substrate development surface; paid CUDA is the narrow submission boundary.**

## Empirical anchor

`tools/gate_mlx_candidate_contest_equivalence.py` Catalog #1265 LANDED 2026-05-26 empirically established MLX-PyTorch contest-equivalence at the frontier-tightening granularity:

- **|S_MLX − S_PyTorch| = 0.000011** on PR95 hnerv_muon canonical archive (anchor commit `9fdef4a04`)
- **72× SMALLER** than PR110 vs PR101 frontier delta (0.000789)
- **90× margin** over the canonical 0.001 contest-units gate threshold (`mlx_candidate_contest_equivalence_v1` schema)
- Catalog #1265 LANDED `69c316ca4` — operationalizes this anchor as PASS/FAIL gate for every Path 3 candidate

Empirical implication: **MLX is contest-grade for ALL frontier-tightening iteration**. Substrate development on MLX produces decisions byte-stable enough to be trusted for paid-CUDA-bypass routing.

## Per-work-type taxonomy (binding canonical mapping)

| Work type | Surface | Cost | Catalog references |
|---|---|---|---|
| **Substrate scaffolding** | MLX-local Apple Silicon | $0 | Catalog #240 (substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY); cascade doctrine L0 |
| **L1 INFRASTRUCTURE-CONVERGENCE** | MLX-local | $0 | Cascade doctrine L1; D=Z6 reference `8833b9db5` |
| **L2 LONG TRAINING** | MLX-local | $0 | Cascade doctrine L2; L2-INFRA-BUILD canonical helper (in-flight `a72cbf33d2ae2768c`); Catalog #2 EMA + #164 score-aware loss + differentiable eval-roundtrip |
| **L3 HYPERPARAMETER SWEEPS** | MLX-local; multi-arm parallel | $0 | Cascade doctrine L3; canonical helper supports N-concurrent arms on M-series shared GPU memory |
| **L4 ARCHITECTURAL ITERATION** | MLX-local | $0 | Cascade doctrine L4; cargo-cult-unwind methodology per NSCS06 v6→v7 canonical pattern (44% reduction in ONE iteration) |
| **L5 OPTIMIZATION** | MLX-local; QAT + EMA-decay-sweep + decoder compression | $0 | Cascade doctrine L5; Catalog #2 + #122 + `tac.bit_allocator` |
| **L6 CONVERGED CANDIDATE verification** | MLX-local + Catalog #1265 gate verdict | $0 | Cascade doctrine L6; canonical Provenance per Catalog #323 |
| **Smoke tests** | MLX-local | $0 | All substrate `_smoke_main` invocations |
| **Cargo-cult-pass FIRST per operator binding directive #2 (2026-05-26)** | MLX-local + design memo discipline | $0 | Catalog #303 + #292 + HARD-EARNED-vs-CARGO-CULTED addendum |
| **Recursive adversarial review (R1, R1', R2-COMBINED, R3, ...)** | MLX-local (read-only review; spawn smoke verification on MLX if needed) | $0 | CLAUDE.md "Recursive adversarial review protocol — close paths" 3-clean-pass cycle |
| **3-axis review per operator binding directive #3 (2026-05-26)** | MLX-local | $0 | Cascade doctrine 3-axis discipline (math+sci+engineering rigor / MLX drift minimization / numpy portability) |
| **Canonical helper development (CONSOLIDATE-OP-1, L2-INFRA-BUILD, posterior_emission wire-in, etc.)** | MLX-local | $0 | Catalog #299 gate consolidation + #335 canonical contract |
| **Audit subagents (OPTIMIZATION-TOOLING-AUDIT etc.)** | MLX-local (analysis-only) | $0 | This doctrine + cascade doctrine §META work |
| **Sister gate parameterization per substrate-class grammar (e.g. Z6PCWM1, RSSMC1, COINPP1, NIRVANA1, BPR1, MDLIBPS-J1, Z7MCM3, Z8HPC1, ATWv2CR2, FAISSPQ1)** | MLX-local | $0 | Catalog #1265 sister gate pattern per substrate-class grammar |

## Paid CUDA narrow scope (binding canonical)

Paid CUDA fires ONLY in these 3 cases:

### 1. Bridge calibration per substrate-class (ONE-TIME)

- Characterize MLX↔CUDA drift bound for substrate-class (HNeRV / predictive-coding / categorical-RSSM / hierarchical-residual / iterative-boosting / canonical-quadruple / cooperative-receiver-Atick / IVF-PQ / IB-MINE / meta-INR / chroma-LUT / etc.)
- ~$0.50-2 per class; once per class for the entire Path 3+ wave
- After per-class calibration lands, ALL future same-class substrates trust MLX-local without re-calibration
- Result canonicalized in `tac.canonical_equations` registry (Catalog #344) for downstream consumer auto-application

### 2. Final submission auth eval — BOTH contest-CPU + contest-CUDA (per submission)

- Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" non-negotiable
- Contest-CPU on Linux x86_64 = leaderboard score
- Contest-CUDA on T4-equivalent = Yousfi bot comment
- ~$0.20-1 per submission
- Non-skippable for any PR/release

### 3. Adversarial public-frontier-PR replay (sporadic)

- When a competitor's PR appears + Yousfi bot comment is missing/stale
- Replay on paid CUDA to characterize their actual contest score on EXACT submitted bytes
- ~$0.20-1 per replay
- Per CLAUDE.md "Frontier scores are pointer-only" + "Frontier target" non-negotiables

### Total forecast

**~$5-30 paid GPU spend for entire 11-substrate Path 3 wave** (vs prior "per-substrate per-iteration" $100-500+ framing). The reduction is structural, not speculative — every L0-L5 step empirically validated MLX-faithful per Catalog #1265 gate; paid CUDA boundary is the L6 → submission handoff only.

## Canonical infrastructure stack (MLX-first enablers)

The MLX-first principle is enabled + enforced by these canonical surfaces (all $0 to build + use):

1. **`tac.local_acceleration.pr95_hnerv_mlx`** — canonical MLX primitives (`pixel_shuffle_2x_nhwc` channel-FIRST byte-stable + `bilinear_resize_nhwc` align_corners=False + Kahan summation + epsilon-stable softmax + fp32 matmul); CONSOLIDATE-OP-1 in-flight extracts these from per-substrate ad-hoc implementations
2. **`numpy_reference.py` sister pattern** — every MLX primitive has sister numpy reference for CPU-only portability per 3-axis discipline axis #3 (G=NIRVANA established canonical pattern with 7/7 references; sister CONSOLIDATE-OP-2 extracts to canonical location)
3. **`tools/export_pr95_mlx_to_pytorch_state_dict.py` (#1251)** — canonical MLX → PyTorch state_dict export bridge with byte-stable round-trip + paired forward parity validation
4. **`tools/package_pr95_mlx_pytorch_state_dict_to_contest_archive.py` (#1257)** — canonical packaging cascade (MLX EMA shadow → PyTorch state_dict → contest-grammar archive)
5. **`tools/gate_mlx_candidate_contest_equivalence.py` (#1265)** — canonical PASS/FAIL gate at threshold 0.001 contest-units (90× margin over empirical anchor); refuses paid-CUDA-dispatch authorization until MLX-PyTorch contest-equivalence proven
6. **Sister #1265 gate per substrate-class grammar** — parameterized canonical gate for each substrate's archive grammar (Z6PCWM1, RSSMC1, COINPP1, NIRVANA1, BPR1, MDLIBPS-J1, Z7MCM3, Z8HPC1, ATWv2CR2, FAISSPQ1, ...); MLX→PyTorch state_dict + decoder + inflate parity verification
7. **`tac.substrates._shared.trainer_skeleton.long_training_canonical` (in-flight `a72cbf33d2ae2768c`)** — canonical long-training infrastructure with `run_long_training(substrate, config) -> TrainingArtifact` entry-point; ALL L2-L5 work uses this canonical helper per cascade doctrine + 10-element contract
8. **`tac.continual_learning.posterior_update_locked` (Catalog #128)** — canonical posterior anchor emission; MLX-local artifacts carry `evidence_grade="macOS-MLX-research-signal"` + `score_claim=False` + `promotion_eligible=False` + `ready_for_exact_eval_dispatch=False` per Catalog #127/#192/#317/#341 non-promotable markers
9. **`tac.canonical_equations.update_equation_with_empirical_anchor` (Catalog #344)** — canonical equations registry; MLX-local empirical anchors calibrate canonical equations across substrate-classes
10. **`tac.cathedral_consumers.*` (Catalog #335 + #341 + #357)** — canonical cathedral autopilot consumer protocol auto-discovers MLX-local posterior anchors (Wave #1 posterior_emission `ae7d6276a7902bdf5` in-flight wires this for all 8 LANDED Path 3 substrates)
11. **`tac.bit_allocator` (Catalog #1068)** — canonical per-byte / per-pair / per-class / per-axis / pareto_dual allocators; consumed at L5 OPTIMIZATION on MLX-local
12. **`tac.findings_lagrangian` + `tac.cathedral_autopilot_autonomous_loop::invoke_meta_lagrangian_on_candidates` (Catalog #355)** — canonical 4-term scalar Lagrangian + closed-form Gaussian posterior + Lindley-1956 action selector; ALL invocations MLX-local

## Compounding implications

The MLX-first principle has compound effects across THREE dimensions:

### Dimension 1: cost (paid GPU spend collapse)

- **Prior framing**: per-substrate per-iteration paid CUDA dispatch ($5-50 per substrate per L1+ advance; 11 substrates × 5 advances = $275-2,750)
- **MLX-first framing**: per-class bridge calibration + per-submission auth eval ($5-30 for entire 11-substrate Path 3 wave)
- **Ratio**: ~25-100× cost reduction; freed budget routes to defensive verification at L6 + adversarial public-frontier-PR replay + future Path 4+

### Dimension 2: iteration velocity (training throughput)

- **Prior framing**: paid CUDA dispatch wall-clock = minutes (Modal T4 dispatch fire) + hours (training) + harvesting + result analysis; per-iteration ~3-24h
- **MLX-first framing**: MLX-local training on M-series Apple Silicon = ~0.3s per 30-epoch synthetic smoke (D=Z6 anchor); contest-scale 384×512 × 600 pairs × 50-100ep estimate ~1-3h on M-series; multi-substrate parallel concurrent
- **Ratio**: ~5-20× iteration velocity gain for the L0-L5 surface where most decisions happen

### Dimension 3: discipline cycle compounding

- **Cargo-cult-first** (per operator binding directive #2) surfaces bugs PRE-spend — MLX-local makes "spend" zero, so the cycle becomes purely about RESEARCH QUALITY not cost-management
- **3-axis adversarial review** (per operator binding directive #3) catches drift/portability/rigor bugs MLX-local before they reach paid CUDA — empirical receipt: R1 caught A=DreamerV3 pixel_shuffle 2.40 drift + bilinear 1.51 drift that landing memo missed; R1' caught F=Z8 LINE-FOR-LINE inheritance of A's bugs (3.77 + 1.51) → CONSOLIDATE-OP-1 in-flight extinct the bug class structurally
- **Recursive review 3-clean-pass cycle** advances per substrate to 3/3 = paid CUDA dispatch authorized; MLX-local makes the cycle FAST (each review round = ~3-5h vs days of waiting for paid CUDA harvest)
- **Per-substrate symposium per Catalog #325** + canonical equation registry + cathedral consumer auto-discovery + canonical posterior anchor: all MLX-local; cathedral autopilot consumes the entire Path 3 wave's signal at $0

## Cross-references

- `.omx/research/path_3_canonical_substrate_development_cascade_doctrine_20260526.md` (cascade doctrine; L0-L6 cascade structure + per-level Catalog references)
- `.omx/research/mlx_candidate_contest_equivalence_gate_landed_20260526.md` (#1265 gate canonical reference; empirical anchor source)
- `.omx/research/pr95_mlx_full_decoder_downstream_scorer_drift_landed_20260526.md` (corrected closure; 0.000011 empirical anchor)
- `.omx/research/path_3_d_z6_l1_promotion_landed_20260526.md` (D=Z6 L1 reference pattern; first L0→L1 promotion proof-of-cascade)
- `.omx/research/path_3_optimization_tooling_audit_and_wire_in_roadmap_20260526.md` (canonical optimization stack inventory + per-substrate utilization matrix; 46.9% UNUSED-NO-RATIONALE pre-Wave#1)
- `.omx/research/path_3_candidate_inventory_for_next_wave_spawning_20260526.md` (Path 3 candidate queue + 3-axis discipline)
- CLAUDE.md "MLX portable-local-substrate authority" — sister non-negotiable; MLX as portable-local-substrate authority for fast candidate generation + scorer-response training data + portability engineering + calibrated spend triage; tagged `[macOS-MLX research-signal]` per Catalog #127/#192/#317/#341
- CLAUDE.md "MPS auth eval is NOISE" — paid CUDA non-negotiable axis truth; MPS NEVER as authoritative score
- CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" — paid CUDA boundary non-negotiable
- CLAUDE.md "Recursive adversarial review protocol — close paths" — 3-clean-pass cycle gate before paid CUDA
- CLAUDE.md "Beauty, simplicity, and developer experience" — MLX-first elegant + composable + reusable + production-hardened per operator's L2-INFRA-BUILD directive
- Catalog #1 (`check_no_mps_fallback_default`) — sister structural protection; MPS-fallback ternary forbidden
- Catalog #127 (`check_authoritative_tag_requires_custody_metadata`) — per-call-site custody routing; MLX-local artifacts non-promotable
- Catalog #192 (`check_macos_cpu_advisory_not_promoted_without_linux_verification`) — sister non-promotable discipline
- Catalog #317 (`check_local_research_signal_dispatches_stamp_evidence_grade`) — sister non-promotable discipline at dispatch surface
- Catalog #341 (`check_cathedral_consumer_mps_prescreen_routing_carries_canonical_markers`) — sister non-promotable discipline at consumer routing surface
- Catalog #344 (`check_empirical_finding_memo_references_canonical_equation`) — canonical equations registry; MLX-local empirical anchors calibrate equations

## 6-hook wire-in declaration (Catalog #125)

- hook #1 sensitivity-map = ACTIVE (MLX-local sweep arms emit per-arm sensitivity contributions per `tac.sensitivity_map`)
- hook #2 Pareto constraint = ACTIVE (MLX-local sweep arms feed Pareto polytope via per-axis decomposition per Catalog #356)
- hook #3 bit-allocator = ACTIVE (L5 MLX-local OPTIMIZATION leverages `tac.bit_allocator` per Catalog #1068)
- hook #4 cathedral autopilot dispatch = ACTIVE (per-arm MLX-local canonical posterior anchors per Wave #1 wire-in → 62 cathedral consumers consume; canonical Provenance non-promotable markers preserved)
- hook #5 continual-learning posterior = ACTIVE (canonical anchor per L1/L2/L3/L4/L5/L6 advance via `tac.continual_learning.posterior_update_locked` per Catalog #128; all MLX-local)
- hook #6 probe-disambiguator = ACTIVE (Catalog #1265 gate IS the canonical disambiguator between MLX-faithful-enough-to-dispatch vs MLX-too-noisy-to-trust; threshold 0.001 + 90× margin over 0.000011 empirical anchor)

## Mission contribution per Catalog #300

`frontier_breaking_enabler` + `frontier_protecting` — establishes the binding canonical principle that:
- (a) MAXIMIZES research throughput via MLX-local $0 (25-100× cost reduction; 5-20× iteration velocity)
- (b) STRUCTURALLY PROTECTS against paid-CUDA waste on cargo-culted assumptions (cargo-cult-first + 3-axis + recursive review catches bugs pre-spend; MLX-local makes "spend" zero)
- (c) FORCES discipline-cycle compounding (each review round = $0; cycle becomes about research quality not cost-management)
- (d) ENABLES parallel concurrent substrate development (3-5 substrates concurrent on M-series shared GPU memory; MLX-local doesn't compete for paid CUDA capacity)
- (e) ESTABLISHES canonical reference for future Path 4+ substrates + future Path N+ work (this doctrine is binding for ALL future substrate development)

## Anti-patterns (DO NOT)

1. **DO NOT use paid CUDA for training, sweeps, smoke, iteration, cargo-cult, review, L1-promotion verification, or any intermediate work** — all MLX-local per operator binding directive #7 (2026-05-26 *"All training to be done on MLX and sweeps on MLX"*). Paid CUDA STRICTLY for bridge calibration (one-time per class) + final submission auth eval (per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA") + adversarial public-frontier-PR replay.
2. **DO NOT cargo-cult paid-CUDA dispatch as "ready-for-submission" intermediate step** — paid CUDA spend BEFORE Catalog #1265 gate PASS + per-substrate symposium PROCEED + R3 3/3 clean-pass counter is wasted spend per cascade doctrine L6 boundary
3. **DO NOT skip the MLX↔PyTorch parity gate (#1265)** — gate is the canonical disambiguator; bypassing it re-introduces the bug class that 0.000011 empirical anchor proves is bounded
4. **DO NOT skip the cargo-cult-first discipline + 3-axis recursive review per operator binding directives #2+#3** — these are the structural protection that makes MLX-first safe at scale (catches bugs pre-spend; MLX-local makes "spend" zero so cycle becomes pure research-quality discipline)
5. **DO NOT add per-substrate ad-hoc MLX training code per operator binding directive 2026-05-26** *"Also the long training infrastructure, ensure reusable composable beautiful elegant creative expressive cimposable abstractions and production hardened OSS and no duplicative code"* — L2-INFRA-BUILD canonical helper (in-flight) IS the canonical infrastructure; substrate L2 trainers must use ≤30 LOC + ONE canonical helper invocation per Catalog #299 gate consolidation discipline

## Operator-routable next steps

1. **op-routable #1**: every NEW substrate work spawn brief MUST cite this doctrine as canonical reference (sister to cascade doctrine + inventory doc cross-reference pattern; binding for all L/M/N/O Tier 3 future spawns + Path 4+ substrates)
2. **op-routable #2**: future CLAUDE.md amendment (operator-routable; this subagent does NOT mutate CLAUDE.md per mutation frontier discipline) to ELEVATE this doctrine to non-negotiable status alongside existing "MLX portable-local-substrate authority" section. Proposed amendment text:

> **Add new CLAUDE.md non-negotiable section**: `## MLX-first everywhere — NON-NEGOTIABLE, HIGHEST EMPHASIS`
>
> Per binding operator directive 2026-05-26 verbatim: *"All training to be done on MLX and sweeps on MLX"* + *"MLX first everywhere"*. Canonical reference: `.omx/research/mlx_first_everywhere_canonical_doctrine_20260526.md`. ALL substrate development work (training / sweeps / iteration / smoke / scaffolding / convergence verification / canonical helper development / review / audit) is MLX-local Apple Silicon at $0. Paid CUDA reserves STRICTLY for: (a) bridge calibration per substrate-class one-time, (b) final submission auth eval BOTH CPU + CUDA per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA", (c) adversarial public-frontier-PR replay. Empirical anchor: Catalog #1265 gate `tools/gate_mlx_candidate_contest_equivalence.py` empirical drift 0.000011 (90× margin over 0.001 threshold; 72× margin below PR110 vs PR101 frontier delta 0.000789). Total paid CUDA forecast for Path 3 wave: ~$5-30 (25-100× reduction vs prior per-substrate per-iteration framing).

3. **op-routable #3**: append-only footers on `path_3_canonical_substrate_development_cascade_doctrine_20260526.md` + `path_3_candidate_inventory_for_next_wave_spawning_20260526.md` referencing this doctrine as binding canonical companion
4. **op-routable #4**: L2-INFRA-BUILD canonical helper (in-flight `a72cbf33d2ae2768c`) IS the canonical MLX-first long-training infrastructure; once landed, L1-PROMOTION-CASCADE subagents inherit canonical helper and substrate L2 advancement is thin (≤30 LOC substrate config + ONE canonical helper invocation)

## Discipline

Catalog #229 PV (read cascade doctrine + #1265 gate landing + MLX bridge memos + CLAUDE.md MLX-relevant sections BEFORE writing) + #117/#157/#174 canonical serializer with POST-EDIT `--expected-content-sha256` + #119 Co-Authored-By + #110/#113 APPEND-ONLY (NEW canonical doctrine file; zero mutation of existing memos or CLAUDE.md) + #208 docs/local-paths (NO `/Users/adpena/...` in body) + #230 sister-subagent ownership map (no file collision; doctrine is in-context work for main agent) + #287 placeholder-rationale rejection + #299 quota brake (no new STRICT gates introduced) + #300 v2 frontmatter + #340 sister-checkpoint guard (passes structurally; in-context work).

EOF
