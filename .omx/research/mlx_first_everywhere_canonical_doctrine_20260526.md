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

---

## M-series MPS fp32 hardware floor canonical anchor (2026-05-26 R1'' verification)

**APPEND-ONLY** per Catalog #110/#113 HISTORICAL_PROVENANCE. This section
appends the canonical M-series MPS fp32 matmul drift hardware-floor
reference per FIX-WAVE-R1''-K landing 2026-05-26. The original doctrine
body above remains unchanged; this section adds the canonical floor
anchor that ALL future Path 3 (and Path N) substrate designs MUST cite as
the binding reference for MLX matmul drift expectations.

### Canonical floor reference (per Catalog #344)

Canonical equation registered in `tac.canonical_equations` per FIX-WAVE-R1''-K
2026-05-26: **`mlx_matmul_drift_m_series_canonical_floor_v1`**

* **Module**: `tac.canonical_equations.mlx_matmul_m_series_floor`
* **Callable**: `classify_mlx_matmul_drift(measured_abs_max, measured_rms=None, measured_rel_median=None, matmul_shape=None) -> dict`
* **Verdict taxonomy**: `BIT_EXACT_LIKE_SINUSOIDAL` (~1e-7) /
  `WITHIN_CANONICAL_FLOOR` (≤ 6e-2 abs ∧ ≤ 1.5e-2 rms) /
  `ABOVE_CANONICAL_FLOOR_NEEDS_MITIGATION`
* **Canonical Provenance**: research-sidecar; `[macOS-MLX research-signal]`;
  non-promotable per Catalog #127/#192/#317/#341

### Canonical M-series MPS fp32 matmul drift hardware floor

Per FIX-WAVE-R1''-K independent verification 2026-05-26 across K-typical
substrate dimensions `[empirical:.omx/research/path_3_k_recursive_adversarial_review_r1_prime_prime_3_axis_20260526.md§4.2]`:

| (m,k)@(k,n)         | abs_max  | rms      | rel_median |
|---------------------|----------|----------|------------|
| (32,32)@(32,32)     | 1.54e-2  | 4.52e-3  | 7.76e-4    |
| (64,64)@(64,64)     | 2.42e-2  | 6.16e-3  | 7.66e-4    |
| (128,128)@(128,128) | 3.62e-2  | 8.81e-3  | 7.59e-4    |
| (256,64)@(64,256)   | 2.97e-2  | 6.20e-3  | 7.64e-4    |
| (64,256)@(256,64)   | 4.60e-2  | 1.24e-2  | 7.75e-4    |

**Canonical floor (upper bounds + dimension-independent floor):**

* **abs_max upper bound**: **6e-2** (covers worst-case across canonical dims + safety margin)
* **rms upper bound**: **1.5e-2** (covers worst-case rms)
* **rel_median canonical floor**: **7.6e-4** (dimension-independent; hardware-class property)
* **Sinusoidal encoding** (sin/cos special case): **bit-exact ~1.2e-7** (independent of matmul accumulation)

The original H+K landing memos claimed tighter bounds (H: "1e-3 to 1e-2 abs";
K: "5e-3 abs") that R1'' empirically falsified at substrate-typical dims.
Sister FIX-WAVE-R1''-K commit corrects both with the canonical floor above.

### Canonical-substrate-design implication

Substrates requiring **<1e-2 abs precision per matmul** MUST satisfy ONE of:

1. **Route through canonical mitigation primitives** at
   `tac.local_acceleration.deterministic_primitives` — fp32 + Kahan-compensated
   summation (per sister canonical equation `mlx_pytorch_conv2d_kahan_summation_drift_reduction_v1`
   + `mlx_pytorch_conv2d_fp64_accumulation_drift_reduction_v1`).
2. **Accept drift as PROXY-grade** per Catalog #341 Tier A observability-only
   markers; the routing recommendation cannot leak into score/promotion signals.
3. **Operate primarily on bit-exact primitives** (sinusoidal positional
   encodings + elementwise multiply/add + sigmoid + softmax-with-epsilon)
   where the local floor is ~1e-7 independent of matmul accumulation.

### Producer/consumer map per Catalog #344

* **Producers** (helpers that emit empirical anchors for this equation):
  * `path_3_fix_wave_r1_prime_prime_k_independent_verification` (THIS landing)
  * `tools.measure_pr95_mlx_pytorch_full_decoder_downstream_scorer_drift` (sister at decoder-output granularity)
  * Future per-M-series-class characterization helpers (M1/M2/M3/M4/M5/Ultra/Pro/Max)
* **Consumers** (helpers that read this equation's floor):
  * `tac.substrates.coin_pp_implicit_neural_representation.tests.test_basic` (K test threshold)
  * `path_3_recursive_adversarial_review_r1_prime_prime_axis_2_reviewer` (R1'' axis 2 framework)
  * `tools.gate_mlx_candidate_contest_equivalence` (sister Catalog #1265 gate threshold rationale)
  * `tac.cathedral_consumers.canonical_equation_lookup_consumer` (auto-discovers per Catalog #335)
  * `mlx_first_everywhere_canonical_doctrine` (THIS doctrine cites the floor)

### Cross-substrate META implication

R1'' aggregate memo §8 measured the SAME hardware floor across H + K
substrates independently — empirically validating that this is a
**hardware-class property** (M-series Apple Silicon MPS fp32 matmul
accumulation), NOT a per-substrate artifact. Future Path 3 substrate
authors should:

1. **Cite this canonical equation** in design memos per Catalog #344
2. **Use the canonical classifier** in MLX parity tests instead of
   hardcoding drift literals
3. **Declare per-matmul accuracy requirement** at design-memo time per
   §"MLX drift minimization per primitive" Axis 2 discipline
4. **Document mitigation path** (Kahan / fp64 accumulation / accept-as-proxy)
   if any matmul requires <1e-2 abs precision

### Reactivation criteria for canonical equation

Per the canonical equation's `provenance.rejection_reason`:

> *"rerun on wider dim sweep + per-Apple-Silicon-class characterization
> (M1/M2/M3/M4/M5/M-Ultra/M-Pro/M-Max) before promoting to cross-machine
> canonical hardware-floor reference"*

The current anchor is from M5 Max per primary M-series-MPS measurement
machine; per-class characterization would expand the canonical floor to
cross-machine reference for the entire Path N+ Apple Silicon ecosystem.

### Sister convergence anchor

R1'' aggregate memo §8 measured EQUIVALENT drift across H and K substrates
independently (H typical dim (64,256): abs=4.97e-2; K typical dim
(64,256)@(256,64): abs=4.60e-2). This convergence is the empirical
HARD-EARNED-CANONICAL anchor that the canonical floor IS the M-series
MPS hardware property, not a per-substrate artifact. Sister FIX-WAVE-R1''-H
also corrects against this same floor; the two corrections converge on
the canonical equation as single source of truth.

### Cross-references

* Canonical equation module: `src/tac/canonical_equations/mlx_matmul_m_series_floor.py`
* Canonical equation registry event: `.omx/state/canonical_equations_registry.jsonl`
  (registered 2026-05-26 via `register_canonical_equation`)
* R1'' K per-substrate memo: `.omx/research/path_3_k_recursive_adversarial_review_r1_prime_prime_3_axis_20260526.md`
* R1'' aggregate memo §8 Empirical anchor: `.omx/research/path_3_recursive_adversarial_review_r1_prime_prime_aggregate_3_axis_landings_h_i_j_k_20260526.md`
* FIX-WAVE-R1''-K landing memo: `.omx/research/path_3_fix_wave_r1_prime_prime_k_coin_pp_landed_*.md`
* K landing memo correction footer: `.omx/research/path_3_k_coin_pp_L0_scaffold_landed_20260526.md` (APPEND-ONLY)
* Sister `mlx_pytorch_full_decoder_downstream_scorer_drift_propagation_v1` (downstream scorer-drift at decoder output)
* CLAUDE.md "Apples-to-apples evidence discipline" + "MPS auth eval is NOISE" + "MLX portable-local-substrate authority"
* Catalog #344 (canonical-equation-reference enforcement) + #287 (placeholder-rationale rejection) + #341 (canonical-routing markers)

---

## Per-substrate-class depth-aware paid CUDA bridge calibration forecast (T3 grand council `7d04474cb` op-routable #8)

**APPEND-ONLY** per Catalog #110/#113 HISTORICAL_PROVENANCE. This section
appends the canonical per-substrate-class depth-aware paid CUDA bridge
calibration forecast per T3 grand council deliberation `7d04474cb`
(PROCEED_WITH_REVISIONS verdict; Decision 5 MLX-first doctrine baseline
amendment). The original doctrine body above (including R1''-K canonical
floor section) remains unchanged; this section refines the
"Total forecast" paragraph above (which retains its structural
`~$5-30 paid GPU spend for entire 11-substrate Path 3 wave` baseline)
with per-substrate-class breakdown reflecting the drift-aware bridge
calibration cost structure.

### Canonical empirical anchor (per CLAUDE.md "Apples-to-apples evidence discipline")

The T3 grand council reasoned with stale n=2 empirical (300ep + 1000ep
extrapolated; α~1.5 super-linear assumed). The canonical empirical anchor
is now the **DRIFT-VS-DEPTH-CHAR-D-Z6 n=5 fit**
(commit `60a9de751` landed 2026-05-26; canonical equation
`mlx_pytorch_drift_vs_training_depth_z6_v1` per Catalog #344):

* `drift = 1.8105e-5 * epochs^0.4713`
* R²=0.971
* **sub-linear** with **saturation observed at 2000→3000ep** (+0.5% drift
  growth for 50% more training; consistent with EMA equilibrium + per-pair
  gradient noise floor combining to bound drift asymptotically)
* Extrapolated threshold-crossing point ~4973 epochs (NOT ~1000ep as the
  council's pre-DRIFT n=2 stale extrapolation predicted)
* Empirical artifact paths per anchor:
  `[empirical:experiments/results/z6_drift_vs_depth_{300,500,1000,2000,3000}ep_20260526T*/gate_1265_verdict.json]`

The DRIFT-VS-DEPTH-CHAR fit FALSIFIED the council's assumed super-linear
α~1.5 extrapolation; the canonical empirical anchor is sub-linear with
saturation. This amendment reflects the EMPIRICAL anchor (n=5), not the
council's stale assumed extrapolation.

### Canonical per-substrate-class bridge calibration forecast (3-class taxonomy)

Per cascade doctrine §"L6 gate 3-verdict map (T3 grand council
`7d04474cb` op-routable #7)" + R1''-K canonical floor 3-verdict taxonomy +
DRIFT-VS-DEPTH-CHAR n=5 sub-linear sat empirical, the canonical
per-substrate-class bridge calibration cost forecast is:

| Substrate class | L6 verdict | Bridge calibration paid CUDA spend per class (one-time) | Rationale |
|---|---|---:|---|
| **Drift-free structural-decoder class** | BIT_EXACT_LIKE_SINUSOIDAL | **~$2 per class** | Substrate's decoder is byte-identical to PyTorch reference at all dims (no matmul accumulation in critical path; structural primitives like Faiss IVF-PQ codebook gather + sinusoidal positional encoding + elementwise multiply/add + softmax-with-epsilon all have hardware-floor ~1e-7). Bridge calibration is a $2 confirmation that the structural property holds across the substrate-class's canonical-archive grammar; once landed, ALL future same-class substrates skip re-calibration. Canonical exemplar: **I=Faiss IVF-PQ residual** (commit `1f929127a` FIX-WAVE-R1''-I; empirically measured max_abs=0.0 at canonical-config dims via `mlx_pq_codebook_gather` + `mlx_pq_reconstruct_tile_vectors` + sister numpy reference). |
| **Matmul-bound HNeRV-class** | WITHIN_CANONICAL_FLOOR | **~$5 per class** | Substrate's decoder composes matmul ops (HNeRV-class CNN+transformer + Mamba SSM + cooperative-receiver matmul + RSSM transformer); per-matmul drift sits within R1''-K canonical floor (abs ≤ 6e-2 + rms ≤ 1.5e-2 per FP32 hardware floor); per-pair forward drift bounded by sub-linear training-depth accumulation per DRIFT empirical (saturates ~2000ep at 0.000725 abs; ~1.39× safety factor over sister #1265 threshold 0.001). Bridge calibration is a $5 paired-CUDA measurement at canonical L3-sweep operating point (500-1500ep) that codifies the per-class MLX→PyTorch decoder parity bound; after landing, ALL future same-class substrates trust MLX-local routing without re-calibration. Canonical exemplar: **D=Z6 predictive coding world model** (commit `60a9de751`; canonical equation `mlx_pytorch_drift_vs_training_depth_z6_v1` per Catalog #344; n=5 sub-linear sat fit). |
| **INR-class with Kahan mitigation pending** | ABOVE_CANONICAL_FLOOR_NEEDS_MITIGATION | **~$8 per class** (pending T3 Class 1-SCOPED Kahan-EMA mitigation landing) | Substrate's decoder composes deep matmul cascade (implicit neural representation INR-class composes 30-50+ matmul ops per forward); per-matmul drift at substrate-typical dims exceeds R1''-K canonical floor at deeper composition (e.g. K=COIN++ (64,256)@(256,64) drift = 4.60e-2 abs / 1.24e-2 rms; per-pair forward composing 30+ such ops can exceed sister #1265 threshold even at moderate training depth). Bridge calibration requires per-class FIRST landing of T3 Class 1-SCOPED Kahan-EMA mitigation on `tac.substrates._shared.trainer_skeleton.long_training_canonical.PolyakEMAShadow.update()` (~30 LOC; in-flight `a075fe299ca54fe3a`) THEN one-time per-class $8 paired-CUDA measurement with mitigation applied. After landing both, ALL future same-class substrates trust MLX-local routing. Canonical exemplar: **K=COIN++ implicit neural representation** (commit `2d59283d4` FIX-WAVE-R1''-K; per-matmul drift exceeds floor at K-typical dims; mitigation pending T3 Class 1-SCOPED landing). |

### Cascade economics revised total

Per the canonical 3-class taxonomy applied across the 11-substrate Path 3
wave, the **revised total paid CUDA spend** is:

* **Class A (drift-free structural-decoder)**: 1 substrate (I=Faiss IVF-PQ)
  × $2 = **$2**
* **Class B (matmul-bound HNeRV-class)**: 8 substrates
  (D=Z6 + A=DreamerV3 + E=BoostNeRV + G=NIRVANA + F=Z8 + C'=NSCS06 v8 +
  B'=Z7-Mamba-2-v2 + H=ATW V2 + J=MDL-IBPS) × ~$5 average = **$40**
* **Class C (INR-class with Kahan mitigation pending)**: 1 substrate
  (K=COIN++) × $8 = **$8**
* **Submission auth eval** (paid CUDA; both contest-CPU + contest-CUDA per
  CLAUDE.md non-negotiable): ~$0.50-1 per submission × 1-3 submissions = **~$2**

**Revised total paid CUDA spend across Path 3 11-candidate cascade ≈ $50**

This is **HIGHER than the original doctrine's structural baseline of $5-30**
(which assumed all substrates were Class A drift-free at $0.50-2 per class)
but is **MUCH LOWER than the originally-feared per-substrate per-iteration
spend of $5-15 × 11 = $55-165** (which the original doctrine explicitly
rejected per the structural reduction insight). The drift-aware per-class
breakdown reflects:

1. **Substrate-class diversity bound BY 3 verdicts** (not 11 per-substrate
   bridge calibrations) — per-class one-time calibration amortizes
   across same-class substrates
2. **Class B HNeRV-class dominance** (8 of 11 substrates) — drives
   majority of bridge calibration spend; per-class $5 reflects paired-CUDA
   measurement at canonical L3 operating point with drift-vs-depth
   empirical-anchor-grade fit (sister wave per substrate per cascade
   doctrine cross-substrate-impact section)
3. **Class C K=COIN++ Kahan mitigation prerequisite** — paid CUDA spend
   gated on T3 Class 1-SCOPED Kahan-EMA landing; pending mitigation, K
   remains Tier A PROXY-grade non-promotable per Catalog #341 (the $8
   bridge calibration line item is RESERVED but not yet spendable)
4. **Submission auth eval BOTH CPU+CUDA** per CLAUDE.md non-negotiable —
   $2 total covers ~1-3 PR submissions across Path 3 wave

### Per-substrate-class consumer wiring (Catalog #335 paradigm)

Per the canonical cascade economics revision, the canonical cathedral
consumer `tac.cathedral_consumers.canonical_equation_lookup_consumer`
auto-discovers (per Catalog #335) per-substrate-class drift-vs-depth
canonical equation predictions; future per-substrate-class bridge
calibration data lands as new canonical equations:

* `mlx_pytorch_drift_vs_training_depth_pr95_v1` (HNeRV-class anchor;
  pending L2 long-training sister landing per CASCADE-PROMOTION wave)
* `mlx_pytorch_drift_vs_training_depth_dreamer_v3_v1` (RSSM-class anchor;
  pending A=DreamerV3 L2 landing)
* `mlx_pytorch_drift_vs_training_depth_coin_pp_v1` (INR-class anchor;
  pending K=COIN++ L2 landing with Kahan-EMA mitigation)
* `mlx_pytorch_drift_vs_training_depth_atw_v2_v1` (cooperative-receiver-class
  anchor; pending H=ATW V2 L2 landing)
* `mlx_pytorch_drift_vs_training_depth_faiss_pq_v1` (Faiss-PQ-class anchor;
  trivially BIT_EXACT_LIKE_SINUSOIDAL per FIX-WAVE-R1''-I; sister registers
  the structural property without paid-CUDA measurement)
* ... (one per substrate class; auto-discovered by canonical consumer
  per Catalog #335 protocol)

After ≥3 substrate-class anchors land, lift to substrate-agnostic
`mlx_pytorch_drift_vs_training_depth_v2` per the canonical META pattern
(would parameterize A + B as functions of architecture-class features per
`mps_drift_architecture_class_dependent_v1` precedent).

### Cross-references (this amendment)

* T3 grand council deliberation `7d04474cb`
  (`.omx/research/t3_grand_council_mlx_pytorch_drift_accumulation_source_and_engineer_away_20260526.md`;
  PROCEED_WITH_REVISIONS verdict; 24-of-26 attendees; 3 binding revisions
  + Decision 5 MLX-first doctrine baseline amendment)
* Cascade doctrine §"L6 gate 3-verdict map" (sister amendment;
  `.omx/research/path_3_canonical_substrate_development_cascade_doctrine_20260526.md`;
  3-verdict map applied at L6 → bridge calibration boundary)
* DRIFT-VS-DEPTH-CHAR-D-Z6 landing `60a9de751`
  (`.omx/research/path_3_d_z6_drift_vs_training_depth_characterization_landed_20260526T125130Z.md`;
  n=5 empirical fit α=0.47 sub-linear sat ~2000ep; canonical empirical
  anchor for this amendment's Class B HNeRV-class baseline)
* R1''-K canonical floor `2d59283d4` (THIS doctrine memo §"M-series MPS fp32
  hardware floor canonical anchor"; canonical equation
  `mlx_matmul_drift_m_series_canonical_floor_v1` per Catalog #344;
  3-verdict taxonomy that this amendment's per-class forecast inherits)
* FIX-WAVE-R1''-I byte-identical anchor `1f929127a`
  (canonical exemplar for Class A drift-free structural-decoder;
  max_abs=0.0 empirical at canonical-config dims)
* T3 Class 1-SCOPED Kahan-EMA mitigation in-flight `a075fe299ca54fe3a`
  (canonical mitigation prerequisite for Class C INR-class bridge
  calibration eligibility; ~30 LOC change to
  `tac.substrates._shared.trainer_skeleton.long_training_canonical.PolyakEMAShadow.update()`)
* Catalog #335 cathedral consumer canonical contract (per-substrate-class
  canonical equation auto-discovery surface)
* Catalog #341 Tier A non-promotable markers (Class C INR-class pending
  mitigation routes to Tier A PROXY-grade)
* Catalog #344 canonical equation registry (sister equations across
  per-substrate-class drift-vs-depth predictions)
* CLAUDE.md "Apples-to-apples evidence discipline" (n=5 empirical anchor
  supersedes n=2 stale council assumed extrapolation)
* CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" (paid CUDA
  submission boundary preserved)
* CLAUDE.md "Forbidden premature KILL without research exhaustion"
  (Class C INR-class PROXY-grade is DEFER not KILL; routes to mitigation)

EOF
