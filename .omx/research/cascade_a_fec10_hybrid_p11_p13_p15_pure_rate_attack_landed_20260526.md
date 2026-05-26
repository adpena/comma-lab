# Cascade A FEC10 Hybrid P11+P13+P15 Pure-Rate-Attack — Landing Memo

**UTC**: 2026-05-26T21:20:00Z
**Subagent (recovery)**: `cascade-a-fec10-hybrid-p11-p13-p15-RECOVERY-1-commit-only-signal-preservation-20260526`
**Predecessor (killed by credit-cap)**: `cascade-a-fec10-hybrid-p11-p13-p15-stack-pure-rate-attack-pr111-candidate-mlx-first-numpy-portable-individually-fractal-20260526`
**Lane**: `lane_cascade_a_fec10_hybrid_p11_p13_p15_stack_pure_rate_attack_pr111_candidate_mlx_first_numpy_portable_individually_fractal_20260526`
**Operator approval**: 2026-05-26 verbatim *"all are approved + follow up are approved + pursue other attacks as well + remember all MLX first + individually fractally optimized"* + RECOVERY directive *"respawn and recover and continue with all, ensure no signal loss"*
**Mission contribution per Catalog #300**: `frontier_breaking_enabler` (PARADIGM-VALIDATED rate attack on selector-stream entropy; sister to NSCS06 v8 Modal CUDA paired in flight as PR111 candidacy chain).

## 1. Empirical verdict (per `cascade_a_fec10_hybrid_empirical.json`)

| Codec | Wire bytes | Δ vs FEC6 | Δ vs FEC8 2nd-order |
|---|---|---|---|
| FEC6 baseline | 249 | 0 | +10 |
| FEC8 1st-order static | 245 | -4 | +6 |
| FEC8 2nd-order static | 239 | -10 | 0 |
| **FEC10 hybrid adaptive blend α=2** | **236** | **-13** | **-3** |

- Roundtrip verified: True
- Encode time: 0.48 ms ; Decode time: 0.73 ms ; 600 pairs ; K=16
- Evidence grade: `[macOS-CPU advisory]` per CLAUDE.md "MLX-first numpy-portable individually-fractal" standing directive; PARADIGM-VALIDATED at the rate-only axis with paired CUDA Modal deferred to operator decision (PR111 candidacy gate).

## 2. Catalog #307 paradigm-vs-implementation classification

**Verdict: PARADIGM-VALIDATED** at the P11 (BEFORE entropy coder) sub-stage via adaptive 1st/2nd-order Markov soft-blend with α=2. Sister sub-stages classified independently per Catalog #307:

- **P13 (per-block model selection flags)** = IMPLEMENTATION-LEVEL FALSIFIED at 600-symbol scale (net +0.5B at block_size=50; 12 flag bits cost > ~10 bits min-selection savings). PARADIGM (model selection) PRESERVED; DEFERRED-PENDING-RESEARCH for longer streams where flag overhead amortizes per CLAUDE.md "Forbidden premature KILL without research exhaustion".
- **P15 (brotli AFTER arithmetic coder)** = IMPLEMENTATION-LEVEL FALSIFIED at 239B-scale (+4B WORSE at q∈{6,9,11}; arith output near Shannon floor). PARADIGM (cross-stream brotli sharing) PRESERVED; DEFERRED-PENDING-RESEARCH for longer streams or compressible sister streams.

Per Catalog #308 alternative-probe-methodology enumeration: sister-disjoint reducer (adaptive-blend) STRICT-DOMINATES P13 + P15 at this scale; both falsified sub-stages enumerate reactivation criteria above.

## 3. Canonical equation registration (Catalog #344)

Registered: `cascade_a_fec10_hybrid_adaptive_blend_savings_v1` → registry grew **52 → 53** equations. Registration ran via `tools/register_cascade_a_fec10_hybrid_adaptive_blend_canonical_equation_20260526.py` after one fix (one_line_summary trimmed from 305 → ≤200 chars per `CanonicalEquation.__post_init__` invariant). Canonical Provenance per Catalog #323 (`build_provenance_for_predicted`; `inputs_sha256=6bae0201fb08...` FEC6 archive sha; `measurement_axis="[macOS-CPU advisory]"`). Excluded contexts include `residual_correction_hybrid_substrates` per Catalog #359 sister discipline (adaptive-blend Markov is BEFORE-entropy-coder context-model enhancement; NOT residual-correction-hybrid stacking-extension).

## 4. Predecessor signal preservation

This RECOVERY-1 subagent preserves the killed predecessor's already-on-disk work without re-exploring:
- Encoder/decoder module (491 LOC) at `submissions/hnerv_fec6_fixed_huffman_k16/encoder/build_pr101_frame_exploit_selector_packet_fec10_hybrid.py`
- Pre-execution gate report (133 LOC) at `.omx/research/cascade_a_fec10_hybrid_p11_p13_p15_pre_execution_gate_report_20260526.md` carrying full PV (Catalog #229), 9-dim checklist (#294), cargo-cult audit (#303), observability surface (#305), canonical-vs-unique decision per layer (#290), predicted-band Dykstra (#296), mission-alignment frontmatter (#300).
- Empirical artifact JSON at `.omx/research/cascade_a_fec10_hybrid_artifacts_20260526/cascade_a_fec10_hybrid_empirical.json`
- Registration script (189 LOC) at `tools/register_cascade_a_fec10_hybrid_adaptive_blend_canonical_equation_20260526.py`

## 5. Required canonical sections (per CLAUDE.md standing directives)

### 3-strategy attack framework coverage

PURE-RATE attack (P11 entropy reduction); FULL-SCORER + DISTORTION untouched (sister attacks per `feedback_fractal_optimization_full_stack_three_strategies_rate_distortion_full_scorer_attack_standing_directive_20260526.md`).

### Entropy-position declaration

P11 (BEFORE entropy coder via adaptive context-model soft-blend); sister-disjoint to P9 (NSCS06 v8 chroma_lut) + P11 (FEC8 sister 1st-order/2nd-order) + P13/P15 (deferred per § 2).

### MLX-first numpy-portable bridge

Encoder + decoder pure-Python int arithmetic; no MLX dep; ≤200 LOC inflate surface per HNeRV parity L4; numpy-portable inflate primitives only.

### Individually-fractal per UNIQUE-AND-COMPLETE-PER-METHOD

NEW codec module sister to FEC8 family; FORK_BECAUSE_SUPPRESSES on context-model factory (FEC8 has 1st-only / 2nd-only; FEC10 has BLEND); ADOPT_CANONICAL on CACM-87 arithmetic coder + EMPIRICAL_* tables.

### Canonical-vs-unique decision per layer (Catalog #290)

See pre-execution gate report § 6 (6 layers: 3 ADOPT_CANONICAL + 3 FORK).

### 9-dimension success checklist evidence (Catalog #294)

See pre-execution gate report § 7 (UNIQUENESS / BEAUTY+ELEGANCE / DISTINCTNESS / RIGOR / OPTIMIZATION-PER-TECHNIQUE / STACK-OF-STACKS-COMPOSABILITY / DETERMINISTIC-REPRODUCIBILITY / EXTREME-OPTIMIZATION-PERFORMANCE / OPTIMAL-MINIMAL-CONTEST-SCORE).

### Cargo-cult audit per assumption (Catalog #303)

See pre-execution gate report § 8 (6 assumptions: 3 HARD-EARNED + 3 CARGO-CULTED with empirical unwind verdicts).

### Observability surface (Catalog #305)

See pre-execution gate report § 9 (6 facets all surfaced).

### Drift / Predicted-band (Catalog #296 Dykstra-feasibility)

Predicted band [234B, 238B]; empirical 236B = mid-band. Δ = 1 byte better than mean prediction; HARD-EARNED-EMPIRICALLY-VERIFIED for `selector_index_stream_pr101_frame_exploit_600_pairs_k16_palette` in-domain context.

### Horizon-class

PLATEAU-ADJACENT at the rate-axis sub-step (-8.66e-6 ΔS contest units per -13B × 25 / 37,545,489); compounds with FRONTIER-PURSUIT sister stacking targets (NSCS06 v8 + grayscale_lut + VQ-VAE indices_blob) per T3 #1335.

### Catalog #344 canonical equation reference

`cascade_a_fec10_hybrid_adaptive_blend_savings_v1` (this landing); sister `markov_context_selector_stream_compression_savings_v1` (FEC8 parent).

### 6-hook wire-in declaration (Catalog #125)

| Hook | Status | Surface |
|---|---|---|
| #1 sensitivity-map | N/A — rate-axis-only sub-step; no per-pixel sensitivity contribution |
| #2 Pareto constraint | ACTIVE — additive ΔS_rate=-8.66e-6 contest units; composable with sister attacks |
| #3 bit-allocator | N/A — single-stream selector-only |
| #4 cathedral autopilot dispatch | ACTIVE — canonical equation #53 auto-discoverable via `tac.cathedral_consumers.canonical_equation_lookup_consumer` per Catalog #335 |
| #5 continual-learning posterior | ACTIVE — canonical equation registry append per Catalog #344 |
| #6 probe-disambiguator | ACTIVE — adaptive-blend rule + α empirical optimum disambiguate FEC8 1st-only vs 2nd-only vs blend |

## 6. Operator-routable next step

**PR111 candidacy paired-CUDA Modal validation OPERATOR-DECISION-REQUIRED** per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" + "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium" + `feedback_t3_council_pr110_stacking_pivot_ordering_landed_20260526.md` ordering ledger. The PARADIGM-VALIDATED [macOS-CPU advisory] anchor is INSUFFICIENT for PR submission; route through Modal A100 paired auth_eval to promote to `[contest-CUDA]` and PR111 candidacy gate.

## 7. Sister coordination

- Disjoint from in-flight sisters at predecessor launch: UNIWARD 6th-order BoostNeRV / Meta-Lagrangian Phase 3 / Cascade B CATALYST cascade composition / NSCS06 v8 Modal CUDA.
- Recovery scope: COMMIT-ONLY signal preservation per credit-cap mitigation directive; NO new exploration; NO subagent spawning.
