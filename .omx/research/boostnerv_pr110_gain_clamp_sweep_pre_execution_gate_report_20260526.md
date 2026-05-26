<!-- SPDX-License-Identifier: MIT -->
<!-- Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE — BoostNeRV-PR110 gain_clamp × epochs sweep pre-execution gate report. DO NOT mutate after landing. -->
<!-- Catalog #229 PV: this memo verifies premises empirically — BoostNeRV L1 EMPIRICAL artifact + PR110 base raw + L1 probe script all verified present + sweep parameterization scope defined BEFORE harness emission. -->
<!-- # FORMALIZATION_PENDING:boostnerv_gain_clamp_sweep_pre_execution_carries_sweep_scope_definition_no_canonical_equation_registration_at_pre_execution_gate_stage_results_landing_memo_will_address -->

# BoostNeRV-PR110 gain_clamp × epochs sweep — PRE-EXECUTION GATE REPORT 2026-05-26

**Subagent**: `boostnerv-pr110-gain-clamp-sweep-20260526`
**Lane**: `lane_path_3_e_boost_nerv_against_pr110_20260526` (extends sister L1 EMPIRICAL lane via sweep-disambiguator follow-up; Catalog #220 operational-mechanism preserved)
**Predecessor**: `boostnerv-pr110-l1-empirical-mlx-respawn-20260526` (commit `b2fd3e587`)
**Operator authority**: 2026-05-26 verbatim approval per Carmack analytical dissent in L1 landing memo + ROADMAP TOP-EV cascade

## Pre-execution checklist

| Item | Status | Evidence |
|---|---|---|
| Mandatory pre-flight: read CLAUDE.md + AGENTS.md | DONE | NON-NEGOTIABLES honored: "MPS auth eval is NOISE" + "Modal `.spawn()` HARVEST OR LOSE" + Catalog #192/#317 MLX advisory tagging + Catalog #287 placeholder rejection + Catalog #340 sister-checkpoint guard + Catalog #343 no hardcoded scores |
| Read both 2026-05-26 NEW standing-directive memos | DONE | (1) `feedback_mlx_cuda_bidirectional_drift_anticipation_standing_directive_20260526.md` 5 drift surfaces; (2) `feedback_pushing_the_frontier_of_research_on_optimization_algorithms_standing_directive_20260526.md` canon-vs-frontier-push framing |
| Read BoostNeRV landing memo Carmack dissent | DONE | `.omx/research/boostnerv_pr110_l1_empirical_landed_20260526.md` lines 11-12: gain_clamp=0.05 too tight; 14B brotli of 1.84MB int8 means residuals nearly all zero post-clamp; HYPOTHESIS clamp-bound NOT converged-quality-bound |
| Read L1 EMPIRICAL probe artifact | DONE | `.omx/state/boostnerv_pr110_residual/l1_empirical_landed_20260526.json` — 1971 params, 30 epochs 0.345s wallclock, BPR1 sidecar 42 bytes, training_loss_reduction_fraction 7.76%, recon_mse_reduction_fraction_proxy 16.2% |
| Read L1 probe script | DONE | `.omx/tmp/boostnerv_pr110_l1_empirical_probe.py` 546 LOC; full parameterization understanding |
| PR110 base raw cache | VERIFIED | `.omx/tmp/boostnerv_pr110_l1_stage0_workdir/output_dir/0.raw` 3492.7 MB (1200 × 874 × 1164 × 3 = 3.66 GB; first ~100 frames suffice for 50-pair probe) |
| Sister-context disjoint scope check | DONE | Cluster slots in flight (#1335 PR110 stacking ordering / cls_stream / NIRVANA / NSCS06 v8 chroma_lut MLX L1 / etc.) all in DIFFERENT substrate IDs OR different files; my scope `src/tac/substrates/boost_nerv_pr110_residual/` + `.omx/tmp/boostnerv*` + `.omx/state/boostnerv_pr110_residual/` + `.omx/research/boostnerv*` is DISJOINT |
| Catalog #340 sister-checkpoint guard for my files | PROCEED | My write targets are NEW research/state files (sweep results JSON + sweep landing memo + sweep harness script) + NEW pre-execution gate report (THIS file); ZERO existing-file mutations planned |
| Catalog #206 subagent checkpoint | EMITTED | Step 1 in_progress checkpoint at 2026-05-26T18:23:59Z |

## Sweep scope definition

**9-cell grid** (3 × 3 cartesian product per Carmack dissent operator-routable #1):

| | epochs=30 | epochs=100 | epochs=300 |
|---|---|---|---|
| **gain_clamp=0.05** | (anchor — already landed in L1 EMPIRICAL) | NEW | NEW |
| **gain_clamp=0.10** | NEW | NEW | NEW |
| **gain_clamp=0.20** | NEW | NEW | NEW |

**Sister anchor**: cell (gain_clamp=0.05, epochs=30) RE-RUNS the L1 EMPIRICAL config from `b2fd3e587` for sweep coherence (sister-cell baseline; expected to match within `5e-5` per AdamW seed determinism on M5 Max MLX); ALL 9 cells re-run with fresh seeded RNG=42 + identical 50-pair × 96×128 fixture for apples-to-apples comparability. Cell (0.05, 30) acts as both anchor + disambiguator probe for MLX→MLX same-config reproducibility (one of Catalog #305 observability facets).

**Per-cell measurement protocol**:
1. **final-epoch loss** (training MSE on composed-vs-GT on internal 96×128 grid; AdamW step-by-step convergence)
2. **recon proxy MSE** (composed-vs-GT per-pair MSE on the 50-pair × 96×128 fixture; advisory per Catalog #192)
3. **post-clamp residual statistics** — histogram bins {0.0, 0.005, 0.01, 0.025, 0.05, 0.10, 0.20} + p99 magnitude — to verify clamp-bound vs unbound regime
4. **BPR1 sidecar bytes** (canonical brotli quality 9 per design memo; mirror the L1 EMPIRICAL canonical helper `build_bpr1_sidecar` with per-cell `gain_clamp` parameterization)
5. **Δrate = 25 × bytes / 37,545,489** (exact contest-axis rate-term cost; deterministic arithmetic)
6. **wallclock seconds** per cell (M5 Max MLX-local; expected 0.3-3s per cell from L1 baseline 0.345s @ 30ep)

**Total compute budget**: $0 (MLX-local ONLY per "Remember all on MLX"). Sequential execution: 9 × ~1-3s = 10-30s training; +data-load amortized (load PR110+GT ONCE shared across all 9 cells). Total expected wallclock ~30-60s. ThreadPoolExecutor parallelization NOT needed given trivial cost.

**Drift surface declaration** per NEW MLX↔CUDA bidirectional drift directive:

| Drift source | Sweep-context mitigation |
|---|---|
| Float32 vs float16 rounding | All cells fp32 throughout (MLX-default + NumPy-default); no float16 demotion |
| NHWC vs NCHW convention | All cells NHWC throughout (no PyTorch sister wire-in this sweep) |
| Conv2d padding semantics | All cells reuse L1 probe's `padding=kernel_size//2` (mirrors PR110 canonical) |
| Tanh + clip ordering | All cells reuse canonical `tanh(conv2(...))` → `clip(±gain_clamp)` → `+base` → `clip(0,1)` order |
| Brotli quality 9 determinism | All cells use identical brotli quality 9 (Python brotli package deterministic) |
| AdamW β₁/β₂ state buffers (NEW from directive sister #3) | All cells reuse MLX `optim.AdamW(learning_rate=1e-3)` defaults (β₁=0.9, β₂=0.999); sister bidirectional verification deferred to future paired-CUDA dispatch |

## Canonical-vs-frontier-push decision (per NEW pushing-the-frontier directive)

The sweep harness itself is **CANON-APPLICATION** at the apparatus level (cartesian-product sweep is canonical experimental-design pattern), BUT the empirical interpretation IS **FRONTIER-PUSH** at the substrate level: the L1 result + sweep heatmap together test whether the canonical boosting-residual paradigm (BoostNeRV-against-PR110) has empirical signal at THIS substrate's specific operating point. The empirical evidence directly informs Catalog #344 candidate equation `residual_hybrid_boosting_savings_v1` proposed-pending-operator-decision in the L1 landing memo — empirical sweep data either supports OR refutes the registration. Per the directive: explicit `## Canonical-vs-frontier-push decision` section will appear in the sweep landing memo with the empirical verdict.

## Pre-execution verdict

**PROCEED** to harness implementation + 9-cell execution.

**Rationale**: all premises empirically verified; sister-context scope-disjoint; drift surface declared; canonical-vs-frontier-push framing applied; canonical helpers reused from L1 probe (no new bug class surface); $0 budget; ~30-60s wallclock; full operator-authority chain (Carmack dissent + ROADMAP TOP-EV) documented; Catalog #229 PV satisfied; Catalog #206 checkpoint emitted.

**Operator-routable**: spawn the harness immediately; no operator gate required.

## Cross-references

- Sister L1 EMPIRICAL landing memo: `.omx/research/boostnerv_pr110_l1_empirical_landed_20260526.md` (commit `b2fd3e587`)
- Sister L1 EMPIRICAL pre-execution gate: `.omx/research/boostnerv_pr110_l1_empirical_pre_execution_gate_report_20260526.md`
- Design memo: `.omx/research/path_3_e_boost_nerv_against_pr110_substrate_design_20260526.md`
- L1 probe script (reused as base): `.omx/tmp/boostnerv_pr110_l1_empirical_probe.py`
- L1 EMPIRICAL artifact: `.omx/state/boostnerv_pr110_residual/l1_empirical_landed_20260526.json`
- Bidirectional MLX↔CUDA drift directive: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_mlx_cuda_bidirectional_drift_anticipation_standing_directive_20260526.md`
- Pushing-the-frontier directive: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_pushing_the_frontier_of_research_on_optimization_algorithms_standing_directive_20260526.md`
- T3 PR110-stacking-ordering memo: `.omx/research/t3_council_pr110_stacking_pivot_ordering_landed_20260526.md` (potential downstream consumer of sweep verdict)
