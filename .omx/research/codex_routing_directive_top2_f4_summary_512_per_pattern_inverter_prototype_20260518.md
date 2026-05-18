# Codex routing directive TOP-2: F4 summary(512) per-pattern inverter prototype
# Date: 2026-05-18
# Authority: SYSTEMATIC RECLAIMABILITY RE-EXAMINATION verdict PROCEED_WITH_REVISIONS (commit `4480d9b14`) — TOP-2 ranking; F4 reclaimable per binary-distillation framework
# Budget envelope: $2-5 paid GPU spend; OPERATOR AUTHORIZATION REQUIRED before execution
# Predicted [prediction] ΔS band: [-0.015, -0.003]; predicted compressed binary 5-15 KB
# Lane class: ASYMPTOTIC-PURSUIT reclamation; second vector in TOP-3 EV ordering
# Gated on: TOP-1 (A1-SPECIALIZED) empirical anchor lands successful — same framework validation

## CANONICAL POINTERS

1. `/Users/adpena/Projects/pact/CLAUDE.md` (FULL; especially "Forbidden premature KILL" + "HNeRV parity discipline" L4 + "Contest vs production target modes" `contest_one_video_replay` + Catalog #270/#325/#313/#167/#199 cluster)
2. `/Users/adpena/Projects/pact/AGENTS.md`
3. `.omx/research/systematic_reclaimability_re_examination_via_binary_distillation_framework_20260518.md` (commit `4480d9b14`; TOP-2 source-of-truth)
4. `.omx/research/a1_binary_distillation_design_memo_zig_sparseness_ablation_plus_canonical_techniques_20260518.md` (commit `0701c323b`; canonical framework — SAME framework as TOP-1, applied to F4 feature space)
5. `.omx/research/rate_attack_legal_receiver_path_audit_codex_f1_finding_relay_20260518.md` (commit `35b06f9ec`; F4 was originally STRICT_SCORER_RULE_VIOLATION; SYSTEMATIC RECLAIMABILITY reclassified RECLAIMABLE per binary-distillation framework)
6. `.omx/research/codex_routing_directive_top1_a1_specialized_per_pattern_vq_vae_inverter_prototype_20260518.md` (sister TOP-1; cite for shared 6-phase execution + same 5 binding revisions + Provenance API)

## STRATEGIC FRAMING

**F4 summary(512)** is PoseNet's `summary` projection layer (upstream/modules.py — `vision(2048) → summary(512)` per CLAUDE.md "Exact scorer architectures" section). The original 43-vector audit classified F4 as `STRICT_SCORER_RULE_VIOLATION` because encoding into summary(512) feature space naively requires the full summary projection at inflate. SYSTEMATIC RECLAIMABILITY reclassified F4 as `RECLAIMABLE_VIA_PER_PATTERN_DISTILL`: a per-pattern PQ (Product Quantization) + K=64 codebook + FP4 + Brotli binary may produce a 5-15 KB specialized inverter. This is `contest_one_video_replay` sanctioned feasibility; scored compliance still requires charged-byte packet proof, no generic scorer behavior, and exact CUDA auth eval.

PQ-8x8 split: summary(512) → 8 sub-vectors of 64 dims each; K=64 codebook per sub-vector → 8 × log2(64) = 48 bits per pattern (vs 512 × 16 = 8192 bits for naive fp16). 170× compression at the feature level before FP4 + Brotli.

Predicted realistic compressed binary: 5-15 KB. Predicted contest-CUDA / contest-CPU ΔS: [-0.015, -0.003] (cite `4480d9b14` per-vector matrix). Marginally higher predicted EV than TOP-1 (different feature space; potentially more headroom) but $2-5 budget vs $1-3 for TOP-1.

## INHERITED 5 BINDING REVISIONS (from `0701c323b` T2 council; same framework)

1. **PROCEED on per-pattern distilled path; NOT generic Hinton student**
2. **TOP architecture composition = PQ-8x8 + K=64 + FP4 + Brotli** (adapted from V2 VQ-VAE; PQ-8x8 better suited to 512-dim feature space than monolithic K=256)
3. **4 reactivation conditions pinned per Catalog #325** (cite `0701c323b` for canonical list)
4. **Reject pure-Zig/Rust binary path** (same rationale as TOP-1; Python in pinned env)
5. **1 mandatory empirical anchor before Phase 2** ($2-5 prototype — THIS DIRECTIVE EXECUTES THAT ANCHOR)

## 6-PHASE EXECUTION

Same 6-phase structure as TOP-1 directive (see `codex_routing_directive_top1_*.md` for canonical Provenance API + probe-outcomes ledger + smoke-before-full pattern). Adapted differences:

### Phase 1: Predecessor-probe consultation (per Catalog #313)
Substrate ID: `f4_summary_512_per_pattern_inverter`. No predecessor expected at landing time.

### Phase 2: Build prototype helper
`tools/build_f4_summary_512_per_pattern_inverter_prototype.py` (~300-500 LOC):
- Reads canonical PoseNet `summary(512)` activations on Lane A archive's per-pattern subset
- Per-pattern PQ codebook training: K=64 per 8-subvector split (Faiss-IVF-PQ-style; see CLAUDE.md DEEP-RESEARCH-WAVE TOP-5 #3 ATW V2-1 + Faiss-IVF-PQ for canonical implementation reference)
- FP4 quantization on PQ codebook entries (`tac.quantization.FakeQuantFP4`)
- 50% magnitude-pruning sparseness mask on residual
- Brotli compression on the resulting blob

### Phase 3: Empirical bit-spend smoke (T4; $2-5 envelope)
Per Catalog #167 smoke-before-full:
- Smoke target: 50 patterns × ~30 seconds each on T4 (Modal T4 ≈ $0.30/100s + overhead; envelope ≈ $2-5)
- Fail-fast threshold: compressed bytes > 20 KB OR proxy distortion >2× predicted

Recipe template: `.omx/operator_authorize_recipes/substrate_f4_summary_512_per_pattern_inverter_modal_t4_smoke.yaml` per Catalog #270 Tier 1+2+3 + #244 NVML + #240 recipe-vs-trainer-state.

### Phase 4: Measurement on 3 archives
Same 3-archive measurement plan as TOP-1: Lane A + Lane 12 sister + apogee_v2 baseline.

### Phase 5: Provenance wrapping per Catalog #323
Same canonical API as TOP-1 (see TOP-1 directive for verified `tac.provenance` builders + `provenance_to_dict` + `ProvenanceEvidenceGrade` enum).

### Phase 6: Probe outcome registration per Catalog #313
Probe ID: `f4_summary_512_per_pattern_inverter_prototype_20260518`.

## DISCIPLINE

Same canonical-discipline cluster as TOP-1 directive (Catalog #229 + #287 + #270 + #313 + #167 + #199 + #244 + #226 + #205 + #316 + #323 + #292 + HARVEST-OR-LOSE).

## OPERATOR-DECISION MATRIX

- **PROCEED-AFTER-TOP-1**: ratify only if TOP-1 (A1-SPECIALIZED) empirical anchor lands successful; same framework validation transfers
- **PROCEED-PARALLEL-WITH-TOP-1**: operator authorizes both TOP-1 + TOP-2 in parallel ($3-8 total) for orthogonal-feature-space validation
- **DEFER**: Codex creates the prototype helper (Phase 2 build only; $0) without dispatch
- **DEFER-PENDING-TOP-1-RESULT**: cheapest disambiguation — TOP-1's empirical anchor reveals whether the framework actually works on the substrate manifold; if TOP-1 fails, TOP-2 may need framework revision before dispatch

## EXIT CRITERIA

- [ ] Phase 1-6 per TOP-1 template
- [ ] Memory entry `feedback_f4_summary_512_per_pattern_inverter_prototype_landed_<YYYYMMDD>.md`

## SISTER COORDINATION

- TOP-1 (A1-SPECIALIZED) is the cheapest framework-validation probe; TOP-2 should ideally fire AFTER TOP-1's positive empirical anchor
- TOP-3 (F5 ResBlock(512)) uses same framework on adjacent feature space; if TOP-2 succeeds, TOP-3 inherits high prior probability
- DYNAMIC PER-CANDIDATE COMPOSITION FRAMEWORK subagent in-flight on disjoint scope

## OPERATOR-FACING NOTE

This routes TOP-2 of the SYSTEMATIC RECLAIMABILITY 5-vector reclamation set. F4 was previously buried as STRICT_SCORER_RULE_VIOLATION but is RECLAIMABLE as a hypothesis via the binary-distillation framework. $2-5 envelope; recommended to fire AFTER TOP-1's empirical anchor lands so framework risk is amortized by actual evidence rather than inherited prose.

— Main-Claude 2026-05-18 (per SYSTEMATIC RECLAIMABILITY `4480d9b14` TOP-2 ranking + shared binary-distillation framework `0701c323b`)
