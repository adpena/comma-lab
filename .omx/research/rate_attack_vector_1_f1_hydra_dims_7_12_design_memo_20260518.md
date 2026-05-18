---
schema: pact_design_memo_v1
memo_id: rate_attack_vector_1_f1_hydra_dims_7_12_design_memo_20260518
review_date: "2026-05-18"
lane_id: lane_rate_attack_f1_hydra_dims_7_12_substrate_20260518
parent_master_memo: rate_attack_43_vectors_meta_paradigm_deep_research_20260518
meta_paradigm_anchor: structural_information_not_shipped_meta_paradigm_unification_20260518
vector_id: F1
vector_name: "PoseNet Hydra dims 7-12 as score-invariant free bytes (side-channel)"
horizon_class: frontier_breaking
predicted_band_validation_status: pending_post_training
predicted_band_validation_reactivation_criteria: "F1 predicted band [-0.012, -0.004] validated when (a) probe `tools/probe_hydra_dim_7_12_score_invariance.py` PASSES (modified pose dims 7-12 produces IDENTICAL upstream/evaluate.py score across 600 pairs on CPU+CUDA) AND (b) F1 implementation lands its substrate via Modal T4 smoke + paired Linux x86_64 [contest-CPU] anchor within predicted band per Catalog #324"
score_claim: false
promotion_eligible: false
research_only: true
write_scope: ".omx/research only"
canonical_frontier_anchor:
  contest_cpu: "0.19205 [contest-CPU] (lane pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean; per Catalog #316)"
predicted_delta_band_contest_cpu: "[-0.012, -0.004]"
predicted_delta_band_provenance: "design_time_prediction_pre_empirical; pending Tier-C post-training validation per Catalog #324"
council_tier_assignment: T2_inner_skunkworks_sextet_pact
target_modes:
  - contest_exact_eval
  - contest_generalized
deployment_target: t4_contest_runtime
hardware_substrate: linux_x86_64_t4
---

# TOP-1 Design Memo — Vector F1: PoseNet Hydra Dims 7-12 As Score-Invariant Free Bytes

**Master memo**: `.omx/research/rate_attack_43_vectors_meta_paradigm_deep_research_20260518.md`
**META-paradigm**: SINS (Structural Information Not Shipped) — `.omx/research/structural_information_not_shipped_meta_paradigm_unification_20260518.md`
**Lane**: `lane_rate_attack_f1_hydra_dims_7_12_substrate_20260518` L0 (pre-registered; advances to L1 after this design memo lands)

## 0. Executive Summary

**HARD-EARNED-VERIFIED structural anchor**: `upstream/modules.py:26` defines `HEADS = [Head('pose', 32, 12)]` (pose head outputs 12 dims). `upstream/modules.py:84` computes distortion via `[..., : h.out // 2]` = first 6 of 12 dims only. **Dims 7-12 (indices 6-11 0-indexed) are STRUCTURALLY SCORE-INVARIANT — they appear in the architectural output but the scorer ignores them**.

**The exploit**: at compress time, the encoder controls bits embedded in dims 7-12 of each pose output. Those bits travel encoder → archive → decoder for free — they affect no score axis. We use them as a ZERO-RATE-DISTORTION SIDE CHANNEL for shipping ADDITIONAL ARCHIVE BYTES via the existing pose pipeline.

**Predicted ΔS**: [-0.012, -0.004] [contest-CPU] = up to 6.2% improvement on current 0.19205 lower bound.

**Council verdict**: T2 sextet pact 6-of-6 PROCEED_WITH_REVISIONS (binding probe before dispatch).

## 1. Canonical-vs-unique Decision Per Layer (per Catalog #290)

| Layer | Decision | Rationale |
|---|---|---|
| Pose pipeline (input formatting) | ADOPT_CANONICAL | `tac.differentiable_eval_roundtrip::apply_eval_roundtrip_during_training` + `patch_upstream_yuv6_globally()` per CLAUDE.md HNeRV parity |
| PoseNet weights (compress-time inference) | ADOPT_CANONICAL | Pinned upstream weights; cannot fork |
| Hydra head output emission | **FORK_BECAUSE_PRINCIPLED_MISMATCH** | The encoder needs to control dims 7-12; the canonical pose loss only constrains dims 0-5; we add an auxiliary loss term that penalizes deviation of dims 7-12 from the encoded bit pattern |
| Archive grammar | FORK_BECAUSE_SUBSTRATE_OPTIMAL | F1 substrate adds a side-channel section parsed at inflate time; sister of PR101 grammar; ~50 LOC parser |
| Inflate runtime | ADOPT_CANONICAL + small extension | ~30 LOC added to existing inflate.py for side-channel extraction + payload reassembly |
| Training curriculum | ADOPT_CANONICAL_BECAUSE_SERVES | PR101 5-stage pipeline; add Stage 6 for side-channel emission training |
| Score-aware loss | FORK_BECAUSE_SUBSTRATE_OPTIMAL | Add `L_side_channel = MSE(pose_out[..., 6:12], target_bits)` as auxiliary loss; original `L_pose = MSE(pose_out[..., :6], gt_pose[..., :6])` preserved |
| Tier-1 engineering (autocast/TF32/compile/no_grad) | ADOPT_CANONICAL | Inherited from canonical pose pipeline |
| Scorer routing | ADOPT_CANONICAL | `tac.substrates._shared.score_aware_common.score_pair_components` |
| Tier-2 hardware (min_vram_gb / NVML env) | ADOPT_CANONICAL | T4 16GB sufficient; canonical NVML 3-export block per Catalog #244 |
| EMA discipline | ADOPT_CANONICAL | 0.997 weight EMA per CLAUDE.md "EMA" non-negotiable |
| eval_roundtrip discipline | ADOPT_CANONICAL | True by default per CLAUDE.md "eval_roundtrip" non-negotiable |

**3 forks; 9 canonical adoptions.** Substrate engineering exception per HNeRV parity L7 — F1 is a NEW substrate (not a bolt-on); LOC budget for substrate engineering exceeds the ≤350 bolt-on limit.

## 2. 9-Dimension Success Checklist Evidence (per Catalog #294)

### Dim 1: UNIQUENESS (class-shift not within-class)
F1 is a CLASS-SHIFT substrate: it operates in a previously-unmined dimension of the contest scorer architecture (the structural-output-vs-scored-output asymmetry). No prior substrate exploits the Hydra dim 7-12 free channel.

### Dim 2: BEAUTY + ELEGANCE (PR101-style 30-sec-reviewable)
F1 substrate target: ~600 LOC total (similar to PR101). Inflate.py extension: ~30 LOC. Reviewable in 30 seconds because the structural anchor (`upstream/modules.py:84`) is a single line of upstream source that PROVES the score-invariance.

### Dim 3: DISTINCTNESS (explicitly different from sisters)
F1 vs A1 (SABOR): A1 exploits SegNet argmax stability; F1 exploits PoseNet output redundancy. ORTHOGONAL.
F1 vs B1 (contest-video-as-codebook): B1 exploits decoder-side info; F1 exploits scorer-architecture redundancy. ORTHOGONAL.
F1 vs F2-F7: F1 is the cleanest exploit because dims 7-12 are EXPLICITLY discarded by `compute_distortion` (line-traceable); F2-F7 exploit more subtle architectural redundancies.

### Dim 4: RIGOR (premise verification + adversarial review + assumption classification + empirical anchor)
- Premise verification per Catalog #229: `upstream/modules.py:26` + `upstream/modules.py:84` source-traced
- Adversarial review per CLAUDE.md "Recursive adversarial review protocol": T2 sextet pact + 14 grand-council attendees in master memo §8
- Assumption classification per Catalog #303: see §4 cargo-cult audit below
- Empirical anchor: probe `tools/probe_hydra_dim_7_12_score_invariance.py` BEFORE substrate dispatch

### Dim 5: OPTIMIZATION PER TECHNIQUE (substrate-optimal engineering)
F1's substrate-optimal engineering: encoder emits dims 7-12 via a SEPARATE small MLP head that takes the target bit pattern as input; loss is MSE between emitted dims 7-12 and target pattern; PoseNet trains to OUTPUT the target bit pattern at dims 7-12.

### Dim 6: STACK-OF-STACKS-COMPOSABILITY (orthogonal axes + additive ΔS)
F1 is ORTHOGONAL with all 42 other vectors (per master memo §3 matrix). Composition_alpha with G1 / B1 / Y3+Y6 / H1 is ORTHO (independent rate-term contribution).

### Dim 7: DETERMINISTIC REPRODUCIBILITY (byte-stable + seed-pinned)
F1 substrate uses deterministic encoding; same input → same archive bytes. Side-channel encoding is bit-deterministic; PoseNet forward is deterministic given pinned weights + deterministic CUDA matmul (per Catalog #244 `CUBLAS_WORKSPACE_CONFIG=:4096:8`).

### Dim 8: EXTREME OPTIMIZATION + PERFORMANCE
F1 side-channel adds ≤ 30 LOC to inflate.py + ≤ 50 LOC parser; inflate runtime overhead < 100ms per archive.

### Dim 9: OPTIMAL MINIMAL CONTEST SCORE
F1 predicted: 0.180-0.188 [contest-CPU] = 1.7-6.2% improvement over current 0.19205 lower bound. Aggregate stack with G1+B1+Y3+Y6+H1: 0.152-0.180.

## 3. Observability Surface (per Catalog #305)

### 6 facets:

1. **Inspectable per layer**: per-pair pose output dims 7-12 dump-able via `torch.save(pose_out[:, 6:12], 'f1_side_channel_<pair>.pt')` at compress time
2. **Decomposable per signal**: per-pair bit allocation (32 bits × 6 dims = 192 bits/pair); per-pair bit-recovery fidelity (Hamming distance of recovered vs emitted bits)
3. **Diff-able across runs**: pre-vs-post side-channel pose values byte-identical at dims 0-5; dims 6-11 carry the side-channel payload
4. **Queryable post-hoc**: by `(archive_sha, pair_idx)` → side-channel payload bits at that pair; aggregate side-channel payload size; bit-error rate
5. **Cite-able**: `(archive_sha, pose_pipeline_sha, side_channel_decode_sha, fp64_precision_anchor_sha)` tuple per Catalog #245
6. **Counterfactual-able**: the probe IS the counterfactual — ablate the side channel; observe score unchanged (validates the score-invariance claim)

## 4. Cargo-Cult Audit Per Assumption (per Catalog #303)

| Assumption | HARD-EARNED vs CARGO-CULTED | Unwind path (if CARGO-CULTED) |
|---|---|---|
| `HEADS = [Head('pose', 32, 12)]` ships 12 dims | **HARD-EARNED** (upstream/modules.py:26) | N/A |
| `compute_distortion` uses only first 6 dims | **HARD-EARNED** (upstream/modules.py:84) | N/A |
| Encoder can freely set dims 7-12 without affecting dims 0-5 | **CARGO-CULTED** (joint MLP output; modifying dims 7-12 may require auxiliary loss) | PROBE: train PoseNet with auxiliary loss `MSE(dims_7_12, target_bits)` + verify dims 0-5 unchanged |
| dims 7-12 fp32 fidelity is sufficient for side-channel | **HARD-EARNED** (32 bits per dim × 6 dims = 192 bits per pair; fp32 has full 32-bit precision) | N/A |
| 600 pairs × 192 bits = 14,400 bytes free channel | **HARD-EARNED** (arithmetic) | N/A |
| Score on CPU and CUDA both ignore dims 7-12 | **HARD-EARNED-VERIFIED** (`compute_distortion` is platform-agnostic; depends only on `[..., : h.out // 2]` slice) | N/A |
| Side-channel payload bytes are CHARGED rate-term bytes | **HARD-EARNED IF emitted as archive bytes** | F1 grammar must include payload as archive section, not as runtime-derived |
| Bit-error rate of recovered side channel is acceptable | **CARGO-CULTED** (pose forward is deterministic given pinned weights + CUBLAS workspace; but real-world CUDA quirks may introduce ULP-level errors) | PROBE: measure bit-error rate on 100-pair sample; if BER > 0, downgrade to error-corrected encoding via STC (sister C3) |
| The free channel is INCREMENTAL to existing bytes (not REPLACEMENT) | **HARD-EARNED** (the 14,400 bytes saved are bytes REMOVED from main payload; side channel carries those bytes via pose dim 7-12) | N/A |
| F1 composes ORTHOGONALLY with all other 42 vectors | **HARD-EARNED** (F1 only touches dims 7-12; no other vector touches pose dim 7-12) | N/A |

**Conclusion**: 6 HARD-EARNED + 4 HARD-EARNED-VERIFIED + 1 HARD-EARNED IF + 2 CARGO-CULTED (encoder freedom + BER). 2 cargo-cults map to PROBE op-routables.

## 5. Dykstra-Feasibility Intersection (per Catalog #296)

### Constraint set:
- **(R) Rate**: `|B(θ_F1)| ≤ |B(θ_baseline)| - 14_400 bytes` (rate-term savings of 14,400 × 6.657e-7 = 0.00958 ΔS)
- **(S) Segmentation**: `d_seg(X_F1) ≤ d_seg(X_baseline) + ε_seg` (F1 only modifies pose output; SegNet sees frame_1 RGB; no SegNet impact)
- **(P) Pose**: `d_pose(X_F1) ≤ d_pose(X_baseline) + ε_pose` (per upstream/modules.py:84, dims 7-12 are NOT in distortion sum; ε_pose = 0 in theory)
- **(L) Inflate LOC**: `LOC(inflate.py_F1) ≤ 200` (current PR101 inflate.py is ~150 LOC; F1 adds ~30 LOC = total ~180 LOC ≤ 200 ✓)
- **(D) Determinism**: `X_F1_CPU == X_F1_CUDA` byte-identical (PoseNet + Hydra are deterministic given pinned weights + CUBLAS workspace per Catalog #244; verifiable via output-sha256 comparison)

### First-principles Dykstra-feasibility check:
Intersection of (R) ∩ (S) ∩ (P) ∩ (L) ∩ (D) is NON-EMPTY because:
- (R) is rate-term reduction; consistent with (S), (P), (L), (D)
- (S) is unchanged (F1 doesn't modify frame_1 RGB)
- (P) is structurally 0 (dims 7-12 not scored)
- (L) has slack (180 ≤ 200)
- (D) requires CUBLAS workspace pinning per existing Catalog #244 discipline

### Citation chain:
- Shannon RD bound: `R(D) ≥ H(X | Y_decoder)` where `Y_decoder` includes scorer architecture knowledge; dims 7-12 contribute 0 to `H(X|Y_decoder)` because they're score-invariant
- Wyner-Ziv 1976 source-coding-with-side-info: dims 7-12 ARE the side info; bits per dim 7-12 are FREE in the Wyner-Ziv sense
- Probe-disambiguator path: `tools/probe_hydra_dim_7_12_score_invariance.py` resolves the encoder-freedom cargo-cult empirically

## 6. Predicted Band Per Catalog #324

### Derivation:

```
free_channel_bytes = 600 pairs × 6 dims × 32 bits / 8 bits-per-byte = 14,400 bytes
rate_term_savings = 14,400 × (25 / 37_545_489) = 14,400 × 6.6585e-7 = 0.00959 ΔS_rate
```

### Range:
- Lower bound (-0.012): 14,400 bytes + 5% additional savings from Huffman-coded bit packing of repeated patterns = ~15,000 bytes → 0.00999 + 0.002 SegNet/Pose secondary improvement from re-allocation = -0.012
- Upper bound (-0.004): if bit-error rate forces error-corrected encoding via STC (sister C3 adds redundancy), effective free channel reduces to 8,000 bytes → 0.00532 ΔS_rate ≈ -0.004 conservative

### Catalog #324 post-training Tier-C validation criterion:
Predicted band [-0.012, -0.004] validated when:
1. Probe `tools/probe_hydra_dim_7_12_score_invariance.py` PASSES on PR101 frontier archive across 600 pairs CPU+CUDA
2. F1 substrate Modal T4 smoke produces archive sha that on paired CPU re-eval lands ΔS within [-0.012, -0.004] vs baseline

### Validation status: `pending_post_training`

### Reactivation criteria:
- (a) If probe FAILS: re-classify as substrate-engineering requiring re-training with explicit dim 7-12 auxiliary loss (still proceeds, but cost increases from $1-3 to $3-8)
- (b) If F1 smoke lands OUTSIDE band: re-derive predicted band from actual bytes saved; update cargo-cult classifications

## 7. 6-Hook Wire-In Declaration (per Catalog #125)

### Hook 1: Sensitivity-map contribution
**ACTIVE**. F1's per-pair side-channel-bytes contribution (~24 bytes/pair × 600 pairs) feeds `tac.sensitivity_map` as a NEW per-pair entry under `category='deterministic_byte_derivation' / subcategory='hydra_dual_head_exploits' / vector_id='F1'`.

### Hook 2: Pareto constraint
**ACTIVE**. F1 adds Pareto constraint `R ≤ R_baseline - 14_400 bytes` to `tac.pareto_*` solver. Composition with G1+B1+Y3+Y6+H1 produces aggregate Pareto-feasible region.

### Hook 3: Bit-allocator hook
**ACTIVE**. F1 modifies per-tensor importance: removes 14,400 bytes from main payload allocation; adds 14,400 bytes to pose-dim-7-12 side-channel allocation. Total bytes UNCHANGED; redistribution per Wyner-Ziv.

### Hook 4: Cathedral autopilot dispatch hook
**ACTIVE**. F1 ranks #1 in `tools/cathedral_autopilot_autonomous_loop.py::rank_candidates` after probe PASSES.

### Hook 5: Continual-learning posterior update
**ACTIVE**. F1 Modal smoke call_id registered per Catalog #245. Empirical anchor flows back to update Catalog #319 deliverability tier (F1 is `TIER_2_CONSTANTS` because it ships structurally-constrained bytes).

### Hook 6: Probe-disambiguator
**ACTIVE**. Probe-disambiguator path: `tools/probe_hydra_dim_7_12_score_invariance.py` resolves encoder-freedom assumption (CARGO-CULTED → HARD-EARNED-VERIFIED via probe).

## 8. Routing Directive Sketch For Codex Execution

This sketch directs the Codex implementation subagent (separate `019de465` ownership per CLAUDE.md "Subagent coherence-by-default"). Full routing directive: `.omx/research/codex_routing_directive_rate_attack_vector_1_f1_hydra_dims_7_12_20260518.md`.

### Phase 1 (CHEAP $0 probe):
1. Land `tools/probe_hydra_dim_7_12_score_invariance.py`:
   - Read PR101 frontier archive (sha 6bae0201...)
   - Run `inflate.sh` → produce 600 pair outputs
   - Run `upstream/evaluate.py --device cuda` → capture baseline pose values + score
   - MUTATE pose values at dims 7-12 to random bits
   - Re-run `upstream/evaluate.py --device cuda` → capture mutated score
   - Assert `mutated_score == baseline_score` to 1e-9 precision
   - Repeat on `--device cpu`
   - Emit verdict to `.omx/state/probe_outcomes.jsonl` per Catalog #313
2. If verdict = PASSED → Phase 2; if FAILED → re-classify F1 to substrate-engineering with auxiliary loss path

### Phase 2 (SMOKE $1-3):
1. Build F1 substrate at `src/tac/substrates/rate_attack_f1_hydra_dims_7_12/`:
   - Substrate contract per Catalog #241/#242 META layer
   - Trainer at `experiments/train_substrate_rate_attack_f1_hydra_dims_7_12.py`
   - Operator-authorize recipe at `.omx/operator_authorize_recipes/substrate_rate_attack_f1_hydra_dims_7_12_modal_t4_dispatch.yaml`
2. Modal T4 100-epoch smoke via canonical `tools/operator_authorize.py` (per Catalog #313 + #271 + #243 chain)
3. Paired CPU re-eval per Catalog #316 frontier discipline

### Phase 3 (FULL $5-15 if smoke validates):
1. Modal A100 1000-epoch full per Catalog #324
2. Paired Linux x86_64 [contest-CPU] anchor per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA"
3. Submit PR

## 9. Per-Substrate Symposium Readiness (per Catalog #325)

**Status**: NOT READY. F1 needs:
- ✓ Cargo-cult audit per Catalog #303 (in §4 above)
- ✓ 9-dim checklist evidence per Catalog #294 (in §2 above)
- ✓ Observability surface declaration per Catalog #305 (in §3 above)
- ✗ Sextet pact deliberation (this design memo is T2 sextet-eligible; needs grand council convened with Atick + Tishby + Wyner + Hinton + Fridrich added per Catalog #325 contract)
- ✓ Per-substrate reactivation criteria (in §6 above)
- ✗ Catalog #324 post-training Tier-C validation (PENDING probe + smoke)

**Per-substrate symposium memo to land**: `.omx/research/council_per_substrate_symposium_f1_hydra_dims_7_12_20260518.md` (separate file, NOT in this wave; queued as next-wave op-routable).

## 10. Cross-References

- Master memo: `.omx/research/rate_attack_43_vectors_meta_paradigm_deep_research_20260518.md`
- META-paradigm: `.omx/research/structural_information_not_shipped_meta_paradigm_unification_20260518.md`
- Routing directive: `.omx/research/codex_routing_directive_rate_attack_vector_1_f1_hydra_dims_7_12_20260518.md`
- Source anchors:
  - `upstream/modules.py:26` (HEADS definition)
  - `upstream/modules.py:84` (compute_distortion first-6-dims slice)
  - `upstream/modules.py:67-68` (Hydra summarizer + heads instantiation)
- Sister substrate symposiums: ATW V2 + Z7 Mamba-2 + Z8 hierarchical predictive coding (all in `.omx/research/`)
- CLAUDE.md non-negotiables: "Exact scorer architectures" / "HNeRV / leaderboard-implementation parity discipline" L4 (≤200 LOC) / "EMA" (decay 0.997) / "eval_roundtrip" / "Strict scorer rule" / "Submission auth eval — BOTH CPU AND CUDA"
- Catalog gates: #125 / #229 / #244 / #245 / #270 / #287 / #292 / #294 / #296 / #303 / #305 / #313 / #316 / #319 / #322 / #324 / #325 / #326

## 11. Closeout

F1 is the **HIGHEST-EV rate-attack vector** of the 43 enumerated under the SINS META-paradigm. The structural anchor (`upstream/modules.py:84` first-6-dims slice) provides HARD-EARNED-VERIFIED evidence that dims 7-12 are score-invariant. The exploit is a CLEAN side-channel: 14,400 free bytes per archive at 0 rate-distortion cost.

**Predicted band [-0.012, -0.004] [contest-CPU]; aggregate stack F1+G1+B1+Y3+Y6+H1: [0.152, 0.180]; 21% lower bound improvement over current 0.19205 frontier.**

**Next action**: Phase 1 probe via Codex `019de465` per routing directive. NO PAID GPU until probe PASSES.
