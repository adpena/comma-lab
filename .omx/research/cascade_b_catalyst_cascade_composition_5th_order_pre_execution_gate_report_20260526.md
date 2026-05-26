# CASCADE B CATALYST CASCADE COMPOSITION — 5th-order recursive doctrine — PRE-EXECUTION GATE REPORT 2026-05-26

**UTC**: 2026-05-26T20:37:30Z
**Subagent**: `cascade-b-catalyst-cascade-composition-p5-qat-p10-bpr1-onto-path-a-foundation-5th-order-recursive-doctrine-mlx-first-numpy-portable-20260526`
**Lane**: `lane_cascade_b_catalyst_cascade_composition_5th_order_distortion_full_scorer_attack_20260526`
**Sister-of**: commit `4c73be3e4` (Cascade B Path A sister wave 1 production-scale 600f×1000ep PARADIGM-VALIDATED-with-plateau)
**Operator approval**: 2026-05-26 verbatim *"all are approved + follow up are approved + pursue other attacks as well + remember all MLX first + individually fractally optimized"*
**Mission contribution per Catalog #300**: `frontier_breaking_enabler` (5th-order CATALYST cascade composition lands FIRST empirical anchor on canonical equation #2 `hinton_kl_distill_enables_qat_catalyst_composition_savings_v1` — currently 0 empirical anchors; this wave establishes the predicted-vs-empirical baseline for the entire CATALYST composition family).

## 1. Three-strategy attack classification

- **PRIMARY = DISTORTION + FULL-SCORER** per CLAUDE.md "Frontier target" + the canonical 3-strategy doctrine.
- DISTORTION axis: KL-distilled student head (Path A foundation) reduces d_seg via tighter softmax-temperature scorer-entropy targeting.
- FULL-SCORER axis: CATALYST composition spans P2 (loss-shape via KL) + P5 (quantization-entropy via FakeQuantFP4 QAT) + P10 (sidecar-entropy via BPR1 brotli-compressed residual). The composition's hypothesis is that KL T=2.0 logit-sharpening at P2 catalyzes tighter post-QAT scorer-entropy targeting at P5, lifting the QAT savings RELATIVE to QAT alone (canonical equation #2 latex form: `ΔS_cat(P2 → P4 → P10) = ΔS_{P4}^{alone} · (1 + α · ΔH_{logits}^{T=2})` with `α ∈ [0.1, 0.2]`).

## 2. Entropy-position discipline § 10 alignment

Per the just-landed entropy-position discipline § 10 CATALYST composition pattern, this wave instantiates the canonical triangular triple (P2 catalyst → P4 enabled → P10 output) on the Hinton substrate:

- **P2 loss-shape (CATALYST)**: Hinton KL T=2.0 distillation via `LearnableConv1x1StudentHead` (Path A foundation; 20 params; production-scale validated at 42% KL reduction).
- **P4/P5 quantization-entropy (ENABLED)**: FakeQuantFP4 STE applied to LearnableConv1x1StudentHead weight + bias (canonical `tac.fp4_quantize.fake_quant_fp4`).
- **P10 sidecar-entropy (OUTPUT)**: BPR1 sign-bitmap sidecar (canonical `tac.substrates.boost_nerv_pr110_residual.bpr1_variant_b_sign_bitmap_codec.build_variant_b_d_sidecar`) consuming POST-QAT student logits as the residual surface; brotli-compressed packed sign-bits + per-pair magnitude scalar.

The composition rule per equation #2 schema: KL distillation BEFORE QAT enables tighter post-quantization scorer-entropy targeting because the KL-distilled student's logit distribution is SHARPER (lower marginal entropy via T=2.0 softening + post-distill weight-space sharpening), so integer-codeword Huffman / brotli at the QAT output has lower entropy → smaller BPR1 sidecar bytes for equivalent reconstruction quality.

## 3. MLX-first + numpy-portable bridge contract

Per CLAUDE.md "MLX portable-local-substrate authority" non-negotiable + Catalog #192/#317 + just-landed MLX-first bridge directive:

- TRAINING: MLX-native via `mx.value_and_grad` on `LearnableConv1x1StudentHead` (Path A foundation reused unchanged).
- QAT INSERTION: Stage B is a STE-style fake-quantization simulation; the canonical `FakeQuantFP4` PyTorch path is the reference oracle. We implement an MLX-native sister of the same codebook + per-block-scale + nearest-codebook-entry pattern so QAT can train inside the same MLX optimizer step without crossing the MLX↔PyTorch boundary at every batch.
- BRIDGE: MLX-trained quantized weights → numpy `.npz` (uncompressed for byte-stable parity) via the sister canonical `tac.local_acceleration.mlx_to_pytorch_export` pattern.
- INFLATE: numpy-portable; FP4 codebook + per-block scales + BPR1 sidecar bytes consumed at decode time by an inflate routine that uses ONLY numpy + brotli (no PyTorch / MLX) per HNeRV parity L4 (≤200 LOC + ≤2 deps).

## 4. Individually-fractal decomposition (5th-order recursive doctrine continuation)

- 1st order: Cascade B Hinton substrate scaffold (commit `15b11c86e`).
- 2nd order: Path A learnable head 50f×100ep PARADIGM-VALIDATED (sister scaffold; 24.8% reduction).
- 3rd order: production-scale 600f×1000ep PARADIGM-VALIDATED-with-plateau (commit `4c73be3e4`; 42.0% reduction; min KL 3.36 sits in PARTIAL_CONVERGENCE_EXTENDED band, near the 20-param capacity ceiling).
- 4th order: capacity-sweep sister (1×1 → 3×3 conv; secondary operator-routable; DEFERRED to sister wave).
- **5th order (THIS WAVE)**: CATALYST cascade composition P5 QAT + P10 BPR1 onto Path A foundation. Tests the hypothesis that the plateau at ~3.4 KL is NOT a paradigm ceiling but a SUBSTRATE-CLASS BOTTLENECK that the CATALYST composition can BYPASS via post-QAT scorer-entropy targeting + BPR1 sidecar bytes.

## 5. Canonical-vs-unique decision per layer (Catalog #290)

| Layer | Decision | Rationale |
|---|---|---|
| Hinton KL T=2.0 loss | ADOPT_CANONICAL | sister Path A foundation; canonical `tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss` |
| LearnableConv1x1StudentHead | ADOPT_CANONICAL | sister Path A foundation; 20-param head proven monotonic-descent capable |
| FakeQuantFP4 codebook | ADOPT_CANONICAL | canonical `tac.fp4_quantize.DEFAULT_CODEBOOK = [0,0.5,1,1.5,2,3,4,6]` matches mask2mask competitor + Quantizr empirical anchor |
| FakeQuantFP4 STE forward path | FORK_BECAUSE_PRINCIPLED_MISMATCH | canonical `FakeQuantFP4` is a `torch.autograd.Function`; MLX has no `torch.autograd.Function`. The fork is a closed-form codebook-nearest projection on MLX arrays; gradients flow via MLX's autodiff over the identity-STE pattern (equivalent semantics; structurally distinct API) |
| BPR1 sign-bitmap codec | ADOPT_CANONICAL | sister `tac.substrates.boost_nerv_pr110_residual.bpr1_variant_b_sign_bitmap_codec` is numpy-only + accepts arbitrary 4D residual tensors |
| brotli quality-9 | ADOPT_CANONICAL | canonical compression for all sidecar bytes |
| Provenance for anchor | ADOPT_CANONICAL | `tac.provenance.builders.build_provenance_for_predicted` per Catalog #323/#341 |
| Canonical equation anchor | ADOPT_CANONICAL | `tac.canonical_equations.update_equation_with_empirical_anchor` per Catalog #344 |

## 6. 9-dimension success checklist evidence (Catalog #294)

| Dimension | Evidence |
|---|---|
| 1. UNIQUENESS | Distinct from sister wave 4th-order capacity-sweep (different axis: composition vs capacity). First CATALYST composition (P2+P5+P10) empirical anchor for the substrate-class. |
| 2. BEAUTY + ELEGANCE | Three orthogonal entropy positions composed via canonical primitives; pipeline reviewable in <1 module (~400 LOC budget). |
| 3. DISTINCTNESS | First wave to compose Path A foundation with FP4 QAT + BPR1 sidecar; sister 3rd order tested Path A alone, sister 4th order tests capacity. |
| 4. RIGOR | Catalog #229 premise verification (read 8 source files); Catalog #292 per-deliberation assumption-statement; Catalog #307 paradigm-vs-implementation classification; Catalog #344 canonical equation anchor. |
| 5. OPTIMIZATION PER TECHNIQUE | KL T=2.0 (Quantizr canonical); FP4 codebook (mask2mask canonical); BPR1 sign-bitmap (Variant B-d frontier-push canonical). |
| 6. STACK-OF-STACKS COMPOSABILITY | Per equation #2 latex form: ΔS_cat = ΔS_{P4}^{alone} · (1 + α · ΔH_{logits}). Composition is multiplicative on the QAT savings axis with bounded `α ∈ [0.1, 0.2]`. |
| 7. DETERMINISTIC REPRODUCIBILITY | seed=0 + canonical fixture + MLX deterministic init (sister Path A pattern reused). |
| 8. EXTREME OPTIMIZATION + PERFORMANCE | MLX-local M5 Max; no paid GPU; ~2-3h budget. |
| 9. OPTIMAL MINIMAL CONTEST SCORE | Predicted ΔS: -2 to -8 score points via DISTORTION axis + additional rate-axis savings via CATALYST QAT bit-rounding compression (canonical equation #2 prediction; bounded `α ∈ [0.1, 0.2]`). Per `[macOS-MLX research-signal]` axis; NOT a contest claim per Catalog #192/#317. |

## 7. Cargo-cult audit (Catalog #303)

| Assumption | Classification | Unwind-test plan |
|---|---|---|
| FP4 codebook `[0,0.5,1,1.5,2,3,4,6]` is optimal for student-head weights | CARGO-CULTED | mask2mask codebook was tuned for renderer weights, not 20-param student heads. Unwind via RESIDUAL_CODEBOOK (denser-near-zero) comparison if FP4-QAT KL diverges. |
| KL distillation before QAT > QAT alone (the CATALYST hypothesis) | CARGO-CULTED-NEEDS-EMPIRICAL | Equation #2 has 0 empirical anchors. This wave is the FIRST anchor. May empirically VINDICATE or FALSIFY. |
| BPR1 sign-bitmap is canonical for arbitrary residual surfaces | HARD-EARNED | sister Variant B-d landing (commit `1075a2f30`) proved the sign-bitmap-only encoding extracts ~all information when post-tanh residuals saturate; the codec is substrate-agnostic. |
| Per-block-size=32 FP4 quantization scale is optimal for 20-param head | CARGO-CULTED | 20 params is less than 1 block; block_size=32 would treat the head as a single block. Unwind via block_size=8 OR block_size=20 (whole-head). |

## 8. Observability surface (Catalog #305)

- **Inspectable per layer**: per-stage MLX array snapshots (Path A logits / FP4-quantized logits / BPR1 residual surface) captured at end of training.
- **Decomposable per signal**: per-axis breakdown (d_seg via KL convergence delta + rate via archive-bytes delta) at every checkpoint.
- **Diff-able across runs**: deterministic init (seed=0); two runs with identical params should produce bit-identical KL trajectory.
- **Queryable post-hoc**: JSON output per-stage; canonical schema `cascade_b_catalyst_cascade_composition_5th_order_verdict_v1_20260526`.
- **Cite-able**: every output row carries canonical Provenance per Catalog #323 + commit SHA + UTC timestamp.
- **Counterfactual-able**: pipeline runs three arms (baseline / Path A alone / CATALYST composition); per-axis residual computable across arms.

## 9. Drift-vs-depth characterization

Sister Path A 3rd-order production scale revealed PLATEAU at ~3.4 KL after 1000ep (asymptotic descent rate slows from -1.82 nats/100ep → -0.022 nats/100ep). The 5th-order CATALYST hypothesis: post-QAT scorer-entropy targeting tightens the achievable logit-distribution sharpness BEYOND the 20-param capacity ceiling because the QAT-induced weight discretization concentrates probability mass on a smaller integer-codeword vocabulary → lower marginal entropy → tighter targeted distribution.

## 10. Predicted-band per Catalog #296 Dykstra-feasibility

Predicted `α` ∈ [0.1, 0.2] per equation #2 schema. Predicted KL reduction from CATALYST composition beyond Path A baseline 3.4 plateau: 5-15% additional reduction (final KL ~2.9-3.2 range). Predicted ΔS: -2 to -8 score points (Shannon R(D) bound: the BPR1 sidecar's compressed residual cost is bounded by H(sign_bitmap) per pixel ≈ 1 bit, so the rate-axis contribution is bounded ~bytes_added/37545489).

Dykstra-feasibility: the three convex constraints (rate ≤ R_budget, d_seg ≤ S_baseline, d_pose ≤ P_baseline) form a convex feasible set; the CATALYST composition's alternating-projections iterate is (a) Path A foundation update via MLX gradient (reduces d_seg) + (b) FP4-QAT projection onto codebook (changes rate via discretization) + (c) BPR1 sidecar emission (additive rate cost bounded). Per Dykstra non-expansive property, the iterate converges to a point in the feasible set; the question is whether the limit point dominates Path A alone.

## 11. Horizon-class per Catalog #309

`horizon_class: plateau_adjacent` (substrate-stacking; sister to NSCS06 v8 PR111 candidacy chain). The CATALYST composition does NOT shift the substrate class (HNeRV-family + Hinton-distilled student remains the substrate); it stacks orthogonal entropy-position primitives.

## 12. Canonical equation #344 anchor declaration

Target equation: `hinton_kl_distill_enables_qat_catalyst_composition_savings_v1` (currently 0 empirical anchors). This wave lands the FIRST empirical anchor regardless of empirical outcome (FALSIFICATION rows are equally valuable per Catalog #344). Anchor ID: `hinton_kl_distill_qat_catalyst_5th_order_cascade_composition_50f_100ep_mlx_local_20260526T<UTC>`.

## 13. 6-hook wire-in declaration (Catalog #125)

- Hook #1 sensitivity-map: ACTIVE (per-axis d_seg + rate residuals contribute to `tac.sensitivity_map.*` via canonical Provenance).
- Hook #2 Pareto constraint: ACTIVE (predicted bands feed `tac.findings_lagrangian.posterior_update_from_anchors` per Catalog #356 sister).
- Hook #3 bit-allocator: ACTIVE (BPR1 sidecar bytes are a first-class bit-allocator contribution).
- Hook #4 cathedral autopilot dispatch: ACTIVE (canonical equation #2 anchor auto-discovered via `tac.cathedral_consumers.canonical_equation_lookup_consumer` per Catalog #335).
- Hook #5 continual-learning posterior: ACTIVE (anchor lands via `tac.canonical_equations.update_equation_with_empirical_anchor` per Catalog #344).
- Hook #6 probe-disambiguator: ACTIVE (3-arm comparison (baseline / Path A alone / CATALYST composition) IS the canonical disambiguator).

## 14. Sister coordination (Catalog #230 ownership map)

Active sisters at launch:
- UNIWARD N+1 real-scorer empirical (#1369; PURE DIST per-pixel; OUT of `uniward_per_pixel_distortion`) — disjoint.
- Meta-Lagrangian Phase 3 typed atom flow (#1372; FULL-SCORER apparatus-scope; OUT of `findings_lagrangian`) — disjoint.
- NSCS06 v8 Modal CUDA in flight (PAID; operationally separate) — disjoint.

THIS wave: STRICTLY confined to `src/tac/substrates/hinton_distilled_scorer_surrogate/` (extends substrate per APPEND-ONLY) + READ-ONLY consumer of `tac.substrates.boost_nerv_pr110_residual.bpr1_variant_b_sign_bitmap_codec` + READ-ONLY consumer of `tac.fp4_quantize.FakeQuantFP4` codebook constants + NEW canonical equation anchor.

## 15. Discipline checklist

- [x] Catalog #229 PV (8 prerequisite files read pre-execution; verified canonical surfaces)
- [ ] Catalog #117/#157/#174 canonical serializer + POST-EDIT --expected-content-sha256 (at commit time)
- [x] Catalog #206 checkpoint discipline (checkpoint #1 emitted; #2 will fire after pipeline lands)
- [x] Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE (NEW files only; sister Path A foundation unchanged)
- [x] Catalog #230 ownership map (zero overlap with active sisters)
- [x] Catalog #287 placeholder-rationale rejection (rationales ≥4 chars throughout)
- [x] Catalog #343 no hardcoded score literals (only KL/distillation/predicted-α values; no contest-axis claims)
- [x] CLAUDE.md "MLX portable-local-substrate authority" (every anchor tagged `[macOS-MLX research-signal]`; score_claim=False)
- [x] CLAUDE.md "QAT pipeline — non-negotiable" (canonical `tac.fp4_quantize.FakeQuantFP4` semantics reused; MLX-native sister codebook projection has equivalent forward/backward STE pattern)
- [x] CLAUDE.md "Forbidden premature KILL" (Catalog #307 verdict taxonomy preserves PARADIGM regardless of implementation outcome)
- [x] CLAUDE.md "GUIDING PRINCIPLE individually fractal" (5th-order continuation; per-substrate 13-ingredient tree honored)
- [x] CLAUDE.md "MLX-first + numpy-portable bridge" (TRAINING in MLX; INFLATE in numpy + brotli only)
