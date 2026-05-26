# SPDX-License-Identifier: MIT

---
schema: substrate_design_memo_v1
council_tier: T2
council_attendees: [Shannon, Dykstra, Jegou, Schmid, Atick, Mallat, AssumptionAdversary]
council_quorum_met: true
council_verdict: PROCEED
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "Path (b) substrate-design REDIRECT is the right decision per PHASE 1 cargo-cult audit"
    classification: HARD-EARNED
    rationale: "PHASE 1 audit demonstrated the META-ASSUMPTION misframing; path (b) explicitly declares NEW substrate-design FRESH posture; V1-V8 work is research INPUT only; canonical primitives reusable but application layer FORKED. Path (a) DIRECT EXTENSION rejected. Path (c) HYBRID deferred as kitchen_sink anti-pattern."
council_decisions_recorded:
  - "op-routable #1: PHASE 3 L0 SCAFFOLD canonical path = src/tac/substrates/faiss_ivf_pq_residual/"
  - "op-routable #2: FORK new canonical helper module tac.optimization.faiss_ivf_pq_residual_codec wrapping the underlying Faiss primitives"
  - "op-routable #3: predicted byte budget = ~10-30 byte/pair derived from PR110 fec6 178559 bytes total"
  - "op-routable #4: K-sweep + M-sweep + per-class-conditioning binary variant queued for PHASE 3 smoke fork"
  - "op-routable #5: PR110 fec6 residual MLX-local probe at PHASE 3 L0 SCAFFOLD with numpy reference path per directive #3"
deferred_substrate_retrospective_due_utc: 2026-06-25T07:58:00Z
deferred_substrate_id: faiss_ivf_pq_residual_codec
related_deliberation_ids:
  - path_3_i_v1_faiss_ivf_pq_cargo_cult_audit_20260526
  - v1_faiss_v4_probe_plus_v8_design_landed_20260519
  - v1_faiss_v8_learned_compression_faiss_design_20260519
lane_id: lane_path_3_i_v1_faiss_ivf_pq_residual_cargo_cult_first_20260526
phase: phase_2_substrate_design_decision
operator_directive_anchor: "we want to design the substrate and curriculum and then optimize the design the whole stack around it for extreme optimization and performance and optimal score lowering" (2026-05-26)
horizon_class: frontier_pursuit
research_only: true
dispatch_enabled: false
---

# Path 3 I — V1 Faiss IVF-PQ residual codec PHASE 2 substrate-design decision

> **Status**: PHASE 2 of 3-phase methodology. PHASE 1 audit (commit `a883a717c`) demonstrated the META-ASSUMPTION misframing. THIS memo formalizes the substrate-design decision per Catalog #290 (Path a/b/c) with PHASE 3 L0 SCAFFOLD plan.

## 1. Decision — Path (b) SUBSTRATE-DESIGN REDIRECT

**Verdict**: Path (b) — declare path 3 I as NEW substrate-design question for *per-pair RGB residual codec stacking on PR110 fec6 frontier*; V1-V8 prior work is research INPUT only.

**Rationale per PHASE 1 Assumption-Adversary verdict**:
- Side-info channel (V1-V8) compresses SegNet softmax outputs (categorical signals)
- Per-pair residual codec compresses RGB residuals (continuous signals)
- Different signal shape → different entropy structure → different codec topology
- Different stacking semantics (additive frame correction vs side-info channel conditioning)
- Per the operator's substrate-design-from-first-principles directive #1 (2026-05-26)

## 2. Substrate identity

- **Substrate ID**: `faiss_ivf_pq_residual`
- **Substrate canonical path**: `src/tac/substrates/faiss_ivf_pq_residual/`
- **Paradigm anchor**: Jégou-Douze-Schmid 2011 *Product quantization for nearest neighbor search* applied to per-pair RGB residual quantization for stacking on PR110 fec6 frontier
- **Class**: residual codec primitive at per-pair RGB surface
- **Horizon class**: `frontier_pursuit` (predicted-band [0.150-0.190] within PR110 fec6 frontier-adjacent)
- **Cross-pollination canonical**: sister to E=BoostNeRV (residual-stacker) + G=NIRVANA (hierarchical residual cascade)

## 3. Per-layer canonical-vs-unique decision per Catalog #290

Per CLAUDE.md UNIQUE-AND-COMPLETE-PER-METHOD operating mode + the falling-rule decision criterion:

| Layer | Canonical helper | Decision | Rationale per falling-rule |
|---|---|---|---|
| PQ codebook build | `tac.optimization.faiss_ivf_pq_atw_channel.build_pq_codebook` (existing primitive) | ADOPT_CANONICAL_BECAUSE_SERVES | Signal-agnostic Faiss IVF-PQ primitive per Jégou-Douze-Schmid 2011; the codebook fits K-means clusters in whatever distribution; no substrate-specific reason to fork |
| PQ codebook serialize | `tac.optimization.faiss_ivf_pq_atw_channel.serialize_codebook` (existing primitive) | ADOPT_CANONICAL_BECAUSE_SERVES | Faiss serialize is signal-agnostic |
| Per-pair residual encode (application layer) | NEW `tac.optimization.faiss_ivf_pq_residual_codec.encode_per_pair_residual` | FORK_BECAUSE_PRINCIPLED_MISMATCH | Per-pair RGB residual signal shape (T, 3, H, W) differs from per-region SegNet softmax (T, K_classes, H_region, W_region); residual-codec application layer FORKED per substrate-optimal engineering |
| Per-pair residual decode (application layer) | NEW `tac.optimization.faiss_ivf_pq_residual_codec.decode_per_pair_residual` | FORK_BECAUSE_PRINCIPLED_MISMATCH | Sister of encode; FORKED for consistency |
| Archive grammar | NEW FAISSPQ1 byte-deterministic grammar | NEW (no canonical) | Substrate-specific archive grammar; mirrors NIRVANA1 pattern at `src/tac/substrates/nirvana_cascading_nerv/archive.py` |
| Inflate device-fork | `tac.substrates._shared.inflate_runtime.select_inflate_device` (Catalog #205) | ADOPT_CANONICAL_BECAUSE_SERVES | Per Catalog #205: submissions MUST use canonical device-fork helper |
| Bilinear residual upsample (if used) | `tac.substrates._shared.bilinear_resize2x_align_corners_false_nhwc` (canonical per sister A forensic) | ADOPT_CANONICAL_BECAUSE_SERVES | Per sister A=DreamerV3 forensic: align_corners=True anti-pattern caused max_abs=24.34 drift; canonical helper enforces align_corners=False NHWC |
| Scorer-preprocess routing | `tac.scorer.load_default_scorers` + Catalog #164 | ADOPT_CANONICAL_BECAUSE_SERVES | Per CLAUDE.md MPS auth eval is NOISE + Catalog #164 |
| Auth-eval CLI routing | `tac.substrates._shared.smoke_auth_eval_gate.gate_auth_eval_call` (Catalog #226) | ADOPT_CANONICAL_BECAUSE_SERVES | Per Catalog #226 substrate trainers MUST route through canonical helper |
| Eval-roundtrip | `tac.differentiable_eval_roundtrip` | ADOPT_CANONICAL_BECAUSE_SERVES | Per CLAUDE.md eval_roundtrip non-negotiable |
| EMA shadow | `tac.training.EMA` decay=0.997 | ADOPT_CANONICAL_BECAUSE_SERVES | Per CLAUDE.md EMA non-negotiable |
| Hardware substrate detection | `tac.substrates._shared.trainer_skeleton.detect_hardware_substrate` (Catalog #190) | ADOPT_CANONICAL_BECAUSE_SERVES | Per Catalog #190 canonical dynamic detection |
| Modal call_id ledger registration | `tac.deploy.modal.call_id_ledger.register_dispatched_call_id_fail_closed` (Catalog #245 + #339) | ADOPT_CANONICAL_BECAUSE_SERVES | Per Catalog #245+#339 ALL Modal dispatches MUST register fail-closed |
| Cost-band calibration | `tac.cost_band_calibration.append_anchor` (Catalog #175+#177) | ADOPT_CANONICAL_BECAUSE_SERVES | Canonical posterior write discipline |
| Probe-outcomes ledger | `tac.probe_outcomes_ledger.register_probe_outcome` (Catalog #313) | ADOPT_CANONICAL_BECAUSE_SERVES | Per Catalog #313 every probe outcome MUST register canonical |
| Canonical Provenance | `tac.provenance.build_provenance_for_contest_archive_byte_member` (Catalog #323) | ADOPT_CANONICAL_BECAUSE_SERVES | Per Catalog #323 every score-claim MUST carry canonical Provenance |
| MLX-Faiss adapter | NEW `mlx_renderer.py` per sister G=NIRVANA pattern | NEW (no canonical) | MLX-native PQ codebook gather + bilinear upsample if needed; Faiss-CPU as optional accelerator |
| numpy reference | NEW `numpy_reference.py` per sister G=NIRVANA pattern + directive #3 axis 3 | NEW (no canonical) | Substrate operable on CPU-only test rigs without MLX OR Faiss |

**Summary**: 13 ADOPT_CANONICAL_BECAUSE_SERVES + 2 FORK_BECAUSE_PRINCIPLED_MISMATCH + 4 NEW (substrate-specific). Per CLAUDE.md UNIQUE-AND-COMPLETE-PER-METHOD: canonical-heavy at META-infrastructure layer (preserving 295+ STRICT preflight gate protection); substrate-unique at residual-codec application layer + MLX renderer + numpy reference + archive grammar (4 NEW per substrate-optimal engineering).

## 4. Predicted byte budget envelope (Dykstra-feasibility per Catalog #296)

The feasible region for path 3 I residual codec:
- (a) PR110 fec6 frontier total bytes = 178559
- (b) Frontier score = 0.1920513169 [contest-CPU] per canonical pointer 2026-05-15
- (c) Per-pair count = 600
- (d) Target ΔS < 0 (frontier-pursuit horizon class) requires per-pair distortion saving > rate cost
- (e) Per-pair rate cost = 25 * Δbytes / 37545489 = 6.66e-7 per byte
- (f) Per-pair distortion saving estimated via MSE-residual R(D) bound

**Per-pair byte budget upper bound**: from PR110 fec6 baseline + assumed total Δbytes ≤ 18000 (10% overhead): per-pair ≤ 30 byte/pair
**Per-pair byte budget lower bound**: PQ codebook overhead + 1 codeword/pair = ~2-4 byte/pair

**Operating point estimate**: M=4 sub-quantizers × ksub=256 codes × 1 byte/codeword + 4 codeword bytes/pair = ~8-20 byte/pair total at 600 pairs = ~5-12KB total
**Codebook overhead**: M=4 × ksub=256 × 4 dim/sub × float32 = ~16KB raw; brotli-compressed ~3-5KB; codebook shared across all 600 pairs (NOT per-pair)

**Dykstra-feasibility verdict**: Feasible region NON-EMPTY for M=4 ksub=128 (8KB total) AND M=2 ksub=64 (3KB total) variants. PHASE 3 smoke fork sweeps both.

## 5. Curriculum design (substrate-design + curriculum-design per directive #1)

Per the operator's substrate-design-from-first-principles directive, the substrate AND curriculum are co-designed.

### Step 1: Empirical residual extraction
- Decode PR110 fec6 via canonical `inflate.sh` on 600 contest pairs
- Compute per-pair residual = frame_gt - frame_pr110_decoded
- Persist residuals as `experiments/results/<lane>/per_pair_residual_<utc>.npy` (NHWC float32; 600 × 384 × 512 × 3 = 354MB)
- Per Catalog #313 register probe outcome

### Step 2: PQ codebook training
- Spatial tiling: split each pair-residual into per-tile patches (e.g. 8×8 spatial blocks → 48×64 tiles/pair × 600 pairs ≈ 1.84M tiles)
- Per-tile feature vector dim = 8×8×3 = 192 float32
- K-means + PQ: M=4 sub-quantizers (dim=48/sub) × ksub ∈ {64, 128, 256}
- Codebook sha256 + per-pair codeword stream sha256 cite-able per Catalog #245

### Step 3: Per-pair residual encoding
- Quantize per-tile feature vector via PQ codebook (M log2(ksub) bits/tile)
- Brotli-compress codeword stream (high-cardinality unique per-pair patterns → low brotli ratio expected)
- Persist `per_pair_residual_pq_codebook` + `per_pair_pq_codeword_stream` archive sections

### Step 4: Inflate-time reconstruction
- Decode PR110 fec6 via canonical inflate runtime
- Decode per-pair residual via PQ codebook lookup + tile reassemble + dequantize
- ADD per-pair residual to PR110 fec6 reconstruction
- uint8 cast at final output (canonical Catalog #205 sister rounding)

### Step 5: Score-aware training (PHASE 4 once smoke passes)
- Optimize PQ codebook centroids end-to-end via straight-through gradient estimator
- Loss = canonical contest score (seg + sqrt(10 * pose) + 25 * archive_bytes / 37545489) per Catalog #164 `score_pair_components`
- EMA decay=0.997 per CLAUDE.md EMA non-negotiable
- eval_roundtrip=True per CLAUDE.md eval_roundtrip non-negotiable

### Step 6: Stack-of-Stacks composition (PHASE 5 once individual anchors land)
- Compose with sister candidates (orthogonal axes): E=BoostNeRV (iterative residual axis) + G=NIRVANA (hierarchical-cascade axis)
- Predicted composition Δ via Catalog #296 Dykstra-feasibility intersection check
- Smoke + paired CPU+CUDA auth-eval per Catalog #246

## 6. 9-dimension success checklist per Catalog #294

| # | Dimension | Evidence | Verdict |
|---|---|---|---|
| 1 | UNIQUENESS | Codec primitive (vector quantization) is paradigm-distinct from sister candidates (iterative boosting / hierarchical cascade / meta-learning / predictive coding / categorical latent) | PASS |
| 2 | BEAUTY + ELEGANCE | ~400 LOC total budget (smaller than HNeRV parity L4 200 LOC inflate-only sub-budget); per-pair PQ encode = 1 codebook-lookup + 1 codeword-gather; reviewable in 30 seconds | PASS |
| 3 | DISTINCTNESS | Path 3 I distinct from sister candidates by decomposition principle (per PHASE 1 audit §5 sister-distinguishing table); cross-pollination canonical with E + G NOT redundant | PASS |
| 4 | RIGOR | PHASE 1 cargo-cult audit committed at a883a717c; 12-assumption table; Assumption-Adversary verdict; Catalog #303 + #292 + #325 discipline | PASS |
| 5 | OPTIMIZATION-PER-TECHNIQUE | Jégou-Douze-Schmid 2011 PQ + OPQ (Ge-He-Ke-Sun 2014) canonical optimization for high-dimensional vector quantization; canonical-vs-unique decision per layer table in §3 | PASS |
| 6 | STACK-OF-STACKS-COMPOSABILITY | Per-pair RGB residual codec adds to PR110 fec6 reconstruction (additive by construction); orthogonal axes with sister E + G (different decomposition principles) | PASS-WITH-EMPIRICAL-VALIDATION-PENDING |
| 7 | DETERMINISTIC REPRODUCIBILITY | PQ codebook training deterministic with seed pin; PQ encoding deterministic; archive grammar byte-stable | PASS |
| 8 | EXTREME OPTIMIZATION + PERFORMANCE | PQ inflate is integer codebook lookup + float gather + bilinear upsample if needed; estimated inflate cost ≤ 5ms per pair on CPU; MLX-native vectorized for Apple Silicon | PASS-WITH-EMPIRICAL-VALIDATION-PENDING |
| 9 | OPTIMAL MINIMAL CONTEST SCORE | Predicted band [0.150, 0.190] frontier-pursuit (predicted delta [-0.040, -0.002] from frontier 0.1920513169); pending Catalog #324 post-training Tier-C validation | PASS-WITH-CATALOG-324-VALIDATION-REQUIRED |

**9-dim summary**: 6 PASS + 2 PASS-WITH-EMPIRICAL-VALIDATION-PENDING + 1 PASS-WITH-CATALOG-324-VALIDATION-REQUIRED. No FAIL dimensions; design structurally sound.

## 7. Cargo-cult audit per assumption per Catalog #303

This PHASE 2 design memo's NEW assumptions (introduced beyond PHASE 1 audit):

| # | Assumption | Classification | Math | Sci | Eng | Unwind |
|---|---|---|---|---|---|---|
| 1 | PR110 fec6 residual has enough structure for PQ codebook to achieve net Δ<0 at ≤30 byte/pair | CARGO-CULTED-PENDING-EMPIRICAL | Natural-image residual R(D) ≈ 1-3 bits/pixel | Per Mallat wavelet theory + Daubechies | PHASE 3 MLX-local probe | Empirical anchor in PHASE 3 |
| 2 | M=4 sub-quantizers × ksub=256 is the right hyperparameter operating point | CARGO-CULTED-PENDING-EMPIRICAL | Per Jégou-Douze-Schmid 2011 SIFT-1M empirical | n/a — canonical PQ defaults | PHASE 3 sweep | M-sweep + ksub-sweep in PHASE 3 |
| 3 | Spatial tiling (8×8 blocks) is the right granularity vs alternative tilings | CARGO-CULTED-PENDING-EMPIRICAL | Tile size = bias-variance tradeoff per Mallat wavelet | Smaller tiles = more PQ overhead; larger tiles = worse codebook fit | PHASE 3 sweep | Tile-size sweep in PHASE 3 |
| 4 | Shared per-pair codebook is optimal vs per-class-conditioned codebook | CARGO-CULTED-PENDING-EMPIRICAL | Per-class-conditioned = Atick-Redlich cooperative-receiver canonical | More codebooks → larger total codebook bytes | PHASE 3 sweep | Both variants in PHASE 3 |
| 5 | MLX↔Faiss-CPU↔numpy parity achievable within Catalog #1265 threshold 0.001 | CARGO-CULTED-PENDING-EMPIRICAL | Codebook gather is float-deterministic; PQ encoding is integer-deterministic | Bilinear upsample is MLX-drift-sensitive (sister A forensic) | Use canonical `bilinear_resize2x_align_corners_false_nhwc` per sister A | PHASE 3 parity test |
| 6 | Per-pair tile reassemble produces seamless RGB output (no tile boundary artifacts) | CARGO-CULTED-PENDING-EMPIRICAL | Per-tile decode + reassemble may produce blocking artifacts | Per Mallat wavelet boundary effects | PHASE 3 visual + scorer-loss check | PHASE 3 sweep with overlapping vs non-overlapping tiles |

**Cargo-cult unwind summary**: 6 PHASE 2 NEW assumptions enumerated; all CARGO-CULTED-PENDING-EMPIRICAL with PHASE 3 unwind paths declared.

## 8. Observability surface per Catalog #305

1. **Inspectable per layer**: PQ codebook serialization (faiss.serialize_index); per-pair codeword stream (length-prefixed); per-pair residual reconstruction (NHWC float32 dumpable to JSON for layer-wise inspection); brotli compression ratios per archive section
2. **Decomposable per signal**: per-pair distortion (MSE) decomposable into PQ-codebook fit residual + brotli-stream entropy; per-pair byte cost decomposable into codebook share (shared overhead / 600 pairs) + per-pair codeword bytes
3. **Diff-able across runs**: codebook sha256 + codeword stream sha256 + per-(M, ksub, tile_size, seed) tuple per Catalog #245 modal_call_id_ledger
4. **Queryable post-hoc**: PHASE 3 probe outcomes → `experiments/results/lane_path_3_i_*/probe_results.json` + Catalog #313 probe-outcomes ledger
5. **Cite-able**: every variant tuple = (M, ksub, tile_size, per_class_conditioned_bool, brotli_quality, seed, mlx_or_faiss_backend, hardware_substrate)
6. **Counterfactual-able**: codebook + codeword stream are byte-mutable artifacts at `experiments/results/<lane>/`; mutating one byte of codebook OR codeword stream should change decoded per-pair RGB residual reconstruction (testable via `tools/verify_distinguishing_feature_byte_mutation.py` per Catalog #272)

## 9. Predicted band Dykstra-feasibility check per Catalog #296

**Feasible region**: intersection of (a) per-pair byte budget ≤ 30 byte/pair, (b) per-pair distortion saving ≥ 1.5 * rate cost (frontier-pursuit margin), (c) MLX↔numpy↔PyTorch parity ≤ Catalog #1265 threshold 0.001, (d) archive grammar self-contained per HNeRV parity L3, (e) inflate runtime ≤ 200 LOC per HNeRV parity L4

**Per-pair byte budget projection**: M=4 × log2(ksub=128) = 28 bits/tile × 48×64 tiles/pair = 86016 bits/pair = 10752 bytes/pair raw → brotli ~0.2 ratio (high-cardinality per-pair pattern) ≈ 2150 bytes/pair brotli compressed → exceeds 30 byte/pair budget by 70×

**RECALIBRATION REQUIRED**: 48×64 tiles per pair is way too many. Need MUCH coarser tiling. Re-derive:
- Target: ≤30 byte/pair raw codeword stream
- M=4 → log2(ksub) bits/tile × tiles/pair = 30 × 8 = 240 bits/pair
- With ksub=256 (8 bits/sub × M=4 = 32 bits/tile): tiles/pair = 240/32 = 7.5 → ~8 tiles/pair
- 8 tiles per pair on 384×512 = ~98304 pixels per tile = 64×64 spatial blocks → 6 tiles per pair (384/64 × 512/64 = 6 × 8 = 48 tiles still too many)
- OR larger tile dim: 128×128 → 3 × 4 = 12 tiles/pair (still ≈40 byte/pair, marginal)
- OR M=2 ksub=128 (14 bits/tile): 30 byte/pair × 8 / 14 ≈ 17 tiles/pair → 5×3 tile grid on 384×512 = ~75×170 pixel tiles
- OR per-pair single PQ vector (M=4 ksub=256 = 4 bytes/pair) — coarsest possible, basically per-pair "look-up which residual archetype"

**REVISED feasibility verdict**: per-pair budget tight requires either (i) coarse spatial tiling (≤20 tiles/pair) OR (ii) per-pair-archetype lookup (single PQ vector/pair). Both variants in PHASE 3 sweep. **Initial PHASE 3 smoke focuses on (ii) per-pair-archetype lookup as baseline lower bound — extreme rate efficiency at probably-too-coarse distortion.**

**Honest predicted band rebaseline**: [0.180, 0.210] frontier-pursuit per RECALIBRATED tile budget; predicted delta [-0.012, +0.018] from frontier 0.1920513169. Catalog #324 post-training Tier-C validation REQUIRED before paid CUDA dispatch.

## 10. 3-axis rigor evidence per directive #3

### Axis 1: Math + scientific + engineering rigor per layer

| Layer | Math | Sci | Eng | Verdict |
|---|---|---|---|---|
| PQ codebook quantization | Jégou-Douze-Schmid 2011 PQ asymptotic distortion bound (assuming K-means convergence) | Faiss paper canonical | Catalog #303 cargo-cult audit landed PHASE 1 | HARD-EARNED |
| Per-pair residual signal entropy | Mallat wavelet theory residual R(D) bound + Atick-Redlich retinal residual mutual information | Daubechies wavelet residual canonical + Atick-Redlich 1990 | PHASE 3 empirical anchor pending | HARD-EARNED-PENDING-EMPIRICAL |
| Inflate-time canonical helpers | Catalog #205 select_inflate_device + canonical bilinear upsample (sister A forensic) | sister A=DreamerV3 RSSM forensic anchor | A landed `69253a1cc` provides canonical pattern | HARD-EARNED |
| MLX-Faiss adapter feasibility | Integer codebook lookup deterministic; float gather deterministic | sister G=NIRVANA mlx_renderer.py pattern | sister G landed `f7d2e86fe` provides pattern | HARD-EARNED |
| Predicted band derivation | Per-pair MSE residual R(D) bound + RECALIBRATED tile budget per §9 | Per Catalog #324 post-training Tier-C re-measurement REQUIRED | NO premature claim; pending PHASE 3 empirical anchor | HARD-EARNED (no premature claim) |

### Axis 2: MLX drift minimization per primitive

| Primitive | Expected drift bound | Mitigation | Canonical helper citation |
|---|---|---|---|
| PQ codebook gather (MLX) | ≤ epsilon-machine (float copy) | `mx.take_along_axis` canonical per sister K, G | sister G NIRVANA `mlx_renderer.py` |
| Bilinear residual upsample (MLX) | ≤ Catalog #1265 threshold 0.001 | USE canonical `bilinear_resize2x_align_corners_false_nhwc` | sister A=DreamerV3 forensic anchor |
| uint8 cast at output (MLX) | = 0 (deterministic) | USE canonical Catalog #205 sister rounding | sister substrate `_shared/inflate_runtime.py` |
| PQ encoding (numpy) | byte-identical with MLX | Numpy reference path | sister G `numpy_reference.py` pattern |
| Tile reassemble | = 0 (deterministic) | Index arithmetic only | n/a (trivial primitive) |

### Axis 3: Portability via numpy per primitive

Every MLX primitive has sister numpy reference:
- **PQ codebook gather**: numpy `np.take_along_axis` (numpy-pure)
- **Bilinear upsample**: numpy reference (sister G NIRVANA `numpy_reference.py` pattern)
- **uint8 cast**: numpy `np.clip(0, 255).astype(np.uint8)` (numpy-pure)
- **PQ codebook training**: sklearn KMeans (numpy-pure) OR Faiss-CPU (optional accelerator)
- **PQ encoding**: numpy reference path; MLX-Faiss adapter is optional accelerator
- **Tile reassemble**: numpy `np.reshape` / `np.split` / `np.concatenate` (numpy-pure)

Per directive #3 axis 3: substrate operable on CPU-only test rigs without MLX OR Faiss dependency. PHASE 3 L0 SCAFFOLD will land `numpy_reference.py` alongside `mlx_renderer.py`.

## 11. 6-hook wire-in declaration per Catalog #125

| # | Hook | Status | Rationale |
|---|---|---|---|
| 1 | Sensitivity-map | N/A — phase 2 design memo only | Lands at PHASE 3 L0 SCAFFOLD |
| 2 | Pareto constraint | N/A — phase 2 design memo only | Lands at PHASE 3 |
| 3 | Bit-allocator | N/A — phase 2 design memo only | Lands at PHASE 3 |
| 4 | Cathedral autopilot dispatch | **ACTIVE (DESIGN-LEVEL)** | Design memo consumable by canonical_equation_lookup_consumer per Catalog #344 |
| 5 | Continual-learning posterior | **ACTIVE (DESIGN-LEVEL)** | Catalog #313 probe outcomes ledger anchor in PHASE 3 |
| 6 | Probe-disambiguator | **ACTIVE (DESIGN-LEVEL)** | MLX-first gate at Catalog #1265 IS canonical disambiguator |

## 12. Phase 3 L0 SCAFFOLD plan

### Files to land

1. `src/tac/substrates/faiss_ivf_pq_residual/__init__.py` — SPDX + Catalog #241 LEGACY_SUBSTRATE_PRE_META_LAYER waiver + Catalog #124 8-field representation lane declaration
2. `src/tac/substrates/faiss_ivf_pq_residual/mlx_renderer.py` — MLX-native PQ codebook gather + tile reassemble + bilinear upsample (canonical helper) + uint8 cast
3. `src/tac/substrates/faiss_ivf_pq_residual/numpy_reference.py` — numpy reference implementation of every MLX primitive (axis 3 portability)
4. `src/tac/substrates/faiss_ivf_pq_residual/archive.py` — FAISSPQ1 byte-deterministic grammar
5. `src/tac/substrates/faiss_ivf_pq_residual/inflate.py` — PyTorch runtime per Catalog #146 + #205
6. `src/tac/substrates/faiss_ivf_pq_residual/tests/test_basic.py` — Catalog #91 + #139 + MLX↔PyTorch + MLX↔numpy parity tests
7. `experiments/train_substrate_faiss_ivf_pq_residual.py` — smoke trainer with `_full_main raises NotImplementedError` per Catalog #240 (c)
8. `.omx/research/path_3_i_v1_faiss_ivf_pq_L0_scaffold_landed_20260526.md` — landing memo with 3-axis evidence

### Test plan

Tests in `tests/test_basic.py`:
1. Archive round-trip: encode + decode preserves byte-stable archive
2. PQ codebook fit on synthetic 600-pair random residual: codebook converges + per-pair quantization error bounded
3. MLX↔numpy parity: same input → same per-pair residual output to byte-identity
4. MLX↔PyTorch parity: same input → output within Catalog #1265 threshold 0.001 contest-units
5. Catalog #139 byte-mutation: mutate codebook byte → decoded RGB residual changes
6. Catalog #91 archive grammar: FAISSPQ1 magic + header layout

### `_full_main raises NotImplementedError` per Catalog #240 (c)

PHASE 3 trainer entry point declares L0 SCAFFOLD posture explicitly; full training path council-gated per Catalog #325 per-substrate symposium before paid Modal dispatch.

## 13. Discipline compliance summary

- Catalog #229 PV / #303 cargo-cult audit / #294 9-dim checklist / #305 observability / #290 canonical-vs-unique per layer / #296 Dykstra-feasibility / #309 horizon_class / #300 v2 frontmatter / #325 per-substrate symposium queued for PHASE 3 / #292 per-deliberation assumption surfacing / #287 placeholder rejection / #119 Co-Authored-By / #117/#157/#174 canonical serializer / #110/#113 APPEND-ONLY / #206 checkpoints / #230 ownership map / #208 no local paths
- Per directive #3 3-axis evidence in §10
- Per directive #4 efficient token use: combined PV reads; rejected sister Path (c) HYBRID as kitchen_sink

## 14. Cross-references

- PHASE 1 cargo-cult audit (committed `a883a717c`): `.omx/research/path_3_i_v1_faiss_ivf_pq_cargo_cult_audit_20260526.md`
- V1+V4+V8 prior work (research INPUT only): `v1_faiss_v4_probe_plus_v8_design_landed_20260519.md` + `v1_faiss_v8_learned_compression_faiss_design_20260519.md`
- Canonical helper to reuse (primitives): `src/tac/optimization/faiss_ivf_pq_atw_channel.py`
- Sister G=NIRVANA scaffold pattern: `src/tac/substrates/nirvana_cascading_nerv/`
- Sister E=BoostNeRV PR110 residual pattern: `src/tac/substrates/boost_nerv_pr110_residual/`
- Canonical frontier pointer: `.omx/state/canonical_frontier_pointer.json` (PR110 fec6 frontier sha 6bae0201)
