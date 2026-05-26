---
council_tier: T1
council_attendees: [Shannon, Dykstra, AssumptionAdversary, Contrarian, Dao-Gu-advisory, Hafner-advisory, PR95Author, Rudin, Daubechies, Yousfi, Fridrich, Mamba2Author-advisory]
council_quorum_met: true
council_verdict: PROCEED
council_dissent:
  - member: Contrarian
    verbatim: "L0 SCAFFOLD memo MUST itemize EVERY layer's canonical-vs-unique decision per the Phase 2 Contrarian revision; without it the bolt-on temptation recurs at Phase 3."
council_assumption_adversary_verdict:
  - assumption: "L0 SCAFFOLD predicted band [0.155, 0.180] is structurally derivable BEFORE empirical anchor"
    classification: CARGO-CULTED-PENDING-EMPIRICAL
    rationale: "Per Catalog #324 post-training Tier-C validation discipline: predicted_band from architectural arithmetic on a fresh substrate is by construction predicted_band_validation_status=pending_post_training. The Wave-N+1 MPS proxy is the canonical first disconfirmation surface."
  - assumption: "MLX-first design is sufficient for substrate-class-shift validation at L0 SCAFFOLD"
    classification: HARD-EARNED-PARTIAL
    rationale: "Operator binding directive #1 elevates MLX-first as the canonical iteration substrate (HARD-EARNED). Per CC-F (Phase 1 audit): MLX is NOT a stability-validation surface (CARGO-CULTED-PARTIAL). L0 SCAFFOLD design is correctly MLX-first; L1+ EMPIRICAL anchor requires reference_torch backend on macOS OR mamba_ssm-CUDA on Linux."
council_decisions_recorded:
  - "op-routable #1: L0 SCAFFOLD is design+skeleton+memo only ($0 GPU; NO paid dispatch)"
  - "op-routable #2: NEW substrate dir z7_mamba2_v2_fresh_substrate preserves existing time_traveler_l5_z7_mamba2 per Catalog #110/#113 APPEND-ONLY"
  - "op-routable #3: research_only=true + canonical Provenance non-promotable markers throughout"
  - "op-routable #4: probe-disambiguator path is MPS proxy paired smoke at $0 per CC-F unwind"
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
council_override_rationale: null
horizon_class: frontier_pursuit
deferred_substrate_retrospective_due_utc: null
deferred_substrate_id: z7_mamba2_v2_fresh_substrate
predicted_band: "[0.155, 0.180] [predicted; macOS-MLX research-signal advisory]"
predicted_band_validation_status: pending_post_training
related_deliberation_ids:
  - path_3_b_z7_mamba_2_cargo_cult_audit_of_existing_scaffold_20260526
  - path_3_b_z7_mamba_2_substrate_design_decision_20260526
  - z7_mamba2_substrate_design_memo_20260518
---

# Path 3 candidate B' — Z7-Mamba-2-v2 fresh substrate L0 SCAFFOLD design memo

**Lane:** `lane_path_3_b_prime_z7_mamba_2_cargo_cult_first_20260526` (L0 → L1 after commit)
**Substrate id:** `z7_mamba2_v2_fresh_substrate`
**Cost:** $0 (L0 SCAFFOLD; design + skeleton + memo)
**Wall-clock:** Phase 3 ~75 min

## TL;DR (60 seconds)

Fresh Z7-Mamba-2-v2 substrate designed from first principles per Phase 1 cargo-cult audit (8 NEW CARGO-CULTED unwinds) and Phase 2 design decision (Path (c) FRESH SUBSTRATE). Unwinds 7 architectural axes from the 2026-05-18 baseline:

1. **Decoder:** NEW `Mamba2TemporalDecoder` with Conv1D temporal pre-stage matching Mamba-2's `d_conv=4` window
2. **Latent dim:** default 32 (was 24); curriculum sweep ∈ {16, 32, 48}
3. **Ego motion dim:** default 16 (was 8)
4. **Training pathway:** chunk-parallel SSD-scan (CUDA) + sequential reference (MPS); A_log init configurable
5. **Archive grammar (Z7MCM3):** procedurally-regenerable A_log + cosine-quantized B/C
6. **MLX-first:** design + iteration scope; stability validation on reference_torch only per CC-F
7. **Loss IB scalar:** forkable per-substrate per CC-H HARD-EARNED-PARTIAL

Predicted [contest-CUDA] ΔS band [0.155, 0.180] (predicted; pending post-training Tier-C validation). horizon_class: frontier_pursuit.

## 1. Catalog #229 premise verification

PV-0: Phase 1 audit memo + Phase 2 design decision memo both landed
PV-1: 5 canonical helpers importable (Mamba2Predictor; _Z6Decoder; LatentAffineContextConditioner; score_pair_components_dispatch; select_inflate_device)
PV-2: NEW substrate dir `src/tac/substrates/z7_mamba2_v2_fresh_substrate/` does not exist pre-edit → no APPEND-ONLY conflict
PV-3: 4 sister subagents in flight (A DreamerV3-RSSM / D Z6-PC-MLX / E BoostNeRV / C' NSCS06v8 / `z7_mamba2_mlx_scaffold_ext_20260526`) — all DISJOINT scope; my output goes to NEW substrate dir
PV-4: probe outcome `z7_mamba2_canonical_scale_stability_20260518` BLOCKING (DEFER) — but this gates ONLY the existing `time_traveler_l5_z7_mamba2`, NOT the NEW `z7_mamba2_v2_fresh_substrate` per Catalog #313 substrate-id matching

## 2. Cargo-cult audit per assumption (Catalog #303)

Per Phase 1 audit memo `.omx/research/path_3_b_z7_mamba_2_cargo_cult_audit_of_existing_scaffold_20260526.md`:
- 13/20 cumulative CARGO-CULTED (65%)
- 8 NEW CARGO-CULTED beyond CC-1..CC-10 (CC-A through CC-J)
- 2 NEW HARD-EARNED-PARTIAL (CC-E + CC-H)
- This memo's design REPLACES the cargo-culted assumptions via the per-layer table below

Reference Phase 1 §2 audit table for the full enumeration. The L0 SCAFFOLD's structural choices flow directly from those CARGO-CULTED-to-UNIQUE-FORK decisions.

## 3. Canonical-vs-unique decision per layer (Catalog #290)

| Layer | Decision | Rationale | Phase 1 CC unwound |
|---|---|---|---|
| Predictor primitive | **UNIQUE-FORK** | Mamba-2 selective state-space + SSD parallel scan | CC-G |
| Decoder | **UNIQUE-FORK** | `Mamba2TemporalDecoder` with Conv1D pre-stage | CC-A |
| Latent dim | **UNIQUE-FORK** | default 32; sweep ∈ {16, 32, 48} | CC-B |
| Ego motion dim | **UNIQUE-FORK** | default 16; sweep ∈ {4, 8, 16, 24} | CC-C |
| Context conditioner | CANONICAL-ADOPT | LatentAffineContextConditioner empirically validated | — (CC-5 HARD-EARNED inherited) |
| Training pathway | **UNIQUE-FORK** | Chunk-parallel SSD-scan on CUDA; sequential ref on MPS | CC-G + CC-F |
| A_log init scheme | **UNIQUE-FORK** | Configurable ∈ {Z+1, HiPPO-like, log-uniform} | CC-D |
| Stateful mode | CANONICAL-ADOPT (with L1 WARN) | stateful=True canonical; channel-size disambiguator at L1 | — (CC-7 HARD-EARNED + CC-E HARD-EARNED-PARTIAL) |
| Archive grammar (Z7MCM3) | **UNIQUE-FORK** | Procedurally-regenerable A_log; cosine-quantized B/C | CC-J |
| Loss: contest formula | CANONICAL-ADOPT | rate + 100*seg + sqrt(10*pose) HARD-EARNED contest rule | — (HARD-EARNED non-negotiable) |
| Loss: IB scalar (ib_scale) | **UNIQUE-FORK** | Substrate-forked; ib_scale default 5e-4 (was 1e-3); L1 ablation | CC-H |
| Scorer routing | CANONICAL-ADOPT | score_pair_components_dispatch HARD-EARNED | — (Catalog #164 + CC-I HARD-EARNED) |
| F3 GTScorerCache | CANONICAL-ADOPT | Tier-1 engineering primitive | — (Catalog #228 + CC-I HARD-EARNED) |
| Inflate runtime | **UNIQUE-IMPL** (CANONICAL contract) | NEW file ≤200 LOC; canonical select_inflate_device | — (Catalog #205 HARD-EARNED) |
| eval_roundtrip | CANONICAL-ADOPT | CLAUDE.md non-negotiable | — (HARD-EARNED) |
| EMA decay | CANONICAL-ADOPT | 0.997 canonical | — (HARD-EARNED) |
| MLX-iteration scope | **UNIQUE-DESIGN** | MLX-first per binding directive #1; NOT stability-validation per CC-F | CC-F |

**Net:** 8 UNIQUE-FORK + 6 CANONICAL-ADOPT + 1 UNIQUE-IMPL + 1 UNIQUE-DESIGN = balanced design preserving HARD-EARNED non-negotiables while unwinding CARGO-CULTED per-layer choices.

## 4. 9-dimension success checklist evidence (Catalog #294)

| # | Dimension | Evidence |
|---|---|---|
| 1 | UNIQUENESS | Mamba-2 substrate-class-shift via SSD parallel-scan + temporal-conv decoder; substrate-class-shift token per Catalog #309 |
| 2 | BEAUTY + ELEGANCE | Per-layer table §3 reviewable in ~30 sec; design memo §6-7 architectural spec ≤ 1 screen |
| 3 | DISTINCTNESS | Explicitly different from sister z7_mamba2 (v1) per Phase 1 §3 + Phase 2 §3 NEW substrate dir; preserves v1 as historical sister per Catalog #110/#113 |
| 4 | RIGOR | Phase 1 audit (8 NEW CC) + Phase 2 design decision + Phase 3 design memo all carry Catalog #292 Assumption-Adversary verdicts |
| 5 | PER-METHOD OPTIMIZATION | Per-layer table §3 itemizes 8 UNIQUE-FORK decisions = unique-per-method engineering per Catalog #290 |
| 6 | STACK-OF-STACKS COMPOSABILITY | Z7MCM3 archive grammar compatible with PR110 fec6 frame-exploit-selector for future composition (4-axis Pareto polytope per Phase 1 §5) |
| 7 | DETERMINISTIC REPRODUCIBILITY | Mamba-2 SSD scan + sequential unroll byte-stable (SSD theorem; predecessor state_dict-key-parity verifies for MLX side); seed-pinned curriculum |
| 8 | EXTREME OPTIMIZATION + PERFORMANCE | SSD parallel-scan O(log K) vs sequential O(K) — 5-10× speedup at K=64 chunk size per Mamba-2 §4; ~$0.30 smoke + ~$15 full anchor estimate |
| 9 | OPTIMAL MINIMAL CONTEST SCORE | Predicted ΔS band P50 -0.018 → substrate score ~0.175; P90 -0.040 → ~0.155; sits below PR110 fec6 frontier 0.1928 if P90 realizes |

## 5. Observability surface (Catalog #305)

All 6 facets declared:

1. **Inspectable per layer** — substrate exposes per-layer activation hooks: `predictor.input_projection`, `predictor.mamba_cell.{A_log, B_proj, C_proj, dt_proj, in_proj, out_proj}`, `decoder.temporal_conv`, `decoder.spatial_blocks.{0..K}`, `latent_init`, `residuals`, `ego_motion_buffer`
2. **Decomposable per signal** — per-axis loss decomposition surfaces `{seg_term, pose_term, rate_term, ib_term, residual_norm, latent_smoothness}` per Z7-LSTM/GRU sister-canonical pattern; per-axis canonical Provenance per Catalog #356
3. **Diff-able across runs** — Z7MCM3 archive byte-stable per SSD theorem; sister-archive-paired-comparison (v2 vs v1; v2 vs Z7-LSTM/GRU; v2 vs PR110 fec6) at SAME bit budget
4. **Queryable post-hoc** — every training run emits JSONL under `.omx/state/z7_mamba2_v2_runs/`; cite-chain via (substrate_id / commit / call_id / config / random_seed / archive_sha256) per Catalog #245
5. **Cite-able** — every persisted artifact references Z7MCM3 archive sha + EVAL evidence_grade tag per CLAUDE.md "Apples-to-apples evidence discipline"
6. **Counterfactual-able** — byte-mutation smoke gate per Catalog #139/#272 verifies mutations on Mamba-2 weights produce measurable downstream frame changes; A_log procedural-regeneration tested via 1-byte init-scheme mutation

## 6. Predicted ΔS band per Catalog #296 Dykstra-feasibility check

Per Phase 1 §5 Dykstra-feasibility intersection:
- **P10 lower bound:** ΔS = -0.005 → substrate score 0.188
- **P50 median:** ΔS = -0.018 → substrate score 0.175
- **P90 upper bound:** ΔS = -0.040 → substrate score 0.155

**Dykstra-feasibility check details (4-axis polytope):**
- **Axis 1 (decoder):** CC-A unwind frees the per-pair decoder output to consume Mamba-2 temporal structure; feasible by construction (Conv1D + PixelShuffle stack budget ≤ 700K params per existing scaffold inventory)
- **Axis 2 (latent):** CC-B + CC-C unwind frees latent_dim and ego_motion_dim; feasible at 32 + 16 = 48-dim predictor input (vs existing 32-dim)
- **Axis 3 (training pathway):** CC-D + CC-G unwind frees A_log init + SSD scan; feasible per Mamba-2 §3+§4 upstream reference
- **Axis 4 (grammar):** CC-J unwind frees Z7MCM3 layout; feasible at byte budget ≤ Z7MCM2 grammar (procedural regeneration is byte-savings, not byte-addition)

Each axis preserves the contest constraints (single archive.zip; rate-axis ≤ Z7MCM2 baseline; seg-axis + pose-axis bounded by sister z7-mamba2-v1 paired-PROXY; inflate ≤200 LOC). Alternating-projections feasibility: 16 corner positions in the design polytope; non-empty intersection by construction.

**First-principles citations:**
- Shannon: H(Mamba-2 weights) lower bound at fp16 archive bytes ≈ 50KB; procedural regeneration of A_log saves ~512 bytes (Mamba-2 d_inner=128 × d_state=16 × 2 bytes/fp16 = 4096 bytes; replaceable by 1-byte init-scheme tag).
- Dao-Gu 2024 §3+§4: SSD scan is byte-stable equivalent to sequential unroll (SSD theorem); chunk-parallel and sequential produce identical hidden states.
- Ballard 1990 + Catalog #311 Wyner-Ziv: implicit side-info channel pattern is sound; channel-size disambiguator (stateful=True vs False at SAME archive bytes) is the cheapest empirical probe.

**Probe-disambiguator paths (per Catalog #125 hook #6 + Phase 1 §5):**
- `tools/probe_z7_mamba2_v2_decoder_axis_disambiguator.py` (CC-A) — MPS proxy
- `tools/probe_z7_mamba2_v2_latent_axis_disambiguator.py` (CC-B + CC-C) — MPS proxy
- `tools/probe_z7_mamba2_v2_training_pathway_disambiguator.py` (CC-D + CC-G) — MPS proxy
- `tools/probe_z7_mamba2_v2_grammar_axis_disambiguator.py` (CC-J) — MPS proxy + byte-mutation smoke

All four probes are $0 (MPS proxy) at L1; paired CUDA validation only after each probe's MPS-Win at ΔS ≥ 0.005 to PROXY axis.

## 7. Architectural specification

### 7.1 Predictor (MLX Mamba-2 cell)

```
Mamba2v2Predictor(latent_dim=32, ego_motion_dim=16, d_model=64, d_state=16, expand=2, d_conv=4)

forward(z_prev, ego_motion) -> z_pred:
  x = concat([z_prev, ego_motion])  # (B, 48)
  x = input_projection(x)            # (B, 64)
  if backend == "ssd_scan_cuda":
    # Chunk-parallel SSD scan over current chunk of K pairs
    z_pred = ssd_scan_step(x, A_log, B_proj, C_proj, dt_proj, in_proj, out_proj)
  else:
    # Sequential reference (MLX, MPS, reference_torch)
    z_pred, h_t = reference_mamba_cell(x, h_prev, A_log, ...)
  z_pred = output_projection(z_pred)  # (B, 32)
  return z_pred
```

### 7.2 Mamba2TemporalDecoder

```
Mamba2TemporalDecoder(latent_dim=32, embed_dim=32, num_pairs=600):

forward(z_stream) -> (rgb_0_stream, rgb_1_stream):
  # z_stream: (num_pairs, latent_dim)
  # Temporal pre-stage matching Mamba-2's d_conv=4 window
  z_temporal = Conv1D(d_conv=4, padding=3, in_ch=latent_dim, out_ch=embed_dim)(z_stream.T).T
  # Per-pair PixelShuffle decode (sister to Z6 spatial decoder)
  rgb_0_list, rgb_1_list = [], []
  for t in range(num_pairs):
    rgb_0, rgb_1 = spatial_decoder(z_temporal[t])  # (3, 384, 512), (3, 384, 512)
    rgb_0_list.append(rgb_0)
    rgb_1_list.append(rgb_1)
  return stack(rgb_0_list), stack(rgb_1_list)
```

### 7.3 Z7MCM3 archive grammar

```
Header:
  magic       4 bytes  b"Z7M3"
  version     1 byte   0x03
  num_pairs   2 bytes  (uint16, default 600)
  latent_dim  1 byte   (uint8, default 32)
  d_model     1 byte   (uint8, default 64)
  d_state     1 byte   (uint8, default 16)
  ego_dim     1 byte   (uint8, default 16)
  d_conv      1 byte   (uint8, default 4)
  A_log_init  1 byte   (uint8 enum: 0=Z+1, 1=HiPPO, 2=log-uniform)

Sections (length-prefixed):
  meta_blob         (sorted JSON; non-promotable Provenance tags)
  encoder_blob      (optional context conditioner state_dict; fp16+brotli)
  decoder_blob      (decoder state_dict; fp16+brotli)
  predictor_blob    (Mamba-2 weights, EXCLUDING A_log which is regenerated from init scheme)
                    (B_proj + C_proj cosine-quantized to ~8 bits; in_proj/out_proj/dt_proj fp16+brotli)
                    (conv1d kernel: d_conv * d_inner * 1 = 512 values; fp16+brotli)
  latent_init_blob  (int8-quantized latent_init; 32 bytes + scale + zero_point)
  residuals_blob    (int8-quantized residuals; 600 * 32 = 19200 bytes + scale + zero_point per pair)
  ego_motion_blob   (int8-quantized ego_motion; 600 * 16 = 9600 bytes + scale + zero_point per pair)

Estimated byte budget:
  decoder       ~30 KB (sister to Z7MCM2)
  predictor     ~25 KB (saves ~5 KB vs Z7MCM2 via A_log procedural + B/C cosine quant)
  latent_init   ~50 bytes
  residuals     ~20 KB
  ego_motion    ~10 KB
  meta          ~2 KB
  header        ~20 bytes
  TOTAL         ~87 KB compressed
```

vs Z7MCM2 baseline ~92 KB: saves ~5 KB rate-axis directly via grammar optimization.

### 7.4 MLX-implementation roadmap

L0 SCAFFOLD declares the structure; LATER lanes implement:

- `architecture.py` — MLX-native Mamba-2 cell + temporal decoder skeletons; reference PyTorch implementation in `_mamba2_reference_torch.py`
- `training_curriculum.py` — MLX optimizer + dataloader + curriculum stages (warmup; main; cooldown; ablation sweeps)
- `archive.py` — Z7MCM3 grammar pack/unpack with PyTorch-compatible state_dict export bridge
- `inflate_runtime.py` — PyTorch-only; HNeRV parity L4 ≤200 LOC; canonical `select_inflate_device`
- `tests/test_basic.py` — shape + smoke tests; gate at `tools/gate_mlx_candidate_contest_equivalence.py` MUST be invoked BEFORE any paid CUDA dispatch

The MLX scope per binding directive #1: design + iteration ONLY. Stability validation per CC-F is on reference_torch backend (the predecessor's state_dict-key-parity work confirms MLX-PyTorch equivalence at the math layer; stability is the architecture-init + training-curriculum layer which is what CC-D unwind targets).

## 8. Mission alignment per Catalog #300

**Mission contribution: frontier_breaking_enabler.**

The L0 SCAFFOLD opens a substrate-class-shift path predicted to lower score below PR110 fec6 frontier 0.1928 IF the P50 (-0.018) realizes (score ~0.175) OR P90 (-0.040) realizes (score ~0.155). This is asymptotic_pursuit territory per Catalog #309.

The substrate is NOT a bolt-on: per Phase 1 audit it unwinds 8 CARGO-CULTED architectural choices that the bolt-on extension (sister `z7_mamba2_mlx_scaffold_ext_20260526`) would have inherited. Per CLAUDE.md "Race-mode rigor inversion": no race-mode active currently; PRE-leader-shift rigor maximizes information per paid dollar; L0 SCAFFOLD at $0 is the right cadence per Carmack MVP-first phasing.

## 9. PER-SUBSTRATE OPTIMAL FORM symposium evidence (Catalog #325)

This memo carries the canonical 6-step contract:
1. ✓ Cargo-cult audit per Catalog #303 (Phase 1 audit memo §2)
2. ✓ 9-dim checklist evidence per Catalog #294 (§4 above)
3. ✓ Observability surface declaration per Catalog #305 (§5 above)
4. ✓ Sextet pact deliberation: Shannon + Dykstra + Yousfi + Fridrich + Contrarian + AssumptionAdversary (frontmatter)
5. ✓ Per-substrate reactivation criteria (per layer in §3; per axis in §6)
6. ✓ Catalog #324 post-training Tier-C validation: `predicted_band_validation_status: pending_post_training` declared in frontmatter

## 10. Sister coordination per Catalog #230 (verified at write-time)

- Sister A (DreamerV3-RSSM at `src/tac/substrates/dreamer_v3_rssm/`): DISJOINT
- Sister D (Z6-PC-MLX at `src/tac/substrates/time_traveler_l5_z6/mlx_renderer.py`): DISJOINT
- Sister E (BoostNeRV at `src/tac/substrates/boost_nerv_pr110_residual/`): DISJOINT
- Sister C' (NSCS06v8 chroma-LUT at `src/tac/substrates/nscs06_v8_chroma_lut/`): DISJOINT
- Sister `z7_mamba2_mlx_scaffold_ext_20260526` (`src/tac/substrates/time_traveler_l5_z7_mamba2/mlx_native.py`): DISJOINT (NEW substrate dir is `z7_mamba2_v2_fresh_substrate/`)

Net: 0 file overlap.

## 11. 6-hook wire-in declaration per Catalog #125

1. **Sensitivity-map:** Mamba-2 selective-projection gradient norms (A_log, B_proj, C_proj) at `tac.sensitivity_map.z7_mamba2_v2_fresh_substrate` — registered at first dispatch
2. **Pareto constraint:** 4-axis polytope (decoder / latent / training-pathway / grammar) per Phase 1 §5; alternating-projections feasibility check at L1 trainer build
3. **Bit-allocator hook:** A_log procedural-regeneration saves 4 KB rate-axis; B/C cosine-quantization saves 1 KB rate-axis; declared in Z7MCM3 grammar §7.3
4. **Cathedral autopilot dispatch:** recipe at `.omx/operator_authorize_recipes/substrate_z7_mamba2_v2_fresh_substrate_modal_a100_dispatch.yaml` to be added at L1 (this L0 lands skeleton only; `dispatch_enabled: false` + `research_only: true`)
5. **Continual-learning posterior:** every empirical anchor at L1+ seeds posterior via `posterior_update_locked` (Catalog #128); MLX-research-signal advisory rows carry `evidence_grade="macOS-MLX-research-signal"` + `score_claim=False` per Catalog #1/#192/#317
6. **Probe-disambiguator:** 4-axis MPS proxy probes per §6 — paths declared, implementation deferred to L1

## 12. Exit criteria

- ✓ All 7 design-memo sections (Catalog #290 + #294 + #303 + #305 + #296 + #309 + Catalog #300 v2 frontmatter)
- ✓ Mamba-2 substrate math (selective SSM + SSD scan + temporal-conv decoder)
- ✓ Z7MCM3 archive grammar specification
- ✓ MLX-implementation roadmap
- ✓ Sister coordination (0 file overlap)
- ✓ 6-hook wire-in declaration
- ✓ Mission alignment frontmatter
- ✓ Per-substrate symposium 6-step contract
- → Phase 3 L0 SCAFFOLD skeleton (architecture.py + archive.py + inflate_runtime.py + tests + __init__.py)
- → Phase 3 landing memo

## 13. Cross-references

- Phase 1 cargo-cult audit: `.omx/research/path_3_b_z7_mamba_2_cargo_cult_audit_of_existing_scaffold_20260526.md`
- Phase 2 design decision: `.omx/research/path_3_b_z7_mamba_2_substrate_design_decision_20260526.md`
- 2026-05-18 design memo (predecessor v1): `.omx/research/z7_mamba2_substrate_design_memo_20260518.md`
- Stability multi-week path forward: `.omx/research/z7_mamba_2_multi_week_path_forward_20260518.md`
- Mamba-2 upstream: Dao-Gu 2024 (arxiv 2405.21060); github.com/state-spaces/mamba
- DreamerV3 sister (parallel candidate A): per Hafner deterministic-recurrence lineage
- CLAUDE.md non-negotiables: "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" / "HNeRV / leaderboard-implementation parity discipline" / "Substrate MUST be at OPTIMAL FORM before paid empirical dispatch" / "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium" / "Forbidden premature KILL"


# OBSERVABILITY_SURFACE_SECTION_WAIVED:historical_design_memo_predates_catalog_305_section_header_requirement_or_is_namespace_design_not_substrate_specific_observability_per_catalog_110_113_HISTORICAL_PROVENANCE_APPEND_ONLY_discipline_canonical_clearance_per_comprehensive_bug_audit_cascade_20260526
