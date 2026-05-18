---
review_kind: asymptotic_stacking_plus_local_utilization_audit
review_id: asymptotic_stacking_plus_local_max_utilization_audit_20260518
review_date: "2026-05-18"
lane_id: lane_local_m5_max_utilization_audit_20260518
operator_directive: "(A) asymptotic stacking of boltons + all optimizations without costly retraining on top of PR95/101/107/106 etc (B) what can be run using local CPU and MPS (if anything for MPS)? are we maximizing use of local M5 Max + 128 GB unified?"
score_claim: false
promotion_eligible: false
provider_spend: false
research_only: true
horizon_class: frontier_breaking
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
related_deliberation_ids:
  - comprehensive_research_wave_20260518
  - council_per_substrate_symposium_tt5l_foveation_lapose_20260517
  - council_per_substrate_symposium_z7_lstm_predictive_coding_20260517
  - council_per_substrate_symposium_atw_v2_reactivation_20260518
  - council_c6_ibps_post_empirical_reactivation_symposium_first_per_substrate_optimal_form_20260518
  - council_per_substrate_symposium_dp1_deep_dive_20260517
  - council_per_substrate_symposium_lane_17_imp_20260517
  - council_per_substrate_symposium_pr106_05_06_reformulated_20260517
  - council_per_substrate_symposium_stc_3a_sidecar_a1_residual_20260517
  - council_per_substrate_symposium_nscs06_v8_path_b_20260517
  - pre_rigor_kill_defer_falsified_inventory_20260517
deferred_substrate_id: null
deferred_substrate_retrospective_due_utc: null
---

# Asymptotic stacking + local-M5-Max-utilization audit — 2026-05-18

**Lane**: `lane_local_m5_max_utilization_audit_20260518` (L0 → L1 at memo landing)
**Subagent**: `c66f93c4af554193`
**Scope**: 2-part operator directive — (A) "asymptotic stacking of bolt-ons + all optimizations possible without costly retraining on top of PR 95/101/107/106 etc" + (B) "what can be run using local CPU and MPS (if anything for MPS)? are we maximizing use of local M5 Max and 128 GB unified?"
**Spend**: $0 GPU, ~3-4h editor. No commits / no dispatches / no substrate code modifications. READ-ONLY everywhere except this memo + checkpoint.
**Live frontier per Catalog #316**: `0.19205 [contest-CPU]` (`pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean`; archive `6bae0201`) / `0.20533 [contest-CUDA]` (`pr106_format0d_latent_score_table`; archive `9cb989cef519`).

The two directives are deeply overlapping: both ask "what cheap zero-retraining bolt-on or local-only work can move the frontier without paid GPU spend". Treated as unified deliverable per parent prompt.

---

## 0. Executive summary

### TOP-5 zero-retraining bolt-on stacking compositions

| # | Composition | Predicted ΔS band | Cost | Cargo-cult-monotonicity risk per #864 |
|---|---|---|---|---|
| 1 | **PR101 fec6 + DP1 driving prior sidecar** (orthogonal: codec-bit-allocator + OOD pretrained prior) | `[-0.012, -0.004]` ⇒ `[0.180, 0.188]` [contest-CPU] | $0 local CPU compose + $10-15 paired-CUDA verify | LOW (DP1 is OOD pretrain so doesn't re-bake fec6 cargo-cults) |
| 2 | **PR106 format0d + Mamba-2 contextual entropy bolt-on** (orthogonal: latent score-table + neural entropy model) | `[-0.008, -0.003]` ⇒ `[0.197, 0.202]` [contest-CUDA] | $0 Mamba inference on MPS + $10-15 paired-CUDA verify | LOW (Mamba entropy model is post-hoc; doesn't perturb format0d weights) |
| 3 | **PR101 fec6 + PR103 arithmetic codec port** (orthogonal: payload selector + entropy stage swap) | `[-0.005, -0.001]` ⇒ `[0.187, 0.191]` [contest-CPU] | $0 local CPU bit-exact compose + $5 paired-CUDA verify | LOW (PR103 already at L2 in registry; no retraining; pure codec swap) |
| 4 | **PR101 fec6 + lane_17_imp Frankle LTH 50% sparsity prune** (post-training pruning of renderer.bin; orthogonal axes) | `[-0.015, -0.005]` ⇒ `[0.177, 0.187]` [contest-CPU] | $1-2 Vast.ai 4090 IMP cycle 0 (per pre-rigor symposium #856) + $0 local CPU rebuild | MED — LTH may regress pose component if pruning targets PoseNet-sensitive weights; #864 anti-pattern applies. Mitigation: per-tensor sensitivity mask. |
| 5 | **PR101 fec6 + STC syndrome-trellis pose-residual sidecar** (orthogonal: codec + payload-side near-Shannon sidecar) | `[-0.020, -0.005]` ⇒ `[0.172, 0.187]` [contest-CPU] | $0.20 local CPU probe per pre-rigor symposium #857 + $5 paired-CUDA verify | LOW (STC operates on pose-residual residual; orthogonal axis to fec6) |

### TOP-5 UNUSED local-M5-Max work items (currently leaving signal on the table)

| # | Local work item | Estimated time-to-value | Why unused now | Predicted impact |
|---|---|---|---|---|
| 1 | **Per-pair master gradient fp64 computation** for ALL PR archives (PR101 + PR106 + fec6 + format0d) — ~176MB per archive; trivial on 128GB unified; produces canonical Lagrangian-dual-derived priority ordering for every bolt-on (per Catalog #319 v2 cascade) | 2-4h editor + 1-2h local fp64 compute per archive | Master gradient only computed for fec6; PR106 + PR101_lc_v2 NOT yet | Drives Catalog #319 cascade properly; unblocks v2 reweight cascade for HIGH_PAIR_INVARIANT class on every archive |
| 2 | **CPU pre-probes for 5+ pending pre-rigor reactivations** (stc clean source $0.20 / PR106 #05+#06 R1+R2 $0 / TT5L MI / DP1 audit) — every one runs in CPU minutes on M5 Max | 4-8h editor + 8-24h local compute | No canonical `tac.optimization.local_cpu_pre_probe_runner` helper exists yet; each pre-probe is hand-rolled | Cheapest possible disambiguator signal per Catalog #313; defers paid GPU until probe verdict |
| 3 | **Faiss-IVF-PQ codebook construction** for ATW V2-1 channel candidate (per-region 16×16 SegNet softmax histogram product-quantized to ≤2KB) — CPU-native; runs in seconds on 128GB | 4-8h editor (single sister subagent landed scaffold already) + 1h compute | ATW V2-1 channel choice depends on Z6 4c outcome; Faiss-IVF-PQ build is the structural prerequisite NOT yet landed | Unblocks ATW V2-1 dispatch wave (per ATW V2 symposium Revisions #1+#2+#3) |
| 4 | **macOS-CPU advisory proxy autopilot ranking** for top-20 candidate queue — `tools/score_macos_cpu_advisory_proxy.py` ALREADY EXISTS but is NOT being run on every candidate; PR107 6e-6 match shows it's HIGH-FIDELITY | 2-4h editor (operator-facing batch runner) + ~5 min/candidate ≈ 100 min total | No batch runner wraps it; called only ad-hoc | Free re-ranking signal for autopilot per Catalog #192 + #317 |
| 5 | **MPS-research-signal sweep** for hyperparameter curve shapes (β-sweep for C6 IBPS / depth-sweep for Z6 / Mamba block-size sweep for Z7) — `tac.optimization.mps_research_signal` ALREADY EXISTS but underused; M5 Max with 128GB unified can run ALL these in <12 hours overnight | 4-8h editor (canonical sweep harness wrapping mps_research_signal) + 8-12h local compute | Each substrate's hyperparameter selection is currently single-pick at design time; sweeps deferred to GPU dispatch | Eliminates the GPU-dispatch-as-hyperparameter-search anti-pattern; cuts dispatch cost ~3-5× via informed prior |

### Recommended HIGH-VALUE NEW LOCAL primitives to canonicalize

| Helper | Canonical location | What it provides | Wires into |
|---|---|---|---|
| `tac.optimization.local_cpu_pre_probe_runner` (NEW) | `src/tac/optimization/local_cpu_pre_probe_runner.py` | Common harness for the 5+ pending CPU pre-probes (stc / PR106 #05+#06 / TT5L MI / DP1 audit / future) with canonical evidence_grade=`local-cpu-pre-probe`, archive_sha provenance, JSONL append to `.omx/state/local_cpu_pre_probes.jsonl` per Catalog #131/#138 | Cathedral autopilot ranker (Hook #4); probe outcomes ledger (Catalog #313); Catalog #323 canonical Provenance umbrella |
| `tac.optimization.macos_cpu_proxy_batch_ranker` (NEW) | `src/tac/optimization/macos_cpu_proxy_batch_ranker.py` | Wraps `score_macos_cpu_advisory_proxy.py` over top-N candidate queue; emits ranked manifest with macOS-CPU advisory grade per Catalog #192 | Autopilot loop; one-arg-local-MPS-vs-Modal switch per Catalog #317 |
| `tac.optimization.mps_research_signal_sweep_harness` (NEW) | `src/tac/optimization/mps_research_signal_sweep_harness.py` | Canonical sweep harness wrapping `mps_research_signal` for hyperparameter curve-shape estimation; runs N config × M epochs on MPS overnight; emits curve-fit prior used by paid GPU dispatch | Z6 Wave 2 4c continuation; Z7 GRU pre-dispatch; C6 IBPS β-sweep; ATW V2-1 channel comparison |
| `tac.local_dispatch.faiss_ivf_pq_channel_builder` (NEW) | `src/tac/local_dispatch/faiss_ivf_pq_channel_builder.py` | CPU-native Faiss-IVF-PQ codebook builder for ATW V2-1 + sister substrate channel construction; ≤2KB shippable budget verification | ATW V2-1 design + DeliverabilityProof per Catalog #319 |
| `tools/run_local_master_gradient_for_archive.py` (NEW) | `tools/run_local_master_gradient_for_archive.py` | Wraps `tools/extract_master_gradient.py --target local-cpu` with sentinel-clean parity + fcntl-locked artifact write to `.omx/state/master_gradient_consumers/`; tool dispatch per Catalog #270 scope clarification (CPU-only) | Catalog #319 v2 cascade; cathedral autopilot reweight |

---

## 1. PR substrate inventory (PR95 / PR101 / PR106 / PR107 + sister)

Per Catalog #316 canonical frontier scan + Catalog #245 modal call-id ledger + `.omx/state/lane_registry.json` (872 lanes; 64 PR-substrate-related; 28 at L2+ promotion-ready).

### 1.1 PR95 HNeRV family (gold-baseline architecture)

- **Anchor**: PR #95 → 2026-04-30 HNeRV gold (0.193 [contest-CUDA]); architecture cloned into pact's `tac.hnerv_decoder_recode` + `tac.hnerv_training_parity_guard` (per Catalog #187)
- **Current best score in pact registry**: `pr101_lc_v2` clone = ~0.193 [contest-CUDA] (matches PR101 GOLD baseline)
- **Archive bytes**: ~178,000 bytes (matches PR102's `afd53348f503`)
- **Bolt-on surfaces (zero-retraining)**:
  - PR101_decoder_storage_order schema swap (registered primitive)
  - PR101_conv4_storage_perms permutation (registered primitive)
  - PR101_decoder_byte_maps sign-encoding (registered primitive; negzig/zig/twos/off/raw_uint8)
  - PR103 arithmetic codec port (L2 lane registered)
  - PR105 packed-state-schema size-sorted (registered primitive)
  - Generic compressors: brotli / lzma / Huffman (registered primitives)

### 1.2 PR101 family (fec6-class frontier holder)

- **Anchor**: 2026-05-15 fec6 selector frontier `0.19205 [contest-CPU]` archive `6bae0201` (per Catalog #316)
- **Architecture**: PR101_lc_v2 (HNeRV gold clone) + Frame_exploit_selector (fec6-class payload selector; per `tac/packet_compiler/pr101_fec7_selector.py`) + fixed Huffman k16 entropy
- **Archive bytes**: ~178,000 bytes (selector_payload ~600-800 bytes; entropy ~177KB)
- **Bolt-on surfaces (zero-retraining)**:
  - FEC7 selector improvement (already-coded primitive registered in `pr101_fec7_selector.py::evaluate_fec7_candidates`)
  - DP1 driving prior sidecar composition (cross-substrate stack; ~700KB DP1 codebook amortized)
  - lane_17_imp Frankle LTH post-training pruning (renderer.bin sparsification)
  - STC syndrome-trellis pose-residual sidecar (per pre-rigor symposium #857)
  - Mamba-2 / DCVC-FM contextual entropy model bolt-on (per research wave §4.1)
  - PR103 arithmetic codec port (entropy stage swap)
  - Brotli-12 production deployment (canonical primitive swap)

### 1.3 PR106 family (latent score-table frontier holder)

- **Anchor**: 2026-05-15 format0d frontier `0.20533 [contest-CUDA]` archive `9cb989cef519` (per Catalog #316)
- **Architecture**: PR106 latent sidecar (~960-byte latent + score table) + Selfcomp 1.017-bpw block-FP weights + 94K-param SegMap (per PR #56 lineage)
- **Archive bytes**: ~120-130KB (latent <1KB + segmap ~94KB + decoder)
- **Bolt-on surfaces (zero-retraining)**:
  - Format0d score-table refinement (in-class refinement; sub-Pareto per Catalog #322 anti-pattern)
  - Wavelet residual sidecar (L2 lane `lane_wavelet_residual_pr106_sidecar_dispatch_ready`)
  - Cool-Chic residual sidecar (L2 lane `lane_cool_chic_residual_pr106_sidecar_dispatch_ready`)
  - C3 residual sidecar (L2 lane `lane_c3_residual_pr106_sidecar_dispatch_ready`)
  - SIREN residual sidecar (L2 lane `lane_siren_residual_pr106_sidecar_dispatch_ready`)
  - Coord-MLP residual sidecar (L2 lane `lane_coord_mlp_residual_pr106_sidecar_dispatch_ready`)
  - PR101_pr103_grammar swap (L2 lane `lane_pr101_pr103_grammar_on_pr106_r2`)
  - Magic-codec PR106 hybrid (L2 lanes `lane_b1_*_magic_codec`)

### 1.4 PR107 family (apogee/PR107 baseline)

- **Anchor**: PR #107 apogee 2026-04-30 (0.229 [contest-CUDA]; 0.1966 [contest-CPU] per M5 Max ↔ GHA Linux 6e-6 match per CLAUDE.md "Submission auth eval")
- **Architecture**: arch_shrink-x0.4 + Lagrangian; baseline for "FP4 quantization + arch shrink" lineage
- **Bolt-on surfaces (zero-retraining)**:
  - PR107 + GPTQ/AWQ modern PTQ (per pre-rigor inventory #6; lane_apogee_int4 reformulated)
  - PR107 + lane_17_imp LTH composition (orthogonal: pruning vs quantization)

### 1.5 Cross-PR cumulative bolt-on registry

| Bolt-on primitive | Source | Already registered? | Composes cleanly with |
|---|---|---|---|
| `pr101_decoder_storage_order` | `tac.composition.registry` | YES | All PR101/PR95 |
| `pr101_conv4_storage_perms` | `tac.composition.registry` | YES (requires decoder_storage_order) | All PR101/PR95 |
| `pr101_decoder_byte_maps` | `tac.composition.registry` | YES | All PR101/PR95 |
| `sign_encoding_{negzig,zig,twos,off,raw_uint8}` | `tac.composition.registry` | YES | All PR-family payload bytes |
| `pr98_cd1_compact_format` | `tac.composition.registry` | YES (schema elision) | PR-family schema |
| `pr100_schema_driven_decoder` | `tac.composition.registry` | YES | PR-family schema |
| `pr105_packed_state_schema` | `tac.composition.registry` | YES (size-sorted) | PR-family state |
| `magic_codec_dense_streams` | `tac.composition.registry` | YES | Most substrates |
| `brotli` (canonical) | `tac.composition.registry` | YES | All payloads |
| `lzma` | `tac.composition.registry` | YES (universal-density) | All payloads |
| `compressai_factorized_prior` | `tac.composition.registry` (per Catalog #169) | YES | Latent streams |
| `compressai_balle_hyperprior` | `tac.composition.registry` (per Catalog #169) | YES | Latent streams |
| `compressai_cheng2020` | `tac.composition.registry` (per Catalog #169) | YES (anchor+attention) | Latent streams |
| `cooperative_receiver_atick_redlich` | `tac.composition.registry` | YES | Score-aware loss |
| `predictive_coding_rao_ballard` | `tac.composition.registry` | YES | Predictor lanes |
| FEC7 selector enhancement | `pr101_fec7_selector.py` | YES (function-level) | fec6/fec7 selector chain |
| PR103 arithmetic codec | `pr103_arithmetic_coding.py` | YES (codec module) | PR101/PR95 entropy stage |
| Frame-exploit-selector grammar | `cooperative_receiver_grammars.py` | YES | PR101 substrate |
| pyppmd (Catalog #203 hard dep) | `pyproject.toml` | YES (Modal image) | Payload compression |
| constriction (ANS/range/arith) | `pyproject.toml` (Catalog #203) + `tac.codec.charm_range_coder` | YES (Rust+Python) | All entropy stages |
| **Wavelet residual basis** | `lane_wavelet_residual_basis_pr106` L1 + `tac/codec/per_tensor_codecs.py` (?) | YES (lane registered) | PR106 sidecar |
| **Cool-Chic v3 residual** | `lane_cool_chic_residual_pr106_sidecar_dispatch_ready` L2 | YES (lane registered) | PR106 sidecar |
| **C3 residual** | `lane_c3_residual_pr106_sidecar_dispatch_ready` L2 | YES (lane registered) | PR106 sidecar |
| **Hessian-block-FP** (Selfcomp PR #56) | `tac/codec/a6_selfcomp_blockfp_hyperprior_compose.py` | YES (~30KB module) | PR106 weights |
| **dual-layer STC AV1 codec** | `tac/codec/dual_layer_stc_av1_codec.py` | YES (~12KB module) | mask channel |
| **pose-filler STC codec** | `tac/codec/pose_filler_stc_codec.py` | YES (~23KB module) | pose residual |
| **syndrome-trellis codec** | `tac/codec/syndrome_trellis_codec.py` | YES (~16KB module) | residual encoding |
| **Wyner-Ziv layer** | `tac/codec/wyner_ziv_layer.py` | YES (~43KB module) | Frame-0 sidecar |
| **DP1 driving prior composition** | `tac/substrates/pretrained_driving_prior/composition.py` + Catalog #210/#211 | YES (canonical Comma2k19 cached per Catalog #213) | Any base substrate |
| **lane_17_imp LTH** | RETIRED (pre-rigor inventory #1 PROCEED) | NO (needs revival) | All weight-bearing substrates |
| **GPTQ/AWQ PTQ** | NOT YET INTEGRATED (pre-rigor inventory #6) | NO | All weight-bearing substrates |
| **Mamba-2 entropy model** | NOT YET INTEGRATED (research wave §4.1) | NO | Any latent stream |
| **Faiss-IVF-PQ channel** | NOT YET INTEGRATED (research wave §4.2 ATW V2-1) | NO | ATW V2-1 + sister cooperative-receiver |
| **DCVC-FM 2024 contextual entropy** | NOT YET INTEGRATED (research wave §4.4 pre-rigor #5) | NO | PR101 + PR106 latent streams |
| **V-JEPA 2 pretraining** | NOT YET INTEGRATED (research wave §4.4 pre-rigor #3 mae_v) | NO | Any encoder (cross-substrate pretrain) |
| **VGGT pose teacher** | NOT YET INTEGRATED (research wave §1.2 TT5L) | NO | TT5L V2 + DP1 pose anchor |

**Empirical observation**: pact already has ~30+ canonical bolt-on primitives registered. The unused surface is dominated by 2024-2026 bleeding-edge integrations (Mamba-2 / Faiss-IVF-PQ / DCVC-FM / V-JEPA 2 / VGGT) and 4 pre-rigor reactivations (lane_17_imp / lane_stc_clean_source / PR106 #05+#06 / lane_pr101_compressai_balle).

---

## 2. Bolt-on primitive enumeration

This section enumerates the ~50 bolt-on primitives across canonical pact + 2024-2026 bleeding-edge OSS. For each: what it does, which PR substrate(s) it composes with, retraining required (zero/minimal/yes), local-runnable (CPU/MPS/GPU-required), predicted ΔS contribution, orthogonal/antagonistic/saturating per Catalog #322 (#329 proposed), 2024-2026 evidence.

### 2.1 Already-canonical pact primitives (zero-retraining, local CPU)

| # | Primitive | What it does | Composes with | Retraining | Local-runnable | Predicted ΔS | Orthogonal/antagonistic | 2024-2026 evidence |
|---|---|---|---|---|---|---|---|---|
| 1 | `pr101_decoder_storage_order` | Schema-driven byte order for PR101 decoder weights | All PR101/PR95 | ZERO | CPU | `[-0.003, +0.001]` already in PR101 gold | ORTHOGONAL with other PR101 storage primitives | Canonical from PR #101 2026-04-30 |
| 2 | `pr101_conv4_storage_perms` | Permutation of conv4 storage for better compressibility | PR101+#1 (required) | ZERO | CPU | `[-0.005, -0.001]` | ORTHOGONAL with #1 | Canonical from PR #101 |
| 3 | `pr101_decoder_byte_maps` | Sign-encoding strategy for byte maps | PR101+#1+sign_enc | ZERO | CPU | `[-0.003, -0.001]` | ORTHOGONAL with sign_enc choice | Canonical from PR #101 |
| 4 | `sign_encoding_negzig/zig/twos/off/raw_uint8` | 5-choice sign encoding per byte stream | All PR-family payload | ZERO | CPU | `[-0.002, +0.002]` per choice | SATURATING (pick best per stream) | Canonical |
| 5 | `pr98_cd1_compact_format` | Schema elision per PR98 CD1 | PR-family schema | ZERO | CPU | `[-0.002, -0.001]` | SATURATING with #6+#7 | Canonical from PR #98 |
| 6 | `pr100_schema_driven_decoder` | Schema-driven decoder storage per PR100 | PR-family schema | ZERO | CPU | `[-0.003, -0.001]` | SATURATING with #5+#7 | Canonical from PR #100 |
| 7 | `pr105_packed_state_schema` | Packed-state-schema size-sorted per PR105 | PR-family state | ZERO | CPU | `[-0.004, -0.001]` | SATURATING with #5+#6 | Canonical from PR #105 |
| 8 | `magic_codec_dense_streams` | Auto-selector bundle for dense streams | Most substrates | ZERO | CPU | `[-0.002, -0.001]` | ORTHOGONAL with substrate-specific codecs | Canonical pact primitive |
| 9 | `brotli` (level 11+) | Contest baseline universal compressor | All payloads | ZERO | CPU | Baseline | SATURATING with #10 | Brotli-12 2024 production |
| 10 | `lzma` | Universal-density entropy coder | All payloads | ZERO | CPU | `[-0.001, +0.005]` vs brotli | SATURATING with #9 | Standard since 1998 |
| 11 | `compressai_factorized_prior` | Ballé 2018 FactorizedPrior baseline | Latent streams | YES (encoder training) | GPU (training) / CPU (inference) | `[-0.005, +0.001]` | ORTHOGONAL with substrate weights | DCVC-FM CVPR 2024 extension |
| 12 | `compressai_balle_hyperprior` | Ballé 2018 Scale Hyperprior | Latent streams | YES (encoder training) | GPU / CPU inference | `[-0.010, -0.003]` | ORTHOGONAL | DCVC-FM CVPR 2024 |
| 13 | `compressai_cheng2020` | Cheng-2020 anchor+attention | Latent streams | YES | GPU / CPU inference | `[-0.012, -0.005]` | ORTHOGONAL | 2024 ELIC extension |
| 14 | `cooperative_receiver_atick_redlich` | Atick-Redlich score-aware loss | Score-aware substrates | YES (training) | GPU training | `[-0.020, -0.005]` | ORTHOGONAL with predictor | Per Catalog #311 |
| 15 | `predictive_coding_rao_ballard` | Rao-Ballard hierarchy residual | Predictor lanes | YES (training) | GPU training | `[-0.025, -0.008]` | ORTHOGONAL with Atick-Redlich | Per Catalog #312 |

### 2.2 Already-canonical pact codec modules (zero-retraining for inference; mostly CPU)

| # | Primitive | What it does | Composes with | Retraining | Local-runnable | Predicted ΔS | Orthogonal/antagonistic | 2024-2026 evidence |
|---|---|---|---|---|---|---|---|---|
| 16 | FEC7 selector (`pr101_fec7_selector`) | Improved frame-exploit-selector vs fec6 | PR101 fec6 | ZERO (post-training) | CPU | `[-0.003, -0.001]` | ORTHOGONAL with fec6 | Canonical pact |
| 17 | PR103 arithmetic codec | Arithmetic codec port for PR101/PR95 entropy | PR101/PR95 | ZERO | CPU | `[-0.004, -0.001]` | SATURATING with brotli/lzma at entropy stage | Canonical pact L2 |
| 18 | Wavelet residual basis (Mallat/Daubechies) | Wavelet decomposition for residual encoding | PR106 sidecar | ZERO if uses canonical pywavelet; MINIMAL if learned wavelet | CPU | `[-0.005, -0.001]` | ORTHOGONAL with base substrate | Per Catalog #277; PyWavelets `PyWavelets/pywt` |
| 19 | Cool-Chic v2/v3 residual | Per-image neural codec residual | PR106 sidecar | YES (per-image overfit) | GPU training / CPU inference | `[-0.008, -0.002]` | ORTHOGONAL with base substrate | Leguay 2024 Cool-Chic v3 |
| 20 | C3 residual | C3 (Cool-Chic v2) residual learned context | PR106 sidecar | YES (per-image) | GPU training | `[-0.010, -0.003]` | ORTHOGONAL | Kim et al. ICLR 2024 |
| 21 | SIREN residual | SIREN coordinate-MLP residual | PR106 sidecar | YES (per-image) | GPU training | `[-0.005, -0.001]` | SATURATING with NeRV substrate | Sitzmann 2020 |
| 22 | Coord-MLP residual | Generic coordinate-MLP residual | PR106 sidecar | YES (per-image) | GPU training | `[-0.005, -0.001]` | SATURATING with SIREN/NeRV | Subsumed by NeRV family |
| 23 | Hessian-block-FP (Selfcomp PR #56) | 1.017-bpw block-FP weight self-compression | PR106 weights | YES (training) | GPU training | `[-0.015, -0.005]` | ORTHOGONAL with quantization choice | Selfcomp PR #56 canon |
| 24 | dual-layer STC AV1 codec | Dual-layer syndrome-trellis on AV1 mask | mask channel | MINIMAL (codec param sweep) | CPU | `[-0.008, -0.002]` | ORTHOGONAL with base mask codec | Filler-Pevný-Fridrich 2011 |
| 25 | pose-filler STC codec | STC encoding for pose residual | pose residual | MINIMAL | CPU | `[-0.010, -0.003]` | ORTHOGONAL with pose substrate | Per pre-rigor #2 |
| 26 | syndrome-trellis codec (generic) | Generic STC for any residual | residual encoding | MINIMAL | CPU | `[-0.008, -0.002]` | ORTHOGONAL | Filler-Pevný-Fridrich 2011 |
| 27 | Wyner-Ziv layer (`tac.codec.wyner_ziv_layer`) | Frame_0 Wyner-Ziv sidecar | D4 frame_0 substrate | YES (training) | GPU training | `[-0.005, +0.005]` | SATURATING with D4 substrate | D4 substrate at L2 |
| 28 | DP1 driving prior composition | Comma2k19-cached DP1 codebook sidecar | Any base substrate | MINIMAL (codebook construction) | CPU (Faiss-IVF-PQ build) | `[-0.012, -0.004]` | ORTHOGONAL with PR101/PR106 (OOD) | Catalog #210/#211/#213 |
| 29 | constriction (ANS/range/arithmetic) | Rust+Python ANS/range/arithmetic library | All entropy stages | ZERO | CPU | Baseline (Shannon-bound entropy coder) | SATURATING with other entropy coders | Bamler 2022 `bamler-lab/constriction` |
| 30 | pyppmd (PPMd context model) | PPMd context-mixing model for non-numeric streams | Text/schema payloads | ZERO | CPU | `[-0.001, +0.002]` vs brotli on numeric | SATURATING | pyppmd >=1.3 (Catalog #203) |

### 2.3 NEW canonical primitives needed (per research wave §6 op-routables 18-21)

| # | Primitive | What it does | Composes with | Retraining | Local-runnable | Predicted ΔS | Integration cost |
|---|---|---|---|---|---|---|---|
| 31 | **Mamba-2** (`state-spaces/mamba`) | O(N) selective state-space; LSTM/GRU replacement | Z7 GRU predictor + Z8 / any sequential codec | YES (training; small models OK on MPS) | MPS (proxy training) / GPU (production) | `[-0.025, -0.008]` per Z7 reformulation | $0 GPU + ~2 days subagent (research wave op-routable 18) |
| 32 | **Faiss-IVF-PQ** (`facebookresearch/faiss` v1.8+) | Product quantization for shippable side-info | ATW V2-1 channel + DP1 codebook | ZERO (post-training codebook construction) | CPU NATIVE (M5 Max with 128GB ideal) | `[-0.015, -0.005]` per ATW V2-1 redesign | $0 GPU + ~1 day subagent (research wave op-routable 19) |
| 33 | **DCVC-FM 2024 contextual entropy** (`microsoft/DCVC`) | Contextual entropy model with Mamba-style; CVPR 2024 SOTA | PR101 + PR106 latent streams | YES (model training) | GPU training / CPU/MPS inference | `[-0.025, -0.008]` per pre-rigor #5 | $0 GPU + ~3 days subagent (research wave op-routable 20) |
| 34 | **VGGT pretrained encoder** (`facebookresearch/vggt`) | CVPR 2025 Best Paper; pretrained pose/depth/cameras | TT5L V2 + DP1 pose anchor + LAPose teacher | ZERO (frozen encoder; finetune small head) | GPU/MPS inference (128GB ideal) | `[-0.020, -0.008]` per TT5L V2 redesign | $0 GPU + ~3 days subagent (research wave op-routable 21) |
| 35 | **DUSt3R/MASt3R** (`naver/dust3r` + `naver/mast3r`) | 2-frame dense reconstruction; pose teacher | TT5L V2 + DP1 pose anchor | ZERO (inference only) | GPU/MPS inference | `[-0.015, -0.005]` complement to VGGT | $0 GPU + ~2 days subagent |
| 36 | **CompressAI DCVC-FM** (`InterDigitalInc/CompressAI` 2024 extension) | Sister to #33; canonical Ballé hyperprior infrastructure | All latent streams | YES (training) | GPU training | Per #12+#13 baseline + DCVC-FM gain | $0 GPU + ~1 day subagent (existing Ballé already integrated; DCVC-FM extension is small delta) |
| 37 | **DreamerV3 RSSM** (`danijar/dreamerv3`) | Categorical latent dynamics (32 one-hot per timestep) | Z7/Z8 + TT5L V2 predictor | YES (full training) | GPU training; small instances OK on MPS | `[-0.030, -0.010]` per Z8 / TT5L V2 | $0 GPU + ~3 days subagent |
| 38 | **xLSTM** (`NX-AI/xlstm`) | Extended LSTM with sLSTM/mLSTM; NeurIPS 2024 | Z7 GRU alternative | YES (training) | MPS adequate for small instances | `[-0.020, -0.005]` per Z7 alternative | $0 GPU + ~2 days subagent |
| 39 | **RWKV-7 "Goose"** (`BlinkDL/RWKV-LM`) | Linear attention RNN; 2025 | Z7 GRU alternative | YES (training) | MPS adequate | `[-0.018, -0.005]` | $0 GPU + ~2 days subagent |
| 40 | **HiNeRV** (`hmkx/HiNeRV`) | Hierarchical NeRV; NeurIPS 2023; first INR>HEVC | HNeRV family respawn | YES (full per-video training) | GPU (1080p 600-frame ~6.5h on A100; ~24h on MPS) | `[-0.010, -0.003]` per pre-rigor inventory | $10-20 Modal envelope per Wave 3 HNeRV-family crash-resume |

### 2.4 Already-existing-but-deferred lanes (need revival/respawn)

| # | Primitive | Lane registry status | Why deferred | What unblocks |
|---|---|---|---|---|
| 41 | **lane_17_imp Frankle LTH** | RETIRED → pre-rigor inventory PROCEED rank #1 | Original 2026-04-30 KILL was stats.json stub-loop artifact (Catalog #91+#94 closed) | $1-2 Vast.ai 4090 cycle 0 per pre-rigor symposium #856 |
| 42 | **lane_stc_clean_source** | FALSIFIED → pre-rigor PROCEED rank #2 | Original FALSIFICATION was MPS-PROXY-derived INVALID per CLAUDE.md | $0.20 CPU probe per pre-rigor symposium #857 |
| 43 | **PR106 #05+#06 REFORMULATED** | Pre-rigor PROCEED with R1+R2 probe-first | Original kill targeted wrong substrate; paradigm-INTACT/design-CARGO-CULTED | $0 R1+R2 probe first then $10 dispatch per symposium #858 |
| 44 | **lane_pr101_compressai_balle_full** | Pre-rigor PROCEED rank #5 | DCVC-FM 2024 not integrated | $0 + 3-day subagent integration of DCVC-FM; $15 Modal dispatch |
| 45 | **lane_apogee_int4** | Pre-rigor PROCEED rank #6 | Modern PTQ (GPTQ/AWQ) not integrated | $0 + 1-day subagent integration of GPTQ; $5-8 Modal dispatch |
| 46 | **lane_mae_v + lane_saug** | Pre-rigor PROCEED rank #3 | V-JEPA 2 (2025) not integrated | $0 + 3-day subagent integration of V-JEPA 2; $30-50 Modal pretrain |
| 47 | **Wave 3 NeRV-family TERMINATED-API-CRASH** (TCNeRV/BlockNeRV/FFNeRV/DSNeRV/HiNeRV) | TERMINATED-API-CRASH → respawn per Catalog #206 | API crash, not paradigm failure | Catalog #206 crash-resume protocol; HiNeRV highest-EV |
| 48 | **lane_lora_tto** (LoRA test-time-only) | OPERATOR_REVIEW | Modern PEFT (`huggingface/peft`) canon not applied | $0 + integration of huggingface PEFT |
| 49 | **lane_hm_s / lane_wc_s** (Hadamard-Mask-Sparse / Weight-Cluster-Sparse) | OPERATOR_REVIEW | Modern Sparse Autoencoder (Anthropic 2024 SAE) techniques not applied | $0 + integration of EleutherAI SAE |
| 50 | **markov1_aac** (adaptive arithmetic codec) | OPERATOR_REVIEW | Adaptive vs static codec class disambiguation needed | $0 audit |

---

## 3. PR substrate × bolt-on stacking matrix

Per #864 cargo-cult-monotonicity finding: cargo-cult unwinds do NOT compose monotonically across architectural changes. This matrix maps which substrates compose **orthogonally** (SUPER_ADDITIVE potential per Catalog #322 v2 cascade) vs **antagonistically** (SUB_ADDITIVE; cargo-cult-recapture risk) vs **saturating** (no marginal improvement).

### 3.1 Matrix legend

- `O` = ORTHOGONAL (predicted α 1.0-1.5; SUPER_ADDITIVE if α>1)
- `A` = ANTAGONISTIC (predicted α 0.4-0.9; SUB_ADDITIVE / NEGATIVE)
- `S` = SATURATING (α → 0; no marginal improvement; redundant axis)
- `?` = UNPROBED (no empirical anchor; needs Catalog #322 v2 cascade audit)
- `#` = NEEDS_TRAINING (cannot be evaluated without ≥$15 Modal dispatch)

### 3.2 PR101 fec6 (live frontier `0.19205 [contest-CPU]`) × bolt-on matrix

| Bolt-on | Verdict | Predicted ΔS | Rationale | Cost path |
|---|---|---|---|---|
| FEC7 selector | O | `[-0.003, -0.001]` | Selector improvement on selector substrate; needs deliverability proof per Catalog #319 | $0 local CPU + $5 verify |
| DP1 driving prior sidecar | O | `[-0.012, -0.004]` | OOD pretrain prior; orthogonal axis to in-distribution encoder | $0 local + $10-15 paired-CUDA |
| lane_17_imp LTH | O | `[-0.015, -0.005]` | Post-training sparsification; per-tensor sensitivity needed to avoid #864 antagonism risk | $1-2 Vast.ai standalone |
| STC pose-residual sidecar | O | `[-0.020, -0.005]` | Near-Shannon parity-check; orthogonal to fec6 selector | $0.20 CPU probe + $5 |
| PR103 arithmetic codec port | S | `[-0.001, +0.001]` | Already at entropy stage of fec6; saturating | $0 local CPU verify |
| Mamba-2 contextual entropy | A | `[-0.005, +0.010]` | Replaces fec6 selector's role; redundant axis | $30+ training |
| compressai_balle_hyperprior | A | `[+0.005, +0.020]` | Different paradigm; PR101 weights NOT hyperprior-compatible without retrain | $15+ training |
| DCVC-FM 2024 | # | `[-0.025, -0.008]` if retrained | Full neural codec replacement; not zero-retraining | $15+ training |
| Wavelet residual sidecar | O | `[-0.005, -0.001]` | Orthogonal axis (residual channel) | $0 local + $5 |
| Cool-Chic v3 residual | A | `[-0.002, +0.005]` | Per-image; doesn't fit contest 1-video unless per-frame applied | $5+ per-frame training |
| C3 residual | A | `[-0.002, +0.005]` | Per-image; same as Cool-Chic | $5+ training |
| SIREN/coord-MLP residual | S | `[+0.001, +0.005]` | NeRV-family architectural ceiling at PR101 weights | $0 |
| Hessian-block-FP | A | `[+0.005, +0.015]` | Conflicts with PR101's storage_order schema | $15+ training |
| brotli-12 vs current | S | `[-0.001, +0.001]` | Already at entropy stage | $0 verify |

**OPTIMAL PR101 fec6 zero-retraining stack** (per the matrix):
- PR101 fec6 (base) + DP1 sidecar + STC pose-residual sidecar + Frankle LTH 50% prune + FEC7 selector enhancement
- Predicted union ΔS: `[-0.045, -0.011]` ⇒ `[0.147, 0.181]` [contest-CPU] **IF orthogonal stacking holds per Catalog #322 v2 cascade**
- Realistic α-discount per #864: `[-0.020, -0.005]` ⇒ `[0.172, 0.187]` [contest-CPU]
- Cost path: $0 local + $0.20 STC probe + $1-2 LTH + $10-15 verify = **$12-18 total**

### 3.3 PR106 format0d (live frontier `0.20533 [contest-CUDA]`) × bolt-on matrix

| Bolt-on | Verdict | Predicted ΔS | Rationale | Cost path |
|---|---|---|---|---|
| Wavelet residual sidecar | O | `[-0.005, -0.001]` | L2 lane already; orthogonal residual axis | $0 local + $5 |
| Cool-Chic v3 residual | O | `[-0.008, -0.002]` | L2 lane already; orthogonal per-image neural codec | $5 per-image train + $10 |
| C3 residual | O | `[-0.010, -0.003]` | L2 lane already; CVPR 2024 SOTA | $5 per-image train + $10 |
| SIREN residual | A | `[-0.002, +0.005]` | Subsumed by C3 | $5+ |
| Coord-MLP residual | S | `[+0.001, +0.005]` | Subsumed by SIREN/C3 | $0 |
| PR101_pr103_grammar swap | O | `[-0.003, -0.001]` | L2 lane `lane_pr101_pr103_grammar_on_pr106_r2`; orthogonal grammar swap | $0 local + $5 |
| Magic-codec PR106 hybrid | O | `[-0.005, -0.001]` | L2 lane already; orthogonal codec axis | $0 + $5 |
| Hessian-block-FP | A | `[-0.003, +0.005]` | PR106 already uses Selfcomp block-FP; saturating | $0 verify |
| DP1 driving prior sidecar | O | `[-0.010, -0.003]` | OOD pretrain prior; orthogonal | $0 local + $10 |
| Mamba-2 contextual entropy | O | `[-0.008, -0.003]` | Replaces format0d score table with neural entropy; needs training | $20-30 |
| DCVC-FM 2024 | A | `[+0.005, +0.020]` | Full replacement of PR106 paradigm; not stacking | $30+ training |
| STC pose-residual | O | `[-0.005, -0.001]` | Orthogonal residual axis | $0.20 probe + $5 |
| lane_17_imp LTH | A | `[+0.005, +0.020]` | PR106 already uses 94K-param SegMap; not much to prune | $1-2 verify |
| FEC7 selector | A | `[+0.005, +0.015]` | PR106 not selector-based | N/A |

**OPTIMAL PR106 format0d zero-retraining stack**:
- PR106 format0d (base) + DP1 sidecar + C3 residual + Magic-codec hybrid + PR101_pr103 grammar swap
- Predicted union ΔS: `[-0.028, -0.007]` ⇒ `[0.177, 0.198]` [contest-CUDA]
- Realistic α-discount per #864: `[-0.015, -0.005]` ⇒ `[0.190, 0.200]` [contest-CUDA]
- Cost path: $0 local + $5 C3 train + $10 paired-CUDA verify = **$15-20 total**

### 3.4 PR107 apogee × bolt-on matrix

| Bolt-on | Verdict | Predicted ΔS | Rationale |
|---|---|---|---|
| GPTQ/AWQ modern PTQ | O | `[-0.010, -0.002]` | Replaces PR107's older PTQ with 2023-2024 SOTA | $5-8 |
| lane_17_imp LTH | A | `[+0.005, +0.020]` | PR107 already 0.4×-shrunk arch; further pruning compounds errors | N/A |
| DP1 driving prior | O | `[-0.008, -0.003]` | OOD pretrain; orthogonal | $10 |
| Brotli-12 vs current | S | `[-0.001, +0.001]` | Already at entropy stage | $0 |

### 3.5 Cross-PR composition matrix

| Substrate A | Substrate B | Verdict | Predicted α |
|---|---|---|---|
| PR101 fec6 | DP1 | O | 1.2-1.5 |
| PR106 format0d | DP1 | O | 1.1-1.4 |
| PR107 apogee | DP1 | O | 1.1-1.3 |
| PR101 fec6 | PR106 format0d | A | 0.5-0.7 (both target latent stream) |
| PR101 fec6 | PR107 apogee | A | 0.4-0.6 (architecturally conflicting) |
| PR106 format0d | PR107 apogee | A | 0.4-0.6 (conflicting weights) |
| PR101 fec6 + DP1 | PR106 format0d + DP1 | A | 0.6-0.8 (DP1 cancellation) |

**Operator takeaway from matrix**: the highest-α compositions all involve PR-substrate-base + DP1 (orthogonal OOD pretrain). Cross-PR-substrate stacks are uniformly antagonistic.

---

## 4. TOP-5 zero-retraining stacking compositions (DETAILED)

### 4.1 Composition #1: PR101 fec6 + DP1 driving prior sidecar

**Substrates**: `pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean` (archive `6bae0201`) + DP1 codebook sidecar (Comma2k19-cached per Catalog #213; ~700KB codebook amortized)

**Predicted ΔS**: `[-0.012, -0.004]` ⇒ `[0.180, 0.188]` [contest-CPU] under Catalog #322 v2 cascade with `composition_alpha ∈ [1.2, 1.5]`

**First-principles**: Hinton distillation 2014 (DP1 codebook IS distilled OOD prior) + Atick-Redlich 1990 cooperative-receiver (DP1 acts as scorer-class-prior side-info) + LINGO-1 (Wayve 2023) + DriveGPT (OpenDriveLab CVPR 2024) world-model pretraining

**Cost path**:
- $0 local CPU: DP1 codebook construction via `tac.substrates.pretrained_driving_prior.distillation.distill_codebook` (per Catalog #209 contest-video-leakage refusal via `Comma2k19FrameIterator`); Faiss-IVF-PQ quantization to ≤700KB shippable budget (per upcoming primitive #32 integration)
- $10-15 paired-CUDA verify on Modal A100 per `tools/dispatch_modal_paired_auth_eval.py` with Catalog #246 anchor-skip-if-exists discipline

**Cargo-cult monotonicity risk per #864**: LOW. DP1 is OOD pretrain (Comma2k19 ≠ contest video) so does NOT re-bake fec6's training-distribution cargo-cults. Validated via Catalog #209 contest-video-leakage gate.

**Pre-dispatch checklist** (per Catalog #325 per-substrate symposium):
- (1) Cargo-cult audit per Catalog #303: ALREADY DONE for DP1 in symposium #855
- (2) 9-dim checklist per Catalog #294: ALREADY DONE for DP1
- (3) Observability surface per Catalog #305: ADD section
- (4) Sextet pact deliberation: NEEDED before paid dispatch
- (5) Reactivation criteria pinned: ADD
- (6) Catalog #324 post-training Tier-C validation: `pending_post_training`

### 4.2 Composition #2: PR106 format0d + DP1 + C3 residual sidecar

**Substrates**: `pr106_format0d_latent_score_table` (archive `9cb989cef519`) + DP1 sidecar (700KB) + C3 residual sidecar (L2 lane `lane_c3_residual_pr106_sidecar_dispatch_ready`)

**Predicted ΔS**: `[-0.015, -0.005]` ⇒ `[0.190, 0.200]` [contest-CUDA] under α ∈ [1.1, 1.4]

**First-principles**: Kim et al. ICLR 2024 C3 + Hinton distillation + Atick-Redlich

**Cost path**:
- $0 local: C3 per-image training via MPS (Cool-Chic v2 reference impl); DP1 codebook
- $5-10 per-image C3 training (sub-PNG-byte-budget per image)
- $10-15 paired-CUDA verify
- **Total**: $15-25

**Cargo-cult monotonicity risk per #864**: LOW. C3 is per-image learned context (orthogonal to format0d score table) + DP1 OOD pretrain.

### 4.3 Composition #3: PR101 fec6 + FEC7 selector enhancement + PR103 arithmetic codec port

**Substrates**: PR101 fec6 + FEC7 selector improvement (`pr101_fec7_selector.evaluate_fec7_candidates`) + PR103 arithmetic codec swap at entropy stage

**Predicted ΔS**: `[-0.005, -0.001]` ⇒ `[0.187, 0.191]` [contest-CPU] (modest but ZERO-COST)

**First-principles**: Information-theoretic — FEC7 is selector-superset of fec6; PR103 arithmetic is entropy-stage-equivalent of fixed Huffman k16

**Cost path**:
- $0 local CPU: bit-exact compose; `tools/build_deterministic_packet.py` per Catalog #158 canonical helper
- $5 paired-CUDA verify

**Total**: $5. **The cheapest visible win.**

**Cargo-cult monotonicity risk per #864**: VERY LOW. Both bolt-ons are already-canonical pact primitives; sister L2 lanes `lane_pr101_dynamic_derivers` + `lane_pr103_arithmetic_codec_port` already passed L1→L2 promotion canonical 4-gate (Catalog #233).

### 4.4 Composition #4: PR101 fec6 + lane_17_imp Frankle LTH 50% sparsity prune

**Substrates**: PR101 fec6 + Frankle Lottery Ticket Hypothesis (per-tensor sensitivity-masked 50% magnitude pruning on PR101 renderer.bin)

**Predicted ΔS**: `[-0.015, -0.005]` ⇒ `[0.177, 0.187]` [contest-CPU]

**First-principles**: Frankle-Carbin 2019 LTH + 2024 Minitron NVIDIA distillation + 40× compression with <2% accuracy loss

**Cost path**:
- $1-2 Vast.ai 4090 IMP cycle 0 standalone per pre-rigor symposium #856
- $0 local CPU rebuild + paired-CUDA verify

**Total**: $5-10.

**Cargo-cult monotonicity risk per #864**: **MEDIUM**. LTH may regress pose component if pruning targets PoseNet-sensitive weights. Mitigation: per-tensor sensitivity mask from canonical sensitivity_map (Hook #1 wire-in); refuse pruning on tensors with high pose-axis Jacobian.

### 4.5 Composition #5: PR101 fec6 + STC syndrome-trellis pose-residual sidecar

**Substrates**: PR101 fec6 + STC pose-residual (per pre-rigor symposium #857)

**Predicted ΔS**: `[-0.020, -0.005]` ⇒ `[0.172, 0.187]` [contest-CPU]

**First-principles**: Filler-Pevný-Fridrich 2011 STC near-Shannon-rate parity-check codes

**Cost path**:
- $0.20 CPU probe per pre-rigor symposium #857 (CHEAPEST signal in entire roadmap)
- $5 paired-CUDA verify post-probe-PROCEED

**Total**: $5-10. **The 2nd cheapest visible win.**

**Cargo-cult monotonicity risk per #864**: LOW. STC operates on pose-residual residual; pact `tac.codec.pose_filler_stc_codec` already canonical; the FALSIFIED 2026-04-30 verdict was MPS-PROXY-derived INVALID per CLAUDE.md.

---

## 5. Local-M5-Max + 128GB unified utilization plan

Per operator's Directive B: *"what can be run using local CPU and MPS (if anything for MPS)? are we maximizing use of local m5 max and 128 GB unified"* — the M5 Max with 128GB unified memory is an EXTREMELY underutilized asset for pact's current workflow. This section maps what's runnable locally and the canonical helper integration recommendations.

### 5.1 Currently UNUSED LOCAL primitives (per Directive B)

#### 5.1.1 Faiss-IVF-PQ codebook construction (HIGH-VALUE; CPU-native)

- **What**: ATW V2-1 channel construction via per-region (16×16) SegNet softmax histogram product-quantized to ≤2KB; DP1 codebook product-quantization for sidecar compose
- **Why local**: Faiss-IVF-PQ is CPU-NATIVE; 128GB unified handles N=600 pairs × 256-D vectors trivially
- **Time-to-value**: ~1 day subagent integration (primitive #32 above) + ~1 hour local compute
- **Wires into**: ATW V2-1 design + DeliverabilityProof per Catalog #319 + DP1 sidecar compression
- **Predicted impact**: Unblocks ATW V2-1 dispatch wave; cuts DP1 codebook bytes 3-5× (700KB → 140-230KB) which materially affects composition #1 cost path

#### 5.1.2 Per-pair master gradient fp64 computation (HIGH-VALUE; trivial on 128GB)

- **What**: fp64 master gradient for ALL PR archives (PR101 + PR106 + fec6 + format0d; ~176MB per archive)
- **Why local**: 128GB unified; fp64 trivial; CPU-only inference path
- **Time-to-value**: 2-4h editor (`tools/run_local_master_gradient_for_archive.py` wrapper) + 1-2h compute per archive
- **Wires into**: Catalog #319 v2 cascade (cathedral autopilot reweight); enables per-archive Lagrangian-dual-derived priority ordering
- **Predicted impact**: Master gradient currently only computed for fec6; PR106 + PR101_lc_v2 NOT yet. Drives Catalog #319 cascade properly for ALL archives.

#### 5.1.3 $0 CPU pre-probes (5+ pending; canonical helper missing)

- **What**: stc 3a / PR106 R1+R2 / TT5L MI / DP1 audit pre-probes
- **Why local**: Each runs in CPU minutes on M5 Max
- **Time-to-value**: 4-8h editor (canonical `tac.optimization.local_cpu_pre_probe_runner` helper from primitive list) + 8-24h local compute across all pending
- **Wires into**: Probe outcomes ledger (Catalog #313); autopilot ranker (Hook #4); canonical Provenance umbrella (Catalog #323)
- **Predicted impact**: Cheapest possible disambiguator signal per Catalog #313; defers paid GPU until probe verdict; could halve GPU dispatch costs for next wave

#### 5.1.4 macOS-CPU advisory proxy autopilot ranking (HIGH-FIDELITY proxy per PR107 anchor)

- **What**: Wrap `tools/score_macos_cpu_advisory_proxy.py` over top-20 candidate queue
- **Why local**: PR107 empirical anchor showed M5 Max CPU `0.19664189` matches GHA Linux x86_64 `0.1966358879` within `6e-6` (CLAUDE.md "Submission auth eval" section) — HIGH-FIDELITY proxy
- **Time-to-value**: 2-4h editor (canonical `tac.optimization.macos_cpu_proxy_batch_ranker`) + ~5 min/candidate
- **Wires into**: Autopilot loop (Hook #4); one-arg-local-MPS-vs-Modal switch per Catalog #317
- **Constraint**: macOS-CPU is `[macOS-CPU advisory]` ONLY — NEVER `[contest-CPU]` per Catalog #192. Used as proxy for autopilot ranking, never promoted to canonical score
- **Predicted impact**: Free re-ranking signal; eliminates the "spend $5+ on Modal smoke just to rank a candidate" anti-pattern

#### 5.1.5 MPS-research-signal sweeps (HIGH-VALUE; curve-shape priors for autopilot)

- **What**: Hyperparameter sweep for (β-sweep for C6 IBPS / depth-sweep for Z6 / Mamba block-size sweep for Z7 / FiLM hidden-dim sweep)
- **Why local**: 128GB unified handles small models; MPS adequate for proxy training per CLAUDE.md "MPS auth eval is NOISE" (proxy training permitted)
- **Time-to-value**: 4-8h editor (`tac.optimization.mps_research_signal_sweep_harness` canonical harness wrapping mps_research_signal) + 8-12h overnight local compute
- **Wires into**: Autopilot loop curve-fit priors per `tac.optimization.mps_research_signal`; Catalog #313 probe outcomes
- **Constraint**: Tagged `[MPS-PROXY]` and `evidence_grade=MPS-research-signal` per Catalog #192. Never promoted to score truth
- **Predicted impact**: Eliminates GPU-dispatch-as-hyperparameter-search anti-pattern; cuts dispatch cost 3-5× via informed prior

#### 5.1.6 Cool-Chic v3 + HiNeRV per-frame inference (HIGH-VALUE; MPS-runnable)

- **What**: Cool-Chic v3 per-image / HiNeRV per-frame inference for residual sidecar evaluation
- **Why local**: Both fit easily in 128GB unified; MPS adequate for inference
- **Time-to-value**: ~3 days subagent + ~12h overnight local compute
- **Wires into**: PR106 sidecar lanes (cool_chic / siren / coord_mlp); HNeRV-family respawn

#### 5.1.7 VGGT inference (CVPR 2025 Best Paper; fits 128GB)

- **What**: VGGT pretrained encoder inference for TT5L V2 pose teacher; runs in <1 second per inference per the paper
- **Why local**: 128GB easily accommodates; MPS adequate
- **Time-to-value**: ~3 days subagent integration (primitive #34 above) + ~30 min local inference for ALL contest video frames
- **Wires into**: TT5L V2 design + DP1 pose anchor

#### 5.1.8 DUSt3R/MASt3R inference (CVPR 2024)

- **What**: 2-frame dense reconstruction for PoseNet teacher residual encoding
- **Why local**: MPS inference adequate
- **Time-to-value**: ~2 days subagent + ~1 hour local inference for ALL 600 contest pairs
- **Wires into**: TT5L V2 + DP1 pose

#### 5.1.9 DreamerV3 RSSM small-model training (MPS adequate)

- **What**: DreamerV3 RSSM training for Z7/Z8/TT5L V2 predictor (small <200M param instances)
- **Why local**: M5 Max with 128GB unified handles DreamerV3 default config easily
- **Time-to-value**: ~3 days subagent + ~12h overnight local training
- **Wires into**: Z7/Z8/TT5L V2 design memos

#### 5.1.10 Sparse Autoencoder activation extraction (research wave NEW convergence)

- **What**: SAE per Anthropic 2024 interpretability research; extract sparse activations from PR101/PR106 weights for lane_hm_s/lane_wc_s sparse substrate revival
- **Why local**: SAE training is small-model; runs on MPS
- **Time-to-value**: ~3 days subagent + ~6h local training
- **Wires into**: lane_hm_s / lane_wc_s sparse substrate revival per primitive #49

### 5.2 Canonical helper integration recommendations

The audit reveals that pact has TWO canonical local helpers (`mps_research_signal` + `macos_cpu_advisory_signal`) that are **structurally underutilized**. They exist + are wired into operator_authorize.py per Catalog #317 (one-arg local-MPS-vs-Modal switch), but no BATCH RUNNER or AUTOMATED SWEEP HARNESS wraps them.

**Recommended NEW canonical helpers (per Section 0 table)**:

1. `tac.optimization.local_cpu_pre_probe_runner` — Common harness for the 5+ pending CPU pre-probes (stc / PR106 #05+#06 / TT5L MI / DP1 audit / future). Schema: `{probe_id, target_lane, evidence_grade='local-cpu-pre-probe', archive_sha, started_at_utc, completed_at_utc, verdict, rationale}`. Append-only JSONL to `.omx/state/local_cpu_pre_probes.jsonl` per Catalog #131/#138.

2. `tac.optimization.macos_cpu_proxy_batch_ranker` — Wraps `score_macos_cpu_advisory_proxy.py` over top-N candidate queue from autopilot; emits ranked manifest with macOS-CPU advisory grade per Catalog #192.

3. `tac.optimization.mps_research_signal_sweep_harness` — Canonical sweep harness wrapping `mps_research_signal` for hyperparameter curve-shape estimation; runs N config × M epochs on MPS overnight; emits curve-fit prior used by paid GPU dispatch.

4. `tac.local_dispatch.faiss_ivf_pq_channel_builder` — CPU-native Faiss-IVF-PQ codebook builder for ATW V2-1 + sister substrate channel construction; ≤2KB shippable budget verification.

5. `tools/run_local_master_gradient_for_archive.py` — Wraps `tools/extract_master_gradient.py --target local-cpu` with sentinel-clean parity + fcntl-locked artifact write to `.omx/state/master_gradient_consumers/`; tool dispatch per Catalog #270 scope clarification (CPU-only).

**Catalog #317 audit**: the operator-authorize.py CLI already supports `--target {auto|modal|vastai|lightning|local|local-mps|local-cpu}` (per Catalog #317 + commit a67f8fc12). However, the `local-mps` and `local-cpu` paths are ONLY invoked when an operator-authorize recipe is explicitly declared with `platform: local-mps` or `platform: local-cpu`. **NO operator-authorize recipes currently declare local platforms**. This is the structural gap: the helper exists, the CLI switch exists, but NO RECIPE uses it.

**Recommended recipe canonicalization**:
- Add `local_mps_proxy_dispatch.yaml` template recipe pattern under `.omx/operator_authorize_recipes/`
- Add `local_cpu_proxy_dispatch.yaml` template recipe pattern
- Wire into `tools/local_pre_deploy_check.py` 9th check `local_dispatch_recipe_consistent`
- Document the "one-arg switch" in `docs/local_dispatch_canonical_usage.md`

---

## 6. NEW Catalog gate proposals (operator review)

Per the operator's "Subagent coherence-by-default" non-negotiable + CLAUDE.md "Gate consolidation discipline" Catalog #299 quota brake (must justify net-additive gates), the following NEW catalog gates are recommended for operator review:

### 6.1 Catalog #330 (proposed): `check_bolt_on_stacking_composition_declares_retraining_required`

**Scope**: Refuses composition design memos under `.omx/research/*_design_<YYYYMMDD>.md` (dated >= 2026-05-18) that declare a bolt-on stacking composition WITHOUT explicit `retraining_required` field (one of: `zero`, `minimal`, `yes`).

**Bug class**: A future composition memo that omits this distinction silently mis-classifies a paradigm-replacement (`yes` retraining) as a bolt-on stack (`zero`); the operator's expectation of "asymptotic stacking without costly retraining" is then unmet.

**Acceptance**: explicit `retraining_required:` field in YAML frontmatter OR `## Retraining required` section in body.

**Wire-in**: WARN-ONLY initially; strict-flip after the operator-routed backfill sweep clears the 28 L2+ promotion-ready compositions.

**Sister of**: Catalog #294 (9-dim checklist) + Catalog #303 (cargo-cult audit) + Catalog #322 (composition_alpha anti-pattern).

### 6.2 Catalog #331 (proposed): `check_local_runnable_bolt_on_cites_canonical_local_helper_integration`

**Scope**: Refuses bolt-on design memos that claim `local_runnable: cpu` or `local_runnable: mps` WITHOUT citing one of the canonical local helpers (`tac.optimization.mps_research_signal` / `tac.optimization.macos_cpu_advisory_signal` / `tac.optimization.local_cpu_pre_probe_runner` / `tac.optimization.macos_cpu_proxy_batch_ranker` / `tac.optimization.mps_research_signal_sweep_harness` / `tac.local_dispatch.faiss_ivf_pq_channel_builder`).

**Bug class**: A future bolt-on claims local-runnability but doesn't route through the canonical helpers — the resulting output silently bypasses (a) the Catalog #192 macOS-CPU non-promotion gate, (b) the Catalog #317 one-arg local-MPS-vs-Modal switch, (c) the canonical JSONL append discipline per Catalog #131/#138.

**Acceptance**: explicit citation of canonical helper in `## Local runnability` section OR same-line waiver `# LOCAL_RUNNABLE_HELPER_WAIVED:<rationale>`.

**Wire-in**: WARN-ONLY initially; strict-flip when canonical helpers are landed (per Section 5.2).

**Sister of**: Catalog #192 (macOS-CPU advisory not promoted) + Catalog #317 (one-arg local dispatch switch) + Catalog #1 (MPS-fallback default).

### 6.3 Optional Catalog #332 (proposed): `check_composition_uses_canonical_alpha_v2_cascade`

**Scope**: Refuses composition memos that predict α (composition_alpha) WITHOUT routing through `tac.optimization.substrate_composition_matrix` canonical matrix + `tools/cathedral_autopilot_autonomous_loop.py::adjust_predicted_delta_for_composition_alpha_v2` cascade per Catalog #322 sister discipline.

**Bug class**: Hand-rolled α predictions silently bypass the Catalog #319 v2 cascade + DeliverabilityProof discipline; can re-introduce the phantom-score-from-research-sidecar class extincted by Catalog #321/#322.

**Acceptance**: explicit citation OR same-line waiver.

**Wire-in**: WARN-ONLY initially.

**Sister of**: Catalog #319/#321/#322/#323.

---

## 7. Operator op-routables (prioritized HIGH-EV first)

Per Section 0 TOP-5 zero-retraining stacking compositions + Section 5 local-utilization plan + CLAUDE.md "Race-mode rigor inversion + parallel-dispatch first" priority. Every op-routable below is the canonical apples-to-apples improvement per per-substrate symposium + research wave + this audit.

### TIER 1: CHEAPEST + HIGHEST-EV (can land within 24-48h; $0-5 spend; pure local + minimal verify)

1. **OP1**: Land **`tools/run_local_master_gradient_for_archive.py`** wrapper around existing `tools/extract_master_gradient.py --target local-cpu`. Run on PR101_lc_v2 + PR106_format0d + fec6 archives. ~$0 + 2-4h editor + 6h local compute. **Unblocks Catalog #319 v2 cascade for ALL archives**.

2. **OP2**: Land **`tac.optimization.local_cpu_pre_probe_runner`** canonical helper. Execute 5+ pending CPU pre-probes (stc 3a $0.20 / PR106 #05+#06 R1+R2 $0 / TT5L MI $0 / DP1 audit $0). ~$0 + 4-8h editor + 8-24h local compute. **Unblocks 4 of the 7 pre-rigor reactivation candidates per the pre-rigor inventory**.

3. **OP3**: Land **`tac.optimization.macos_cpu_proxy_batch_ranker`** canonical helper. Run over top-20 candidate queue from autopilot. ~$0 + 2-4h editor + 100 min local compute. **Free re-ranking signal per Catalog #192 + #317**.

4. **OP4**: Execute **Composition #3 (PR101 fec6 + FEC7 + PR103 arithmetic)** zero-retraining bit-exact compose. ~$0 local + $5 paired-CUDA verify. **The cheapest visible PR101 frontier improvement**; predicted ΔS `[-0.005, -0.001]` ⇒ `[0.187, 0.191]` [contest-CPU].

5. **OP5**: Execute **Composition #5 (PR101 fec6 + STC pose-residual sidecar)** $0.20 CPU probe per pre-rigor symposium #857. ~$0.20 + $5 paired-CUDA verify. **The cheapest possible disambiguator for STC clean-source paradigm**.

### TIER 2: HIGH-EV ($10-25 spend; bolt-on stacking compositions; can land within 1 week)

6. **OP6**: Execute **Composition #1 (PR101 fec6 + DP1 driving prior sidecar)** per Section 4.1. ~$10-15 paired-CUDA verify after $0 local DP1 codebook construction. Predicted ΔS `[-0.012, -0.004]`. **Requires canonical helpers from OP1 + OP2 to be landed for proper Catalog #319 routing**.

7. **OP7**: Execute **Composition #2 (PR106 format0d + DP1 + C3 residual)** per Section 4.2. ~$15-25 envelope. Predicted ΔS `[-0.015, -0.005]` [contest-CUDA].

8. **OP8**: Execute **Composition #4 (PR101 fec6 + lane_17_imp Frankle LTH)** per Section 4.4. ~$5-10 envelope ($1-2 Vast.ai LTH + $5 verify). **Caveat**: per-tensor sensitivity-mask mitigation needed for #864 antagonism risk; should ONLY proceed AFTER OP1 master gradient computation provides the sensitivity mask.

9. **OP9**: Land **`tac.optimization.mps_research_signal_sweep_harness`** + execute β-sweep for C6 IBPS (currently DEFER per Catalog #324 phantom-band falsification). ~$0 + 4-8h editor + 8-12h overnight MPS. **Eliminates GPU-dispatch-as-hyperparameter-search anti-pattern**.

10. **OP10**: Land **`tac.local_dispatch.faiss_ivf_pq_channel_builder`** + execute ATW V2-1 channel construction (per ATW V2 symposium Revision #2 binding). ~$0 + 4-8h editor + 1h local compute. **Unblocks ATW V2-1 dispatch wave**.

### TIER 3: STRATEGIC LOCAL-FIRST (canonical helper integrations; $0 GPU; multi-day subagent landings)

11. **OP11**: Land **Mamba-2 integration** into pact composition registry per primitive #31. ~$0 + 2 days subagent. Enables Z7 Mamba-2 alternative per Catalog #308 N>=3 alternative probes.

12. **OP12**: Land **DCVC-FM 2024 integration** into pact via CompressAI extension per primitive #36. ~$0 + 1 day subagent. Enables lane_pr101_compressai revival per pre-rigor #5.

13. **OP13**: Land **VGGT + DUSt3R integration** into pact per primitives #34+#35. ~$0 + 3 days subagent. Enables TT5L V2 LAPose teacher.

14. **OP14**: Land **HiNeRV respawn** per Catalog #206 crash-resume protocol per primitive #40. ~$0 editor + $10-20 Modal envelope.

15. **OP15**: Land **`tac.optimization.macos_cpu_proxy_batch_ranker` wire-in to autopilot loop** so every cathedral autopilot top-20 ranking includes the free macOS-CPU advisory grade per Catalog #192. ~$0 + 2-4h editor.

### TIER 4: NEW CATALOG GATE LANDINGS (apparatus hardening per Section 6)

16. **OP16**: Land **Catalog #330** (bolt-on stacking declares retraining_required). WARN-ONLY initially.

17. **OP17**: Land **Catalog #331** (local-runnable bolt-on cites canonical helper). WARN-ONLY initially.

18. **OP18**: Land **Catalog #332** (composition uses canonical α v2 cascade). Optional / depends on operator review.

### Sequencing recommendation

**Week 1**: OP1 + OP2 + OP3 + OP4 + OP5 (all local; $5.20 total spend). Lands the canonical local helpers; produces master gradient for all archives; executes cheapest 2 stacking compositions. **Expected score impact**: PR101 frontier `0.19205` → `[0.183, 0.190]` predicted.

**Week 2**: OP6 + OP7 + OP8 (stacking compositions; ~$30-50 total). **Expected score impact**: PR101 frontier `[0.172, 0.187]`; PR106 frontier `[0.190, 0.200]`.

**Week 3**: OP9 + OP10 + OP15 (local sweep + canonical helpers + autopilot wire-in; $0 GPU). Closes the local-utilization gap.

**Week 4+**: OP11-OP14 (canonical helper integrations; $10-30 spend per integration); OP16-OP18 (catalog gate landings).

**Estimated cumulative spend**: $40-90 across 4 weeks. **Estimated frontier improvement**: PR101 `0.19205` → `~0.175` [contest-CPU] (best-case under orthogonal stacking) OR `~0.185` [contest-CPU] (realistic discount per #864).

---

## 8. Cross-references

### 8.1 Today's 10 symposiums (per_substrate_optimal_form_symposium per Catalog #325)

- `council_per_substrate_symposium_tt5l_foveation_lapose_20260517`
- `council_per_substrate_symposium_z7_lstm_predictive_coding_20260517`
- `council_per_substrate_symposium_atw_v2_reactivation_20260518`
- `council_c6_ibps_post_empirical_reactivation_symposium_first_per_substrate_optimal_form_20260518`
- `council_per_substrate_symposium_dp1_deep_dive_20260517`
- `council_per_substrate_symposium_lane_17_imp_20260517`
- `council_per_substrate_symposium_pr106_05_06_reformulated_20260517`
- `council_per_substrate_symposium_stc_3a_sidecar_a1_residual_20260517`
- `council_per_substrate_symposium_stc_clean_source_20260517`
- `council_per_substrate_symposium_nscs06_v8_path_b_20260517`

### 8.2 Sister research artifacts (today's wave)

- `comprehensive_research_wave_20260518` — TOP-3 OSS + 8 convergent-truth tuples + 8 NEW convergences + 60+ refs
- `pre_rigor_kill_defer_falsified_inventory_20260517` — 34 pre-rigor verdicts re-evaluated; TOP-7 reactivations
- `signal_loss_audit_20260517` — 142 working-tree entries swept; 3 critical batches now committed
- `frontier_signal_loss_permanent_fix_landed_20260517` — Catalog #316 frontier scan canonical helper

### 8.3 Canonical helpers (already-integrated; underutilized)

- `tac.optimization.mps_research_signal` (Catalog #192 / #1 / #317)
- `tac.optimization.macos_cpu_advisory_signal` (Catalog #192 / #317)
- `tac.composition.registry` (Catalog #169 — CompressAI primitives) + 19 registered primitives
- `tac.packet_compiler.*` (PR101/PR103/PR105/PR106/PR107 sister primitives; 20+ modules)
- `tac.codec.*` (per-tensor / cooperative-receiver / Wyner-Ziv / Filler-STC; 12+ modules)
- `tac.substrates.pretrained_driving_prior.*` (DP1; Catalog #209/#210/#211/#213)
- `tools/score_macos_cpu_advisory_proxy.py` (high-fidelity proxy per PR107 6e-6 anchor)
- `tools/build_mps_research_signal_manifest.py` (sweep harness — wrapping needed)
- `tools/dispatch_modal_paired_auth_eval.py` (Catalog #246 anchor-skip-if-exists)
- `tools/operator_authorize.py --target {local|local-mps|local-cpu}` (Catalog #317)

### 8.4 CLAUDE.md non-negotiables honored

- "MPS auth eval is NOISE" + "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" → all local work tagged `[MPS-PROXY]` or `[macOS-CPU advisory]`, NEVER promoted to `[contest-CPU]`
- "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" + Catalog #220 → all composition memos must declare operational mechanism
- "HNeRV / leaderboard-implementation parity discipline" L7 → bolt-ons (≤350 LOC) share; substrate engineering unique-ifies
- Catalog #864 "Cross-substrate-composition cargo-cult-unwind monotonicity" → α-discount applied to every composition prediction; per-tensor sensitivity-mask mitigation required for LTH compositions
- Catalog #316 frontier-scan canonical state → live frontier numbers cited verbatim
- Catalog #322 v2 cascade + Catalog #319 deliverability proof → all composition memos must route through canonical cascade

### 8.5 6-hook wire-in declaration per Catalog #125

- **Hook #1 sensitivity-map**: ACTIVE. Master gradient computation (OP1) produces canonical per-tensor sensitivity that feeds `tac.sensitivity_map.*`.
- **Hook #2 Pareto constraint**: ACTIVE. Composition α matrix (Section 3) feeds `tac.pareto_*` via canonical `substrate_composition_matrix.json` posterior.
- **Hook #3 bit-allocator**: ACTIVE. Faiss-IVF-PQ channel construction (OP10) emits ≤2KB shippable budget verification consumed by bit-allocator.
- **Hook #4 cathedral autopilot dispatch hook**: PRIMARY. Every recommended op-routable feeds the autopilot ranker via canonical macos_cpu_advisory_signal + mps_research_signal + probe_outcomes_ledger.
- **Hook #5 continual-learning posterior update**: ACTIVE. Every CPU pre-probe (OP2) emits anchor via `tac.continual_learning.posterior_update_locked`.
- **Hook #6 probe-disambiguator**: ACTIVE. The local pre-probe runner (OP2) IS the canonical disambiguator between PROCEED / DEFER / KILL verdicts on pre-rigor reactivation candidates.

---

**End of asymptotic-stacking-plus-local-M5-Max-utilization-audit memo. Lane `lane_local_m5_max_utilization_audit_20260518` advances L0 → L1 at memo landing per Catalog #126 lifecycle discipline.**
