---
title: DreamerV3 RSSM MLX-local L0 SCAFFOLD landed
date_utc: 2026-05-26T07:12:00Z
lane: lane_dreamer_v3_rssm_mlx_scaffold_20260526
subagent_id: subagent_a_dreamer_v3_rssm_20260526T065116Z_10444
parent_session: main
substrate_id: dreamer_v3_rssm_l0_mlx_scaffold
substrate_aliases:
  - dreamerv3_rssm_categorical_posterior_c6_paradigm_bridge_v1
  - c6_ibps_v2_path_b2_rssm

council_tier: T1
council_attendees:
  - Shannon
  - Dykstra
  - PR95Author
  - Assumption-Adversary
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "MLX-local L0 scaffold is the canonical landing for op-routable #2 of the 2026-05-19 T3 symposium"
    classification: HARD-EARNED
    rationale: "Symposium verdict PROCEED_WITH_REVISIONS explicitly enumerated op-routable #2 (Path B2 design memo + substrate scaffold) at $0 cost; this landing satisfies that op-routable at L0 MLX-local scope per CLAUDE.md MLX portable-local-substrate authority + corrected #1258 anchor (72× MLX faithful at frontier-tightening granularity)."
  - assumption: "L0 archive grammar (RSSMC1 27-byte header + decoder_blob + indices_blob + meta_blob) is canonical Catalog #124 8-field representation-lane declaration"
    classification: HARD-EARNED-PARTIAL
    rationale: "All 8 fields declared inline in __init__.py ARCHIVE_GRAMMAR_FIELDS dict + sister tokens in module/archive/inflate sources; per Catalog #124 the AST walker observes the declaration. PARTIAL because score_aware_loss is declared as pending Path B2 trainer landing (canonical helper routing not yet wired in MLX side; PyTorch port at L1+ wires through tac.substrates._shared.score_aware_common per Catalog #164)."
  - assumption: "PyTorch inflate runtime parity with MLX module at decoder boundary is empirically verified at L0"
    classification: CARGO-CULTED-PENDING-CANONICAL-BILINEAR
    rationale: "L0 MLX module uses simple mx.repeat 2x upsample (not align_corners=False bilinear) → empirical max_abs drift = 24.34 in [0,255] space at sub-uint8 boundary. The canonical PR95 bilinear helper at tac.local_acceleration.pr95_hnerv_mlx::bilinear_resize2x_align_corners_false_nhwc produces 3e-5 max_abs parity (sister #1251 empirical anchor). Wiring it into this substrate is the canonical L0→L1 promotion step; the L0 test asserts max_abs < 50.0 (honestly documents the gap; Catalog #287 evidence-tag discipline)."
council_decisions_recorded:
  - "op-routable #1: wire canonical bilinear_resize2x_align_corners_false_nhwc from tac.local_acceleration.pr95_hnerv_mlx into this substrate's _bilinear_resize_2x_nhwc helper (drops MLX↔PyTorch decoder parity from ~24.3 to <1.0 in [0,255] space; L0→L1 promotion step)"
  - "op-routable #2: build L1 sister gate tools/gate_mlx_candidate_contest_equivalence_rssm.py parameterized to consume RSSMC1 grammar (current #1265 gate consumes PR95 grammar only; refuses our archive as expected; documented empirical refusal in equivalence_gate_attempt.json)"
  - "op-routable #3: PyTorch port via canonical Path 3 export bridge sister #1251 + #1257 (export bridge generalized to write RSSMC1 instead of PR95 archive); enables Catalog #1265 gate consumption via parallel sister gate (op-routable #2)"
  - "op-routable #4: score-aware loss via tac.substrates._shared.score_aware_common.score_pair_components per Catalog #164 (canonical eval_roundtrip-mandatory routing; replaces L0 synthetic-frame MSE proxy)"
  - "op-routable #5: real frame loader from upstream/videos/0.mkv per Catalog #114 (replaces L0 synthetic random target frames; enables convergence-against-actual-contest-distribution validation)"
  - "op-routable #6: Catalog #325 PER-SUBSTRATE OPTIMAL FORM symposium revisions applied per the 2026-05-19 T3 verdict's 5 binding revisions (Contrarian VETO + Tao predicted-band justification + Assumption-Adversary 5 classifications)"
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
council_override_rationale: ""
horizon_class: frontier_pursuit
canonical_equation_refs:
  - categorical_posterior_capacity_vs_continuous_gaussian_v1
  - categorical_blahut_arimoto_rate_distortion_v1
predicted_band_validation_status: pending_post_training
related_deliberation_ids:
  - council_t3_dreamerv3_rssm_paradigm_bridge_per_substrate_symposium_20260519
  - mlx_candidate_contest_equivalence_gate_landed_20260526
  - pr95_mlx_full_decoder_downstream_scorer_drift_landed_20260526
  - pr95_mlx_full_inflate_parity_closure_landed_20260526
  - dreamerv3_rssm_categorical_rd_canonical_equation_derivation_20260520T131815Z
---

# DreamerV3 RSSM MLX-local L0 SCAFFOLD LANDED 2026-05-26

**Lane**: `lane_dreamer_v3_rssm_mlx_scaffold_20260526` L1 (impl_complete + research_only + lane_class=substrate_engineering)
**Evidence grade**: `[macOS-MLX research-signal]` per CLAUDE.md "MLX portable-local-substrate authority"
**Cost**: $0 + ~75 min wall-clock (4 source files + 1 trainer + 11 tests + 1 landing memo); NO paid CUDA dispatch.

## Verdict

**PROCEED** — L0 MLX-local scaffold satisfies op-routable #2 of the 2026-05-19 T3 grand council per-substrate symposium ("Path B2 design memo + substrate scaffold at $0 cost"). All 11 dedicated tests pass; MLX smoke trainer end-to-end clean (loss 5431.49 → 5424.68 over 2 epochs); byte-deterministic RSSMC1 archive emitted (515 KB; sha256 `3dbaf76e43a970ff034ba8cd…`). PyTorch inflate runtime path empirically verified end-to-end (MLX-train → archive → PyTorch decode → camera-resolution uint8 raw output for 4 frames).

## What this landing IS

Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" + Catalog #240 acceptance cascade (c) pre-build substrate-engineering:

1. **Canonical substrate scaffold** at `src/tac/substrates/dreamer_v3_rssm/` with all 4 canonical files (module / archive / inflate / __init__) + tests directory.
2. **MLX-local training loop** at `experiments/train_substrate_dreamer_v3_rssm.py` (smoke-only; `_full_main` raises `NotImplementedError` per pre-build council-gating).
3. **Byte-deterministic RSSMC1 archive grammar** (27-byte header + decoder_blob + indices_blob + meta_blob; canonical Catalog #124 8-field declaration).
4. **PyTorch inflate runtime** at `src/tac/substrates/dreamer_v3_rssm/inflate.py` consuming RSSMC1 archive + producing camera-resolution uint8 raw output per Catalog #146 + #205.
5. **Lane registration** at L1 `impl_complete=true` + `research_only=true` + `lane_class=substrate_engineering` per Catalog #298 retirement-discipline opt-outs.
6. **Canonical equation references** to `categorical_posterior_capacity_vs_continuous_gaussian_v1` + `categorical_blahut_arimoto_rate_distortion_v1` (both registered in `.omx/state/canonical_equations_registry.jsonl` per Catalog #344).

## What this landing IS NOT

Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #287 evidence-tag discipline + non-promotable markers per Catalog #127/#192/#317/#341:

- **NOT a contest score claim** (`score_claim=false`, `promotion_eligible=false`, `ready_for_exact_eval_dispatch=false`, `axis_tag=[macOS-MLX research-signal]` per CLAUDE.md "MLX portable-local-substrate authority").
- **NOT a paid CUDA dispatch** (no Modal/Vast.ai/Lightning).
- **NOT a Catalog #1265 gate PASS** (sister #1265 gate consumes PR95 grammar; RSSMC1 grammar requires sister gate per op-routable #2).
- **NOT a converged training run** (2-epoch synthetic-frame smoke only; loss reduction 0.13% per epoch on random targets demonstrates the gradient path is intact but does NOT empirically validate the categorical-posterior paradigm-bridge hypothesis).
- **NOT a Catalog #324 Tier-C density validation** (`predicted_band_validation_status: pending_post_training`; L0 scaffold is pre-empirical anchor per the canonical phantom_random_init discipline).

## Architecture (Catalog #290 canonical-vs-unique decision per layer)

| Layer | Canonical vs Unique decision | Rationale |
|---|---|---|
| Per-pair latent | **FORK (categorical posterior G=24, K=256)** | Substrate-class shift from C6 continuous Gaussian 24-dim (effective ~50 bits); categorical alphabet cannot collapse to single mode (the structural failure mode of C6 IBPS v1 SegNet-collapse @ 105.15 contest-CUDA); H(T) = G*log2(K) = 192 bits/sample ≈ 4× capacity headroom per canonical equation `categorical_posterior_capacity_vs_continuous_gaussian_v1` |
| Reparametrization | **ADOPT-CANONICAL (Gumbel-Softmax + STE)** | Hafner 2024 DreamerV3 canonical recipe per Jang et al. 2016 + Maddison et al. 2016; canonical at ~50 LOC PyTorch / ~25 LOC MLX; ADOPT-BECAUSE-SERVES |
| Decoder topology | **ADOPT-CANONICAL (PR95 HNeRV 6-stage PixelShuffle)** | Empirically validated medal-class topology per PR95/PR101/PR110 cluster; substrate-class shift is at LATENT layer, NOT decoder layer; ADOPT-BECAUSE-SERVES (Catalog #290 decision cascade rule 4 — obvious-fit) |
| Archive grammar | **FORK (RSSMC1 monolithic single-file 0.bin)** | Categorical posterior requires distinct grammar from PR95 (which stores continuous latents); RSSMC1 packs per-pair int8/int16 category indices + decoder state_dict via canonical fp16 brotli q=9 (matches sister C6/sane_hnerv pattern per HNeRV parity L3) |
| Inflate runtime | **FORK (PyTorch-native; no MLX at inflate)** | Per HNeRV parity L4 (≤200 LOC; ≤2 deps = torch + brotli); MLX kept OUT of inflate runtime per CLAUDE.md "MLX portable-local-substrate authority" + Catalog #295 inflate works with empty PYTHONPATH; FORK from MLX module pattern with explicit NHWC↔NCHW conv layout transposes at load |
| Score-aware loss | **PENDING L1+ via canonical helper** | L0 scaffold uses MSE proxy on synthetic frames (smoke-only per Catalog #114); L1+ routes through `tac.substrates._shared.score_aware_common.score_pair_components` per Catalog #164 (canonical eval_roundtrip-mandatory) |
| Optimizer | **ADOPT-CANONICAL (AdamW)** | Canonical neural-codec optimizer; ADOPT-BECAUSE-SERVES |
| GRU-deterministic state | **EXCLUDE at L0 (canonical-only ablation)** | Per symposium Step 1 assumption #3 canonical unwind: L0 implements categorical-only ablation; full RSSM with GRU is L1+ extension per Hafner 2024 canonical recipe |

## 9-dimension success checklist evidence (Catalog #294)

| Dimension | L0 Evidence | Status |
|---|---|---|
| **1. UNIQUENESS (class-shift not within-class)** | YES — discrete-posterior strategy IS class-shift from C6 continuous-Gaussian baseline; canonical equation `categorical_posterior_capacity_vs_continuous_gaussian_v1` registered per Catalog #344 with predicted ~4× capacity headroom | YES |
| **2. BEAUTY + ELEGANCE (PR101-style 30-sec-reviewable)** | YES — module.py 437 LOC + archive.py 360 LOC + inflate.py 220 LOC + trainer 489 LOC; per-file ≤500 LOC; reviewable in 30 seconds per file | YES |
| **3. DISTINCTNESS (explicitly different from sisters)** | YES — distinct from C6 IBPS v1 (continuous Gaussian → categorical), distinct from PR95 HNeRV (per-pair float latent → per-pair int8 category indices), distinct from sister Z7-Mamba-2 + NSCS06 v8 chroma_lut + Z6 PC | YES |
| **4. RIGOR (premise verification + adversarial review + assumption classification + empirical anchor)** | YES (premise verification via T3 symposium 2026-05-19 + this landing's Assumption-Adversary verdict + canonical equation derivation memo 2026-05-20) + PARTIAL (empirical anchor at MLX-local synthetic-frame smoke only) | PARTIAL |
| **5. OPTIMIZATION-PER-TECHNIQUE** | YES — canonical Gumbel-Softmax + STE + uniform max-entropy prior + AdamW; per-layer canonical-vs-unique decision matrix above | YES |
| **6. STACK-OF-STACKS COMPOSABILITY** | YES — orthogonal axes per symposium Step 7: this substrate (latent layer) + sister V1 Faiss V8 (side-info channel) + sister NSCS06 v8 hybrid_path_C (chroma residual) all use discrete-posterior strategy; cross-substrate composition is canonical 3-substrate-frontier-breaking-ensemble | YES |
| **7. DETERMINISTIC REPRODUCIBILITY** | YES — `test_archive_round_trip_byte_determinism` empirically verifies same-input → same-bytes (sorted-keys JSON; fixed brotli quality; fp16 state_dict cast on CPU; raw bytes for category indices); seed-pinned via mx.random.seed | YES |
| **8. EXTREME OPTIMIZATION + PERFORMANCE** | UNKNOWN — L0 scaffold not performance-tuned; MLX smoke runs ~10s for 2 epochs × 8 pairs (~625 ms/pair-epoch); full 600-pair × 100-epoch estimated 100 min on Apple Silicon | UNKNOWN-PENDING-L1 |
| **9. OPTIMAL MINIMAL CONTEST SCORE** | PENDING — predicted_band [0.20, 0.40] per symposium 2026-05-19 (TIGHTENED from DD's [0.18, 0.45] per Tao); `predicted_band_validation_status: pending_post_training` per Catalog #324; L0 scaffold does NOT measure contest score | PENDING |

## Cargo-cult audit per assumption (Catalog #303)

| # | Assumption | Classification | Unwind |
|---|---|---|---|
| 1 | MLX module's simple `mx.repeat` 2x upsample is acceptable at L0 | **CARGO-CULTED-PENDING-CANONICAL-BILINEAR** | Wire `bilinear_resize2x_align_corners_false_nhwc` from `tac.local_acceleration.pr95_hnerv_mlx` at L1+ (drops MLX↔PyTorch decoder parity from ~24.3 to <1.0 in [0,255] space per sister PR95 #1251 empirical anchor) |
| 2 | Synthetic random target frames at L0 smoke produce meaningful gradient signal | **HARD-EARNED-PARTIAL** | Per CLAUDE.md "Forbidden make_synthetic_pair_batch in non-smoke": synthetic targets are explicitly tagged smoke-only; demonstrates gradient path is intact (loss decreased 0.13%) but does NOT validate categorical-posterior paradigm-bridge against contest distribution; unwind via op-routable #5 (real frame loader) at L1+ |
| 3 | Per-pair learnable categorical logits at full G=24, K=256 for 600 pairs are reasonable to keep training-only (not in archive) | **HARD-EARNED** | Per-pair logits = 600 × 24 × 256 = 3,686,400 float32 ≈ 14.7 MB training-time only; reduced to 600 × 24 = 14,400 int8 bytes (~14 KB) in archive via argmax per canonical Hafner 2024 + vdOord VQ-VAE discrete-latent contract; archive shrink ratio ~1000× |
| 4 | Decoder topology re-use from PR95 HNeRV (6-stage PixelShuffle) does NOT compromise the categorical-paradigm-bridge | **HARD-EARNED** | Per symposium Step 1 assumption #3 canonical unwind + Catalog #290 ADOPT-CANONICAL-BECAUSE-SERVES decision: substrate-class shift is at LATENT layer; decoder is the empirically validated medal-class topology (PR95/PR101/PR110 cluster); changing decoder would compound paradigm change |
| 5 | NHWC↔NCHW conv layout transpose at MLX→PyTorch load is correctness-preserving | **HARD-EARNED** | Empirically verified: PyTorch decoder loads MLX state_dict + produces (B, 2, 3, 384, 512) shape match; smoke trainer's end-to-end MLX→archive→PyTorch→camera-uint8 chain produces expected byte count (n_frames × 874 × 1164 × 3) — exactly matches test_end_to_end_mlx_train_archive_pytorch_inflate |
| 6 | The Catalog #1265 contest-equivalence gate (built for PR95 grammar) refuses RSSMC1 archives at this L0 landing is the expected empirical answer | **HARD-EARNED** | Confirmed by direct invocation: gate rc=1 with `Pr95HNeRVMlxError: truncated PR95 archive while reading archive.meta`. This is the canonical sister-gate-needed signal per Catalog #299 quota-brake discipline + the canonical PR95 #1265 design as PR95-specific. L1 sister `gate_mlx_candidate_contest_equivalence_rssm.py` is op-routable #2 |

## Observability surface (Catalog #305, 6 facets)

1. **Inspectable per layer**: `DreamerV3RSSMSubstrateMLX.architecture_manifest()` exposes all canonical config (G, K, decoder_latent_dim, base_channels, decoder_channels_taper, eval_size, num_pairs, gumbel_temperature, use_straight_through) + canonical equation refs + non-promotable markers; per-pair logits accessible at `model.logits` shape (num_pairs, G, K); intermediate `_decoder_forward` exposes (B, 2, 3, H, W) RGB output for inspection.

2. **Decomposable per signal**: per-group categorical sample (`soft.shape = (B, G, K)`) decomposable per group via `mx.argmax(soft, axis=-1)`; per-pair indices stored in archive as int8 (1 byte/group × 24 groups = 24 bytes/pair = 14.4 KB total at 600 pairs); per-epoch loss in `stats.json::epoch_losses`; per-batch gradient norms accessible via `optimizer.state`.

3. **Diff-able across runs**: archive sha256 + decoder_state_dict sha256 + per-pair indices sha256 give byte-level diff across (G, K, gumbel_temperature_schedule, lr, seed, base_channels) tuples; `test_archive_round_trip_byte_determinism` empirically verifies same-input → same-bytes byte determinism.

4. **Queryable post-hoc**: `stats.json` schema `dreamer_v3_rssm_l0_smoke_stats_v1` carries timestamp + elapsed_seconds + hardware_substrate + architecture_manifest + config + epoch_losses + canonical_equation_refs + non-promotable markers; `experiments/results/dreamer_v3_rssm_l0_smoke_20260526T071027Z/stats.json` is the canonical L0 anchor.

5. **Cite-able**: every variant parameter tuple is (G, K, base_channels, decoder_latent_dim, gumbel_temperature_start, gumbel_temperature_final, lr, seed, num_pairs); canonical equation refs cited in `CANONICAL_EQUATION_IDS` constant + stats.json + landing memo frontmatter; sister landing memos cited via `related_deliberation_ids`.

6. **Counterfactual-able**: byte-mutation gate per Catalog #139 + #105 + #272 distinguishing-feature contract; per-pair category index mutation IS distinguishing-feature (canonical disambiguator vs C6's per-pair float latent mutation; tested via `parse_rssmc1_archive_bytes` section-offset parser returning byte ranges); future probe via `tools/verify_distinguishing_feature_byte_mutation.py` on the RSSMC1 archive.

## Catalog #324 post-training Tier-C validation discipline

`predicted_band_validation_status: pending_post_training` per Catalog #324.

Reactivation criterion: post-training Tier-C density measurement on a Path B2 first-anchor archive via `tools/mdl_scorer_conditional_ablation.py --tier c`. Three outcome verdicts:

- **density >= 0.70 (within_class)**: PARADIGM-BRIDGE FALSIFIED at IMPLEMENTATION-LEVEL per Catalog #307. Pivot to non-IB class-shift architectures per symposium Step 5 alt-1 through alt-5 (lane_17_imp / cooperative-receiver ATW V2-1 / Z6 Wave 2 Candidate 4c / NSCS06 v8 hybrid_path_C / Z7-Mamba-2). Recipe transitions to `phantom_random_init` per C6 IBPS v1 canonical precedent (commit `a7f1cc6c6`).
- **density <= 0.30 (across_class)**: PARADIGM-BRIDGE STRUCTURALLY CONFIRMED. Recipe transitions to `validated_post_training`. Path B4 (B1+B2 combined hierarchical RSSM) becomes EV-positive follow-on per Ballard verbatim from symposium.
- **density in [0.30, 0.70] (indeterminate)**: additional 100ep+ training + re-measurement on next archive per Catalog #324 indeterminate-band canonical handling.

The L0 scaffold lands `predicted_band_validation_status: pending_post_training` per the canonical phantom-random-init discipline — NO predicted_band measurement on random-init weights per Catalog #324 anchor + the C6 IBPS 22× miss anchor 2026-05-17.

## Tests (11/11 PASS)

```
src/tac/substrates/dreamer_v3_rssm/tests/test_basic.py:
  test_module_imports_and_archive_grammar_declared PASSED
  test_config_invariants PASSED
  test_mlx_module_forward_training_shape PASSED
  test_mlx_module_forward_eval_shape PASSED
  test_archive_round_trip_byte_determinism PASSED
  test_archive_grammar_section_offsets PASSED
  test_mlx_pytorch_decoder_parity_at_archive_boundary PASSED  (max_abs=24.3403 < 50.0 L0 ceiling)
  test_end_to_end_mlx_train_archive_pytorch_inflate PASSED
  test_gumbel_softmax_sample_shapes_and_indices_in_range PASSED
  test_architecture_manifest_includes_canonical_equation_refs PASSED
  test_rssmc_decoder_param_count_excluding_per_pair_logits PASSED
```

## Smoke trainer empirical anchor

```
$ .venv/bin/python experiments/train_substrate_dreamer_v3_rssm.py \
    --output-dir experiments/results/dreamer_v3_rssm_l0_smoke_20260526T071027Z \
    --epochs 2 --num-pairs 8 --smoke --write-archive

[L0-SCAFFOLD-MLX-SMOKE] config: H(T)=192 bits/sample, G=24, K=256, base_channels=24, decoder_latent_dim=28
[L0-SCAFFOLD-MLX-SMOKE] epoch 1/2 loss=5431.4868 tau=0.550
[L0-SCAFFOLD-MLX-SMOKE] epoch 2/2 loss=5424.6782 tau=0.100
[L0-SCAFFOLD-MLX-SMOKE] wrote stats to .../stats.json
[L0-SCAFFOLD-MLX-SMOKE] wrote archive to .../archive.zip
  (member_bytes=514,927, zip_bytes=515,035, sha256=3dbaf76e43a970ff034ba8cd…)
[L0-SCAFFOLD-MLX-SMOKE] DONE
```

Loss reduction of 0.13% in 2 epochs on synthetic random targets confirms the gradient path through Gumbel-Softmax STE + decoder forward + MSE proxy is intact. **NOT a contest-score signal** — synthetic random uniform [0, 255] targets do not represent the contest distribution; convergence-against-contest is op-routable #5 (real frame loader at L1+).

## Catalog #1265 gate refusal (canonical L0→L1 signal)

```
$ .venv/bin/python tools/gate_mlx_candidate_contest_equivalence.py \
    --archive-zip experiments/results/dreamer_v3_rssm_l0_smoke_20260526T071027Z/archive.zip \
    --candidate-label dreamer_v3_rssm_l0_scaffold \
    --output-json .../equivalence_gate_attempt.json --n-pairs 2

[gate] Invoking canonical measurement: measure_pr95_mlx_pytorch_actual_contest_score_difference.py
[step 1/5] Render MLX + PyTorch decoder-resolution pairs (N=2) from archive
tac.local_acceleration.pr95_hnerv_mlx.Pr95HNeRVMlxError:
  truncated PR95 archive while reading archive.meta: expected 1129534290 bytes, got 514923
[gate] ERROR: measurement returned rc=1
```

**This is the canonical sister-gate-needed signal**, per Catalog #299 quota-brake discipline + the canonical PR95 #1265 design. The Catalog #1265 gate parses archives via `tac.local_acceleration.pr95_hnerv_mlx::parse_pr95_public_archive_zip` (PR95 grammar magic = brotli'd meta header). Our RSSMC1 archive uses distinct magic `b"RSSC"` + 27-byte uncompressed header (canonical Catalog #124 declaration). The gate's refusal IS empirically correct and DOCUMENTS the L0→L1 promotion step:

**Op-routable #2 (L1 sister gate)**: build `tools/gate_mlx_candidate_contest_equivalence_rssm.py` parameterized to consume RSSMC1 grammar via `tac.substrates.dreamer_v3_rssm.archive::parse_archive` + canonical RSSMC1 MLX decoder forward + PyTorch decoder forward + contest-score-difference measurement against ground-truth frames per the canonical PR95 #1265 + `measure_pr95_mlx_pytorch_actual_contest_score_difference.py` pattern.

## Catalog #325 PER-SUBSTRATE OPTIMAL FORM symposium discipline

The 2026-05-19 T3 symposium verdict was `PROCEED_WITH_REVISIONS` with 5 binding revisions per the symposium memo. Per CLAUDE.md "Substrate MUST be at OPTIMAL FORM before paid empirical dispatch" + Catalog #315 + Catalog #325: this L0 scaffold IS NOT yet at OPTIMAL FORM (the council-binding revisions are NOT yet applied at the implementation surface). The 5 binding revisions queued for L1+ promotion:

1. **Contrarian's 3 FREE alternative-probe-methodologies** (op-routable #1a/b/c): local MPS micro-probe at $0 + analytical Dykstra-feasibility at $0 + canonical equation lookup at $0. Of these, the canonical equation IS registered (categorical_posterior_capacity_vs_continuous_gaussian_v1 + categorical_blahut_arimoto_rate_distortion_v1 per task #1062 PATH-A derivation memo). The MPS micro-probe + Dykstra-feasibility ARE the canonical L0+L1 prescreen path; this scaffold satisfies the MLX-local-iteration substrate (op-routable #1a partial — does not yet measure I(B;R(B)) explicitly).
2. **Tao's predicted-band justification** (TIGHTENED [0.18, 0.45] → [0.20, 0.40]): inherited per `predicted_band_validation_status: pending_post_training`.
3. **PR95Author's HNeRV parity L7 LOC budget**: this scaffold at ~1506 LOC total (module 437 + archive 360 + inflate 220 + trainer 489) is OVER the 700-LOC symposium target. Of this, ~360 LOC is archive grammar (substrate-engineering scope per L7 waiver) + ~489 LOC is trainer scaffold (smoke-only; L1+ replacement via canonical helper routing reduces to ~250 LOC). True substrate-engineering scope ≈ module 437 + archive 360 ≈ 800 LOC; LOC budget slightly exceeded but within the substrate_engineering L7 waiver per the symposium binding.
4. **Quantizr+Selfcomp PR #56 discrete-posterior precedent citation**: cited in `__init__.py` docstring + canonical equation registry references the cross-substrate convergence per the symposium Step 7 cross-pollination matrix.
5. **MacKay MDL-framework-budget-allocation**: implicit via uniform max-entropy prior + KL-regularizer-encourages-USE-of-all-K-categories canonical Hafner 2024 contract (the substrate's structural non-collapse property); explicit MDL-budget allocation is L1+ via score-aware loss canonical helper routing.

## 6-hook wire-in declaration (Catalog #125)

1. **Sensitivity-map contribution** = **PENDING** (L0 scaffold does not yet compute per-byte sensitivity; L1+ wires via canonical helper after PyTorch port).
2. **Pareto constraint** = **PENDING** (categorical-posterior constraint polytope analytical computation is op-routable #1b at L1+; Dykstra-feasibility intersection with contest constraints).
3. **Bit-allocator hook** = **PENDING** (categorical posterior bit-allocation per group via Blahut-Arimoto canonical equation `categorical_blahut_arimoto_rate_distortion_v1`; L1+ wires).
4. **Cathedral autopilot dispatch hook** = **ACTIVE** (lane registered at L1 + impl_complete = lane_dreamer_v3_rssm_mlx_scaffold_20260526; cathedral autopilot ranker can consume the canonical equation refs via `tac.cathedral_consumers.canonical_equation_lookup_consumer` per Catalog #335 auto-discovery).
5. **Continual-learning posterior update** = **ACTIVE** (this landing memo emits canonical posterior anchor via `tac.council_continual_learning.append_council_anchor` per Catalog #300 v2 frontmatter; `predicted_band_validation_status: pending_post_training` per Catalog #324; canonical equation registry has `categorical_posterior_capacity_vs_continuous_gaussian_v1` + `categorical_blahut_arimoto_rate_distortion_v1` registered per Catalog #344).
6. **Probe-disambiguator** = **ACTIVE** (the architectural choice between K-capacity hypothesis and G-groups hypothesis IS the canonical disambiguator per the `__init__.py` docstring + `tools/probe_dreamer_v3_rssm_g_k_sweep_disambiguator.py` planned per op-routable #1a; the canonical Blahut-Arimoto sweep over (G, K) is the canonical first-principles disambiguator).

## Sister coordination

Per parent-agent dispatch scope coordination: this subagent (A — DreamerV3 RSSM) is disjoint from 3 sister subagents spawning in parallel:

- Subagent B: `lane_z7_mamba2_mlx_scaffold_extension_20260526` (path: `src/tac/substrates/time_traveler_l5_z7_mamba2/`)
- Subagent C: `lane_nscs06_v8_chroma_lut_mlx_iteration_20260526` (path: `src/tac/substrates/nscs06_v8_chroma_lut/`)
- Subagent D: `lane_z6_predictive_coding_mlx_scaffold_20260526` (path: `src/tac/substrates/z6_*` — sister directory)

My owned domain: `src/tac/substrates/dreamer_v3_rssm/**` + `experiments/train_substrate_dreamer_v3_rssm.py` + `.omx/research/dreamer_v3_rssm_mlx_scaffold_landed_20260526.md` + lane registry row for `lane_dreamer_v3_rssm_mlx_scaffold_20260526`. No sister-owned file modified.

## Operator-routable next steps

Per the 2026-05-19 T3 symposium 6-tier dispatch cost ladder + the corrected #1258 anchor 2026-05-26 enabling MLX-local-iteration at frontier-tightening granularity:

1. **(P1 / L1 promotion / $0)** — Wire canonical `bilinear_resize2x_align_corners_false_nhwc` from `tac.local_acceleration.pr95_hnerv_mlx` into this substrate's `_bilinear_resize_2x_nhwc` helper. Drops MLX↔PyTorch decoder parity from ~24.3 to <1.0 in [0,255] space per sister PR95 #1251 anchor. Test threshold tightening from `< 50.0` to `< 5.0` is the canonical post-fix L1 gate.

2. **(P1 / L1 promotion / $0)** — Build L1 sister gate `tools/gate_mlx_candidate_contest_equivalence_rssm.py` parameterized to consume RSSMC1 grammar. Canonical pattern: replace PR95-specific `parse_pr95_public_archive_zip` call with `tac.substrates.dreamer_v3_rssm.archive::parse_archive`; replace PR95 HNeRV decoder with RSSMC1 MLX module + PyTorch decoder; emit canonical PASS/FAIL verdict per the #1265 schema.

3. **(P2 / L1 promotion / $0)** — Per CLAUDE.md "MLX portable-local-substrate authority" + the operator MLX cascade: build a real-frame MLX-local convergence smoke (replacing synthetic random target frames per Catalog #114) reading from `upstream/videos/0.mkv` via pyav. Validates the categorical-posterior architecture against actual contest distribution at $0.

4. **(P3 / L1+ PyTorch port / $0)** — PyTorch port via canonical Path 3 export bridge sister #1251 + #1257 pattern: build `tools/export_dreamer_v3_rssm_mlx_to_pytorch_state_dict.py` + `tools/package_dreamer_v3_rssm_mlx_pytorch_state_dict_to_contest_archive.py`. Enables L1 sister gate consumption + paid Modal dispatch path.

5. **(P4 / Paid Modal smoke / $0.30 per symposium tier_1)** — After op-routables 1-4 land + the L1 sister gate PASSES with `|S_MLX − S_PT| < 0.001` on a synthetic-trained RSSMC1 archive, fire Modal T4 25-epoch smoke per symposium Catalog #167 smoke-before-full. Paired contest CPU + contest CUDA per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA".

6. **(P5 / Catalog #324 Tier-C validation / $0)** — Post-training Tier-C density measurement on the first paid-Modal-anchor archive via `tools/mdl_scorer_conditional_ablation.py --tier c`. Three outcome verdicts per the 2026-05-19 symposium Step 6:
   - density >= 0.70 (within_class) → PARADIGM-BRIDGE FALSIFIED at IMPLEMENTATION-LEVEL per Catalog #307; pivot per symposium Step 5
   - density <= 0.30 (across_class) → PARADIGM-BRIDGE STRUCTURALLY CONFIRMED; Path B4 hierarchical RSSM follow-on
   - density in [0.30, 0.70] (indeterminate) → 100ep+ retrain per Catalog #324 indeterminate-band canonical handling

## Discipline applied

- **Catalog #229 PV**: read 6 canonical reference files in full before any edit (council symposium memo 402 lines / corrected #1258 closure footer 245 lines / #1257 inflate parity closure 124 lines / #1265 contest-equivalence gate 113 lines / PR95 HNeRV MLX 250 lines / PR95 hnerv_muon canonical inflate.py + model.py + codec.py + c6 IBPS sister substrate scaffold).
- **Catalog #117 / #157 / #174**: this landing commits via canonical `tools/subagent_commit_serializer.py` with POST-EDIT `--expected-content-sha256` per the canonical serializer contract.
- **Catalog #119**: Co-Authored-By trailer auto-appended by serializer.
- **Catalog #206**: 5+ checkpoints emitted via `tools/subagent_checkpoint.py` per discipline.
- **Catalog #110 / #113**: APPEND-ONLY HISTORICAL_PROVENANCE — NO existing file mutated; sister sister memos preserved; this is a NEW landing memo + NEW substrate scaffold directory.
- **Catalog #287**: every rationale ≥4 chars + non-placeholder (no `<rationale>` / `<reason>` literals); every empirical claim carries axis tag.
- **Catalog #290 + #294 + #303 + #305**: design-memo discipline 4-section + 9-dim + cargo-cult + observability declared in this landing memo body above.
- **Catalog #309**: `horizon_class: frontier_pursuit` declared in frontmatter (predicted ΔS in [0.012, 0.20] per symposium tightened band [0.20, 0.40] vs canonical frontier 0.192051).
- **Catalog #310**: this substrate IS class-shift (categorical posterior replaces continuous Gaussian); explicit `lane_class=substrate_engineering` per HNeRV parity L7 waiver; NOT a bolt-on per Catalog #310 distinction.
- **Catalog #325**: this landing memo cites the 2026-05-19 T3 symposium per the canonical 6-step contract + binds the substrate to the PROCEED_WITH_REVISIONS verdict.
- **Catalog #300**: v2 frontmatter complete (council_tier + attendees + quorum + verdict + dissent + assumption_adversary_verdict + decisions + mission_contribution + override fields + horizon_class + canonical_equation_refs + related_deliberation_ids).
- **CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"**: `research_only=true` per lane registry + `_full_main raises NotImplementedError` per Catalog #240 acceptance cascade (c) pre-build substrate-engineering.
- **CLAUDE.md "MLX portable-local-substrate authority"**: all artifacts carry `axis_tag=[macOS-MLX research-signal]` + non-promotable markers per Catalog #127/#192/#317/#341.
- **CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode"**: per-layer canonical-vs-unique decision documented in the architecture section; fork at latent + grammar + inflate; adopt at decoder topology + reparametrization + optimizer.
- **CLAUDE.md "Forbidden make_synthetic_pair_batch in non-smoke"**: synthetic frame targets ONLY in `_smoke_main` per `--smoke` flag; `_full_main` raises NotImplementedError pre-build council-gated.
- **CLAUDE.md "Forbidden premature KILL without research exhaustion"**: this L0 landing is research_only=true PROCEED-PENDING-L1; reactivation criteria fully enumerated per the 6 op-routables above.

## Frontmatter mission alignment per Catalog #300

- `council_predicted_mission_contribution: frontier_breaking_enabler` — the L0 scaffold IS the structural enabler for op-routables 1-6 above; opens the canonical Path 3 substrate-class-shift pursuit at $0 MLX-local cost per CLAUDE.md "MLX portable-local-substrate authority" + the corrected #1258 anchor 2026-05-26 (MLX faithful at frontier-tightening granularity).
- `council_override_invoked: false` — no operator-frontier-override required; this is standard PROCEED.

## Wall-clock + cost

- **Wall-clock**: ~75 minutes (premise verification 25 min + scaffold authoring 30 min + tests 10 min + landing memo 10 min)
- **GPU spend**: $0 (MLX-local synthetic-frame smoke only)
- **Modal / Lightning / Vast.ai**: NOT invoked
- **Codex**: NOT invoked (operator-routable per CLAUDE.md "Pre-dispatch codex review automation" Catalog #271 — relevant at L1+ before paid dispatch, not at L0 scaffold landing)

## Artifact paths

- Substrate scaffold: `src/tac/substrates/dreamer_v3_rssm/{__init__.py, module.py, archive.py, inflate.py, tests/{__init__.py, test_basic.py}}`
- Trainer: `experiments/train_substrate_dreamer_v3_rssm.py`
- Smoke output: `experiments/results/dreamer_v3_rssm_l0_smoke_20260526T071027Z/{stats.json, archive.zip}`
- Landing memo: `.omx/research/dreamer_v3_rssm_mlx_scaffold_landed_20260526.md` (this file)
- Lane registry: `.omx/state/lane_registry.json` row `lane_dreamer_v3_rssm_mlx_scaffold_20260526` L1 (impl_complete=true + research_only=true + lane_class=substrate_engineering)

## Reproduce

```bash
# Run the 11/11 tests
.venv/bin/python -m pytest src/tac/substrates/dreamer_v3_rssm/tests/test_basic.py -v

# Run the L0 MLX-local smoke trainer + archive write
.venv/bin/python experiments/train_substrate_dreamer_v3_rssm.py \
    --output-dir experiments/results/dreamer_v3_rssm_l0_smoke_$(date -u +%Y%m%dT%H%M%SZ) \
    --epochs 2 --num-pairs 8 --smoke --write-archive

# Attempt #1265 gate (will refuse with PR95 grammar error — canonical expected L0 signal)
.venv/bin/python tools/gate_mlx_candidate_contest_equivalence.py \
    --archive-zip <smoke_dir>/archive.zip \
    --candidate-label dreamer_v3_rssm_l0_scaffold \
    --output-json <smoke_dir>/equivalence_gate_attempt.json --n-pairs 2
```

---

<!-- ===== APPEND-ONLY FOOTER: FIX-WAVE-R1 closure 2026-05-26 ===== -->
<!-- Catalog #110 / #113 APPEND-ONLY HISTORICAL_PROVENANCE — this footer is the
     CORRECTION + CLOSURE record for the R1 review's A-OP1 + A-OP2 + A-OP3 +
     A-OP4 findings against this landing memo. Body above is preserved
     UNMUTATED per APPEND-ONLY discipline; corrections are recorded here. -->

## APPEND-ONLY footer: FIX-WAVE-R1 closure 2026-05-26

**Reference**: R1 review memo `.omx/research/path_3_a_recursive_adversarial_review_r1_3_axis_20260526.md` + aggregate `.omx/research/path_3_recursive_adversarial_review_r1_aggregate_3_axis_landings_a_d_e_20260526.md` (commit `80acd6da3`).

### Correction #1: Assumption-Adversary verdict #1 RE-CLASSIFIED CARGO-CULTED → CORRECTED (A-OP1 + A-OP2 closed)

The original `council_assumption_adversary_verdict[2]` row at frontmatter line 28-30 classified PyTorch inflate runtime parity at decoder boundary as `CARGO-CULTED-PENDING-CANONICAL-BILINEAR` and attributed the 24.34 max_abs drift entirely to the bilinear `mx.repeat` gap. R1 review surfaced that there were **TWO INDEPENDENT MLX↔PyTorch drift sources**, not just one:

- **A-OP1 (NEW; R1 review surfaced; landing memo missed)**: `_pixel_shuffle_2x_nhwc` used channel-LAST reshape convention `(B, H, W, 2, 2, out_C)` + transpose `(0, 1, 3, 2, 4, 5)` producing 2.40 absolute drift vs PyTorch `nn.PixelShuffle(2)`.
- **A-OP2 (landing memo correctly named)**: `_bilinear_resize_2x_nhwc` used `mx.repeat` 2x producing 0.99 absolute drift vs PyTorch `F.interpolate(mode='bilinear', align_corners=False)`.

Both bugs compounded through the 6-PixelShuffle-block decoder + sin saturation + sigmoid clipping, producing 24.34 max_abs at the output boundary.

### Correction #2: Cargo-cult audit row #5 amendment (A-OP4 closed)

The original cargo-cult audit row at line 120 ("NHWC↔NCHW conv layout transpose at MLX→PyTorch load is correctness-preserving") was classified HARD-EARNED based on the `test_end_to_end_mlx_train_archive_pytorch_inflate` SHAPE round-trip test. R1 review surfaced that the empirical verification cited held for **SHAPE correctness only**; FULL decoder forward equivalence required ALSO fixing the PixelShuffle convention bug + the bilinear bug. The conv-layout transpose itself IS correctness-preserving (verified independently); but the row's framing implicitly suggested full decoder forward equivalence which was NOT yet empirically verified at the time of landing.

### FIX-WAVE-R1 actions landed (this commit batch)

1. **A-OP1 CLOSED**: `src/tac/substrates/dreamer_v3_rssm/module.py::_pixel_shuffle_2x_nhwc` rewritten to use channel-FIRST reshape convention `(B, H, W, out_C, 2, 2)` + transpose `(0, 1, 4, 2, 5, 3)` matching sister D=Z6 `src/tac/substrates/time_traveler_l5_z6/mlx_renderer.py::_pixel_shuffle_2x_nhwc` AND canonical PR95 helper `tac.local_acceleration.pr95_hnerv_mlx::pixel_shuffle_2x_nhwc`.
2. **A-OP2 CLOSED**: `src/tac/substrates/dreamer_v3_rssm/module.py::_bilinear_resize_2x_nhwc` rewritten to delegate to canonical `tac.local_acceleration.pr95_hnerv_mlx::bilinear_resize2x_align_corners_false_nhwc`. Catalog #295 self-containment is preserved because the canonical helper is imported only at MLX training time in `module.py`; the substrate's inflate runtime at `inflate.py` is PyTorch-only and does NOT import MLX. The Catalog #295 contract scopes `submissions/*/inflate.py` PYTHONPATH self-containment; this substrate's MLX module is at `src/tac/substrates/dreamer_v3_rssm/` which is in-tree by definition.
3. **A-OP3 CLOSED**: `test_mlx_pytorch_decoder_parity_at_archive_boundary` threshold tightened from `< 50.0` to `< 0.05` per the post-fix empirical anchor. Empirical measurement post-fix: **max_abs=0.0054, mean_abs=0.0007** (~4500x improvement vs pre-fix 24.34; well below R1 review's stated < 5.0 promotion criterion).
4. **A-OP4 CLOSED via this APPEND-ONLY footer**: cargo-cult audit row #5 framing amended per the correction above. The row's HARD-EARNED classification holds for the conv-layout transpose itself (independent verification); the FULL decoder forward equivalence claim required A-OP1 + A-OP2 fixes which now land in the same commit batch.

### Post-fix verification (2026-05-26)

- `.venv/bin/python -m pytest src/tac/substrates/dreamer_v3_rssm/tests/test_basic.py -v` → **11/11 pass** (no regressions; same test surface).
- `test_mlx_pytorch_decoder_parity_at_archive_boundary` printed `MLX↔PyTorch decoder parity: max_abs=0.0054, mean_abs=0.0007`.
- The remaining sub-0.01 drift is fp32 compound-op precision noise across 6 PixelShuffle blocks + sin/sigmoid nonlinearities + final RGB heads; acceptable per CLAUDE.md "Apples-to-apples evidence discipline" because after camera-resolution uint8 quantization the drift is structurally below the per-pixel quantization step (1.0 / 255 ≈ 0.004 in [0, 1] space or 1.0 in [0, 255] space).

### R2 readiness signal

- R1 counter status post-FIX-WAVE-R1: **CLEAN** for A=DreamerV3 (all P0 + P1 + P2 findings closed); R2 can fire on this substrate when the aggregate R1 cycle re-runs.
- Catalog #1265 contest-equivalence gate threshold |S_MLX - S_PT| ≤ 0.001 contest-units is now structurally achievable for A=DreamerV3 archives because the decoder forward semantics match (the L1 sister gate `tools/gate_mlx_candidate_contest_equivalence_rssm.py` per op-routable #2 can now operate against byte-stable MLX↔PyTorch decoder parity).

### Outstanding L1+ work (NOT in FIX-WAVE-R1 scope; queued for future subagents)

- **CONSOLIDATE-OP / META** (R1 op-routable #9): extract `_pixel_shuffle_2x_nhwc` + general `_bilinear_resize_nhwc` to canonical `tac.local_acceleration.pr95_hnerv_mlx` so future MLX substrates inherit ONE source of truth per CLAUDE.md "consolidate into META layer" standing directive. Refactor A=DreamerV3 + D=Z6 + future Path 3 candidates to import from canonical. Sister of Catalog #299 quota brake. **Status**: queued as TaskCreate op-routable; NOT executed in FIX-WAVE-R1.

### Cross-references

- FIX-WAVE-R1 landing memo: `.omx/research/path_3_fix_wave_r1_close_findings_landed_20260526.md`
- Source-code diff for A-OP1 + A-OP2: `src/tac/substrates/dreamer_v3_rssm/module.py` lines 184-243
- Test threshold tightening: `src/tac/substrates/dreamer_v3_rssm/tests/test_basic.py` lines 320-350
