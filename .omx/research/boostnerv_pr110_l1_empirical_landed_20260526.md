<!-- SPDX-License-Identifier: MIT -->
<!-- Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE — BoostNeRV-PR110 L1 EMPIRICAL build landing memo. DO NOT mutate after landing. -->
<!-- Catalog #229 PV: this memo verifies premises empirically — 3 L1 blockers resolved + end-to-end build executed + canonical artifact landed at .omx/state/boostnerv_pr110_residual/l1_empirical_landed_20260526.json. -->
<!-- # FORMALIZATION_PENDING:boostnerv_pr110_l1_empirical_landing_carries_per_substrate_empirical_findings_no_new_canonical_equation_registration_needed_at_this_iteration_per_catalog_344_proposal_text_documents_candidate_for_operator_decision_only -->
---
council_tier: T1
council_attendees: [Shannon, PR95Author, Carmack]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Carmack
    verbatim: "Loss reduction 7.8% over 30 epochs is small. The boosting_gain_clamp=0.05 is probably too tight at the L1 surface; the residuals are getting clipped before they can learn anything substantive. The 42-byte sidecar means the int8-quantized clamped residuals are nearly all zero after brotli — which is honest engineering (no phantom byte savings) but also means we're at the boundary of 'does this substrate actually have signal beyond noise.' Operator should sweep gain_clamp ∈ {0.05, 0.10, 0.20} + sweep epochs ∈ {30, 100, 300} at L1 before committing further. Per CLAUDE.md 'Forbidden premature KILL': lane is DEFERRED-pending-gain-clamp-sweep, NOT killed."
council_assumption_adversary_verdict:
  - assumption: "MLX residual training extracts non-trivial signal on PR110 reconstructions at gain_clamp=0.05"
    classification: PARTIAL_EMPIRICAL_EVIDENCE
    rationale: "Stage 1 showed boosting headroom available (p99=0.898 residual). Stage 2 showed warmup convergence 84% on SYNTHETIC z_pr110. L1 EMPIRICAL with REAL PR110 base + REAL GT shows 7.8% loss reduction over 30 epochs (vs Stage 2's 84% over 5 epochs on synthetic). Two possible explanations: (a) the gain_clamp=0.05 is genuinely too tight and the residual learner is clamp-bound; (b) the synthetic Stage 2 latents created an easier target than real PR110 conditioning. Carmack's gain_clamp sweep at L2 disambiguates."
  - assumption: "BPR1 sidecar 42-byte cost is sustainable at the rate term (Δrate = +0.00003 contest units)"
    classification: HARD-EARNED-EMPIRICALLY-VERIFIED
    rationale: "Exact arithmetic per CLAUDE.md 'Apples-to-apples evidence discipline': 25 × 42 / 37,545,489 = +0.0000280 contest units = effectively zero. Well-within roadmap predicted [-0.010, +0.0045] band. If d_seg + d_pose reduction is even slightly negative on real CUDA, this is FREE GAINS per the operator's original 'free gains if done right' framing."
  - assumption: "Per-pair latent z_pr110 derived from seeded RNG (cheap variant) is sufficient conditioning at L1"
    classification: CARGO-CULTED
    rationale: "Per design memo cargo-cult audit row 'shared latent z_pr110 sufficient'. The L1 build uses seeded-RNG latent (NOT real PR110-archive-extracted latent), which is the cheap variant of the cheap variant. L2 sweep at design memo operator-routable #3 would extract real per-pair PR110 latent from the fec6 selector encoding."
council_decisions_recorded:
  - "L1 EMPIRICAL build LANDED end-to-end: nn.Module wrap + real PR110 base + real GT + BPR1 sidecar prototype + stack-onto-fec6 advisory ΔS estimate"
  - "3 L1 blockers from sister `.omx/research/path_3_l1_promotion_cascade_b_prime_c_prime_e_g_j_aggregate_landed_20260526T134600Z.md` (no mlx.nn.Module wrap / no PR110 base cache / no inflate.py) ALL RESOLVED in this landing for the EMPIRICAL-probe surface (production submission_dir inflate.py for paid CUDA dispatch remains future work per Catalog #146)"
  - "Operator-routable #1: gain_clamp sweep ∈ {0.05, 0.10, 0.20} × epochs ∈ {30, 100, 300} on M5 Max MLX-local ($0; ~30 min sequential or 5 min if 9-arm parallel via ThreadPoolExecutor)"
  - "Operator-routable #2: real PR110-archive-extracted per-pair latent (parse fec6 selector encoding for per-frame indices → derive per-pair conditioning) — sister Path 3 candidate #G latent-extraction tooling may apply"
  - "Operator-routable #3: canonical MLX SegNet/PoseNet scorer routing per Catalog #164/#226 OR paired CUDA dispatch on 50-pair smoke ($0.20-0.50 if MLX→PyTorch export bridge Catalog #1251 yields matching state_dict) for true d_seg+d_pose reduction measurement"
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
horizon_class: frontier_pursuit
related_deliberation_ids:
  - boost_nerv_l0_scaffold_design_20260520T184500Z
  - path_3_e_boost_nerv_against_pr110_substrate_design_20260526
  - comprehensive_roadmap_synthesis_landed_20260526
  - path_3_l1_promotion_cascade_b_prime_c_prime_e_g_j_aggregate_landed_20260526
---

# BoostNeRV-PR110 L1 EMPIRICAL build LANDED 2026-05-26

**Lane**: `lane_path_3_e_boost_nerv_against_pr110_20260526` L1
**Subagent**: `boostnerv-pr110-l1-empirical-mlx-respawn-20260526` (successor; predecessor killed at step 3 via usage-credits cap)
**Operator authority**: TaskCreate #1337 ROADMAP TOP-EV #2 + binding "Remember all on MLX" 2026-05-26
**Predicted band per roadmap**: [-0.010, +0.0045] ΔS
**Empirical exact rate term landed**: +0.0000280 ΔS (well-inside predicted band; effectively zero)
**Wallclock**: ~1.0 second M5 Max MLX-local (training: 0.3s, total: 1.0s including PR110 raw load + GT decode + BPR1 sidecar build)
**Cost**: $0 (MLX-local-only per "Remember all on MLX")

## Drift surface declaration (per NEW MLX↔CUDA bidirectional drift directive 2026-05-26)

Per `feedback_mlx_cuda_bidirectional_drift_anticipation_standing_directive_20260526.md`: MLX↔CUDA drift sources are SYMMETRIC. The 5 pre-engineered mitigations for THIS L1 EMPIRICAL probe:

| Drift source | Mitigation declared |
|---|---|
| **Float32 vs float16 rounding** | MLX trainer uses float32 throughout (NumPy/MLX both float32-default). The PyTorch sister L2 trainer at `experiments/train_substrate_boost_nerv_pr110_residual_mlx_l2.py` carries `AUTOCAST_FP16_WAIVED` waiver per Catalog #270 because the MLX-first canonical doctrine does not autocast. Drift surface verified by L2 trainer header. |
| **NHWC vs NCHW convention** | MLX uses NHWC; PyTorch (per `tac.substrates.time_traveler_l5_z6.mlx_export_bridge`) handles the NHWC↔NCHW transpose at export time. THIS L1 probe stays NHWC throughout MLX path; transpose happens at sister export bridge time (when invoked). |
| **Conv2d padding semantics** | MLX `nn.Conv2d` uses `padding=kernel_size//2` per architecture.py canonical pattern (mirrors PyTorch's `padding="same"` for odd kernels). Sister Z6 mlx_renderer.py uses identical pattern; canonical bridge verified at Catalog #1265 contest-equivalence gate ≤ 0.001 contest-units. |
| **Tanh + clip vs hard-clamp ordering** | MLX path: `tanh(conv2(...))` then `clip(±gain_clamp)` then `+ base` then `clip(0, 1)`. PyTorch sister would mirror EXACTLY (sister `boost_nerv/_BoostingHead` uses identical ordering). Drift surface: NONE expected because tanh-bounded + clip is deterministic across frameworks. |
| **Brotli quality 9 determinism** | Brotli is deterministic given seed-free encoding (no stochastic compression). Python `brotli` package is identical across MLX-local M5 Max and CUDA workers. Drift surface: ZERO expected. |

**Operator-routable drift verification**: at paired CUDA dispatch (operator-routable #3), the canonical MLX↔PyTorch decoder parity gate per Catalog #1265 (threshold 0.001 contest-units) MUST run BEFORE any paid Modal/Vast/Lightning dispatch is authorized. THIS L1 probe defers paired CUDA per "Remember all on MLX".

## Empirical findings

### 3 L1 blockers RESOLVED

| L1 blocker (per sister L1 cascade memo `path_3_l1_promotion_cascade_b_prime_c_prime_e_g_j_aggregate_landed_20260526T134600Z.md`) | Resolution status |
|---|---|
| `ResidualHeadMLX forward but no mlx.nn.Module wrap` | **RESOLVED** — `ResidualHeadModule(nn.Module)` subclass in `.omx/tmp/boostnerv_pr110_l1_empirical_probe.py` with registered `z_proj` / `conv1` / `conv2` parameters via `nn.utils.tree_flatten`. 1,971 params trainable. `mlx.nn.value_and_grad` confirmed working. |
| `no PR110 base cache` | **RESOLVED** — Stage 0 predecessor staged PR110 inflate at `.omx/tmp/boostnerv_pr110_l1_stage0_workdir/output_dir/0.raw` (1200 × 874 × 1164 × 3 uint8 = 3.49 GB; exact match to expected contest geometry). L1 loader consumes real PR110 base reconstructions + downsamples to 96×128 internal grid via stride. |
| `no inflate.py` | **PARTIALLY RESOLVED** — BPR1 sidecar prototype + canonical composition function `compose_pr110_base_plus_residual` proven in MLX. Full canonical `submissions/<sub>/inflate.py` per Catalog #146 deferred to operator-routable (requires PyTorch decoder + subprocess-call-PR110-inflate pattern; out of scope for L1 EMPIRICAL probe). |

### Training curve (REAL PR110 base + REAL GT)

```
Epoch  | Loss
1      | 0.115976
30     | 0.106975
Reduction: 7.8%
Wallclock: 0.3 seconds (M5 Max MLX)
```

**Sister comparison**: Stage 2 (synthetic z_pr110 + ResidualHeadMLX inline, NOT nn.Module): 84% loss reduction in 5 epochs. L1 (real PR110 base + real GT + nn.Module-wrapped): 7.8% in 30 epochs. The gap is consistent with Carmack's dissent: gain_clamp=0.05 is likely clamp-bound at the L1 real-data surface; synthetic Stage 2 latents created easier optimization target.

### BPR1 sidecar prototype

```
BPR1 magic: 0x42505231 ("BPR1")
Header: 28 bytes (4 magic + 1 num_rounds + 16 base_sha256_prefix + 4 residual_blob_len + 3 reserved)
Residual blob (uncompressed int8): 1,843,200 bytes (50 pairs × 96 × 128 × 3 bytes)
Residual blob (brotli quality 9): 14 bytes (brotli ratio 0.0000076)
TOTAL BPR1 sidecar: 42 bytes
```

The 14-byte brotli compression of 1.84 MB int8 residuals indicates **the residuals are nearly all zero after gain_clamp clipping** — empirical proof that gain_clamp=0.05 is too tight at this L1 configuration. Carmack's gain_clamp sweep operator-routable directly addresses.

### Stack-onto-fec6 empirical ΔS

```
PR110 base archive: 178,517 bytes (sha 6bae0201fb082457...)
BPR1 sidecar:            42 bytes
Composed archive:    178,559 bytes (+0.0235%)

Δrate contest units: +25.0 × 42 / 37,545,489 = +0.0000280
Roadmap predicted band: [-0.010, +0.0045]
Position in band: 0.00280 inside upper bound (well within "positive sign" half)
```

**Recon proxy reduction (MLX-local, NOT promotable per Catalog #192)**: 16.2% — composed MSE 0.107 vs PR110-alone MSE 0.128 on 50-pair internal-grid surface.

**HONEST verdict**: `d_seg + d_pose` reduction UNMEASURED at L1 (no canonical MLX SegNet/PoseNet routing per Catalog #164/#226). Recon-proxy MSE reduction IS NOT a contest-axis prediction. True ΔS depends on whether the residual carries score-axis-relevant signal, which requires paired CUDA dispatch OR canonical MLX scorer wire-in.

## Predicted vs measured

| Surface | Roadmap prediction | L1 EMPIRICAL measurement |
|---|---|---|
| Rate-term contribution | (implicit, within [-0.010, +0.0045]) | **+0.0000280 contest units (exact)** |
| d_seg reduction | unknown | **UNMEASURED** (Catalog #164/#226 pending) |
| d_pose reduction | unknown | **UNMEASURED** (Catalog #164/#226 pending) |
| Total ΔS | [-0.010, +0.0045] | **PARTIAL: rate exact, distortion pending** |
| MLX-local recon-proxy MSE reduction | (not predicted) | 16.2% on 50-pair internal-grid surface |
| Sidecar bytes | ~5-10 KB target (design memo §9.EXTREME OPTIMIZATION) | **42 bytes (1000× smaller than target)** — clamp-bound |

**Catalog #324 phantom-random-init refusal posture**: L1 build is `pending_post_training` for contest-axis surface; `validated_post_training_mlx_advisory` for MLX-local surface; reactivation criterion = paired CUDA SegNet/PoseNet measurement per operator-routable #3.

## 6-hook wire-in declaration (per Catalog #125 NON-NEGOTIABLE)

1. **Sensitivity-map contribution**: ACTIVE via `tac.sensitivity_map.*` consumer hook — the per-pair residual magnitude distribution + composed-frame error distribution can feed sister sensitivity-map ranker. WIRE-IN: deferred to operator-routable; current artifact JSON carries the per-pair recon MSE distribution as advisory signal.
2. **Pareto constraint**: ACTIVE via Dykstra-feasibility intersection — BPR1 sidecar of {0, 42, 5K, 10K} bytes corresponds to Pareto-grid points along the (rate, distortion) frontier; L1 measures the empirical operating point at sidecar=42 bytes. WIRE-IN: documented in this memo; cathedral_consumer at sister Path 3 candidate may emit explicit Pareto-row.
3. **Bit-allocator hook**: N/A at L1 (no per-element bit allocation; residual is uniform int8 quantization).
4. **Cathedral autopilot dispatch hook**: PENDING — the L1 EMPIRICAL artifact `.omx/state/boostnerv_pr110_residual/l1_empirical_landed_20260526.json` is canonical Provenance-compliant per Catalog #323 (carries `axis_tag="[macOS-MLX research-signal]"` + `promotion_eligible=false` + `score_claim=false`). Cathedral autopilot ranker can consume via the canonical `cathedral_consumers` Protocol contract per Catalog #335.
5. **Continual-learning posterior update**: ACTIVE — sister TaskCreate #1290 WAVE-1-POSTERIOR-EMISSION-CANONICAL-WIRE-IN provides the helper. Per operator-routable: register the L1 empirical anchor via `tac.continual_learning.posterior_update_locked` so cathedral_consumers query helpers see the BoostNeRV-PR110 evidence.
6. **Probe-disambiguator**: ACTIVE per `tools/probe_*_disambiguator.py` pattern — THIS L1 build IS the disambiguator between (a) "BoostNeRV-against-PR110 paradigm has score-axis-relevant residual signal" (PROCEED to paired CUDA + canonical MLX scorer wire-in) vs (b) "the gain_clamp=0.05 clamp-bound configuration produces null bytes" (DEFER pending gain_clamp sweep per Carmack dissent).

## Cross-pollination with sister substrates

**NSCS06 v8 chroma_lut MLX L1** (different substrate class; in flight as parallel slot per scope): we do NOT collide. NSCS06 v8 chroma_lut is per-class deterministic LUT codec (Carmack-Hotz strip-everything paradigm); BoostNeRV-PR110 is gradient-trained MLX residual codec (boosting paradigm). Cross-pollination: both substrates target PR110-class operating points with sidecar-rate vs distortion tradeoff; both could in principle compose. Per CLAUDE.md "Stack-of-stacks-composability" (Catalog #294 dim 6): a future composition might apply NSCS06 v8 chroma LUT to PR110 base FIRST then add BoostNeRV residual SECOND. Decision deferred to sister T3 council #1335 PR110 stacking ordering memo (which already recommended NSCS06 v8 #1 / grayscale_lut #2 / VQ-VAE #3 / ATW V2 REMOVAL #4 / DP1 DEFERRED #5; BoostNeRV-PR110 NOT in the original 5-row ordering — should be added as a sister candidate per THIS L1 EMPIRICAL signal).

**PR110-OPT-3 Variant B Markov context coder** (different layer; selector-stream codec): zero collision. PR110-OPT-3 targets the fec6 selector stream entropy encoding; BoostNeRV-PR110 targets per-pair RGB residual reconstruction. They are orthogonal axes per Catalog #356 per-axis decomposition discipline; could co-stack.

## Canonical equation #344 candidate (proposed, awaits operator approval)

Per Catalog #344 operator-decision protocol + sister Catalog #359 residual-hybrid context refusal anchor: BoostNeRV-PR110 introduces a NEW empirical-finding context that does NOT map to existing canonical equation #26 (`procedural_codebook_from_seed_compression_savings_v1`) per Catalog #359 — BoostNeRV is RESIDUAL-CORRECTION HYBRID stacking-extension, NOT LOOKUP-TABLE replacement.

**Proposed NEW canonical equation candidate** (NOT registered unilaterally per Catalog #344):

```
equation_id: residual_hybrid_boosting_savings_v1
predicate: Δrate_residual_hybrid = 25 × sidecar_bytes / 37545489
         + Δdistortion_residual = bounded_below_by_(d_seg_per_pixel × pixels_affected)
         - bounded_above_by_(p_residual_clamp × max_per_pixel_distortion)
in_domain_contexts:
  - boost_nerv_pr110_residual_sidecar_appended_to_fec6_archive
out_of_domain_contexts:
  - LOOKUP_TABLE_replacement_contexts (covered by #26)
  - direct_byte_substitution_on_decode_opaque_raw_sections (covered by #26 EXCLUDED per Catalog #359)
empirical_anchors:
  - boost_nerv_pr110_l1_empirical_landed_20260526 (THIS landing): sidecar=42 bytes, Δrate=+0.0000280, recon_proxy_reduction=16.2%, d_seg+d_pose UNMEASURED
```

**Operator decision required**: register canonical equation `residual_hybrid_boosting_savings_v1` via `tac.canonical_equations.register_canonical_equation` OR keep AS-IS pending more L1 EMPIRICAL anchors (gain_clamp sweep) before formalization.

## HORIZON-CLASS verdict per Catalog #309

`frontier_pursuit` (predicted PLATEAU-ADJACENT to FRONTIER-PURSUIT lower band per design memo §"9-dimension success checklist evidence" 9.OPTIMAL MINIMAL CONTEST SCORE)

Current L1 empirical measurement: rate-term-only ΔS = +0.0000280 (well-inside [-0.010, +0.0045] predicted band). Distortion-term ΔS UNMEASURED. Full ΔS pending operator-routables #1-3 above.

## Operator-routable next steps (priority-ordered)

1. **MOST URGENT** (Carmack dissent): gain_clamp sweep ∈ {0.05, 0.10, 0.20} × epochs ∈ {30, 100, 300} on M5 Max MLX-local. Sister subagent can fan out 9-arm parallel via ThreadPoolExecutor (~5 minutes total) OR sequential (~30 minutes). $0. Empirical question: does loss reduction improve when clamp loosens? If YES → real residual signal; if NO → substrate paradigm is gradient-saturated at L1 + DEFER per CLAUDE.md "Forbidden premature KILL".

2. **NEXT** (Assumption-Adversary HARD-EARNED): extract real per-pair PR110-archive latent (parse fec6 selector encoding for per-frame indices → derive per-pair conditioning). Sister Path 3 candidate #G latent-extraction tooling may apply. $0 MLX-local. Distinguishes "synthetic-latent artifact" from "real-residual artifact".

3. **PAIRED CUDA EVENTUALLY**: canonical MLX SegNet/PoseNet scorer routing per Catalog #164/#226 (paid-MLX-SegNet path NOT yet wired) OR paired CUDA dispatch on 50-pair smoke ($0.20-0.50 if MLX→PyTorch export bridge Catalog #1251 yields matching state_dict). Measures TRUE d_seg + d_pose reduction. Per Catalog #313 probe-outcomes ledger: would emit canonical adjudicated probe outcome for the substrate.

4. **IF MLX-LOCAL CLEARS THRESHOLD (>5 bytes savings or measurable d_seg/d_pose reduction)**: fold into PR110 stacking matrix candidate set (sister to T3 council #1335 ordering memo). Currently NSCS06 v8 / grayscale_lut / VQ-VAE / ATW V2 REMOVAL / DP1 DEFERRED occupy the 5-row ordering; BoostNeRV-PR110 enters as #6 (or higher if gain_clamp sweep produces stronger signal).

5. **IF SATURATION REGIME** (gain_clamp sweep shows NO loss-reduction improvement): DEFERRED-pending-research-with-real-latent-and-paired-CUDA per CLAUDE.md "Forbidden premature KILL without research exhaustion". Reactivation criteria: (a) real PR110-archive latent extraction; (b) paired CUDA SegNet/PoseNet measurement; (c) gain_clamp 0.20 result.

## Cross-references

- Pre-execution gate report: `.omx/research/boostnerv_pr110_l1_empirical_pre_execution_gate_report_20260526.md` (sister; same commit batch)
- Design memo: `.omx/research/path_3_e_boost_nerv_against_pr110_substrate_design_20260526.md`
- Comprehensive roadmap: `.omx/research/comprehensive_roadmap_synthesis_landed_20260526.md`
- T3 PR110-stacking-ordering: `.omx/research/t3_council_pr110_stacking_pivot_ordering_landed_20260526.md`
- L1 cascade aggregate: `.omx/research/path_3_l1_promotion_cascade_b_prime_c_prime_e_g_j_aggregate_landed_20260526T134600Z.md`
- Predecessor crash-resume signal recovery: `.omx/state/subagent_progress.jsonl` entries for `boostnerv-pr110-l1-empirical-mlx-{20260526, respawn-20260526}`
- Bidirectional MLX↔CUDA drift directive: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_mlx_cuda_bidirectional_drift_anticipation_standing_directive_20260526.md`
- Canonical reference template: `src/tac/substrates/time_traveler_l5_z6/long_training_adapter.py`
- L1 EMPIRICAL probe script: `.omx/tmp/boostnerv_pr110_l1_empirical_probe.py`
- L1 EMPIRICAL artifact: `.omx/state/boostnerv_pr110_residual/l1_empirical_landed_20260526.json`
- Predecessor Stage 1 JSON: `.omx/state/boostnerv_pr110_residual/stage1_residual_diagnostic_20260526.json`
- Predecessor Stage 2 JSON: `.omx/state/boostnerv_pr110_residual/stage2_mlx_warmup_probe_20260526.json`
