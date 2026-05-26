# Canonical equation #344 anchor PROPOSAL — atick_redlich_asymmetric_scorer_channel_lagrangian_routing_savings_v1

- **subagent_id**: `cascade-c-prime-option-a-build-substrate-scaffold-8-prerequisite-artifacts-closes-highest-ev-unclosed-loop-pr111-sub-frontier-candidate-mlx-first-numpy-portable-20260526`
- **lane_id**: `lane_cascade_c_prime_option_a_build_scaffold_20260526`
- **date_utc**: 2026-05-26T20:57:00Z
- **status**: PROPOSED-pending-paired-CUDA-validation per Catalog #344 sister discipline
- **canonical_equation_registry_path**: `.omx/state/canonical_equations_registry.jsonl`
- **registry_growth**: 52 → 53 entries (post-validation registration)
- **FORMALIZATION_PENDING** rationale: pre-empirical synthesis prediction; paired-CUDA validation required per Catalog #324 before registration

<!-- FORMALIZATION_PENDING:proposed_canonical_equation_344_anchor_pending_paired_cuda_validation_per_catalog_324_post_training_tier_c_density_re_measurement_on_landed_paired_smoke_archive_sha -->

## Proposed canonical equation

### `atick_redlich_asymmetric_scorer_channel_lagrangian_routing_savings_v1`

**Mathematical statement**:

Given a per-pair frame-0 mode menu `M_0 = {m_0^1, ..., m_0^{K_0}}` and frame-1 mode menu `M_1 = {m_1^1, ..., m_1^{K_1}}`, the per-pair Lagrangian dual routing decision selects:

```
r_i = argmin_{m ∈ M_0 ∪ M_1} [
    100 × d_seg_i(m)
    + sqrt(10 × max(pose_avg_baseline + d_pose_i(m), eps))
    - sqrt(10 × pose_avg_baseline)
    + 25/37545489 × bytes_per_pair(m)
]
```

**Atick-Redlich asymmetric channel invariant** (1990 cooperative-receiver theorem):
```
d_seg_i(m) = 0 ∀ m ∈ M_0   (SegNet's x[:,-1,...] slice never sees frame-0)
d_seg_i(m) > 0 ∀ m ∈ M_1   (SegNet sees frame-1; pays M_seg cost)
```

**Aggregated score savings** (vs frame-0-only baseline):
```
ΔS_total = -Σ_i [L_frame_0_only_baseline_i - L_joint_menu_argmin_i]
         + 25/37545489 × routing_decision_sidecar_bytes
         + 25/37545489 × menu_index_stream_overhead
```

where `routing_decision_sidecar_bytes ~ 79 bytes` for 600 pairs (Cascade C' synthesis Option B brotli-compressed 1-bit-per-pair packed).

### Predicted value at PR106 frontier operating point

- `pose_avg_baseline = 3.4e-5`
- Cascade C' synthesis (Option B 1-bit sidecar): `ΔS_total = -0.058820` [macOS-MLX research-signal]
- Per-pair frame-1 routing distribution: 25.17% (cross-validates sister #1324 PoseNet-null 22.3% within 3pp)
- 48-cell sensitivity sweep: 41 PARADIGM / 7 MARGINAL / 0 NULL

### Empirical confirmation status

**PENDING-PAIRED-CUDA-VALIDATION** per Catalog #324:

- Synthesis only (rng.gamma() distribution shapes derived from sister #1324 PoseNet-null artifact + Atick-Redlich theory)
- Production paired-CUDA Modal T4 smoke + paired-CPU Linux x86_64 anchor required per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" non-negotiable
- Operator-routable Option A (THIS lane) builds substrate scaffold; 7th-order iteration builds MLX-first trainer + inflate.sh + paired-CUDA dispatch

## Sister equations in registry

Per just-landed canonical-equations-registry framework (commit per `feedback_canonical_equations_and_models_registry_formalization_landed_20260519.md`):

- `procedural_codebook_from_seed_compression_savings_v1` (canonical equation #26 IN-DOMAIN context family) — pure REPLACEMENT savings via LUT replacement (chroma_lut / grayscale_lut / vq_vae)
- `procedural_predictor_plus_residual_correction_savings_v1` (CARGO-CULT FALSIFICATION sister per WAVE-3-MAGIC-CODEC-PAIR-1-2-ENGINEERING-FIX-RE-RUN; residual-correction-hybrid context EXCLUDED from equation #26 per Catalog #359)
- `per_pair_master_gradient_score_impact_taylor_v1` (canonical Taylor expansion + Cauchy-Schwarz per slot 9 sister)
- `master_gradient_locality_violation_by_codec_v1` (per slot 15+17+18 sister findings)

This proposed equation is **DISTINCT from equation #26 family** because:
- Equation #26 family covers REPLACEMENT savings (procedural codebook replaces inline LUT)
- This equation covers ASYMMETRIC CHANNEL ROUTING savings (per-pair Lagrangian dual selects min over joint menu)

## Sister registration tool pattern

Per Catalog #344 sister registration patterns (commits `7ab5f58ae` + `04f34ea40`):

Sister 7th-order subagent will create `tools/register_atick_redlich_asymmetric_scorer_channel_canonical_equation_20260526.py` (pattern of sister `tools/register_*_canonical_equation_*.py` files) which:

1. Imports `tac.canonical_equations.{register_canonical_equation, EmpiricalAnchor, Provenance}`
2. Builds `CanonicalEquation(id="atick_redlich_asymmetric_scorer_channel_lagrangian_routing_savings_v1", ...)` with:
   - `canonical_producers = ["tac.substrates.cascade_c_prime_frame_1_segnet_waterfill.architecture.compute_per_pair_lagrangian_dual_routing"]`
   - `canonical_consumers = ["tac.cathedral_consumers.canonical_equation_lookup_consumer"]` (per Catalog #335)
   - `mathematical_statement = <the Lagrangian formula above>`
   - `Provenance(kind="empirical_paired_cuda", axis_tag="[contest-CUDA T4]", evidence_grade="contest_cuda", ...)`
3. Appends initial `EmpiricalAnchor` from paired-CUDA Modal T4 smoke once it lands
4. Registers via fcntl-locked APPEND-ONLY write per Catalog #131/#138/#245

## Cathedral autopilot consumer wire-in per Catalog #335

When the canonical equation #344 anchor lands, the cathedral autopilot's
`canonical_equation_lookup_consumer` (sister of NSCS06 v8 chroma_lut pattern)
auto-discovers the equation via `tac.cathedral_consumers.canonical_equation_lookup_consumer`
and emits observability-only `[predicted]` annotations on candidate substrates
whose context strings match `atick_redlich_asymmetric_scorer_channel` tokens.

Per Catalog #287/#323 canonical Provenance + Catalog #341 canonical-routing-markers:
the consumer's `consume_candidate(...)` returns all 3 canonical non-promotable markers:
- `predicted_delta_adjustment=0.0` (observability-only)
- `promotable=False`
- `axis_tag="[predicted]"`

## Cross-references

- Pre-execution gate report: `.omx/research/cascade_c_prime_option_a_build_scaffold_pre_execution_gate_report_20260526.md`
- Cascade C' parent synthesis landing memo (commit `2d5337f27`; subagent `aa563bbb31adadfd6`)
- Cascade C' Modal validation DEFERRED verdict (commit `aa1a9cf32`; subagent `a1d16a40f4a722e26`)
- Substrate package: `src/tac/substrates/cascade_c_prime_frame_1_segnet_waterfill/`
- Operator-authorize recipe: `.omx/operator_authorize_recipes/substrate_cascade_c_prime_frame_1_segnet_waterfill_modal_t4_dispatch.yaml`
- Per-substrate symposium memo: `.omx/research/council_t2_cascade_c_prime_frame_1_segnet_waterfill_per_substrate_symposium_20260526.md`
- CLAUDE.md "Canonical equations + models registry — NON-NEGOTIABLE"
- Catalog #344 sister discipline (canonical equation registration + cathedral consumer + FORMALIZATION_PENDING waiver)
- CLAUDE.md "Forbidden premature KILL without research exhaustion" (PENDING-PAIRED-CUDA-VALIDATION ≠ KILL)
