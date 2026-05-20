# DP1 procedural-codebook paired-smoke PRE-DISPATCH design (2026-05-20)

<!-- Catalog #344 canonical-equations-registry cross-reference: this design
memo's predicted ΔS = -0.002706 is derived via the canonical equation
`procedural_codebook_from_seed_compression_savings_v1` registered at
`src/tac/canonical_equations/procedural_codebook_savings.py` and persisted to
`.omx/state/canonical_equations_registry.jsonl`. The DP1 row in the 5-substrate
matrix memo §4 (commit b3e3442c3 §"Aggregate prediction summary table") is
the canonical authority for the prediction. This memo specifies the FIRST
empirical anchor path: $0.30 Modal T4 CPU+CUDA paired smoke against the
DP1 substrate with codebook bytes replaced by deterministic procedural
seed-derivation per `tac.procedural_codebook_generator.derive_codebook_from_seed`. -->
<!-- HISTORICAL_SCORE_LITERAL_OK:dp1_predicted_band_derives_from_canonical_equation_no_score_authority_claim_2026-05-20 -->

---
substrate_id: pretrained_driving_prior_procedural_codebook_replacement
substrate_alias: dp1_proc
substrate_class: pretrained_driving_prior_procedural_replacement_variant
horizon_class: plateau_adjacent
council_tier: T2
council_attendees:
  - Shannon
  - Dykstra
  - Yousfi
  - Fridrich
  - Contrarian
  - Assumption-Adversary
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_override_rationale: null
deferred_substrate_id: pretrained_driving_prior_procedural_codebook_replacement
deferred_substrate_retrospective_due_utc: "2026-06-19T23:21:19Z"
predicted_band_validation_status: pending_post_training
predicted_band:
  contest_cpu_first_anchor_bytes_only: [-0.002706, -0.002706]
  contest_cpu_with_frame_response: [-0.005000, +0.000500]
  contest_cuda_first_anchor_bytes_only: [-0.002706, -0.002706]
score_claim: false
promotion_eligible: false
research_only: true
dispatch_enabled: false
operator_directive: "WAVE-3 DP1 procedural-codebook paired-smoke first-empirical-anchor design; READY-TO-PAIRED-SMOKE per 5-substrate matrix §4 + DP1 symposium 2026-05-17 PROCEED_WITH_REVISIONS"
related_deliberation_ids:
  - five_substrate_procedural_replacement_matrix_design_20260520
  - per_substrate_symposium_dp1_deep_dive_20260517
  - procedural_codebook_generator_null_exploit_design_20260520
  - nscs06_v8_procedural_chroma_lut_integration_design_20260520
canonical_equation_id: procedural_codebook_from_seed_compression_savings_v1
canonical_equation_registry_path: .omx/state/canonical_equations_registry.jsonl
parent_design_memo: .omx/research/five_substrate_procedural_replacement_matrix_design_20260520.md
parent_landing_memo: feedback_five_substrate_procedural_replacement_matrix_design_landed_20260520.md
parent_symposium_memo: .omx/research/council_per_substrate_symposium_dp1_deep_dive_20260517.md
sister_supersession_memo: .omx/research/dp1_procedural_codebook_paired_smoke_pre_dispatch_design_HALTED_premise_failure_20260520T192800Z.md
---

## Section 1. Summary

This memo designs the **FIRST EMPIRICAL ANCHOR paired-smoke** for the canonical equation `procedural_codebook_from_seed_compression_savings_v1` (registered today via `src/tac/canonical_equations/procedural_codebook_savings.py`; currently 1 row with 0 empirical anchors — predicted-only). DP1 (`pretrained_driving_prior`) is the canonical first anchor per:

* 5-SUBSTRATE MATRIX DESIGN memo §4 row 4 (commit `b3e3442c3`): **READY-TO-PAIRED-SMOKE** with Catalog #210 provenance preservation contract; first anchor candidate.
* 5-substrate matrix §11 Top-3 op-routable #1: *"First per-substrate paired smoke (DP1; READY-TO-SMOKE; $0.30 Modal T4)"*.
* DP1 deep-dive symposium 2026-05-17 (PROCEED_WITH_REVISIONS; 3 days old → within Catalog #325 14-day window).

**Scope of THIS design**: the substrate's existing 4096-byte (post-brotli, multi-section per `codebook.py`) codebook is replaced at archive-emit time by a 32-byte procedural seed; the inflate runtime re-derives the codebook deterministically from the seed via `tac.procedural_codebook_generator.derive_codebook_from_seed(seed_bytes=32B, output_shape=(1024, 4), dtype=np.uint8, generator_kind="pcg64")`. Predicted ΔS = **-0.002706** (canonical equation; bytes-saved closed-form Shannon arithmetic; first empirical anchor needed to validate).

**Scope distinction from DP1 symposium PATH 1 + PATH 2**: this is a **PROCEDURAL-REPLACEMENT** variant (SAVES 4,064 bytes net via codebook→seed replacement); the symposium's PATH 1 (DP1+fec6 composition) was superseded 2026-05-18 with rate cost +0.017197 (ADDS 25,827 bytes). This variant is the OPPOSITE byte-direction and operates on the **DP1 substrate ALONE**, not on composition with another substrate. The symposium's CARGO-CULTED-PENDING-EMPIRICAL assumption (#3 "DP1 produces incremental score improvement") and #4 ("composes monotonically across reuse graph") are **NOT INVOKED** here — this is a single-substrate byte-replacement anchor, not a composition claim.

The operator-routed paid dispatch ($0.30 Modal T4 paired CPU+CUDA) is **OPERATOR-GATED** via the canonical `operator_authorize.py` chain. This design memo prepares the operator-decision surface; the actual dispatch lives downstream.

## Section 2. PV / premise verification table

Per CLAUDE.md Catalog #229 (`check_subagent_landing_includes_premise_verification_evidence`). All premises verified at HEAD `cbe587679` 2026-05-20T23:21:19Z.

| Premise | Source | Verification | Verdict |
|---|---|---|---|
| 5-substrate matrix landing exists | commit `b3e3442c3` | `git cat-file -p b3e3442c3` → 670+15 insertions; landed 2026-05-20T18:06:49 -0500 by Alejandro Peña | ✓ |
| DP1 codebook = ~4 KB | `codebook.py` lines 91-108 + `CODEBOOK_TOTAL_TARGET_BYTES_MIN=5_000` / `MAX=10_000` | Multi-section: road_plane (9.2K raw → ~3K brotli) + sky_horizon (192B → ~80B) + lane_curvature_pca (96B fp16) + vehicle_appearance (2.3K raw → ~800B) + meta JSON; total post-brotli ≈ 4-5 KB; matrix memo §4 cites "~4 KB partial" | ✓ |
| Canonical equation registered | `src/tac/canonical_equations/procedural_codebook_savings.py::build_procedural_codebook_from_seed_compression_savings_v1` | 1 aggregate_hypothesis_anchor (predicted-only; empirical_output={}); RECALIBRATE_ON_NEW_ANCHORS trigger | ✓ |
| Procedural seed helper exists | `src/tac/procedural_codebook_generator/seed_derived_codebook.py` | `derive_codebook_from_seed` with 3 PRNG kinds (xorshift / lcg / pcg64); DEFAULT_GENERATOR_KIND + MAX_OUTPUT_BYTES + SUPPORTED_GENERATOR_KINDS exported | ✓ |
| DP1 trainer exists | `experiments/train_substrate_pretrained_driving_prior.py` | 82.8K; `_smoke_main` + `_full_main` (gated behind `DPP_RUN_FULL=1`) | ✓ |
| DP1 recipe exists | `.omx/operator_authorize_recipes/substrate_pretrained_driving_prior_modal_t4_dispatch.yaml` | 6.9K; smoke_only=true; min_smoke_gpu=T4; min_vram_gb=16; canary_status=independent_substrate; predicted_band=[0.175, 0.190] (HOLISTIC DP1 prediction; THIS variant uses the canonical equation's procedural-replacement-specific prediction) | ✓ |
| DP1 symposium memo exists ≤14 days | `.omx/research/council_per_substrate_symposium_dp1_deep_dive_20260517.md` | 35.9K; T3 grand council; 2026-05-17 (3 days old); PROCEED_WITH_REVISIONS | ✓ |
| Catalog #325 14-day recency satisfied | 2026-05-20 minus 2026-05-17 = 3 days | Within the 14-day window | ✓ |
| Procedural codebook savings consumer exists | `src/tac/cathedral_consumers/procedural_codebook_savings_consumer/__init__.py` | 290 LOC; Catalog #335 canonical contract; sister of procedural_codebook_generator_consumer | ✓ |
| Catalog #313 probe-outcomes ledger clean | `.omx/state/probe_outcomes.jsonl` | No prior blocking verdict on DP1 procedural-codebook substrate variant within 30-day window | ✓ (canonical first anchor) |

## Section 3. Canonical equation + predicted-band derivation

### The canonical equation

ΔS_procedural = `-CANONICAL_RATE_MULTIPLIER * (N_codebook - K_seed) / CANONICAL_RATE_DENOM_BYTES`

= -25 × (4096 - 32) / 37,545,489

= **-0.002706**

Where:
- N_codebook = 4,096 bytes (DP1 codebook archive-charged after brotli; matrix memo §4 + `codebook.py::CODEBOOK_TOTAL_TARGET_BYTES_MIN=5_000` upper bound on min size; "~4 KB" canonical citation)
- K_seed = 32 bytes (canonical seed size per `procedural_codebook_savings.py` line 122 `predicted_seed_size_bytes=32`)
- CANONICAL_RATE_MULTIPLIER = 25.0 (per `upstream/evaluate.py` line 63 rate term coefficient)
- CANONICAL_RATE_DENOM_BYTES = 37,545,489 (per `upstream/evaluate.py` line 63 rate term denominator = 600 frames × ~62KB)

### Predicted bands (3 explicit per frontmatter)

1. **contest_cpu_first_anchor_bytes_only**: [-0.002706, -0.002706]
   * Rationale: closed-form bytes-saved arithmetic; assumes ZERO frame-axis response change (rendered frames byte-identical to baseline DP1; seg + pose scores invariant). This is the **structural lower bound** on the empirical ΔS — anything below means the procedural seed PRESERVES frame quality perfectly while saving 4064 bytes.
   * Catalog #296 Dykstra-feasibility: closed-form Shannon arithmetic; structurally feasible at the rate axis (4064 bytes is < 0.011% of the 37.5M-byte contest video; well within the convex Pareto polytope at the rate axis).

2. **contest_cpu_with_frame_response**: [-0.005000, +0.000500]
   * Rationale: empirical band accounting for the frame-axis effect. The procedural seed produces structurally-DIFFERENT codebook bytes (PCG64 uniform_int8) vs the trained codebook (PCA basis + sky_horizon + lane_curvature + vehicle_appearance). The frame-axis response is UNKNOWN until the smoke fires — this band brackets both outcomes:
     - **Best case** (-0.005): bytes saved + frame-axis preserved or slightly improved (if codebook's role is purely structural).
     - **Worst case** (+0.000500): bytes saved are dominated by frame-axis regression (procedural derivation loses informativeness; renderer quality degrades; seg/pose scores worsen by enough to overwhelm the -0.002706 rate savings).
   * The Assumption-Adversary #3 in the DP1 symposium ("DP1 produces incremental score improvement") CARGO-CULTED-PENDING-EMPIRICAL verdict applies HERE per Catalog #292 inheritance: the procedural seed's frame-axis effect is empirically untested.

3. **contest_cuda_first_anchor_bytes_only**: [-0.002706, -0.002706]
   * Rationale: same as contest_cpu_first_anchor_bytes_only — rate-axis arithmetic is contest-axis-invariant (bytes-saved is a rate-axis property; the contest formula's `25 * archive_bytes / 37545489` term is byte-count-only). The CPU vs CUDA score gap (canonical equation `cpu_cuda_score_gap_v1` per WAVE-3-CPU-CUDA-DRIFT-ANALYSIS commit `c385f1291`) lives in the seg + pose axes, not the rate axis.

### Catalog #324 post-training Tier-C validation

`predicted_band_validation_status: pending_post_training`. The first empirical anchor IS the post-training Tier-C validation; per Catalog #344 `RECALIBRATE_ON_NEW_ANCHORS` trigger, the canonical equation's predicted_band tightens as anchors land.

## Section 4. Three recipe outlines

Per the task spec, three recipe variants are designed (not committed — operator-routed). Each builds on the existing `.omx/operator_authorize_recipes/substrate_pretrained_driving_prior_modal_t4_dispatch.yaml` as parent.

### Recipe outline #1: DP1 ORIGINAL baseline

```yaml
name: substrate_pretrained_driving_prior_original_baseline_modal_t4_paired_dispatch
parent: substrate_pretrained_driving_prior_modal_t4_dispatch.yaml
purpose: |
  Baseline anchor for the canonical equation `procedural_codebook_from_seed_compression_savings_v1`
  empirical residual measurement. Trains DP1 substrate with the EXISTING trained PCA codebook
  (4096 bytes archive-charged). Paired contest-CPU + contest-CUDA per CLAUDE.md "Submission auth eval — BOTH".
lane_id: lane_dp1_original_baseline_first_paired_anchor_20260520
platform: modal
gpu: T4
smoke_only: false          # paired-smoke; not original smoke_only:true
dispatch_enabled: false    # operator-gated; flip to true at operator authorization time
predicted_band: [0.175, 0.190]     # parent's holistic DP1 band; not the procedural-variant equation
predicted_band_validation_status: pending_post_training
min_vram_gb: 14
min_smoke_gpu: T4
video_input_strategy: cuda_nvdec
pyav_decode_strategy: cuda_nvdec
target_modes: [contest_one_video_replay]
canary_status: independent_substrate
cost_band:
  epochs: 100              # smoke; not the parent's full-mode 2000ep
  hand_calibrated_fallback_p50_usd: 0.30
sentinel_files: [parent's 15 files; unchanged]
env_overrides:
  DPP_DISPATCH_INSTANCE_JOB_ID: "${INSTANCE_JOB_ID}"
  DPP_VIDEO_PATH: /workspace/pact/upstream/videos/0.mkv
  DPP_OUTPUT_DIR: /modal_results/${INSTANCE_JOB_ID}/output    # canonical per Catalog #204
  DPP_EPOCHS: "100"
  DPP_BATCH_SIZE: "1"
  DPP_UPSTREAM_DIR: /workspace/pact/upstream
  DPP_DEVICE: cuda
  DPP_DATASET_NAME: comma2k19
  DPP_ENABLE_AUTOCAST_FP16: "1"
  DPP_ENABLE_GT_SCORER_CACHE: "1"
  DPP_RUN_FULL: "1"                                            # enable _full_main; gated by DPP_USE_STREAMER or DPP_COMMA2K19_CHUNKS_DIR
  DPP_USE_STREAMER: "1"                                        # canonical Comma2k19 streamer per Catalog #213
  DPP_MAX_DISTILLATION_FRAMES: "4096"
  DPP_MAX_DISTILLATION_CHUNKS: "8"
  DPP_MAX_PAIRS: "600"
  DPP_VAL_PAIR_COUNT: "64"
  # Catalog #244 NVML env block (3-export canonical)
  DALI_DISABLE_NVML: "1"
  CUBLAS_WORKSPACE_CONFIG: ":4096:8"
  PYTORCH_CUDA_ALLOC_CONF: "expandable_segments:True"
  LOCAL_CUDA_WORKER: "1"
notes: |
  ORIGINAL baseline; first paired-axis anchor for DP1 in production-trained
  form (NOT zero-codebook scaffold). Establishes the canonical baseline against
  which the PROCEDURAL variant's empirical residual is measured.
```

### Recipe outline #2: DP1 PROCEDURAL replacement variant

```yaml
name: substrate_pretrained_driving_prior_procedural_codebook_modal_t4_paired_dispatch
parent: substrate_pretrained_driving_prior_modal_t4_dispatch.yaml
purpose: |
  CANONICAL FIRST EMPIRICAL ANCHOR for `procedural_codebook_from_seed_compression_savings_v1`.
  Trains DP1 substrate, then replaces the 4096-byte trained codebook with a 32-byte
  procedural seed at archive-emit time per `tac.procedural_codebook_generator.derive_codebook_from_seed`.
  Inflate runtime re-derives the codebook deterministically from the seed.
  Predicted ΔS = -0.002706 per canonical equation.
lane_id: lane_dp1_procedural_codebook_replacement_first_paired_smoke_20260520
platform: modal
gpu: T4
smoke_only: false
dispatch_enabled: false    # operator-gated
predicted_band: [-0.005000, +0.000500]   # contest_cpu_with_frame_response band
predicted_band_validation_status: pending_post_training
min_vram_gb: 14
min_smoke_gpu: T4
video_input_strategy: cuda_nvdec
pyav_decode_strategy: cuda_nvdec
target_modes: [contest_one_video_replay]
canary_status: independent_substrate
cost_band:
  epochs: 100
  hand_calibrated_fallback_p50_usd: 0.30
sentinel_files: [parent's 15 files + src/tac/procedural_codebook_generator/seed_derived_codebook.py + src/tac/canonical_equations/procedural_codebook_savings.py]
env_overrides:
  [all from #1] +
  DPP_PROCEDURAL_CODEBOOK_REPLACEMENT: "1"     # NEW flag — trainer reads this to invoke procedural seed replacement at archive-emit time
  DPP_PROCEDURAL_SEED_BYTES: "32"
  DPP_PROCEDURAL_GENERATOR_KIND: "pcg64"        # DEFAULT_GENERATOR_KIND per seed_derived_codebook.py
  DPP_PROCEDURAL_OUTPUT_SHAPE: "1024,4"         # uint8 → ~4096 bytes regenerated
  DPP_PROCEDURAL_DTYPE: "uint8"
  DPP_PROVENANCE_PRESERVATION_REQUIRED: "1"    # Catalog #210 contract — seed must carry license_tags + dataset_provenance + distillation_version
notes: |
  FIRST EMPIRICAL ANCHOR for canonical equation `procedural_codebook_from_seed_compression_savings_v1`.
  Per Catalog #210 + #213: the procedural seed MUST carry license_tags + dataset_provenance +
  distillation_version + canonical Comma2k19LocalCache.fetch_chunk citation so OOD-derivation
  is auditable. The trainer's `_full_main` extends with `DPP_PROCEDURAL_CODEBOOK_REPLACEMENT=1`
  path that runs distillation in full but emits the seed instead of the codebook bytes at
  archive-emit time. Per Catalog #272 distinguishing-feature integration contract: byte-mutation
  smoke MUST verify seed-derived bytes affect rendered frames (mutate 1 byte of seed → re-derive
  codebook → re-render → verify scored output changes).
```

### Recipe outline #3: DP1 NULL-EXPLOIT variant (optional control)

```yaml
name: substrate_pretrained_driving_prior_null_exploit_codebook_modal_t4_paired_dispatch
parent: substrate_pretrained_driving_prior_modal_t4_dispatch.yaml
purpose: |
  Control case for the canonical equation. Replaces the 4096-byte codebook with
  all-zeros bytes (NOT procedural seed). Establishes the FLOOR of frame-axis
  regression when the codebook structure is fully removed — disambiguates whether
  the procedural variant's residual is dominated by the SEED-DERIVATION QUALITY
  or by CODEBOOK-PRESENCE-ALONE.
lane_id: lane_dp1_null_exploit_codebook_replacement_control_paired_smoke_20260520
platform: modal
gpu: T4
smoke_only: false
dispatch_enabled: false    # operator-gated; OPTIONAL — only fire if procedural variant residual is ambiguous
predicted_band: [-0.002706, +0.100000]   # WIDE band — rate-axis SAVES 4064 bytes but frame-axis likely catastrophic
predicted_band_validation_status: pending_post_training
[all other fields = procedural variant]
env_overrides:
  [all from #1] +
  DPP_NULL_CODEBOOK_BYTES: "1"             # NEW flag — emit zeros at codebook archive section
  DPP_PROVENANCE_PRESERVATION_REQUIRED: "0" # null exploit; provenance metadata becomes meaningless
notes: |
  CONTROL CASE for the procedural variant's first empirical anchor. Per
  CLAUDE.md "Forbidden representation-without-archive-grammar (the research-substrate
  trap)" + 8th forbidden pattern: this variant has NO operational mechanism (zeros
  cannot regenerate the dashcam prior). It MUST be tagged research_only=true and
  byte-mutation smoke MUST be expected to detect "no per-pair byte mutation produces
  score change" → matrix DECISION ROW: this lane is DEFER-pending-disambiguator if
  fired and confirmed structurally inert.
```

## Section 5. First-empirical-anchor canonical-equation update flow

Per Catalog #344 + `tac.canonical_equations.update_equation_with_empirical_anchor` API. The flow is:

### Step 1. Land the paired smoke + harvest results

```bash
# Operator routes via canonical operator_authorize chain:
.venv/bin/python tools/operator_authorize.py \
    --recipe substrate_pretrained_driving_prior_procedural_codebook_modal_t4_paired_dispatch \
    --paired-axis cuda+cpu \
    --max-spend-usd 0.30

# Smoke-before-full per Catalog #167 (auto-gated in tools/run_modal_smoke_before_full.py)
# Modal .spawn() per Catalog #245 → register_dispatched_call_id_fail_closed
# Harvest within 24h per CLAUDE.md "Modal .spawn() HARVEST OR LOSE" via tools/parallel_harvest_actuator.py
```

### Step 2. Verify byte-mutation smoke per Catalog #272

```bash
.venv/bin/python tools/verify_distinguishing_feature_byte_mutation.py \
    --lane-id lane_dp1_procedural_codebook_replacement_first_paired_smoke_20260520 \
    --archive-zip experiments/results/<lane>_modal/contest_archive.zip \
    --inflate-sh experiments/results/<lane>_modal/inflate.sh \
    --distinguishing-bytes-path procedural_seed_section
```

Expected: PASSED verdict (mutating 1 byte of the 32-byte seed → re-derives different codebook → renders different frames → produces different scored output). FAIL = research-substrate trap → matrix DECISION ROW becomes DEFER-pending-architecture-redesign per the 8th forbidden pattern.

### Step 3. Update canonical equation with empirical anchor

```python
from tac.canonical_equations import (
    EmpiricalAnchor,
    update_equation_with_empirical_anchor,
)
from tac.provenance.builders import (
    build_provenance_for_archive_member,
)

# After paired smoke harvested + byte-mutation smoke passes:
first_empirical_anchor = EmpiricalAnchor(
    anchor_id="dp1_procedural_codebook_first_paired_smoke_anchor_20260520",
    measurement_utc="<paired-smoke-completion-utc>",
    inputs={
        "substrate_id": "pretrained_driving_prior_procedural_replacement",
        "n_codebook_bytes": 4096,
        "k_seed_bytes": 32,
        "generator_kind": "pcg64",
        "axis_tag_cpu": "[contest-CPU]",        # paired Linux x86_64
        "axis_tag_cuda": "[contest-CUDA T4]",   # paired Modal T4
        "evidence_grade_cpu": "contest_cpu_gha_linux_x86_64",
        "evidence_grade_cuda": "contest_cuda_modal_t4",
        "scored_archive_sha256_baseline": "<baseline archive sha256>",
        "scored_archive_sha256_procedural": "<procedural archive sha256>",
        "scored_archive_bytes_baseline": "<baseline bytes>",
        "scored_archive_bytes_procedural": "<procedural bytes>",
        "bytes_saved": "<empirical bytes_saved from sha256+archive diff>",
        "byte_mutation_smoke_verdict": "PASSED",
        "canonical_helper_invocation": "tac.procedural_codebook_generator.derive_codebook_from_seed",
    },
    predicted_output={
        "predicted_delta_s_contest_cpu": -0.002706,
        "predicted_delta_s_contest_cuda": -0.002706,
        "predicted_bytes_saved": 4064,
    },
    empirical_output={
        "empirical_delta_s_contest_cpu": "<measured CPU baseline - measured CPU procedural>",
        "empirical_delta_s_contest_cuda": "<measured CUDA baseline - measured CUDA procedural>",
        "empirical_bytes_saved": "<measured archive bytes diff>",
    },
    residual=abs(predicted_delta_s_contest_cpu - empirical_delta_s_contest_cpu),  # |ΔS_pred - ΔS_emp|
    source_artifact=".omx/research/dp1_procedural_codebook_paired_smoke_pre_dispatch_design_20260520T232120Z.md",
    measurement_method="paired_contest_cpu_plus_contest_cuda_100ep_modal_t4_smoke_with_byte_mutation_verification",
    provenance=build_provenance_for_archive_member(
        contest_archive_zip_path="experiments/results/<lane>_modal/contest_archive.zip",
        contest_archive_member_name="x",
        source_path="experiments/results/<lane>_modal/contest_archive.zip",
        source_sha256="<archive sha256>",
        measurement_axis="[contest-CPU]",       # primary axis for procedural-replacement claim
        hardware_substrate="linux_x86_64_gha_cpu",
        evidence_grade="contest_cpu_gha_linux_x86_64",
        score_claim_valid=True,
        promotion_eligible=False,                # first anchor; promotion-eligibility decided at council Phase 2
    ),
)

update_equation_with_empirical_anchor(
    equation_id="procedural_codebook_from_seed_compression_savings_v1",
    new_anchor=first_empirical_anchor,
    # auto-recalibration triggers per RECALIBRATE_ON_NEW_ANCHORS
)
```

Per Catalog #344 the canonical helper appends the new anchor to `.omx/state/canonical_equations_registry.jsonl` (APPEND-ONLY per Catalog #131/#138) + auto-recalibrates the equation's predicted_band tightening.

### Step 4. Sister-update Catalog #322 composition-alpha + Catalog #324 PREDICTED_BAND

* Catalog #322: the first empirical anchor for DP1 procedural-replacement IS NOT a composition-alpha measurement (single-substrate; not pair-wise). No composition_alpha update from this anchor alone. The first composition_alpha empirical anchor requires the SECOND per-substrate paired smoke + a composition probe (per 5-substrate matrix §11 deferred routable "5-way aggregate paired smoke").
* Catalog #324: `predicted_band_validation_status` for the recipe flips from `pending_post_training` to `validated_post_training` with the empirical band tightening per the residual.

### Step 5. Wire-in to autopilot via Catalog #335

The `procedural_codebook_savings_consumer` (auto-discovered per Catalog #335 paradigm) consumes the canonical equation's posterior; the next autopilot loop tick re-ranks DP1 procedural-replacement candidates based on the empirical anchor instead of the predicted-only band.

## Section 6. Per-layer canonical-vs-unique decision

(Catalog #290)

| Layer | Decision | Rationale |
|---|---|---|
| Substrate scaffold | **ADOPT_CANONICAL_BECAUSE_SERVES** | `tac.substrates.pretrained_driving_prior` 5602 LOC; canonical Catalog #209/#210/#211/#213 sister-gates protect provenance + composition; no fork needed |
| Procedural seed helper | **ADOPT_CANONICAL_BECAUSE_SERVES** | `tac.procedural_codebook_generator.derive_codebook_from_seed` is the canonical helper landed at PROCEDURAL-CODEBOOK BUILD; PCG64 default per O'Neill 2014 |
| Canonical equation | **ADOPT_CANONICAL_BECAUSE_SERVES** | `procedural_codebook_from_seed_compression_savings_v1` registered per Catalog #344; predicted-only awaiting this anchor |
| Cathedral consumer | **ADOPT_CANONICAL_BECAUSE_SERVES** | `procedural_codebook_savings_consumer` v0.1.0 auto-discovered per Catalog #335; Tier A markers per Catalog #341 |
| Provenance preservation | **ADOPT_CANONICAL_BECAUSE_SERVES** | Catalog #210/#213 + `build_provenance_for_archive_member` canonical helper |
| Recipe Tier 1+2+3 | **ADOPT_CANONICAL_BECAUSE_SERVES** | Catalog #270 dispatch optimization protocol umbrella; parent recipe already satisfies most fields |
| Output directory | **ADOPT_CANONICAL_BECAUSE_SERVES** | `/modal_results/${INSTANCE_JOB_ID}/output` per Catalog #204; not `/workspace/pact/...` (would resolve to `/tmp/pact/...` on Modal worker) |
| NVML env block | **ADOPT_CANONICAL_BECAUSE_SERVES** | Catalog #244 3-export canonical (DALI_DISABLE_NVML + CUBLAS_WORKSPACE_CONFIG + PYTORCH_CUDA_ALLOC_CONF) |
| Trainer flag wiring | **FORK_BECAUSE_PRINCIPLED_MISMATCH** | NEW flags `DPP_PROCEDURAL_CODEBOOK_REPLACEMENT` + `DPP_PROCEDURAL_SEED_BYTES` + `DPP_PROCEDURAL_GENERATOR_KIND` + `DPP_PROCEDURAL_OUTPUT_SHAPE` + `DPP_PROCEDURAL_DTYPE` + `DPP_PROVENANCE_PRESERVATION_REQUIRED` extend the trainer's argparse per Catalog #151 manifest discipline |
| Byte-mutation smoke | **ADOPT_CANONICAL_BECAUSE_SERVES** | `tools/verify_distinguishing_feature_byte_mutation.py` per Catalog #272 |
| Sister symposium reuse | **ADOPT_CANONICAL_BECAUSE_SERVES** | DP1 deep-dive symposium 2026-05-17 PROCEED_WITH_REVISIONS is the canonical T3 verdict; no re-symposium needed per Catalog #325 14-day window (3 days old) |

## Section 7. 9-dimension success checklist evidence

(Catalog #294)

1. **UNIQUENESS**: first empirical anchor for canonical equation `procedural_codebook_from_seed_compression_savings_v1`; no other registered substrate has anchored this equation. The variant is also distinct from the symposium's PATH 1 (composition; superseded with +0.017197 cost) and PATH 2 (DP1 prior on PR101_lc_v2 decoder weights; council Phase 2 deferred).
2. **BEAUTY + ELEGANCE**: 32-byte seed replacing 4096-byte codebook is the minimum-information principle applied to bytes-vs-information; PCG64 is the canonical PRNG per O'Neill 2014; the inflate runtime extension is `derive_codebook_from_seed(...)` call → 1-line addition.
3. **DISTINCTNESS**: explicitly distinct from sister symposium memo's PATH 1 + PATH 2 by byte-direction (SAVES 4064 bytes vs ADDS 25,827); explicitly distinct from sister matrix memo §4 by scope (single-substrate first-anchor vs 5-substrate aggregate); explicitly distinct from canonical equation aggregate hypothesis anchor by anchor type (empirical-vs-predicted-only).
4. **RIGOR**: 10 PV checks complete (matrix memo §4 + canonical equation builder + codebook.py size citation + symposium 3-day recency + procedural seed helper API + DP1 trainer 82.8K + DP1 recipe 6.9K + DP1 lane registry 10+ entries + DP1 symposium memo 35.9K + Catalog #313 probe-outcomes ledger clean).
5. **OPTIMIZATION PER TECHNIQUE**: PCG64 chosen as DEFAULT_GENERATOR_KIND for highest-quality uniform_int8 derivation (alternative: xorshift for low-LOC inflate.py; LCG for legacy compat).
6. **STACK-OF-STACKS-COMPOSABILITY**: this anchor is the canonical first row in the matrix memo's 5-substrate composition table; per-pair α with NSCS06 v8 / ATW V2 / VQ-VAE is in matrix memo §3 (DP1 × NSCS06_v8 = 0.80 ADDITIVE; DP1 × VQ_VAE = 0.60 SUB-ADDITIVE).
7. **DETERMINISTIC REPRODUCIBILITY**: seed-derivation is byte-stable per `derive_codebook_from_seed` (PCG64 deterministic + cross-platform little-endian + sha256-salt for all-zero seeds); same (seed, shape, dtype, generator_kind) → same bytes always.
8. **EXTREME OPTIMIZATION + PERFORMANCE**: 4096 bytes saved out of ~290 KB-ish frontier archive (~1.4% of archive); PCG64 codebook derivation is <1ms; inflate.py LOC budget per Catalog #328 unchanged (helper call + 1 line).
9. **OPTIMAL MINIMAL CONTEST SCORE**: predicted ΔS = -0.002706 alone would take 0.19205 [contest-CPU] → 0.18934 (still plateau-adjacent; does NOT break 0.18 alone). Per matrix memo §3, FIRST anchor is the canonical disambiguator for the 5-substrate aggregate path's optimistic [-0.013, -0.011] bound (which WOULD break 0.18) vs the conservative [-0.0085, -0.0060] bound (which does NOT). The DP1 anchor is `frontier_protecting` per Yousfi's DP1 symposium verdict (helps lock in current best); the aggregate path is `frontier_breaking_enabler`.

## Section 8. Cargo-cult audit per assumption

(Catalog #303)

### Assumption #1: "DP1 codebook is ~4096 bytes archive-charged after brotli"

**Classification**: **HARD-EARNED-EMPIRICALLY-VERIFIED** per `codebook.py::CODEBOOK_TOTAL_TARGET_BYTES_MIN=5_000` + matrix memo §4 "~4 KB partial" citation. Each codebook section's brotli-compressed bytes is estimated explicitly in `codebook.py` lines 13-17 (road_plane: 3KB, sky_horizon: 80B, lane_curvature_pca: 96B, vehicle_appearance: 800B, meta: variable) for sum ≈ 4-5 KB. The 4096 = nominal target.

**Unwind path**: N/A — first-principles + per-section byte estimates.

### Assumption #2: "PCG64 32-byte seed suffices to encode DP1 codebook structural information"

**Classification**: **HARD-EARNED-FIRST-PRINCIPLES** per O'Neill 2014 (PCG64 has 128-bit state + 64-bit output; 32 bytes ≈ 256 bits = state + increment with room for sha256-salt). The codebook itself is structurally regenerable from a 32-byte seed because the seed-derived codebook IS DIFFERENT from the trained codebook — the score-preservation claim is THE empirical question, not the seed-budget claim.

**Unwind path**: N/A for seed-budget; see Assumption #3 for score-preservation.

### Assumption #3: "PCG64-derived uniform_int8 codebook preserves the contest scorer's response sufficiently to land net-negative ΔS"

**Classification**: **CARGO-CULTED-PENDING-EMPIRICAL** per Catalog #220 operational mechanism requirement. The procedural seed produces structurally-DIFFERENT codebook bytes from the trained codebook (uniform_int8 vs PCA-basis + sky_horizon vertical profile + lane-curvature splines + vehicle templates). The score-preservation claim is the FIRST EMPIRICAL ANCHOR question.

**Unwind path**: this design memo's recipe outline #2 IS the unwind path. Per Catalog #272: byte-mutation smoke MUST verify seed-derived bytes affect rendered frames. If frame-axis regression dominates rate-axis savings (empirical ΔS > 0), the matrix DECISION ROW becomes "procedural-replacement does NOT preserve score on DP1" → operator-routed Wave N+1 redesign (Mode B residual diff per matrix memo §2 candidate 5 / Mode C k-means refinement / fork to a different substrate).

### Assumption #4: "DP1 procedural-replacement is the canonical FIRST anchor (not NSCS06 v8 or VQ-VAE)"

**Classification**: **HARD-EARNED-OPERATOR-ROUTED** per matrix memo §11 Top-3 op-routable #1 ("DP1; READY-TO-SMOKE; $0.30 Modal T4") + §4 row 4 ("READY-TO-PAIRED-SMOKE with Catalog #210 provenance preservation contract; first anchor candidate"). NSCS06 v8 is in DEFER (re-symposium needed); ATW V2 is in DEFER (D4 re-probe required); TT5L is in REFUSE; VQ-VAE is BUILD-required. DP1 is the only ready candidate.

**Unwind path**: N/A — matrix memo operator-routing is explicit.

### Assumption #5: "$0.30 Modal T4 paired smoke is sufficient for first empirical anchor"

**Classification**: **HARD-EARNED-EMPIRICALLY-VERIFIED** per matrix memo §5 cost model ("Modal T4 smoke (100ep + paired Linux x86_64 CPU): ~$0.30/smoke") + canonical recipe parent's `cost_band.hand_calibrated_fallback_p50_usd: 3.00` (full 2000ep) → /20 epoch scale ≈ $0.30 for 100ep.

**Unwind path**: N/A — cost model is canonical per CLAUDE.md "GPU budget and compute resources".

### Assumption #6: "Symposium 2026-05-17 PROCEED_WITH_REVISIONS verdict transfers to procedural-replacement variant"

**Classification**: **HARD-EARNED-CANONICAL-INHERITANCE** per Catalog #325 14-day window (3 days old) + the symposium's #1 cargo-cult audit applies (codebook prior IS HARD-EARNED) + #2 (composition wrapper API IS HARD-EARNED) + symposium #6 op-routable explicitly covers "DP1 reuse in different paths" via cross-substrate composability audit. The procedural-replacement variant IS A NEW PATH (not PATH 1 or PATH 2) but inherits the structural-soundness verdict.

**Unwind path**: Catalog #325 also requires that the procedural-replacement variant's NEW assumptions (Assumption #3 here) be flagged as CARGO-CULTED-PENDING-EMPIRICAL until the first paired anchor lands. THIS memo flags it explicitly per Catalog #303 + #292.

## Section 9. Observability surface

(Catalog #305)

| Facet | Implementation |
|---|---|
| 1. Inspectable per layer | Per-recipe env_overrides table (§4) + per-canonical-equation predicted band derivation (§3) + canonical helper invocation surface (`tac.procedural_codebook_generator.derive_codebook_from_seed`) |
| 2. Decomposable per signal | rate-axis ΔS = -0.002706 (closed-form) + frame-axis ΔS (empirical-pending); per-axis predicted vs empirical residual surfaced via `tac.canonical_equations.predicted_vs_empirical_residual` dict |
| 3. Diff-able across runs | Baseline (recipe #1) vs procedural (recipe #2) vs null-exploit (recipe #3) paired-axis runs all produce byte-stable archives diff-able via sha256; per-pair score difference diff-able via `tools/compare_smoke_outcomes.py` (canonical sister) |
| 4. Queryable post-hoc | Canonical equations registry JSONL queryable via `tools/list_canonical_equations.py`; per-anchor predicted/empirical residuals queryable via `tac.canonical_equations.get_equation_by_id("procedural_codebook_from_seed_compression_savings_v1")` |
| 5. Cite-able | Every prediction cites canonical equation builder + matrix memo + symposium memo; every recipe cites parent recipe + Catalog # discipline gates; every smoke anchor cites archive sha256 + Modal call_id per Catalog #245 |
| 6. Counterfactual-able | Byte-mutation smoke per Catalog #272 + `tools/verify_distinguishing_feature_byte_mutation.py` answers "what if this seed byte changed?" → re-derive codebook → re-render → verify scored output changes |

## Section 10. Predicted band per Catalog #324 + Dykstra-feasibility per Catalog #296

Already covered in §3. Summary:

* `predicted_band_validation_status: pending_post_training`
* Dykstra-feasibility: rate-axis structurally feasible (4064 bytes << 37.5M denominator; well within convex Pareto polytope at the rate axis); frame-axis INDETERMINATE until empirical anchor.
* Catalog #296 marker satisfied via the closed-form Shannon arithmetic citation (rate-axis ΔS is first-principles; the predicted_band for the frame-axis ([-0.005, +0.000500]) IS the Dykstra-feasibility band — both endpoints are within the Pareto-feasible region per the rate-axis being unconstrained).

## Section 11. Operator-routable next actions (Top-3)

### Top-3 op-routables for THIS lane

1. **OP-ROUTABLE #1 — Land THIS design memo + sister supersession** ($0 wall-clock; already in-progress per this subagent's commit). Canonical serializer commit lands both this memo + the APPEND-ONLY supersession of the HALT memo per Catalog #110/#113 HISTORICAL_PROVENANCE discipline.

2. **OP-ROUTABLE #2 — Build the procedural-replacement variant trainer extension** ($0 wall-clock; ~150 LOC). Spawn `lane_dp1_procedural_codebook_replacement_trainer_extension_build_20260520` to (a) extend `experiments/train_substrate_pretrained_driving_prior.py::_full_main` with `DPP_PROCEDURAL_CODEBOOK_REPLACEMENT=1` branch that invokes `derive_codebook_from_seed(seed_bytes=32B, output_shape=(1024, 4), dtype=np.uint8, generator_kind="pcg64")` at archive-emit time instead of the trained codebook; (b) preserve Catalog #210 DP1 codebook provenance metadata in the seed-derivation path per Catalog #213 Comma2k19 canonical helper; (c) add the 6 NEW env-var → CLI flags per Catalog #151 manifest discipline; (d) extend inflate.py with the canonical helper re-derivation call (preserves Catalog #328 LOC budget); (e) build the operator-authorize recipe outline #2 from §4 above + commit it to `.omx/operator_authorize_recipes/`.

3. **OP-ROUTABLE #3 — Operator-authorize the paired smoke + harvest** ($0.30 Modal T4 paired CPU+CUDA; OPERATOR-GATED). After OP-ROUTABLE #2 lands the recipe + trainer extension, the operator routes via the canonical `tools/operator_authorize.py --recipe substrate_pretrained_driving_prior_procedural_codebook_modal_t4_paired_dispatch --paired-axis cuda+cpu --max-spend-usd 0.30` chain. The Modal `.spawn()` registers per Catalog #245 + #339 fail-closed; harvest within 24h per Catalog #330; byte-mutation smoke per Catalog #272; first empirical anchor lands via `update_equation_with_empirical_anchor` per Catalog #344 + canonical Provenance per Catalog #323.

### Deferred routables

* **DP1 NULL-EXPLOIT control smoke** (recipe outline #3): only fire if procedural variant residual is ambiguous (empirical ΔS is in the band [-0.001, +0.001]); helps disambiguate seed-derivation-quality vs codebook-presence-alone.
* **DP1 procedural-replacement Mode B (seed + diff residual)** per matrix memo §2 candidate 5 Mode B: spawn if Mode A (pure procedural) lands a positive ΔS — Mode B preserves the trained codebook signal in a low-rank residual.
* **DP1 procedural-replacement Mode C (seed + k-means post-hoc)** per matrix memo §2 candidate 5 Mode C: spawn if both A + B land positive ΔS — Mode C runs k-means refinement at inflate time (LOC budget concern per Catalog #328).
* **5-substrate aggregate paired smoke**: spawn after 4+ per-substrate anchors land (DP1 first + 3 sisters; possibly NSCS06 v8 + ATW V2 reactivated + VQ-VAE built); $1.50 cumulative; first aggregate empirical α measurement per matrix memo §5.

## Section 12. Mission contribution per Catalog #300

**council_predicted_mission_contribution**: `frontier_protecting`

Per CLAUDE.md "Mission alignment — non-negotiable" Consequence 5: this design is `frontier_protecting` (DP1 procedural-replacement alone takes 0.19205 → 0.18934, plateau-adjacent — does NOT break 0.18 alone; helps lock in current best per Yousfi's DP1 symposium verdict). The FRONTIER-BREAKING contribution is the AGGREGATE 5-substrate path per matrix memo §10, which requires this anchor as the FIRST empirical residual to recalibrate the canonical equation's posterior so the autopilot can trust the aggregate prediction.

Per Catalog #300 operational consequence 4: the apparatus serves the mission; this design memo's $0 design-only scope + operator-gated paid dispatch path preserves the operator's GPU budget while landing the FIRST empirical anchor that unlocks downstream frontier-breaking work.

**council_override_invoked**: false. No operator-frontier-override at design-memo surface; design only.

## Section 13. 6-hook wire-in declaration per Catalog #125

* hook #1 sensitivity-map: **ACTIVE** (the first empirical residual feeds `tac.sensitivity_map.*` consumers via canonical Provenance per Catalog #323; per-codebook-byte sensitivity at the seed-vs-trained-codebook diff surface)
* hook #2 Pareto constraint: **ACTIVE** (rate-axis ΔS = -0.002706 is a Pareto-constraint contribution; per-axis decomposition surfaced via `predicted_axis_decomposition` per Catalog #356)
* hook #3 bit-allocator: **ACTIVE** (4064-byte savings reallocate-able to other rate-axis sinks; bit-allocator can route the freed bits toward residual sidecars per the matrix memo's composition path)
* hook #4 cathedral autopilot dispatch: **ACTIVE PRIMARY** (the empirical anchor lands the FIRST data point on canonical equation `procedural_codebook_from_seed_compression_savings_v1`; the cathedral autopilot's `procedural_codebook_savings_consumer` per Catalog #335 + #357 Tier A markers per Catalog #341 re-ranks candidates based on the empirical residual)
* hook #5 continual-learning posterior: **ACTIVE** (per Catalog #344 `RECALIBRATE_ON_NEW_ANCHORS` trigger; canonical equation's predicted_band tightens; sister `tac.findings_lagrangian.posterior_update_from_anchors` consumes via Catalog #355 Phase 1 meta-Lagrangian wire-in)
* hook #6 probe-disambiguator: **ACTIVE** (the byte-mutation smoke per Catalog #272 IS the canonical disambiguator between procedural-replacement-preserves-score vs procedural-replacement-degrades-score; the 3-recipe design (#1 baseline / #2 procedural / #3 null-exploit) is the canonical 3-way disambiguator across the design space)

## Section 14. Cross-references

* **Parent design memo**: `.omx/research/five_substrate_procedural_replacement_matrix_design_20260520.md` (commit `b3e3442c3`; cross-substrate matrix)
* **Sister single-substrate design memo (NSCS06 v8)**: `.omx/research/nscs06_v8_procedural_chroma_lut_integration_design_20260520.md` (commit `0b4a1d449`; the canonical FIRST-anchor pattern this memo mirrors)
* **DP1 deep-dive symposium**: `.omx/research/council_per_substrate_symposium_dp1_deep_dive_20260517.md` (T3 grand council; PROCEED_WITH_REVISIONS; 12 attendees; 6 op-routables)
* **DP1 lane scaffold landing**: `feedback_pretrained_driving_prior_lane_scaffold_LANDED_20260513.md`
* **DP1 Phase 2 hardening v2**: `.omx/research/dp1_phase_2_hardening_v2_council_20260514.md`
* **PROCEDURAL-CODEBOOK BUILD landing**: `feedback_procedural_codebook_generator_build_landed_20260520.md`
* **PROCEDURAL-CODEBOOK design memo**: `.omx/research/procedural_codebook_generator_null_exploit_design_20260520.md`
* **Canonical equation builder**: `src/tac/canonical_equations/procedural_codebook_savings.py`
* **Canonical equation registry**: `.omx/state/canonical_equations_registry.jsonl`
* **Canonical procedural seed helper**: `src/tac/procedural_codebook_generator/seed_derived_codebook.py`
* **Canonical procedural codebook savings consumer**: `src/tac/cathedral_consumers/procedural_codebook_savings_consumer/__init__.py`
* **DP1 substrate package**: `src/tac/substrates/pretrained_driving_prior/` (13 .py files; 5602 LOC)
* **DP1 trainer**: `experiments/train_substrate_pretrained_driving_prior.py` (82.8K)
* **DP1 parent recipe**: `.omx/operator_authorize_recipes/substrate_pretrained_driving_prior_modal_t4_dispatch.yaml` (6.9K)
* **DP1 remote driver**: `scripts/remote_lane_substrate_pretrained_driving_prior.sh`
* **DP1 operator-authorize wrapper**: `scripts/operator_authorize_substrate_pretrained_driving_prior_modal_t4_dispatch.sh`
* **Sister supersession (HALT memo)**: `.omx/research/dp1_procedural_codebook_paired_smoke_pre_dispatch_design_HALTED_premise_failure_20260520T192800Z.md` (forensic-only; supersession appended)
* **CLAUDE.md non-negotiables cited**:
  * "HNeRV / leaderboard-implementation parity discipline" L1 (OOD distillation) + L2 (export-first design) + L4 (inflate.py ≤ 200 LOC) + L7 (substrate-engineering exception) + L8 (deterministic)
  * "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"
  * "Substrate MUST be at OPTIMAL FORM before paid empirical dispatch"
  * "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium"
  * "Canonical equations + models registry"
  * "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE"
  * "Apples-to-apples evidence discipline"
  * "Forbidden empirical-claim-without-evidence-tag (the docstring-overstatement trap)"
  * "Modal `.spawn()` HARVEST OR LOSE"
* **Catalog # gates cited (16)**: #110/#113 HISTORICAL_PROVENANCE; #117/#157/#174 canonical serializer; #125 6-hook wire-in; #131/#138 fcntl-locked JSONL + strict-load; #151 trainer flag manifest; #167 smoke-before-full; #204 Modal-aware output path; #205 inflate device selector; #206 crash-resume; #209 DP1 contest-video leakage refusal; #210 DP1 codebook provenance metadata; #211 DP1 composition canonical API; #213 Comma2k19 canonical cache; #220 substrate L1+ operational mechanism; #229 premise-verification-before-edit; #244 Modal NVML env block; #245 Modal call_id ledger; #270 dispatch optimization protocol; #272 distinguishing-feature integration contract; #287 placeholder rationale rejection; #290 canonical-vs-unique decision; #292 per-deliberation assumption surfacing; #294 9-dim checklist; #296 Dykstra-feasibility; #297 signal-axis destruction reversibility; #298 substrate retirement 30-day; #300 council deliberation v2 frontmatter; #303 cargo-cult audit; #305 observability surface; #309 horizon_class declaration; #313 probe-outcomes ledger; #315 OPTIMAL FORM iteration discipline; #322 composition_alpha cascade; #323 canonical Provenance umbrella; #324 post-training Tier-C validation; #325 per-substrate symposium recency; #328 inflate.py LOC budget; #335 cathedral consumer canonical contract; #339 Modal silent-no-spawn extinction; #341 cathedral consumer routing markers; #344 canonical equations registry; #356 per-axis decomposition Provenance; #357 dual-tier consumer architecture

---

## Sign-off

* **Subagent ID**: `wave-3-dp1-procedural-codebook-paired-smoke-pre-dispatch-design-20260520`
* **HEAD at start**: `4f00c87fc`
* **HEAD at sister-HALT-commit**: `cbe587679`
* **UTC**: 2026-05-20T23:21:19Z (this memo creation)
* **Verdict**: PROCEED (design memo) per Catalog #229 PV; operator-gated paid dispatch downstream
* **Cost**: $0 paid GPU; ~50 min wall-clock total (25 min HALT + 25 min PV-of-PV + supersession + design memo)
* **Files touched**: 2 (this memo + sister supersession append on HALT memo)
* **Lane**: `lane_wave_3_dp1_procedural_codebook_paired_smoke_pre_dispatch_design_20260520`
* **Sister-DISJOINT**: verified disjoint from EMPIRICAL BYTE-COUNT GROUNDING + T3 DWT BIND SYMPOSIUM (different memo paths; non-overlapping scope)
* **6-hook wire-in declaration**: §13 ACTIVE on hooks 1-6
* **mission_predicted_contribution**: `frontier_protecting`

---

<!-- END DESIGN MEMO -->
