---
name: Continual-learning posterior validation across all 2026-05-11 anchors
date: 2026-05-11
research_only: false
lane_class: substrate_engineering
lane_id: lane_continual_learning_posterior_validation
related:
  - feedback_a1_dual_cuda_dispatch_landed_20260509.md
  - feedback_pr106_r2_paired_cpu_eval_landed_20260511.md
  - feedback_v_dispatch_consolidation_landed_20260511.md
  - feedback_l2_sparse_aware_encoders_first_dispatch_landed_20260511.md
  - feedback_pr101_pr103_grammar_on_pr106_r2_landed_20260511.md
---

# Continual-learning posterior validation 2026-05-11

## Bottom line

The continual-learning posterior at `.omx/state/continual_learning_posterior.json`
(schema `tac_continual_learning_posterior_v1`) currently contains
**6 accepted anchors** across 5 architecture classes. Cross-referencing
the contest-eval result tree under `experiments/results/modal_auth_eval/`
and `experiments/results/modal_auth_eval_cpu/` for 2026-05-11 produces
**22 unique authoritative anchor candidates** (16 contest-CUDA + 6
contest-CPU on 1:1 contest-compliant hardware per CLAUDE.md "Submission
auth eval — BOTH CPU AND CUDA").

**Result**: 6/22 anchors are reflected in the posterior; **16 candidate
anchors are ORPHANED** (not yet posterior-updated). 0 anchors are
miscategorized within the posterior. 0 anchors carry a paired-axis drift
outside the HNeRV-cluster predictor's expected range. The orphan count
is a tooling-state artifact rather than a custody failure: each orphan
candidate has its custody intact in the per-result `contest_auth_eval.json`
file; the gap is that no agent has called `posterior_update_locked` on
those rows yet.

This validation pass surfaces **3 operator decisions** for the
custody-vs-posterior loop closure but does NOT auto-update the posterior
(per CLAUDE.md "Subagent coherence-by-default" + "Continual-learning
posterior update" non-negotiable; bulk back-fill would shift the
posterior n_anchors count beyond the operator's awareness window).

## Posterior state at validation time

| Field | Value |
|---|---|
| schema | `tac_continual_learning_posterior_v1` |
| evidence_grade | `[continual-learning posterior; non-authoritative]` |
| accepted_anchor_count | **6** |
| refused_anchor_count | 6 |
| last_updated_utc | 2026-05-11T20:43:16.318536+00:00 |
| track_correction_posteriors | {} (empty; no tracks have updated yet) |
| source_rho_posteriors | {} (empty; no source-rho estimates yet) |

## Per-anchor validation in the posterior

| # | axis | architecture_class | sha256 (16 char) | bytes | score | hardware_substrate | observed_at | Custody verdict |
|---|---|---|---|---:|---:|---|---|---|
| 1 | cpu | hnerv_ft_microcodec | 87ec7ca5f2f328a8 | 178262 | 0.19284758 | linux_x86_64_gha_cpu | 2026-05-09T02:03:11Z | ✓ A1 CPU per Catalog #127 (validate_custody PASS) |
| 2 | cuda | hnerv_ft_microcodec | 87ec7ca5f2f328a8 | 178262 | 0.22635202 | linux_x86_64_t4 | 2026-05-09T02:42:55Z | ✓ A1 CUDA per Catalog #127 (validate_custody PASS) |
| 3 | cuda | lane_pr106_latent_sidecar_r2_pr101_grammar | c48631e11a9bb18d | 186780 | 0.20661814 | linux_x86_64_t4 | 2026-05-11T18:09:15Z | ✓ PR101 grammar paired runtime CUDA per Catalog #127 |
| 4 | cuda | lane_c3_residual_pr106_sidecar_dispatch_ready | eafb1a027f706575 | 186832 | 0.20663364 | linux_x86_64_t4 | 2026-05-11T18:13:10Z | ✓ c3 residual L1 empty CUDA per Catalog #127 |
| 5 | cuda | lane_wavelet_residual_pr106_sidecar_dispatch_ready | ed90a2250e948ed7 | 186832 | 0.20663364 | linux_x86_64_t4 | 2026-05-11T18:13:16Z | ✓ wavelet residual L1 empty CUDA per Catalog #127 |
| 6 | cuda | lane_c3_residual_pr106_sidecar_l2_sparse_aware_dispatch | 8e61ff2d5a42f463 | 186832 | 0.20663364 | linux_x86_64_t4 | 2026-05-11T20:43:16Z | ✓ c3 sparse-aware L2 CUDA per Catalog #127 |

**Custody verdict: 6/6 PASS.** All entries pass the typed
`ContestResult.validate_custody` check (per Catalog #127): each carries
a recognized `(tag, axis, hardware_substrate)` triplet whose tag's
required axis matches the axis AND whose hardware_substrate prefix is
in the tag's allowed set. No `tag_axis_mismatch`, no
`cpu_tag_non_gha_linux`, no `cuda_tag_unknown_substrate`, no
`macos_substrate`, no `missing_metadata`, no `advisory_grade` refusals.

## Substrate-class consistency check

| architecture_class | family | n_anchors in posterior | sister anchors known to exist |
|---|---|---:|---:|
| hnerv_ft_microcodec | A1 | 2 (CPU+CUDA paired) | 0 additional |
| lane_pr106_latent_sidecar_r2_pr101_grammar | PR101-grammar-on-PR106 | 1 (CUDA only) | 1 paired CPU at 0.22806463 (orphan) |
| lane_c3_residual_pr106_sidecar_dispatch_ready | c3 family L1 | 1 (CUDA only) | 1 paired CPU at 0.22810213 (orphan) |
| lane_wavelet_residual_pr106_sidecar_dispatch_ready | wavelet family L1 | 1 (CUDA only) | 1 paired CPU at 0.22810213 (orphan) |
| lane_c3_residual_pr106_sidecar_l2_sparse_aware_dispatch | c3 sparse-aware L2 | 1 (CUDA only) | 0 paired CPU known yet |

**Substrate-class consistency**: 5 distinct architecture_class labels,
each consistent with the empirical evidence trail. NO miscategorization.

## Per-axis (CPU vs CUDA) drift profile

The HNeRV-cluster CUDA-CPU drift profile predicts +0.033 ± 0.001 per
the cluster predictor (per `feedback_cuda_cpu_axis_profile_learning_layer_20260508.md`).

| Architecture class | CUDA score | CPU score | Δ (CUDA − CPU) | Predicted Δ | Within ±0.001 tolerance? |
|---|---:|---:|---:|---:|:---:|
| hnerv_ft_microcodec (A1) | 0.22635202 | 0.19284758 | +0.0335 | +0.033 ± 0.001 | YES |
| lane_pr106_latent_sidecar_r2_pr101_grammar | 0.20661814 | (orphan: 0.22806463 in custody) | (would be -0.0214) | predicted +0.033 | NO — sign-flipped |
| lane_c3_residual_pr106_sidecar_dispatch_ready | 0.20663364 | (orphan: 0.22810213 in custody) | (would be -0.0215) | predicted +0.033 | NO — sign-flipped |
| lane_wavelet_residual_pr106_sidecar_dispatch_ready | 0.20663364 | (orphan: 0.22810213 in custody) | (would be -0.0215) | predicted +0.033 | NO — sign-flipped |

**Critical observation**: A1 (HNeRV-FT-microcodec architecture class)
follows the HNeRV-cluster CUDA-CPU drift predictor. The PR106-derived
families (PR101 grammar, c3, wavelet) all show **sign-flipped device-axis
behavior** — CPU score is HIGHER (worse) than CUDA score by ~0.0215
points. This is the **PR106-cluster device-axis profile** (per V landing
2026-05-11 + handoff "Device-axis rule"), NOT a posterior anomaly.

**The PR106 family is its own substrate class** with its own
CUDA-CPU drift profile. Per the handoff:
> Public PR100/101/102/103/105 rows are mostly CPU/comment evidence and
> often look better than local T4 replays. PR106-derived local packets
> are currently CUDA-favored.

The posterior currently has insufficient n_anchors per architecture_class
to compute per-class drift posteriors (each class has n=1 paired or
n=2 paired-A1-only). The HNeRV-cluster predictor is the ONLY cluster
with N=1 paired anchor (A1), and that anchor matches the predicted
+0.033 drift within 0.0005.

## Orphan anchor candidates (NOT in posterior)

The following 16 contest-CUDA anchors + 6 contest-CPU anchors exist
in custody under `experiments/results/modal_auth_eval/` and
`experiments/results/modal_auth_eval_cpu/` from 2026-05-10/2026-05-11
but have NOT been posterior-updated:

### contest-CUDA orphans (16)

| Result dir | sha256 (16 char) | bytes | canonical_score | architecture-class hint |
|---|---|---:|---:|---|
| pr101_a5_channel_qbits_dp_qsum200_exact_cuda_modal_20260510T194943Z | efc0466bc38edb9c | 178014 | 0.23396252 | pr101_lossy_coarsening |
| pr101_a5_trust_q7_all_exact_cuda_modal_20260510T193821Z | 39dbfd05d4861c6c | 177928 | 0.23488498 | pr101_lossy_coarsening |
| pr101_bias_refine_exact_cuda_modal_20260510T165632Z | b83bf3488625dbd7 | 178258 | 0.22650343 | pr101_lossy_coarsening |
| pr101_kaggle_proxy_runtime_packet_exact_cuda_modal_20260510T194142Z | b83bf3488625dbd7 | 178258 | 0.22688161 | pr101_lossy_coarsening |
| pr103_global_combo_12b_exact_cuda_modal_20260510T2257Z | 578c8f4e86eafc9d | 178211 | 0.22777018 | pr103_arithmetic_coding |
| pr103_global_combo_mid32_latent_hi_16b_clean_runtime_exact_cuda_modal_20260510T2346Z | 8460014d70855ce9 | 178207 | 0.22776743 | pr103_arithmetic_coding |
| pr103_global_combo_mid32_latent_hi_16b_exact_cuda_modal_20260510T2339Z | 8460014d70855ce9 | 178207 | 0.22776743 | pr103_arithmetic_coding (dup of above) |
| pr103_histogram_8b_packet_exact_cuda_modal_20260510T221146Z | 2427cbb7f68e8e3b | 178215 | 0.22777268 | pr103_arithmetic_coding |
| pr103_pr106_cuda_auto_current_runtime_modal_20260511T0638Z | ec0890c2d2317dca | 185578 | 0.20898305 | pr103_on_pr106 |
| pr103_pr106_cuda_raw_manifest_modal_20260511T060115Z | ec0890c2d2317dca | 185578 | 0.20898305 | pr103_on_pr106 (dup) |
| pr103_pr106_dual_runtime_cuda_20260511T021405Z | ec0890c2d2317dca | 185578 | 0.20898305 | pr103_on_pr106 (dup) |
| pr103_pr106_dual_runtime_cuda_v2_20260511T022553Z | ec0890c2d2317dca | 185578 | 0.20898305 | pr103_on_pr106 (dup) |
| pr103_source_same_runtime_cuda_baseline_modal_20260510T2300Z | 31881b2d23d027e6 | 178223 | 0.22777818 | pr103_arithmetic_coding |
| pr106_latent_sidecar_20260511T150517Z | 947b85e8a69db295 | 186808 | 0.20739428 | pr106_latent_sidecar_r1 |
| pr106_latent_sidecar_r2_20260511T160358Z | 7f926bc3e213af1c | 186822 | 0.20664589 | pr106_latent_sidecar_r2 |
| lane_cool_chic_residual_pr106_sidecar_20260511T200000Z | d48600da99bad7a7 | 186832 | 0.20663364 | cool_chic family L1 |
| lane_coord_mlp_residual_pr106_sidecar_20260511T200000Z | 01df6f12e562fdd1 | 186832 | 0.20663364 | coord_mlp family L1 |
| lane_siren_residual_pr106_sidecar_20260511T200000Z | f373b308b08059e4 | 186832 | 0.20663364 | siren family L1 |

After dedup by (sha256, axis): **13 unique CUDA orphans**.

### contest-CPU orphans (6)

| Result dir | sha256 (16 char) | bytes | canonical_score | architecture-class hint |
|---|---|---:|---:|---|
| lane_c3_residual_pr106_sidecar_20260511T203000Z | eafb1a027f706575 | 186832 | 0.22810213 | c3 family L1 paired CPU |
| lane_wavelet_residual_pr106_sidecar_20260511T203000Z | ed90a2250e948ed7 | 186832 | 0.22810213 | wavelet family L1 paired CPU |
| pr103_pr106_cpu_raw_manifest_modal_20260511T060714Z | ec0890c2d2317dca | 185578 | 0.22966566 | pr103_on_pr106 paired CPU |
| pr106_latent_sidecar_20260511T151955Z | 947b85e8a69db295 | 186808 | 0.22868028 | pr106_latent_sidecar_r1 paired CPU |
| pr106_latent_sidecar_r2_20260511T171453Z | 7f926bc3e213af1c | 186822 | 0.22809238 | pr106_latent_sidecar_r2 paired CPU |
| pr106_latent_sidecar_r2_pr101_grammar_20260511T200000Z | c48631e11a9bb18d | 186780 | 0.22806463 | PR101 grammar paired CPU |

**6 unique CPU orphans.**

## Recommendations

### Custody-OK orphans

All 13 CUDA orphans + 6 CPU orphans are CUSTODY-VALID per the
per-result `contest_auth_eval.json` schema (`schema_version=1`,
`evidence_grade` set, `score_axis` set, `provenance.archive_sha256`
recorded). Each is posterior-update-eligible per Catalog #127.

### Why the orphan gap exists

Per CLAUDE.md "Subagent coherence-by-default" 6-hook wire-in: every
landing memo SHOULD declare a `Continual-learning posterior update`
hook contribution. Reviewing the 2026-05-11 landings:

- 6 anchors that ARE in the posterior were updated by their respective
  landing-memo subagents calling `posterior_update_locked` directly.
- 16 anchors that are NOT in the posterior come from landings that
  declared the hook as `applied to local custody only — bulk update
  deferred to validation pass` or similar deferral language.

This validation pass IS the deferred hook landing. However, per
CLAUDE.md "Continual-learning posterior — non-negotiable" + the
operator-trigger discipline, **bulk back-fill of 19 orphans without
explicit operator AskUserQuestion would shift the posterior's
n_anchors from 6 to 25 in a single update**, which is large enough
to materially change downstream posteriors (`track_correction_posteriors`,
`source_rho_posteriors`) for the next planner run.

### Surfaced operator decisions (3)

**OD-A**: Authorize bulk back-fill of 19 custody-OK orphans into the
continual-learning posterior in this validation pass?
- The back-fill would call `posterior_update_locked` per orphan;
  posterior n_anchors would go 6 → 25.
- Idempotence: `posterior_update_locked` deduplicates by
  (sha256, axis, hardware_substrate); re-running is safe.
- Cost: $0 GPU (pure file-update).
- Risk: posterior shift may affect downstream planner runs that
  assume the n=6 posterior.

**OD-B**: Authorize bulk back-fill of all PR101/PR103 CUDA-only orphans
to seed the per-architecture-class drift posteriors (3+ paired CPU
anchors per class enables posterior estimation per
`tac.optimization.cuda_cpu_axis_profile_registry`)?
- Currently 0 architecture classes have 3+ paired anchors;
  back-filling would advance the posterior toward per-class drift
  estimation which gates Phase 3 dispatch readiness.

**OD-C**: Authorize a permanent fix to enforce the
"6-hook wire-in declaration" non-negotiable at the SCRIPT level so
future landings auto-update the posterior at landing-time, eliminating
the orphan accumulation pattern?
- This would be a new STRICT preflight check (Catalog #151) refusing
  any new landing memo that declares the continual-learning hook as
  "deferred" without an explicit operator-trigger justification.
- Cost: $0 GPU; lane-engineering work.

ALL THREE decisions are surfaced ONLY here; NONE are auto-acted.

## Net validation verdict

| Metric | Count |
|---|---:|
| Anchors validated in posterior | **6** |
| Orphan anchors with intact custody | **19** (13 CUDA + 6 CPU) |
| Anchors miscategorized within posterior | **0** |
| Anchors with paired-axis drift outside HNeRV-cluster predictor | **0** within posterior; **3 candidate substrate classes** in orphan custody show **sign-flipped PR106-cluster drift** (CUDA < CPU; opposite of HNeRV +0.033) — NOT an anomaly, but evidence of a SECOND substrate cluster that the posterior should learn separately |
| Custody validation refusals | **0/6** PASS per Catalog #127 |
| Multi-axis lock-write integrity | PASS per Catalog #128 (`posterior_update_locked` is the only writer) |

## CLAUDE.md compliance tags

- `predicted_band_only_no_score_claim`
- `posterior_validation_advisory_only`
- `operator_gate_non_negotiable_at_every_dispatch`
- `halt_and_ask_default_on`
- `no_tmp_paths`
- `continual_learning_posterior_aware`

## 6-hook wire-in declaration (per CLAUDE.md "Subagent coherence-by-default")

1. **Sensitivity-map**: validation surfaces per-class drift sensitivity
   priors for `tac.sensitivity_map.*` (HNeRV-cluster +0.033 vs
   PR106-cluster -0.0215).
2. **Pareto constraint**: each orphan anchor IS a Pareto point in
   (cost, predicted_delta, axis) space; back-fill enriches the Pareto
   frontier in `tac.pareto_*`.
3. **Bit-allocator hook**: posterior n_anchors grows imply tighter
   confidence intervals on per-class drift, which inform
   per-architecture-class bit allocation.
4. **Cathedral autopilot dispatch hook**: validated posterior IS the
   primary input to autopilot's CandidateRow predicted_score_delta
   estimation.
5. **Continual-learning posterior update**: THIS IS that hook;
   surfaced via OD-A.
6. **Probe-disambiguator**: HNeRV-cluster vs PR106-cluster IS a
   regime-conditional probe-disambiguator on substrate-class drift.
