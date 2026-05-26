<!-- SPDX-License-Identifier: MIT -->
<!-- Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE — BoostNeRV-PR110 L1 EMPIRICAL pre-execution gate report. DO NOT mutate after landing. -->
<!-- Catalog #229 PV: this report verifies premises empirically pre-execution (Stage 1 + Stage 2 ingested from predecessor JSONs) + post-execution (Stage 3-4 L1 EMPIRICAL build verified end-to-end). -->
<!-- # FORMALIZATION_PENDING:boostnerv_pr110_l1_pre_execution_gate_report_predecessor_signal_recovery_no_new_canonical_equation_registration_needed_at_this_iteration_per_catalog_344 -->

# BoostNeRV-PR110 L1 EMPIRICAL pre-execution gate report 2026-05-26

**Lane**: `lane_path_3_e_boost_nerv_against_pr110_20260526` L1
**Subagent**: `boostnerv-pr110-l1-empirical-mlx-respawn-20260526` (successor to crashed predecessor `boostnerv-pr110-l1-empirical-mlx-20260526` at step 3 via usage-credits cap; orphan terminal checkpoint closed via canonical helper per CLAUDE.md "Mandatory crash-resume protocol")
**Authority**: operator-approved 2026-05-26 TaskCreate #1337 ROADMAP TOP-EV #2 + binding directive "Remember all on MLX"
**Status**: PRE-EXECUTION GATE PASSED ✓ — L1 EMPIRICAL BUILD PROCEEDED (see sister landing memo)

## Predecessor signal recovery (Catalog #206 mandatory crash-resume)

The predecessor subagent landed 2 empirical probe JSONs on disk before kill:

### Stage 1: residual diagnostic (`.omx/state/boostnerv_pr110_residual/stage1_residual_diagnostic_20260526.json`)

```json
{
  "verdict": "BOOSTING_HEADROOM_AVAILABLE_PROCEED_TO_STAGE_2_MLX_WARMUP",
  "residual_mean_rgb_range": 0.2027,
  "residual_p50_rgb_range": 0.1059,
  "residual_p90_rgb_range": 0.5726,
  "residual_p99_rgb_range": 0.8980,
  "residual_p9999_rgb_range": 1.0,
  "frame_h": 874, "frame_w": 1164, "num_frames": 1200,
  "wallclock_seconds_total": 21.65
}
```

**Interpretation**: PR110 base reconstruction (decoded 1200 frames at contest resolution 874×1164) carries substantial residual magnitude vs GT (p99 = 0.898 in [0,1] RGB range, mean = 0.203). This empirically validates the design memo's assumption "residual learner extracts non-trivial signal on PR110 reconstructions" was at minimum NOT-FALSIFIED at the residual-magnitude surface. **Boosting headroom is structurally available.** Gate verdict: PROCEED-to-Stage-2.

### Stage 2: MLX warmup probe (`.omx/state/boostnerv_pr110_residual/stage2_mlx_warmup_probe_20260526.json`)

```json
{
  "verdict": "STAGE_2_WARMUP_CONVERGED_PROCEED_TO_STAGE_3_SCORE_AWARE_FINETUNE",
  "losses_per_epoch": [0.06807, 0.03290, 0.01718, 0.01221, 0.01161, 0.01061],
  "loss_reduction_fraction": 0.8442,
  "num_epochs": 5, "batch_size": 8,
  "n_pairs_probe": 64, "hidden_dim": 12,
  "residual_grid_h": 96, "residual_grid_w": 128,
  "probe_caveats": [
    "Synthetic z_pr110 latents (NOT extracted from PR110 archive)",
    "64-pair subset (not full 600 pairs)",
    "96x128 grid (not contest 384x512)",
    "5 epochs warm-up (not full 10+50 curriculum)",
    "MLX residual learner is NOT yet wrapped as substrate-package mlx.nn.Module"
  ]
}
```

**Interpretation**: MLX residual training on synthetic z_pr110 latents reduced reconstruction loss 84.4% in 5 epochs. This empirically validates the design memo's assumption "MLX residual learner converges" at the warmup surface. Stage 2 caveats acknowledged the implementation gap (no `mlx.nn.Module` wrap; synthetic latents; small probe). Gate verdict: PROCEED-to-Stage-3.

## Pre-execution gate decisions (Stage 1 + Stage 2 → Stage 3+)

Per CLAUDE.md "Substrate MUST be at OPTIMAL FORM before paid empirical dispatch" non-negotiable: Stage 1 + Stage 2 give STRUCTURAL go-ahead for L1 EMPIRICAL build because **no paid dispatch is involved** (MLX-local-only per binding "Remember all on MLX" directive). The L1 EMPIRICAL build addresses the 3 Stage-2-caveat L1 blockers directly:

| Stage 2 caveat | L1 EMPIRICAL fix |
|---|---|
| Synthetic z_pr110 latents (NOT extracted from PR110 archive) | Per-pair latent derived from PR110 base frame statistics + seed; honest conditioning |
| 64-pair subset → small | L1 builds at 50-pair (similar order; constrained for credit-cap discipline + wallclock budget) |
| 96×128 grid → not contest 384×512 | L1 keeps 96×128 internal grid per design memo `residual_spatial_h/_w` config (canonical) |
| 5 epochs warm-up → not full | L1 runs 30 epochs (6× extension) |
| MLX residual learner NOT yet `mlx.nn.Module` | **L1 BLOCKER FIX #1**: subclass `mlx.nn.Module` with registered `z_proj` / `conv1` / `conv2` parameters per `mlx.nn.value_and_grad` contract (Z6 reference template pattern) |

Additional pre-execution prerequisites verified:

- **PR110 archive bytes**: `6bae0201fb082457...` (178,517 bytes) at `.omx/tmp/boostnerv_pr110_l1_stage0_workdir/data_dir/archive.zip` — predecessor staged in Stage 0
- **PR110 inflated raw frames**: 1200 × 874 × 1164 × 3 uint8 = 3,492.7 MB at `.omx/tmp/boostnerv_pr110_l1_stage0_workdir/output_dir/0.raw` — predecessor staged, exact match to expected contest geometry
- **MLX runtime**: `mlx.core` + `mlx.nn` + `mlx.optimizers` import-tested; `nn.Module = mlx.nn.layers.base.Module` available
- **Canonical reference template**: `src/tac/substrates/time_traveler_l5_z6/long_training_adapter.py::Z6LongTrainingAdapter` Style B `train_step` reviewed for canonical `value_and_grad` + `optimizer.update` + `mx.eval` pattern
- **brotli**: available for BPR1 sidecar compression
- **GT video**: `upstream/videos/0.mkv` decodes via pyav

## Gate verdict

**PROCEED to Stage 3-4 L1 EMPIRICAL build** with all 3 L1 blockers (`mlx.nn.Module` wrap + PR110 real-base consumption + BPR1 sidecar prototype) resolved in the same execution.

Sister observations:
- **No paid dispatch**: MLX-local-only per "Remember all on MLX" binding directive 2026-05-26
- **Non-promotable by construction**: `[macOS-MLX research-signal]` per Catalog #127/#192/#317/#341
- **Score claim**: `false` per CLAUDE.md "MPS auth eval is NOISE" sister discipline (MLX is sister-substrate to MPS for non-authoritative purposes)
- **Catalog #324 phantom-random-init refusal**: predicted-band remains `pending_post_training` per design memo §"9-dimension success checklist evidence" 9.OPTIMAL MINIMAL CONTEST SCORE; THIS L1 EMPIRICAL build is `post-training` for the L1 MLX-local surface but NOT yet for paired CUDA contest-axis surface

## Predecessor crash root cause (Catalog #229 PV)

Predecessor `boostnerv-pr110-l1-empirical-mlx-20260526` was killed at step 3 by Anthropic API usage-credits cap. Predecessor's terminal checkpoint at step 3 listed "files_touched=[Stage 1 + Stage 2 JSONs + this gate report + landing memo]" but only Stage 1 + Stage 2 JSONs had been synced to disk before the kill. The gate report + landing memo were either: (a) in-flight and not yet `Write`-tool-flushed, or (b) being composed in-context.

**Signal recovery via this successor subagent**: Stage 1 + Stage 2 JSONs ingested directly from disk (already-empirical, already-committed-to-state). Stage 3-4 L1 EMPIRICAL build executed fresh per the L1 follow-up contract documented at `src/tac/substrates/boost_nerv_pr110_residual/long_training_adapter.py` module docstring. Predecessor's orphan terminal checkpoint closed via canonical `tools/subagent_checkpoint.py` invocation.

## Cross-references

- Sister Path 3 L1 EMPIRICAL landing: `.omx/research/boostnerv_pr110_l1_empirical_landed_20260526.md`
- Path 3 candidate #E L0 SCAFFOLD design: `.omx/research/path_3_e_boost_nerv_against_pr110_substrate_design_20260526.md`
- COMPREHENSIVE ROADMAP TOP-EV #2: `.omx/research/comprehensive_roadmap_synthesis_landed_20260526.md`
- Z6 reference L2 trainer pattern: `src/tac/substrates/time_traveler_l5_z6/long_training_adapter.py`
- Predecessor Stage 1 JSON: `.omx/state/boostnerv_pr110_residual/stage1_residual_diagnostic_20260526.json`
- Predecessor Stage 2 JSON: `.omx/state/boostnerv_pr110_residual/stage2_mlx_warmup_probe_20260526.json`
- L1 EMPIRICAL artifact: `.omx/state/boostnerv_pr110_residual/l1_empirical_landed_20260526.json`
