# Canonical L2 Long-Training Infrastructure

**Status**: CANONICAL adopted 2026-05-26 per L2-INFRA-BUILD landing (Path 3 cascade doctrine commit `fb270e9b6` §"L2 LONG-TRAINING INFRASTRUCTURE — CANONICAL + REUSABLE + COMPOSABLE + PRODUCTION-HARDENED").

**Purpose**: substrate-agnostic L2 long-training contract + canonical entry-point. Each substrate's L2 trainer is ≤30 LOC of substrate-specific config + ONE canonical helper invocation, replacing ~600 LOC of per-substrate hand-rolled training boilerplate.

## Quick start (operator + sister-subagent)

```python
from pathlib import Path
from tac.training.long_training_canonical import (
    CurriculumStage, LongTrainingConfig, run_long_training,
)
from tac.substrates.<my_substrate>.long_training_adapter import MyLongTrainingAdapter

# 1) Build substrate-specific adapter (Protocol-conforming).
adapter = MyLongTrainingAdapter(...)

# 2) Build canonical config.
config = LongTrainingConfig(
    substrate_id="my_substrate",
    lane_id="lane_my_substrate_l2_20260601",
    epochs=3000,
    batch_pair_indices_per_step=2,
    curriculum_stages=(  # or use PR95_8STAGE_CURRICULUM_DEFAULT
        CurriculumStage(name="warmup", start_epoch=0, end_epoch=300, lr_scale=0.1),
        CurriculumStage(name="main",   start_epoch=300, end_epoch=2400, lr_scale=1.0),
        CurriculumStage(name="finetune", start_epoch=2400, end_epoch=3000, lr_scale=0.1),
    ),
    learning_rate=1e-3,
    seed=42,
    output_dir=Path("experiments/results/my_substrate_l2_20260601"),
    device="mlx",
    checkpoint_interval_epochs=100,
    early_stopping_patience=500,
    evidence_grade="[macOS-MLX research-signal]",
)

# 3) ONE canonical helper invocation runs everything.
artifact = run_long_training(adapter, config)

# 4) Inspect canonical artifact.
print(f"Trained {artifact.total_epochs_completed} epochs in {artifact.total_wall_clock_seconds:.1f}s")
print(f"EMA shadow checkpoint: {artifact.ema_shadow_checkpoint_path}")
print(f"Canonical archive sha256: {artifact.archive_sha256}")
print(f"Non-promotable per Catalog #127/#192/#317/#341: promotable={artifact.promotable}")
```

## What the canonical helper handles for you

Per the canonical 10-element contract per the doctrine §L2 LONG-TRAINING INFRASTRUCTURE:

1. **`run_long_training(adapter, config) -> TrainingArtifact`** — canonical entry-point.
2. **`LongTrainingConfig`** frozen dataclass — schema validated on construction.
3. **`CurriculumStage`** frozen dataclass — per-stage hparams with strict invariants.
4. **Checkpoint+resume** — periodic + final checkpoints with `substrate_id` + `lane_id` + `curriculum_hash` validation refusing cross-substrate / cross-curriculum resume.
5. **Per-arm canonical Provenance + posterior anchor** — auto-stamped via `tac.provenance.build_provenance_for_predicted` + `tac.substrates._shared.posterior_emission_helper.emit_substrate_landing_posterior_anchor`. All artifacts carry non-promotable markers per Catalog #127/#192/#317/#341.
6. **EMA shadow** — canonical Polyak averaging via `PolyakEMAShadow` (decay=0.997 per Catalog #2 NON-NEGOTIABLE); snapshot+restore pattern at archive emission per CLAUDE.md "EMA — NON-NEGOTIABLE".
7. **Multi-arm parallel dispatch** — `run_long_training_multi_arm` orchestrates N concurrent substrate arms; sequential-safe default per Catalog #302 sister-subagent scope overlap.
8. **OOM-safe step runner** — `OOMSafeStepRunner` halves batch size on OOM up to `max_retries` (default 4); raises typed error after exhaustion so caller can crash-recover via checkpoint resume.
9. **Observability surface per Catalog #305** — `TelemetrySink` emits per-epoch `PerEpochMetrics` JSONL rows (loss + loss_components + per_axis_decomposition + EMA-drift + wall-clock).
10. **OSS-clean public API** — narrow `__all__`; canonical docstrings; SPDX-License-Identifier: MIT header; zero `/Users/...` paths.

## SubstrateLongTrainingAdapter Protocol contract

Every substrate adapter must declare these attributes + methods (Style A vs Style B):

**Required attributes**:
- `substrate_id: str` — canonical id (matches `src/tac/substrates/<substrate_id>/`).
- `model: Any` — trainable parameters container (torch.nn.Module OR MLX module).

**Required methods**:
- `sample_batch(batch_size, seed) -> Any` — substrate-specific batch sampler.
- `loss_fn(model, batch, loss_weights) -> {"total": ...}` — REQUIRED for Style A (torch).
- `optimizer_step(model, loss, learning_rate) -> None` — REQUIRED for Style A.
- `export_state_dict(model, path) -> None` — substrate-specific checkpoint emission.
- `export_archive(model, output_dir) -> (path, sha256, bytes) | None` — optional.
- `score_aware_components(model, batch) -> {"d_seg": ..., "d_pose": ..., "rate": ...} | None` — optional per Catalog #356.

**Optional (Style B)**:
- `train_step(batch, learning_rate, loss_weights) -> {"total": ...}` — combined value+grad+update (MLX-natural via `mlx.nn.value_and_grad`).

The canonical helper auto-detects `train_step` and prefers it over `loss_fn` + `optimizer_step` when present.

## Worked example: D=Z6 predictive coding

**L1 promotion** (hand-rolled, ~600 LOC): `experiments/train_substrate_z6_predictive_coding_mlx.py`

**L2 canonical pattern** (~30 LOC substrate-specific config + ONE helper call): `experiments/train_substrate_z6_predictive_coding_mlx_l2.py`

**Z6 adapter** (Style B because MLX uses `value_and_grad`): `src/tac/substrates/time_traveler_l5_z6/long_training_adapter.py`

The full L2 trainer:

```python
# (After argparse + video decode setup, the substrate-specific surface is:)
cfg = Z6PredictiveCodingConfig(latent_dim=args.latent_dim, num_pairs=args.num_pairs, ...)
adapter = Z6LongTrainingAdapter(config=cfg, target_rgb_0=t0, target_rgb_1=t1, ...)
canonical_config = LongTrainingConfig(
    substrate_id="time_traveler_l5_z6",
    lane_id="lane_path_3_d_z6_l2_long_training_canonical_proof_20260526",
    epochs=args.epochs,
    curriculum_stages=(CurriculumStage(name="z6_l2_recon_full", start_epoch=0, end_epoch=args.epochs),),
    output_dir=args.output_dir,
    device="mlx",
)
artifact = run_long_training(adapter, canonical_config)
```

That's it. Everything else (EMA, checkpointing, Provenance, posterior anchor, archive export, telemetry, OOM-safety) is the canonical helper's responsibility.

## Composability with sister tooling

The canonical TrainingArtifact integrates with:

- **Cathedral autopilot** (`tools/cathedral_autopilot_autonomous_loop.py`) — canonical posterior anchor at `.omx/state/mps_research_signal_manifest.jsonl` is auto-discovered by the 62-cathedral-consumer cascade per Catalog #335.
- **Canonical equations registry** (`tac.canonical_equations`) — TrainingArtifact's empirical anchors can calibrate canonical equations via `update_equation_with_empirical_anchor` per Catalog #344.
- **Bit allocator** (`tac.bit_allocator`) — `PerEpochMetrics.per_axis_decomposition` is the per-axis input bit_allocator consumes per Catalog #1068.
- **Findings Lagrangian** (`tac.findings_lagrangian`) — sister meta-Lagrangian invocation per Catalog #355 consumes canonical Provenance.
- **Master-gradient consumers** (Catalog #354 BUNDLE) — multi-arm dispatch results feed `master_gradient_xray_consumer` for cross-arm diff.

## Catalog discipline (binding)

- **Catalog #2** EMA NON-NEGOTIABLE (decay=0.997)
- **Catalog #127/#192/#317/#341** canonical non-promotable markers
- **Catalog #128** fcntl-locked posterior write discipline
- **Catalog #131** bare-write to .omx/state/ refusal
- **Catalog #146** contest-compliant inflate runtime contract
- **Catalog #190** hardware_substrate auto-detection
- **Catalog #206** subagent crash-resume discipline (sister pattern)
- **Catalog #208** docs/local-paths discipline (no `/Users/...` paths)
- **Catalog #229** premise verification (file hashes + config snapshot)
- **Catalog #265/#335** canonical contract pattern
- **Catalog #287** placeholder-rationale rejection
- **Catalog #290** canonical-vs-unique decision per layer
- **Catalog #299** gate consolidation discipline
- **Catalog #305** observability surface 6-facet
- **Catalog #323** canonical Provenance umbrella
- **Catalog #356** per-axis decomposition per dual-tier contract

## Sister modules + composition contract

- **`tac.training.long_training_canonical`** (THIS module) — canonical substrate-AGNOSTIC L2 helper; substrate-conforming adapters plug in.
- **`tac.local_acceleration.pr95_hnerv_mlx_long_training`** — canonical PR95-HNeRV-SPECIFIC L2 helper (hardcoded `HNeRVDecoderMLX` + RGB-MSE + PyAV pipeline). Per Catalog #290 canonical-vs-unique: legitimate fork because PR95-HNeRV training has substrate-specific structure.
- **`tac.substrates._shared.trainer_skeleton`** — canonical substrate-trainer utilities (seeds, EVAL_HW, decode_real_pairs, device_or_die).
- **`tac.substrates._shared.posterior_emission_helper`** — canonical L0/L1 landing posterior emission; THIS module's per-arm posterior anchor invokes it.

## Conformance testing

Every new substrate adapter MUST pass:

1. **Protocol conformance**: `validate_substrate_adapter(adapter)` succeeds without exception.
2. **End-to-end smoke**: at `epochs=5` `num_pairs=8` the canonical helper completes + emits valid TrainingArtifact.
3. **Catalog #335 conformance test row**: add a parametrized test entry to `src/tac/substrates/_shared/tests/test_substrate_long_training_conformance.py` (sister of the cathedral consumer canonical contract validator).

## Operator-routable next steps

After L2-INFRA-BUILD lands:

1. **D=Z6 L2 first long-training run** — proof-of-canonical-helper end-to-end at `epochs=100` `num_pairs=50` (smoke already verified at `epochs=5` `num_pairs=8`); canonical Provenance + posterior anchor lands; cathedral consumer cascade observes.
2. **L1-PROMOTION-CASCADE per non-L2-ready substrates** — spawn L1 promotion subagents for E=BoostNeRV / G=NIRVANA / C'=NSCS06 / B'=Z7-Mamba-2-v2 / J=MDL-IBPS. Each L1 promotion produces a substrate-specific `long_training_adapter.py` per the Z6 reference pattern; L2 trainer entry-point is ~30 LOC.
3. **L2 LONG-TRAINING-CASCADE concurrent on M-series** — 3-5 substrates parallel via `run_long_training_multi_arm` once 3-5 L1 promotions land.

## Anti-patterns (DO NOT)

- **Per-substrate long-training code**: each substrate writes its own training loop, checkpoint logic, EMA wrapping, score-aware loss invocation, etc. → DUPLICATIVE CODE that violates operator directive + Catalog #299.
- **Substrate-coupled trainer infrastructure**: training helper hard-codes substrate-specific architecture details (e.g. `dreamer_v3_specific_training_loop()`) → BREAKS COMPOSABILITY.
- **Skip-L2-infra-and-spawn-L2-training**: each substrate gets ad-hoc training code → DUPLICATIVE + non-production-hardened + non-OSS-clean.
- **`device='mps'`**: REFUSED at LongTrainingConfig construction per CLAUDE.md "MPS auth eval is NOISE" non-negotiable. Use `'mlx'` for Apple Silicon (acceptable as research-signal) or `'cuda'` for promotion-grade.
- **`output_dir` under `/tmp/`**: REFUSED at LongTrainingConfig construction per CLAUDE.md "Forbidden /tmp paths in any persisted artifact" + Catalog #220 transient-evidence trap. Use `experiments/results/<lane_id>_<timestamp>/`.

## Cross-references

- Path 3 cascade doctrine: `.omx/research/path_3_canonical_substrate_development_cascade_doctrine_20260526.md`
- D=Z6 L1 promotion: `.omx/research/path_3_d_z6_l1_promotion_landed_20260526.md`
- PR95 sister-module reference: `src/tac/local_acceleration/pr95_hnerv_mlx_long_training.py`
- Cathedral consumer canonical contract: `src/tac/cathedral/consumer_contract.py`
- CLAUDE.md "Beauty, simplicity, and developer experience"
- CLAUDE.md "Subagent coherence-by-default" (6-hook wire-in)
- CLAUDE.md "MLX portable-local-substrate authority"
- CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA"
