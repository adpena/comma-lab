# Codex Findings: Z7 Canonical SSD MLX Runtime Bridge

UTC: 2026-05-31T01:45:20Z

## Verdict

The prior Z7 canonical SSD MLX landing repaired provenance but intentionally
blocked export at `canonical_ssd_mlx_pytorch_bridge_export_not_wired`. That
blocker is now closed for archive/runtime custody: `Z7Mamba2MLXModule` exports
canonical SSD-shaped predictor weights, `Z7MCM2` stores/loads
`backend="ssd_reference"`, and the generated submission runtime vendors the
SSD helper plus framework backend needed for scorer-free CPU inflate.

Authority remains fail-closed. Canonical SSD MLX rows are still
`[macOS-MLX research-signal]` and carry
`canonical_ssd_mlx_exact_cpu_cuda_replay_required`; they are not score,
promotion, rank/kill, or exact-dispatch authority until contest CPU/CUDA auth
replay signs the archive/runtime packet.

## Concrete Changes

- `mlx_module.py` exports the SSD recurrent core through the exact PyTorch SSD
  state dict keys consumed by `Mamba2Predictor(backend="ssd_reference")`,
  including `A_log`, `B_proj`, `C_proj`, `dt_proj`, `D`, gate projection,
  output projection, decoder weights, latents, residuals, and ego-motion.
- `archive_candidate.py` translates SSD MLX configs into
  `Z7Mamba2PredictiveCodingConfig(backend="ssd_reference")`, stamps runtime
  bridge metadata, packs SSD exports through the normal `Z7MCM2` grammar, and
  vendors `tac.substrates._shared.mamba2_ssd` plus
  `tac.framework_agnostic.backend` into generated runtimes.
- `archive.py` parses SSD metadata back into `ssd_reference` configs and
  appends the exact CPU/CUDA replay blocker to archive authority metadata.
- Regression coverage now proves both local parse/inflate and generated
  `submission/inflate.sh` runtime consumption for the MLX-exported SSD packet.

## Verification

- `.venv/bin/ruff check` on touched source/tests: PASS
- `.venv/bin/pytest src/tac/tests/test_z7_mamba2_mlx_backend_lineage.py src/tac/tests/test_z7_mamba2_mlx_module_smoke.py src/tac/tests/test_z7_mamba2_canonical_helper_rewire.py src/tac/tests/test_z7_mamba2_substrate_full_landing.py -q`: 43 passed
- `.venv/bin/pytest src/tac/substrates/_shared/mamba2_ssd/tests/test_mamba2_ssd.py src/tac/tests/test_z7_mamba2_mlx_backend_lineage.py src/tac/tests/test_z7_mamba2_mlx_module_smoke.py src/tac/tests/test_z7_mamba2_canonical_helper_rewire.py src/tac/tests/test_z7_mamba2_substrate_full_landing.py -q`: 76 passed

## Next Integration Blocker

The bridge now emits byte-closed SSD archives and receiver proof, but exact
score lowering still needs the autonomous runner to treat these packets as
exact-ready inputs only after archive/runtime custody, preclaim gating, and
contest CPU/CUDA replay. The next code slice should wire this SSD bridge into
the contract-first acquisition/materializer path so MLX-trained Z7 candidates
flow automatically into exact-axis blockers or dispatch packets instead of
remaining only trainer-local exports.
