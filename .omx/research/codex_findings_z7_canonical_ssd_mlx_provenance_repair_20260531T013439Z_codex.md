# Codex Findings: Z7 Canonical SSD-MLX Provenance Repair

UTC: 2026-05-31T01:34:39Z

## Verdict

Z7-Mamba-2 MLX now has a real opt-in path that executes the canonical
`tac.substrates._shared.mamba2_ssd` MLX recurrence helper. This repairs the
prior provenance ambiguity: default Z7 MLX remains the bridge-compatible
reference S6 recurrence, while `--use-canonical-ssd-mlx-backend` records
canonical SSD recurrence custody and simultaneously blocks PyTorch/archive
export until the matching runtime adapter exists.

## Mechanism

- `Z7Mamba2MLXRenderConfig` validates SSD shape provenance and rejects
  `canonical_ssd_mlx_backend_wired=true` unless the SSD backend is actually
  requested.
- `Z7Mamba2MLXModule` adds the SSD projection wrapper used by the canonical
  PyTorch helper pattern: per-head scalar `A_log`, per-head B/C/dt projections,
  zero-initialized D skip, Z7 gate, and output projection.
- The SSD recurrent update delegates to `mamba2_ssd_step_mlx`; tests spy on the
  helper call so this cannot regress into a fake label.
- `Z7Mamba2MLXNativeRenderer` refuses SSD configs because it preserves only the
  old S6-shaped PyTorch bridge path.
- `export_state_dict`, `load_state_dict_from_numpy`, and Z7 archive packing fail
  closed for SSD configs with `canonical_ssd_mlx_pytorch_bridge_export_not_wired`.

## Stack-Of-Stacks Provenance Note

The predictive-coding stack-of-stacks should compose validated members only:
Z8 hierarchical PC, Z7 Mamba state-space PC, DreamerV3 RSSM, Z6-v2 ego-motion PC,
and Z4 Atick-Redlich cooperative receiver. Compound C remains excluded until its
phantom-provenance blocker is repaired. This landing prevents the same failure
mode by separating recurrence-core provenance from archive/runtime custody.

## Verification

- `ruff check` on touched files passed.
- `pytest src/tac/tests/test_z7_mamba2_mlx_backend_lineage.py src/tac/tests/test_z7_mamba2_mlx_module_smoke.py -q` passed: 10 tests.
- `pytest src/tac/substrates/_shared/mamba2_ssd/tests/test_mamba2_ssd.py src/tac/tests/test_z7_mamba2_mlx_backend_lineage.py src/tac/tests/test_z7_mamba2_mlx_module_smoke.py -q` passed: 43 tests.

## Remaining Blocker

Canonical SSD-MLX Z7 training is real MLX research signal, but byte-closed
promotion remains blocked until a dedicated PyTorch/runtime/archive adapter
exports and consumes the SSD parameterization.
