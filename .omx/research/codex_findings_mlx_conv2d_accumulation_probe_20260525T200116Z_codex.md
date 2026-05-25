# Codex Findings: MLX Conv2d Accumulation Probe

UTC: 2026-05-25T20:01:16Z

## Verdict

The partner audit was materially correct: the MLX scorer parity stack had PyTorch-side deterministic backend controls, but Kahan compensated accumulation, fp64 Conv2d accumulation, and MLX-side deterministic-reduction accounting were not durable planner-consumable evidence.

This tranche turns that gap into executable local diagnostics, not authority. `mlx_conv2d_accumulation_probe.v1` now measures optimized MLX Conv2d, fixed-order fp32, Kahan fp32, and fixed-order fp64 against a PyTorch Conv2d reference. The manifest records PyTorch backend knobs and the visible MLX runtime determinism contract.

## Integration

- Reusable primitive: `MLXReferenceConv2dAdapter` in `src/tac/local_acceleration/mlx_scorer_adapters.py`.
- Canonical manifest: `build_mlx_conv2d_accumulation_probe_manifest(...)` in `src/tac/local_acceleration/mlx_scorer_torch_parity.py`.
- Operator CLI: `tools/probe_mlx_conv2d_accumulation.py`.
- Production gate consumer: `tools/check_mlx_scorer_production_contract.py` accepts `--conv2d-accumulation-probe` and fails closed on supplied probe failures.
- Regression coverage: `src/tac/tests/test_mlx_scorer_torch_parity.py`.
- Local probe artifact: `.omx/research/codex_mlx_conv2d_accumulation_probe_20260525T200132Z/mlx_conv2d_accumulation_probe.synthetic_grouped_3x3.json`.

## Authority

All emitted evidence remains `[macOS-MLX research-signal]` / local implementation diagnostic:

- `score_claim=false`
- `promotion_eligible=false`
- `promotable=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- exact CPU/CUDA auth eval is still required before score, promotion, rank, or kill decisions

The MLX runtime contract records the asymmetry directly. If no public deterministic-reduction flag is visible from Python, the manifest carries `mlx_public_deterministic_reduction_flag_not_observed` under `mlx_backend.blockers`. That blocker prevents future agents from silently treating PyTorch deterministic pinning as symmetric MLX determinism.

## Probe Snapshot

Synthetic grouped 3x3 Conv2d smoke:

- input shape: `(2, 4, 11, 13)`
- weight shape: `(6, 2, 3, 3)`
- groups: `2`
- PyTorch device: `cpu`
- MLX device: `cpu`
- optimized MLX Conv2d max abs delta vs PyTorch: `8.940696716308594e-08`
- fixed-order fp32 max abs delta: `1.1920928955078125e-07`
- Kahan fp32 max abs delta: `8.940696716308594e-08`
- fixed-order fp64 max abs delta: `8.940696716308594e-08`
- visible MLX deterministic-reduction flag: absent
- MLX PRNG seed/key APIs: present

## Remaining Work

This is a diagnostic probe, not a scorer-production replacement. The production contract now consumes supplied probe manifests, so the next high-EV follow-up is to thread the same gate into the MLX scorer-response spend-triage planner and queue definitions, where missing/failed numerical-mitigation probes can steer local optimizer selection automatically.
