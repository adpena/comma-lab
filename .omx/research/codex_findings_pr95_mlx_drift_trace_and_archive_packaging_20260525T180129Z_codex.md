# Codex Findings: PR95 MLX Drift Trace And Archive Packaging

UTC: 2026-05-25T18:01:29Z
Agent: codex
Scope: PR95/HNeRV MLX reproduction lane, MLX/PyTorch drift, byte-closed archive packaging.

## Findings

1. The PR95 MLX -> PyTorch forward drift is bounded and reproducible at the Stage 8 checkpoint: output `max_abs=3.0517578125e-05`, `mean_abs=3.217932923386494e-06`, within the canonical `1e-4` max / `1e-5` mean attested tolerance.
2. Decoder-boundary tracing localizes the visible drift cliff to the RGB heads. Stem, upsample blocks, refine residual, and feature tensor remain below `~2.8e-7` max_abs; `rgb_0`, `rgb_1`, and `output` reach `3.0517578125e-05`.
3. Naive deterministic substitution of bilinear/sin/sigmoid is not the right fix: isolated probes classify those operations as byte-stable by default. The remaining mechanism is framework arithmetic around conv/head execution plus sigmoid and `255` scaling.
4. The correct production contract is attested-tolerance portability plus false-authority markers and exact-auth escalation, not treating MLX/local CPU as contest score authority.
5. PR95 MLX byte-closed archive packaging is now executable: MLX-exported PyTorch `state_dict` + source archive latents/meta -> canonical PR95 archive grammar + vendored self-contained runtime. Generated runtime payloads remain ignored; compact JSON reports are tracked.

## Artifacts

- `.omx/research/codex_pr95_stage6_stage7_full_profile_queue_20260525T1714Z/matrix/stage8/pr95_stage8_muon_adamw_mlx/seed17_c36_0666bb51ac1f/pytorch_mlx_decoder_trace.json`
- `.omx/research/codex_pr95_stage6_stage7_full_profile_queue_20260525T1714Z/matrix/stage8/pr95_stage8_muon_adamw_mlx/seed17_c36_0666bb51ac1f/pytorch_export_forward_parity_with_decoder_trace.json`
- `.omx/research/codex_pr95_stage6_stage7_full_profile_queue_20260525T1714Z/matrix/stage8/pr95_stage8_muon_adamw_mlx/seed17_c36_0666bb51ac1f/pr95_mlx_pytorch_per_op_drift.json`
- `.omx/research/codex_pr95_stage6_stage7_full_profile_queue_20260525T1714Z/matrix/stage8/pr95_stage8_muon_adamw_mlx/seed17_c36_0666bb51ac1f/pr95_packaged_submission_report.json`
- `.omx/state/probe_outcomes.jsonl` row `pr95_mlx_decoder_drift_boundary_trace_stage8_20260525`

## Next Engineering

The next reducer is full-frame inflate parity for the packaged PR95 archive, then scorer cache/component trace, then paired contest CPU/CUDA auth anchors. The subagent-audited scorer-drift packet remains the next generalization target for non-PR95 MLX calibration: aggregate cache audit, scorer response, parity sweep, quality/speed delta, score calibration, production contract, and component trace into one false-authority `mlx_drift_packet.v1`.
