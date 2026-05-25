# MLX Dynamic Learned Sweep Plan

- Score claim: `False`
- Ready for exact eval dispatch: `False`
- Local ready rows: `5`
- Observation rows: `0`
- Suppressed observed rows: `0`
- Optimizer pairings: `50`

| rank | candidate | config | pass | acq | mean | lcb | local ready | next action |
|---:|---|---|---|---:|---:|---:|---|---|
| 1 | `distortion_axis:uniward_per_instance_multi_scale_wavelet_combined_v1` | `mlx_local_response` | `smoke` | 0.0630167783062 | 0.182028282957 | 0.174528282957 | `True` | `run_local_mlx_or_cpu_sweep_then_append_observation` |
| 2 | `distortion_axis:uniward_per_instance_multi_scale_wavelet_combined_v1` | `macos_cpu_advisory` | `smoke` | 0.0201457624885 | 0.182028282957 | 0.174528282957 | `True` | `run_local_mlx_or_cpu_sweep_then_append_observation` |
| 3 | `distortion_axis:uniward_per_instance_multi_scale_wavelet_combined_v1` | `mlx_local_response` | `micro` | 0.0171489751906 | 0.182028282957 | 0.174528282957 | `True` | `run_local_mlx_or_cpu_sweep_then_append_observation` |
| 4 | `distortion_axis:uniward_per_instance_multi_scale_wavelet_combined_v1` | `contest_cpu_exact_candidate` | `smoke` | 0.00964296610414 | 0.182028282957 | 0.174528282957 | `False` | `materialize_controls_claim_then_exact_eval_in_separate_flow` |
| 5 | `distortion_axis:uniward_per_instance_multi_scale_wavelet_combined_v1` | `contest_cuda_diagnostic` | `smoke` | 0.00675394539624 | 0.182028282957 | 0.174528282957 | `False` | `materialize_controls_claim_then_exact_eval_in_separate_flow` |
| 6 | `distortion_axis:uniward_per_instance_multi_scale_wavelet_combined_v1` | `macos_cpu_advisory` | `micro` | 0.00507145235335 | 0.182028282957 | 0.174528282957 | `True` | `run_local_mlx_or_cpu_sweep_then_append_observation` |
| 7 | `distortion_axis:uniward_per_instance_multi_scale_wavelet_combined_v1` | `mlx_local_response` | `intermediate` | 0.00463593895115 | 0.182028282957 | 0.174528282957 | `True` | `run_local_mlx_or_cpu_sweep_then_append_observation` |
| 8 | `distortion_axis:uniward_per_instance_multi_scale_wavelet_combined_v1` | `contest_cpu_exact_candidate` | `micro` | 0.00221328982159 | 0.182028282957 | 0.174528282957 | `False` | `materialize_controls_claim_then_exact_eval_in_separate_flow` |

Authority: planning only. Local MLX/CPU rows may drive more local sweeps; exact CPU/CUDA rows still require materialization, controls, lane claim, canonical auth eval, and harvest before any score claim.
