import json
from tac.local_acceleration.mlx_scorer_torch_parity import build_mlx_segnet_layer_trace_manifest
m = build_mlx_segnet_layer_trace_manifest(
    cache_dir='experiments/results/mlx_scorer_input_cache_reference_video_20260521T2304Z_full600',
    repo_root='.', device_type='gpu', start_pair=0, max_pairs=2,
    run_id='gpu_trace', allow_gpu_research_signal=True, cliff_threshold=1e-4)
print('drift_cliff', json.dumps(m['drift_cliff']))
print('seg_argmax_diff_pixels', m['segnet_argmax_diff_pixels'])
# print top-5 rows by abs delta
rows = sorted(m['rows'], key=lambda r: -(r.get('max_abs_delta') or 0))
for r in rows[:8]:
    print(f"{r['name']:42s} max_abs={r.get('max_abs_delta'):.5g} mean_abs={r.get('mean_abs_delta'):.5g} cliff={r['exceeds_cliff_threshold']}")
